"""Tests for `IndexMap.outcome_distributions` of compiled graphical models.

Aggregates over the distributions are checked against brute force over every deterministic
policy, with each policy's outcome distribution computed by pyAgrum inference.
"""

import functools
import itertools
import json

import numpy as np
import pyagrum as gum
import pytest

from desdeo.problem.graphical import (
    DiscreteNetwork,
    DiscreteVariable,
    IndexMap,
    ObjectiveNode,
    Potential,
    RiskMeasure,
    build_outcome_distributions,
    build_rooted_junction_tree,
    compile_diagram,
    effective_information_sets,
)
from desdeo.problem.graphical.adapters.pyagrum import to_pyagrum
from desdeo.problem.graphical.demo import build_irrigation_demo, build_oil_wildcatter_demo
from desdeo.tools.gurobipy_solver_interfaces import GurobipySolver
from desdeo.tools.outcome_distribution import add_cvar_dd, add_expected_value_dd, add_worst_case_dd

pytestmark = [pytest.mark.graphical, pytest.mark.gurobipy]


def _irrigation_with_loss():
    """The irrigation demo plus a minimized `loss = -profit`, so a minimized risk measure is nontrivial."""
    network, annotation = build_irrigation_demo()
    profit = annotation.objectives[0]
    loss = ObjectiveNode(
        symbol="loss",
        parents=profit.parents,
        utility=Potential(
            variables=profit.utility.variables,
            shape=profit.utility.shape,
            values=(-np.array(profit.utility.values)).tolist(),
        ),
        maximize=False,
        unit="EUR",
    )
    return network, annotation.model_copy(update={"objectives": [profit, loss]})


DEMOS = [build_irrigation_demo, build_oil_wildcatter_demo, _irrigation_with_loss]
IDS = ["irrigation", "oil_wildcatter", "irrigation_loss"]


def _cardinality(network, annotation, node_name):
    """Return the number of states for a chance or decision node."""
    for var in network.variables:
        if var.name == node_name:
            return len(var.labels)
    for dec in annotation.decisions:
        if dec.symbol == node_name:
            return len(dec.states)
    raise ValueError(f"Unknown node: {node_name}")


def _make_cpt_values(n_states, info_cards, mapping):
    """A deterministic CPT: 1.0 at each information state's chosen state."""
    values = np.zeros([n_states, *info_cards])
    for info_state, chosen in mapping.items():
        values[(chosen, *info_state)] = 1.0
    return values.tolist()


def _policies(network, annotation):
    """Yield every deterministic policy as a dict mapping decision symbols to (info_order, mapping)."""
    decisions = annotation.decisions
    if not decisions:
        yield {}
        return

    info_sets = effective_information_sets(annotation)
    decision_infos = []
    for dec in decisions:
        info_order = sorted(info_sets.get(dec.symbol, set()))
        info_cards = [_cardinality(network, annotation, n) for n in info_order]
        info_states = list(itertools.product(*[range(c) for c in info_cards]))
        n_states = len(dec.states)
        all_choices = list(itertools.product(*[range(n_states)] * len(info_states)))
        decision_infos.append((dec.symbol, info_order, info_states, all_choices))

    for combo in itertools.product(*[di[3] for di in decision_infos]):
        policy = {}
        for i, (sym, info_order, info_states, _) in enumerate(decision_infos):
            mapping = {info_states[j]: combo[i][j] for j in range(len(info_states))}
            policy[sym] = (info_order, mapping)
        yield policy


def _policy_network(network, annotation, policy):
    """Build a DiscreteNetwork with decisions replaced by deterministic chance nodes."""
    new_variables = list(network.variables)
    new_potentials = dict(network.potentials)

    for dec in annotation.decisions:
        sym = dec.symbol
        states = dec.states
        info_order, mapping = policy[sym]
        info_cards = [_cardinality(network, annotation, n) for n in info_order]
        new_variables.append(DiscreteVariable(name=sym, labels=list(states)))
        shape = [len(states), *info_cards]
        values = _make_cpt_values(len(states), info_cards, mapping)
        new_potentials[sym] = Potential(variables=[sym, *info_order], shape=shape, values=values)

    arcs = []
    for pot in new_potentials.values():
        child = pot.variables[0]
        for parent in pot.variables[1:]:
            arcs.append((parent, child))

    return DiscreteNetwork(name=network.name, variables=new_variables, arcs=arcs, potentials=new_potentials)


def _outcome_distribution(network, annotation, policy, objective_symbol):
    """Compute the outcome distribution for a policy and objective via pyAgrum."""
    objective = next(o for o in annotation.objectives if o.symbol == objective_symbol)
    pn = _policy_network(network, annotation, policy)
    bn = to_pyagrum(pn)

    nodes = list(objective.parents)
    inference = gum.LazyPropagation(bn)
    inference.addJointTarget(set(nodes))
    inference.makeInference()
    posterior = inference.jointPosterior(set(nodes))
    order = [v.name() for v in posterior.variablesSequence()]
    probs = np.transpose(posterior.toarray())
    probs = np.moveaxis(probs, [order.index(n) for n in nodes], range(len(nodes)))

    utility_values = objective.utility.values
    cards = [_cardinality(network, annotation, n) for n in nodes]
    dist = {}
    for combo in itertools.product(*[range(c) for c in cards]):
        p = float(probs[combo])
        if p > 0:
            util = utility_values
            for idx in combo:
                util = util[idx]
            util = float(util)
            dist[util] = dist.get(util, 0.0) + p
    return dist


def _cvar_max(dist, alpha):
    """CVaR for a maximizing objective: mean of the lowest alpha probability mass."""
    items = sorted(dist.items())
    total = 0.0
    weighted = 0.0
    for val, prob in items:
        if total + prob <= alpha:
            total += prob
            weighted += val * prob
        else:
            remaining = alpha - total
            weighted += val * remaining
            total = alpha
            break
    return weighted / alpha


def _brute_force(network, annotation, objective_symbol, measure):
    """Brute-force the best policy value for the given measure."""
    objective = next(o for o in annotation.objectives if o.symbol == objective_symbol)
    best = None

    for policy in _policies(network, annotation):
        dist = _outcome_distribution(network, annotation, policy, objective_symbol)

        if measure == "expected":
            val = sum(v * p for v, p in dist.items())
        elif measure == "worst":
            candidates = [v for v, p in dist.items() if p > 1e-6]
            val = min(candidates) if objective.maximize else max(candidates)
        elif isinstance(measure, tuple) and measure[0] == "cvar":
            alpha = measure[1]
            val = _cvar_max(dist, alpha) if objective.maximize else -_cvar_max({-v: p for v, p in dist.items()}, alpha)
        else:
            raise ValueError(f"Unknown measure: {measure}")

        if best is None:
            best = val
        elif objective.maximize:
            best = max(best, val)
        else:
            best = min(best, val)

    return best


@functools.cache
def _compiled(builder):
    """Cache the compiled problem per demo builder."""
    network, annotation = builder()
    problem, index_map = compile_diagram(network, annotation)
    return network, annotation, problem, index_map


@pytest.mark.parametrize("builder", DEMOS, ids=IDS)
def test_probabilities_sum_to_one_at_optimum(builder):
    """Each outcome distribution's probabilities sum to 1 at the optimum."""
    _, _, problem, index_map = _compiled(builder)
    for obj in problem.objectives:
        result = GurobipySolver(problem).solve(obj.symbol)
        assert result.success, result.message
        for dist in index_map.outcome_distributions.values():
            total = sum(result.extra_func_values[p] for p in dist.probabilities)
            assert total == pytest.approx(1, abs=1e-6)


@pytest.mark.parametrize("builder", DEMOS, ids=IDS)
def test_expected_value_over_distributions_matches_objective(builder):
    """Expected value of the outcome distribution equals the objective value."""
    network, annotation, problem, index_map = _compiled(builder)
    symbols = [o.symbol for o in problem.objectives]
    new_problem, _ = add_expected_value_dd(problem, index_map.outcome_distributions, symbols)
    for sym in symbols:
        result = GurobipySolver(new_problem).solve(sym)
        assert result.success, result.message
        assert result.optimal_objectives[f"E_{sym}"] == pytest.approx(result.optimal_objectives[sym], abs=1e-6)
        assert result.optimal_objectives[sym] == pytest.approx(
            _brute_force(network, annotation, sym, "expected"), abs=1e-6
        )


@pytest.mark.parametrize("builder", DEMOS, ids=IDS)
def test_worst_case_matches_brute_force(builder):
    """Worst-case risk measure matches brute-force enumeration."""
    network, annotation, problem, index_map = _compiled(builder)
    for obj in problem.objectives:
        sym = obj.symbol
        new_problem, mapping = add_worst_case_dd(problem, index_map.outcome_distributions, [sym])
        new_sym = mapping[sym]
        result = GurobipySolver(new_problem).solve(new_sym)
        assert result.success, result.message
        expected = _brute_force(network, annotation, sym, "worst")
        assert result.optimal_objectives[new_sym] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("alpha", [0.1, 0.5])
@pytest.mark.parametrize("builder", DEMOS, ids=IDS)
def test_cvar_matches_brute_force(builder, alpha):
    """CVaR risk measure matches brute-force enumeration."""
    network, annotation, problem, index_map = _compiled(builder)
    for obj in problem.objectives:
        sym = obj.symbol
        new_problem, mapping = add_cvar_dd(problem, index_map.outcome_distributions, [sym], alpha, prefix="CVAR_")
        new_sym = mapping[sym]
        result = GurobipySolver(new_problem).solve(new_sym)
        assert result.success, result.message
        expected = _brute_force(network, annotation, sym, ("cvar", alpha))
        assert result.optimal_objectives[new_sym] == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("builder", DEMOS, ids=IDS)
def test_index_map_without_outcome_distributions_loads(builder):
    """IndexMap reloads correctly when outcome_distributions key is absent."""
    _, _, _, index_map = _compiled(builder)
    data = json.loads(index_map.model_dump_json())
    del data["outcome_distributions"]
    reloaded = IndexMap.model_validate_json(json.dumps(data))
    assert reloaded.outcome_distributions == {}
    assert reloaded.model_copy(update={"outcome_distributions": index_map.outcome_distributions}) == index_map


@pytest.mark.parametrize("builder", DEMOS, ids=IDS)
def test_build_outcome_distributions_matches_compiled(builder):
    """Standalone build matches the compiled index map and extra funcs."""
    network, annotation = builder()
    rjt = build_rooted_junction_tree(network, annotation)
    problem, index_map = compile_diagram(network, annotation, rjt=rjt)
    distributions, extra_funcs = build_outcome_distributions(network, annotation, rjt, index_map.clusters)
    assert distributions == index_map.outcome_distributions
    symbols = {f.symbol for f in extra_funcs}
    compiled_extra = [f for f in problem.extra_funcs if f.symbol in symbols]
    assert [f.model_dump() for f in compiled_extra] == [f.model_dump() for f in extra_funcs]


def _extract_indices(func):
    """The 0-based mu indices a `pu_*` function's `Sum(Extract(mu, ...))` selects."""
    _, (_, _, *spec) = func
    if len(spec) == 1 and isinstance(spec[0], list):
        _, first, last = spec[0]
        return list(range(first - 1, last))
    return [i - 1 for i in spec]


@pytest.mark.parametrize("builder", DEMOS, ids=IDS)
def test_probabilities_partition_containing_cluster(builder):
    """Each objective's `pu_*` functions select every entry of one cluster's mu exactly once."""
    _, _, problem, index_map = _compiled(builder)
    funcs = {f.symbol: f.func for f in problem.extra_funcs}
    sizes = {c.mu_symbol: c.state_space_size for c in index_map.clusters.values()}
    for dist in index_map.outcome_distributions.values():
        mu_symbols = {funcs[p][1][1] for p in dist.probabilities}
        assert len(mu_symbols) == 1
        (mu,) = mu_symbols
        indices = [i for p in dist.probabilities for i in _extract_indices(funcs[p])]
        assert sorted(indices) == list(range(sizes[mu]))


@pytest.mark.parametrize("alpha", [0.1, 0.5])
@pytest.mark.parametrize("builder", DEMOS, ids=IDS)
def test_compiled_risk_measure_matches_add_cvar_dd(builder, alpha):
    """A compile-time `RiskMeasure` has the same optimum as `add_cvar_dd` over the distributions."""
    network, annotation = builder()
    risk_measures = [
        RiskMeasure(symbol=f"cvar_{o.symbol}", target=o.symbol, alpha=alpha) for o in annotation.objectives
    ]
    risk_problem, _ = compile_diagram(network, annotation.model_copy(update={"risk_measures": risk_measures}))
    _, _, problem, index_map = _compiled(builder)
    for risk_measure in risk_measures:
        compiled = GurobipySolver(risk_problem).solve(risk_measure.symbol)
        assert compiled.success, compiled.message
        new_problem, mapping = add_cvar_dd(
            problem, index_map.outcome_distributions, [risk_measure.target], alpha, prefix="CVAR_"
        )
        result = GurobipySolver(new_problem).solve(mapping[risk_measure.target])
        assert result.success, result.message
        assert compiled.optimal_objectives[risk_measure.symbol] == pytest.approx(
            result.optimal_objectives[mapping[risk_measure.target]], abs=1e-6
        )


def test_non_finite_utility_raises():
    """A non-finite utility value is rejected with a message naming the objective."""
    network, annotation = build_irrigation_demo()
    profit = annotation.objectives[0]
    values = np.array(profit.utility.values, dtype=float)
    values.flat[0] = np.inf
    bad = profit.model_copy(
        update={
            "utility": Potential(variables=profit.utility.variables, shape=profit.utility.shape, values=values.tolist())
        }
    )
    with pytest.raises(ValueError, match=f"Objective '{profit.symbol}' has non-finite utility"):
        compile_diagram(network, annotation.model_copy(update={"objectives": [bad]}))
