"""Tests for `desdeo.problem.graphical.compile`.

Each case is checked against independent oracles as well as the compiled MILP:

1. **Brute force** (`_brute_force_optimum`): enumerates every deterministic policy and chance-node
   joint state in plain Python/NumPy. Supports multi-objective annotations; tiny diagrams only.
2. **pyAgrum** (`to_pyagrum_influence_diagram` + `ShaferShenoyLIMIDInference`/`LazyPropagation`):
   single-objective only (see `_pyagrum_meu`).
3. Property-based random small networks (`gum.randomBN`) with a decision and objective grafted on.

Diagram constraints use `_brute_force_optimum_feasible` and `_brute_force_marginal`. Risk measures
(CVaR, Herrala, Terho & Oliveira 2025) use `_brute_force_cvar`, plus the Rockafellar & Uryasev
relationships `CVaR_alpha -> E[utility]` as `alpha -> 1` and `CVaR_alpha <= E[utility]`.

Every test solves via gurobipy, so the module is marked `gurobipy` as well as `graphical`.
"""

import itertools

import numpy as np
import pyagrum as gum
import pytest

from desdeo.problem.graphical import (
    Cluster,
    DecisionAnnotation,
    DecisionNode,
    DiagramConstraint,
    DiagramConstraintKind,
    DiscreteNetwork,
    DiscreteVariable,
    IndexMap,
    ObjectiveNode,
    Potential,
    RiskMeasure,
    RootedJunctionTree,
    build_rooted_junction_tree,
    compile_diagram,
    read_policy_from_solution,
)
from desdeo.problem.graphical.adapters.pyagrum import from_pyagrum, to_pyagrum, to_pyagrum_influence_diagram
from desdeo.problem.graphical.index_map import mixed_radix_decode
from desdeo.problem.graphical.network import kahn_topological_order
from desdeo.problem.graphical.rooted_junction_tree import effective_information_sets
from desdeo.problem.schema import ConstraintTypeEnum
from desdeo.tools.gurobipy_solver_interfaces import GurobipySolver

pytestmark = [pytest.mark.graphical, pytest.mark.gurobipy]


# ── Oracle 1: brute force over policies and chance-node joint states ───────────────────────────


def _cardinalities(network: DiscreteNetwork, annotation: DecisionAnnotation) -> dict[str, int]:
    cardinality = {variable.name: len(variable.labels) for variable in network.variables}
    cardinality.update({decision.symbol: len(decision.states) for decision in annotation.decisions})
    return cardinality


def _network_topological_order(network: DiscreteNetwork) -> list[str]:
    """Topological order of just the chance nodes -- usable before any annotation exists."""
    names = [variable.name for variable in network.variables]
    predecessors: dict[str, set[str]] = {name: set() for name in names}
    for variable in network.variables:
        predecessors[variable.name] |= set(network.potentials[variable.name].variables[1:])
    return kahn_topological_order(names, predecessors)


def _topological_order(network: DiscreteNetwork, annotation: DecisionAnnotation) -> list[str]:
    cardinality = _cardinalities(network, annotation)
    eff = effective_information_sets(annotation)
    predecessors: dict[str, set[str]] = {node: set() for node in cardinality}
    for variable in network.variables:
        predecessors[variable.name] |= set(network.potentials[variable.name].variables[1:])
    for decision in annotation.decisions:
        predecessors[decision.symbol] |= eff[decision.symbol]
    return kahn_topological_order(list(cardinality), predecessors)


def _enumerate_policies(annotation: DecisionAnnotation, cardinality: dict[str, int], eff: dict[str, set[str]]):
    """Yield one `{decision_symbol: (info_order, policy_dict)}` per deterministic policy combo."""
    per_decision = []
    for decision in annotation.decisions:
        info_order = sorted(eff[decision.symbol])
        info_states = list(itertools.product(*(range(cardinality[node]) for node in info_order))) or [()]
        state_choices = list(itertools.product(range(len(decision.states)), repeat=len(info_states)))
        per_decision.append(
            [(decision.symbol, info_order, dict(zip(info_states, choice, strict=True))) for choice in state_choices]
        )
    for combo in itertools.product(*per_decision) if per_decision else [()]:
        yield {symbol: (info_order, policy) for symbol, info_order, policy in combo}


def _expected_utilities(
    network: DiscreteNetwork,
    annotation: DecisionAnnotation,
    order: list[str],
    cardinality: dict[str, int],
    policy_by_decision: dict[str, tuple[list[str], dict]],
) -> dict[str, float]:
    var_names = [variable.name for variable in network.variables]
    totals = {objective.symbol: 0.0 for objective in annotation.objectives}

    for chance_assignment in itertools.product(*(range(cardinality[name]) for name in var_names)):
        state = dict(zip(var_names, chance_assignment, strict=True))
        for node in order:
            if node in state:
                continue
            info_order, policy = policy_by_decision[node]
            state[node] = policy[tuple(state[n] for n in info_order)]

        prob = 1.0
        for variable in network.variables:
            potential = network.potentials[variable.name]
            idx = tuple(state[name] for name in potential.variables)
            prob *= float(np.array(potential.values)[idx])

        for objective in annotation.objectives:
            utility_array = np.array(objective.utility.values, dtype=float)
            if objective.parents:
                idx = tuple(state[parent] for parent in objective.parents)
                util = float(utility_array[idx])
            else:
                util = float(utility_array)
            totals[objective.symbol] += prob * util

    return totals


def _brute_force_optimum(network: DiscreteNetwork, annotation: DecisionAnnotation, objective_symbol: str) -> float:
    """The exact MEU for `objective_symbol`, by enumerating every deterministic policy."""
    cardinality = _cardinalities(network, annotation)
    eff = effective_information_sets(annotation)
    order = _topological_order(network, annotation)
    objective = next(o for o in annotation.objectives if o.symbol == objective_symbol)

    best = None
    for policy in _enumerate_policies(annotation, cardinality, eff):
        value = _expected_utilities(network, annotation, order, cardinality, policy)[objective_symbol]
        if best is None or (value > best if objective.maximize else value < best):
            best = value
    return best


def _brute_force_optimum_feasible(
    network: DiscreteNetwork,
    annotation: DecisionAnnotation,
    objective_symbol: str,
    feasible,
) -> float:
    """`_brute_force_optimum`, restricted to policies for which `feasible(policy)` is true."""
    cardinality = _cardinalities(network, annotation)
    eff = effective_information_sets(annotation)
    order = _topological_order(network, annotation)
    objective = next(o for o in annotation.objectives if o.symbol == objective_symbol)

    best = None
    for policy in _enumerate_policies(annotation, cardinality, eff):
        if not feasible(policy):
            continue
        value = _expected_utilities(network, annotation, order, cardinality, policy)[objective_symbol]
        if best is None or (value > best if objective.maximize else value < best):
            best = value
    return best


def _brute_force_marginal(
    network: DiscreteNetwork,
    annotation: DecisionAnnotation,
    order: list[str],
    cardinality: dict[str, int],
    policy_by_decision: dict[str, tuple[list[str], dict]],
    node: str,
    state_index: int,
) -> float:
    """P(`node` == `state_index`) under one fixed deterministic policy, from the network's CPTs."""
    var_names = [variable.name for variable in network.variables]
    total = 0.0

    for chance_assignment in itertools.product(*(range(cardinality[name]) for name in var_names)):
        state = dict(zip(var_names, chance_assignment, strict=True))
        for n in order:
            if n in state:
                continue
            info_order, policy = policy_by_decision[n]
            state[n] = policy[tuple(state[i] for i in info_order)]

        prob = 1.0
        for variable in network.variables:
            potential = network.potentials[variable.name]
            idx = tuple(state[name] for name in potential.variables)
            prob *= float(np.array(potential.values)[idx])

        if state[node] == state_index:
            total += prob

    return total


def _utility_distribution_for_policy(
    network: DiscreteNetwork,
    annotation: DecisionAnnotation,
    order: list[str],
    cardinality: dict[str, int],
    policy_by_decision: dict[str, tuple[list[str], dict]],
    objective_symbol: str,
) -> dict[float, float]:
    """`{utility_value: probability_mass}` for one objective under one fixed deterministic policy."""
    var_names = [variable.name for variable in network.variables]
    objective = next(o for o in annotation.objectives if o.symbol == objective_symbol)
    utility_array = np.array(objective.utility.values, dtype=float)
    distribution: dict[float, float] = {}

    for chance_assignment in itertools.product(*(range(cardinality[name]) for name in var_names)):
        state = dict(zip(var_names, chance_assignment, strict=True))
        for node in order:
            if node in state:
                continue
            info_order, policy = policy_by_decision[node]
            state[node] = policy[tuple(state[n] for n in info_order)]

        prob = 1.0
        for variable in network.variables:
            potential = network.potentials[variable.name]
            idx = tuple(state[name] for name in potential.variables)
            prob *= float(np.array(potential.values)[idx])

        if objective.parents:
            idx = tuple(state[parent] for parent in objective.parents)
            util = float(utility_array[idx])
        else:
            util = float(utility_array)

        distribution[util] = distribution.get(util, 0.0) + prob

    return distribution


def _cvar_from_distribution(mass_by_value: dict[float, float], alpha: float) -> float:
    """Textbook CVaR_alpha of `mass_by_value`'s *lower* tail (Rockafellar & Uryasev 2000).

    The boundary value's mass is taken fractionally so the tail is exactly `alpha`.
    """
    remaining = alpha
    weighted = 0.0
    for value in sorted(mass_by_value):
        take = min(mass_by_value[value], remaining)
        weighted += take * value
        remaining -= take
        if remaining <= 1e-12:
            break
    return weighted / alpha


def _brute_force_cvar(
    network: DiscreteNetwork, annotation: DecisionAnnotation, objective_symbol: str, alpha: float
) -> float:
    """The exact CVaR_alpha optimum for `objective_symbol`, by enumerating every policy.

    Minimized objectives are negated, take the lower-tail CVaR, and negate back.
    """
    cardinality = _cardinalities(network, annotation)
    eff = effective_information_sets(annotation)
    order = _topological_order(network, annotation)
    objective = next(o for o in annotation.objectives if o.symbol == objective_symbol)

    best = None
    for policy in _enumerate_policies(annotation, cardinality, eff):
        distribution = _utility_distribution_for_policy(
            network, annotation, order, cardinality, policy, objective_symbol
        )
        if objective.maximize:
            value = _cvar_from_distribution(distribution, alpha)
        else:
            negated = {-u: p for u, p in distribution.items()}
            value = -_cvar_from_distribution(negated, alpha)
        if best is None or (value > best if objective.maximize else value < best):
            best = value
    return best


# ── Oracle 2: pyAgrum's own LIMID inference ─────────────────────────────────────────────────────


def _pyagrum_meu(network: DiscreteNetwork, annotation: DecisionAnnotation, objective_symbol: str) -> float:
    """The expected utility for `objective_symbol`, computed via pyAgrum's own inference.

    Single-objective only: `MEU()` sums every utility node, and `posteriorUtility(node)` doesn't
    reduce to a scalar. Diagrams without decisions use `LazyPropagation`, since
    `ShaferShenoyLIMIDInference` refuses them.
    """
    if len(annotation.objectives) != 1:
        msg = "_pyagrum_meu only supports single-objective annotations (see the docstring)."
        raise NotImplementedError(msg)
    objective = annotation.objectives[0]
    assert objective.symbol == objective_symbol

    if not annotation.decisions:
        bn = to_pyagrum(network)
        inference = gum.LazyPropagation(bn)

        if not objective.parents:
            return float(np.array(objective.utility.values, dtype=float))

        # `jointPosterior` only ever answers pre-declared joint targets ("not addressable in
        # this optimized inference" otherwise) -- must be registered before `makeInference`.
        inference.addJointTarget(set(objective.parents))
        inference.makeInference()
        posterior = inference.jointPosterior(set(objective.parents))
        order = [v.name() for v in posterior.variablesSequence()]
        probs = np.transpose(posterior.toarray())  # now indexed in `order`'s order
        utility = np.array(objective.utility.values, dtype=float)
        # `order` (pyAgrum's own joint-posterior variable order) need not match
        # `objective.parents`'s order, so align before contracting.
        probs = np.moveaxis(probs, [order.index(p) for p in objective.parents], range(len(objective.parents)))
        return float(np.sum(probs * utility))

    diagram = to_pyagrum_influence_diagram(network, annotation)
    inference = gum.ShaferShenoyLIMIDInference(diagram)

    if annotation.no_forgetting:
        # Needed only with `no_forgetting`, not `decision_order` alone. Takes decision names;
        # the id overload raises `NotFound`.
        inference.addNoForgettingAssumption(list(annotation.decision_order))

    inference.makeInference()
    # `to_pyagrum_influence_diagram` writes the utility table in pyAgrum's own maximize-only sign
    # convention (negated when `objective.maximize` is False -- see that function's docstring), so
    # `MEU()["mean"]` must be negated back by the same sign to land in `objective`'s own units.
    sign = 1.0 if objective.maximize else -1.0
    return sign * inference.MEU()["mean"]


# ── Oracle-checked compiled MILP value ──────────────────────────────────────────────────────────


def _compiled_optimum(network: DiscreteNetwork, annotation: DecisionAnnotation, objective_symbol: str) -> float:
    problem, _ = compile_diagram(network, annotation)
    result = GurobipySolver(problem).solve(objective_symbol)
    assert result.success, result.message
    return result.optimal_objectives[objective_symbol]


def _read_back_policy(annotation: DecisionAnnotation, index_map: IndexMap, result):
    """Reconstruct `{decision_symbol: (info_order, {info_state: chosen_state})}` from a solved `z`.

    Wraps `read_policy_from_solution` in the shape the brute-force oracle uses, and asserts each
    `z` block has exactly one 1.0.
    """
    policies = read_policy_from_solution(
        index_map, {sym: result.optimal_variables[sym] for sym in (dm.z_symbol for dm in index_map.decisions.values())}
    )
    policy_by_decision = {}
    for decision in annotation.decisions:
        decision_map = index_map.decisions[decision.symbol]
        decision_policy = policies[decision.symbol]
        z_values = result.optimal_variables[decision_map.z_symbol]
        n_states = len(decision_map.states)

        policy: dict[tuple[int, ...], int] = {}
        for info_flat, chosen in decision_policy.policy.items():
            block = z_values[info_flat * n_states : (info_flat + 1) * n_states]
            assert block[chosen] == pytest.approx(1.0, abs=1e-6)
            info_state = (
                mixed_radix_decode(info_flat, decision_map.information_cardinalities)
                if decision_map.information_set
                else ()
            )
            policy[info_state] = chosen
        policy_by_decision[decision.symbol] = (decision_map.information_set, policy)
    return policy_by_decision


def _read_back_policy_and_reverify(
    network: DiscreteNetwork, annotation: DecisionAnnotation, objective_symbol: str
) -> None:
    """Reconstruct the solved policy from `IndexMap` + solved `z`, confirm brute force agrees.

    Catches `IndexMap` ordering errors, which objective values alone wouldn't reveal.
    """
    problem, index_map = compile_diagram(network, annotation)
    result = GurobipySolver(problem).solve(objective_symbol)
    assert result.success, result.message

    cardinality = _cardinalities(network, annotation)
    policy_by_decision = _read_back_policy(annotation, index_map, result)

    order = _topological_order(network, annotation)
    readback = _expected_utilities(network, annotation, order, cardinality, policy_by_decision)[objective_symbol]
    assert readback == pytest.approx(result.optimal_objectives[objective_symbol], abs=1e-6)


def _assert_all_agree(network: DiscreteNetwork, annotation: DecisionAnnotation, objective_symbol: str) -> None:
    brute_force = _brute_force_optimum(network, annotation, objective_symbol)
    pyagrum_value = _pyagrum_meu(network, annotation, objective_symbol)
    compiled = _compiled_optimum(network, annotation, objective_symbol)

    assert compiled == pytest.approx(brute_force, abs=1e-6)
    assert compiled == pytest.approx(pyagrum_value, abs=1e-6)

    _read_back_policy_and_reverify(network, annotation, objective_symbol)


# ── Structured cases ─────────────────────────────────────────────────────────────────────────────


def test_single_chance_node_expectation():
    """No decisions at all: the objective's compiled value is just a plain expectation."""
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6])},
    )
    annotation = DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="f1", parents=["A"], utility=Potential(variables=["A"], shape=[2], values=[10.0, 20.0])
            )
        ]
    )
    _assert_all_agree(network, annotation, "f1")


def test_chain_network_expectation():
    """A -> B, objective over the downstream node B only (exercises the CPT chain rule)."""
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"]), DiscreteVariable(name="B", labels=["lo", "hi"])],
        arcs=[("A", "B")],
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6]),
            "B": Potential(variables=["B", "A"], shape=[2, 2], values=[[0.9, 0.2], [0.1, 0.8]]),
        },
    )
    annotation = DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="fB",
                parents=["B"],
                utility=Potential(variables=["B"], shape=[2], values=[1.0, 0.0]),
                maximize=True,
            )
        ]
    )
    _assert_all_agree(network, annotation, "fB")


def test_fork_network_joint_objective():
    """A -> B, A -> C; objective jointly over (B, C) -- exercises a 3-node cluster marginalisation."""
    network = DiscreteNetwork(
        variables=[
            DiscreteVariable(name="A", labels=["lo", "hi"]),
            DiscreteVariable(name="B", labels=["lo", "hi"]),
            DiscreteVariable(name="C", labels=["lo", "hi"]),
        ],
        arcs=[("A", "B"), ("A", "C")],
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6]),
            "B": Potential(variables=["B", "A"], shape=[2, 2], values=[[0.9, 0.2], [0.1, 0.8]]),
            "C": Potential(variables=["C", "A"], shape=[2, 2], values=[[0.7, 0.3], [0.3, 0.7]]),
        },
    )
    annotation = DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="fBC",
                parents=["B", "C"],
                utility=Potential(variables=["B", "C"], shape=[2, 2], values=[[0.0, 0.0], [0.0, 1.0]]),
                maximize=True,
            )
        ]
    )
    _assert_all_agree(network, annotation, "fBC")


def test_bandit_decision_no_information():
    """A single decision with an empty information set: an unconditional argmax."""
    network = DiscreteNetwork(variables=[], arcs=[], potentials={})
    annotation = DecisionAnnotation(
        decisions=[DecisionNode(symbol="D", states=["s0", "s1"])],
        objectives=[
            ObjectiveNode(
                symbol="fD",
                parents=["D"],
                utility=Potential(variables=["D"], shape=[2], values=[5.0, 8.0]),
                maximize=True,
            )
        ],
    )
    _assert_all_agree(network, annotation, "fD")


def test_decision_observes_chance_node():
    """D observes A, reward for D matching A -- a genuine (not just unconditional) policy."""
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[2], values=[0.3, 0.7])},
    )
    annotation = DecisionAnnotation(
        decisions=[DecisionNode(symbol="D", states=["s0", "s1"], information_set=["A"])],
        objectives=[
            ObjectiveNode(
                symbol="fmatch",
                parents=["A", "D"],
                utility=Potential(variables=["A", "D"], shape=[2, 2], values=[[10.0, 0.0], [0.0, 10.0]]),
                maximize=True,
            )
        ],
    )
    _assert_all_agree(network, annotation, "fmatch")


def test_chance_node_cpt_conditions_on_decision():
    """B's own CPT conditions on both a chance parent A and a decision D.

    D has no information set, so the optimal policy is driven by D's effect on B's distribution:
    P(B=hi | D=low) = 0.3, P(B=hi | D=high) = 0.75.
    """
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"]), DiscreteVariable(name="B", labels=["lo", "hi"])],
        arcs=[("A", "B")],
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=[0.5, 0.5]),
            "B": Potential(
                variables=["B", "A", "D"],
                shape=[2, 2, 2],
                values=[[[0.8, 0.4], [0.6, 0.1]], [[0.2, 0.6], [0.4, 0.9]]],
            ),
        },
    )
    annotation = DecisionAnnotation(
        decisions=[DecisionNode(symbol="D", states=["low", "high"])],
        objectives=[
            ObjectiveNode(
                symbol="fB",
                parents=["B"],
                utility=Potential(variables=["B"], shape=[2], values=[0.0, 1.0]),
                maximize=True,
            )
        ],
    )
    _assert_all_agree(network, annotation, "fB")


def test_read_policy_from_solution_returns_full_policy_function():
    """`read_policy_from_solution` reports a per-information-state function.

    D observes A and is rewarded for matching it, so the policy differs by information state.
    """
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[2], values=[0.3, 0.7])},
    )
    annotation = DecisionAnnotation(
        decisions=[DecisionNode(symbol="D", states=["s0", "s1"], information_set=["A"])],
        objectives=[
            ObjectiveNode(
                symbol="fmatch",
                parents=["A", "D"],
                utility=Potential(variables=["A", "D"], shape=[2, 2], values=[[10.0, 0.0], [0.0, 10.0]]),
                maximize=True,
            )
        ],
    )
    problem, index_map = compile_diagram(network, annotation)
    result = GurobipySolver(problem).solve("fmatch")
    assert result.success, result.message

    policies = read_policy_from_solution(index_map, result.optimal_variables)
    decision_policy = policies["D"]

    assert decision_policy.symbol == "D"
    assert decision_policy.information_set == ["A"]
    assert decision_policy.states == ["s0", "s1"]
    # Two distinct information states (A=lo, A=hi), each its own entry -- not a single value.
    assert len(decision_policy.policy) == 2
    assert decision_policy.policy[0] == 0  # A="lo" (flat 0) -> D="s0" (index 0)
    assert decision_policy.policy[1] == 1  # A="hi" (flat 1) -> D="s1" (index 1)
    # The two information states genuinely disagree -- confirms this isn't a collapsed single
    # choice silently repeated for every information state.
    assert decision_policy.policy[0] != decision_policy.policy[1]


def test_two_decisions_with_no_forgetting():
    """D1 observes A; D2, with no_forgetting, implicitly also observes D1 (and hence A)."""
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[2], values=[0.35, 0.65])},
    )
    annotation = DecisionAnnotation(
        decisions=[
            DecisionNode(symbol="D1", states=["s0", "s1"], information_set=["A"]),
            DecisionNode(symbol="D2", states=["t0", "t1"]),
        ],
        objectives=[
            ObjectiveNode(
                symbol="fchain",
                parents=["A", "D1", "D2"],
                utility=Potential(
                    variables=["A", "D1", "D2"],
                    shape=[2, 2, 2],
                    # Rewarded for D1 matching A, and D2 matching D1 -- no_forgetting is what
                    # makes D2's own perfect match achievable, since D2 never observes A or D1
                    # by itself.
                    values=[[[10.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 10.0]]],
                ),
                maximize=True,
            )
        ],
        decision_order=["D1", "D2"],
        no_forgetting=True,
    )
    _assert_all_agree(network, annotation, "fchain")


def test_two_decisions_without_no_forgetting_is_worse_or_equal():
    """The same diagram without no_forgetting can't do better -- D2 is blind to A and D1."""
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[2], values=[0.35, 0.65])},
    )

    def _make(no_forgetting: bool) -> DecisionAnnotation:
        return DecisionAnnotation(
            decisions=[
                DecisionNode(symbol="D1", states=["s0", "s1"], information_set=["A"]),
                DecisionNode(symbol="D2", states=["t0", "t1"]),
            ],
            objectives=[
                ObjectiveNode(
                    symbol="fchain",
                    parents=["A", "D1", "D2"],
                    utility=Potential(
                        variables=["A", "D1", "D2"],
                        shape=[2, 2, 2],
                        values=[[[10.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 10.0]]],
                    ),
                    maximize=True,
                )
            ],
            decision_order=["D1", "D2"],
            no_forgetting=no_forgetting,
        )

    with_memory = _make(True)
    without_memory = _make(False)
    _assert_all_agree(network, with_memory, "fchain")

    # pyAgrum can't derive a decision order here ("not solvable"), so only brute force checks it.
    brute_force_without_memory = _brute_force_optimum(network, without_memory, "fchain")
    compiled_without_memory = _compiled_optimum(network, without_memory, "fchain")
    assert compiled_without_memory == pytest.approx(brute_force_without_memory, abs=1e-6)

    with_memory_value = _compiled_optimum(network, with_memory, "fchain")
    assert with_memory_value >= compiled_without_memory - 1e-9


def test_multi_objective_including_all_zero_utility():
    """Two objectives compiled together, one with an all-zero utility table.

    Checked against brute force only, since the pyAgrum oracle is single-objective.
    """
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6])},
    )
    annotation = DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="f_real", parents=["A"], utility=Potential(variables=["A"], shape=[2], values=[10.0, 20.0])
            ),
            ObjectiveNode(
                symbol="f_zero", parents=["A"], utility=Potential(variables=["A"], shape=[2], values=[0.0, 0.0])
            ),
        ]
    )
    problem, _ = compile_diagram(network, annotation)

    for symbol, expected in (("f_real", 16.0), ("f_zero", 0.0)):
        brute_force = _brute_force_optimum(network, annotation, symbol)
        assert brute_force == pytest.approx(expected, abs=1e-9)
        result = GurobipySolver(problem).solve(symbol)
        assert result.success, result.message
        assert result.optimal_objectives[symbol] == pytest.approx(brute_force, abs=1e-6)


# ── Property-based: random small networks, a grafted-on decision and objective ─────────────────


def _graft_decision_and_objective(network: DiscreteNetwork, rng: np.random.Generator) -> DecisionAnnotation:
    """A single decision (observing a random subset of the earlier nodes) and one objective."""
    order = _network_topological_order(network)
    decision_node_position = len(order) - 1
    info_candidates = order[:decision_node_position]
    info_set = [node for node in info_candidates if rng.random() < 0.5]

    n_decision_states = 2
    decision = DecisionNode(symbol="DEC", states=[f"d{i}" for i in range(n_decision_states)], information_set=info_set)

    cardinality = {variable.name: len(variable.labels) for variable in network.variables}
    parents = [*info_set, order[decision_node_position], "DEC"]
    shape = [cardinality[p] if p != "DEC" else n_decision_states for p in parents]
    utility_values = rng.uniform(0.0, 10.0, size=shape).tolist()

    objective = ObjectiveNode(
        symbol="obj",
        parents=parents,
        utility=Potential(variables=parents, shape=shape, values=utility_values),
        maximize=True,
    )
    return DecisionAnnotation(decisions=[decision], objectives=[objective])


@pytest.mark.parametrize(
    ("seed", "n", "domain_size", "ratio_arc"),
    # gum.randomBN refuses n <= 3 ("A BN can not be randomly generated from less than 4 nodes.").
    [(101, 4, 2, 1.2), (102, 4, 2, 1.0), (103, 5, 3, 1.5)],
)
def test_random_small_diagrams_all_oracles_agree(seed, n, domain_size, ratio_arc):
    """A random small BN plus a grafted-on decision/objective, checked against both oracles."""
    gum.initRandom(seed)
    network = from_pyagrum(gum.randomBN(n=n, domain_size=domain_size, ratio_arc=ratio_arc))
    annotation = _graft_decision_and_objective(network, np.random.default_rng(seed))

    # Also exercise build_rooted_junction_tree directly once, so a failure here points at
    # whether the tree itself or the compiler on top of it is at fault.
    build_rooted_junction_tree(network, annotation)

    _assert_all_agree(network, annotation, "obj")


# ── compile_diagram robustness against a caller-supplied rjt ────────────────────────────────────


def test_malformed_root_cluster_separator_raises():
    """A caller-supplied `rjt` whose root cluster has a non-empty separator is rejected."""
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"]), DiscreteVariable(name="B", labels=["lo", "hi"])],
        arcs=[("A", "B")],
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6]),
            "B": Potential(variables=["B", "A"], shape=[2, 2], values=[[0.9, 0.2], [0.1, 0.8]]),
        },
    )
    annotation = DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="fA", parents=["A"], utility=Potential(variables=["A"], shape=[2], values=[1.0, 2.0])
            )
        ]
    )
    bad_rjt = RootedJunctionTree(
        clusters={
            "A": Cluster(ownership="A", nodes=["A", "B"], parent=None, separator=["B"], order=0, state_space_size=4),
            "B": Cluster(ownership="B", nodes=["A", "B"], parent="A", separator=["A"], order=1, state_space_size=4),
        },
        root="A",
    )
    with pytest.raises(ValueError, match="non-empty separator"):
        compile_diagram(network, annotation, rjt=bad_rjt)


# ── DiagramConstraint compilation ───────────────────────────────────────────────────────────────


def test_decision_constraint_exclusivity_changes_optimum():
    """A `kind == "decision"` exclusivity constraint over two unconditional decisions' states.

    Checked against `_brute_force_optimum_feasible` with the same exclusivity condition.
    """
    network = DiscreteNetwork(variables=[], arcs=[], potentials={})
    annotation = DecisionAnnotation(
        decisions=[
            DecisionNode(symbol="D1", states=["s0", "s1"]),
            DecisionNode(symbol="D2", states=["s0", "s1"]),
        ],
        objectives=[
            ObjectiveNode(
                symbol="fD",
                parents=["D1", "D2"],
                utility=Potential(variables=["D1", "D2"], shape=[2, 2], values=[[1.0, 3.0], [3.0, 10.0]]),
                maximize=True,
            )
        ],
    )
    unconstrained = _compiled_optimum(network, annotation, "fD")
    assert unconstrained == pytest.approx(10.0, abs=1e-6)

    constrained = annotation.model_copy(
        update={
            "constraints": [
                DiagramConstraint(
                    symbol="excl",
                    kind=DiagramConstraintKind.decision,
                    nodes=["D1", "D2"],
                    # At most one of D1/D2 may be in their (0-based) state 1, i.e. "s1".
                    func=["Subtract", ["Add", ["At", "D1", 2], ["At", "D2", 2]], 1],
                    cons_type=ConstraintTypeEnum.LTE,
                )
            ]
        }
    )
    constrained_optimum = _compiled_optimum(network, constrained, "fD")
    assert constrained_optimum == pytest.approx(3.0, abs=1e-6)
    assert constrained_optimum < unconstrained - 1e-6

    def feasible(policy) -> bool:
        chosen_d1 = policy["D1"][1][()]
        chosen_d2 = policy["D2"][1][()]
        return not (chosen_d1 == 1 and chosen_d2 == 1)

    brute_force = _brute_force_optimum_feasible(network, constrained, "fD", feasible)
    assert constrained_optimum == pytest.approx(brute_force, abs=1e-6)


def test_probability_constraint_on_decision_marginal_changes_optimum():
    """A `kind == "probability"` bound on a decision's own marginal, P(D == "s1") <= 0.5.

    The unconstrained optimum gives P(D == "s1") == 0.7, so the bound is binding. Checked against
    `_brute_force_marginal` on the solved policy and against `_brute_force_optimum_feasible`.
    """
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[2], values=[0.3, 0.7])},
    )
    base = DecisionAnnotation(
        decisions=[DecisionNode(symbol="D", states=["s0", "s1"], information_set=["A"])],
        objectives=[
            ObjectiveNode(
                symbol="fmatch",
                parents=["A", "D"],
                utility=Potential(variables=["A", "D"], shape=[2, 2], values=[[10.0, 0.0], [0.0, 10.0]]),
                maximize=True,
            )
        ],
    )
    unconstrained = _compiled_optimum(network, base, "fmatch")
    assert unconstrained == pytest.approx(10.0, abs=1e-6)

    constrained = base.model_copy(
        update={
            "constraints": [
                DiagramConstraint(
                    symbol="p_d_hi",
                    kind=DiagramConstraintKind.probability,
                    nodes=["D"],
                    func=["Subtract", ["At", "D", 2], 0.5],
                    cons_type=ConstraintTypeEnum.LTE,
                )
            ]
        }
    )
    problem, index_map = compile_diagram(network, constrained)
    result = GurobipySolver(problem).solve("fmatch")
    assert result.success, result.message
    assert result.optimal_objectives["fmatch"] == pytest.approx(3.0, abs=1e-6)
    assert result.optimal_objectives["fmatch"] < unconstrained - 1e-6

    order = _topological_order(network, constrained)
    cardinality = _cardinalities(network, constrained)
    policy_by_decision = _read_back_policy(constrained, index_map, result)

    marginal_hi = _brute_force_marginal(network, constrained, order, cardinality, policy_by_decision, "D", 1)
    assert marginal_hi <= 0.5 + 1e-6

    readback_reward = _expected_utilities(network, constrained, order, cardinality, policy_by_decision)["fmatch"]
    assert readback_reward == pytest.approx(result.optimal_objectives["fmatch"], abs=1e-6)

    def feasible(policy) -> bool:
        return _brute_force_marginal(network, constrained, order, cardinality, policy, "D", 1) <= 0.5 + 1e-9

    brute_force = _brute_force_optimum_feasible(network, constrained, "fmatch", feasible)
    assert result.optimal_objectives["fmatch"] == pytest.approx(brute_force, abs=1e-6)


def test_expected_value_constraint_on_decision_cost_changes_optimum():
    """A `kind == "expected_value"` bound on a weighted cost over D's own states.

    2 * P(D == "s0") + 5 * P(D == "s1") <= 3.0, exercising substitution under `Add`/`Multiply`.
    """
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[2], values=[0.3, 0.7])},
    )
    base = DecisionAnnotation(
        decisions=[DecisionNode(symbol="D", states=["s0", "s1"], information_set=["A"])],
        objectives=[
            ObjectiveNode(
                symbol="fmatch",
                parents=["A", "D"],
                utility=Potential(variables=["A", "D"], shape=[2, 2], values=[[10.0, 0.0], [0.0, 10.0]]),
                maximize=True,
            )
        ],
    )
    unconstrained = _compiled_optimum(network, base, "fmatch")
    assert unconstrained == pytest.approx(10.0, abs=1e-6)

    constrained = base.model_copy(
        update={
            "constraints": [
                DiagramConstraint(
                    symbol="cost_budget",
                    kind=DiagramConstraintKind.expected_value,
                    nodes=["D"],
                    func=[
                        "Subtract",
                        ["Add", ["Multiply", 2.0, ["At", "D", 1]], ["Multiply", 5.0, ["At", "D", 2]]],
                        3.0,
                    ],
                    cons_type=ConstraintTypeEnum.LTE,
                )
            ]
        }
    )
    problem, index_map = compile_diagram(network, constrained)
    result = GurobipySolver(problem).solve("fmatch")
    assert result.success, result.message
    assert result.optimal_objectives["fmatch"] == pytest.approx(3.0, abs=1e-6)
    assert result.optimal_objectives["fmatch"] < unconstrained - 1e-6

    order = _topological_order(network, constrained)
    cardinality = _cardinalities(network, constrained)
    policy_by_decision = _read_back_policy(constrained, index_map, result)

    p_lo = _brute_force_marginal(network, constrained, order, cardinality, policy_by_decision, "D", 0)
    p_hi = _brute_force_marginal(network, constrained, order, cardinality, policy_by_decision, "D", 1)
    assert 2.0 * p_lo + 5.0 * p_hi <= 3.0 + 1e-6

    readback_reward = _expected_utilities(network, constrained, order, cardinality, policy_by_decision)["fmatch"]
    assert readback_reward == pytest.approx(result.optimal_objectives["fmatch"], abs=1e-6)

    def feasible(policy) -> bool:
        lo = _brute_force_marginal(network, constrained, order, cardinality, policy, "D", 0)
        hi = _brute_force_marginal(network, constrained, order, cardinality, policy, "D", 1)
        return 2.0 * lo + 5.0 * hi <= 3.0 + 1e-9

    brute_force = _brute_force_optimum_feasible(network, constrained, "fmatch", feasible)
    assert result.optimal_objectives["fmatch"] == pytest.approx(brute_force, abs=1e-6)


def test_tail_constraint_raises():
    """`kind == "tail"` is rejected at compile time; it needs a risk-measure transform."""
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6])},
    )
    annotation = DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="f1", parents=["A"], utility=Potential(variables=["A"], shape=[2], values=[10.0, 20.0])
            )
        ],
        constraints=[
            DiagramConstraint(
                symbol="tail1",
                kind=DiagramConstraintKind.tail,
                nodes=["f1"],
                func=["At", "f1", 1],
                cons_type=ConstraintTypeEnum.LTE,
            )
        ],
    )
    with pytest.raises(ValueError, match="tail"):
        compile_diagram(network, annotation)


def test_decision_constraint_with_nonempty_information_set_raises():
    """`kind == "decision"` referencing a decision with a non-empty effective info set is rejected."""
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[2], values=[0.3, 0.7])},
    )
    annotation = DecisionAnnotation(
        decisions=[DecisionNode(symbol="D", states=["s0", "s1"], information_set=["A"])],
        objectives=[
            ObjectiveNode(
                symbol="fmatch",
                parents=["A", "D"],
                utility=Potential(variables=["A", "D"], shape=[2, 2], values=[[10.0, 0.0], [0.0, 10.0]]),
                maximize=True,
            )
        ],
        constraints=[
            DiagramConstraint(
                symbol="bad_decision_constraint",
                kind=DiagramConstraintKind.decision,
                nodes=["D"],
                func=["Subtract", ["At", "D", 1], 1],
                cons_type=ConstraintTypeEnum.LTE,
            )
        ],
    )
    with pytest.raises(ValueError, match="effective information set"):
        compile_diagram(network, annotation)


@pytest.mark.parametrize(
    "bad_func",
    [
        ["Add", "A", 1],  # bare symbol leaf, not wrapped in `At`
        ["Sum", ["Extract", "A", ["Tuple", 1, 2]]],  # Extract/Sum target, not `At`
    ],
)
def test_constraint_func_unsupported_node_reference_raises(bad_func):
    """`func` referencing a `nodes` member outside the supported `["At", node, k]` form is rejected."""
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6])},
    )
    annotation = DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="f1", parents=["A"], utility=Potential(variables=["A"], shape=[2], values=[10.0, 20.0])
            )
        ],
        constraints=[
            DiagramConstraint(
                symbol="bad_form",
                kind=DiagramConstraintKind.probability,
                nodes=["A"],
                func=bad_func,
                cons_type=ConstraintTypeEnum.LTE,
            )
        ],
    )
    with pytest.raises(ValueError, match="unsupported form"):
        compile_diagram(network, annotation)


def test_constraint_at_index_out_of_range_raises():
    """An out-of-range `["At", node, k]` state index raises a clear error, not a bare `IndexError`."""
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6])},
    )
    annotation = DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="f1", parents=["A"], utility=Potential(variables=["A"], shape=[2], values=[10.0, 20.0])
            )
        ],
        constraints=[
            DiagramConstraint(
                symbol="out_of_range",
                kind=DiagramConstraintKind.probability,
                nodes=["A"],
                func=["Subtract", ["At", "A", 5], 0.1],
                cons_type=ConstraintTypeEnum.LTE,
            )
        ],
    )
    with pytest.raises(ValueError, match="out of range"):
        compile_diagram(network, annotation)


def test_constraint_multiply_of_two_node_terms_raises():
    """A `Multiply` of two node-derived `At` terms is bilinear, not linear -- rejected."""
    network = DiscreteNetwork(
        variables=[
            DiscreteVariable(name="A", labels=["lo", "hi"]),
            DiscreteVariable(name="B", labels=["lo", "hi"]),
        ],
        arcs=[],
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6]),
            "B": Potential(variables=["B"], shape=[2], values=[0.5, 0.5]),
        },
    )
    annotation = DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="f1", parents=["A"], utility=Potential(variables=["A"], shape=[2], values=[10.0, 20.0])
            )
        ],
        constraints=[
            DiagramConstraint(
                symbol="c2",
                kind=DiagramConstraintKind.probability,
                nodes=["A", "B"],
                func=["Subtract", ["Multiply", ["At", "A", 1], ["At", "B", 1]], 0.1],
                cons_type=ConstraintTypeEnum.LTE,
            )
        ],
    )
    with pytest.raises(ValueError, match="node-derived"):
        compile_diagram(network, annotation)


def test_constraint_referencing_undeclared_node_raises():
    """A node referenced in `func` but missing from `DiagramConstraint.nodes` is rejected."""
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6])},
    )
    annotation = DecisionAnnotation(
        # D observes A, so the diagram is connected (build_rooted_junction_tree requires a single
        # connected component) even though the constraint below deliberately doesn't declare "A".
        decisions=[DecisionNode(symbol="D", states=["s0", "s1"], information_set=["A"])],
        objectives=[
            ObjectiveNode(
                symbol="fD",
                parents=["D"],
                utility=Potential(variables=["D"], shape=[2], values=[5.0, 8.0]),
                maximize=True,
            )
        ],
        constraints=[
            DiagramConstraint(
                symbol="leaky",
                kind=DiagramConstraintKind.probability,
                nodes=["D"],
                func=["Subtract", ["At", "A", 1], 0.5],
                cons_type=ConstraintTypeEnum.LTE,
            )
        ],
    )
    with pytest.raises(ValueError, match="not declared in this constraint's nodes"):
        compile_diagram(network, annotation)


def test_constraint_referencing_colliding_objective_symbol_raises():
    """A bare `func` reference to a name colliding with an `ObjectiveNode.symbol` is rejected."""
    network = DiscreteNetwork(
        variables=[
            DiscreteVariable(name="A", labels=["lo", "hi"]),
            DiscreteVariable(name="B", labels=["lo", "hi"]),
        ],
        # A -> B, so the diagram is connected (build_rooted_junction_tree requires a single
        # connected component) even though the constraint below deliberately doesn't declare "A".
        arcs=[("A", "B")],
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6]),
            "B": Potential(variables=["B", "A"], shape=[2, 2], values=[[0.9, 0.2], [0.1, 0.8]]),
        },
    )
    annotation = DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="A",  # Collides with chance node "A"'s own name -- allowed by the schema.
                parents=["B"],
                utility=Potential(variables=["B"], shape=[2], values=[1.0, 2.0]),
            )
        ],
        constraints=[
            DiagramConstraint(
                symbol="leaky_objective_ref",
                kind=DiagramConstraintKind.probability,
                nodes=["B"],
                func=["Subtract", "A", 0.1],
                cons_type=ConstraintTypeEnum.LTE,
            )
        ],
    )
    with pytest.raises(ValueError, match="not declared in this constraint's nodes"):
        compile_diagram(network, annotation)


def test_constraint_referencing_risk_measure_symbol_raises():
    """A bare `func` reference to a `RiskMeasure.symbol` is rejected, the same as an `ObjectiveNode`."""
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="W", labels=["lose", "win"])],
        arcs=[],
        potentials={"W": Potential(variables=["W"], shape=[2], values=[0.5, 0.5])},
    )
    annotation = DecisionAnnotation(
        decisions=[DecisionNode(symbol="D", states=["safe", "gamble"])],
        objectives=[
            ObjectiveNode(
                symbol="payoff",
                parents=["D", "W"],
                utility=Potential(variables=["D", "W"], shape=[2, 2], values=[[5.0, 5.0], [0.0, 12.0]]),
                maximize=True,
            )
        ],
        risk_measures=[RiskMeasure(symbol="cvar_payoff", target="payoff", alpha=0.5)],
        constraints=[
            DiagramConstraint(
                symbol="leaky_risk_measure_ref",
                kind=DiagramConstraintKind.probability,
                nodes=["D"],
                func=["Subtract", 6.0, "cvar_payoff"],
                cons_type=ConstraintTypeEnum.LTE,
            )
        ],
    )
    with pytest.raises(ValueError, match="not declared in this constraint's nodes"):
        compile_diagram(network, annotation)


def test_constraint_referencing_risk_measure_internal_symbol_raises():
    """A bare `func` reference to a risk measure's own internal `eta` variable is rejected."""
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="W", labels=["lose", "win"])],
        arcs=[],
        potentials={"W": Potential(variables=["W"], shape=[2], values=[0.5, 0.5])},
    )
    annotation = DecisionAnnotation(
        decisions=[DecisionNode(symbol="D", states=["safe", "gamble"])],
        objectives=[
            ObjectiveNode(
                symbol="payoff",
                parents=["D", "W"],
                utility=Potential(variables=["D", "W"], shape=[2, 2], values=[[5.0, 5.0], [0.0, 12.0]]),
                maximize=True,
            )
        ],
        risk_measures=[RiskMeasure(symbol="cvar_payoff", target="payoff", alpha=0.5)],
        constraints=[
            DiagramConstraint(
                symbol="leaky_internal_ref",
                kind=DiagramConstraintKind.probability,
                nodes=["D"],
                func=["Subtract", "eta_cvar_payoff", 100.0],
                cons_type=ConstraintTypeEnum.LTE,
            )
        ],
    )
    with pytest.raises(ValueError, match="not declared in this constraint's nodes"):
        compile_diagram(network, annotation)


def test_constraint_referencing_compiler_internal_mu_symbol_raises():
    """A bare `func` reference to this module's own internal `mu_<node>` symbol is rejected."""
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6])},
    )
    annotation = DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="f1", parents=["A"], utility=Potential(variables=["A"], shape=[2], values=[10.0, 20.0])
            )
        ],
        constraints=[
            DiagramConstraint(
                symbol="leaky_mu_ref",
                kind=DiagramConstraintKind.probability,
                nodes=["A"],
                func=["Subtract", ["Multiply", "mu_A", "mu_A"], 0.1],
                cons_type=ConstraintTypeEnum.LTE,
            )
        ],
    )
    with pytest.raises(ValueError, match="not declared in this constraint's nodes"):
        compile_diagram(network, annotation)


def test_constraint_at_unknown_symbol_raises():
    """`["At", sym, k]` where `sym` isn't even a known node/objective/risk-measure symbol is rejected."""
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6])},
    )
    annotation = DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="f1", parents=["A"], utility=Potential(variables=["A"], shape=[2], values=[10.0, 20.0])
            )
        ],
        constraints=[
            DiagramConstraint(
                symbol="unknown_at_ref",
                kind=DiagramConstraintKind.probability,
                nodes=["A"],
                func=["Subtract", ["At", "Z", 1], 0.5],
                cons_type=ConstraintTypeEnum.LTE,
            )
        ],
    )
    with pytest.raises(ValueError, match="unknown symbol"):
        compile_diagram(network, annotation)


def test_constraint_node_in_operator_position_raises():
    """A node symbol appearing in operator position (`func[0]`) is rejected."""
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6])},
    )
    annotation = DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="f1", parents=["A"], utility=Potential(variables=["A"], shape=[2], values=[10.0, 20.0])
            )
        ],
        constraints=[
            DiagramConstraint(
                symbol="node_as_operator",
                kind=DiagramConstraintKind.probability,
                nodes=["A"],
                func=["Subtract", ["A", 1], 0.5],
                cons_type=ConstraintTypeEnum.LTE,
            )
        ],
    )
    with pytest.raises(ValueError, match="operator position"):
        compile_diagram(network, annotation)


def test_probability_constraint_on_chance_node_via_nonowning_cluster():
    """A `kind == "probability"` bound on a chance node read out of a *non-owning* cluster's mu.

    `X`'s own cluster (state space 48) is larger than `Y`'s (`{X, Y}`, 6), so
    `find_containing_cluster(["X"])` picks `Y`'s. `X`'s CPT ignores its parents, so P(X == k) is
    read directly off it.
    """
    network = DiscreteNetwork(
        variables=[
            DiscreteVariable(name="P1", labels=["a", "b", "c", "d"]),
            DiscreteVariable(name="P2", labels=["a", "b", "c", "d"]),
            DiscreteVariable(name="X", labels=["x0", "x1", "x2"]),
            DiscreteVariable(name="Y", labels=["y0", "y1"]),
        ],
        arcs=[("P1", "X"), ("P2", "X"), ("X", "Y")],
        potentials={
            "P1": Potential(variables=["P1"], shape=[4], values=[0.1, 0.2, 0.3, 0.4]),
            "P2": Potential(variables=["P2"], shape=[4], values=[0.4, 0.3, 0.2, 0.1]),
            "X": Potential(
                variables=["X", "P1", "P2"],
                shape=[3, 4, 4],
                # Same distribution over X for every (P1, P2) combination -- P(X == k) is then
                # just cpt_x[k], independent of P1/P2's own marginals.
                values=[[[0.2] * 4] * 4, [[0.5] * 4] * 4, [[0.3] * 4] * 4],
            ),
            "Y": Potential(variables=["Y", "X"], shape=[2, 3], values=[[0.6, 0.5, 0.4], [0.4, 0.5, 0.6]]),
        },
    )
    annotation = DecisionAnnotation(
        objectives=[
            ObjectiveNode(symbol="fY", parents=["Y"], utility=Potential(variables=["Y"], shape=[2], values=[1.0, 2.0]))
        ],
    )
    rjt = build_rooted_junction_tree(network, annotation)
    assert rjt.find_containing_cluster(["X"]) == "Y"

    def make(bound: float) -> DecisionAnnotation:
        return annotation.model_copy(
            update={
                "constraints": [
                    DiagramConstraint(
                        symbol="p_x1",
                        kind=DiagramConstraintKind.probability,
                        nodes=["X"],
                        func=["Subtract", ["At", "X", 2], bound],
                        cons_type=ConstraintTypeEnum.LTE,
                    )
                ]
            }
        )

    feasible_problem, index_map = compile_diagram(network, make(0.6))
    feasible_result = GurobipySolver(feasible_problem).solve("fY")
    assert feasible_result.success, feasible_result.message

    cluster_map = index_map.clusters["Y"]
    assert cluster_map.nodes == ["X", "Y"]
    mu_y = feasible_result.optimal_variables[cluster_map.mu_symbol]
    marginal_x1 = mu_y[2] + mu_y[3]  # X == state index 1 (0-based), Y summed out.
    assert marginal_x1 == pytest.approx(0.5, abs=1e-6)

    infeasible_problem, _ = compile_diagram(network, make(0.4))
    infeasible_result = GurobipySolver(infeasible_problem).solve("fY")
    assert not infeasible_result.success


# ── RiskMeasure / CVaR compilation ──────────────────────────────────────────────────────────────


def test_cvar_hand_computed_maximize():
    """A hand-computed CVaR value, not just self-consistency against `_brute_force_cvar`.

    A has 3 states, probabilities [0.5, 0.3, 0.2], utilities [10, 20, 30] (maximize). At
    alpha=0.6, sorted ascending: u=10 (mass 0.5, fully included, remaining 0.1), u=20 (mass 0.3,
    0.1 of it included to reach the 0.6 tail exactly) -- CVaR_0.6 = (0.5*10 + 0.1*20) / 0.6 =
    7.0 / 0.6.
    """
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["s0", "s1", "s2"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[3], values=[0.5, 0.3, 0.2])},
    )
    annotation = DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="f1",
                parents=["A"],
                utility=Potential(variables=["A"], shape=[3], values=[10.0, 20.0, 30.0]),
                maximize=True,
            )
        ],
        risk_measures=[RiskMeasure(symbol="cvar_f1", target="f1", alpha=0.6)],
    )
    expected = (0.5 * 10.0 + 0.1 * 20.0) / 0.6

    compiled = _compiled_optimum(network, annotation, "cvar_f1")
    assert compiled == pytest.approx(expected, abs=1e-6)

    brute_force = _brute_force_cvar(network, annotation, "f1", 0.6)
    assert compiled == pytest.approx(brute_force, abs=1e-6)


def test_cvar_hand_computed_minimize_sign_convention():
    """The same distribution/values as above, but as a cost to *minimize*.

    The tail is now the highest costs: (0.2*30 + 0.3*20 + 0.1*10) / 0.6 = 13.0 / 0.6.
    """
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["s0", "s1", "s2"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[3], values=[0.5, 0.3, 0.2])},
    )
    annotation = DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="cost",
                parents=["A"],
                utility=Potential(variables=["A"], shape=[3], values=[10.0, 20.0, 30.0]),
                maximize=False,
            )
        ],
        risk_measures=[RiskMeasure(symbol="cvar_cost", target="cost", alpha=0.6)],
    )
    expected = (0.2 * 30.0 + 0.3 * 20.0 + 0.1 * 10.0) / 0.6

    compiled = _compiled_optimum(network, annotation, "cvar_cost")
    assert compiled == pytest.approx(expected, abs=1e-6)

    brute_force = _brute_force_cvar(network, annotation, "cost", 0.6)
    assert compiled == pytest.approx(brute_force, abs=1e-6)


def test_cvar_degenerate_single_utility_value():
    """A constant utility (|U| == 1): the epsilon/M degenerate case in `_cvar_epsilon`/`big_m`."""
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[2], values=[0.5, 0.5])},
    )
    annotation = DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="f1",
                parents=["A"],
                utility=Potential(variables=["A"], shape=[2], values=[7.0, 7.0]),
                maximize=True,
            )
        ],
        risk_measures=[RiskMeasure(symbol="cvar_f1", target="f1", alpha=0.37)],
    )
    compiled = _compiled_optimum(network, annotation, "cvar_f1")
    assert compiled == pytest.approx(7.0, abs=1e-6)


def test_risk_averse_decision_prefers_safe_over_risky_bet():
    """EV-optimal and CVaR-optimal policies differ.

    D chooses a safe payoff of 5 or a gamble paying 0 or 12 with equal probability. EV prefers the
    gamble (6 > 5); at alpha=0.5, CVaR prefers safe (5 > 0).
    """
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="W", labels=["lose", "win"])],
        arcs=[],
        potentials={"W": Potential(variables=["W"], shape=[2], values=[0.5, 0.5])},
    )
    annotation = DecisionAnnotation(
        decisions=[DecisionNode(symbol="D", states=["safe", "gamble"])],
        objectives=[
            ObjectiveNode(
                symbol="payoff",
                parents=["D", "W"],
                utility=Potential(variables=["D", "W"], shape=[2, 2], values=[[5.0, 5.0], [0.0, 12.0]]),
                maximize=True,
            )
        ],
        risk_measures=[RiskMeasure(symbol="cvar_payoff", target="payoff", alpha=0.5)],
    )

    unconstrained_ev = _compiled_optimum(network, annotation, "payoff")
    assert unconstrained_ev == pytest.approx(6.0, abs=1e-6)  # EV-optimal policy gambles.

    compiled_cvar = _compiled_optimum(network, annotation, "cvar_payoff")
    assert compiled_cvar == pytest.approx(5.0, abs=1e-6)  # CVaR-optimal policy plays safe.

    brute_force_cvar = _brute_force_cvar(network, annotation, "payoff", 0.5)
    assert compiled_cvar == pytest.approx(brute_force_cvar, abs=1e-6)


def _pig_farm_network() -> DiscreteNetwork:
    """The single-value-node pig farm problem, reduced to one treatment period.

    Herrala, Terho & Oliveira 2025, Section 2.1: healthy price 1000 DKK, ill price 300 DKK,
    treatment cost 100 DKK, test specificity 80%/sensitivity 90%, cure probability 0.5 with
    treatment / 0.1 without, disease-development probability 0.1 with treatment / 0.2 without,
    initial disease probability 0.1.

    The treatment-dependent transition is modelled as two counterfactual chance nodes conditioned
    on `H1` only, `CT` (treated) and `CN` (not treated); the utility table selects the one
    matching `D1`. This gives the same joint distribution as a decision-conditioned CPT.
    """
    return DiscreteNetwork(
        name="pig_farm",
        variables=[
            DiscreteVariable(name="H1", labels=["healthy", "ill"]),
            DiscreteVariable(name="T1", labels=["neg", "pos"]),
            DiscreteVariable(name="CT", labels=["healthy", "ill"]),
            DiscreteVariable(name="CN", labels=["healthy", "ill"]),
        ],
        arcs=[("H1", "T1"), ("H1", "CT"), ("H1", "CN")],
        potentials={
            "H1": Potential(variables=["H1"], shape=[2], values=[0.9, 0.1]),
            # Specificity 80% (P(neg|healthy)=0.8), sensitivity 90% (P(pos|ill)=0.9).
            "T1": Potential(variables=["T1", "H1"], shape=[2, 2], values=[[0.8, 0.1], [0.2, 0.9]]),
            # Treated: cure prob 0.5 if ill; disease-development prob 0.1 if healthy.
            "CT": Potential(variables=["CT", "H1"], shape=[2, 2], values=[[0.9, 0.5], [0.1, 0.5]]),
            # Untreated: spontaneous cure prob 0.1 if ill; disease-development prob 0.2 if healthy.
            "CN": Potential(variables=["CN", "H1"], shape=[2, 2], values=[[0.8, 0.1], [0.2, 0.9]]),
        },
    )


def _pig_farm_annotation(alpha: float) -> DecisionAnnotation:
    """`_pig_farm_network`'s decision/objective/risk overlay.

    One treatment decision informed by the test result, one profit objective (market price minus
    treatment cost), and a `RiskMeasure` at `alpha` (the paper's Section 4.1 uses `alpha = 0.15`).
    """
    return DecisionAnnotation(
        decisions=[DecisionNode(symbol="D1", states=["treat", "notreat"], information_set=["T1"])],
        objectives=[
            ObjectiveNode(
                symbol="profit",
                parents=["D1", "CT", "CN"],
                utility=Potential(
                    variables=["D1", "CT", "CN"],
                    shape=[2, 2, 2],
                    # D1=treat: 1000/300 DKK market price by CT's outcome, minus the 100 DKK
                    # treatment cost, independent of CN (never actually realized under this
                    # choice). D1=notreat: 1000/300 DKK by CN's outcome, no cost, independent of CT.
                    values=[
                        [[900.0, 900.0], [200.0, 200.0]],
                        [[1000.0, 300.0], [1000.0, 300.0]],
                    ],
                ),
                maximize=True,
            )
        ],
        risk_measures=[RiskMeasure(symbol="cvar_profit", target="profit", alpha=alpha)],
    )


def test_pig_farm_cvar_matches_brute_force_oracle():
    """The paper's running example (Section 2.1) at its Section 4.1 level, `alpha = 0.15`.

    The paper publishes no exact CVaR value, so this checks against `_brute_force_cvar`, and that
    the read-back `eta` lies within the utility range.
    """
    network = _pig_farm_network()
    annotation = _pig_farm_annotation(alpha=0.15)

    problem, index_map = compile_diagram(network, annotation)
    result = GurobipySolver(problem).solve("cvar_profit")
    assert result.success, result.message
    compiled = result.optimal_objectives["cvar_profit"]

    brute_force = _brute_force_cvar(network, annotation, "profit", 0.15)
    assert compiled == pytest.approx(brute_force, abs=1e-6)

    risk_map = index_map.risk_measures["cvar_profit"]
    assert risk_map.target == "profit"
    assert risk_map.target_maximizes is True
    eta_value = result.optimal_variables[risk_map.eta_symbol]
    assert min(risk_map.utility_values) - 1e-6 <= eta_value <= max(risk_map.utility_values) + 1e-6


@pytest.mark.parametrize(
    ("seed", "n", "domain_size", "ratio_arc", "alpha"),
    # gum.randomBN refuses n <= 3 ("A BN can not be randomly generated from less than 4 nodes.").
    [(301, 4, 2, 1.2, 0.3), (302, 4, 2, 1.0, 0.6), (303, 5, 3, 1.5, 0.25)],
)
def test_random_small_diagrams_cvar_matches_brute_force(seed, n, domain_size, ratio_arc, alpha):
    """A random small BN plus a grafted-on decision/objective/risk measure, against `_brute_force_cvar`."""
    gum.initRandom(seed)
    network = from_pyagrum(gum.randomBN(n=n, domain_size=domain_size, ratio_arc=ratio_arc))
    base = _graft_decision_and_objective(network, np.random.default_rng(seed))
    annotation = base.model_copy(update={"risk_measures": [RiskMeasure(symbol="cvar_obj", target="obj", alpha=alpha)]})

    compiled = _compiled_optimum(network, annotation, "cvar_obj")
    brute_force = _brute_force_cvar(network, annotation, "obj", alpha)
    assert compiled == pytest.approx(brute_force, abs=1e-6)


def test_cvar_approaches_expected_value_and_is_bounded_and_monotone():
    """Two relationships from Rockafellar & Uryasev (2000), on a no-decision diagram.

    - `CVaR_alpha -> E[utility]` as `alpha -> 1`: at `alpha=0.999`, within
      `(1 - alpha) * (max(u) - min(u))` of the expected-value objective `f1` from the same solve.
    - `CVaR_alpha <= E[utility]` for every `alpha`, and `CVaR_alpha` is non-decreasing in `alpha`.
    """
    utility_values = [10.0, 12.0, 15.0, 20.0]
    probabilities = [0.4, 0.3, 0.2, 0.1]
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["s0", "s1", "s2", "s3"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[4], values=probabilities)},
    )
    alphas = [0.1, 0.3, 0.5, 0.7, 0.9, 0.999]
    symbols = [f"cvar_{int(a * 1000)}" for a in alphas]
    annotation = DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="f1",
                parents=["A"],
                utility=Potential(variables=["A"], shape=[4], values=utility_values),
                maximize=True,
            )
        ],
        risk_measures=[
            RiskMeasure(symbol=symbol, target="f1", alpha=alpha) for symbol, alpha in zip(symbols, alphas, strict=True)
        ],
    )

    problem, _ = compile_diagram(network, annotation)
    result = GurobipySolver(problem).solve("f1")
    assert result.success, result.message

    expected_value = result.optimal_objectives["f1"]
    assert expected_value == pytest.approx(sum(p * u for p, u in zip(probabilities, utility_values, strict=True)))

    cvars = [result.optimal_objectives[symbol] for symbol in symbols]

    for alpha, cvar in zip(alphas, cvars, strict=True):
        assert cvar <= expected_value + 1e-6, f"CVaR_{alpha} exceeded E[U]"

    for earlier, later in itertools.pairwise(cvars):
        assert later >= earlier - 1e-6  # non-decreasing in alpha

    value_range = max(utility_values) - min(utility_values)
    tolerance = (1.0 - 0.999) * value_range + 1e-9
    assert cvars[-1] == pytest.approx(expected_value, abs=tolerance)

    # Cross-check the alpha=0.999 case directly against `_brute_force_cvar` too.
    brute_force_0999 = _brute_force_cvar(network, annotation, "f1", 0.999)
    assert cvars[-1] == pytest.approx(brute_force_0999, abs=1e-6)
