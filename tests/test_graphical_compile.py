"""Tests for `desdeo.problem.graphical.compile`.

Phase 4 of `.claude/desdeo-graphical-models-plan.md`. Per the plan's checklist, the oracle tests
are built before (well, alongside) the compiler, not after, and every case is checked against
*two* independent oracles plus the compiler under test:

1. **Brute force** (`_brute_force_optimum`): enumerates every deterministic policy and every
   chance-node joint state directly in plain Python/NumPy, with no dependency on this package's
   own RJT/MILP machinery (only `kahn_topological_order` and `effective_information_sets`, which
   are Phase 1/3 utilities *about* the diagram's semantics, not about compiling one). Obviously
   correct, catches everything, but scales combinatorially -- kept to tiny diagrams. The only
   oracle that supports multi-objective annotations (`_expected_utilities` already returns a
   dict keyed by objective symbol) -- see `_pyagrum_meu`'s own docstring for why pyAgrum can't.
2. **pyAgrum** (`_to_pyagrum_influence_diagram` + `ShaferShenoyLIMIDInference`/`LazyPropagation`):
   builds the same diagram directly in pyAgrum and asks its own inference engine for the MEU.
   Restricted to single-objective annotations -- `posteriorUtility(node)`, the per-utility-node
   escape hatch the Phase 2 findings expected to need for multi-objective comparison, turns out
   *not* to collapse to a scalar (confirmed empirically; see `_pyagrum_meu`'s docstring), so
   `MEU()['mean']` is used instead, matching what the plan's checklist actually asked for
   ("single-objective unconstrained cases"). `addNoForgettingAssumption` is required to match
   classical (non-LIMID) semantics whenever `annotation.no_forgetting` is set (not merely
   `decision_order` -- also confirmed empirically, see `_pyagrum_meu`).
3. Property-based generation of small random Bayesian networks (`gum.randomBN`, same style as
   `test_graphical_pyagrum_adapter.py`) with a decision and an objective grafted on, checked
   against both oracles above.

pyAgrum is a hard dependency (see the plan's design decisions) but every test here also solves
the compiled MILP via gurobipy, so the module is marked both `graphical` and `gurobipy` for
selection (`pytest -m "graphical and gurobipy"`), not skip -- `pytest -m "not gurobipy"` still
deselects it correctly for a machine/CI run without a Gurobi license.

**Phase 5 additions** (`_build_diagram_constraint`, `DiagramConstraint` compilation): the
"decision"-kind and "probability"/"expected_value"-kind cases below get their own oracle checks,
since neither existing oracle (`_pyagrum_meu` doesn't model diagram-level side constraints at all;
`_brute_force_optimum` enumerates *every* policy, not just feasible ones) covers a constrained
diagram as-is. `_brute_force_optimum_feasible` restricts `_brute_force_optimum`'s own enumeration
to policies passing a feasibility predicate (decision-kind); `_brute_force_marginal` computes
P(node == state) under one fixed policy directly from the network's own CPTs, independent of the
compiler's mu/`IndexMap` machinery (probability/expected_value-kind, read back via `_read_back_policy`
-- factored out of what was `_read_back_policy_and_reverify`'s own inline block, now shared).

**Phase 6 additions** (`_build_risk_measure`, `RiskMeasure`/CVaR compilation, Herrala, Terho &
Oliveira 2025): the paper's own Section 4/Appendix A only report solve-time plots for their
risk-averse pig farm experiment, no exact reproducible published CVaR number, so validation here
does not (cannot) check against a published numeric value -- see `test_pig_farm_cvar_matches_
brute_force_oracle`'s docstring. Instead: (1) `_brute_force_cvar` (built on
`_utility_distribution_for_policy` + `_cvar_from_distribution`, both independent of
`_build_risk_measure`'s own eta/lambda/rho machinery -- the textbook VaR/CVaR definition applied
directly to a from-first-principles utility distribution) is the CVaR analogue of
`_brute_force_optimum`, checked against a literal instance of the paper's own pig farm running
example (Section 2.1) plus several small hand-built and property-based random diagrams; (2) two
literature relationships from Rockafellar & Uryasev (2000, cited by the paper itself for CVaR):
`CVaR_alpha -> E[utility]` as `alpha -> 1`, checked at `alpha=0.999` against Phase 4's own
already-oracle-tested `_build_objective` expected-value `Objective`, compiled *side by side on the
same `Problem`* as the `RiskMeasure`'s own `Objective` -- a real regression-style cross-check
between two independently-built code paths in this module, not just self-consistency; and
`CVaR_alpha <= E[utility]` for every `alpha`, plus `CVaR_alpha` monotone non-decreasing in `alpha`.
"""

import itertools

import numpy as np
import pyagrum as gum
import pytest

from desdeo.problem.graphical import (
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
    build_rooted_junction_tree,
    compile_diagram,
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
    """`_brute_force_optimum`, restricted to policies for which `feasible(policy)` is true.

    `policy` is the same `{decision_symbol: (info_order, {info_state: chosen_state})}` mapping
    `_enumerate_policies` yields -- the ground truth for a `kind == "decision"` `DiagramConstraint`
    test, which is exactly a hard restriction on which deterministic policies are legal.
    """
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
    """P(`node` == `state_index`) under one fixed deterministic policy.

    Direct from the network's own CPTs -- independent of the compiler's own mu/`IndexMap`
    marginalisation machinery, so it's a real check of a `kind in {"probability",
    "expected_value"}` `DiagramConstraint`'s effect, not a restatement of it. Same iteration shape
    as `_expected_utilities`, accumulating probability mass instead of utility.
    """
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
    """`{utility_value: probability_mass}` for one objective under one fixed deterministic policy.

    Same iteration shape as `_expected_utilities`, but bucketing probability mass by utility value
    instead of summing `prob * util` into one scalar -- the full realizable distribution a Phase 6
    `RiskMeasure`/CVaR oracle needs, independent of this package's own mu/`_build_risk_measure`
    machinery (only network CPTs and the objective's own utility table are read).
    """
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

    Sort ascending, accumulate probability mass until `alpha` is reached (the boundary value's own
    mass taken fractionally so the tail used is exactly `alpha`, matching VaR_alpha's definition as
    the alpha-quantile), and return the probability-weighted average of that tail. No dependency on
    this package's own `_build_risk_measure`/eta/lambda/rho machinery -- a from-first-principles
    oracle, not a restatement of the MILP.
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
    """The exact CVaR_alpha optimum for `objective_symbol`.

    Enumerates every deterministic policy and computes each one's full utility distribution from
    first principles. Mirrors `_brute_force_optimum`'s policy search, but optimizes
    `_cvar_from_distribution` (in the same sign convention `_build_risk_measure`'s compiled
    `Objective` uses -- see the module docstring's "Sign convention for a minimizing target":
    negate the distribution, take the lower-tail CVaR of *that*, negate back) instead of the plain
    expectation.
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
# `_to_pyagrum_influence_diagram` was promoted to a public adapter function,
# `adapters.pyagrum.to_pyagrum_influence_diagram`, since Phase 7's `/graphical/whatif` needs the
# exact same diagram-building logic outside this test module. Imported above, not redefined here.


def _pyagrum_meu(network: DiscreteNetwork, annotation: DecisionAnnotation, objective_symbol: str) -> float:
    """The expected utility for `objective_symbol`, computed via pyAgrum's own inference.

    Restricted to single-objective diagrams, matching the plan's own oracle-test scoping ("pyAgrum
    `ShaferShenoyLIMIDInference.MEU()` for single-objective unconstrained cases") -- `MEU()` sums
    every utility node into one scalar (the Phase 2 finding), and `posteriorUtility(node)` turns
    out *not* to be the per-node escape hatch that finding suggested: it stays indexed by that
    node's own remaining decision/chance parents rather than collapsing to a single expected
    value (confirmed empirically -- it returns the raw, un-reduced utility table), so recovering
    one objective's share of a multi-objective MEU needs more than reading it directly. Out of
    scope here; every test case in this module has exactly one `ObjectiveNode`.

    `ShaferShenoyLIMIDInference` also refuses to run at all ("This LIMID/Influence Diagram is not
    solvable") on a diagram with *zero* decision nodes -- unsurprising, since ID solving is
    fundamentally about policies -- so the decision-less case (a plain expectation over a
    Bayesian network) uses a different pyAgrum entry point: `LazyPropagation` over the plain
    `BayesNet` part, contracted against the utility table directly.
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
        # `decision_order` alone is *not* enough to trigger this: per `DecisionAnnotation`'s own
        # docstring, `no_forgetting` is a separate flag that must also be set -- `decision_order`
        # without it still means each decision uses only its own declared `information_set`
        # (LIMID semantics), just with an extra temporal-ordering constraint. Confirmed empirically
        # that this overload wants decision *names*, not the node ids the docstring itself claims
        # -- the id overload raises `NotFound: No element with the key <>` even for a plainly
        # valid id.
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

    Factored out of what was `_read_back_policy_and_reverify`'s own inline block, so Phase 5's
    probability/expected_value constraint tests can also read back the actual solved policy (to
    compute the marginal a `DiagramConstraint` bounds) without duplicating this bookkeeping.
    """
    policy_by_decision = {}
    for decision in annotation.decisions:
        decision_map = index_map.decisions[decision.symbol]
        z_values = result.optimal_variables[decision_map.z_symbol]
        n_states = len(decision_map.states)
        n_info_states = len(z_values) // n_states

        policy: dict[tuple[int, ...], int] = {}
        for info_flat in range(n_info_states):
            block = z_values[info_flat * n_states : (info_flat + 1) * n_states]
            chosen = max(range(n_states), key=lambda s: block[s])
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

    Exercises `IndexMap` itself -- the plan calls it "load-bearing for reading policies back out
    of solutions" (Phase E), but nothing else in this module reads it. Every other assertion in
    this suite only checks the compiled *objective value*; a silently permuted
    `DecisionIndexMap.information_set`/`states` ordering would still pass all of those (the MILP
    itself doesn't care what order `IndexMap` reports) yet break exactly this readback.
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

    # pyAgrum's own solver refuses this second diagram outright ("This LIMID/Influence Diagram
    # is not solvable") -- confirmed empirically that `decisionOrderExists()` is False for it:
    # with no_forgetting off, D2 has no information set and no arc from/through D1 either, so
    # pyAgrum can't derive *any* decision order from the graph, even though brute force and the
    # compiler both handle "decisions with no temporal graph link between them" just fine (there
    # is nothing to order -- each is an independent choice). So only brute force checks this half.
    brute_force_without_memory = _brute_force_optimum(network, without_memory, "fchain")
    compiled_without_memory = _compiled_optimum(network, without_memory, "fchain")
    assert compiled_without_memory == pytest.approx(brute_force_without_memory, abs=1e-6)

    with_memory_value = _compiled_optimum(network, with_memory, "fchain")
    assert with_memory_value >= compiled_without_memory - 1e-9


def test_multi_objective_including_all_zero_utility():
    """Two objectives compiled together, one with an all-zero utility table.

    Regression test for a bug the code review caught: an all-zero utility table used to compile
    to `func = [0.0]`, which looks like valid MathJSON (a single-literal terminal) but isn't a
    node any backend's parser matches as an entire `func` -- and since `GurobipyEvaluator` parses
    every objective at construction, one degenerate objective made the *whole* compiled `Problem`
    unsolvable, not just its own. pyAgrum's own oracle can't check multi-objective diagrams (see
    `_pyagrum_meu`'s docstring), so this is checked against brute force only -- `_expected_utilities`
    already returns a value per objective symbol, exactly for this case.
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


# ── Phase 5: DiagramConstraint compilation ──────────────────────────────────────────────────────


def test_decision_constraint_exclusivity_changes_optimum():
    """A `kind == "decision"` exclusivity constraint over two unconditional decisions' states.

    Checked against `_brute_force_optimum_feasible`, restricted to policies that actually satisfy
    the same exclusivity condition -- the "enumerate only feasible policies" oracle the task asks
    for, not just a before/after objective-value comparison.
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

    D perfectly matching A (the unconstrained optimum) would put P(D == "s1") == P(A == "hi")
    == 0.7, violating the bound -- so the constraint is genuinely binding. Checked against
    `_brute_force_marginal` (independent of the compiler's own mu machinery) on the actual solved
    policy, read back via `IndexMap` the same way `_read_back_policy_and_reverify` does -- and,
    with real power (unlike a bare "the returned policy satisfies the bound" check, which every
    policy in this tiny 4-policy space would trivially pass or fail regardless of whether the
    *optimum* was computed correctly), against `_brute_force_optimum_feasible`'s own enumeration
    restricted to the same bound via `_brute_force_marginal`.
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

    Same diagram as the probability test above, but the constraint is a genuine weighted linear
    combination (2 * P(D == "s0") + 5 * P(D == "s1") <= 3.0) rather than a single state's
    probability -- `probability`/`expected_value` compile identically (see the module docstring),
    so this exercises that the substitution composes correctly under `Add`/`Multiply`, not just a
    lone `At`. Checked against `_brute_force_optimum_feasible`, restricted by the same weighted
    bound via `_brute_force_marginal`, for real oracle power (see the probability test above).
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
    """`kind == "tail"` is rejected at compile time -- Phase 6's risk-measure transform, not this."""
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
    """A `Multiply` of two node-derived `At` terms is bilinear, not linear -- rejected.

    Regression test for the code review's most severe finding: this used to compile silently
    with `is_linear=True`/`is_convex=True`/`is_twice_differentiable=True` hardcoded regardless,
    corrupting `Problem.is_linear` (an `all()` over every element -- one lying constraint poisons
    solver selection for the whole `Problem`) and risking a genuinely quadratic constraint being
    handed to a pure-MILP solver on the strength of that false flag. `["Multiply", At_A, At_B]`
    would compute an independence-assuming *product* of two marginals, not a joint-event
    probability either way -- out of this compiler's scope regardless (see the module docstring).
    """
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
    """A node referenced in `func` but missing from `DiagramConstraint.nodes` is rejected.

    Regression test: this used to compile cleanly and only fail at *solve* time with an opaque
    `GurobipyEvaluatorError: No expression with name A found in the gurobipy model`, naming
    neither the constraint nor the missing `nodes` entry -- exactly the class of error this module
    claims to prevent at compile time.
    """
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
    """A bare `func` reference to a name colliding with an `ObjectiveNode.symbol` is rejected.

    Regression test for the code review's second finding, collision variant: nothing in the
    schema forbids an `ObjectiveNode.symbol` from colliding with a chance-node name (only
    decision/chance collisions are checked -- see `DecisionAnnotation.validate_against`). Before
    the fix, a bare reference like this fell straight through `_substitute_at_nodes` unsubstituted
    (its scope check only covered `diagram_constraint.nodes`) and would have silently resolved at
    solve time to whatever compiled `Problem` symbol happens to share that name -- here, the
    objective's own compiled expression -- instead of erroring, producing a constraint meaning
    something completely different from what was written, with no error anywhere.
    """
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
    """A bare `func` reference to a `RiskMeasure.symbol` is rejected, the same as an `ObjectiveNode`.

    Regression test for a Phase 6 code-review finding: `RiskMeasure.symbol` compiles to a real
    `Objective` in the emitted `Problem`, exactly like an `ObjectiveNode.symbol` -- but
    `compile_diagram`'s `objective_symbols` (built only from `annotation.objectives`) used to leave
    every risk-measure symbol invisible to both `_build_diagram_constraint`'s `referenced_objectives`
    rejection and `_substitute_at_nodes`'s undeclared-reference guard. Before the fix, a
    `probability`/`expected_value`-kind constraint referencing a risk measure's own symbol compiled
    silently and was *actually enforced* as a live bound on the compiled CVaR value -- a working
    back-door around the deliberate `kind == "tail"` rejection.
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
    """A bare `func` reference to a risk measure's own internal `eta` variable is rejected.

    Regression test for the same Phase 6 code-review finding as above, internal-symbol variant:
    `eta_<risk_measure_symbol>` (and the `lambda`/`lambda_bar`/`rho`/`rho_bar` symbols alongside it)
    are this module's own reserved MILP implementation variables, never meant to be reachable from
    a `DiagramConstraint` under any `kind`. Before the fix this silently bound against the internal
    eta variable directly and only failed at *solve* time with a bare `IndexError` from inside the
    evaluator -- exactly the "silently passed through unsubstituted" failure mode the module
    docstring says is supposed to be rejected at compile time.
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


def test_probability_constraint_on_chance_node_via_nonowning_cluster():
    """A `kind == "probability"` bound on a chance node read out of a *non-owning* cluster's mu.

    `RootedJunctionTree.find_containing_cluster([node])` doesn't always return `node`'s own
    cluster: here `X` (cardinality 3) has two cardinality-4 CPT parents `P1`/`P2`, so `X`'s own
    cluster (`{P1, P2, X}`, state space 48) is larger than `Y`'s cluster (`{X, Y}`, state space 6
    -- `Y`'s only CPT parent is `X`), so `find_containing_cluster(["X"])` picks `Y`'s cluster
    instead (the "smallest by state_space_size" rule -- see `Cluster.find_containing_cluster`).
    This is exactly the configuration most likely to hide an index/ordering bug in the
    marginal-substitution machinery -- nothing else in this suite exercises it. `X`'s CPT is
    deliberately independent of `P1`/`P2`'s actual values (the same distribution over `X` for
    every `(P1, P2)` combination), so the true marginal P(X == k) is just that fixed
    distribution's own k'th entry, checkable by inspection rather than by hand-summing a 4x4 joint.
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


# ── Phase 6: RiskMeasure / CVaR compilation ─────────────────────────────────────────────────────


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

    Exercises `_build_risk_measure`'s sign-flip path directly: internally CVaR is computed over
    the negated (maximize-convention) values, then negated back -- the result should be the
    risk-averse average of the worst (highest) alpha-tail *costs*, i.e.
    (0.2*30 + 0.3*20 + 0.1*10) / 0.6 = 13.0 / 0.6 (compare against the maximize case above: here
    the *high* values 30/20 dominate the tail, not the low value 10, since a cost-minimizer's
    risk-averse tail is the expensive outcomes).
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
    """A constant utility (|U| == 1): the epsilon/M degenerate case in `_cvar_epsilon`/`big_m`.

    CVaR_alpha of a one-point distribution is that point, for any alpha -- checked directly, not
    just against `_brute_force_cvar` (which would degenerate the same way for the same reason).
    """
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
    """A genuine EV-optimal vs. CVaR-optimal policy divergence -- the point of a risk measure.

    D chooses between a deterministic "safe" payoff of 5 and a "gamble" that pays 0 or 12 with
    equal probability (EV 6 > 5, so an *expected-value*-maximizing policy always gambles -- see
    `unconstrained_ev` below). At alpha=0.5, CVaR_0.5(gamble) = 0.0 (the worst half of its own
    outcomes is the losing branch) while CVaR_0.5(safe) = 5.0 (constant, so equal to itself for
    any alpha) -- a risk-averse policy should choose "safe" instead, the opposite of the
    EV-optimal choice. Checked against `_brute_force_cvar`'s own independent policy search, not
    just the hand-computed values.
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

    Herrala, Terho & Oliveira 2025, Section 2.1 -- exact parameters from the paper, confirmed
    against the PDF directly (page 2): healthy price 1000 DKK, ill price 300 DKK, treatment cost
    100 DKK, test specificity 80%/sensitivity 90%, cure probability 0.5 with treatment / 0.1
    without (spontaneous), disease-development probability 0.1 with treatment / 0.2 without,
    initial disease probability 0.1.

    **A genuine reformulation, not a simplification of the numbers.** DESDEO's `DiscreteNetwork`
    cannot express the paper's own structure directly: a chance node's CPT may only be conditioned
    on other chance nodes, never a decision (`rooted_junction_tree.py`'s own module docstring,
    Phase 3's confirmed finding), but the pig farm ID's core feature is exactly that the
    treatment decision D1 changes the *chance* transition into the next period's health state --
    an arc pyAgrum's own `InfluenceDiagram` supports (decisions are ordinary DAG parents there)
    but this package's `DiscreteNetwork`/`DecisionAnnotation` split deliberately doesn't allow.
    Reformulated here as two independent chance nodes, both conditioned on `H1` only (never on
    `D1`): `CT` ("counterfactual outcome if treated") and `CN` ("counterfactual outcome if not
    treated") -- both are always drawn, jointly with `H1`, regardless of which choice `D1` actually
    makes, and the `ObjectiveNode`'s own utility table (built by `_pig_farm_annotation`) selects
    the branch matching `D1`'s actual value while ignoring the other. Since `CT`/`CN` are
    conditionally independent given `H1` by construction, this reproduces exactly the same joint
    distribution over (H1, D1, realized outcome) as the paper's own decision-conditioned CPT would
    -- a faithful reformulation forced by this compiler's current expressiveness, not a different
    problem. Worth a maintainer's attention if a future phase wants genuinely decision-conditioned
    chance nodes.
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
    treatment cost), and a `RiskMeasure` at `alpha` (the paper's own Section 4.1 experiment uses
    `1 - alpha = 0.85`, i.e. `alpha = 0.15`, confirmed against the PDF page 7 -- see
    `test_pig_farm_cvar_matches_brute_force_oracle`).
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
    """A literal instance of the paper's own running example (Section 2.1).

    Compiled at the paper's own Section 4.1 experimental confidence level
    (`1 - alpha = 0.85`, i.e. `alpha = 0.15`). The paper's own Section 4/Appendix A only report
    solve-time comparison plots for this exact experiment (checked directly against the PDF and
    the full extracted text -- no exact reproducible numeric CVaR value is published anywhere in
    the paper), so this cannot be checked against a published number. Checked instead against
    `_brute_force_cvar` -- a from-first-principles oracle enumerating every deterministic policy
    and every chance realization directly, independent of `_build_risk_measure`'s own
    eta/lambda/rho machinery -- and via `IndexMap` read-back of `eta`, confirming it lands within
    `utility_values`' own range (not exact membership -- see `RiskMeasureIndexMap.eta_symbol`'s own
    docstring: constraints (11)-(18) only pin `eta` to *some* alpha-quantile of the compiled
    distribution, not necessarily a realizable utility value itself; the code review pass caught
    this exact test asserting more than the formulation actually guarantees).
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
    """A random small BN plus a grafted-on decision/objective/risk measure.

    Checked against `_brute_force_cvar` -- same generation style as Phase 4's
    `test_random_small_diagrams_all_oracles_agree`, extended with a `RiskMeasure`.
    """
    gum.initRandom(seed)
    network = from_pyagrum(gum.randomBN(n=n, domain_size=domain_size, ratio_arc=ratio_arc))
    base = _graft_decision_and_objective(network, np.random.default_rng(seed))
    annotation = base.model_copy(update={"risk_measures": [RiskMeasure(symbol="cvar_obj", target="obj", alpha=alpha)]})

    compiled = _compiled_optimum(network, annotation, "cvar_obj")
    brute_force = _brute_force_cvar(network, annotation, "obj", alpha)
    assert compiled == pytest.approx(brute_force, abs=1e-6)


def test_cvar_approaches_expected_value_and_is_bounded_and_monotone():
    """Two literature relationships from Rockafellar & Uryasev (2000), cited by the paper itself.

    Checked as an oracle independent of this module's own internal consistency, on a no-decision
    diagram (so mu is uniquely pinned -- see the module docstring -- and every objective/risk
    measure compiled against the same `Problem` reads the identical fixed distribution, letting
    the comparisons below isolate the mathematical property from any policy-search confound):

    - `CVaR_alpha -> E[utility]` as `alpha -> 1`: checked at `alpha=0.999` against `f1`, Phase 4's
      own already-oracle-tested `_build_objective` expected-value `Objective` -- a real
      regression-style cross-check between two independently-built code paths
      (`_build_objective`/`_build_risk_measure`) in this same module, both compiled onto the same
      `Problem` and read off the *same* solve (no decisions, so which objective is "solved for"
      doesn't matter -- the feasible mu is unique either way). The tolerance is derived, not
      arbitrary: excluding at most `1 - alpha` probability mass from CVaR_alpha's tail can move it
      by at most `(1 - alpha) * (max(u) - min(u))` away from the full expectation.
    - `CVaR_alpha <= E[utility]` for every `alpha`, and `CVaR_alpha` monotone non-decreasing in
      `alpha` -- both checked directly against `f1`'s own compiled expected value and across a
      sweep of `RiskMeasure`s at different `alpha`, all compiled into the one `Problem`.
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
