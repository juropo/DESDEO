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
"""

import itertools

import numpy as np
import pyagrum as gum
import pytest

from desdeo.problem.graphical import (
    DecisionAnnotation,
    DecisionNode,
    DiscreteNetwork,
    DiscreteVariable,
    ObjectiveNode,
    Potential,
    build_rooted_junction_tree,
    compile_diagram,
)
from desdeo.problem.graphical.adapters.pyagrum import _fill_cpt, _variable_to_pyagrum, from_pyagrum, to_pyagrum
from desdeo.problem.graphical.index_map import mixed_radix_decode
from desdeo.problem.graphical.network import kahn_topological_order
from desdeo.problem.graphical.rooted_junction_tree import effective_information_sets
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


# ── Oracle 2: pyAgrum's own LIMID inference ─────────────────────────────────────────────────────


def _to_pyagrum_influence_diagram(network: DiscreteNetwork, annotation: DecisionAnnotation) -> gum.InfluenceDiagram:
    """Build the same diagram directly in pyAgrum, independent of this package's own compiler."""
    diagram = gum.InfluenceDiagram()

    for variable in network.variables:
        diagram.addChanceNode(_variable_to_pyagrum(variable))
    for node_name, potential in network.potentials.items():
        for parent in potential.variables[1:]:
            diagram.addArc(parent, node_name)
    for node_name, potential in network.potentials.items():
        _fill_cpt(diagram.cpt(node_name), potential)

    for decision in annotation.decisions:
        diagram.addDecisionNode(gum.LabelizedVariable(decision.symbol, decision.symbol, decision.states))
    for decision in annotation.decisions:
        for parent in decision.information_set:
            diagram.addArc(parent, decision.symbol)

    for objective in annotation.objectives:
        util_name = f"util_{objective.symbol}"
        diagram.addUtilityNode(gum.LabelizedVariable(util_name, util_name, 1))
        for parent in objective.parents:
            diagram.addArc(parent, util_name)
        fake_potential = Potential(
            variables=[util_name, *objective.parents],
            shape=[1, *objective.utility.shape],
            values=np.expand_dims(np.array(objective.utility.values, dtype=float), axis=0).tolist(),
        )
        _fill_cpt(diagram.utility(util_name), fake_potential)

    return diagram


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

    diagram = _to_pyagrum_influence_diagram(network, annotation)
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
    return inference.MEU()["mean"]


# ── Oracle-checked compiled MILP value ──────────────────────────────────────────────────────────


def _compiled_optimum(network: DiscreteNetwork, annotation: DecisionAnnotation, objective_symbol: str) -> float:
    problem, _ = compile_diagram(network, annotation)
    result = GurobipySolver(problem).solve(objective_symbol)
    assert result.success, result.message
    return result.optimal_objectives[objective_symbol]


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
