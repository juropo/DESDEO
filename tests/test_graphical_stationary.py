"""Tests for `desdeo.problem.graphical.dynamic`.

Oracles:

1. **Value iteration / exact policy evaluation** on the explicit MDP read off `mdp_2tbn_network`.
2. **pyAgrum `LazyPropagation`** on the 2TBN, for the slice-0 joint over the state nodes.
3. **pyAgrum-unrolled marginals** (`dynamicBN.unroll2TBN`): the discounted per-step average
   Σ_{t≥1} (1 - gamma) gamma^{t-1} E[u_t], truncated at `HORIZON` slices.
"""

import itertools

import numpy as np
import pyagrum as gum
import pytest
from fixtures import (
    atemporal_2tbn_network,
    binary_variable,
    ci_healthcare_mirror_network,
    mdp_2tbn_network,
    same_slice_2tbn_network,
)
from pyagrum.lib.dynamicBN import unroll2TBN

from desdeo.problem.graphical import (
    DecisionAnnotation,
    DecisionNode,
    DecisionPolicy,
    DiagramConstraint,
    DiagramConstraintKind,
    DiscreteNetwork,
    ObjectiveNode,
    Potential,
    StationaryDiagram,
    classify_2tbn,
    compile_stationary_diagram,
    evaluate_stationary_policy,
    initial_state_distribution,
    promote_to_decisions,
    read_policy_from_solution,
)
from desdeo.problem.graphical.adapters.pyagrum import to_pyagrum
from desdeo.problem.graphical.compile import _group_flat_indices_by_projection
from desdeo.problem.graphical.dynamic import _contract
from desdeo.problem.graphical.index_map import mixed_radix_encode
from desdeo.tools import add_asf_diff
from desdeo.tools.gurobipy_solver_interfaces import GurobipySolver

pytestmark = pytest.mark.graphical

GAMMA = 0.9
HORIZON = 400
X_UTILITY = [1.0, 3.0]
Y_UTILITY = [2.0, -1.0, 0.5]


def _objective(symbol: str, parents: list[str], utility: list) -> ObjectiveNode:
    values = np.array(utility, dtype=float)
    return ObjectiveNode(
        symbol=symbol,
        parents=parents,
        utility=Potential(variables=parents, shape=list(values.shape), values=values.tolist()),
        maximize=True,
    )


def _policy(decision: DecisionNode, information_set: list[str], choices: list[int]) -> DecisionPolicy:
    return DecisionPolicy(
        symbol=decision.symbol,
        information_set=information_set,
        states=decision.states,
        policy=dict(enumerate(choices)),
    )


def _replace_cpt(network: DiscreteNetwork, node: str, potential: Potential) -> DiscreteNetwork:
    """`network` with `node`'s CPT, and so its parents, replaced by `potential`."""
    arcs = [(parent, child) for parent, child in network.arcs if child != node]
    arcs += [(parent, node) for parent in potential.variables[1:]]
    return DiscreteNetwork(
        name=network.name,
        variables=network.variables,
        arcs=arcs,
        potentials={**network.potentials, node: potential},
    )


def _deterministic_cpt(node: str, n_states: int, information_set: list[str], cards: list[int], choices: list[int]):
    """The CPT of `node` choosing `choices[mixed_radix_encode(info_state, cards)]`."""
    values = np.zeros([n_states, *cards])
    for info_state in itertools.product(*(range(card) for card in cards)):
        values[(choices[mixed_radix_encode(info_state, cards)], *info_state)] = 1.0
    return Potential(variables=[node, *information_set], shape=list(values.shape), values=values.tolist())


# ── Oracle 1: value iteration on the explicit MDP ──────────────────────────────────────────────


def _explicit_mdp(network: DiscreteNetwork, utility: list[float]):
    """(P[s, a, s'], R[s, a], mu[s]) of `mdp_2tbn_network`, with R the expected utility of `Xt`."""
    xt = np.array(network.potentials["Xt"].values)  # [Xt, X0, Dt]
    transition = np.transpose(xt, (1, 2, 0))
    reward = transition @ np.array(utility)
    mu = np.array(network.potentials["X0"].values)
    return transition, reward, mu


def _value_iteration(transition, reward, gamma: float, tol: float = 1e-13, max_iter: int = 100_000):
    """Bellman backup V(s) <- max_a [R(s, a) + gamma Σ_s' P(s'|s, a) V(s')]."""
    value = np.zeros(reward.shape[0])
    q = reward.copy()
    for _ in range(max_iter):
        q = reward + gamma * np.einsum("sat,t->sa", transition, value)
        new_value = q.max(axis=1)
        if np.max(np.abs(new_value - value)) < tol:
            value = new_value
            break
        value = new_value
    return value, q.argmax(axis=1)


def _evaluate_deterministic_policy(transition, reward, policy, gamma: float):
    """V = R_pi + gamma P_pi V, solved exactly."""
    states = np.arange(reward.shape[0])
    return np.linalg.solve(np.eye(len(states)) - gamma * transition[states, policy], reward[states, policy])


def _mdp_diagram() -> StationaryDiagram:
    return StationaryDiagram.with_default_information_sets(
        mdp_2tbn_network(), {"Dt": None}, [_objective("u", ["Xt"], X_UTILITY)], discount=GAMMA
    )


def test_mdp_matches_value_iteration_oracle():
    """Every deterministic policy matches exact policy evaluation, and the best one matches value iteration."""
    diagram = _mdp_diagram()
    decision = diagram.decisions[0]
    transition, reward, mu = _explicit_mdp(diagram.network, X_UTILITY)

    averages = {}
    for choices in itertools.product(range(3), repeat=2):
        evaluation = evaluate_stationary_policy(diagram, {"Dt": _policy(decision, ["X0"], list(choices))})
        value = _evaluate_deterministic_policy(transition, reward, np.array(choices), GAMMA)
        assert evaluation.objectives["u"] == pytest.approx((1 - GAMMA) * mu @ value, abs=1e-9)
        assert np.sum(evaluation.occupation.values) == pytest.approx(1.0, abs=1e-12)
        averages[choices] = evaluation.objectives["u"]

    optimal_value, optimal_policy = _value_iteration(transition, reward, GAMMA)
    best = max(averages, key=averages.get)
    assert averages[best] == pytest.approx((1 - GAMMA) * mu @ optimal_value, abs=1e-9)
    assert list(best) == optimal_policy.tolist()


# ── classify_2tbn ──────────────────────────────────────────────────────────────────────────────


def test_classify_mdp():
    """The MDP's state is X0, carried to Xt."""
    diagram = _mdp_diagram()
    structure = classify_2tbn(diagram.network, diagram.decisions)
    assert structure.slice0 == {"X0", "D0"}
    assert structure.slicet == {"Xt", "Dt"}
    assert structure.atemporal == set()
    assert structure.state_nodes == ["X0"]
    assert structure.next_of == {"X0": "Xt"}


def test_classify_same_slice_observation():
    """A decision observing a same-slice node adds nothing to the state."""
    network = same_slice_2tbn_network()
    diagram = StationaryDiagram.with_default_information_sets(
        network, {"Dt": None}, [_objective("u", ["Xt"], X_UTILITY)], discount=GAMMA
    )
    assert diagram.decisions[0].information_set == ["Yt"]
    assert diagram.decisions[0].states == ["lo", "hi"]
    structure = diagram.structure
    assert structure.state_nodes == ["X0"]
    assert structure.next_of == {"X0": "Xt"}


def test_classify_atemporal():
    """Atemporal chance nodes are carried by identity; an atemporal decision is not state."""
    network = atemporal_2tbn_network()
    decision = DecisionNode(symbol="G", states=["lo", "mid", "hi"])

    structure = classify_2tbn(network, [decision])
    assert structure.atemporal == {"A", "G"}
    assert structure.state_nodes == ["A", "X0"]
    assert structure.next_of == {"A": "A", "X0": "Xt"}

    status_quo = classify_2tbn(network)
    assert status_quo.state_nodes == ["A", "G", "X0"]
    assert status_quo.next_of == {"A": "A", "G": "G", "X0": "Xt"}


def test_classify_ci_healthcare_mirror():
    """With GER and LYHt decided, the state is OMA and the slice-0 PIS, PIH, LYS, LYH; HHA is excluded.

    GER is a parent of AST0, which isn't state, so no initial distribution is needed.
    """
    network = ci_healthcare_mirror_network()
    assert ("GER", "AST0") in network.arcs
    decisions = [
        DecisionNode(symbol="GER", states=["lo", "hi"]),
        DecisionNode(
            symbol="LYHt",
            states=["lo", "mid", "hi"],
            information_set=network.potentials["LYHt"].variables[1:],
        ),
    ]
    structure = classify_2tbn(network, decisions)
    assert structure.atemporal == {"HHA", "GER", "OMA"}
    assert structure.state_nodes == ["OMA", "PIS0", "PIH0", "LYS0", "LYH0"]
    assert structure.next_of == {"OMA": "OMA", "PIS0": "PISt", "PIH0": "PIHt", "LYS0": "LYSt", "LYH0": "LYHt"}

    diagram = StationaryDiagram.with_default_information_sets(
        network, {"GER": None, "LYHt": None}, [_objective("u", ["KUOt"], [0.0, 1.0])], discount=GAMMA
    )
    assert diagram.initial_distribution is None
    assert diagram.structure.state_nodes == structure.state_nodes
    assert initial_state_distribution(diagram).shape == (3, 3, 2, 3, 3)


# ── Oracle 2: initial state distribution against LazyPropagation ─────────────────────────────────


def _lazy_propagation_joint(network: DiscreteNetwork, nodes: list[str]) -> np.ndarray:
    engine = gum.LazyPropagation(to_pyagrum(network))
    engine.addJointTarget(set(nodes))
    engine.makeInference()
    joint = engine.jointPosterior(set(nodes))
    axes = [variable.name() for variable in joint.variablesSequence()]
    return np.transpose(np.transpose(joint.toarray()), [axes.index(node) for node in nodes])


@pytest.mark.parametrize(
    ("network_builder", "decisions"),
    [
        (mdp_2tbn_network, {"Dt": None}),
        (same_slice_2tbn_network, {"Dt": None}),
        (atemporal_2tbn_network, {"G": None}),
        (atemporal_2tbn_network, {}),
        (ci_healthcare_mirror_network, {"GER": None, "LYHt": None}),
    ],
)
def test_initial_state_distribution_matches_lazy_propagation(network_builder, decisions):
    """The slice-0 joint over the state nodes equals pyAgrum's joint posterior."""
    network = network_builder()
    objective = (
        _objective("u", ["KUOt"], [0.0, 1.0]) if "KUOt" in network.potentials else _objective("u", ["Xt"], X_UTILITY)
    )
    diagram = StationaryDiagram.with_default_information_sets(network, decisions, [objective], discount=GAMMA)
    state_nodes = diagram.structure.state_nodes

    mu = initial_state_distribution(diagram)
    assert mu.shape == tuple(len(network.potentials[name].values) for name in state_nodes)
    np.testing.assert_allclose(mu, _lazy_propagation_joint(network, state_nodes), rtol=0, atol=1e-12)


# ── Oracle 3: discounted averages of pyAgrum-unrolled marginals ─────────────────────────────────


def _unrolled_average(network: DiscreteNetwork, node: str, utility: list[float]) -> float:
    """Σ_{t=1}^{HORIZON-1} (1 - gamma) gamma^{t-1} E[utility(node at t)] on the unrolled 2TBN."""
    engine = gum.LazyPropagation(unroll2TBN(to_pyagrum(network), HORIZON))
    engine.makeInference()
    return sum(
        (1 - GAMMA) * GAMMA ** (t - 1) * float(engine.posterior(f"{node[:-1]}{t}").toarray() @ np.array(utility))
        for t in range(1, HORIZON)
    )


@pytest.mark.parametrize(
    ("network_builder", "decisions"),
    [
        (atemporal_2tbn_network, []),
        (atemporal_2tbn_network, ["G"]),
        (same_slice_2tbn_network, []),
        (same_slice_2tbn_network, ["Dt"]),
    ],
)
def test_status_quo_matches_unrolled_network(network_builder, decisions):
    """Status-quo CPTs, with or without the node declared a decision, match the unrolled network."""
    network = network_builder()
    objectives = [_objective("x", ["Xt"], X_UTILITY)]
    if "Yt" in network.potentials:
        objectives.append(_objective("y", ["Yt"], Y_UTILITY))
    diagram = StationaryDiagram.with_default_information_sets(
        network, dict.fromkeys(decisions), objectives, discount=GAMMA
    )

    evaluation = evaluate_stationary_policy(diagram, {})
    assert evaluation.objectives["x"] == pytest.approx(_unrolled_average(network, "Xt", X_UTILITY), abs=1e-6)
    if "y" in evaluation.objectives:
        assert evaluation.objectives["y"] == pytest.approx(_unrolled_average(network, "Yt", Y_UTILITY), abs=1e-6)
    assert evaluation.occupation.variables == diagram.structure.state_nodes


@pytest.mark.parametrize("choices", [[0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 0, 1]])
def test_same_slice_policy_matches_unrolled_network(choices):
    """A policy over `[X0, Yt]` (not the network's parent order) is decoded most-significant-first."""
    network = same_slice_2tbn_network()
    information_set = ["X0", "Yt"]
    diagram = StationaryDiagram.with_default_information_sets(
        network, {"Dt": information_set}, [_objective("x", ["Xt"], X_UTILITY)], discount=GAMMA
    )

    evaluation = evaluate_stationary_policy(diagram, {"Dt": _policy(diagram.decisions[0], information_set, choices)})
    pinned = _replace_cpt(network, "Dt", _deterministic_cpt("Dt", 2, information_set, [2, 3], choices))
    assert evaluation.objectives["x"] == pytest.approx(_unrolled_average(pinned, "Xt", X_UTILITY), abs=1e-6)


@pytest.mark.parametrize("choices", [[0, 1, 1, 0, 1, 0], [1, 1, 0, 0, 0, 1]])
def test_policy_order_differs_from_decision_information_set(choices):
    """A policy over `[X0, Yt]` is read in its own order when the decision observes `[Yt, X0]`."""
    network = same_slice_2tbn_network()
    diagram = StationaryDiagram.with_default_information_sets(
        network, {"Dt": ["Yt", "X0"]}, [_objective("x", ["Xt"], X_UTILITY)], discount=GAMMA
    )
    assert diagram.decisions[0].information_set == ["Yt", "X0"]

    policy_information_set = ["X0", "Yt"]
    policy = _policy(diagram.decisions[0], policy_information_set, choices)
    evaluation = evaluate_stationary_policy(diagram, {"Dt": policy})
    pinned = _replace_cpt(network, "Dt", _deterministic_cpt("Dt", 2, policy_information_set, [2, 3], choices))
    assert evaluation.objectives["x"] == pytest.approx(_unrolled_average(pinned, "Xt", X_UTILITY), abs=1e-6)


@pytest.mark.parametrize("choice", [0, 1, 2])
def test_atemporal_policy_matches_unrolled_network(choice):
    """An atemporal decision is fixed once, as a point-mass CPT in the unrolled network."""
    network = atemporal_2tbn_network()
    diagram = StationaryDiagram.with_default_information_sets(
        network,
        {"G": None},
        [_objective("x", ["Xt"], X_UTILITY), _objective("g", ["G"], [0.0, 1.0, 5.0])],
        discount=GAMMA,
    )

    evaluation = evaluate_stationary_policy(diagram, {"G": _policy(diagram.decisions[0], [], [choice])})
    pinned = _replace_cpt(network, "G", _deterministic_cpt("G", 3, [], [], [choice]))
    assert evaluation.objectives["x"] == pytest.approx(_unrolled_average(pinned, "Xt", X_UTILITY), abs=1e-6)
    assert evaluation.objectives["g"] == pytest.approx([0.0, 1.0, 5.0][choice], abs=1e-12)


def test_explicit_initial_distribution_for_atemporal_decision_ancestor():
    """An explicit initial distribution is used when a decision is an ancestor of the state."""
    network = atemporal_2tbn_network(g_affects_slice0=True)
    choice = 1
    pinned = _replace_cpt(network, "G", _deterministic_cpt("G", 3, [], [], [choice]))
    mu = _lazy_propagation_joint(pinned, ["A", "X0"])
    diagram = StationaryDiagram.with_default_information_sets(
        network,
        {"G": None},
        [_objective("x", ["Xt"], X_UTILITY)],
        discount=GAMMA,
        initial_distribution=Potential(variables=["A", "X0"], shape=[2, 2], values=mu.tolist()),
    )

    np.testing.assert_array_equal(initial_state_distribution(diagram), mu)
    evaluation = evaluate_stationary_policy(diagram, {"G": _policy(diagram.decisions[0], [], [choice])})
    assert evaluation.objectives["x"] == pytest.approx(_unrolled_average(pinned, "Xt", X_UTILITY), abs=1e-6)


def test_state_node_outside_transition_matches_unrolled_network():
    """A state node whose only slice-t child doesn't feed the next state still gets a kernel axis."""
    network = _network({"Y0": [], "Yt": [], "W0": [], "Wt": ["Y0"]})
    network = _replace_cpt(network, "Y0", Potential(variables=["Y0"], shape=[2], values=[0.3, 0.7]))
    network = _replace_cpt(network, "Yt", Potential(variables=["Yt"], shape=[2], values=[0.8, 0.2]))
    wt = Potential(variables=["Wt", "Y0"], shape=[2, 2], values=[[0.9, 0.4], [0.1, 0.6]])
    network = _replace_cpt(network, "Wt", wt)
    diagram = StationaryDiagram(
        network=network, decisions=[], objectives=[_objective("w", ["Wt"], [0.0, 1.0])], discount=GAMMA
    )
    assert diagram.structure.state_nodes == ["Y0"]

    evaluation = evaluate_stationary_policy(diagram, {})
    assert evaluation.objectives["w"] == pytest.approx(_unrolled_average(network, "Wt", [0.0, 1.0]), abs=1e-6)


# ── compile_stationary_diagram ─────────────────────────────────────────────────────────────────


def _all_policies(diagram: StationaryDiagram):
    """Every deterministic policy of `diagram`'s decisions, as `{symbol: DecisionPolicy}`."""
    cardinality = {variable.name: len(variable.labels) for variable in diagram.network.variables}
    per_decision = []
    for decision in diagram.decisions:
        n_info = int(np.prod([cardinality[name] for name in decision.information_set]))
        per_decision.append(
            [
                _policy(decision, decision.information_set, list(choices))
                for choices in itertools.product(range(len(decision.states)), repeat=n_info)
            ]
        )
    for combination in itertools.product(*per_decision):
        yield {policy.symbol: policy for policy in combination}


def _solve(diagram: StationaryDiagram, objective_symbol: str):
    """Solve `diagram`'s MILP for one objective; return the result, read-back policies and their evaluation."""
    problem, stationary_map = compile_stationary_diagram(diagram)
    result = GurobipySolver(problem).solve(objective_symbol)
    assert result.success, result.message
    policies = read_policy_from_solution(stationary_map.index_map, result.optimal_variables)
    evaluation = evaluate_stationary_policy(diagram, policies)
    for symbol, value in evaluation.objectives.items():
        assert result.optimal_objectives[symbol] == pytest.approx(value, abs=1e-6)
    return result, policies, evaluation, stationary_map


def _same_slice_diagram(information_set: list[str] | None) -> StationaryDiagram:
    return StationaryDiagram.with_default_information_sets(
        same_slice_2tbn_network(),
        {"Dt": information_set},
        [_objective("x", ["Xt"], X_UTILITY), _objective("y", ["Yt"], Y_UTILITY)],
        discount=GAMMA,
    )


def _atemporal_diagram() -> StationaryDiagram:
    return StationaryDiagram.with_default_information_sets(
        atemporal_2tbn_network(),
        {"G": None},
        [_objective("x", ["Xt"], X_UTILITY), _objective("g", ["G"], [0.0, -0.3, -0.6])],
        discount=GAMMA,
    )


@pytest.mark.gurobipy
@pytest.mark.parametrize("gamma", [0.5, 0.9, 0.99])
def test_compiled_mdp_matches_value_iteration(gamma):
    """The MILP optimum and policy of the fully observed MDP equal value iteration's."""
    diagram = StationaryDiagram.with_default_information_sets(
        mdp_2tbn_network(), {"Dt": None}, [_objective("u", ["Xt"], X_UTILITY)], discount=gamma
    )
    transition, reward, mu = _explicit_mdp(diagram.network, X_UTILITY)
    optimal_value, optimal_policy = _value_iteration(transition, reward, gamma)

    result, policies, _, _ = _solve(diagram, "u")
    assert result.optimal_objectives["u"] == pytest.approx((1 - gamma) * mu @ optimal_value, abs=1e-6)
    assert policies["Dt"].information_set == ["X0"]
    assert [policies["Dt"].policy[s] for s in range(2)] == optimal_policy.tolist()


@pytest.mark.gurobipy
@pytest.mark.parametrize("information_set", [None, ["X0", "Yt"]])
@pytest.mark.parametrize("objective_symbol", ["x", "y"])
def test_compiled_same_slice_matches_brute_force(information_set, objective_symbol):
    """With a same-slice observation, the MILP optimum is the best enumerated deterministic policy."""
    diagram = _same_slice_diagram(information_set)
    best = max(
        evaluate_stationary_policy(diagram, policies).objectives[objective_symbol]
        for policies in _all_policies(diagram)
    )

    result, _, _, _ = _solve(diagram, objective_symbol)
    assert result.optimal_objectives[objective_symbol] == pytest.approx(best, abs=1e-6)


@pytest.mark.gurobipy
@pytest.mark.parametrize("objective_symbol", ["x", "g"])
def test_compiled_atemporal_decision_matches_brute_force(objective_symbol):
    """An atemporal decision with an atemporal chance state: the MILP picks the brute-force argmax."""
    diagram = _atemporal_diagram()
    values = {
        policies["G"].policy[0]: evaluate_stationary_policy(diagram, policies).objectives[objective_symbol]
        for policies in _all_policies(diagram)
    }
    best = max(values, key=values.get)

    result, policies, _, _ = _solve(diagram, objective_symbol)
    assert result.optimal_objectives[objective_symbol] == pytest.approx(values[best], abs=1e-6)
    assert policies["G"].policy == {0: best}


@pytest.mark.gurobipy
def test_compiled_asf_solution_is_not_dominated():
    """An ASF solution's policy evaluates to the solver's objectives and no deterministic policy dominates it."""
    diagram = _same_slice_diagram(["X0", "Yt"])
    problem, stationary_map = compile_stationary_diagram(diagram)
    enumerated = [evaluate_stationary_policy(diagram, policies).objectives for policies in _all_policies(diagram)]
    ideal = {symbol: max(values[symbol] for values in enumerated) for symbol in ("x", "y")}
    nadir = {symbol: min(values[symbol] for values in enumerated) for symbol in ("x", "y")}
    reference_point = {symbol: (ideal[symbol] + nadir[symbol]) / 2 for symbol in ("x", "y")}

    problem = problem.update_ideal_and_nadir(ideal, nadir)
    scalarized, target = add_asf_diff(problem, "asf", reference_point)
    result = GurobipySolver(scalarized).solve(target)
    assert result.success, result.message

    policies = read_policy_from_solution(stationary_map.index_map, result.optimal_variables)
    evaluation = evaluate_stationary_policy(diagram, policies)
    for symbol in ("x", "y"):
        assert result.optimal_objectives[symbol] == pytest.approx(evaluation.objectives[symbol], abs=1e-6)
    for values in enumerated:
        at_least = all(values[symbol] >= evaluation.objectives[symbol] - 1e-9 for symbol in ("x", "y"))
        better = any(values[symbol] > evaluation.objectives[symbol] + 1e-9 for symbol in ("x", "y"))
        assert not (at_least and better), values


@pytest.mark.gurobipy
@pytest.mark.parametrize("diagram_builder", [lambda: _same_slice_diagram(["X0", "Yt"]), _atemporal_diagram])
def test_compiled_occupation_matches_evaluation(diagram_builder):
    """The solved occupation cluster's marginal over the state nodes is the evaluated occupation measure."""
    diagram = diagram_builder()
    result, _, evaluation, stationary_map = _solve(diagram, "x")
    cluster = stationary_map.index_map.clusters[stationary_map.occupation_owner]
    cardinalities = evaluation.occupation.shape
    groups = _group_flat_indices_by_projection(
        cluster.nodes, cluster.cardinalities, stationary_map.state_nodes, cardinalities
    )
    mu = np.array(result.optimal_variables[cluster.mu_symbol])
    occupation = np.array([mu[group].sum() for group in groups])

    assert occupation.sum() == pytest.approx(1.0, abs=1e-6)
    np.testing.assert_allclose(occupation, np.ravel(evaluation.occupation.values), rtol=0, atol=1e-6)


def test_compile_stationary_sink_name_clash_raises():
    """A symbol named like the internal state sink is refused."""
    diagram = StationaryDiagram.with_default_information_sets(
        mdp_2tbn_network(), {"Dt": None}, [_objective("__state_t", ["Xt"], X_UTILITY)], discount=GAMMA
    )
    with pytest.raises(ValueError, match="reserved"):
        compile_stationary_diagram(diagram)


def test_compile_stationary_flow_name_clash_raises():
    """A constraint symbol named like a flow-balance constraint is refused."""
    constraint = DiagramConstraint(
        symbol="flow_1", kind=DiagramConstraintKind.probability, nodes=["Xt"], func=["Subtract", ["At", "Xt", 1], 0.9]
    )
    diagram = StationaryDiagram.with_default_information_sets(
        mdp_2tbn_network(),
        {"Dt": None},
        [_objective("u", ["Xt"], X_UTILITY)],
        discount=GAMMA,
        constraints=[constraint],
    )
    with pytest.raises(ValueError, match="reserved for the flow-balance constraints"):
        compile_stationary_diagram(diagram)


def _sink_apart_network() -> DiscreteNetwork:
    """`Wt | Y0, Dt` and `Dt | Y0`, with `Yt` a root, so the state sink is apart from the free chain."""
    network = _network({"Y0": [], "Yt": [], "W0": [], "Wt": ["Y0", "Dt"], "D0": [], "Dt": ["Y0"]})
    network = _replace_cpt(network, "Yt", Potential(variables=["Yt"], shape=[2], values=[0.8, 0.2]))
    win = np.array([[0.2, 0.7], [0.9, 0.4]])
    return _replace_cpt(
        network,
        "Wt",
        Potential(variables=["Wt", "Y0", "Dt"], shape=[2, 2, 2], values=np.stack([1 - win, win]).tolist()),
    )


def _barren_node_network() -> DiscreteNetwork:
    """The MDP with an isolated slice pair `Q0`, `Qt`."""
    network = mdp_2tbn_network()
    return DiscreteNetwork(
        variables=[*network.variables, binary_variable("Q0"), binary_variable("Qt")],
        arcs=network.arcs,
        potentials={
            **network.potentials,
            "Q0": Potential(variables=["Q0"], shape=[2], values=[0.5, 0.5]),
            "Qt": Potential(variables=["Qt"], shape=[2], values=[0.4, 0.6]),
        },
    )


@pytest.mark.gurobipy
@pytest.mark.parametrize(
    ("network_builder", "objective"),
    [(_sink_apart_network, ["Wt"]), (_barren_node_network, ["Xt"])],
)
def test_compiled_disconnected_step_matches_brute_force(network_builder, objective):
    """A step network that would split into several components compiles to the brute-force optimum."""
    diagram = StationaryDiagram.with_default_information_sets(
        network_builder(), {"Dt": None}, [_objective("u", objective, [1.0, 3.0])], discount=GAMMA
    )
    best = max(evaluate_stationary_policy(diagram, policies).objectives["u"] for policies in _all_policies(diagram))

    result, _, _, stationary_map = _solve(diagram, "u")
    assert result.optimal_objectives["u"] == pytest.approx(best, abs=1e-6)
    assert "Qt" not in stationary_map.index_map.clusters


@pytest.mark.gurobipy
@pytest.mark.parametrize("objective_symbol", ["e", "x"])
def test_compiled_decision_sink_parent_matches_brute_force(objective_symbol):
    """A lone decision `Et` joins the state sink as an extra parent and compiles to the brute-force optimum."""
    network = _network({"Et": [], "E0": [], "X0": [], "Xt": ["X0"]})
    network = _replace_cpt(
        network, "Xt", Potential(variables=["Xt", "X0"], shape=[2, 2], values=[[0.6, 0.2], [0.4, 0.8]])
    )
    diagram = StationaryDiagram.with_default_information_sets(
        network,
        {"Et": None},
        [_objective("e", ["Et"], [0.0, 2.0]), _objective("x", ["Xt"], [1.0, 3.0])],
        discount=GAMMA,
    )
    best = max(
        evaluate_stationary_policy(diagram, policies).objectives[objective_symbol]
        for policies in _all_policies(diagram)
    )

    result, _, _, _ = _solve(diagram, objective_symbol)
    assert result.optimal_objectives[objective_symbol] == pytest.approx(best, abs=1e-6)


# ── promote_to_decisions ───────────────────────────────────────────────────────────────────────


def test_promote_to_decisions_keeps_decision_axes():
    """Promoted decisions leave the network, but children keep their axes for `DecisionAnnotation`."""
    diagram = _mdp_diagram()
    promoted = promote_to_decisions(diagram.network, diagram.decisions)

    assert [variable.name for variable in promoted.variables] == ["X0", "Xt", "D0"]
    assert all("Dt" not in arc for arc in promoted.arcs)
    assert "Dt" not in promoted.potentials
    assert promoted.potentials["Xt"].variables == ["Xt", "X0", "Dt"]
    DecisionAnnotation(decisions=diagram.decisions, objectives=diagram.objectives).validate_against(promoted)


# ── Errors ─────────────────────────────────────────────────────────────────────────────────────


def _network(parents: dict[str, list[str]]) -> DiscreteNetwork:
    """Binary nodes with uniform CPTs and the given parents."""
    return DiscreteNetwork(
        variables=[binary_variable(name) for name in parents],
        arcs=[(parent, child) for child, child_parents in parents.items() for parent in child_parents],
        potentials={
            name: Potential(
                variables=[name, *node_parents],
                shape=[2] * (1 + len(node_parents)),
                values=np.full([2] * (1 + len(node_parents)), 0.5).tolist(),
            )
            for name, node_parents in parents.items()
        },
    )


def test_slice_t_to_slice_0_arc_raises():
    """An arc from slice t back to slice 0 is refused."""
    network = _network({"X0": [], "Xt": ["X0"], "Y0": ["Xt"], "Yt": ["X0"]})
    with pytest.raises(ValueError, match="from slice t back to slice 0"):
        classify_2tbn(network)


def test_slice_t_node_without_slice_0_copy_raises():
    """A name ending in `t` needs its `0` copy."""
    with pytest.raises(ValueError, match="slice-0 copy 'Y0' is missing"):
        classify_2tbn(_network({"X0": [], "Xt": ["X0"], "Yt": ["Xt"]}))


def test_atemporal_node_with_time_slice_parent_raises():
    """An atemporal node can't depend on a time slice."""
    with pytest.raises(ValueError, match="Atemporal node 'A' has parent 'X0'"):
        classify_2tbn(_network({"X0": [], "Xt": ["X0"], "A": ["X0"]}))


def test_decision_not_in_network_raises():
    """Decisions must override network nodes."""
    with pytest.raises(ValueError, match="must name network nodes"):
        StationaryDiagram(
            network=mdp_2tbn_network(),
            decisions=[DecisionNode(symbol="Q", states=["a", "b"])],
            objectives=[_objective("u", ["Xt"], X_UTILITY)],
            discount=GAMMA,
        )


def test_objective_on_slice_0_raises():
    """Objectives can't reference slice 0."""
    with pytest.raises(ValueError, match="references slice-0 node"):
        StationaryDiagram.with_default_information_sets(
            mdp_2tbn_network(), {"Dt": None}, [_objective("u", ["X0"], X_UTILITY)], discount=GAMMA
        )


@pytest.mark.parametrize("discount", [0.0, 1.0])
def test_discount_out_of_range_raises(discount):
    """The discount must be strictly between 0 and 1."""
    with pytest.raises(ValueError, match="discount must be in"):
        StationaryDiagram.with_default_information_sets(
            mdp_2tbn_network(), {"Dt": None}, [_objective("u", ["Xt"], X_UTILITY)], discount=discount
        )


@pytest.mark.parametrize(
    ("values", "reason"),
    [([0.6, 0.7], "got sum"), ([1.2, -0.2], "got negative"), ([float("nan"), 1.0], "got non-finite")],
)
def test_unnormalised_initial_distribution_raises(values, reason):
    """The initial distribution must be a probability distribution."""
    with pytest.raises(ValueError, match=f"finite, nonnegative and sum to 1, {reason}"):
        StationaryDiagram.with_default_information_sets(
            mdp_2tbn_network(),
            {"Dt": None},
            [_objective("u", ["Xt"], X_UTILITY)],
            discount=GAMMA,
            initial_distribution=Potential(variables=["X0"], shape=[2], values=values),
        )


@pytest.mark.parametrize("node", ["Y0", "D0"])
def test_constraint_on_dropped_node_raises(node):
    """Constraints can't reference slice-0 nodes that aren't state."""
    constraint = DiagramConstraint(
        symbol="c", kind=DiagramConstraintKind.probability, nodes=[node], func=["Subtract", ["At", node, 1], 0.5]
    )
    with pytest.raises(ValueError, match="must be slice-t nodes, decisions, state nodes"):
        StationaryDiagram.with_default_information_sets(
            same_slice_2tbn_network(),
            {"Dt": None},
            [_objective("x", ["Xt"], X_UTILITY)],
            discount=GAMMA,
            constraints=[constraint],
        )


def test_atemporal_decision_ancestor_of_state_without_initial_distribution_raises():
    """A decision upstream of the state needs an explicit initial distribution."""
    with pytest.raises(ValueError, match="give initial_distribution"):
        StationaryDiagram.with_default_information_sets(
            atemporal_2tbn_network(g_affects_slice0=True),
            {"G": None},
            [_objective("x", ["Xt"], X_UTILITY)],
            discount=GAMMA,
        )


def test_policy_with_wrong_information_set_raises():
    """A policy must observe the decision's information set."""
    diagram = _mdp_diagram()
    with pytest.raises(ValueError, match="information set"):
        evaluate_stationary_policy(diagram, {"Dt": _policy(diagram.decisions[0], [], [0])})


def test_policy_with_wrong_state_labels_raises():
    """A policy's states must be the decision's labels, not just as many."""
    diagram = _mdp_diagram()
    policy = DecisionPolicy(symbol="Dt", information_set=["X0"], states=["a", "b", "c"], policy={0: 0, 1: 1})
    with pytest.raises(ValueError, match="has states"):
        evaluate_stationary_policy(diagram, {"Dt": policy})


def test_policy_with_out_of_range_key_raises():
    """A policy's keys must be flat indices of the information states."""
    diagram = _mdp_diagram()
    policy = _policy(diagram.decisions[0], ["X0"], [0, 1, 2])
    with pytest.raises(ValueError, match=r"keys \[2\] outside the 2 information states"):
        evaluate_stationary_policy(diagram, {"Dt": policy})


def test_policy_with_out_of_range_choice_raises():
    """A policy's choices must index the decision's states."""
    diagram = _mdp_diagram()
    policy = _policy(diagram.decisions[0], ["X0"], [0, 3])
    with pytest.raises(ValueError, match="chooses state 3, out of range"):
        evaluate_stationary_policy(diagram, {"Dt": policy})


def test_contraction_beyond_einsum_axis_limit_raises():
    """A contraction with more distinct axes than einsum supports gets a clear error."""
    factors = [([f"V{i}"], np.ones(2)) for i in range(53)]
    with pytest.raises(ValueError, match="at most 52"):
        _contract(factors, [])
