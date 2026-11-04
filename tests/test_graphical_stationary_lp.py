"""Tests for `desdeo.problem.graphical.stationary_lp`.

Oracles:

1. **`evaluate_stationary_policy`**, exact policy evaluation on the 2TBN, against a linear solve on
   the flattened kernel.
2. **Brute-force enumeration** of deterministic policies, against `stationary_value_iteration`.
3. **pyAgrum-unrolled marginals** (`_unrolled_average`), against randomised policy evaluation.
"""

import itertools
from collections.abc import Callable, Iterator
from math import prod

import numpy as np
import pytest
import scipy.sparse as sp
from fixtures import atemporal_2tbn_network, ci_healthcare_mirror_network, mdp_2tbn_network, same_slice_2tbn_network
from test_graphical_stationary import GAMMA, _replace_cpt, _unrolled_average

from desdeo.problem.graphical import (
    DecisionNode,
    DecisionPolicy,
    DiagramConstraint,
    DiagramConstraintKind,
    FlattenedStep,
    ObjectiveNode,
    Potential,
    RandomisedPolicy,
    StationaryDiagram,
    evaluate_stationary_policy,
    flatten_stationary_step,
    full_observation_information_set,
    stationary_value_iteration,
)
from desdeo.problem.graphical.index_map import mixed_radix_encode

pytestmark = pytest.mark.graphical


def _nested_shape(values: list) -> list[int]:
    """Shape of a nested list."""
    if isinstance(values[0], list):
        return [len(values), *_nested_shape(values[0])]
    return [len(values)]


def _objective(symbol: str, parents: list[str], utility: list, *, maximize: bool) -> ObjectiveNode:
    """Build an ObjectiveNode from a nested-list utility."""
    return ObjectiveNode(
        symbol=symbol,
        parents=parents,
        utility=Potential(variables=parents, shape=_nested_shape(utility), values=utility),
        maximize=maximize,
    )


def _mdp_diagram() -> StationaryDiagram:
    """Mdp diagram with full-observation Dt."""
    network = mdp_2tbn_network()
    objectives = [
        _objective("x", ["Xt"], [1.0, 3.0], maximize=True),
        _objective("c", ["Dt"], [0.0, 0.4, 1.0], maximize=False),
    ]
    provisional = StationaryDiagram.with_default_information_sets(network, {"Dt": None}, objectives, discount=GAMMA)
    full_set = full_observation_information_set(provisional, "Dt")
    return StationaryDiagram.with_default_information_sets(network, {"Dt": full_set}, objectives, discount=GAMMA)


def _same_slice_diagram(*, impossible_observation: bool = False) -> StationaryDiagram:
    """Same-slice diagram with full-observation Dt; `Yt = 2` can't follow `X0 = 0` if `impossible_observation`."""
    network = same_slice_2tbn_network()
    if impossible_observation:
        yt = Potential(variables=["Yt", "X0"], shape=[3, 2], values=[[0.30, 0.40], [0.70, 0.35], [0.0, 0.25]])
        network = _replace_cpt(network, "Yt", yt)
    objectives = [
        _objective("x", ["Xt"], [1.0, 3.0], maximize=True),
        _objective("y", ["Yt", "Dt"], [[2.0, 0.0], [-1.0, 0.5], [0.5, 1.5]], maximize=False),
    ]
    provisional = StationaryDiagram.with_default_information_sets(network, {"Dt": None}, objectives, discount=GAMMA)
    full_set = full_observation_information_set(provisional, "Dt")
    return StationaryDiagram.with_default_information_sets(network, {"Dt": full_set}, objectives, discount=GAMMA)


def _mirror_diagram() -> StationaryDiagram:
    """Mirror diagram with atemporal GER and full-observation LYHt."""
    network = ci_healthcare_mirror_network()
    objectives = [
        _objective("k", ["KUOt"], [0.0, 1.0], maximize=False),
        _objective("a", ["ASTt", "LYHt"], [[0.0, 0.5, 1.0], [0.3, 0.2, 0.9]], maximize=True),
        _objective(
            "m",
            ["OMA", "PIHt", "GER"],
            [[[0.0, 0.1], [0.2, 0.3]], [[0.4, 0.5], [0.6, 0.7]], [[0.8, 0.9], [1.0, 0.0]]],
            maximize=False,
        ),
    ]
    provisional = StationaryDiagram.with_default_information_sets(
        network, {"GER": [], "LYHt": None}, objectives, discount=GAMMA
    )
    full_set = full_observation_information_set(provisional, "LYHt")
    return StationaryDiagram.with_default_information_sets(
        network, {"GER": [], "LYHt": full_set}, objectives, discount=GAMMA
    )


def _flat_value(flat: FlattenedStep, branch_index: int, probabilities: np.ndarray) -> dict[str, float]:
    """Oracle: linear solve on the flattened kernel for one branch."""
    branch = flat.branches[branch_index]
    n_observations = prod(flat.observation_cardinalities)
    n_actions = len(flat.decision.states)
    n_states = prod(flat.state_cardinalities)

    pairs = branch.pairs
    s = pairs // n_observations
    o = pairs % n_observations

    s_indices = np.unravel_index(s, flat.state_cardinalities)
    o_indices = np.unravel_index(o, flat.observation_cardinalities) if flat.observation_nodes else ()

    values: dict[str, np.ndarray] = {}
    for name, idx in zip(flat.state_nodes, s_indices, strict=True):
        values[name] = idx
    for name, idx in zip(flat.observation_nodes, o_indices, strict=True):
        values[name] = idx
    for symbol, val in branch.atemporal.items():
        values[symbol] = np.full(len(pairs), val, dtype=np.int64)

    cards: dict[str, int] = {}
    for name in flat.state_nodes:
        cards[name] = flat.state_cardinalities[flat.state_nodes.index(name)]
    for name in flat.observation_nodes:
        cards[name] = flat.observation_cardinalities[flat.observation_nodes.index(name)]
    for atemporal in flat.atemporal_decisions:
        cards[atemporal.symbol] = len(atemporal.states)

    info_set = flat.decision.information_set
    info_cards = [cards[name] for name in info_set]
    keys = np.array(
        [mixed_radix_encode(tuple(int(values[name][i]) for name in info_set), info_cards) for i in range(len(pairs))],
        dtype=np.int64,
    )

    pi = probabilities[keys]  # (n_pairs, n_actions)

    kernel_dense = branch.kernel.toarray()  # (n_states, n_pairs * n_actions)
    transition = np.zeros((n_states, n_states))
    for k in range(len(pairs)):
        s_k = int(s[k])
        transition[:, s_k] += kernel_dense[:, k * n_actions : (k + 1) * n_actions] @ pi[k]

    gamma = flat.discount
    x = np.linalg.solve(np.eye(n_states) - gamma * transition, (1.0 - gamma) * flat.mu0)

    w = np.zeros(len(pairs) * n_actions)
    for k in range(len(pairs)):
        w[k * n_actions : (k + 1) * n_actions] = x[int(s[k])] * pi[k]

    return {sym: float(coeffs @ w) for sym, coeffs in branch.objective_coefficients.items()}


def _deterministic_policies(decision: DecisionNode, n_info: int) -> Iterator[tuple[tuple[int, ...], DecisionPolicy]]:
    """Yield every deterministic policy over `decision`'s information set."""
    for choices in itertools.product(range(len(decision.states)), repeat=n_info):
        policy = DecisionPolicy(
            symbol=decision.symbol,
            information_set=decision.information_set,
            states=decision.states,
            policy={i: choices[i] for i in range(n_info)},
        )
        yield choices, policy


# ── Tests ──────────────────────────────────────────────────────────────────────────────────────


def test_full_observation_information_set() -> None:
    """The three fixtures' full observation sets match the expected lists."""
    mdp_net = mdp_2tbn_network()
    obj = [_objective("x", ["Xt"], [1.0, 3.0], maximize=True)]
    prov = StationaryDiagram.with_default_information_sets(mdp_net, {"Dt": None}, obj, discount=GAMMA)
    assert full_observation_information_set(prov, "Dt") == ["X0"]

    ss_net = same_slice_2tbn_network()
    prov2 = StationaryDiagram.with_default_information_sets(ss_net, {"Dt": None}, obj, discount=GAMMA)
    assert full_observation_information_set(prov2, "Dt") == ["X0", "Yt"]

    mir_net = ci_healthcare_mirror_network()
    mir_obj = [_objective("k", ["KUOt"], [0.0, 1.0], maximize=False)]
    prov3 = StationaryDiagram.with_default_information_sets(mir_net, {"GER": [], "LYHt": None}, mir_obj, discount=GAMMA)
    expected = ["GER", "OMA", "PIS0", "PIH0", "LYS0", "LYH0", "PISt", "PIHt", "LYSt"]
    assert full_observation_information_set(prov3, "LYHt") == expected


@pytest.mark.parametrize(
    "builder", [_mdp_diagram, _same_slice_diagram, lambda: _same_slice_diagram(impossible_observation=True)]
)
def test_kernel_matches_evaluation_small(builder: Callable[[], StationaryDiagram]) -> None:
    """_flat_value matches evaluate_stationary_policy for every deterministic policy."""
    diagram = builder()
    flat = flatten_stationary_step(diagram)
    decision = flat.decision
    n_info = prod(flat.state_cardinalities) * prod(flat.observation_cardinalities)
    n_actions = len(decision.states)

    for choices, policy in _deterministic_policies(decision, n_info):
        one_hot = np.zeros((n_info, n_actions))
        for i, c in enumerate(choices):
            one_hot[i, c] = 1.0
        expected = _flat_value(flat, 0, one_hot)
        actual = evaluate_stationary_policy(diagram, {decision.symbol: policy}).objectives
        for sym in expected:
            assert actual[sym] == pytest.approx(expected[sym], abs=1e-10)


def test_impossible_observation_pairs_are_dropped() -> None:
    """The pair (X0 = 0, Yt = 2) has probability 0, so it and its columns are dropped."""
    flat = flatten_stationary_step(_same_slice_diagram(impossible_observation=True))
    branch = flat.branches[0]
    assert branch.pairs.tolist() == [0, 1, 3, 4, 5]
    assert branch.kernel.shape == (2, 10)
    assert all(len(coefficients) == 10 for coefficients in branch.objective_coefficients.values())


def test_kernel_matches_evaluation_mirror() -> None:
    """_flat_value matches evaluate_stationary_policy on the mirror with randomised policies."""
    diagram = _mirror_diagram()
    flat = flatten_stationary_step(diagram)
    rng = np.random.default_rng(0)

    n_info = (
        prod(flat.state_cardinalities)
        * prod(flat.observation_cardinalities)
        * prod(len(a.states) for a in flat.atemporal_decisions)
    )
    n_actions = len(flat.decision.states)
    ger_states = next(a.states for a in flat.atemporal_decisions if a.symbol == "GER")

    for _ in range(50):
        g = int(rng.integers(0, 2))
        probs = rng.dirichlet(np.ones(n_actions), size=n_info)
        lyht_policy = RandomisedPolicy(
            symbol="LYHt",
            information_set=flat.decision.information_set,
            states=flat.decision.states,
            probabilities={i: probs[i].tolist() for i in range(n_info)},
        )
        ger_policy = DecisionPolicy(symbol="GER", information_set=[], states=ger_states, policy={0: g})

        assert flat.branches[g].atemporal == {"GER": g}
        expected = _flat_value(flat, g, probs)
        actual = evaluate_stationary_policy(diagram, {"LYHt": lyht_policy, "GER": ger_policy}).objectives
        for sym in expected:
            assert actual[sym] == pytest.approx(expected[sym], abs=1e-10)


@pytest.mark.parametrize(
    ("builder", "symbol"),
    [
        (_mdp_diagram, "x"),
        (_mdp_diagram, "c"),
        (_same_slice_diagram, "x"),
        (_same_slice_diagram, "y"),
        (lambda: _same_slice_diagram(impossible_observation=True), "y"),
    ],
)
def test_value_iteration_matches_brute_force(builder: Callable[[], StationaryDiagram], symbol: str) -> None:
    """stationary_value_iteration matches brute-force enumeration of deterministic policies."""
    diagram = builder()
    flat = flatten_stationary_step(diagram)
    decision = flat.decision
    n_info = prod(flat.state_cardinalities) * prod(flat.observation_cardinalities)
    maximize = flat.maximize[symbol]

    best = -np.inf if maximize else np.inf
    for _, policy in _deterministic_policies(decision, n_info):
        val = evaluate_stationary_policy(diagram, {decision.symbol: policy}).objectives[symbol]
        best = max(best, val) if maximize else min(best, val)

    vi_value, vi_policies = stationary_value_iteration(flat, symbol)
    assert vi_value == pytest.approx(best, abs=1e-10)
    eval_val = evaluate_stationary_policy(diagram, vi_policies).objectives[symbol]
    assert eval_val == pytest.approx(vi_value, abs=1e-10)


@pytest.mark.parametrize("discount", [0.99, 0.999])
@pytest.mark.parametrize("builder", [_mdp_diagram, _same_slice_diagram])
def test_value_iteration_exact_at_high_discount(builder: Callable[[], StationaryDiagram], discount: float) -> None:
    """At discounts near 1, VI matches brute force and the exact evaluation of its own policy within 1e-10."""
    diagram = StationaryDiagram(**{**dict(builder()), "discount": discount})
    flat = flatten_stationary_step(diagram)
    decision = flat.decision
    n_info = prod(flat.state_cardinalities) * prod(flat.observation_cardinalities)

    for symbol, maximize in flat.maximize.items():
        values = [
            evaluate_stationary_policy(diagram, {decision.symbol: policy}).objectives[symbol]
            for _, policy in _deterministic_policies(decision, n_info)
        ]
        vi_value, vi_policies = stationary_value_iteration(flat, symbol)
        assert vi_value == pytest.approx(max(values) if maximize else min(values), abs=1e-10)
        assert evaluate_stationary_policy(diagram, vi_policies).objectives[symbol] == pytest.approx(vi_value, abs=1e-10)


def test_value_iteration_atemporal_matches_brute_force() -> None:
    """With atemporal G, VI matches the best over every G value and deterministic Xt policy."""
    network = atemporal_2tbn_network()
    objectives = [
        _objective("x", ["A", "Xt"], [[1.0, 0.0], [0.2, 0.9]], maximize=True),
        _objective(
            "g",
            ["G", "A", "Xt"],
            [[[1.0, 0.2], [0.9, 0.6]], [[0.4, 0.8], [0.1, 0.7]], [[0.3, 0.5], [0.6, 0.0]]],
            maximize=False,
        ),
    ]
    provisional = StationaryDiagram.with_default_information_sets(
        network, {"G": [], "Xt": None}, objectives, discount=GAMMA
    )
    full_set = full_observation_information_set(provisional, "Xt")
    diagram = StationaryDiagram.with_default_information_sets(
        network, {"G": [], "Xt": full_set}, objectives, discount=GAMMA
    )
    flat = flatten_stationary_step(diagram)
    decision = flat.decision
    g_decision = flat.atemporal_decisions[0]
    cards = {"A": 2, "G": 3, "X0": 2}
    info_cards = [cards[name] for name in decision.information_set]
    n_info = prod(info_cards)
    g_axis = decision.information_set.index("G")

    for symbol, maximize in flat.maximize.items():
        values = []
        for g in range(len(g_decision.states)):
            # Only the information states with this G value are reached.
            keys = [k for k in range(n_info) if np.unravel_index(k, info_cards)[g_axis] == g]
            for choices in itertools.product(range(len(decision.states)), repeat=len(keys)):
                policy = DecisionPolicy(
                    symbol=decision.symbol,
                    information_set=decision.information_set,
                    states=decision.states,
                    policy=dict.fromkeys(range(n_info), 0) | dict(zip(keys, choices, strict=True)),
                )
                g_policy = DecisionPolicy(symbol="G", information_set=[], states=g_decision.states, policy={0: g})
                values.append(evaluate_stationary_policy(diagram, {"Xt": policy, "G": g_policy}).objectives[symbol])
        vi_value, vi_policies = stationary_value_iteration(flat, symbol)
        assert vi_value == pytest.approx(max(values) if maximize else min(values), abs=1e-10)
        assert evaluate_stationary_policy(diagram, vi_policies).objectives[symbol] == pytest.approx(vi_value, abs=1e-10)


def test_value_iteration_on_mirror_beats_random_policies() -> None:
    """VI value on the mirror is at least as good as 20 random randomised policies per objective."""
    diagram = _mirror_diagram()
    flat = flatten_stationary_step(diagram)
    rng = np.random.default_rng(42)

    n_info = (
        prod(flat.state_cardinalities)
        * prod(flat.observation_cardinalities)
        * prod(len(a.states) for a in flat.atemporal_decisions)
    )
    n_actions = len(flat.decision.states)
    ger_states = next(a.states for a in flat.atemporal_decisions if a.symbol == "GER")

    for symbol in flat.maximize:
        vi_value, vi_policies = stationary_value_iteration(flat, symbol)
        eval_val = evaluate_stationary_policy(diagram, vi_policies).objectives[symbol]
        assert eval_val == pytest.approx(vi_value, abs=1e-10)

        for _ in range(20):
            g = int(rng.integers(0, 2))
            probs = rng.dirichlet(np.ones(n_actions), size=n_info)
            lyht_policy = RandomisedPolicy(
                symbol="LYHt",
                information_set=flat.decision.information_set,
                states=flat.decision.states,
                probabilities={i: probs[i].tolist() for i in range(n_info)},
            )
            ger_policy = DecisionPolicy(symbol="GER", information_set=[], states=ger_states, policy={0: g})
            rand_val = evaluate_stationary_policy(diagram, {"LYHt": lyht_policy, "GER": ger_policy}).objectives[symbol]
            if flat.maximize[symbol]:
                assert vi_value >= rand_val - 1e-12
            else:
                assert vi_value <= rand_val + 1e-12


def test_randomised_policy_matches_unrolled_network() -> None:
    """RandomisedPolicy on same_slice matches the CPT-pinned unrolled 2TBN average."""
    network = same_slice_2tbn_network()
    objectives = [_objective("x", ["Xt"], [1.0, 3.0], maximize=True)]
    provisional = StationaryDiagram.with_default_information_sets(network, {"Dt": None}, objectives, discount=GAMMA)
    full_set = full_observation_information_set(provisional, "Dt")
    diagram = StationaryDiagram.with_default_information_sets(network, {"Dt": full_set}, objectives, discount=GAMMA)

    rng = np.random.default_rng(3)
    n_info = 2 * 3  # X0 (2) x Yt (3)
    n_actions = 2  # Dt is binary
    probs = rng.dirichlet(np.ones(n_actions), size=n_info)

    policy = RandomisedPolicy(
        symbol="Dt",
        information_set=full_set,
        states=diagram.decisions[0].states,
        probabilities={i: probs[i].tolist() for i in range(n_info)},
    )
    evaluation = evaluate_stationary_policy(diagram, {"Dt": policy})

    pinned = _replace_cpt(
        network,
        "Dt",
        Potential(
            variables=["Dt", "X0", "Yt"],
            shape=[2, 2, 3],
            values=probs.T.reshape(2, 2, 3).tolist(),
        ),
    )
    expected = _unrolled_average(pinned, "Xt", [1.0, 3.0])
    assert evaluation.objectives["x"] == pytest.approx(expected, abs=1e-6)


def test_flatten_zero_slicet_decisions() -> None:
    """`flatten_stationary_step` raises when there is no slice-t decision."""
    network = atemporal_2tbn_network()
    objectives = [_objective("x", ["Xt"], [1.0, 3.0], maximize=True)]
    diagram = StationaryDiagram.with_default_information_sets(network, {"G": None}, objectives, discount=GAMMA)
    with pytest.raises(ValueError, match="exactly one slice-t decision"):
        flatten_stationary_step(diagram)


def test_flatten_two_slicet_decisions() -> None:
    """`flatten_stationary_step` raises when there are two slice-t decisions."""
    network = ci_healthcare_mirror_network()
    objectives = [_objective("k", ["KUOt"], [0.0, 1.0], maximize=False)]
    diagram = StationaryDiagram.with_default_information_sets(
        network, {"GER": [], "LYHt": None, "ASTt": None}, objectives, discount=GAMMA
    )
    with pytest.raises(ValueError, match="exactly one slice-t decision"):
        flatten_stationary_step(diagram)


def test_flatten_partial_information_set() -> None:
    """`flatten_stationary_step` raises when the decision's info set is not the full observation set."""
    network = same_slice_2tbn_network()
    objectives = [_objective("x", ["Xt"], [1.0, 3.0], maximize=True)]
    diagram = StationaryDiagram.with_default_information_sets(network, {"Dt": ["Yt"]}, objectives, discount=GAMMA)
    with pytest.raises(ValueError, match="full observation set"):
        flatten_stationary_step(diagram)


def test_flatten_atemporal_observations() -> None:
    """`flatten_stationary_step` raises when an atemporal decision observes something."""
    network = ci_healthcare_mirror_network()
    objectives = [
        _objective("k", ["KUOt"], [0.0, 1.0], maximize=False),
    ]
    full_set = full_observation_information_set(_mirror_diagram(), "LYHt")
    diagram = StationaryDiagram.with_default_information_sets(
        network, {"GER": ["OMA"], "LYHt": full_set}, objectives, discount=GAMMA
    )
    with pytest.raises(ValueError, match="Atemporal decision 'GER' observes"):
        flatten_stationary_step(diagram)


def test_flatten_with_constraints() -> None:
    """`flatten_stationary_step` raises when the diagram has constraints."""
    constraint = DiagramConstraint(
        symbol="cap", kind=DiagramConstraintKind.probability, nodes=["Dt"], func=["Subtract", ["At", "Dt", 1], 0.5]
    )
    diagram = _same_slice_diagram()
    diagram = StationaryDiagram(**{**dict(diagram), "constraints": [constraint]})
    with pytest.raises(ValueError, match="doesn't support diagram constraints"):
        flatten_stationary_step(diagram)


def test_full_observation_non_slicet() -> None:
    """full_observation_information_set raises for a non-slice-t decision."""
    network = ci_healthcare_mirror_network()
    objectives = [_objective("k", ["KUOt"], [0.0, 1.0], maximize=False)]
    diagram = StationaryDiagram.with_default_information_sets(
        network, {"GER": [], "LYHt": None}, objectives, discount=GAMMA
    )
    with pytest.raises(ValueError, match="isn't a slice-t decision"):
        full_observation_information_set(diagram, "GER")


def test_value_iteration_unknown_objective() -> None:
    """stationary_value_iteration raises for an unknown objective symbol."""
    flat = flatten_stationary_step(_mdp_diagram())
    with pytest.raises(ValueError, match="Unknown objective"):
        stationary_value_iteration(flat, "nonexistent")


def test_randomised_policy_wrong_length() -> None:
    """RandomisedPolicy rejects a distribution with the wrong number of entries."""
    with pytest.raises(ValueError, match="probabilities, expected"):
        RandomisedPolicy(symbol="D", information_set=[], states=["a", "b"], probabilities={0: [0.5]})


def test_randomised_policy_not_summing_to_one() -> None:
    """RandomisedPolicy rejects a distribution that does not sum to 1."""
    with pytest.raises(ValueError, match="sums to"):
        RandomisedPolicy(symbol="D", information_set=[], states=["a", "b"], probabilities={0: [0.3, 0.4]})


def test_randomised_policy_negative_entry() -> None:
    """RandomisedPolicy rejects a distribution with a negative entry."""
    with pytest.raises(ValueError, match="negative"):
        RandomisedPolicy(symbol="D", information_set=[], states=["a", "b"], probabilities={0: [1.5, -0.5]})


def test_evaluate_missing_information_state() -> None:
    """evaluate_stationary_policy raises when a RandomisedPolicy is missing an info state."""
    diagram = _mdp_diagram()
    decision = diagram.decisions[0]
    partial = RandomisedPolicy(
        symbol=decision.symbol,
        information_set=decision.information_set,
        states=decision.states,
        probabilities={0: [0.0, 0.0, 1.0]},
    )
    with pytest.raises(ValueError, match="no choice for information states"):
        evaluate_stationary_policy(diagram, {decision.symbol: partial})


@pytest.mark.parametrize("tol", [0.0, np.inf, np.nan])
def test_value_iteration_invalid_tol(tol: float) -> None:
    """stationary_value_iteration raises for a tolerance that isn't finite and positive."""
    flat = flatten_stationary_step(_mdp_diagram())
    with pytest.raises(ValueError, match="tol must be finite and positive"):
        stationary_value_iteration(flat, "x", tol=tol)


@pytest.mark.parametrize("max_iterations", [0, -1, 1.5, True])
def test_value_iteration_invalid_max_iterations(max_iterations: int) -> None:
    """stationary_value_iteration raises for a max_iterations that isn't a positive integer."""
    flat = flatten_stationary_step(_mdp_diagram())
    with pytest.raises(ValueError, match="max_iterations must be a positive integer"):
        stationary_value_iteration(flat, "x", max_iterations=max_iterations)


def test_value_iteration_non_finite_coefficients() -> None:
    """stationary_value_iteration raises for an objective with a non-finite coefficient."""
    flat = flatten_stationary_step(_mdp_diagram())
    branch = flat.branches[0]
    coefficients = branch.objective_coefficients["x"].copy()
    coefficients[0] = np.nan
    branch = branch.model_copy(update={"objective_coefficients": {**branch.objective_coefficients, "x": coefficients}})
    flat = flat.model_copy(update={"branches": [branch]})
    with pytest.raises(ValueError, match="non-finite coefficients"):
        stationary_value_iteration(flat, "x")


def test_value_iteration_max_iterations() -> None:
    """stationary_value_iteration raises when policy iteration needs more steps than max_iterations."""
    flat = flatten_stationary_step(_mdp_diagram())
    # In state 0, action lo is free but slowly leaks into the absorbing costly state 1, which the
    # value-iteration warm start undervalues, so policy iteration needs a second step to switch to mid.
    kernel = sp.csr_array(np.array([[0.999, 1.0, 1.0, 0.0, 0.0, 0.0], [0.001, 0.0, 0.0, 1.0, 1.0, 1.0]]))
    coefficients = np.array([0.0, 0.5, 0.5, 1.0, 1.0, 1.0])
    branch = flat.branches[0]
    branch = branch.model_copy(
        update={"kernel": kernel, "objective_coefficients": {**branch.objective_coefficients, "c": coefficients}}
    )
    flat = flat.model_copy(update={"branches": [branch], "discount": 0.9999})
    with pytest.raises(RuntimeError, match="Policy iteration didn't converge"):
        stationary_value_iteration(flat, "c", max_iterations=1)
    _, policies = stationary_value_iteration(flat, "c", max_iterations=2)
    assert policies[flat.decision.symbol].probabilities[0] == [0.0, 1.0, 0.0]


def test_value_iteration_mirror_near_one_discount() -> None:
    """At discount 0.9999 on the mirror, policy iteration converges and its value is its policy's exact value."""
    diagram = StationaryDiagram(**{**dict(_mirror_diagram()), "discount": 0.9999})
    flat = flatten_stationary_step(diagram)
    value, policies = stationary_value_iteration(flat, "k")
    assert evaluate_stationary_policy(diagram, policies).objectives["k"] == pytest.approx(value, abs=1e-10)
