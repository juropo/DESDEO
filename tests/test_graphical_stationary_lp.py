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
from fixtures import (
    atemporal_2tbn_network,
    ci_healthcare_mirror_network,
    ci_healthcare_per_step_mirror_network,
    decision_chain_2tbn_network,
    mdp_2tbn_network,
    same_slice_2tbn_network,
)
from test_graphical_stationary import GAMMA, _replace_cpt, _unrolled_average

from desdeo.problem.graphical import (
    DecisionPolicy,
    DiagramConstraint,
    DiagramConstraintKind,
    DiscreteNetwork,
    FlattenedStep,
    ObjectiveNode,
    Potential,
    RandomisedPolicy,
    StationaryDiagram,
    dynamic,
    evaluate_stationary_policy,
    flatten_stationary_step,
    full_observation_information_set,
    initial_state_distribution,
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


def _full_observation_diagram(
    network: DiscreteNetwork, decisions: list[str], objectives: list[ObjectiveNode]
) -> StationaryDiagram:
    """`network` with `decisions` observing their full-observation sets."""
    provisional = StationaryDiagram.with_default_information_sets(
        network, dict.fromkeys(decisions), objectives, discount=GAMMA
    )
    full_sets = {symbol: full_observation_information_set(provisional, symbol) for symbol in decisions}
    return StationaryDiagram.with_default_information_sets(network, full_sets, objectives, discount=GAMMA)


def _mdp_diagram() -> StationaryDiagram:
    """Mdp diagram with full-observation Dt."""
    objectives = [
        _objective("x", ["Xt"], [1.0, 3.0], maximize=True),
        _objective("c", ["Dt"], [0.0, 0.4, 1.0], maximize=False),
    ]
    return _full_observation_diagram(mdp_2tbn_network(), ["Dt"], objectives)


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
    return _full_observation_diagram(network, ["Dt"], objectives)


def _atemporal_diagram() -> StationaryDiagram:
    """`atemporal_2tbn_network` with atemporal G and full-observation Xt."""
    objectives = [
        _objective("x", ["A", "Xt"], [[1.0, 0.0], [0.2, 0.9]], maximize=True),
        _objective(
            "g",
            ["G", "A", "Xt"],
            [[[1.0, 0.2], [0.9, 0.6]], [[0.4, 0.8], [0.1, 0.7]], [[0.3, 0.5], [0.6, 0.0]]],
            maximize=False,
        ),
    ]
    return _full_observation_diagram(atemporal_2tbn_network(), ["G", "Xt"], objectives)


def _chain_diagram(*, atemporal: bool = False) -> StationaryDiagram:
    """`decision_chain_2tbn_network` with full-observation Dt and Et, and atemporal G if `atemporal`."""
    objectives = [
        _objective("x", ["Xt"], [1.0, 3.0], maximize=True),
        _objective("c", ["Dt", "Yt", "Et"], [[[0.3, 0.8], [0.5, 0.5]], [[0.1, 0.9], [1.0, 0.2]]], maximize=False),
    ]
    decisions = ["G", "Dt", "Et"] if atemporal else ["Dt", "Et"]
    return _full_observation_diagram(decision_chain_2tbn_network(atemporal), decisions, objectives)


_MIRROR_OBJECTIVES = [
    _objective("k", ["KUOt"], [0.0, 1.0], maximize=False),
    _objective("a", ["ASTt", "LYHt"], [[0.0, 0.5, 1.0], [0.3, 0.2, 0.9]], maximize=True),
]


def _mirror_diagram() -> StationaryDiagram:
    """Mirror diagram with atemporal GER and full-observation LYHt."""
    objectives = [
        *_MIRROR_OBJECTIVES,
        _objective(
            "m",
            ["OMA", "PIHt", "GER"],
            [[[0.0, 0.1], [0.2, 0.3]], [[0.4, 0.5], [0.6, 0.7]], [[0.8, 0.9], [1.0, 0.0]]],
            maximize=False,
        ),
    ]
    return _full_observation_diagram(ci_healthcare_mirror_network(), ["GER", "LYHt"], objectives)


def _per_step_mirror_diagram() -> StationaryDiagram:
    """Per-step mirror with atemporal OMA and the full-observation chain GERt -> LYHt."""
    objectives = [
        *_MIRROR_OBJECTIVES,
        _objective(
            "m",
            ["OMA", "PIHt", "GERt"],
            [[[0.0, 0.1], [0.2, 0.3]], [[0.4, 0.5], [0.6, 0.7]], [[0.8, 0.9], [1.0, 0.0]]],
            maximize=False,
        ),
    ]
    return _full_observation_diagram(ci_healthcare_per_step_mirror_network(), ["OMA", "GERt", "LYHt"], objectives)


_SMALL_DIAGRAMS = {
    "mdp": _mdp_diagram,
    "same_slice": _same_slice_diagram,
    "same_slice_impossible": lambda: _same_slice_diagram(impossible_observation=True),
    "atemporal": _atemporal_diagram,
    "chain": _chain_diagram,
    "chain_atemporal": lambda: _chain_diagram(atemporal=True),
}


def _n_information(diagram: StationaryDiagram, symbol: str) -> int:
    """The number of information states of decision `symbol`."""
    cardinality = {variable.name: len(variable.labels) for variable in diagram.network.variables}
    decision = next(decision for decision in diagram.decisions if decision.symbol == symbol)
    return prod(cardinality[name] for name in decision.information_set)


def _prefix_information_keys(flat: FlattenedStep, branch_index: int, level: int) -> np.ndarray:
    """Chain decision `level`'s information-state key for each kept prefix of that level."""
    branch = flat.branches[branch_index]
    names, cards = list(flat.state_nodes), list(flat.state_cardinalities)
    for k in range(level):
        names += [*flat.observation_nodes[k], flat.decisions[k].symbol]
        cards += [*flat.observation_cardinalities[k], len(flat.decisions[k].states)]
    names += flat.observation_nodes[level]
    cards += flat.observation_cardinalities[level]
    prefixes = branch.prefixes[level]
    values = dict(zip(names, np.unravel_index(prefixes, cards), strict=True))
    values |= {symbol: np.full(len(prefixes), value) for symbol, value in branch.atemporal.items()}
    cardinality = dict(zip(names, cards, strict=True))
    cardinality |= {atemporal.symbol: len(atemporal.states) for atemporal in flat.atemporal_decisions}
    information_set = flat.decisions[level].information_set
    return np.array(
        [
            mixed_radix_encode(
                tuple(int(values[name][i]) for name in information_set), [cardinality[n] for n in information_set]
            )
            for i in range(len(prefixes))
        ],
        dtype=np.int64,
    )


def _closed_loop(
    flat: FlattenedStep, branch_index: int, tables: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Oracle: one branch's state transition `P[s', s]` and per-state rewards under the chain's policy tables."""
    branch = flat.branches[branch_index]
    n_states = prod(flat.state_cardinalities)
    weight, owner = np.ones(n_states), np.arange(n_states)
    for level, decision in enumerate(flat.decisions):
        pi = tables[decision.symbol][_prefix_information_keys(flat, branch_index, level)]
        parent = branch.parents[level]
        weight = (weight[parent][:, None] * pi).reshape(-1)
        owner = np.repeat(owner[parent], len(decision.states))
    owner_matrix = np.eye(n_states)[owner]
    transition = (branch.kernel.toarray() * weight) @ owner_matrix
    rewards = {
        symbol: (coefficients * weight) @ owner_matrix for symbol, coefficients in branch.objective_coefficients.items()
    }
    return transition, rewards


def _discounted_values(flat: FlattenedStep, transition: np.ndarray, rewards: dict[str, np.ndarray]) -> dict[str, float]:
    """Per-step averages from a state transition and per-state rewards."""
    gamma = flat.discount
    x = np.linalg.solve(np.eye(len(transition)) - gamma * transition, (1.0 - gamma) * flat.mu0)
    return {symbol: float(x @ reward) for symbol, reward in rewards.items()}


def _flat_value(
    flat: FlattenedStep, tables: dict[str, np.ndarray], marginals: dict[str, np.ndarray] | None = None
) -> dict[str, float]:
    """Oracle: linear solves on the flattened kernels, mixed over the branches by the atemporal marginals."""
    marginals = marginals or {}
    values = dict.fromkeys(flat.maximize, 0.0)
    for index, branch in enumerate(flat.branches):
        weight = prod(
            marginals[symbol][value] if symbol in marginals else 1.0 for symbol, value in branch.atemporal.items()
        )
        if weight == 0:
            continue
        for symbol, value in _discounted_values(flat, *_closed_loop(flat, index, tables)).items():
            values[symbol] += weight * value
    return values


def _deterministic_combinations(
    diagram: StationaryDiagram, flat: FlattenedStep
) -> Iterator[tuple[dict[str, float], dict[str, np.ndarray], dict[str, DecisionPolicy]]]:
    """Every branch with every deterministic chain policy on its kept prefixes; other information states choose 0.

    Yields the branch's atemporal point masses, the chain's policy tables and the matching policies.
    """
    for index, branch in enumerate(flat.branches):
        keys = [_prefix_information_keys(flat, index, level) for level in range(len(flat.decisions))]
        choice_sets = [
            itertools.product(range(len(decision.states)), repeat=len(level_keys))
            for decision, level_keys in zip(flat.decisions, keys, strict=True)
        ]
        marginals = {
            atemporal.symbol: np.eye(len(atemporal.states))[branch.atemporal[atemporal.symbol]]
            for atemporal in flat.atemporal_decisions
        }
        atemporal_policies = {
            atemporal.symbol: DecisionPolicy(
                symbol=atemporal.symbol,
                information_set=[],
                states=atemporal.states,
                policy={0: branch.atemporal[atemporal.symbol]},
            )
            for atemporal in flat.atemporal_decisions
        }
        for choices in itertools.product(*choice_sets):
            tables, policies = {}, dict(atemporal_policies)
            for decision, level_keys, level_choices in zip(flat.decisions, keys, choices, strict=True):
                choice = np.zeros(_n_information(diagram, decision.symbol), dtype=np.int64)
                choice[level_keys] = level_choices
                tables[decision.symbol] = np.eye(len(decision.states))[choice]
                policies[decision.symbol] = DecisionPolicy(
                    symbol=decision.symbol,
                    information_set=decision.information_set,
                    states=decision.states,
                    policy=dict(enumerate(choice.tolist())),
                )
            yield marginals, tables, policies


def _random_policies(
    diagram: StationaryDiagram, rng: np.random.Generator
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, RandomisedPolicy]]:
    """Random randomised policies for every decision, as atemporal marginals, chain tables and policies."""
    structure = diagram.structure
    marginals, tables, policies = {}, {}, {}
    for decision in diagram.decisions:
        table = rng.dirichlet(np.ones(len(decision.states)), size=_n_information(diagram, decision.symbol))
        if decision.symbol in structure.atemporal:
            marginals[decision.symbol] = table[0]
        else:
            tables[decision.symbol] = table
        policies[decision.symbol] = RandomisedPolicy(
            symbol=decision.symbol,
            information_set=decision.information_set,
            states=decision.states,
            probabilities=dict(enumerate(table.tolist())),
        )
    return marginals, tables, policies


# ── Tests ──────────────────────────────────────────────────────────────────────────────────────


def test_full_observation_information_set() -> None:
    """The fixtures' full observation sets match the expected lists."""
    assert _mdp_diagram().decisions[0].information_set == ["X0"]
    assert _same_slice_diagram().decisions[0].information_set == ["X0", "Yt"]

    mirror = {decision.symbol: decision.information_set for decision in _mirror_diagram().decisions}
    assert mirror == {"GER": [], "LYHt": ["GER", "OMA", "PIS0", "PIH0", "LYS0", "LYH0", "PISt", "PIHt", "LYSt"]}

    chain = {decision.symbol: decision.information_set for decision in _chain_diagram(atemporal=True).decisions}
    assert chain == {"G": [], "Dt": ["G", "X0", "Zt"], "Et": ["G", "X0", "Zt", "Dt", "Yt"]}


def test_full_observation_information_set_per_step_mirror() -> None:
    """On the per-step mirror, GERt observes the state, OMA and PISt; LYHt also GERt, PIHt and LYSt."""
    diagram = _per_step_mirror_diagram()
    state = ["PIS0", "PIH0", "LYS0", "LYH0"]
    assert diagram.structure.state_nodes == state
    observed = {decision.symbol: decision.information_set for decision in diagram.decisions}
    assert observed == {
        "OMA": [],
        "GERt": ["OMA", *state, "PISt"],
        "LYHt": ["GERt", "OMA", *state, "PISt", "PIHt", "LYSt"],
    }
    flat = flatten_stationary_step(diagram)
    assert [decision.symbol for decision in flat.decisions] == ["GERt", "LYHt"]
    assert flat.observation_nodes == [["PISt"], ["PIHt", "LYSt"]]
    assert [atemporal.symbol for atemporal in flat.atemporal_decisions] == ["OMA"]


def test_chain_prefixes_and_parents() -> None:
    """`Zt = hi` can't follow `X0 = lo` and `Yt = hi` can't follow `Dt = lo`, so those prefixes are dropped."""
    flat = flatten_stationary_step(_chain_diagram())
    assert flat.observation_nodes == [["Zt"], ["Yt"]]
    branch = flat.branches[0]
    # Level 1 is (X0, Zt); level 2 is (X0, Zt, Dt, Yt).
    assert branch.prefixes[0].tolist() == [0, 2, 3]
    assert branch.parents[0].tolist() == [0, 1, 1]
    assert branch.prefixes[1].tolist() == [0, 2, 3, 8, 10, 11, 12, 14, 15]
    assert branch.parents[1].tolist() == [0, 1, 1, 2, 3, 3, 4, 5, 5]
    assert branch.kernel.shape == (2, 18)
    assert all(len(coefficients) == 18 for coefficients in branch.objective_coefficients.values())


def test_impossible_observation_prefixes_are_dropped() -> None:
    """The prefix (X0 = 0, Yt = 2) has probability 0, so it and its columns are dropped."""
    flat = flatten_stationary_step(_same_slice_diagram(impossible_observation=True))
    branch = flat.branches[0]
    assert branch.prefixes[0].tolist() == [0, 1, 3, 4, 5]
    assert branch.parents[0].tolist() == [0, 0, 1, 1, 1]
    assert branch.kernel.shape == (2, 10)
    assert all(len(coefficients) == 10 for coefficients in branch.objective_coefficients.values())


@pytest.mark.parametrize("name", list(_SMALL_DIAGRAMS))
def test_kernel_matches_evaluation_small(name: str) -> None:
    """The flattened oracle matches evaluate_stationary_policy for every deterministic policy combination."""
    diagram = _SMALL_DIAGRAMS[name]()
    flat = flatten_stationary_step(diagram)
    for marginals, tables, policies in _deterministic_combinations(diagram, flat):
        expected = _flat_value(flat, tables, marginals)
        actual = evaluate_stationary_policy(diagram, policies).objectives
        for symbol in expected:
            assert actual[symbol] == pytest.approx(expected[symbol], abs=1e-10)


@pytest.mark.parametrize("builder", [_mirror_diagram, _per_step_mirror_diagram])
def test_kernel_matches_evaluation_mirror(builder: Callable[[], StationaryDiagram]) -> None:
    """The flattened oracle matches evaluate_stationary_policy on the mirrors with randomised policies."""
    diagram = builder()
    flat = flatten_stationary_step(diagram)
    rng = np.random.default_rng(0)
    for _ in range(50):
        marginals, tables, policies = _random_policies(diagram, rng)
        expected = _flat_value(flat, tables, marginals)
        actual = evaluate_stationary_policy(diagram, policies).objectives
        for symbol in expected:
            assert actual[symbol] == pytest.approx(expected[symbol], abs=1e-10)


@pytest.mark.parametrize("name", list(_SMALL_DIAGRAMS))
def test_value_iteration_matches_brute_force(name: str) -> None:
    """stationary_value_iteration matches the best enumerated deterministic policy combination."""
    diagram = _SMALL_DIAGRAMS[name]()
    flat = flatten_stationary_step(diagram)
    values = [
        evaluate_stationary_policy(diagram, policies).objectives
        for _, _, policies in _deterministic_combinations(diagram, flat)
    ]
    for symbol, maximize in flat.maximize.items():
        best = (max if maximize else min)(value[symbol] for value in values)
        vi_value, vi_policies = stationary_value_iteration(flat, symbol)
        assert vi_value == pytest.approx(best, abs=1e-10)
        assert set(vi_policies) == {decision.symbol for decision in diagram.decisions}
        assert evaluate_stationary_policy(diagram, vi_policies).objectives[symbol] == pytest.approx(vi_value, abs=1e-10)


@pytest.mark.parametrize("discount", [0.99, 0.999])
@pytest.mark.parametrize("builder", [_mdp_diagram, _same_slice_diagram, _chain_diagram])
def test_value_iteration_exact_at_high_discount(builder: Callable[[], StationaryDiagram], discount: float) -> None:
    """At discounts near 1, VI matches brute force and the exact evaluation of its own policy within 1e-10."""
    diagram = StationaryDiagram(**{**dict(builder()), "discount": discount})
    flat = flatten_stationary_step(diagram)
    values = [
        evaluate_stationary_policy(diagram, policies).objectives
        for _, _, policies in _deterministic_combinations(diagram, flat)
    ]
    for symbol, maximize in flat.maximize.items():
        vi_value, vi_policies = stationary_value_iteration(flat, symbol)
        assert vi_value == pytest.approx((max if maximize else min)(value[symbol] for value in values), abs=1e-10)
        assert evaluate_stationary_policy(diagram, vi_policies).objectives[symbol] == pytest.approx(vi_value, abs=1e-10)


@pytest.mark.parametrize("builder", [_mirror_diagram, _per_step_mirror_diagram])
def test_value_iteration_on_mirror_beats_random_policies(builder: Callable[[], StationaryDiagram]) -> None:
    """VI value on the mirrors is at least as good as 20 random randomised policies per objective."""
    diagram = builder()
    flat = flatten_stationary_step(diagram)
    rng = np.random.default_rng(42)
    for symbol, maximize in flat.maximize.items():
        vi_value, vi_policies = stationary_value_iteration(flat, symbol)
        assert evaluate_stationary_policy(diagram, vi_policies).objectives[symbol] == pytest.approx(vi_value, abs=1e-10)
        for _ in range(20):
            random_value = evaluate_stationary_policy(diagram, _random_policies(diagram, rng)[2]).objectives[symbol]
            if maximize:
                assert vi_value >= random_value - 1e-12
            else:
                assert vi_value <= random_value + 1e-12


def test_randomised_policy_matches_unrolled_network() -> None:
    """RandomisedPolicy on same_slice matches the CPT-pinned unrolled 2TBN average."""
    diagram = _same_slice_diagram()
    decision = diagram.decisions[0]
    rng = np.random.default_rng(3)
    probs = rng.dirichlet(np.ones(2), size=6)
    policy = RandomisedPolicy(
        symbol="Dt",
        information_set=decision.information_set,
        states=decision.states,
        probabilities=dict(enumerate(probs.tolist())),
    )
    evaluation = evaluate_stationary_policy(diagram, {"Dt": policy})

    pinned = _replace_cpt(
        diagram.network,
        "Dt",
        Potential(variables=["Dt", "X0", "Yt"], shape=[2, 2, 3], values=probs.T.reshape(2, 2, 3).tolist()),
    )
    assert evaluation.objectives["x"] == pytest.approx(_unrolled_average(pinned, "Xt", [1.0, 3.0]), abs=1e-6)


def test_randomised_chain_matches_unrolled_network() -> None:
    """Randomised G, Dt and Et on the chain fixture match the unrolled 2TBN, which draws G once."""
    diagram = _chain_diagram(atemporal=True)
    _, _, policies = _random_policies(diagram, np.random.default_rng(5))
    evaluation = evaluate_stationary_policy(diagram, policies)

    network = diagram.network
    for decision in diagram.decisions:
        rows = np.array(
            [
                policies[decision.symbol].probabilities[key]
                for key in range(len(policies[decision.symbol].probabilities))
            ]
        )
        shape = [2] * (len(decision.information_set) + 1)
        cpt = Potential(
            variables=[decision.symbol, *decision.information_set], shape=shape, values=rows.T.reshape(shape).tolist()
        )
        network = _replace_cpt(network, decision.symbol, cpt)
    assert evaluation.objectives["x"] == pytest.approx(_unrolled_average(network, "Xt", [1.0, 3.0]), abs=1e-6)


def test_randomised_atemporal_policy_is_drawn_once() -> None:
    """GER at 0.35/0.65 on the mirror evaluates to the mix of its point masses, not to a per-step redraw."""
    diagram = _mirror_diagram()
    flat = flatten_stationary_step(diagram)
    _, tables, policies = _random_policies(diagram, np.random.default_rng(1))
    masses = [0.35, 0.65]
    ger = policies["GER"].model_copy(update={"probabilities": {0: masses}})
    evaluation = evaluate_stationary_policy(diagram, {**policies, "GER": ger})

    point_masses = [
        evaluate_stationary_policy(
            diagram,
            {**policies, "GER": DecisionPolicy(symbol="GER", information_set=[], states=ger.states, policy={0: g})},
        )
        for g in range(2)
    ]
    for symbol, value in evaluation.objectives.items():
        assert value == pytest.approx(
            sum(m * e.objectives[symbol] for m, e in zip(masses, point_masses, strict=True)), abs=1e-12
        )
    mixed_occupation = sum(m * np.array(e.occupation.values) for m, e in zip(masses, point_masses, strict=True))
    np.testing.assert_allclose(evaluation.occupation.values, mixed_occupation, rtol=0, atol=1e-12)
    assert evaluation.objectives == pytest.approx(_flat_value(flat, tables, {"GER": np.array(masses)}), abs=1e-10)

    # Redrawing GER every step mixes the branches' transitions instead of their values.
    loops = [_closed_loop(flat, g, tables) for g in range(2)]
    transition = sum(m * loop[0] for m, loop in zip(masses, loops, strict=True))
    rewards = {
        symbol: sum(m * loop[1][symbol] for m, loop in zip(masses, loops, strict=True)) for symbol in flat.maximize
    }
    redrawn = _discounted_values(flat, transition, rewards)
    assert abs(evaluation.objectives["a"] - redrawn["a"]) > 5e-4


def test_point_mass_randomised_atemporal_policy_matches_deterministic() -> None:
    """A point-mass RandomisedPolicy for an atemporal decision evaluates exactly like its DecisionPolicy."""
    diagram = _per_step_mirror_diagram()
    _, _, policies = _random_policies(diagram, np.random.default_rng(2))
    oma = policies["OMA"]
    randomised = evaluate_stationary_policy(
        diagram, {**policies, "OMA": oma.model_copy(update={"probabilities": {0: [0.0, 1.0, 0.0]}})}
    )
    deterministic = evaluate_stationary_policy(
        diagram, {**policies, "OMA": DecisionPolicy(symbol="OMA", information_set=[], states=oma.states, policy={0: 1})}
    )
    assert randomised == deterministic


def test_two_randomised_atemporal_decisions() -> None:
    """Randomised GER and OMA on the mirror match the oracle mixed by the product of their marginals.

    With OMA at its status quo, its CPT is its marginal.
    """
    diagram = _full_observation_diagram(
        ci_healthcare_mirror_network(), ["GER", "OMA", "LYHt"], _mirror_diagram().objectives
    )
    flat = flatten_stationary_step(diagram)
    assert len(flat.branches) == 6
    rng = np.random.default_rng(7)
    for _ in range(3):
        marginals, tables, policies = _random_policies(diagram, rng)
        evaluation = evaluate_stationary_policy(diagram, policies)
        assert evaluation.objectives == pytest.approx(_flat_value(flat, tables, marginals), abs=1e-10)

    marginals, tables, policies = _random_policies(diagram, rng)
    evaluation = evaluate_stationary_policy(diagram, {**policies, "OMA": None})
    oma = np.array(diagram.network.potentials["OMA"].values)
    expected = _flat_value(flat, tables, {"GER": marginals["GER"], "OMA": oma})
    assert evaluation.objectives == pytest.approx(expected, abs=1e-10)


def _observing_atemporal_policy() -> tuple[StationaryDiagram, RandomisedPolicy, np.ndarray]:
    """`atemporal_2tbn_network` with G observing A, and a randomised policy for G with its rows."""
    objectives = [_objective("x", ["Xt"], [1.0, 3.0], maximize=True)]
    diagram = StationaryDiagram.with_default_information_sets(
        atemporal_2tbn_network(), {"G": ["A"]}, objectives, discount=GAMMA
    )
    rows = np.array([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]])
    policy = RandomisedPolicy(
        symbol="G",
        information_set=["A"],
        states=diagram.decisions[0].states,
        probabilities=dict(enumerate(rows.tolist())),
    )
    return diagram, policy, rows


def test_randomised_atemporal_policy_with_observations_matches_unrolled_network() -> None:
    """G drawn once given A matches the unrolled 2TBN with G's CPT pinned to the policy."""
    diagram, policy, rows = _observing_atemporal_policy()
    evaluation = evaluate_stationary_policy(diagram, {"G": policy})

    pinned = _replace_cpt(diagram.network, "G", Potential(variables=["G", "A"], shape=[3, 2], values=rows.T.tolist()))
    assert evaluation.objectives["x"] == pytest.approx(_unrolled_average(pinned, "Xt", [1.0, 3.0]), abs=1e-6)


def test_randomised_atemporal_draws_with_initial_distribution(monkeypatch: pytest.MonkeyPatch) -> None:
    """With initial_distribution set, mixing G's 9 draws matches the drawn-once evaluation; too many draws raise."""
    diagram, policy, _ = _observing_atemporal_policy()
    expected = evaluate_stationary_policy(diagram, {"G": policy})
    state_nodes = diagram.structure.state_nodes
    mu = initial_state_distribution(diagram)
    explicit = StationaryDiagram(
        **{
            **dict(diagram),
            "initial_distribution": Potential(variables=state_nodes, shape=list(mu.shape), values=mu.tolist()),
        }
    )
    evaluation = evaluate_stationary_policy(explicit, {"G": policy})
    assert evaluation.objectives == pytest.approx(expected.objectives, abs=1e-12)
    np.testing.assert_allclose(evaluation.occupation.values, expected.occupation.values, rtol=0, atol=1e-12)

    monkeypatch.setattr(dynamic, "_MAX_ATEMPORAL_DRAWS", 8)
    with pytest.raises(ValueError, match="9 deterministic draws"):
        evaluate_stationary_policy(explicit, {"G": policy})


def test_flatten_without_slicet_decisions() -> None:
    """`flatten_stationary_step` raises when there is no slice-t decision."""
    network = atemporal_2tbn_network()
    objectives = [_objective("x", ["Xt"], [1.0, 3.0], maximize=True)]
    diagram = StationaryDiagram.with_default_information_sets(network, {"G": None}, objectives, discount=GAMMA)
    with pytest.raises(ValueError, match="at least one slice-t decision"):
        flatten_stationary_step(diagram)


def test_flatten_slicet_decisions_not_a_chain() -> None:
    """`flatten_stationary_step` raises when neither of two slice-t decisions is an ancestor of the other."""
    network = ci_healthcare_mirror_network()
    objectives = [_objective("k", ["KUOt"], [0.0, 1.0], maximize=False)]
    diagram = StationaryDiagram.with_default_information_sets(
        network, {"GER": [], "ASTt": None, "KUOt": None}, objectives, discount=GAMMA
    )
    with pytest.raises(ValueError, match="form a chain"):
        flatten_stationary_step(diagram)
    with pytest.raises(ValueError, match="form a chain"):
        full_observation_information_set(diagram, "ASTt")


@pytest.mark.parametrize(
    ("builder", "decisions"),
    [
        (same_slice_2tbn_network, {"Dt": ["Yt"]}),
        (decision_chain_2tbn_network, {"Dt": ["X0"], "Et": ["X0", "Zt", "Dt", "Yt"]}),
        (decision_chain_2tbn_network, {"Dt": ["X0", "Zt"], "Et": ["X0", "Zt", "Yt"]}),
    ],
)
def test_flatten_partial_information_set(
    builder: Callable[[], DiscreteNetwork], decisions: dict[str, list[str]]
) -> None:
    """`flatten_stationary_step` raises when a chain decision's info set is not its full observation set."""
    objectives = [_objective("x", ["Xt"], [1.0, 3.0], maximize=True)]
    diagram = StationaryDiagram.with_default_information_sets(builder(), decisions, objectives, discount=GAMMA)
    with pytest.raises(ValueError, match="full observation set"):
        flatten_stationary_step(diagram)


def test_flatten_atemporal_observations() -> None:
    """`flatten_stationary_step` raises when an atemporal decision observes something."""
    network = ci_healthcare_mirror_network()
    objectives = [_objective("k", ["KUOt"], [0.0, 1.0], maximize=False)]
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


def test_full_observation_of_non_decision() -> None:
    """full_observation_information_set raises for a node that isn't a decision."""
    with pytest.raises(ValueError, match="isn't a decision"):
        full_observation_information_set(_mirror_diagram(), "PISt")


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
    assert policies[flat.decisions[0].symbol].probabilities[0] == [0.0, 1.0, 0.0]


def test_value_iteration_mirror_near_one_discount() -> None:
    """At discount 0.9999 on the mirror, policy iteration converges and its value is its policy's exact value."""
    diagram = StationaryDiagram(**{**dict(_mirror_diagram()), "discount": 0.9999})
    flat = flatten_stationary_step(diagram)
    value, policies = stationary_value_iteration(flat, "k")
    assert evaluate_stationary_policy(diagram, policies).objectives["k"] == pytest.approx(value, abs=1e-10)
