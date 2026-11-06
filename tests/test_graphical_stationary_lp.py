"""Tests for `desdeo.problem.graphical.stationary_lp`.

Oracles:

1. **`evaluate_stationary_policy`**, exact policy evaluation on the 2TBN, against a linear solve on
   the flattened kernel.
2. **Brute-force enumeration** of deterministic policies, against `stationary_value_iteration`.
3. **pyAgrum-unrolled marginals** (`_unrolled_average`), against randomised policy evaluation.
4. **`stationary_value_iteration`** and enumeration, against `compile_stationary_lp`'s LP optimum.
5. **A propagated population** (`_renewal_oracle`), whose renewed mass restarts from mu0 with fresh
   atemporal draws, against evaluation with renewal.
6. **An explicit einsum of the step joint** (`_joint_oracle`) over the flattened oracle's occupation,
   against `stationary_step_posteriors`.
"""

import inspect
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
from scipy.linalg import block_diag
from test_graphical_stationary import GAMMA, _replace_cpt, _unrolled_average

from desdeo.problem.graphical import (
    DecisionNode,
    DecisionPolicy,
    DiagramConstraint,
    DiagramConstraintKind,
    DiscreteNetwork,
    FlattenedStep,
    ObjectiveNode,
    Potential,
    RandomisedPolicy,
    StationaryDiagram,
    StationaryLPBranch,
    StationaryLPIndexMap,
    compile_stationary_diagram,
    compile_stationary_lp,
    dynamic,
    evaluate_stationary_policy,
    flatten_stationary_step,
    full_observation_information_set,
    initial_state_distribution,
    read_randomised_policy,
    stationary_payoff_table,
    stationary_step_posteriors,
    stationary_value_iteration,
)
from desdeo.problem.graphical.index_map import mixed_radix_encode
from desdeo.problem.schema import Constraint, ConstraintTypeEnum, Problem
from desdeo.problem.testproblems import ci_healthcare_lp_diagram, ci_healthcare_lp_problem
from desdeo.problem.testproblems.ci_healthcare_problem import _PURE_BIF_PATH, _load_normalised_network
from desdeo.tools.cvxpy_solver_interfaces import CVXPYSolver, CVXPYSolverOptions
from desdeo.tools.generics import SolverResults
from desdeo.tools.gurobipy_solver_interfaces import GurobipySolver
from desdeo.tools.scalarization import add_asf_diff

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


def _per_step_mirror_diagram(*, renewal: bool = False) -> StationaryDiagram:
    """Per-step mirror with atemporal OMA and the full-observation chain GERt -> LYHt; `KUOt = 0` renews if asked."""
    objectives = [
        *_MIRROR_OBJECTIVES,
        _objective(
            "m",
            ["OMA", "PIHt", "GERt"],
            [[[0.0, 0.1], [0.2, 0.3]], [[0.4, 0.5], [0.6, 0.7]], [[0.8, 0.9], [1.0, 0.0]]],
            maximize=False,
        ),
    ]
    diagram = _full_observation_diagram(ci_healthcare_per_step_mirror_network(), ["OMA", "GERt", "LYHt"], objectives)
    return StationaryDiagram(**{**dict(diagram), "renewal": {"KUOt": [0]}}) if renewal else diagram


def _renewal_diagram(*, chain: bool = False, atemporal: bool = False) -> StationaryDiagram:
    """`decision_chain_2tbn_network` with `Kt = hi` renewing; full-observation Dt, Et if `chain`, G if `atemporal`."""
    objectives = [
        _objective("x", ["Xt"], [1.0, 3.0], maximize=True),
        _objective("c", ["Dt", "Yt", "Kt"], [[[0.3, 2.0], [0.5, 1.5]], [[0.1, 1.9], [1.0, 1.2]]], maximize=False),
    ]
    decisions = [*(["G"] if atemporal else []), "Dt", *(["Et"] if chain else [])]
    network = decision_chain_2tbn_network(atemporal, death=True)
    diagram = _full_observation_diagram(network, decisions, objectives)
    return StationaryDiagram(**{**dict(diagram), "renewal": {"Kt": [1]}})


_SMALL_DIAGRAMS = {
    "mdp": _mdp_diagram,
    "same_slice": _same_slice_diagram,
    "same_slice_impossible": lambda: _same_slice_diagram(impossible_observation=True),
    "atemporal": _atemporal_diagram,
    "chain": _chain_diagram,
    "chain_atemporal": lambda: _chain_diagram(atemporal=True),
    "renewal": _renewal_diagram,
    "renewal_atemporal": lambda: _renewal_diagram(atemporal=True),
    "renewal_chain": lambda: _renewal_diagram(chain=True),
    "renewal_chain_atemporal": lambda: _renewal_diagram(chain=True, atemporal=True),
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
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Oracle: one branch's transition `P[s', s]` without renewals, rewards and renewal probabilities per state.

    Under the chain's policy tables; the renewal probabilities are 0 without renewal.
    """
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
    renewed = np.zeros(n_states) if branch.renewal is None else (branch.renewal * weight) @ owner_matrix
    rewards = {
        symbol: (coefficients * weight) @ owner_matrix for symbol, coefficients in branch.objective_coefficients.items()
    }
    return transition, rewards, renewed


def _discounted_values(flat: FlattenedStep, transition: np.ndarray, rewards: dict[str, np.ndarray]) -> dict[str, float]:
    """Per-step averages from a state transition and per-state rewards."""
    gamma = flat.discount
    x = np.linalg.solve(np.eye(len(transition)) - gamma * transition, (1.0 - gamma) * flat.mu0)
    return {symbol: float(x @ reward) for symbol, reward in rewards.items()}


def _branch_occupation(
    flat: FlattenedStep, tables: dict[str, np.ndarray], marginals: dict[str, np.ndarray] | None = None
) -> tuple[np.ndarray, list[tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]]]:
    """Oracle: the occupation over (branch, state) by a linear solve on the flattened kernels, and each branch's loop.

    Every arrival, initial or renewed, draws its branch from the product of the atemporal marginals
    and its state from mu0.
    """
    marginals = marginals or {}
    weights = np.array(
        [
            prod(marginals[symbol][value] if symbol in marginals else 1.0 for symbol, value in branch.atemporal.items())
            for branch in flat.branches
        ]
    )
    loops = [_closed_loop(flat, index, tables) for index in range(len(flat.branches))]
    arrivals = np.kron(weights, flat.mu0)
    transition = block_diag(*(loop[0] for loop in loops))
    transition += np.outer(arrivals, np.concatenate([loop[2] for loop in loops]))
    gamma = flat.discount
    return np.linalg.solve(np.eye(len(arrivals)) - gamma * transition, (1.0 - gamma) * arrivals), loops


def _flat_value(
    flat: FlattenedStep, tables: dict[str, np.ndarray], marginals: dict[str, np.ndarray] | None = None
) -> dict[str, float]:
    """Oracle: per-step averages from `_branch_occupation`."""
    x, loops = _branch_occupation(flat, tables, marginals)
    return {symbol: float(x @ np.concatenate([loop[1][symbol] for loop in loops])) for symbol in flat.maximize}


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


def _pinned_network(diagram: StationaryDiagram, policies: dict[str, RandomisedPolicy]) -> DiscreteNetwork:
    """`diagram.network` with each decision in `policies` having its CPT replaced by its policy."""
    cardinality = {variable.name: len(variable.labels) for variable in diagram.network.variables}
    network = diagram.network
    for decision in diagram.decisions:
        if decision.symbol not in policies:
            continue
        probabilities = policies[decision.symbol].probabilities
        rows = np.array([probabilities[key] for key in range(len(probabilities))])
        shape = [len(decision.states), *(cardinality[name] for name in decision.information_set)]
        cpt = Potential(
            variables=[decision.symbol, *decision.information_set], shape=shape, values=rows.T.reshape(shape).tolist()
        )
        network = _replace_cpt(network, decision.symbol, cpt)
    return network


def _cpt_product(cpts: dict[str, tuple[list[str], np.ndarray]], nodes: list[str], assignment: dict[str, int]) -> float:
    """The product of `nodes`' CPT entries at `assignment`."""
    return prod(float(cpts[name][1][tuple(assignment[axis] for axis in cpts[name][0])]) for name in nodes)


def _renewal_oracle(
    network: DiscreteNetwork,
    next_of: dict[str, str],
    renewal: dict[str, list[int]],
    objectives: list[ObjectiveNode],
    gamma: float,
    *,
    horizon: int = 2000,
) -> dict[str, float]:
    """Oracle: per-step averages over `horizon` steps of a population whose renewed members restart from mu0.

    Enumerates the joint of the state nodes (`next_of`'s keys, whose parents must be state nodes)
    and the slice-t nodes from `network`'s CPTs. An atemporal node maps to itself in `next_of`: it
    is drawn from its CPT in mu0 and kept while alive. Each step, the state distribution moves along
    the steps without a renewal, and the renewed mass is reset to mu0, which redraws the atemporal
    nodes.
    """
    cpts = {name: (potential.variables, np.array(potential.values)) for name, potential in network.potentials.items()}
    cards = {variable.name: len(variable.labels) for variable in network.variables}
    state = list(next_of)
    step = [variable.name for variable in network.variables if variable.name.endswith("t")]
    state_cards = [cards[name] for name in state]
    n_states = prod(state_cards)

    mu0 = np.zeros(n_states)
    for values in itertools.product(*(range(card) for card in state_cards)):
        mu0[np.ravel_multi_index(values, state_cards)] = _cpt_product(
            cpts, state, dict(zip(state, values, strict=True))
        )

    alive, renewed = np.zeros((n_states, n_states)), np.zeros(n_states)
    utilities = {objective.symbol: np.zeros(n_states) for objective in objectives}
    nodes = [*state, *step]
    for values in itertools.product(*(range(cards[name]) for name in nodes)):
        assignment = dict(zip(nodes, values, strict=True))
        weight = _cpt_product(cpts, step, assignment)
        s = np.ravel_multi_index([assignment[name] for name in state], state_cards)
        if any(assignment[name] in states for name, states in renewal.items()):
            renewed[s] += weight
        else:
            alive[s, np.ravel_multi_index([assignment[next_of[name]] for name in state], state_cards)] += weight
        for objective in objectives:
            utility = np.array(objective.utility.values)[tuple(assignment[parent] for parent in objective.parents)]
            utilities[objective.symbol][s] += weight * utility

    distribution, totals = mu0, dict.fromkeys(utilities, 0.0)
    for k in range(horizon):
        for symbol, utility in utilities.items():
            totals[symbol] += gamma**k * float(distribution @ utility)
        distribution = distribution @ alive + float(distribution @ renewed) * mu0
    return {symbol: (1 - gamma) * total for symbol, total in totals.items()}


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


@pytest.mark.parametrize(
    "builder",
    [_mirror_diagram, _per_step_mirror_diagram, lambda: _per_step_mirror_diagram(renewal=True)],
    ids=["mirror", "per_step_mirror", "per_step_mirror_renewal"],
)
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


@pytest.mark.parametrize(
    "builder",
    [_mirror_diagram, _per_step_mirror_diagram, lambda: _per_step_mirror_diagram(renewal=True)],
    ids=["mirror", "per_step_mirror", "per_step_mirror_renewal"],
)
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


@pytest.mark.parametrize("chain", [False, True])
@pytest.mark.parametrize("atemporal", [False, True])
def test_renewal_matches_propagated_population(chain: bool, atemporal: bool) -> None:
    """The status quo and random randomised policies evaluate to the propagated population's averages.

    G, status quo or randomised, is a state node drawn from its CPT or policy in mu0, so every
    renewal redraws it.
    """
    diagram = _renewal_diagram(chain=chain, atemporal=atemporal)
    next_of = {"G": "G", "X0": "Xt"} if atemporal else {"X0": "Xt"}

    def oracle(network: DiscreteNetwork) -> dict[str, float]:
        return _renewal_oracle(network, next_of, diagram.renewal, diagram.objectives, diagram.discount)

    status_quo = evaluate_stationary_policy(diagram, {}).objectives
    assert status_quo == pytest.approx(oracle(diagram.network), abs=1e-8)
    rng = np.random.default_rng(11)
    for _ in range(3):
        _, _, policies = _random_policies(diagram, rng)
        expected = oracle(_pinned_network(diagram, policies))
        assert evaluate_stationary_policy(diagram, policies).objectives == pytest.approx(expected, abs=1e-8)


def test_renewal_redraws_randomised_atemporal_policy() -> None:
    """A fractional G with renewal differs from mixing its point-mass evaluations, which keep G across renewals."""
    diagram = _renewal_diagram(chain=True, atemporal=True)
    _, _, policies = _random_policies(diagram, np.random.default_rng(5))
    q = [0.3, 0.7]
    fresh = evaluate_stationary_policy(
        diagram, {**policies, "G": policies["G"].model_copy(update={"probabilities": {0: q}})}
    ).objectives
    point_masses = [
        evaluate_stationary_policy(
            diagram,
            {**policies, "G": DecisionPolicy(symbol="G", information_set=[], states=["lo", "hi"], policy={0: g})},
        ).objectives
        for g in range(2)
    ]
    kept = {
        symbol: sum(weight * values[symbol] for weight, values in zip(q, point_masses, strict=True)) for symbol in fresh
    }
    assert max(abs(fresh[symbol] - kept[symbol]) for symbol in fresh) > 1e-6


def test_renewal_with_initial_distribution_and_randomised_atemporal_policy_raises() -> None:
    """With renewal and initial_distribution, a randomised atemporal policy raises; a point mass doesn't."""
    diagram = _renewal_diagram(atemporal=True)
    mu = initial_state_distribution(diagram)
    explicit = StationaryDiagram(
        **{
            **dict(diagram),
            "initial_distribution": Potential(
                variables=diagram.structure.state_nodes, shape=list(mu.shape), values=mu.tolist()
            ),
        }
    )
    _, _, policies = _random_policies(diagram, np.random.default_rng(3))
    with pytest.raises(NotImplementedError, match="initial_distribution and renewal"):
        evaluate_stationary_policy(explicit, policies)
    point_mass = {**policies, "G": policies["G"].model_copy(update={"probabilities": {0: [0.0, 1.0]}})}
    assert evaluate_stationary_policy(explicit, point_mass) == evaluate_stationary_policy(diagram, point_mass)


def test_renewal_changes_the_value() -> None:
    """Without renewal the oracle matches the plain evaluation, and the renewal moves both objectives."""
    diagram = _renewal_diagram(chain=True)
    plain = StationaryDiagram(**{**dict(diagram), "renewal": None})
    without = evaluate_stationary_policy(plain, {}).objectives
    oracle = _renewal_oracle(diagram.network, {"X0": "Xt"}, {}, diagram.objectives, diagram.discount)
    assert without == pytest.approx(oracle, abs=1e-8)
    with_renewal = evaluate_stationary_policy(diagram, {}).objectives
    assert all(abs(with_renewal[symbol] - without[symbol]) > 1e-3 for symbol in without)


@pytest.mark.parametrize(
    ("renewal", "match"),
    [
        ({"Dt": [0]}, "is a decision"),
        ({"X0": [0]}, "must be a slice-t chance node"),
        ({"Kt": [2]}, "out of range"),
        ({"Kt": []}, "no renewal states"),
    ],
)
def test_invalid_renewal_raises(renewal: dict[str, list[int]], match: str) -> None:
    """A decision or state node as renewal node, or an out-of-range renewal state, raises."""
    with pytest.raises(ValueError, match=match):
        StationaryDiagram(**{**dict(_renewal_diagram()), "renewal": renewal})


def test_compile_stationary_diagram_with_renewal_raises() -> None:
    """The MILP path doesn't support renewal."""
    with pytest.raises(NotImplementedError, match="renewal"):
        compile_stationary_diagram(_renewal_diagram())


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


_SOLVERS = [pytest.param("cvxpy", marks=pytest.mark.cvxpy), pytest.param("gurobipy", marks=pytest.mark.gurobipy)]


def _solve(problem: Problem, target: str, solver: str, options: dict[str, object] | None = None) -> SolverResults:
    """Solve `problem` for `target` with CVXPY + HiGHS or gurobipy, passing `options` to the solver."""
    if solver == "cvxpy":
        results = CVXPYSolver(problem, CVXPYSolverOptions(solver="HIGHS", extra_options=options)).solve(target)
    else:
        results = GurobipySolver(problem, options).solve(target)
    assert results.success, results.message
    return results


def _with_fixed_masses(problem: Problem, index_map: StationaryLPIndexMap, masses: list[float]) -> Problem:
    """`problem` with each branch mass fixed through its bounds."""
    fixed = {branch.m_symbol: mass for branch, mass in zip(index_map.branches, masses, strict=True)}
    variables = [
        variable.model_copy(update={"lowerbound": fixed[variable.symbol], "upperbound": fixed[variable.symbol]})
        if variable.symbol in fixed
        else variable
        for variable in problem.variables
    ]
    return problem.model_copy(update={"variables": variables})


def _with_arrival_probabilities(problem: Problem, index_map: StationaryLPIndexMap, q: list[float]) -> Problem:
    """`problem` with renewal and each branch's arrival probability fixed by `nu_g - q_g Σ_h nu_h == 0`."""
    inflows = [branch.nu_symbol for branch in index_map.branches]
    return problem.add_constraints(
        [
            Constraint(
                name=f"Arrival probability of branch {g}",
                symbol=f"q_{g}",
                cons_type=ConstraintTypeEnum.EQ,
                func=["Subtract", inflow, ["Multiply", probability, ["Add", *inflows]]],
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
            for g, (inflow, probability) in enumerate(zip(inflows, q, strict=True))
        ]
    )


def _check_lp_matches_value_iteration(diagram: StationaryDiagram, solver: str) -> None:
    """The LP optimum equals VI for each objective, and the read-back policies evaluate to it."""
    flat = flatten_stationary_step(diagram)
    problem, index_map = compile_stationary_lp(diagram)
    assert len(problem.constraints) == len(flat.branches) * (len(flat.decisions) + 1 + bool(diagram.renewal)) + 1
    for symbol in flat.maximize:
        results = _solve(problem, symbol, solver)
        value = results.optimal_objectives[symbol]
        assert value == pytest.approx(stationary_value_iteration(flat, symbol)[0], abs=1e-8)
        policies = read_randomised_policy(index_map, results.optimal_variables)
        assert set(policies) == {decision.symbol for decision in diagram.decisions}
        assert evaluate_stationary_policy(diagram, policies).objectives[symbol] == pytest.approx(value, abs=1e-8)


@pytest.mark.parametrize("solver", _SOLVERS)
@pytest.mark.parametrize("name", list(_SMALL_DIAGRAMS))
def test_lp_matches_value_iteration(name: str, solver: str) -> None:
    """On the small fixtures, the LP optimum equals VI and the read-back policies evaluate to it."""
    _check_lp_matches_value_iteration(_SMALL_DIAGRAMS[name](), solver)


@pytest.mark.parametrize(
    ("renewal", "solver"),
    [
        pytest.param(False, "cvxpy", marks=pytest.mark.cvxpy),
        pytest.param(True, "cvxpy", marks=pytest.mark.cvxpy),
        pytest.param(True, "gurobipy", marks=[pytest.mark.gurobipy, pytest.mark.githubskip]),
    ],
)
def test_lp_matches_value_iteration_per_step_mirror(renewal: bool, solver: str) -> None:
    """On the per-step mirror, the LP optimum equals VI and the read-back policies evaluate to it.

    With renewal the LP is too large for gurobipy's size-limited license.
    """
    _check_lp_matches_value_iteration(_per_step_mirror_diagram(renewal=renewal), solver)


@pytest.mark.cvxpy
@pytest.mark.parametrize("masses", [[0.35, 0.65, 0.0], [0.2, 0.35, 0.45]])
def test_lp_fixed_fractional_masses_per_step_mirror(masses: list[float]) -> None:
    """With OMA's masses fixed, the LP value weights each branch's optimum by its mass; read-back policies attain it."""
    diagram = _per_step_mirror_diagram()
    compiled, index_map = compile_stationary_lp(diagram)
    problem = _with_fixed_masses(compiled, index_map, masses)
    units = np.eye(len(index_map.branches)).tolist()
    for objective in diagram.objectives:
        results = _solve(problem, objective.symbol, "cvxpy")
        single = [
            _solve(_with_fixed_masses(compiled, index_map, unit), objective.symbol, "cvxpy").optimal_objectives[
                objective.symbol
            ]
            for unit in units
        ]
        assert results.optimal_objectives[objective.symbol] == pytest.approx(np.dot(masses, single), abs=1e-8)
        policies = read_randomised_policy(index_map, results.optimal_variables)
        assert policies["OMA"].probabilities[0] == pytest.approx(masses, abs=1e-8)
        assert evaluate_stationary_policy(diagram, policies).objectives[objective.symbol] == pytest.approx(
            results.optimal_objectives[objective.symbol], abs=1e-8
        )


@pytest.mark.cvxpy
@pytest.mark.parametrize("q", [[0.35, 0.65, 0.0], [0.2, 0.35, 0.45]])
def test_lp_fixed_arrival_probabilities_per_step_mirror_renewal(q: list[float]) -> None:
    """With renewal and OMA's arrival probabilities fixed, the read-back gives them and attains the LP value.

    The propagated population with OMA drawn from q at every arrival agrees.
    """
    diagram = _per_step_mirror_diagram(renewal=True)
    compiled, index_map = compile_stationary_lp(diagram)
    problem = _with_arrival_probabilities(compiled, index_map, q)
    next_of = {"HHA": "HHA", "OMA": "OMA", "PIS0": "PISt", "PIH0": "PIHt", "LYS0": "LYSt", "LYH0": "LYHt"}
    for objective in diagram.objectives:
        results = _solve(problem, objective.symbol, "cvxpy")
        value = results.optimal_objectives[objective.symbol]
        policies = read_randomised_policy(index_map, results.optimal_variables)
        assert policies["OMA"].probabilities[0] == pytest.approx(q, abs=1e-8)
        assert evaluate_stationary_policy(diagram, policies).objectives[objective.symbol] == pytest.approx(
            value, abs=1e-8
        )
        oracle = _renewal_oracle(
            _pinned_network(diagram, policies), next_of, diagram.renewal, diagram.objectives, diagram.discount
        )
        assert oracle[objective.symbol] == pytest.approx(value, abs=1e-8)


def _asf(values: dict[str, float], problem: Problem, reference: dict[str, float], rho: float, delta: float) -> float:
    """`add_asf_diff`'s scalarization at objective values `values`, with the problem's ideal and nadir."""
    terms, augmentation = [], 0.0
    for objective in problem.objectives:
        sign = -1.0 if objective.maximize else 1.0
        scale = sign * objective.nadir - (sign * objective.ideal - delta)
        terms.append(sign * (values[objective.symbol] - reference[objective.symbol]) / scale)
        augmentation += sign * values[objective.symbol] / scale
    return max(terms) + rho * augmentation


def _three_objective_chain_diagram() -> StationaryDiagram:
    """`_chain_diagram` with a third objective, the share of `Dt = hi`, minimised."""
    objectives = [*_chain_diagram().objectives, _objective("d", ["Dt"], [0.0, 1.0], maximize=False)]
    return _full_observation_diagram(decision_chain_2tbn_network(), ["Dt", "Et"], objectives)


def _fractional_counts(index_map: StationaryLPIndexMap, variables: dict[str, list[float] | float]) -> list[int]:
    """Per chain decision, the number of reached prefixes at which the LP's policy isn't a point mass."""
    counts = []
    for level, decision in enumerate(index_map.decisions):
        count = 0
        for branch in index_map.branches:
            y = np.array(variables[branch.y_symbols[level]]).reshape(-1, len(decision.states))
            totals = y.sum(axis=1)
            reached = totals > 1e-8
            count += int(((y[reached] / totals[reached, None]).max(axis=1) < 1 - 1e-6).sum())
        counts.append(count)
    return counts


@pytest.mark.parametrize("solver", _SOLVERS)
@pytest.mark.parametrize(
    "builder",
    [
        _mdp_diagram,
        _same_slice_diagram,
        _chain_diagram,
        lambda: _chain_diagram(atemporal=True),
        _three_objective_chain_diagram,
    ],
    ids=["mdp", "same_slice", "chain", "chain_atemporal", "chain_three_objectives"],
)
def test_lp_asf_beats_deterministic_policies(builder: Callable[[], StationaryDiagram], solver: str) -> None:
    """The ASF optimum reads back exactly, is no worse than any deterministic policy's ASF and isn't dominated.

    The reference point is the midpoint of the best and worst deterministic values. On the
    three-objective chain, the optimum randomises both chain decisions.
    """
    diagram = builder()
    flat = flatten_stationary_step(diagram)
    deterministic = [
        evaluate_stationary_policy(diagram, policies).objectives
        for _, _, policies in _deterministic_combinations(diagram, flat)
    ]
    best = {s: (max if m else min)(values[s] for values in deterministic) for s, m in flat.maximize.items()}
    worst = {s: (min if m else max)(values[s] for values in deterministic) for s, m in flat.maximize.items()}
    reference = {s: (best[s] + worst[s]) / 2 for s in flat.maximize}

    problem, index_map = compile_stationary_lp(diagram)
    problem = problem.update_ideal_and_nadir(new_ideal=best, new_nadir=worst)
    rho, delta = 1e-6, 1e-6
    asf_problem, target = add_asf_diff(problem, "asf", reference, rho=rho, delta=delta)
    results = _solve(asf_problem, target, solver)
    values = results.optimal_objectives

    policies = read_randomised_policy(index_map, results.optimal_variables)
    evaluated = evaluate_stationary_policy(diagram, policies).objectives
    for symbol in flat.maximize:
        assert evaluated[symbol] == pytest.approx(values[symbol], abs=1e-8)

    asf_value = _asf(values, problem, reference, rho, delta)
    assert results.scalarization_values[target] == pytest.approx(asf_value, abs=1e-8)
    assert asf_value <= min(_asf(other, problem, reference, rho, delta) for other in deterministic) + 1e-8
    signs = {s: -1.0 if m else 1.0 for s, m in flat.maximize.items()}
    for other in deterministic:
        no_worse = all(signs[s] * other[s] <= signs[s] * values[s] + 1e-9 for s in signs)
        better = any(signs[s] * other[s] < signs[s] * values[s] - 1e-9 for s in signs)
        assert not (no_worse and better)
    if builder is _three_objective_chain_diagram:
        assert all(count > 0 for count in _fractional_counts(index_map, results.optimal_variables))


@pytest.mark.parametrize("solver", _SOLVERS)
@pytest.mark.parametrize(
    "builder",
    [
        _atemporal_diagram,
        lambda: _chain_diagram(atemporal=True),
        lambda: _renewal_diagram(atemporal=True),
        lambda: _renewal_diagram(chain=True, atemporal=True),
    ],
    ids=["atemporal", "chain_atemporal", "renewal_atemporal", "renewal_chain_atemporal"],
)
def test_lp_atemporal_options(builder: Callable[[], StationaryDiagram], solver: str) -> None:
    """Deterministic and randomised atemporal choices both give the best single-branch LP."""
    diagram = builder()
    randomised, index_map = compile_stationary_lp(diagram)
    deterministic, deterministic_map = compile_stationary_lp(diagram, atemporal="deterministic")
    n_branches = len(index_map.branches)
    masses = [branch.z_symbol or branch.m_symbol for branch in deterministic_map.branches]
    fix = _with_arrival_probabilities if diagram.renewal else _with_fixed_masses

    for objective in diagram.objectives:
        symbol, sign = objective.symbol, -1.0 if objective.maximize else 1.0
        single = [
            sign
            * _solve(fix(randomised, index_map, np.eye(n_branches)[g].tolist()), symbol, solver).optimal_objectives[
                symbol
            ]
            for g in range(n_branches)
        ]
        results = _solve(deterministic, symbol, solver)
        assert sign * results.optimal_objectives[symbol] == pytest.approx(min(single), abs=1e-8)
        for mass in masses:
            assert min(abs(results.optimal_variables[mass]), abs(results.optimal_variables[mass] - 1)) < 1e-8
        assert sign * _solve(randomised, symbol, solver).optimal_objectives[symbol] == pytest.approx(
            min(single), abs=1e-8
        )


def test_compile_lp_reserved_symbol() -> None:
    """An objective symbol with the reserved prefix raises."""
    diagram = _mdp_diagram()
    objectives = [diagram.objectives[0].model_copy(update={"symbol": "slp_x"}), diagram.objectives[1]]
    diagram = StationaryDiagram(**{**dict(diagram), "objectives": objectives})
    with pytest.raises(ValueError, match="slp_x"):
        compile_stationary_lp(diagram)


def test_compile_lp_invalid_atemporal() -> None:
    """An unknown `atemporal` option raises."""
    with pytest.raises(ValueError, match="atemporal must be"):
        compile_stationary_lp(_mdp_diagram(), atemporal="mixed")


def test_compile_lp_atemporal_observations() -> None:
    """An atemporal decision with observations raises."""
    full_set = full_observation_information_set(_mirror_diagram(), "LYHt")
    diagram = StationaryDiagram.with_default_information_sets(
        ci_healthcare_mirror_network(), {"GER": ["OMA"], "LYHt": full_set}, _MIRROR_OBJECTIVES, discount=GAMMA
    )
    with pytest.raises(ValueError, match="Atemporal decision 'GER' observes"):
        compile_stationary_lp(diagram)


def _two_atemporal_index_map() -> StationaryLPIndexMap:
    """A hand-built index map: state S (2), atemporal decisions G1 (2) and G2 (3), decision D (2) observing all."""
    g1 = DecisionNode(symbol="G1", states=["a", "b"])
    g2 = DecisionNode(symbol="G2", states=["a", "b", "c"])
    decision = DecisionNode(symbol="D", states=["u", "v"], information_set=["S", "G1", "G2"])
    branches = [
        StationaryLPBranch(
            atemporal={"G1": i, "G2": j},
            prefixes=[[0, 1]],
            parents=[[0, 1]],
            m_symbol=f"m_{k}",
            x_symbol=f"x_{k}",
            y_symbols=[f"w_{k}"],
            kernel_symbol=f"T_{k}",
            coefficient_symbols={},
            block_symbols=[f"B_{k}"],
            parent_symbols=[f"P_{k}"],
            consistency_symbols=[f"c_{k}"],
            flow_symbol=f"flow_{k}",
        )
        for k, (i, j) in enumerate(itertools.product(range(2), range(3)))
    ]
    return StationaryLPIndexMap(
        state_nodes=["S"],
        state_cardinalities=[2],
        decisions=[decision],
        observation_nodes=[[]],
        observation_cardinalities=[[]],
        atemporal_decisions=[g1, g2],
        branches=branches,
        mu0_symbol="mu0",
        mass_symbol="mass",
        discount=GAMMA,
    )


def _branch_solution(masses: np.ndarray) -> dict[str, list[float] | float]:
    """A solution of `_two_atemporal_index_map` with branch masses `masses`, D choosing u in S=0 and v in S=1."""
    variables: dict[str, list[float] | float] = {}
    for k, mass in enumerate(masses):
        variables[f"m_{k}"] = float(mass)
        variables[f"x_{k}"] = [mass / 2, mass / 2]
        variables[f"w_{k}"] = [mass / 2, 0.0, 0.0, mass / 2]
    return variables


def test_read_policy_two_atemporal_product() -> None:
    """A product joint of masses reads back as its marginals."""
    index_map = _two_atemporal_index_map()
    g1, g2 = np.array([0.3, 0.7]), np.array([0.2, 0.5, 0.3])
    policies = read_randomised_policy(index_map, _branch_solution(np.outer(g1, g2).ravel()))

    assert policies["G1"].probabilities[0] == pytest.approx(g1.tolist(), abs=1e-12)
    assert policies["G2"].probabilities[0] == pytest.approx(g2.tolist(), abs=1e-12)
    for g1_value, g2_value in itertools.product(range(2), range(3)):
        for s_value, action in [(0, [1.0, 0.0]), (1, [0.0, 1.0])]:
            key = mixed_radix_encode([s_value, g1_value, g2_value], [2, 2, 3])
            assert policies["D"].probabilities[key] == pytest.approx(action)


def test_read_policy_two_atemporal_correlated() -> None:
    """A joint of masses that isn't a product of its marginals raises."""
    masses = np.array([0.5, 0.0, 0.0, 0.0, 0.5, 0.0])
    with pytest.raises(ValueError, match="product of their marginals"):
        read_randomised_policy(_two_atemporal_index_map(), _branch_solution(masses))


def test_read_policy_colliding_keys() -> None:
    """A chain decision whose information set omits the atemporal decisions raises."""
    index_map = _two_atemporal_index_map()
    decision = index_map.decisions[0].model_copy(update={"information_set": ["S"]})
    index_map = index_map.model_copy(update={"decisions": [decision]})
    with pytest.raises(ValueError, match="information-state keys of decision 'D'"):
        read_randomised_policy(index_map, _branch_solution(np.outer([0.3, 0.7], [0.2, 0.5, 0.3]).ravel()))


def test_read_policy_no_mass() -> None:
    """All masses at or below `tol` raise."""
    with pytest.raises(ValueError, match="No branch mass"):
        read_randomised_policy(_two_atemporal_index_map(), _branch_solution(np.full(6, 1e-12)))


def _joint_oracle(
    diagram: StationaryDiagram,
    flat: FlattenedStep,
    marginals: dict[str, np.ndarray],
    tables: dict[str, np.ndarray],
    policies: dict[str, RandomisedPolicy],
) -> tuple[list[str], np.ndarray]:
    """Oracle: the step joint over the atemporal decisions, the state nodes and slice t, by an explicit einsum.

    `_branch_occupation`'s x over (branch, state) times the slice-t CPTs of `_pinned_network`.
    """
    x, _ = _branch_occupation(flat, tables, marginals)
    atemporal = [decision.symbol for decision in flat.atemporal_decisions]
    x = x.reshape([*(len(decision.states) for decision in flat.atemporal_decisions), *flat.state_cardinalities])
    network = _pinned_network(diagram, policies)
    slicet = [variable.name for variable in network.variables if variable.name in diagram.structure.slicet]
    nodes = [*atemporal, *flat.state_nodes, *slicet]
    letters = {name: chr(ord("a") + i) for i, name in enumerate(nodes)}
    subscripts = ["".join(letters[name] for name in [*atemporal, *flat.state_nodes])]
    operands = [x]
    for name in slicet:
        potential = network.potentials[name]
        subscripts.append("".join(letters[axis] for axis in potential.variables))
        operands.append(np.array(potential.values))
    return nodes, np.einsum(f"{','.join(subscripts)}->{''.join(letters.values())}", *operands)


def _conditioned(
    nodes: list[str], joint: np.ndarray, evidence: dict[str, int], objectives: list[ObjectiveNode]
) -> tuple[dict[str, np.ndarray], dict[str, float], float]:
    """Oracle: posteriors, conditional expected utilities and evidence probability from a full joint."""
    for name, index in evidence.items():
        mask = np.zeros(joint.shape[nodes.index(name)])
        mask[index] = 1.0
        joint = joint * mask.reshape([-1 if node == name else 1 for node in nodes])
    probability = float(joint.sum())
    posteriors = {
        name: joint.sum(axis=tuple(i for i in range(len(nodes)) if i != axis)) / probability
        for axis, name in enumerate(nodes)
    }
    letters = {name: chr(ord("a") + i) for i, name in enumerate(nodes)}
    utilities = {
        objective.symbol: float(
            np.einsum(
                f"{''.join(letters.values())},{''.join(letters[parent] for parent in objective.parents)}->",
                joint,
                np.array(objective.utility.values),
            )
        )
        / probability
        for objective in objectives
    }
    return posteriors, utilities, probability


def _status_quo_tables(diagram: StationaryDiagram) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Every decision's network CPT as atemporal marginals and chain tables over its information states."""
    cardinality = {variable.name: len(variable.labels) for variable in diagram.network.variables}
    marginals, tables = {}, {}
    for decision in diagram.decisions:
        potential = diagram.network.potentials[decision.symbol]
        values = np.array(potential.values)
        information_set = decision.information_set
        rows = []
        for key in itertools.product(*(range(cardinality[name]) for name in information_set)):
            observed = dict(zip(information_set, key, strict=True))
            rows.append(values[(slice(None), *(observed[name] for name in potential.variables[1:]))])
        table = np.array(rows)
        if decision.symbol in diagram.structure.atemporal:
            marginals[decision.symbol] = table[0]
        else:
            tables[decision.symbol] = table
    return marginals, tables


def _assert_posteriors_match_joint_oracle(
    diagram: StationaryDiagram,
    marginals: dict[str, np.ndarray],
    tables: dict[str, np.ndarray],
    policies: dict[str, RandomisedPolicy],
    evidence: dict[str, int],
) -> None:
    """`stationary_step_posteriors` under `policies` matches `_joint_oracle` conditioned on `evidence`."""
    nodes, joint = _joint_oracle(diagram, flatten_stationary_step(diagram), marginals, tables, policies)
    expected_posteriors, expected_utilities, probability = _conditioned(nodes, joint, evidence, diagram.objectives)

    result = stationary_step_posteriors(diagram, policies, evidence)
    assert set(result.posteriors) == set(nodes)
    for node, posterior in expected_posteriors.items():
        np.testing.assert_allclose(result.posteriors[node], posterior, rtol=0, atol=1e-10)
    assert result.utilities == pytest.approx(expected_utilities, abs=1e-10)
    assert result.evidence_probability == pytest.approx(probability, abs=1e-10)


@pytest.mark.parametrize("name", ["chain_atemporal", "renewal", "renewal_chain_atemporal"])
@pytest.mark.parametrize("evidence", [{}, {"Dt": 1}, {"X0": 0, "Yt": 1}], ids=["none", "one", "two"])
def test_step_posteriors_match_joint_oracle(name: str, evidence: dict[str, int]) -> None:
    """Posteriors and conditional utilities under random randomised policies match the explicit step joint."""
    diagram = _SMALL_DIAGRAMS[name]()
    rng = np.random.default_rng(7)
    for _ in range(3):
        marginals, tables, policies = _random_policies(diagram, rng)
        _assert_posteriors_match_joint_oracle(diagram, marginals, tables, policies, evidence)


@pytest.mark.parametrize("name", ["chain_atemporal", "renewal_atemporal", "renewal_chain_atemporal"])
@pytest.mark.parametrize("g", [0, 1])
@pytest.mark.parametrize("evidence", ["none", "Dt", "G"])
def test_step_posteriors_one_hot_atemporal_match_joint_oracle(name: str, g: int, evidence: str) -> None:
    """A one-hot G, a step factor rather than a state node, matches the explicit step joint."""
    diagram = _SMALL_DIAGRAMS[name]()
    marginals, tables, policies = _random_policies(diagram, np.random.default_rng(7))
    one_hot = np.eye(2)[g]
    policies = {**policies, "G": policies["G"].model_copy(update={"probabilities": {0: one_hot.tolist()}})}
    conditions = {"none": {}, "Dt": {"Dt": 1}, "G": {"G": g}}[evidence]
    _assert_posteriors_match_joint_oracle(diagram, {**marginals, "G": one_hot}, tables, policies, conditions)


@pytest.mark.parametrize("name", ["chain", "chain_atemporal", "renewal", "renewal_chain_atemporal"])
@pytest.mark.parametrize("evidence", [{}, {"Dt": 1}, {"X0": 0, "Yt": 1}], ids=["none", "one", "two"])
def test_step_posteriors_status_quo_match_joint_oracle(name: str, evidence: dict[str, int]) -> None:
    """The status quo matches the explicit step joint with every decision at its network CPT."""
    diagram = _SMALL_DIAGRAMS[name]()
    marginals, tables = _status_quo_tables(diagram)
    _assert_posteriors_match_joint_oracle(diagram, marginals, tables, {}, evidence)


@pytest.mark.parametrize("name", ["mdp", "same_slice", "chain_atemporal", "renewal_atemporal"])
def test_step_posteriors_without_evidence_match_evaluation(name: str) -> None:
    """Without evidence the utilities are the objectives and the state posteriors the occupation's marginals."""
    diagram = _SMALL_DIAGRAMS[name]()
    rng = np.random.default_rng(3)
    for policies in [{}, _random_policies(diagram, rng)[2]]:
        evaluation = evaluate_stationary_policy(diagram, policies)
        result = stationary_step_posteriors(diagram, policies)
        assert result.utilities == pytest.approx(evaluation.objectives, abs=1e-12)
        assert result.evidence_probability == pytest.approx(1.0, abs=1e-12)
        occupation = np.array(evaluation.occupation.values)
        state_nodes = evaluation.occupation.variables
        for axis, node in enumerate(state_nodes):
            marginal = occupation.sum(axis=tuple(i for i in range(len(state_nodes)) if i != axis))
            np.testing.assert_allclose(result.posteriors[node], marginal, rtol=0, atol=1e-12)


def test_step_posteriors_point_mass_policy_on_mdp() -> None:
    """A deterministic Dt's posterior is the occupation-weighted share of its choices."""
    diagram = _mdp_diagram()
    policy = DecisionPolicy(symbol="Dt", information_set=["X0"], states=["lo", "mid", "hi"], policy={0: 2, 1: 0})
    x0 = evaluate_stationary_policy(diagram, {"Dt": policy}).occupation.values
    result = stationary_step_posteriors(diagram, {"Dt": policy})
    np.testing.assert_allclose(result.posteriors["Dt"], [x0[1], 0.0, x0[0]], rtol=0, atol=1e-12)
    conditioned = stationary_step_posteriors(diagram, {"Dt": policy}, {"X0": 1})
    np.testing.assert_allclose(conditioned.posteriors["Dt"], [1.0, 0.0, 0.0], rtol=0, atol=1e-12)
    assert conditioned.evidence_probability == pytest.approx(x0[1], abs=1e-12)


@pytest.mark.parametrize(
    ("evidence", "match"),
    [
        ({"Q": 0}, "aren't in the step joint"),
        ({"D0": 0}, "aren't in the step joint"),
        ({"Dt": 2}, "isn't an integer in range"),
        ({"Yt": -1}, "isn't an integer in range"),
        ({"Dt": 1.0}, "isn't an integer in range"),
        ({"Dt": True}, "isn't an integer in range"),
        ({"X0": 0, "Zt": 1}, "probability 0"),
    ],
)
def test_step_posteriors_invalid_evidence_raises(evidence: dict[str, int], match: str) -> None:
    """Unknown evidence nodes, non-integer or out-of-range states and impossible evidence raise."""
    with pytest.raises(ValueError, match=match):
        stationary_step_posteriors(_chain_diagram(), {}, evidence)


def test_step_posteriors_mixed_atemporal_draws_raise() -> None:
    """A randomised atemporal policy with initial_distribution raises, as its evaluation mixes draws."""
    diagram = _chain_diagram(atemporal=True)
    mu = initial_state_distribution(diagram)
    explicit = StationaryDiagram(
        **{
            **dict(diagram),
            "initial_distribution": Potential(
                variables=diagram.structure.state_nodes, shape=list(mu.shape), values=mu.tolist()
            ),
        }
    )
    _, _, policies = _random_policies(diagram, np.random.default_rng(4))
    with pytest.raises(NotImplementedError, match="initial_distribution"):
        stationary_step_posteriors(explicit, policies)


@pytest.mark.parametrize("name", list(_SMALL_DIAGRAMS))
def test_payoff_table_matches_brute_force(name: str) -> None:
    """The ideal and nadir match the payoff table of the best enumerated deterministic policy per objective."""
    diagram = _SMALL_DIAGRAMS[name]()
    flat = flatten_stationary_step(diagram)
    values = [
        evaluate_stationary_policy(diagram, policies).objectives
        for _, _, policies in _deterministic_combinations(diagram, flat)
    ]
    rows = {
        symbol: (max if maximize else min)(values, key=lambda value, symbol=symbol: value[symbol])
        for symbol, maximize in flat.maximize.items()
    }
    ideal, nadir = stationary_payoff_table(diagram)
    for symbol, maximize in flat.maximize.items():
        assert ideal[symbol] == pytest.approx(rows[symbol][symbol], abs=1e-10)
        worst = (min if maximize else max)(row[symbol] for row in rows.values())
        assert nadir[symbol] == pytest.approx(worst, abs=1e-10)
        assert (ideal[symbol] >= nadir[symbol]) if maximize else (ideal[symbol] <= nadir[symbol])


_CI_SOLVER_OPTIONS = {
    "cvxpy": {"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10},
    "gurobipy": {"FeasibilityTol": 1e-9, "OptimalityTol": 1e-9},
}
"""The tightest tolerances, without which the near-zero optima and their read-back policies miss VI."""


@pytest.fixture(scope="module")
def ci_diagram() -> tuple[StationaryDiagram, FlattenedStep]:
    """`ci_healthcare_lp_diagram` and its flattened step."""
    diagram = ci_healthcare_lp_diagram()
    return diagram, flatten_stationary_step(diagram)


@pytest.fixture(scope="module")
def ci_lp(ci_diagram: tuple[StationaryDiagram, FlattenedStep]) -> tuple[Problem, StationaryLPIndexMap]:
    """`ci_healthcare_lp_problem`, checked against compiling `ci_diagram`."""
    problem, index_map = ci_healthcare_lp_problem()
    expected_problem, expected_index_map = compile_stationary_lp(ci_diagram[0])
    assert index_map == expected_index_map
    assert [v.symbol for v in problem.variables] == [v.symbol for v in expected_problem.variables]
    assert [c.symbol for c in problem.constraints] == [c.symbol for c in expected_problem.constraints]
    assert [o.symbol for o in problem.objectives] == [o.symbol for o in expected_problem.objectives]
    return problem, index_map


@pytest.mark.slow
@pytest.mark.parametrize(
    "solver",
    [
        pytest.param("cvxpy", marks=[pytest.mark.cvxpy, pytest.mark.githubskip]),
        pytest.param("gurobipy", marks=[pytest.mark.gurobipy, pytest.mark.githubskip]),
    ],
)
@pytest.mark.parametrize("objective_symbol", ["cost", "dissatisfaction", "mortality", "illness"])
def test_ci_healthcare_lp_matches_value_iteration(
    ci_diagram: tuple[StationaryDiagram, FlattenedStep],
    ci_lp: tuple[Problem, StationaryLPIndexMap],
    objective_symbol: str,
    solver: str,
) -> None:
    """On ci-healthcare, the LP optimum equals VI and the read-back policies evaluate to it.

    The LP is too large for gurobipy's size-limited license. For near-zero optima, the LP-vs-VI
    check effectively runs at its 1e-9 absolute tolerance. The read-back value can miss the LP's
    because flow-constraint residuals at the solver's feasibility tolerance let the LP occupation
    differ from the read-back policies' exact one, by up to about residual times max utility
    (measured on cost: Gurobi 9.5e-9, HiGHS 1.4e-9). 2e-8 absolute is an empirical margin over these.
    """
    diagram, flat = ci_diagram
    problem, index_map = ci_lp
    results = _solve(problem, objective_symbol, solver, _CI_SOLVER_OPTIONS[solver])
    value = results.optimal_objectives[objective_symbol]
    assert value == pytest.approx(stationary_value_iteration(flat, objective_symbol)[0], rel=1e-6, abs=1e-9)

    policies = read_randomised_policy(index_map, results.optimal_variables)
    evaluation = evaluate_stationary_policy(diagram, policies).objectives[objective_symbol]
    assert evaluation == pytest.approx(value, rel=1e-6, abs=2e-8)


@pytest.mark.slow
@pytest.mark.githubskip
def test_ci_healthcare_lp_beats_ger_chosen_once(ci_diagram: tuple[StationaryDiagram, FlattenedStep]) -> None:
    """Each per-step optimum is no worse than that with OMA at its prior and GER chosen once."""
    diagram, flat = ci_diagram
    cpt_floor = inspect.signature(ci_healthcare_lp_diagram).parameters["cpt_floor"].default
    network = _load_normalised_network(_PURE_BIF_PATH, cpt_floor)
    provisional = StationaryDiagram.with_default_information_sets(
        network, {"GER": [], "LYHt": None}, diagram.objectives, discount=diagram.discount, renewal=diagram.renewal
    )
    reference = StationaryDiagram.with_default_information_sets(
        network,
        {"GER": [], "LYHt": full_observation_information_set(provisional, "LYHt")},
        diagram.objectives,
        discount=diagram.discount,
        renewal=diagram.renewal,
    )
    reference_flat = flatten_stationary_step(reference)
    for symbol, maximize in flat.maximize.items():
        sign = -1.0 if maximize else 1.0
        value = stationary_value_iteration(flat, symbol)[0]
        assert sign * value <= sign * stationary_value_iteration(reference_flat, symbol)[0] + 1e-9
