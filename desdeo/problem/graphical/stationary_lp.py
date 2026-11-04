"""Randomised stationary policies under full observation, on the flattened 2-slice step.

The one slice-t decision D observes every state node, every atemporal decision and every slice-t
ancestor of D. With s the joint state and o the joint slice-t ancestors, the step flattens into a
kernel `T[s', (s, o, a)] = P(o, s' | s, do(a))` and objective coefficients
`W[(s, o, a)] = Σ P(o, rest | s, do(a)) u`. Each joint value of the atemporal decisions is a branch
with its own kernel and coefficients. Over the discounted occupation `w(s, o, a) = x(s) π(a | s, o)`
the step is linear, and `stationary_value_iteration` solves it for one objective.
"""

import itertools
from collections.abc import Hashable
from math import prod

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pydantic import BaseModel, ConfigDict

from .dynamic import (
    Factor,
    RandomisedPolicy,
    StationaryDiagram,
    _ancestors,
    _contract,
    _network_factors,
    initial_state_distribution,
)
from .schema import DecisionNode


class FlattenedBranch(BaseModel):
    """The flattened step for one joint value of the atemporal decisions."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    atemporal: dict[str, int]
    """Atemporal decision symbol -> its state index in this branch."""
    pairs: np.ndarray
    """The kept flat (s, o) indices `s * O + o`, ascending: those with `P(o | s) > 0`. The `k`-th kept
    pair and action `a` are the flat (s, o, a) column `k * A + a`."""
    kernel: sp.csr_array
    """`P(o, s' | s, do(a))`, rows the next state s', columns the flat (s, o, a) of the kept pairs."""
    objective_coefficients: dict[str, np.ndarray]
    """Objective symbol -> `Σ P(o, rest | s, do(a)) u` per flat (s, o, a) of the kept pairs, unsigned."""


class FlattenedStep(BaseModel):
    """The 2-slice step of a fully observed stationary diagram, flattened over (s, o, a).

    s is mixed-radix over `state_nodes`, o over `observation_nodes`, and a is the decision's state
    index, varying fastest.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    state_nodes: list[str]
    """`TwoSliceStructure.state_nodes`."""
    state_cardinalities: list[int]
    """The state nodes' numbers of states."""
    observation_nodes: list[str]
    """The decision's slice-t ancestors, in network declaration order."""
    observation_cardinalities: list[int]
    """The observation nodes' numbers of states."""
    decision: DecisionNode
    """The slice-t decision."""
    atemporal_decisions: list[DecisionNode]
    """The atemporal decisions, each chosen once without observations."""
    branches: list[FlattenedBranch]
    """One per joint value of `atemporal_decisions`, the last decision varying fastest."""
    mu0: np.ndarray
    """The slice-0 joint over `state_nodes`, flat."""
    discount: float
    """`StationaryDiagram.discount`."""
    maximize: dict[str, bool]
    """Objective symbol -> whether it is maximised."""


def _slice_t_decision(diagram: StationaryDiagram) -> DecisionNode:
    """The diagram's one slice-t decision; raise if there isn't exactly one."""
    slicet = diagram.structure.slicet
    decisions = [decision for decision in diagram.decisions if decision.symbol in slicet]
    if len(decisions) != 1:
        msg = f"The LP path needs exactly one slice-t decision, got {[decision.symbol for decision in decisions]}."
        raise ValueError(msg)
    return decisions[0]


def full_observation_information_set(diagram: StationaryDiagram, decision_symbol: str) -> list[str]:
    """The state nodes, the atemporal decisions and the slice-t ancestors of a slice-t decision.

    Ancestors follow the network's CPT parents. The state nodes come from `diagram.structure`, so a
    slice-0 node only the decision observes is included only if the decision already observes it.
    Nodes are in network declaration order.

    Raises:
        ValueError: `decision_symbol` isn't one of the diagram's slice-t decisions.
    """
    structure = diagram.structure
    decisions = {decision.symbol for decision in diagram.decisions}
    if decision_symbol not in decisions & structure.slicet:
        msg = f"'{decision_symbol}' isn't a slice-t decision of the diagram."
        raise ValueError(msg)
    parents = {name: potential.variables[1:] for name, potential in diagram.network.potentials.items()}
    observed = set(structure.state_nodes) | (decisions & structure.atemporal)
    observed |= _ancestors(parents[decision_symbol], parents, structure.slicet)
    return [variable.name for variable in diagram.network.variables if variable.name in observed]


def _flat_indices(axis_indices: list[np.ndarray], cardinalities: list[int], size: int) -> np.ndarray:
    """Mixed-radix flat index, most significant first, of each entry's per-axis indices."""
    if not axis_indices:
        return np.zeros(size, dtype=np.int64)
    return np.ravel_multi_index(axis_indices, cardinalities)


def flatten_stationary_step(diagram: StationaryDiagram) -> FlattenedStep:
    """Flatten `diagram`'s 2-slice step over (state, observation, action), one branch per atemporal choice.

    The decision's own CPT is replaced by an all-ones factor on its axis, and each atemporal
    decision's by a point mass on the branch's value. Pairs (s, o) with `P(o | s) == 0` exactly are
    dropped, together with their columns.

    Raises:
        ValueError: the diagram has constraints, doesn't have exactly one slice-t decision, that
            decision's information set isn't `full_observation_information_set`, or an atemporal
            decision observes anything.
    """
    if diagram.constraints:
        msg = "The LP path doesn't support diagram constraints."
        raise ValueError(msg)
    structure = diagram.structure
    decision = _slice_t_decision(diagram)
    atemporal_decisions = [d for d in diagram.decisions if d.symbol in structure.atemporal]
    for atemporal in atemporal_decisions:
        if atemporal.information_set:
            msg = (
                f"Atemporal decision '{atemporal.symbol}' observes {atemporal.information_set}; on the LP path "
                "atemporal decisions are chosen once without observations."
            )
            raise ValueError(msg)
    full = full_observation_information_set(diagram, decision.symbol)
    if sorted(decision.information_set) != sorted(full):
        msg = (
            f"Decision '{decision.symbol}' observes {decision.information_set}, but the LP path needs its full "
            f"observation set {full}."
        )
        raise ValueError(msg)

    cardinality = {variable.name: len(variable.labels) for variable in diagram.network.variables}
    state_nodes = structure.state_nodes
    observation_nodes = [name for name in full if name in structure.slicet]
    state_cards = [cardinality[name] for name in state_nodes]
    observation_cards = [cardinality[name] for name in observation_nodes]
    n_states, n_observations = prod(state_cards), prod(observation_cards)
    n_actions = len(decision.states)
    symbol = decision.symbol

    factors = _network_factors(diagram.network)
    del factors[symbol]
    parents = {name: axes[1:] for name, (axes, _) in factors.items()}
    parents[symbol] = decision.information_set
    for atemporal in atemporal_decisions:
        parents[atemporal.symbol] = []
    step = structure.slicet | {atemporal.symbol for atemporal in atemporal_decisions}

    # An atemporal state node keeps its value; its next-step axis is tied to it by an identity.
    next_axes: list[Hashable] = []
    identities: list[Factor] = []
    for name in state_nodes:
        if name in structure.atemporal:
            next_axes.append((name, "next"))
            identities.append(([name, (name, "next")], np.eye(cardinality[name])))
        else:
            next_axes.append(structure.next_of[name])
    # All-ones factors keep the state and decision axes no remaining CPT mentions.
    ones = [([name], np.ones(cardinality[name])) for name in [*state_nodes, symbol]]
    kernel_axes = [*state_nodes, *dict.fromkeys([*observation_nodes, symbol, *next_axes])]
    position = {axis: i for i, axis in enumerate(kernel_axes)}

    next_slicet = [axis for axis in next_axes if axis in structure.slicet]
    transition_names = sorted(_ancestors([*next_slicet, *observation_nodes], parents, step) - {symbol})
    observation_names = sorted(_ancestors(observation_nodes, parents, step))
    objective_names = {
        objective.symbol: sorted(_ancestors([*objective.parents, *observation_nodes], parents, step) - {symbol})
        for objective in diagram.objectives
    }

    branches = []
    for values in itertools.product(*(range(len(atemporal.states)) for atemporal in atemporal_decisions)):
        branch_factors = dict(factors)
        for atemporal, value in zip(atemporal_decisions, values, strict=True):
            branch_factors[atemporal.symbol] = ([atemporal.symbol], np.eye(len(atemporal.states))[value])
        observed = _contract(
            [branch_factors[name] for name in observation_names] + ones[:-1], [*state_nodes, *observation_nodes]
        )
        pairs = np.flatnonzero(observed.reshape(-1))
        column_of = np.full(n_states * n_observations, -1, dtype=np.int64)
        column_of[pairs] = np.arange(len(pairs))

        transition = _contract([branch_factors[name] for name in transition_names] + identities + ones, kernel_axes)
        entries = np.nonzero(transition)
        size = len(entries[0])
        s = _flat_indices([entries[position[name]] for name in state_nodes], state_cards, size)
        o = _flat_indices([entries[position[name]] for name in observation_nodes], observation_cards, size)
        s_next = _flat_indices([entries[position[axis]] for axis in next_axes], state_cards, size)
        pair_column = column_of[s * n_observations + o]
        kept = pair_column >= 0
        kernel = sp.csr_array(
            (
                transition[entries][kept],
                (s_next[kept], pair_column[kept] * n_actions + entries[position[symbol]][kept]),
            ),
            shape=(n_states, len(pairs) * n_actions),
        )

        coefficients = {}
        for objective in diagram.objectives:
            utility = (objective.parents, np.array(objective.utility.values, dtype=float))
            table = _contract(
                [branch_factors[name] for name in objective_names[objective.symbol]] + ones + [utility],
                [*state_nodes, *observation_nodes, symbol],
            )
            coefficients[objective.symbol] = table.reshape(-1, n_actions)[pairs].reshape(-1)

        branches.append(
            FlattenedBranch(
                atemporal={
                    atemporal.symbol: value for atemporal, value in zip(atemporal_decisions, values, strict=True)
                },
                pairs=pairs,
                kernel=kernel,
                objective_coefficients=coefficients,
            )
        )

    return FlattenedStep(
        state_nodes=state_nodes,
        state_cardinalities=state_cards,
        observation_nodes=observation_nodes,
        observation_cardinalities=observation_cards,
        decision=decision,
        atemporal_decisions=atemporal_decisions,
        branches=branches,
        mu0=initial_state_distribution(diagram).reshape(-1),
        discount=diagram.discount,
        maximize={objective.symbol: objective.maximize for objective in diagram.objectives},
    )


def _information_keys(flat: FlattenedStep, branch: FlattenedBranch) -> np.ndarray:
    """The decision's information-state flat index for each flat (s, o) of `branch`, keyed like `RandomisedPolicy`."""
    nodes = [*flat.state_nodes, *flat.observation_nodes]
    cards = [*flat.state_cardinalities, *flat.observation_cardinalities]
    size = prod(cards)
    # The flat (s, o) index is mixed-radix over the state nodes followed by the observation nodes.
    values = dict(zip(nodes, np.unravel_index(np.arange(size), cards) if cards else (), strict=True))
    values |= {symbol: np.full(size, value) for symbol, value in branch.atemporal.items()}
    cardinality = dict(zip(nodes, cards, strict=True))
    cardinality |= {atemporal.symbol: len(atemporal.states) for atemporal in flat.atemporal_decisions}
    information_set = flat.decision.information_set
    return _flat_indices(
        [values[name] for name in information_set], [cardinality[name] for name in information_set], size
    )


def stationary_value_iteration(
    flat: FlattenedStep, objective_symbol: str, *, tol: float = 1e-12, max_iterations: int = 1000
) -> tuple[float, dict[str, RandomisedPolicy]]:
    """The full-observation optimum of one objective, by policy iteration warm-started by value iteration.

    On each branch, 100 value-iteration sweeps of `V(s) = Σ_o min_a [W + gamma T V]`, with W
    negated for a maximised objective, give a starting greedy policy. Policy iteration then
    evaluates the greedy policy exactly by a sparse linear solve and re-greedies on that V,
    switching an action only where it lowers Q by more than
    `max((1 - gamma) * tol * max(1, ‖V‖∞), 64 * eps * ‖Q‖∞)`, until no action switches. The
    atemporal decisions take the branch with the best exact value.

    Args:
        flat: the flattened step.
        objective_symbol: the objective to optimise.
        tol: the relative policy-iteration switching tolerance, as above. Must be finite and positive.
        max_iterations: the most policy-iteration steps per branch. Must be a positive integer.

    Returns:
        tuple[float, dict[str, RandomisedPolicy]]: the exact per-step average `(1 - gamma) mu0 V` of
            the returned policy, in the objective's own sign, and a point-mass policy for every
            decision. Each branch's information states get that branch's greedy action; dropped
            (s, o) pairs get action 0.

    Raises:
        ValueError: `objective_symbol` isn't an objective of `flat`, `tol` isn't finite and positive,
            `max_iterations` isn't a positive integer, or the objective's coefficients aren't finite.
        RuntimeError: policy iteration doesn't converge within `max_iterations` steps.
    """
    if objective_symbol not in flat.maximize:
        msg = f"Unknown objective '{objective_symbol}'; expected one of {sorted(flat.maximize)}."
        raise ValueError(msg)
    if not (np.isfinite(tol) and tol > 0):
        msg = f"tol must be finite and positive, got {tol}."
        raise ValueError(msg)
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        msg = f"max_iterations must be a positive integer, got {max_iterations}."
        raise ValueError(msg)
    if not all(np.isfinite(branch.objective_coefficients[objective_symbol]).all() for branch in flat.branches):
        msg = f"Objective '{objective_symbol}' has non-finite coefficients."
        raise ValueError(msg)
    sign = -1.0 if flat.maximize[objective_symbol] else 1.0
    gamma = flat.discount
    n_states = prod(flat.state_cardinalities)
    n_observations = prod(flat.observation_cardinalities)
    n_actions = len(flat.decision.states)
    n_information = n_states * n_observations * prod(len(atemporal.states) for atemporal in flat.atemporal_decisions)
    probabilities = np.zeros((n_information, n_actions))
    eps = np.finfo(float).eps

    best_value, best_branch = np.inf, flat.branches[0]
    for branch in flat.branches:
        owner = branch.pairs // n_observations
        backward = branch.kernel.T.tocsr()
        reward = sign * branch.objective_coefficients[objective_symbol].reshape(-1, n_actions)
        # A fixed number of value-iteration sweeps gives the starting policy.
        value = np.zeros(n_states)
        for _ in range(100):
            q = reward + gamma * (backward @ value).reshape(-1, n_actions)
            value = np.bincount(owner, weights=q.min(axis=1), minlength=n_states)

        # Policy iteration from the greedy policy, with each policy's value from a linear solve.
        q = reward + gamma * (backward @ value).reshape(-1, n_actions)
        greedy = q.argmin(axis=1)
        rows = np.arange(len(owner))
        owner_matrix = sp.csr_array((np.ones(len(owner)), (rows, owner)), shape=(len(owner), n_states))
        for _ in range(max_iterations):
            closed_loop = branch.kernel[:, rows * n_actions + greedy] @ owner_matrix
            step_reward = np.bincount(owner, weights=reward[rows, greedy], minlength=n_states)
            value = spla.spsolve((sp.eye_array(n_states) - gamma * closed_loop.T).tocsc(), step_reward)
            q = reward + gamma * (backward @ value).reshape(-1, n_actions)
            # The floor keeps rounding noise in Q from switching actions near gamma = 1.
            threshold = max((1 - gamma) * tol * max(1.0, np.max(np.abs(value))), 64 * eps * np.max(np.abs(q)))
            switch = q[rows, greedy] - q.min(axis=1) > threshold
            if not switch.any():
                break
            greedy = np.where(switch, q.argmin(axis=1), greedy)
        else:
            msg = f"Policy iteration didn't converge within {max_iterations} iterations."
            raise RuntimeError(msg)

        choices = np.zeros(n_states * n_observations, dtype=np.int64)
        choices[branch.pairs] = greedy
        probabilities[_information_keys(flat, branch), choices] = 1.0

        branch_value = (1 - gamma) * flat.mu0 @ value
        if branch_value < best_value:
            best_value, best_branch = branch_value, branch

    policies = {
        flat.decision.symbol: RandomisedPolicy(
            symbol=flat.decision.symbol,
            information_set=flat.decision.information_set,
            states=flat.decision.states,
            probabilities=dict(enumerate(probabilities.tolist())),
        )
    }
    for atemporal in flat.atemporal_decisions:
        point_mass = np.eye(len(atemporal.states))[best_branch.atemporal[atemporal.symbol]]
        policies[atemporal.symbol] = RandomisedPolicy(
            symbol=atemporal.symbol, information_set=[], states=atemporal.states, probabilities={0: point_mass.tolist()}
        )
    return float(sign * best_value), policies
