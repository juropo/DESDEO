"""Randomised stationary policies under full observation, on the flattened 2-slice step.

The slice-t decisions D_1, ..., D_n form a chain, each an ancestor of the next. D_n observes every
state node, every atemporal decision and every slice-t ancestor of D_n, and D_k observes what
D_{k+1} observes except D_k and its descendants. With s the joint state, o_k the slice-t nodes D_k
observes and D_{k-1} doesn't, and a_k D_k's action, the step flattens into a kernel
`T[s', (s, o_1, a_1, ..., o_n, a_n)] = P(o_1..o_n, s' | s, do(a_1..a_n))` and objective
coefficients `W[(s, o_1, a_1, ..., o_n, a_n)] = Σ P(o_1..o_n, rest | s, do(a_1..a_n)) u`. Each joint
value of the atemporal decisions is a branch with its own kernel and coefficients. Over the
discounted occupations `y_0 = x(s)` and `y_k = y_{k-1} π_k(a_k | s, o_1, a_1, ..., o_k)` the step is
linear, and `stationary_value_iteration` solves it for one objective.
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
    prefixes: list[np.ndarray]
    """Per level k, the kept prefixes p_k as flat indices into the mixed-radix space over
    (s, o_1, a_1, ..., o_k), ascending: those whose newest observation has positive probability and
    whose parent is kept. The `i`-th kept p_k and action a_k are entry `i * A_k + a_k` of y_k."""
    parents: list[np.ndarray]
    """Per level k, each kept p_k's entry in y_{k-1}: its state s for level 1, otherwise
    `j * A_{k-1} + a_{k-1}` with j its parent's position among level k-1's kept prefixes."""
    kernel: sp.csr_array
    """`P(o_1..o_n, s' | s, do(a_1..a_n))`, rows the next state s', columns the entries of y_n."""
    objective_coefficients: dict[str, np.ndarray]
    """Objective symbol -> `Σ P(o_1..o_n, rest | s, do(a_1..a_n)) u` per entry of y_n, unsigned."""


class FlattenedStep(BaseModel):
    """The 2-slice step of a fully observed stationary diagram, flattened over its decision chain.

    The flat order is (s, o_1, a_1, ..., o_n, a_n), mixed-radix with the last axis varying fastest:
    s over `state_nodes`, o_k over `observation_nodes[k]` and a_k over `decisions[k].states`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    state_nodes: list[str]
    """`TwoSliceStructure.state_nodes`."""
    state_cardinalities: list[int]
    """The state nodes' numbers of states."""
    decisions: list[DecisionNode]
    """The slice-t decisions, each an ancestor of the next."""
    observation_nodes: list[list[str]]
    """Per decision, the slice-t chance nodes it observes and the previous decision doesn't, in
    network declaration order."""
    observation_cardinalities: list[list[int]]
    """The observation nodes' numbers of states."""
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


def _cpt_parents(diagram: StationaryDiagram) -> dict[str, list[str]]:
    """Node -> its parents in the network's CPTs."""
    return {name: potential.variables[1:] for name, potential in diagram.network.potentials.items()}


def _decision_chain(diagram: StationaryDiagram) -> list[DecisionNode]:
    """The slice-t decisions, each an ancestor of the next under the network's CPT parents.

    Raises:
        ValueError: there are no slice-t decisions, or they don't form a chain.
    """
    slicet = diagram.structure.slicet
    chain = [decision for decision in diagram.decisions if decision.symbol in slicet]
    if not chain:
        msg = "The LP path needs at least one slice-t decision."
        raise ValueError(msg)
    parents = _cpt_parents(diagram)
    ancestors = {decision.symbol: _ancestors(parents[decision.symbol], parents, slicet) for decision in chain}
    chain.sort(key=lambda decision: len(ancestors[decision.symbol]))
    for earlier, later in itertools.pairwise(chain):
        if earlier.symbol not in ancestors[later.symbol]:
            msg = (
                f"The LP path needs the slice-t decisions to form a chain, each an ancestor of the next "
                f"through the network's CPT parents, not the information sets; '{earlier.symbol}' isn't an "
                f"ancestor of '{later.symbol}'."
            )
            raise ValueError(msg)
    return chain


def full_observation_information_set(diagram: StationaryDiagram, decision_symbol: str) -> list[str]:
    """A decision's full-observation information set, in network declaration order.

    An atemporal decision observes nothing. The last slice-t decision of the chain observes the
    state nodes, the atemporal decisions and its slice-t ancestors; an earlier one observes those
    except itself and its descendants. Ancestors and descendants follow the network's CPT parents.
    The state nodes come from `diagram.structure`, so a slice-0 node only a decision observes is
    included only if that decision already observes it.

    Raises:
        ValueError: `decision_symbol` isn't one of the diagram's decisions, or the slice-t
            decisions don't form a chain.
    """
    structure = diagram.structure
    decisions = {decision.symbol for decision in diagram.decisions}
    if decision_symbol in decisions & structure.atemporal:
        return []
    if decision_symbol not in decisions & structure.slicet:
        msg = f"'{decision_symbol}' isn't a decision of the diagram."
        raise ValueError(msg)
    last = _decision_chain(diagram)[-1].symbol
    parents = _cpt_parents(diagram)
    children: dict[str, list[str]] = {name: [] for name in parents}
    for name, axes in parents.items():
        for parent in axes:
            children[parent].append(name)
    observed = set(structure.state_nodes) | (decisions & structure.atemporal)
    observed |= _ancestors(parents[last], parents, structure.slicet)
    observed -= _ancestors([decision_symbol], children, structure.slicet)
    return [variable.name for variable in diagram.network.variables if variable.name in observed]


def _flat_indices(axis_indices: list[np.ndarray], cardinalities: list[int], size: int) -> np.ndarray:
    """Mixed-radix flat index, most significant first, of each entry's per-axis indices."""
    if not axis_indices:
        return np.zeros(size, dtype=np.int64)
    return np.ravel_multi_index(axis_indices, cardinalities)


def _prefix_axes(flat: FlattenedStep, level: int) -> tuple[list[str], list[int]]:
    """The names and cardinalities of the prefix axes (s, o_1, a_1, ..., o_k) of chain decision `level`, from 0."""
    names, cards = [*flat.state_nodes], [*flat.state_cardinalities]
    for k in range(level):
        names += [*flat.observation_nodes[k], flat.decisions[k].symbol]
        cards += [*flat.observation_cardinalities[k], len(flat.decisions[k].states)]
    return [*names, *flat.observation_nodes[level]], [*cards, *flat.observation_cardinalities[level]]


def _kept_prefixes(
    tables: list[np.ndarray], observation_counts: list[int], n_actions: list[int]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per level, the kept prefixes and their entries in y_{k-1}, from each level's prefix probabilities."""
    prefixes: list[np.ndarray] = []
    parents: list[np.ndarray] = []
    row_of = np.zeros(0, dtype=np.int64)
    for k, table in enumerate(tables):
        kept = np.flatnonzero(table.reshape(-1))
        above = kept // observation_counts[k]
        if k == 0:
            parent = above
        else:
            previous, action = np.divmod(above, n_actions[k - 1])
            reached = row_of[previous] >= 0
            kept, parent = kept[reached], row_of[previous[reached]] * n_actions[k - 1] + action[reached]
        row_of = np.full(table.size, -1, dtype=np.int64)
        row_of[kept] = np.arange(len(kept))
        prefixes.append(kept)
        parents.append(parent)
    return prefixes, parents


def flatten_stationary_step(diagram: StationaryDiagram) -> FlattenedStep:
    """Flatten `diagram`'s 2-slice step over its decision chain, one branch per atemporal choice.

    Each chain decision's CPT is replaced by an all-ones factor on its axis, and each atemporal
    decision's by a point mass on the branch's value. A prefix p_k whose newest observation has
    probability exactly 0 given its parent is dropped, together with everything below it.

    Raises:
        ValueError: the diagram has constraints, has no slice-t decisions, its slice-t decisions
            don't form a chain, one of them doesn't observe exactly its
            `full_observation_information_set`, or an atemporal decision observes anything.
    """
    if diagram.constraints:
        msg = "The LP path doesn't support diagram constraints."
        raise ValueError(msg)
    structure = diagram.structure
    chain = _decision_chain(diagram)
    atemporal_decisions = [d for d in diagram.decisions if d.symbol in structure.atemporal]
    for atemporal in atemporal_decisions:
        if atemporal.information_set:
            msg = (
                f"Atemporal decision '{atemporal.symbol}' observes {atemporal.information_set}; on the LP path "
                "atemporal decisions are chosen once without observations."
            )
            raise ValueError(msg)
    chain_symbols = [decision.symbol for decision in chain]
    cardinality = {variable.name: len(variable.labels) for variable in diagram.network.variables}
    observation_nodes: list[list[str]] = []
    observed: set[str] = set()
    for decision in chain:
        full = full_observation_information_set(diagram, decision.symbol)
        if sorted(decision.information_set) != sorted(full):
            msg = (
                f"Decision '{decision.symbol}' observes {decision.information_set}, but the LP path needs its full "
                f"observation set {full}."
            )
            raise ValueError(msg)
        newest = [name for name in full if name in structure.slicet and name not in {*chain_symbols, *observed}]
        observation_nodes.append(newest)
        observed.update(newest)

    state_nodes = structure.state_nodes
    state_cards = [cardinality[name] for name in state_nodes]
    observation_cards = [[cardinality[name] for name in nodes] for nodes in observation_nodes]
    n_states = prod(state_cards)
    n_actions = [len(decision.states) for decision in chain]
    level_axes = []
    flat_axes = [*state_nodes]
    for nodes, symbol in zip(observation_nodes, chain_symbols, strict=True):
        level_axes.append([*flat_axes, *nodes])
        flat_axes += [*nodes, symbol]
    flat_cards = [cardinality[name] for name in flat_axes]

    factors = _network_factors(diagram.network)
    for decision in chain:
        del factors[decision.symbol]
    parents = {name: axes[1:] for name, (axes, _) in factors.items()}
    for decision in chain:
        parents[decision.symbol] = decision.information_set
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
    ones = {name: ([name], np.ones(cardinality[name])) for name in [*state_nodes, *chain_symbols]}
    kernel_axes = [*state_nodes, *dict.fromkeys([*flat_axes[len(state_nodes) :], *next_axes])]
    position = {axis: i for i, axis in enumerate(kernel_axes)}

    all_observations = [name for nodes in observation_nodes for name in nodes]
    next_slicet = [axis for axis in next_axes if axis in structure.slicet]
    transition_names = sorted(_ancestors([*next_slicet, *all_observations], parents, step) - set(chain_symbols))
    prefix_names = [
        sorted(
            _ancestors([name for nodes in observation_nodes[: k + 1] for name in nodes], parents, step)
            - set(chain_symbols)
        )
        for k in range(len(chain))
    ]
    objective_names = {
        objective.symbol: sorted(
            _ancestors([*objective.parents, *all_observations], parents, step) - set(chain_symbols)
        )
        for objective in diagram.objectives
    }

    branches = []
    for values in itertools.product(*(range(len(atemporal.states)) for atemporal in atemporal_decisions)):
        branch_factors = dict(factors)
        for atemporal, value in zip(atemporal_decisions, values, strict=True):
            branch_factors[atemporal.symbol] = ([atemporal.symbol], np.eye(len(atemporal.states))[value])

        tables = [
            _contract(
                [branch_factors[name] for name in prefix_names[k]]
                + [ones[name] for name in [*state_nodes, *chain_symbols[:k]]],
                axes,
            )
            for k, axes in enumerate(level_axes)
        ]
        prefixes, prefix_parents = _kept_prefixes(tables, [prod(cards) for cards in observation_cards], n_actions)
        row_of = np.full(tables[-1].size, -1, dtype=np.int64)
        row_of[prefixes[-1]] = np.arange(len(prefixes[-1]))

        transition = _contract(
            [branch_factors[name] for name in transition_names] + identities + list(ones.values()), kernel_axes
        )
        entries = np.nonzero(transition)
        size = len(entries[0])
        flat_index = _flat_indices([entries[position[name]] for name in flat_axes], flat_cards, size)
        s_next = _flat_indices([entries[position[axis]] for axis in next_axes], state_cards, size)
        prefix, action = np.divmod(flat_index, n_actions[-1])
        row = row_of[prefix]
        kept = row >= 0
        kernel = sp.csr_array(
            (transition[entries][kept], (s_next[kept], row[kept] * n_actions[-1] + action[kept])),
            shape=(n_states, len(prefixes[-1]) * n_actions[-1]),
        )

        coefficients = {}
        for objective in diagram.objectives:
            utility = (objective.parents, np.array(objective.utility.values, dtype=float))
            table = _contract(
                [branch_factors[name] for name in objective_names[objective.symbol]] + list(ones.values()) + [utility],
                flat_axes,
            )
            coefficients[objective.symbol] = table.reshape(-1, n_actions[-1])[prefixes[-1]].reshape(-1)

        branches.append(
            FlattenedBranch(
                atemporal={
                    atemporal.symbol: value for atemporal, value in zip(atemporal_decisions, values, strict=True)
                },
                prefixes=prefixes,
                parents=prefix_parents,
                kernel=kernel,
                objective_coefficients=coefficients,
            )
        )

    return FlattenedStep(
        state_nodes=state_nodes,
        state_cardinalities=state_cards,
        decisions=chain,
        observation_nodes=observation_nodes,
        observation_cardinalities=observation_cards,
        atemporal_decisions=atemporal_decisions,
        branches=branches,
        mu0=initial_state_distribution(diagram).reshape(-1),
        discount=diagram.discount,
        maximize={objective.symbol: objective.maximize for objective in diagram.objectives},
    )


def _information_keys(flat: FlattenedStep, branch: FlattenedBranch, level: int) -> np.ndarray:
    """The information-state key of chain decision `level` for each of `branch`'s kept prefixes at that level.

    Keys are as in that decision's `RandomisedPolicy`, with the branch's atemporal values.
    """
    names, cards = _prefix_axes(flat, level)
    prefixes = branch.prefixes[level]
    values = dict(zip(names, np.unravel_index(prefixes, cards) if cards else (), strict=True))
    values |= {symbol: np.full(len(prefixes), value) for symbol, value in branch.atemporal.items()}
    cardinality = dict(zip(names, cards, strict=True))
    cardinality |= {atemporal.symbol: len(atemporal.states) for atemporal in flat.atemporal_decisions}
    information_set = flat.decisions[level].information_set
    return _flat_indices(
        [values[name] for name in information_set], [cardinality[name] for name in information_set], len(prefixes)
    )


def _nested_backup(
    branch: FlattenedBranch, n_actions: list[int], n_states: int, q_last: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray]:
    """Per-level Q tables, from the last level's, minimising over a_k and summing over o_k upwards, and V(s)."""
    qs = [q_last]
    for k in range(len(n_actions) - 1, 0, -1):
        totals = np.bincount(
            branch.parents[k], weights=qs[0].min(axis=1), minlength=len(branch.prefixes[k - 1]) * n_actions[k - 1]
        )
        qs.insert(0, totals.reshape(-1, n_actions[k - 1]))
    return qs, np.bincount(branch.parents[0], weights=qs[0].min(axis=1), minlength=n_states)


def _reached_entries(
    branch: FlattenedBranch, n_actions: list[int], greedy: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """The entries of y_n a deterministic chain policy reaches, and the state s each one starts from."""
    owner = branch.parents[0]
    active = np.arange(len(owner))
    for k in range(len(n_actions) - 1):
        entry_owner = np.full(len(branch.prefixes[k]) * n_actions[k], -1, dtype=np.int64)
        entry_owner[active * n_actions[k] + greedy[k][active]] = owner
        child_owner = entry_owner[branch.parents[k + 1]]
        active = np.flatnonzero(child_owner >= 0)
        owner = child_owner[active]
    return active * n_actions[-1] + greedy[-1][active], owner


def stationary_value_iteration(
    flat: FlattenedStep, objective_symbol: str, *, tol: float = 1e-12, max_iterations: int = 1000
) -> tuple[float, dict[str, RandomisedPolicy]]:
    """The full-observation optimum of one objective, by policy iteration warm-started by value iteration.

    On each branch, 100 value-iteration sweeps of the nested backup `V(s) = Σ_{o_1} min_{a_1} ...
    Σ_{o_n} min_{a_n} [W + gamma T V]`, with W negated for a maximised objective, give a starting
    greedy policy per chain decision. Policy iteration then evaluates the greedy policies exactly by
    a sparse linear solve and re-greedies on that V, from the last level up: Q_k sums level k+1's Q
    at its chosen actions, and an action of level k switches only where that lowers Q_k by more than
    `max((1 - gamma) * tol * max(1, ‖V‖∞), 64 * eps * ‖Q_k‖∞)`. It stops when no action switches.
    The atemporal decisions take the branch with the best exact value.

    Args:
        flat: the flattened step.
        objective_symbol: the objective to optimise.
        tol: the relative policy-iteration switching tolerance, as above. Must be finite and positive.
        max_iterations: the most policy-iteration steps per branch. Must be a positive integer.

    Returns:
        tuple[float, dict[str, RandomisedPolicy]]: the exact per-step average `(1 - gamma) mu0 V` of
            the returned policies, in the objective's own sign, and a point-mass policy for every
            decision. Each branch's information states get that branch's greedy action; those of
            dropped prefixes get action 0.

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
    n_actions = [len(decision.states) for decision in flat.decisions]
    n_branches = prod(len(atemporal.states) for atemporal in flat.atemporal_decisions)
    probabilities = []
    for level, actions in enumerate(n_actions):
        table = np.zeros((prod(_prefix_axes(flat, level)[1]) * n_branches, actions))
        table[:, 0] = 1.0
        probabilities.append(table)
    eps = np.finfo(float).eps

    best_value, best_branch = np.inf, flat.branches[0]
    for branch in flat.branches:
        backward = branch.kernel.T.tocsr()
        reward = sign * branch.objective_coefficients[objective_symbol]
        # A fixed number of value-iteration sweeps gives the starting policy.
        value = np.zeros(n_states)
        for _ in range(100):
            q_last = (reward + gamma * (backward @ value)).reshape(-1, n_actions[-1])
            _, value = _nested_backup(branch, n_actions, n_states, q_last)

        # Policy iteration from the greedy policy, with each policy's value from a linear solve.
        q_last = (reward + gamma * (backward @ value)).reshape(-1, n_actions[-1])
        qs, _ = _nested_backup(branch, n_actions, n_states, q_last)
        greedy = [q.argmin(axis=1) for q in qs]
        for _ in range(max_iterations):
            columns, owner = _reached_entries(branch, n_actions, greedy)
            owner_matrix = sp.csr_array(
                (np.ones(len(owner)), (np.arange(len(owner)), owner)), shape=(len(owner), n_states)
            )
            closed_loop = branch.kernel[:, columns] @ owner_matrix
            step_reward = np.bincount(owner, weights=reward[columns], minlength=n_states)
            value = spla.spsolve((sp.eye_array(n_states) - gamma * closed_loop.T).tocsc(), step_reward)
            # Switch bottom-up: Q_k sums the realised Q_{k+1} at level k+1's chosen actions.
            q = (reward + gamma * (backward @ value)).reshape(-1, n_actions[-1])
            switched = False
            for k in range(len(n_actions) - 1, -1, -1):
                # The floor keeps rounding noise in Q from switching actions near gamma = 1.
                threshold = max((1 - gamma) * tol * max(1.0, np.max(np.abs(value))), 64 * eps * np.max(np.abs(q)))
                switch = q[np.arange(len(q)), greedy[k]] - q.min(axis=1) > threshold
                greedy[k] = np.where(switch, q.argmin(axis=1), greedy[k])
                switched = switched or bool(switch.any())
                if k > 0:
                    q = np.bincount(
                        branch.parents[k],
                        weights=q[np.arange(len(q)), greedy[k]],
                        minlength=len(branch.prefixes[k - 1]) * n_actions[k - 1],
                    ).reshape(-1, n_actions[k - 1])
            if not switched:
                break
        else:
            msg = f"Policy iteration didn't converge within {max_iterations} iterations."
            raise RuntimeError(msg)

        for level, (table, actions) in enumerate(zip(probabilities, n_actions, strict=True)):
            table[_information_keys(flat, branch, level)] = np.eye(actions)[greedy[level]]

        branch_value = (1 - gamma) * flat.mu0 @ value
        if branch_value < best_value:
            best_value, best_branch = branch_value, branch

    policies = {
        decision.symbol: RandomisedPolicy(
            symbol=decision.symbol,
            information_set=decision.information_set,
            states=decision.states,
            probabilities=dict(enumerate(table.tolist())),
        )
        for decision, table in zip(flat.decisions, probabilities, strict=True)
    }
    for atemporal in flat.atemporal_decisions:
        point_mass = np.eye(len(atemporal.states))[best_branch.atemporal[atemporal.symbol]]
        policies[atemporal.symbol] = RandomisedPolicy(
            symbol=atemporal.symbol, information_set=[], states=atemporal.states, probabilities={0: point_mass.tolist()}
        )
    return float(sign * best_value), policies
