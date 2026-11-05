"""Randomised stationary policies under full observation, on the flattened 2-slice step.

The slice-t decisions D_1, ..., D_n form a chain, each an ancestor of the next. D_n observes every
state node, every atemporal decision and every slice-t ancestor of D_n, and D_k observes what
D_{k+1} observes except D_k and its descendants. With s the joint state, o_k the slice-t nodes D_k
observes and D_{k-1} doesn't, and a_k D_k's action, the step flattens into a kernel
`T[s', (s, o_1, a_1, ..., o_n, a_n)] = P(o_1..o_n, s' | s, do(a_1..a_n))` and objective
coefficients `W[(s, o_1, a_1, ..., o_n, a_n)] = Σ P(o_1..o_n, rest | s, do(a_1..a_n)) u`. Each joint
value of the atemporal decisions is a branch with its own kernel and coefficients. With renewal,
the kernel holds the steps without a renewal, and a vector `R[(s, o_1, a_1, ..., o_n, a_n)] =
P(o_1..o_n, renew | s, do(a_1..a_n))` sends the renewed mass back to mu0. Over the
discounted occupations `y_0 = x(s)` and `y_k = y_{k-1} π_k(a_k | s, o_1, a_1, ..., o_k)` the step is
linear: `stationary_value_iteration` solves it for one objective, and `compile_stationary_lp` compiles it
into an LP `Problem` over the occupations.
"""

import itertools
from collections.abc import Hashable
from functools import reduce
from math import prod
from typing import Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from pydantic import BaseModel, ConfigDict

from desdeo.problem.schema import (
    Constraint,
    ConstraintTypeEnum,
    Objective,
    Problem,
    SparseTensorConstant,
    TensorConstant,
    TensorVariable,
    Variable,
    VariableTypeEnum,
)

from .dynamic import (
    Factor,
    RandomisedPolicy,
    StationaryDiagram,
    _ancestors,
    _contract,
    _network_factors,
    _renewal_factors,
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
    """`P(o_1..o_n, s', no renewal | s, do(a_1..a_n))`, rows the next state s', columns the entries of y_n."""
    objective_coefficients: dict[str, np.ndarray]
    """Objective symbol -> `Σ P(o_1..o_n, rest | s, do(a_1..a_n)) u` per entry of y_n, unsigned."""
    renewal: np.ndarray | None = None
    """`P(o_1..o_n, renewal | s, do(a_1..a_n))` per entry of y_n; `None` without `StationaryDiagram.renewal`."""


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


def _prefix_axes(flat: "FlattenedStep | StationaryLPIndexMap", level: int) -> tuple[list[str], list[int]]:
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


def flatten_stationary_step(diagram: StationaryDiagram) -> FlattenedStep:  # noqa: C901
    """Flatten `diagram`'s 2-slice step over its decision chain, one branch per atemporal choice.

    Each chain decision's CPT is replaced by an all-ones factor on its axis, and each atemporal
    decision's by a point mass on the branch's value. A prefix p_k whose newest observation has
    probability exactly 0 given its parent is dropped, together with everything below it. With
    `diagram.renewal`, each branch's kernel holds only the steps without a renewal, and its
    `renewal` vector the probability of one.

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
    renewal, alive, renew = _renewal_factors(diagram.renewal, cardinality)
    transition_names = sorted(
        _ancestors([*next_slicet, *all_observations, *renewal], parents, step) - set(chain_symbols)
    )
    renewal_names = sorted(_ancestors([*renewal, *all_observations], parents, step) - set(chain_symbols))
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
            [branch_factors[name] for name in transition_names] + identities + list(ones.values()) + alive,
            kernel_axes,
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

        renewal_vector = None
        if renew:
            table = _contract([branch_factors[name] for name in renewal_names] + list(ones.values()) + renew, flat_axes)
            renewal_vector = table.reshape(-1, n_actions[-1])[prefixes[-1]].reshape(-1)

        branches.append(
            FlattenedBranch(
                atemporal={
                    atemporal.symbol: value for atemporal, value in zip(atemporal_decisions, values, strict=True)
                },
                prefixes=prefixes,
                parents=prefix_parents,
                kernel=kernel,
                objective_coefficients=coefficients,
                renewal=renewal_vector,
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


def _information_keys(
    flat: "FlattenedStep | StationaryLPIndexMap", branch: "FlattenedBranch | StationaryLPBranch", level: int
) -> np.ndarray:
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


def _next_values(branch: FlattenedBranch, backward: sp.csr_array, mu0: np.ndarray, value: np.ndarray) -> np.ndarray:
    """E[V(s')] per entry of y_n: by the kernel, plus the renewal probability times `mu0 V`."""
    expected = backward @ value
    if branch.renewal is None:
        return expected
    return expected + branch.renewal * (mu0 @ value)


def _policy_value(
    flat: FlattenedStep, branch: FlattenedBranch, columns: np.ndarray, owner: np.ndarray, reward: np.ndarray
) -> np.ndarray:
    """V(s) of the deterministic chain policy reaching the entries `columns` of y_n from the states `owner`.

    The rank-one renewal term `gamma R mu0 V` is solved for by Sherman-Morrison.
    """
    gamma, mu0 = flat.discount, flat.mu0
    n_states = len(mu0)
    owner_matrix = sp.csr_array((np.ones(len(owner)), (np.arange(len(owner)), owner)), shape=(len(owner), n_states))
    closed_loop = branch.kernel[:, columns] @ owner_matrix
    step_reward = np.bincount(owner, weights=reward[columns], minlength=n_states)
    system = (sp.eye_array(n_states) - gamma * closed_loop.T).tocsc()
    if branch.renewal is None:
        return spla.spsolve(system, step_reward)
    renewed = np.bincount(owner, weights=branch.renewal[columns], minlength=n_states)
    base, response = spla.spsolve(system, np.column_stack([step_reward, renewed])).T
    return base + gamma * response * (mu0 @ base) / (1 - gamma * mu0 @ response)


def stationary_value_iteration(
    flat: FlattenedStep, objective_symbol: str, *, tol: float = 1e-12, max_iterations: int = 1000
) -> tuple[float, dict[str, RandomisedPolicy]]:
    """The full-observation optimum of one objective, by policy iteration warm-started by value iteration.

    On each branch, 100 value-iteration sweeps of the nested backup `V(s) = Σ_{o_1} min_{a_1} ...
    Σ_{o_n} min_{a_n} [W + gamma (T V + R mu0 V)]`, with W negated for a maximised objective and R
    the renewal vector (0 without renewal), give a starting greedy policy per chain decision. Policy
    iteration then evaluates the greedy policies exactly by a sparse linear solve and re-greedies on
    that V, from the last level up: Q_k sums level k+1's Q at its chosen actions, and an action of
    level k switches only where that lowers Q_k by more than
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
            q_last = (reward + gamma * _next_values(branch, backward, flat.mu0, value)).reshape(-1, n_actions[-1])
            _, value = _nested_backup(branch, n_actions, n_states, q_last)

        # Policy iteration from the greedy policy, with each policy's value from a linear solve.
        q_last = (reward + gamma * _next_values(branch, backward, flat.mu0, value)).reshape(-1, n_actions[-1])
        qs, _ = _nested_backup(branch, n_actions, n_states, q_last)
        greedy = [q.argmin(axis=1) for q in qs]
        for _ in range(max_iterations):
            columns, owner = _reached_entries(branch, n_actions, greedy)
            value = _policy_value(flat, branch, columns, owner, reward)
            # Switch bottom-up: Q_k sums the realised Q_{k+1} at level k+1's chosen actions.
            q = (reward + gamma * _next_values(branch, backward, flat.mu0, value)).reshape(-1, n_actions[-1])
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


_PREFIX = "slp_"
"""Prefix of the symbols `compile_stationary_lp` adds; objective symbols may not start with it."""


class StationaryLPBranch(BaseModel):
    """One branch's prefixes and symbols in the problem `compile_stationary_lp` builds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    atemporal: dict[str, int]
    """`FlattenedBranch.atemporal`."""
    prefixes: list[list[int]]
    """`FlattenedBranch.prefixes`."""
    parents: list[list[int]]
    """`FlattenedBranch.parents`."""
    m_symbol: str | None
    """The `Variable` of the branch's mass, the probability of its atemporal values; `None` with renewal."""
    nu_symbol: str | None = None
    """The `Variable` of the branch's inflow `nu`, the discounted mass arriving in it; `None` without renewal."""
    z_symbol: str | None = None
    """The binary `Variable` choosing the branch, with renewal and deterministic atemporal decisions;
    otherwise `None`."""
    nu_bound_symbol: str | None = None
    """The constraint `nu - z <= 0`; `None` without `z_symbol`."""
    x_symbol: str
    """The `TensorVariable` of the state occupation y_0 = x, one entry per joint state s."""
    y_symbols: list[str]
    """Per level k, the `TensorVariable` of the occupation y_k over (p_k, a_k); the last is w."""
    kernel_symbol: str
    """The `SparseTensorConstant` of `FlattenedBranch.kernel`."""
    coefficient_symbols: dict[str, str]
    """Objective symbol -> the 1-D `SparseTensorConstant` of its coefficients over w."""
    block_symbols: list[str]
    """Per level k, the `SparseTensorConstant` B_k summing y_k over a_k for each kept p_k."""
    parent_symbols: list[str]
    """Per level k, the `SparseTensorConstant` P_k selecting each kept p_k's entry in y_{k-1}."""
    consistency_symbols: list[str]
    """Per level k, the vector-valued constraint `B_k y_k - P_k y_{k-1} == 0`."""
    flow_symbol: str
    """The vector-valued flow constraint `x - gamma T w - (1 - gamma) m mu0 == 0`, or `x - gamma T w -
    nu mu0 == 0` with renewal."""
    delta_symbol: str | None = None
    """The `Variable` of the renewed mass `delta = R w`; `None` without renewal."""
    renewal_symbol: str | None = None
    """The 1-D `SparseTensorConstant` R of `FlattenedBranch.renewal`; `None` without renewal."""
    renewal_balance_symbol: str | None = None
    """The constraint `delta - R w == 0`; `None` without renewal."""


class StationaryLPIndexMap(BaseModel):
    """`compile_stationary_lp`'s bookkeeping, for `read_randomised_policy`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    state_nodes: list[str]
    """`FlattenedStep.state_nodes`."""
    state_cardinalities: list[int]
    """`FlattenedStep.state_cardinalities`."""
    decisions: list[DecisionNode]
    """`FlattenedStep.decisions`."""
    observation_nodes: list[list[str]]
    """`FlattenedStep.observation_nodes`."""
    observation_cardinalities: list[list[int]]
    """`FlattenedStep.observation_cardinalities`."""
    atemporal_decisions: list[DecisionNode]
    """`FlattenedStep.atemporal_decisions`."""
    branches: list[StationaryLPBranch]
    """One per `FlattenedStep.branches`, in the same order."""
    mu0_symbol: str
    """The `TensorConstant` of `FlattenedStep.mu0`."""
    mass_symbol: str | None
    """The constraint `Σ_g m_g == 1`; `None` with renewal."""
    inflow_symbol: str | None = None
    """The constraint `Σ_g nu_g - gamma Σ_g delta_g == 1 - gamma`; `None` without renewal."""
    choice_symbol: str | None = None
    """The constraint `Σ_g z_g == 1`; `None` without `StationaryLPBranch.z_symbol`."""
    discount: float
    """`StationaryDiagram.discount`."""


def _sparse_constant(name: str, symbol: str, matrix: sp.sparray, *, vector: bool = False) -> SparseTensorConstant:
    """`matrix` as a `SparseTensorConstant`, 1-D over its columns if `vector`."""
    coo = matrix.tocoo()
    if vector:
        return SparseTensorConstant(
            name=name, symbol=symbol, shape=[coo.shape[1]], indices=[coo.col.tolist()], values=coo.data.tolist()
        )
    return SparseTensorConstant(
        name=name,
        symbol=symbol,
        shape=list(coo.shape),
        indices=[coo.row.tolist(), coo.col.tolist()],
        values=coo.data.tolist(),
    )


def _occupation_variable(name: str, symbol: str, size: int) -> TensorVariable:
    """A real occupation vector in [0, 1]^size."""
    return TensorVariable(
        name=name,
        symbol=symbol,
        variable_type=VariableTypeEnum.real,
        shape=[size],
        lowerbounds=0.0,
        upperbounds=1.0,
    )


def compile_stationary_lp(
    diagram: StationaryDiagram, *, atemporal: Literal["randomised", "deterministic"] = "randomised"
) -> tuple[Problem, StationaryLPIndexMap]:
    """Compile `diagram` into an LP for its optimal randomised stationary policy under full observation.

    The step is flattened with `flatten_stationary_step`. Each branch g, one joint value of the
    atemporal decisions, has a mass `m_g`, a state occupation `x_g = y_{0,g}` and per chain level k
    an occupation `y_{k,g}` over the kept (p_k, a_k), with `w_g = y_{n,g}` and

    - `B_k y_{k,g} - P_k y_{k-1,g} = 0` per level, `Σ_{a_k} y_k(p_k, a_k) = y_{k-1}(parent of p_k)`,
    - `x_g = (1 - gamma) m_g mu0 + gamma T_g w_g`,
    - `Σ_g m_g = 1`.

    With `diagram.renewal`, each new patient, initial or renewed, draws its branch afresh. Each
    branch then has the renewed mass `delta_g = R_g w_g` and an inflow `nu_g >= 0` instead of `m_g`:

    - `delta_g = R_g w_g`,
    - `x_g = nu_g mu0 + gamma T_g w_g`,
    - `Σ_g nu_g - gamma Σ_g delta_g = 1 - gamma`, instead of `Σ_g m_g = 1`.

    Every arrival draws the same branch probabilities, so they are `q_g = nu_g / Σ_h nu_h`.

    Each objective is `Σ_g W_g w_g`, the per-step average as in `evaluate_stationary_policy`;
    multiply by 1 / (1 - gamma) for the total discounted sum. All added symbols start with `slp_`.

    With several atemporal decisions, the masses are a joint over their values, so the LP optimises
    over correlated choices, a relaxation of independent randomisation; `read_randomised_policy`
    raises when the optimal joint isn't a product of its marginals.

    Args:
        diagram: the stationary diagram; its chain decisions must observe their full observation sets.
        atemporal: `"randomised"` lets the masses take any value in [0, 1]; `"deterministic"` makes
            them binary when there is more than one branch, so the atemporal decisions take one value.
            With renewal and more than one branch, `"deterministic"` adds binary `z_g` with
            `Σ_g z_g = 1` and `nu_g <= z_g`.

    Returns:
        tuple[Problem, StationaryLPIndexMap]: the LP (a MILP for deterministic atemporal decisions)
            and its bookkeeping.

    Raises:
        ValueError: `atemporal` isn't one of its options, an objective symbol starts with `slp_`, or
            `flatten_stationary_step` raises.
    """
    if atemporal not in ("randomised", "deterministic"):
        msg = f"atemporal must be 'randomised' or 'deterministic', got {atemporal!r}."
        raise ValueError(msg)
    clashes = sorted(objective.symbol for objective in diagram.objectives if objective.symbol.startswith(_PREFIX))
    if clashes:
        msg = f"Objective symbols {clashes} start with '{_PREFIX}', which is reserved for the LP's own symbols."
        raise ValueError(msg)
    flat = flatten_stationary_step(diagram)
    gamma = flat.discount
    n_states = prod(flat.state_cardinalities)
    n_actions = [len(decision.states) for decision in flat.decisions]
    binary = atemporal == "deterministic" and len(flat.branches) > 1
    renewing = bool(diagram.renewal)
    linear = {"is_linear": True, "is_convex": True, "is_twice_differentiable": True}

    mu0_symbol = f"{_PREFIX}mu0"
    constants: list[TensorConstant | SparseTensorConstant] = [
        TensorConstant(name="Initial state distribution", symbol=mu0_symbol, shape=[n_states], values=flat.mu0.tolist())
    ]
    variables: list[Variable | TensorVariable] = []
    constraints: list[Constraint] = []
    branches: list[StationaryLPBranch] = []
    for g, branch in enumerate(flat.branches):
        m, x, kernel, flow = (f"{_PREFIX}{name}_{g}" for name in ("m", "x", "T", "flow"))
        ys, blocks, selections, consistencies = (
            [f"{_PREFIX}{name}_{g}_{k}" for k in range(len(n_actions))] for name in ("y", "B", "P", "consistency")
        )
        nu = z = nu_bound = None
        if renewing:
            m, nu = None, f"{_PREFIX}nu_{g}"
            variables.append(
                Variable(
                    name=f"Inflow of branch {g}",
                    symbol=nu,
                    variable_type=VariableTypeEnum.real,
                    lowerbound=0.0,
                    upperbound=1.0,
                )
            )
            if binary:
                z, nu_bound = f"{_PREFIX}z_{g}", f"{_PREFIX}nu_bound_{g}"
                variables.append(
                    Variable(
                        name=f"Choice of branch {g}",
                        symbol=z,
                        variable_type=VariableTypeEnum.binary,
                        lowerbound=0.0,
                        upperbound=1.0,
                    )
                )
                constraints.append(
                    Constraint(
                        name=f"Inflow of branch {g} only if chosen",
                        symbol=nu_bound,
                        cons_type=ConstraintTypeEnum.LTE,
                        func=["Subtract", nu, z],
                        **linear,
                    )
                )
        else:
            variables.append(
                Variable(
                    name=f"Mass of branch {g}",
                    symbol=m,
                    variable_type=VariableTypeEnum.binary if binary else VariableTypeEnum.real,
                    lowerbound=0.0,
                    upperbound=1.0,
                )
            )
        variables.append(_occupation_variable(f"State occupation of branch {g}", x, n_states))
        previous, previous_size = x, n_states
        for k, (prefixes, parents, actions) in enumerate(zip(branch.prefixes, branch.parents, n_actions, strict=True)):
            rows = len(prefixes)
            variables.append(_occupation_variable(f"Occupation of level {k} in branch {g}", ys[k], rows * actions))
            block = sp.csr_array(
                (np.ones(rows * actions), (np.repeat(np.arange(rows), actions), np.arange(rows * actions))),
                shape=(rows, rows * actions),
            )
            selection = sp.csr_array((np.ones(rows), (np.arange(rows), parents)), shape=(rows, previous_size))
            constants += [
                _sparse_constant(f"Action sums of level {k} in branch {g}", blocks[k], block),
                _sparse_constant(f"Parent entries of level {k} in branch {g}", selections[k], selection),
            ]
            constraints.append(
                Constraint(
                    name=f"Consistency of level {k} in branch {g}",
                    symbol=consistencies[k],
                    cons_type=ConstraintTypeEnum.EQ,
                    func=["Subtract", ["MatMul", blocks[k], ys[k]], ["MatMul", selections[k], previous]],
                    **linear,
                )
            )
            previous, previous_size = ys[k], rows * actions

        constants.append(_sparse_constant(f"Transition kernel of branch {g}", kernel, branch.kernel))
        arrivals = ["Multiply", nu, mu0_symbol] if renewing else ["Multiply", m, ["Multiply", 1 - gamma, mu0_symbol]]
        delta = renewal = renewal_balance = None
        if renewing:
            delta, renewal, renewal_balance = (f"{_PREFIX}{name}_{g}" for name in ("delta", "R", "renewal"))
            variables.append(
                Variable(
                    name=f"Renewed mass of branch {g}",
                    symbol=delta,
                    variable_type=VariableTypeEnum.real,
                    lowerbound=0.0,
                    upperbound=1.0,
                )
            )
            constants.append(
                _sparse_constant(
                    f"Renewal probabilities of branch {g}", renewal, sp.csr_array(branch.renewal[None, :]), vector=True
                )
            )
            constraints.append(
                Constraint(
                    name=f"Renewed mass of branch {g}",
                    symbol=renewal_balance,
                    cons_type=ConstraintTypeEnum.EQ,
                    func=["Subtract", delta, ["MatMul", renewal, ys[-1]]],
                    **linear,
                )
            )
        coefficient_symbols = {}
        for i, (symbol, coefficients) in enumerate(branch.objective_coefficients.items()):
            coefficient_symbols[symbol] = f"{_PREFIX}W_{g}_{i}"
            constants.append(
                _sparse_constant(
                    f"Coefficients of objective '{symbol}' in branch {g}",
                    coefficient_symbols[symbol],
                    sp.csr_array(coefficients[None, :]),
                    vector=True,
                )
            )
        constraints.append(
            Constraint(
                name=f"Flow balance of branch {g}",
                symbol=flow,
                cons_type=ConstraintTypeEnum.EQ,
                func=["Subtract", x, ["Multiply", gamma, ["MatMul", kernel, ys[-1]]], arrivals],
                **linear,
            )
        )
        branches.append(
            StationaryLPBranch(
                atemporal=branch.atemporal,
                prefixes=[prefixes.tolist() for prefixes in branch.prefixes],
                parents=[parents.tolist() for parents in branch.parents],
                m_symbol=m,
                nu_symbol=nu,
                z_symbol=z,
                nu_bound_symbol=nu_bound,
                x_symbol=x,
                y_symbols=ys,
                kernel_symbol=kernel,
                coefficient_symbols=coefficient_symbols,
                block_symbols=blocks,
                parent_symbols=selections,
                consistency_symbols=consistencies,
                flow_symbol=flow,
                delta_symbol=delta,
                renewal_symbol=renewal,
                renewal_balance_symbol=renewal_balance,
            )
        )

    def total(symbols: list[str]) -> str | list:
        return ["Add", *symbols] if len(symbols) > 1 else symbols[0]

    mass_symbol = inflow_symbol = choice_symbol = None
    if renewing:
        inflow_symbol = f"{_PREFIX}inflow"
        inflows = total([branch.nu_symbol for branch in branches])
        renewed = total([branch.delta_symbol for branch in branches])
        constraints.append(
            Constraint(
                name="Branch inflows are the initial and the renewed mass",
                symbol=inflow_symbol,
                cons_type=ConstraintTypeEnum.EQ,
                func=["Subtract", inflows, ["Multiply", gamma, renewed], 1 - gamma],
                **linear,
            )
        )
        if binary:
            choice_symbol = f"{_PREFIX}choice"
            constraints.append(
                Constraint(
                    name="One branch is chosen",
                    symbol=choice_symbol,
                    cons_type=ConstraintTypeEnum.EQ,
                    func=["Subtract", total([branch.z_symbol for branch in branches]), 1],
                    **linear,
                )
            )
    else:
        mass_symbol = f"{_PREFIX}mass"
        constraints.append(
            Constraint(
                name="Branch masses sum to one",
                symbol=mass_symbol,
                cons_type=ConstraintTypeEnum.EQ,
                func=["Subtract", total([branch.m_symbol for branch in branches]), 1],
                **linear,
            )
        )

    objectives = []
    for objective in diagram.objectives:
        terms = [["MatMul", branch.coefficient_symbols[objective.symbol], branch.y_symbols[-1]] for branch in branches]
        objectives.append(
            Objective(
                name=objective.name or objective.symbol,
                symbol=objective.symbol,
                unit=objective.unit,
                func=["Add", *terms] if len(terms) > 1 else terms[0],
                maximize=objective.maximize,
                **linear,
            )
        )

    problem = Problem(
        name="Stationary randomised policy LP",
        description=(
            "The occupation-measure LP of a stationary diagram's randomised policies under full observation; "
            "objectives are per-step averages."
        ),
        constants=constants,
        variables=variables,
        objectives=objectives,
        constraints=constraints,
    )
    return problem, StationaryLPIndexMap(
        state_nodes=flat.state_nodes,
        state_cardinalities=flat.state_cardinalities,
        decisions=flat.decisions,
        observation_nodes=flat.observation_nodes,
        observation_cardinalities=flat.observation_cardinalities,
        atemporal_decisions=flat.atemporal_decisions,
        branches=branches,
        mu0_symbol=mu0_symbol,
        mass_symbol=mass_symbol,
        inflow_symbol=inflow_symbol,
        choice_symbol=choice_symbol,
        discount=gamma,
    )


def read_randomised_policy(
    index_map: StationaryLPIndexMap, optimal_variables: dict[str, list[float] | float], *, tol: float = 1e-10
) -> dict[str, RandomisedPolicy]:
    """Read the randomised policies from a solution of `compile_stationary_lp`'s problem.

    Chain decision k takes `π_k(a | p_k) = y_k(p_k, a) / Σ_a y_k(p_k, ·)`, negative entries clipped
    to 0, where the parent entry of p_k in y_{k-1} is above `tol` in a branch with normalised mass
    above 0; every other information state, unreachable or dropped, gets the uniform distribution.
    Each atemporal decision gets the marginal of the normalised masses, those `<= tol` set to 0.
    With renewal, the inflows take the masses' place, so the marginals are those of the arrival
    probabilities `q_g = nu_g / Σ_h nu_h`.

    Args:
        index_map: the `StationaryLPIndexMap` of the problem.
        optimal_variables: `SolverResults.optimal_variables` of a solve of it.
        tol: the occupation and mass or inflow at or below which a branch or prefix counts as unreached.

    Returns:
        dict[str, RandomisedPolicy]: a policy for each chain decision and each atemporal decision.

    Raises:
        ValueError: no mass or inflow is above `tol`; a decision's information keys collide across branches;
            or, with several atemporal decisions, the masses aren't the product of their marginals,
            so independent policies can't represent them.
    """
    atemporal_cards = [len(atemporal.states) for atemporal in index_map.atemporal_decisions]
    n_actions = [len(decision.states) for decision in index_map.decisions]
    probabilities = [
        np.full((prod(_prefix_axes(index_map, level)[1]) * prod(atemporal_cards), actions), 1.0 / actions)
        for level, actions in enumerate(n_actions)
    ]

    masses = np.array([float(optimal_variables[branch.nu_symbol or branch.m_symbol]) for branch in index_map.branches])
    masses[masses <= tol] = 0.0
    if masses.sum() == 0:
        msg = f"No branch mass or inflow is above {tol}."
        raise ValueError(msg)
    masses /= masses.sum()

    written = [[] for _ in n_actions]
    for branch, mass in zip(index_map.branches, masses, strict=True):
        if mass == 0:
            continue
        previous = np.asarray(optimal_variables[branch.x_symbol], dtype=float)
        for level, (table, actions) in enumerate(zip(probabilities, n_actions, strict=True)):
            y = np.clip(np.asarray(optimal_variables[branch.y_symbols[level]], dtype=float), 0.0, None)
            y = y.reshape(-1, actions)
            totals = y.sum(axis=1)
            reached = (previous[branch.parents[level]] > tol) & (totals > 0)
            rows = np.full_like(y, 1.0 / actions)
            rows[reached] = y[reached] / totals[reached, None]
            keys = _information_keys(index_map, branch, level)
            table[keys] = rows
            written[level].append(keys)
            previous = y.reshape(-1)

    for decision, level_keys in zip(index_map.decisions, written, strict=True):
        keys = np.concatenate(level_keys)
        if len(np.unique(keys)) < len(keys):
            msg = (
                f"Branches write the same information-state keys of decision '{decision.symbol}'; its information "
                "set must cover the prefix axes and all atemporal decisions."
            )
            raise ValueError(msg)

    policies = {
        decision.symbol: RandomisedPolicy(
            symbol=decision.symbol,
            information_set=decision.information_set,
            states=decision.states,
            probabilities=dict(enumerate(table.tolist())),
        )
        for decision, table in zip(index_map.decisions, probabilities, strict=True)
    }
    if not index_map.atemporal_decisions:
        return policies
    # Branches enumerate the atemporal decisions' joint values, the last varying fastest.
    joint = masses.reshape(atemporal_cards)
    marginals = [joint.sum(axis=tuple(j for j in range(joint.ndim) if j != i)) for i in range(joint.ndim)]
    if not np.allclose(joint, reduce(np.multiply.outer, marginals), rtol=0, atol=tol):
        msg = "The atemporal decisions' joint masses aren't a product of their marginals."
        raise ValueError(msg)
    for atemporal, marginal in zip(index_map.atemporal_decisions, marginals, strict=True):
        policies[atemporal.symbol] = RandomisedPolicy(
            symbol=atemporal.symbol, information_set=[], states=atemporal.states, probabilities={0: marginal.tolist()}
        )
    return policies
