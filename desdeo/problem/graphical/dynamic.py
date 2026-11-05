"""Stationary policies for dynamic influence diagrams given as 2-time-slice networks (2TBNs).

A `StationaryDiagram` is a status-quo 2TBN in pyAgrum's naming (`X0` is the previous slice, `Xt`
the current one, unsuffixed nodes are atemporal) with some of its nodes overridden as decisions.
One stationary policy is applied at every step. Objectives are per-step averages under the
normalised discounted occupation measure x = (1 - gamma) Σ_k gamma^k P(state_k), whose initial state is the
status-quo slice-0 population.
"""

import itertools
from collections.abc import Hashable, Sequence
from math import prod
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from desdeo.problem.schema import Constraint, ConstraintTypeEnum, Problem

from .compile import (
    DecisionPolicy,
    IndexMap,
    _extract_expr,
    _group_flat_indices_by_projection,
    compile_diagram,
)
from .index_map import mixed_radix_decode
from .network import DiscreteNetwork, DiscreteVariable, Potential, kahn_topological_order
from .schema import DecisionAnnotation, DecisionNode, DiagramConstraint, ObjectiveNode, _check_unique_symbols

Factor = tuple[Sequence[Hashable], np.ndarray]
"""A table and the names of its axes, in order."""


_EINSUM_MAX_AXES = 52
"""numpy's einsum accepts at most this many distinct integer axis labels."""

_INITIAL_SUM_TOLERANCE = 1e-9
"""How far `StationaryDiagram.initial_distribution` may sum from 1."""

_STATE_SINK = "__state_t"
"""One-state chance node whose cluster holds the slice-t state joint."""

_PROBABILITY_SUM_TOLERANCE = 1e-9
"""How far each `RandomisedPolicy` distribution may sum from 1."""

_MAX_ATEMPORAL_DRAWS = 1024
"""The most deterministic draws `_mix_atemporal_draws` evaluates."""


def _check_known_decisions(symbols: set[str], node_names: set[str]) -> None:
    """Raise `ValueError` if a decision symbol isn't a network node."""
    unknown = sorted(symbols - node_names)
    if unknown:
        msg = f"Decision symbols {unknown} must name network nodes."
        raise ValueError(msg)


def _check_no_decided_ancestors(ancestors: set[str], decided: set[str]) -> None:
    """Raise `ValueError` if a decided node is among the state nodes' `ancestors`."""
    decided_ancestors = sorted(ancestors & decided)
    if decided_ancestors:
        msg = (
            f"Decisions {decided_ancestors} are ancestors of the state nodes, so the initial distribution "
            "depends on them; give initial_distribution."
        )
        raise ValueError(msg)


class TwoSliceStructure(BaseModel):
    """The time slices of a 2TBN's nodes and its Markov state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    slice0: set[str]
    """Nodes `X0` whose `Xt` exists."""
    slicet: set[str]
    """Nodes `Xt`."""
    atemporal: set[str]
    """All other nodes."""
    state_nodes: list[str]
    """Slice-0 and atemporal chance nodes with a child in slice t or among the atemporal decisions.
    Atemporal nodes come first, then slice-0 nodes, each in topological order."""
    next_of: dict[str, str]
    """State node -> its value one step later: `X0` -> `Xt`, an atemporal node -> itself."""


def classify_2tbn(network: DiscreteNetwork, decisions: Sequence[DecisionNode] = ()) -> TwoSliceStructure:
    """Split `network`'s nodes into time slices and find the Markov state.

    A name ending in `t` is a slice-t node, and its slice-0 copy (the same name ending in `0`)
    must exist with the same number of states. A name ending in `0` is a slice-0 node if its
    slice-t copy exists. Every other name is atemporal.

    `decisions` override network nodes; each decision's parents are its information set instead
    of its network parents.

    Raises:
        ValueError: a slice-t node lacks its slice-0 copy or their domains differ, a decision isn't
            a network node or is in slice 0, an information set references an unknown node, an
            arc goes from slice t to slice 0, an atemporal node has a parent in a time slice, or
            the information sets create a cycle.
    """
    names = [variable.name for variable in network.variables]
    name_set = set(names)
    cardinality = {variable.name: len(variable.labels) for variable in network.variables}

    slice0: set[str] = set()
    slicet: set[str] = set()
    atemporal: set[str] = set()
    for name in names:
        if name.endswith("t"):
            partner = f"{name[:-1]}0"
            if partner not in name_set:
                msg = f"Node '{name}' ends in 't', so it is in slice t, but its slice-0 copy '{partner}' is missing."
                raise ValueError(msg)
            if cardinality[partner] != cardinality[name]:
                msg = f"Nodes '{partner}' and '{name}' must have the same number of states."
                raise ValueError(msg)
            slicet.add(name)
        elif name.endswith("0") and f"{name[:-1]}t" in name_set:
            slice0.add(name)
        else:
            atemporal.add(name)

    decision_symbols = {decision.symbol for decision in decisions}
    _check_known_decisions(decision_symbols, name_set)
    in_slice0 = sorted(decision_symbols & slice0)
    if in_slice0:
        msg = f"Decisions {in_slice0} are in slice 0, which holds the status-quo population; decide the slice-t nodes."
        raise ValueError(msg)

    parents: dict[str, set[str]] = {name: set() for name in names}
    for parent, child in network.arcs:
        parents[child].add(parent)
    arcs = list(network.arcs)
    for decision in decisions:
        unknown = [name for name in decision.information_set if name not in name_set]
        if unknown:
            msg = f"Decision '{decision.symbol}': information_set references unknown node(s) {unknown}."
            raise ValueError(msg)
        parents[decision.symbol] = set(decision.information_set)
        arcs.extend((name, decision.symbol) for name in decision.information_set)

    for parent, child in arcs:
        if child in slice0 and parent in slicet:
            msg = f"Arc '{parent}' -> '{child}' goes from slice t back to slice 0."
            raise ValueError(msg)
        if child in atemporal and parent not in atemporal:
            msg = f"Atemporal node '{child}' has parent '{parent}' in a time slice."
            raise ValueError(msg)

    order = kahn_topological_order(names, parents)
    if len(order) != len(names):
        msg = f"Decision information sets create a cycle among {sorted(name_set - set(order))}."
        raise ValueError(msg)

    drivers = slicet | (atemporal & decision_symbols)
    state = {parent for child in drivers for parent in parents[child]} - drivers - decision_symbols
    state_nodes = [name for name in order if name in state and name in atemporal]
    state_nodes += [name for name in order if name in state and name in slice0]
    next_of = {name: name if name in atemporal else f"{name[:-1]}t" for name in state_nodes}

    return TwoSliceStructure(
        slice0=slice0, slicet=slicet, atemporal=atemporal, state_nodes=state_nodes, next_of=next_of
    )


def promote_to_decisions(network: DiscreteNetwork, decisions: Sequence[DecisionNode]) -> DiscreteNetwork:
    """`network` without the decision nodes, their potentials and every arc touching them.

    The children's potentials keep the decision axes, as `DecisionAnnotation` expects.
    """
    symbols = {decision.symbol for decision in decisions}
    _check_known_decisions(symbols, {variable.name for variable in network.variables})

    return DiscreteNetwork(
        name=network.name,
        variables=[variable for variable in network.variables if variable.name not in symbols],
        arcs=[(parent, child) for parent, child in network.arcs if parent not in symbols and child not in symbols],
        potentials={name: potential for name, potential in network.potentials.items() if name not in symbols},
    )


def _ancestors(nodes: Sequence[str], parents: dict[str, Sequence[str]], within: set[str] | None = None) -> set[str]:
    """`nodes` and their ancestors under `parents`, walking only through `within` if given."""
    found: set[str] = set()
    stack = [node for node in nodes if within is None or node in within]
    while stack:
        node = stack.pop()
        if node in found:
            continue
        found.add(node)
        stack.extend(parent for parent in parents[node] if within is None or parent in within)
    return found


def _check_undecided_atemporal_state(
    structure: TwoSliceStructure, parents: dict[str, Sequence[str]], decided: set[str]
) -> None:
    """Raise `ValueError` if a decided node is an ancestor of an atemporal state node."""
    atemporal_state = [name for name in structure.state_nodes if name in structure.atemporal]
    decided_ancestors = sorted(_ancestors(atemporal_state, parents) & decided)
    if decided_ancestors:
        msg = (
            f"Decisions {decided_ancestors} are ancestors of atemporal state nodes {atemporal_state}, which "
            "stay fixed across steps; an atemporal chance state node can't depend on a decision."
        )
        raise ValueError(msg)


def _check_objectives(
    objectives: list[ObjectiveNode], structure: TwoSliceStructure, decisions: set[str], labels: dict[str, list[str]]
) -> None:
    """Raise `ValueError` if an objective's parents aren't slice-t, decision or atemporal state nodes."""
    atemporal_state = [name for name in structure.state_nodes if name in structure.atemporal]
    allowed = structure.slicet | decisions | set(atemporal_state)
    for objective in objectives:
        in_slice0 = [parent for parent in objective.parents if parent in structure.slice0]
        if in_slice0:
            msg = f"Objective '{objective.symbol}' references slice-0 node(s) {in_slice0}; use the slice-t copies."
            raise ValueError(msg)
        not_allowed = [parent for parent in objective.parents if parent not in allowed]
        if not_allowed:
            msg = (
                f"Objective '{objective.symbol}': parents {not_allowed} must be slice-t nodes, decisions or "
                f"atemporal state nodes {atemporal_state}."
            )
            raise ValueError(msg)
        expected_shape = [len(labels[parent]) for parent in objective.parents]
        if objective.utility.shape != expected_shape:
            msg = (
                f"Objective '{objective.symbol}': utility has shape {objective.utility.shape}, "
                f"expected {expected_shape} from its parents' domains."
            )
            raise ValueError(msg)


class StationaryDiagram(BaseModel):
    """A status-quo 2TBN with some nodes overridden as decisions taking one stationary policy.

    Decisions act from slice t onwards; slice 0 is the status-quo population. Each decision's
    `information_set` replaces its network parents, and its `states` must equal the node's labels.
    An atemporal decision's information set may only contain atemporal nodes, and no atemporal
    chance state node may have a decision ancestor.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    network: DiscreteNetwork
    """The full status-quo 2TBN."""
    decisions: list[DecisionNode]
    """Each symbol names a network node whose CPT the decision overrides."""
    objectives: list[ObjectiveNode]
    """Parents are slice-t nodes, decision symbols or atemporal state nodes."""
    constraints: list[DiagramConstraint] = Field(default_factory=list)
    """Nodes are slice-t nodes, decisions, state nodes or objective symbols; a constraint on a slice-0
    or atemporal state node constrains its occupation marginal."""
    discount: float
    """The discount factor gamma, in (0, 1)."""
    initial_distribution: Potential | None = None
    """Joint over `TwoSliceStructure.state_nodes`, in that order, nonnegative and summing to 1. `None`
    contracts the slice-0 and atemporal CPTs."""

    @model_validator(mode="after")
    def _check_diagram(self) -> Self:
        if not 0.0 < self.discount < 1.0:
            msg = f"discount must be in (0, 1), got {self.discount}."
            raise ValueError(msg)

        node_names = {variable.name for variable in self.network.variables}
        labels = {variable.name: variable.labels for variable in self.network.variables}
        for name, potential in self.network.potentials.items():
            external = [axis for axis in potential.variables if axis not in node_names]
            if external:
                msg = (
                    f"Potential for '{name}' has axes {external} outside the network; decisions must be network nodes."
                )
                raise ValueError(msg)

        decision_symbols = [decision.symbol for decision in self.decisions]
        objective_symbols = [objective.symbol for objective in self.objectives]
        constraint_symbols = [constraint.symbol for constraint in self.constraints]
        _check_unique_symbols("within decisions", decision_symbols)
        _check_unique_symbols("across objectives/constraints", objective_symbols + constraint_symbols)
        collisions = sorted(node_names & {*objective_symbols, *constraint_symbols})
        if collisions:
            msg = f"Objective/constraint symbols collide with network node names: {collisions}."
            raise ValueError(msg)
        if len(self.objectives) == 0:
            msg = "StationaryDiagram must have at least one objective."
            raise ValueError(msg)

        structure = classify_2tbn(self.network, self.decisions)

        for decision in self.decisions:
            if decision.states != labels[decision.symbol]:
                msg = (
                    f"Decision '{decision.symbol}': states {decision.states} must equal the node's labels "
                    f"{labels[decision.symbol]}."
                )
                raise ValueError(msg)

        _check_objectives(self.objectives, structure, set(decision_symbols), labels)

        referenceable = structure.slicet | set(decision_symbols) | set(structure.state_nodes) | set(objective_symbols)
        for constraint in self.constraints:
            not_allowed = [node for node in constraint.nodes if node not in referenceable]
            if not_allowed:
                msg = (
                    f"Constraint '{constraint.symbol}': nodes {not_allowed} must be slice-t nodes, decisions, "
                    f"state nodes {structure.state_nodes} or objective symbols."
                )
                raise ValueError(msg)

        parents = {name: potential.variables[1:] for name, potential in self.network.potentials.items()}
        _check_undecided_atemporal_state(structure, parents, set(decision_symbols))

        if self.initial_distribution is not None:
            expected_shape = [len(labels[name]) for name in structure.state_nodes]
            if (
                self.initial_distribution.variables != structure.state_nodes
                or self.initial_distribution.shape != expected_shape
            ):
                msg = (
                    f"initial_distribution must be over {structure.state_nodes} with shape {expected_shape}, got "
                    f"{self.initial_distribution.variables} with shape {self.initial_distribution.shape}."
                )
                raise ValueError(msg)
            values = np.array(self.initial_distribution.values, dtype=float)
            if not np.isfinite(values).all():
                msg = "initial_distribution must be finite, nonnegative and sum to 1, got non-finite values."
                raise ValueError(msg)
            if (values < 0).any():
                msg = "initial_distribution must be finite, nonnegative and sum to 1, got negative values."
                raise ValueError(msg)
            if abs(values.sum() - 1.0) > _INITIAL_SUM_TOLERANCE:
                msg = f"initial_distribution must be finite, nonnegative and sum to 1, got sum {values.sum()}."
                raise ValueError(msg)
        else:
            _check_no_decided_ancestors(_ancestors(structure.state_nodes, parents), set(decision_symbols))

        return self

    @property
    def structure(self) -> TwoSliceStructure:
        """`classify_2tbn` of `network` with `decisions`."""
        return classify_2tbn(self.network, self.decisions)

    @classmethod
    def with_default_information_sets(
        cls,
        network: DiscreteNetwork,
        decisions: dict[str, list[str] | None],
        objectives: list[ObjectiveNode],
        *,
        discount: float,
        constraints: list[DiagramConstraint] | None = None,
        initial_distribution: Potential | None = None,
    ) -> Self:
        """Build a diagram whose decisions take their states and default information sets from `network`.

        Args:
            network: the full status-quo 2TBN.
            decisions: node name -> information set; `None` uses the node's parents in `network`,
                in its potential's order.
            objectives: as in `StationaryDiagram.objectives`.
            discount: as in `StationaryDiagram.discount`.
            constraints: as in `StationaryDiagram.constraints`.
            initial_distribution: as in `StationaryDiagram.initial_distribution`.
        """
        variables = {variable.name: variable for variable in network.variables}
        _check_known_decisions(set(decisions), set(variables))

        decision_nodes = [
            DecisionNode(
                symbol=symbol,
                states=variables[symbol].labels,
                information_set=network.potentials[symbol].variables[1:]
                if information_set is None
                else information_set,
            )
            for symbol, information_set in decisions.items()
        ]
        return cls(
            network=network,
            decisions=decision_nodes,
            objectives=objectives,
            constraints=constraints or [],
            discount=discount,
            initial_distribution=initial_distribution,
        )


class StationaryEvaluation(BaseModel):
    """Objective values and occupation measure of one stationary policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    objectives: dict[str, float]
    """Objective symbol -> per-step average Σ_s x(s) E[utility | s], the compiled objectives' scale.
    Multiply by 1 / (1 - gamma) for the total discounted sum."""
    occupation: Potential
    """The occupation measure x over `TwoSliceStructure.state_nodes`."""


def _contract(factors: Sequence[Factor], output: Sequence[Hashable]) -> np.ndarray:
    """Multiply `factors` and sum out every axis not in `output`."""
    if not factors:
        return np.ones(())
    axis_ids: dict[Hashable, int] = {}
    operands: list = []
    for axes, values in factors:
        operands += [values, [axis_ids.setdefault(axis, len(axis_ids)) for axis in axes]]
    if len(axis_ids) > _EINSUM_MAX_AXES:
        msg = f"Contraction has {len(axis_ids)} distinct axes; numpy's einsum allows at most {_EINSUM_MAX_AXES}."
        raise ValueError(msg)
    return np.einsum(*operands, [axis_ids[axis] for axis in output], optimize=True)


def _initial_joint(
    factors: dict[str, Factor], parents: dict[str, Sequence[str]], state_nodes: list[str], decided: set[str]
) -> np.ndarray:
    """Joint over `state_nodes` from the CPTs of their ancestors; raise if a decided node is one."""
    ancestors = _ancestors(state_nodes, parents)
    _check_no_decided_ancestors(ancestors, decided)
    return _contract([factors[name] for name in sorted(ancestors)], state_nodes)


def _network_factors(network: DiscreteNetwork) -> dict[str, Factor]:
    return {
        name: (potential.variables, np.array(potential.values, dtype=float))
        for name, potential in network.potentials.items()
    }


def initial_state_distribution(diagram: StationaryDiagram) -> np.ndarray:
    """The slice-0 joint over `diagram.structure.state_nodes`, axes in that order.

    `diagram.initial_distribution` if set, otherwise the contraction of the slice-0 and atemporal
    CPTs that are ancestors of the state nodes.

    Raises:
        ValueError: a decision is among those ancestors and no `initial_distribution` was given.
    """
    if diagram.initial_distribution is not None:
        return np.array(diagram.initial_distribution.values, dtype=float)

    factors = _network_factors(diagram.network)
    parents = {name: axes[1:] for name, (axes, _) in factors.items()}
    decided = {decision.symbol for decision in diagram.decisions}
    return _initial_joint(factors, parents, diagram.structure.state_nodes, decided)


class RandomisedPolicy(BaseModel):
    """One decision's randomised policy: a distribution over its states per information state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    """The decision's symbol."""
    information_set: list[str]
    """Observed nodes, most-significant-first in `probabilities`' key encoding."""
    states: list[str]
    """The decision's alternatives, in state-index order."""
    probabilities: dict[int, list[float]]
    """Information-state flat index (mixed-radix over `information_set`, `0` when it is empty) -> the
    probability of each state in `states`."""

    @model_validator(mode="after")
    def _check_distributions(self) -> Self:
        for flat_index, distribution in self.probabilities.items():
            if len(distribution) != len(self.states):
                msg = (
                    f"Policy for '{self.symbol}': information state {flat_index} has {len(distribution)} "
                    f"probabilities, expected {len(self.states)}."
                )
                raise ValueError(msg)
            values = np.array(distribution, dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                msg = f"Policy for '{self.symbol}': information state {flat_index} has non-finite or negative values."
                raise ValueError(msg)
            if abs(values.sum() - 1.0) > _PROBABILITY_SUM_TOLERANCE:
                msg = f"Policy for '{self.symbol}': information state {flat_index} sums to {values.sum()}, not 1."
                raise ValueError(msg)
        return self


def _policy_factor(
    decision: DecisionNode, policy: DecisionPolicy | RandomisedPolicy, cardinality: dict[str, int]
) -> Factor:
    """A CPT over `[decision, *policy.information_set]` from `policy`'s flat-index keys."""
    if policy.symbol != decision.symbol:
        msg = f"Policy for '{policy.symbol}' given for decision '{decision.symbol}'."
        raise ValueError(msg)
    if sorted(policy.information_set) != sorted(decision.information_set):
        msg = (
            f"Policy for '{decision.symbol}' has information set {policy.information_set}, but the decision "
            f"observes {decision.information_set}."
        )
        raise ValueError(msg)
    if policy.states != decision.states:
        msg = f"Policy for '{decision.symbol}' has states {policy.states}, expected {decision.states}."
        raise ValueError(msg)

    cards = [cardinality[name] for name in policy.information_set]
    keys = policy.probabilities if isinstance(policy, RandomisedPolicy) else policy.policy
    missing = sorted(set(range(prod(cards))) - set(keys))
    if missing:
        msg = f"Policy for '{decision.symbol}' has no choice for information states {missing}."
        raise ValueError(msg)
    extra = sorted(set(keys) - set(range(prod(cards))))
    if extra:
        msg = f"Policy for '{decision.symbol}' has keys {extra} outside the {prod(cards)} information states."
        raise ValueError(msg)

    if isinstance(policy, RandomisedPolicy):
        # Mixed-radix keys, most significant first, are C order over the information set's axes.
        rows = np.array([policy.probabilities[flat_index] for flat_index in range(prod(cards))], dtype=float)
        return [decision.symbol, *policy.information_set], rows.T.reshape([len(decision.states), *cards])

    values = np.zeros([len(decision.states), *cards])
    for flat_index in range(prod(cards)):
        choice = policy.policy[flat_index]
        if not 0 <= choice < len(decision.states):
            msg = f"Policy for '{decision.symbol}' chooses state {choice}, out of range."
            raise ValueError(msg)
        values[(choice, *mixed_radix_decode(flat_index, cards))] = 1.0
    return [decision.symbol, *policy.information_set], values


def _mix_atemporal_draws(
    diagram: StationaryDiagram,
    policies: dict[str, DecisionPolicy | RandomisedPolicy | None],
    drawn: dict[str, Factor],
) -> StationaryEvaluation:
    """Mix the evaluations of every deterministic draw of the atemporal policies `drawn`.

    Each draw picks one state per information state of each drawn decision and is weighted by the
    product of those states' probabilities.

    Raises:
        ValueError: there are more than `_MAX_ATEMPORAL_DRAWS` draws.
    """
    n_draws = prod(
        int(support)
        for _, values in drawn.values()
        for support in np.count_nonzero(values.reshape(len(values), -1), axis=0)
    )
    if n_draws > _MAX_ATEMPORAL_DRAWS:
        msg = (
            f"With initial_distribution set, randomised atemporal policies are evaluated by mixing their "
            f"{n_draws} deterministic draws, more than the {_MAX_ATEMPORAL_DRAWS} allowed; leave "
            f"initial_distribution unset or randomise fewer information states."
        )
        raise ValueError(msg)
    alternatives = []
    for symbol, (axes, values) in drawn.items():
        decision = next(decision for decision in diagram.decisions if decision.symbol == symbol)
        # Rows in C order over the information set are its mixed-radix keys.
        rows = values.reshape(len(decision.states), -1).T
        alternatives.append(
            [
                (
                    prod(row[choice] for row, choice in zip(rows, choices, strict=True)),
                    DecisionPolicy(
                        symbol=symbol,
                        information_set=list(axes[1:]),
                        states=decision.states,
                        policy={key: int(choice) for key, choice in enumerate(choices)},
                    ),
                )
                for choices in itertools.product(*(np.flatnonzero(row) for row in rows))
            ]
        )

    objectives = dict.fromkeys((objective.symbol for objective in diagram.objectives), 0.0)
    occupation = 0.0
    for draw in itertools.product(*alternatives):
        weight = prod(probability for probability, _ in draw)
        evaluation = evaluate_stationary_policy(diagram, policies | {policy.symbol: policy for _, policy in draw})
        for symbol, value in evaluation.objectives.items():
            objectives[symbol] += weight * value
        occupation = occupation + weight * np.array(evaluation.occupation.values, dtype=float)
    return StationaryEvaluation(
        objectives=objectives,
        occupation=Potential(
            variables=evaluation.occupation.variables, shape=evaluation.occupation.shape, values=occupation.tolist()
        ),
    )


def evaluate_stationary_policy(
    diagram: StationaryDiagram, policies: dict[str, DecisionPolicy | RandomisedPolicy | None]
) -> StationaryEvaluation:
    """Objective values and occupation measure of `diagram` under `policies`.

    A randomised atemporal policy is drawn once, not at every step. Without
    `diagram.initial_distribution`, every atemporal policy is then a state node's CPT, like a
    status-quo atemporal node's. With it, the objectives and the occupation are the
    probability-weighted mix of those of the deterministic draws.

    Args:
        diagram: the stationary diagram.
        policies: decision symbol -> deterministic or randomised policy, keyed as
            `read_policy_from_solution` produces them. `None` or a missing key uses the node's
            status-quo CPT from `diagram.network`.

    Raises:
        ValueError: a policy doesn't match its decision, status-quo decisions need state nodes
            outside `diagram.initial_distribution`'s, or there are too many draws to mix.
    """
    network = diagram.network
    structure = diagram.structure
    decisions = {decision.symbol: decision for decision in diagram.decisions}
    unknown = sorted(set(policies) - set(decisions))
    if unknown:
        msg = f"Policies given for unknown decisions {unknown}."
        raise ValueError(msg)

    cardinality = {variable.name: len(variable.labels) for variable in network.variables}
    factors = _network_factors(network)
    decided: set[str] = set()
    for symbol, policy in policies.items():
        if policy is not None:
            factors[symbol] = _policy_factor(decisions[symbol], policy, cardinality)
            decided.add(symbol)
    drawn = {
        symbol: factors[symbol]
        for symbol in policies
        if symbol in decided & structure.atemporal and not np.isin(factors[symbol][1], (0.0, 1.0)).all()
    }
    if drawn:
        if diagram.initial_distribution is not None:
            return _mix_atemporal_draws(diagram, policies, drawn)
        # Undecided atemporal nodes are state nodes, drawn once from their CPTs.
        decided -= structure.atemporal
    parents = {name: axes[1:] for name, (axes, _) in factors.items()}

    # Status-quo decisions are chance nodes here, so the state may need more nodes than the diagram's.
    step = structure.slicet | (structure.atemporal & decided)
    needed = set(structure.state_nodes)
    for name in step:
        needed.update(parent for parent in parents[name] if parent not in step)
    for objective in diagram.objectives:
        needed.update(parent for parent in objective.parents if parent not in step)
    names = [variable.name for variable in network.variables]
    order = kahn_topological_order(names, {name: set(axes) for name, axes in parents.items()})
    if len(order) != len(names):
        msg = f"Policies and status-quo CPTs create a cycle among {sorted(set(names) - set(order))}."
        raise ValueError(msg)
    state = [name for name in order if name in needed and name in structure.atemporal]
    state += [name for name in order if name in needed and name in structure.slice0]

    if diagram.initial_distribution is not None:
        if state != structure.state_nodes:
            msg = (
                f"Evaluating these policies needs an initial distribution over {state}, but initial_distribution "
                f"is over {structure.state_nodes}; give policies for the status-quo decisions."
            )
            raise ValueError(msg)
        mu = np.array(diagram.initial_distribution.values, dtype=float)
    else:
        mu = _initial_joint(factors, parents, state, decided)

    # An atemporal state node keeps its value; its next-step axis is tied to it by an identity.
    next_axes: list[Hashable] = []
    next_slicet: list[str] = []
    identities: list[Factor] = []
    for name in state:
        if name in structure.atemporal:
            next_axes.append((name, "next"))
            identities.append(([name, (name, "next")], np.eye(cardinality[name])))
        else:
            next_axes.append(f"{name[:-1]}t")
            next_slicet.append(f"{name[:-1]}t")
    transition = [factors[name] for name in sorted(_ancestors(next_slicet, parents, step))]
    # All-ones factors keep state axes the transition CPTs don't mention.
    state_axes = [([name], np.ones(cardinality[name])) for name in state]
    size = mu.size
    kernel = _contract(transition + identities + state_axes, [*state, *next_axes]).reshape(size, size)

    gamma = diagram.discount
    x = np.linalg.solve(np.eye(size) - gamma * kernel.T, (1 - gamma) * mu.reshape(size))
    x = x.reshape([cardinality[name] for name in state])

    values = {}
    for objective in diagram.objectives:
        objective_factors = [factors[name] for name in sorted(_ancestors(objective.parents, parents, step))]
        utility = (objective.parents, np.array(objective.utility.values, dtype=float))
        values[objective.symbol] = float(_contract([(state, x), *objective_factors, utility], []))

    occupation = _contract([(state, x)], structure.state_nodes)
    return StationaryEvaluation(
        objectives=values,
        occupation=Potential(variables=structure.state_nodes, shape=list(occupation.shape), values=occupation.tolist()),
    )


class StationaryIndexMap(BaseModel):
    """`compile_stationary_diagram`'s bookkeeping on top of the compiled step's `IndexMap`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    index_map: IndexMap
    """The compiled 2-slice step; pass it to `read_policy_from_solution`."""
    state_nodes: list[str]
    """`TwoSliceStructure.state_nodes`; flow balance `i` is their `i`-th joint state, mixed-radix."""
    next_of: dict[str, str]
    """`TwoSliceStructure.next_of`."""
    discount: float
    """`StationaryDiagram.discount`."""
    occupation_owner: str
    """The last state node, whose cluster holds the occupation measure x over `state_nodes`."""
    sink_owner: str
    """The sink node, whose cluster holds the slice-t state joint over `next_of`'s values."""


def _step_network(diagram: StationaryDiagram) -> DiscreteNetwork:
    """One 2-slice step: a uniform free chain over the state, slice t, the decisions and the state sink.

    The state nodes are declared first, so each state node's cluster holds only state nodes.
    Each state node takes all earlier state nodes as parents. Slice-0 and atemporal chance nodes
    that aren't state are dropped, and so are slice-t chance nodes that no next-step state node,
    objective, decision or constraint depends on. The sink's parents are the next-step state nodes
    and one node of each connected component they miss; its all-ones CPT keeps the semantics.
    """
    network = diagram.network
    structure = diagram.structure
    state_nodes = structure.state_nodes
    state = set(state_nodes)
    decisions = {decision.symbol for decision in diagram.decisions}
    next_nodes = [structure.next_of[name] for name in state_nodes]

    parents = {name: potential.variables[1:] for name, potential in network.potentials.items()}
    parents.update({decision.symbol: decision.information_set for decision in diagram.decisions})
    targets = [*next_nodes, *decisions]
    targets += [parent for objective in diagram.objectives for parent in objective.parents]
    targets += [node for constraint in diagram.constraints for node in constraint.nodes if node in parents]
    kept = state | _ancestors(targets, parents, structure.slicet | (structure.atemporal & decisions))
    by_name = {variable.name: variable for variable in network.variables}
    variables = [by_name[name] for name in state_nodes]
    variables += [variable for variable in network.variables if variable.name in kept - state]
    cardinality = {variable.name: len(variable.labels) for variable in variables}

    arcs = [(parent, child) for parent, child in network.arcs if child in kept - state and parent in kept]
    potentials = {name: potential for name, potential in network.potentials.items() if name in kept - state}
    for i, name in enumerate(state_nodes):
        parents[name] = state_nodes[:i]
        shape = [cardinality[name], *(cardinality[parent] for parent in parents[name])]
        arcs.extend((parent, name) for parent in parents[name])
        potentials[name] = Potential(
            variables=[name, *parents[name]], shape=shape, values=np.full(shape, 1.0 / shape[0]).tolist()
        )

    neighbours: dict[str, set[str]] = {name: set() for name in kept}
    for child in kept:
        for parent in parents[child]:
            neighbours[child].add(parent)
            neighbours[parent].add(child)
    sink_parents = list(next_nodes)
    reached = _ancestors(next_nodes, neighbours)
    for variable in variables:
        if variable.name not in reached:
            sink_parents.append(variable.name)
            reached |= _ancestors([variable.name], neighbours)

    variables.append(DiscreteVariable(name=_STATE_SINK, labels=["state"]))
    arcs.extend((name, _STATE_SINK) for name in sink_parents)
    shape = [1, *(cardinality[name] for name in sink_parents)]
    potentials[_STATE_SINK] = Potential(
        variables=[_STATE_SINK, *sink_parents], shape=shape, values=np.ones(shape).tolist()
    )

    return DiscreteNetwork(name=network.name, variables=variables, arcs=arcs, potentials=potentials)


def _marginal_exprs(
    index_map: IndexMap, owner: str, target_order: list[str], target_cardinalities: list[int]
) -> list[list]:
    """`Sum(Extract(mu, ...))` per joint state of `target_order` in `owner`'s cluster."""
    cluster = index_map.clusters[owner]
    groups = _group_flat_indices_by_projection(cluster.nodes, cluster.cardinalities, target_order, target_cardinalities)
    return [_extract_expr(cluster.mu_symbol, group) for group in groups]


def compile_stationary_diagram(
    diagram: StationaryDiagram,
    *,
    max_cluster_size: int | None = None,
    on_exceed: Literal["warn", "raise"] = "warn",
) -> tuple[Problem, StationaryIndexMap]:
    """Compile `diagram` into a MILP for its optimal deterministic stationary policy.

    One 2-slice step is compiled with `compile_diagram`, the slice-0 state joint left free as
    the occupation measure x. One flow-balance equality per joint state s of the state nodes,
    `x(s) = (1 - gamma) mu_init(s) + gamma P(state_t = s)`, ties it to the step's own slice-t
    state joint. The objectives are per-step averages under x, as in `evaluate_stationary_policy`;
    multiply by 1 / (1 - gamma) for the total discounted sum.

    Args:
        diagram: the stationary diagram.
        max_cluster_size: forwarded to `compile_diagram`.
        on_exceed: forwarded to `compile_diagram`.

    Returns:
        tuple[Problem, StationaryIndexMap]: the compiled MILP and its bookkeeping.

    Raises:
        ValueError: the diagram has no state nodes, a network node or objective/constraint
            symbol is named like the internal sink node, an objective/constraint symbol is named
            like a flow-balance constraint `flow_{i}`, or `compile_diagram` raises.
    """
    structure = diagram.structure
    state_nodes = structure.state_nodes
    if not state_nodes:
        msg = "The diagram has no state nodes, so nothing carries over between steps."
        raise ValueError(msg)
    symbols = {objective.symbol for objective in diagram.objectives}
    symbols |= {constraint.symbol for constraint in diagram.constraints}
    if _STATE_SINK in symbols | {variable.name for variable in diagram.network.variables}:
        msg = f"'{_STATE_SINK}' is reserved for the compiled state sink; rename that node or symbol."
        raise ValueError(msg)
    cardinality = {variable.name: len(variable.labels) for variable in diagram.network.variables}
    cardinalities = [cardinality[name] for name in state_nodes]
    flow_clashes = sorted(symbols & {f"flow_{i}" for i in range(prod(cardinalities))})
    if flow_clashes:
        msg = f"Symbols {flow_clashes} are reserved for the flow-balance constraints; rename them."
        raise ValueError(msg)

    network = promote_to_decisions(_step_network(diagram), diagram.decisions)
    annotation = DecisionAnnotation(
        decisions=diagram.decisions, objectives=diagram.objectives, constraints=diagram.constraints
    )
    problem, index_map = compile_diagram(
        network,
        annotation,
        max_cluster_size=max_cluster_size,
        on_exceed=on_exceed,
        free_nodes=frozenset(state_nodes),
    )

    for name in state_nodes:
        extra = sorted(set(index_map.clusters[name].nodes) - set(state_nodes))
        if extra:
            msg = f"Internal error: state node '{name}' shares its cluster with non-state nodes {extra}."
            raise ValueError(msg)

    occupation_owner = state_nodes[-1]
    current = _marginal_exprs(index_map, occupation_owner, state_nodes, cardinalities)
    following = _marginal_exprs(
        index_map, _STATE_SINK, [structure.next_of[name] for name in state_nodes], cardinalities
    )
    gamma = diagram.discount
    mu_init = initial_state_distribution(diagram).reshape(-1)
    flow = [
        Constraint(
            name=f"Flow balance at state {i}",
            symbol=f"flow_{i}",
            cons_type=ConstraintTypeEnum.EQ,
            func=["Subtract", ["Subtract", x, ["Multiply", gamma, x_next]], float((1 - gamma) * mu_init[i])],
            is_linear=True,
            is_convex=True,
            is_twice_differentiable=True,
        )
        for i, (x, x_next) in enumerate(zip(current, following, strict=True))
    ]

    return problem.add_constraints(flow), StationaryIndexMap(
        index_map=index_map,
        state_nodes=state_nodes,
        next_of=structure.next_of,
        discount=gamma,
        occupation_owner=occupation_owner,
        sink_owner=_STATE_SINK,
    )
