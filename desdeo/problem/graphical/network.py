"""`DiscreteNetwork`: a pydantic mirror of pyAgrum's Bayesian network model.

Mirrors pyAgrum's variable kinds, CPT indexing and label conventions so that
`adapters/pyagrum.to_pyagrum`/`from_pyagrum` round-trip losslessly, while staying serialisable
for the DB and frontend. Does not import pyAgrum.
"""

import heapq
from enum import StrEnum
from itertools import product
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


def kahn_topological_order(
    nodes: list[str], predecessors: dict[str, set[str]], priority: dict[str, int] | None = None
) -> list[str]:
    """Kahn's algorithm: as much of a topological order over `nodes` as `predecessors` allows.

    Args:
        nodes: every node to order; `predecessors` may omit nodes with no predecessors.
        predecessors: `predecessors[node]` is the set of nodes that must come before `node`.
        priority: tiebreaker for nodes that become ready at the same time, ascending (lower
            first); declaration order (`nodes`'s own order) if omitted, for reproducible output.

    Returns:
        list[str]: as many nodes as could be topologically ordered. If shorter than `nodes`, the
            nodes missing from the result participate in a cycle among `predecessors` --
            `set(nodes) - set(result)` names them.
    """
    if priority is None:
        priority = {node: i for i, node in enumerate(nodes)}

    remaining = {node: set(predecessors.get(node, ())) for node in nodes}
    successors: dict[str, list[str]] = {node: [] for node in nodes}
    for node, preds in remaining.items():
        for pred in preds:
            successors[pred].append(node)

    ready = [(priority[node], node) for node in nodes if not remaining[node]]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        _, node = heapq.heappop(ready)
        order.append(node)
        for successor in successors[node]:
            remaining[successor].discard(node)
            if not remaining[successor]:
                heapq.heappush(ready, (priority[successor], successor))

    return order


class DiscreteVariableKind(StrEnum):
    """Mirrors pyAgrum's `VarType` discriminator for `DiscreteVariable` subclasses."""

    labelized = "labelized"
    """An arbitrary, unordered set of string labels (pyAgrum's `LabelizedVariable`)."""
    range = "range"
    """A contiguous integer range (pyAgrum's `RangeVariable`)."""
    integer = "integer"
    """An explicit, ordered set of integers (pyAgrum's `IntegerVariable`)."""
    discretized = "discretized"
    """A continuous domain cut into bins by a set of ticks (pyAgrum's `DiscretizedVariable`)."""
    numerical = "numerical"
    """An explicit, ordered set of floats (pyAgrum's `NumericalDiscreteVariable`)."""


class DiscreteVariable(BaseModel):
    """A discrete chance node's domain, mirroring one pyAgrum `DiscreteVariable` subclass.

    `labels` holds the domain in state-index order; the kind-specific fields allow reconstructing
    the same pyAgrum subclass.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    """Node name. Used as the node's identity throughout `DiscreteNetwork` (no separate id)."""
    description: str = ""
    """Optional longer description, mirroring pyAgrum's per-variable description string."""
    kind: DiscreteVariableKind = DiscreteVariableKind.labelized
    """Which pyAgrum `DiscreteVariable` subclass this mirrors."""
    labels: list[str]
    """Domain labels in state-index order. Always present, regardless of `kind`."""
    lower_bound: int | None = None
    """`kind == range` only: the range's inclusive lower bound."""
    upper_bound: int | None = None
    """`kind == range` only: the range's inclusive upper bound."""
    integer_values: list[int] | None = None
    """`kind == integer` only: the explicit, ordered domain."""
    ticks: list[float] | None = None
    """`kind == discretized` only: the bin edges (`len(labels) == len(ticks) - 1`)."""
    numerical_values: list[float] | None = None
    """`kind == numerical` only: the explicit, ordered domain."""

    @model_validator(mode="after")
    def _check_kind_fields(self) -> Self:
        """Every kind-specific field set must be present and internally consistent."""
        if len(self.labels) == 0:
            msg = f"Variable '{self.name}' must have a non-empty domain."
            raise ValueError(msg)

        match self.kind:
            case DiscreteVariableKind.range:
                if self.lower_bound is None or self.upper_bound is None:
                    msg = f"Variable '{self.name}' is kind 'range' but lacks lower_bound/upper_bound."
                    raise ValueError(msg)
                if self.upper_bound - self.lower_bound + 1 != len(self.labels):
                    msg = f"Variable '{self.name}': range bounds don't match the number of labels."
                    raise ValueError(msg)
            case DiscreteVariableKind.integer:
                if self.integer_values is None:
                    msg = f"Variable '{self.name}' is kind 'integer' but lacks integer_values."
                    raise ValueError(msg)
                if len(self.integer_values) != len(self.labels):
                    msg = f"Variable '{self.name}': integer_values doesn't match the number of labels."
                    raise ValueError(msg)
            case DiscreteVariableKind.discretized:
                if self.ticks is None:
                    msg = f"Variable '{self.name}' is kind 'discretized' but lacks ticks."
                    raise ValueError(msg)
                if len(self.ticks) - 1 != len(self.labels):
                    msg = f"Variable '{self.name}': ticks don't match the number of labels (bins)."
                    raise ValueError(msg)
            case DiscreteVariableKind.numerical:
                if self.numerical_values is None:
                    msg = f"Variable '{self.name}' is kind 'numerical' but lacks numerical_values."
                    raise ValueError(msg)
                if len(self.numerical_values) != len(self.labels):
                    msg = f"Variable '{self.name}': numerical_values doesn't match the number of labels."
                    raise ValueError(msg)
            case DiscreteVariableKind.labelized:
                pass

        return self


class Potential(BaseModel):
    """A conditional probability table over one or more `DiscreteVariable`s.

    `variables` is the node itself first, then its parents in arc-insertion order. `values`'s axes
    match `variables` (the reverse of pyAgrum's `Tensor.toarray()`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    variables: list[str]
    """Variable names, node-then-parents order. `values`'s axes follow this order 1:1."""
    shape: list[int]
    """Cardinality per axis, same order as `variables`."""
    values: list
    """Nested list of probabilities; `np.array(values).shape == tuple(shape)`."""

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        if len(self.variables) != len(self.shape):
            msg = f"Potential over {self.variables}: shape {self.shape} has the wrong number of axes."
            raise ValueError(msg)

        if len(set(self.variables)) != len(self.variables):
            msg = f"Potential's variables must be unique, got {self.variables}."
            raise ValueError(msg)

        actual_shape = np.array(self.values, dtype=float).shape
        if actual_shape != tuple(self.shape):
            msg = f"Potential over {self.variables}: values has shape {actual_shape}, expected {tuple(self.shape)}."
            raise ValueError(msg)

        return self


class DiscreteNetwork(BaseModel):
    """A discrete Bayesian network: chance nodes, arcs between them, and one CPT per node.

    Near-isomorphic to pyAgrum's `BayesNet`; see `adapters/pyagrum.py`. Nodes are addressed by name.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = ""
    variables: list[DiscreteVariable]
    arcs: list[tuple[str, str]]
    """`(parent, child)` name pairs."""
    potentials: dict[str, Potential]
    """One CPT per node, keyed by the node's own name (the CPT's first `variables` entry)."""

    @model_validator(mode="after")
    def _check_structure(self) -> Self:
        names = [var.name for var in self.variables]
        if len(names) != len(set(names)):
            msg = f"Variable names must be unique, got: {names}."
            raise ValueError(msg)
        name_set = set(names)
        cardinalities = {var.name: len(var.labels) for var in self.variables}

        for parent, child in self.arcs:
            if parent not in name_set:
                msg = f"Arc references unknown parent '{parent}'."
                raise ValueError(msg)
            if child not in name_set:
                msg = f"Arc references unknown child '{child}'."
                raise ValueError(msg)
            if parent == child:
                msg = f"Arc '{parent}' -> '{child}' is a self-loop."
                raise ValueError(msg)

        if self._has_cycle(name_set, self.arcs):
            msg = "Arcs form a cycle; DiscreteNetwork must be a DAG."
            raise ValueError(msg)

        if set(self.potentials) != name_set:
            msg = (
                f"Every variable must have exactly one potential; "
                f"variables={name_set}, potentials={set(self.potentials)}."
            )
            raise ValueError(msg)

        parents_of: dict[str, set[str]] = {name: set() for name in names}
        for parent, child in self.arcs:
            parents_of[child].add(parent)

        for node, potential in self.potentials.items():
            if potential.variables[0] != node:
                msg = (
                    f"Potential for '{node}' must list '{node}' as its own (first) variable, got {potential.variables}."
                )
                raise ValueError(msg)

            # Names outside this network (e.g. decision symbols) are checked by
            # `DecisionAnnotation.validate_against`.
            known_names = [var_name for var_name in potential.variables if var_name in name_set]
            known_parents = set(known_names[1:])
            if known_parents != parents_of[node]:
                msg = (
                    f"Potential for '{node}' covers known chance parents {known_parents}, "
                    f"but the network's arcs give it parents {parents_of[node]}."
                )
                raise ValueError(msg)

            known_shape = [
                shape
                for var_name, shape in zip(potential.variables, potential.shape, strict=True)
                if var_name in name_set
            ]
            expected_known_shape = [cardinalities[var_name] for var_name in known_names]
            if known_shape != expected_known_shape:
                msg = (
                    f"Potential for '{node}' has shape {potential.shape} over {potential.variables}, but axes for "
                    f"this network's own variables {known_names} expected {expected_known_shape}."
                )
                raise ValueError(msg)

        return self

    @staticmethod
    def _has_cycle(names: set[str], arcs: list[tuple[str, str]]) -> bool:
        """`kahn_topological_order`; cheap enough at construction time and independent of pyAgrum."""
        predecessors: dict[str, set[str]] = {name: set() for name in names}
        for parent, child in arcs:
            predecessors[child].add(parent)

        order = kahn_topological_order(sorted(names), predecessors)
        return len(order) != len(names)

    def joint_state_count(self) -> int:
        """Total number of joint states over all variables (product of all cardinalities)."""
        return int(np.prod([len(var.labels) for var in self.variables])) if self.variables else 0

    def all_states(self, variable_names: list[str]) -> list[tuple[int, ...]]:
        """All joint 0-based state tuples over `variable_names`, in the order given."""
        cardinalities = {var.name: len(var.labels) for var in self.variables}
        return list(product(*(range(cardinalities[name]) for name in variable_names)))
