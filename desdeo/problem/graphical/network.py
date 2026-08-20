"""`DiscreteNetwork`: a near-isomorphic pydantic mirror of pyAgrum's Bayesian network model.

Phase 1 of `.claude/desdeo-graphical-models-plan.md`. Mirrors pyAgrum's own node categories
(`LabelizedVariable`, `RangeVariable`, `IntegerVariable`, `DiscretizedVariable`,
`NumericalDiscreteVariable`), potential (CPT) indexing order, and label conventions, so that
`adapters/pyagrum.to_pyagrum`/`from_pyagrum` round-trip losslessly. A plain pydantic model is
used rather than pyAgrum's own (SWIG-wrapped) objects because SWIG objects can't be pydantic
fields, DB rows, or JSON for the frontend.

This module does not import pyAgrum — see the package-level dependency rule in the plan's
cross-cutting section.
"""

from enum import StrEnum
from itertools import product
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


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

    `labels` always holds the domain in state-index order (what pyAgrum's own `.labels()`
    returns for any of its variable subclasses, regardless of kind) and is what CPT/`Potential`
    indexing refers to. The kind-specific fields below carry exactly what's needed to
    reconstruct the same pyAgrum subclass rather than always falling back to a generic
    labelized variable.
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

    `variables` follows pyAgrum's own per-CPT ordering: the node itself first, then its parents
    in the order pyAgrum lists them (arc-insertion order). `values`'s axes match `variables`
    1:1 in that same order -- note this is the *reverse* of pyAgrum's own `Tensor.toarray()`,
    whose axis order is reversed relative to its `variablesSequence()`; the adapter transposes
    between the two conventions so this schema's axis order stays intuitive to read.
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

        actual_shape = np.array(self.values, dtype=float).shape
        if actual_shape != tuple(self.shape):
            msg = f"Potential over {self.variables}: values has shape {actual_shape}, expected {tuple(self.shape)}."
            raise ValueError(msg)

        return self


class DiscreteNetwork(BaseModel):
    """A discrete Bayesian network: chance nodes, arcs between them, and one CPT per node.

    Near-isomorphic to pyAgrum's `BayesNet` -- see `adapters/pyagrum.py` for the conversion.
    Nodes are addressed by name throughout, mirroring how the rest of DESDEO addresses `Problem`
    elements by `symbol` rather than by a separate integer id.
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
            if set(potential.variables[1:]) != parents_of[node]:
                msg = (
                    f"Potential for '{node}' covers parents {set(potential.variables[1:])}, "
                    f"but the network's arcs give it parents {parents_of[node]}."
                )
                raise ValueError(msg)
            expected_shape = [cardinalities[var_name] for var_name in potential.variables]
            if potential.shape != expected_shape:
                msg = (
                    f"Potential for '{node}' has shape {potential.shape}, "
                    f"expected {expected_shape} from variable domains."
                )
                raise ValueError(msg)

        return self

    @staticmethod
    def _has_cycle(names: set[str], arcs: list[tuple[str, str]]) -> bool:
        """Kahn's algorithm; cheap enough at construction time and independent of pyAgrum."""
        children: dict[str, list[str]] = {name: [] for name in names}
        indegree: dict[str, int] = dict.fromkeys(names, 0)
        for parent, child in arcs:
            children[parent].append(child)
            indegree[child] += 1

        queue = [name for name, degree in indegree.items() if degree == 0]
        visited = 0
        while queue:
            node = queue.pop()
            visited += 1
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)

        return visited != len(names)

    def joint_state_count(self) -> int:
        """Total number of joint states over all variables (product of all cardinalities)."""
        return int(np.prod([len(var.labels) for var in self.variables])) if self.variables else 0

    def all_states(self, variable_names: list[str]) -> list[tuple[int, ...]]:
        """All joint 0-based state tuples over `variable_names`, in the order given."""
        cardinalities = {var.name: len(var.labels) for var in self.variables}
        return list(product(*(range(cardinalities[name]) for name in variable_names)))
