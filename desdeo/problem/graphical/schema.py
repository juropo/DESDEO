"""`DecisionAnnotation`: the decision/objective overlay on top of a `DiscreteNetwork`.

`DiscreteNetwork` holds only chance nodes and CPTs. Decisions, objectives, diagram-level
constraints and risk measures live here and reference network nodes by name, so CPTs can be
relearned (`learn_network(data, structure=...)`) without touching the annotation.

Elements are addressed by `symbol`, like `Variable`/`Objective`/`Constraint` in
`problem/schema.py`. Decision symbols and chance-node names must be disjoint.
"""

from collections import Counter
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from desdeo.problem.graphical.network import DiscreteNetwork, Potential
from desdeo.problem.schema import ConstraintTypeEnum

CURRENT_SCHEMA_VERSION = 1
"""Bumped on breaking changes to `DecisionAnnotation`'s schema."""


def _check_unique_symbols(context: str, symbols: list[str]) -> None:
    """Raise `ValueError` naming each symbol occurring more than once in `symbols`."""
    counts = Counter(symbols)
    duplicates = {symbol: count for symbol, count in counts.items() if count > 1}
    if not duplicates:
        return
    msg = f"Symbols must be unique {context}."
    for symbol, count in duplicates.items():
        msg += f" Symbol '{symbol}' occurs {count} times."
    raise ValueError(msg)


def _check_references_known(
    owner_label: str, owner_symbol: str, field_label: str, refs: list[str], known: set[str]
) -> None:
    """Raise `ValueError` naming any of `refs` not present in `known`."""
    unknown = [ref for ref in refs if ref not in known]
    if unknown:
        msg = f"{owner_label} '{owner_symbol}': {field_label} unknown node(s) {unknown}."
        raise ValueError(msg)


class DecisionNode(BaseModel):
    """A decision (choice) node. Not part of `DiscreteNetwork` -- it has no CPT, only a domain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    """Identifies this decision throughout the annotation. Must not clash with chance-node names."""
    name: str = ""
    """Optional descriptive name, for UI/visualisation. Mirrors `Objective.name`'s role."""
    states: list[str]
    """The decision's alternatives, in state-index order."""
    information_set: list[str] = Field(default_factory=list)
    """Chance-node names and/or decision symbols observed before this decision is taken."""

    @model_validator(mode="after")
    def _check_states(self) -> Self:
        if len(self.states) == 0:
            msg = f"Decision '{self.symbol}' must have a non-empty set of states."
            raise ValueError(msg)
        if len(set(self.states)) != len(self.states):
            msg = f"Decision '{self.symbol}' has duplicate states: {self.states}."
            raise ValueError(msg)
        if self.symbol in self.information_set:
            msg = f"Decision '{self.symbol}' cannot include itself in its own information set."
            raise ValueError(msg)
        return self


class ObjectiveNode(BaseModel):
    """A utility node: one linear objective, valued over its parents' joint state.

    Objectives are kept separate rather than summed into one utility as pyAgrum does.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    """Distinct namespace from `DiscreteNetwork` variable names."""
    name: str = ""
    parents: list[str]
    """Names of nodes this objective's value depends on: `DiscreteNetwork` chance-node names
    and/or `DecisionNode.symbol`s. Order matches `utility.variables` 1:1."""
    utility: Potential
    """The utility table over `parents`' joint state. `utility.variables` must equal `parents`;
    its shape is checked by `DecisionAnnotation.validate_against`."""
    maximize: bool = False
    """Whether this objective is to be maximized or minimized, as in `Objective.maximize`."""

    @model_validator(mode="after")
    def _check_utility_matches_parents(self) -> Self:
        if self.utility.variables != self.parents:
            msg = (
                f"Objective '{self.symbol}': utility.variables {self.utility.variables} must "
                f"match parents {self.parents} exactly (order included)."
            )
            raise ValueError(msg)
        if self.symbol in self.parents:
            msg = f"Objective '{self.symbol}' cannot depend on itself."
            raise ValueError(msg)
        if len(set(self.parents)) != len(self.parents):
            msg = f"Objective '{self.symbol}' has duplicate entries in parents: {self.parents}."
            raise ValueError(msg)
        return self


class DiagramConstraintKind(StrEnum):
    """Which quantity a `DiagramConstraint` is linear in."""

    decision = "decision"
    """Linear in z (decision indicators): budget, logic, exclusivity among alternatives. Free."""
    probability = "probability"
    """Linear in mu (cluster moments): an event-probability bound. Free."""
    expected_value = "expected_value"
    """Linear in mu: an expected-value bound. Free."""
    tail = "tail"
    """Distributional / tail constraint. Requires a risk-measure transform first."""


class DiagramConstraint(BaseModel):
    """A diagram-level constraint, expressed in terms of node symbols rather than compiled variables.

    `kind` determines how `func` is compiled.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    name: str = ""
    kind: DiagramConstraintKind
    nodes: list[str]
    """Chance-node names, decision symbols and/or objective symbols this constraint refers to."""
    func: list
    """MathJSON expression over `nodes`, in the same standard form `Constraint.func` uses --
    left-hand side only, compared against zero by `cons_type`."""
    cons_type: ConstraintTypeEnum = ConstraintTypeEnum.LTE

    @model_validator(mode="after")
    def _check_func_not_empty(self) -> Self:
        if len(self.func) == 0:
            msg = f"DiagramConstraint '{self.symbol}' must have a non-empty func."
            raise ValueError(msg)
        return self


class RiskMeasureKind(StrEnum):
    """Which risk measure to apply. Only `cvar` is implemented."""

    cvar = "cvar"


class RiskMeasure(BaseModel):
    """A risk transform recorded against one objective, applied to the diagram before compiling."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    target: str
    """`ObjectiveNode.symbol` this risk measure replaces/augments."""
    kind: RiskMeasureKind = RiskMeasureKind.cvar
    alpha: float
    """Confidence/quantile level, e.g. 0.95 for CVaR_0.95. Must be in (0, 1)."""

    @model_validator(mode="after")
    def _check_alpha(self) -> Self:
        if not 0.0 < self.alpha < 1.0:
            msg = f"RiskMeasure '{self.symbol}': alpha must be in (0, 1), got {self.alpha}."
            raise ValueError(msg)
        return self


class DecisionAnnotation(BaseModel):
    """The full decision/objective overlay for one `DiscreteNetwork`.

    Construction checks internal consistency only; references into a network are checked by
    `validate_against`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = CURRENT_SCHEMA_VERSION
    """Bumped on breaking changes; lets a persisted annotation be checked for compatibility
    before being reapplied to a relearned/reloaded network."""
    decisions: list[DecisionNode] = Field(default_factory=list)
    objectives: list[ObjectiveNode]
    constraints: list[DiagramConstraint] = Field(default_factory=list)
    risk_measures: list[RiskMeasure] = Field(default_factory=list)
    decision_order: list[str] | None = None
    """Temporal order of `DecisionNode.symbol`s. `None` => LIMID semantics: only each decision's
    own `information_set` matters, no total order is assumed. A list enforces classical
    influence-diagram semantics, and must be a permutation of `decisions`' symbols."""
    no_forgetting: bool = False
    """If true (only meaningful when `decision_order` is set), each decision's information set
    also includes every earlier decision and its parents, as pyAgrum's `addNoForgettingAssumption`."""

    @model_validator(mode="after")
    def _check_internal_structure(self) -> Self:
        decision_symbols = [d.symbol for d in self.decisions]
        objective_symbols = [o.symbol for o in self.objectives]
        constraint_symbols = [c.symbol for c in self.constraints]
        risk_symbols = [r.symbol for r in self.risk_measures]

        for label, symbols in (
            ("decisions", decision_symbols),
            ("objectives", objective_symbols),
            ("constraints", constraint_symbols),
            ("risk_measures", risk_symbols),
        ):
            _check_unique_symbols(f"within {label}", symbols)

        all_symbols = decision_symbols + objective_symbols + constraint_symbols + risk_symbols
        _check_unique_symbols("across decisions/objectives/constraints/risk_measures", all_symbols)

        if len(objective_symbols) == 0:
            msg = "DecisionAnnotation must have at least one objective."
            raise ValueError(msg)

        # Chance-node references are checked in `validate_against`.

        objective_symbol_set = set(objective_symbols)
        for risk_measure in self.risk_measures:
            if risk_measure.target not in objective_symbol_set:
                msg = f"RiskMeasure '{risk_measure.symbol}' targets unknown objective '{risk_measure.target}'."
                raise ValueError(msg)

        if self.decision_order is not None:
            if sorted(self.decision_order) != sorted(decision_symbols):
                msg = (
                    f"decision_order {self.decision_order} must be a permutation of the decision "
                    f"symbols {decision_symbols}."
                )
                raise ValueError(msg)
        elif self.no_forgetting:
            msg = "no_forgetting requires decision_order to be set (LIMID semantics have no total order)."
            raise ValueError(msg)

        return self

    def validate_against(self, network: DiscreteNetwork) -> None:
        """Raise `ValueError` if any node reference or state-set doesn't match `network`.

        Checks existence of every reference and utility-table shapes against cardinalities. CPT
        values are not read, so a relearned network with the same structure still validates.
        """
        chance_cardinality = {var.name: len(var.labels) for var in network.variables}
        decision_cardinality = {d.symbol: len(d.states) for d in self.decisions}
        objective_symbols = {o.symbol for o in self.objectives}

        # Decision symbols and chance-node names must be disjoint.
        collisions = set(chance_cardinality) & set(decision_cardinality)
        if collisions:
            msg = f"Decision symbols collide with network variable names: {sorted(collisions)}."
            raise ValueError(msg)

        observable = set(chance_cardinality) | set(decision_cardinality)
        cardinalities = {**chance_cardinality, **decision_cardinality}

        for decision in self.decisions:
            _check_references_known(
                "Decision", decision.symbol, "information_set references", decision.information_set, observable
            )

        for objective in self.objectives:
            _check_references_known("Objective", objective.symbol, "parents reference", objective.parents, observable)

            expected_shape = [cardinalities[parent] for parent in objective.parents]
            if objective.utility.shape != expected_shape:
                msg = (
                    f"Objective '{objective.symbol}': utility has shape {objective.utility.shape}, "
                    f"expected {expected_shape} from its parents' domains."
                )
                raise ValueError(msg)

        # Decision axes of chance-node potentials aren't checked by `DiscreteNetwork`.
        for variable in network.variables:
            potential = network.potentials[variable.name]
            external_parents = [name for name in potential.variables[1:] if name not in chance_cardinality]
            _check_references_known(
                "Chance node",
                variable.name,
                "potential's external parents reference",
                external_parents,
                decision_cardinality,
            )

            expected_full_shape = [cardinalities[name] for name in potential.variables]
            if potential.shape != expected_full_shape:
                msg = (
                    f"Chance node '{variable.name}': potential has shape {potential.shape}, "
                    f"expected {expected_full_shape} from its variables' domains."
                )
                raise ValueError(msg)

        referenceable = observable | objective_symbols
        for constraint in self.constraints:
            _check_references_known("Constraint", constraint.symbol, "nodes reference", constraint.nodes, referenceable)
