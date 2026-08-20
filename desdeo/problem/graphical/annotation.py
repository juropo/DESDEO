"""`DecisionAnnotation`: the decision/objective overlay on top of a `DiscreteNetwork`.

Phase 2 of `.claude/desdeo-graphical-models-plan.md`. `DiscreteNetwork` (Phase 1) is a pure
Bayesian network: chance nodes and CPTs only, near-isomorphic to pyAgrum's `BayesNet`. Decision
structure -- which choices exist, what an objective values, diagram-level constraints, and risk
transforms -- lives here instead, referencing network nodes **by name** rather than embedding
into the network itself. That separation is what lets `learn_network(data, structure=...)`
(Phase 1) relearn CPTs from new data without disturbing decision modelling: the same
`DecisionAnnotation` instance is expected to keep validating against a network whose DAG and
variable domains are unchanged but whose potentials have been relearned.

Unlike `DiscreteNetwork`, this module's schemas aren't mirroring a pyAgrum data model, so they
follow the rest of DESDEO's `Problem` convention instead: elements are addressed by `symbol`
(matching `Variable`/`Objective`/`Constraint` in `problem/schema.py`), not by `name` the way
`DiscreteNetwork`'s chance nodes are. `DiagramConstraint` reuses `ConstraintTypeEnum` from
`problem/schema.py` for the same reason -- it is compiled into an ordinary `Constraint` in
Phase 4/5, so the standard-form convention (`func <=/= 0`) should already match at this layer.

This module does not import pyAgrum -- see the package-level dependency rule in the plan's
cross-cutting section. It does import from `problem.schema`; that dependency runs one way only
(nothing in `problem/schema.py` imports `problem/graphical`).
"""

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from desdeo.problem.graphical.network import DiscreteNetwork, Potential
from desdeo.problem.schema import ConstraintTypeEnum

CURRENT_SCHEMA_VERSION = 1
"""Bumped on breaking changes to `DecisionAnnotation`'s schema."""


class DecisionNode(BaseModel):
    """A decision (choice) node. Not part of `DiscreteNetwork` -- it has no CPT, only a domain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    """Identifies this decision throughout the annotation (`decision_order`, other nodes'
    `information_set`/`parents`, `DiagramConstraint.nodes`). Distinct namespace from
    `DiscreteNetwork` variable names -- see the module docstring."""
    name: str = ""
    """Optional descriptive name, for UI/visualisation. Mirrors `Objective.name`'s role."""
    states: list[str]
    """The decision's alternatives, in state-index order -- plays the same role `labels` plays
    on `DiscreteVariable`, since a decision node has a domain but no network-side counterpart."""
    information_set: list[str] = Field(default_factory=list)
    """Names of nodes observed before this decision is taken: `DiscreteNetwork` chance-node
    names and/or other decisions' `symbol`s. Empty for a decision made with no prior
    observations. Resolved and validated against a network by `DecisionAnnotation.validate_against`."""

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

    Deliberately kept singular and vector-friendly -- `DecisionAnnotation` holds a *list* of
    these rather than following pyAgrum's influence-diagram convention of implicitly summing
    every utility node into one scalar (`ShaferShenoyLIMIDInference.MEU()` does exactly that;
    confirmed empirically -- see the plan's Phase 2 findings). Multiobjective handling stays with
    DESDEO's existing methods, one objective per `ObjectiveNode` (design decisions table).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    """Distinct namespace from `DiscreteNetwork` variable names -- see the module docstring."""
    name: str = ""
    parents: list[str]
    """Names of nodes this objective's value depends on: `DiscreteNetwork` chance-node names
    and/or `DecisionNode.symbol`s. Order matches `utility.variables` 1:1."""
    utility: Potential
    """The utility table over `parents`' joint state. Reuses `Potential`'s tensor+shape schema
    (variables/shape/values) -- nothing in `Potential` requires its values to be a normalised
    probability, so the same shape-checked container serves both roles. `utility.variables` must
    equal `parents` exactly (order included); `utility.shape` is checked against the parents'
    actual cardinalities by `DecisionAnnotation.validate_against`, since cardinalities live on
    the network (or on a sibling `DecisionNode`), not on this schema alone."""
    maximize: bool = False
    """Whether this objective is to be maximized or minimized -- mirrors `Objective.maximize`
    in `problem/schema.py`, since Phase 4 compiles one `Objective` per `ObjectiveNode`."""

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
        return self


class DiagramConstraintKind(StrEnum):
    """Which quantity a `DiagramConstraint` is linear in, per the Phase 5 constraint taxonomy.

    Recorded here (Phase 2) so it can be surfaced in the API ahead of Phase 5 actually compiling
    against it; the taxonomy itself doesn't change.
    """

    decision = "decision"
    """Linear in z (decision indicators): budget, logic, exclusivity among alternatives. Free."""
    probability = "probability"
    """Linear in mu (cluster moments): an event-probability bound. Free."""
    expected_value = "expected_value"
    """Linear in mu: an expected-value bound. Free."""
    tail = "tail"
    """Distributional / tail constraint. Requires a Phase 6 risk-measure transform first."""


class DiagramConstraint(BaseModel):
    """A diagram-level constraint, expressed in terms of node symbols rather than compiled variables.

    mu/z variables only exist once Phase 4 compiles a specific `RootedJunctionTree`. `kind` fixes
    which of the Phase 5 taxonomy rows this constraint belongs to; `func` is resolved against that
    taxonomy at compile time.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    name: str = ""
    kind: DiagramConstraintKind
    nodes: list[str]
    """Node names/symbols this constraint refers to -- `DiscreteNetwork` chance-node names,
    `DecisionNode.symbol`s, and/or `ObjectiveNode.symbol`s. Validated for existence by
    `DecisionAnnotation.validate_against`, same as every other cross-reference in this schema."""
    func: list
    """MathJSON expression over `nodes`, in the same standard form `Constraint.func` uses --
    left-hand side only, compared against zero by `cons_type`."""
    cons_type: ConstraintTypeEnum = ConstraintTypeEnum.LTE


class RiskMeasureKind(StrEnum):
    """Which risk measure to apply.

    Only `cvar` is implemented end to end (Phase 6); kept as an enum so the schema doesn't need
    to change shape when a second risk measure is added.
    """

    cvar = "cvar"


class RiskMeasure(BaseModel):
    """A risk transform recorded against one objective, applied before Phase 4 emits variables.

    Applied as a transformation of the diagram/RJT, not as a constraint appended afterward -- see
    the plan's Phase 6 section. This schema only records intent; the transformation itself is
    Phase 6 work.
    """

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

    Contains nothing the network already provides -- no chance-node domains or CPTs, only what
    turns a Bayesian network into an influence diagram: decisions, objectives, diagram-level
    constraints, and risk measures, plus the temporal/memory assumptions governing them.

    Construction validates internal consistency only (symbol uniqueness, cross-references among
    this annotation's own decisions/objectives/constraints/risk measures). Cross-references into
    a `DiscreteNetwork` -- existence and state-set matching -- are deliberately *not* checked
    here; see `validate_against`, which is the entry point the "relearn CPTs, reapply the same
    annotation" round trip depends on.
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
    is treated as also including every earlier decision (per `decision_order`) and its parents --
    mirrors pyAgrum's `addNoForgettingAssumption`. Left for Phase 3/4 to apply; recorded here
    since it's a property of the decision structure, not of the RJT or compiler."""

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
            if len(symbols) != len(set(symbols)):
                msg = f"Duplicate symbols within {label}: {symbols}."
                raise ValueError(msg)

        all_symbols = decision_symbols + objective_symbols + constraint_symbols + risk_symbols
        if len(all_symbols) != len(set(all_symbols)):
            msg = f"Symbols must be unique across decisions/objectives/constraints/risk_measures, got: {all_symbols}."
            raise ValueError(msg)

        if len(objective_symbols) == 0:
            msg = "DecisionAnnotation must have at least one objective."
            raise ValueError(msg)

        # Decision `information_set`/objective `parents` may reference chance-node names, which
        # aren't known here (no network available) -- that existence check is `validate_against`'s
        # job, once a network is on hand. Only symbol-level self-consistency happens here.

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

        Checks every cross-reference this annotation makes into `network` (chance-node names)
        and within itself (decision/objective symbols): existence, and -- for `ObjectiveNode`
        utility tables -- that shapes match the referenced nodes' actual cardinalities. This is
        the entry point the "relearn CPTs from new data, reapply the same annotation" round trip
        depends on: a network with the same DAG and variable domains but freshly relearned CPTs
        must still validate, since none of what's checked here reads potentials.
        """
        chance_cardinality = {var.name: len(var.labels) for var in network.variables}
        decision_cardinality = {d.symbol: len(d.states) for d in self.decisions}
        objective_symbols = {o.symbol for o in self.objectives}

        # A decision's own symbol namespace and the network's chance-node namespace are
        # expected to be disjoint (see the module docstring); enforce that here, since it's only
        # checkable once both are on hand together.
        collisions = set(chance_cardinality) & set(decision_cardinality)
        if collisions:
            msg = f"Decision symbols collide with network variable names: {sorted(collisions)}."
            raise ValueError(msg)

        observable = set(chance_cardinality) | set(decision_cardinality)
        for decision in self.decisions:
            unknown = [ref for ref in decision.information_set if ref not in observable]
            if unknown:
                msg = f"Decision '{decision.symbol}': information_set references unknown node(s) {unknown}."
                raise ValueError(msg)

        for objective in self.objectives:
            unknown = [ref for ref in objective.parents if ref not in observable]
            if unknown:
                msg = f"Objective '{objective.symbol}': parents reference unknown node(s) {unknown}."
                raise ValueError(msg)

            cardinalities = {**chance_cardinality, **decision_cardinality}
            expected_shape = [cardinalities[parent] for parent in objective.parents]
            if objective.utility.shape != expected_shape:
                msg = (
                    f"Objective '{objective.symbol}': utility has shape {objective.utility.shape}, "
                    f"expected {expected_shape} from its parents' domains."
                )
                raise ValueError(msg)

        referenceable = observable | objective_symbols
        for constraint in self.constraints:
            unknown = [ref for ref in constraint.nodes if ref not in referenceable]
            if unknown:
                msg = f"Constraint '{constraint.symbol}': nodes reference unknown node(s) {unknown}."
                raise ValueError(msg)
