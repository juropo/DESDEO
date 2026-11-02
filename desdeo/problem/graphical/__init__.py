"""Compiler for graphical models (Bayesian networks, influence diagrams) into `Problem` instances.

See `.claude/desdeo-graphical-models-plan.md` for the design. This package is being built up
phase by phase; only what a given phase needs exists at any point in time.

`adapters.pyagrum` is deliberately not imported here (nor anywhere else in this package) even
though pyAgrum is an ordinary hard dependency -- see the dependency rule in the plan's
cross-cutting section. Import it explicitly: `from desdeo.problem.graphical.adapters.pyagrum
import to_pyagrum, from_pyagrum`.
"""

from .assumptions import (
    DiagramAssumptions,
    compile_with_assumptions,
    condition_network,
    fixed_decision_constraints,
    validate_assumptions,
)
from .compile import (
    ClusterIndexMap,
    DecisionIndexMap,
    DecisionPolicy,
    IndexMap,
    RiskMeasureIndexMap,
    build_outcome_distributions,
    compile_diagram,
    read_policy_from_solution,
)
from .network import DiscreteNetwork, DiscreteVariable, DiscreteVariableKind, Potential
from .rooted_junction_tree import (
    Cluster,
    ClusterSizeWarning,
    RootedJunctionTree,
    build_rooted_junction_tree,
    clear_rjt_cache,
    effective_information_sets,
)
from .schema import (
    CURRENT_SCHEMA_VERSION,
    DecisionAnnotation,
    DecisionNode,
    DiagramConstraint,
    DiagramConstraintKind,
    ObjectiveNode,
    RiskMeasure,
    RiskMeasureKind,
)

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "Cluster",
    "ClusterIndexMap",
    "ClusterSizeWarning",
    "DecisionAnnotation",
    "DecisionIndexMap",
    "DecisionNode",
    "DecisionPolicy",
    "DiagramAssumptions",
    "DiagramConstraint",
    "DiagramConstraintKind",
    "DiscreteNetwork",
    "DiscreteVariable",
    "DiscreteVariableKind",
    "IndexMap",
    "ObjectiveNode",
    "Potential",
    "RiskMeasure",
    "RiskMeasureIndexMap",
    "RiskMeasureKind",
    "RootedJunctionTree",
    "build_outcome_distributions",
    "build_rooted_junction_tree",
    "clear_rjt_cache",
    "compile_diagram",
    "compile_with_assumptions",
    "condition_network",
    "effective_information_sets",
    "fixed_decision_constraints",
    "read_policy_from_solution",
    "validate_assumptions",
]
