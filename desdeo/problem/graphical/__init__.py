"""Compiler for graphical models (Bayesian networks, influence diagrams) into `Problem` instances.

See `.claude/desdeo-graphical-models-plan.md` for the design. This package is being built up
phase by phase; only what a given phase needs exists at any point in time.

`adapters.pyagrum` is deliberately not imported here (nor anywhere else in this package) even
though pyAgrum is an ordinary hard dependency -- see the dependency rule in the plan's
cross-cutting section. Import it explicitly: `from desdeo.problem.graphical.adapters.pyagrum
import to_pyagrum, from_pyagrum`.
"""

from .compile import ClusterIndexMap, DecisionIndexMap, IndexMap, compile_diagram
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
    "DiagramConstraint",
    "DiagramConstraintKind",
    "DiscreteNetwork",
    "DiscreteVariable",
    "DiscreteVariableKind",
    "IndexMap",
    "ObjectiveNode",
    "Potential",
    "RiskMeasure",
    "RiskMeasureKind",
    "RootedJunctionTree",
    "build_rooted_junction_tree",
    "clear_rjt_cache",
    "compile_diagram",
    "effective_information_sets",
]
