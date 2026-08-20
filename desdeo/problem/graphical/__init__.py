"""Compiler for graphical models (Bayesian networks, influence diagrams) into `Problem` instances.

See `.claude/desdeo-graphical-models-plan.md` for the design. This package is being built up
phase by phase; only what a given phase needs exists at any point in time.

`adapters.pyagrum` is deliberately not imported here (nor anywhere else in this package) even
though pyAgrum is an ordinary hard dependency -- see the dependency rule in the plan's
cross-cutting section. Import it explicitly: `from desdeo.problem.graphical.adapters.pyagrum
import to_pyagrum, from_pyagrum`.
"""

from .annotation import (
    CURRENT_SCHEMA_VERSION,
    DecisionAnnotation,
    DecisionNode,
    DiagramConstraint,
    DiagramConstraintKind,
    ObjectiveNode,
    RiskMeasure,
    RiskMeasureKind,
)
from .network import DiscreteNetwork, DiscreteVariable, DiscreteVariableKind, Potential
from .rjt import Cluster, ClusterSizeWarning, RootedJunctionTree, build_rooted_junction_tree, clear_rjt_cache

__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "Cluster",
    "ClusterSizeWarning",
    "DecisionAnnotation",
    "DecisionNode",
    "DiagramConstraint",
    "DiagramConstraintKind",
    "DiscreteNetwork",
    "DiscreteVariable",
    "DiscreteVariableKind",
    "ObjectiveNode",
    "Potential",
    "RiskMeasure",
    "RiskMeasureKind",
    "RootedJunctionTree",
    "build_rooted_junction_tree",
    "clear_rjt_cache",
]
