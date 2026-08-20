"""Bidirectional conversion between `DiscreteNetwork` and pyAgrum's `BayesNet`.

Also the import/export/learning entry points that go through pyAgrum: `load_network` and
`save_network` (pyAgrum's own readers/writers -- BIF, XDSL, `.net`, XMLBIF, ...) and
`learn_network` (a thin wrapper around `pyagrum.BNLearner`).

This is the only module in `desdeo/problem/graphical/` allowed to import pyAgrum -- see the
dependency rule in the plan's cross-cutting section. pyAgrum is an ordinary hard dependency
(like gurobipy/pyomo/cvxpy already are), not an optional extra -- see the design decisions
table in `.claude/desdeo-graphical-models-plan.md`.
"""

from pathlib import Path

import numpy as np
import pyagrum as gum

from desdeo.problem.graphical.network import (
    DiscreteNetwork,
    DiscreteVariable,
    DiscreteVariableKind,
    Potential,
)

_KIND_FROM_PYAGRUM = {
    gum.VarType_LABELIZED: DiscreteVariableKind.labelized,
    gum.VarType_RANGE: DiscreteVariableKind.range,
    gum.VarType_INTEGER: DiscreteVariableKind.integer,
    gum.VarType_DISCRETIZED: DiscreteVariableKind.discretized,
    gum.VarType_NUMERICAL: DiscreteVariableKind.numerical,
}


def _variable_from_pyagrum(variable: gum.DiscreteVariable) -> DiscreteVariable:
    """Convert one pyAgrum discrete variable to a `DiscreteVariable`, dispatching on `varType()`.

    `asRangeVar`/`asIntegerVar`/`asDiscretizedVar`/`asNumericalDiscreteVar` downcast a
    base-typed `DiscreteVariable` reference (what `BayesNet.variable()` returns) to the concrete
    subclass so its kind-specific accessors become available.
    """
    kind = _KIND_FROM_PYAGRUM.get(variable.varType())
    if kind is None:
        msg = f"Unsupported pyAgrum variable type {variable.varType()} for variable '{variable.name()}'."
        raise ValueError(msg)

    common = {
        "name": variable.name(),
        "description": variable.description(),
        "kind": kind,
        "labels": list(variable.labels()),
    }
    match kind:
        case DiscreteVariableKind.range:
            range_var = variable.asRangeVar()
            return DiscreteVariable(**common, lower_bound=range_var.minVal(), upper_bound=range_var.maxVal())
        case DiscreteVariableKind.integer:
            integer_var = variable.asIntegerVar()
            return DiscreteVariable(**common, integer_values=list(integer_var.integerDomain()))
        case DiscreteVariableKind.discretized:
            discretized_var = variable.asDiscretizedVar()
            return DiscreteVariable(**common, ticks=list(discretized_var.ticks()))
        case DiscreteVariableKind.numerical:
            numerical_var = variable.asNumericalDiscreteVar()
            return DiscreteVariable(**common, numerical_values=list(numerical_var.numericalDomain()))
        case DiscreteVariableKind.labelized:
            return DiscreteVariable(**common)


def _variable_to_pyagrum(variable: DiscreteVariable) -> gum.DiscreteVariable:
    """Convert one `DiscreteVariable` to the pyAgrum subclass its `kind` mirrors."""
    match variable.kind:
        case DiscreteVariableKind.labelized:
            pyagrum_var = gum.LabelizedVariable(variable.name, variable.description, 0)
            for label in variable.labels:
                pyagrum_var.addLabel(label)
            return pyagrum_var
        case DiscreteVariableKind.range:
            return gum.RangeVariable(variable.name, variable.description, variable.lower_bound, variable.upper_bound)
        case DiscreteVariableKind.integer:
            return gum.IntegerVariable(variable.name, variable.description, variable.integer_values)
        case DiscreteVariableKind.discretized:
            pyagrum_var = gum.DiscretizedVariable(variable.name, variable.description)
            for tick in variable.ticks:
                pyagrum_var.addTick(tick)
            return pyagrum_var
        case DiscreteVariableKind.numerical:
            return gum.NumericalDiscreteVariable(variable.name, variable.description, variable.numerical_values)


def _potential_from_pyagrum(tensor: gum.Tensor) -> Potential:
    """Convert one pyAgrum CPT to a `Potential`.

    `Tensor.toarray()`'s axis order is the reverse of `variablesSequence()`; transposing (which
    reverses all axes for an N-D array) brings it back in line with `variablesSequence()`'s
    node-then-parents order, which is the convention `Potential.variables` uses.
    """
    variables = [v.name() for v in tensor.variablesSequence()]
    values = np.transpose(tensor.toarray())
    return Potential(variables=variables, shape=list(values.shape), values=values.tolist())


def _fill_cpt(tensor: gum.Tensor, potential: Potential) -> None:
    """Fill a freshly-created pyAgrum CPT from a `Potential`.

    Positional, not name-aware: relies on `tensor.variablesSequence()` already matching
    `potential.variables`, which `to_pyagrum` guarantees by adding each node's arcs in its own
    `Potential`'s parent order before filling any CPTs.
    """
    actual = [v.name() for v in tensor.variablesSequence()]
    if actual != potential.variables:
        msg = (
            f"CPT variable order {actual} doesn't match Potential.variables {potential.variables}; "
            "arcs must be added in the same order the Potential's parents are listed in."
        )
        raise ValueError(msg)

    values = np.array(potential.values, dtype=float)
    # fillWith requires a C-contiguous array; transpose alone only produces a reversed-stride
    # view, so it must be copied back into contiguous memory before handing it to pyAgrum.
    tensor.fillWith(np.ascontiguousarray(np.transpose(values)))


def to_pyagrum(network: DiscreteNetwork) -> gum.BayesNet:
    """Convert a `DiscreteNetwork` to a pyAgrum `BayesNet`."""
    bn = gum.BayesNet(network.name)
    bn.setProperty("name", network.name)

    for variable in network.variables:
        bn.add(_variable_to_pyagrum(variable))

    # Add each node's arcs in its own Potential's parent order, not `network.arcs`'s list
    # order: pyAgrum records a node's parent order by arc-addition chronology, and `_fill_cpt`
    # below is positional, so this is what keeps CPT values landing on the right parent axis.
    for node_name, potential in network.potentials.items():
        for parent_name in potential.variables[1:]:
            bn.addArc(parent_name, node_name)

    for node_name, potential in network.potentials.items():
        _fill_cpt(bn.cpt(node_name), potential)

    return bn


def from_pyagrum(bn: gum.BayesNet) -> DiscreteNetwork:
    """Convert a pyAgrum `BayesNet` to a `DiscreteNetwork`.

    Nodes are listed in topological order. Each node's arcs are read off its own CPT's
    `variablesSequence()` rather than `BayesNet.arcs()` (an unordered set), so that a node's
    parent order -- and therefore `Potential.variables` order -- survives `to_pyagrum(from_pyagrum(bn))`.
    """
    order = list(bn.topologicalOrder())
    variables = [_variable_from_pyagrum(bn.variable(node_id)) for node_id in order]

    arcs: list[tuple[str, str]] = []
    potentials: dict[str, Potential] = {}
    for node_id in order:
        node_name = bn.variable(node_id).name()
        potential = _potential_from_pyagrum(bn.cpt(node_id))
        potentials[node_name] = potential
        arcs.extend((parent_name, node_name) for parent_name in potential.variables[1:])

    return DiscreteNetwork(
        name=bn.propertyWithDefault("name", ""),
        variables=variables,
        arcs=arcs,
        potentials=potentials,
    )


def load_network(path: str | Path, **kwargs) -> DiscreteNetwork:
    """Import a network from disk via pyAgrum's own readers.

    No hand-written parsers: `gum.loadBN` dispatches on the file extension. See
    `gum.availableBNExts()` for the supported suffixes (BIF, XDSL, `.net`, XMLBIF, and others).
    """
    return from_pyagrum(gum.loadBN(str(path), **kwargs))


def save_network(network: DiscreteNetwork, path: str | Path) -> None:
    """Export a `DiscreteNetwork` to disk via pyAgrum's own writers."""
    gum.saveBN(to_pyagrum(network), str(path))


def learn_network(
    data: str | Path,
    *,
    structure: DiscreteNetwork | None = None,
    missing_symbols: list[str] | None = None,
) -> DiscreteNetwork:
    """Learn a `DiscreteNetwork` from data via pyAgrum's `BNLearner`.

    If `structure` is given, only CPT parameters are relearned against that fixed DAG
    (`BNLearner.learnParameters`) -- this is the entry point the "decision structure lives in a
    separate overlay" design decision relies on: relearning CPTs from new data must not require
    redoing decision modelling, so passing the previous network back in as `structure` keeps the
    DAG (and therefore anything a `DecisionAnnotation` references on it) untouched. Otherwise
    both structure and parameters are learned from scratch, via pyAgrum's default search
    (greedy hill-climbing + BIC).

    Args:
        data: path to a CSV (or other pyAgrum-readable) dataset.
        structure: an existing network whose DAG to keep fixed while relearning CPTs.
        missing_symbols: strings in the data that denote a missing value.

    Returns:
        DiscreteNetwork: the learned network.
    """
    learner_kwargs = {"missingSymbols": missing_symbols} if missing_symbols is not None else {}

    if structure is not None:
        source_bn = to_pyagrum(structure)
        learner = gum.BNLearner(str(data), source_bn, **learner_kwargs)
        learned_bn = learner.learnParameters(source_bn)
    else:
        learner = gum.BNLearner(str(data), **learner_kwargs)
        learned_bn = learner.learnBN()

    return from_pyagrum(learned_bn)
