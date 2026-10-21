"""Conversion between `DiscreteNetwork` and pyAgrum's `BayesNet`/`InfluenceDiagram`.

Also pyAgrum-backed I/O and learning: `load_network`, `save_network`, `learn_network`. This is the
only module in `desdeo/problem/graphical/` that imports pyAgrum.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyagrum as gum

from desdeo.problem.graphical.network import (
    DiscreteNetwork,
    DiscreteVariable,
    DiscreteVariableKind,
    Potential,
)
from desdeo.problem.graphical.schema import DecisionAnnotation


@dataclass(frozen=True)
class _VariableKindAdapter:
    """Everything needed to convert one `DiscreteVariableKind` to/from a pyAgrum variable."""

    pyagrum_type: int
    """The `gum.VarType_*` constant this kind mirrors."""
    from_pyagrum: Callable[[gum.DiscreteVariable], dict]
    """Given a pyAgrum variable already known to be this kind, the kind-specific `DiscreteVariable`
    constructor kwargs beyond the common ones (`name`/`description`/`kind`/`labels`)."""
    to_pyagrum: Callable[[DiscreteVariable], gum.DiscreteVariable]
    """Given a `DiscreteVariable` of this kind, build the equivalent pyAgrum variable."""


def _labelized_to_pyagrum(variable: DiscreteVariable) -> gum.DiscreteVariable:
    pyagrum_var = gum.LabelizedVariable(variable.name, variable.description, 0)
    for label in variable.labels:
        pyagrum_var.addLabel(label)
    return pyagrum_var


def _range_to_pyagrum(variable: DiscreteVariable) -> gum.DiscreteVariable:
    return gum.RangeVariable(variable.name, variable.description, variable.lower_bound, variable.upper_bound)


def _integer_to_pyagrum(variable: DiscreteVariable) -> gum.DiscreteVariable:
    return gum.IntegerVariable(variable.name, variable.description, variable.integer_values)


def _discretized_to_pyagrum(variable: DiscreteVariable) -> gum.DiscreteVariable:
    pyagrum_var = gum.DiscretizedVariable(variable.name, variable.description)
    for tick in variable.ticks:
        pyagrum_var.addTick(tick)
    return pyagrum_var


def _numerical_to_pyagrum(variable: DiscreteVariable) -> gum.DiscreteVariable:
    return gum.NumericalDiscreteVariable(variable.name, variable.description, variable.numerical_values)


_VARIABLE_KIND_ADAPTERS: dict[DiscreteVariableKind, _VariableKindAdapter] = {
    DiscreteVariableKind.labelized: _VariableKindAdapter(
        pyagrum_type=gum.VarType_LABELIZED,
        from_pyagrum=lambda _variable: {},
        to_pyagrum=_labelized_to_pyagrum,
    ),
    DiscreteVariableKind.range: _VariableKindAdapter(
        pyagrum_type=gum.VarType_RANGE,
        from_pyagrum=lambda variable: {
            "lower_bound": variable.asRangeVar().minVal(),
            "upper_bound": variable.asRangeVar().maxVal(),
        },
        to_pyagrum=_range_to_pyagrum,
    ),
    DiscreteVariableKind.integer: _VariableKindAdapter(
        pyagrum_type=gum.VarType_INTEGER,
        from_pyagrum=lambda variable: {"integer_values": list(variable.asIntegerVar().integerDomain())},
        to_pyagrum=_integer_to_pyagrum,
    ),
    DiscreteVariableKind.discretized: _VariableKindAdapter(
        pyagrum_type=gum.VarType_DISCRETIZED,
        from_pyagrum=lambda variable: {"ticks": list(variable.asDiscretizedVar().ticks())},
        to_pyagrum=_discretized_to_pyagrum,
    ),
    DiscreteVariableKind.numerical: _VariableKindAdapter(
        pyagrum_type=gum.VarType_NUMERICAL,
        from_pyagrum=lambda variable: {"numerical_values": list(variable.asNumericalDiscreteVar().numericalDomain())},
        to_pyagrum=_numerical_to_pyagrum,
    ),
}

_KIND_FROM_PYAGRUM = {adapter.pyagrum_type: kind for kind, adapter in _VARIABLE_KIND_ADAPTERS.items()}


def _variable_from_pyagrum(variable: gum.DiscreteVariable) -> DiscreteVariable:
    """Convert one pyAgrum discrete variable to a `DiscreteVariable`, dispatching on `varType()`."""
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
    extra = _VARIABLE_KIND_ADAPTERS[kind].from_pyagrum(variable)
    return DiscreteVariable(**common, **extra)


def _variable_to_pyagrum(variable: DiscreteVariable) -> gum.DiscreteVariable:
    """Convert one `DiscreteVariable` to the pyAgrum subclass its `kind` mirrors."""
    adapter = _VARIABLE_KIND_ADAPTERS.get(variable.kind)
    if adapter is None:
        msg = f"No pyAgrum adapter registered for DiscreteVariableKind '{variable.kind}'."
        raise ValueError(msg)
    return adapter.to_pyagrum(variable)


def _potential_from_pyagrum(tensor: gum.Tensor) -> Potential:
    """Convert one pyAgrum CPT to a `Potential`.

    `Tensor.toarray()`'s axes are reversed relative to `variablesSequence()`, so transpose them.
    """
    variables = [v.name() for v in tensor.variablesSequence()]
    values = np.transpose(tensor.toarray())
    return Potential(variables=variables, shape=list(values.shape), values=values.tolist())


def _fill_cpt(tensor: gum.Tensor, potential: Potential) -> None:
    """Fill a freshly-created pyAgrum CPT from a `Potential`.

    Positional: relies on `tensor.variablesSequence()` already matching `potential.variables`.
    """
    actual = [v.name() for v in tensor.variablesSequence()]
    if actual != potential.variables:
        msg = (
            f"CPT variable order {actual} doesn't match Potential.variables {potential.variables}; "
            "arcs must be added in the same order the Potential's parents are listed in."
        )
        raise ValueError(msg)

    values = np.array(potential.values, dtype=float)
    # fillWith requires a C-contiguous array; transpose only gives a strided view.
    tensor.fillWith(np.ascontiguousarray(np.transpose(values)))


def to_pyagrum(network: DiscreteNetwork, decision_pins: dict[str, int] | None = None) -> gum.BayesNet:
    """Convert a `DiscreteNetwork` to a pyAgrum `BayesNet`.

    Args:
        network: the Bayesian network (chance nodes + CPTs).
        decision_pins: decision symbol -> 0-based state index. Any potential with a pinned decision
            parent is sliced at that index, since a `BayesNet` can't hold decisions.
    """
    bn = gum.BayesNet(network.name)
    bn.setProperty("name", network.name)

    for variable in network.variables:
        bn.add(_variable_to_pyagrum(variable))

    pinned_potentials = {
        node_name: _pin_potential(potential, decision_pins) for node_name, potential in network.potentials.items()
    }

    chance_names = {variable.name for variable in network.variables}
    for node_name, potential in pinned_potentials.items():
        for parent in potential.variables[1:]:
            if parent not in chance_names:
                msg = (
                    f"Node '{node_name}'s potential depends on decision '{parent}', which was not "
                    "resolved via decision_pins; pass decision_pins or use "
                    "to_pyagrum_influence_diagram instead."
                )
                raise ValueError(msg)

    # pyAgrum orders a node's parents by arc insertion, and `_fill_cpt` is positional, so add arcs
    # in each Potential's parent order.
    for node_name, potential in pinned_potentials.items():
        for parent_name in potential.variables[1:]:
            bn.addArc(parent_name, node_name)

    for node_name, potential in pinned_potentials.items():
        _fill_cpt(bn.cpt(node_name), potential)

    return bn


def _pin_potential(potential: Potential, decision_pins: dict[str, int] | None) -> Potential:
    """Slice out any `decision_pins` axis from `potential`'s parents, leaving it decision-free."""
    if not decision_pins:
        return potential

    variables = list(potential.variables)
    values = np.array(potential.values, dtype=float)
    for symbol, state in decision_pins.items():
        if symbol not in variables[1:]:
            continue
        axis = variables.index(symbol, 1)
        size = values.shape[axis]
        if not (0 <= state < size):
            msg = f"decision_pins['{symbol}'] = {state} is out of range for '{symbol}' with {size} states."
            raise ValueError(msg)
        values = values.take(state, axis=axis)
        variables.pop(axis)

    return Potential(variables=variables, shape=list(values.shape), values=values.tolist())


def from_pyagrum(bn: gum.BayesNet) -> DiscreteNetwork:
    """Convert a pyAgrum `BayesNet` to a `DiscreteNetwork`.

    Nodes are listed in topological order. Arcs are read from each CPT's `variablesSequence()`
    rather than the unordered `BayesNet.arcs()`, so parent order survives a round trip.
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
    """Import a network from disk via `gum.loadBN`, which dispatches on the file extension."""
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

    If `structure` is given, only the CPTs are relearned on its fixed DAG, so a
    `DecisionAnnotation` on it stays valid. Otherwise structure and parameters are both learned
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


def to_pyagrum_influence_diagram(network: DiscreteNetwork, annotation: DecisionAnnotation) -> gum.InfluenceDiagram:
    """Build a pyAgrum `InfluenceDiagram` from a `(DiscreteNetwork, DecisionAnnotation)` pair.

    Decisions get arcs from their declared `information_set` (no-forgetting is left to the
    inference engine). Each `ObjectiveNode` gets its own utility node `util_<symbol>`, negated when
    minimized since pyAgrum always maximizes.

    Args:
        network: the Bayesian network (chance nodes + CPTs).
        annotation: the decision/objective overlay. Not validated against `network` here.

    Returns:
        gum.InfluenceDiagram: the equivalent pyAgrum diagram.
    """
    diagram = gum.InfluenceDiagram()

    for variable in network.variables:
        diagram.addChanceNode(_variable_to_pyagrum(variable))
    # Decisions first, since chance-node CPTs may name them as parents.
    for decision in annotation.decisions:
        diagram.addDecisionNode(gum.LabelizedVariable(decision.symbol, decision.symbol, decision.states))

    for node_name, potential in network.potentials.items():
        for parent in potential.variables[1:]:
            diagram.addArc(parent, node_name)
    for node_name, potential in network.potentials.items():
        _fill_cpt(diagram.cpt(node_name), potential)

    for decision in annotation.decisions:
        for parent in decision.information_set:
            diagram.addArc(parent, decision.symbol)

    for objective in annotation.objectives:
        util_name = f"util_{objective.symbol}"
        diagram.addUtilityNode(gum.LabelizedVariable(util_name, util_name, 1))
        for parent in objective.parents:
            diagram.addArc(parent, util_name)
        # pyAgrum maximizes, so minimized objectives are negated. Utility-derived values read back
        # (`MEU()`, `posteriorUtility`) must be negated again.
        sign = 1.0 if objective.maximize else -1.0
        fake_potential = Potential(
            variables=[util_name, *objective.parents],
            shape=[1, *objective.utility.shape],
            values=np.expand_dims(sign * np.array(objective.utility.values, dtype=float), axis=0).tolist(),
        )
        _fill_cpt(diagram.utility(util_name), fake_potential)

    return diagram
