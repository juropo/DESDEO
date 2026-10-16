"""Bidirectional conversion between `DiscreteNetwork` and pyAgrum's `BayesNet`.

Also the import/export/learning entry points that go through pyAgrum: `load_network` and
`save_network` (pyAgrum's own readers/writers -- BIF, XDSL, `.net`, XMLBIF, ...) and
`learn_network` (a thin wrapper around `pyagrum.BNLearner`); and `to_pyagrum_influence_diagram`,
which additionally folds in a `DecisionAnnotation` to build a full `gum.InfluenceDiagram` --
decision nodes plus one utility node per `ObjectiveNode` -- for Phase 7's `/graphical/whatif`
inference (and, before that, `tests/test_graphical_compile.py`'s own pyAgrum oracle).

This is the only module in `desdeo/problem/graphical/` allowed to import pyAgrum -- see the
dependency rule in the plan's cross-cutting section. pyAgrum is an ordinary hard dependency
(like gurobipy/pyomo/cvxpy already are), not an optional extra -- see the design decisions
table in `.claude/desdeo-graphical-models-plan.md`.
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
    """Everything needed to convert one `DiscreteVariableKind` to/from a pyAgrum variable.

    One registry entry per kind (`_VARIABLE_KIND_ADAPTERS` below) replaces what used to be three
    independent kind-dispatch points here (a `gum.VarType_*` lookup dict plus two separate
    `match` statements) -- adding a kind now means adding one entry instead of remembering to
    update three, and a kind missing from the registry now fails with an explicit `KeyError`/
    `ValueError` at the lookup site instead of a `match` silently falling through. (This can't
    also absorb `DiscreteVariable._check_kind_fields`'s validation in `network.py` -- that module
    is the one that must *not* import pyAgrum, per this package's dependency rule -- so this
    registry only covers pyAgrum-specific conversion, one of the two things kind-dispatch is
    used for in this package.)
    """

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
    """Convert one pyAgrum discrete variable to a `DiscreteVariable`, dispatching on `varType()`.

    `asRangeVar`/`asIntegerVar`/`asDiscretizedVar`/`asNumericalDiscreteVar` (called from within
    `_VARIABLE_KIND_ADAPTERS`'s `from_pyagrum` entries) downcast a base-typed `DiscreteVariable`
    reference (what `BayesNet.variable()` returns) to the concrete subclass so its kind-specific
    accessors become available.
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


def to_pyagrum_influence_diagram(network: DiscreteNetwork, annotation: DecisionAnnotation) -> gum.InfluenceDiagram:
    """Build a pyAgrum `InfluenceDiagram` from a `(DiscreteNetwork, DecisionAnnotation)` pair.

    Lifted from `tests/test_graphical_compile.py`'s own `_to_pyagrum_influence_diagram` (Phase 4's
    pyAgrum oracle -- see that module's docstring), now public so Phase 7's `/graphical/whatif`
    can build the same diagram pyAgrum's own inference engines (`ShaferShenoyLIMIDInference`) need,
    not just the test suite. Chance nodes and their CPTs come straight from `to_pyagrum`-style
    conversion (inlined here rather than delegated, since a `gum.BayesNet` can't be upgraded to a
    `gum.InfluenceDiagram` in place); decision nodes get one arc per `information_set` member
    (the *declared* one -- `no_forgetting` expansion is an inference-engine concern, applied via
    `ShaferShenoyLIMIDInference.addNoForgettingAssumption`, not a structural change to the diagram
    itself); and each `ObjectiveNode` gets its own utility node (`util_<symbol>`), arcs from its
    `parents`, and its own utility table -- deliberately *not* summed into one utility node, per
    the Phase 2 finding that pyAgrum's own multi-utility-node convention does that summing itself
    (`ShaferShenoyLIMIDInference.MEU()`), which is not the semantics `ObjectiveNode` wants. Each
    utility table is written in *pyAgrum's* maximize-only sign convention, negated from
    `objective.utility`'s own values whenever `objective.maximize` is `False` -- see the inline
    comment at the utility-table loop below for why, and for what a caller must do to read a
    utility-node-derived scalar back out in `objective`'s own units.

    Args:
        network: the Bayesian network (chance nodes + CPTs).
        annotation: the decision/objective overlay. Not validated against `network` here --
            callers needing that should call `annotation.validate_against(network)` themselves.

    Returns:
        gum.InfluenceDiagram: the equivalent pyAgrum diagram.
    """
    diagram = gum.InfluenceDiagram()

    for variable in network.variables:
        diagram.addChanceNode(_variable_to_pyagrum(variable))
    for node_name, potential in network.potentials.items():
        for parent in potential.variables[1:]:
            diagram.addArc(parent, node_name)
    for node_name, potential in network.potentials.items():
        _fill_cpt(diagram.cpt(node_name), potential)

    for decision in annotation.decisions:
        diagram.addDecisionNode(gum.LabelizedVariable(decision.symbol, decision.symbol, decision.states))
    for decision in annotation.decisions:
        for parent in decision.information_set:
            diagram.addArc(parent, decision.symbol)

    for objective in annotation.objectives:
        util_name = f"util_{objective.symbol}"
        diagram.addUtilityNode(gum.LabelizedVariable(util_name, util_name, 1))
        for parent in objective.parents:
            diagram.addArc(parent, util_name)
        # pyAgrum's own ID-solving semantics (`ShaferShenoyLIMIDInference`'s policy optimisation,
        # MEU()) always *maximize* the utility node(s). `ObjectiveNode.maximize` can be `False`
        # (a cost -- it's the field's own default), so a minimizing objective's utility values are
        # negated here before being written into pyAgrum's own utility table -- the same
        # `sign = 1.0 if objective.maximize else -1.0` convention `compile.py`'s `_build_objective`
        # already uses for the compiled MILP's `Objective`, applied here so pyAgrum's policy search
        # optimises in the right direction for a minimizing objective too. A caller reading a
        # utility-node-derived scalar back out of this diagram (`MEU()["mean"]`,
        # `posteriorUtility`, ...) must negate again by the same sign to recover the value in
        # `objective`'s own units -- the diagram's own decision/chance-node posteriors need no such
        # correction, since a *choice* has no sign.
        sign = 1.0 if objective.maximize else -1.0
        fake_potential = Potential(
            variables=[util_name, *objective.parents],
            shape=[1, *objective.utility.shape],
            values=np.expand_dims(sign * np.array(objective.utility.values, dtype=float), axis=0).tolist(),
        )
        _fill_cpt(diagram.utility(util_name), fake_potential)

    return diagram
