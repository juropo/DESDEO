"""Tests for `desdeo.problem.graphical.schema` (`DecisionAnnotation` and friends).

The round-trip test at the bottom uses pyAgrum-backed `learn_network` and is marked `graphical`.
"""

from pathlib import Path

import pyagrum as gum
import pytest
from fixtures import two_node_chain_network as _chance_network
from pydantic import ValidationError

from desdeo.problem.graphical import (
    DecisionAnnotation,
    DecisionNode,
    DiagramConstraint,
    DiagramConstraintKind,
    DiscreteNetwork,
    DiscreteVariable,
    ObjectiveNode,
    Potential,
    RiskMeasure,
)
from desdeo.problem.graphical.adapters.pyagrum import from_pyagrum, learn_network, to_pyagrum

# `_chance_network` is `tests/fixtures/graphical.py`'s `two_node_chain_network` (A -> B, two
# binary chance nodes), imported above under its old private name so call sites didn't need
# touching.


def _decision_node() -> DecisionNode:
    """D, a binary decision observing A before it's taken."""
    return DecisionNode(symbol="D", states=["reject", "accept"], information_set=["A"])


def _objective_node() -> ObjectiveNode:
    """F, valued over (B, D)."""
    return ObjectiveNode(
        symbol="F",
        parents=["B", "D"],
        utility=Potential(variables=["B", "D"], shape=[2, 2], values=[[0.0, 10.0], [5.0, -5.0]]),
        maximize=True,
    )


def _annotation(**overrides) -> DecisionAnnotation:
    kwargs = {"decisions": [_decision_node()], "objectives": [_objective_node()]}
    kwargs.update(overrides)
    return DecisionAnnotation(**kwargs)


# ── DecisionNode ─────────────────────────────────────────────────────────────────────────────


def test_decision_node_valid():
    """A well-formed decision node keeps its states as given."""
    node = _decision_node()
    assert node.states == ["reject", "accept"]


def test_decision_node_empty_states_raises():
    """A decision must have at least one alternative."""
    with pytest.raises(ValidationError, match="non-empty set of states"):
        DecisionNode(symbol="D", states=[])


def test_decision_node_duplicate_states_raises():
    """States must be distinct."""
    with pytest.raises(ValidationError, match="duplicate states"):
        DecisionNode(symbol="D", states=["a", "a"])


def test_decision_node_self_reference_in_information_set_raises():
    """A decision can't be observed before it's taken."""
    with pytest.raises(ValidationError, match="cannot include itself"):
        DecisionNode(symbol="D", states=["a", "b"], information_set=["D"])


def test_decision_node_information_set_defaults_empty():
    """No `information_set` given means a decision taken with no prior observations."""
    node = DecisionNode(symbol="D", states=["a", "b"])
    assert node.information_set == []


# ── ObjectiveNode ────────────────────────────────────────────────────────────────────────────


def test_objective_node_valid():
    """A well-formed objective node keeps its `maximize` sense as given."""
    node = _objective_node()
    assert node.maximize is True


def test_objective_node_utility_variables_must_match_parents():
    """`utility.variables` must equal `parents` exactly, order included."""
    with pytest.raises(ValidationError, match="must match parents"):
        ObjectiveNode(
            symbol="F",
            parents=["B", "D"],
            utility=Potential(variables=["D", "B"], shape=[2, 2], values=[[0.0, 1.0], [2.0, 3.0]]),
        )


def test_objective_node_self_dependency_raises():
    """An objective can't be its own parent."""
    with pytest.raises(ValidationError, match="cannot depend on itself"):
        ObjectiveNode(
            symbol="F",
            parents=["F"],
            utility=Potential(variables=["F"], shape=[2], values=[0.0, 1.0]),
        )


def test_objective_node_duplicate_parents_raises():
    """Duplicate entries in `parents` are rejected."""
    with pytest.raises(ValidationError, match="variables must be unique"):
        ObjectiveNode(
            symbol="F",
            parents=["A", "A"],
            utility=Potential(variables=["A", "A"], shape=[2, 2], values=[[0.0, 1.0], [2.0, 3.0]]),
        )


# ── RiskMeasure ──────────────────────────────────────────────────────────────────────────────


def test_risk_measure_alpha_out_of_range_raises():
    """`alpha` outside (0, 1) is rejected."""
    with pytest.raises(ValidationError, match="alpha must be in"):
        RiskMeasure(symbol="R", target="F", alpha=1.5)


def test_risk_measure_alpha_valid():
    """A valid `alpha` is kept as given."""
    risk = RiskMeasure(symbol="R", target="F", alpha=0.95)
    assert risk.alpha == 0.95


# ── DecisionAnnotation: internal consistency ────────────────────────────────────────────────


def test_annotation_defaults():
    """Only `objectives` is required; everything else defaults to empty/None/False."""
    annotation = DecisionAnnotation(objectives=[_objective_node()])
    assert annotation.decisions == []
    assert annotation.constraints == []
    assert annotation.risk_measures == []
    assert annotation.decision_order is None
    assert annotation.no_forgetting is False
    assert annotation.schema_version == 1


def test_annotation_requires_at_least_one_objective():
    """An annotation with no objectives at all is rejected."""
    with pytest.raises(ValidationError, match="at least one objective"):
        DecisionAnnotation(objectives=[])


def test_annotation_duplicate_symbols_across_categories_raises():
    """A decision and an objective sharing a symbol is rejected, not just within-category dupes."""
    clashing_objective = ObjectiveNode(
        symbol="D",
        parents=["A"],
        utility=Potential(variables=["A"], shape=[2], values=[0.0, 1.0]),
    )
    with pytest.raises(ValidationError, match="must be unique across"):
        DecisionAnnotation(decisions=[_decision_node()], objectives=[clashing_objective])


def test_annotation_risk_measure_unknown_target_raises():
    """A risk measure targeting an objective symbol that doesn't exist is rejected."""
    risk = RiskMeasure(symbol="R", target="nonexistent", alpha=0.9)
    with pytest.raises(ValidationError, match="unknown objective"):
        _annotation(risk_measures=[risk])


def test_annotation_risk_measure_known_target_ok():
    """A risk measure targeting a real objective symbol is accepted."""
    risk = RiskMeasure(symbol="R", target="F", alpha=0.9)
    annotation = _annotation(risk_measures=[risk])
    assert annotation.risk_measures[0].target == "F"


def test_annotation_decision_order_must_be_permutation_of_decisions():
    """`decision_order` naming a decision that doesn't exist (or omitting one that does) is rejected."""
    with pytest.raises(ValidationError, match="permutation"):
        _annotation(decision_order=["D", "E"])


def test_annotation_decision_order_valid():
    """A `decision_order` that's exactly a permutation of the decision symbols is accepted."""
    annotation = _annotation(decision_order=["D"])
    assert annotation.decision_order == ["D"]


def test_annotation_no_forgetting_without_decision_order_raises():
    """`no_forgetting` is meaningless without a total order, so it requires `decision_order`."""
    with pytest.raises(ValidationError, match="requires decision_order"):
        _annotation(no_forgetting=True)


def test_annotation_no_forgetting_with_decision_order_ok():
    """`no_forgetting` alongside a `decision_order` is accepted."""
    annotation = _annotation(decision_order=["D"], no_forgetting=True)
    assert annotation.no_forgetting is True


# ── DiagramConstraint ────────────────────────────────────────────────────────────────────────


def test_diagram_constraint_construction():
    """A well-formed diagram constraint keeps its `kind` as given."""
    constraint = DiagramConstraint(
        symbol="C1",
        kind=DiagramConstraintKind.decision,
        nodes=["D"],
        func=["D"],
    )
    annotation = _annotation(constraints=[constraint])
    assert annotation.constraints[0].kind == DiagramConstraintKind.decision


def test_diagram_constraint_duplicate_symbol_with_objective_raises():
    """A constraint symbol colliding with an objective symbol is rejected."""
    constraint = DiagramConstraint(symbol="F", kind=DiagramConstraintKind.decision, nodes=["D"], func=["D"])
    with pytest.raises(ValidationError, match="must be unique across"):
        _annotation(constraints=[constraint])


def test_diagram_constraint_empty_func_raises():
    """An empty `func` is rejected at construction."""
    with pytest.raises(ValidationError, match="non-empty func"):
        DiagramConstraint(symbol="C1", kind=DiagramConstraintKind.decision, nodes=["D"], func=[])


# ── validate_against ─────────────────────────────────────────────────────────────────────────


def test_validate_against_passes_for_consistent_network():
    """The happy path: a fully consistent annotation raises nothing."""
    _annotation().validate_against(_chance_network())


def test_validate_against_unknown_information_set_node_raises():
    """A decision's information set naming a node that exists nowhere is rejected."""
    decision = DecisionNode(symbol="D", states=["reject", "accept"], information_set=["nonexistent"])
    annotation = _annotation(decisions=[decision])
    with pytest.raises(ValueError, match="information_set references unknown"):
        annotation.validate_against(_chance_network())


def test_validate_against_unknown_objective_parent_raises():
    """An objective depending on a node that exists nowhere is rejected."""
    objective = ObjectiveNode(
        symbol="F",
        parents=["nonexistent"],
        utility=Potential(variables=["nonexistent"], shape=[2], values=[0.0, 1.0]),
    )
    annotation = _annotation(objectives=[objective])
    with pytest.raises(ValueError, match="parents reference unknown"):
        annotation.validate_against(_chance_network())


def test_validate_against_utility_shape_mismatch_raises():
    """B and D are both binary, so a 3-state utility axis for B is a shape mismatch.

    Distinct failure mode from an unknown reference: B is a real node, just the wrong shape.
    """
    objective = ObjectiveNode(
        symbol="F",
        parents=["B", "D"],
        utility=Potential(variables=["B", "D"], shape=[3, 2], values=[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
    )
    annotation = _annotation(objectives=[objective])
    with pytest.raises(ValueError, match="expected"):
        annotation.validate_against(_chance_network())


def test_validate_against_constraint_unknown_node_raises():
    """A constraint naming a node that exists nowhere is rejected."""
    constraint = DiagramConstraint(
        symbol="C1", kind=DiagramConstraintKind.probability, nodes=["nonexistent"], func=["nonexistent"]
    )
    annotation = _annotation(constraints=[constraint])
    with pytest.raises(ValueError, match="nodes reference unknown"):
        annotation.validate_against(_chance_network())


def test_validate_against_constraint_may_reference_objective_node():
    """A constraint may point at an objective node, unlike information sets/objective parents.

    E.g. a tail constraint on F's own distribution.
    """
    constraint = DiagramConstraint(symbol="C1", kind=DiagramConstraintKind.tail, nodes=["F"], func=["F"])
    _annotation(constraints=[constraint]).validate_against(_chance_network())


def test_validate_against_decision_symbol_colliding_with_network_name_raises():
    """A decision symbol reusing a chance-node name is ambiguous -- rejected once both are on hand."""
    decision = DecisionNode(symbol="A", states=["reject", "accept"])
    objective = ObjectiveNode(
        symbol="F",
        parents=["A"],
        utility=Potential(variables=["A"], shape=[2], values=[0.0, 1.0]),
    )
    annotation = DecisionAnnotation(decisions=[decision], objectives=[objective])
    with pytest.raises(ValueError, match="collide with network variable names"):
        annotation.validate_against(_chance_network())


# ── validate_against: a chance node's potential referencing a decision ──


def _network_with_decision_referencing_potential():
    """A (prior) -> B, B's own CPT also conditions on decision 'D' (not a network variable)."""
    a = DiscreteVariable(name="A", labels=["lo", "hi"])
    b = DiscreteVariable(name="B", labels=["lo", "hi"])
    return DiscreteNetwork(
        variables=[a, b],
        arcs=[("A", "B")],
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6]),
            "B": Potential(
                variables=["B", "A", "D"],
                shape=[2, 2, 2],
                values=[[[0.8, 0.4], [0.6, 0.1]], [[0.2, 0.6], [0.4, 0.9]]],
            ),
        },
    )


def test_validate_against_chance_potential_referencing_known_decision_passes():
    """B's CPT naming decision 'D' as a parent validates once 'D' is a declared decision."""
    network = _network_with_decision_referencing_potential()
    decision = DecisionNode(symbol="D", states=["low", "high"])
    objective = ObjectiveNode(
        symbol="F", parents=["B"], utility=Potential(variables=["B"], shape=[2], values=[0.0, 1.0])
    )
    annotation = DecisionAnnotation(decisions=[decision], objectives=[objective])

    annotation.validate_against(network)  # raises nothing


def test_validate_against_chance_potential_unknown_external_parent_raises():
    """B's CPT naming 'D' as a parent is rejected when no decision 'D' exists at all."""
    network = _network_with_decision_referencing_potential()
    objective = ObjectiveNode(
        symbol="F", parents=["B"], utility=Potential(variables=["B"], shape=[2], values=[0.0, 1.0])
    )
    annotation = DecisionAnnotation(objectives=[objective])

    with pytest.raises(ValueError, match="external parents reference unknown"):
        annotation.validate_against(network)


def test_validate_against_chance_potential_decision_shape_mismatch_raises():
    """'D' exists, but with 3 states -- a shape mismatch, distinct from an unknown reference."""
    network = _network_with_decision_referencing_potential()
    decision = DecisionNode(symbol="D", states=["low", "mid", "high"])
    objective = ObjectiveNode(
        symbol="F", parents=["B"], utility=Potential(variables=["B"], shape=[2], values=[0.0, 1.0])
    )
    annotation = DecisionAnnotation(decisions=[decision], objectives=[objective])

    with pytest.raises(ValueError, match="potential has shape"):
        annotation.validate_against(network)


# ── Round trip: relearn CPTs, reapply the same annotation ──────────────────────────────────


@pytest.mark.graphical
def test_annotation_still_validates_after_relearning_cpts(tmp_path: Path):
    """The "decision structure lives in a separate overlay" design decision, end to end.

    Relearning CPTs from new data must not require redoing decision modelling.
    """
    net = _chance_network()
    annotation = _annotation()
    annotation.validate_against(net)  # sanity: valid before any relearning happens

    bn = to_pyagrum(net)
    generator = gum.BNDatabaseGenerator(bn)
    generator.drawSamples(2000)
    csv_path = tmp_path / "samples.csv"
    generator.toCSV(str(csv_path))

    relearned = learn_network(csv_path, structure=net)

    # Same DAG, same variable domains, different (relearned) CPTs -- the annotation references
    # none of the CPT values, so it must still validate.
    annotation.validate_against(relearned)

    # And a network reloaded straight from pyAgrum with the identical structure validates too.
    annotation.validate_against(from_pyagrum(bn))
