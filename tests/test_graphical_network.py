"""Schema tests for `DiscreteVariable`, `Potential`, and `DiscreteNetwork`'s structural validators."""

import pytest
from pydantic import ValidationError

from desdeo.problem.graphical import DiscreteNetwork, DiscreteVariable, DiscreteVariableKind, Potential

pytestmark = pytest.mark.graphical


# ── DiscreteVariable ─────────────────────────────────────────────────────────────────────────


def test_labelized_variable_is_the_default_kind():
    """A bare `labels` list, with no kind given, is a labelized variable."""
    variable = DiscreteVariable(name="A", labels=["yes", "no"])
    assert variable.kind == DiscreteVariableKind.labelized


def test_range_variable_valid():
    """A range variable's bounds must span exactly its labels."""
    variable = DiscreteVariable(
        name="R", kind=DiscreteVariableKind.range, labels=["2", "3", "4", "5"], lower_bound=2, upper_bound=5
    )
    assert variable.upper_bound - variable.lower_bound + 1 == len(variable.labels)


def test_range_variable_mismatched_bounds_raises():
    """Bounds spanning a different count than `labels` are rejected."""
    with pytest.raises(ValidationError, match="range bounds"):
        DiscreteVariable(name="R", kind=DiscreteVariableKind.range, labels=["2", "3"], lower_bound=2, upper_bound=5)


def test_range_variable_missing_bounds_raises():
    """`kind='range'` without bounds is rejected rather than silently defaulting."""
    with pytest.raises(ValidationError, match="lacks lower_bound"):
        DiscreteVariable(name="R", kind=DiscreteVariableKind.range, labels=["2", "3"])


def test_integer_variable_valid():
    """An integer variable's explicit values must match its labels one-to-one."""
    variable = DiscreteVariable(
        name="I", kind=DiscreteVariableKind.integer, labels=["1", "3", "7"], integer_values=[1, 3, 7]
    )
    assert variable.integer_values == [1, 3, 7]


def test_integer_variable_missing_values_raises():
    """`kind='integer'` without `integer_values` is rejected."""
    with pytest.raises(ValidationError, match="lacks integer_values"):
        DiscreteVariable(name="I", kind=DiscreteVariableKind.integer, labels=["1", "3", "7"])


def test_discretized_variable_valid():
    """A discretized variable needs one more tick than it has labels (bins)."""
    variable = DiscreteVariable(
        name="D", kind=DiscreteVariableKind.discretized, labels=["[0;1[", "[1;2.5]"], ticks=[0.0, 1.0, 2.5]
    )
    assert len(variable.ticks) == len(variable.labels) + 1


def test_discretized_variable_wrong_tick_count_raises():
    """`len(ticks)` must be exactly `len(labels) + 1`."""
    with pytest.raises(ValidationError, match="ticks don't match"):
        DiscreteVariable(name="D", kind=DiscreteVariableKind.discretized, labels=["[0;1[", "[1;2.5]"], ticks=[0.0, 1.0])


def test_numerical_variable_valid():
    """A numerical variable's explicit float values must match its labels one-to-one."""
    variable = DiscreteVariable(
        name="N", kind=DiscreteVariableKind.numerical, labels=["0.1", "0.2", "0.5"], numerical_values=[0.1, 0.2, 0.5]
    )
    assert variable.numerical_values == [0.1, 0.2, 0.5]


def test_numerical_variable_missing_values_raises():
    """`kind='numerical'` without `numerical_values` is rejected."""
    with pytest.raises(ValidationError, match="lacks numerical_values"):
        DiscreteVariable(name="N", kind=DiscreteVariableKind.numerical, labels=["0.1", "0.2"])


def test_empty_labels_raises():
    """A variable with no states at all is meaningless."""
    with pytest.raises(ValidationError, match="non-empty domain"):
        DiscreteVariable(name="A", labels=[])


def test_discrete_variable_is_frozen():
    """DiscreteVariable is immutable, matching the rest of the frozen problem schema."""
    variable = DiscreteVariable(name="A", labels=["yes", "no"])
    with pytest.raises(ValidationError):
        variable.name = "B"


# ── Potential ────────────────────────────────────────────────────────────────────────────────


def test_potential_valid_shape():
    """A 3-axis Potential with matching `variables`/`shape`/`values` lengths validates."""
    potential = Potential(variables=["C", "A", "B"], shape=[2, 2, 3], values=[[[0.0] * 3] * 2] * 2)
    assert potential.shape == [2, 2, 3]


def test_potential_shape_mismatched_axis_count_raises():
    """`len(shape)` must equal `len(variables)`."""
    with pytest.raises(ValidationError, match="wrong number of axes"):
        Potential(variables=["C", "A"], shape=[2, 2, 3], values=[[[0.0] * 3] * 2] * 2)


def test_potential_values_shape_mismatch_raises():
    """`values`'s actual nested shape must match the declared `shape`."""
    with pytest.raises(ValidationError, match="has shape"):
        Potential(variables=["C"], shape=[3], values=[0.5, 0.5])


def test_potential_duplicate_variables_raises():
    """A repeated name in `variables` (e.g. a decision parent listed twice) is rejected."""
    with pytest.raises(ValidationError, match="variables must be unique"):
        Potential(
            variables=["B", "A", "D", "D"],
            shape=[2, 2, 2, 2],
            values=[[[[0.0] * 2] * 2] * 2] * 2,
        )


# ── DiscreteNetwork ──────────────────────────────────────────────────────────────────────────


def _two_node_chain() -> DiscreteNetwork:
    """A -> B, both binary, A a prior and B's CPT conditioned on A."""
    a = DiscreteVariable(name="A", labels=["low", "high"])
    b = DiscreteVariable(name="B", labels=["low", "high"])
    return DiscreteNetwork(
        name="chain",
        variables=[a, b],
        arcs=[("A", "B")],
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6]),
            "B": Potential(variables=["B", "A"], shape=[2, 2], values=[[0.9, 0.2], [0.1, 0.8]]),
        },
    )


def test_valid_chain_network():
    """A minimal two-node chain constructs successfully."""
    network = _two_node_chain()
    assert [var.name for var in network.variables] == ["A", "B"]
    assert network.arcs == [("A", "B")]


def test_duplicate_variable_names_raises():
    """Two variables with the same name are rejected -- names are the only node identity."""
    a1 = DiscreteVariable(name="A", labels=["0", "1"])
    a2 = DiscreteVariable(name="A", labels=["0", "1"])
    with pytest.raises(ValidationError, match="unique"):
        DiscreteNetwork(
            variables=[a1, a2],
            arcs=[],
            potentials={"A": Potential(variables=["A"], shape=[2], values=[0.5, 0.5])},
        )


def test_arc_to_unknown_node_raises():
    """An arc naming a node that isn't in `variables` is rejected."""
    a = DiscreteVariable(name="A", labels=["0", "1"])
    with pytest.raises(ValidationError, match="unknown child"):
        DiscreteNetwork(
            variables=[a],
            arcs=[("A", "Z")],
            potentials={"A": Potential(variables=["A"], shape=[2], values=[0.5, 0.5])},
        )


def test_self_loop_raises():
    """An arc from a node to itself is rejected."""
    a = DiscreteVariable(name="A", labels=["0", "1"])
    with pytest.raises(ValidationError, match="self-loop"):
        DiscreteNetwork(
            variables=[a],
            arcs=[("A", "A")],
            potentials={"A": Potential(variables=["A"], shape=[2], values=[0.5, 0.5])},
        )


def test_cycle_raises():
    """A -> B -> A is not a DAG and is rejected."""
    a = DiscreteVariable(name="A", labels=["0", "1"])
    b = DiscreteVariable(name="B", labels=["0", "1"])
    with pytest.raises(ValidationError, match="cycle"):
        DiscreteNetwork(
            variables=[a, b],
            arcs=[("A", "B"), ("B", "A")],
            potentials={
                "A": Potential(variables=["A", "B"], shape=[2, 2], values=[[0.5, 0.5], [0.5, 0.5]]),
                "B": Potential(variables=["B", "A"], shape=[2, 2], values=[[0.5, 0.5], [0.5, 0.5]]),
            },
        )


def test_missing_potential_raises():
    """Every variable needs exactly one potential; B has none here."""
    a = DiscreteVariable(name="A", labels=["0", "1"])
    b = DiscreteVariable(name="B", labels=["0", "1"])
    with pytest.raises(ValidationError, match="exactly one potential"):
        DiscreteNetwork(
            variables=[a, b],
            arcs=[],
            potentials={"A": Potential(variables=["A"], shape=[2], values=[0.5, 0.5])},
        )


def test_potential_wrong_own_variable_raises():
    """A's potential must list 'A' first; listing 'Z' instead is rejected."""
    a = DiscreteVariable(name="A", labels=["0", "1"])
    with pytest.raises(ValidationError, match="own \\(first\\) variable"):
        DiscreteNetwork(
            variables=[a],
            arcs=[],
            potentials={"A": Potential(variables=["Z"], shape=[2], values=[0.5, 0.5])},
        )


def test_potential_parent_mismatch_raises():
    """A's Potential claims B as a parent, but no such arc was declared."""
    a = DiscreteVariable(name="A", labels=["0", "1"])
    b = DiscreteVariable(name="B", labels=["0", "1"])
    with pytest.raises(ValidationError, match="but the network's arcs give it parents"):
        DiscreteNetwork(
            variables=[a, b],
            arcs=[],
            potentials={
                "A": Potential(variables=["A", "B"], shape=[2, 2], values=[[0.5, 0.5], [0.5, 0.5]]),
                "B": Potential(variables=["B"], shape=[2], values=[0.5, 0.5]),
            },
        )


def test_potential_shape_disagrees_with_variable_domains_raises():
    """B has 3 labels, but its Potential's shape claims 2."""
    a = DiscreteVariable(name="A", labels=["0", "1"])
    b = DiscreteVariable(name="B", labels=["low", "mid", "high"])
    with pytest.raises(ValidationError, match="expected"):
        DiscreteNetwork(
            variables=[a, b],
            arcs=[("A", "B")],
            potentials={
                "A": Potential(variables=["A"], shape=[2], values=[0.5, 0.5]),
                "B": Potential(variables=["B", "A"], shape=[2, 2], values=[[0.5, 0.5], [0.5, 0.5]]),
            },
        )


def test_potential_external_parent_is_left_unchecked():
    """A potential naming a parent outside the network (e.g. a decision symbol) is not rejected.

    That cross-check belongs to `DecisionAnnotation.validate_against`.
    """
    a = DiscreteVariable(name="A", labels=["0", "1"])
    network = DiscreteNetwork(
        variables=[a],
        arcs=[],
        potentials={"A": Potential(variables=["A", "D"], shape=[2, 2], values=[[0.5, 0.5], [0.5, 0.5]])},
    )
    assert network.potentials["A"].variables == ["A", "D"]


def test_potential_known_parent_mismatch_raises_alongside_external_parent():
    """A known (chance) parent mismatch still raises even with an external parent also present."""
    a = DiscreteVariable(name="A", labels=["0", "1"])
    b = DiscreteVariable(name="B", labels=["0", "1"])
    with pytest.raises(ValidationError, match="but the network's arcs give it parents"):
        DiscreteNetwork(
            variables=[a, b],
            arcs=[],
            potentials={
                "A": Potential(variables=["A"], shape=[2], values=[0.5, 0.5]),
                "B": Potential(
                    variables=["B", "A", "D"],
                    shape=[2, 2, 2],
                    values=[[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]],
                ),
            },
        )


def test_potential_known_shape_mismatch_raises_alongside_external_parent():
    """A known-axis shape mismatch still raises even with an external parent also present."""
    a = DiscreteVariable(name="A", labels=["0", "1"])
    b = DiscreteVariable(name="B", labels=["low", "mid", "high"])
    with pytest.raises(ValidationError, match="expected"):
        DiscreteNetwork(
            variables=[a, b],
            arcs=[("A", "B")],
            potentials={
                "A": Potential(variables=["A"], shape=[2], values=[0.5, 0.5]),
                "B": Potential(
                    variables=["B", "A", "D"],
                    shape=[2, 2, 2],
                    values=[[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]],
                ),
            },
        )


def test_joint_state_count():
    """Two binary variables have 4 joint states."""
    network = _two_node_chain()
    assert network.joint_state_count() == 4


def test_all_states_enumerates_in_given_order():
    """`all_states` enumerates in the requested variable order, last variable fastest."""
    network = _two_node_chain()
    assert network.all_states(["A", "B"]) == [(0, 0), (0, 1), (1, 0), (1, 1)]
