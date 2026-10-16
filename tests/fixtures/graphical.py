"""Shared `DiscreteNetwork`/`Potential` builders for `problem/graphical` tests.

Plain functions, not `@pytest.fixture`s (`tests/fixtures/problems.py`'s pattern) -- several of
these take arguments (`independent_network`'s node names), and every call site wants a fresh,
independently-constructed network rather than one shared per-test instance. Consolidates what
used to be near-identical private builders hand-rolled separately in
`tests/test_graphical_rjt.py`, `tests/test_graphical_annotation.py`,
`tests/test_graphical_pyagrum_adapter.py`, and `desdeo/api/tests/test_graphical_model_metadata.py`.
"""

from desdeo.problem.graphical import DiscreteNetwork, DiscreteVariable, DiscreteVariableKind, Potential


def binary_variable(name: str) -> DiscreteVariable:
    """A two-state (`"lo"`/`"hi"`) chance-node domain."""
    return DiscreteVariable(name=name, labels=["lo", "hi"])


def two_node_chain_network() -> DiscreteNetwork:
    """A -> B, two binary chance nodes."""
    return DiscreteNetwork(
        variables=[binary_variable("A"), binary_variable("B")],
        arcs=[("A", "B")],
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6]),
            "B": Potential(variables=["B", "A"], shape=[2, 2], values=[[0.9, 0.1], [0.2, 0.8]]),
        },
    )


def chain_network() -> DiscreteNetwork:
    """A -> B -> C, three binary chance nodes."""
    return DiscreteNetwork(
        variables=[binary_variable("A"), binary_variable("B"), binary_variable("C")],
        arcs=[("A", "B"), ("B", "C")],
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6]),
            "B": Potential(variables=["B", "A"], shape=[2, 2], values=[[0.9, 0.1], [0.2, 0.8]]),
            "C": Potential(variables=["C", "B"], shape=[2, 2], values=[[0.7, 0.3], [0.3, 0.7]]),
        },
    )


def fork_network() -> DiscreteNetwork:
    """A -> B, A -> C: a common-cause fork. A has two real children."""
    return DiscreteNetwork(
        variables=[binary_variable("A"), binary_variable("B"), binary_variable("C")],
        arcs=[("A", "B"), ("A", "C")],
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6]),
            "B": Potential(variables=["B", "A"], shape=[2, 2], values=[[0.9, 0.1], [0.2, 0.8]]),
            "C": Potential(variables=["C", "A"], shape=[2, 2], values=[[0.7, 0.3], [0.3, 0.7]]),
        },
    )


def independent_network(*names: str) -> DiscreteNetwork:
    """`len(names)` binary chance nodes with no arcs between them at all."""
    variables = [binary_variable(name) for name in names]
    return DiscreteNetwork(
        variables=variables,
        arcs=[],
        potentials={name: Potential(variables=[name], shape=[2], values=[0.5, 0.5]) for name in names},
    )


def diverse_kind_network() -> DiscreteNetwork:
    """One independent root node per `DiscreteVariableKind`, so none constrain each other."""
    variables = [
        DiscreteVariable(name="A", kind=DiscreteVariableKind.labelized, labels=["yes", "no"]),
        DiscreteVariable(
            name="B", kind=DiscreteVariableKind.range, labels=["2", "3", "4"], lower_bound=2, upper_bound=4
        ),
        DiscreteVariable(name="C", kind=DiscreteVariableKind.integer, labels=["1", "5", "9"], integer_values=[1, 5, 9]),
        DiscreteVariable(
            name="D",
            kind=DiscreteVariableKind.discretized,
            labels=["[0;1[", "[1;2.5]"],
            ticks=[0.0, 1.0, 2.5],
        ),
        DiscreteVariable(
            name="E",
            kind=DiscreteVariableKind.numerical,
            labels=["0.1", "0.2", "0.5"],
            numerical_values=[0.1, 0.2, 0.5],
        ),
    ]
    potentials = {
        "A": Potential(variables=["A"], shape=[2], values=[0.3, 0.7]),
        "B": Potential(variables=["B"], shape=[3], values=[0.2, 0.3, 0.5]),
        "C": Potential(variables=["C"], shape=[3], values=[0.2, 0.3, 0.5]),
        "D": Potential(variables=["D"], shape=[2], values=[0.4, 0.6]),
        "E": Potential(variables=["E"], shape=[3], values=[0.2, 0.3, 0.5]),
    }
    return DiscreteNetwork(name="diverse_kinds", variables=variables, arcs=[], potentials=potentials)
