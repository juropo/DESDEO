"""Shared `DiscreteNetwork`/`Potential` builders for `problem/graphical` tests.

Plain functions, not `@pytest.fixture`s (`tests/fixtures/problems.py`'s pattern) -- several of
these take arguments (`independent_network`'s node names), and every call site wants a fresh,
independently-constructed network rather than one shared per-test instance. Consolidates what
used to be near-identical private builders hand-rolled separately in
`tests/test_graphical_rjt.py`, `tests/test_graphical_annotation.py`,
`tests/test_graphical_pyagrum_adapter.py`, and `desdeo/api/tests/test_graphical_model_metadata.py`.
"""

import numpy as np

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
            "B": Potential(variables=["B", "A"], shape=[2, 2], values=[[0.9, 0.2], [0.1, 0.8]]),
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


def ternary_variable(name: str) -> DiscreteVariable:
    """A three-state (`"lo"`/`"mid"`/`"hi"`) chance-node domain."""
    return DiscreteVariable(name=name, labels=["lo", "mid", "hi"])


def mdp_2tbn_network() -> DiscreteNetwork:
    """Fully observed MDP 2TBN: X binary, D ternary; `Dt | X0`, `Xt | X0, Dt`."""
    return DiscreteNetwork(
        variables=[
            binary_variable("X0"),
            binary_variable("Xt"),
            ternary_variable("D0"),
            ternary_variable("Dt"),
        ],
        arcs=[("X0", "D0"), ("X0", "Dt"), ("X0", "Xt"), ("Dt", "Xt")],
        potentials={
            "X0": Potential(variables=["X0"], shape=[2], values=[0.30, 0.70]),
            "D0": Potential(
                variables=["D0", "X0"],
                shape=[3, 2],
                values=[[0.45, 0.15], [0.35, 0.50], [0.20, 0.35]],
            ),
            "Dt": Potential(
                variables=["Dt", "X0"],
                shape=[3, 2],
                values=[[0.30, 0.55], [0.45, 0.25], [0.25, 0.20]],
            ),
            "Xt": Potential(
                variables=["Xt", "X0", "Dt"],
                shape=[2, 2, 3],
                values=[
                    [[0.60, 0.30, 0.20], [0.40, 0.70, 0.55]],
                    [[0.40, 0.70, 0.80], [0.60, 0.30, 0.45]],
                ],
            ),
        },
    )


def same_slice_2tbn_network() -> DiscreteNetwork:
    """Same-slice observation 2TBN: X, D binary, Y ternary; `Yt | X0`, `Dt | Yt`, `Xt | X0, Yt, Dt`."""
    return DiscreteNetwork(
        variables=[
            binary_variable("X0"),
            binary_variable("Xt"),
            ternary_variable("Y0"),
            ternary_variable("Yt"),
            binary_variable("D0"),
            binary_variable("Dt"),
        ],
        arcs=[
            ("X0", "Y0"),
            ("X0", "Yt"),
            ("Y0", "D0"),
            ("Yt", "Dt"),
            ("X0", "Xt"),
            ("Yt", "Xt"),
            ("Dt", "Xt"),
        ],
        potentials={
            "X0": Potential(variables=["X0"], shape=[2], values=[0.35, 0.65]),
            "Y0": Potential(
                variables=["Y0", "X0"],
                shape=[3, 2],
                values=[[0.50, 0.20], [0.30, 0.40], [0.20, 0.40]],
            ),
            "Yt": Potential(
                variables=["Yt", "X0"],
                shape=[3, 2],
                values=[[0.30, 0.40], [0.50, 0.35], [0.20, 0.25]],
            ),
            "D0": Potential(
                variables=["D0", "Y0"],
                shape=[2, 3],
                values=[[0.70, 0.40, 0.25], [0.30, 0.60, 0.75]],
            ),
            "Dt": Potential(
                variables=["Dt", "Yt"],
                shape=[2, 3],
                values=[[0.60, 0.30, 0.55], [0.40, 0.70, 0.45]],
            ),
            "Xt": Potential(
                variables=["Xt", "X0", "Yt", "Dt"],
                shape=[2, 2, 3, 2],
                values=[
                    [
                        [
                            [0.30, 0.40],
                            [0.50, 0.60],
                            [0.20, 0.30],
                        ],
                        [
                            [0.40, 0.50],
                            [0.30, 0.20],
                            [0.60, 0.40],
                        ],
                    ],
                    [
                        [
                            [0.70, 0.60],
                            [0.50, 0.40],
                            [0.80, 0.70],
                        ],
                        [
                            [0.60, 0.50],
                            [0.70, 0.80],
                            [0.40, 0.60],
                        ],
                    ],
                ],
            ),
        },
    )


def atemporal_2tbn_network(g_affects_slice0: bool = False) -> DiscreteNetwork:
    """Atemporal A (binary) and G (ternary), both parents of `Xt`; G is a parent of X0 iff `g_affects_slice0`."""
    variables = [
        binary_variable("A"),
        ternary_variable("G"),
        binary_variable("X0"),
        binary_variable("Xt"),
    ]
    if g_affects_slice0:
        x0_pot = Potential(
            variables=["X0", "A", "G"],
            shape=[2, 2, 3],
            values=[
                [[0.70, 0.55, 0.40], [0.30, 0.20, 0.15]],
                [[0.30, 0.45, 0.60], [0.70, 0.80, 0.85]],
            ],
        )
        arcs = [("A", "X0"), ("G", "X0"), ("A", "Xt"), ("X0", "Xt"), ("G", "Xt")]
    else:
        x0_pot = Potential(
            variables=["X0", "A"],
            shape=[2, 2],
            values=[[0.60, 0.30], [0.40, 0.70]],
        )
        arcs = [("A", "X0"), ("A", "Xt"), ("X0", "Xt"), ("G", "Xt")]
    xt_pot = Potential(
        variables=["Xt", "X0", "A", "G"],
        shape=[2, 2, 2, 3],
        values=[
            [[[0.70, 0.55, 0.40], [0.30, 0.25, 0.15]], [[0.60, 0.45, 0.35], [0.20, 0.10, 0.08]]],
            [[[0.30, 0.45, 0.60], [0.70, 0.75, 0.85]], [[0.40, 0.55, 0.65], [0.80, 0.90, 0.92]]],
        ],
    )
    return DiscreteNetwork(
        variables=variables,
        arcs=arcs,
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=[0.40, 0.60]),
            "G": Potential(variables=["G"], shape=[3], values=[0.30, 0.40, 0.30]),
            "X0": x0_pot,
            "Xt": xt_pot,
        },
    )


def ci_healthcare_mirror_network(seed: int = 0) -> DiscreteNetwork:
    """The arcs of a cross-impact healthcare 2TBN with random CPTs; HHA, GER, OMA are atemporal."""
    parents: dict[str, list[str]] = {
        "HHA": [],
        "GER": [],
        "OMA": [],
        "PIS0": [],
        "PIH0": ["HHA", "PIS0"],
        "LYS0": ["PIH0", "PIS0"],
        "LYH0": ["HHA", "LYS0", "PIH0", "PIS0"],
        "AST0": ["GER", "LYH0", "OMA", "PIH0"],
        "KUO0": ["LYH0", "LYS0", "PIH0", "PIS0"],
        "PISt": ["LYH0", "LYS0", "PIH0", "PIS0"],
        "PIHt": ["GER", "LYH0", "OMA", "PIH0", "PISt"],
        "LYSt": ["GER", "LYH0", "LYS0", "OMA", "PIHt", "PISt"],
        "LYHt": ["GER", "LYH0", "LYSt", "OMA", "PIHt", "PISt"],
        "ASTt": ["LYHt", "PIHt"],
        "KUOt": ["LYHt", "LYSt", "PIHt", "PISt"],
    }
    card: dict[str, int] = {
        "HHA": 2,
        "GER": 2,
        "OMA": 3,
        "PIS0": 3,
        "PIH0": 2,
        "LYS0": 3,
        "LYH0": 3,
        "AST0": 2,
        "KUO0": 2,
        "PISt": 3,
        "PIHt": 2,
        "LYSt": 3,
        "LYHt": 3,
        "ASTt": 2,
        "KUOt": 2,
    }
    variables = [binary_variable(name) if card[name] == 2 else ternary_variable(name) for name in parents]
    arcs: list[tuple[str, str]] = [(p, node) for node, pars in parents.items() for p in pars]
    rng = np.random.default_rng(seed)
    potentials: dict[str, Potential] = {}
    for node, node_parents in parents.items():
        node_card = card[node]
        parent_shape = [card[p] for p in node_parents]
        n_configs = int(np.prod(parent_shape))
        arr = rng.dirichlet(np.ones(node_card), size=n_configs)
        arr = arr.reshape([*parent_shape, node_card])
        arr = np.moveaxis(arr, -1, 0)
        potentials[node] = Potential(
            variables=[node, *node_parents],
            shape=[node_card, *parent_shape],
            values=arr.tolist(),
        )
    return DiscreteNetwork(variables=variables, arcs=arcs, potentials=potentials)
