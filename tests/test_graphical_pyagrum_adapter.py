"""Tests for `desdeo.problem.graphical.adapters.pyagrum`.

Phase 1 of `.claude/desdeo-graphical-models-plan.md`. pyAgrum is a hard dependency (see the
plan's design decisions), so these tests need no install-skip logic -- just the `graphical`
marker for selection (`pytest -m graphical`), same as `gurobipy`/`cvxpy`/`pyomo`.
"""

from pathlib import Path

import numpy as np
import pyagrum as gum
import pytest

from desdeo.problem.graphical import DiscreteNetwork, DiscreteVariable, DiscreteVariableKind, Potential
from desdeo.problem.graphical.adapters.pyagrum import (
    from_pyagrum,
    learn_network,
    load_network,
    save_network,
    to_pyagrum,
)

pytestmark = pytest.mark.graphical


def _assert_networks_equivalent(a: DiscreteNetwork, b: DiscreteNetwork) -> None:
    """Structure exact, CPT floats with a tolerance -- per the plan's round-trip test note.

    Compares by variable/node name rather than list position: a DAG's topological order isn't
    unique, so `DiscreteNetwork.variables`'s list order is allowed to differ across an
    equivalent round trip (see `from_pyagrum`'s docstring) as long as every node's own
    `Potential.variables` order -- the one that's actually load-bearing for axis alignment --
    matches exactly.
    """
    variables_a = {var.name: var for var in a.variables}
    variables_b = {var.name: var for var in b.variables}
    assert variables_a == variables_b

    assert sorted(a.arcs) == sorted(b.arcs)

    assert set(a.potentials) == set(b.potentials)
    for name in a.potentials:
        potential_a, potential_b = a.potentials[name], b.potentials[name]
        assert potential_a.variables == potential_b.variables
        assert potential_a.shape == potential_b.shape
        np.testing.assert_allclose(np.array(potential_a.values), np.array(potential_b.values))


# ── Structural round trip against random networks ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("seed", "n", "domain_size", "ratio_arc"),
    [
        (1, 5, 2, 1.2),
        (2, 6, 3, 1.5),
        (3, 7, 2, 1.0),
        (4, 8, 4, 1.3),
        (5, 10, 2, 2.0),
    ],
)
def test_round_trip_random_networks(seed, n, domain_size, ratio_arc):
    """`from_pyagrum(to_pyagrum(net)) == net` (up to the tolerance in `_assert_networks_equivalent`).

    `net` here is `from_pyagrum(gum.randomBN(...))` -- see the docstring note on why the
    property is anchored at our own schema rather than at an independently-generated one: the
    round trip direction the checklist item names is `from_pyagrum(to_pyagrum(net))`, which
    requires starting from a `DiscreteNetwork`, and pyAgrum's own generator is the simplest
    source of realistic random *networks* to seed one from.
    """
    gum.initRandom(seed)
    bn = gum.randomBN(n=n, domain_size=domain_size, ratio_arc=ratio_arc)
    net = from_pyagrum(bn)

    round_tripped = from_pyagrum(to_pyagrum(net))

    _assert_networks_equivalent(net, round_tripped)


def test_round_trip_is_a_fixed_point():
    """A second round trip changes nothing further, i.e. the representation has stabilised."""
    gum.initRandom(11)
    net = from_pyagrum(gum.randomBN(n=6, domain_size=3, ratio_arc=1.4))

    once = from_pyagrum(to_pyagrum(net))
    twice = from_pyagrum(to_pyagrum(once))

    _assert_networks_equivalent(once, twice)


# ── Variable kind fidelity ──────────────────────────────────────────────────────────────────


def _diverse_kind_network() -> DiscreteNetwork:
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


def test_round_trip_preserves_every_variable_kind():
    """Each `DiscreteVariableKind`'s kind-specific fields (not just `labels`) survive the round trip.

    Compared by name (see `_assert_networks_equivalent`): with no arcs constraining them, these
    five independent roots aren't guaranteed to keep the same list position through
    `to_pyagrum`/`from_pyagrum`.
    """
    net = _diverse_kind_network()

    round_tripped = from_pyagrum(to_pyagrum(net))

    _assert_networks_equivalent(net, round_tripped)
    by_name = {var.name: var for var in round_tripped.variables}
    assert by_name["A"].kind == DiscreteVariableKind.labelized
    assert by_name["B"].kind == DiscreteVariableKind.range
    assert by_name["C"].kind == DiscreteVariableKind.integer
    assert by_name["D"].kind == DiscreteVariableKind.discretized
    assert by_name["E"].kind == DiscreteVariableKind.numerical


# ── to_pyagrum produces a genuinely usable BayesNet ─────────────────────────────────────────


def test_to_pyagrum_cpts_are_valid_distributions():
    """Every CPT the adapter fills sums to 1 over the node's own axis, for every parent context."""
    gum.initRandom(21)
    net = from_pyagrum(gum.randomBN(n=6, domain_size=3, ratio_arc=1.4))

    bn = to_pyagrum(net)

    for node_id in bn.nodes():
        array = bn.cpt(node_id).toarray()
        # `Tensor.toarray()`'s last axis is the node's own axis (see `_potential_from_pyagrum`).
        np.testing.assert_allclose(array.sum(axis=-1), 1.0)


def test_to_pyagrum_inference_matches_hand_computed_posterior():
    """A converted two-node chain gives the exact posterior an independent hand computation does."""
    a = DiscreteVariable(name="A", labels=["low", "high"])
    b = DiscreteVariable(name="B", labels=["low", "high"])
    prior = [0.4, 0.6]
    # P(B | A): rows are A's state, columns are B's state.
    conditional = [[0.9, 0.1], [0.2, 0.8]]
    net = DiscreteNetwork(
        variables=[a, b],
        arcs=[("A", "B")],
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=prior),
            "B": Potential(
                variables=["B", "A"], shape=[2, 2], values=[list(row) for row in zip(*conditional, strict=True)]
            ),
        },
    )

    bn = to_pyagrum(net)
    inference = gum.LazyPropagation(bn)
    inference.makeInference()
    posterior_b = inference.posterior("B").toarray()

    expected_b_low = prior[0] * conditional[0][0] + prior[1] * conditional[1][0]
    expected_b_high = prior[0] * conditional[0][1] + prior[1] * conditional[1][1]
    np.testing.assert_allclose(sorted(posterior_b), sorted([expected_b_low, expected_b_high]))


# ── File import/export via pyAgrum's own readers/writers ───────────────────────────────────


def test_save_and_load_network_round_trip(tmp_path: Path):
    """BIF is a plain-text format pyAgrum both reads and writes; structure survives the trip.

    BIF normalises every variable to a generic labelized-like representation, so `kind` isn't
    expected to survive here -- that's what `test_round_trip_preserves_every_variable_kind`
    (an in-memory `to_pyagrum`/`from_pyagrum` trip, no file format involved) covers instead.
    """
    gum.initRandom(31)
    net = from_pyagrum(gum.randomBN(n=6, domain_size=3, ratio_arc=1.3))
    bif_path = tmp_path / "network.bif"

    save_network(net, bif_path)
    reloaded = load_network(bif_path)

    assert {var.name for var in net.variables} == {var.name for var in reloaded.variables}
    assert sorted(net.arcs) == sorted(reloaded.arcs)
    for name in net.potentials:
        np.testing.assert_allclose(
            np.array(net.potentials[name].values), np.array(reloaded.potentials[name].values), atol=1e-6
        )


# ── Learning from data ───────────────────────────────────────────────────────────────────────


@pytest.fixture
def sampled_csv(tmp_path: Path) -> tuple[Path, DiscreteNetwork]:
    """A CSV of samples drawn from a small random network, plus the network it came from."""
    gum.initRandom(41)
    bn = gum.randomBN(n=5, domain_size=2, ratio_arc=1.3)
    net = from_pyagrum(bn)

    generator = gum.BNDatabaseGenerator(bn)
    generator.drawSamples(3000)
    csv_path = tmp_path / "samples.csv"
    generator.toCSV(str(csv_path))

    return csv_path, net


def test_learn_network_with_fixed_structure_keeps_the_dag(sampled_csv):
    """Structure-fixed learning (the CPT-relearning path) never changes the DAG.

    This is the entry point the "decision structure lives in a separate overlay" design
    decision relies on -- see `learn_network`'s docstring.
    """
    csv_path, net = sampled_csv

    learned = learn_network(csv_path, structure=net)

    assert {var.name for var in learned.variables} == {var.name for var in net.variables}
    assert sorted(learned.arcs) == sorted(net.arcs)
    for name in net.potentials:
        assert learned.potentials[name].variables == net.potentials[name].variables


def test_learn_network_with_fixed_structure_recovers_approximately_correct_cpts(sampled_csv):
    """With 3000 samples, learned CPTs should be close to the network they were drawn from."""
    csv_path, net = sampled_csv

    learned = learn_network(csv_path, structure=net)

    for name, potential in net.potentials.items():
        np.testing.assert_allclose(np.array(learned.potentials[name].values), np.array(potential.values), atol=0.15)


def test_learn_network_without_structure_recovers_the_same_variables(sampled_csv):
    """Free structure learning still recovers the right variable set and valid CPTs."""
    csv_path, net = sampled_csv

    learned = learn_network(csv_path)

    assert {var.name for var in learned.variables} == {var.name for var in net.variables}
    for potential in learned.potentials.values():
        array = np.array(potential.values)
        np.testing.assert_allclose(array.sum(axis=0), 1.0, atol=1e-6)
