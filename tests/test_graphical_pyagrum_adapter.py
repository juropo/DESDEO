"""Tests for `desdeo.problem.graphical.adapters.pyagrum`."""

from pathlib import Path

import numpy as np
import pyagrum as gum
import pytest
from fixtures import diverse_kind_network as _diverse_kind_network

from desdeo.problem.graphical import (
    DecisionAnnotation,
    DiscreteNetwork,
    DiscreteVariable,
    DiscreteVariableKind,
    Potential,
    compile_diagram,
)
from desdeo.problem.graphical.adapters.pyagrum import (
    from_pyagrum,
    from_pyagrum_influence_diagram,
    learn_network,
    load_influence_diagram,
    load_network,
    save_network,
    to_pyagrum,
    to_pyagrum_influence_diagram,
)
from desdeo.problem.graphical.demo import build_irrigation_demo, build_oil_wildcatter_demo
from desdeo.tools.gurobipy_solver_interfaces import GurobipySolver

pytestmark = pytest.mark.graphical


def _assert_networks_equivalent(a: DiscreteNetwork, b: DiscreteNetwork) -> None:
    """Structure exact, CPT floats with a tolerance.

    Compared by name, since a DAG's topological order isn't unique; each `Potential.variables`
    order must match exactly.
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
    """`from_pyagrum(to_pyagrum(net)) == net`, seeded from `from_pyagrum(gum.randomBN(...))`."""
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


def test_round_trip_preserves_every_variable_kind():
    """Each `DiscreteVariableKind`'s kind-specific fields (not just `labels`) survive the round trip."""
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


def test_to_pyagrum_decision_pins_slices_out_the_pinned_axis():
    """`decision_pins` slices a chance node's decision parent out of its CPT."""
    a = DiscreteVariable(name="A", labels=["lo", "hi"])
    b = DiscreteVariable(name="B", labels=["lo", "hi"])
    network = DiscreteNetwork(
        variables=[a, b],
        arcs=[("A", "B")],
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=[0.5, 0.5]),
            "B": Potential(
                variables=["B", "A", "D"],
                shape=[2, 2, 2],
                values=[[[0.8, 0.4], [0.6, 0.1]], [[0.2, 0.6], [0.4, 0.9]]],
            ),
        },
    )

    bn = to_pyagrum(network, decision_pins={"D": 1})

    assert {v.name() for v in bn.cpt("B").variablesSequence()} == {"B", "A"}
    np.testing.assert_allclose(bn.cpt("B")[{"A": "lo"}].tolist(), [0.4, 0.6])
    np.testing.assert_allclose(bn.cpt("B")[{"A": "hi"}].tolist(), [0.1, 0.9])


def test_to_pyagrum_decision_pin_matching_own_node_name_is_not_sliced():
    """`decision_pins` naming a chance node's own name must not slice that node's own axis."""
    a = DiscreteVariable(name="A", labels=["lo", "hi"])
    b = DiscreteVariable(name="B", labels=["lo", "hi"])
    network = DiscreteNetwork(
        variables=[a, b],
        arcs=[],
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=[0.5, 0.5]),
            "B": Potential(variables=["B"], shape=[2], values=[0.3, 0.7]),
        },
    )

    bn = to_pyagrum(network, decision_pins={"A": 0})

    assert {v.name() for v in bn.cpt("A").variablesSequence()} == {"A"}
    np.testing.assert_allclose(bn.cpt("A").toarray(), [0.5, 0.5])


def test_to_pyagrum_decision_pin_out_of_range_raises_value_error():
    """An out-of-range pin state raises a clear `ValueError`, not a bare numpy `IndexError`."""
    a = DiscreteVariable(name="A", labels=["lo", "hi"])
    b = DiscreteVariable(name="B", labels=["lo", "hi"])
    network = DiscreteNetwork(
        variables=[a, b],
        arcs=[("A", "B")],
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=[0.5, 0.5]),
            "B": Potential(
                variables=["B", "A", "D"],
                shape=[2, 2, 2],
                values=[[[0.8, 0.4], [0.6, 0.1]], [[0.2, 0.6], [0.4, 0.9]]],
            ),
        },
    )

    with pytest.raises(ValueError, match="out of range"):
        to_pyagrum(network, decision_pins={"D": 5})

    with pytest.raises(ValueError, match="out of range"):
        to_pyagrum(network, decision_pins={"D": -1})


def test_to_pyagrum_unresolved_decision_parent_raises_value_error():
    """No (or incomplete) `decision_pins` for a decision-referencing potential: a clear `ValueError`."""
    a = DiscreteVariable(name="A", labels=["lo", "hi"])
    b = DiscreteVariable(name="B", labels=["lo", "hi"])
    network = DiscreteNetwork(
        variables=[a, b],
        arcs=[("A", "B")],
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=[0.5, 0.5]),
            "B": Potential(
                variables=["B", "A", "D"],
                shape=[2, 2, 2],
                values=[[[0.8, 0.4], [0.6, 0.1]], [[0.2, 0.6], [0.4, 0.9]]],
            ),
        },
    )

    with pytest.raises(ValueError, match=r"'D'.*'B'|'B'.*'D'"):
        to_pyagrum(network)


# ── File import/export via pyAgrum's own readers/writers ───────────────────────────────────


def test_save_and_load_network_round_trip(tmp_path: Path):
    """BIF is a plain-text format pyAgrum both reads and writes; structure survives the trip.

    BIF doesn't preserve `kind`.
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
    """Structure-fixed learning (the CPT-relearning path) never changes the DAG."""
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


# ── Influence-diagram import ─────────────────────────────────────────────────────────────────


def _assert_round_trip_equal(
    network: DiscreteNetwork,
    annotation: DecisionAnnotation,
    imported: tuple[DiscreteNetwork, DecisionAnnotation],
) -> None:
    imported_network, imported_annotation = imported
    _assert_networks_equivalent(network, imported_network)
    assert imported_annotation == annotation


@pytest.mark.parametrize(
    ("build", "minimize", "units", "no_forgetting"),
    [
        (build_irrigation_demo, (), None, False),
        (
            build_oil_wildcatter_demo,
            {"investment"},
            {"investment": "USD", "oil_produced": "barrels", "profit": "USD"},
            True,
        ),
    ],
    ids=["irrigation", "oil_wildcatter"],
)
def test_influence_diagram_round_trip_on_demos(build, minimize, units, no_forgetting):
    """Exporting a demo and importing it back with the matching options gives the same pair."""
    network, annotation = build()

    imported = from_pyagrum_influence_diagram(
        to_pyagrum_influence_diagram(network, annotation),
        minimize=minimize,
        units=units,
        no_forgetting=no_forgetting,
    )

    _assert_round_trip_equal(network, annotation, imported)
    assert imported[0].name == network.name


# Cost by [D1, Y]. The expected cost is lowest for `D1 = some`, and transposing the table changes
# both the minimizer and the minimum.
HAND_BUILT_COST = np.array([[8.0, 5.0, 3.0], [9.0, 4.0, 2.0], [16.0, 13.0, 10.0]])


def _hand_built_diagram(utility_names: tuple[str, ...] = ("util_gain", "cost")) -> gum.InfluenceDiagram:
    """`W -> Y <- D1`, `D2` observing `Y` and `D1`; `util_gain(Y, D2)` and `cost(D1, Y)`."""
    diagram = gum.InfluenceDiagram()
    diagram.addChanceNode(gum.LabelizedVariable("W", "Weather", ["bad", "good"]))
    diagram.addChanceNode(gum.LabelizedVariable("Y", "Yield", ["low", "mid", "high"]))
    diagram.addDecisionNode(gum.LabelizedVariable("D1", "Fertilise", ["none", "some", "lots"]))
    diagram.addDecisionNode(gum.LabelizedVariable("D2", "D2", ["sell", "store"]))
    diagram.addArc("W", "Y")
    diagram.addArc("D1", "Y")
    diagram.addArc("Y", "D2")
    diagram.addArc("D1", "D2")
    diagram.cpt("W").fillWith([0.4, 0.6])
    # Axes of `cpt("Y")` are [Y, W, D1]; `fillWith` takes them reversed.
    y_given = np.array(
        [
            [[0.7, 0.2, 0.1], [0.5, 0.3, 0.2], [0.4, 0.3, 0.3]],
            [[0.3, 0.5, 0.2], [0.1, 0.5, 0.4], [0.05, 0.25, 0.7]],
        ]
    )  # [W, D1, Y]
    diagram.cpt("Y").fillWith(np.ascontiguousarray(np.transpose(y_given, (1, 0, 2))))

    if "util_gain" in utility_names:
        diagram.addUtilityNode(gum.LabelizedVariable("util_gain", "Gain", 1))
        diagram.addArc("Y", "util_gain")
        diagram.addArc("D2", "util_gain")
        # [D2, Y, util_gain] reversed.
        diagram.utility("util_gain").fillWith(
            np.ascontiguousarray(np.array([[10.0, 40.0, 90.0], [25.0, 35.0, 60.0]])[:, :, None])
        )
    if "cost" in utility_names:
        diagram.addUtilityNode(gum.LabelizedVariable("cost", "cost", 1))
        diagram.addArc("D1", "cost")
        diagram.addArc("Y", "cost")
        # pyAgrum maximizes, so the cost is stored negated. [cost, D1, Y] reversed.
        diagram.utility("cost").fillWith(np.ascontiguousarray(-HAND_BUILT_COST.T[:, :, None]))
    return diagram


def test_hand_built_diagram_imports_decision_parents_and_strips_util_prefix():
    """A CPT keeps its decision parent; arcs hold only chance-to-chance arcs; `util_` is stripped."""
    network, annotation = from_pyagrum_influence_diagram(_hand_built_diagram(), minimize={"cost"})

    assert network.potentials["Y"].variables == ["Y", "W", "D1"]
    assert network.arcs == [("W", "Y")]
    decisions = {decision.symbol: decision for decision in annotation.decisions}
    assert decisions["D1"].name == "Fertilise"
    assert decisions["D2"].name == ""
    assert set(decisions["D2"].information_set) == {"Y", "D1"}
    objectives = {objective.symbol: objective for objective in annotation.objectives}
    assert set(objectives) == {"gain", "cost"}
    assert objectives["gain"].name == "Gain"
    assert objectives["gain"].parents == ["Y", "D2"]
    assert objectives["gain"].maximize
    assert annotation.decision_order is None


def test_minimize_negates_utilities_back():
    """A minimized objective's utility is the negation of the pyAgrum table, and `maximize` is off."""
    _, annotation = from_pyagrum_influence_diagram(_hand_built_diagram(), minimize={"cost"}, units={"cost": "EUR"})

    cost = next(objective for objective in annotation.objectives if objective.symbol == "cost")
    assert not cost.maximize
    assert cost.unit == "EUR"
    assert cost.parents == ["D1", "Y"]
    np.testing.assert_allclose(np.array(cost.utility.values), HAND_BUILT_COST)


@pytest.mark.parametrize("utility_name", ["util_gain", "cost"])
def test_imported_diagram_optimum_matches_pyagrum_meu(utility_name):
    """Per utility node, the compiled DESDEO optimum equals `ShaferShenoyLIMIDInference`'s MEU."""
    diagram = _hand_built_diagram((utility_name,))
    symbol = utility_name.removeprefix("util_")
    minimize = {"cost"} if symbol == "cost" else set()

    inference = gum.ShaferShenoyLIMIDInference(diagram)
    assert inference.isSolvable()
    inference.makeInference()
    sign = -1.0 if minimize else 1.0
    expected = sign * inference.MEU()["mean"]

    network, annotation = from_pyagrum_influence_diagram(diagram, minimize=minimize)
    problem, _ = compile_diagram(network, annotation)
    result = GurobipySolver(problem).solve(symbol)
    assert result.success, result.message

    assert result.optimal_objectives[symbol] == pytest.approx(expected)


@pytest.mark.parametrize("option", ["minimize", "units"])
def test_unknown_objective_names_raise_value_error(option):
    """`minimize`/`units` must only name imported objectives, so the unstripped name is rejected."""
    with pytest.raises(ValueError, match="unknown objective"):
        from_pyagrum_influence_diagram(_hand_built_diagram(), **{option: {"util_gain": "x"}})


def test_minimize_as_a_bare_string_raises_type_error():
    """A string is a collection of characters, so `minimize="cost"` is rejected rather than iterated."""
    with pytest.raises(TypeError, match="not the string"):
        from_pyagrum_influence_diagram(_hand_built_diagram(), minimize="cost")


def _diagram_with_utility(utility_name: str, *, chance_name: str = "A") -> gum.InfluenceDiagram:
    """A chance node and decision `x`, both parents of the given utility node."""
    diagram = gum.InfluenceDiagram()
    diagram.addChanceNode(gum.LabelizedVariable(chance_name, chance_name, 2))
    diagram.addDecisionNode(gum.LabelizedVariable("x", "x", 2))
    diagram.addUtilityNode(gum.LabelizedVariable(utility_name, utility_name, 1))
    diagram.addArc(chance_name, utility_name)
    diagram.addArc("x", utility_name)
    return diagram


def test_utility_named_only_the_prefix_raises_value_error():
    """`util_` would import with an empty symbol."""
    with pytest.raises(ValueError, match="empty symbol"):
        from_pyagrum_influence_diagram(_diagram_with_utility("util_"))


@pytest.mark.parametrize(("utility_name", "clashing_node"), [("util_A", "A"), ("util_x", "x")])
def test_stripped_utility_symbol_clashing_with_another_node_raises_value_error(utility_name, clashing_node):
    """A stripped utility symbol must not equal a chance-node name or decision symbol."""
    with pytest.raises(ValueError, match=rf"'{utility_name}'.*clashes with node '{clashing_node}'"):
        from_pyagrum_influence_diagram(_diagram_with_utility(utility_name))


def test_two_utilities_stripping_to_the_same_symbol_raise_value_error():
    """`u` and `util_u` would both import as objective `u`."""
    diagram = _diagram_with_utility("u")
    diagram.addUtilityNode(gum.LabelizedVariable("util_u", "util_u", 1))
    diagram.addArc("x", "util_u")

    with pytest.raises(ValueError, match=r"'util_u'.*clashes with node 'u'"):
        from_pyagrum_influence_diagram(diagram)


def test_parentless_utility_raises_value_error():
    """A constant utility has no parents to build an `ObjectiveNode` utility table over."""
    diagram = _diagram_with_utility("u")
    diagram.addUtilityNode(gum.LabelizedVariable("util_const", "util_const", 1))

    with pytest.raises(ValueError, match="'util_const' has no parents"):
        from_pyagrum_influence_diagram(diagram)


def test_unnamed_decision_and_objective_round_trip():
    """Export writes the symbol as the description, which imports back as `name=""`."""
    network, annotation = build_irrigation_demo()
    unnamed = annotation.model_copy(
        update={
            "decisions": [decision.model_copy(update={"name": ""}) for decision in annotation.decisions],
            "objectives": [objective.model_copy(update={"name": ""}) for objective in annotation.objectives],
        }
    )

    imported = from_pyagrum_influence_diagram(to_pyagrum_influence_diagram(network, unnamed))

    _assert_round_trip_equal(network, unnamed, imported)


def test_no_forgetting_without_a_decision_order_raises_value_error():
    """Two decisions with no directed path between them have no total order."""
    diagram = gum.InfluenceDiagram()
    diagram.addDecisionNode(gum.LabelizedVariable("D1", "D1", 2))
    diagram.addDecisionNode(gum.LabelizedVariable("D2", "D2", 2))
    diagram.addUtilityNode(gum.LabelizedVariable("u", "u", 1))
    diagram.addArc("D1", "u")
    diagram.addArc("D2", "u")

    with pytest.raises(ValueError, match="total order"):
        from_pyagrum_influence_diagram(diagram, no_forgetting=True)


def test_no_forgetting_reads_the_decision_order():
    """`no_forgetting=True` takes `decision_order` from the diagram's `decisionOrder()`."""
    _, annotation = from_pyagrum_influence_diagram(_hand_built_diagram(), no_forgetting=True)

    assert annotation.decision_order == ["D1", "D2"]
    assert annotation.no_forgetting


def test_save_and_load_influence_diagram_round_trip(tmp_path: Path):
    """BIFXML keeps descriptions, so names survive `gum.saveID` + `load_influence_diagram`.

    `gum.loadID` names the diagram after the file, so the network name isn't compared.
    """
    network, annotation = build_oil_wildcatter_demo()
    path = tmp_path / "oil_wildcatter.bifxml"

    gum.saveID(to_pyagrum_influence_diagram(network, annotation), str(path))
    imported = load_influence_diagram(
        path,
        minimize={"investment"},
        units={"investment": "USD", "oil_produced": "barrels", "profit": "USD"},
        no_forgetting=True,
    )

    _assert_round_trip_equal(network, annotation, imported)
