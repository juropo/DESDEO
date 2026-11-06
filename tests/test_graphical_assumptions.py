"""Tests for `desdeo.problem.graphical.assumptions`."""

import numpy as np
import pytest

from desdeo.problem.graphical import (
    DiagramAssumptions,
    compile_with_assumptions,
    condition_network,
    validate_assumptions,
)
from desdeo.problem.graphical.demo import (
    DRILL_STATES,
    OIL_PRICE_PRIOR,
    OIL_PRICE_STATES,
    STRUCTURE_STATES,
    TEST_STATES,
    build_oil_wildcatter_demo,
)
from desdeo.tools import CVXPYSolver

pytestmark = pytest.mark.graphical


def _expected_utilities(annotation, structure: str, test: str, drill: str) -> dict[str, float]:
    """Each objective's expected utility given `structure`, `test` and `drill`, over the oil price prior."""
    utilities = {}
    for objective in annotation.objectives:
        assert objective.parents == ["test", "drill", "structure", "oil_price"]
        values = np.array(objective.utility.values)[
            TEST_STATES.index(test), DRILL_STATES.index(drill), STRUCTURE_STATES.index(structure)
        ]
        utilities[objective.symbol] = float(np.dot(values, OIL_PRICE_PRIOR))
    return utilities


def test_validate_rejects_chance_node_with_parents():
    """Conditioning on a chance node with parents would change its ancestors, so it is refused."""
    network, annotation = build_oil_wildcatter_demo()
    with pytest.raises(ValueError, match="has parents"):
        validate_assumptions(network, annotation, DiagramAssumptions(chance={"test_result": "NS"}))


@pytest.mark.parametrize(
    ("assumptions", "match"),
    [
        (DiagramAssumptions(chance={"nope": "dry"}), "isn't a chance node"),
        (DiagramAssumptions(chance={"structure": "nope"}), "isn't a state of 'structure'"),
        (DiagramAssumptions(decisions={"nope": "yes"}), "isn't a decision"),
        (DiagramAssumptions(decisions={"drill": "maybe"}), "isn't a state of decision 'drill'"),
    ],
)
def test_validate_rejects_unknown_nodes_and_states(assumptions, match):
    """Unknown nodes and states are refused with a clear message."""
    network, annotation = build_oil_wildcatter_demo()
    with pytest.raises(ValueError, match=match):
        validate_assumptions(network, annotation, assumptions)


def test_merged_with_combines_and_rejects_conflicts():
    """Stacking keeps earlier assumptions, allows repeating one, and refuses a different state."""
    base = DiagramAssumptions(chance={"structure": "soaking"})

    merged = base.merged_with(DiagramAssumptions(chance={"structure": "soaking"}, decisions={"drill": "yes"}))
    assert merged == DiagramAssumptions(chance={"structure": "soaking"}, decisions={"drill": "yes"})
    assert merged.label() == "structure = soaking, drill = yes"

    with pytest.raises(ValueError, match="structure = soaking"):
        base.merged_with(DiagramAssumptions(chance={"structure": "dry"}))


def test_condition_network_sets_point_mass_and_leaves_original():
    """Only the assumed prior changes, and the input network is not modified."""
    network, _ = build_oil_wildcatter_demo()

    conditioned = condition_network(network, DiagramAssumptions(chance={"structure": "soaking"}))

    assert conditioned.potentials["structure"].values == [0.0, 0.0, 1.0]
    assert conditioned.potentials["oil_price"] == network.potentials["oil_price"]
    assert network.potentials["structure"].values == [0.5, 0.3, 0.2]


@pytest.mark.cvxpy
def test_all_assumed_objectives_match_hand_computed_expectations():
    """With `structure`, `test` and `drill` all assumed, every objective equals its expectation over `oil_price`."""
    network, annotation = build_oil_wildcatter_demo()
    assumptions = DiagramAssumptions(chance={"structure": "soaking"}, decisions={"test": "no", "drill": "yes"})

    problem, _, _, _ = compile_with_assumptions(network, annotation, assumptions)
    result = CVXPYSolver(problem).solve("profit_min")

    assert result.success
    expected = _expected_utilities(annotation, "soaking", "no", "yes")
    for symbol, value in expected.items():
        assert result.optimal_objectives[symbol] == pytest.approx(value, rel=1e-6)


@pytest.mark.cvxpy
def test_unassumed_decision_is_still_optimized():
    """With `structure` and `drill` assumed, maximizing profit picks the better `test` choice."""
    network, annotation = build_oil_wildcatter_demo()
    assumptions = DiagramAssumptions(chance={"structure": "soaking"}, decisions={"drill": "yes"})

    problem, _, _, _ = compile_with_assumptions(network, annotation, assumptions)
    result = CVXPYSolver(problem).solve("profit_min")

    assert result.success
    best = max(_expected_utilities(annotation, "soaking", test, "yes")["profit"] for test in TEST_STATES)
    assert result.optimal_objectives["profit"] == pytest.approx(best, rel=1e-6)


def test_oil_price_is_parentless_and_assumable():
    """Every `oil_price` state can be assumed."""
    network, annotation = build_oil_wildcatter_demo()
    for state in OIL_PRICE_STATES:
        validate_assumptions(network, annotation, DiagramAssumptions(chance={"oil_price": state}))
