"""Tests for `desdeo/problem/graphical/demo.py` -- Phase F of the graphical-models plan.

Not a production compiler fixture (see the plan's Phase F checklist item 1: "it doesn't need to be
exhaustively tested like a production compiler fixture, but it does need `build_rooted_junction_
tree`/`compile_diagram` to actually succeed on it"). Checked here: it does compile and solve, and
the solved values agree with a from-first-principles hand computation over the network's own CPTs
and the objectives' own utility tables -- an oracle check, not just "the solver returned success".

`gurobipy` is marked per-test, not at module level (matching `desdeo/api/tests/
test_graphical_router.py`'s own convention) -- a review pass caught the original module-level
`pytestmark = [pytest.mark.graphical, pytest.mark.gurobipy]` marking every test here `gurobipy`
even though only `test_profit_solve_matches_hand_computed_expectation` actually solves via
`GurobipySolver`; the other two never touch gurobipy at all. That made `pytest -m "graphical and
not gurobipy"` skip tests the plan's own checklist calls out as the ones it actually requires
(validate_against succeeding, and the compiler itself accepting the diagram).
"""

import pytest

from desdeo.problem.graphical import build_rooted_junction_tree, compile_diagram
from desdeo.problem.graphical.demo import (
    IRRIGATE_STATES,
    build_irrigation_demo,
    build_irrigation_demo_annotation,
    build_irrigation_demo_network,
)
from desdeo.tools.gurobipy_solver_interfaces import GurobipySolver

pytestmark = pytest.mark.graphical


def test_network_and_annotation_validate_against_each_other():
    """`build_irrigation_demo` runs `validate_against` itself; also checked standalone here."""
    network = build_irrigation_demo_network()
    annotation = build_irrigation_demo_annotation()
    annotation.validate_against(network)  # raises on failure


def test_rjt_and_compile_succeed():
    """The one thing the plan's checklist actually requires: the compiler accepts this diagram."""
    network, annotation = build_irrigation_demo()
    rjt = build_rooted_junction_tree(network, annotation)
    problem, index_map = compile_diagram(network, annotation, rjt=rjt)

    assert {o.symbol for o in problem.objectives} == {"profit"}
    assert set(index_map.decisions) == {"irrigate"}


def _hand_computed_expected_profit() -> dict[str, float]:
    """Expected profit for each `irrigate` choice.

    Computed directly from the network's own CPTs and the objective's own utility values --
    independent of `compile_diagram`'s own machinery.
    """
    weather = {"dry": 0.3, "normal": 0.4, "wet": 0.3}
    soil_given_weather = {
        "dry": {"low": 0.60, "medium": 0.35, "high": 0.05},
        "normal": {"low": 0.20, "medium": 0.60, "high": 0.20},
        "wet": {"low": 0.05, "medium": 0.35, "high": 0.60},
    }
    yield_given_soil = {
        "none": {
            "low": {"poor": 0.70, "average": 0.25, "good": 0.05},
            "medium": {"poor": 0.20, "average": 0.60, "good": 0.20},
            "high": {"poor": 0.05, "average": 0.35, "good": 0.60},
        },
        "light": {
            "low": {"poor": 0.40, "average": 0.45, "good": 0.15},
            "medium": {"poor": 0.10, "average": 0.50, "good": 0.40},
            "high": {"poor": 0.05, "average": 0.30, "good": 0.65},
        },
        "heavy": {
            "low": {"poor": 0.15, "average": 0.45, "good": 0.40},
            "medium": {"poor": 0.05, "average": 0.35, "good": 0.60},
            "high": {"poor": 0.10, "average": 0.30, "good": 0.60},
        },
    }
    revenue = {"poor": 1000.0, "average": 2500.0, "good": 4000.0}
    water_cost = {"none": 0.0, "light": 150.0, "heavy": 400.0}

    soil_marginal = {
        soil: sum(weather[w] * soil_given_weather[w][soil] for w in weather) for soil in ("low", "medium", "high")
    }

    expected: dict[str, float] = {}
    for choice in IRRIGATE_STATES:
        yield_marginal = {
            state: sum(soil_marginal[soil] * yield_given_soil[choice][soil][state] for soil in soil_marginal)
            for state in ("poor", "average", "good")
        }
        expected_revenue = sum(yield_marginal[state] * revenue[state] for state in yield_marginal)
        expected[choice] = expected_revenue - water_cost[choice]
    return expected


@pytest.mark.gurobipy
def test_profit_solve_matches_hand_computed_expectation():
    """Maximizing `profit` picks the choice with the best hand-computed expected profit."""
    network, annotation = build_irrigation_demo()
    problem, _ = compile_diagram(network, annotation)
    result = GurobipySolver(problem).solve("profit")
    assert result.success, result.message

    expected = _hand_computed_expected_profit()
    best_choice = max(expected, key=expected.get)
    assert result.optimal_objectives["profit"] == pytest.approx(expected[best_choice], abs=1e-6)
    # A genuine, non-trivial choice -- not every irrigation level ties, so this actually exercises
    # the decision rather than being indifferent to it.
    assert len(set(expected.values())) == len(expected)
