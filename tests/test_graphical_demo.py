"""Tests for `desdeo/problem/graphical/demo.py`.

Each fixture must compile and solve, and the solved values are checked against a hand computation
over the network's CPTs and the objectives' utility tables.
"""

import pytest

from desdeo.problem.graphical import build_rooted_junction_tree, compile_diagram
from desdeo.problem.graphical.demo import (
    IRRIGATE_STATES,
    POWER_SOURCE_STATES,
    build_cabin_energy_annotation,
    build_cabin_energy_demo,
    build_cabin_energy_network,
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
    """The compiler accepts the irrigation diagram."""
    network, annotation = build_irrigation_demo()
    rjt = build_rooted_junction_tree(network, annotation)
    problem, index_map = compile_diagram(network, annotation, rjt=rjt)

    assert {o.symbol for o in problem.objectives} == {"profit"}
    assert set(index_map.decisions) == {"irrigate"}


def _hand_computed_expected_profit() -> dict[str, float]:
    """Expected profit for each `irrigate` choice.

    Computed directly from the network's CPTs and the objective's utility values, independently
    of `compile_diagram`.
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
    # The choices don't tie, so the decision actually matters.
    assert len(set(expected.values())) == len(expected)


def test_cabin_energy_network_and_annotation_validate_against_each_other():
    """`build_cabin_energy_demo` runs `validate_against` itself; also checked standalone here."""
    network = build_cabin_energy_network()
    annotation = build_cabin_energy_annotation()
    annotation.validate_against(network)  # raises on failure


def test_cabin_energy_rjt_and_compile_succeeds():
    """The compiler accepts a diagram with two objectives and a decision."""
    network, annotation = build_cabin_energy_demo()
    rjt = build_rooted_junction_tree(network, annotation)
    problem, index_map = compile_diagram(network, annotation, rjt=rjt)

    assert {o.symbol for o in problem.objectives} == {"expected_cost", "expected_unmet_demand"}
    assert set(index_map.decisions) == {"power_source"}


def _hand_computed_expected_cabin_energy() -> dict[str, dict[str, float]]:
    """Expected `expected_cost`/`expected_unmet_demand` for each `power_source` choice.

    Computed directly from the network's CPTs and the objectives' utility values, independently
    of `compile_diagram`.
    """
    weather = {"sunny": 0.6, "cloudy": 0.4}
    solar_given_weather = {
        "sunny": {"low": 0.2, "high": 0.8},
        "cloudy": {"low": 0.8, "high": 0.2},
    }
    grid_outage = {"stable": 0.9, "outage": 0.1}

    solar_marginal = {
        state: sum(weather[w] * solar_given_weather[w][state] for w in weather) for state in ("low", "high")
    }

    cost = {
        "grid": {"low": {"stable": 5.0, "outage": 50.0}, "high": {"stable": 5.0, "outage": 50.0}},
        "battery": {"low": {"stable": 20.0, "outage": 20.0}, "high": {"stable": 0.0, "outage": 0.0}},
    }
    unmet_demand = {
        "grid": {"low": {"stable": 0.0, "outage": 3.0}, "high": {"stable": 0.0, "outage": 3.0}},
        "battery": {"low": {"stable": 1.0, "outage": 1.0}, "high": {"stable": 0.0, "outage": 0.0}},
    }

    expected_cost: dict[str, float] = {}
    expected_unmet_demand: dict[str, float] = {}
    for choice in POWER_SOURCE_STATES:
        expected_cost[choice] = sum(
            solar_marginal[solar] * grid_outage[grid] * cost[choice][solar][grid]
            for solar in solar_marginal
            for grid in grid_outage
        )
        expected_unmet_demand[choice] = sum(
            solar_marginal[solar] * grid_outage[grid] * unmet_demand[choice][solar][grid]
            for solar in solar_marginal
            for grid in grid_outage
        )
    return {"expected_cost": expected_cost, "expected_unmet_demand": expected_unmet_demand}


@pytest.mark.gurobipy
def test_cabin_energy_solve_matches_hand_computed_expectation_and_conflicts():
    """Minimizing `expected_cost` alone picks `battery`; minimizing `expected_unmet_demand` alone picks `grid`."""
    network, annotation = build_cabin_energy_demo()
    problem, _ = compile_diagram(network, annotation)
    expected = _hand_computed_expected_cabin_energy()

    cost_result = GurobipySolver(problem).solve("expected_cost")
    assert cost_result.success, cost_result.message
    best_cost_choice = min(expected["expected_cost"], key=expected["expected_cost"].get)
    assert cost_result.optimal_objectives["expected_cost"] == pytest.approx(
        expected["expected_cost"][best_cost_choice], abs=1e-6
    )

    unmet_result = GurobipySolver(problem).solve("expected_unmet_demand")
    assert unmet_result.success, unmet_result.message
    best_unmet_choice = min(expected["expected_unmet_demand"], key=expected["expected_unmet_demand"].get)
    assert unmet_result.optimal_objectives["expected_unmet_demand"] == pytest.approx(
        expected["expected_unmet_demand"][best_unmet_choice], abs=1e-6
    )

    # The objectives conflict.
    assert best_cost_choice != best_unmet_choice
