"""Tests for `desdeo/problem/graphical/demo.py`.

Each fixture must compile and solve, and the solved values are checked against a hand computation
over the network's CPTs and the objectives' utility tables.
"""

from math import prod

import numpy as np
import pytest

from desdeo.problem.graphical import build_rooted_junction_tree, compile_diagram, read_policy_from_solution
from desdeo.problem.graphical.demo import (
    IRRIGATE_STATES,
    TEST_RESULT_STATES,
    TEST_STATES,
    build_irrigation_demo,
    build_irrigation_demo_annotation,
    build_irrigation_demo_network,
    build_oil_wildcatter_annotation,
    build_oil_wildcatter_demo,
    build_oil_wildcatter_network,
)
from desdeo.problem.graphical.network import DiscreteNetwork, Potential
from desdeo.problem.graphical.schema import DecisionAnnotation, DecisionNode, ObjectiveNode
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


def test_oil_wildcatter_network_and_annotation_validate_against_each_other():
    """`build_oil_wildcatter_demo` runs `validate_against` itself; also checked standalone here."""
    network = build_oil_wildcatter_network()
    annotation = build_oil_wildcatter_annotation()
    annotation.validate_against(network)  # raises on failure


def test_oil_wildcatter_rjt_and_compile_succeeds():
    """The compiler accepts a diagram with three objectives and two decisions (`test`, `drill`)."""
    network, annotation = build_oil_wildcatter_demo()
    rjt = build_rooted_junction_tree(network, annotation)
    problem, index_map = compile_diagram(network, annotation, rjt=rjt)

    assert {o.symbol for o in problem.objectives} == {"investment", "oil_produced", "profit"}
    assert set(index_map.decisions) == {"test", "drill"}


# ── Independent hand computation of the case's own numbers ──────────────────────────────────────
#
# These re-type the case's figures rather than importing them from `demo.py`.

_STRUCTURE_PRIOR = {"dry": 0.5, "wet": 0.3, "soaking": 0.2}
_BARRELS_BY_STRUCTURE = {"dry": 0.0, "wet": 1200.0, "soaking": 2700.0}
_PRICE_PER_BARREL = {"low": 70.0, "baseline": 100.0, "high": 130.0}
_OIL_PRICE_PRIOR = {"low": 0.3, "baseline": 0.5, "high": 0.2}
_TEST_COST = 10_000.0
_DRILL_COST = 70_000.0
# P(test_result | structure, test=yes): the case's own NS/OS/CS joint-probability table, divided
# by its own structure marginals.
_TEST_RESULT_GIVEN_STRUCTURE = {
    "dry": {"NS": 0.6, "OS": 0.3, "CS": 0.1},
    "wet": {"NS": 0.3, "OS": 0.4, "CS": 0.3},
    "soaking": {"NS": 0.1, "OS": 0.4, "CS": 0.5},
}

_EXPECTED_PRICE = sum(_OIL_PRICE_PRIOR[p] * _PRICE_PER_BARREL[p] for p in _OIL_PRICE_PRIOR)
_EXPECTED_BARRELS_PRIOR = sum(_STRUCTURE_PRIOR[s] * _BARRELS_BY_STRUCTURE[s] for s in _STRUCTURE_PRIOR)


def _posterior_structure_given_result(result: str) -> tuple[dict[str, float], float]:
    """`(P(structure | test_result=result, test=yes), P(test_result=result | test=yes))`."""
    joint = {s: _STRUCTURE_PRIOR[s] * _TEST_RESULT_GIVEN_STRUCTURE[s][result] for s in _STRUCTURE_PRIOR}
    marginal = sum(joint.values())
    return {s: p / marginal for s, p in joint.items()}, marginal


def _hand_computed_optimal_test_yes_policy() -> tuple[dict[str, bool], float]:
    """The textbook policy (drill only on favorable test results) and its expected `profit`."""
    drill_policy: dict[str, bool] = {}
    marginal_result: dict[str, float] = {}
    expected_revenue_by_result: dict[str, float] = {}
    for result in ("NS", "OS", "CS"):
        posterior, marginal = _posterior_structure_given_result(result)
        expected_barrels = sum(posterior[s] * _BARRELS_BY_STRUCTURE[s] for s in posterior)
        expected_revenue_by_result[result] = expected_barrels * _EXPECTED_PRICE
        marginal_result[result] = marginal
        drill_policy[result] = expected_revenue_by_result[result] > _DRILL_COST

    expected_profit = (
        sum(
            marginal_result[result] * (expected_revenue_by_result[result] - _DRILL_COST if drill else 0.0)
            for result, drill in drill_policy.items()
        )
        - _TEST_COST
    )
    return drill_policy, expected_profit


def _expected_profit_at_price(drill_policy: dict[str, bool], price: float) -> float:
    """Expected profit under a fixed drill policy, if `oil_price` were a flat constant `price`."""
    total = -_TEST_COST
    for result, drill in drill_policy.items():
        posterior, marginal = _posterior_structure_given_result(result)
        if drill:
            expected_barrels = sum(posterior[s] * _BARRELS_BY_STRUCTURE[s] for s in posterior)
            total += marginal * (expected_barrels * price - _DRILL_COST)
    return total


def _hand_computed_profit_optimum_investment_and_oil_produced() -> tuple[float, float]:
    """`(investment, oil_produced)` expectations under the profit-optimal (`test=yes`) policy."""
    drill_policy, _ = _hand_computed_optimal_test_yes_policy()
    expected_investment = _TEST_COST
    expected_oil_produced = 0.0
    for result, drill in drill_policy.items():
        posterior, marginal = _posterior_structure_given_result(result)
        if drill:
            expected_investment += marginal * _DRILL_COST
            expected_oil_produced += marginal * sum(posterior[s] * _BARRELS_BY_STRUCTURE[s] for s in posterior)
    return expected_investment, expected_oil_produced


def _decode_drill_info_state(flat_index: int, drill_map) -> dict[str, str]:
    """`{node: state_label}` for one flat `drill` info-state index over `[test, test_result]`."""
    cardinalities = drill_map.information_cardinalities
    remaining = flat_index
    state_indices = []
    for i in range(len(cardinalities)):
        trailing = prod(cardinalities[i + 1 :])
        state_indices.append(remaining // trailing)
        remaining %= trailing
    labels = {"test": TEST_STATES, "test_result": TEST_RESULT_STATES}
    return {node: labels[node][idx] for node, idx in zip(drill_map.information_set, state_indices, strict=True)}


def _decode_drill_policy_given_test_yes(index_map, result) -> dict[str, bool]:
    """The solved `drill` policy's `{NS/OS/CS: did_drill}` restricted to the `test='yes'` branch."""
    policies = read_policy_from_solution(
        index_map, {sym: result.optimal_variables[sym] for sym in (dm.z_symbol for dm in index_map.decisions.values())}
    )
    drill_map = index_map.decisions["drill"]

    decoded: dict[str, bool] = {}
    for flat_index, state_index in policies["drill"].policy.items():
        node_states = _decode_drill_info_state(flat_index, drill_map)
        if node_states["test"] == "yes" and node_states["test_result"] in ("NS", "OS", "CS"):
            decoded[node_states["test_result"]] = drill_map.states[state_index] == "yes"
    return decoded


def _decode_reachable_drill_choices(index_map, result) -> list[bool]:
    """The solved `drill` policy's choices at every info-state reachable given the solved `test`.

    Unreachable info-states have zero probability, so the solver's choice there is arbitrary.
    """
    policies = read_policy_from_solution(
        index_map, {sym: result.optimal_variables[sym] for sym in (dm.z_symbol for dm in index_map.decisions.values())}
    )
    test_policy = policies["test"]
    test_state = test_policy.states[test_policy.policy[0]]
    reachable_test_results = {"no_info"} if test_state == "no" else {"NS", "OS", "CS"}

    drill_map = index_map.decisions["drill"]
    choices: list[bool] = []
    for flat_index, state_index in policies["drill"].policy.items():
        node_states = _decode_drill_info_state(flat_index, drill_map)
        if node_states["test"] == test_state and node_states["test_result"] in reachable_test_results:
            choices.append(drill_map.states[state_index] == "yes")
    return choices


def _restrict_test_to_no(network, annotation):
    """Variant of the oil-wildcatter diagram where `test`'s only available choice is `'no'`."""
    no_index = TEST_STATES.index("no")

    def slice_potential(potential: Potential) -> Potential:
        if "test" not in potential.variables:
            return potential
        axis = potential.variables.index("test")
        values = np.array(potential.values)
        sliced = np.take(values, [no_index], axis=axis)
        return Potential(variables=potential.variables, shape=list(sliced.shape), values=sliced.tolist())

    new_potentials = {name: slice_potential(p) for name, p in network.potentials.items()}
    new_network = DiscreteNetwork(
        name=network.name, variables=network.variables, arcs=network.arcs, potentials=new_potentials
    )

    new_decisions = [
        DecisionNode(symbol=d.symbol, name=d.name, states=["no"], information_set=d.information_set)
        if d.symbol == "test"
        else d
        for d in annotation.decisions
    ]
    new_objectives = [
        ObjectiveNode(
            symbol=o.symbol, name=o.name, parents=o.parents, utility=slice_potential(o.utility), maximize=o.maximize
        )
        for o in annotation.objectives
    ]
    new_annotation = DecisionAnnotation(
        schema_version=annotation.schema_version,
        decisions=new_decisions,
        objectives=new_objectives,
        constraints=annotation.constraints,
        risk_measures=annotation.risk_measures,
        decision_order=annotation.decision_order,
        no_forgetting=annotation.no_forgetting,
    )
    new_annotation.validate_against(new_network)
    return new_network, new_annotation


@pytest.mark.gurobipy
def test_profit_optimal_policy_drills_only_on_favorable_test_results():
    """Maximizing `profit` picks `test=yes`, drilling only when the result justifies the $70,000 cost."""
    drill_policy, expected_profit = _hand_computed_optimal_test_yes_policy()
    # The textbook policy: a clean (no-structure) reading isn't worth drilling on; either
    # structure reading is.
    assert drill_policy == {"NS": False, "OS": True, "CS": True}

    network, annotation = build_oil_wildcatter_demo()
    problem, index_map = compile_diagram(network, annotation)
    result = GurobipySolver(problem).solve("profit")
    assert result.success, result.message
    assert result.optimal_objectives["profit"] == pytest.approx(expected_profit, abs=1e-6)

    policies = read_policy_from_solution(
        index_map, {sym: result.optimal_variables[sym] for sym in (dm.z_symbol for dm in index_map.decisions.values())}
    )
    test_policy = policies["test"]
    assert test_policy.states[test_policy.policy[0]] == "yes"

    # The concrete drill policy the test's information actually buys: drill on OS/CS, not NS.
    recovered_drill_policy = _decode_drill_policy_given_test_yes(index_map, result)
    assert recovered_drill_policy == {"NS": False, "OS": True, "CS": True}


@pytest.mark.gurobipy
def test_profit_depends_on_oil_price_prior():
    """`profit`'s expected value depends on `oil_price`'s asymmetric prior.

    `E[price] = 97`, so the solved `profit` must match the 97-based hand computation and not the
    flat-$100 one.
    """
    drill_policy, expected_profit = _hand_computed_optimal_test_yes_policy()
    assert pytest.approx(97.0, abs=1e-9) == _EXPECTED_PRICE

    expected_profit_if_price_were_inert = _expected_profit_at_price(drill_policy, 100.0)
    assert expected_profit != pytest.approx(expected_profit_if_price_were_inert, abs=1e-6)

    network, annotation = build_oil_wildcatter_demo()
    problem, _ = compile_diagram(network, annotation)
    result = GurobipySolver(problem).solve("profit")
    assert result.success, result.message
    assert result.optimal_objectives["profit"] == pytest.approx(expected_profit, abs=1e-6)
    assert result.optimal_objectives["profit"] != pytest.approx(expected_profit_if_price_were_inert, abs=1e-6)


@pytest.mark.gurobipy
def test_profit_test_is_worth_its_cost_evsi_exceeds_price():
    """The $10,000 seismic test is worth its cost: skipping it caps `profit` below the optimum.

    The best `test='no'` policy, solved on a restricted variant, reaches at most $17,300.
    """
    network, annotation = build_oil_wildcatter_demo()

    # `test_result` must not leak structure when `test='no'`.
    test_result_potential = network.potentials["test_result"]
    assert test_result_potential.variables == ["test_result", "structure", "test"]
    no_index = TEST_STATES.index("no")
    yes_index = TEST_STATES.index("yes")
    no_info_index = TEST_RESULT_STATES.index("no_info")
    structure_count = test_result_potential.shape[1]
    for s in range(structure_count):
        no_column = [test_result_potential.values[r][s][no_index] for r in range(len(TEST_RESULT_STATES))]
        assert no_column == [0, 0, 0, 1], f"test='no' leaks structure at structure index {s}: {no_column}"
        assert test_result_potential.values[no_info_index][s][yes_index] == 0, (
            f"test='yes' leaves no_info nonzero at structure index {s}"
        )

    best_no_test_profit = max(0.0, _EXPECTED_BARRELS_PRIOR * _EXPECTED_PRICE - _DRILL_COST)
    assert best_no_test_profit == pytest.approx(17_300.0, abs=1e-6)
    no_test_network, no_test_annotation = _restrict_test_to_no(network, annotation)
    no_test_problem, _ = compile_diagram(no_test_network, no_test_annotation)
    no_test_result = GurobipySolver(no_test_problem).solve("profit")
    assert no_test_result.success, no_test_result.message
    assert no_test_result.optimal_objectives["profit"] == pytest.approx(best_no_test_profit, abs=1e-6)

    problem, _ = compile_diagram(network, annotation)
    result = GurobipySolver(problem).solve("profit")
    assert result.success, result.message
    assert result.optimal_objectives["profit"] > best_no_test_profit + 1e-6


@pytest.mark.gurobipy
def test_investment_minimized_alone_skips_test_and_drilling():
    """Minimizing `investment` alone -- ignoring profit and oil entirely -- picks `test=no, drill=no`."""
    network, annotation = build_oil_wildcatter_demo()
    problem, index_map = compile_diagram(network, annotation)
    result = GurobipySolver(problem).solve("investment")
    assert result.success, result.message
    assert result.optimal_objectives["investment"] == pytest.approx(0.0, abs=1e-6)

    policies = read_policy_from_solution(
        index_map, {sym: result.optimal_variables[sym] for sym in (dm.z_symbol for dm in index_map.decisions.values())}
    )
    test_policy = policies["test"]
    assert test_policy.states[test_policy.policy[0]] == "no"


@pytest.mark.gurobipy
def test_objectives_conflict_three_ways():
    """Minimizing `investment`, maximizing `oil_produced`, and maximizing `profit` each pick a different policy.

    Checked against hand-computed `(investment, oil_produced)` pairs. Maximizing `oil_produced`
    ties between `test='yes'` and `test='no'`, so its `test` choice isn't asserted.
    """
    network, annotation = build_oil_wildcatter_demo()
    problem, index_map = compile_diagram(network, annotation)

    investment_result = GurobipySolver(problem).solve("investment")
    assert investment_result.success, investment_result.message
    oil_result = GurobipySolver(problem).solve("oil_produced")
    assert oil_result.success, oil_result.message
    profit_result = GurobipySolver(problem).solve("profit")
    assert profit_result.success, profit_result.message

    # Minimizing investment: skip the test and never drill -- nothing spent, nothing produced.
    assert investment_result.optimal_objectives["investment"] == pytest.approx(0.0, abs=1e-6)
    assert investment_result.optimal_objectives["oil_produced"] == pytest.approx(0.0, abs=1e-6)

    # Maximizing oil_produced: drill unconditionally (cost doesn't matter to this objective), which
    # reaches the full 900-barrel prior expectation regardless of whether the test was run.
    assert oil_result.optimal_objectives["oil_produced"] == pytest.approx(_EXPECTED_BARRELS_PRIOR, abs=1e-6)
    assert oil_result.optimal_objectives["investment"] == pytest.approx(70_000.0, abs=1e-6) or (
        oil_result.optimal_objectives["investment"] == pytest.approx(80_000.0, abs=1e-6)
    )
    oil_policies = read_policy_from_solution(
        index_map, {sym: oil_result.optimal_variables[sym] for sym in (dm.z_symbol for dm in index_map.decisions.values())}
    )
    oil_test_policy = oil_policies["test"]
    oil_test_state = oil_test_policy.states[oil_test_policy.policy[0]]
    reachable_drill_choices = _decode_reachable_drill_choices(index_map, oil_result)
    # `all([])` is vacuously `True`, so the count must be checked too: 3 reachable info-states
    # (NS/OS/CS) if `test='yes'` was solved, 1 (`no_info`) if `test='no'` was solved.
    assert len(reachable_drill_choices) == (3 if oil_test_state == "yes" else 1)
    assert all(reachable_drill_choices)

    # Maximizing profit: the textbook test='yes', drill-only-on-favorable-result policy.
    expected_investment, expected_oil_produced = _hand_computed_profit_optimum_investment_and_oil_produced()
    assert profit_result.optimal_objectives["investment"] == pytest.approx(expected_investment, abs=1e-6)
    assert profit_result.optimal_objectives["oil_produced"] == pytest.approx(expected_oil_produced, abs=1e-6)
    _, expected_profit = _hand_computed_optimal_test_yes_policy()
    assert profit_result.optimal_objectives["profit"] == pytest.approx(expected_profit, abs=1e-6)
