"""Tests for DESDEO outcome distribution aggregators."""

import pytest

from desdeo.problem.scenario import Scenario, ScenarioModel
from desdeo.problem.schema import (
    Constant,
    Constraint,
    ConstraintTypeEnum,
    ExtraFunction,
    Objective,
    Problem,
    ScalarizationFunction,
    Variable,
    VariableTypeEnum,
)
from desdeo.tools import (
    GurobipySolver,
    OutcomeDistribution,
    add_conditional_value_at_risk,
    add_cvar_dd,
    add_expected_value_dd,
    add_worst_case_dd,
)
from desdeo.tools.outcome_distribution import cvar_dd_constraint_symbols


def _ef(symbol, func):
    return ExtraFunction(
        name=symbol,
        symbol=symbol,
        func=func,
        is_linear=True,
        is_convex=True,
        is_twice_differentiable=True,
    )


def _expected(values, probs):
    """Compute the expected value of a discrete distribution."""
    return sum(v * p for v, p in zip(values, probs, strict=True))


def _worst(values, probs, maximize):
    """Return the worst outcome value (smallest if maximize, largest otherwise)."""
    eligible = [v for v, p in zip(values, probs, strict=True) if p > 1e-9]
    return min(eligible) if maximize else max(eligible)


def _cvar(values, probs, alpha, maximize):
    """Compute CVaR as the mass-weighted average over the worst alpha probability mass."""
    pairs = list(zip(values, probs, strict=True))
    if maximize:
        pairs.sort(key=lambda p: p[0])
    else:
        pairs.sort(key=lambda p: p[0], reverse=True)
    total_mass = 0.0
    weighted_sum = 0.0
    for val, prob in pairs:
        if total_mass >= alpha:
            break
        remaining = alpha - total_mass
        if prob <= remaining:
            total_mass += prob
            weighted_sum += val * prob
        else:
            weighted_sum += val * remaining
            total_mass = alpha
            break
    return weighted_sum / alpha


def _make_simple_problem(maximize=True):
    """Build a minimal single-binary-variable problem with two probability extras."""
    return Problem(
        name="t",
        description="t",
        variables=[
            Variable(
                name="x",
                symbol="x",
                variable_type=VariableTypeEnum.binary,
                lowerbound=0,
                upperbound=1,
            )
        ],
        extra_funcs=[_ef("p_0", "x"), _ef("p_1", "1 - x")],
        objectives=[
            Objective(
                name="f",
                symbol="f",
                func=["Multiply", 0, "x"],
                maximize=maximize,
                unit="EUR",
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
        ],
    )


def _build_bruteforce_problem(maximize):
    """Build a two-binary-variable problem with three outcomes for brute-force testing.

    Some combinations zero the lowest or the highest outcome, so the worst case and CVaR vary
    across combinations in both directions.
    """
    problem = Problem(
        name="bf",
        description="brute force test",
        variables=[
            Variable(name="x", symbol="x", variable_type=VariableTypeEnum.binary, lowerbound=0, upperbound=1),
            Variable(name="y", symbol="y", variable_type=VariableTypeEnum.binary, lowerbound=0, upperbound=1),
        ],
        extra_funcs=[
            _ef("p_0", "0.2 - 0.2*x + 0.1*y"),
            _ef("p_1", "0.3 + 0.4*y"),
            _ef("p_2", "0.5 + 0.2*x - 0.5*y"),
        ],
        objectives=[
            Objective(
                name="f",
                symbol="f",
                func=["Multiply", 0, "x"],
                maximize=maximize,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            ),
        ],
    )
    dists = {
        "f": OutcomeDistribution(
            symbol="f",
            name="f",
            maximize=maximize,
            values=[0.0, 40.0, 100.0],
            probabilities=["p_0", "p_1", "p_2"],
        ),
    }
    prob_coeffs = {
        "p_0": (0.2, -0.2, 0.1),
        "p_1": (0.3, 0.0, 0.4),
        "p_2": (0.5, 0.2, -0.5),
    }
    return problem, dists, prob_coeffs


def _apply_aggregator(problem, dists, symbols, agg):
    """Dispatch to the correct aggregator function based on the agg tag string."""
    if agg == "expected":
        return add_expected_value_dd(problem, dists, symbols)
    if agg == "worst":
        return add_worst_case_dd(problem, dists, symbols)
    alpha = float(agg.split("_")[1])
    return add_cvar_dd(problem, dists, symbols, alpha)


def _reference_aggregate(values, probs, agg, maximize):
    """Evaluate the reference aggregator for a given probability vector."""
    if agg == "expected":
        return _expected(values, probs)
    if agg == "worst":
        return _worst(values, probs, maximize)
    alpha = float(agg.split("_")[1])
    return _cvar(values, probs, alpha, maximize)


AGGREGATORS = [
    pytest.param("expected", id="expected"),
    pytest.param("worst", id="worst"),
    pytest.param("cvar_0.25", id="cvar_0.25"),
    pytest.param("cvar_0.5", id="cvar_0.5"),
]


@pytest.mark.gurobipy
@pytest.mark.parametrize("maximize", [True, False], ids=["maximize", "minimize"])
@pytest.mark.parametrize("agg", AGGREGATORS)
def test_bruteforce_two_binary_three_outcomes(maximize, agg):
    """Aggregators match brute-force optimum over all four (x, y) combinations."""
    problem, dists, coeffs = _build_bruteforce_problem(maximize)
    new_problem, mapping = _apply_aggregator(problem, dists, ["f"], agg)
    new_sym = mapping["f"]
    values = [0.0, 40.0, 100.0]

    def compute_probs(x, y):
        return [
            coeffs["p_0"][0] + coeffs["p_0"][1] * x + coeffs["p_0"][2] * y,
            coeffs["p_1"][0] + coeffs["p_1"][1] * x + coeffs["p_1"][2] * y,
            coeffs["p_2"][0] + coeffs["p_2"][1] * x + coeffs["p_2"][2] * y,
        ]

    combos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    agg_values = [_reference_aggregate(values, compute_probs(x, y), agg, maximize) for x, y in combos]
    assert len(set(agg_values)) > 1
    brute_opt = max(agg_values) if maximize else min(agg_values)

    result = GurobipySolver(new_problem).solve(new_sym)
    assert result.success
    assert result.optimal_objectives[new_sym] == pytest.approx(brute_opt, abs=1e-6)

    sx = round(result.optimal_variables["x"])
    sy = round(result.optimal_variables["y"])
    assert _reference_aggregate(values, compute_probs(sx, sy), agg, maximize) == pytest.approx(brute_opt, abs=1e-6)


@pytest.mark.gurobipy
@pytest.mark.parametrize("maximize", [True, False], ids=["maximize", "minimize"])
@pytest.mark.parametrize("agg", AGGREGATORS)
def test_two_outcome_one_binary(maximize, agg):
    """Aggregators on a two-outcome binary problem match brute force."""
    problem = _make_simple_problem(maximize)
    dists = {
        "f": OutcomeDistribution(
            symbol="f",
            name="f",
            maximize=maximize,
            values=[0.0, 100.0],
            probabilities=["p_0", "p_1"],
        ),
    }
    new_problem, mapping = _apply_aggregator(problem, dists, ["f"], agg)
    new_sym = mapping["f"]
    values = [0.0, 100.0]

    def agg_at(x):
        return _reference_aggregate(values, [x, 1 - x], agg, maximize)

    brute_opt = max(agg_at(0), agg_at(1)) if maximize else min(agg_at(0), agg_at(1))

    result = GurobipySolver(new_problem).solve(new_sym)
    assert result.success
    assert result.optimal_objectives[new_sym] == pytest.approx(brute_opt, abs=1e-6)


@pytest.mark.gurobipy
@pytest.mark.parametrize("maximize", [True, False], ids=["maximize", "minimize"])
def test_cvar_fixed_distribution_limits(maximize):
    """CVaR at alpha near 1 approximates expected value; at worst-outcome mass gives worst value."""
    problem = Problem(
        name="fixed",
        description="fixed distribution",
        variables=[
            Variable(name="x", symbol="x", variable_type=VariableTypeEnum.real, lowerbound=0, upperbound=0),
        ],
        extra_funcs=[
            _ef("p_0", ["Add", 0.2, ["Multiply", 0, "x"]]),
            _ef("p_1", ["Add", 0.3, ["Multiply", 0, "x"]]),
            _ef("p_2", ["Add", 0.5, ["Multiply", 0, "x"]]),
        ],
        objectives=[
            Objective(
                name="f",
                symbol="f",
                func=["Multiply", 0, "x"],
                maximize=maximize,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            ),
        ],
    )
    dists = {
        "f": OutcomeDistribution(
            symbol="f",
            name="f",
            maximize=maximize,
            values=[0.0, 40.0, 100.0],
            probabilities=["p_0", "p_1", "p_2"],
        ),
    }
    values = [0.0, 40.0, 100.0]
    probs = [0.2, 0.3, 0.5]

    new_problem, mapping = add_cvar_dd(problem, dists, ["f"], 0.999999)
    new_sym = mapping["f"]
    result = GurobipySolver(new_problem).solve(new_sym)
    assert result.success
    assert result.optimal_objectives[new_sym] == pytest.approx(_expected(values, probs), abs=1e-3)

    if maximize:
        worst_prob = 0.2
        worst_val = 0.0
    else:
        worst_prob = 0.5
        worst_val = 100.0

    for alpha in [worst_prob, worst_prob / 2]:
        new_problem, mapping = add_cvar_dd(problem, dists, ["f"], alpha)
        new_sym = mapping["f"]
        result = GurobipySolver(new_problem).solve(new_sym)
        assert result.success
        assert result.optimal_objectives[new_sym] == pytest.approx(worst_val, abs=1e-6)


def test_cvar_invalid_alpha():
    """CVaR with alpha outside (0, 1) raises ValueError mentioning alpha."""
    problem = _make_simple_problem()
    dists = {
        "f": OutcomeDistribution(
            symbol="f",
            name="f",
            maximize=True,
            values=[0.0, 100.0],
            probabilities=["p_0", "p_1"],
        ),
    }
    for bad_alpha in [0.0, 1.0, -0.1, 1.5]:
        with pytest.raises(ValueError, match="alpha"):
            add_cvar_dd(problem, dists, ["f"], bad_alpha)


def test_unknown_symbol_raises():
    """A symbol not present in distributions raises ValueError."""
    problem = _make_simple_problem()
    dists = {
        "f": OutcomeDistribution(
            symbol="f",
            name="f",
            maximize=True,
            values=[0.0, 100.0],
            probabilities=["p_0", "p_1"],
        ),
    }
    with pytest.raises(ValueError):
        add_expected_value_dd(problem, dists, ["nonexistent"])


def test_probability_symbol_not_in_problem_raises():
    """A probability symbol missing from the problem's extra functions raises ValueError."""
    problem = _make_simple_problem()
    dists = {
        "f": OutcomeDistribution(
            symbol="f",
            name="f",
            maximize=True,
            values=[0.0, 100.0],
            probabilities=["p_0", "p_missing"],
        ),
    }
    with pytest.raises(ValueError):
        add_expected_value_dd(problem, dists, ["f"])


def test_outcome_distribution_validation():
    """OutcomeDistribution rejects unsorted, duplicate, or mismatched-length inputs."""
    with pytest.raises(ValueError):
        OutcomeDistribution(
            symbol="f",
            name="f",
            maximize=True,
            values=[100.0, 0.0],
            probabilities=["p_0", "p_1"],
        )
    with pytest.raises(ValueError):
        OutcomeDistribution(
            symbol="f",
            name="f",
            maximize=True,
            values=[0.0, 0.0],
            probabilities=["p_0", "p_1"],
        )
    with pytest.raises(ValueError):
        OutcomeDistribution(
            symbol="f",
            name="f",
            maximize=True,
            values=[0.0, 100.0],
            probabilities=["p_0"],
        )
    with pytest.raises(ValueError, match="finite"):
        OutcomeDistribution(
            symbol="f",
            name="f",
            maximize=True,
            values=[0.0, float("nan")],
            probabilities=["p_0", "p_1"],
        )


@pytest.mark.parametrize("maximize", [True, False], ids=["maximize", "minimize"])
def test_added_element_is_objective(maximize):
    """The new element is an Objective with the correct maximize flag."""
    problem = _make_simple_problem(maximize)
    dists = {
        "f": OutcomeDistribution(
            symbol="f",
            name="f",
            maximize=maximize,
            values=[0.0, 100.0],
            probabilities=["p_0", "p_1"],
        ),
    }
    new_problem, mapping = add_expected_value_dd(problem, dists, ["f"])
    new_sym = mapping["f"]
    obj = new_problem.get_objective(new_sym)
    assert obj is not None
    assert obj.maximize == maximize


def _scenario_cvar(maximize, alpha):
    """Scenario CVaR of u = c with c = 0 (p = 0.1) or 100 (p = 0.9)."""
    base_problem = Problem(
        name="s",
        description="s",
        variables=[Variable(name="x", symbol="x", variable_type=VariableTypeEnum.real, lowerbound=0, upperbound=0)],
        constants=[Constant(name="c", symbol="c", value=0.0)],
        objectives=[
            Objective(
                name="u",
                symbol="u",
                func="x + c",
                maximize=maximize,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
        ],
    )
    model = ScenarioModel(
        scenario_tree={"ROOT": ["s_1", "s_2"], "s_1": [], "s_2": []},
        scenario_probabilities={"s_1": 0.1, "s_2": 0.9},
        base_problem=base_problem,
        constants=[
            Constant(name="c (s_1)", symbol="c", value=0.0),
            Constant(name="c (s_2)", symbol="c", value=100.0),
        ],
        scenarios={"s_1": Scenario(constants={"c": 0}), "s_2": Scenario(constants={"c": 1})},
    )
    problem, added = add_conditional_value_at_risk(model, ["u"], alpha=alpha)
    result = GurobipySolver(problem).solve(added["u"])
    assert result.success
    return result.optimal_objectives[added["u"]]


@pytest.mark.gurobipy
@pytest.mark.parametrize("maximize", [True, False], ids=["maximize", "minimize"])
@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.3, 0.9, 0.95])
def test_cvar_matches_scenario_cvar(maximize, alpha):
    """Scenario CVaR and add_cvar_dd agree on the same fixed distribution."""
    problem = Problem(
        name="fixed",
        description="fixed distribution",
        variables=[Variable(name="x", symbol="x", variable_type=VariableTypeEnum.real, lowerbound=0, upperbound=0)],
        extra_funcs=[
            _ef("p_0", ["Add", 0.1, ["Multiply", 0, "x"]]),
            _ef("p_1", ["Add", 0.9, ["Multiply", 0, "x"]]),
        ],
        objectives=[
            Objective(
                name="u",
                symbol="u",
                func=["Multiply", 0, "x"],
                maximize=maximize,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
        ],
    )
    dists = {
        "u": OutcomeDistribution(
            symbol="u", name="u", maximize=maximize, values=[0.0, 100.0], probabilities=["p_0", "p_1"]
        )
    }
    new_problem, added = add_cvar_dd(problem, dists, ["u"], alpha)
    result = GurobipySolver(new_problem).solve(added["u"])

    assert result.success
    assert result.optimal_objectives[added["u"]] == pytest.approx(_scenario_cvar(maximize, alpha), abs=1e-6)
    assert result.optimal_objectives[added["u"]] == pytest.approx(
        _cvar([0.0, 100.0], [0.1, 0.9], alpha, maximize), abs=1e-6
    )


def _continuous_probs(x, y):
    """Outcome probabilities of the continuous problem."""
    return [0.2 - 0.05 * x - 0.1 * y, 0.25 + 0.15 * x + 0.3 * y, 0.55 - 0.1 * x - 0.2 * y]


def _build_continuous_problem(maximize):
    """Build a two-continuous-variable problem with three outcomes and a coupling constraint.

    The constraint x + y <= 1.3 adds the vertices (0.3, 1) and (1, 0.3) to the box, so an optimum
    can lie at an interior x.
    """
    problem = Problem(
        name="cont",
        description="continuous decision",
        variables=[
            Variable(name="x", symbol="x", variable_type=VariableTypeEnum.real, lowerbound=0, upperbound=1),
            Variable(name="y", symbol="y", variable_type=VariableTypeEnum.real, lowerbound=0, upperbound=1),
        ],
        constraints=[
            Constraint(
                name="g",
                symbol="g",
                cons_type=ConstraintTypeEnum.LTE,
                func="x + y - 1.3",
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
        ],
        extra_funcs=[
            _ef("p_0", "0.2 - 0.05*x - 0.1*y"),
            _ef("p_1", "0.25 + 0.15*x + 0.3*y"),
            _ef("p_2", "0.55 - 0.1*x - 0.2*y"),
        ],
        objectives=[
            Objective(
                name="f",
                symbol="f",
                func=["Multiply", 0, "x"],
                maximize=maximize,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
        ],
    )
    dists = {
        "f": OutcomeDistribution(
            symbol="f",
            name="f",
            maximize=maximize,
            values=[0.0, 40.0, 100.0],
            probabilities=["p_0", "p_1", "p_2"],
        )
    }
    return problem, dists


@pytest.mark.gurobipy
@pytest.mark.parametrize("maximize", [True, False], ids=["maximize", "minimize"])
@pytest.mark.parametrize("agg", ["expected", "cvar_0.25", "cvar_0.5"])
def test_continuous_matches_grid(maximize, agg):
    """E and CVaR with continuous decisions match the optimum over a fine grid of (x, y)."""
    problem, dists = _build_continuous_problem(maximize)
    new_problem, mapping = _apply_aggregator(problem, dists, ["f"], agg)
    values = [0.0, 40.0, 100.0]

    def agg_at(x, y):
        return _reference_aggregate(values, _continuous_probs(x, y), agg, maximize)

    n = 200
    grid = [(agg_at(i / n, j / n), i / n) for i in range(n + 1) for j in range(n + 1) if i + j <= 1.3 * n]
    grid_opt, grid_x = max(grid) if maximize else min(grid)
    if agg == "cvar_0.5":
        # Both directions have their unique CVaR optimum at (0.3, 1).
        assert 0.05 < grid_x < 0.95

    result = GurobipySolver(new_problem).solve(mapping["f"])
    assert result.success
    assert result.optimal_objectives[mapping["f"]] == pytest.approx(grid_opt, abs=1e-6)
    assert agg_at(result.optimal_variables["x"], result.optimal_variables["y"]) == pytest.approx(grid_opt, abs=1e-6)
    if agg == "cvar_0.5":
        assert result.optimal_variables["x"] == pytest.approx(grid_x, abs=1e-6)


@pytest.mark.gurobipy
@pytest.mark.parametrize("maximize", [True, False], ids=["maximize", "minimize"])
@pytest.mark.parametrize(("upperbound", "bad_possible"), [(1.0, False), (0.99, True)], ids=["p_zero", "p_above"])
def test_worst_case_continuous(maximize, upperbound, bad_possible):
    """The bad outcome is ignored at p = 0 and counted when p stays clearly above the tolerance."""
    # p_bad = 0.5 - 0.5 x is 0 at x = 1 and at least 0.005 when x <= 0.99.
    bad, good = (0.0, 100.0) if maximize else (100.0, 0.0)
    problem = Problem(
        name="wc",
        description="worst case",
        variables=[
            Variable(name="x", symbol="x", variable_type=VariableTypeEnum.real, lowerbound=0, upperbound=upperbound)
        ],
        extra_funcs=[_ef("p_bad", "0.5 - 0.5*x"), _ef("p_good", "0.5 + 0.5*x")],
        objectives=[
            Objective(
                name="f",
                symbol="f",
                func=["Multiply", 0, "x"],
                maximize=maximize,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
        ],
    )
    probabilities = ["p_bad", "p_good"] if maximize else ["p_good", "p_bad"]
    dists = {
        "f": OutcomeDistribution(
            symbol="f", name="f", maximize=maximize, values=[0.0, 100.0], probabilities=probabilities
        )
    }
    new_problem, mapping = add_worst_case_dd(problem, dists, ["f"])
    result = GurobipySolver(new_problem).solve(mapping["f"])

    assert result.success
    assert result.optimal_objectives[mapping["f"]] == pytest.approx(bad if bad_possible else good, abs=1e-6)


def test_direction_mismatch_raises():
    """A distribution whose maximize differs from its base objective's raises ValueError."""
    problem = _make_simple_problem(maximize=True)
    dists = {
        "f": OutcomeDistribution(
            symbol="f", name="f", maximize=False, values=[0.0, 100.0], probabilities=["p_0", "p_1"]
        )
    }
    for add in (add_expected_value_dd, add_worst_case_dd):
        with pytest.raises(ValueError, match="maximize"):
            add(problem, dists, ["f"])
    with pytest.raises(ValueError, match="maximize"):
        add_cvar_dd(problem, dists, ["f"], 0.5)


def test_symbol_collisions_raise():
    """New symbols clashing with existing ones or with each other raise ValueError."""
    problem = _make_simple_problem()
    dists = {
        "f": OutcomeDistribution(symbol="f", name="f", maximize=True, values=[0.0, 100.0], probabilities=["p_0", "p_1"])
    }

    with pytest.raises(ValueError, match="duplicate"):
        add_expected_value_dd(problem, dists, ["f"], prefix="")
    with pytest.raises(ValueError, match="duplicate"):
        add_expected_value_dd(problem, dists, ["f", "f"])

    worst_problem, _ = add_worst_case_dd(problem, dists, ["f"])
    with pytest.raises(ValueError, match="duplicate"):
        add_worst_case_dd(worst_problem, dists, ["f"])

    cvar_problem, _ = add_cvar_dd(problem, dists, ["f"], 0.5)
    with pytest.raises(ValueError, match="duplicate"):
        add_cvar_dd(cvar_problem, dists, ["f"], 0.25)


@pytest.mark.parametrize("add", ["expected", "worst", "cvar_0.5"])
def test_objective_aggregate_keeps_unit(add):
    """Objective aggregates take the base objective's unit."""
    problem = _make_simple_problem()
    dists = {
        "f": OutcomeDistribution(symbol="f", name="f", maximize=True, values=[0.0, 100.0], probabilities=["p_0", "p_1"])
    }
    new_problem, mapping = _apply_aggregator(problem, dists, ["f"], add)
    assert new_problem.get_objective(mapping["f"]).unit == "EUR"


def test_cvar_key_differs_from_distribution_symbol():
    """A key different from the distribution's symbol names the CVaR objective and its variables."""
    problem = _make_simple_problem()
    dist_f = OutcomeDistribution(symbol="f", name="f", maximize=True, values=[0.0, 100.0], probabilities=["p_0", "p_1"])
    new_problem, mapping = add_cvar_dd(problem, {"cvar_f": dist_f}, ["cvar_f"], alpha=0.25, prefix="")

    assert mapping == {"cvar_f": "cvar_f"}
    objective = new_problem.get_objective("cvar_f")
    assert objective is not None
    assert objective.maximize
    assert objective.unit == "EUR"
    variable_symbols = {v.symbol for v in new_problem.variables}
    assert {"eta_cvar_f", "lambda_cvar_f", "lambdabar_cvar_f", "rho_cvar_f", "rhobar_cvar_f"} <= variable_symbols


def test_non_objective_base_elements():
    """Scalarization function bases give scalarization functions; others give extra functions."""
    problem = _make_simple_problem().add_scalarization(
        ScalarizationFunction(name="s", symbol="s", func=["Multiply", 1, "x"], is_linear=True, is_convex=True)
    )
    dists = {
        sym: OutcomeDistribution(
            symbol=sym, name=sym, maximize=False, values=[0.0, 100.0], probabilities=["p_0", "p_1"]
        )
        for sym in ("s", "g")
    }
    new_problem, mapping = add_expected_value_dd(problem, dists, ["s", "g"])

    assert mapping == {"s": "E_s", "g": "E_g"}
    assert "E_s" in {s.symbol for s in new_problem.scalarization_funcs}
    assert "E_g" in {e.symbol for e in new_problem.extra_funcs}
    assert {o.symbol for o in new_problem.objectives} == {"f"}


def test_cvar_dd_constraint_symbols_match_added_constraints():
    """cvar_dd_constraint_symbols names exactly the constraints add_cvar_dd adds."""
    problem, dists, _ = _build_bruteforce_problem(maximize=True)
    problem = problem.model_copy(
        update={
            "objectives": [
                *problem.objectives,
                Objective(
                    name="g",
                    symbol="g",
                    func=["Multiply", 0, "y"],
                    maximize=False,
                    is_linear=True,
                    is_convex=True,
                    is_twice_differentiable=True,
                ),
            ]
        }
    )
    dists["g"] = OutcomeDistribution(
        symbol="g", name="g", maximize=False, values=[1.0, 3.0, 5.0], probabilities=["p_0", "p_1", "p_2"]
    )
    new_problem, mapping = add_cvar_dd(problem, dists, ["f", "g"], alpha=0.3)

    added = {c.symbol for c in new_problem.constraints} - {c.symbol for c in (problem.constraints or [])}
    expected = set().union(*(cvar_dd_constraint_symbols(mapping[s], 3) for s in ("f", "g")))
    assert added == expected
