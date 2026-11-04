"""Tests for the gurobipy solver."""

import gurobipy as gp
import numpy as np
import pytest

from desdeo.problem import (
    Constraint,
    ConstraintTypeEnum,
    Objective,
    ObjectiveTypeEnum,
    Problem,
    ScalarizationFunction,
    TensorVariable,
    Variable,
    VariableTypeEnum,
)
from desdeo.problem.gurobipy_evaluator import GurobipyEvaluator, GurobipyEvaluatorError
from desdeo.problem.testproblems import (
    simple_constrained_quadratic_tensor_test_problem,
    simple_knapsack_vectors,
    simple_linear_test_problem,
)
from desdeo.tools import GurobipySolver, PersistentGurobipySolver


@pytest.mark.slow
@pytest.mark.gurobipy
def test_gurobipy_solver():
    """Tests the bonmin solver."""
    problem = simple_linear_test_problem()
    solver = GurobipySolver(problem)

    results = solver.solve("f_1")

    assert results.success

    xs = results.optimal_variables
    assert np.isclose(xs["x_1"], 4.2, atol=1e-8)
    assert np.isclose(xs["x_2"], 2.1, atol=1e-8)


@pytest.mark.slow
@pytest.mark.gurobipy
def test_gurobipy_persistent_solver():
    """Tests the gurobipy solver."""
    problem = simple_linear_test_problem()
    solver = PersistentGurobipySolver(problem)

    results = solver.solve("f_1")

    assert results.success

    xs = results.optimal_variables
    assert np.isclose(xs["x_1"], 4.2, atol=1e-8)
    assert np.isclose(xs["x_2"], 2.1, atol=1e-8)

    testvar = Variable(name="test_y", symbol="y", variable_type=VariableTypeEnum.integer, lowerbound=-20, upperbound=30)
    solver.add_variable(testvar)
    assert isinstance(solver.evaluator.get_expression_by_name("y"), gp.Var)

    testconstr = Constraint(
        name="testconstraint", symbol="c_test", cons_type=ConstraintTypeEnum.EQ, func=["Add", "x_1", "x_2", "y", -20]
    )
    solver.add_constraint(testconstr)
    assert solver.evaluator.model.getConstrByName("c_test") is not None

    testobjective = Objective(name="testobjective", symbol="f_test", func=["Add", "y"])
    solver.add_objective(testobjective)
    assert isinstance(solver.evaluator.get_expression_by_name("f_test"), gp.Var)

    testscal = ScalarizationFunction(name="test scalarization function", symbol="scal", func=["Add", "f_test", "f_1"])
    solver.evaluator.add_scalarization_function(testscal)
    assert isinstance(solver.evaluator.get_expression_by_name("scal"), gp.LinExpr)

    solver.solve("scal")
    assert np.isclose(solver.evaluator.get_expression_by_name("scal").getValue(), 20)

    solver.remove_constraint("c_test")
    assert solver.evaluator.model.getConstrByName("c_test") is None

    solver.remove_variable("y")
    with pytest.raises(GurobipyEvaluatorError):
        solver.evaluator.get_expression_by_name("y")

    # Check that the solver can still solve the original problem
    # after removing the added variables and constraints
    results = solver.solve("f_1")

    assert results.success

    xs = results.optimal_variables
    assert np.isclose(xs["x_1"], 4.2, atol=1e-8)
    assert np.isclose(xs["x_2"], 2.1, atol=1e-8)

    testvar = TensorVariable(
        name="test_y",
        symbol="y",
        variable_type=VariableTypeEnum.integer,
        shape=(2, 2),
        lowerbounds=[[-20, -20], [-20, -20]],
        upperbounds=[[30, 30], [30, 30]],
    )
    solver.add_variable(testvar)
    assert isinstance(solver.evaluator.get_expression_by_name("y"), gp.MVar)

    solver.remove_variable("y")
    with pytest.raises(GurobipyEvaluatorError):
        solver.evaluator.get_expression_by_name("y")


@pytest.mark.slow
@pytest.mark.gurobipy
def test_gurobipy_solver_with_tensors():
    """Test gurobipy solver with a problem with TensorVariables."""
    problem = simple_knapsack_vectors()
    solver = GurobipySolver(problem)

    results = solver.solve("f_1_min")

    assert results.success
    xs, ys = results.optimal_variables, results.optimal_objectives

    assert np.allclose(xs["X"], [1.0, 1.0, 0.0, 0.0])
    assert np.isclose(ys["f_1"], 8.0)
    assert np.isclose(ys["f_2"], 6.0)

    results = solver.solve("f_2_min")

    assert results.success
    xs, ys = results.optimal_variables, results.optimal_objectives

    assert np.allclose(xs["X"], [0.0, 0.0, 1.0, 0.0])
    assert np.isclose(ys["f_1"], 6.0)
    assert np.isclose(ys["f_2"], 7.0)


@pytest.mark.slow
@pytest.mark.gurobipy
def test_gurobipy_solver_qp_with_tensors():
    """Test gurobipy solver with a quadratic problem with TensorVariables."""
    problem = simple_constrained_quadratic_tensor_test_problem()
    solver = GurobipySolver(problem)

    results = solver.solve("f_1_min")

    assert results.success
    xs, ys = results.optimal_variables, results.optimal_objectives

    assert np.allclose(xs["X"], [2 / 3, 2 / 3])
    assert np.isclose(ys["f_1"], -((2 / 3) ** 2))


def _chained_objectives_problem() -> Problem:
    """A problem whose second objective references the first one by symbol."""
    return Problem(
        name="Chained objectives",
        description="f_2 is defined in terms of f_1's symbol.",
        variables=[
            Variable(
                name="x",
                symbol="x",
                variable_type=VariableTypeEnum.real,
                lowerbound=0.0,
                upperbound=10.0,
                initial_value=1.0,
            )
        ],
        objectives=[
            Objective(
                name="f_1",
                symbol="f_1",
                func=["Add", "x", 1],
                maximize=False,
                objective_type=ObjectiveTypeEnum.analytical,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            ),
            Objective(
                name="f_2",
                symbol="f_2",
                func=["Multiply", 2, "f_1"],
                maximize=False,
                objective_type=ObjectiveTypeEnum.analytical,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            ),
        ],
    )


@pytest.mark.gurobipy
def test_objective_may_reference_earlier_objective():
    """An objective can be defined in terms of an objective declared before it.

    The expressions are registered on the evaluator as they are parsed, so a later
    objective resolves an earlier one's symbol.  They used to be collected in a local
    dict published only once every objective had been parsed, which made such a
    reference unresolvable and left gurobipy out of step with the other evaluators.
    """
    problem = _chained_objectives_problem()
    evaluator = GurobipyEvaluator(problem)

    # f_1 = x + 1 was substituted into f_2, giving f_2 = 2x + 2.
    f_1, f_2 = evaluator.objective_functions["f_1"], evaluator.objective_functions["f_2"]
    assert f_2.getConstant() == pytest.approx(2 * f_1.getConstant())
    assert f_2.size() == f_1.size() == 1
    assert f_2.getVar(0).VarName == f_1.getVar(0).VarName
    assert f_2.getCoeff(0) == pytest.approx(2 * f_1.getCoeff(0))

    # And the problem solves: minimising 2x + 2 over x in [0, 10] puts x at 0.
    results = GurobipySolver(problem).solve("f_2")
    assert results.success
    assert results.optimal_objectives["f_2"] == pytest.approx(2 * results.optimal_objectives["f_1"])
    assert results.optimal_objectives["f_2"] == pytest.approx(2.0)


@pytest.mark.gurobipy
def test_objective_referencing_unknown_symbol_still_raises():
    """Referencing a symbol that is not defined anywhere remains an error."""
    problem = _chained_objectives_problem()
    broken = problem.model_copy(
        update={
            "objectives": [
                problem.objectives[0],
                problem.objectives[1].model_copy(update={"func": ["Multiply", 2, "f_nonexistent"]}),
            ]
        }
    )

    with pytest.raises(GurobipyEvaluatorError, match="f_nonexistent"):
        GurobipyEvaluator(broken)


@pytest.mark.gurobipy
@pytest.mark.parametrize("solver_class", [GurobipySolver, PersistentGurobipySolver])
def test_custom_options_keep_defaults(solver_class):
    """Custom options are set alongside the defaults, and override them where they overlap."""
    solver = solver_class(simple_linear_test_problem(), options={"TimeLimit": 30, "OutputFlag": 1})
    params = solver.evaluator.model.Params

    assert params.IntegralityFocus == 1
    assert params.LogToConsole == 0
    assert params.TimeLimit == 30
    assert params.OutputFlag == 1


def _knapsack_problem() -> tuple[Problem, list[int], int]:
    """A small binary knapsack, returned with its item weights and capacity."""
    values = [10, 13, 7, 8, 9, 12, 6, 11]
    weights = [5, 7, 4, 5, 6, 8, 3, 6]
    capacity = 20
    symbols = [f"x_{i + 1}" for i in range(len(values))]

    return (
        Problem(
            name="Knapsack",
            description="Binary knapsack.",
            variables=[
                Variable(name=s, symbol=s, variable_type=VariableTypeEnum.binary, lowerbound=0, upperbound=1)
                for s in symbols
            ],
            objectives=[
                Objective(
                    name="f",
                    symbol="f",
                    func=["Add", *[["Multiply", v, s] for v, s in zip(values, symbols, strict=True)]],
                    maximize=True,
                    objective_type=ObjectiveTypeEnum.analytical,
                    is_linear=True,
                    is_convex=True,
                    is_twice_differentiable=True,
                )
            ],
            constraints=[
                Constraint(
                    name="capacity",
                    symbol="g",
                    func=["Add", *[["Multiply", w, s] for w, s in zip(weights, symbols, strict=True)], -capacity],
                    cons_type=ConstraintTypeEnum.LTE,
                    is_linear=True,
                    is_convex=True,
                    is_twice_differentiable=True,
                )
            ],
        ),
        weights,
        capacity,
    )


@pytest.mark.gurobipy
@pytest.mark.parametrize("solver_class", [GurobipySolver, PersistentGurobipySolver])
def test_incumbent_returned_when_not_optimal(solver_class):
    """When the solve stops with a feasible incumbent, its values are returned with success=False."""
    problem, weights, capacity = _knapsack_problem()
    solver = solver_class(problem, options={"SolutionLimit": 1})
    results = solver.solve("f")

    assert solver.evaluator.model.status == gp.GRB.SOLUTION_LIMIT
    assert results.success is False
    assert "not proven optimal" in results.message

    xs = [results.optimal_variables[f"x_{i + 1}"] for i in range(len(weights))]
    assert all(np.isfinite(x) for x in xs)
    assert all(np.isclose(x, 0) or np.isclose(x, 1) for x in xs)
    assert sum(w * x for w, x in zip(weights, xs, strict=True)) <= capacity + 1e-6
    assert np.isfinite(results.optimal_objectives["f"])


@pytest.mark.gurobipy
@pytest.mark.parametrize("solver_class", [GurobipySolver, PersistentGurobipySolver])
def test_infeasible_returns_nan(solver_class):
    """An infeasible model has no incumbent, so NaN values are returned with success=False."""
    problem, _, _ = _knapsack_problem()
    infeasible = problem.model_copy(
        update={
            "constraints": [
                *problem.constraints,
                Constraint(
                    name="impossible",
                    symbol="h",
                    func=["Add", "x_1", 1],
                    cons_type=ConstraintTypeEnum.LTE,
                    is_linear=True,
                    is_convex=True,
                    is_twice_differentiable=True,
                ),
            ]
        }
    )
    results = solver_class(infeasible).solve("f")

    assert results.success is False
    assert all(np.isnan(x) for x in results.optimal_variables.values())
    assert all(np.isnan(f) for f in results.optimal_objectives.values())
