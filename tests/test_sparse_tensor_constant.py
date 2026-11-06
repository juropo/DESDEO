"""Tests for `SparseTensorConstant` and matrix-vector `MatMul` constraints across the evaluators."""

import numpy as np
import pytest
from pydantic import ValidationError

from desdeo.problem import (
    Constant,
    Constraint,
    ConstraintTypeEnum,
    Objective,
    PolarsEvaluator,
    Problem,
    PyomoEvaluator,
    SparseTensorConstant,
    SympyEvaluator,
    TensorConstant,
    TensorVariable,
    Variable,
    VariableTypeEnum,
)
from desdeo.problem.json_parser import FormatEnum, MathParser, ParserError
from desdeo.tools import CVXPYSolver, GurobipySolver
from desdeo.tools.cvxpy_solver_interfaces import CVXPYSolverOptions
from desdeo.tools.utils import available_solvers, find_compatible_solvers, guess_best_solver

TOL = 1e-9
HIGHS = CVXPYSolverOptions(solver="HIGHS")
GUROBI_OPTIONS = {"OutputFlag": 0}
LINEAR = {"is_linear": True, "is_convex": True, "is_twice_differentiable": True}


def _sparse(symbol: str, dense: np.ndarray) -> SparseTensorConstant:
    """Build a sparse constant from the nonzero entries of a dense array."""
    nonzero = np.nonzero(dense)
    return SparseTensorConstant(
        name=symbol,
        symbol=symbol,
        shape=list(dense.shape),
        indices=[axis.tolist() for axis in nonzero],
        values=dense[nonzero].tolist(),
    )


def _constant(symbol: str, dense: np.ndarray, *, sparse: bool) -> TensorConstant | SparseTensorConstant:
    if sparse:
        return _sparse(symbol, dense)
    return TensorConstant(name=symbol, symbol=symbol, shape=list(dense.shape), values=dense.tolist())


def _vector_variable(symbol: str, size: int, lower: float, upper: float) -> TensorVariable:
    return TensorVariable(
        name=symbol,
        symbol=symbol,
        variable_type=VariableTypeEnum.real,
        shape=[size],
        lowerbounds=[lower] * size,
        upperbounds=[upper] * size,
    )


def _evaluate(problem: Problem, values: dict[str, np.ndarray]):
    return PolarsEvaluator(problem).evaluate({symbol: [np.asarray(v).tolist()] for symbol, v in values.items()})


# Knapsack-like LP over 10 variables: min 0.5 * c.w subject to a.w <= 2, 0 <= w <= 1.
KNAPSACK_C = np.array([-3.0, 0.0, -1.0, 0.0, 2.0, 0.0, -2.0, 0.0, 0.0, -4.0])
KNAPSACK_A = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0])


def _knapsack_problem(*, sparse: bool) -> Problem:
    return Problem(
        name="Knapsack LP",
        description="A small LP with vector constants in the objective and a constraint.",
        constants=[
            _constant("c", KNAPSACK_C, sparse=sparse),
            _constant("a", KNAPSACK_A, sparse=sparse),
        ],
        variables=[_vector_variable("w", 10, 0.0, 1.0)],
        objectives=[Objective(name="f", symbol="f", func=["MatMul", ["Multiply", 0.5, "c"], "w"], **LINEAR)],
        constraints=[
            Constraint(
                name="capacity",
                symbol="g",
                cons_type=ConstraintTypeEnum.LTE,
                func=["Subtract", ["MatMul", "a", "w"], 2],
                **LINEAR,
            )
        ],
    )


# Flow-like LP: x = b + 0.9 A w, w on the simplex, min c.w + e.x. The optimum is a unique vertex.
_rng = np.random.default_rng(7)
FLOW_A = np.round((_rng.random((5, 12)) < 0.4) * _rng.random((5, 12)), 3)
FLOW_B = np.round(_rng.random(5), 3)
FLOW_C = np.round(_rng.random(12) - 0.5, 3)
FLOW_E = np.round(_rng.random(5), 3)


def _flow_problem(form: str) -> Problem:
    """The flow LP with the flow constraint as one vector constraint ('dense', 'sparse') or as scalar rows ('rows')."""
    if form == "rows":
        flow_constraints = [
            Constraint(
                name=f"flow row {i + 1}",
                symbol=f"flow_{i + 1}",
                cons_type=ConstraintTypeEnum.EQ,
                func=[
                    "Subtract",
                    ["At", "x", i + 1],
                    [
                        "Multiply",
                        0.9,
                        [
                            "Add",
                            0,
                            *[
                                ["Multiply", float(FLOW_A[i, j]), ["At", "w", int(j) + 1]]
                                for j in np.nonzero(FLOW_A[i])[0]
                            ],
                        ],
                    ],
                    float(FLOW_B[i]),
                ],
                **LINEAR,
            )
            for i in range(FLOW_A.shape[0])
        ]
        constants = []
    else:
        flow_constraints = [
            Constraint(
                name="flow",
                symbol="flow",
                cons_type=ConstraintTypeEnum.EQ,
                func=["Subtract", "x", ["Multiply", 0.9, ["MatMul", "A", "w"]], "b"],
                **LINEAR,
            )
        ]
        constants = [
            _constant("A", FLOW_A, sparse=form == "sparse"),
            TensorConstant(name="b", symbol="b", shape=[5], values=FLOW_B.tolist()),
        ]

    return Problem(
        name="Flow LP",
        description="A small LP with a vector-valued flow constraint.",
        constants=[
            *constants,
            TensorConstant(name="c", symbol="c", shape=[12], values=FLOW_C.tolist()),
            TensorConstant(name="e", symbol="e", shape=[5], values=FLOW_E.tolist()),
        ],
        variables=[_vector_variable("w", 12, 0.0, 1.0), _vector_variable("x", 5, -10.0, 10.0)],
        objectives=[
            Objective(name="f", symbol="f", func=["Add", ["MatMul", "c", "w"], ["MatMul", "e", "x"]], **LINEAR)
        ],
        constraints=[
            *flow_constraints,
            Constraint(
                name="simplex",
                symbol="simplex",
                cons_type=ConstraintTypeEnum.EQ,
                func=["Subtract", ["Sum", "w"], 1],
                **LINEAR,
            ),
        ],
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"shape": [3], "indices": [[3]], "values": [1.0]}, "out of bounds"),
        ({"shape": [2, 3], "indices": [[0, 1], [2, -1]], "values": [1.0, 2.0]}, "out of bounds"),
        ({"shape": [3], "indices": [[0, 1]], "values": [1.0]}, "same length"),
        ({"shape": [2, 3], "indices": [[0, 1]], "values": [1.0, 2.0]}, "one index list per axis"),
        ({"shape": [2, 3], "indices": [[0, 0], [1, 1]], "values": [1.0, 2.0]}, "duplicate"),
        ({"shape": [2, 2, 2], "indices": [[0], [0], [0]], "values": [1.0]}, "one or two positive axes"),
        ({"shape": [0], "indices": [[]], "values": []}, "one or two positive axes"),
    ],
)
def test_validation_errors(kwargs, match):
    """Invalid shapes and coordinates are rejected."""
    with pytest.raises(ValidationError, match=match):
        SparseTensorConstant(name="T", symbol="T", **kwargs)


def test_to_scipy_and_to_dense():
    """A vector becomes a (1, n) CSR array, a matrix keeps its shape, and the dense values match."""
    vector = SparseTensorConstant(name="v", symbol="v", shape=[4], indices=[[1, 3]], values=[2.0, -1.0])
    assert vector.to_scipy().shape == (1, 4)
    assert vector.to_scipy().format == "csr"
    np.testing.assert_array_equal(vector.to_dense(), [0.0, 2.0, 0.0, -1.0])

    matrix = SparseTensorConstant(name="M", symbol="M", shape=[2, 3], indices=[[0, 1], [2, 0]], values=[5.0, 7.0])
    assert matrix.to_scipy().shape == (2, 3)
    np.testing.assert_array_equal(matrix.to_dense(), [[0.0, 0.0, 5.0], [7.0, 0.0, 0.0]])


@pytest.mark.gurobipy
def test_knapsack_gurobipy_sparse_matches_dense():
    """Gurobi gives the same optimum for the dense and the sparse knapsack LP."""
    dense = GurobipySolver(_knapsack_problem(sparse=False), options=GUROBI_OPTIONS).solve("f")
    sparse = GurobipySolver(_knapsack_problem(sparse=True), options=GUROBI_OPTIONS).solve("f")

    assert dense.success and sparse.success
    assert np.isclose(dense.optimal_objectives["f"], -3.5, atol=TOL)
    assert abs(sparse.optimal_objectives["f"] - dense.optimal_objectives["f"]) < TOL
    assert abs(sparse.constraint_values["g"] - dense.constraint_values["g"]) < TOL
    np.testing.assert_allclose(sparse.optimal_variables["w"], dense.optimal_variables["w"], atol=TOL)


@pytest.mark.cvxpy
@pytest.mark.polars
def test_knapsack_cvxpy_and_polars_sparse_matches_dense():
    """CVXPY with HiGHS, and Polars evaluation of its optimum, agree for the dense and the sparse knapsack LP."""
    dense_problem = _knapsack_problem(sparse=False)
    sparse_problem = _knapsack_problem(sparse=True)
    dense = CVXPYSolver(dense_problem, HIGHS).solve("f")
    sparse = CVXPYSolver(sparse_problem, HIGHS).solve("f")

    assert dense.success and sparse.success
    assert np.isclose(dense.optimal_objectives["f"], -3.5, atol=TOL)
    assert abs(sparse.optimal_objectives["f"] - dense.optimal_objectives["f"]) < TOL
    np.testing.assert_allclose(sparse.optimal_variables["w"], dense.optimal_variables["w"], atol=TOL)

    dense_eval = _evaluate(dense_problem, dense.optimal_variables)
    sparse_eval = _evaluate(sparse_problem, dense.optimal_variables)
    for symbol in ("f", "g"):
        assert abs(sparse_eval[symbol][0] - dense_eval[symbol][0]) < TOL
    assert abs(sparse_eval["f"][0] - dense.optimal_objectives["f"]) < TOL


@pytest.mark.polars
@pytest.mark.parametrize("shape", [[100_000], [1, 100_000]])
def test_polars_large_sparse_constant_is_not_replicated(shape):
    """A sparse constant stays out of the evaluation dataframe, and MatMul with it is evaluated per row."""
    n = 100_000
    indices = np.arange(0, n, 1000)
    dense = np.zeros(n)
    dense[indices] = np.arange(1, len(indices) + 1)
    sparse = SparseTensorConstant(
        name="T",
        symbol="T",
        shape=shape,
        indices=[[0] * len(indices), indices.tolist()] if len(shape) == 2 else [indices.tolist()],
        values=dense[indices].tolist(),
    )
    problem = Problem(
        name="Large sparse",
        description="A long sparse vector.",
        constants=[sparse],
        variables=[_vector_variable("w", n, 0.0, 1.0)],
        objectives=[Objective(name="f", symbol="f", func=["MatMul", "T", "w"], **LINEAR)],
    )

    ws = np.vstack([np.full(n, 1.0), np.linspace(0.0, 1.0, n), np.zeros(n)])
    result = PolarsEvaluator(problem).evaluate({"w": ws.tolist()})

    assert "T" not in result.columns
    expected = ws @ dense
    values = np.array(result["f"].to_list(), dtype=float).reshape(3, -1)
    assert values.shape == (3, 1)
    np.testing.assert_allclose(values[:, 0], expected, rtol=1e-12)


def test_pyomo_and_sympy_evaluators_raise():
    """The Pyomo and Sympy evaluators reject sparse constants, naming the constant."""
    with pytest.raises(NotImplementedError, match="'c'"):
        PyomoEvaluator(_knapsack_problem(sparse=True))

    scalar_problem = Problem(
        name="Scalar",
        description="Scalar variables with a sparse constant.",
        constants=[SparseTensorConstant(name="c", symbol="c", shape=[2], indices=[[0]], values=[1.0])],
        variables=[Variable(name="x", symbol="x", variable_type=VariableTypeEnum.real, lowerbound=0, upperbound=1)],
        objectives=[Objective(name="f", symbol="f", func=["Multiply", 2, "x"], **LINEAR)],
    )
    with pytest.raises(NotImplementedError, match="'c'"):
        SympyEvaluator(scalar_problem)


def test_solver_selection_excludes_pyomo_and_sympy_backed_solvers():
    """`find_compatible_solvers` and `guess_best_solver` don't offer Pyomo- or Sympy-backed solvers."""
    excluded = {
        available_solvers[name]["constructor"]
        for name in ("pyomo_bonmin", "pyomo_ipopt", "pyomo_gurobi", "pyomo_cbc", "nevergrad")
    }
    problem = _knapsack_problem(sparse=True)

    solvers = find_compatible_solvers(problem)
    assert solvers
    assert not excluded.intersection(solvers)
    assert guess_best_solver(problem) not in excluded

    dense_solvers = find_compatible_solvers(_knapsack_problem(sparse=False))
    assert set(solvers) <= set(dense_solvers)


@pytest.mark.json
def test_json_round_trip():
    """A problem with sparse constants round-trips through JSON."""
    problem = _knapsack_problem(sparse=True)
    reloaded = Problem.model_validate_json(problem.model_dump_json())

    assert reloaded == problem
    assert all(isinstance(const, SparseTensorConstant) for const in reloaded.constants)


def _scaled_flow_problem(*, sparse: bool, scalar_first: bool, multiplier: str | list = "k") -> Problem:
    """The flow LP with a multiplier built from the scalar constant `k` scaling the 1-D `c` and the 2-D `A`."""

    def scaled(symbol: str) -> list:
        return ["Multiply", multiplier, symbol] if scalar_first else ["Multiply", symbol, multiplier]

    return Problem(
        name="Scaled flow LP",
        description="A small LP with sparse constants multiplied by a named scalar constant.",
        constants=[
            Constant(name="k", symbol="k", value=0.9),
            _constant("A", FLOW_A, sparse=sparse),
            _constant("c", FLOW_C, sparse=sparse),
            TensorConstant(name="b", symbol="b", shape=[5], values=FLOW_B.tolist()),
            TensorConstant(name="e", symbol="e", shape=[5], values=FLOW_E.tolist()),
        ],
        variables=[_vector_variable("w", 12, 0.0, 1.0), _vector_variable("x", 5, -10.0, 10.0)],
        objectives=[
            Objective(name="f", symbol="f", func=["Add", ["MatMul", scaled("c"), "w"], ["MatMul", "e", "x"]], **LINEAR)
        ],
        constraints=[
            Constraint(
                name="flow",
                symbol="flow",
                cons_type=ConstraintTypeEnum.EQ,
                func=["Subtract", "x", ["MatMul", scaled("A"), "w"], "b"],
                **LINEAR,
            ),
            Constraint(
                name="budget",
                symbol="budget",
                cons_type=ConstraintTypeEnum.LTE,
                func=["Subtract", ["MatMul", scaled("c"), "w"], 0.1],
                **LINEAR,
            ),
            Constraint(
                name="simplex",
                symbol="simplex",
                cons_type=ConstraintTypeEnum.EQ,
                func=["Subtract", ["Sum", "w"], 1],
                **LINEAR,
            ),
        ],
    )


@pytest.mark.gurobipy
@pytest.mark.cvxpy
@pytest.mark.polars
@pytest.mark.parametrize("scalar_first", [True, False])
@pytest.mark.parametrize("multiplier", ["k", ["Multiply", "k", 2], ["Subtract", 1, "k"], ["Negate", "k"]])
def test_named_scalar_constant_times_sparse(scalar_first, multiplier):
    """A scalar constant expression times 1-D and 2-D sparse constants agrees with the dense model in every backend."""
    dense_problem = _scaled_flow_problem(sparse=False, scalar_first=scalar_first, multiplier=multiplier)
    sparse_problem = _scaled_flow_problem(sparse=True, scalar_first=scalar_first, multiplier=multiplier)

    for solve in (
        lambda p: GurobipySolver(p, options=GUROBI_OPTIONS).solve("f"),
        lambda p: CVXPYSolver(p, HIGHS).solve("f"),
    ):
        dense = solve(dense_problem)
        sparse = solve(sparse_problem)
        assert dense.success and sparse.success
        assert abs(sparse.optimal_objectives["f"] - dense.optimal_objectives["f"]) < TOL
        for symbol in ("flow", "budget"):
            np.testing.assert_allclose(
                sparse.constraint_values[symbol], dense.constraint_values[symbol], atol=TOL, rtol=0
            )

        dense_eval = _evaluate(dense_problem, sparse.optimal_variables)
        sparse_eval = _evaluate(sparse_problem, sparse.optimal_variables)
        assert abs(sparse_eval["f"][0] - dense_eval["f"][0]) < TOL
        assert abs(sparse_eval["f"][0] - sparse.optimal_objectives["f"]) < TOL
        assert abs(sparse_eval["budget"][0] - dense_eval["budget"][0]) < TOL
        np.testing.assert_allclose(sparse_eval["flow"].to_list()[0], dense_eval["flow"].to_list()[0], atol=TOL, rtol=0)


@pytest.mark.polars
@pytest.mark.parametrize("symbol", ["A", "c"])
def test_polars_sparse_times_dense_matrix_chain(symbol):
    """A sparse constant followed by a dense matrix in a `MatMul` chain evaluates like the numpy chain, per row."""
    rng = np.random.default_rng(3)
    matrix = np.round(rng.random((12, 12)) - 0.5, 3)
    values = {"w": [np.ones(12).tolist(), np.round(rng.random(12), 3).tolist()]}
    problem = Problem(
        name="Sparse chain",
        description="A sparse constant times a dense matrix times a vector.",
        constants=[
            _sparse("A", FLOW_A),
            _sparse("c", FLOW_C),
            TensorConstant(name="B", symbol="B", shape=[12, 12], values=matrix.tolist()),
        ],
        variables=[_vector_variable("w", 12, 0.0, 1.0)],
        objectives=[Objective(name="f", symbol="f", func=["Sum", ["MatMul", symbol, "B", "w"]], **LINEAR)],
    )

    dense_problem = problem.model_copy(
        update={
            "constants": [
                _constant("A", FLOW_A, sparse=False),
                _constant("c", FLOW_C, sparse=False),
                *problem.constants[2:],
            ]
        }
    )

    result = PolarsEvaluator(problem).evaluate(values)["f"].to_numpy()
    dense_result = PolarsEvaluator(dense_problem).evaluate(values)["f"].to_numpy()

    dense = FLOW_A if symbol == "A" else FLOW_C
    expected = [np.sum(dense @ matrix @ np.asarray(w)) for w in values["w"]]
    np.testing.assert_allclose(result, expected, atol=TOL, rtol=0)
    np.testing.assert_allclose(result, dense_result, atol=TOL, rtol=0)


@pytest.mark.polars
@pytest.mark.parametrize("n_rows", [1, 3])
@pytest.mark.parametrize(
    ("func", "expected", "scalar"),
    [
        (["MatMul", "c", "W"], lambda c, a, b, w, v, wm: c @ wm, False),
        (["MatMul", "c", "B", "w"], lambda c, a, b, w, v, wm: c @ b @ w, True),
        (["MatMul", "c", "W", "v"], lambda c, a, b, w, v, wm: c @ wm @ v, True),
        (["MatMul", "A", "B", "w"], lambda c, a, b, w, v, wm: a @ b @ w, False),
    ],
)
def test_polars_dense_matmul_chain(func, expected, scalar, n_rows):
    """A dense `MatMul` chain with per-row vectors and matrices evaluates like the numpy chain, per row."""
    rng = np.random.default_rng(5)
    c, a, b = rng.random(6), rng.random((4, 6)), rng.random((6, 6))
    problem = Problem(
        name="Dense chain",
        description="Dense constants times per-row vectors and matrices.",
        constants=[
            _constant("c", c, sparse=False),
            _constant("A", a, sparse=False),
            _constant("B", b, sparse=False),
        ],
        variables=[
            _vector_variable("w", 6, 0.0, 1.0),
            _vector_variable("v", 3, 0.0, 1.0),
            TensorVariable(
                name="W",
                symbol="W",
                variable_type=VariableTypeEnum.real,
                shape=[6, 3],
                lowerbounds=np.zeros((6, 3)).tolist(),
                upperbounds=np.ones((6, 3)).tolist(),
            ),
        ],
        objectives=[Objective(name="f", symbol="f", func=func, **LINEAR)],
    )
    rows = [(rng.random(6), rng.random(3), rng.random((6, 3))) for _ in range(n_rows)]
    values = {
        "w": [w.tolist() for w, _, _ in rows],
        "v": [v.tolist() for _, v, _ in rows],
        "W": [wm.tolist() for _, _, wm in rows],
    }

    result = PolarsEvaluator(problem).evaluate(values)["f"].to_list()

    assert len(result) == n_rows
    for value, (w, v, wm) in zip(result, rows, strict=True):
        if scalar:
            assert isinstance(value, float)
        np.testing.assert_allclose(value, expected(c, a, b, w, v, wm), atol=TOL, rtol=0)


@pytest.mark.polars
@pytest.mark.parametrize("func", ["c", ["Multiply", 2, "c"]])
def test_polars_bare_sparse_expression_raises(func):
    """A function that evaluates to a bare sparse constant raises a parser error naming it, if it can."""
    parser = MathParser(sparse_constants={"c": _sparse("c", KNAPSACK_C).to_scipy_operand()})

    with pytest.raises(ParserError, match="'c'" if func == "c" else "sparse constant"):
        parser.parse(func)


@pytest.mark.parametrize(
    "func",
    [
        ["Add", "A", 1],
        ["MatMul", "w", "A"],
        ["Multiply", "A", "w"],
        ["Sum", "A"],
        ["Extract", "A", 1],
    ],
)
@pytest.mark.parametrize("to_format", [FormatEnum.polars, FormatEnum.gurobipy, FormatEnum.cvxpy])
def test_unsupported_operations_raise(func, to_format):
    """Operations other than left-operand MatMul and scalar Multiply on a sparse constant raise."""
    matrix = _sparse("A", FLOW_A).to_scipy()
    parser = MathParser(to_format=to_format, sparse_constants={"A": matrix})

    if to_format == FormatEnum.polars:
        with pytest.raises(ParserError, match="sparse constant"):
            parser.parse(func)
    else:
        scope = {"A": matrix, "w": np.ones(12)}
        with pytest.raises(ParserError, match="sparse constant"):
            parser.parse(func, scope.__getitem__)


def _assert_flow_results_agree(results: dict):
    rows = results["rows"]
    assert rows.success
    for form in ("dense", "sparse"):
        assert results[form].success
        assert abs(results[form].optimal_objectives["f"] - rows.optimal_objectives["f"]) < TOL
        for symbol in ("w", "x"):
            np.testing.assert_allclose(
                results[form].optimal_variables[symbol], rows.optimal_variables[symbol], atol=TOL, rtol=0
            )
        assert np.asarray(results[form].constraint_values["flow"]).shape == (5,)
        np.testing.assert_allclose(results[form].constraint_values["flow"], 0.0, atol=TOL)

        evaluated = _evaluate(_flow_problem(form), results[form].optimal_variables)
        residuals = np.array(evaluated["flow"].to_list()[0])
        assert residuals.shape == (5,)
        np.testing.assert_allclose(residuals, 0.0, atol=TOL)
        assert abs(evaluated["f"][0] - rows.optimal_objectives["f"]) < TOL


@pytest.mark.gurobipy
@pytest.mark.polars
def test_vector_constraint_gurobipy():
    """Gurobi solves the vector flow constraint, dense and sparse, like the scalar-row model."""
    results = {
        form: GurobipySolver(_flow_problem(form), options=GUROBI_OPTIONS).solve("f")
        for form in ("rows", "dense", "sparse")
    }
    _assert_flow_results_agree(results)


@pytest.mark.cvxpy
@pytest.mark.polars
def test_vector_constraint_cvxpy():
    """CVXPY with HiGHS solves the vector flow constraint, dense and sparse, like the scalar-row model."""
    results = {form: CVXPYSolver(_flow_problem(form), HIGHS).solve("f") for form in ("rows", "dense", "sparse")}
    _assert_flow_results_agree(results)
