"""Tests that SparseTensorConstants survive the round trip through `ProblemDB`."""

import pytest

from desdeo.api.models import ProblemDB
from desdeo.problem.schema import (
    Constant,
    Objective,
    Problem,
    SparseTensorConstant,
    TensorConstant,
    TensorVariable,
    VariableTypeEnum,
)

pytestmark = pytest.mark.api


def test_sparse_tensor_constant_round_trip_through_problemdb(session_and_user):
    """A problem with sparse, dense and scalar constants is stored and reloaded unchanged."""
    session = session_and_user["session"]
    user = session_and_user["user"]

    problem = Problem(
        name="Sparse constant problem",
        description="A problem with a sparse matrix and a sparse vector.",
        constants=[
            Constant(name="gamma", symbol="gamma", value=0.9),
            TensorConstant(name="b", symbol="b", shape=[2], values=[1.0, 2.0]),
            SparseTensorConstant(name="A", symbol="A", shape=[2, 3], indices=[[0, 1], [2, 0]], values=[0.5, 1.5]),
            SparseTensorConstant(name="c", symbol="c", shape=[3], indices=[[1]], values=[-1.0]),
        ],
        variables=[
            TensorVariable(
                name="w",
                symbol="w",
                variable_type=VariableTypeEnum.real,
                shape=[3],
                lowerbounds=[0.0, 0.0, 0.0],
                upperbounds=[1.0, 1.0, 1.0],
            )
        ],
        objectives=[
            Objective(
                name="f",
                symbol="f",
                func=["MatMul", "c", "w"],
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
        ],
    )

    problem_db = ProblemDB.from_problem(problem, user=user)
    session.add(problem_db)
    session.commit()
    session.refresh(problem_db)

    reloaded = Problem.from_problemdb(session.get(ProblemDB, problem_db.id))

    assert len(problem_db.sparse_tensor_constants) == 2
    by_symbol = {const.symbol: const for const in reloaded.constants}
    assert by_symbol == {const.symbol: const for const in problem.constants}
    assert isinstance(by_symbol["A"], SparseTensorConstant)
