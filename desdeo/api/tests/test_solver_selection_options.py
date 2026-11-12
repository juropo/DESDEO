"""Tests for solver selection options in `SolverSelectionMetadata`."""

import logging
from collections import OrderedDict

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from desdeo.api.models import ProblemDB, SolverSelectionMetadata, StateDB
from desdeo.api.models.problem import ProblemMetaDataDB
from desdeo.api.routers import cumulus as cumulus_router
from desdeo.api.routers import utils
from desdeo.api.routers.problem import _solver_selection, effective_solver_options
from desdeo.api.routers.utils import copy_problem_metadata
from desdeo.problem import Problem
from desdeo.tools import CVXPYSolver, GurobipyOptions, GurobipySolver
from desdeo.tools.cvxpy_solver_interfaces import CVXPYSolverOptions
from desdeo.tools.utils import available_solvers, normalize_solver_options

from .conftest import login, post_json
from .test_cumulus_hypothetical import _compile_oil_wildcatter, _hypothetical
from .test_stationary_model_metadata import _add

pytestmark = pytest.mark.graphical


def _stored_options(problem_db: ProblemDB):
    """The options on the problem's last solver selection row."""
    return _solver_selection(problem_db).solver_options


def test_assign_solver_with_options_gurobipy(client: TestClient, session_and_user: dict):
    """Gurobipy options are stored and round-trip via get_metadata and _stored_options."""
    session: Session = session_and_user["session"]
    problem_db = _add(session_and_user)
    token = login(client)

    resp = post_json(
        client,
        "/problem/assign_solver",
        {
            "problem_id": problem_db.id,
            "solver_string_representation": "gurobipy",
            "solver_options": {"FeasibilityTol": 1e-9},
        },
        token,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"message": "OK"}

    resp = post_json(
        client,
        "/problem/get_metadata",
        {"problem_id": problem_db.id, "metadata_type": "solver_selection_metadata"},
        token,
    )
    assert resp.status_code == 200, resp.text
    rows = resp.json()
    assert len(rows) == 1
    row = SolverSelectionMetadata.model_validate(rows[0])
    assert row.solver_options == {"FeasibilityTol": 1e-9}

    session.refresh(problem_db)
    session.refresh(problem_db.problem_metadata)
    assert _stored_options(problem_db) == {"FeasibilityTol": 1e-9}


def test_assign_solver_rejects_invalid_gurobi_value(client: TestClient, session_and_user: dict):
    """A Gurobi parameter of the wrong type gives 422 at /assign_solver instead of failing at solve time."""
    problem_db = _add(session_and_user)
    resp = post_json(
        client,
        "/problem/assign_solver",
        {
            "problem_id": problem_db.id,
            "solver_string_representation": "gurobipy",
            "solver_options": {"FeasibilityTol": "abc"},
        },
        login(client),
    )
    assert resp.status_code == 422
    assert "FeasibilityTol" in resp.text


def test_assign_solver_invalid_and_no_options(client: TestClient, session_and_user: dict):
    """Invalid cvxpy options give 422; omitting options stores None and returns 200."""
    session: Session = session_and_user["session"]
    problem_db = _add(session_and_user)
    token = login(client)

    resp = post_json(
        client,
        "/problem/assign_solver",
        {
            "problem_id": problem_db.id,
            "solver_string_representation": "cvxpy",
            "solver_options": {"verbose": "not a bool"},
        },
        token,
    )
    assert resp.status_code == 422

    resp = post_json(
        client,
        "/problem/assign_solver",
        {"problem_id": problem_db.id, "solver_string_representation": "cvxpy"},
        token,
    )
    assert resp.status_code == 200, resp.text

    session.refresh(problem_db)
    session.refresh(problem_db.problem_metadata)
    solver_rows = [
        m for m in problem_db.problem_metadata.all_metadata if m.metadata_type == "solver_selection_metadata"
    ]
    assert solver_rows[-1].solver_options is None


def test_stored_options_last_row(client: TestClient, session_and_user: dict):
    """_stored_options returns the last row's options; None when no solver is set."""
    session: Session = session_and_user["session"]
    problem_db = _add(session_and_user)
    token = login(client)

    resp1 = post_json(
        client,
        "/problem/assign_solver",
        {"problem_id": problem_db.id, "solver_string_representation": "gurobipy", "solver_options": {"A": 1}},
        token,
    )
    assert resp1.status_code == 200, resp1.text
    resp2 = post_json(
        client,
        "/problem/assign_solver",
        {"problem_id": problem_db.id, "solver_string_representation": "gurobipy", "solver_options": {"B": 2}},
        token,
    )
    assert resp2.status_code == 200, resp2.text

    session.refresh(problem_db)
    session.refresh(problem_db.problem_metadata)
    assert _stored_options(problem_db) == {"B": 2}

    no_solver_db = _add(session_and_user)
    assert _solver_selection(no_solver_db) is None


def test_effective_solver_options_merging(session_and_user: dict):
    """effective_solver_options merges stored and request options; validates for model-based solvers."""
    problem_db = _add(
        session_and_user,
        solver="gurobipy",
        solver_options={"FeasibilityTol": 1e-9, "OptimalityTol": 1e-9},
    )
    result = effective_solver_options(problem_db, {"OptimalityTol": 1e-6, "Method": 2})
    assert isinstance(result, GurobipyOptions)
    assert result.model_dump(exclude_unset=True) == {"FeasibilityTol": 1e-9, "OptimalityTol": 1e-6, "Method": 2}

    result = effective_solver_options(problem_db, None)
    assert result.model_dump(exclude_unset=True) == {"FeasibilityTol": 1e-9, "OptimalityTol": 1e-9}

    with pytest.raises(HTTPException) as exc_info:
        effective_solver_options(problem_db, {"FeasibilityTol": "abc"})
    assert exc_info.value.status_code == 422

    plain_db = _add(session_and_user)
    assert effective_solver_options(plain_db, None) is None

    cvxpy_db = _add(session_and_user, solver="cvxpy", solver_options={"verbose": False})
    result = effective_solver_options(cvxpy_db, None)
    assert isinstance(result, CVXPYSolverOptions)
    assert result.verbose is False

    with pytest.raises(HTTPException) as exc_info:
        effective_solver_options(cvxpy_db, {"verbose": "not a bool"})
    assert exc_info.value.status_code == 422


@pytest.mark.gurobipy
def test_effective_solver_options_case_variant_request_wins(session_and_user):
    """Request options with case-variant name override stored options."""
    problem_db = _add(session_and_user, solver="gurobipy", solver_options={"feasibilitytol": 1e-6})
    options = effective_solver_options(problem_db, {"FeasibilityTol": 1e-9})
    assert options.model_dump(exclude_unset=True) == {"FeasibilityTol": 1e-9}
    solver = GurobipySolver(Problem.from_problemdb(problem_db), options)
    assert solver.evaluator.model.Params.FeasibilityTol == 1e-9


def _cumulus_solve(client: TestClient, problem_db: ProblemDB, solver_options: dict, expected_status: int = 200):
    """Initialize CUMULUS on `problem_db` and solve once with `solver_options` in the request."""
    token = login(client)
    resp = post_json(client, "/method/cumulus/get-or-initialize", {"problem_id": problem_db.id}, token)
    assert resp.status_code == 200, resp.text
    init = resp.json()

    current = init["current_solutions"][0]["objective_values"]
    ideal = {o.symbol: o.ideal for o in problem_db.objectives}
    reference_point = {symbol: (current[symbol] + ideal[symbol]) / 2 for symbol in current}
    resp = post_json(
        client,
        "/method/cumulus/solve",
        {
            "problem_id": problem_db.id,
            "preference": {"preference_type": "reference_point", "aspiration_levels": reference_point},
            "current_objectives": current,
            "parent_state_id": init["state_id"],
            "solver_options": solver_options,
        },
        token,
    )
    assert resp.status_code == expected_status, resp.text
    return resp


@pytest.mark.cvxpy
def test_cumulus_with_solver_options(client: TestClient, session_and_user: dict, monkeypatch: pytest.MonkeyPatch):
    """CUMULUS solve passes merged solver_options to the solver constructor."""
    monkeypatch.setattr(utils, "_problem_cache", OrderedDict())
    problem_db = _add(session_and_user, solver="cvxpy", solver_options={"verbose": False})

    captured: list = []

    class CapturingSolver(CVXPYSolver):
        def __init__(self, problem, options=None):
            captured.append(options)
            if options is not None:
                super().__init__(problem, options)
            else:
                super().__init__(problem)

    monkeypatch.setitem(available_solvers["cvxpy"], "constructor", CapturingSolver)
    _cumulus_solve(client, problem_db, {"gp": False})

    assert captured
    assert all(isinstance(opt, CVXPYSolverOptions) for opt in captured)
    assert captured[-1].model_fields_set >= {"verbose", "gp"}
    assert any(opt.model_fields_set == {"verbose"} for opt in captured)


@pytest.mark.gurobipy
def test_cumulus_passes_gurobi_parameters(client: TestClient, session_and_user: dict, monkeypatch: pytest.MonkeyPatch):
    """A stored Gurobi parameter reaches `GurobipySolver` as `GurobipyOptions` and is set on the model."""
    monkeypatch.setattr(utils, "_problem_cache", OrderedDict())
    problem_db = _add(session_and_user, solver="gurobipy", solver_options={"FeasibilityTol": 1e-9})

    captured: list = []

    class CapturingSolver(GurobipySolver):
        def __init__(self, problem, options=None):
            super().__init__(problem, options)
            captured.append((options, self.evaluator.model.Params.FeasibilityTol))

    monkeypatch.setitem(available_solvers["gurobipy"], "constructor", CapturingSolver)
    _cumulus_solve(client, problem_db, {"OptimalityTol": 1e-8})

    assert captured
    assert all(isinstance(options, GurobipyOptions) and tol == 1e-9 for options, tol in captured)
    assert captured[-1][0].model_dump(exclude_unset=True) == {"FeasibilityTol": 1e-9, "OptimalityTol": 1e-8}


def test_copy_problem_metadata_copies_solver_options(session_and_user: dict):
    """copy_problem_metadata preserves solver_options on the SolverSelectionMetadata row."""
    session: Session = session_and_user["session"]
    problem_db = _add(session_and_user, solver="gurobipy", solver_options={"FeasibilityTol": 1e-9})
    copy_db = ProblemDB.from_problem(Problem.from_problemdb(problem_db), user=session_and_user["user"])
    session.add(copy_db)
    session.commit()
    session.refresh(copy_db)

    copy_problem_metadata(problem_db, copy_db, session)
    session.commit()

    session.refresh(copy_db)
    session.refresh(copy_db.problem_metadata)
    assert _stored_options(copy_db) == {"FeasibilityTol": 1e-9}


def test_add_stationary_lp_problem_stores_solver_options(session_and_user: dict):
    """add_stationary_lp_problem stores solver_options; raises ValueError for missing solver or invalid options."""
    problem_db = _add(session_and_user, solver="gurobipy", solver_options={"FeasibilityTol": 1e-9})
    rows = [m for m in problem_db.problem_metadata.all_metadata if m.metadata_type == "solver_selection_metadata"]
    assert len(rows) == 1
    assert rows[0].solver_options == {"FeasibilityTol": 1e-9}

    with pytest.raises(ValueError, match="need a"):
        _add(session_and_user, solver_options={"FeasibilityTol": 1e-9})

    with pytest.raises(ValueError, match="verbose"):
        _add(session_and_user, solver="cvxpy", solver_options={"verbose": "not a bool"})


def test_assign_solver_keeps_unknown_options(client: TestClient, session_and_user: dict, caplog):
    """Unknown cvxpy keys are kept alongside validated fields and a warning is logged."""
    session: Session = session_and_user["session"]
    problem_db = _add(session_and_user)
    token = login(client)

    caplog.set_level(logging.WARNING, logger="desdeo.tools.utils")
    resp = post_json(
        client,
        "/problem/assign_solver",
        {
            "problem_id": problem_db.id,
            "solver_string_representation": "cvxpy",
            "solver_options": {"verbos": True, "verbose": False},
        },
        token,
    )
    assert resp.status_code == 200, resp.text

    session.refresh(problem_db)
    session.refresh(problem_db.problem_metadata)
    assert _stored_options(problem_db) == {"verbos": True, "verbose": False}

    warning_records = [r for r in caplog.records if r.levelname == "WARNING" and r.name == "desdeo.tools.utils"]
    assert any("verbos" in r.getMessage() and "cvxpy" in r.getMessage() for r in warning_records)

    result = effective_solver_options(problem_db)
    assert isinstance(result, CVXPYSolverOptions)
    assert result.verbose is False


def test_alias_and_field_name_merge(session_and_user: dict):
    """Stored alias and request field-name merge correctly for pyomo_cbc."""
    session: Session = session_and_user["session"]

    problem_db = _add(session_and_user)
    session.add(
        SolverSelectionMetadata(
            metadata_id=problem_db.problem_metadata.id,
            solver_string_representation="pyomo_cbc",
            solver_options={"logLevel": 1},
        )
    )
    session.commit()
    session.refresh(problem_db)
    session.refresh(problem_db.problem_metadata)

    result = effective_solver_options(problem_db, {"log_level": 2})
    assert result.log_level == 2

    problem_db2 = _add(session_and_user, solver="pyomo_cbc", solver_options={"logLevel": 1})
    rows = [m for m in problem_db2.problem_metadata.all_metadata if m.metadata_type == "solver_selection_metadata"]
    assert rows[0].solver_options == {"log_level": 1}

    result2 = effective_solver_options(problem_db2, {"logLevel": 3})
    assert result2.log_level == 3


def test_constrained_variant_copies_solver_options(client: TestClient, session_and_user: dict):
    """A constrained variant copies the parent's solver_options from the last SolverSelectionMetadata row."""
    session: Session = session_and_user["session"]
    token = login(client)
    river_db = session.exec(select(ProblemDB).where(ProblemDB.name == "The river pollution problem")).first()

    resp = post_json(
        client,
        "/problem/assign_solver",
        {
            "problem_id": river_db.id,
            "solver_string_representation": "gurobipy",
            "solver_options": {"FeasibilityTol": 1e-9},
        },
        token,
    )
    assert resp.status_code == 200, resp.text

    resp = post_json(
        client,
        f"/problem/{river_db.id}/constrained_variant",
        {"variable_fixings": [{"variable_symbol": "x_1", "fixed_value": 0.5}]},
        token,
    )
    assert resp.status_code == 200, resp.text
    variant_id = resp.json()["problem_id"]

    assert _stored_options(session.get(ProblemDB, variant_id)) == {"FeasibilityTol": 1e-9}


def test_invalid_stored_options_leave_no_modification_rows(client: TestClient, session_and_user: dict):
    """Invalid stored cvxpy options cause modify-problem to return 422 without creating rows."""
    session: Session = session_and_user["session"]
    token = login(client)

    meta = ProblemMetaDataDB(problem_id=1)
    session.add(meta)
    session.commit()
    session.refresh(meta)
    session.add(
        SolverSelectionMetadata(
            metadata_id=meta.id,
            solver_string_representation="cvxpy",
            solver_options={"verbose": "not a bool"},
        )
    )
    session.commit()

    problem_count_before = len(session.exec(select(ProblemDB)).all())
    state_count_before = len(session.exec(select(StateDB)).all())

    resp = post_json(
        client,
        "/method/cumulus/modify-problem",
        {
            "problem_id": 1,
            "original_problem_id": 1,
            "modifications": {
                "constraints": [{"name": "x_1 upper", "symbol": "g_new", "func": "x_1 - 0.8", "cons_type": "<="}]
            },
        },
        token,
    )
    assert resp.status_code == 422
    assert "Invalid options" in resp.text

    assert len(session.exec(select(ProblemDB)).all()) == problem_count_before
    assert len(session.exec(select(StateDB)).all()) == state_count_before


def test_invalid_stored_options_leave_no_hypothetical_rows(client: TestClient, session_and_user: dict):
    """Invalid stored cvxpy options cause hypothetical to return 422 without creating rows."""
    session: Session = session_and_user["session"]
    token = login(client)
    pid = _compile_oil_wildcatter(client, token)

    problem_db = session.get(ProblemDB, pid)
    session.add(
        SolverSelectionMetadata(
            metadata_id=problem_db.problem_metadata.id,
            solver_string_representation="cvxpy",
            solver_options={"verbose": "not a bool"},
        )
    )
    session.commit()

    problem_count_before = len(session.exec(select(ProblemDB)).all())
    state_count_before = len(session.exec(select(StateDB)).all())

    resp = _hypothetical(client, token, pid, chance={"structure": "soaking"})
    assert resp.status_code == 422
    assert "Invalid options" in resp.text

    assert len(session.exec(select(ProblemDB)).all()) == problem_count_before
    assert len(session.exec(select(StateDB)).all()) == state_count_before


def test_options_reach_modification_background_checks(
    client: TestClient, session_and_user: dict, monkeypatch: pytest.MonkeyPatch
):
    """Normalized solver_options are passed as keyword to _run_modification_checks_background."""
    token = login(client)
    captured: list = []

    def fake(*args, solver_options=None, **kwargs):
        captured.append(solver_options)

    monkeypatch.setattr(cumulus_router, "_run_modification_checks_background", fake)

    # dtlz2 modify-problem
    resp = post_json(
        client,
        "/problem/assign_solver",
        {"problem_id": 1, "solver_string_representation": "cvxpy", "solver_options": {"verbose": False}},
        token,
    )
    assert resp.status_code == 200, resp.text

    resp = post_json(
        client,
        "/method/cumulus/modify-problem",
        {
            "problem_id": 1,
            "original_problem_id": 1,
            "modifications": {
                "constraints": [{"name": "x_1 upper", "symbol": "g_new", "func": "x_1 - 0.8", "cons_type": "<="}]
            },
        },
        token,
    )
    assert resp.status_code == 200, resp.text
    assert captured[-1] is not None
    assert isinstance(captured[-1], CVXPYSolverOptions)
    assert captured[-1].verbose is False
    assert captured[-1].model_fields_set == {"verbose"}

    # oil wildcatter hypothetical
    pid = _compile_oil_wildcatter(client, token)
    resp = post_json(
        client,
        "/problem/assign_solver",
        {"problem_id": pid, "solver_string_representation": "cvxpy", "solver_options": {"gp": False}},
        token,
    )
    assert resp.status_code == 200, resp.text

    resp = _hypothetical(client, token, pid, chance={"structure": "soaking"})
    assert resp.status_code == 200, resp.text
    assert captured[-1].gp is False


@pytest.mark.cvxpy
def test_cumulus_solve_rejects_invalid_request_options(
    client: TestClient, session_and_user: dict, monkeypatch: pytest.MonkeyPatch
):
    """A /method/cumulus/solve request with invalid solver_options returns 422."""
    monkeypatch.setattr(utils, "_problem_cache", OrderedDict())
    problem_db = _add(session_and_user, solver="cvxpy", solver_options={"verbose": False})

    resp = _cumulus_solve(client, problem_db, {"verbose": "not a bool"}, expected_status=422)
    assert "verbose" in resp.text


def test_normalize_keeps_python_values_and_warns_on_name_and_alias(caplog: pytest.LogCaptureFixture) -> None:
    """Normalising keeps non-JSON values such as infinity, and warns when a field is given by name and alias."""
    assert normalize_solver_options("scipy_de", {"de_kwargs": {"atol": float("inf")}}) == {
        "de_kwargs": {"atol": float("inf")}
    }
    with caplog.at_level(logging.WARNING):
        assert normalize_solver_options("pyomo_cbc", {"log_level": 1, "logLevel": 3}) == {"log_level": 3}
    assert "both name and alias" in caplog.text
