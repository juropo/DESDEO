"""Tests for CUMULUS graphical-model aggregate measures and scenario availability."""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlmodel import Session

from desdeo.api.models import ProblemDB, StateDB
from desdeo.api.models.cumulus import UncertaintyMeasureSpec
from desdeo.api.routers.cumulus import _retarget_description_parts
from desdeo.api.routers.utils import _apply_graphical_uncertainty_measures, find_graphical_metadata
from desdeo.problem import Problem
from desdeo.problem.graphical import RiskMeasure
from desdeo.problem.graphical.compile import IndexMap
from desdeo.problem.graphical.demo import build_oil_wildcatter_demo
from desdeo.tools.outcome_distribution import cvar_dd_internal_symbols

from .conftest import get_json, login, post_json
from .test_cumulus_hypothetical import _compile_oil_wildcatter, _hypothetical, _run_checks
from .test_routes import _setup_simple_scenario

pytestmark = pytest.mark.graphical


def _measures(cvar_alpha: float = 0.2) -> list[dict]:
    return [
        {"measure_type": "expected_value", "symbols": ["investment", "oil_produced", "profit"]},
        {"measure_type": "worst_case_robust", "symbols": ["investment", "oil_produced", "profit"]},
        {"measure_type": "conditional_value_at_risk", "symbols": ["profit"], "alpha": cvar_alpha},
    ]


def _objective_symbols(session: Session, problem_id: int) -> list[str]:
    session.expire_all()
    problem = session.get(ProblemDB, problem_id)
    return [o.symbol for o in problem.objectives]


def _ideals(session: Session, problem_id: int) -> dict[str, float]:
    session.expire_all()
    problem = session.get(ProblemDB, problem_id)
    return {o.symbol: o.ideal for o in problem.objectives}


def _aggregate(client: TestClient, token: str, problem_id: int, measures: list[dict]) -> dict:
    resp = post_json(
        client,
        "/method/cumulus/get-or-initialize",
        {"problem_id": problem_id, "uncertainty_measures": measures},
        token,
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    return resp.json()


def test_availability_graphical(client: TestClient, session_and_user: dict) -> None:
    """Availability endpoint reports graphical source and correct fields for a compiled problem."""
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    resp = get_json(client, f"/method/cumulus/scenario-availability?problem_id={problem_id}", token)
    assert resp.status_code == 200
    data = resp.json()
    assert data["available"] is True
    assert data["source"] == "graphical"
    assert data["scenario_model_id"] is None
    assert data["base_problem_id"] == problem_id
    assert data["objective_symbols"] == ["investment", "oil_produced", "profit"]
    assert data["applied_measures"] == []
    assert "weighted_scenarios" not in data["measure_types"]
    for t in ("expected_value", "worst_case_robust", "conditional_value_at_risk"):
        assert t in data["measure_types"]


def test_availability_scenario_model(client: TestClient, session_and_user: dict) -> None:
    """Availability endpoint reports scenario_model source and includes weighted_scenarios."""
    token = login(client)
    problem_id, sm_id = _setup_simple_scenario(session_and_user)
    resp = get_json(client, f"/method/cumulus/scenario-availability?problem_id={problem_id}", token)
    assert resp.status_code == 200
    data = resp.json()
    assert data["source"] == "scenario_model"
    assert data["scenario_model_id"] == sm_id
    assert "weighted_scenarios" in data["measure_types"]


@pytest.mark.cvxpy
def test_get_or_initialize_setup_then_init(client: TestClient, session_and_user: dict) -> None:
    """First call returns scenario_setup; second with empty measures initializes on the same problem."""
    session = session_and_user["session"]
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    resp = post_json(client, "/method/cumulus/get-or-initialize", {"problem_id": problem_id}, token)
    assert resp.status_code == 200
    data = resp.json()
    assert data["response_type"] == "cumulus.scenario_setup"
    assert data["source"] == "graphical"
    assert data["scenario_model_id"] is None
    assert data["objective_symbols"] == ["investment", "oil_produced", "profit"]
    assert "weighted_scenarios" not in data["measure_types"]
    resp2 = post_json(
        client,
        "/method/cumulus/get-or-initialize",
        {"problem_id": problem_id, "uncertainty_measures": []},
        token,
    )
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["response_type"] == "cumulus.initialization"
    assert data2["problem_id"] == problem_id
    assert _objective_symbols(session, problem_id) == ["investment", "oil_produced", "profit"]


@pytest.mark.cvxpy
def test_aggregate_measures_initialization(client: TestClient, session_and_user: dict) -> None:
    """Aggregated problem has correct objectives, extra funcs, availability, and unconditioned ideals."""
    session = session_and_user["session"]
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    data = _aggregate(client, token, problem_id, _measures())
    assert data["response_type"] == "cumulus.initialization"
    agg_id = data["problem_id"]
    assert agg_id != problem_id
    expected_symbols = [
        "E_investment",
        "E_oil_produced",
        "E_profit",
        "robust_investment",
        "robust_oil_produced",
        "robust_profit",
        "CVAR_profit",
    ]
    assert _objective_symbols(session, agg_id) == expected_symbols
    session.expire_all()
    original = session.get(ProblemDB, problem_id)
    aggregated = session.get(ProblemDB, agg_id)
    orig_funcs = {e.symbol for e in original.extra_funcs}
    agg_funcs = {e.symbol for e in aggregated.extra_funcs}
    assert orig_funcs.issubset(agg_funcs)
    resp = get_json(client, f"/method/cumulus/scenario-availability?problem_id={agg_id}", token)
    assert resp.status_code == 200
    avail = resp.json()
    assert avail["base_problem_id"] == problem_id
    assert avail["applied_measures"] == ["conditional_value_at_risk", "expected_value", "worst_case_robust"]
    ideals = _ideals(session, agg_id)
    assert ideals["E_oil_produced"] == pytest.approx(900.0, rel=1e-6)
    assert ideals["robust_oil_produced"] == pytest.approx(0.0, abs=1e-6)
    assert ideals["robust_profit"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.cvxpy
def test_apply_scenario_measures_new_problem(client: TestClient, session_and_user: dict) -> None:
    """apply-scenario-measures on an aggregated problem builds a new problem from the original base."""
    session = session_and_user["session"]
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    data = _aggregate(client, token, problem_id, _measures())
    agg_id = data["problem_id"]
    resp = post_json(
        client,
        "/method/cumulus/apply-scenario-measures",
        {"problem_id": agg_id, "uncertainty_measures": [{"measure_type": "expected_value", "symbols": ["profit"]}]},
        token,
    )
    assert resp.status_code == 200
    new_id = resp.json()["problem_id"]
    session.expire_all()
    new_problem = session.get(ProblemDB, new_id)
    assert new_problem.parent_problem_id == problem_id
    assert _objective_symbols(session, new_id) == ["E_profit"]


@pytest.mark.cvxpy
def test_weighted_scenarios_rejected(client: TestClient, session_and_user: dict) -> None:
    """weighted_scenarios measure type returns 422 for graphical problems."""
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    resp = post_json(
        client,
        "/method/cumulus/get-or-initialize",
        {
            "problem_id": problem_id,
            "uncertainty_measures": [
                {"measure_type": "weighted_scenarios", "symbols": ["profit"], "leaf_weights": {"a": 1.0}},
            ],
        },
        token,
    )
    assert resp.status_code == 422


@pytest.mark.cvxpy
def test_unknown_symbol_rejected(client: TestClient, session_and_user: dict) -> None:
    """Unknown symbol in measures returns 422 with the symbol name in the detail."""
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    resp = post_json(
        client,
        "/method/cumulus/get-or-initialize",
        {
            "problem_id": problem_id,
            "uncertainty_measures": [
                {"measure_type": "expected_value", "symbols": ["nonexistent"]},
            ],
        },
        token,
    )
    assert resp.status_code == 422
    assert "nonexistent" in resp.text


@pytest.mark.cvxpy
def test_aggregated_graphical_endpoints(client: TestClient, session_and_user: dict) -> None:
    """Aggregated problem serves /graphical/structure and /graphical/policy with the initial solution."""
    session = session_and_user["session"]
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    data = _aggregate(client, token, problem_id, _measures())
    agg_id = data["problem_id"]
    state_id = data["state_id"]
    resp = get_json(client, f"/graphical/{agg_id}/structure", token)
    assert resp.status_code == 200
    session.expire_all()
    state = session.get(StateDB, state_id)
    var_values = state.state.solver_results.optimal_variables
    resp2 = post_json(client, f"/graphical/{agg_id}/policy", {"variable_values": var_values}, token)
    assert resp2.status_code == 200
    assert "policy" in resp2.json()


@pytest.mark.cvxpy
def test_hypothetical_from_aggregated(
    client: TestClient, session_and_user: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hypothetical from an aggregated problem returns scenario_setup with the original symbols."""
    session = session_and_user["session"]
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    data = _aggregate(client, token, problem_id, _measures())
    agg_id = data["problem_id"]
    hyp_resp = _hypothetical(client, token, agg_id, chance={"structure": "soaking"})
    assert hyp_resp.status_code == 200, hyp_resp.text
    hyp_id = hyp_resp.json()["problem_id"]
    hyp_state_id = hyp_resp.json()["state_id"]
    _run_checks(session, monkeypatch, hyp_id, hyp_state_id)
    resp = post_json(client, "/method/cumulus/get-or-initialize", {"problem_id": hyp_id}, token)
    assert resp.status_code == 200
    d = resp.json()
    assert d["response_type"] == "cumulus.scenario_setup"
    assert d["source"] == "graphical"
    assert d["objective_symbols"] == ["investment", "oil_produced", "profit"]


@pytest.mark.cvxpy
def test_hypothetical_conditioned_ideals(
    client: TestClient, session_and_user: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Conditioned hypothetical aggregates differ from unconditioned ideals and parent is the hypothetical."""
    session = session_and_user["session"]
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    data_uncond = _aggregate(client, token, problem_id, _measures())
    uncond_id = data_uncond["problem_id"]
    hyp_resp = _hypothetical(client, token, problem_id, chance={"structure": "soaking"})
    assert hyp_resp.status_code == 200, hyp_resp.text
    hyp_id = hyp_resp.json()["problem_id"]
    hyp_state_id = hyp_resp.json()["state_id"]
    _run_checks(session, monkeypatch, hyp_id, hyp_state_id)
    data_cond = _aggregate(client, token, hyp_id, _measures())
    cond_id = data_cond["problem_id"]
    session.expire_all()
    cond_problem = session.get(ProblemDB, cond_id)
    assert cond_problem.parent_problem_id == hyp_id
    uncond_ideals = _ideals(session, uncond_id)
    cond_ideals = _ideals(session, cond_id)
    assert cond_ideals["robust_oil_produced"] == pytest.approx(2700.0, rel=1e-6)
    assert uncond_ideals["robust_oil_produced"] == pytest.approx(0.0, abs=1e-6)
    assert cond_ideals["E_oil_produced"] == pytest.approx(2700.0, rel=1e-6)
    assert uncond_ideals["E_oil_produced"] == pytest.approx(900.0, rel=1e-6)
    assert cond_ideals["robust_profit"] == pytest.approx(119000.0, rel=1e-6)
    assert uncond_ideals["robust_profit"] == pytest.approx(0.0, abs=1e-6)
    assert cond_ideals["CVAR_profit"] == pytest.approx(119000.0, rel=1e-6)


def test_empty_symbols_rejected(client: TestClient, session_and_user: dict) -> None:
    """Empty symbols list in uncertainty measure spec returns 422 from request validation."""
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    resp = post_json(
        client,
        "/method/cumulus/get-or-initialize",
        {"problem_id": problem_id, "uncertainty_measures": [{"measure_type": "expected_value", "symbols": []}]},
        token,
    )
    assert resp.status_code == 422


def test_no_aggregates_rejected(client: TestClient, session_and_user: dict) -> None:
    """_apply_graphical_uncertainty_measures raises 422 when no aggregates are produced from empty symbols."""
    session = session_and_user["session"]
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    spec = UncertaintyMeasureSpec.model_construct(measure_type="expected_value", symbols=[])
    problem = Problem.from_problemdb(session.get(ProblemDB, problem_id))
    meta = find_graphical_metadata(problem_id, session)
    with pytest.raises(HTTPException) as exc:
        _apply_graphical_uncertainty_measures(problem, [spec], meta)
    assert exc.value.status_code == 422


@pytest.mark.cvxpy
def test_risk_measure_aux_removed(client: TestClient, session_and_user: dict) -> None:
    """CVaR risk measure aux variables are removed from aggregated problem but present in base."""
    session = session_and_user["session"]
    token = login(client)
    network, annotation = build_oil_wildcatter_demo()
    annotation = annotation.model_copy(
        update={"risk_measures": [RiskMeasure(symbol="cvar_profit", target="profit", alpha=0.5)]}
    )
    resp = post_json(
        client,
        "/graphical/compile",
        {"network": network.model_dump(mode="json"), "annotation": annotation.model_dump(mode="json")},
        token,
    )
    assert resp.status_code == 200, resp.text
    pid = resp.json()["problem_id"]
    setup_resp = post_json(client, "/method/cumulus/get-or-initialize", {"problem_id": pid}, token)
    assert setup_resp.status_code == 200
    setup_data = setup_resp.json()
    assert setup_data["objective_symbols"] == ["investment", "oil_produced", "profit"]
    data = _aggregate(client, token, pid, [{"measure_type": "expected_value", "symbols": ["profit"]}])
    assert data["response_type"] == "cumulus.initialization"
    agg_id = data["problem_id"]
    assert _objective_symbols(session, agg_id) == ["E_profit"]
    aux_symbols = set(cvar_dd_internal_symbols("cvar_profit").values())
    agg_problem = Problem.from_problemdb(session.get(ProblemDB, agg_id))
    agg_var_symbols = {v.symbol for v in agg_problem.variables}
    assert aux_symbols.isdisjoint(agg_var_symbols)
    agg_constraint_symbols = {c.symbol for c in (agg_problem.constraints or [])}
    assert not any("cvar_profit" in s for s in agg_constraint_symbols)
    base_problem = Problem.from_problemdb(session.get(ProblemDB, pid))
    base_var_symbols = {v.symbol for v in base_problem.variables}
    assert aux_symbols.issubset(base_var_symbols)


def test_modify_problem_graphical_base(client: TestClient, session_and_user: dict) -> None:
    """Modify-problem on a base graphical problem with uncertainty measures succeeds and updates objectives."""
    session = session_and_user["session"]
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    resp = post_json(
        client,
        "/method/cumulus/modify-problem",
        {
            "problem_id": problem_id,
            "original_problem_id": problem_id,
            "modifications": {"uncertainty_measures": [{"measure_type": "expected_value", "symbols": ["profit"]}]},
        },
        token,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["response_type"] == "cumulus.modify"
    new_id = data["problem_id"]
    assert new_id != problem_id
    assert _objective_symbols(session, new_id) == ["E_profit"]


@pytest.mark.cvxpy
def test_modify_problem_aggregated_rejected(client: TestClient, session_and_user: dict) -> None:
    """Modify-problem on an aggregated problem with measures on a non-objective symbol returns 422."""
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    data = _aggregate(client, token, problem_id, [{"measure_type": "expected_value", "symbols": ["profit"]}])
    agg_id = data["problem_id"]
    resp = post_json(
        client,
        "/method/cumulus/modify-problem",
        {
            "problem_id": agg_id,
            "original_problem_id": agg_id,
            "modifications": {"uncertainty_measures": [{"measure_type": "expected_value", "symbols": ["profit"]}]},
        },
        token,
    )
    assert resp.status_code == 422


def test_modify_problem_dropped_objective_reference_rejected(client: TestClient, session_and_user: dict) -> None:
    """Modify-problem with constraint on dropped objective and measure on remaining objective returns 422."""
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    resp = post_json(
        client,
        "/method/cumulus/modify-problem",
        {
            "problem_id": problem_id,
            "original_problem_id": problem_id,
            "modifications": {
                "constraints": [
                    {
                        "name": "profit nonneg",
                        "symbol": "profit_nonneg",
                        "cons_type": "<=",
                        "func": ["Multiply", -1, "profit"],
                    }
                ],
                "uncertainty_measures": [{"measure_type": "expected_value", "symbols": ["investment"]}],
            },
        },
        token,
    )
    assert resp.status_code == 422
    assert "profit_nonneg" in resp.text


def test_modify_problem_risk_aux_reference_rejected(client: TestClient, session_and_user: dict) -> None:
    """A user constraint on a dropped risk measure's auxiliary variable returns 422 instead of being removed."""
    token = login(client)
    network, annotation = build_oil_wildcatter_demo()
    annotation = annotation.model_copy(
        update={"risk_measures": [RiskMeasure(symbol="cvar_profit", target="profit", alpha=0.5)]}
    )
    resp = post_json(
        client,
        "/graphical/compile",
        {"network": network.model_dump(mode="json"), "annotation": annotation.model_dump(mode="json")},
        token,
    )
    assert resp.status_code == 200, resp.text
    problem_id = resp.json()["problem_id"]
    eta = cvar_dd_internal_symbols("cvar_profit")["eta"]
    resp = post_json(
        client,
        "/method/cumulus/modify-problem",
        {
            "problem_id": problem_id,
            "original_problem_id": problem_id,
            "modifications": {
                "constraints": [
                    {"name": "eta bound", "symbol": "eta_bound", "cons_type": "<=", "func": ["Subtract", eta, 1000]}
                ],
                "uncertainty_measures": [{"measure_type": "expected_value", "symbols": ["profit"]}],
            },
        },
        token,
    )
    assert resp.status_code == 422
    assert "eta_bound" in resp.text


@pytest.mark.cvxpy
def test_apply_scenario_measures_conflicting_distributions(client: TestClient, session_and_user: dict) -> None:
    """Conflicting stored pu_* functions give a 422 on apply-scenario-measures but not on the other endpoints."""
    session = session_and_user["session"]
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)

    meta = find_graphical_metadata(problem_id, session)
    index_map = IndexMap.model_validate_json(meta.index_map_json)
    meta.index_map_json = index_map.model_copy(update={"outcome_distributions": {}}).model_dump_json()
    session.add(meta)
    pu_func = next(f for f in session.get(ProblemDB, problem_id).extra_funcs if f.symbol.startswith("pu_"))
    pu_func.func = ["Add", pu_func.func, 1]
    session.add(pu_func)
    session.commit()
    pu_symbol = pu_func.symbol

    resp = post_json(
        client,
        "/method/cumulus/apply-scenario-measures",
        {"problem_id": problem_id, "uncertainty_measures": [{"measure_type": "expected_value", "symbols": ["profit"]}]},
        token,
    )
    assert resp.status_code == 422
    assert pu_symbol in resp.json()["detail"]

    resp = get_json(client, f"/method/cumulus/scenario-availability?problem_id={problem_id}", token)
    assert resp.status_code == 200
    assert resp.json()["available"] is False

    resp = post_json(client, "/method/cumulus/get-or-initialize", {"problem_id": problem_id}, token)
    assert resp.status_code == 200
    assert resp.json()["response_type"] == "cumulus.initialization"


@pytest.mark.cvxpy
def test_reopen_aggregated_in_fresh_session(client: TestClient, session_and_user: dict) -> None:
    """Opening an aggregated problem in a fresh session returns initialization with same problem_id."""
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    data = _aggregate(client, token, problem_id, [{"measure_type": "expected_value", "symbols": ["profit"]}])
    agg_id = data["problem_id"]
    assert post_json(client, "/session/new", {"info": "fresh"}, token).status_code == 200
    resp = post_json(client, "/method/cumulus/get-or-initialize", {"problem_id": agg_id}, token)
    assert resp.status_code == 200
    d = resp.json()
    assert d["response_type"] == "cumulus.initialization"
    assert d["problem_id"] == agg_id


def _solve(client: TestClient, token: str, init: dict, reference_point: dict[str, float]) -> dict:
    resp = post_json(
        client,
        "/method/cumulus/solve",
        {
            "problem_id": init["problem_id"],
            "preference": {"preference_type": "reference_point", "aspiration_levels": reference_point},
            "current_objectives": init["current_solutions"][0]["objective_values"],
            "scalarizations": ["cumulonimbus"],
            "parent_state_id": init["state_id"],
        },
        token,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


@pytest.mark.gurobipy
def test_solve_with_cvar_aggregates(client: TestClient, session_and_user: dict) -> None:
    """Cumulonimbus finds a solution on an E + worst-case + CVaR build of the oil wildcatter."""
    session = session_and_user["session"]
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    symbols = ["investment", "oil_produced", "profit"]
    measures = [
        {"measure_type": "expected_value", "symbols": symbols},
        {"measure_type": "worst_case_robust", "symbols": symbols},
        {"measure_type": "conditional_value_at_risk", "symbols": symbols, "alpha": 0.05},
    ]
    init = _aggregate(client, token, problem_id, measures)
    reference_point = dict.fromkeys(_objective_symbols(session, init["problem_id"]), 0.0)
    reference_point |= {"E_oil_produced": 900.0, "E_profit": 20286.0}
    data = _solve(client, token, init, reference_point)
    assert len(data["current_solutions"]) >= 1
    assert data["warnings"] == []


@pytest.mark.gurobipy
def test_solve_failure_warns(client: TestClient, session_and_user: dict) -> None:
    """A cumulonimbus solve that finds no solution says so in the warnings."""
    session = session_and_user["session"]
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    init = _aggregate(client, token, problem_id, [{"measure_type": "expected_value", "symbols": ["profit"]}])
    session.expire_all()
    nadir = {o.symbol: o.nadir for o in session.get(ProblemDB, init["problem_id"]).objectives}
    data = _solve(client, token, init, nadir)
    assert data["current_solutions"] == []
    assert any("Cumulonimbus scalarization" in w for w in data["warnings"])


@pytest.mark.cvxpy
def test_aggregate_solution_description(client: TestClient, session_and_user: dict) -> None:
    """A replaced objective's description part reads its expected value, or is left out without one."""
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    parts = [
        {"text": "Expected outcomes:"},
        {"label": "Profit", "symbol": "profit", "format_spec": ".0f"},
        {"label": "Investment", "expression": ["Multiply", 2, "investment"], "format_spec": ".0f"},
    ]
    resp = post_json(client, "/solution-description/update_metadata", {"problem_id": problem_id, "parts": parts}, token)
    assert resp.status_code == 200, resp.text
    init = _aggregate(
        client,
        token,
        problem_id,
        [
            {"measure_type": "expected_value", "symbols": ["profit"]},
            {"measure_type": "worst_case_robust", "symbols": ["investment"]},
        ],
    )
    resp = post_json(
        client,
        "/solution-description/get",
        {"problem_id": init["problem_id"], "solution": {"state_id": init["state_id"], "solution_index": 0}},
        token,
    )
    assert resp.status_code == 200, resp.text
    profit = init["current_solutions"][0]["objective_values"]["E_profit"]
    assert resp.json()["description"] == f"Expected outcomes:\nProfit: {profit:.0f}"

    # Without any value part left, the header goes too; a later rebuild still has the parts.
    robust = _aggregate(client, token, problem_id, [{"measure_type": "worst_case_robust", "symbols": ["profit"]}])
    resp = post_json(
        client,
        "/solution-description/get",
        {"problem_id": robust["problem_id"], "solution": {"state_id": robust["state_id"], "solution_index": 0}},
        token,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["description"] == ""
    resp = post_json(
        client,
        "/method/cumulus/apply-scenario-measures",
        {
            "problem_id": robust["problem_id"],
            "uncertainty_measures": [{"measure_type": "expected_value", "symbols": ["profit"]}],
        },
        token,
    )
    assert resp.status_code == 200, resp.text
    reopened = resp.json()
    resp = post_json(
        client,
        "/solution-description/get",
        {"problem_id": reopened["problem_id"], "solution": {"state_id": reopened["state_id"], "solution_index": 0}},
        token,
    )
    assert resp.status_code == 200, resp.text
    profit = reopened["current_solutions"][0]["objective_values"]["E_profit"]
    assert resp.json()["description"] == f"Expected outcomes:\nProfit: {profit:.0f}"


@pytest.mark.parametrize(
    "parts",
    [
        [{"label": "Investment", "symbol": "investment"}],
        [{"label": "Investment", "symbol": "investment"}, {"text": "Outcomes follow."}],
    ],
)
def test_aggregate_solution_description_without_header(
    client: TestClient, session_and_user: dict, parts: list[dict]
) -> None:
    """A description whose first part is a value part left out is rebuilt without error."""
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    resp = post_json(client, "/solution-description/update_metadata", {"problem_id": problem_id, "parts": parts}, token)
    assert resp.status_code == 200, resp.text
    robust = _aggregate(client, token, problem_id, [{"measure_type": "worst_case_robust", "symbols": ["investment"]}])
    resp = post_json(
        client,
        "/solution-description/get",
        {"problem_id": robust["problem_id"], "solution": {"state_id": robust["state_id"], "solution_index": 0}},
        token,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["description"] == ("Outcomes follow." if len(parts) > 1 else "")


def test_retarget_description_parts(client: TestClient, session_and_user: dict) -> None:
    """Value parts on missing symbols are left out, with a text part whose value parts all are."""
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    robust = _aggregate(client, token, problem_id, [{"measure_type": "worst_case_robust", "symbols": ["investment"]}])
    session = session_and_user["session"]
    session.expire_all()
    problem = Problem.from_problemdb(session.get(ProblemDB, robust["problem_id"]))

    def texts(raw_parts: list[dict], renamed: dict[str, str] | None = None) -> list[str | None]:
        return [p["text"] or p["symbol"] for p in _retarget_description_parts(raw_parts, problem, renamed or {})]

    investment = {"symbol": "investment"}
    robust_investment = {"symbol": "robust_investment"}
    assert texts([]) == []
    assert texts([investment]) == []
    assert texts([investment, {"text": "a"}]) == ["a"]
    assert texts([{"text": "a"}, investment]) == []
    assert texts([{"text": "a"}, investment, {"text": "b"}, robust_investment]) == ["b", "robust_investment"]
    assert texts([{"text": "a"}, investment, robust_investment]) == ["a", "robust_investment"]
    assert texts([{"text": "a"}, {"text": "b"}]) == ["a", "b"]
    assert texts([{"text": "a"}, {"symbol": "x"}], {"x": "robust_investment"}) == ["a", "robust_investment"]


@pytest.mark.cvxpy
@pytest.mark.parametrize(
    "measures",
    [
        [{"measure_type": "expected_value", "symbols": ["profit"]}],
        [{"measure_type": "worst_case_robust", "symbols": ["profit"]}],
    ],
)
def test_hypothetical_from_aggregate_solution_description(
    client: TestClient, session_and_user: dict, monkeypatch: pytest.MonkeyPatch, measures: list[dict]
) -> None:
    """A hypothetical made from an aggregate problem describes solutions with the original objectives."""
    session = session_and_user["session"]
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    parts = [
        {"text": "Outcomes:"},
        {"label": "Profit", "symbol": "profit", "format_spec": ".0f"},
        {"label": "Double profit", "expression": ["Multiply", 2, "profit"], "format_spec": ".0f"},
    ]
    resp = post_json(client, "/solution-description/update_metadata", {"problem_id": problem_id, "parts": parts}, token)
    assert resp.status_code == 200, resp.text
    agg_id = _aggregate(client, token, problem_id, measures)["problem_id"]

    hyp_resp = _hypothetical(client, token, agg_id, chance={"structure": "soaking"})
    assert hyp_resp.status_code == 200, hyp_resp.text
    hyp_id = hyp_resp.json()["problem_id"]
    _run_checks(session, monkeypatch, hyp_id, hyp_resp.json()["state_id"])
    resp = post_json(client, "/method/cumulus/initialize", {"problem_id": hyp_id}, token)
    assert resp.status_code == 200, resp.text
    init = resp.json()

    resp = post_json(
        client,
        "/solution-description/get",
        {"problem_id": hyp_id, "solution": {"state_id": init["state_id"], "solution_index": 0}},
        token,
    )
    assert resp.status_code == 200, resp.text
    profit = init["current_solutions"][0]["objective_values"]["profit"]
    assert resp.json()["description"] == f"Outcomes:\nProfit: {profit:.0f}\nDouble profit: {2 * profit:.0f}"


def test_cvar_alpha_out_of_range_rejected(client: TestClient, session_and_user: dict) -> None:
    """A CVaR alpha outside (0, 1) returns 422 from request validation."""
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    resp = post_json(
        client,
        "/method/cumulus/get-or-initialize",
        {"problem_id": problem_id, "uncertainty_measures": _measures(cvar_alpha=1.5)},
        token,
    )
    assert resp.status_code == 422


@pytest.mark.cvxpy
def test_aggregate_stores_rebuilt_outcome_distributions(client: TestClient, session_and_user: dict) -> None:
    """An aggregate built from an index map without outcome distributions stores the rebuilt ones."""
    session = session_and_user["session"]
    token = login(client)
    problem_id = _compile_oil_wildcatter(client, token)
    meta = find_graphical_metadata(problem_id, session)
    index_map = IndexMap.model_validate_json(meta.index_map_json)
    meta.index_map_json = index_map.model_copy(update={"outcome_distributions": {}}).model_dump_json()
    session.add(meta)
    session.commit()

    agg_id = _aggregate(client, token, problem_id, [{"measure_type": "expected_value", "symbols": ["profit"]}])[
        "problem_id"
    ]
    session.expire_all()
    stored = IndexMap.model_validate_json(find_graphical_metadata(agg_id, session).index_map_json)
    assert stored.outcome_distributions == index_map.outcome_distributions
