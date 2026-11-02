"""Tests for `POST /method/cumulus/hypothetical`."""

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from desdeo.api.models import CumulusModificationState, ProblemDB
from desdeo.api.routers import cumulus as cumulus_router
from desdeo.api.routers.graphical import _fetch_graphical_metadata
from desdeo.problem import Problem
from desdeo.problem.graphical import DiagramAssumptions, DiscreteNetwork, compile_with_assumptions
from desdeo.problem.graphical.demo import build_oil_wildcatter_demo
from desdeo.tools import CVXPYSolver

from .conftest import get_json, login, post_json

pytestmark = pytest.mark.graphical


def _compile_oil_wildcatter(client: TestClient, access_token: str) -> int:
    network, annotation = build_oil_wildcatter_demo()
    payload = {"network": network.model_dump(mode="json"), "annotation": annotation.model_dump(mode="json")}
    response = post_json(client, "/graphical/compile", payload, access_token)
    assert response.status_code == 200, response.text
    return response.json()["problem_id"]


def _hypothetical(client: TestClient, access_token: str, problem_id: int, chance=None, decisions=None):
    payload = {"problem_id": problem_id, "assumptions": {"chance": chance or {}, "decisions": decisions or {}}}
    return post_json(client, "/method/cumulus/hypothetical", payload, access_token)


def _run_checks(session: Session, monkeypatch: pytest.MonkeyPatch, problem_id: int, state_id: int) -> None:
    """Run the background checks against the test database."""
    monkeypatch.setattr(cumulus_router, "engine", session.get_bind())
    symbols = [c.symbol for c in session.get(ProblemDB, problem_id).constraints if c.symbol.startswith("assume_")]
    cumulus_router._run_modification_checks_background(problem_id, state_id, None, symbols, feasibility_witnessed=False)
    session.expire_all()


def test_hypothetical_creates_linked_problem_and_leaves_original(client: TestClient, session_and_user: dict):
    """The new problem has its own conditioned metadata; the original's metadata and bounds don't change."""
    session: Session = session_and_user["session"]
    access_token = login(client)
    problem_id = _compile_oil_wildcatter(client, access_token)
    original_meta = _fetch_graphical_metadata(problem_id, session).model_dump()
    original_bounds = [(o.ideal, o.nadir) for o in session.get(ProblemDB, problem_id).objectives]

    response = _hypothetical(client, access_token, problem_id, chance={"structure": "soaking"})

    assert response.status_code == 200, response.text
    new_id = response.json()["problem_id"]
    new_problem = session.get(ProblemDB, new_id)
    assert new_problem.parent_problem_id == problem_id
    assert new_problem.name.endswith("| structure = soaking")
    assert new_problem.is_temporary

    new_meta = _fetch_graphical_metadata(new_id, session)
    assert DiscreteNetwork.model_validate_json(new_meta.network_json).potentials["structure"].values == [0, 0, 1]
    assert DiagramAssumptions.model_validate_json(new_meta.assumptions_json) == DiagramAssumptions(
        chance={"structure": "soaking"}
    )

    session.expire_all()
    assert _fetch_graphical_metadata(problem_id, session).model_dump() == original_meta
    assert [(o.ideal, o.nadir) for o in session.get(ProblemDB, problem_id).objectives] == original_bounds


@pytest.mark.cvxpy
def test_hypothetical_background_checks_set_its_own_ideal_and_nadir(
    client: TestClient, session_and_user: dict, monkeypatch: pytest.MonkeyPatch
):
    """After the checks, the hypothetical is a normal problem whose ideal is what's attainable under the assumptions."""
    session: Session = session_and_user["session"]
    access_token = login(client)
    problem_id = _compile_oil_wildcatter(client, access_token)

    response = _hypothetical(
        client, access_token, problem_id, chance={"structure": "soaking"}, decisions={"drill": "yes"}
    )
    assert response.status_code == 200, response.text
    new_id, state_id = response.json()["problem_id"], response.json()["state_id"]

    _run_checks(session, monkeypatch, new_id, state_id)

    mod_state = session.get(CumulusModificationState, state_id)
    assert mod_state.is_ready
    assert mod_state.error is None
    new_problem = session.get(ProblemDB, new_id)
    assert not new_problem.is_temporary

    network, annotation = build_oil_wildcatter_demo()
    expected, *_ = compile_with_assumptions(
        network, annotation, DiagramAssumptions(chance={"structure": "soaking"}, decisions={"drill": "yes"})
    )
    best_profit = CVXPYSolver(expected).solve("profit_min").optimal_objectives["profit"]
    profit = next(o for o in new_problem.objectives if o.symbol == "profit")
    assert profit.ideal == pytest.approx(best_profit, rel=1e-6)
    assert Problem.from_problemdb(new_problem).get_nadir_point()["profit"] is not None


def test_hypothetical_stacks_and_rejects_conflicts(client: TestClient, session_and_user: dict):
    """A hypothetical of a hypothetical keeps the earlier assumptions; contradicting or repeating them is a 400."""
    session: Session = session_and_user["session"]
    access_token = login(client)
    problem_id = _compile_oil_wildcatter(client, access_token)
    first_id = _hypothetical(client, access_token, problem_id, chance={"structure": "soaking"}).json()["problem_id"]

    response = _hypothetical(client, access_token, first_id, decisions={"drill": "yes"})
    assert response.status_code == 200, response.text
    second_meta = _fetch_graphical_metadata(response.json()["problem_id"], session)
    assert DiagramAssumptions.model_validate_json(second_meta.assumptions_json) == DiagramAssumptions(
        chance={"structure": "soaking"}, decisions={"drill": "yes"}
    )

    conflict = _hypothetical(client, access_token, first_id, chance={"structure": "dry"})
    assert conflict.status_code == 400
    assert "structure = soaking" in conflict.json()["detail"]

    repeat = _hypothetical(client, access_token, first_id, chance={"structure": "soaking"})
    assert repeat.status_code == 400
    assert "nothing" in repeat.json()["detail"]


def test_stacked_hypothetical_keeps_assumed_decisions(client: TestClient, session_and_user: dict):
    """Stacking a chance assumption on a decision assumption keeps the decision fixed."""
    session: Session = session_and_user["session"]
    access_token = login(client)
    problem_id = _compile_oil_wildcatter(client, access_token)
    first_id = _hypothetical(client, access_token, problem_id, decisions={"drill": "yes"}).json()["problem_id"]

    response = _hypothetical(client, access_token, first_id, chance={"oil_price": "high"})

    assert response.status_code == 200, response.text
    stacked = session.get(ProblemDB, response.json()["problem_id"])
    assert any(c.symbol.startswith("assume_drill_") for c in stacked.constraints)
    assert stacked.name.endswith("| drill = yes | oil_price = high")


def test_hypothetical_rejects_chance_node_with_parents(client: TestClient, session_and_user: dict):
    """Only decisions and parentless chance nodes can be assumed."""
    access_token = login(client)
    problem_id = _compile_oil_wildcatter(client, access_token)

    response = _hypothetical(client, access_token, problem_id, chance={"test_result": "NS"})

    assert response.status_code == 400
    assert "has parents" in response.json()["detail"]


def test_hypothetical_of_non_graphical_problem_is_404(client: TestClient, session_and_user: dict):
    """A problem without a graphical model can't have a hypothetical."""
    access_token = login(client)

    response = _hypothetical(client, access_token, 1, chance={"structure": "soaking"})

    assert response.status_code == 404


def test_structure_and_whatif_follow_the_assumptions(client: TestClient, session_and_user: dict):
    """`/structure` reports the assumptions, and `/whatif` pins assumed decisions and rejects contradicting pins."""
    access_token = login(client)
    problem_id = _compile_oil_wildcatter(client, access_token)
    new_id = _hypothetical(
        client, access_token, problem_id, chance={"structure": "soaking"}, decisions={"test": "no", "drill": "yes"}
    ).json()["problem_id"]

    structure = get_json(client, f"/graphical/{new_id}/structure", access_token)
    assert structure.status_code == 200, structure.text
    assert structure.json()["assumptions"] == {
        "chance": {"structure": "soaking"},
        "decisions": {"test": "no", "drill": "yes"},
    }
    original = get_json(client, f"/graphical/{problem_id}/structure", access_token)
    assert original.json()["assumptions"] == {"chance": {}, "decisions": {}}

    whatif = post_json(client, "/graphical/whatif", {"problem_id": new_id, "evidence": {}}, access_token)
    assert whatif.status_code == 200, whatif.text
    assert whatif.json()["policy"] == {"test": "no", "drill": "yes"}
    assert whatif.json()["posteriors"]["structure"] == pytest.approx([0.0, 0.0, 1.0])
    assert whatif.json()["decision_posteriors"] == {"test": [0.0, 1.0], "drill": [1.0, 0.0]}

    contradicting = post_json(
        client,
        "/graphical/whatif",
        {"problem_id": new_id, "evidence": {}, "decision_evidence": {"drill": "no"}},
        access_token,
    )
    assert contradicting.status_code == 400
    assert "drill = yes" in contradicting.json()["detail"]


def test_sensitivity_and_voi_refuse_assumed_decisions(client: TestClient, session_and_user: dict):
    """These endpoints refuse with 501, since pyAgrum's `MEU()` would ignore an assumed decision."""
    access_token = login(client)
    problem_id = _compile_oil_wildcatter(client, access_token)
    new_id = _hypothetical(client, access_token, problem_id, decisions={"drill": "yes"}).json()["problem_id"]

    sensitivity = post_json(client, "/graphical/sensitivity", {"problem_id": new_id, "target": "profit"}, access_token)
    voi = post_json(
        client, "/graphical/voi", {"problem_id": new_id, "node": "structure", "target": "profit"}, access_token
    )

    assert sensitivity.status_code == 501
    assert voi.status_code == 501
    assert "hypothetical" in voi.json()["detail"]
