"""Tests for `/graphical/{id}/structure` and `/graphical/whatif` on stationary LP problems."""

from collections import OrderedDict

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from desdeo.api.models import CumulusClassificationState, ProblemDB, ReferencePoint, StateDB, StationaryModelMetaData
from desdeo.api.routers import graphical as graphical_router
from desdeo.api.routers.utils import load_problem
from desdeo.api.utils.stationary import add_stationary_lp_problem
from desdeo.problem.graphical import (
    DiscreteNetwork,
    ObjectiveNode,
    Potential,
    StationaryDiagram,
    compile_stationary_lp,
    evaluate_stationary_policy,
    full_observation_information_set,
    read_randomised_policy,
    stationary_step_posteriors,
)
from desdeo.tools import CVXPYSolver, CVXPYSolverOptions, SolverResults
from tests.fixtures import decision_chain_2tbn_network

from .conftest import delete_json, get_json, login, post_json

pytestmark = pytest.mark.graphical


def _objective(symbol: str, parents: list[str], utility: list, *, maximize: bool) -> ObjectiveNode:
    return ObjectiveNode(
        symbol=symbol,
        parents=parents,
        utility=Potential(variables=parents, shape=list(np.shape(utility)), values=utility),
        maximize=maximize,
    )


def _assert_close(actual: dict[str, list[float]], expected: dict, atol: float) -> None:
    assert set(actual) == set(expected)
    for key, values in expected.items():
        np.testing.assert_allclose(actual[key], values, atol=atol, rtol=0)


def _diagram(network: DiscreteNetwork | None = None) -> StationaryDiagram:
    """`decision_chain_2tbn_network` with atemporal G and full-observation Dt and Et."""
    if network is None:
        network = decision_chain_2tbn_network(atemporal=True)
    objectives = [
        _objective("x", ["Xt"], [1.0, 3.0], maximize=True),
        _objective("c", ["Dt", "Yt", "Et"], [[[0.3, 0.8], [0.5, 0.5]], [[0.1, 0.9], [1.0, 0.2]]], maximize=False),
    ]
    decisions = ["G", "Dt", "Et"]
    provisional = StationaryDiagram.with_default_information_sets(
        network, dict.fromkeys(decisions), objectives, discount=0.9
    )
    full_sets = {symbol: full_observation_information_set(provisional, symbol) for symbol in decisions}
    return StationaryDiagram.with_default_information_sets(network, full_sets, objectives, discount=0.9)


def _store_solution_state(session: Session, problem_id: int, results: list[SolverResults]) -> StateDB:
    """Persist `results` as a CUMULUS solve state of `problem_id`."""
    state = StateDB.create(
        session,
        problem_id=problem_id,
        state=CumulusClassificationState(preferences=ReferencePoint(aspiration_levels={}), solver_results=results),
    )
    session.commit()
    session.refresh(state)
    return state


def _solved(session_and_user: dict) -> tuple[ProblemDB, SolverResults, StateDB]:
    """The fixture problem, solved for `c` with CVXPY + HiGHS and stored as a state."""
    session: Session = session_and_user["session"]
    user = session_and_user["user"]
    problem_db = add_stationary_lp_problem(session, user, _diagram(), name="Stationary chain")
    problem = load_problem(problem_db)
    result = CVXPYSolver(problem, CVXPYSolverOptions(solver="HIGHS")).solve("c")
    assert result.success
    state = _store_solution_state(session, problem_db.id, [result])
    return problem_db, result, state


def _find_zero_prob_evidence() -> dict[str, str] | None:
    """Search node-pair evidence that triggers a zero-probability ValueError."""
    diagram = _diagram()
    nodes = ["X0", "Xt", "Zt", "Yt", "Et", "Dt", "G"]
    labels = ["lo", "hi"]
    for i, n1 in enumerate(nodes):
        for i1, l1 in enumerate(labels):
            for n2 in nodes[i + 1 :]:
                for i2, l2 in enumerate(labels):
                    try:
                        stationary_step_posteriors(diagram, {}, {n1: i1, n2: i2})
                    except ValueError as exc:
                        if "probability 0" in str(exc):
                            return {n1: l1, n2: l2}
    return None


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch: pytest.MonkeyPatch):
    """A fresh `_stationary_cache` per test, since SQLite reuses ids across tests."""
    monkeypatch.setattr(graphical_router, "_stationary_cache", OrderedDict())


def test_structure(client: TestClient, session_and_user: dict):
    """Structure lists every network node plus objectives with correct kinds, slices, and edges."""
    token = login(client)
    session: Session = session_and_user["session"]
    user = session_and_user["user"]
    problem_db = add_stationary_lp_problem(session, user, _diagram(), name="Stationary chain")
    diagram = _diagram()

    resp = get_json(client, f"/graphical/{problem_db.id}/structure", token)
    assert resp.status_code == 200
    data = resp.json()
    nodes = {n["id"]: n for n in data["nodes"]}

    names = [variable.name for variable in diagram.network.variables]
    assert set(nodes) == {*names, "util_x", "util_c"}
    for name in names:
        assert nodes[name]["kind"] == ("decision" if name in ("G", "Dt", "Et") else "chance")

    assert nodes["util_x"]["kind"] == "objective"
    assert nodes["util_x"]["slice"] is None
    assert nodes["util_c"]["kind"] == "objective"
    assert nodes["util_c"]["slice"] is None

    for name in diagram.structure.slice0:
        assert nodes[name]["slice"] == "slice0"
    for name in diagram.structure.slicet:
        assert nodes[name]["slice"] == "slicet"
    for name in diagram.structure.atemporal:
        assert nodes[name]["slice"] == "atemporal"

    edges = {(e["source"], e["target"]) for e in data["edges"]}
    for decision in diagram.decisions:
        for parent in decision.information_set:
            assert (parent, decision.symbol) in edges
    for parent in ["Dt", "Yt", "Et"]:
        assert (parent, "util_c") in edges


def test_whatif_status_quo(client: TestClient, session_and_user: dict):
    """Status-quo whatif returns correct posteriors, utilities, and stationary metadata."""
    token = login(client)
    session: Session = session_and_user["session"]
    user = session_and_user["user"]
    problem_db = add_stationary_lp_problem(session, user, _diagram(), name="Stationary chain")
    diagram = _diagram()

    resp = post_json(client, "/graphical/whatif", {"problem_id": problem_db.id}, token)
    assert resp.status_code == 200
    data = resp.json()

    expected = stationary_step_posteriors(diagram, {})
    decisions = {decision.symbol for decision in diagram.decisions}
    exp_dec = {k: v for k, v in expected.posteriors.items() if k in decisions}
    exp_non = {k: v for k, v in expected.posteriors.items() if k not in decisions}

    _assert_close(data["decision_posteriors"], exp_dec, atol=1e-12)
    _assert_close(data["posteriors"], exp_non, atol=1e-12)
    assert data["utilities"] == pytest.approx(expected.utilities, abs=1e-12)
    assert "D0" not in data["posteriors"]
    assert data["stationary"]["objectives"] == pytest.approx(
        evaluate_stationary_policy(diagram, {}).objectives, abs=1e-12
    )
    assert data["stationary"]["discount"] == 0.9
    assert sum(data["stationary"]["arrival_shares"]["G"]) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(data["stationary"]["arrival_shares"]["G"], expected.posteriors["G"], atol=1e-12, rtol=0)
    assert data["policy"] == {}
    assert data["utility_std_devs"] == {}
    assert data["off_policy_decisions"] == []


def test_arrival_shares_with_atemporal_parent(client: TestClient, session_and_user: dict):
    """Status-quo arrival shares contract the atemporal decision's atemporal ancestors."""
    token = login(client)
    session: Session = session_and_user["session"]
    user = session_and_user["user"]

    network = decision_chain_2tbn_network(atemporal=True)
    a = network.variables[[v.name for v in network.variables].index("G")].model_copy(update={"name": "A"})
    potentials = {
        **network.potentials,
        "A": Potential(variables=["A"], shape=[2], values=[0.25, 0.75]),
        "G": Potential(variables=["G", "A"], shape=[2, 2], values=[[0.9, 0.3], [0.1, 0.7]]),
    }
    network = DiscreteNetwork(
        variables=[*network.variables, a], arcs=[*network.arcs, ("A", "G")], potentials=potentials
    )

    diagram = _diagram(network)
    problem_db = add_stationary_lp_problem(session, user, diagram, name="Stationary chain with A")

    resp = post_json(client, "/graphical/whatif", {"problem_id": problem_db.id}, token)
    assert resp.status_code == 200
    data = resp.json()

    # P(G) = sum_A P(G | A) P(A).
    expected_arrival_shares = [0.9 * 0.25 + 0.3 * 0.75, 0.1 * 0.25 + 0.7 * 0.75]
    np.testing.assert_allclose(data["stationary"]["arrival_shares"]["G"], expected_arrival_shares, atol=1e-12, rtol=0)


@pytest.mark.cvxpy
def test_whatif_solution(client: TestClient, session_and_user: dict):
    """Solution whatif posteriors and objectives match read_randomised_policy results."""
    token = login(client)
    problem_db, result, state = _solved(session_and_user)
    diagram = _diagram()
    _, index_map = compile_stationary_lp(diagram)
    policies = read_randomised_policy(index_map, result.optimal_variables)

    resp = post_json(
        client,
        "/graphical/whatif",
        {"problem_id": problem_db.id, "solution": {"state_id": state.id, "solution_index": 0}},
        token,
    )
    assert resp.status_code == 200
    data = resp.json()

    expected = stationary_step_posteriors(diagram, policies)
    decisions = {decision.symbol for decision in diagram.decisions}
    exp_dec = {k: v for k, v in expected.posteriors.items() if k in decisions}
    exp_non = {k: v for k, v in expected.posteriors.items() if k not in decisions}

    _assert_close(data["decision_posteriors"], exp_dec, atol=1e-10)
    _assert_close(data["posteriors"], exp_non, atol=1e-10)
    assert data["stationary"]["objectives"] == pytest.approx(
        evaluate_stationary_policy(diagram, policies).objectives, abs=1e-10
    )
    assert data["stationary"]["objectives"] == pytest.approx(result.optimal_objectives, abs=1e-6)
    assert sum(data["stationary"]["arrival_shares"]["G"]) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(
        data["stationary"]["arrival_shares"]["G"], policies["G"].probabilities[0], atol=1e-12, rtol=0
    )


@pytest.mark.cvxpy
def test_whatif_evidence(client: TestClient, session_and_user: dict):
    """Evidence and decision_evidence condition posteriors; stationary objectives unchanged."""
    token = login(client)
    problem_db, result, state = _solved(session_and_user)
    diagram = _diagram()
    _, index_map = compile_stationary_lp(diagram)
    policies = read_randomised_policy(index_map, result.optimal_variables)

    # Pin Dt to a state the policy takes when X0 = "hi".
    dt = int(np.argmax(stationary_step_posteriors(diagram, policies, {"X0": 1}).posteriors["Dt"]))
    expected = stationary_step_posteriors(diagram, policies, {"X0": 1, "Dt": dt})

    resp = post_json(
        client,
        "/graphical/whatif",
        {
            "problem_id": problem_db.id,
            "evidence": {"X0": "hi"},
            "decision_evidence": {"Dt": ["lo", "hi"][dt]},
            "solution": {"state_id": state.id, "solution_index": 0},
        },
        token,
    )
    assert resp.status_code == 200
    data = resp.json()

    decisions = {decision.symbol for decision in diagram.decisions}
    exp_dec = {k: v for k, v in expected.posteriors.items() if k in decisions}
    exp_non = {k: v for k, v in expected.posteriors.items() if k not in decisions}

    _assert_close(data["decision_posteriors"], exp_dec, atol=1e-10)
    _assert_close(data["posteriors"], exp_non, atol=1e-10)
    assert data["utilities"] == pytest.approx(expected.utilities, abs=1e-10)
    assert data["stationary"]["objectives"] == pytest.approx(
        evaluate_stationary_policy(diagram, policies).objectives, abs=1e-10
    )


def test_evidence_errors(client: TestClient, session_and_user: dict):
    """Invalid evidence (bad label, bad node, chance in decision_evidence, D0, zero-prob) -> 422."""
    token = login(client)
    session: Session = session_and_user["session"]
    user = session_and_user["user"]
    problem_db = add_stationary_lp_problem(session, user, _diagram(), name="Stationary chain")

    resp = post_json(client, "/graphical/whatif", {"problem_id": problem_db.id, "evidence": {"X0": "medium"}}, token)
    assert resp.status_code == 422

    resp = post_json(client, "/graphical/whatif", {"problem_id": problem_db.id, "evidence": {"nope": "lo"}}, token)
    assert resp.status_code == 422

    resp = post_json(
        client, "/graphical/whatif", {"problem_id": problem_db.id, "decision_evidence": {"X0": "lo"}}, token
    )
    assert resp.status_code == 422

    resp = post_json(client, "/graphical/whatif", {"problem_id": problem_db.id, "evidence": {"D0": "lo"}}, token)
    assert resp.status_code == 422

    zero_ev = _find_zero_prob_evidence()
    assert zero_ev is not None
    resp = post_json(client, "/graphical/whatif", {"problem_id": problem_db.id, "evidence": zero_ev}, token)
    assert resp.status_code == 422
    assert "probability 0" in resp.json()["detail"]


def test_unsupported_and_404(client: TestClient, session_and_user: dict):
    """Sensitivity/VOI/policy/solve return 501; structure/whatif on plain problem return 404."""
    token = login(client)
    session: Session = session_and_user["session"]
    user = session_and_user["user"]
    problem_db = add_stationary_lp_problem(session, user, _diagram(), name="Stationary chain")

    resp = post_json(client, "/graphical/sensitivity", {"problem_id": problem_db.id, "target": "Xt"}, token)
    assert resp.status_code == 501

    resp = post_json(client, "/graphical/voi", {"problem_id": problem_db.id, "node": "X0", "target": "x"}, token)
    assert resp.status_code == 501

    resp = post_json(client, f"/graphical/{problem_db.id}/policy", {"variable_values": {}}, token)
    assert resp.status_code == 501

    resp = post_json(client, f"/graphical/{problem_db.id}/solve", {"objective_symbol": "c"}, token)
    assert resp.status_code == 501

    plain = session.exec(select(ProblemDB)).first()
    assert plain is not None
    resp = get_json(client, f"/graphical/{plain.id}/structure", token)
    assert resp.status_code == 404

    resp = post_json(client, "/graphical/whatif", {"problem_id": plain.id}, token)
    assert resp.status_code == 404


def test_evidence_sets(client: TestClient, session_and_user: dict):
    """Evidence sets can be created, listed, and deleted on a stationary problem."""
    token = login(client)
    session: Session = session_and_user["session"]
    user = session_and_user["user"]
    problem_db = add_stationary_lp_problem(session, user, _diagram(), name="Stationary chain")

    resp = post_json(
        client,
        f"/graphical/{problem_db.id}/evidence-sets",
        {"name": "pinned", "evidence": {"X0": "hi"}},
        token,
    )
    assert resp.status_code == 200
    set_id = resp.json()["id"]

    resp = get_json(client, f"/graphical/{problem_db.id}/evidence-sets", token)
    assert resp.status_code == 200
    assert set_id in [e["id"] for e in resp.json()]

    resp = delete_json(client, f"/graphical/evidence-sets/{set_id}", token)
    assert resp.status_code < 300


@pytest.mark.cvxpy
def test_cache(monkeypatch: pytest.MonkeyPatch, client: TestClient, session_and_user: dict):
    """read_randomised_policy is called only once when the cache is warm."""
    token = login(client)
    problem_db, _, state = _solved(session_and_user)

    calls: list[int] = []
    original = graphical_router.read_randomised_policy

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(graphical_router, "read_randomised_policy", counting)

    base = {"problem_id": problem_db.id, "solution": {"state_id": state.id, "solution_index": 0}}
    resp1 = post_json(client, "/graphical/whatif", base, token)
    assert resp1.status_code == 200

    resp2 = post_json(client, "/graphical/whatif", {**base, "evidence": {"X0": "hi"}}, token)
    assert resp2.status_code == 200

    assert len(calls) == 1


@pytest.mark.cvxpy
def test_cache_evicts_deleted_state(client: TestClient, session_and_user: dict):
    """A new state that reuses a deleted state's id gets its own policies, not the cached ones."""
    token = login(client)
    session: Session = session_and_user["session"]
    problem_db, _, state = _solved(session_and_user)
    ref = {"problem_id": problem_db.id, "solution": {"state_id": state.id, "solution_index": 0}}
    assert post_json(client, "/graphical/whatif", ref, token).status_code == 200
    state_id = state.id
    session.delete(state)
    session.commit()
    result_x = CVXPYSolver(load_problem(problem_db), CVXPYSolverOptions(solver="HIGHS")).solve("x")
    new_state = _store_solution_state(session, problem_db.id, [result_x])
    assert new_state.id == state_id
    resp = post_json(client, "/graphical/whatif", ref, token)
    assert resp.status_code == 200
    assert resp.json()["stationary"]["objectives"] == pytest.approx(result_x.optimal_objectives, abs=1e-6)


@pytest.mark.cvxpy
def test_cache_misses_replaced_state(client: TestClient, session_and_user: dict):
    """A state replaced in place under the same ids, as by a reseed, misses the cache."""
    token = login(client)
    session: Session = session_and_user["session"]
    problem_db, _, state = _solved(session_and_user)
    ref = {"problem_id": problem_db.id, "solution": {"state_id": state.id, "solution_index": 0}}
    resp1 = post_json(client, "/graphical/whatif", ref, token)
    assert resp1.status_code == 200
    result_x = CVXPYSolver(load_problem(problem_db), CVXPYSolverOptions(solver="HIGHS")).solve("x")
    state.state.solver_results = [result_x]
    state.base_state.date_time = "2000-01-01T00:00:00"
    session.add(state)
    session.commit()
    resp2 = post_json(client, "/graphical/whatif", ref, token)
    assert resp2.status_code == 200
    assert resp2.json()["stationary"]["objectives"] == pytest.approx(result_x.optimal_objectives, abs=1e-6)
    assert resp2.json()["stationary"]["objectives"] != resp1.json()["stationary"]["objectives"]


def test_cache_misses_replaced_metadata(client: TestClient, session_and_user: dict):
    """Metadata replaced in place under the same ids, as by a reseed, misses the cache."""
    token = login(client)
    session: Session = session_and_user["session"]
    user = session_and_user["user"]
    problem_db = add_stationary_lp_problem(session, user, _diagram(), name="Stationary chain")
    resp1 = post_json(client, "/graphical/whatif", {"problem_id": problem_db.id}, token)
    assert resp1.status_code == 200
    assert resp1.json()["stationary"]["discount"] == 0.9
    metas = session.exec(select(StationaryModelMetaData)).all()
    meta = next(m for m in metas if m.metadata_instance.problem_id == problem_db.id)
    meta.diagram_json = _diagram().model_copy(update={"discount": 0.8}).model_dump_json()
    session.add(meta)
    session.commit()
    resp2 = post_json(client, "/graphical/whatif", {"problem_id": problem_db.id}, token)
    assert resp2.status_code == 200
    assert resp2.json()["stationary"]["discount"] == pytest.approx(0.8)
