"""Tests for `desdeo/api/routers/graphical.py` (Phase 7 of the graphical-models plan).

Marked `graphical` (registered repo-wide in `pyproject.toml`, applies regardless of path). No
gurobipy needed: `/compile` only builds a `Problem` (`compile_diagram`), and `/whatif` only ever
calls pyAgrum's own inference -- neither solves the compiled MILP.

`_engine_cache` (the module-level session-scoped engine cache `graphical.py` defines) is cleared
before and after every test in this module -- it is a plain process-global dict, so leftover
entries from an earlier test in the same pytest process could otherwise make a later test's
"was the engine actually rebuilt/reused" assertions non-deterministic (see `_clear_engine_cache`).
"""

import pyagrum as gum
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from desdeo.api.models import (
    GraphicalModelMetaData,
    InteractiveSessionDB,
    ProblemDB,
    ProblemMetaDataDB,
    User,
    UserRole,
)
from desdeo.api.routers import graphical as graphical_router
from desdeo.api.routers.user_authentication import get_password_hash
from desdeo.problem.graphical import (
    DecisionAnnotation,
    DecisionNode,
    DiscreteNetwork,
    DiscreteVariable,
    IndexMap,
    ObjectiveNode,
    Potential,
    RootedJunctionTree,
    build_rooted_junction_tree,
    compile_diagram,
)
from desdeo.problem.graphical.adapters.pyagrum import to_pyagrum_influence_diagram

from .conftest import get_json, login, post_json

pytestmark = pytest.mark.graphical


@pytest.fixture(autouse=True)
def _clear_engine_cache():
    graphical_router._engine_cache.clear()
    yield
    graphical_router._engine_cache.clear()


# --- Fixture builders ---------------------------------------------------------------------------


def _chain_network() -> DiscreteNetwork:
    """A -> B, two binary chance nodes, with a genuinely *normalised* CPT for B.

    Unlike `tests/fixtures/graphical.py`'s own `chain_network`, which is deliberately unnormalised
    for structure-only testing -- see `compile.py`'s `_check_cpts_normalized` finding. `/whatif`
    doesn't itself enforce normalisation, but pyAgrum silently renormalises an unnormalised CPT
    row at inference time, which would make a hand-computed "independent oracle" expectation
    wrong unless the table is already a valid distribution to begin with.

    `values[b][a] == P(B=b | A=a)`: column `a=0` (`A=lo`) is `[0.9, 0.1]`, column `a=1` (`A=hi`)
    is `[0.2, 0.8]`, each summing to 1.
    """
    return DiscreteNetwork(
        variables=[
            DiscreteVariable(name="A", labels=["lo", "hi"]),
            DiscreteVariable(name="B", labels=["lo", "hi"]),
        ],
        arcs=[("A", "B")],
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6]),
            "B": Potential(variables=["B", "A"], shape=[2, 2], values=[[0.9, 0.2], [0.1, 0.8]]),
        },
    )


def _no_decision_annotation() -> DecisionAnnotation:
    return DecisionAnnotation(
        objectives=[
            ObjectiveNode(symbol="f1", parents=["B"], utility=Potential(variables=["B"], shape=[2], values=[0.0, 10.0]))
        ]
    )


def _decision_network() -> DiscreteNetwork:
    """A single binary chance node -- the decision `D` (built in the annotation) observes it."""
    return DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[2], values=[0.3, 0.7])},
    )


def _decision_annotation() -> DecisionAnnotation:
    """D observes A; reward for D matching A (mirrors `test_graphical_compile.py`'s own fixture)."""
    return DecisionAnnotation(
        decisions=[DecisionNode(symbol="D", states=["s0", "s1"], information_set=["A"])],
        objectives=[
            ObjectiveNode(
                symbol="fmatch",
                parents=["A", "D"],
                utility=Potential(variables=["A", "D"], shape=[2, 2], values=[[10.0, 0.0], [0.0, 10.0]]),
                maximize=True,
            )
        ],
    )


def _persist_graphical_metadata(
    session: Session, problem_id: int, network: DiscreteNetwork, annotation: DecisionAnnotation
) -> GraphicalModelMetaData:
    """Attach a `GraphicalModelMetaData` row directly, bypassing `/compile`.

    For tests that only exercise `/structure`/`/whatif` -- no need to also build/persist an
    RJT/IndexMap for those.
    """
    metadata_db = ProblemMetaDataDB(problem_id=problem_id)
    session.add(metadata_db)
    session.commit()
    session.refresh(metadata_db)

    graphical_metadata = GraphicalModelMetaData(
        metadata_id=metadata_db.id,
        network_json=network.model_dump_json(),
        annotation_json=annotation.model_dump_json(),
    )
    session.add(graphical_metadata)
    session.commit()
    session.refresh(graphical_metadata)
    return graphical_metadata


# --- POST /graphical/compile ---------------------------------------------------------------------


def test_compile_persists_problem_and_metadata(client: TestClient, session_and_user: dict):
    """A successful compile persists a `Problem` plus a `GraphicalModelMetaData`.

    All four JSON columns are populated, including `index_map_json` (only produced once Phase 4
    has run).
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    network = _chain_network()
    annotation = _no_decision_annotation()
    payload = {
        "network": network.model_dump(mode="json"),
        "annotation": annotation.model_dump(mode="json"),
        "name": "compiled_test_model",
    }
    response = post_json(client, "/graphical/compile", payload, access_token)
    assert response.status_code == 200, response.text
    problem_id = response.json()["problem_id"]

    problem_db = session.get(ProblemDB, problem_id)
    assert problem_db is not None
    assert problem_db.name == "compiled_test_model"
    assert len(problem_db.objectives) == 1

    metadata_db = session.exec(select(ProblemMetaDataDB).where(ProblemMetaDataDB.problem_id == problem_id)).first()
    graphical_rows = [m for m in metadata_db.all_metadata if isinstance(m, GraphicalModelMetaData)]
    assert len(graphical_rows) == 1
    meta = graphical_rows[0]

    # Round-trip, not just "is not None" -- a review pass caught the original version of this test
    # only checking presence, which would pass even if a column held garbage JSON. `rjt`/
    # `index_map` are compared against freshly built ones for the same `(network, annotation)`
    # (Phase 3/4's own suites already oracle-test *correctness* of that construction -- this is a
    # persistence-layer check: did the endpoint store what it computed, not "is the computation
    # right").
    assert DiscreteNetwork.model_validate_json(meta.network_json) == network
    assert DecisionAnnotation.model_validate_json(meta.annotation_json) == annotation
    expected_rjt = build_rooted_junction_tree(network, annotation, use_cache=False)
    _, expected_index_map = compile_diagram(network, annotation, rjt=expected_rjt)
    assert RootedJunctionTree.model_validate_json(meta.rjt_json) == expected_rjt
    assert IndexMap.model_validate_json(meta.index_map_json) == expected_index_map


def test_compile_bad_annotation_returns_400(client: TestClient, session_and_user: dict):
    """An annotation that doesn't validate against the network is a 400, not a bare 500.

    `ValueError` from `validate_against` (here: an objective referencing an unknown parent) is
    translated, not left to surface raw.
    """
    access_token = login(client)
    network = _chain_network()
    bad_annotation = DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="f1",
                parents=["ZZZ"],
                utility=Potential(variables=["ZZZ"], shape=[2], values=[0.0, 1.0]),
            )
        ]
    )
    payload = {"network": network.model_dump(mode="json"), "annotation": bad_annotation.model_dump(mode="json")}
    response = post_json(client, "/graphical/compile", payload, access_token)
    assert response.status_code == 400


def test_compile_guest_rejected(client: TestClient, session_and_user: dict):
    """Guest users may not compile graphical models, mirroring `routers/problem.py`'s `add_problem`."""
    session: Session = session_and_user["session"]
    guest = User(username="guest_user", password_hash=get_password_hash("guest_pw"), role=UserRole.guest, group="test")
    session.add(guest)
    session.commit()

    access_token = login(client, username="guest_user", password="guest_pw")  # noqa: S106
    network = _chain_network()
    annotation = _no_decision_annotation()
    payload = {"network": network.model_dump(mode="json"), "annotation": annotation.model_dump(mode="json")}
    response = post_json(client, "/graphical/compile", payload, access_token)
    assert response.status_code == 401


# --- GET /graphical/{problem_id}/structure --------------------------------------------------------


def test_structure_returns_nodes_and_edges(client: TestClient, session_and_user: dict):
    """Nodes get the right `kind`/`states`; edges cover CPT parents, information sets, and objectives.

    Uses a fixture with at least one decision and one objective.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _decision_network()
    annotation = _decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    response = get_json(client, f"/graphical/{problem.id}/structure", access_token)
    assert response.status_code == 200
    data = response.json()

    nodes_by_id = {n["id"]: n for n in data["nodes"]}
    assert nodes_by_id["A"]["kind"] == "chance"
    assert nodes_by_id["A"]["states"] == ["lo", "hi"]
    assert nodes_by_id["D"]["kind"] == "decision"
    assert nodes_by_id["D"]["states"] == ["s0", "s1"]
    # Objective node ids are namespaced `util_<symbol>` (not the bare symbol) so they can never
    # collide with a chance-node id -- nothing forbids an ObjectiveNode.symbol from equalling a
    # chance-node name (see get_structure's own comment).
    assert nodes_by_id["util_fmatch"]["kind"] == "objective"
    assert nodes_by_id["util_fmatch"]["label"] == "fmatch"
    assert nodes_by_id["util_fmatch"]["states"] is None

    edges = {(e["source"], e["target"]) for e in data["edges"]}
    assert ("A", "D") in edges  # D's information_set
    assert ("A", "util_fmatch") in edges  # fmatch's parents
    assert ("D", "util_fmatch") in edges
    # Every edge carries a stable id derived from its endpoints.
    assert all(e["id"] == f"{e['source']}->{e['target']}" for e in data["edges"])


def test_structure_404_when_no_graphical_metadata(client: TestClient, session_and_user: dict):
    """A problem with `ProblemMetaDataDB` but no `GraphicalModelMetaData` row is a 404."""
    session: Session = session_and_user["session"]
    access_token = login(client)
    problem = session.exec(select(ProblemDB).where(ProblemDB.name == "The river pollution problem")).first()

    response = get_json(client, f"/graphical/{problem.id}/structure", access_token)
    assert response.status_code == 404


def test_structure_ownership_enforced(client: TestClient, session_and_user: dict):
    """A `dm`-role user cannot read another user's problem's structure.

    Analyst/admin roles bypass ownership by design -- see `fetch_problem_with_role_check`.
    """
    session: Session = session_and_user["session"]
    dm_user = User(username="dm_user", password_hash=get_password_hash("dm_pw"), role=UserRole.dm, group="test")
    session.add(dm_user)
    session.commit()

    problem = session.exec(select(ProblemDB)).first()  # owned by the default "analyst" test user
    network = _chain_network()
    annotation = _no_decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    access_token = login(client, username="dm_user", password="dm_pw")  # noqa: S106
    response = get_json(client, f"/graphical/{problem.id}/structure", access_token)
    assert response.status_code in (403, 404)


# --- POST /graphical/whatif ------------------------------------------------------------------------


def test_whatif_no_decision_posteriors_and_utilities(client: TestClient, session_and_user: dict):
    """Posteriors and utilities under hard evidence, checked against a from-scratch computation.

    Computed directly off the network's own CPTs/utility table -- not self-consistency.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _chain_network()
    annotation = _no_decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {"A": "hi"}}
    response = post_json(client, "/graphical/whatif", payload, access_token)
    assert response.status_code == 200
    data = response.json()

    # Independent oracle: A is pinned to "hi", so B's posterior is exactly B's own CPT row for
    # A="hi" -- [0.2, 0.8] per `_chain_network`'s own table, read directly, no dependency on this
    # endpoint's own inference machinery.
    assert data["posteriors"]["A"] == pytest.approx([0.0, 1.0])
    assert data["posteriors"]["B"] == pytest.approx([0.2, 0.8])
    assert data["policy"] is None
    assert data["utilities"]["f1"] == pytest.approx(0.0 * 0.2 + 10.0 * 0.8)


def test_whatif_with_decision_single_objective_matches_pyagrum_oracle(client: TestClient, session_and_user: dict):
    """Policy and utility for a single-objective decision diagram, checked against two oracles.

    A hand-computed literal value first -- a review pass caught the original version of this test
    checking only against a `ShaferShenoyLIMIDInference` oracle built the same way the endpoint
    itself builds it (`to_pyagrum_influence_diagram` + `ShaferShenoyLIMIDInference` +
    `MEU()["mean"]`/`posterior("D").argmax()`), which cannot fail on a bug shared by both sides --
    exactly the `maximize`-sign bug this same review pass found (see the plan's Phase 7 findings).
    With `A` pinned to `"lo"`, `D` (which observes `A`) knows exactly which branch it's in, so the
    optimal choice is deterministic and the expected utility is the single matching-reward cell,
    no averaging: `D="s0"` (matches `A="lo"`), utility `= 10.0` (`_decision_annotation`'s own
    utility table, `utility[A=lo, D=s0] = 10.0`). The pyAgrum-oracle cross-check is kept as a
    second, independent confirmation.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _decision_network()
    annotation = _decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    evidence = {"A": "lo"}
    payload = {"problem_id": problem.id, "evidence": evidence}
    response = post_json(client, "/graphical/whatif", payload, access_token)
    assert response.status_code == 200
    data = response.json()

    assert data["utilities"]["fmatch"] == pytest.approx(10.0)
    assert data["policy"]["D"] == "s0"

    diagram = to_pyagrum_influence_diagram(network, annotation)
    inference = gum.ShaferShenoyLIMIDInference(diagram)
    inference.setEvidence(evidence)
    inference.makeInference()

    assert data["utilities"]["fmatch"] == pytest.approx(inference.MEU()["mean"])
    expected_probs = inference.posterior("D").toarray()
    expected_choice = annotation.decisions[0].states[int(expected_probs.argmax())]
    assert data["policy"]["D"] == expected_choice


def test_whatif_minimizing_objective_with_decision_picks_and_reports_the_minimum(
    client: TestClient, session_and_user: dict
):
    """A `maximize=False` decision diagram picks the *minimum*-cost choice, not the maximum.

    Regression test for the code review's first finding: `to_pyagrum_influence_diagram` used to
    hand pyAgrum's own (always-maximizing) `MEU()`/policy search the utility table verbatim, so a
    minimizing (cost) objective -- `ObjectiveNode.maximize`'s own default -- silently got the
    *worst* achievable value reported as though it were optimal. `D` observing `A` and paying a
    mismatch cost (`cost[A=lo,D=s0]=0`, `cost[A=lo,D=s1]=10`, `cost[A=hi,D=s0]=10`,
    `cost[A=hi,D=s1]=0`) should choose `D="s0"` when `A="lo"` (cost `0.0`), the mirror image of
    `_decision_annotation`'s maximize=True reward table.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _decision_network()
    annotation = DecisionAnnotation(
        decisions=[DecisionNode(symbol="D", states=["s0", "s1"], information_set=["A"])],
        objectives=[
            ObjectiveNode(
                symbol="cost",
                parents=["A", "D"],
                utility=Potential(variables=["A", "D"], shape=[2, 2], values=[[0.0, 10.0], [10.0, 0.0]]),
                maximize=False,
            )
        ],
    )
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {"A": "lo"}}
    response = post_json(client, "/graphical/whatif", payload, access_token)
    assert response.status_code == 200
    data = response.json()

    assert data["policy"]["D"] == "s0"
    assert data["utilities"]["cost"] == pytest.approx(0.0)


def test_whatif_policy_omits_decision_not_fully_determined_by_evidence(client: TestClient, session_and_user: dict):
    """A decision left open by partial/no evidence is absent from `policy`, not a marginal guess.

    Regression test for the code review's second finding: with no evidence at all, `D`'s
    `posterior` is the evidence-induced *marginal* `[0.3, 0.7]` (`_decision_network`'s own prior on
    `A`), not a deterministic policy -- reporting its argmax as `policy["D"]` used to claim a
    confident "chosen state" that no actual policy produces (the true optimum is contingent:
    choose `s0` when `A` turns out `lo`, `s1` when `A` turns out `hi`).
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _decision_network()
    annotation = _decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {}}
    response = post_json(client, "/graphical/whatif", payload, access_token)
    assert response.status_code == 200
    data = response.json()

    assert "D" not in (data["policy"] or {})


def test_whatif_engine_cache_reused_and_evidence_does_not_leak(
    client: TestClient, session_and_user: dict, monkeypatch: pytest.MonkeyPatch
):
    """Two calls in the same session with different evidence don't leak evidence into each other.

    Each call's posteriors reflect only its own evidence, and the engine/diagram is built only
    once (proven via a call-counter on `_build_engine`, not just "trust the code").
    """
    session: Session = session_and_user["session"]
    user = session_and_user["user"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _chain_network()
    annotation = _no_decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    interactive_session = InteractiveSessionDB(user_id=user.id, info="whatif cache test session")
    session.add(interactive_session)
    session.commit()
    session.refresh(interactive_session)

    build_calls = 0
    original_build_engine = graphical_router._build_engine

    def _counting_build_engine(network_, annotation_):
        nonlocal build_calls
        build_calls += 1
        return original_build_engine(network_, annotation_)

    monkeypatch.setattr(graphical_router, "_build_engine", _counting_build_engine)

    payload_hi = {"problem_id": problem.id, "session_id": interactive_session.id, "evidence": {"A": "hi"}}
    payload_lo = {"problem_id": problem.id, "session_id": interactive_session.id, "evidence": {"A": "lo"}}

    response_hi = post_json(client, "/graphical/whatif", payload_hi, access_token)
    response_lo = post_json(client, "/graphical/whatif", payload_lo, access_token)
    assert response_hi.status_code == 200
    assert response_lo.status_code == 200
    assert build_calls == 1  # second call reused the cached engine

    data_hi = response_hi.json()
    data_lo = response_lo.json()
    assert data_hi["posteriors"]["A"] == pytest.approx([0.0, 1.0])
    assert data_lo["posteriors"]["A"] == pytest.approx([1.0, 0.0])
    assert data_hi["posteriors"]["B"] == pytest.approx([0.2, 0.8])
    assert data_lo["posteriors"]["B"] == pytest.approx([0.9, 0.1])


def test_whatif_engine_cache_invalidated_by_content_change(
    client: TestClient, session_and_user: dict, monkeypatch: pytest.MonkeyPatch
):
    """Editing a persisted model's network/annotation invalidates its cached engine.

    Regression coverage for `_get_or_build_entry`'s content-hash mechanism -- a review pass flagged
    this as entirely untested (the one thing the cache exists to get right: never serve inference
    against a stale structure/CPTs after a recompile). Simulates an in-place edit by mutating the
    same `GraphicalModelMetaData` row's `network_json` directly, then confirms a second `/whatif`
    call in the same session rebuilds (not reuses) the engine and reflects the *new* network.
    """
    session: Session = session_and_user["session"]
    user = session_and_user["user"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _chain_network()
    annotation = _no_decision_annotation()
    meta = _persist_graphical_metadata(session, problem.id, network, annotation)

    interactive_session = InteractiveSessionDB(user_id=user.id, info="cache invalidation test")
    session.add(interactive_session)
    session.commit()
    session.refresh(interactive_session)

    build_calls = 0
    original_build_engine = graphical_router._build_engine

    def _counting_build_engine(network_, annotation_):
        nonlocal build_calls
        build_calls += 1
        return original_build_engine(network_, annotation_)

    monkeypatch.setattr(graphical_router, "_build_engine", _counting_build_engine)

    payload_no_evidence = {"problem_id": problem.id, "session_id": interactive_session.id, "evidence": {}}
    first = post_json(client, "/graphical/whatif", payload_no_evidence, access_token)
    assert first.status_code == 200
    assert build_calls == 1
    assert first.json()["posteriors"]["A"] == pytest.approx([0.4, 0.6])  # _chain_network's own prior

    # Same structure, different content: A's prior changes from [0.4, 0.6] to [0.9, 0.1].
    edited_network = network.model_copy(
        update={"potentials": {**network.potentials, "A": Potential(variables=["A"], shape=[2], values=[0.9, 0.1])}}
    )
    meta.network_json = edited_network.model_dump_json()
    session.add(meta)
    session.commit()

    second = post_json(client, "/graphical/whatif", payload_no_evidence, access_token)
    assert second.status_code == 200
    assert build_calls == 2  # content changed -> rebuilt, not reused
    assert second.json()["posteriors"]["A"] == pytest.approx([0.9, 0.1])  # reflects the edit

    # A third, unedited call reuses the (now up to date) cached engine again.
    third = post_json(client, "/graphical/whatif", payload_no_evidence, access_token)
    assert third.status_code == 200
    assert build_calls == 2


def test_whatif_multi_objective_with_decision_raises_501(client: TestClient, session_and_user: dict):
    """More than one objective plus at least one decision is an honest 501, not a wrong number.

    Regression coverage for the deliberate scope cut in `_compute_utilities`: confirmed empirically
    (module docstring) that `ShaferShenoyLIMIDInference` exposes no joint-posterior primitive to
    recover one objective's own share of a jointly-solved policy independent of the others.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _decision_network()
    annotation = DecisionAnnotation(
        decisions=[DecisionNode(symbol="D", states=["s0", "s1"], information_set=["A"])],
        objectives=[
            ObjectiveNode(symbol="f1", parents=["A"], utility=Potential(variables=["A"], shape=[2], values=[0.0, 1.0])),
            ObjectiveNode(symbol="f2", parents=["D"], utility=Potential(variables=["D"], shape=[2], values=[1.0, 0.0])),
        ],
    )
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {}}
    response = post_json(client, "/graphical/whatif", payload, access_token)
    assert response.status_code == 501


def test_whatif_ownership_enforced_for_problem(client: TestClient, session_and_user: dict):
    """A `dm`-role user cannot call `/whatif` against another user's problem."""
    session: Session = session_and_user["session"]
    dm_user = User(username="dm_user2", password_hash=get_password_hash("dm_pw"), role=UserRole.dm, group="test")
    session.add(dm_user)
    session.commit()

    problem = session.exec(select(ProblemDB)).first()  # owned by the default "analyst" test user
    network = _chain_network()
    annotation = _no_decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    access_token = login(client, username="dm_user2", password="dm_pw")  # noqa: S106
    payload = {"problem_id": problem.id, "evidence": {}}
    response = post_json(client, "/graphical/whatif", payload, access_token)
    assert response.status_code == 404


def test_whatif_ownership_enforced_for_session_id(client: TestClient, session_and_user: dict):
    """A user cannot pass someone else's `session_id`, even for a problem they own themselves."""
    session: Session = session_and_user["session"]
    analyst = session_and_user["user"]

    dm_user = User(username="dm_user3", password_hash=get_password_hash("dm_pw"), role=UserRole.dm, group="test")
    session.add(dm_user)
    session.commit()

    dm_access_token = login(client, username="dm_user3", password="dm_pw")  # noqa: S106
    network = _chain_network()
    annotation = _no_decision_annotation()
    compile_payload = {"network": network.model_dump(mode="json"), "annotation": annotation.model_dump(mode="json")}
    compile_response = post_json(client, "/graphical/compile", compile_payload, dm_access_token)
    assert compile_response.status_code == 200
    dm_problem_id = compile_response.json()["problem_id"]

    analyst_session = InteractiveSessionDB(user_id=analyst.id, info="analyst's own session")
    session.add(analyst_session)
    session.commit()
    session.refresh(analyst_session)

    payload = {"problem_id": dm_problem_id, "session_id": analyst_session.id, "evidence": {}}
    response = post_json(client, "/graphical/whatif", payload, dm_access_token)
    assert response.status_code == 404
