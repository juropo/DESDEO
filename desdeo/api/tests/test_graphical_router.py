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
    Group,
    InteractiveSessionDB,
    ProblemDB,
    ProblemMetaDataDB,
    User,
    UserRole,
    UserSavedSolutionDB,
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
    RiskMeasure,
    RootedJunctionTree,
    build_rooted_junction_tree,
    compile_diagram,
)
from desdeo.problem.graphical.adapters.pyagrum import to_pyagrum_influence_diagram
from desdeo.tools.gurobipy_solver_interfaces import GurobipySolver

from .conftest import delete_json, get_json, login, post_json

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


def _persist_graphical_metadata_with_index_map(
    session: Session, problem_id: int, network: DiscreteNetwork, annotation: DecisionAnnotation
) -> GraphicalModelMetaData:
    """Like `_persist_graphical_metadata`, but also compiles and persists `index_map_json`.

    `/policy` (Phase E) needs a real `IndexMap` -- unlike `/structure`/`/whatif`, which only ever
    read `network_json`/`annotation_json`.
    """
    _, index_map = compile_diagram(network, annotation)
    metadata_db = ProblemMetaDataDB(problem_id=problem_id)
    session.add(metadata_db)
    session.commit()
    session.refresh(metadata_db)

    graphical_metadata = GraphicalModelMetaData(
        metadata_id=metadata_db.id,
        network_json=network.model_dump_json(),
        annotation_json=annotation.model_dump_json(),
        index_map_json=index_map.model_dump_json(),
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
    # Phase E: MEU with its standard deviation. mean=8.0 (above); variance = E[U^2] - mean^2 =
    # (0.0^2 * 0.2 + 10.0^2 * 0.8) - 8.0^2 = 80.0 - 64.0 = 16.0, std = 4.0 -- hand-computed, not
    # just "the endpoint returns some number".
    assert data["utility_std_devs"]["f1"] == pytest.approx(4.0)


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


def test_whatif_utility_std_dev_with_decision_matches_pyagrum_meu_variance(client: TestClient, session_and_user: dict):
    """Phase E, decision-diagram branch: `utility_std_devs` == `sqrt(MEU()["variance"])`.

    `D` is deliberately *blind* (`information_set=[]`, unlike `_decision_annotation`'s `D`, which
    observes `A` and would drive the achieved utility to a single deterministic cell -- zero
    variance, degenerate for this test). Blind, `D`'s single fixed optimal choice still lands on
    different utility cells depending on `A`'s own realization, so the achieved-utility
    distribution genuinely has spread: `D="s0"` (`E[U]=0.3*10+0.7*0=3.0`, beating `D="s1"`'s
    `0.3*0+0.7*3=2.1`), so the achieved outcome is `10.0` w.p. `0.3` or `0.0` w.p. `0.7` --
    `mean=3.0`, `variance=E[U^2]-mean^2=0.3*100-9=21.0`.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _decision_network()
    annotation = DecisionAnnotation(
        decisions=[DecisionNode(symbol="D", states=["s0", "s1"], information_set=[])],
        objectives=[
            ObjectiveNode(
                symbol="fmatch",
                parents=["A", "D"],
                utility=Potential(variables=["A", "D"], shape=[2, 2], values=[[10.0, 0.0], [0.0, 3.0]]),
                maximize=True,
            )
        ],
    )
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {}}
    response = post_json(client, "/graphical/whatif", payload, access_token)
    assert response.status_code == 200
    data = response.json()

    diagram = to_pyagrum_influence_diagram(network, annotation)
    inference = gum.ShaferShenoyLIMIDInference(diagram)
    inference.makeInference()
    meu = inference.MEU()

    assert data["utilities"]["fmatch"] == pytest.approx(3.0)
    assert data["utility_std_devs"]["fmatch"] == pytest.approx(21.0**0.5)
    assert data["utilities"]["fmatch"] == pytest.approx(meu["mean"])
    assert data["utility_std_devs"]["fmatch"] == pytest.approx(meu["variance"] ** 0.5)


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


# --- POST /graphical/{problem_id}/policy (Phase E) ------------------------------------------------
#
# Distinct from `/whatif`'s own `policy` field (a live pyAgrum engine's evidence-conditioned
# posterior) -- this decodes a *specific solved `Problem`*'s `z` values via `IndexMap`/
# `read_policy_from_solution`, with no pyAgrum call at all. Needs an actual solved MILP, so these
# tests (unlike the rest of this module) are marked `gurobipy` individually.


def _solved_decision_problem(network: DiscreteNetwork, annotation: DecisionAnnotation, objective_symbol: str):
    """Compile + solve `(network, annotation)`, returning `(index_map, optimal_variables)`."""
    problem, index_map = compile_diagram(network, annotation)
    result = GurobipySolver(problem).solve(objective_symbol)
    assert result.success, result.message
    return index_map, result.optimal_variables


@pytest.mark.gurobipy
def test_policy_decodes_solved_z_given_evidence(client: TestClient, session_and_user: dict):
    """`/policy` reports `D`'s solved choice for the information state `evidence` picks out.

    `_decision_annotation`'s `D` observes `A` and is rewarded for matching it, so the solved
    policy is `D="s0"` when `A="lo"` and `D="s1"` when `A="hi"` -- checked for both, from the
    *same* solved `optimal_variables`, confirming this is a genuine per-information-state
    read-back and not a fluke on one evidence value.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _decision_network()
    annotation = _decision_annotation()
    _persist_graphical_metadata_with_index_map(session, problem.id, network, annotation)

    _, optimal_variables = _solved_decision_problem(network, annotation, "fmatch")

    for a_state, expected_choice in (("lo", "s0"), ("hi", "s1")):
        payload = {
            "variable_values": optimal_variables,
            "evidence": {"A": a_state},
        }
        response = post_json(client, f"/graphical/{problem.id}/policy", payload, access_token)
        assert response.status_code == 200, response.text
        assert response.json()["policy"] == {"D": expected_choice}


@pytest.mark.gurobipy
def test_policy_omits_decision_not_pinned_by_evidence(client: TestClient, session_and_user: dict):
    """No evidence on `A` -> `D`'s effective information set isn't pinned -> `D` is absent.

    Mirrors `/whatif`'s own `_read_policy` "omit, don't guess" semantics for the live-engine case
    (`test_whatif_policy_omits_decision_not_fully_determined_by_evidence`).
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _decision_network()
    annotation = _decision_annotation()
    _persist_graphical_metadata_with_index_map(session, problem.id, network, annotation)

    _, optimal_variables = _solved_decision_problem(network, annotation, "fmatch")

    payload = {"variable_values": optimal_variables, "evidence": {}}
    response = post_json(client, f"/graphical/{problem.id}/policy", payload, access_token)
    assert response.status_code == 200, response.text
    assert response.json()["policy"] == {}


def _two_decision_no_forgetting_network() -> DiscreteNetwork:
    return DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[2], values=[0.35, 0.65])},
    )


def _two_decision_no_forgetting_annotation() -> DecisionAnnotation:
    """D1 observes A; D2, with no_forgetting, implicitly also observes D1 (and hence A).

    Same fixture shape as `tests/test_graphical_compile.py::test_two_decisions_with_no_forgetting`
    -- D2's *effective* information set (`no_forgetting`-expanded) includes D1, not just chance
    nodes, which is exactly the case `/policy`'s decoding needs to resolve in order.
    """
    return DecisionAnnotation(
        decisions=[
            DecisionNode(symbol="D1", states=["s0", "s1"], information_set=["A"]),
            DecisionNode(symbol="D2", states=["t0", "t1"]),
        ],
        objectives=[
            ObjectiveNode(
                symbol="fchain",
                parents=["A", "D1", "D2"],
                utility=Potential(
                    variables=["A", "D1", "D2"],
                    shape=[2, 2, 2],
                    # Rewarded for D1 matching A, and D2 matching D1.
                    values=[[[10.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 10.0]]],
                ),
                maximize=True,
            )
        ],
        decision_order=["D1", "D2"],
        no_forgetting=True,
    )


@pytest.mark.gurobipy
def test_policy_resolves_multi_decision_no_forgetting_against_whatif(client: TestClient, session_and_user: dict):
    """`/policy` reports BOTH decisions for a `no_forgetting` diagram, matching `/whatif`'s own.

    Regression test for a review-pass finding: `_decode_solved_policy` used to only include a
    decision when every member of its *effective* information set was present in the (validated
    chance-node-only) `evidence` -- but under `no_forgetting`, `D2`'s effective information set
    includes `D1` (a decision, not a chance node), which chance-only `evidence` could never
    satisfy on its own, so `D2` was unconditionally omitted with no error/warning even though only
    evidence pinning the chance node `A` is needed to fully determine the whole policy. Cross-
    checked directly against `/whatif`'s own live-engine `_read_policy`, which already resolves
    this correctly for the identical evidence.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _two_decision_no_forgetting_network()
    annotation = _two_decision_no_forgetting_annotation()
    _persist_graphical_metadata_with_index_map(session, problem.id, network, annotation)

    _, optimal_variables = _solved_decision_problem(network, annotation, "fchain")

    evidence = {"A": "lo"}
    policy_response = post_json(
        client,
        f"/graphical/{problem.id}/policy",
        {"variable_values": optimal_variables, "evidence": evidence},
        access_token,
    )
    assert policy_response.status_code == 200, policy_response.text
    policy = policy_response.json()["policy"]
    assert set(policy) == {"D1", "D2"}

    whatif_response = post_json(
        client, "/graphical/whatif", {"problem_id": problem.id, "evidence": evidence}, access_token
    )
    assert whatif_response.status_code == 200, whatif_response.text
    assert whatif_response.json()["policy"] == policy


@pytest.mark.gurobipy
def test_policy_via_solution_id(client: TestClient, session_and_user: dict):
    """`solution_id` (a `UserSavedSolutionDB` row) is an equally valid source.

    Per the plan's own "or a reference to an already-stored solution" wording -- same decoded
    answer either way.
    """
    session: Session = session_and_user["session"]
    user = session_and_user["user"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _decision_network()
    annotation = _decision_annotation()
    _persist_graphical_metadata_with_index_map(session, problem.id, network, annotation)

    _, optimal_variables = _solved_decision_problem(network, annotation, "fmatch")

    saved = UserSavedSolutionDB(
        name="solved",
        objective_values={"fmatch": 10.0},
        variable_values=optimal_variables,
        solution_index=0,
        user_id=user.id,
        problem_id=problem.id,
    )
    session.add(saved)
    session.commit()
    session.refresh(saved)

    payload = {"solution_id": saved.id, "evidence": {"A": "lo"}}
    response = post_json(client, f"/graphical/{problem.id}/policy", payload, access_token)
    assert response.status_code == 200, response.text
    assert response.json()["policy"] == {"D": "s0"}


def test_policy_requires_exactly_one_of_variable_values_or_solution_id(client: TestClient, session_and_user: dict):
    """400 when neither, or both, of `variable_values`/`solution_id` are given."""
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _decision_network()
    annotation = _decision_annotation()
    _persist_graphical_metadata_with_index_map(session, problem.id, network, annotation)

    neither = post_json(client, f"/graphical/{problem.id}/policy", {"evidence": {}}, access_token)
    assert neither.status_code == 400

    both = post_json(
        client,
        f"/graphical/{problem.id}/policy",
        {"variable_values": {"z_D": [1.0, 0.0]}, "solution_id": 1, "evidence": {}},
        access_token,
    )
    assert both.status_code == 400


def test_policy_404_when_no_index_map(client: TestClient, session_and_user: dict):
    """A `GraphicalModelMetaData` row persisted without ever running `compile_diagram` 404s.

    `_persist_graphical_metadata` deliberately does this for the `/structure`/`/whatif`-only
    tests, so `index_map_json` is `None` -- `/policy` needs it and 404s rather than crashing.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _chain_network()
    annotation = _no_decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"variable_values": {}, "evidence": {}}
    response = post_json(client, f"/graphical/{problem.id}/policy", payload, access_token)
    assert response.status_code == 404


@pytest.mark.gurobipy
def test_policy_missing_z_symbol_entry_is_400(client: TestClient, session_and_user: dict):
    """A `variable_values` missing a decision's own `z_symbol` entry is a clear 400."""
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _decision_network()
    annotation = _decision_annotation()
    _persist_graphical_metadata_with_index_map(session, problem.id, network, annotation)

    payload = {"variable_values": {"not_the_z_symbol": [1.0, 0.0]}, "evidence": {}}
    response = post_json(client, f"/graphical/{problem.id}/policy", payload, access_token)
    assert response.status_code == 400


def test_policy_wrong_length_z_symbol_is_400_not_500(client: TestClient, session_and_user: dict):
    """A present but wrong-length `z_symbol` entry is a clean 400, not an uncaught `KeyError`/500.

    Regression test for a review-pass finding: `read_policy_from_solution` used to infer
    `n_info_states` from `len(z_values) // n_states` instead of validating it against
    `prod(information_cardinalities) * len(states)` -- a wrong-length `z` (e.g. a stale saved
    solution from before a recompile changed a cardinality) used to silently produce a policy dict
    missing some info-state keys, which then raised an uncaught `KeyError` (-> 500) the first time
    `_decode_solved_policy` looked up a missing information state, instead of failing loudly and
    cleanly at read-back time. `D`'s effective information set is `["A"]` (cardinality 2) and `D`
    has 2 states, so a genuine `z_D` has length 4 -- this one has 3.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _decision_network()
    annotation = _decision_annotation()
    _persist_graphical_metadata_with_index_map(session, problem.id, network, annotation)

    payload = {"variable_values": {"z_D": [1.0, 0.0, 1.0]}, "evidence": {}}
    response = post_json(client, f"/graphical/{problem.id}/policy", payload, access_token)
    assert response.status_code == 400, response.text


@pytest.mark.gurobipy
def test_policy_accepts_scalar_risk_measure_eta_without_422(client: TestClient, session_and_user: dict):
    """`variable_values` accepts a real solved-solution dict with a scalar RiskMeasure eta.

    A Phase 6 `RiskMeasure`'s scalar `eta` entry (a bare `float`, not `list[float]`) alongside the
    decision's own list-valued `z`. Regression test for a review-pass finding: `PolicyRequest.variable_values` was typed
    `dict[str, list[float]] | None`, stricter than `solution_id`'s equivalent path
    (`UserSavedSolutionDB.variable_values` already tolerates a bare `float` at runtime -- see
    Phase E's own findings) -- posting a real `optimal_variables` dict for a RiskMeasure-bearing
    diagram through this field used to 422.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="W", labels=["lose", "win"])],
        arcs=[],
        potentials={"W": Potential(variables=["W"], shape=[2], values=[0.5, 0.5])},
    )
    annotation = DecisionAnnotation(
        decisions=[DecisionNode(symbol="D", states=["safe", "gamble"])],
        objectives=[
            ObjectiveNode(
                symbol="payoff",
                parents=["D", "W"],
                utility=Potential(variables=["D", "W"], shape=[2, 2], values=[[5.0, 5.0], [0.0, 12.0]]),
                maximize=True,
            )
        ],
        risk_measures=[RiskMeasure(symbol="cvar_payoff", target="payoff", alpha=0.5)],
    )
    _persist_graphical_metadata_with_index_map(session, problem.id, network, annotation)

    compiled_problem, index_map = compile_diagram(network, annotation)
    result = GurobipySolver(compiled_problem).solve("cvar_payoff")
    assert result.success, result.message

    eta_symbol = index_map.risk_measures["cvar_payoff"].eta_symbol
    # Confirms the fixture actually exercises the fix: a genuinely scalar (non-list) entry.
    assert isinstance(result.optimal_variables[eta_symbol], float)

    payload = {"variable_values": result.optimal_variables, "evidence": {}}
    response = post_json(client, f"/graphical/{problem.id}/policy", payload, access_token)
    assert response.status_code != 422, response.text
    assert response.status_code == 200, response.text


# --- Named evidence sets (Phase E) -----------------------------------------------------------------


def test_evidence_set_create_list_get_roundtrip(client: TestClient, session_and_user: dict):
    """Create, list (for the problem), and fetch-by-id (the "shareable by URL" path) all agree."""
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _chain_network()
    annotation = _no_decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    create_response = post_json(
        client,
        f"/graphical/{problem.id}/evidence-sets",
        {"name": "baseline A=hi", "evidence": {"A": "hi"}},
        access_token,
    )
    assert create_response.status_code == 200, create_response.text
    created = create_response.json()
    assert created["name"] == "baseline A=hi"
    assert created["evidence"] == {"A": "hi"}
    assert created["problem_id"] == problem.id

    list_response = get_json(client, f"/graphical/{problem.id}/evidence-sets", access_token)
    assert list_response.status_code == 200
    listed = list_response.json()
    assert len(listed) == 1
    assert listed[0]["id"] == created["id"]

    get_response = get_json(client, f"/graphical/evidence-sets/{created['id']}", access_token)
    assert get_response.status_code == 200
    assert get_response.json() == created


def test_evidence_set_create_rejects_non_chance_evidence(client: TestClient, session_and_user: dict):
    """A reference to a node that isn't a chance node is rejected eagerly, at save time."""
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _chain_network()
    annotation = _no_decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    response = post_json(
        client,
        f"/graphical/{problem.id}/evidence-sets",
        {"name": "bad", "evidence": {"ZZZ": "hi"}},
        access_token,
    )
    assert response.status_code == 400


def test_evidence_set_create_rejects_invalid_state_label(client: TestClient, session_and_user: dict):
    """A real chance-node name with a typo'd state label is rejected eagerly, at save time.

    Regression test for a review-pass finding: `_validate_chance_evidence` used to check node
    names but never state labels, so a typo'd label used to save successfully (200) and only fail
    later, opaquely, the first time anything (`/whatif`/`/sensitivity`/`/voi`) tried to actually
    use the set.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _chain_network()
    annotation = _no_decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    response = post_json(
        client,
        f"/graphical/{problem.id}/evidence-sets",
        {"name": "bad label", "evidence": {"A": "bogus-label"}},
        access_token,
    )
    assert response.status_code == 400, response.text


def test_evidence_set_get_visible_to_other_user_with_problem_visibility(client: TestClient, session_and_user: dict):
    """Shareable by URL: any user who can already see the problem can fetch the set by id.

    Not only the user who saved it -- analyst/admin bypass ownership
    (`fetch_problem_with_role_check`).
    """
    session: Session = session_and_user["session"]
    access_token = login(client)  # default "analyst" user, bypasses ownership

    problem = session.exec(select(ProblemDB)).first()
    network = _chain_network()
    annotation = _no_decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    admin = User(username="admin_user", password_hash=get_password_hash("admin_pw"), role=UserRole.admin, group="test")
    session.add(admin)
    session.commit()
    admin_token = login(client, username="admin_user", password="admin_pw")  # noqa: S106

    create_response = post_json(
        client, f"/graphical/{problem.id}/evidence-sets", {"name": "shared", "evidence": {}}, access_token
    )
    evidence_set_id = create_response.json()["id"]

    get_response = get_json(client, f"/graphical/evidence-sets/{evidence_set_id}", admin_token)
    assert get_response.status_code == 200
    assert get_response.json()["evidence"] == {}


def test_evidence_set_delete_owner_only(client: TestClient, session_and_user: dict):
    """Delete is restricted to the saving user (or an admin/analyst) -- reading/listing are not.

    `dm_user4` here has no visibility into the problem at all (not its owner, not in any group
    attached to it), so `fetch_problem_with_role_check` returns `None` for them and this only
    exercises the "can't even see it" -> 404 branch of `delete_evidence_set`, *not* the "can see it
    but isn't the owner" -> 403 branch -- those are genuinely different code paths (see
    `test_evidence_set_delete_visible_non_owner_is_403` below for the latter, added by a review
    pass that caught this test's previous `in (403, 404)` assertion papering over not knowing
    which branch it exercised).
    """
    session: Session = session_and_user["session"]
    access_token = login(client)  # default "analyst" user -- bypasses ownership on reads/deletes

    problem = session.exec(select(ProblemDB)).first()
    network = _chain_network()
    annotation = _no_decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    dm_user = User(username="dm_user4", password_hash=get_password_hash("dm_pw"), role=UserRole.dm, group="test")
    session.add(dm_user)
    session.commit()
    dm_token = login(client, username="dm_user4", password="dm_pw")  # noqa: S106

    create_response = post_json(
        client, f"/graphical/{problem.id}/evidence-sets", {"name": "mine", "evidence": {}}, access_token
    )
    evidence_set_id = create_response.json()["id"]

    # A user without visibility into the problem can't see it well enough to delete it either --
    # a clean 404 (not a 403, which would leak "this exists" to someone who can't even see the
    # underlying problem).
    forbidden = delete_json(client, f"/graphical/evidence-sets/{evidence_set_id}", dm_token)
    assert forbidden.status_code == 404

    ok = delete_json(client, f"/graphical/evidence-sets/{evidence_set_id}", access_token)
    assert ok.status_code == 200

    gone = get_json(client, f"/graphical/evidence-sets/{evidence_set_id}", access_token)
    assert gone.status_code == 404


def test_evidence_set_delete_visible_non_owner_is_403(client: TestClient, session_and_user: dict):
    """A user who CAN see the problem (via a shared `Group`) but didn't save the set gets a 403.

    Distinct from `test_evidence_set_delete_owner_only`'s "no visibility at all" -> 404 case above
    -- uses the same `Group`-membership visibility path `fetch_problem_with_role_check`
    (`routers/utils.py`) checks and `test_models.py::test_group` exercises directly to give
    `dm_user5` real visibility into the problem without making them its owner.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)  # default "analyst" user -- owns the problem

    problem = session.exec(select(ProblemDB)).first()
    network = _chain_network()
    annotation = _no_decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    dm_user = User(username="dm_user5", password_hash=get_password_hash("dm_pw"), role=UserRole.dm, group="test")
    session.add(dm_user)
    session.commit()
    session.refresh(dm_user)
    dm_token = login(client, username="dm_user5", password="dm_pw")  # noqa: S106

    group = Group(name="shared", owner_id=session_and_user["user"].id, user_ids=[dm_user.id], problem_id=problem.id)
    session.add(group)
    session.commit()

    create_response = post_json(
        client, f"/graphical/{problem.id}/evidence-sets", {"name": "mine", "evidence": {}}, access_token
    )
    evidence_set_id = create_response.json()["id"]

    # dm_user5 can see the problem (group membership), just isn't the one who saved this set.
    forbidden = delete_json(client, f"/graphical/evidence-sets/{evidence_set_id}", dm_token)
    assert forbidden.status_code == 403

    # And reading/listing (unlike deleting) are not owner-restricted -- still visible.
    still_visible = get_json(client, f"/graphical/evidence-sets/{evidence_set_id}", dm_token)
    assert still_visible.status_code == 200


def test_evidence_set_404_for_nonexistent_id(client: TestClient, session_and_user: dict):
    """A nonexistent id is a plain 404, not a crash."""
    access_token = login(client)
    response = get_json(client, "/graphical/evidence-sets/999999", access_token)
    assert response.status_code == 404


# --- POST /graphical/sensitivity (Phase E) ---------------------------------------------------------


def _sensitivity_fixture_network() -> DiscreteNetwork:
    """Three independent binary chance nodes: A, B, C -- no arcs at all.

    An objective depending only on A (not B, not C) makes the expected sensitivity result
    hand-computable: sweeping A moves the objective, sweeping B or C (independent of A and not
    themselves parents of the objective) does not, at all -- a clean oracle for "sorted by
    impact, biggest bar first" and for a target-value swing of exactly zero.
    """
    return DiscreteNetwork(
        variables=[
            DiscreteVariable(name="A", labels=["lo", "hi"]),
            DiscreteVariable(name="B", labels=["lo", "hi"]),
            DiscreteVariable(name="C", labels=["lo", "hi"]),
        ],
        arcs=[],
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=[0.5, 0.5]),
            "B": Potential(variables=["B"], shape=[2], values=[0.3, 0.7]),
            "C": Potential(variables=["C"], shape=[2], values=[0.5, 0.5]),
        },
    )


def _sensitivity_fixture_annotation() -> DecisionAnnotation:
    """`f1` depends only on `A` (`utility[A] = [0.0, 100.0]`) -- `B`/`C` have zero true swing."""
    return DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="f1", parents=["A"], utility=Potential(variables=["A"], shape=[2], values=[0.0, 100.0])
            )
        ]
    )


def test_sensitivity_objective_target_sorted_by_swing(client: TestClient, session_and_user: dict):
    """A drives the objective, B/C don't -- tornado results sorted descending by swing agree."""
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _sensitivity_fixture_network()
    annotation = _sensitivity_fixture_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {}, "target": "f1"}
    response = post_json(client, "/graphical/sensitivity", payload, access_token)
    assert response.status_code == 200, response.text
    data = response.json()

    # No evidence: E[f1] = 0.5 * 0.0 + 0.5 * 100.0 = 50.0.
    assert data["base_value"] == pytest.approx(50.0)

    results_by_node = {r["node"]: r for r in data["results"]}
    assert set(results_by_node) == {"A", "B", "C"}

    # A: pinning A="lo" makes E[f1]=0.0, A="hi" makes E[f1]=100.0 -- a genuine swing of 100.0.
    assert results_by_node["A"]["low"] == pytest.approx(0.0)
    assert results_by_node["A"]["high"] == pytest.approx(100.0)
    assert results_by_node["A"]["swing"] == pytest.approx(100.0)

    # B, C: independent of A and not objective parents -- pinning either leaves E[f1] unchanged.
    assert results_by_node["B"]["swing"] == pytest.approx(0.0)
    assert results_by_node["C"]["swing"] == pytest.approx(0.0)

    # Sorted descending by swing -- A (the real driver) comes first.
    assert data["results"][0]["node"] == "A"


def test_sensitivity_node_target(client: TestClient, session_and_user: dict):
    """`target` can be a chance node's own marginal, not just an objective."""
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _sensitivity_fixture_network()
    annotation = _sensitivity_fixture_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {}, "target": "A", "target_state": "hi"}
    response = post_json(client, "/graphical/sensitivity", payload, access_token)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["base_value"] == pytest.approx(0.5)  # P(A=hi) unconditioned
    results_by_node = {r["node"]: r for r in data["results"]}
    assert set(results_by_node) == {"B", "C"}  # A itself is excluded (it's the target)
    # A is independent of B and C -- P(A=hi) doesn't move no matter what B/C are pinned to.
    assert results_by_node["B"]["swing"] == pytest.approx(0.0)
    assert results_by_node["C"]["swing"] == pytest.approx(0.0)


def test_sensitivity_node_target_requires_target_state(client: TestClient, session_and_user: dict):
    """A node `target` without `target_state` is a clear 400, not a crash."""
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _sensitivity_fixture_network()
    annotation = _sensitivity_fixture_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {}, "target": "A"}
    response = post_json(client, "/graphical/sensitivity", payload, access_token)
    assert response.status_code == 400


def test_sensitivity_excludes_already_evidenced_node_from_sweep(client: TestClient, session_and_user: dict):
    """A node already pinned by `evidence` isn't swept -- nothing left to sweep for it."""
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _sensitivity_fixture_network()
    annotation = _sensitivity_fixture_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {"B": "lo"}, "target": "f1"}
    response = post_json(client, "/graphical/sensitivity", payload, access_token)
    assert response.status_code == 200
    swept_nodes = {r["node"] for r in response.json()["results"]}
    assert swept_nodes == {"A", "C"}


def test_sensitivity_bad_evidence_state_label_is_400_not_500(client: TestClient, session_and_user: dict):
    """A bad state label in `evidence` itself (not a bad node name) is a clean 400.

    Regression test for a review-pass finding: the sweep's own `finally` block restores the
    engine's evidence via `entry.engine.setEvidence(request.evidence)` -- if `request.evidence`
    itself is what pyAgrum just rejected inside the `try` (about to raise a documented 400), that
    same restore call used to raise again, and a raise inside `finally` replaces the in-flight
    `HTTPException`, turning the 400 into an uncaught 500 (confirmed empirically pre-fix). Also
    exercises `_validate_chance_evidence`'s new label check (a separate review-pass fix), which
    now rejects this evidence before the engine is ever touched -- either fix alone is enough to
    make this test pass, so it's a genuine regression test for both.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _sensitivity_fixture_network()
    annotation = _sensitivity_fixture_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {"A": "bogus-label"}, "target": "f1"}
    response = post_json(client, "/graphical/sensitivity", payload, access_token)
    assert response.status_code == 400, response.text


# --- POST /graphical/voi (Phase E) -----------------------------------------------------------------


def test_voi_hand_computed(client: TestClient, session_and_user: dict):
    """VoI(A) for a blind decision that would benefit from observing A first -- hand-computed.

    `D` is blind (`information_set=[]`); `utility[A,D] = [[10,0],[0,3]]`. Blind, `D`'s single best
    choice is `D="s0"` (`E=0.4*10+0.6*0=4.0` vs `D="s1"`'s `0.4*0+0.6*3=1.8`), so
    `base_value=4.0`. If `D` could observe `A` first: `A="lo"` (prob 0.4) -> best is `D="s0"`
    (payoff 10); `A="hi"` (prob 0.6) -> best is `D="s1"` (payoff 3). Weighted:
    `expected_value_with_observation = 0.4*10 + 0.6*3 = 5.8`. `VoI = 5.8 - 4.0 = 1.8`.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = DiscreteNetwork(
        variables=[DiscreteVariable(name="A", labels=["lo", "hi"])],
        arcs=[],
        potentials={"A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6])},
    )
    annotation = DecisionAnnotation(
        decisions=[DecisionNode(symbol="D", states=["s0", "s1"], information_set=[])],
        objectives=[
            ObjectiveNode(
                symbol="fmatch",
                parents=["A", "D"],
                utility=Potential(variables=["A", "D"], shape=[2, 2], values=[[10.0, 0.0], [0.0, 3.0]]),
                maximize=True,
            )
        ],
    )
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {}, "node": "A", "target": "fmatch"}
    response = post_json(client, "/graphical/voi", payload, access_token)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["base_value"] == pytest.approx(4.0)
    assert data["expected_value_with_observation"] == pytest.approx(5.8)
    assert data["value_of_information"] == pytest.approx(1.8)
    assert data["marginal"] == pytest.approx({"lo": 0.4, "hi": 0.6})


def test_voi_no_decision_is_400(client: TestClient, session_and_user: dict):
    """VoI is only meaningful for a diagram with at least one decision.

    A clear 4xx, not a silent 0.0 or a crash (see `value_of_information`'s own docstring for why
    4xx was chosen).
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _chain_network()
    annotation = _no_decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {}, "node": "A", "target": "f1"}
    response = post_json(client, "/graphical/voi", payload, access_token)
    assert response.status_code == 400


def test_voi_node_already_evidenced_is_400(client: TestClient, session_and_user: dict):
    """A node already pinned by `evidence` has nothing left to observe -- 400."""
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _decision_network()
    annotation = _decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {"A": "lo"}, "node": "A", "target": "fmatch"}
    response = post_json(client, "/graphical/voi", payload, access_token)
    assert response.status_code == 400


def test_voi_bad_evidence_state_label_is_400_not_500(client: TestClient, session_and_user: dict):
    """A bad state label in `evidence` (for a node other than `node` itself) is a clean 400.

    Mirrors `test_sensitivity_bad_evidence_state_label_is_400_not_500`'s identical regression --
    see that test's own docstring for the mechanism (a bad `finally`-block restore could otherwise
    replace an in-flight 400 with an uncaught 500). Needs a second chance node distinct from
    `node`/`target`'s own information set, since `node in evidence` is itself rejected earlier
    (`test_voi_node_already_evidenced_is_400`) -- `B` here is otherwise unused by the diagram.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = DiscreteNetwork(
        variables=[
            DiscreteVariable(name="A", labels=["lo", "hi"]),
            DiscreteVariable(name="B", labels=["lo", "hi"]),
        ],
        arcs=[],
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=[0.3, 0.7]),
            "B": Potential(variables=["B"], shape=[2], values=[0.5, 0.5]),
        },
    )
    annotation = _decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {"B": "bogus-label"}, "node": "A", "target": "fmatch"}
    response = post_json(client, "/graphical/voi", payload, access_token)
    assert response.status_code == 400, response.text


def test_voi_target_not_an_objective_is_400(client: TestClient, session_and_user: dict):
    """`target` must name an `ObjectiveNode.symbol` -- anything else is a 400."""
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _decision_network()
    annotation = _decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {}, "node": "A", "target": "not_an_objective"}
    response = post_json(client, "/graphical/voi", payload, access_token)
    assert response.status_code == 400


# --- POST /graphical/demo/irrigation, POST /graphical/{problem_id}/solve (Phase F) ---------------
#
# "Real backend integration for the CUMULUS demo" -- the webui's dev-only network-structure demo
# dialog now compiles/solves this canned diagram for real instead of binding to a static TypeScript
# fixture. `/solve` needs an actually-solved MILP, so those tests are marked `gurobipy`
# individually, matching `/policy`'s own convention above.


def test_compile_demo_irrigation_persists_expected_structure(client: TestClient, session_and_user: dict):
    """The demo endpoint persists a compiled `Problem` whose structure matches `demo.py`'s model."""
    access_token = login(client)

    response = post_json(client, "/graphical/demo/irrigation", {}, access_token)
    assert response.status_code == 200, response.text
    problem_id = response.json()["problem_id"]

    structure = get_json(client, f"/graphical/{problem_id}/structure", access_token)
    assert structure.status_code == 200
    data = structure.json()
    nodes_by_id = {n["id"]: n for n in data["nodes"]}
    assert nodes_by_id["weather"]["kind"] == "chance"
    assert nodes_by_id["soil_moisture"]["kind"] == "chance"
    assert nodes_by_id["irrigate"]["kind"] == "decision"
    assert nodes_by_id["irrigate"]["states"] == ["none", "light", "heavy"]
    assert nodes_by_id["util_profit"]["kind"] == "objective"
    for choice in ("none", "light", "heavy"):
        assert nodes_by_id[f"crop_yield_{choice}"]["kind"] == "chance"


def test_compile_demo_irrigation_guest_rejected(client: TestClient, session_and_user: dict):
    """Guest users may not compile the demo model either -- same rule as `/compile`."""
    session: Session = session_and_user["session"]
    guest = User(username="guest_demo", password_hash=get_password_hash("guest_pw"), role=UserRole.guest, group="test")
    session.add(guest)
    session.commit()

    access_token = login(client, username="guest_demo", password="guest_pw")  # noqa: S106
    response = post_json(client, "/graphical/demo/irrigation", {}, access_token)
    assert response.status_code == 401


@pytest.mark.gurobipy
def test_solve_profit_picks_heavy_irrigation_and_feeds_policy(client: TestClient, session_and_user: dict):
    """Solving `profit` picks `irrigate == 'heavy'`; the readback feeds `/policy`.

    `heavy` is the hand-verified expected-profit-maximizing choice for `demo.py`'s own numbers
    (`tests/test_graphical_demo.py::test_profit_solve_matches_hand_computed_expectation`) -- this
    test additionally cross-checks the whole demo path end to end: compile via the API, solve via
    the API, and use the returned `variable_values` as `/policy`'s own `variable_values` -- exactly
    how the webui's `NetworkPanel` (`policySource={{ variableValues: ... }}`) is meant to use this
    endpoint's output. Also cross-checked against `/whatif`'s own live-engine policy: with exactly
    one `ObjectiveNode`, both should agree -- unlike a would-be two-objective version of this
    diagram, which `/whatif` can't even answer at all (see `demo.py`'s own module docstring).
    """
    access_token = login(client)

    compile_response = post_json(client, "/graphical/demo/irrigation", {}, access_token)
    assert compile_response.status_code == 200, compile_response.text
    problem_id = compile_response.json()["problem_id"]

    solve_response = post_json(client, f"/graphical/{problem_id}/solve", {"objective_symbol": "profit"}, access_token)
    assert solve_response.status_code == 200, solve_response.text
    variable_values = solve_response.json()["variable_values"]

    policy_response = post_json(
        client,
        f"/graphical/{problem_id}/policy",
        {"variable_values": variable_values, "evidence": {}},
        access_token,
    )
    assert policy_response.status_code == 200, policy_response.text
    assert policy_response.json()["policy"] == {"irrigate": "heavy"}

    whatif_response = post_json(client, "/graphical/whatif", {"problem_id": problem_id, "evidence": {}}, access_token)
    assert whatif_response.status_code == 200, whatif_response.text
    assert whatif_response.json()["policy"] == {"irrigate": "heavy"}


@pytest.mark.gurobipy
def test_solve_unknown_objective_symbol_is_400(client: TestClient, session_and_user: dict):
    """`objective_symbol` must be one of the compiled problem's own objectives."""
    access_token = login(client)

    compile_response = post_json(client, "/graphical/demo/irrigation", {}, access_token)
    assert compile_response.status_code == 200, compile_response.text
    problem_id = compile_response.json()["problem_id"]

    response = post_json(
        client, f"/graphical/{problem_id}/solve", {"objective_symbol": "not_a_real_objective"}, access_token
    )
    assert response.status_code == 400


def test_solve_problem_not_found_is_404(client: TestClient, session_and_user: dict):
    """A nonexistent problem id is a 404, not a 500 from a failed `ProblemDB` fetch."""
    access_token = login(client)
    response = post_json(client, "/graphical/999999/solve", {"objective_symbol": "profit"}, access_token)
    assert response.status_code == 404


@pytest.mark.gurobipy
def test_solve_visible_non_owner_succeeds(client: TestClient, session_and_user: dict):
    """A user who can see the problem only via a shared `Group` (not its owner) can still solve it.

    Same `Group`-membership visibility pattern `test_evidence_set_delete_visible_non_owner_is_403`
    uses -- `/solve` has no owner-only action of its own (unlike deleting an evidence set), so this
    confirms visibility (not ownership) is what actually gates it, matching every sibling
    `/graphical/{problem_id}/*` endpoint's `fetch_problem_with_role_check` semantics.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)  # default "analyst" user -- owns the problem once compiled

    compile_response = post_json(client, "/graphical/demo/irrigation", {}, access_token)
    assert compile_response.status_code == 200, compile_response.text
    problem_id = compile_response.json()["problem_id"]

    dm_user = User(username="dm_user6", password_hash=get_password_hash("dm_pw"), role=UserRole.dm, group="test")
    session.add(dm_user)
    session.commit()
    session.refresh(dm_user)
    dm_token = login(client, username="dm_user6", password="dm_pw")  # noqa: S106

    group = Group(
        name="shared-solve", owner_id=session_and_user["user"].id, user_ids=[dm_user.id], problem_id=problem_id
    )
    session.add(group)
    session.commit()

    response = post_json(client, f"/graphical/{problem_id}/solve", {"objective_symbol": "profit"}, dm_token)
    assert response.status_code == 200, response.text


def test_solve_non_graphical_problem_is_404(client: TestClient, session_and_user: dict):
    """`/solve` rejects an ordinary (non-graphical) `ProblemDB` -- same 404 every sibling endpoint gives.

    A review pass caught `/solve` skipping the `_fetch_graphical_metadata` check every other
    `/graphical/{problem_id}/*` endpoint runs first, despite its own docstring claiming otherwise --
    this doubled it as an accidental general-purpose solve API for arbitrary visible problems. Uses
    "The river pollution problem" -- the same plain, non-graphical fixture
    `test_structure_404_when_no_graphical_metadata` already uses for the identical check on
    `/structure`.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)
    problem = session.exec(select(ProblemDB).where(ProblemDB.name == "The river pollution problem")).first()

    response = post_json(client, f"/graphical/{problem.id}/solve", {"objective_symbol": "f_1"}, access_token)
    assert response.status_code == 404


def test_demo_irrigation_idempotent_same_user_returns_same_problem_id(client: TestClient, session_and_user: dict):
    """Two `/demo/irrigation` calls from the same user reuse the same compiled `problem_id`.

    A review pass caught the original endpoint persisting a brand-new ~127 KB `Problem` + metadata
    on *every* call, with no cleanup story -- every browser tab, every reload without a live
    `sessionStorage` cache, and every "Recompile demo" click accumulated a fresh entry.
    """
    access_token = login(client)

    first = post_json(client, "/graphical/demo/irrigation", {}, access_token)
    assert first.status_code == 200, first.text
    second = post_json(client, "/graphical/demo/irrigation", {}, access_token)
    assert second.status_code == 200, second.text

    assert first.json()["problem_id"] == second.json()["problem_id"]


def test_demo_irrigation_force_recompiles(client: TestClient, session_and_user: dict):
    """`force: true` bypasses the idempotency lookup and always compiles a fresh `problem_id`."""
    access_token = login(client)

    first = post_json(client, "/graphical/demo/irrigation", {}, access_token)
    assert first.status_code == 200, first.text
    forced = post_json(client, "/graphical/demo/irrigation", {"force": True}, access_token)
    assert forced.status_code == 200, forced.text

    assert first.json()["problem_id"] != forced.json()["problem_id"]

    # And a plain, non-forced call afterward picks up the *most recently* compiled copy -- not the
    # first one, and not an arbitrary one among the now-two candidates.
    third = post_json(client, "/graphical/demo/irrigation", {}, access_token)
    assert third.status_code == 200, third.text
    assert third.json()["problem_id"] == forced.json()["problem_id"]
