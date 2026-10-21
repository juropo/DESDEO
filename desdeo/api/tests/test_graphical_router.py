"""Tests for `desdeo/api/routers/graphical.py`.

Only tests that solve the compiled MILP are marked `gurobipy`. `_engine_cache` is cleared around
every test so engine reuse assertions are deterministic.
"""

import numpy as np
import pyagrum as gum
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

import desdeo.tools.utils as tools_utils
from desdeo.api.models import (
    CumulusClassificationState,
    GraphicalModelMetaData,
    Group,
    InteractiveSessionDB,
    ObjectiveDB,
    ProblemDB,
    ProblemMetaDataDB,
    ReferencePoint,
    StateDB,
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
from desdeo.problem.graphical.adapters.pyagrum import to_pyagrum, to_pyagrum_influence_diagram
from desdeo.problem.graphical.demo import (
    OIL_PRICE_PRIOR,
    STRUCTURE_PRIOR,
    build_irrigation_demo,
    build_oil_wildcatter_demo,
)
from desdeo.tools import SolverResults
from desdeo.tools.gurobipy_solver_interfaces import GurobipySolver

from .conftest import delete_json, get_json, login, post_json

pytestmark = pytest.mark.graphical


@pytest.fixture(autouse=True)
def _clear_engine_cache():
    graphical_router._engine_cache.clear()
    yield
    graphical_router._engine_cache.clear()


# --- Fixture builders -----------------------------------------------------------------------------


def _chain_network() -> DiscreteNetwork:
    """A -> B, two binary chance nodes, with a genuinely *normalised* CPT for B.

    Unlike `tests/fixtures/graphical.py`'s `chain_network`, which is unnormalised; pyAgrum would
    silently renormalise it, breaking hand-computed expectations.

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


def _impossible_evidence_network() -> DiscreteNetwork:
    """`Structure` (left unobserved by the tests using this) -> `Observed`, whose CPT is `"e0"` with certainty.

    Pinning `Observed="e1"` has zero joint probability, so reading `Structure`'s posterior raises
    pyAgrum's bare `Exception`.
    """
    return DiscreteNetwork(
        variables=[
            DiscreteVariable(name="Structure", labels=["s0", "s1", "s2"]),
            DiscreteVariable(name="Observed", labels=["e0", "e1"]),
        ],
        arcs=[("Structure", "Observed")],
        potentials={
            "Structure": Potential(variables=["Structure"], shape=[3], values=[0.5, 0.3, 0.2]),
            "Observed": Potential(
                variables=["Observed", "Structure"], shape=[2, 3], values=[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]
            ),
        },
    )


def _impossible_evidence_no_decision_annotation() -> DecisionAnnotation:
    return DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="f1",
                parents=["Observed"],
                utility=Potential(variables=["Observed"], shape=[2], values=[0.0, 10.0]),
            )
        ]
    )


def _impossible_evidence_single_decision_annotation() -> DecisionAnnotation:
    """One blind decision `D`, with `f1` parented on `[D, Structure]` so its MEU needs `Structure`'s posterior.

    Tests leave `D` unpinned so that `MEU()` is used.
    """
    return DecisionAnnotation(
        decisions=[DecisionNode(symbol="D", states=["s0", "s1"], information_set=[])],
        objectives=[
            ObjectiveNode(
                symbol="f1",
                parents=["D", "Structure"],
                utility=Potential(
                    variables=["D", "Structure"], shape=[2, 3], values=[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
                ),
                maximize=True,
            )
        ],
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

    Enough for `/structure`/`/whatif`, which don't need the RJT or IndexMap.
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

    Needed by `/policy`.
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


# --- POST /graphical/compile ----------------------------------------------------------------------


def test_compile_persists_problem_and_metadata(client: TestClient, session_and_user: dict):
    """A successful compile persists a `Problem` plus a `GraphicalModelMetaData`.

    All four JSON columns are populated.
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

    # Round-trip each column and compare against freshly built objects.
    assert DiscreteNetwork.model_validate_json(meta.network_json) == network
    assert DecisionAnnotation.model_validate_json(meta.annotation_json) == annotation
    expected_rjt = build_rooted_junction_tree(network, annotation, use_cache=False)
    _, expected_index_map = compile_diagram(network, annotation, rjt=expected_rjt)
    assert RootedJunctionTree.model_validate_json(meta.rjt_json) == expected_rjt
    assert IndexMap.model_validate_json(meta.index_map_json) == expected_index_map


def test_compile_bad_annotation_returns_400(client: TestClient, session_and_user: dict):
    """An annotation that doesn't validate against the network is a 400, not a bare 500.

    Here, an objective references an unknown parent.
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


def test_compile_many_small_disjoint_clusters_exceeding_total_cap_is_400(
    client: TestClient, session_and_user: dict, monkeypatch: pytest.MonkeyPatch
):
    """Many small, disjoint-clique clusters whose *sum* exceeds the total cap are a clean 400.

    Five `A_i -> B_i` pairs (cardinality 20, joined through a shared `Root` so the network is
    connected) give a largest cluster of 400, far under `_SERVER_MAX_CLUSTER_SIZE`.
    `_SERVER_MAX_TOTAL_CLUSTER_SIZE` is monkeypatched down so the total exceeds it.
    """
    monkeypatch.setattr(graphical_router, "_SERVER_MAX_TOTAL_CLUSTER_SIZE", 1000)

    access_token = login(client)

    n_pairs, card = 5, 20
    variables = [DiscreteVariable(name="Root", labels=["r0", "r1"])]
    arcs: list[tuple[str, str]] = []
    potentials = {"Root": Potential(variables=["Root"], shape=[2], values=[0.5, 0.5])}
    for i in range(n_pairs):
        a_name, b_name = f"A{i}", f"B{i}"
        variables.append(DiscreteVariable(name=a_name, labels=[f"a{i}_{j}" for j in range(card)]))
        variables.append(DiscreteVariable(name=b_name, labels=[f"b{i}_{j}" for j in range(card)]))
        arcs.append(("Root", a_name))
        arcs.append((a_name, b_name))
        potentials[a_name] = Potential(
            variables=[a_name, "Root"], shape=[card, 2], values=(np.ones((card, 2)) / card).tolist()
        )
        potentials[b_name] = Potential(
            variables=[b_name, a_name], shape=[card, card], values=(np.ones((card, card)) / card).tolist()
        )

    network = DiscreteNetwork(variables=variables, arcs=arcs, potentials=potentials)
    annotation = DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="f1", parents=["Root"], utility=Potential(variables=["Root"], shape=[2], values=[0.0, 1.0])
            )
        ]
    )

    payload = {"network": network.model_dump(mode="json"), "annotation": annotation.model_dump(mode="json")}
    response = post_json(client, "/graphical/compile", payload, access_token)
    assert response.status_code == 400, response.text
    assert "combined" in response.json()["detail"].lower()


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
    # Objective node ids are prefixed with `util_` to avoid collisions with chance nodes.
    assert nodes_by_id["util_fmatch"]["kind"] == "objective"
    assert nodes_by_id["util_fmatch"]["label"] == "fmatch"
    assert nodes_by_id["util_fmatch"]["states"] is None

    edges = {(e["source"], e["target"]) for e in data["edges"]}
    assert ("A", "D") in edges  # D's information_set
    assert ("A", "util_fmatch") in edges  # fmatch's parents
    assert ("D", "util_fmatch") in edges
    # Every edge carries a stable id derived from its endpoints.
    assert all(e["id"] == f"{e['source']}->{e['target']}" for e in data["edges"])


def test_structure_includes_decision_parent_edge_from_chance_potential(client: TestClient, session_and_user: dict):
    """A chance node's own potential naming a decision parent must appear as an edge too."""
    session: Session = session_and_user["session"]
    access_token = login(client)
    problem = session.exec(select(ProblemDB)).first()

    network = DiscreteNetwork(
        variables=[
            DiscreteVariable(name="A", labels=["lo", "hi"]),
            DiscreteVariable(name="B", labels=["lo", "hi"]),
        ],
        arcs=[("A", "B")],
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6]),
            "B": Potential(
                variables=["B", "A", "D"],
                shape=[2, 2, 2],
                values=[[[0.8, 0.4], [0.6, 0.1]], [[0.2, 0.6], [0.4, 0.9]]],
            ),
        },
    )
    annotation = DecisionAnnotation(
        decisions=[DecisionNode(symbol="D", states=["s0", "s1"])],
        objectives=[
            ObjectiveNode(
                symbol="f1", parents=["B"], utility=Potential(variables=["B"], shape=[2], values=[0.0, 10.0])
            )
        ],
    )
    _persist_graphical_metadata(session, problem.id, network, annotation)

    response = get_json(client, f"/graphical/{problem.id}/structure", access_token)
    assert response.status_code == 200
    data = response.json()

    edges = {(e["source"], e["target"]) for e in data["edges"]}
    assert ("D", "B") in edges


def test_structure_404_when_no_graphical_metadata(client: TestClient, session_and_user: dict):
    """A problem with `ProblemMetaDataDB` but no `GraphicalModelMetaData` row is a 404."""
    session: Session = session_and_user["session"]
    access_token = login(client)
    problem = session.exec(select(ProblemDB).where(ProblemDB.name == "The river pollution problem")).first()

    response = get_json(client, f"/graphical/{problem.id}/structure", access_token)
    assert response.status_code == 404


def test_structure_ownership_enforced(client: TestClient, session_and_user: dict):
    """A `dm`-role user cannot read another user's problem's structure.

    (Analyst/admin roles bypass ownership.)
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


# --- POST /graphical/whatif -----------------------------------------------------------------------


def test_whatif_no_decision_posteriors_and_utilities(client: TestClient, session_and_user: dict):
    """Posteriors and utilities under hard evidence, checked against a from-scratch computation.

    Computed directly from the network's CPTs and utility table.
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

    # A is pinned to "hi", so B's posterior is B's CPT column for A="hi".
    assert data["posteriors"]["A"] == pytest.approx([0.0, 1.0])
    assert data["posteriors"]["B"] == pytest.approx([0.2, 0.8])
    assert data["policy"] is None
    assert data["utilities"]["f1"] == pytest.approx(0.0 * 0.2 + 10.0 * 0.8)
    # variance = E[U^2] - mean^2 = (0.0^2 * 0.2 + 10.0^2 * 0.8) - 8.0^2 = 16.0, std = 4.0
    assert data["utility_std_devs"]["f1"] == pytest.approx(4.0)


def _no_decision_annotation_matching_dtlz2_f1(utility_values: list[float]) -> DecisionAnnotation:
    """A no-decision annotation whose objective symbol (`"f_1"`) matches `dtlz2(5, 3)`'s own.

    Lets the ratchet tests reuse the seeded `dtlz2(5, 3)` problem (`f_1` minimized, `ideal=0.0`,
    `nadir=1.0`), matching objectives by symbol as `_ratchet_ideal_nadir` does.
    """
    return DecisionAnnotation(
        objectives=[
            ObjectiveNode(
                symbol="f_1", parents=["B"], utility=Potential(variables=["B"], shape=[2], values=utility_values)
            )
        ]
    )


def _f1_ideal_nadir(session: Session, problem_id: int) -> tuple[float | None, float | None]:
    obj_db = session.exec(
        select(ObjectiveDB).where(ObjectiveDB.problem_id == problem_id, ObjectiveDB.symbol == "f_1")
    ).first()
    return obj_db.ideal, obj_db.nadir


def test_whatif_commit_false_does_not_ratchet_or_return_bounds(client: TestClient, session_and_user: dict):
    """`commit=False` (the default): no DB write, and `ideal`/`nadir` stay `None` on the response.

    The computed utility (-10.0) would beat `f_1`'s ideal (0.0) if the ratchet ran.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _chain_network()
    annotation = _no_decision_annotation_matching_dtlz2_f1([-10.0, -10.0])
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {"A": "hi"}, "commit": False}
    response = post_json(client, "/graphical/whatif", payload, access_token)
    assert response.status_code == 200
    data = response.json()

    assert data["utilities"]["f_1"] == pytest.approx(-10.0)
    assert data["ideal"] is None
    assert data["nadir"] is None

    session.expire_all()
    ideal, nadir = _f1_ideal_nadir(session, problem.id)
    assert ideal == pytest.approx(0.0)
    assert nadir == pytest.approx(1.0)


def test_whatif_commit_true_ratchets_ideal_open_on_violation(client: TestClient, session_and_user: dict):
    """`commit=True` with a utility beating the current ideal widens and persists it.

    `f_1` minimizes with `ideal=0.0`/`nadir=1.0`; a utility of -10.0 replaces the ideal and leaves
    the nadir untouched.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _chain_network()
    annotation = _no_decision_annotation_matching_dtlz2_f1([-10.0, -10.0])
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {"A": "hi"}, "commit": True}
    response = post_json(client, "/graphical/whatif", payload, access_token)
    assert response.status_code == 200
    data = response.json()

    assert data["utilities"]["f_1"] == pytest.approx(-10.0)
    assert data["ideal"]["f_1"] == pytest.approx(-10.0)
    assert data["nadir"]["f_1"] == pytest.approx(1.0)

    session.expire_all()
    ideal, nadir = _f1_ideal_nadir(session, problem.id)
    assert ideal == pytest.approx(-10.0)
    assert nadir == pytest.approx(1.0)


def test_whatif_commit_true_inside_range_does_not_change_bounds(client: TestClient, session_and_user: dict):
    """`commit=True` with a utility inside the current ideal/nadir range changes nothing.

    Bounds are only ever widened, never narrowed.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _chain_network()
    # Expected utility = 0.2 * 0.3 + 0.8 * 0.3 = 0.3, strictly inside [ideal=0.0, nadir=1.0].
    annotation = _no_decision_annotation_matching_dtlz2_f1([0.3, 0.3])
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {"A": "hi"}, "commit": True}
    response = post_json(client, "/graphical/whatif", payload, access_token)
    assert response.status_code == 200
    data = response.json()

    assert data["utilities"]["f_1"] == pytest.approx(0.3)
    assert data["ideal"]["f_1"] == pytest.approx(0.0)
    assert data["nadir"]["f_1"] == pytest.approx(1.0)

    session.expire_all()
    ideal, nadir = _f1_ideal_nadir(session, problem.id)
    assert ideal == pytest.approx(0.0)
    assert nadir == pytest.approx(1.0)


def test_whatif_commit_true_multi_objective_with_decision_still_raises_501(
    client: TestClient, session_and_user: dict
):
    """`commit=True` must not change or bypass `_compute_utilities`'s existing 501 scope cut.

    Same fixture as `test_whatif_multi_objective_with_decision_raises_501`, plus `commit=True`.
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

    payload = {"problem_id": problem.id, "evidence": {}, "commit": True}
    response = post_json(client, "/graphical/whatif", payload, access_token)
    assert response.status_code == 501


def test_whatif_commit_true_both_bounds_none_skips_ratchet(client: TestClient, session_and_user: dict):
    """A freshly-*compiled* graphical-model problem starts with `ideal`/`nadir` both `None`.

    The ratchet must skip it rather than collapse both bounds to the same value.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    network = _chain_network()
    annotation = _no_decision_annotation()
    compile_payload = {"network": network.model_dump(mode="json"), "annotation": annotation.model_dump(mode="json")}
    compile_response = post_json(client, "/graphical/compile", compile_payload, access_token)
    assert compile_response.status_code == 200, compile_response.text
    problem_id = compile_response.json()["problem_id"]

    problem_db = session.get(ProblemDB, problem_id)
    assert all(obj_db.ideal is None and obj_db.nadir is None for obj_db in problem_db.objectives)

    payload = {"problem_id": problem_id, "evidence": {"A": "hi"}, "commit": True}
    response = post_json(client, "/graphical/whatif", payload, access_token)
    assert response.status_code == 200
    data = response.json()

    assert data["utilities"]["f1"] == pytest.approx(0.0 * 0.2 + 10.0 * 0.8)
    # Both bounds were skipped, so "f1" is absent from the returned dicts.
    assert "f1" not in data["ideal"]
    assert "f1" not in data["nadir"]

    session.expire_all()
    obj_db = session.exec(
        select(ObjectiveDB).where(ObjectiveDB.problem_id == problem_id, ObjectiveDB.symbol == "f1")
    ).first()
    assert obj_db.ideal is None
    assert obj_db.nadir is None


def test_whatif_commit_non_owner_without_privilege_is_403(client: TestClient, session_and_user: dict):
    """A user who can SEE the problem (via a shared `Group`) but doesn't own it can't `commit`.

    A non-committing call from the same user still works.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)  # default "analyst" user -- owns the problem

    problem = session.exec(select(ProblemDB)).first()
    network = _chain_network()
    annotation = _no_decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    dm_user = User(username="dm_user6", password_hash=get_password_hash("dm_pw"), role=UserRole.dm, group="test")
    session.add(dm_user)
    session.commit()
    session.refresh(dm_user)
    dm_token = login(client, username="dm_user6", password="dm_pw")  # noqa: S106

    group = Group(name="shared6", owner_id=session_and_user["user"].id, user_ids=[dm_user.id], problem_id=problem.id)
    session.add(group)
    session.commit()

    # dm_user6 can see the problem (group membership) -- a non-committing call succeeds.
    plain_payload = {"problem_id": problem.id, "evidence": {"A": "hi"}}
    plain_response = post_json(client, "/graphical/whatif", plain_payload, dm_token)
    assert plain_response.status_code == 200

    # ... but isn't the owner, so a committing call is forbidden.
    commit_payload = {"problem_id": problem.id, "evidence": {"A": "hi"}, "commit": True}
    forbidden = post_json(client, "/graphical/whatif", commit_payload, dm_token)
    assert forbidden.status_code == 403


def test_whatif_with_decision_single_objective_matches_pyagrum_oracle(client: TestClient, session_and_user: dict):
    """Policy and utility for a single-objective decision diagram, checked against two oracles.

    First against a hand-computed value: with `A` pinned to `"lo"`, `D` (which observes `A`)
    chooses `"s0"`, with utility `utility[A=lo, D=s0] = 10.0`. Then against a pyAgrum oracle.
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
    """With decisions, `utility_std_devs` == `sqrt(MEU()["variance"])`.

    `D` is blind (`information_set=[]`) so the utility has non-zero variance. `D="s0"`
    (`E[U]=0.3*10+0.7*0=3.0`, beating `D="s1"`'s `0.3*0+0.7*3=2.1`), so the outcome is `10.0`
    w.p. `0.3` or `0.0` w.p. `0.7`: `mean=3.0`, `variance=0.3*100-9=21.0`.
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

    pyAgrum always maximizes, so the utility table must be negated for minimized objectives. `D`
    observes `A` and pays a mismatch cost (`cost[A=lo,D=s0]=0`, `cost[A=lo,D=s1]=10`,
    `cost[A=hi,D=s0]=10`, `cost[A=hi,D=s1]=0`), so it should choose `D="s0"` when `A="lo"`.
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

    With no evidence, `D`'s posterior is the marginal `[0.3, 0.7]`, not a policy: the optimal
    choice depends on `A`.
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

    Also checks, by counting `_build_engine` calls, that the engine is built only once.
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

    Mutates the `GraphicalModelMetaData` row's `network_json` directly, then checks that the next
    `/whatif` call in the same session rebuilds the engine and reflects the new network.
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

    See `_compute_utilities`. The message points at `decision_evidence` as the way out.
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
    assert "decision_evidence" in response.json()["detail"]


def test_whatif_oil_wildcatter_partially_pinned_or_unpinned_decisions_still_raises_501(
    client: TestClient, session_and_user: dict
):
    """`decision_evidence` must cover *every* decision to lift the 501, not just some of them.

    Uses the real Oil Wildcatter fixture (three objectives, two decisions): leaving `decision_evidence`
    empty, or pinning only `test` (leaving `drill` unresolved), both still 501.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network, annotation = build_oil_wildcatter_demo()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    unpinned_payload = {"problem_id": problem.id, "evidence": {}}
    unpinned_response = post_json(client, "/graphical/whatif", unpinned_payload, access_token)
    assert unpinned_response.status_code == 501

    partially_pinned_payload = {
        "problem_id": problem.id,
        "evidence": {},
        "decision_evidence": {"test": "yes"},
    }
    partially_pinned_response = post_json(client, "/graphical/whatif", partially_pinned_payload, access_token)
    assert partially_pinned_response.status_code == 501


def test_whatif_decision_evidence_unknown_decision_or_bad_label_is_400(
    client: TestClient, session_and_user: dict
):
    """`decision_evidence` referencing an unknown decision, or a label outside its own states, is 400."""
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _decision_network()
    annotation = _decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    unknown_decision_payload = {
        "problem_id": problem.id,
        "evidence": {},
        "decision_evidence": {"ZZZ": "s0"},
    }
    unknown_response = post_json(client, "/graphical/whatif", unknown_decision_payload, access_token)
    assert unknown_response.status_code == 400

    bad_label_payload = {
        "problem_id": problem.id,
        "evidence": {},
        "decision_evidence": {"D": "not_a_state"},
    }
    bad_label_response = post_json(client, "/graphical/whatif", bad_label_payload, access_token)
    assert bad_label_response.status_code == 400


def test_whatif_fully_pinned_decisions_multi_objective_matches_hand_computed_expectation(
    client: TestClient, session_and_user: dict
):
    """Pinning every decision via `decision_evidence` lifts the 501 for the Oil Wildcatter demo.

    `test`/`drill` are both pinned to `"yes"`. Expectations are hand-computed from the demo's
    constants.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network, annotation = build_oil_wildcatter_demo()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {
        "problem_id": problem.id,
        "evidence": {},
        "decision_evidence": {"test": "yes", "drill": "yes"},
    }
    response = post_json(client, "/graphical/whatif", payload, access_token)
    assert response.status_code == 200, response.text
    data = response.json()

    structure_prior = np.array(STRUCTURE_PRIOR)  # [dry, wet, soaking]
    oil_price_prior = np.array(OIL_PRICE_PRIOR)  # [low, baseline, high]
    barrels = np.array([0.0, 1200.0, 2700.0])  # oil_produced by structure, drill="yes"
    price = np.array([70.0, 100.0, 130.0])  # $/barrel by oil_price
    investment = 10_000.0 + 70_000.0  # test cost + drill cost, both run

    expected_oil = float(np.dot(structure_prior, barrels))
    expected_oil_var = float(np.dot(structure_prior, barrels**2)) - expected_oil**2

    profit_table = np.outer(barrels, price) - investment  # [structure, oil_price]
    joint = np.outer(structure_prior, oil_price_prior)
    expected_profit = float(np.sum(joint * profit_table))
    expected_profit_var = float(np.sum(joint * profit_table**2)) - expected_profit**2

    assert data["utilities"]["investment"] == pytest.approx(investment)
    assert data["utility_std_devs"]["investment"] == pytest.approx(0.0)
    assert data["utilities"]["oil_produced"] == pytest.approx(expected_oil)
    assert data["utility_std_devs"]["oil_produced"] == pytest.approx(expected_oil_var**0.5)
    assert data["utilities"]["profit"] == pytest.approx(expected_profit)
    assert data["utility_std_devs"]["profit"] == pytest.approx(expected_profit_var**0.5)
    assert data["policy"] == {"test": "yes", "drill": "yes"}

    # `test_result`'s CPT resolves to the right conditional for each `test` pin.
    test_yes_engine = gum.LazyPropagation(to_pyagrum(network, decision_pins={"test": 0, "drill": 0}))
    test_yes_engine.makeInference()
    # P(test_result | structure, test="yes"), rows = structure (dry/wet/soaking).
    test_result_given_structure = np.array(
        [[0.6, 0.3, 0.1, 0.0], [0.3, 0.4, 0.3, 0.0], [0.1, 0.4, 0.5, 0.0]]
    )
    expected_test_result_given_yes = structure_prior @ test_result_given_structure
    assert test_yes_engine.posterior("test_result").toarray() == pytest.approx(expected_test_result_given_yes)

    # The response's own `posteriors` must come from this same pinned engine, not the cached
    # LIMID engine's optimal (and here different) policy -- see `whatif`'s use of
    # `_compute_utilities`'s `posteriors_override`.
    assert data["posteriors"]["test_result"] == pytest.approx(expected_test_result_given_yes.tolist())

    test_no_engine = gum.LazyPropagation(to_pyagrum(network, decision_pins={"test": 1, "drill": 1}))
    test_no_engine.makeInference()
    assert test_no_engine.posterior("test_result").toarray() == pytest.approx([0.0, 0.0, 0.0, 1.0])

    # Mixed pins, since both "yes" pins above share index 0.
    mixed_payload_a = {
        "problem_id": problem.id,
        "evidence": {},
        "decision_evidence": {"test": "no", "drill": "yes"},
    }
    mixed_response_a = post_json(client, "/graphical/whatif", mixed_payload_a, access_token)
    assert mixed_response_a.status_code == 200, mixed_response_a.text
    mixed_data_a = mixed_response_a.json()
    assert mixed_data_a["utilities"]["investment"] == pytest.approx(70_000.0)
    assert mixed_data_a["utilities"]["oil_produced"] == pytest.approx(900.0)
    assert mixed_data_a["utilities"]["profit"] == pytest.approx(17_300.0)

    mixed_payload_b = {
        "problem_id": problem.id,
        "evidence": {},
        "decision_evidence": {"test": "yes", "drill": "no"},
    }
    mixed_response_b = post_json(client, "/graphical/whatif", mixed_payload_b, access_token)
    assert mixed_response_b.status_code == 200, mixed_response_b.text
    mixed_data_b = mixed_response_b.json()
    assert mixed_data_b["utilities"]["investment"] == pytest.approx(10_000.0)
    assert mixed_data_b["utilities"]["oil_produced"] == pytest.approx(0.0)
    assert mixed_data_b["utilities"]["profit"] == pytest.approx(-10_000.0)


def test_whatif_decision_evidence_incompatible_with_evidence_is_400(client: TestClient, session_and_user: dict):
    """Evidence that's impossible given the pinned decisions is a 400, not an uncaught 500.

    With `test="no"` pinned, `test_result="NS"` has zero joint probability.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network, annotation = build_oil_wildcatter_demo()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {
        "problem_id": problem.id,
        "evidence": {"test_result": "NS"},
        "decision_evidence": {"test": "no", "drill": "no"},
    }
    response = post_json(client, "/graphical/whatif", payload, access_token)
    assert response.status_code == 400, response.text
    assert "incompatible" in response.json()["detail"].lower()


def test_whatif_no_decision_impossible_evidence_combination_is_400(client: TestClient, session_and_user: dict):
    """A no-decision network's zero-joint-probability evidence is a 400, not an uncaught 500."""
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _impossible_evidence_network()
    annotation = _impossible_evidence_no_decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {"Observed": "e1"}}
    response = post_json(client, "/graphical/whatif", payload, access_token)
    assert response.status_code == 400, response.text
    assert "incompatible" in response.json()["detail"].lower()


def test_whatif_single_objective_decision_nan_meu_is_400(client: TestClient, session_and_user: dict):
    """A decision diagram whose `MEU()` returns `nan` (zero-probability evidence) is a 400, not a serialization 500."""
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _impossible_evidence_network()
    annotation = _impossible_evidence_single_decision_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {"Structure": "s0", "Observed": "e1"}}
    response = post_json(client, "/graphical/whatif", payload, access_token)
    assert response.status_code == 400, response.text
    assert "incompatible" in response.json()["detail"].lower()


def test_whatif_partial_decision_evidence_single_objective_has_no_effect(
    client: TestClient, session_and_user: dict
):
    """A partial `decision_evidence` on a single-objective, two-decision diagram changes nothing.

    Neither decision is in `policy` without evidence on `A`; pinning only `D2` changes neither
    `policy` nor `utilities`.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _two_decision_no_forgetting_network()
    annotation = _two_decision_no_forgetting_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    baseline_payload = {"problem_id": problem.id, "evidence": {}, "decision_evidence": {}}
    baseline_response = post_json(client, "/graphical/whatif", baseline_payload, access_token)
    assert baseline_response.status_code == 200, baseline_response.text
    baseline_data = baseline_response.json()

    partial_payload = {"problem_id": problem.id, "evidence": {}, "decision_evidence": {"D2": "t0"}}
    partial_response = post_json(client, "/graphical/whatif", partial_payload, access_token)
    assert partial_response.status_code == 200, partial_response.text
    partial_data = partial_response.json()

    assert partial_data == baseline_data
    assert "D2" not in (partial_data["policy"] or {})


def test_whatif_fully_pinned_single_objective_decision_returns_pinned_policys_utility_not_meu(
    client: TestClient, session_and_user: dict
):
    """A fully pinned `decision_evidence` reports that policy's own expected utility, not the MEU.

    Irrigation demo: `"heavy"` is optimal (`2780.625`); pinning `"none"` must report its own
    expected profit (`2458.75`, hand-computed).
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network, annotation = build_irrigation_demo()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    diagram = to_pyagrum_influence_diagram(network, annotation)
    inference = gum.ShaferShenoyLIMIDInference(diagram)
    inference.makeInference()
    meu = inference.MEU()["mean"]
    assert meu == pytest.approx(2780.625)  # "heavy" is the actual optimum, not the pin below

    payload = {"problem_id": problem.id, "evidence": {}, "decision_evidence": {"irrigate": "none"}}
    response = post_json(client, "/graphical/whatif", payload, access_token)
    assert response.status_code == 200, response.text
    data = response.json()

    assert data["policy"] == {"irrigate": "none"}
    assert data["utilities"]["profit"] == pytest.approx(2458.75)
    assert data["utilities"]["profit"] != pytest.approx(meu)


def _store_solution_state(
    session: Session, problem_id: int, results: list[SolverResults], session_id: int | None = None
) -> StateDB:
    """Persist `results` as a CUMULUS solve state of `problem_id`, like a real CUMULUS iteration."""
    state = StateDB.create(
        session,
        problem_id=problem_id,
        session_id=session_id,
        state=CumulusClassificationState(preferences=ReferencePoint(aspiration_levels={}), solver_results=results),
    )
    session.commit()
    session.refresh(state)
    return state


def _oil_wildcatter_with_solved_state(session: Session) -> tuple[ProblemDB, SolverResults, StateDB]:
    """Oil Wildcatter attached to a problem, solved for expected `profit`, stored as a state."""
    problem = session.exec(select(ProblemDB)).first()
    network, annotation = build_oil_wildcatter_demo()
    _persist_graphical_metadata_with_index_map(session, problem.id, network, annotation)
    compiled, _ = compile_diagram(network, annotation)
    result = GurobipySolver(compiled).solve("profit")
    assert result.success, result.message
    return problem, result, _store_solution_state(session, problem.id, [result])


@pytest.mark.gurobipy
def test_whatif_solution_unpinned_decisions_follow_solved_policy(client: TestClient, session_and_user: dict):
    """With `solution` and no pins, utilities are the solved policy's own expected values.

    The expected-`profit` optimum runs the test and drills only on an open/closed-structure
    reading. Its expected values are hand-computed from the demo's constants, and must also match
    the solved MILP's own objective values, which are expectations too.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)
    problem, result, state = _oil_wildcatter_with_solved_state(session)

    payload = {
        "problem_id": problem.id,
        "evidence": {},
        "solution": {"state_id": state.id, "solution_index": 0},
    }
    response = post_json(client, "/graphical/whatif", payload, access_token)
    assert response.status_code == 200, response.text
    data = response.json()

    structure_prior = np.array(STRUCTURE_PRIOR)  # [dry, wet, soaking]
    barrels = np.array([0.0, 1200.0, 2700.0])
    # P(test_result in {OS, CS} | structure, test="yes").
    p_drill_given_structure = np.array([0.3 + 0.1, 0.4 + 0.3, 0.4 + 0.5])
    expected_price = float(np.dot(OIL_PRICE_PRIOR, [70.0, 100.0, 130.0]))
    expected_oil = float(np.sum(structure_prior * p_drill_given_structure * barrels))
    expected_investment = 10_000.0 + 70_000.0 * float(np.dot(structure_prior, p_drill_given_structure))
    expected_profit = expected_price * expected_oil - expected_investment

    assert data["utilities"]["investment"] == pytest.approx(expected_investment)
    assert data["utilities"]["oil_produced"] == pytest.approx(expected_oil)
    assert data["utilities"]["profit"] == pytest.approx(expected_profit)
    for symbol, value in result.optimal_objectives.items():
        assert data["utilities"][symbol] == pytest.approx(value)

    # `drill` depends on `test_result`, which isn't observed, so only `test` is resolved.
    assert data["policy"] == {"test": "yes"}
    assert data["posteriors"]["test_result"][3] == pytest.approx(0.0)  # the test was run

    observed_payload = {**payload, "evidence": {"test_result": "CS"}}
    observed_response = post_json(client, "/graphical/whatif", observed_payload, access_token)
    assert observed_response.status_code == 200, observed_response.text
    assert observed_response.json()["policy"] == {"test": "yes", "drill": "yes"}
    assert observed_response.json()["off_policy_decisions"] == []


@pytest.mark.gurobipy
def test_whatif_solution_decision_posteriors(client: TestClient, session_and_user: dict):
    """`decision_posteriors` holds each followed decision's state probabilities under the solution.

    The profit-optimal policy always tests and drills on an OS/CS reading, so `P(drill="yes")` is
    `P(test_result in {OS, CS} | test="yes")`. Pinned and off-policy decisions are omitted.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)
    problem, _result, state = _oil_wildcatter_with_solved_state(session)
    solution = {"state_id": state.id, "solution_index": 0}

    def _whatif(**fields) -> dict:
        response = post_json(client, "/graphical/whatif", {"problem_id": problem.id, **fields}, access_token)
        assert response.status_code == 200, response.text
        return response.json()

    data = _whatif(solution=solution)
    p_drill = float(np.dot(STRUCTURE_PRIOR, [0.3 + 0.1, 0.4 + 0.3, 0.4 + 0.5]))
    assert data["decision_posteriors"]["test"] == pytest.approx([1.0, 0.0])  # [yes, no]
    assert data["decision_posteriors"]["drill"] == pytest.approx([p_drill, 1.0 - p_drill])
    assert 0.0 < p_drill < 1.0

    observed = _whatif(solution=solution, evidence={"test_result": "CS"})
    assert observed["decision_posteriors"]["drill"] == pytest.approx([1.0, 0.0])

    off_path = _whatif(solution=solution, decision_evidence={"test": "no"})
    assert off_path["off_policy_decisions"] == ["drill"]
    assert off_path["decision_posteriors"] == {}

    drill_pinned = _whatif(solution=solution, decision_evidence={"drill": "yes"})
    assert drill_pinned["decision_posteriors"] == {"test": pytest.approx([1.0, 0.0])}

    assert _whatif(solution=solution, decision_evidence={"test": "yes", "drill": "no"})["decision_posteriors"] == {}
    assert _whatif(decision_evidence={"test": "yes", "drill": "no"})["decision_posteriors"] == {}


@pytest.mark.gurobipy
def test_whatif_solution_pin_overrides_solved_policy(client: TestClient, session_and_user: dict):
    """A pinned decision overrides the solution; the other decision still follows it.

    Pinning `drill="yes"` keeps the solved `test="yes"`. With both decisions pinned the solution
    is irrelevant, so the result matches the pins-only path.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)
    problem, _result, state = _oil_wildcatter_with_solved_state(session)
    solution = {"state_id": state.id, "solution_index": 0}

    drill_payload = {"problem_id": problem.id, "decision_evidence": {"drill": "yes"}, "solution": solution}
    drill_response = post_json(client, "/graphical/whatif", drill_payload, access_token)
    assert drill_response.status_code == 200, drill_response.text
    drill_data = drill_response.json()
    assert drill_data["utilities"]["investment"] == pytest.approx(80_000.0)
    assert drill_data["policy"] == {"test": "yes", "drill": "yes"}
    assert drill_data["off_policy_decisions"] == []

    # Deterministic `investment` (test followed, drill pinned): roundoff in the variance is
    # reported as an exact zero, not e.g. 0.000122.
    no_drill_payload = {
        "problem_id": problem.id,
        "evidence": {"test_result": "OS"},
        "decision_evidence": {"drill": "no"},
        "solution": solution,
    }
    no_drill_response = post_json(client, "/graphical/whatif", no_drill_payload, access_token)
    assert no_drill_response.status_code == 200, no_drill_response.text
    assert no_drill_response.json()["utilities"]["investment"] == pytest.approx(10_000.0)
    assert no_drill_response.json()["utility_std_devs"]["investment"] == 0.0

    for both in ({"test": "yes", "drill": "no"}, {"test": "no", "drill": "no"}):
        with_solution = post_json(
            client,
            "/graphical/whatif",
            {"problem_id": problem.id, "decision_evidence": both, "solution": solution},
            access_token,
        )
        without_solution = post_json(
            client, "/graphical/whatif", {"problem_id": problem.id, "decision_evidence": both}, access_token
        )
        assert with_solution.status_code == 200, with_solution.text
        assert without_solution.status_code == 200, without_solution.text
        assert with_solution.json()["utilities"] == pytest.approx(without_solution.json()["utilities"])
        assert with_solution.json()["posteriors"] == pytest.approx(without_solution.json()["posteriors"])
        assert with_solution.json()["off_policy_decisions"] == []


@pytest.mark.gurobipy
def test_whatif_solution_pin_moving_followed_decision_off_path_reports_it(
    client: TestClient, session_and_user: dict
):
    """A pin that puts a followed decision where the solution never goes yields no utilities.

    The profit-optimal solution runs the test, so `drill`'s choice at `test="no"` is left
    unconstrained by the MILP. Pinning `test="no"` reports `drill` as off-policy instead of
    using that arbitrary choice.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)
    problem, _result, state = _oil_wildcatter_with_solved_state(session)
    solution = {"state_id": state.id, "solution_index": 0}

    payload = {"problem_id": problem.id, "decision_evidence": {"test": "no"}, "solution": solution}
    response = post_json(client, "/graphical/whatif", payload, access_token)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["off_policy_decisions"] == ["drill"]
    assert data["utilities"] == {}
    assert data["utility_std_devs"] == {}
    assert data["policy"] == {"test": "no"}
    # Upstream of `drill`, the posteriors still reflect the pin.
    assert data["posteriors"]["test_result"] == pytest.approx([0.0, 0.0, 0.0, 1.0])


@pytest.mark.gurobipy
def test_whatif_solution_unusable_reference_is_404_or_400(client: TestClient, session_and_user: dict):
    """Missing / inaccessible / other-problem states are 404; a bad index or stale `z` is 400."""
    session: Session = session_and_user["session"]
    owner = session_and_user["user"]
    access_token = login(client)
    problem, result, state = _oil_wildcatter_with_solved_state(session)

    def whatif_with(solution: dict, token: str = access_token):
        return post_json(client, "/graphical/whatif", {"problem_id": problem.id, "solution": solution}, token)

    assert whatif_with({"state_id": 999_999, "solution_index": 0}).status_code == 404

    other_problem = session.exec(select(ProblemDB)).all()[1]
    other_state = _store_solution_state(session, other_problem.id, [result])
    assert whatif_with({"state_id": other_state.id, "solution_index": 0}).status_code == 404

    # A state in the owner's own interactive session, seen by a user who can see the problem.
    owner_session = InteractiveSessionDB(user_id=owner.id, info="owner's session")
    session.add(owner_session)
    session.commit()
    session.refresh(owner_session)
    private_state = _store_solution_state(session, problem.id, [result], session_id=owner_session.id)
    dm_user = User(username="dm_user_whatif", password_hash=get_password_hash("dm_pw"), role=UserRole.dm, group="test")
    session.add(dm_user)
    session.commit()
    session.refresh(dm_user)
    session.add(Group(name="shared-whatif", owner_id=owner.id, user_ids=[dm_user.id], problem_id=problem.id))
    session.commit()
    dm_token = login(client, username="dm_user_whatif", password="dm_pw")  # noqa: S106
    assert whatif_with({"state_id": private_state.id, "solution_index": 0}, dm_token).status_code == 404
    assert whatif_with({"state_id": private_state.id, "solution_index": 0}).status_code == 200

    assert whatif_with({"state_id": state.id, "solution_index": 1}).status_code == 400
    assert whatif_with({"state_id": state.id, "solution_index": -1}).status_code == 400

    _, index_map = compile_diagram(*build_oil_wildcatter_demo())
    stale_variables = {**result.optimal_variables, index_map.decisions["test"].z_symbol: [1.0]}
    stale_result = result.model_copy(update={"optimal_variables": stale_variables})
    stale_state = _store_solution_state(session, problem.id, [stale_result])
    assert whatif_with({"state_id": stale_state.id, "solution_index": 0}).status_code == 400


def test_whatif_solution_without_index_map_is_404(client: TestClient, session_and_user: dict):
    """A problem whose metadata has no `IndexMap` can't decode a solution's policy, unless it's not needed."""
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network, annotation = build_oil_wildcatter_demo()
    _persist_graphical_metadata(session, problem.id, network, annotation)
    results = SolverResults(optimal_objectives={}, optimal_variables={}, success=True, message="")
    state = _store_solution_state(session, problem.id, [results])

    payload = {"problem_id": problem.id, "solution": {"state_id": state.id, "solution_index": 0}}
    response = post_json(client, "/graphical/whatif", payload, access_token)
    assert response.status_code == 404, response.text

    # With every decision pinned the solution isn't needed, so it isn't decoded.
    pinned_payload = {**payload, "decision_evidence": {"test": "yes", "drill": "yes"}}
    pinned_response = post_json(client, "/graphical/whatif", pinned_payload, access_token)
    assert pinned_response.status_code == 200, pinned_response.text


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


# --- POST /graphical/{problem_id}/policy ----------------------------------------------------------


def _solved_decision_problem(network: DiscreteNetwork, annotation: DecisionAnnotation, objective_symbol: str):
    """Compile + solve `(network, annotation)`, returning `(index_map, optimal_variables)`."""
    problem, index_map = compile_diagram(network, annotation)
    result = GurobipySolver(problem).solve(objective_symbol)
    assert result.success, result.message
    return index_map, result.optimal_variables


@pytest.mark.gurobipy
def test_policy_decodes_solved_z_given_evidence(client: TestClient, session_and_user: dict):
    """`/policy` reports `D`'s solved choice for the information state `evidence` picks out.

    `D` observes `A` and is rewarded for matching it, so the solved policy is `D="s0"` when
    `A="lo"` and `D="s1"` when `A="hi"`. Both are checked from the same solution.
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

    Same semantics as `/whatif`.
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

    D2's effective information set includes the decision D1, so `/policy` must decode in order.
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

    Pinning `A` determines both decisions, even though D2's information set includes D1.
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

    Gives the same answer as passing `variable_values`.
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


@pytest.mark.gurobipy
def test_policy_solution_id_visible_non_owner_is_403(client: TestClient, session_and_user: dict):
    """A user who can see the problem (via a shared `Group`) but didn't save the solution gets 403.

    Visibility into the problem doesn't grant access to another user's saved solution.
    """
    session: Session = session_and_user["session"]
    owner = session_and_user["user"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _decision_network()
    annotation = _decision_annotation()
    _persist_graphical_metadata_with_index_map(session, problem.id, network, annotation)

    _, optimal_variables = _solved_decision_problem(network, annotation, "fmatch")

    saved = UserSavedSolutionDB(
        name="owner's solution",
        objective_values={"fmatch": 10.0},
        variable_values=optimal_variables,
        solution_index=0,
        user_id=owner.id,
        problem_id=problem.id,
    )
    session.add(saved)
    session.commit()
    session.refresh(saved)

    dm_user = User(username="dm_user_policy", password_hash=get_password_hash("dm_pw"), role=UserRole.dm, group="test")
    session.add(dm_user)
    session.commit()
    session.refresh(dm_user)
    dm_token = login(client, username="dm_user_policy", password="dm_pw")  # noqa: S106

    group = Group(name="shared-policy", owner_id=owner.id, user_ids=[dm_user.id], problem_id=problem.id)
    session.add(group)
    session.commit()

    payload = {"solution_id": saved.id, "evidence": {"A": "lo"}}
    response = post_json(client, f"/graphical/{problem.id}/policy", payload, dm_token)
    assert response.status_code == 403, response.text


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

    `index_map_json` is `None`, which `/policy` needs.
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

    E.g. a stale saved solution. `D`'s information set is `["A"]` (cardinality 2) and `D` has 2
    states, so `z_D` should have length 4; this one has 3.
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


def test_policy_scalar_z_symbol_entry_is_400_not_500(client: TestClient, session_and_user: dict):
    """A bare scalar on a decision's own `z_symbol` entry is a clean 400, not an uncaught `TypeError`/500.

    `variable_values` accepts scalars (for a `RiskMeasure`'s `eta`), but not for a `z_symbol`.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _decision_network()
    annotation = _decision_annotation()
    _persist_graphical_metadata_with_index_map(session, problem.id, network, annotation)

    payload = {"variable_values": {"z_D": 1.0}, "evidence": {}}
    response = post_json(client, f"/graphical/{problem.id}/policy", payload, access_token)
    assert response.status_code == 400, response.text


@pytest.mark.gurobipy
def test_policy_accepts_scalar_risk_measure_eta_without_422(client: TestClient, session_and_user: dict):
    """`variable_values` accepts a real solved-solution dict with a scalar RiskMeasure eta.

    The scalar `eta` sits alongside the decision's list-valued `z`.
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


# --- Named evidence sets --------------------------------------------------------------------------


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

    Otherwise the set would only fail later, when used.
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

    Not only the user who saved it.
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

    `dm_user4` can't see the problem at all, so this covers the 404 branch. See
    `test_evidence_set_delete_visible_non_owner_is_403` for the 403 branch.
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

    # 404 rather than 403, so as not to reveal that the set exists.
    forbidden = delete_json(client, f"/graphical/evidence-sets/{evidence_set_id}", dm_token)
    assert forbidden.status_code == 404

    ok = delete_json(client, f"/graphical/evidence-sets/{evidence_set_id}", access_token)
    assert ok.status_code == 200

    gone = get_json(client, f"/graphical/evidence-sets/{evidence_set_id}", access_token)
    assert gone.status_code == 404


def test_evidence_set_delete_visible_non_owner_is_403(client: TestClient, session_and_user: dict):
    """A user who CAN see the problem (via a shared `Group`) but didn't save the set gets a 403.

    `dm_user5` gets visibility through a `Group` without owning the problem.
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


# --- POST /graphical/sensitivity ------------------------------------------------------------------


def _sensitivity_fixture_network() -> DiscreteNetwork:
    """Three independent binary chance nodes: A, B, C -- no arcs at all.

    With an objective depending only on A, sweeping A moves the objective and sweeping B or C
    gives a swing of exactly zero.
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

    Covers both `_validate_chance_evidence`'s label check and the `finally` block's restore not
    replacing the 400 with a 500.
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


def test_sensitivity_nodes_filter_restricts_sweep(client: TestClient, session_and_user: dict):
    """`nodes` restricts the sweep to only the given node(s), not every chance node."""
    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _sensitivity_fixture_network()
    annotation = _sensitivity_fixture_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {}, "target": "f1", "nodes": ["B"]}
    response = post_json(client, "/graphical/sensitivity", payload, access_token)
    assert response.status_code == 200, response.text
    swept_nodes = {r["node"] for r in response.json()["results"]}
    assert swept_nodes == {"B"}  # not {"A", "B", "C"} -- C is excluded by the filter


def test_sensitivity_sweep_cap_exceeded_is_400(client: TestClient, session_and_user: dict, monkeypatch: pytest.MonkeyPatch):
    """Exceeding `_MAX_SENSITIVITY_SWEEP_NODES` (after any `nodes` filtering) is a clean 400.

    The cap is monkeypatched down to 1; `B` and `C` are both swept by default.
    """
    monkeypatch.setattr(graphical_router, "_MAX_SENSITIVITY_SWEEP_NODES", 1)

    session: Session = session_and_user["session"]
    access_token = login(client)

    problem = session.exec(select(ProblemDB)).first()
    network = _sensitivity_fixture_network()
    annotation = _sensitivity_fixture_annotation()
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {}, "target": "f1"}
    response = post_json(client, "/graphical/sensitivity", payload, access_token)
    assert response.status_code == 400, response.text


def test_sensitivity_guest_rejected(client: TestClient, session_and_user: dict):
    """Guest users may not run `/sensitivity`, matching `/compile`'s own guest check."""
    session: Session = session_and_user["session"]
    guest = User(
        username="guest_sensitivity", password_hash=get_password_hash("guest_pw"), role=UserRole.guest, group="test"
    )
    session.add(guest)
    session.commit()

    access_token = login(client, username="guest_sensitivity", password="guest_pw")  # noqa: S106
    payload = {"problem_id": 1, "evidence": {}, "target": "f1"}
    response = post_json(client, "/graphical/sensitivity", payload, access_token)
    assert response.status_code == 401


def test_sensitivity_excludes_near_impossible_state_from_values(client: TestClient, session_and_user: dict):
    """A swept state whose own marginal under the base evidence is ~0 is excluded from `values`.

    Here, `B="lo"` has prior probability `1e-9`.
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
            "A": Potential(variables=["A"], shape=[2], values=[0.5, 0.5]),
            "B": Potential(variables=["B"], shape=[2], values=[1e-9, 1 - 1e-9]),
        },
    )
    annotation = DecisionAnnotation(
        objectives=[
            ObjectiveNode(symbol="f1", parents=["A"], utility=Potential(variables=["A"], shape=[2], values=[0.0, 100.0]))
        ]
    )
    _persist_graphical_metadata(session, problem.id, network, annotation)

    payload = {"problem_id": problem.id, "evidence": {}, "target": "f1"}
    response = post_json(client, "/graphical/sensitivity", payload, access_token)
    assert response.status_code == 200, response.text

    results_by_node = {r["node"]: r for r in response.json()["results"]}
    assert set(results_by_node["B"]["values"]) == {"hi"}  # "lo" (prob 1e-9) is excluded


# --- POST /graphical/voi --------------------------------------------------------------------------


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

    A no-decision diagram gets a 400 rather than 0.0.
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

    See `test_sensitivity_bad_evidence_state_label_is_400_not_500`. Uses a second chance node `B`,
    since evidence on `node` itself is rejected earlier.
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


# --- POST /graphical/demo/irrigation, POST /graphical/{problem_id}/solve --------------------------


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

    `heavy` is the hand-verified optimum (see `tests/test_graphical_demo.py`). Exercises the demo
    path end to end: compile, solve, and decode the policy with `/policy`, as `NetworkPanel` does.
    Also cross-checked against `/whatif`'s policy.
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

    `/solve` is gated by visibility, not ownership.
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


def test_solve_without_gurobi_license_never_crashes_on_cvxpy_mip(
    client: TestClient, session_and_user: dict, monkeypatch: pytest.MonkeyPatch
):
    """Without a Gurobi license, `/solve` uses a MIP-capable fallback or returns 400.

    It must not fall back to `CVXPYSolver`, which may lack a MIP backend. Simulates a missing
    license by monkeypatching `check_gurobi_license` in `desdeo.tools.utils`.
    """
    monkeypatch.setattr(tools_utils, "check_gurobi_license", lambda: False)

    access_token = login(client)

    compile_response = post_json(client, "/graphical/demo/irrigation", {}, access_token)
    assert compile_response.status_code == 200, compile_response.text
    problem_id = compile_response.json()["problem_id"]

    response = post_json(client, f"/graphical/{problem_id}/solve", {"objective_symbol": "profit"}, access_token)
    # Either a MIP-capable fallback worked, or none is available.
    assert response.status_code in (200, 400), response.text


def test_solve_non_graphical_problem_is_404(client: TestClient, session_and_user: dict):
    """`/solve` rejects an ordinary (non-graphical) `ProblemDB` -- same 404 every sibling endpoint gives.

    Uses "The river pollution problem", as `test_structure_404_when_no_graphical_metadata` does.
    """
    session: Session = session_and_user["session"]
    access_token = login(client)
    problem = session.exec(select(ProblemDB).where(ProblemDB.name == "The river pollution problem")).first()

    response = post_json(client, f"/graphical/{problem.id}/solve", {"objective_symbol": "f_1"}, access_token)
    assert response.status_code == 404


def test_demo_irrigation_idempotent_same_user_returns_same_problem_id(client: TestClient, session_and_user: dict):
    """Two `/demo/irrigation` calls from the same user reuse the same compiled `problem_id`."""
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

    # A non-forced call afterwards returns the most recent copy.
    third = post_json(client, "/graphical/demo/irrigation", {}, access_token)
    assert third.status_code == 200, third.text
    assert third.json()["problem_id"] == forced.json()["problem_id"]


def test_demo_irrigation_rejects_impostor_same_name_different_diagram(client: TestClient, session_and_user: dict):
    """A same-named `ProblemDB` whose `GraphicalModelMetaData` doesn't match the demo isn't adopted.

    Compiles an unrelated diagram under the demo's name first, then checks that
    `/demo/irrigation` compiles the real demo instead of returning it.
    """
    access_token = login(client)

    impostor_network = _chain_network()
    impostor_annotation = _no_decision_annotation()
    impostor_payload = {
        "network": impostor_network.model_dump(mode="json"),
        "annotation": impostor_annotation.model_dump(mode="json"),
        "name": graphical_router._DEMO_IRRIGATION_PROBLEM_NAME,
    }
    impostor_response = post_json(client, "/graphical/compile", impostor_payload, access_token)
    assert impostor_response.status_code == 200, impostor_response.text
    impostor_problem_id = impostor_response.json()["problem_id"]

    demo_response = post_json(client, "/graphical/demo/irrigation", {}, access_token)
    assert demo_response.status_code == 200, demo_response.text
    demo_problem_id = demo_response.json()["problem_id"]

    # The impostor wasn't adopted.
    assert demo_problem_id != impostor_problem_id

    structure = get_json(client, f"/graphical/{demo_problem_id}/structure", access_token)
    assert structure.status_code == 200
    assert "irrigate" in {n["id"] for n in structure.json()["nodes"]}
