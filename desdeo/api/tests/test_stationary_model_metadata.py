"""Tests for `StationaryModelMetaData` and `add_stationary_lp_problem`."""

from collections import OrderedDict

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from desdeo.api.models import ProblemDB
from desdeo.api.routers import utils
from desdeo.api.routers.utils import copy_problem_metadata, find_stationary_metadata, load_problem
from desdeo.api.utils.stationary import add_stationary_lp_problem
from desdeo.problem import Problem
from desdeo.problem.graphical import (
    ObjectiveNode,
    Potential,
    StationaryDiagram,
    StationaryLPIndexMap,
    compile_stationary_lp,
    full_observation_information_set,
    stationary_payoff_table,
)
from tests.fixtures import decision_chain_2tbn_network

from .conftest import get_json, login, post_json

pytestmark = pytest.mark.graphical


def _objective(symbol: str, parents: list[str], utility: list, *, maximize: bool) -> ObjectiveNode:
    return ObjectiveNode(
        symbol=symbol,
        parents=parents,
        utility=Potential(variables=parents, shape=list(np.shape(utility)), values=utility),
        maximize=maximize,
    )


def _diagram() -> StationaryDiagram:
    """`decision_chain_2tbn_network` with atemporal G and full-observation Dt and Et."""
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


def _add(session_and_user: dict, **kwargs) -> ProblemDB:
    return add_stationary_lp_problem(
        session_and_user["session"], session_and_user["user"], _diagram(), name="Stationary chain", **kwargs
    )


def test_add_stationary_lp_problem_round_trips(session_and_user: dict):
    """The persisted problem loads back as the compiled LP with the payoff table's ideal and nadir."""
    diagram = _diagram()
    problem_db = _add(session_and_user, description="A chain.")

    compiled, _ = compile_stationary_lp(diagram)
    ideal, nadir = stationary_payoff_table(diagram)
    expected = compiled.update_ideal_and_nadir(ideal, nadir).model_copy(
        update={"name": "Stationary chain", "description": "A chain."}
    )
    loaded = Problem.from_problemdb(problem_db)

    # The database keeps scalar and tensor variables apart, so only their order changes.
    def by_symbol(problem: Problem) -> Problem:
        return problem.model_copy(update={"variables": sorted(problem.variables, key=lambda v: v.symbol)})

    assert by_symbol(loaded) == by_symbol(expected)
    assert loaded.get_ideal_point() == ideal
    assert loaded.get_nadir_point() == nadir
    assert all(value is not None for value in [*ideal.values(), *nadir.values()])


def test_stationary_metadata_parses_back(session_and_user: dict):
    """The metadata row holds the diagram and the index map, and shows up in `all_metadata`."""
    diagram = _diagram()
    problem_db = _add(session_and_user)

    meta = find_stationary_metadata(problem_db.id, session_and_user["session"])
    assert meta is not None
    assert meta.metadata_type == "stationary_model_metadata"
    assert meta in problem_db.problem_metadata.all_metadata
    assert StationaryDiagram.model_validate_json(meta.diagram_json) == diagram
    assert StationaryLPIndexMap.model_validate_json(meta.index_map_json) == compile_stationary_lp(diagram)[1]


def test_stationary_metadata_in_problem_info(client: TestClient, session_and_user: dict):
    """`ProblemMetaDataPublic` carries the stationary metadata without the diagram and index map."""
    problem_db = _add(session_and_user)
    token = login(client)

    resp = get_json(client, "/problem/all_info", token)
    assert resp.status_code == 200
    info = next(p for p in resp.json() if p["id"] == problem_db.id)
    rows = info["problem_metadata"]["stationary_model_metadata"]
    assert len(rows) == 1
    assert rows[0].keys() == {"id", "metadata_type"}
    assert rows[0]["metadata_type"] == "stationary_model_metadata"
    assert info["problem_metadata"]["graphical_model_metadata"] == []


def test_solver_selection(session_and_user: dict):
    """`solver` is stored as the preferred solver; an unknown one raises."""
    problem_db = _add(session_and_user, solver="cvxpy")
    solvers = [
        m.solver_string_representation
        for m in problem_db.problem_metadata.all_metadata
        if m.metadata_type == "solver_selection_metadata"
    ]
    assert solvers == ["cvxpy"]

    with pytest.raises(ValueError, match="Unknown solver"):
        _add(session_and_user, solver="no_such_solver")


def test_copy_problem_metadata_copies_stationary(session_and_user: dict):
    """A problem copy gets its own copy of the stationary metadata."""
    session: Session = session_and_user["session"]
    problem_db = _add(session_and_user)
    copy_db = ProblemDB.from_problem(Problem.from_problemdb(problem_db), user=session_and_user["user"])
    session.add(copy_db)
    session.commit()
    session.refresh(copy_db)

    copy_problem_metadata(problem_db, copy_db, session)
    session.commit()

    original = find_stationary_metadata(problem_db.id, session)
    copied = find_stationary_metadata(copy_db.id, session)
    assert copied is not None
    assert copied.id != original.id
    assert (copied.diagram_json, copied.index_map_json) == (original.diagram_json, original.index_map_json)


def test_load_problem_cache(session_and_user: dict):
    """`load_problem` returns the cached `Problem` until the problem's rows change."""
    session: Session = session_and_user["session"]
    problem_db = _add(session_and_user)

    first = load_problem(problem_db)
    assert load_problem(problem_db) is first

    objective_db = problem_db.objectives[0]
    objective_db.ideal = objective_db.ideal - 1.0
    session.add(objective_db)
    session.commit()
    session.refresh(problem_db)

    changed = load_problem(problem_db)
    assert changed is not first
    assert changed == Problem.from_problemdb(problem_db)


def test_load_problem_cache_key_survives_expiry(session_and_user: dict):
    """A freshly fetched row and the same row expired by a commit hit the same cache entry."""
    session: Session = session_and_user["session"]
    problem_id = _add(session_and_user).id
    session.expunge_all()

    first = load_problem(session.get(ProblemDB, problem_id))
    session.commit()
    assert load_problem(session.get(ProblemDB, problem_id)) is first


def test_load_problem_cache_evicts_deleted_problem(session_and_user: dict):
    """Deleting a problem drops its cache entry."""
    session: Session = session_and_user["session"]
    problem_db = _add(session_and_user)
    problem_id = problem_db.id
    load_problem(problem_db)
    assert any(cached_id == problem_id for cached_id, _ in utils._problem_cache.values())

    session.delete(problem_db)
    session.commit()
    assert all(cached_id != problem_id for cached_id, _ in utils._problem_cache.values())


@pytest.mark.cvxpy
def test_cumulus_classification_on_stationary_problem(
    client: TestClient, session_and_user: dict, monkeypatch: pytest.MonkeyPatch
):
    """A CUMULUS classification solve returns solutions without the occupation variables."""
    monkeypatch.setattr(utils, "_problem_cache", OrderedDict())
    problem_db = _add(session_and_user, solver="cvxpy")
    token = login(client)

    resp = post_json(client, "/method/cumulus/get-or-initialize", {"problem_id": problem_db.id}, token)
    assert resp.status_code == 200, resp.text
    init = resp.json()
    assert init["response_type"] == "cumulus.initialization"

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
        },
        token,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["current_solutions"]
    assert "slp_" not in resp.text
    # Initialize and solve load the problem from differently loaded rows but share one cache entry.
    assert [cached_id for cached_id, _ in utils._problem_cache.values()] == [problem_db.id]

    solution = data["current_solutions"][0]
    resp = post_json(
        client,
        "/method/cumulus/finalize",
        {
            "problem_id": problem_db.id,
            "solution_info": {"state_id": solution["state_id"], "solution_index": solution["solution_index"]},
            "parent_state_id": data["state_id"],
        },
        token,
    )
    assert resp.status_code == 200, resp.text
    final_solution = resp.json()["final_solution"]
    assert final_solution["variable_values"] is None
    assert final_solution["objective_values"].keys() == current.keys()
    assert "slp_" not in resp.text

    resp = post_json(client, "/method/cumulus/get-or-initialize", {"problem_id": problem_db.id}, token)
    assert resp.status_code == 200, resp.text
    assert resp.json()["response_type"] == "cumulus.finalize"
    assert resp.json()["final_solution"]["variable_values"] is None
    assert "slp_" not in resp.text
