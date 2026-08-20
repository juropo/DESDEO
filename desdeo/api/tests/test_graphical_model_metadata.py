"""Tests for `GraphicalModelMetaData`, the Phase 3 persistence side-table.

Model-level only (mirrors `tests/test_site_selection.py`'s "Model tests" section) -- no
router/endpoint exists yet for this metadata kind (that's Phase 7's job), so this just verifies
the DB model round-trips real `problem/graphical` pydantic models through its JSON columns, the
same pattern `SiteSelectionMetaData` uses for its own JSON blobs.
"""

from sqlmodel import Session, select

from desdeo.api.models import GraphicalModelMetaData, ProblemDB, ProblemMetaDataDB
from desdeo.problem.graphical import (
    DecisionAnnotation,
    DiscreteNetwork,
    DiscreteVariable,
    ObjectiveNode,
    Potential,
    RootedJunctionTree,
    build_rooted_junction_tree,
)


def _network() -> DiscreteNetwork:
    a = DiscreteVariable(name="A", labels=["low", "high"])
    b = DiscreteVariable(name="B", labels=["low", "high"])
    return DiscreteNetwork(
        variables=[a, b],
        arcs=[("A", "B")],
        potentials={
            "A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6]),
            "B": Potential(variables=["B", "A"], shape=[2, 2], values=[[0.9, 0.1], [0.2, 0.8]]),
        },
    )


def _annotation() -> DecisionAnnotation:
    return DecisionAnnotation(
        objectives=[
            ObjectiveNode(symbol="F", parents=["B"], utility=Potential(variables=["B"], shape=[2], values=[0.0, 10.0]))
        ]
    )


def test_graphical_model_metadata_round_trip(session_and_user: dict):
    """Create/store a `DiscreteNetwork` + `DecisionAnnotation` + `RootedJunctionTree`, reload."""
    session: Session = session_and_user["session"]

    problem = session.exec(select(ProblemDB)).first()
    metadata_db = ProblemMetaDataDB(problem_id=problem.id)
    session.add(metadata_db)
    session.commit()
    session.refresh(metadata_db)

    network = _network()
    annotation = _annotation()
    rjt = build_rooted_junction_tree(network, annotation, use_cache=False)

    graphical_metadata = GraphicalModelMetaData(
        metadata_id=metadata_db.id,
        network_json=network.model_dump_json(),
        annotation_json=annotation.model_dump_json(),
        rjt_json=rjt.model_dump_json(),
    )
    session.add(graphical_metadata)
    session.commit()
    session.refresh(graphical_metadata)

    loaded = session.get(GraphicalModelMetaData, graphical_metadata.id)
    assert loaded is not None
    assert loaded.index_map_json is None  # Phase 4 hasn't compiled anything yet

    reloaded_network = DiscreteNetwork.model_validate_json(loaded.network_json)
    reloaded_annotation = DecisionAnnotation.model_validate_json(loaded.annotation_json)
    assert reloaded_network == network
    assert reloaded_annotation == annotation

    reloaded_rjt = RootedJunctionTree.model_validate_json(loaded.rjt_json)
    assert reloaded_rjt == rjt


def test_graphical_model_metadata_appears_in_all_metadata(session_and_user: dict):
    """`ProblemMetaDataDB.all_metadata` picks up `GraphicalModelMetaData` rows too."""
    session: Session = session_and_user["session"]

    problem = session.exec(select(ProblemDB)).first()
    metadata_db = ProblemMetaDataDB(problem_id=problem.id)
    session.add(metadata_db)
    session.commit()
    session.refresh(metadata_db)

    network = _network()
    annotation = _annotation()
    graphical_metadata = GraphicalModelMetaData(
        metadata_id=metadata_db.id,
        network_json=network.model_dump_json(),
        annotation_json=annotation.model_dump_json(),
    )
    session.add(graphical_metadata)
    session.commit()
    session.refresh(metadata_db)

    assert graphical_metadata in metadata_db.all_metadata
