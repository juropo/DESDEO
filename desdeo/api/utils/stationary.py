"""Persisting problems compiled from stationary 2TBN diagrams."""

from typing import Literal

from sqlmodel import Session

from desdeo.api.models import ProblemDB, User
from desdeo.api.models.problem import ProblemMetaDataDB, SolverSelectionMetadata, StationaryModelMetaData
from desdeo.problem.graphical import StationaryDiagram, compile_stationary_lp, stationary_payoff_table
from desdeo.tools.utils import available_solvers


def add_stationary_lp_problem(
    session: Session,
    user: User,
    diagram: StationaryDiagram,
    *,
    name: str,
    description: str | None = None,
    solver: str | None = None,
    atemporal: Literal["randomised", "deterministic"] = "randomised",
) -> ProblemDB:
    """Compile `diagram` with `compile_stationary_lp` and persist it with its `StationaryModelMetaData`.

    The ideal and nadir come from `stationary_payoff_table`. Everything is added in one commit.

    Args:
        session: the database session.
        user: the owner of the problem.
        diagram: the stationary diagram.
        name: the problem's name.
        description: the problem's description; the compiled problem's description if `None`.
        solver: a key of `available_solvers` stored as the problem's preferred solver, if given.
        atemporal: passed to `compile_stationary_lp`.

    Returns:
        ProblemDB: the persisted problem.

    Raises:
        ValueError: `solver` isn't a key of `available_solvers`, or `compile_stationary_lp` raises.
    """
    if solver is not None and solver not in available_solvers:
        msg = f"Unknown solver {solver!r}; expected one of {sorted(available_solvers)}."
        raise ValueError(msg)

    problem, index_map = compile_stationary_lp(diagram, atemporal=atemporal)
    ideal, nadir = stationary_payoff_table(diagram)
    problem = problem.update_ideal_and_nadir(ideal, nadir).model_copy(
        update={"name": name, "description": description if description is not None else problem.description}
    )

    # `flush()` assigns ids for the FKs without committing, so a failure rolls everything back.
    problem_db = ProblemDB.from_problem(problem, user=user)
    session.add(problem_db)
    session.flush()

    metadata_db = ProblemMetaDataDB(problem_id=problem_db.id)
    session.add(metadata_db)
    session.flush()

    session.add(
        StationaryModelMetaData(
            metadata_id=metadata_db.id,
            diagram_json=diagram.model_dump_json(),
            index_map_json=index_map.model_dump_json(),
        )
    )
    if solver is not None:
        session.add(SolverSelectionMetadata(metadata_id=metadata_db.id, solver_string_representation=solver))
    session.commit()
    session.refresh(problem_db)
    return problem_db
