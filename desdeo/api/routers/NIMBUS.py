"""Router for NIMBUS."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from numpy import allclose
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy.orm import Session

from desdeo.api.db import get_db
from desdeo.api.db_models import Method, Preference, SolutionArchive
from desdeo.api.db_models import Problem as ProblemInDB
from desdeo.api.routers.UserAuth import get_current_user
from desdeo.api.schema import Methods, User
from desdeo.mcdm.nimbus import generate_starting_point, solve_intermediate_solutions, solve_sub_problems
from desdeo.problem.schema import Problem
from desdeo.tools.utils import available_solvers

router = APIRouter(prefix="/nimbus")


class InitRequest(BaseModel):
    """The request to initialize the NIMBUS."""

    problem_id: int = Field(description="The ID of the problem to navigate.")
    method_id: int = Field(description="The ID of the method being used.")


class NIMBUSResponse(BaseModel):
    """The response from most NIMBUS endpoints."""

    objective_symbols: list[str] = Field(description="The symbols of the objectives.")
    objective_long_names: list[str] = Field(description="The names of the objectives.")
    units: list[str | None] | None = Field(description="The units of the objectives.")
    is_maximized: list[bool] = Field(description="Whether the objectives are to be maximized or minimized.")
    lower_bounds: list[float] = Field(description="The lower bounds of the objectives.")
    upper_bounds: list[float] = Field(description="The upper bounds of the objectives.")
    previous_preference: list[float] = Field(description="The previous preference used.")
    current_solutions: list[list[float]] = Field(description="The solutions from the current interation of nimbus.")
    saved_solutions: list[list[float]] = Field(description="The best candidate solutions saved by the decision maker.")
    all_solutions: list[list[float]] = Field(description="All solutions generated by NIMBUS in all iterations.")


class FakeNIMBUSResponse(BaseModel):
    """fake response for testing purposes."""

    message: str = Field(description="A simple message.")


class NIMBUSIterateRequest(BaseModel):
    """The request to iterate the NIMBUS algorithm."""

    problem_id: int = Field(description="The ID of the problem to be solved.")
    method_id: int = Field(description="The ID of the method being used.")
    preference: list[float] = Field(
        description=(
            "The preference as a reference point. Note, NIMBUS uses classification preference,"
            " we can construct it using this reference point and the reference solution."
        )
    )
    reference_solution: list[float] = Field(
        description="The reference solution to be used in the classification preference."
    )
    num_solutions: int | None = Field(
        description="The number of solutions to be generated in the iteration.", default=1
    )


class NIMBUSIntermediateSolutionRequest(BaseModel):
    """The request to generate an intermediate solution in NIMBUS."""

    problem_id: int = Field(description="The ID of the problem to be solved.")
    method_id: int = Field(description="The ID of the method being used.")

    reference_solution_1: list[float] = Field(
        description="The first reference solution to be used in the classification preference."
    )
    reference_solution_2: list[float] = Field(
        description="The reference solution to be used in the classification preference."
    )
    num_solutions: int | None = Field(
        description="The number of solutions to be generated in the iteration.", default=1
    )


class SaveRequest(BaseModel):
    """The request to save the solutions."""

    problem_id: int = Field(description="The ID of the problem to be solved.")
    method_id: int = Field(description="The ID of the method being used.")
    solutions: list[list[float]] = Field(description="The solutions to be saved.")


class ChooseRequest(BaseModel):
    """The request to choose the final solution."""

    problem_id: int = Field(description="The ID of the problem to be solved.")
    method_id: int = Field(description="The ID of the method being used.")
    solution: list[float] = Field(description="The chosen solution.")


@router.post("/initialize")
def init_nimbus(
    init_request: InitRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> NIMBUSResponse | FakeNIMBUSResponse:
    """Initialize the NIMBUS algorithm.

    Args:
        init_request (InitRequest): The request to initialize the NIMBUS.
        user (Annotated[User, Depends(get_current_user)]): The current user.
        db (Annotated[Session, Depends(get_db)]): The database session.

    Returns:
        The response from the NIMBUS algorithm.
    """
    # Do database stuff here.
    problem_id = init_request.problem_id
    # The request is supposed to contain method id, but I don't want to deal with frontend code
    init_request.method_id = get_nimbus_method_id(db)
    method_id = init_request.method_id

    problem, solver = read_problem_from_db(db=db, problem_id=problem_id, user_id=user.index)

    # See if there are previous solutions in the database for this problem
    solutions = read_solutions_from_db(db, problem_id, user.index, method_id)

    # Calculate bounds here, just to make sure that they have been properly defined in the problem
    lower_bounds, upper_bounds = calculate_bounds(problem)

    # If there are no solutions, generate a starting point for NIMBUS
    if not solutions:
        start_result = generate_starting_point(problem=problem, solver=available_solvers[solver] if solver else None)
        save_results_to_db(
            db=db, user_id=user.index, request=init_request, results=[start_result], previous_solutions=solutions
        )
        solutions = read_solutions_from_db(db, problem_id, user.index, method_id)

    # If there is a solution marked as current, use that. Otherwise just use the first solution in the db
    current_solution = next((sol for sol in solutions if sol.current), solutions[0])

    # return FakeNIMBUSResponse(message="NIMBUS initialized.")
    return NIMBUSResponse(
        objective_symbols=[obj.symbol for obj in problem.objectives],
        objective_long_names=[obj.name for obj in problem.objectives],
        units=[obj.unit for obj in problem.objectives],
        is_maximized=[obj.maximize for obj in problem.objectives],
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        previous_preference=current_solution.objectives,
        current_solutions=[current_solution.objectives],
        saved_solutions=[sol.objectives for sol in solutions if sol.saved],
        all_solutions=[sol.objectives for sol in solutions],
    )


@router.post("/iterate")
def iterate(
    request: NIMBUSIterateRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> NIMBUSResponse | FakeNIMBUSResponse:
    """Iterate the NIMBUS algorithm.

    Args:
        request: The request body for a NIMBUS iteration.
        user (Annotated[User, Depends(get_current_user)]): The current user.
        db (Annotated[Session, Depends(get_db)]): The database session.

    Returns:
        The response from the NIMBUS algorithm.
    """
    # Do database stuff here.
    problem_id = request.problem_id
    # The request is supposed to contain method id, but I don't want to deal with frontend code
    request.method_id = get_nimbus_method_id(db)
    method_id = request.method_id

    problem, solver = read_problem_from_db(db=db, problem_id=problem_id, user_id=user.index)

    previous_solutions = read_solutions_from_db(db, problem_id, user.index, method_id)

    if not previous_solutions:
        raise HTTPException(status_code=404, detail="Problem not found in the database.")

    # Calculate bounds here, just to make sure that they have been properly defined in the problem
    lower_bounds, upper_bounds = calculate_bounds(problem)

    # Do NIMBUS stuff here.
    results = solve_sub_problems(
        problem=problem,
        current_objectives=dict(
            zip([obj.symbol for obj in problem.objectives], request.reference_solution, strict=True)
        ),
        reference_point=dict(zip([obj.symbol for obj in problem.objectives], request.preference, strict=True)),
        num_desired=request.num_solutions,
        solver=available_solvers[solver] if solver else None,
    )

    # Do database stuff again.
    save_results_to_db(
        db=db, user_id=user.index, request=request, results=results, previous_solutions=previous_solutions
    )

    solutions = read_solutions_from_db(db, problem_id, user.index, method_id)

    return NIMBUSResponse(
        objective_symbols=[obj.symbol for obj in problem.objectives],
        objective_long_names=[obj.name for obj in problem.objectives],
        units=[obj.unit for obj in problem.objectives],
        is_maximized=[obj.maximize for obj in problem.objectives],
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        previous_preference=request.preference,
        current_solutions=[sol.objectives for sol in solutions if sol.current],
        saved_solutions=[sol.objectives for sol in solutions if sol.saved],
        all_solutions=[sol.objectives for sol in solutions],
    )


@router.post("/intermediate")
def intermediate(
    request: NIMBUSIntermediateSolutionRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> NIMBUSResponse | FakeNIMBUSResponse:
    """Get solutions between two solutions using NIMBUS.

    Args:
        request: The request body for a NIMBUS iteration.
        user (Annotated[User, Depends(get_current_user)]): The current user.
        db (Annotated[Session, Depends(get_db)]): The database session.

    Returns:
        The response from the NIMBUS algorithm.
    """
    # Do database stuff here.
    problem_id = request.problem_id
    # The request is supposed to contain method id, but I don't want to deal with frontend code
    request.method_id = get_nimbus_method_id(db)
    method_id = request.method_id

    problem, solver = read_problem_from_db(db=db, problem_id=problem_id, user_id=user.index)

    previous_solutions = read_solutions_from_db(db, problem_id, user.index, method_id)

    if not previous_solutions:
        raise HTTPException(status_code=404, detail="Problem not found in the database.")

    # Calculate bounds here, just to make sure that they have been properly defined in the problem
    lower_bounds, upper_bounds = calculate_bounds(problem)

    # Do NIMBUS stuff here.
    results = solve_intermediate_solutions(
        problem=problem,
        solution_1=dict(zip(problem.objectives, request.reference_solution_1, strict=True)),
        solution_2=dict(zip(problem.objectives, request.reference_solution_2, strict=True)),
        num_desired=request.num_solutions,
        solver=available_solvers[solver] if solver else None,
    )

    # Do database stuff again.
    save_results_to_db(
        db=db, user_id=user.index, request=request, results=results, previous_solutions=previous_solutions
    )

    solutions = read_solutions_from_db(db, problem_id, user.index, method_id)

    return NIMBUSResponse(
        objective_symbols=[obj.symbol for obj in problem.objectives],
        objective_long_names=[obj.name for obj in problem.objectives],
        units=[obj.unit for obj in problem.objectives],
        is_maximized=[obj.maximize for obj in problem.objectives],
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        previous_preference=request.preference,
        current_solutions=[sol.objectives for sol in solutions if sol.current],
        saved_solutions=[sol.objectives for sol in solutions if sol.saved],
        all_solutions=[sol.objectives for sol in solutions],
    )


@router.post("/save")
def save(
    request: SaveRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> NIMBUSResponse | FakeNIMBUSResponse:
    """Save the solutions to the database.

    Args:
        request: The request body for saving solutions.
        user (Annotated[User, Depends(get_current_user)]): The current user.
        db (Annotated[Session, Depends(get_db)]): The database session.

    Returns:
        The response from the NIMBUS algorithm.
    """
    # Get the solutions from database.
    problem_id = request.problem_id
    method_id = get_nimbus_method_id(db)

    previous_solutions = read_solutions_from_db(db, problem_id, user.index, method_id)

    if not previous_solutions:
        raise HTTPException(status_code=404, detail="Problem not found in the database.")

    # Find the requested solutions and mark them as saved.
    for sol in request.solutions:
        for prev in previous_solutions:
            if allclose(sol, prev.objectives):
                prev.saved = True
    db.commit()

    return NIMBUSResponse(
        objective_symbols=[],
        objective_long_names=[],
        units=[],
        is_maximized=[],
        lower_bounds=[],
        upper_bounds=[],
        previous_preference=[],
        current_solutions=[sol.objectives for sol in previous_solutions if sol.current],
        saved_solutions=[sol.objectives for sol in previous_solutions if sol.saved],
        all_solutions=[sol.objectives for sol in previous_solutions],
    )


@router.post("/choose")
def choose(
    request: ChooseRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[Session, Depends(get_db)],
) -> FakeNIMBUSResponse:
    """Choose a solution as the final solution for NIMBUS.

    Args:
        request: The request body for saving solutions.
        user (Annotated[User, Depends(get_current_user)]): The current user.
        db (Annotated[Session, Depends(get_db)]): The database session.

    Returns:
        The response from the NIMBUS algorithm.
    """
    # Get the solutions from database.
    problem_id = request.problem_id
    method_id = get_nimbus_method_id(db)

    previous_solutions = read_solutions_from_db(db, problem_id, user.index, method_id)

    if not previous_solutions:
        raise HTTPException(status_code=404, detail="Problem not found in the database.")

    # Find the requested solution and mark it as chosen.
    for prev in previous_solutions:
        if allclose(request.solution, prev.objectives):
            prev.chosen = True
            db.commit()
            break
    else:
        raise HTTPException(status_code=404, detail="The chosen solution was not found in the database.")

    return FakeNIMBUSResponse(message="Solution chosen.")


def flatten(lst) -> list[float]:
    """Takes a nested list and flattens it into a single list.

    Args:
        lst: The list that needs flattening

    Returns:
        The flattened list.
    """
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list


def get_nimbus_method_id(db: Session) -> int:
    """Queries the database to find the id for NIMBUS method.

    Args:
        db: Database session

    Returns:
        The method id
    """
    nimbus_method = db.query(Method).filter(Method.kind == Methods.NIMBUS).first()
    return nimbus_method.id


def read_problem_from_db(db: Session, problem_id: int, user_id: int) -> tuple[Problem, str]:
    """Reads the problem from database.

    Args:
        db (Session): Database session to be used
        problem_id (int): Id of the problem
        method_id (int): Id of the method
        user_id (int): Index of the user

    Raises:
        HTTPException: _description_
        HTTPException: _description_
        HTTPException: _description_

    Returns:
        tuple[Problem, str]: Returns the problem as a desdeo problem class and the name of the solver
    """
    problem = db.query(ProblemInDB).filter(ProblemInDB.id == problem_id).first()

    if problem is None:
        raise HTTPException(status_code=404, detail="Problem not found.")
    if problem.owner != user_id and problem.owner is not None:
        raise HTTPException(status_code=403, detail="Unauthorized to access chosen problem.")
    try:
        solver = problem.solver.value
        problem = Problem.model_validate(problem.value)
    except ValidationError:
        raise HTTPException(status_code=500, detail="Error in parsing the problem.") from ValidationError
    return problem, solver


def read_solutions_from_db(db: Session, problem_id: int, user_id: int, method_id: int) -> list[SolutionArchive]:
    """Reads the previous solutions from the database.

    Args:
        db (Session): _description_
        problem_id (int): _description_
        user_id (int): _description_
        method_id (int): _description_

    Returns:
        list[SolutionArchive]: _description_
    """
    return (
        db.query(SolutionArchive)
        .filter(
            SolutionArchive.problem == problem_id, SolutionArchive.user == user_id, SolutionArchive.method == method_id
        )
        .all()
    )


def save_results_to_db(
    db: Session,
    user_id: int,
    request: InitRequest | NIMBUSIterateRequest | NIMBUSIntermediateSolutionRequest,
    results: list,
    previous_solutions: list[SolutionArchive],
):
    """Saves the results to the database.

    Args:
        db (Session): _description_
        user_id (int): _description_
        request (_type_): _description_
        results (list): _description_
        previous_solutions (list[SolutionArchive]): _description_
    """
    problem_id = request.problem_id
    method_id = request.method_id

    if type(request) is InitRequest:
        pref = None
    else:
        pref = Preference(
            user=user_id,
            problem=problem_id,
            method=method_id,
            kind="NIMBUS" if type(type(request) is NIMBUSIterateRequest) else "NIMBUS_intermediate",
            value=request.model_dump(mode="json"),
        )
        db.add(pref)
        db.commit()

    # See if the results include duplicates and remove them
    duplicate_indices = set()
    for i in range(len(results) - 1):
        for j in range(i + 1, len(results)):
            if allclose(list(results[i].optimal_objectives.values()), list(results[j].optimal_objectives.values())):
                duplicate_indices.add(j)

    for index in sorted(duplicate_indices, reverse=True):
        results.pop(index)

    old_current_solutions = (
        db.query(SolutionArchive)
        .filter(SolutionArchive.problem == problem_id, SolutionArchive.user == user_id, SolutionArchive.current)
        .all()
    )

    # Mark all the old solutions as not current
    for old in old_current_solutions:
        old.current = False

    for res in results:
        # Check if the results already exist in the database
        duplicate = False
        for prev in previous_solutions:
            if allclose(list(res.optimal_objectives.values()), list(prev.objectives)):
                prev.current = True
                duplicate = True
                break
        # If the solution was not found in the database, add it
        if not duplicate:
            db.add(
                SolutionArchive(
                    user=user_id,
                    problem=problem_id,
                    method=method_id,
                    preference=pref.id if pref is not None else None,
                    decision_variables=flatten(list(res.optimal_variables.values())),
                    objectives=list(res.optimal_objectives.values()),
                    saved=False,
                    current=True,
                    chosen=False,
                )
            )
    db.commit()


def calculate_bounds(problem: Problem) -> tuple[list[float, list[float]]]:
    """Calculates upper and lower bounds for the objectives.

    Args:
        problem (Problem): _description_

    Raises:
        HTTPException: _description_

    Returns:
        tuple[list[float, list[float]]]: tuple containing a list of lower bound values and a list of upper bound values
    """
    ideal = problem.get_ideal_point()
    nadir = problem.get_nadir_point()
    if None in ideal or None in nadir:
        raise HTTPException(status_code=500, detail="Problem missing ideal or nadir value.")

    lower_bounds = [0.0 for x in range(len(problem.objectives))]
    upper_bounds = [0.0 for x in range(len(problem.objectives))]
    for i in range(len(problem.objectives)):
        if problem.objectives[i].maximize:
            lower_bounds[i] = nadir[problem.objectives[i].symbol]
            upper_bounds[i] = ideal[problem.objectives[i].symbol]
        else:
            lower_bounds[i] = ideal[problem.objectives[i].symbol]
            upper_bounds[i] = nadir[problem.objectives[i].symbol]

    return lower_bounds, upper_bounds
