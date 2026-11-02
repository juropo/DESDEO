"""Defines end-points to access functionalities related to the CUMULUS method."""

import logging
from dataclasses import dataclass
from math import isfinite
from typing import Annotated, Any

import numpy as np
import polars as pl
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from sqlmodel import Session, select

from desdeo.api.db import engine, get_session
from desdeo.api.models import (
    CumulusFinalState,
    CumulusInitializationState,
    CumulusModificationState,
    CumulusObjectiveConstraintState,
    CumulusSaveState,
    IntermediateSolutionRequest,
    IntermediateSolutionState,
    ReferencePoint,
    StateDB,
    User,
    UserSavedSolutionDB,
)
from desdeo.api.models.cumulus import (
    CumulusActiveObjectiveConstraintsResponse,
    CumulusClassificationRequest,
    CumulusClassificationResponse,
    CumulusDeleteSaveRequest,
    CumulusDeleteSaveResponse,
    CumulusFinalizeRequest,
    CumulusFinalizeResponse,
    CumulusHypotheticalRequest,
    CumulusInitializationRequest,
    CumulusInitializationResponse,
    CumulusIntermediateSolutionResponse,
    CumulusModificationRequest,
    CumulusModificationResponse,
    CumulusObjectiveConstraintInfo,
    CumulusObjectiveConstraintRequest,
    CumulusObjectiveConstraintResponse,
    CumulusSaveRequest,
    CumulusSaveResponse,
    CumulusScenarioAvailabilityResponse,
    CumulusScenarioSetupResponse,
    ProblemModification,
    UncertaintyMeasureSpec,
    UncertaintyMeasureType,
    UncertaintySource,
)
from desdeo.api.models.generic import SolutionInfo
from desdeo.api.models.generic_states import SolutionReference, SolutionReferenceResponse, StateKind
from desdeo.api.models.problem import (
    ConstraintDB,
    DescriptionPart,
    GraphicalModelMetaData,
    ProblemDB,
    SolutionDescriptionMetaData,
)
from desdeo.api.models.scenario import ScenarioModelDB
from desdeo.api.models.state import CumulusClassificationState
from desdeo.api.routers.graphical import _fetch_graphical_metadata
from desdeo.api.routers.problem import check_solver
from desdeo.api.routers.solution_description import _substitute_values, validate_description_parts
from desdeo.api.routers.user_authentication import get_current_user
from desdeo.mcdm.cumulus import (
    CumulusScalarization,
    _minimize_soft_constraint_violations,
    generate_starting_point,
    soft_constraint_violations,
    solve_sub_problems,
)
from desdeo.mcdm.reference_point_method import rpm_intermediate_solutions
from desdeo.problem import PolarsEvaluator, Problem, ScenarioModel
from desdeo.problem.graphical import (
    DecisionAnnotation,
    DiagramAssumptions,
    DiscreteNetwork,
    compile_with_assumptions,
)
from desdeo.problem.json_parser import FormatEnum, MathParser
from desdeo.problem.schema import Constraint, ConstraintTypeEnum, TensorVariable
from desdeo.tools import SolverResults, guess_best_solver, is_heuristic_solver
from desdeo.tools.utils import payoff_table_method

from .utils import (
    ContextField,
    SessionContext,
    SessionContextGuard,
    apply_problem_modifications,
    collect_all_solutions,
    collect_saved_solutions,
    copy_problem_metadata,
    determine_unstated_constraint_properties,
    find_graphical_metadata,
    get_solver_results_at,
    graphical_outcome_distributions,
    iter_states_of_kinds,
    leaves_properties_unstated,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/method/cumulus")

# Constraint values are normalised to `func <= 0` (or `== 0`), so a small positive value is
# rounding rather than a genuine violation.
_CONSTRAINT_TOLERANCE = 1e-6


def _get_scenario_model_id(
    problem_id: int,
    session_id: int | None,
    db_session: Session,
) -> int | None:
    """Return the scenario_model_id from the most recent CUMULUS init or solve state that has one."""
    for state in iter_states_of_kinds(
        problem_id, session_id, db_session, (StateKind.CUMULUS_INIT, StateKind.CUMULUS_SOLVE)
    ):
        sm_id = getattr(state, "scenario_model_id", None)
        if sm_id is not None:
            return sm_id
    return None


_MAX_PARENT_HOPS = 10

_SCALARIZATION_NAMES = {
    CumulusScalarization.CUMULONIMBUS: "Cumulonimbus",
    CumulusScalarization.ASF_PARTIAL: "ASF (partial)",
}

_MEASURE_PREFIXES = {
    "E_": "expected_value",
    "robust_": "worst_case_robust",
    "CVAR_": "conditional_value_at_risk",
    "weighted_": "weighted_scenarios",
}


_MEASURE_TYPES: dict[UncertaintySource, list[UncertaintyMeasureType]] = {
    "scenario_model": ["expected_value", "worst_case_robust", "conditional_value_at_risk", "weighted_scenarios"],
    "graphical": ["expected_value", "worst_case_robust", "conditional_value_at_risk"],
}


@dataclass(frozen=True)
class _UncertaintySource:
    """What uncertainty measures for a problem are taken over, and the problem they are applied to."""

    base_problem_db: ProblemDB
    objective_symbols: list[str]
    scenario_model_db: ScenarioModelDB | None = None
    graphical_meta: GraphicalModelMetaData | None = None

    @property
    def source(self) -> UncertaintySource:
        return "scenario_model" if self.scenario_model_db is not None else "graphical"


def _resolve_uncertainty_source(
    problem_db: ProblemDB, db_session: Session, *, strict: bool = False
) -> _UncertaintySource | None:
    """Walk up `parent_problem_id` to the problem (if any) that uncertainty measures are applied to.

    That is the problem carrying a scenario model, or a graphical-model problem with an objective
    that has an outcome distribution. A problem whose objectives are all aggregates is passed
    over, so this finds the base problem whether `problem_db` is the original problem or one
    already combined from it. A problem whose outcome distributions can't be read is logged and
    passed over too.

    Raises:
        HTTPException 422: `strict` is set and `problem_db`'s outcome distributions can't be read.
    """
    current: ProblemDB | None = problem_db
    for _ in range(_MAX_PARENT_HOPS):
        if current is None:
            return None
        if current.scenario_models:
            scenario_model_db = current.scenario_models[0]
            # Include objectives from both the base problem and the scenario pool (e.g. f_3).
            symbols = [obj.symbol for obj in current.objectives] + [obj.symbol for obj in scenario_model_db.objectives]
            return _UncertaintySource(current, list(dict.fromkeys(symbols)), scenario_model_db=scenario_model_db)
        meta = find_graphical_metadata(current.id, db_session)
        if meta is not None:
            try:
                _, distributions = graphical_outcome_distributions(Problem.from_problemdb(current), meta)
            except ValueError as e:
                if strict and current is problem_db:
                    raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
                logger.warning("Outcome distributions of problem %s can't be read: %s", current.id, e)
                distributions = {}
            symbols = [obj.symbol for obj in current.objectives if obj.symbol in distributions]
            if symbols:
                return _UncertaintySource(current, symbols, graphical_meta=meta)
        current = db_session.get(ProblemDB, current.parent_problem_id) if current.parent_problem_id else None
    return None


def _detect_applied_measures(problem_db: ProblemDB) -> list[str]:
    """Best-effort detection of which uncertainty measures are already on a problem's objectives."""
    found: set[str] = set()
    for obj in problem_db.objectives or []:
        for prefix, measure in _MEASURE_PREFIXES.items():
            if obj.symbol.startswith(prefix):
                found.add(measure)
    return sorted(found)


def _retarget_description_parts(raw_parts: list[dict], problem: Problem, renamed: dict[str, str]) -> list[dict]:
    """Point a solution description's parts at `problem`, renaming symbols by `renamed`.

    Parts still referring to symbols the problem lacks are left out, and so is a text part all
    of whose value parts (up to the next text part) are left out.
    """
    parts: list[dict] = []
    # Whether a value part since the last text part was left out.
    dropped = False
    for raw in raw_parts:
        part = DescriptionPart.model_validate(raw)
        if part.text is not None:
            if dropped and parts and parts[-1]["text"] is not None:
                parts.pop()
            dropped = False
            parts.append(part.model_dump())
            continue
        part = part.model_copy(
            update={
                "symbol": renamed.get(part.symbol, part.symbol),
                "expression": _substitute_values(part.expression, renamed),
            }
        )
        try:
            validate_description_parts([part], problem)
        except ValueError:
            dropped = True
            continue
        parts.append(part.model_dump())
    if dropped and parts and parts[-1]["text"] is not None:
        parts.pop()
    return parts


def _retargeted_solution_descriptions(
    base_problem_db: ProblemDB, problem: Problem, specs: list[UncertaintyMeasureSpec], db_session: Session
) -> list[tuple[list[dict], str]]:
    """Solution descriptions for a graphical aggregate `problem` built from `base_problem_db`.

    The description is taken from the problem the aggregates are built on, so parts left out of
    an earlier aggregate problem come back. A replaced objective is the expected value of its
    outcome, so a part on it reads the objective's expected-value aggregate where there is one.
    """
    source = _resolve_uncertainty_source(base_problem_db, db_session)
    source_metadata = source.base_problem_db.problem_metadata if source is not None else None
    metadata = source_metadata or base_problem_db.problem_metadata
    source_items = (metadata.solution_description_metadata or []) if metadata is not None else []
    renamed = {
        symbol: f"{spec.prefix or 'E_'}{symbol}"
        for spec in specs
        if spec.measure_type == "expected_value"
        for symbol in spec.symbols
    }
    return [(_retarget_description_parts(item.parts or [], problem, renamed), item.separator) for item in source_items]


def _replace_solution_descriptions(
    problem_db: ProblemDB, descriptions: list[tuple[list[dict], str]], db_session: Session
) -> None:
    """Replace the solution descriptions of `problem_db` with `descriptions`."""
    db_session.refresh(problem_db)
    metadata = problem_db.problem_metadata
    if metadata is None:
        return
    for item in metadata.solution_description_metadata or []:
        db_session.delete(item)
    for parts, separator in descriptions:
        db_session.add(SolutionDescriptionMetaData(metadata_id=metadata.id, parts=parts, separator=separator))
    db_session.commit()


def _build_combined_problem_and_initialize(
    request: CumulusInitializationRequest,
    context: SessionContext,
    base_problem_db: ProblemDB,
    parent_state: StateDB | None,
) -> CumulusInitializationResponse:
    """Build a combined problem from `request.uncertainty_measures` and initialize CUMULUS on it."""
    db_session = context.db_session
    user = context.user
    interactive_session = context.interactive_session

    problem = Problem.from_problemdb(base_problem_db)
    mods = ProblemModification(uncertainty_measures=request.uncertainty_measures)
    graphical_meta = find_graphical_metadata(base_problem_db.id, db_session)
    try:
        combined_problem = apply_problem_modifications(problem, mods, db_session, graphical_meta)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to build combined scenario problem: {e}",
        ) from e

    # Estimate ideal/nadir via payoff table so scalarizations are well-scaled.
    solver = check_solver(problem_db=base_problem_db)
    try:
        estimated_ideal, estimated_nadir = payoff_table_method(combined_problem, solver)
        combined_problem = combined_problem.update_ideal_and_nadir(estimated_ideal, estimated_nadir)
    except Exception:  # noqa: S110
        pass  # proceed without ideal/nadir if payoff table fails

    descriptions = (
        _retargeted_solution_descriptions(base_problem_db, combined_problem, request.uncertainty_measures, db_session)
        if graphical_meta is not None
        else None
    )

    new_problem_db = ProblemDB.from_problem(combined_problem, user=user)
    new_problem_db.parent_problem_id = base_problem_db.id
    if request.name:
        new_problem_db.name = request.name
    original_desc = base_problem_db.description or ""
    combined_desc = new_problem_db.description or ""
    if original_desc and combined_desc != original_desc:
        new_problem_db.description = original_desc + "\n\n" + combined_desc
    db_session.add(new_problem_db)
    db_session.commit()
    db_session.refresh(new_problem_db)
    copy_problem_metadata(base_problem_db, new_problem_db, db_session)
    db_session.commit()
    if descriptions is not None:
        _replace_solution_descriptions(new_problem_db, descriptions, db_session)

    new_context = SessionContext(
        user=user,
        db_session=db_session,
        problem_db=new_problem_db,
        interactive_session=interactive_session,
        parent_state=parent_state,
    )
    new_request = request.model_copy(
        update={
            "problem_id": new_problem_db.id,
            "original_problem_id": request.original_problem_id or base_problem_db.id,
            "uncertainty_measures": None,
        }
    )
    return initialize(new_request, new_context)


def _load_scenario_model(
    scenario_model_id: int,
    db_session: Session,
) -> ScenarioModel | None:
    """Load a :class:`ScenarioModel` from the DB given its id."""
    sm_db = db_session.get(ScenarioModelDB, scenario_model_id)
    if sm_db is None or sm_db.base_problem is None:
        return None
    return sm_db.to_scenario_model(base_problem=Problem.from_problemdb(sm_db.base_problem))


def _get_objective_constraint_ids(
    problem_id: int,
    session_id: int | None,
    db_session: Session,
) -> list[int]:
    """Return the soft constraint ids from the latest objective-constraint state."""
    for state in iter_states_of_kinds(problem_id, session_id, db_session, (StateKind.CUMULUS_OBJ_CONSTRAINT,)):
        if isinstance(state, CumulusObjectiveConstraintState):
            return state.soft_constraint_ids
    return []


@router.post("/solve")
def solve_solutions(
    request: CumulusClassificationRequest,
    context: Annotated[SessionContext, Depends(SessionContextGuard(require=[ContextField.PROBLEM]).post)],
) -> CumulusClassificationResponse:
    """Solve the problem using the CUMULUS method."""
    db_session = context.db_session
    user = context.user
    problem_db = context.problem_db
    interactive_session = context.interactive_session
    parent_state = context.parent_state

    solver = check_solver(problem_db=problem_db)
    problem = Problem.from_problemdb(problem_db)

    try:
        scalarizations = [CumulusScalarization(s) for s in request.scalarizations]
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e

    soft_ids = _get_objective_constraint_ids(
        problem_db.id, interactive_session.id if interactive_session else None, db_session
    )
    soft_constraints = [
        Constraint.model_validate(row, from_attributes=True)
        for row in db_session.exec(select(ConstraintDB).where(ConstraintDB.id.in_(soft_ids))).all()
    ] or None

    session_id = interactive_session.id if interactive_session else None
    scenario_model_id = _get_scenario_model_id(problem_db.id, session_id, db_session)
    scenario_model = _load_scenario_model(scenario_model_id, db_session) if scenario_model_id is not None else None

    results_by_scalarization = solve_sub_problems(
        problem=problem,
        current_objectives=request.current_objectives,
        reference_point=request.preference.aspiration_levels,
        scalarizations=scalarizations,
        scalarization_options=request.scalarization_options,
        solver=solver,
        solver_options=request.solver_options,
        soft_constraints=soft_constraints,
        scenario_model=scenario_model,
    )
    # Filter out infeasible scalarizations (None values); store only valid results.
    solver_results: list[SolverResults] = [r for r in results_by_scalarization.values() if r is not None]

    warnings: list[str] = []
    for sf in scalarizations:
        if results_by_scalarization.get(sf) is not None:
            continue
        if sf == CumulusScalarization.CUMULONIMBUS and soft_ids:
            # Cumulonimbus keeps every objective within what the classification allows, so it fails
            # when no solution there meets the soft constraints as closely as they can be met at all.
            warnings.append(
                "Cumulonimbus found no solution. Your reference point does not let some objectives get "
                "worse than in the current solution, and no solution within those limits meets your "
                f"{len(soft_ids)} soft objective constraint(s) as closely as they can be met. To get a "
                "Cumulonimbus solution, allow those objectives to get worse in the reference point, or "
                "loosen or remove the constraints."
            )
        else:
            warning = (
                f"The {_SCALARIZATION_NAMES[sf]} scalarization found no solution: the solver could not solve it, "
                "so it may be infeasible or unbounded for this reference point."
            )
            if sf == CumulusScalarization.CUMULONIMBUS:
                warning += " If every objective may worsen freely, ask at least one objective to improve."
            warnings.append(warning)

    # The largest relaxation of each soft constraint over the returned solutions.
    violated: dict[str, float] = {}
    for result in solver_results:
        for symbol, amount in soft_constraint_violations(result, soft_constraints or []).items():
            if amount > _CONSTRAINT_TOLERANCE:
                violated[symbol] = max(violated.get(symbol, 0.0), amount)
    if violated:
        names = {c.symbol: f"{c.name} {c.cons_type.value} 0" for c in soft_constraints}
        details = ", ".join(f"{names[symbol]} (by {amount:.4g})" for symbol, amount in violated.items())
        warnings.append(
            "Your soft objective constraints could not all be met, so they were relaxed as little as "
            f"possible. Violated: {details}."
        )

    cumulus_state = CumulusClassificationState(
        preferences=request.preference,
        scalarization_options=request.scalarization_options,
        solver=request.solver,
        solver_options=request.solver_options,
        solver_results=solver_results,
        current_objectives=request.current_objectives,
        scalarizations=request.scalarizations,
        original_problem_id=request.original_problem_id,
        scenario_model_id=scenario_model_id,
    )

    state = StateDB.create(
        database_session=db_session,
        problem_id=problem_db.id,
        session_id=interactive_session.id if interactive_session is not None else None,
        parent_id=parent_state.id if parent_state is not None else None,
        state=cumulus_state,
    )

    db_session.add(state)
    db_session.commit()
    db_session.refresh(state)

    current_solutions: list[SolutionReference] = [
        SolutionReference(state=state, solution_index=i) for i, _ in enumerate(solver_results)
    ]
    saved_solutions = collect_saved_solutions(user, request.problem_id, db_session)
    all_solutions = collect_all_solutions(user, request.problem_id, db_session)

    return CumulusClassificationResponse(
        state_id=state.id,
        scenario_model_id=scenario_model_id,
        previous_preference=request.preference,
        previous_objectives=request.current_objectives,
        current_solutions=current_solutions,
        saved_solutions=saved_solutions,
        all_solutions=all_solutions,
        warnings=warnings,
    )


@router.post("/initialize")
def initialize(
    request: CumulusInitializationRequest,
    context: Annotated[SessionContext, Depends(SessionContextGuard(require=[ContextField.PROBLEM]).post)],
) -> CumulusInitializationResponse:
    """Initialize the problem for the CUMULUS method."""
    db_session = context.db_session
    user = context.user
    problem_db = context.problem_db
    interactive_session = context.interactive_session
    parent_state = context.parent_state

    solver = check_solver(problem_db=problem_db)
    problem = Problem.from_problemdb(problem_db)

    # Tracked per field, so the fallback below never overwrites a value that is already set.
    missing_ideal = {symbol for symbol, value in problem.get_ideal_point().items() if value is None}
    missing_nadir = {symbol for symbol, value in problem.get_nadir_point().items() if value is None}
    if missing_ideal or missing_nadir:
        # Some problems (e.g. ones compiled from a graphical model) have no ideal/nadir, which
        # `generate_starting_point` requires. Estimate them with a payoff table and persist the
        # result, as `_build_combined_problem_and_initialize` does. The estimate depends only on
        # the problem, so it is persisted regardless of who owns the problem.
        try:
            estimated_ideal, estimated_nadir = payoff_table_method(problem, solver, require_success=True)
        except Exception:  # noqa: S110
            pass  # proceed without ideal/nadir -- generate_starting_point raises its own clean error
        else:
            # Only fill in missing values, and skip non-finite estimates: a NaN would never be
            # retried, since it isn't None. Skipped symbols stay None for a later attempt.
            new_ideal = {
                symbol: value
                for symbol, value in estimated_ideal.items()
                if symbol in missing_ideal and isfinite(value)
            }
            new_nadir = {
                symbol: value
                for symbol, value in estimated_nadir.items()
                if symbol in missing_nadir and isfinite(value)
            }
            problem = problem.update_ideal_and_nadir(new_ideal, new_nadir)

            any_changed = False
            for obj_db in problem_db.objectives:
                changed = False
                if obj_db.symbol in new_ideal:
                    obj_db.ideal = new_ideal[obj_db.symbol]
                    changed = True
                if obj_db.symbol in new_nadir:
                    obj_db.nadir = new_nadir[obj_db.symbol]
                    changed = True
                if changed:
                    db_session.add(obj_db)
                    any_changed = True
            if any_changed:
                db_session.commit()

    if isinstance(ref_point := request.starting_point, ReferencePoint):
        starting_point = ref_point.aspiration_levels
    elif isinstance(info := request.starting_point, SolutionInfo):
        state = db_session.exec(select(StateDB).where(StateDB.id == info.state_id)).first()
        if state is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"StateDB with index {info.state_id} could not be found.",
            )
        starting_point = state.state.result_objective_values[info.solution_index]
    else:
        starting_point = None

    start_result = generate_starting_point(
        problem=problem,
        reference_point=starting_point,
        scalarization_options=request.scalarization_options,
        solver=solver,
        solver_options=request.solver_options,
    )

    effective_original_problem_id = (
        request.original_problem_id if request.original_problem_id is not None else request.problem_id
    )

    # Only associate a scenario model when a combined problem was actually built from one,
    # i.e., when the problem being initialized differs from the original problem.
    _scenario_model_id: int | None = None
    if effective_original_problem_id != request.problem_id:
        _orig_db = db_session.get(ProblemDB, effective_original_problem_id)
        if _orig_db is not None and _orig_db.scenario_models:
            _scenario_model_id = _orig_db.scenario_models[0].id

    initialization_state = CumulusInitializationState(
        reference_point=starting_point,
        scalarization_options=request.scalarization_options,
        solver=request.solver,
        solver_results=start_result,
        original_problem_id=effective_original_problem_id,
        scenario_model_id=_scenario_model_id,
    )

    state = StateDB.create(
        database_session=db_session,
        problem_id=problem_db.id,
        session_id=interactive_session.id if interactive_session is not None else None,
        parent_id=parent_state.id if parent_state is not None else None,
        state=initialization_state,
        kind=StateKind.CUMULUS_INIT,
    )

    db_session.add(state)
    db_session.commit()
    db_session.refresh(state)

    current_solutions = [SolutionReference(state=state, solution_index=0)]
    saved_solutions = collect_saved_solutions(user, request.problem_id, db_session)
    all_solutions = collect_all_solutions(user, request.problem_id, db_session)
    session_id = interactive_session.id if interactive_session is not None else None
    soft_ids = _get_objective_constraint_ids(problem_db.id, session_id, db_session)

    return CumulusInitializationResponse(
        state_id=state.id,
        problem_id=problem_db.id,
        current_solutions=current_solutions,
        saved_solutions=saved_solutions,
        all_solutions=all_solutions,
        soft_constraint_ids=soft_ids,
    )


@router.post("/save")
def save(
    request: CumulusSaveRequest,
    context: Annotated[SessionContext, Depends(SessionContextGuard().post)],
) -> CumulusSaveResponse:
    """Save solutions."""
    db_session = context.db_session
    user = context.user
    interactive_session = context.interactive_session
    parent_state = context.parent_state

    if request.parent_state_id is None:
        parent_state = (
            interactive_session.states[-1]
            if (interactive_session is not None and len(interactive_session.states) > 0)
            else None
        )
    else:
        parent_state = db_session.exec(select(StateDB).where(StateDB.id == request.parent_state_id)).first()
        if parent_state is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not find state with id={request.parent_state_id}",
            )

    updated_solutions: list[UserSavedSolutionDB] = []
    new_solutions: list[UserSavedSolutionDB] = []

    for info in request.solution_info:
        existing_solution = db_session.exec(
            select(UserSavedSolutionDB).where(
                UserSavedSolutionDB.origin_state_id == info.state_id,
                UserSavedSolutionDB.solution_index == info.solution_index,
            )
        ).first()

        if existing_solution is not None:
            existing_solution.name = info.name
            db_session.add(existing_solution)
            updated_solutions.append(existing_solution)
        else:
            new_solution = UserSavedSolutionDB.from_state_info(
                db_session, user.id, request.problem_id, info.state_id, info.solution_index, info.name
            )
            db_session.add(new_solution)
            new_solutions.append(new_solution)

    if updated_solutions or new_solutions:
        db_session.commit()
        [db_session.refresh(row) for row in updated_solutions + new_solutions]

    save_state = CumulusSaveState(
        solutions=updated_solutions + new_solutions,
        original_problem_id=request.original_problem_id,
    )

    state = StateDB.create(
        database_session=db_session,
        problem_id=request.problem_id,
        session_id=interactive_session.id if interactive_session is not None else None,
        parent_id=parent_state.id if parent_state is not None else None,
        state=save_state,
        kind=StateKind.CUMULUS_SAVE,
    )

    db_session.add(state)
    db_session.commit()
    db_session.refresh(state)

    return CumulusSaveResponse(state_id=state.id)


@router.post("/get-or-initialize")
def get_or_initialize(
    request: CumulusInitializationRequest,
    context: Annotated[SessionContext, Depends(SessionContextGuard(require=[ContextField.PROBLEM]).post)],
) -> (
    CumulusInitializationResponse
    | CumulusClassificationResponse
    | CumulusFinalizeResponse
    | CumulusScenarioSetupResponse
):
    """Get the latest CUMULUS state if it exists, or initialize a new one if it doesn't.

    If the problem has associated scenarios (a scenario model, or a graphical model whose
    objectives are not yet aggregated) and no uncertainty measures have been specified, a
    ``CumulusScenarioSetupResponse`` is returned so the frontend can ask the decision maker which
    aggregates to include.  When the frontend re-calls with ``uncertainty_measures`` populated, a
    combined problem is built, saved to the database, and used for initialization.
    """
    db_session = context.db_session
    user = context.user
    interactive_session = context.interactive_session
    problem_db = context.problem_db

    statement = (
        select(StateDB)
        .where(
            StateDB.problem_id == request.problem_id,
            StateDB.session_id == (interactive_session.id if interactive_session else user.active_session_id),
        )
        .order_by(StateDB.id.desc())
    )
    states = db_session.exec(statement).all()

    # Save states hold the saved solutions, not the current ones, so they are skipped.
    latest_state = None
    for state in states:
        if state.base_state is not None and state.base_state.kind in (
            StateKind.CUMULUS_SOLVE,
            StateKind.CUMULUS_INIT,
            StateKind.CUMULUS_FINAL,
        ):
            latest_state = state
            break

    if latest_state is not None:
        saved_solutions = collect_saved_solutions(user, request.problem_id, db_session)
        all_solutions = collect_all_solutions(user, request.problem_id, db_session)

        current_solutions = [
            SolutionReference(state=latest_state, solution_index=i) for i in range(latest_state.state.num_solutions)
        ]

        if isinstance(latest_state.state, CumulusClassificationState):
            return CumulusClassificationResponse(
                state_id=latest_state.id,
                previous_preference=latest_state.state.preferences,
                previous_objectives=latest_state.state.current_objectives,
                current_solutions=current_solutions,
                saved_solutions=saved_solutions,
                all_solutions=all_solutions,
            )

        if isinstance(latest_state.state, CumulusFinalState):
            solution_index = latest_state.state.solution_result_index
            origin_state_id = latest_state.state.solution_origin_state_id

            final_solution_ref_res = SolutionReferenceResponse(
                solution_index=solution_index,
                state_id=origin_state_id,
                objective_values=latest_state.state.solver_results.optimal_objectives,
                variable_values=latest_state.state.solver_results.optimal_variables,
            )

            return CumulusFinalizeResponse(
                state_id=latest_state.id,
                final_solution=final_solution_ref_res,
                saved_solutions=saved_solutions,
                all_solutions=all_solutions,
            )

        # CumulusInitializationState
        _session_id = interactive_session.id if interactive_session is not None else user.active_session_id
        soft_ids = _get_objective_constraint_ids(request.problem_id, _session_id, db_session)
        return CumulusInitializationResponse(
            state_id=latest_state.id,
            current_solutions=current_solutions,
            saved_solutions=saved_solutions,
            all_solutions=all_solutions,
            soft_constraint_ids=soft_ids,
        )

    # No existing state — check whether the problem itself is the one measures are applied to.
    uncertainty_source = _resolve_uncertainty_source(problem_db, db_session)
    if uncertainty_source is not None and uncertainty_source.base_problem_db.id == problem_db.id:
        if request.uncertainty_measures is None:
            # Ask the frontend which uncertainty aggregates the DM wants to include.
            scenario_model_db = uncertainty_source.scenario_model_db
            return CumulusScenarioSetupResponse(
                source=uncertainty_source.source,
                scenario_model_id=scenario_model_db.id if scenario_model_db is not None else None,
                objective_symbols=uncertainty_source.objective_symbols,
                measure_types=_MEASURE_TYPES[uncertainty_source.source],
            )

        if not request.uncertainty_measures:
            # DM explicitly declined to use scenarios — initialize with the original problem.
            return initialize(request, context)

        # The DM has chosen measures — build and persist the combined problem.
        return _build_combined_problem_and_initialize(request, context, problem_db, context.parent_state)

    return initialize(request, context)


@router.get("/scenario-availability")
def get_scenario_availability(
    context: Annotated[SessionContext, Depends(SessionContextGuard(require=[ContextField.PROBLEM]).get)],
) -> CumulusScenarioAvailabilityResponse:
    """Report whether uncertainty measures can be added to (or changed on) a problem.

    Looks past any combined problem already built from it (via `parent_problem_id`),
    so this stays accurate whether `problem_id` is the original problem or one already combined
    from it. Used to decide whether to show the "Add uncertainty measures" UI action at all.
    """
    problem_db = context.problem_db
    uncertainty_source = _resolve_uncertainty_source(problem_db, context.db_session)
    if uncertainty_source is None:
        return CumulusScenarioAvailabilityResponse(available=False)

    scenario_model_db = uncertainty_source.scenario_model_db
    return CumulusScenarioAvailabilityResponse(
        available=True,
        source=uncertainty_source.source,
        base_problem_id=uncertainty_source.base_problem_db.id,
        scenario_model_id=scenario_model_db.id if scenario_model_db is not None else None,
        objective_symbols=uncertainty_source.objective_symbols,
        applied_measures=(
            [] if uncertainty_source.base_problem_db.id == problem_db.id else _detect_applied_measures(problem_db)
        ),
        measure_types=_MEASURE_TYPES[uncertainty_source.source],
    )


@router.post("/apply-scenario-measures")
def apply_scenario_measures(
    request: CumulusInitializationRequest,
    context: Annotated[SessionContext, Depends(SessionContextGuard(require=[ContextField.PROBLEM]).post)],
) -> CumulusInitializationResponse:
    """Build (or rebuild) a combined scenario problem with the given uncertainty measures.

    Unlike `/get-or-initialize`, this always builds a fresh combined problem and starts CUMULUS
    on it, even when a CUMULUS state already exists for the current problem. Used by the "Add
    uncertainty measures" UI action to add measures after the method has already started, or to
    rebuild the combined problem with a different set of measures.
    """
    if not request.uncertainty_measures:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="uncertainty_measures must be provided and non-empty.",
        )

    uncertainty_source = _resolve_uncertainty_source(context.problem_db, context.db_session, strict=True)
    if uncertainty_source is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="This problem has no associated scenario model or graphical model.",
        )

    return _build_combined_problem_and_initialize(
        request, context, uncertainty_source.base_problem_db, parent_state=None
    )


@router.post("/finalize")
def finalize(
    request: CumulusFinalizeRequest,
    context: Annotated[SessionContext, Depends(SessionContextGuard(require=[ContextField.PROBLEM]).post)],
) -> CumulusFinalizeResponse:
    """Finalize the CUMULUS process by selecting a solution."""
    db_session = context.db_session
    user = context.user
    interactive_session = context.interactive_session
    parent_state = context.parent_state
    problem_db = context.problem_db

    solution_state_id = request.solution_info.state_id
    solution_index = request.solution_info.solution_index

    state = db_session.exec(select(StateDB).where(StateDB.id == solution_state_id)).first()
    actual_state = state.state if state else None
    if actual_state is None:
        raise HTTPException(
            detail="No concrete substate!",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    final_state = CumulusFinalState(
        solution_origin_state_id=solution_state_id,
        solution_result_index=solution_index,
        solver_results=get_solver_results_at(actual_state, solution_index),
        original_problem_id=request.original_problem_id,
    )

    state = StateDB.create(
        database_session=db_session,
        problem_id=problem_db.id,
        session_id=interactive_session.id if interactive_session is not None else None,
        parent_id=parent_state.id if parent_state is not None else None,
        state=final_state,
        kind=StateKind.CUMULUS_FINAL,
    )

    db_session.add(state)
    db_session.commit()
    db_session.refresh(state)

    solution_reference_response = SolutionReferenceResponse(
        solution_index=solution_index,
        state_id=solution_state_id,
        objective_values=final_state.solver_results.optimal_objectives,
        variable_values=final_state.solver_results.optimal_variables,
    )

    return CumulusFinalizeResponse(
        state_id=state.id,
        final_solution=solution_reference_response,
        saved_solutions=collect_saved_solutions(user=user, problem_id=problem_db.id, session=db_session),
        all_solutions=collect_all_solutions(user=user, problem_id=problem_db.id, session=db_session),
    )


@router.post("/intermediate")
def solve_intermediate(
    request: IntermediateSolutionRequest,
    context: Annotated[SessionContext, Depends(SessionContextGuard(require=[ContextField.PROBLEM]).post)],
) -> CumulusIntermediateSolutionResponse:
    """Solve intermediate solutions between two solutions using rpm_intermediate_solutions."""
    db_session = context.db_session
    user = context.user
    problem_db = context.problem_db
    interactive_session = context.interactive_session
    parent_state = context.parent_state

    var_and_obj_values_of_references: list[tuple[dict, dict]] = []
    reference_states = []

    for solution_info in [request.reference_solution_1, request.reference_solution_2]:
        solution_state = db_session.exec(select(StateDB).where(StateDB.id == solution_info.state_id)).first()

        if solution_state is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Could not find a state with id={solution_info.state_id}.",
            )

        reference_states.append(solution_state)

        try:
            _var_values = solution_state.state.result_variable_values
            var_values = _var_values[solution_info.solution_index]
        except IndexError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"The index {solution_info.solution_index} is out of bounds "
                f"for results with len={len(_var_values)}",
            ) from exc

        try:
            _obj_values = solution_state.state.result_objective_values
            obj_values = _obj_values[solution_info.solution_index]
        except IndexError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"The index {solution_info.solution_index} is out of bounds "
                f"for results with len={len(_obj_values)}",
            ) from exc

        var_and_obj_values_of_references.append((var_values, obj_values))

    problem = Problem.from_problemdb(problem_db)
    solver = check_solver(problem_db=problem_db)

    solver_results: list[SolverResults] = rpm_intermediate_solutions(
        problem=problem,
        solution_1=var_and_obj_values_of_references[0][1],
        solution_2=var_and_obj_values_of_references[1][1],
        num_desired=request.num_desired or 1,
        scalarization_options=request.scalarization_options,
        solver=solver,
        solver_options=request.solver_options,
    )

    intermediate_state = IntermediateSolutionState(
        scalarization_options=request.scalarization_options,
        context="cumulus",
        solver=request.solver,
        solver_options=request.solver_options,
        solver_results=solver_results,
        num_desired=request.num_desired or 1,
        reference_solution_1=var_and_obj_values_of_references[0][1],
        reference_solution_2=var_and_obj_values_of_references[1][1],
    )

    state = StateDB.create(
        database_session=db_session,
        problem_id=problem_db.id,
        session_id=interactive_session.id if interactive_session is not None else None,
        parent_id=parent_state.id if parent_state is not None else None,
        state=intermediate_state,
    )

    db_session.add(state)
    db_session.commit()
    db_session.refresh(state)

    saved_solutions = collect_saved_solutions(user, request.problem_id, db_session)
    all_solutions = collect_all_solutions(user, request.problem_id, db_session)

    return CumulusIntermediateSolutionResponse(
        state_id=state.id,
        reference_solution_1=var_and_obj_values_of_references[0][1],
        reference_solution_2=var_and_obj_values_of_references[1][1],
        current_solutions=[SolutionReference(state=state, solution_index=i) for i in range(state.state.num_solutions)],
        saved_solutions=saved_solutions,
        all_solutions=all_solutions,
    )


@router.post("/delete_save")
def delete_save(
    request: CumulusDeleteSaveRequest,
    context: Annotated[SessionContext, Depends(SessionContextGuard().post)],
) -> CumulusDeleteSaveResponse:
    """Delete a saved CUMULUS solution."""
    db_session = context.db_session

    to_be_deleted = db_session.exec(
        select(UserSavedSolutionDB).where(
            UserSavedSolutionDB.origin_state_id == request.state_id,
            UserSavedSolutionDB.solution_index == request.solution_index,
        )
    ).first()

    if to_be_deleted is None:
        raise HTTPException(detail="Unable to find a saved solution!", status_code=status.HTTP_404_NOT_FOUND)

    db_session.delete(to_be_deleted)
    db_session.commit()

    to_be_deleted = db_session.exec(
        select(UserSavedSolutionDB).where(
            UserSavedSolutionDB.origin_state_id == request.state_id,
            UserSavedSolutionDB.solution_index == request.solution_index,
        )
    ).first()

    if to_be_deleted is not None:
        raise HTTPException(
            detail="Could not delete the saved solution!", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

    return CumulusDeleteSaveResponse(message="Save deleted.")


def _fill_in_soft_constraint_properties(
    problem_db: ProblemDB, raws: list[Any], constraint_dbs: list[ConstraintDB]
) -> None:
    """Derive the math properties of soft constraints whose request left them unstated.

    The rows are stored detached from any problem, but ``solve_sub_problems`` feeds them to
    ``add_soft_constraint`` against the one in play, so their flags end up describing the
    problem that is actually solved: a bound that claims not to be twice differentiable drags
    the whole problem onto the non-differentiable path with it, which not every solver can
    express. The analysis therefore runs against the problem they will be added to, with the
    constraints attached, so that the symbols they reference resolve.

    Modifies the rows in `constraint_dbs` in place. `raws` are the request entries they were
    validated from, positionally matched, and are consulted to see which properties the request
    stated for itself.
    """
    pending = [
        constraint_db
        for raw, constraint_db in zip(raws, constraint_dbs, strict=True)
        if leaves_properties_unstated(raw)
    ]
    if not pending:
        return

    problem = Problem.from_problemdb(problem_db)
    # `from_attributes` reads just the fields `Constraint` declares, leaving the columns that
    # only exist on the DB row (which `Constraint` forbids) behind.
    attached = [Constraint.model_validate(constraint_db, from_attributes=True) for constraint_db in pending]

    analyzed = determine_unstated_constraint_properties(
        problem.model_copy(update={"constraints": [*(problem.constraints or []), *attached]}),
        [constraint.symbol for constraint in attached],
    )
    by_symbol = {constraint.symbol: constraint for constraint in (analyzed.constraints or [])}

    for constraint_db in pending:
        result = by_symbol.get(constraint_db.symbol)
        if result is None:
            continue
        constraint_db.is_linear = result.is_linear
        constraint_db.is_convex = result.is_convex
        constraint_db.is_twice_differentiable = result.is_twice_differentiable


@router.get("/objective-constraint")
def get_objective_constraints(
    context: Annotated[SessionContext, Depends(SessionContextGuard(require=[ContextField.PROBLEM]).get)],
) -> CumulusActiveObjectiveConstraintsResponse:
    """Return the soft objective constraints in effect for the problem, so the UI can show them."""
    db_session = context.db_session
    session_id = context.interactive_session.id if context.interactive_session else None
    soft_ids = _get_objective_constraint_ids(context.problem_db.id, session_id, db_session)
    rows = db_session.exec(select(ConstraintDB).where(ConstraintDB.id.in_(soft_ids))).all()
    by_id = {row.id: row for row in rows}
    return CumulusActiveObjectiveConstraintsResponse(
        soft_constraints=[
            CumulusObjectiveConstraintInfo(
                name=row.name, symbol=row.symbol, cons_type=str(row.cons_type), func=row.func
            )
            for row in (by_id[i] for i in soft_ids if i in by_id)
        ]
    )


@router.post("/objective-constraint")
def set_objective_constraints(
    request: CumulusObjectiveConstraintRequest,
    context: Annotated[SessionContext, Depends(SessionContextGuard(require=[ContextField.PROBLEM]).post)],
) -> CumulusObjectiveConstraintResponse:
    """Set (replace) the list of soft objective constraints for the current CUMULUS session.

    Creates ConstraintDB rows in the database without adding them to the active problem.
    A new state records the IDs of the current constraint list.

    Hard bounds change the feasible region and need the method reinitialized, so they go
    through ``/method/cumulus/modify-problem`` instead.
    """
    db_session = context.db_session
    interactive_session = context.interactive_session
    parent_state = context.parent_state
    problem_db = context.problem_db

    soft_constraints: list[ConstraintDB] = []
    for raw in request.soft_constraints:
        try:
            constraint_db = ConstraintDB.model_validate(raw)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
        constraint_db.id = None
        constraint_db.problem_id = None
        soft_constraints.append(constraint_db)

    # Before persisting: the stored flags are what later solves read back.
    _fill_in_soft_constraint_properties(problem_db, list(request.soft_constraints), soft_constraints)
    for constraint_db in soft_constraints:
        db_session.add(constraint_db)

    db_session.flush()
    soft_constraint_ids = [c.id for c in soft_constraints]

    obj_constraint_state = CumulusObjectiveConstraintState(
        problem_id=problem_db.id,
        original_problem_id=request.original_problem_id,
        soft_constraint_ids=soft_constraint_ids,
    )

    state = StateDB.create(
        database_session=db_session,
        problem_id=problem_db.id,
        session_id=interactive_session.id if interactive_session is not None else None,
        parent_id=parent_state.id if parent_state is not None else None,
        state=obj_constraint_state,
    )

    db_session.add(state)
    db_session.commit()
    db_session.refresh(state)

    return CumulusObjectiveConstraintResponse(
        state_id=state.id,
        soft_constraint_ids=soft_constraint_ids,
    )


def _solution_satisfies(problem: Problem, results: SolverResults, constraints: list[Constraint]) -> bool:
    """Whether one stored solution satisfies `constraints`; False also when that cannot be told.

    The constraints are evaluated against the values the solver reported, and only failing that
    against the problem re-evaluated at the stored decision vector: the evaluator cannot build a
    frame for a combined scenario problem at all, so re-evaluating first would leave every bound on
    such a problem undecided.
    """
    parser = MathParser(to_format=FormatEnum.polars)

    def values_at(scope: dict[str, float]) -> list[float] | None:
        frame = pl.DataFrame({symbol: [value] for symbol, value in scope.items()})
        try:
            return [float(frame.select(parser.parse(c.func).alias("_value")).row(0)[0]) for c in constraints]
        except Exception:
            return None

    # Scalars only: tensor values cannot share a frame with them, and a bound never needs them.
    reported = (
        results.optimal_variables,
        results.optimal_objectives,
        results.extra_func_values or {},
        results.scalarization_values or {},
    )
    scope = {sym: float(val) for source in reported for sym, val in source.items() if isinstance(val, int | float)}
    # Constraints may reference the always-minimized variant of an objective.
    for objective in problem.objectives:
        if objective.symbol in scope:
            value = scope[objective.symbol]
            scope.setdefault(f"{objective.symbol}_min", -value if objective.maximize else value)
    values = values_at(scope)

    if values is None:
        # A symbol the reported values lack.  Only the problem's own variables are passed: a stored
        # solution may also carry auxiliary ones (a CVaR `_alpha`, say) with no evaluator column.
        try:
            xs = {variable.symbol: [results.optimal_variables[variable.symbol]] for variable in problem.variables}
            evaluated = PolarsEvaluator(problem).evaluate(xs).row(0, named=True)
        except Exception:
            evaluated = {}
        evaluated = {sym: float(val) for sym, val in evaluated.items() if isinstance(val, int | float)}
        values = values_at({**evaluated, **scope})

    if values is None:
        logger.warning(
            "Could not tell whether a current solution satisfies the changed constraints (%s).",
            ", ".join(c.symbol for c in constraints),
        )
        return False

    # LTE is `func <= 0`; EQ is `func == 0`.
    return all(
        (abs(v) if c.cons_type == ConstraintTypeEnum.EQ else v) <= _CONSTRAINT_TOLERANCE
        for c, v in zip(constraints, values, strict=True)
    )


def _some_current_solution_satisfies(problem: Problem, parent_state, changed_constraints: list[Constraint]) -> bool:
    """Whether some solution current at `parent_state` satisfies the changed constraints.

    Such a solution proves the modified problem feasible without a solve.  Only meaningful when
    the modification changed nothing but constraints: then the others were already satisfied
    before it.
    """
    if not changed_constraints:
        return True
    # A simulator or surrogate constraint has no function to evaluate at a stored solution.
    if any(c.func is None for c in changed_constraints):
        return False
    try:
        num_solutions = parent_state.state.num_solutions
    except (AttributeError, NotImplementedError):
        return False

    for index in range(num_solutions):
        try:
            results = get_solver_results_at(parent_state.state, index)
        except (AttributeError, IndexError, KeyError, TypeError):
            continue
        if _solution_satisfies(problem, results, changed_constraints):
            return True
    return False


def _all_variables_bounded(problem: Problem) -> bool:
    """Whether every variable, tensor elements included, has finite lower and upper bounds."""
    scalars = [s for v in problem.variables for s in (v.to_variables() if isinstance(v, TensorVariable) else [v])]
    return all(bound is not None and np.isfinite(bound) for s in scalars for bound in (s.lowerbound, s.upperbound))


def _changed_by_modification(old: Problem, new: Problem) -> tuple[list[Constraint], bool]:
    """Which constraints of `new` the modification may have made unsatisfiable.

    Returns:
        (constraints, only_constraints_changed).  When only constraints were added or replaced,
        those are the ones returned; the rest held before and still do.  When anything a
        constraint can depend on changed (variables, constants, objectives, extra functions),
        every constraint of `new` is returned.
    """

    def by_symbol(elements) -> dict:
        return {e.symbol: e.model_dump() for e in elements or []}

    others_changed = any(
        by_symbol(getattr(old, field)) != by_symbol(getattr(new, field))
        for field in ("variables", "constants", "objectives", "extra_funcs")
    )
    if others_changed:
        return list(new.constraints or []), False

    old_constraints = by_symbol(old.constraints)
    changed = [c for c in new.constraints or [] if old_constraints.get(c.symbol) != c.model_dump()]
    return changed, True


def _probe_new_problem_feasibility(
    base_problem: Problem, constraints: list[Constraint], solver_class, solver_options=None
) -> tuple[bool | None, float, str | None]:
    """Is `base_problem` with `constraints` added feasible? Decided with one solve.

    The constraints are attached to `base_problem` as soft ones and the total violation is
    minimised.  A minimum of zero means some point satisfies all of them at once, so the problem
    is feasible; anything larger is the amount by which they cannot jointly be met.  The
    constraints already in `base_problem` stay hard, so they must be known to be satisfiable.

    Minimising violation rather than projecting a reference point also covers constraints over
    several objectives at once, which a reference point cannot express.

    Returns:
        (feasible, minimum total violation, message or None).  `feasible` is None when the solve
        failed, or when a heuristic solver found a positive violation, so feasibility is
        undetermined rather than refuted; the caller rejects both.
    """
    if not constraints:
        return True, 0.0, None
    try:
        _, _, violation = _minimize_soft_constraint_violations(
            base_problem, None, constraints, solver_class, solver_options
        )
    except Exception as e:
        return (
            None,
            float("nan"),
            (f"Could not determine whether the modified problem is feasible ({e}), so the modification was discarded."),
        )
    if violation > _CONSTRAINT_TOLERANCE:
        symbols = ", ".join(c.symbol for c in constraints)
        if is_heuristic_solver(solver_class):
            return (
                None,
                violation,
                (
                    "Could not determine whether the modified problem is feasible. The smallest total "
                    f"violation of the constraints ({symbols}) found was {violation:.6g}, but the "
                    f"solver ({solver_class.__name__}) is a heuristic one, so a better solution may "
                    "exist. The modification was discarded."
                ),
            )
        return (
            False,
            violation,
            (
                "The modified problem has no feasible solution. The constraints "
                f"({symbols}) cannot all be satisfied at once; the smallest total violation "
                f"achievable is {violation:.6g}. The modification was discarded."
            ),
        )
    return True, violation, None


def _run_modification_checks_background(
    new_problem_id: int,
    cumulus_state_id: int,
    solver_class,
    probe_constraint_symbols: list[str],
    feasibility_witnessed: bool,
) -> None:
    """Background: confirm the modified problem is feasible and bounded, and refresh its ideal and nadir.

    Feasibility is settled first because it costs one solve, where the payoff table costs one per
    objective.  The constraints in `probe_constraint_symbols` are softened and the rest kept hard.
    The probe is skipped when `feasibility_witnessed`, i.e. a current solution already satisfies
    the modified problem.  The payoff table then runs for every modification: optimizing each
    objective on its own is also what reveals one left unbounded, e.g. by a removed constraint.

    A modification that fails either check, or whose checks cannot be completed, is discarded
    outright: its ProblemDB row is deleted, which cascades to the metadata created alongside it,
    and the decision maker stays on the problem they were already using.  The modification state
    is moved to the parent problem first, so the status endpoint can still report the reason.

    The checks are deliberately conservative: a modification that cannot be shown feasible and
    bounded is discarded even when it might be fine.  In particular:

    - The probe uses the same solver as the modified problem.  When that is a heuristic one (e.g.
      Nevergrad for non-smooth objectives), any positive violation it finds is treated as
      undetermined and discarded, so most probed modifications of such problems are rejected.
    - A probe over a simulator or surrogate constraint (no `func`) cannot be built, so it is
      discarded.  Changing variables, constants, objectives or extra functions probes every
      constraint, so on problems with such constraints these modifications are always rejected.
    - A positive violation from a local solver (Ipopt, Bonmin, SLSQP) on non-convex constraints
      is reported as infeasibility, though it only shows the solver found no feasible point.
    - With a heuristic solver the payoff table cannot reveal unboundedness, so the modification is
      discarded unless every variable has finite bounds.

    Other routes for modifying or defining problems remain for the cases rejected here.
    """
    try:
        with Session(engine) as session:
            problem_db = session.get(ProblemDB, new_problem_id)
            mod_state = session.get(CumulusModificationState, cumulus_state_id)
            if problem_db is None or mod_state is None:
                return

            def discard(message: str) -> None:
                # Move the state to the parent problem so the cascade spares it and the status
                # endpoint can still report why the modification was discarded.
                mod_state.is_ready = True
                mod_state.error = message
                mod_state.problem_deleted = True
                session.add(mod_state)
                state_db = session.get(StateDB, cumulus_state_id)
                if state_db is not None:
                    state_db.problem_id = problem_db.parent_problem_id
                    session.add(state_db)
                session.commit()
                session.delete(problem_db)
                session.commit()

            problem = Problem.from_problemdb(problem_db)
            effective_solver = solver_class if solver_class is not None else guess_best_solver(problem)

            if probe_constraint_symbols and not feasibility_witnessed:
                soft = [c for c in problem.constraints or [] if c.symbol in set(probe_constraint_symbols)]
                hard = [c for c in problem.constraints or [] if c.symbol not in set(probe_constraint_symbols)]
                base_problem = problem.model_copy(update={"constraints": hard or None})
                feasible, _, probe_message = _probe_new_problem_feasibility(base_problem, soft, effective_solver)
                if not feasible:
                    discard(probe_message)
                    return

            # A heuristic solver reports success whatever it finds, so its payoff table cannot reveal
            # an unbounded objective; only a fully bounded box of variables rules one out.
            if is_heuristic_solver(effective_solver) and not _all_variables_bounded(problem):
                discard(
                    f"The solver ({effective_solver.__name__}) is a heuristic one, so it cannot tell whether "
                    "the modified problem is bounded when some variable lacks a finite lower or upper "
                    "bound. The modification was discarded."
                )
                return

            try:
                estimated_ideal, estimated_nadir = payoff_table_method(
                    problem, solver=effective_solver, require_success=True
                )
            except Exception as e:
                discard(
                    f"Could not compute the ideal and nadir points of the modified problem: {e} "
                    "The modification was discarded."
                )
                return

            for obj_db in problem_db.objectives:
                if obj_db.symbol in estimated_ideal:
                    obj_db.ideal = estimated_ideal[obj_db.symbol]
                if obj_db.symbol in estimated_nadir:
                    obj_db.nadir = estimated_nadir[obj_db.symbol]
                session.add(obj_db)

            # The modification survived: it is a real problem now, not a provisional one.
            problem_db.is_temporary = False
            session.add(problem_db)
            mod_state.is_ready = True
            mod_state.error = None
            session.add(mod_state)
            session.commit()
    except Exception:  # noqa: S110
        # Background tasks cannot surface errors to the caller; is_ready stays False.
        pass


@router.post("/modify-problem")
def modify_problem(
    request: CumulusModificationRequest,
    background_tasks: BackgroundTasks,
    context: Annotated[SessionContext, Depends(SessionContextGuard(require=[ContextField.PROBLEM]).post)],
) -> CumulusModificationResponse:
    """Apply modifications to the current problem, verify feasibility, and save as a new problem."""
    db_session = context.db_session
    user = context.user
    problem_db = context.problem_db
    interactive_session = context.interactive_session
    parent_state = context.parent_state

    problem = Problem.from_problemdb(problem_db)
    graphical_meta = find_graphical_metadata(problem_db.id, db_session)
    modified_problem = apply_problem_modifications(problem, request.modifications, db_session, graphical_meta)

    changed_constraints, only_constraints_changed = _changed_by_modification(problem, modified_problem)

    # Cheap and synchronous: a current solution that satisfies the changed constraints proves the
    # modified problem feasible, and spares the probe.  Only a constraints-only modification can be
    # judged this way: once variables, objectives or the like change, the stored values no longer
    # describe the problem.
    # The stored solutions must belong to this problem, or they need not satisfy its constraints.
    current_solution_feasible = (
        only_constraints_changed
        and parent_state is not None
        and parent_state.problem_id == problem_db.id
        and _some_current_solution_satisfies(modified_problem, parent_state, changed_constraints)
    )

    descriptions = (
        _retargeted_solution_descriptions(
            problem_db, modified_problem, request.modifications.uncertainty_measures, db_session
        )
        if graphical_meta is not None and request.modifications.uncertainty_measures
        else None
    )

    # Save the modified problem immediately (ideal/nadir will be updated by the background task).
    # It stays temporary until the background check confirms it is feasible.
    new_problem_db = ProblemDB.from_problem(modified_problem, user=user)
    new_problem_db.is_temporary = True
    new_problem_db.parent_problem_id = problem_db.id
    if request.name is not None:
        new_problem_db.name = request.name
    original_desc = problem_db.description or ""
    modified_desc = new_problem_db.description or ""
    if original_desc and modified_desc != original_desc:
        new_problem_db.description = original_desc + "\n\n" + modified_desc
    db_session.add(new_problem_db)
    db_session.commit()
    db_session.refresh(new_problem_db)
    copy_problem_metadata(problem_db, new_problem_db, db_session)
    db_session.commit()
    if descriptions is not None:
        _replace_solution_descriptions(new_problem_db, descriptions, db_session)

    original_problem_id = request.original_problem_id if request.original_problem_id is not None else problem_db.id

    mod_session_id = interactive_session.id if interactive_session else None
    mod_scenario_model_id = _get_scenario_model_id(problem_db.id, mod_session_id, db_session)

    modification_state = CumulusModificationState(
        problem_id=new_problem_db.id,
        original_problem_id=original_problem_id,
        scenario_model_id=mod_scenario_model_id,
        is_ready=False,
        current_solution_feasible=current_solution_feasible,
    )

    state = StateDB.create(
        database_session=db_session,
        problem_id=new_problem_db.id,
        session_id=interactive_session.id if interactive_session is not None else None,
        parent_id=parent_state.id if parent_state is not None else None,
        state=modification_state,
    )

    db_session.add(state)
    db_session.commit()
    db_session.refresh(state)

    solver_class = check_solver(problem_db=problem_db)

    background_tasks.add_task(
        _run_modification_checks_background,
        new_problem_db.id,
        state.base_state.id,
        solver_class,
        [c.symbol for c in changed_constraints],
        current_solution_feasible,
    )

    return CumulusModificationResponse(
        state_id=state.id,
        problem_id=new_problem_db.id,
        original_problem_id=original_problem_id,
        scenario_model_id=mod_scenario_model_id,
        is_ready=False,
        current_solution_feasible=current_solution_feasible,
    )


@router.post("/hypothetical")
def create_hypothetical(
    request: CumulusHypotheticalRequest,
    background_tasks: BackgroundTasks,
    context: Annotated[SessionContext, Depends(SessionContextGuard(require=[ContextField.PROBLEM]).post)],
) -> CumulusModificationResponse:
    """Create a new problem from a graphical-model problem under hypothetical assumptions.

    The new problem is recompiled from the conditioned diagram and gets its own copy of the
    graphical metadata; the original problem is left as it is. Like a modification, it stays
    temporary until the background checks confirm it is feasible and set its ideal and nadir.
    Poll `/modify-problem/status/{state_id}` for the outcome.

    Raises:
        HTTPException: 404 if the problem has no graphical model; 400 if the assumptions are
            invalid, add nothing new, or contradict what the problem already assumes.
    """
    db_session = context.db_session
    problem_db = context.problem_db
    parent_state = context.parent_state

    meta = _fetch_graphical_metadata(problem_db.id, db_session)
    network = DiscreteNetwork.model_validate_json(meta.network_json)
    annotation = DecisionAnnotation.model_validate_json(meta.annotation_json)
    existing = (
        DiagramAssumptions.model_validate_json(meta.assumptions_json) if meta.assumptions_json else DiagramAssumptions()
    )

    try:
        assumptions = existing.merged_with(request.assumptions)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    if assumptions == existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The assumptions add nothing to what this problem already assumes.",
        )
    try:
        # Re-conditioning on `existing.chance` is a no-op; its decisions still need fixing.
        problem, index_map, rjt, conditioned = compile_with_assumptions(network, annotation, assumptions)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    problem = problem.model_copy(
        update={
            "name": f"{problem_db.name} | {request.assumptions.label()}",
            "description": problem_db.description,
        }
    )
    new_problem_db = ProblemDB.from_problem(problem, user=context.user)
    new_problem_db.is_temporary = True
    new_problem_db.parent_problem_id = problem_db.id
    db_session.add(new_problem_db)
    db_session.commit()
    db_session.refresh(new_problem_db)

    copy_problem_metadata(problem_db, new_problem_db, db_session)
    db_session.commit()
    new_meta = _fetch_graphical_metadata(new_problem_db.id, db_session)
    new_meta.network_json = conditioned.model_dump_json()
    new_meta.rjt_json = rjt.model_dump_json()
    new_meta.index_map_json = index_map.model_dump_json()
    new_meta.assumptions_json = assumptions.model_dump_json()
    db_session.add(new_meta)

    state = StateDB.create(
        database_session=db_session,
        problem_id=new_problem_db.id,
        session_id=context.interactive_session.id if context.interactive_session is not None else None,
        parent_id=parent_state.id if parent_state is not None else None,
        state=CumulusModificationState(
            problem_id=new_problem_db.id,
            original_problem_id=problem_db.id,
            is_ready=False,
            # The current solutions belong to another problem, so they prove nothing.
            current_solution_feasible=False,
        ),
    )
    db_session.add(state)
    db_session.commit()
    db_session.refresh(state)

    # Probe the assumed decisions; the payoff table catches anything else that fails.
    assumed_decision_symbols = [c.symbol for c in problem.constraints or [] if c.symbol.startswith("assume_")]
    background_tasks.add_task(
        _run_modification_checks_background,
        new_problem_db.id,
        state.base_state.id,
        check_solver(problem_db=problem_db),
        assumed_decision_symbols,
        feasibility_witnessed=False,
    )

    return CumulusModificationResponse(
        state_id=state.id,
        problem_id=new_problem_db.id,
        original_problem_id=problem_db.id,
        is_ready=False,
        current_solution_feasible=False,
    )


@router.get("/modify-problem/status/{state_id}")
def modify_problem_status(
    state_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> CumulusModificationResponse:
    """Poll the status of a pending modify-problem operation."""
    state = db_session.exec(select(StateDB).where(StateDB.id == state_id)).first()
    if state is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"State {state_id} not found.")

    mod_state = state.state
    if not isinstance(mod_state, CumulusModificationState):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"State {state_id} is not a Cumulus modification state.",
        )

    return CumulusModificationResponse(
        state_id=state_id,
        scenario_model_id=mod_state.scenario_model_id,
        problem_id=mod_state.problem_id,
        original_problem_id=mod_state.original_problem_id,
        is_ready=mod_state.is_ready,
        error=mod_state.error,
        current_solution_feasible=mod_state.current_solution_feasible,
        problem_deleted=mod_state.problem_deleted,
    )
