"""Endpoints for compiling, inspecting, and querying graphical models.

Bayesian networks and influence diagrams. Unlike `desdeo/problem/graphical/`, this module uses
`pyagrum` directly for live inference.

pyAgrum behaviour relied on below (verified empirically):

- `setEvidence(...)` on `LazyPropagation`/`ShaferShenoyLIMIDInference` *replaces* the whole
  evidence set (`setEvidence({})` clears it). This makes cached engines safe to reuse across calls
  without leaking evidence.
- `LazyPropagation.addJointTarget(...)` registered once stays valid across later
  `setEvidence`/`makeInference` cycles.
- `ShaferShenoyLIMIDInference.posterior(decision)` is a 0/1 vector when the decision's information
  set is fully pinned by evidence, and a marginal over choices otherwise (see `_read_policy`).
- `ShaferShenoyLIMIDInference` has no `jointPosterior`; `MEU()` sums all utility nodes, and
  `posteriorUtility(node)` doesn't collapse to a scalar. So there is no way to get one objective's
  expected utility when a diagram has more than one objective and a decision (501 in
  `_compute_utilities`).
- `gum.GumException` is the base of pyAgrum's errors (unknown node/label, unsolvable LIMID, ...);
  these are reported as 400.
- `ShaferShenoyLIMIDInference.MEU()` returns `{"mean": ..., "variance": ...}`.
"""

import hashlib
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite, sqrt
from typing import Annotated, Literal

import numpy as np
import pyagrum as gum
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlmodel import Session, select

from desdeo.api.db import get_session
from desdeo.api.models import (
    GraphicalEvidenceSetDB,
    GraphicalModelMetaData,
    ProblemDB,
    ProblemMetaDataDB,
    User,
    UserRole,
    UserSavedSolutionDB,
)
from desdeo.api.routers.user_authentication import get_current_user
from desdeo.api.routers.utils import fetch_interactive_session_with_role_check, fetch_problem_with_role_check
from desdeo.problem import Problem, VariableDomainTypeEnum
from desdeo.problem.graphical import (
    DecisionAnnotation,
    DiscreteNetwork,
    IndexMap,
    build_rooted_junction_tree,
    compile_diagram,
    read_policy_from_solution,
)
from desdeo.problem.graphical.adapters.pyagrum import to_pyagrum, to_pyagrum_influence_diagram
from desdeo.problem.graphical.demo import build_irrigation_demo
from desdeo.problem.graphical.index_map import mixed_radix_encode
from desdeo.tools import find_compatible_solvers, guess_best_solver
from desdeo.tools.gurobipy_solver_interfaces import GurobipySolver
from desdeo.tools.pyomo_solver_interfaces import PyomoBonminSolver, PyomoCBCSolver, PyomoGurobiSolver

router = APIRouter(prefix="/graphical", tags=["Graphical models"])


# --- Request / response schemas -------------------------------------------------------------------


class CompileRequest(BaseModel):
    """Body for `POST /graphical/compile`."""

    network: DiscreteNetwork
    annotation: DecisionAnnotation
    name: str | None = None
    max_cluster_size: int | None = None
    on_exceed: Literal["warn", "raise"] = "warn"


class CompileResponse(BaseModel):
    """Response for `POST /graphical/compile`."""

    problem_id: int


class StructureNode(BaseModel):
    """One node in a `GET /graphical/{problem_id}/structure` response."""

    id: str
    label: str
    kind: Literal["chance", "decision", "objective"]
    states: list[str] | None = None
    """Chance nodes: `DiscreteVariable.labels`. Decisions: `DecisionNode.states`. Objectives have
    no states (a utility value, not a discrete domain) -- always `None`."""
    description: str | None = None
    """Chance nodes only: `DiscreteVariable.description`, or `None` if empty. Always `None` for
    decision/objective nodes."""


class StructureEdge(BaseModel):
    """One directed edge in a `GET /graphical/{problem_id}/structure` response."""

    id: str
    """`f"{source}->{target}"`, unique since a DAG has at most one edge per pair."""
    source: str
    target: str


class StructureResponse(BaseModel):
    """Response for `GET /graphical/{problem_id}/structure`."""

    nodes: list[StructureNode]
    edges: list[StructureEdge]


class WhatIfRequest(BaseModel):
    """Body for `POST /graphical/whatif`."""

    problem_id: int
    session_id: int | None = None
    evidence: dict[str, str] = {}
    """Chance-node name -> pinned state label (hard evidence only)."""
    commit: bool = False
    """When `True`, a computed utility outside the problem's persisted ideal/nadir widens that bound
    (see `_ratchet_ideal_nadir`) and the full ideal/nadir are returned. When `False`, ideal/nadir
    are left untouched."""


class WhatIfResponse(BaseModel):
    """Response for `POST /graphical/whatif`."""

    posteriors: dict[str, list[float]]
    """Chance-node name -> per-state probability, in that node's own `labels` order."""
    policy: dict[str, str] | None = None
    """Decision symbol -> chosen state label, only for decisions whose information set is fully
    pinned by the evidence (see `_read_policy`). `None` when there are no decisions."""
    utilities: dict[str, float]
    """Objective symbol -> expected utility under the given evidence."""
    utility_std_devs: dict[str, float]
    """Objective symbol -> standard deviation of the utility under the given evidence."""
    ideal: dict[str, float] | None = None
    """Objective symbol -> the problem's ideal value after any ratchet. Only set when `commit` is
    `True`, otherwise `None`.

    Since the ratchet adopts evidence-conditioned utilities, these bounds can be wider than what
    any actual solution of the problem attains."""
    nadir: dict[str, float] | None = None
    """Objective symbol -> the problem's current nadir value, same population rule as `ideal`."""


# --- POST /graphical/compile ----------------------------------------------------------------------


_SERVER_MAX_CLUSTER_SIZE = 200_000
"""Upper bound on any cluster's state space, enforced regardless of the request's
`max_cluster_size`, so one synchronous `/compile` call can't block a worker indefinitely. A request
may ask for a stricter cap but never a looser one (see `_effective_cluster_cap`)."""

_SERVER_MAX_TOTAL_CLUSTER_SIZE = 2_000_000
"""Upper bound on the *sum* of all clusters' state spaces, since many small clusters can still
make a huge problem. Not overridable by the request."""


def _effective_cluster_cap(request: CompileRequest) -> tuple[int, Literal["warn", "raise"]]:
    """Clamp `request.max_cluster_size`/`on_exceed` to never exceed `_SERVER_MAX_CLUSTER_SIZE`.

    A cap at or under the ceiling is used as given. A missing or looser cap is replaced by the
    ceiling with `"raise"`.
    """
    if request.max_cluster_size is not None and request.max_cluster_size <= _SERVER_MAX_CLUSTER_SIZE:
        return request.max_cluster_size, request.on_exceed
    return _SERVER_MAX_CLUSTER_SIZE, "raise"


@router.post("/compile")
def compile_graphical_model(
    request: CompileRequest,
    user: Annotated[User, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> CompileResponse:
    """Compile a `(DiscreteNetwork, DecisionAnnotation)` pair into a `Problem`, and persist it.

    Persists the `ProblemDB` plus a `GraphicalModelMetaData` row holding the network, annotation,
    `RootedJunctionTree`, and `IndexMap`, in one commit. Guest users may not compile.

    Raises:
        HTTPException: 401 for a guest user; 400 for a `ValueError` from validation or
            compilation (e.g. `max_cluster_size` exceeded), or if the total cluster state space
            exceeds `_SERVER_MAX_TOTAL_CLUSTER_SIZE`.
    """
    if user.role == UserRole.guest:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Guest users are not allowed to compile graphical models.",
        )

    max_cluster_size, on_exceed = _effective_cluster_cap(request)
    try:
        rjt = build_rooted_junction_tree(
            request.network, request.annotation, max_cluster_size=max_cluster_size, on_exceed=on_exceed
        )
        total_cluster_size = sum(cluster.state_space_size for cluster in rjt.clusters.values())
        if total_cluster_size > _SERVER_MAX_TOTAL_CLUSTER_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"Combined cluster state space ({total_cluster_size}) exceeds the server "
                    f"limit of {_SERVER_MAX_TOTAL_CLUSTER_SIZE}, even though every individual "
                    f"cluster is under max_cluster_size={max_cluster_size}."
                ),
            )
        problem, index_map = compile_diagram(request.network, request.annotation, rjt=rjt)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    if request.name is not None:
        problem = problem.model_copy(update={"name": request.name})

    # `flush()` assigns ids for the FKs without committing, so a failure rolls everything back.
    problem_db = ProblemDB.from_problem(problem, user=user)
    db_session.add(problem_db)
    db_session.flush()

    metadata_db = ProblemMetaDataDB(problem_id=problem_db.id)
    db_session.add(metadata_db)
    db_session.flush()

    graphical_metadata = GraphicalModelMetaData(
        metadata_id=metadata_db.id,
        network_json=request.network.model_dump_json(),
        annotation_json=request.annotation.model_dump_json(),
        rjt_json=rjt.model_dump_json(),
        index_map_json=index_map.model_dump_json(),
    )
    db_session.add(graphical_metadata)
    db_session.commit()
    db_session.refresh(problem_db)

    return CompileResponse(problem_id=problem_db.id)


# --- GET /graphical/{problem_id}/structure --------------------------------------------------------


def _fetch_graphical_metadata(problem_id: int, db_session: Session) -> GraphicalModelMetaData:
    """The most recently added `GraphicalModelMetaData` row for `problem_id`, or 404.

    Queried directly with an explicit order, since `ProblemMetaDataDB.all_metadata` is unordered.
    """
    row = db_session.exec(
        select(GraphicalModelMetaData)
        .join(ProblemMetaDataDB, GraphicalModelMetaData.metadata_id == ProblemMetaDataDB.id)
        .where(ProblemMetaDataDB.problem_id == problem_id)
        .order_by(GraphicalModelMetaData.id.desc())
    ).first()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No graphical model metadata found for problem {problem_id}.",
        )
    return row


@router.get("/{problem_id}/structure")
def get_structure(
    problem_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> StructureResponse:
    """The DAG and node types of a graphical model, without inference.

    Raises:
        HTTPException: 404 if the problem doesn't exist, isn't visible to `user`, or has no
            `GraphicalModelMetaData` row.
    """
    problem_db = fetch_problem_with_role_check(user, problem_id, db_session)
    if problem_db is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Problem with id={problem_id} not found.")

    meta = _fetch_graphical_metadata(problem_id, db_session)
    network = DiscreteNetwork.model_validate_json(meta.network_json)
    annotation = DecisionAnnotation.model_validate_json(meta.annotation_json)

    nodes: list[StructureNode] = []
    edges: list[StructureEdge] = []

    def _add_edge(source: str, target: str) -> None:
        edges.append(StructureEdge(id=f"{source}->{target}", source=source, target=target))

    for variable in network.variables:
        nodes.append(
            StructureNode(
                id=variable.name,
                label=variable.name,
                kind="chance",
                states=variable.labels,
                description=variable.description or None,
            )
        )
    for parent, child in network.arcs:
        _add_edge(parent, child)

    for decision in annotation.decisions:
        # `validate_against` already rejects decision symbols that collide with chance nodes.
        nodes.append(
            StructureNode(
                id=decision.symbol, label=decision.name or decision.symbol, kind="decision", states=decision.states
            )
        )
        for parent in decision.information_set:
            _add_edge(parent, decision.symbol)

    for objective in annotation.objectives:
        # Objective symbols may collide with chance-node names, so prefix them with `util_` (the
        # same prefix `to_pyagrum_influence_diagram` uses).
        objective_id = f"util_{objective.symbol}"
        nodes.append(StructureNode(id=objective_id, label=objective.name or objective.symbol, kind="objective"))
        for parent in objective.parents:
            _add_edge(parent, objective_id)

    return StructureResponse(nodes=nodes, edges=edges)


# --- Session-scoped inference engine cache --------------------------------------------------------

_CACHE_TTL_SECONDS = 30 * 60
"""Idle time after which a cached engine is evicted."""

_CACHE_MAX_SIZE = 128
"""Maximum number of cached engines; least-recently-used are evicted first."""

_cache_lock = threading.Lock()
"""Guards `_engine_cache` itself; each entry's engine is guarded by `_EngineCacheEntry.lock`."""


@dataclass
class _EngineCacheEntry:
    """One cached pyAgrum inference engine plus the bookkeeping needed to reuse it safely."""

    engine: "gum.LazyPropagation | gum.ShaferShenoyLIMIDInference"
    content_hash: str
    """`_content_hash(network_json, annotation_json)` at build time; a mismatch forces a rebuild."""
    last_used: float
    """`time.monotonic()` of the last access, for TTL and LRU eviction."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    """Serialises evidence-setting, inference, and read-back on this entry's engine, so concurrent
    requests sharing a cache key don't interleave."""


_engine_cache: dict[str, _EngineCacheEntry] = {}
"""In-memory cache of pyAgrum inference engines, keyed by session id or, without one, by
`user:<id>:problem:<id>`. One entry per key, so switching models within a session forces a rebuild.

Process-local: a multi-worker deployment needs sticky sessions, since live pyAgrum engines can't
be shared between processes. Evidence is not cached; each request sets its own.
"""


def _content_hash(network_json: str, annotation_json: str) -> str:
    """Fingerprint of `(network, annotation)`, used to invalidate cached engines."""
    return hashlib.sha256(f"{network_json}\x00{annotation_json}".encode()).hexdigest()


def _evict_stale(now: float) -> None:
    """Drop TTL-expired entries, then trim to `_CACHE_MAX_SIZE` by least-recently-used.

    Called on every cache access; there is no background sweep.
    """
    expired = [key for key, entry in _engine_cache.items() if now - entry.last_used > _CACHE_TTL_SECONDS]
    for key in expired:
        del _engine_cache[key]

    if len(_engine_cache) > _CACHE_MAX_SIZE:
        by_recency = sorted(_engine_cache.items(), key=lambda item: item[1].last_used)
        for key, _entry in by_recency[: len(_engine_cache) - _CACHE_MAX_SIZE]:
            del _engine_cache[key]


def _build_engine(network: DiscreteNetwork, annotation: DecisionAnnotation):
    """Build a fresh pyAgrum inference engine for `(network, annotation)`.

    Without decisions, uses `LazyPropagation` on a `BayesNet`, since `ShaferShenoyLIMIDInference`
    refuses diagrams with no decisions. Each objective's parents are registered as a joint target.
    Every chance node is also registered as a target, because declaring any joint target makes
    `LazyPropagation` drop its default of targeting every node.
    """
    if not annotation.decisions:
        engine = gum.LazyPropagation(to_pyagrum(network))
        for objective in annotation.objectives:
            if objective.parents:
                engine.addJointTarget(set(objective.parents))
        for variable in network.variables:
            engine.addTarget(variable.name)
        return engine

    diagram = to_pyagrum_influence_diagram(network, annotation)
    engine = gum.ShaferShenoyLIMIDInference(diagram)
    if annotation.no_forgetting:
        engine.addNoForgettingAssumption(list(annotation.decision_order))
    return engine


def _get_or_build_entry(
    cache_key: str, network_json: str, annotation_json: str, network: DiscreteNetwork, annotation: DecisionAnnotation
) -> _EngineCacheEntry:
    """Look up `cache_key`'s cached engine, rebuilding it if missing/stale/content-mismatched.

    The raw JSON strings are passed alongside the parsed models to avoid re-serialising them for
    the content hash.
    """
    content_hash = _content_hash(network_json, annotation_json)
    now = time.monotonic()

    with _cache_lock:
        _evict_stale(now)
        entry = _engine_cache.get(cache_key)
        if entry is not None and entry.content_hash == content_hash:
            entry.last_used = now
            return entry

    # Built outside the global lock so a slow build doesn't block other requests. Concurrent misses
    # for the same key may both build; the re-check below keeps the first one.
    built = _EngineCacheEntry(engine=_build_engine(network, annotation), content_hash=content_hash, last_used=now)

    with _cache_lock:
        _evict_stale(now)
        entry = _engine_cache.get(cache_key)
        if entry is not None and entry.content_hash == content_hash:
            entry.last_used = now
            return entry
        _engine_cache[cache_key] = built
        return built


# --- POST /graphical/whatif -----------------------------------------------------------------------


def _read_posteriors(engine, network: DiscreteNetwork) -> dict[str, list[float]]:
    """`{chance_node_name: [P(state) for state in variable.labels]}`, current evidence applied."""
    return {variable.name: engine.posterior(variable.name).toarray().tolist() for variable in network.variables}


def _read_policy(engine, annotation: DecisionAnnotation) -> dict[str, str] | None:
    """`{decision_symbol: chosen_state_label}` for decisions fully determined by the evidence.

    `None` if there are no decisions. A decision's posterior is one-hot exactly when its
    information set is fully pinned; otherwise it is a marginal over choices, and the decision is
    omitted rather than reporting the argmax as if it were the policy.
    """
    if not annotation.decisions:
        return None
    policy: dict[str, str] = {}
    for decision in annotation.decisions:
        probs = np.asarray(engine.posterior(decision.symbol).toarray())
        best = int(np.argmax(probs))
        if np.isclose(probs[best], 1.0, atol=1e-6):
            policy[decision.symbol] = decision.states[best]
    return policy


def _expected_value_no_decision(engine, objective) -> tuple[float, float]:
    """One `ObjectiveNode`'s `(expected_utility, variance)` from a `LazyPropagation` engine.

    Contracts the joint posterior over the objective's parents against its utility table. The
    variance `E[U^2] - E[U]^2` is clamped at zero against roundoff. Validation guarantees
    `objective.parents` is non-empty.
    """
    posterior = engine.jointPosterior(set(objective.parents))
    order = [v.name() for v in posterior.variablesSequence()]
    probs = np.transpose(posterior.toarray())
    utility = np.array(objective.utility.values, dtype=float)
    probs = np.moveaxis(probs, [order.index(p) for p in objective.parents], range(len(objective.parents)))
    mean = float(np.sum(probs * utility))
    variance = max(float(np.sum(probs * utility**2) - mean**2), 0.0)
    return mean, variance


def _compute_utilities(engine, annotation: DecisionAnnotation) -> tuple[dict[str, float], dict[str, float]]:
    """`({objective_symbol: expected_utility}, {objective_symbol: std_dev})`, current evidence.

    - No decisions: per-objective contraction (`_expected_value_no_decision`).
    - One objective with decisions: `MEU()`. The utility table is negated for minimized
      objectives (pyAgrum maximizes), so the mean is negated back; the variance is unaffected.
    - Several objectives with decisions: not implemented (501), see the module docstring.
    """
    if not annotation.decisions:
        means: dict[str, float] = {}
        std_devs: dict[str, float] = {}
        for objective in annotation.objectives:
            mean, variance = _expected_value_no_decision(engine, objective)
            means[objective.symbol] = mean
            std_devs[objective.symbol] = sqrt(variance)
        return means, std_devs

    if len(annotation.objectives) == 1:
        objective = annotation.objectives[0]
        # Undo the negation `to_pyagrum_influence_diagram` applies to minimized objectives.
        sign = 1.0 if objective.maximize else -1.0
        meu = engine.MEU()
        return {objective.symbol: sign * meu["mean"]}, {objective.symbol: sqrt(max(meu["variance"], 0.0))}

    msg = (
        "Computing per-objective expected utility is not implemented for a diagram with more "
        "than one ObjectiveNode and at least one decision: ShaferShenoyLIMIDInference exposes no "
        "joint-posterior primitive, MEU() sums every utility node into one scalar, and "
        "posteriorUtility(node) does not collapse to a scalar either."
    )
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=msg)


def _resolve_cache_key(user: User, session_id: int | None, problem_id: int, db_session: Session) -> str:
    """`f"session:{id}"` (ownership-checked) if `session_id` is given, else a per-user-per-problem key."""
    if session_id is not None:
        # Raises 404 itself if not found/owned -- see fetch_interactive_session_with_role_check.
        fetch_interactive_session_with_role_check(user, session_id, db_session)
        return f"session:{session_id}"
    return f"user:{user.id}:problem:{problem_id}"


def _validate_chance_evidence(network: DiscreteNetwork, evidence: dict[str, str]) -> None:
    """400 if `evidence` references a node that isn't a chance node, or a label it doesn't have."""
    chance_labels = {variable.name: variable.labels for variable in network.variables}
    unknown_evidence = sorted(set(evidence) - set(chance_labels))
    if unknown_evidence:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Evidence references node(s) that aren't DiscreteNetwork chance nodes: "
                f"{unknown_evidence}. Only hard evidence on chance nodes is supported."
            ),
        )
    for node, value in evidence.items():
        if value not in chance_labels[node]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Evidence value '{value}' for node '{node}' isn't one of its own states {chance_labels[node]}.",
            )


def _ratchet_ideal_nadir(problem_db: ProblemDB, utilities: dict[str, float], db_session: Session) -> bool:
    """Widen `problem_db`'s persisted ideal/nadir to cover `utilities`, never narrow them.

    For a maximized objective, `ideal` only increases and `nadir` only decreases (reversed when
    minimizing). Since the bounds can never be narrowed again:

    - Non-finite values are skipped.
    - Objectives with neither bound set are skipped, since adopting the value for both would
      give a zero-width range. The initial range must come from e.g. a payoff table.
    - If only one bound is set, the missing one is adopted directly.

    Changed rows are added to `db_session`; committing is left to the caller. Not safe against
    concurrent commits on the same problem (lost updates are possible).

    Returns:
        bool: whether any `ObjectiveDB` row was changed.
    """
    any_changed = False
    for obj_db in problem_db.objectives:
        if obj_db.symbol not in utilities:
            continue
        value = utilities[obj_db.symbol]
        if not isfinite(value):
            continue
        if obj_db.ideal is None and obj_db.nadir is None:
            # No range to widen yet.
            continue

        # A `None` bound (with the other set) is always adopted.
        if obj_db.maximize:
            better_ideal = obj_db.ideal is None or value > obj_db.ideal
            worse_nadir = obj_db.nadir is None or value < obj_db.nadir
        else:
            better_ideal = obj_db.ideal is None or value < obj_db.ideal
            worse_nadir = obj_db.nadir is None or value > obj_db.nadir
        if better_ideal:
            obj_db.ideal = value
        if worse_nadir:
            obj_db.nadir = value
        if better_ideal or worse_nadir:
            db_session.add(obj_db)
            any_changed = True
    return any_changed


@router.post("/whatif")
def whatif(
    request: WhatIfRequest,
    user: Annotated[User, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> WhatIfResponse:
    """Set hard evidence, run inference, and return posteriors/policy/utilities in one payload.

    Uses a cached engine (see `_engine_cache`). With `request.commit=True`, also ratchets the
    problem's ideal/nadir (see `_ratchet_ideal_nadir`) and returns them.

    Raises:
        HTTPException: 404 if the problem or session isn't found or visible, or the problem has no
            `GraphicalModelMetaData`; 403 if `commit` is `True` and `user` is neither the owner
            nor an admin/analyst; 400 for invalid evidence or a diagram pyAgrum rejects; 501 for
            several objectives with decisions (see `_compute_utilities`).
    """
    problem_db = fetch_problem_with_role_check(user, request.problem_id, db_session)
    if problem_db is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Problem with id={request.problem_id} not found."
        )

    if request.commit and problem_db.user_id != user.id and user.role not in (UserRole.analyst, UserRole.admin):
        # Visibility (e.g. via a group) is not enough to write to the problem.
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the problem's owner (or an admin/analyst) may commit a what-if result.",
        )

    meta = _fetch_graphical_metadata(request.problem_id, db_session)
    network = DiscreteNetwork.model_validate_json(meta.network_json)
    annotation = DecisionAnnotation.model_validate_json(meta.annotation_json)

    _validate_chance_evidence(network, request.evidence)
    cache_key = _resolve_cache_key(user, request.session_id, request.problem_id, db_session)
    entry = _get_or_build_entry(cache_key, meta.network_json, meta.annotation_json, network, annotation)

    with entry.lock:
        entry.last_used = time.monotonic()
        try:
            entry.engine.setEvidence(request.evidence)
            entry.engine.makeInference()
        except gum.GumException as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

        posteriors = _read_posteriors(entry.engine, network)
        policy = _read_policy(entry.engine, annotation)
        utilities, utility_std_devs = _compute_utilities(entry.engine, annotation)

    ideal: dict[str, float] | None = None
    nadir: dict[str, float] | None = None
    if request.commit:
        if _ratchet_ideal_nadir(problem_db, utilities, db_session):
            db_session.commit()
        ideal = {obj_db.symbol: obj_db.ideal for obj_db in problem_db.objectives if obj_db.ideal is not None}
        nadir = {obj_db.symbol: obj_db.nadir for obj_db in problem_db.objectives if obj_db.nadir is not None}

    return WhatIfResponse(
        posteriors=posteriors,
        policy=policy,
        utilities=utilities,
        utility_std_devs=utility_std_devs,
        ideal=ideal,
        nadir=nadir,
    )


# --- POST /graphical/{problem_id}/policy ----------------------------------------------------------
#
# Unlike `/whatif`'s `policy` (from a live pyAgrum engine), this decodes the policy of a specific
# solved `Problem` from its `z` variables via `IndexMap`, without pyAgrum.


class PolicyRequest(BaseModel):
    """Body for `POST /graphical/{problem_id}/policy`."""

    variable_values: dict[str, list[float] | float] | None = None
    """A solved `Problem`'s `optimal_variables`, including every decision's `z_symbol`. Exactly one
    of this or `solution_id` must be given. Scalars are allowed, e.g. a `RiskMeasure`'s `eta`."""
    solution_id: int | None = None
    """Id of a `UserSavedSolutionDB` row for this `problem_id` to read the variable values from."""
    evidence: dict[str, str] = {}
    """Chance-node name -> pinned state label, used to pick each decision's information state."""


class PolicyResponse(BaseModel):
    """Response for `POST /graphical/{problem_id}/policy`."""

    policy: dict[str, str]
    """Decision symbol -> chosen state label, only for decisions whose information set is fully
    pinned by `evidence`. Same semantics as `WhatIfResponse.policy`."""


def _decode_solved_policy(
    index_map: IndexMap,
    network: DiscreteNetwork,
    annotation: DecisionAnnotation,
    variable_values: dict[str, list[float] | float],
    evidence: dict[str, str],
) -> dict[str, str]:
    """`{decision_symbol: chosen_state_label}` -- `read_policy_from_solution`, collapsed via `evidence`.

    A decision is included only when all of its information-set members are resolved. Decisions
    are decoded in `annotation.decision_order` (or declaration order), adding each chosen label to
    the resolved set, so that later decisions whose information set includes an earlier decision
    can be resolved too.
    """
    label_lists: dict[str, list[str]] = {variable.name: variable.labels for variable in network.variables}
    label_lists.update({decision.symbol: decision.states for decision in annotation.decisions})

    try:
        policies = read_policy_from_solution(index_map, variable_values)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"variable_values is missing a required entry: {e}.",
        ) from e
    except (ValueError, TypeError) as e:
        # `ValueError`: a `z_symbol` entry has the wrong length (e.g. a stale saved solution).
        # `TypeError`: a bare scalar was given for a `z_symbol`.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    decoding_order = annotation.decision_order or [decision.symbol for decision in annotation.decisions]
    resolved = dict(evidence)
    policy: dict[str, str] = {}
    for symbol in decoding_order:
        decision_policy = policies.get(symbol)
        if decision_policy is None:
            continue
        info_indices: list[int] = []
        for node in decision_policy.information_set:
            if node not in resolved:
                break
            labels = label_lists.get(node)
            if labels is None or resolved[node] not in labels:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Evidence value for '{node}' isn't one of its own states.",
                )
            info_indices.append(labels.index(resolved[node]))
        else:
            info_cardinalities = [len(label_lists[node]) for node in decision_policy.information_set]
            info_flat = mixed_radix_encode(info_indices, info_cardinalities)
            chosen = decision_policy.policy[info_flat]
            chosen_label = decision_policy.states[chosen]
            policy[symbol] = chosen_label
            resolved[symbol] = chosen_label
        # On `break` (an unresolved information-set member), the decision is omitted.
    return policy


@router.post("/{problem_id}/policy")
def get_policy(
    problem_id: int,
    request: PolicyRequest,
    user: Annotated[User, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> PolicyResponse:
    """Decode a solved `Problem`'s policy against `evidence`.

    Raises:
        HTTPException: 404 if the problem isn't found or visible, has no graphical metadata or
            `IndexMap`, or `solution_id` isn't a saved solution of `problem_id`; 403 if the saved
            solution belongs to another user and `user` is not an admin/analyst; 400 if not
            exactly one of `variable_values`/`solution_id` is given, or for invalid evidence or
            variable values.
    """
    problem_db = fetch_problem_with_role_check(user, problem_id, db_session)
    if problem_db is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Problem with id={problem_id} not found.")

    meta = _fetch_graphical_metadata(problem_id, db_session)
    if meta.index_map_json is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Problem {problem_id} has no IndexMap yet -- it wasn't compiled via /graphical/compile.",
        )

    network = DiscreteNetwork.model_validate_json(meta.network_json)
    annotation = DecisionAnnotation.model_validate_json(meta.annotation_json)
    index_map = IndexMap.model_validate_json(meta.index_map_json)

    if (request.variable_values is None) == (request.solution_id is None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Exactly one of variable_values or solution_id must be given.",
        )

    if request.variable_values is not None:
        variable_values = request.variable_values
    else:
        solution = db_session.get(UserSavedSolutionDB, request.solution_id)
        if solution is None or solution.problem_id != problem_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No saved solution with id={request.solution_id} for problem {problem_id}.",
            )
        # Visibility into the problem doesn't grant access to another user's saved solution.
        if solution.user_id != user.id and user.role not in (UserRole.analyst, UserRole.admin):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the saving user (or an admin/analyst) may read back this solution's policy.",
            )
        variable_values = solution.variable_values

    _validate_chance_evidence(network, request.evidence)
    policy = _decode_solved_policy(index_map, network, annotation, variable_values, request.evidence)
    return PolicyResponse(policy=policy)


# --- Named evidence sets --------------------------------------------------------------------------


class EvidenceSetCreateRequest(BaseModel):
    """Body for `POST /graphical/{problem_id}/evidence-sets`."""

    name: str
    evidence: dict[str, str] = {}


class EvidenceSetResponse(BaseModel):
    """Response shape for every named-evidence-set endpoint below."""

    id: int
    problem_id: int
    name: str
    evidence: dict[str, str]
    created_at: datetime

    @classmethod
    def from_db(cls, row: GraphicalEvidenceSetDB) -> "EvidenceSetResponse":
        """Build the public response shape from a DB row."""
        return cls(
            id=row.id,
            problem_id=row.problem_id,
            name=row.name,
            evidence=row.evidence,
            created_at=row.created_at,
        )


@router.post("/{problem_id}/evidence-sets")
def create_evidence_set(
    problem_id: int,
    request: EvidenceSetCreateRequest,
    user: Annotated[User, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> EvidenceSetResponse:
    """Save the current evidence dict under a name, for later reuse/sharing by id.

    Raises:
        HTTPException: 404 if the problem doesn't exist, isn't visible to `user`, or has no
            `GraphicalModelMetaData`; 400 for invalid evidence.
    """
    problem_db = fetch_problem_with_role_check(user, problem_id, db_session)
    if problem_db is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Problem with id={problem_id} not found.")

    meta = _fetch_graphical_metadata(problem_id, db_session)
    network = DiscreteNetwork.model_validate_json(meta.network_json)
    _validate_chance_evidence(network, request.evidence)

    row = GraphicalEvidenceSetDB(
        problem_id=problem_id, user_id=user.id, name=request.name, evidence=dict(request.evidence)
    )
    db_session.add(row)
    db_session.commit()
    db_session.refresh(row)
    return EvidenceSetResponse.from_db(row)


@router.get("/{problem_id}/evidence-sets")
def list_evidence_sets(
    problem_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> list[EvidenceSetResponse]:
    """Every named evidence set saved against `problem_id`, regardless of which user saved it.

    Evidence sets are shareable: anyone who can see the problem can read them, while deletion is
    restricted to the owner.

    Raises:
        HTTPException: 404 if the problem doesn't exist or isn't visible to `user`.
    """
    problem_db = fetch_problem_with_role_check(user, problem_id, db_session)
    if problem_db is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Problem with id={problem_id} not found.")

    rows = db_session.exec(
        select(GraphicalEvidenceSetDB)
        .where(GraphicalEvidenceSetDB.problem_id == problem_id)
        .order_by(GraphicalEvidenceSetDB.created_at.desc())
    ).all()
    return [EvidenceSetResponse.from_db(row) for row in rows]


@router.get("/evidence-sets/{evidence_set_id}")
def get_evidence_set(
    evidence_set_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> EvidenceSetResponse:
    """Fetch one named evidence set by id, so it can be shared by URL.

    Visibility is checked against the set's problem.

    Raises:
        HTTPException: 404 if no such evidence set exists, or its problem isn't visible to `user`.
    """
    row = db_session.get(GraphicalEvidenceSetDB, evidence_set_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No evidence set with id={evidence_set_id}.")
    problem_db = fetch_problem_with_role_check(user, row.problem_id, db_session)
    if problem_db is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No evidence set with id={evidence_set_id}.")
    return EvidenceSetResponse.from_db(row)


@router.delete("/evidence-sets/{evidence_set_id}")
def delete_evidence_set(
    evidence_set_id: int,
    user: Annotated[User, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> None:
    """Delete a named evidence set. Owner (or admin/analyst) only, unlike reading/listing.

    Raises:
        HTTPException: 404 if no such evidence set exists, or its problem isn't visible to `user`;
            403 if `user` neither saved it nor is an admin/analyst.
    """
    row = db_session.get(GraphicalEvidenceSetDB, evidence_set_id)
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No evidence set with id={evidence_set_id}.")
    problem_db = fetch_problem_with_role_check(user, row.problem_id, db_session)
    if problem_db is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"No evidence set with id={evidence_set_id}.")
    if row.user_id != user.id and user.role not in (UserRole.analyst, UserRole.admin):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Only the saving user (or an admin/analyst) may delete it."
        )
    db_session.delete(row)
    db_session.commit()


# --- POST /graphical/sensitivity, POST /graphical/voi ---------------------------------------------
#
# Both use the cached engines and the shared `_sweep_node` primitive.


def _resolve_target_value(
    engine, network: DiscreteNetwork, annotation: DecisionAnnotation, target: str, target_state: str | None
) -> float:
    """`target`'s current scalar value under `engine`'s current evidence state.

    For an `ObjectiveNode.symbol`, its expected utility. For a chance/decision node,
    `P(target == target_state)`, which requires `target_state`.
    """
    objective_by_symbol = {objective.symbol: objective for objective in annotation.objectives}
    if target in objective_by_symbol:
        if not annotation.decisions:
            mean, _variance = _expected_value_no_decision(engine, objective_by_symbol[target])
            return mean
        means, _ = _compute_utilities(engine, annotation)
        return means[target]

    if target_state is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'target_state' is required when target ('{target}') isn't an ObjectiveNode symbol.",
        )

    label_lists: dict[str, list[str]] = {variable.name: variable.labels for variable in network.variables}
    label_lists.update({decision.symbol: decision.states for decision in annotation.decisions})
    labels = label_lists.get(target)
    if labels is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{target}' isn't a chance node, decision node, or ObjectiveNode symbol of this diagram.",
        )
    if target_state not in labels:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{target_state}' isn't one of '{target}''s own states {labels}.",
        )
    probs = np.asarray(engine.posterior(target).toarray())
    return float(probs[labels.index(target_state)])


def _sweep_node(
    engine,
    network: DiscreteNetwork,
    annotation: DecisionAnnotation,
    base_evidence: dict[str, str],
    sweep_node_name: str,
    target: str,
    target_state: str | None,
    states: list[str] | None = None,
) -> dict[str, float]:
    """`{state_label: target_value}` for every state of `sweep_node_name`.

    Pins one state at a time on top of `base_evidence`. Mutates `engine`'s evidence; callers must
    hold `entry.lock` and restore the evidence afterwards.

    `states`: sweep only these states instead of all of the node's labels.
    """
    label_lists: dict[str, list[str]] = {variable.name: variable.labels for variable in network.variables}
    labels = states if states is not None else label_lists[sweep_node_name]

    values: dict[str, float] = {}
    for state in labels:
        engine.setEvidence({**base_evidence, sweep_node_name: state})
        engine.makeInference()
        values[state] = _resolve_target_value(engine, network, annotation, target, target_state)
    return values


_SENSITIVITY_NEGLIGIBLE_MARGINAL = 1e-6
"""Swept states with a base marginal below this are skipped: pyAgrum doesn't raise on
near-impossible evidence, but the resulting values are meaningless."""

_MAX_SENSITIVITY_SWEEP_NODES = 50
"""Maximum number of chance nodes one `/sensitivity` request may sweep, to bound the work done by a
synchronous request."""


class SensitivityRequest(BaseModel):
    """Body for `POST /graphical/sensitivity`."""

    problem_id: int
    session_id: int | None = None
    evidence: dict[str, str] = {}
    target: str
    """An `ObjectiveNode.symbol`, or a chance/decision node's own name (with `target_state` set)."""
    target_state: str | None = None
    """Required when `target` is a chance/decision node rather than an objective."""
    nodes: list[str] | None = None
    """Restrict the sweep to these chance nodes. `None` sweeps every chance node, subject to
    `_MAX_SENSITIVITY_SWEEP_NODES`."""


class SensitivityNodeResult(BaseModel):
    """One swept node's tornado-bar entry."""

    node: str
    values: dict[str, float]
    """State label -> `target`'s value with `node` pinned to that state. States with a negligible
    base marginal are omitted (see `_SENSITIVITY_NEGLIGIBLE_MARGINAL`)."""
    low: float
    high: float
    swing: float
    """`high - low`; `results` is sorted by this, descending."""


class SensitivityResponse(BaseModel):
    """Response for `POST /graphical/sensitivity`."""

    base_value: float
    """`target`'s value under `evidence`, without sweeping."""
    results: list[SensitivityNodeResult]
    """One entry per swept chance node (excluding `target` and evidenced nodes), sorted by `swing`
    descending."""


@router.post("/sensitivity")
def sensitivity(
    request: SensitivityRequest,
    user: Annotated[User, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> SensitivityResponse:
    """Single-node hard-evidence sweep, one node at a time, holding everything else at `evidence`.

    Sweeps every chance node other than `target` and the evidenced ones (decision nodes are not
    swept), using the cached engine.

    Raises:
        HTTPException: 401 for a guest user; 404 if the problem isn't found or visible, or has no
            `GraphicalModelMetaData`; 400 for invalid evidence, `target`, `target_state` or
            `nodes`, or too many nodes to sweep; 501 if `target` is an objective of a diagram
            with several objectives and a decision (see `_compute_utilities`).
    """
    if user.role == UserRole.guest:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Guest users are not allowed to run sensitivity analysis on graphical models.",
        )

    problem_db = fetch_problem_with_role_check(user, request.problem_id, db_session)
    if problem_db is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Problem with id={request.problem_id} not found."
        )

    meta = _fetch_graphical_metadata(request.problem_id, db_session)
    network = DiscreteNetwork.model_validate_json(meta.network_json)
    annotation = DecisionAnnotation.model_validate_json(meta.annotation_json)

    _validate_chance_evidence(network, request.evidence)

    chance_node_names = {variable.name for variable in network.variables}
    candidate_names = [
        variable.name
        for variable in network.variables
        if variable.name != request.target and variable.name not in request.evidence
    ]
    if request.nodes is not None:
        unknown_nodes = sorted(set(request.nodes) - chance_node_names)
        if unknown_nodes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"'nodes' references node(s) that aren't DiscreteNetwork chance nodes: {unknown_nodes}.",
            )
        requested = set(request.nodes)
        candidate_names = [name for name in candidate_names if name in requested]
    if len(candidate_names) > _MAX_SENSITIVITY_SWEEP_NODES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Sweeping {len(candidate_names)} node(s) exceeds the server limit of "
                f"{_MAX_SENSITIVITY_SWEEP_NODES} per /sensitivity request -- narrow 'nodes'."
            ),
        )

    cache_key = _resolve_cache_key(user, request.session_id, request.problem_id, db_session)
    entry = _get_or_build_entry(cache_key, meta.network_json, meta.annotation_json, network, annotation)

    with entry.lock:
        entry.last_used = time.monotonic()
        try:
            entry.engine.setEvidence(request.evidence)
            entry.engine.makeInference()
            base_value = _resolve_target_value(entry.engine, network, annotation, request.target, request.target_state)

            # Marginals under the base evidence, to skip near-impossible states.
            label_lists = {variable.name: variable.labels for variable in network.variables}
            base_marginals = {
                name: dict(
                    zip(label_lists[name], np.asarray(entry.engine.posterior(name).toarray()).tolist(), strict=True)
                )
                for name in candidate_names
            }

            results: list[SensitivityNodeResult] = []
            for name in candidate_names:
                surviving_states = [
                    state
                    for state in label_lists[name]
                    if base_marginals[name][state] >= _SENSITIVITY_NEGLIGIBLE_MARGINAL
                ]
                if not surviving_states:
                    # Shouldn't happen with normalised CPTs; avoids min/max on an empty dict.
                    continue
                values = _sweep_node(
                    entry.engine,
                    network,
                    annotation,
                    request.evidence,
                    name,
                    request.target,
                    request.target_state,
                    states=surviving_states,
                )
                low, high = min(values.values()), max(values.values())
                results.append(SensitivityNodeResult(node=name, values=values, low=low, high=high, swing=high - low))
        except gum.GumException as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
        finally:
            # Best-effort restore of the base evidence (every caller sets its own evidence anyway).
            # Errors are swallowed: if pyAgrum just rejected `request.evidence`, raising here
            # would replace the 400 above with a 500.
            try:
                entry.engine.setEvidence(request.evidence)
                entry.engine.makeInference()
            except gum.GumException:
                pass

    results.sort(key=lambda r: r.swing, reverse=True)
    return SensitivityResponse(base_value=base_value, results=results)


class VoiRequest(BaseModel):
    """Body for `POST /graphical/voi`."""

    problem_id: int
    session_id: int | None = None
    evidence: dict[str, str] = {}
    node: str
    """The chance node whose value of information (before deciding) is being asked about."""
    target: str
    """An `ObjectiveNode.symbol`."""


class VoiResponse(BaseModel):
    """Response for `POST /graphical/voi`."""

    base_value: float
    """`target`'s expected utility under `evidence` alone, `node` unobserved."""
    expected_value_with_observation: float
    """`sum_s P(node=s | evidence) * target_value(evidence, node=s)`: the expected utility if
    `node` were observed before deciding."""
    value_of_information: float
    """`expected_value_with_observation - base_value`. Not clamped at zero, so floating-point noise
    stays visible."""
    marginal: dict[str, float]
    """`P(node=state | evidence)`, the weights used for `expected_value_with_observation`."""


@router.post("/voi")
def value_of_information(
    request: VoiRequest,
    user: Annotated[User, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> VoiResponse:
    """Value of information for observing `node` before deciding, as a gain in expected utility.

    `VoI(node) = sum_s P(node=s) * value(node=s) - value()`, computed with `_sweep_node`. Only
    defined for diagrams with at least one decision; otherwise there is nothing the observation
    could change, so the request is rejected rather than returning zero.

    Raises:
        HTTPException: 404 if the problem isn't found or visible, or has no
            `GraphicalModelMetaData`; 400 if the diagram has no decisions, or for an invalid
            `target`, `node` (or one already in `evidence`), or evidence; 501 for several
            objectives with a decision (see `_compute_utilities`).
    """
    problem_db = fetch_problem_with_role_check(user, request.problem_id, db_session)
    if problem_db is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Problem with id={request.problem_id} not found."
        )

    meta = _fetch_graphical_metadata(request.problem_id, db_session)
    network = DiscreteNetwork.model_validate_json(meta.network_json)
    annotation = DecisionAnnotation.model_validate_json(meta.annotation_json)

    if not annotation.decisions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Value of information is only meaningful for a diagram with at least one decision "
                "-- 'EU gain from observing before deciding' presupposes something being decided."
            ),
        )
    if request.target not in {objective.symbol for objective in annotation.objectives}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{request.target}' isn't an ObjectiveNode symbol of this diagram.",
        )
    chance_names = {variable.name: variable.labels for variable in network.variables}
    if request.node not in chance_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"'{request.node}' isn't a DiscreteNetwork chance node."
        )
    if request.node in request.evidence:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{request.node}' is already pinned by evidence; there is nothing left to observe.",
        )
    _validate_chance_evidence(network, request.evidence)

    cache_key = _resolve_cache_key(user, request.session_id, request.problem_id, db_session)
    entry = _get_or_build_entry(cache_key, meta.network_json, meta.annotation_json, network, annotation)

    with entry.lock:
        entry.last_used = time.monotonic()
        try:
            entry.engine.setEvidence(request.evidence)
            entry.engine.makeInference()
            base_value = _resolve_target_value(entry.engine, network, annotation, request.target, None)

            labels = chance_names[request.node]
            marginal_probs = np.asarray(entry.engine.posterior(request.node).toarray())
            marginal = dict(zip(labels, marginal_probs.tolist(), strict=True))

            values = _sweep_node(
                entry.engine, network, annotation, request.evidence, request.node, request.target, None
            )
            expected_with_observation = sum(marginal[state] * values[state] for state in labels)
        except gum.GumException as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
        finally:
            # Best-effort restore, see `sensitivity`.
            try:
                entry.engine.setEvidence(request.evidence)
                entry.engine.makeInference()
            except gum.GumException:
                pass

    return VoiResponse(
        base_value=base_value,
        expected_value_with_observation=expected_with_observation,
        value_of_information=expected_with_observation - base_value,
        marginal=marginal,
    )


# --- POST /graphical/demo/irrigation, POST /graphical/{problem_id}/solve --------------------------
#
# Minimal endpoints backing the webui's network demo dialog: a compiled demo problem, and a solved
# policy to display.


_DEMO_IRRIGATION_PROBLEM_NAME = "CUMULUS network demo (irrigation)"
"""Name the demo problem is saved under, also used to find an existing copy."""


class DemoIrrigationRequest(BaseModel):
    """Body for `POST /graphical/demo/irrigation`. Empty body (`{}`) is a normal, expected call."""

    force: bool = False
    """If `False`, reuse the caller's existing demo problem if there is one. If `True`, always
    compile a new copy."""


def _demo_irrigation_metadata_matches(problem_id: int, db_session: Session) -> bool:
    """Whether `problem_id`'s `GraphicalModelMetaData` network matches `build_irrigation_demo()`.

    A name match alone could pick up an unrelated problem saved under the same name.
    """
    row = db_session.exec(
        select(GraphicalModelMetaData)
        .join(ProblemMetaDataDB, GraphicalModelMetaData.metadata_id == ProblemMetaDataDB.id)
        .where(ProblemMetaDataDB.problem_id == problem_id)
        .order_by(GraphicalModelMetaData.id.desc())
    ).first()
    if row is None:
        return False
    demo_network, _demo_annotation = build_irrigation_demo()
    return row.network_json == demo_network.model_dump_json()


@router.post("/demo/irrigation")
def compile_demo_irrigation_model(
    request: DemoIrrigationRequest,
    user: Annotated[User, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> CompileResponse:
    """Compile the canned irrigation demo diagram (`desdeo/problem/graphical/demo.py`).

    Builds the model server-side and delegates to `compile_graphical_model`.

    Unless `request.force` is set, returns the caller's most recent problem with the demo name if
    its network matches the demo (see `_demo_irrigation_metadata_matches`), and compiles a new
    copy otherwise.

    Raises:
        HTTPException: whatever `compile_graphical_model` raises (e.g. 401 for a guest user).
    """
    if not request.force:
        existing = db_session.exec(
            select(ProblemDB)
            .where(ProblemDB.user_id == user.id, ProblemDB.name == _DEMO_IRRIGATION_PROBLEM_NAME)
            .order_by(ProblemDB.id.desc())
        ).first()
        if existing is not None and _demo_irrigation_metadata_matches(existing.id, db_session):
            return CompileResponse(problem_id=existing.id)

    network, annotation = build_irrigation_demo()
    compile_request = CompileRequest(network=network, annotation=annotation, name=_DEMO_IRRIGATION_PROBLEM_NAME)
    return compile_graphical_model(compile_request, user, db_session)


class SolveRequest(BaseModel):
    """Body for `POST /graphical/{problem_id}/solve`."""

    objective_symbol: str


class SolveResponse(BaseModel):
    """Response for `POST /graphical/{problem_id}/solve`."""

    variable_values: dict[str, list[float] | float]
    """`SolverResults.optimal_variables`, usable as `PolicyRequest.variable_values`."""


@router.post("/{problem_id}/solve")
def solve_graphical_model(
    problem_id: int,
    request: SolveRequest,
    user: Annotated[User, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> SolveResponse:
    """Solve a compiled diagram's `Problem` for one objective, via `guess_best_solver`.

    A minimal single-objective solve, so the policy of a solved problem can be shown via
    `/policy`. Only compiled graphical diagrams are accepted.

    Raises:
        HTTPException: 404 if the problem isn't found or visible, or isn't a compiled graphical
            diagram; 400 for an unknown `objective_symbol` or if no compatible solver exists;
            500 if the solver raises or reports failure.
    """
    problem_db = fetch_problem_with_role_check(user, problem_id, db_session)
    if problem_db is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Problem with id={problem_id} not found.")
    _fetch_graphical_metadata(problem_id, db_session)

    problem = Problem.from_problemdb(problem_db)
    if request.objective_symbol not in {objective.symbol for objective in problem.objectives}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{request.objective_symbol}' isn't one of problem {problem_id}'s own objectives.",
        )

    solvers = find_compatible_solvers(problem)
    if not solvers:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"No compatible solver found for problem {problem_id}."
        )

    # The `z` variables are binary, but `find_compatible_solvers` may still offer `CVXPYSolver`,
    # which may lack a MIP-capable backend. Keep only solvers known to handle MILPs.
    if problem.variable_domain in (VariableDomainTypeEnum.integer, VariableDomainTypeEnum.mixed):
        mip_capable_solvers = {GurobipySolver, PyomoGurobiSolver, PyomoCBCSolver, PyomoBonminSolver}
        solvers = [solver for solver in solvers if solver in mip_capable_solvers]
        if not solvers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No compatible MIP-capable solver found for problem {problem_id}.",
            )

    # `guess_best_solver` always returns something, so only use it if it passed the filters above.
    solver_class = guess_best_solver(problem)
    if solver_class not in solvers:
        solver_class = solvers[0]

    try:
        result = solver_class(problem).solve(request.objective_symbol)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Solver '{solver_class.__name__}' failed to solve problem {problem_id} "
            f"for '{request.objective_symbol}': {e}",
        ) from e
    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Solver failed to solve problem {problem_id} for '{request.objective_symbol}': {result.message}",
        )

    return SolveResponse(variable_values=result.optimal_variables)
