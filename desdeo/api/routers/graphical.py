"""Endpoints for compiling, inspecting, and querying graphical models.

Bayesian networks and influence diagrams -- Phase 7 of `.claude/desdeo-graphical-models-plan.md`.
Own router, own schemas -- request/response shapes are plain pydantic `BaseModel`s defined here,
not threaded through any existing router's request types, matching the plan's own rationale ("the
web-API is volatile and this keeps upstream churn to rebases rather than rewrites").

This module imports `pyagrum` directly (for `/graphical/whatif`'s live inference engines), unlike
`desdeo/problem/graphical/`'s own modules (`network.py`/`schema.py`/`rooted_junction_tree.py`/
`compile.py`/`risk.py`), which never do -- that dependency rule (see the plan's cross-cutting
section) is scoped to keeping the *compiler* package's algorithmic code testable without pyAgrum
in the loop; it does not extend to the API layer, whose whole job here is calling pyAgrum's own
inference engines. `to_pyagrum`/`to_pyagrum_influence_diagram` (structural conversion, reusable
outside this router) still live in `adapters/pyagrum.py`; the engine cache, evidence-setting, and
posterior/policy/utility read-back below are specific to this endpoint and stay here, per the
plan's "no threading through existing code paths" instruction.

**pyAgrum API facts used below, confirmed empirically (`uv run python -c "..."`), not assumed:**

- `LazyPropagation.setEvidence(dict[str, str])`/`ShaferShenoyLIMIDInference.setEvidence(...)`
  *replace* the engine's entire evidence set rather than merging with whatever was set on a
  previous call (confirmed: setting `{"A": "hi", "B": "lo"}` then `{"C": "hi"}` leaves only `C`
  evidenced, `nbrEvidence() == 1`; `setEvidence({})` clears everything). This is exactly what
  makes engine reuse across `/whatif` calls safe without ever caching evidence itself: each call's
  full evidence dict fully determines the engine's evidence state for that call, with no leakage
  from a previous call's evidence.
- `LazyPropagation.addJointTarget(...)` may be registered once, before the first `makeInference`,
  and then queried via `jointPosterior(...)` repeatedly across many later `setEvidence`/
  `makeInference` cycles on the same engine instance -- confirmed by inference under three
  different evidence states (including no evidence) all correctly reflecting `jointPosterior` for
  a target registered only once at the start. This is what lets one cached `LazyPropagation`
  engine serve many `/whatif` calls without re-declaring joint targets per call.
- `ShaferShenoyLIMIDInference.posterior(decision_symbol)` collapses to a deterministic 0/1 vector
  (matching the argmax of `optimalDecision(decision_symbol)`'s policy table) whenever the
  decision's effective information set is fully pinned by evidence -- confirmed on a 1-chance/
  1-decision diagram. Without full pinning it instead returns the evidence-induced *marginal*
  probability of each choice (a mixture over the still-uncertain information states), which is
  still a sensible per-node summary but not a single deterministic policy -- this is the intended,
  documented reading for this endpoint's `policy` field (see `_read_policy`).
- `ShaferShenoyLIMIDInference` has **no `addJointTarget`/`jointPosterior`** at all (confirmed via
  `dir()`) -- only single-node `posterior`/`posteriorUtility`. Combined with the Phase 2/4 findings
  (`MEU()` sums every utility node into one scalar; `posteriorUtility(node)` stays indexed by that
  node's own remaining parents rather than collapsing to a scalar), there is no verified way to
  recover one objective's own expected utility, independent of the others, once a diagram has
  *both* more than one `ObjectiveNode` and at least one decision -- see `_compute_utilities`'s
  `HTTPException(501, ...)` branch.
- `gum.GumException` is the common base of pyAgrum's own typed exceptions (`NotFound`,
  `OutOfBounds`, `FatalError`, ...) raised by e.g. `setEvidence` on an unknown node/label or
  `makeInference` on an unsolvable LIMID (a diagram with independent decisions and no derivable
  order, confirmed in Phase 4 -- see `compile.py`'s findings) -- caught here and reported as a 400
  rather than surfacing as an opaque 500.
"""

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Annotated, Literal

import numpy as np
import pyagrum as gum
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlmodel import Session, select

from desdeo.api.db import get_session
from desdeo.api.models import GraphicalModelMetaData, ProblemDB, ProblemMetaDataDB, User, UserRole
from desdeo.api.routers.user_authentication import get_current_user
from desdeo.api.routers.utils import fetch_interactive_session_with_role_check, fetch_problem_with_role_check
from desdeo.problem.graphical import DecisionAnnotation, DiscreteNetwork, build_rooted_junction_tree, compile_diagram
from desdeo.problem.graphical.adapters.pyagrum import to_pyagrum, to_pyagrum_influence_diagram

router = APIRouter(prefix="/graphical", tags=["Graphical models"])


# --- Request / response schemas -------------------------------------------------------------


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


class StructureEdge(BaseModel):
    """One directed edge in a `GET /graphical/{problem_id}/structure` response."""

    id: str
    """`f"{source}->{target}"` -- a stable id per edge (edges have no symbol of their own to use).
    A DAG has at most one edge per `(source, target)` pair, so this is always unique."""
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
    """Chance-node name -> pinned state label. Hard evidence only, matching
    `webui/.../network/types.ts`'s `NetworkEvidence`. Every call's evidence fully replaces
    whatever was set on a previous call against the same cached engine -- see the module
    docstring's "pyAgrum API facts" -- so evidence from one call never leaks into another."""


class WhatIfResponse(BaseModel):
    """Response for `POST /graphical/whatif`."""

    posteriors: dict[str, list[float]]
    """Chance-node name -> per-state probability, in that node's own `labels` order."""
    policy: dict[str, str] | None = None
    """Decision symbol -> chosen state label, for decisions whose effective information set is
    fully pinned by the given evidence (see `_read_policy`); `None` when `annotation.decisions` is
    empty. A decision left open by partial evidence is simply absent from this dict rather than
    reporting a most-likely-choice guess."""
    utilities: dict[str, float]
    """Objective symbol -> expected utility under the given evidence."""


# --- POST /graphical/compile ------------------------------------------------------------------


_SERVER_MAX_CLUSTER_SIZE = 200_000
"""Hard ceiling `/compile` enforces regardless of the caller's own `max_cluster_size`/`on_exceed`
choice. `/compile` runs synchronously (an anyio-threadpool-worker FastAPI endpoint), and
`compile_diagram`'s own cost is `O(state space)` per cluster (see its module docstring), so an
absent or overly generous client-supplied `max_cluster_size` would let one request block a worker
for an unbounded time -- a review pass caught the original version trusting the client's own value
entirely, with no server-side floor under it at all. Picked from Phase 4's own docstring note that
"`state_space_size` in the hundreds of thousands starts to matter" -- generous enough for any
reasonably-sized diagram, tight enough to bound one request's worst case. A request may ask for a
*stricter* cap than this (respected as given, at the caller's own `on_exceed`), but never a looser
one -- see `_effective_cluster_cap`."""


def _effective_cluster_cap(request: CompileRequest) -> tuple[int, Literal["warn", "raise"]]:
    """Clamp `request.max_cluster_size`/`on_exceed` to never exceed `_SERVER_MAX_CLUSTER_SIZE`.

    A caller-requested cap at or under the server ceiling is honoured exactly (including a `"warn"`
    choice, since it's already at least as strict as the server would enforce). A missing or looser
    caller-requested cap is replaced by the server ceiling at a forced `"raise"` -- silently
    downgrading to `"warn"` here would defeat the point of having a ceiling at all.
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

    Persists the compiled `Problem` (`ProblemDB`) plus a `GraphicalModelMetaData` side-table row
    carrying the network, annotation, `RootedJunctionTree`, and `IndexMap` -- all four JSON
    columns, matching Phase 3/4's persistence design (see `GraphicalModelMetaData`'s own
    docstring), in one atomic commit (a review pass caught an earlier version committing the
    `ProblemDB`, then the `ProblemMetaDataDB`, then the `GraphicalModelMetaData` as three separate
    transactions -- a failure on the last one left a permanently-orphaned problem with no graphical
    metadata and no repair path). Mirrors `routers/problem.py`'s `add_problem` guest check: guest
    users may not create new problems, graphical or otherwise.

    Raises:
        HTTPException: 401 for a guest user; 400 for a `ValueError` from `validate_against`/
            `build_rooted_junction_tree`/`compile_diagram` (bad/inconsistent annotation,
            `max_cluster_size` exceeded with `on_exceed="raise"`, an out-of-scope
            `DiagramConstraint`, ...) -- translated rather than left as an opaque 500.
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
        problem, index_map = compile_diagram(request.network, request.annotation, rjt=rjt)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e

    if request.name is not None:
        problem = problem.model_copy(update={"name": request.name})

    # One atomic commit at the end: `flush()` assigns each row's id (needed for the next row's FK)
    # without ending the transaction, so a failure anywhere in this block rolls everything back
    # together rather than leaving a partially-persisted problem.
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


# --- GET /graphical/{problem_id}/structure ----------------------------------------------------


def _fetch_graphical_metadata(problem_id: int, db_session: Session) -> GraphicalModelMetaData:
    """The most recently added `GraphicalModelMetaData` row for `problem_id`, or 404.

    Queries `GraphicalModelMetaData` directly, joined through `ProblemMetaDataDB`, ordered by
    `id.desc()` -- a review pass caught an earlier version picking `all_metadata`'s last element
    instead, which is **not** ordered by insertion at all (`ProblemMetaDataDB.all_metadata`
    concatenates plain, `order_by`-less `Relationship` lists). Nothing today ever inserts a second
    `GraphicalModelMetaData` row for the same problem, so this was latent rather than live, but an
    explicit `order_by` is what actually backs the "most recently added" claim this function makes,
    rather than relying on incidental list order.
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
    """DAG and node types only -- no inference, no pyAgrum. Serves the frontend's phase A.

    Ownership is checked via `fetch_problem_with_role_check` rather than `site_selection.py`'s
    own hand-rolled `_role_rank` dict -- both encode the same admin/analyst-bypass-ownership
    idea, but `fetch_problem_with_role_check` is the newer, already-shared helper (also used by
    `SessionContextGuard`) and additionally covers group-shared problems, which this read-only
    structure view has no reason to exclude.

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
        nodes.append(StructureNode(id=variable.name, label=variable.name, kind="chance", states=variable.labels))
    for parent, child in network.arcs:
        _add_edge(parent, child)

    for decision in annotation.decisions:
        # No id collision possible against a chance node here: `validate_against` (already run at
        # `/compile` time) rejects a decision symbol colliding with a chance-node name.
        nodes.append(
            StructureNode(
                id=decision.symbol, label=decision.name or decision.symbol, kind="decision", states=decision.states
            )
        )
        for parent in decision.information_set:
            _add_edge(parent, decision.symbol)

    for objective in annotation.objectives:
        # Unlike decisions, nothing forbids an `ObjectiveNode.symbol` from colliding with a
        # chance-node name (a review pass caught this -- `DecisionAnnotation.validate_against`
        # only checks the decision/chance namespaces against each other, deliberately left that
        # way so an existing regression test in `tests/test_graphical_compile.py` can exercise the
        # collision at the constraint-substitution layer instead -- see that test's own docstring).
        # Namespaced here as `util_<symbol>`, the same prefix `to_pyagrum_influence_diagram`
        # already uses internally for a diagram's utility nodes, so this endpoint's node ids can
        # never collide across kinds even though the underlying symbols could.
        objective_id = f"util_{objective.symbol}"
        nodes.append(StructureNode(id=objective_id, label=objective.name or objective.symbol, kind="objective"))
        for parent in objective.parents:
            _add_edge(parent, objective_id)

    return StructureResponse(nodes=nodes, edges=edges)


# --- Session-scoped inference engine cache -----------------------------------------------------

_CACHE_TTL_SECONDS = 30 * 60
"""Idle-eviction threshold. 30 minutes: long enough that a DM exploring what-if scenarios in one
sitting isn't punished by a full rebuild between clicks, short enough that a session nobody ever
explicitly tears down doesn't hold a live engine indefinitely."""

_CACHE_MAX_SIZE = 128
"""LRU cap, evicted least-recently-used first once exceeded -- same bounded-cache precedent as
`rooted_junction_tree.py`'s RJT cache and `compile.py`'s `_position_map`, sized for "many distinct
sessions/diagrams explored over one server's lifetime", not for any one deployment's expected
concurrency."""

_cache_lock = threading.Lock()
"""Guards `_engine_cache`'s dict operations (lookup/insert/evict) only -- not each entry's own
`lock`, which guards using that entry's engine (see `_EngineCacheEntry.lock`)."""


@dataclass
class _EngineCacheEntry:
    """One cached pyAgrum inference engine plus the bookkeeping needed to reuse it safely."""

    engine: "gum.LazyPropagation | gum.ShaferShenoyLIMIDInference"
    content_hash: str
    """`_content_hash(network_json, annotation_json)` at build time -- a cache hit requires an
    exact match, so a recompiled/re-edited model (different network or annotation JSON) is
    detected and the engine rebuilt, rather than silently serving inference against a stale
    structure/CPTs."""
    last_used: float
    """`time.monotonic()` timestamp, updated on every access -- drives both TTL and LRU eviction."""
    lock: threading.Lock = field(default_factory=threading.Lock)
    """Serialises `setEvidence`/`makeInference`/read-back for *this* entry specifically, so two
    concurrent `/whatif` calls sharing one cache key (same session, or same synthetic per-user-
    per-problem key) can't interleave their evidence-set/inference/read steps -- without this, one
    call's `setEvidence` could run before another's read, corrupting both calls' results even
    though `setEvidence` itself is safe to call repeatedly. Per-entry (not one global lock) so
    unrelated sessions/keys still run concurrently."""


_engine_cache: dict[str, _EngineCacheEntry] = {}
"""Process-local, in-memory cache of pyAgrum inference engines, keyed by session id (when given)
or a synthetic `user:<id>:problem:<id>` key otherwise. One entry per key, matching the plan's own
`{session_id: (model_hash, engine)}` shape literally -- not `{(session_id, problem_id): ...}` --
so switching between two different graphical models within the *same* session id thrashes this
cache (each switch is a hash mismatch, forcing a rebuild) rather than keeping both warm. Accepted
as a minor, documented inefficiency rather than a correctness issue: a rebuild is always correct,
just not free, and the plan's own literal cache-key shape doesn't nest by problem.

**Single-worker only.** This is a plain module-level dict, not shared across worker processes
(`uvicorn --workers > 1`, gunicorn) or machines -- a live pyAgrum engine is a SWIG-wrapped C++
object, not something that can be pickled into an external cache (redis, ...) without first
solving a separate serialisation problem this phase doesn't attempt. A multi-worker deployment
needs either sticky sessions (route every request for a given key to the same worker process) or
an external cache holding a serialisable representation instead of a live engine, before this
scales past one worker -- see the plan's cross-cutting risk table entry for exactly this. Flagged
here, not solved, per the plan's explicit Phase 7 scope.

**Evidence is never stored here.** Only the built engine (network structure + CPTs/utility
tables) is cached; every `/graphical/whatif` call supplies its own full evidence dict via
`setEvidence`, which *replaces* the engine's evidence state rather than merging with a previous
call's (confirmed empirically -- see the module docstring's "pyAgrum API facts"), so cached-engine
reuse never lets one call's evidence leak into another's.
"""


def _content_hash(network_json: str, annotation_json: str) -> str:
    """Fingerprint of `(network, annotation)`'s *content* -- what invalidates a cached engine."""
    return hashlib.sha256(f"{network_json}\x00{annotation_json}".encode()).hexdigest()


def _evict_stale(now: float) -> None:
    """Drop TTL-expired entries, then trim to `_CACHE_MAX_SIZE` by least-recently-used.

    Called opportunistically on every cache access (no background sweep thread -- see the cache's
    own docstring on why that's an acceptable simplification here): "eventually evicted" is enough
    for a process-local convenience cache, not "evicted at exactly the TTL boundary".
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

    No-decision case: a plain `BayesNet` + `LazyPropagation` -- `ShaferShenoyLIMIDInference`
    refuses outright ("This LIMID/Influence Diagram is not solvable") on a diagram with zero
    decision nodes (Phase 4 finding, `compile.py`'s module docstring). Every `ObjectiveNode`'s
    joint target is registered once, here, before any `makeInference` call -- see the module
    docstring's "pyAgrum API facts" for why that's what lets the cached engine answer many later
    `/whatif` calls under different evidence without re-declaring targets each time.

    With-decision case: `to_pyagrum_influence_diagram` + `ShaferShenoyLIMIDInference`, applying
    `addNoForgettingAssumption` only when `annotation.no_forgetting` is set -- `decision_order`
    alone is *not* sufficient (Phase 4 finding), and the assumption wants decision *names*
    (`list[str]`), not the ids its own docstring claims.
    """
    if not annotation.decisions:
        engine = gum.LazyPropagation(to_pyagrum(network))
        for objective in annotation.objectives:
            if objective.parents:
                engine.addJointTarget(set(objective.parents))
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

    Takes `network_json`/`annotation_json` as the raw strings already sitting in
    `GraphicalModelMetaData` (the caller just parsed them into `network`/`annotation` and needs
    those parsed forms regardless, for `_build_engine` on a miss and for the rest of `whatif`) --
    a review pass caught an earlier version re-serialising `network`/`annotation` back to JSON via
    `model_dump_json()` on every call just to fingerprint them, a full parse-then-reserialise round
    trip of the entire network (every CPT included) on the hot path, for data that was already a
    string in hand.
    """
    content_hash = _content_hash(network_json, annotation_json)
    now = time.monotonic()

    with _cache_lock:
        _evict_stale(now)
        entry = _engine_cache.get(cache_key)
        if entry is not None and entry.content_hash == content_hash:
            entry.last_used = now
            return entry

    # Built outside the lock: a review pass caught the original version building (the expensive
    # step -- compiling a pyAgrum diagram and constructing its inference engine) while holding
    # `_cache_lock`, a single process-global lock, which serialised *every* concurrent `/whatif`
    # request in the process -- cache hits included -- behind whichever request happened to be
    # building. Two concurrent misses for the same key duplicate a build rather than one stalling
    # behind the other; the loser's result is discarded by the re-check below.
    built = _EngineCacheEntry(engine=_build_engine(network, annotation), content_hash=content_hash, last_used=now)

    with _cache_lock:
        _evict_stale(now)
        entry = _engine_cache.get(cache_key)
        if entry is not None and entry.content_hash == content_hash:
            entry.last_used = now
            return entry
        _engine_cache[cache_key] = built
        return built


# --- POST /graphical/whatif ---------------------------------------------------------------------


def _read_posteriors(engine, network: DiscreteNetwork) -> dict[str, list[float]]:
    """`{chance_node_name: [P(state) for state in variable.labels]}`, current evidence applied."""
    return {variable.name: engine.posterior(variable.name).toarray().tolist() for variable in network.variables}


def _read_policy(engine, annotation: DecisionAnnotation) -> dict[str, str] | None:
    """`{decision_symbol: chosen_state_label}`, decisions fully determined by evidence only.

    `None` if there are no decisions at all; a decision whose choice isn't yet determined is
    simply absent from the dict (see below), never `None`-valued.

    `posterior(decision_symbol)` collapses to a deterministic one-hot vector exactly when the
    decision's effective information set is fully pinned (confirmed empirically, see the module
    docstring); otherwise it's an evidence-induced *marginal* over the still-uncertain choices, not
    a single policy. A review pass caught an earlier version of this function reporting that
    marginal's argmax as if it were "the" chosen state regardless -- e.g. an unconditional
    marginal of `[0.3, 0.7]` (no evidence at all pinning the decision's own information set) used
    to report a confident-looking `"s1"` even though `s1` is not what any actual policy chooses
    when the information state turns out to be the 0.3 branch. Fixed by only including a decision
    here when its posterior is genuinely one-hot; a decision left open by partial evidence is
    omitted from the response instead of a most-likely-choice guess. This also sidesteps having to
    reason about `information_set` structure directly -- it may reference other decisions, not
    just chance nodes (`DecisionNode.information_set`'s own docstring) -- since the one-hot check
    is agnostic to *why* a decision's outcome is or isn't yet determined.
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


def _expected_value_no_decision(engine, objective) -> float:
    """One `ObjectiveNode`'s expected utility from a `LazyPropagation` engine (no decisions).

    Generalizes `tests/test_graphical_compile.py`'s `_pyagrum_meu` no-decision contraction
    (`addJointTarget` + `jointPosterior` + contract against `objective.utility`) to run
    independently per objective -- no summing-across-objectives issue in this branch, since each
    objective's own joint target/posterior is looked up by its own `parents` set.

    `objective.parents == []` (a constant, parent-less objective) is intentionally not
    special-cased here: `ObjectiveNode._check_utility_matches_parents` forces
    `utility.variables == parents` and `validate_against` forces `utility.shape` to match, so an
    empty `parents` would need `shape == []`, which `Potential`'s own shape validator can never
    satisfy -- the identical dead branch Phase 4's review pass already found and deleted from
    `compile.py` for exactly this reason (see the plan's Phase 4 findings). This copy, inherited
    from the test helper this function was generalised from, is removed rather than kept "just in
    case", matching that precedent.
    """
    posterior = engine.jointPosterior(set(objective.parents))
    order = [v.name() for v in posterior.variablesSequence()]
    probs = np.transpose(posterior.toarray())
    utility = np.array(objective.utility.values, dtype=float)
    probs = np.moveaxis(probs, [order.index(p) for p in objective.parents], range(len(objective.parents)))
    return float(np.sum(probs * utility))


def _compute_utilities(engine, annotation: DecisionAnnotation) -> dict[str, float]:
    """`{objective_symbol: expected_utility}` under the engine's current evidence.

    No decisions: independent per-objective contraction (`_expected_value_no_decision`), any
    number of objectives.

    Exactly one decision-diagram objective: `ShaferShenoyLIMIDInference.MEU()["mean"]` is
    oracle-proven correct against brute force (Phase 4 finding) -- with only one utility node,
    pyAgrum's own utility-summing has nothing else to sum in.

    More than one objective *and* at least one decision: **not implemented.** Confirmed
    empirically (module docstring) that `ShaferShenoyLIMIDInference` has no joint-posterior
    primitive at all, so there is no verified way to contract each objective's own utility table
    against the induced joint distribution over its own parents the way the no-decision branch
    does; `MEU()` would silently sum every objective into one scalar (Phase 2 finding), and
    `posteriorUtility(node)` doesn't collapse to a scalar either (Phase 4 finding). Raising rather
    than shipping an unverified number, the same scope-cut style Phase 5 used for
    `DiagramConstraint`'s `kind == "tail"`.
    """
    if not annotation.decisions:
        return {objective.symbol: _expected_value_no_decision(engine, objective) for objective in annotation.objectives}

    if len(annotation.objectives) == 1:
        objective = annotation.objectives[0]
        # `to_pyagrum_influence_diagram` writes the utility table in pyAgrum's own maximize-only
        # sign convention (negated when `objective.maximize` is False -- see that function's
        # docstring), so `MEU()["mean"]` must be negated back by the same sign to report the
        # value in `objective`'s own units.
        sign = 1.0 if objective.maximize else -1.0
        return {objective.symbol: sign * engine.MEU()["mean"]}

    msg = (
        "Computing per-objective expected utility is not implemented for a diagram with more "
        "than one ObjectiveNode and at least one decision: ShaferShenoyLIMIDInference exposes no "
        "joint-posterior primitive to contract an objective's own utility table against the "
        "induced distribution over its own parents (confirmed empirically -- see this module's "
        "docstring), MEU() sums every utility node into one scalar, and posteriorUtility(node) "
        "does not collapse to a scalar either (see the plan's Phase 2/4 findings)."
    )
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=msg)


@router.post("/whatif")
def whatif(
    request: WhatIfRequest,
    user: Annotated[User, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> WhatIfResponse:
    """Set hard evidence, run inference, and return posteriors/policy/utilities in one payload.

    Reuses a session- (or, without a `session_id`, a per-user-per-problem-) scoped cached pyAgrum
    engine across calls (see `_engine_cache`'s docstring) -- compiling/rebuilding the underlying
    diagram is comparatively expensive, evidence propagation is cheap, so the model itself is
    never resent or rebuilt unless it actually changed.

    Raises:
        HTTPException: 404 if the problem doesn't exist, isn't visible to `user`, or has no
            `GraphicalModelMetaData`; 404 (via `fetch_interactive_session_with_role_check`) if
            `session_id` is given but doesn't exist or isn't owned by `user`; 400 if `evidence`
            references a node that isn't one of `network`'s chance nodes, or if pyAgrum itself
            rejects the evidence/diagram (`gum.GumException` -- an unknown node/label, or an
            unsolvable LIMID); 501 for the open multi-objective-with-decisions utilities case
            (see `_compute_utilities`).
    """
    problem_db = fetch_problem_with_role_check(user, request.problem_id, db_session)
    if problem_db is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Problem with id={request.problem_id} not found."
        )

    meta = _fetch_graphical_metadata(request.problem_id, db_session)
    network = DiscreteNetwork.model_validate_json(meta.network_json)
    annotation = DecisionAnnotation.model_validate_json(meta.annotation_json)

    chance_names = {variable.name for variable in network.variables}
    unknown_evidence = sorted(set(request.evidence) - chance_names)
    if unknown_evidence:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Evidence references node(s) that aren't DiscreteNetwork chance nodes: "
                f"{unknown_evidence}. Only hard evidence on chance nodes is supported."
            ),
        )

    if request.session_id is not None:
        # Raises 404 itself if not found/owned -- see fetch_interactive_session_with_role_check.
        fetch_interactive_session_with_role_check(user, request.session_id, db_session)
        cache_key = f"session:{request.session_id}"
    else:
        cache_key = f"user:{user.id}:problem:{request.problem_id}"

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
        utilities = _compute_utilities(entry.engine, annotation)

    return WhatIfResponse(posteriors=posteriors, policy=policy, utilities=utilities)
