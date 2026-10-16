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
- `ShaferShenoyLIMIDInference.MEU()` returns `{"mean": ..., "variance": ...}` (confirmed
  empirically -- `uv run python -c "..."` against a small hand-built `InfluenceDiagram`, not
  assumed from the plan's own Phase E checklist wording, which only promised "mean and variance"
  without naming the keys). Used by `_compute_utilities`'s standard-deviation return (Phase E, MEU
  with its standard deviation) -- see that function's own docstring for the sign-convention
  reasoning (the mean needs the same maximize/minimize sign flip `MEU()["mean"]` already needed;
  the variance does not, since `Var(-X) == Var(X)`).
"""

import hashlib
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from math import sqrt
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
    description: str | None = None
    """Chance nodes only: `DiscreteVariable.description`, when non-empty -- `None` for a chance
    node with no description set, and always `None` for decision/objective nodes (`DecisionNode`/
    `ObjectiveNode` have no description field of their own to thread through). A review pass
    caught `desdeo/problem/graphical/demo.py` setting a `description` on each of its
    `DiscreteVariable`s with no way for it to ever reach the frontend (`StructureNode` had no such
    field at all) -- added here, and threaded through by `get_structure` below, rather than
    dropping `demo.py`'s descriptions as dead data, since `NetworkModel`'s own node shape
    (`webui/.../network/types.ts`) already has a `description` field ready to receive it."""


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
    utility_std_devs: dict[str, float]
    """Objective symbol -> standard deviation of the utility distribution under the given
    evidence (Phase E: "MEU with its standard deviation"), same keys/shape as `utilities` --
    `sqrt(MEU()["variance"])` for the decision-diagram case, `sqrt(E[U^2] - E[U]^2)` (the same
    joint-posterior contraction `utilities`' own no-decision branch already computes) otherwise.
    See `_compute_utilities`'s own docstring for why (unlike `utilities`) this never needs the
    maximize/minimize sign flip."""


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

_SERVER_MAX_TOTAL_CLUSTER_SIZE = 2_000_000
"""Hard ceiling on the *sum* of every cluster's own `state_space_size`, unconditional (no
caller-supplied override, unlike `_SERVER_MAX_CLUSTER_SIZE`). `_SERVER_MAX_CLUSTER_SIZE` only
bounds the single largest cluster -- a network built from many small, disjoint cliques (each well
under that cap individually) can still produce an arbitrarily large *total* compiled `Problem`,
since `compile_diagram`'s overall cost scales with the sum across clusters, not just its max. Set
an order of magnitude above `_SERVER_MAX_CLUSTER_SIZE` (10x): generous enough that a legitimate
diagram with several independent, moderately-sized clusters still compiles, tight enough to still
bound one request's worst-case total work."""


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
            `DiagramConstraint`, ...) -- translated rather than left as an opaque 500; 400 if the
            sum of every cluster's own state space exceeds `_SERVER_MAX_TOTAL_CLUSTER_SIZE` (a
            many-small-disjoint-clusters diagram that never trips the single-largest-cluster cap).
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
    `/whatif` calls under different evidence without re-declaring targets each time. Every chance
    node is *also* registered as its own single-node target here -- confirmed empirically that
    `LazyPropagation.addJointTarget(...)` silently narrows its default "every node is a target"
    behaviour to *only* the explicitly declared (joint) targets plus any hard-evidenced node, so
    without this, `posterior(node)` for any chance node that's neither a declared joint target's
    own member nor evidenced raises `gum.UndefinedElement` ("does not belong to this optimized
    inference") -- a review pass caught this breaking `/sensitivity`'s own near-zero-marginal
    check (`sensitivity`'s `base_marginals`) for a chance node unrelated to every objective's own
    parents. `ShaferShenoyLIMIDInference` (the with-decision branch below) has no such
    restriction -- confirmed empirically it answers `posterior` for any chance/decision node
    regardless of target declarations -- so this only applies here.
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


def _expected_value_no_decision(engine, objective) -> tuple[float, float]:
    """One `ObjectiveNode`'s `(expected_utility, variance)` from a `LazyPropagation` engine.

    Generalizes `tests/test_graphical_compile.py`'s `_pyagrum_meu` no-decision contraction
    (`addJointTarget` + `jointPosterior` + contract against `objective.utility`) to run
    independently per objective -- no summing-across-objectives issue in this branch, since each
    objective's own joint target/posterior is looked up by its own `parents` set.

    Variance (Phase E, "MEU with its standard deviation") is `E[U^2] - E[U]^2` over the identical
    `probs`/`utility` pair the mean already needs -- computed alongside it, not as a second
    `jointPosterior` pass. `LazyPropagation` has no `MEU()` of its own to ask (that's a
    `ShaferShenoyLIMIDInference`-only method -- see `_compute_utilities`'s other branch), so this
    is the no-decision-case equivalent. Clamped at `0.0`: floating-point roundoff on
    `E[U^2] - E[U]^2` can go very slightly negative for a (near-)deterministic distribution whose
    true variance is exactly `0`.

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
    mean = float(np.sum(probs * utility))
    variance = max(float(np.sum(probs * utility**2) - mean**2), 0.0)
    return mean, variance


def _compute_utilities(engine, annotation: DecisionAnnotation) -> tuple[dict[str, float], dict[str, float]]:
    """`({objective_symbol: expected_utility}, {objective_symbol: std_dev})`, current evidence.

    No decisions: independent per-objective contraction (`_expected_value_no_decision`), any
    number of objectives.

    Exactly one decision-diagram objective: `ShaferShenoyLIMIDInference.MEU()` returns
    `{"mean": ..., "variance": ...}` (confirmed empirically -- see the module docstring's "pyAgrum
    API facts"); `["mean"]` is oracle-proven correct against brute force (Phase 4 finding) -- with
    only one utility node, pyAgrum's own utility-summing has nothing else to sum in.

    Sign convention (Phase E): `to_pyagrum_influence_diagram` writes the utility table in
    pyAgrum's own maximize-only sign convention (negated when `objective.maximize` is False -- see
    that function's docstring), so the *mean* needs the same negate-back `sign` correction
    `MEU()["mean"]` already needed before Phase E existed. The *variance* does not: `Var(-X) ==
    Var(X)` (`Var(aX) = a^2 * Var(X)`, and `a = -1` here), so a variance computed over the
    negated-for-pyAgrum utility table is already numerically identical to the variance over the
    objective's own (un-negated) values -- flipping its sign would be wrong twice over (variance
    is already non-negative by construction, and the quantity doesn't change sign in the first
    place). This is worth stating explicitly rather than assumed, per the plan's own Phase E
    checklist wording ("think through this rather than assuming").

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
        means: dict[str, float] = {}
        std_devs: dict[str, float] = {}
        for objective in annotation.objectives:
            mean, variance = _expected_value_no_decision(engine, objective)
            means[objective.symbol] = mean
            std_devs[objective.symbol] = sqrt(variance)
        return means, std_devs

    if len(annotation.objectives) == 1:
        objective = annotation.objectives[0]
        # `to_pyagrum_influence_diagram` writes the utility table in pyAgrum's own maximize-only
        # sign convention (negated when `objective.maximize` is False -- see that function's
        # docstring), so `MEU()["mean"]` must be negated back by the same sign to report the
        # value in `objective`'s own units. The variance needs no such correction -- see this
        # function's own docstring.
        sign = 1.0 if objective.maximize else -1.0
        meu = engine.MEU()
        return {objective.symbol: sign * meu["mean"]}, {objective.symbol: sqrt(max(meu["variance"], 0.0))}

    msg = (
        "Computing per-objective expected utility is not implemented for a diagram with more "
        "than one ObjectiveNode and at least one decision: ShaferShenoyLIMIDInference exposes no "
        "joint-posterior primitive to contract an objective's own utility table against the "
        "induced distribution over its own parents (confirmed empirically -- see this module's "
        "docstring), MEU() sums every utility node into one scalar, and posteriorUtility(node) "
        "does not collapse to a scalar either (see the plan's Phase 2/4 findings)."
    )
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail=msg)


def _resolve_cache_key(user: User, session_id: int | None, problem_id: int, db_session: Session) -> str:
    """`f"session:{id}"` (ownership-checked) if `session_id` is given, else a per-user-per-problem key.

    Factored out of what was `whatif`'s own inline block so `/sensitivity`/`/voi` (Phase E) share
    the identical session-resolution/ownership-check semantics rather than a second, potentially
    drifting copy -- all three endpoints reuse the same session-scoped engine cache (`_engine_
    cache`) for the same reason (compiling/rebuilding the diagram is expensive, evidence
    propagation is cheap).
    """
    if session_id is not None:
        # Raises 404 itself if not found/owned -- see fetch_interactive_session_with_role_check.
        fetch_interactive_session_with_role_check(user, session_id, db_session)
        return f"session:{session_id}"
    return f"user:{user.id}:problem:{problem_id}"


def _validate_chance_evidence(network: DiscreteNetwork, evidence: dict[str, str]) -> None:
    """400 if `evidence` references a bad chance node or state label.

    A node that isn't one of `network`'s own chance nodes, or a state label that isn't one of
    that node's own `labels`. Factored out of `whatif`'s own inline check so `/sensitivity`/`/voi`/evidence-set creation
    (Phase E) enforce the identical "hard evidence is chance-nodes-only, with a real state label"
    restriction, matching `/whatif`'s own. The label check is new (a review pass caught evidence-
    set creation validating node names but never state labels: a typo'd label used to save
    successfully -- 200 -- and only fail later, opaquely, the first time anything tried to
    actually use the set, since pyAgrum's own `setEvidence` is what used to catch it, well
    downstream of the save). `/whatif` already turns that same pyAgrum rejection into a 400 via
    its `gum.GumException` handler, so centralising the check here only tightens its error message
    (a clean, this-module-authored 400 instead of a raw pyAgrum exception string), not its status
    code -- no behavioural change to `/whatif` itself.
    """
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

    return WhatIfResponse(posteriors=posteriors, policy=policy, utilities=utilities, utility_std_devs=utility_std_devs)


# --- POST /graphical/{problem_id}/policy --------------------------------------------------------
#
# Phase E ("Read decision policies out of solutions via IndexMap; display read-only in
# decisionMode: 'solver', updating as the DM navigates"). Distinct from `/whatif`'s own `policy`
# field: that one reads a *live pyAgrum engine's* evidence-conditioned posterior (an ID-solving
# query, "what would the optimal policy choose"); this one decodes a *specific, already-solved*
# `Problem`'s `z` values via `IndexMap`/`read_policy_from_solution` (`problem/graphical/compile.py`)
# -- "what does *this* solved policy choose given what's currently observed", with no pyAgrum call
# at all. Both answer the question a `decisionMode: 'solver'` UI needs, from two different sources
# of truth (a live ID-solve vs. a specific stored solution) -- see the plan's own Phase E wording.


class PolicyRequest(BaseModel):
    """Body for `POST /graphical/{problem_id}/policy`."""

    variable_values: dict[str, list[float] | float] | None = None
    """A solved `Problem`'s `SolverResults.optimal_variables` -- must include every decision's own
    `z_symbol` (`GraphicalModelMetaData.index_map_json` / `IndexMap.decisions`). Exactly one of
    this or `solution_id` must be given. Typed to accept a bare scalar per entry, not just
    `list[float]`: a diagram compiled with a Phase 6 `RiskMeasure` has a scalar `eta` `Variable` in
    `optimal_variables` (a bare `float`, not a length-1 list) alongside every decision's own
    list-valued `z_symbol` -- a review pass caught the narrower `dict[str, list[float]]` 422ing on
    exactly this real shape of solved-solution dict, the same shape `solution_id`'s own
    `UserSavedSolutionDB.variable_values` path already accepts without complaint."""
    solution_id: int | None = None
    """Id of a `UserSavedSolutionDB` row already holding a solved `Problem`'s variable values for
    this same `problem_id` -- the plan's own "or a reference to an already-stored solution"
    wording, so a caller doesn't have to resend the whole solved vector on every navigation step.
    Must belong to `problem_id`."""
    evidence: dict[str, str] = {}
    """Current hard evidence (chance-node name -> pinned state label) -- same restriction as
    `WhatIfRequest.evidence`. Used to decode which of a decision's information states is 'the'
    currently applicable one, exactly the way `_read_policy` does for the live-engine case."""


class PolicyResponse(BaseModel):
    """Response for `POST /graphical/{problem_id}/policy`."""

    policy: dict[str, str]
    """Decision symbol -> chosen state label, for decisions whose effective information set is
    fully pinned by `evidence` -- same shape/semantics as `WhatIfResponse.policy` (minus the
    `None`-when-no-decisions case: an empty dict already conveys that here), so a
    `decisionMode: 'solver'` UI can treat a live-engine (`/whatif`) and a solved-MILP (`/policy`)
    answer identically, reusing the same `NetworkPolicy` badge plumbing. A decision left open by
    partial evidence is simply absent, never a most-likely-choice guess."""


def _decode_solved_policy(
    index_map: IndexMap,
    network: DiscreteNetwork,
    annotation: DecisionAnnotation,
    variable_values: dict[str, list[float] | float],
    evidence: dict[str, str],
) -> dict[str, str]:
    """`{decision_symbol: chosen_state_label}` -- `read_policy_from_solution`, collapsed via `evidence`.

    `read_policy_from_solution` (`compile.py`) returns each decision's *full* index-based policy
    function; this is the API-layer translation the plan's own checklist item asks for -- decode
    which single information state `evidence` currently picks out (labels -> indices, via each
    info-set member's own state list), and look up that state's chosen label. A decision is
    included only when every one of its effective information-set members is resolved (mirroring
    `_read_policy`'s own "omit, don't guess" semantics for the live-engine case) --
    `information_set` may itself reference other decisions, not just chance nodes
    (`DecisionNode.information_set`'s own docstring), so `label_lists` covers both namespaces.

    Decoded in `annotation.decision_order` (falling back to `annotation.decisions`'s own
    declaration order when unset -- LIMID semantics with no total order) rather than
    `policies`' own dict-insertion order, accumulating each resolved decision's chosen label into
    a working copy of `evidence` before resolving the next decision. This is what lets a
    downstream decision whose *effective* information set includes an *upstream* decision (the
    `no_forgetting`/multi-decision case: `DecisionIndexMap.information_set` is the *effective*,
    `no_forgetting`-expanded set) resolve correctly -- `evidence` itself is validated to be
    chance-node-only (`_validate_chance_evidence`), so it alone could never satisfy a
    decision-valued information-set member; a review pass caught the previous per-`policies`-item
    version unconditionally omitting every such downstream decision, with no error or warning,
    even though `/whatif`'s own live-engine `_read_policy` correctly resolves it for the identical
    evidence.
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
        # `read_policy_from_solution` raises `ValueError` when a `z_symbol` entry's length
        # doesn't match this IndexMap's own expected shape (e.g. a stale saved solution from
        # before a recompile changed a cardinality) -- a client-correctable 400, not an
        # uncaught 500 (a review pass caught this path falling through uncaught). `TypeError` is
        # caught alongside it for the same reason: `variable_values` is typed to accept a bare
        # scalar per entry (the Phase 6 `RiskMeasure` `eta` case -- see `PolicyRequest.
        # variable_values`'s own docstring), but nothing stops a caller putting a bare scalar on a
        # decision's own `z_symbol` key instead, and `len(z_values)` on a bare `float` raises
        # `TypeError` rather than `ValueError` (a review pass caught this escaping as an uncaught
        # 500 instead of the same clean 400 a wrong-length list already gets).
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
        # `break` above (an information-set member not yet resolved) skips the `for...else` body
        # -- this decision (and, transitively, anything downstream of it) is simply omitted,
        # matching `_read_policy`'s "don't guess" rule.
    return policy


@router.post("/{problem_id}/policy")
def get_policy(
    problem_id: int,
    request: PolicyRequest,
    user: Annotated[User, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> PolicyResponse:
    """Decode a *solved* `Problem`'s policy against `evidence` -- no live pyAgrum engine involved.

    Raises:
        HTTPException: 404 if the problem doesn't exist, isn't visible to `user`, has no
            `GraphicalModelMetaData`, has no `index_map_json` yet (compiled via a path that never
            ran `compile_diagram`), or `solution_id` doesn't reference a saved solution belonging
            to `problem_id`; 403 if `solution_id` references a solution saved by a different user
            and `user` is neither its owner nor an admin/analyst (visibility into `problem_id`
            alone is not ownership of one specific saved solution); 400 if neither/both of
            `variable_values`/`solution_id` are given, if `evidence` references a node that isn't
            a chance node, if `variable_values` is missing a required `z_symbol` entry or has one
            in the wrong shape, or if an evidence value isn't one of its own node's states.
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
        # Group-visibility into `problem_id` (checked above, via `fetch_problem_with_role_check`)
        # is not ownership of this *specific* saved solution -- mirrors `delete_evidence_set`'s own
        # role-aware ownership check (`row.user_id != user.id and user.role not in (...)`), the
        # precedent `test_evidence_set_delete_visible_non_owner_is_403` already establishes for
        # exactly this kind of check elsewhere in this router.
        if solution.user_id != user.id and user.role not in (UserRole.analyst, UserRole.admin):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only the saving user (or an admin/analyst) may read back this solution's policy.",
            )
        variable_values = solution.variable_values

    _validate_chance_evidence(network, request.evidence)
    policy = _decode_solved_policy(index_map, network, annotation, variable_values, request.evidence)
    return PolicyResponse(policy=policy)


# --- Named evidence sets (Phase E) --------------------------------------------------------------
#
# "Named evidence sets stored server-side, selectable as comparison baseline, shareable by URL" --
# see `GraphicalEvidenceSetDB`'s own docstring (`api/models/graphical.py`) for why this is its own
# simple table rather than routed through `ProblemMetaDataDB`'s polymorphic pattern.


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
            `GraphicalModelMetaData`; 400 if `evidence` references a node that isn't one of the
            network's own chance nodes -- validated eagerly here (not deferred to whatever
            eventually loads this set) so a bad save fails immediately, at the point the DM can
            still fix it.
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

    Listing is deliberately *not* restricted to the requesting user's own saved sets: a saved
    evidence set is meant to be a reproducible, shareable bookmark ("a figure corresponds to a
    scenario id, not a click sequence" -- the plan's own rationale) for anyone with visibility
    into the underlying problem, the same visibility rule `/structure`/`/whatif` already use --
    not a private-to-the-creator list. Creation/deletion stay owner-restricted (see those
    endpoints); reading does not.

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
    """Fetch one named evidence set by its own id -- "shareable by URL", literally.

    Not scoped under `/{problem_id}/...`: the id alone is the stable, addressable handle the plan's
    checklist asks for ("given the set's id, another request ... can fetch it and get back the
    same evidence dict, deterministically"); visibility is still checked against the row's own
    `problem_id`, so sharing the id with someone who can't see the underlying problem still 404s.

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


# --- POST /graphical/sensitivity, POST /graphical/voi -------------------------------------------
#
# Phase E: "Sensitivity tornado: single-node evidence sweep, Sigma|states| queries on a warm
# engine" and "Value of information, same machinery, framed as EU gain from observing before
# deciding". Both reuse `_engine_cache`/`_get_or_build_entry` (the same warm-engine cache
# `/whatif` already uses) and the shared `_sweep_node` primitive below -- see that function's own
# docstring for why VoI is a thin transform over the identical sweep rather than a parallel
# implementation.


def _resolve_target_value(
    engine, network: DiscreteNetwork, annotation: DecisionAnnotation, target: str, target_state: str | None
) -> float:
    """`target`'s current scalar value under `engine`'s current evidence state.

    `target` an `ObjectiveNode.symbol` -> its expected utility. `target` a chance/decision node's
    own name -> `P(target == target_state)` (`target_state` required in that case, and must be one
    of that node's own states) -- `engine.posterior(...)` works for chance and decision nodes
    alike (already relied on by `_read_posteriors`/`_read_policy`).

    No-decision diagrams call `_expected_value_no_decision` directly for just `target`, rather
    than routing through `_compute_utilities`'s multi-objective wrapper and discarding every
    result but `target`'s own -- a review pass caught the original version doing exactly that: on
    a diagram with N objectives, every swept state (a caller here is always inside `_sweep_node`,
    a per-state loop) redundantly recomputed all N objectives' expected utilities instead of just
    the one being asked for, defeating the "Sigma|states| queries on a warm engine" cost model
    `/sensitivity`/`/voi` are built around. `_expected_value_no_decision` is already exactly the
    single-objective primitive needed here -- `_compute_utilities`'s own no-decision branch is
    just this, looped over every objective.

    With-decision diagrams still route through `_compute_utilities` unchanged, including its
    multi-objective-with-decision `501` scope-cut (see that function's own docstring): there is no
    verified single-objective primitive to call directly for that case the way
    `_expected_value_no_decision` is one for the no-decision case.
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

    One hard-evidence value at a time, everything else held at `base_evidence` -- the "single-node
    evidence sweep" the
    plan's own checklist names for both `/sensitivity` (as-is, one node at a time, reported
    directly) and `/voi` (the same per-state values, reduced to a single weighted scalar by the
    node's own marginal -- see `value_of_information`). One `setEvidence`/`makeInference`/read
    cycle per state of `sweep_node_name`, `Sigma|states|` total across a caller's full sweep of
    every candidate node, exactly the checklist's own wording.

    Mutates `engine`'s evidence state as a side effect (every state in turn); callers hold
    `entry.lock` for the whole sweep and are responsible for restoring `engine`'s evidence back to
    `base_evidence` afterward, since a cached engine is shared across later calls on the same
    cache key.

    `states`: sweep exactly these states instead of `sweep_node_name`'s full label list -- lets a
    caller (`/sensitivity`) exclude a state it already knows is ~impossible under `base_evidence`
    *before* spending a `setEvidence`/`makeInference` call on it, rather than discarding the result
    afterward. `None` (default, and always for `/voi`'s own call): sweep every state, unchanged.
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
"""A swept state with its own marginal probability below this, under the base evidence alone, is
excluded from `/sensitivity`'s results entirely -- pyAgrum doesn't raise on near-impossible
evidence, so hard-pinning such a state would otherwise produce a numerically meaningless value
that still ends up in the tornado chart."""

_MAX_SENSITIVITY_SWEEP_NODES = 50
"""Hard cap on how many chance nodes one `/sensitivity` request may sweep (after `nodes`
filtering, if given). Each swept node costs `Sigma|states|` `setEvidence`/`makeInference` calls on
a synchronous endpoint -- unconditionally sweeping every chance node in a large network (no `nodes`
filter given at all) could otherwise run an unbounded number of inference calls per request, the
same worker-blocking concern `_SERVER_MAX_CLUSTER_SIZE` already guards against for `/compile`.
Picked generously for a tornado chart, which is normally read a handful of bars at a time anyway."""


class SensitivityRequest(BaseModel):
    """Body for `POST /graphical/sensitivity`."""

    problem_id: int
    session_id: int | None = None
    evidence: dict[str, str] = {}
    target: str
    """An `ObjectiveNode.symbol`, or a chance/decision node's own name (with `target_state` set)."""
    target_state: str | None = None
    """Required when `target` is a chance/decision node rather than an objective symbol -- see
    `_resolve_target_value`."""
    nodes: list[str] | None = None
    """Restrict the sweep to these chance nodes only (still excluding `target`/already-evidenced
    ones). `None` (default): sweep every chance node in the network, as before -- subject to
    `_MAX_SENSITIVITY_SWEEP_NODES`."""


class SensitivityNodeResult(BaseModel):
    """One swept node's tornado-bar entry."""

    node: str
    values: dict[str, float]
    """State label -> `target`'s value with `node` hard-pinned to that state. Omits any state
    whose own marginal probability under the base `evidence` is ~0 (below `1e-6`) -- pyAgrum
    doesn't raise on such near-impossible evidence, so hard-pinning it would otherwise produce a
    numerically meaningless value in the tornado chart."""
    low: float
    high: float
    swing: float
    """`high - low` -- the tornado bar's own magnitude; `results` is sorted descending by this."""


class SensitivityResponse(BaseModel):
    """Response for `POST /graphical/sensitivity`."""

    base_value: float
    """`target`'s own value under `evidence` exactly as given, no sweep -- the tornado's reference
    line."""
    results: list[SensitivityNodeResult]
    """One entry per swept chance node -- every chance node other than `target`/already in
    `evidence`, or, when `SensitivityRequest.nodes` is given, only those (still excluding
    `target`/already-evidenced ones) -- sorted descending by `swing` -- the standard
    tornado-diagram ordering, biggest bar first. A candidate node all of whose own states are
    ~impossible under the base evidence is omitted entirely (see `SensitivityNodeResult.values`)."""


@router.post("/sensitivity")
def sensitivity(
    request: SensitivityRequest,
    user: Annotated[User, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> SensitivityResponse:
    """Single-node hard-evidence sweep, one node at a time, holding everything else at `evidence`.

    Every other chance node (not `target` itself, not already pinned by `evidence` -- re-pinning
    an already-evidenced node to itself is a degenerate single-value "sweep") is swept over its
    own states via `_sweep_node`, on the same warm cached engine `/whatif` uses -- "Sigma|states|
    queries on a warm engine", literally: this endpoint issues exactly that many `setEvidence`/
    `makeInference` calls, none of which rebuild the diagram. Decision nodes are not swept:
    hard-evidence sweeping is a "what if we also observed this" question, restricted to chance
    nodes for the same reason `/whatif`'s own evidence is (see `_validate_chance_evidence`).

    Raises:
        HTTPException: 401 for a guest user (this endpoint can trigger `Sigma|states|` expensive
            inference calls much like `/compile`'s own concern); 404 if the problem doesn't exist,
            isn't visible to `user`, or has no `GraphicalModelMetaData`; 400 if `evidence`
            references a non-chance node, if `target` isn't a chance/decision/objective node of
            this diagram, if `target` is a chance/decision node and `target_state` is
            missing/invalid, if `nodes` references a node that isn't a chance node, or if the
            number of nodes to sweep (after `nodes` filtering, if given) exceeds
            `_MAX_SENSITIVITY_SWEEP_NODES`; 501 if `target` is an objective and the diagram has
            more than one objective and at least one decision (see `_compute_utilities`).
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

    # (a) `nodes` filter: restrict the sweep to these chance nodes only, still excluding `target`/
    # already-evidenced ones. (b) cap: bound how many nodes one request may sweep, applied after
    # the filter -- see `_MAX_SENSITIVITY_SWEEP_NODES`'s own docstring.
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

            # Each candidate's own marginal under `request.evidence` alone (before any sweeping
            # mutates the engine's evidence) -- used below to drop a swept state whose own
            # probability is ~0 given the rest of the evidence: pyAgrum doesn't raise on
            # near-impossible evidence, so hard-pinning such a state produces a numerically
            # meaningless value that would otherwise still end up in the tornado chart.
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
                    # Every one of this node's own states is ~impossible under the base evidence
                    # (shouldn't happen for a normalised CPT, but defensively skip rather than
                    # crash on an empty min/max below).
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
            # `_sweep_node` mutated the cached engine's evidence repeatedly -- restore it to
            # exactly `request.evidence` so a later call sharing this cache entry is left in a
            # consistent state. Best-effort only: every caller downstream of a sweep
            # (`/whatif`/`/sensitivity`/`/voi`) calls `setEvidence` with its *own* full evidence
            # dict before reading anything (`setEvidence` fully replaces the engine's evidence
            # state -- see the module docstring's "pyAgrum API facts"), so no caller can actually
            # observe leftover sweep evidence even if this restore is skipped; it's kept purely so
            # the cached engine isn't left sitting on `request.evidence`'s *last swept value*
            # between calls, not because anything downstream depends on it. Swallowed rather than
            # left to raise: if `request.evidence` itself is what pyAgrum just rejected (the
            # `except` branch above, about to raise a 400), re-running the exact same
            # `setEvidence` here raises again -- a raise inside `finally` replaces the in-flight
            # `HTTPException`, turning a documented 400 into an opaque 500 (confirmed empirically;
            # a review pass caught this).
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
    """An `ObjectiveNode.symbol` -- VoI is only defined relative to a decision's own expected
    utility (see `value_of_information`'s docstring for why a diagram needs at least one
    decision)."""


class VoiResponse(BaseModel):
    """Response for `POST /graphical/voi`."""

    base_value: float
    """`target`'s expected utility under `evidence` alone, `node` unobserved."""
    expected_value_with_observation: float
    """`sum_s P(node=s | evidence) * target_value(evidence, node=s)` -- the expected utility if
    `node` were observed (and then optimally acted on) before deciding."""
    value_of_information: float
    """`expected_value_with_observation - base_value`. Not clamped at `0` here: for a rational
    decision maker with a correctly specified diagram this is `>= 0` up to solver/inference
    floating-point noise (observing more before deciding can never make the achievable expected
    utility worse) -- a small negative value is itself useful evidence of that noise, not
    something to silently hide by clamping."""
    marginal: dict[str, float]
    """`node`'s own current marginal (`P(node=state | evidence)`) -- the weights
    `expected_value_with_observation` was averaged over."""


@router.post("/voi")
def value_of_information(
    request: VoiRequest,
    user: Annotated[User, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> VoiResponse:
    """Value of information for observing `node` before deciding, framed as EU gain (Phase E).

    `VoI(node) = E[MEU | observe node before deciding] - MEU(no observation)` -- the weighted
    (by `node`'s own current marginal) average of `target`'s value under each hard-evidence value
    of `node` in turn, minus `target`'s value under `evidence` alone. Implemented as a thin
    transform over `_sweep_node` -- the identical "set evidence, run inference, read a scalar"
    primitive `/sensitivity` uses, reduced to one weighted scalar instead of reported per-state;
    this held up cleanly (no divergence from the shared primitive needed), matching the plan's own
    "if it doesn't hold up cleanly, document why and diverge" framing for the case where it does.

    **Only meaningful for a diagram with at least one decision -- rejected with 400 otherwise.**
    "EU gain from observing before deciding" (the plan's own framing) presupposes something being
    decided; a no-decision diagram has nothing whose policy could change in response to the
    observation, so VoI doesn't type-check for it, not merely evaluate to zero. Returning `0.0` in
    that case would be a well-formed-looking number for a question that doesn't apply, silently
    masking the distinction between "meaningfully zero" and "not applicable" -- the explicit
    4xx makes the scope cut visible rather than accidental (matching the plan's own precedent, see
    `DiagramConstraint`'s `kind == "tail"`/multi-objective-with-decision 501, for picking a
    documented rejection over a guess).

    Raises:
        HTTPException: 404 if the problem doesn't exist, isn't visible to `user`, or has no
            `GraphicalModelMetaData`; 400 if the diagram has no decisions, if `target` isn't an
            `ObjectiveNode.symbol`, if `node` isn't a chance node, if `node` is already pinned by
            `evidence`, or if `evidence` itself references a non-chance node; 501 if the diagram
            has more than one objective and at least one decision (see `_compute_utilities`,
            reached via `_resolve_target_value`).
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
            # See `sensitivity`'s identical, more thoroughly commented `finally` block: this
            # restore is best-effort/defensive only (no downstream caller can actually observe
            # leftover sweep evidence), and swallowed rather than left to raise so a bad
            # `request.evidence` doesn't turn an in-flight 400 into an opaque 500.
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


# --- POST /graphical/demo/irrigation, POST /graphical/{problem_id}/solve ------------------------
#
# Phase F of the plan ("Real backend integration for the CUMULUS demo"): the webui's dev-only
# network-structure demo dialog needs a `problem_id` to hand to `<NetworkPanel>`, and a genuinely
# *solved* policy to show in `decisionMode: 'solver'` -- neither existed before this phase (the
# dialog previously bound `<NetworkView>` to a static TypeScript fixture with a hardcoded demo
# policy constant). Both endpoints below are deliberately minimal and demo-scoped, not
# general-purpose "compile any canned network"/"solve any problem" APIs -- see each one's own
# docstring.


_DEMO_IRRIGATION_PROBLEM_NAME = "CUMULUS network demo (irrigation)"
"""`ProblemDB.name` this endpoint always compiles under -- also what `DemoIrrigationRequest`'s own
idempotency check matches on to find an already-compiled copy (see `compile_demo_irrigation_model`'s
own docstring)."""


class DemoIrrigationRequest(BaseModel):
    """Body for `POST /graphical/demo/irrigation`. Empty body (`{}`) is a normal, expected call."""

    force: bool = False
    """`False` (default): reuse an already-compiled demo problem owned by the caller, if one
    exists, instead of compiling a new one. `True`: always compile a fresh copy, regardless of
    what already exists -- see `compile_demo_irrigation_model`'s own docstring for why this needs
    an explicit opt-in rather than being the only behaviour."""


def _demo_irrigation_metadata_matches(problem_id: int, db_session: Session) -> bool:
    """Whether `problem_id` actually carries a `GraphicalModelMetaData` row matching the demo.

    Backs `compile_demo_irrigation_model`'s idempotency lookup: matching on `ProblemDB.name` +
    `user_id` alone would also silently adopt a same-named problem created through the ordinary
    `/problem/add` path (which carries no `GraphicalModelMetaData` at all), or a genuinely
    different diagram someone happened to save under this exact name -- neither is actually "the"
    irrigation demo. Compared against `build_irrigation_demo()`'s own serialized network, the same
    "most precise, fully compiled" signal `/structure` and friends already trust, rather than
    anything looser (e.g. just "has some `GraphicalModelMetaData`").
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

    Rather than have the webui hand-construct a `DiscreteNetwork`/`DecisionAnnotation` JSON payload
    itself (a large, easy-to-get-subtly-wrong duplication of a Python-side model's CPT/utility
    tensors purely for a demo dialog), this endpoint builds the same canned model server-side and
    delegates straight to `compile_graphical_model` -- an ordinary Python function call, not a
    second HTTP round trip -- so persistence stays exactly `/compile`'s own atomic-commit path, not
    a parallel copy of it. The webui calls this lazily, once, the first time the CUMULUS demo
    dialog opens, and caches the resulting `problem_id` client-side (see `CUMULUS/+page.svelte`)
    rather than recompiling on every open.

    **Idempotent by default.** Every call used to persist a brand-new ~127 KB `Problem` +
    metadata into the caller's own problem list unconditionally -- a review pass caught this
    meaning every browser tab, every reload without a live `sessionStorage` cache, and every
    "Recompile demo" click accumulated a fresh, never-cleaned-up
    `"CUMULUS network demo (irrigation)"` entry. Unless `request.force` is set, this now looks for
    a `ProblemDB` already owned by `user` with that exact name, *and* carrying a
    `GraphicalModelMetaData` row whose own network actually matches `build_irrigation_demo()`'s
    (see `_demo_irrigation_metadata_matches`) -- name alone isn't enough: a review pass caught a
    same-named problem created through the ordinary `/problem/add` path (or a different diagram
    saved under this exact name) getting silently adopted as if it were the demo. A name match
    that doesn't also qualify falls through to a fresh compile rather than being adopted. The
    lookup itself only inspects the single most-recently-created row with this name, not the
    most-recent *qualifying* one -- so a higher-id, same-named-but-unrelated row (e.g. from
    `/problem/add`) shadows an older genuine demo copy and forces a redundant fresh compile. This
    self-heals after that one extra compile, though: the fresh copy is itself the new
    highest-id row with this name, so it qualifies on every call after. `force=True` bypasses
    this lookup entirely and always compiles fresh, for the webui's own "Recompile demo" button.

    Raises:
        HTTPException: whatever `compile_graphical_model` itself raises (401 for a guest user; a
            400 here would mean a bug in `build_irrigation_demo`, since its own `validate_against`
            call already checks what `/compile` would otherwise catch on a bad annotation).
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
    """A solved `Problem`'s `SolverResults.optimal_variables` -- directly usable as
    `PolicyRequest.variable_values` (see `POST /graphical/{problem_id}/policy` above) to decode a
    policy from this solve."""


@router.post("/{problem_id}/solve")
def solve_graphical_model(
    problem_id: int,
    request: SolveRequest,
    user: Annotated[User, Depends(get_current_user)],
    db_session: Annotated[Session, Depends(get_session)],
) -> SolveResponse:
    """Solve a compiled diagram's `Problem` for one objective, via `guess_best_solver`.

    Minimal, single-objective solve -- not a general-purpose solve API (this repo's method
    routers, e.g. `cumulus.py`, already own that job for their own problems). Exists so a
    `decisionMode: 'solver'` UI (the CUMULUS demo dialog, in particular) can drive
    `POST /graphical/{problem_id}/policy` off a genuinely solved policy instead of a hardcoded demo
    constant. Calls the general `guess_best_solver`/`find_compatible_solvers` machinery rather than
    hardcoding `GurobipySolver` directly -- Phase 4's own design decision ("targets gurobipy only,
    for now") means `guess_best_solver` resolves to `GurobipySolver` for every diagram
    `compile_diagram` can produce today regardless, but this keeps the endpoint correct unchanged
    if a second backend is added later (see the plan's own Phase 4 rationale).

    Only solves *compiled graphical diagrams* -- like every sibling `/graphical/{problem_id}/*`
    endpoint, `_fetch_graphical_metadata` is called first, so an ordinary (non-graphical)
    `ProblemDB` the user happens to be able to see is rejected with a 404 rather than solved. A
    review pass caught an earlier version skipping that check, which doubled this endpoint as an
    accidental general-purpose solve API for arbitrary visible problems -- with no size/cost bound
    analogous to `/compile`'s own server-side cluster cap, and no defence against a problem whose
    `optimal_variables` doesn't fit `SolveResponse`'s declared shape (e.g. a multi-dimensional
    `TensorVariable`, which a compiled diagram's own `Problem` never has, but an arbitrary ordinary
    problem might).

    Raises:
        HTTPException: 404 if the problem doesn't exist, isn't visible to `user`, or has no
            `GraphicalModelMetaData` (i.e. isn't a compiled graphical diagram); 400 if
            `objective_symbol` isn't one of the problem's own objectives, or if no compatible
            solver exists for it; 500 if constructing/running the solver itself raises, or if it
            reports failure (`SolverResults.success` is `False`) -- surfaced with the solver's own
            message rather than an opaque, unhandled exception escaping as a raw 500.
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

    # A compiled graphical diagram's `z` variables are always binary, so `variable_domain` is
    # `integer` or `mixed` for essentially every real diagram. `find_compatible_solvers`'s own
    # vetted list can still include `CVXPYSolver` in that case purely on convexity grounds
    # (`check_cvxpy_suitability` doesn't check for a MIP-capable backend at all) -- a review pass
    # caught the earlier "fall back to a vetted solver" fix (below) not actually preventing its own
    # target scenario, since `CVXPYSolver` being vetted-but-not-MIP-capable means plain `in solvers`
    # membership isn't enough on a license-less machine with no MIP-capable CVXPY backend
    # installed. Narrow to solvers known to actually handle MILPs before applying that fallback.
    if problem.variable_domain in (VariableDomainTypeEnum.integer, VariableDomainTypeEnum.mixed):
        # `PyomoBonminSolver` (BONMIN) is a real MINLP/branch-and-bound solver that handles binary
        # `z` vars fine -- it's the one `find_compatible_solvers` itself appends specifically
        # *because* `variable_domain` is `integer`/`mixed` (see its own "mixed integer" branch), so
        # leaving it out of this allowlist would reject the one MIP-capable solver actually present
        # on a machine with `bonmin` installed but no Gurobi/cbc.
        mip_capable_solvers = {GurobipySolver, PyomoGurobiSolver, PyomoCBCSolver, PyomoBonminSolver}
        solvers = [solver for solver in solvers if solver in mip_capable_solvers]
        if not solvers:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No compatible MIP-capable solver found for problem {problem_id}.",
            )

    # `guess_best_solver` never returns nothing -- it falls through to `NevergradGenericSolver` --
    # and its own branch order differs from `find_compatible_solvers`'s (a review pass caught this
    # landing on `CVXPYSolver` instead of `GurobipySolver` for a mixed-integer diagram on a machine
    # without a Gurobi license, raising an unhandled `cvxpy` exception when the local CVXPY install
    # lacks a MIP-capable backend). Trust the guess only when it's actually one of the solvers
    # `find_compatible_solvers` itself already vetted (and, above, actually MIP-capable when
    # needed); fall back to its own first choice otherwise.
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
