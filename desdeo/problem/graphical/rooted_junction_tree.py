r"""`RootedJunctionTree` construction from a `(DiscreteNetwork, DecisionAnnotation)` pair.

Phase 3 of `.claude/desdeo-graphical-models-plan.md`. Implements Algorithm 1 ("Create a Minimal
Gradual RJT Given a Topological Order") of Parmentier, Cohen, Leclère, Obozinski & Salmon,
"Integer Programming on the Junction Tree Polytope for Influence Diagrams" (arXiv:1902.07039,
INFORMS J. on Optimization 2020), whose *gradual
RJT* terminology (Definition 2: "an RJT is gradual if, for every v, the offspring of C_v is
exactly {v}") is what Phase 6's target paper -- Herrala, Terho & Oliveira, "Risk-averse decision
strategies for influence diagrams using rooted junction trees" (Operations Research Letters 61
(2025) 107308, arXiv:2401.03734) -- also uses, unchanged. Concretely, a gradual RJT over node
set N has exactly one cluster
per node -- each cluster the unique "root cluster" of the single node that owns it -- running
intersection holds, and every node's information set/CPT-parents lie in *its own* cluster, not
merely some ancestor's.

**Algorithm 1, translated.** Fix a topological order ⪯ on N (parents before children; a decision
before -- rather than after -- what it observes would make ⪯ impossible to construct, which is
exactly the cycle `_topological_order` raises on). Process nodes from ⪯-last to ⪯-first,
maintaining a working cluster `C'_v` and a *node-level* parent map (not yet a tree over clusters):

    C'_v <- fa(v) union (union of (C'_w \ {w}) over every w with an already-recorded arc (v, w))
    if C'_v \ {v} is non-empty:
        u <- the (already-computed) argmax_prec (C'_v \ {v})
        record arc (u, v)

`fa(v) = {v} union parents(v)`. The subtlety the paper's own proof (Proposition 3/4) spells out
and this module's tests exercise directly: the union in the first line only ever draws from
nodes `w` that *themselves*, during their own (earlier) turn, picked `v` as their `u` -- a
dynamic relationship discovered arc-by-arc as the algorithm runs, not "every node whose family
happens to mention v" (a static relationship computable up front). Conflating the two was an
actual bug in an earlier version of this module: it made a common-cause fork (`A -> B`, `A -> C`)
raise a spurious error, because both B's and C's turns independently tried to claim A as *their
own* child using the static notion. Implemented per the paper's dynamic one (`_pending_children`
below, populated only when an arc is actually recorded), the fork case needs no special-casing
at all: A's turn is a real algorithmic step that later absorbs both, once B's and C's arcs have
each been recorded as `(A, B)` and `(A, C)`.

**Why the root is the ⪯-first node, not the last.** The paper's own proof establishes
`(u in C_v) => (u prec v)` (Proposition 3/4's proof, "we first prove that prec is a topological
order on G"): every element of a node's own cluster -- other than the node itself -- is
⪯-earlier than it. Walking from any node up through recorded arcs therefore strictly decreases
⪯, so the tree's root -- the one node whose turn never produces an arc -- is the global ⪯-minimum
(topologically earliest). This also matches the compilation need this module exists for
(Phase 4): a chance node's CPT constraint ties its own moment to its *parents'* (⪯-earlier)
moment, which only reads naturally if the parents' cluster is the one nearer the root.

**What counts as N here, and why objectives/constraints aren't in it.** Parmentier et al.'s own
N is *every* diagram vertex, including utility/value vertices (their `V^r`) -- their MEU
objective sums `<r_v, mu_v>` over each one independently (Eq. 11a), with no special-casing needed
for more than one, since each just gets its own ordinary cluster via `fa(v) = {v} union
parents(v)` like any other vertex. (Herrala et al.'s own "Algorithm 1", confusingly numbered the
same, is a *different*, optional single-value-node preprocessing step, needed for their CVaR
reformulation specifically, not for building a base gradual RJT -- Phase 6's problem, not this
one's.) DESDEO's `ObjectiveNode` doesn't carry a state count the way a value vertex's domain
does -- it's a `Potential` over `parents` directly, with no domain of its own -- so it can't
become a first-class member of N without inventing a cardinality that isn't part of its schema.
Instead, an `ObjectiveNode`'s parents (or a `DiagramConstraint`'s node set) are folded, as a
plain clique requirement, into the family of whichever of their own members is ⪯-latest --
routed into an *existing* real node's family. Since that anchor node is already ⪯-later than
every other clique member, this can only add already-valid predecessor relationships, never a
cycle, and it guarantees `RootedJunctionTree.find_containing_cluster` succeeds for every
objective/constraint without needing them to own a cluster of their own -- see
`_extend_family_for_clique`. This is also why decisions can't parameterise a chance node's CPT in
this framework: `DiscreteNetwork.potentials`'s parent references validate against
`DiscreteNetwork`'s own variable names only (`network.py`'s `_check_structure`), so a chance
node's family never references a decision -- decisions only ever feed objectives/constraints.

This module does not import pyAgrum (see the package-level dependency rule in the plan's
cross-cutting section): pyAgrum's `JunctionTreeGenerator` does accept an elimination
partial-order and could seed a triangulation (confirmed interactively -- it produces a correct,
connected join tree given the right partial order), but it triangulates on *node count*, not
joint state-space size, and offers no direct way to fold in objective/constraint cliques or to
get the one-cluster-per-node "gradual" property. Implementing Algorithm 1 directly ended up both
simpler and exactly what the papers specify -- the "otherwise implement directly" branch of the
Phase 3 checklist.

**Limitation.** Per the paper's own proof ("if G is not connected, the algorithm is equivalent to
its separate application on each of the connected components, which each yield a tree"), a
diagram whose chance/decision nodes (plus any dependency implied by `decision_order` and by
folded-in objective/constraint cliques) split into more than one component would produce a
*forest*, not a single RJT. This module raises `ValueError` in that case instead of returning
one -- most diagrams with actual decisions/objectives to jointly optimise are connected by
construction, and `RootedJunctionTree.root` being a single node is a simplifying assumption worth
keeping until a real use case needs otherwise.
"""

import hashlib
import json
import warnings
from collections import OrderedDict, deque
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict

from desdeo.problem.graphical.network import DiscreteNetwork, kahn_topological_order
from desdeo.problem.graphical.schema import DecisionAnnotation


class ClusterSizeWarning(UserWarning):
    """Raised (as a warning) when a constructed cluster's flattened state space is large.

    See `build_rooted_junction_tree`'s `max_cluster_size`/`on_exceed` parameters.
    """


class Cluster(BaseModel):
    """One cluster of a `RootedJunctionTree`: the "root cluster" of exactly one node.

    See the module docstring for what "root cluster" and "gradual" mean here. `nodes` is in no
    particular order at this stage -- Phase 0's separator-major mixed-radix ordering is recorded
    in `IndexMap`, built once cardinalities and a concrete `Problem` exist (Phase 4), not here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ownership: str
    """The chance-node name or decision symbol this cluster is built for. Exactly one cluster
    owns each node of the diagram's chance+decision node set -- see the module docstring."""
    nodes: list[str]
    """Every node whose joint state this cluster's moment vector mu_C would be indexed over.
    Always a superset of `{ownership}` and of `ownership`'s own information set/CPT parents."""
    parent: str | None
    """`ownership` of this cluster's tree-parent (the neighbour closer to the root), or `None`
    for the root cluster."""
    separator: list[str]
    """`nodes` shared with the parent cluster; `[]` for the root cluster."""
    order: int
    """BFS visitation order from the root: `0` at the root, strictly increasing along any path
    away from it (a node is always visited after its tree-parent). This is tree-depth ordering,
    *not* a restatement of ⪯: a ⪯-earlier decision (e.g. an earlier one per `decision_order`) is
    **not** guaranteed to have a smaller `order` than a ⪯-later one -- BFS order also depends on
    how many hops each node's path to the root takes, which differs once decisions have
    differently sized information sets (an earlier decision with a larger information set can
    end up *deeper* than a later decision with a smaller one). What the paper's proof does
    guarantee (`(u in C_v) => (u prec v)`, see the module docstring) is narrower: a node's own
    tree-parent is always ⪯-earlier than it, so walking from any cluster toward the root strictly
    decreases ⪯ -- just not in lockstep with BFS depth."""
    state_space_size: int
    """Product of the cardinalities of `nodes`: the length mu_C would have under the Phase 0
    mixed-radix flattening. Drives `RootedJunctionTree.max_cluster_size`."""


class RootedJunctionTree(BaseModel):
    """A gradual rooted junction tree over one diagram's chance and decision nodes.

    Build via `build_rooted_junction_tree`, not directly -- the constructor performs no
    structural validation of its own (that all lives in the builder, which is the only place
    with enough context -- cardinalities, information sets, decision order -- to do it).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    clusters: dict[str, Cluster]
    """Keyed by `Cluster.ownership`, matching how the rest of `problem/graphical` addresses
    elements by symbol/name rather than by a separate integer id."""
    root: str
    """`ownership` of the root cluster."""

    @property
    def max_cluster_size(self) -> int:
        """The largest `Cluster.state_space_size` across all clusters; `0` if there are none."""
        return max((cluster.state_space_size for cluster in self.clusters.values()), default=0)

    def children_of(self, ownership: str) -> list[str]:
        """`ownership` symbols of every cluster whose tree-parent is `ownership`."""
        return sorted(cluster.ownership for cluster in self.clusters.values() if cluster.parent == ownership)

    def find_containing_cluster(self, nodes: Iterable[str]) -> str | None:
        """The smallest cluster whose `nodes` is a superset of `nodes`, or `None` if none is.

        "Smallest" by `state_space_size` (ties broken by `order`, for determinism), since a
        smaller qualifying cluster is always at least as good for Phase 4's purposes (evaluating
        an `ObjectiveNode`/`DiagramConstraint` against a cluster's mu vector). Guaranteed to
        find one for every `ObjectiveNode.parents`/`DiagramConstraint.nodes` that
        `build_rooted_junction_tree` was given -- see the module docstring's
        `_extend_family_for_clique` paragraph -- but is a general query, useful for any node set.
        """
        needed = set(nodes)
        candidates = [cluster for cluster in self.clusters.values() if needed <= set(cluster.nodes)]
        if not candidates:
            return None
        return min(candidates, key=lambda cluster: (cluster.state_space_size, cluster.order)).ownership


def effective_information_sets(annotation: DecisionAnnotation) -> dict[str, set[str]]:
    """Each decision's information set, expanded per `no_forgetting` if that's set.

    Without `no_forgetting` (or without `decision_order`, which it requires), a decision's
    declared `information_set` is used as-is -- LIMID semantics, no assumption beyond what's
    stated. With it, `DecisionAnnotation`'s own docstring specifies the expansion: "each
    decision's information set is treated as also including every earlier decision (per
    decision_order) and its parents" -- applied incrementally along `decision_order` so each
    earlier decision's *own* (already-expanded) effective information set is folded in too, not
    just its declared one.

    Public (not module-private) because Phase 4's compiler (`compile.py`) needs the exact same
    expansion to decide what a decision's z (policy indicator) variable is conditioned on --
    conditioning z on only the *declared* `information_set` while the cluster it lives in was
    built from the *effective* one (this function's output) would silently ignore
    `no_forgetting` in the compiled MILP.
    """
    raw = {decision.symbol: set(decision.information_set) for decision in annotation.decisions}
    if not (annotation.decision_order and annotation.no_forgetting):
        return raw

    effective: dict[str, set[str]] = {}
    accumulated: set[str] = set()
    for symbol in annotation.decision_order:
        effective[symbol] = raw[symbol] | accumulated
        accumulated = accumulated | effective[symbol] | {symbol}
    return effective


_MIN_CLIQUE_SIZE = 2
"""A clique of fewer than two nodes needs no folding -- nothing to connect."""


def _extend_family_for_clique(family: dict[str, set[str]], members: list[str], order_index: dict[str, int]) -> None:
    """Fold a required clique into the family of its ⪯-latest member; see the module docstring."""
    member_set = set(members)
    if len(member_set) < _MIN_CLIQUE_SIZE:
        return
    anchor = max(member_set, key=lambda node: order_index[node])
    family[anchor] |= member_set


def _resolve_clique_members(
    members: list[str], node_set: set[str], objective_parents: dict[str, list[str]]
) -> list[str]:
    """`members` filtered to `node_set`, with objective symbols expanded to their own parents.

    Any `ObjectiveNode` symbol among `members` is expanded to *that objective's own parents*
    instead of being dropped. `DiagramConstraint.nodes` may legally reference an objective symbol
    directly (e.g. "a tail
    constraint on F's own distribution" --
    `test_validate_against_constraint_may_reference_objective_node`), but an `ObjectiveNode` has
    no cardinality and can never own a cluster (see the module docstring), so folding it in
    verbatim would silently record no requirement at all. Since an objective's value is a
    function of its own `parents`' joint state, the clique this constraint actually needs is
    over those parents -- so that's what gets folded instead. `ObjectiveNode.parents` can only
    reference chance/decision nodes (never another objective, per its own schema), so this
    expansion never needs to recurse.
    """
    resolved: set[str] = set()
    for member in members:
        if member in node_set:
            resolved.add(member)
        elif member in objective_parents:
            resolved.update(p for p in objective_parents[member] if p in node_set)
    return list(resolved)


def _topological_order(
    nodes: list[str], predecessors: dict[str, set[str]], declared_index: dict[str, int]
) -> list[str]:
    """`kahn_topological_order`, breaking ties by declaration order for reproducible output.

    Reproducibility matters here specifically because the resulting order feeds a cache key
    (`build_rooted_junction_tree`'s cache is keyed on a hash of the *inputs*, but two runs over
    the same inputs must produce the same RJT for the cache to be sound, not just fast).
    """
    order = kahn_topological_order(nodes, predecessors, priority=declared_index)

    if len(order) != len(nodes):
        cyclic = sorted(set(nodes) - set(order))
        msg = (
            "Cannot construct a rooted junction tree: chance-node CPT-parent arcs, decision "
            f"information sets, and decision_order together form a cycle involving {cyclic}. "
            "A common cause is decision_order contradicting an information_set -- a decision "
            "observing a node that decision_order says is only revealed after it."
        )
        raise ValueError(msg)

    return order


def _structure_hash(network: DiscreteNetwork, annotation: DecisionAnnotation) -> str:
    """Hash of everything that determines RJT *structure* -- never CPT/utility values.

    So relearning CPTs (`learn_network(..., structure=...)`, Phase 1) or editing an
    `ObjectiveNode.utility` table without changing `parents` doesn't invalidate the cache.
    """
    payload = {
        "variables": [[variable.name, len(variable.labels)] for variable in network.variables],
        "arcs": sorted(f"{parent}->{child}" for parent, child in network.arcs),
        "decisions": sorted(
            [decision.symbol, len(decision.states), sorted(decision.information_set)]
            for decision in annotation.decisions
        ),
        "objectives": sorted([objective.symbol, sorted(objective.parents)] for objective in annotation.objectives),
        "constraints": sorted(
            [constraint.symbol, constraint.kind.value, sorted(constraint.nodes)]
            for constraint in annotation.constraints
        ),
        "decision_order": annotation.decision_order,
        "no_forgetting": annotation.no_forgetting,
    }
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


_CACHE_MAX_SIZE = 128
"""Cap on `_CACHE`'s size before its least-recently-used entry is evicted. Without a bound, a
long-running process (e.g. the API server) building many distinct network/annotation structures
over its lifetime would grow `_CACHE` -- including each tree's full cluster node sets -- without
limit."""

_CACHE: OrderedDict[str, RootedJunctionTree] = OrderedDict()
"""Keyed on `_structure_hash`, in least-to-most-recently-used order (see `_cache_get`/
`_cache_put`). Module-level and process-local -- fine for a bounded library cache; a
persisted/multi-process cache is the API layer's concern (see `GraphicalModelMetaData`)."""


def clear_rjt_cache() -> None:
    """Drop every cached `RootedJunctionTree`. Mainly for tests."""
    _CACHE.clear()


def _cache_get(key: str) -> RootedJunctionTree | None:
    """`_CACHE[key]` if present (marking it most-recently-used), else `None`."""
    if key not in _CACHE:
        return None
    _CACHE.move_to_end(key)
    return _CACHE[key]


def _cache_put(key: str, value: RootedJunctionTree) -> None:
    """Store `value` under `key`, evicting the least-recently-used entry past `_CACHE_MAX_SIZE`."""
    _CACHE[key] = value
    _CACHE.move_to_end(key)
    if len(_CACHE) > _CACHE_MAX_SIZE:
        _CACHE.popitem(last=False)


def _run_algorithm_1(
    nodes: list[str], family: dict[str, set[str]], order_index: dict[str, int]
) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Algorithm 1 itself: cluster contents plus the node-level parent map, in one reverse pass.

    `parent_of[v] = u` means arc `(u, v)` was recorded during `v`'s own turn -- `u` becomes
    `v`'s tree-parent once cluster-level arcs are formed (Algorithm 1's step 10). Each node
    records at most one arc, during its *own* turn, so `parent_of` can never receive conflicting
    writes -- see the module docstring for why an earlier, non-conforming implementation of this
    same idea (unioning from every node whose family happened to mention `v`, rather than only
    from nodes that actually recorded an arc to `v`) could and did conflict.
    """
    reverse_order = sorted(nodes, key=lambda node: order_index[node], reverse=True)

    pending_children: dict[str, list[str]] = {node: [] for node in nodes}
    cluster_nodes: dict[str, set[str]] = {}
    parent_of: dict[str, str] = {}

    for v in reverse_order:
        content = set(family[v])
        for w in pending_children[v]:
            content |= cluster_nodes[w] - {w}
        cluster_nodes[v] = content

        rest = content - {v}
        if rest:
            u = max(rest, key=lambda node: order_index[node])
            parent_of[v] = u
            pending_children[u].append(v)

    return cluster_nodes, parent_of


def _assign_bfs_order(nodes: list[str], root: str, parent_of: dict[str, str]) -> dict[str, int]:
    """`0` at `root`, strictly increasing outward -- see `Cluster.order`'s docstring."""
    children: dict[str, list[str]] = {node: [] for node in nodes}
    for child, parent in parent_of.items():
        children[parent].append(child)

    bfs_order: dict[str, int] = {root: 0}
    queue: deque[str] = deque([root])
    while queue:
        current = queue.popleft()
        for child in sorted(children[current]):
            bfs_order[child] = len(bfs_order)
            queue.append(child)

    return bfs_order


def _root_and_bfs_order(nodes: list[str], parent_of: dict[str, str]) -> tuple[str, dict[str, int]]:
    """Find the single root in `parent_of` and BFS-order every node from it, or raise.

    `len(roots) == 1` only implies a single connected tree given the invariant `_run_algorithm_1`
    proves (every `parent_of[v]` precedes `v` in ⪯, so `parent_of` can't contain a cycle disjoint
    from the root) -- so the BFS reachability is re-checked directly here too, as
    defense-in-depth: a future violation of that invariant then fails with a clear, diagnostic
    `ValueError` naming the unreached nodes, rather than a bare `KeyError` from a `bfs_order`
    lookup downstream.
    """
    roots = [node for node in nodes if node not in parent_of]
    if len(roots) != 1:
        msg = (
            f"Rooted junction tree construction found {len(roots)} root(s) ({sorted(roots)}) "
            f"instead of one over {len(nodes)} node(s); the diagram's chance/decision nodes "
            "(plus any objective/constraint clique requirements) must form a single connected "
            "component -- see the module docstring's Limitation paragraph."
        )
        raise ValueError(msg)
    root = roots[0]

    bfs_order = _assign_bfs_order(nodes, root, parent_of)
    unreached = set(nodes) - set(bfs_order)
    if unreached:
        msg = (
            f"Rooted junction tree construction produced a single root but "
            f"{len(unreached)} node(s) unreachable from it: {sorted(unreached)}. This points at "
            "a bug in the construction algorithm itself (parent_of should never contain a cycle "
            "disjoint from the root) rather than a malformed diagram."
        )
        raise ValueError(msg)

    return root, bfs_order


def build_rooted_junction_tree(
    network: DiscreteNetwork,
    annotation: DecisionAnnotation,
    *,
    max_cluster_size: int | None = None,
    on_exceed: Literal["warn", "raise"] = "warn",
    use_cache: bool = True,
) -> RootedJunctionTree:
    """Construct the gradual `RootedJunctionTree` for `(network, annotation)`.

    Args:
        network: the Bayesian network (chance nodes + CPTs).
        annotation: the decision/objective overlay; validated against `network` first
            (`DecisionAnnotation.validate_against`), so a mismatched reference fails here with
            that validator's message rather than a confusing downstream `KeyError`.
        max_cluster_size: if given, the resulting tree's `max_cluster_size` (a flattened
            state-space size, not a node count -- see `Cluster.state_space_size`) is checked
            against it.
        on_exceed: `"warn"` (default, via `ClusterSizeWarning`) or `"raise"` (`ValueError`) when
            `max_cluster_size` is exceeded.
        use_cache: look up/store the result in the module-level cache, keyed on
            `_structure_hash` -- CPT/utility values never affect the key, so relearning CPTs
            without touching structure reuses the cached tree instead of re-triangulating. A
            cache hit skips `validate_against` too: the hash already captures everything that
            check reads (variable/decision cardinalities, objective/constraint node references),
            so if this exact structure validated once, it validates again.

    Returns:
        RootedJunctionTree: the constructed tree.

    Raises:
        ValueError: `annotation` doesn't validate against `network`; the induced chance/decision
            dependency graph has a cycle; the graph is disconnected (unsupported, see the module
            docstring); or `max_cluster_size` is exceeded and `on_exceed == "raise"`.
    """
    cache_key = _structure_hash(network, annotation)
    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    annotation.validate_against(network)

    cardinalities: dict[str, int] = {variable.name: len(variable.labels) for variable in network.variables}
    cardinalities.update({decision.symbol: len(decision.states) for decision in annotation.decisions})

    nodes = list(cardinalities)
    if not nodes:
        msg = "Cannot construct a rooted junction tree: network and annotation have no chance or decision nodes."
        raise ValueError(msg)
    node_set = set(nodes)

    family: dict[str, set[str]] = {
        variable.name: {variable.name, *network.potentials[variable.name].variables[1:]}
        for variable in network.variables
    }
    decision_effective_info_sets = effective_information_sets(annotation)
    family.update(
        {
            decision.symbol: {decision.symbol, *decision_effective_info_sets[decision.symbol]}
            for decision in annotation.decisions
        }
    )

    declared_index: dict[str, int] = {variable.name: i for i, variable in enumerate(network.variables)}
    offset = len(network.variables)
    declared_index.update({decision.symbol: offset + i for i, decision in enumerate(annotation.decisions)})

    predecessors: dict[str, set[str]] = {node: set() for node in nodes}
    for child, fam in family.items():
        predecessors[child] |= fam - {child}
    if annotation.decision_order:
        for earlier, later in zip(annotation.decision_order, annotation.decision_order[1:], strict=False):
            predecessors[later].add(earlier)

    order = _topological_order(nodes, predecessors, declared_index)
    order_index = {node: i for i, node in enumerate(order)}

    objective_parents = {objective.symbol: objective.parents for objective in annotation.objectives}
    for objective in annotation.objectives:
        members = _resolve_clique_members(objective.parents, node_set, objective_parents)
        _extend_family_for_clique(family, members, order_index)
    for constraint in annotation.constraints:
        members = _resolve_clique_members(constraint.nodes, node_set, objective_parents)
        _extend_family_for_clique(family, members, order_index)

    cluster_nodes, parent_of = _run_algorithm_1(nodes, family, order_index)
    root, bfs_order = _root_and_bfs_order(nodes, parent_of)

    clusters: dict[str, Cluster] = {}
    for v in nodes:
        parent = parent_of.get(v)
        separator = sorted(cluster_nodes[v] & cluster_nodes[parent]) if parent is not None else []
        state_space_size = 1
        for member in cluster_nodes[v]:
            state_space_size *= cardinalities[member]
        clusters[v] = Cluster(
            ownership=v,
            nodes=sorted(cluster_nodes[v]),
            parent=parent,
            separator=separator,
            order=bfs_order[v],
            state_space_size=state_space_size,
        )

    rjt = RootedJunctionTree(clusters=clusters, root=root)

    if max_cluster_size is not None and rjt.max_cluster_size > max_cluster_size:
        msg = (
            f"Rooted junction tree's largest cluster has state space {rjt.max_cluster_size}, "
            f"exceeding max_cluster_size={max_cluster_size} (owned by "
            f"'{max(clusters.values(), key=lambda c: c.state_space_size).ownership}')."
        )
        if on_exceed == "raise":
            raise ValueError(msg)
        warnings.warn(msg, ClusterSizeWarning, stacklevel=2)

    if use_cache:
        _cache_put(cache_key, rjt)

    return rjt
