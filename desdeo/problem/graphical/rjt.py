r"""`RootedJunctionTree` construction from a `(DiscreteNetwork, DecisionAnnotation)` pair.

Phase 3 of `.claude/desdeo-graphical-models-plan.md`. Implements a *gradual rooted junction
tree* (RJT) in the sense of Parmentier, Cohen, Leclère, Obozinski & Salmon, "Integer Programming
on the Junction Tree Polytope for Influence Diagrams" (arXiv:1902.07039, INFORMS J. on
Optimization 2020) and Herrala, Terho & Oliveira, "Risk-averse decision strategies for influence
diagrams using rooted junction trees" (arXiv:2401.03734, Operations Research Letters 2025) --
the paper Phase 6 (risk measures) targets, and the source of the "gradual" terminology this
module's checklist item uses: a rooted junction tree over node set N is *gradual* if (a) running
intersection holds (any cluster on the tree-path between two clusters contains their
intersection), (b) it has exactly one cluster per node of N -- each cluster the unique "root
cluster" of the single node that owns it -- and (c) a node's information set/CPT-parents are
contained in *its own* cluster (not merely some ancestor's).

**Caveat on fidelity to the cited papers.** Both papers were only reachable here through lossy,
AI-paraphrased extraction (the source PDF renders as compressed binary to the available tools),
and one extracted claim -- "the final/sink node becomes the root" -- turned out to be
irreconcilable with the CPT-chain-rule equation extracted *from the same source*: computing a
child's cluster moment as (parent's moment) x (child's own conditional probability) only makes
sense if the *parent* is closer to the root, so the child's moment can be derived from an
already-available ancestor moment, not the reverse. Rather than trust either paraphrase over the
other, this module fixes its own root/orientation convention from that equation and from first
principles (see "Root and orientation" below), and leans on its own tests -- running
intersection, information-set containment, gradual node/cluster correspondence -- to verify
correctness directly, rather than on reproducing a specific external example. The *technique*
(elimination-order-driven, one-cluster-per-node construction) and the "gradual" property it's
named for are still taken from these papers; the specific rooting direction is this module's own
choice, verified rather than assumed.

**Construction.** Given a topological order ⪯ over N (parents before children, decisions'
information sets before the decisions that need them, and -- when `decision_order` is set --
earlier decisions before later ones), process nodes from ⪯-last to ⪯-first. For node v:
`C_v = fa(v) U (union of (C_w \ {w}) over every G-child w of v)`, where `fa(v) = {v} U parents(v)`
and every G-child w of v has *already* been processed (children are ⪯-later, hence visited
earlier in this reverse walk). If `C_v \ {v}` is non-empty, let `u = argmax_⪯(C_v \ {v})`;
record an (undirected, at this stage) tree edge between u and v.

**Root and orientation.** Edges are collected undirected and only *then* rooted -- deliberately,
because tying "root" to which side of each edge-adding step happened first (the "u"/"v" roles
above) breaks down for a node with two or more real dependents: e.g. `A -> B`, `A -> C` (A is
the CPT-parent of both B and C) makes both B's and C's processing pick `u = A` and try to make A
*their* child, an artefact of the single-edge-per-step bookkeeping, not a real structural
conflict -- A legitimately ends up with two children, B and C, once the edge set is rooted and
oriented by BFS. Root = the node with the smallest ⪯ position (topologically earliest); this
puts CPT-ancestors and earlier `decision_order` entries near the root and their dependents
further out, consistent with the chain-rule direction above. Rooting a valid tree at any node
preserves running intersection (rooting is bookkeeping, not structure), so this choice does not
need to be "the" canonical one -- it only needs to be deterministic and to make `Cluster.order`
(BFS visitation order from the root) increase along `decision_order`, which is what
`tests/test_graphical_rjt.py` checks.

**What counts as N here, and why objectives/constraints aren't in it.** N is chance nodes and
decision nodes only -- nodes with an actual discrete state space, since a cluster's moment
vector mu_C is indexed by the *joint state* of its members (Phase 0), and `ObjectiveNode`/
`DiagramConstraint` have no state space of their own. This is also why decisions can't
parameterise a chance node's CPT in this framework: `DiscreteNetwork.potentials`'s parent
references are validated against `DiscreteNetwork`'s own variable names only (`network.py`'s
`_check_structure`), so chance-node families never reference a decision. Decisions only ever
feed into objectives/constraints, never into CPTs -- an explicit modelling restriction worth
recording here since it's what keeps `family_of` acyclic without further checking on that front.

Parmentier et al.'s own construction handles multiple utility/value nodes by consolidating them
into one synthetic node v̄ with arcs (p, v̄) for every value node's parent p (their Section 3.1) --
forcing a single shared root. DESDEO deliberately keeps objectives *separate* (design decisions
table: "Multiobjective handled by existing methods"), so that trick isn't available as-is, and
adding each `ObjectiveNode` as its own real, separately-owned cluster would need the same v̄-style
consolidation to avoid unrelated objectives conflicting over a shared ancestor. Instead, an
objective's parents (or a constraint's node set) are folded, as a plain clique requirement, into
the family of whichever of their own members is ⪯-latest -- routed into an *existing* real
node's family rather than a synthetic one. Since that anchor node is already ⪯-later than every
other clique member, this can only add already-valid predecessor relationships, never a new one
that could introduce a cycle, and (like the A/B/C example above) any resulting multi-claim on a
shared ancestor is resolved by the same undirected-edges-then-BFS orientation, not a special
case. This guarantees `RootedJunctionTree.find_containing_cluster` succeeds for every
objective/constraint without needing them to own a cluster of their own -- see
`_extend_family_for_clique`.

This module does not import pyAgrum (see the package-level dependency rule in the plan's
cross-cutting section): pyAgrum's `JunctionTreeGenerator` does accept an elimination
partial-order and could seed a triangulation (confirmed interactively -- it produces a correct,
connected join tree given the right partial order), but it triangulates on *node count*, not
joint state-space size, and offers no direct way to fold in objective/constraint cliques or to
get the one-cluster-per-node "gradual" property. Implementing the closed-form algorithm directly
is both simpler and more faithful to the cited papers -- the "otherwise implement directly"
branch of the Phase 3 checklist.

**Limitation.** The induced graph over N (chance/decision nodes, plus any dependency implied by
`decision_order` and by folded-in objective/constraint cliques) must be a single connected
component; a diagram whose chance/decision nodes split into independent pieces raises
`ValueError` rather than returning a forest. Revisit if a real use case needs it -- most diagrams
with actual decisions/objectives to jointly optimise are connected by construction.
"""

import hashlib
import heapq
import json
import warnings
from collections import deque
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict

from desdeo.problem.graphical.annotation import DecisionAnnotation
from desdeo.problem.graphical.network import DiscreteNetwork


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
    away from it (a node is always visited after its tree-parent). See "Root and orientation" in
    the module docstring for why this -- not a global elimination sequence number -- is what's
    guaranteed to move monotonically along `decision_order`."""
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


def _effective_information_sets(annotation: DecisionAnnotation) -> dict[str, set[str]]:
    """Each decision's information set, expanded per `no_forgetting` if that's set.

    Without `no_forgetting` (or without `decision_order`, which it requires), a decision's
    declared `information_set` is used as-is -- LIMID semantics, no assumption beyond what's
    stated. With it, `DecisionAnnotation`'s own docstring specifies the expansion: "each
    decision's information set is treated as also including every earlier decision (per
    decision_order) and its parents" -- applied incrementally along `decision_order` so each
    earlier decision's *own* (already-expanded) effective information set is folded in too, not
    just its declared one.
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


def _topological_order(
    nodes: list[str], predecessors: dict[str, set[str]], declared_index: dict[str, int]
) -> list[str]:
    """Kahn's algorithm, breaking ties by declaration order for reproducible output.

    Reproducibility matters here specifically because the resulting order feeds a cache key
    (`build_rooted_junction_tree`'s cache is keyed on a hash of the *inputs*, but two runs over
    the same inputs must produce the same RJT for the cache to be sound, not just fast).
    """
    remaining = {node: set(predecessors[node]) for node in nodes}
    successors: dict[str, list[str]] = {node: [] for node in nodes}
    for node, preds in predecessors.items():
        for pred in preds:
            successors[pred].append(node)

    ready = [(declared_index[node], node) for node in nodes if not remaining[node]]
    heapq.heapify(ready)
    order: list[str] = []
    while ready:
        _, node = heapq.heappop(ready)
        order.append(node)
        for successor in successors[node]:
            remaining[successor].discard(node)
            if not remaining[successor]:
                heapq.heappush(ready, (declared_index[successor], successor))

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


_CACHE: dict[str, RootedJunctionTree] = {}
"""Keyed on `_structure_hash`. Module-level and process-local -- fine for a library cache; a
persisted/multi-process cache is the API layer's concern (see `GraphicalModelMetaData`)."""


def clear_rjt_cache() -> None:
    """Drop every cached `RootedJunctionTree`. Mainly for tests."""
    _CACHE.clear()


def _build_cluster_contents(
    nodes: list[str], family: dict[str, set[str]], order_index: dict[str, int]
) -> tuple[dict[str, set[str]], set[frozenset[str]]]:
    """The content-propagation half of the construction.

    Returns `cluster_nodes` plus an undirected edge set. Root/orientation is deliberately *not*
    decided here -- see the module docstring.
    """
    children: dict[str, set[str]] = {node: set() for node in nodes}
    for parent_node, fam in family.items():
        for member in fam:
            if member != parent_node:
                children[member].add(parent_node)

    reverse_order = sorted(nodes, key=lambda node: order_index[node], reverse=True)

    cluster_nodes: dict[str, set[str]] = {}
    edges: set[frozenset[str]] = set()
    for v in reverse_order:
        content = set(family[v])
        for w in children[v]:
            content |= cluster_nodes[w] - {w}
        cluster_nodes[v] = content

        rest = content - {v}
        if rest:
            u = max(rest, key=lambda node: order_index[node])
            edges.add(frozenset((u, v)))

    return cluster_nodes, edges


def _root_and_orient(
    nodes: list[str], edges: set[frozenset[str]], order_index: dict[str, int]
) -> tuple[str, dict[str, str | None], dict[str, int]]:
    """Pick a deterministic root and BFS-orient `edges` from it.

    Raises `ValueError` if `edges` (however it was produced) doesn't span `nodes` as a single
    connected, acyclic component -- the general connectivity/cycle check the module docstring's
    "Limitation" paragraph describes, done directly on the edge set rather than assumed.
    """
    adjacency: dict[str, set[str]] = {node: set() for node in nodes}
    for edge in edges:
        a, b = tuple(edge)
        adjacency[a].add(b)
        adjacency[b].add(a)

    root = min(nodes, key=lambda node: order_index[node])

    parent_of: dict[str, str | None] = {root: None}
    bfs_order: dict[str, int] = {root: 0}
    queue: deque[str] = deque([root])
    visited = {root}
    while queue:
        current = queue.popleft()
        for neighbour in sorted(adjacency[current]):
            if neighbour not in visited:
                visited.add(neighbour)
                parent_of[neighbour] = current
                bfs_order[neighbour] = len(visited) - 1
                queue.append(neighbour)

    unreachable = set(nodes) - visited
    if unreachable or len(edges) != len(nodes) - 1:
        msg = (
            f"Rooted junction tree construction did not produce a single connected tree over "
            f"{len(nodes)} node(s): {len(edges)} edge(s) found, {len(unreachable)} node(s) "
            f"unreachable from the root ({sorted(unreachable)}). The diagram's chance/decision "
            "nodes (plus any objective/constraint clique requirements) must form a single "
            "connected component -- see the module docstring's Limitation paragraph."
        )
        raise ValueError(msg)

    return root, parent_of, bfs_order


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
            without touching structure reuses the cached tree instead of re-triangulating.

    Returns:
        RootedJunctionTree: the constructed tree.

    Raises:
        ValueError: `annotation` doesn't validate against `network`; the induced chance/decision
            dependency graph has a cycle; the graph is disconnected (unsupported, see the module
            docstring); or `max_cluster_size` is exceeded and `on_exceed == "raise"`.
    """
    annotation.validate_against(network)

    cache_key = _structure_hash(network, annotation)
    if use_cache and cache_key in _CACHE:
        return _CACHE[cache_key]

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
    effective_information_sets = _effective_information_sets(annotation)
    family.update(
        {
            decision.symbol: {decision.symbol, *effective_information_sets[decision.symbol]}
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

    for objective in annotation.objectives:
        _extend_family_for_clique(family, [p for p in objective.parents if p in node_set], order_index)
    for constraint in annotation.constraints:
        _extend_family_for_clique(family, [n for n in constraint.nodes if n in node_set], order_index)

    cluster_nodes, edges = _build_cluster_contents(nodes, family, order_index)
    root, parent_of, bfs_order = _root_and_orient(nodes, edges, order_index)

    clusters: dict[str, Cluster] = {}
    for v in nodes:
        parent = parent_of[v]
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
        _CACHE[cache_key] = rjt

    return rjt
