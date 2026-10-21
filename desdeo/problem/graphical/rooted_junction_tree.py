r"""`RootedJunctionTree` construction from a `(DiscreteNetwork, DecisionAnnotation)` pair.

Implements Algorithm 1 ("Create a Minimal Gradual RJT Given a Topological Order") of Parmentier,
Cohen, Leclère, Obozinski & Salmon, "Integer Programming on the Junction Tree Polytope for
Influence Diagrams" (arXiv:1902.07039). A gradual RJT has exactly one cluster per node, satisfies
running intersection, and every node's information set/CPT parents lie in its own cluster.

**Algorithm 1.** Fix a topological order ⪯ on N and process nodes from ⪯-last to ⪯-first:

    C'_v <- fa(v) union (union of (C'_w \ {w}) over every w with an already-recorded arc (v, w))
    if C'_v \ {v} is non-empty:
        u <- argmax_prec (C'_v \ {v})
        record arc (u, v)

where `fa(v) = {v} union parents(v)`. The union only draws from nodes `w` that picked `v` as their
`u` during their own turn (`_pending_children`), not every node whose family mentions `v`. The
root is the ⪯-first node, since every other member of a cluster is ⪯-earlier than its owner.

**Objectives and constraints** have no cardinality, so they don't own clusters. Instead, an
objective's parents (or a constraint's nodes) are folded as a clique into the family of their
⪯-latest member (`_extend_family_for_clique`), so `find_containing_cluster` always succeeds.

**Limitation.** A disconnected diagram would give a forest; this raises `ValueError` instead.
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

    `nodes` is unordered; the mixed-radix ordering lives in `IndexMap`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ownership: str
    """The chance-node name or decision symbol this cluster is built for."""
    nodes: list[str]
    """Every node whose joint state this cluster's moment vector mu_C would be indexed over.
    Always a superset of `{ownership}` and of `ownership`'s own information set/CPT parents."""
    parent: str | None
    """`ownership` of this cluster's tree-parent (the neighbour closer to the root), or `None`
    for the root cluster."""
    separator: list[str]
    """`nodes` shared with the parent cluster; `[]` for the root cluster."""
    order: int
    """BFS visitation order from the root: `0` at the root, increasing away from it. Not the same
    as ⪯: an earlier decision can end up deeper than a later one."""
    state_space_size: int
    """Product of the cardinalities of `nodes`, i.e. the length of mu_C."""


class RootedJunctionTree(BaseModel):
    """A gradual rooted junction tree over one diagram's chance and decision nodes.

    Build via `build_rooted_junction_tree`; the constructor does no structural validation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    clusters: dict[str, Cluster]
    """Keyed by `Cluster.ownership`."""
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

        "Smallest" by `state_space_size`, ties broken by `order`.
        """
        needed = set(nodes)
        candidates = [cluster for cluster in self.clusters.values() if needed <= set(cluster.nodes)]
        if not candidates:
            return None
        return min(candidates, key=lambda cluster: (cluster.state_space_size, cluster.order)).ownership


def effective_information_sets(annotation: DecisionAnnotation) -> dict[str, set[str]]:
    """Each decision's information set, expanded per `no_forgetting` if that's set.

    With `no_forgetting` and `decision_order`, each decision also observes every earlier decision
    and its (effective) information set. Otherwise the declared `information_set` is used as-is.
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

    An objective can't own a cluster, so a constraint referencing it needs a clique over the
    objective's parents instead.
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
    """`kahn_topological_order`, breaking ties by declaration order for reproducible output."""
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
    """Hash of everything that determines RJT *structure* -- never CPT/utility values."""
    chance_names = {variable.name for variable in network.variables}
    payload = {
        "variables": [[variable.name, len(variable.labels)] for variable in network.variables],
        "arcs": sorted(f"{parent}->{child}" for parent, child in network.arcs),
        # Decision parents of chance nodes (and their cardinalities) change the family too.
        "decision_parents": sorted(
            f"{parent}:{potential.shape[axis]}->{node_name}"
            for node_name, potential in network.potentials.items()
            for axis, parent in enumerate(potential.variables[1:], start=1)
            if parent not in chance_names
        ),
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

    `parent_of[v] = u` means arc `(u, v)` was recorded during `v`'s own turn.
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

    Reachability is re-checked so an unreached node fails with a clear `ValueError`.
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
        annotation: the decision/objective overlay; validated against `network` first.
        max_cluster_size: if given, the largest cluster's flattened state-space size is checked
            against it, also on a cache hit.
        on_exceed: `"warn"` (default, via `ClusterSizeWarning`) or `"raise"` (`ValueError`) when
            `max_cluster_size` is exceeded.
        use_cache: look up/store the result in the module-level cache, keyed on
            `_structure_hash`. A cache hit skips `validate_against`, since the hash covers
            everything it checks.

    Returns:
        RootedJunctionTree: the constructed tree.

    Raises:
        ValueError: `annotation` doesn't validate against `network`; the induced chance/decision
            dependency graph has a cycle; the graph is disconnected; or `max_cluster_size` is
            exceeded and `on_exceed == "raise"`.
    """
    cache_key = _structure_hash(network, annotation)
    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            _check_max_cluster_size(cached, max_cluster_size, on_exceed)
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

    _check_max_cluster_size(rjt, max_cluster_size, on_exceed)

    if use_cache:
        _cache_put(cache_key, rjt)

    return rjt


def _check_max_cluster_size(
    rjt: RootedJunctionTree, max_cluster_size: int | None, on_exceed: Literal["warn", "raise"]
) -> None:
    """Warn or raise if `rjt.max_cluster_size` exceeds `max_cluster_size`."""
    if max_cluster_size is None or rjt.max_cluster_size <= max_cluster_size:
        return
    msg = (
        f"Rooted junction tree's largest cluster has state space {rjt.max_cluster_size}, "
        f"exceeding max_cluster_size={max_cluster_size} (owned by "
        f"'{max(rjt.clusters.values(), key=lambda c: c.state_space_size).ownership}')."
    )
    if on_exceed == "raise":
        raise ValueError(msg)
    warnings.warn(msg, ClusterSizeWarning, stacklevel=3)
