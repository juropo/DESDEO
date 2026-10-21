"""Tests for `desdeo.problem.graphical.rooted_junction_tree` (`RootedJunctionTree` construction).

Checked directly against the gradual RJT definition: running intersection, information-set
containment, one cluster per node.
"""

from collections import deque

import pytest
from fixtures import chain_network as _chain_network
from fixtures import fork_network as _fork_network
from fixtures import independent_network as _independent_network

from desdeo.problem.graphical import (
    DecisionAnnotation,
    DecisionNode,
    DiagramConstraint,
    DiagramConstraintKind,
    DiscreteNetwork,
    DiscreteVariable,
    ObjectiveNode,
    Potential,
    RootedJunctionTree,
    build_rooted_junction_tree,
    clear_rjt_cache,
)
from desdeo.problem.graphical.rooted_junction_tree import _CACHE, _CACHE_MAX_SIZE, ClusterSizeWarning, _structure_hash

# ── Helpers ──────────────────────────────────────────────────────────────────────────────────


def _running_intersection_holds(rjt: RootedJunctionTree) -> bool:
    """For every node appearing in >=2 clusters, those clusters form a connected subtree."""
    owners_containing: dict[str, set[str]] = {}
    for cluster in rjt.clusters.values():
        for member in cluster.nodes:
            owners_containing.setdefault(member, set()).add(cluster.ownership)

    for owners in owners_containing.values():
        if len(owners) <= 1:
            continue
        start = next(iter(owners))
        seen = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            current_cluster = rjt.clusters[current]
            neighbours = list(rjt.children_of(current))
            if current_cluster.parent is not None:
                neighbours.append(current_cluster.parent)
            for neighbour in neighbours:
                if neighbour in owners and neighbour not in seen:
                    seen.add(neighbour)
                    queue.append(neighbour)
        if seen != owners:
            return False
    return True


def _is_gradual(rjt: RootedJunctionTree, expected_nodes: set[str]) -> bool:
    """Definition 2.1(a)+(b): running intersection, and exactly one cluster per node."""
    if set(rjt.clusters) != expected_nodes:
        return False
    for symbol, cluster in rjt.clusters.items():
        if cluster.ownership != symbol:
            return False
    return _running_intersection_holds(rjt)


# ── Fixtures ─────────────────────────────────────────────────────────────────────────────────


def _objective(symbol: str, parents: list[str], *, cardinality: int = 2) -> ObjectiveNode:
    shape = [cardinality] * len(parents)
    size = cardinality ** len(parents)
    values = _nest([float(i) for i in range(size)], shape)
    return ObjectiveNode(
        symbol=symbol, parents=parents, utility=Potential(variables=parents, shape=shape, values=values)
    )


def _nest(flat: list[float], shape: list[int]):
    if len(shape) == 1:
        return flat
    step = len(flat) // shape[0]
    return [_nest(flat[i * step : (i + 1) * step], shape[1:]) for i in range(shape[0])]


# ── Gradual property / running intersection ─────────────────────────────────────────────────


def test_chain_network_is_gradual():
    """A -> B -> C with no decisions: three clusters, one per chance node, valid tree."""
    network = _chain_network()
    annotation = DecisionAnnotation(objectives=[_objective("F", ["C"])])
    rjt = build_rooted_junction_tree(network, annotation, use_cache=False)

    assert _is_gradual(rjt, {"A", "B", "C"})
    # Every node's own CPT-parents must be contained in its own cluster (Definition 2.1(c)).
    assert set(network.potentials["B"].variables[1:]) <= set(rjt.clusters["B"].nodes)
    assert set(network.potentials["C"].variables[1:]) <= set(rjt.clusters["C"].nodes)


def test_fork_network_hub_has_two_children():
    """A -> B, A -> C: A must end up owning a cluster with two tree-children, not raise.

    The objective touches only B, to keep the fork structure unperturbed.
    """
    network = _fork_network()
    annotation = DecisionAnnotation(objectives=[_objective("F", ["B"])])
    rjt = build_rooted_junction_tree(network, annotation, use_cache=False)

    assert _is_gradual(rjt, {"A", "B", "C"})
    assert len(rjt.children_of("A")) == 2


def test_independent_chance_nodes_connected_via_shared_objective():
    """A, B share no arc; an objective over both must still connect them in the tree."""
    network = _independent_network("A", "B")
    annotation = DecisionAnnotation(objectives=[_objective("F", ["A", "B"])])
    rjt = build_rooted_junction_tree(network, annotation, use_cache=False)

    assert _is_gradual(rjt, {"A", "B"})
    assert rjt.find_containing_cluster(["A", "B"]) is not None


def test_two_objectives_sharing_one_ancestor_does_not_conflict():
    """A, B, C mutually independent; objectives need {A,B} and {A,C} respectively.

    Both anchors want A, which must resolve to A having two children rather than raising.
    """
    network = _independent_network("A", "B", "C")
    annotation = DecisionAnnotation(objectives=[_objective("F1", ["A", "B"]), _objective("F2", ["A", "C"])])
    rjt = build_rooted_junction_tree(network, annotation, use_cache=False)

    assert _is_gradual(rjt, {"A", "B", "C"})
    assert rjt.find_containing_cluster(["A", "B"]) is not None
    assert rjt.find_containing_cluster(["A", "C"]) is not None


def test_objective_spanning_non_adjacent_chain_nodes_forces_a_bigger_cluster():
    """In A -> B -> C, an objective over {A, C} (skipping B) must still find a cluster."""
    network = _chain_network()
    annotation = DecisionAnnotation(objectives=[_objective("F", ["A", "C"])])
    rjt = build_rooted_junction_tree(network, annotation, use_cache=False)

    assert _is_gradual(rjt, {"A", "B", "C"})
    assert rjt.find_containing_cluster(["A", "C"]) is not None


def test_diagram_constraint_clique_is_satisfiable():
    """A `DiagramConstraint` over two otherwise-unrelated nodes also gets a home cluster."""
    network = _independent_network("A", "B")
    constraint = DiagramConstraint(
        symbol="c1", kind=DiagramConstraintKind.probability, nodes=["A", "B"], func=["Add", "A", "B"]
    )
    annotation = DecisionAnnotation(objectives=[_objective("F", ["A"])], constraints=[constraint])
    rjt = build_rooted_junction_tree(network, annotation, use_cache=False)

    assert rjt.find_containing_cluster(["A", "B"]) is not None


def test_constraint_referencing_objective_node_folds_its_parents():
    """A constraint naming an `ObjectiveNode` symbol must still get a home cluster.

    An objective can't own a cluster, so the clique is over F's own parents.
    """
    network = _independent_network("A", "B")
    constraint = DiagramConstraint(symbol="c1", kind=DiagramConstraintKind.tail, nodes=["F"], func=["F"])
    annotation = DecisionAnnotation(objectives=[_objective("F", ["A", "B"])], constraints=[constraint])
    rjt = build_rooted_junction_tree(network, annotation, use_cache=False)

    assert _is_gradual(rjt, {"A", "B"})
    assert rjt.find_containing_cluster(["A", "B"]) is not None


# ── Decisions: information sets and decision_order ──────────────────────────────────────────


def test_decision_information_set_contained_in_own_cluster():
    """A LIMID-style decision (no decision_order): I(D) must be a subset of C_D."""
    network = _chain_network()
    decision = DecisionNode(symbol="D", states=["a", "b"], information_set=["A", "B"])
    annotation = DecisionAnnotation(decisions=[decision], objectives=[_objective("F", ["C", "D"])])
    rjt = build_rooted_junction_tree(network, annotation, use_cache=False)

    assert _is_gradual(rjt, {"A", "B", "C", "D"})
    assert {"A", "B"} <= set(rjt.clusters["D"].nodes)


def test_decision_order_gives_monotonically_increasing_cluster_order():
    """Two decisions D1, D2 with decision_order=[D1, D2]: D1's cluster is closer to the root."""
    network = _chain_network()
    d1 = DecisionNode(symbol="D1", states=["a", "b"], information_set=["A"])
    d2 = DecisionNode(symbol="D2", states=["a", "b"], information_set=["B"])
    annotation = DecisionAnnotation(
        decisions=[d1, d2],
        objectives=[_objective("F", ["C", "D1", "D2"])],
        decision_order=["D1", "D2"],
    )
    rjt = build_rooted_junction_tree(network, annotation, use_cache=False)

    assert _is_gradual(rjt, {"A", "B", "C", "D1", "D2"})
    assert rjt.clusters["D1"].order < rjt.clusters["D2"].order


def test_no_forgetting_expands_later_decisions_information_set():
    """With no_forgetting, D2 must observe D1 even though its own information_set doesn't say so."""
    network = _chain_network()
    d1 = DecisionNode(symbol="D1", states=["a", "b"], information_set=["A"])
    d2 = DecisionNode(symbol="D2", states=["a", "b"], information_set=["B"])  # doesn't declare D1
    annotation = DecisionAnnotation(
        decisions=[d1, d2],
        objectives=[_objective("F", ["C", "D1", "D2"])],
        decision_order=["D1", "D2"],
        no_forgetting=True,
    )
    rjt = build_rooted_junction_tree(network, annotation, use_cache=False)

    assert "D1" in rjt.clusters["D2"].nodes
    assert rjt.clusters["D1"].order < rjt.clusters["D2"].order


def test_decision_order_contradicting_information_set_raises():
    """decision_order says D1 before D2, but D1's info_set observes D2 -- a genuine cycle."""
    network = _independent_network("A")
    d1 = DecisionNode(symbol="D1", states=["a", "b"], information_set=["D2"])
    d2 = DecisionNode(symbol="D2", states=["a", "b"], information_set=[])
    annotation = DecisionAnnotation(
        decisions=[d1, d2], objectives=[_objective("F", ["A"])], decision_order=["D1", "D2"]
    )

    with pytest.raises(ValueError, match="cycle"):
        build_rooted_junction_tree(network, annotation, use_cache=False)


# ── Errors ───────────────────────────────────────────────────────────────────────────────────


def test_disconnected_diagram_raises():
    """A, B independent, with objectives that never mention them together: no way to connect them."""
    network = _independent_network("A", "B")
    annotation = DecisionAnnotation(objectives=[_objective("FA", ["A"]), _objective("FB", ["B"])])

    with pytest.raises(ValueError, match="connected"):
        build_rooted_junction_tree(network, annotation, use_cache=False)


# ── max_cluster_size ─────────────────────────────────────────────────────────────────────────


def test_max_cluster_size_property():
    """`max_cluster_size` is the largest `state_space_size` across all clusters."""
    network = _fork_network()
    annotation = DecisionAnnotation(objectives=[_objective("F", ["B", "C"])])
    rjt = build_rooted_junction_tree(network, annotation, use_cache=False)

    assert rjt.max_cluster_size == max(cluster.state_space_size for cluster in rjt.clusters.values())
    assert rjt.max_cluster_size >= 4


def test_max_cluster_size_warns_by_default():
    """Exceeding `max_cluster_size` warns (`ClusterSizeWarning`) rather than raising by default."""
    network = _fork_network()
    annotation = DecisionAnnotation(objectives=[_objective("F", ["B", "C"])])

    with pytest.warns(ClusterSizeWarning):
        build_rooted_junction_tree(network, annotation, max_cluster_size=1, use_cache=False)


def test_max_cluster_size_raises_when_requested():
    """`on_exceed="raise"` turns the same threshold breach into a `ValueError`."""
    network = _fork_network()
    annotation = DecisionAnnotation(objectives=[_objective("F", ["B", "C"])])

    with pytest.raises(ValueError, match="max_cluster_size"):
        build_rooted_junction_tree(network, annotation, max_cluster_size=1, on_exceed="raise", use_cache=False)


# ── Caching ──────────────────────────────────────────────────────────────────────────────────


def test_cache_returns_same_object_for_same_structure():
    """Building the same `(network, annotation)` twice returns the identical cached object."""
    clear_rjt_cache()
    network = _chain_network()
    annotation = DecisionAnnotation(objectives=[_objective("F", ["C"])])

    first = build_rooted_junction_tree(network, annotation)
    second = build_rooted_junction_tree(network, annotation)
    assert first is second
    clear_rjt_cache()


def test_cache_ignores_cpt_value_changes():
    """Relearning CPTs (same structure, different potential values) must reuse the cached RJT."""
    clear_rjt_cache()
    network = _chain_network()
    reweighted = DiscreteNetwork(
        variables=network.variables,
        arcs=network.arcs,
        potentials={
            **network.potentials,
            "A": Potential(variables=["A"], shape=[2], values=[0.1, 0.9]),
        },
    )
    annotation = DecisionAnnotation(objectives=[_objective("F", ["C"])])

    first = build_rooted_junction_tree(network, annotation)
    second = build_rooted_junction_tree(reweighted, annotation)
    assert first is second
    clear_rjt_cache()


def test_cache_distinguishes_different_structure():
    """Different network/annotation structures never share a cache entry."""
    clear_rjt_cache()
    annotation = DecisionAnnotation(objectives=[_objective("F", ["C"])])
    chain = build_rooted_junction_tree(_chain_network(), annotation)
    fork_annotation = DecisionAnnotation(objectives=[_objective("F", ["B", "C"])])
    fork = build_rooted_junction_tree(_fork_network(), fork_annotation)

    assert chain is not fork
    clear_rjt_cache()


def test_cache_distinguishes_decision_parent_cardinality():
    """Networks differing only in B's decision-parent axis size must not share a structure hash."""
    clear_rjt_cache()
    a = DiscreteVariable(name="A", labels=["lo", "hi"])
    b = DiscreteVariable(name="B", labels=["lo", "hi"])
    objective = ObjectiveNode(
        symbol="F", parents=["B"], utility=Potential(variables=["B"], shape=[2], values=[0.0, 1.0])
    )
    annotation = DecisionAnnotation(decisions=[DecisionNode(symbol="D", states=["lo", "hi"])], objectives=[objective])

    def _network_with_decision_parent(d_axis_size: int) -> DiscreteNetwork:
        shape = [2, 2, d_axis_size]
        row = [0.1 * (i + 1) for i in range(d_axis_size)]
        return DiscreteNetwork(
            variables=[a, b],
            arcs=[("A", "B")],
            potentials={
                "A": Potential(variables=["A"], shape=[2], values=[0.4, 0.6]),
                "B": Potential(variables=["B", "A", "D"], shape=shape, values=[[row, row], [row, row]]),
            },
        )

    network_ok = _network_with_decision_parent(2)  # D's axis matches annotation's 2-state D.
    network_bad = _network_with_decision_parent(3)  # D's axis doesn't -- invalid against annotation.

    assert _structure_hash(network_ok, annotation) != _structure_hash(network_bad, annotation)

    build_rooted_junction_tree(network_ok, annotation)  # Warms the cache with the valid structure.
    with pytest.raises(ValueError, match="shape"):
        build_rooted_junction_tree(network_bad, annotation)
    clear_rjt_cache()


def test_use_cache_false_bypasses_cache():
    """`use_cache=False` rebuilds (a distinct, though equal, object) every call."""
    clear_rjt_cache()
    network = _chain_network()
    annotation = DecisionAnnotation(objectives=[_objective("F", ["C"])])

    first = build_rooted_junction_tree(network, annotation, use_cache=False)
    second = build_rooted_junction_tree(network, annotation, use_cache=False)
    assert first is not second
    assert first == second
    clear_rjt_cache()


def test_cache_evicts_least_recently_used_past_max_size():
    """Filling the cache past `_CACHE_MAX_SIZE` evicts the least-recently-used entry, not others."""
    clear_rjt_cache()
    try:
        for i in range(_CACHE_MAX_SIZE):
            network = _independent_network(f"A{i}")
            annotation = DecisionAnnotation(objectives=[_objective("F", [f"A{i}"])])
            build_rooted_junction_tree(network, annotation)
        first_key = next(iter(_CACHE))

        # Touch every entry except the first, moving them all ahead of it in LRU order.
        for i in range(1, _CACHE_MAX_SIZE):
            network = _independent_network(f"A{i}")
            annotation = DecisionAnnotation(objectives=[_objective("F", [f"A{i}"])])
            build_rooted_junction_tree(network, annotation)

        overflow_network = _independent_network("Aoverflow")
        overflow_annotation = DecisionAnnotation(objectives=[_objective("F", ["Aoverflow"])])
        build_rooted_junction_tree(overflow_network, overflow_annotation)

        assert len(_CACHE) == _CACHE_MAX_SIZE
        assert first_key not in _CACHE
    finally:
        clear_rjt_cache()
