r"""MILP compiler: `(DiscreteNetwork, DecisionAnnotation)` -> `Problem`.

Phase 4 of `.claude/desdeo-graphical-models-plan.md`. Compiles a gradual `RootedJunctionTree`
(Phase 3) into an ordinary, solver-agnostic `Problem` implementing the MILP reformulation of
Parmentier, Cohen, Leclère, Obozinski & Salmon (arXiv:1902.07039), Section 4. Targets gurobipy
only for now (see the plan's design decisions) -- everything emitted is a plain `Problem`, so
other backends can be added later through `guess_best_solver`/`find_compatible_solvers` without
redoing this module.

**No separate "mu_check{C}" variable.** The paper introduces a derived moment mu_check{Cv} :=
sum_{xv} mu_{Cv} (the marginal of a cluster's own moment vector with its owning node summed out)
and uses it in two places: the CPT/decision-link equation (mu_{Cv} = mu_check{Cv} * p_{v|pa(v)}),
and the local-consistency equation tying mu_check{Cv} to the matching marginal computed from the
tree-parent's own cluster. Both uses only ever read mu_check{Cv} through a linear expression --
never as an independent decision variable in its own right -- so this module never materialises
it as a `TensorVariable`. Instead, one `ExtraFunction` per (cluster, separator state) names the
block-sum expression once (see `_sep_symbol`), and every equation that would reference
mu_check{Cv}[j] references that extra function's symbol instead. This is exactly the same
`Sum(Extract(...))` mechanism Phase 0 validated, just wrapped in a name so it isn't
re-derived (and re-serialised) at every one of the `state_space_size` CPT/link equations that
need it.

**Gradual property collapses "the separator" to one thing.** Because the RJT is gradual
(`offspring(Cv) = {v}`, Phase 3's `Cluster.separator == Cluster.nodes - {ownership}` always), a
cluster's *own* upward marginalisation (summing out just its own node) is always the cheap,
contiguous case from Phase 0 -- the owning node is placed as the single least-significant digit,
so every separator state's block is `separator_block_range`. What is *not* always contiguous is
matching that block sum against the marginalisation of the tree-*parent's* cluster onto the same
node set: the parent orders itself for *its own* upward link (toward the grandparent), not for
any one child's separator, so for a parent with more than one child at most one child's separator
can align with that ordering -- "accept strided access for the others" (Phase 0). This module
does not try to special-case the aligned child; `_group_flat_indices_by_projection` handles both
uniformly, and `_extract_expr` opportunistically emits a compact `Tuple` range instead of an
explicit index list whenever the resulting group happens to be contiguous.

**Root normalisation.** Every non-root cluster's separator-marginal is pinned by the
local-consistency equation against its parent. Nothing pins the *root's* total probability mass,
since CPT/decision-link equations are homogeneous in mu_{Cv} (uniformly rescaling every mu ANd
every mu_check by the same constant still satisfies them) -- exactly the extra degree of freedom
the MDP LP formulation's `mu^0_{s0} = 1` constraint exists to remove (Section 1.2 of the paper).
One explicit equality constraint (`Sum(mu_root) == 1`) is added for the root cluster to fix it.

**McCormick bound.** `b = 1` (the simplest admissible bound in Eq. 16 of the paper) is used for
the decision-link linearisation. The paper notes this makes the LP relaxation as loose as
possible (Remark 4) -- tightening it, and the Section 5 valid cuts, are optimisation-quality
concerns for a later phase, not correctness ones; the MILP (with `z` integrality) is exact
regardless of which valid `b` is used.

**Scope.** Only what Phase 4's checklist lists: mu/z variables, local consistency, CPT
constraints, decision-link constraints, and one linear `Objective` per `ObjectiveNode`.
`DiagramConstraint`/`RiskMeasure` compilation is Phase 5/6's job, not this module's -- see the
plan.

**Reserved symbol namespace.** `mu_*`/`z_*`/`sep_*_*` (variables/extra functions) and
`cpt_*`/`mc_hi_*`/`mc_lo_*`/`zsum_*`/`lc_*`/`mass_*` (constraints) are this module's own; an
`ObjectiveNode`/`DiagramConstraint` symbol that happens to collide surfaces as an opaque
`Problem` "non-unique symbols" error, not a message naming these as reserved -- worth knowing if
one of those errors ever needs debugging.
"""

from functools import lru_cache
from math import prod
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from desdeo.problem.graphical.index_map import mixed_radix_decode, mixed_radix_encode, separator_block_range
from desdeo.problem.graphical.network import DiscreteNetwork, Potential
from desdeo.problem.graphical.rooted_junction_tree import (
    RootedJunctionTree,
    build_rooted_junction_tree,
    effective_information_sets,
)
from desdeo.problem.graphical.schema import DecisionAnnotation, DecisionNode, ObjectiveNode
from desdeo.problem.schema import (
    Constraint,
    ConstraintTypeEnum,
    ExtraFunction,
    Objective,
    Problem,
    TensorVariable,
    VariableTypeEnum,
)


def _mu_symbol(node: str) -> str:
    return f"mu_{node}"


def _z_symbol(node: str) -> str:
    return f"z_{node}"


def _sep_symbol(node: str, separator_state: int) -> str:
    return f"sep_{node}_{separator_state}"


class ClusterIndexMap(BaseModel):
    """The mixed-radix digit order used to flatten one cluster's moment vector mu_C.

    `nodes`' order is always `sorted(separator) + [ownership]` -- separator nodes (most
    significant, canonically sorted for determinism) followed by the cluster's own owned node
    (least significant, a single digit). This is what makes `separator_symbols` -- this cluster's
    *own* upward marginalisation -- always the cheap, contiguous case (see the module docstring).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ownership: str
    nodes: list[str]
    """Digit order, most significant first. `nodes[-1] == ownership` always."""
    cardinalities: list[int]
    """Matches `nodes` 1:1."""
    mu_symbol: str
    """Symbol of the `TensorVariable` holding mu_C, flattened per `nodes`' order."""
    separator_symbols: list[str]
    """`ExtraFunction` symbol per separator state (`nodes[:-1]`'s joint state, 0-based index into
    this list), each the `Sum(Extract(mu_symbol, ...))` block-sum naming mu_check{C}[state] --
    see the module docstring for why this is an `ExtraFunction` and not its own variable."""

    @property
    def separator_nodes(self) -> list[str]:
        """`nodes[:-1]` -- the separator toward this cluster's tree-parent (`[]` for the root)."""
        return self.nodes[:-1]

    @property
    def separator_cardinalities(self) -> list[int]:
        """`cardinalities[:-1]`, matching `separator_nodes` 1:1."""
        return self.cardinalities[:-1]

    @property
    def state_space_size(self) -> int:
        """`prod(cardinalities)` -- the length of `mu_symbol`'s `TensorVariable`."""
        return prod(self.cardinalities)


class DecisionIndexMap(BaseModel):
    """Bookkeeping for one decision node's z (deterministic-policy indicator) variable.

    `z_symbol`'s `TensorVariable` is flattened as `information_set`-major, `states`-minor (the
    same separator-major-then-own-node convention `ClusterIndexMap` uses, with the decision's own
    state playing the role of the owned node) -- so, symmetrically to `ClusterIndexMap`, summing
    out just the decision's own state per information state is a contiguous block.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    states: list[str]
    """The decision's own alternatives, in state-index order (matches `DecisionNode.states`)."""
    information_set: list[str]
    """The *effective* information set (`rooted_junction_tree.effective_information_sets` --
    `DecisionNode.information_set` expanded per `no_forgetting`, unchanged otherwise), canonically
    sorted -- the node order `z_symbol`'s flattening uses, most significant first."""
    information_cardinalities: list[int]
    """Matches `information_set` 1:1."""
    z_symbol: str
    """Symbol of the binary `TensorVariable`, length `prod(information_cardinalities) * len(states)`."""


class IndexMap(BaseModel):
    """Everything needed to read a compiled solution back into cluster moments / policies.

    Returned alongside the compiled `Problem` by `compile_diagram` for the library path; the
    persisted/API path stores it in `GraphicalModelMetaData` instead (see Phase 3's findings and
    the plan's design decisions) -- either way it is the load-bearing artifact for Phase E.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    clusters: dict[str, ClusterIndexMap]
    """Keyed by `Cluster.ownership`, matching `RootedJunctionTree.clusters`."""
    decisions: dict[str, DecisionIndexMap]
    """Keyed by `DecisionNode.symbol`. Subset of `clusters`' keys (every decision owns exactly
    one cluster, same as every chance node), containing only the decision ones."""
    root: str
    """`RootedJunctionTree.root`, carried through for convenience."""


@lru_cache(maxsize=2048)
def _position_map(order: tuple[str, ...]) -> dict[str, int]:
    """`{node: index in order}`, cached per distinct `order` -- called once per cluster state.

    `_decode_projection` is on the hot path (once per flat index, and flat index counts scale
    with cluster state space -- see `Cluster.state_space_size`); rebuilding this dict from
    scratch on every call was measured to dominate `compile_diagram`'s own runtime at realistic
    cluster sizes. Bounded like the RJT builder's own `_CACHE_MAX_SIZE` (`rooted_junction_tree.py`)
    rather than unbounded, since a long-running process (e.g. the API server) compiling many
    distinct diagrams over its lifetime would otherwise grow this without limit.
    """
    return {node: i for i, node in enumerate(order)}


def _decode_projection(
    flat_index: int, order: list[str], cardinalities: list[int], target_order: list[str]
) -> tuple[int, ...]:
    """Decode `flat_index` under `(order, cardinalities)`, project onto `target_order`.

    `target_order` must be a subset of `order`; the returned tuple gives each `target_order`
    node's own 0-based state, in `target_order`'s order -- regardless of where in `order` it
    originally sat (this is what lets the projection be non-contiguous/"strided").
    """
    position = _position_map(tuple(order))
    full = mixed_radix_decode(flat_index, cardinalities)
    return tuple(full[position[node]] for node in target_order)


def _project_flat(
    flat_index: int,
    order: list[str],
    cardinalities: list[int],
    target_order: list[str],
    target_cardinalities: list[int],
) -> int:
    """`_decode_projection` followed by re-encoding onto `target_order`'s own flat index space."""
    projection = _decode_projection(flat_index, order, cardinalities, target_order)
    return mixed_radix_encode(list(projection), target_cardinalities)


def _group_flat_indices_by_projection(
    order: list[str],
    cardinalities: list[int],
    target_order: list[str],
    target_cardinalities: list[int],
) -> list[list[int]]:
    """Partition every flat index of `(order, cardinalities)` by its projection onto `target_order`.

    Returns a list indexed by the *target's* own flat index, each entry the (ascending, so
    trivially contiguity-checkable) list of *source* 0-based flat indices sharing that
    projection. A single O(source state space) pass, not one lookup per target state -- see the
    module docstring's note on strided access; this is the general marginalisation both local
    consistency (parent cluster -> child separator) and objective evaluation (cluster -> an
    objective's parents) reduce to.
    """
    groups: list[list[int]] = [[] for _ in range(prod(target_cardinalities))]
    for flat in range(prod(cardinalities)):
        target_flat = _project_flat(flat, order, cardinalities, target_order, target_cardinalities)
        groups[target_flat].append(flat)
    return groups


def _extract_expr(symbol: str, flat_indices_0based: list[int]) -> list:
    """`Sum(Extract(symbol, ...))` over `flat_indices_0based`, compact when the group allows it.

    Emits a single `At` for a lone index, a `Tuple` range when the (already-ascending) indices
    happen to be contiguous, and an explicit index list otherwise -- the "accept strided access"
    fallback from the module docstring. Never a dense matrix (Phase 0's finding).
    """
    if len(flat_indices_0based) == 1:
        return ["At", symbol, flat_indices_0based[0] + 1]

    first, last = flat_indices_0based[0], flat_indices_0based[-1]
    if last - first + 1 == len(flat_indices_0based):
        return ["Sum", ["Extract", symbol, ["Tuple", first + 1, last + 1]]]

    return ["Sum", ["Extract", symbol, *[i + 1 for i in flat_indices_0based]]]


def _build_cluster_index_map(
    ownership: str, separator: list[str], cardinality: dict[str, int]
) -> tuple[ClusterIndexMap, list[ExtraFunction]]:
    """One cluster's `ClusterIndexMap` plus the `ExtraFunction`s naming its separator block sums.

    `separator` is `Cluster.separator` (`== Cluster.nodes - {ownership}`, always, by the gradual
    property -- see the module docstring); `nodes`' own order is derived from it, not read off
    `Cluster.nodes` directly, since `Cluster.nodes` carries no ordering of its own.
    """
    nodes = [*sorted(separator), ownership]
    cardinalities = [cardinality[node] for node in nodes]
    own_cardinality = cardinality[ownership]
    separator_state_count = prod(cardinalities[:-1])

    symbols: list[str] = []
    extra_funcs: list[ExtraFunction] = []
    for state in range(separator_state_count):
        start, stop = separator_block_range(state, own_cardinality)
        block_expr = ["Sum", ["Extract", _mu_symbol(ownership), ["Tuple", start, stop]]]
        symbol = _sep_symbol(ownership, state)
        symbols.append(symbol)
        extra_funcs.append(
            ExtraFunction(
                name=f"mu_check[{ownership}] at separator state {state}",
                symbol=symbol,
                func=block_expr,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
        )

    index_map = ClusterIndexMap(
        ownership=ownership,
        nodes=nodes,
        cardinalities=cardinalities,
        mu_symbol=_mu_symbol(ownership),
        separator_symbols=symbols,
    )
    return index_map, extra_funcs


def _build_decision(
    ownership: str,
    decision: DecisionNode,
    index_map: ClusterIndexMap,
    cardinality: dict[str, int],
    info_order: list[str],
) -> tuple[TensorVariable, list[Constraint], DecisionIndexMap]:
    """The z `TensorVariable`, its McCormick/policy `Constraint`s, and its `DecisionIndexMap`.

    `info_order` is the decision's *effective* (no_forgetting-expanded) information set, already
    resolved by the caller (`compile_diagram`) via `effective_information_sets` -- see that
    function's own note on why the raw `DecisionNode.information_set` isn't used directly.
    """
    own_cardinality = index_map.cardinalities[-1]
    info_cardinalities = [cardinality[node] for node in info_order]
    z_target_order = [*info_order, ownership]
    z_target_cardinalities = [*info_cardinalities, own_cardinality]
    z_symbol = _z_symbol(ownership)
    n_info_states = prod(info_cardinalities)

    decision_map = DecisionIndexMap(
        symbol=ownership,
        states=decision.states,
        information_set=info_order,
        information_cardinalities=info_cardinalities,
        z_symbol=z_symbol,
    )
    variable = TensorVariable(
        name=f"z[{ownership}]",
        symbol=z_symbol,
        variable_type=VariableTypeEnum.binary,
        shape=[n_info_states * own_cardinality],
    )

    constraints: list[Constraint] = []
    for i in range(index_map.state_space_size):
        z_idx = _project_flat(i, index_map.nodes, index_map.cardinalities, z_target_order, z_target_cardinalities)
        sep_ref = index_map.separator_symbols[i // own_cardinality]
        mu_i = ["At", index_map.mu_symbol, i + 1]
        z_ref = ["At", z_symbol, z_idx + 1]

        constraints.append(
            Constraint(
                name=f"McCormick upper bound for mu[{ownership}][{i}]",
                symbol=f"mc_hi_{ownership}_{i}",
                cons_type=ConstraintTypeEnum.LTE,
                func=["Subtract", mu_i, z_ref],
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
        )
        constraints.append(
            Constraint(
                name=f"McCormick lower bound for mu[{ownership}][{i}]",
                symbol=f"mc_lo_{ownership}_{i}",
                cons_type=ConstraintTypeEnum.LTE,
                func=["Subtract", ["Add", sep_ref, ["Subtract", z_ref, 1]], mu_i],
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
        )

    for info_state in range(n_info_states):
        start, stop = separator_block_range(info_state, own_cardinality)
        constraints.append(
            Constraint(
                name=f"Deterministic policy at {ownership} for information state {info_state}",
                symbol=f"zsum_{ownership}_{info_state}",
                cons_type=ConstraintTypeEnum.EQ,
                func=["Subtract", ["Sum", ["Extract", z_symbol, ["Tuple", start, stop]]], 1],
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
        )

    return variable, constraints, decision_map


def _build_cpt_constraints(ownership: str, potential: Potential, index_map: ClusterIndexMap) -> list[Constraint]:
    """One equality per cluster state, tying mu_C to mu_check{C} via the node's own CPT."""
    values = np.array(potential.values, dtype=float)
    own_cardinality = index_map.cardinalities[-1]
    constraints: list[Constraint] = []

    for i in range(index_map.state_space_size):
        projection = _decode_projection(i, index_map.nodes, index_map.cardinalities, potential.variables)
        p_i = float(values[projection])
        sep_ref = index_map.separator_symbols[i // own_cardinality]
        mu_i = ["At", index_map.mu_symbol, i + 1]

        constraints.append(
            Constraint(
                name=f"CPT tie for mu[{ownership}][{i}]",
                symbol=f"cpt_{ownership}_{i}",
                cons_type=ConstraintTypeEnum.EQ,
                func=["Subtract", mu_i, ["Multiply", p_i, sep_ref]],
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
        )

    return constraints


def _build_local_consistency_constraints(
    ownership: str, parent_ownership: str, index_map: ClusterIndexMap, parent_map: ClusterIndexMap
) -> list[Constraint]:
    """Tie this cluster's own separator marginal to the matching marginal of its tree-parent."""
    groups = _group_flat_indices_by_projection(
        parent_map.nodes, parent_map.cardinalities, index_map.separator_nodes, index_map.separator_cardinalities
    )
    constraints: list[Constraint] = []
    for state, group in enumerate(groups):
        parent_marginal = _extract_expr(parent_map.mu_symbol, group)
        constraints.append(
            Constraint(
                name=f"Local consistency between {ownership} and its parent {parent_ownership}",
                symbol=f"lc_{ownership}_{state}",
                cons_type=ConstraintTypeEnum.EQ,
                func=["Subtract", index_map.separator_symbols[state], parent_marginal],
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
        )
    return constraints


def _build_objective(
    objective: ObjectiveNode,
    rjt: RootedJunctionTree,
    cluster_maps: dict[str, ClusterIndexMap],
    cardinality: dict[str, int],
) -> Objective:
    """One linear `Objective`, evaluated as a sum over its parents' joint states.

    `objective.parents` is never empty in practice: `ObjectiveNode.utility` (a `Potential`) would
    need `shape=[]`/a 0-d `values` to represent a parent-less (constant) utility, and
    `Potential`'s own shape validator (`network.py`) rejects that -- `np.array(values).shape` for
    any constructible `values` is never `()`. No branch for it here, accordingly.
    """
    utility_array = np.array(objective.utility.values, dtype=float)

    containing = rjt.find_containing_cluster(objective.parents)
    if containing is None:
        # Defensive: build_rooted_junction_tree folds every ObjectiveNode's parents into some
        # cluster's family (see rooted_junction_tree.py's _extend_family_for_clique), so this
        # should be unreachable for a tree it built. Only reachable if a caller's own `rjt`
        # (passed in explicitly) omitted that clique requirement.
        msg = (
            f"No cluster contains objective '{objective.symbol}''s parents "
            f"{objective.parents}. The given rjt was not built for this annotation."
        )
        raise ValueError(msg)

    cluster_map = cluster_maps[containing]
    target_cardinalities = [cardinality[parent] for parent in objective.parents]
    groups = _group_flat_indices_by_projection(
        cluster_map.nodes, cluster_map.cardinalities, objective.parents, target_cardinalities
    )

    terms = []
    for target_flat, group in enumerate(groups):
        state = mixed_radix_decode(target_flat, target_cardinalities)
        utility_value = float(utility_array[state])
        if utility_value == 0.0:
            continue
        terms.append(["Multiply", utility_value, _extract_expr(cluster_map.mu_symbol, group)])

    # A bare `[0.0]` looks like valid MathJSON (a single-literal terminal) but isn't a node any
    # backend's parser matches when it's the *entire* func -- confirmed to raise `ParserError` in
    # the gurobipy/polars/cvxpy parsers alike. An all-zero utility table (every `utility_value`
    # skipped above) must still emit a genuine linear expression, so fall back to an explicit
    # zero-coefficient term against the containing cluster's own mu rather than a bare literal.
    func = ["Add", *terms] if terms else ["Multiply", 0.0, ["At", cluster_map.mu_symbol, 1]]

    return Objective(
        name=objective.name or objective.symbol,
        symbol=objective.symbol,
        func=func,
        maximize=objective.maximize,
        is_linear=True,
        is_convex=True,
        is_twice_differentiable=True,
    )


def _check_cpts_normalized(network: DiscreteNetwork) -> None:
    """Every chance node's CPT must sum to 1 over its own states, for every parent configuration.

    Not enforced by `Potential`/`DiscreteNetwork` themselves (`network.py`'s validator checks
    shape only -- deliberately, since `ObjectiveNode.utility` reuses `Potential`'s schema for
    un-normalised utility tables, and Phases 1-3's own fixtures rely on that being unenforced).
    But the McCormick linearisation's `b = 1` bound (see the module docstring) is only a valid
    upper bound on mu_check *because* every CPT normalises to 1 -- an unnormalised CPT compiles
    without error and fails silently, surfacing only as an opaque solver-reported infeasibility
    with nothing pointing back at the actual cause. Checked here, at the one place that actually
    depends on the assumption, rather than retroactively tightening `DiscreteNetwork`'s own schema.
    """
    for variable in network.variables:
        potential = network.potentials[variable.name]
        values = np.array(potential.values, dtype=float)
        sums = values.sum(axis=0)
        if not np.allclose(sums, 1.0):
            msg = (
                f"Chance node '{variable.name}''s CPT doesn't sum to 1 over its own states for "
                f"every parent configuration (got sums {np.atleast_1d(sums).tolist()}). The MILP "
                "compiler's McCormick linearisation assumes normalised CPTs; an unnormalised one "
                "would compile without error but produce an infeasible or silently wrong model."
            )
            raise ValueError(msg)


def _check_family_contained(required: set[str], cluster_nodes: set[str], ownership: str, label: str) -> None:
    """`required` must lie inside `cluster_nodes`, or `rjt` wasn't built for this annotation.

    Guards `_build_decision`/`_build_cpt_constraints` (whose CPT/information-set projections
    assume every referenced node is in the cluster they're given) and the local-consistency step
    (whose separator must lie inside the tree-parent's own cluster) against a caller-supplied
    `rjt` that's stale relative to the `annotation` actually being compiled -- see
    `compile_diagram`'s `rjt` docstring. Without this check, a mismatch surfaces only as a bare
    `KeyError` deep inside `_decode_projection`, naming neither the cluster nor the cause.
    """
    missing = required - cluster_nodes
    if missing:
        msg = (
            f"{label} for '{ownership}' -- {sorted(missing)} -- isn't contained in its cluster "
            f"{sorted(cluster_nodes)}. The given rjt was not built for this (network, annotation) "
            "pair; omit compile_diagram's rjt= argument to have it built fresh."
        )
        raise ValueError(msg)


def compile_diagram(
    network: DiscreteNetwork,
    annotation: DecisionAnnotation,
    *,
    rjt: RootedJunctionTree | None = None,
    max_cluster_size: int | None = None,
    on_exceed: Literal["warn", "raise"] = "warn",
) -> tuple[Problem, IndexMap]:
    """Compile `(network, annotation)` into a `Problem` implementing the RJT MILP formulation.

    Args:
        network: the Bayesian network (chance nodes + CPTs).
        annotation: the decision/objective overlay; validated against `network` first.
        rjt: a pre-built `RootedJunctionTree` for `(network, annotation)` -- built with
            `build_rooted_junction_tree`'s defaults if omitted. Passing one in lets a caller
            reuse a tree across recompiles (e.g. after `RiskMeasure` transforms, Phase 6).
            Checked for the expected node set and for family containment (every chance node's
            CPT parents, every decision's effective information set, and every cluster's
            separator toward its parent must lie inside the cluster that's supposed to contain
            them) rather than recomputed from scratch, if given -- a stale tree from a materially
            different annotation fails here with a clear message instead of a bare `KeyError` deep
            in constraint construction.
        max_cluster_size: forwarded to `build_rooted_junction_tree` when `rjt` isn't given (has
            no effect on a caller-supplied `rjt`, which is already built). `compile_diagram`'s own
            constraint count is `O(cluster state space)` per cluster (see the module docstring),
            so this is worth setting before compiling an unfamiliar diagram -- the default of
            `None` performs no size check at all, and `state_space_size` in the hundreds of
            thousands starts to matter (see `_position_map`'s docstring on where that cost lands).
        on_exceed: `"warn"` (default) or `"raise"`, forwarded alongside `max_cluster_size`.

    Returns:
        tuple[Problem, IndexMap]: the compiled MILP, and the per-cluster/per-decision bookkeeping
            needed to read a solution back into cluster moments and deterministic policies.

    Raises:
        ValueError: `annotation` doesn't validate against `network`; a chance node's CPT isn't
            normalised; `rjt` was given but doesn't match `(network, annotation)`'s expected node
            set or family containment; or (defensively -- should not happen for a
            `RootedJunctionTree` built by `build_rooted_junction_tree`) an `ObjectiveNode`'s
            parents aren't contained in any cluster.
    """
    annotation.validate_against(network)
    _check_cpts_normalized(network)

    cardinality: dict[str, int] = {variable.name: len(variable.labels) for variable in network.variables}
    cardinality.update({decision.symbol: len(decision.states) for decision in annotation.decisions})
    expected_nodes = set(cardinality)

    if rjt is None:
        rjt = build_rooted_junction_tree(network, annotation, max_cluster_size=max_cluster_size, on_exceed=on_exceed)
    elif set(rjt.clusters) != expected_nodes:
        msg = (
            f"The given rjt's node set {sorted(rjt.clusters)} doesn't match the expected chance/"
            f"decision node set {sorted(expected_nodes)} for this (network, annotation) pair."
        )
        raise ValueError(msg)

    decision_by_symbol = {decision.symbol: decision for decision in annotation.decisions}
    decision_effective_info_sets = effective_information_sets(annotation)

    cluster_maps: dict[str, ClusterIndexMap] = {}
    variables: list[TensorVariable] = []
    extra_funcs: list[ExtraFunction] = []

    for ownership, cluster in rjt.clusters.items():
        index_map, sep_funcs = _build_cluster_index_map(ownership, cluster.separator, cardinality)
        if set(index_map.nodes) != set(cluster.nodes):
            # Defensive: guaranteed by the gradual property (Cluster.separator == nodes -
            # {ownership} always -- see rooted_junction_tree.py's module docstring), which
            # Phase 3's own tests already exercise. A mismatch here would mean that invariant
            # broke, not that this diagram is unusual.
            msg = (
                f"Cluster '{ownership}': separator {cluster.separator} plus ownership doesn't "
                f"reconstruct the cluster's own node set {cluster.nodes}. This indicates the "
                "gradual RJT invariant (Cluster.separator == nodes - {ownership}) was violated."
            )
            raise ValueError(msg)

        cluster_maps[ownership] = index_map
        extra_funcs.extend(sep_funcs)
        variables.append(
            TensorVariable(
                name=f"mu[{ownership}]",
                symbol=index_map.mu_symbol,
                variable_type=VariableTypeEnum.real,
                shape=[index_map.state_space_size],
                lowerbounds=0.0,
                upperbounds=1.0,
            )
        )

    decision_maps: dict[str, DecisionIndexMap] = {}
    constraints: list[Constraint] = []

    for ownership, cluster in rjt.clusters.items():
        index_map = cluster_maps[ownership]

        if ownership in decision_by_symbol:
            info_order = sorted(decision_effective_info_sets[ownership])
            _check_family_contained(
                set(info_order), set(index_map.nodes), ownership, "Decision's effective information set"
            )
            variable, decision_constraints, decision_map = _build_decision(
                ownership, decision_by_symbol[ownership], index_map, cardinality, info_order
            )
            variables.append(variable)
            constraints.extend(decision_constraints)
            decision_maps[ownership] = decision_map
        else:
            potential = network.potentials[ownership]
            _check_family_contained(
                set(potential.variables), set(index_map.nodes), ownership, "Chance node's CPT parents"
            )
            constraints.extend(_build_cpt_constraints(ownership, potential, index_map))

        if cluster.parent is not None:
            parent_map = cluster_maps[cluster.parent]
            _check_family_contained(
                set(index_map.separator_nodes), set(parent_map.nodes), ownership, "Cluster's separator"
            )
            constraints.extend(_build_local_consistency_constraints(ownership, cluster.parent, index_map, parent_map))

    root_map = cluster_maps[rjt.root]
    constraints.append(
        Constraint(
            name=f"Total probability mass at root {rjt.root}",
            symbol=f"mass_{rjt.root}",
            cons_type=ConstraintTypeEnum.EQ,
            func=["Subtract", root_map.separator_symbols[0], 1],
            is_linear=True,
            is_convex=True,
            is_twice_differentiable=True,
        )
    )

    objectives = [_build_objective(objective, rjt, cluster_maps, cardinality) for objective in annotation.objectives]

    problem = Problem(
        name=network.name or "compiled_diagram",
        description="Compiled from a DiscreteNetwork/DecisionAnnotation via the RJT MILP formulation.",
        variables=variables,
        objectives=objectives,
        constraints=constraints,
        extra_funcs=extra_funcs,
    )

    index_map = IndexMap(clusters=cluster_maps, decisions=decision_maps, root=rjt.root)

    return problem, index_map
