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

**Scope.** Phase 4's mu/z variables, local consistency, CPT constraints, decision-link
constraints, and one linear `Objective` per `ObjectiveNode`, plus Phase 5's `DiagramConstraint`
compilation (`_build_diagram_constraint`) and Phase 6's `RiskMeasure`/CVaR compilation
(`_build_risk_measure`).

**Phase 6: CVaR (`RiskMeasure`), following Herrala, Terho & Oliveira, "Risk-averse decision
strategies for influence diagrams using rooted junction trees" (Operations Research Letters 61
(2025) 107308, arXiv:2401.03734), Section 3.4 and Table 1 (equations 11-21).** Applied as part of
this same compile pass, using the target `ObjectiveNode`'s own containing cluster's mu -- not as a
`Constraint` appended to an already-finished `Problem` the way a naive reading of "transformation"
might suggest (see the plan's Phase 6 preamble, "applied as transformations... not as constraints
appended afterward"). Concretely: `_build_risk_measure` takes the same `_objective_groups` result
`_build_objective` uses for its target objective (`compile_diagram` computes `_objective_groups`
once per distinct target objective symbol and shares it -- see `objective_groups_by_symbol` --
rather than each caller recomputing an `O(cluster state space)` pass; a real inefficiency the code
review pass caught, see the plan's Findings) but merges the per-state groups by
*equal utility value* instead of summing them -- giving the target objective's own realizable
utility support U and, for each u in U, a `Sum`/`Extract` expression over mu computing its
(unnormalised, since mu is only pinned to total 1 at the root) probability p(u). One eta
(`Variable`, playing the paper's own VaR_alpha role -- see `RiskMeasureIndexMap.eta_symbol`'s own
docstring for why the *solved* value isn't guaranteed to equal the canonical VaR_alpha), and four
length-|U| `TensorVariable`s (lambda, lambda_bar binary; rho, rho_bar in [0,1]) implement equations
(11)-(20) exactly (`M = max(U) - min(U)`, `epsilon = 0.5 *
min pairwise gap between distinct u values` -- both per the paper's own convention), and one
`Objective` per `RiskMeasure` computes `CVaR_alpha = (1/alpha) * sum_u rho_bar(u) * u` (Eq. before
Table 1, page 7).

**A real, confirmed erratum in the published Table 1, not a lossy-fetch artifact.** Table 1's own
description of rho_bar(u) at u = eta reads "alpha - sum_{u in U} p(u)" in the rendered PDF (checked
by direct glyph/font inspection, not the lossy pypdf dump -- the character genuinely is Latin
italic 'p', not a bar-dropped rho). That's dimensionally wrong: sum_{u in U} p(u) = 1 always (p is
a full probability mass function), so alpha - 1 is negative for any alpha < 1, violating rho_bar's
own [0,1] bound (20). The paper's own body text one paragraph later (page 7, "the value of
rho_bar(u) must be alpha - sum_{u in U} rho(u) for u = eta") uses rho(u), not p(u) -- and since
rho(u) = 0 for every u >= eta (established earlier in that same paragraph), summing rho(u) over
all of U is exactly the tail sum sum_{u < eta} p(u), matching what equation (18) (sum_u rho_bar(u)
= alpha) actually forces once rho_bar(u) = p(u) for u < eta and 0 for u > eta. This module
implements the algebraically-derived, self-consistent behaviour (rho_bar(u) is only ever
constrained by (11)-(18) directly -- no formula is hardcoded), so the erratum doesn't affect
correctness here; it's recorded for anyone cross-checking against the paper's own Table 1 directly.

**Sign convention for a minimizing target.** The paper's own u is a value/utility to be
maximized throughout -- CVaR_alpha averages the *lower* alpha-tail (VaR_alpha down to the worst
outcome), the risk-averse direction for maximization. DESDEO's `ObjectiveNode.maximize` can be
`False` (a cost to minimize); for that case, `_build_risk_measure` runs the identical paper
formulation over the *negated* utility values, then negates the resulting CVaR expression back
(and reports the compiled `Objective` with `maximize=objective.maximize`, mirroring the target's
own sense rather than always `True`) -- which is exactly the standard Rockafellar-Uryasev
CVaR-of-loss framing (the risk-averse average of the worst, i.e. *highest*, cost outcomes,
minimized), not something the paper itself needed to address since it has no minimizing case.

**`DiagramConstraint.func` node-reference scope, Phase 5.** The only supported way `func` may
reference a node in `DiagramConstraint.nodes` is `["At", "<node-symbol>", k]` -- a single 1-based
own-state index. This module only compiles **linear combinations of single-node marginals**: any
number of such `At` terms may be combined with `Add`/`Subtract`, and multiplied by a plain numeric
constant (`Multiply`, at most one operand node-derived), but combining two or more node-derived
subtrees under `Multiply`/`Divide`-by-a-node-derived-denominator/any other non-linear operator is
rejected at compile time (see `_substitute_at_nodes`) -- a joint-event probability like
`P(A == lo AND D == s0)` is genuinely *not* expressible this way: the natural-looking
`["Multiply", At_A, At_D]` encoding would compute an independence-assuming *product* of two
marginals, a different (and usually wrong) quantity, not the joint probability -- exactly why it's
rejected rather than silently computed. Every reference to a node *anywhere else* in `func` -- a
bare symbol leaf, in operator position, inside `Extract`/`Sum`, as `At`'s `k` argument with a
non-literal-int index, or a node/objective symbol that exists elsewhere in the compiled model but
isn't declared in this constraint's own `nodes` -- raises `ValueError` instead of being silently
misinterpreted or silently passed through unsubstituted (the latter is a real risk if the
reference happens to collide with another compiled symbol, e.g. an `ObjectiveNode.symbol`, since
it would then silently resolve to *that* symbol's own compiled expression instead of erroring).
`kind == decision` substitutes directly against that decision's own `z` (only supported for
decisions with an empty *effective* information set -- see `_build_diagram_constraint`); `kind in
{probability, expected_value}` substitutes against the marginal of that node's own state, read out
of its containing cluster's `mu` (both kinds compile identically -- `kind` only documents the
constraint author's intent, per the plan's taxonomy table); `kind == tail` is rejected outright --
Phase 6 adds `RiskMeasure`/CVaR compilation (a different mechanism, see the module docstring), not
a `DiagramConstraint` `kind == tail` implementation, so this remains unsupported.

**Reserved symbol namespace.** `mu_*`/`z_*`/`sep_*_*`/`eta_*`/`lambda_*`/`lambdabar_*`/`rho_*`/
`rhobar_*` (variables/extra functions) and `cpt_*`/`mc_hi_*`/`mc_lo_*`/`zsum_*`/`lc_*`/`mass_*`/
`cvar_11_*`...`cvar_18_*` (constraints) are this module's own; an `ObjectiveNode`/
`DiagramConstraint`/`RiskMeasure` symbol that happens to collide surfaces as an opaque `Problem`
"non-unique symbols" error, not a message naming these as reserved -- worth knowing if one of
those errors ever needs debugging.
"""

import itertools
from collections.abc import Callable
from functools import lru_cache
from math import prod
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from desdeo.problem.graphical.index_map import mixed_radix_decode, mixed_radix_encode, separator_block_range
from desdeo.problem.graphical.network import DiscreteNetwork, Potential
from desdeo.problem.graphical.rooted_junction_tree import (
    RootedJunctionTree,
    build_rooted_junction_tree,
    effective_information_sets,
)
from desdeo.problem.graphical.schema import (
    DecisionAnnotation,
    DecisionNode,
    DiagramConstraint,
    DiagramConstraintKind,
    ObjectiveNode,
    RiskMeasure,
)
from desdeo.problem.schema import (
    Constraint,
    ConstraintTypeEnum,
    ExtraFunction,
    Objective,
    Problem,
    TensorVariable,
    Variable,
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


class RiskMeasureIndexMap(BaseModel):
    """Bookkeeping for one compiled `RiskMeasure` (Phase 6, CVaR -- see the module docstring).

    `utility_values` and `eta_symbol` are in *internal sign convention*: unchanged from the target
    `ObjectiveNode`'s own utility values when it maximizes, negated when it minimizes (see the
    module docstring's "Sign convention for a minimizing target" paragraph) -- `target_maximizes`
    records which, so a caller reading `eta_symbol`'s solved value back would know whether to
    negate it (`value if target_maximizes else -value`, same rule for `utility_values`) *if*
    `eta_symbol` reliably held a value worth reading back this way -- see its own docstring below
    for why it doesn't, in general.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    """`RiskMeasure.symbol`; also the compiled CVaR `Objective`'s own symbol."""
    target: str
    """`RiskMeasure.target` -- the `ObjectiveNode.symbol` this risk measure was built for."""
    target_maximizes: bool
    """`ObjectiveNode.maximize` of the target objective -- see the class docstring."""
    utility_values: list[float]
    """U's support, ascending, in internal sign convention -- index `i` (0-based) is the u that
    `lambda_symbol`/`lambda_bar_symbol`/`rho_symbol`/`rho_bar_symbol`'s element `i+1` (1-based, per
    `TensorVariable` convention) refers to."""
    eta_symbol: str
    """Symbol of the scalar `Variable` (11)-(18) constrain to *some* alpha-quantile of the
    compiled distribution -- **not necessarily the canonical lower VaR_alpha, and not necessarily
    equal to any element of `utility_values`.** Confirmed by the code review pass (see the plan's
    Findings): whenever the CDF hits `alpha` exactly at a support point, the whole open interval up
    to the next support point is equally feasible for `eta` and the CVaR objective is indifferent
    to which point in it the solver picks -- e.g. `p=[0.5,0.3,0.2]`, `u=[10,20,30]`, `alpha=0.5`
    compiles a correct `CVaR_0.5=10.0`, yet a solved `eta` of `15.0` (between the first two support
    points, not equal to either) also satisfies every constraint. `CVaR_alpha` itself (the
    corresponding compiled `Objective`'s value) has no such ambiguity -- only `eta`'s own solved
    value does. A canonical VaR read-back, if ever needed, should be derived from the solved
    `rho_bar`/mu (the smallest `u` whose cumulative `rho_bar` reaches `alpha`) rather than trusted
    from `eta` directly -- not implemented here, out of this phase's scope."""
    lambda_symbol: str
    lambda_bar_symbol: str
    rho_symbol: str
    rho_bar_symbol: str
    """Symbols of the four length-`len(utility_values)` `TensorVariable`s implementing (11)-(20)."""


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
    risk_measures: dict[str, RiskMeasureIndexMap] = Field(default_factory=dict)
    """Keyed by `RiskMeasure.symbol`. Empty when `annotation.risk_measures` is empty."""


class DecisionPolicy(BaseModel):
    """One decision's full solved policy, read back from a solved `z` via `IndexMap`.

    Phase E of `.claude/desdeo-graphical-models-plan.md`: promoted from what was
    `tests/test_graphical_compile.py`'s own test-private `_read_back_policy` helper (already
    exercised by Phase 4/5's own oracle-readback tests) into real library code -- the plan's own
    checklist names this exact mechanism "load-bearing for reading policies back out of
    solutions". Unlike `routers/graphical.py`'s `/whatif`/`_read_policy` (a *live pyAgrum engine's*
    evidence-conditioned posterior, collapsed to a single choice only when fully pinned), this
    reads a *solved MILP*'s `z` directly and returns the whole policy function, not just whatever
    a particular evidence state happens to collapse to -- the two are complementary read-back
    paths over the same underlying `IndexMap`/`DecisionIndexMap` bookkeeping, not alternatives for
    the same question.

    Index-based, not label-based: `IndexMap` carries no node-label information (`DiscreteNetwork`/
    `DecisionAnnotation` own that) -- translating `information_set`/`states` indices to labels,
    and collapsing to "the choice given this evidence", is the caller's job. See
    `routers/graphical.py`'s `_decode_solved_policy` for the API-layer translation actually used.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    """`DecisionIndexMap.symbol` -- this decision's own symbol."""
    information_set: list[str]
    """`DecisionIndexMap.information_set` -- the *effective* (no_forgetting-expanded) information
    set, most-significant-first, matching `policy`'s own key encoding (`mixed_radix_encode` over
    `information_cardinalities`)."""
    states: list[str]
    """`DecisionIndexMap.states` -- this decision's own alternatives, in state-index order."""
    policy: dict[int, int]
    """Information-state flat index (0-based, `0` -- the only key -- when `information_set` is
    empty) -> the chosen state's own 0-based index into `states`. The *full* policy function, not
    a single collapsed choice: a decision with a non-empty information set genuinely answers
    differently per information state, which is exactly what this preserves."""


def read_policy_from_solution(
    index_map: IndexMap, optimal_variables: dict[str, list[float]]
) -> dict[str, DecisionPolicy]:
    """Read every decision's full solved policy out of a solved `Problem`'s `optimal_variables`.

    For each decision, `optimal_variables[decision_map.z_symbol]` (the same flat `z` array
    `SolverResults.optimal_variables` reports for any `TensorVariable`) is split into
    `n_info_states` contiguous blocks of `len(states)` entries each (`DecisionIndexMap`'s own
    information-set-major/states-minor flattening -- see its class docstring), and each block's
    argmax is read off as that information state's chosen state. A genuinely solved binary `z`
    has exactly one `1.0` per block (Phase 4/5's own `_zsum_*` constraint enforces this), so
    argmax recovers the deterministic choice regardless of exact floating-point equality to `1.0`
    -- no closeness assertion is made here (that belongs to a test oracle checking the *solver*
    behaved correctly, not to this read-back step, which should read whatever `optimal_variables`
    actually contains).

    Args:
        index_map: the `IndexMap` `compile_diagram` returned alongside the `Problem` that was
            solved (or the equivalent persisted `GraphicalModelMetaData.index_map_json`).
        optimal_variables: a solved `Problem`'s `SolverResults.optimal_variables` (or an
            equivalent mapping, e.g. a `UserSavedSolutionDB.variable_values` row) -- must contain
            an entry for every `index_map.decisions[...].z_symbol`.

    Returns:
        dict[str, DecisionPolicy]: keyed by decision symbol, matching `index_map.decisions`.

    Raises:
        KeyError: `optimal_variables` is missing an entry for one of `index_map.decisions`'s
            `z_symbol`s.
        ValueError: a `z_symbol` entry's length doesn't match
            `prod(decision_map.information_cardinalities) * len(decision_map.states)` -- e.g. a
            stale saved solution's `z` array left over from before a recompile changed a
            cardinality. Checked explicitly rather than silently inferring `n_info_states` from
            `len(z_values) // n_states` (a review pass caught the original version doing this,
            which produces a policy dict missing some info-state keys instead of failing loudly).
    """
    policies: dict[str, DecisionPolicy] = {}
    for symbol, decision_map in index_map.decisions.items():
        z_values = optimal_variables[decision_map.z_symbol]
        n_states = len(decision_map.states)
        n_info_states = prod(decision_map.information_cardinalities)
        expected_len = n_info_states * n_states
        if len(z_values) != expected_len:
            msg = (
                f"Decision '{symbol}': z_symbol '{decision_map.z_symbol}' has length "
                f"{len(z_values)}, expected {expected_len} "
                f"(prod(information_cardinalities)={n_info_states} * len(states)={n_states}). "
                "This usually means optimal_variables came from a solution solved against a "
                "different (e.g. since-recompiled) IndexMap."
            )
            raise ValueError(msg)

        policy: dict[int, int] = {}
        for info_flat in range(n_info_states):
            block = z_values[info_flat * n_states : (info_flat + 1) * n_states]
            policy[info_flat] = max(range(n_states), key=lambda s: block[s])

        policies[symbol] = DecisionPolicy(
            symbol=symbol,
            information_set=decision_map.information_set,
            states=decision_map.states,
            policy=policy,
        )
    return policies


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


def _objective_groups(
    objective: ObjectiveNode,
    rjt: RootedJunctionTree,
    cluster_maps: dict[str, ClusterIndexMap],
    cardinality: dict[str, int],
) -> tuple[ClusterIndexMap, np.ndarray, list[list[int]], list[int]]:
    """The containing cluster, utility array, and per-target-state mu flat-index groups.

    Shared by `_build_objective` (which sums these into one linear expected-value `Objective`) and
    `_build_risk_measure` (Phase 6, which instead merges them by equal utility value to get the
    target's realizable utility distribution -- see that function's docstring).

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
    return cluster_map, utility_array, groups, target_cardinalities


def _build_objective(
    objective: ObjectiveNode,
    cluster_map: ClusterIndexMap,
    utility_array: np.ndarray,
    groups: list[list[int]],
    target_cardinalities: list[int],
) -> Objective:
    """One linear `Objective`, evaluated as a sum over its parents' joint states.

    Takes `_objective_groups`' own return values directly rather than calling it itself, so
    `compile_diagram` can compute `_objective_groups` once per distinct target objective and share
    the result with every `RiskMeasure` targeting the same objective (`_build_risk_measure` takes
    the identical shape) -- see `compile_diagram`'s `objective_groups_by_symbol`. `_objective_groups`
    is an `O(cluster state space)` pass; recomputing it once per `RiskMeasure` sharing a target was
    a real, measured inefficiency the code review caught (see the plan's Phase 6 review-pass
    Findings).
    """
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


def _eta_symbol(symbol: str) -> str:
    return f"eta_{symbol}"


def _lambda_symbol(symbol: str) -> str:
    return f"lambda_{symbol}"


def _lambda_bar_symbol(symbol: str) -> str:
    return f"lambdabar_{symbol}"


def _rho_symbol(symbol: str) -> str:
    return f"rho_{symbol}"


def _rho_bar_symbol(symbol: str) -> str:
    return f"rhobar_{symbol}"


def _risk_measure_internal_symbols(risk_measures: list[RiskMeasure]) -> set[str]:
    """Every eta/lambda/lambda_bar/rho/rho_bar symbol `_build_risk_measure` would emit for `risk_measures`.

    Computed directly from each `RiskMeasure.symbol` (the naming is deterministic, no RJT/cluster
    context needed) so `compile_diagram` can pass this to `_build_diagram_constraint` *before*
    `_build_risk_measure` itself runs. Used to keep `DiagramConstraint.func` from silently reaching
    into these reserved internal MILP variables (see the module docstring's reserved-namespace
    note and `_build_diagram_constraint`'s own docstring) -- a gap the code review pass flagged
    alongside the `objective_like_symbols` one.
    """
    symbols: set[str] = set()
    for risk_measure in risk_measures:
        symbols |= {
            _eta_symbol(risk_measure.symbol),
            _lambda_symbol(risk_measure.symbol),
            _lambda_bar_symbol(risk_measure.symbol),
            _rho_symbol(risk_measure.symbol),
            _rho_bar_symbol(risk_measure.symbol),
        }
    return symbols


_MIN_DISTINCT_UTILITY_VALUES_FOR_GAP = 2
"""Fewer than two distinct utility values means no pair to compute a gap between -- see
`_cvar_epsilon`'s degenerate-case fallback."""


def _cvar_epsilon(distinct_us: list[float]) -> float:
    """0.5 * the minimum gap between adjacent distinct utility values.

    `distinct_us` is sorted ascending -- this is the paper's own convention (Section 3.4, citing
    Salo et al. [22]) for choosing the smallest epsilon that still models (11)-(14)'s strict
    inequalities correctly.

    A single distinct utility value has no pair to compare, but also needs no real separation --
    `eta` is then forced to equal that one value by (11)/(12) alone (`M = 0` in that case, since
    `max(U) == min(U)`), for *any* positive epsilon. `1.0` is used as an arbitrary positive
    placeholder in that degenerate case.
    """
    if len(distinct_us) < _MIN_DISTINCT_UTILITY_VALUES_FOR_GAP:
        return 1.0
    return 0.5 * min(b - a for a, b in itertools.pairwise(distinct_us))


def _build_risk_measure(
    risk_measure: RiskMeasure,
    objective: ObjectiveNode,
    cluster_map: ClusterIndexMap,
    utility_array: np.ndarray,
    groups: list[list[int]],
    target_cardinalities: list[int],
) -> tuple[list[Variable | TensorVariable], list[Constraint], Objective, RiskMeasureIndexMap]:
    """One `RiskMeasure`'s CVaR reformulation.

    Builds eta/lambda/lambda_bar/rho/rho_bar plus (11)-(20), and one `Objective` computing
    `CVaR_alpha = (1/alpha) * sum_u rho_bar(u) * u` -- see the module docstring for the paper
    citation, the Table 1 erratum this deliberately does *not* reproduce, and the sign convention
    used when `objective.maximize` is `False`.

    `objective` is the target `ObjectiveNode` (`annotation.objectives[i]` where
    `.symbol == risk_measure.target` -- resolved by the caller, `compile_diagram`). Takes
    `_objective_groups`' own return values directly (`cluster_map`/`utility_array`/`groups`/
    `target_cardinalities`) rather than calling it itself -- see `_build_objective`'s docstring for
    why (shared, memoised once per target objective by `compile_diagram`, not recomputed per
    `RiskMeasure`). U -- the target's realizable utility support -- is obtained by merging
    `groups` by equal utility value (unlike `_build_objective`, zero-utility states are *not*
    skipped here: they're a real point of U's support with their own probability mass, not a term
    that drops out of a sum).
    """
    sign = 1.0 if objective.maximize else -1.0

    by_value: dict[float, list[int]] = {}
    for target_flat, group in enumerate(groups):
        if not group:
            continue
        state = mixed_radix_decode(target_flat, target_cardinalities)
        u = sign * float(utility_array[state])
        by_value.setdefault(u, []).extend(group)

    us = sorted(by_value)
    n = len(us)
    p_exprs = [_extract_expr(cluster_map.mu_symbol, sorted(by_value[u])) for u in us]

    alpha = risk_measure.alpha
    symbol = risk_measure.symbol
    big_m = us[-1] - us[0]
    epsilon = _cvar_epsilon(us)

    eta_symbol = _eta_symbol(symbol)
    lambda_symbol = _lambda_symbol(symbol)
    lambda_bar_symbol = _lambda_bar_symbol(symbol)
    rho_symbol = _rho_symbol(symbol)
    rho_bar_symbol = _rho_bar_symbol(symbol)

    variables: list[Variable | TensorVariable] = [
        Variable(name=f"VaR_alpha[{symbol}]", symbol=eta_symbol, variable_type=VariableTypeEnum.real),
        TensorVariable(
            name=f"lambda[{symbol}]", symbol=lambda_symbol, variable_type=VariableTypeEnum.binary, shape=[n]
        ),
        TensorVariable(
            name=f"lambda_bar[{symbol}]",
            symbol=lambda_bar_symbol,
            variable_type=VariableTypeEnum.binary,
            shape=[n],
        ),
        TensorVariable(
            name=f"rho[{symbol}]",
            symbol=rho_symbol,
            variable_type=VariableTypeEnum.real,
            shape=[n],
            lowerbounds=0.0,
            upperbounds=1.0,
        ),
        TensorVariable(
            name=f"rho_bar[{symbol}]",
            symbol=rho_bar_symbol,
            variable_type=VariableTypeEnum.real,
            shape=[n],
            lowerbounds=0.0,
            upperbounds=1.0,
        ),
    ]

    def _linear(name: str, cons_symbol: str, func: list) -> Constraint:
        return Constraint(
            name=name,
            symbol=cons_symbol,
            cons_type=ConstraintTypeEnum.LTE,
            func=func,
            is_linear=True,
            is_convex=True,
            is_twice_differentiable=True,
        )

    constraints: list[Constraint] = []
    for i, u in enumerate(us):
        lam = ["At", lambda_symbol, i + 1]
        lam_bar = ["At", lambda_bar_symbol, i + 1]
        rho = ["At", rho_symbol, i + 1]
        rho_bar = ["At", rho_bar_symbol, i + 1]
        p = p_exprs[i]
        eta_minus_u = ["Subtract", eta_symbol, u]

        # (11) eta - u <= M * lambda(u)
        constraints.append(
            _linear(
                f"CVaR VaR upper link (11) for {symbol} at u={u}",
                f"cvar_11_{symbol}_{i}",
                ["Subtract", eta_minus_u, ["Multiply", big_m, lam]],
            )
        )
        # (12) eta - u >= (M+eps)*lambda(u) - M
        constraints.append(
            _linear(
                f"CVaR VaR lower link (12) for {symbol} at u={u}",
                f"cvar_12_{symbol}_{i}",
                ["Subtract", ["Subtract", ["Multiply", big_m + epsilon, lam], big_m], eta_minus_u],
            )
        )
        # (13) eta - u <= (M+eps)*lambda_bar(u) - eps
        constraints.append(
            _linear(
                f"CVaR VaR upper link (13) for {symbol} at u={u}",
                f"cvar_13_{symbol}_{i}",
                ["Subtract", eta_minus_u, ["Subtract", ["Multiply", big_m + epsilon, lam_bar], epsilon]],
            )
        )
        # (14) eta - u >= M*(lambda_bar(u) - 1)
        constraints.append(
            _linear(
                f"CVaR VaR lower link (14) for {symbol} at u={u}",
                f"cvar_14_{symbol}_{i}",
                ["Subtract", ["Multiply", big_m, ["Subtract", lam_bar, 1]], eta_minus_u],
            )
        )
        # (15) rho_bar(u) <= lambda_bar(u)
        constraints.append(
            _linear(
                f"CVaR rho_bar <= lambda_bar (15) for {symbol} at u={u}",
                f"cvar_15_{symbol}_{i}",
                ["Subtract", rho_bar, lam_bar],
            )
        )
        # (16) p(u) - (1 - lambda(u)) <= rho(u) <= lambda(u)
        constraints.append(
            _linear(
                f"CVaR rho lower bound (16) for {symbol} at u={u}",
                f"cvar_16lo_{symbol}_{i}",
                ["Subtract", ["Subtract", p, ["Subtract", 1, lam]], rho],
            )
        )
        constraints.append(
            _linear(
                f"CVaR rho upper bound (16) for {symbol} at u={u}",
                f"cvar_16hi_{symbol}_{i}",
                ["Subtract", rho, lam],
            )
        )
        # (17) rho(u) <= rho_bar(u) <= p(u)
        constraints.append(
            _linear(
                f"CVaR rho <= rho_bar (17) for {symbol} at u={u}",
                f"cvar_17lo_{symbol}_{i}",
                ["Subtract", rho, rho_bar],
            )
        )
        constraints.append(
            _linear(
                f"CVaR rho_bar <= p (17) for {symbol} at u={u}",
                f"cvar_17hi_{symbol}_{i}",
                ["Subtract", rho_bar, p],
            )
        )

    # (18) sum_u rho_bar(u) = alpha
    constraints.append(
        Constraint(
            name=f"CVaR tail mass total (18) for {symbol}",
            symbol=f"cvar_18_{symbol}",
            cons_type=ConstraintTypeEnum.EQ,
            func=["Subtract", ["Sum", rho_bar_symbol], alpha],
            is_linear=True,
            is_convex=True,
            is_twice_differentiable=True,
        )
    )

    cvar_terms = [["Multiply", u, ["At", rho_bar_symbol, i + 1]] for i, u in enumerate(us)]
    cvar_expr: list = ["Multiply", 1.0 / alpha, ["Add", *cvar_terms]]
    if not objective.maximize:
        # See the module docstring's "Sign convention for a minimizing target": `cvar_expr` here
        # is CVaR of the *negated* utility, i.e. -1 times the risk-averse average of the worst
        # (highest) cost outcomes; negate back so the compiled Objective reports that quantity
        # directly, in the target's own (cost) units, to be minimized.
        cvar_expr = ["Multiply", -1.0, cvar_expr]

    risk_objective = Objective(
        name=risk_measure.symbol,
        symbol=risk_measure.symbol,
        func=cvar_expr,
        maximize=objective.maximize,
        is_linear=True,
        is_convex=True,
        is_twice_differentiable=True,
    )

    index_map = RiskMeasureIndexMap(
        symbol=symbol,
        target=objective.symbol,
        target_maximizes=objective.maximize,
        utility_values=us,
        eta_symbol=eta_symbol,
        lambda_symbol=lambda_symbol,
        lambda_bar_symbol=lambda_bar_symbol,
        rho_symbol=rho_symbol,
        rho_bar_symbol=rho_bar_symbol,
    )

    return variables, constraints, risk_objective, index_map


_AT_NODE_EXPR_LENGTH = 3
"""Length of a matchable `["At", "<node>", k]` sub-list -- see `_substitute_at_nodes`."""

_ALWAYS_LINEAR_OPERATORS = {"Add", "Subtract"}
"""Operators that stay linear no matter how many node-derived operands they combine."""


def _check_node_derived_linear(operator: str, derived_flags: list[bool], constraint_symbol: str) -> None:
    """Reject a non-linear combination of node-derived subtrees under `operator`.

    A "node-derived" subtree is one containing at least one substituted `At` reference (a
    marginal/z value, per `_substitute_at_nodes`) -- called once per compiled operator node that
    has at least one node-derived operand. `Add`/`Subtract` combine any number of them linearly
    (the entire point of the `At`-only design -- see the module docstring). `Multiply` stays
    linear only with at most one node-derived operand (the other side a plain constant, e.g.
    "weight * probability" for an `expected_value` constraint); two or more node-derived operands
    would multiply marginals together -- an independence-assuming product, not a linear
    combination, and not the same quantity as a joint-event probability either. `Divide` stays
    linear only when the denominator (every operand but the first, matching this module's/the
    parser's left-to-right `reduce` semantics) isn't node-derived. Any other operator applied to a
    node-derived operand (`Power`, etc.) is rejected outright.
    """
    if operator in _ALWAYS_LINEAR_OPERATORS:
        return

    if operator == "Multiply":
        if sum(derived_flags) <= 1:
            return
        msg = (
            f"DiagramConstraint '{constraint_symbol}': func multiplies two or more node-derived "
            "terms together (e.g. two At-substituted marginals). This computes an "
            "independence-assuming product, not a linear combination -- and isn't the same "
            "quantity as a joint-event probability either. Only 'plain numeric constant * a "
            "single node-derived term' is supported; see the module docstring's "
            "single-node-marginals-only scope."
        )
        raise ValueError(msg)

    if operator == "Divide":
        if not any(derived_flags[1:]):
            return
        msg = (
            f"DiagramConstraint '{constraint_symbol}': func divides by a node-derived "
            "denominator (e.g. an At-substituted marginal) -- only dividing by a plain numeric "
            "constant is a linear form; see the module docstring's single-node-marginals-only "
            "scope."
        )
        raise ValueError(msg)

    msg = (
        f"DiagramConstraint '{constraint_symbol}': func applies operator '{operator}' to a "
        "node-derived term (e.g. an At-substituted marginal) -- only Add/Subtract (any "
        "combination) and Multiply/Divide by a plain numeric constant are supported linear "
        "forms; see the module docstring's single-node-marginals-only scope."
    )
    raise ValueError(msg)


def _substitute_at_nodes(
    func: list | str | int | float,
    nodes: set[str],
    make_expr: Callable[[str, int], list],
    constraint_symbol: str,
    known_symbols: set[str],
) -> tuple[list | str | int | float, bool]:
    """Walk `func`, replacing every `["At", "<node>", k]` sub-list for `node` in `nodes`.

    `make_expr(node, k)` builds the substituted MathJSON expression for one match (differs
    between `kind == decision` and `kind in {probability, expected_value}` -- see
    `_build_diagram_constraint`). `known_symbols` is every chance/decision/objective symbol that
    exists anywhere in the compiled model (a superset of `nodes`) -- any occurrence of one of
    these *outside* `nodes`, in any form, raises `ValueError` as "referenced but not declared in
    nodes" (a node this constraint didn't declare, previously silently passed through
    unsubstituted -- and, worse, silently resolving to another compiled symbol's own expression
    at solve time if the name happened to collide with one); any occurrence of a `nodes` member
    outside `["At", node, k]` -- a bare symbol leaf, in operator position, inside `Extract`/`Sum`,
    or as `At`'s `k` argument with a non-literal-int index -- raises `ValueError` as "unsupported
    form". Returns `(substituted, is_node_derived)`: `is_node_derived` is true iff the returned
    subtree contains at least one substituted reference, used by the caller (via
    `_check_node_derived_linear`) to reject non-linear combinations of node-derived subtrees --
    this is the module docstring's `At`-only, linear-combinations-only scope restriction, enforced
    here rather than assumed.
    """
    if isinstance(func, list) and len(func) == _AT_NODE_EXPR_LENGTH and func[0] == "At" and isinstance(func[1], str):
        node = func[1]
        if node in known_symbols:
            if node not in nodes:
                msg = (
                    f"DiagramConstraint '{constraint_symbol}': func references node '{node}' via "
                    f'["At", "{node}", ...], but "{node}" is not declared in this constraint\'s '
                    f"nodes {sorted(nodes)}. Add it to nodes, or remove the reference."
                )
                raise ValueError(msg)
            k = func[2]
            if not isinstance(k, int) or isinstance(k, bool):
                msg = (
                    f"DiagramConstraint '{constraint_symbol}': the state index in "
                    f'["At", "{node}", {k!r}] must be a literal 1-based integer, not {type(k).__name__}.'
                )
                raise ValueError(msg)
            return make_expr(node, k), True

    if isinstance(func, list):
        operator = func[0]
        substituted: list = [operator]
        derived_flags: list[bool] = []
        for item in func[1:]:
            sub_item, is_derived = _substitute_at_nodes(item, nodes, make_expr, constraint_symbol, known_symbols)
            substituted.append(sub_item)
            derived_flags.append(is_derived)

        is_node_derived = any(derived_flags)
        if is_node_derived:
            _check_node_derived_linear(operator, derived_flags, constraint_symbol)

        return substituted, is_node_derived

    if isinstance(func, str) and func in known_symbols:
        if func in nodes:
            msg = (
                f"DiagramConstraint '{constraint_symbol}': func references node '{func}' in an "
                'unsupported form. Only ["At", "<node>", k] (a single 1-based own-state index) is '
                "supported for referencing a node's state -- combine several such At terms with "
                "Add/Multiply/Subtract for anything more complex."
            )
        else:
            msg = (
                f"DiagramConstraint '{constraint_symbol}': func references '{func}', which is not "
                f"declared in this constraint's nodes {sorted(nodes)}. Add it to nodes, or remove "
                "the reference."
            )
        raise ValueError(msg)

    return func, False


def _build_diagram_constraint(
    diagram_constraint: DiagramConstraint,
    rjt: RootedJunctionTree,
    cluster_maps: dict[str, ClusterIndexMap],
    decision_maps: dict[str, DecisionIndexMap],
    decision_effective_info_sets: dict[str, set[str]],
    cardinality: dict[str, int],
    objective_like_symbols: set[str],
    reserved_risk_measure_symbols: set[str] = frozenset(),
) -> Constraint:
    """One `Constraint`, compiled per the Phase 5 taxonomy table (see the module docstring).

    `kind == tail` is rejected -- Phase 6 adds `RiskMeasure`/CVaR compilation
    (`_build_risk_measure`), a different mechanism than a `DiagramConstraint`, and doesn't give
    `kind == tail` a compilation either. `kind == decision` is only supported for decisions whose
    *effective* information set
    (`effective_information_sets`) is empty: `z_<node>` is then just `own_cardinality` entries
    (no information-state axis), so `["At", node, k]` substitutes directly to `["At", z_node, k]`
    with no marginalisation needed -- an info-conditional decision constraint would need
    per-information-state replication, out of scope here. `kind in {probability, expected_value}`
    substitutes the marginal of `node`'s own state k, read out of its containing cluster's `mu`,
    built the same way `_build_objective` evaluates an objective (`find_containing_cluster` +
    `_group_flat_indices_by_projection` + `_extract_expr`) -- both kinds compile identically,
    `kind` only records the constraint author's intent. The hardcoded `is_linear=True` etc. on the
    returned `Constraint` are safe to hardcode *because* `_substitute_at_nodes` (via
    `_check_node_derived_linear`) rejects anything that isn't actually a linear combination of
    single-node marginals before this returns -- not merely assumed from the taxonomy table.

    `objective_like_symbols` is every `ObjectiveNode.symbol` *and* every `RiskMeasure.symbol` --
    both compile to an ordinary `Objective` in the emitted `Problem`, owning no RJT cluster or z,
    so both must be rejected identically by the check just below, and both feed `known_symbols`
    (below) so a bare/`At` reference to either outside `nodes` is rejected too, not silently passed
    through. `reserved_risk_measure_symbols` (every risk measure's own internal eta/lambda/
    lambda_bar/rho/rho_bar variable symbols -- see `_risk_measure_internal_symbols`) is folded into
    `known_symbols` only, not this rejection: these aren't objective-like at all (a `DiagramConstraint`
    was never meant to reference them under any kind, "tail" included), so they get the generic
    "not declared in this constraint's nodes" rejection via `_substitute_at_nodes` rather than this
    objective-specific one. Defaults to `frozenset()` for callers with no risk measures at all.
    """
    symbol = diagram_constraint.symbol
    nodes = set(diagram_constraint.nodes)

    if diagram_constraint.kind == DiagramConstraintKind.tail:
        msg = (
            f"DiagramConstraint '{symbol}' has kind 'tail' -- Phase 6 adds `RiskMeasure`/CVaR "
            "compilation, a different mechanism than a DiagramConstraint, and doesn't give "
            "kind 'tail' a compilation either. Not supported by this compiler."
        )
        raise ValueError(msg)

    referenced_objectives = nodes & objective_like_symbols
    if referenced_objectives:
        msg = (
            f"DiagramConstraint '{symbol}' (kind '{diagram_constraint.kind}') references "
            f"objective or risk-measure symbol(s) {sorted(referenced_objectives)} -- neither owns "
            "an RJT cluster or z, and can only legally be referenced by a 'tail' constraint."
        )
        raise ValueError(msg)

    if diagram_constraint.kind == DiagramConstraintKind.decision:

        def make_expr(node: str, k: int) -> list:
            if node not in decision_maps:
                msg = f"DiagramConstraint '{symbol}' (kind 'decision'): '{node}' isn't a DecisionNode."
                raise ValueError(msg)
            if decision_effective_info_sets[node]:
                msg = (
                    f"DiagramConstraint '{symbol}' (kind 'decision'): decision '{node}' has a "
                    f"non-empty effective information set {sorted(decision_effective_info_sets[node])} "
                    "-- info-conditional decision constraints are out of scope for this phase "
                    "(they would need per-information-state replication)."
                )
                raise ValueError(msg)

            own_cardinality = cardinality[node]
            if not 1 <= k <= own_cardinality:
                msg = (
                    f"DiagramConstraint '{symbol}': state index {k} for decision '{node}' is out "
                    f"of range (decision has {own_cardinality} states, expected 1..{own_cardinality}; "
                    "only positive 1-based indices are supported here, not DESDEO's general "
                    "end-relative negative indexing)."
                )
                raise ValueError(msg)

            return ["At", decision_maps[node].z_symbol, k]

    else:

        def make_expr(node: str, k: int) -> list:
            if node not in cardinality:
                # Defensive: `_substitute_at_nodes` only ever calls `make_expr` for a `node` in
                # `nodes`, and `nodes` is checked against `known_symbols` (below) before
                # `make_expr` can be reached at all -- combined with the `referenced_objectives`
                # rejection above (an objective/risk-measure symbol can't be in `nodes` here),
                # every reachable `node` is guaranteed to be in `cardinality`. Should be
                # unreachable.
                msg = f"DiagramConstraint '{symbol}': '{node}' isn't a chance or decision node."
                raise ValueError(msg)

            own_cardinality = cardinality[node]
            if not 1 <= k <= own_cardinality:
                msg = (
                    f"DiagramConstraint '{symbol}': state index {k} for node '{node}' is out of "
                    f"range (node has {own_cardinality} states, expected 1..{own_cardinality}; only "
                    "positive 1-based indices are supported here, not DESDEO's general "
                    "end-relative negative indexing)."
                )
                raise ValueError(msg)

            containing = rjt.find_containing_cluster([node])
            if containing is None:
                # Defensive: every node owns a cluster containing at least itself, by RJT
                # construction (see build_rooted_junction_tree) -- should be unreachable.
                msg = f"DiagramConstraint '{symbol}': no cluster contains node '{node}'."
                raise ValueError(msg)

            cluster_map = cluster_maps[containing]
            groups = _group_flat_indices_by_projection(
                cluster_map.nodes, cluster_map.cardinalities, [node], [own_cardinality]
            )
            return _extract_expr(cluster_map.mu_symbol, groups[k - 1])

    known_symbols = set(cardinality) | objective_like_symbols | reserved_risk_measure_symbols
    func, _ = _substitute_at_nodes(diagram_constraint.func, nodes, make_expr, symbol, known_symbols)

    return Constraint(
        name=diagram_constraint.name or diagram_constraint.symbol,
        symbol=symbol,
        cons_type=diagram_constraint.cons_type,
        func=func,
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
            reuse a tree across recompiles. No RJT-level transform is needed for `RiskMeasure`
            (Phase 6): `build_rooted_junction_tree`'s `_extend_family_for_clique` already folds
            every `ObjectiveNode`'s parents -- including a `RiskMeasure`'s own target -- into some
            cluster's family unconditionally (see `_build_risk_measure`'s docstring), so the same
            tree that compiles a diagram's expected-value objectives already contains what CVaR
            compilation needs. Checked for the expected node set and for family containment (every
            chance node's CPT parents, every decision's effective information set, and every cluster's
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
            set or family containment; (defensively -- should not happen for a
            `RootedJunctionTree` built by `build_rooted_junction_tree`) an `ObjectiveNode`'s
            parents aren't contained in any cluster; or a `DiagramConstraint` is out of Phase 5's
            scope (`kind == "tail"`, references an objective symbol, references a decision with a
            non-empty effective information set under `kind == "decision"`, references a `nodes`
            member in an unsupported MathJSON form, references a node/objective symbol that isn't
            declared in this constraint's own `nodes`, combines node-derived subtrees under a
            non-linear operator, or an `At` state index is out of range -- see
            `_build_diagram_constraint`/`_substitute_at_nodes`).
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
    variables: list[Variable | TensorVariable] = []
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

    # Computed once per distinct target objective symbol and shared below by both `_build_objective`
    # and every `RiskMeasure` targeting that objective -- `_objective_groups` is an
    # `O(cluster state space)` pass, and recomputing it once per `RiskMeasure` sharing a target
    # (on top of `_build_objective`'s own call for the same objective) was a real, measured
    # inefficiency the code review pass caught; see the plan's Phase 6 review-pass Findings.
    objective_groups_by_symbol = {
        objective.symbol: _objective_groups(objective, rjt, cluster_maps, cardinality)
        for objective in annotation.objectives
    }
    objectives = [
        _build_objective(objective, *objective_groups_by_symbol[objective.symbol])
        for objective in annotation.objectives
    ]

    objective_symbols = {objective.symbol for objective in annotation.objectives}
    risk_measure_symbols = {risk_measure.symbol for risk_measure in annotation.risk_measures}
    # Every `RiskMeasure.symbol` compiles to a real `Objective` in the emitted `Problem`, exactly
    # like an `ObjectiveNode.symbol` -- both must be treated identically by `_build_diagram_
    # constraint`'s "an objective/risk-measure isn't a chance/decision node" rejection and its
    # `known_symbols` undeclared-reference guard, or a `DiagramConstraint` can silently reach in
    # and bind against a risk measure's own compiled CVaR value (a working back-door around the
    # deliberate `kind == "tail"` rejection) -- a real hole the code review pass caught and this
    # closes; see the plan's Findings.
    objective_like_symbols = objective_symbols | risk_measure_symbols
    reserved_risk_measure_symbols = _risk_measure_internal_symbols(annotation.risk_measures)
    constraints.extend(
        _build_diagram_constraint(
            diagram_constraint,
            rjt,
            cluster_maps,
            decision_maps,
            decision_effective_info_sets,
            cardinality,
            objective_like_symbols,
            reserved_risk_measure_symbols,
        )
        for diagram_constraint in annotation.constraints
    )

    objective_by_symbol = {objective.symbol: objective for objective in annotation.objectives}
    risk_index_maps: dict[str, RiskMeasureIndexMap] = {}
    for risk_measure in annotation.risk_measures:
        # `risk_measure.target` is guaranteed to name a real objective by
        # `DecisionAnnotation._check_internal_structure` (runs on every construction, since the
        # model is frozen) -- no defensive check needed here.
        target_objective = objective_by_symbol[risk_measure.target]
        risk_variables, risk_constraints, risk_objective, risk_index_map = _build_risk_measure(
            risk_measure, target_objective, *objective_groups_by_symbol[risk_measure.target]
        )
        variables.extend(risk_variables)
        constraints.extend(risk_constraints)
        objectives.append(risk_objective)
        risk_index_maps[risk_measure.symbol] = risk_index_map

    problem = Problem(
        name=network.name or "compiled_diagram",
        description="Compiled from a DiscreteNetwork/DecisionAnnotation via the RJT MILP formulation.",
        variables=variables,
        objectives=objectives,
        constraints=constraints,
        extra_funcs=extra_funcs,
    )

    index_map = IndexMap(clusters=cluster_maps, decisions=decision_maps, root=rjt.root, risk_measures=risk_index_maps)

    return problem, index_map
