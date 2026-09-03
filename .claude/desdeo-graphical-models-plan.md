# Graphical models in DESDEO — implementation plan

Backend: compile Bayesian networks and influence diagrams (learned from data or elicited
from experts) into DESDEO `Problem` instances via the rooted junction tree MILP formulation.

Frontend: a reusable network visualisation component for what-if queries and conveying
dependency structure, embeddable in method-specific UIs.

The two tracks are deliberately decoupled — UI phase A ships before any backend work exists.

---

## Working through this plan

- One phase at a time. Follow the dependency order in "Sequencing" (cross-cutting section) —
  not the top-to-bottom phase numbering, which is exposition order, not execution order.
- Check off checklist items directly in this file as they're completed. This file is the
  progress tracker; don't keep a separate TODO for it.
- Stop and report at the end of each phase rather than cascading into the next unprompted —
  say what got checked off and what's next. This is fragile, exploratory work (see the risk
  table), worth reviewing direction on before compounding.
- If something comes up that this plan doesn't answer, surface it rather than resolving it
  silently.

---

## Design decisions already settled

| Decision | Rationale |
|---|---|
| RJT formulation, not decision programming | Path-based formulation is exponential and empirically worse; the RJT reformulations now also carry risk measures and side constraints |
| Output is a plain `Problem` | Every existing interactive method works on compiler output on day one |
| Not `ScenarioModel` for the base compiler | Its scenario tree is a path/branching structure; encoding the ID that way *is* the path-based formulation |
| Own pydantic schema, mirroring pyAgrum's data model | SWIG objects can't be pydantic fields, DB rows, or JSON for the frontend |
| pyAgrum for import, learning, and inference only | Hard dependency, like gurobipy/pyomo/cvxpy already are — no extras mechanism exists in `pyproject.toml`, so this follows the existing convention rather than inventing `desdeo[graphical]`. Still confined to one adapter module; nothing outside `adapters/pyagrum.py` imports it |
| Decision structure lives in a separate overlay | Relearning CPTs from new data must not require redoing decision modelling |
| Multiobjective handled by existing methods | Objective nodes emit one linear objective each over a shared feasible set |
| Compiled-model metadata (`IndexMap`, network, annotation) persists via the API's `ProblemMetaDataDB` side-table pattern, not a field on core `Problem` | Core `Problem` is frozen with `extra="forbid"` and has no metadata field; `ProblemMetaDataDB` already exists for exactly this — problem-specific side data plus a signal for the frontend to show a specialised view (see `SiteSelectionMetaData` in `desdeo/api/models/problem.py`) |
| Phase 4 targets gurobipy only, for now | Already a hard dependency, strongest MILP performance, fastest path to a working v1. Compiler output stays a plain `Problem`, so pyomo/other backends can be added later through the existing `guess_best_solver`/`find_compatible_solvers` machinery without redoing Phase 4 |
| Frontend Phase B integrates into Cumulus first | Concrete target chosen up front rather than decided mid-phase; existing backend (`desdeo/mcdm/cumulus.py`, `desdeo/api/routers/cumulus.py`) and webui route (`webui/src/routes/interactive_methods/CUMULUS`) already exist to integrate against |

---

## Backend

### Phase 0 — Encoding and constraint-generation spike

`TensorVariable` and tensor constraints transmit as tensors to solvers that support them,
but **few solvers handle more than two dimensions**. Cluster moments μ_C are indexed by the
joint state of every node in C, and clusters of 4–6 nodes are the normal case — so the
natural representation is exactly the one that doesn't survive the solver interface.

**Resolution: flatten the cluster state space.** μ_C becomes a vector of length ∏|S_j| under
a mixed-radix encoding. Local consistency, CPT constraints and μ/z linking are then all
linear maps — matrices times vectors, never above two dimensions.

**Ordering trick.** Within each cluster, make the separator variables toward the parent the
most significant digits. Marginalising onto that separator is then a sum over contiguous
blocks, one constraint per separator state, with no scattered index arithmetic. A cluster
with multiple children can only be contiguous for one separator; pick the parent's and accept
strided access for the others. Record the ordering per cluster in `IndexMap`.

- [x] Implement mixed-radix encode/decode with separator-major ordering
- [x] Write one cluster pair's local-consistency block at realistic cardinality
      (say 4 nodes × 5 states = 625 entries, separator 125)
- [x] Determine whether a slice sum can be expressed compactly, or needs explicit enumeration
- [x] Check whether a sparse aggregation matrix is workable as `TensorConstant`; a 125×625
      marginalisation matrix is ~98% zeros, so if tensors are dense, generate slice-sum
      equations directly instead
- [x] Measure `Problem` construction time and serialization size at that scale

Note the *number* of local-consistency constraints is the separator state count regardless —
that's inherent, not a representation artifact. What's variable is whether each one is a
compact expression or a large explicit string.

**Findings.** `desdeo/problem/graphical/index_map.py` implements
`mixed_radix_encode`/`mixed_radix_decode` (matches NumPy's row-major order) and
`separator_block_range`, which turns a 0-based separator state into the 1-based inclusive flat
index range of cluster states sharing it. Verified in `tests/test_graphical_index_map.py` at the
625/125 scale:

- A slice sum *is* compactly expressible, with existing operators — no new MathJSON operator or
  explicit index enumeration needed: `["Sum", ["Extract", mu_c, ["Tuple", start, stop]]]` per
  separator state, `start`/`stop` from `separator_block_range`. Confirmed correct against a
  NumPy ground-truth marginal through the gurobipy backend.
- A dense aggregation matrix (`TensorConstant` + `MatMul`, one vectorised constraint) is
  correct too, but is not the better representation: at this scale it cost ~307KiB serialized
  and ~85–95ms to construct, versus ~36KiB and ~1ms for 125 separate slice-sum constraints — the
  ~98%-zero matrix dominates on both construction time and wire size, consistent with the
  checklist's expectation. **Decision: the compiler generates slice-sum equations directly
  (Phase 4) and never materialises a marginalisation matrix.**
- The number of local-consistency constraints is separator-state-count regardless of
  representation, as expected — what the finding above sets is which of the two *equivalent*
  expression forms Phase 4 should emit.

**Outcome.** Sets the practical treewidth threshold for Phase 3 against gurobipy specifically
(the settled Phase 4 target — see design decisions), and tells you whether the compiler should
warn as clusters approach it. Revisit if a second backend is added later.

### Phase 1 — `DiscreteNetwork` and the pyAgrum adapter

Near-isomorphic pydantic mirror of pyAgrum's model. Same node categories, same potential
indexing order, same label conventions.

- [x] `DiscreteNetwork`, `DiscreteVariable`, `Potential` schemas
- [x] `desdeo/problem/graphical/adapters/pyagrum.py`: `to_pyagrum` / `from_pyagrum`
- [x] Import via pyAgrum's readers (BIF, XDSL, `.net`, XMLBIF) — no hand-written parsers
- [x] `BNLearner` wrapper for learning from data
- [x] **Round-trip property test**: `from_pyagrum(to_pyagrum(net)) == net` over generated
      random networks. This is the test that catches index-order drift. Compare structure
      exactly but potentials/CPT floats with a tolerance (`pytest.approx`), not `==` on the
      whole model — exact float equality through a round trip will be flaky.
- [x] Register a `graphical` pytest marker for this suite, for selection (`pytest -m graphical`)
      the same way `gurobipy`/`cvxpy`/`pyomo` markers are used today — no install-skip logic
      needed, pyAgrum is a hard dependency (see design decisions).

Adapter is confined to `adapters/pyagrum.py`; nothing downstream imports it, even though
pyAgrum itself is an ordinary hard dependency.

**Findings.** `pyagrum` (PyPI package name is lowercase, unlike the `pyAgrum` import used in
older docs/versions) added via `uv add pyagrum` — resolved cleanly, no extras mechanism needed,
matching the design decision. Installed pyAgrum 3.1.1, which renamed `Potential` to `Tensor`
internally; our own schema keeps the name `Potential` regardless, since it mirrors the CPT
*concept*, not pyAgrum's current internal class name.

Two indexing subtleties worth recording for anyone touching this adapter later:

- `Tensor.toarray()`'s axis order is the *reverse* of `Tensor.variablesSequence()` (pyAgrum's
  own per-CPT node-then-parents order). `Potential.variables`/`values` use the
  `variablesSequence()` convention (node first) since it reads more intuitively; the adapter
  transposes between the two. `fillWith()` requires the array to be C-contiguous, so the
  transpose needs an explicit `np.ascontiguousarray` — a bare transpose is just a reversed-stride
  view and pyAgrum rejects it.
- `BayesNet.arcs()` is an unordered set, so it can't be used to recover a node's *parent order*
  (which determines CPT axis order). `from_pyagrum` instead reads each node's parent order off
  its own CPT's `variablesSequence()`, and `to_pyagrum` adds each node's arcs in its own
  `Potential`'s parent order (confirmed empirically: pyAgrum's CPT parent order follows
  `addArc` call chronology) rather than in `DiscreteNetwork.arcs`'s list order. A DAG's
  topological order isn't unique, either — `DiscreteNetwork.variables`'s top-level list order
  can (and empirically does, for independent branches) shuffle across a round trip even though
  every node's own potential is unaffected; the round-trip tests compare by name/set rather than
  by list position for that reason.

`DiscreteVariable` carries kind-specific fields (`lower_bound`/`upper_bound` for `range`,
`integer_values` for `integer`, `ticks` for `discretized`, `numerical_values` for `numerical`)
alongside the always-present `labels`, so `to_pyagrum` reconstructs the same pyAgrum subclass
rather than flattening everything to a generic labelized variable. Confirmed round-tripping
through `to_pyagrum`/`from_pyagrum` for all five kinds. BIF (used for the `load_network`/
`save_network` test) normalises everything to a generic labelized-like representation on
reload, as expected of a plain-text CPT format — `kind` fidelity is only guaranteed for the
in-memory `to_pyagrum`/`from_pyagrum` path, not through a file round trip.

`learn_network(data, structure=...)` wraps `BNLearner.learnParameters`, keeping the DAG fixed
while relearning CPTs — this is the entry point the "decision structure lives in a separate
overlay" design decision depends on. Verified against a network `BNDatabaseGenerator` sampled
3000 rows from: recovered CPTs within 0.15 absolute tolerance of the source, and the DAG
unchanged. `learn_network(data)` without `structure` wraps `BNLearner.learnBN()` for the
free-structure case, verified to recover a valid probability distribution per node but not
checked against a specific structure (structure learning is inherently approximate; there's no
single correct answer to assert against for that path, unlike the fixed-structure case).

Tests: `tests/test_graphical_network.py` (26, pure-schema validators, no pyAgrum), and
`tests/test_graphical_pyagrum_adapter.py` (13, marked `graphical`, includes the round-trip
property test over `pytest.mark.parametrize`d random networks). All 928 tests in the suite still
collect cleanly; the 39 new ones pass under the same `-n auto -m "not fixme" --disable-warnings`
invocation `just test` uses.

### Phase 2 — `DecisionAnnotation`

Overlay referencing network nodes by name. Contains nothing the network provides.

- [x] `DecisionNode` (symbol, information set), `ObjectiveNode` (symbol, parents, utility
      tensor, sense), `DiagramConstraint`, `RiskMeasure`
- [x] `decision_order` (None ⇒ LIMID semantics) and `no_forgetting` flag
- [x] `validate_against(network)` — every referenced node exists, every state set matches
- [x] Schema version field
- [x] Round-trip test: relearn CPTs from new data, reapply annotation, assert it still validates

**Verify early:** pyAgrum's ID semantics additively sum multiple utility nodes into a single
scalar. Objective nodes here must stay a *vector*. Confirm this before using pyAgrum's
multi-utility support for anything, since a silent sum produces plausible wrong answers.

**Findings.** `desdeo/problem/graphical/annotation.py` adds `DecisionNode`, `ObjectiveNode`,
`DiagramConstraintKind`/`DiagramConstraint`, `RiskMeasureKind`/`RiskMeasure`, and
`DecisionAnnotation`. Unlike `DiscreteNetwork`, none of this mirrors a pyAgrum data model, so
these schemas follow the rest of DESDEO's `Problem` convention instead: elements are addressed
by `symbol` (matching `Variable`/`Objective`/`Constraint`), a deliberately distinct namespace
from `DiscreteNetwork`'s chance-node `name`s. `ObjectiveNode.utility` reuses `Potential` as-is
for the utility table — nothing in `Potential`'s schema requires normalised probabilities, only
shape consistency, so the same container serves both roles without a parallel schema.
`DiagramConstraint.cons_type` reuses `ConstraintTypeEnum` from `problem/schema.py` (the module's
one intentional import out of the graphical package), since it compiles into an ordinary
`Constraint` later and the standard-form convention should already match at this layer;
`DiagramConstraint.func`/`.nodes` stay at the node-symbol level (not mu/z), since compiled
variables only exist once Phase 4 has a specific `RootedJunctionTree` — `kind` records which
Phase 5 taxonomy row a constraint belongs to ahead of that.

Construction validates only *internal* consistency (symbol uniqueness across all four element
kinds, `risk_measures` targeting a real objective, `decision_order` being an exact permutation
of the decision symbols, `no_forgetting` requiring `decision_order`) since no network is
available at that point. `validate_against(network)` is the separate entry point for
network-dependent checks — node existence for `information_set`/`parents`/constraint `nodes`,
plus utility-tensor shape against the referenced nodes' actual cardinalities — and is
deliberately blind to potentials, so it's exactly what the "relearn CPTs, reapply annotation"
round trip needs. `tests/test_graphical_annotation.py` (29 tests, 1 marked `graphical` for the
relearning round trip, the rest pure-schema like `test_graphical_network.py`) covers both
validators plus the collision case where a decision symbol reuses a network variable name.

The "verify early" pyAgrum check: built a 2-utility-node `InfluenceDiagram` in pyAgrum directly
(binary decision `D`, utilities 10/0 and 0/100 for `D=0`/`D=1`) and ran
`ShaferShenoyLIMIDInference.MEU()`. Confirmed it returns one scalar (`{'mean': 100.0, ...}`),
i.e. the two utility nodes are silently summed. `posteriorUtility(node)` does return each
utility node's own table un-summed, though, which is the escape hatch Phase 4's oracle test 2
(single-objective pyAgrum comparison) will need for multi-objective cases: compare each
`ObjectiveNode` against its own `posteriorUtility`, never against a summed `MEU()`.

### Phase 3 — Rooted junction tree construction

The expensive, fragile stage. Persisted, not transient.

- [x] Persist via a `GraphicalModelMetaData` API side-table (JSON-blob columns for the network,
      annotation, RJT and `IndexMap`), mirroring `SiteSelectionMetaData`'s pattern — not a
      `ScenarioModelDB`-style relational mapping. Confirm first that nothing needs to query into
      RJT substructure at the SQL level; if it does, that pushes back toward the relational
      approach `ScenarioModelDB` had to use for its polymorphic pools.
- [x] `RootedJunctionTree` and `Cluster` schemas, with `ownership` and `order`
- [x] Construction from `(DiscreteNetwork, DecisionAnnotation)`, following Parmentier et al.'s
      gradual/rooted construction — seed from pyAgrum's triangulation if it adapts, otherwise
      implement directly
- [x] `max_cluster_size` property; refuse or warn above a configurable threshold
- [x] Cache keyed on hash of (structure, annotation) so CPT changes don't re-triangulate
- [x] Tests: gradual property holds, every node's cluster contains its information set,
      cluster order consistent with `decision_order`

**Findings.** `desdeo/problem/graphical/rjt.py` adds `Cluster`/`RootedJunctionTree` and
`build_rooted_junction_tree`, implementing Algorithm 1 ("Create a Minimal Gradual RJT Given a
Topological Order") of Parmentier, Cohen, Leclère, Obozinski & Salmon (arXiv:1902.07039), whose
"gradual RJT" terminology (Definition 2) is what Herrala, Terho & Oliveira's paper (the Phase 6
target) also uses unchanged. First pass at this used only lossy, AI-paraphrased fetches of both
papers (the PDFs initially rendered as opaque compressed binary to the available tools), and one
extracted claim from that pass — "the sink/value node becomes the root" — was wrong: it
contradicted the CPT-chain-rule equation extracted from the same lossy pass, and, once the
actual PDF text was extracted directly (`pypdf`, added ephemerally via `uv run --with pypdf`,
no project dependency change) and Algorithm 1's real pseudocode read line by line, turned out to
also contradict the paper's own proof — Proposition 3/4 establishes `(u ∈ C_v) ⇒ (u ⪯ v)`, i.e.
every node's cluster contains only ⪯-*earlier* nodes, which makes the root the ⪯-*first* node,
not the last. Re-reading the primary source directly, rather than continuing to reconcile
paraphrases against each other, is what actually resolved this — worth remembering next time a
paper is only reachable through a lossy fetch and something extracted from it doesn't add up.

**A second, more consequential bug fell out of the same re-read.** Algorithm 1's content-update
step unions into `C_v` only from nodes `w` that *themselves*, during their own (earlier) turn,
picked `v` as their arc target — a dynamic relationship discovered arc-by-arc as the algorithm
runs. An earlier version of this module instead unioned from every node whose *family* happened
to mention `v` (a static, precomputed relationship) — a plausible-looking shortcut that isn't
what the algorithm says, and it broke on the most ordinary Bayesian network shape there is: a
common-cause fork (`A -> B`, `A -> C`). Both B's and C's processing independently computed
`rest = {A}` and each tried to claim A as *its own* child, raising a spurious "double claim"
error on a perfectly valid structure (A legitimately ends up with two tree-children). The
symptom was patched once already (collect edges undirected, root and orient via a separate BFS
pass afterward) before the real cause was found; implemented per the algorithm's actual dynamic
`pending_children` mechanism (`_run_algorithm_1`), the fork case needs no BFS-reorientation
special-casing at all — each node records at most one arc, during its own turn, so the map from
child to parent can never receive a conflicting write, and A's own turn naturally absorbs both
B's and C's leftover content once their arcs have been recorded. This is the kind of thing
brute-force/oracle tests (Phase 4) would have caught eventually, but finding it here — from
reading the actual algorithm, not from a failing MILP — was considerably cheaper. The existing
tests (`test_fork_network_hub_has_two_children`,
`test_two_objectives_sharing_one_ancestor_does_not_conflict`) still cover exactly this case and
did not need to change; only the construction they exercise did.

**pyAgrum triangulation, tried and set aside.** `JunctionTreeGenerator.junctionTree(dag,
partial_order)` does accept an elimination partial-order and was confirmed interactively to
produce a correct, connected join tree given the right partial order — a genuine candidate for
"seed from pyAgrum's triangulation." (Separately, `ShaferShenoyLIMIDInference.junctionTree()`
was also tried first and rejected: on a small hand-built `InfluenceDiagram` its output failed a
basic clique-cover-every-moral-edge check, and nothing in pyAgrum's public API explains why
without reverse-engineering LIMID-reduction internals not worth depending on.) But
`JunctionTreeGenerator` triangulates on node count, not joint state-space size (Phase 0's actual
cost driver), and has no way to fold in `ObjectiveNode`/`DiagramConstraint` clique requirements
or to get the one-cluster-per-node "gradual" property. Implementing Algorithm 1 directly ended
up both simpler and exactly what the paper specifies — the "otherwise implement directly" branch.

**Objectives/constraints stay outside the tree's own node set.** Parmentier et al.'s own N is
*every* diagram vertex, including utility/value vertices — their MEU objective sums `⟨r_v,μ_v⟩`
over each one independently (their Eq. 11a), no special-casing needed for more than one, since
each just gets an ordinary cluster like any other vertex. (Herrala et al.'s own "Algorithm 1",
confusingly numbered the same, is a *different*, optional single-value-node preprocessing step
needed only for their CVaR reformulation — Phase 6's problem, not this one's; earlier notes in
this file conflating the two, from the lossy-fetch pass, were wrong.) DESDEO's `ObjectiveNode`
doesn't carry a state count the way a value vertex's domain does — it's a `Potential` over
`parents` directly, no domain of its own — so it can't become a first-class member of N without
inventing a cardinality that isn't part of its schema; this, not anything about multiple
objectives specifically, is why they're excluded. This also surfaced a real modelling fact worth
recording: `DiscreteNetwork.potentials`'s parent references only validate against
`DiscreteNetwork`'s own variable names, so a chance node's CPT can never depend on a decision in
this framework — decisions only ever feed `ObjectiveNode`/`DiagramConstraint`, never a CPT. The
chosen alternative for objectives/constraints: fold an objective's parents (or a constraint's
node set) into the family of whichever of their own members is ⪯-latest, routing the requirement
into an *existing* real node rather than a synthetic one.

`no_forgetting` expansion (each decision's information set folded with every earlier decision's,
per `decision_order`) is applied in this module, not deferred to Phase 4, since it directly
determines required cluster content. `_structure_hash` covers DAG/domain-size/decision-annotation
*structure* only — never CPT or utility values — so `learn_network(..., structure=...)`
relearning CPTs reuses the cached tree instead of re-triangulating, confirmed by
`test_cache_ignores_cpt_value_changes`.

`GraphicalModelMetaData` (in `desdeo/api/models/problem.py`) mirrors `SiteSelectionMetaData`'s
JSON-blob pattern exactly: `network_json`/`annotation_json`/`rjt_json`/`index_map_json` (the last
nullable — Phase 4 hasn't run yet). Confirmed nothing in the current API queries into a stored
problem's substructure at the SQL level the way `ScenarioModelDB`'s relational mapping needs to
for its polymorphic pools, so the side-table/JSON-blob approach applies as expected. Wired into
`ProblemMetaDataDB.all_metadata`/`ProblemMetaDataPublic` alongside the other metadata kinds; no
router yet (Phase 7's job — `POST /graphical/compile` is what will actually populate a row).

Tests: `tests/test_graphical_rjt.py` (18, pure-Python, no pyAgrum — correctness is checked
directly against the gradual-RJT definition via a running-intersection helper, not against a
specific external worked example) and `desdeo/api/tests/test_graphical_model_metadata.py` (2,
DB round-trip + `all_metadata` aggregation). `just test`'s full run (API + core) still passes
(273 tests across the touched suites, including the pre-existing Phase 0–2 graphical tests).

### Phase 4 — MILP compiler

```python
def compile_diagram(
    network: DiscreteNetwork,
    annotation: DecisionAnnotation,
    *,
    rjt: RootedJunctionTree | None = None,
    max_cluster_size: int | None = None,
    on_exceed: Literal["warn", "raise"] = "warn",
) -> tuple[Problem, IndexMap]:
```

- [x] μ vector per cluster under mixed-radix encoding (see Phase 0), z binaries per
      (decision, state, information state)
- [x] Local consistency between adjacent clusters, as slice sums under separator-major ordering
- [x] CPT constraints tying chance node moments to their potentials
- [x] Linking constraints between μ and z
- [x] One linear objective per objective node, over the shared feasible set
- [x] `IndexMap` returned alongside `Problem` for the library path (direct `compile_diagram`
      callers). For the persisted/API path it — along with the network and annotation — lives
      in the `GraphicalModelMetaData` side-table (see design decisions and phase 3), not on
      core `Problem`. Either way it carries the per-cluster digit ordering and is load-bearing
      for reading policies back out of solutions (phase E), not just for the UI

**Oracle tests — build these before the compiler, not after:**

1. Brute force. Enumerate policies for tiny diagrams. Obviously correct, catches everything.
2. pyAgrum `ShaferShenoyLIMIDInference.MEU()` for single-objective unconstrained cases.
   Note it treats models as LIMIDs by default — apply `addNoForgettingAssumption` when
   comparing against classical ID semantics, or the oracle answers a different question.
   `addNoForgettingAssumption` must be gated on `annotation.no_forgetting`, not merely on
   `annotation.decision_order` being set (confirmed empirically — see Findings).
3. Property-based generation of small random IDs, all three must agree.

**Findings.** `desdeo/problem/graphical/compile.py` implements the paper's Eq. 11/18 MILP
directly over the gradual RJT: one `TensorVariable` mu_C per cluster (mixed-radix flattened,
bounds `[0,1]`), one binary `TensorVariable` z per decision, `Constraint`s for CPT/decision-link
and local consistency, and one `Objective` per `ObjectiveNode`. `ClusterIndexMap`/
`DecisionIndexMap`/`IndexMap` (all in `compile.py`) carry the per-cluster/per-decision digit
ordering `compile_diagram` returns alongside the `Problem`.

**No separate mu_check{C} variable.** The paper's own convenience variable mu_check{Cv} :=
sum_{xv} mu_{Cv} is never materialised as a `TensorVariable` here -- every place it would be
referenced (the CPT/decision-link equation and the local-consistency equation) only ever reads
it through a linear expression, so one `ExtraFunction` per (cluster, separator state) names the
`Sum(Extract(mu_C, ...))` block-sum once instead, and every equation that needs mu_check{Cv}[j]
references that extra function's symbol. This is the same mechanism Phase 0 validated for
marginalisation, just named so it isn't re-derived (and re-serialised) at every one of the
`state_space_size` equations that need it.

**Gradual property fixes one cluster's own ordering, not its relationship to every neighbour.**
Because the RJT is gradual (`Cluster.separator == Cluster.nodes - {ownership}` always, Phase 3),
placing a cluster's separator nodes (canonically sorted) most-significant and its own owned node
least-significant makes that cluster's *own* upward marginalisation (summing out just its own
node) always the cheap, contiguous `separator_block_range` case from Phase 0. What Phase 0 didn't
yet need to resolve is the *other* direction: tying that separator-marginal to the matching
marginalisation of the tree-*parent's* cluster, which is ordered for its own upward link (toward
the grandparent), not for any one child's separator -- so for a parent with more than one child,
at most one child's separator can align with that ordering; the rest are strided. A single general
helper (`_group_flat_indices_by_projection`: one O(cluster state space) pass, decode-then-reencode
per state, no dense matrix) handles both the aligned and the strided case uniformly, and
`_extract_expr` opportunistically emits a compact `Tuple` range instead of an explicit index list
whenever a resulting group happens to be contiguous. The same general marginalisation is what
evaluating an `ObjectiveNode` over its containing cluster (`RootedJunctionTree.find_containing_cluster`)
reduces to, since an objective's parents are, in general, an arbitrary (not prefix-aligned)
subset of that cluster's nodes.

**Root normalisation, easy to miss.** CPT/decision-link equations are homogeneous in mu_{Cv}
(uniformly rescaling every mu and every mu_check by the same constant still satisfies them), so
nothing pins the root's total probability mass on its own -- exactly the extra degree of freedom
the classical MDP LP formulation's `mu^0_{s0} = 1` constraint exists to remove (Section 1.2 of the
paper). One explicit `Sum(mu_root) == 1` equality constraint fixes it; without it the compiled
model is under-determined (feasible at *any* overall scale) even though every other constraint
looks satisfied.

**McCormick bound `b = 1`.** The simplest admissible bound (Eq. 16); tightening it and the
Section 5 valid cuts are LP-relaxation-quality concerns for a later pass, not correctness ones --
the MILP (with `z` integrality) is exact regardless of which valid `b` is used, and nothing in
Phase 4's checklist asked for the cuts.

**A real bug caught by writing the oracle tests, not by inspection.** The z-linking code
originally conditioned each decision's z on its *declared* `DecisionNode.information_set` --
correct without `no_forgetting`, silently wrong with it, since the cluster the decision lives in
was built from the *effective* (no_forgetting-expanded) information set (Phase 3's
`_effective_information_sets`), not the declared one. Conditioning z on a smaller set than the
cluster it's embedded in is exactly the kind of mismatch that produces a plausible-looking but
under-informed policy rather than an obvious crash. Fixed by promoting `effective_information_sets`
from module-private to public in `rooted_junction_tree.py` (it was already exactly the right
function, just not exported) and using it in `compile.py` too. Caught only once
`test_two_decisions_with_no_forgetting` was checked against pyAgrum with `addNoForgettingAssumption`
-- brute force alone wouldn't have caught it, since brute force was (at the time) using the same
declared-information-set assumption the buggy compiler code did.

**pyAgrum oracle quirks, all confirmed empirically, none documented clearly enough to predict up
front:**

- `ShaferShenoyLIMIDInference` refuses outright ("This LIMID/Influence Diagram is not solvable")
  on a diagram with *zero* decision nodes -- unsurprising in hindsight (ID solving is fundamentally
  about policies), but it means the decision-less case (a plain expectation over a Bayesian
  network) needs a different pyAgrum entry point: `LazyPropagation` + `addJointTarget` (declared
  *before* `makeInference`, or `jointPosterior` raises "not addressable in this optimized
  inference") over the plain `BayesNet` part, contracted against the utility table directly.
- `posteriorUtility(node)` -- the escape hatch the Phase 2 findings flagged for multi-objective
  comparison -- does **not** collapse to a single expected value; it stays indexed by that node's
  own remaining decision/chance parents (confirmed by printing it: identical to the raw,
  un-reduced utility table). Recovering one objective's share of a multi-objective MEU needs more
  than reading it directly, so it's out of scope for this phase's oracle (every test case here has
  exactly one `ObjectiveNode`); `MEU()['mean']` is used instead, matching what the plan's checklist
  actually asked for ("single-objective unconstrained cases").
- `addNoForgettingAssumption` wants a list of decision **names** (`list[str]`), not the node ids
  its own docstring claims (`ids: list[int]`) -- the id overload raises
  `NotFound: No element with the key <>` even for a plainly valid id; the name overload works.
- `decision_order` being set on a `DecisionAnnotation` does **not** by itself mean
  `no_forgetting` semantics for pyAgrum either -- confirmed by a test that toggles only
  `no_forgetting` with `decision_order` fixed and gets two different (both oracle-agreeing)
  answers, so `addNoForgettingAssumption` must be gated on `annotation.no_forgetting`, not on
  `annotation.decision_order` being truthy.
- Two independent decisions with no chance-node or decision-to-decision arc between them at all
  (each blind to the other, no `no_forgetting`) leave `diagram.decisionOrderExists()` `False`, and
  `ShaferShenoyLIMIDInference` refuses to solve that diagram too -- pyAgrum apparently requires
  *some* derivable decision order even for a LIMID with genuinely independent decisions, which
  brute force and the compiled MILP both handle without complaint (there's nothing to order). One
  test case (`test_two_decisions_without_no_forgetting_is_worse_or_equal`) checks that half only
  against brute force, with a comment explaining why pyAgrum is skipped there specifically.

**Tests.** `tests/test_graphical_compile.py` (11 tests, marked both `graphical` and `gurobipy`
since every test solves via `GurobipySolver`): six structured cases (single chance node, a 2-node
chain, a 3-node fork with a joint 2-parent objective, an unconditional bandit decision, a
decision observing a chance node, two decisions with/without `no_forgetting`) plus a
multi-objective regression case and 3 property-based cases (`gum.randomBN`-seeded small networks,
per the adapter tests' own style, with a decision and objective grafted on), each checked against
brute force, (where solvable) pyAgrum, and an `IndexMap`-readback re-verification. `just test`'s
full run and the rest of the `graphical`-marked suite (109 tests across `problem/graphical`)
still pass.

**Code review (deep-dive pass, before merging).** A domain-aware review ran a 254-case
differential fuzz sweep of `compile_diagram` against the brute-force oracle (varying node count,
cardinality, decision information sets, `decision_order`/`no_forgetting`, objective parent order,
and multi-objective annotations) plus hand-built structural edge cases (a child separator that
projects onto a *permutation* of the parent's digit order; a parent cluster strictly larger than
a child's separator). The indexing arithmetic itself (`_decode_projection`/`_project_flat`/
`_group_flat_indices_by_projection`, 1-based conversions, `_extract_expr`'s contiguity test) held
up under all of it — zero numerical mismatches there. One real bug surfaced, plus several
secondary findings, all fixed:

- **Correctness (the one real bug).** An all-zero `ObjectiveNode.utility` table compiled to
  `func = [0.0]` — syntactically a MathJSON terminal, but not a node any backend's parser
  (`_parse_to_gurobipy`/`_parse_to_polars`/`_parse_to_cvxpy`) matches as an entire `func`, since
  none of the "operator" or "single-symbol" branches fire for a bare numeric list. Because
  `GurobipyEvaluator` parses every objective at construction, one degenerate all-zero-utility
  objective made the *whole compiled `Problem`* fail to parse, silently taking down every sibling
  objective too. Fixed: the all-zero fallback now emits `["Multiply", 0.0, ["At", mu_symbol, 1]]`
  — a genuine (if trivial) linear expression — instead of a bare literal. The equally-affected
  `utility_array.ndim == 0` branch (meant for a parent-less/constant objective) was removed
  outright rather than patched the same way: `ObjectiveNode.utility` is a `Potential`, and
  `Potential`'s own shape validator can't actually be satisfied with `shape=[]` (`np.array(values)`
  is never 0-d for any constructible `values`), so that branch was dead code with the identical
  latent bug, not a reachable path. Caught only by the fuzz sweep — every hand-picked test case
  used a nonzero utility table, so the 10 pre-review tests never exercised it. A regression test
  (`test_multi_objective_including_all_zero_utility`) now pins it directly.
- **Robustness.** A caller-supplied `rjt` (the `rjt=` parameter, meant for reuse across recompiles
  — e.g. after a Phase 6 `RiskMeasure` transform) was checked only for its node *set* matching,
  not for the family containment every downstream helper assumes (a chance node's CPT parents, a
  decision's effective information set, a cluster's separator toward its parent, all inside the
  cluster that's supposed to contain them). A stale tree from a materially different annotation
  used to fail with a bare `KeyError` naming neither the cluster nor the cause. Fixed:
  `_check_family_contained`, called at all three sites, raises a clear `ValueError` instead.
- **Efficiency.** `_decode_projection` rebuilt a `{node: index}` position dict from scratch on
  every call — on the hot path, once per cluster state, so cost scales with `state_space_size`.
  Measured (reviewer's profile): a 15,625-state cluster spent ~0.85s in `mixed_radix_decode` and
  ~0.74s cumulative rebuilding position dicts, dwarfing the ~0.24s actually spent building the
  resulting `Problem`. Partially fixed: `_position_map` now memoises the dict per distinct
  `order` via a bounded `lru_cache` (same bounding precedent as the RJT builder's own
  `_CACHE_MAX_SIZE`), which removes the *rebuild-per-call* cost cheaply and safely. The reviewer's
  further suggestion — vectorising the whole decode/group step with NumPy — was deliberately not
  done in this pass: a larger rewrite of exactly the arithmetic the fuzz sweep just finished
  validating carries more correctness risk than the win justifies here, especially now that
  `max_cluster_size`/`on_exceed` (next item) gives callers a way to catch pathological scale
  before it's spent. Worth revisiting if a real workload's cluster sizes make the constant factor
  matter.
- **Usability.** `compile_diagram` had no `max_cluster_size`/`on_exceed` passthrough to
  `build_rooted_junction_tree`, so the *default* entry point (no pre-built `rjt`) gave no warning
  before starting to emit `O(state space)` constraints for an oversized cluster — exactly the
  scenario Phase 3's `ClusterSizeWarning` exists to flag. Fixed: both parameters now exist on
  `compile_diagram` and forward through when it builds its own tree.
- **Invariant.** Nothing checked that a chance node's CPT actually normalises to 1 — the
  McCormick `b = 1` bound (see above) is only a valid upper bound *because* of that, and an
  unnormalised CPT used to compile without complaint and fail only as an opaque solver-reported
  infeasibility. Deliberately **not** fixed by tightening `Potential`/`DiscreteNetwork`'s own
  schema validator (Phases 1–3's fixtures — `tests/fixtures/graphical.py`'s `chain_network`/
  `fork_network`/etc. — use deliberately non-normalised tables for structure-only testing, and
  `ObjectiveNode.utility` reuses `Potential`'s schema specifically to *avoid* a normalisation
  requirement); instead added as `_check_cpts_normalized`, called once at the top of
  `compile_diagram` itself, scoped to exactly where the assumption is actually used.
- **Test-convention/coverage.** The test module solves via `GurobipySolver` in every test but was
  only marked `graphical`, not `gurobipy` (inconsistent with every other Gurobi-dependent suite in
  the repo) — fixed. No test compiled more than one `ObjectiveNode` together (this is exactly why
  the all-zero-utility bug's blast radius on sibling objectives went unnoticed) or read `IndexMap`
  back at all (the plan calls it "load-bearing for Phase E", but nothing exercised it) — both
  fixed: a multi-objective regression test, and an `IndexMap`+solved-`z` policy-readback
  re-verification added to every existing test via `_assert_all_agree`.
- **Minor, applied:** every compiled `Constraint`/`Objective`/`ExtraFunction` now also sets
  `is_convex=True`/`is_twice_differentiable=True` alongside `is_linear=True` (previously left at
  their `False` defaults despite the whole compiled `Problem` genuinely being both, which silently
  excluded it from twice-differentiable-gated solver fallbacks like `pyomo_bonmin`); a
  pre-existing dead duplicate `return order` in `rooted_junction_tree.py` (unrelated to this
  phase, but in a file this phase already touches) was deleted; `_build_cpt_constraints`/
  `_build_decision` no longer take a redundant `own_cardinality` parameter that duplicated
  `ClusterIndexMap.cardinalities[-1]`; the module docstring now notes the `mu_*`/`z_*`/`sep_*_*`/
  `cpt_*`/`mc_hi_*`/`mc_lo_*`/`zsum_*`/`lc_*`/`mass_*` symbol prefixes are reserved.

### Phase 5 — Constraints

Taxonomy worth surfacing in the API, since it tells users what's free:

| Constraint on | Linear in | Cost |
|---|---|---|
| Decisions (budget, logic, exclusivity) | z | Free |
| Event probabilities | μ | Free |
| Expected values | μ | Free |
| Tail / distributional | — | Requires phase 6 |

- [x] `_build_diagram_constraint` (`compile.py`) compiles one `DiagramConstraint` into one
      `Constraint` on the `Problem` `compile_diagram` already builds; wired into `compile_diagram`
      right after the objectives are built, added to the same `constraints` list
- [x] `kind == "decision"`: every referenced node must be a `DecisionNode` with an empty
      *effective* information set (`effective_information_sets`); `["At", node, k]` substitutes
      directly to `["At", z_node, k]` (no marginalisation needed, since z is just
      `own_cardinality` entries for such a decision). A decision with a non-empty effective
      information set raises `ValueError` (info-conditional decision constraints are out of scope
      for this phase)
- [x] `kind in {"probability", "expected_value"}`: every referenced node (chance or decision)
      gets `["At", node, k]` substituted with that node's own-state marginal, read out of its
      containing cluster's `mu` (`rjt.find_containing_cluster([node])` +
      `_group_flat_indices_by_projection` + `_extract_expr`, the same machinery
      `_build_objective` already uses). Both kinds compile through the identical code path --
      `kind` only records the constraint author's intent, per the taxonomy table
- [x] `kind == "tail"` raises `ValueError` at compile time, naming Phase 6 as the missing piece
- [x] A referenced `ObjectiveNode` symbol is rejected for `decision`/`probability`/`expected_value`
      kinds (an objective owns no cluster/`z`) -- only legal for `tail`, which isn't implemented
- [x] `func` may only reference a `DiagramConstraint.nodes` member as `["At", "<node>", k]`, a
      single 1-based own-state index; any other form (bare symbol leaf, operator position,
      `Extract`/`Sum`) raises `ValueError` via a dedicated tree walk (`_substitute_at_nodes`),
      not the generic `replace_str` helper (unsafe for a targeted, `At`-only substitution).
      Arbitrary combinations across states are still expressible via several `At` terms combined
      with `Add`/`Multiply`/`Subtract`, exactly like every `func` this module itself emits
- [x] Out-of-range `At` state indices (`k` outside `1..cardinality[node]`) raise a clear
      `ValueError`, not a bare `IndexError`
- [x] Tests (`tests/test_graphical_compile.py`, extended in place -- `pytest.mark.graphical` +
      `pytest.mark.gurobipy`, same as Phase 4's suite): a `decision`-kind exclusivity constraint
      checked against a brute-force oracle restricted to feasible policies
      (`_brute_force_optimum_feasible`); a `probability`-kind bound on a decision's own marginal,
      and an `expected_value`-kind weighted-cost bound, both checked by reading the actual solved
      policy back via `IndexMap` and verifying the bound against an independent brute-force
      marginal (`_brute_force_marginal`, no dependency on the compiler's own mu machinery); `tail`
      raising `ValueError`; a decision constraint on a non-empty-effective-information-set
      decision raising `ValueError`; unsupported node-reference forms (bare symbol, `Extract`)
      raising `ValueError`; an out-of-range `At` index raising `ValueError`

**Findings.** `_build_diagram_constraint` (`compile.py`) compiles each `DiagramConstraint` per the
`kind` split the plan's taxonomy table already implies: `decision` is a hard, realization-
independent restriction on the policy itself (linear in z), while `probability`/`expected_value`
are restrictions on the induced distribution (linear in mu) -- genuinely different substitution
targets, not two names for the same thing, even though the substitution *mechanism* (walk `func`,
replace `At` matches, recurse) and the resulting `Constraint`'s shape (`is_linear`/`is_convex`/
`is_twice_differentiable = True`, symbol/name/cons_type taken straight from the
`DiagramConstraint`, mirroring `_build_objective`) are identical either way. The recommended
design from the task brief was implemented as given, with no deviations.

**The `At`-only scope restriction is deliberate, not a shortcut that quietly limits expressiveness.**
`_substitute_at_nodes` walks `func` once, matching `["At", "<node>", k]` sub-lists exactly and
raising on any other appearance of a `nodes` member (bare leaf, operator position, `Extract`/`Sum`
target, or a non-literal-int `k`) -- confirmed by `test_constraint_func_unsupported_node_reference_raises`,
parametrised over a bare-symbol and an `Extract` case. Nothing expressible is actually lost: a
constraint over a node's *combination* of states (e.g. "P(A in {lo, mid}) <= 0.5") is just
`["Add", ["At", "A", 1], ["At", "A", 2]]` compared against the bound, same as how this module's
own emitted `func`s (e.g. `_build_objective`'s `terms`) are themselves built from `Add`/`Multiply`
over individual `At`/`Extract` pieces, not one opaque tensor op.

**`decision`-kind substitutes against z directly; `probability`/`expected_value` substitute
against a cluster marginal of mu -- picking the right one per `kind`, not always the cluster
marginal, is what keeps a decision constraint an *exact* (not LP-relaxed) restriction on the
deterministic policy.** Marginalising a decision's `mu` instead (the `probability`/`expected_value`
path) would give the induced *probability* of that decision's state under the current policy --
correct for a probability/expected-value constraint on a decision's outcome (`test_probability_
constraint_on_decision_marginal_changes_optimum`, `test_expected_value_constraint_on_decision_
cost_changes_optimum`, both deliberately pick a decision node exactly to exercise the "decisions
also own an RJT cluster and hence a mu" case in the task brief), but would be the wrong substitution
for a `decision`-kind constraint, which needs a hard 0/1 restriction on *which* policy is legal, not
a bound on the resulting distribution. `test_decision_constraint_exclusivity_changes_optimum` is the
`decision`-kind analogue, checked against `_brute_force_optimum_feasible` (a `_brute_force_optimum`
variant restricted to policies passing a feasibility predicate) rather than a marginal read-back,
since a `decision`-kind constraint has no meaningful "marginal" reading in the first place.

**Restricting `kind == "decision"` to decisions with an empty effective information set was the
one place this phase's scope genuinely bites, and it bites for a structural reason, not
laziness.** A decision with a non-empty effective information set has `z_node` shaped
`information_set`-major/`states`-minor (`DecisionIndexMap`) -- `["At", node, k]` would then be
ambiguous (which information state's z?) without a per-information-state replication of the
constraint the plan brief explicitly scoped out. Caught and rejected with a clear message
(`test_decision_constraint_with_nonempty_information_set_raises`) rather than silently picking an
arbitrary information state or summing across them (either of which would silently answer a
different question than the one the constraint author asked).

**No real bugs this phase, unlike Phase 4's fuzz-sweep findings -- worth recording precisely
because it's notable.** Two things that could plausibly have gone wrong and didn't, confirmed by
building the oracle tests rather than assumed: (1) `rjt.find_containing_cluster([node])` (a
*singleton* target, deliberately not the joint `DiagramConstraint.nodes` clique the RJT builder's
own `_extend_family_for_clique` already folds in for `constraint.nodes` as a whole) always
succeeds, since every node trivially owns a cluster containing at least itself by RJT
construction -- the defensive `containing is None` branch in `_build_diagram_constraint` is
genuinely unreachable in every test built here, exactly as expected; (2) the weighted-sum
`expected_value` test (`2 * P(D=="s0") + 5 * P(D=="s1") <= 3.0`, two separate `At` matches
combined with `Add`/`Multiply` in one `func`) compiled and solved correctly on the first attempt,
confirming `_substitute_at_nodes`'s per-match substitution composes correctly under ordinary
MathJSON arithmetic exactly as the module docstring claims, not just for a single lone `At`.

**Tests.** `tests/test_graphical_compile.py` grew from 11 to 19 tests (8 new, all marked both
`graphical` and `gurobipy`, consistent with the rest of the suite): the seven cases the task asked
for, plus a shared `_read_back_policy` helper factored out of what was
`_read_back_policy_and_reverify`'s own inline block (so the new probability/expected-value tests
can read back the actual solved policy the same way, without duplicating that bookkeeping) --
`_read_back_policy_and_reverify` itself is otherwise unchanged, confirmed by rerunning it. All 19
tests in `tests/test_graphical_compile.py` pass (`pytest -m "graphical and gurobipy"`, ~1015s, one
gurobipy solve per test). The full `graphical`-marked suite (`pytest -m graphical tests/`, 59
tests across `problem/graphical`) passes with no regressions from the `compile.py` changes.

**Code review pass (before merging).** A deep-dive review of this phase's diff found two real,
confirmed bugs, both fixed, plus several smaller issues:

- **Correctness (most severe): a nonlinear `func` was silently flagged as linear.**
  `_build_diagram_constraint` hardcoded `is_linear=True`/`is_convex=True`/
  `is_twice_differentiable=True` on every compiled `Constraint` without ever checking that the
  substituted expression is actually linear. A `probability`/`expected_value`-kind constraint
  multiplying two node-derived `At` terms together (e.g. `["Multiply", ["At","A",1],
  ["At","D",1]]`, which a constraint author might reach for meaning "P(A) and P(D)" without
  realizing it computes an independence-assuming *product*, not a joint-event probability) still
  substituted to a genuinely bilinear expression and reported it as linear/convex/twice-
  differentiable regardless -- corrupting `Problem.is_linear` (an `all()` over every element, so
  one lying constraint poisons solver selection for the whole `Problem`) and risking a quadratic
  constraint being handed to a pure-MILP solver on the strength of the false flag. Fixed by
  rejecting at compile time rather than silently handling it: `_substitute_at_nodes` now returns
  `(substituted_subtree, is_node_derived)` per recursive call, and a new `_check_node_derived_linear`
  rejects any operator combining node-derived operands non-linearly (`Multiply` with more than one
  node-derived operand, `Divide` with a node-derived denominator, any other operator applied to a
  node-derived subtree) -- `Add`/`Subtract` (any combination) and `Multiply`/`Divide` by a plain
  numeric constant remain unrestricted, since that's the normal "weight × probability" case
  `expected_value` constraints need. Regression tests added: a `Multiply` of two `At` terms now
  raises `ValueError` at compile time (`test_constraint_multiply_of_two_node_terms_raises`); the
  legitimate `Multiply`-by-constant case was already covered
  (`test_expected_value_constraint_on_decision_cost_changes_optimum`) and still passes.
- **Correctness: a node referenced in `func` but missing from `DiagramConstraint.nodes` was
  silently passed through unsubstituted.** `_substitute_at_nodes`'s "reserved leaf, reject if
  referenced in an unsupported form" check was scoped only to `diagram_constraint.nodes`, not to
  every symbol that actually exists in the compiled model -- so a `func` referencing an undeclared
  chance/decision node used to compile cleanly and only fail at *solve* time with an opaque
  `GurobipyEvaluatorError` naming neither the constraint nor the missing `nodes` entry, and, worse,
  if the leftover (never validated) string happened to collide with another compiled symbol --
  nothing currently forbids an `ObjectiveNode.symbol` from colliding with a chance/decision node's
  own name -- it would silently resolve at solve time to *that* symbol's own compiled expression
  instead of erroring, with no error anywhere. Fixed: `_substitute_at_nodes` now takes a
  `known_symbols` parameter (`cardinality | objective_symbols`, i.e. every chance/decision/
  objective symbol that exists anywhere in the compiled model, passed from `compile_diagram` via
  `_build_diagram_constraint`) and raises `ValueError` for any occurrence of a `known_symbols`
  member outside `nodes`, in any form -- the same class of clear compile-time error as the
  existing "declared but referenced in an unsupported form" case, not a silent fall-through.
  Regression tests added: an undeclared chance-node reference now raises
  (`test_constraint_referencing_undeclared_node_raises`, mirroring the review's own example), and a
  `func` reference colliding with an `ObjectiveNode.symbol` is rejected rather than silently
  resolving to the objective (`test_constraint_referencing_colliding_objective_symbol_raises`).
- **Docstring accuracy.** The module docstring and `_build_diagram_constraint`'s docstring used to
  imply "arbitrary combinations across states are expressible by combining several `At` terms"
  without qualifying that this only ever covers **linear combinations of single-node marginals** --
  a joint-event probability (e.g. `P(A == lo AND D == s0)`) is genuinely not expressible this way,
  and the natural-looking `Multiply` encoding computes a different (and usually wrong) quantity, an
  independence-assuming product, exactly why the first bug fix above now rejects it. Reworded to
  say so plainly, in both docstrings.
- **Error message wording.** The out-of-range `At` index error read like a bug report rather than
  a documented scope choice for a legitimate DESDEO tensor-indexing form (negative/end-relative
  indices are valid MathJSON elsewhere in the codebase). Reworded (cheaper than normalising `k < 0`
  and supporting it, and no caller needed negative indices here) to state explicitly that only
  positive 1-based indices are supported by this constraint form, not DESDEO's general end-relative
  indexing.
- **Dead code / redundancy.** The `if node not in cardinality` defensive branch inside the
  `probability`/`expected_value` marginal-building code was confirmed unreachable -- every `node`
  reaching `make_expr` is guaranteed (by the `known_symbols` check above, plus the pre-existing
  `referenced_objectives` rejection ruling out objectives) to already be in `cardinality` -- so it
  now carries a `# Defensive:` comment matching this module's existing convention, rather than
  being silently trusted or deleted outright. `decision_by_symbol` was confirmed redundant as a
  parameter to `_build_diagram_constraint` specifically (its only use there was a membership test,
  and `decision_maps` -- built from the same key set by construction, by the time
  `_build_diagram_constraint` runs -- already answers it); simplified by dropping the parameter and
  checking `decision_maps` instead. `decision_by_symbol` itself stays in `compile_diagram`, since
  it's still needed elsewhere (fetching the actual `DecisionNode` for `_build_decision`). The
  unused `network` parameter on the test module's `_read_back_policy` helper was removed.
- **Test quality.** The `probability`/`expected_value` decision-marginal tests previously only
  asserted the compiled optimum against a hand-computed literal and checked that the *returned*
  policy satisfies the bound -- little power for these specific fixtures, since almost any policy
  trivially satisfies it. Both now also check the compiled optimum against
  `_brute_force_optimum_feasible` restricted by a feasibility predicate built from
  `_brute_force_marginal`, a real constrained-brute-force oracle rather than a restated assumption.
  A new positive `probability`-kind test on a *chance* node (not a decision), with cardinality ≥ 3,
  deliberately constructed so its containing cluster (`rjt.find_containing_cluster([node])`) is a
  *different* node's cluster, not its own
  (`test_probability_constraint_on_chance_node_via_nonowning_cluster`) -- exactly the configuration
  most likely to hide an index/ordering bug in the marginal-substitution machinery, and nothing in
  the pre-review suite exercised it.
- **Deliberately deferred, not fixed here.** The reviewer separately flagged that
  `build_rooted_junction_tree` still folds every `DiagramConstraint.nodes` set into one joint
  clique (Phase 3's `_extend_family_for_clique`) even though Phase 5's `At`-only design never
  actually uses that joint clique -- every reference is a single-node marginal, read out
  independently per node, never a joint lookup across the whole `nodes` set together. For a
  constraint spanning many otherwise-unrelated nodes this is a real, avoidable efficiency cost
  (an unnecessarily large fused cluster). Left unfixed in this pass: it's Phase 3 code with its
  own deliberate, separate test coverage
  (`test_diagram_constraint_over_two_unrelated_nodes_also_gets_a_home_cluster`), and reversing that
  design as a side effect of a Phase 5 bugfix pass risks a regression there for a concern that's
  about performance, not correctness -- worth its own follow-up if a real workload's constraints
  make the extra fusion cost matter.

All 23 tests in `tests/test_graphical_compile.py` (19 + 4 new: the `Multiply`-of-two-`At`-terms
regression, the undeclared-reference regression, the colliding-objective-symbol regression, and
the non-owning-cluster positive case -- see above) pass
(`pytest -m "graphical and gurobipy" tests/test_graphical_compile.py`). The full `graphical`-marked
suite (`pytest -m graphical tests/`, 63 tests) passes with no regressions from this review pass's
changes.

### Phase 6 — Risk measures

Applied as **transformations of the diagram or the RJT, before variables are emitted** —
not as constraints appended afterward. Architecting them as the latter will have to be unpicked.

- [x] `RiskMeasure` handling inside phase 3, driven by `annotation.risk_measures`
- [x] CVaR first; follow Herrala, Terho & Oliveira (2025)
- [x] Validate against published benchmark results, not just self-consistency

**Findings.** `desdeo/problem/graphical/compile.py` gains `_build_risk_measure` (plus the shared
`_objective_groups` helper, factored out of `_build_objective` for reuse) implementing Herrala,
Terho & Oliveira's Eq. 11-21/Table 1 exactly, one eta (scalar `Variable`) plus four length-`|U|`
`TensorVariable`s (`lambda`/`lambda_bar` binary, `rho`/`rho_bar` in `[0,1]`) per `RiskMeasure`, and
one `Objective` per `RiskMeasure` (`symbol = RiskMeasure.symbol`) computing
`CVaR_alpha = (1/alpha) * sum_u rho_bar(u) * u`. Wired into `compile_diagram` right after the
`DiagramConstraint`s, in the same pass that builds mu/z/objectives/constraints -- reading the
target `ObjectiveNode`'s realizable utility support `U` and each `p(u)` as a genuine linear
expression over the target's containing cluster's own `mu` (via `_objective_groups` +
`_group_flat_indices_by_projection` + `_extract_expr`, the identical machinery `_build_objective`
and Phase 5's `_build_diagram_constraint` already use), not read off an already-finished `Problem`.
This is what the plan's own "applied as transformations... not as constraints appended afterward"
note (this section's preamble) is read to mean here: the CVaR variables/constraints are emitted as
part of the same compile pass that builds every other variable, using the target's own
already-in-progress `mu`, rather than bolted onto a finished `Problem` externally the way Phase 5
deliberately rejected for `DiagramConstraint`'s own `kind == "tail"`.

**"RiskMeasure handling inside phase 3" resolved to "already true, confirmed by reading the code,
nothing new needed" -- not a new change to `rooted_junction_tree.py`.** `build_rooted_junction_
tree`'s `_extend_family_for_clique` already folds *every* `ObjectiveNode`'s parents into some
cluster's family unconditionally, for every objective, not gated on anything risk-measure-related
(confirmed by reading `rooted_junction_tree.py` directly, not assumed) -- exactly the structural
guarantee the paper's own (confusingly identically-numbered, but unrelated) "Algorithm 1"
single-value-node preprocessing step exists to provide for their path-based decision-programming
baseline. DESDEO's `ObjectiveNode` never needed that preprocessing to begin with: Phase 2's design
("deliberately kept singular and vector-friendly", one cluster-worthy clique per objective by
construction) already gives a `RiskMeasure`'s own target exactly the guarantee Algorithm 1/2 exist
to establish elsewhere. `compile_diagram`'s own `rjt` docstring now records this explicitly (no
RJT-level transform needed for `RiskMeasure`, same tree serves both the expected-value objectives
and the CVaR ones).

**A real, confirmed erratum in the paper's own Table 1 -- found by reading the rendered PDF image
directly at the character/glyph level, not by trusting the (correctly-flagged-as-unreliable) pypdf
text dump.** The task brief's own warning was that pypdf silently drops the overline distinguishing
`lambda(u)`/`lambda_bar(u)` and `rho(u)`/`rho_bar(u)` -- true, and avoided by reading Section
3.4/Table 1 as a rendered image (`pymupdf`, added ephemerally via `uv run --with pymupdf`, no
project dependency change, matching Phase 3's own `pypdf`-ephemeral-fetch precedent) at up to
2400 dpi. But the *image itself*, independent of any extraction tooling, contains a genuine
content error: Table 1's own description of `rho_bar(u)` at `u = eta` reads "`alpha -
sum_{u in U} p(u)`" (confirmed by inspecting the PDF's actual glyph/font data, not just eyeballing
the render -- the character is a plain Latin italic `p`, font `NimbusRomNo9L-ReguItal`, not a
bar-dropped `rho`). That's dimensionally impossible: `sum_{u in U} p(u) = 1` always (p is a full
probability mass function), so `alpha - 1` is negative for any `alpha < 1`, violating `rho_bar`'s
own `[0,1]` bound (Eq. 20). The paper's own body text one paragraph later (page 7) gives the
algebraically consistent version: "`rho_bar(u)` must be `alpha - sum_{u in U} rho(u)` for
`u = eta`" -- using `rho(u)`, not `p(u)`; since `rho(u) = 0` for every `u >= eta` (established
earlier in that same paragraph, and independently re-derivable from (16)-(17) directly), summing
`rho(u)` over the whole of `U` is exactly the tail sum `sum_{u < eta} p(u)`, which is what (18)
actually forces once `rho_bar(u) = p(u)` for `u < eta` and `0` for `u > eta`. This module doesn't
reproduce either textual formula -- `rho_bar(u)` is only ever pinned by (11)-(18) directly, derived
independently from the constraints themselves (see the module docstring) -- so the erratum doesn't
affect correctness here, but it's recorded for anyone cross-checking this module against the
paper's own Table 1 directly, the same spirit as Phase 3's lossy-fetch story: read the primary
source at the level of detail the claim actually needs, not the summary a step above it.

**Sign convention for a minimizing target -- the paper's own reasoning, extended, not something the
paper itself addresses.** The paper's own `u` is a value/utility to be maximized throughout,
consistent with `CVaR_alpha` averaging the *lower* alpha-tail being the risk-averse direction for
maximization. `ObjectiveNode.maximize` can be `False` in DESDEO (a cost); `_build_risk_measure`
handles that by running the identical paper formulation over the *negated* utility values
internally, then negating the resulting `CVaR_alpha` expression back before reporting it as the
compiled `Objective` (`maximize=objective.maximize`, mirroring the target's own sense rather than
hardcoding `True`) -- exactly the standard Rockafellar & Uryasev CVaR-*of-loss* framing (the
risk-averse average of the worst, i.e. highest, cost outcomes, minimized). `RiskMeasureIndexMap`
(new, alongside `ClusterIndexMap`/`DecisionIndexMap` in the same bookkeeping style) records
`utility_values`/`eta_symbol` in this *internal* sign convention plus a `target_maximizes` flag, so
a caller reading a solved `eta` back as `VaR_alpha` in the target's own units knows whether to
negate it -- documented explicitly in the class docstring, the same "load-bearing for reading a
solution back" role Phase 4's `IndexMap` already plays for mu/z.

**A genuine expressiveness limitation surfaced while building the pig farm fixture, not a Phase 6
bug.** `DiscreteNetwork` chance-node CPTs may only be conditioned on other chance nodes, never a
decision (confirmed, matches Phase 3's own already-recorded finding in `rooted_junction_tree.py`'s
module docstring) -- but the pig farm ID's defining structural feature is exactly that the
treatment decision changes the *chance* transition into the next period's health state, an arc
pyAgrum's own `InfluenceDiagram` supports natively but this package's `DiscreteNetwork`/
`DecisionAnnotation` split deliberately doesn't allow. Resolved with a faithful reformulation
(`tests/test_graphical_compile.py`'s `_pig_farm_network`/`_pig_farm_annotation`): two independent
"counterfactual outcome" chance nodes (`CT`/`CN`, "outcome if treated"/"outcome if not treated"),
both conditioned on `H1` only and always drawn regardless of which choice the decision actually
makes, with the decision-dependent *selection* pushed into the `ObjectiveNode`'s own utility table
instead (which legitimately can have a decision as a parent). Since `CT`/`CN` are conditionally
independent given `H1` by construction, this reproduces exactly the same joint distribution over
`(H1, D1, realized outcome)` as the paper's own decision-conditioned CPT would -- a faithful
reformulation forced by this compiler's current expressiveness, not a different problem, and
correct enough to serve as the literal-instance fixture the task asked for. Still, this is a real,
structural limitation worth a maintainer's attention beyond Phase 6: any future genuinely
decision-conditioned chance transition (the general POMDP-style case, or the paper's own
multi-period pig farm beyond one treatment period) needs the same reformulation by hand, or a
schema change letting `DiscreteNetwork.potentials` reference a decision -- out of scope here.

**Validation -- what it established, and what it plainly couldn't.** The paper's own Section 4 and
Appendix A (re-confirmed directly from the full extracted text and the PDF, not assumed from the
task brief) report only solve-time comparison plots for the risk-averse pig farm experiment, no
exact reproducible numeric CVaR value anywhere -- so "validate against published benchmark
results" could not mean "reproduce a published number" here, and this doesn't fabricate one to make
the checklist item look more thoroughly matched than it is. What was actually established, all in
`tests/test_graphical_compile.py`:

1. A literal (necessarily reformulated -- see above) instance of the paper's own pig farm running
   example (Section 2.1's exact numbers, confirmed against the PDF page 2: 1000/300 DKK
   healthy/ill price, 100 DKK treatment cost, 80%/90% test specificity/sensitivity, 0.5/0.1 cure
   probability with/without treatment, 0.1/0.2 disease-development probability with/without prior
   treatment, 0.1 initial disease probability), compiled at the paper's own Section 4.1
   experimental confidence level (`1 - alpha = 0.85`, i.e. `alpha = 0.15`, confirmed against the
   PDF page 7) -- `test_pig_farm_cvar_matches_brute_force_oracle`.
2. `_brute_force_cvar` (built on `_utility_distribution_for_policy` + `_cvar_from_distribution`,
   the textbook Rockafellar & Uryasev VaR/CVaR definition -- sort ascending, accumulate mass to
   `alpha`, weight the boundary atom fractionally -- applied to a from-first-principles utility
   distribution computed directly from the network's own CPTs and the objective's own utility
   table), fully independent of `_build_risk_measure`'s own eta/lambda/rho machinery: two
   hand-computed cases (`test_cvar_hand_computed_maximize`/`_minimize_sign_convention`, exact
   fractional CVaR values worked out by hand and checked with `pytest.approx`, one exercising the
   maximize case and one the minimize sign-flip directly), a degenerate `|U| == 1` case
   (`test_cvar_degenerate_single_utility_value`, exercising `_cvar_epsilon`'s/`big_m`'s
   zero-separation fallback), a genuine EV-optimal-vs-CVaR-optimal policy divergence
   ("safe" guaranteed payoff vs. a higher-expectation "gamble",
   `test_risk_averse_decision_prefers_safe_over_risky_bet` -- the actual point of a risk measure,
   not just a fixed-distribution check), the pig farm instance above, and 3 property-based
   `gum.randomBN`-seeded small random diagrams with a grafted decision/objective/risk measure
   (`test_random_small_diagrams_cvar_matches_brute_force`, same generation style as Phase 4's own
   property tests).
3. Two literature relationships (Rockafellar & Uryasev 2000, cited by the paper itself for CVaR),
   `test_cvar_approaches_expected_value_and_is_bounded_and_monotone`: `CVaR_alpha -> E[utility]` as
   `alpha -> 1`, checked at `alpha = 0.999` against `f1`, Phase 4's own already-oracle-tested
   `_build_objective` expected-value `Objective` -- compiled onto the identical `Problem` as a
   sweep of `RiskMeasure`s at different `alpha`s over a no-decision diagram (so `mu` is uniquely
   pinned regardless of which objective is "solved for", isolating the mathematical property from
   any policy-search confound) and read off *one* solve, a real regression-style cross-check
   between two independently-built code paths in this module, not mere self-consistency, checked
   against a mathematically-derived tolerance (`(1 - alpha) * (max(U) - min(U))`, the largest
   amount excluding at most `1 - alpha` tail mass could move the average by), not an arbitrary
   fudge factor; `CVaR_alpha <= E[utility]` for every swept `alpha`; and `CVaR_alpha` monotone
   non-decreasing in `alpha` across the same sweep.

**No real bugs found by building these oracle tests this phase, unlike Phase 4's fuzz-sweep
findings -- worth recording precisely because it's notable, the same way Phase 5 did.** Every test
above passed on its first run once written; the paper's equations were transcribed directly from
the verified PDF image (Table 1's own erratum aside, which never entered the implementation) and
composed with Phase 4/5's existing, already-oracle-tested marginalisation machinery rather than
new indexing arithmetic, which is plausibly why -- Phase 4's own real bug (the all-zero-utility
`func = [0.0]` case) and Phase 5's two real bugs (the silently-mislabelled-nonlinear constraint,
the undeclared-node-reference leak) both lived in genuinely new substitution/indexing code that had
no earlier oracle coverage; Phase 6's new code reuses `_group_flat_indices_by_projection`/
`_extract_expr` as data (merged by utility value) rather than reimplementing anything at that
layer.

**Tests.** `tests/test_graphical_compile.py` grew from 23 to 32 tests (9 new -- 6 standalone plus a
3-way `pytest.mark.parametrize` on the property-based case -- all marked both `graphical` and
`gurobipy`): the cases listed above. All 32 tests pass
(`pytest -m "graphical and gurobipy" tests/test_graphical_compile.py`). The full `graphical`-marked
suite (`pytest -m graphical tests/`) passes with no regressions from this phase's changes.

### Phase 7 — Web-API

- [x] Session-scoped inference engine cache `{session_id: (model_hash, engine)}` — compilation
      is expensive, evidence propagation is cheap, never resend the model. Define eviction
      (LRU/TTL). An in-process dict only works single-worker: decide explicitly whether that's
      an accepted v1 limitation (document it) or whether multi-worker deployment needs sticky
      sessions or an external cache before this ships
- [x] `POST /graphical/whatif` — evidence set in, all posteriors and utilities out in one payload
- [x] `POST /graphical/compile` — network + annotation → `Problem` id
- [x] `GET /graphical/{id}/structure` — DAG and node types only, no inference (serves UI phase A)
- [x] Own router, own schemas, no threading through existing code paths — the web-API is
      volatile and this keeps upstream churn to rebases rather than rewrites

**Findings.** `desdeo/api/routers/graphical.py` adds the three endpoints above, mounted at
`/graphical` and registered in `api/app.py`. `/compile` builds the RJT + compiles via
`compile_diagram`, persisting `ProblemDB` + `ProblemMetaDataDB` + `GraphicalModelMetaData` (all
four JSON columns, including `index_map_json`) in one atomic commit. `/structure` reads only
`network_json`/`annotation_json` — no pyAgrum, no inference, matching the checklist's own "serves
UI phase A" note. `/whatif` is the one genuinely new piece of machinery: a process-local,
session-keyed cache of live pyAgrum inference engines (`LazyPropagation` for a no-decision
network, `ShaferShenoyLIMIDInference` for a diagram with decisions), keyed on `session_id` when
given (falling back to a synthetic per-user-per-problem key otherwise), invalidated by a content
hash of `(network_json, annotation_json)` so an edited/recompiled model is never silently served
stale inference. Evidence itself is **never** cached — every call's full evidence dict is set
fresh via `setEvidence` (confirmed empirically to *replace*, not merge, the engine's evidence
state), so cached-engine reuse can't leak one call's evidence into another's. TTL (30 min idle)
and LRU (128 entries) eviction run opportunistically on each cache access; the module docstring
documents, per the plan's own risk-table item, that this is accepted as a single-worker-only
limitation (a live pyAgrum engine is a SWIG-wrapped C++ object, not something serialisable into an
external cache without solving a separate problem this phase doesn't attempt) — not solved here.

To formalise the pyAgrum side of `/whatif`, `desdeo/problem/graphical/adapters/pyagrum.py` gained
a new public function, `to_pyagrum_influence_diagram(network, annotation) -> gum.InfluenceDiagram`,
lifted from what had been a test-private helper in `tests/test_graphical_compile.py`'s own pyAgrum
oracle (`_pyagrum_meu`) since Phase 4 — the same proven construction, now shared by both the test
oracle and the live endpoint rather than duplicated.

**Code review pass (before merging).** A deep-dive review found two real, confirmed correctness
bugs plus a cluster of robustness/efficiency/test-quality issues, all fixed:

- **Correctness (most severe): a minimizing `ObjectiveNode` silently got its *worst* value
  reported as optimal.** `to_pyagrum_influence_diagram` wrote a utility node's table verbatim, but
  pyAgrum's own `MEU()`/policy search always *maximizes* — and `ObjectiveNode.maximize` defaults
  to `False` (a cost). Every existing pyAgrum-oracle test in `test_graphical_compile.py` happens to
  use `maximize=True`, so this was untested territory the moment `/whatif` started relying on the
  adapter for real (not just oracle-checking a specific already-known-correct case). Fixed the same
  way `compile.py`'s own `_build_objective` already handles this: the utility table is negated
  internally when `objective.maximize` is `False` (`sign = 1.0 if objective.maximize else -1.0`),
  and every reader of a utility-node-derived scalar (`/whatif`'s `_compute_utilities`,
  `test_graphical_compile.py`'s `_pyagrum_meu`) negates the same way before reporting a value in
  the objective's own units. Regression test:
  `test_whatif_minimizing_objective_with_decision_picks_and_reports_the_minimum`.
- **Correctness: `policy` reported a marginal's argmax as if it were a deterministic choice.**
  `_read_policy` used to take `argmax(posterior(decision))` unconditionally; that's only exact once
  the decision's effective information set is fully pinned by evidence — otherwise it's an
  evidence-induced *mixture* over still-uncertain choices, and reporting its argmax claims a
  confident single answer no actual policy produces. Fixed: a decision is only included in
  `policy` when its posterior is genuinely one-hot; a decision left open by partial/no evidence is
  simply absent rather than a most-likely-choice guess. Regression test:
  `test_whatif_policy_omits_decision_not_fully_determined_by_evidence`.
- **Correctness/robustness: `max_cluster_size`/`on_exceed` was checked only on a freshly-built
  tree, never on a cache hit.** `build_rooted_junction_tree`'s process-global, structure-keyed
  cache returned early before the size check ran, so a `on_exceed="warn"` call that warmed the
  cache silently disabled a *later, stricter* `on_exceed="raise"` call for the same structure —
  including across different users/problems, since the cache is process-global. Fixed by factoring
  the check into `_check_max_cluster_size` and running it on both the cache-hit and freshly-built
  paths (`RootedJunctionTree.max_cluster_size` is a property of the returned tree, so re-checking
  it on a hit is free — no re-triangulation needed). `/compile` additionally now clamps a
  caller-supplied `max_cluster_size` to a server-side ceiling (`_SERVER_MAX_CLUSTER_SIZE`,
  `200_000`, from Phase 4's own docstring note on where cost starts to matter) forced to
  `on_exceed="raise"`, since `/compile` runs synchronously and an absent or overly generous
  client-supplied cap would let one request block a worker indefinitely.
- **Robustness: `/compile`'s three separate commits could leave a permanently-orphaned problem.**
  `ProblemDB`, then `ProblemMetaDataDB`, then `GraphicalModelMetaData` were committed as three
  separate transactions; a failure on the last one left a durably-persisted problem with no
  graphical metadata and no repair path (`/structure`/`/whatif` 404 on it forever). Fixed: build
  all three with `db_session.flush()` between them (to obtain each row's id for the next row's FK
  without ending the transaction) and commit once at the end, so a failure anywhere rolls
  everything back together.
- **Robustness: `copy_problem_metadata` (pre-existing, `routers/utils.py`) silently dropped
  `GraphicalModelMetaData` on copy.** Phase 3 added `graphical_model_metadata` to
  `ProblemMetaDataDB`/`all_metadata` but never to this helper's copy loop — any flow going through
  it (e.g. Cumulus's problem-modification path) lost a compiled diagram's graphical metadata
  silently. Fixed by adding the missing loop, matching the existing per-kind pattern.
- **Correctness (latent): `_fetch_graphical_metadata`'s "most recently added" wasn't ordered by
  anything.** Picked `all_metadata`'s last element, but `all_metadata` concatenates plain,
  `order_by`-less `Relationship` lists — nothing today ever inserts a second
  `GraphicalModelMetaData` row for the same problem, so this was latent, not live, but the
  docstring asserted a guarantee the code didn't provide. Fixed by querying
  `GraphicalModelMetaData` directly, joined through `ProblemMetaDataDB`, explicitly
  `order_by(GraphicalModelMetaData.id.desc())`.
- **Correctness (low likelihood): `/structure` could emit two nodes with the same id.** Nothing
  forbids an `ObjectiveNode.symbol` from colliding with a chance-node name (a Phase 5 finding,
  deliberately not tightened at the schema level — an existing regression test in
  `test_graphical_compile.py` exercises that exact collision at the constraint-substitution layer
  instead). Fixed locally in `/structure`: objective node ids are namespaced `util_<symbol>`, the
  same prefix `to_pyagrum_influence_diagram` already uses internally, so this endpoint's ids can
  never collide across kinds even though the underlying symbols could. Edges also gained a stable
  `id` field (`f"{source}->{target}"`) — needed by the id-keyed reconciliation the webui's
  `<NetworkView>` (Phase B) already does.
- **Efficiency: the engine cache's own lock serialised every concurrent `/whatif` request in the
  process.** The per-entry lock's own docstring claimed unrelated sessions run concurrently, but
  `_get_or_build_entry` built a fresh engine *inside* the single process-global `_cache_lock` — a
  slow build (compiling a pyAgrum diagram) stalled every other request, cache hits included. Fixed
  with double-checked construction: look up under the lock, release, build outside it, re-acquire
  and insert (discarding the loser's build on a race) — two concurrent misses for the same key now
  duplicate cheap work instead of one stalling behind the other.
- **Efficiency: the content-hash fingerprint re-serialised data that was already a string.**
  `_get_or_build_entry` used to call `model_dump_json()` on the just-parsed `network`/`annotation`
  purely to hash them, a full parse-then-reserialise round trip of the entire network (every CPT
  included) on the hot path. Fixed by hashing `meta.network_json`/`meta.annotation_json` directly —
  the raw strings the caller already had in hand.
- **Simplification:** a dead `if not objective.parents: return float(...)` branch in
  `_expected_value_no_decision`, inherited from the test helper it was generalised from, was
  removed — `ObjectiveNode`'s own validators make `parents == []` require `utility.shape == []`,
  which `Potential`'s shape validator can never satisfy, the identical dead branch Phase 4's own
  review pass already found and deleted from `compile.py`.
- **Test quality.** The single-objective-with-decision oracle test used to check *only* against a
  `ShaferShenoyLIMIDInference` oracle built the same way the endpoint itself builds one — unable to
  catch a bug (like the sign bug above) shared by both sides. Added a hand-computed literal
  expectation alongside the existing pyAgrum cross-check. Also added: a maximize=False regression
  test, a policy-omitted-when-undetermined regression test, a content-hash cache-invalidation test
  (previously entirely untested — the one property the cache exists to get right), a
  multi-objective-with-decision 501 test, and round-trip assertions (not just presence checks) on
  `/compile`'s persisted `rjt_json`/`index_map_json` against freshly-built equivalents. `just
  test-api` (193 tests, up from 189) and the full `graphical`-marked suite (`pytest -m graphical`,
  both with and without `gurobipy`) pass with no regressions.

**Deliberately not fixed in this pass, flagged rather than silently left implicit:**

- **Multi-objective expected utility once at least one decision exists is an open problem,
  answered here with an honest `HTTPException(501)`, not a guess.** Confirmed empirically that
  `ShaferShenoyLIMIDInference` has no `addJointTarget`/`jointPosterior` at all (only single-node
  `posterior`/`posteriorUtility`), and Phase 2/4 already established `MEU()` sums every utility
  node into one scalar while `posteriorUtility` doesn't collapse to a per-node scalar either — so
  there is no verified way to recover one objective's own share once a diagram has *both* more
  than one `ObjectiveNode` and at least one decision. Exactly one objective (with or without
  decisions) and any number of objectives with zero decisions are both fully implemented and
  oracle-tested; the combination is rejected with a clear message pointing at this finding, the
  same scope-cut style Phase 5 used for `DiagramConstraint`'s `kind == "tail"`.
- **The wire shapes here don't yet match webui's Phase B `<NetworkPanel>` placeholder client
  exactly** — `WhatIfRequest` uses `problem_id` where `NetworkPanel.svelte`'s `defaultFetchWhatIf`
  posts `model_id`; `/structure`'s edges now carry a stable `id` (fixed above) but nothing yet
  adapts the response into the exact `NetworkModel` shape `types.ts` expects; and `/whatif` rejects
  a decision-symbol evidence key outright (manual-mode decision toggling, per Phase B's
  `isNodeEditable`, isn't wired to any backend semantics here). This is expected, not an oversight:
  Phase B's own findings already anticipated this exact gap ("it will 404 until phase 7 lands, by
  design; swapping in the generated OpenAPI client at that point is a prop change, not a rewrite").
  Wiring the two together is Phase C/D's job, not this one's — recorded here so it isn't
  rediscovered as a surprise then.
- **`no_forgetting` gating through the live endpoint, and the TTL/LRU eviction paths, have no
  dedicated test in `test_graphical_router.py`.** Both were checked by the review pass by reading
  the code directly (confirmed correct — `no_forgetting`, not `decision_order`, gates
  `addNoForgettingAssumption`, matching Phase 4's own already-hardened finding; eviction runs on
  every access and does bound the cache) rather than by a new test, a deliberate time-tradeoff
  given both are thin passthroughs of already-oracle-tested (Phase 4) or already-reviewed logic.
  Worth adding if this endpoint sees substantial further change.

### Later — parameter ambiguity layer

Deferred, but the shape is known: `ScenarioModel` whose leaves are alternative
parameterizations (expert disagreement, posterior uncertainty over learned CPTs), policy
binaries pinned across leaves via `anticipation_stop` — which solvers already honour.

Flag when building: `leaf_scenarios` falls back to uniform weights when
`scenario_probabilities` is empty, which silently asserts a uniform second-order prior over
experts. Make it explicit either way.

---

## Frontend

### Phase A — Structure-only component (no backend)

Ships standalone and already delivers the "understand the dependency structure" half.

- [x] `<NetworkView>` on Svelte Flow (`@xyflow/svelte`), custom nodes per type
      (chance / decision / objective / deterministic)
- [x] Layout via elkjs, computed **once** and persisted with the model; "re-layout" is an
      explicit button. Re-running layout on every mount destroys spatial memory across method UIs.
- [x] Level of detail driven by container width *and* zoom — labels only at ~320px, bars at
      panel width, full detail expanded. `compact` prop plus resize observer.
- [x] libdagitty client-side: ancestor sets, d-separation, active trail highlighting
- [x] Table view alongside the canvas — beyond a dozen nodes, the table is what people read
- [x] Nail a minimal JSON contract for `model` (nodes, types, edges) before writing components,
      independent of the eventual pydantic/OpenAPI shape — phase A ships before phase 1's schema
      is final, so this avoids reworking the component once the real types and generated
      OpenAPI client land

**Findings.** Package lives at `webui/src/lib/components/network/` (the path the plan's own
phase B item names), exporting only from `index.ts`. Added `@xyflow/svelte` (1.6.3) and `elkjs`
(0.12.0) as real dependencies — `npm install` needed Node ≥21.7.3 (`@cortex-js/compute-engine`'s
engine range), which this checkout's default Node (v20.20.2) doesn't satisfy; used nvm's
v24.15.0 (already installed locally) to run `npm install`/`vitest`/`svelte-check`. `package.json`/
`package-lock.json` diffs are otherwise clean (just the two new deps).

There is no `libdagitty`/`dagitty`/`js-dagitty` npm package — dagitty.js was never published to
npm, only distributed as part of the R `dagitty` package via V8 — so "libdagitty client-side"
is implemented directly as plain TypeScript graph algorithms in `dag-algorithms.ts`, not a
wrapped dependency: `ancestors`/`descendants` (closure via parent/child adjacency), `dSeparated`
(moralize-the-ancestral-graph construction), and `activeTrailNodes` (the Koller & Friedman
"Reachable" / Bayes-Ball algorithm, used for hover/select highlighting). All three are pure
functions over `{nodes, edges}` only — no inference, no backend — which is what makes them usable
in phase A. Covered by `dag-algorithms.test.ts` (9 tests: chain/fork/collider d-separation,
active-trail propagation and blocking, collider-opened-by-evidence-on-a-descendant). Hooked up
in `NetworkView` as a free slice of phase C: hovering or selecting a node highlights its ancestor
set now, ahead of schedule — full evidence-aware active-trail highlighting still needs phase D's
what-if state, so `HighlightState`/`activeTrailNodes` are ready for that but not wired to evidence
yet.

The JSON contract (`types.ts`: `NetworkModel` = `{nodes, edges, layout?}`, `NetworkNodeModel` =
`{id, label, kind, states?, description?}`) is deliberately thin and carries no positions inside
node data — positions live only in `NetworkModel.layout.positions` (node id → `{x,y}`), computed
by elk's layered algorithm (`layout.ts`) and written back onto the `$bindable model` prop, once on
mount if `layout` doesn't already cover every node, or on explicit "Re-layout". `NetworkView`
reconciles Svelte-Flow-facing node/edge arrays from `model` in one `$effect`, preserving existing
(including manually dragged) positions via an id-keyed map rather than rebuilding from `layout` on
every reactive tick — `layout.ts` and the reconciliation effect are independent, so re-layout is
the only path that overwrites positions wholesale, matching the "computed once, explicit button"
requirement.

LOD (`lod.ts`, `computeDetailLevel`) takes the more conservative of a width tier (compact <320px,
normal <560px, expanded above) and a zoom tier (compact <0.55, normal <0.85, expanded above), or
is pinned to `'compact'` by the `compact` prop — both a wide-but-zoomed-out canvas and a
narrow-but-zoomed-in one degrade gracefully instead of only checking one dimension. Canvas and
table are both permanently mounted (visibility/flex-sizing toggled by a `Canvas / Split / Table`
segmented control, not conditionally rendered) specifically so `containerEl`'s `ResizeObserver`
and the Svelte Flow viewport survive mode switching — an earlier version that conditionally
rendered per mode was tearing down and recreating both PoV state on every toggle.

Verified: `dag-algorithms.test.ts` (9/9 pass, vitest). `svelte-check` reports zero errors/warnings
in every new file (network/, nodes/, network-view.stories.svelte) against a baseline that already
has ~66 pre-existing errors elsewhere in the webui tree, unrelated to this change. Could **not**
verify against `npm run lint`/`just lint`: `webui/eslint.config.js` currently throws
`ReferenceError: ts is not defined` (missing `typescript-eslint` import — pre-existing, unrelated
to this branch, reproduces on `master` too) and `prettier --check` crashes on **every** `.svelte`
file in the tree, including ones this change never touched (`TypeError: getVisitorKeys is not a
function`, from `prettier`@3.8.0 vs `prettier-plugin-svelte`@3.4.1 in the committed lockfile) —
both reproduce identically under Node v20.20.2 and v24.15.0, so neither is a version artifact of
this session. Worth a maintainer heads-up; new files were hand-matched to the repo's `.prettierrc`
(tabs, single quotes, no trailing commas, 100 cols) and to neighbouring components' style instead.

Demo data: `sample-network.ts` (a small irrigation influence diagram exercising all four node
kinds) backs `webui/src/stories/network-view.stories.svelte` (Default / Compact panel / No
persisted layout, the last exercising the auto-layout-on-mount path) — not exported from the
package's `index.ts` since it's a fixture, not part of the public surface.

### Phase B — Embedding contract

- [x] Props: `model`, `evidence` (`$bindable`), `posteriors`, `policy`, `highlight`,
      `compact`, `decisionMode: 'solver' | 'manual'` (governs whether decision nodes are
      read-only outputs of a solved policy or directly manipulable — see phase E)
- [x] Events: `evidenceChange`, `nodeSelect`, `nodeHover`
- [x] Snippet prop for node chrome so a method UI can inject a badge without forking
- [x] Thin `<NetworkPanel>` wrapper that owns fetching, for the standalone case only
- [x] Own workspace package (or `$lib/components/network/`) with **no imports from any
      method store** — that constraint is what actually keeps it reusable
- [x] Integrate into Cumulus end to end before generalising (see design decisions)

**Findings.** All additions land in the existing `webui/src/lib/components/custom/network/`
package (phase A's actual location — the plan's own phase B item text names
`$lib/components/network/`, which doesn't exist; phase A's findings already recorded the real
path and this continues from there).

**No backend to integrate against yet — resolved by the plan's own sequencing note, not by
asking.** Backend phases 4-7 are unbuilt, so there is no compiled diagram, no evidence
propagation, and no `/graphical/*` endpoints to call. The "Sequencing" section already answers
what this means for phase B: *"Getting the embedding contract right against a real method UI
(phase B) before the inference plumbing exists is the point of ordering the UI this way."*
Concretely: the component-library contract (props/events/snippet/`NetworkPanel`) is built and
tested independent of any backend, and the Cumulus integration wires `<NetworkView>` into a real
method UI's actual chrome (a dropdown-triggered dialog, `SegmentedControl`, `Dialog`) using a
sample diagram and fixed demo posteriors/policy — clearly labelled "(demo)" in the UI and in
comments — standing in for what a compiled diagram's structure/whatif response will supply once
phase 7 ships. `<NetworkPanel>` itself (the piece that *does* need real endpoints) is written
against the plan's committed endpoint shapes (`GET /graphical/{id}/structure`,
`POST /graphical/whatif`, via the `/api/...` proxy) with `fetchStructure`/`fetchWhatIf` as
injectable functions defaulting to those paths — it will 404 until phase 7 lands, by design;
swapping in the generated OpenAPI client at that point is a prop change, not a rewrite.

**Wire shapes added to `types.ts`:** `DecisionMode` ('solver' | 'manual'), `NetworkEvidence`
(node id → pinned state label — hard evidence only; soft evidence and an explicit
observe-vs-intervene mode are phase D's job per the plan), `NetworkPosteriors` (node id →
per-state probability array aligned with `states`), `NetworkPolicy` (decision node id → chosen
state), and `WhatIfResult` (the `POST /graphical/whatif` response shape `NetworkPanel` expects).

**Node interactivity, factored out as pure functions for the same reason `dag-algorithms.ts` is
pure** (this repo has no component-testing setup — see `dag-algorithms.test.ts`'s precedent):
`interaction.ts` adds `isNodeEditable(kind, decisionMode)` (chance nodes always; decision nodes
only in `'manual'`; objective/deterministic never — a deterministic node's value follows from its
parents rather than being observed or chosen, so it's excluded on the same logic as objectives),
`toggleEvidenceState` (immutable set/clear on click), and `mergeHighlightSets` (unions the
existing hover/select ancestor-closure highlight with the new externally supplied `highlight`
prop — ahead of phase C, which will be the first real caller of external highlighting).
`interaction.test.ts` covers all three (13 tests).

**`GraphNode.svelte`** renders the new data: an evidence pin badge (corner overlay, any detail
level — pinned evidence is important enough to survive `compact`), a policy badge with a check
icon for decision nodes in `'solver'` mode, and per-state chips that become clickable buttons
when `editable` (click stops propagation so it doesn't also toggle node selection). Revisiting the
lod.ts's original design note — *"labels only at ~320px, bars at panel width, full detail
expanded"* — the `'normal'` tier's "bars" was written before posteriors existed to show; it now
renders a compact unlabelled bar row (segment width = probability, `title` attr carries the
label) when `posteriors` is supplied, falling back to the old "N states" text otherwise, and
`'expanded'` renders full labelled chips with an inset probability fill. Both tiers' bars are
static, non-animated — bar-transition animation is explicitly phase D's job.

**`nodeChrome` snippet** is threaded through `NetworkNodeData.chrome` (one per Svelte-Flow node's
`data`, since xyflow instantiates custom node types by `type` alone with no way to pass an
extra prop through the type map) rather than rendered by `NetworkView` itself — `GraphNode` calls
`{@render nodeData.chrome({ id, kind, label })}` in a corner overlay. Verified against the
Cumulus integration, which is the first real, non-trivial consumer: a per-node flag button
(`toggle_network_flag`) that is entirely Cumulus-owned state, requiring zero changes to
`NetworkView`/`GraphNode` to add.

**`NetworkPanel.svelte`** deliberately does not import `stores/uiState` (or anything else
method-specific) for its loading/error state — local `$state` instead — which is what actually
keeps the "no imports from any method store" property true for the fetching wrapper, not just
the display component. No debounce/cancellation/staleness-dimming (phase D's job, layered above
the fetch this component owns).

**Cumulus integration**: a new "View network structure (demo)" item in the existing
hamburger-menu `DropdownMenu` opens a `Dialog` containing `<NetworkView>` bound to
Cumulus-page-local `network_model`/`network_evidence` state, a `SegmentedControl` toggling
`decisionMode`, fixed demo `posteriors`/`policy` constants, and the flag-button `nodeChrome`
described above. `network_decision_mode_str` is kept as a plain `string` (not `DecisionMode`)
for the `SegmentedControl` binding, mirroring `NetworkView`'s own `viewMode` workaround for the
same bindable-prop-type mismatch, with a `$derived` cast at the read site.

**Verification.** `npx vitest run` (Node v24.15.0 via nvm, per phase A's findings): 38/38 tests
pass across all 4 suites (`dag-algorithms.test.ts` 9, `interaction.test.ts` 13, plus the
pre-existing 2 non-network suites). `npx svelte-check`: 66 errors / 13 warnings, all in files this
change never touched — identical count to phase A's recorded baseline, confirming zero new errors
introduced by any of `types.ts`, `view-types.ts`, `interaction.ts`, `NetworkView.svelte`,
`GraphNode.svelte`, `NetworkPanel.svelte`, `index.ts`, the story file, or the Cumulus route.
`npm run lint`/`prettier --check` still cannot run at all, for the same pre-existing,
branch-independent reasons phase A recorded (`eslint.config.js`'s missing `typescript-eslint`
import, `prettier`/`prettier-plugin-svelte` version mismatch) — new/changed code was hand-matched
to the repo's `.prettierrc` and neighbouring style instead, as phase A also did.

**End-to-end verification against the running app** (`webui-testing` skill; analyst login,
`db_init_summer_cabin.py`-seeded DB, Simple knapsack problem) found one real gap and, along the
way, two pre-existing bugs unrelated to phase B that were fixed since they blocked testing
entirely:

- **The gap**: at `'normal'` detail, an editable node (chance always, or a decision in `'manual'`
  mode) with no `posteriors` entry fell back to the same plain, non-interactive "N states" text
  used for genuinely non-editable nodes — silently losing the click affordance below `'expanded'`
  detail purely because there was no probability to draw a bar with. Fixed in `GraphNode.svelte`:
  that fallback now renders an equal-width segment row (still clickable, no probability fill)
  whenever the node is editable, reserving plain text for the actually-non-interactive case
  (objective/deterministic nodes, or a decision in `'solver'` mode). Confirmed live: the demo's
  "Irrigate" decision node was exactly this case (no posterior — it's a decision, not a chance
  node) and was unclickable in Manual mode before this fix, clickable after.
- **Bug 1 (severe, pre-existing, unrelated to this branch's UI work)**: hard/direct navigation to
  *any* route whose `+page.ts` calls a generated API function 500'd, always, regardless of login
  state — `/problems`, `/interactive_methods/CUMULUS`, and 8 other routes (`groups`,
  `methods/initialize`, `XNIMBUS`, `NIMBUS`, `EMO`, `SCORE-bands`, `GNIMBUS`,
  `GDM-SCORE-bands`) all shared the pattern. Root cause: `customFetch` (`lib/api/new-client.ts`)
  only routes through SvelteKit's own `fetch` — the one `hooks.server.ts`'s `handleFetch` actually
  intercepts to attach the `access_token` cookie — when a caller passes `fetchImpl: fetch` from its
  own `load`. None of these ten loaders did; they called the generated client bare, so on a hard
  navigation (no browser-side hydration yet to fall back on) the SSR call went out via plain global
  `fetch` with zero cookies, 401'd, and the loader's `throw new Error(...)` surfaced as a 500. Only
  ever masked because normal usage always navigates client-side after the first login redirect,
  where `getUrl`'s browser branch (`/api/...` through the proxy) carries real browser cookies
  regardless. Fixed by threading `{ fetch }` from each loader into `{ fetchImpl: fetch }` on every
  generated-client call — the exact mechanism `handlers.ts`'s `refreshAccessTokenRefreshPost` call
  already used, just never applied to these ten `load` functions.
- **Bug 2 (surfaced by fixing bug 1)**: routing through SvelteKit's fetch immediately hit a second,
  independent issue — SvelteKit strips response headers from a `load`-function `fetch()` call
  before serializing it for client hydration unless explicitly allow-listed via
  `filterSerializedResponseHeaders`, and `hooks.server.ts` had no `handle` hook defining one.
  `getBody` (`new-client.ts`) reads `content-type` on every response, so every affected route
  traded a 401-turned-500 for a "Failed to get response header 'content-type'"-turned-500. Fixed by
  adding `export const handle: Handle` to `hooks.server.ts`, allow-listing `content-type` (the only
  header any application code reads off these responses — confirmed by grep).
- **Also found, not fixed (out of scope for this pass)**: a `[svelte] hydration_mismatch` console
  warning on every hard navigation to a page reading `methodSelection` (localStorage-backed) during
  render — SSR always sees the store's pre-hydration default, so a previously selected
  problem/method causes a client/server markup mismatch right after hydration. Doesn't break
  anything observed (the page still renders and functions correctly afterward) but is a real,
  separate bug worth a future pass; flagging it here rather than fixing it silently since it's
  unrelated to phase B's actual scope.

With both bugs fixed, the demo dialog was driven live end-to-end: opened from CUMULUS's hamburger
menu, Manual mode's per-state chips toggle hard evidence on/off (verified the pin badge and
highlighted segment appear and clear correctly), Solver mode shows the fixed demo policy as a
check-badge on "Irrigate", and the `nodeChrome`-injected flag button toggles Cumulus-owned
`network_flagged` state and updates the dialog's "Flagged for follow-up" text — confirming the
snippet-injection contract works against a real method UI, not just Storybook. No console errors
or failed network requests attributable to any of this; the only console warning is the
pre-existing hydration mismatch noted above.

**Post-hoc `/code-review` pass (8 parallel angle-agents), 14 findings, all addressed:**

- Two state-staleness bugs in `NetworkPanel`: evidence/posteriors/policy weren't reset on a
  `modelId` change (a shared node id across models could show a leftover badge/bars/checkmark),
  and a what-if response with no `policy` field kept the previous one forever via `?? policy`
  instead of clearing it. Fixed by resetting all three at the start of `loadStructure`, and by
  making `refreshWhatIf` treat `result.policy` as authoritative including when `undefined`.
- `mergeHighlightSets` collapsed an explicit empty external highlight (`highlight={[]}`) to
  `null`, contradicting its own documented null-vs-empty-but-active contract — a real trap for
  phase C, the feature this function exists for. Fixed to only treat `null`/`undefined` as
  "absent"; tests updated to assert the corrected contract.
- The split-view Table only ever saw the internal hover/select id, never the externally-merged
  `highlightedIds`, so an externally-driven highlight would show on the canvas but not the table.
  Fixed by changing `NetworkTable`'s prop from a single `highlightedId` to a `highlightedIds` set.
- A decision node's read-only policy checkmark could render simultaneously with its editable
  click targets if `decisionMode` switched to `'manual'` while a stale `policy` was still set —
  `policyState` is now gated on `decisionMode === 'solver'` at the point `NetworkView` builds each
  node's data, rather than trusting the caller never to pass a mode/policy combination that
  shouldn't coexist.
- `hooks.server.ts`'s `handleFetch` had no de-duplication for concurrent 401 refreshes, newly
  reachable now that ~10 loaders route through it; two calls 401ing at once could each
  independently (and, if the backend rotates refresh tokens on use, destructively) call
  `refreshAccessTokenRefreshPost`. Fixed with a `WeakMap<RequestEvent, Promise<...>>` so concurrent
  calls on the same request await one shared refresh.
- `NetworkPanel`'s structure fetch had no in-flight-request guard: a fast `modelId` change could
  let an older, slower response resolve after a newer one and clobber it. Fixed with a
  request-generation counter that discards superseded responses.
- `groups/+page.ts` fetched each group's (then each problem's) info in a sequential `for`-loop
  despite each iteration being independent — switched both to `Promise.all`.
- The reconciliation effect rebuilt every Svelte-Flow node (including a fresh `onStateClick`
  closure each) on every hover/select, not just structural changes. Split into a structural effect
  (nodes/edges/detail/evidence/policy/posteriors/mode/chrome) and a separate lightweight effect
  that only patches each node's `highlight` field, leaving closures and other fields untouched.
- The `{ fetchImpl: fetch } as RequestInit` fix (plus its explanatory comment) was duplicated
  verbatim across all ~10 loaders. Centralized into `$lib/api/ssr-fetch.ts`'s `withFetch(fetch)`,
  unit-tested, with the one type-unsafe cast now living in one place instead of ten.
- `isPinned` was copy-pasted inline three times in `GraphNode.svelte`. Extracted as
  `isStatePinned` in `interaction.ts`, unit-tested alongside the module's other pure helpers.
- `CUMULUS/+page.svelte`'s `toggle_network_flag` reimplemented the exact "toggle Set membership"
  logic already in `problems/+page.svelte` as a page-local `toggleSet`. Both now import a single
  generic `toggleSet<T>` from `$lib/helpers`.
- Several new lines exceeded the project's printWidth:100 — fixed by extracting per-state class
  strings into small named functions (which also gave `isStatePinned` a natural call site),
  pulling a fully-static class string to a module constant, and wrapping two long
  comment/attribute lines.
- Scope/altitude concern: the CUMULUS demo menu item was wired into the real production dropdown
  with no gate, so a real user could open a fabricated network diagram styled identically to the
  three genuine actions beside it. Wrapped in `{#if dev}` (`$app/environment`) so it only exists in
  local/dev builds, preserving its value for exercising the embedding contract against a real
  method UI (the reason it's `<NetworkView>` + demo data rather than Storybook-only) without that
  production-exposure risk.

Verified after all fixes: 46/46 vitest tests (up from 38 — new coverage for `isStatePinned`, the
corrected `mergeHighlightSets` contract, `withFetch`, and the concurrent-refresh dedup), and
`svelte-check` still exactly at the 66-error/13-warning pre-existing baseline.

### Phase C — Cross-highlighting

Cheapest high-value feature in the whole plan. Hover objective *f₂* in the reference point
widget → its ancestor subgraph lights up, everything else greys out. Hover a chance node →
the objectives it can influence highlight in the panel.

- [x] Shared highlight store keyed on node ids only
- [x] Ancestor sets computed client-side (already available from phase A)

Objective nodes show **no distributions** — their values live in the existing objective
components. Their job in the graph is purely structural.

**Findings.** The second checklist item needed no new code — `ancestors`/`descendants`/
`ancestralSet` (`dag-algorithms.ts`) already existed from phase A and were already exported from
`index.ts`; this phase only reuses them. Objective-node distribution-suppression was likewise
already true by construction (`NetworkNodeModel.states` is absent for `objective` nodes per
`types.ts`'s own doc comment, so `GraphNode.svelte`'s posterior-bar block, gated on
`states?.length`, never renders for one) — nothing to add there either.

The actual new piece: `webui/src/lib/components/custom/network/highlight-store.ts` adds
`createNetworkHighlightStore()`, a **factory** (not a module-level singleton) returning a
`Writable<Set<string> | null>` — a generic, model-agnostic, node-id-keyed cross-highlight bus a
method UI creates once per embedded diagram and shares between `<NetworkView highlight={$store}>`
(the externally-driven `highlight` prop phase B already built) and its own widgets. Factory, not
singleton, for the same reason `network_model`/`network_evidence` in the Cumulus demo are
page-local `$state` rather than global stores — a singleton would leak stale highlight state
across independent diagram instances/navigations. `set`/`update` defensively copy into a fresh
`Set` before writing, so an immutability-contract violation (a caller calling `.add`/`.delete` on
a value read out of the store, which would otherwise be a silent no-op to every subscriber, since
`$derived`'s reference-equality check would never see it) can't corrupt shared state. Exported
alongside a `NetworkHighlightStore` type alias from `index.ts`, along with `buildAdjacency`/
`Adjacency` (previously internal-only, needed by any external caller wanting to avoid an O(V+E)
rebuild per hover event — see below). `highlight-store.test.ts` (6 tests) covers initial value,
set/subscribe, independent instances not sharing state, subscription isolation (a `vi.fn()`
subscriber on one store is never notified by another store's `.set()`), unsubscribe actually
stopping notifications, and the defensive-copy behavior.

Wired end-to-end (not just Storybook) into the existing dev-only Cumulus network demo dialog
(phase B's `{#if dev}`-gated "View network structure (demo)" item), proving both directions of
this phase's own worked example against a real method UI: a small "(demo) Reference point" panel
lists the sample diagram's two `objective` nodes as a stand-in for a real reference-point widget;
hovering/focusing a row sets the shared store to that node's ancestor closure (`ancestralSet`,
inclusive of the seed), which flows into `<NetworkView>`'s canvas via `highlight`. The reverse
direction — hovering *or selecting* a canvas node — mirrors `NetworkView`'s own
`hoveredId ?? selectedId` semantics locally (`network_focus_id`) and pushes the node's own id plus
its descendants (inclusive, matching the panel→canvas direction's inclusive convention so
self-highlighting agrees both ways) into the *same* shared store, while separately intersecting
that same set against the two objective ids to light up matching panel rows. Driving canvas→panel
and panel→canvas through one shared store (rather than two independently-computed highlight sets)
is what keeps the canvas and the panel from disagreeing about which objectives are relevant. An
`$effect` resets all of this (hovered/selected ids, the shared store, the panel's own highlight
set) when the dialog closes, since a dialog close (e.g. Escape) isn't guaranteed to fire
`mouseleave`/`blur` on whatever was focused, and the store's lifetime is page-scoped while the
diagram's is dialog-scoped. Panel rows are `<button>`s with `onfocus`/`onblur` mirroring their
`onmouseenter`/`onmouseleave`, so keyboard-only navigation gets the same highlighting a mouse does.

**Code review pass (before merging).** A deep-dive review of the initial implementation found 10
issues, all fixed: a stale-highlight-survives-dialog-reopen bug (finding 1 above, the `$effect`
reset); the reference-point panel reading objectives from the static `sampleIrrigationNetwork`
fixture instead of the live (potentially reassigned) `network_model`, now `$derived` off the
latter; canvas-hover and panel-hover computing two independently-derived, contradictory highlight
sets (canvas hover dimmed exactly the objectives the panel simultaneously lit up), resolved by
routing both directions through the one shared store described above; an inclusive/exclusive
mismatch between `ancestralSet` (seed-inclusive) and a plain `descendants` call (seed-exclusive)
that made self-highlighting asymmetric between the two directions, resolved by making the reverse
direction seed-inclusive too; `buildAdjacency`/`Adjacency` not being exported from `index.ts` even
though its own doc comment describes exactly the per-hover-event-rebuild anti-pattern a caller
needs it to avoid, and this phase being the first external caller — fixed by exporting both and
building the adjacency once per `network_model` change via `$derived`; the store's undocumented,
unenforced immutability contract (see above); a missing `NetworkHighlightStore` type alias, needed
the moment a real (non-demo) reference-point widget wants to type a `highlightStore` prop; the
"independent instances" test proving only value isolation, not subscription isolation (the
property that actually matters for a shared bus) — fixed with the `vi.fn()`-subscriber tests
described above; reference-point rows having no keyboard-accessible trigger; and canvas
selection not surviving hover-out in the panel's own highlighting the way it already does in
`NetworkView`'s internal highlighting, fixed by tracking hovered/selected ids separately and
deriving focus the same way `NetworkView` itself does.

**Tests.** `webui`'s vitest suite grew from 46 to 52 (6 new, all in `highlight-store.test.ts`).
`npx vitest run`: 52/52 pass. `npx svelte-check`: 66 errors / 13 warnings — exactly phase A/B's
recorded pre-existing baseline, confirmed none of the newly-touched files (`highlight-store.ts`,
`index.ts`, `CUMULUS/+page.svelte`) contribute any of them. `npm run lint`/`prettier --check`
still cannot run, for the same pre-existing, branch-independent reasons phases A and B both
recorded; new/changed code was hand-matched to `.prettierrc` and neighbouring style instead.

### Phase D — What-if

- [x] Prior (or committed scenario) as ghost mark behind each posterior bar, plus signed delta.
      Without a baseline a posterior conveys nothing.
- [x] Hard evidence, soft evidence (likelihood), and observe-vs-intervene as an explicit mode
      with distinct node marks. If only observation is supported, say so in the UI rather than
      letting people read causal claims into conditional ones.
- [x] Debounce ~150ms, cancel in-flight requests, dim the whole canvas while stale (partial
      updates read as bugs)
- [x] Animate bar transitions 200–300ms — motion is what communicates change. Never animate layout.
- [x] Explain nulls: when evidence changes nothing, show the d-separation rather than letting
      the user assume the tool is broken
- [x] Ghost/delta/highlight encoding is colour-blind-safe by default — pair colour with
      shape/position for "changed" vs "unchanged", never colour alone

**Exploratory vs committed evidence.** Exploratory updates objective components as a ghost
overlay without touching method state. Committed evidence changes the achievable ranges —
so ideal and nadir move, and any reference point or bound the DM already set was expressed
against the old ranges. Decide explicitly: recompute and warn that preferences were rescaled,
or hold ranges fixed and show clipping. Failing to choose is what produces the bug reports.

**Findings.** Started by reconciling `NetworkPanel`'s default fetchers against the real,
already-shipped `/graphical/whatif`/`/graphical/{id}/structure` endpoints (`desdeo/api/routers/
graphical.py`, backend phase 7), per this phase's brief — `defaultFetchStructure` turned out to
need no change at all (`StructureResponse`'s `nodes[].{id,label,kind,states}`/`edges[].{id,
source,target}` already match `NetworkModel` field-for-field); `defaultFetchWhatIf` had two real
bugs, both fixed: it posted `model_id` where the real `WhatIfRequest` expects `problem_id`, and it
silently dropped `WhatIfResponse.utilities` entirely, which `WhatIfResult`/`types.ts` didn't even
have a field for. Fixed by adding `NetworkUtilities` (`Record<string, number>`, node-id-keyed like
every other per-node map in this contract) to `WhatIfResult`, and by having `defaultFetchWhatIf`
remap `utilities`' keys from the real endpoint's raw objective *symbol* to the `util_<symbol>` node
id `get_structure` itself already uses for objective nodes — otherwise the two responses'
objective-keying conventions would silently disagree.

**Where utilities end up.** Not rendered inline on the objective node shape. Phase C already
decided objective nodes show *no distributions* — "their job in the graph is purely structural,"
their values live in a method's own objective-value UI — and a scalar expected utility is exactly
that kind of value, not a distribution, so extending `GraphNode` to draw it would undo that
decision for no good reason. Instead `NetworkPanel` tracks `utilities` as state (reset on
`modelId` change, same as `posteriors`/`policy`) and forwards every what-if response through a new
`onWhatIfResult` callback prop, so a host method UI that wants to show "expected profit given this
evidence" next to its own reference-point widget has a real, tested hook to read it from, instead
of the data being silently thrown away as it was before this phase. The Cumulus demo (no live
backend, no compiled diagram to call `/whatif` against) doesn't exercise this specific path live;
`NetworkPanel`'s `defaultFetchWhatIf` remap and `onWhatIfResult` wiring are verified by reading and
by `svelte-check`, not by a running demo — flagged rather than glossed over.

**Ghost/delta mechanism.** `computeDelta(baseline, current)` (`whatif.ts`) is the one function that
turns two posterior arrays into a per-state delta array (`null` on any absence or length
mismatch — a mismatched pair is never meaningful to diff, e.g. a stale posterior from a different
model). `GraphNode` derives its own `delta`/`dominant` from the two raw arrays `NetworkView`
already puts on `nodeData` (`posterior`, new `ghostPosterior`) rather than being handed a
precomputed third array — keeps "what does a bar look like" in one place. The **baseline** itself
("prior, or committed scenario", literally, per the plan's own wording) is owned by whoever runs
the fetch: `NetworkPanel` fetches the true unconditioned prior once, right after structure loads
(`fetchWhatIf(id, {})`, generation-guarded against a `modelId` race the same way the structure
fetch already was, fire-and-forget relative to the panel's own `loading` state since a missing
ghost baseline is a degraded-but-safe "no ghost mark" state, not an error), and `ghostPosteriors =
committedPosteriors ?? priorPosteriors` — i.e. the prior until the DM commits some evidence, after
which that committed evidence's own posteriors become the new baseline. Rendered as: an
`expanded`-tier chip gets a dashed vertical *line* at the ghost width behind the solid current-value
*fill* (a shape difference, not a shade difference); a `normal`-tier segmented bar gets a second,
outlined (not filled) ghost bar directly beneath it, same segmented layout, one tier down. A single
per-node "dominant delta" callout (largest-|Δ| state, `dominantDeltaIndex`, thresholded at one
percentage point to filter floating-point/imperceptible noise) is shown as an up/down arrow *icon*
(`IconArrowUp`/`IconArrowDown`, not just red/green text) plus the signed percentage in text — every
individual state's exact delta is additionally always available as plain tooltip text
(`stateTitle`), so the "pair colour with shape/position, never colour alone" rule (item 6) has a
fully-accessible fallback beyond just the one dominant callout.

**Evidence-mode scope decision (item 2).** The real backend (`WhatIfRequest.evidence: dict[str,
str]`, fed straight into pyAgrum's `setEvidence`) only ever conditions on evidence — there is no
soft-evidence/likelihood parameter and no do-operator intervention anywhere in `graphical.py`.
Built no mode toggle and no soft-evidence/intervene affordance at all (not even a visibly-disabled
one) — a disabled slider still implies "this is a dial that will work later," and nothing in the
plan or the backend suggests intervention support is imminent; a dead control would just be UI
debt. Instead: `NetworkEvidence`'s own doc comment in `types.ts` spells out why observation-only is
the whole story, `EVIDENCE_MODE_NOTE` (`interaction.ts`) is one exported, reused string so
`NetworkView`'s toolbar and any future caller say the identical thing, and it's shown as a
permanently-visible (not hover-only) icon+label in `NetworkView`'s toolbar — "Observational evidence
only" — every time evidence controls are on screen, not just discoverable via tooltip. The evidence
pin badge itself (already existing, `GraphNode`) now reads "Observed: `<state>`" instead of
"Evidence: `<state>`", since with only one mode ever possible, that one mode's own node mark
*is* the "distinct mark" the checklist item asks for — a second, always-identical mark for a
mode that can never occur would be pure noise.

**Exploratory vs committed evidence.** Chose **hold-ranges-fixed-and-show-clipping** over
recompute-and-warn, for one concrete reason: this package still has zero access to a method's own
range-recomputation machinery (deliberately — "no imports from any method store" — and no such
machinery exists in any method UI yet regardless), so a recompute-and-warn UI could only ever
*claim* to have rescaled preferences without actually doing it, which is worse than not building
the feature. "Hold fixed, show clipping" degrades honestly instead: nothing about a DM's existing
reference point/bound is silently reinterpreted, and once real range-recompute machinery exists
elsewhere, a host method UI has exactly one integration point to hang it off — the new
`onCommitEvidence` event — without this package needing to change at all. Concretely, exploratory
vs. committed is now a first-class distinction in the contract, not just a comment: `NetworkView`
gained `ghostPosteriors` (caller-owned baseline), `committedEvidence` (caller-owned "what's already
accepted"), and `onCommitEvidence` (fired by a new "Commit evidence" toolbar button, enabled only
when `evidence` diverges from `committedEvidence` — `hasEvidenceDiverged`, order-independent
map comparison). Toggling a state chip (`evidence`) is exploratory by construction — it only ever
updates the *live* `posteriors`/ghost-delta display, never `ghostPosteriors`/`committedEvidence`
themselves, so it reads as a ghost overlay without touching anything else. `NetworkPanel`'s
`handleCommitEvidence` is the reference implementation of what a host does with the event: promote
`committedEvidence`/`committedPosteriors` (the just-fetched live result — already computed for
exactly this evidence, no need to refetch) to be the new ghost baseline, then bubble the same event
up via its own `onCommitEvidence` prop for a real method UI to eventually hang range recomputation
off. The Cumulus demo wires the identical pattern against its mock. This phase does **not** add any
actual ideal/nadir recomputation or clipping-indicator UI — there is no range machinery to recompute
against yet, exactly as the task scoped it — only the distinction and the commit event/hook itself.

**Debounce/cancel/dim.** `createWhatIfRunner` (`whatif.ts`) is the one state machine both
`NetworkPanel` (real fetch) and the Cumulus demo (mock fetch) drive, specifically to avoid two
independently-written, independently-buggy copies of exactly the stale-response-clobbering class of
bug prior phases' code-review passes kept finding. Debounces via `createDebouncer` (150ms,
`EVIDENCE_DEBOUNCE_MS`), then on fire aborts any previous in-flight call via a fresh
`AbortController` per request *and* independently generation-guards the result (so a `fetchWhatIf`
override that ignores its `signal` — a real possibility, since it's caller-injectable — still can't
let a stale response clobber a newer one; belt-and-suspenders, not either/or).
`onStale(true)` fires **synchronously at `request()`**, not once the network call actually starts —
a burst of clicks has to dim immediately, not only after the debounce window elapses, or the whole
window reads as "nothing happened yet" instead of "working on it." `cancel()` aborts and clears a
pending debounce without ever calling `onResult`/`onError`/`onStale(false)` — deliberately: a caller
that cancels because it's moving on (a `modelId` change, a dialog close) needs every trace of the
old query gone, including its eventual `stale=false`, which is why both `NetworkPanel.loadStructure`
and the Cumulus demo's dialog-close effect explicitly reset their own `stale`/`network_stale` flag
right after calling `cancel()` — `cancel()` itself intentionally leaves that to the caller (see the
"watch for stale state" class of bug called out for this phase: a naive `cancel()` that also fired
`onStale(false)` would be indistinguishable, from a subscriber's perspective, from the cancelled
request having actually completed). `NetworkView`'s `stale` prop dims the *entire* canvas+table pane
(a wrapping `opacity-50` transition plus a small "Updating…" pill) rather than any individual node,
per the plan's "partial updates read as bugs" reasoning — a partially-refreshed canvas is worse than
a fully dimmed one. Ported the debounce/cancel/dim UX into `NetworkPanel.svelte`, whose own
doc comment previously said this belonged "layered above the fetch, not inside this wrapper's one
job of owning it" — updated: `NetworkPanel` *is* that layer now, since it's the thing that actually
owns the fetch such a policy needs to wrap, and inventing a second wrapper around it would only add
indirection.

**Animation.** Bar-width transitions are plain CSS (`transition-[width]`, `duration-[250ms]` —
inside the plan's 200–300ms window), applied only to `width`/`background-color` on posterior
fills/segments, never to a Svelte-Flow node's `position` (unaffected, still driven entirely by
`layout`/manual drag as before) — "never animate layout" holds by construction, since nothing here
touches the reconciliation effect that assigns `position`.

**Null-explanation mechanism.** `explainNoChange` (`null-explanation.ts`, new module, same
pure/testable pattern as `dag-algorithms.ts`/`interaction.ts`) takes a node id, the current
evidence, and an already-computed delta, and returns an explanation string only when: evidence is
actually set, the node isn't itself the pinned evidence (trivially "explained" by being the thing
that was set), the delta is non-null and *actually* zero (floating-point-noise-tolerant, `1e-9`),
and — the one expensive check, done last so it's only paid when every cheaper guard already passed —
`dSeparated(model, [nodeId], evidenceIds)` from `dag-algorithms.ts` (phase A) confirms it
structurally. `NetworkView` computes this per node in its existing structural-reconciliation effect
and stores it on `nodeData.nullExplanation`; `GraphNode` shows it as a small "unaffected" badge
(`IconInfo` + tooltip carrying the full sentence) exactly when there's no dominant-delta callout to
show instead (the two are mutually exclusive by construction — a real change and "provably no
change" can't both be true for the same node). Exercised live in the Cumulus demo: the existing
`sampleIrrigationNetwork` fixture already has the needed structure for free (`crop_yield` is an
unconditioned collider between `irrigate` and `weather` via `soil_moisture`), so pinning "Irrigate"
alone provably never reaches "Weather" — no fixture changes were needed, confirmed with a dedicated
`dSeparated` regression test (`sample-network.test.ts`) precisely so a future edit to that fixture
that broke this property would fail a test instead of silently breaking the demo's own worked
example.

**Colour-blind-safe encoding (item 6).** Every "changed vs unchanged"/"ghost vs current" pair in
this phase carries a non-colour cue: ghost marks are a different *shape* than the value they're
compared against (a dashed line/outline vs. a solid fill — not a paler tint of the same fill); the
dominant-delta callout pairs its red/green text with a distinct up/down *icon* and an explicit sign
character, never colour alone; the null-explanation badge is a wholly separate icon+text state, not
a colour variant of anything; and every state's exact delta is available as plain accessible text
(tooltip) regardless of any of the above. The evidence pin/policy-check badges (pre-existing, phases
B/A) already followed this convention (icon + position, not colour-only) and are unchanged in kind
here, only in wording (pin badge: "Observed" vs "Evidence").

**Tests.** `webui`'s vitest suite grew from 52 (end of phase C) to 91 (39 new): `whatif.test.ts`
(24 — `createDebouncer` timer semantics, `createWhatIfRunner`'s full debounce/abort/generation-guard
matrix including the slow-request-resolves-after-a-faster-one-supersedes-it and
cancel-during-in-flight cases, `computeDelta`, `dominantDeltaIndex`, `hasEvidenceDiverged`),
`null-explanation.test.ts` (7), `sample-network.test.ts` (8 — including the `dSeparated` regression
guarding the demo's own worked example described above). `npx vitest run`: 91/91 pass. `npx
svelte-check`: 66 errors / 13 warnings — exactly the phase A/B/C baseline; one real regression was
caught and fixed along the way (`NetworkNodeData.nullExplanation` is typed `string | undefined`
everywhere else in this contract, but `explainNoChange` returns `string | null` — `?? undefined` at
the one assignment site in `NetworkView`, confirmed back to baseline afterward). `npx vitest run`
and `npx svelte-check` both run under Node v24.15.0 via `nvm use 24`, matching every prior phase's
recorded requirement (default `node -v` on this machine is v20.20.2, too old). `npm run lint` is
still broken for the same pre-existing, branch-independent reasons phases A–C recorded, but
(unlike those phases) plain `npx prettier --check`/`--write` turned out to work fine here for
`.ts`/`.svelte` files in isolation — used to format every file this phase created outright
(`whatif.ts`, `null-explanation.ts`, both `.test.ts` files, `sample-network.ts`'s additions) and
every file this phase substantially rewrote (`NetworkView.svelte`, `NetworkPanel.svelte`,
`GraphNode.svelte`). Deliberately *not* run on `CUMULUS/+page.svelte`: confirmed (by copying the
pre-phase-D committed version aside and checking it in isolation) that file already had pre-existing
Prettier drift unrelated to this change before this phase touched it, so running the formatter
across the whole file would have reformatted unrelated, untouched code — hand-matched this phase's
additions to the surrounding style there instead, per every earlier phase's precedent.

**Not done / explicitly scope-cut.** No live exercise of `onWhatIfResult`/`utilities` against a
running backend (no compiled diagram exists in this dev sandbox to call `/graphical/whatif`
against — same limitation phase 7 itself operated under); no real ideal/nadir range recomputation
or clipping-indicator UI (no such machinery exists anywhere yet, and building a fake one would
misrepresent this phase's own "hold fixed, don't lie" decision above); no `session_id` plumbing
through `NetworkPanel` (the real `WhatIfRequest.session_id` field is optional and simply unused —
every `NetworkPanel` call falls back to the backend's own per-user-per-problem cache key — a small,
worth-flagging gap for whichever phase first wires a real interactive session through this
component, not required by anything in this phase's checklist).

**Code review pass (before merging).** A deep-dive review of the initial implementation found 13
issues, all fixed:

- **`WhatIfRunner.cancel()` didn't bump `generation`.** The JSDoc and this section's own original
  text both claimed a cancelled request's callbacks never fire, but `cancel()` only called
  `debouncer.cancel()`/`controller?.abort()` — an in-flight fetcher past its own debounce (e.g. one
  whose `fetchWhatIf` override ignores `signal`, a documented possibility) still had
  `myGeneration === generation` when it resolved, so `onResult`/`onStale(false)` fired anyway.
  Concretely: switching `modelId` mid-request could let the old model's what-if response land after
  the new model's structure had already rendered, painting stale posteriors/policy/utilities onto
  the new (possibly same-id) nodes. Fixed by incrementing `generation` inside `cancel()`, before
  aborting — every outstanding fetcher's generation check now fails regardless of whether it
  honours the abort signal. `whatif.test.ts` gained a case using a fetcher that never even looks at
  its `signal` (`Promise.resolve('a')`), firing the debounce synchronously, calling `cancel()`
  before the already-resolved promise's `.then` is processed, and asserting `onResult`/
  `onStale(false)` never fire — the previous "never resolves" cancel test couldn't have caught this
  at all.
- **A what-if fetch failure destroyed the whole panel, not just the stale result.** `NetworkPanel`
  wrote both the structure fetch's and the what-if fetch's failures into one `error` state, and the
  template rendered `{:else if error}` instead of `<NetworkView>` — so one failed what-if POST
  (transient 500, or the 400 the next finding was producing) wiped the canvas, table, evidence,
  layout, and commit state, with no recovery except changing `modelId`. Split into `structureError`
  (fatal, replaces the pane, unchanged behaviour) and `whatIfError` (non-fatal, a small dismissible
  banner shown above a still-rendered `<NetworkView>`, cleared automatically on the next successful
  result and when a new evidence-driven request starts) — mirrors how the prior-baseline fetch just
  below it already treated its own failure as non-fatal.
- **`decisionMode="manual"` evidence 400s the real backend** — and, before the previous fix, took
  the whole panel down with it. `graphical.py`'s `/whatif` hard-rejects any evidence key that isn't
  a `DiscreteNetwork` chance-node name, but `isNodeEditable` allows clicking decision-node states in
  `'manual'` mode and `NetworkPanel` posted the whole evidence map, decision picks included. This is
  exactly the "what-if restricted to chance nodes" constraint the plan's own phase E entry already
  names. Fixed with `filterChanceEvidence` (`interaction.ts`, new, tested) — restricts outgoing
  evidence to chance-node ids (via `model.nodes[].kind`) at the two `NetworkPanel` call sites into
  `fetchWhatIf`; a decision pick stays local-only (still shown as a pin on the node) rather than
  being sent to the backend as if it were an observation.
- **Committing evidence while a what-if was still in flight/debounced captured the wrong ghost
  baseline.** `handleCommitEvidence`'s `committedPosteriors = posteriors` assumed `posteriors`
  already reflected the evidence being committed — true only when `stale === false` — but the
  Commit button wasn't disabled on `stale`, only on `!diverged`. Fixed by disabling Commit while
  `stale` too (`NetworkView`'s toolbar button, `disabled={!diverged || stale}`) — simpler and just
  as correct as deferring the promotion to a matching response, and it also means the DM never sees
  a Commit affordance that would silently commit last cycle's numbers.
- **The d-separation "why didn't this change" explanation ignored `committedEvidence`, going silent
  right after every commit.** `explainNoChange` always queried d-separation with an empty
  conditioning set — correct only while the ghost baseline was still the unconditioned prior. Once
  a commit makes `ghostPosteriors` the committed scenario, the right query is "is this node
  d-separated from the evidence *added since the commit*, given the committed evidence" (e.g. chain
  weather→soil_moisture→crop_yield: commit `soil_moisture`, then explore `weather` alone —
  `crop_yield`'s posterior is correctly unaffected, since the committed evidence on the *middle*
  node of the chain already blocks it, even though `weather`/`crop_yield` are *not* unconditionally
  d-separated). Fixed: `explainNoChange` now takes `committedEvidence` as a required parameter,
  queries only the evidence entries added/changed relative to it, and conditions on the committed
  entries; `NetworkView` threads its own `committedEvidence` prop through. `null-explanation.test.ts`
  gained a dedicated `describe` block proving the same zero delta *is* explained given the committed
  context and *is not* explained without it (against the identical evidence/delta), plus a
  precondition test confirming the unconditioned query really would have gotten it wrong. Related
  copy bug fixed alongside: `GraphNode`'s per-state tooltip hardcoded "(prior X%, ...)" even when
  the ghost baseline was a committed scenario, not the prior — reworded to "(baseline X%, ...)",
  correct either way without needing to plumb an extra "is this the prior or a commit" flag through.
- **Cumulus demo: closing the dialog mid-flight left evidence and posteriors permanently out of
  sync.** The close effect cancelled the runner and reset `network_stale`, but not
  `network_evidence`/`network_posteriors`/commit state, and the evidence-watching effect only
  re-fires on an actual *change* — so reopening showed a pinned evidence badge against posteriors
  that were never actually refreshed for it, with nothing to trigger a refresh. Fixed by resetting
  `network_evidence`/`network_posteriors`/`network_committed_evidence`/`network_committed_posteriors`
  in the same close effect, mirroring `NetworkPanel.loadStructure`'s own reset list.
- **`mockWhatIf`'s evidenced node showed no visible change at all** — the one node the DM had just
  clicked. `nodeId in evidence` was classified `unaffected` and returned the baseline verbatim, but
  real hard evidence (pyAgrum `setEvidence`) collapses that node to one-hot on the observed state.
  Fixed to emit a one-hot vector at the observed state's index (via a `nodeId -> states` lookup);
  `sample-network.test.ts`'s test claiming evidence "pins it directly" (while asserting the
  baseline, unpinned vector) was itself wrong and rewritten to assert the one-hot collapse for two
  different observed states.
- **`NetworkPanel`'s load effect could be retriggered by an inline `fetchStructure`/`fetchWhatIf`
  prop on any unrelated parent re-render**, wiping evidence/posteriors/policy/utilities/commit state
  mid-session. `loadStructure` read `fetchStructure` (a destructured-prop, i.e. reactive, getter)
  synchronously before its first `await`, while still inside the effect's own synchronous
  execution — the natural call pattern `fetchStructure={(id) => client.structure(id, token)}`
  creates a new closure every parent render, and that new prop identity became a tracked dependency
  of the effect. Fixed by capturing `modelId` in the effect and reading `fetchStructure`/
  `fetchWhatIf` via `untrack(...)`, passing them into `loadStructure` as plain arguments instead of
  having it read the reactive props itself — only `modelId` drives re-fetching now.
- **No cancellation on component teardown.** Neither `NetworkPanel` nor the Cumulus page cancelled
  the what-if runner on unmount (only on `modelId` change / dialog close) — a pending debounce timer
  or in-flight fetch could outlive navigation away and still write state / call `onWhatIfResult`
  afterward. Fixed with a dedicated no-dependency `$effect(() => () => whatIfRunner.cancel())` in
  both places (distinct from the Cumulus dialog's own close-effect cancel, since the dialog can
  reopen within the same page lifetime).
- **Efficiency: `explainNoChange` rebuilt the adjacency/ancestral closure from scratch, via
  `dSeparated`, for every node on every reconciliation** — evidence toggle, posterior arrival,
  zoom-driven detail-level change, resize — the identical anti-pattern phase C's own review already
  fixed once for `ancestralSet`. Fixed by giving `dSeparated` an optional `adjacency` parameter
  (defaulting to `buildAdjacency(model)` for standalone callers, so every existing call site keeps
  working unchanged), threading it through `explainNoChange`, and passing `NetworkView`'s existing
  `$derived.by` adjacency into both.
- **The "observational evidence only" disclosure was `hidden` below the `sm` breakpoint**,
  contradicting this section's own commitment that the note is "permanently visible ... every time
  evidence controls are on screen" — on a narrow viewport the evidence chips/pins/Commit button
  stayed fully interactive while the disclaimer vanished, the opposite of the intended safety
  property; the full sentence was also `title`-only, unreachable on touch. Fixed by dropping the
  `sm:` gate entirely (the toolbar already `flex-wrap`s) and adding `aria-label` alongside `title`
  so the full sentence is the element's accessible name regardless of hover/touch.
- **Dead/misleading export**: `BAR_TRANSITION_MS` was documented as "the CSS transition duration
  `GraphNode.svelte`'s bars use," but `GraphNode` hardcoded `duration-[250ms]` Tailwind classes
  (arbitrary-value classes can't consume a JS constant) and never imported it — changing the
  constant would have silently done nothing. Fixed by wiring it via `style:transition-duration`
  (inline styles win over the classes' own transition-*property* declaration, so `transition-
  [width]`/`transition-[width,background-color]` stay in the class string for *which* properties
  animate, while the constant now genuinely drives *how long*).
- **Lower-priority, fixed alongside:** `NetworkPanel` now seeds `posteriors` (not just
  `priorPosteriors`) from the prior-baseline fetch when no exploratory evidence has been set yet, so
  the first render shows the real evidence-free distribution instead of every editable chance node's
  equal-width "no posterior" placeholder; the Cumulus demo's `network_posteriors` initial `$state`
  now wraps `sampleIrrigationBaselinePosteriors` in `structuredClone`, matching `network_model`'s own
  clone two lines above; `NetworkView`'s doc comment near the Commit button had its disabled/enabled
  logic backwards ("always-enabled-until-first-evidence") and was reworded to match actual behaviour
  (disabled until evidence diverges from committed); `NetworkPanel.handleEvidenceChange` no longer
  redundantly reassigns `evidence` (already kept in sync via `bind:evidence`) alongside triggering
  the what-if request; `<NetworkView>` in `NetworkPanel` is now wrapped in `{#key modelId}` so
  hover/select ids and dragged node positions from a previous model can no longer bleed onto a new
  one that happens to share node ids, which previously undercut this phase's own "everything is
  reset on model switch" claim.

Verified after all fixes: `npx vitest run` 102/102 pass (up from 91 -- new coverage for the
`cancel()`/generation-bump fix, `filterChanceEvidence`, the committed-evidence-conditioned
`explainNoChange` behaviour, the `dSeparated` adjacency-reuse parameter, and the one-hot
`mockWhatIf` collapse). `npx svelte-check`: 66 errors / 13 warnings, exactly the phase A/B/C
baseline, confirmed none of the newly-touched files (`whatif.ts`, `null-explanation.ts`,
`dag-algorithms.ts`, `interaction.ts`, `sample-network.ts`, `NetworkView.svelte`,
`NetworkPanel.svelte`, `GraphNode.svelte`, `CUMULUS/+page.svelte`, and every associated `.test.ts`)
contribute any of them. Both commands still run under Node v24.15.0 via `nvm use 24`. `npx prettier
--write` re-applied to every file this pass touched, with the same `CUMULUS/+page.svelte` exception
(pre-existing, unrelated formatting drift already present in that file before this phase) as the
rest of phase D — new/changed lines there were instead hand-wrapped to stay under printWidth.

### Phase E — Policy display and derived views

- [x] Read decision policies out of solutions via `IndexMap`; display read-only in
      `decisionMode: 'solver'`, updating as the DM navigates. What-if inside method UIs is
      then restricted to chance nodes, which is conceptually right — "if A happens" is a
      question about the world, not about what we choose.
- [x] MEU with its standard deviation (pyAgrum's `MEU()` returns mean and variance)
- [x] Named evidence sets stored server-side, selectable as comparison baseline, shareable by
      URL — gives reproducibility, a figure corresponds to a scenario id not a click sequence
- [x] Sensitivity tornado: single-node evidence sweep, ΣΙstatesΙ queries on a warm engine
- [x] Value of information, same machinery, framed as EU gain from observing before deciding

**Findings.** Backend additions live in `desdeo/problem/graphical/compile.py` (the promoted
policy-readback function) and `desdeo/api/routers/graphical.py` (four new endpoints alongside
`/compile`/`/structure`/`/whatif`); frontend additions live in the existing
`webui/src/lib/components/custom/network/` package. Kept proportionate to "one implementation
pass, not five phases' worth of prose" per the plan's own instruction for this section.

**1. Policy readback.** `_read_back_policy`'s logic (`tests/test_graphical_compile.py`) is now a
public `read_policy_from_solution(index_map, optimal_variables) -> dict[str, DecisionPolicy]` in
`compile.py`, alongside a new `DecisionPolicy` model. It reports each decision's *whole* policy
function (`information_state_flat_index -> chosen_state_index`), not one collapsed choice —
regression test `test_read_policy_from_solution_returns_full_policy_function` exercises exactly
this on a decision with two genuinely different per-information-state choices (a single-value
result would have passed a lazier test). It is deliberately **index-based, not label-based**:
`IndexMap` carries no node-label information (that lives in `DiscreteNetwork`/`DecisionAnnotation`),
so translating indices to labels is left to the caller. `tests/test_graphical_compile.py`'s own
`_read_back_policy` now delegates to this function (a touch to an already-checked-off Phase 4/5
test helper, justified as literally the promotion this checklist item asks for — its brute-force
oracle shape needs tuple-keyed info states, so it decodes the library function's flat-int keys
locally rather than duplicating the underlying bookkeeping).

Exposed via `POST /graphical/{problem_id}/policy`: takes either `variable_values` (a solved
`Problem`'s `SolverResults.optimal_variables`) or `solution_id` (a `UserSavedSolutionDB` row) plus
`evidence`, and returns `{policy: dict[str, str]}` — deliberately the *same shape* `/whatif`'s own
`policy` field already returns, so a `decisionMode: 'solver'` UI can't tell (and doesn't need to)
whether an answer came from a live pyAgrum ID-solve or a specific stored solution's decoded `z`.
One real gotcha found while wiring `solution_id`: `UserSavedSolutionDB.variable_values` is declared
`dict[str, VariableType]` (`VariableType = float | int | bool`, no `list`) — apparently unable to
hold a `TensorVariable`'s flat z-array at all. Confirmed empirically that it works anyway: the
field is `sa_column=Column(JSON)`, and neither construction nor DB round-trip actually validates
against the declared type for it, so a list survives untouched. `/policy` relies on this working
behaviour but doesn't fix the misleading annotation — flagged here for whoever next touches that
model, out of this phase's remit to correct.

Frontend: `NetworkPanel` gained `policySource`/`fetchPolicy` props (default `null`/
`defaultFetchPolicy`, purely additive). When `policySource` is set, `fetchWhatIfWithPolicy` runs
the `/whatif` and `/policy` calls together under the *same* `whatIfRunner` debounce/cancel/
generation-guard cycle (`whatif.ts`, phase D) rather than a second, independently-timed fetch, and
overlays the solved-policy decode onto `policy` — reusing the existing `NetworkPolicy`/policy-badge
rendering in `NetworkView`/`GraphNode` (phase B) unchanged. Scope note: like the pre-existing
`/whatif`-derived `policy`, the very first render (before any evidence-driven request has fired)
shows no policy yet — not a new gap, the same limitation Phase D's own `priorPosteriors` fetch
already had for `policy` (it only ever seeded posteriors, never policy).

**2. MEU with its standard deviation.** Confirmed empirically, not assumed:
`ShaferShenoyLIMIDInference.MEU()` returns `{"mean": ..., "variance": ...}` (built a small 1-utility
`InfluenceDiagram` and printed it). `_expected_value_no_decision` (the no-decision-diagram branch)
now returns `(mean, variance)` from the identical joint-posterior contraction the mean already
needed (`variance = E[U^2] - E[U]^2`, clamped at `0.0` for float roundoff on a near-deterministic
distribution); `_compute_utilities` now returns `(means, std_devs)` — one signature change to an
existing Phase 7 function, with exactly one call site (`/whatif`) updated to match.

**Sign convention, reasoned through rather than assumed, per the checklist's own instruction.** The
mean needs the identical maximize/minimize negate `MEU()["mean"]` already needed (pyAgrum's utility
table is written in a maximize-only sign convention, negated internally for a minimizing
objective — see `to_pyagrum_influence_diagram`). The **variance does not**: `Var(-X) == Var(X)`
(`Var(aX) = a^2 * Var(X)`, `a = -1`), so a variance computed over the negated-for-pyAgrum table is
already numerically identical to the variance over the objective's own values — negating it would
be wrong twice over (variance is already non-negative, and the quantity doesn't change sign to
begin with). `WhatIfResponse` gained `utility_std_devs: dict[str, float]` (same keys as
`utilities`); `types.ts` gained `NetworkUtilityStdDevs`/`WhatIfResult.utilityStdDevs`, remapped
through the identical `util_<symbol>` prefix `utilities` already needed, and threaded through the
same `onWhatIfResult` hook Phase D built for `utilities` — no second hook.

Two hand-computed regression tests, both real gotchas caught while writing them: the no-decision
case (mean `8.0`, `variance = 80.0 - 64.0 = 16.0`, `std = 4.0`) is straightforward, but the first
attempt at a decision-diagram std-dev test reused `_decision_annotation`'s own fixture (`D` fully
observes `A`) and got a **degenerate zero variance** — `D` matches `A` perfectly every time, so the
achieved utility is deterministic regardless of `A`'s realization. Fixed by building a *blind* `D`
(`information_set=[]`) instead, whose one fixed optimal choice still lands on different utility
cells depending on `A`, giving a genuine `variance = 21.0` cross-checked directly against
`MEU()["variance"]`.

**3. Named evidence sets.** Persistence-shape decision, weighed and documented (per the checklist's
own instruction) in `GraphicalEvidenceSetDB`'s docstring (`api/models/graphical.py`, new module):
**own simple table, not `ProblemMetaDataDB`'s polymorphic side-table pattern.** That pattern exists
for data that *describes the problem itself* (network/annotation/RJT/`IndexMap`, geographic
data, ...) — one coherent row set per problem, doubling as the frontend's signal to render a
specialised view. A saved evidence set is a different shape of thing: a user-owned, independently
addressable *bookmark* into a diagram someone is exploring, closer in kind to
`UserSavedSolutionDB` (`api/models/generic_states.py`) than to problem metadata — so it gets its
own table, `problem_id`/`user_id` directly, the same shape `UserSavedSolutionDB` already uses.

CRUD-ish endpoints: `POST/GET /graphical/{problem_id}/evidence-sets` (create; list — *every* set
saved against the problem, not just the requesting user's own, since a saved set is meant to be a
reproducible, shareable bookmark for anyone with visibility into the problem, not a private list;
creation validates `evidence` eagerly via the same `_validate_chance_evidence` helper `/whatif`
uses, so a bad save fails immediately rather than only once something eventually loads it) and
`GET/DELETE /graphical/evidence-sets/{evidence_set_id}` (addressed by the set's own id, independent
of `problem_id` in the path — this is the actual "shareable by URL" mechanism: visibility is
re-checked against the row's own stored `problem_id`, so sharing an id with someone who can't see
the underlying problem still 404s; delete is owner/admin/analyst-restricted, unlike reading).

Frontend: `evidence-sets.ts` (`mapEvidenceSetResponse` — the one pure, tested piece — plus thin
default fetchers, the same "no component-testing setup, so pure logic is what's covered" pattern
`whatif.ts`/`sensitivity.ts` already follow). `NetworkPanel` gained `fetchEvidenceSets`/
`saveEvidenceSet` props defaulting to the real endpoints (the same "on by default, injectable"
convention `fetchStructure`/`fetchWhatIf` already use, rather than opt-in-only), a save
input+button, and a baseline `<select>`. Selecting a saved set calls `loadEvidenceSetAsBaseline`,
which fetches that set's own what-if result and promotes it into `committedEvidence`/
`committedPosteriors` via the *exact same* mechanism `handleCommitEvidence` already uses for a live
DM commit (phase D) — not a parallel baseline mechanism, per the checklist's own wording ("become
that baseline via the existing mechanism").

**4. Sensitivity tornado.** New `POST /graphical/sensitivity`, reusing `_engine_cache`/
`_get_or_build_entry` — the same warm, session-scoped pyAgrum engine `/whatif` already maintains,
per the checklist's own "on a warm engine" wording. `_resolve_cache_key`/`_validate_chance_evidence`
were factored out of `/whatif`'s own inline logic (a minor touch to already-checked-off Phase 7
code) so `/sensitivity`/`/voi` share the identical session-resolution and evidence-validation rules
rather than drifting copies.

`_sweep_node(engine, ..., sweep_node_name, target, target_state)`: one `setEvidence`/
`makeInference`/read cycle per state of `sweep_node_name`, everything else held at the caller's
`evidence` — `Σ|states|` queries across a full sweep, literally, one call per candidate node's own
cardinality. `target` is either an `ObjectiveNode.symbol` (routed straight through
`_compute_utilities`'s own mean, including its existing multi-objective-with-decision `501` —
reused, not reimplemented) or a chance/decision node's own name (`target_state` required; its
marginal probability under the swept evidence). Sweep candidates are restricted to chance nodes not
already in `evidence` and not the target itself, matching `/whatif`'s own chance-nodes-only
restriction. The whole sweep runs under `entry.lock` (many inference calls share one cached, mutable
engine) with a `finally` that restores the engine's evidence to exactly the caller's `evidence`
afterward — without it, a `gum.GumException` raised mid-sweep (or just reaching the end of a normal
sweep) would leave the shared cached engine sitting on whatever state the last swept value set,
corrupting the next caller sharing that cache key.

Frontend: `sensitivity.ts` (camelCase mapping of both `/sensitivity` and `/voi` responses, plus
`tornadoDomain`/`sortByImpact` — pure, tested) and `TornadoChart.svelte`, a presentational,
plain-percentage-`<div>` bar chart (no SVG/d3 dependency) — checked `visualizations/horizontal-bar/`
first per the checklist's own instruction, and didn't reuse it: different component-authoring
convention (Svelte 4 `export let` there, this package is Svelte 5 runes throughout) and a different
job (two *draggable* bounds on one bar, not N independent, read-only range bars). Deliberately
**not** embedded inside `NetworkPanel`/`NetworkView`: mirrors phase C's own "objective values live
in the method's own UI, the graph stays purely structural" precedent — a tornado chart is a
method-level analysis view a host page composes next to `<NetworkPanel>`, exported alongside plain
fetch defaults, not something the reusable canvas package should own end-to-end.

**5. Value of information.** New `POST /graphical/voi`, implemented as a thin transform over
`_sweep_node` — exactly the primitive the checklist suggested trying first, and it held up cleanly
(worth recording precisely because the checklist explicitly anticipated it might not):
`VoI(node) = Σ_s P(node=s | evidence) * target_value(evidence, node=s) - target_value(evidence)`,
the weighted-by-`node`'s-own-current-marginal average of `_sweep_node`'s per-state values, minus the
unswept base value.

**Only meaningful for a diagram with at least one decision — a `400`, decided and documented rather
than left an accidental crash or a silent `0.0`.** "EU gain from observing before deciding"
presupposes something being decided; a no-decision diagram has no policy that could respond to the
observation, so the question doesn't type-check for it at all, not merely evaluate to zero.
Returning a well-formed-looking `0.0` would silently conflate "provably no value" with "the
question doesn't apply here" — the same "pick the smaller, honest, well-documented scope" call
Phase 5 (`DiagramConstraint` `kind == "tail"`) and Phase 7 (multi-objective-with-decision `501`)
already made for their own scope cuts. `value_of_information` itself is **not** clamped at `0` in
the response: `VoI >= 0` is a theorem for a rational decision maker under a correctly specified
diagram, not something the arithmetic here enforces, so a small negative result is real, useful
evidence of solver/inference floating-point noise rather than something to hide by clamping it away.

Hand-computed oracle test (`test_voi_hand_computed`): a *blind* `D` (mirroring MEU std dev's own
fixture choice above, and for the identical reason) with `VoI(A) = 1.8` worked out by hand —
`_decision_annotation`'s fully-observing `D` would again have made VoI trivially `0.0` (nothing left
to learn from observing `A`, since `D` already structurally sees it), the same degenerate-test trap
item 2's std-dev test hit, avoided the same way.

**Tests.** Backend: `tests/test_graphical_compile.py` grew from 29 to 30 tests (one new, exercising
`read_policy_from_solution` directly against a genuine two-information-state policy — marked
`graphical`/`gurobipy`, since it solves via `GurobipySolver`); all 35 tests in that module pass
(`pytest -m "graphical and gurobipy" tests/test_graphical_compile.py`, ~340s). `desdeo/api/tests/
test_graphical_router.py` grew from 15 to 35 tests (20 new, covering `/policy`'s happy paths,
`solution_id`, both validation-error cases, missing-`index_map`/missing-`z_symbol`; evidence-set
create/list/get/delete/visibility/validation; sensitivity's objective-target and node-target cases,
target-state validation, evidence-exclusion-from-sweep; VoI's hand-computed case and its three
scope-guard rejections; plus two new/extended `/whatif` assertions for `utility_std_devs`) — all 35
pass. `just test-api`: 213/213 pass (up from 193 recorded at the end of Phase 7), no regressions.

Frontend: `npx vitest run` (Node v24.15.0 via `nvm use 24`, matching every prior phase's recorded
requirement): 114/114 pass, up from 102 at the end of phase D (12 new — 10 in
`sensitivity.test.ts`, 2 in `evidence-sets.test.ts`). `npx svelte-check`: 66 errors/13 warnings —
re-confirmed as the current baseline (not trusted stale, per this task's own instruction) before
starting, and exactly matched after. One genuine new-error pair was caught and fixed along the way:
giving `fetchEvidenceSets`/`saveEvidenceSet` real (non-`undefined`) defaults made TS correctly flag
the `{#if fetchEvidenceSets && saveEvidenceSet}` guard around the evidence-set UI as "always true,
did you mean to call it?" — fixed by removing the now-dead guard (the UI is unconditionally
rendered, matching the props' own new "on by default" contract) rather than suppressing the
warning. `npx prettier --write` applied to every file this phase created or substantially edited,
per every prior phase's own workaround for `npm run lint`'s still pre-existing, branch-independent
brokenness (`eslint.config.js`'s missing import, the `prettier`/`prettier-plugin-svelte` version
mismatch) — not re-diagnosed here, per this task's own instruction not to burn time on it.

**Scope cuts, stated plainly rather than left implicit:**

- Cumulus's dev-only network demo dialog was **not** extended to exercise `policySource`/
  evidence-sets/the tornado chart/VoI live end-to-end — there is still no compiled diagram in this
  dev sandbox to call any of these endpoints against, the identical limitation phase D's own
  findings already recorded for `onWhatIfResult`/`utilities` ("no live exercise ... same limitation
  phase 7 itself operated under"). Verified by reading the code plus `vitest`/`svelte-check`, not by
  driving a running app, exactly as phase D did for its own equivalent gap.
- No bundled "SensitivityPanel" wrapper analogous to `NetworkPanel` (i.e. no packaged
  fetch/loading/error state machine around `TornadoChart`/VoI) — kept to a presentational chart
  component plus exported fetch defaults, on the same "method-level analysis view, not something
  the reusable canvas package should own end-to-end" reasoning phase C already established for
  keeping objective values out of the graph itself.
- `UserSavedSolutionDB.variable_values`'s misleading `dict[str, VariableType]` type annotation
  (works correctly at runtime for `/policy`'s `solution_id` path, per the finding above) is flagged,
  not repaired — pre-existing, and tightening a shared model's type annotation is outside this
  phase's remit.
- No dedicated API-layer test for `no_forgetting`/multi-decision policy decoding via `/policy`
  (only single-decision fixtures) — `read_policy_from_solution` decodes each decision independently
  with no `no_forgetting`-specific branching of its own, and Phase 4/5's own oracle suite already
  exercises the underlying `DecisionIndexMap`/effective-information-set machinery for multi-decision
  and `no_forgetting` diagrams directly; not re-proven a second time through the API layer here, a
  reasonable line given the remaining scope of this phase. **Superseded by the code review pass
  below** — that gap turned out to hide a real bug (`/policy` silently dropping `D2`), not just an
  undertested-but-correct path, so a dedicated test now exists after all.

**Code review pass.** A review of this phase's diff found 10 issues, all fixed. Three real
correctness bugs: `/sensitivity`/`/voi`'s sweep `finally` block called `entry.engine.setEvidence
(request.evidence)` to restore the cached engine's state, unguarded — when `request.evidence`
itself was what pyAgrum had just rejected, that same call raised again inside `finally`, silently
replacing the in-flight `HTTPException(400)` with an uncaught 500 (confirmed empirically); fixed by
wrapping the restore in its own `try/except gum.GumException: pass`, since no caller downstream of a
sweep can actually observe leftover sweep evidence anyway (every caller sets its own full evidence
before reading). `_decode_solved_policy` only included a decision when every member of its
*effective* information set was present in the (validated chance-only) `evidence` — under
`no_forgetting`, a downstream decision's effective information set includes an *upstream* decision,
which chance-only evidence could never satisfy on its own, so that decision was unconditionally
dropped with no error; fixed by decoding in `decision_order` and threading each resolved decision's
own choice into a working evidence-like lookup for the next. `_resolve_target_value`'s no-decision
branch routed through `_compute_utilities` (which computes *every* objective's expected utility) and
discarded all but the target's own value — an N-times redundant cost per swept state on an
N-objective diagram, undercutting the "Σ|states| queries" cost model the checklist promises; fixed
by calling `_expected_value_no_decision` directly for just the target, preserving
`_compute_utilities`'s multi-objective-with-decision `501` for the with-decision branch unchanged.

Two robustness gaps in `/policy`: `read_policy_from_solution` inferred `n_info_states` from
`len(z_values) // n_states` instead of validating it against `prod(information_cardinalities) *
len(states)`, so a wrong-length `z` (e.g. a stale saved solution from before a recompile changed a
cardinality) silently produced a policy dict missing some info-state keys, surfacing as an uncaught
`KeyError`/500 the first time something looked one up; now raises a documented `ValueError`, mapped
to a 400 alongside the existing `KeyError` case. `PolicyRequest.variable_values` was typed
`dict[str, list[float]] | None`, stricter than `solution_id`'s equivalent path — a diagram compiled
with a Phase 6 `RiskMeasure` has a scalar `eta` `Variable` in `optimal_variables` (a bare `float`,
not a list), and posting a real solved-solution dict through this field 422'd; widened to
`dict[str, list[float] | float] | None`.

One validation gap: `_validate_chance_evidence` (evidence-set creation, and — shared — `/whatif`/
`/sensitivity`/`/voi`) checked node names but never state labels, so a typo'd label saved
successfully (200) and only failed later, opaquely, the first time anything tried to use the set;
now also checks each value against that node's own `labels`. Doesn't change `/whatif`'s status code
(it already 400'd via `gum.GumException`), only tightens the message.

Four frontend issues in `NetworkPanel.svelte`, all missing the generation-guard/filtering
conventions the rest of the component already follows: `loadEvidenceSetAsBaseline` had no
generation guard at all (unlike `loadStructure`/the prior-posteriors fetch), so a fast baseline
reselect or a `modelId` change mid-request could install a stale response as the committed
baseline — fixed with the same `requestId`-vs-`structureRequestId` pattern plus a dedicated
`baselineRequestId` for the reselect case. The evidence-set save/load path bypassed
`filterChanceEvidence` (used everywhere else evidence is posted) — `handleSaveEvidenceSet`'s
disabled-guard and POST body, and `loadEvidenceSetAsBaseline`'s use of a loaded set's `evidence`,
now all go through it. The "Prior (default)" baseline option was inert (`Number('')` is `0`, no
evidence set has `id === 0`) — fixed by handling the empty-value case explicitly and binding the
select's value to real component state (`selectedBaselineId`, also cleared on a live DM commit) so
the dropdown can't visually disagree with the active baseline. The evidence-set list fetch was
needlessly serialized in front of the (unrelated) prior-baseline fetch in `loadStructure`, delaying
the ghost overlay for no reason — decoupled to run independently; a stale prop doc comment claiming
the evidence-set UI was conditional on `fetchEvidenceSets`/`saveEvidenceSet` both being given (true
before this phase's own real-defaults change, not after) was corrected to match.

One test-coverage gap addressed with judgment rather than a hard fix: `test_evidence_set_delete_
owner_only`'s `in (403, 404)` assertion couldn't distinguish "no visibility" (404) from "visible but
not owner" (403), since its only non-owner user had no visibility into the problem at all. Rather
than leave the ambiguous assertion, split it: the existing test now asserts a plain 404 (its actual
branch), and a new `test_evidence_set_delete_visible_non_owner_is_403` uses a shared `Group`
(the same visibility path `fetch_problem_with_role_check` checks, `test_models.py::test_group`
exercises directly) to give a second user real visibility without ownership, and asserts a genuine
403.

**Tests.** `desdeo/api/tests/test_graphical_router.py` grew from 35 to 42 (7 new: bad evidence
state label -> 400 for `/sensitivity`/`/voi`/evidence-set creation; a wrong-length `z_symbol` -> 400
not 500; a `RiskMeasure`'s scalar `eta` through `variable_values` -> not 422; a two-decision
`no_forgetting` diagram's `/policy` cross-checked against `/whatif`'s live policy; the split
403-vs-404 evidence-set-delete case). `just test-api`: 220/220 pass (up from 213). `pytest -m
"graphical and gurobipy" tests/test_graphical_compile.py`: 35/35 pass, unchanged — confirms
`read_policy_from_solution`'s new length check doesn't reject any genuinely-solved policy read-back.
`pytest -m graphical tests/`: 75/75 pass. `npx vitest run`: 114/114 pass, unchanged (no pure-logic
module touched by this pass). `npx svelte-check`: 66 errors/13 warnings, exactly the baseline this
phase's own implementation pass recorded — confirmed none of them trace to `NetworkPanel.svelte`.

### Phase F — Real backend integration for the CUMULUS demo

Unscheduled work that fell through the cracks of every prior phase: the CUMULUS network-structure
demo dialog (phase B) still ran entirely on hand-mocked data (`sample-network.ts`'s
`sampleIrrigationNetwork`/`mockWhatIf`), even though `NetworkPanel` (built in phase D, extended in
phase E) is a fully real component that has never actually been driven against a live compiled
diagram in the running app. This phase closes that gap.

- [x] A real backend graphical model matching (as closely as the compiler allows) the demo diagram
- [x] Compile + solve it, reachable from the running app, with a `problem_id` the webui can use
- [x] Swap `CUMULUS/+page.svelte`'s `<NetworkView>` + mock-fetch state for `<NetworkPanel>`
- [x] Handle compile-taking-a-moment / compile-or-solve-failing / `/whatif`-erroring, reusing
      `NetworkPanel`'s own loading/error UI rather than a second copy

**1. The demo model, and a real capability gap it surfaced.** `desdeo/problem/graphical/demo.py`
(`build_irrigation_demo`/`_network`/`_annotation`) builds a `DiscreteNetwork`/`DecisionAnnotation`
pair by hand, in the same style as `tests/fixtures/graphical.py`/the pig farm fixture, *not* a
literal structural copy of `sample-network.ts`. Two faithful reformulations were needed, both for
reasons already documented elsewhere in this file, not new discoveries:

- `sample-network.ts`'s `irrigate -> crop_yield` edge is a chance node conditioned on a decision —
  `DiscreteNetwork` can't express that (`rooted_junction_tree.py`'s own module docstring; the same
  limitation the pig farm fixture hit in Phase 6). Reformulated the same way: three counterfactual
  chance nodes (`crop_yield_none`/`_light`/`_heavy`), each conditioned on `soil_moisture` only,
  always drawn; `profit`'s own utility table (which legitimately *can* depend on a decision)
  selects the branch matching `irrigate`'s realized state.
- `sample-network.ts`'s `water_usage` (kind `"deterministic"`, parent `irrigate`) has no backend
  counterpart at all — `StructureNode.kind` is `Literal["chance", "decision", "objective"]`, no
  fourth kind exists. Folded into `profit`'s own utility table as a direct function of `irrigate`
  instead; nothing about this is even a compiler limitation, `"deterministic"` was simply never a
  representable node kind here.

**A third difference was not anticipated going in and is the more consequential one: the demo
diagram is single-objective, not two (`profit`/`water_cost`), and this is a hard requirement, not
a simplification.** `POST /graphical/whatif` calls `_compute_utilities` *unconditionally* on every
request (`graphical.py`, phase 7), and `_compute_utilities`'s own docstring already documents that
a diagram with more than one `ObjectiveNode` *and* at least one decision is an open problem
(`ShaferShenoyLIMIDInference` exposes no joint-posterior primitive to recover one objective's own
share) — resolved there with an `HTTPException(501)`, not an approximation. Building the demo
network with both `profit` and `water_cost` as originally planned would have made `/whatif` respond
501 to *every* evidence query, not merely omit `utilities` — the entire what-if exploration half of
the dialog would be dead on arrival. Caught before it shipped by actually running the intended
two-objective version against a local API instance while drafting this phase, not by inspection.
Dropped `water_cost` as a separate `ObjectiveNode` entirely; `profit`'s own utility table already
nets out water/irrigation cost per choice, so the traded-off-explicitly framing the frontend
fixture's two-objective story described is preserved in spirit, just not as a second, independently
explorable node. Documented in `demo.py`'s own module docstring so the reasoning isn't lost.

Verified the model itself is sound, not just schema-valid: hand-computed expected profit for each
of `irrigate`'s three choices directly from the network's own CPTs (`tests/test_graphical_demo.py`)
gives `heavy` as the maximizing choice (`2780.625`, `light` `2696.875`, `none` `2458.75`) — matches
`GurobipySolver`'s solve exactly. `build_rooted_junction_tree`/`compile_diagram` both succeed on it
(max cluster size 81, well under the server-side cap); `tests/test_graphical_demo.py` (3 tests,
`graphical`, `gurobipy` on the one that solves) checks compile success and the hand-computed cross-check, not exhaustive
compiler coverage — matching this task's own "doesn't need to be exhaustively tested like a
production compiler fixture" scoping.

**2. Compile + solve mechanism.** Two small additions to `desdeo/api/routers/graphical.py`, both
deliberately minimal and demo-scoped rather than general-purpose:

- `POST /graphical/demo/irrigation` (no body): builds `build_irrigation_demo()`'s network/
  annotation server-side and delegates straight to `compile_graphical_model` — an ordinary Python
  function call (`compile_graphical_model(request, user, db_session)`), not a second HTTP round
  trip, so persistence stays exactly `/compile`'s own atomic-commit path rather than a parallel
  copy of it. Chosen over having the webui construct the `DiscreteNetwork`/`DecisionAnnotation` JSON
  payload itself (the plan's other suggested mechanism): the network's CPT/utility tensors are
  large, order-sensitive nested arrays, and hand-duplicating that construction in TypeScript purely
  for a demo is exactly the kind of easy-to-get-subtly-wrong duplication this task's own framing
  warned about — a thin server-side endpoint reusing the real compiler is both less code and less
  risk than a client-side reimplementation. Also avoids touching `db_init.py`/seed scripts, per the
  plan's own "avoids touching db_init.py" framing for the lazy-compile option. The webui calls this
  lazily, once, the first time the dialog opens, and caches the resulting `problem_id` in
  `sessionStorage` (`CUMULUS/+page.svelte`'s `ensure_network_demo_compiled`) so repeat opens within
  the same tab don't recompile.
- `POST /graphical/{problem_id}/solve` (body: `{objective_symbol}`): the "small solve step" the
  task called for — no prior endpoint solves an arbitrary compiled graphical `Problem` and hands
  back variable values outside a method-specific router. Reconstructs the pydantic `Problem` via
  `Problem.from_problemdb` (the same reverse conversion `cumulus.py`'s own background payoff-table
  task already uses), resolves a solver via `guess_best_solver`/`find_compatible_solvers` (per the
  task's own instruction, not a hardcoded `GurobipySolver` import, even though Phase 4's own design
  decision means it resolves to `GurobipySolver` for every diagram this compiler can produce
  today), solves for the one requested objective, and returns `SolverResults.optimal_variables`
  directly — already the exact shape `POST /graphical/{id}/policy`'s `variable_values` field
  expects, confirmed by round-tripping it through `/policy` in
  `test_solve_profit_picks_heavy_irrigation_and_feeds_policy`.

  **Worth keeping despite `/whatif`'s own live-engine `policy` already being correct for this
  single-objective diagram.** With only one `ObjectiveNode`, pyAgrum's own ID-solve (behind
  `/whatif`'s `_read_policy`) and this endpoint's MILP solve necessarily agree — confirmed directly
  by the same test asserting both `/policy`'s and `/whatif`'s `policy` equal
  `{"irrigate": "heavy"}` from one solve. The solve/`/policy` path is still real, load-bearing
  wiring, not redundant: it's what actually exercises `policySource`/`fetchPolicy` (phase E's own
  "read decision policies out of solutions via `IndexMap`" checklist item) against a live app for
  the first time — before this phase, that whole code path was verified only by `vitest`/reading
  code, per phase E's own "not done" note. A would-be *two*-objective demo (closer to the original
  `sample-network.ts` story) would have made this distinction matter for real, since `/whatif`'s
  policy would then reflect pyAgrum's implicit utility-summing across both objectives rather than
  either one alone — recorded here in case a future phase revisits the single-objective scope cut
  above and needs the reasoning.

**3. The swap.** `CUMULUS/+page.svelte` now imports `NetworkPanel` instead of `NetworkView` +
`sample-network.ts`'s mock helpers. Deleted outright (dead now that `NetworkPanel` owns the job):
`network_posteriors`/`network_committed_evidence`/`network_committed_posteriors`/
`network_ghost_posteriors`/`network_stale`, `fetch_mock_whatif`, `network_whatif_runner` and its
two `$effect`s, `handle_network_commit`, `NETWORK_DEMO_POLICY` — roughly 90 lines net. What's left
is genuinely page-owned: the decision-mode toggle, the flag-button `nodeChrome` (phase B), the
phase C cross-highlight demo (kept — see below), and the new lazy-compile/solve bookkeeping.
`<NetworkPanel>` is now only mounted `{#if network_problem_id !== null}` — conditionally rendering
it (rather than keeping it always-mounted and resetting its internal state manually the way the old
mock-backed version had to on every dialog close) means every dialog reopen is a fresh mount, so
`NetworkPanel`'s own `loadStructure` reset logic (evidence/posteriors/policy/utilities/commit state,
phase D/E) already gives a clean session for free — the dialog-close `$effect` that used to
manually reset four pieces of mock state now only resets this page's own hover/select/highlight
bookkeeping.

**The phase C cross-highlight "Reference point" demo panel was kept, not deleted as dead
mock-related code — it was never mock-backed to begin with.** It's pure client-side graph
algorithms (`ancestralSet`/`descendants`/`buildAdjacency`) over the loaded model structure, no
inference involved either before or after this phase. The one real friction: `NetworkPanel` owns
`model` as internal state, not a prop, so this page had no way to read the loaded structure for its
own objective list/adjacency. Fixed with one small, additive, callback-shaped prop —
`onStructureLoad?: (model: NetworkModel) => void`, fired once per successful structure fetch
alongside the existing `model = result` assignment — following the same pattern
`onWhatIfResult`/`onCommitEvidence` (phase D/E) already established for "purely additive hook, does
nothing if unset". This is the one deliberate touch to an already-checked-off phase (phase B built
`NetworkPanel`) this task's own instructions allowed for ("where genuinely necessary... note any
such touch") — the alternative (a second, independent structure fetch from `CUMULUS/+page.svelte`
itself, duplicating `NetworkPanel`'s own `defaultFetchStructure`) was rejected as needless
duplication for a one-line addition.

**4. Failure modes.** Compile taking a moment: `network_compiling`, shown as a plain "Compiling
demo diagram…" message in the dialog body before `network_problem_id` is known (there is no
`modelId` for `NetworkPanel` to render anything against yet, so this can't be delegated to it).
Compile failing: `network_compile_error`, shown with a "Retry" button that clears the
`sessionStorage` cache and recompiles — covers both a genuine failure and a stale cached
`problem_id` from a since-reset local dev DB (a "Recompile demo" button in the dialog's toolbar does
the same reset unconditionally, for the case where the cached id is stale but no error has
surfaced yet, e.g. `/structure` 404ing inside `NetworkPanel` rather than the compile call itself
failing). Solve failing: non-fatal, `network_solve_error` shown as inline text next to the
decision-mode toggle noting Solver mode falls back to `/whatif`'s own live policy — genuinely true,
not just a reassuring message, per point 2 above. `/whatif` erroring during exploration (a 400 from
bad evidence, a transient 500): already `NetworkPanel`'s own job (`whatIfError`, phase D's non-fatal
banner) — nothing added here, per this task's own instruction to reuse it rather than build a second
copy.

**Scope cuts, stated plainly.** No sensitivity tornado / VoI wiring in the demo dialog — Phase E's
own finding already established these are method-level analysis views a host page composes
alongside `NetworkPanel`, not something every embedding needs
(`TornadoChart`/`defaultFetchSensitivity`/`defaultFetchVoi` remain unused by this dialog, same as
before this phase); adding them here would be scope creep on top of "one focused piece of wiring".
No server-side seed step (`db_init.py`) — the lazy-compile mechanism above was judged less invasive,
per point 2's reasoning, and this is a dev-only, `{#if dev}`-gated dialog, not a fixture other tests
or seed data depend on.

**Tests.** Backend: `tests/test_graphical_demo.py` (3 new, `graphical`+`gurobipy`) exercises
`demo.py` directly, per above. `desdeo/api/tests/test_graphical_router.py` grew by 5 tests (marked
consistently with the rest of that module — `gurobipy` only on the ones that actually solve):
`/demo/irrigation` persists the expected structure and rejects guest users; `/solve` picks `heavy`
irrigation for `profit` and round-trips through `/policy` *and* cross-checks against `/whatif`'s own
policy (see point 2); an unknown `objective_symbol` is a 400; a nonexistent `problem_id` is a 404.
`pytest desdeo/api/tests/test_graphical_router.py tests/test_graphical_demo.py`: 50/50 pass.
`just test-api`: 225/225 pass (re-run after the final single-objective revision to `demo.py`/its
tests, so this number reflects the code as it actually ships), no regressions. `pytest -m "graphical
and not gurobipy"`: 79/79 pass. `ruff check`/`ruff format --check` on every file this phase touched:
clean (one real pydocstyle fix along the way -- `tests/test_graphical_demo.py`'s
`_hand_computed_expected_profit` docstring needed a blank line after its summary line, D205/D209).

Frontend: `npx vitest run`: 114/114 pass, unchanged — no pure-logic module touched by this phase
(the changes are in `NetworkPanel.svelte`'s markup/props and `CUMULUS/+page.svelte`'s state, neither
covered by this repo's "no component-testing setup" constraint's usual pure-function escape hatch,
since neither is a new pure module). `npx svelte-check`: confirmed the baseline fresh before
starting (66 errors/13 warnings, matching every prior phase's recorded number) and again after —
one real new-error pair surfaced and was fixed along the way: `let network_model: NetworkModel |
null = $state(null)` (a variable-level type annotation) type-checked as `never` for
`network_model?.nodes` two lines later, while every existing nullable-state declaration in this
package (`NetworkPanel.svelte`'s own `model`/`posteriors`/etc.) uses `$state<T | null>(...)`'s
generic form instead — switching to that form fixed it, and is now the only form used in this file
for a nullable `$state`, matching the rest of the package's own idiom rather than introducing a
second one. `npm run lint`/`prettier --check` remain broken for the same pre-existing,
branch-independent reasons every prior phase recorded (not re-diagnosed here, per this task's own
instruction); `CUMULUS/+page.svelte` was hand-matched to its surrounding style rather than run
through `prettier --write`, per every prior phase's identical precedent for that one file — tried
once, confirmed it reformats large unrelated pre-existing sections of the file exactly as phase D's
own findings already documented, and reverted.

**Not exercised, and why.** No live check that a *second* browser tab (or a fresh `sessionStorage`)
correctly triggers a brand-new compile rather than colliding with another tab's cached id — the
cache key has no per-tab/per-session scoping beyond `sessionStorage`'s own browser-tab isolation,
which is exactly the intended behaviour (two tabs get two independently compiled demo problems), but
this wasn't driven end-to-end in a running browser, only read and reasoned through; no live exercise
of the "Recompile demo" button's actual network calls (reasoned through, not clicked in a running
app) — this phase's verification was `vitest`/`svelte-check` plus the backend's own API tests, the
same standard phase D/E's own "no live compiled diagram" findings held to before this phase existed
to remove that limitation; a follow-up pass driving the dialog live end-to-end in a browser (the
`webui-testing` skill phase B used) would be worth doing before this dialog's contract is trusted
much further, though its scope is a dev-only preview, not production surface.

**Code review pass.** A review of this phase's diff found seven issues, all fixed.

Two real correctness/robustness gaps in `NetworkPanel.svelte`/the CUMULUS page: opening the demo
dialog straight into `decisionMode: 'solver'` showed no policy checkmark at all until the DM pinned
some evidence -- `policy` was only ever written from the what-if runner's own `onResult`, triggered
by an evidence change; the initial structure-load seed fetched `posteriors` from an evidence-free
response but never routed it through `policySource` at all. Fixed two ways, since in practice
`/solve` resolves after `/structure`, so either alone is insufficient: `loadStructure`'s own
prior-baseline fetch now runs through `fetchWhatIfWithPolicy` (so a `policySource` already set by
load time seeds `policy` immediately, the same call that already seeds `posteriors`), and a new
effect fires once when `policySource` transitions from unset to non-null *after* the initial load
(the CUMULUS dialog's actual ordering), re-running the identical evidence-driven fetch
`handleEvidenceChange` already uses. Separately, a stale/foreign cached `problem_id` in
`sessionStorage` (unscoped to the logged-in user -- a login/logout in the same tab could adopt
another user's cached id, or a local DB reset could leave a dead one) had no automatic recovery
path, only an unrelated "Recompile demo" button the DM had to notice; fixed with a new
`onStructureError` prop on `NetworkPanel` (fired when the structure fetch itself fails, distinct
from a non-fatal `/whatif` failure), which the CUMULUS page uses to drop the cache and recompile
once, automatically, guarded against retrying forever on a persistently failing backend. The cache
key now also includes the logged-in user's own id, narrowing the collision window further.

Two real backend robustness gaps in `graphical.py`: `/graphical/{problem_id}/solve`'s compatibility
check (`find_compatible_solvers`) and the solver it actually used (`guess_best_solver`) were two
different functions with different fallback behavior and no exception handling around the solve
call -- on a machine without a Gurobi license, a compiled (mixed-integer) diagram could land on
`CVXPYSolver` instead of `GurobipySolver`, and if the local CVXPY install lacked a MIP-capable
backend, that raised an unhandled exception escaping as a raw 500. Fixed by trusting
`guess_best_solver`'s guess only when it's actually one of `find_compatible_solvers`'s own vetted
list, falling back to that list's first entry otherwise, and wrapping solver construction/`.solve`
in a `try/except` that reports a solver-side failure as a clean 500 with the solver's own message.
Separately, `/solve` skipped the `_fetch_graphical_metadata` check every sibling
`/graphical/{problem_id}/*` endpoint runs first, despite its own docstring claiming otherwise --
doubling it as an accidental general-purpose solve endpoint for arbitrary visible `ProblemDB` rows,
with two concrete bad outcomes: an ordinary problem with a multi-dimensional `TensorVariable`
produced an `optimal_variables` shape that failed `SolveResponse`'s declared response type
(an opaque `ResponseValidationError` 500), and there was no size/cost bound analogous to
`/compile`'s server-side cluster cap. Fixed by calling `_fetch_graphical_metadata` before solving,
matching every sibling endpoint.

One real resource-accumulation issue: `POST /graphical/demo/irrigation` had no idempotency --
every call (every browser tab, every reload without a live `sessionStorage` cache, every "Recompile
demo" click) persisted a brand-new ~127 KB `Problem` + metadata into the caller's own problem list,
with no cleanup story. Fixed by looking up an existing `ProblemDB` owned by the calling user with
the demo's fixed name before compiling, returning its `problem_id` instead of recompiling unless a
new `force: bool = False` request field says otherwise; the CUMULUS page's "Recompile demo" button
now passes `force: true`, the only caller that wants an unconditional fresh compile.

Two minor issues, both fixed: `reset_network_demo` cleared the cached problem id and error state but
not the loaded `network_model`, so a failed recompile could leave the "(demo) Reference point" panel
showing the *previous* model's objectives next to a fatal error pane -- now cleared alongside the
rest. `NetworkPolicySource.variableValues` (TypeScript) was typed `Record<string, number[]>`,
narrower than the backend's actual `dict[str, list[float] | float]` response shape -- the identical
mismatch Phase E's own review already fixed on the Python side (`PolicyRequest.variable_values`),
reintroduced here on the TypeScript side; widened to `Record<string, number[] | number>`, and the
CUMULUS page's own inline `variable_values` type updated to match. The dialog's description text
said 'pin "Weather" to see "Soil moisture" respond', but `get_structure` renders node labels as the
raw variable name (`weather`/`soil_moisture`) -- fixed the copy. Separately, `demo.py` set a
`description` on each `DiscreteVariable` with no way for it to ever reach the frontend
(`StructureNode` had no such field at all, despite `NetworkModel`'s own node type already having one
ready to receive it) -- added `description` to `StructureNode` and threaded it through
`get_structure`'s chance-node loop (decision/objective nodes have no description field of their own
to thread, so those stay `None`), rather than leaving `demo.py`'s descriptions as unexplained dead
data.

One test-marking issue: `tests/test_graphical_demo.py`'s module-level `pytestmark` marked *all three*
tests `gurobipy`, even though only `test_profit_solve_matches_hand_computed_expectation` actually
solves via `GurobipySolver` -- the other two (`validate_against` succeeding, and the compiler itself
accepting the diagram) never touch gurobipy, so `pytest -m "graphical and not gurobipy"` skipped
tests the plan's own checklist calls out as the ones it actually requires. Fixed by moving to
per-test `gurobipy` marking, matching `test_graphical_router.py`'s own convention. Also corrected
this write-up's own "4 tests" claim above to the actual 3.

**Tests.** `desdeo/api/tests/test_graphical_router.py` grew by 4 (a `Group`-shared, non-owner user
successfully solving -- confirming `/solve` is visibility-, not ownership-, gated, the same pattern
`test_evidence_set_delete_visible_non_owner_is_403` uses; an ordinary non-graphical `ProblemDB`
-- "The river pollution problem", the same fixture `test_structure_404_when_no_graphical_metadata`
uses -- now 404s from `/solve` the same way every sibling endpoint already does; two `/demo/
irrigation` idempotency tests, same-user-same-`problem_id` and `force: true` returning a different
one). `pytest desdeo/api/tests/test_graphical_router.py -q`: 51/51 pass. `just test-api`: 229/229
pass (up from 225), no regressions. `pytest -m graphical tests/`: 78/78 pass. `ruff check` on every
file touched: clean. `npx vitest run`: 114/114 pass, unchanged. `npx svelte-check`: 66 errors/13
warnings, exactly the pre-existing baseline -- confirmed none trace to `NetworkPanel.svelte`,
`CUMULUS/+page.svelte`, or `network/types.ts`.

---

## Cross-cutting

**Package layout**

```
desdeo/problem/graphical/
    network.py                DiscreteNetwork, DiscreteVariable, Potential
    schema.py                 DecisionAnnotation and friends (renamed from annotation.py, to
                               match the rest of the codebase's `schema.py` convention)
    rooted_junction_tree.py   RootedJunctionTree construction (renamed from rjt.py)
    compile.py                MILP generation, IndexMap
    risk.py                   diagram/tree transformations
    adapters/
        pyagrum.py             optional; import, learning, inference oracle
```

Dependency rule: `network`, `schema`, `rooted_junction_tree`, `compile`, `risk` never import pyAgrum.
The interesting algorithmic code stays testable without a C++ extension in the loop.

**Risks**

| Risk | Mitigation |
|---|---|
| Cluster moments exceed 2 tensor dimensions | Flattened mixed-radix encoding; Phase 0 fixes the scheme before anything depends on it |
| Gradual RJT construction harder than expected | Timebox; fall back to hand-specified cluster trees for early models |
| Learned networks have unusable treewidth | Guard in phase 3, surfaced in the UI before compile |
| Web-API churn | Self-contained router and schemas |
| pyAgrum utility-summing semantics | Verify in phase 2, before anything depends on it |
| In-memory engine cache doesn't share across workers | Decide sticky sessions vs. an external cache before multi-worker deployment, or document the single-worker limitation explicitly (phase 7) |

**Sequencing.** Phase 0 no longer blocks — run it alongside Phase 1, since its output is a
threshold for Phase 3 rather than a go/no-go. UI phase A blocks nothing and can start
immediately. Phases C and D need backend phase 7. Phase E needs backend phase 4.
Getting the embedding contract right against a real method UI (phase B) before the inference
plumbing exists is the point of ordering the UI this way.

**Talk to the maintainers early.** The web-API has a dedicated tracking issue and the
maintainers explicitly invite contributors on Discord. Probabilistic modelling of uncertainty
is close enough to the group's scenario-based MOO work that parallel efforts may already exist.
