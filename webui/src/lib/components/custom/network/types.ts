/**
 * Frontend-only JSON contract for `<NetworkView>` (graphical models plan, frontend phase A).
 *
 * Deliberately independent of the eventual pydantic/OpenAPI shape: `DiscreteNetwork` and
 * `DecisionAnnotation` (backend phases 1-2, `desdeo/problem/graphical/`) land on a separate
 * track, and UI phase A is meant to ship before that schema is final. When the generated
 * OpenAPI client exists, an adapter maps it onto this shape — this shape does not change to
 * match whatever the backend happens to emit. See `.claude/desdeo-graphical-models-plan.md`.
 */

/** Node categories the component renders distinctly. `deterministic` covers functional/derived
 *  nodes (fixed given their parents, no CPT) as used in influence diagrams. */
export type NetworkNodeKind = 'chance' | 'decision' | 'objective' | 'deterministic';

export interface NetworkNodeModel {
	/** Stable id, unique within the model. Matches the backend symbol/name once one exists. */
	id: string;
	/** Display label. */
	label: string;
	kind: NetworkNodeKind;
	/** Discrete state labels. Absent for `objective` nodes, which carry a scalar utility rather
	 *  than a state space. */
	states?: string[];
	description?: string;
}

export interface NetworkEdgeModel {
	id: string;
	/** `NetworkNodeModel.id` this edge starts from. */
	source: string;
	/** `NetworkNodeModel.id` this edge points to. */
	target: string;
}

export interface NetworkLayoutPosition {
	x: number;
	y: number;
}

export interface NetworkLayoutModel {
	/** Node id -> canvas position. Computed once (elkjs) and persisted with the model —
	 *  re-running layout on every mount destroys spatial memory across method UIs. */
	positions: Record<string, NetworkLayoutPosition>;
	/** Informational only; not used for cache invalidation. */
	computedAt?: string;
}

export interface NetworkModel {
	nodes: NetworkNodeModel[];
	edges: NetworkEdgeModel[];
	/** Absent (or missing entries for some nodes) triggers an automatic one-off elk layout;
	 *  see `computeElkLayout` in `layout.ts`. */
	layout?: NetworkLayoutModel;
}

/**
 * Embedding contract additions (graphical models plan, frontend phase B). These are the wire
 * shapes a method UI (or the eventual `<NetworkPanel>` fetch wrapper) binds against — deliberately
 * committed to now, ahead of backend phase 7, the same way `NetworkModel` above was committed to
 * ahead of backend phases 1-2. See `.claude/desdeo-graphical-models-plan.md`.
 */

/** Governs whether decision nodes are read-only outputs of a solved policy ('solver') or directly
 *  settable by the DM ('manual') — see the plan's frontend phase B/E split. */
export type DecisionMode = 'solver' | 'manual';

/**
 * Hard evidence only: node id -> the single state label pinned as observed. This is the *only*
 * evidence mode the real backend supports (phase D, reconciled against `desdeo/api/routers/
 * graphical.py`, backend phase 7): `WhatIfRequest.evidence: dict[str, str]` feeds straight into
 * pyAgrum's `setEvidence`, which is conditioning/observation only — there is no soft-evidence
 * (likelihood) parameter and no do-operator intervention anywhere in that endpoint. Building a
 * soft-evidence or intervene affordance in this component would silently promise a capability the
 * backend can't perform, so phase D deliberately does not add one; `NetworkView`'s evidence note
 * (see `EVIDENCE_MODE_NOTE` in `interaction.ts`) says so explicitly in the UI instead of leaving
 * it to be inferred. Revisit this type (and add an explicit mode discriminator) only once/if the
 * backend actually grows a second evidence kind.
 */
export type NetworkEvidence = Record<string, string>;

/** Node id -> per-state probability, aligned index-for-index with that node's `states`. A missing
 *  entry (or the whole map being absent) means "no posterior available yet" — e.g. before any
 *  evidence has been set — not "all zero". */
export type NetworkPosteriors = Record<string, number[]>;

/** Decision node id -> the state a solved policy chose for it. */
export type NetworkPolicy = Record<string, string>;

/** Objective node id -> expected utility under the currently applied evidence (`WhatIfResponse.
 *  utilities`, keyed by node id the same way `NetworkPosteriors`/`NetworkPolicy` are — see
 *  `defaultFetchWhatIf` in `NetworkPanel.svelte` for the `util_<symbol>` remap this requires
 *  against the real endpoint). Not rendered inline on the (deliberately posterior-free, per phase
 *  C) objective node shape — threaded through to a caller via `NetworkPanel`'s `onWhatIfResult`
 *  instead, for a method UI's own objective-value display to consume. */
export type NetworkUtilities = Record<string, number>;

/** Objective node id -> standard deviation of the utility distribution under the currently
 *  applied evidence (`WhatIfResponse.utility_std_devs`, phase E: "MEU with its standard
 *  deviation"). Same keying convention as `NetworkUtilities` — the two are always fetched, and
 *  remapped from `util_<symbol>`, together (see `defaultFetchWhatIf`). Not rendered on the graph
 *  itself for the same reason `NetworkUtilities` isn't (phase C: objective nodes stay
 *  posterior-free) — surfaced only via `NetworkPanel`'s `onWhatIfResult`. */
export type NetworkUtilityStdDevs = Record<string, number>;

/** Shape of a `POST /graphical/whatif` response (backend phase 7; see `WhatIfResponse` in
 *  `desdeo/api/routers/graphical.py`) — what `<NetworkPanel>`'s `fetchWhatIf` resolves to. */
export interface WhatIfResult {
	posteriors: NetworkPosteriors;
	policy?: NetworkPolicy;
	utilities?: NetworkUtilities;
	utilityStdDevs?: NetworkUtilityStdDevs;
}

/**
 * Phase E: reads a *solved* `Problem`'s policy back (via `POST /graphical/{id}/policy`), as
 * distinct from `WhatIfResult.policy` (a live pyAgrum engine's evidence-conditioned posterior —
 * see `desdeo/api/routers/graphical.py`'s own module docstring for why these are two different,
 * complementary sources of truth for the same `NetworkPolicy` shape). Exactly one of
 * `variableValues`/`solutionId` identifies which solved solution to decode.
 *
 * `variableValues` is `Record<string, number[] | number>`, not just `Record<string, number[]>` —
 * matching `PolicyRequest.variable_values`'s own declared `dict[str, list[float] | float]` shape
 * (`desdeo/api/routers/graphical.py`). A review pass caught this narrower type reintroducing, on
 * the TypeScript side, the identical mismatch Phase E's own review already fixed on the Python
 * side: a diagram compiled with a `RiskMeasure` has a scalar `eta` `Variable` in a solved
 * `Problem`'s `optimal_variables` (a bare number, not a length-1 array) alongside every decision's
 * own list-valued `z_symbol`.
 */
export type NetworkPolicySource =
	| { variableValues: Record<string, number[] | number> }
	| { solutionId: number };

/** A named hard-evidence set saved server-side (`GET/POST/DELETE /graphical/.../evidence-sets`,
 *  phase E) — selectable as a comparison baseline (via `NetworkView`'s existing
 *  `ghostPosteriors`/`committedEvidence`/`onCommitEvidence` machinery, see `NetworkPanel`'s
 *  `loadEvidenceSetAsBaseline`) and addressable/shareable by its own `id`. */
export interface NetworkEvidenceSet {
	id: number;
	problemId: number;
	name: string;
	evidence: NetworkEvidence;
	createdAt: string;
}
