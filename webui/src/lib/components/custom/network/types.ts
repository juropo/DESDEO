/**
 * Frontend JSON contract for `<NetworkView>`. Independent of the backend's schema: backend
 * responses are mapped onto these shapes (see `NetworkPanel.svelte`).
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

/** Shapes a method UI or `<NetworkPanel>` binds against when embedding a network. */

/** Whether decision nodes are read-only outputs of a solved policy ('solver') or directly
 *  settable by the DM ('manual'). */
export type DecisionMode = 'solver' | 'manual';

/**
 * Hard evidence: node id -> the state label observed. The backend supports neither soft evidence
 * nor interventions (see `EVIDENCE_MODE_NOTE` in `interaction.ts`).
 */
export type NetworkEvidence = Record<string, string>;

/** Node id -> per-state probability, aligned index-for-index with that node's `states`. A missing
 *  entry (or the whole map being absent) means "no posterior available yet" — e.g. before any
 *  evidence has been set — not "all zero". */
export type NetworkPosteriors = Record<string, number[]>;

/** Decision node id -> the state a solved policy chose for it. */
export type NetworkPolicy = Record<string, string>;

/** Objective node id -> expected utility under the current evidence. Not rendered on the graph;
 *  passed to the host via `NetworkPanel`'s `onWhatIfResult`. */
export type NetworkUtilities = Record<string, number>;

/** Objective node id -> standard deviation of the utility under the current evidence. Same
 *  keying and delivery as `NetworkUtilities`. */
export type NetworkUtilityStdDevs = Record<string, number>;

/** A `POST /graphical/whatif` response, as `<NetworkPanel>`'s `fetchWhatIf` resolves it. */
export interface WhatIfResult {
	posteriors: NetworkPosteriors;
	policy?: NetworkPolicy;
	utilities?: NetworkUtilities;
	utilityStdDevs?: NetworkUtilityStdDevs;
	/** The problem's ideal/nadir, keyed by objective symbol (not node id). Only set for
	 *  `commit: true` requests. */
	ideal?: Record<string, number> | null;
	nadir?: Record<string, number> | null;
}

/**
 * Which solved solution to decode a policy from (via `POST /graphical/{id}/policy`), as opposed
 * to `WhatIfResult.policy`, which comes from live inference. `variableValues` may contain scalars
 * (e.g. a `RiskMeasure`'s `eta`).
 */
export type NetworkPolicySource =
	| { variableValues: Record<string, number[] | number> }
	| { solutionId: number };

/** A named evidence set saved server-side, usable as a comparison baseline (see `NetworkPanel`'s
 *  `loadEvidenceSetAsBaseline`) and shareable by `id`. */
export interface NetworkEvidenceSet {
	id: number;
	problemId: number;
	name: string;
	evidence: NetworkEvidence;
	createdAt: string;
}
