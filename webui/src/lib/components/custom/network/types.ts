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

/** Hard evidence only for now: node id -> the single state label pinned as observed/set. Soft
 *  evidence (likelihoods) and an explicit observe-vs-intervene mode are phase D's job — this is
 *  just the wire shape phase B commits to so callers can start binding against it. */
export type NetworkEvidence = Record<string, string>;

/** Node id -> per-state probability, aligned index-for-index with that node's `states`. A missing
 *  entry (or the whole map being absent) means "no posterior available yet" — e.g. before any
 *  evidence has been set — not "all zero". */
export type NetworkPosteriors = Record<string, number[]>;

/** Decision node id -> the state a solved policy chose for it. */
export type NetworkPolicy = Record<string, string>;

/** Shape of a `POST /graphical/whatif` response (backend phase 7, not built yet) — what
 *  `<NetworkPanel>`'s `fetchWhatIf` resolves to. */
export interface WhatIfResult {
	posteriors: NetworkPosteriors;
	policy?: NetworkPolicy;
}
