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
