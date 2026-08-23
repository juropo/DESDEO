/** Types for the Svelte-Flow-facing side of `<NetworkView>` — kept separate from `types.ts`
 *  (the model's JSON contract) since these describe rendering state, not the model itself. */

import type { Snippet } from 'svelte';
import type { Edge, Node } from '@xyflow/svelte';
import type { NetworkNodeKind } from './types';

/** How much a node renders: label only, label + state count, or label + full state list.
 *  Driven by container width *and* zoom — see `computeDetailLevel` in `lod.ts`. */
export type DetailLevel = 'compact' | 'normal' | 'expanded';

/** Cross-highlight state for a node — see phase C (ancestor/active-trail highlighting) and
 *  `dag-algorithms.ts`. Phase A only ever sets `'normal'`; the field exists now so phase C
 *  doesn't have to touch every node component again. */
export type HighlightState = 'normal' | 'dimmed' | 'active';

/** Context passed to the `nodeChrome` snippet (phase B) — just enough for a method UI to decide
 *  what badge, if any, to render for a given node, without needing the whole `NetworkNodeModel`. */
export interface NetworkChromeContext {
	id: string;
	kind: NetworkNodeKind;
	label: string;
}

/** Data payload carried on every Svelte Flow node (`Node<NetworkNodeData>`). */
export interface NetworkNodeData extends Record<string, unknown> {
	kind: NetworkNodeKind;
	label: string;
	states?: string[];
	description?: string;
	detail: DetailLevel;
	highlight: HighlightState;
	/** Whether clicking one of this node's states sets/clears hard evidence on it — true for
	 *  chance nodes always, and for decision nodes only in `decisionMode: 'manual'`. See
	 *  `isNodeEditable` in `interaction.ts`. */
	editable: boolean;
	/** This node's currently pinned hard-evidence state, if any (phase B contract; see
	 *  `NetworkEvidence`). */
	evidenceState?: string;
	/** This node's state under the currently solved policy — decision nodes only, and only
	 *  meaningful in `decisionMode: 'solver'` (see `NetworkPolicy`). */
	policyState?: string;
	/** Per-state posterior probabilities, aligned with `states` (see `NetworkPosteriors`). */
	posterior?: number[];
	/** Fired when an editable state is clicked; `NetworkView` supplies a closure bound to this
	 *  node's id. */
	onStateClick?: (state: string) => void;
	/** Per-node chrome injected by the embedding method UI — see the `nodeChrome` prop on
	 *  `NetworkView`. */
	chrome?: Snippet<[NetworkChromeContext]>;
}

export type NetworkFlowNode = Node<NetworkNodeData>;
export type NetworkFlowEdge = Edge;
