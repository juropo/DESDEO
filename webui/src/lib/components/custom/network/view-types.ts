/** Types for the Svelte-Flow-facing side of `<NetworkView>` — kept separate from `types.ts`
 *  (the model's JSON contract) since these describe rendering state, not the model itself. */

import type { Edge, Node } from '@xyflow/svelte';
import type { NetworkNodeKind } from './types';

/** How much a node renders: label only, label + state count, or label + full state list.
 *  Driven by container width *and* zoom — see `computeDetailLevel` in `lod.ts`. */
export type DetailLevel = 'compact' | 'normal' | 'expanded';

/** Cross-highlight state for a node — see phase C (ancestor/active-trail highlighting) and
 *  `dag-algorithms.ts`. Phase A only ever sets `'normal'`; the field exists now so phase C
 *  doesn't have to touch every node component again. */
export type HighlightState = 'normal' | 'dimmed' | 'active';

/** Data payload carried on every Svelte Flow node (`Node<NetworkNodeData>`). */
export interface NetworkNodeData extends Record<string, unknown> {
	kind: NetworkNodeKind;
	label: string;
	states?: string[];
	description?: string;
	detail: DetailLevel;
	highlight: HighlightState;
}

export type NetworkFlowNode = Node<NetworkNodeData>;
export type NetworkFlowEdge = Edge;
