import type { NodeTypes } from '@xyflow/svelte';
import GraphNode from './GraphNode.svelte';

export { GraphNode };

/** `node.type` -> component. All four kinds render through the same `GraphNode`, which reads its
 *  own `type` prop to look up `NODE_KIND_STYLE` (shape/color/icon) -- see `GraphNode.svelte`. */
export const NETWORK_NODE_TYPES: NodeTypes = {
	chance: GraphNode,
	decision: GraphNode,
	objective: GraphNode,
	deterministic: GraphNode
};
