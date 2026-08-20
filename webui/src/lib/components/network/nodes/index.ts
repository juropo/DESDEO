import type { NodeTypes } from '@xyflow/svelte';
import ChanceNode from './ChanceNode.svelte';
import DecisionNode from './DecisionNode.svelte';
import ObjectiveNode from './ObjectiveNode.svelte';
import DeterministicNode from './DeterministicNode.svelte';

export { ChanceNode, DecisionNode, ObjectiveNode, DeterministicNode };

/** `node.type` -> component, keyed to match `NetworkNodeKind` exactly so building this map from
 *  a `NetworkModel` is a direct lookup (see `NetworkView.svelte`). */
export const NETWORK_NODE_TYPES: NodeTypes = {
	chance: ChanceNode,
	decision: DecisionNode,
	objective: ObjectiveNode,
	deterministic: DeterministicNode
};
