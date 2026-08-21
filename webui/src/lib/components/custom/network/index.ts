/** Public surface of the network visualisation package. No imports from any method store —
 *  that constraint is what keeps this reusable across method UIs (see the plan, frontend
 *  phase B). Everything downstream should import from here rather than reaching into
 *  individual files. */

export { default as NetworkView } from './NetworkView.svelte';
export { default as NetworkTable } from './NetworkTable.svelte';

export type {
	NetworkModel,
	NetworkNodeModel,
	NetworkEdgeModel,
	NetworkNodeKind,
	NetworkLayoutModel,
	NetworkLayoutPosition
} from './types';

export type { DetailLevel, HighlightState, NetworkFlowNode, NetworkFlowEdge } from './view-types';

export { NODE_KIND_STYLE } from './node-style';
export type { NodeShape, NodeKindStyle } from './node-style';

export { computeElkLayout, layoutCoversModel } from './layout';
export type { ElkLayoutOptions } from './layout';

export {
	ancestors,
	descendants,
	ancestralSet,
	dSeparated,
	activeTrailNodes,
	nodeById,
	incidentEdges
} from './dag-algorithms';

export { computeDetailLevel } from './lod';
