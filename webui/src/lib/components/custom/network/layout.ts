/**
 * elkjs layout helper. Computed once (on demand, via the "re-layout" action) and persisted with
 * the model — re-running this on every mount would destroy spatial memory across method UIs, so
 * callers are expected to cache the result on `NetworkModel.layout` rather than call this on mount
 * whenever a persisted layout already exists.
 */

import ELK from 'elkjs/lib/elk.bundled.js';
import type { NetworkLayoutModel, NetworkModel } from './types';

const elk = new ELK();

export interface ElkLayoutOptions {
	direction?: 'DOWN' | 'RIGHT' | 'UP' | 'LEFT';
	nodeWidth?: number;
	nodeHeight?: number;
	spacingNodeNode?: number;
	spacingLayer?: number;
}

const DEFAULT_OPTIONS: Required<ElkLayoutOptions> = {
	direction: 'DOWN',
	nodeWidth: 180,
	nodeHeight: 64,
	spacingNodeNode: 48,
	spacingLayer: 96
};

/** Runs elk's layered algorithm over `model`'s structure and returns fresh node positions.
 *  Does not read or write `model.layout` — that's the caller's job (see `NetworkView`'s
 *  re-layout action), keeping this function a pure structure -> positions map. */
export async function computeElkLayout(
	model: Pick<NetworkModel, 'nodes' | 'edges'>,
	options: ElkLayoutOptions = {}
): Promise<NetworkLayoutModel> {
	const opts = { ...DEFAULT_OPTIONS, ...options };

	const graph = {
		id: 'root',
		layoutOptions: {
			'elk.algorithm': 'layered',
			'elk.direction': opts.direction,
			'elk.spacing.nodeNode': String(opts.spacingNodeNode),
			'elk.layered.spacing.nodeNodeBetweenLayers': String(opts.spacingLayer),
			'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
			'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP'
		},
		children: model.nodes.map((n) => ({
			id: n.id,
			width: opts.nodeWidth,
			height: opts.nodeHeight
		})),
		edges: model.edges.map((e) => ({ id: e.id, sources: [e.source], targets: [e.target] }))
	};

	const result = await elk.layout(graph);

	const positions: NetworkLayoutModel['positions'] = {};
	for (const child of result.children ?? []) {
		positions[child.id] = { x: child.x ?? 0, y: child.y ?? 0 };
	}
	return { positions, computedAt: new Date().toISOString() };
}

/** True when `layout` has a position for every node in `model` — i.e. it's safe to render
 *  without falling back to an automatic layout pass first. */
export function layoutCoversModel(
	model: Pick<NetworkModel, 'nodes'>,
	layout: NetworkLayoutModel | undefined
): boolean {
	if (!layout) return false;
	return model.nodes.every((n) => layout.positions[n.id] !== undefined);
}
