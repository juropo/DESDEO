/**
 * Client-side graph algorithms over `NetworkModel` — ancestor sets, d-separation, and active-trail
 * reachability. This is the "libdagitty" line item in the graphical-models plan's frontend phase A:
 * there is no npm package by that name (dagitty.js was never published), so this reimplements the
 * relevant graph theory directly rather than wrapping something that doesn't exist as a dependency.
 *
 * All functions are pure and operate only on the structure (`nodes`/`edges`) of a `NetworkModel` —
 * no inference, no backend. That's what makes them usable in phase A, before any compiler exists.
 */

import type { NetworkEdgeModel, NetworkModel, NetworkNodeModel } from './types';

interface Adjacency {
	parents: Map<string, string[]>;
	children: Map<string, string[]>;
}

function buildAdjacency(model: Pick<NetworkModel, 'nodes' | 'edges'>): Adjacency {
	const parents = new Map<string, string[]>();
	const children = new Map<string, string[]>();
	for (const node of model.nodes) {
		parents.set(node.id, []);
		children.set(node.id, []);
	}
	for (const edge of model.edges) {
		children.get(edge.source)?.push(edge.target);
		parents.get(edge.target)?.push(edge.source);
	}
	return { parents, children };
}

/** All strict ancestors of `nodeId` (not including `nodeId` itself). */
export function ancestors(
	model: Pick<NetworkModel, 'nodes' | 'edges'>,
	nodeId: string
): Set<string> {
	const { parents } = buildAdjacency(model);
	const seen = new Set<string>();
	const stack = [...(parents.get(nodeId) ?? [])];
	while (stack.length > 0) {
		const id = stack.pop()!;
		if (seen.has(id)) continue;
		seen.add(id);
		stack.push(...(parents.get(id) ?? []));
	}
	return seen;
}

/** All strict descendants of `nodeId` (not including `nodeId` itself). */
export function descendants(
	model: Pick<NetworkModel, 'nodes' | 'edges'>,
	nodeId: string
): Set<string> {
	const { children } = buildAdjacency(model);
	const seen = new Set<string>();
	const stack = [...(children.get(nodeId) ?? [])];
	while (stack.length > 0) {
		const id = stack.pop()!;
		if (seen.has(id)) continue;
		seen.add(id);
		stack.push(...(children.get(id) ?? []));
	}
	return seen;
}

/** Union of `nodeIds` with every one of their ancestors (including `nodeIds` themselves). */
export function ancestralSet(
	model: Pick<NetworkModel, 'nodes' | 'edges'>,
	nodeIds: Iterable<string>
): Set<string> {
	const result = new Set(nodeIds);
	for (const id of [...result]) {
		for (const a of ancestors(model, id)) result.add(a);
	}
	return result;
}

/**
 * Tests whether `x` and `y` are d-separated given `z`, via the standard "moralize the ancestral
 * graph" construction: restrict to the ancestral closure of x ∪ y ∪ z, marry parents of every
 * node in that closure (moralization), drop z, and check reachability between x and y in the
 * resulting undirected graph. No path ⇒ d-separated.
 */
export function dSeparated(
	model: Pick<NetworkModel, 'nodes' | 'edges'>,
	x: string[],
	y: string[],
	z: string[] = []
): boolean {
	const zSet = new Set(z);
	const closure = ancestralSet(model, [...x, ...y, ...z]);
	const { parents } = buildAdjacency(model);

	// Undirected adjacency restricted to `closure`, plus moralization edges between every pair of
	// a common child's parents that both lie in `closure`.
	const moral = new Map<string, Set<string>>();
	const link = (a: string, b: string) => {
		if (a === b) return;
		if (!moral.has(a)) moral.set(a, new Set());
		if (!moral.has(b)) moral.set(b, new Set());
		moral.get(a)!.add(b);
		moral.get(b)!.add(a);
	};
	for (const edge of model.edges) {
		if (closure.has(edge.source) && closure.has(edge.target)) link(edge.source, edge.target);
	}
	for (const node of closure) {
		const ps = (parents.get(node) ?? []).filter((p) => closure.has(p));
		for (let i = 0; i < ps.length; i++) {
			for (let j = i + 1; j < ps.length; j++) link(ps[i], ps[j]);
		}
	}

	// BFS from x to y, never stepping through a node in z.
	const visited = new Set<string>(x.filter((id) => !zSet.has(id)));
	const queue = [...visited];
	while (queue.length > 0) {
		const cur = queue.shift()!;
		if (y.includes(cur)) return false;
		for (const next of moral.get(cur) ?? []) {
			if (visited.has(next) || zSet.has(next)) continue;
			visited.add(next);
			queue.push(next);
		}
	}
	return true;
}

type TrailDirection = 'up' | 'down';

/**
 * Nodes reachable from `sourceIds` via an active trail given evidence `evidenceIds` — the
 * "Reachable" algorithm (Koller & Friedman, *Probabilistic Graphical Models*, alg. 3.1 /
 * Shachter's Bayes-Ball). Used to drive active-trail highlighting: hovering a node shows every
 * node it can currently influence or be influenced by, given whatever evidence is set.
 *
 * Source nodes are included in the result (unless they are themselves evidence).
 */
export function activeTrailNodes(
	model: Pick<NetworkModel, 'nodes' | 'edges'>,
	sourceIds: string[],
	evidenceIds: string[] = []
): Set<string> {
	const { parents, children } = buildAdjacency(model);
	const z = new Set(evidenceIds);

	// Phase 1: ancestors of z (needed to decide whether a v-structure is "opened" by evidence
	// on one of its descendants).
	const ancestorsOfZ = new Set<string>();
	{
		const toVisit = [...z];
		while (toVisit.length > 0) {
			const node = toVisit.pop()!;
			if (ancestorsOfZ.has(node)) continue;
			ancestorsOfZ.add(node);
			toVisit.push(...(parents.get(node) ?? []));
		}
	}

	// Phase 2: traverse active trails from the source set.
	const reachable = new Set<string>();
	const visited = new Set<string>(); // "node|direction" pairs already processed
	const queue: [string, TrailDirection][] = sourceIds.map((id) => [id, 'up']);

	while (queue.length > 0) {
		const [node, direction] = queue.pop()!;
		const key = `${node}|${direction}`;
		if (visited.has(key)) continue;
		visited.add(key);

		if (!z.has(node)) reachable.add(node);

		if (direction === 'up' && !z.has(node)) {
			// Trail continues upward through a parent (chain/fork, not blocked), and downward
			// into a child (fork tail).
			for (const p of parents.get(node) ?? []) queue.push([p, 'up']);
			for (const c of children.get(node) ?? []) queue.push([c, 'down']);
		} else if (direction === 'down') {
			if (!z.has(node)) {
				// Chain/fork continues through children.
				for (const c of children.get(node) ?? []) queue.push([c, 'down']);
			}
			if (ancestorsOfZ.has(node)) {
				// v-structure at `node` is opened (node or a descendant is evidence): trail can
				// continue back up through its parents.
				for (const p of parents.get(node) ?? []) queue.push([p, 'up']);
			}
		}
	}
	return reachable;
}

/** Convenience: node lookup by id, throwing if not found — used by table/canvas views that
 *  already assume a valid model. */
export function nodeById(model: Pick<NetworkModel, 'nodes'>, id: string): NetworkNodeModel {
	const node = model.nodes.find((n) => n.id === id);
	if (!node) throw new Error(`NetworkModel has no node with id "${id}"`);
	return node;
}

/** Edges incident to `nodeId`, either direction. */
export function incidentEdges(
	model: Pick<NetworkModel, 'edges'>,
	nodeId: string
): NetworkEdgeModel[] {
	return model.edges.filter((e) => e.source === nodeId || e.target === nodeId);
}
