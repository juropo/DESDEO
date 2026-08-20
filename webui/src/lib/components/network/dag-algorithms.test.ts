import { describe, expect, it } from 'vitest';
import { activeTrailNodes, ancestors, dSeparated, descendants } from './dag-algorithms';
import type { NetworkModel } from './types';

function model(edges: [string, string][], extraNodeIds: string[] = []): NetworkModel {
	const ids = new Set<string>(extraNodeIds);
	for (const [s, t] of edges) {
		ids.add(s);
		ids.add(t);
	}
	return {
		nodes: [...ids].map((id) => ({ id, label: id, kind: 'chance' as const })),
		edges: edges.map(([source, target]) => ({ id: `${source}->${target}`, source, target }))
	};
}

describe('ancestors / descendants', () => {
	// Diamond: A -> B -> D, A -> C -> D
	const diamond = model([
		['A', 'B'],
		['A', 'C'],
		['B', 'D'],
		['C', 'D']
	]);

	it('collects all strict ancestors', () => {
		expect(ancestors(diamond, 'D')).toEqual(new Set(['A', 'B', 'C']));
		expect(ancestors(diamond, 'A')).toEqual(new Set());
	});

	it('collects all strict descendants', () => {
		expect(descendants(diamond, 'A')).toEqual(new Set(['B', 'C', 'D']));
		expect(descendants(diamond, 'D')).toEqual(new Set());
	});
});

describe('dSeparated', () => {
	it('a chain is blocked by evidence on the middle node', () => {
		// A -> B -> C
		const chain = model([
			['A', 'B'],
			['B', 'C']
		]);
		expect(dSeparated(chain, ['A'], ['C'], [])).toBe(false); // active: unblocked chain
		expect(dSeparated(chain, ['A'], ['C'], ['B'])).toBe(true); // blocked by evidence on B
	});

	it('a fork is blocked by evidence on the shared parent', () => {
		// A <- B -> C
		const fork = model([
			['B', 'A'],
			['B', 'C']
		]);
		expect(dSeparated(fork, ['A'], ['C'], [])).toBe(false);
		expect(dSeparated(fork, ['A'], ['C'], ['B'])).toBe(true);
	});

	it('a collider is d-separated unless it (or a descendant) is conditioned on', () => {
		// A -> C <- B, C -> D
		const collider = model([
			['A', 'C'],
			['B', 'C'],
			['C', 'D']
		]);
		expect(dSeparated(collider, ['A'], ['B'], [])).toBe(true); // blocked collider
		expect(dSeparated(collider, ['A'], ['B'], ['C'])).toBe(false); // opened by evidence on C
		expect(dSeparated(collider, ['A'], ['B'], ['D'])).toBe(false); // opened by evidence on a descendant
	});
});

describe('activeTrailNodes', () => {
	it('propagates down an unblocked chain', () => {
		const chain = model([
			['A', 'B'],
			['B', 'C']
		]);
		expect(activeTrailNodes(chain, ['A'], [])).toEqual(new Set(['A', 'B', 'C']));
	});

	it('stops at evidence in the middle of a chain', () => {
		const chain = model([
			['A', 'B'],
			['B', 'C']
		]);
		// B is evidence: reachable includes B's Markov blanket boundary but not C beyond it.
		expect(activeTrailNodes(chain, ['A'], ['B'])).toEqual(new Set(['A']));
	});

	it('opens a collider only when it or a descendant is evidence', () => {
		// A -> C <- B, C -> D
		const collider = model([
			['A', 'C'],
			['B', 'C'],
			['C', 'D']
		]);
		expect(activeTrailNodes(collider, ['A'], [])).toEqual(new Set(['A', 'C', 'D']));
		expect(activeTrailNodes(collider, ['A'], ['C'])).toEqual(new Set(['A', 'B']));
		expect(activeTrailNodes(collider, ['A'], ['D'])).toEqual(new Set(['A', 'B', 'C']));
	});

	it('handles disconnected nodes and isolated inputs', () => {
		const isolated = model([['A', 'B']], ['Z']);
		expect(activeTrailNodes(isolated, ['Z'], [])).toEqual(new Set(['Z']));
	});
});
