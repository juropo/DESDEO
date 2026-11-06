import { describe, expect, it } from 'vitest';
import { dSeparated } from './dag-algorithms';
import { explainNoChange } from './null-explanation';
import type { NetworkModel } from './types';

// weather -> soil_moisture -> crop_yield <- irrigate; crop_yield is a collider, so evidence on
// `irrigate` alone (no evidence/descendant-evidence on crop_yield) never reaches `weather`.
// weather -> soil_moisture -> crop_yield is also a direct chain, so committing soil_moisture (the
// middle node) is what lets a *further* exploration of weather alone leave crop_yield unaffected.
const model: NetworkModel = {
	nodes: [
		{ id: 'weather', label: 'Weather', kind: 'chance', states: ['dry', 'wet'] },
		{ id: 'soil_moisture', label: 'Soil moisture', kind: 'chance', states: ['low', 'high'] },
		{ id: 'irrigate', label: 'Irrigate', kind: 'decision', states: ['none', 'light'] },
		{ id: 'crop_yield', label: 'Crop yield', kind: 'chance', states: ['poor', 'good'] }
	],
	edges: [
		{ id: 'e1', source: 'weather', target: 'soil_moisture' },
		{ id: 'e2', source: 'soil_moisture', target: 'crop_yield' },
		{ id: 'e3', source: 'irrigate', target: 'crop_yield' }
	]
};

describe('explainNoChange', () => {
	it('returns null when nothing has changed since the (possibly empty) committed baseline', () => {
		expect(explainNoChange(model, 'weather', {}, {}, [0, 0])).toBeNull();
	});

	it('returns null for the node that is itself pinned as current evidence', () => {
		expect(explainNoChange(model, 'irrigate', { irrigate: 'light' }, {}, [0, 0])).toBeNull();
	});

	it('returns null when there is no delta to compare', () => {
		expect(explainNoChange(model, 'weather', { irrigate: 'light' }, {}, null)).toBeNull();
		expect(explainNoChange(model, 'weather', { irrigate: 'light' }, {}, [])).toBeNull();
	});

	it('returns null when the delta is not actually zero', () => {
		expect(explainNoChange(model, 'crop_yield', { irrigate: 'light' }, {}, [0.1, -0.1])).toBeNull();
	});

	it('returns null for a zero delta that is not provably d-separated', () => {
		// crop_yield is a direct child of irrigate -- not d-separated -- even though this
		// particular (contrived) delta happens to be zero, that's not something this function
		// can attribute to the graph structure.
		expect(explainNoChange(model, 'crop_yield', { irrigate: 'light' }, {}, [0, 0])).toBeNull();
	});

	it('explains a zero delta for a node d-separated from the (uncommitted) evidence', () => {
		const result = explainNoChange(model, 'weather', { irrigate: 'light' }, {}, [0, 0]);
		expect(result).toContain('irrigate');
		expect(result).toMatch(/d-separated/i);
	});

	it('treats near-zero (floating point noise) deltas as zero', () => {
		const result = explainNoChange(model, 'weather', { irrigate: 'light' }, {}, [1e-12, -1e-12]);
		expect(result).not.toBeNull();
	});

	it('returns null once every current-evidence entry matches the committed baseline (nothing added)', () => {
		const committed = { soil_moisture: 'low' };
		expect(explainNoChange(model, 'crop_yield', committed, committed, [0, 0])).toBeNull();
	});

	describe('conditioning on committed evidence (the newly-added-evidence-only query)', () => {
		it('is unconditionally NOT d-separated -- weather reaches crop_yield via the direct chain', () => {
			// Sanity precondition for the test below: without conditioning on soil_moisture, this
			// is the case an unconditioned query would get wrong.
			expect(dSeparated(model, ['crop_yield'], ['weather'])).toBe(false);
		});

		it('explains a zero delta once the blocking evidence is already committed', () => {
			// soil_moisture committed; weather explored on top of it. crop_yield's chain to weather
			// runs through soil_moisture, which is already conditioned on -- so weather alone can't
			// move crop_yield any further, even though weather and crop_yield are *not*
			// unconditionally d-separated (see the precondition test above).
			const committed = { soil_moisture: 'low' };
			const evidence = { soil_moisture: 'low', weather: 'dry' };
			const result = explainNoChange(model, 'crop_yield', evidence, committed, [0, 0]);
			expect(result).not.toBeNull();
			expect(result).toContain('weather');
			expect(result).toContain('soil_moisture');
		});

		it('does NOT explain the same zero delta against an unconditioned (empty) committed set', () => {
			// Same evidence/delta as above, but without the committed-evidence context -- the
			// unconditioned query correctly refuses to explain it (see the precondition test).
			const evidence = { soil_moisture: 'low', weather: 'dry' };
			expect(explainNoChange(model, 'crop_yield', evidence, {}, [0, 0])).toBeNull();
		});
	});
});
