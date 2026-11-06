import { describe, expect, it } from 'vitest';
import { mapSensitivityResponse, mapVoiResponse, sortByImpact, tornadoDomain } from './sensitivity';
import type { SensitivityNodeResult } from './sensitivity';

describe('mapSensitivityResponse', () => {
	it('maps snake_case fields to camelCase', () => {
		const mapped = mapSensitivityResponse({
			base_value: 50,
			results: [{ node: 'A', values: { lo: 0, hi: 100 }, low: 0, high: 100, swing: 100 }]
		});
		expect(mapped).toEqual({
			baseValue: 50,
			results: [{ node: 'A', values: { lo: 0, hi: 100 }, low: 0, high: 100, swing: 100 }]
		});
	});

	it('maps an empty results list', () => {
		expect(mapSensitivityResponse({ base_value: 1, results: [] })).toEqual({
			baseValue: 1,
			results: []
		});
	});
});

describe('mapVoiResponse', () => {
	it('maps snake_case fields to camelCase', () => {
		const mapped = mapVoiResponse({
			base_value: 4.0,
			expected_value_with_observation: 5.8,
			value_of_information: 1.8,
			marginal: { lo: 0.4, hi: 0.6 }
		});
		expect(mapped).toEqual({
			baseValue: 4.0,
			expectedValueWithObservation: 5.8,
			valueOfInformation: 1.8,
			marginal: { lo: 0.4, hi: 0.6 }
		});
	});
});

describe('sortByImpact', () => {
	const a: SensitivityNodeResult = { node: 'a', values: {}, low: 0, high: 10, swing: 10 };
	const b: SensitivityNodeResult = { node: 'b', values: {}, low: 0, high: 100, swing: 100 };
	const c: SensitivityNodeResult = { node: 'c', values: {}, low: 0, high: 1, swing: 1 };

	it('sorts descending by swing', () => {
		expect(sortByImpact([a, c, b]).map((r) => r.node)).toEqual(['b', 'a', 'c']);
	});

	it('does not mutate the input array', () => {
		const input = [a, c, b];
		const original = [...input];
		sortByImpact(input);
		expect(input).toEqual(original);
	});

	it('handles an empty array', () => {
		expect(sortByImpact([])).toEqual([]);
	});
});

describe('tornadoDomain', () => {
	it('covers every result low/high plus baseValue', () => {
		const results: SensitivityNodeResult[] = [
			{ node: 'A', values: {}, low: 0, high: 100, swing: 100 },
			{ node: 'B', values: {}, low: 40, high: 60, swing: 20 }
		];
		expect(tornadoDomain(results, 50)).toEqual([0, 100]);
	});

	it('extends beyond every swept range when baseValue falls outside it', () => {
		const results: SensitivityNodeResult[] = [
			{ node: 'A', values: {}, low: 10, high: 20, swing: 10 }
		];
		expect(tornadoDomain(results, 0)).toEqual([0, 20]);
		expect(tornadoDomain(results, 30)).toEqual([10, 30]);
	});

	it('widens a degenerate (all-equal) domain instead of returning a zero-width range', () => {
		const results: SensitivityNodeResult[] = [{ node: 'A', values: {}, low: 5, high: 5, swing: 0 }];
		expect(tornadoDomain(results, 5)).toEqual([4, 6]);
	});

	it('handles no results at all -- domain collapses to just baseValue, widened', () => {
		expect(tornadoDomain([], 7)).toEqual([6, 8]);
	});
});
