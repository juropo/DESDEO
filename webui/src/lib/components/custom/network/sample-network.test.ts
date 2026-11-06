import { describe, expect, it } from 'vitest';
import { dSeparated } from './dag-algorithms';
import {
	mockWhatIf,
	sampleIrrigationBaselinePosteriors,
	sampleIrrigationNetwork
} from './sample-network';

describe('sampleIrrigationNetwork structure (mockWhatIf/null-explanation precondition)', () => {
	it('weather is d-separated from irrigate alone (crop_yield is an unconditioned collider)', () => {
		expect(dSeparated(sampleIrrigationNetwork, ['weather'], ['irrigate'])).toBe(true);
	});

	it('crop_yield is not d-separated from weather (direct chain through soil_moisture)', () => {
		expect(dSeparated(sampleIrrigationNetwork, ['crop_yield'], ['weather'])).toBe(false);
	});
});

describe('mockWhatIf', () => {
	it('returns the baseline untouched for empty evidence', () => {
		expect(mockWhatIf({})).toEqual(sampleIrrigationBaselinePosteriors);
	});

	it('leaves a d-separated node exactly at baseline', () => {
		const result = mockWhatIf({ irrigate: 'light' });
		expect(result.weather).toEqual(sampleIrrigationBaselinePosteriors.weather);
	});

	it('shifts a node reachable from the evidence away from baseline', () => {
		const result = mockWhatIf({ weather: 'dry' });
		expect(result.soil_moisture).not.toEqual(sampleIrrigationBaselinePosteriors.soil_moisture);
		expect(result.crop_yield).not.toEqual(sampleIrrigationBaselinePosteriors.crop_yield);
	});

	it('collapses an evidenced node to one-hot at the observed state (real hard evidence does too)', () => {
		const result = mockWhatIf({ weather: 'dry' });
		// 'dry' is states[0] for weather (see sample-network.ts).
		expect(result.weather).toEqual([1, 0, 0]);
	});

	it('one-hot state tracks whichever state was actually observed, not always index 0', () => {
		const result = mockWhatIf({ weather: 'wet' });
		// 'wet' is states[2] for weather.
		expect(result.weather).toEqual([0, 0, 1]);
	});

	it('every returned distribution still (approximately) sums to 1', () => {
		const result = mockWhatIf({ weather: 'dry', irrigate: 'heavy' });
		for (const probs of Object.values(result)) {
			expect(probs.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
		}
	});

	it('is deterministic for the same evidence', () => {
		expect(mockWhatIf({ weather: 'wet' })).toEqual(mockWhatIf({ weather: 'wet' }));
	});
});
