/** Small illustrative network (an irrigation-decision influence diagram) used by the Storybook
 *  story and available for ad-hoc manual testing. Not part of the public package surface — it's
 *  a fixture, not a component/type this package exports for consumers to build on. */

import { dSeparated } from './dag-algorithms';
import type { NetworkEvidence, NetworkModel, NetworkPosteriors } from './types';

export const sampleIrrigationNetwork: NetworkModel = {
	nodes: [
		{
			id: 'weather',
			label: 'Weather',
			kind: 'chance',
			states: ['dry', 'normal', 'wet'],
			description: 'Growing-season weather outlook.'
		},
		{
			id: 'soil_moisture',
			label: 'Soil moisture',
			kind: 'chance',
			states: ['low', 'medium', 'high'],
			description: 'Soil moisture at planting, driven by weather.'
		},
		{
			id: 'irrigate',
			label: 'Irrigate',
			kind: 'decision',
			states: ['none', 'light', 'heavy'],
			description: 'How much supplemental irrigation to apply.'
		},
		{
			id: 'water_usage',
			label: 'Water usage',
			kind: 'deterministic',
			description: 'Water volume used, a fixed function of the irrigation decision.'
		},
		{
			id: 'crop_yield',
			label: 'Crop yield',
			kind: 'chance',
			states: ['poor', 'average', 'good'],
			description: 'Harvest yield, depending on soil moisture and irrigation.'
		},
		{
			id: 'profit',
			label: 'Profit',
			kind: 'objective',
			description: 'Revenue from yield minus water and irrigation cost.'
		},
		{
			id: 'water_cost',
			label: 'Water cost',
			kind: 'objective',
			description: 'Cost of water used, kept separate so it can be traded off explicitly.'
		}
	],
	edges: [
		{ id: 'e1', source: 'weather', target: 'soil_moisture' },
		{ id: 'e2', source: 'soil_moisture', target: 'crop_yield' },
		{ id: 'e3', source: 'irrigate', target: 'crop_yield' },
		{ id: 'e4', source: 'irrigate', target: 'water_usage' },
		{ id: 'e5', source: 'crop_yield', target: 'profit' },
		{ id: 'e6', source: 'water_usage', target: 'profit' },
		{ id: 'e7', source: 'water_usage', target: 'water_cost' }
	]
};

/** Unconditioned prior for every chance node in `sampleIrrigationNetwork` -- the ghost baseline
 *  (phase D) `mockWhatIf` below diffs against before any evidence is set. `irrigate` (a decision,
 *  not a chance node) deliberately has no entry here, matching the real backend's own
 *  `WhatIfResponse.posteriors` convention of only ever covering chance nodes. */
export const sampleIrrigationBaselinePosteriors: NetworkPosteriors = {
	weather: [0.3, 0.4, 0.3],
	soil_moisture: [0.25, 0.5, 0.25],
	crop_yield: [0.2, 0.5, 0.3]
};

/** `nodeId -> states` lookup for `sampleIrrigationNetwork`'s own nodes -- what `mockWhatIf` uses to
 *  turn a pinned evidence *label* (e.g. `"dry"`) into the state *index* a one-hot posterior needs. */
const NODE_STATES = new Map(sampleIrrigationNetwork.nodes.map((n) => [n.id, n.states]));

/**
 * Deterministic, structure-aware mock "what-if" for the dev-only Cumulus demo dialog (graphical
 * models plan, frontend phase D) -- **not** real inference, just enough to make the ghost/delta/
 * null-explanation UI show something believable without a live backend (no compiled diagram
 * exists to call `/graphical/whatif` against in this dev sandbox; see the plan's phase D findings).
 *
 * A node that is itself pinned as evidence collapses to a one-hot vector at the observed state --
 * real hard evidence (pyAgrum `setEvidence`) does exactly this, and showing the evidenced node's
 * own bar snap to 100% is the most immediate, legible feedback that a click did something at all.
 * Every *other* node reachable from the evidence set via *some* active trail (i.e. **not**
 * `dSeparated(model, [nodeId], evidenceIds)`) gets its baseline nudged toward a deterministic,
 * evidence-derived state index; every node that's neither is returned completely untouched --
 * which is exactly what makes `sampleIrrigationNetwork` able to exercise both a real posterior
 * change (e.g. evidence on `weather` shifts `soil_moisture`/`crop_yield`) *and* a provably
 * d-separated no-op (evidence on `irrigate` alone never reaches `weather`, since `crop_yield` -- a
 * collider on that path -- is never itself evidenced) deterministically, every time.
 */
export function mockWhatIf(evidence: NetworkEvidence): NetworkPosteriors {
	const model = sampleIrrigationNetwork;
	const evidenceIds = Object.keys(evidence);
	const result: NetworkPosteriors = {};
	for (const [nodeId, baseline] of Object.entries(sampleIrrigationBaselinePosteriors)) {
		if (nodeId in evidence) {
			const stateIndex = NODE_STATES.get(nodeId)?.indexOf(evidence[nodeId]) ?? -1;
			// Defensive fallback for an evidence label that isn't actually one of this node's
			// states (shouldn't happen via the UI, which only ever posts a real state label) --
			// leave the baseline untouched rather than emit a nonsensical all-zero vector.
			result[nodeId] = stateIndex >= 0 ? oneHot(baseline.length, stateIndex) : baseline;
			continue;
		}
		const unaffected = evidenceIds.length === 0 || dSeparated(model, [nodeId], evidenceIds);
		result[nodeId] = unaffected ? baseline : shiftTowardEvidence(baseline, evidence);
	}
	return result;
}

function oneHot(length: number, index: number): number[] {
	return Array.from({ length }, (_, i) => (i === index ? 1 : 0));
}

/** Deterministically nudges `baseline` toward a state index derived from `evidence`'s own values
 *  (a stable string hash, not randomness -- the same evidence always produces the same shift) --
 *  plausible-looking, not statistically meaningful. */
function shiftTowardEvidence(baseline: number[], evidence: NetworkEvidence): number[] {
	const seed = Object.values(evidence).join('|');
	let hash = 0;
	for (let i = 0; i < seed.length; i++) hash = (hash * 31 + seed.charCodeAt(i)) >>> 0;
	const target = hash % baseline.length;
	const boost = 0.3;
	const otherShare = baseline.length > 1 ? boost / (baseline.length - 1) : 0;
	const shifted = baseline.map((p, i) => (i === target ? p + boost : Math.max(0, p - otherShare)));
	const total = shifted.reduce((sum, p) => sum + p, 0);
	return shifted.map((p) => p / total);
}
