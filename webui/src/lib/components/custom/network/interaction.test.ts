import { describe, expect, it } from 'vitest';
import {
	decisionsBlockingUtilities,
	describeDecisions,
	filterChanceEvidence,
	filterDecisionEvidence,
	isNodeEditable,
	isStatePinned,
	mergeHighlightSets,
	solutionStateTier,
	toggleEvidenceState
} from './interaction';
import type { NetworkModel } from './types';

describe('isNodeEditable', () => {
	it('chance nodes are always editable', () => {
		expect(isNodeEditable('chance', 'solver')).toBe(true);
		expect(isNodeEditable('chance', 'manual')).toBe(true);
	});

	it('decision nodes are editable only in manual mode', () => {
		expect(isNodeEditable('decision', 'solver')).toBe(false);
		expect(isNodeEditable('decision', 'manual')).toBe(true);
	});

	it('objective and deterministic nodes are never editable', () => {
		expect(isNodeEditable('objective', 'manual')).toBe(false);
		expect(isNodeEditable('deterministic', 'manual')).toBe(false);
	});
});

describe('toggleEvidenceState', () => {
	it('sets a new evidence state', () => {
		expect(toggleEvidenceState({}, 'weather', 'dry')).toEqual({ weather: 'dry' });
	});

	it('clears the state if it was already pinned', () => {
		expect(toggleEvidenceState({ weather: 'dry' }, 'weather', 'dry')).toEqual({});
	});

	it('overwrites a different pinned state on the same node', () => {
		expect(toggleEvidenceState({ weather: 'dry' }, 'weather', 'wet')).toEqual({ weather: 'wet' });
	});

	it('leaves other nodes untouched', () => {
		expect(toggleEvidenceState({ irrigate: 'light' }, 'weather', 'dry')).toEqual({
			irrigate: 'light',
			weather: 'dry'
		});
	});

	it('does not mutate the input map', () => {
		const evidence = { weather: 'dry' };
		toggleEvidenceState(evidence, 'weather', 'wet');
		expect(evidence).toEqual({ weather: 'dry' });
	});
});

describe('isStatePinned', () => {
	it('is pinned when it matches the evidence state', () => {
		expect(isStatePinned({ evidenceState: 'dry' }, 'dry')).toBe(true);
	});

	it('is pinned when it matches the policy state', () => {
		expect(isStatePinned({ policyState: 'light' }, 'light')).toBe(true);
	});

	it('is not pinned otherwise', () => {
		expect(isStatePinned({ evidenceState: 'dry' }, 'wet')).toBe(false);
		expect(isStatePinned({}, 'dry')).toBe(false);
	});
});

describe('solutionStateTier', () => {
	it('is certain at probability 1, up to roundoff', () => {
		expect(solutionStateTier([1, 0], 0)).toBe('certain');
		expect(solutionStateTier([1 - 1e-12, 1e-12], 0)).toBe('certain');
	});

	it('is possible for any probability above roundoff', () => {
		expect(solutionStateTier([0.62, 0.38], 0)).toBe('possible');
		expect(solutionStateTier([1e-6, 1 - 1e-6], 0)).toBe('possible');
	});

	it('is null at probability 0, up to roundoff', () => {
		expect(solutionStateTier([1, 0], 1)).toBeNull();
		expect(solutionStateTier([1 - 1e-12, 1e-12], 1)).toBeNull();
	});

	it('is null without a posterior or for a missing index', () => {
		expect(solutionStateTier(undefined, 0)).toBeNull();
		expect(solutionStateTier([1, 0], 2)).toBeNull();
	});
});

describe('filterChanceEvidence', () => {
	const model: NetworkModel = {
		nodes: [
			{ id: 'weather', label: 'Weather', kind: 'chance', states: ['dry', 'wet'] },
			{ id: 'irrigate', label: 'Irrigate', kind: 'decision', states: ['none', 'light'] },
			{ id: 'profit', label: 'Profit', kind: 'objective' }
		],
		edges: []
	};

	it('keeps only chance-node evidence entries', () => {
		expect(filterChanceEvidence(model, { weather: 'dry', irrigate: 'light' })).toEqual({
			weather: 'dry'
		});
	});

	it('drops evidence for ids not present in the model at all', () => {
		expect(filterChanceEvidence(model, { unknown_id: 'x' })).toEqual({});
	});

	it('returns an empty map unchanged', () => {
		expect(filterChanceEvidence(model, {})).toEqual({});
	});

	it('is a no-op when every entry is already chance-node evidence', () => {
		expect(filterChanceEvidence(model, { weather: 'wet' })).toEqual({ weather: 'wet' });
	});
});

describe('filterDecisionEvidence', () => {
	const model: NetworkModel = {
		nodes: [
			{ id: 'weather', label: 'Weather', kind: 'chance', states: ['dry', 'wet'] },
			{ id: 'irrigate', label: 'Irrigate', kind: 'decision', states: ['none', 'light'] },
			{ id: 'profit', label: 'Profit', kind: 'objective' }
		],
		edges: []
	};

	it('keeps only decision-node evidence entries', () => {
		expect(filterDecisionEvidence(model, { weather: 'dry', irrigate: 'light' })).toEqual({
			irrigate: 'light'
		});
	});

	it('drops evidence for ids not present in the model at all', () => {
		expect(filterDecisionEvidence(model, { unknown_id: 'x' })).toEqual({});
	});

	it('returns an empty map unchanged', () => {
		expect(filterDecisionEvidence(model, {})).toEqual({});
	});
});

describe('mergeHighlightSets', () => {
	it('returns null when both are absent', () => {
		expect(mergeHighlightSets(null, null)).toBeNull();
		expect(mergeHighlightSets(null, undefined)).toBeNull();
	});

	it('treats an explicit empty external set as active-but-empty, not absent', () => {
		// Distinct from the "both absent" case above: a caller passing `highlight={[]}` means
		// "actively highlighting zero nodes" (dim everything), not "no highlight prop given".
		const result = mergeHighlightSets(null, []);
		expect(result).not.toBeNull();
		expect(result).toEqual(new Set());
	});

	it('returns the internal set alone when external is absent', () => {
		expect(mergeHighlightSets(new Set(['a']), null)).toEqual(new Set(['a']));
	});

	it('returns the external set alone when internal is absent', () => {
		expect(mergeHighlightSets(null, ['b'])).toEqual(new Set(['b']));
	});

	it('unions both when present', () => {
		expect(mergeHighlightSets(new Set(['a']), ['b'])).toEqual(new Set(['a', 'b']));
	});

	it('unions internal with an active-but-empty external, keeping just the internal set', () => {
		expect(mergeHighlightSets(new Set(['a']), [])).toEqual(new Set(['a']));
	});
});

// Oil Wildcatter shape: `test` is blind, `drill` observes `test_result`.
const OIL_MODEL: NetworkModel = {
	nodes: [
		{ id: 'test_result', label: 'test_result', kind: 'chance', states: ['NS', 'CS'] },
		{ id: 'test', label: 'Run seismic test', kind: 'decision', states: ['yes', 'no'] },
		{ id: 'drill', label: 'Drill', kind: 'decision', states: ['yes', 'no'] },
		{ id: 'util_profit', label: 'Profit', kind: 'objective' },
		{ id: 'util_investment', label: 'Investment', kind: 'objective' }
	],
	edges: [
		{ id: 'test->test_result', source: 'test', target: 'test_result' },
		{ id: 'test_result->drill', source: 'test_result', target: 'drill' }
	]
};

describe('decisionsBlockingUtilities', () => {
	it('lists unpinned decisions for a multi-objective diagram without a solution', () => {
		expect(decisionsBlockingUtilities(OIL_MODEL, { test: 'yes' }, false)).toEqual(['drill']);
	});

	it('is empty once every decision is pinned', () => {
		expect(decisionsBlockingUtilities(OIL_MODEL, { test: 'yes', drill: 'no' }, false)).toEqual([]);
	});

	it('is empty with a solution to follow', () => {
		expect(decisionsBlockingUtilities(OIL_MODEL, {}, true)).toEqual([]);
	});

	it('is empty for a single-objective diagram', () => {
		const single = { nodes: OIL_MODEL.nodes.filter((n) => n.id !== 'util_investment') };
		expect(decisionsBlockingUtilities(single, {}, false)).toEqual([]);
	});
});

describe('describeDecisions', () => {
	it('reports pinned, solution-chosen and unresolved decisions', () => {
		expect(describeDecisions(OIL_MODEL, { test: 'no' }, { test: 'no' })).toEqual([
			{ id: 'test', label: 'Run seismic test', state: 'no', source: 'pinned', dependsOn: [] },
			{ id: 'drill', label: 'Drill', source: 'unresolved', dependsOn: ['test_result'] }
		]);
	});

	it('takes a decision from the policy when not pinned', () => {
		const statuses = describeDecisions(
			OIL_MODEL,
			{ test_result: 'CS' },
			{ test: 'yes', drill: 'yes' }
		);
		expect(statuses.map((s) => [s.id, s.state, s.source])).toEqual([
			['test', 'yes', 'solution'],
			['drill', 'yes', 'solution']
		]);
	});

	it('marks a decision moved off the solution path', () => {
		const [, drill] = describeDecisions(OIL_MODEL, { test: 'no' }, { test: 'no' }, ['drill']);
		expect(drill).toEqual({ id: 'drill', label: 'Drill', source: 'off-policy', dependsOn: [] });
	});

	it('does not list a parent resolved by evidence as a dependency', () => {
		const [, drill] = describeDecisions(OIL_MODEL, { test_result: 'CS' }, undefined);
		expect(drill).toMatchObject({ source: 'unresolved', dependsOn: [] });
	});
});
