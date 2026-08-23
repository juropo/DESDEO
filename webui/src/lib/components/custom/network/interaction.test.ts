import { describe, expect, it } from 'vitest';
import {
	isNodeEditable,
	isStatePinned,
	mergeHighlightSets,
	toggleEvidenceState
} from './interaction';

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
