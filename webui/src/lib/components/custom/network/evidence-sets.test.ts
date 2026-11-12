import { describe, expect, it } from 'vitest';
import { mapEvidenceSetResponse } from './evidence-sets';

describe('mapEvidenceSetResponse', () => {
	it('maps snake_case fields to camelCase', () => {
		expect(
			mapEvidenceSetResponse({
				id: 3,
				problem_id: 7,
				name: 'baseline',
				evidence: { A: 'hi' },
				created_at: '2026-08-24T00:00:00Z'
			})
		).toEqual({
			id: 3,
			problemId: 7,
			name: 'baseline',
			evidence: { A: 'hi' },
			createdAt: '2026-08-24T00:00:00Z'
		});
	});

	it('preserves an empty evidence dict', () => {
		expect(
			mapEvidenceSetResponse({
				id: 1,
				problem_id: 1,
				name: 'no evidence',
				evidence: {},
				created_at: '2026-08-24T00:00:00Z'
			}).evidence
		).toEqual({});
	});
});
