/**
 * Named evidence sets (graphical models plan, phase E, checklist item 3) — save the current
 * evidence dict under a name, list saved sets for a model, and fetch one by id ("shareable by
 * URL"). Backed by `POST/GET /graphical/{id}/evidence-sets` and `GET/DELETE
 * /graphical/evidence-sets/{id}` (`desdeo/api/routers/graphical.py`).
 *
 * `mapEvidenceSetResponse` is the one pure, testable piece — the fetch wrappers below it are thin
 * (same "no component-testing setup, so pure logic is what's covered" reasoning as `whatif.ts`/
 * `sensitivity.ts`; `defaultFetchWhatIf`/`defaultFetchStructure` in `NetworkPanel.svelte` are the
 * existing precedent for a thin, untested-in-isolation default fetcher).
 */

import type { NetworkEvidence, NetworkEvidenceSet } from './types';

interface RawEvidenceSetResponse {
	id: number;
	problem_id: number;
	name: string;
	evidence: NetworkEvidence;
	created_at: string;
}

export function mapEvidenceSetResponse(raw: RawEvidenceSetResponse): NetworkEvidenceSet {
	return {
		id: raw.id,
		problemId: raw.problem_id,
		name: raw.name,
		evidence: raw.evidence,
		createdAt: raw.created_at
	};
}

async function parseJsonOrThrow(res: Response, what: string): Promise<unknown> {
	if (!res.ok) throw new Error(`Failed to ${what} (status ${res.status})`);
	return res.json();
}

/** Default `fetchEvidenceSets` — every named evidence set saved against `modelId`. */
export async function defaultFetchEvidenceSets(
	modelId: string | number
): Promise<NetworkEvidenceSet[]> {
	const res = await fetch(`/api/graphical/${modelId}/evidence-sets`);
	const raw = (await parseJsonOrThrow(res, 'list evidence sets')) as RawEvidenceSetResponse[];
	return raw.map(mapEvidenceSetResponse);
}

/** Default `saveEvidenceSet` — saves the given evidence under `name` for `modelId`. */
export async function defaultSaveEvidenceSet(
	modelId: string | number,
	name: string,
	evidence: NetworkEvidence
): Promise<NetworkEvidenceSet> {
	const res = await fetch(`/api/graphical/${modelId}/evidence-sets`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ name, evidence })
	});
	return mapEvidenceSetResponse(
		(await parseJsonOrThrow(res, 'save evidence set')) as RawEvidenceSetResponse
	);
}

/** Default `deleteEvidenceSet` — deletes a saved evidence set by its own id. */
export async function defaultDeleteEvidenceSet(evidenceSetId: number): Promise<void> {
	const res = await fetch(`/api/graphical/evidence-sets/${evidenceSetId}`, { method: 'DELETE' });
	if (!res.ok) throw new Error(`Failed to delete evidence set (status ${res.status})`);
}

/** Fetches one evidence set by its own id — "shareable by URL": given only the id (e.g. from a
 *  route param), this is the one call a page needs to resolve it to its evidence dict. */
export async function defaultFetchEvidenceSetById(
	evidenceSetId: number
): Promise<NetworkEvidenceSet> {
	const res = await fetch(`/api/graphical/evidence-sets/${evidenceSetId}`);
	return mapEvidenceSetResponse(
		(await parseJsonOrThrow(res, 'fetch evidence set')) as RawEvidenceSetResponse
	);
}
