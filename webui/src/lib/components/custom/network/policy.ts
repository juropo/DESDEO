/**
 * Phase E, checklist item 1: read a *solved* `Problem`'s policy back, as distinct from
 * `/whatif`'s own `policy` field (a live pyAgrum engine's evidence-conditioned posterior — see
 * `NetworkPolicySource`'s own doc comment in `types.ts`, and `desdeo/api/routers/graphical.py`'s
 * module docstring, for the full reasoning). Backed by `POST /graphical/{id}/policy`.
 */

import type { NetworkEvidence, NetworkPolicy, NetworkPolicySource } from './types';

/** Default `fetchPolicy` — `POST /graphical/{modelId}/policy` via the `/api/...` proxy, matching
 *  `NetworkPanel.svelte`'s own `defaultFetchWhatIf`/`defaultFetchStructure` convention
 *  (injectable, swappable for the generated OpenAPI client later without changing call sites). */
export async function defaultFetchPolicy(
	modelId: string | number,
	source: NetworkPolicySource,
	evidence: NetworkEvidence,
	signal?: AbortSignal
): Promise<NetworkPolicy> {
	const body =
		'solutionId' in source
			? { solution_id: source.solutionId, evidence }
			: { variable_values: source.variableValues, evidence };
	const res = await fetch(`/api/graphical/${modelId}/policy`, {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify(body),
		signal
	});
	if (!res.ok) throw new Error(`Failed to fetch solved policy (status ${res.status})`);
	const result: { policy: NetworkPolicy } = await res.json();
	return result.policy;
}
