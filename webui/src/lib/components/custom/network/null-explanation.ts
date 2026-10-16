/**
 * "Why did nothing happen" — graphical models plan, frontend phase D. When evidence changes and a
 * node's posterior comes back identical to its ghost baseline, the honest thing to show is *why*
 * (d-separated from the evidence set, per the network's own structure) rather than let the DM
 * assume the tool silently failed to update. Pure and testable for the same reason
 * `dag-algorithms.ts`/`whatif.ts` are — see `null-explanation.test.ts`.
 */

import { buildAdjacency, dSeparated, type Adjacency } from './dag-algorithms';
import type { NetworkEvidence, NetworkModel } from './types';

/**
 * Explains a zero-delta posterior for `nodeId`, or returns `null` when there's nothing to
 * explain: nothing has changed since `committedEvidence` (no new/changed evidence entries),
 * `nodeId` is itself pinned as *current* evidence (trivially "explained" by being the thing that
 * was just set), there's no `delta` to compare (no ghost baseline or no posterior yet), the delta
 * is *not* actually zero (a real change needs no explanation), or the zero delta isn't provably
 * due to d-separation (some other reason a real inference engine produced an unchanged value — not
 * something this client-side check can explain, so it says nothing rather than guessing).
 *
 * Queries d-separation between `nodeId` and the evidence *added since the last commit* (not the
 * whole current evidence set), conditioned on `committedEvidence` — the ghost baseline this delta
 * is measured against already reflects `committedEvidence`, so the only thing that can explain a
 * *further* zero change is that the newly added/changed evidence is d-separated from `nodeId`
 * given what was already committed. Querying against the unconditioned graph (ignoring
 * `committedEvidence` as the conditioning set) is wrong the moment anything has ever been
 * committed: e.g. a chain weather → soil_moisture → crop_yield, with `soil_moisture` committed —
 * `crop_yield`'s posterior is correctly unaffected by further exploring `weather` alone (blocked
 * by the committed evidence on `soil_moisture`), even though `weather` and `crop_yield` are *not*
 * unconditionally d-separated (there's a direct chain between them).
 *
 * `adjacency` may be passed in (built once per `model`, e.g. by `NetworkView`'s own
 * `$derived.by`) to avoid `dSeparated` rebuilding it from scratch on every node, every
 * reconciliation — see `dSeparated`'s own doc comment.
 */
export function explainNoChange(
	model: Pick<NetworkModel, 'nodes' | 'edges'>,
	nodeId: string,
	evidence: NetworkEvidence,
	committedEvidence: NetworkEvidence,
	delta: number[] | null,
	adjacency: Adjacency = buildAdjacency(model)
): string | null {
	if (nodeId in evidence) return null;
	// "Added since the commit" -- a key absent from `committedEvidence`, or present with a
	// different value (re-pinning the same node to a different state also counts as new evidence
	// relative to the committed baseline).
	const addedEvidenceIds = Object.keys(evidence).filter(
		(id) => evidence[id] !== committedEvidence[id]
	);
	if (addedEvidenceIds.length === 0) return null;
	if (!delta || delta.length === 0) return null;
	const changed = delta.some((d) => Math.abs(d) > 1e-9);
	if (changed) return null;
	const committedIds = Object.keys(committedEvidence);
	if (!dSeparated(model, [nodeId], addedEvidenceIds, committedIds, adjacency)) return null;
	const givenClause =
		committedIds.length > 0
			? ` given the already-committed evidence (${committedIds.join(', ')})`
			: '';
	return `Unaffected by the current evidence (${addedEvidenceIds.join(', ')}) — d-separated from it${givenClause} in the network structure.`;
}
