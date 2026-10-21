/**
 * Pure helpers for evidence toggling, decision-node editability, and highlight merging, kept out
 * of `NetworkView.svelte` so they're unit-testable.
 */

import type {
	DecisionMode,
	NetworkEvidence,
	NetworkModel,
	NetworkNodeKind,
	NetworkPolicy
} from './types';

/**
 * The explanation `NetworkView` shows next to its evidence controls: the backend only conditions
 * on evidence (pyAgrum `setEvidence`), never interventions or soft evidence.
 */
export const EVIDENCE_MODE_NOTE =
	'Evidence here is observational (conditioning on what is known), not a causal intervention — setting a state updates belief given it, it does not simulate changing it.';

/** Minimal shape `isStatePinned` needs off a node's data. */
interface PinnableNodeData {
	evidenceState?: string;
	policyState?: string;
}

/**
 * Chance nodes are always editable. Decision nodes are editable only in `decisionMode: 'manual'`;
 * in `'solver'` mode they show a solved policy read-only. Objective and deterministic nodes are
 * never editable.
 */
export function isNodeEditable(kind: NetworkNodeKind, decisionMode: DecisionMode): boolean {
	return kind === 'chance' || (kind === 'decision' && decisionMode === 'manual');
}

/**
 * Restricts `evidence` to chance-node ids, since `/graphical/whatif` rejects other keys. Decision
 * picks still show locally but go through `filterDecisionEvidence` instead.
 */
export function filterChanceEvidence(
	model: Pick<NetworkModel, 'nodes'>,
	evidence: NetworkEvidence
): NetworkEvidence {
	const chanceIds = new Set(model.nodes.filter((n) => n.kind === 'chance').map((n) => n.id));
	return Object.fromEntries(Object.entries(evidence).filter(([id]) => chanceIds.has(id)));
}

/**
 * Restricts `evidence` to decision-node ids, for `WhatIfRequest.decision_evidence`.
 */
export function filterDecisionEvidence(
	model: Pick<NetworkModel, 'nodes'>,
	evidence: NetworkEvidence
): NetworkEvidence {
	const decisionIds = new Set(model.nodes.filter((n) => n.kind === 'decision').map((n) => n.id));
	return Object.fromEntries(Object.entries(evidence).filter(([id]) => decisionIds.has(id)));
}

/**
 * Decision ids that must still be pinned before `/graphical/whatif` can compute utilities: with
 * more than one objective and no solution to follow, the backend needs every decision pinned (it
 * returns 501 otherwise). Empty when the request can be served.
 */
export function decisionsBlockingUtilities(
	model: Pick<NetworkModel, 'nodes'>,
	decisionEvidence: NetworkEvidence,
	hasSolution: boolean
): string[] {
	if (hasSolution) return [];
	if (model.nodes.filter((n) => n.kind === 'objective').length <= 1) return [];
	return model.nodes
		.filter((n) => n.kind === 'decision' && !(n.id in decisionEvidence))
		.map((n) => n.id);
}

/** One decision's state and where it comes from, as listed by `describeDecisions`. */
export interface DecisionStatus {
	id: string;
	label: string;
	state?: string;
	source: 'pinned' | 'solution' | 'off-policy' | 'unresolved';
	/** Unresolved only: labels of the decision's parents that are neither in the evidence nor the
	 *  policy. */
	dependsOn: string[];
}

/**
 * Each decision's state: pinned in `evidence`, moved off the followed solution's path
 * (`offPolicy`, see `WhatIfResult.offPolicyDecisions`), chosen by `policy` (the followed
 * solution's choice, which also repeats the pins), or unresolved.
 */
export function describeDecisions(
	model: Pick<NetworkModel, 'nodes' | 'edges'>,
	evidence: NetworkEvidence,
	policy: NetworkPolicy | undefined,
	offPolicy: string[] = []
): DecisionStatus[] {
	const labels = new Map(model.nodes.map((n) => [n.id, n.label]));
	return model.nodes
		.filter((n) => n.kind === 'decision')
		.map((n) => {
			if (n.id in evidence) {
				return { id: n.id, label: n.label, state: evidence[n.id], source: 'pinned', dependsOn: [] };
			}
			if (offPolicy.includes(n.id)) {
				return { id: n.id, label: n.label, source: 'off-policy', dependsOn: [] };
			}
			const chosen = policy?.[n.id];
			if (chosen !== undefined) {
				return { id: n.id, label: n.label, state: chosen, source: 'solution', dependsOn: [] };
			}
			const dependsOn = model.edges
				.filter(
					(e) => e.target === n.id && !(e.source in evidence) && !(e.source in (policy ?? {}))
				)
				.map((e) => labels.get(e.source) ?? e.source);
			return { id: n.id, label: n.label, source: 'unresolved', dependsOn };
		});
}

/**
 * Sets `state` as `nodeId`'s hard evidence, or clears it if `state` was already pinned there.
 * Immutable — returns a new map, never mutates `evidence`.
 */
export function toggleEvidenceState(
	evidence: NetworkEvidence,
	nodeId: string,
	state: string
): NetworkEvidence {
	if (evidence[nodeId] === state) {
		const next = { ...evidence };
		delete next[nodeId];
		return next;
	}
	return { ...evidence, [nodeId]: state };
}

/**
 * Whether `state` is the pinned hard-evidence state or the solved-policy state for a node —
 * the predicate every detail-level branch in `GraphNode.svelte` uses to decide whether a state
 * chip/segment renders as "selected".
 */
export function isStatePinned(nodeData: PinnableNodeData, state: string): boolean {
	return nodeData.evidenceState === state || nodeData.policyState === state;
}

/**
 * How strongly a followed solution's policy picks state `index` of a decision, given its
 * `solutionPosterior`: `'certain'` at probability 1, `'possible'` above 0, else `null`.
 */
export function solutionStateTier(
	solutionPosterior: number[] | undefined,
	index: number
): 'certain' | 'possible' | null {
	const p = solutionPosterior?.[index] ?? 0;
	if (p >= 1 - 1e-9) return 'certain';
	if (p > 1e-9) return 'possible';
	return null;
}

/**
 * Union of the internal hover/select highlight set and the external `highlight` prop. `null` means
 * no highlighting; an empty external set means "highlight nothing" and dims every node.
 */
export function mergeHighlightSets(
	internal: Set<string> | null,
	external: Iterable<string> | null | undefined
): Set<string> | null {
	if (external == null) return internal;
	const ext = new Set(external);
	return internal ? new Set([...internal, ...ext]) : ext;
}
