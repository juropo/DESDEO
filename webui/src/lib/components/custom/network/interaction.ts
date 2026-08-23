/**
 * Pure helpers behind the phase B embedding contract — evidence toggling, decision-node
 * editability, and cross-highlight merging — kept separate from `NetworkView.svelte` so they're
 * unit-testable the same way `dag-algorithms.ts` is, without needing a component-testing setup
 * (none exists in this repo yet; `dag-algorithms.test.ts` is the precedent this follows).
 */

import type { DecisionMode, NetworkEvidence, NetworkNodeKind } from './types';

/** Minimal shape `isStatePinned` needs off a node's data — matches the relevant slice of
 *  `NetworkNodeData` without importing it (that type lives in `view-types.ts`, the rendering-only
 *  side of the split this module's own docs describe; this helper is about evidence/policy data,
 *  not rendering). */
interface PinnableNodeData {
	evidenceState?: string;
	policyState?: string;
}

/**
 * Chance nodes are always editable — clicking a state sets/clears hard evidence on it. Decision
 * nodes are editable only in `decisionMode: 'manual'`; in `'solver'` mode they display a solved
 * policy's choice read-only instead (see `NetworkPolicy`). Objective and deterministic nodes are
 * never editable: an objective has no state space of its own, and a deterministic node's value
 * follows from its parents rather than being observed or chosen.
 */
export function isNodeEditable(kind: NetworkNodeKind, decisionMode: DecisionMode): boolean {
	return kind === 'chance' || (kind === 'decision' && decisionMode === 'manual');
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
 * Union of an internally computed highlight set (the hover/select ancestor closure `NetworkView`
 * already maintains) and an externally supplied one (the `highlight` prop, for cross-highlighting
 * driven by other UI — see phase C). Either side may be absent or empty; `null` means "no
 * highlighting active at all", as distinct from an empty-but-active set — `external` being an
 * empty (but present, i.e. not `null`/`undefined`) iterable is a caller actively saying "highlight
 * nothing", which must dim every node, not read as "no highlight prop was given".
 */
export function mergeHighlightSets(
	internal: Set<string> | null,
	external: Iterable<string> | null | undefined
): Set<string> | null {
	if (external == null) return internal;
	const ext = new Set(external);
	return internal ? new Set([...internal, ...ext]) : ext;
}
