/**
 * Pure helpers behind the phase B embedding contract — evidence toggling, decision-node
 * editability, and cross-highlight merging — kept separate from `NetworkView.svelte` so they're
 * unit-testable the same way `dag-algorithms.ts` is, without needing a component-testing setup
 * (none exists in this repo yet; `dag-algorithms.test.ts` is the precedent this follows).
 */

import type { DecisionMode, NetworkEvidence, NetworkModel, NetworkNodeKind } from './types';

/**
 * The one-line explanation `NetworkView` shows next to its evidence controls (phase D). The real
 * backend (`WhatIfRequest.evidence`, `desdeo/api/routers/graphical.py`) only ever calls pyAgrum's
 * `setEvidence` — conditioning/observation, never a do-operator intervention — and has no
 * soft-evidence/likelihood parameter at all. Rather than build a mode toggle or a "soft evidence"
 * slider the backend can't honour, this is said plainly in the UI instead: see `NetworkEvidence`'s
 * own doc comment in `types.ts` for the full reasoning, and the plan's phase D findings for why
 * this was chosen over a visibly-disabled affordance.
 */
export const EVIDENCE_MODE_NOTE =
	'Evidence here is observational (conditioning on what is known), not a causal intervention — setting a state updates belief given it, it does not simulate changing it.';

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
 * Restricts `evidence` to chance-node ids only, per `model.nodes`' own `kind`. The real
 * `/graphical/whatif` endpoint (`desdeo/api/routers/graphical.py`) hard-rejects (400) any evidence
 * key that isn't one of the compiled `DiscreteNetwork`'s chance nodes — a decision node picked
 * while exploring `decisionMode: 'manual'` is a real, clickable state per `isNodeEditable` above,
 * but posting it to `/whatif` would 400 the whole request. This is also exactly the constraint the
 * plan's own phase E entry names ("what-if inside method UIs is then restricted to chance nodes ...
 * 'if A happens' is a question about the world, not about what we choose") — enforced here at the
 * wire boundary (`NetworkPanel`'s calls into `fetchWhatIf`) rather than by disallowing the click
 * itself, so a decision pick still shows locally (evidence pin, `isStatePinned`) without ever being
 * sent to the backend as if it were an observation.
 */
export function filterChanceEvidence(
	model: Pick<NetworkModel, 'nodes'>,
	evidence: NetworkEvidence
): NetworkEvidence {
	const chanceIds = new Set(model.nodes.filter((n) => n.kind === 'chance').map((n) => n.id));
	return Object.fromEntries(Object.entries(evidence).filter(([id]) => chanceIds.has(id)));
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
