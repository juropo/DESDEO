<script lang="ts">
	/**
	 * `<NetworkView>` — structure-only network visualisation (graphical models plan, frontend
	 * phase A). No backend calls: renders a `NetworkModel`'s DAG structure on a Svelte Flow canvas
	 * plus a table view, with client-side ancestor-set highlighting on hover/select.
	 *
	 * Layout is elk-computed on demand (mount, if the model carries no usable layout yet, or the
	 * explicit "Re-layout" button) and written back onto the bindable `model` prop so callers can
	 * persist it — re-running layout on every mount would destroy spatial memory across method UIs.
	 *
	 * Level of detail (`compact` / `normal` / `expanded`, see `lod.ts`) is driven by the canvas
	 * container's width *and* the current zoom, or pinned to `'compact'` via the `compact` prop.
	 */
	import { onMount, untrack, type Snippet } from 'svelte';
	import { SvelteFlow, Background, Controls, type Viewport } from '@xyflow/svelte';
	import '@xyflow/svelte/dist/base.css';
	import { Button } from '$lib/components/ui/button';
	import { SegmentedControl } from '$lib/components/custom/segmented-control';
	import IconRefreshCw from '@lucide/svelte/icons/refresh-cw';
	import IconCheck from '@lucide/svelte/icons/check';
	import IconInfo from '@lucide/svelte/icons/info';
	import NetworkTable from './NetworkTable.svelte';
	import { NETWORK_NODE_TYPES } from './nodes';
	import { computeElkLayout, layoutCoversModel } from './layout';
	import { computeDetailLevel } from './lod';
	import {
		EVIDENCE_MODE_NOTE,
		isNodeEditable,
		mergeHighlightSets,
		toggleEvidenceState
	} from './interaction';
	import { computeDelta, hasEvidenceDiverged } from './whatif';
	import { explainNoChange } from './null-explanation';
	import { ancestralSet, buildAdjacency, type Adjacency } from './dag-algorithms';
	import type {
		DecisionMode,
		NetworkEvidence,
		NetworkModel,
		NetworkPolicy,
		NetworkPosteriors
	} from './types';
	import type {
		HighlightState,
		NetworkChromeContext,
		NetworkFlowEdge,
		NetworkFlowNode
	} from './view-types';

	interface Props {
		/** Bindable so a computed layout (auto, or from "Re-layout") can be persisted by the
		 *  caller — see the plan's design decision on where compiled-model metadata lives. */
		model: NetworkModel;
		/** Hard evidence pinned per node id. Bindable: editable nodes (see `isNodeEditable`) toggle
		 *  it on click, but a method UI or `<NetworkPanel>` is free to set/read it directly too —
		 *  e.g. to seed evidence from elsewhere in its own UI. */
		evidence?: NetworkEvidence;
		/** Per-node posterior probabilities to render as state bars (phase B is UI-only — this is
		 *  the wire shape a real compiled-diagram what-if response fills in later, not a live
		 *  computation done here). */
		posteriors?: NetworkPosteriors;
		/** Baseline distribution each of `posteriors`' bars is diffed against — the prior (no
		 *  evidence at all) or, once the DM has committed some evidence via `onCommitEvidence`,
		 *  that committed evidence's own posteriors. Entirely caller-owned (phase D): this
		 *  component never fetches or computes it itself, only renders the ghost mark/signed delta
		 *  it implies (see `GraphNode`'s `delta`/`ghostPosterior` handling) and decides whether
		 *  `evidence` currently diverges from `committedEvidence` for the Commit button below. */
		ghostPosteriors?: NetworkPosteriors;
		/** The evidence the caller currently considers "accepted" — e.g. already folded into a
		 *  method's own reference-point ranges. Compared against `evidence` (via
		 *  `hasEvidenceDiverged`) purely to decide whether there's anything exploratory left to
		 *  commit; this component never mutates it itself. Defaults to `{}` (nothing committed
		 *  yet), so a caller that never wires up commit machinery still gets a correctly-disabled
		 *  Commit button (nothing to diverge from) rather than a crash. */
		committedEvidence?: NetworkEvidence;
		/** A solved policy's chosen state per decision node, shown read-only in
		 *  `decisionMode: 'solver'`. */
		policy?: NetworkPolicy;
		/** True while a what-if refresh triggered by the current `evidence` is in flight (or
		 *  debounced, about to be) — dims the whole canvas+table rather than let a partial update
		 *  (some node's bars already refreshed, others not yet) read as a bug. Entirely
		 *  caller-driven: this component owns no fetching of its own (see `NetworkPanel`, or the
		 *  Cumulus demo's own debounced mock, for what actually sets this). */
		stale?: boolean;
		/** `'solver'` (default): decision nodes display `policy`'s choice, read-only. `'manual'`:
		 *  decision nodes become clickable like chance nodes, writing the chosen state into
		 *  `evidence` instead. */
		decisionMode?: DecisionMode;
		/** Externally driven highlight set — e.g. hovering an objective in a reference-point widget
		 *  (phase C) — unioned with the hover/select ancestor highlighting this component already
		 *  computes on its own. */
		highlight?: Iterable<string> | null;
		/** Injected per-node chrome (a badge, icon, etc.) so a method UI can decorate nodes without
		 *  forking this component or `GraphNode`. Receives `{ id, kind, label }` for the node being
		 *  rendered. */
		nodeChrome?: Snippet<[NetworkChromeContext]>;
		/** Pins the level of detail to `'compact'` regardless of container width/zoom — set this
		 *  when the caller already knows it's embedding into a small panel. */
		compact?: boolean;
		class?: string;
		onNodeSelect?: (id: string | null) => void;
		onNodeHover?: (id: string | null) => void;
		onEvidenceChange?: (evidence: NetworkEvidence) => void;
		/** Fired when the DM clicks "Commit evidence" — the explicit exploratory-vs-committed
		 *  boundary the plan's phase D calls for. This component does nothing else on commit
		 *  (doesn't touch `evidence`, `ghostPosteriors`, or `committedEvidence` itself): promoting
		 *  the current evidence/posteriors into the new baseline, or triggering a real method's own
		 *  range recomputation, is entirely the caller's decision to make — see the plan's phase D
		 *  findings for why ("hold ranges fixed, show clipping" was chosen over silent
		 *  recompute-and-warn, precisely because no such recompute machinery exists yet to make a
		 *  "warn" claim honest). */
		onCommitEvidence?: (evidence: NetworkEvidence) => void;
	}

	let {
		model = $bindable(),
		evidence = $bindable({}),
		posteriors,
		ghostPosteriors,
		committedEvidence = {},
		policy,
		stale = false,
		decisionMode = 'solver',
		highlight = null,
		nodeChrome,
		compact = false,
		class: className = '',
		onNodeSelect,
		onNodeHover,
		onEvidenceChange,
		onCommitEvidence
	}: Props = $props();

	let containerEl = $state<HTMLDivElement | null>(null);
	let containerWidth = $state(560);
	let viewport = $state<Viewport>({ x: 0, y: 0, zoom: 1 });
	// Plain `string` (not a literal union) so it type-matches `SegmentedControl`'s bindable
	// `value: string` prop; the option `value`s below are the only strings it will ever hold.
	let viewMode = $state<string>('split');
	let hoveredId = $state<string | null>(null);
	let selectedId = $state<string | null>(null);
	let relayouting = $state(false);

	let detailLevel = $derived(computeDetailLevel(containerWidth, viewport.zoom, compact));

	// Rebuilt only when `model.nodes`/`model.edges` themselves change (that's all `buildAdjacency`
	// reads), not on every hover/select -- `highlightedIds` below used to call `ancestralSet`
	// without this, which rebuilt the adjacency maps from scratch on every single hover event.
	let adjacency = $derived.by((): Adjacency => buildAdjacency(model));

	// Free (structural, backend-independent) slice of phase C: hovering/selecting a node lights
	// up its ancestor set. Full active-trail-given-evidence highlighting needs evidence state
	// that only exists once phase D's what-if machinery lands.
	let highlightFocusId = $derived(hoveredId ?? selectedId);
	// Unioned with the externally supplied `highlight` prop (phase C: cross-highlighting driven
	// by other UI, e.g. hovering an objective in a reference-point widget) — see `mergeHighlightSets`.
	let highlightedIds = $derived.by((): Set<string> | null => {
		const internal = highlightFocusId ? ancestralSet(model, [highlightFocusId], adjacency) : null;
		return mergeHighlightSets(internal, highlight);
	});

	let flowNodes = $state.raw<NetworkFlowNode[]>([]);
	let flowEdges = $state.raw<NetworkFlowEdge[]>([]);

	// Whether there's anything exploratory left to commit -- the Commit button below is disabled
	// once `evidence` matches `committedEvidence` exactly (nothing new to accept), which is also
	// true before any evidence has ever been set at all.
	let diverged = $derived(hasEvidenceDiverged(evidence, committedEvidence));

	function handleCommit() {
		onCommitEvidence?.(evidence);
	}

	function highlightStateFor(id: string, highlights: Set<string> | null): HighlightState {
		if (!highlights) return 'normal';
		return highlights.has(id) ? 'active' : 'dimmed';
	}

	// Structural reconciliation: reruns on node/edge-set changes, detail-level changes
	// (resize/zoom), and evidence/policy/posteriors/decisionMode/chrome changes -- deliberately
	// *not* on highlight changes (hover/select), which the lightweight effect below handles
	// instead. Rebuilding this whole array (including a fresh `onStateClick` closure per node)
	// on every hover/select used to make each mouse move over the canvas reallocate N node
	// objects and N throwaway closures for what is otherwise only a `highlight` field flip.
	// Existing node positions (including manual drags) are preserved via `prevPositions`; only
	// genuinely new nodes fall back to the persisted layout or the origin.
	$effect(() => {
		const nodes = model.nodes;
		const edges = model.edges;
		const layout = model.layout;
		const detail = detailLevel;
		const evidenceSnapshot = evidence;
		const committedSnapshot = committedEvidence;
		const policySnapshot = policy;
		const posteriorSnapshot = posteriors;
		const ghostSnapshot = ghostPosteriors;
		const mode = decisionMode;
		const chrome = nodeChrome;

		untrack(() => {
			// Read without subscribing -- this effect only needs *a* value to seed with; the
			// highlight-patch effect below is what keeps it in sync as it changes.
			const highlights = highlightedIds;
			const prevPositions = new Map(flowNodes.map((n) => [n.id, n.position]));
			flowNodes = nodes.map((n) => {
				const posterior = posteriorSnapshot?.[n.id];
				const ghostPosterior = ghostSnapshot?.[n.id];
				return {
					id: n.id,
					type: n.kind,
					position: prevPositions.get(n.id) ?? layout?.positions[n.id] ?? { x: 0, y: 0 },
					data: {
						kind: n.kind,
						label: n.label,
						states: n.states,
						description: n.description,
						detail,
						highlight: highlightStateFor(n.id, highlights),
						editable: isNodeEditable(n.kind, mode),
						evidenceState: evidenceSnapshot[n.id],
						// Only meaningful read-only in 'solver' mode -- in 'manual' mode the node is
						// interactive instead, and shouldn't also show a (possibly stale) solved
						// policy's checkmark alongside its clickable states.
						policyState:
							n.kind === 'decision' && mode === 'solver' ? policySnapshot?.[n.id] : undefined,
						posterior,
						ghostPosterior,
						// Only worth explaining a *zero* delta -- explainNoChange itself no-ops when
						// `computeDelta` finds an actual change (or nothing to compare at all).
						// `committedSnapshot` is the conditioning set: `ghostPosterior` already
						// reflects it, so the d-separation query is "is nodeId separated from the
						// evidence *added since the commit*, given the commit" -- see
						// `explainNoChange`'s own doc comment for why an unconditioned query is
						// wrong the moment anything has ever been committed. `adjacency` (already
						// `$derived.by` off `model` above) is passed through so this doesn't rebuild
						// it from scratch per node, per reconciliation.
						// `?? undefined`: NetworkNodeData's contract uses "absent" (not `null`) for
						// "nothing to explain", matching every other optional field on it.
						nullExplanation:
							explainNoChange(
								model,
								n.id,
								evidenceSnapshot,
								committedSnapshot,
								computeDelta(ghostPosterior, posterior),
								adjacency
							) ?? undefined,
						chrome,
						onStateClick: (state: string) => handleStateClick(n.id, state)
					}
				};
			});
			flowEdges = edges.map((e) => ({ id: e.id, source: e.source, target: e.target }));
		});
	});

	// Lightweight highlight-only patch: keeps `flowNodes` in sync with `highlightedIds` (hover/
	// select, or an externally supplied `highlight` prop) without rebuilding the rest of each
	// node's data or its closures -- see the effect above.
	$effect(() => {
		const highlights = highlightedIds;
		untrack(() => {
			flowNodes = flowNodes.map((n) => {
				const nextHighlight = highlightStateFor(n.id, highlights);
				return n.data.highlight === nextHighlight
					? n
					: { ...n, data: { ...n.data, highlight: nextHighlight } };
			});
		});
	});

	/** Bound into each node's `data.onStateClick` above — toggles hard evidence on that node's
	 *  state and notifies the caller. Only reachable for editable nodes; `GraphNode` gates the
	 *  click affordance on `data.editable`, itself driven by `isNodeEditable`. */
	function handleStateClick(nodeId: string, state: string) {
		const next = toggleEvidenceState(evidence, nodeId, state);
		evidence = next;
		onEvidenceChange?.(next);
	}

	async function relayout() {
		relayouting = true;
		try {
			const layout = await computeElkLayout(model);
			model = { ...model, layout };
			flowNodes = flowNodes.map((n) => ({ ...n, position: layout.positions[n.id] ?? n.position }));
		} finally {
			relayouting = false;
		}
	}

	onMount(() => {
		if (!containerEl) return;
		const observer = new ResizeObserver((entries) => {
			const width = entries[0]?.contentRect.width;
			if (width) containerWidth = width;
		});
		observer.observe(containerEl);

		if (!layoutCoversModel(model, model.layout)) {
			void relayout();
		}

		return () => observer.disconnect();
	});

	function handleNodeClick({ node }: { node: NetworkFlowNode }) {
		selectedId = selectedId === node.id ? null : node.id;
		onNodeSelect?.(selectedId);
	}

	function handleNodePointerEnter({ node }: { node: NetworkFlowNode }) {
		hoveredId = node.id;
		onNodeHover?.(node.id);
	}

	function handleNodePointerLeave() {
		hoveredId = null;
		onNodeHover?.(null);
	}

	function handleTableHover(id: string | null) {
		hoveredId = id;
		onNodeHover?.(id);
	}

	function handleTableSelect(id: string) {
		selectedId = selectedId === id ? null : id;
		onNodeSelect?.(selectedId);
	}
</script>

<div class="flex h-full min-h-0 flex-col gap-2 {className}">
	<div class="flex flex-wrap items-center justify-between gap-2">
		<SegmentedControl
			size="sm"
			bind:value={viewMode}
			options={[
				{ value: 'canvas', label: 'Canvas' },
				{ value: 'split', label: 'Split' },
				{ value: 'table', label: 'Table' }
			]}
		/>
		<div class="flex items-center gap-2">
			<!-- Evidence-mode disclosure (phase D item 2): the backend only ever conditions on
			     evidence (pyAgrum `setEvidence`) -- never a causal intervention -- and has no
			     soft-evidence/likelihood parameter. Said plainly here rather than building a mode
			     toggle or a "soft evidence" affordance the backend can't actually honour. Always
			     visible (not gated to wider viewports -- the toolbar already `flex-wrap`s instead of
			     needing to hide anything to fit) since this is a safety disclosure, not decoration:
			     the evidence chips/pins/Commit button stay fully interactive at every width, so the
			     note explaining what that evidence actually means has to too. `aria-label` (not just
			     `title`, which a touch UI has no way to trigger) carries the full sentence as this
			     element's accessible name; the short visible text is what a sighted user reads
			     without needing to hover. -->
			<span
				class="text-muted-foreground flex items-center gap-1 text-xs"
				title={EVIDENCE_MODE_NOTE}
				aria-label={EVIDENCE_MODE_NOTE}
			>
				<IconInfo class="size-3.5 shrink-0" />
				Observational evidence only
			</span>
			<Button
				variant="outline"
				size="sm"
				disabled={!diverged || stale}
				onclick={handleCommit}
				title={stale
					? 'Waiting for the current evidence to finish refreshing before it can be committed.'
					: 'Accept the current evidence as the new baseline for future ghost/delta comparisons.'}
			>
				<IconCheck class="size-3.5" />
				Commit evidence
			</Button>
			<Button variant="outline" size="sm" onclick={relayout} disabled={relayouting}>
				<IconRefreshCw class="size-3.5 {relayouting ? 'animate-spin' : ''}" />
				Re-layout
			</Button>
		</div>
	</div>

	<!--
		Both panes stay permanently mounted; only visibility/sizing changes with `viewMode`. Using
		{#if} to swap DOM structure per mode would tear down and recreate the canvas container
		(and with it `containerEl`, the ResizeObserver target, and the SvelteFlow viewport) every
		time the user toggled modes — this way canvas zoom/pan state and the ResizeObserver both
		survive mode switches.
	-->
	<div class="relative flex min-h-0 flex-1 flex-col gap-2">
		<!-- Dims the *whole* canvas+table pane, not just the nodes whose bars are about to change --
		     a partial update (some nodes refreshed, others not yet) reads as a bug, so staleness is
		     shown before any per-node data actually changes (see `stale`'s own doc comment). -->
		<div
			class="pointer-events-none absolute inset-0 z-20 flex items-start justify-center pt-2 transition-opacity duration-300"
			class:opacity-0={!stale}
			class:opacity-100={stale}
			aria-hidden={!stale}
		>
			<span class="bg-card text-muted-foreground rounded-md border px-2 py-1 text-xs shadow-sm">
				Updating…
			</span>
		</div>
		<div
			class="flex min-h-0 flex-1 flex-col gap-2 transition-opacity duration-300"
			class:opacity-50={stale}
		>
			<div
				bind:this={containerEl}
				class="border-border min-h-[180px] rounded-md border"
				class:hidden={viewMode === 'table'}
				class:flex-1={viewMode !== 'split'}
				style={viewMode === 'split' ? 'flex: 6 1 0;' : ''}
			>
				<SvelteFlow
					bind:nodes={flowNodes}
					bind:edges={flowEdges}
					bind:viewport
					nodeTypes={NETWORK_NODE_TYPES}
					fitView
					minZoom={0.2}
					maxZoom={2}
					onnodeclick={handleNodeClick}
					onnodepointerenter={handleNodePointerEnter}
					onnodepointerleave={handleNodePointerLeave}
				>
					<Background />
					<Controls showLock={false} />
				</SvelteFlow>
			</div>
			<div
				class="min-h-0 overflow-auto"
				class:hidden={viewMode === 'canvas'}
				class:flex-1={viewMode !== 'split'}
				style={viewMode === 'split' ? 'flex: 4 1 0;' : ''}
			>
				<NetworkTable
					{model}
					{highlightedIds}
					onRowHover={handleTableHover}
					onRowSelect={handleTableSelect}
				/>
			</div>
		</div>
	</div>
</div>
