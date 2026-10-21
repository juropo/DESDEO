<script lang="ts">
	/**
	 * `<NetworkView>` — structure-only network visualisation. No backend calls: renders a
	 * `NetworkModel`'s DAG on a Svelte Flow canvas plus a table view, with ancestor-set highlighting
	 * on hover/select.
	 *
	 * Layout is elk-computed on mount (if the model has no usable layout) or via "Re-layout", and
	 * written back onto the bindable `model` so callers can persist it.
	 *
	 * Level of detail (see `lod.ts`) follows container width and zoom, or is pinned by `compact`.
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
	import FitViewOnReady from './FitViewOnReady.svelte';
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
		NetworkPosteriors,
		NetworkUtilities,
		NetworkUtilityStdDevs
	} from './types';
	import type {
		HighlightState,
		NetworkChromeContext,
		NetworkFlowEdge,
		NetworkFlowNode
	} from './view-types';

	interface Props {
		/** Bindable so a computed layout can be persisted by the caller. */
		model: NetworkModel;
		/** Hard evidence pinned per node id. Bindable: editable nodes (see `isNodeEditable`) toggle
		 *  it on click, but a method UI or `<NetworkPanel>` is free to set/read it directly too —
		 *  e.g. to seed evidence from elsewhere in its own UI. */
		evidence?: NetworkEvidence;
		/** Per-node posterior probabilities to render as state bars. */
		posteriors?: NetworkPosteriors;
		/** Baseline distribution each posterior bar is diffed against (the prior, or the committed
		 *  evidence's posteriors). Caller-owned. */
		ghostPosteriors?: NetworkPosteriors;
		/** The evidence the caller considers accepted. Compared against `evidence` to enable the
		 *  Commit button; never mutated here. */
		committedEvidence?: NetworkEvidence;
		/** A solved policy's chosen state per decision node, shown read-only in
		 *  `decisionMode: 'solver'`. */
		policy?: NetworkPolicy;
		/** Expected utility (and its std dev) per objective node id, shown on objective nodes. */
		utilities?: NetworkUtilities;
		utilityStdDevs?: NetworkUtilityStdDevs;
		/** Per-state probability of each decision under a followed solution's policy, highlighted
		 *  on unpinned decision nodes (see `WhatIfResult.decisionPosteriors`). */
		decisionPosteriors?: NetworkPosteriors;
		/** Followed decisions moved off the solution's path, flagged on their nodes. */
		offPolicyDecisions?: string[];
		/** True while a what-if refresh is in flight or debounced; dims the canvas and table so
		 *  partial updates don't show. Caller-driven. */
		stale?: boolean;
		/** `'solver'` (default): decision nodes display `policy`'s choice, read-only. `'manual'`:
		 *  decision nodes become clickable like chance nodes, writing the chosen state into
		 *  `evidence` instead. */
		decisionMode?: DecisionMode;
		/** Externally driven highlight set, unioned with the hover/select ancestor highlighting. */
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
		/** Fired when the DM clicks "Commit evidence". What committing means is up to the caller. */
		onCommitEvidence?: (evidence: NetworkEvidence) => void;
	}

	let {
		model = $bindable(),
		evidence = $bindable({}),
		posteriors,
		ghostPosteriors,
		committedEvidence = {},
		policy,
		utilities,
		utilityStdDevs,
		decisionPosteriors,
		offPolicyDecisions,
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
	let layoutVersion = $state(0);

	let detailLevel = $derived(computeDetailLevel(containerWidth, viewport.zoom, compact));

	// Rebuilt only when nodes/edges change, not on every hover/select.
	let adjacency = $derived.by((): Adjacency => buildAdjacency(model));

	// Hovering/selecting a node lights up its ancestor set.
	let highlightFocusId = $derived(hoveredId ?? selectedId);
	// Unioned with the externally supplied `highlight` prop.
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

	// Structural reconciliation. Deliberately not rerun on highlight changes, which the lighter
	// effect below handles. Existing node positions, including manual drags, are preserved.
	$effect(() => {
		const nodes = model.nodes;
		const edges = model.edges;
		const layout = model.layout;
		const detail = detailLevel;
		const evidenceSnapshot = evidence;
		const committedSnapshot = committedEvidence;
		const policySnapshot = policy;
		const utilitiesSnapshot = utilities;
		const utilityStdDevsSnapshot = utilityStdDevs;
		const decisionPosteriorsSnapshot = decisionPosteriors;
		const offPolicySnapshot = offPolicyDecisions;
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
						solutionPosterior:
							n.kind === 'decision' && evidenceSnapshot[n.id] === undefined
								? decisionPosteriorsSnapshot?.[n.id]
								: undefined,
						offPolicy: evidenceSnapshot[n.id] === undefined && offPolicySnapshot?.includes(n.id),
						utility: utilitiesSnapshot?.[n.id],
						utilityStdDev: utilityStdDevsSnapshot?.[n.id],
						// Explains a zero delta via d-separation from the evidence added since the
						// commit, given the committed evidence.
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
			layoutVersion++;
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
			<!-- Evidence is always conditioned on (pyAgrum `setEvidence`), never an intervention or
			     soft evidence. `aria-label` carries the full sentence. -->
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
		Both panes stay mounted; only visibility changes with `viewMode`, so canvas zoom/pan and the
		ResizeObserver survive mode switches.
	-->
	<div class="relative flex min-h-0 flex-1 flex-col gap-2">
		<!-- Dims the whole pane so a partial update doesn't show. -->
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
					<FitViewOnReady trigger={layoutVersion} />
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
