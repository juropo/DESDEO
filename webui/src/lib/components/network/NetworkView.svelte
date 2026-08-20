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
	import { onMount, untrack } from 'svelte';
	import { SvelteFlow, Background, Controls, type Viewport } from '@xyflow/svelte';
	import '@xyflow/svelte/dist/base.css';
	import { Button } from '$lib/components/ui/button';
	import { SegmentedControl } from '$lib/components/custom/segmented-control';
	import IconRefreshCw from '@lucide/svelte/icons/refresh-cw';
	import NetworkTable from './NetworkTable.svelte';
	import { NETWORK_NODE_TYPES } from './nodes';
	import { computeElkLayout, layoutCoversModel } from './layout';
	import { computeDetailLevel } from './lod';
	import { ancestralSet } from './dag-algorithms';
	import type { NetworkModel } from './types';
	import type { NetworkFlowEdge, NetworkFlowNode } from './view-types';

	interface Props {
		/** Bindable so a computed layout (auto, or from "Re-layout") can be persisted by the
		 *  caller — see the plan's design decision on where compiled-model metadata lives. */
		model: NetworkModel;
		/** Pins the level of detail to `'compact'` regardless of container width/zoom — set this
		 *  when the caller already knows it's embedding into a small panel. */
		compact?: boolean;
		class?: string;
		onNodeSelect?: (id: string | null) => void;
		onNodeHover?: (id: string | null) => void;
	}

	let {
		model = $bindable(),
		compact = false,
		class: className = '',
		onNodeSelect,
		onNodeHover
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

	// Free (structural, backend-independent) slice of phase C: hovering/selecting a node lights
	// up its ancestor set. Full active-trail-given-evidence highlighting needs evidence state
	// that only exists once phase D's what-if machinery lands.
	let highlightFocusId = $derived(hoveredId ?? selectedId);
	let highlightedIds = $derived.by((): Set<string> | null => {
		if (!highlightFocusId) return null;
		return ancestralSet(model, [highlightFocusId]);
	});

	let flowNodes = $state.raw<NetworkFlowNode[]>([]);
	let flowEdges = $state.raw<NetworkFlowEdge[]>([]);

	// Single reconciliation point: reruns on structural changes (node/edge set), detail-level
	// changes (resize/zoom), and highlight changes (hover/select). Existing node positions
	// (including manual drags) are preserved via `prevPositions`; only genuinely new nodes fall
	// back to the persisted layout or the origin.
	$effect(() => {
		const nodes = model.nodes;
		const edges = model.edges;
		const layout = model.layout;
		const detail = detailLevel;
		const focusId = highlightFocusId;
		const highlights = highlightedIds;

		untrack(() => {
			const prevPositions = new Map(flowNodes.map((n) => [n.id, n.position]));
			flowNodes = nodes.map((n) => ({
				id: n.id,
				type: n.kind,
				position: prevPositions.get(n.id) ?? layout?.positions[n.id] ?? { x: 0, y: 0 },
				data: {
					kind: n.kind,
					label: n.label,
					states: n.states,
					description: n.description,
					detail,
					highlight: focusId
						? n.id === focusId
							? ('active' as const)
							: highlights?.has(n.id)
								? ('active' as const)
								: ('dimmed' as const)
						: ('normal' as const)
				}
			}));
			flowEdges = edges.map((e) => ({ id: e.id, source: e.source, target: e.target }));
		});
	});

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
	<div class="flex items-center justify-between gap-2">
		<SegmentedControl
			size="sm"
			bind:value={viewMode}
			options={[
				{ value: 'canvas', label: 'Canvas' },
				{ value: 'split', label: 'Split' },
				{ value: 'table', label: 'Table' }
			]}
		/>
		<Button variant="outline" size="sm" onclick={relayout} disabled={relayouting}>
			<IconRefreshCw class="size-3.5 {relayouting ? 'animate-spin' : ''}" />
			Re-layout
		</Button>
	</div>

	<!--
		Both panes stay permanently mounted; only visibility/sizing changes with `viewMode`. Using
		{#if} to swap DOM structure per mode would tear down and recreate the canvas container
		(and with it `containerEl`, the ResizeObserver target, and the SvelteFlow viewport) every
		time the user toggled modes — this way canvas zoom/pan state and the ResizeObserver both
		survive mode switches.
	-->
	<div class="flex min-h-0 flex-1 flex-col gap-2">
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
				highlightedId={highlightFocusId}
				onRowHover={handleTableHover}
				onRowSelect={handleTableSelect}
			/>
		</div>
	</div>
</div>
