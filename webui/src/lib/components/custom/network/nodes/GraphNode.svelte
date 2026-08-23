<script lang="ts">
	/**
	 * The single node component registered for every kind (chance/decision/objective/
	 * deterministic) in `NETWORK_NODE_TYPES` (`./index.ts`). xyflow passes each node's own `type`
	 * as a prop, so one component that looks up `NODE_KIND_STYLE[type]` for its shape/color/icon
	 * replaces what used to be four near-identical wrapper files (ChanceNode/DecisionNode/
	 * DeterministicNode/ObjectiveNode.svelte) differing only in which icon they imported and
	 * which `NODE_KIND_STYLE` key they hardcoded.
	 */
	import { Handle, Position, type NodeProps } from '@xyflow/svelte';
	import IconPin from '@lucide/svelte/icons/pin';
	import IconCheck from '@lucide/svelte/icons/check';
	import { NODE_KIND_STYLE } from '../node-style';
	import { isStatePinned } from '../interaction';
	import type { NetworkNodeData } from '../view-types';
	import type { NetworkNodeKind } from '../types';

	let props: NodeProps = $props();
	let nodeData = $derived(props.data as NetworkNodeData);
	let selected = $derived(props.selected ?? false);
	// `NodeProps.type` is typed `any` by xyflow; the cast is what NETWORK_NODE_TYPES (./index.ts)
	// guarantees is safe -- every kind this component is registered under is a `NetworkNodeKind`.
	let kind = $derived(props.type as NetworkNodeKind);
	let style = $derived(NODE_KIND_STYLE[kind]);
	let Icon = $derived(style.icon);

	// Fully static -- pulled out of the template (rather than left as an inline `class="..."`
	// string) purely to keep that line under the project's 100-col printWidth.
	const EVIDENCE_BADGE_CLASS = `border-background bg-primary text-primary-foreground absolute
		-top-1.5 -right-1.5 z-10 flex size-4 items-center justify-center rounded-full border-2`;

	/** Clicking a state chip toggles evidence -- only wired up when `nodeData.editable` (chance
	 *  nodes always, decision nodes only in `decisionMode: 'manual'`; see `isNodeEditable` in
	 *  `interaction.ts`). `stopPropagation` keeps the click from also toggling node selection. */
	function handleStateClick(event: MouseEvent, state: string) {
		event.stopPropagation();
		nodeData.onStateClick?.(state);
	}

	/** Class string for one state chip at `'expanded'` detail -- factored out (rather than an
	 *  inline ternary in the template) to keep the template line under printWidth and to give
	 *  `isPinned` one place to be read from, instead of the three near-identical inline
	 *  `evidenceState === state || policyState === state` checks this replaces. */
	function expandedChipClass(pinned: boolean, editable: boolean): string {
		const tone = pinned ? 'border-primary text-primary' : 'border-border/70 text-muted-foreground';
		const interactive = editable ? 'cursor-pointer hover:border-primary/70' : 'cursor-default';
		return `bg-muted/50 relative overflow-hidden rounded-full border px-1.5 py-0.5
			text-[10px] leading-none transition-colors ${tone} ${interactive}`;
	}

	/** Class string for one bar segment at `'normal'` detail when a posterior is available. */
	function posteriorBarClass(pinned: boolean, editable: boolean): string {
		const tone = pinned ? 'bg-primary' : 'bg-primary/40';
		const interactive = editable ? 'cursor-pointer hover:bg-primary/70' : 'cursor-default';
		return `h-full transition-colors ${tone} ${interactive}`;
	}

	// Diamond uses clip-path on a background layer rather than rotating the whole node, so the
	// label stays horizontal and unclipped instead of needing a counter-rotation hack.
	let bgStyle = $derived(
		style.shape === 'diamond'
			? 'clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%);'
			: style.shape === 'ellipse'
				? 'border-radius: 9999px;'
				: 'border-radius: 0.5rem;'
	);
</script>

<Handle type="target" position={Position.Top} class="!bg-background !h-2 !w-2 !border-2" />
<div
	class="network-node group bg-card relative flex min-h-14 min-w-28 flex-col items-center justify-center gap-1 border-2 px-3 py-2 shadow-sm transition-opacity"
	class:opacity-35={nodeData.highlight === 'dimmed'}
	class:shadow-lg={nodeData.highlight === 'active' || selected}
	class:ring-2={nodeData.highlight === 'active' || selected}
	class:ring-offset-2={nodeData.highlight === 'active' || selected}
	class:ring-offset-background={nodeData.highlight === 'active' || selected}
	style:border-color={style.colorVar}
	style:--tw-ring-color={style.colorVar}
	style={style.shape !== 'double-rect' ? bgStyle : 'border-radius: 0.5rem;'}
>
	{#if style.shape === 'double-rect'}
		<div
			class="pointer-events-none absolute inset-[3px] rounded-[5px] border"
			style:border-color={style.colorVar}
		></div>
	{/if}

	{#if nodeData.evidenceState}
		<div class={EVIDENCE_BADGE_CLASS} title="Evidence: {nodeData.evidenceState}">
			<IconPin class="size-2.5" />
		</div>
	{/if}

	{#if nodeData.chrome}
		<div class="absolute -top-1.5 -left-1.5 z-10">
			{@render nodeData.chrome({ id: props.id, kind, label: nodeData.label })}
		</div>
	{/if}

	<div class="flex items-center gap-1.5">
		<Icon class="size-3.5 shrink-0" />
		<span class="text-foreground max-w-40 truncate text-xs font-medium" title={nodeData.label}>
			{nodeData.label}
		</span>
	</div>

	{#if kind === 'decision' && nodeData.policyState}
		<span class="text-primary flex items-center gap-0.5 text-[10px] leading-none font-medium">
			<IconCheck class="size-2.5" />
			{nodeData.policyState}
		</span>
	{/if}

	{#if nodeData.detail !== 'compact' && nodeData.states?.length}
		{#if nodeData.detail === 'expanded'}
			<div class="flex max-w-40 flex-wrap justify-center gap-1">
				{#each nodeData.states as state, index (state)}
					{@const prob = nodeData.posterior?.[index]}
					{@const pinned = isStatePinned(nodeData, state)}
					<button
						type="button"
						disabled={!nodeData.editable}
						class={expandedChipClass(pinned, nodeData.editable)}
						title={prob != null ? `${state}: ${Math.round(prob * 100)}%` : state}
						onclick={(event) => handleStateClick(event, state)}
					>
						{#if prob != null}
							<span
								class="bg-primary/20 pointer-events-none absolute inset-y-0 left-0"
								style:width="{Math.round(prob * 100)}%"
							></span>
						{/if}
						<span class="relative">{state}</span>
					</button>
				{/each}
			</div>
		{:else if nodeData.posterior}
			<div
				class="bg-muted/50 flex h-1.5 w-24 overflow-hidden rounded-full"
				title="Posterior distribution"
			>
				{#each nodeData.states as state, index (state)}
					{@const prob = nodeData.posterior[index] ?? 0}
					{@const pinned = isStatePinned(nodeData, state)}
					<button
						type="button"
						disabled={!nodeData.editable}
						class={posteriorBarClass(pinned, nodeData.editable)}
						style:width="{Math.round(prob * 100)}%"
						title="{state}: {Math.round(prob * 100)}%"
						onclick={(event) => handleStateClick(event, state)}
					></button>
				{/each}
			</div>
		{:else if nodeData.editable}
			<!-- No posterior yet (e.g. a decision node pre-solve), but still editable -- an
			     equal-width segment row keeps the click affordance available at this detail level
			     too, rather than only at 'expanded'. Plain text below is for genuinely
			     non-interactive nodes (objective/deterministic, or a decision in 'solver' mode). -->
			<div class="bg-muted/50 flex h-1.5 w-24 overflow-hidden rounded-full gap-px">
				{#each nodeData.states as state (state)}
					{@const pinned = isStatePinned(nodeData, state)}
					<button
						type="button"
						class="h-full flex-1 cursor-pointer transition-colors hover:bg-primary/70 {pinned
							? 'bg-primary'
							: 'bg-primary/30'}"
						title={state}
						onclick={(event) => handleStateClick(event, state)}
					></button>
				{/each}
			</div>
		{:else}
			<span class="text-muted-foreground text-[10px] leading-none">
				{nodeData.states.length} states
			</span>
		{/if}
	{/if}
</div>
<Handle type="source" position={Position.Bottom} class="!bg-background !h-2 !w-2 !border-2" />
