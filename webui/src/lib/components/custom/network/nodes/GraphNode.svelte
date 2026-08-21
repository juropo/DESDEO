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
	import { NODE_KIND_STYLE } from '../node-style';
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
	<div class="flex items-center gap-1.5">
		<Icon class="size-3.5 shrink-0" />
		<span class="text-foreground max-w-40 truncate text-xs font-medium" title={nodeData.label}>
			{nodeData.label}
		</span>
	</div>

	{#if nodeData.detail !== 'compact' && nodeData.states?.length}
		{#if nodeData.detail === 'expanded'}
			<div class="flex max-w-40 flex-wrap justify-center gap-1">
				{#each nodeData.states as state (state)}
					<span
						class="border-border/70 bg-muted/50 text-muted-foreground rounded-full border px-1.5 py-0.5 text-[10px] leading-none"
					>
						{state}
					</span>
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
