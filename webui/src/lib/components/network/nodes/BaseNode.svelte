<script lang="ts">
	/**
	 * Shared chrome for the four per-kind node components (Chance/Decision/Objective/Deterministic).
	 * Not registered in `nodeTypes` itself — each per-kind component is a thin wrapper that fixes
	 * `shape`/`colorVar`/`icon` and forwards `data`/`selected` here, keeping the visual language in
	 * one place while still giving xyflow four distinct component identities to dispatch on.
	 */
	import type { Component } from 'svelte';
	import { Handle, Position } from '@xyflow/svelte';
	import type { NetworkNodeData } from '../view-types';
	import type { NodeShape } from '../node-style';

	interface Props {
		data: NetworkNodeData;
		selected?: boolean;
		shape: NodeShape;
		colorVar: string;
		icon: Component<{ class?: string }>;
	}

	let { data, selected = false, shape, colorVar, icon: Icon }: Props = $props();

	// Diamond uses clip-path on a background layer rather than rotating the whole node, so the
	// label stays horizontal and unclipped instead of needing a counter-rotation hack.
	let bgStyle = $derived(
		shape === 'diamond'
			? 'clip-path: polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%);'
			: shape === 'ellipse'
				? 'border-radius: 9999px;'
				: 'border-radius: 0.5rem;'
	);
</script>

<Handle type="target" position={Position.Top} class="!bg-background !h-2 !w-2 !border-2" />
<div
	class="network-node group bg-card relative flex min-h-14 min-w-28 flex-col items-center justify-center gap-1 border-2 px-3 py-2 shadow-sm transition-opacity"
	class:opacity-35={data.highlight === 'dimmed'}
	class:shadow-lg={data.highlight === 'active' || selected}
	class:ring-2={data.highlight === 'active' || selected}
	class:ring-offset-2={data.highlight === 'active' || selected}
	class:ring-offset-background={data.highlight === 'active' || selected}
	style:border-color={colorVar}
	style:--tw-ring-color={colorVar}
	style={shape !== 'double-rect' ? bgStyle : 'border-radius: 0.5rem;'}
>
	{#if shape === 'double-rect'}
		<div
			class="pointer-events-none absolute inset-[3px] rounded-[5px] border"
			style:border-color={colorVar}
		></div>
	{/if}
	<div class="flex items-center gap-1.5">
		<Icon class="size-3.5 shrink-0" />
		<span class="text-foreground max-w-40 truncate text-xs font-medium" title={data.label}>
			{data.label}
		</span>
	</div>

	{#if data.detail !== 'compact' && data.states?.length}
		{#if data.detail === 'expanded'}
			<div class="flex max-w-40 flex-wrap justify-center gap-1">
				{#each data.states as state (state)}
					<span
						class="border-border/70 bg-muted/50 text-muted-foreground rounded-full border px-1.5 py-0.5 text-[10px] leading-none"
					>
						{state}
					</span>
				{/each}
			</div>
		{:else}
			<span class="text-muted-foreground text-[10px] leading-none">
				{data.states.length} states
			</span>
		{/if}
	{/if}
</div>
<Handle type="source" position={Position.Bottom} class="!bg-background !h-2 !w-2 !border-2" />
