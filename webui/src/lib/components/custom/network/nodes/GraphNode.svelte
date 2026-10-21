<script lang="ts">
	/**
	 * The node component for every kind in `NETWORK_NODE_TYPES` (`./index.ts`); shape, color and
	 * icon come from `NODE_KIND_STYLE[type]`.
	 */
	import { Handle, Position, type NodeProps } from '@xyflow/svelte';
	import IconPin from '@lucide/svelte/icons/pin';
	import IconCheck from '@lucide/svelte/icons/check';
	import IconArrowUp from '@lucide/svelte/icons/arrow-up';
	import IconArrowDown from '@lucide/svelte/icons/arrow-down';
	import IconInfo from '@lucide/svelte/icons/info';
	import { NODE_KIND_STYLE } from '../node-style';
	import { isStatePinned, solutionStateTier } from '../interaction';
	import { BAR_TRANSITION_MS, computeDelta, dominantDeltaIndex, formatUtility } from '../whatif';
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

	// `delta[i]` is posterior[i] minus the ghost baseline; `dominant` is the largest change worth
	// calling out (see `dominantDeltaIndex`).
	let delta = $derived(computeDelta(nodeData.ghostPosterior, nodeData.posterior));
	let dominant = $derived(dominantDeltaIndex(delta));

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

	// A followed solution's choice per state (see `solutionStateTier`): dashed/lighter, unlike a pin.
	type SolutionTier = ReturnType<typeof solutionStateTier>;

	// The compact policy line: the solution's certain choice, or `null` if it depends on more.
	let solutionChoice = $derived(
		nodeData.solutionPosterior
			? (nodeData.states?.find(
					(_, index) => solutionStateTier(nodeData.solutionPosterior, index) === 'certain'
				) ?? null)
			: undefined
	);

	/** Class string for one state chip at `'expanded'` detail. */
	function expandedChipClass(pinned: boolean, editable: boolean, tier: SolutionTier): string {
		const tone = pinned
			? 'border-primary text-primary'
			: tier === 'certain'
				? 'border-primary border-dashed text-primary'
				: tier === 'possible'
					? 'border-primary/40 border-dashed text-primary/70'
					: 'border-border/70 text-muted-foreground';
		const interactive = editable ? 'cursor-pointer hover:border-primary/70' : 'cursor-default';
		return `bg-muted/50 relative overflow-hidden rounded-full border px-1.5 py-0.5
			text-[10px] leading-none transition-colors ${tone} ${interactive}`;
	}

	/** Class string for one equal-width segment at `'normal'` detail when there's no posterior. */
	function segmentClass(pinned: boolean, tier: SolutionTier, hasSolution: boolean): string {
		const tone = pinned
			? 'bg-primary'
			: tier === 'certain'
				? 'bg-primary/60'
				: tier === 'possible'
					? 'bg-primary/25'
					: hasSolution
						? 'bg-primary/10'
						: 'bg-primary/30';
		return `hover:bg-primary/70 h-full flex-1 cursor-pointer transition-colors ${tone}`;
	}

	/** Tooltip text for one state the followed solution may choose. */
	function solutionTitle(state: string, prob: number, tier: SolutionTier): string {
		return tier === 'certain'
			? `${state}: From solution`
			: `${state}: From solution: ${Math.round(prob * 100)}%`;
	}

	/** Class string for one bar segment at `'normal'` detail when a posterior is available. Width
	 *  changes animate; duration comes from `BAR_TRANSITION_MS`. */
	function posteriorBarClass(pinned: boolean, editable: boolean): string {
		const tone = pinned ? 'bg-primary' : 'bg-primary/40';
		const interactive = editable ? 'cursor-pointer hover:bg-primary/70' : 'cursor-default';
		return `h-full transition-[width,background-color] ${tone} ${interactive}`;
	}

	/** Tooltip text for one state: probability, plus the signed change from the ghost baseline
	 *  when both are available, so the delta isn't conveyed by color alone. */
	function stateTitle(state: string, prob: number | undefined, ghost: number | undefined): string {
		if (prob == null) return state;
		const pct = Math.round(prob * 100);
		if (ghost == null) return `${state}: ${pct}%`;
		const deltaPct = Math.round((prob - ghost) * 100);
		const sign = deltaPct > 0 ? '+' : '';
		return `${state}: ${pct}% (baseline ${Math.round(ghost * 100)}%, ${sign}${deltaPct}pp)`;
	}

	// Diamond clips a background layer rather than rotating the node, so the label stays upright.
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
	title={nodeData.offPolicy ? "Off the solution's path — pin it" : undefined}
>
	{#if nodeData.offPolicy}
		<div
			class="pointer-events-none absolute -inset-1 rounded-[0.6rem] ring-2 ring-amber-500/50"
		></div>
	{/if}
	{#if style.shape === 'double-rect'}
		<div
			class="pointer-events-none absolute inset-[3px] rounded-[5px] border"
			style:border-color={style.colorVar}
		></div>
	{/if}

	{#if nodeData.evidenceState}
		<!-- "Observed", not "set to" -- this is conditioning evidence (pyAgrum `setEvidence`),
		     never a causal intervention; see EVIDENCE_MODE_NOTE in interaction.ts. -->
		<div class={EVIDENCE_BADGE_CLASS} title="Observed: {nodeData.evidenceState}">
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
	{:else if kind === 'decision' && nodeData.detail === 'compact' && solutionChoice !== undefined}
		<span class="text-muted-foreground flex items-center gap-0.5 text-[10px] leading-none">
			{#if solutionChoice !== null}
				<IconCheck class="size-2.5" />
				{solutionChoice} (solution)
			{:else}
				depends
			{/if}
		</span>
	{/if}

	{#if kind === 'objective' && nodeData.detail !== 'compact' && nodeData.utility != null}
		<span
			class="text-muted-foreground text-[10px] leading-none tabular-nums"
			title="Expected value ± standard deviation"
		>
			{formatUtility(nodeData.utility, nodeData.utilityStdDev)}
		</span>
	{/if}

	{#if nodeData.detail !== 'compact' && nodeData.states?.length}
		{#if nodeData.detail === 'expanded'}
			<div class="flex max-w-40 flex-wrap justify-center gap-1">
				{#each nodeData.states as state, index (state)}
					{@const prob = nodeData.posterior?.[index]}
					{@const ghost = nodeData.ghostPosterior?.[index]}
					{@const pinned = isStatePinned(nodeData, state)}
					{@const tier = solutionStateTier(nodeData.solutionPosterior, index)}
					<button
						type="button"
						disabled={!nodeData.editable}
						class={expandedChipClass(pinned, nodeData.editable, tier)}
						title={tier
							? solutionTitle(state, nodeData.solutionPosterior?.[index] ?? 0, tier)
							: stateTitle(state, prob, ghost)}
						onclick={(event) => handleStateClick(event, state)}
					>
						{#if ghost != null}
							<!-- Ghost mark: a dashed line at the baseline width, distinguishable from the
							     fill without relying on color. -->
							<span
								class="border-foreground/50 pointer-events-none absolute inset-y-0 border-l border-dashed"
								style:left="{Math.round(ghost * 100)}%"
							></span>
						{/if}
						{#if prob != null}
							<span
								class="bg-primary/20 pointer-events-none absolute inset-y-0 left-0 transition-[width]"
								style:width="{Math.round(prob * 100)}%"
								style:transition-duration="{BAR_TRANSITION_MS}ms"
							></span>
						{/if}
						<span class="relative">{state}</span>
					</button>
				{/each}
			</div>
		{:else if nodeData.posterior}
			<div class="flex flex-col items-center gap-0.5">
				<div
					class="bg-muted/50 flex h-1.5 w-24 overflow-hidden rounded-full"
					title="Posterior distribution"
				>
					{#each nodeData.states as state, index (state)}
						{@const prob = nodeData.posterior[index] ?? 0}
						{@const ghost = nodeData.ghostPosterior?.[index]}
						{@const pinned = isStatePinned(nodeData, state)}
						<button
							type="button"
							disabled={!nodeData.editable}
							class={posteriorBarClass(pinned, nodeData.editable)}
							style:width="{Math.round(prob * 100)}%"
							style:transition-duration="{BAR_TRANSITION_MS}ms"
							title={stateTitle(state, prob, ghost)}
							onclick={(event) => handleStateClick(event, state)}
						></button>
					{/each}
				</div>
				{#if nodeData.ghostPosterior}
					<!-- Ghost bar: the baseline distribution, outlined rather than filled. -->
					<div
						class="border-muted-foreground/50 flex h-1 w-24 overflow-hidden rounded-full border border-dashed"
						title="Prior/baseline distribution"
					>
						{#each nodeData.states as state, index (state)}
							{@const ghostProb = nodeData.ghostPosterior[index] ?? 0}
							<span
								class="border-muted-foreground/40 h-full border-r last:border-r-0"
								style:width="{Math.round(ghostProb * 100)}%"
								title={state}
							></span>
						{/each}
					</div>
				{/if}
			</div>
		{:else if nodeData.editable}
			<!-- No posterior yet but editable: equal-width segments keep states clickable. -->
			<div class="bg-muted/50 flex h-1.5 w-24 gap-px overflow-hidden rounded-full">
				{#each nodeData.states as state, index (state)}
					{@const pinned = isStatePinned(nodeData, state)}
					{@const tier = solutionStateTier(nodeData.solutionPosterior, index)}
					<button
						type="button"
						class={segmentClass(pinned, tier, !!nodeData.solutionPosterior)}
						title={tier
							? solutionTitle(state, nodeData.solutionPosterior?.[index] ?? 0, tier)
							: state}
						onclick={(event) => handleStateClick(event, state)}
					></button>
				{/each}
			</div>
		{:else}
			<span class="text-muted-foreground text-[10px] leading-none">
				{nodeData.states.length} states
			</span>
		{/if}

		{#if dominant !== null && delta}
			{@const d = delta[dominant]}
			<!-- Signed delta callout: an arrow shape and spelled-out sign pair with the color. -->
			<span
				class="flex items-center gap-0.5 text-[10px] leading-none font-medium {d > 0
					? 'text-emerald-600 dark:text-emerald-400'
					: 'text-rose-600 dark:text-rose-400'}"
				title="Largest change since the ghost baseline"
			>
				{#if d > 0}
					<IconArrowUp class="size-2.5" />
				{:else}
					<IconArrowDown class="size-2.5" />
				{/if}
				{nodeData.states[dominant]}
				{d > 0 ? '+' : ''}{Math.round(d * 100)}%
			</span>
		{:else if nodeData.nullExplanation}
			<span
				class="text-muted-foreground flex items-center gap-0.5 text-[10px] leading-none"
				title={nodeData.nullExplanation}
			>
				<IconInfo class="size-2.5" />
				unaffected
			</span>
		{/if}
	{/if}
</div>
<Handle type="source" position={Position.Bottom} class="!bg-background !h-2 !w-2 !border-2" />
