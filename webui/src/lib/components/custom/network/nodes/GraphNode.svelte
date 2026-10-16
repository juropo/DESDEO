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
	import IconArrowUp from '@lucide/svelte/icons/arrow-up';
	import IconArrowDown from '@lucide/svelte/icons/arrow-down';
	import IconInfo from '@lucide/svelte/icons/info';
	import { NODE_KIND_STYLE } from '../node-style';
	import { isStatePinned } from '../interaction';
	import { BAR_TRANSITION_MS, computeDelta, dominantDeltaIndex } from '../whatif';
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

	// Ghost/delta (phase D): `delta[i]` is this node's posterior[i] minus its ghost (prior-or-
	// last-committed) baseline -- computed here, from the two raw arrays `NetworkView` already
	// puts on `nodeData`, rather than threaded through as a third precomputed array, so this stays
	// the one place that owns "what does a bar look like" instead of splitting that across two
	// components. `dominant` is the single largest-magnitude per-state change worth calling out;
	// see `dominantDeltaIndex`'s own doc comment for why only one, not a full per-state list.
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

	/** Class string for one bar segment at `'normal'` detail when a posterior is available.
	 *  `transition-[width]` (alongside the existing `transition-colors`) is what animates a
	 *  posterior change (phase D: 200-300ms, "motion is what communicates change") -- applied to
	 *  `width` only, never to a node's canvas `position`, which is never animated. The actual
	 *  duration is set via `style:transition-duration` (see the two call sites below) from
	 *  `BAR_TRANSITION_MS`, not hardcoded in this class string, so that constant stays load-bearing
	 *  rather than a value nothing reads. */
	function posteriorBarClass(pinned: boolean, editable: boolean): string {
		const tone = pinned ? 'bg-primary' : 'bg-primary/40';
		const interactive = editable ? 'cursor-pointer hover:bg-primary/70' : 'cursor-default';
		return `h-full transition-[width,background-color] ${tone} ${interactive}`;
	}

	/** Tooltip text for one state -- current probability, plus the signed change from the ghost
	 *  baseline when both are available. "Baseline" (not "prior"): the ghost baseline is the
	 *  unconditioned prior only until the DM commits some evidence, after which it's that committed
	 *  scenario instead -- see `NetworkView`'s `ghostPosteriors` doc comment. Delta is surfaced here
	 *  (plain text, always present regardless of color vision) as well as via the visual
	 *  dominant-delta callout below, so the "pair colour with shape/position, never colour alone"
	 *  rule (phase D) has a fully-accessible fallback for every state, not just the single dominant
	 *  one. */
	function stateTitle(state: string, prob: number | undefined, ghost: number | undefined): string {
		if (prob == null) return state;
		const pct = Math.round(prob * 100);
		if (ghost == null) return `${state}: ${pct}%`;
		const deltaPct = Math.round((prob - ghost) * 100);
		const sign = deltaPct > 0 ? '+' : '';
		return `${state}: ${pct}% (baseline ${Math.round(ghost * 100)}%, ${sign}${deltaPct}pp)`;
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
	{/if}

	{#if nodeData.detail !== 'compact' && nodeData.states?.length}
		{#if nodeData.detail === 'expanded'}
			<div class="flex max-w-40 flex-wrap justify-center gap-1">
				{#each nodeData.states as state, index (state)}
					{@const prob = nodeData.posterior?.[index]}
					{@const ghost = nodeData.ghostPosterior?.[index]}
					{@const pinned = isStatePinned(nodeData, state)}
					<button
						type="button"
						disabled={!nodeData.editable}
						class={expandedChipClass(pinned, nodeData.editable)}
						title={stateTitle(state, prob, ghost)}
						onclick={(event) => handleStateClick(event, state)}
					>
						{#if ghost != null}
							<!-- Ghost mark: a dashed *line* at the prior/baseline width, distinct in
							     shape (a line vs. an area fill) from the solid current-value fill
							     below it -- not just a paler shade of the same color -- so the two
							     stay distinguishable without relying on color vision (phase D). -->
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
					<!-- Ghost bar: the same segmented layout, one tier down, outlined rather than
					     filled -- the whole-bar baseline this posterior is diffed against (phase D
					     item 1: "without a baseline a posterior conveys nothing"). -->
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
			<!-- No posterior yet (e.g. a decision node pre-solve), but still editable -- an
			     equal-width segment row keeps the click affordance available at this detail level
			     too, rather than only at 'expanded'. Plain text below is for genuinely
			     non-interactive nodes (objective/deterministic, or a decision in 'solver' mode). -->
			<div class="bg-muted/50 flex h-1.5 w-24 gap-px overflow-hidden rounded-full">
				{#each nodeData.states as state (state)}
					{@const pinned = isStatePinned(nodeData, state)}
					<button
						type="button"
						class="hover:bg-primary/70 h-full flex-1 cursor-pointer transition-colors {pinned
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

		{#if dominant !== null && delta}
			{@const d = delta[dominant]}
			<!-- Signed delta callout (phase D item 1/6): an up/down *arrow shape* pairs with the
			     sign/color rather than relying on color alone, and the sign is spelled out in text
			     too ("+"/no prefix for negative, since Math.round already carries the "-"). -->
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
