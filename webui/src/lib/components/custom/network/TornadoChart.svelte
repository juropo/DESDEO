<script lang="ts">
	/**
	 * A reasonably simple tornado-diagram chart (graphical models plan, phase E: "sensitivity
	 * tornado ... a reasonably simple tornado-chart component"). Purely presentational — takes
	 * already-fetched/mapped `SensitivityResult` data (`sensitivity.ts`) and renders one horizontal
	 * bar per swept node, sorted by impact magnitude, plus a vertical reference line at `baseValue`.
	 *
	 * No SVG/d3 dependency: plain percentage-positioned `<div>`s against a shared domain
	 * (`tornadoDomain`), the same "flex/width, not a canvas library" approach `GraphNode.svelte`'s
	 * own posterior bars already use — consistent with this package's existing style rather than
	 * introducing a new charting primitive. `visualizations/horizontal-bar/` (D3, Svelte-4-style
	 * `export let` props) was checked first and not reused: different component-authoring
	 * convention (this package is Svelte 5 runes throughout) and built for a different job
	 * (two *draggable* bounds on one bar, not N independent, non-interactive range bars).
	 */
	import { sortByImpact, tornadoDomain, type SensitivityNodeResult } from './sensitivity';

	interface Props {
		results: SensitivityNodeResult[];
		baseValue: number;
		/** Optional unit suffix appended to every displayed number (e.g. `"%"`, `" DKK"`). */
		unit?: string;
		class?: string;
	}

	let { results, baseValue, unit = '', class: className = '' }: Props = $props();

	let sorted = $derived(sortByImpact(results));
	let domain = $derived(tornadoDomain(results, baseValue));

	function percent(value: number): number {
		const [min, max] = domain;
		if (max === min) return 50;
		return ((value - min) / (max - min)) * 100;
	}

	function format(value: number): string {
		return `${Number(value.toFixed(3))}${unit}`;
	}
</script>

<div class="flex flex-col gap-1.5 {className}">
	{#if sorted.length === 0}
		<p class="text-muted-foreground text-xs">No other nodes to sweep.</p>
	{:else}
		{#each sorted as result (result.node)}
			{@const barLeft = Math.min(percent(result.low), percent(result.high))}
			{@const barWidth = Math.abs(percent(result.high) - percent(result.low))}
			<div class="flex items-center gap-2 text-xs">
				<span class="w-24 shrink-0 truncate font-medium" title={result.node}>{result.node}</span>
				<div class="bg-muted relative h-4 min-w-0 flex-1 overflow-hidden rounded-sm">
					<div
						class="bg-primary/60 absolute inset-y-0 rounded-sm"
						style="left: {barLeft}%; width: {barWidth}%"
						title="{format(result.low)} – {format(result.high)}"
					></div>
					<div
						class="bg-foreground absolute inset-y-0 w-px"
						style="left: {percent(baseValue)}%"
						title="Base value: {format(baseValue)}"
					></div>
				</div>
				<span class="text-muted-foreground w-28 shrink-0 text-right tabular-nums">
					{format(result.low)} – {format(result.high)}
				</span>
			</div>
		{/each}
	{/if}
</div>
