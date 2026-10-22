<script lang="ts">
	/**
	 * Checkboxes for the leaf scenarios whose outcomes the chart shows, each with its colour; the
	 * baseline leaf is marked ★.
	 */
	import { Button } from '$lib/components/ui/button';
	import Checkbox from '$lib/components/ui/checkbox/checkbox.svelte';
	import { scenarioColor } from './scenario-chart';

	let {
		leaves,
		baselineLeaf = null,
		checked = $bindable([])
	}: {
		leaves: string[];
		baselineLeaf?: string | null;
		checked?: string[];
	} = $props();

	function toggle(leaf: string, on: boolean) {
		// Kept in leaf order so lines and legend follow the scenario model.
		checked = leaves.filter((l) => (l === leaf ? on : checked.includes(l)));
	}
</script>

<div class="space-y-1">
	<div class="flex items-center justify-between gap-2">
		<span class="text-sm font-semibold">Scenarios in chart</span>
		<span class="flex gap-1">
			<Button
				variant="ghost"
				size="sm"
				class="h-6 px-2 text-xs"
				onclick={() => (checked = [...leaves])}>All</Button
			>
			<Button variant="ghost" size="sm" class="h-6 px-2 text-xs" onclick={() => (checked = [])}
				>None</Button
			>
		</span>
	</div>
	<p class="text-xs text-gray-500">
		{checked.length === 0
			? 'None checked: the chart shows the target values.'
			: 'The chart shows the objective values of each design in the checked scenarios.'}
	</p>
	<div class="flex max-h-28 flex-wrap gap-x-3 gap-y-1 overflow-y-auto">
		{#each leaves as leaf, i (leaf)}
			<div class="flex min-w-0 items-center gap-1">
				<Checkbox
					id="scenario-{leaf}"
					checked={checked.includes(leaf)}
					onCheckedChange={(on) => toggle(leaf, !!on)}
				/>
				<span
					class="size-3 shrink-0 rounded-sm"
					style="background-color: {scenarioColor(i)}"
					aria-hidden="true"
				></span>
				<label
					for="scenario-{leaf}"
					class="max-w-[9rem] cursor-pointer truncate text-xs"
					style="color: {scenarioColor(i)}"
					title={leaf === baselineLeaf ? `${leaf} (baseline leaf)` : leaf}
				>
					{leaf === baselineLeaf ? '★ ' : ''}{leaf}
				</label>
			</div>
		{/each}
	</div>
</div>
