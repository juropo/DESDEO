<script lang="ts">
	/**
	 * Evaluates designs in the leaves of a scenario model of the problem: the setup's own model,
	 * or another one as a stress test. For another model a fallback reference point over that
	 * model's objectives is sent for the objectives a design's own reference point does not cover;
	 * empty entries are left out.
	 */
	import * as Select from '$lib/components/ui/select/index.js';
	import { Button } from '$lib/components/ui/button';
	import Checkbox from '$lib/components/ui/checkbox/checkbox.svelte';
	import { Label } from '$lib/components/ui/label';
	import type {
		JinaDesignInfo,
		JinaScenarioModelInfo,
		ProblemInfo
	} from '$lib/gen/endpoints/DESDEOFastAPI';
	import ObjectivePointInputs from './ObjectivePointInputs.svelte';
	import { finitePoint, midpointPrefill, type Range } from './reference-point';
	import { modelLabel } from './labels';

	let {
		problem,
		setupModelId,
		scenarioModels,
		designs,
		wishlist,
		objectiveRanges,
		onEvaluate
	}: {
		problem: ProblemInfo;
		setupModelId: number;
		scenarioModels: JinaScenarioModelInfo[];
		designs: JinaDesignInfo[];
		wishlist: number[];
		objectiveRanges: Record<string, Range>;
		onEvaluate: (request: {
			scenario_model_id: number;
			design_ids: number[];
			reference_point: Record<string, number> | null;
		}) => void;
	} = $props();

	let modelId = $state('');
	let chosen = $state<number[]>([]);
	let fallback = $state<Record<string, number | null>>({});

	$effect(() => {
		if (!modelId) modelId = String(setupModelId);
	});
	// Prefills each objective of the chosen model once, with the midpoint of its range if known.
	$effect(() => {
		const missing = (model?.objectives ?? []).filter((obj) => !(obj in fallback));
		if (missing.length === 0) return;
		fallback = {
			...fallback,
			...midpointPrefill(missing, objectiveRanges)
		};
	});
	$effect(() => {
		const known = new Set(designs.map((d) => d.id));
		const kept = chosen.filter((id) => known.has(id));
		if (kept.length !== chosen.length) chosen = kept;
	});

	let isOtherModel = $derived(modelId !== String(setupModelId));
	let model = $derived(scenarioModels.find((m) => String(m.id) === modelId));

	function label(m: JinaScenarioModelInfo, full = false): string {
		return m.id === setupModelId ? `${modelLabel(m, full)} (setup)` : modelLabel(m, full);
	}

	function run() {
		if (!model) return;
		onEvaluate({
			scenario_model_id: model.id,
			design_ids: chosen,
			reference_point: isOtherModel ? finitePoint(fallback, model.objectives) : null
		});
	}
</script>

<div class="min-w-0 space-y-3">
	<div class="grid grid-cols-[minmax(0,1fr)] gap-1">
		<Label for="stress-model">Scenario model</Label>
		<Select.Root type="single" bind:value={modelId}>
			<Select.Trigger id="stress-model" class="w-full min-w-0">
				<span class="min-w-0 truncate" title={model ? label(model, true) : ''}>
					{model ? label(model) : 'Select a scenario model'}
				</span>
			</Select.Trigger>
			<Select.Content>
				{#each scenarioModels as m (m.id)}
					<Select.Item value={String(m.id)} title={label(m, true)}>
						<span class="truncate">{label(m)}</span>
					</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>
	</div>

	<div class="grid grid-cols-[minmax(0,1fr)] gap-1">
		<div class="flex items-center justify-between">
			<span class="text-sm font-medium">Designs</span>
			<span class="flex gap-1">
				<Button
					variant="ghost"
					size="sm"
					class="h-6 px-2 text-xs"
					onclick={() => (chosen = [...wishlist])}>Wishlist</Button
				>
				<Button
					variant="ghost"
					size="sm"
					class="h-6 px-2 text-xs"
					onclick={() => (chosen = designs.map((d) => d.id))}>All</Button
				>
			</span>
		</div>
		<div class="flex max-h-28 flex-wrap gap-x-3 gap-y-1 overflow-y-auto">
			{#each designs as design (design.id)}
				<div class="flex items-center gap-1">
					<Checkbox
						id="stress-design-{design.id}"
						checked={chosen.includes(design.id)}
						onCheckedChange={(checked) =>
							(chosen = checked ? [...chosen, design.id] : chosen.filter((id) => id !== design.id))}
					/>
					<label for="stress-design-{design.id}" class="text-sm">#{design.id}</label>
				</div>
			{/each}
		</div>
	</div>

	{#if isOtherModel}
		<div class="grid grid-cols-[minmax(0,1fr)] gap-1">
			<span class="text-sm font-medium">Fallback reference point</span>
			<p class="text-xs text-gray-500">
				Used for objectives a design's own reference point does not cover in this model.
			</p>
			<ObjectivePointInputs
				{problem}
				objectives={model?.objectives ?? []}
				idPrefix="fallback"
				bind:values={fallback}
			/>
		</div>
	{/if}

	<Button size="sm" disabled={!model || chosen.length === 0} onclick={run}>Evaluate</Button>
</div>
