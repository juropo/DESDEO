<script lang="ts">
	/**
	 * Inputs of the robustness analysis of an evaluation over the objectives in its outcomes:
	 * domain-criterion thresholds per objective (prefilled with the midpoint of each objective's
	 * ideal–nadir range) and, when the setup has a baseline leaf, optional absolute and loss floors
	 * of the Taleb classes. Empty entries are left out of the request; the result is put in `result`.
	 * The entries are kept per evaluation in `inputs`.
	 */
	import { Button } from '$lib/components/ui/button';
	import { Label } from '$lib/components/ui/label';
	import type { JinaAnalysisResponse, ProblemInfo } from '$lib/gen/endpoints/DESDEOFastAPI';
	import ObjectivePointInputs from './ObjectivePointInputs.svelte';
	import { finitePoint, midpointPrefill, type Range } from './reference-point';
	import type { AnalysisInputs, AnalysisResult, EvaluateState } from './types';

	let {
		problem,
		evaluation,
		baselineLeaf,
		objectiveRanges,
		onAnalyse,
		result = $bindable(null),
		inputs = $bindable({})
	}: {
		problem: ProblemInfo;
		evaluation: EvaluateState | null;
		baselineLeaf: string | null;
		objectiveRanges: Record<string, Range>;
		onAnalyse: (
			evaluateStateId: number,
			request: {
				thresholds: Record<string, number>;
				absolute_floors: Record<string, number> | null;
				loss_floors: Record<string, number> | null;
			}
		) => Promise<JinaAnalysisResponse | null>;
		result?: AnalysisResult | null;
		inputs?: Record<number, AnalysisInputs>;
	} = $props();

	let saved = $derived(evaluation ? inputs[evaluation.state_id] : undefined);
	// The objectives with a value in the outcomes of the evaluation.
	let objectives = $derived([
		...new Set(
			Object.values(evaluation?.result.outcomes ?? {}).flatMap((leaves) =>
				Object.values(leaves).flatMap((outcome) => Object.keys(outcome.objectives ?? {}))
			)
		)
	]);
	let nLeaves = $derived(
		evaluation ? Object.keys(Object.values(evaluation.result.outcomes)[0] ?? {}).length : 0
	);

	// Creates the evaluation's entries once, prefilling thresholds without a saved value with the
	// midpoint of the objective's range if known.
	$effect(() => {
		if (!evaluation) return;
		const current = inputs[evaluation.state_id];
		const missing = objectives.filter((obj) => !current || !(obj in current.thresholds));
		if (current && missing.length === 0) return;
		inputs[evaluation.state_id] = {
			thresholds: { ...midpointPrefill(missing, objectiveRanges), ...current?.thresholds },
			absoluteFloors: current?.absoluteFloors ?? {},
			lossFloors: current?.lossFloors ?? {}
		};
	});

	function floors(values: Record<string, number | null>): Record<string, number> | null {
		const point = finitePoint(values, objectives);
		return baselineLeaf && Object.keys(point).length > 0 ? point : null;
	}

	async function run() {
		if (!evaluation || !saved) return;
		const used = finitePoint(saved.thresholds, objectives);
		const analysis = await onAnalyse(evaluation.state_id, {
			thresholds: used,
			absolute_floors: floors(saved.absoluteFloors),
			loss_floors: floors(saved.lossFloors)
		});
		if (analysis) {
			result = {
				evaluateStateId: evaluation.state_id,
				scenarioModelId: evaluation.result.scenario_model_id,
				analysis,
				objectives,
				thresholds: used,
				nLeaves
			};
		}
	}
</script>

{#if !evaluation || !saved}
	<p class="text-sm text-gray-500">Choose a finished evaluation to analyse.</p>
{:else}
	<div class="min-w-0 space-y-3">
		<div class="grid gap-1">
			<Label>Domain-criterion thresholds</Label>
			<p class="text-xs text-gray-500">The scenarios meeting each threshold are counted.</p>
			<ObjectivePointInputs
				{problem}
				{objectives}
				idPrefix="threshold"
				bind:values={saved.thresholds}
			/>
		</div>

		{#if baselineLeaf}
			<div class="grid gap-1">
				<Label>Absolute floors (optional)</Label>
				<p class="text-xs text-gray-500">
					Worst acceptable outcome; a design beyond it in any scenario is fragile. Leave empty for
					none.
				</p>
				<ObjectivePointInputs
					{problem}
					{objectives}
					idPrefix="absolute-floor"
					bind:values={saved.absoluteFloors}
				/>
			</div>
			<div class="grid gap-1">
				<Label>Loss floors (optional)</Label>
				<p class="text-xs text-gray-500">
					Largest acceptable loss relative to {baselineLeaf}, in the objective's units; a design
					losing more in any scenario is fragile. Leave empty for none.
				</p>
				<ObjectivePointInputs
					{problem}
					{objectives}
					idPrefix="loss-floor"
					bind:values={saved.lossFloors}
				/>
			</div>
		{/if}

		<Button size="sm" onclick={run}>Analyse evaluation {evaluation.state_id}</Button>
	</div>
{/if}
