<script lang="ts">
	/**
	 * The robustness tab: evaluate designs in a scenario model, pick an evaluation, see its
	 * outcomes by scenario on the objective axes, analyse it and chart the results. Scrolls
	 * vertically inside the tab.
	 */
	import VisualizationsPanel from '$lib/components/custom/visualizations-panel/visualizations-panel.svelte';
	import { PREFERENCE_TYPES } from '$lib/constants';
	import type {
		JinaAnalysisResponse,
		JinaDesignInfo,
		JinaScenarioModelInfo,
		ProblemInfo
	} from '$lib/gen/endpoints/DESDEOFastAPI';
	import AnalysisPanel from './AnalysisPanel.svelte';
	import RobustnessCharts from './RobustnessCharts.svelte';
	import ScenarioBreakdown from './ScenarioBreakdown.svelte';
	import ScenarioPicker from './ScenarioPicker.svelte';
	import StressTestPanel from './StressTestPanel.svelte';
	import { formatNumber, objectiveLabel, plural } from './labels';
	import type { Range } from './reference-point';
	import {
		evaluatedLeafObjectives,
		evaluatedLeaves,
		evaluatedObjectives
	} from './robustness-charts';
	import { scenarioColor, scenarioLines, scenarioProblem } from './scenario-chart';
	import type { AnalysisInputs, AnalysisResult, EvaluateState, SetupState } from './types';

	let {
		problem,
		setup,
		scenarioModels,
		designs,
		wishlist,
		evaluations,
		objectiveRanges,
		selectedId = $bindable(null),
		result = $bindable(null),
		analysisInputs = $bindable({}),
		onEvaluate,
		onAnalyse
	}: {
		problem: ProblemInfo;
		setup: SetupState;
		scenarioModels: JinaScenarioModelInfo[];
		designs: JinaDesignInfo[];
		wishlist: number[];
		evaluations: EvaluateState[];
		objectiveRanges: Record<string, Range>;
		selectedId?: number | null;
		result?: AnalysisResult | null;
		analysisInputs?: Record<number, AnalysisInputs>;
		onEvaluate: (request: {
			scenario_model_id: number;
			design_ids: number[];
			reference_point: Record<string, number> | null;
		}) => void;
		onAnalyse: (
			evaluateStateId: number,
			request: {
				thresholds: Record<string, number>;
				absolute_floors: Record<string, number> | null;
				loss_floors: Record<string, number> | null;
			}
		) => Promise<JinaAnalysisResponse | null>;
	} = $props();

	let evaluationId = $state<number | null>(null);
	// Scenarios plotted per evaluation; null plots all.
	let checked = $state<Record<number, string[] | null>>({});

	let finished = $derived(evaluations.filter((e) => e.is_ready && !e.error));
	let evaluation = $derived(
		finished.find((e) => e.state_id === evaluationId) ?? finished.at(-1) ?? null
	);
	let model = $derived(
		evaluation
			? scenarioModels.find((m) => m.id === evaluation.result.scenario_model_id)
			: undefined
	);
	let outcomes = $derived(evaluation?.result.outcomes ?? {});
	let leaves = $derived(model?.leaves ?? evaluatedLeaves(outcomes));
	let leafColors = $derived(Object.fromEntries(leaves.map((l, i) => [l, scenarioColor(i)])));
	let checkedLeaves = $derived(evaluation ? (checked[evaluation.state_id] ?? leaves) : []);
	let sameModel = $derived(evaluation?.result.scenario_model_id === setup.result.scenario_model_id);
	let baselineLeaf = $derived(sameModel ? setup.result.baseline_leaf : null);

	// Objective axes over the evaluated objectives; the setup's ranges apply only in its own model.
	let chart = $derived.by(() => {
		if (!evaluation) return null;
		const objectives = evaluatedObjectives(outcomes);
		const chartProblem = scenarioProblem(
			problem,
			(setup.result.targets?.per_scenario ?? []).filter((t) => objectives.includes(t.objective)),
			sameModel ? ((setup.result.ranges?.per_scenario ?? {}) as Record<string, Range>) : {},
			checkedLeaves
		);
		const lines = scenarioLines(
			[...evaluation.result.design_ids].sort((a, b) => a - b),
			evaluatedLeafObjectives(outcomes),
			checkedLeaves,
			leaves,
			chartProblem.objectives.map((o) => o.symbol),
			selectedId
		);
		return { problem: chartProblem, lines };
	});

	let selectedOutcomes = $derived(
		evaluation && selectedId !== null ? outcomes[String(selectedId)] : undefined
	);

	function statusOf(e: EvaluateState): string {
		if (!e.is_ready) return 'running…';
		return e.error ? 'failed' : plural(e.result.design_ids.length, 'design');
	}

	function modelName(id: number): string {
		return id === setup.result.scenario_model_id
			? `model #${id} (setup)`
			: `model #${id} (stress test)`;
	}
</script>

<!-- The scroller is positioned out of flow, so the layout's auto-sized grid row cannot grow to the
     content's height, and it anchors the chart's absolutely positioned tooltip. -->
<div class="relative h-full min-h-0 min-w-0">
	<div class="@container absolute inset-0 space-y-4 overflow-y-auto p-3">
		<div class="grid gap-4 @2xl:grid-cols-2 @5xl:grid-cols-3">
			<section class="min-w-0 rounded border bg-white p-3">
				<h3 class="mb-2 text-sm font-semibold">Evaluate designs</h3>
				<StressTestPanel
					{problem}
					setupModelId={setup.result.scenario_model_id}
					{scenarioModels}
					{designs}
					{wishlist}
					{objectiveRanges}
					{onEvaluate}
				/>
			</section>

			<section class="min-w-0 rounded border bg-white p-3">
				<h3 class="mb-2 text-sm font-semibold">Evaluations</h3>
				{#if evaluations.length === 0}
					<p class="text-sm text-gray-500">No evaluations yet.</p>
				{/if}
				<ul class="max-h-64 space-y-1 overflow-y-auto">
					{#each [...evaluations].reverse() as e (e.state_id)}
						<li>
							<button
								class="w-full rounded px-2 py-1 text-left text-sm {e.state_id ===
								evaluation?.state_id
									? 'bg-blue-50 font-semibold'
									: 'hover:bg-gray-50'}"
								disabled={!e.is_ready || !!e.error}
								title={e.error ?? undefined}
								onclick={() => (evaluationId = e.state_id)}
							>
								Evaluation {e.state_id} — {modelName(e.result.scenario_model_id)}
								<span class="text-xs font-normal {e.error ? 'text-red-700' : 'text-gray-500'}">
									· {statusOf(e)}
								</span>
							</button>
						</li>
					{/each}
				</ul>
				{#if evaluation?.result.reference_point}
					<p class="mt-2 text-xs text-gray-500">
						Fallback reference point:
						{Object.entries(evaluation.result.reference_point)
							.map(([obj, v]) => `${objectiveLabel(problem, obj)} ${formatNumber(v)}`)
							.join(', ')}
					</p>
				{/if}
			</section>

			<section class="min-w-0 rounded border bg-white p-3">
				<h3 class="mb-2 text-sm font-semibold">Analyse</h3>
				<AnalysisPanel
					{problem}
					{evaluation}
					{baselineLeaf}
					{objectiveRanges}
					{onAnalyse}
					bind:result
					bind:inputs={analysisInputs}
				/>
			</section>
		</div>

		{#if evaluation && chart}
			<section class="rounded border bg-white p-3">
				<h3 class="text-sm font-semibold">
					Outcomes of evaluation {evaluation.state_id} by scenario
				</h3>
				<div class="grid gap-3 @3xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
					<div class="h-80 min-w-0">
						{#if chart.lines.designIds.length > 0}
							<VisualizationsPanel
								problem={chart.problem}
								solutionsObjectiveValues={chart.lines.values}
								previousPreferenceType={PREFERENCE_TYPES.ReferencePoint}
								currentPreferenceType={PREFERENCE_TYPES.ReferencePoint}
								externalSelectedIndexes={chart.lines.selected}
								onSelectSolution={(index: number) => {
									selectedId = chart?.lines.designIds[index] ?? selectedId;
								}}
								lineLabels={Object.fromEntries(
									chart.lines.labels.map((label, i) => [String(i), label])
								)}
								colorByIndex={chart.lines.colors}
								showHeader={false}
								verticalAxisLabels
								compactTickLabels
								fitMargins
								minAxisSpacing={80}
							/>
						{:else}
							<p class="text-sm text-gray-500">No outcomes in the checked scenarios.</p>
						{/if}
					</div>
					<div class="min-w-0 space-y-3">
						<ScenarioPicker
							{leaves}
							{baselineLeaf}
							bind:checked={
								() => checkedLeaves,
								(next: string[]) => {
									if (evaluation) checked = { ...checked, [evaluation.state_id]: next };
								}
							}
						/>
						{#if selectedOutcomes}
							<div class="border-t pt-2">
								<h4 class="mb-1 text-sm font-semibold">Design #{selectedId} by scenario</h4>
								<ScenarioBreakdown
									{problem}
									{baselineLeaf}
									{leafColors}
									{checkedLeaves}
									leafObjectives={Object.fromEntries(
										Object.entries(selectedOutcomes)
											.filter(([, o]) => o.status)
											.map(([leaf, o]) => [leaf, o.objectives ?? {}])
									)}
									failed={Object.fromEntries(
										Object.entries(selectedOutcomes)
											.filter(([, o]) => !o.status)
											.map(([leaf, o]) => [leaf, o.message])
									)}
								/>
							</div>
						{:else}
							<p class="text-xs text-gray-500">Select an evaluated design to see its outcomes.</p>
						{/if}
					</div>
				</div>
			</section>
		{/if}

		{#if evaluation && result?.evaluateStateId === evaluation.state_id}
			<RobustnessCharts {problem} {result} {outcomes} {baselineLeaf} bind:selectedId />
		{/if}
	</div>
</div>
