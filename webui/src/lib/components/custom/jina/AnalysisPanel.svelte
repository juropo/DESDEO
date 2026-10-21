<script lang="ts">
	/**
	 * Robustness analysis of an evaluation over the objectives in its outcomes: max regret, the
	 * domain criterion at thresholds per objective (prefilled with the midpoint of each objective's
	 * ideal–nadir range) and, when the setup has a baseline leaf, antifragility with Taleb classes
	 * set by optional absolute and loss floors. Empty entries are left out of the request.
	 */
	import * as Select from '$lib/components/ui/select/index.js';
	import * as Table from '$lib/components/ui/table/index.js';
	import { Button } from '$lib/components/ui/button';
	import { Label } from '$lib/components/ui/label';
	import type { JinaAnalysisResponse, ProblemInfo } from '$lib/gen/endpoints/DESDEOFastAPI';
	import ObjectivePointInputs from './ObjectivePointInputs.svelte';
	import { finitePoint, midpointPrefill, type Range } from './reference-point';
	import { formatNumber, objectiveLabel } from './labels';
	import type { EvaluateState } from './types';

	let {
		problem,
		evaluations,
		baselineLeaf,
		objectiveRanges,
		onAnalyse
	}: {
		problem: ProblemInfo;
		evaluations: EvaluateState[];
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
	} = $props();

	const TALEB_STYLES: Record<string, string> = {
		fragile: 'bg-red-100 text-red-800',
		robust: 'bg-gray-100 text-gray-800',
		antifragile: 'bg-green-100 text-green-800'
	};

	let evaluationId = $state('');
	let thresholds = $state<Record<string, number | null>>({});
	let absoluteFloors = $state<Record<string, number | null>>({});
	let lossFloors = $state<Record<string, number | null>>({});
	let analysis = $state<JinaAnalysisResponse | null>(null);
	let analysedId = $state<number | null>(null);
	let analysedObjectives = $state<string[]>([]);

	let finished = $derived(evaluations.filter((e) => e.is_ready && !e.error));
	let evaluation = $derived(
		finished.find((e) => String(e.state_id) === evaluationId) ?? finished.at(-1)
	);
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

	// Prefills each threshold once, with the midpoint of the objective's range if known.
	$effect(() => {
		const missing = objectives.filter((obj) => !(obj in thresholds));
		if (missing.length === 0) return;
		thresholds = {
			...thresholds,
			...midpointPrefill(missing, objectiveRanges)
		};
	});

	function floors(values: Record<string, number | null>): Record<string, number> | null {
		const point = finitePoint(values, objectives);
		return baselineLeaf && Object.keys(point).length > 0 ? point : null;
	}

	async function run() {
		if (!evaluation) return;
		const result = await onAnalyse(evaluation.state_id, {
			thresholds: finitePoint(thresholds, objectives),
			absolute_floors: floors(absoluteFloors),
			loss_floors: floors(lossFloors)
		});
		if (result) {
			analysis = result;
			analysedId = evaluation.state_id;
			analysedObjectives = objectives;
		}
	}

	function designIds(record: Record<string, unknown>): string[] {
		return Object.keys(record);
	}
</script>

{#if finished.length === 0 || !evaluation}
	<p class="text-sm text-gray-500">Evaluate designs first to analyse their robustness.</p>
{:else}
	<div class="min-w-0 space-y-3">
		<Select.Root type="single" bind:value={evaluationId}>
			<Select.Trigger class="w-full text-xs">
				Evaluation {evaluation.state_id} — model #{evaluation.result.scenario_model_id}
			</Select.Trigger>
			<Select.Content>
				{#each finished as e (e.state_id)}
					<Select.Item value={String(e.state_id)}>
						Evaluation {e.state_id} — model #{e.result.scenario_model_id}
					</Select.Item>
				{/each}
			</Select.Content>
		</Select.Root>

		<div class="grid gap-1">
			<Label>Domain-criterion thresholds</Label>
			<p class="text-xs text-gray-500">The scenarios meeting each threshold are counted.</p>
			<ObjectivePointInputs {problem} {objectives} idPrefix="threshold" bind:values={thresholds} />
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
					bind:values={absoluteFloors}
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
					bind:values={lossFloors}
				/>
			</div>
		{/if}

		<Button size="sm" onclick={run}>Analyse</Button>

		{#if analysis && analysedId !== null}
			<div class="space-y-4 border-t pt-2">
				<p class="text-xs text-gray-500">Results of evaluation {analysedId}.</p>
				<section>
					<h4 class="text-sm font-semibold">Max regret</h4>
					<Table.Root>
						<Table.Header>
							<Table.Row>
								<Table.Head>Design</Table.Head>
								{#each analysedObjectives as obj (obj)}
									<Table.Head class="text-right">{objectiveLabel(problem, obj)}</Table.Head>
								{/each}
								<Table.Head class="text-right">Mean</Table.Head>
							</Table.Row>
						</Table.Header>
						<Table.Body>
							{#each designIds(analysis.max_regret) as id (id)}
								{@const regret = analysis.max_regret[id]}
								<Table.Row>
									<Table.Cell>#{id}</Table.Cell>
									{#each analysedObjectives as obj (obj)}
										<Table.Cell class="text-right">{formatNumber(regret.regret[obj])}</Table.Cell>
									{/each}
									<Table.Cell class="text-right">{formatNumber(regret.mean_regret)}</Table.Cell>
								</Table.Row>
							{/each}
						</Table.Body>
					</Table.Root>
				</section>

				<section>
					<h4 class="text-sm font-semibold">Domain criterion</h4>
					<p class="text-xs text-gray-500">Scenarios meeting each threshold, of {nLeaves}.</p>
					<Table.Root>
						<Table.Header>
							<Table.Row>
								<Table.Head>Design</Table.Head>
								{#each analysedObjectives as obj (obj)}
									<Table.Head class="text-right">{objectiveLabel(problem, obj)}</Table.Head>
								{/each}
								<Table.Head class="text-right">Mean</Table.Head>
							</Table.Row>
						</Table.Header>
						<Table.Body>
							{#each designIds(analysis.domain_criterion) as id (id)}
								{@const domain = analysis.domain_criterion[id]}
								<Table.Row>
									<Table.Cell>#{id}</Table.Cell>
									{#each analysedObjectives as obj (obj)}
										<Table.Cell class="text-right">{domain.counts[obj] ?? '—'}</Table.Cell>
									{/each}
									<Table.Cell class="text-right">{formatNumber(domain.mean_count)}</Table.Cell>
								</Table.Row>
							{/each}
						</Table.Body>
					</Table.Root>
				</section>

				{#if baselineLeaf && analysis.antifragility}
					{@const fragility = analysis.antifragility}
					<section>
						<h4 class="text-sm font-semibold">Antifragility relative to {baselineLeaf}</h4>
						<p class="text-xs text-gray-500">
							Index from −1 (only losses) to 1 (only gains), with gains / losses in scenarios.
						</p>
						<Table.Root>
							<Table.Header>
								<Table.Row>
									<Table.Head>Design</Table.Head>
									{#each analysedObjectives as obj (obj)}
										<Table.Head class="text-right">{objectiveLabel(problem, obj)}</Table.Head>
									{/each}
								</Table.Row>
							</Table.Header>
							<Table.Body>
								{#each designIds(fragility) as id (id)}
									<Table.Row>
										<Table.Cell>#{id}</Table.Cell>
										{#each analysedObjectives as obj (obj)}
											{@const summary = fragility[id].objectives[obj]}
											<Table.Cell class="text-right">
												{#if summary}
													{formatNumber(summary.antifragility, 3)}
													<span class="text-xs text-gray-500">
														({summary.n_gain}/{summary.n_loss})
													</span>
													{#if summary.taleb_class}
														<span
															class="ml-1 rounded px-1 text-xs {TALEB_STYLES[summary.taleb_class]}"
														>
															{summary.taleb_class}
														</span>
													{/if}
												{:else}
													—
												{/if}
											</Table.Cell>
										{/each}
									</Table.Row>
								{/each}
							</Table.Body>
						</Table.Root>
						{#if analysis.antifragile_objectives}
							<p class="mt-1 text-xs text-gray-600">
								Objectives with an upside for some design:
								{analysis.antifragile_objectives.length > 0
									? analysis.antifragile_objectives
											.map((o) => objectiveLabel(problem, o))
											.join(', ')
									: 'none'}
							</p>
						{/if}
					</section>
				{/if}
			</div>
		{/if}
	</div>
{/if}
