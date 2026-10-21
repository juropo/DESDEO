<script lang="ts">
	import { onMount } from 'svelte';
	import { BaseLayout } from '$lib/components/custom/method_layout/index.js';
	import { methodSelection } from '../../../stores/methodSelection';
	import { errorMessage, isLoading } from '../../../stores/uiState';
	import LoadingSpinner from '$lib/components/custom/notifications/loading-spinner.svelte';
	import Alert from '$lib/components/custom/notifications/alert.svelte';
	import { SegmentedControl } from '$lib/components/custom/segmented-control';
	import * as Resizable from '$lib/components/ui/resizable/index.js';
	import ResizableHandle from '$lib/components/ui/resizable/resizable-handle.svelte';
	import * as Select from '$lib/components/ui/select/index.js';
	import * as Tabs from '$lib/components/ui/tabs/index.js';
	import { Button } from '$lib/components/ui/button';
	import ParallelCoordinates from '$lib/components/visualizations/parallel-coordinates/parallel-coordinates.svelte';

	import SetupDialog from '$lib/components/custom/jina/SetupDialog.svelte';
	import ReferencePointEditor from '$lib/components/custom/jina/ReferencePointEditor.svelte';
	import DesignTable from '$lib/components/custom/jina/DesignTable.svelte';
	import ScenarioBreakdown from '$lib/components/custom/jina/ScenarioBreakdown.svelte';
	import WishlistPanel from '$lib/components/custom/jina/WishlistPanel.svelte';
	import StressTestPanel from '$lib/components/custom/jina/StressTestPanel.svelte';
	import AnalysisPanel from '$lib/components/custom/jina/AnalysisPanel.svelte';
	import JobStatus from '$lib/components/custom/jina/JobStatus.svelte';
	import {
		midpoint,
		objectiveRanges,
		type Range
	} from '$lib/components/custom/jina/reference-point';
	import { MEASURE_LABELS, targetLabel } from '$lib/components/custom/jina/labels';

	import type { ProblemInfo } from '$lib/gen/endpoints/DESDEOFastAPI';
	import type {
		EvaluateState,
		IterateState,
		JinaDesignInfo,
		JinaScenarioModelInfo,
		JinaStateResponse,
		SelectState,
		SetupState,
		Target,
		View
	} from './types';
	import {
		analyse,
		evaluate,
		fetchDesigns,
		loadSession,
		importDesigns,
		iterate,
		select,
		startSetup
	} from './handlers';

	const { data } = $props<{ data: { problems: ProblemInfo[] } }>();
	let problemList = $derived(data.problems ?? []);

	type Tab = View | 'pool';
	const TABS: { value: Tab; label: string }[] = [
		{ value: 'aggregated', label: 'Aggregated' },
		{ value: 'per_scenario', label: 'Per-scenario' },
		{ value: 'pool', label: 'Pool' }
	];
	const VIEW_OPTIONS: { value: View; label: string }[] = [
		{ value: 'aggregated', label: 'Aggregated' },
		{ value: 'per_scenario', label: 'Per-scenario' }
	];

	let problem = $state<ProblemInfo | null>(null);
	let scenarioModels = $state<JinaScenarioModelInfo[]>([]);
	let setup = $state<SetupState | null>(null);
	let states = $state<JinaStateResponse[]>([]);
	let designs = $state<JinaDesignInfo[]>([]);
	let pending = $state<{ id: number; label: string }[]>([]);

	let showSetup = $state(false);
	let setupJobId = $state<number | null>(null);
	let designsRequest = 0;
	let tab = $state<Tab>('aggregated');
	let poolView = $state<View>('aggregated');
	let selectedDesignId = $state<number | null>(null);
	let shownIteration = $state<Record<Tab, string>>({ aggregated: '', per_scenario: '', pool: '' });
	let referencePoints = $state<Record<View, Record<string, number>>>({
		aggregated: {},
		per_scenario: {}
	});

	let ready = $derived(!!setup && setup.is_ready && !setup.error && !!setup.result.ranges);
	let view = $derived<View>(tab === 'pool' ? poolView : tab);
	let targets = $derived<Target[]>(setup?.result.targets?.[view] ?? []);
	let ranges = $derived<Record<string, Range>>(
		(setup?.result.ranges?.[view] ?? {}) as Record<string, Range>
	);

	let iterations = $derived(
		states.filter((s): s is IterateState => s.result.kind === 'jina.iterate')
	);
	let evaluations = $derived(
		states.filter((s): s is EvaluateState => s.result.kind === 'jina.evaluate')
	);
	let lastSelect = $derived(
		states.filter((s): s is SelectState => s.result.kind === 'jina.select').at(-1)
	);
	let wishlist = $derived(lastSelect?.result.wishlist ?? []);
	let finalId = $derived(lastSelect?.result.final_design_id ?? null);

	let tabIterations = $derived(
		iterations.filter((s) =>
			tab === 'pool' ? s.result.mode === 'pool' : s.result.mode === 'solve' && s.result.view === tab
		)
	);
	let iteration = $derived(
		tabIterations.find((s) => String(s.state_id) === shownIteration[tab]) ?? tabIterations.at(-1)
	);
	let iterationTargets = $derived<Target[]>(
		iteration?.result.view ? (setup?.result.targets?.[iteration.result.view] ?? []) : []
	);

	// The latest reference point iterated with in each view.
	let previousPoints = $derived.by(() => {
		const result: Partial<Record<View, Record<string, number>>> = {};
		for (const s of iterations) {
			if (s.result.view) result[s.result.view] = s.result.reference_point;
		}
		return result;
	});

	let objRanges = $derived.by(() => {
		const result: Record<string, Range> = {};
		for (const objective of problem?.objectives ?? []) {
			if (objective.ideal != null && objective.nadir != null) {
				result[objective.symbol] = [objective.ideal, objective.nadir];
			}
		}
		const perScenario = setup?.result.targets?.per_scenario ?? [];
		const perScenarioRanges = (setup?.result.ranges?.per_scenario ?? {}) as Record<string, Range>;
		return { ...result, ...objectiveRanges(perScenario, perScenarioRanges) };
	});

	let selectedDesign = $derived(designs.find((d) => d.id === selectedDesignId) ?? null);
	let representativeSets = $derived(
		(problem?.problem_metadata?.representative_nd_metadata ?? [])
			.filter((s) => s.id != null)
			.map((s) => ({ id: s.id as number, name: s.name }))
	);
	let setupModel = $derived(scenarioModels.find((m) => m.id === setup?.result.scenario_model_id));

	// Starts each view's reference point at its previous one, or at the middle of each range.
	$effect(() => {
		if (!ready || !setup) return;
		for (const v of ['aggregated', 'per_scenario'] as View[]) {
			const viewTargets = setup.result.targets?.[v] ?? [];
			if (Object.keys(referencePoints[v]).length > 0 || viewTargets.length === 0) continue;
			const viewRanges = (setup.result.ranges?.[v] ?? {}) as Record<string, Range>;
			const start =
				previousPoints[v] ??
				Object.fromEntries(viewTargets.map((t) => [t.symbol, midpoint(viewRanges[t.symbol])]));
			referencePoints[v] = { ...start };
		}
	});

	onMount(async () => {
		const id = $methodSelection.selectedProblemId;
		if (!id) return;
		problem = problemList.find((p: ProblemInfo) => String(p.id) === String(id)) ?? null;
		if (!problem) return;

		const { models, restored } = await loadSession(problem.id);
		scenarioModels = models ?? [];
		if (!restored) return;
		if (!restored.setup) {
			showSetup = true;
			return;
		}
		applySetup(restored.setup as SetupState, restored.states);
		if (!ready) setupJobId = restored.setup.state_id;
	});

	function applySetup(state: SetupState, children: JinaStateResponse[] = []) {
		setup = state;
		states = children;
		designs = [];
		selectedDesignId = null;
		referencePoints = { aggregated: {}, per_scenario: {} };
		shownIteration = { aggregated: '', per_scenario: '', pool: '' };
		pending = children
			.filter((s) => !s.is_ready)
			.map((s) => ({ id: s.state_id, label: jobLabel(s) }));
		if (ready) refreshDesigns();
	}

	// Only the response to the latest request for the current setup is kept.
	async function refreshDesigns() {
		if (!problem || !setup) return;
		const request = ++designsRequest;
		const setupId = setup.state_id;
		const result = await fetchDesigns(problem.id, setupId);
		if (result && request === designsRequest && setup?.state_id === setupId) designs = result;
	}

	function jobLabel(state: JinaStateResponse): string {
		if (state.result.kind === 'jina.evaluate') return `Evaluation ${state.state_id}`;
		if (state.result.kind === 'jina.iterate' && state.result.mode === 'import') {
			return `Import ${state.state_id}`;
		}
		return `Iteration ${state.state_id}`;
	}

	function upsert(state: JinaStateResponse) {
		const index = states.findIndex((s) => s.state_id === state.state_id);
		states = index < 0 ? [...states, state] : states.with(index, state);
	}

	function track(state: JinaStateResponse) {
		upsert(state);
		if (!state.is_ready) pending = [...pending, { id: state.state_id, label: jobLabel(state) }];
		else afterIterate(state);
	}

	function jobDone(state: JinaStateResponse) {
		upsert(state);
		if (!state.error) {
			pending = pending.filter((p) => p.id !== state.state_id);
			afterIterate(state);
		}
	}

	function afterIterate(state: JinaStateResponse) {
		if (state.result.kind !== 'jina.iterate') return;
		refreshDesigns();
		const first = state.result.results.find((r) => r.design_id != null)?.design_id;
		if (state.result.mode !== 'import' && first != null) selectedDesignId = first;
	}

	async function runIterate() {
		if (!problem || !setup) return;
		const point = Object.fromEntries(
			targets
				.filter((t) => referencePoints[view][t.symbol] !== undefined)
				.map((t) => [t.symbol, referencePoints[view][t.symbol]])
		);
		const state = await iterate({
			problem_id: problem.id,
			setup_state_id: setup.state_id,
			mode: tab === 'pool' ? 'pool' : 'solve',
			view,
			reference_point: point
		});
		if (state) {
			shownIteration[tab] = '';
			track(state);
		}
	}

	async function saveSelection(nextWishlist: number[], nextFinal: number | null) {
		if (!problem || !setup) return;
		const state = await select({
			problem_id: problem.id,
			setup_state_id: setup.state_id,
			wishlist: nextWishlist,
			final_design_id: nextFinal
		});
		if (state) upsert(state);
	}

	async function runImport(setId: number, referencePoint: Record<string, number>) {
		if (!problem || !setup) return;
		const state = await importDesigns({
			problem_id: problem.id,
			setup_state_id: setup.state_id,
			representative_set_id: setId,
			reference_point: referencePoint
		});
		if (state) track(state);
	}

	async function runEvaluate(request: {
		scenario_model_id: number;
		design_ids: number[];
		reference_point: Record<string, number> | null;
	}) {
		if (!problem || !setup) return;
		const state = await evaluate({
			problem_id: problem.id,
			setup_state_id: setup.state_id,
			...request
		});
		if (state) track(state);
	}

	function runAnalysis(
		evaluateStateId: number,
		request: {
			thresholds: Record<string, number>;
			absolute_floors: Record<string, number> | null;
			loss_floors: Record<string, number> | null;
		}
	) {
		if (!problem || !setup) return Promise.resolve(null);
		return analyse({
			problem_id: problem.id,
			setup_state_id: setup.state_id,
			evaluate_state_id: evaluateStateId,
			...request
		});
	}

	function setupDone(state: JinaStateResponse) {
		if (state.error) return;
		setupJobId = null;
		applySetup(state as SetupState);
	}

	function iterationLabel(s: IterateState, index: number): string {
		const viewLabel =
			tab === 'pool' && s.result.view ? ` (${s.result.view.replace('_', '-')})` : '';
		return `Iteration ${index + 1}${viewLabel}`;
	}

	let pcDimensions = $derived(
		problem
			? iterationTargets.map((t) => ({
					symbol: t.symbol,
					name: targetLabel(problem!, t),
					direction: (t.maximize ? 'max' : 'min') as 'max' | 'min'
				}))
			: []
	);
	let emptyMessage = $derived(
		iteration?.result.mode === 'pool'
			? 'No unpicked designs left.'
			: 'No designs in this iteration.'
	);
	let pcRows = $derived(
		(iteration?.result.results ?? []).filter((r) => r.success && r.design_id != null)
	);
	let pcSelected = $derived.by(() => {
		const index = pcRows.findIndex((r) => r.design_id === selectedDesignId);
		return index < 0 ? null : index;
	});
</script>

<svelte:head>
	<title>JINA | DESDEO</title>
	<meta
		name="description"
		content="This page implements the JINA interactive method for scenario-based design in DESDEO"
	/>
</svelte:head>

{#if $isLoading}
	<LoadingSpinner />
{/if}

{#if $errorMessage}
	<Alert title="Error" message={$errorMessage} variant="destructive" />
{/if}

{#if problem}
	<SetupDialog
		bind:open={showSetup}
		problemId={problem.id}
		{scenarioModels}
		onStart={startSetup}
		onStarted={(state) => (setupJobId = state.state_id)}
	/>
{/if}

<BaseLayout
	showLeftSidebar={!!problem}
	showRightSidebar={ready}
	leftSidebarWidth="20rem"
	rightSidebarWidth="24rem"
	bottomPanelTitle={tab === 'pool' ? 'Pool picks' : 'Iteration results'}
>
	{#snippet leftSidebar()}
		<div class="space-y-4 p-3">
			{#if setup && setupModel}
				<div class="space-y-1 text-xs text-gray-600">
					<p>
						Scenario model #{setupModel.id}: {setupModel.leaves.join(', ')}
					</p>
					<p>
						Aggregation: {MEASURE_LABELS[setup.result.measure]}{setup.result.measure === 'cvar'
							? ` (α = ${setup.result.cvar_alpha})`
							: ''}
					</p>
					<p>Design variables: {setup.result.design_variables.join(', ')}</p>
					<p>Baseline leaf: {setup.result.baseline_leaf ?? 'none'}</p>
				</div>
			{/if}
			<Button variant="outline" size="sm" onclick={() => (showSetup = true)}>
				{setup ? 'New setup' : 'Set up'}
			</Button>
			{#if setupJobId !== null}
				<JobStatus
					stateId={setupJobId}
					label="Computing target ranges"
					onDone={setupDone}
					onDismiss={() => (setupJobId = null)}
				/>
			{/if}

			{#if ready && problem}
				{#if tab === 'pool'}
					<div class="space-y-1">
						<span class="text-sm font-medium">Targets of view</span>
						<SegmentedControl bind:value={poolView} options={VIEW_OPTIONS} size="sm" />
					</div>
				{/if}
				<h3 class="text-sm font-semibold">Reference point</h3>
				<ReferencePointEditor
					{problem}
					{targets}
					{ranges}
					previous={previousPoints[view] ?? null}
					bind:value={referencePoints[view]}
				/>
				<Button class="w-full" onclick={runIterate}>
					{tab === 'pool' ? 'Pick from pool' : 'Solve'}
				</Button>
			{/if}
		</div>
	{/snippet}

	{#snippet explorerTitle()}
		<span class="whitespace-nowrap">Solution Explorer</span>
	{/snippet}

	{#snippet explorerControls()}
		{#if ready}
			<!-- Wraps onto further lines rather than overflowing a narrow pane. -->
			<div class="flex min-w-0 flex-wrap items-center justify-end gap-2">
				<SegmentedControl bind:value={tab} options={TABS} />
				{#if tabIterations.length > 0 && iteration}
					<Select.Root type="single" bind:value={shownIteration[tab]}>
						<Select.Trigger class="w-56 max-w-full min-w-0">
							<span class="min-w-0 truncate">
								{iterationLabel(iteration, tabIterations.indexOf(iteration))}
							</span>
						</Select.Trigger>
						<Select.Content>
							{#each tabIterations as s, i (s.state_id)}
								<Select.Item value={String(s.state_id)}>{iterationLabel(s, i)}</Select.Item>
							{/each}
						</Select.Content>
					</Select.Root>
				{/if}
				{#if finalId !== null}
					<span
						class="shrink-0 rounded bg-green-100 px-1.5 py-0.5 text-xs whitespace-nowrap text-green-800"
						title="Final design"
					>
						Final #{finalId}
					</span>
				{/if}
			</div>
		{/if}
	{/snippet}

	{#snippet visualizationArea()}
		{#if !problem}
			<div class="flex h-full items-center justify-center text-gray-500">
				Select a problem with a scenario model to use JINA.
			</div>
		{:else if !ready}
			<div class="flex h-full items-center justify-center text-gray-500">Set up JINA to start.</div>
		{:else if !iteration}
			<div class="flex h-full items-center justify-center text-gray-500">
				{tab === 'pool'
					? 'Pick designs from the pool of designs found so far.'
					: 'Give a reference point and solve.'}
			</div>
		{:else}
			<Resizable.PaneGroup direction="horizontal" class="h-full">
				<Resizable.Pane defaultSize={60} minSize={30} class="h-full p-2">
					{#if !iteration.is_ready}
						<div class="flex h-full items-center justify-center text-gray-500">Solving…</div>
					{:else if pcRows.length === 0}
						<div class="flex h-full items-center justify-center text-gray-500">
							{iteration.error ?? emptyMessage}
						</div>
					{:else}
						<ParallelCoordinates
							data={pcRows.map((r) => r.target_values ?? {})}
							dimensions={pcDimensions}
							lineLabels={Object.fromEntries(
								pcRows.map((r, i) => [String(i), `#${r.design_id} ${r.name}`])
							)}
							referenceData={{
								referencePoint: {
									values: iteration.result.reference_point,
									label: 'Reference point'
								}
							}}
							selectedIndex={pcSelected}
							verticalAxisLabels
							compactTickLabels
							fitMargins
							onLineSelect={(index) => {
								if (index !== null) selectedDesignId = pcRows[index]?.design_id ?? null;
							}}
						/>
					{/if}
				</Resizable.Pane>
				<ResizableHandle withHandle />
				<Resizable.Pane defaultSize={40} minSize={20} class="h-full">
					<div class="h-full min-w-0 overflow-auto p-2">
						{#if selectedDesign}
							<h3 class="mb-1 text-sm font-semibold">Design #{selectedDesign.id} by scenario</h3>
							<ScenarioBreakdown
								{problem}
								leafObjectives={selectedDesign.leaf_objectives ?? {}}
								baselineLeaf={setup?.result.baseline_leaf ?? null}
							/>
						{:else}
							<p class="text-sm text-gray-500">Select a design to see its values by scenario.</p>
						{/if}
					</div>
				</Resizable.Pane>
			</Resizable.PaneGroup>
		{/if}
	{/snippet}

	{#snippet numericalValues()}
		{#if problem && setup && iteration?.is_ready && iteration.result.results.length === 0}
			<p class="p-2 text-sm text-gray-500">{iteration.error ?? emptyMessage}</p>
		{:else if problem && setup && iteration}
			<DesignTable
				{problem}
				targets={iterationTargets}
				results={iteration.result.results}
				{designs}
				designVariables={setup.result.design_variables}
				nameLabel={tab === 'pool' ? 'Scorer' : 'Variant'}
				bind:selectedId={selectedDesignId}
			/>
		{/if}
	{/snippet}

	{#snippet rightSidebar()}
		{#if problem && setup}
			{#key setup.state_id}
				<div class="min-w-0 space-y-3 p-3">
					{#each pending as job (job.id)}
						<JobStatus
							stateId={job.id}
							label={job.label}
							onDone={jobDone}
							onDismiss={() => (pending = pending.filter((p) => p.id !== job.id))}
						/>
					{/each}
					<Tabs.Root value="designs" class="min-w-0">
						<Tabs.List class="w-full">
							<Tabs.Trigger value="designs">Designs</Tabs.Trigger>
							<Tabs.Trigger value="evaluate">Evaluate</Tabs.Trigger>
							<Tabs.Trigger value="analysis">Analysis</Tabs.Trigger>
						</Tabs.List>
						<Tabs.Content value="designs" class="min-w-0 pt-2">
							<WishlistPanel
								{problem}
								{designs}
								{wishlist}
								{finalId}
								bind:selectedId={selectedDesignId}
								{representativeSets}
								importObjectives={setupModel?.objectives ?? []}
								objectiveRanges={objRanges}
								onSave={saveSelection}
								onImport={runImport}
							/>
						</Tabs.Content>
						<Tabs.Content value="evaluate" class="min-w-0 pt-2">
							<StressTestPanel
								{problem}
								setupModelId={setup.result.scenario_model_id}
								baselineLeaf={setup.result.baseline_leaf}
								{scenarioModels}
								{designs}
								{wishlist}
								{evaluations}
								objectiveRanges={objRanges}
								onEvaluate={runEvaluate}
							/>
						</Tabs.Content>
						<Tabs.Content value="analysis" class="min-w-0 pt-2">
							<AnalysisPanel
								{problem}
								{evaluations}
								baselineLeaf={setup.result.baseline_leaf}
								objectiveRanges={objRanges}
								onAnalyse={runAnalysis}
							/>
						</Tabs.Content>
					</Tabs.Root>
				</div>
			{/key}
		{/if}
	{/snippet}
</BaseLayout>
