<script lang="ts">
	/** Tables of a robustness analysis: max regret, domain-criterion counts and antifragility. */
	import * as Table from '$lib/components/ui/table/index.js';
	import type { ProblemInfo } from '$lib/gen/endpoints/DESDEOFastAPI';
	import { formatNumber, objectiveLabel } from './labels';
	import type { AnalysisResult } from './types';

	let {
		problem,
		result,
		baselineLeaf,
		selectedId = null
	}: {
		problem: ProblemInfo;
		result: AnalysisResult;
		baselineLeaf: string | null;
		selectedId?: number | null;
	} = $props();

	const TALEB_STYLES: Record<string, string> = {
		fragile: 'bg-red-100 text-red-800',
		robust: 'bg-gray-100 text-gray-800',
		antifragile: 'bg-green-100 text-green-800'
	};

	function designIds(record: Record<string, unknown>): string[] {
		return Object.keys(record);
	}
</script>

<div class="min-w-0 space-y-4 p-2">
	<p class="text-xs text-gray-500">Results of evaluation {result.evaluateStateId}.</p>
	<section>
		<h4 class="text-sm font-semibold">Max regret</h4>
		<Table.Root>
			<Table.Header>
				<Table.Row>
					<Table.Head>Design</Table.Head>
					{#each result.objectives as obj (obj)}
						<Table.Head class="text-right">{objectiveLabel(problem, obj)}</Table.Head>
					{/each}
					<Table.Head class="text-right">Mean</Table.Head>
				</Table.Row>
			</Table.Header>
			<Table.Body>
				{#each designIds(result.analysis.max_regret) as id (id)}
					{@const regret = result.analysis.max_regret[id]}
					<Table.Row class={Number(id) === selectedId ? 'bg-blue-50 font-semibold' : ''}>
						<Table.Cell>#{id}</Table.Cell>
						{#each result.objectives as obj (obj)}
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
		<p class="text-xs text-gray-500">Scenarios meeting each threshold, of {result.nLeaves}.</p>
		<Table.Root>
			<Table.Header>
				<Table.Row>
					<Table.Head>Design</Table.Head>
					{#each result.objectives as obj (obj)}
						<Table.Head class="text-right">{objectiveLabel(problem, obj)}</Table.Head>
					{/each}
					<Table.Head class="text-right">Mean</Table.Head>
				</Table.Row>
			</Table.Header>
			<Table.Body>
				{#each designIds(result.analysis.domain_criterion) as id (id)}
					{@const domain = result.analysis.domain_criterion[id]}
					<Table.Row class={Number(id) === selectedId ? 'bg-blue-50 font-semibold' : ''}>
						<Table.Cell>#{id}</Table.Cell>
						{#each result.objectives as obj (obj)}
							<Table.Cell class="text-right">{domain.counts[obj] ?? '—'}</Table.Cell>
						{/each}
						<Table.Cell class="text-right">{formatNumber(domain.mean_count)}</Table.Cell>
					</Table.Row>
				{/each}
			</Table.Body>
		</Table.Root>
	</section>

	{#if baselineLeaf && result.analysis.antifragility}
		{@const fragility = result.analysis.antifragility}
		<section>
			<h4 class="text-sm font-semibold">Antifragility relative to {baselineLeaf}</h4>
			<p class="text-xs text-gray-500">
				Index from −1 (only losses) to 1 (only gains), with gains / losses in scenarios.
			</p>
			<Table.Root>
				<Table.Header>
					<Table.Row>
						<Table.Head>Design</Table.Head>
						{#each result.objectives as obj (obj)}
							<Table.Head class="text-right">{objectiveLabel(problem, obj)}</Table.Head>
						{/each}
					</Table.Row>
				</Table.Header>
				<Table.Body>
					{#each designIds(fragility) as id (id)}
						<Table.Row class={Number(id) === selectedId ? 'bg-blue-50 font-semibold' : ''}>
							<Table.Cell>#{id}</Table.Cell>
							{#each result.objectives as obj (obj)}
								{@const summary = fragility[id].objectives[obj]}
								<Table.Cell class="text-right">
									{#if summary}
										{formatNumber(summary.antifragility, 3)}
										<span class="text-xs text-gray-500">
											({summary.n_gain}/{summary.n_loss})
										</span>
										{#if summary.taleb_class}
											<span class="ml-1 rounded px-1 text-xs {TALEB_STYLES[summary.taleb_class]}">
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
			{#if result.analysis.antifragile_objectives}
				<p class="mt-1 text-xs text-gray-600">
					Objectives with an upside for some design:
					{result.analysis.antifragile_objectives.length > 0
						? result.analysis.antifragile_objectives
								.map((o) => objectiveLabel(problem, o))
								.join(', ')
						: 'none'}
				</p>
			{/if}
		</section>
	{/if}
</div>
