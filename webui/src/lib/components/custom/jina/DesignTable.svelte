<script lang="ts">
	/**
	 * Table of the results of a JINA iteration: one row per variant or pool scorer with its
	 * design id, target values (binding targets marked) and design variable values. Clicking
	 * a row selects its design.
	 */
	import * as Table from '$lib/components/ui/table/index.js';
	import type {
		JinaDesignInfo,
		JinaIterateResult,
		ProblemInfo,
		Target
	} from '$lib/gen/endpoints/DESDEOFastAPI';
	import { formatNumber, formatValue, targetLabel } from './labels';

	let {
		problem,
		targets,
		results,
		designs,
		designVariables,
		nameLabel = 'Variant',
		selectedId = $bindable(null)
	}: {
		problem: ProblemInfo;
		targets: Target[];
		results: JinaIterateResult[];
		designs: JinaDesignInfo[];
		designVariables: string[];
		nameLabel?: string;
		selectedId?: number | null;
	} = $props();

	let byId = $derived(new Map(designs.map((d) => [d.id, d])));
	let showBinding = $derived(results.some((r) => (r.binding ?? []).length > 0));
</script>

<div class="h-full overflow-auto">
	<Table.Root>
		<Table.Header>
			<Table.Row>
				<Table.Head>Design</Table.Head>
				<Table.Head>{nameLabel}</Table.Head>
				{#each targets as target (target.symbol)}
					<Table.Head class="text-right">{targetLabel(problem, target)}</Table.Head>
				{/each}
				{#each designVariables as symbol (symbol)}
					<Table.Head class="text-right text-gray-500">{symbol}</Table.Head>
				{/each}
			</Table.Row>
		</Table.Header>
		<Table.Body>
			{#each results as result, i (i)}
				{@const design = result.design_id != null ? byId.get(result.design_id) : undefined}
				<Table.Row
					class="cursor-pointer {result.design_id != null && result.design_id === selectedId
						? 'bg-blue-50'
						: ''}"
					onclick={() => {
						if (result.design_id != null) selectedId = result.design_id;
					}}
				>
					<Table.Cell>{result.design_id != null ? `#${result.design_id}` : '—'}</Table.Cell>
					<Table.Cell>
						{result.name}
						{#if !result.success}
							<span class="block text-xs text-red-700">{result.message}</span>
						{/if}
					</Table.Cell>
					{#each targets as target (target.symbol)}
						{@const binding = (result.binding ?? []).includes(target.symbol)}
						<Table.Cell
							class="text-right {binding ? 'font-semibold' : ''}"
							title={binding ? 'Binding at this reference point' : undefined}
						>
							{formatNumber(result.target_values?.[target.symbol])}{binding ? ' ●' : ''}
						</Table.Cell>
					{/each}
					{#each designVariables as symbol (symbol)}
						<Table.Cell class="text-right text-gray-600">
							{design ? formatValue(design.values[symbol]) : '—'}
						</Table.Cell>
					{/each}
				</Table.Row>
			{/each}
		</Table.Body>
	</Table.Root>
	{#if showBinding}
		<p class="px-2 py-1 text-xs text-gray-500">● binding target at this reference point</p>
	{/if}
</div>
