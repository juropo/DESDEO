<script lang="ts">
	/**
	 * Objective values of one design in each leaf scenario. Leaves listed in `failed` show their
	 * message instead of values.
	 */
	import * as Table from '$lib/components/ui/table/index.js';
	import type { ProblemInfo } from '$lib/gen/endpoints/DESDEOFastAPI';
	import { formatNumber, objectiveLabel } from './labels';

	let {
		problem,
		leafObjectives,
		failed = {},
		baselineLeaf = null
	}: {
		problem: ProblemInfo;
		leafObjectives: Record<string, Record<string, number>>;
		failed?: Record<string, string>;
		baselineLeaf?: string | null;
	} = $props();

	let leaves = $derived([...new Set([...Object.keys(leafObjectives), ...Object.keys(failed)])]);
	let objectives = $derived([
		...new Set(Object.values(leafObjectives).flatMap((values) => Object.keys(values)))
	]);
</script>

{#if leaves.length === 0}
	<p class="p-2 text-sm text-gray-500">No per-scenario values.</p>
{:else}
	<Table.Root>
		<Table.Header>
			<Table.Row>
				<Table.Head>Scenario</Table.Head>
				{#each objectives as symbol (symbol)}
					<Table.Head class="text-right">{objectiveLabel(problem, symbol)}</Table.Head>
				{/each}
			</Table.Row>
		</Table.Header>
		<Table.Body>
			{#each leaves as leaf (leaf)}
				<Table.Row>
					<Table.Cell>
						{leaf}
						{#if leaf === baselineLeaf}<span class="text-xs text-gray-500"> (baseline)</span>{/if}
					</Table.Cell>
					{#if failed[leaf] !== undefined}
						<Table.Cell colspan={objectives.length} class="text-xs text-red-700">
							{failed[leaf]}
						</Table.Cell>
					{:else}
						{#each objectives as symbol (symbol)}
							<Table.Cell class="text-right">
								{formatNumber(leafObjectives[leaf]?.[symbol])}
							</Table.Cell>
						{/each}
					{/if}
				</Table.Row>
			{/each}
		</Table.Body>
	</Table.Root>
{/if}
