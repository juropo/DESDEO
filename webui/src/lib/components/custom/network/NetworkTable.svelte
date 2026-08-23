<script lang="ts">
	/**
	 * Table view of a `NetworkModel`, shown alongside the canvas. Per the plan: "beyond a dozen
	 * nodes, the table is what people read" — sortable by column, one row per node, parent/child
	 * counts and full state lists that would be cut off in the canvas at anything but 'expanded'
	 * detail.
	 */
	import * as Table from '$lib/components/ui/table';
	import { Badge } from '$lib/components/ui/badge';
	import type { NetworkModel, NetworkNodeModel } from './types';
	import { NODE_KIND_STYLE } from './node-style';

	interface Props {
		model: NetworkModel;
		/** Node ids to highlight, for row <-> canvas cross-highlighting -- the same merged set
		 *  (hover/select ancestor closure unioned with any externally supplied `highlight` prop)
		 *  the canvas uses, so both panes of a split view agree on what's highlighted. */
		highlightedIds?: Set<string> | null;
		onRowHover?: (id: string | null) => void;
		onRowSelect?: (id: string) => void;
	}

	let { model, highlightedIds = null, onRowHover, onRowSelect }: Props = $props();

	type SortKey = 'label' | 'kind' | 'parents' | 'children';
	let sortKey = $state<SortKey>('label');
	let sortAsc = $state(true);

	/** One O(nodes + edges) pass building every node's parent/child count, instead of the O(edges)
	 *  scan `counts()` used to redo per node (making the old `rows` derivation O(nodes * edges)). */
	function buildCounts(m: NetworkModel): Record<string, { parents: number; children: number }> {
		const counts: Record<string, { parents: number; children: number }> = {};
		for (const node of m.nodes) counts[node.id] = { parents: 0, children: 0 };
		for (const edge of m.edges) {
			if (counts[edge.target]) counts[edge.target].parents++;
			if (counts[edge.source]) counts[edge.source].children++;
		}
		return counts;
	}

	let rows = $derived.by(() => {
		const counts = buildCounts(model);
		const withCounts = model.nodes.map((node) => ({
			node,
			...(counts[node.id] ?? { parents: 0, children: 0 })
		}));
		const dir = sortAsc ? 1 : -1;
		return withCounts.sort((a, b) => {
			switch (sortKey) {
				case 'kind':
					return dir * a.node.kind.localeCompare(b.node.kind);
				case 'parents':
					return dir * (a.parents - b.parents);
				case 'children':
					return dir * (a.children - b.children);
				default:
					return dir * a.node.label.localeCompare(b.node.label);
			}
		});
	});

	function setSort(key: SortKey) {
		if (sortKey === key) {
			sortAsc = !sortAsc;
		} else {
			sortKey = key;
			sortAsc = true;
		}
	}

	function stateSummary(node: NetworkNodeModel): string {
		if (!node.states || node.states.length === 0) return '—';
		return node.states.join(', ');
	}
</script>

<div class="border-border overflow-auto rounded-md border">
	<Table.Root>
		<Table.Header>
			<Table.Row>
				<Table.Head class="cursor-pointer select-none" onclick={() => setSort('label')}>
					Node {sortKey === 'label' ? (sortAsc ? '▲' : '▼') : ''}
				</Table.Head>
				<Table.Head class="cursor-pointer select-none" onclick={() => setSort('kind')}>
					Kind {sortKey === 'kind' ? (sortAsc ? '▲' : '▼') : ''}
				</Table.Head>
				<Table.Head
					class="cursor-pointer text-right select-none"
					onclick={() => setSort('parents')}
				>
					Parents {sortKey === 'parents' ? (sortAsc ? '▲' : '▼') : ''}
				</Table.Head>
				<Table.Head
					class="cursor-pointer text-right select-none"
					onclick={() => setSort('children')}
				>
					Children {sortKey === 'children' ? (sortAsc ? '▲' : '▼') : ''}
				</Table.Head>
				<Table.Head>States</Table.Head>
			</Table.Row>
		</Table.Header>
		<Table.Body>
			{#each rows as row (row.node.id)}
				<Table.Row
					class="cursor-pointer {highlightedIds?.has(row.node.id) ? 'bg-accent/60' : ''}"
					onmouseenter={() => onRowHover?.(row.node.id)}
					onmouseleave={() => onRowHover?.(null)}
					onclick={() => onRowSelect?.(row.node.id)}
				>
					<Table.Cell class="font-medium" title={row.node.description}>{row.node.label}</Table.Cell>
					<Table.Cell>
						<Badge
							variant="outline"
							style="border-color: {NODE_KIND_STYLE[row.node.kind]
								.colorVar}; color: {NODE_KIND_STYLE[row.node.kind].colorVar};"
						>
							{NODE_KIND_STYLE[row.node.kind].label}
						</Badge>
					</Table.Cell>
					<Table.Cell class="text-right tabular-nums">{row.parents}</Table.Cell>
					<Table.Cell class="text-right tabular-nums">{row.children}</Table.Cell>
					<Table.Cell
						class="text-muted-foreground max-w-72 truncate"
						title={stateSummary(row.node)}
					>
						{stateSummary(row.node)}
					</Table.Cell>
				</Table.Row>
			{/each}
			{#if rows.length === 0}
				<Table.Row>
					<Table.Cell colspan={5} class="text-muted-foreground text-center">No nodes.</Table.Cell>
				</Table.Row>
			{/if}
		</Table.Body>
	</Table.Root>
</div>
