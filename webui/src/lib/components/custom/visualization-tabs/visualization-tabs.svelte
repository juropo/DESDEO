<script lang="ts" module>
	import type { Snippet } from 'svelte';

	/** One problem-specific visualization, e.g. a map or a solution description. */
	export interface VisualizationTab {
		id: string;
		label: string;
		content: Snippet;
	}
</script>

<script lang="ts">
	/**
	 * Shows the problem-specific visualizations that are available as tabs. A single one is shown
	 * without a tab bar. If the selected tab goes away (e.g. the problem changed), the first one is
	 * shown instead.
	 */
	import * as Tabs from '$lib/components/ui/tabs';

	interface Props {
		tabs: VisualizationTab[];
		/** Id of the selected tab. Bindable, so a host can remember it. */
		value?: string;
		class?: string;
	}

	let { tabs, value = $bindable(''), class: className = '' }: Props = $props();

	let selected = $derived(tabs.some((t) => t.id === value) ? value : (tabs[0]?.id ?? ''));
</script>

{#if tabs.length === 1}
	<div class="h-full min-h-0 {className}">
		{@render tabs[0].content()}
	</div>
{:else if tabs.length > 1}
	<Tabs.Root
		value={selected}
		onValueChange={(next) => (value = next)}
		class="flex h-full min-h-0 flex-col {className}"
	>
		<Tabs.List class="shrink-0">
			{#each tabs as tab (tab.id)}
				<Tabs.Trigger value={tab.id}>{tab.label}</Tabs.Trigger>
			{/each}
		</Tabs.List>
		{#each tabs as tab (tab.id)}
			<Tabs.Content value={tab.id} class="min-h-0 flex-1">
				<!-- Only the active tab is mounted, so a hidden one doesn't render at zero size. -->
				{#if tab.id === selected}
					{@render tab.content()}
				{/if}
			</Tabs.Content>
		{/each}
	</Tabs.Root>
{/if}
