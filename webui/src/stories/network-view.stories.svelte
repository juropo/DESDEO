<script module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { NetworkView } from '$lib/components/network';

	const { Story } = defineMeta({
		title: 'Example/NetworkView',
		component: NetworkView,
		tags: ['autodocs']
	});
</script>

<script lang="ts">
	import { sampleIrrigationNetwork } from '$lib/components/network/sample-network';

	// `model` is `$bindable` on NetworkView (layout gets written back onto it), so each story
	// needs its own local state rather than a shared `args.model` — see NetworkView's docs.
	let defaultModel = $state(structuredClone(sampleIrrigationNetwork));
	let compactModel = $state(structuredClone(sampleIrrigationNetwork));
	let noLayoutModel = $state({
		nodes: structuredClone(sampleIrrigationNetwork.nodes),
		edges: structuredClone(sampleIrrigationNetwork.edges)
	});
</script>

<Story name="Default">
	<div style="height: 640px;">
		<NetworkView bind:model={defaultModel} />
	</div>
</Story>

<Story name="Compact panel">
	<div style="height: 480px; width: 300px;">
		<NetworkView bind:model={compactModel} compact />
	</div>
</Story>

<Story name="No persisted layout (auto elk layout on mount)">
	<div style="height: 640px;">
		<NetworkView bind:model={noLayoutModel} />
	</div>
</Story>
