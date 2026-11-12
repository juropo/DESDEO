<script module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { NetworkView } from '$lib/components/custom/network';

	const { Story } = defineMeta({
		title: 'Example/NetworkView',
		component: NetworkView,
		tags: ['autodocs']
	});
</script>

<script lang="ts">
	import { sampleIrrigationNetwork } from '$lib/components/custom/network/sample-network';
	import type {
		NetworkEvidence,
		NetworkPolicy,
		NetworkPosteriors
	} from '$lib/components/custom/network';

	// `model` is `$bindable` on NetworkView (layout gets written back onto it), so each story
	// needs its own local state rather than a shared `args.model` — see NetworkView's docs.
	let defaultModel = $state(structuredClone(sampleIrrigationNetwork));
	let compactModel = $state(structuredClone(sampleIrrigationNetwork));
	let noLayoutModel = $state({
		nodes: structuredClone(sampleIrrigationNetwork.nodes),
		edges: structuredClone(sampleIrrigationNetwork.edges)
	});

	// Phase B embedding contract: evidence/posteriors/policy/decisionMode/nodeChrome.
	let whatIfModel = $state(structuredClone(sampleIrrigationNetwork));
	let whatIfEvidence: NetworkEvidence = $state({ weather: 'dry' });
	const whatIfPosteriors: NetworkPosteriors = {
		weather: [1, 0, 0],
		soil_moisture: [0.6, 0.3, 0.1],
		crop_yield: [0.5, 0.35, 0.15]
	};

	let solverModel = $state(structuredClone(sampleIrrigationNetwork));
	let solverEvidence: NetworkEvidence = $state({});
	const solverPolicy: NetworkPolicy = { irrigate: 'light' };
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

<Story name="Manual evidence with posteriors">
	<div style="height: 640px;">
		<NetworkView
			bind:model={whatIfModel}
			bind:evidence={whatIfEvidence}
			posteriors={whatIfPosteriors}
			decisionMode="manual"
		/>
	</div>
</Story>

<Story name="Solver mode with a fixed policy and injected chrome">
	<div style="height: 640px;">
		<NetworkView
			bind:model={solverModel}
			bind:evidence={solverEvidence}
			policy={solverPolicy}
			decisionMode="solver"
		>
			{#snippet nodeChrome({ kind })}
				{#if kind === 'objective'}
					<span
						class="flex size-4 items-center justify-center rounded-full border bg-background
							text-[9px] shadow-sm"
						title="Injected by the embedding method UI (nodeChrome snippet)"
					>
						★
					</span>
				{/if}
			{/snippet}
		</NetworkView>
	</div>
</Story>
