<script lang="ts">
	import { untrack } from 'svelte';
	import { useSvelteFlow } from '@xyflow/svelte';

	/**
	 * Re-fits the viewport after an ELK relayout. `fitView()` is deferred by a macrotask because
	 * SvelteFlow's node-bounds cache is still stale in the same update. `trigger` is bumped after
	 * each relayout; `untrack` keeps `fitView()`'s own writes from re-triggering the effect.
	 */
	let { trigger }: { trigger: number } = $props();

	const { fitView } = useSvelteFlow();

	$effect(() => {
		if (trigger > 0) {
			const id = setTimeout(() => untrack(() => void fitView()), 150);
			return () => clearTimeout(id);
		}
	});
</script>
