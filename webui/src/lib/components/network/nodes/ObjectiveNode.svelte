<script lang="ts">
	import type { NodeProps } from '@xyflow/svelte';
	import IconTarget from '@lucide/svelte/icons/target';
	import BaseNode from './BaseNode.svelte';
	import { NODE_KIND_STYLE } from '../node-style';
	import type { NetworkNodeData } from '../view-types';

	// Objective nodes show no distributions/values here — their job in the graph is purely
	// structural, per the plan (phase C). `NetworkNodeModel.states` is absent for objectives
	// (they carry a scalar utility, not a state space), so BaseNode simply has nothing state-ish
	// to render for this kind regardless of detail level.
	let { data, selected = false }: NodeProps = $props();
	const style = NODE_KIND_STYLE.objective;
</script>

<BaseNode
	data={data as NetworkNodeData}
	{selected}
	shape={style.shape}
	colorVar={style.colorVar}
	icon={IconTarget}
/>
