<script lang="ts">
	import type { NodeProps } from '@xyflow/svelte';
	import IconSigma from '@lucide/svelte/icons/sigma';
	import BaseNode from './BaseNode.svelte';
	import { NODE_KIND_STYLE } from '../node-style';
	import type { NetworkNodeData } from '../view-types';

	let { data, selected = false }: NodeProps = $props();
	const style = NODE_KIND_STYLE.deterministic;
</script>

<BaseNode
	data={data as NetworkNodeData}
	{selected}
	shape={style.shape}
	colorVar={style.colorVar}
	icon={IconSigma}
/>
