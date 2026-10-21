<script lang="ts">
	/**
	 * Edits a reference point with one aspiration level per target.
	 *
	 * `value` holds the raw aspiration levels in the objectives' original direction. They can be
	 * entered as raw values, as percent of the ideal–nadir range (100 = ideal, 0 = nadir) or as a
	 * change from the previous reference point in percent of the range toward the ideal.
	 */
	import { SegmentedControl } from '$lib/components/custom/segmented-control';
	import type { ProblemInfo, Target } from '$lib/gen/endpoints/DESDEOFastAPI';
	import {
		changeToRaw,
		percentToRaw,
		rawToChange,
		rawToPercent,
		type Range
	} from './reference-point';
	import { formatNumber, targetLabel } from './labels';

	let {
		problem,
		targets,
		ranges,
		previous = null,
		value = $bindable({})
	}: {
		problem: ProblemInfo;
		targets: Target[];
		ranges: Record<string, Range>;
		previous?: Record<string, number> | null;
		value?: Record<string, number>;
	} = $props();

	type Mode = 'percent' | 'raw' | 'change';
	let mode = $state<Mode>('percent');

	let modes = $derived([
		{ value: 'percent', label: 'Percent' },
		{ value: 'raw', label: 'Value' },
		...(previous ? [{ value: 'change', label: 'Change' }] : [])
	]);

	$effect(() => {
		if (mode === 'change' && !previous) mode = 'percent';
	});

	function shown(target: Target): number | null {
		const raw = value[target.symbol];
		const range = ranges[target.symbol];
		if (raw === undefined || !range) return null;
		if (mode === 'percent') return Number(rawToPercent(raw, range).toFixed(1));
		if (mode === 'change') {
			const prev = previous?.[target.symbol];
			return prev === undefined ? null : Number(rawToChange(raw, prev, range).toFixed(1));
		}
		return Number(raw.toPrecision(6));
	}

	/** Apply an entered value; a rejected one resets the input to the value that will be sent. */
	function update(target: Target, input: HTMLInputElement) {
		const raw = toRaw(target, input.value);
		if (raw === null) {
			const current = shown(target);
			input.value = current === null ? '' : String(current);
			return;
		}
		value = { ...value, [target.symbol]: raw };
	}

	function toRaw(target: Target, input: string): number | null {
		const entered = Number(input);
		const range = ranges[target.symbol];
		if (input === '' || !Number.isFinite(entered) || !range) return null;
		if (mode === 'percent') return percentToRaw(entered, range);
		if (mode === 'change') {
			const prev = previous?.[target.symbol];
			return prev === undefined ? null : changeToRaw(entered, prev, range);
		}
		return entered;
	}
</script>

<div class="space-y-3">
	<SegmentedControl bind:value={mode} options={modes} size="sm" />
	<p class="text-xs text-gray-500">
		{#if mode === 'percent'}
			100 is the ideal and 0 the nadir of each target.
		{:else if mode === 'change'}
			Percent of each range to move from the previous reference point; positive moves toward the
			ideal.
		{:else}
			Aspiration levels in the objectives' own units.
		{/if}
	</p>
	{#each targets as target (target.symbol)}
		{@const range = ranges[target.symbol]}
		<div class="grid gap-1">
			<label for="rp-{target.symbol}" class="text-sm font-medium">
				{targetLabel(problem, target)}
				<span class="text-gray-500">({target.maximize ? 'max' : 'min'})</span>
			</label>
			<div class="flex items-center gap-2">
				<input
					id="rp-{target.symbol}"
					type="number"
					step="any"
					class="w-28 rounded border px-2 py-1 text-sm"
					value={shown(target)}
					onchange={(e) => update(target, e.currentTarget)}
				/>
				{#if mode !== 'raw'}
					<span class="text-xs text-gray-600">= {formatNumber(value[target.symbol])}</span>
				{/if}
			</div>
			{#if range}
				<span class="text-xs text-gray-500">
					ideal {formatNumber(range[0])} · nadir {formatNumber(range[1])}
					{#if previous?.[target.symbol] !== undefined}
						· previous {formatNumber(previous[target.symbol])}
					{/if}
				</span>
			{/if}
		</div>
	{/each}
</div>
