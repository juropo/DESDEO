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
		applyEntry,
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

	// Text being typed per target, shown instead of the converted value until the input is left,
	// and the value each input had when it was focused.
	let drafts = $state<Record<string, string>>({});
	let warnings = $state<Record<string, string>>({});
	let atFocus: Record<string, number | undefined> = {};

	$effect(() => {
		void mode;
		drafts = {};
		warnings = {};
	});

	function without<T>(record: Record<string, T>, key: string): Record<string, T> {
		const { [key]: _, ...rest } = record;
		return rest;
	}

	function focused(target: Target) {
		atFocus[target.symbol] = value[target.symbol];
	}

	/** Apply each keystroke at once, so a typed value is never lost; an invalid entry restores the value at focus. */
	function typed(target: Target, input: string) {
		drafts = { ...drafts, [target.symbol]: input };
		const range = ranges[target.symbol];
		const initial = atFocus[target.symbol] ?? value[target.symbol];
		const prev = previous?.[target.symbol];
		if (!range || initial === undefined || (mode === 'change' && prev === undefined)) return;
		const convert =
			mode === 'percent'
				? (v: number) => percentToRaw(v, range)
				: mode === 'change'
					? (v: number) => changeToRaw(v, prev!, range)
					: (v: number) => v;
		const result = applyEntry(input, initial, convert, mode === 'percent' ? [0, 100] : undefined);
		warnings = result.warning
			? { ...warnings, [target.symbol]: result.warning }
			: without(warnings, target.symbol);
		value = { ...value, [target.symbol]: result.value };
	}

	/** On leaving the input, show the value that will be sent. */
	function left(target: Target) {
		drafts = without(drafts, target.symbol);
		warnings = without(warnings, target.symbol);
		atFocus[target.symbol] = undefined;
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
		<div class="grid grid-cols-[minmax(0,1fr)] gap-1">
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
					min={mode === 'percent' ? 0 : undefined}
					max={mode === 'percent' ? 100 : undefined}
					value={drafts[target.symbol] ?? shown(target)}
					onfocus={() => focused(target)}
					oninput={(e) => typed(target, e.currentTarget.value)}
					onblur={() => left(target)}
				/>
				{#if mode !== 'raw'}
					<span class="text-xs text-gray-600">= {formatNumber(value[target.symbol])}</span>
				{/if}
			</div>
			<!-- The warning takes the place of the range line, so nothing below moves when it appears. -->
			{#if warnings[target.symbol]}
				<span class="truncate text-xs text-amber-700" role="alert" title={warnings[target.symbol]}>
					{warnings[target.symbol]}
				</span>
			{:else if range}
				{@const hint = `ideal ${formatNumber(range[0])} · nadir ${formatNumber(range[1])}${
					previous?.[target.symbol] !== undefined
						? ` · previous ${formatNumber(previous[target.symbol])}`
						: ''
				}`}
				<span class="truncate text-xs text-gray-500" title={hint}>{hint}</span>
			{/if}
		</div>
	{/each}
</div>
