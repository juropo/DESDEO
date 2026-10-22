<script lang="ts">
	/**
	 * Charts of a robustness analysis: max regret and the domain criterion as heatmaps by design and
	 * objective, the share of scenarios meeting all thresholds as bars, and with a baseline leaf the
	 * antifragility index on a diverging scale with Taleb classes, and upside against downside.
	 * Clicking a design selects it; the selected design's label is highlighted.
	 */
	import type { EChartsOption } from 'echarts';
	import EChartsComponent from '$lib/components/custom/nimbus/echarts-component.svelte';
	import type { ProblemInfo } from '$lib/gen/endpoints/DESDEOFastAPI';
	import {
		antifragilityHeatmap,
		domainShares,
		evaluatedLeafObjectives,
		evaluatedLeaves,
		regretHeatmap,
		upsideDownside,
		type HeatmapData,
		type Outcomes
	} from './robustness-charts';
	import { formatNumber, objectiveLabel } from './labels';
	import type { AnalysisResult } from './types';

	let {
		problem,
		result,
		outcomes,
		baselineLeaf,
		selectedId = $bindable(null)
	}: {
		problem: ProblemInfo;
		result: AnalysisResult;
		outcomes: Outcomes;
		baselineLeaf: string | null;
		selectedId?: number | null;
	} = $props();

	const SEQUENTIAL = ['#f8fafc', '#94a3b8', '#1e293b'];
	const DIVERGING = ['#b91c1c', '#f3f4f6', '#15803d'];
	const BAR = '#64748b';
	const BAR_SELECTED = '#1e293b';

	let maximize = $derived(
		Object.fromEntries(problem.objectives.map((o) => [o.symbol, !!o.maximize]))
	);
	let labels = $derived(result.objectives.map((o) => objectiveLabel(problem, o)));
	let regret = $derived(regretHeatmap(result.analysis.max_regret, result.objectives));
	let domain = $derived(
		domainShares(
			evaluatedLeafObjectives(outcomes),
			regret.rows,
			evaluatedLeaves(outcomes),
			result.thresholds,
			maximize
		)
	);
	let fragility = $derived(
		baselineLeaf && result.analysis.antifragility
			? antifragilityHeatmap(result.analysis.antifragility, result.objectives)
			: null
	);
	let points = $derived(
		baselineLeaf && result.analysis.antifragility
			? upsideDownside(result.analysis.antifragility, result.objectives)
			: []
	);

	/** Design labels on a category axis, the selected one bold. */
	function designAxis(rows: number[]) {
		return {
			type: 'category' as const,
			data: rows.map((id) => `#${id}`),
			inverse: true,
			axisLabel: {
				formatter: (value: string) => (value === `#${selectedId}` ? `{sel|${value}}` : value),
				rich: { sel: { fontWeight: 'bold' as const, color: '#1d4ed8' } }
			}
		};
	}

	type Cell = [number, number, number | string];

	/** The [column, row, value] of a heatmap event, whose data is the cell or an object holding it. */
	function cellOf(p: unknown): Cell {
		return (p as { value: Cell }).value;
	}

	function heatmapOption(spec: {
		data: HeatmapData;
		columns: string[];
		range: [number, number];
		rangeText: [string, string];
		colors: string[];
		label: (value: number | null, index: number) => string;
		cellStyle?: (index: number) => Record<string, unknown> | undefined;
		tooltipExtra?: (index: number) => string;
	}): EChartsOption {
		const { data, columns } = spec;
		return {
			tooltip: {
				trigger: 'item',
				formatter: (p: unknown) => {
					const [x, y, v] = cellOf(p);
					const extra = spec.tooltipExtra?.((p as { dataIndex: number }).dataIndex) ?? '';
					return `#${data.rows[y]} — ${columns[x]}: ${typeof v === 'number' ? formatNumber(v) : '—'}${extra}`;
				}
			},
			grid: { left: 60, right: 20, top: 10, bottom: 80 },
			xAxis: {
				type: 'category',
				data: columns,
				axisLabel: { interval: 0, width: 85, overflow: 'break', lineHeight: 12, fontSize: 11 }
			},
			yAxis: designAxis(data.rows),
			visualMap: {
				min: spec.range[0],
				max: spec.range[1],
				calculable: false,
				orient: 'horizontal',
				left: 'right',
				bottom: 0,
				itemHeight: 80,
				text: spec.rangeText,
				inRange: { color: spec.colors }
			},
			series: [
				{
					type: 'heatmap',
					data: data.cells.map(([x, y, v], index) => {
						const value: Cell = [x, y, v ?? '-'];
						const itemStyle = spec.cellStyle?.(index);
						return itemStyle ? { value, itemStyle } : value;
					}),
					label: {
						show: true,
						fontSize: 10,
						rich: { fragile: { fontWeight: 'bold', color: '#7f1d1d', fontSize: 10 } },
						formatter: (p: unknown) => {
							const v = cellOf(p)[2];
							return spec.label(
								typeof v === 'number' ? v : null,
								(p as { dataIndex: number }).dataIndex
							);
						}
					}
				}
			]
		};
	}

	// Regret with an extra column of each design's worst regret.
	let regretOption = $derived(
		heatmapOption({
			data: {
				...regret,
				cells: [
					...regret.cells,
					...regret.worst.map((w, y): [number, number, number | null] => [labels.length, y, w])
				]
			},
			columns: [...labels, 'Worst'],
			range: [0, 1],
			rangeText: ['1 (worst)', '0 (best)'],
			colors: SEQUENTIAL,
			label: (v) => (v === null ? '—' : formatNumber(v, 2))
		})
	);

	let domainOption = $derived(
		heatmapOption({
			data: domain.perObjective,
			columns: Object.keys(result.thresholds).map((o) => objectiveLabel(problem, o)),
			range: [0, 1],
			rangeText: ['100%', '0%'],
			colors: SEQUENTIAL,
			label: (v) => (v === null ? '—' : `${Math.round(v * 100)}%`)
		})
	);

	let allOption = $derived<EChartsOption>({
		tooltip: { trigger: 'axis', valueFormatter: (v) => `${Math.round(Number(v) * 100)}%` },
		grid: { left: 60, right: 50, top: 10, bottom: 30 },
		xAxis: {
			type: 'value',
			min: 0,
			max: 1,
			axisLabel: { formatter: (v: number) => `${v * 100}%` }
		},
		yAxis: designAxis(regret.rows),
		series: [
			{
				type: 'bar',
				data: domain.all.map((v, i) => ({
					value: v ?? 0,
					itemStyle: { color: regret.rows[i] === selectedId ? BAR_SELECTED : BAR }
				})),
				label: {
					show: true,
					position: 'right',
					formatter: (p: unknown) => `${Math.round((p as { value: number }).value * 100)}%`
				}
			}
		]
	});

	// The cell colour is the antifragility index; with floors the label is the Taleb class, and
	// fragile cells are outlined.
	let hasClasses = $derived(!!fragility?.classes.some((c) => c !== null));
	let fragilityOption = $derived(
		fragility
			? heatmapOption({
					data: fragility,
					columns: labels,
					range: [-1, 1],
					rangeText: ['+1 (gains)', '−1 (losses)'],
					colors: DIVERGING,
					label: (v, i) => {
						const cls = fragility.classes[i];
						if (hasClasses) return cls === 'fragile' ? '{fragile|fragile}' : (cls ?? '—');
						return v === null ? '—' : formatNumber(v, 2);
					},
					cellStyle: (i) =>
						fragility.classes[i] === 'fragile'
							? { borderColor: '#7f1d1d', borderWidth: 2 }
							: undefined,
					tooltipExtra: (i) => (fragility.classes[i] ? ` (${fragility.classes[i]})` : '')
				})
			: null
	);

	// One upside-against-downside scatter per objective, each on its own scale.
	let scatterOptions = $derived(
		result.objectives.map((obj, k): EChartsOption => {
			const own = points.filter((p) => p.objective === obj);
			const top = Math.max(...own.flatMap((p) => [p.upside, p.downside]), 0) * 1.1 || 1;
			return {
				title: {
					text: labels[k],
					textStyle: { fontSize: 11, fontWeight: 'normal' },
					left: 'center'
				},
				tooltip: {
					trigger: 'item',
					alwaysShowContent: false,
					enterable: false,
					hideDelay: 0,
					formatter: (p: unknown) => {
						const { data } = p as { data: { designId: number; value: [number, number] } };
						return `#${data.designId} — ${labels[k]}<br>downside ${formatNumber(data.value[0], 3)}, upside ${formatNumber(data.value[1], 3)}`;
					}
				},
				grid: { left: 45, right: 15, top: 30, bottom: 35 },
				xAxis: {
					type: 'value',
					name: 'Downside',
					nameLocation: 'middle',
					nameGap: 22,
					min: 0,
					max: top,
					splitNumber: 3,
					axisLabel: { formatter: (v: number) => formatNumber(v, 2), hideOverlap: true }
				},
				yAxis: {
					type: 'value',
					name: 'Upside',
					min: 0,
					max: top,
					splitNumber: 3,
					axisLabel: { formatter: (v: number) => formatNumber(v, 2), hideOverlap: true }
				},
				series: [
					{
						type: 'scatter',
						colorBy: 'series',
						data: own.map((p) => ({
							value: [p.downside, p.upside],
							designId: p.designId,
							symbolSize: p.designId === selectedId ? 14 : 9,
							itemStyle: { color: p.designId === selectedId ? BAR_SELECTED : BAR }
						})),
						label: {
							show: true,
							position: 'right',
							fontSize: 10,
							formatter: (p: unknown) => `#${(p as { data: { designId: number } }).data.designId}`
						},
						// Coinciding points would stack their labels; the tooltip still names each design.
						labelLayout: { hideOverlap: true },
						markLine: {
							silent: true,
							symbol: 'none',
							lineStyle: { type: 'dashed', color: '#9ca3af' },
							data: [[{ coord: [0, 0] }, { coord: [top, top] }]]
						}
					}
				]
			};
		})
	);

	// Clicks select the design of the clicked cell, bar or point.
	function rowClick(rows: () => number[]) {
		return {
			click: (p: unknown) => {
				selectedId = rows()[cellOf(p)[1]] ?? selectedId;
			}
		};
	}
	const regretEvents = rowClick(() => regret.rows);
	const domainEvents = rowClick(() => domain.perObjective.rows);
	const fragilityEvents = rowClick(() => fragility?.rows ?? []);
	const barEvents = {
		click: (p: { dataIndex: number }) => {
			selectedId = regret.rows[p.dataIndex] ?? selectedId;
		}
	};
	const scatterEvents = {
		click: (p: unknown) => {
			const id = (p as { data: { designId?: number } }).data.designId;
			if (typeof id === 'number') selectedId = id;
		}
	};

	// Heights grow with the designs; widths with the objectives, scrolling inside the card.
	let rowsHeight = $derived(Math.max(160, regret.rows.length * 28 + 90));
	let columnsWidth = $derived(`min-width: ${120 + (labels.length + 1) * 90}px`);
</script>

<div class="space-y-4">
	<section class="rounded border bg-white p-3">
		<h3 class="text-sm font-semibold">Max regret</h3>
		<p class="text-xs text-gray-500">
			Gap to the best design in each scenario, scaled by the spread over designs (0 = best in every
			scenario), and each design's worst.
		</p>
		<div class="overflow-x-auto">
			<div style="height: {rowsHeight}px; {columnsWidth}">
				<EChartsComponent option={regretOption} events={regretEvents} />
			</div>
		</div>
	</section>

	<section class="rounded border bg-white p-3">
		<h3 class="text-sm font-semibold">Domain criterion</h3>
		{#if Object.keys(result.thresholds).length === 0}
			<p class="text-xs text-gray-500">No thresholds were given.</p>
		{:else}
			<p class="text-xs text-gray-500">
				Share of the {result.nLeaves} scenarios meeting all thresholds, and each threshold.
			</p>
			<div class="grid gap-3 @3xl:grid-cols-2">
				<div style="height: {rowsHeight}px">
					<EChartsComponent option={allOption} events={barEvents} />
				</div>
				<div class="overflow-x-auto">
					<div style="height: {rowsHeight}px; {columnsWidth}">
						<EChartsComponent option={domainOption} events={domainEvents} />
					</div>
				</div>
			</div>
		{/if}
	</section>

	{#if baselineLeaf}
		<section class="rounded border bg-white p-3">
			<h3 class="text-sm font-semibold">Antifragility relative to {baselineLeaf}</h3>
			{#if fragilityOption}
				<p class="text-xs text-gray-500">
					Colour = antifragility index, from −1 (only losses, red) to +1 (only gains, green).
					{hasClasses
						? 'Label = Taleb class from the floors; fragile cells are outlined.'
						: 'Give floors to see Taleb classes.'}
					Below: upside against downside per objective, each on its own scale; above the dashed line gains
					outweigh losses.
				</p>
				<div class="space-y-3">
					<div class="overflow-x-auto">
						<div style="height: {rowsHeight}px; {columnsWidth}">
							<EChartsComponent option={fragilityOption} events={fragilityEvents} />
						</div>
					</div>
					<div class="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-2">
						{#each scatterOptions as option, k (result.objectives[k])}
							<div style="height: 220px">
								<EChartsComponent {option} events={scatterEvents} />
							</div>
						{/each}
					</div>
				</div>
			{:else}
				<p class="text-xs text-gray-500">No design has an outcome in {baselineLeaf}.</p>
			{/if}
		</section>
	{/if}
</div>
