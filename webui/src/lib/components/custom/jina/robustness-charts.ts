/* Data for the robustness charts: evaluated outcomes, max regret, domain criterion, antifragility. */

import type { Antifragility, LeafOutcome, MaxRegret } from '$lib/gen/endpoints/DESDEOFastAPI';

/** Outcomes of an evaluation by design id (as a string) and leaf. */
export type Outcomes = Record<string, Record<string, LeafOutcome>>;

/** Rows of designs by columns of objectives; each cell is [column, row, value or null]. */
export interface HeatmapData {
	rows: number[];
	columns: string[];
	cells: [number, number, number | null][];
}

/** The design ids keying a record, in ascending order. */
export function evaluatedDesigns(byDesign: Record<string, unknown>): number[] {
	return Object.keys(byDesign)
		.map(Number)
		.sort((a, b) => a - b);
}

/** The leaves of the outcomes in order of first appearance. */
export function evaluatedLeaves(outcomes: Outcomes): string[] {
	return [...new Set(Object.values(outcomes).flatMap((leaves) => Object.keys(leaves)))];
}

/** The objectives with a value in the outcomes, in order of first appearance. */
export function evaluatedObjectives(outcomes: Outcomes): string[] {
	return [
		...new Set(
			Object.values(outcomes).flatMap((leaves) =>
				Object.values(leaves).flatMap((outcome) => Object.keys(outcome.objectives ?? {}))
			)
		)
	];
}

/** Objective values by design, leaf and objective, from the leaves where the evaluation succeeded. */
export function evaluatedLeafObjectives(
	outcomes: Outcomes
): Record<number, Record<string, Record<string, number>>> {
	return Object.fromEntries(
		Object.entries(outcomes).map(([id, leaves]) => [
			Number(id),
			Object.fromEntries(
				Object.entries(leaves).flatMap(([leaf, outcome]) =>
					outcome.status ? [[leaf, outcome.objectives ?? {}]] : []
				)
			)
		])
	);
}

function heatmap(
	rows: number[],
	columns: string[],
	value: (row: number, column: string) => number | null | undefined
): HeatmapData {
	return {
		rows,
		columns,
		cells: rows.flatMap((row, y) =>
			columns.map((column, x): [number, number, number | null] => [
				x,
				y,
				value(row, column) ?? null
			])
		)
	};
}

/** Max regret by design and objective, with each design's worst regret over the objectives. */
export function regretHeatmap(
	maxRegret: Record<string, MaxRegret>,
	objectives: string[]
): HeatmapData & { worst: (number | null)[] } {
	const rows = evaluatedDesigns(maxRegret);
	const data = heatmap(rows, objectives, (id, obj) => maxRegret[String(id)]?.regret[obj]);
	const worst = rows.map((id) => {
		const values = objectives
			.map((obj) => maxRegret[String(id)]?.regret[obj])
			.filter((v): v is number => typeof v === 'number');
		return values.length > 0 ? Math.max(...values) : null;
	});
	return { ...data, worst };
}

/** Whether a value meets a threshold: at most it for a minimised, at least it for a maximised objective. */
export function meetsThreshold(value: number, threshold: number, maximize: boolean): boolean {
	return maximize ? value >= threshold : value <= threshold;
}

/**
 * Shares of the scenarios where each design meets the thresholds: of all thresholds at once, and
 * of each objective's threshold. A scenario where the design has no value does not meet it.
 */
export function domainShares(
	leafObjectives: Record<number, Record<string, Record<string, number>>>,
	designs: number[],
	leaves: string[],
	thresholds: Record<string, number>,
	maximize: Record<string, boolean>
): { all: (number | null)[]; perObjective: HeatmapData } {
	const objectives = Object.keys(thresholds);
	const meets = (id: number, leaf: string, obj: string) => {
		const value = leafObjectives[id]?.[leaf]?.[obj];
		return typeof value === 'number' && meetsThreshold(value, thresholds[obj], !!maximize[obj]);
	};
	const share = (count: number) => (leaves.length > 0 ? count / leaves.length : null);
	return {
		all: designs.map((id) =>
			objectives.length === 0
				? null
				: share(leaves.filter((leaf) => objectives.every((obj) => meets(id, leaf, obj))).length)
		),
		perObjective: heatmap(designs, objectives, (id, obj) =>
			share(leaves.filter((leaf) => meets(id, leaf, obj)).length)
		)
	};
}

/** The antifragility index by design and objective, with the Taleb class of each cell. */
export function antifragilityHeatmap(
	antifragility: Record<string, Antifragility>,
	objectives: string[]
): HeatmapData & { classes: (string | null)[] } {
	const rows = evaluatedDesigns(antifragility);
	const data = heatmap(
		rows,
		objectives,
		(id, obj) => antifragility[String(id)]?.objectives[obj]?.antifragility
	);
	const classes = data.cells.map(
		([x, y]) => antifragility[String(rows[y])]?.objectives[objectives[x]]?.taleb_class ?? null
	);
	return { ...data, classes };
}

/** The upside and downside of each design and objective that has both. */
export function upsideDownside(
	antifragility: Record<string, Antifragility>,
	objectives: string[]
): { designId: number; objective: string; upside: number; downside: number }[] {
	return evaluatedDesigns(antifragility).flatMap((id) =>
		objectives.flatMap((objective) => {
			const summary = antifragility[String(id)]?.objectives[objective];
			return summary?.upside != null && summary.downside !== null
				? [{ designId: id, objective, upside: summary.upside, downside: summary.downside }]
				: [];
		})
	);
}
