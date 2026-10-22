import { describe, it, expect } from 'vitest';
import type { Antifragility, MaxRegret } from '$lib/gen/endpoints/DESDEOFastAPI';
import {
	antifragilityHeatmap,
	domainShares,
	evaluatedDesigns,
	evaluatedLeafObjectives,
	evaluatedLeaves,
	evaluatedObjectives,
	meetsThreshold,
	regretHeatmap,
	upsideDownside,
	type Outcomes
} from './robustness-charts';

const outcomes: Outcomes = {
	'10': {
		a: { status: true, message: '', objectives: { f: 1, g: 9 } },
		b: { status: true, message: '', objectives: { f: 4, g: 5 } }
	},
	'2': {
		a: { status: true, message: '', objectives: { f: 2, g: 7 } },
		b: { status: false, message: 'infeasible' }
	}
};

describe('evaluated outcomes', () => {
	it('list designs ascending, leaves and objectives in order of appearance', () => {
		expect(evaluatedDesigns(outcomes)).toEqual([2, 10]);
		expect(evaluatedLeaves(outcomes)).toEqual(['a', 'b']);
		expect(evaluatedObjectives(outcomes)).toEqual(['f', 'g']);
	});

	it('keep only the leaves where the evaluation succeeded', () => {
		expect(evaluatedLeafObjectives(outcomes)).toEqual({
			10: { a: { f: 1, g: 9 }, b: { f: 4, g: 5 } },
			2: { a: { f: 2, g: 7 } }
		});
	});
});

describe('regretHeatmap', () => {
	it('gives a cell per design and objective and each design worst regret', () => {
		const regret: Record<string, MaxRegret> = {
			'10': { regret: { f: 0.5, g: null }, mean_regret: 0.5 },
			'2': { regret: { f: 0, g: 1 }, mean_regret: 0.5 }
		};
		expect(regretHeatmap(regret, ['f', 'g'])).toEqual({
			rows: [2, 10],
			columns: ['f', 'g'],
			cells: [
				[0, 0, 0],
				[1, 0, 1],
				[0, 1, 0.5],
				[1, 1, null]
			],
			worst: [1, 0.5]
		});
	});
});

describe('domainShares', () => {
	const leafObjectives = evaluatedLeafObjectives(outcomes);

	it('counts scenarios meeting all and each threshold, direction-aware, missing as not met', () => {
		// f minimised at most 3, g maximised at least 6.
		const shares = domainShares(
			leafObjectives,
			[2, 10],
			['a', 'b'],
			{ f: 3, g: 6 },
			{ f: false, g: true }
		);
		expect(shares.all).toEqual([0.5, 0.5]);
		expect(shares.perObjective.cells).toEqual([
			[0, 0, 0.5],
			[1, 0, 0.5],
			[0, 1, 0.5],
			[1, 1, 0.5]
		]);
	});

	it('gives no overall share without thresholds', () => {
		expect(domainShares(leafObjectives, [2], ['a'], {}, {}).all).toEqual([null]);
	});

	it('compares in the objective direction', () => {
		expect(meetsThreshold(5, 5, false)).toBe(true);
		expect(meetsThreshold(6, 5, false)).toBe(false);
		expect(meetsThreshold(6, 5, true)).toBe(true);
	});
});

const summary = (antifragility: number | null, up: number | null, down: number | null) => ({
	upside: up,
	downside: down,
	antifragility,
	n_gain: 0,
	n_loss: 0,
	upside_raw: null,
	downside_raw: null,
	max_loss_raw: null,
	best_gain_leaf: null,
	worst_loss_leaf: null,
	worst_outcome: null,
	taleb_class: antifragility !== null && antifragility > 0 ? ('antifragile' as const) : null,
	deviations: []
});

describe('antifragility charts', () => {
	const af: Record<string, Antifragility> = {
		'3': { objectives: { f: summary(0.5, 0.6, 0.2), g: summary(null, null, null) } },
		'1': { objectives: { f: summary(-1, 0, 0.7), g: summary(0, 0.1, 0.1) } }
	};

	it('give the index and class per design and objective', () => {
		const data = antifragilityHeatmap(af, ['f', 'g']);
		expect(data.rows).toEqual([1, 3]);
		expect(data.cells).toEqual([
			[0, 0, -1],
			[1, 0, 0],
			[0, 1, 0.5],
			[1, 1, null]
		]);
		expect(data.classes).toEqual([null, null, 'antifragile', null]);
	});

	it('give upside and downside where both exist', () => {
		expect(upsideDownside(af, ['f', 'g'])).toEqual([
			{ designId: 1, objective: 'f', upside: 0, downside: 0.7 },
			{ designId: 1, objective: 'g', upside: 0.1, downside: 0.1 },
			{ designId: 3, objective: 'f', upside: 0.6, downside: 0.2 }
		]);
	});
});
