import { describe, it, expect, vi, beforeEach } from 'vitest';
import { get } from 'svelte/store';

const setObjectiveConstraints = vi.fn<(...args: unknown[]) => unknown>();
const getObjectiveConstraints = vi.fn<(...args: unknown[]) => unknown>();
const getScenarioAvailability = vi.fn<(...args: unknown[]) => unknown>();
const applyScenarioMeasures = vi.fn<(...args: unknown[]) => unknown>();

vi.mock('$lib/gen/endpoints/DESDEOFastAPI', () => ({
	setObjectiveConstraintsMethodCumulusObjectiveConstraintPost: (...args: unknown[]) =>
		setObjectiveConstraints(...args),
	getObjectiveConstraintsMethodCumulusObjectiveConstraintGet: (...args: unknown[]) =>
		getObjectiveConstraints(...args),
	getScenarioAvailabilityMethodCumulusScenarioAvailabilityGet: (...args: unknown[]) =>
		getScenarioAvailability(...args),
	applyScenarioMeasuresMethodCumulusApplyScenarioMeasuresPost: (...args: unknown[]) =>
		applyScenarioMeasures(...args),
	// The module is imported wholesale, so every other endpoint the handlers pull in has to exist.
	solveSolutionsMethodCumulusSolvePost: vi.fn(),
	getOrInitializeMethodCumulusGetOrInitializePost: vi.fn(),
	saveMethodCumulusSavePost: vi.fn(),
	deleteSaveMethodCumulusDeleteSavePost: vi.fn(),
	finalizeMethodCumulusFinalizePost: vi.fn(),
	solveIntermediateMethodCumulusIntermediatePost: vi.fn(),
	getUtopiaDataUtopiaPost: vi.fn(),
	getSolutionDescriptionSolutionDescriptionGetPost: vi.fn()
}));

const {
	set_objective_constraints,
	get_objective_constraints,
	check_scenario_availability,
	apply_scenario_measures,
	apply_network_bounds
} = await import('./handlers');
const { errorMessage } = await import('../../../stores/uiState');

import type { ProblemInfo } from '$lib/gen/endpoints/DESDEOFastAPI';

/** A problem stub carrying only what the constraint handlers actually read. */
function make_problem(): ProblemInfo {
	return { id: 7 } as unknown as ProblemInfo;
}

const SOFT_OK = {
	status: 200,
	data: { state_id: 99, soft_constraint_ids: [11] }
};

beforeEach(() => {
	vi.clearAllMocks();
	errorMessage.set(null);
	setObjectiveConstraints.mockResolvedValue(SOFT_OK);
});

describe('set_objective_constraints', () => {
	it('sends the constraints as the soft list of the current problem', async () => {
		const result = await set_objective_constraints(make_problem(), 5, [
			{ func: 'f_1 - 100', type: '<=0' }
		]);

		expect(setObjectiveConstraints).toHaveBeenCalledWith({
			problem_id: 7,
			parent_state_id: 5,
			soft_constraints: [{ name: 'f_1 - 100', symbol: 'oc_1', func: 'f_1 - 100', cons_type: '<=' }]
		});
		expect(result).toMatchObject({ state_id: 99 });
	});

	it('still posts an empty soft list, so removing every constraint clears them', async () => {
		await set_objective_constraints(make_problem(), null, []);

		expect(setObjectiveConstraints).toHaveBeenCalledWith(
			expect.objectContaining({ soft_constraints: [] })
		);
	});

	it('reports the endpoint error and returns null when the list is refused', async () => {
		setObjectiveConstraints.mockResolvedValue({ status: 422, data: { detail: 'bad expression' } });

		const result = await set_objective_constraints(make_problem(), 5, [
			{ func: 'nonsense(', type: '<=0' }
		]);

		expect(result).toBeNull();
		expect(get(errorMessage)).toContain('bad expression');
	});
});

describe('get_objective_constraints', () => {
	it('turns the stored constraints back into what the dialog edits', async () => {
		getObjectiveConstraints.mockResolvedValue({
			status: 200,
			data: {
				soft_constraints: [
					{ name: 'f_1 - 100', symbol: 'oc_1', cons_type: '<=', func: ['Subtract', 'f_1', 100] },
					{ name: 'f_2 - 5', symbol: 'oc_2', cons_type: '=', func: ['Subtract', 'f_2', 5] }
				]
			}
		});

		const constraints = await get_objective_constraints(7);

		expect(getObjectiveConstraints).toHaveBeenCalledWith({ problem_id: 7 });
		expect(constraints).toEqual([
			{ func: 'f_1 - 100', type: '<=0' },
			{ func: 'f_2 - 5', type: '=0' }
		]);
	});

	it('returns null when the endpoint fails', async () => {
		getObjectiveConstraints.mockResolvedValue({ status: 500, data: { detail: 'boom' } });

		expect(await get_objective_constraints(7)).toBeNull();
	});
});

describe('check_scenario_availability', () => {
	it('reports availability straight from the endpoint response', async () => {
		getScenarioAvailability.mockResolvedValue({
			status: 200,
			data: {
				available: true,
				base_problem_id: 7,
				scenario_model_id: 3,
				objective_symbols: ['f_1']
			}
		});

		const result = await check_scenario_availability(7);

		expect(getScenarioAvailability).toHaveBeenCalledWith({ problem_id: 7 });
		expect(result).toEqual({
			available: true,
			base_problem_id: 7,
			scenario_model_id: 3,
			objective_symbols: ['f_1']
		});
	});

	it('returns null when the endpoint fails', async () => {
		getScenarioAvailability.mockResolvedValue({ status: 500, data: { detail: 'boom' } });

		const result = await check_scenario_availability(7);

		expect(result).toBeNull();
	});
});

describe('apply_scenario_measures', () => {
	it('sends the chosen measures and returns the rebuilt problem state', async () => {
		applyScenarioMeasures.mockResolvedValue({ status: 200, data: { problem_id: 9 } });

		const result = await apply_scenario_measures(
			7,
			3,
			['f_1', 'f_2'],
			['expected_value'],
			{},
			'combined'
		);

		expect(applyScenarioMeasures).toHaveBeenCalledWith({
			problem_id: 7,
			name: 'combined',
			uncertainty_measures: [
				{ measure_type: 'expected_value', scenario_model_id: 3, symbols: ['f_1', 'f_2'] }
			]
		});
		expect(result).toEqual({ problem_id: 9 });
	});

	it('reports an error and returns null when the endpoint rejects the measures', async () => {
		applyScenarioMeasures.mockResolvedValue({ status: 422, data: { detail: 'no scenario model' } });

		const result = await apply_scenario_measures(7, 3, ['f_1'], ['expected_value']);

		expect(result).toBeNull();
		expect(get(errorMessage)).toContain('422');
	});
});

describe('apply_network_bounds', () => {
	/** A problem stub carrying only what this pure fold actually reads/writes. */
	function make_bounded_problem(): ProblemInfo {
		return {
			id: 7,
			objectives: [
				{ symbol: 'f_1', ideal: 0, nadir: 10 },
				{ symbol: 'f_2', ideal: -5, nadir: 5 }
			]
		} as unknown as ProblemInfo;
	}

	it('widens ideal/nadir for an objective the bounds mention', () => {
		const result = apply_network_bounds(make_bounded_problem(), {
			ideal: { f_1: -2 },
			nadir: { f_1: 12 }
		});

		expect(result.objectives.find((o) => o.symbol === 'f_1')).toMatchObject({
			ideal: -2,
			nadir: 12
		});
	});

	it('leaves an objective the bounds do not mention untouched', () => {
		const original = make_bounded_problem();
		const result = apply_network_bounds(original, { ideal: { f_1: -2 } });

		expect(result.objectives.find((o) => o.symbol === 'f_2')).toEqual(
			original.objectives.find((o) => o.symbol === 'f_2')
		);
	});

	it('only overwrites the direction the bounds actually mention for a touched objective', () => {
		const result = apply_network_bounds(make_bounded_problem(), { ideal: { f_1: -2 } });

		expect(result.objectives.find((o) => o.symbol === 'f_1')).toMatchObject({
			ideal: -2,
			nadir: 10 // unchanged -- bounds carried no `nadir.f_1` entry
		});
	});

	it('does not mutate the input problem', () => {
		const original = make_bounded_problem();
		const snapshot = structuredClone(original);
		apply_network_bounds(original, { ideal: { f_1: -2 } });

		expect(original).toEqual(snapshot);
	});

	it('returns the identical problem reference when bounds carry nothing new', () => {
		const original = make_bounded_problem();
		const result = apply_network_bounds(original, {
			ideal: { f_1: 0, f_2: -5 },
			nadir: { f_1: 10, f_2: 5 }
		});

		expect(result).toBe(original);
	});

	it('returns the identical problem reference for an empty bounds object', () => {
		const original = make_bounded_problem();
		expect(apply_network_bounds(original, {})).toBe(original);
	});

	it('returns a new problem reference, but reuses untouched objective references, when something actually moves', () => {
		const original = make_bounded_problem();
		const result = apply_network_bounds(original, { ideal: { f_1: -2 } });

		expect(result).not.toBe(original);
		expect(result.objectives.find((o) => o.symbol === 'f_2')).toBe(
			original.objectives.find((o) => o.symbol === 'f_2')
		);
	});
});
