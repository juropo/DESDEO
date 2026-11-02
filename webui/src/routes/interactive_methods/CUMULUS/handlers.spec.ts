import { describe, it, expect, vi, beforeEach } from 'vitest';
import { get } from 'svelte/store';

const setObjectiveConstraints = vi.fn<(...args: unknown[]) => unknown>();
const getObjectiveConstraints = vi.fn<(...args: unknown[]) => unknown>();
const getScenarioAvailability = vi.fn<(...args: unknown[]) => unknown>();
const applyScenarioMeasures = vi.fn<(...args: unknown[]) => unknown>();
const createHypothetical = vi.fn<(...args: unknown[]) => unknown>();
const modifyProblemStatus = vi.fn<(...args: unknown[]) => unknown>();

vi.mock('$lib/gen/endpoints/DESDEOFastAPI', () => ({
	setObjectiveConstraintsMethodCumulusObjectiveConstraintPost: (...args: unknown[]) =>
		setObjectiveConstraints(...args),
	getObjectiveConstraintsMethodCumulusObjectiveConstraintGet: (...args: unknown[]) =>
		getObjectiveConstraints(...args),
	getScenarioAvailabilityMethodCumulusScenarioAvailabilityGet: (...args: unknown[]) =>
		getScenarioAvailability(...args),
	applyScenarioMeasuresMethodCumulusApplyScenarioMeasuresPost: (...args: unknown[]) =>
		applyScenarioMeasures(...args),
	createHypotheticalMethodCumulusHypotheticalPost: (...args: unknown[]) =>
		createHypothetical(...args),
	modifyProblemStatusMethodCumulusModifyProblemStatusStateIdGet: (...args: unknown[]) =>
		modifyProblemStatus(...args),
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
	create_hypothetical,
	default_solution_name
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

	it('sends CVaR with the worst 5 % as the default tail probability', async () => {
		applyScenarioMeasures.mockResolvedValue({ status: 200, data: { problem_id: 9 } });

		await apply_scenario_measures(7, 3, ['f_1'], ['conditional_value_at_risk']);

		expect(applyScenarioMeasures).toHaveBeenCalledWith({
			problem_id: 7,
			name: null,
			uncertainty_measures: [
				{
					measure_type: 'conditional_value_at_risk',
					scenario_model_id: 3,
					symbols: ['f_1'],
					alpha: 0.05
				}
			]
		});
	});

	it('reports an error and returns null when the endpoint rejects the measures', async () => {
		applyScenarioMeasures.mockResolvedValue({ status: 422, data: { detail: 'no scenario model' } });

		const result = await apply_scenario_measures(7, 3, ['f_1'], ['expected_value']);

		expect(result).toBeNull();
		expect(get(errorMessage)).toContain('422');
	});
});

describe('create_hypothetical', () => {
	const assumptions = { chance: { structure: 'soaking' }, decisions: {} };

	beforeEach(() => {
		createHypothetical.mockReset();
		modifyProblemStatus.mockReset();
	});

	it('creates the problem, polls until ready, and returns its id', async () => {
		createHypothetical.mockResolvedValue({ status: 200, data: { state_id: 3, problem_id: 12 } });
		modifyProblemStatus
			.mockResolvedValueOnce({ status: 200, data: { is_ready: false } })
			.mockResolvedValueOnce({ status: 200, data: { is_ready: true, problem_id: 12 } });

		const result = await create_hypothetical(7, 2, assumptions, { poll_interval_ms: 0 });

		expect(createHypothetical).toHaveBeenCalledWith({
			problem_id: 7,
			parent_state_id: 2,
			assumptions
		});
		expect(modifyProblemStatus).toHaveBeenCalledWith(3);
		expect(result).toBe(12);
	});

	it('reports the reason and returns null when the new problem is discarded', async () => {
		createHypothetical.mockResolvedValue({ status: 200, data: { state_id: 3, problem_id: 12 } });
		modifyProblemStatus.mockResolvedValue({
			status: 200,
			data: { is_ready: true, problem_deleted: true, error: 'infeasible' }
		});

		const result = await create_hypothetical(7, null, assumptions, { poll_interval_ms: 0 });

		expect(result).toBeNull();
		expect(get(errorMessage)).toBe('infeasible');
	});

	it('reports the backend detail and does not poll when creation is refused', async () => {
		createHypothetical.mockResolvedValue({
			status: 400,
			data: { detail: 'Already assumed differently: structure = dry.' }
		});

		const result = await create_hypothetical(7, null, assumptions);

		expect(result).toBeNull();
		expect(modifyProblemStatus).not.toHaveBeenCalled();
		expect(get(errorMessage)).toContain('structure = dry');
	});
});

describe('default_solution_name', () => {
	it('names a solution by its state id and 1-based index', () => {
		expect(default_solution_name({ state_id: 57, solution_index: 1 })).toBe('Solution 57.2');
	});

	it('treats a missing index as the first solution', () => {
		expect(default_solution_name({ state_id: 3, solution_index: null })).toBe('Solution 3.1');
	});
});
