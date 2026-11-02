/**
 * CUMULUS API Client-Side Handlers
 */

import {
	solveSolutionsMethodCumulusSolvePost,
	getOrInitializeMethodCumulusGetOrInitializePost,
	saveMethodCumulusSavePost,
	deleteSaveMethodCumulusDeleteSavePost,
	finalizeMethodCumulusFinalizePost,
	solveIntermediateMethodCumulusIntermediatePost,
	getUtopiaDataUtopiaPost,
	setObjectiveConstraintsMethodCumulusObjectiveConstraintPost,
	getObjectiveConstraintsMethodCumulusObjectiveConstraintGet,
	getSolutionDescriptionSolutionDescriptionGetPost,
	getScenarioAvailabilityMethodCumulusScenarioAvailabilityGet,
	applyScenarioMeasuresMethodCumulusApplyScenarioMeasuresPost,
	createHypotheticalMethodCumulusHypotheticalPost,
	modifyProblemStatusMethodCumulusModifyProblemStatusStateIdGet
} from '$lib/gen/endpoints/DESDEOFastAPI';
import type {
	CumulusClassificationRequest,
	CumulusInitializationRequest,
	CumulusSaveRequest,
	CumulusDeleteSaveRequest,
	CumulusFinalizeRequest,
	IntermediateSolutionRequest,
	SolutionInfo,
	CumulusObjectiveConstraintRequest,
	CumulusScenarioAvailabilityResponse,
	CumulusModificationResponse,
	DiagramAssumptions,
	SolutionDescriptionRequest,
	ConstraintTypeEnum,
	ProblemInfo,
	SolutionReferenceResponse as Solution,
	ReferencePoint
} from '$lib/gen/endpoints/DESDEOFastAPI';
import type { MapOptions, Response } from './types';
import { errorMessage, isLoading } from '../../../stores/uiState';

/** Name of an unnamed solution: the id of the state that produced it, and its index there. */
export function default_solution_name(
	solution: Pick<Solution, 'state_id' | 'solution_index'>
): string {
	return `Solution ${solution.state_id}.${(solution.solution_index ?? 0) + 1}`;
}

/** `value`, or null when it is missing or empty. */
export function blankToNull(value: string | null | undefined): string | null {
	if (!value) return null;
	return value;
}

/** Convert a Solution to a SolutionInfo for API requests. */
function toSolutionInfo(solution: Solution, name?: string | null): SolutionInfo {
	return {
		state_id: solution.state_id,
		solution_index: solution.solution_index ?? 0,
		name: name ?? solution.name
	};
}

/**
 * Handles the generation of intermediate solutions between two selected reference solutions.
 */
export async function handle_intermediate(
	problem: ProblemInfo | null,
	selected_solutions: Solution[],
	num_desired: number
): Promise<Response | null> {
	if (!problem) {
		errorMessage.set('No problem selected');
		return null;
	}
	if (selected_solutions.length !== 2) {
		errorMessage.set('Exactly 2 solutions must be selected for intermediate solutions');
		return null;
	}

	isLoading.set(true);
	errorMessage.set(null);

	try {
		const request: IntermediateSolutionRequest = {
			problem_id: problem.id,
			reference_solution_1: toSolutionInfo(selected_solutions[0]),
			reference_solution_2: toSolutionInfo(selected_solutions[1]),
			num_desired: num_desired
		};

		const response = await solveIntermediateMethodCumulusIntermediatePost(request);

		if (response.status !== 200) {
			errorMessage.set(`Intermediate solutions failed with status ${response.status}`);
			return null;
		}

		return response.data as unknown as Response;
	} catch (error) {
		const msg = error instanceof Error ? error.message : 'Unknown error';
		errorMessage.set(msg);
		console.error('Error in handle_intermediate:', msg);
		return null;
	} finally {
		isLoading.set(false);
	}
}

/**
 * Handles a CUMULUS iteration based on user-defined reference point preferences.
 */
export async function handle_iterate(
	problem: ProblemInfo,
	current_preference: number[],
	selected_iteration_objectives: Record<string, number>,
	_current_num_iteration_solutions: number,
	scalarizations: string[] = ['cumulonimbus']
): Promise<Response | null> {
	isLoading.set(true);
	errorMessage.set(null);

	try {
		const preference: ReferencePoint = {
			preference_type: 'reference_point',
			aspiration_levels: problem.objectives.reduce(
				(acc, obj, idx) => {
					acc[obj.symbol] = current_preference[idx];
					return acc;
				},
				{} as Record<string, number>
			)
		};

		const request: CumulusClassificationRequest = {
			problem_id: problem.id,
			current_objectives: selected_iteration_objectives,
			preference: preference,
			scalarizations: scalarizations
		};

		const response = await solveSolutionsMethodCumulusSolvePost(request);

		if (response.status !== 200) {
			errorMessage.set(`Iteration failed with status ${response.status}`);
			return null;
		}

		return response.data as unknown as Response;
	} catch (error) {
		const msg = error instanceof Error ? error.message : 'Unknown error';
		errorMessage.set(msg);
		console.error('Error in handle_iterate:', msg);
		return null;
	} finally {
		isLoading.set(false);
	}
}

/**
 * Saves a solution with an optional user-provided name.
 */
export async function handle_save(
	problem: ProblemInfo | null,
	solution: Solution,
	name: string | undefined
): Promise<boolean> {
	if (!problem) {
		errorMessage.set('No problem selected');
		return false;
	}

	isLoading.set(true);
	errorMessage.set(null);

	try {
		const request: CumulusSaveRequest = {
			problem_id: problem.id,
			solution_info: [toSolutionInfo(solution, name ?? null)]
		};

		const response = await saveMethodCumulusSavePost(request);

		if (response.status !== 200) {
			errorMessage.set(`Save failed with status ${response.status}`);
			return false;
		}

		return true;
	} catch (error) {
		const msg = error instanceof Error ? error.message : 'Unknown error';
		errorMessage.set(msg);
		console.error('Error in handle_save:', msg);
		return false;
	} finally {
		isLoading.set(false);
	}
}

/**
 * Removes a previously saved solution.
 */
export async function handle_remove_saved(
	problem: ProblemInfo | null,
	solution: Solution
): Promise<boolean> {
	if (!problem) {
		errorMessage.set('No problem selected');
		return false;
	}

	isLoading.set(true);
	errorMessage.set(null);

	try {
		const request: CumulusDeleteSaveRequest = {
			state_id: solution.state_id,
			solution_index: solution.solution_index ?? 0,
			problem_id: problem.id
		};

		const response = await deleteSaveMethodCumulusDeleteSavePost(request);

		if (response.status !== 200) {
			errorMessage.set(`Delete save failed with status ${response.status}`);
			return false;
		}

		return true;
	} catch (error) {
		const msg = error instanceof Error ? error.message : 'Unknown error';
		errorMessage.set(msg);
		console.error('Error in handle_remove_saved:', msg);
		return false;
	} finally {
		isLoading.set(false);
	}
}

/**
 * Marks a solution as the final chosen solution for the session.
 */
export async function handle_finish(
	problem: ProblemInfo | null,
	solution: Solution
): Promise<boolean> {
	if (!problem) {
		errorMessage.set('No problem selected');
		return false;
	}

	isLoading.set(true);
	errorMessage.set(null);

	try {
		const request: CumulusFinalizeRequest = {
			problem_id: problem.id,
			solution_info: toSolutionInfo(solution)
		};

		const response = await finalizeMethodCumulusFinalizePost(request);

		if (response.status !== 200) {
			errorMessage.set(`Finalize failed with status ${response.status}`);
			return false;
		}

		return true;
	} catch (error) {
		const msg = error instanceof Error ? error.message : 'Unknown error';
		errorMessage.set(msg);
		console.error('Error in handle_finish:', msg);
		return false;
	} finally {
		isLoading.set(false);
	}
}

/**
 * Fetches map data related to a specific solution for UTOPIA visualization.
 */
export async function get_maps(
	problem: ProblemInfo,
	solution: Solution
): Promise<{
	years: string[];
	options: Record<string, MapOptions>;
	map_json: object;
	map_name: string;
	description: string;
} | null> {
	isLoading.set(true);
	errorMessage.set(null);

	try {
		const response = await getUtopiaDataUtopiaPost({
			problem_id: problem.id,
			solution: toSolutionInfo(solution)
		});

		if (response.status !== 200) {
			errorMessage.set(`Get maps failed with status ${response.status}`);
			return null;
		}

		const result = response.data;
		const options = result.options as Record<string, MapOptions>;
		for (const year of result.years) {
			const tooltip = options[year]?.tooltip;
			if (tooltip?.formatterEnabled) {
				tooltip.formatter = (params) => params.name;
			}
		}

		return { ...result, options };
	} catch (error) {
		const msg = error instanceof Error ? error.message : 'Unknown error';
		errorMessage.set(msg);
		console.error('Error in get_maps:', msg);
		return null;
	} finally {
		isLoading.set(false);
	}
}

/**
 * Fetches a textual solution description from the solution_description endpoint.
 */
export async function get_solution_description(
	problem: ProblemInfo,
	solution: Solution
): Promise<string | null> {
	try {
		const request: SolutionDescriptionRequest = {
			problem_id: problem.id,
			solution: toSolutionInfo(solution)
		};
		const response = await getSolutionDescriptionSolutionDescriptionGetPost(request);

		if (response.status !== 200 || !response.data.available) return null;
		return response.data.description;
	} catch {
		return null;
	}
}

/**
 * Initializes a new CUMULUS state for a given problem, or retrieves the latest one.
 *
 * Pass `skip_scenarios = true` to bypass scenario detection and force plain initialization
 * (used when the DM declines to build a combined scenario problem).
 */
export async function initialize_cumulus_state(
	problem_id: number,
	skip_scenarios = false
): Promise<Response | null> {
	isLoading.set(true);
	errorMessage.set(null);

	try {
		const request: CumulusInitializationRequest = {
			problem_id: problem_id,
			// An empty array signals "skip scenario detection"; null triggers it.
			uncertainty_measures: skip_scenarios ? [] : null
		};

		const response = await getOrInitializeMethodCumulusGetOrInitializePost(request);

		if (response.status !== 200) {
			errorMessage.set(`Initialization failed with status ${response.status}`);
			return null;
		}

		return response.data as unknown as Response;
	} catch (error) {
		const msg = error instanceof Error ? error.message : 'Unknown error';
		errorMessage.set(msg);
		console.error('Error in initialize_cumulus_state:', msg);
		return null;
	} finally {
		isLoading.set(false);
	}
}

export type MeasureType =
	| 'expected_value'
	| 'worst_case_robust'
	| 'conditional_value_at_risk'
	| 'weighted_scenarios';

export interface MeasureOptions {
	cvar_alpha?: number;
}

function build_uncertainty_measures_request(
	problem_id: number,
	scenario_model_id: number,
	objective_symbols: string[],
	selected_measures: MeasureType[],
	measure_options: MeasureOptions,
	name?: string
): CumulusInitializationRequest {
	return {
		problem_id: problem_id,
		name: blankToNull(name),
		uncertainty_measures: selected_measures.map((measure_type) => ({
			measure_type,
			scenario_model_id,
			symbols: objective_symbols,
			...(measure_type === 'conditional_value_at_risk' && {
				alpha: measure_options.cvar_alpha ?? 0.05
			})
		}))
	};
}

/**
 * Re-calls get-or-initialize with the DM-chosen uncertainty measures applied,
 * building a combined multi-scenario problem before starting the method.
 */
export async function initialize_cumulus_state_with_scenarios(
	problem_id: number,
	scenario_model_id: number,
	objective_symbols: string[],
	selected_measures: MeasureType[],
	measure_options: MeasureOptions = {},
	name?: string
): Promise<Response | null> {
	isLoading.set(true);
	errorMessage.set(null);

	try {
		const request = build_uncertainty_measures_request(
			problem_id,
			scenario_model_id,
			objective_symbols,
			selected_measures,
			measure_options,
			name
		);

		const response = await getOrInitializeMethodCumulusGetOrInitializePost(request);

		if (response.status !== 200) {
			errorMessage.set(`Scenario initialization failed with status ${response.status}`);
			return null;
		}

		return response.data as unknown as Response;
	} catch (error) {
		const msg = error instanceof Error ? error.message : 'Unknown error';
		errorMessage.set(msg);
		console.error('Error in initialize_cumulus_state_with_scenarios:', msg);
		return null;
	} finally {
		isLoading.set(false);
	}
}

/**
 * Checks whether a problem (or the base problem it was combined from) has an associated
 * scenario model, so the "Add uncertainty measures" action can be shown or hidden accordingly.
 */
export async function check_scenario_availability(
	problem_id: number
): Promise<CumulusScenarioAvailabilityResponse | null> {
	try {
		const response = await getScenarioAvailabilityMethodCumulusScenarioAvailabilityGet({
			problem_id
		});

		if (response.status !== 200) {
			return null;
		}

		return response.data as unknown as CumulusScenarioAvailabilityResponse;
	} catch (error) {
		console.error(
			'Error in check_scenario_availability:',
			error instanceof Error ? error.message : error
		);
		return null;
	}
}

/**
 * Builds (or rebuilds) a combined scenario problem with the given uncertainty measures,
 * regardless of whether CUMULUS has already been initialized for the current problem.
 * Used by the "Add uncertainty measures" menu action, which can be used both before the
 * method has started and later, to change which measures are included.
 */
export async function apply_scenario_measures(
	problem_id: number,
	scenario_model_id: number,
	objective_symbols: string[],
	selected_measures: MeasureType[],
	measure_options: MeasureOptions = {},
	name?: string
): Promise<Response | null> {
	isLoading.set(true);
	errorMessage.set(null);

	try {
		const request = build_uncertainty_measures_request(
			problem_id,
			scenario_model_id,
			objective_symbols,
			selected_measures,
			measure_options,
			name
		);

		const response = await applyScenarioMeasuresMethodCumulusApplyScenarioMeasuresPost(request);

		if (response.status !== 200) {
			errorMessage.set(`Applying uncertainty measures failed with status ${response.status}`);
			return null;
		}

		return response.data as unknown as Response;
	} catch (error) {
		const msg = error instanceof Error ? error.message : 'Unknown error';
		errorMessage.set(msg);
		console.error('Error in apply_scenario_measures:', msg);
		return null;
	} finally {
		isLoading.set(false);
	}
}

/** Poll cadence and give-up point for the background checks of a new problem. */
const MODIFICATION_POLL_INTERVAL_MS = 1000;
const MODIFICATION_POLL_TIMEOUT_MS = 10 * 60 * 1000;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

const DISCARDED_MESSAGE = 'The new problem could not be used and was discarded.';

export type ModificationOutcome =
	/** The background checks passed and the new problem is ready to be used. */
	| { kind: 'accepted'; response: CumulusModificationResponse }
	/** The checks failed and the new problem was discarded. */
	| { kind: 'rejected'; reason: string }
	/** Neither could be established: polling itself failed or ran out of time. */
	| { kind: 'failed' };

/**
 * Polls /modify-problem/status until the background checks settle.
 *
 * `is_ready` only means the checks have finished, so a ready response still has to be read for
 * the verdict.
 */
export async function poll_modification_status(
	state_id: number,
	{
		interval_ms = MODIFICATION_POLL_INTERVAL_MS,
		timeout_ms = MODIFICATION_POLL_TIMEOUT_MS
	}: { interval_ms?: number; timeout_ms?: number } = {}
): Promise<ModificationOutcome> {
	const deadline = Date.now() + timeout_ms;

	for (;;) {
		const response = await modifyProblemStatusMethodCumulusModifyProblemStatusStateIdGet(state_id);

		// A discarded problem can take its state with it, so a missing state means rejection.
		if ((response.status as number) === 404) {
			return { kind: 'rejected', reason: DISCARDED_MESSAGE };
		}

		if (response.status !== 200) {
			errorMessage.set(
				errorDetail(response.data) ??
					`Checking the new problem failed with status ${response.status}`
			);
			return { kind: 'failed' };
		}

		if (response.data.is_ready) {
			if (response.data.problem_deleted || response.data.error) {
				return { kind: 'rejected', reason: response.data.error ?? DISCARDED_MESSAGE };
			}
			return { kind: 'accepted', response: response.data };
		}

		if (Date.now() >= deadline) {
			errorMessage.set(
				'Checking the new problem is taking longer than expected; try again in a moment.'
			);
			return { kind: 'failed' };
		}

		await sleep(interval_ms);
	}
}

/**
 * Creates a hypothetical problem from a graphical-model problem under `assumptions` and waits
 * for its background checks. Returns the new problem's id, or null after reporting an error.
 */
export async function create_hypothetical(
	problem_id: number,
	parent_state_id: number | null,
	assumptions: DiagramAssumptions,
	{ poll_interval_ms }: { poll_interval_ms?: number } = {}
): Promise<number | null> {
	errorMessage.set(null);
	const response = await createHypotheticalMethodCumulusHypotheticalPost({
		problem_id,
		parent_state_id,
		assumptions
	});
	if (response.status !== 200) {
		errorMessage.set(
			errorDetail(response.data) ??
				`Creating the hypothetical problem failed with status ${response.status}`
		);
		return null;
	}

	const outcome = await poll_modification_status(response.data.state_id, {
		interval_ms: poll_interval_ms
	});
	if (outcome.kind === 'rejected') errorMessage.set(outcome.reason);
	return outcome.kind === 'accepted' ? response.data.problem_id : null;
}

type UIConstraintType = '<=0' | '=0';

export interface UIConstraint {
	func: string;
	type: UIConstraintType;
}

export interface ObjectiveConstraintResult {
	/** The objective-constraint state holding the soft list now in effect. */
	state_id: number;
	soft_constraint_ids: number[];
}

/** FastAPI reports an HTTPException as a plain string detail, validation errors as a list. */
function errorDetail(data: unknown): string | null {
	const detail = (data as { detail?: unknown } | null)?.detail;
	return typeof detail === 'string' ? detail : null;
}

/**
 * Fetches the soft objective constraints in effect, in the form the dialog edits them.
 *
 * The dialog stores the expression the decision maker typed as the constraint's name, so that
 * is what is shown again.
 */
export async function get_objective_constraints(
	problem_id: number
): Promise<UIConstraint[] | null> {
	try {
		const response = await getObjectiveConstraintsMethodCumulusObjectiveConstraintGet({
			problem_id
		});
		if (response.status !== 200) return null;

		return response.data.soft_constraints.map((c) => ({
			func: c.name,
			type: c.cons_type === '<=' ? '<=0' : '=0'
		}));
	} catch (error) {
		console.error(
			'Error in get_objective_constraints:',
			error instanceof Error ? error.message : error
		);
		return null;
	}
}

/**
 * Sets the soft objective constraints of the current session, replacing any earlier list.
 */
export async function set_objective_constraints(
	problem: ProblemInfo,
	parent_state_id: number | null,
	constraints: UIConstraint[]
): Promise<ObjectiveConstraintResult | null> {
	isLoading.set(true);
	errorMessage.set(null);

	// Sent even when empty: the endpoint replaces the stored list, which is how the decision
	// maker clears soft constraints they have removed from the dialog.
	const request: CumulusObjectiveConstraintRequest = {
		problem_id: problem.id,
		parent_state_id: parent_state_id ?? null,
		soft_constraints: constraints.map((c, i) => ({
			name: c.func,
			symbol: `oc_${i + 1}`,
			func: c.func,
			cons_type: (c.type === '<=0' ? '<=' : '=') as ConstraintTypeEnum
		}))
	};

	try {
		const response = await setObjectiveConstraintsMethodCumulusObjectiveConstraintPost(request);

		if (response.status !== 200) {
			errorMessage.set(
				errorDetail(response.data) ??
					`Setting objective constraints failed with status ${response.status}`
			);
			return null;
		}

		return response.data;
	} catch (error) {
		const msg = error instanceof Error ? error.message : 'Unknown error';
		errorMessage.set(msg);
		console.error('Error in set_objective_constraints:', msg);
		return null;
	} finally {
		isLoading.set(false);
	}
}
