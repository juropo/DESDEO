/* JINA states with their result kind narrowed. */

import type {
	JinaAnalysisResponse,
	JinaStateResponse,
	JinaSetupResult,
	JinaIterateStateResult,
	JinaEvaluateResult,
	JinaSelectResult
} from '$lib/gen/endpoints/DESDEOFastAPI';

export type View = 'aggregated' | 'per_scenario';
export type Measure = JinaSetupResult['measure'];

export type SetupState = JinaStateResponse & { result: JinaSetupResult };
export type IterateState = JinaStateResponse & { result: JinaIterateStateResult };
export type EvaluateState = JinaStateResponse & { result: JinaEvaluateResult };
export type SelectState = JinaStateResponse & { result: JinaSelectResult };

/** A robustness analysis with the evaluation, objectives and thresholds it was computed for. */
export interface AnalysisResult {
	evaluateStateId: number;
	scenarioModelId: number;
	analysis: JinaAnalysisResponse;
	objectives: string[];
	thresholds: Record<string, number>;
	nLeaves: number;
}

/** The analysis entries of an evaluation; an emptied entry holds null. */
export interface AnalysisInputs {
	thresholds: Record<string, number | null>;
	absoluteFloors: Record<string, number | null>;
	lossFloors: Record<string, number | null>;
}
