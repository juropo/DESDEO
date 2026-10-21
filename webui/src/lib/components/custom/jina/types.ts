/* JINA states with their result kind narrowed. */

import type {
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
