/**
 * Sensitivity tornado + value-of-information wire shapes and pure logic (graphical models plan,
 * frontend phase E). Mirrors `whatif.ts`'s own split: fetch/mapping is here as plain, testable
 * functions, `TornadoChart.svelte` only ever renders already-mapped data — the same
 * pure-logic-behind-a-thin-component pattern `dag-algorithms.ts`/`interaction.ts`/`whatif.ts`
 * already follow, for the same reason (no component-testing setup in this repo).
 *
 * Backed by `POST /graphical/sensitivity` / `POST /graphical/voi` (`desdeo/api/routers/
 * graphical.py`, backend phase E) — a single-node hard-evidence sweep on the same warm,
 * session-scoped pyAgrum engine `/whatif` already uses.
 */

/** One swept node's tornado-bar entry — camelCase mirror of `SensitivityNodeResult`
 *  (`graphical.py`). */
export interface SensitivityNodeResult {
	node: string;
	/** State label -> the target's value with `node` hard-pinned to that state. */
	values: Record<string, number>;
	low: number;
	high: number;
	/** `high - low` — the tornado bar's own magnitude. */
	swing: number;
}

/** Camelcase mirror of `SensitivityResponse` (`graphical.py`) — what `fetchSensitivity` resolves
 *  to. `results` is already sorted descending by `swing` server-side; `sortByImpact` below is for
 *  a caller that re-derives/re-orders a subset rather than trusting that ordering blindly. */
export interface SensitivityResult {
	baseValue: number;
	results: SensitivityNodeResult[];
}

/** Camelcase mirror of `VoiResponse` (`graphical.py`) — what `fetchVoi` resolves to. */
export interface VoiResult {
	baseValue: number;
	expectedValueWithObservation: number;
	valueOfInformation: number;
	marginal: Record<string, number>;
}

/** Raw JSON shape `POST /graphical/sensitivity` returns — snake_case, matching `SensitivityResponse`. */
interface RawSensitivityResponse {
	base_value: number;
	results: {
		node: string;
		values: Record<string, number>;
		low: number;
		high: number;
		swing: number;
	}[];
}

/** Raw JSON shape `POST /graphical/voi` returns — snake_case, matching `VoiResponse`. */
interface RawVoiResponse {
	base_value: number;
	expected_value_with_observation: number;
	value_of_information: number;
	marginal: Record<string, number>;
}

export function mapSensitivityResponse(raw: RawSensitivityResponse): SensitivityResult {
	return {
		baseValue: raw.base_value,
		results: raw.results.map((r) => ({
			node: r.node,
			values: r.values,
			low: r.low,
			high: r.high,
			swing: r.swing
		}))
	};
}

export function mapVoiResponse(raw: RawVoiResponse): VoiResult {
	return {
		baseValue: raw.base_value,
		expectedValueWithObservation: raw.expected_value_with_observation,
		valueOfInformation: raw.value_of_information,
		marginal: raw.marginal
	};
}

/** `results`, sorted descending by `swing` (biggest bar first) — the standard tornado-diagram
 *  ordering. `SensitivityResult.results` already arrives sorted this way from the backend; this
 *  is for a caller that filters/recombines results and needs to re-establish the ordering rather
 *  than trusting it was preserved. Immutable — returns a new array. */
export function sortByImpact(results: SensitivityNodeResult[]): SensitivityNodeResult[] {
	return [...results].sort((a, b) => b.swing - a.swing);
}

/** The `[min, max]` domain a tornado chart's shared horizontal scale must cover — every result's
 *  `low`/`high` plus `baseValue` itself (the reference line has to fit on the axis too, even if
 *  it happens to fall outside every swept range, e.g. a target with strong evidence already set). */
export function tornadoDomain(
	results: SensitivityNodeResult[],
	baseValue: number
): [number, number] {
	let min = baseValue;
	let max = baseValue;
	for (const r of results) {
		if (r.low < min) min = r.low;
		if (r.high > max) max = r.high;
	}
	// A degenerate domain (every value identical, e.g. all-zero swings) still needs a non-zero
	// width to render a bar chart against — widen symmetrically rather than divide by zero later.
	if (min === max) {
		return [min - 1, max + 1];
	}
	return [min, max];
}

async function parseJsonOrThrow(res: Response, what: string): Promise<unknown> {
	if (!res.ok) throw new Error(`Failed to fetch ${what} (status ${res.status})`);
	return res.json();
}

/** Default `fetchSensitivity` — `POST /graphical/sensitivity` via the `/api/...` proxy, matching
 *  `NetworkPanel.svelte`'s own `defaultFetchWhatIf` convention (injectable, swappable for the
 *  generated OpenAPI client later). */
export async function defaultFetchSensitivity(
	modelId: string | number,
	evidence: Record<string, string>,
	target: string,
	targetState?: string,
	signal?: AbortSignal
): Promise<SensitivityResult> {
	const res = await fetch('/api/graphical/sensitivity', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ problem_id: modelId, evidence, target, target_state: targetState }),
		signal
	});
	return mapSensitivityResponse(
		(await parseJsonOrThrow(res, 'sensitivity')) as RawSensitivityResponse
	);
}

/** Default `fetchVoi` — `POST /graphical/voi` via the `/api/...` proxy. */
export async function defaultFetchVoi(
	modelId: string | number,
	evidence: Record<string, string>,
	node: string,
	target: string,
	signal?: AbortSignal
): Promise<VoiResult> {
	const res = await fetch('/api/graphical/voi', {
		method: 'POST',
		headers: { 'Content-Type': 'application/json' },
		body: JSON.stringify({ problem_id: modelId, evidence, node, target }),
		signal
	});
	return mapVoiResponse((await parseJsonOrThrow(res, 'value of information')) as RawVoiResponse);
}
