/**
 * Framework-agnostic what-if logic: debouncing, cancellation, staleness, and delta arithmetic.
 * Kept out of the Svelte components so it can be unit tested (see `whatif.test.ts`).
 */

import type { NetworkEvidence } from './types';

/** Coalesces a burst of state-chip clicks into one request. */
export const EVIDENCE_DEBOUNCE_MS = 150;

/** Transition duration for `GraphNode.svelte`'s posterior bar widths (not node positions). */
export const BAR_TRANSITION_MS = 250;

/** Minimum |delta| (as a probability) for `GraphNode` to show a delta callout. */
export const DELTA_DISPLAY_THRESHOLD = 0.01;

export interface Debouncer {
	/** Schedules `fn` after the delay, replacing any pending call. */
	run(fn: () => void): void;
	/** Cancels a pending scheduling, if any, without running it. */
	cancel(): void;
}

/** A minimal debouncer. */
export function createDebouncer(delayMs: number): Debouncer {
	let timer: ReturnType<typeof setTimeout> | null = null;
	return {
		run(fn) {
			if (timer !== null) clearTimeout(timer);
			timer = setTimeout(() => {
				timer = null;
				fn();
			}, delayMs);
		},
		cancel() {
			if (timer !== null) {
				clearTimeout(timer);
				timer = null;
			}
		}
	};
}

export interface WhatIfRunner<T> {
	/** Debounce, then run `fetcher` with a fresh `AbortSignal`. A previous in-flight call is
	 *  aborted, and its result ignored even if the fetcher doesn't honour `signal`.
	 *  `onStale(true)` fires immediately, before the debounce delay. */
	request(fetcher: (signal: AbortSignal) => Promise<T>): void;
	/** Cancels any pending debounce and in-flight request, without calling any callback for it. */
	cancel(): void;
}

/**
 * Debounced, cancellable orchestration for repeated what-if requests.
 *
 * Callbacks are only called for the latest `request(...)`; superseded or aborted calls are silent.
 */
export function createWhatIfRunner<T>(
	delayMs: number,
	onResult: (result: T) => void,
	onError: (error: unknown) => void,
	onStale: (stale: boolean) => void
): WhatIfRunner<T> {
	const debouncer = createDebouncer(delayMs);
	let generation = 0;
	let controller: AbortController | null = null;

	function cancel() {
		// Bumping `generation` silences fetchers that ignore the abort signal.
		generation += 1;
		debouncer.cancel();
		controller?.abort();
		controller = null;
	}

	function request(fetcher: (signal: AbortSignal) => Promise<T>) {
		generation += 1;
		const myGeneration = generation;
		onStale(true);
		debouncer.run(() => {
			// Abort a previous request that may still be in flight.
			controller?.abort();
			const myController = new AbortController();
			controller = myController;
			fetcher(myController.signal).then(
				(result) => {
					if (myGeneration !== generation) return; // superseded by a newer request()
					onResult(result);
					if (myGeneration === generation) onStale(false);
				},
				(error: unknown) => {
					if (myGeneration !== generation) return;
					if (!(error instanceof DOMException && error.name === 'AbortError')) {
						onError(error);
					}
					if (myGeneration === generation) onStale(false);
				}
			);
		});
	}

	return { request, cancel };
}

/** Elementwise `current - baseline`, or `null` if either is missing or the lengths differ. */
export function computeDelta(
	baseline: number[] | undefined,
	current: number[] | undefined
): number[] | null {
	if (!baseline || baseline.length !== current?.length) return null;
	return current.map((value, index) => value - baseline[index]);
}

/**
 * Index of `delta`'s largest-magnitude entry, or `null` if it is below `threshold` (or `delta` is
 * `null`/empty).
 */
export function dominantDeltaIndex(
	delta: number[] | null,
	threshold: number = DELTA_DISPLAY_THRESHOLD
): number | null {
	if (!delta || delta.length === 0) return null;
	let best = 0;
	for (let i = 1; i < delta.length; i++) {
		if (Math.abs(delta[i]) > Math.abs(delta[best])) best = i;
	}
	return Math.abs(delta[best]) >= threshold ? best : null;
}

/** Whether two evidence maps differ, ignoring key order. */
export function hasEvidenceDiverged(a: NetworkEvidence, b: NetworkEvidence): boolean {
	const aKeys = Object.keys(a);
	const bKeys = Object.keys(b);
	if (aKeys.length !== bKeys.length) return true;
	return aKeys.some((key) => a[key] !== b[key]);
}

/** Ideal/nadir passed to the host's `onCommitEvidence` after the backend's ratchet has run. */
export interface CommitBounds {
	ideal?: Record<string, number>;
	nadir?: Record<string, number>;
}

/** Result of `fetchCommitBounds`: either `bounds` or an `error` message. */
export interface FetchCommitBoundsResult {
	bounds?: CommitBounds;
	error?: string;
}

/**
 * Re-runs `evidence` through `fetchWhatIf` with `commit: true`, which triggers the backend's
 * ideal/nadir ratchet, and resolves to the resulting bounds. Never rejects; failures resolve to
 * an `error` message for the caller to show.
 *
 * `evidence` must already be filtered to chance nodes.
 */
export async function fetchCommitBounds(
	fetchWhatIf: (
		modelId: string | number,
		evidence: NetworkEvidence,
		signal: AbortSignal | undefined,
		commit: boolean
	) => Promise<{ ideal?: Record<string, number> | null; nadir?: Record<string, number> | null }>,
	modelId: string | number,
	evidence: NetworkEvidence
): Promise<FetchCommitBoundsResult> {
	try {
		const result = await fetchWhatIf(modelId, evidence, undefined, true);
		return { bounds: { ideal: result.ideal ?? undefined, nadir: result.nadir ?? undefined } };
	} catch (e) {
		return { error: e instanceof Error ? e.message : 'Failed to fetch commit bounds' };
	}
}
