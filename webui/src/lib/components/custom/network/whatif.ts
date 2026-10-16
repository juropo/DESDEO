/**
 * Pure, framework-agnostic what-if plumbing (graphical models plan, frontend phase D) — the
 * debounce/cancel/staleness state machine and the ghost/delta arithmetic, kept out of
 * `NetworkPanel.svelte`/`NetworkView.svelte` for the same reason `interaction.ts`/
 * `dag-algorithms.ts` are pure: this package has no component-testing setup, so anything worth
 * getting right under test has to be a plain `.ts` function (see `whatif.test.ts`).
 */

import type { NetworkEvidence } from './types';

/** ~150ms per the plan — long enough to coalesce a burst of state-chip clicks into one request,
 *  short enough that a single click still feels responsive. */
export const EVIDENCE_DEBOUNCE_MS = 150;

/** 200–300ms per the plan ("motion is what communicates change") — the CSS transition duration
 *  `GraphNode.svelte`'s posterior bars/chips use for width changes. Never applied to node
 *  position/layout, only to bar fill. */
export const BAR_TRANSITION_MS = 250;

/** Minimum |delta| (as a probability, i.e. 0.01 = one percentage point) before `GraphNode` bothers
 *  surfacing a dominant-delta callout — filters out floating-point noise and genuinely
 *  imperceptible shifts rather than flagging every node as "changed". */
export const DELTA_DISPLAY_THRESHOLD = 0.01;

export interface Debouncer {
	/** Schedules `fn` to run after the debouncer's delay, cancelling any not-yet-fired previous
	 *  scheduling — the standard "reset the timer on every call" debounce. */
	run(fn: () => void): void;
	/** Cancels a pending scheduling, if any, without running it. */
	cancel(): void;
}

/** A minimal, dependency-free debouncer. Kept separate from `createWhatIfRunner` below so the
 *  timing primitive itself (easy to get subtly wrong: does a second call within the window reset
 *  the timer or no-op? does `cancel` actually prevent a pending run?) has its own direct test
 *  coverage, independent of the request-generation/abort logic layered on top of it. */
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
	/** Debounce, then run `fetcher` with a fresh `AbortSignal`. Any previously in-flight call is
	 *  aborted (so a fetcher that honours `signal` stops the actual network request, not just its
	 *  own callback) and, independently, generation-guarded (so a fetcher that *doesn't* honour
	 *  `signal` — e.g. a synchronous demo mock — still can't let a stale response clobber a newer
	 *  one). `onStale(true)` fires synchronously, the moment a new request is scheduled — dimming
	 *  has to start immediately, not once the network call actually begins, or a burst of clicks
	 *  reads as "nothing happened" for the whole debounce window. */
	request(fetcher: (signal: AbortSignal) => Promise<T>): void;
	/** Cancels any pending debounce and aborts any in-flight request, without ever calling
	 *  `onResult`/`onError`/`onStale(false)` for it — for a caller switching away from whatever
	 *  this runner was querying (e.g. `NetworkPanel`'s `modelId` changing) and needing every
	 *  trace of the old query gone, not just superseded. */
	cancel(): void;
}

/**
 * Debounced, cancellable, generation-guarded orchestration for repeated what-if refreshes —
 * shared by `NetworkPanel.svelte` (the real `/graphical/whatif` fetch) and the Cumulus demo
 * dialog's mock what-if, so both exercise (and both get to rely on) exactly the same
 * stale-response/staleness-dimming semantics rather than two independently-written, independently
 * buggy copies.
 *
 * `onResult`/`onError`/`onStale` are called only for the *latest* `request(...)` call — an
 * intentionally superseded or aborted call is silent (no `onError` for an `AbortError`, no
 * `onResult`/`onStale(false)` for a generation that's no longer current).
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
		// Bump `generation` *before* aborting -- an in-flight fetcher (past its own debounce
		// already) may legitimately ignore `signal` (see `WhatIfRunner.request`'s own doc comment),
		// in which case `abort()` alone changes nothing about whether it still resolves; bumping
		// `generation` is what actually makes its eventual `.then`/`.catch` see a stale
		// `myGeneration !== generation` and stay silent, regardless of whether the fetcher itself
		// ever notices the abort.
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
			// A previous call's request may still be in flight if it was itself debounced past
			// its own delay before this one fired (e.g. delayMs shortened mid-flight) -- abort it
			// defensively even though the generation guard below would also catch its result.
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

/**
 * Elementwise `current - baseline`, or `null` when either side is absent or the arrays don't
 * line up (a mismatched-length pair is never meaningful to diff, e.g. a posterior fetched for a
 * different model). The ghost-mark/delta arithmetic behind every posterior bar in `GraphNode`.
 */
export function computeDelta(
	baseline: number[] | undefined,
	current: number[] | undefined
): number[] | null {
	if (!baseline || baseline.length !== current?.length) return null;
	return current.map((value, index) => value - baseline[index]);
}

/**
 * Index of `delta`'s largest-magnitude entry, or `null` if every entry is below
 * `DELTA_DISPLAY_THRESHOLD` (or `delta` itself is `null`/empty) — the single callout `GraphNode`
 * shows per node rather than a full per-state delta list, which wouldn't fit at any detail tier.
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

/**
 * Whether two evidence maps differ — the "is there anything exploratory to commit" check behind
 * `NetworkView`'s Commit button. Order-independent (compares as sets of entries, not insertion
 * order), which is what `NetworkEvidence`'s own contract (a plain record) already implies.
 */
export function hasEvidenceDiverged(a: NetworkEvidence, b: NetworkEvidence): boolean {
	const aKeys = Object.keys(a);
	const bKeys = Object.keys(b);
	if (aKeys.length !== bKeys.length) return true;
	return aKeys.some((key) => a[key] !== b[key]);
}
