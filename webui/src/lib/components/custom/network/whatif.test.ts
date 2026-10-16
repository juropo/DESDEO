import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
	computeDelta,
	createDebouncer,
	createWhatIfRunner,
	dominantDeltaIndex,
	hasEvidenceDiverged
} from './whatif';

describe('createDebouncer', () => {
	beforeEach(() => vi.useFakeTimers());
	afterEach(() => vi.useRealTimers());

	it('runs fn after the delay', () => {
		const debouncer = createDebouncer(150);
		const fn = vi.fn();
		debouncer.run(fn);
		vi.advanceTimersByTime(149);
		expect(fn).not.toHaveBeenCalled();
		vi.advanceTimersByTime(1);
		expect(fn).toHaveBeenCalledTimes(1);
	});

	it('a second call within the delay resets the timer, coalescing into one run', () => {
		const debouncer = createDebouncer(150);
		const fn = vi.fn();
		debouncer.run(fn);
		vi.advanceTimersByTime(100);
		debouncer.run(fn);
		vi.advanceTimersByTime(100);
		expect(fn).not.toHaveBeenCalled(); // only 100ms since the second call
		vi.advanceTimersByTime(50);
		expect(fn).toHaveBeenCalledTimes(1);
	});

	it('cancel prevents a pending run', () => {
		const debouncer = createDebouncer(150);
		const fn = vi.fn();
		debouncer.run(fn);
		debouncer.cancel();
		vi.advanceTimersByTime(500);
		expect(fn).not.toHaveBeenCalled();
	});

	it('cancel with nothing pending is a no-op', () => {
		expect(() => createDebouncer(150).cancel()).not.toThrow();
	});
});

describe('createWhatIfRunner', () => {
	beforeEach(() => vi.useFakeTimers());
	afterEach(() => vi.useRealTimers());

	function makeRunner() {
		const onResult = vi.fn();
		const onError = vi.fn();
		const staleTransitions: boolean[] = [];
		const runner = createWhatIfRunner<string>(150, onResult, onError, (stale) =>
			staleTransitions.push(stale)
		);
		return { runner, onResult, onError, staleTransitions };
	}

	it('calls onStale(true) synchronously, then onResult and onStale(false) once resolved', async () => {
		const { runner, onResult, staleTransitions } = makeRunner();
		runner.request(() => Promise.resolve('a'));
		expect(staleTransitions).toEqual([true]);
		await vi.advanceTimersByTimeAsync(150);
		expect(onResult).toHaveBeenCalledWith('a');
		expect(staleTransitions).toEqual([true, false]);
	});

	it('a fetcher is not even called until the debounce delay elapses', () => {
		const { runner } = makeRunner();
		const fetcher = vi.fn(() => Promise.resolve('a'));
		runner.request(fetcher);
		vi.advanceTimersByTime(100);
		expect(fetcher).not.toHaveBeenCalled();
	});

	it('rapid successive requests only ever call the last fetcher', async () => {
		const { runner } = makeRunner();
		const fetcherA = vi.fn(() => Promise.resolve('a'));
		const fetcherB = vi.fn(() => Promise.resolve('b'));
		runner.request(fetcherA);
		vi.advanceTimersByTime(50);
		runner.request(fetcherB);
		await vi.advanceTimersByTimeAsync(150);
		expect(fetcherA).not.toHaveBeenCalled();
		expect(fetcherB).toHaveBeenCalledTimes(1);
	});

	it('a slower earlier request resolving after a faster later one is discarded, not clobbering', async () => {
		const { runner, onResult } = makeRunner();
		let resolveSlow: (value: string) => void = () => undefined;
		const slow = new Promise<string>((resolve) => {
			resolveSlow = resolve;
		});
		runner.request(() => slow);
		await vi.advanceTimersByTimeAsync(150); // slow's fetcher is now in flight

		// A second, independent request supersedes it before the slow one resolves.
		runner.request(() => Promise.resolve('fast'));
		await vi.advanceTimersByTimeAsync(150);
		expect(onResult).toHaveBeenCalledWith('fast');

		// The slow request finally resolves -- must not overwrite the already-current 'fast'.
		resolveSlow('slow');
		await Promise.resolve();
		await Promise.resolve();
		expect(onResult).toHaveBeenCalledTimes(1);
		expect(onResult).not.toHaveBeenCalledWith('slow');
	});

	it('AbortError from an aborted fetcher does not call onError', async () => {
		const { runner, onError, onResult } = makeRunner();
		runner.request(
			(signal) =>
				new Promise((_resolve, reject) => {
					signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
				})
		);
		await vi.advanceTimersByTimeAsync(150); // first fetcher now in flight, holding `signal`

		runner.request(() => Promise.resolve('b')); // supersedes and aborts the first
		await vi.advanceTimersByTimeAsync(150);

		expect(onError).not.toHaveBeenCalled();
		expect(onResult).toHaveBeenCalledWith('b');
	});

	it('a genuine (non-abort) error calls onError and still clears staleness', async () => {
		const { runner, onError, staleTransitions } = makeRunner();
		runner.request(() => Promise.reject(new Error('boom')));
		await vi.advanceTimersByTimeAsync(150);
		expect(onError).toHaveBeenCalledWith(new Error('boom'));
		expect(staleTransitions).toEqual([true, false]);
	});

	it('cancel() aborts an in-flight request and prevents its callbacks from firing', async () => {
		const { runner, onResult, staleTransitions } = makeRunner();
		let aborted = false;
		runner.request(
			(signal) =>
				new Promise((resolve) => {
					signal.addEventListener('abort', () => {
						aborted = true;
					});
					// Never resolves on its own within the test.
					void resolve;
				})
		);
		await vi.advanceTimersByTimeAsync(150);
		runner.cancel();
		expect(aborted).toBe(true);
		expect(onResult).not.toHaveBeenCalled();
		// onStale(true) fired on request(), but cancel() never calls onStale(false) itself --
		// callers own resetting their own staleness flag on cancel if they need to.
		expect(staleTransitions).toEqual([true]);
	});

	it('cancel() prevents callbacks even for a fetcher that ignores the abort signal entirely', async () => {
		// The component doc comments are explicit that a caller-supplied `fetchWhatIf` is free to
		// ignore `signal` -- `Promise.resolve('a')` below is exactly that: it never even looks at
		// its `signal` argument, so `controller.abort()` alone changes nothing about whether it
		// resolves. The only thing that can still stop `onResult`/`onStale(false)` from firing is
		// `cancel()` bumping `generation` itself.
		const { runner, onResult, staleTransitions } = makeRunner();
		runner.request(() => Promise.resolve('a'));
		// Fires the debounce synchronously -- the fetcher runs and its `.then` is now a pending
		// microtask, not yet processed (no `await`/microtask-flush has happened yet).
		vi.advanceTimersByTime(150);
		runner.cancel();
		// Flush the pending microtask (the fetcher's already-resolved `.then`) without advancing
		// any fake timers -- if `cancel()` didn't bump `generation`, this is exactly where the
		// stale `onResult`/`onStale(false)` would fire.
		await Promise.resolve();
		await Promise.resolve();
		expect(onResult).not.toHaveBeenCalled();
		expect(staleTransitions).toEqual([true]);
	});

	it('cancel() also clears a still-pending (not yet fired) debounce', () => {
		const { runner } = makeRunner();
		const fetcher = vi.fn(() => Promise.resolve('a'));
		runner.request(fetcher);
		runner.cancel();
		vi.advanceTimersByTime(500);
		expect(fetcher).not.toHaveBeenCalled();
	});
});

describe('computeDelta', () => {
	it('returns null when either side is missing', () => {
		expect(computeDelta(undefined, [0.1, 0.9])).toBeNull();
		expect(computeDelta([0.1, 0.9], undefined)).toBeNull();
	});

	it('returns null on a length mismatch', () => {
		expect(computeDelta([0.1, 0.9], [1])).toBeNull();
	});

	it('computes elementwise current - baseline', () => {
		const [a, b] = computeDelta([0.3, 0.7], [0.5, 0.5])!;
		expect(a).toBeCloseTo(0.2);
		expect(b).toBeCloseTo(-0.2);
	});
});

describe('dominantDeltaIndex', () => {
	it('returns null for null/empty delta', () => {
		expect(dominantDeltaIndex(null)).toBeNull();
		expect(dominantDeltaIndex([])).toBeNull();
	});

	it('returns null when every entry is below the threshold', () => {
		expect(dominantDeltaIndex([0.001, -0.002, 0.003])).toBeNull();
	});

	it('returns the index of the largest-magnitude entry', () => {
		expect(dominantDeltaIndex([0.02, -0.2, 0.05])).toBe(1);
	});

	it('respects a custom threshold', () => {
		expect(dominantDeltaIndex([0.02, -0.03], 0.05)).toBeNull();
		expect(dominantDeltaIndex([0.02, -0.03], 0.02)).toBe(1);
	});
});

describe('hasEvidenceDiverged', () => {
	it('false for two empty maps', () => {
		expect(hasEvidenceDiverged({}, {})).toBe(false);
	});

	it('false for identical maps', () => {
		expect(hasEvidenceDiverged({ weather: 'dry' }, { weather: 'dry' })).toBe(false);
	});

	it('true when a key differs in value', () => {
		expect(hasEvidenceDiverged({ weather: 'dry' }, { weather: 'wet' })).toBe(true);
	});

	it('true when key sets differ in size', () => {
		expect(hasEvidenceDiverged({ weather: 'dry' }, {})).toBe(true);
		expect(hasEvidenceDiverged({}, { weather: 'dry' })).toBe(true);
	});

	it('is order-independent', () => {
		expect(hasEvidenceDiverged({ a: '1', b: '2' }, { b: '2', a: '1' })).toBe(false);
	});
});
