import { get } from 'svelte/store';
import { describe, expect, it, vi } from 'vitest';
import { createNetworkHighlightStore } from './highlight-store';

describe('createNetworkHighlightStore', () => {
	it('starts out null', () => {
		const store = createNetworkHighlightStore();
		expect(get(store)).toBeNull();
	});

	it('set/subscribe works', () => {
		const store = createNetworkHighlightStore();
		store.set(new Set(['a', 'b']));
		expect(get(store)).toEqual(new Set(['a', 'b']));

		store.set(null);
		expect(get(store)).toBeNull();
	});

	it('independent instances do not share state', () => {
		const storeA = createNetworkHighlightStore();
		const storeB = createNetworkHighlightStore();

		storeA.set(new Set(['a']));

		expect(get(storeA)).toEqual(new Set(['a']));
		expect(get(storeB)).toBeNull();
	});

	it('independent instances do not share subscriptions', () => {
		const storeA = createNetworkHighlightStore();
		const storeB = createNetworkHighlightStore();
		const subscriberA = vi.fn();
		const subscriberB = vi.fn();
		storeA.subscribe(subscriberA);
		storeB.subscribe(subscriberB);

		subscriberA.mockClear();
		subscriberB.mockClear();

		storeA.set(new Set(['a']));

		expect(subscriberA).toHaveBeenCalledTimes(1);
		expect(subscriberA).toHaveBeenCalledWith(new Set(['a']));
		expect(subscriberB).not.toHaveBeenCalled();
	});

	it('stops notifying a subscriber once it has unsubscribed', () => {
		const store = createNetworkHighlightStore();
		const subscriber = vi.fn();
		const unsubscribe = store.subscribe(subscriber);

		subscriber.mockClear();
		unsubscribe();

		store.set(new Set(['a']));

		expect(subscriber).not.toHaveBeenCalled();
	});

	it('defensively copies on set, so mutating the input afterwards does not affect the store', () => {
		const store = createNetworkHighlightStore();
		const input = new Set(['a']);
		store.set(input);

		input.add('b');

		expect(get(store)).toEqual(new Set(['a']));
	});
});
