/**
 * Shared cross-highlight bus (graphical models plan, frontend phase C). A generic,
 * model-agnostic, node-id-keyed store: a method UI page creates one instance and passes it to
 * both `<NetworkView highlight={$store}>` and any other UI it owns (e.g. a reference-point
 * widget), so hovering either side highlights the other -- hover objective *f₂* in the
 * reference-point widget and its ancestor subgraph lights up on the canvas, or hover a chance
 * node and the objectives it can influence highlight in the widget.
 *
 * This file holds no ancestor/descendant computation itself -- that's `dag-algorithms.ts`'s job
 * (`ancestors`/`descendants`/`ancestralSet`). This store is just the shared piece of mutable
 * state other code writes into and `<NetworkView highlight={...}>` already knows how to read
 * (via `mergeHighlightSets` in `interaction.ts`).
 *
 * `createNetworkHighlightStore` is a factory, not a module-level singleton, so independent
 * diagram instances/pages/navigations never share stale highlight state -- the same reasoning
 * as why the Cumulus demo's `network_model`/`network_evidence` are page-local `$state`, not
 * global stores.
 *
 * **Immutability contract**: values are treated as immutable snapshots. Always replace the
 * current set via `set(...)` (or `update` returning a new set) -- never call `.add`/`.delete`
 * on the value you read from the store and expect that to propagate. Svelte's `$derived`
 * (and any other reference-equality-based reactivity) compares the *reference*, not contents,
 * so an in-place mutation is a silent no-op to every subscriber. To make misuse harmless rather
 * than just documented, `set`/`update` below defensively copy into a fresh `Set` before writing,
 * so even a caller that hands in a set it keeps mutating afterwards can't corrupt this store's
 * state out from under other subscribers.
 */

import { writable, type Writable } from 'svelte/store';

export type NetworkHighlightStore = Writable<Set<string> | null>;

/** Creates a new, independent cross-highlight store, initialized to `null` (no highlight
 *  active). Call once per embedded diagram/page -- do not share one instance across unrelated
 *  method UIs or navigations. See this module's doc comment re: the immutability contract. */
export function createNetworkHighlightStore(): NetworkHighlightStore {
	const store = writable<Set<string> | null>(null);
	return {
		subscribe: store.subscribe,
		set(value) {
			store.set(value == null ? null : new Set(value));
		},
		update(updater) {
			store.update((current) => {
				const next = updater(current);
				return next == null ? null : new Set(next);
			});
		}
	};
}
