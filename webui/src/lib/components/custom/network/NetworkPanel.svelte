<script lang="ts">
	/**
	 * `<NetworkPanel>` — thin wrapper that owns fetching, for the standalone case only (graphical
	 * models plan, frontend phase B). A method UI that already holds a compiled model's
	 * structure/evidence/posteriors in its own state (the Cumulus integration does, for now with
	 * demo data — see the Cumulus demo dialog's own debounced mock, phase D) should use
	 * `<NetworkView>` directly instead — this wrapper is for callers that just want to point at a
	 * compiled model id and get a working panel.
	 *
	 * The backend endpoints this calls (`GET /graphical/{id}/structure`, `POST /graphical/whatif`)
	 * are backend phase 7, now built — see `desdeo/api/routers/graphical.py`'s `StructureResponse`/
	 * `WhatIfRequest`/`WhatIfResponse`. `fetchStructure`/`fetchWhatIf` stay injectable regardless
	 * (unit-testable against a fake, swappable for the generated OpenAPI client later without
	 * changing this component), but the defaults below are now written against the real, shipped
	 * request/response shapes rather than the placeholder ones phase B guessed at — see
	 * `defaultFetchWhatIf`'s own comment for what changed and why.
	 *
	 * Debounce (~150ms) / in-flight cancellation / whole-canvas staleness dimming (phase D) live
	 * here, via `createWhatIfRunner` (`whatif.ts`) — this is the component that actually owns the
	 * fetch a debounce/cancel/dim policy needs to wrap, per the plan's own phase B note that this
	 * UX "belongs layered above the fetch, not inside this wrapper's one job of owning it" (true of
	 * phase B's *bare* fetch; phase D is exactly that layer, added here rather than invented a new
	 * wrapper around this one). Also deliberately no import from any method store (`stores/
	 * uiState` etc.) — loading/error state is local, which is what keeps this reusable across
	 * method UIs the same way the rest of this package is.
	 *
	 * A failed structure fetch (`structureError`) and a failed what-if refresh (`whatIfError`) are
	 * kept as two separate pieces of state, not one shared `error` — a transient what-if failure
	 * (a 500, or a 400 from evidence the backend rejects) must not tear down an already-rendered
	 * `<NetworkView>` and everything the DM has built up in it (evidence, layout, commit state);
	 * only a structure-fetch failure is fatal enough to replace the whole pane. See the code-review
	 * pass in the plan's phase D findings for the bug this fixes.
	 */
	import { untrack, type Snippet } from 'svelte';
	import NetworkView from './NetworkView.svelte';
	import { createWhatIfRunner, EVIDENCE_DEBOUNCE_MS } from './whatif';
	import { filterChanceEvidence } from './interaction';
	import type {
		DecisionMode,
		NetworkEvidence,
		NetworkModel,
		NetworkPolicy,
		NetworkPosteriors,
		NetworkUtilities,
		WhatIfResult
	} from './types';
	import type { NetworkChromeContext } from './view-types';

	/** Field-for-field identical to `StructureResponse` (`graphical.py`) -- `nodes[].{id,label,
	 *  kind,states}` and `edges[].{id,source,target}` both already match `NetworkModel`'s own
	 *  shape, so unlike `defaultFetchWhatIf` below, no remapping is needed here. */
	async function defaultFetchStructure(modelId: string | number): Promise<NetworkModel> {
		const res = await fetch(`/api/graphical/${modelId}/structure`);
		if (!res.ok) throw new Error(`Failed to fetch network structure (status ${res.status})`);
		return res.json();
	}

	/**
	 * Two real mismatches against `WhatIfRequest`/`WhatIfResponse` (`graphical.py`), fixed here
	 * rather than left for a future pass (phase 7's own findings flagged this as "phase C/D's job"):
	 *
	 * 1. The request body key is `problem_id`, not `model_id` -- `modelId` here *is* a `Problem`
	 *    id (see the `modelId` prop's own doc comment), so this was always a naming bug, not a
	 *    deliberate abstraction.
	 * 2. `WhatIfResponse.utilities` is keyed by the *raw objective symbol*, but `get_structure`
	 *    emits each objective node's id as `util_<symbol>` (to keep it from colliding with a
	 *    same-named chance node) -- so `NetworkUtilities` (keyed by node id, like every other
	 *    per-node map in this contract) needs that prefix re-applied before it lines up with the
	 *    node ids `defaultFetchStructure` already returned.
	 *
	 * `signal` (phase D: in-flight cancellation) is threaded straight through to `fetch` -- a
	 * caller-supplied `fetchWhatIf` is free to ignore it, which is exactly why `createWhatIfRunner`
	 * *also* generation-guards every response rather than relying on abort alone.
	 *
	 * Callers of this function (`loadStructure`'s prior-baseline fetch, `handleEvidenceChange`) are
	 * responsible for restricting `evidence` to chance-node ids first (`filterChanceEvidence`) --
	 * the real endpoint 400s on any other node (`graphical.py` ~632-641), matching the plan's own
	 * phase E note that what-if is restricted to chance nodes.
	 */
	async function defaultFetchWhatIf(
		modelId: string | number,
		evidence: NetworkEvidence,
		signal?: AbortSignal
	): Promise<WhatIfResult> {
		const res = await fetch(`/api/graphical/whatif`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ problem_id: modelId, evidence }),
			signal
		});
		if (!res.ok) throw new Error(`Failed to fetch what-if result (status ${res.status})`);
		const result: {
			posteriors: NetworkPosteriors;
			policy?: NetworkPolicy;
			utilities?: NetworkUtilities;
		} = await res.json();
		const utilities: NetworkUtilities | undefined = result.utilities
			? Object.fromEntries(
					Object.entries(result.utilities).map(([symbol, value]) => [`util_${symbol}`, value])
				)
			: undefined;
		return { posteriors: result.posteriors, policy: result.policy, utilities };
	}

	interface Props {
		/** Id of a compiled diagram (Problem id, once backend phase 4 exists). */
		modelId: string | number;
		fetchStructure?: (modelId: string | number) => Promise<NetworkModel>;
		fetchWhatIf?: (
			modelId: string | number,
			evidence: NetworkEvidence,
			signal?: AbortSignal
		) => Promise<WhatIfResult>;
		decisionMode?: DecisionMode;
		highlight?: Iterable<string> | null;
		nodeChrome?: Snippet<[NetworkChromeContext]>;
		compact?: boolean;
		class?: string;
		onNodeSelect?: (id: string | null) => void;
		onNodeHover?: (id: string | null) => void;
		/** Fired with every what-if response this panel receives -- `posteriors`/`policy` are
		 *  already rendered by the embedded `<NetworkView>`, but `utilities` (objective-node
		 *  expected utility, phase D) deliberately isn't: objective nodes stay posterior-free per
		 *  phase C's "purely structural" decision, so this is the hook a host method UI uses to
		 *  read utilities for its own objective-value display instead. */
		onWhatIfResult?: (result: WhatIfResult) => void;
		/** Fired when the DM commits the current evidence -- see `NetworkView`'s own
		 *  `onCommitEvidence` doc comment for the exploratory-vs-committed contract this bubbles
		 *  up from. */
		onCommitEvidence?: (evidence: NetworkEvidence) => void;
	}

	let {
		modelId,
		fetchStructure = defaultFetchStructure,
		fetchWhatIf = defaultFetchWhatIf,
		decisionMode = 'solver',
		highlight = null,
		nodeChrome,
		compact = false,
		class: className = '',
		onNodeSelect,
		onNodeHover,
		onWhatIfResult,
		onCommitEvidence
	}: Props = $props();

	let model = $state<NetworkModel | null>(null);
	let evidence = $state<NetworkEvidence>({});
	let posteriors = $state<NetworkPosteriors | undefined>(undefined);
	let policy = $state<NetworkPolicy | undefined>(undefined);
	let utilities = $state<NetworkUtilities | undefined>(undefined);
	let loading = $state(true);
	// Fatal -- replaces the whole pane below, the same as this component's single `error` used to.
	let structureError = $state<string | null>(null);
	// Non-fatal -- shown as a dismissible banner above a still-rendered `<NetworkView>`. Cleared on
	// the next successful what-if result, and when a new evidence-driven request starts (a stale
	// error banner sitting on top of a fresh "Updating…" dim reads as worse than either alone).
	let whatIfError = $state<string | null>(null);
	let stale = $state(false);

	// Ghost baseline (phase D item 1): the unconditioned prior, fetched once right after structure
	// loads, until the DM commits some evidence -- at which point `committedPosteriors` (that
	// evidence's own what-if result) takes over as the baseline instead. "Prior (or committed
	// scenario)", per the plan, literally.
	let priorPosteriors = $state<NetworkPosteriors | undefined>(undefined);
	let committedEvidence = $state<NetworkEvidence>({});
	let committedPosteriors = $state<NetworkPosteriors | undefined>(undefined);
	let ghostPosteriors = $derived(committedPosteriors ?? priorPosteriors);

	// Bumped on every `loadStructure` call and captured per-call as `requestId`, so a response
	// that resolves after a *newer* call has already started (e.g. `modelId` changed twice in
	// quick succession, and the first model's fetch resolves last) is discarded instead of
	// clobbering the already-loaded, more current model.
	let structureRequestId = 0;

	// See `whatif.ts`'s own doc comment -- debounces evidence-driven what-if refreshes, aborts a
	// superseded in-flight request, and generation-guards the result besides (belt-and-suspenders
	// for a `fetchWhatIf` override that ignores its `signal`). Declared once (not `$state`) since
	// its identity never needs to change -- only the `$state` variables its callbacks close over do.
	const whatIfRunner = createWhatIfRunner<WhatIfResult>(
		EVIDENCE_DEBOUNCE_MS,
		(result) => {
			posteriors = result.posteriors;
			// Always take the response's own `policy` as authoritative, including when it's
			// `undefined` -- a what-if response that legitimately carries no policy for the current
			// evidence must clear a previously-solved one, not leave a stale checkmark showing.
			policy = result.policy;
			utilities = result.utilities;
			whatIfError = null;
			onWhatIfResult?.(result);
		},
		(e) => {
			whatIfError = e instanceof Error ? e.message : 'Failed to fetch what-if result';
		},
		(next) => {
			stale = next;
		}
	);

	// A pending debounce timer or in-flight what-if fetch must not outlive this component -- without
	// this, navigating away mid-request still eventually calls `onResult`/`onWhatIfResult` (and
	// writes to `$state` that no longer has any rendered `<NetworkView>` to reflect it in) well after
	// unmount. Registered once; the effect itself has no reactive dependencies, so its cleanup only
	// ever runs on unmount, not on every render.
	$effect(() => {
		return () => whatIfRunner.cancel();
	});

	async function loadStructure(
		id: string | number,
		structureFetcher: (modelId: string | number) => Promise<NetworkModel>,
		priorFetcher: (
			modelId: string | number,
			evidence: NetworkEvidence,
			signal?: AbortSignal
		) => Promise<WhatIfResult>
	) {
		const requestId = ++structureRequestId;
		// A `modelId` change means any what-if refresh still chasing the *previous* model must not
		// be allowed to land here -- cancel it outright rather than merely let it lose the
		// generation race, since its `stale` flip could otherwise briefly resurrect the dimming
		// overlay for a model the DM has already navigated away from.
		whatIfRunner.cancel();
		stale = false;
		loading = true;
		structureError = null;
		whatIfError = null;
		// A new model id means A's evidence/posteriors/policy/utilities/ghost-baseline are
		// meaningless for B -- without this, a node id shared between models would briefly show a
		// pinned evidence badge, posterior bars, or a policy checkmark left over from the previous
		// model.
		evidence = {};
		posteriors = undefined;
		policy = undefined;
		utilities = undefined;
		priorPosteriors = undefined;
		committedEvidence = {};
		committedPosteriors = undefined;
		try {
			const result = await structureFetcher(id);
			if (requestId !== structureRequestId) return; // superseded by a newer `loadStructure` call
			model = result;
		} catch (e) {
			if (requestId !== structureRequestId) return;
			structureError = e instanceof Error ? e.message : 'Failed to load network structure';
			return;
		} finally {
			if (requestId === structureRequestId) loading = false;
		}
		// Seed the ghost baseline with the unconditioned prior -- fire-and-forget relative to
		// `loadStructure` itself (the panel is already usable without it; the ghost overlay just
		// stays absent until this resolves), but still generation-guarded against a `modelId`
		// change racing it.
		try {
			const prior = await priorFetcher(id, {});
			if (requestId !== structureRequestId) return;
			priorPosteriors = prior.posteriors;
			// Also seed the *live* posteriors from the same (evidence-free) response, so the first
			// render shows the actual baseline distribution instead of the equal-width "no posterior
			// yet" placeholder every editable chance node would otherwise fall back to. Only when no
			// exploratory evidence has been set in the meantime -- a fast click before this resolves
			// must not let this stale, unconditioned response clobber a newer, evidence-driven result
			// that `handleEvidenceChange` may already be chasing.
			if (Object.keys(evidence).length === 0) posteriors = prior.posteriors;
		} catch {
			// No prior baseline is a degraded-but-safe state (no ghost mark, no delta) -- not
			// worth surfacing as the panel's own `structureError` over a structure fetch that *did*
			// succeed.
		}
	}

	$effect(() => {
		const id = modelId;
		// `fetchStructure`/`fetchWhatIf` are read *inside* `untrack`, then passed into
		// `loadStructure` as plain arguments, rather than left for `loadStructure` to read the
		// reactive props itself -- reading them at this effect's own (tracked) top level would make
		// a fresh closure identity on every unrelated parent re-render (a very natural call pattern,
		// e.g. `fetchStructure={(id) => client.structure(id, token)}`) a tracked dependency of this
		// effect, retriggering `loadStructure` -- and with it wiping evidence/posteriors/policy/
		// utilities/commit state -- on every such render, not just an actual `modelId` change.
		untrack(() => {
			void loadStructure(id, fetchStructure, fetchWhatIf);
		});
	});

	function handleEvidenceChange(next: NetworkEvidence) {
		// `evidence` itself is already kept in sync via `bind:evidence` below -- `NetworkView`
		// assigns its own bindable `evidence` before calling `onEvidenceChange`, so reassigning it
		// again here would just be a second, redundant write down the same channel.
		if (!model) return;
		whatIfError = null;
		// Only chance-node evidence is ever sent to the real backend -- see `defaultFetchWhatIf`'s
		// own doc comment and `filterChanceEvidence`'s. A decision-node pick made while exploring
		// `decisionMode: 'manual'` stays local-only (still shown as a pin on the node itself) rather
		// than being posted and 400ing the whole request.
		const chanceEvidence = filterChanceEvidence(model, next);
		whatIfRunner.request((signal) => fetchWhatIf(modelId, chanceEvidence, signal));
	}

	function handleCommitEvidence(committed: NetworkEvidence) {
		committedEvidence = committed;
		// The current live `posteriors` *are* that committed evidence's what-if result (they were
		// fetched for exactly this `evidence`) -- no need to re-fetch just to promote them.
		// `<NetworkView>`'s own Commit button is disabled while `stale`, so `posteriors` is
		// guaranteed to already reflect `committed` by the time this fires.
		committedPosteriors = posteriors;
		onCommitEvidence?.(committed);
	}
</script>

<div class="flex h-full min-h-0 flex-col {className}">
	{#if loading}
		<div class="text-muted-foreground flex flex-1 items-center justify-center text-sm">
			Loading network…
		</div>
	{:else if structureError}
		<div class="text-destructive flex flex-1 items-center justify-center text-sm">
			{structureError}
		</div>
	{:else if model}
		<div class="flex h-full min-h-0 flex-col gap-2">
			{#if whatIfError}
				<div
					class="border-destructive/40 bg-destructive/10 text-destructive flex items-center justify-between gap-2 rounded-md border px-2 py-1 text-xs"
				>
					<span>{whatIfError}</span>
					<button type="button" class="underline" onclick={() => (whatIfError = null)}>
						Dismiss
					</button>
				</div>
			{/if}
			<!-- Keyed on `modelId` -- without this, hover/select ids and dragged node positions from
			     a previous model can bleed onto a new one that happens to share node ids, undercutting
			     the "everything else is reset on model switch" guarantee `loadStructure` otherwise
			     makes for evidence/posteriors/policy/utilities/commit state. -->
			{#key modelId}
				<NetworkView
					bind:model
					bind:evidence
					{posteriors}
					{ghostPosteriors}
					{committedEvidence}
					{policy}
					{stale}
					{decisionMode}
					{highlight}
					{compact}
					{nodeChrome}
					{onNodeSelect}
					{onNodeHover}
					onEvidenceChange={handleEvidenceChange}
					onCommitEvidence={handleCommitEvidence}
				/>
			{/key}
		</div>
	{/if}
</div>
