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
	import { defaultFetchPolicy } from './policy';
	import { defaultFetchEvidenceSets, defaultSaveEvidenceSet } from './evidence-sets';
	import type {
		DecisionMode,
		NetworkEvidence,
		NetworkEvidenceSet,
		NetworkModel,
		NetworkPolicy,
		NetworkPolicySource,
		NetworkPosteriors,
		NetworkUtilities,
		NetworkUtilityStdDevs,
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
	 * 2. `WhatIfResponse.utilities` (and, phase E, `utility_std_devs`) is keyed by the *raw
	 *    objective symbol*, but `get_structure` emits each objective node's id as `util_<symbol>`
	 *    (to keep it from colliding with a same-named chance node) -- so `NetworkUtilities`/
	 *    `NetworkUtilityStdDevs` (keyed by node id, like every other per-node map in this contract)
	 *    need that prefix re-applied before they line up with the node ids `defaultFetchStructure`
	 *    already returned.
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
			utility_std_devs?: NetworkUtilityStdDevs;
		} = await res.json();
		const withUtilPrefix = <T extends Record<string, number>>(map: T | undefined): T | undefined =>
			map
				? (Object.fromEntries(
						Object.entries(map).map(([symbol, value]) => [`util_${symbol}`, value])
					) as T)
				: undefined;
		return {
			posteriors: result.posteriors,
			policy: result.policy,
			utilities: withUtilPrefix(result.utilities),
			utilityStdDevs: withUtilPrefix(result.utility_std_devs)
		};
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
		/** A specific solved `Problem`'s policy (phase E, checklist item 1) -- when given, every
		 *  evidence-driven refresh also decodes `policy` from `POST /graphical/{id}/policy` against
		 *  this solution, *overriding* whatever `fetchWhatIf`'s own (live-engine) `policy` field
		 *  returned. `null`/absent (the default) leaves `policy` sourced from `fetchWhatIf` alone,
		 *  exactly as before this prop existed -- purely additive, no behaviour change for existing
		 *  callers. See `NetworkPolicySource`'s own doc comment for why these are two different,
		 *  complementary sources of truth for the same `NetworkPolicy` shape `<NetworkView>` renders
		 *  identically either way. */
		policySource?: NetworkPolicySource | null;
		fetchPolicy?: (
			modelId: string | number,
			source: NetworkPolicySource,
			evidence: NetworkEvidence,
			signal?: AbortSignal
		) => Promise<NetworkPolicy>;
		/** Named evidence sets (phase E, checklist item 3) -- default to the real
		 *  `GET/POST /graphical/{id}/evidence-sets` endpoints, the same "on by default, injectable
		 *  for a fake/the generated OpenAPI client" convention `fetchStructure`/`fetchWhatIf` already
		 *  follow. The save/load UI itself is unconditionally rendered -- not gated behind either
		 *  prop being given (a review pass caught this comment and an actual `{#if fetchEvidenceSets
		 *  && saveEvidenceSet}` guard both claiming otherwise after the guard was removed in favour
		 *  of these real defaults). Pass `fetchEvidenceSets={() => Promise.resolve([])}` to point a
		 *  specific embedding at a backend without these routes mounted without erroring, but there
		 *  is currently no prop-level way to hide the UI entirely -- passing `saveEvidenceSet=
		 *  {undefined}` does *not* do that, since an `undefined` prop just falls back to this
		 *  default like any other. */
		fetchEvidenceSets?: (modelId: string | number) => Promise<NetworkEvidenceSet[]>;
		saveEvidenceSet?: (
			modelId: string | number,
			name: string,
			evidence: NetworkEvidence
		) => Promise<NetworkEvidenceSet>;
		decisionMode?: DecisionMode;
		highlight?: Iterable<string> | null;
		nodeChrome?: Snippet<[NetworkChromeContext]>;
		compact?: boolean;
		class?: string;
		onNodeSelect?: (id: string | null) => void;
		onNodeHover?: (id: string | null) => void;
		/** Fired with every what-if response this panel receives -- `posteriors`/`policy` are
		 *  already rendered by the embedded `<NetworkView>`, but `utilities`/`utilityStdDevs`
		 *  (objective-node expected utility and its standard deviation, phase D/E) deliberately
		 *  aren't: objective nodes stay posterior-free per phase C's "purely structural" decision,
		 *  so this is the hook a host method UI uses to read them for its own objective-value
		 *  display instead. */
		onWhatIfResult?: (result: WhatIfResult) => void;
		/** Fired when the DM commits the current evidence -- see `NetworkView`'s own
		 *  `onCommitEvidence` doc comment for the exploratory-vs-committed contract this bubbles
		 *  up from. Also fired by `loadEvidenceSetAsBaseline` (phase E item 3): selecting a saved
		 *  evidence set as the comparison baseline goes through this identical mechanism, not a
		 *  parallel one. */
		onCommitEvidence?: (evidence: NetworkEvidence) => void;
		/** Fired once, with the freshly loaded `NetworkModel`, right after a successful structure
		 *  fetch (phase F: a host method UI wanting its own read of the model -- e.g. to build a
		 *  reference-point widget's own node list, or an adjacency map for cross-highlighting --
		 *  has no other way to get one, since `model` itself is this component's own internal
		 *  state, not a prop). Purely additive; does nothing if unset. */
		onStructureLoad?: (model: NetworkModel) => void;
		/** Fired when the structure fetch itself fails (`structureError`, the fatal case that
		 *  replaces the whole pane) -- distinct from `whatIfError` (non-fatal, no hook of its own,
		 *  see this component's own doc comment for why). A review pass caught a real failure mode
		 *  this closes: a cached `modelId` a host page holds onto across a login/logout or a DB
		 *  reset can 404 here for reasons that have nothing to do with the DM's own actions, and
		 *  without a way to observe *this specific* failure, a host page has no way to distinguish
		 *  "recover automatically" (drop the cache, refetch) from "surface it" the way it can for
		 *  `onWhatIfResult`'s success case. Purely additive; does nothing if unset. */
		onStructureError?: (error: string) => void;
	}

	let {
		modelId,
		fetchStructure = defaultFetchStructure,
		fetchWhatIf = defaultFetchWhatIf,
		policySource = null,
		fetchPolicy = defaultFetchPolicy,
		fetchEvidenceSets = defaultFetchEvidenceSets,
		saveEvidenceSet = defaultSaveEvidenceSet,
		decisionMode = 'solver',
		highlight = null,
		nodeChrome,
		compact = false,
		class: className = '',
		onNodeSelect,
		onNodeHover,
		onWhatIfResult,
		onCommitEvidence,
		onStructureLoad,
		onStructureError
	}: Props = $props();

	let model = $state<NetworkModel | null>(null);
	let evidence = $state<NetworkEvidence>({});
	let posteriors = $state<NetworkPosteriors | undefined>(undefined);
	let policy = $state<NetworkPolicy | undefined>(undefined);
	let utilities = $state<NetworkUtilities | undefined>(undefined);
	let utilityStdDevs = $state<NetworkUtilityStdDevs | undefined>(undefined);
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

	// Named evidence sets (phase E item 3) -- `fetchEvidenceSets`/`saveEvidenceSet` default to the
	// real endpoints (see the props' own doc comments), so this UI is on by default, not
	// conditional on either prop being given (a review pass caught the props' own doc comments and
	// a stale `{#if fetchEvidenceSets && saveEvidenceSet}` guard both claiming otherwise after the
	// guard was removed -- the doc comments were the one straggler, fixed here to match); loaded
	// once alongside structure, refreshed after a successful save.
	let evidenceSets = $state<NetworkEvidenceSet[]>([]);
	let evidenceSetsError = $state<string | null>(null);
	let newEvidenceSetName = $state('');
	let savingEvidenceSet = $state(false);
	// The baseline `<select>`'s own displayed value, `bind:value`d below so the dropdown can never
	// visually disagree with which baseline (a saved set's id, or `''` for "Prior (default)") is
	// actually active -- reset alongside every other commit-adjacent piece of state (`loadStructure`,
	// `handleCommitEvidence`) rather than left to drift once a saved set has been picked once.
	let selectedBaselineId = $state('');
	// Guards `loadEvidenceSetAsBaseline`'s own writes against a *newer* baseline selection
	// resolving out of order (a fast select-then-reselect) -- see that function's own comment.
	let baselineRequestId = 0;

	// Bumped on every `loadStructure` call and captured per-call as `requestId`, so a response
	// that resolves after a *newer* call has already started (e.g. `modelId` changed twice in
	// quick succession, and the first model's fetch resolves last) is discarded instead of
	// clobbering the already-loaded, more current model.
	let structureRequestId = 0;

	// Phase F finding 1: whether `policySource` was already non-null the *last* time the
	// policySource-transition effect below ran -- plain variable, not `$state`, since it's read-
	// modify-write bookkeeping for that effect alone, not something any markup renders off of.
	// Reset on every `loadStructure` call (a fresh model has no "already handled this transition"
	// history of its own).
	let hadPolicySource = false;

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
			utilityStdDevs = result.utilityStdDevs;
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
		utilityStdDevs = undefined;
		priorPosteriors = undefined;
		committedEvidence = {};
		committedPosteriors = undefined;
		evidenceSets = [];
		evidenceSetsError = null;
		newEvidenceSetName = '';
		selectedBaselineId = '';
		hadPolicySource = false;
		try {
			const result = await structureFetcher(id);
			if (requestId !== structureRequestId) return; // superseded by a newer `loadStructure` call
			model = result;
			onStructureLoad?.(result);
		} catch (e) {
			if (requestId !== structureRequestId) return;
			structureError = e instanceof Error ? e.message : 'Failed to load network structure';
			onStructureError?.(structureError);
			return;
		} finally {
			if (requestId === structureRequestId) loading = false;
		}
		// Named evidence sets (phase E item 3) -- fire-and-forget, same "degraded-but-safe absence"
		// treatment as the prior-baseline fetch just below: a listing failure shouldn't tear down an
		// already-rendered panel (or skip the rest of this function), just leave the save/load UI
		// showing nothing to select yet. Deliberately *not* awaited here -- this list is unrelated
		// to the prior-baseline fetch just below, which has no reason to sit behind it (a review
		// pass caught the two needlessly serialized, delaying the ghost overlay behind a fetch that
		// has nothing to do with it).
		void (async () => {
			try {
				const sets = await fetchEvidenceSets(id);
				if (requestId === structureRequestId) evidenceSets = sets;
			} catch (e) {
				if (requestId === structureRequestId) {
					evidenceSetsError = e instanceof Error ? e.message : 'Failed to load saved evidence sets';
				}
			}
		})();
		// Seed the ghost baseline with the unconditioned prior -- fire-and-forget relative to
		// `loadStructure` itself (the panel is already usable without it; the ghost overlay just
		// stays absent until this resolves), but still generation-guarded against a `modelId`
		// change racing it. `priorFetcher` is expected to already be `policySource`-aware (see the
		// effect below, which passes a `fetchWhatIfWithPolicy`-backed wrapper) -- so this is also
		// what seeds the *policy* badge without requiring the DM to touch evidence first (phase F
		// finding 1: a review pass caught `policy` only ever being written from the what-if
		// runner's own `onResult`, triggered by an evidence change -- opening the dialog straight
		// into `decisionMode: 'solver'` showed no policy checkmark at all until the DM pinned
		// something).
		try {
			const prior = await priorFetcher(id, {});
			if (requestId !== structureRequestId) return;
			priorPosteriors = prior.posteriors;
			// Also seed the *live* posteriors (and, phase F: policy) from the same (evidence-free)
			// response, so the first render shows the actual baseline distribution instead of the
			// equal-width "no posterior yet" placeholder every editable chance node would otherwise
			// fall back to. Only when no exploratory evidence has been set in the meantime -- a fast
			// click before this resolves must not let this stale, unconditioned response clobber a
			// newer, evidence-driven result that `handleEvidenceChange` may already be chasing.
			if (Object.keys(evidence).length === 0) {
				posteriors = prior.posteriors;
				policy = prior.policy;
			}
		} catch {
			// No prior baseline is a degraded-but-safe state (no ghost mark, no delta, no seeded
			// policy) -- not worth surfacing as the panel's own `structureError` over a structure
			// fetch that *did* succeed.
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
		//
		// The prior-baseline fetcher is `fetchWhatIfWithPolicy` (not the bare `fetchWhatIf`) --
		// phase F finding 1: when `policySource` is already set by the time this initial load runs,
		// this is what lets `loadStructure`'s own seeding (see its own comment) apply the solved
		// policy immediately, the same way every evidence-driven refresh already does.
		untrack(() => {
			void loadStructure(id, fetchStructure, (_id, evidenceForRequest, signal) =>
				fetchWhatIfWithPolicy(evidenceForRequest, signal)
			);
		});
	});

	// Phase F finding 1: `policySource` transitioning from unset to a real solved policy *after*
	// the initial load already ran -- in practice, the CUMULUS demo dialog's own `/solve` call
	// resolves after `/structure` does, so the seed above alone doesn't reliably cover this. A
	// review pass caught the policy badge otherwise only ever appearing once the DM made their own
	// evidence change. Fires once per such transition (guarded by `hadPolicySource`, reset on every
	// `loadStructure` call), re-running the identical evidence-driven fetch `handleEvidenceChange`
	// already uses, for whatever `evidence` currently is.
	$effect(() => {
		const hasSource = policySource !== null && policySource !== undefined;
		const currentModel = model;
		if (hasSource && !hadPolicySource && currentModel) {
			untrack(() => {
				handleEvidenceChange(evidence);
			});
		}
		hadPolicySource = hasSource;
	});

	/**
	 * `fetchWhatIf`, optionally overlaid with a *solved* policy (phase E item 1) -- when
	 * `policySource` is given, runs both requests together (so they share one debounce/abort/dim
	 * cycle via `whatIfRunner`, rather than a second, independently-timed fetch) and returns the
	 * what-if result with `policy` replaced by the solved-policy decode for the same `evidence`.
	 * This is what makes the displayed policy badge "update as the DM navigates": every evidence
	 * change re-runs this, so a newly-pinned-enough information set is decoded fresh each time.
	 *
	 * `signal` is optional (unlike `WhatIfRunner.request`'s own fetcher signature, which always
	 * supplies one) -- `loadStructure`'s prior-baseline fetch (phase F finding 1: routed through
	 * here too, so `policySource` is applied at initial load, not just on later evidence changes)
	 * calls `priorFetcher(id, {})` with no signal at all.
	 */
	async function fetchWhatIfWithPolicy(
		evidenceForRequest: NetworkEvidence,
		signal?: AbortSignal
	): Promise<WhatIfResult> {
		if (!policySource) return fetchWhatIf(modelId, evidenceForRequest, signal);
		const [result, solvedPolicy] = await Promise.all([
			fetchWhatIf(modelId, evidenceForRequest, signal),
			fetchPolicy(modelId, policySource, evidenceForRequest, signal)
		]);
		return { ...result, policy: solvedPolicy };
	}

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
		whatIfRunner.request((signal) => fetchWhatIfWithPolicy(chanceEvidence, signal));
	}

	function handleCommitEvidence(committed: NetworkEvidence) {
		committedEvidence = committed;
		// The current live `posteriors` *are* that committed evidence's what-if result (they were
		// fetched for exactly this `evidence`) -- no need to re-fetch just to promote them.
		// `<NetworkView>`'s own Commit button is disabled while `stale`, so `posteriors` is
		// guaranteed to already reflect `committed` by the time this fires.
		committedPosteriors = posteriors;
		// A live DM commit is a new, different baseline than whatever saved set (if any) was
		// previously selected -- clear the dropdown's own selection so it doesn't keep showing a
		// saved set's name once that set is no longer actually the active baseline.
		selectedBaselineId = '';
		onCommitEvidence?.(committed);
	}

	/** Save the current live `evidence` as a named set (phase E item 3). No-ops without a name, or
	 *  no *chance* evidence (nothing meaningful to name/reuse) -- see `chanceEvidenceToSave`'s own
	 *  comment for why this is `filterChanceEvidence(model, evidence)`, not raw `evidence`. */
	let chanceEvidenceToSave = $derived(model ? filterChanceEvidence(model, evidence) : {});
	async function handleSaveEvidenceSet() {
		if (!newEvidenceSetName.trim() || Object.keys(chanceEvidenceToSave).length === 0) return;
		savingEvidenceSet = true;
		evidenceSetsError = null;
		try {
			const saved = await saveEvidenceSet(modelId, newEvidenceSetName.trim(), chanceEvidenceToSave);
			evidenceSets = [saved, ...evidenceSets];
			newEvidenceSetName = '';
		} catch (e) {
			evidenceSetsError = e instanceof Error ? e.message : 'Failed to save evidence set';
		} finally {
			savingEvidenceSet = false;
		}
	}

	/**
	 * Loads a saved evidence set as the comparison baseline -- the "ghost" `NetworkView` already
	 * renders behind every posterior bar. Goes through the *identical* `committedEvidence`/
	 * `committedPosteriors`/`onCommitEvidence` mechanism `handleCommitEvidence` uses for a live
	 * commit, not a parallel one, per the plan's own phase E wording -- a saved set becomes the
	 * baseline the same way accepting the DM's own current evidence would. Deliberately doesn't
	 * touch the live, exploratory `evidence`/`posteriors` -- selecting a baseline to compare
	 * against isn't the same action as adopting it as what's currently being explored.
	 *
	 * Generation-guarded against two different races (a review pass caught this function missing
	 * both, unlike every other async write in this component): `requestId` mirrors `loadStructure`'s
	 * own pattern, so a response that resolves after a `modelId` change doesn't install the *old*
	 * model's evidence/posteriors as the *new* model's committed baseline; `ownRequestId` guards
	 * against a *newer* baseline selection resolving first (a fast select-then-reselect), so an
	 * older response can't win and overwrite a more current one.
	 */
	async function loadEvidenceSetAsBaseline(set: NetworkEvidenceSet) {
		if (!model) return;
		const requestId = structureRequestId;
		const ownRequestId = ++baselineRequestId;
		evidenceSetsError = null;
		// Defensive: the model may have been recompiled since this set was saved, and only chance
		// evidence is ever valid to post to `/whatif` -- same restriction `handleEvidenceChange`/
		// `handleSaveEvidenceSet` already enforce.
		const chanceEvidence = filterChanceEvidence(model, set.evidence);
		try {
			const result = await fetchWhatIf(modelId, chanceEvidence);
			if (requestId !== structureRequestId || ownRequestId !== baselineRequestId) return;
			committedEvidence = chanceEvidence;
			committedPosteriors = result.posteriors;
			onCommitEvidence?.(chanceEvidence);
		} catch (e) {
			if (requestId !== structureRequestId || ownRequestId !== baselineRequestId) return;
			evidenceSetsError =
				e instanceof Error ? e.message : 'Failed to load evidence set as baseline';
		}
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
			<!-- Named evidence sets (phase E item 3) -- `fetchEvidenceSets`/`saveEvidenceSet` default
			     to real endpoint calls (see those props' own doc comments), so this is on by default,
			     unconditionally rendered rather than gated behind an `{#if}` on either prop. -->
			<div class="flex flex-wrap items-center gap-2 text-xs">
				<label class="text-muted-foreground flex items-center gap-1">
					Baseline:
					<select
						class="border-border bg-background rounded-md border px-1.5 py-0.5"
						bind:value={selectedBaselineId}
						onchange={(e) => {
							const raw = (e.currentTarget as HTMLSelectElement).value;
							if (raw === '') {
								// "Prior (default)" -- back to the unconditioned prior, the same reset
								// `loadStructure` performs on a fresh model. `ghostPosteriors`'s own
								// derivation already falls back to `priorPosteriors` once
								// `committedPosteriors` is unset again.
								committedEvidence = {};
								committedPosteriors = undefined;
								return;
							}
							const set = evidenceSets.find((s) => String(s.id) === raw);
							if (set) void loadEvidenceSetAsBaseline(set);
						}}
					>
						<option value="">Prior (default)</option>
						{#each evidenceSets as set (set.id)}
							<option value={set.id}>{set.name}</option>
						{/each}
					</select>
				</label>
				<input
					class="border-border bg-background w-40 rounded-md border px-1.5 py-0.5"
					type="text"
					placeholder="Name this evidence set…"
					bind:value={newEvidenceSetName}
				/>
				<button
					type="button"
					class="border-border rounded-md border px-2 py-0.5 disabled:opacity-50"
					disabled={savingEvidenceSet ||
						!newEvidenceSetName.trim() ||
						Object.keys(chanceEvidenceToSave).length === 0}
					onclick={handleSaveEvidenceSet}
				>
					Save current evidence
				</button>
				{#if evidenceSetsError}
					<span class="text-destructive">{evidenceSetsError}</span>
				{/if}
			</div>
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
