<script lang="ts">
	/**
	 * `<NetworkPanel>` — a `<NetworkView>` that fetches its own data from the graphical-model
	 * endpoints, given a compiled model id. Method UIs that already hold the model's state should
	 * use `<NetworkView>` directly.
	 *
	 * The fetch functions are injectable (for tests or another client). What-if requests are
	 * debounced, cancellable, and dim the canvas while stale (via `createWhatIfRunner`). State is
	 * kept local, with no method-store imports, so the panel stays reusable.
	 *
	 * A failed structure fetch (`structureError`) replaces the whole pane; a failed what-if refresh
	 * (`whatIfError`) is only a banner, so it doesn't discard the DM's evidence and layout.
	 */
	import { untrack, type Snippet } from 'svelte';
	import NetworkView from './NetworkView.svelte';
	import { createWhatIfRunner, EVIDENCE_DEBOUNCE_MS } from './whatif';
	import {
		assumableEvidence,
		decisionsBlockingUtilities,
		filterChanceEvidence,
		filterDecisionEvidence
	} from './interaction';
	import { defaultFetchPolicy } from './policy';
	import { defaultFetchEvidenceSets, defaultSaveEvidenceSet } from './evidence-sets';
	import type {
		DecisionMode,
		NetworkAssumptions,
		NetworkEvidence,
		NetworkEvidenceSet,
		NetworkModel,
		NetworkPolicy,
		NetworkPolicySource,
		NetworkPosteriors,
		NetworkSolutionRef,
		NetworkUtilities,
		NetworkUtilityStdDevs,
		WhatIfResult
	} from './types';
	import type { NetworkChromeContext } from './view-types';

	/** `StructureResponse` already matches `NetworkModel`, so no remapping is needed. */
	async function defaultFetchStructure(modelId: string | number): Promise<NetworkModel> {
		const res = await fetch(`/api/graphical/${modelId}/structure`);
		if (!res.ok) throw new Error(`Failed to fetch network structure (status ${res.status})`);
		return res.json() as Promise<NetworkModel>;
	}

	/**
	 * Calls `POST /graphical/whatif` and maps the response onto `WhatIfResult`. The backend keys
	 * `utilities`/`utility_std_devs` by objective symbol, while objective node ids are
	 * `util_<symbol>`, so the prefix is added here.
	 *
	 * `evidence` must already be filtered to chance nodes (`filterChanceEvidence`), and
	 * `decisionEvidence` to decision nodes (`filterDecisionEvidence`); the endpoint rejects
	 * anything else in either field.
	 * `solution`: a solved solution that unpinned decisions follow.
	 */
	async function defaultFetchWhatIf(
		modelId: string | number,
		evidence: NetworkEvidence,
		signal?: AbortSignal,
		decisionEvidence: NetworkEvidence = {},
		solution: NetworkSolutionRef | null = null
	): Promise<WhatIfResult> {
		const res = await fetch(`/api/graphical/whatif`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({
				problem_id: modelId,
				evidence,
				decision_evidence: decisionEvidence,
				solution: solution
					? { state_id: solution.stateId, solution_index: solution.solutionIndex }
					: null
			}),
			signal
		});
		if (!res.ok) {
			const detail: unknown = await res.json().then(
				(body: { detail?: unknown }) => body?.detail,
				() => undefined
			);
			const suffix = typeof detail === 'string' ? `: ${detail}` : '';
			throw new Error(`Failed to fetch what-if result (status ${res.status})${suffix}`);
		}
		const result = (await res.json()) as {
			posteriors: NetworkPosteriors;
			policy?: NetworkPolicy;
			utilities?: NetworkUtilities;
			utility_std_devs?: NetworkUtilityStdDevs;
			off_policy_decisions?: string[];
			decision_posteriors?: NetworkPosteriors;
		};
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
			utilityStdDevs: withUtilPrefix(result.utility_std_devs),
			offPolicyDecisions: result.off_policy_decisions,
			decisionPosteriors: result.decision_posteriors
		};
	}

	interface Props {
		/** Problem id of a compiled diagram. */
		modelId: string | number;
		fetchStructure?: (modelId: string | number) => Promise<NetworkModel>;
		fetchWhatIf?: (
			modelId: string | number,
			evidence: NetworkEvidence,
			signal?: AbortSignal,
			/** Pinned decision states (`decisionMode: 'manual'` only), filtered via
			 *  `filterDecisionEvidence`. */
			decisionEvidence?: NetworkEvidence,
			/** `policySolution` (`decisionMode: 'manual'` only). */
			solution?: NetworkSolutionRef | null
		) => Promise<WhatIfResult>;
		/** In `decisionMode: 'manual'`, a solved solution whose policy every unpinned decision
		 *  follows in what-if requests. Refetches when it changes. */
		policySolution?: NetworkSolutionRef | null;
		/** Current evidence, including pinned decisions in `decisionMode: 'manual'`. Bindable, e.g.
		 *  so the host can list the pins. Kept on mount, so a host can restore it after a remount;
		 *  reset when `modelId` changes. */
		evidence?: NetworkEvidence;
		/** A solved solution to decode the policy from (via `POST /graphical/{id}/policy`) on every
		 *  evidence change, overriding the live-inference policy from `fetchWhatIf`. */
		policySource?: NetworkPolicySource | null;
		fetchPolicy?: (
			modelId: string | number,
			source: NetworkPolicySource,
			evidence: NetworkEvidence,
			signal?: AbortSignal
		) => Promise<NetworkPolicy>;
		/** Named evidence sets; default to the `/graphical/{id}/evidence-sets` endpoints. The
		 *  save/load UI is always rendered. */
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
		/** Fired with every what-if response (including the initial prior), e.g. so the host can
		 *  list `utilities`. Fired with an empty result when `decisionsBlockingUtilities` skips
		 *  the request. */
		onWhatIfResult?: (result: WhatIfResult) => void;
		/** Fired when the DM commits evidence as the comparison baseline, or selects a saved
		 *  evidence set as the baseline. Local to the panel; nothing is persisted. */
		onCommitEvidence?: (evidence: NetworkEvidence) => void;
		/** Fired when the DM asks to explore the current evidence as a hypothetical problem. The
		 *  button is only shown when this is set. */
		onExploreAssumptions?: (assumptions: NetworkAssumptions) => void;
		/** Label of the explore button. */
		exploreLabel?: string;
		/** Disables the explore button while the host is creating the hypothetical. */
		exploring?: boolean;
		/** Why the last explore attempt failed, shown next to the button. */
		exploreError?: string | null;
		/** Passed to `<NetworkView>`: host content for a resizable sidebar, above the node table. */
		aside?: Snippet;
		asideAutoSaveId?: string;
		/** Passed to `<NetworkView>`; also hides the evidence-set and explore controls. */
		readonly?: boolean;
		/** Fired with the loaded `NetworkModel` after a successful structure fetch. */
		onStructureLoad?: (model: NetworkModel) => void;
		/** Fired when the structure fetch fails, e.g. so a host can drop a stale cached `modelId`. */
		onStructureError?: (error: string) => void;
	}

	let {
		modelId,
		fetchStructure = defaultFetchStructure,
		fetchWhatIf = defaultFetchWhatIf,
		policySolution = null,
		evidence = $bindable({}),
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
		onExploreAssumptions,
		exploreLabel = 'Explore these assumptions',
		exploring = false,
		exploreError = null,
		aside,
		asideAutoSaveId,
		readonly = false,
		onStructureLoad,
		onStructureError
	}: Props = $props();

	let model = $state<NetworkModel | null>(null);
	let posteriors = $state<NetworkPosteriors | undefined>(undefined);
	let policy = $state<NetworkPolicy | undefined>(undefined);
	let utilities = $state<NetworkUtilities | undefined>(undefined);
	let utilityStdDevs = $state<NetworkUtilityStdDevs | undefined>(undefined);
	let decisionPosteriors = $state<NetworkPosteriors | undefined>(undefined);
	let offPolicyDecisions = $state<string[] | undefined>(undefined);
	// The solution sent with what-if requests: only followed in 'manual' mode.
	let requestSolution = $derived(decisionMode === 'manual' ? policySolution : null);
	let loading = $state(true);
	// Fatal: replaces the whole pane.
	let structureError = $state<string | null>(null);
	// Non-fatal: a dismissible banner. Cleared on the next successful result or new request.
	let whatIfError = $state<string | null>(null);
	let stale = $state(false);

	// Ghost baseline: the prior until the DM commits evidence, then the committed posteriors.
	let priorPosteriors = $state<NetworkPosteriors | undefined>(undefined);
	let committedEvidence = $state<NetworkEvidence>({});
	let committedPosteriors = $state<NetworkPosteriors | undefined>(undefined);
	let ghostPosteriors = $derived(committedPosteriors ?? priorPosteriors);

	// Named evidence sets: loaded with the structure, refreshed after a save.
	let evidenceSets = $state<NetworkEvidenceSet[]>([]);
	let evidenceSetsError = $state<string | null>(null);
	let newEvidenceSetName = $state('');
	let savingEvidenceSet = $state(false);
	// The baseline dropdown's value: a saved set's id, or `''` for the prior.
	let selectedBaselineId = $state('');
	// Discards out-of-order responses in `loadEvidenceSetAsBaseline`.
	let baselineRequestId = 0;

	// Bumped on every `loadStructure` call, so responses for a superseded model are discarded.
	let structureRequestId = 0;
	// Whether a model has been loaded since mount; the first load keeps the given `evidence`.
	let hasLoadedStructure = false;
	// Discards out-of-order responses in `seedPrior`.
	let priorRequestId = 0;

	// Whether `policySource` was set the last time the effect below ran. Reset by `loadStructure`.
	let hadPolicySource = false;

	// `decisionMode` the last time the effect below ran, so a refetch does NOT fire on mount or an
	// unrelated re-render -- only on an actual later change to `decisionMode`.
	let previousDecisionMode = untrack(() => decisionMode);

	const whatIfRunner = createWhatIfRunner<WhatIfResult>(
		EVIDENCE_DEBOUNCE_MS,
		(result) => {
			posteriors = result.posteriors;
			// Also when `undefined`, so a stale policy is cleared.
			policy = result.policy;
			utilities = result.utilities;
			utilityStdDevs = result.utilityStdDevs;
			decisionPosteriors = result.decisionPosteriors;
			offPolicyDecisions = result.offPolicyDecisions;
			whatIfError = null;
			onWhatIfResult?.(result);
		},
		(e) => {
			// Aborted requests never get here. Clear the previous result, same as a skipped request.
			whatIfError = e instanceof Error ? e.message : 'Failed to fetch what-if result';
			posteriors = undefined;
			policy = undefined;
			utilities = undefined;
			utilityStdDevs = undefined;
			decisionPosteriors = undefined;
			offPolicyDecisions = undefined;
			onWhatIfResult?.({ posteriors: {} });
		},
		(next) => {
			stale = next;
		}
	);

	// Cancel pending what-if requests on unmount.
	$effect(() => {
		return () => whatIfRunner.cancel();
	});

	async function loadStructure(
		id: string | number,
		structureFetcher: (modelId: string | number) => Promise<NetworkModel>,
		priorFetcher: (evidence: NetworkEvidence) => Promise<WhatIfResult>
	) {
		const requestId = ++structureRequestId;
		// Cancel what-if requests for the previous model.
		whatIfRunner.cancel();
		stale = false;
		loading = true;
		structureError = null;
		whatIfError = null;
		// Reset all state tied to the previous model.
		const keepEvidence = !hasLoadedStructure;
		hasLoadedStructure = true;
		if (!keepEvidence) evidence = {};
		posteriors = undefined;
		policy = undefined;
		utilities = undefined;
		utilityStdDevs = undefined;
		decisionPosteriors = undefined;
		offPolicyDecisions = undefined;
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
		// Load saved evidence sets in parallel with the prior fetch below; failure is non-fatal.
		if (!readonly)
			void (async () => {
				try {
					const sets = await fetchEvidenceSets(id);
					if (requestId === structureRequestId) evidenceSets = sets;
				} catch (e) {
					if (requestId === structureRequestId) {
						evidenceSetsError =
							e instanceof Error ? e.message : 'Failed to load saved evidence sets';
					}
				}
			})();
		if (keepEvidence && Object.keys(evidence).length > 0) handleEvidenceChange(evidence);
		await seedPrior(requestId, priorFetcher);
	}

	/**
	 * Seeds the ghost baseline with the prior. `priorFetcher` is `policySource`/`policySolution`-
	 * aware, so this also shows the solved policy before the DM sets any evidence. Skipped when the
	 * request is known to fail (see `decisionsBlockingUtilities`).
	 */
	async function seedPrior(
		requestId: number,
		priorFetcher: (evidence: NetworkEvidence) => Promise<WhatIfResult>
	) {
		const ownRequestId = ++priorRequestId;
		if (!model || decisionsBlockingUtilities(model, {}, !!requestSolution).length > 0) {
			priorPosteriors = undefined;
			return;
		}
		try {
			const prior = await priorFetcher({});
			if (requestId !== structureRequestId || ownRequestId !== priorRequestId) return;
			priorPosteriors = prior.posteriors;
			// Also seed the live state, unless the DM has set evidence meanwhile.
			if (Object.keys(evidence).length === 0) {
				posteriors = prior.posteriors;
				policy = prior.policy;
				utilities = prior.utilities;
				utilityStdDevs = prior.utilityStdDevs;
				decisionPosteriors = prior.decisionPosteriors;
				offPolicyDecisions = prior.offPolicyDecisions;
				onWhatIfResult?.(prior);
			}
		} catch (e) {
			// Non-fatal: the panel just has no ghost baseline. Shown unless the DM has moved on.
			if (requestId !== structureRequestId || ownRequestId !== priorRequestId) return;
			if (Object.keys(evidence).length === 0) {
				whatIfError = e instanceof Error ? e.message : 'Failed to fetch what-if result';
			}
		}
	}

	$effect(() => {
		const id = modelId;
		// Only `modelId` is tracked: the fetcher props are read in `untrack`, so a parent passing a
		// new inline closure on re-render doesn't reload (and reset) the panel.
		untrack(() => {
			void loadStructure(id, fetchStructure, (evidenceForRequest) =>
				fetchWhatIfWithPolicy(evidenceForRequest)
			);
		});
	});

	// Refetch, including the prior, when the followed solution changes. Compared by value, since a
	// host may pass a fresh object for the same solution on re-render.
	const solutionKey = (solution: NetworkSolutionRef | null | undefined) =>
		solution ? `${solution.stateId}:${solution.solutionIndex}` : '';
	let previousSolutionKey = untrack(() => solutionKey(requestSolution));
	$effect(() => {
		const key = solutionKey(requestSolution);
		const currentModel = model;
		if (key !== previousSolutionKey && currentModel) {
			untrack(() => {
				handleEvidenceChange(evidence);
				void seedPrior(structureRequestId, (evidenceForRequest) =>
					fetchWhatIfWithPolicy(evidenceForRequest)
				);
			});
		}
		previousSolutionKey = key;
	});

	// Refetch when `policySource` becomes set after the initial load (e.g. a solve finishing after
	// the structure loaded), so the solved policy shows without an evidence change.
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

	// Refetch on a `decisionMode` switch, so stale pinned-mode utilities/posteriors don't linger
	// after switching to 'solver' (and vice versa): `handleEvidenceChange` sends an empty
	// `decision_evidence` in 'solver' mode, the current pins in 'manual' mode.
	$effect(() => {
		const mode = decisionMode;
		const currentModel = model;
		if (mode !== previousDecisionMode && currentModel) {
			untrack(() => {
				handleEvidenceChange(evidence);
			});
		}
		previousDecisionMode = mode;
	});

	/**
	 * `fetchWhatIf`, with `policy` replaced by the decoded solved policy when `policySource` is
	 * set. Both requests run together so they share one debounce/abort cycle.
	 */
	async function fetchWhatIfWithPolicy(
		evidenceForRequest: NetworkEvidence,
		signal?: AbortSignal,
		decisionEvidenceForRequest: NetworkEvidence = {}
	): Promise<WhatIfResult> {
		const request = () =>
			fetchWhatIf(modelId, evidenceForRequest, signal, decisionEvidenceForRequest, requestSolution);
		if (!policySource) return request();
		const [result, solvedPolicy] = await Promise.all([
			request(),
			fetchPolicy(modelId, policySource, evidenceForRequest, signal)
		]);
		return { ...result, policy: solvedPolicy };
	}

	function handleEvidenceChange(next: NetworkEvidence) {
		// `evidence` is already updated via `bind:evidence`.
		if (!model) return;
		whatIfError = null;
		// Only chance evidence is sent as `evidence`; decision-node picks (in 'manual' mode) go
		// through `decision_evidence` instead, so they're pinned rather than treated as observed.
		const chanceEvidence = filterChanceEvidence(model, next);
		const decisionEvidence = decisionMode === 'manual' ? filterDecisionEvidence(model, next) : {};
		if (decisionsBlockingUtilities(model, decisionEvidence, !!requestSolution).length > 0) {
			// Known to fail (501) until every decision is pinned, so skip it and clear the results.
			whatIfRunner.cancel();
			stale = false;
			posteriors = undefined;
			policy = undefined;
			utilities = undefined;
			utilityStdDevs = undefined;
			decisionPosteriors = undefined;
			offPolicyDecisions = undefined;
			onWhatIfResult?.({ posteriors: {} });
			return;
		}
		whatIfRunner.request((signal) =>
			fetchWhatIfWithPolicy(chanceEvidence, signal, decisionEvidence)
		);
	}

	function handleCommitEvidence(committed: NetworkEvidence) {
		committedEvidence = committed;
		// The live posteriors already match `committed`: Commit is disabled while stale.
		committedPosteriors = posteriors;
		// The baseline is no longer a saved set.
		selectedBaselineId = '';
		onCommitEvidence?.(committed);
	}

	let assumable = $derived(model ? assumableEvidence(model, evidence, decisionMode) : null);
	let assumedLabel = $derived.by(() => {
		const assumed = model?.assumptions;
		if (!assumed) return '';
		return Object.entries({ ...assumed.chance, ...assumed.decisions })
			.map(([id, state]) => `${id} = ${state}`)
			.join(', ');
	});
	let exploreBlockedReason = $derived.by(() => {
		if (!assumable) return 'The network is still loading.';
		if (assumable.conflicts.length > 0) {
			return `This problem already assumes ${assumable.conflicts.join(', ')}.`;
		}
		if (assumable.blocking.length > 0) {
			return (
				`Only decisions and chance nodes without parents can be assumed. Clear the evidence on ` +
				`${assumable.blocking.join(', ')} first.`
			);
		}
		const { chance, decisions } = assumable.assumptions;
		if (Object.keys(chance).length === 0 && Object.keys(decisions).length === 0) {
			return decisionMode === 'manual'
				? 'Set evidence on a chance node without parents, or pin a decision, to assume it.'
				: 'Set evidence on a chance node without parents to assume it.';
		}
		return null;
	});

	/** Save the current chance evidence as a named set. No-ops without a name or evidence. */
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
	 * Loads a saved evidence set as the comparison baseline, like a commit but without changing
	 * the live evidence. Responses are discarded if the model or the selection changed meanwhile.
	 */
	async function loadEvidenceSetAsBaseline(set: NetworkEvidenceSet) {
		if (!model) return;
		const requestId = structureRequestId;
		const ownRequestId = ++baselineRequestId;
		const loadModelId = modelId;
		evidenceSetsError = null;
		const chanceEvidence = filterChanceEvidence(model, set.evidence);
		// Saved sets only ever hold chance evidence (the backend rejects anything else on save);
		// any decision pinned live (in 'manual' mode) still applies on top of the loaded baseline.
		const decisionEvidence =
			decisionMode === 'manual' ? filterDecisionEvidence(model, evidence) : {};
		try {
			const result = await fetchWhatIf(
				loadModelId,
				chanceEvidence,
				undefined,
				decisionEvidence,
				requestSolution
			);
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
			<!-- Named evidence sets -->
			{#if !readonly}
				<div class="flex flex-wrap items-center gap-2 text-xs">
					<label class="text-muted-foreground flex items-center gap-1">
						Baseline:
						<select
							class="border-border bg-background rounded-md border px-1.5 py-0.5"
							bind:value={selectedBaselineId}
							onchange={(e) => {
								const raw = (e.currentTarget as HTMLSelectElement).value;
								if (raw === '') {
									// Back to the prior as the baseline.
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
			{/if}
			{#if assumedLabel || onExploreAssumptions}
				<div class="flex flex-wrap items-center gap-2 text-xs">
					{#if assumedLabel}
						<span class="text-muted-foreground">Assumes: {assumedLabel}</span>
					{/if}
					{#if onExploreAssumptions && !readonly}
						<button
							type="button"
							class="border-border rounded-md border px-2 py-0.5 disabled:opacity-50"
							disabled={exploring || exploreBlockedReason !== null}
							title={exploreBlockedReason ??
								'Creates a new problem under these assumptions and switches to it.'}
							onclick={() => assumable && onExploreAssumptions(assumable.assumptions)}
						>
							{exploring ? 'Creating…' : exploreLabel}
						</button>
						{#if exploreError}
							<span class="text-destructive">{exploreError}</span>
						{/if}
					{/if}
				</div>
			{/if}
			<!-- Keyed on `modelId` so hover/selection and node positions reset with the model. -->
			{#key modelId}
				<NetworkView
					bind:model
					bind:evidence
					{posteriors}
					{ghostPosteriors}
					{committedEvidence}
					{policy}
					{utilities}
					{utilityStdDevs}
					{decisionPosteriors}
					{offPolicyDecisions}
					{stale}
					{decisionMode}
					{highlight}
					{compact}
					{nodeChrome}
					{onNodeSelect}
					{onNodeHover}
					onEvidenceChange={handleEvidenceChange}
					onCommitEvidence={handleCommitEvidence}
					{aside}
					{asideAutoSaveId}
					{readonly}
				/>
			{/key}
		</div>
	{/if}
</div>
