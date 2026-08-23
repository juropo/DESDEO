<script lang="ts">
	/**
	 * `<NetworkPanel>` — thin wrapper that owns fetching, for the standalone case only (graphical
	 * models plan, frontend phase B). A method UI that already holds a compiled model's
	 * structure/evidence/posteriors in its own state (the Cumulus integration does, for now with
	 * demo data) should use `<NetworkView>` directly instead — this wrapper is for callers that
	 * just want to point at a compiled model id and get a working panel.
	 *
	 * The backend endpoints this calls (`GET /graphical/{id}/structure`, `POST /graphical/whatif`)
	 * are backend phase 7 — not built yet. `fetchStructure`/`fetchWhatIf` are therefore injectable,
	 * so this can be exercised (and unit-tested) against a fake today and pointed at the generated
	 * OpenAPI client once phase 7 ships without changing this component. The defaults hit the
	 * endpoint paths the plan commits to (via the `/api/...` proxy — see the webui's API-proxy
	 * pattern), so wiring up real fetching later is a prop removal, not a rewrite.
	 *
	 * Deliberately no debounce / in-flight cancellation / staleness dimming here — that UX (phase D)
	 * belongs layered above the fetch, not inside this wrapper's one job of owning it. Also
	 * deliberately no import from any method store (`stores/uiState` etc.) — loading/error state is
	 * local, which is what keeps this reusable across method UIs the same way the rest of this
	 * package is.
	 */
	import type { Snippet } from 'svelte';
	import NetworkView from './NetworkView.svelte';
	import type {
		DecisionMode,
		NetworkEvidence,
		NetworkModel,
		NetworkPolicy,
		NetworkPosteriors,
		WhatIfResult
	} from './types';
	import type { NetworkChromeContext } from './view-types';

	async function defaultFetchStructure(modelId: string | number): Promise<NetworkModel> {
		const res = await fetch(`/api/graphical/${modelId}/structure`);
		if (!res.ok) throw new Error(`Failed to fetch network structure (status ${res.status})`);
		return res.json();
	}

	async function defaultFetchWhatIf(
		modelId: string | number,
		evidence: NetworkEvidence
	): Promise<WhatIfResult> {
		const res = await fetch(`/api/graphical/whatif`, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ model_id: modelId, evidence })
		});
		if (!res.ok) throw new Error(`Failed to fetch what-if result (status ${res.status})`);
		return res.json();
	}

	interface Props {
		/** Id of a compiled diagram (Problem id, once backend phase 4 exists). */
		modelId: string | number;
		fetchStructure?: (modelId: string | number) => Promise<NetworkModel>;
		fetchWhatIf?: (modelId: string | number, evidence: NetworkEvidence) => Promise<WhatIfResult>;
		decisionMode?: DecisionMode;
		highlight?: Iterable<string> | null;
		nodeChrome?: Snippet<[NetworkChromeContext]>;
		compact?: boolean;
		class?: string;
		onNodeSelect?: (id: string | null) => void;
		onNodeHover?: (id: string | null) => void;
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
		onNodeHover
	}: Props = $props();

	let model = $state<NetworkModel | null>(null);
	let evidence = $state<NetworkEvidence>({});
	let posteriors = $state<NetworkPosteriors | undefined>(undefined);
	let policy = $state<NetworkPolicy | undefined>(undefined);
	let loading = $state(true);
	let error = $state<string | null>(null);

	// Bumped on every `loadStructure` call and captured per-call as `requestId`, so a response
	// that resolves after a *newer* call has already started (e.g. `modelId` changed twice in
	// quick succession, and the first model's fetch resolves last) is discarded instead of
	// clobbering the already-loaded, more current model.
	let structureRequestId = 0;

	async function loadStructure(id: string | number) {
		const requestId = ++structureRequestId;
		loading = true;
		error = null;
		// A new model id means A's evidence/posteriors/policy are meaningless for B -- without
		// this, a node id shared between models would briefly show a pinned evidence badge,
		// posterior bars, or a policy checkmark left over from the previous model.
		evidence = {};
		posteriors = undefined;
		policy = undefined;
		try {
			const result = await fetchStructure(id);
			if (requestId !== structureRequestId) return; // superseded by a newer `loadStructure` call
			model = result;
		} catch (e) {
			if (requestId !== structureRequestId) return;
			error = e instanceof Error ? e.message : 'Failed to load network structure';
		} finally {
			if (requestId === structureRequestId) loading = false;
		}
	}

	async function refreshWhatIf(id: string | number, currentEvidence: NetworkEvidence) {
		try {
			const result = await fetchWhatIf(id, currentEvidence);
			posteriors = result.posteriors;
			// Always take the response's own `policy` as authoritative, including when it's
			// `undefined` -- a what-if response that legitimately carries no policy for the current
			// evidence must clear a previously-solved one, not leave a stale checkmark showing.
			policy = result.policy;
		} catch (e) {
			error = e instanceof Error ? e.message : 'Failed to fetch what-if result';
		}
	}

	$effect(() => {
		void loadStructure(modelId);
	});

	function handleEvidenceChange(next: NetworkEvidence) {
		evidence = next;
		void refreshWhatIf(modelId, next);
	}
</script>

<div class="flex h-full min-h-0 flex-col {className}">
	{#if loading}
		<div class="text-muted-foreground flex flex-1 items-center justify-center text-sm">
			Loading network…
		</div>
	{:else if error}
		<div class="text-destructive flex flex-1 items-center justify-center text-sm">{error}</div>
	{:else if model}
		<NetworkView
			bind:model
			bind:evidence
			{posteriors}
			{policy}
			{decisionMode}
			{highlight}
			{compact}
			{nodeChrome}
			{onNodeSelect}
			{onNodeHover}
			onEvidenceChange={handleEvidenceChange}
		/>
	{/if}
</div>
