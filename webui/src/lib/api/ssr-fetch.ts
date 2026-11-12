/**
 * Wraps a `load` function's own `fetch` for passing to a generated API client call as its
 * `options` argument.
 *
 * Generated functions (`$lib/gen/endpoints/DESDEOFastAPI`) type `options` as plain `RequestInit`
 * (orval's default), but `customFetch` (`new-client.ts`) additionally reads a `fetchImpl` field
 * off it. Passing `withFetch(fetch)` from inside a `load` function is what makes
 * `hooks.server.ts`'s `handleFetch` (cookie forwarding + refresh-on-401) actually run for that
 * call's SSR request -- see `getUrl` in `new-client.ts`. Without it, a hard/direct navigation's
 * SSR call uses the bare global `fetch` (no cookies at all) and always 401s, even though the same
 * call succeeds client-side after hydration (where `getUrl` routes through the `/api` proxy with
 * real browser cookies).
 *
 * Centralized here — instead of `{ fetchImpl: fetch } as RequestInit` repeated at every call site
 * — so the one type-unsafe cast (`fetchImpl` isn't part of `RequestInit`'s real type) lives in one
 * place, and any future change to what an SSR call needs (extra headers, a different auth
 * mechanism) is a one-file edit instead of an N-call-site one where missing a site silently
 * reintroduces the SSR-401 bug this exists to prevent.
 */
export function withFetch(fetch: typeof globalThis.fetch): RequestInit {
	return { fetchImpl: fetch } as RequestInit;
}
