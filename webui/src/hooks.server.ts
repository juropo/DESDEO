import type { Handle, HandleFetch, RequestEvent } from '@sveltejs/kit';
import { refreshAccessTokenRefreshPost } from '$lib/gen/endpoints/DESDEOFastAPI';
import { dev } from '$app/environment';

// De-duplicates concurrent token refreshes within a single request: `handleFetch` runs once per
// outgoing `fetch()` call from a `load` function, and several loaders now issue multiple calls per
// request (`Promise.all`, or several loads sharing one navigation). If the access token expires
// right as two of those calls go out, both 401 near-simultaneously and, without this, would each
// independently call `refreshAccessTokenRefreshPost` with the same refresh token -- if the backend
// rotates/invalidates a refresh token on use, the second concurrent refresh fails and that request
// stays 401 for a user who is actually still authenticated. Keyed on `event` (the same object
// instance is passed to every `handleFetch` call within one request), so concurrent calls await
// the same in-flight refresh instead of racing; a `WeakMap` means the entry is reclaimed once the
// request finishes without any manual cleanup.
const refreshPromises = new WeakMap<RequestEvent, Promise<string | null>>();

// const API = process.env.API_BASE_URL ?? '/';

// SvelteKit strips response headers from `fetch()` calls made inside `load` functions before
// serializing them for client hydration, unless explicitly allow-listed here -- otherwise reading
// `response.headers.get(...)` throws. `getBody` in `src/lib/api/new-client.ts` reads `content-type`
// on every generated-client response to pick a body parser, so it must be allow-listed. This only
// bites universal (`+page.ts`) loads that route through SvelteKit's own `fetch` (i.e. anything
// passing `fetchImpl: fetch` — see those loaders' comments) on a hard/direct navigation; a
// client-side navigation never serializes the response at all, which is why this was easy to miss.
export const handle: Handle = async ({ event, resolve }) =>
	resolve(event, {
		filterSerializedResponseHeaders: (name) => name.toLowerCase() === 'content-type'
	});

export const handleFetch: HandleFetch = async ({ event, request, fetch }) => {
	// Buffer the body upfront so we have concrete data instead of a
	// ReadableStream. This avoids Node/undici "expected non-null body source"
	// errors when reconstructing or cloning Requests with streaming bodies.
	const hasBody = !['GET', 'HEAD'].includes(request.method);
	const bodyBuffer = hasBody ? await request.arrayBuffer() : undefined;
	const originalHeaders = Object.fromEntries(request.headers.entries());

	// Build a fresh Request with the buffered body and a given cookie header.
	// ArrayBuffers are reusable (unlike ReadableStreams), so this can be
	// called multiple times for retries.
	const makeRequest = (cookieHeader: string) =>
		new Request(request.url, {
			method: request.method,
			headers: new Headers({
				...originalHeaders,
				cookie: cookieHeader
			}),
			body: bodyBuffer,
			// @ts-expect-error — duplex is required for streaming bodies in Node 18+
			duplex: 'half'
		});

	// Forward access_token cookie to all API requests
	const accessToken = event.cookies.get('access_token');
	const originalCookie = originalHeaders['cookie'] ?? '';

	let res = await fetch(makeRequest(accessToken ? `access_token=${accessToken}` : originalCookie));

	// No auth errors, pass through.
	if (res.status !== 401) return res;

	// 401 — try refreshing the access token ONCE (shared across concurrent 401s on this same
	// request, see `refreshPromises` above), then retry.
	const refreshToken = event.cookies.get('refresh_token');
	if (!refreshToken) return res;

	let refreshPromise = refreshPromises.get(event);
	if (!refreshPromise) {
		// Explicit return type: `data` on the success branch is typed `unknown` (the generated
		// client doesn't narrow it further), so without a target type to check each `return`
		// against, accessing `.access_token` more than once here confuses inference into losing
		// the narrowing the single access in the pre-de-duplication version of this code relied
		// on. Reading it into `newAccessToken` once, right after the narrowing check, is also
		// just plain clearer than re-deriving it at each use site.
		refreshPromise = (async (): Promise<string | null> => {
			try {
				const response_with_new_cookies = await refreshAccessTokenRefreshPost({
					fetchImpl: fetch,
					headers: {
						cookie: `refresh_token=${refreshToken}`
					}
				} as RequestInit);

				if (
					response_with_new_cookies.status != 200 ||
					!response_with_new_cookies.data?.access_token
				) {
					return null;
				}
				const newAccessToken = response_with_new_cookies.data.access_token;

				// Update the access_token cookie for the browser.
				event.cookies.set('access_token', newAccessToken, {
					httpOnly: true,
					secure: !dev,
					sameSite: 'lax',
					path: '/'
				});

				return newAccessToken;
			} catch (err) {
				console.error('[handleFetch] token refresh failed:', err);
				console.error('[handleFetch] cause:', (err as any)?.cause);
				return null; // degrade to 401, not 500
			}
		})();
		refreshPromises.set(event, refreshPromise);
	}

	const newAccessToken = await refreshPromise;
	if (!newAccessToken) return res;

	// Retry with the new access token, reuses the buffered body. Reads the full cookie jar
	// (rather than just `newAccessToken`) so a concurrent caller that lost the refresh race still
	// retries with the one refresh's cookie, not a stale one from before this request began.
	const cookieHeader = event.cookies
		.getAll()
		.map(({ name, value }) => `${name}=${value}`)
		.join('; ');
	res = await fetch(makeRequest(cookieHeader));

	return res;
};
