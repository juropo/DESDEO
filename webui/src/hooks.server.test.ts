import { beforeEach, describe, expect, it, vi } from 'vitest';
import { handleFetch } from './hooks.server';
import { refreshAccessTokenRefreshPost } from '$lib/gen/endpoints/DESDEOFastAPI';

vi.mock('$lib/gen/endpoints/DESDEOFastAPI', () => ({
	refreshAccessTokenRefreshPost: vi.fn()
}));

vi.mock('$app/environment', () => ({
	dev: true
}));

const createCookieStore = (initial: Record<string, string>) => {
	const store = new Map(Object.entries(initial));

	return {
		get: (name: string) => store.get(name),
		getAll: () => Array.from(store.entries()).map(([name, value]) => ({ name, value })),
		set: vi.fn((name: string, value: string) => {
			store.set(name, value);
		}),
		/** Snapshot of the store for assertions. */
		_dump: () => Object.fromEntries(store)
	};
};

type HookArgs = Parameters<typeof handleFetch>[0];

const callHook = (opts: {
	request: Request;
	cookies: ReturnType<typeof createCookieStore>;
	fetchMock: ReturnType<typeof vi.fn>;
	/** Pass the same `event` across calls to simulate concurrent fetches within one request --
	 *  SvelteKit invokes `handleFetch` with the same `event` instance for every fetch a `load`
	 *  function issues. Defaults to a fresh object per call, matching every other test here where
	 *  each call is its own independent request. */
	event?: HookArgs['event'];
}) =>
	handleFetch({
		event: opts.event ?? ({ cookies: opts.cookies } as unknown as HookArgs['event']),
		request: opts.request,
		fetch: opts.fetchMock
	} as HookArgs);

const mockRefreshSuccess = (newToken = 'new-access-token') => {
	vi.mocked(refreshAccessTokenRefreshPost).mockResolvedValue({
		status: 200,
		data: { access_token: newToken },
		headers: new Headers()
	});
};

const mockRefreshFailure = (status = 401) => {
	vi.mocked(refreshAccessTokenRefreshPost).mockResolvedValue({
		status,
		data: null,
		headers: new Headers()
	} as unknown as Awaited<ReturnType<typeof refreshAccessTokenRefreshPost>>);
};

const ok200 = (body = '{"ok":true}') =>
	new Response(body, {
		status: 200,
		headers: { 'content-type': 'application/json' }
	});

const unauthorized401 = () => new Response(null, { status: 401 });

// Tests

describe('handleFetch', () => {
	beforeEach(() => {
		vi.clearAllMocks();
	});

	// Basic pass-through (no 401)

	describe('when the initial request succeeds', () => {
		it('forwards access_token cookie on a GET request', async () => {
			const cookies = createCookieStore({ access_token: 'tok-123' });
			const fetchMock = vi.fn().mockResolvedValueOnce(ok200());

			const request = new Request('http://example.test/api/resource');
			const response = await callHook({ request, cookies, fetchMock });

			expect(response.status).toBe(200);
			expect(fetchMock).toHaveBeenCalledTimes(1);

			const sentRequest = fetchMock.mock.calls[0][0] as Request;
			expect(sentRequest.headers.get('cookie')).toBe('access_token=tok-123');
		});

		it('forwards access_token cookie on a POST request and preserves the body', async () => {
			const body = JSON.stringify({ problem_id: 42 });
			const cookies = createCookieStore({ access_token: 'tok-456' });
			const fetchMock = vi.fn().mockResolvedValueOnce(ok200());

			const request = new Request('http://example.test/api/method/nimbus/solve', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body
			});
			const response = await callHook({ request, cookies, fetchMock });

			expect(response.status).toBe(200);
			expect(fetchMock).toHaveBeenCalledTimes(1);

			const sentRequest = fetchMock.mock.calls[0][0] as Request;
			expect(sentRequest.headers.get('cookie')).toBe('access_token=tok-456');
			const sentBody = await sentRequest.text();
			expect(sentBody).toBe(body);
		});

		it('sends request with original cookie header when no access_token exists', async () => {
			const cookies = createCookieStore({});
			const fetchMock = vi.fn().mockResolvedValueOnce(ok200());

			const request = new Request('http://example.test/api/resource');
			const response = await callHook({ request, cookies, fetchMock });

			expect(response.status).toBe(200);
			expect(fetchMock).toHaveBeenCalledTimes(1);
		});
	});

	// 401 without refresh token

	describe('when the initial request returns 401 and no refresh token exists', () => {
		it('returns the 401 response without attempting refresh', async () => {
			const cookies = createCookieStore({});
			const fetchMock = vi.fn().mockResolvedValueOnce(unauthorized401());

			const request = new Request('http://example.test/api/resource');
			const response = await callHook({ request, cookies, fetchMock });

			expect(response.status).toBe(401);
			expect(refreshAccessTokenRefreshPost).not.toHaveBeenCalled();
			expect(fetchMock).toHaveBeenCalledTimes(1);
		});
	});

	// 401 -> refresh -> retry (GET)

	describe('when a GET request returns 401 and refresh succeeds', () => {
		it('retries with the new access token and returns 200', async () => {
			const cookies = createCookieStore({ refresh_token: 'refresh-abc' });
			const fetchMock = vi
				.fn()
				.mockResolvedValueOnce(unauthorized401())
				.mockResolvedValueOnce(ok200());

			mockRefreshSuccess('fresh-token');

			const request = new Request('http://example.test/api/resource');
			const response = await callHook({ request, cookies, fetchMock });

			expect(response.status).toBe(200);
			expect(fetchMock).toHaveBeenCalledTimes(2);

			// Refresh was called with the refresh_token cookie
			expect(refreshAccessTokenRefreshPost).toHaveBeenCalledWith(
				expect.objectContaining({
					headers: { cookie: 'refresh_token=refresh-abc' }
				})
			);
		});

		it('updates the access_token in the cookie store', async () => {
			const cookies = createCookieStore({ refresh_token: 'refresh-abc' });
			const fetchMock = vi
				.fn()
				.mockResolvedValueOnce(unauthorized401())
				.mockResolvedValueOnce(ok200());

			mockRefreshSuccess('fresh-token');

			const request = new Request('http://example.test/api/resource');
			await callHook({ request, cookies, fetchMock });

			expect(cookies.set).toHaveBeenCalledWith(
				'access_token',
				'fresh-token',
				expect.objectContaining({
					httpOnly: true,
					path: '/',
					sameSite: 'lax'
				})
			);
		});

		it('includes all cookies in the retry request', async () => {
			const cookies = createCookieStore({ refresh_token: 'refresh-abc' });
			const fetchMock = vi
				.fn()
				.mockResolvedValueOnce(unauthorized401())
				.mockResolvedValueOnce(ok200());

			mockRefreshSuccess('fresh-token');

			const request = new Request('http://example.test/api/resource');
			await callHook({ request, cookies, fetchMock });

			const retryRequest = fetchMock.mock.calls[1][0] as Request;
			const retryCookie = retryRequest.headers.get('cookie');
			expect(retryCookie).toContain('access_token=fresh-token');
			expect(retryCookie).toContain('refresh_token=refresh-abc');
		});
	});

	// 401 -> refresh -> retry (POST with body)

	describe('when a POST request returns 401 and refresh succeeds', () => {
		it('replays the original body on the retry request', async () => {
			const body = JSON.stringify({ problem_id: 7, num_desired: 3 });
			const cookies = createCookieStore({ refresh_token: 'refresh-xyz' });
			const fetchMock = vi
				.fn()
				.mockResolvedValueOnce(unauthorized401())
				.mockResolvedValueOnce(ok200('{"solutions":[]}'));

			mockRefreshSuccess();

			const request = new Request('http://example.test/api/method/nimbus/solve', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body
			});
			const response = await callHook({ request, cookies, fetchMock });

			expect(response.status).toBe(200);
			expect(fetchMock).toHaveBeenCalledTimes(2);

			// Verify the retry is also a POST with the same body
			const retryRequest = fetchMock.mock.calls[1][0] as Request;
			expect(retryRequest.method).toBe('POST');
			const retryBody = await retryRequest.text();
			expect(retryBody).toBe(body);
		});

		it('preserves content-type header on the retry', async () => {
			const cookies = createCookieStore({ refresh_token: 'refresh-xyz' });
			const fetchMock = vi
				.fn()
				.mockResolvedValueOnce(unauthorized401())
				.mockResolvedValueOnce(ok200());

			mockRefreshSuccess();

			const request = new Request('http://example.test/api/method/nimbus/solve', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: '{}'
			});
			await callHook({ request, cookies, fetchMock });

			const retryRequest = fetchMock.mock.calls[1][0] as Request;
			expect(retryRequest.headers.get('content-type')).toBe('application/json');
		});

		it('preserves the URL and method across the retry', async () => {
			const cookies = createCookieStore({ refresh_token: 'r' });
			const fetchMock = vi
				.fn()
				.mockResolvedValueOnce(unauthorized401())
				.mockResolvedValueOnce(ok200());

			mockRefreshSuccess();

			const url = 'http://example.test/api/method/e-nautilus/iterate?step=2';
			const request = new Request(url, {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: '{"ref_point":[1,2,3]}'
			});
			await callHook({ request, cookies, fetchMock });

			const initialRequest = fetchMock.mock.calls[0][0] as Request;
			const retryRequest = fetchMock.mock.calls[1][0] as Request;

			expect(initialRequest.url).toBe(url);
			expect(retryRequest.url).toBe(url);
			expect(initialRequest.method).toBe('POST');
			expect(retryRequest.method).toBe('POST');
		});
	});

	// Refresh failure cases

	describe('when the refresh call fails', () => {
		it('returns the original 401 if refresh returns non-200', async () => {
			const cookies = createCookieStore({ refresh_token: 'refresh-abc' });
			const fetchMock = vi.fn().mockResolvedValueOnce(unauthorized401());

			mockRefreshFailure(403);

			const request = new Request('http://example.test/api/resource');
			const response = await callHook({ request, cookies, fetchMock });

			expect(response.status).toBe(401);
			expect(fetchMock).toHaveBeenCalledTimes(1); // no retry
		});

		it('returns the original 401 if refresh returns 200 but no access_token', async () => {
			const cookies = createCookieStore({ refresh_token: 'refresh-abc' });
			const fetchMock = vi.fn().mockResolvedValueOnce(unauthorized401());

			vi.mocked(refreshAccessTokenRefreshPost).mockResolvedValue({
				status: 200,
				data: {}, // missing access_token
				headers: new Headers()
			});

			const request = new Request('http://example.test/api/resource');
			const response = await callHook({ request, cookies, fetchMock });

			expect(response.status).toBe(401);
			expect(fetchMock).toHaveBeenCalledTimes(1);
		});

		it('returns the original 401 (not 500) if refresh throws', async () => {
			const cookies = createCookieStore({ refresh_token: 'refresh-abc' });
			const fetchMock = vi.fn().mockResolvedValueOnce(unauthorized401());

			vi.mocked(refreshAccessTokenRefreshPost).mockRejectedValue(new TypeError('fetch failed'));

			const request = new Request('http://example.test/api/resource');
			const response = await callHook({ request, cookies, fetchMock });

			// Must degrade to 401, not propagate as an uncaught 500
			expect(response.status).toBe(401);
			expect(fetchMock).toHaveBeenCalledTimes(1);
		});

		it('does not update the cookie store when refresh fails', async () => {
			const cookies = createCookieStore({ refresh_token: 'refresh-abc' });
			const fetchMock = vi.fn().mockResolvedValueOnce(unauthorized401());

			mockRefreshFailure(401);

			const request = new Request('http://example.test/api/resource');
			await callHook({ request, cookies, fetchMock });

			expect(cookies.set).not.toHaveBeenCalled();
		});
	});

	// Concurrent 401s on the same request share one refresh (see `refreshPromises`)

	describe('concurrent 401s within the same request', () => {
		it('shares one refresh call across concurrent fetches on the same event', async () => {
			const cookies = createCookieStore({ refresh_token: 'refresh-shared' });
			// Same `event` for both calls -- this is what actually makes them "concurrent 401s on
			// one request" rather than two unrelated requests; every other test in this file uses
			// a fresh `event` per call, which is why this needs its own scenario.
			const event = { cookies } as unknown as HookArgs['event'];

			const fetchMockA = vi
				.fn()
				.mockResolvedValueOnce(unauthorized401())
				.mockResolvedValueOnce(ok200('{"a":true}'));
			const fetchMockB = vi
				.fn()
				.mockResolvedValueOnce(unauthorized401())
				.mockResolvedValueOnce(ok200('{"b":true}'));

			mockRefreshSuccess('shared-fresh-token');

			const [responseA, responseB] = await Promise.all([
				callHook({
					request: new Request('http://example.test/api/a'),
					cookies,
					fetchMock: fetchMockA,
					event
				}),
				callHook({
					request: new Request('http://example.test/api/b'),
					cookies,
					fetchMock: fetchMockB,
					event
				})
			]);

			expect(responseA.status).toBe(200);
			expect(responseB.status).toBe(200);
			// The one-refresh guarantee this test exists for: without de-duplication, two
			// concurrent 401s would each call this independently.
			expect(refreshAccessTokenRefreshPost).toHaveBeenCalledTimes(1);
		});

		it('does not share a refresh call across different events (independent requests)', async () => {
			const cookiesA = createCookieStore({ refresh_token: 'refresh-a' });
			const cookiesB = createCookieStore({ refresh_token: 'refresh-b' });

			const fetchMockA = vi
				.fn()
				.mockResolvedValueOnce(unauthorized401())
				.mockResolvedValueOnce(ok200());
			const fetchMockB = vi
				.fn()
				.mockResolvedValueOnce(unauthorized401())
				.mockResolvedValueOnce(ok200());

			mockRefreshSuccess('fresh-token');

			await Promise.all([
				callHook({
					request: new Request('http://example.test/api/a'),
					cookies: cookiesA,
					fetchMock: fetchMockA
				}),
				callHook({
					request: new Request('http://example.test/api/b'),
					cookies: cookiesB,
					fetchMock: fetchMockB
				})
			]);

			expect(refreshAccessTokenRefreshPost).toHaveBeenCalledTimes(2);
		});
	});

	// Non-401 error codes are not intercepted

	describe('non-401 error responses', () => {
		it('passes through 403 without attempting refresh', async () => {
			const cookies = createCookieStore({
				access_token: 'tok',
				refresh_token: 'ref'
			});
			const fetchMock = vi.fn().mockResolvedValueOnce(new Response(null, { status: 403 }));

			const request = new Request('http://example.test/api/resource');
			const response = await callHook({ request, cookies, fetchMock });

			expect(response.status).toBe(403);
			expect(refreshAccessTokenRefreshPost).not.toHaveBeenCalled();
			expect(fetchMock).toHaveBeenCalledTimes(1);
		});

		it('passes through 500 without attempting refresh', async () => {
			const cookies = createCookieStore({
				access_token: 'tok',
				refresh_token: 'ref'
			});
			const fetchMock = vi.fn().mockResolvedValueOnce(new Response(null, { status: 500 }));

			const request = new Request('http://example.test/api/resource');
			const response = await callHook({ request, cookies, fetchMock });

			expect(response.status).toBe(500);
			expect(refreshAccessTokenRefreshPost).not.toHaveBeenCalled();
		});
	});
});
