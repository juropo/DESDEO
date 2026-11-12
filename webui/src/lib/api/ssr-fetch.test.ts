import { describe, expect, it } from 'vitest';
import { withFetch } from './ssr-fetch';

describe('withFetch', () => {
	it('wraps the given fetch as fetchImpl', () => {
		const fetchMock = (() => {}) as unknown as typeof globalThis.fetch;
		const result = withFetch(fetchMock) as RequestInit & { fetchImpl?: typeof fetch };
		expect(result.fetchImpl).toBe(fetchMock);
	});
});
