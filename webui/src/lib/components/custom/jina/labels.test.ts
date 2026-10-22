import { describe, it, expect } from 'vitest';
import { compactNumber } from './labels';

describe('compactNumber', () => {
	it('abbreviates large values with SI suffixes', () => {
		expect(compactNumber(-32_123_456)).toBe('-32.1M');
		expect(compactNumber(7_890)).toBe('7.9k');
		expect(compactNumber(2_500_000_000)).toBe('2.5B');
	});

	it('keeps small values short', () => {
		expect(compactNumber(158.4)).toBe('158');
		expect(compactNumber(3.14159)).toBe('3.14');
		expect(compactNumber(0.012345)).toBe('0.012');
		expect(compactNumber(0)).toBe('0');
		expect(compactNumber(null)).toBe('—');
	});
});
