import type { DetailLevel } from './view-types';

const TIER_ORDER: Record<DetailLevel, number> = { compact: 0, normal: 1, expanded: 2 };

/** Container-width breakpoints, per the plan: "labels only at ~320px, bars at panel width, full
 *  detail expanded". `width` is the canvas container's own width, not the diagram's extent. */
function widthTier(width: number): DetailLevel {
	if (width < 320) return 'compact';
	if (width < 560) return 'normal';
	return 'expanded';
}

/** Zoom breakpoints: text and state chips become unreadable well before xyflow's own min-zoom,
 *  so detail has to back off at low zoom even inside a wide panel. */
function zoomTier(zoom: number): DetailLevel {
	if (zoom < 0.55) return 'compact';
	if (zoom < 0.85) return 'normal';
	return 'expanded';
}

/** Effective detail level: the more conservative (lower) of the width and zoom tiers, unless
 *  `forceCompact` (the `compact` prop) pins it to `'compact'` regardless. */
export function computeDetailLevel(
	width: number,
	zoom: number,
	forceCompact: boolean
): DetailLevel {
	if (forceCompact) return 'compact';
	const w = widthTier(width);
	const z = zoomTier(zoom);
	return TIER_ORDER[w] <= TIER_ORDER[z] ? w : z;
}
