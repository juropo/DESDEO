/** Per-kind visual conventions, shared between the custom node components and the table view so
 *  the two stay in sync without duplicating the mapping. Shapes follow the usual influence-diagram
 *  convention: chance = ellipse, decision = rectangle, objective/utility = diamond, deterministic
 *  = double-bordered rectangle. */

import type { NetworkNodeKind } from './types';

export type NodeShape = 'ellipse' | 'rect' | 'diamond' | 'double-rect';

export interface NodeKindStyle {
	shape: NodeShape;
	/** CSS color token (a `--chart-N` var already defined in `app.css`), reused as-is so the
	 *  network view stays visually consistent with the rest of the app instead of inventing a
	 *  parallel palette. */
	colorVar: string;
	label: string;
	/** Icon path for `@lucide/svelte/icons/<icon>`. */
	icon: string;
}

export const NODE_KIND_STYLE: Record<NetworkNodeKind, NodeKindStyle> = {
	chance: { shape: 'ellipse', colorVar: 'var(--chart-2)', label: 'Chance', icon: 'dices' },
	decision: {
		shape: 'rect',
		colorVar: 'var(--chart-4)',
		label: 'Decision',
		icon: 'square-check'
	},
	objective: { shape: 'diamond', colorVar: 'var(--chart-1)', label: 'Objective', icon: 'target' },
	deterministic: {
		shape: 'double-rect',
		colorVar: 'var(--chart-3)',
		label: 'Deterministic',
		icon: 'sigma'
	}
};
