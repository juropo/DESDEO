/** Per-kind visual conventions, shared between the custom node components and the table view so
 *  the two stay in sync without duplicating the mapping. Shapes follow the usual influence-diagram
 *  convention: chance = ellipse, decision = rectangle, objective/utility = diamond, deterministic
 *  = double-bordered rectangle. */

import type { Component } from 'svelte';
import IconDices from '@lucide/svelte/icons/dices';
import IconSquareCheck from '@lucide/svelte/icons/square-check';
import IconTarget from '@lucide/svelte/icons/target';
import IconSigma from '@lucide/svelte/icons/sigma';
import type { NetworkNodeKind } from './types';

export type NodeShape = 'ellipse' | 'rect' | 'diamond' | 'double-rect';

export interface NodeKindStyle {
	shape: NodeShape;
	/** CSS color token (a `--chart-N` var already defined in `app.css`), reused as-is so the
	 *  network view stays visually consistent with the rest of the app instead of inventing a
	 *  parallel palette. */
	colorVar: string;
	label: string;
	/** The icon component for this kind -- `nodes/GraphNode.svelte`'s only source for it, so
	 *  changing a kind's icon means changing it here and nowhere else. (Previously a separate
	 *  icon *name* string that nothing actually read: each per-kind wrapper component hardcoded
	 *  its own icon import instead, so this field could drift from what actually rendered.) */
	icon: Component<{ class?: string }>;
}

export const NODE_KIND_STYLE: Record<NetworkNodeKind, NodeKindStyle> = {
	chance: { shape: 'ellipse', colorVar: 'var(--chart-2)', label: 'Chance', icon: IconDices },
	decision: {
		shape: 'rect',
		colorVar: 'var(--chart-4)',
		label: 'Decision',
		icon: IconSquareCheck
	},
	objective: { shape: 'diamond', colorVar: 'var(--chart-1)', label: 'Objective', icon: IconTarget },
	deterministic: {
		shape: 'double-rect',
		colorVar: 'var(--chart-3)',
		label: 'Deterministic',
		icon: IconSigma
	}
};
