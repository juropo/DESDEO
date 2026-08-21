/** Small illustrative network (an irrigation-decision influence diagram) used by the Storybook
 *  story and available for ad-hoc manual testing. Not part of the public package surface — it's
 *  a fixture, not a component/type this package exports for consumers to build on. */

import type { NetworkModel } from './types';

export const sampleIrrigationNetwork: NetworkModel = {
	nodes: [
		{
			id: 'weather',
			label: 'Weather',
			kind: 'chance',
			states: ['dry', 'normal', 'wet'],
			description: 'Growing-season weather outlook.'
		},
		{
			id: 'soil_moisture',
			label: 'Soil moisture',
			kind: 'chance',
			states: ['low', 'medium', 'high'],
			description: 'Soil moisture at planting, driven by weather.'
		},
		{
			id: 'irrigate',
			label: 'Irrigate',
			kind: 'decision',
			states: ['none', 'light', 'heavy'],
			description: 'How much supplemental irrigation to apply.'
		},
		{
			id: 'water_usage',
			label: 'Water usage',
			kind: 'deterministic',
			description: 'Water volume used, a fixed function of the irrigation decision.'
		},
		{
			id: 'crop_yield',
			label: 'Crop yield',
			kind: 'chance',
			states: ['poor', 'average', 'good'],
			description: 'Harvest yield, depending on soil moisture and irrigation.'
		},
		{
			id: 'profit',
			label: 'Profit',
			kind: 'objective',
			description: 'Revenue from yield minus water and irrigation cost.'
		},
		{
			id: 'water_cost',
			label: 'Water cost',
			kind: 'objective',
			description: 'Cost of water used, kept separate so it can be traded off explicitly.'
		}
	],
	edges: [
		{ id: 'e1', source: 'weather', target: 'soil_moisture' },
		{ id: 'e2', source: 'soil_moisture', target: 'crop_yield' },
		{ id: 'e3', source: 'irrigate', target: 'crop_yield' },
		{ id: 'e4', source: 'irrigate', target: 'water_usage' },
		{ id: 'e5', source: 'crop_yield', target: 'profit' },
		{ id: 'e6', source: 'water_usage', target: 'profit' },
		{ id: 'e7', source: 'water_usage', target: 'water_cost' }
	]
};
