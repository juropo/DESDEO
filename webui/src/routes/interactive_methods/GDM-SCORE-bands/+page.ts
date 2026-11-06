import type { PageLoad } from './$types';
import {
	getGroupInfoGdmGetGroupInfoPost,
	getProblemProblemProblemIdGet
} from '$lib/gen/endpoints/DESDEOFastAPI';
import type { ProblemInfo } from '$lib/gen/endpoints/DESDEOFastAPI';
import { withFetch } from '$lib/api/ssr-fetch';

type LoadData = {
	refreshToken?: string;
};

type Problem = ProblemInfo;

export const load: PageLoad<LoadData> = async ({ url, data, fetch }) => {
	const groupId = url.searchParams.get('group');
	if (!groupId) throw new Error('No group ID provided');

	const group = await getGroupInfoGdmGetGroupInfoPost(
		{ group_id: parseInt(groupId) },
		withFetch(fetch)
	);
	if (group.status !== 200) throw new Error('Failed to fetch group info');
	const problemId = group.data.problem_id;

	const resProblem = await getProblemProblemProblemIdGet(problemId, undefined, withFetch(fetch));
	if (resProblem.status !== 200) throw new Error('Failed to fetch problem info');

	return {
		problem: resProblem.data as Problem,
		refreshToken: data.refreshToken,
		group: group.data
	};
};
