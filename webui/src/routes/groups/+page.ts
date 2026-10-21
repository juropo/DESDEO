import type { PageLoad } from './$types';
import {
	getCurrentUserInfoUserInfoGet,
	getGroupInfoGdmGetGroupInfoPost,
	getProblemProblemProblemIdGet
} from '$lib/gen/endpoints/DESDEOFastAPI';
import { withFetch } from '$lib/api/ssr-fetch';

export const load: PageLoad = async ({ fetch }) => {
	const user = await getCurrentUserInfoUserInfoGet(withFetch(fetch));
	if (user.status !== 200) {
		throw new Error('Failed to fetch user info');
	}

	const groupIds = user.data.group_ids;
	if (!groupIds || groupIds.length === 0) {
		return { problemList: [], groupList: [] };
	}

	// Each group's info (and, below, each group's problem info) is independent of the others --
	// fetched with Promise.all instead of a sequential for-loop, which used to cost roughly 2×N
	// sequential round-trips for a user in N groups.
	const groupList = await Promise.all(
		groupIds.map(async (id) => {
			const info = await getGroupInfoGdmGetGroupInfoPost({ group_id: id }, withFetch(fetch));
			if (info.status !== 200) {
				throw new Error(`Failed to fetch group info for group ID ${id}`);
			}
			return info.data;
		})
	);

	const problemList = await Promise.all(
		groupList.map(async (group) => {
			const problem = await getProblemProblemProblemIdGet(
				group.problem_id,
				undefined,
				withFetch(fetch)
			);
			if (problem.status !== 200) {
				throw new Error(`Failed to fetch problem info for problem ID ${group.problem_id}`);
			}
			return problem.data;
		})
	);

	return {
		problemList,
		groupList
	};
};
