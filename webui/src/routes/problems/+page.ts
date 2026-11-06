import type { PageLoad } from './$types';
import { getDmUsersUsersDmsGet, getProblemsInfoProblemAllInfoGet } from '$lib/gen/endpoints/DESDEOFastAPI';
import type { UserPublic } from '$lib/gen/endpoints/DESDEOFastAPI';
import { withFetch } from '$lib/api/ssr-fetch';

export const load: PageLoad = async ({ fetch }) => {
	const [problemsRes, dmsRes] = await Promise.all([
		getProblemsInfoProblemAllInfoGet(withFetch(fetch)),
		getDmUsersUsersDmsGet(withFetch(fetch)).catch(() => null)
	]);

	if (problemsRes.status !== 200) {
		throw new Error('Failed to fetch problems');
	}

	const dmUsers: UserPublic[] = dmsRes?.status === 200 ? (dmsRes.data as UserPublic[]) : [];

	return {
		problemList: problemsRes.data,
		dmUsers
	};
};
