import type { PageLoad } from './$types';
import { getProblemsInfoProblemAllInfoGet } from '$lib/gen/endpoints/DESDEOFastAPI';
import type { ProblemInfo } from '$lib/gen/endpoints/DESDEOFastAPI';
import { withFetch } from '$lib/api/ssr-fetch';

type ProblemList = ProblemInfo[];

export const load: PageLoad = async ({ fetch }) => {
	const res = await getProblemsInfoProblemAllInfoGet(withFetch(fetch));
	if (res.status !== 200) throw new Error('Failed to fetch problems');
	return { problems: res.data satisfies ProblemList };
};
