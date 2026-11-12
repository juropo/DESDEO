"""DB model for named evidence sets (Phase E of `.claude/desdeo-graphical-models-plan.md`).

**Own simple table, not `ProblemMetaDataDB`'s polymorphic side-table pattern.** That pattern
(`GraphicalModelMetaData` and friends in `api/models/problem.py`) exists for data that
*describes the problem itself* -- network/annotation/RJT/`IndexMap`, geographic data, ... -- one
coherent row set per problem, doubling as the frontend's signal to render a specialised view (see
`api/models/problem.py`'s own module docstring). A saved evidence set is a different shape of
thing entirely: a user-owned, independently addressable *bookmark* into a diagram someone is
exploring ("a figure corresponds to a scenario id, not a click sequence" -- the plan's own
rationale), closer in kind to `UserSavedSolutionDB` (`api/models/generic_states.py`, a
user/problem-scoped saved row with its own id) than to problem metadata. So it gets its own
table, keyed by `problem_id`/`user_id` directly, the same shape `UserSavedSolutionDB` already
uses -- not routed through the polymorphic union.
"""

from datetime import UTC, datetime

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class GraphicalEvidenceSetDB(SQLModel, table=True):
    """A named hard-evidence dict (`NetworkEvidence`, webui) saved against a compiled diagram.

    Addressable by its own `id` alone, independent of `problem_id` -- "shareable by URL" (the
    plan's own checklist wording) means exactly that: given the id, `GET /graphical/evidence-
    sets/{id}` returns the identical evidence dict deterministically to *any* user with visibility
    into the underlying problem (checked via `fetch_problem_with_role_check`, the same rule
    `/structure`/`/whatif` already use for problem visibility) -- not only the user who saved it.
    Creation and deletion, by contrast, are restricted to the saving user (or an admin/analyst,
    mirroring the rest of this router's ownership convention) -- see `routers/graphical.py`.
    """

    id: int | None = Field(primary_key=True, default=None)
    problem_id: int = Field(foreign_key="problemdb.id")
    user_id: int = Field(foreign_key="user.id")
    name: str
    evidence: dict[str, str] = Field(sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
