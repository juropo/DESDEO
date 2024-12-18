"""This is a docstring."""

import json

from desdeo.api.db import SessionLocal
from desdeo.api.db_models import Problem as ProblemInDB
from desdeo.api.db_models import User, Utopia
from desdeo.utopia_stuff.utopia_problem import utopia_problem


def reinit_user(username: str):
    """Recreates the forest problem for the specified user.

    Args:
        username (str): username of the forest owner
    """
    db = SessionLocal()
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise Exception("User not found")  # noqa: TRY002
    problem_in_db = db.query(ProblemInDB).filter(ProblemInDB.owner == user.id).first()
    map_info = db.query(Utopia).filter(Utopia.problem == problem_in_db.id, Utopia.user == user.id)

    with open("C:/MyTemp/data/forest_owners_punkaharju.json") as file:  # noqa: PTH123
        fo_dict = json.load(file)

    print(fo_dict[username])

    problem, schedule_dict = utopia_problem(
        data_dir=fo_dict[username]["data_folder"], problem_name="Metsänhoitosuunnitelma", separator=","
    )

    problem_in_db.value = problem.model_dump(mode="json")
    map_info.schedule_dict = schedule_dict

    db.commit()


if __name__ == "__main__":
    reinit_user("kero")
