"""This is a docstring."""

import json
import pandas as pd

from desdeo.api.db import SessionLocal
from desdeo.api.db_models import Method, SolutionArchive, User, Utopia
from desdeo.api.db_models import Problem as ProblemInDB
from desdeo.api.schema import Methods
from desdeo.utopia_stuff.utopia_problem import utopia_problem


def read_solutions_from_db(problemnames: list[str]):
    """Reads all solutions associated with the given problem and does something with them.

    Args:
        problemname (str): name of the problem
    """
    db = SessionLocal()
    problems = []
    for name in problemnames:
        problems.extend(db.query(ProblemInDB).filter(ProblemInDB.name == name).all())

    nimbus_method = db.query(Method).filter(Method.kind == Methods.NIMBUS).first()

    objective_names = ["NPV", "Timber volume", "Harvest income", "Carbon storage"]

    data = []
    for p in problems:
        solution = (
            db.query(SolutionArchive)
            .filter(SolutionArchive.problem == p.id, SolutionArchive.method == nimbus_method.id, SolutionArchive.chosen)
            .first()
        )
        if not solution:
            continue
        user = db.query(User).filter(User.id == solution.user).first()
        row = {"UserName": user.username, "Problem": p.name}
        i = 0
        for obj in solution.objectives:
            row[objective_names[i]] = obj
            i += 1
        data.append(row)
    df = pd.DataFrame(data)
    print(df)
    df.to_csv("workshop_final_solutions.csv", sep="\t", index=True)
    data = []
    for p in problems:
        solutions = (
            db.query(SolutionArchive)
            .filter(SolutionArchive.problem == p.id, SolutionArchive.method == nimbus_method.id)
            .all()
        )
        for solution in solutions:
            user = db.query(User).filter(User.id == solution.user).first()
            row = {
                "UserName": user.username,
                "Problem": p.name,
                "Saved": solution.saved,
                "Shared": solution.shared,
                "Chosen": solution.chosen,
                "Decsions": solution.decision_variables,
            }
            i = 0
            for obj in solution.objectives:
                row[objective_names[i]] = obj
                i += 1
            data.append(row)
    df = pd.DataFrame(data)
    print(df)
    df.to_csv("workshop_all_solutions.csv", sep="\t", index=True)


if __name__ == "__main__":
    read_solutions_from_db(["Metsänhoitosuunnitelma"])
