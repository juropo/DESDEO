import json  # noqa: D100

from desdeo.api.db_init import *  # noqa: F403

# from desdeo.utopia_stuff.utopia_problem import utopia_problem
from desdeo.utopia_stuff.utopia_problem_CO2 import utopia_problem as problem_CO2

with open("C:/MyTemp/code/users_and_passwords.json") as file:  # noqa: PTH123
    userdict = json.load(file)

db = SessionLocal()


# Actual forest owners start from here
# The contents of this file are not supposed to be found on github
"""
The json file contents look something like this
{
  "jane_smith": {
    "password": "password123",
    "simulation_results": "C:/MyTemp/data/alternatives/fake_location/alternatives.csv",
    "treatment_key": "C:/MyTemp/data/alternatives/fake_location/alternatives_key.csv",
    "mapjson": "C:/MyTemp/data/alternatives/fake_location/map.geojson",
    "stand_id": "id",
    "stand_descriptor": "number",
    "holding_descriptor": "estate_code",
    "extension":"extension"
  }
}
"""
with open("C:/MyTemp/data/forest_owners_ws2.json") as file:  # noqa: PTH123
    fo_dict = json.load(file)

with open("C:/MyTemp/data/reference_solutions.json") as file:
    ref_solutions = json.load(file)


def _generate_descriptions(mapjson: dict, sid: str, stand: str, holding: str, extension: str) -> dict:
    descriptions = {}
    if holding:
        for feat in mapjson["features"]:
            if feat["properties"][extension]:  # noqa: SIM108
                ext = f".{feat["properties"][extension]}"
            else:
                ext = ""
            descriptions[feat["properties"][sid]] = (
                f"Ala {feat["properties"][holding].split("-")[-1]} kuvio {feat["properties"][stand]}{ext}: "
            )
    else:
        for feat in mapjson["features"]:
            if feat["properties"][extension]:  # noqa: SIM108
                ext = f".{feat["properties"][extension]}"
            else:
                ext = ""
            descriptions[feat["properties"][sid]] = f"Kuvio {feat["properties"][stand]}{ext}: "
    return descriptions


for name in fo_dict:
    print(name)
    user = db_models.User(
        username=name,
        password_hash=get_password_hash(fo_dict[name]["password"]),
        role=UserRole.DM,
        privilages=[],
        user_group="",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    problem, schedule_dict = problem_CO2(
        data_dir=fo_dict[name]["data_folder"], problem_name="Metsänhoitosuunnitelma CO2", compensation=30
    )
    problem_in_db = db_models.Problem(
        owner=user.id,
        name="Vaihe 3",
        kind=ProblemKind.CONTINUOUS,
        obj_kind=ObjectiveKind.ANALYTICAL,
        solver=Solvers.GUROBIPY,
        presumed_ideal=problem.get_ideal_point(),
        presumed_nadir=problem.get_nadir_point(),
        value=problem.model_dump(mode="json"),
    )
    db.add(problem_in_db)
    db.commit()
    db.refresh(problem_in_db)

    # The info about the map and decision alternatives now goes into the database
    with open(fo_dict[name]["mapjson"]) as f:  # noqa: PTH123
        forest_map = f.read()
    map_info = db_models.Utopia(
        problem=problem_in_db.id,
        user=user.id,
        map_json=forest_map,
        schedule_dict=schedule_dict,
        years=["5", "10", "20"],
        stand_id_field=fo_dict[name]["stand_id"],
        stand_descriptor=_generate_descriptions(
            json.loads(forest_map),
            fo_dict[name]["stand_id"],
            fo_dict[name]["stand_descriptor"],
            fo_dict[name]["holding_descriptor"],
            fo_dict[name]["extension"],
        ),
    )
    db.add(map_info)

    problem_access = db_models.UserProblemAccess(
        user_id=user.id,
        problem_access=problem_in_db.id,
    )
    db.add(problem_access)

    # Add the reference solution from previous workshop as the starting value
    reference_solution = db_models.SolutionArchive(
        user=user.id,
        problem=problem_in_db.id,
        method=nimbus_id,
        decision_variables=ref_solutions[name]["decisions_vars"],
        objectives=ref_solutions[name]["objectives"] + [0],
        saved=True,
        current=True,
        chosen=False,
        shared=True,
    )
    db.add(reference_solution)

    db.commit()

# Extra problem ends here

db.close()
