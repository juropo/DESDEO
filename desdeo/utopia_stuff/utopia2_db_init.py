import json  # noqa: D100

from desdeo.api.db_init import *  # noqa: F403

# from desdeo.utopia_stuff.utopia_problem import utopia_problem
from desdeo.utopia_stuff.utopia_problem_CO2 import utopia_problem as problem_CO2


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
with open("C:/MyTemp/data/forest_owners_punkaharju.json") as file:  # noqa: PTH123
    fo_dict = json.load(file)

with open("C:/MyTemp/data/reference_solutions_punkaharju.json") as file:
    ref_solutions = json.load(file)


def _generate_descriptions(mapjson: dict, sid: str, stand: str, holding: str, extension: str) -> dict:
    descriptions = {}
    if holding:
        for feat in mapjson["features"]:
            if False:  # noqa: SIM108
                ext = f".{feat["properties"][extension]}"
            else:
                ext = ""
            descriptions[feat["properties"][sid]] = (
                f"Ala {feat["properties"][holding].split("-")[-1]} kuvio {feat["properties"][stand]}{ext}: "
            )
    else:
        for feat in mapjson["features"]:
            if False:  # noqa: SIM108
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

    carbon_prices = [0.0, 15.0, 30.0]
    discounting_factor = 3

    for index in range(1, 4):
        # compensation = carbon_price / 100
        compensation = carbon_prices[index - 1] * discounting_factor / (1 - (1 + discounting_factor) ^ -100)

        problem, schedule_dict = problem_CO2(
            data_dir=fo_dict[name]["data_folder"],
            problem_name=f"Vaihe {index}",
            compensation=compensation,
            discounting_factor=discounting_factor,
        )
        problem_in_db = db_models.Problem(
            owner=user.id,
            name=f"Vaihe {index}",
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
            compensation=compensation,
        )
        db.add(map_info)

        problem_access = db_models.UserProblemAccess(
            user_id=user.id,
            problem_access=problem_in_db.id,
        )
        db.add(problem_access)

        # Add the reference solution from previous workshop as the starting value
        if isinstance(ref_solutions[name]["decisions_vars"], str):
            decision_vars = json.loads(ref_solutions[name]["decisions_vars"])
        else:
            decision_vars = ref_solutions[name]["decisions_vars"]
        decision_vars["C_1"] = 0
        decision_vars["C_2"] = 0
        decision_vars["C_3"] = 0
        reference_solution = db_models.SolutionArchive(
            user=user.id,
            problem=problem_in_db.id,
            method=nimbus_id,
            decision_variables=decision_vars,
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
