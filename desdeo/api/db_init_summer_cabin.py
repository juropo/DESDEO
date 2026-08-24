"""Initialize the database with the summer cabin EV problem and its solution description metadata.

Includes everything from db_init.py and adds:
- summer_cabin_battery_robust_ev_problem with stripped objectives
  (only E_f_1, f_2, E_f_3 kept as objectives; all others demoted to extra functions)
- SolutionDescriptionMetaData for that problem
- the district heating problem from district_heating/district_heating.py

Run from desdeo/api/:
    python db_init_summer_cabin.py

The district heating problem needs hourly electricity and gas price data. It is read
from the directory given by the ``DISTRICT_HEATING_DATA_DIR`` environment variable
(default ``~/data/jenni``); if those files are not available, deterministic synthetic
prices are used instead so the database can still be built. The horizon defaults to
168 hours (one week) and can be set with ``DISTRICT_HEATING_HOURS``.
"""
# ruff: noqa: T201

import os
import sys
import warnings
from pathlib import Path

from sqlalchemy_utils import database_exists
from sqlmodel import Session, SQLModel

from desdeo.api.config import ServerConfig, SettingsConfig
from desdeo.api.db import engine
from desdeo.api.models import ProblemDB, User, UserRole
from desdeo.api.models.problem import DescriptionPart, ProblemMetaDataDB, SolutionDescriptionMetaData
from desdeo.api.models.scenario import ScenarioModelDB
from desdeo.api.routers.solution_description import validate_description_parts
from desdeo.api.routers.user_authentication import get_password_hash
from desdeo.problem.schema import ExtraFunction, Objective
from desdeo.problem.testproblems import dtlz2, river_pollution_problem, simple_knapsack
from desdeo.problem.testproblems.summer_cabin_electricity import (
    summer_cabin_battery_problem_split_scenario,
    summer_cabin_battery_robust_ev_problem,
)

# The district heating problem lives in the repository root, outside the desdeo package.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from district_heating.district_heating import (  # noqa: E402
    district_heating_problem,
    load_district_heating_prices,
)

DISTRICT_HEATING_DATA_DIR = Path(os.environ.get("DISTRICT_HEATING_DATA_DIR", Path.home() / "data" / "jenni"))
DISTRICT_HEATING_YEAR = int(os.environ.get("DISTRICT_HEATING_YEAR", "2022"))
DISTRICT_HEATING_HOURS = int(os.environ.get("DISTRICT_HEATING_HOURS", "168"))


def objective_to_extra_function(obj: Objective) -> ExtraFunction:
    """Convert an Objective to an ExtraFunction, keeping the same symbol."""
    return ExtraFunction(
        name=obj.name,
        symbol=obj.symbol,
        func=obj.func,
        is_linear=obj.is_linear,
        is_convex=obj.is_convex,
        is_twice_differentiable=obj.is_twice_differentiable,
    )


def build_summer_cabin_problem_full():
    """Build the summer cabin problem with all objectives kept intact."""
    print("Building summer_cabin_battery_robust_ev_problem full (this may take a moment)...")
    problem = summer_cabin_battery_robust_ev_problem()
    return problem.model_copy(
        update={
            "name": "Summer cabin energy management full",
            "description": (
                "Robust multi-objective energy management for a summer cabin with battery storage, "
                "solar panels, and an optional EV charger. "
                "All original objectives are retained. "
                "Scenario tree: 4 leaves covering grid outages in segments 2 and/or 3."
            ),
        }
    )


def build_summer_cabin_problem():
    """Build the summer cabin problem with demoted objectives."""
    print("Building summer_cabin_battery_robust_ev_problem (this may take a moment)...")
    problem = summer_cabin_battery_robust_ev_problem()

    keep = {"E_f_1", "f_2", "E_f_3"}
    rename = {
        "E_f_1": "Expected electricity cost /€",
        "f_2": "Investment cost /€",
        "E_f_3": "Expected outage duration /h",
    }
    new_objectives = [
        o.model_copy(update={"name": rename[o.symbol]}) if o.symbol in rename else o
        for o in problem.objectives
        if o.symbol in keep
    ]
    demoted = [objective_to_extra_function(o) for o in problem.objectives if o.symbol not in keep]
    existing_extra = list(problem.extra_funcs or [])

    return problem.model_copy(
        update={
            "name": "Summer cabin energy management (robust EV)",
            "description": (
                "Robust multi-objective energy management for a summer cabin with battery storage, "
                "solar panels, and an optional EV charger. "
                "Objectives: expected electricity cost (E_f_1), investment cost (f_2), "
                "expected unserved hours (E_f_3). "
                "Scenario tree: 4 leaves covering grid outages in segments 2 and/or 3."
            ),
            "objectives": new_objectives,
            "extra_funcs": existing_extra + demoted,
        }
    )


def build_district_heating_problem():
    """Build the district heating problem over a short horizon using the real price data."""
    hours = DISTRICT_HEATING_HOURS
    print(
        f"Building district heating problem ({hours} h of {DISTRICT_HEATING_YEAR} prices "
        f"from {DISTRICT_HEATING_DATA_DIR})..."
    )

    spot, gas = load_district_heating_prices(DISTRICT_HEATING_DATA_DIR, DISTRICT_HEATING_YEAR)
    if len(spot) < hours:
        raise ValueError(f"Price data has only {len(spot)} hours, but {hours} were requested.")
    spot, gas = spot[:hours], gas[:hours]

    problem = district_heating_problem(spot, gas)
    return problem.model_copy(
        update={
            "description": (
                f"{problem.description} Horizon: the first {hours} hours of "
                f"{DISTRICT_HEATING_YEAR} all-in electricity and gas prices."
            ),
        }
    )


def add_solution_description(session, problem_db, parts, problem) -> None:
    """Attach solution description parts to a problem, checking that they can be rendered.

    The parts go through the same validation as the update_metadata endpoint, so a template
    referring to a symbol that does not exist is reported here rather than when a user asks
    for a description. Unlike the endpoint this only warns, because a description may be
    written for a problem derived from this one, which inherits the metadata and has symbols
    the problem itself does not.
    """
    try:
        validate_description_parts(parts, problem)
    except ValueError as e:
        warnings.warn(f"Description for {problem_db.name!r}: {e}", stacklevel=2)

    metadata_db = ProblemMetaDataDB(problem_id=problem_db.id, problem=problem_db)
    session.add(metadata_db)
    session.commit()
    session.refresh(metadata_db)

    session.add(
        SolutionDescriptionMetaData(
            metadata_id=metadata_db.id,
            parts=[p.model_dump() for p in parts],
            separator="\n",
        )
    )
    session.commit()


def tensor_constant_values(problem, symbol: str) -> list[float]:
    """Return the plain list of values of a problem's TensorConstant.

    The stored values are a MathJSON list, i.e. ``["List", v1, v2, ...]``.
    """
    constant = next(c for c in problem.constants if c.symbol == symbol)
    return [float(v) for v in constant.values[1:]]


def district_heating_description_parts(problem) -> list[DescriptionPart]:
    """Build the solution description parts for the district heating problem.

    Besides the objective totals, this reports the highest cost paid in any single hour
    for electricity and for gas. Each is a single number: the maximum over the horizon of
    the hourly price times that hour's boiler input (steam plus storage charging). Both
    the prices and the boiler inputs are referred to by their tensor element symbols,
    ``p_el_1``, ``el_steam_1`` and so on: the prices come from the problem's constants and
    the inputs from the solution.
    """
    hours = range(1, len(tensor_constant_values(problem, "p_el")) + 1)

    hourly_el_cost = [["Multiply", f"p_el_{t}", ["Add", f"el_steam_{t}", f"el_charge_{t}"]] for t in hours]
    hourly_gas_cost = [["Multiply", f"p_gas_{t}", ["Add", f"gas_steam_{t}", f"gas_charge_{t}"]] for t in hours]

    return [
        DescriptionPart(text="Total cost over the horizon:"),
        DescriptionPart(label="  Electricity", symbol="f_el", format_spec=".2f", suffix=" €"),
        DescriptionPart(label="  Gas", symbol="f_gas", format_spec=".2f", suffix=" €"),
        DescriptionPart(
            label="  Combined",
            expression=["Add", "f_el", "f_gas"],
            format_spec=".2f",
            suffix=" €",
        ),
        DescriptionPart(text="Most expensive single hour:"),
        DescriptionPart(
            label="  Electricity",
            expression=["Max", *hourly_el_cost],
            format_spec=".2f",
            suffix=" €",
        ),
        DescriptionPart(
            label="  Gas",
            expression=["Max", *hourly_gas_cost],
            format_spec=".2f",
            suffix=" €",
        ),
        DescriptionPart(text="Thermal storage:"),
        DescriptionPart(
            label="  Peak storage level",
            expression=["Max", *[f"storage_{t}" for t in hours]],
            format_spec=".2f",
            suffix=" MWh",
        ),
    ]


DESCRIPTION_PARTS = [
    DescriptionPart(text="Investment decisions:"),
    DescriptionPart(label="  Battery installed", symbol="y", format_spec=".0f"),
    DescriptionPart(label="  Battery capacity", symbol="E", format_spec=".1f", suffix=" kWh"),
    DescriptionPart(label="  Solar panels", symbol="n", format_spec=".0f"),
    DescriptionPart(text="Investment cost breakdown:"),
    DescriptionPart(
        label="  Battery fixed cost",
        expression=["Multiply", 2000, "y"],
        format_spec=".0f",
        suffix=" €",
    ),
    DescriptionPart(
        label="  Battery capacity cost",
        expression=["Multiply", 310, "E"],
        format_spec=".0f",
        suffix=" €",
    ),
    DescriptionPart(
        label="  Solar panel cost",
        expression=["Multiply", 200, "n"],
        format_spec=".0f",
        suffix=" €",
    ),
    DescriptionPart(label="  Total investment cost", symbol="E_f_2", format_spec=".0f", suffix=" €"),
    DescriptionPart(text="Expected performance:"),
    DescriptionPart(label="  Expected electricity cost", symbol="E_f_1", format_spec=".2f", suffix=" €"),
    DescriptionPart(label="  Expected unserved hours", symbol="E_f_3", format_spec=".2f", suffix=" h"),
    DescriptionPart(text="Worst-case performance:"),
    DescriptionPart(label="  Worst-case electricity cost", symbol="robust_f_1", format_spec=".2f", suffix=" €"),
    DescriptionPart(label="  Worst-case investment cost", symbol="robust_f_2", format_spec=".0f", suffix=" €"),
    DescriptionPart(label="  Worst-case unserved hours", symbol="robust_f_3", format_spec=".1f", suffix=" h"),
]


if __name__ == "__main__":
    if SettingsConfig.debug:
        # Built first so that missing price data fails before the database is touched.
        district_heating_prob = build_district_heating_problem()

        print("Creating database tables.")
        if not database_exists(engine.url):
            SQLModel.metadata.create_all(engine)
        else:
            warnings.warn("Database already exists. Clearing it.", stacklevel=1)
            SQLModel.metadata.reflect(bind=engine)
            SQLModel.metadata.drop_all(bind=engine)
            SQLModel.metadata.create_all(engine)
        print("Database tables created.")

        with Session(engine) as session:
            user_analyst = User(
                username=ServerConfig.test_user_analyst_name,
                password_hash=get_password_hash(ServerConfig.test_user_analyst_password),
                role=UserRole.analyst,
                group="test",
            )
            session.add(user_analyst)
            session.commit()
            session.refresh(user_analyst)

            # Standard test problems from db_init.py
            for problem in [dtlz2(10, 3), simple_knapsack(), river_pollution_problem()]:
                session.add(ProblemDB.from_problem(problem, user_analyst))
            session.commit()
            print("Standard test problems added.")

            # Summer cabin problem with all objectives (unstripped)
            cabin_full_problem = build_summer_cabin_problem_full()
            cabin_full_db = ProblemDB.from_problem(cabin_full_problem, user_analyst)
            session.add(cabin_full_db)
            session.commit()
            session.refresh(cabin_full_db)
            print(f"Summer cabin full problem added (id={cabin_full_db.id}).")

            # Summer cabin problem with stripped objectives
            cabin_problem = build_summer_cabin_problem()
            cabin_db = ProblemDB.from_problem(cabin_problem, user_analyst)
            session.add(cabin_db)
            session.commit()
            session.refresh(cabin_db)
            print(f"Summer cabin problem added (id={cabin_db.id}).")

            # Solution description metadata for full problem
            add_solution_description(session, cabin_full_db, DESCRIPTION_PARTS, cabin_full_problem)
            print("Solution description metadata added for full problem.")

            # Solution description metadata for stripped problem
            add_solution_description(session, cabin_db, DESCRIPTION_PARTS, cabin_problem)
            print("Solution description metadata added for stripped problem.")

            # Base summer cabin problem with its scenario model (for testing scenario detection in CUMULUS)
            print("Building summer_cabin_battery_problem_split_scenario (this may take a moment)...")
            scenario_model = summer_cabin_battery_problem_split_scenario()
            base_problem = scenario_model.base_problem.model_copy(
                update={
                    "name": "Summer cabin energy management (with scenarios)",
                    "description": (
                        "Base summer cabin battery problem with an associated 4-leaf scenario model "
                        "covering grid outages in segments 2 and/or 3. "
                        "Use CUMULUS to optionally build a combined scenario problem with uncertainty measures."
                    ),
                }
            )
            cabin_scenario_base_db = ProblemDB.from_problem(base_problem, user_analyst)
            session.add(cabin_scenario_base_db)
            session.commit()
            session.refresh(cabin_scenario_base_db)
            print(f"Summer cabin base problem added (id={cabin_scenario_base_db.id}).")

            scenario_model_db = ScenarioModelDB.from_scenario_model(
                scenario_model, user=user_analyst, base_problem_id=cabin_scenario_base_db.id
            )
            session.add(scenario_model_db)
            session.commit()
            session.refresh(scenario_model_db)
            print(
                f"ScenarioModelDB added (id={scenario_model_db.id}) linked to problem id={cabin_scenario_base_db.id}."
            )

            # This warns that the description refers to symbols the base problem does not have
            # (E_f_1, robust_f_1, ...), which is expected: they appear in the combined scenario
            # problem CUMULUS builds from it, and that problem inherits this metadata.
            add_solution_description(session, cabin_scenario_base_db, DESCRIPTION_PARTS, base_problem)
            print("Solution description metadata added for scenario base problem.")

            # District heating problem
            district_heating_db = ProblemDB.from_problem(district_heating_prob, user_analyst)
            session.add(district_heating_db)
            session.commit()
            session.refresh(district_heating_db)
            print(f"District heating problem added (id={district_heating_db.id}).")

            add_solution_description(
                session,
                district_heating_db,
                district_heating_description_parts(district_heating_prob),
                district_heating_prob,
            )
            print("Solution description metadata added for district heating problem.")

        print("Done.")

    else:
        pass
