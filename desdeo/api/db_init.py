"""This module initializes the database."""

import warnings

import numpy as np
import polars as pl
from sqlalchemy import text
from sqlalchemy_utils import create_database, database_exists, drop_database

from desdeo.api import db_models
from desdeo.api.config import DBConfig
from desdeo.api.db import Base, SessionLocal, engine
from desdeo.api.routers.UserAuth import get_password_hash
from desdeo.api.schema import Methods, ObjectiveKind, ProblemKind, Solvers, UserPrivileges, UserRole
from desdeo.problem.schema import DiscreteRepresentation, Objective, Problem, Variable
from desdeo.problem.testproblems import binh_and_korn, forest_problem, nimbus_test_problem, river_pollution_problem
from desdeo.utopia_stuff.utopia_problem_old import utopia_problem_old

TEST_USER = "test"
TEST_PASSWORD = "test"  # NOQA: S105 # TODO: Remove this line and create a proper user creation system.

# The following line creates the database and tables. This is not ideal, but it is simple for now.
# It recreates the tables every time the server starts. Any data saved in the database will be lost.
# TODO: Remove this line and create a proper database migration system.
print("Creating database tables.")
if not database_exists(engine.url):
    create_database(engine.url)
else:
    warnings.warn("Database already exists. Clearing it.", stacklevel=1)

    # Drop all active connections
    db = SessionLocal()
    terminate_connections_sql = text("""
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE datname = :db_name AND pid <> pg_backend_pid();
    """)
    db.execute(terminate_connections_sql, {"db_name": DBConfig.db_database})

    # Drop all tables
    Base.metadata.drop_all(bind=engine)
print("Database tables created.")

# Create the tables in the database.
Base.metadata.create_all(bind=engine)

# Create test users
db = SessionLocal()


# I guess we need to have methods in the database as well
nimbus = db_models.Method(
    kind=Methods.NIMBUS,
    properties=[],
    name="NIMBUS",
)
db.add(nimbus)
db.commit()

nimbus_id = nimbus.id

db.close()
