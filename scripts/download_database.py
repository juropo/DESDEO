import os

import pandas
from sqlmodel import create_engine
from sqlalchemy import text

filepath = "c:/MyTemp/data/testbackup/"


DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

print("Starting engine")
engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

with engine.connect() as conn:
    df = pandas.read_sql(
        text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_name;
        """),
        conn,
    )

tables = df["table_name"].tolist()

for table in tables:
    with engine.connect() as conn:
        df = pandas.read_sql_table(table_name=table, con=conn)

        print(df)
        if df.empty:
            print(f"Skipping {table}: empty dataframe")
            continue

        df.to_csv(f"{filepath}{table}.csv", sep=";", index=False)
        # conn.commit()

"""    fake_conn = engine.raw_connection()
    fake_cur = fake_conn.cursor()
    fake_cur.copy_expert(copy_sql, dbcopy_f)
    fake_conn.commit()"""
