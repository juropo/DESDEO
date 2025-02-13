import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from desdeo.problem.schema import (
    Constant,
    Constraint,
    ConstraintTypeEnum,
    DiscreteRepresentation,
    ExtraFunction,
    Objective,
    ObjectiveTypeEnum,
    Problem,
    TensorConstant,
    TensorVariable,
    Variable,
    VariableTypeEnum,
)
from desdeo.tools.utils import available_solvers, payoff_table_method


def utopia_problem(
    data_dir: str,
    problem_name: str = "Forest problem",
    holding: int = 1,
    compensation: float = 0,
    discounting_factor: int = 3,
) -> tuple[Problem, dict]:
    r"""Defines a test forest problem that has TensorConstants and TensorVariables.

    The problem has TensorConstants V, W and P as vectors taking values from a data file and
    TensorVariables X_n, where n is the number of units in the data, as vectors matching the constants in shape.
    The variables are binary and each variable vector X_i has one variable with the value 1 while others have value 0.
    The variable with the value 1 for each vector X_i represents the optimal plan for the corresponding unit i.
    The three objective functions f_1, f_2, f_3 represent the net present value, wood volume at the end of
    the planning period, and the profit from harvesting.
    All of the objective functions are to be maximized.
    The problem is defined as follows:

    \begin{align}
        \mbox{maximize~} & \sum_{j=1}^N\sum_{i \in I_j} v_{ij} x_{ij} & \\
        \mbox{maximize~} & \sum_{j=1}^N\sum_{i \in I_j} w_{ij} x_{ij} & \\
        \mbox{maximize~} & \sum_{j=1}^N\sum_{i \in I_j} p_{ij} x_{ij} & \\
        \nonumber\\
        \mbox{subject to~} &  \sum\limits_{i \in I_j} x_{ij} = 1, & \forall j = 1 \ldots N \\
        & x_{ij}\in \{0,1\}& \forall j = 1 \ldots N, ~\forall i\in I_j,
    \end{align}

    where $x_{ij}$ are decision variables representing the choice of implementing management plan $i$ in stand $j$,
    and $I_j$ is the set of available management plans for stand $j$. For each plan $i$ in stand $j$
    the net present value, wood volume at the end of the planning period, and the profit from harvesting
    are represented by $v_{ij}$, $w_{ij}$, and $p_{ij}$ respectively.

    Args:
        data_dir (str): The directory of the data. Has to contain files 'alternatives.csv', 'alternatives_key.csv',
            'carbon.json' and 'dec_vars.json'. 'carbon.json' has the CO2 tons computed for each stand as a dict,
            'dec_vars.json' has the reference solution's decision variable values as a list.
        problem_name (str, optional): The name of the problem. Defalts to 'Forest problem'.
        holding (int, optional): The number of the holding to be optimized. Defaults to 1.

    Returns:
        Problem: An instance of the test forest problem.
    """
    schedule_dict = {}

    # This can be 1, 2, 3, 4 or 5. It represents %
    discounting = [
        (1 - 0.01 * discounting_factor) ** 2,
        (1 - 0.01 * discounting_factor) ** 7,
        (1 - 0.01 * discounting_factor) ** 17,
    ]

    df = pl.read_csv(
        Path(f"{data_dir}/alternatives.csv"), schema_overrides={"unit": pl.Float64}, infer_schema_length=1000
    )
    separator = ","
    unfiltered_df_key = pl.read_csv(
        Path(f"{data_dir}/alternatives_key.csv"),
        schema_overrides={"unit": pl.Float64},
        separator=separator,
    )

    # Remove the decision alternatives that are not found in the filter file
    filter_df = pl.read_csv(
        Path(f"{data_dir}/filter.csv"),
        schema_overrides={"unit": pl.Float64},
        separator=separator,
    )

    df = df.join(filter_df, on=["holding", "unit", "schedule"])
    df_key = unfiltered_df_key.join(filter_df, on=["holding", "unit", "schedule"])

    # Calculate the total wood volume at the start
    selected_df_v0 = df.select(["unit", "stock_0"]).unique()
    wood_volume_0 = int(selected_df_v0["stock_0"].sum())

    selected_df_v = df.select(["unit", "schedule", f"npv_{discounting_factor}_percent"])
    unique_units = selected_df_v.unique(["unit"], maintain_order=True).get_column("unit")
    selected_df_v.group_by(["unit", "schedule"])
    rows_by_key = selected_df_v.rows_by_key(key=["unit", "schedule"])
    v_array = np.zeros((selected_df_v["unit"].n_unique(), selected_df_v["schedule"].n_unique()))
    for i in range(np.shape(v_array)[0]):
        for j in range(np.shape(v_array)[1]):
            if (unique_units[i], j) in rows_by_key:
                v_array[i][j] = rows_by_key[(unique_units[i], j)][0][0]

    selected_df_w = df.select(["unit", "schedule", "stock_20"])
    selected_df_w.group_by(["unit", "schedule"])
    rows_by_key = selected_df_w.rows_by_key(key=["unit", "schedule"])
    selected_df_key_w = df_key.select(["unit", "schedule", "treatment"])
    selected_df_key_w.group_by(["unit", "schedule"])
    rows_by_key_df_key = selected_df_key_w.rows_by_key(key=["unit", "schedule"])
    w_array = np.zeros((selected_df_w["unit"].n_unique(), selected_df_w["schedule"].n_unique()))
    for i in range(np.shape(w_array)[0]):
        for j in range(np.shape(w_array)[1]):
            if len(rows_by_key_df_key[(unique_units[i], j)]) == 0:
                continue
            if (unique_units[i], j) in rows_by_key:
                w_array[i][j] = rows_by_key[(unique_units[i], j)][0][0]

    """
    selected_df_p = df.filter(pl.col("holding") == holding).select(
        ["unit", "schedule", "harvest_value_period_2025", "harvest_value_period_2030", "harvest_value_period_2035"]
    )
    selected_df_p.group_by(["unit", "schedule"])
    rows_by_key = selected_df_p.rows_by_key(key=["unit", "schedule"])
    p_array = np.zeros((selected_df_p["unit"].n_unique(), selected_df_p["schedule"].n_unique()))
    discounting = [0.95**5, 0.95**10, 0.95**15]
    for i in range(np.shape(p_array)[0]):
        for j in range(np.shape(p_array)[1]):
            if (unique_units[i], j) in rows_by_key:
                p_array[i][j] = (
                    sum(x * y for x, y in zip(rows_by_key[(unique_units[i], j)][0], discounting, strict=True)) + 1e-6
                )  # the 1E-6 is to deal with an annoying corner case, don't worry about it
                v_array[i][j] += p_array[i][j]
    """

    selected_df_p1 = df.select(["unit", "schedule", "harvest_value_5"])
    selected_df_p1.group_by(["unit", "schedule"])
    rows_by_key = selected_df_p1.rows_by_key(key=["unit", "schedule"])
    p1_array = np.zeros((selected_df_p1["unit"].n_unique(), selected_df_p1["schedule"].n_unique()))
    for i in range(np.shape(p1_array)[0]):
        for j in range(np.shape(p1_array)[1]):
            if (unique_units[i], j) in rows_by_key:
                p1_array[i][j] = rows_by_key[(unique_units[i], j)][0][0] + 1e-6

    selected_df_p2 = df.select(["unit", "schedule", "harvest_value_10"])
    selected_df_p2.group_by(["unit", "schedule"])
    rows_by_key = selected_df_p2.rows_by_key(key=["unit", "schedule"])
    p2_array = np.zeros((selected_df_p2["unit"].n_unique(), selected_df_p2["schedule"].n_unique()))
    for i in range(np.shape(p2_array)[0]):
        for j in range(np.shape(p2_array)[1]):
            if (unique_units[i], j) in rows_by_key:
                p2_array[i][j] = rows_by_key[(unique_units[i], j)][0][0] + 1e-6

    selected_df_p3 = df.select(["unit", "schedule", "harvest_value_20"])
    selected_df_p3.group_by(["unit", "schedule"])
    rows_by_key = selected_df_p3.rows_by_key(key=["unit", "schedule"])
    p3_array = np.zeros((selected_df_p3["unit"].n_unique(), selected_df_p3["schedule"].n_unique()))
    for i in range(np.shape(p3_array)[0]):
        for j in range(np.shape(p3_array)[1]):
            if (unique_units[i], j) in rows_by_key:
                p3_array[i][j] = rows_by_key[(unique_units[i], j)][0][0] + 1e-6

    constants = []
    variables = []
    constraints = []
    f_1_func = []
    f_2_func = []
    p1_func = []
    p2_func = []
    p3_func = []
    # define the constants V, W and P, decision variable X, constraints, and objective function expressions in one loop
    for i in range(np.shape(v_array)[0]):
        # Constants V, W and P
        v = TensorConstant(
            name=f"V_{i+1}",
            symbol=f"V_{i+1}",
            shape=[np.shape(v_array)[1]],  # NOTE: vectors have to be of form [2] instead of [2,1] or [1,2]
            values=v_array[i].tolist(),
        )
        constants.append(v)
        w = TensorConstant(
            name=f"W_{i+1}",
            symbol=f"W_{i+1}",
            shape=[np.shape(w_array)[1]],  # NOTE: vectors have to be of form [2] instead of [2,1] or [1,2]
            values=w_array[i].tolist(),
        )
        constants.append(w)
        p1 = TensorConstant(
            name=f"P1_{i+1}",
            symbol=f"P1_{i+1}",
            shape=[np.shape(p1_array)[1]],  # NOTE: vectors have to be of form [2] instead of [2,1] or [1,2]
            values=p1_array[i].tolist(),
        )
        constants.append(p1)

        p2 = TensorConstant(
            name=f"P2_{i+1}",
            symbol=f"P2_{i+1}",
            shape=[np.shape(p2_array)[1]],  # NOTE: vectors have to be of form [2] instead of [2,1] or [1,2]
            values=p2_array[i].tolist(),
        )
        constants.append(p2)

        p3 = TensorConstant(
            name=f"P3_{i+1}",
            symbol=f"P3_{i+1}",
            shape=[np.shape(p3_array)[1]],  # NOTE: vectors have to be of form [2] instead of [2,1] or [1,2]
            values=p3_array[i].tolist(),
        )
        constants.append(p3)

        # Decision variable X
        x = TensorVariable(
            name=f"X_{i+1}",
            symbol=f"X_{i+1}",
            variable_type=VariableTypeEnum.binary,
            shape=[np.shape(v_array)[1]],  # NOTE: vectors have to be of form [2] instead of [2,1] or [1,2]
            lowerbounds=np.shape(v_array)[1] * [0],
            upperbounds=np.shape(v_array)[1] * [1],
            initial_values=np.shape(v_array)[1] * [0],
        )
        variables.append(x)

        # Fill out the dict with information about treatments associated with X_{i+1}
        treatment_list = unfiltered_df_key.filter(pl.col("unit") == unique_units[i]).get_column("treatment").to_list()
        schedule_dict[f"X_{i+1}"] = dict(zip(range(len(treatment_list)), treatment_list, strict=True))
        schedule_dict[f"X_{i+1}"]["unit"] = unique_units[i]

        # Constraints
        con = Constraint(
            name=f"x_con_{i+1}",
            symbol=f"x_con_{i+1}",
            cons_type=ConstraintTypeEnum.EQ,
            func=f"Sum(X_{i+1}) - 1",
            is_twice_differentiable=True,
        )
        constraints.append(con)

        # Objective function expressions
        exprs = f"V_{i+1}@X_{i+1}"
        f_1_func.append(exprs)

        exprs = f"W_{i+1}@X_{i+1}"
        f_2_func.append(exprs)

        exprs = f"P1_{i+1}@X_{i+1}"
        p1_func.append(exprs)

        exprs = f"P2_{i+1}@X_{i+1}"
        p2_func.append(exprs)

        exprs = f"P3_{i+1}@X_{i+1}"
        p3_func.append(exprs)

    for i in range(1, 4):
        pvar = Variable(name=f"P_{i}", symbol=f"P_{i}", variable_type=VariableTypeEnum.real, lowerbound=0)
        variables.append(pvar)

    vvar = Variable(name="V_end", symbol="V_end", variable_type=VariableTypeEnum.real, lowerbound=0)
    variables.append(vvar)

    # get the remainder value of the forest into decision variable V_end
    v_func = "V_end - " + " - ".join(f_1_func)
    con = Constraint(
        name="v_con", symbol="v_con", cons_type=ConstraintTypeEnum.EQ, func=v_func, is_twice_differentiable=True
    )
    constraints.append(con)

    # These are here, so that we can get the harvesting incomes into decision variables P_i
    p1_func = "P_1 - " + " - ".join(p1_func)
    con = Constraint(
        name="p1_con", symbol="p1_con", cons_type=ConstraintTypeEnum.EQ, func=p1_func, is_twice_differentiable=True
    )
    constraints.append(con)

    p2_func = "P_2 - " + " - ".join(p2_func)
    con = Constraint(
        name="p2_con", symbol="p2_con", cons_type=ConstraintTypeEnum.EQ, func=p2_func, is_twice_differentiable=True
    )
    constraints.append(con)

    p3_func = "P_3 - " + " - ".join(p3_func)
    con = Constraint(
        name="p3_con", symbol="p3_con", cons_type=ConstraintTypeEnum.EQ, func=p3_func, is_twice_differentiable=True
    )
    constraints.append(con)

    # print(v_func)
    # print(p1_func)
    # print(p2_func)
    # print(p3_func)

    with Path.open(f"{data_dir}/carbon.json", "r") as f:
        carbon_dict = json.load(f)

    for i in range(np.shape(v_array)[0]):
        # store the carbon tons of the stand from each year as TensorConstants whose values are a list
        # of carbon tons with each schedule

        # need to add some code for this to work with the filter file, let's hope this works
        plans = filter_df.filter(pl.col("unit") == unique_units[i]).select("schedule").to_series().to_list()

        values = [carbon_dict[str(unique_units[i])]["0"][j] for j in plans]
        values = np.pad(values, (0, x.shape[0] - len(values)), mode="constant", constant_values=0).tolist()
        c0 = TensorConstant(
            name=f"C0_{i+1}",
            symbol=f"C0_{i+1}",
            shape=x.shape,  # NOTE: vectors have to be of form [2] instead of [2,1] or [1,2]
            values=values,
        )
        constants.append(c0)

        values = [carbon_dict[str(unique_units[i])]["5"][j] for j in plans]
        values = np.pad(values, (0, x.shape[0] - len(values)), mode="constant", constant_values=0).tolist()
        c5 = TensorConstant(
            name=f"C5_{i+1}",
            symbol=f"C5_{i+1}",
            shape=x.shape,  # NOTE: vectors have to be of form [2] instead of [2,1] or [1,2]
            values=values,
        )
        constants.append(c5)

        values = [carbon_dict[str(unique_units[i])]["10"][j] for j in plans]
        values = np.pad(values, (0, x.shape[0] - len(values)), mode="constant", constant_values=0).tolist()
        c10 = TensorConstant(
            name=f"C10_{i+1}",
            symbol=f"C10_{i+1}",
            shape=x.shape,  # NOTE: vectors have to be of form [2] instead of [2,1] or [1,2]
            values=values,
        )
        constants.append(c10)

        values = [carbon_dict[str(unique_units[i])]["20"][j] for j in plans]
        values = np.pad(values, (0, x.shape[0] - len(values)), mode="constant", constant_values=0).tolist()
        c20 = TensorConstant(
            name=f"C20_{i+1}",
            symbol=f"C20_{i+1}",
            shape=x.shape,  # NOTE: vectors have to be of form [2] instead of [2,1] or [1,2]
            values=values,
        )
        constants.append(c20)

    if Path(f"{data_dir}/dec_vars.json").is_file():
        with Path.open(f"{data_dir}/dec_vars.json", "r") as f:
            CO2_baseline = json.load(f)  # noqa: N806
    else:
        CO2_baseline = None  # noqa: N806

    c1_func = []
    c2_func = []
    c3_func = []
    for i in range(np.shape(v_array)[0]):
        # if given a baseline (e.g., some previous solution's decision variables) basically maximize the difference
        if CO2_baseline is not None:
            # no lists allowed in the parser as such so we need to make the baseline into constants
            b = TensorConstant(
                name=f"B_{i+1}",
                symbol=f"B_{i+1}",
                shape=[len(CO2_baseline[i])],  # NOTE: vectors have to be of form [2] instead of [2,1] or [1,2]
                values=CO2_baseline[i],
            )
            constants.append(b)
            c1_func.append(f"((C5_{i+1}@X_{i+1}) - (C5_{i+1}@B_{i+1}))*5")
            c2_func.append(f"((C10_{i+1}@X_{i+1}) - (C10_{i+1}@B_{i+1}))*10")
            c3_func.append(f"((C20_{i+1}@X_{i+1}) - (C20_{i+1}@B_{i+1}))*5")

        # if no baseline given, use the beginning state's volumes
        else:
            print("Baseline for CO2 not found!")
            c1_func.append(f"((C5_{i+1}@X_{i+1}) - (C0_{i+1}@X_{i+1}))*5")
            c2_func.append(f"((C10_{i+1}@X_{i+1}) - (C0_{i+1}@X_{i+1}))*10")
            c3_func.append(f"((C20_{i+1}@X_{i+1}) - (C0_{i+1}@X_{i+1}))*5")

    for i in range(1, 5):
        cvar = Variable(name=f"C_{i}", symbol=f"C_{i}", variable_type=VariableTypeEnum.real)
        variables.append(cvar)

    # These are here, so that we can get the carbon storage into decision variables C_i
    c1_func = "C_1 - " + " - ".join(c1_func)
    con = Constraint(
        name="c1_con", symbol="c1_con", cons_type=ConstraintTypeEnum.EQ, func=c1_func, is_twice_differentiable=True
    )
    constraints.append(con)

    c2_func = "C_2 - " + " - ".join(c2_func)
    con = Constraint(
        name="c2_con", symbol="c2_con", cons_type=ConstraintTypeEnum.EQ, func=c2_func, is_twice_differentiable=True
    )
    constraints.append(con)

    c3_func = "C_3 - " + " - ".join(c3_func)
    con = Constraint(
        name="c3_con", symbol="c3_con", cons_type=ConstraintTypeEnum.EQ, func=c3_func, is_twice_differentiable=True
    )
    constraints.append(con)

    # Add a constraint stating that we cannot go below reference plan in carbon storage
    co2_func = "- (C_1 + C_2 + C_3 + 1E-3)"
    con = Constraint(
        name="cmin_con",
        symbol="cmin_con",
        cons_type=ConstraintTypeEnum.LTE,
        func=co2_func,
        is_twice_differentiable=True,
    )
    constraints.append(con)

    # form the objective function sums
    f_2_func = " + ".join(f_2_func)
    if compensation == 0:
        f_3_func = f"{discounting[0]} * P_1 + {discounting[1]} * P_2 + {discounting[2]} * P_3"
    else:
        f_3_func = (
            f"{discounting[0]} * (P_1 + {compensation}*C_1) + "
            + f"{discounting[1]} * (P_2 + {compensation}*C_2) + "
            + f"{discounting[2]} * (P_3 + {compensation}*C_3)"
        )
    f_1_func = "V_end + " + f_3_func
    f_4_func = "C_1 + C_2 + C_3"

    # print(f_1_func)
    # print(f_2_func)
    # print(f_3_func)

    f_1 = Objective(
        name="Net present value",
        symbol="f_1",
        func=f_1_func,
        maximize=True,
        objective_type=ObjectiveTypeEnum.analytical,
        is_linear=True,
        is_convex=False,  # not checked
        is_twice_differentiable=True,
    )

    f_2 = Objective(
        name="Wood stock volume",
        symbol="f_2",
        func=f_2_func,
        maximize=True,
        objective_type=ObjectiveTypeEnum.analytical,
        is_linear=True,
        is_convex=False,  # not checked
        is_twice_differentiable=True,
    )

    f_3 = Objective(
        name="Harvest value",
        symbol="f_3",
        func=f_3_func,
        maximize=True,
        objective_type=ObjectiveTypeEnum.analytical,
        is_linear=True,
        is_convex=False,  # not checked
        is_twice_differentiable=True,
    )

    f_4 = Objective(
        name="Carbon storage",
        symbol="f_4",
        func=f_4_func,
        maximize=True,
        objective_type=ObjectiveTypeEnum.analytical,
        is_linear=True,
        is_convex=False,  # not checked
        is_twice_differentiable=True,
    )

    # This is so bad, but we currently don't have a better way
    ideals, nadirs = payoff_table_method(
        problem=Problem(
            name=problem_name,
            description="A test forest problem.",
            constants=constants,
            variables=variables,
            objectives=[f_1, f_2, f_3, f_4],
            constraints=constraints,
        ),
        solver=available_solvers["gurobipy"],
    )

    print(ideals)
    print(nadirs)

    f_1 = Objective(
        name="Nettonykyarvo / €",
        symbol="f_1",
        func=f_1_func,
        maximize=True,
        ideal=math.ceil(ideals["f_1"]),
        nadir=math.floor(nadirs["f_1"]),
        objective_type=ObjectiveTypeEnum.analytical,
        is_linear=True,
        is_convex=False,  # not checked
        is_twice_differentiable=True,
    )

    f_2 = Objective(
        name=f"Puuston tilavuus / m^3\n(alussa {wood_volume_0}m^3)",
        symbol="f_2",
        func=f_2_func,
        maximize=True,
        ideal=math.ceil(ideals["f_2"]),
        nadir=math.floor(nadirs["f_2"]),
        objective_type=ObjectiveTypeEnum.analytical,
        is_linear=True,
        is_convex=False,  # not checked
        is_twice_differentiable=True,
    )

    f_3 = Objective(
        name="Hakkuiden tuotto / €",
        symbol="f_3",
        func=f_3_func,
        maximize=True,
        ideal=math.ceil(ideals["f_3"]),
        nadir=math.floor(nadirs["f_3"]),
        objective_type=ObjectiveTypeEnum.analytical,
        is_linear=True,
        is_convex=False,  # not checked
        is_twice_differentiable=True,
    )

    f_4 = Objective(
        name="Sidottu hiilidioksidi / (v·t)",
        symbol="f_4",
        func=f_4_func,
        maximize=True,
        ideal=math.ceil(ideals["f_4"]),
        nadir=0.0,
        objective_type=ObjectiveTypeEnum.analytical,
        is_linear=True,
        is_convex=False,  # not checked
        is_twice_differentiable=True,
    )

    return Problem(
        name=problem_name,
        description="A test forest problem.",
        constants=constants,
        variables=variables,
        objectives=[f_1, f_2, f_3, f_4],
        constraints=constraints,
    ), schedule_dict
