"""District heating optimization: electric/gas boilers with thermal storage.

Implements a bi-objective MILP for scheduling an electric boiler, a gas boiler,
and a thermal storage tank to satisfy a constant 1 MW heat demand.  Corresponds
to "Scenario 3" in the original code1.py study.

Typical use::

    from pathlib import Path
    from district_heating.district_heating import load_district_heating_data, district_heating_problem

    spot, gas = load_district_heating_data(Path("/home/juho/data/jenni"), year=2022)
    # Use first week (168 hours) to keep the MILP tractable
    problem = district_heating_problem(spot[:168], gas[:168])
"""

# ruff: noqa: PLR2004

from pathlib import Path

import numpy as np
import pandas as pd

from desdeo.problem.schema import (
    Constraint,
    ConstraintTypeEnum,
    Objective,
    Problem,
    TensorConstant,
    TensorVariable,
    VariableTypeEnum,
)

# ── Physical parameters ───────────────────────────────────────────────────────
DEMAND_MW = 1.0
ELEC_BOILER_CAPACITY_MW = 2.0
GAS_BOILER_CAPACITY_MW = 1.0
ELEC_BOILER_EFF = 0.95
GAS_BOILER_EFF = 0.8
STORAGE_CAPACITY_MWH = 5.0
MAX_CHARGE_POWER_MW = 2.0
MAX_DISCHARGE_POWER_MW = 2.0
HEAT_LOSS_FRACTION = 0.001
CHARGING_EFF = 0.95
DISCHARGING_EFF = 0.95

# ── Price / fee parameters (EUR/MWh) ─────────────────────────────────────────
DISTRIBUTION_FEE_PEAK = 12.8
DISTRIBUTION_FEE_OFF_PEAK = 4.4
ENERGY_CONTENT_TAX = 0.63
GAS_ENERGY_TAX = 23.354

# ── Derived compound constants ────────────────────────────────────────────────
_ELEC_STEAM_MAX = DEMAND_MW / ELEC_BOILER_EFF  # max electric input for steam
_ELEC_TOTAL_MAX = ELEC_BOILER_CAPACITY_MW / ELEC_BOILER_EFF  # max total electric input
_GAS_STEAM_MAX = DEMAND_MW / GAS_BOILER_EFF  # max gas input for steam
_GAS_TOTAL_MAX = GAS_BOILER_CAPACITY_MW / GAS_BOILER_EFF  # max total gas input
_ELEC_STORE_EFF = ELEC_BOILER_EFF * CHARGING_EFF  # net storage efficiency (electric)
_GAS_STORE_EFF = GAS_BOILER_EFF * CHARGING_EFF  # net storage efficiency (gas)
_DISCHARGE_LOSS = 1.0 / DISCHARGING_EFF  # multiplier for storage losses on discharge
_HEAT_RETENTION = 1.0 - HEAT_LOSS_FRACTION  # fraction of heat retained per hour


def _distribution_fee(time_local: pd.Timestamp) -> float:
    is_winter = time_local.month in {12, 1, 2}
    is_weekday = time_local.weekday() < 5
    is_daytime = 7 <= time_local.hour < 21
    if is_winter and is_weekday and is_daytime:
        return DISTRIBUTION_FEE_PEAK
    return DISTRIBUTION_FEE_OFF_PEAK


def load_district_heating_data(data_dir: Path | str, year: int) -> tuple[np.ndarray, np.ndarray]:
    """Load hourly all-in electricity and gas price arrays for a given year.

    Reads the ENTSO-E day-ahead electricity price file and the monthly gas price
    file from *data_dir*.  Adds distribution fees, energy-content tax, and gas
    energy tax so the returned values represent all-in variable costs:

    * electricity: day-ahead spot + distribution fee + energy content tax
    * gas:         gas commodity price (LHV basis) + gas energy tax

    2025 electricity data is in 15-minute resolution and is resampled to hourly
    by averaging.

    Args:
        data_dir: directory containing ``Electricity_price_FI_<year>.xlsx`` and
                  ``Gas_Prices.xlsx`` (the Jenni dataset).
        year:     calendar year to extract (2019-2025).

    Returns:
        Tuple ``(spot_prices, gas_prices)`` of NumPy float64 arrays each with
        shape ``(T,)`` where *T* is the number of hours in *year*.
    """
    data_dir = Path(data_dir)

    # ── Electricity prices ────────────────────────────────────────────────────
    df_el = pd.read_excel(data_dir / f"Electricity_price_FI_{year}.xlsx")
    df_el["Date"] = pd.to_datetime(df_el["MTU (UTC)"].str.split(" - ").str[0], format="%d/%m/%Y %H:%M:%S")
    df_el = df_el[["Date", "Day-ahead Price (EUR/MWh)"]].copy()
    df_el["Date"] = df_el["Date"].dt.tz_localize("UTC")

    if year == 2025:  # 15-min data → hourly average
        df_el = df_el.set_index("Date").resample("h").mean().reset_index()

    df_el["Date_local"] = df_el["Date"].dt.tz_convert("Europe/Helsinki")
    df_el = df_el[df_el["Date_local"].dt.year == year].copy()
    df_el["distribution_fee"] = df_el["Date_local"].apply(_distribution_fee)
    df_el["total_price"] = df_el["Day-ahead Price (EUR/MWh)"] + df_el["distribution_fee"] + ENERGY_CONTENT_TAX
    df_el["Month"] = df_el["Date_local"].dt.strftime("%m/%Y")

    # ── Gas prices (monthly → hourly via merge) ───────────────────────────────
    df_gas = pd.read_excel(data_dir / "Gas_Prices.xlsx")
    df_gas["Month"] = pd.to_datetime(df_gas["Month"], format="%m/%Y").dt.strftime("%m/%Y")
    df_gas["total_gas_price"] = df_gas["Gas Price (EUR/MWh)"] + GAS_ENERGY_TAX
    df_gas = df_gas[["Month", "total_gas_price"]]

    df = df_el.merge(df_gas, on="Month", how="left")
    return df["total_price"].to_numpy(), df["total_gas_price"].to_numpy()


def district_heating_problem(
    spot_prices: np.ndarray,
    gas_prices: np.ndarray,
    initial_storage_level: float = 0.0,
) -> Problem:
    """Build a bi-objective MILP for district heating scheduling.

    Models the joint dispatch of an electric boiler (2 MW), a gas boiler (1 MW),
    and a 5 MWh thermal storage tank over a time horizon *T* given by the length
    of the price arrays.  The system must satisfy a constant heat demand of 1 MW
    at every hour.

    Decision variables (TensorVariables, all shape ``[T]``):

    * ``el_steam``  - electric boiler electricity input for steam production (MW)
    * ``el_charge`` - electric boiler electricity input for storage charging (MW)
    * ``gas_steam`` - gas boiler gas input for steam production (MW)
    * ``gas_charge``- gas boiler gas input for storage charging (MW)
    * ``discharge`` - heat discharged from storage (MW)
    * ``storage``   - thermal energy in storage (MWh)
    * ``u_el``      - binary: electric boiler is charging storage
    * ``u_gas``     - binary: gas boiler is charging storage
    * ``u_dis``     - binary: storage is discharging

    Constraints:

    1. **Storage dynamics t=1**: linearised heat balance at the first timestep,
       anchored at *initial_storage_level*.
    2. **Storage dynamics t=2..T**: storage[t] = retention · storage[t-1] + charged - discharged.
    3. **Heat demand balance**: total heat output equals 1 MW every hour.
    4. **Electric boiler capacity**: combined steam + charging input ≤ 2/η_el MW.
    5. **Gas boiler capacity**: combined steam + charging input ≤ 1/η_gas MW.
    6. **Mutual exclusivity**: at most one of {charge-el, charge-gas, discharge} per hour.
    7-9. **Big-M bounds** linking continuous flows to their binary indicators.

    Objectives:

    * ``f_el``  - total electricity cost (EUR), minimised.
    * ``f_gas`` - total gas cost (EUR), minimised.

    Note:
        For a full year (T = 8760) this is a large MILP (≈ 79 k constraints,
        26 k binary variables).  For interactive use, pass a week's worth of data
        (T = 168) to keep solve times manageable.

    Args:
        spot_prices: hourly all-in electricity prices (EUR/MWh), shape ``(T,)``.
        gas_prices:  hourly all-in gas prices (EUR/MWh), shape ``(T,)``.
        initial_storage_level: initial heat in storage (MWh), default 0.

    Returns:
        A DESDEO :class:`~desdeo.problem.schema.Problem` (MILP).
    """
    T = len(spot_prices)  # noqa: N806
    if T < 2:
        raise ValueError("Time horizon must be at least 2 hours.")

    # ── Variables ─────────────────────────────────────────────────────────────
    el_steam = TensorVariable(
        name="Electric boiler electricity input for steam (MW)",
        symbol="el_steam",
        shape=[T],
        variable_type=VariableTypeEnum.real,
        lowerbounds=0.0,
        upperbounds=_ELEC_STEAM_MAX,
        initial_values=0.0,
    )
    el_charge = TensorVariable(
        name="Electric boiler electricity input for storage charging (MW)",
        symbol="el_charge",
        shape=[T],
        variable_type=VariableTypeEnum.real,
        lowerbounds=0.0,
        upperbounds=_ELEC_TOTAL_MAX,
        initial_values=0.0,
    )
    gas_steam = TensorVariable(
        name="Gas boiler gas input for steam (MW)",
        symbol="gas_steam",
        shape=[T],
        variable_type=VariableTypeEnum.real,
        lowerbounds=0.0,
        upperbounds=_GAS_STEAM_MAX,
        initial_values=0.0,
    )
    gas_charge = TensorVariable(
        name="Gas boiler gas input for storage charging (MW)",
        symbol="gas_charge",
        shape=[T],
        variable_type=VariableTypeEnum.real,
        lowerbounds=0.0,
        upperbounds=_GAS_TOTAL_MAX,
        initial_values=0.0,
    )
    discharge = TensorVariable(
        name="Heat discharged from storage (MW)",
        symbol="discharge",
        shape=[T],
        variable_type=VariableTypeEnum.real,
        lowerbounds=0.0,
        upperbounds=MAX_DISCHARGE_POWER_MW,
        initial_values=0.0,
    )
    storage = TensorVariable(
        name="Thermal energy in storage (MWh)",
        symbol="storage",
        shape=[T],
        variable_type=VariableTypeEnum.real,
        lowerbounds=0.0,
        upperbounds=STORAGE_CAPACITY_MWH,
        initial_values=initial_storage_level,
    )
    u_el = TensorVariable(
        name="Electric boiler charging storage (binary indicator)",
        symbol="u_el",
        shape=[T],
        variable_type=VariableTypeEnum.binary,
        lowerbounds=0,
        upperbounds=1,
        initial_values=0,
    )
    u_gas = TensorVariable(
        name="Gas boiler charging storage (binary indicator)",
        symbol="u_gas",
        shape=[T],
        variable_type=VariableTypeEnum.binary,
        lowerbounds=0,
        upperbounds=1,
        initial_values=0,
    )
    u_dis = TensorVariable(
        name="Storage discharging (binary indicator)",
        symbol="u_dis",
        shape=[T],
        variable_type=VariableTypeEnum.binary,
        lowerbounds=0,
        upperbounds=1,
        initial_values=0,
    )

    # ── Constants ─────────────────────────────────────────────────────────────
    p_el = TensorConstant(
        name="All-in electricity price (EUR/MWh)",
        symbol="p_el",
        shape=[T],
        values=spot_prices.tolist(),
    )
    p_gas = TensorConstant(
        name="All-in gas price (EUR/MWh)",
        symbol="p_gas",
        shape=[T],
        values=gas_prices.tolist(),
    )

    # ── Constraints ───────────────────────────────────────────────────────────
    # Storage dynamics at t=1 (scalar EQ):
    #   storage[1] = retention * initial + el_charge[1]*E_eff + gas_charge[1]*G_eff
    #                - discharge[1] * discharge_loss
    # → storage[1] - retention*initial - E_eff*el_charge[1] - G_eff*gas_charge[1]
    #   + discharge_loss*discharge[1] = 0
    stor_dyn_1 = Constraint(
        name="Storage dynamics t=1",
        symbol="stor_dyn_1",
        func=[
            "Add",
            ["At", "storage", 1],
            -(_HEAT_RETENTION * initial_storage_level),
            ["Negate", ["Multiply", _ELEC_STORE_EFF, ["At", "el_charge", 1]]],
            ["Negate", ["Multiply", _GAS_STORE_EFF, ["At", "gas_charge", 1]]],
            ["Multiply", _DISCHARGE_LOSS, ["At", "discharge", 1]],
        ],
        cons_type=ConstraintTypeEnum.EQ,
        is_linear=True,
        is_convex=True,
        is_twice_differentiable=True,
    )

    # Storage dynamics t=2..T (vector EQ of length T-1):
    #   storage[t] - retention*storage[t-1] - E_eff*el_charge[t] - G_eff*gas_charge[t]
    #   + discharge_loss*discharge[t] = 0
    stor_dyn = Constraint(
        name="Storage dynamics t=2..T",
        symbol="stor_dyn",
        func=[
            "Add",
            ["Extract", "storage", ["Tuple", 2, T]],
            ["Negate", ["Multiply", _HEAT_RETENTION, ["Extract", "storage", ["Tuple", 1, T - 1]]]],
            ["Negate", ["Multiply", _ELEC_STORE_EFF, ["Extract", "el_charge", ["Tuple", 2, T]]]],
            ["Negate", ["Multiply", _GAS_STORE_EFF, ["Extract", "gas_charge", ["Tuple", 2, T]]]],
            ["Multiply", _DISCHARGE_LOSS, ["Extract", "discharge", ["Tuple", 2, T]]],
        ],
        cons_type=ConstraintTypeEnum.EQ,
        is_linear=True,
        is_convex=True,
        is_twice_differentiable=True,
    )

    # Heat demand balance (vector EQ of length T):
    #   el_steam * ELEC_EFF + gas_steam * GAS_EFF + discharge = DEMAND
    # → ELEC_EFF*el_steam + GAS_EFF*gas_steam + discharge - DEMAND = 0
    demand_bal = Constraint(
        name="Heat demand balance",
        symbol="demand_bal",
        func=[
            "Add",
            ["Multiply", ELEC_BOILER_EFF, "el_steam"],
            ["Multiply", GAS_BOILER_EFF, "gas_steam"],
            "discharge",
            -DEMAND_MW,
        ],
        cons_type=ConstraintTypeEnum.EQ,
        is_linear=True,
        is_convex=True,
        is_twice_differentiable=True,
    )

    # Electric boiler total capacity (vector LTE):
    #   el_steam + el_charge <= ELEC_BOILER_CAPACITY / ELEC_EFF
    el_cap = Constraint(
        name="Electric boiler total capacity",
        symbol="el_cap",
        func=["Add", "el_steam", "el_charge", -_ELEC_TOTAL_MAX],
        cons_type=ConstraintTypeEnum.LTE,
        is_linear=True,
        is_convex=True,
        is_twice_differentiable=True,
    )

    # Gas boiler total capacity (vector LTE):
    #   gas_steam + gas_charge <= GAS_BOILER_CAPACITY / GAS_EFF
    gas_cap = Constraint(
        name="Gas boiler total capacity",
        symbol="gas_cap",
        func=["Add", "gas_steam", "gas_charge", -_GAS_TOTAL_MAX],
        cons_type=ConstraintTypeEnum.LTE,
        is_linear=True,
        is_convex=True,
        is_twice_differentiable=True,
    )

    # Mutual exclusivity: at most one of {charge-el, charge-gas, discharge} per hour (vector LTE):
    #   u_el + u_gas + u_dis <= 1
    mutex = Constraint(
        name="Mutual exclusivity of charge/discharge modes",
        symbol="mutex",
        func=["Add", "u_el", "u_gas", "u_dis", -1.0],
        cons_type=ConstraintTypeEnum.LTE,
        is_linear=True,
        is_convex=True,
        is_twice_differentiable=True,
    )

    # Big-M: electric charging only active when u_el=1 (vector LTE):
    #   el_charge * ELEC_EFF <= MAX_CHARGE * u_el
    el_bigm = Constraint(
        name="Electric charging big-M",
        symbol="el_bigm",
        func=[
            "Add",
            ["Multiply", ELEC_BOILER_EFF, "el_charge"],
            ["Negate", ["Multiply", MAX_CHARGE_POWER_MW, "u_el"]],
        ],
        cons_type=ConstraintTypeEnum.LTE,
        is_linear=True,
        is_convex=True,
        is_twice_differentiable=True,
    )

    # Big-M: gas charging only active when u_gas=1 (vector LTE):
    #   gas_charge * GAS_EFF <= MAX_CHARGE * u_gas
    gas_bigm = Constraint(
        name="Gas charging big-M",
        symbol="gas_bigm",
        func=[
            "Add",
            ["Multiply", GAS_BOILER_EFF, "gas_charge"],
            ["Negate", ["Multiply", MAX_CHARGE_POWER_MW, "u_gas"]],
        ],
        cons_type=ConstraintTypeEnum.LTE,
        is_linear=True,
        is_convex=True,
        is_twice_differentiable=True,
    )

    # Big-M: discharge only active when u_dis=1 (vector LTE):
    #   discharge <= DEMAND * u_dis
    dis_bigm = Constraint(
        name="Discharge big-M",
        symbol="dis_bigm",
        func=["Add", "discharge", ["Negate", ["Multiply", DEMAND_MW, "u_dis"]]],
        cons_type=ConstraintTypeEnum.LTE,
        is_linear=True,
        is_convex=True,
        is_twice_differentiable=True,
    )

    # ── Objectives ────────────────────────────────────────────────────────────
    objectives = [
        Objective(
            name="Electricity cost",
            description="Total electricity cost: sum of price × (steam + charging) input",  # noqa: RUF001
            symbol="f_el",
            func=["Add", ["MatMul", "p_el", "el_steam"], ["MatMul", "p_el", "el_charge"]],
            unit="EUR",
            maximize=False,
            is_linear=True,
            is_convex=True,
            is_twice_differentiable=True,
        ),
        Objective(
            name="Gas cost",
            description="Total gas cost: sum of price × (steam + charging) input",  # noqa: RUF001
            symbol="f_gas",
            func=["Add", ["MatMul", "p_gas", "gas_steam"], ["MatMul", "p_gas", "gas_charge"]],
            unit="EUR",
            maximize=False,
            is_linear=True,
            is_convex=True,
            is_twice_differentiable=True,
        ),
    ]

    return Problem(
        name="District heating: electric/gas boilers with thermal storage",
        description=(
            "Bi-objective MILP for scheduling a 2 MW electric boiler, a 1 MW gas boiler, "
            "and a 5 MWh thermal storage tank to satisfy a constant 1 MW heat demand. "
            "Minimise electricity cost (f_el) and gas cost (f_gas) independently. "
            "Binary indicators enforce that at most one of {charge-electric, charge-gas, "
            "discharge} is active at any hour."
        ),
        variables=[
            el_steam,
            el_charge,
            gas_steam,
            gas_charge,
            discharge,
            storage,
            u_el,
            u_gas,
            u_dis,
        ],
        constants=[p_el, p_gas],
        objectives=objectives,
        constraints=[
            stor_dyn_1,
            stor_dyn,
            demand_bal,
            el_cap,
            gas_cap,
            mutex,
            el_bigm,
            gas_bigm,
            dis_bigm,
        ],
    )
