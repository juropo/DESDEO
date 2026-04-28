"""Example problem optimizing electricity use at a summer cabin with uncertain temperature and usage patterns."""

# ruff: noqa: PLR2004, N806
from pathlib import Path

import numpy as np
import pandas as pd

from desdeo.problem.scenario import Scenario, ScenarioModel
from desdeo.problem.schema import (
    Constraint,
    ConstraintTypeEnum,
    Objective,
    Problem,
    TensorConstant,
    TensorVariable,
    Variable,
    VariableTypeEnum,
)

_PRICES_PATH = Path(__file__).parents[3] / "datasets" / "prices_summer.npz"


def generate_summer_cabin_electricity_data():  # noqa: C901
    """Generates synthetic hourly electricity load and temperature data for a summer cabin from June 1 to August 31."""
    rng = np.random.default_rng(6969)

    time_index = pd.date_range("2025-06-01", "2025-08-31 23:00", freq="H")

    # --- Temperature model ---
    def seasonal_temp(ts):
        match ts.month:
            case 6:
                return 14
            case 7:
                return 18
            case _:
                return 15

    def daily_temp(hour):
        # peak at 15:00, trough at 04:00
        return 5 * np.sin((hour - 15) / 24 * 2 * np.pi)

    def temperature(ts):
        return seasonal_temp(ts) + daily_temp(ts.hour) + rng.normal(0, 1.5)

    # --- Heating ---
    def heating_load(temp):
        threshold = 18
        k = 0.2
        return max(0, (threshold - temp) * k)

    # --- Base load ---
    def base_load():
        return rng.uniform(0.2, 0.35)

    # --- Daily pattern ---
    def daily_usage(hour):
        if hour < 6:
            return 0.1
        if hour < 9:
            return 0.8
        if hour < 17:
            return 0.4
        if hour < 22:
            return 1.0
        return 0.3

    # --- Cooking spikes ---
    def cooking_spike(ts):
        if ts.hour in [7, 12, 18] and rng.random() < 0.5:
            return rng.uniform(0.8, 1.8)
        return 0

    # --- Sauna ---
    def sauna_spike(ts):
        if ts.weekday() in [4, 5, 6]:  # Fri-Sun  # noqa: SIM102
            if 18 <= ts.hour <= 21 and rng.random() < 0.25:
                return rng.uniform(5, 8)
        return 0

    loads = []
    temps = []

    for ts in time_index:
        temps.append(temperature(ts))

        load = 0
        load += base_load()
        load += daily_usage(ts.hour)
        load += heating_load(temperature(ts))
        load += cooking_spike(ts)
        # load += sauna_spike(ts) # no one uses electric sauna in the summer cabin
        load += rng.normal(0, 0.05)

        loads.append(max(load, 0.15))

    prices_eur_kwh = np.load(_PRICES_PATH)["prices"] / 1000.0

    return pd.DataFrame(
        {"load_kWh": loads, "temperature_C": temps, "price_EUR_kWh": prices_eur_kwh},
        index=time_index,
    )


def generate_solar_profile() -> np.ndarray:
    """Generate hourly solar production per 160 W panel from June 1 to August 31.

    Uses a clear-sky sine curve based on approximate sunrise/sunset times for 60°N latitude,
    scaled by a daily cloudiness factor sampled uniformly from [0.2, 1.0] (up to 80% reduction).
    On a clear summer day the panel produces its nominal 0.16 kWh/h at solar noon.

    Returns:
        A numpy array of shape (2208,) with kWh output per panel per hour.
    """
    rng = np.random.default_rng(42)
    time_index = pd.date_range("2025-06-01", "2025-08-31 23:00", freq="h")
    panel_peak_kw = 0.16  # kW nominal per panel
    solar_noon = 13.0  # hour of peak irradiance (local time)
    lat = np.radians(60.0)

    days = pd.date_range("2025-06-01", "2025-08-31", freq="D")
    cloudiness = {d.date(): rng.uniform(0.2, 1.0) for d in days}

    profile = []
    for ts in time_index:
        doy = ts.day_of_year
        dec = np.radians(23.45 * np.sin(np.radians(360.0 / 365.0 * (doy - 80))))
        cos_ha = np.clip(-np.tan(lat) * np.tan(dec), -1.0, 1.0)
        day_len = 2.0 * np.degrees(np.arccos(cos_ha)) / 15.0
        sunrise = solar_noon - day_len / 2.0
        sunset = solar_noon + day_len / 2.0

        hour = ts.hour
        if sunrise <= hour <= sunset:
            clear_sky = panel_peak_kw * np.sin(np.pi * (hour - sunrise) / (sunset - sunrise))
        else:
            clear_sky = 0.0

        profile.append(max(0.0, clear_sky * cloudiness[ts.date()]))

    return np.array(profile)


def summer_cabin_battery_problem(initial_soc: float = 0.0, n_panels_max: int = 50) -> "Problem":
    """Build a bi-objective MILP for battery + solar investment and scheduling at the summer cabin.

    Decision variables:
    - y ∈ {0,1}: whether the battery is installed
    - E ∈ [0, 42] kWh: battery capacity (14-42 kWh if installed, 0 otherwise)
    - n ∈ {0,...,n_panels_max}: number of 160 W solar panels
    - c_t, d_t ∈ [0, 10] kW: hourly charge/discharge rate
    - soc_t ∈ [0, 42] kWh: state of charge
    - buy_t ≥ 0: electricity purchased from the grid (kWh/h)
    - sell_t ≥ 0: electricity sold to the grid (kWh/h)

    Objectives:
    - f_1: total electricity cost (EUR) = Σ q_t·buy_t - Σ p_t·sell_t
      where q = p + 0.05 EUR/kWh (spot + transmission) for buying,
      and p is the spot price for selling (no transmission).
    - f_2: total investment cost (EUR) = 2000·y + 310·E + 200·n

    Args:
        initial_soc: initial state of charge in kWh. Defaults to 0.0 (empty).
        n_panels_max: upper bound on number of solar panels. Defaults to 50.

    Returns:
        A DESDEO Problem instance.
    """
    df = generate_summer_cabin_electricity_data()
    prices = df["price_EUR_kWh"].to_numpy()
    loads = df["load_kWh"].to_numpy()
    solar = generate_solar_profile()
    T = len(prices)

    # --- Variables ---
    installed = Variable(
        name="Battery installed",
        symbol="y",
        variable_type=VariableTypeEnum.binary,
        lowerbound=0,
        upperbound=1,
        initial_value=0,
    )
    capacity = Variable(
        name="Battery capacity (kWh)",
        symbol="E",
        variable_type=VariableTypeEnum.real,
        lowerbound=0.0,
        upperbound=42.0,
        initial_value=0.0,
    )
    n_panels = Variable(
        name="Number of solar panels",
        symbol="n",
        variable_type=VariableTypeEnum.integer,
        lowerbound=0,
        upperbound=n_panels_max,
        initial_value=0,
    )
    charge = TensorVariable(
        name="Charging power (kW)",
        symbol="c",
        shape=[T],
        variable_type=VariableTypeEnum.real,
        lowerbounds=0.0,
        upperbounds=10.0,
        initial_values=0.0,
    )
    discharge = TensorVariable(
        name="Discharging power (kW)",
        symbol="d",
        shape=[T],
        variable_type=VariableTypeEnum.real,
        lowerbounds=0.0,
        upperbounds=10.0,
        initial_values=0.0,
    )
    soc = TensorVariable(
        name="State of charge (kWh)",
        symbol="soc",
        shape=[T],
        variable_type=VariableTypeEnum.real,
        lowerbounds=0.0,
        upperbounds=42.0,
        initial_values=initial_soc,
    )
    buy = TensorVariable(
        name="Grid electricity purchased (kWh/h)",
        symbol="buy",
        shape=[T],
        variable_type=VariableTypeEnum.real,
        lowerbounds=0.0,
        upperbounds=None,
        initial_values=0.0,
    )
    sell = TensorVariable(
        name="Grid electricity sold (kWh/h)",
        symbol="sell",
        shape=[T],
        variable_type=VariableTypeEnum.real,
        lowerbounds=0.0,
        upperbounds=None,
        initial_values=0.0,
    )

    # --- Constants ---
    price_const = TensorConstant(
        name="Electricity spot price (EUR/kWh)",
        symbol="p",
        shape=[T],
        values=prices.tolist(),
    )
    load_const = TensorConstant(
        name="Electricity load (kWh/h)",
        symbol="l",
        shape=[T],
        values=loads.tolist(),
    )
    solar_const = TensorConstant(
        name="Solar production per panel (kWh/h per panel)",
        symbol="sol",
        shape=[T],
        values=solar.tolist(),
    )

    # --- Constraints ---
    constraints = [
        # Battery capacity links: E = 0 if y = 0; E in [14, 42] if y = 1
        Constraint(
            name="Capacity zero if not installed",
            symbol="cap_ub_con",
            func=["Add", "E", ["Negate", ["Multiply", 42.0, "y"]]],
            cons_type=ConstraintTypeEnum.LTE,
            is_linear=True,
            is_convex=True,
            is_twice_differentiable=True,
        ),
        Constraint(
            name="Minimum capacity if installed",
            symbol="cap_lb_con",
            func=["Add", ["Multiply", 14.0, "y"], ["Negate", "E"]],
            cons_type=ConstraintTypeEnum.LTE,
            is_linear=True,
            is_convex=True,
            is_twice_differentiable=True,
        ),
        # SOC dynamics: soc_t = soc_{t-1} + c_t - d_t
        Constraint(
            name="SOC dynamics t=1",
            symbol="soc_con_1",
            func=["Add", ["At", "soc", 1], -initial_soc, ["Negate", ["At", "c", 1]], ["At", "d", 1]],
            cons_type=ConstraintTypeEnum.EQ,
            is_linear=True,
            is_convex=True,
            is_twice_differentiable=True,
        ),
        *[
            Constraint(
                name=f"SOC dynamics t={t}",
                symbol=f"soc_con_{t}",
                func=[
                    "Add",
                    ["At", "soc", t],
                    ["Negate", ["At", "soc", t - 1]],
                    ["Negate", ["At", "c", t]],
                    ["At", "d", t],
                ],
                cons_type=ConstraintTypeEnum.EQ,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
            for t in range(2, T + 1)
        ],
        # soc_t <= E
        *[
            Constraint(
                name=f"SOC capacity upper bound t={t}",
                symbol=f"soc_cap_con_{t}",
                func=["Add", ["At", "soc", t], ["Negate", "E"]],
                cons_type=ConstraintTypeEnum.LTE,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
            for t in range(1, T + 1)
        ],
        # Energy balance: buy_t - sell_t + d_t - c_t + n*sol_t = l_t
        *[
            Constraint(
                name=f"Energy balance t={t}",
                symbol=f"energy_bal_{t}",
                func=[
                    "Add",
                    ["At", "buy", t],
                    ["Negate", ["At", "sell", t]],
                    ["At", "d", t],
                    ["Negate", ["At", "c", t]],
                    ["Multiply", "n", ["At", "sol", t]],
                    ["Negate", ["At", "l", t]],
                ],
                cons_type=ConstraintTypeEnum.EQ,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
            for t in range(1, T + 1)
        ],
    ]

    return Problem(
        name="Summer cabin battery and solar investment optimization",
        description=(
            "Jointly optimize battery and solar panel investment with hourly scheduling over a summer season. "
            "Battery: 2000 EUR fixed + 310 EUR/kWh, 10 kW rate. Solar: 200 EUR per 160 W panel. "
            "Buying price includes 0.05 EUR/kWh transmission; selling uses spot price only."
        ),
        variables=[installed, capacity, n_panels, charge, discharge, soc, buy, sell],
        constants=[price_const, load_const, solar_const],
        objectives=[
            Objective(
                name="Total electricity cost",
                symbol="f_1",
                func=[
                    "Add",
                    ["MatMul", "p", "buy"],
                    ["Multiply", 0.05, ["Sum", "buy"]],
                    ["Negate", ["MatMul", "p", "sell"]],
                ],
                unit="EUR",
                maximize=False,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            ),
            Objective(
                name="Total investment cost",
                symbol="f_2",
                func=["Add", ["Multiply", 2000.0, "y"], ["Multiply", 310.0, "E"], ["Multiply", 200.0, "n"]],
                unit="EUR",
                maximize=False,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            ),
        ],
        constraints=constraints,
    )


def summer_cabin_battery_problem_split(initial_soc: float = 0.0, n_panels_max: int = 50) -> "Problem":
    """Same problem as summer_cabin_battery_problem but the time horizon is split into 3 equal segments.

    Each segment has its own TensorVariables and TensorConstants.  SOC continuity across
    segment boundaries is enforced by an equality constraint that reads the last element of the
    previous segment's SOC variable via the At operator.  Objective values are identical to the
    monolithic form.

    Args:
        initial_soc: initial state of charge in kWh. Defaults to 0.0 (empty).
        n_panels_max: upper bound on number of solar panels. Defaults to 50.

    Returns:
        A DESDEO Problem instance.
    """
    df = generate_summer_cabin_electricity_data()
    prices = df["price_EUR_kWh"].to_numpy()
    loads = df["load_kWh"].to_numpy()
    solar = generate_solar_profile()
    T = len(prices)

    N_SEG = 3
    S = T // N_SEG  # 736 hours per segment

    # --- Shared scalar investment variables ---
    installed = Variable(
        name="Battery installed",
        symbol="y",
        variable_type=VariableTypeEnum.binary,
        lowerbound=0,
        upperbound=1,
        initial_value=0,
    )
    capacity = Variable(
        name="Battery capacity (kWh)",
        symbol="E",
        variable_type=VariableTypeEnum.real,
        lowerbound=0.0,
        upperbound=42.0,
        initial_value=0.0,
    )
    n_panels = Variable(
        name="Number of solar panels",
        symbol="n",
        variable_type=VariableTypeEnum.integer,
        lowerbound=0,
        upperbound=n_panels_max,
        initial_value=0,
    )

    # --- Per-segment variables and constants ---
    seg_vars = []    # seg_vars[k-1] = [charge_k, discharge_k, soc_k, buy_k, sell_k]
    seg_consts = []  # seg_consts[k-1] = [price_k, load_k, solar_k]

    for k in range(1, N_SEG + 1):
        sl = slice((k - 1) * S, k * S)
        sk = f"s{k}"

        seg_vars.append([
            TensorVariable(
                name=f"Charging power segment {k} (kW)", symbol=f"c_{sk}", shape=[S],
                variable_type=VariableTypeEnum.real, lowerbounds=0.0, upperbounds=10.0, initial_values=0.0,
            ),
            TensorVariable(
                name=f"Discharging power segment {k} (kW)", symbol=f"d_{sk}", shape=[S],
                variable_type=VariableTypeEnum.real, lowerbounds=0.0, upperbounds=10.0, initial_values=0.0,
            ),
            TensorVariable(
                name=f"State of charge segment {k} (kWh)", symbol=f"soc_{sk}", shape=[S],
                variable_type=VariableTypeEnum.real, lowerbounds=0.0, upperbounds=42.0, initial_values=initial_soc,
            ),
            TensorVariable(
                name=f"Grid purchased segment {k} (kWh/h)", symbol=f"buy_{sk}", shape=[S],
                variable_type=VariableTypeEnum.real, lowerbounds=0.0, upperbounds=None, initial_values=0.0,
            ),
            TensorVariable(
                name=f"Grid sold segment {k} (kWh/h)", symbol=f"sell_{sk}", shape=[S],
                variable_type=VariableTypeEnum.real, lowerbounds=0.0, upperbounds=None, initial_values=0.0,
            ),
        ])
        seg_consts.append([
            TensorConstant(
                name=f"Spot price segment {k} (EUR/kWh)", symbol=f"p_{sk}",
                shape=[S], values=prices[sl].tolist(),
            ),
            TensorConstant(
                name=f"Load segment {k} (kWh/h)", symbol=f"l_{sk}",
                shape=[S], values=loads[sl].tolist(),
            ),
            TensorConstant(
                name=f"Solar per panel segment {k} (kWh/h)", symbol=f"sol_{sk}",
                shape=[S], values=solar[sl].tolist(),
            ),
        ])

    # --- Constraints ---
    constraints = [
        Constraint(
            name="Capacity zero if not installed",
            symbol="cap_ub_con",
            func=["Add", "E", ["Negate", ["Multiply", 42.0, "y"]]],
            cons_type=ConstraintTypeEnum.LTE,
            is_linear=True, is_convex=True, is_twice_differentiable=True,
        ),
        Constraint(
            name="Minimum capacity if installed",
            symbol="cap_lb_con",
            func=["Add", ["Multiply", 14.0, "y"], ["Negate", "E"]],
            cons_type=ConstraintTypeEnum.LTE,
            is_linear=True, is_convex=True, is_twice_differentiable=True,
        ),
    ]

    for k in range(1, N_SEG + 1):
        sk = f"s{k}"
        c_sym = f"c_{sk}"
        d_sym = f"d_{sk}"
        soc_sym = f"soc_{sk}"
        buy_sym = f"buy_{sk}"
        sell_sym = f"sell_{sk}"
        sol_sym = f"sol_{sk}"
        l_sym = f"l_{sk}"

        # SOC at t=1: reference initial_soc for segment 1, or last element of previous segment
        prev_soc_term = -initial_soc if k == 1 else ["Negate", ["At", f"soc_s{k - 1}", S]]

        constraints.append(Constraint(
            name=f"SOC dynamics segment {k} t=1",
            symbol=f"soc_con_{sk}_1",
            func=["Add", ["At", soc_sym, 1], prev_soc_term,
                  ["Negate", ["At", c_sym, 1]], ["At", d_sym, 1]],
            cons_type=ConstraintTypeEnum.EQ,
            is_linear=True, is_convex=True, is_twice_differentiable=True,
        ))

        constraints.extend(
            Constraint(
                name=f"SOC dynamics segment {k} t={t}",
                symbol=f"soc_con_{sk}_{t}",
                func=["Add", ["At", soc_sym, t], ["Negate", ["At", soc_sym, t - 1]],
                      ["Negate", ["At", c_sym, t]], ["At", d_sym, t]],
                cons_type=ConstraintTypeEnum.EQ,
                is_linear=True, is_convex=True, is_twice_differentiable=True,
            )
            for t in range(2, S + 1)
        )

        constraints.extend(
            Constraint(
                name=f"SOC capacity upper bound segment {k} t={t}",
                symbol=f"soc_cap_con_{sk}_{t}",
                func=["Add", ["At", soc_sym, t], ["Negate", "E"]],
                cons_type=ConstraintTypeEnum.LTE,
                is_linear=True, is_convex=True, is_twice_differentiable=True,
            )
            for t in range(1, S + 1)
        )

        constraints.extend(
            Constraint(
                name=f"Energy balance segment {k} t={t}",
                symbol=f"energy_bal_{sk}_{t}",
                func=["Add",
                      ["At", buy_sym, t], ["Negate", ["At", sell_sym, t]],
                      ["At", d_sym, t], ["Negate", ["At", c_sym, t]],
                      ["Multiply", "n", ["At", sol_sym, t]],
                      ["Negate", ["At", l_sym, t]]],
                cons_type=ConstraintTypeEnum.EQ,
                is_linear=True, is_convex=True, is_twice_differentiable=True,
            )
            for t in range(1, S + 1)
        )

    # f_1 = sum over all segments of (p_sk · buy_sk + 0.05*sum(buy_sk) - p_sk · sell_sk)
    f1_terms = []
    for k in range(1, N_SEG + 1):
        sk = f"s{k}"
        f1_terms += [
            ["MatMul", f"p_{sk}", f"buy_{sk}"],
            ["Multiply", 0.05, ["Sum", f"buy_{sk}"]],
            ["Negate", ["MatMul", f"p_{sk}", f"sell_{sk}"]],
        ]

    all_variables = [installed, capacity, n_panels]
    all_constants = []
    for k in range(N_SEG):
        all_variables.extend(seg_vars[k])
        all_constants.extend(seg_consts[k])

    return Problem(
        name="Summer cabin battery and solar investment optimization (split)",
        description=(
            "Same as summer_cabin_battery_problem but the time horizon is split into 3 segments "
            "to reduce per-block tensor sizes while preserving solution equivalence."
        ),
        variables=all_variables,
        constants=all_constants,
        objectives=[
            Objective(
                name="Total electricity cost",
                symbol="f_1",
                func=["Add", *f1_terms],
                unit="EUR",
                maximize=False,
                is_linear=True, is_convex=True, is_twice_differentiable=True,
            ),
            Objective(
                name="Total investment cost",
                symbol="f_2",
                func=["Add", ["Multiply", 2000.0, "y"], ["Multiply", 310.0, "E"], ["Multiply", 200.0, "n"]],
                unit="EUR",
                maximize=False,
                is_linear=True, is_convex=True, is_twice_differentiable=True,
            ),
        ],
        constraints=constraints,
    )


_M_UNMET = 15.0  # kWh upper bound for big-M constraints (exceeds any single-hour load)
_N_OUTAGE_HRS = 4  # hours of grid outage at the start of the affected segment


def summer_cabin_battery_problem_split_scenario(  # noqa: C901
    initial_soc: float = 0.0,
    n_panels_max: int = 50,
) -> ScenarioModel:
    """Build a ScenarioModel for the split summer cabin battery problem with two-level outage scenarios.

    Scenario tree (4 leaf scenarios):
        ROOT → [S1, S2]   — at the start of segment 2, S2 has a 4-hour grid outage
        S1   → [S1a, S1b] — at the start of segment 3, S1b has a 4-hour grid outage
        S2   → [S2a, S2b] — at the start of segment 3, S2b has a 4-hour grid outage

    During an outage, grid buy and sell are forced to zero.  An unmet-demand slack
    variable absorbs any energy balance gap; a binary indicator records whether the
    hour went unserved.

    Unmet-demand variables and the f_3 objective are only added to scenarios where
    an outage can actually occur, keeping each scenario problem as small as possible:
        S1a — no outage, no extra variables, no f_3
        S1b — s3 outage: 8 extra vars, f_3 = Sum(z_s3)
        S2a — s2 outage: 8 extra vars, f_3 = Sum(z_s2)
        S2b — both:     16 extra vars, f_3 = Sum(z_s2 + z_s3)

    The base problem is the unmodified split problem.  All additions live entirely in
    the ScenarioModel pool.

    Non-anticipativity:
        ROOT: investment variables (y, E, n) and all segment-1 schedule variables.
        S1/S2: segment-2 schedule variables within each branch.

    Args:
        initial_soc: initial state of charge in kWh.
        n_panels_max: upper bound on number of solar panels.

    Returns:
        A ScenarioModel instance.
    """
    base = summer_cabin_battery_problem_split(initial_soc=initial_soc, n_panels_max=n_panels_max)
    H = _N_OUTAGE_HRS

    # ------------------------------------------------------------------ #
    # Variable pool
    # indices 0..3   : unmet_s2_1..4
    # indices 4..7   : unmet_s3_1..4
    # indices 8..11  : z_s2_1..4
    # indices 12..15 : z_s3_1..4
    #
    # For segment k: unmet at (k-2)*H + (t-1), z at 2*H + (k-2)*H + (t-1)
    # ------------------------------------------------------------------ #
    var_pool: list[Variable] = []
    for k in (2, 3):
        for t in range(1, H + 1):
            var_pool.append(Variable(
                name=f"Unmet demand s{k} t={t} (kWh)",
                symbol=f"unmet_s{k}_{t}",
                variable_type=VariableTypeEnum.real,
                lowerbound=0.0,
                upperbound=None,
                initial_value=0.0,
            ))
    for k in (2, 3):
        for t in range(1, H + 1):
            var_pool.append(Variable(
                name=f"Demand unserved indicator s{k} t={t}",
                symbol=f"z_s{k}_{t}",
                variable_type=VariableTypeEnum.binary,
                lowerbound=0,
                upperbound=1,
                initial_value=0,
            ))

    def _seg_vars(k: int) -> dict[str, int]:
        """Variable pool indices for outage variables of segment k."""
        return {
            **{f"unmet_s{k}_{t}": (k - 2) * H + (t - 1) for t in range(1, H + 1)},
            **{f"z_s{k}_{t}": 2 * H + (k - 2) * H + (t - 1) for t in range(1, H + 1)},
        }

    # ------------------------------------------------------------------ #
    # Objective pool
    # index 0: f_3 = Sum(z_s2)          — S2a
    # index 1: f_3 = Sum(z_s3)          — S1b
    # index 2: f_3 = Sum(z_s2 + z_s3)   — S2b
    # ------------------------------------------------------------------ #
    def _f3(segments: tuple[int, ...]) -> Objective:
        z_syms = [f"z_s{k}_{t}" for k in segments for t in range(1, H + 1)]
        return Objective(
            name="Hours with unserved electricity demand",
            symbol="f_3",
            func=["Add", *z_syms],
            unit="h",
            maximize=False,
            is_linear=True, is_convex=True, is_twice_differentiable=True,
        )

    _f3_zero = Objective(
        name="Hours with unserved electricity demand",
        symbol="f_3",
        func=["Multiply", 0, "y"],  # always 0 — no outage possible in this scenario
        unit="h",
        maximize=False,
        is_linear=True, is_convex=True, is_twice_differentiable=True,
    )
    obj_pool: list[Objective] = [_f3((2,)), _f3((3,)), _f3((2, 3)), _f3_zero]
    _f3_idx = {(2,): 0, (3,): 1, (2, 3): 2}

    # ------------------------------------------------------------------ #
    # Constraint pool
    # indices 0..3   : energy_bal_s2_1..4  (replace — same symbol)
    # indices 4..7   : energy_bal_s3_1..4  (replace — same symbol)
    # indices 8..11  : bigm_s2_1..4        (new)
    # indices 12..15 : bigm_s3_1..4        (new)
    # indices 16..19 : outage_buy_s2_1..4  (new)
    # indices 20..23 : outage_sell_s2_1..4 (new)
    # indices 24..27 : outage_buy_s3_1..4  (new)
    # indices 28..31 : outage_sell_s3_1..4 (new)
    #
    # For segment k: eb at (k-2)*H+(t-1), bigm at 2*H+(k-2)*H+(t-1),
    #                outage_buy at 4*H+(k-2)*2*H+(t-1),
    #                outage_sell at 4*H+(k-2)*2*H+H+(t-1)
    # ------------------------------------------------------------------ #
    con_pool: list[Constraint] = []

    for k in (2, 3):
        sk = f"s{k}"
        for t in range(1, H + 1):
            con_pool.append(Constraint(
                name=f"Energy balance s{k} t={t} (with unmet slack)",
                symbol=f"energy_bal_{sk}_{t}",
                func=["Add",
                      ["At", f"buy_{sk}", t], ["Negate", ["At", f"sell_{sk}", t]],
                      ["At", f"d_{sk}", t], ["Negate", ["At", f"c_{sk}", t]],
                      ["Multiply", "n", ["At", f"sol_{sk}", t]],
                      f"unmet_{sk}_{t}",
                      ["Negate", ["At", f"l_{sk}", t]]],
                cons_type=ConstraintTypeEnum.EQ,
                is_linear=True, is_convex=True, is_twice_differentiable=True,
            ))

    for k in (2, 3):
        for t in range(1, H + 1):
            con_pool.append(Constraint(
                name=f"Big-M unmet indicator s{k} t={t}",
                symbol=f"bigm_s{k}_{t}",
                func=["Add", f"unmet_s{k}_{t}", ["Negate", ["Multiply", _M_UNMET, f"z_s{k}_{t}"]]],
                cons_type=ConstraintTypeEnum.LTE,
                is_linear=True, is_convex=True, is_twice_differentiable=True,
            ))

    for k in (2, 3):
        sk = f"s{k}"
        for t in range(1, H + 1):
            con_pool.append(Constraint(
                name=f"Outage no-buy s{k} t={t}",
                symbol=f"outage_buy_{sk}_{t}",
                func=["At", f"buy_{sk}", t],
                cons_type=ConstraintTypeEnum.EQ,
                is_linear=True, is_convex=True, is_twice_differentiable=True,
            ))
        for t in range(1, H + 1):
            con_pool.append(Constraint(
                name=f"Outage no-sell s{k} t={t}",
                symbol=f"outage_sell_{sk}_{t}",
                func=["At", f"sell_{sk}", t],
                cons_type=ConstraintTypeEnum.EQ,
                is_linear=True, is_convex=True, is_twice_differentiable=True,
            ))

    def _seg_cons(k: int) -> dict[str, int]:
        """Constraint pool indices for all outage-related constraints of segment k."""
        return {
            **{f"energy_bal_s{k}_{t}": (k - 2) * H + (t - 1) for t in range(1, H + 1)},
            **{f"bigm_s{k}_{t}": 2 * H + (k - 2) * H + (t - 1) for t in range(1, H + 1)},
            **{f"outage_buy_s{k}_{t}": 4 * H + (k - 2) * 2 * H + (t - 1) for t in range(1, H + 1)},
            **{f"outage_sell_s{k}_{t}": 4 * H + (k - 2) * 2 * H + H + (t - 1) for t in range(1, H + 1)},
        }

    def _scenario(outage_segs: tuple[int, ...]) -> Scenario:
        return Scenario(
            variables={k: v for d in [_seg_vars(k) for k in outage_segs] for k, v in d.items()},
            objectives={"f_3": _f3_idx[outage_segs]},
            constraints={k: v for d in [_seg_cons(k) for k in outage_segs] for k, v in d.items()},
        )

    s1_sched = ["c_s1", "d_s1", "soc_s1", "buy_s1", "sell_s1"]
    s2_sched = ["c_s2", "d_s2", "soc_s2", "buy_s2", "sell_s2"]
    s2_unmet = [f"unmet_s2_{t}" for t in range(1, H + 1)]
    s2_z = [f"z_s2_{t}" for t in range(1, H + 1)]

    return ScenarioModel(
        scenario_tree={
            "ROOT": ["S1", "S2"],
            "S1": ["S1a", "S1b"],
            "S2": ["S2a", "S2b"],
            "S1a": [], "S1b": [], "S2a": [], "S2b": [],
        },
        scenario_probabilities={
            "S1": 0.5, "S2": 0.5,
            "S1a": 0.25, "S1b": 0.25,
            "S2a": 0.25, "S2b": 0.25,
        },
        base_problem=base,
        variables=tuple(var_pool),
        objectives=tuple(obj_pool),
        constraints=tuple(con_pool),
        scenarios={
            "S1a": Scenario(objectives={"f_3": 3}),
            "S1b": _scenario((3,)),
            "S2a": _scenario((2,)),
            "S2b": _scenario((2, 3)),
        },
        anticipation_stop={
            "ROOT": ["y", "E", "n", *s1_sched],
            "S1": s2_sched,
            "S2": [*s2_sched, *s2_unmet, *s2_z],
        },
    )
