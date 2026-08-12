import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""Operational cost comparison of five industrial steam production scenarios (2019–2025).

Evaluates the following scenarios for a district heating system with a constant
1 MW heat demand, using Finnish day-ahead electricity and gas market data for
2019-2025:

    1. Electric boiler only (no storage, no optimisation)
    2. Electric boiler + 5 MWh thermal storage (rolling-horizon MILP, 36 h window)
    3. Electric + gas boiler + thermal storage (rolling-horizon MILP, 36 h window)
    4. Electric + gas boiler, no storage (greedy per-hour dispatch)
    5. Gas boiler only (no storage, no optimisation)

Each scenario is evaluated per year.  Annual OPEX, energy use, and hourly
dispatch are written to ``all_years_results.xlsx`` and PNG figures.

Reference:
    Leinonen, J. (2026). *Economics and CO2 Emission Reductions of Electrifying
    Industrial Steam Production in Volatile Electricity Markets*.
    Master's thesis, Aalto University.
"""

# ruff: noqa

import pulp
from tqdm import tqdm

# Define parameters
demand = 1  # Power demand for steam production MW
gas_boiler_capacity = 1  # Gas boiler capacity
electric_boiler_capacity = 2  # Electric boiler capacity
elec_boiler_efficiency = 0.95  # Electric boiler  efficiency
gas_boiler_efficiency = 0.8  # Gas boiler efficiency

storage_capacity = 5  # Storage capacity MWh
initial_storage_level = 0  # Initial storage level, MWh
max_charge_power = 2  # Storage max charging power MW
max_discharge_power = 2  # Storage max discharging power MW
heat_loss = 0.001  # MW/hour
charging_eff = 0.95  # Storage charging efficiency
discharging_eff = 0.95  # Storage discharging efficiency

# Electrity fees
electricity_basic_fee = 1760  # Electricity basic fee €/month
electricity_power_fee = 1710  # Electricity power bee €/MW/month
energy_cost = 0.63  # Energy content tax and strategic stockpile fee €/MWh

# Gas fees
gas_basic_fee = 1.3629355 * (0.3 * demand * 1000 / gas_boiler_efficiency + 108)  # Gas basic fee €/MW/month
gas_energy_tax = 23.354  # Gas energy tax €/MWh

# --- Load and combine all Excel files for 2019-2025 ---
# Data for electricity
elec_files = [
    r"C:\Users\leinonj10\OneDrive - Aalto University\Electricity_price_FI_2019.xlsx",
    r"C:\Users\leinonj10\OneDrive - Aalto University\Electricity_price_FI_2020.xlsx",
    r"C:\Users\leinonj10\OneDrive - Aalto University\Electricity_price_FI_2021.xlsx",
    r"C:\Users\leinonj10\OneDrive - Aalto University\Electricity_price_FI_2022.xlsx",
    r"C:\Users\leinonj10\OneDrive - Aalto University\Electricity_price_FI_2023.xlsx",
    r"C:\Users\leinonj10\OneDrive - Aalto University\Electricity_price_FI_2024.xlsx",
    r"C:\Users\leinonj10\OneDrive - Aalto University\Electricity_price_FI_2025.xlsx",
]
# Data for gas
gas_file = [r"C:\Users\leinonj10\OneDrive - Aalto University\Gas_prices.xlsx"]

results_Energyuse = []

############ CALCULATIONS ##################

# Electricity prices

dfs = []  # dataframes
results_summary = []
for f in elec_files:
    df = pd.read_excel(f)

    # Extract the start time (before the " - ")
    df["Date"] = df["MTU (UTC)"].str.split(" - ").str[0]

    # Convert to datetime
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M:%S")

    # Keep only relevant columns
    df = df[["Date", "Day-ahead Price (EUR/MWh)"]]
    df["Date"] = df["Date"].dt.tz_localize("UTC")

    # If year is 2025 when 15min data, combine for hourly data
    if "2025" in f:
        # 15-min → hourly
        df = df.set_index("Date").resample("h").mean().reset_index()
    dfs.append(df)

# Combine into one dataframe and index by date
df_el_prices = pd.concat(dfs).sort_values("Date").reset_index(drop=True)
# Convert to 1D array
spot_price = df_el_prices["Day-ahead Price (EUR/MWh)"].to_numpy()  # shape: (8760,)
# Save original UTC timesteps
df_el_prices["Date_UTC"] = df_el_prices["Date"]

# Finnish local time for distribution costs
df_el_prices["Date_local"] = df_el_prices["Date_UTC"].dt.tz_convert("Europe/Helsinki")
df_el_prices["Year"] = df_el_prices["Date_local"].dt.year
years = sorted(df_el_prices["Year"].unique())

# Replace negative values with zero
# spot_price[spot_price < 0] = 0

# Electricity annual fixed fee
electricity_fixed_annual = (
    electricity_basic_fee + electricity_power_fee * (electric_boiler_capacity / elec_boiler_efficiency)
) * 12  # €/year


# Adding distribution fees to electricity prices
def distribution_fee_eur_per_mwh(time_local):
    is_winter = time_local.month in [12, 1, 2]
    is_weekday = time_local.weekday() < 5  # mon=0 … fri=4
    is_daytime = 7 <= time_local.hour < 21

    if is_winter and is_weekday and is_daytime:
        return 12.8  # EUR/MWh
    return 4.4  # EUR/MWh


df_el_prices["distribution_fee"] = df_el_prices["Date_local"].apply(distribution_fee_eur_per_mwh)

df_el_prices["total_price"] = df_el_prices["Day-ahead Price (EUR/MWh)"] + df_el_prices["distribution_fee"] + energy_cost
spot_price = df_el_prices["total_price"].to_numpy()

# Ignoring negative prices
# spot_price[spot_price < 0] = 0

# Gas prices
# Gas annual fixed fee
gas_fixed_annual = gas_basic_fee * 12  # €/year
# Gas variable fees
df_gas = pd.read_excel(gas_file[0])
df_gas["Month"] = pd.to_datetime(df_gas["Month"], format="%m/%Y").dt.strftime("%m/%Y")
df_gas["total_gas_price"] = df_gas["Gas Price (EUR/MWh)"] + gas_energy_tax
df_gas = df_gas[["Month", "total_gas_price"]]

# Combine hourly electricity prices and monthly gas prices
df_el_prices["Month"] = df_el_prices["Date_local"].dt.strftime("%m/%Y")
df_prices = df_el_prices.merge(df_gas, on="Month", how="left")

# Determine the latest year for which gas data exists
df_gas_dt = pd.to_datetime(df_gas["Month"], format="%m/%Y")
max_gas_year = df_gas_dt.dt.year.max()
# Only rows for 2019-2025, not 2026
df_prices = df_prices[df_prices["Date_local"].dt.year <= max_gas_year].copy()

if df_prices["total_gas_price"].isna().any():
    print("Warning: No gas price for all hours!")
gas_price = df_prices["total_gas_price"].to_numpy()

# Use Finnish local years for electricity, but cut at max_gas_year
years = sorted(df_el_prices["Date_local"].dt.year.unique())
years = [y for y in years if y <= max_gas_year]

### OPERATIONAL COSTS ########


# Costs for electric boiler without storage (Scenario 1)
def cost_without_storage(spot_prices, demand, elec_boiler_efficiency):
    elec_boiler_input = demand / elec_boiler_efficiency
    return np.sum(spot_prices * elec_boiler_input)


# ---  Daily 36h optimisation function for electric boiler and storage (Scenario 2)---
def optimise_tank_actions_dynamic(
    spot_prices, storage_capacity, initial_storage_level, window_size=36, action_period=24
):
    elec_boiler_steam_input_max = demand / elec_boiler_efficiency
    elec_boiler_total_input_max = electric_boiler_capacity / elec_boiler_efficiency

    # Initialize lists for decision variables
    charge_actions = []
    discharge_actions = []
    storage_levels = []
    current_time = 0
    elec_boiler_charge_actions = []
    elec_boiler_steam_actions = []

    # Start iteration through the year
    num_periods = len(spot_prices)  # Total number of time periods
    # Initialise progress bar (does not affect results)
    total_steps = (num_periods - window_size) // action_period
    progress_bar = tqdm(total=total_steps, desc="Optimising storage actions")

    while current_time + window_size <= num_periods:
        window_spot_prices = spot_prices[current_time : current_time + window_size]

        # LP model
        model = pulp.LpProblem("Optimise_Tank_Actions", pulp.LpMinimize)

        # Decision variables
        elec_boiler_steam = [
            pulp.LpVariable(f"boiler_steam{t}", lowBound=0, upBound=elec_boiler_steam_input_max, cat="Continuous")
            for t in range(window_size)
        ]
        elec_boiler_charge = [
            pulp.LpVariable(f"boiler_charge{t}", lowBound=0, upBound=elec_boiler_total_input_max, cat="Continuous")
            for t in range(window_size)
        ]
        el_discharge_storage = [
            pulp.LpVariable(f"discharge_{t}", lowBound=0, upBound=max_discharge_power, cat="Continuous")
            for t in range(window_size)
        ]
        storage = [
            pulp.LpVariable(f"storage_{t}", lowBound=0, upBound=storage_capacity, cat="Continuous")
            for t in range(window_size + 1)
        ]

        # Indicator variables to enforce mutual exclusivity
        charge_indicator = [
            pulp.LpVariable(f"charge_indicator_{t}", lowBound=0, upBound=1, cat="Binary") for t in range(window_size)
        ]
        discharge_indicator = [
            pulp.LpVariable(f"discharge_indicator_{t}", lowBound=0, upBound=1, cat="Binary") for t in range(window_size)
        ]
        # Initial storage level constraint
        model += storage[0] == initial_storage_level

        # Constraints
        for t in range(window_size):
            model += storage[t + 1] == (
                storage[t] * (1 - heat_loss)
                + elec_boiler_charge[t] * elec_boiler_efficiency * charging_eff
                - el_discharge_storage[t] / discharging_eff
            )  # Storage constraint
            model += (
                elec_boiler_steam[t] * elec_boiler_efficiency + el_discharge_storage[t] == demand
            )  # Demand constraint
            model += elec_boiler_steam[t] + elec_boiler_charge[t] <= elec_boiler_total_input_max  # Boiler constraint
            model += (
                charge_indicator[t] + discharge_indicator[t] <= 1
            )  # Constraint forbidding charging and discharging at the same time
            model += (
                elec_boiler_charge[t] * elec_boiler_efficiency <= max_charge_power * charge_indicator[t]
            )  # Charging constraint
            model += el_discharge_storage[t] <= demand * discharge_indicator[t]  # Discharging constraint
            # model += storage[t] >=1  #MWh  #min level of storage, not used

        # model += storage[window_size] >= storage[0]  #terminal constraint, not used

        # Objective: minimise cost for each window
        model += pulp.lpSum(
            [
                (window_spot_prices[t] * elec_boiler_steam[t] + window_spot_prices[t] * elec_boiler_charge[t])
                for t in range(window_size)
            ]
        )

        solver = pulp.GUROBI_CMD(msg=False, options=[("TimeLimit", 10)])
        status = model.solve(solver)

        # TO AVOID TOO LONG OPTIMISATION
        if status != pulp.LpStatusOptimal and status != pulp.LpStatusNotSolved:
            # fill with zeroes or repeat last known value
            charge_actions.extend([0] * action_period)
            discharge_actions.extend([0] * action_period)
            storage_levels.extend([initial_storage_level] * action_period)
            elec_boiler_charge_actions.extend([0] * action_period)
            elec_boiler_steam_actions.extend([0] * action_period)
        else:
            # Store results for the action period
            charge_actions.extend(
                [pulp.value(elec_boiler_charge[t] * elec_boiler_efficiency) for t in range(action_period)]
            )  # storage
            discharge_actions.extend([pulp.value(el_discharge_storage[t]) for t in range(action_period)])
            storage_levels.extend([pulp.value(storage[t]) for t in range(action_period)])
            elec_boiler_charge_actions.extend(
                [pulp.value(elec_boiler_charge[t]) for t in range(action_period)]
            )  # boiler
            elec_boiler_steam_actions.extend([pulp.value(elec_boiler_steam[t]) for t in range(action_period)])

            # Update the initial storage level for the next window
            initial_storage_level = pulp.value(storage[action_period])

        # Move to the next action period
        current_time += action_period
        # pdb.set_trace()
        progress_bar.update(1)  # progress bar update

    progress_bar.close()  # close progress bar when program is finished

    # Fill the remaining hours with no action if the loop doesn't cover the entire period
    s = initial_storage_level
    if current_time < num_periods:
        s = s * (1 - heat_loss)
        charge_actions.extend([0] * (num_periods - current_time))
        discharge_actions.extend([0] * (num_periods - current_time))
        storage_levels.extend([s] * (num_periods - current_time))
        elec_boiler_charge_actions.extend([0] * (num_periods - current_time))
        elec_boiler_steam_actions.extend([0] * (num_periods - current_time))

    # STORAGE OPERATION METRICS
    results_df = pd.DataFrame(
        {
            "time": np.arange(len(charge_actions)),
            "charge": charge_actions,
            "discharge": discharge_actions,
            "storage_level": storage_levels,
            "elec_boiler_steam": elec_boiler_steam_actions,
            "boiler_charge": elec_boiler_charge_actions,
            "spot_price": spot_prices[: len(charge_actions)],
        }
    )

    return (
        np.array(charge_actions),
        np.array(discharge_actions),
        np.array(storage_levels),
        np.array(elec_boiler_steam_actions),
        np.array(elec_boiler_charge_actions),
    )


# Costs for storage with electric boiler (Scenario 2)
def cost_with_storage(spot_prices, elec_boiler_steam, elec_boiler_charge):
    return np.sum(spot_prices * (elec_boiler_steam + elec_boiler_charge))


# ---  Daily 36h optimisation function for both boilers and storage (Scenario 3) ---
def optimise_both_boilers_tank_actions(
    spot_prices, gas_prices, initial_storage_level, storage_capacity, window_size=36, action_period=24
):
    elec_boiler_steam_input_max = demand / elec_boiler_efficiency
    elec_boiler_total_input_max = electric_boiler_capacity / elec_boiler_efficiency
    gas_boiler_steam_input_max = demand / gas_boiler_efficiency
    gas_boiler_total_input_max = gas_boiler_capacity / gas_boiler_efficiency

    # Initialise lists for decision variables
    charge_actions = []
    gas_charge_actions = []
    discharge_actions = []
    storage_levels = []
    current_time = 0
    elec_boiler_charge_actions = []
    gas_boiler_charge_actions = []
    elec_boiler_steam_actions = []
    gas_boiler_steam_actions = []

    # Start iteration through the year
    num_periods = len(spot_prices)  # Total number of time periods
    # Initialise progress bar (does not affect results)
    total_steps = (num_periods - window_size) // action_period
    progress_bar = tqdm(total=total_steps, desc="Optimising storage actions")

    while current_time + window_size <= num_periods:
        window_spot_prices = spot_prices[current_time : current_time + window_size]
        window_gas_prices = gas_prices[current_time : current_time + window_size]

        # Create LP model
        model = pulp.LpProblem("Optimise_Tank_Actions", pulp.LpMinimize)

        # Initialize decision variables
        elec_boiler_steam = [
            pulp.LpVariable(f"boiler_steam{t}", lowBound=0, upBound=elec_boiler_steam_input_max, cat="Continuous")
            for t in range(window_size)
        ]
        gas_boiler_steam = [
            pulp.LpVariable(f"gas_boiler_steam{t}", lowBound=0, upBound=gas_boiler_steam_input_max, cat="Continuous")
            for t in range(window_size)
        ]
        elec_boiler_charge = [
            pulp.LpVariable(f"boiler_charge{t}", lowBound=0, upBound=elec_boiler_total_input_max, cat="Continuous")
            for t in range(window_size)
        ]
        gas_boiler_charge = [
            pulp.LpVariable(f"gas_boiler_charge{t}", lowBound=0, upBound=gas_boiler_total_input_max, cat="Continuous")
            for t in range(window_size)
        ]
        discharge_storage = [
            pulp.LpVariable(f"discharge_{t}", lowBound=0, upBound=max_discharge_power, cat="Continuous")
            for t in range(window_size)
        ]
        storage = [
            pulp.LpVariable(f"storage_{t}", lowBound=0, upBound=storage_capacity, cat="Continuous")
            for t in range(window_size + 1)
        ]

        # Indicator variables to enforce mutual exclusivity
        charge_indicator_el = [
            pulp.LpVariable(f"charge_indicator_el{t}", lowBound=0, upBound=1, cat="Binary") for t in range(window_size)
        ]
        charge_indicator_gas = [
            pulp.LpVariable(f"charge_indicator_gas{t}", lowBound=0, upBound=1, cat="Binary") for t in range(window_size)
        ]
        discharge_indicator = [
            pulp.LpVariable(f"discharge_indicator_{t}", lowBound=0, upBound=1, cat="Binary") for t in range(window_size)
        ]

        # Initial storage level constraint
        model += storage[0] == initial_storage_level

        # Constraints
        for t in range(window_size):
            model += storage[t + 1] == (
                storage[t] * (1 - heat_loss)
                + (elec_boiler_charge[t] * elec_boiler_efficiency + gas_boiler_charge[t] * gas_boiler_efficiency)
                * charging_eff
                - discharge_storage[t] / discharging_eff
            )  # Storage constraint
            model += (
                elec_boiler_steam[t] * elec_boiler_efficiency
                + gas_boiler_steam[t] * gas_boiler_efficiency
                + discharge_storage[t]
                == demand
            )  # Demand constraint
            model += elec_boiler_steam[t] + elec_boiler_charge[t] <= elec_boiler_total_input_max
            model += gas_boiler_steam[t] + gas_boiler_charge[t] <= gas_boiler_total_input_max
            model += (
                charge_indicator_el[t] + charge_indicator_gas[t] + discharge_indicator[t] <= 1
            )  # constraints forbidding charging and discharging at the same time
            model += elec_boiler_charge[t] * elec_boiler_efficiency <= max_charge_power * charge_indicator_el[t]
            model += gas_boiler_charge[t] * gas_boiler_efficiency <= max_charge_power * charge_indicator_gas[t]
            model += discharge_storage[t] <= demand * discharge_indicator[t]
            # model += storage[t] >= 1  #MWh #Storage minimum level constraint, not used
            # model += el_boiler_indicator[t] + gas_boiler_indicator[t] <= 1  #constraint for using either gas or electricity (in case of having only one hybrid boiler)
        # model += storage[window_size] >= storage[0]  # terminal constraint, not used

        # Objective: minimise cost for each window
        model += pulp.lpSum(
            [
                (window_spot_prices[t] * (elec_boiler_steam[t] + elec_boiler_charge[t]))
                + (window_gas_prices[t] * (gas_boiler_steam[t] + gas_boiler_charge[t]))
                for t in range(window_size)
            ]
        )
        solver = pulp.GUROBI_CMD(msg=False, options=[("TimeLimit", 10)])
        status = model.solve(solver)

        # TO AVOID TOO LONG OPTIMISATION
        if status != pulp.LpStatusOptimal and status != pulp.LpStatusNotSolved:
            # fill with zeroes or repeat last known value
            charge_actions.extend([0] * action_period)
            gas_charge_actions.extend([0] * action_period)
            discharge_actions.extend([0] * action_period)
            storage_levels.extend([initial_storage_level] * action_period)
            elec_boiler_charge_actions.extend([0] * action_period)
            gas_boiler_charge_actions.extend([0] * action_period)
            elec_boiler_steam_actions.extend([0] * action_period)
            gas_boiler_steam_actions.extend([0] * action_period)
        else:
            # Store results for the action period
            charge_actions.extend([pulp.value(elec_boiler_charge[t]) for t in range(action_period)])
            gas_charge_actions.extend([pulp.value(gas_boiler_charge[t]) for t in range(action_period)])
            discharge_actions.extend([pulp.value(discharge_storage[t]) for t in range(action_period)])
            storage_levels.extend([pulp.value(storage[t]) for t in range(action_period)])
            elec_boiler_charge_actions.extend([pulp.value(elec_boiler_charge[t]) for t in range(action_period)])
            gas_boiler_charge_actions.extend([pulp.value(gas_boiler_charge[t]) for t in range(action_period)])
            elec_boiler_steam_actions.extend([pulp.value(elec_boiler_steam[t]) for t in range(action_period)])
            gas_boiler_steam_actions.extend([pulp.value(gas_boiler_steam[t]) for t in range(action_period)])
            # Update the initial storage level for the next window
            initial_storage_level = pulp.value(storage[action_period])
        # Move to the next action period
        current_time += action_period
        # pdb.set_trace()
        progress_bar.update(1)  # progress bar update

    progress_bar.close()  # close progress bar when program is finished

    # Fill the remaining hours with no action if the loop doesn't cover the entire period
    s = initial_storage_level
    if current_time < num_periods:
        s = s * (1 - heat_loss)
        charge_actions.extend([0] * (num_periods - current_time))
        gas_charge_actions.extend([0] * (num_periods - current_time))
        discharge_actions.extend([0] * (num_periods - current_time))
        storage_levels.extend([s] * (num_periods - current_time))
        elec_boiler_charge_actions.extend([0] * (num_periods - current_time))
        gas_boiler_charge_actions.extend([0] * (num_periods - current_time))
        elec_boiler_steam_actions.extend([0] * (num_periods - current_time))
        gas_boiler_steam_actions.extend([0] * (num_periods - current_time))

    # STORAGE OPERATION METRICS
    results_df = pd.DataFrame(
        {
            "time": np.arange(len(charge_actions)),
            "charge_elec": charge_actions,
            "charge_gas": gas_charge_actions,
            "discharge": discharge_actions,
            "storage_level": storage_levels,
            "elec_boiler_steam": elec_boiler_steam_actions,
            "gas_boiler_steam": gas_boiler_steam_actions,
            "spot_price": spot_prices[: len(charge_actions)],
            "gas_price": gas_prices[: len(charge_actions)],
        }
    )

    return (
        np.array(charge_actions),
        np.array(gas_charge_actions),
        np.array(discharge_actions),
        np.array(storage_levels),
        np.array(elec_boiler_steam_actions),
        np.array(gas_boiler_steam_actions),
        np.array(elec_boiler_charge_actions),
        np.array(gas_boiler_charge_actions),
    )


# Costs for storage + both boilers (Scenario 3)
def cost_with_storage_and_both_boilers(spot_prices, gas_prices, elec_steam, elec_charge, gas_steam, gas_charge):
    return np.sum(spot_prices * (elec_steam + elec_charge)) + np.sum(gas_prices * (gas_steam + gas_charge))


# Costs for both boilers without storage (Scenario 4)
def both_boilers_without_storage(gas_prices, spot_prices, demand, gas_boiler_efficiency, elec_boiler_efficiency):
    gas_boiler_input = demand / gas_boiler_efficiency
    el_boiler_input = demand / elec_boiler_efficiency

    elec_boiler_steam_nostorage = []
    gasboiler_steam_nostorage = []

    for t in range(len(spot_prices)):
        el_steam = spot_prices[t] * el_boiler_input
        gas_steam = gas_prices[t] * gas_boiler_input

        if el_steam <= gas_steam:
            elec_boiler_steam_nostorage.append(demand)
            gasboiler_steam_nostorage.append(0)
        else:
            gasboiler_steam_nostorage.append(demand)
            elec_boiler_steam_nostorage.append(0)

    boiler_steam_both_boilers = np.array(elec_boiler_steam_nostorage)
    gas_boiler_steam_both_boilers = np.array(gasboiler_steam_nostorage)

    both_boilers_nostorage_total = np.sum(
        boiler_steam_both_boilers / elec_boiler_efficiency * spot_prices
        + gas_boiler_steam_both_boilers / gas_boiler_efficiency * gas_prices
    )

    return both_boilers_nostorage_total


# Costs for gas boiler base case (Scenario 5)
def cost_gas_without_storage(gas_prices, demand, gas_boiler_efficiency):
    gas_boiler_input = demand / gas_boiler_efficiency
    return np.sum(gas_prices * gas_boiler_input)


#####  Calculating the results  #####

## Creating lists for later use
both_boilers_storage_elec_MWh = []
both_boilers_storage_gas_MWh = []
both_boilers_nostorage_elec_MWh = []
both_boilers_nostorage_gas_MWh = []
electric_storage_MWh = []
electric_nostorage_MWh = []
gas_only_MWh = []

both_boilers_storage_elec_week = []
both_boilers_storage_gas_week = []
both_boilers_nostorage_elec_week = []
both_boilers_nostorage_gas_week = []

el_steam_hours_per_year = []
gas_steam_hours_per_year = []
discharge_hours_per_year = []

# Defining plot style
plt.rcParams.update(
    {
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 12,
    }
)

# Writing results to Excel
with pd.ExcelWriter("all_years_results.xlsx", engine="openpyxl") as writer:
    for year in years:
        print("\n============================")
        print(f"Results year {year}")
        print("============================")

        df_year_el = df_el_prices[df_el_prices["Date_local"].dt.year == year].copy()
        df_year_gas = df_prices[df_prices["Date_local"].dt.year == year].copy()

        spot_price = df_year_el["total_price"].to_numpy()
        gas_price = df_year_gas["total_gas_price"].to_numpy()

        ######### Calculating costs for each scenario ##########

        # Electric only without storage (Scenario 1)
        electric_only_total = cost_without_storage(spot_price, demand, elec_boiler_efficiency)

        # Electric boiler optimisation with storage (Scenario 2)
        (charge_actions, discharge_actions, storage_levels, elec_boiler_steam_actions, elec_boiler_charge_actions) = (
            optimise_tank_actions_dynamic(spot_price, storage_capacity, initial_storage_level)
        )
        # Electric boiler + storage cost
        storage_cost = cost_with_storage(spot_price, elec_boiler_steam_actions, elec_boiler_charge_actions)
        storage_df = pd.DataFrame(
            {
                "charge": charge_actions,
                "discharge": discharge_actions,
                "storage_level": storage_levels,
                "elec_boiler_steam": elec_boiler_steam_actions,
                "elec_boiler_charge": elec_boiler_charge_actions,
                "spot_price": spot_price[: len(charge_actions)],
            }
        )
        storage_sheet = f"{year}_storage"
        storage_df.to_excel(writer, sheet_name=storage_sheet, index=False)

        # Both boilers boiler with storage (Scenario 3)
        (
            charge_elec,
            charge_gas,
            discharge,
            storage_levels,
            steam_elec,
            steam_gas,
            charge_elec_boiler,
            charge_gas_boiler,
        ) = optimise_both_boilers_tank_actions(spot_price, gas_price, initial_storage_level, storage_capacity)
        # Both boilers + storage cost
        both_boilers_storage_cost = cost_with_storage_and_both_boilers(
            spot_price, gas_price, steam_elec, charge_elec_boiler, steam_gas, charge_gas_boiler
        )
        both_boilers_df = pd.DataFrame(
            {
                "charge_elec": charge_elec,
                "charge_gas": charge_gas,
                "discharge": discharge,
                "storage_level": storage_levels,
                "steam_elec": steam_elec,
                "steam_gas": steam_gas,
                "spot_price": spot_price[: len(charge_actions)],
                "gas_price": gas_price[: len(charge_actions)],
            }
        )

        both_boilers_sheet = f"{year}_both_boilers"
        both_boilers_df.to_excel(writer, sheet_name=both_boilers_sheet, index=False)

        # Both boilers without storage (Scenario 4)
        both_boilers_nostorage_total = both_boilers_without_storage(
            gas_price, spot_price, demand, gas_boiler_efficiency, elec_boiler_efficiency
        )

        # Gas only (Scenario 5)
        baseline_gas_cost = cost_gas_without_storage(gas_price, demand, gas_boiler_efficiency)

        # Count operating hours (hours with non-zero activity)
        eps = 1e-9  # tolerance for > 0
        el_hours = int(np.count_nonzero(steam_elec > eps))  # electric boiler steam hours
        gas_hours = int(np.count_nonzero(steam_gas > eps))  # gas boiler steam hours
        dis_hours = int(np.count_nonzero(discharge > eps))  # discharge hours

        el_steam_hours_per_year.append(el_hours)
        gas_steam_hours_per_year.append(gas_hours)
        discharge_hours_per_year.append(dis_hours)

        # Calculating and plotting Hourly costs

        # Electric without storage
        spot_hourly = spot_price * (demand / elec_boiler_efficiency)

        # Electric with storage
        storage_hourly = spot_price * (elec_boiler_steam_actions + elec_boiler_charge_actions)

        # Both boilers + Storage hourly
        both_boilers_storage_hourly = spot_price * (steam_elec + charge_elec_boiler) + gas_price * (
            steam_gas + charge_gas_boiler
        )

        # Both boilers without storage
        el_input = demand / elec_boiler_efficiency
        gas_input = demand / gas_boiler_efficiency

        both_boilers_nostorage_hourly = np.minimum(spot_price * el_input, gas_price * gas_input)

        # Gas only
        gas_only_hourly = gas_price * (demand / gas_boiler_efficiency)

        # Plot absolute hourly costs
        plt.figure(figsize=(12, 6))

        plt.plot(spot_hourly, label="Electric only", alpha=0.7)
        plt.plot(storage_hourly, label="Storage (electric)", alpha=0.7)
        plt.plot(both_boilers_storage_hourly, label="Both boilers + Storage", alpha=0.7)
        plt.plot(both_boilers_nostorage_hourly, label="Both boilers no storage", alpha=0.7)
        plt.plot(gas_only_hourly, label="Gas only", alpha=0.7)

        plt.xlabel("Hour of Year")
        plt.ylabel("Hourly Cost (€)")
        plt.title(f"Hourly Cost – {year}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plt.savefig(f"HourlyCost_{year}.png", dpi=300)

        # Calculating and plotting cumulative costs
        cum_spot = np.cumsum(spot_hourly)
        cum_storage = np.cumsum(storage_hourly)
        cum_both_boilers_storage = np.cumsum(both_boilers_storage_hourly)
        cum_both_boilers_nostorage = np.cumsum(both_boilers_nostorage_hourly)
        cum_gas = np.cumsum(gas_only_hourly)

        # Plot cumulative costs
        plt.figure(figsize=(10, 6))

        plt.plot(cum_spot, label="Electric only", color="blue", linewidth=2)
        plt.plot(cum_storage, label="Storage (electric)", color="orange", linewidth=2)
        plt.plot(cum_both_boilers_storage, label="Both boilers + Storage", color="green", linewidth=2)
        plt.plot(cum_both_boilers_nostorage, label="Both boilers no storage", color="red", linewidth=2)
        plt.plot(cum_gas, label="Gas only", color="purple", linewidth=2)
        plt.xlabel("Hour", fontsize=16)
        plt.ylabel(f"Cumulative Cost {year}", color="black")
        plt.title(f"Cumulative Cost – {year}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"Cumulativecost{year}.png", dpi=300, bbox_inches="tight")

        # Calculate and plot Daily costs
        # Create daily dataframe
        daily_df = pd.DataFrame(
            {
                "Day": df_year_el["Date_local"].dt.dayofyear,
                "Electric only": spot_hourly,
                "Storage (electric)": storage_hourly,
                "Both boilers + Storage": both_boilers_storage_hourly,
                "Both boilers no storage": both_boilers_nostorage_hourly,
                "Gas only": gas_only_hourly,
            }
        )

        # Group by day and sum
        daily_costs = daily_df.groupby("Day", as_index=False).sum()

        plt.figure(figsize=(10, 6))

        plt.plot(daily_costs["Day"], daily_costs["Electric only"], color="blue", label="Electric only")

        plt.plot(daily_costs["Day"], daily_costs["Storage (electric)"], color="orange", label="Storage (electric)")

        plt.plot(
            daily_costs["Day"], daily_costs["Both boilers + Storage"], color="green", label="Both boilers + Storage"
        )

        plt.plot(
            daily_costs["Day"], daily_costs["Both boilers no storage"], color="red", label="Both boilers no storage"
        )

        plt.plot(daily_costs["Day"], daily_costs["Gas only"], color="purple", label="Gas only")

        plt.xlabel("Date")
        plt.ylabel("Daily Cost (€)")
        plt.title(f"Daily Cost Comparison – {year}")
        plt.xticks(range(0, 366, 30))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"DailyCost_{year}.png", dpi=300)
        plt.close()

        # Calculating and plotting Monthly costs

        months = df_year_el["Date_local"].dt.month.values
        monthly_df = pd.DataFrame(
            {
                "month": months,
                "Electric only": spot_hourly,
                "Storage (electric)": storage_hourly,
                "Both boilers + Storage": both_boilers_storage_hourly,
                "Both boilers no storage": both_boilers_nostorage_hourly,
                "Gas only": gas_only_hourly,
            }
        )

        monthly_costs = monthly_df.groupby("month").sum().reset_index()

        plt.figure(figsize=(10, 6))

        plt.plot(
            monthly_costs["month"], monthly_costs["Electric only"], marker="o", color="blue", label="Electric only"
        )

        plt.plot(
            monthly_costs["month"],
            monthly_costs["Storage (electric)"],
            marker="o",
            color="orange",
            label="Storage (electric)",
        )

        plt.plot(
            monthly_costs["month"],
            monthly_costs["Both boilers + Storage"],
            marker="o",
            color="green",
            label="Both boilers + Storage",
        )

        plt.plot(
            monthly_costs["month"],
            monthly_costs["Both boilers no storage"],
            marker="o",
            color="red",
            label="Both boilers no storage",
        )

        plt.plot(monthly_costs["month"], monthly_costs["Gas only"], marker="o", color="purple", label="Gas only")

        plt.xlabel("Month")
        plt.ylabel("Monthly Cost (€)")
        plt.title(f"Monthly Cost Comparison – {year}")
        plt.xticks(range(1, 13))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(f"MonthlyCost_{year}.png", dpi=300)
        plt.close()

        # ----- Adding annual fixed costs to the total costs---#

        # Add fixed fees to scenario costs
        electric_only_total_with_fix = electric_only_total + electricity_fixed_annual
        storage_cost_with_fix = storage_cost + electricity_fixed_annual

        both_boilers_storage_cost_with_fix = both_boilers_storage_cost + electricity_fixed_annual + gas_fixed_annual
        both_boilers_nostorage_total_with_fix = (
            both_boilers_nostorage_total + electricity_fixed_annual + gas_fixed_annual
        )

        baseline_gas_cost_with_fix = baseline_gas_cost + gas_fixed_annual

        results_summary.append(
            {
                "year": year,
                "electric_only": electric_only_total_with_fix,
                "storage_only": storage_cost_with_fix,
                "both_boilers_storage": both_boilers_storage_cost_with_fix,
                "both_boilers_nostorage": both_boilers_nostorage_total_with_fix,
                "gas_only": baseline_gas_cost_with_fix,
            }
        )

        # ------Calculating energy use for each scenario-----

        # Electric boiler with spot (Scenario 1)
        hours = len(spot_price)
        elecspot_MWh = (demand / elec_boiler_efficiency) * hours
        electric_nostorage_MWh.append(elecspot_MWh)

        # Electric boiler with storage (Scenario 2)
        elecstorage_MWh = (storage_df["elec_boiler_charge"] + storage_df["elec_boiler_steam"]).sum()
        electric_storage_MWh.append(elecstorage_MWh)

        # Both boilers boiler with storage (Scenario 3)

        elec_MWh = (both_boilers_df["charge_elec"] + both_boilers_df["steam_elec"]).sum()

        gas_MWh = (both_boilers_df["charge_gas"] + both_boilers_df["steam_gas"]).sum()

        both_boilers_storage_elec_MWh.append(elec_MWh)
        both_boilers_storage_gas_MWh.append(gas_MWh)

        # Both boilers without storage (Scenario 4)
        cost_el = spot_price * el_input
        cost_gas = gas_price * gas_input

        steam_el = np.where(cost_el <= cost_gas, demand / elec_boiler_efficiency, 0)
        steam_gas = np.where(cost_gas < cost_el, demand / gas_boiler_efficiency, 0)

        both_boilers_nostorage_elec_MWh.append(steam_el.sum())
        both_boilers_nostorage_gas_MWh.append(steam_gas.sum())

        # Gas only (Scenario 5)
        hours = len(spot_price)
        gasboiler_MWh = (demand / gas_boiler_efficiency) * hours
        gas_only_MWh.append(gasboiler_MWh)

        results_Energyuse.append(
            {
                "year": year,
                "electric_storage_MWh": elecstorage_MWh,
                "electric_nostorage_MWh": elecspot_MWh,
                "both_boilers_storage_elec_MWh": elec_MWh,
                "both_boilers_storage_gas_MWh": gas_MWh,
                "both_boilers_nostorage_elec_MWh": both_boilers_nostorage_elec_MWh[-1],
                "both_boilers_nostorage_gas_MWh": both_boilers_nostorage_gas_MWh[-1],
                "gas_only_MWh": gasboiler_MWh,
            }
        )

        # Calculating first week electricity and gas energy
        # First 7 days of year
        first_week_mask = df_year_el["Date_local"].dt.dayofyear <= 7
        mask = first_week_mask.values

        # Both boilers + Storage
        elec_first_week = (steam_elec[mask] + charge_elec_boiler[mask]).sum()
        gas_first_week = (steam_gas[mask] + charge_gas_boiler[mask]).sum()

        # Both boilers without storage
        cost_el = spot_price * (demand / elec_boiler_efficiency)
        cost_gas = gas_price * (demand / gas_boiler_efficiency)

        steam_el = np.where(cost_el <= cost_gas, demand / elec_boiler_efficiency, 0)
        steam_gas = np.where(cost_gas < cost_el, demand / gas_boiler_efficiency, 0)

        elec_first_week_nostorage = steam_el[mask].sum()
        gas_first_week_nostorage = steam_gas[mask].sum()

        # Store results
        both_boilers_storage_elec_week.append(elec_first_week)
        both_boilers_storage_gas_week.append(gas_first_week)
        both_boilers_nostorage_elec_week.append(elec_first_week_nostorage)
        both_boilers_nostorage_gas_week.append(gas_first_week_nostorage)

    # Print and save results
    print("\n===== RESULTS 2019–2025 =====\n")

    summary_df = pd.DataFrame(results_summary)
    Energyuse_df = pd.DataFrame(results_Energyuse)

    # Order columns
    summary_df = summary_df[
        ["year", "electric_only", "storage_only", "both_boilers_storage", "both_boilers_nostorage", "gas_only"]
    ]

    Energyuse_df = Energyuse_df[
        [
            "year",
            "electric_storage_MWh",
            "electric_nostorage_MWh",
            "both_boilers_storage_elec_MWh",
            "both_boilers_storage_gas_MWh",
            "both_boilers_nostorage_elec_MWh",
            "both_boilers_nostorage_gas_MWh",
            "gas_only_MWh",
        ]
    ]

    summary_df.to_excel(writer, sheet_name="Summary", index=False)

    for r in results_summary:
        print(f"{r['year']}:")
        print(f"  1. Electric only:        {r['electric_only']:,.0f} €")
        print(f"  2. Storage (electric):   {r['storage_only']:,.0f} €")
        print(f"  3. Both boilers + Storage:     {r['both_boilers_storage']:,.0f} €")
        print(f"  4. Both boilers no storage:    {r['both_boilers_nostorage']:,.0f} €")
        print(f"  5. Gas only:             {r['gas_only']:,.0f} €")
        print()

    Energyuse_df.to_excel(writer, sheet_name="Energy_use", index=False)

    for r in results_Energyuse:
        print(f"{r['year']}:")
        print(f"  1. Electric only:        {r['electric_nostorage_MWh']:,.0f} MWh")
        print(f"  2. Storage (electric):   {r['electric_storage_MWh']:,.0f} MWh")
        print(f"  3. Both boilers + Storage:     {r['both_boilers_storage_elec_MWh']:,.0f} MWh")
        print(f"  3. Both boilers + Storage:     {r['both_boilers_storage_gas_MWh']:,.0f} MWh")
        print(f"  4. Both boilers no storage:    {r['both_boilers_nostorage_elec_MWh']:,.0f} MWh")
        print(f"  4. Both boilers no storage:    {r['both_boilers_nostorage_gas_MWh']:,.0f} MWh")
        print(f"  5. Gas only:             {r['gas_only_MWh']:,.0f} MWh")
        print()

import matplotlib.pyplot as plt

# Plot annual OPEX
plt.figure(figsize=(10, 6))

plt.plot(summary_df["year"], summary_df["electric_only"], marker="o", label="Electric only")
plt.plot(summary_df["year"], summary_df["storage_only"], marker="o", label="Storage (electric)")
plt.plot(summary_df["year"], summary_df["both_boilers_storage"], marker="o", label="Both boilers + Storage")
plt.plot(summary_df["year"], summary_df["both_boilers_nostorage"], marker="o", label="Both boilers no storage")
plt.plot(summary_df["year"], summary_df["gas_only"], marker="o", label="Gas only")

plt.xlabel("Year")
plt.ylabel("Annual Cost (€)")
plt.title("Annual Cost Comparison 2019–2025")
plt.grid(True)
plt.legend()
plt.xticks(summary_df["year"])

plt.tight_layout()
plt.savefig("PriceGraph.png", dpi=300)


# Plot annual electricity and propane energy
years = summary_df["year"].to_numpy()

both_boilers_storage_hours_elec = np.array(both_boilers_storage_elec_MWh)
both_boilers_storage_hours_gas = np.array(both_boilers_storage_gas_MWh)
both_boilers_nostorage_hours_elec = np.array(both_boilers_nostorage_elec_MWh)
both_boilers_nostorage_hours_gas = np.array(both_boilers_nostorage_gas_MWh)

width = 0.2
x = np.arange(len(years))

plt.figure(figsize=(12, 6))

# Both boilers + Storage
plt.bar(x - width / 2, both_boilers_storage_hours_elec, width=width, color="green", label="Both boilers+Storage Elec")

plt.bar(
    x - width / 2,
    both_boilers_storage_hours_gas,
    width=width,
    bottom=both_boilers_storage_hours_elec,
    color="grey",
    label="Both boilers+Storage Gas",
)

# Both boilers no Storage
plt.bar(
    x + width / 2,
    both_boilers_nostorage_hours_elec,
    width=width,
    color="lightgreen",
    label="Both boilers no Storage Elec",
)

plt.bar(
    x + width / 2,
    both_boilers_nostorage_hours_gas,
    width=width,
    bottom=both_boilers_nostorage_hours_elec,
    color="lightgrey",
    label="Both boilers no Storage Gas",
)

plt.xlabel("Year")
plt.ylabel("Energy use electricity/gas (MWh)")
plt.title("Energy run on electricity and gas when having both boilers")
plt.xticks(x, years)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("Elec_and_propane_energy.png", dpi=300)
plt.close()

# Plot Electricity and gas use in a first week of January

years = summary_df["year"].to_numpy()

width = 0.2
x = np.arange(len(years))

plt.figure(figsize=(12, 6))

# Both boilers + Storage
plt.bar(
    x - 1.5 * width, both_boilers_storage_elec_week, width=width, color="green", label="Both boilers + Storage Elec"
)

plt.bar(x - 0.5 * width, both_boilers_storage_gas_week, width=width, color="grey", label="Both boilers + Storage Gas")

# Both boilers no Storage
plt.bar(
    x + 0.5 * width,
    both_boilers_nostorage_elec_week,
    width=width,
    color="lightgreen",
    label="Both boilers no Storage Elec",
)

plt.bar(
    x + 1.5 * width, both_boilers_nostorage_gas_week, width=width, color="darkgrey", label="Both boilers no Storage Gas"
)

plt.xlabel("Year")
plt.ylabel("Energy use electricity/gas (MWh)")
plt.title("Electricity vs Gas Use – First Week of Each Year")
plt.xticks(x, years)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("FirstWeek_FuelUse.png", dpi=300)
plt.close()

#   Plot Gas and electricity price correlation graphs
for year in years:
    df_y = df_prices.copy()

    df_y["Date_local"] = df_y["Date"].dt.tz_convert("Europe/Helsinki")

    df_y = df_y[(df_y["Date_local"] >= f"{year}-01-01") & (df_y["Date_local"] < f"{year + 1}-01-01")].copy()

    df_y = df_y.set_index("Date_local").sort_index()

    elec = df_y["total_price"].astype(float)
    gas = df_y["total_gas_price"].astype(float)

    plt.figure(figsize=(12, 6))
    plt.plot(df_y.index, elec, label="Electricity (EUR/MWh)")
    plt.plot(df_y.index, gas, label="Gas (EUR/MWh)")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.xticks(rotation=0)
    plt.xlim(df_y.index.min(), df_y.index.max())
    plt.legend(fontsize=16)
    plt.title(f"Electricity and Gas Prices – {year}")
    plt.xlabel("Date")
    plt.ylabel("Price (EUR/MWh)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Gas_Elec_Prices_{year}.png", dpi=300)
    plt.close()

#  Plot annual Gas boiler, electric boiler and discharge hours #
# Build x-axis (years) in the same order as results were computed
years_arr = np.array([r["year"] for r in results_summary])
x = np.arange(len(years_arr))

# Electric boiler steam hours
plt.figure(figsize=(10, 5))
plt.bar(x, el_steam_hours_per_year, width=0.35, color="#4C9F70")
plt.xticks(x, years_arr)
plt.xlabel("Year")
plt.ylabel("Hours")
plt.title("Electric Boiler Steam Hours per Year")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("Hours_ElectricSteam_byYear.png", dpi=300)
plt.close()

# Gas boiler steam hours
plt.figure(figsize=(10, 5))
plt.bar(x, gas_steam_hours_per_year, width=0.35, color="#7F7F7F")
plt.xticks(x, years_arr)
plt.xlabel("Year")
plt.ylabel("Hours")
plt.title("Gas Boiler Steam Hours per Year")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("Hours_GasSteam_byYear.png", dpi=300)
plt.close()

# Discharge hours
plt.figure(figsize=(10, 5))
plt.bar(x, discharge_hours_per_year, width=0.35, color="#1F77B4")
plt.xticks(x, years_arr)
plt.xlabel("Year")
plt.ylabel("Hours")
plt.title("Discharge Hours per Year")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("Hours_Discharge_byYear.png", dpi=300)
plt.close()
