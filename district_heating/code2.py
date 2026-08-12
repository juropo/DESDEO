"""Economic and emissions post-processing of five industrial steam scenarios.

Post-processes the results produced by code1.py to compute:

* Levelised Cost of Heat (LCOH) per scenario and year (2019-2025)
* Annual CO2 emissions per scenario
* Emission-reduction cost (€/tCO2) relative to the gas-only baseline
* Sensitivity analyses for storage capacity, storage investment cost, and WACC

Reads ``all_years_results.xlsx`` (written by code1.py) and a separate
``sensitivity_analysis.xlsx``.  Outputs are written to
``lcoh_emission_results.xlsx`` and PNG figures.

Reference:
    Leinonen, J. (2026). *Economics and CO2 Emission Reductions of Electrifying
    Industrial Steam Production in Volatile Electricity Markets*.
    Master's thesis, Aalto University.
"""

# ruff: noqa

import matplotlib.pyplot as plt
import numpy_financial as npf
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator

# Define parameters
investment_cost_elec_boiler = 100000  # per MW
elec_boiler_capacity = 2  # MW
investment_cost_storage = 200000  # per MWh
storage_capacity = 5  # MWh
emissionfactor_gas = 200.052  # kg/MWh
emissionfactor_electricity = 26  # kg/MWh
WACC = 0.05
lifetime = 30
yearly_demand = 8760  # 8760 MW

# ---  Excel file for energy costs and used energy in each scenario for 2019-2025 ---
energycost_file = r"C:\Users\leinonj10\OneDrive - Aalto University\all_years_results.xlsx"


# --------functions-------#
def cost_averages_for_scenarios():
    # read the specific sheet
    df = pd.read_excel(energycost_file, sheet_name="Summary")

    # compute column means with pandas
    elec_average = df["electric_only"].mean()
    elecstorage_average = df["storage_only"].mean()
    both_boilers_storage_average = df["both_boilers_storage"].mean()
    both_boilers_average = df["both_boilers_nostorage"].mean()
    gas_average = df["gas_only"].mean()

    return elec_average, elecstorage_average, both_boilers_storage_average, both_boilers_average, gas_average


def emission_averages_from_summary(emissions_summary_df):
    columns = ["electric_only", "storage_only", "both_boilers_storage", "both_boilers_nostorage", "gas_only"]
    return emissions_summary_df[columns].mean()


def lcoh(WACC, lifetime, investment_cost, energy_cost, yearly_demand):
    discounted_costs = 0
    discounted_heat = 0

    for t in range(1, lifetime + 1):
        discounted_costs += energy_cost / (1 + WACC) ** t
        discounted_heat += yearly_demand / (1 + WACC) ** t

    lcoh = (investment_cost + discounted_costs) / discounted_heat

    return lcoh


def emissions(gas_energy, electricity_energy, emissionfactor_gas, emissionfactor_electricity):
    return gas_energy * emissionfactor_gas + electricity_energy * emissionfactor_electricity


# emission reduction costs (base on LCOH)
def cost_of_emission_reduction(costreduction, emissionreduction):
    return costreduction / emissionreduction


# sensitivity analysis for storage size
def plot_sensitivity_analysis(excel, investment_cost_storage):
    # Extract storage capacities and total cost benefits
    capacity_df = pd.read_excel(excel, index_col=0)

    # Storage capacities
    storage_capacities = capacity_df.columns.str.replace("MWh", "", regex=False).astype(float).tolist()

    # Cost benefits
    total_cost_benefits = capacity_df.loc["Cost benefits"].tolist()

    # Capital costs
    capital_costs = [investment_cost_storage * cap for cap in storage_capacities]

    # Calculate IRR
    irr_values = []
    for capex, cost_benefit in zip(capital_costs, total_cost_benefits):
        if cost_benefit == 0:
            irr_values.append(None)
            continue

        # Cash flow: initial investment + yearly benefits
        cash_flows = [-capex] + [cost_benefit] * lifetime

        irr = npf.irr(cash_flows)
        irr_values.append(irr)

    irr_values_percent = [irr * 100 if irr is not None else None for irr in irr_values]
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot on primary y-axis (left)
    ax1.plot(storage_capacities, total_cost_benefits, "o-", color="blue", label="Cost benefit in a year")
    ax1.plot(storage_capacities, capital_costs, "s--", color="red", label="Capital cost")

    # --- x-axis: Storage capacity in MWh ---
    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{round(x, 1):.1f} "))
    ax1.set_xlabel("Storage capacity (MWh)", fontsize=21, labelpad=3)
    ax1.set_ylabel("Cost (€)", color="black", fontsize=21)

    ax1.grid(True, which="major", linestyle="-", linewidth=0.75)
    ax1.minorticks_on()
    ax1.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.5)

    # Create secondary y-axis (right)
    ax2 = ax1.twinx()
    ax2.plot(storage_capacities, irr_values_percent, "d-", color="green", label="IRR (%)")
    ax2.set_ylabel("IRR (%)", color="green", fontsize=21)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax2.tick_params(axis="y", labelcolor="green")

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper center",
        bbox_to_anchor=(0.52, -0.20),
        ncol=3,
        frameon=False,
        fontsize=19,
    )

    plt.title("Sensitivity Analysis: Storage size", fontsize=22)
    plt.tight_layout()
    plt.savefig("Sensitivityanalysis_storages_x.png", dpi=300, bbox_inches="tight")

    return storage_capacities, total_cost_benefits, irr_values


# sensitivity analysis for investment costs
def plot_sensitivity_analysis2(excel, storage_capacity):
    # Extract storage capacities and total cost benefits
    cost_df = pd.read_excel(excel, index_col=0)

    # Investment costs
    investment_costs2 = cost_df.loc["Investment costs storage € / MWh"].astype(float).tolist()

    # Cost benefits
    total_cost_benefits2 = cost_df.loc["Cost benefits (investment costs)"].tolist()

    # Capital costs
    capital_costs2 = [storage_capacity * inv for inv in investment_costs2]

    # Calculate IRR
    irr_values2 = []
    for capex, cost_benefit in zip(capital_costs2, total_cost_benefits2):
        if cost_benefit == 0:
            irr_values2.append(None)
            continue

        # Cash flow: initial investment + yearly benefits
        cash_flows2 = [-capex] + [cost_benefit] * lifetime

        irr2 = npf.irr(cash_flows2)
        irr_values2.append(irr2)
    irr_values_percent2 = [irr2 * 100 if irr2 is not None else None for irr2 in irr_values2]
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot on primary y-axis (left)
    ax1.plot(investment_costs2, total_cost_benefits2, "o-", color="blue", label="Cost benefit in a year")
    ax1.plot(investment_costs2, capital_costs2, "s--", color="red", label="Investment cost")

    # --- x-axis: Storage capacity in MWh ---
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{round(x, 1):.1f} "))
    ax1.set_xlabel("Investment costs (€/MWh)", fontsize=21, labelpad=3)
    ax1.set_ylabel("Cost (€)", color="black", fontsize=21)

    ax1.grid(True, which="major", linestyle="-", linewidth=0.75)
    ax1.minorticks_on()
    ax1.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.5)

    # Create secondary y-axis (right)
    ax2 = ax1.twinx()
    ax2.plot(investment_costs2, irr_values_percent2, "d-", color="green", label="IRR (%)")
    ax2.set_ylabel("IRR (%)", color="green", fontsize=21)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax2.tick_params(axis="y", labelcolor="green")

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper center",
        bbox_to_anchor=(0.52, -0.20),
        ncol=3,
        frameon=False,
        fontsize=19,
    )
    plt.title("Sensitivity Analysis: Investment Cost", fontsize=22)
    plt.tight_layout()
    plt.savefig("Sensitivityanalysis_capex_x.png", dpi=300, bbox_inches="tight")
    return investment_costs2, total_cost_benefits2, irr_values2


# sensitivity analysis for WACC
def plot_sensitivity_analysis3(excel, storage_capacity):
    # Extract storage capacities and total cost benefits
    wacc_df = pd.read_excel(excel, index_col=0)

    # WACCs
    waccs = wacc_df.loc["WACC"].astype(float).tolist()
    waccs_percent = [w * 100 for w in waccs]

    # Cost benefits
    total_cost_benefit3 = wacc_df.loc["Cost benefits (WACCs)"]

    # Capital costs
    capital_costs3 = storage_capacity * investment_cost_storage

    # Calculate NPV
    npv_values = []
    for wacc, cost_benefit in zip(waccs, total_cost_benefit3):
        if cost_benefit == 0:
            npv_values.append(None)
            continue

        # Cash flow: initial investment + yearly benefits
        cash_flow = [-capital_costs3] + [cost_benefit] * lifetime

        npv = sum(cf / ((1 + wacc) ** t) for t, cf in enumerate(cash_flow))
        npv_values.append(npv)

    # Create figure and primary axis

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot on primary y-axis (left)
    ax1.plot(waccs_percent, total_cost_benefit3, "o-", color="blue", label="Cost benefit in a year")
    ax1.plot(waccs_percent, [capital_costs3] * len(waccs_percent), "s--", color="red", label="Investment cost")

    # --- x-axis: Storage capacity in MWh ---
    ax1.set_xlabel("WACC %", fontsize=21, labelpad=3)
    ax1.set_ylabel("Cost (€)", color="black", fontsize=21)

    ax1.grid(True, which="major", linestyle="-", linewidth=0.75)
    ax1.minorticks_on()
    ax1.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.5)

    # Create secondary y-axis (right)
    ax2 = ax1.twinx()
    ax2.plot(waccs_percent, npv_values, "d-", color="green", label="NPV")
    ax2.set_ylabel("NPV", color="green", fontsize=21)
    ax2.tick_params(axis="y", labelcolor="green")

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper center",
        bbox_to_anchor=(0.52, -0.20),
        ncol=3,
        frameon=False,
        fontsize=19,
    )
    plt.title("Sensitivity Analysis: WACC", fontsize=22)
    plt.tight_layout()
    plt.savefig("Sensitivityanalysis_wacc_x.png", dpi=300, bbox_inches="tight")
    return waccs, total_cost_benefit3, npv_values


# -------Calculations------------

# --- #average prices for each scenario:---
summary_df = pd.read_excel(energycost_file, sheet_name="Summary")
energy_use_df = pd.read_excel(energycost_file, sheet_name="Energy_use")

lcoh_summary = []
emissions_summary = []
reductioncost_summary = []
storage_summary = []
investment_cost_summary = []

with pd.ExcelWriter("lcoh_emission_results.xlsx", engine="openpyxl") as writer:
    # --- #LCOH for each scenario:---
    for _, year_data in summary_df.iterrows():
        year = year_data["year"]

        # pick the matching row from Energy_use
        energy_year = energy_use_df[energy_use_df["year"] == year].iloc[0]

        # LCOH per scenario
        lcoh_spot = lcoh(
            WACC,
            lifetime,
            investment_cost_elec_boiler * elec_boiler_capacity,
            year_data["electric_only"],
            yearly_demand,
        )
        lcoh_el_sto = lcoh(
            WACC,
            lifetime,
            investment_cost_elec_boiler * elec_boiler_capacity + investment_cost_storage * storage_capacity,
            year_data["storage_only"],
            yearly_demand,
        )
        lcoh_both_boilers_storage = lcoh(
            WACC,
            lifetime,
            investment_cost_elec_boiler * elec_boiler_capacity + investment_cost_storage * storage_capacity,
            year_data["both_boilers_storage"],
            yearly_demand,
        )
        lcoh_both_boilers_nostorage = lcoh(
            WACC,
            lifetime,
            investment_cost_elec_boiler * elec_boiler_capacity,
            year_data["both_boilers_nostorage"],
            yearly_demand,
        )
        lcoh_gas = lcoh(WACC, lifetime, 0, year_data["gas_only"], yearly_demand)

        lcoh_summary.append(
            {
                "year": year,
                "electric_only": lcoh_spot,
                "storage_only": lcoh_el_sto,
                "both_boilers_storage": lcoh_both_boilers_storage,
                "both_boilers_nostorage": lcoh_both_boilers_nostorage,
                "gas_only": lcoh_gas,
            }
        )

        # --- #Emissions for each scenario:---#

        # Electricity-only (spot)
        spot_emissions = emissions(
            gas_energy=0,
            electricity_energy=energy_year["electric_nostorage_MWh"],
            emissionfactor_gas=emissionfactor_gas,
            emissionfactor_electricity=emissionfactor_electricity,
        )

        # Electricity with storage
        storageonly_emissions = emissions(
            gas_energy=0,
            electricity_energy=energy_year["electric_storage_MWh"],
            emissionfactor_gas=emissionfactor_gas,
            emissionfactor_electricity=emissionfactor_electricity,
        )

        # Both boilers with storage
        both_boilers_storage_emissions = emissions(
            gas_energy=energy_year["both_boilers_storage_gas_MWh"],
            electricity_energy=energy_year["both_boilers_storage_elec_MWh"],
            emissionfactor_gas=emissionfactor_gas,
            emissionfactor_electricity=emissionfactor_electricity,
        )

        # Both boilers without storage
        both_boilers_nostorage_emissions = emissions(
            gas_energy=energy_year["both_boilers_nostorage_gas_MWh"],
            electricity_energy=energy_year["both_boilers_nostorage_elec_MWh"],
            emissionfactor_gas=emissionfactor_gas,
            emissionfactor_electricity=emissionfactor_electricity,
        )

        # Gas only
        gas_emissions = emissions(
            gas_energy=energy_year["gas_only_MWh"],
            electricity_energy=0,
            emissionfactor_gas=emissionfactor_gas,
            emissionfactor_electricity=emissionfactor_electricity,
        )

        emissions_summary.append(
            {
                "year": year,
                "electric_only": spot_emissions,
                "storage_only": storageonly_emissions,
                "both_boilers_storage": both_boilers_storage_emissions,
                "both_boilers_nostorage": both_boilers_nostorage_emissions,
                "gas_only": gas_emissions,
            }
        )

        # Convert results to DataFrames for further analysis and plotting
        lcoh_summary_df = pd.DataFrame(lcoh_summary)
        emissions_summary_df = pd.DataFrame(emissions_summary)
        lcoh_summary_df.to_excel(writer, sheet_name="lcoh_summary", index=False)
        emissions_summary_df.to_excel(writer, sheet_name="emissions_summary", index=False)

    # -----Emission reduction profits for each scenario -------

    # compute column means with pandas
    ele_average = lcoh_summary_df["electric_only"].mean()
    elestorage_average = lcoh_summary_df["storage_only"].mean()
    hybstorage_average = lcoh_summary_df["both_boilers_storage"].mean()
    hyb_average = lcoh_summary_df["both_boilers_nostorage"].mean()
    g_average = lcoh_summary_df["gas_only"].mean()

    # Emission averages
    emissions_averages = emissions_summary_df[
        ["electric_only", "storage_only", "both_boilers_storage", "both_boilers_nostorage", "gas_only"]
    ].mean()

    print(" Profit of emission reductions €/kgCO2")

    # Spot only
    costreduction = (g_average - ele_average) * 8760
    emissionreduction = emissions_averages["gas_only"] - emissions_averages["electric_only"]
    cost_of_reduction_spot = cost_of_emission_reduction(costreduction, emissionreduction)
    print(f"  1. Electric only:        {cost_of_reduction_spot:,.4f} €/kgCO2")

    # Electricity with storage
    costreduction = (g_average - elestorage_average) * 8760
    emissionreduction = emissions_averages["gas_only"] - emissions_averages["storage_only"]
    cost_of_reduction_storage = cost_of_emission_reduction(costreduction, emissionreduction)
    print(f"  2. Electric with storage:        {cost_of_reduction_storage:,.4f} €/kgCO2")

    # Both boilers with storage
    costreduction = (g_average - hybstorage_average) * 8760
    emissionreduction = emissions_averages["gas_only"] - emissions_averages["both_boilers_storage"]
    cost_of_reduction_both_boilers_storage = cost_of_emission_reduction(costreduction, emissionreduction)
    print(f"  3. Both boilers with storage:        {cost_of_reduction_both_boilers_storage:,.4f} €/kgCO2")

    # Both boilers without storage
    costreduction = (g_average - hyb_average) * 8760
    emissionreduction = emissions_averages["gas_only"] - emissions_averages["both_boilers_nostorage"]
    cost_of_reduction_both_boilers_nostorage = cost_of_emission_reduction(costreduction, emissionreduction)
    print(f"  4. Both boilers without storage:        {cost_of_reduction_both_boilers_nostorage:,.4f} €/kgCO2")

    reductioncost_summary.append(
        {
            "electric_only": cost_of_reduction_spot,
            "storage_only": cost_of_reduction_storage,
            "both_boilers_storage": cost_of_reduction_both_boilers_storage,
            "both_boilers_nostorage": cost_of_reduction_both_boilers_nostorage,
        }
    )

    reductions_summary_df = pd.DataFrame(reductioncost_summary)
    reductions_summary_df.to_excel(writer, sheet_name="reductions_summary", index=False)

    # -------Sensitivity analyses-------#
    excel = r"C:\Users\leinonj10\OneDrive - Aalto University\sensitivity_analysis.xlsx"

    # Storage capacity
    storage_capacities, total_cost_benefits, irr_values = plot_sensitivity_analysis(excel, investment_cost_storage)

    print("\n=== Sensitivity analysis – STORAGE SIZES ===")
    for cap, benefit, irr in zip(storage_capacities, total_cost_benefits, irr_values):
        print(f"\nStorage capacity: {cap:.0f} MWh")
        print(f"Cost benefit: {benefit:,.0f}")
        print(f"IRR: {irr:.2f} ")

        storage_summary.append({"storage_capacity": cap, "Cost benefit": benefit, "IRR": irr})
        storage_summary_df = pd.DataFrame(storage_summary)
        storage_summary_df.to_excel(writer, sheet_name="storage_summary", index=False)

    ### Investment cost
    investment_costs2, total_cost_benefits2, irr2 = plot_sensitivity_analysis2(excel, storage_capacity)

    print("\n=== Sensitivity analysis – INVESTMENT COSTS ===")
    for cap, benefit, irr in zip(investment_costs2, total_cost_benefits2, irr2):
        print(f"\nInvestment cost: {cap:.0f} €/MWh")
        print(f"Cost benefit: {benefit:,.0f}")
        print(f"IRR: {irr:.2f}")

    #### WACC ##
    waccs, total_cost_benefit3, npv = plot_sensitivity_analysis3(excel, storage_capacity)

    print("\n=== Sensitivity analysis – WACCs ===")
    for wacc, benefit, npv in zip(waccs, total_cost_benefit3, npv):
        print(f"\nWACCs: {wacc * 100:.0f} %")
        print(f"Cost benefit: {benefit:,.0f} €")
        print(f"NPV: {npv:.2f}€")

# ------PLOTS------

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

# ------LCOH plot------

plt.figure(figsize=(10, 6))

plt.plot(lcoh_summary_df["year"], lcoh_summary_df["electric_only"], marker="o", label="Electric only")
plt.plot(lcoh_summary_df["year"], lcoh_summary_df["storage_only"], marker="o", label="Storage (electric)")
plt.plot(lcoh_summary_df["year"], lcoh_summary_df["both_boilers_storage"], marker="o", label="Both boilers + Storage")
plt.plot(
    lcoh_summary_df["year"], lcoh_summary_df["both_boilers_nostorage"], marker="o", label="Both boilers no storage"
)
plt.plot(lcoh_summary_df["year"], lcoh_summary_df["gas_only"], marker="o", label="Gas only")

plt.xlabel("Year")
plt.ylabel("LCOH (€/MWh)")
plt.title("Annual LCOH Comparison 2019–2025")
plt.grid(True)
plt.legend()
plt.xticks(summary_df["year"])

plt.tight_layout()
plt.savefig("LCOHGraph.png", dpi=300)

# ------Emission plot------

plt.figure(figsize=(10, 6))

plt.plot(
    emissions_summary_df["year"],
    emissions_summary_df["electric_only"] / 1000000,
    marker="o",
    label="Electric only",
    alpha=0.8,
)
plt.plot(
    emissions_summary_df["year"],
    emissions_summary_df["storage_only"] / 1000000,
    marker="o",
    label="Storage (electric)",
    alpha=0.8,
)
plt.plot(
    emissions_summary_df["year"],
    emissions_summary_df["both_boilers_storage"] / 1000000,
    marker="o",
    label="Both boilers + Storage",
    alpha=0.8,
)
plt.plot(
    emissions_summary_df["year"],
    emissions_summary_df["both_boilers_nostorage"] / 1000000,
    marker="o",
    label="Both boilers no storage",
    alpha=0.8,
)
plt.plot(
    emissions_summary_df["year"], emissions_summary_df["gas_only"] / 1000000, marker="o", label="Gas only", alpha=0.8
)

plt.xlabel("Year")
plt.ylabel("Annual Emissions (M kg CO₂)")
plt.title("Annual Emission Comparison 2019–2025")
plt.grid(True)
plt.legend()
plt.xticks(summary_df["year"])

plt.tight_layout()
plt.savefig("EmissionGraph.png", dpi=300)

# -----Emission reduction plot----#

width = 0.35
x = 4

plt.figure(figsize=(12, 6))

# Spot only
plt.bar(1, cost_of_reduction_spot * 1000, width=width, color="blue", label="Electric only")

# Elec with storage
plt.bar(2, cost_of_reduction_storage * 1000, width=width, color="orange", label="Electric with storage")

# Both boilers with Storage
plt.bar(3, cost_of_reduction_both_boilers_storage * 1000, width=width, color="green", label="Both boilers with storage")

# Both boilers without Storage
plt.bar(
    4, cost_of_reduction_both_boilers_nostorage * 1000, width=width, color="red", label="Both boilers without Storage"
)

plt.xlabel("Scenario")
plt.ylabel("Profit €/tCO2")
plt.title("Emission reduction Profits ")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("Emission_reduction_profits.png", dpi=300)
plt.close()
