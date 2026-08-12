"""Compare DESDEO district_heating_problem against the original code1.py PuLP formulation.

Runs both solvers on the same 36-hour price window (first window of 2022) and prints
a side-by-side comparison of costs and hourly dispatch decisions.

Usage::

    python district_heating/compare_code1.py
"""

import sys
from pathlib import Path

import numpy as np
import pulp

sys.path.insert(0, str(Path(__file__).parent))

from desdeo.tools.cvxpy_solver_interfaces import CVXPYSolver
from desdeo.tools.scalarization import add_weighted_sums
from district_heating import (
    CHARGING_EFF,
    DEMAND_MW,
    DISCHARGING_EFF,
    ELEC_BOILER_CAPACITY_MW,
    ELEC_BOILER_EFF,
    GAS_BOILER_CAPACITY_MW,
    GAS_BOILER_EFF,
    HEAT_LOSS_FRACTION,
    MAX_CHARGE_POWER_MW,
    MAX_DISCHARGE_POWER_MW,
    STORAGE_CAPACITY_MWH,
    district_heating_problem,
    load_district_heating_data,
)

DATA_DIR = Path("/home/juho/data/jenni")
YEAR = 2022
WINDOW_START = 3696  # h=3696 has 22/36 hours where gas is cheaper than electricity
WINDOW = 36  # same as code1.py rolling-horizon window size
INITIAL_STORAGE = 0.0


# ── PuLP reference (code1.py Scenario 3 logic) ───────────────────────────────


def solve_pulp(spot: np.ndarray, gas: np.ndarray) -> dict:
    """Replicates the code1.py Scenario 3 (both boilers + storage) LP for one window."""
    T = len(spot)  # noqa: N806
    el_steam_max = DEMAND_MW / ELEC_BOILER_EFF
    el_total_max = ELEC_BOILER_CAPACITY_MW / ELEC_BOILER_EFF
    gas_steam_max = DEMAND_MW / GAS_BOILER_EFF
    gas_total_max = GAS_BOILER_CAPACITY_MW / GAS_BOILER_EFF

    model = pulp.LpProblem("district_heating_code1", pulp.LpMinimize)

    el_steam = [pulp.LpVariable(f"el_steam_{t}", 0, el_steam_max) for t in range(T)]
    gas_steam = [pulp.LpVariable(f"gas_steam_{t}", 0, gas_steam_max) for t in range(T)]
    el_charge = [pulp.LpVariable(f"el_charge_{t}", 0, el_total_max) for t in range(T)]
    gas_charge = [pulp.LpVariable(f"gas_charge_{t}", 0, gas_total_max) for t in range(T)]
    discharge = [pulp.LpVariable(f"discharge_{t}", 0, MAX_DISCHARGE_POWER_MW) for t in range(T)]
    storage = [pulp.LpVariable(f"storage_{t}", 0, STORAGE_CAPACITY_MWH) for t in range(T + 1)]
    u_el = [pulp.LpVariable(f"u_el_{t}", 0, 1, cat="Binary") for t in range(T)]
    u_gas = [pulp.LpVariable(f"u_gas_{t}", 0, 1, cat="Binary") for t in range(T)]
    u_dis = [pulp.LpVariable(f"u_dis_{t}", 0, 1, cat="Binary") for t in range(T)]

    model += storage[0] == INITIAL_STORAGE

    for t in range(T):
        model += storage[t + 1] == (
            storage[t] * (1 - HEAT_LOSS_FRACTION)
            + (el_charge[t] * ELEC_BOILER_EFF + gas_charge[t] * GAS_BOILER_EFF) * CHARGING_EFF
            - discharge[t] / DISCHARGING_EFF
        )
        model += el_steam[t] * ELEC_BOILER_EFF + gas_steam[t] * GAS_BOILER_EFF + discharge[t] == DEMAND_MW
        model += el_steam[t] + el_charge[t] <= el_total_max
        model += gas_steam[t] + gas_charge[t] <= gas_total_max
        model += u_el[t] + u_gas[t] + u_dis[t] <= 1
        model += el_charge[t] * ELEC_BOILER_EFF <= MAX_CHARGE_POWER_MW * u_el[t]
        model += gas_charge[t] * GAS_BOILER_EFF <= MAX_CHARGE_POWER_MW * u_gas[t]
        model += discharge[t] <= DEMAND_MW * u_dis[t]

    model += pulp.lpSum(
        spot[t] * (el_steam[t] + el_charge[t]) + gas[t] * (gas_steam[t] + gas_charge[t]) for t in range(T)
    )

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    f_el = sum(spot[t] * (pulp.value(el_steam[t]) + pulp.value(el_charge[t])) for t in range(T))
    f_gas = sum(gas[t] * (pulp.value(gas_steam[t]) + pulp.value(gas_charge[t])) for t in range(T))

    return {
        "status": pulp.LpStatus[model.status],
        "f_el": f_el,
        "f_gas": f_gas,
        "f_total": f_el + f_gas,
        "el_steam": np.array([pulp.value(v) for v in el_steam]),
        "el_charge": np.array([pulp.value(v) for v in el_charge]),
        "gas_steam": np.array([pulp.value(v) for v in gas_steam]),
        "gas_charge": np.array([pulp.value(v) for v in gas_charge]),
        "discharge": np.array([pulp.value(v) for v in discharge]),
        # storage[t+1] in PuLP = end-of-hour-t state, matches DESDEO's storage[t]
        "storage": np.array([pulp.value(v) for v in storage[1:]]),
    }


# ── DESDEO solver ─────────────────────────────────────────────────────────────


def solve_desdeo(spot: np.ndarray, gas: np.ndarray) -> dict:
    """Solves the same problem via DESDEO CVXPYSolver, minimising f_el + f_gas."""
    prob = district_heating_problem(spot, gas, initial_storage_level=INITIAL_STORAGE)
    # Equal-weight sum replicates code1's single combined-cost objective
    prob_scal, scal_sym = add_weighted_sums(prob, "f_total", {"f_el": 1.0, "f_gas": 1.0})
    solver = CVXPYSolver(prob_scal)
    result = solver.solve(scal_sym)

    T = len(spot)  # noqa: N806
    vars_ = result.optimal_variables
    return {
        "status": "Optimal" if result.success else "Failed",
        "f_el": result.optimal_objectives["f_el"],
        "f_gas": result.optimal_objectives["f_gas"],
        "f_total": result.optimal_objectives["f_el"] + result.optimal_objectives["f_gas"],
        "el_steam": np.array(vars_["el_steam"]).flatten()[:T],
        "el_charge": np.array(vars_["el_charge"]).flatten()[:T],
        "gas_steam": np.array(vars_["gas_steam"]).flatten()[:T],
        "gas_charge": np.array(vars_["gas_charge"]).flatten()[:T],
        "discharge": np.array(vars_["discharge"]).flatten()[:T],
        "storage": np.array(vars_["storage"]).flatten()[:T],
    }


# ── Main ──────────────────────────────────────────────────────────────────────
# ruff: noqa: T201


def main():  # noqa: D103
    print(f"Loading {YEAR} prices from {DATA_DIR} ...")
    spot, gas = load_district_heating_data(DATA_DIR, YEAR)
    spot_w = spot[WINDOW_START : WINDOW_START + WINDOW]
    gas_w = gas[WINDOW_START : WINDOW_START + WINDOW]

    print(f"\nSolving with PuLP (CBC) — {WINDOW}-hour window (starting at h={WINDOW_START}) ...")
    p = solve_pulp(spot_w, gas_w)
    print(f"\nSolving with DESDEO (CVXPY/HiGHS) — {WINDOW}-hour window ...")
    d = solve_desdeo(spot_w, gas_w)

    print("\n" + "=" * 55)
    print(f"{'Metric':<28} {'PuLP':>12} {'DESDEO':>12}")
    print("=" * 55)
    print(f"{'Status':<28} {p['status']:>12} {d['status']:>12}")
    print(f"{'f_el  (electricity cost €)':<28} {p['f_el']:>12.4f} {d['f_el']:>12.4f}")
    print(f"{'f_gas (gas cost €)':<28} {p['f_gas']:>12.4f} {d['f_gas']:>12.4f}")
    print(f"{'f_total (combined cost €)':<28} {p['f_total']:>12.4f} {d['f_total']:>12.4f}")
    print("=" * 55)

    # Max absolute difference over all dispatch variables
    diffs = {}
    for key in ("el_steam", "el_charge", "gas_steam", "gas_charge", "discharge", "storage"):
        diffs[key] = np.max(np.abs(p[key] - d[key]))

    print("\nMax |PuLP - DESDEO| per dispatch variable (MW or MWh):")
    for key, diff in diffs.items():
        print(f"  {key:<12}: {diff:.6f}")

    max_cost_diff = abs(p["f_total"] - d["f_total"])
    rel_diff = max_cost_diff / max(abs(p["f_total"]), 1e-9) * 100
    print(f"\nTotal cost difference: {max_cost_diff:.4f} € ({rel_diff:.4f} %)")

    print("\nHourly dispatch comparison (first 12 hours):")
    print(
        f"{'h':>3}  {'p_el':>6}  {'p_gas':>6}  "
        f"{'el_stm_P':>9}{'el_stm_D':>9}  "
        f"{'gs_stm_P':>9}{'gs_stm_D':>9}  "
        f"{'stor_P':>7}{'stor_D':>7}"
    )
    for t in range(min(12, WINDOW)):
        print(
            f"{t:>3}  {spot_w[t]:>6.1f}  {gas_w[t]:>6.1f}  "
            f"{p['el_steam'][t]:>9.4f}{d['el_steam'][t]:>9.4f}  "
            f"{p['gas_steam'][t]:>9.4f}{d['gas_steam'][t]:>9.4f}  "
            f"{p['storage'][t]:>7.4f}{d['storage'][t]:>7.4f}"
        )


if __name__ == "__main__":
    main()
