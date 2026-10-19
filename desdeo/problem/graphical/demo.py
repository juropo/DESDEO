"""Small, hand-built influence diagrams used to exercise the compiler end to end.

- `build_irrigation_demo`: a single-objective diagram (weather -> soil moisture -> crop yield,
  irrigation decision, profit objective). Backend counterpart to the webui's
  `sampleIrrigationNetwork` fixture (`webui/.../network/sample-network.ts`).
- `build_cabin_energy_demo`: a two-objective diagram with a decision, used as a realistic
  multi-objective CUMULUS problem.

The irrigation diagram is not a literal copy of `sample-network.ts`, because the schema can't
express everything the frontend fixture does:

- A chance node's CPT can't be conditioned on a decision, so `crop_yield` is split into one
  counterfactual node per irrigation choice (`crop_yield_none`/`_light`/`_heavy`), each
  conditioned on `soil_moisture` only. `profit`'s utility table selects the node matching
  `irrigate`'s state. Since the three are conditionally independent given `soil_moisture`, this
  gives the same joint distribution as a decision-conditioned CPT would.
- There is no `"deterministic"` node kind, so `water_usage` is folded into `profit`'s utility
  table as a function of `irrigate`.
- The second objective, `water_cost`, is dropped: with more than one objective and a decision,
  `_compute_utilities` in `desdeo/api/routers/graphical.py` raises 501, and `/whatif` calls it on
  every request. `profit` already nets out water cost.
"""

import numpy as np

from desdeo.problem.graphical.network import DiscreteNetwork, DiscreteVariable, Potential
from desdeo.problem.graphical.schema import DecisionAnnotation, DecisionNode, ObjectiveNode

WEATHER_STATES = ["dry", "normal", "wet"]
SOIL_MOISTURE_STATES = ["low", "medium", "high"]
YIELD_STATES = ["poor", "average", "good"]
IRRIGATE_STATES = ["none", "light", "heavy"]

_YIELD_NODES = {"none": "crop_yield_none", "light": "crop_yield_light", "heavy": "crop_yield_heavy"}
_REVENUE_BY_YIELD = {"poor": 1000.0, "average": 2500.0, "good": 4000.0}
_WATER_COST_BY_IRRIGATION = {"none": 0.0, "light": 150.0, "heavy": 400.0}


def build_irrigation_demo_network() -> DiscreteNetwork:
    """The chance-node half of the demo diagram: `weather -> soil_moisture -> crop_yield_*`.

    The `crop_yield_*` nodes are conditionally independent given `soil_moisture`. The numbers are
    illustrative: irrigation helps at low soil moisture, and heavy irrigation is slightly worse
    than light at high soil moisture (waterlogging).
    """
    # P(soil_moisture | weather), rows = soil_moisture, columns = weather, each column sums to 1.
    soil_moisture_given_weather = np.array(
        [
            [0.60, 0.20, 0.05],  # low
            [0.35, 0.60, 0.35],  # medium
            [0.05, 0.20, 0.60],  # high
        ]
    )
    # P(crop_yield_<choice> | soil_moisture), rows = yield, columns = soil_moisture.
    yield_given_soil_moisture = {
        "none": np.array(
            [
                [0.70, 0.20, 0.05],  # poor
                [0.25, 0.60, 0.35],  # average
                [0.05, 0.20, 0.60],  # good
            ]
        ),
        "light": np.array(
            [
                [0.40, 0.10, 0.05],
                [0.45, 0.50, 0.30],
                [0.15, 0.40, 0.65],
            ]
        ),
        "heavy": np.array(
            [
                [0.15, 0.05, 0.10],
                [0.45, 0.35, 0.30],
                [0.40, 0.60, 0.60],
            ]
        ),
    }

    variables = [
        DiscreteVariable(name="weather", labels=WEATHER_STATES, description="Growing-season weather outlook."),
        DiscreteVariable(
            name="soil_moisture",
            labels=SOIL_MOISTURE_STATES,
            description="Soil moisture at planting, driven by weather.",
        ),
    ]
    arcs = [("weather", "soil_moisture")]
    potentials = {
        "weather": Potential(variables=["weather"], shape=[3], values=[0.3, 0.4, 0.3]),
        "soil_moisture": Potential(
            variables=["soil_moisture", "weather"],
            shape=[3, 3],
            values=soil_moisture_given_weather.tolist(),
        ),
    }

    for choice, node_name in _YIELD_NODES.items():
        variables.append(
            DiscreteVariable(
                name=node_name,
                labels=YIELD_STATES,
                description=f"Counterfactual crop yield if irrigation choice were '{choice}'.",
            )
        )
        arcs.append(("soil_moisture", node_name))
        potentials[node_name] = Potential(
            variables=[node_name, "soil_moisture"], shape=[3, 3], values=yield_given_soil_moisture[choice].tolist()
        )

    return DiscreteNetwork(name="irrigation_demo", variables=variables, arcs=arcs, potentials=potentials)


def build_irrigation_demo_annotation() -> DecisionAnnotation:
    """The decision/objective overlay for `build_irrigation_demo_network`.

    `irrigate` picks one of `IRRIGATE_STATES` with no prior observation. `profit` depends on
    `irrigate` and all three counterfactual yield nodes, but only the node matching `irrigate`'s
    state affects its value. Water cost per choice is netted out of `profit`.
    """
    yield_node_order = [_YIELD_NODES[choice] for choice in IRRIGATE_STATES]
    profit_shape = [len(IRRIGATE_STATES), *([len(YIELD_STATES)] * len(yield_node_order))]
    profit_values = np.zeros(profit_shape)
    for i, irrigate_choice in enumerate(IRRIGATE_STATES):
        # `yield_node_order` follows `IRRIGATE_STATES`, so the relevant yield axis is `i`.
        realized_axis = i
        for realized_state_idx, realized_state in enumerate(YIELD_STATES):
            revenue = _REVENUE_BY_YIELD[realized_state] - _WATER_COST_BY_IRRIGATION[irrigate_choice]
            index = [i] + [slice(None)] * len(yield_node_order)
            index[1 + realized_axis] = realized_state_idx
            profit_values[tuple(index)] = revenue

    return DecisionAnnotation(
        decisions=[DecisionNode(symbol="irrigate", name="Irrigate", states=IRRIGATE_STATES)],
        objectives=[
            ObjectiveNode(
                symbol="profit",
                name="Profit",
                parents=["irrigate", *yield_node_order],
                utility=Potential(
                    variables=["irrigate", *yield_node_order], shape=profit_shape, values=profit_values.tolist()
                ),
                maximize=True,
            ),
        ],
    )


def build_irrigation_demo() -> tuple[DiscreteNetwork, DecisionAnnotation]:
    """`(network, annotation)` for the CUMULUS network-structure demo dialog. See module docstring."""
    network = build_irrigation_demo_network()
    annotation = build_irrigation_demo_annotation()
    annotation.validate_against(network)
    return network, annotation


CABIN_WEATHER_STATES = ["sunny", "cloudy"]
SOLAR_STATES = ["low", "high"]
GRID_STATES = ["stable", "outage"]
POWER_SOURCE_STATES = ["grid", "battery"]


def build_cabin_energy_network() -> DiscreteNetwork:
    """The chance-node half of the demo diagram: `weather -> solar_output`, plus independent `grid_outage`.

    `grid_outage` is an exogenous risk independent of the weather, so it has no parents.
    """
    # P(solar_output | weather), rows = solar_output, columns = weather, each column sums to 1.
    solar_given_weather = [
        [0.2, 0.8],  # low
        [0.8, 0.2],  # high
    ]

    variables = [
        DiscreteVariable(name="weather", labels=CABIN_WEATHER_STATES, description="Today's weather outlook."),
        DiscreteVariable(
            name="solar_output", labels=SOLAR_STATES, description="Solar panel output, driven by weather."
        ),
        DiscreteVariable(
            name="grid_outage",
            labels=GRID_STATES,
            description="Grid reliability -- an independent exogenous risk, not caused by weather.",
        ),
    ]
    arcs = [("weather", "solar_output")]
    potentials = {
        "weather": Potential(variables=["weather"], shape=[2], values=[0.6, 0.4]),
        "solar_output": Potential(variables=["solar_output", "weather"], shape=[2, 2], values=solar_given_weather),
        "grid_outage": Potential(variables=["grid_outage"], shape=[2], values=[0.9, 0.1]),
    }

    return DiscreteNetwork(name="cabin_energy_demo", variables=variables, arcs=arcs, potentials=potentials)


def build_cabin_energy_annotation() -> DecisionAnnotation:
    """The decision/objective overlay for `build_cabin_energy_network`.

    `power_source` picks `grid` or `battery` with no prior observation. There are two objectives,
    `expected_cost` and `expected_unmet_demand`, which depend on `power_source`, `solar_output`
    and `grid_outage`.

    Known limitation: `_compute_utilities` (`desdeo/api/routers/graphical.py`) raises 501 for
    diagrams with more than one objective and a decision, so every `/whatif` call against this
    problem fails. In the webui's network dialog this means no priors are shown, and pinning
    evidence shows an error. CUMULUS itself does not use `/whatif` and is unaffected.
    """
    objective_shape = [len(POWER_SOURCE_STATES), len(SOLAR_STATES), len(GRID_STATES)]
    parents = ["power_source", "solar_output", "grid_outage"]

    # Indexed [power_source][solar_output][grid_outage].
    cost_values = [
        [[5.0, 50.0], [5.0, 50.0]],  # grid: low/high solar x stable/outage
        [[20.0, 20.0], [0.0, 0.0]],  # battery: low/high solar x stable/outage
    ]
    unmet_demand_values = [
        [[0.0, 3.0], [0.0, 3.0]],  # grid
        [[1.0, 1.0], [0.0, 0.0]],  # battery
    ]

    return DecisionAnnotation(
        decisions=[DecisionNode(symbol="power_source", name="Power source", states=POWER_SOURCE_STATES)],
        objectives=[
            ObjectiveNode(
                symbol="expected_cost",
                name="Expected electricity cost",
                parents=parents,
                utility=Potential(variables=parents, shape=objective_shape, values=cost_values),
                maximize=False,
            ),
            ObjectiveNode(
                symbol="expected_unmet_demand",
                name="Expected unmet demand",
                parents=parents,
                utility=Potential(variables=parents, shape=objective_shape, values=unmet_demand_values),
                maximize=False,
            ),
        ],
    )


def build_cabin_energy_demo() -> tuple[DiscreteNetwork, DecisionAnnotation]:
    """`(network, annotation)` for a two-objective cabin-energy CUMULUS problem. See `build_cabin_energy_annotation`."""
    network = build_cabin_energy_network()
    annotation = build_cabin_energy_annotation()
    annotation.validate_against(network)
    return network, annotation
