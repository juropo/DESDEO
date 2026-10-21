"""Small, hand-built influence diagrams used to exercise the compiler end to end.

- `build_irrigation_demo`: a single-objective diagram (weather -> soil moisture -> crop yield,
  irrigation decision, profit objective). Backend counterpart to the webui's
  `sampleIrrigationNetwork` fixture (`webui/.../network/sample-network.ts`).
- `build_oil_wildcatter_demo`: the classic Raiffa "Oil Wildcatter" case (seismic test, drill
  decision), with three objectives and two sequential decisions.

The irrigation diagram differs from `sample-network.ts`:

- `crop_yield` is split into one counterfactual node per irrigation choice, each conditioned on
  `soil_moisture` only; `profit`'s utility table selects the one matching `irrigate`.
- `water_usage` is folded into `profit`'s utility table as a function of `irrigate`.
- There is no `water_cost` objective (so `/whatif` works without pinning `irrigate`); `profit`
  already nets out water cost.
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


STRUCTURE_STATES = ["dry", "wet", "soaking"]
TEST_RESULT_STATES = ["NS", "OS", "CS", "no_info"]
OIL_PRICE_STATES = ["low", "baseline", "high"]
TEST_STATES = ["yes", "no"]
DRILL_STATES = ["yes", "no"]

STRUCTURE_PRIOR = [0.5, 0.3, 0.2]
# Asymmetric on purpose: E[price] = 97, not the $100 baseline, so `oil_price` affects expected profit.
OIL_PRICE_PRIOR = [0.3, 0.5, 0.2]

# P(test_result | structure, test=yes): the case's own NS/OS/CS joint-probability table, divided
# by its own structure marginals. `no_info` is impossible once the test is actually run.
_TEST_RESULT_GIVEN_STRUCTURE_WHEN_TESTED = {
    "dry": [0.6, 0.3, 0.1, 0.0],
    "wet": [0.3, 0.4, 0.3, 0.0],
    "soaking": [0.1, 0.4, 0.5, 0.0],
}

_BARRELS_BY_STRUCTURE = {"dry": 0.0, "wet": 1200.0, "soaking": 2700.0}
_PRICE_PER_BARREL_BY_OIL_PRICE = {"low": 70.0, "baseline": 100.0, "high": 130.0}
_TEST_COST = 10_000.0
_DRILL_COST = 70_000.0


def build_oil_wildcatter_network() -> DiscreteNetwork:
    """The chance-node half of the demo diagram: `structure`, `test_result`, `oil_price`.

    `structure` is the usual dry/wet/soaking state of nature. `test_result` is conditioned on
    `structure` is the dry/wet/soaking state of nature. `test_result` depends on `structure` and
    the `test` decision: one of NS/OS/CS if tested, else `no_info` with certainty. `oil_price` is
    an independent exogenous risk.
    """
    # P(test_result | structure, test), axes [test_result, structure, test].
    test_result_values = np.zeros((len(TEST_RESULT_STATES), len(STRUCTURE_STATES), len(TEST_STATES)))
    for s, structure in enumerate(STRUCTURE_STATES):
        for t, test in enumerate(TEST_STATES):
            probs = (
                _TEST_RESULT_GIVEN_STRUCTURE_WHEN_TESTED[structure]
                if test == "yes"
                else [0.0, 0.0, 0.0, 1.0]  # no_info certain when the test isn't run
            )
            for r in range(len(TEST_RESULT_STATES)):
                test_result_values[r, s, t] = probs[r]

    variables = [
        DiscreteVariable(
            name="structure", labels=STRUCTURE_STATES, description="True geological state of the drilling site."
        ),
        DiscreteVariable(
            name="test_result",
            labels=TEST_RESULT_STATES,
            description="Seismic test reading: no-structure/open-structure/closed-structure, or no "
            "information if the test wasn't run.",
        ),
        DiscreteVariable(
            name="oil_price",
            labels=OIL_PRICE_STATES,
            description="Price-per-barrel regime at the time of drilling, relative to the $100 baseline.",
        ),
    ]
    arcs = [("structure", "test_result")]
    potentials = {
        "structure": Potential(variables=["structure"], shape=[3], values=STRUCTURE_PRIOR),
        "test_result": Potential(
            variables=["test_result", "structure", "test"],
            shape=[len(TEST_RESULT_STATES), len(STRUCTURE_STATES), len(TEST_STATES)],
            values=test_result_values.tolist(),
        ),
        "oil_price": Potential(variables=["oil_price"], shape=[3], values=OIL_PRICE_PRIOR),
    }

    return DiscreteNetwork(name="oil_wildcatter_demo", variables=variables, arcs=arcs, potentials=potentials)


def build_oil_wildcatter_annotation() -> DecisionAnnotation:
    """The decision/objective overlay for `build_oil_wildcatter_network`.

    `test` (run the $10,000 seismic test or not) is taken first with no prior observation; `drill`
    observes `test_result` and is taken second -- `decision_order=["test", "drill"]` with
    `no_forgetting=True`, so `drill` also effectively observes `test`. There are three objectives,
    all parented on `[test, drill, structure, oil_price]`:

    - `investment` (minimize): the $10,000 test cost (if run) plus the $70,000 drilling cost (if
      drilled).
    - `oil_produced` (maximize, barrels): 0 if not drilled, else the structure's barrel count
      (dry/wet/soaking -> 0/1200/2700, backed out of the case's net payoffs at $100/barrel).
    - `profit` (maximize): `oil_produced * price_per_barrel[oil_price] - investment` -- the real
      trade-off objective, balancing the test's information value against drilling cost and price
      risk.

    `/graphical/whatif` raises 501 for this diagram unless both decisions are pinned or a solution
    is given.
    """
    parents = ["test", "drill", "structure", "oil_price"]
    objective_shape = [len(TEST_STATES), len(DRILL_STATES), len(STRUCTURE_STATES), len(OIL_PRICE_STATES)]

    investment_values = np.zeros(objective_shape)
    oil_produced_values = np.zeros(objective_shape)
    profit_values = np.zeros(objective_shape)
    for t, test in enumerate(TEST_STATES):
        for d, drill in enumerate(DRILL_STATES):
            investment = (_TEST_COST if test == "yes" else 0.0) + (_DRILL_COST if drill == "yes" else 0.0)
            for s, structure in enumerate(STRUCTURE_STATES):
                oil_produced = 0.0 if drill == "no" else _BARRELS_BY_STRUCTURE[structure]
                for p, oil_price in enumerate(OIL_PRICE_STATES):
                    profit = oil_produced * _PRICE_PER_BARREL_BY_OIL_PRICE[oil_price] - investment
                    investment_values[t, d, s, p] = investment
                    oil_produced_values[t, d, s, p] = oil_produced
                    profit_values[t, d, s, p] = profit

    return DecisionAnnotation(
        decisions=[
            DecisionNode(symbol="test", name="Run seismic test", states=TEST_STATES),
            DecisionNode(symbol="drill", name="Drill", states=DRILL_STATES, information_set=["test_result"]),
        ],
        objectives=[
            ObjectiveNode(
                symbol="investment",
                name="Investment cost",
                parents=parents,
                utility=Potential(variables=parents, shape=objective_shape, values=investment_values.tolist()),
                maximize=False,
            ),
            ObjectiveNode(
                symbol="oil_produced",
                name="Oil produced",
                parents=parents,
                utility=Potential(variables=parents, shape=objective_shape, values=oil_produced_values.tolist()),
                maximize=True,
            ),
            ObjectiveNode(
                symbol="profit",
                name="Profit",
                parents=parents,
                utility=Potential(variables=parents, shape=objective_shape, values=profit_values.tolist()),
                maximize=True,
            ),
        ],
        decision_order=["test", "drill"],
        no_forgetting=True,
    )


def build_oil_wildcatter_demo() -> tuple[DiscreteNetwork, DecisionAnnotation]:
    """`(network, annotation)` for the Oil Wildcatter CUMULUS problem. See `build_oil_wildcatter_annotation`."""
    network = build_oil_wildcatter_network()
    annotation = build_oil_wildcatter_annotation()
    annotation.validate_against(network)
    return network, annotation
