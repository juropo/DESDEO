"""A small, hand-built irrigation influence diagram used to exercise the compiler end to end.

Backend counterpart to `webui/src/lib/components/custom/network/sample-network.ts`'s
`sampleIrrigationNetwork` fixture -- built for Phase F of `.claude/desdeo-graphical-models-plan.md`
("Real backend integration for the CUMULUS demo"), the first time the webui's dev-only network
demo dialog is pointed at an actually-compiled diagram instead of a static TypeScript fixture.

**Not a literal structural match to `sample-network.ts`.** The frontend fixture has an edge
`irrigate -> crop_yield` (a chance node's CPT conditioned on a decision) and a `water_usage` node
of kind `"deterministic"` with `irrigate` as its only parent. Neither is expressible by
`DiscreteNetwork`/`DecisionAnnotation`: a chance node's CPT may only be conditioned on other chance
nodes, never a decision (`rooted_junction_tree.py`'s own module docstring; the same limitation
Phase 6's pig farm fixture -- `tests/test_graphical_compile.py`'s `_pig_farm_network` -- hit and
documented), and there is no `"deterministic"` node kind in this package's schema at all (only
chance/decision/objective -- see `StructureNode.kind` in `desdeo/api/routers/graphical.py`).

Reformulated the same way the pig farm fixture was, faithfully rather than by simplifying away the
decision dependence:

- `crop_yield` becomes three independent chance nodes, one per irrigation choice
  (`crop_yield_none`/`crop_yield_light`/`crop_yield_heavy`, "what the yield would be under this
  choice"), each conditioned on `soil_moisture` only, all three always drawn regardless of which
  choice `irrigate` actually makes. `profit`'s own utility table (which legitimately *can* depend
  on a decision, since it isn't a chance-node CPT) then selects the branch matching `irrigate`'s
  realized state while ignoring the other two -- since the three counterfactual nodes are
  conditionally independent given `soil_moisture` by construction, this reproduces the same joint
  distribution over `(soil_moisture, irrigate, realized yield)` a direct decision-conditioned CPT
  would have.
- `water_usage` is dropped as a separate node entirely and folded into `profit`'s own utility table
  as a direct function of `irrigate` -- it was already purely a deterministic function of the
  decision with no chance content of its own, so there is nothing a separate node would add that a
  decision-indexed utility table doesn't already capture, and (unlike the counterfactual
  reformulation above) this isn't even a compiler limitation: `"deterministic"` was never a
  representable node kind here to begin with.
- The frontend fixture's second objective, `water_cost`, is **dropped entirely, not just
  reformulated** -- a genuine backend capability gap, not a modelling convenience. A diagram with
  more than one `ObjectiveNode` *and* at least one decision is a confirmed, documented open problem
  for the live-inference side of this package (`_compute_utilities`'s own docstring in
  `desdeo/api/routers/graphical.py`: `ShaferShenoyLIMIDInference` exposes no joint-posterior
  primitive to recover one objective's own share once more than one utility node and a decision
  coexist), and critically, `POST /graphical/whatif` calls `_compute_utilities` *unconditionally*
  on every request, regardless of whether the caller cares about `utilities` -- so a two-objective
  version of this diagram would make `/whatif` respond `501` to *every* evidence query, not just
  degrade some secondary field. `profit`'s own utility table already bakes in water/irrigation
  cost (see below), so the traded-off-explicitly framing sample-network.ts's `water_cost` node
  described is preserved in spirit -- just not as a second, independently explorable node here.

This changes the diagram's node *count* and *objective count* relative to the frontend fixture, but
preserves the same story: weather drives soil moisture, which (together with the irrigation choice)
drives crop yield; profit is revenue from the realized yield minus water/irrigation cost. See the
CUMULUS demo dialog's own description text for how this is presented to a user.
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

    `crop_yield_none`/`crop_yield_light`/`crop_yield_heavy` are conditionally independent given
    `soil_moisture` -- irrigation narrows the gap a low natural soil moisture leaves, with heavy
    irrigation giving marginally lower "good"-yield odds than light irrigation at already-high
    soil moisture (a mild waterlogging effect), illustrative rather than agronomically precise.
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

    `irrigate` picks one of `IRRIGATE_STATES` with no prior observation. `profit` (the diagram's
    *only* objective -- see the module docstring for why a second one would break `/whatif`)
    depends on `irrigate` and all three counterfactual yield nodes -- only the one matching
    `irrigate`'s own realized state actually determines the value, the other two axes are constant
    slices (mirrors `_pig_farm_annotation`'s `profit` utility table in
    `tests/test_graphical_compile.py`) -- and already nets out water/irrigation cost per choice.
    """
    yield_node_order = [_YIELD_NODES[choice] for choice in IRRIGATE_STATES]
    profit_shape = [len(IRRIGATE_STATES), *([len(YIELD_STATES)] * len(yield_node_order))]
    profit_values = np.zeros(profit_shape)
    for i, irrigate_choice in enumerate(IRRIGATE_STATES):
        # Index of the yield axis that matters for this irrigation choice -- `i` itself, since
        # `yield_node_order` is built in the same `IRRIGATE_STATES` order.
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
