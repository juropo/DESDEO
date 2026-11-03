"""A cross-impact healthcare 2TBN as a stationary influence diagram.

The network, `datasets/ci_healthcare_2tbn.bif`, is a 2-time-slice Bayesian network fitted to
expert estimates of how a patient's illnesses and care evolve. Its states are labelled by index;
`NODE_NAMES` and `STATE_NAMES` give their meanings. The decisions are acute care `LYHt` and
whether a geriatric team works at the emergency department, `GER`.
"""

from pathlib import Path

import numpy as np

from desdeo.problem.graphical import (
    DiscreteNetwork,
    ObjectiveNode,
    Potential,
    StationaryDiagram,
    StationaryIndexMap,
    compile_stationary_diagram,
)
from desdeo.problem.schema import Problem

NODE_NAMES = {
    "LYS": "Short-term illness (Lyhytaikainen sairaus)",
    "PIS": "Long-term illness (Pitkäaikainen sairaus)",
    "HHA": "Care-seeking propensity (Hoitoonhakeutumisalttius)",
    "LYH": "Acute care (Lyhytaikainen hoito)",
    "PIH": "Long-term care (Pitkäaikainen hoito)",
    "GER": "Geriatric team at the emergency department (Geriatriapalikka)",
    "OMA": "Own doctor (Omalääkäri)",
    "AST": "Customer satisfaction (Asiakastyytyväisyys)",
    "KUO": "Died (Kuollut)",
}
"""Node name without its slice suffix -> readable name."""

STATE_NAMES = {
    "LYS": [
        "Healthy (Terve)",
        "Mild, likely heals without care (Lievä)",
        "Severe, needs care or risks long-term problems (Vakava)",
        "Life-threatening (Hengenvaarallinen)",
    ],
    "PIS": [
        "Healthy (Terve)",
        "Mild, no active care needed (Lievä)",
        "Severe, needs continuous care (Vakava)",
        "Life-threatening (Hengenvaarallinen)",
    ],
    "HHA": [
        "Overuser, seeks care needlessly (Ramppaaja)",
        "Normal, seeks care mainly when serious (Normaali)",
        "Reluctant, avoids care unless forced (Änkyrä)",
    ],
    "LYH": [
        "None (Ei mitään)",
        "Health centre (Terveyskeskus)",
        "Emergency department (Päivystys)",
        "Hospital (Sairaala)",
        "Home hospital (Kotisairaala)",
        "Short-term rehabilitation (Lyhytaikainen kuntoutus)",
        "Assessment unit (Arviointiyksikkö)",
    ],
    "PIH": [
        "None (Ei mitään)",
        "Family care (Omaishoito)",
        "Home care (Kotihoito)",
        "24-hour assisted living (Ympärivuorokautinen palveluasuminen)",
        "Institutional care (Laitospalvelu)",
    ],
    "GER": ["Yes (Kyllä)", "No (Ei)"],
    "OMA": ["Yes (Kyllä)", "No (Ei)"],
    "AST": ["Good (Hyvä)", "Average (Keskinkertainen)", "Poor (Huono)"],
    "KUO": ["Yes (Kyllä)", "No (Ei)"],
}
"""Node name without its slice suffix -> readable name of each state, in state-index order."""

PIH_COSTS = [0.0, 1.0, 2.0, 4.0, 6.0]
"""Placeholder per-step cost of each long-term care state, in `STATE_NAMES["PIH"]` order."""

LYH_COSTS = [0.0, 1.0, 2.0, 5.0, 3.0, 3.0, 2.0]
"""Placeholder per-step cost of each acute care state, in `STATE_NAMES["LYH"]` order."""

_BIF_PATH = Path(__file__).parent.parent.parent.parent / "datasets/ci_healthcare_2tbn.bif"


def _objective(symbol: str, name: str, parents: list[str], utility: np.ndarray, unit: str) -> ObjectiveNode:
    return ObjectiveNode(
        symbol=symbol,
        name=name,
        parents=parents,
        utility=Potential(variables=parents, shape=list(utility.shape), values=utility.tolist()),
        unit=unit,
    )


def _load_normalised_network() -> DiscreteNetwork:
    """The BIF's network with each CPT rescaled to sum to 1, which the file's rounding breaks."""
    from desdeo.problem.graphical.adapters.pyagrum import load_network  # noqa: PLC0415

    network = load_network(_BIF_PATH)
    potentials = {}
    for name, potential in network.potentials.items():
        values = np.array(potential.values, dtype=float)
        potentials[name] = potential.model_copy(update={"values": (values / values.sum(axis=0)).tolist()})
    return network.model_copy(update={"potentials": potentials})


def ci_healthcare_stationary_diagram(
    *, lyh_information_set: list[str] | None = None, discount: float = 0.95
) -> StationaryDiagram:
    """The cross-impact healthcare 2TBN with `LYHt` and `GER` as stationary decisions.

    `GER` observes nothing. All objectives are minimised per-step averages under the discounted
    occupation measure of living patients:

    - `cost`: placeholder care cost `PIH_COSTS[PIHt] + LYH_COSTS[LYHt]`;
    - `dissatisfaction`: `ASTt`'s state index, 0 good to 2 poor;
    - `mortality`: the probability `KUOt` is "Yes", the per-step death hazard;
    - `illness`: `PISt`'s state index, 0 healthy to 3 life-threatening.

    Args:
        lyh_information_set: the nodes `LYHt` observes; `None` uses its network parents.
        discount: the discount factor gamma, in (0, 1).

    Returns:
        StationaryDiagram: the diagram.
    """
    network = _load_normalised_network()

    objectives = [
        _objective(
            "cost",
            "Care cost",
            ["PIHt", "LYHt"],
            np.add.outer(PIH_COSTS, LYH_COSTS),
            "cost units per step",
        ),
        _objective("dissatisfaction", "Customer dissatisfaction", ["ASTt"], np.arange(3.0), "dissatisfaction level"),
        _objective("mortality", "Mortality", ["KUOt"], np.array([1.0, 0.0]), "probability per step"),
        _objective("illness", "Long-term illness severity", ["PISt"], np.arange(4.0), "severity level"),
    ]
    diagram = StationaryDiagram.with_default_information_sets(
        network, {"LYHt": lyh_information_set, "GER": []}, objectives, discount=discount
    )
    names = {"LYHt": NODE_NAMES["LYH"], "GER": NODE_NAMES["GER"]}
    decisions = [decision.model_copy(update={"name": names[decision.symbol]}) for decision in diagram.decisions]
    return diagram.model_copy(update={"decisions": decisions})


def ci_healthcare_problem(
    *, lyh_information_set: list[str] | None = None, discount: float = 0.95
) -> tuple[Problem, StationaryIndexMap]:
    """`ci_healthcare_stationary_diagram` compiled with `compile_stationary_diagram`.

    Args:
        lyh_information_set: as in `ci_healthcare_stationary_diagram`.
        discount: as in `ci_healthcare_stationary_diagram`.

    Returns:
        tuple[Problem, StationaryIndexMap]: the MILP and its bookkeeping.
    """
    return compile_stationary_diagram(
        ci_healthcare_stationary_diagram(lyh_information_set=lyh_information_set, discount=discount)
    )
