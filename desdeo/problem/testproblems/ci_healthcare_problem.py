"""A cross-impact healthcare 2TBN as a stationary influence diagram.

The network, `datasets/ci_healthcare_2tbn.bif`, is a 2-time-slice Bayesian network fitted to
expert estimates of how a patient's illnesses and care evolve. Its states are labelled by index;
`NODE_NAMES` and `STATE_NAMES` give their meanings. The decisions are acute care `LYHt` and
whether a geriatric team works at the emergency department, `GER`.

The LP path, `ci_healthcare_lp_diagram`, uses the updated `datasets/ci_healthcare_2tbn_pure.bif`
by default, with `GER` decided every step as `GERt` and the own doctor `OMA` decided once per
patient.
"""

from pathlib import Path
from typing import Literal

import numpy as np

from desdeo.problem.graphical import (
    DiscreteNetwork,
    ObjectiveNode,
    Potential,
    StationaryDiagram,
    StationaryIndexMap,
    StationaryLPIndexMap,
    compile_stationary_diagram,
    compile_stationary_lp,
    full_observation_information_set,
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

_DATASETS = Path(__file__).parent.parent.parent.parent / "datasets"
_BIF_PATH = _DATASETS / "ci_healthcare_2tbn.bif"
_PURE_BIF_PATH = _DATASETS / "ci_healthcare_2tbn_pure.bif"


def _objective(symbol: str, name: str, parents: list[str], utility: np.ndarray, unit: str) -> ObjectiveNode:
    return ObjectiveNode(
        symbol=symbol,
        name=name,
        parents=parents,
        utility=Potential(variables=parents, shape=list(utility.shape), values=utility.tolist()),
        unit=unit,
    )


def _objectives() -> list[ObjectiveNode]:
    """The four minimised objectives, as described in `ci_healthcare_stationary_diagram`."""
    return [
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


def _load_normalised_network(bif_path: Path, cpt_floor: float) -> DiscreteNetwork:
    """The BIF's network with CPT entries below `cpt_floor` set to 0 and each CPT rescaled to sum to 1."""
    from desdeo.problem.graphical.adapters.pyagrum import load_network  # noqa: PLC0415

    network = load_network(bif_path)
    max_cardinality = max(potential.shape[0] for potential in network.potentials.values())
    if not 0 <= cpt_floor < 1 / max_cardinality:
        msg = f"cpt_floor must be in [0, 1/{max_cardinality}), got {cpt_floor}."
        raise ValueError(msg)
    potentials = {}
    for name, potential in network.potentials.items():
        values = np.array(potential.values, dtype=float)
        values[values < cpt_floor] = 0.0
        potentials[name] = potential.model_copy(update={"values": (values / values.sum(axis=0)).tolist()})
    return network.model_copy(update={"potentials": potentials})


def ci_healthcare_stationary_diagram(
    *, lyh_information_set: list[str] | None = None, discount: float = 0.95, cpt_floor: float = 1e-8
) -> StationaryDiagram:
    """The cross-impact healthcare 2TBN with `LYHt` and `GER` as stationary decisions.

    `GER` observes nothing. All objectives are minimised per-step averages under the discounted
    occupation measure of living patients:

    - `cost`: placeholder care cost `PIH_COSTS[PIHt] + LYH_COSTS[LYHt]`;
    - `dissatisfaction`: `ASTt`'s state index, 0 good to 2 poor;
    - `mortality`: the probability `KUOt` is "Yes", the per-step death hazard;
    - `illness`: `PISt`'s state index, 0 healthy to 3 life-threatening.

    CPT entries below `cpt_floor` are set to 0 before each CPT is renormalised, because tiny
    entries make solver presolve wrongly prune the compiled MILP. The default floor changes the
    status-quo occupation by at most about 1e-8 and the status-quo objectives by less than 1e-6
    relative.

    Args:
        lyh_information_set: the nodes `LYHt` observes; `None` uses its network parents.
        discount: the discount factor gamma, in (0, 1).
        cpt_floor: entries below it are set to 0, in [0, 1 / largest node cardinality).

    Returns:
        StationaryDiagram: the diagram.
    """
    network = _load_normalised_network(_BIF_PATH, cpt_floor)
    diagram = StationaryDiagram.with_default_information_sets(
        network, {"LYHt": lyh_information_set, "GER": []}, _objectives(), discount=discount
    )
    names = {"LYHt": NODE_NAMES["LYH"], "GER": NODE_NAMES["GER"]}
    decisions = [decision.model_copy(update={"name": names[decision.symbol]}) for decision in diagram.decisions]
    return diagram.model_copy(update={"decisions": decisions})


def ci_healthcare_problem(
    *, lyh_information_set: list[str] | None = None, discount: float = 0.95, cpt_floor: float = 1e-8
) -> tuple[Problem, StationaryIndexMap]:
    """`ci_healthcare_stationary_diagram` compiled with `compile_stationary_diagram`.

    Args:
        lyh_information_set: as in `ci_healthcare_stationary_diagram`.
        discount: as in `ci_healthcare_stationary_diagram`.
        cpt_floor: as in `ci_healthcare_stationary_diagram`.

    Returns:
        tuple[Problem, StationaryIndexMap]: the MILP and its bookkeeping.
    """
    return compile_stationary_diagram(
        ci_healthcare_stationary_diagram(
            lyh_information_set=lyh_information_set, discount=discount, cpt_floor=cpt_floor
        )
    )


def _renamed(network: DiscreteNetwork, old: str, new: str) -> DiscreteNetwork:
    """`network` with node `old` renamed to `new` in its variables, arcs and CPTs."""

    def rename(name: str) -> str:
        return new if name == old else name

    return DiscreteNetwork(
        name=network.name,
        variables=[variable.model_copy(update={"name": rename(variable.name)}) for variable in network.variables],
        arcs=[(rename(parent), rename(child)) for parent, child in network.arcs],
        potentials={
            rename(name): potential.model_copy(update={"variables": [rename(v) for v in potential.variables]})
            for name, potential in network.potentials.items()
        },
    )


def ci_healthcare_lp_diagram(
    *, bif_path: Path = _PURE_BIF_PATH, discount: float = 0.95, cpt_floor: float = 1e-8, renewal: bool = True
) -> StationaryDiagram:
    """The cross-impact healthcare 2TBN for `compile_stationary_lp`.

    `GER` is renamed to the slice-t decision `GERt`, made every step. `OMA` is an atemporal
    decision that observes nothing, drawn once per patient. `GERt` and `LYHt` observe their
    full-observation sets. With `renewal`, a patient who dies (`KUOt` "Yes") is replaced by a new
    patient drawn from the slice-0 distribution, so the objectives are per-step averages over a
    constant population. The objectives and the CPT floor are as in
    `ci_healthcare_stationary_diagram`.

    Args:
        bif_path: the 2TBN, with no arcs from `GER` into slice-0 nodes, since `GERt` is slice-t;
            the default is `datasets/ci_healthcare_2tbn_pure.bif`. The pure and reduced files
            qualify; the original's `GER` -> `AST0` arc is rejected by `classify_2tbn`.
        discount: the discount factor gamma, in (0, 1).
        cpt_floor: entries below it are set to 0, in [0, 1 / largest node cardinality).
        renewal: whether death renews the patient.

    Returns:
        StationaryDiagram: the diagram.
    """
    network = _renamed(_load_normalised_network(bif_path, cpt_floor), "GER", "GERt")
    objectives = _objectives()
    renewal_states = {"KUOt": [0]} if renewal else None
    provisional = StationaryDiagram.with_default_information_sets(
        network, {"OMA": [], "GERt": None, "LYHt": None}, objectives, discount=discount, renewal=renewal_states
    )
    information_sets = {
        "OMA": [],
        "GERt": full_observation_information_set(provisional, "GERt"),
        "LYHt": full_observation_information_set(provisional, "LYHt"),
    }
    diagram = StationaryDiagram.with_default_information_sets(
        network, information_sets, objectives, discount=discount, renewal=renewal_states
    )
    names = {"OMA": NODE_NAMES["OMA"], "GERt": NODE_NAMES["GER"], "LYHt": NODE_NAMES["LYH"]}
    decisions = [decision.model_copy(update={"name": names[decision.symbol]}) for decision in diagram.decisions]
    return diagram.model_copy(update={"decisions": decisions})


def ci_healthcare_lp_problem(
    *,
    bif_path: Path = _PURE_BIF_PATH,
    discount: float = 0.95,
    cpt_floor: float = 1e-8,
    renewal: bool = True,
    atemporal: Literal["randomised", "deterministic"] = "randomised",
) -> tuple[Problem, StationaryLPIndexMap]:
    """`ci_healthcare_lp_diagram` compiled with `compile_stationary_lp`.

    Args:
        bif_path: as in `ci_healthcare_lp_diagram`.
        discount: as in `ci_healthcare_lp_diagram`.
        cpt_floor: as in `ci_healthcare_lp_diagram`.
        renewal: as in `ci_healthcare_lp_diagram`.
        atemporal: as in `compile_stationary_lp`.

    Returns:
        tuple[Problem, StationaryLPIndexMap]: the LP and its bookkeeping.
    """
    return compile_stationary_lp(
        ci_healthcare_lp_diagram(bif_path=bif_path, discount=discount, cpt_floor=cpt_floor, renewal=renewal),
        atemporal=atemporal,
    )
