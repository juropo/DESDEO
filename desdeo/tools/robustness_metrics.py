"""Robustness and antifragility metrics for comparing designs evaluated across scenarios.

Outcomes are given per design, per scenario leaf and per objective. Provides max regret and the domain
criterion (Shavazipour et al., Eq. 2 and 3) and an upside-downside antifragility index with
fragile/robust/antifragile classes (Taleb, *Antifragile*, 2012).

A design that is infeasible (or failed to solve) in a leaf should be passed with that leaf, or the affected
objectives, left out. How such missing outcomes are treated is set by the ``missing`` argument.
"""

import warnings
from collections.abc import Hashable, Mapping
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

DesignId = Hashable
Outcomes = Mapping[DesignId, Mapping[str, Mapping[str, float]]]
EventClass = Literal["gain", "untouched", "bounded_loss", "fragile"]
TalebClass = Literal["fragile", "robust", "antifragile"]
MissingPolicy = Literal["worst", "skip"]

# Improvements within this tolerance of zero count as neither gains nor losses.
_TOL = 1e-9


class MaxRegret(BaseModel):
    """Per-design maximum regret summary."""

    model_config = ConfigDict(frozen=True)

    regret: dict[str, float | None]
    """Per-objective maximum regret across leaves (None if no leaf has a value)."""
    mean_regret: float | None
    """Mean over objectives with a non-None regret; None if none."""


class DomainCriterion(BaseModel):
    """Per-design domain-criterion counts."""

    model_config = ConfigDict(frozen=True)

    counts: dict[str, int]
    """Number of leaves meeting the threshold, per objective that has one."""
    mean_count: float | None
    """Mean of counts; None if no thresholds."""


class Deviation(BaseModel):
    """Deviation of one design and objective in one non-baseline leaf."""

    model_config = ConfigDict(frozen=True)

    leaf: str
    """The non-baseline leaf."""
    value: float
    """The design's outcome in this leaf, original direction."""
    improvement: float
    """Change vs baseline normalised by the leaf's spread, positive = better."""
    improvement_raw: float
    """Unnormalised change vs baseline, positive = better."""
    event_class: EventClass
    """Classification of this deviation."""


class ObjectiveAntifragility(BaseModel):
    """Antifragility summary for one objective of one design."""

    model_config = ConfigDict(frozen=True)

    upside: float | None
    """U: largest (or mean) normalised gain across non-baseline leaves."""
    downside: float | None
    """D: largest normalised loss across non-baseline leaves."""
    antifragility: float | None
    """AF = (U - D) / (U + D + eps), in [-1, 1]."""
    n_gain: int
    """Number of leaves with a gain."""
    n_loss: int
    """Number of leaves with a loss."""
    upside_raw: float | None
    """Unnormalised gain in the best-gain leaf (0 if no gain)."""
    downside_raw: float | None
    """Unnormalised loss in the leaf with the largest normalised loss (0 if no loss)."""
    max_loss_raw: float | None
    """Largest unnormalised loss over the non-baseline leaves (0 if no loss)."""
    best_gain_leaf: str | None
    """Leaf with the largest normalised gain, if any."""
    worst_loss_leaf: str | None
    """Leaf with the largest normalised loss, if any."""
    worst_outcome: float | None
    """Worst value over all leaves including the baseline, original direction."""
    taleb_class: TalebClass | None
    """Taleb-style class; only set for objectives with a floor."""
    deviations: list[Deviation]
    """Deviations in all non-baseline leaves where the design has a value (after filling missing values)."""


class Antifragility(BaseModel):
    """Antifragility results for a single design."""

    model_config = ConfigDict(frozen=True)

    objectives: dict[str, ObjectiveAntifragility]
    """Per-objective antifragility summary."""


def _to_min_array(
    outcomes: Outcomes,
    maximize: Mapping[str, bool],
) -> tuple[list[DesignId], list[str], list[str], np.ndarray]:
    """Convert outcomes to a 3-D min-space array.

    Args:
        outcomes: design -> leaf -> objective -> value.
        maximize: objective -> True if the objective is maximised.

    Returns:
        Tuple of (design_ids, leaves, objectives, arr) where ``arr`` has shape
        (n_designs, n_leaves, n_objectives) in min-space (maximised objectives
        are negated). Missing entries are NaN.

    Raises:
        ValueError: If an outcome mentions an objective not in ``maximize``.
    """
    objectives = list(maximize)
    obj_index = {o: i for i, o in enumerate(objectives)}
    design_ids = list(outcomes)
    leaf_index: dict[str, int] = {}
    for leaves_map in outcomes.values():
        for leaf, values in leaves_map.items():
            leaf_index.setdefault(leaf, len(leaf_index))
            for obj in values:
                if obj not in obj_index:
                    raise ValueError(f"Objective '{obj}' not in maximize mapping.")

    arr = np.full((len(design_ids), len(leaf_index), len(objectives)), np.nan)
    for di, design_id in enumerate(design_ids):
        for leaf, values in outcomes[design_id].items():
            for obj, val in values.items():
                arr[di, leaf_index[leaf], obj_index[obj]] = -float(val) if maximize[obj] else float(val)

    return design_ids, list(leaf_index), objectives, arr


def _fill_missing(arr: np.ndarray, missing: MissingPolicy) -> np.ndarray:
    """Apply the missing-outcome policy to a min-space array.

    With ``"worst"``, a missing entry becomes the worst (largest min-space) value of its leaf and objective
    over the designs; entries where no design has a value stay NaN. With ``"skip"``, the array is unchanged.

    Raises:
        ValueError: If ``missing`` is not "worst" or "skip".
    """
    if missing not in ("worst", "skip"):
        raise ValueError(f"missing must be 'worst' or 'skip', got {missing!r}.")
    if missing == "skip":
        return arr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        worst = np.nanmax(arr, axis=0)
    return np.where(np.isnan(arr), worst, arr)


def _leaf_best_and_spread(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-leaf ideal (nanmin) and spread (nanmax - nanmin) over designs.

    Args:
        arr: shape (n_designs, n_leaves, n_objectives) in min-space.

    Returns:
        (best, spread), each of shape (n_leaves, n_objectives); NaN where no design has a value.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        best = np.nanmin(arr, axis=0)
        worst = np.nanmax(arr, axis=0)
    return best, worst - best


def max_regret(
    outcomes: Outcomes,
    maximize: Mapping[str, bool],
    *,
    normalize: bool = True,
    missing: MissingPolicy = "worst",
) -> dict[DesignId, MaxRegret]:
    """Compute the max regret of each design (Shavazipour et al., Eq. 2).

    In each leaf, the regret of a design is its gap to the best design in that leaf, divided by the spread
    (worst - best) over the designs in that leaf; leaves where all designs tie give regret 0. The max regret
    of an objective is the largest regret over the leaves where the design has a value. Regret is relative
    to the compared set of designs: adding or removing designs changes the per-leaf best and spread, and
    therefore the regret.

    Pass infeasible leaves as missing outcomes. With ``missing="worst"``, a design missing a leaf and objective
    that other designs have gets the worst value among them, i.e. regret 1 (the full spread when not
    normalised) unless all designs with a value tie. With ``missing="skip"``, the leaf is left out of that
    design's max. Leaves and objectives where no design has a value are skipped in both modes.

    Args:
        outcomes: design -> leaf -> objective -> value.
        maximize: objective -> True if the objective is maximised.
        normalize: if True, divide the gap by the per-leaf spread; otherwise report the raw gap.
        missing: how to treat missing outcomes: ``"worst"`` or ``"skip"``.

    Returns:
        Mapping from each design id to its ``MaxRegret``.

    Raises:
        ValueError: If an outcome mentions an objective not in ``maximize``, or ``missing`` is not "worst" or
            "skip".
    """
    design_ids, _leaves, objectives, arr = _to_min_array(outcomes, maximize)
    arr = _fill_missing(arr, missing)
    best, spread = _leaf_best_and_spread(arr)

    gap = arr - best
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            regret = np.where(spread > 0, gap / spread, 0.0)
        # Missing outcomes stay missing instead of counting as zero regret.
        regret[np.isnan(arr)] = np.nan
    else:
        regret = gap

    result: dict[DesignId, MaxRegret] = {}
    for di, design_id in enumerate(design_ids):
        per_obj: dict[str, float | None] = {}
        for oi, obj in enumerate(objectives):
            vals = regret[di, :, oi]
            per_obj[obj] = float(np.nanmax(vals)) if not np.isnan(vals).all() else None
        present = [v for v in per_obj.values() if v is not None]
        result[design_id] = MaxRegret(regret=per_obj, mean_regret=float(np.mean(present)) if present else None)

    return result


def domain_criterion(
    outcomes: Outcomes,
    maximize: Mapping[str, bool],
    thresholds: Mapping[str, float],
) -> dict[DesignId, DomainCriterion]:
    """Compute the domain criterion of each design (Shavazipour et al., Eq. 3).

    For each design and each objective in ``thresholds``, counts the leaves where the outcome meets the
    threshold (value <= threshold for minimised, value >= threshold for maximised objectives). Missing
    entries, such as infeasible leaves, count as not meeting the threshold.

    Args:
        outcomes: design -> leaf -> objective -> value.
        maximize: objective -> True if the objective is maximised.
        thresholds: objective -> threshold value, original direction.

    Returns:
        Mapping from each design id to its ``DomainCriterion``.

    Raises:
        ValueError: If a threshold key is not in ``maximize``, or an outcome mentions an objective not in
            ``maximize``.
    """
    for key in thresholds:
        if key not in maximize:
            raise ValueError(f"Threshold key '{key}' not in maximize mapping.")

    design_ids, _leaves, objectives, arr = _to_min_array(outcomes, maximize)

    result: dict[DesignId, DomainCriterion] = {}
    for di, design_id in enumerate(design_ids):
        counts: dict[str, int] = {}
        for obj, threshold in thresholds.items():
            vals = arr[di, :, objectives.index(obj)]
            min_threshold = -threshold if maximize[obj] else threshold
            # NaN comparisons are False, so missing entries are not counted.
            counts[obj] = int(np.sum(vals <= min_threshold))
        mean_count = float(np.mean(list(counts.values()))) if counts else None
        result[design_id] = DomainCriterion(counts=counts, mean_count=mean_count)

    return result


def _event_class(
    improvement: float, improvement_raw: float, value: float, min_floor: float | None, loss_floor: float | None
) -> EventClass:
    """Classify one deviation; ``value`` and ``min_floor`` are in min-space."""
    if (min_floor is not None and value > min_floor) or (loss_floor is not None and -improvement_raw > loss_floor):
        return "fragile"
    if improvement > _TOL:
        return "gain"
    if improvement < -_TOL:
        return "bounded_loss"
    return "untouched"


def _objective_antifragility(
    vals: np.ndarray,
    spread: np.ndarray,
    leaves: list[str],
    *,
    baseline_idx: int,
    maximize: bool,
    upside: str,
    absolute_floor: float | None,
    loss_floor: float | None,
    eps: float,
) -> ObjectiveAntifragility:
    """Antifragility summary of one design and objective from its min-space values and per-leaf spreads."""
    sign = -1.0 if maximize else 1.0
    min_floor = None if absolute_floor is None else sign * absolute_floor
    worst = None if np.isnan(vals).all() else float(np.nanmax(vals))
    base = vals[baseline_idx]

    deviations: list[Deviation] = []
    if not np.isnan(base):
        for li, value in enumerate(vals):
            if li == baseline_idx or np.isnan(value):
                continue
            imp_raw = float(base - value)
            imp = imp_raw / spread[li] if spread[li] > 0 else 0.0
            deviations.append(
                Deviation(
                    leaf=leaves[li],
                    value=sign * float(value),
                    improvement=imp,
                    improvement_raw=imp_raw,
                    event_class=_event_class(imp, imp_raw, float(value), min_floor, loss_floor),
                )
            )

    up = down = af = up_raw = down_raw = max_loss_raw = None
    gain_leaf = loss_leaf = None
    n_gain = n_loss = 0
    if deviations:
        imp = np.array([d.improvement for d in deviations])
        gains = np.maximum(imp, 0.0)
        down = float(np.maximum(-imp, 0.0).max())
        up = float(gains.mean() if upside == "mean" else gains.max())
        af = (up - down) / (up + down + eps)
        n_gain = int((imp > _TOL).sum())
        n_loss = int((imp < -_TOL).sum())
        loss_dev = deviations[int(np.argmin(imp))]
        gain_dev = deviations[int(np.argmax(imp))]
        down_raw = max(-loss_dev.improvement_raw, 0.0)
        max_loss_raw = max(0.0, *(-d.improvement_raw for d in deviations))
        up_raw = max(gain_dev.improvement_raw, 0.0)
        loss_leaf = loss_dev.leaf if loss_dev.improvement < 0 else None
        gain_leaf = gain_dev.leaf if gain_dev.improvement > 0 else None

    taleb_class: TalebClass | None = None
    if absolute_floor is not None or loss_floor is not None:
        fragile = (loss_floor is not None and max_loss_raw is not None and max_loss_raw > loss_floor) or (
            min_floor is not None and worst is not None and worst > min_floor
        )
        if fragile:
            taleb_class = "fragile"
        elif deviations:
            taleb_class = "antifragile" if n_gain >= 1 else "robust"

    return ObjectiveAntifragility(
        upside=up,
        downside=down,
        antifragility=af,
        n_gain=n_gain,
        n_loss=n_loss,
        upside_raw=up_raw,
        downside_raw=down_raw,
        max_loss_raw=max_loss_raw,
        best_gain_leaf=gain_leaf,
        worst_loss_leaf=loss_leaf,
        worst_outcome=None if worst is None else sign * worst,
        taleb_class=taleb_class,
        deviations=deviations,
    )


def antifragility(
    outcomes: Outcomes,
    maximize: Mapping[str, bool],
    baseline_leaf: str,
    *,
    upside: Literal["max", "mean"] = "max",
    absolute_floors: Mapping[str, float] | None = None,
    loss_floors: Mapping[str, float] | None = None,
    eps: float = 1e-9,
    missing: MissingPolicy = "worst",
) -> dict[DesignId, Antifragility]:
    """Compute the upside-downside antifragility index of each design.

    For each design and objective, every non-baseline leaf where the design has a value gives a deviation
    from the design's own outcome in ``baseline_leaf``, normalised by that leaf's spread (worst - best) over
    the compared set of designs (deviation 0 where the spread is 0). D is the largest loss, U the largest
    (or mean) gain, and AF = (U - D) / (U + D + eps) lies in [-1, 1]: -1 means only downside, 1 only upside.

    For objectives with a floor, designs are classified following Taleb (*Antifragile*, 2012):
    ``"fragile"`` if any deviation is fragile or the baseline outcome breaches the absolute floor, i.e. the
    worst outcome over all leaves breaches the absolute floor or the largest raw loss over the non-baseline
    leaves (``max_loss_raw``) exceeds the loss floor; otherwise ``"antifragile"`` if the design gains in at
    least one leaf, else ``"robust"``. Individual deviations are classified into
    gain/untouched/bounded_loss/fragile with the same floors.

    Pass infeasible leaves as missing outcomes. With ``missing="worst"``, a design missing a leaf and objective
    that other designs have takes the worst value among them there, so it counts towards the design's
    deviations and worst outcome (including in the baseline leaf). With ``missing="skip"``, such leaves give
    no deviation. Leaves and objectives where no design has a value are skipped in both modes.

    Args:
        outcomes: design -> leaf -> objective -> value.
        maximize: objective -> True if the objective is maximised.
        baseline_leaf: the leaf each design is compared against.
        upside: how to aggregate gains: ``"max"`` or ``"mean"`` over the non-baseline leaves.
        absolute_floors: objective -> worst acceptable outcome in the original direction (an upper bound for
            minimised, a lower bound for maximised objectives).
        loss_floors: objective -> largest acceptable loss relative to the baseline, in original units.
        eps: small constant that keeps AF defined when U = D = 0.
        missing: how to treat missing outcomes: ``"worst"`` or ``"skip"``.

    Returns:
        Mapping from each design id to its ``Antifragility``. Designs or objectives without a baseline value
        (after filling missing values) get no deviations and None for the index.

    Raises:
        ValueError: If ``baseline_leaf`` is not a leaf of any design, a floor key is not in ``maximize``,
            ``upside`` is not "max" or "mean", ``missing`` is not "worst" or "skip", or an outcome mentions an
            objective not in ``maximize``.
    """
    if upside not in ("max", "mean"):
        raise ValueError(f"upside must be 'max' or 'mean', got {upside!r}.")
    absolute_floors = absolute_floors or {}
    loss_floors = loss_floors or {}
    for key in [*absolute_floors, *loss_floors]:
        if key not in maximize:
            raise ValueError(f"Floor key '{key}' not in maximize mapping.")

    design_ids, leaves, objectives, arr = _to_min_array(outcomes, maximize)
    if baseline_leaf not in leaves:
        raise ValueError(f"Baseline leaf '{baseline_leaf}' not found in any design.")
    baseline_idx = leaves.index(baseline_leaf)
    arr = _fill_missing(arr, missing)
    _, spread = _leaf_best_and_spread(arr)

    return {
        design_id: Antifragility(
            objectives={
                obj: _objective_antifragility(
                    arr[di, :, oi],
                    spread[:, oi],
                    leaves,
                    baseline_idx=baseline_idx,
                    maximize=maximize[obj],
                    upside=upside,
                    absolute_floor=absolute_floors.get(obj),
                    loss_floor=loss_floors.get(obj),
                    eps=eps,
                )
                for oi, obj in enumerate(objectives)
            }
        )
        for di, design_id in enumerate(design_ids)
    }


def antifragile_objectives(results: Mapping[DesignId, Antifragility], tol: float = 1e-9) -> list[str]:
    """List the objectives for which some design has an upside above ``tol``.

    Args:
        results: output of ``antifragility``.
        tol: minimum upside for an objective to be listed.

    Returns:
        Objective names in order of first qualifying appearance.
    """
    out: list[str] = []
    for result in results.values():
        for obj, summary in result.objectives.items():
            if summary.upside is not None and summary.upside > tol and obj not in out:
                out.append(obj)
    return out
