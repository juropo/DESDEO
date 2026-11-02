"""Aggregations over outcome distributions whose probabilities depend on the decisions.

Scenario-based aggregations (``desdeo.tools.stochastic``, ``desdeo.tools.robust``) work on a
fixed set of scenarios with constant probabilities and decision-dependent values. Here it is
the other way around: an element can take one of finitely many constant values ``u``, and the
probability ``p(u)`` of each value is an ``ExtraFunction`` of the decision variables, as in
problems compiled from graphical models. Since ``p(u)`` is linear in the decision variables,
each aggregate has a linear or mixed-integer linear form.

The aggregation functions take ``distributions`` keyed by symbol. A key forms the new element's
symbol (``{prefix}{key}``), and the distribution's own ``symbol`` decides the new element's type.
The two are usually equal, but need not be, e.g. for several CVaR levels of one objective.

No weighted-scenario variant is provided: with constant values and user-given weights,
``sum_u w_u * u`` is a constant that does not depend on the decisions.
"""

import itertools
import math
from collections import Counter
from typing import TYPE_CHECKING, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from desdeo.problem.schema import (
    Constraint,
    ConstraintTypeEnum,
    TensorVariable,
    Variable,
    VariableTypeEnum,
)
from desdeo.tools.scenarios import append_aggregated_elem

if TYPE_CHECKING:
    from desdeo.problem.schema import Problem


class OutcomeDistribution(BaseModel):
    """The distribution of an element's value over finitely many constant outcomes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(description="Symbol of the aggregated element.")
    name: str = Field(description="Name of the aggregated element.")
    maximize: bool = Field(description="Whether the aggregated element is maximized.")
    values: tuple[float, ...] = Field(description="The possible values of the element, ascending and distinct.")
    probabilities: tuple[str, ...] = Field(
        description=(
            "Symbols of the extra functions giving the probability of each value, in the same order "
            "as `values`. They should be linear in the decision variables."
        )
    )

    @model_validator(mode="after")
    def _check_values(self) -> "OutcomeDistribution":
        if not self.values:
            msg = "values must not be empty."
            raise ValueError(msg)
        if len(self.values) != len(self.probabilities):
            msg = (
                f"values and probabilities must have equal lengths, got {len(self.values)} "
                f"and {len(self.probabilities)}."
            )
            raise ValueError(msg)
        if not all(math.isfinite(v) for v in self.values):
            msg = f"values must be finite, got {self.values}."
            raise ValueError(msg)
        if any(b <= a for a, b in itertools.pairwise(self.values)):
            msg = f"values must be ascending and distinct, got {self.values}."
            raise ValueError(msg)
        return self


class _Resolution(NamedTuple):
    """Per-symbol metadata shared by the aggregation functions."""

    dist: OutcomeDistribution
    found_type: str
    unit: str | None
    is_linear: bool
    is_twice_diff: bool


def _resolve(problem: "Problem", distributions: dict[str, OutcomeDistribution], sym: str) -> _Resolution:
    """Look up the distribution for ``sym`` and the element type of its base element.

    The aggregate is an objective if the base element is an objective, a scalarization function
    if it is one, and an extra function otherwise. An objective aggregate takes the base
    objective's unit.

    Raises:
        ValueError: if ``sym`` has no distribution, a probability symbol is not an extra
            function of ``problem``, or the distribution's ``maximize`` differs from the base
            objective's.
    """
    if sym not in distributions:
        msg = f"Symbol '{sym}' not found in distributions."
        raise ValueError(msg)
    dist = distributions[sym]

    extra_funcs = {e.symbol: e for e in problem.extra_funcs or []}
    missing = [p for p in dist.probabilities if p not in extra_funcs]
    if missing:
        msg = f"Probabilities of '{sym}' are not extra functions of the problem: {missing}."
        raise ValueError(msg)
    prob_funcs = [extra_funcs[p] for p in dist.probabilities]

    objective = problem.get_objective(dist.symbol)
    unit = None
    if objective is not None:
        if objective.maximize != dist.maximize:
            msg = (
                f"Distribution of '{sym}' has maximize={dist.maximize}, but objective "
                f"'{dist.symbol}' has maximize={objective.maximize}."
            )
            raise ValueError(msg)
        found_type = "objectives"
        unit = objective.unit
    elif any(s.symbol == dist.symbol for s in problem.scalarization_funcs or []):
        found_type = "scalarization_funcs"
    else:
        found_type = "extra_funcs"

    return _Resolution(
        dist=dist,
        found_type=found_type,
        unit=unit,
        is_linear=all(f.is_linear for f in prob_funcs),
        is_twice_diff=all(f.is_twice_differentiable for f in prob_funcs),
    )


def _copy_unique(problem: "Problem", update: dict) -> "Problem":
    """Copy ``problem`` with ``update`` applied, rejecting duplicate symbols."""
    new_problem = problem.model_copy(update=update)
    counts = Counter(s for s in new_problem.get_all_symbols() if s is not None)
    duplicates = sorted(s for s, c in counts.items() if c > 1)
    if duplicates:
        msg = f"The aggregation would create duplicate symbols: {duplicates}."
        raise ValueError(msg)
    return new_problem


def _lte(name: str, symbol: str, func: list, *, is_linear: bool = True, is_twice_diff: bool = True) -> Constraint:
    return Constraint(
        name=name,
        symbol=symbol,
        cons_type=ConstraintTypeEnum.LTE,
        func=func,
        is_linear=is_linear,
        is_convex=is_linear,
        is_twice_differentiable=is_twice_diff,
    )


def add_expected_value_dd(
    problem: "Problem",
    distributions: dict[str, OutcomeDistribution],
    symbols: list[str],
    prefix: str = "E_",
) -> "tuple[Problem, dict[str, str]]":
    """Add expected values ``sum_u u * p(u)`` for selected symbols.

    The new element type matches the base element's (see ``OutcomeDistribution.symbol``):
    objectives stay objectives, scalarization functions stay scalarization functions, and
    everything else becomes an extra function.

    Args:
        problem: the problem defining the probability extra functions.
        distributions: mapping from symbol to its outcome distribution.
        symbols: symbols whose expected values should be added.
        prefix: prefix prepended to each symbol to form the new symbol. Defaults to ``'E_'``.

    Returns:
        A tuple of the Problem with the expected-value elements appended, and a dict mapping
        each symbol to its new expected-value symbol.

    Raises:
        ValueError: if a symbol has no distribution, a probability symbol is not an extra
            function of ``problem``, the distribution's direction differs from its base
            objective's, or a new symbol clashes with an existing one.
    """
    new_objectives = list(problem.objectives)
    new_scal_funcs = list(problem.scalarization_funcs or [])
    new_extra_funcs = list(problem.extra_funcs or [])
    added_symbols: dict[str, str] = {}

    for sym in symbols:
        info = _resolve(problem, distributions, sym)
        dist = info.dist
        expected_sym = f"{prefix}{sym}"
        added_symbols[sym] = expected_sym

        terms = [["Multiply", u, p] for u, p in zip(dist.values, dist.probabilities, strict=True)]

        append_aggregated_elem(
            info.found_type,
            new_objectives,
            new_scal_funcs,
            new_extra_funcs,
            name=f"Expected {dist.name}",
            description=f"Expected value of {dist.name}",
            symbol=expected_sym,
            func=terms[0] if len(terms) == 1 else ["Add", *terms],
            maximize=dist.maximize,
            unit=info.unit,
            is_linear=info.is_linear,
            is_convex=info.is_linear,
            is_twice_differentiable=info.is_twice_diff,
        )

    return _copy_unique(
        problem,
        {
            "objectives": new_objectives,
            "scalarization_funcs": new_scal_funcs or None,
            "extra_funcs": new_extra_funcs or None,
        },
    ), added_symbols


def add_worst_case_dd(
    problem: "Problem",
    distributions: dict[str, OutcomeDistribution],
    symbols: list[str],
    prefix: str = "robust_",
    tolerance: float = 1e-6,
) -> "tuple[Problem, dict[str, str]]":
    """Add the worst possible outcome for selected symbols.

    An outcome ``u`` is possible when ``p(u) > tolerance``. Under a MIP solver the effective
    threshold is about ``tolerance`` plus the solver's feasibility and integrality tolerances
    (e.g. ``1e-6`` and ``1e-5`` in Gurobi), so an outcome with a smaller ``p(u)`` may be
    ignored. The measure is discontinuous in ``p(u)``, so with continuous decisions the solver
    may push a bad outcome's probability to just below that threshold. With ``M = max U - min U``, a
    binary ``delta_u`` per outcome, and a variable ``t`` bounded to ``[min U, max U]``:

    * ``p(u) - tolerance <= delta_u``: a possible outcome has ``delta_u = 1``.
    * **Maximise**: ``t <= u + M * (1 - delta_u)``, and ``t`` is maximized, so ``t`` is the
      smallest possible outcome.
    * **Minimise**: ``t >= u - M * (1 - delta_u)``, and ``t`` is minimized, so ``t`` is the
      largest possible outcome.

    The variables are named ``_t_{prefix}{sym}`` and ``_delta_{prefix}{sym}_{i}`` (``i`` the
    0-based index into ``values``), and the returned element is ``{prefix}{sym}``, of the same
    type as the base element.

    Args:
        problem: the problem defining the probability extra functions.
        distributions: mapping from symbol to its outcome distribution.
        symbols: symbols whose worst cases should be added.
        prefix: prefix prepended to each symbol to form the new symbol. Defaults to ``'robust_'``.
        tolerance: probabilities at or below this are treated as impossible outcomes.

    Returns:
        A tuple of the Problem with the worst-case elements appended, and a dict mapping each
        symbol to its new worst-case symbol.

    Raises:
        ValueError: if a symbol has no distribution, a probability symbol is not an extra
            function of ``problem``, the distribution's direction differs from its base
            objective's, or a new symbol clashes with an existing one.
    """
    new_variables = list(problem.variables)
    new_objectives = list(problem.objectives)
    new_scal_funcs = list(problem.scalarization_funcs or [])
    new_extra_funcs = list(problem.extra_funcs or [])
    new_constraints = list(problem.constraints or [])
    added_symbols: dict[str, str] = {}

    for sym in symbols:
        info = _resolve(problem, distributions, sym)
        dist = info.dist
        robust_sym = f"{prefix}{sym}"
        t_sym = f"_t_{robust_sym}"
        added_symbols[sym] = robust_sym
        big_m = dist.values[-1] - dist.values[0]

        new_variables.append(
            Variable(
                name=f"Worst-case epigraph variable for {dist.name}",
                symbol=t_sym,
                variable_type=VariableTypeEnum.real,
                lowerbound=dist.values[0],
                upperbound=dist.values[-1],
                initial_value=dist.values[0],
            )
        )

        for i, (u, p) in enumerate(zip(dist.values, dist.probabilities, strict=True)):
            delta_sym = f"_delta_{robust_sym}_{i}"
            new_variables.append(
                Variable(
                    name=f"Possible outcome indicator for {dist.name} at u={u}",
                    symbol=delta_sym,
                    variable_type=VariableTypeEnum.binary,
                    lowerbound=0,
                    upperbound=1,
                    initial_value=1,
                )
            )
            new_constraints.append(
                _lte(
                    f"Possible outcome of {dist.name} at u={u}",
                    f"{robust_sym}_possible_{i}",
                    ["Subtract", ["Subtract", p, tolerance], delta_sym],
                    is_linear=info.is_linear,
                    is_twice_diff=info.is_twice_diff,
                )
            )
            # Maximise: t - u - M(1 - delta) <= 0.  Minimise: u - M(1 - delta) - t <= 0.
            slack = ["Multiply", big_m, ["Subtract", 1, delta_sym]]
            con_func = (
                ["Subtract", ["Subtract", t_sym, u], slack]
                if dist.maximize
                else ["Subtract", ["Subtract", u, slack], t_sym]
            )
            new_constraints.append(
                _lte(f"Worst-case bound for {dist.name} at u={u}", f"{robust_sym}_bound_{i}", con_func)
            )

        append_aggregated_elem(
            info.found_type,
            new_objectives,
            new_scal_funcs,
            new_extra_funcs,
            name=f"Worst-case {dist.name}",
            description=f"Worst-case value of {dist.name}",
            symbol=robust_sym,
            func=t_sym,
            maximize=dist.maximize,
            unit=info.unit,
            is_linear=True,
            is_convex=True,
            is_twice_differentiable=True,
        )

    return _copy_unique(
        problem,
        {
            "variables": new_variables,
            "objectives": new_objectives,
            "scalarization_funcs": new_scal_funcs or None,
            "extra_funcs": new_extra_funcs or None,
            "constraints": new_constraints or None,
        },
    ), added_symbols


def cvar_dd_internal_symbols(cvar_symbol: str) -> dict[str, str]:
    """The auxiliary variable symbols ``add_cvar_dd`` creates for the CVaR element ``cvar_symbol``.

    Keys are ``'eta'``, ``'lambda'``, ``'lambda_bar'``, ``'rho'`` and ``'rho_bar'``.
    """
    return {
        "eta": f"eta_{cvar_symbol}",
        "lambda": f"lambda_{cvar_symbol}",
        "lambda_bar": f"lambdabar_{cvar_symbol}",
        "rho": f"rho_{cvar_symbol}",
        "rho_bar": f"rhobar_{cvar_symbol}",
    }


def add_cvar_dd(
    problem: "Problem",
    distributions: dict[str, OutcomeDistribution],
    symbols: list[str],
    alpha: float,
    prefix: str = "CVAR_",
) -> "tuple[Problem, dict[str, str]]":
    """Add CVaR aggregations using the MILP of Herrala, Terho and Oliveira.

    Follows "Risk-averse decision strategies for influence diagrams using rooted junction
    trees" (Operations Research Letters 61 (2025) 107308, arXiv:2401.03734), Section 3.4 and
    Table 1, equations (11)-(20). The formulation is stated for a maximized ``u`` and averages
    the lowest ``alpha`` probability mass. For a minimized element it is applied to the negated
    values ``-u`` and the resulting CVaR is negated back, so it averages the highest ``alpha``
    mass and stays minimized.

    With U the (sign-adjusted) values in ascending order, ``M = max U - min U`` and
    ``epsilon`` half the smallest gap between adjacent values (1.0 when there is only one
    value), the variables are a VaR threshold ``eta``, binary ``lambda(u)`` and
    ``lambda_bar(u)``, and ``rho(u)``, ``rho_bar(u)`` in [0, 1]. The constraints are

    * (11) ``eta - u <= M * lambda(u)``
    * (12) ``eta - u >= (M + epsilon) * lambda(u) - M``
    * (13) ``eta - u <= (M + epsilon) * lambda_bar(u) - epsilon``
    * (14) ``eta - u >= M * (lambda_bar(u) - 1)``
    * (15) ``rho_bar(u) <= lambda_bar(u)``
    * (16) ``p(u) - (1 - lambda(u)) <= rho(u) <= lambda(u)``
    * (17) ``rho(u) <= rho_bar(u) <= p(u)``
    * (18) ``sum_u rho_bar(u) = alpha``

    and ``CVaR = (1 / alpha) * sum_u u * rho_bar(u)``. Table 1 of the paper describes
    ``rho_bar(u)`` at ``u = eta`` as ``alpha - sum_u p(u)``; the body text (page 7) gives
    ``alpha - sum_u rho(u)``, which is what (18) implies. Only the constraints above are imposed,
    so this erratum does not affect the result.

    ``lambda``, ``lambda_bar``, ``rho`` and ``rho_bar`` are length-``|U|`` tensor variables whose
    element ``i + 1`` belongs to the ``i``-th sign-adjusted value in ascending order: the values
    in order when maximizing and in reverse order when minimizing. Their symbols, and that of
    ``eta``, are given by ``cvar_dd_internal_symbols({prefix}{sym})``. The constraints are named
    ``cvar_{k}_{prefix}{sym}_{i}`` for ``k`` in 11, 12, 13, 14, 15, 16lo, 16hi, 17lo, 17hi,
    and ``cvar_18_{prefix}{sym}``.

    Args:
        problem: the problem defining the probability extra functions.
        distributions: mapping from symbol to its outcome distribution.
        symbols: symbols whose CVaR should be added.
        alpha: probability mass of the averaged tail of worst outcomes in the element's own
            direction, 0 < alpha < 1, e.g. 0.05 for the worst 5 %.
        prefix: prefix prepended to each symbol to form the new symbol. Defaults to ``'CVAR_'``.

    Returns:
        A tuple of the Problem with the CVaR elements appended, and a dict mapping each symbol
        to its new CVaR symbol.

    Raises:
        ValueError: if alpha is not in (0, 1), a symbol has no distribution, a probability
            symbol is not an extra function of ``problem``, the distribution's direction differs
            from its base objective's, or a new symbol clashes with an existing one.
    """
    if not 0.0 < alpha < 1.0:
        msg = f"alpha must be in (0, 1), got {alpha}."
        raise ValueError(msg)

    new_variables = list(problem.variables)
    new_objectives = list(problem.objectives)
    new_scal_funcs = list(problem.scalarization_funcs or [])
    new_extra_funcs = list(problem.extra_funcs or [])
    new_constraints = list(problem.constraints or [])
    added_symbols: dict[str, str] = {}

    for sym in symbols:
        info = _resolve(problem, distributions, sym)
        dist = info.dist
        cvar_sym = f"{prefix}{sym}"
        added_symbols[sym] = cvar_sym
        aux = cvar_dd_internal_symbols(cvar_sym)

        sign = 1.0 if dist.maximize else -1.0
        outcomes = sorted(zip((sign * v for v in dist.values), dist.probabilities, strict=True))
        us = [u for u, _ in outcomes]
        n = len(us)
        big_m = us[-1] - us[0]
        epsilon = 0.5 * min(b - a for a, b in itertools.pairwise(us)) if n > 1 else 1.0

        new_variables.extend(
            [
                Variable(name=f"VaR_alpha[{cvar_sym}]", symbol=aux["eta"], variable_type=VariableTypeEnum.real),
                TensorVariable(
                    name=f"lambda[{cvar_sym}]",
                    symbol=aux["lambda"],
                    variable_type=VariableTypeEnum.binary,
                    shape=[n],
                ),
                TensorVariable(
                    name=f"lambda_bar[{cvar_sym}]",
                    symbol=aux["lambda_bar"],
                    variable_type=VariableTypeEnum.binary,
                    shape=[n],
                ),
                TensorVariable(
                    name=f"rho[{cvar_sym}]",
                    symbol=aux["rho"],
                    variable_type=VariableTypeEnum.real,
                    shape=[n],
                    lowerbounds=0.0,
                    upperbounds=1.0,
                ),
                TensorVariable(
                    name=f"rho_bar[{cvar_sym}]",
                    symbol=aux["rho_bar"],
                    variable_type=VariableTypeEnum.real,
                    shape=[n],
                    lowerbounds=0.0,
                    upperbounds=1.0,
                ),
            ]
        )

        for i, (u, p) in enumerate(outcomes):
            lam = ["At", aux["lambda"], i + 1]
            lam_bar = ["At", aux["lambda_bar"], i + 1]
            rho = ["At", aux["rho"], i + 1]
            rho_bar = ["At", aux["rho_bar"], i + 1]
            eta_minus_u = ["Subtract", aux["eta"], u]

            new_constraints.extend(
                [
                    _lte(
                        f"CVaR VaR upper link (11) for {cvar_sym} at u={u}",
                        f"cvar_11_{cvar_sym}_{i}",
                        ["Subtract", eta_minus_u, ["Multiply", big_m, lam]],
                    ),
                    _lte(
                        f"CVaR VaR lower link (12) for {cvar_sym} at u={u}",
                        f"cvar_12_{cvar_sym}_{i}",
                        ["Subtract", ["Subtract", ["Multiply", big_m + epsilon, lam], big_m], eta_minus_u],
                    ),
                    _lte(
                        f"CVaR VaR upper link (13) for {cvar_sym} at u={u}",
                        f"cvar_13_{cvar_sym}_{i}",
                        ["Subtract", eta_minus_u, ["Subtract", ["Multiply", big_m + epsilon, lam_bar], epsilon]],
                    ),
                    _lte(
                        f"CVaR VaR lower link (14) for {cvar_sym} at u={u}",
                        f"cvar_14_{cvar_sym}_{i}",
                        ["Subtract", ["Multiply", big_m, ["Subtract", lam_bar, 1]], eta_minus_u],
                    ),
                    _lte(
                        f"CVaR rho_bar <= lambda_bar (15) for {cvar_sym} at u={u}",
                        f"cvar_15_{cvar_sym}_{i}",
                        ["Subtract", rho_bar, lam_bar],
                    ),
                    _lte(
                        f"CVaR rho lower bound (16) for {cvar_sym} at u={u}",
                        f"cvar_16lo_{cvar_sym}_{i}",
                        ["Subtract", ["Subtract", p, ["Subtract", 1, lam]], rho],
                        is_linear=info.is_linear,
                        is_twice_diff=info.is_twice_diff,
                    ),
                    _lte(
                        f"CVaR rho upper bound (16) for {cvar_sym} at u={u}",
                        f"cvar_16hi_{cvar_sym}_{i}",
                        ["Subtract", rho, lam],
                    ),
                    _lte(
                        f"CVaR rho <= rho_bar (17) for {cvar_sym} at u={u}",
                        f"cvar_17lo_{cvar_sym}_{i}",
                        ["Subtract", rho, rho_bar],
                    ),
                    _lte(
                        f"CVaR rho_bar <= p (17) for {cvar_sym} at u={u}",
                        f"cvar_17hi_{cvar_sym}_{i}",
                        ["Subtract", rho_bar, p],
                        is_linear=info.is_linear,
                        is_twice_diff=info.is_twice_diff,
                    ),
                ]
            )

        new_constraints.append(
            Constraint(
                name=f"CVaR tail mass total (18) for {cvar_sym}",
                symbol=f"cvar_18_{cvar_sym}",
                cons_type=ConstraintTypeEnum.EQ,
                func=["Subtract", ["Sum", aux["rho_bar"]], alpha],
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
        )

        terms = [["Multiply", u, ["At", aux["rho_bar"], i + 1]] for i, u in enumerate(us)]
        weighted = terms[0] if n == 1 else ["Add", *terms]

        append_aggregated_elem(
            info.found_type,
            new_objectives,
            new_scal_funcs,
            new_extra_funcs,
            name=f"CVaR of {dist.name}",
            description=f"CVaR of {dist.name}",
            symbol=cvar_sym,
            # sign negates back to the element's own units when minimizing.
            func=["Multiply", sign / alpha, weighted],
            maximize=dist.maximize,
            unit=info.unit,
            is_linear=True,
            is_convex=True,
            is_twice_differentiable=True,
        )

    return _copy_unique(
        problem,
        {
            "variables": new_variables,
            "objectives": new_objectives,
            "scalarization_funcs": new_scal_funcs or None,
            "extra_funcs": new_extra_funcs or None,
            "constraints": new_constraints or None,
        },
    ), added_symbols
