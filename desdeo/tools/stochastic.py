"""Tools for solving scenario-based optimization problems."""

from typing import TYPE_CHECKING

from desdeo.problem.schema import (
    Constraint,
    ConstraintTypeEnum,
    Variable,
    VariableTypeEnum,
)
from desdeo.tools.scalarization import add_asf_diff, add_asf_nondiff
from desdeo.tools.scenarios import (
    append_aggregated_elem,
    build_combined_scenario_problem,
    leaf_expr,
    resolve_elem,
    weighted_sum_expr,
)

if TYPE_CHECKING:
    from desdeo.problem.scenario import ScenarioModel
    from desdeo.problem.schema import Problem


def add_expected_asf(
    scenario_model: "ScenarioModel",
    symbol: str,
    reference_point: dict[str, float],
    *,
    ideal: dict[str, float] | None = None,
    nadir: dict[str, float] | None = None,
    rho: float = 1e-6,
    delta: float = 1e-6,
) -> "tuple[Problem, str]":
    """Build a combined scenario problem with an expected value of ASF scalarization.

    This is here mostly as a convenience for testing, but it can also be used as a template
    for users who want to build their own scenario problems expected-value scalarizations.

    Args:
        scenario_model: the ScenarioModel to expand and scalarize.
        symbol: symbol for the scalarization function added to the problem.
        reference_point: maps objective symbols to reference point values.
        ideal: maps objective symbols to ideal values. If None, the problem's
            ideal is used.
        nadir: maps objective symbols to nadir values. If None, the problem's
            nadir is used.
        rho: augmentation term weight for the ASF.
        delta: small perturbation for the differentiable ASF variant.

    Returns:
        A tuple of the combined Problem and the scalarization function symbol.
    """
    base_problem = scenario_model.base_problem
    if base_problem.is_twice_differentiable or base_problem.is_linear:
        scal_problem, scal = add_asf_diff(
            problem=base_problem,
            symbol=symbol,
            reference_point=reference_point,
            ideal=ideal,
            nadir=nadir,
            rho=rho,
            delta=delta,
        )
    else:
        scal_problem, scal = add_asf_nondiff(
            problem=base_problem,
            symbol=symbol,
            reference_point=reference_point,
            ideal=ideal,
            nadir=nadir,
            rho=rho,
            delta=delta,
        )

    modified_model = scenario_model.with_base_problem(problem=scal_problem, validate=True)
    combined, symbol_maps = build_combined_scenario_problem(modified_model)
    combined, added = add_expected_value(modified_model, [scal], combined=combined, symbol_maps=symbol_maps)

    return combined, added[scal]


def add_expected_value(
    scenario_model: "ScenarioModel",
    symbols: list[str],
    *,
    prefix: str = "E_",
    combined: "Problem | None" = None,
    symbol_maps: "dict[str, dict[str, dict[str, str]]] | None" = None,
) -> "tuple[Problem, dict[str, str]]":
    """Add expected-value aggregations for selected symbols to the combined scenario problem.

    For each symbol the expected value is a probability-weighted sum of the per-leaf
    copies of that symbol in the combined problem.  The new element type matches the
    original: objectives stay objectives, scalarization functions stay scalarization
    functions, and everything else (extra functions, constraints, …) becomes an extra
    function.

    Args:
        scenario_model: the ScenarioModel to expand.
        symbols: original symbols whose expected values should be added.
        prefix: prefix prepended to each original symbol to form the new symbol.
            Defaults to ``'E_'``.
        combined: pre-built combined Problem. If provided together with
            ``symbol_maps``, ``build_combined_scenario_problem`` is not called.
            Must match ``scenario_model``.
        symbol_maps: pre-built symbol maps from ``build_combined_scenario_problem``.
            Must be provided together with ``combined``; ignored otherwise.

    Returns:
        A tuple of the combined Problem with the expected-value elements appended,
        and a dict mapping each original symbol to its new expected-value symbol.

    Raises:
        ValueError: if a requested symbol is not found in the combined problem.
    """
    if combined is None or symbol_maps is None:
        combined, symbol_maps = build_combined_scenario_problem(scenario_model)
    weights = scenario_model.leaf_scenarios

    new_objectives = list(combined.objectives or [])
    new_scal_funcs = list(combined.scalarization_funcs or [])
    new_extra_funcs = list(combined.extra_funcs or [])
    added_symbols: dict[str, str] = {}

    for sym in symbols:
        info = resolve_elem(sym, symbol_maps, combined, scenario_model)
        expected_sym = f"{prefix}{sym}"
        added_symbols[sym] = expected_sym

        expected_expr = weighted_sum_expr(info, weights, combined)

        append_aggregated_elem(
            info.found_type,
            new_objectives,
            new_scal_funcs,
            new_extra_funcs,
            name=f"Expected {info.elem_name}",
            description=f"Expected value of {info.elem_desc}"
            if info.elem_desc
            else f"Expected value of {info.elem_name}",
            symbol=expected_sym,
            func=expected_expr,
            maximize=info.maximize,
            is_linear=info.is_linear,
            is_convex=info.is_convex,
            is_twice_differentiable=info.is_twice_diff,
        )

    return combined.model_copy(
        update={
            "objectives": new_objectives or None,
            "scalarization_funcs": new_scal_funcs or None,
            "extra_funcs": new_extra_funcs or None,
        }
    ), added_symbols


_DISPERSION_LABELS = {"variance": "Variance", "std": "Standard deviation", "mad": "Mean absolute deviation"}


def _real_variable(name: str, symbol: str, lowerbound: float | None = None) -> Variable:
    return Variable(
        name=name,
        symbol=symbol,
        variable_type=VariableTypeEnum.real,
        lowerbound=lowerbound,
        upperbound=None,
        initial_value=0.0,
    )


def _ensure_helper(
    variable: Variable,
    defining: list[Constraint],
    variables: dict[str, Variable],
    constraints: dict[str, Constraint],
    taken: set[str],
) -> None:
    """Add a helper variable and its defining constraints, or reuse an identical existing one.

    Helpers are named after the aggregated symbol rather than the aggregate, so variance,
    standard deviation and mean absolute deviation of the same symbol share them.

    Raises:
        ValueError: if a helper's symbol is in use by anything other than the same helper.
    """
    symbols = [variable.symbol, *(con.symbol for con in defining)]
    if variable.symbol in variables and all(
        con.symbol in constraints
        and constraints[con.symbol].func == con.func
        and constraints[con.symbol].cons_type == con.cons_type
        for con in defining
    ):
        return
    if clashes := sorted(set(symbols) & taken):
        msg = f"Helper symbols {', '.join(clashes)} are already in use by other elements."
        raise ValueError(msg)

    variables[variable.symbol] = variable
    constraints.update({con.symbol: con for con in defining})
    taken.update(symbols)


def _ensure_deviations(
    sym: str,
    info,
    weights: "dict[str, float]",
    *,
    variables: dict[str, Variable],
    constraints: dict[str, Constraint],
    taken: set[str],
) -> dict[str, str]:
    """Ensure the mean ``_mean_{sym}`` and per-leaf deviations ``_dev_{leaf}_{sym}``; return the latter."""
    # Constraints are evaluated last, so they can name every per-leaf copy directly.
    flags = {"is_linear": info.is_linear, "is_convex": info.is_linear, "is_twice_differentiable": info.is_twice_diff}

    mean_sym = f"_mean_{sym}"
    _ensure_helper(
        _real_variable(f"Mean of {info.elem_name} over scenarios", mean_sym),
        [
            Constraint(
                name=f"Mean of {info.elem_name} over scenarios",
                symbol=f"{mean_sym}_con",
                func=["Add", mean_sym, *[["Multiply", -weights[leaf], info.per_leaf[leaf]] for leaf in weights]],
                cons_type=ConstraintTypeEnum.EQ,
                **flags,
            )
        ],
        variables,
        constraints,
        taken,
    )

    dev_syms = {leaf: f"_dev_{leaf}_{sym}" for leaf in weights}
    for leaf, dev_sym in dev_syms.items():
        _ensure_helper(
            _real_variable(f"Deviation of {info.elem_name} from its mean in {leaf}", dev_sym),
            [
                Constraint(
                    name=f"Deviation of {info.elem_name} from its mean in {leaf}",
                    symbol=f"{dev_sym}_con",
                    func=["Add", dev_sym, ["Negate", info.per_leaf[leaf]], mean_sym],
                    cons_type=ConstraintTypeEnum.EQ,
                    **flags,
                )
            ],
            variables,
            constraints,
            taken,
        )
    return dev_syms


def _ensure_abs_deviations(
    sym: str,
    info,
    dev_syms: dict[str, str],
    *,
    variables: dict[str, Variable],
    constraints: dict[str, Constraint],
    taken: set[str],
) -> dict[str, str]:
    """Ensure per-leaf upper bounds ``_absdev_{leaf}_{sym} >= |d_s|``; return their symbols."""
    flags = {"is_linear": True, "is_convex": True, "is_twice_differentiable": True}
    abs_syms = {leaf: f"_absdev_{leaf}_{sym}" for leaf in dev_syms}
    for leaf, abs_sym in abs_syms.items():
        dev_sym = dev_syms[leaf]
        _ensure_helper(
            _real_variable(f"Absolute deviation of {info.elem_name} in {leaf}", abs_sym, lowerbound=0.0),
            [
                Constraint(
                    name=f"Absolute deviation of {info.elem_name} in {leaf} (positive side)",
                    symbol=f"{abs_sym}_pos",
                    func=["Add", dev_sym, ["Negate", abs_sym]],
                    cons_type=ConstraintTypeEnum.LTE,
                    **flags,
                ),
                Constraint(
                    name=f"Absolute deviation of {info.elem_name} in {leaf} (negative side)",
                    symbol=f"{abs_sym}_neg",
                    func=["Add", ["Negate", dev_sym], ["Negate", abs_sym]],
                    cons_type=ConstraintTypeEnum.LTE,
                    **flags,
                ),
            ],
            variables,
            constraints,
            taken,
        )
    return abs_syms


def _add_dispersion(
    scenario_model: "ScenarioModel",
    symbols: list[str],
    *,
    kind: str,
    prefix: str,
    combined: "Problem | None",
    symbol_maps: "dict[str, dict[str, dict[str, str]]] | None",
) -> "tuple[Problem, dict[str, str]]":
    """Shared implementation of the variance, standard deviation and mean absolute deviation.

    The mean and the per-leaf deviations are helper variables tied to the per-leaf copies
    by linear equality constraints, so the aggregates only square or bound single
    variables. Squaring the per-leaf expressions themselves can be very large for
    tensor-valued problems.
    """
    if combined is None or symbol_maps is None:
        combined, symbol_maps = build_combined_scenario_problem(scenario_model)
    weights = scenario_model.leaf_scenarios
    label = _DISPERSION_LABELS[kind]

    variables = {v.symbol: v for v in combined.variables}
    constraints = {c.symbol: c for c in combined.constraints or []}
    new_objectives = list(combined.objectives or [])
    new_scal_funcs = list(combined.scalarization_funcs or [])
    new_extra_funcs = list(combined.extra_funcs or [])
    taken = set(combined.get_all_symbols())
    added_symbols: dict[str, str] = {}

    for sym in symbols:
        info = resolve_elem(sym, symbol_maps, combined, scenario_model)
        new_sym = f"{prefix}{sym}"
        if new_sym in taken:
            msg = f"Symbol '{new_sym}' for the {label.lower()} of '{sym}' is already in use; choose another prefix."
            raise ValueError(msg)
        taken.add(new_sym)
        added_symbols[sym] = new_sym

        dev_syms = _ensure_deviations(sym, info, weights, variables=variables, constraints=constraints, taken=taken)
        if kind == "mad":
            abs_syms = _ensure_abs_deviations(
                sym, info, dev_syms, variables=variables, constraints=constraints, taken=taken
            )
            terms = [["Multiply", weights[leaf], abs_syms[leaf]] for leaf in weights]
        else:
            terms = [["Multiply", weights[leaf], ["Square", dev_syms[leaf]]] for leaf in weights]
        expr = terms[0] if len(terms) == 1 else ["Add", *terms]
        if kind == "std":
            expr = ["Sqrt", expr]

        append_aggregated_elem(
            info.found_type,
            new_objectives,
            new_scal_funcs,
            new_extra_funcs,
            name=f"{label} of {info.elem_name}",
            description=f"{label} of {info.elem_desc}" if info.elem_desc else f"{label} of {info.elem_name}",
            symbol=new_sym,
            func=expr,
            # Dispersion is a risk to reduce whatever the direction of the original objective.
            maximize=False,
            is_linear=kind == "mad",
            # Weighted sums of squares or of nonnegative variables are convex, as is the
            # square root of the former (a norm).
            is_convex=True,
            is_twice_differentiable=True,
        )

    return combined.model_copy(
        update={
            "variables": list(variables.values()),
            "objectives": new_objectives or None,
            "scalarization_funcs": new_scal_funcs or None,
            "extra_funcs": new_extra_funcs or None,
            "constraints": list(constraints.values()) or None,
        }
    ), added_symbols


def add_variance(
    scenario_model: "ScenarioModel",
    symbols: list[str],
    *,
    prefix: str = "VARIANCE_",
    combined: "Problem | None" = None,
    symbol_maps: "dict[str, dict[str, dict[str, str]]] | None" = None,
) -> "tuple[Problem, dict[str, str]]":
    """Add variance aggregations for selected symbols to the combined scenario problem.

    For each symbol the variance over the leaf scenarios is ``sum_s p_s * d_s^2`` where
    ``d_s = sym_s - m`` and ``m = sum_t p_t * sym_t``. It adds, unless already present:

    - ``_mean_{sym}``: the mean ``m``, fixed by the equality constraint ``_mean_{sym}_con``.
    - ``_dev_{leaf}_{sym}``: the per-leaf deviation ``d_s``, fixed by ``_dev_{leaf}_{sym}_con``.

    These helpers are shared with :func:`add_standard_deviation` and
    :func:`add_mean_absolute_deviation` on the same symbol. The constraints are linear when
    ``sym`` is, and the variance only squares single variables.

    The new ``{prefix}{sym}`` element keeps the type of the original (objective stays
    objective, scalarization function stays scalarization function, anything else becomes
    an extra function). Variance objectives are always minimized. The default prefix is
    ``'VARIANCE_'`` rather than ``'VAR_'``, which :func:`add_conditional_value_at_risk`
    uses for its VaR threshold variables.

    Args:
        scenario_model: the ScenarioModel used to build or match ``combined``.
        symbols: original symbols whose variance should be added.
        prefix: prefix prepended to each original symbol to form the new symbol.
            Defaults to ``'VARIANCE_'``.
        combined: pre-built combined Problem. If provided together with
            ``symbol_maps``, ``build_combined_scenario_problem`` is not called.
        symbol_maps: pre-built symbol maps; required together with ``combined``.

    Returns:
        A tuple of the updated combined Problem and a dict mapping each original
            symbol to its variance symbol.

    Raises:
        ValueError: if a requested symbol is not found in the combined problem, or a
            new symbol is already in use.
    """
    return _add_dispersion(
        scenario_model, symbols, kind="variance", prefix=prefix, combined=combined, symbol_maps=symbol_maps
    )


def add_standard_deviation(
    scenario_model: "ScenarioModel",
    symbols: list[str],
    *,
    prefix: str = "STD_",
    combined: "Problem | None" = None,
    symbol_maps: "dict[str, dict[str, dict[str, str]]] | None" = None,
) -> "tuple[Problem, dict[str, str]]":
    """Add standard-deviation aggregations for selected symbols to the combined scenario problem.

    The standard deviation is the square root of the variance built by
    :func:`add_variance`, using the same shared helpers. Standard deviation objectives are
    always minimized.

    Args:
        scenario_model: the ScenarioModel used to build or match ``combined``.
        symbols: original symbols whose standard deviation should be added.
        prefix: prefix prepended to each original symbol to form the new symbol.
            Defaults to ``'STD_'``.
        combined: pre-built combined Problem. If provided together with
            ``symbol_maps``, ``build_combined_scenario_problem`` is not called.
        symbol_maps: pre-built symbol maps; required together with ``combined``.

    Returns:
        A tuple of the updated combined Problem and a dict mapping each original
            symbol to its standard-deviation symbol.

    Raises:
        ValueError: if a requested symbol is not found in the combined problem, or a
            new symbol is already in use.
    """
    return _add_dispersion(
        scenario_model, symbols, kind="std", prefix=prefix, combined=combined, symbol_maps=symbol_maps
    )


def add_mean_absolute_deviation(
    scenario_model: "ScenarioModel",
    symbols: list[str],
    *,
    prefix: str = "MAD_",
    combined: "Problem | None" = None,
    symbol_maps: "dict[str, dict[str, dict[str, str]]] | None" = None,
) -> "tuple[Problem, dict[str, str]]":
    """Add mean-absolute-deviation aggregations for selected symbols to the combined scenario problem.

    The mean absolute deviation is ``sum_s p_s * |d_s|`` with the deviations ``d_s`` shared
    with :func:`add_variance`. The absolute values use the linear epigraph form: helpers
    ``_absdev_{leaf}_{sym} >= 0`` with the constraints ``_absdev_{leaf}_{sym}_pos``
    (``d_s <= u_s``) and ``_absdev_{leaf}_{sym}_neg`` (``-d_s <= u_s``), and the aggregate is
    ``sum_s p_s * u_s``. The aggregate is linear when ``sym`` is.

    The epigraph is exact only where something pushes ``u_s`` down: when the aggregate is
    minimized or bounded from above. Elsewhere ``u_s`` may exceed ``|d_s|``, so the value
    is an upper bound; recompute it from the deviations if it must be exact. Mean absolute
    deviation objectives are always minimized.

    Args:
        scenario_model: the ScenarioModel used to build or match ``combined``.
        symbols: original symbols whose mean absolute deviation should be added.
        prefix: prefix prepended to each original symbol to form the new symbol.
            Defaults to ``'MAD_'``.
        combined: pre-built combined Problem. If provided together with
            ``symbol_maps``, ``build_combined_scenario_problem`` is not called.
        symbol_maps: pre-built symbol maps; required together with ``combined``.

    Returns:
        A tuple of the updated combined Problem and a dict mapping each original
            symbol to its mean-absolute-deviation symbol.

    Raises:
        ValueError: if a requested symbol is not found in the combined problem, or a
            new symbol is already in use.
    """
    return _add_dispersion(
        scenario_model, symbols, kind="mad", prefix=prefix, combined=combined, symbol_maps=symbol_maps
    )


def add_conditional_value_at_risk(
    scenario_model: "ScenarioModel",
    symbols: list[str],
    alpha: float,
    *,
    var_prefix: str = "VAR_",
    cvar_prefix: str = "CVAR_",
    combined: "Problem | None" = None,
    symbol_maps: "dict[str, dict[str, dict[str, str]]] | None" = None,
) -> "tuple[Problem, dict[str, str]]":
    """Add CVaR aggregations using the Rockafellar-Uryasev LP formulation.

    For each symbol in ``symbols``, introduces:

    - ``{var_prefix}{sym}``: VaR threshold variable η (shared across scenarios).
    - ``{leaf}_{var_prefix}{sym}``: per-leaf auxiliary z_s ≥ 0 such that
      z_s >= sym_s - eta.
    - ``{leaf}_{var_prefix}{sym}_con``: per-leaf LTE constraint sym_s - eta - z_s <= 0.
    - ``{cvar_prefix}{sym}``: CVaR = eta + 1/(1-alpha) * sum_s p_s * z_s.

    For maximized objectives the bad tail is the low values, so the constraints are
    eta - sym_s - z_s <= 0 and CVaR = eta - 1/(1-alpha) * sum_s p_s * z_s, which stays
    maximized. Non-objective elements always use the minimization form.

    The CVaR element preserves the type of the original element (objective stays
    objective, scalarization function stays scalarization function, anything else
    becomes an extra function).

    Args:
        scenario_model: the ScenarioModel used to build or match ``combined``.
        symbols: original symbols whose CVaR should be added.
        alpha: confidence level, 0 < alpha < 1 (e.g., 0.95 for 95% CVaR).
        var_prefix: prefix for the VaR threshold and auxiliary variables.
            Defaults to ``'VAR_'``.
        cvar_prefix: prefix for the resulting CVaR element. Defaults to ``'CVAR_'``.
        combined: pre-built combined Problem. If provided together with
            ``symbol_maps``, ``build_combined_scenario_problem`` is not called.
        symbol_maps: pre-built symbol maps; required together with ``combined``.

    Returns:
        A tuple of the updated combined Problem and a dict mapping each original
            symbol to its CVaR symbol.

    Raises:
        ValueError: if a requested symbol is not found in the combined problem.
    """
    if combined is None or symbol_maps is None:
        combined, symbol_maps = build_combined_scenario_problem(scenario_model)
    weights = scenario_model.leaf_scenarios
    scale = 1.0 / (1.0 - alpha)

    new_variables = list(combined.variables)
    new_objectives = list(combined.objectives or [])
    new_scal_funcs = list(combined.scalarization_funcs or [])
    new_extra_funcs = list(combined.extra_funcs or [])
    new_constraints = list(combined.constraints or [])
    added_symbols: dict[str, str] = {}

    for sym in symbols:
        info = resolve_elem(sym, symbol_maps, combined, scenario_model)
        var_sym = f"{var_prefix}{sym}"
        cvar_sym = f"{cvar_prefix}{sym}"
        added_symbols[sym] = cvar_sym
        is_maximize = info.found_type == "objectives" and info.maximize

        # VaR threshold η — shared across all scenarios.
        new_variables.append(
            Variable(
                name=f"VaR threshold for {info.elem_name}",
                symbol=var_sym,
                variable_type=VariableTypeEnum.real,
                lowerbound=None,
                upperbound=None,
                initial_value=0.0,
            )
        )

        # Per-leaf auxiliary variables z_s and their constraints.
        leaf_z_syms: dict[str, str] = {}
        for leaf in info.per_leaf:
            z_sym = f"{leaf}_{var_sym}"
            leaf_z_syms[leaf] = z_sym

            new_variables.append(
                Variable(
                    name=f"CVaR auxiliary for {info.elem_name} in {leaf}",
                    symbol=z_sym,
                    variable_type=VariableTypeEnum.real,
                    lowerbound=0.0,
                    upperbound=None,
                    initial_value=0.0,
                )
            )

            # Minimize: z_s >= sym_s - eta; maximize: z_s >= eta - sym_s
            leaf_func = leaf_expr(info, leaf)
            con_func = (
                ["Add", var_sym, ["Negate", leaf_func], ["Negate", z_sym]]
                if is_maximize
                else ["Add", leaf_func, ["Negate", var_sym], ["Negate", z_sym]]
            )
            new_constraints.append(
                Constraint(
                    name=f"CVaR constraint for {info.elem_name} in {leaf}",
                    symbol=f"{z_sym}_con",
                    func=con_func,
                    cons_type=ConstraintTypeEnum.LTE,
                    is_linear=info.is_linear,
                    is_convex=info.is_convex,
                    is_twice_differentiable=info.is_twice_diff,
                )
            )

        # CVaR = η ± scale · Σ_s p_s · z_s (minus for maximized objectives)
        weighted_z = [["Multiply", weights[leaf], leaf_z_syms[leaf]] for leaf in weights]
        sum_z = weighted_z[0] if len(weighted_z) == 1 else ["Add", *weighted_z]

        append_aggregated_elem(
            info.found_type,
            new_objectives,
            new_scal_funcs,
            new_extra_funcs,
            name=f"CVaR of {info.elem_name}",
            description=f"CVaR of {info.elem_desc}" if info.elem_desc else f"CVaR of {info.elem_name}",
            symbol=cvar_sym,
            func=["Add", var_sym, ["Multiply", -scale if is_maximize else scale, sum_z]],
            maximize=is_maximize,
            is_linear=True,
            is_convex=True,
            is_twice_differentiable=True,
        )

    return combined.model_copy(
        update={
            "variables": new_variables,
            "objectives": new_objectives or None,
            "scalarization_funcs": new_scal_funcs or None,
            "extra_funcs": new_extra_funcs or None,
            "constraints": new_constraints or None,
        }
    ), added_symbols
