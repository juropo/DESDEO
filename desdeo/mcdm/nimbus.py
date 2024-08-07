"""Functions related to the Sychronous NIMBUS method.

References:
    Miettinen, K., & Mäkelä, M. M. (2006). Synchronous approach in interactive
        multiobjective optimization. European Journal of Operational Research,
        170(3), 909–922.
"""  # noqa: RUF002

import numpy as np

from desdeo.problem import GenericEvaluator, Problem, VariableType, variable_dict_to_numpy_array
from desdeo.tools import (
    BaseSolver,
    SolverOptions,
    SolverResults,
    add_asf_diff,
    add_asf_nondiff,
    add_guess_sf_diff,
    add_guess_sf_nondiff,
    add_nimbus_sf_diff,
    add_nimbus_sf_nondiff,
    add_stom_sf_diff,
    add_stom_sf_nondiff,
    guess_best_solver,
)


class NimbusError(Exception):
    """Raised when an error with a NIMBUS method is encountered."""


def solve_intermediate_solutions(  # noqa: PLR0913
    problem: Problem,
    solution_1: dict[str, VariableType],
    solution_2: dict[str, VariableType],
    num_desired: int,
    scalarization_options: dict | None = None,
    solver: BaseSolver | None = None,
    solver_options: SolverOptions | None = None,
) -> list[SolverResults]:
    """Generates a desired number of intermediate solutions between two given solutions.

    Generates a desires number of intermediate solutions given two Pareto optimal solutions.
    The solutions are generated by taking n number of steps between the two solutions in the
    objective space. The objective vectors corresponding to these solutions are then
    utilized as reference points in the achievement scalarizing function. Solving the functions
    for each reference point will project the reference point on the Pareto optimal
    front of the problem. These projected solutions are then returned. Note that the
    intermediate solutions are generated _between_ the two given solutions, this means the
    returned solutions will not include the original points.

    Args:
        problem (Problem): the problem being solved.
        solution_1 (dict[str, VariableType]): the first of the solutions between which the intermediate
            solutions are to be generated.
        solution_2 (dict[str, VariableType]): the second of the solutions between which the intermediate
            solutions are to be generated.
        num_desired (int): the number of desired intermediate solutions to be generated. Must be at least `1`.
        scalarization_options (dict | None, optional): optional kwargs passed to the scalarization function.
            Defaults to None.
        solver (BaseSolver | None, optional): solver used to solve the problem.
            If not given, an appropriate solver will be automatically determined based on the features of `problem`.
            Defaults to None.
        solver_options (SolverOptions | None, optional): optional options passed
            to the `solver`. Ignored if `solver` is `None`.
            Defaults to None.

    Returns:
        list[SolverResults]: a list with the projected intermediate solutions as
            `SolverResults` objects.
    """
    if int(num_desired) < 1:
        msg = f"The given number of desired intermediate ({num_desired=}) solutions must be at least 1."
        raise NimbusError(msg)

    init_solver = guess_best_solver(problem) if solver is None else solver
    _solver_options = None if solver_options is None or solver is None else solver_options

    # compute the element-wise difference between each solution (in the decision space)
    solution_1_arr = variable_dict_to_numpy_array(problem, solution_1)
    solution_2_arr = variable_dict_to_numpy_array(problem, solution_2)
    delta = solution_1_arr - solution_2_arr

    # the '2' is in the denominator because we want to calculate the steps
    # between the two given points; we are not interested in the given points themselves.
    step_size = delta / (2 + num_desired)

    intermediate_points = np.array([solution_2_arr + i * step_size for i in range(1, num_desired + 1)])

    xs = {f"{variable.symbol}": intermediate_points[:, i].tolist() for (i, variable) in enumerate(problem.variables)}

    # evaluate the intermediate points to get reference points
    # TODO(gialmisi): an evaluator might have to be selected depending on the problem
    evaluator = GenericEvaluator(problem)

    reference_points: list[dict[str, float]] = (
        evaluator.evaluate(xs).select([obj.symbol for obj in problem.objectives]).to_dicts()
    )

    # for each reference point, add and solve the ASF scalarization problem
    # projecting the reference point onto the Pareto optimal front of the problem.
    # TODO(gialmisi): this can be done in parallel.
    intermediate_solutions = []
    for rp in reference_points:
        # add scalarization
        # TODO(gialmisi): add logic that selects correct variant of the ASF
        # depending on problem properties (either diff or non-diff)
        asf_problem, target = add_asf_diff(problem, "target", rp, **(scalarization_options or {}))

        solver = init_solver(asf_problem, _solver_options)

        # solve and store results
        result: SolverResults = solver.solve(target)

        intermediate_solutions.append(result)

    return intermediate_solutions


def infer_classifications(
    problem: Problem, current_objectives: dict[str, float], reference_point: dict[str, float]
) -> dict[str, tuple[str, float | None]]:
    r"""Infers NIMBUS classifications based on a reference point and current objective values.

    Infers the classifications based on a given reference point and current objective function
    values. The following classifications are inferred for each objective:

    - $I^{<}$: values that should improve, the reference point value of an objective
        function is equal to its ideal value;
    - $I^{\leq}$: values that should improve until a given aspiration level, the reference point
        value of an objective function is better than the current value;
    - $I^{=}$: values that should stay as they are, the reference point value of an objective
        function is equal to the current value;
    - $I^{\geq}$: values that can be impaired until some reservation level, the reference point
        value of an objective function is worse than the current value; and
    - $I^{\diamond}$: values that are allowed to change freely, the reference point value of
        and objective function is equal to its nadir value.

    The aspiration levels and the reservation levels are then given for each classification, when relevant, in
    the return value of this function as the following example demonstrates:

    ```python
    classifications = {
        "f_1": ("<", None),
        "f_2": ("<=", 42.1),
        "f_3": (">=", 22.2),
        "f_4": ("0", None)
        }
    ```

    Raises:
        NimbusError: the ideal or nadir point, or both, of the given
            problem are undefined.
        NimbusError: the reference point or current objectives are missing
            entries for one or more of the objective functions defined in
            the problem.

    Args:
        problem (Problem): the problem the current objectives and the reference point
            are related to.
        current_objectives (dict[str, float]): an objective dictionary with the current
            objective functions values.
        reference_point (dict[str, float]): an objective dictionary with the reference point
            values.

    Returns:
        dict[str, tuple[str, float | None]]: a dict with keys corresponding to the
            symbols of the objective functions defined for the problem and with values
            of tuples, where the first element is the classification (str) and the second
            element is either a reservation level (in case of classification `>=`) or an
            aspiration level (in case of classification `<=`).
    """
    if None in problem.get_ideal_point() or None in problem.get_nadir_point():
        msg = "The given problem must have both an ideal and nadir point defined."
        raise NimbusError(msg)

    if not all(obj.symbol in reference_point for obj in problem.objectives):
        msg = f"The reference point {reference_point} is missing entries " "for one or more of the objective functions."
        raise NimbusError(msg)

    if not all(obj.symbol in current_objectives for obj in problem.objectives):
        msg = f"The current point {reference_point} is missing entries " "for one or more of the objective functions."
        raise NimbusError(msg)

    # derive the classifications based on the reference point and and previous
    # objective function values
    classifications = {}

    for obj in problem.objectives:
        if np.isclose(reference_point[obj.symbol], obj.nadir):
            # the objective is free to change
            classification = {obj.symbol: ("0", None)}
        elif np.isclose(reference_point[obj.symbol], obj.ideal):
            # the objective should improve
            classification = {obj.symbol: ("<", None)}
        elif np.isclose(reference_point[obj.symbol], current_objectives[obj.symbol]):
            # the objective should stay as it is
            classification = {obj.symbol: ("=", None)}
        elif not obj.maximize and reference_point[obj.symbol] < current_objectives[obj.symbol]:
            # minimizing objective, reference value smaller, this is an aspiration level
            # improve until
            classification = {obj.symbol: ("<=", reference_point[obj.symbol])}
        elif not obj.maximize and reference_point[obj.symbol] > current_objectives[obj.symbol]:
            # minimizing objective, reference value is greater, this is a reservations level
            # impair until
            classification = {obj.symbol: (">=", reference_point[obj.symbol])}
        elif obj.maximize and reference_point[obj.symbol] < current_objectives[obj.symbol]:
            # maximizing objective, reference value is smaller, this is a reservation level
            # impair until
            classification = {obj.symbol: (">=", reference_point[obj.symbol])}
        elif obj.maximize and reference_point[obj.symbol] > current_objectives[obj.symbol]:
            # maximizing objective, reference value is greater, this is an aspiration level
            # improve until
            classification = {obj.symbol: ("<=", reference_point[obj.symbol])}
        else:
            # could not figure classification
            msg = f"Warning: NIMBUS could not figure out the classification for objective {obj.symbol}."

        classifications |= classification

    return classifications


def solve_sub_problems(  # noqa: PLR0913
    problem: Problem,
    current_objectives: dict[str, float],
    reference_point: dict[str, float],
    num_desired: int,
    scalarization_options: dict | None = None,
    solver: BaseSolver | None = None,
    solver_options: SolverOptions | None = None,
) -> list[SolverResults]:
    r"""Solves a desired number of sub-problems as defined in the NIMBUS methods.

    Solves 1-4 scalarized problems utilizing different scalarization
    functions. The scalarizations are based on the classification of a
    solutions provided by a decision maker. The classifications
    are represented by a reference point. Returns a number of new solutions
    corresponding to the number of scalarization functions solved.

    Depending on `num_desired`, solves the following scalarized problems corresponding
    the the following scalarization functions:

    1.  the NIMBUS scalarization function,
    2.  the STOM scalarization function,
    3.  the achievement scalarizing function, and
    4.  the GUESS scalarization function.

    Raises:
        NimbusError: the given problem has an undefined ideal or nadir point, or both.
        NimbusError: either the reference point of current objective functions value are
            missing entries for one or more of the objective functions defined in the problem.

    Args:
        problem (Problem): the problem being solved.
        current_objectives (dict[str, float]): an objective dictionary with the objective functions values
            the classifications have been given with respect to.
        reference_point (dict[str, float]): an objective dictionary with a reference point.
            The classifications utilized in the sub problems are derived from
            the reference point.
        num_desired (int): the number of desired solutions to be solved. Solves as
            many scalarized problems. The value must be in the range 1-4.
        scalarization_options (dict | None, optional): optional kwargs passed to the scalarization function.
            Defaults to None.
        solver (BaseSolver | None, optional): solver used to solve the problem.
            If not given, an appropriate solver will be automatically determined based on the features of `problem`.
            Defaults to None.
        solver_options (SolverOptions | None, optional): optional options passed
            to the `solver`. Ignored if `solver` is `None`.
            Defaults to None.

    Returns:
        list[SolverResults]: a list of `SolverResults` objects. Contains as many elements
            as defined in `num_desired`.
    """
    if None in problem.get_ideal_point() or None in problem.get_nadir_point():
        msg = "The given problem must have both an ideal and nadir point defined."
        raise NimbusError(msg)

    if not all(obj.symbol in reference_point for obj in problem.objectives):
        msg = f"The reference point {reference_point} is missing entries " "for one or more of the objective functions."
        raise NimbusError(msg)

    if not all(obj.symbol in current_objectives for obj in problem.objectives):
        msg = f"The current point {reference_point} is missing entries " "for one or more of the objective functions."
        raise NimbusError(msg)

    init_solver = solver if solver is not None else guess_best_solver(problem)
    _solver_options = solver_options if solver_options is not None else None

    # derive the classifications based on the reference point and and previous
    # objective function values
    classifications = infer_classifications(problem, current_objectives, reference_point)

    # TODO(gialmisi): this info should come from the problem
    is_smooth = True

    solutions = []

    # solve the nimbus scalarization problem, this is done always
    add_nimbus_sf = add_nimbus_sf_diff if is_smooth else add_nimbus_sf_nondiff

    problem_w_nimbus, nimbus_target = add_nimbus_sf(
        problem, "nimbus_sf", classifications, current_objectives, **(scalarization_options or {})
    )

    if _solver_options:
        nimbus_solver = init_solver(problem_w_nimbus, _solver_options)
    else:
        nimbus_solver = init_solver(problem_w_nimbus)

    solutions.append(nimbus_solver.solve(nimbus_target))

    if num_desired > 1:
        # solve STOM
        add_stom_sf = add_stom_sf_diff if is_smooth else add_stom_sf_nondiff

        problem_w_stom, stom_target = add_stom_sf(problem, "stom_sf", reference_point, **(scalarization_options or {}))
        if _solver_options:
            stom_solver = init_solver(problem_w_stom, _solver_options)
        else:
            stom_solver = init_solver(problem_w_stom)

        solutions.append(stom_solver.solve(stom_target))

    if num_desired > 2:  # noqa: PLR2004
        # solve ASF
        add_asf = add_asf_diff if is_smooth else add_asf_nondiff

        problem_w_asf, asf_target = add_asf(problem, "asf", reference_point, **(scalarization_options or {}))

        if _solver_options:
            asf_solver = init_solver(problem_w_asf, _solver_options)
        else:
            asf_solver = init_solver(problem_w_asf)

        solutions.append(asf_solver.solve(asf_target))

    if num_desired > 3:  # noqa: PLR2004
        # solve GUESS
        add_guess_sf = add_guess_sf_diff if is_smooth else add_guess_sf_nondiff

        problem_w_guess, guess_target = add_guess_sf(
            problem, "guess_sf", reference_point, **(scalarization_options or {})
        )

        if _solver_options:
            guess_solver = init_solver(problem_w_guess, _solver_options)
        else:
            guess_solver = init_solver(problem_w_guess)

        solutions.append(guess_solver.solve(guess_target))

    return solutions


def generate_starting_point(
    problem: Problem,
    reference_point: dict[str, float] | None = None,
    scalarization_options: dict | None = None,
    create_solver: CreateSolverType | None = None,
    solver_options: SolverOptions | None = None,
) -> SolverResults:
    r"""Generates a starting point for the NIMBUS method.

    Using the given reference point and achievement scalarizing function, finds one pareto
    optimal solution that can be used as a starting point for the NIMBUS method.
    If no reference point is given, ideal is used as the reference point.

    Instead of using this function, the user can provide a starting point.

    Raises:
        NimbusError: the given problem has an undefined ideal or nadir point, or both.

    Args:
        problem (Problem): the problem being solved.
        reference_point (dict[str, float]|None): an objective dictionary with a reference point.
            If not given, ideal will be used as reference point.
        scalarization_options (dict | None, optional): optional kwargs passed to the scalarization function.
            Defaults to None.
        create_solver (CreateSolverType | None, optional): a function that given a problem, will return a solver.
            If not given, an appropriate solver will be automatically determined based on the features of `problem`.
            Defaults to None.
        solver_options (SolverOptions | None, optional): optional options passed
            to the `create_solver` routine. Ignored if `create_solver` is `None`.
            Defaults to None.

    Returns:
        list[SolverResults]: a list of `SolverResults` objects. Contains as many elements
            as defined in `num_desired`.
    """
    ideal = problem.get_ideal_point()
    nadir = problem.get_nadir_point()
    if None in ideal or None in nadir:
        msg = "The given problem must have both an ideal and nadir point defined."
        raise NimbusError(msg)

    if reference_point is None:
        reference_point = {}
    for obj in problem.objectives:
        if obj.symbol not in reference_point:
            reference_point[obj.symbol] = ideal[obj.symbol]

    init_solver = create_solver if create_solver is not None else guess_best_solver(problem)
    _solver_options = solver_options if solver_options is not None else None

    # TODO(gialmisi): this info should come from the problem
    is_smooth = True

    # solve ASF
    add_asf = add_asf_diff if is_smooth else add_asf_nondiff

    problem_w_asf, asf_target = add_asf(problem, "asf", reference_point, **(scalarization_options or {}))
    if _solver_options:
        asf_solver = init_solver(problem_w_asf, _solver_options)
    else:
        asf_solver = init_solver(problem_w_asf)

    return asf_solver.solve(asf_target)
