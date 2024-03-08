"""Defines templates for scalarization functions and utilities to handle the templates.

Note that when scalarization functions are defined, they must add the post-fix
'_min' to any symbol representing objective functions so that the maximization
or minimization of the corresponding objective functions may be correctly
accounted for when computing scalarization function values.
"""

import json
import pprint

from desdeo.problem import (
    Constraint,
    ConstraintTypeEnum,
    InfixExpressionParser,
    Problem,
    ScalarizationFunction,
)


class ScalarizationError(Exception):
    """Raised when issues with creating or adding scalarization functions are encountered."""


class Op:
    """Defines the supported operators in the MathJSON format."""

    # TODO: move this to problem/schema.py, make it use this, and import it here from there
    # Basic arithmetic operators
    NEGATE = "Negate"
    ADD = "Add"
    SUB = "Subtract"
    MUL = "Multiply"
    DIV = "Divide"

    # Exponentation and logarithms
    EXP = "Exp"
    LN = "Ln"
    LB = "Lb"
    LG = "Lg"
    LOP = "LogOnePlus"
    SQRT = "Sqrt"
    SQUARE = "Square"
    POW = "Power"

    # Rounding operators
    ABS = "Abs"
    CEIL = "Ceil"
    FLOOR = "Floor"

    # Trigonometric operations
    ARCCOS = "Arccos"
    ARCCOSH = "Arccosh"
    ARCSIN = "Arcsin"
    ARCSINH = "Arcsinh"
    ARCTAN = "Arctan"
    ARCTANH = "Arctanh"
    COS = "Cos"
    COSH = "Cosh"
    SIN = "Sin"
    SINH = "Sinh"
    TAN = "Tan"
    TANH = "Tanh"

    # Comparison operators
    EQUAL = "Equal"
    GREATER = "Greater"
    GREATER_EQUAL = "GreaterEqual"
    LESS = "Less"
    LESS_EQUAL = "LessEqual"
    NOT_EQUAL = "NotEqual"

    # Other operators
    MAX = "Max"
    RATIONAL = "Rational"


def get_corrected_ideal_and_nadir(problem: Problem) -> tuple[dict[str, float | None], dict[str, float | None] | None]:
    """Compute the corrected ideal and nadir points depending if an objective function is to be maximized or not.

    I.e., the ideal and nadir point element for objectives to be maximized will be multiplied by -1.

    Args:
        problem (Problem): the problem with the ideal and nadir points.

    Returns:
        tuple[list[float], list[float]]: a list with the corrected ideal point
            and a list with the corrected nadir point. Will return None for missing
            elements.
    """
    ideal_point = {
        objective.symbol: objective.ideal if not objective.maximize else -objective.ideal
        for objective in problem.objectives
    }
    nadir_point = {
        objective.symbol: objective.nadir if not objective.maximize else -objective.nadir
        for objective in problem.objectives
    }

    return ideal_point, nadir_point


def add_asf_nondiff(
    problem: Problem,
    symbol: str,
    reference_point: dict[str, float],
    reference_in_aug=False,
    delta: float = 0.000001,
    rho: float = 0.000001,
) -> tuple[Problem, str]:
    """Creates and add the achievement scalarizing function for the given problem and reference point.

    This is the non-differentiable variant of the achievement scalarizing function, which
    means the resulting scalarization function is non-differentiable.
    Requires that the ideal and nadir point have been defined for the problem.

    Args:
        problem (Problem): the problem to which the scalarization function should be added.
        symbol (str): the symbol to reference the added scalarization function.
        reference_point (dict[str, float]): a reference point as an objective dict.
        reference_in_aug (bool): whether the reference point should be used in
            the augmentation term as well. Defaults to False.
        delta (float, optional): the scalar value used to define the utopian point (ideal - delta).
            Defaults to 0.000001.
        rho (float, optional): the weight factor used in the augmentation term. Defaults to 0.000001.

    Raises:
        ScalarizationError: there are missing elements in the reference point, or if any of the ideal or nadir
            point values are undefined (None).

    Returns:
        tuple[Problem, str]: A tuple containing a copy of the problem with the scalarization function added,
            and the symbol of the added scalarization function.
    """
    # check that the reference point has all the objective components
    if not all(obj.symbol in reference_point for obj in problem.objectives):
        msg = f"The given reference point {reference_point} does not have a component defined for all the objectives."
        raise ScalarizationError(msg)

    # check if minimizing or maximizing and adjust ideal and nadir values correspondingly
    ideal_point, nadir_point = get_corrected_ideal_and_nadir(problem)

    if any(value is None for value in ideal_point.values()) or any(value is None for value in nadir_point.values()):
        msg = f"There are undefined values in either the ideal ({ideal_point}) or the nadir point ({nadir_point})."
        raise ScalarizationError(msg)

    # Build the max term
    max_operands = [
        (
            f"({obj.symbol}_min - {reference_point[obj.symbol]}{" * -1" if obj.maximize else ''}) "
            f"/ ({nadir_point[obj.symbol]} - ({ideal_point[obj.symbol]} - {delta}))"
        )
        for obj in problem.objectives
    ]
    max_term = f"{Op.MAX}({', '.join(max_operands)})"

    # Build the augmentation term
    if not reference_in_aug:
        aug_operands = [
            f"{obj.symbol}_min / ({nadir_point[obj.symbol]} - ({ideal_point[obj.symbol]} - {delta}))"
            for obj in problem.objectives
        ]
    else:
        aug_operands = [
            (
                f"({obj.symbol}_min - {reference_point[obj.symbol]}{" * -1" if obj.maximize else 1}) "
                f"/ ({nadir_point[obj.symbol]} - ({ideal_point[obj.symbol]} - {delta}))"
            )
            for obj in problem.objectives
        ]

    aug_term = " + ".join(aug_operands)

    asf_function = f"{max_term} + {rho} * ({aug_term})"

    # Add the function to the problem
    scalarization_function = ScalarizationFunction(
        name="Achievement scalarizing function",
        symbol=symbol,
        func=asf_function,
    )
    return problem.add_scalarization(scalarization_function), symbol


def add_asf_generic_nondiff(
    problem: Problem,
    symbol: str,
    reference_point: dict[str, float],
    weights: dict[str, float],
    reference_in_aug=False,
    rho: float = 0.000001,
) -> tuple[Problem, str]:
    """Creates the generic achievement scalarizing function for the given problem and reference point, and weights.

    This is the non-differentiable variant of the generic achievement scalarizing function, which
    means the resulting scalarization function is non-differentiable.
    Requires that the ideal and nadir point have been defined for the problem.

    Args:
        problem (Problem): the problem to which the scalarization function should be added.
        symbol (str): the symbol to reference the added scalarization function.
        reference_point (dict[str, float]): a reference point with as many components as there are objectives.
        weights (dict[str, float]): the weights to be used in the scalarization function. must be positive.
        reference_in_aug (bool, optional): Whether the reference point should be used in the augmentation term.
            Defaults to False.
        rho (float, optional): the weight factor used in the augmentation term. Defaults to 0.000001.

    Raises:
        ScalarizationError: If either the reference point or the weights given are missing any of the objective
            components.
        ScalarizationError: If any of the ideal or nadir point values are undefined (None).

    Returns:
        tuple[Problem, str]: A tuple containing a copy of the problem with the scalarization function added,
            and the symbol of the added scalarization function.
    """
    # check that the reference point has all the objective components
    if not all(obj.symbol in reference_point for obj in problem.objectives):
        msg = f"The given reference point {reference_point} does not have a component defined for all the objectives."
        raise ScalarizationError(msg)

    # check that the weights have all the objective components
    if not all(obj.symbol in weights for obj in problem.objectives):
        msg = f"The given weight vector {weights} does not have a component defined for all the objectives."
        raise ScalarizationError(msg)

    # check if minimizing or maximizing and adjust ideal and nadir values correspondingly
    ideal_point, nadir_point = get_corrected_ideal_and_nadir(problem)

    if any(value is None for value in ideal_point.values()) or any(value is None for value in nadir_point.values()):
        msg = f"There are undefined values in either the ideal ({ideal_point}) or the nadir point ({nadir_point})."
        raise ScalarizationError(msg)

    # Build the max term
    max_operands = [
        (f"({obj.symbol}_min - {reference_point[obj.symbol]} * {-1 if obj.maximize else 1}) / ({weights[obj.symbol]})")
        for obj in problem.objectives
    ]
    max_term = f"{Op.MAX}({', '.join(max_operands)})"

    # Build the augmentation term
    if not reference_in_aug:
        aug_operands = [f"{obj.symbol}_min / ({weights[obj.symbol]})" for obj in problem.objectives]
    else:
        aug_operands = [
            (
                f"({obj.symbol}_min - {reference_point[obj.symbol]}) * {-1 if obj.maximize else 1} / "
                f"({weights[obj.symbol]})"
            )
            for obj in problem.objectives
        ]

    aug_term = " + ".join(aug_operands)

    # Collect the terms
    sf = f"{max_term} + {rho} * ({aug_term})"

    # Add the function to the problem
    scalarization_function = ScalarizationFunction(
        name="Generic achievement scalarizing function",
        symbol=symbol,
        func=sf,
    )
    return problem.add_scalarization(scalarization_function), symbol


def add_weighted_sums(problem: Problem, symbol: str, weights: dict[str, float]) -> tuple[Problem, str]:
    """Add the weighted sums scalarization to a problem.

    It is assumed that the weights add to 1.

    Warning:
        The weighted sums scalarization is often not capable of finding most Pareto optimal
            solutions when optimized. It is advised to utilize some better scalarization
            functions.

    Args:
        problem (Problem): the problem to which the scalarization should be added.
        symbol (str): the symbol to reference the added scalarization function.
        weights (dict[str, float]): the weights. For the method to work, the weights
            should sum to 1. However, this is not a condition that is checked.

    Raises:
        ScalarizationError: if the weights are missing any of the objective components.

    Returns:
        tuple[Problem, str]: A tuple containing a copy of the problem with the scalarization function added,
            and the symbol of the added scalarization function.
    """
    # check that the weights have all the objective components
    if not all(obj.symbol in weights for obj in problem.objectives):
        msg = f"The given weight vector {weights} does not have a component defined for all the objectives."
        raise ScalarizationError(msg)

    # Build the sum
    sum_terms = [f"({weights[obj.symbol]} * {obj.symbol}_min)" for obj in problem.objectives]

    # aggregate the terms
    sf = " + ".join(sum_terms)

    # Add the function to the problem
    scalarization_function = ScalarizationFunction(
        name="Weighted sums scalarization function",
        symbol=symbol,
        func=sf,
    )
    return problem.add_scalarization(scalarization_function), symbol


def add_objective_as_scalarization(problem: Problem, symbol: str, objective_symbol: str) -> tuple[Problem, str]:
    """Creates a scalarization where one of the problem's objective functions is optimized.

    Args:
        problem (Problem): the problem to which the scalarization should be added.
        symbol (str): the symbol to reference the added scalarization function.
        objective_symbol (str): the symbol of the objective function to be optimized.

    Raises:
        ScalarizationError: the given objective_symbol does not exist in the problem.

    Returns:
        tuple[Problem, str]: A tuple containing a copy of the problem with the scalarization function added,
            and the symbol of the added scalarization function.
    """
    # check that symbol exists
    if objective_symbol not in (correct_symbols := [objective.symbol for objective in problem.objectives]):
        msg = f"The given objective symbol {objective_symbol} should be one of {correct_symbols}."
        raise ScalarizationError(msg)

    sf = ["Multiply", 1, f"{objective_symbol}_min"]

    # Add the function to the problem
    scalarization_function = ScalarizationFunction(
        name=f"Objective {objective_symbol}",
        symbol=symbol,
        func=sf,
    )
    return problem.add_scalarization(scalarization_function), symbol


def add_epsilon_constraints(
    problem: Problem, symbol: str, constraint_symbols: dict[str, str], objective_symbol: str, epsilons: dict[str, float]
) -> tuple[Problem, str, list[str]]:
    """Creates expressions for an epsilon constraints scalarization and constraints.

    It is assumed that epsilon have been given in a format where each objective is to be minimized.

    Args:
        problem (Problem): the problem to scalarize.
        symbol (str): the symbol of the added objective function to be optimized.
        constraint_symbols (dict[str, str]): a dict with the symbols to be used with the added
            constraints. The key indicates the name of the objective function the constraint
            is related to, and the value is the symbol to be used when defining the constraint.
        objective_symbol (str): the objective used as the objective in the epsilon constraint scalarization.
        epsilons (dict[str, float]): the epsilon constraint values in a dict
            with each key being an objective's symbol. The corresponding value
            is then used as the epsilon value for the respective objective function.

    Raises:
        ScalarizationError: `objective_symbol` not found in problem definition.

    Returns:
        tuple[Problem, str, list[str]]: A triple with the first element being a copy of the
            problem with the added epsilon constraints. The second element is the symbol of
            the objective to be optimized. The last element is a list with the symbols
            of the added constraints to the problem.
    """
    if objective_symbol not in (correct_symbols := [objective.symbol for objective in problem.objectives]):
        msg = f"The given objective symbol {objective_symbol} should be one of {correct_symbols}."
        raise ScalarizationError(msg)

    _problem, _ = add_objective_as_scalarization(problem, symbol, objective_symbol)

    # the epsilons must be given such that each objective function is to be minimized
    # TODO: check if objective function is linear
    constraints = [
        Constraint(
            name=f"Epsilon for {obj.symbol}",
            symbol=constraint_symbols[obj.symbol],
            func=["Add", f"{obj.symbol}_min", ["Negate", epsilons[obj.symbol]]],
            cons_type=ConstraintTypeEnum.LTE,
        )
        for obj in problem.objectives
        if obj.symbol != objective_symbol
    ]

    _problem = _problem.add_constraints(constraints)

    return _problem, symbol, [con.symbol for con in constraints]


def create_epsilon_constraints_json(
    problem: Problem, objective_symbol: str, epsilons: dict[str, float]
) -> tuple[list[str | int | float], list[str]]:
    """Creates JSON expressions for an epsilon constraints scalarization and constraints.

    It is assumed that epsilon have been given in a format where each objective is to be minimized.

    Args:
        problem (Problem): the problem to scalarize.
        objective_symbol (str): the objective used as the objective in the epsilon constraint scalarization.
        epsilons (dict[str, float]): the epsilon constraint values in a dict
            with each key being an objective's symbol.

    Raises:
        ScalarizationError: `objective_symbol` not found in problem definition.

    Returns:
        tuple[list, list]: the first element is the expression of the scalarized objective expressed in MathJSON format.
        The second element is a list of expressions of the constraints expressed in MathJSON format.
            The constraints are in less than or equal format.
    """
    correct_symbols = [objective.symbol for objective in problem.objectives]
    if objective_symbol not in correct_symbols:
        msg = f"The given objective symbol {objective_symbol} should be one of {correct_symbols}."
        raise ScalarizationError(msg)
    correct_symbols.remove(objective_symbol)

    scalarization_expr = ["Multiply", 1, f"{objective_symbol}_min"]

    # the epsilons must be given such that each objective function is to be minimized
    constraint_exprs = [["Add", f"{obj}_min", ["Negate", epsilons[obj]]] for obj in correct_symbols]

    return scalarization_expr, constraint_exprs


def add_scalarization_function(
    problem: Problem,
    func: str,
    symbol: str,
    name: str | None = None,
) -> tuple[Problem, str]:
    """Adds a scalarization function to a Problem.

    Returns a new instance of the Problem with the new scalarization function
    and the symbol of the scalarization function added.

    Args:
        problem (Problem): the problem to which the scalarization function should be added.
        func (str): the scalarization function to be added as a string in infix notation.
        symbol (str): the symbol reference the added scalarization function.
            This is important when the added scalarization function should be
            utilized when optimizing a problem.
        name (str, optional): the name to be given to the scalarization
            function. If None, the symbol is used as the name. Defaults to None.

    Returns:
        tuple[Problem, str]: A tuple with the new Problem with the added
            scalarization function and the function's symbol.
    """
    scalarization_function = ScalarizationFunction(
        name=symbol if name is None else name,
        symbol=symbol,
        func=func,
    )
    return problem.add_scalarization(scalarization_function), symbol


def add_lte_constraints(
    problem: Problem, funcs: list[str], symbols: list[str], names: list[str | None] | None = None
) -> Problem:
    """Adds constraints to a problem that are defined in the less than or equal format.

    Is is assumed that the constraints expression at position funcs[i] is symbolized by the
    symbol at position symbols[i] for all i.

    Does not modify problem, but makes a copy of it instead and returns it.

    Args:
        problem (Problem): the problem to which the constraints are added.
        funcs (list[str]): the expressions of the constraints.
        symbols (list[str]): the symbols of the constraints. In order.
        names (list[str  |  None] | None, optional): The names of the
            constraints. For any name with 'None' the symbol is used as the name. If
            names is None, then the symbol is used as the name for all the
            constraints. Defaults to None.

    Raises:
        ScalarizationError: if the lengths of the arguments do not match.

    Returns:
        Problem: a copy of the original problem with the constraints added.
    """
    if (len_f := len(funcs)) != (len_s := len(symbols)) and names is not None and (len_n := len(names)) != len(funcs):
        msg = (
            f"The lengths of ({len_f=}) and 'symbols' ({len_s=}) must match. "
            f"If 'names' is not None, then its length ({len_n=}) must also match."
        )
        raise ScalarizationError(msg)

    if names is None:
        names = symbols

    return problem.model_copy(
        update={
            "constraints": [
                *(problem.constraints if problem.constraints is not None else []),
                *[
                    Constraint(
                        name=(name if (name := names[i]) is not None else symbols[i]),
                        symbol=symbols[i],
                        cons_type=ConstraintTypeEnum.LTE,
                        func=funcs[i],
                    )
                    for i in range(len(funcs))
                ],
            ]
        }
    )
