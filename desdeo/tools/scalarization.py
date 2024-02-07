"""Defines templates for scalariztation functions and utilities to handle the templates.

Note that when scalarization functions are defined, they must add the post-fix
'_min' to any symbol representing objective functions so that the maximization
or minimization of the corresponding objective functions may be correctly
accounted for when computing scalarization function values.
"""

from jinja2 import Environment, FileSystemLoader, Template
import json, pprint

from desdeo.problem import InfixExpressionParser
from desdeo.problem import Problem, ScalarizationFunction
from desdeo.problem import binh_and_korn

# TODO: fix
TEMPLATE_PATH = "desdeo/tools/scalarization_templates"

env = Environment(loader=FileSystemLoader(TEMPLATE_PATH))


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


def create_asf(problem: Problem, reference_point: list[float], delta: float = 0.000001, rho: float = 0.000001) -> str:
    """Creates the achievement scalarizing function for the given problem and reference point.

    Args:
        problem (Problem): _description_
        reference_point (list[float]): _description_.
        delta (float, optional): _description_. Defaults to 0.000001.
        rho (float, optional): _description_. Defaults to 0.000001.

    Returns:
        str: The scalarization function for the problem and given reference point in infix format.
    """
    objective_symbols = [objective.symbol for objective in problem.objectives]
    ideal_point = [objective.ideal for objective in problem.objectives]
    nadir_point = [objective.nadir for objective in problem.objectives]

    # Build the max term
    max_operands = [
        f"({objective_symbols[i]}_min - {reference_point[i]}) / ({nadir_point[i]} - ({ideal_point[i]} - {delta}))"
        for i in range(len(problem.objectives))
    ]
    max_term = f"{Op.MAX}({', '.join(max_operands)})"

    # Build the augmentation term
    aug_operands = [
        f"{objective_symbols[i]}_min / ({nadir_point[i]} - ({ideal_point[i]} - {delta}))"
        for i in range(len(problem.objectives))
    ]
    aug_term = " + ".join(aug_operands)

    # Return the whole scalarization function
    return f"{max_term} + {rho} * ({aug_term})"


def add_scalarization_function(
    problem: Problem,
    func: str,
    symbol: str,
    name: str | None = None,
    description: str | None = None,
) -> tuple[Problem, str]:
    """Adds a scalarization function to a Problem.

    Returns a new instanse of the Problem with the new scalarization function
    and the symbol of the scalarization function added.

    Args:
        problem (Problem): the problem to which the scalarization function should be added.
        func (str): the scalarization function to be added as a string in infix notation.
        symbol (str): the symbol reference the added scalarization function.
            This is important when the added scalarization function should be
            utilized when optimizing a problem.
        name (str, optional): the name to be given to the scalarization
            function. If None, the symbol is used as the name. Defaults to None.
        description (str, optional): the description of the scalarization
            function. If None, the string representation of the scalarization
            function is utilized. Defaults to None.

    Returns:
        tuple[Problem, str]: A tuple with the new Problem with the added
            scalarization function and the function's symbol.
    """
    scalarization_function = ScalarizationFunction(
        description=func if description is None else description,
        name=symbol if name is None else name,
        symbol=symbol,
        func=func,
    )
    return problem.add_scalarization(scalarization_function), symbol


if __name__ == "__main__":
    problem = binh_and_korn()
    problem = problem.model_copy(
        update={
            "objectives": [
                objective.model_copy(update={"ideal": 0.5, "nadir": 5.5}) for objective in problem.objectives
            ]
        }
    )
    res = create_asf(problem, [5.0, 2.0])

    parser = InfixExpressionParser()
    print(f"Infix:\n\n{res}\n")
    dump = json.dumps(parser.parse(res), indent=2)
    print("JSON:\n")
    pprint.pprint(json.loads(dump))
