"""Defines solver interfaces for gurobipy."""

import contextlib
import io
import logging
import sys
from typing import Any, Self

import gurobipy as gp
from pydantic import BaseModel, ConfigDict, Field, model_validator

from desdeo.problem import (
    Constraint,
    GurobipyEvaluator,
    Objective,
    Problem,
    ScalarizationFunction,
    TensorVariable,
    Variable,
)
from desdeo.tools.generics import BaseSolver, PersistentSolver, SolverResults

logger = logging.getLogger(__name__)

_GUROBI_PARAMETER_NAMES = {name.lower(): name for name in dir(gp.GRB.Param) if not name.startswith("_")}
"""Gurobi's parameter names by their lowercased name (Gurobi matches them case-insensitively)."""


class GurobipyOptions(BaseModel):
    """Defines a pydantic model to store and pass Gurobi parameters to the gurobipy solvers.

    Fields are named by Gurobi's parameter names. Any other Gurobi parameter can be given by its Gurobi name
    and is passed on as is; Gurobi validates the values. Names are matched case-insensitively, as Gurobi does,
    and normalised to the field's or Gurobi's spelling. An explicit `None` for an optional field means Gurobi's
    default, also overriding a stored value. For available parameters see
    https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html
    """

    model_config = ConfigDict(extra="allow")

    OutputFlag: int = Field(description="Whether Gurobi logs its output.", default=0)
    LogToConsole: int = Field(description="Whether Gurobi logs to the console.", default=0)
    IntegralityFocus: int = Field(
        description=(
            "Whether Gurobi keeps MIP solutions feasible after rounding binaries, so that big-M "
            "formulations cannot pass flow through nearly-zero binaries."
        ),
        default=1,
    )
    FeasibilityTol: float | None = Field(
        description="Primal feasibility tolerance. None leaves Gurobi's default.", default=None
    )
    OptimalityTol: float | None = Field(
        description="Dual feasibility tolerance. None leaves Gurobi's default.", default=None
    )
    TimeLimit: float | None = Field(description="Time limit in seconds. None leaves Gurobi's default.", default=None)
    MIPGap: float | None = Field(description="Relative MIP optimality gap. None leaves Gurobi's default.", default=None)
    Threads: int | None = Field(
        description="Number of threads, 0 letting Gurobi choose. None leaves Gurobi's default.", default=None
    )
    NumericFocus: int | None = Field(
        description="Numerical caution; higher values make Gurobi more careful. None leaves Gurobi's default.",
        default=None,
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_parameter_names(cls, data: Any) -> Any:
        """Rename keys to the declared field's or Gurobi's spelling; of case-variants, keep the one in that spelling."""
        if not isinstance(data, dict):
            return data
        declared = {name.lower(): name for name in cls.model_fields}
        normalized = {}
        for key, value in data.items():
            if not isinstance(key, str):
                normalized[key] = value
                continue
            name = declared.get(key.lower()) or _GUROBI_PARAMETER_NAMES.get(key.lower(), key)
            if name in normalized:
                logger.warning(
                    "Gurobi parameter %r is given in several spellings (including %r); the spelling %r is kept if "
                    "given, else the first.",
                    name,
                    key,
                    name,
                )
                if key != name:
                    continue
            normalized[name] = value
        return normalized

    @model_validator(mode="after")
    def _warn_unknown_parameters(self) -> Self:
        """Warn about extra parameters the installed Gurobi doesn't know; they are kept."""
        unknown = sorted(key for key in self.model_extra if key.lower() not in _GUROBI_PARAMETER_NAMES)
        if unknown:
            logger.warning(
                "Unknown Gurobi parameters %s are kept; Gurobi rejects them when the solver is constructed.", unknown
            )
        return self


def _set_gurobi_parameters(model: gp.Model, options: GurobipyOptions | dict[str, Any] | None) -> None:
    """Set the non-None fields and the extras of `options` (a dict is validated first) on `model`."""
    if not isinstance(options, GurobipyOptions):
        options = GurobipyOptions.model_validate(options or {})
    for key, value in options.model_dump(exclude_none=True).items():
        model.setParam(key, value)


def _constraint_contains_objective(constraint: Constraint, objective_symbol: str) -> bool:
    """Check if a constraint's function contains a specific objective symbol."""
    func_str = str(constraint.func)
    return objective_symbol in func_str


def parse_gurobipy_optimizer_results(problem: Problem, evaluator: GurobipyEvaluator) -> SolverResults:
    """Parses results from GurobipyEvaluator's model into DESDEO SolverResults.

    Args:
        problem (Problem): the problem being solved.
        evaluator (GurobipyEvaluator): the evaluator utilized to solve the problem.

    Returns:
        SolverResults: DESDEO solver results.
    """
    model_status = evaluator.model.status
    success = model_status == gp.GRB.OPTIMAL
    if model_status == gp.GRB.OPTIMAL:
        status = "Optimal solution found."
    elif model_status == gp.GRB.INFEASIBLE:
        status = "Model is infeasible."
    elif model_status == gp.GRB.UNBOUNDED:
        status = "Model is unbounded."
    elif model_status == gp.GRB.INF_OR_UNBD:
        status = "Model is either infeasible or unbounded."
    elif model_status == gp.GRB.TIME_LIMIT:
        status = "Time limit reached."
    elif model_status == gp.GRB.SOLUTION_LIMIT:
        status = "Solution limit reached."
    else:
        status = f"Optimization ended with status: {model_status}"
    msg = f"Gurobipy solver status is: '{status}'"

    # An incumbent is returned unless the model is infeasible or unbounded.
    has_incumbent = evaluator.model.SolCount > 0 and model_status not in (
        gp.GRB.INFEASIBLE,
        gp.GRB.UNBOUNDED,
        gp.GRB.INF_OR_UNBD,
    )
    if not success and has_incumbent:
        msg += " A feasible solution was found, but it is not proven optimal."
        if evaluator.model.IsMIP:
            with contextlib.suppress(AttributeError):
                msg += f" MIP gap: {evaluator.model.MIPGap:.4g}."

    if not success and not has_incumbent:
        return SolverResults(
            optimal_variables={var.symbol: float("nan") for var in problem.variables},
            optimal_objectives={obj.symbol: float("nan") for obj in problem.objectives},
            constraint_values=None,
            extra_func_values=None,
            scalarization_values=None,
            lagrange_multipliers=None,
            success=False,
            message=msg,
        )

    results = evaluator.get_values()

    variable_values = {var.symbol: results[var.symbol] for var in problem.variables}
    objective_values = {obj.symbol: results[obj.symbol] for obj in problem.objectives}
    constraint_values = (
        {con.symbol: results[con.symbol] for con in problem.constraints} if problem.constraints is not None else None
    )
    extra_func_values = (
        {extra.symbol: results[extra.symbol] for extra in problem.extra_funcs}
        if problem.extra_funcs is not None
        else None
    )
    scalarization_values = (
        {scal.symbol: results[scal.symbol] for scal in problem.scalarization_funcs}
        if problem.scalarization_funcs is not None
        else None
    )
    lagrange_multipliers = None
    if problem.constraints is not None:
        objective_symbols = set(objective_values.keys())
        lagrange_multipliers = {
            con.symbol: results[con.symbol]
            for con in problem.constraints
            if any(obj_sym in con.symbol for obj_sym in objective_symbols)
        }

    return SolverResults(
        optimal_variables=variable_values,
        optimal_objectives=objective_values,
        constraint_values=constraint_values,
        extra_func_values=extra_func_values,
        scalarization_values=scalarization_values,
        lagrange_multipliers=lagrange_multipliers,
        success=success,
        message=msg,
    )


class GurobipySolver(BaseSolver):
    """Creates a gurobipy solver that utilizes gurobi's own Python implementation."""

    def __init__(self, problem: Problem, options: GurobipyOptions | dict[str, Any] | None = None):
        """The solver is initialized by supplying a problem and options.

        Unlike with Pyomo you do not need to have gurobi installed on your system
        for this to work. Suitable for solving mixed-integer linear and quadratic optimization
        problems.

        Args:
            problem (Problem): the problem to be solved.
            options (GurobipyOptions | dict[str, Any] | None): Gurobi parameters to set. A dict is validated
                as `GurobipyOptions`. If `None`, the defaults of `GurobipyOptions` are used.
        """
        self.evaluator = GurobipyEvaluator(problem)
        self.problem = problem
        _set_gurobi_parameters(self.evaluator.model, options)

    def solve(self, target: str) -> SolverResults:
        """Solve the problem for the given target.

        Args:
            target (str): the symbol of the function to be optimized, and which is
                defined in the problem given when initializing the solver.

        Returns:
            SolverResults: the results of the optimization.
        """
        self.evaluator.set_optimization_target(target)
        self.evaluator.model.optimize()
        return parse_gurobipy_optimizer_results(self.problem, self.evaluator)


class PersistentGurobipySolver(PersistentSolver):
    """A persistent solver class utlizing gurobipy.

    Use this instead of create_gurobipy_solver when re-initializing the
    solver every time the problem is changed is not practical.
    """

    evaluator: GurobipyEvaluator

    def __init__(self, problem: Problem, options: GurobipyOptions | dict[str, Any] | None = None):
        """Initializer for the persistent solver.

        Args:
            problem (Problem): the problem to be transformed in a GurobipyModel.
            options (GurobipyOptions | dict[str, Any] | None): Gurobi parameters to set. A dict is validated
                as `GurobipyOptions`. If `None`, the defaults of `GurobipyOptions` are used.
        """
        self.problem = problem
        self.evaluator = GurobipyEvaluator(problem)
        _set_gurobi_parameters(self.evaluator.model, options)

    def add_constraint(self, constraint: Constraint | list[Constraint]) -> gp.Constr | list[gp.Constr]:
        """Add one or more constraint expressions to the solver.

        If adding a lot of constraints or dealing with a large model, this function
        may end up being very slow compared to adding the constraints to the model
        stored in the evaluator directly.

        Args:
            constraint (Constraint): the constraint function expression or a list of
                constraint function expressions.

        Raises:
            GurobipyEvaluatorError: when an unsupported constraint type is encountered.

        Returns:
            gurobipy.Constr: The gurobipy constraint that was added or a list of gurobipy
                constraints if the constraint argument was a list.
        """
        if isinstance(constraint, list):
            cons_list = list[gp.Constr]
            for cons in constraint:
                cons_list.append(self.evaluator.add_constraint(cons))
            return cons_list

        return self.evaluator.add_constraint(constraint)

    def add_objective(self, objective: Objective | list[Objective]):
        """Adds an objective function expression to the solver.

        Does not yet add any actual gurobipy optimization objectives, only adds them to the dict
        containing the expressions of the objectives. The objective expressions are stored in the
        evaluator and the evaluator must add the appropiate gurobipy objective before solving.

        Args:
            objective (Objective): an objective function expression or a list of objective function
                expressions to be added.
        """
        if not isinstance(objective, list):
            objective = [objective]

        for obj in objective:
            self.evaluator.add_objective(obj)

    def add_scalarization_function(self, scalarization: ScalarizationFunction | list[ScalarizationFunction]):
        """Adds a scalrization expression to the solver.

        Scalarizations work identically to objectives, except they are stored in a different
        dict in the evaluator. If you want to solve the problem using a scalarization, the
        evaluator needs to set it as an optimization target first.

        Args:
            scalarization (ScalarizationFunction): A scalarization function or a list of
                scalarization functions to be added.
        """
        if not isinstance(scalarization, list):
            scalarization = [scalarization]

        for scal in scalarization:
            self.evaluator.add_scalarization_function(scal)

    def add_variable(
        self, variable: Variable | TensorVariable | list[Variable] | list[TensorVariable]
    ) -> gp.Var | gp.MVar | list[gp.Var] | list[gp.MVar]:
        """Add one or more variables to the solver.

        If adding a lot of variables or dealing with a large model, this function
        may end up being very slow compared to adding the variables to the model
        stored in the evaluator directly.

        Args:
            variable (Variable): The definition of the variable or a list of variables to be added.

        Raises:
            GurobipyEvaluatorError: when a problem in extracting the variables is encountered.
                I.e., the variables are of a non supported type.

        Returns:
            gp.Var: the variable that was added to the model or a list of variables if
                variable argument was a list.
        """
        if isinstance(variable, list):
            var_list = list[gp.Var | gp.MVar]
            for var in variable:
                var_list.append(self.evaluator.add_variable(var))
            return var_list

        return self.evaluator.add_variable(variable)

    def remove_constraint(self, symbol: str | list[str]):
        """Removes a constraint from the solver.

        If removing a lot of constraints or dealing with a very large model this function
        may be slow because of the model.update() calls. Accessing the model stored in the
        evaluator directly may be faster.

        Args:
            symbol (str): a str representing the symbol of the constraint to be removed.
                Can also be a list of multiple symbols.
        """
        if not isinstance(symbol, list):
            symbol = [symbol]
        for s in symbol:
            self.evaluator.remove_constraint(s)

    def remove_variable(self, symbol: str | list[str]):
        """Removes a variable from the model.

        If removing a lot of variables or dealing with a very large model this function
        may be slow because of the model.update() calls. Accessing the model stored in
        the evaluator directly may be faster.

        Args:
            symbol (str): a str representing the symbol of the variable to be removed.
                Can also be a list of multiple symbols.
        """
        self.evaluator.remove_variable(symbol)

    def solve(self, target: str) -> SolverResults:
        """Solves the current problem with the specified target.

        Args:
            target (str): a str representing the symbol of the target function.

        Returns:
            SolverResults: The results of the solver
        """
        self.evaluator.set_optimization_target(target)
        self.evaluator.model.optimize()
        return parse_gurobipy_optimizer_results(self.problem, self.evaluator)


def check_gurobi_license():
    """Check if Gurobi is using a full license (not trial).

    Returns:
        (str | bool): True if using full academic/commercial license
            False if using trial license or no license found
    """
    captured_output = io.StringIO()
    original_stdout = sys.stdout

    try:
        sys.stdout = captured_output
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 1)
            env.start()
        sys.stdout = original_stdout

        output = captured_output.getvalue()
        return "Restricted license - for non-production use only" not in output  # noqa: TRY300

    except Exception:
        sys.stdout = original_stdout
        return False
