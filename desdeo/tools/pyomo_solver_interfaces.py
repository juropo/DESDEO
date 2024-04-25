"""Defines solver interfaces for pyomo."""

from collections.abc import Callable

import pyomo.environ as pyomo
from pydantic import BaseModel, Field
from pyomo.opt import SolverResults as _pyomo_SolverResults
from pyomo.opt import SolverStatus as _pyomo_SolverStatus
from pyomo.opt import TerminationCondition as _pyomo_TerminationCondition

from desdeo.problem import Problem, PyomoEvaluator
from desdeo.tools.generics import SolverResults


class BonminOptions(BaseModel):
    """Defines a pydantic model to store and pass options to the Bonmin solver.

    Because Bonmin utilizes many sub-solver, the options specific to Bonmin
    must be prefixed in their name with 'bonmin.{option_name}',
    e.g., `bonmin.integer_tolerance`. For a list of options, see
    https://www.coin-or.org/Bonmin/options_list.html

    Note:
        Not all options are available through this model.
        Please add options as they are needed and make a pull request.
    """

    tol: float = Field(description="Sets the convergence tolerance of ipopt. Defaults to 1e-8.", default=1e-8)
    """Sets the convergence tolerance of ipopt. Defaults to 1e-8."""

    bonmin_integer_tolerance: float = Field(
        description="Numbers within this value of an integer are considered integers. Defaults to 1e-6.", default=1e-6
    )
    """Numbers within this value of an integer are considered integers. Defaults to 1e-6."""

    bonmin_algorithm: str = Field(
        description=(
            "Presets some of the options in Bonmin based on the algorithm choice. Defaults to 'B-BB'. "
            "A good first option to try is 'B-Hyb'."
        ),
        default="B-BB",
    )
    """Presets some of the options in Bonmin based on the algorithm choice. Defaults to 'B-BB'.
    A good first option to try is 'B-Hyb'.
    """

    def asdict(self) -> dict[str, float]:
        """Converts the Pydantic model into a dict so that Bonmin specific options are in the correct format.

        This means that the attributes starting with `bonmin_optionname` will be
        converted to keys in the format `bonmin.optionname` in the returned dict.
        """
        output = {}
        for field_name, _ in self.__fields__.items():
            if (rest := field_name.split(sep="_"))[0] == "bonmin":
                # Convert to Bonmin specific format
                output[f"bonmin.{'_'.join(rest[1:])}"] = getattr(self, field_name)
            else:
                # Keep the field as is
                output[field_name] = getattr(self, field_name)

        return output


class IpoptOptions(BaseModel):
    """Defines a pydantic dataclass to pass options to the Ipopt solver.

    For more information and documentation on the options,
    see https://coin-or.github.io/Ipopt/

    Note:
        Not all options are available through this model.
        Please add options as they are needed and make a pull request.
    """

    tol: float = Field(description="The desired relative convergence tolerance. Defaults to 1e-8.", default=1e-8)
    """The desired relative convergence tolerance. Defaults to 1e-8."""

    max_iter: int = Field(description="Maximum number of iterations. Must be >1. Defaults to 3000.", default=3000)
    """Maximum number of iterations. Must be >1. Defaults to 3000."""


_default_bonmin_options = BonminOptions()
"""Defines Bonmin options with default values."""

_default_ipopt_options = IpoptOptions()
"""Defines Ipopt optins with default values."""


def parse_pyomo_optimizer_results(
    opt_res: _pyomo_SolverResults, problem: Problem, evaluator: PyomoEvaluator
) -> SolverResults:
    """Parses pyomo SolverResults into DESDEO SolverResults.

    Args:
        opt_res (SolverResults): the pyomo solver results.
        problem (Problem): the problem being solved.
        evaluator (PyomoEvaluator): the evaluator utilized to get the pyomo solver results.

    Returns:
        SolverResults: DESDEO solver results.
    """
    results = evaluator.get_values()

    variable_values = {var.symbol: results[var.symbol] for var in problem.variables}
    objective_values = {obj.symbol: results[obj.symbol] for obj in problem.objectives}
    constraint_values = (
        {con.symbol: results[con.symbol] for con in problem.constraints} if problem.constraints else None
    )
    success = (
        opt_res.solver.status == _pyomo_SolverStatus.ok
        and opt_res.solver.termination_condition == _pyomo_TerminationCondition.optimal
    )
    msg = (
        f"Pyomo solver status is: '{opt_res.solver.status}', with termination condition: "
        f"'{opt_res.solver.termination_condition}'."
    )

    return SolverResults(
        optimal_variables=variable_values,
        optimal_objectives=objective_values,
        constraint_values=constraint_values,
        success=success,
        message=msg,
    )


class PyomoBonminSolver:
    """Creates pyomo solvers that utilize bonmin."""

    def __init__(self, problem: Problem, options: BonminOptions | None = _default_bonmin_options):
        """The solver is initialized with a problem and solver options.

        Suitable for mixed-integer problems. The objective function being minimized
        (target) and the constraint functions must be twice continuously
        differentiable. When the objective functions and constraints are convex, the
        solution is exact. When the objective or any of the constraints, or both,
        are non-convex, then the solution is based on heuristics.

        For more info about bonmin, see: https://www.coin-or.org/Bonmin/

        Note:
            Bonmin must be installed on the system running DESDEO, and its executable
                must be defined in the PATH.

        Args:
            problem (Problem): the problem to be solved.
            options (BonminOptions, optional): options to be passed to the Bonmin solver.
                If `None` is passed, defaults to `_default_bonmin_options` defined in
                this source file. Defaults to `None`.
        """
        self.problem = problem
        self.evaluator = PyomoEvaluator(problem)

        if options is None:
            self.options = _default_bonmin_options
        else:
            self.options = options

    def solve(self, target: str) -> SolverResults:
        """Solve the problem for a given target.

        Args:
            target (str): the symbol of the objective function to be optimized.

        Returns:
            SolverResults: the results of the optimization.
        """
        self.evaluator.set_optimization_target(target)

        opt = pyomo.SolverFactory("bonmin", tee=True)

        # set solver options
        for key, value in self.options.asdict().items():
            opt.options[key] = value
        opt_res = opt.solve(self.evaluator.model)

        return parse_pyomo_optimizer_results(opt_res, self.problem, self.evaluator)


class PyomoIpoptSolver:
    """Create a pyomo solver that utilizes Ipopt."""

    def __init__(self, problem: Problem, options: IpoptOptions | None = _default_ipopt_options):
        """The solver is initialized with a problem and solver options.

        Suitable for non-linear, twice differentiable constrained problems.
        The problem may be convex or non-convex.

        For more information, see https://coin-or.github.io/Ipopt/

        Note:
            Ipopt must be installed on the system running DESDEO, and its executable
                must be defined in the PATH.

        Args:
            problem (Problem): the problem being solved.
            options (IpoptOptions, optional): options to be passed to the Ipopt solver.
                If `None` is passed, defaults to `_default_ipopt_options` defined in
                this source file. Defaults to `None`.
        """
        self.problem = problem
        self.evaluator = PyomoEvaluator(problem)

        if options is None:
            self.options = _default_ipopt_options
        else:
            self.options = options

    def solve(self, target: str) -> SolverResults:
        """Solve the problem for a given target.

        Args:
            target (str): the symbol of the objective function to be optimized.

        Returns:
            SolverResults: results of the Optimization.
        """
        self.evaluator.set_optimization_target(target)

        opt = pyomo.SolverFactory("ipopt", tee=True)
        opt_res = opt.solve(self.evaluator.model)
        return parse_pyomo_optimizer_results(opt_res, self.problem, self.evaluator)


class PyomoGurobiSolver:
    """Creates a pyomo solver that utilized Gurobi."""

    def __init__(self, problem: Problem, options: dict[str, any] | None = None):
        """Creates a pyomo solver that utilizes gurobi.

        You need to have gurobi installed on your system for this to work.

        Suitable for solving mixed-integer linear and quadratic optimization
        problems.

        Args:
            problem (Problem): the problem to be solved.
            options (GurobiOptions): Dictionary of Gurobi parameters to set.
                This is passed to pyomo as is, so it works the same as options
                would for calling pyomo SolverFactory directly.
                See https://www.gurobi.com/documentation/current/refman/parameters.html
                for information on the available options
        """
        self.problem = problem
        self.evaluator = PyomoEvaluator(problem)

        if options is None:
            self.options = {}
        else:
            self.options = options

    def solve(self, target: str) -> SolverResults:
        """Solve the problem for a given target.

        Args:
            target (str): the symbol of the objective function to be optimized.

        Returns:
            SolverResults: the results of the optimization.
        """
        self.evaluator.set_optimization_target(target)

        opt = pyomo.SolverFactory("gurobi", solver_io="python", options=self.options)

        with pyomo.SolverFactory("gurobi", solver_io="python") as opt:
            opt_res = opt.solve(self.evaluator.model)
            return parse_pyomo_optimizer_results(opt_res, self.problem, self.evaluator)
