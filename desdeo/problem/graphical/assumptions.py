"""Hypothetical assumptions on a diagram: fixed states for decisions and parentless chance nodes.

A parentless chance node is assumed by replacing its prior with a point mass, which is exactly
conditioning on it. A decision is assumed by fixing its choice at every information state of the
compiled problem. Chance nodes with parents aren't supported, since conditioning on them would
also change their ancestors' distributions.
"""

from math import prod
from typing import Self

from pydantic import BaseModel, ConfigDict, Field

from desdeo.problem.schema import Constraint, ConstraintTypeEnum, Problem

from .compile import IndexMap, compile_diagram
from .network import DiscreteNetwork, Potential
from .rooted_junction_tree import RootedJunctionTree, build_rooted_junction_tree
from .schema import DecisionAnnotation


class DiagramAssumptions(BaseModel):
    """States assumed for some of a diagram's parentless chance nodes and decisions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chance: dict[str, str] = Field(default_factory=dict)
    """Parentless chance node name -> assumed state label."""
    decisions: dict[str, str] = Field(default_factory=dict)
    """Decision symbol -> fixed state label."""

    def is_empty(self) -> bool:
        """Whether nothing is assumed."""
        return not self.chance and not self.decisions

    def merged_with(self, other: "DiagramAssumptions") -> Self:
        """`self` plus `other`'s assumptions.

        Raises:
            ValueError: `other` assumes a different state for a node `self` already assumes.
        """
        for mine, theirs in ((self.chance, other.chance), (self.decisions, other.decisions)):
            conflicts = sorted(node for node in mine.keys() & theirs.keys() if mine[node] != theirs[node])
            if conflicts:
                msg = f"Already assumed differently: {', '.join(f'{node} = {mine[node]}' for node in conflicts)}."
                raise ValueError(msg)
        return type(self)(chance={**self.chance, **other.chance}, decisions={**self.decisions, **other.decisions})

    def label(self) -> str:
        """E.g. `"structure = soaking, drill = yes"`."""
        return ", ".join(f"{node} = {state}" for node, state in {**self.chance, **self.decisions}.items())


def validate_assumptions(
    network: DiscreteNetwork, annotation: DecisionAnnotation, assumptions: DiagramAssumptions
) -> None:
    """Check that every assumed node exists, is assumable, and has the assumed state.

    Raises:
        ValueError: an assumed chance node is unknown, has parents, or lacks the state; or an
            assumed decision is unknown or lacks the state.
    """
    labels = {variable.name: variable.labels for variable in network.variables}
    for node, state in assumptions.chance.items():
        if node not in labels:
            msg = f"'{node}' isn't a chance node of this diagram."
            raise ValueError(msg)
        if len(network.potentials[node].variables) > 1:
            msg = f"'{node}' has parents; only decisions and chance nodes without parents can be assumed."
            raise ValueError(msg)
        if state not in labels[node]:
            msg = f"'{state}' isn't a state of '{node}' (states: {labels[node]})."
            raise ValueError(msg)

    decision_states = {decision.symbol: decision.states for decision in annotation.decisions}
    for symbol, state in assumptions.decisions.items():
        if symbol not in decision_states:
            msg = f"'{symbol}' isn't a decision of this diagram."
            raise ValueError(msg)
        if state not in decision_states[symbol]:
            msg = f"'{state}' isn't a state of decision '{symbol}' (states: {decision_states[symbol]})."
            raise ValueError(msg)


def condition_network(network: DiscreteNetwork, assumptions: DiagramAssumptions) -> DiscreteNetwork:
    """`network` with each assumed chance node's prior replaced by a point mass on its state."""
    labels = {variable.name: variable.labels for variable in network.variables}
    potentials = dict(network.potentials)
    for node, state in assumptions.chance.items():
        values = [1.0 if label == state else 0.0 for label in labels[node]]
        potentials[node] = Potential(variables=[node], shape=[len(values)], values=values)
    return network.model_copy(update={"potentials": potentials})


def fixed_decision_constraints(index_map: IndexMap, assumptions: DiagramAssumptions) -> list[Constraint]:
    """Equality constraints fixing each assumed decision's choice at every information state."""
    constraints: list[Constraint] = []
    for symbol, state in assumptions.decisions.items():
        decision_map = index_map.decisions[symbol]
        own_cardinality = len(decision_map.states)
        state_index = decision_map.states.index(state)

        for info_state in range(prod(decision_map.information_cardinalities)):
            z_index = info_state * own_cardinality + state_index
            constraints.append(
                Constraint(
                    name=f"Assume {symbol} = {state} at information state {info_state}",
                    symbol=f"assume_{symbol}_{info_state}",
                    cons_type=ConstraintTypeEnum.EQ,
                    func=["Subtract", ["At", decision_map.z_symbol, z_index + 1], 1],
                    is_linear=True,
                    is_convex=True,
                    is_twice_differentiable=True,
                )
            )
    return constraints


def compile_with_assumptions(
    network: DiscreteNetwork, annotation: DecisionAnnotation, assumptions: DiagramAssumptions, **rjt_kwargs
) -> tuple[Problem, IndexMap, RootedJunctionTree, DiscreteNetwork]:
    """Compile `(network, annotation)` under `assumptions`.

    `rjt_kwargs` are passed to `build_rooted_junction_tree`.

    Returns:
        tuple[Problem, IndexMap, RootedJunctionTree, DiscreteNetwork]: the compiled problem with
            the fixed-decision constraints added, its index map and tree, and the conditioned
            network to store alongside it.

    Raises:
        ValueError: `assumptions` doesn't validate against the diagram, or compiling fails.
    """
    validate_assumptions(network, annotation, assumptions)
    conditioned = condition_network(network, assumptions)
    rjt = build_rooted_junction_tree(conditioned, annotation, **rjt_kwargs)
    problem, index_map = compile_diagram(conditioned, annotation, rjt=rjt)
    fixed = fixed_decision_constraints(index_map, assumptions)
    if fixed:
        problem = problem.add_constraints(fixed)
    return problem, index_map, rjt, conditioned
