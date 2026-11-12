"""Phase 0 spike: mixed-radix cluster-state encoding and the local-consistency representation.

See the "Phase 0 -- Encoding and constraint-generation spike" section of
`.claude/desdeo-graphical-models-plan.md`. This module both exercises the mixed-radix
encode/decode helpers and answers the representation question the phase was scoped to settle:
whether a cluster-to-separator marginalisation constraint should be a dense aggregation matrix
(`TensorConstant` + `MatMul`) or a set of compact per-separator-state slice sums
(`Extract`/`Tuple` + `Sum`).

No `graphical` pytest marker yet -- that's registered in Phase 1 alongside the rest of the
`DiscreteNetwork`/pyAgrum adapter suite this module will eventually sit next to.
"""

import itertools
import json
import time

import numpy as np
import pytest

from desdeo.problem import (
    Constraint,
    ConstraintTypeEnum,
    FormatEnum,
    MathParser,
    Objective,
    Problem,
    TensorConstant,
    TensorVariable,
    VariableTypeEnum,
)
from desdeo.problem.graphical.index_map import (
    mixed_radix_decode,
    mixed_radix_encode,
    separator_block_range,
)

# A realistic-scale cluster: 4 nodes x 5 states each (625 joint states), with 3 of the 4 nodes
# forming the separator toward the parent cluster (125 separator states, block size 5). Matches
# the plan's "4 nodes x 5 states = 625 entries, separator 125" scoping.
CARDINALITIES = [5, 5, 5, 5]  # separator-major: first 3 are the separator, last is private
SEPARATOR_CARDINALITIES = CARDINALITIES[:3]
SEPARATOR_SIZE = 5 * 5 * 5  # 125
BLOCK_SIZE = 5  # product of the non-separator (private) cardinalities
CLUSTER_SIZE = SEPARATOR_SIZE * BLOCK_SIZE  # 625


def test_mixed_radix_round_trip_exhaustive():
    """Every flat index in a small joint state space round-trips through encode(decode(.))."""
    cardinalities = [3, 2, 4]
    for flat_index in range(np.prod(cardinalities)):
        digits = mixed_radix_decode(flat_index, cardinalities)
        assert mixed_radix_encode(digits, cardinalities) == flat_index

    for digits in itertools.product(*(range(c) for c in cardinalities)):
        flat_index = mixed_radix_encode(list(digits), cardinalities)
        assert mixed_radix_decode(flat_index, cardinalities) == digits


def test_mixed_radix_matches_numpy_row_major_order():
    """The encoding matches NumPy's default (C, row-major) ravel/unravel.

    That's the order `TensorVariable`/`TensorConstant` values already assume elsewhere in the
    codebase.
    """
    cardinalities = [4, 3, 5]
    for digits in itertools.product(*(range(c) for c in cardinalities)):
        expected = int(np.ravel_multi_index(digits, cardinalities))
        assert mixed_radix_encode(list(digits), cardinalities) == expected
        assert mixed_radix_decode(expected, cardinalities) == digits


def test_mixed_radix_rejects_out_of_range_digit():
    """A digit past its variable's cardinality is rejected rather than silently wrapped."""
    with pytest.raises(ValueError, match="out of range"):
        mixed_radix_encode([0, 5], [3, 3])


def test_mixed_radix_rejects_out_of_range_flat_index():
    """A flat index past the joint state space size is rejected."""
    with pytest.raises(ValueError, match="out of range"):
        mixed_radix_decode(9, [3, 3])


def test_mixed_radix_rejects_mismatched_lengths():
    """`digits` and `cardinalities` must describe the same variables."""
    with pytest.raises(ValueError, match="same length"):
        mixed_radix_encode([0, 1], [3, 3, 3])


def test_separator_major_states_form_contiguous_blocks():
    """Fixing the separator digits and varying only the private digits yields a contiguous run.

    This is the defining property the whole scheme rests on, and it should match
    `separator_block_range`.
    """
    for sep_digits in itertools.product(*(range(c) for c in SEPARATOR_CARDINALITIES)):
        sep_state = mixed_radix_encode(list(sep_digits), SEPARATOR_CARDINALITIES)
        flat_indices = sorted(
            mixed_radix_encode([*sep_digits, private_digit], CARDINALITIES)
            for private_digit in range(CARDINALITIES[-1])
        )
        start, stop = separator_block_range(sep_state, BLOCK_SIZE)
        # separator_block_range is 1-based inclusive; flat_indices are 0-based.
        assert flat_indices == list(range(start - 1, stop))


# ── Representation comparison: dense aggregation matrix vs. compact slice sums ──────────────


def _true_marginal(mu_c: np.ndarray) -> np.ndarray:
    """Ground truth: reshape to (separator_size, block_size) and sum over the block axis."""
    return mu_c.reshape(SEPARATOR_SIZE, BLOCK_SIZE).sum(axis=1)


def test_slice_sum_expression_is_compact_and_correct():
    """A compact per-separator-state slice-sum expression reproduces the true marginal.

    Answers the "can a slice sum be expressed compactly" checklist item: yes, via
    ``["Sum", ["Extract", mu_c, ["Tuple", start, stop]]]``, i.e. existing `Extract`/`Tuple`/`Sum`
    MathJSON operators, no explicit index enumeration needed.
    """
    rng = np.random.default_rng(0)
    mu_c = rng.random(CLUSTER_SIZE)
    expected = _true_marginal(mu_c)

    parser = MathParser(to_format=FormatEnum.gurobipy)

    def callback(name):
        if name == "mu_c":
            return mu_c
        raise KeyError(name)

    for sep_state in range(SEPARATOR_SIZE):
        start, stop = separator_block_range(sep_state, BLOCK_SIZE)
        expr = ["Sum", ["Extract", "mu_c", ["Tuple", start, stop]]]
        assert parser.parse(expr, callback) == pytest.approx(expected[sep_state])


def _build_marginalization_matrix() -> np.ndarray:
    """Dense (separator_size x cluster_size) 0/1 aggregation matrix for the `MatMul` approach."""
    matrix = np.zeros((SEPARATOR_SIZE, CLUSTER_SIZE))
    for sep_state in range(SEPARATOR_SIZE):
        start, stop = separator_block_range(sep_state, BLOCK_SIZE)
        matrix[sep_state, start - 1 : stop] = 1.0
    return matrix


def _build_problem_dense_matmul() -> Problem:
    """Approach A: one vectorised constraint, `M @ mu_C - mu_S = 0`, `M` a dense TensorConstant."""
    mu_c = TensorVariable(
        name="Cluster moments",
        symbol="mu_c",
        shape=[CLUSTER_SIZE],
        variable_type=VariableTypeEnum.real,
        lowerbounds=0.0,
        upperbounds=1.0,
        initial_values=0.0,
    )
    mu_s = TensorVariable(
        name="Separator moments",
        symbol="mu_s",
        shape=[SEPARATOR_SIZE],
        variable_type=VariableTypeEnum.real,
        lowerbounds=0.0,
        upperbounds=1.0,
        initial_values=0.0,
    )
    agg = TensorConstant(
        name="Marginalisation matrix",
        symbol="agg",
        shape=[SEPARATOR_SIZE, CLUSTER_SIZE],
        values=_build_marginalization_matrix().tolist(),
    )
    constraint = Constraint(
        name="Local consistency (dense matmul)",
        symbol="local_consistency",
        func=["Subtract", ["MatMul", "agg", "mu_c"], "mu_s"],
        cons_type=ConstraintTypeEnum.EQ,
        is_linear=True,
        is_convex=True,
        is_twice_differentiable=True,
    )
    return Problem(
        name="Phase 0 spike: dense matmul local consistency",
        description="Benchmark problem for the dense-aggregation-matrix representation.",
        variables=[mu_c, mu_s],
        constants=[agg],
        objectives=[
            Objective(
                name="Dummy",
                symbol="f_1",
                func=["Sum", "mu_c"],
                maximize=False,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
        ],
        constraints=[constraint],
    )


def _build_problem_slice_sums() -> Problem:
    """Approach B: one small `Extract`+`Sum` constraint per separator state, no dense matrix."""
    mu_c = TensorVariable(
        name="Cluster moments",
        symbol="mu_c",
        shape=[CLUSTER_SIZE],
        variable_type=VariableTypeEnum.real,
        lowerbounds=0.0,
        upperbounds=1.0,
        initial_values=0.0,
    )
    mu_s = TensorVariable(
        name="Separator moments",
        symbol="mu_s",
        shape=[SEPARATOR_SIZE],
        variable_type=VariableTypeEnum.real,
        lowerbounds=0.0,
        upperbounds=1.0,
        initial_values=0.0,
    )
    constraints = []
    for sep_state in range(SEPARATOR_SIZE):
        start, stop = separator_block_range(sep_state, BLOCK_SIZE)
        constraints.append(
            Constraint(
                name=f"Local consistency (separator state {sep_state})",
                symbol=f"local_consistency_{sep_state}",
                func=["Subtract", ["Sum", ["Extract", "mu_c", ["Tuple", start, stop]]], ["At", "mu_s", sep_state + 1]],
                cons_type=ConstraintTypeEnum.EQ,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
        )
    return Problem(
        name="Phase 0 spike: slice-sum local consistency",
        description="Benchmark problem for the compact per-separator-state representation.",
        variables=[mu_c, mu_s],
        objectives=[
            Objective(
                name="Dummy",
                symbol="f_1",
                func=["Sum", "mu_c"],
                maximize=False,
                is_linear=True,
                is_convex=True,
                is_twice_differentiable=True,
            )
        ],
        constraints=constraints,
    )


def test_dense_matmul_and_slice_sum_problems_agree_with_ground_truth():
    """Both representations reproduce the same marginal through the gurobipy backend.

    They're equivalent, so the choice between them is purely about representation size and
    construction cost, not correctness.
    """
    rng = np.random.default_rng(1)
    mu_c_values = rng.random(CLUSTER_SIZE)
    expected = _true_marginal(mu_c_values)

    dense_problem = _build_problem_dense_matmul()
    sliced_problem = _build_problem_slice_sums()

    parser = MathParser(to_format=FormatEnum.gurobipy)

    def dense_callback(name):
        if name == "mu_c":
            return mu_c_values
        if name == "agg":
            return _build_marginalization_matrix()
        raise KeyError(name)

    dense_result = np.asarray(parser.parse(dense_problem.constraints[0].func[1], dense_callback))
    np.testing.assert_allclose(dense_result, expected)

    def sliced_callback(name):
        if name == "mu_c":
            return mu_c_values
        raise KeyError(name)

    for sep_state, constraint in enumerate(sliced_problem.constraints):
        # func is `["Subtract", ["Sum", ["Extract", ...]], ["At", "mu_s", ...]]`; evaluate just
        # the slice-sum half (the `mu_s` term is a decision variable, not a constant here).
        slice_sum_expr = constraint.func[1]
        value = parser.parse(slice_sum_expr, sliced_callback)
        assert value == pytest.approx(expected[sep_state])


@pytest.mark.slow
def test_representation_size_and_construction_time_favor_slice_sums():
    """`Problem` construction time and serialised size, measured at the checklist's scale.

    4 nodes x 5 states (cluster_size=625, separator_size=125). Sample measurements on a dev
    machine: dense matmul construction ~85-95ms / ~307KiB JSON, vs. slice sums ~1ms / ~36KiB
    JSON -- roughly two orders of magnitude on construction time and ~8.5x on wire size. Not
    asserted as tight numbers (both are noisy and machine-dependent), but the *direction* is the
    actual finding: at this scale the ~98%-zero 125x625 matrix costs far more, both to validate
    and to transmit, than generating 125 small slice-sum equations, so the compiler should do the
    latter and never materialise the matrix.
    """
    dense_start = time.perf_counter()
    dense_problem = _build_problem_dense_matmul()
    dense_construction_time = time.perf_counter() - dense_start
    dense_json_size = len(dense_problem.model_dump_json().encode())

    sliced_start = time.perf_counter()
    sliced_problem = _build_problem_slice_sums()
    sliced_construction_time = time.perf_counter() - sliced_start
    sliced_json_size = len(sliced_problem.model_dump_json().encode())

    # The dense matrix is ~98% zeros by construction (see _build_marginalization_matrix); the
    # slice-sum representation should never be larger on the wire or slower to build.
    assert sliced_json_size < dense_json_size
    assert sliced_construction_time < dense_construction_time

    # Sanity: both should actually parse back losslessly (Problem fields like `is_convex_` are
    # aliased, so round-tripping needs `by_alias=True` on the way out).
    assert Problem.model_validate(json.loads(dense_problem.model_dump_json(by_alias=True)))
    assert Problem.model_validate(json.loads(sliced_problem.model_dump_json(by_alias=True)))
