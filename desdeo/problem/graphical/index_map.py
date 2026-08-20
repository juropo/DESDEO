"""Mixed-radix encoding for flattened cluster state spaces.

Phase 0 of `.claude/desdeo-graphical-models-plan.md`. Cluster moments in the rooted junction
tree formulation are indexed by the joint state of every node in a cluster, which is naturally
an N-dimensional tensor for N nodes in the cluster. `TensorVariable`/`TensorConstant` transmit
as tensors to solvers that support them, but few solvers handle more than two dimensions, and
clusters of 4-6 nodes are the normal case. The resolution is to flatten a cluster's joint state
space into a single vector under a mixed-radix encoding, so local consistency, CPT constraints,
and moment linking all become expressions over a 1-D `TensorVariable`.

**Separator-major ordering.** Within a cluster, the variables shared with its parent in the
junction tree (the separator) are placed first, i.e. as the most significant digits. Every joint
state agreeing on the separator state then occupies a contiguous run of flat indices, so
marginalising a cluster's moment vector onto its separator is a sum over contiguous blocks, one
block per separator state, expressible with `Extract`/`Tuple` + `Sum` (see
`separator_block_range` and the correctness/size findings in the plan's Phase 0 section) instead
of a dense marginalisation matrix. A cluster with more than one child can only be contiguous for
one separator at a time; pick the parent's and accept strided access for the others.

The encode/decode convention here matches NumPy's default (C, row-major) order: the first entry
of `cardinalities`/`digits` is the most significant (slowest-varying) axis.
"""

from collections.abc import Sequence
from math import prod


def mixed_radix_encode(digits: Sequence[int], cardinalities: Sequence[int]) -> int:
    """Encode a joint state into a flat 0-based index under a mixed-radix (row-major) scheme.

    Args:
        digits: one 0-based state index per variable, most significant (e.g. separator)
            variable first.
        cardinalities: the number of states of each variable, in the same order as `digits`.

    Raises:
        ValueError: `digits` and `cardinalities` have different lengths, or a digit is out of
            range for its cardinality.

    Returns:
        int: the flat 0-based index, in `[0, prod(cardinalities))`.
    """
    if len(digits) != len(cardinalities):
        msg = f"digits and cardinalities must have the same length, got {len(digits)} and {len(cardinalities)}."
        raise ValueError(msg)

    flat_index = 0
    for digit, cardinality in zip(digits, cardinalities, strict=True):
        if not 0 <= digit < cardinality:
            msg = f"digit {digit} out of range for cardinality {cardinality}."
            raise ValueError(msg)
        flat_index = flat_index * cardinality + digit

    return flat_index


def mixed_radix_decode(flat_index: int, cardinalities: Sequence[int]) -> tuple[int, ...]:
    """Decode a flat 0-based index back into a joint state (inverse of `mixed_radix_encode`).

    Args:
        flat_index: the flat 0-based index to decode.
        cardinalities: the number of states of each variable, most significant first.

    Raises:
        ValueError: `flat_index` is out of range for `cardinalities`.

    Returns:
        tuple[int, ...]: one 0-based state index per variable, in the same order as
            `cardinalities`.
    """
    size = prod(cardinalities)
    if not 0 <= flat_index < size:
        msg = f"flat_index {flat_index} out of range for state space of size {size}."
        raise ValueError(msg)

    digits = [0] * len(cardinalities)
    remainder = flat_index
    for i in range(len(cardinalities) - 1, -1, -1):
        remainder, digits[i] = divmod(remainder, cardinalities[i])

    return tuple(digits)


def separator_block_range(separator_state: int, block_size: int) -> tuple[int, int]:
    """1-based inclusive flat-index range covering one separator state's block.

    Under separator-major ordering, every joint cluster state sharing a given separator state
    occupies a contiguous run of `block_size` flat indices (`block_size` being the product of
    the non-separator variables' cardinalities). The returned bounds are 1-based and inclusive,
    matching MathJSON's `["Extract", cluster_symbol, ["Tuple", start, stop]]` indexing
    convention, so a cluster-to-separator marginalisation constraint for this separator state can
    be written directly as ``["Sum", ["Extract", mu_C, ["Tuple", start, stop]]]``.

    Args:
        separator_state: the 0-based flat index of the separator's own joint state.
        block_size: the number of cluster states sharing one separator state, i.e. the product
            of the cardinalities of the cluster's non-separator variables.

    Returns:
        tuple[int, int]: the `(start, stop)` 1-based inclusive bounds of the block.
    """
    start0 = separator_state * block_size
    return start0 + 1, start0 + block_size
