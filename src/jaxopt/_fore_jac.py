from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
from jax import numpy as jnp

if TYPE_CHECKING:
    from jaxopt._constraint_graph import GraphConstraints
    from jaxopt._block_row_sparse import BlockRowSparseMatrix


def make_point_jacobi_precoditioner(
    A_blocksparse: BlockRowSparseMatrix,
) -> Callable[[jax.Array], jax.Array]:
    """Returns a point Jacobi (diagonal) preconditioner."""
    # NOTE: approximate the inverse of the diagonal of A^T A by summing the components of each row of A^T A.
    # return a lambda function that takes in a vector and returns the vector divided by the ATA diagonals.
    ATA_diagonals = jnp.zeros(A_blocksparse.shape[1])

    for block_row in A_blocksparse.block_rows:
        (n_blocks, rows, cols_concat) = block_row.blocks_concat.shape
        del rows
        del cols_concat
        assert block_row.blocks_concat.ndim == 3  # (N_block, rows, cols)
        assert block_row.start_cols[0].shape == (n_blocks,)
        block_l2_cols = jnp.sum(block_row.blocks_concat**2, axis=1).flatten()
        indices = jnp.concatenate(
            [
                (start_col[:, None] + jnp.arange(block_cols)[None, :])
                for start_col, block_cols in zip(
                    block_row.start_cols, block_row.block_num_cols
                )
            ],
            axis=1,
        ).flatten()
        ATA_diagonals = ATA_diagonals.at[indices].add(block_l2_cols)

    return lambda vec: vec / ATA_diagonals


def make_block_jacobi_precoditioner(
    graph: GraphConstraints, A_blocksparse: BlockRowSparseMatrix
) -> Callable[[jax.Array], jax.Array]:
    """Returns a block Jacobi preconditioner."""
    # NOTE:     Purpose: Approximates the inverse of block diagonal parts of the matrix A^T A, capturing more structure than point Jacobi.
    # NOTE: `A_blocksparse` 's shape is (num_factors, residual_dim, num_vars * tangent_dim)

    # Mathematical Background: For each variable type, constructs a block diagonal matrix where each block is the sum of
    #  outer products of Jacobian blocks associated with that variable across all factors.
    #     Inverts these blocks to form the preconditioner.
    # Implementation Details:
    #     Initializes gram diagonal blocks for each variable type.
    #     Computes gramian blocks using einsum for factor-variable interactions.
    #     Accumulates gramian blocks and inverts them.
    #     Preconditioner function splits the input vector, applies inverse blocks, and concatenates the result.

    # This list will store block diagonal gram matrices corresponding to each
    # variable.
    gram_diagonal_blocks = list[jax.Array]()
    # NOTE: Numerical Stability: Regularization with a small identity matrix prevents issues with singular blocks
    for var_type, ids in graph.manifold_order.ordered_dict_items(
        graph.sorted_ids_from_var_type
    ):
        (num_vars,) = ids.shape
        gram_diagonal_blocks.append(
            jnp.zeros((num_vars, var_type.tangent_dim, var_type.tangent_dim))
            + jnp.eye(var_type.tangent_dim) * 1e-6
        )

    assert len(graph.aggregated_constraints) == len(A_blocksparse.block_rows)
    for factor, block_row in zip(graph.aggregated_constraints, A_blocksparse.block_rows):
        assert block_row.blocks_concat.ndim == 3  # (N_block, rows, cols)

        # Current index we're looking at in the blocks_concat array.
        start_concat_col = 0

        # NOTE: graph.stacked_factors represents the number of factor groups (not individual factors). Each factor group contains similar constraint functionss.
        # that have been vectorized together for efficient computation. Each factor group produces one block-row in the Jacobian matrix `A_blocksparse`.
        # The individual factors within each group are stacked along the batch dimension.

        for var_type, ids in graph.manifold_order.ordered_dict_items(
            factor.sorted_ids_from_var_type
        ):
            (num_factors, num_vars) = ids.shape
            var_type_idx = graph.manifold_order.order_from_type[var_type]

            # Extract the blocks corresponding to the current variable type.
            end_concat_col = start_concat_col + num_vars * var_type.tangent_dim
            A_blocks = block_row.blocks_concat[
                :, :, start_concat_col:end_concat_col
            ].reshape(
                (
                    num_factors,
                    factor.residual_dim,
                    num_vars,
                    var_type.tangent_dim,
                )
            )

            # f: factor, r: residual, v: variable, t/a: tangent
            gram_blocks = jnp.einsum("frvt,frva->fvta", A_blocks, A_blocks)
            assert gram_blocks.shape == (
                num_factors,
                num_vars,
                var_type.tangent_dim,
                var_type.tangent_dim,
            )

            start_concat_col = end_concat_col
            del end_concat_col

            gram_diagonal_blocks[var_type_idx] = (
                gram_diagonal_blocks[var_type_idx]
                .at[jnp.searchsorted(graph.sorted_ids_from_var_type[var_type], ids)]
                .add(gram_blocks)
            )

    inv_block_diagonals = [
        jnp.linalg.inv(batched_block) for batched_block in gram_diagonal_blocks
    ]

    def preconditioner(vec: jax.Array) -> jax.Array:
        """Compute block Jacobi preconditioning."""
        precond_parts = []
        offset = 0
        for inv_batched_block in inv_block_diagonals:
            num_blocks, block_dim, block_dim_ = inv_batched_block.shape
            assert block_dim == block_dim_
            precond_parts.append(
                jnp.einsum(
                    "bij,bj->bi",
                    inv_batched_block,
                    vec[offset : offset + num_blocks * block_dim].reshape(
                        (num_blocks, block_dim)
                    ),
                ).flatten()
            )
            offset += num_blocks * block_dim
        out = jnp.concatenate(precond_parts, axis=0)
        assert out.shape == vec.shape
        return out

    return preconditioner
