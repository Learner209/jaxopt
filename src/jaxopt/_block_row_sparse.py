from __future__ import annotations

from typing import Hashable, Sequence, TypeVar, Protocol
from dataclasses import field

import jax
import jax.experimental.sparse
import jax_dataclasses as jdc
from jax import numpy as jnp

T = TypeVar('T', bound='SparseMatrixBase')

class SparseMatrixProtocol(Protocol):
    """Protocol defining interface for sparse matrix implementations.
    
    All sparse matrix implementations should implement these methods
    to ensure consistent interface across different formats.
    """
    
    def to_dense(self) -> jax.Array:
        """Convert sparse matrix to dense representation."""
        ...

    def multiply(self, vector: jax.Array) -> jax.Array:
        """Perform sparse matrix-vector multiplication.
        
        Args:
            vector: Input vector for multiplication
            
        Returns:
            Result of matrix-vector multiplication
        """
        ...

@jdc.pytree_dataclass
class SparseBlockRow:
    """A sparse block-row matrix representation.
    
    Each block-row contains a sequence of blocks with equal number of rows but
    potentially different numbers of columns. The blocks are stored concatenated
    along the column axis with their starting column indices tracked separately.
    
    Attributes:
        num_cols: Total width of the block-row including zero columns
        start_cols: Starting column index for each block
        block_num_cols: Number of columns in each block
        blocks_concat: Concatenated block values along column axis
        
    Shape Conventions:
        - start_cols: ([num_block_rows],)
        - blocks_concat: ([num_block_rows,] rows, cols)
    """

    num_cols: jdc.Static[int] = field(frozen=True)
    start_cols: tuple[jax.Array, ...] = field(frozen=True)
    block_num_cols: jdc.Static[tuple[int, ...]] = field(frozen=True) 
    blocks_concat: jax.Array = field(frozen=True)

    @classmethod
    def validate_dimensions(cls, start_cols: Sequence[jax.Array], 
                          block_num_cols: Sequence[int],
                          blocks_concat: jax.Array) -> None:
        """Validate matrix dimensions and block structure.
        
        Args:
            start_cols: Starting column indices
            block_num_cols: Number of columns per block
            blocks_concat: Concatenated block values
            
        Raises:
            ValueError: If dimensions are invalid
        """
        if len(start_cols) != len(block_num_cols):
            raise ValueError("Mismatched number of blocks and starting indices")
        
        if blocks_concat.ndim not in (2, 3):
            raise ValueError("blocks_concat must be 2D or 3D array")

    def __post_init__(self):
        """Validate matrix dimensions on initialization."""
        self.validate_dimensions(self.start_cols, self.block_num_cols, 
                               self.blocks_concat)

    @property
    def treedef(self) -> Hashable:
        """Get pytree definition based on block shapes."""
        return tuple(block.shape for block in self.blocks_concat)

    def to_dense(self) -> jax.Array:
        """Convert to dense matrix representation.
        
        Returns:
            Dense matrix array with shape (rows, num_cols) for single block-row
            or (num_block_rows * rows, num_cols) for batched block-rows.
        """
        if self.blocks_concat.ndim == 3:
            # Handle batched block-rows
            num_block_rows, num_rows, _ = self.blocks_concat.shape
            return jax.vmap(SparseBlockRow.to_dense)(self).reshape(
                (num_block_rows * num_rows, self.num_cols)
            )

        # Handle single block-row
        num_rows, num_cols_concat = self.blocks_concat.shape
        out = jnp.zeros((num_rows, self.num_cols))

        start_concat_col = 0
        for start_col, block_width in zip(self.start_cols, self.block_num_cols):
            end_concat_col = start_concat_col + block_width
            block = self.blocks_concat[:, start_concat_col:end_concat_col]
            out = jax.lax.dynamic_update_slice(
                out, block, start_indices=(0, start_col)
            )
            start_concat_col = end_concat_col

        return out


@jdc.pytree_dataclass
class BlockRowSparseMatrix:
    """Sparse matrix composed of block-rows.
    
    Attributes:
        block_rows: Tuple of SparseBlockRow objects representing the matrix
        shape: Overall shape of the matrix
    """
    block_rows: tuple[SparseBlockRow, ...]
    shape: jdc.Static[tuple[int, int]]

    def multiply(self, target: jax.Array) -> jax.Array:
        """Perform sparse-dense matrix multiplication.
        
        Args:
            target: Dense vector to multiply with
            
        Returns:
            Resulting dense vector after multiplication
        """
        if target.ndim != 1:
            raise ValueError("Target vector must be 1-dimensional")

        out_slices = []
        for block_row in self.block_rows:
            num_blocks, block_rows, block_nonzero_cols = block_row.blocks_concat.shape

            # Extract slices corresponding to nonzero terms in block-row.
            target_slices = [
                jax.vmap(
                    lambda start_idx: jax.lax.dynamic_slice_in_dim(
                        target, start_index=start_idx, slice_size=width, axis=0
                    )
                )(start_indices)
                for start_indices, width in zip(block_row.start_cols, block_row.block_num_cols)
            ]

            # Concatenate slices to form target slice.
            target_slice = jnp.concatenate(target_slices, axis=1)

            # Multiply block-rows with target slice.
            out_slices.append(
                jnp.einsum("bij,bj->bi", block_row.blocks_concat, target_slice).flatten()
            )

        return jnp.concatenate(out_slices, axis=0)

    def to_dense(self) -> jax.Array:
        """Convert the sparse matrix to a dense matrix representation.
        
        Returns:
            Dense matrix array with the same shape as the sparse matrix
        """
        dense_matrix = jnp.concatenate(
            [block_row.to_dense() for block_row in self.block_rows],
            axis=0,
        )
        if dense_matrix.shape != self.shape:
            raise ValueError("Dense matrix shape does not match expected shape")
        return dense_matrix


@jdc.pytree_dataclass
class SparseCsrCoordinates:
    """Coordinates for CSR sparse matrix format.
    
    Attributes:
        indices: Column indices of non-zero entries
        indptr: Index of start to each row
        shape: Shape of the matrix
    """
    indices: jax.Array
    indptr: jax.Array
    shape: jdc.Static[tuple[int, int]]


@jdc.pytree_dataclass
class SparseCsrMatrix:
    """Sparse matrix in CSR format.
    
    Attributes:
        values: Non-zero matrix values
        coords: Indices describing non-zero entries
    """
    values: jax.Array
    coords: SparseCsrCoordinates

    def as_jax_bcsr(self) -> jax.experimental.sparse.BCSR:
        """Convert to JAX BCSR format."""
        return jax.experimental.sparse.BCSR(
            args=(self.values, self.coords.indices, self.coords.indptr),
            shape=self.coords.shape,
            indices_sorted=True,
            unique_indices=True,
        )


@jdc.pytree_dataclass
class SparseCooCoordinates:
    """Coordinates for COO sparse matrix format.
    
    Attributes:
        rows: Row indices of non-zero entries
        cols: Column indices of non-zero entries
        shape: Shape of the matrix
    """
    rows: jax.Array
    cols: jax.Array
    shape: jdc.Static[tuple[int, int]]


@jdc.pytree_dataclass
class SparseCooMatrix:
    """Sparse matrix in COO format.
    
    Attributes:
        values: Non-zero matrix values
        coords: Indices describing non-zero entries
    """
    values: jax.Array
    coords: SparseCooCoordinates

    def as_jax_bcoo(self) -> jax.experimental.sparse.BCOO:
        """Convert to JAX BCOO format."""
        return jax.experimental.sparse.BCOO(
            args=(
                self.values,
                jnp.stack([self.coords.rows, self.coords.cols], axis=-1),
            ),
            shape=self.coords.shape,
            indices_sorted=True,
            unique_indices=True,
        )
