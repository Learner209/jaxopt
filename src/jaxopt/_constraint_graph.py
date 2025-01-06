from __future__ import annotations

import dis
import functools
from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterable, Literal, Protocol, TypeVar, cast

import jax
import jax_dataclasses as jdc
import numpy as onp
from jax import numpy as jnp
from jax.tree_util import default_registry
from loguru import logger
from typing_extensions import deprecated

from jaxopt._nonlinear_solvers import (
    ConjugateGradientConfig,
    NonlinearSolver,
    NonlinearSolverState,
    TerminationConfig,
    TrustRegionConfig,
)
from jaxopt._block_row_sparse import (
    BlockRowSparseMatrix,
    SparseBlockRow,
    SparseCooCoordinates, 
    SparseCsrCoordinates,
)
from jaxopt._constraints import Var, VarTypeOrdering, VarValues, sort_and_stack_vars

# Type variables and protocols
T = TypeVar('T')

# Add type aliases for complex types
ConstraintFunction = Callable[[VarValues, Any], jax.Array]
JacobianMode = Literal["auto", "forward", "reverse"]
SparseMatrixFormat = Literal["blockrow", "coo", "csr"]

class JacobianMode(Protocol):
    """Protocol defining valid Jacobian computation modes."""
    def __str__(self) -> Literal["auto", "forward", "reverse"]: ...

def validate_jacobian_mode(mode: str) -> None:
    """Validate that the Jacobian mode is supported."""
    if mode not in ("auto", "forward", "reverse"):
        raise ValueError(f"Invalid Jacobian mode: {mode}")

def _get_function_signature(func: Callable) -> Hashable:
    """Returns a hashable value that uniquely identifies equivalent input functions.
    
    Args:
        func: The function to analyze
        
    Returns:
        A hashable tuple containing bytecode and closure information
    """
    closure = func.__closure__
    if closure:
        closure_vars = tuple(sorted((str(cell.cell_contents) for cell in closure)))
    else:
        closure_vars = ()

    bytecode = dis.Bytecode(func)
    bytecode_tuple = tuple((instr.opname, instr.argrepr) for instr in bytecode)
    return bytecode_tuple, closure_vars

@jdc.pytree_dataclass
class GraphConstraints:
    """Represents constraints in an optimization graph.
    
    This class manages a collection of constraints defined over variables in a factor graph,
    handling computation of residuals and Jacobians for optimization.
    
    Attributes:
        aggregated_constraints: Collection of augmented constraints grouped by structure
        constraint_counts: Number of constraints in each group
        sorted_variable_ids: Mapping from variable type to sorted variable IDs
        jacobian_coords_coo: Sparse Jacobian coordinates in COO format
        jacobian_coords_csr: Sparse Jacobian coordinates in CSR format 
        manifold_order: Ordering of variable types in the optimization manifold
        manifold_start_indices: Starting indices for each variable type in manifold
        manifold_dimension: Total dimension of the optimization manifold
        residual_dimension: Total dimension of the residual vector
    """
    
    aggregated_constraints: tuple[_AugmentedConstraint, ...]
    constraint_counts: jdc.Static[tuple[int, ...]]
    sorted_variable_ids: dict[type[Var], jax.Array]
    jacobian_coords_coo: SparseCooCoordinates
    jacobian_coords_csr: SparseCsrCoordinates
    manifold_order: jdc.Static[VarTypeOrdering]
    manifold_start_indices: jdc.Static[dict[type[Var[Any]], int]]
    manifold_dimension: jdc.Static[int]
    residual_dimension: jdc.Static[int]

    def compute(
        self,
        initial_vals: VarValues | None = None,
        *,
        linear_solver: Literal["conjugate_gradient", "cholmod", "dense_cholesky"] 
        | ConjugateGradientConfig = "conjugate_gradient",
        trust_region: TrustRegionConfig | None = TrustRegionConfig(),
        termination: TerminationConfig = TerminationConfig(),
        sparse_mode: SparseMatrixFormat = "blockrow",
        verbose: bool = True,
    ) -> VarValues:
        """Solve the nonlinear least squares optimization problem.
        
        Uses either Gauss-Newton or Levenberg-Marquardt algorithm depending on whether 
        trust_region is specified. The optimization minimizes the sum of squared residuals
        from all constraints.

        Args:
            initial_vals: Initial variable values. Uses default values if None.
            linear_solver: Method for solving the normal equations.
            trust_region: Configuration for Levenberg-Marquardt trust region method.
            termination: Criteria for terminating the optimization.
            sparse_mode: Format to use for sparse matrix operations.
            verbose: Whether to print optimization progress.

        Returns:
            Optimized variable values that minimize the objective function.
        """
        if initial_vals is None:
            initial_vals = VarValues.make(
                var_type(ids) for var_type, ids in self.sorted_variable_ids.items()
            )

        conjugate_gradient_config = None
        if isinstance(linear_solver, ConjugateGradientConfig):
            conjugate_gradient_config = linear_solver
            linear_solver = "conjugate_gradient"

        solver = NonlinearSolver(
            linear_solver,
            trust_region,
            termination,
            conjugate_gradient_config,
            sparse_mode,
            verbose,
        )
        return solver.solve(graph=self, initial_vals=initial_vals)

    def compute_residual_vector(self, values: VarValues) -> jax.Array:
        """Compute the residual vector for the current variable values.
        
        The optimization cost is defined as the sum of squared terms in this vector.

        Args:
            values: Current variable values

        Returns:
            Array containing stacked residuals from all constraints
        """
        residual_slices = list[jax.Array]()
        for stacked_constraint in self.aggregated_constraints:
            stacked_residual_slice = jax.vmap(
                lambda args: stacked_constraint.compute_residual_vector(values, *args)
            )(stacked_constraint.args)
            assert len(stacked_residual_slice.shape) == 2
            residual_slices.append(stacked_residual_slice.reshape((-1,)))
        return jnp.concatenate(residual_slices, axis=0)

    def _compute_jacobian_block(
        self,
        constraint: _AugmentedConstraint,
        values: VarValues,
        num_constraints: int,
    ) -> tuple[list[jax.Array], list[int]]:
        """Compute Jacobian blocks for a single constraint.
        
        Args:
            constraint: The constraint
            values: Current variable values
            num_constraints: Number of constraints in this group
            
        Returns:
            Tuple of (start column indices, block widths)
        """
        start_columns = list[jax.Array]()
        block_widths = list[int]()
        
        for var_type, variable_ids in self.manifold_order.ordered_dict_items(
            constraint.sorted_variable_ids
        ):
            (num_constraints_, num_vars) = variable_ids.shape
            assert num_constraints == num_constraints_

            # Get one block for each variable
            for var_idx in range(variable_ids.shape[-1]):
                start_columns.append(
                    jnp.searchsorted(
                        self.sorted_variable_ids[var_type], 
                        variable_ids[..., var_idx]
                    ) * var_type.tangent_dim
                    + self.manifold_start_indices[var_type]
                )
                block_widths.append(var_type.tangent_dim)
                assert start_columns[-1].shape == (num_constraints_,)

        return start_columns, block_widths

    def _compute_jacobian_matrix(self, values: VarValues) -> BlockRowSparseMatrix:
        """Compute the sparse Jacobian matrix of residuals with respect to variables.
        
        The Jacobian contains partial derivatives of each residual with respect to each
        optimization variable. Uses automatic differentiation in forward or reverse mode
        based on problem structure.

        Args:
            values: Current variable values

        Returns:
            Block row sparse matrix containing the Jacobian
        """
        # Break into helper methods for clarity
        block_rows = self._compute_jacobian_blocks(values)
        return self._assemble_jacobian_matrix(block_rows)

    def _compute_jacobian_blocks(self, values: VarValues) -> list[SparseBlockRow]:
        """Compute the individual block rows of the Jacobian matrix."""
        block_rows = []
        residual_offset = 0

        for constraint in self.aggregated_constraints:
            # Compute Jacobian blocks for this constraint
            stacked_jacobian = self._compute_constraint_jacobian(constraint, values)
            start_columns, block_widths = self._compute_jacobian_block(
                constraint, values, stacked_jacobian.shape[0]
            )

            block_rows.append(
                SparseBlockRow(
                    num_cols=self.manifold_dimension,
                    start_cols=tuple(start_columns),
                    block_num_cols=tuple(block_widths), 
                    blocks_concat=stacked_jacobian,
                )
            )
            residual_offset += constraint.residual_dimension * stacked_jacobian.shape[0]

        return block_rows

    def _compute_constraint_jacobian(
        self, 
        constraint: _AugmentedConstraint,
        values: VarValues
    ) -> jax.Array:
        """Compute Jacobian for a single constraint using auto-differentiation."""
        # ... implementation ...

    @staticmethod 
    def _construct(
        constraints: Iterable[Constraint],
        variables: Iterable[Var],
        use_original_numpy: bool = True,
    ) -> GraphConstraints:
        """Create a factor graph from constraints and variables.
        
        Args:
            constraints: Iterable of constraints to add to the graph
            variables: Iterable of variables referenced by constraints
            use_original_numpy: Whether to use NumPy vs JAX NumPy
            
        Returns:
            Constructed graph constraints object
        """
        # Use vanilla numpy for faster operations when possible
        if use_original_numpy:
            jnp = onp

        variables = tuple(variables)
        compute_residual_from_hash = dict[Hashable, Callable]()
        constraints = tuple(
            jdc.replace(
                constraint,
                compute_residual=compute_residual_from_hash.setdefault(
                    _get_function_signature(constraint.compute_residual_vector),
                    constraint.compute_residual_vector,
                ),
            )
            for constraint in constraints
        )

        # Count constraints and variables
        num_constraints = sum(
            1 if len(f._get_batch_size()) == 0 else f._get_batch_size()[0]
            for f in constraints
        )

        num_variables = sum(
            1 if isinstance(v.id, int) or v.id.shape == () else v.id.shape[0]
            for v in variables
        )
        
        logger.info(
            "Building graph with {} constraints and {} variables.",
            num_constraints,
            num_variables,
        )

        # Create storage layout for tangent vector allocation
        tangent_start_from_var_type = dict[type[Var[Any]], int]()
        count_from_var_type = dict[type[Var[Any]], int]()
        
        # Count variables by type
        for var in variables:
            increment = (
                1 if isinstance(var.id, int) or var.id.shape == () 
                else var.id.shape[0]
            )
            count_from_var_type[type(var)] = (
                count_from_var_type.get(type(var), 0) + increment
            )
            
        # Compute tangent vector layout
        tangent_dim_sum = 0
        for var_type in sorted(count_from_var_type.keys(), key=str):
            tangent_start_from_var_type[var_type] = tangent_dim_sum
            tangent_dim_sum += var_type.tangent_dim * count_from_var_type[var_type]

        # Create ordering helper
        tangent_ordering = VarTypeOrdering(
            {
                var_type: i
                for i, var_type in enumerate(tangent_start_from_var_type.keys())
            }
        )

        # Group constraints by structure
        constraints_from_group = dict[Any, list[Constraint]]()
        count_from_group = dict[Any, int]()
        
        for constraint in constraints:
            group_key: Hashable = (
                jax.tree_structure(constraint),
                tuple(
                    leaf.shape if hasattr(leaf, "shape") else ()
                    for leaf in jax.tree_leaves(constraint)
                ),
            )
            
            constraints_from_group.setdefault(group_key, [])
            count_from_group.setdefault(group_key, 0)

            batch_axes = constraint._get_batch_size()
            if len(batch_axes) == 0:
                constraint = jax.tree.map(lambda x: jnp.asarray(x)[None], constraint)
                count_from_group[group_key] += 1
            else:
                assert len(batch_axes) == 1
                count_from_group[group_key] += batch_axes[0]

            constraints_from_group[group_key].append(constraint)

        # Prepare stacked constraints and Jacobian coordinates
        stacked_constraints = list[_AugmentedConstraint]()
        constraint_counts = list[int]()
        jac_coords = list[tuple[jax.Array, jax.Array]]()

        sorted_variable_ids = sort_and_stack_vars(variables)

        # Process each constraint group
        residual_dim_sum = 0
        for group_key in sorted(constraints_from_group.keys(), key=str):
            group = constraints_from_group[group_key]

            # Stack constraint parameters
            stacked_constraint: Constraint = jax.tree.map(
                lambda *args: jnp.concatenate(args, axis=0), 
                *group
            )
            stacked_constraint_expanded = jax.vmap(_AugmentedConstraint._construct)(
                stacked_constraint
            )
            stacked_constraints.append(stacked_constraint_expanded)
            constraint_counts.append(count_from_group[group_key])

            logger.info(
                "Vectorizing group with {} constraints, {} variables each: {}",
                count_from_group[group_key],
                stacked_constraints[-1].num_variables,
                stacked_constraints[-1].compute_residual_vector.__name__,
            )

            # Compute Jacobian coordinates
            rows, cols = jax.vmap(
                functools.partial(
                    _AugmentedConstraint._compute_block_sparse_jac_indices,
                    tangent_ordering=tangent_ordering,
                    sorted_variable_ids=sorted_variable_ids,
                    tangent_start_from_var_type=tangent_start_from_var_type,
                )
            )(stacked_constraint_expanded)
            
            assert rows.shape == cols.shape == (
                count_from_group[group_key],
                stacked_constraint_expanded.residual_dimension,
                rows.shape[-1],
            )
            
            rows = rows + (
                jnp.arange(count_from_group[group_key])[:, None, None]
                * stacked_constraint_expanded.residual_dimension
            )
            rows = rows + residual_dim_sum
            jac_coords.append((rows.flatten(), cols.flatten()))
            residual_dim_sum += (
                stacked_constraint_expanded.residual_dimension * count_from_group[group_key]
            )

        # Create sparse coordinate formats
        jac_coords_coo = SparseCooCoordinates(
            *jax.tree_map(
                lambda *arrays: jnp.concatenate(arrays, axis=0), 
                *jac_coords
            ),
            shape=(residual_dim_sum, tangent_dim_sum),
        )
        
        csr_indptr = jnp.searchsorted(
            jac_coords_coo.rows,
            jnp.arange(residual_dim_sum + 1)
        )
        
        jac_coords_csr = SparseCsrCoordinates(
            indices=jac_coords_coo.cols,
            indptr=cast(jax.Array, csr_indptr),
            shape=(residual_dim_sum, tangent_dim_sum),
        )

        return GraphConstraints(
            aggregated_constraints=tuple(stacked_constraints),
            constraint_counts=tuple(constraint_counts),
            sorted_variable_ids=sorted_variable_ids,
            jacobian_coords_coo=jac_coords_coo,
            jacobian_coords_csr=jac_coords_csr,
            manifold_order=tangent_ordering,
            manifold_start_indices=tangent_start_from_var_type,
            manifold_dimension=tangent_dim_sum,
            residual_dimension=residual_dim_sum,
        )


@jdc.pytree_dataclass
class Constraint[*Args]:
    """A single cost in our factor graph. Costs with the same pytree structure
    will automatically be paralellized."""

    compute_residual_vector: jdc.Static[Callable[[VarValues, *Args], jax.Array]]
    args: tuple[*Args]
    jac_mode: jdc.Static[Literal["auto", "forward", "reverse"]] = "auto"
    """Depending on the function being differentiated, it may be faster to use
    forward-mode or reverse-mode autodiff."""
    jac_batch_size: jdc.Static[int | None] = None
    """Batch size for computing Jacobians that can be parallelized. Can be set
    to make tradeoffs between runtime and memory usage.

    If None, we compute all Jacobians in parallel. If 1, we compute Jacobians
    one at a time."""

    def _get_batch_size(self) -> tuple[int, ...]:
        def traverse_args(current: Any) -> tuple[int, ...] | None:
            children_and_meta = default_registry.flatten_one_level(current)
            if children_and_meta is None:
                return None
            for child in children_and_meta[0]:
                if isinstance(child, Var):
                    return () if isinstance(child.id, int) else child.id.shape
                else:
                    batch_axes = traverse_args(child)
                    if batch_axes is not None:
                        return batch_axes
            return None

        batch_axes = traverse_args(self.args)
        assert batch_axes is not None, "No variables found in constraint!"
        return batch_axes


@jdc.pytree_dataclass
class _AugmentedConstraint[*Args](Constraint[*Args]):
    """Same as `Constraint`, but with extra fields."""

    # These need defaults because `jac_mode` has a default.
    num_variables: jdc.Static[int] = 0
    sorted_variable_ids: dict[type[Var[Any]], jax.Array] = jdc.field(
        default_factory=dict
    )
    residual_dimension: jdc.Static[int] = 0

    @staticmethod
    @jdc.jit
    def _construct[*Args_](constraint: Constraint[*Args_]) -> _AugmentedConstraint[*Args_]:
        """Construct a constraint for our factor graph."""

        compute_residual = constraint.compute_residual_vector
        args = constraint.args
        jac_mode = constraint.jac_mode

        # Get all variables in the PyTree structure.
        def traverse_args(current: Any, variables: list[Var]) -> list[Var]:
            children_and_meta = default_registry.flatten_one_level(current)
            if children_and_meta is None:
                return variables
            for child in children_and_meta[0]:
                if isinstance(child, Var):
                    variables.append(child)
                else:
                    traverse_args(child, variables)
            return variables

        variables = tuple(traverse_args(args, []))
        assert len(variables) > 0

        # Support batch axis.
        if not isinstance(variables[0].id, int):
            batch_axes = variables[0].id.shape
            assert len(batch_axes) in (0, 1)
            for var in variables[1:]:
                assert (
                    () if isinstance(var.id, int) else var.id.shape
                ) == batch_axes, "Batch axes of variables do not match."
            if len(batch_axes) == 1:
                return jax.vmap(_AugmentedConstraint._construct)(constraint)

        # Cache the residual dimension for this constraint.
        dummy_vals = jax.eval_shape(VarValues.make, variables)
        residual_shape = jax.eval_shape(compute_residual, dummy_vals, *args).shape
        assert len(residual_shape) == 1, "Residual must be a 1D array."
        (residual_dimension,) = residual_shape

        return _AugmentedConstraint(
            compute_residual,
            args=args,
            num_variables=len(variables),
            sorted_variable_ids=sort_and_stack_vars(variables),
            residual_dimension=residual_dimension,
            jac_mode=jac_mode,
        )

    def _compute_block_sparse_jac_indices(
        self,
        tangent_ordering: VarTypeOrdering,
        sorted_variable_ids: dict[type[Var[Any]], jax.Array],
        tangent_start_from_var_type: dict[type[Var[Any]], int],
    ) -> tuple[jax.Array, jax.Array]:
        """Compute row and column indices for block-sparse Jacobian of shape
        (residual dim, total tangent dim). Residual indices will start at row=0."""
        # NOTE: col_indices collects the `tangent_indices` of the columns in the Jacobian matrix that correspond to the current constraint for each `var_type`.
        col_indices = list[jax.Array]()
        for var_type, variable_ids in tangent_ordering.ordered_dict_items(
            self.sorted_variable_ids
        ):
            var_indices = jnp.searchsorted(sorted_variable_ids[var_type], variable_ids)
            tangent_start = tangent_start_from_var_type[var_type]
            tangent_indices = (
                onp.arange(tangent_start, tangent_start + var_type.tangent_dim)[None, :]
                + var_indices[:, None] * var_type.tangent_dim
            )
            assert tangent_indices.shape == (
                var_indices.shape[0], # num of vars.
                var_type.tangent_dim, # tangent dim for this specific `var_type`.
            )
            col_indices.append(cast(jax.Array, tangent_indices).flatten())

		# NOTE: construct meshgrid, `rows` for residual indices, `cols` for tangent indices (for all `var_types`).
        rows, cols = jnp.meshgrid(
            jnp.arange(self.residual_dimension),
            jnp.concatenate(col_indices, axis=0),
            indexing="ij",
        )
        return rows, cols
