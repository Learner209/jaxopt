from __future__ import annotations

import functools
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Hashable,
    Literal,
    Optional,
    Tuple,
    Union,
    assert_never,
    cast,
)

import jax
import jax.experimental.sparse
import jax.flatten_util
import jax_dataclasses as jdc
import numpy as np
import scipy
import scipy.sparse
from jax import numpy as jnp

from jaxopt._fore_jac import (
    make_block_jacobi_precoditioner,
    make_point_jacobi_precoditioner,
)
from jaxopt._block_row_sparse import BlockRowSparseMatrix, SparseCooMatrix, SparseCsrMatrix
from jaxopt._constraint_graph import GraphConstraints
from jaxopt._constraints import VarTypeOrdering, VarValues
from jaxopt.utilities import jax_log

if TYPE_CHECKING:
    import sksparse.cholmod
    from jaxopt._constraint_graph import GraphConstraints

# Type aliases for improved readability
ArrayType = Union[float, jax.Array]
LinearSolverType = Literal["conjugate_gradient", "cholmod", "dense_cholesky"]
SparseMatrixMode = Literal["blockrow", "coo", "csr"]
PreconditionerType = Literal["block_jacobi", "point_jacobi"]

# Cache for CHOLMOD analysis results
_cholmod_analyze_cache: Dict[Hashable, "sksparse.cholmod.Factor"] = {}
MAX_CACHE_SIZE = 512

def _cholmod_solve(
    A: SparseCsrMatrix,
    ATb: jax.Array,
    lambd: ArrayType,
) -> jax.Array:
    """Solve a linear system using CHOLMOD with JIT support.
    
    Args:
        A: Sparse CSR matrix
        ATb: Right-hand side vector
        lambd: Damping factor
        
    Returns:
        Solution vector
    """
    return jax.pure_callback(
        _cholmod_solve_on_host,
        ATb,  # Result shape/dtype
        A,
        ATb,
        lambd,
        vectorized=False,
    )

def _cholmod_solve_on_host(
    A: SparseCsrMatrix,
    ATb: jax.Array,
    lambd: ArrayType,
) -> jax.Array:
    """Solve a linear system using CHOLMOD on the host.
    
    This function solves the normal equation (A^T A + λI)x = A^T b using sparse 
    Cholesky factorization, where:
    - A is the Jacobian matrix of residuals
    - b is the residual vector  
    - λ is the damping factor
    
    Args:
        A: Sparse CSR matrix
        ATb: Right-hand side vector
        lambd: Damping factor
        
    Returns:
        Solution vector
    """
    import sksparse.cholmod

    # Convert CSR to CSC (matrix is transposed)
    A_T_scipy = scipy.sparse.csc_matrix(
        (A.values, A.coords.indices, A.coords.indptr),
        shape=A.coords.shape[::-1]
    )

    # Cache sparsity pattern analysis
    cache_key = (
        A.coords.indices.tobytes(),
        A.coords.indptr.tobytes(),
        A.coords.shape,
    )
    
    factor = _get_or_create_cholmod_factor(A_T_scipy, cache_key)
    
    # Factorize and solve
    factor = factor.cholesky_AAt(
        A_T_scipy,
        beta=lambd + 1e-5,  # Add small regularization term
    )
    return factor.solve_A(ATb)

def _get_or_create_cholmod_factor(
    A_T_scipy: scipy.sparse.csc_matrix,
    cache_key: Hashable,
) -> "sksparse.cholmod.Factor":
    """Get cached CHOLMOD factor or create new one.
    
    Args:
        A_T_scipy: Transposed sparse matrix in CSC format
        cache_key: Cache key for factor lookup
        
    Returns:
        CHOLMOD factor
    """
    import sksparse.cholmod
    
    factor = _cholmod_analyze_cache.get(cache_key)
    if factor is None:
        factor = sksparse.cholmod.analyze_AAt(A_T_scipy)
        _cholmod_analyze_cache[cache_key] = factor

        if len(_cholmod_analyze_cache) > MAX_CACHE_SIZE:
            _cholmod_analyze_cache.pop(next(iter(_cholmod_analyze_cache)))
            
    return factor

@jdc.pytree_dataclass
class ConjugateGradientState:
    """State for Eisenstat-Walker criterion in conjugate gradient solver."""
    
    ATb_norm_prev: ArrayType
    """Previous norm of ATb."""
    
    eta: ArrayType
    """Current tolerance."""

@jdc.pytree_dataclass
class ConjugateGradientConfig:
    """Configuration for conjugate gradient iterative solver.
    
    This solver uses the Eisenstat-Walker criterion for inexact steps.
    See: "Choosing the Forcing Terms in an Inexact Newton Method",
    Eisenstat & Walker, 1996.
    
    Attributes:
        tolerance_min: Minimum tolerance threshold
        tolerance_max: Maximum tolerance threshold
        eisenstat_walker_gamma: Controls tolerance decrease rate (0.5-0.9)
        eisenstat_walker_alpha: Controls tolerance sensitivity (1.5-2.0)
        preconditioner: Type of preconditioner to use
    """

    tolerance_min: float = 1e-7
    tolerance_max: float = 1e-2
    eisenstat_walker_gamma: float = 0.9
    eisenstat_walker_alpha: float = 2.0
    preconditioner: jdc.Static[Optional[PreconditionerType]] = "block_jacobi"

    def _solve(
        self,
        graph: GraphConstraints,
        A_blocksparse: BlockRowSparseMatrix,
        ATA_multiply: Callable[[jax.Array], jax.Array],
        ATb: jax.Array,
        prev_linear_state: ConjugateGradientState,
    ) -> Tuple[jax.Array, ConjugateGradientState]:
        """Solve linear system using conjugate gradient method.
        
        Args:
            graph: Graph constraints
            A_blocksparse: Block sparse matrix
            ATA_multiply: Function to multiply by A^T A
            ATb: Right-hand side vector
            prev_linear_state: Previous solver state
            
        Returns:
            Tuple of (solution vector, new solver state)
        """
        assert len(ATb.shape) == 1, "ATb must be 1-dimensional"

        # Setup preconditioner
        preconditioner = self._setup_preconditioner(graph, A_blocksparse)

        # Calculate tolerance using Eisenstat-Walker criterion
        ATb_norm = jnp.linalg.norm(ATb)
        current_eta = self._compute_eisenstat_walker_tolerance(
            ATb_norm, prev_linear_state
        )

        # Solve with conjugate gradient
        solution_values, _ = jax.scipy.sparse.linalg.cg(
            A=ATA_multiply,
            b=ATb,
            x0=jnp.zeros_like(ATb),
            maxiter=len(ATb),
            tol=float(current_eta),  # Cast to float for type safety
            M=preconditioner,
        )
        
        return solution_values, ConjugateGradientState(
            ATb_norm_prev=ATb_norm,
            eta=current_eta
        )

    def _setup_preconditioner(
        self,
        graph: GraphConstraints,
        A_blocksparse: BlockRowSparseMatrix,
    ) -> Callable[[jax.Array], jax.Array]:
        """Setup preconditioner based on configuration.
        
        Args:
            graph: Graph constraints
            A_blocksparse: Block sparse matrix
            
        Returns:
            Preconditioner function
        """
        if self.preconditioner == "block_jacobi":
            return make_block_jacobi_precoditioner(graph, A_blocksparse)
        elif self.preconditioner == "point_jacobi":
            return make_point_jacobi_precoditioner(A_blocksparse)
        elif self.preconditioner is None:
            return lambda x: x
        else:
            assert_never(self.preconditioner)

    def _compute_eisenstat_walker_tolerance(
        self,
        ATb_norm: ArrayType,
        prev_state: ConjugateGradientState,
    ) -> ArrayType:
        """Compute tolerance using Eisenstat-Walker criterion.
        
        Args:
            ATb_norm: Current norm of A^T b
            prev_state: Previous solver state
            
        Returns:
            New tolerance value
        """
        current_eta = jnp.minimum(
            self.eisenstat_walker_gamma
            * (ATb_norm / (prev_state.ATb_norm_prev + 1e-7))
            ** self.eisenstat_walker_alpha,
            self.tolerance_max,
        )
        return jnp.maximum(
            self.tolerance_min,
            jnp.minimum(current_eta, prev_state.eta)
        )


# Nonlinear solvers.


@jdc.pytree_dataclass
class NonlinearSolverState:
    """State of the nonlinear solver."""
    
    iterations: ArrayType
    """Number of iterations performed."""
    
    vals: VarValues
    """Current variable values."""
    
    cost: ArrayType
    """Current cost value."""
    
    residual_vector: jax.Array
    """Current residual vector."""
    
    termination_criteria: jax.Array
    """Array of boolean flags indicating termination criteria met."""
    
    termination_deltas: jax.Array
    """Array of delta values used in termination checks."""
    
    lambd: ArrayType
    """Current damping factor for Levenberg-Marquardt."""
    
    cg_state: Optional[ConjugateGradientState]
    """State for conjugate gradient solver if used."""


@jdc.pytree_dataclass
class TrustRegionConfig:
    """Configuration for trust region optimization."""
    
    lambda_initial: float = 5e-4
    """Initial damping factor for Levenberg-Marquardt."""
    
    lambda_factor: float = 2.0
    """Factor to increase/decrease damping."""
    
    lambda_min: float = 1e-5
    """Minimum damping factor."""
    
    lambda_max: float = 1e10
    """Maximum damping factor."""
    
    step_quality_min: float = 1e-3
    """Minimum acceptable step quality."""


@jdc.pytree_dataclass
class TerminationConfig:
    """Configuration for optimization termination criteria."""
    
    max_iterations: int = 100
    """Maximum number of iterations."""
    
    cost_tolerance: float = 1e-5
    """Relative cost change tolerance for termination."""
    
    gradient_tolerance: float = 1e-4
    """Gradient norm tolerance for termination."""
    
    gradient_tolerance_start_step: int = 10
    """Step to start checking gradient tolerance."""
    
    parameter_tolerance: float = 1e-6
    """Parameter change tolerance for termination."""

    def _check_convergence(
        self,
        state_prev: NonlinearSolverState,
        cost_updated: jax.Array,
        tangent: jax.Array,
        tangent_ordering: VarTypeOrdering,
        ATb: jax.Array,
        accept_flag: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, jax.Array]:
        """Check convergence criteria.
        
        Args:
            state_prev: Previous solver state
            cost_updated: Updated cost value
            tangent: Search direction
            tangent_ordering: Variable ordering
            ATb: Gradient vector
            accept_flag: Optional flag indicating step acceptance
            
        Returns:
            Tuple of (termination flags, termination deltas)
        """
        # Cost tolerance check
        cost_absdelta = jnp.abs(cost_updated - state_prev.cost)
        cost_reldelta = cost_absdelta / state_prev.cost
        converged_cost = cost_reldelta < self.cost_tolerance

        # Gradient tolerance check
        flat_vals = jax.flatten_util.ravel_pytree(state_prev.vals)[0]
        gradient_mag = jnp.max(
            flat_vals
            - jax.flatten_util.ravel_pytree(
                state_prev.vals._retract(ATb, tangent_ordering)
            )[0]
        )
        converged_gradient = jnp.where(
            state_prev.iterations >= self.gradient_tolerance_start_step,
            gradient_mag < self.gradient_tolerance,
            False,
        )

        # Parameter tolerance check
        param_delta = jnp.linalg.norm(jnp.abs(tangent)) / (
            jnp.linalg.norm(flat_vals) + self.parameter_tolerance
        )
        converged_parameters = param_delta < self.parameter_tolerance

        # Combine termination flags
        term_flags = jnp.array(
            [
                converged_cost,
                converged_gradient,
                converged_parameters,
                state_prev.iterations >= (self.max_iterations - 1),
            ]
        )

        # Apply acceptance criteria
        if accept_flag is not None:
            term_flags = term_flags.at[:3].set(
                jnp.logical_and(
                    term_flags[:3],
                    jnp.logical_or(accept_flag, cost_absdelta == 0.0),
                )
            )

        return term_flags, jnp.array([cost_reldelta, gradient_mag, param_delta])


@jdc.pytree_dataclass
class NonlinearSolver:
    """Nonlinear solver using Gauss-Newton or Levenberg-Marquardt methods."""
    
    linear_solver: jdc.Static[LinearSolverType]
    """Type of linear solver to use."""
    
    trust_region: Optional[TrustRegionConfig]
    """Trust region configuration for Levenberg-Marquardt."""
    
    termination: TerminationConfig
    """Termination criteria configuration."""
    
    conjugate_gradient_config: Optional[ConjugateGradientConfig]
    """Configuration for conjugate gradient solver."""
    
    sparse_mode: jdc.Static[SparseMatrixMode]
    """Sparse matrix format to use."""
    
    verbose: jdc.Static[bool]
    """Whether to print verbose output."""

    @jdc.jit
    def solve(
        self,
        graph: GraphConstraints,
        initial_vals: VarValues
    ) -> VarValues:
        """Solve the nonlinear optimization problem.
        
        Args:
            graph: Graph constraints
            initial_vals: Initial variable values
            
        Returns:
            Optimized variable values
        """
        vals = initial_vals
        residual_vector = graph.compute_res_vec(vals)

        state = NonlinearSolverState(
            iterations=0,
            vals=vals,
            cost=jnp.sum(residual_vector**2),
            residual_vector=residual_vector,
            termination_criteria=jnp.array([False, False, False, False]),
            termination_deltas=jnp.zeros(3),
            lambd=self.trust_region.lambda_initial
            if self.trust_region is not None
            else 0.0,
            cg_state=None
            if self.linear_solver != "conjugate_gradient"
            else ConjugateGradientState(
                ATb_norm_prev=0.0,
                eta=(
                    ConjugateGradientConfig()
                    if self.conjugate_gradient_config is None
                    else self.conjugate_gradient_config
                ).tolerance_max,
            ),
        )

        # Run optimization loop
        state = jax.lax.while_loop(
            cond_fun=lambda s: jnp.logical_not(jnp.any(s.termination_criteria)),
            body_fun=functools.partial(self.step, graph),
            init_val=state,
        )
        
        if self.verbose:
            self._log_termination(state, graph)
            
        return state.vals

    def _log_termination(self, state: NonlinearSolverState, graph: GraphConstraints) -> None:
        """Log termination status and statistics.
        
        Args:
            state: Current solver state
            graph: Graph constraints
        """
        jax_log(
            "Terminated @ iteration #{i}: cost={cost:.4f} criteria={criteria}, "
            "term_deltas={cost_delta:.1e},{grad_mag:.1e},{param_delta:.1e}",
            i=state.iterations,
            cost=state.cost,
            criteria=state.termination_criteria.astype(jnp.int32),
            cost_delta=state.termination_deltas[0],
            grad_mag=state.termination_deltas[1],
            param_delta=state.termination_deltas[2],
        )
        
        residual_index = 0
        for f, count in zip(graph.aggregated_constraints, graph.constraints_counts):
            stacked_dim = count * f.residual_dim
            partial_cost = jnp.sum(
                state.residual_vector[residual_index : residual_index + stacked_dim]
                ** 2
            )
            residual_index += stacked_dim
            jax_log(
                "     - {name}({count}): {cost:.5f} (avg {avg:.5f})",
                name=f.__class__.__name__,
                count=count,
                cost=partial_cost,
                avg=partial_cost / stacked_dim,
                ordered=True,
            )

    def step(
        self,
        graph: GraphConstraints,
        state: NonlinearSolverState
    ) -> NonlinearSolverState:
        """Perform one optimization step.
        
        Args:
            graph: Graph constraints
            state: Current solver state
            
        Returns:
            Updated solver state
        """
        # Compute Jacobian
        A_blocksparse = graph._compute_jacobian_mat(state.vals)
        jac_values = jnp.concatenate(
            [block_row.blocks_concat.flatten() for block_row in A_blocksparse.block_rows],
            axis=0,
        )

        # Setup matrix operations based on sparse mode
        A_multiply, AT_multiply = self._setup_matrix_ops(
            A_blocksparse, jac_values, graph
        )

        # Compute gradient
        ATb = -AT_multiply(state.residual_vector)

        # Solve linear system
        linear_state = None
        local_delta = self._solve_linear_system(
            graph,
            A_blocksparse,
            A_multiply,
            AT_multiply,
            ATb,
            state,
        )

        # Update variables
        vals = state.vals._retract(local_delta, graph.manifold_order)
        
        with jdc.copy_and_mutate(state) as state_next:
            proposed_residual_vector = graph.compute_res_vec(vals)
            proposed_cost = jnp.sum(proposed_residual_vector**2)

            if linear_state is not None:
                state_next.cg_state = linear_state

            # Update state based on step acceptance
            if self.trust_region is None:
                self._update_state_gauss_newton(
                    state_next, vals, proposed_residual_vector, proposed_cost
                )
                accept_flag = None
            else:
                accept_flag = self._update_state_levenberg_marquardt(
                    state_next,
                    vals,
                    proposed_residual_vector,
                    proposed_cost,
                    A_blocksparse,
                    local_delta,
                )

            # Check convergence
            state_next.termination_criteria, state_next.termination_deltas = (
                self.termination._check_convergence(
                    state,
                    proposed_cost,
                    local_delta,
                    graph.manifold_order,
                    ATb,
                    accept_flag,
                )
            )

            state_next.iterations += 1
            
        return state_next

    def _setup_matrix_ops(
        self,
        A_blocksparse: BlockRowSparseMatrix,
        jac_values: jax.Array,
        graph: GraphConstraints,
    ) -> Tuple[Callable[[jax.Array], jax.Array], Callable[[jax.Array], jax.Array]]:
        """Setup matrix multiplication operations based on sparse mode.
        
        Args:
            A_blocksparse: Block sparse matrix
            jac_values: Jacobian values
            graph: Graph constraints
            
        Returns:
            Tuple of (forward multiply, transpose multiply) functions
        """
        if self.sparse_mode == "blockrow":
            A_multiply = A_blocksparse.multiply
            AT_multiply_ = jax.linear_transpose(
                A_multiply, jnp.zeros((A_blocksparse.shape[1],))
            )
            AT_multiply = lambda vec: AT_multiply_(vec)[0]
            return A_multiply, AT_multiply
        elif self.sparse_mode == "coo":
            A_coo = SparseCooMatrix(
                values=jac_values,
                coords=graph.jac_coords_coo
            ).as_jax_bcoo()
            AT_coo = A_coo.transpose()
            A_multiply = lambda vec: A_coo @ vec
            AT_multiply = lambda vec: AT_coo @ vec
            return A_multiply, AT_multiply
        elif self.sparse_mode == "csr":
            A_csr = SparseCsrMatrix(
                values=jac_values,
                coords=graph.jac_coords_csr
            ).as_jax_bcsr()
            A_multiply = lambda vec: A_csr @ vec
            AT_multiply_ = jax.linear_transpose(
                A_multiply, jnp.zeros((A_blocksparse.shape[1],))
            )
            AT_multiply = lambda vec: AT_multiply_(vec)[0]
            return A_multiply, AT_multiply
        else:
            assert_never(self.sparse_mode)

    def _solve_linear_system(
        self,
        graph: GraphConstraints,
        A_blocksparse: BlockRowSparseMatrix,
        A_multiply: Callable[[jax.Array], jax.Array],
        AT_multiply: Callable[[jax.Array], jax.Array],
        ATb: jax.Array,
        state: NonlinearSolverState,
    ) -> jax.Array:
        """Solve the linear system using the configured solver.
        
        Args:
            graph: Graph constraints
            A_blocksparse: Block sparse matrix
            A_multiply: Forward multiplication function
            AT_multiply: Transpose multiplication function
            ATb: Right-hand side vector
            state: Current solver state
            
        Returns:
            Solution vector
        """
        if (
            isinstance(self.linear_solver, ConjugateGradientConfig)
            or self.linear_solver == "conjugate_gradient"
        ):
            cg_config = (
                ConjugateGradientConfig()
                if self.linear_solver == "conjugate_gradient"
                else self.linear_solver
            )
            assert isinstance(state.cg_state, ConjugateGradientState)
            local_delta, _ = cg_config._solve(
                graph,
                A_blocksparse,
                lambda vec: AT_multiply(A_multiply(vec)) + state.lambd * vec,
                ATb=ATb,
                prev_linear_state=state.cg_state,
            )
        elif self.linear_solver == "cholmod":
            # Get Jacobian values from A_blocksparse
            jac_values = jnp.concatenate(
                [block_row.blocks_concat.flatten() for block_row in A_blocksparse.block_rows],
                axis=0,
            )
            A_csr = SparseCsrMatrix(jac_values, graph.jac_coords_csr)
            local_delta = _cholmod_solve(A_csr, ATb, lambd=state.lambd)
        elif self.linear_solver == "dense_cholesky":
            A_dense = A_blocksparse.to_dense()
            normal_matrix = A_dense.T @ A_dense
            diag_idx = jnp.arange(normal_matrix.shape[0])
            normal_matrix = normal_matrix.at[diag_idx, diag_idx].add(state.lambd)
            cho_factor = jax.scipy.linalg.cho_factor(normal_matrix)
            local_delta = jax.scipy.linalg.cho_solve(cho_factor, ATb)
        else:
            assert_never(self.linear_solver)
            
        return local_delta

    def _update_state_gauss_newton(
        self,
        state: NonlinearSolverState,
        vals: VarValues,
        residual_vector: jax.Array,
        cost: ArrayType,
    ) -> None:
        """Update state for Gauss-Newton step.
        
        Args:
            state: State to update
            vals: New variable values
            residual_vector: New residual vector
            cost: New cost value
        """
        state.vals = vals
        state.residual_vector = residual_vector
        state.cost = cost

    def _update_state_levenberg_marquardt(
        self,
        state: NonlinearSolverState,
        vals: VarValues,
        residual_vector: jax.Array,
        cost: ArrayType,
        A_blocksparse: BlockRowSparseMatrix,
        local_delta: jax.Array,
    ) -> jax.Array:
        """Update state for Levenberg-Marquardt step.
        
        Args:
            state: State to update
            vals: New variable values
            residual_vector: New residual vector
            cost: New cost value
            A_blocksparse: Block sparse matrix
            local_delta: Solution vector
            
        Returns:
            Step acceptance flag
        """
        assert self.trust_region is not None
        
        # Evaluate step quality
        step_quality = (cost - state.cost) / (
            jnp.sum(
                (A_blocksparse.multiply(local_delta) + state.residual_vector)
                ** 2
            )
            - state.cost
        )
        accept_flag = step_quality >= self.trust_region.step_quality_min

        # Update state conditionally
        state.vals = jax.tree_map(
            lambda proposed, current: jnp.where(accept_flag, proposed, current),
            vals,
            state.vals,
        )
        state.residual_vector = jnp.where(
            accept_flag, residual_vector, state.residual_vector
        )
        state.cost = jnp.where(accept_flag, cost, state.cost)
        
        # Update damping parameter
        state.lambd = jnp.where(
            accept_flag,
            state.lambd / self.trust_region.lambda_factor,
            jnp.maximum(
                self.trust_region.lambda_min,
                jnp.minimum(
                    state.lambd * self.trust_region.lambda_factor,
                    self.trust_region.lambda_max,
                ),
            ),
        )
        
        return accept_flag
