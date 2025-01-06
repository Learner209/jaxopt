import pathlib
from typing import Literal

import jax
import jaxopt
import matplotlib.pyplot as plt
import tyro

from . import _g2o_utils


def main(
    g2o_file_path: pathlib.Path = pathlib.Path(__file__).parent / "data/input_M3500_g2o.g2o",
    solver_type: Literal[
        "conjugate_gradient", "cholmod", "dense_cholesky"
    ] = "conjugate_gradient",
    termination_config: jaxopt.TerminationConfig = jaxopt.TerminationConfig(),
    trust_region_config: jaxopt.TrustRegionConfig = jaxopt.TrustRegionConfig(),
    sparse_format: Literal["blockrow", "coo", "csr"] = "blockrow",
    output_directory: pathlib.Path = pathlib.Path(__file__).parent / "exp/figs",
    is_verbose: bool = True,
) -> None:
    """Main function to solve pose graph optimization problems.

    Args:
        g2o_file_path: Path to the .g2o file containing the pose graph.
        solver_type: Type of linear solver to use.
        termination_config: Configuration for termination criteria.
        trust_region_config: Configuration for trust region.
        sparse_format: Sparse matrix format to use.
        output_directory: Directory to save output figures.
        is_verbose: Flag to enable verbose output.
    """

    # Parse g2o file.
    with jaxopt.utilities.stopwatch("Reading g2o file"):
        g2o_data: _g2o_utils.G2OData = _g2o_utils.parse_g2o(g2o_file_path)
        jax.block_until_ready(g2o_data)

    # Construct graph from parsed data.
    with jaxopt.utilities.stopwatch("Constructing graph"):
        graph_constraints = jaxopt.GraphConstraints._construct(
            constraints=g2o_data.factors, variables=g2o_data.pose_vars
        )
        jax.block_until_ready(graph_constraints)

    # Initialize solver with initial values.
    with jaxopt.utilities.stopwatch("Initializing solver"):
        initial_values = jaxopt.VarValues.make(
            (
                pose_var.with_value(pose)
                for pose_var, pose in zip(g2o_data.pose_vars, g2o_data.initial_poses)
            )
        )

    # Run the solver to compute optimized values.
    with jaxopt.utilities.stopwatch("Running solver"):
        optimized_values = graph_constraints.compute(
            initial_values,
            trust_region=None,
            linear_solver=solver_type,
            termination=termination_config,
            sparse_mode=sparse_format,
            verbose=is_verbose,
        )

    # Plot results.
    plot_results(g2o_data, initial_values, optimized_values, output_directory, solver_type, sparse_format, termination_config, trust_region_config)


def plot_results(g2o_data, initial_values, optimized_values, output_directory, solver_type, sparse_format, termination_config, trust_region_config):
    """Plot initial and optimized pose graphs.

    Args:
        g2o_data: Parsed G2O data.
        initial_values: Initial pose values.
        optimized_values: Optimized pose values.
        output_directory: Directory to save output figures.
        solver_type: Type of linear solver used.
        sparse_format: Sparse matrix format used.
        termination_config: Termination configuration used.
        trust_region_config: Trust region configuration used.
    """
    figure = plt.figure()
    if isinstance(g2o_data.pose_vars[0], jaxopt.SE2Var):
        plt.plot(
            *(initial_values.get_stacked_value(jaxopt.SE2Var).translation().T),
            c="r",
            label="Initial",
        )
        plt.plot(
            *(optimized_values.get_stacked_value(jaxopt.SE2Var).translation().T),
            c="b",
            label="Optimized",
        )
    elif isinstance(g2o_data.pose_vars[0], jaxopt.SE3Var):
        ax = plt.axes(projection="3d")
        ax.set_box_aspect([1.0, 1.0, 1.0])
        ax.plot3D(
            *(initial_values.get_stacked_value(jaxopt.SE3Var).translation().T),
            c="r",
            label="Initial",
        )
        ax.plot3D(
            *(optimized_values.get_stacked_value(jaxopt.SE3Var).translation().T),
            c="b",
            label="Optimized",
        )
    else:
        raise TypeError("Unsupported pose variable type")

    plot_id = (
        f"{g2o_file_path.stem}_{solver_type}_{sparse_format}_"
        f"term_config_{termination_config.max_iterations}_"
        f"{termination_config.cost_tolerance}_"
        f"{termination_config.gradient_tolerance}_"
        f"{termination_config.parameter_tolerance}_"
        f"trust_region_{trust_region_config.lambda_initial}_"
        f"{trust_region_config.lambda_factor}_"
        f"{trust_region_config.lambda_min}_"
        f"{trust_region_config.lambda_max}_"
        f"{trust_region_config.step_quality_min}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    plt.legend()
    plt.tight_layout()
    figure.savefig(output_directory / f"{plot_id}.pdf", bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    tyro.cli(main)
