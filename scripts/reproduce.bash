# 3d.
g2o_3d_datasets="sphere2500.g2o cubicle.g2o torus3D.g2o grid3D.g2o rim.g2o sphere_bignoise_vertex3.g2o parking-garage.g2o"
# default term and trust region config
# for dataset in $g2o_3d_datasets; do
#     for method in cholmod conjugate_gradient dense_cholesky; do
#         python ./examples/pose_graph_g2o.py --g2o_path ./examples/data/3d/$dataset --linear_solver $method --output_dir ./examples/exp/figs/default/$dataset
#     done
# done

g2o_2d_datasets="input_INTEL_g2o.g2o input_M3500_g2o.g2o input_M3500a_g2o.g2o input_M3500b_g2o.g2o input_M3500c_g2o.g2o input_MITb_g2o.g2o"
# for dataset in $g2o_2d_datasets; do
#     for method in cholmod conjugate_gradient dense_cholesky; do
#         python ./examples/pose_graph_g2o.py --g2o_path ./examples/data/2d/$dataset --linear_solver $method --output_dir ./examples/exp/figs/default/$dataset
#     done
# done


g2o_toro_graph="CSAIL_P_toro.graph FR079_P_toro.graph FRH_P_toro.graph INTEL_P_toro.graph M3500.graph M10000_P_toro.graph"
# for dataset in $g2o_toro_graph; do
#     for method in cholmod conjugate_gradient dense_cholesky; do
#         python ./examples/pose_graph_g2o.py --g2o_path ./examples/data/toro_2d/$dataset --linear_solver $method --output_dir ./examples/exp/figs/default/$dataset
#     done
# done

# studies on max_iterations

# 1. on sphere 2500

# for method in cholmod conjugate_gradient; do
#     for i in {1..6}; do
#         python ./examples/pose_graph_g2o.py --g2o_path ./examples/data/3d/sphere2500.g2o --linear_solver $method --output_dir ./examples/exp/figs/max_iterations/sphere2500 --termination.max_iterations $i
#     done
# done

# 2. on torus3D
# for method in cholmod conjugate_gradient; do
#     for i in {1..6}; do
#         python ./examples/pose_graph_g2o.py --g2o_path ./examples/data/3d/torus3D.g2o --linear_solver $method --output_dir ./examples/exp/figs/max_iterations/torus3D --termination.max_iterations $i
#     done
# done

# 3. on parking-garage.g2o
for method in cholmod conjugate_gradient; do
    for i in {1..6}; do
        python ./examples/pose_graph_g2o.py --g2o_path ./examples/data/3d/parking-garage.g2o --linear_solver $method --output_dir ./examples/exp/figs/max_iterations/parking-garage --termination.max_iterations $i
    done
done

# 4. on input_M3500_g2o.g2o
for method in cholmod conjugate_gradient; do
    for i in {1..6}; do
        python ./examples/pose_graph_g2o.py --g2o_path ./examples/data/2d/input_M3500_g2o.g2o --linear_solver $method --output_dir ./examples/exp/figs/max_iterations/input_M3500_g2o --termination.max_iterations $i
    done
done

# 5. on g2o_3d_datasets.
for dataset in $g2o_3d_datasets; do
    for method in cholmod conjugate_gradient dense_cholesky; do
        for i in {1..6}; do
            python ./examples/pose_graph_g2o.py --g2o_path ./examples/data/3d/$dataset --linear_solver $method --output_dir ./examples/exp/figs/max_iterations_full/$dataset --termination.max_iterations $i
        done
    done
done


# # 6. on g2o_2d_datasets.
for dataset in $g2o_2d_datasets; do
    for method in cholmod conjugate_gradient dense_cholesky; do
        for i in {1..6}; do
            python ./examples/pose_graph_g2o.py --g2o_path ./examples/data/2d/$dataset --linear_solver $method --output_dir ./examples/exp/figs/max_iterations_full/$dataset --termination.max_iterations $i
        done
    done
done

# 7. on g2o_toro_graph.
for dataset in $g2o_toro_graph; do
    for method in cholmod conjugate_gradient dense_cholesky; do
        for i in {1..6}; do
            python ./examples/pose_graph_g2o.py --g2o_path ./examples/data/toro_2d/$dataset --linear_solver $method --output_dir ./examples/exp/figs/max_iterations_full/$dataset --termination.max_iterations $i
        done
    done
done

