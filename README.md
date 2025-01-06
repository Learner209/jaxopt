#<p align="center">
  <h3 align="center"><strong>JAXOPT: <br>NLLS and IRLS in JAX</strong></h3>

<p align="center">
    <a href="https://github.com/Learner209">Minghao Liu</a><sup>1</sup>,
    <br>
    <br>
    <sup>1</sup>SJTU
    <br>
</p>

<div align="center">

<img src="https://img.shields.io/badge/Python-v3-E97040?logo=python&logoColor=white" /> &nbsp;&nbsp;&nbsp;&nbsp;
<img alt="powered by JAX" src="https://img.shields.io/badge/JAX-❤️-F8C6B5?logo=jax&logoColor=white"> &nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://img.shields.io/badge/Conda-Supported-lightgreen?style=social&logo=anaconda" /> &nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://github.com/brentyi/jaxopt'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
 <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fbrentyi%2Fjaxopt&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
</div>

## ❖ Contents

- [❖ Contents](#-contents)
- [❖ Introduction:](#-introduction)
- [❖ Installation](#-installation)
  - [Installation](#installation)
- [❖ Usage](#-usage)
  - [Pose Graph Example](#pose-graph-example)
- [❖ Examples](#-examples)
  - [✨Stars/forks/issues/PRs are all welcome!](#starsforksissuesprs-are-all-welcome)
- [❖ Last but Not Least](#-last-but-not-least)

## ❖ Introduction:

**JAXOPT** is a library for nonlinear least squares optimization in JAX. It provides a factor graph interface for specifying and solving least squares problems. The library accelerates optimization by analyzing the structure of graphs: repeated factor and variable types are vectorized, and the sparsity of adjacency in the graph is translated into sparse matrix operations.

Use cases are primarily in least squares problems that are (1) sparse and (2) inefficient to solve with gradient-based methods.

Currently supported:

- Automatic sparse Jacobians.
- Optimization on manifolds.
  - Examples provided for SO(2), SO(3), SE(2), and SE(3).
- Nonlinear solvers: Levenberg-Marquardt and Gauss-Newton.
- Linear subproblem solvers:
  - Sparse iterative with Conjugate Gradient.
    - Preconditioning: block and point Jacobi.
    - Inexact Newton via Eisenstat-Walker.
  - Sparse direct with Cholesky / CHOLMOD, on CPU.
  - Dense Cholesky for smaller problems.

For additional references, see inspirations like [GTSAM](https://gtsam.org/), [Ceres Solver](http://ceres-solver.org/), [minisam](https://github.com/dongjing3309/minisam), [SwiftFusion](https://github.com/borglab/SwiftFusion), and [g2o](https://github.com/RainerKuemmerle/g2o).

<!-- 🤗 Please cite [JAXOPT](https://github.com/brentyi/jaxopt) in your publications if it helps with your work. Please star🌟 this repo to help others notice **JAXOPT** if you think it is useful. Thank you! 😉 -->

## ❖ Installation

**JAXOPT** supports `python>=3.12`:

```bash
pip install git+https://github.com/brentyi/jaxopt.git jaxopt
```
**`jaxopt`** is a library for nonlinear least squares in JAX.

We provide a factor graph interface for specifying and solving least squares
problems. We accelerate optimization by analyzing the structure of graphs:
repeated factor and variable types are vectorized, and the sparsity of adjacency
in the graph is translated into sparse matrix operations.

Use cases are primarily in least squares problems that are (1) sparse and (2)
inefficient to solve with gradient-based methods.

Currently supported:

- Automatic sparse Jacobians.
- Optimization on manifolds.
  - Examples provided for SO(2), SO(3), SE(2), and SE(3).
- Nonlinear solvers: Levenberg-Marquardt and Gauss-Newton.
- Linear subproblem solvers:
  - Sparse iterative with Conjugate Gradient.
    - Preconditioning: block and point Jacobi.
    - Inexact Newton via Eisenstat-Walker.
  - Sparse direct with Cholesky / CHOLMOD, on CPU.
  - Dense Cholesky for smaller problems.

For the first iteration of this library, written for [IROS 2021](https://github.com/brentyi/dfgo), see
[jaxfg](https://github.com/brentyi/jaxfg). `jaxopt` is a rewrite that is faster
and easier to use. For additional references, see inspirations like
[GTSAM](https://gtsam.org/), [Ceres Solver](http://ceres-solver.org/),
[minisam](https://github.com/dongjing3309/minisam),
[SwiftFusion](https://github.com/borglab/SwiftFusion),
[g2o](https://github.com/RainerKuemmerle/g2o).

### Installation

`jaxopt` supports `python>=3.12`:
```bash
pip install git+https://github.com/brentyi/jaxopt.git
```

## ❖ Usage

### Pose Graph Example

```python
import jaxopt
import jaxlie
```

**Defining variables.** Each variable is given an integer ID. They don't need to be contiguous.

```python
pose_vars = [jaxopt.SE2Var(0), jaxopt.SE2Var(1)]
```

**Defining factors.** Factors are defined using a callable cost function and a set of arguments.

```python
# Factors take two arguments:
# - A callable with signature `(jaxopt.VarValues, *Args) -> jax.Array`.
# - A tuple of arguments: the type should be `tuple[*Args]`.
#
# All arguments should be PyTree structures. Variable types within the PyTree
# will be automatically detected.
factors = [
    # Cost on pose 0.
    jaxopt.Factor(
        lambda vals, var, init: (vals[var] @ init.inverse()).log(),
        (pose_vars[0], jaxlie.SE2.from_xy_theta(0.0, 0.0, 0.0)),
    ),
    # Cost on pose 1.
    jaxopt.Factor(
        lambda vals, var, init: (vals[var] @ init.inverse()).log(),
        (pose_vars[1], jaxlie.SE2.from_xy_theta(2.0, 0.0, 0.0)),
    ),
    # Cost between poses.
    jaxopt.Factor(
        lambda vals, var0, var1, delta: (
            (vals[var0].inverse() @ vals[var1]) @ delta.inverse()
        ).log(),
        (pose_vars[0], pose_vars[1], jaxlie.SE2.from_xy_theta(1.0, 0.0, 0.0)),
    ),
]
```

Factors with similar structure, like the first two in this example, will be vectorized under-the-hood.

**Solving optimization problems.** We can set up the optimization problem, solve it, and print the solutions:

```python
graph = jaxopt.FactorGraph.make(factors, pose_vars)
solution = graph.solve()
print("All solutions", solution)
print("Pose 0", solution[pose_vars[0]])
print("Pose 1", solution[pose_vars[1]])
```

## ❖ Examples

For more examples, please refer to the [examples directory](./examples).

### ✨Stars/forks/issues/PRs are all welcome!

## ❖ Last but Not Least

If you have any additional questions or have interests in collaboration, please feel free to contact me at [Minghao Liu](lmh209@sjtu.edu.cn) 😃.
