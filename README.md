# PDE-constrained optimization for insulation components of buildings under High-dimensional uncertainty

## Summary
This repository contains a predictive model of PDE-constrained optimization with application to a L-shaped insulation component. There are two methods used for predicting the optimal design under uncertainty: multi-objective optimization and optimization with chance constraints. 

## Key Features

* 2D and 3D simulations
* Multi-objective optimization
* Chance constraints
* Spatially-correlated uncertainty


## Credits
This software uses the following open source packages:
'SOUPy'implements scalable algorithms for the optimization of large-scale complex systems governed by partial differential equations (PDEs) under high-dimensional uncertainty. The library also provides built-in parallel implementations of optimization algorithms (e.g. BFGS, Inexact Newton CG).
`FEniCS` is a popular open-source computing platform for solving partial differential equations (PDEs). `FEniCS` enables users to quickly translate scientific models into efficient finite element code. With the high-level Python and C++ interfaces to `FEniCS`, it is easy to get started, but `FEniCS` offers also powerful capabilities for more experienced programmers. `FEniCS` runs on a multitude of platforms ranging from laptops to high-performance clusters.
`hIPPYlib` implements state-of-the-art scalable algorithms for
deterministic and Bayesian inverse problems governed by partial differential equations (PDEs).
It builds on `FEniCS` for the discretization of the PDE
and on `PETSc` for scalable and efficient linear
algebra operations and solvers.
