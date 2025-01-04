# PDE-constrained optimization for insulation components of buildings under High-dimensional uncertainty

## Summary
This repository contains a predictive model of PDE-constrained optimization with application to a L-shaped insulation component. The model uses scalable algorithms under high-dimensional uncertainty which are independent of the parameter dimension. There are two methods used for predicting the optimal design under uncertainty: Multi-objective Optimization and Optimization with Chance constraints. 

## Key Features

* 2D and 3D simulations
* Multi-objective optimization
* Chance constraints
* Spatially-correlated uncertainty


## Credits
This software uses the following open source packages:
`SOUPy` implements scalable algorithms for the optimization of large-scale complex systems governed by partial differential equations (PDEs) under high-dimensional uncertainty. The library also provides built-in parallel implementations of optimization algorithms (e.g. BFGS, Inexact Newton CG). SOUPy is built on the open-source `hIPPYlib` library, which provides adjoint-based methods for deterministic and Bayesian inverse problems governed by PDEs, and makes use of `FEniCS` for the high-level formulation, discretization, and solution of PDEs.

`FEniCS` is a popular open-source computing platform for solving partial differential equations (PDEs). `FEniCS` enables users to quickly translate scientific models into efficient finite element code. With the high-level Python and C++ interfaces to `FEniCS`, it is easy to get started, but `FEniCS` offers also powerful capabilities for more experienced programmers. `FEniCS` runs on a multitude of platforms ranging from laptops to high-performance clusters.

`hIPPYlib` implements state-of-the-art scalable algorithms for
deterministic and Bayesian inverse problems governed by partial differential equations (PDEs).
It builds on `FEniCS` for the discretization of the PDE
and on `PETSc` for scalable and efficient linear
algebra operations and solvers.

## Acknowledgement 
The authors gratefully acknowledge the financial support
received from the U.S. National Science Foundation (NSF) CAREER Award CMMI-2143662. 
