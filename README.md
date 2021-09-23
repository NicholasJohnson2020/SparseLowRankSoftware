# SparseLowRankSoftware
Software supplement for the paper  "Sparse Plus Low Rank Matrix Decomposition: A Discrete Optimization Approach"  by Dimitris Bertsimas, Ryan Cory-Wright and Nicholas A. G. Johnson

## Introduction

The software in this package is designed to provide high quality feasible solutions at scale and certifiably near-optimal solutions for small instances of the Sparse Plus Low Rank optimization problem given by

`min ||U - X - Y||_F^2 + \lambda * ||X||_F^2 + \mu * ||Y||_F^2`

`s.t. rank(X) <= k_rank, ||Y||_0 <= k_sparse`

using algorithms described in the paper "Sparse Plus Low Rank Matrix Decomposition: A Discrete Optimization Approach"  by Dimitris Bertsimas, Ryan Cory-Wright and Nicholas A. G. Johnson. Specifically, alternating minimization is used to compute scalable high quality feasible solutions and a custom branch and bound algorithm that leverages alternating minimization and a novel convex relaxation is used to compute certfiably near-optimal solutions.

## Installation and set up

In order to run this software, you must install a recent version of Julia from http://julialang.org/downloads/, and a recent version of the Mosek solver (academic licenses are freely available at https://www.mosek.com/products/academic-licenses/). The most recent version of Julia at the time this code was last tested was Julia 1.5.1 using Mosek version 9.2.

Several packages must be installed in Julia before the code can be run.  These packages can be found in "SparseLowRankSoftware.jl". The code was last tested using the following package versions:

- Distributions v0.25.0
- JuMP v0.21.4
- LowRankApprox v0.5.0
- Mosek v1.1.3
- MosekTools v0.9.4
- SCS v0.7.1
- StatsBase v0.33.8
- TSVD v0.4.3

## Use of the SLR_AM() and SLR_BnB() functions

The two key methods in this package are SLR_AM() and SLR_BnB().  They both take five required  arguments: `U`, `\mu`, `\lambda`, `k_sparse` and `k_rank`, as well as several optional arguments which are described in the respective function docstring. The five required arguments correspond to the input data to the optimization problem.

## Thank you

Thank you for your interest in SparseLowRankSoftware. Please let us know if you encounter any issues using this code, or have comments or questions.  Feel free to email us anytime.

Dimitris Bertsimas
dbertsim@mit.edu

Ryan Cory-Wright
ryancw@mit.edu

Nicholas A. G. Johnson
nagj@mit.edu
