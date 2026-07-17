# Nonlinear multiphase SQP with one-copy global parameters

This note records the end-to-end nonlinear experiment built on top of the
bordered Riccati kernel. It is a complete equality-constrained Gauss--Newton
SQP loop and remains an independent reference implementation. The same
generated problem is now also exposed through `ParametricImplicitNlpOcp` and
solved by FATROP's public primal-dual interior-point algorithm; the executable
can compare that native path directly with IPOPT.

## 1. Nonlinear KKT system

Let `w` contain all stage variables and let

```text
theta in R^p
```

be a one-copy vector of global static parameters. Consider

```text
minimize    phi(w, theta)
subject to  c(w, theta) = 0.
```

For equality multipliers `lambda`, define

```text
r_w     = gradient_w phi + A^T lambda,
r_theta = gradient_theta phi + E^T lambda,
A       = partial c / partial w,
E       = partial c / partial theta.
```

The Gauss--Newton SQP step solves

```text
[ H_ww       A^T   H_wtheta ] [delta_w    ]   [r_w    ]
[ A           0    E        ] [delta_lambda] =-[c      ],
[ H_wtheta^T E^T   H_theta  ] [delta_theta]   [r_theta]
```

where the Hessian blocks are formed from least-squares objective residuals.
Constraint second derivatives are intentionally omitted. This keeps the
experiment focused on the global-parameter linear algebra while retaining a
genuinely nonlinear outer iteration.

## 2. Exact bordered solve inside every SQP iteration

Define the ordinary OCP block and its parameter border as

```text
K = [H_ww A^T],
    [A       0 ],

B = [H_wtheta],
    [E         ].
```

Every SQP iteration performs:

1. one FATROP factorization and base solve

   ```text
   K z_0 = -[r_w; c];
   ```

2. one blocked multi-right-hand-side traversal

   ```text
   K Y = -B;
   ```

3. one dense global-parameter solve

   ```text
   (H_theta + B^T Y) delta_theta
       = -r_theta - B^T z_0;
   ```

4. one back-substitution

   ```text
   [delta_w; delta_lambda]
       = z_0 + Y delta_theta.
   ```

This is algebraically identical to solving the full SQP KKT matrix. The final
dense factorization has cubic cost in `p`; the trajectory work retains the OCP
chain structure.

## 3. Nonlinear implicit Radau collocation

For each phase and collocation point `j`, the benchmark imposes

```text
M_f(X_j, theta)
    sum_(r=0)^d C_(rj) X_r
- h_f f_f(X_j, u, theta)
= 0.
```

The mass matrix is state- and parameter-dependent, so these are nonlinear
implicit dynamics rather than an explicit ODE written in collocation form.
The endpoint equation is

```text
x_(k+1) = sum_(r=0)^d D_r X_r.
```

Collocation states are local controls of the OCP stage, the implicit residuals
are local path equalities, and only the mesh endpoint is propagated. This is
direct Radau collocation of degree 1, 2, or 3.

At phase boundaries, a nonlinear rectangular reset map connects different
state dimensions:

```text
x_(f+1,0) = R_f(x_(f,end), theta).
```

The generated problems therefore combine:

- heterogeneous state dimensions;
- heterogeneous and optionally zero control dimensions;
- nonlinear implicit dynamics;
- direct collocation;
- multiple phases and nonlinear reset maps;
- one-copy global static parameters.

## 4. Globalization and stopping test

The outer algorithm uses an exact `l1` merit function

```text
phi(w, theta) + rho ||c(w, theta)||_1
```

with backtracking. The penalty is kept above the current multiplier estimate.
Primal and multiplier steps use the accepted line-search step length.

Convergence requires both

```text
||c||_infinity
```

and the primal/parameter Lagrangian stationarity residuals to be below the
requested tolerance.

## 5. Independent checks

The executable performs three different checks:

1. a central finite-difference directional check for the objective gradient
   and complete constraint Jacobian;
2. the residual of every structured bordered KKT step;
3. on small cases, comparison of the first structured step with a dense
   full-pivoting KKT solve.

For the default three-phase Radau-3 case:

```text
objective directional-derivative error  2.32e-11
constraint directional-derivative error 4.42e-11
largest structured step residual         3.56e-16
dense first-step difference              3.33e-16
```

The same problem converges in ten SQP iterations to

```text
constraint infinity norm 7.64e-15
dual infinity norm       5.36e-9
```

and agrees with IPOPT's objective to printed precision.

## 6. Factorial nonlinear sweep

The original reference-SQP sweep covers 144 cases:

```text
phases             in {1, 2, 4, 8}
dimension pattern  in {alternating, ramp, zigzag}
dimension scale    in {1, 2}
global p           in {0, 2, 8}
Radau degree       in {1, 3}
```

All 144 cases converged. Across the grid:

```text
largest constraint residual  8.62e-13
largest dual residual        9.93e-9
largest linear-step residual 6.01e-16
largest iteration count      55
```

The largest problem has 1914 trajectory variables, 1748 equality constraints,
eight global parameters, eight phases, and degree-3 collocation.

## 7. IPOPT comparison and interpretation

The current fair comparison gives native FATROP and IPOPT+MA57 the same
analytic functions, sparse first derivatives, supplied Gauss--Newton Hessian,
stopping tolerance, and three timed repetitions.

The short-mesh matrix varies 1/2/4/8 phases, three heterogeneous dimension
patterns, two dimension scales, `p={1,3,6}`, and Radau degrees 1 and 3. Both
solvers converge in all 144 cases. FATROP is faster in 113 cases, but summed
`IPOPT/FATROP` is `0.909`: degree 1 favors FATROP (`1.200`), whereas degree 3
favors MA57 (`0.864`). These problems have only four/five nodes per phase and
can put large dense collocation blocks into each FATROP stage.

With 20/21 nodes per phase and Radau-3, FATROP wins all 12 combinations of
1/2/4/8 phases and 1/3/6 global parameters. Summed native time is `485.710 ms`
versus `1862.490 ms` for IPOPT, or `3.835x`. The result is a crossover map, not
a universal ranking: local collocation condensation, globalization quality,
and problem geometry still matter.

## 8. Remaining integration work

The main missing pieces are:

- IPOPT-grade globalization and broad infeasible/degenerate-start
  qualification in the native interior-point path;
- exact Lagrangian Hessians for nonlinear implicit residuals;
- automatic selection of phase-aware elimination for phase-local static
  parameters;
- local condensation of internal collocation states and equations;
- an end-to-end nonlinear FATROP baseline that copies the global parameter
  into every state;
- a tiled matrix-valued response path for stabilized or nonzero-cross-curvature
  implicit dynamics (the unstabilized zero-cross-curvature full-rank case now
  normalizes to the explicit blocked traversal);
- automatic-differentiation and generated-function interfaces;
- comparisons against MOTO, acados/HPIPM, and other structure-exploiting
  solvers on identical derivatives.

The current experiment is therefore an end-to-end proof of feasibility and a
correctness/performance baseline, not by itself a scientific novelty claim.
