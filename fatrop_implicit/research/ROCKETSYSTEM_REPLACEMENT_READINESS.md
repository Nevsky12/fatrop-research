# Standalone readiness for a future RocketSystem/LOPT backend

Date: 2026-07-17

This document is a qualification contract, not an integration plan. FATROP is
not linked from RocketSystem, PSOPT/LOPT, or UCHORT. The current work remains
inside the standalone FATROP repository until the gates below are satisfied.

## Target problem class

The intended backend must solve a multiphase direct-collocation NLP with:

- heterogeneous phase state, control, mesh, and static-parameter dimensions;
- optional one-copy static variables, including phase times and design data;
- explicit or implicit dynamics and full-rank or rank-deficient next-state
  Jacobians;
- stage equalities, two-sided inequalities, and bounds on local variables and
  static parameters;
- adjacent phase reset/linkage equations;
- exact or Gauss--Newton Lagrangian Hessians;
- complete primal, equality-dual, inequality/bound-dual, slack, and barrier
  warm starts across repeated solves;
- reliable status, iteration, and last-finite-solution output.

## Implemented and verified in standalone FATROP

- `ProblemDims` and `ProblemInfo` store a global parameter vector exactly once.
- `ParametricImplicitOcpAbstract` and `ParametricImplicitNlpOcp` expose local
  stage callbacks plus global first- and second-derivative blocks.
- The same native API accepts both `p=0` and `p>0`.
- Static-parameter, state, and control bounds participate in the native
  primal-dual barrier and restoration system.
- `GlobalParameterKktSolver` performs an exact bordered Schur reduction for
  explicit and implicit trajectory systems.
- Full-rank zero-cross-curvature implicit systems normalize a general
  nonsingular `dF/dx[k+1]` once. Small borders reuse scalar explicit factors;
  wider borders use a matrix-valued traversal.
- Exact adjacent-stage Hessian blocks are verified against dense KKT systems
  for full-rank and rank-deficient implicit dynamics, with and without
  equality-dual stabilization, for both the factorizing solve and a different
  retained-RHS solve.
- Stabilized full-rank systems retain the verified modified recursion.
  Rank-deficient systems retain a verified structured fallback.
- `PhaseArrowOcpKktSolver` exactly condenses heterogeneous phase-local blocks
  into a block-tridiagonal phase chain plus a genuinely global arrow.
- `PhaseArrowPdSolver` applies the same two-level elimination to the complete
  per-phase primal-dual barrier Newton system, including inequality slacks,
  lower/upper complementarity, exact adjacent-stage curvature, and optional
  equality-dual stabilization. Dense full-system residual tests cover
  heterogeneous implicit phases and both `p=0` and `p>0`.
- The native IPM returns primal variables, slacks, equality multipliers, lower
  and upper bound multipliers, barrier parameter, and iteration count.
- A complete copied warm start `(x,s,lambda,zL,zU,mu)` can be reused.
- Dense-reference regressions cover global borders, rank deficiency,
  stabilization, restoration, and heterogeneous phase-arrow systems.
- The linearized native matrix currently covers 204 combinations of phase
  count, `(nx,nu,np)`, and implicit collocation degree, including `np=0`.
- A separate dense-reference matrix covers 324 combinations of
  `phases={1,2,4,8}`, heterogeneous `(nx,nu)`, `np={0,2,8}`, transition rank
  loss `{0,1,2}`, and Gauss--Newton, exact-adjacent-curvature, or
  inertia-corrected Hessians.
- Fair full-solver benchmarks compare native FATROP and IPOPT+MA57 on identical
  generated models, derivatives, Hessian approximations, and repeat counts.
- A clean standalone build completes all targets, and all 157 registered CTest
  cases pass.

## Missing before production substitution

### 1. Native nonlinear phase-aware builder and solver selection

The phase-arrow solver now handles a complete assembled primal-dual barrier
Newton step, rather than only an equality-constrained KKT micro-kernel. The
ordinary native `IpAlgorithm`, however, still builds one flattened implicit
chain and selects `PdSolverOrig`; it does not yet create the per-phase views,
reduced linkage/static blocks, and ownership maps required by
`PhaseArrowPdSolver` at every ordinary and restoration iteration. That
phase-aware nonlinear problem builder and linear-solver selector must be added
and tested entirely in standalone FATROP before any application adapter exists.

### 2. Structural representation of nonlocal constraints

The native problem API directly represents stage-local constraints and
adjacent implicit transitions. Aggregate integrals, two-endpoint events, and
general linkage inequalities need one of the following structure-preserving
representations before a future adapter is written:

- auxiliary integral/interface states;
- phase-local or global bordered constraints and slacks;
- an exact adjacent transition lifting.

A generic sparse NLP callback is intentionally insufficient because it hides
the MOCP structure that provides the speed advantage.

### 3. Direct-collocation local condensation

Internal collocation states and equations are currently stage-local blocks in
the Riccati recursion. They are not condensed analytically before the horizon
solve. This is the main short-mesh/high-degree performance gap against MA57.

### 4. Remaining implicit batch cases

The full-rank unstabilized exact-curvature path reuses its structured factors.
For correctness, rank-deficient exact-curvature responses and stabilized
exact-curvature responses currently refactor through the same linear-time
structured recursion for each new RHS. These corners still need cached/tiled
multi-RHS implementations; this is a performance limitation, not a known KKT
correctness limitation.

### 5. Robustness qualification

Synthetic convergence is not enough. The native IPM still needs a standalone
stress suite for poor/infeasible starts, dependent constraints, strong scaling
contrasts, indefinite exact Hessians, evaluation failures, and restoration
cycling. A strict cold-start multiphase rocket mission must converge without
an IPOPT fallback before integration is considered.

### 6. Application-facing ergonomics

The standalone callbacks are low-level BLASFEO callbacks. A future integration
will need derivative/code-generation adapters, canonical scaling and dual-sign
mapping, mesh-refinement warm-start interpolation, cancellation, diagnostics,
and stable status translation. None of these should be added to RocketSystem
until the numerical core passes the gates below.

## Qualification gates

1. All dense-reference and 204-case dimension/collocation regressions continue
   to pass under sanitizers and a broader randomized rank/inertia matrix.
2. Nonlinear sweeps cover `phases={1,2,4,8}`, heterogeneous `(nx,nu)`, and
   `np={0,1,3,6}` for short and long meshes.
3. Trapezoidal, Hermite--Simpson, and Lobatto-4 structural formulations are
   tested explicitly, not inferred only from a generic Radau degree.
4. Cold, primal-only, and complete primal-dual/barrier warm starts converge
   across mesh changes after externally supplied interpolation.
5. Exact KKT residuals and physical constraint violations meet the same
   acceptance thresholds as IPOPT+MA57.
6. A representative multiphase rocket mission converges repeatedly from the
   production initial guess with no fallback.
7. Only after gates 1--6 should a separate LOPT structured-backend adapter be
   introduced. Until then RocketSystem/PSOPT remains FATROP-free.
