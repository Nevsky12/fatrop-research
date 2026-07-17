# Global-parameter and collocation research benchmarks

The current candidate scientific contribution is documented in
[`INTERVAL_SCOPED_RICCATI.md`](INTERVAL_SCOPED_RICCATI.md): a fill-free reduced
KKT factorization for one-copy variables with contiguous phase scopes. The
note includes the theorem, complexity bound, closest prior art, assumptions,
and falsifiable claims. Reduced, complete implicit-KKT, and primal-dual
prototypes now pass dense-reference tests, and a reproducible scaling
experiment shows the predicted crossover against an optimistic packed-global
baseline. A standalone exact-Hessian primal-dual Radau prototype now handles
two-sided nonlinear inequalities and matches the same NLP in IPOPT+MA57 over
a heterogeneous `(F,nx,nu,p)` sweep. It remains a research hypothesis until
production `IpAlgorithm` integration, real-application qualification, and an
independent prior-art review are complete.

This directory contains experimental benchmarks and the standalone FATROP
implementation. The reusable bordered linear-system kernel is exposed as
`fatrop::GlobalParameterKktSolver` and
`fatrop::ImplicitGlobalParameterKktSolver`. FATROP now also has a native
`ParametricImplicitOcpAbstract` problem API, one-copy parameter storage,
parameter bounds, bordered primal-dual Newton steps, and restoration-phase
coverage. The public `IpAlgorithm` also accepts a copied full primal-dual warm
start `(x,s,lambda,z_L,z_U,mu)`. The same native problem API now also accepts
`p=0`, so parameter-free and parametric problems do not require different
solver plumbing. The current standalone tests include an end-to-end IPM solve,
warm reuse of a one-copy parameter solution, a 204-case
matrix varying phase count, `(nx,nu,np)`, and implicit direct-collocation
degree, and like-for-like full-solver comparisons with IPOPT+MA57.

**Scope note (2026-07-17):** the earlier experimental LOPT/PSOPT/UCHORT
integration was deliberately removed. FATROP is not currently wired into
RocketSystem. All LOPT/UCHORT results and implementation descriptions below
are retained only as historical evidence; they do not describe the current
source tree. The current code answers ten questions:

1. Can a small, one-copy vector of global static parameters be handled as a
   dense border around the OCP KKT system instead of copying it into every
   state?
2. Does the same construction remain valid and useful when an implicit ODE is
   transcribed with direct Radau collocation?
3. Does introducing multiple phases and explicit inter-phase reset/linkage
   equations change the structured-solver comparison?
4. Can all global-parameter sensitivity columns be propagated in one blocked
   Riccati traversal instead of one full horizon sweep per parameter?
5. Does the same exact reduction work for heterogeneous phases with different
   state, control, and phase-local parameter dimensions?
6. What remains valid when the next-state Jacobian is rank deficient, the
   exact Hessian couples adjacent stages, or inertia correction is required?
7. Does the bordered recursion remain correct and useful inside a complete
   nonlinear multiphase SQP iteration with implicit direct collocation?
8. Does the native FATROP primal-dual IPM carry the same structure through a
   complete bounded nonlinear solve, rather than only through a KKT kernel?
9. On identical generated NLPs, callbacks, Hessian models, and repeat counts,
   where does native FATROP outperform IPOPT+MA57 and where does it not?
10. Can phase-local parameter/linkage blocks be eliminated by a second
    phase-arrow recursion, leaving only genuinely mission-global variables in
    the final dense block, and what then limits the crossover?

The short answer from the current experiments is:

- the bordered solve is algebraically correct;
- a full primal-dual solution can be copied back into the standalone solver;
  global `p` is part of the primal vector and is therefore warm-started once,
  while boundary slacks and nonpositive bound multipliers are repaired before
  the next iteration;
- it is faster than naive state augmentation for the tested non-identity
  implicit dynamics;
- a blocked multi-right-hand-side Riccati traversal now propagates all
  parameter columns in one horizon pass and is exact for shooting and Radau
  degree 1--3; it improves the parameter-sensitivity kernel for moderate
  parameter dimensions, while one-column and very wide batches still expose
  workspace and cache overhead;
- at fixed total node count, replacing ordinary transitions by phase linkages
  has negligible timing impact; at fixed nodes per phase, both structured
  methods scale approximately linearly with the number of phases;
- heterogeneous phases, rectangular full-row-rank reset maps, and phase-local
  static parameters agree with a dense KKT reference; putting every local
  parameter into one dense border is correct but does not yet exploit their
  phase locality;
- rank-deficient implicit dynamics are exact through compact rank-aware
  preprocessing for both zero and nonzero adjacent-stage Hessian blocks. A
  universal lifted formulation remains an independent oracle and a route to
  rectangular residual counts that the compact public interface does not yet
  expose;
- exact adjacent-stage curvature is checked against dense KKT references for
  full-rank and rank-deficient dynamics, with and without stabilization, for
  both the factorizing solve and a different retained RHS. Rank-deficient and
  stabilized exact-curvature RHS paths currently refactor through the same
  structured recursion, so their remaining limitation is factor reuse and
  batching performance rather than known algebraic correctness;
- an equality-constrained Gauss--Newton SQP now solves nonlinear multiphase
  implicit Radau problems end to end; all 144 tested nonlinear combinations of phase
  count and `(nx, nu, p)` converged, and every bordered Newton step retained a
  residual below `6.1e-16`;
- a separate interval-scoped exact-Hessian predictor-corrector IPM keeps
  phase, interface, segment, and global variables in one copy, represents
  heterogeneous nonlinear phase resets with `[f,f+1]` separator states, and
  supports nonlinear two-sided path inequalities.  Its affine and corrector
  steps share one numeric factorization.  It matches the same NLP in
  IPOPT+MA57 on an eleven-profile `(F,nx,nu,p)` sweep and exposes the expected
  loss when the active front grows to `omega=58`;
- the bordered factor/blocked-response/Schur solve is now a reusable public
  FATROP component rather than benchmark-local code, with dense-reference
  unit tests over heterogeneous `(nx, nu, p)` configurations;
- on the fair 96-case quadratic full-solver matrix, native FATROP was faster
  in all 96 cases: summed IPOPT/FATROP was `1.357` for shooting and `1.693`
  for Radau-3, with solution/multiplier differences below `8.9e-15`;
- on the harder 144-case nonlinear short-mesh matrix, both solvers converged
  in all cases and FATROP won 113/144, but the summed ratio was only `0.909`:
  degree-1 favored FATROP (`1.200`), while degree-3 with large dense local
  collocation blocks favored MA57 (`0.864`);
- increasing the nonlinear Radau-3 mesh to 20/21 nodes per phase changes the
  crossover: FATROP won all 12 combinations of 1/2/4/8 phases and 1/3/6
  global parameters, with a summed IPOPT/FATROP ratio of `3.835`;
- the historical, now-removed LOPT backend solved trapezoidal,
  Hermite--Simpson, and Lobatto-4
  direct collocation with finite bounds, local/adjacent/dense inequalities,
  aggregate integrals/events/linkages, free phase times, one-copy static
  parameters, exact nonlocal curvature, heterogeneous phases, and full
  primal-dual/barrier warm starts;
- equality-dual stabilization handles cases where the trajectory-only KKT
  block is singular but the complete parameter-bordered system is regular.
  Full-rank implicit systems preprocess once and expose stabilized responses
  through a batch interface. For unstabilized systems with zero cross-stage
  curvature, any nonsingular next-state block is normalized once and then
  uses an adaptive scalar/blocked explicit traversal; stabilized and
  nonzero-cross-curvature cases still use the modified recursion, and
  rank-deficient systems retain a verified scalar fallback;
- the historical 216-case LOPT factorial matrix over 1/2/4/8 phases, three state sizes, two
  control sizes, three static-parameter sizes, and all three collocation
  schemes converges and agrees with IPOPT+MA57 in every case;
- a reusable two-level phase-arrow solver now factors each phase trajectory
  independently, condenses phase-local interface/parameter coordinates into a
  block-tridiagonal phase system, and retains only genuinely global coordinates
  in the final arrow. A primal-dual wrapper now performs the complete local
  slack/bound-dual reduction and reconstructs the full barrier Newton step.
  Dense-reference tests cover heterogeneous implicit phases with 1/2/4/8 KKT
  phase blocks and full primal-dual residual tests cover 1/2/4 phases with
  `p=0/2`;
- sparse active-column detection and compact BLAS Schur assembly reduce the
  eight-phase representative medians to `2.433 ms` (trapezoidal), `5.075 ms`
  (Hermite--Simpson), and `9.898 ms` (Lobatto-4), versus IPOPT+MA57 at
  `2.057`, `3.450`, and `4.394 ms`. The remaining dominant cost is the
  nominally batched implicit response path, which still loops over scalar
  Riccati right-hand sides;
- the post-safeguard 216-case matrix still converges and agrees with IPOPT in
  every case. FATROP is faster in 91/216 single-repeat timings; mean
  `IPOPT/FATROP` is `2.095`, `1.129`, `0.804`, and `0.671` for 1/2/4/8
  phases respectively. This is a crossover map, not a universal speed win;
- the removed UCHORT experiment had an explicit `ipopt|fatrop` backend selector. Its representative
  two-phase mission completes both a cold solve and a `2 -> 4` interval mesh
  refinement through FATROP without invoking the generic fallback. Fixed
  zero-width NLP boxes are projected exactly on output, preventing scaled KKT
  tolerances from becoming physical arc-domain violations after unscaling;
- the removed LOPT experiment kept a shifted objective/violation filter and invoked
  a regularized phase-arrow feasibility normal-step when the ordinary SQP line
  search is exhausted. It also has inequality-multiplier recentering, a dual
  watchdog, an early convexified-Hessian fallback, and a guarded structured
  second-order correction. Dedicated regressions exercise restoration, SOC,
  and an indefinite exact Hessian. UCHORT exposes an explicit numerical
  failure policy: strict return, IPOPT from the original start, or IPOPT from
  FATROP's last finite primal;
- a strict five-phase Delta-III cold-start qualification is still negative.
  IPOPT+MA57 solved its first coarse-grid NLP in 36 iterations, while FATROP
  reached its 240 accepted-step limit at `kkt=3.03e-3` and scaled feasibility
  `2.74e-3`. This is the current production-robustness blocker.

See [RESULTS.md](RESULTS.md) for measurements and conclusions.
See [ROCKETSYSTEM_REPLACEMENT_READINESS.md](ROCKETSYSTEM_REPLACEMENT_READINESS.md)
for the standalone qualification gates that must be met before any future
LOPT/RocketSystem integration.

## Executables

| Executable | Purpose |
| --- | --- |
| `global_parameter_kkt_benchmark` | Synthetic single- or multi-phase implicit shooting, direct Radau collocation, and a DTOC3-derived quadratic problem; compares sequential and blocked bordered KKT solves, naive FATROP state augmentation, dense LU on small cases, and optionally IPOPT. |
| `parametric_riccati_reference` | Small dense reference for the exact `P/S/T` Riccati recursion with a one-copy global parameter vector. |
| `heterogeneous_phase_kkt_benchmark` | Shooting and Radau problems with per-phase `nx`, `nu`, and local static-parameter dimensions plus one global vector; compares sequential and blocked bordered solves and dense LU. |
| `interval_scoped_kkt_benchmark` | Reduced interval-graph factorization versus Eigen SparseLU and dense LU, varying phase count and active width. |
| `interval_scoped_ocp_benchmark` | Complete heterogeneous implicit-OCP KKT solve versus an optimistic dense packed-global baseline using identical phase kernels and active response columns. |
| `rank_deficient_implicit_kkt_benchmark` | Heterogeneous multiphase implicit dynamics with prescribed Jacobian rank loss, global parameters, and inertia correction; compares compact rank-aware preprocessing, universal lifting, and dense KKT. |
| `nonlinear_global_parameter_sqp_benchmark` | Complete equality-constrained Gauss--Newton SQP for nonlinear implicit Radau collocation, heterogeneous phases, nonlinear reset maps, and one-copy global parameters; optionally compares against IPOPT. |
| `nonlinear_interval_scoped_sqp_benchmark` | Exact-Hessian nonlinear implicit Radau predictor-corrector IPM with two-sided path inequalities, one-copy phase/interface/segment/global variables, and heterogeneous interface separators; compares the same analytic NLP against IPOPT. |
| `dtoc_fatrop_benchmark` | End-to-end FATROP solves for exact DTOC1L, DTOC3, and DTOC6 formulations derived from CUTEst SIF files. |
| `cutest_probe` | Loads a decoded CUTEst problem and reports dimensions, derivative sparsity, and locality before and after an OCP stage ordering. |
| `cutest_ipopt_benchmark` | Generic CUTEst-to-IPOPT adapter with exact sparse first and second derivatives. |
| `bench_fatrop_backend` (historical, removed LOPT tree) | Archived end-to-end FATROP versus IPOPT+MA57 measurements; this executable is not part of the current standalone tree. |

The two CUTEst executables are built only when a CUTEst installation is found.

## Global-parameter formulation

Let `w` contain all stage variables and equality multipliers, and let
`theta` be a global parameter vector of dimension `p`. A Newton/KKT system has
the bordered form

```text
[ K       B ] [delta_w    ] = -[r_w    ]
[ B^T  Htheta] [delta_theta]    [r_theta]
```

where `K` retains the ordinary block-banded OCP structure. The experimental
solve:

1. factors `K` once with FATROP's OCP recursion;
2. solves the base right-hand side;
3. applies the same factorization to the `p` columns of `B`, either as
   sequential vector solves or as one blocked matrix-valued Riccati traversal;
4. forms and solves a dense `p x p` Schur complement;
5. back-substitutes into the stage variables and multipliers.

The naive comparison appends `theta_k` to every state and imposes

```text
theta[k + 1] - theta[k] = 0.
```

For full-rank, non-identity `dF/dx[k+1]`, the naive baseline uses the
experimental `ImplicitOcpType`. For DTOC3 and collocation, whose dynamics have
already been normalized to `dF/dx[k+1] = -I`, it uses `OcpType`. This distinction
is important: running the implicit preprocessing for an already explicit
problem gives an unfairly slow naive baseline.

### Reusable core kernel

`fatrop/ocp/global_parameter_kkt_solver.hpp` implements the same exact
reduction as reusable explicit and implicit aliases. It owns the ordinary OCP
factorization, propagates `H_xp` and `J_p`, forms the dense parameter Schur
complement, and back-substitutes the trajectory and equality multipliers.
The explicit path uses `solve_rhs_batch`. Full-rank implicit systems cache the
LU/permutation preprocessing for both stabilized and unstabilized solves, so
they never refactorize per border column. In the unstabilized zero-cross-stage-
curvature case, a full-rank non-identity `dF/dx[k+1]` is normalized once and
delegated to the explicit recursion. Batches of one to three columns use its
allocation-free scalar factors; wider batches use the matrix-valued traversal.
An exact `-I` block skips implicit preprocessing entirely. Stabilized systems
and exact Hessians with nonzero adjacent-stage curvature retain the modified
recursion. The rank-deficient path retains the verified scalar
reused-factorization fallback.

The corresponding unit test compares the complete bordered solution against
a dense KKT solve for horizons of 2, 5, and 8 stages, heterogeneous state and
control dimensions, parameter dimensions 1, 3, and 8, rank-deficient implicit
next-state Jacobians, and a stabilized singular-trajectory border. A singular
parameter Schur complement is reported as `NOFULL_RANK`.

The linear kernel is now connected to FATROP's own nonlinear stack through
`ParametricImplicitNlpOcp`. The native path assembles objective and constraint
derivatives with respect to one-copy parameters, converts parameter box bounds
to structured terminal inequalities, includes the border in the primal-dual
Newton step, and preserves it through restoration reduction.

### Native primal-dual warm start

After a solve, the complete iterate can be snapshotted before the next
`optimize()` call:

```cpp
algorithm->set_warm_start_from_solution();
auto status = algorithm->optimize();
```

The explicit overload accepts `primal_x`, `primal_s`, equality multipliers,
lower/upper bound multipliers, and `mu`. Inputs are copied immediately and
validated for dimensions and finite values. `clear_warm_start()` restores the
NLP-provided cold start. This is standalone functionality only; no
RocketSystem mesh-refinement driver calls it yet.

### Historical LOPT integration backend (removed)

The former companion LOPT worktree had a structure-preserving backend route
and an initial `FatropEqualitySqpSolver`. It consumed the exact scaled/staged
`AnyNlpProblem` together with its MOCP model and layout, then flattens
heterogeneous phases into one implicit FATROP chain. Nodal controls are carried
as stage states; synthetic next-control inputs and consensus transitions make
two-point direct collocation square without changing the canonical NLP.
Transparent consensus bridges concatenated independent phase chains. None of
that adapter code is present in PSOPT/RocketSystem now.

The validated subset is:

- trapezoidal, Hermite--Simpson, and Lobatto-4 local direct collocation;
- fixed, one-sided, and two-sided variable bounds;
- stage-local, adjacent-stage, and dense inequality slacks/barriers;
- aggregate integral/event/linkage constraints and exact promoted nonlocal
  Lagrangian curvature;
- one-copy phase parameters and fixed/free phase times;
- heterogeneous phase counts and `(nx, nu, np)`;
- exact NLP gradient, Jacobian, and Lagrangian Hessian callbacks;
- primal, constraint-dual, bound-dual, inequality-slack, and barrier warm
  starts across repeated NLP solves.

The driver tests verify both the scaled structured-first route and unchanged
fallback to IPOPT. See the LOPT integration checkpoint in
[RESULTS.md](RESULTS.md). Compatible affine sequential phase-boundary
equalities are consumed by the implicit bridge; nonlinear-curvature boundaries
use the exact promoted-border path, and non-defect adjacent equalities inside a
phase remain unsupported. Phase-arrow elimination is now integrated. A true
blocked implicit multi-right-hand-side recursion, native FATROP problem-API
support for global variables in its own restoration machinery, and broad
mission robustness validation remain outside the current backend. The LOPT
wrapper itself now has a persistent filter, a feasibility normal-step, a
targeted dual watchdog, and a guarded one-step SOC. It still lacks an
IPOPT-grade composite normal/tangential step, soft restoration, and broadly
qualified globalization safeguards.

## Tested transcriptions

### Implicit shooting

The synthetic shooting problem has a full-rank, generally non-identity
next-state Jacobian:

```text
F_k(u_k, x_k, x_{k+1}, theta) = 0.
```

The bordered method normalizes each dynamics block before applying the ordinary
FATROP recursion. `normalization_ms` is reported separately.

### Direct Radau collocation

The synthetic collocation problem discretizes

```text
M xdot = A x + B u + E theta + d
```

with Radau degree 1, 2, or 3. Collocation states are local stage variables,
collocation residuals are stage equality constraints, and the mesh endpoint is
the state passed to the next stage. This is direct collocation, not multiple
shooting disguised as collocation.

The KKT-only executables use linear-quadratic models and fixed steps to isolate
the linear algebra. `nonlinear_global_parameter_sqp_benchmark` separately
uses state- and parameter-dependent mass matrices, nonlinear vector fields,
and nonlinear phase resets inside a complete SQP loop. Derivatives in that
executable are analytic rather than generated by automatic differentiation.

### Multiple phases

For the synthetic problems, `--phases F` partitions the requested total number
of nodes as evenly as possible among `F` phases. Every phase has at least two
nodes. Ordinary dynamics/collocation transitions remain inside a phase. The
last node of every nonterminal phase is connected to the first node of the next
phase by a separate parameter-dependent reset map

```text
x[phase + 1, 0] = R_phase x[phase, end] + E_phase theta + d_phase.
```

The phase dynamics and integration step vary deterministically with the phase.
The reset map is represented as an OCP dynamics block, so the flattened KKT
system remains a chain but contains genuine inter-phase linkage equations.
`global_parameter_kkt_benchmark` uses common `nx` and `nu` values inside one
problem so it can compare against its naive augmentation and IPOPT baselines.

`heterogeneous_phase_kkt_benchmark` removes that restriction. Each phase has
independent `nx_f`, `nu_f`, and `p_f`, together with a shared `p_g`. A reset
from `nx_f` to `nx_(f+1)` is rectangular but remains full row rank because the
incoming-state Jacobian is `-I`. Stage functions couple only to `p_g` and their
own `p_f`; linkages may couple `p_g`, `p_f`, and `p_(f+1)`.

See [HETEROGENEOUS_PHASE_PARAMETERS.md](HETEROGENEOUS_PHASE_PARAMETERS.md)
for the exact reduction and why a dense border in
`p_g + sum_f p_f` is a correctness reference rather than the final scalable
phase-local algorithm.

### Rank-deficient implicit dynamics

For a linearized residual

```text
A_k delta_y_k + J_k delta_x[k+1] + E_k delta_theta = -r_k,
```

the ordinary normalization requires a square nonsingular `J_k`. The
rank-aware prototype instead separates determined and free incoming-state
coordinates. Free coordinates become controls of the next stage, while
unused dynamics equations become path equalities.

The independent universal baseline introduces a local copy
`v_k = x[k+1]`. The implicit residual is then a stage equality in
`(y_k, v_k)`, and the copy equation is an ordinary explicit transition. This
lifting works for singular or rectangular residual Jacobians and makes an
exact `y_k`--`x[k+1]` Hessian block stage-local.

See [RANK_DEFICIENT_IMPLICIT.md](RANK_DEFICIENT_IMPLICIT.md) for the
rank-aware transformation, the lifting proof, and the reduced-Hessian inertia
condition.

### Nonlinear multiphase SQP

The end-to-end benchmark solves

```text
minimize    phi(w, theta)
subject to  c(w, theta) = 0
```

with an `l1` merit-function line search. Every Gauss--Newton SQP iteration
factors the ordinary trajectory KKT block once, propagates all global
parameter columns with the blocked Riccati traversal, and solves the final
`p x p` Schur complement.

Its direct-collocation residual is

```text
M_f(X_j, theta) sum_r C_rj X_r
- h_f f_f(X_j, u, theta) = 0,
```

so the dynamics are nonlinear and implicit. State and control dimensions may
change at nonlinear inter-phase reset maps. See
[NONLINEAR_GLOBAL_PARAMETER_SQP.md](NONLINEAR_GLOBAL_PARAMETER_SQP.md) for the
KKT derivation, globalization, verification, and scope of the IPOPT
comparison.

### DTOC problems

DTOC1L, DTOC3, and DTOC6 are reconstructed from their CUTEst SIF definitions.
FATROP represents the fixed initial state as a stage equality. CUTEst keeps the
same condition as fixed variable bounds, so the FATROP constraint count is
larger by the state dimension although the mathematical problem is equivalent.

## Build

The benchmarks require Eigen and IPOPT. CUTEst is optional.

```bash
cmake -S . -B build-research -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_WITH_C_INTERFACE=OFF \
  -DBUILD_RESEARCH_BENCHMARKS=ON \
  -DWITH_BUILD_BLASFEO=ON \
  -DCUTEST_ROOT="$PWD/../research-deps/install"

cmake --build build-research -j
```

If `CUTEST_ROOT` is omitted, CMake also checks
`../research-deps/install`.

## Representative commands

Small implicit-shooting validation against the naive formulation, dense LU, and
IPOPT:

```bash
./build-research/research/global_parameter_kkt_benchmark \
  --problem synthetic \
  --stages 20 \
  --nx 6 \
  --nu 3 \
  --parameters 2 \
  --repeats 11 \
  --ipopt
```

Large implicit-shooting KKT:

```bash
./build-research/research/global_parameter_kkt_benchmark \
  --problem synthetic \
  --stages 2000 \
  --nx 20 \
  --nu 10 \
  --parameters 4 \
  --repeats 9 \
  --ipopt \
  --no-dense-validation
```

Degree-3 direct Radau collocation:

```bash
./build-research/research/global_parameter_kkt_benchmark \
  --problem synthetic \
  --stages 1961 \
  --nx 6 \
  --nu 3 \
  --parameters 4 \
  --collocation-degree 3 \
  --repeats 9 \
  --ipopt \
  --no-dense-validation
```

Four-phase degree-3 Radau collocation:

```bash
./build-research/research/global_parameter_kkt_benchmark \
  --problem synthetic \
  --stages 240 \
  --phases 4 \
  --nx 12 \
  --nu 5 \
  --parameters 8 \
  --collocation-degree 3 \
  --repeats 11 \
  --ipopt \
  --no-dense-validation
```

Heterogeneous three-phase Radau collocation with different `(nx, nu, p_f)`:

```bash
./build-research/research/heterogeneous_phase_kkt_benchmark \
  --phase-nodes 8,7,9 \
  --phase-nx 4,7,3 \
  --phase-nu 2,3,1 \
  --phase-parameters 1,2,1 \
  --global-parameters 2 \
  --collocation-degree 3 \
  --repeats 11
```

Rank-deficient heterogeneous implicit dynamics with automatic inertia
correction:

```bash
./build-research/research/rank_deficient_implicit_kkt_benchmark \
  --phase-nodes 5,4,6 \
  --phase-nx 4,7,3 \
  --phase-nu 2,3,1 \
  --rank-deficiency 2 \
  --parameters 8 \
  --negative-curvature 4 \
  --repeats 11
```

The exact-Hessian mode sets a nonzero adjacent-stage block. The lifted result
remains the validated solution; the current compact rank-aware result is
reported as a diagnostic:

```bash
./build-research/research/rank_deficient_implicit_kkt_benchmark \
  --cross-hessian-scale 0.01
```

Complete nonlinear three-phase Radau-3 SQP, including IPOPT:

```bash
./build-research/research/nonlinear_global_parameter_sqp_benchmark \
  --phase-nodes 5,4,6 \
  --phase-nx 4,7,3 \
  --phase-nu 2,3,1 \
  --parameters 3 \
  --collocation-degree 3 \
  --repeats 5 \
  --full-solvers \
  --ipopt-linear-solver ma57
```

DTOC3-derived quadratic problem with four global parameters:

```bash
./build-research/research/global_parameter_kkt_benchmark \
  --problem dtoc3 \
  --stages 30000 \
  --parameters 4 \
  --repeats 9 \
  --ipopt \
  --no-dense-validation
```

End-to-end FATROP solves:

```bash
./build-research/research/dtoc_fatrop_benchmark \
  --problem dtoc1l --stages 6667 --controls 5 --states 10

./build-research/research/dtoc_fatrop_benchmark \
  --problem dtoc3 --stages 30000

./build-research/research/dtoc_fatrop_benchmark \
  --problem dtoc6 --stages 50000
```

`run_preliminary_benchmarks.sh` runs the complete reported suite
sequentially. Set `BENCH_CPU` to select the logical CPU, or to an empty string
to disable affinity:

```bash
BENCH_CPU=8 ./research/run_preliminary_benchmarks.sh
```

`run_multiphase_sweep.sh` produces a CSV factorial sweep over phase count and
`(nx, nu, p)`. It runs both a fixed-total-node experiment, which isolates the
effect of additional phase boundaries, and a fixed-nodes-per-phase experiment,
which measures scaling as phases are added:

```bash
PHASES_LIST="1 2 4 8" \
NX_LIST="6 20" \
NU_LIST="3 10" \
NP_LIST="0 1 4 8" \
REPEATS=11 \
BENCH_CPU=8 \
./research/run_multiphase_sweep.sh > multiphase.csv
```

IPOPT is disabled in the full sweep by default. Set `RUN_IPOPT=1` for smaller
kernel/context comparisons, `RUN_NATIVE_IPM=1` for the native IPM, or
`RUN_FULL_SOLVERS=1 IPOPT_LINEAR_SOLVER=ma57` for a like-for-like complete
solver comparison. Cases with `p=0` are skipped by the parametric native-IPM
route. See `./research/run_multiphase_sweep.sh --help` for all controls.

`run_heterogeneous_phase_sweep.sh` generates heterogeneous phase patterns and
varies the phase count, global dimension, local dimensions, and transcription:

```bash
PHASES_LIST="1 2 4 8" \
GLOBAL_PARAMETERS_LIST="0 2 8" \
PHASE_PATTERNS="alternating ramp zigzag" \
TRANSCRIPTIONS="shooting radau1 radau2 radau3" \
NODES_PER_PHASE=12 \
REPEATS=7 \
BENCH_CPU=8 \
./research/run_heterogeneous_phase_sweep.sh > heterogeneous.csv
```

Small cases use a dense KKT reference by default. The full sweep disables it
to avoid cubic dense work; set `DENSE_VALIDATION=1` for a deliberately small
grid.

`run_rank_deficient_implicit_sweep.sh` varies phase count, heterogeneous
`(nx, nu)`, global parameter dimension, rank loss, exact-Hessian mode, and
negative-curvature/inertia mode:

```bash
PHASES_LIST="1 2 4 8" \
PARAMETERS_LIST="1 3 6" \
RANK_DEFICIENCY_LIST="0 1 2" \
PHASE_PATTERNS="alternating ramp zigzag" \
MODES="gauss_newton exact_hessian inertia" \
REPEATS=7 \
BENCH_CPU=8 \
./research/run_rank_deficient_implicit_sweep.sh \
  > rank-deficient.csv
```

`run_nonlinear_sqp_sweep.sh` runs the end-to-end nonlinear factorial grid over
phase count, heterogeneous dimension pattern, dimension scale, global
parameter dimension, and Radau degree:

```bash
PHASES_LIST="1 2 4 8" \
PARAMETERS_LIST="0 2 8" \
PHASE_PATTERNS="alternating ramp zigzag" \
DIMENSION_SCALES="1 2" \
TRANSCRIPTIONS="radau1 radau3" \
NODES_PER_PHASE=4 \
MAX_ITERATIONS=80 \
REPEATS=3 \
BENCH_CPU=8 \
./research/run_nonlinear_sqp_sweep.sh \
  > nonlinear-sqp.csv
```

The full sweep disables dense-step validation and complete solvers by default.
Set `DENSE_VALIDATION=1` for small correctness grids or
`RUN_FULL_SOLVERS=1 IPOPT_LINEAR_SOLVER=ma57` to run native FATROP and IPOPT
with the same Gauss--Newton Hessian and the same repeat count.

`run_nonlinear_interval_scoped_sweep.sh` runs the newer exact-Hessian
interval-scope primal-dual experiment.  It covers 1/2/4/8 phases,
Radau-1/Radau-3, heterogeneous states and controls, two-sided nonlinear path
constraints, zero/narrow/wide static-parameter profiles, and the matched IPOPT
solve. `PATH_INEQUALITIES` selects the default count per stage (default 2):

```bash
REPEATS=7 \
CPU=8 \
HSL_LIBRARY=/path/to/libhsl.so \
IPOPT_LINEAR_SOLVER=ma57 \
./research/run_nonlinear_interval_scoped_sweep.sh build-research \
  > nonlinear-interval-scoped.csv
```

`HSL_LIBRARY` may be omitted when the selected IPOPT linear solver is already
discoverable by the system loader.  Every row is an executable equivalence
gate, not only a timing sample: analytic first/second derivatives, the full
primal-dual dense KKT corrector, complementarity, convergence, and solution
agreement with IPOPT must all pass. Pass `--no-factor-reuse` directly to the
executable for the predictor/corrector refactorization ablation.

## Optional CUTEst setup

The measurements in `RESULTS.md` used:

- CUTEst `v2.7.1`;
- SIFDecode commit `d8666ba`;
- SIF commit `ba729ae`.

One reproducible source installation is:

```bash
mkdir -p ../research-deps
git clone https://github.com/ralna/CUTEst.git ../research-deps/CUTEst
git clone https://github.com/ralna/SIFDecode.git ../research-deps/SIFDecode
git clone https://github.com/ralna/SIF.git ../research-deps/SIF

git -C ../research-deps/CUTEst checkout v2.7.1
git -C ../research-deps/SIFDecode checkout d8666ba9552b539de2913647e0faa8f10c26af8b
git -C ../research-deps/SIF checkout ba729aec456b4a38f02917a305d8719b26b4c9fc

python3 -m venv ../research-deps/venv
../research-deps/venv/bin/pip install meson ninja

../research-deps/venv/bin/meson setup \
  ../research-deps/SIFDecode/build \
  ../research-deps/SIFDecode \
  --prefix "$PWD/../research-deps/install"
../research-deps/venv/bin/meson compile \
  -C ../research-deps/SIFDecode/build
../research-deps/venv/bin/meson install \
  -C ../research-deps/SIFDecode/build

../research-deps/venv/bin/meson setup \
  ../research-deps/CUTEst/build \
  ../research-deps/CUTEst \
  --prefix "$PWD/../research-deps/install" \
  -Ddefault_library=shared
../research-deps/venv/bin/meson compile \
  -C ../research-deps/CUTEst/build
../research-deps/venv/bin/meson install \
  -C ../research-deps/CUTEst/build
```

Decode a large DTOC3 case and create the problem-specific shared library:

```bash
case_dir="$PWD/../research-deps/problems/DTOC3_N30000"
mkdir -p "$case_dir"
cd "$case_dir"

../../install/bin/sifdecoder \
  -dp -s 3 -force -param N=30000 \
  ../../SIF/DTOC3.SIF

gfortran -O3 -shared -fPIC \
  -o libDTOC3_N30000.so \
  ELFUN.f GROUP.f RANGE.f
```

The other reported parameter sets are:

```text
DTOC3:  N=50, N=5000, N=30000
DTOC1L: N=1000,NX=5,NY=10
         N=6667,NX=5,NY=10
         N=9091,NX=1,NY=10
DTOC6:  N=5001, N=50000
```

Values absent from the SIF file's predefined list require `-force`.

Run the generic adapters as follows:

```bash
./build-research/research/cutest_probe \
  ../research-deps/problems/DTOC3_N30000/OUTSDIF.d \
  ../research-deps/problems/DTOC3_N30000/libDTOC3_N30000.so

./build-research/research/cutest_ipopt_benchmark \
  ../research-deps/problems/DTOC3_N30000/OUTSDIF.d \
  ../research-deps/problems/DTOC3_N30000/libDTOC3_N30000.so
```

## Validation and timing scope

`global_parameter_kkt_benchmark` checks:

- bordered KKT residuals;
- agreement of the blocked and sequential bordered solutions;
- equality consensus between all naive parameter copies;
- agreement with the naive FATROP solve;
- the same checks across parameter-dependent inter-phase reset/linkage
  equations;
- agreement with a dense pivoted-LU solve when the total KKT dimension is at
  most 2500;
- agreement with IPOPT when `--ipopt` is enabled.

`heterogeneous_phase_kkt_benchmark` checks the same sequential-versus-blocked
agreement and structured residuals for variable phase dimensions. It also
compares against dense pivoted LU when the total KKT dimension is at most
2500.

`rank_deficient_implicit_kkt_benchmark` checks detected transition ranks,
original KKT residuals after postprocessing, dense-reference agreement,
reduced-Hessian regularization, and the full KKT inertia. Its compact and
lifted solutions are required to validate for both zero and nonzero `FuFx`.
Dense forward differences use the scale `1 + ||solution||_inf`, while the
original structured KKT residual is checked independently.

`nonlinear_global_parameter_sqp_benchmark` checks objective and constraint
derivatives with central directional differences, verifies every bordered SQP
step by its full KKT residual, compares the first step with dense full-pivoting
LU on small cases, and optionally solves the same nonlinear program with
IPOPT. Dense validation is performed outside the reported SQP timing.

The bordered and naive KKT timings exclude problem generation and solver
construction. They include numerical normalization, factorization, solves, and
the dense parameter Schur complement. The reported component medians need not
sum exactly to the median total.

The legacy IPOPT column measures `OptimizeTNLP` after `IpoptApplication`
initialization. The quadratic cases converge in one iteration, but those
values are broader than the bordered/naive KKT-only timings. The newer
`--full-solvers` columns instead time complete native FATROP and IPOPT calls,
repeat each solver equally, and report the median.

`dtoc_fatrop_benchmark` and `cutest_ipopt_benchmark` measure complete
optimization calls, but not process startup or SIF decoding.

The reference nonlinear SQP timing includes analytic model evaluation, all
factorizations and parameter Schur solves, merit-function line searches, and
convergence checks. In the current full-solver comparison, native FATROP and
IPOPT use the same analytic functions, sparse first derivatives, and the same
user-supplied Gauss--Newton Hessian blocks.

## Current limitations

- Global parameters are integrated into FATROP's native dimensions, problem
  API, primal-dual Newton step, parameter-bound handling, and restoration
  reduction. Complete primal-dual warm-start input/output is available at the
  `IpAlgorithm` level. What is still missing is broad nonlinear robustness
  qualification, production code generation/AD ergonomics, and a stable
  compatibility layer matching the generic IPOPT-facing application API.
- The phase-arrow path now solves a complete assembled primal-dual barrier
  Newton system, but it is still separate from the ordinary native nonlinear
  builder. `IpAlgorithm` currently flattens sequential phases into one implicit
  chain and selects `PdSolverOrig`; it does not yet automatically partition
  phase-local variables and linkage coordinates, assemble their reduced blocks,
  or select the two-level solver in ordinary and restoration iterations.
- The full-rank implicit solver exposes a multi-RHS API and caches its
  preprocessing. Unstabilized systems with zero adjacent-stage curvature now
  normalize any nonsingular `Jt` once and choose scalar explicit reuse for
  one to three columns or the matrix-valued explicit traversal for wider
  borders. Full-rank unstabilized exact-curvature systems reuse the modified
  factors, but their wide batches are not yet tiled. Stabilized exact-curvature
  responses currently refactor through the modified recursion. The remaining
  small reduced solves use scalar pivoted LU.
- Rank-deficient zero-curvature responses retain scalar factorization reuse.
  Rank-deficient exact-curvature and stabilized responses currently refactor
  through the structured recursion for each RHS; a cached/tiled implementation
  is still missing.
- The KKT microbenchmarks use a linear implicit ODE. The nonlinear SQP
  benchmark adds nonlinear implicit Radau dynamics but uses analytic
  derivatives and a Gauss--Newton objective Hessian rather than an exact
  Lagrangian Hessian or automatic differentiation. The separate
  interval-scoped benchmark does use the exact Lagrangian Hessian and a full
  primal-dual inequality system.
- The standalone heterogeneous benchmark intentionally keeps all global and
  phase-local static variables in one dense border as a correctness oracle.
  The phase-arrow KKT and primal-dual kernels perform the phase-aware
  elimination and are checked against full-system references, but are not yet
  selected by the native nonlinear IPM.
- The interval-scoped predictor-corrector is likewise a standalone research
  driver. It does not yet inherit the production filter, restoration, inertia
  correction, warm-start, or difficult-start machinery of `IpAlgorithm`.
- There is not yet a naive state-augmentation baseline for heterogeneous
  phase-local parameters.
- The compact rank-aware implicit preprocessing is verified for nonzero
  cross-stage Hessian blocks `FuFx` when the residual block is square. The
  universal lifting remains the reference for rectangular residual layouts.
- `ImplicitOcpType` still stores one square dynamics residual block per
  incoming state. The lifting derivation permits a rectangular residual count,
  but that count is not yet exposed by the compact interface.
- The native IPM covers structured inequalities and parameter bounds, but its
  globalization is not yet an IPOPT-grade composite normal/tangential method
  with soft restoration, and there is no comparable degeneracy/infeasibility
  stress suite.
- Radau collocation currently treats `[u, X_1, ..., X_d]` as one dense local
  stage block. It does not yet condense the internal collocation states and
  equations locally. This is the main observed short-horizon Radau-3
  performance gap against MA57.
- The original global-parameter benchmark still generates only `FuFx=0`;
  exact-curvature coverage lives in the rank-deficient benchmark and focused
  dense-reference unit regressions.
- Non-defect adjacent equality rows inside a phase are deliberately declined;
  sequential inter-phase boundary equalities are already represented as
  implicit bridge transitions.
- FATROP is deliberately not connected to PSOPT/LOPT/UCHORT/RocketSystem in
  the current tree. Re-integration must wait for broad real-mission,
  infeasible-start, failure-diagnostic, and mesh-refinement qualification.
- The installed FATROP shared library and headers must be versioned and
  deployed atomically because the public batched KKT API is not ABI-compatible
  with an older library.
- No memory/cache-counter comparison and no same-problem comparison with MOTO,
  acados, or HPIPM has been collected. Results from one CPU and IPOPT+MA57 are
  evidence, not a general performance theorem.
