# Preliminary results: global parameters and direct collocation

Date: 2026-07-17

These are research measurements, not production performance claims. They
evaluate the exactness and cost of a dense global-parameter border around the
FATROP OCP KKT recursion.

**Current scope (2026-07-17):** the experimental LOPT/PSOPT/UCHORT integration
described in later historical sections has been removed, and FATROP is not
wired into RocketSystem. Since those measurements were collected, the
standalone FATROP branch gained native one-copy global-parameter dimensions,
`ParametricImplicitOcpAbstract`, parameter bounds, bordered primal-dual steps,
restoration coverage, full primal-dual warm-start input/output, and a single
native problem API for both `p=0` and `p>0`. New native regressions include an
end-to-end implicit IPM solve plus 204 linearized combinations of
phase count, `(nx,nu,np)`, and collocation degree. Historical LOPT results
remain useful comparison data, not a statement about the current source tree.
A clean build currently registers 165 CTest cases and all 165 pass. The suite
now includes full-rank and rank-deficient exact inter-stage Hessian regressions,
with and without equality-dual stabilization, and checks both the factorizing
solve and a different retained RHS.

## Environment and protocol

- FATROP branch: `research/global-parameter-collocation`
- FATROP base commit: `4587bef`
- CPU: AMD Ryzen 9 9950X3D
- Compiler: GCC 13.3.0
- CMake: 3.28.3
- IPOPT: 3.14.20; legacy Sections 1--11 use MUMPS, historical LOPT tables use
  MA57, and the current full-solver checkpoints below use MA57
- CUTEst: 2.7.1
- Build: `Release`, bundled BLASFEO
- Affinity: logical CPU 8
- Thread limits: `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, `BLIS_NUM_THREADS=1`

Bordered and naive KKT-kernel values are medians after an untimed warm-up. The
legacy IPOPT columns in those tables time `OptimizeTNLP` after application
initialization and are not equivalent to an isolated KKT factorization. The
new full-solver checkpoints repeat both native FATROP and IPOPT equally in the
same process and report their medians. Times are in milliseconds.

## Correctness

### Interval-scoped implicit-OCP factorization

The new interval-scoped prototype keeps phase-local, interface, segment, and
mission-global decision blocks exactly once. Its reduced graph is an interval
graph and is eliminated in increasing-right-endpoint order. Complete implicit
OCP KKT tests match a monolithic dense reference for 1, 2, 4, and 8 phases,
heterogeneous `(nx,nu,p)`, and a rank-deficient implicit case. Separate tests
reconstruct the full primal-dual barrier step for 1, 2, and 4 phases.
The optimized kernels pass dedicated ASan+UBSan runs, and the full Release
suite passes all 165 registered tests.

The controlled end-to-end KKT benchmark gives both methods the same phase
Riccati kernels and the same nonzero response columns. The baseline is the
optimistic naive extension that packs every static variable into one dense
global Schur block. For fixed active width `omega=8`:

| Phases | Scoped dimension | Interval, us | Packed global, us | Packed / interval | KKT residual |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6 | `16.06` | `15.95` | `0.993` | `5.0e-16` |
| 8 | 29 | `179.28` | `178.69` | `0.997` | `9.4e-16` |
| 32 | 113 | `745.79` | `804.51` | `1.079` | `1.3e-15` |
| 64 | 225 | `1555.70` | `2023.76` | `1.301` | `8.8e-15` |
| 128 | 449 | `3942.23` | `7539.10` | `1.912` | `3.7e-15` |
| 256 | 897 | `8826.58` | `37718.34` | `4.273` | `6.0e-14` |

At 128 phases, enlarging `(nx,nu)` and scoped parameter blocks from
`(2,1)/(1,1,1,1)` to `(12..14,6..7)/(8,4,8,8)` increases the measured
full-KKT speedup from `1.60x` to `11.24x`; all solutions agree within
`5.4e-15`, and complete residuals remain below `2.1e-13`.

This is a linear-solver result, not yet an IPOPT+MA57 or nonlinear-iteration
claim. The optimized reduced kernel beats Eigen SparseLU for bounded
`omega=8` through 1024 phases, but crosses behind it once overlapping scopes
make the active width roughly 38 or larger in the tested 256-phase family. See
[INTERVAL_SCOPED_RICCATI.md](INTERVAL_SCOPED_RICCATI.md) for the theorem,
protocol, limitations, and prior-art boundary.

### Nonlinear interval-scoped direct collocation

The interval-scoped path now has a matched nonlinear experiment rather than
only a linear-KKT benchmark.  Each heterogeneous phase is an independent
implicit Radau leaf; a one-copy boundary separator with scope `[f,f+1]`
represents the nonlinear reset between adjacent phases.  Phase, interface,
segment, and global static parameters use the same scope mechanism.  The
structured primal-dual solver and IPOPT adapter share one analytic evaluator,
two-sided nonlinear path bounds, and the exact Lagrangian Hessian, including
dynamics, reset, and inequality curvature.  Each Mehrotra iteration reuses
the affine trajectory/interval factors for its second-order corrector RHS.

Seven-repeat Release medians on pinned logical CPU 8 are:

| Phases | Degree | `omega` | Scoped IPM, ms | IPOPT+MA57, ms | IPOPT / scoped |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 6 | `0.445` | `1.137` | `2.553` |
| 2 | 1 | 13 | `2.141` | `3.142` | `1.468` |
| 4 | 1 | 17 | `5.169` | `6.946` | `1.344` |
| 8 | 1 | 20 | `14.302` | `15.773` | `1.103` |
| 1 | 3 | 6 | `1.153` | `1.545` | `1.340` |
| 2 | 3 | 13 | `5.205` | `5.216` | `1.002` |
| 4 | 3 | 17 | `10.543` | `11.967` | `1.135` |
| 8 | 3 | 20 | `26.357` | `29.332` | `1.113` |

Both solvers converged in all eleven profiles varying `(F,nx,nu,p)` and the
inequality dimension.  Structured primal, dual, complementarity, and full
linear-step residuals stayed below `5.1e-9`, and the exact-Hessian directional
error stayed below `9.8e-11`.  The first second-order corrector agrees with a
dense full primal-dual KKT solve within `2.6e-10`.  Disabling retained-factor
correction on the eight-phase Radau-3 case raises total time from `26.340` to
`35.688 ms`.

The parameter-heavy counterexample is equally important.  With four phases,
Radau-3, `nx=(8,12,6,10)`, `nu=(4,6,3,5)`, static block dimensions
`(6,4,8,12)`, four inequalities per stage, and `omega=58`, scoped IPM took
`80.023 ms` versus `66.968 ms` for IPOPT+MA57.  The method wins or reaches
parity for bounded active width in this sweep, not for arbitrary dense global
coupling.  The script `run_nonlinear_interval_scoped_sweep.sh` reproduces the
matrix; equality and inequality CTests plus representative ASan+UBSan runs
pass.

### Native FATROP global-parameter regressions

The current standalone implementation adds two native matrices beyond the
historical benchmark-only checks:

- 96 implicit multiphase Newton systems over `phases={1,2,4,8}`,
  `nx={1,3,6}`, `nu={1,2}`, and `np={0,1,3,6}`;
- 108 implicit direct-collocation systems over `phases={1,2,4}`,
  Radau-like local degrees `{1,2,3}`, `nx={1,3}`, `nu={1,2}`, and
  `np={0,1,3}`.

Every case solves through `PdSolverOrig<ImplicitOcpType>` with the parameter
vector stored once, and the complete primal-dual residual is below `2e-8`.
Separate tests cover dense-reference bordered solves, explicit and implicit
native PD systems, parameter bounds, and restoration reduction. A complete
native IPM regression solves an implicit transfer problem from an infeasible
terminal guess and recovers `u=0.4`, `p=0.6`. A second end-to-end regression
snapshots `(x,s,lambda,z_L,z_U,mu)`, reuses it after reset, and verifies that
the one-copy global parameter is retained. Initializer tests cover owned-copy
semantics, dimensions/NaN rejection, strict slack interiority, and positive
bound-dual repair.

### Current like-for-like full-solver checkpoints

These measurements supersede the earlier statement that a fair native
full-solver benchmark was still missing. Both solvers receive the same
generated NLP, analytic callbacks, Hessian model, stopping tolerance, and
repeat count.

The quadratic 96-case matrix uses 80 total stages,
`phases={1,2,4,8}`, `nx={3,6}`, `nu={2,4}`, `p={1,3,6}`, shooting and
Radau-3, five repeats, and IPOPT+MA57. Both methods converge in one nonlinear
iteration. The maximum primal/multiplier difference is `8.882e-15`.

| Transcription | Cases FATROP faster | Native FATROP mean, ms | IPOPT+MA57 mean, ms | Summed IPOPT/FATROP | Per-case range |
| --- | ---: | ---: | ---: | ---: | ---: |
| shooting | 48 / 48 | `0.5719` | `0.7763` | `1.357` | `1.139--1.825` |
| Radau-3 | 48 / 48 | `1.2442` | `2.1066` | `1.693` | `1.238--2.355` |

The nonlinear short-mesh matrix uses `phases={1,2,4,8}`, three heterogeneous
dimension patterns, two dimension scales, `p={1,3,6}`, Radau-1/Radau-3,
four/five nodes per phase, and three repeats. IPOPT and native FATROP use the
same supplied Gauss--Newton Hessian. Both converge in all 144 cases. The
largest native constraint and dual residuals are `1.097e-12` and `4.736e-8`;
the largest differences from the independent reference SQP are `4.129e-6` in
the trajectory and `3.258e-9` in `p`.

| Slice | FATROP faster | Summed IPOPT/FATROP | Interpretation |
| --- | ---: | ---: | --- |
| all short-mesh cases | 113 / 144 | `0.909` | FATROP is about 10% slower in summed time |
| Radau-1 | -- | `1.200` | FATROP is about 20% faster |
| Radau-3 | -- | `0.864` | MA57 is about 16% faster |

The Radau-3 loss occurs when a very short horizon contains large dense local
stage blocks. FATROP currently treats `[u,X_1,...,X_d]` as one dense control
block instead of locally condensing collocation variables.

A long-mesh Radau-3 checkpoint uses 20/21 nodes per phase, the alternating
dimension pattern, `phases={1,2,4,8}`, `p={1,3,6}`, and three repeats. FATROP
wins all 12 cases; individual ratios range from `1.164` to `6.897`. Summed
native time is `485.710 ms` versus `1862.490 ms` for IPOPT+MA57, a ratio of
`3.835`. This is the expected Riccati crossover: enough stages amortize the
dense local work and expose MA57's whole-NLP sparse-factorization cost.

### Historical KKT-only fixed-horizon crossover sample

A fresh 96-case Release run used 80 total stages, phases `{1,2,4,8}`,
`nx={3,6}`, `nu={2,4}`, `np={1,3,6}`, shooting and degree-3 Radau, five
kernel repeats, one pinned CPU, and IPOPT 3.14.20 with dynamically loaded
MA57. Values below are means of each twelve-case phase/transcription slice:

| Phases / transcription | Bordered FATROP, ms | Naive copied-`p` FATROP, ms | IPOPT+MA57, ms | Naive / bordered | IPOPT / bordered |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 / shooting | 0.099 | 0.176 | 0.966 | 1.74 | 10.02 |
| 2 / shooting | 0.098 | 0.175 | 0.982 | 1.73 | 10.11 |
| 4 / shooting | 0.098 | 0.176 | 0.980 | 1.74 | 10.13 |
| 8 / shooting | 0.097 | 0.174 | 0.970 | 1.75 | 10.18 |
| 1 / Radau-3 | 0.443 | 0.312 | 2.593 | 0.69 | 6.30 |
| 2 / Radau-3 | 0.440 | 0.308 | 2.616 | 0.68 | 6.34 |
| 4 / Radau-3 | 0.430 | 0.303 | 2.603 | 0.69 | 6.44 |
| 8 / Radau-3 | 0.410 | 0.287 | 2.620 | 0.68 | 6.69 |

The one-copy border beats copied parameters in 45/48 shooting cases, but in
0/48 degree-3 Radau cases: the current implicit response/preprocessing path
cost is still decisive for collocation. The IPOPT column is a complete
one-iteration `OptimizeTNLP` call, whereas the two FATROP columns isolate KKT
work, so its 6--10x ratio is context, not a fair end-to-end speed claim. The
like-for-like native results above are the current solver-level comparison.

For a small implicit-shooting case (`N=20`, `nx=6`, `nu=3`, `p=2`):

| Check | Maximum absolute difference/residual |
| --- | ---: |
| Bordered KKT residual | `3.77e-16` |
| Naive KKT residual | `3.82e-16` |
| Bordered vs naive primal | `7.35e-17` |
| Bordered vs naive multipliers | `7.91e-16` |
| Bordered vs naive parameters | `6.94e-18` |
| Parameter-copy consensus | `1.39e-17` |
| Bordered vs dense pivoted LU | `2.61e-15` |
| Bordered vs IPOPT primal | `6.94e-17` |

All reported large cases also pass the executable's `5e-7` validation
tolerance. Typical residuals are between `1e-15` and `1e-12`.

## 1. Full-rank implicit shooting

The next-state Jacobian is non-identity and must be normalized before the
ordinary FATROP recursion.

| N | nx | nu | p | Base primal / eq. | Border total | Normalize | Factor | p RHS | Schur | Naive implicit | IPOPT | Naive / border | IPOPT / border |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 20 | 10 | 4 | 5,990 / 3,980 | 4.176 | 2.769 | 0.783 | 0.580 | 0.059 | 9.349 | 23.795 | 2.24x | 5.70x |
| 2,000 | 20 | 10 | 4 | 59,990 / 39,980 | 35.429 | 18.261 | 8.133 | 8.455 | 0.593 | 73.270 | 565.864 | 2.07x | 15.97x |

Result: the one-copy border wins over naive parameter copies in the tested
implicit formulation. It avoids enlarging every implicit state and its
full-rank preprocessing.

## 2. Direct Radau collocation

These cases use degree-3 Radau collocation. The fair naive baseline is the
explicit `OcpType` recursion after normalization to
`dF/dx[k+1] = -I`.

### Dependence on the number of global parameters

`N=200`, `nx=6`, `nu=3`, base KKT dimensions 5,379 primal and 4,776 equality
constraints:

| p | Border | Factor | p RHS | Schur | Naive explicit | IPOPT | Naive / border |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.362 | 1.166 | 0.179 | 0.008 | 1.186 | 38.290 | 0.871 |
| 4 | 1.898 | 1.172 | 0.694 | 0.023 | 1.301 | 40.132 | 0.685 |
| 8 | 2.636 | 1.185 | 1.394 | 0.044 | 1.479 | 41.799 | 0.561 |

The dense Schur complement is cheap at these values of `p`; the sequential
full-horizon parameter sweeps dominate.

### Approximately 100,000 KKT rows and columns

All cases use `p=4`. The stage count is adjusted so that the base primal plus
equality dimensions are approximately 100,000.

| N | nx | nu | Base primal / eq. | Border | Naive explicit | IPOPT | Naive / border | IPOPT / border |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,961 | 6 | 3 | 52,926 / 47,040 | 35.513 | 20.079 | 470.390 | 0.565 | 13.25x |
| 589 | 20 | 10 | 52,940 / 47,040 | 111.130 | 85.115 | 716.062 | 0.766 | 6.44x |
| 236 | 50 | 25 | 52,925 / 47,000 | 509.956 | 463.418 | 1,000.294 | 0.909 | 1.96x |

Result: the current border is slower than naive FATROP state augmentation for
all three collocation shapes. The gap narrows as the physical state grows,
because increasing the state by four parameters becomes relatively more
expensive, but the present sequential multi-solve scheme still does not cross
over.

## 3. DTOC3-derived problem with global parameters

The standard DTOC3 problem has `nx=2`, so it is deliberately unfavorable to
repeated full-horizon solves. Global parameters enter every dynamics block and
the objective.

`N=30,000`, base dimensions 89,999 primal and 60,000 equalities:

| p | Naive primal / eq. | Border | Naive explicit | IPOPT | Naive / border | IPOPT / border |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 89,999 / 60,000 | 19.822 | 19.675 | 119.161 | 0.993 | 6.01x |
| 1 | 119,999 / 89,999 | 34.968 | 19.410 | 361.713 | 0.555 | 10.34x |
| 4 | 209,999 / 179,996 | 73.558 | 28.084 | 920.169 | 0.382 | 12.51x |

Even though four copied parameters more than double the naive problem
dimensions, one larger recursion is 2.62 times faster than five sequential
recursion sweeps. Both structured approaches remain much faster than the
corresponding one-iteration IPOPT run, with the timing-scope caveat stated
above.

## 4. CUTEst locality

CUTEst's native variable ordering hides the local OCP structure. The probe
recognizes DTOC variable names and computes the largest constraint column span
after a stage ordering.

| Problem | n | m | nnz(J) | nnz(H) | Native max span | Stage-ordered max span |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DTOC3, N=30,000 | 89,999 | 59,998 | 209,993 | 89,997 | 89,999 | 6 |
| DTOC1L, N=6,667, NX=5, NY=10 | 100,000 | 66,660 | 553,278 | 100,000 | 66,675 | 30 |
| DTOC1L, N=9,091, NX=1, NY=10 | 100,000 | 90,900 | 427,230 | 100,000 | 90,911 | 22 |
| DTOC6, N=50,000 | 99,999 | 49,999 | 149,997 | 149,997 | 50,001 | 4 |

This confirms that the selected CUTEst cases have strong OCP locality even
though a generic sparse NLP adapter does not expose stages to the solver.

## 5. End-to-end FATROP versus generic CUTEst/IPOPT

The FATROP implementations reproduce the SIF formulations directly through
`OcpAbstract`; IPOPT receives the decoded CUTEst model through a generic sparse
TNLP adapter. Both use exact Hessians.

FATROP adds the fixed initial state as equality constraints. CUTEst represents
the same values as fixed bounds, so their `m` values differ by `nx`.

| Problem | Size | FATROP n / eq. | CUTEst n / m | Iter. F/I | FATROP | IPOPT | IPOPT / FATROP | Objective F/I |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DTOC1L | medium | 14,995 / 10,000 | 14,995 / 9,990 | 6/6 | 12.180 | 121.851 | 10.00x | 125.338129736 |
| DTOC1L | 100k/67k eq. | 100,000 / 66,670 | 100,000 / 66,660 | 6/6 | 107.970 | 628.344 | 5.82x | 836.220228484 |
| DTOC1L | 100k/91k eq. | 100,000 / 90,910 | 100,000 / 90,900 | 7/14 | 136.544 | 1,398.025 | 10.24x | 40.9402410631 / 40.9402410625 |
| DTOC3 | medium | 14,999 / 10,000 | 14,999 / 9,998 | 1/1 | 7.787 | 26.361 | 3.38x | 235.262481035 |
| DTOC3 | ~100k | 89,999 / 60,000 | 89,999 / 59,998 | 1/1 | 70.676 | 113.493 | 1.61x | 235.278942674 / 235.278942679 |
| DTOC6 | medium | 10,001 / 5,001 | 10,001 / 5,000 | 11/11 | 32.434 | 67.957 | 2.10x | 134850.616259 |
| DTOC6 | ~100k | 99,999 / 50,000 | 99,999 / 49,999 | 13/13 | 591.438 | 626.114 | 1.06x | 2259344.28645 |

Objectives agree to the shown precision, apart from the final few digits of
the 100k/91k DTOC1L and large DTOC3 solves. Maximum equality violations range
from approximately `2e-16` to `2e-11`.

The end-to-end table also shows why KKT microbenchmarks are insufficient:
FATROP's advantage ranges from about 1.06x to 10.24x once function evaluation,
globalization, convergence checks, and all iterations are included.

## 6. Multiphase sweep

The synthetic benchmark now supports true phase boundaries. Each nonterminal
phase ends with a parameter-dependent reset/linkage equation connecting its
terminal state to a separate initial state of the next phase. The values below
use common `nx` and `nu` within a problem, but phase-dependent dynamics,
integration steps, and reset maps.

### Fixed total number of nodes

These cases keep `N=240`, `nx=6`, `nu=3`, and `p=4` fixed. Increasing the phase
count replaces ordinary integration transitions by linkages, so this experiment
isolates phase-boundary overhead instead of increasing the total problem size.
Values are medians of 21 timed solves.

| Phases | Shooting border | Shooting naive | Naive / border | Radau-3 border | Radau-3 naive | Naive / border |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.405 | 0.720 | 1.78x | 2.189 | 1.502 | 0.686 |
| 2 | 0.408 | 0.729 | 1.79x | 2.141 | 1.481 | 0.692 |
| 4 | 0.406 | 0.733 | 1.81x | 2.128 | 1.479 | 0.695 |
| 8 | 0.407 | 0.724 | 1.78x | 2.097 | 1.450 | 0.691 |

Result: adjacent phase linkages do not introduce a phase-count-dependent dense
cost. At fixed total node count, one through eight phases have essentially the
same structured-solver timing. The small Radau decrease is explained by phase
endpoint stages having no control or collocation variables.

On the same four cases, IPOPT took approximately `4.28--4.35 ms` for shooting
and `44.5--45.5 ms` for Radau-3. It converged in one iteration and agreed with
the bordered solution to between `8e-16` and `1.2e-13`.

### Fixed nodes per phase

Here every phase has 60 nodes, so total problem size grows with the number of
phases.

| Phases | Total nodes | Shooting border | us/node | Shooting naive | Radau-3 border | us/node | Radau-3 naive |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 60 | 0.097 | 1.62 | 0.174 | 0.511 | 8.52 | 0.362 |
| 2 | 120 | 0.196 | 1.63 | 0.359 | 1.054 | 8.78 | 0.735 |
| 4 | 240 | 0.420 | 1.75 | 0.745 | 2.288 | 9.54 | 1.630 |
| 8 | 480 | 1.048 | 2.18 | 1.858 | 5.406 | 11.26 | 4.188 |

Both structured formulations scale approximately linearly with the total
number of nodes. The per-node increase at 480 nodes is consistent with cache
and memory effects, not a new asymptotic dependence on the number of phases.

### Dependence on the global parameter dimension

`N=240`, four phases, `nx=6`, and `nu=3`:

| p | Shooting border | Shooting p-RHS | Shooting naive | Naive / border | Radau-3 border | Radau p-RHS | Radau-3 naive | Naive / border |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.236 | 0.000 | 0.355 | 1.50x | 1.334 | 0.000 | 1.305 | 0.979 |
| 1 | 0.282 | 0.036 | 0.453 | 1.60x | 1.559 | 0.205 | 1.386 | 0.889 |
| 4 | 0.408 | 0.142 | 0.730 | 1.79x | 2.248 | 0.815 | 1.564 | 0.696 |
| 8 | 0.621 | 0.295 | 1.439 | 2.32x | 3.191 | 1.633 | 1.985 | 0.622 |
| 16 | 1.202 | 0.623 | 4.423 | 3.68x | 5.113 | 3.264 | 2.883 | 0.564 |

The parameter right-hand-side time grows nearly linearly with `p`. This helps
implicit shooting because copying `p` into every state is even more expensive,
but it is exactly the bottleneck that makes the current Radau border lose.

### Dependence on state and control dimensions

`N=240`, four phases, and `p=4`:

| nx | nu | Shooting naive / border | Radau-3 naive / border |
| ---: | ---: | ---: | ---: |
| 4 | 1 | 1.76x | 0.592 |
| 4 | 10 | 1.52x | 0.644 |
| 12 | 1 | 2.10x | 0.846 |
| 12 | 10 | 1.95x | 0.849 |
| 20 | 1 | 2.19x | 0.817 |
| 20 | 10 | 2.06x | 0.797 |
| 40 | 1 | 2.99x | 0.873 |
| 40 | 10 | 2.93x | 0.881 |

The bordered implicit-shooting solve wins throughout this grid. For Radau, the
border gets closer as the physical state becomes larger, because adding four
copied state components becomes relatively more costly, but it still does not
cross over. The tested range was `nx in {4,12,20,40}` and
`nu in {1,5,10}`; every case passed the KKT validation tolerance.

## 7. Blocked multi-right-hand-side Riccati traversal

The solver now has a matrix-valued `solve_rhs_batch` path. It reuses the
existing FATROP factorization and carries all parameter-border columns through
one backward/forward horizon traversal using BLASFEO GEMM and TRSM kernels.
The sequential vector path remains in the benchmark as an independent
reference.

During validation, the blocked implementation exposed a pre-existing bug in
the offset matrix form of `PermutationMatrix::apply_inverse_on_rows`: pivot
swaps were replayed in forward instead of reverse order. This was invisible
for identity/simple permutations but corrupted Radau stages with several
eliminated equality constraints. The row and column inverse operations now
replay swaps in reverse order and have dedicated regression tests.

Correctness grids:

- 576 degree-3 cases covering `phases in {1,2,4,8}`,
  `nx in {2,6,12}`, `nu in {1,3,6}`, `p in {0,1,4,8}`, shooting
  and Radau, with both fixed-total-node and fixed-nodes-per-phase scaling;
- 288 additional Radau degree-1/2 cases covering the same phase counts,
  `nx in {2,6,12}`, `nu in {1,3}`, and `p in {0,4,8}`;
- a randomized unit-test problem with stage equalities, inequalities, and
  three simultaneous right-hand sides.

Across the 576-case degree-3 grid, the largest blocked KKT residual was
`1.34e-14`; the largest blocked-versus-sequential differences were
`1.53e-16` in the primal variables, `8.88e-15` in the multipliers, and
`5.00e-16` in the global parameters. The degree-1/2 grid produced comparable
maxima.

The following table reports averages over 12 shapes for each `p`:
`N=160`, `phases in {1,4,8}`, `nx in {6,12}`, and `nu in {3,6}`.
Each case is the median of seven pinned-CPU runs. A ratio above one means the
blocked path is faster.

| p | Shooting p-RHS speedup | Shooting total speedup | Radau-3 p-RHS speedup | Radau-3 total speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.402x | 0.870x | 0.544x | 0.928x |
| 4 | 1.299x | 1.076x | 1.413x | 1.105x |
| 8 | 1.726x | 1.228x | 1.655x | 1.218x |
| 16 | 0.984x | 1.046x | 1.362x | 1.189x |
| 32 | 0.958x | 1.070x | 1.370x | 1.247x |

Result: the fused traversal removes a substantial part of the sequential
parameter-sweep bottleneck for moderate `p`, especially around `p=4--8`.
One-column batches lose because matrix workspace setup dominates. Very wide
shooting batches stop improving, indicating that persistent/tiled workspace
and cache-aware blocking are the next kernel optimizations; Radau retains a
roughly `1.36x` right-hand-side gain at `p=16--32`.

## 8. Heterogeneous phases and phase-local parameters

The separate heterogeneous benchmark assigns independent `nx_f`, `nu_f`, and
`p_f` dimensions to every phase and one shared global dimension `p_g`.
Ordinary stage functions depend only on `(p_g, p_f)`. Adjacent reset maps can
depend on `(p_g, p_f, p_(f+1))` and map between different state dimensions.
The incoming-state block remains `-I`, so these rectangular linkages are full
row rank.

The validation sweep contains 144 cases:

```text
phases       in {1, 2, 4, 8}
patterns     in {alternating, ramp, zigzag}
global p     in {0, 2, 8}
transcription in {shooting, Radau-1, Radau-2, Radau-3}
```

The patterns vary every phase's state, control, and local-parameter dimension,
including zero-control and zero-local-parameter phases. Across the full sweep:

| Check | Largest value |
| --- | ---: |
| Sequential KKT residual | `9.08e-15` |
| Blocked KKT residual | `1.08e-14` |
| Blocked vs sequential primal | `6.25e-17` |
| Blocked vs sequential multipliers | `6.22e-15` |
| Blocked vs sequential parameters | `9.71e-17` |

Dense validation was also run on representative shooting and Radau degree
1--3 cases. This included a four-phase case with

```text
nx = 1:8:2:10,  nu = 0:4:1:0,
p_g = 0,        p_f = 0:3:1:2,
```

whose degree-3 solution agreed with the full dense KKT solve to `4.80e-14`.
An eight-phase degree-2 case with 21 total static variables also passed with a
blocked residual of `2.39e-15`.

The following averages summarize the 144-case pinned-CPU sweep with 11 timed
repetitions per case. A speedup above one means the blocked traversal is faster
than sequential parameter columns.

| Transcription | Phases | Cases | Total speedup | Parameter-RHS speedup |
| --- | ---: | ---: | ---: | ---: |
| Shooting | 1 | 9 | `1.006x` | `1.003x` |
| Shooting | 2 | 9 | `1.071x` | `1.104x` |
| Shooting | 4 | 9 | `1.353x` | `1.534x` |
| Shooting | 8 | 9 | `1.447x` | `1.629x` |
| Radau 1--3 | 1 | 27 | `1.041x` | `1.073x` |
| Radau 1--3 | 2 | 27 | `1.083x` | `1.182x` |
| Radau 1--3 | 4 | 27 | `1.242x` | `1.529x` |
| Radau 1--3 | 8 | 27 | `1.438x` | `1.802x` |

The correctness result is stronger than the performance result: one dense
border in

```text
p_total = p_g + sum_f p_f
```

is exact, but its final Schur complement is generally dense after trajectory
elimination. It therefore has cubic factorization cost in `p_total` and does
not exploit the original phase locality. The next scalable construction is a
block-arrow recursion that eliminates each `p_f` with its phase and leaves
only `p_g` as the persistent dense border. The derivation and benchmark
contract are recorded in
[HETEROGENEOUS_PHASE_PARAMETERS.md](HETEROGENEOUS_PHASE_PARAMETERS.md).

## 9. Rank-deficient implicit dynamics and inertia

The rank-deficient benchmark uses heterogeneous multiphase dimensions,
prescribes a rank loss in every ordinary transition and phase linkage, and
adds one-copy global parameters. It compares:

- compact rank-aware `ImplicitOcpType` preprocessing;
- a universal lifting with a local copy of each incoming state;
- dense full KKT solves on representative small cases.

The lifting turns the implicit residual into a stage equality and imposes the
copy with an explicit transition. It therefore retains the OCP chain for
singular or rectangular Jacobians and makes an exact adjacent-stage Hessian
block local.

The solver defects fixed around this sweep include:

- preprocessing now restores the original state/control/equality partition
  before every base or right-hand-side solve;
- terminal preprocessing no longer accesses `number_of_states[K]` and
  `number_of_controls[K]`;
- transformed adjacent-stage Hessian stripes now propagate through every
  backward stage and retained-RHS affine/multiplier terms;
- stabilized rank-deficient promotion keeps hard dynamics rows outside the
  path-equality damping diagonal.

Active regressions perform different retained-RHS solves for full-rank and
rank-deficient systems, stabilized and unstabilized, and verify normalized
original-KKT residuals. Dense forward differences in the sweep are normalized
by `1 + ||solution||_inf`; this avoids rejecting two mutually agreeing
structured solutions solely because an ill-conditioned dense LU reference has
a larger absolute forward error.

### Correctness grid

The pinned-CPU sweep contains 324 cases:

```text
phases           in {1, 2, 4, 8}
dimension pattern in {alternating, ramp, zigzag}
global p         in {0, 2, 8}
rank deficiency  in {0, 1, 2}
mode             in {Gauss--Newton, exact Hessian, inertia}
```

The table below is the pre-fix snapshot that exposed the compact-recursion
defect. The Gauss--Newton mode has `FuFx=0`. Exact-Hessian mode uses
`FuFx` scale `0.01`. Inertia mode subtracts `4 I` from the generated Hessian
and applies the independently computed uniform correction needed to give the
reduced Hessian a `0.1` eigenvalue margin.

| Mode | Largest compact residual | Largest lifted residual | Largest compact/lifted difference | Largest regularization |
| --- | ---: | ---: | ---: | ---: |
| Gauss--Newton | `4.30e-12` | `2.31e-12` | `1.64e-11` | `0` |
| Inertia corrected | `9.31e-13` | `2.49e-13` | `3.18e-12` | `1.10504` |
| Exact cross Hessian | `1.84e-1` | `2.07e-12` | `4.85` | `0` |

For the first two modes, both structured formulations agree with one another
and the dense KKT reference. Every inertia case has the expected base inertia

```text
(n_primal, n_constraints, 0)
```

and the parameter-augmented inertia

```text
(n_primal + p, n_constraints, 0).
```

The old exact-Hessian result was decisive: it showed that the compact
implementation was wrong even when the next-state Jacobian was full rank,
while the lifted formulation remained exact. On the default three-phase case,
the lifted
dense-reference difference is `3.10e-11`; its structured residual is
`1.43e-13`, versus `5.66e-2` for the compact path.

That defect has since been repaired. Focused dense-reference regressions now
pass for full-rank and rank-deficient implicit dynamics, stabilized and
unstabilized, including a different retained RHS. A validation rerun of all
324 combinations with dense references and one repeat also passes. Consequently
the `Exact cross Hessian` row above is historical failure evidence, not a
current solver result; a pinned seven-repeat timing sweep is still needed
before publishing replacement timings for that row.

### Kernel timing

The table averages 27 shapes per row with seven timed repetitions per case.
`lifted / compact` below one means that the larger lifted formulation is
faster.

| Mode | Phases | Compact ms | Lifted ms | Lifted / compact |
| --- | ---: | ---: | ---: | ---: |
| Gauss--Newton | 1 | `0.0487` | `0.0162` | `0.433` |
| Gauss--Newton | 2 | `0.1012` | `0.0345` | `0.443` |
| Gauss--Newton | 4 | `0.2330` | `0.0760` | `0.432` |
| Gauss--Newton | 8 | `0.4884` | `0.1578` | `0.425` |
| Exact Hessian | 1 | `0.0481` | `0.0157` | `0.431` |
| Exact Hessian | 2 | `0.1010` | `0.0341` | `0.438` |
| Exact Hessian | 4 | `0.2383` | `0.0782` | `0.431` |
| Exact Hessian | 8 | `0.4710` | `0.1551` | `0.431` |
| Inertia | 1 | `0.0495` | `0.0162` | `0.428` |
| Inertia | 2 | `0.0993` | `0.0331` | `0.435` |
| Inertia | 4 | `0.2326` | `0.0764` | `0.431` |
| Inertia | 8 | `0.4768` | `0.1535` | `0.423` |

The timing is not evidence that full lifting is asymptotically superior. The
compact prototype currently repeats rank decomposition and all preprocessing
for every parameter right-hand side, whereas the lifted problem directly
reuses the ordinary FATROP factorization. For the default problem, lifting
changes the base dimensions from `93/62` primal/equality variables to
`155/124`, yet still wins because the compact repeated-preprocessing overhead
dominates.

The remaining engineering target is to cache/tile the rank-aware and
stabilized exact-curvature transformations across right-hand sides and rerun
the compact-versus-lifted crossover study. The algebra and inertia conditions are detailed in
[RANK_DEFICIENT_IMPLICIT.md](RANK_DEFICIENT_IMPLICIT.md).

## 10. Reference nonlinear multiphase SQP

This section records the earlier standalone equality-SQP experiment and its
historical limited-memory IPOPT comparison. It remains an independent
correctness oracle. The current native-IPM versus matched-Hessian IPOPT
results are reported near the top of this document.

The final experiment embeds the global-parameter border in a complete
equality-constrained Gauss--Newton SQP loop. The generated NLP uses nonlinear
state- and parameter-dependent mass matrices, nonlinear implicit Radau
collocation residuals, heterogeneous phases, and nonlinear rectangular reset
maps.

At every iteration, the trajectory KKT block is factored once, all global
parameter columns are propagated by the blocked multi-right-hand-side
traversal, and a dense `p x p` Schur complement is solved. An `l1` merit
function provides globalization.

### Independent correctness checks

For the default three-phase Radau-3 problem:

| Check | Error |
| --- | ---: |
| Objective directional derivative | `2.32e-11` |
| Constraint directional derivative | `4.42e-11` |
| Largest bordered linear-step residual | `3.56e-16` |
| First-step difference from dense full KKT | `3.33e-16` |

The nonlinear solve converges in ten iterations to a constraint infinity norm
of `7.64e-15` and a dual infinity norm of `5.36e-9`.

### 144-case factorial sweep

The pinned-CPU sweep covers:

```text
phases             in {1, 2, 4, 8}
dimension pattern  in {alternating, ramp, zigzag}
dimension scale    in {1, 2}
global p           in {0, 2, 8}
Radau degree       in {1, 3}
```

All 144 cases converged. The largest residuals over the complete grid are:

```text
constraint residual   8.62e-13
dual residual         9.93e-9
linear-step residual  6.01e-16
```

The largest case has 1914 trajectory variables, 1748 equality constraints,
eight phases, and eight global parameters. It converged in 55 iterations.

Average end-to-end timings over 18 dimension/parameter combinations in each
row are:

| Phases | Radau degree | SQP ms | KKT ms | Parameter RHS ms | Iterations |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | `0.365` | `0.157` | `0.030` | `5.39` |
| 1 | 3 | `2.630` | `1.826` | `0.139` | `8.72` |
| 2 | 1 | `1.048` | `0.396` | `0.094` | `8.89` |
| 2 | 3 | `7.637` | `4.830` | `0.489` | `16.28` |
| 4 | 1 | `2.879` | `1.089` | `0.256` | `10.50` |
| 4 | 3 | `25.750` | `17.125` | `1.688` | `20.61` |
| 8 | 1 | `5.687` | `2.070` | `0.516` | `10.61` |
| 8 | 3 | `49.897` | `31.972` | `3.473` | `20.67` |

The phase-count trend includes a growing total number of nodes, because every
phase has four or five nodes. Both model evaluation and structured KKT work
therefore grow with the full problem size; the table is not a fixed-size
phase-boundary experiment.

The global parameter dimension affects both the linear algebra and, on these
generated problems, the nonlinear iteration count:

| Degree | p | SQP ms | KKT ms | Parameter RHS ms | Iterations |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | `1.159` | `0.609` | `0` | `8.08` |
| 1 | 2 | `1.985` | `0.882` | `0.214` | `8.33` |
| 1 | 8 | `4.341` | `1.294` | `0.458` | `10.13` |
| 3 | 0 | `14.863` | `11.570` | `0` | `15.58` |
| 3 | 2 | `17.301` | `12.145` | `1.080` | `15.04` |
| 3 | 8 | `32.272` | `18.100` | `3.263` | `19.08` |

### Representative IPOPT comparison

In this earlier protocol, twelve cases were also solved by IPOPT with the same
analytic functions and a sparse Jacobian pattern, but IPOPT used its
limited-memory Hessian approximation.
The objective difference is at most `1.1e-13`, the largest IPOPT constraint
residual is `4.3e-13`, and the largest parameter difference is `2.7e-7`.

For the six `p=8` cases:

| Phases | Degree | Trajectory variables | SQP ms | IPOPT ms | IPOPT / SQP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 34 | `0.100` | `6.822` | `68.4x` |
| 1 | 3 | 58 | `0.497` | `12.723` | `25.6x` |
| 4 | 1 | 284 | `2.253` | `90.620` | `40.2x` |
| 4 | 3 | 500 | `8.857` | `182.991` | `20.7x` |
| 8 | 1 | 432 | `4.231` | `280.368` | `66.3x` |
| 8 | 3 | 752 | `12.496` | `327.590` | `26.2x` |

Across all twelve cases, the measured range is `6.1x--85.7x`. This is a
favorable equality-constrained benchmark for the specialized method, not a
general solver ranking. The research SQP has no inequality barrier,
restoration phase, or production-level safeguards, while IPOPT provides all
of that general infrastructure.

The formulation and verification contract are detailed in
[NONLINEAR_GLOBAL_PARAMETER_SQP.md](NONLINEAR_GLOBAL_PARAMETER_SQP.md).

## 11. Initial LOPT integration checkpoint (historical)

The bordered implicit kernel was exercised through the actual LOPT
transcription and driver interfaces rather than only through generated
research models. That integration has since been removed. This section records
the first equality-only checkpoint and is retained for chronology; Section 12
supersedes its historical capability and timing statements.

The initial `FatropEqualitySqpSolver` accepts equality-constrained
trapezoidal direct collocation with fixed or unbounded variables. It assembles
LOPT's exact objective gradient, constraint Jacobian, and Lagrangian Hessian
into `ImplicitGlobalParameterKktSolver`. Global phase parameters and free
times remain one-copy border variables. Nodal controls are carried as FATROP
states with synthetic next-control consensus equations, and independent
phase chains are joined by algebraically transparent consensus bridges.

### Numerical validation

A scalar convex problem

```text
xdot = u + p
min integral(u^2 + 0.25 p^2) dt
x(0) = 0, x(1) = 1
```

has the analytic solution `u=0.2`, `p=0.8`, and objective `0.2`.
The LOPT/FATROP backend reproduces all three values and agrees with IPOPT on
the same transcribed NLP.

The heterogeneous validation matrix is:

```text
1 phase:  (nx,nu,np,N) = (2,1,2,2)
2 phases: (2,1,2,2), (3,2,1,3)
4 phases: (1,0,2,2), (2,1,1,3), (3,2,4,2), (4,3,2,2)
```

All cases converge, remain below the test's `2e-7` constraint tolerance, and
agree with IPOPT within `2e-6` in objective. The four-phase case includes a
control-free phase and changes every one of `nx`, `nu`, and `np`.

The production-route tests additionally establish that:

- LOPT scaling is applied before the structured backend and correctly undone
  afterward;
- a supported problem invokes FATROP without invoking the generic backend;
- finite variable bounds cause a pre-numerical decline and exactly one IPOPT
  fallback call;
- Hermite--Simpson is recognized as chain-local but declined until
  interior-stage condensation is implemented.

The comparison used IPOPT 3.14.20 with MUMPS. No IPOPT+MA57 timing has yet
been collected for this new LOPT backend.

### Singular trajectory block and stabilization

The dimension matrix exposed an important bordered-factorization condition.
For example, with two physical states, one control, and two static parameters,
the trajectory-only KKT block can be rank deficient even though the complete
bordered KKT matrix is regular: parameter variations are required to satisfy
independent endpoint conditions.

The public kernel now accepts an equality-dual diagonal `D_eq`. A focused
dense-reference test constructs duplicate trajectory constraint rows whose
independence is restored by the parameter border. The stabilized implicit
bordered solve matches the complete dense KKT solution and residual within
`2e-10`.

FATROP's current stabilized `solve_rhs` reuse did not reproduce that KKT
action. The correctness-first implementation therefore refactors the
trajectory system for every border column when `D_eq` is active. Its present
linear-algebra cost is

```text
O((np + 1) N) structured factor/solves + O(np^3) dense Schur
```

rather than one factorization plus reused responses. Repairing and blocking
the stabilized reuse is required before making a speed claim against
IPOPT+MA57 or naive parameter lifting.

## 12. Historical LOPT/UCHORT replacement checkpoint (integration removed)

The initial slice has since been extended into a complete research backend for
the LOPT transcription used by UCHORT. It now supports:

- trapezoidal, Hermite--Simpson, and Lobatto-4 direct collocation;
- fixed, one-sided, and two-sided bounds on trajectory variables, phase times,
  and one-copy static parameters;
- stage-local, adjacent-stage, and dense inequalities with primal-dual barrier
  slacks;
- aggregate integrals, events, phase linkages, rates, and derived rates;
- exact Lagrangian curvature, including sparse nonlocal curvature promoted
  into the bordered system;
- heterogeneous phases, free times, complete bound/constraint dual output,
  adaptive inertia correction, and full primal-dual/barrier warm starts.

The stabilized implicit response implementation was also changed materially.
For full-rank next-state Jacobians it now preprocesses the LU/permutation data
once, propagates every border response as a batch, and forms the Schur terms
with BLASFEO matrix kernels. A multistage dense-reference regression includes
non-diagonal pivoted dynamics, local equalities and inequalities, `D_x`,
`D_eq`, `D_s`, and four simultaneous parameter right-hand sides. The verified
rank-deficient path remains a scalar reused-factorization fallback.

### Validation matrix

After these changes:

| Layer | Result |
| --- | ---: |
| Core bordered-KKT dense-reference tests | `5 / 5` |
| Core phase-arrow block solve tests | `3 / 3` |
| Core implicit OCP phase-arrow dense oracle | `1 / 1` (1/2/4/8 heterogeneous phases) |
| Core implicit phase-arrow primal-dual full-system residual | `1 / 1` (1/2/4 phases, `p=0/2`) |
| LOPT FATROP backend tests, including restoration, SOC, and indefinite-Hessian fallback | `22 / 22` |
| LOPT structured failure-policy tests | `7 / 7` |
| UCHORT `test_solve` suite, including solve, refinement, and failure handoff | `62 / 62` |
| UCHORT solve-app parser/output suite | `96 / 96` |
| End-to-end LOPT factorial benchmark | `216 / 216` |

The 216 benchmark cases are the Cartesian product

```text
phases  in {1, 2, 4, 8}
nx      in {2, 6, 12}
nu      in {1, 3}
np      in {1, 4, 8}
scheme  in {trapezoidal, Hermite--Simpson, Lobatto-4}
```

Every case reached the requested FATROP success status, remained below
`2e-7` constraint violation, and agreed with IPOPT within `2e-6` relative
objective difference.

### Pre-phase-arrow end-to-end IPOPT+MA57 comparison

The following measurements record the monolithic-border checkpoint before the
two-level phase-arrow solve was integrated. They use a `Release` build, CPU 8,
single-threaded BLAS, two intervals per phase, one untimed warm-up, and the
median of seven cold nonlinear solves. IPOPT 3.14.20 uses MA57, an exact
Lagrangian Hessian, tolerance `1e-8`, and no acceptable-iteration shortcut.
The last column is `IPOPT time / FATROP time`, so a value above one favors
FATROP.

| Scheme | Phases | `(nx,nu,np)` | Persistent dense border | FATROP ms | IPOPT+MA57 ms | IPOPT / FATROP |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Trapezoidal | 1 | `(2,1,1)` | 1 | `0.111` | `0.438` | `3.94x` |
| Trapezoidal | 2 | `(4,1,2)` | 6 | `0.136` | `0.610` | `4.50x` |
| Trapezoidal | 4 | `(6,3,4)` | 28 | `2.619` | `0.977` | `0.373x` |
| Trapezoidal | 8 | `(8,4,8)` | 120 | `6.564` | `2.059` | `0.314x` |
| Hermite--Simpson | 1 | `(2,1,1)` | 1 | `0.171` | `0.489` | `2.86x` |
| Hermite--Simpson | 2 | `(4,1,2)` | 6 | `0.225` | `0.898` | `3.99x` |
| Hermite--Simpson | 4 | `(6,3,4)` | 28 | `5.592` | `1.473` | `0.263x` |
| Hermite--Simpson | 8 | `(8,4,8)` | 120 | `12.754` | `3.446` | `0.270x` |
| Lobatto-4 | 1 | `(2,1,1)` | 1 | `0.259` | `0.572` | `2.21x` |
| Lobatto-4 | 2 | `(4,1,2)` | 6 | `0.364` | `1.176` | `3.23x` |
| Lobatto-4 | 4 | `(6,3,4)` | 28 | `10.186` | `1.819` | `0.179x` |
| Lobatto-4 | 8 | `(8,4,8)` | 120 | `22.502` | `4.376` | `0.194x` |

The new batched preprocessing and matrix Schur assembly are substantial: the
representative eight-phase Lobatto-4 case first fell from approximately
`519 ms` to `33 ms`. Embedding the benchmark's affine sequential state-linkage
rows directly in the implicit phase bridge then reduced the border from 176 to
120 and the time
again, from `33 ms` to `22.5 ms`. The same case is nevertheless still about
`5.1x` slower than IPOPT+MA57.

The benchmark reports the actual persistent border decomposition. For the
eight-phase `(8,4,8)` problem it contains

```text
64  free phase-parameter columns
56  global-only parameter-linkage multipliers
 0  inter-phase state-linkage multipliers (now implicit bridge rows)
---
120 dense border coordinates
```

More generally, for this linked benchmark the dimension is

```text
F*np + (F - 1)*np.
```

The entire block is currently condensed and factorized as dense. Its border
cost therefore grows with `np` and phase count even when total mesh work is
held fixed; `nx` now affects the structured trajectory recursion instead of
adding state-linkage duals to the border. A second 216-case run held the total
number of intervals at 16.
Aggregating all `(nx,nu,np)` cases gives:

| Phases | Trapezoidal | Hermite--Simpson | Lobatto-4 |
| ---: | ---: | ---: | ---: |
| 1 | `0.932x` | `0.793x` | `0.368x` |
| 2 | `0.738x` | `0.636x` | `0.305x` |
| 4 | `0.461x` | `0.475x` | `0.240x` |
| 8 | `0.276x` | `0.318x` | `0.170x` |

These are ratios of summed IPOPT time to summed FATROP time from a single
timed solve per case and are trend diagnostics, not stable per-case medians.
They isolate the phase-border overhead from growth in the number of mesh
intervals.

### Two-level phase-arrow checkpoint

The monolithic dense border above has now been replaced by a reusable
two-level solver. Each phase trajectory is factorized independently; its
phase-local border responses are condensed into a block-tridiagonal phase
system with a final arrow for genuinely global coordinates. The public core
has separate dense-oracle regressions for the block-arrow algebra and for the
complete implicit OCP condensation over 1/2/4/8 heterogeneous phases.

The reusable core now also performs the complete per-phase primal-dual
reduction used by a barrier Newton step: it eliminates slack and bound-dual
increments, sends the augmented trajectory blocks through the two-level
solver, and reconstructs `(dx,ds,dlambda,dzL,dzU)`. A full-system residual
regression covers 1/2/4 heterogeneous implicit phases, `p=0/2`, nonidentity
implicit dynamics, exact adjacent-stage curvature, inequalities, and both
stabilized and unstabilized equality-dual blocks. This validates the assembled
linear step; the ordinary nonlinear `IpAlgorithm` does not select it yet.

The implementation skips structurally zero incident response columns and
forms the compact Schur terms with BLAS matrix products. Representative
seven-repeat medians with two intervals per phase are:

| Scheme | Phases | FATROP ms | IPOPT+MA57 ms | IPOPT / FATROP |
| --- | ---: | ---: | ---: | ---: |
| Trapezoidal | 4 | `1.974` | `0.983` | `0.498x` |
| Trapezoidal | 8 | `2.433` | `2.057` | `0.845x` |
| Hermite--Simpson | 4 | `4.264` | `1.470` | `0.345x` |
| Hermite--Simpson | 8 | `5.075` | `3.450` | `0.680x` |
| Lobatto-4 | 4 | `8.102` | `1.802` | `0.222x` |
| Lobatto-4 | 8 | `9.898` | `4.394` | `0.444x` |

The post-arrow factorial sweep again solved all 216 cases. After enabling the
LOPT filter, feasibility normal-step, inequality-dual recentering, dual
watchdog, convexified-Hessian fallback, and guarded SOC, a fresh sweep still
solved all 216 cases. The maximum FATROP constraint violation was `1.77e-13`;
the largest relative objective difference from IPOPT was `2.60e-7`. FATROP was
faster in 91/216 single-repeat timings.
The mean `IPOPT/FATROP` ratio by phase count was:

| Phases | 1 | 2 | 4 | 8 |
| ---: | ---: | ---: | ---: | ---: |
| Mean ratio | `2.095x` | `1.129x` | `0.804x` | `0.671x` |

The phase-count bottleneck is no longer dense condensation: in the
eight-phase Lobatto-4 profile, compact Schur assembly takes about `0.09 ms`.
The dominant measured component is now trajectory response propagation
(`4.58 ms`, versus `2.52 ms` for factorization). Despite its batch API, the
full-rank implicit `solve_rhs_batch` currently invokes the scalar modified
Riccati solve once per active column. A true blocked matrix-valued implicit
backward/forward traversal is therefore the next performance kernel.

This bridge optimization is applied only when the boundary has no structural
cross-phase curvature. Nonlinear-curvature boundaries retain the transparent
full-rank bridge and exact dense linkage dual. Compact rank-deficient implicit
preprocessing is now verified with nonzero `FuFx`; rank-deficient and
stabilized exact-curvature responses conservatively refactor for each RHS.
The phase-arrow path eliminates phase-local blocks, while exact promoted
curvature and genuinely global couplings remain in the final arrow. The next
measured performance gap is cached, blocked implicit response propagation.

UCHORT now exposes `ipopt|fatrop` in both `SolveOptions` and the solve-app
configuration/CLI. Its representative two-phase mission converges through the
structured backend without invoking the generic fallback, agrees with IPOPT,
and also completes a two-pass `2 -> 4` interval mesh refinement with FATROP on
both passes. Zero-width variable boxes are returned at their exact bound, so a
scaled equality tolerance cannot turn the launch surface into an arc-domain
violation after unscaling. Numerical failure after FATROP has accepted a
structure is now an explicit policy rather than an accidental status collapse:
`strict`, IPOPT from the original start, or IPOPT from FATROP's last finite
primal. LOPT unit tests verify both handoff starts and a UCHORT integration test
verifies the last-primal route end to end.

The LOPT nonlinear wrapper now also retains a shifted
`(barrier objective, constraint violation)` filter and, after exhausting the
ordinary SQP line search and inertia retries, assembles a separate regularized
feasibility normal-step through the same phase-arrow KKT solver. A deliberately
bad static-parameter start with the nonlinear event `p^2=1` and only one
ordinary line-search trial fails with status 4 when restoration is disabled;
with restoration enabled it converges to `p=1` below `1e-8` constraint
violation. The same rejected nonlinear trial is also covered by a separate
structured second-order-correction regression. A concave bounded-control case
forces the convexified-Hessian fallback, and inequality multipliers are
recentered after every accepted primal step. The accepted-step counter no
longer charges inertia/globalization retries against `maxIterations`; UCHORT
exposes both that limit and trace frequency. These are genuine structured
safeguards, but they are not yet equivalent to IPOPT's composite
normal/tangential step and soft-restoration machinery.

### Strict five-phase UCHORT qualification

The coarse-grid `delta_III_classic` case was run from the `fast` cold-start
guess with five phases, four intervals per phase, one mesh pass, strict FATROP
failure handling, and a 240 accepted-step limit. The retained first attempt
ended with

```text
kkt                  = 3.02794e-3
primal stationarity  = 3.02794e-3
scaled feasibility   = 2.74370e-3
dense feasibility    = 1.24282e-5
```

The baseline-seed retry failed and was rejected. On the same first coarse-grid
NLP, IPOPT+MA57 reached requested success in 36 iterations with unscaled
constraint violation `2.35e-9`; its reported core solver time was about
`19 ms`. The comparison is not a clean end-to-end timing benchmark because
UCHORT performs additional reporting and retry work, but it is decisive as a
robustness result: the then-current research FATROP backend could not replace
IPOPT for this real multiphase mission. That adapter failed this qualification
and has now been removed.

No FATROP backend is currently registered in RocketSystem. Any future
re-integration must wait for broader real-mission, bad-initialization,
infeasibility, degeneracy, and refinement qualification.

## Main conclusion

The proposed one-copy global-parameter generalization is correct, but the
current implementation is not a universal improvement over state
augmentation.

- It is already useful for the tested full-rank implicit shooting problems.
- Parameter-dependent phase linkages preserve linear scaling when they are
  embedded as chain transitions. The two-level phase-arrow solve now also
  preserves the locality of phase-owned parameter/linkage blocks instead of
  factorizing their union as one dense border.
- The blocked traversal is algebraically equivalent to sequential solves and
  improves moderate-size parameter blocks. Cached full-rank implicit
  preprocessing removes the former refactorization cost, but wide-border cache
  effects and the dense Schur factorization still prevent a universal speedup.
- Different state/control dimensions and phase-local static parameters are
  algebraically supported by the dense-border formulation, including
  rectangular full-row-rank phase reset maps.
- Treating every phase-local vector as globally dense remains a useful
  correctness oracle; the standalone phase-arrow KKT and primal-dual kernels
  are the scalable algorithmic reference and agree with full-system oracles,
  but the native nonlinear IPM does not select them automatically yet.
- Rank-deficient implicit dynamics with global parameters and controlled
  inertia are now validated. Compact preprocessing also passes exact
  adjacent-stage Hessian regressions for square residual blocks; universal
  lifting remains the oracle for rectangular residual layouts.
- The compact rank-aware implementation is exact in the verified zero- and
  nonzero-`FuFx` cases. Rank-deficient and stabilized exact-curvature paths
  still repeat the structured factorization for every parameter RHS.
- The same one-copy border works inside the generated nonlinear
  equality-constrained SQP and the native FATROP primal-dual IPM. Native
  dimensions, the implicit problem API, parameter-bound inequalities, Newton
  steps, restoration reduction, and complete primal-dual warm starts now carry
  global variables.
- The removed historical LOPT route demonstrated scaled, bounded,
  heterogeneous multiphase trapezoidal, Hermite--Simpson, and Lobatto-4
  transcriptions, but none of that adapter is present in RocketSystem now.
- Explicit-normalized direct collocation and small-state DTOC3 remain the
  hardest crossover cases against one larger recursion with copied parameters.
- At the full-solver level, FATROP beats IPOPT+MA57 on all tested quadratic
  cases and on long nonlinear horizons, but loses the summed short-mesh
  Radau-3 slice. Local condensation of internal collocation states is the
  clearest performance target.
- The next formulation work is an exact cached rank-aware implicit
  transformation with rectangular residual counts; the next kernel work is
  collocation-local condensation plus a true blocked non-identity implicit
  multi-right-hand-side recursion.

Only after the structured globalization passes broad infeasible-start and
degeneracy qualification, gains a composite normal/tangential step with soft
restoration, completes broad standalone mesh-refinement qualification, and
receives collocation/implicit-response optimization plus same-problem
comparisons against other structure-aware solvers would a broad replacement
or novelty claim be justified. Re-integration is explicitly outside the
current work.

## Known gaps

- the bordered KKT solve is reusable through the explicit and implicit
  global-parameter aliases and is connected to FATROP's native implicit
  problem API, IPM, parameter-bound handling, restoration reduction, and
  complete primal-dual warm-start input/output; the remaining gap is
  production-grade globalization and qualification rather than basic
  parameter plumbing;
- compatible affine sequential state-linkage multipliers have moved into
  implicit bridge transitions, and phase-local border blocks are now condensed
  by the phase-arrow solver. Nonlinear-curvature boundaries remain exact via
  promoted coordinates and the corrected compact rank-deficient recursion;
- stabilized and unstabilized full-rank implicit preprocessing is cached and
  exposed through a batch API. With zero adjacent-stage curvature, the
  unstabilized path now normalizes any nonsingular non-identity next-state
  block once and adaptively uses scalar explicit reuse for fewer than four
  columns or a matrix-valued explicit traversal for wider borders. The
  full-rank unstabilized exact-curvature path reuses the modified factors.
  Stabilized exact-curvature and rank-deficient exact-curvature responses
  refactor per RHS, and wide instances of those cases are not tiled;
- the dense border uses a small scalar pivoted LU rather than an optimized
  blocked factorization, although this is secondary to exploiting phase
  sparsity;
- the benchmark's nonlinear implicit Hessian is Gauss--Newton and its
  derivatives are analytic; the public parametric API can receive exact blocks
  but does not yet have production AD/code-generation ergonomics;
- internal Radau states/equations are not locally condensed, so short horizons
  with large degree-3 stage blocks can be slower than IPOPT+MA57;
- the two-level phase-arrow solver can perform a complete assembled barrier
  Newton step, but is not yet selected by the native nonlinear IPM; the
  phase-aware nonlinear builder, ownership map, and ordinary/restoration
  orchestration are still missing;
- no heterogeneous naive-augmentation baseline;
- no end-to-end nonlinear copied-parameter FATROP baseline;
- no cached multi-RHS factor reuse for rank-deficient/stabilized
  exact-curvature cases, and no compact rectangular residual interface;
- native bounds, parameter bounds, restoration reduction, and warm starts have
  focused regressions, but there is no severe-degeneracy,
  contradictory-constraint, or large-active-set stress suite comparable with
  IPOPT's safeguards; a composite normal/tangential step and soft restoration
  are still absent. The watchdog/SOC regressions described in the historical
  section belonged to the removed adapter;
- the corrected compact nonzero-`FuFx` implementation has focused deterministic
  regressions and passes the 324-case phase/dimension/rank/inertia matrix with
  dense validation; sanitizer qualification and wider randomized coverage are
  still needed before a broad exact-Hessian readiness claim;
- no memory or cache-counter measurements;
- non-defect adjacent equalities inside a phase are still declined; sequential
  phase-boundary equalities are supported by the implicit bridge;
- the former UCHORT FATROP selector and handoff policies were removed. The
  strict five-phase Delta-III cold-start result remains historical evidence of
  the robustness gap; broad multi-mission, infeasible-start, and mesh-refinement
  qualification is still missing before any re-integration;
- installed headers and `libfatrop` must be deployed atomically; an older
  shared library cannot satisfy the new public batched-response ABI;
- no comparison against other structure-exploiting solvers such as acados,
  MOTO, or HPIPM on the same generated derivatives;
- one machine and one compiler; the current full-solver checkpoints and the
  historical LOPT measurements use IPOPT+MA57, while older standalone research
  tables used MUMPS as stated in their protocol.
