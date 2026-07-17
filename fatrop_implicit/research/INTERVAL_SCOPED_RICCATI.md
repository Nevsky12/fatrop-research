# Interval-Scoped Riccati Factorization

Status: research hypothesis with verified reduced-KKT, complete implicit-OCP
KKT, primal-dual barrier, and nonlinear exact-Hessian direct-collocation IPM
prototypes. This note deliberately separates the generic
sparse-linear-algebra fact from the possible optimal-control contribution. It
is not yet a claim of publication-level novelty: production `IpAlgorithm`
integration, difficult-start globalization, and independent prior-art review
remain open.

## 1. Motivation

A multiphase optimal-control model often contains several kinds of one-copy
variables:

- a phase parameter used only in phase `f`;
- a linkage multiplier or switching variable used at phases `f` and `f+1`;
- a calibration parameter used over a contiguous subset of phases;
- a mission-global parameter used in every phase.

Treating all of them as global makes the final Schur complement dense in their
total dimension. Duplicating them as states increases every shooting or
collocation stage and introduces artificial continuity equations. A
phase-arrow factorization is better, but grouping all quantities into one
block per phase discards the distinction between phase-local, interface, and
segment scopes.

The proposed abstraction keeps every variable exactly once and records the
closed phase interval on which it is allowed to occur.

## 2. Condensed Newton system

Let `f = 0,...,F-1` index phases. Let `y_f` contain all trajectory variables
and local primal-dual quantities eliminated by the phase solver. Let
`v_j in R^{d_j}` be a one-copy scoped block with support

```text
I_j = [l_j, r_j],  0 <= l_j <= r_j < F.
```

After the usual slack/complementarity elimination, a regularized primal-dual
Newton system can be permuted into

```text
[ K_0                                    B_0 ] [dy_0]   [b_0]
[     K_1                                B_1 ] [dy_1]   [b_1]
[          ...                           ... ] [ ...] = [ ...]
[               K_{F-1}             B_{F-1}] [dy_F]   [b_F]
[ B_0^T B_1^T ... B_{F-1}^T             H  ] [ dv ]   [b_v].
```

Here a column block `B_{fj}` is structurally zero whenever `f` is not in
`I_j`. A linkage constraint can be represented without coupling two phase
trajectory blocks directly: its multiplier is an interface-scoped block that
couples independently to both endpoint phases.

Assume for the moment that the regularized phase pivots `K_f` are nonsingular.
Independent phase elimination produces

```text
S_ij = H_ij - sum_{f in I_i intersection I_j} B_fi^T K_f^{-1} B_fj,
q_i  = b_vi - sum_{f in I_i} B_fi^T K_f^{-1} b_f.
```

Therefore `S_ij` is structurally zero when `I_i` and `I_j` are disjoint. Each
phase contributes one dense clique only over its active set

```text
A_f = {j | f in I_j}.
```

This is the assembly boundary implemented by
`IntervalScopedKktSolver::add_phase_contribution()`.

### 2.1 Exact inequality reduction and predictor-corrector reuse

For a two-sided path constraint, introduce `h(y,v)-s=0` and distances
`s_L=s-l`, `s_U=u-s`.  With the FATROP sign convention, the slack and
complementarity residuals are

```text
r_s = -lambda_i - z_L + z_U,
r_L = s_L z_L - mu,
r_U = s_U z_U - mu.
```

Eliminating `(ds,dz_L,dz_U)` from the full primal-dual Newton system gives

```text
Sigma       = D_s + z_L/s_L + z_U/s_U,
r_bar       = r_s + r_L/s_L - r_U/s_U,
D_ineq_bar  = Sigma^(-1) + D_i,
r_ineq_bar  = r_(h-s) + Sigma^(-1) r_bar.
```

Thus the phase Riccati leaf sees an ordinary augmented equality row with
diagonal `D_ineq_bar`; after solving it, the eliminated directions are
reconstructed exactly.  The direct interval-scoped variables remain in the
same reduced graph because neither slacks nor bound duals create cross-phase
incidence.

For Mehrotra's corrector, replace the complementarity residuals by

```text
r_L^corr = s_L z_L - sigma mu + ds_aff dz_L_aff,
r_U^corr = s_U z_U - sigma mu - ds_aff dz_U_aff.
```

At a fixed iterate, these corrections change only the right-hand side:
`H_L`, the Jacobian, `Sigma`, regularization, phase response matrices, and the
interval reduced matrix are unchanged.  The implementation therefore applies
the retained phase and interval factors to the corrector rather than
factorizing twice.  This is an exact algebraic reuse, not an approximation.

## 3. Fill-free block elimination theorem

**Proposition.** Suppose the structural reduced graph contains one vertex per
scoped block and an edge `(i,j)` exactly when `I_i` and `I_j` overlap. Order
the blocks by nondecreasing right endpoint `r_j`. Prescribed block Gaussian
elimination in this order creates no structural fill.

**Proof sketch.** Consider the interval `I_i = [l_i,r_i]` eliminated next.
Every remaining neighbour `I_j` has `r_j >= r_i`. Since `I_i` and `I_j`
overlap, `l_j <= r_i`; hence `r_i` belongs to `I_j`. All remaining neighbours
of `i` therefore contain the common point `r_i` and are pairwise adjacent.
They already form a clique, so eliminating `i` adds no edge. Repeating the
argument proves the result.

This proposition is an application of the standard perfect-elimination
property of interval graphs; that graph-theoretic fact is not claimed as new.
The research question is whether the complete optimal-control construction
below is absent from prior work:

1. one-copy variables with arbitrary contiguous phase scopes;
2. generalized implicit/rank-aware Riccati phase pivots;
3. exact indefinite, regularized primal-dual Newton systems with inequalities;
4. direct collocation and multiphase linkage assembly through the same API.

## 4. Complexity

Define

```text
P     = sum_j d_j,
omega = max_f sum_{j: f in I_j} d_j.
```

The scalar dimension of every elimination front is at most `omega`. If `w_j`
is the dimension of the front in which block `j` is eliminated, the reduced
factorization takes

```text
O(sum_j [d_j^3 + d_j^2 (w_j-d_j) + d_j (w_j-d_j)^2])
    subset O(P omega^2)
```

work and `O(P omega)` block storage. The phase condensation additionally
requires solves only for the locally active response columns:

```text
sum_f cost_solve(K_f, 1 + a_f),
a_f = sum_{j in A_f} d_j,
```

rather than a response to every static variable in the mission.

Consequences:

- if every parameter is genuinely global, `omega = P` and the method reduces
  to the known dense global Schur complement;
- if scope width and local dimensions stay bounded as phases are added, both
  reduced work and storage grow linearly in the number of phases;
- the method does not promise a gain when the active width itself grows with
  the full problem.

## 5. Relationship to prior work

The closest results found in the July 2026 audit are:

- Sousa-Pinto and Orban, *Dual-Regularized Riccati Recursions for
  Interior-Point Optimal Control* ([arXiv:2509.16370](https://arxiv.org/abs/2509.16370)),
  already support a one-copy global `theta`, stage equalities and inequalities,
  dual regularization, and a full primal-dual IPM. Their cross-stage-variable
  reduction uses `p+1` Riccati solves followed by one dense `p x p` Schur
  complement. Thus global parameters plus IPM are prior art.
- Katayama and Ohtsuka, *Riccati Recursion for Optimal Control Problems of
  Nonlinear Switched Systems*
  ([arXiv:2102.02065](https://arxiv.org/abs/2102.02065)), insert switching-time
  variables into specialised boundary Riccati steps. This is a close partial
  precedent for interface-scoped variables.
- Schwan, Kuhn, and Jones, *Exploiting Multistage Optimization Structure in
  Proximal Solvers* ([arXiv:2503.12664](https://arxiv.org/abs/2503.12664)),
  factor a block-tridiagonal-arrow system for convex/proximal multistage QPs
  and support one global decision block.
- Rehfeldt et al., *A Massively Parallel Interior-Point Solver for LPs with
  Generalized Arrowhead Structure* ([DOI:10.1016/j.ejor.2021.06.063](https://doi.org/10.1016/j.ejor.2021.06.063)),
  show that constraints local to two adjacent blocks produce a sparse,
  doubly-bordered block-tridiagonal Schur complement with linearly growing
  nonzeros.  They also identify the analogous pattern for linking variables.
- Kempke, Rehfeldt, and Koch, *A Massively Parallel Interior-Point Method for
  Arrowhead Linear Programs with Local Linking Structure*
  ([DOI:10.1137/24M1716288](https://doi.org/10.1137/24M1716288)), give an exact
  hierarchical Schur factorization for local links in PIPS-IPM++.  They state
  that links over multiple adjacent blocks and local linking variables extend
  analogously, although the latter are not implemented in their experiments.
  This is the closest generic structural precedent found: border locality,
  hierarchical Schur elimination, and linear scaling are not new claims.
- Vanroye, De Schutter, and Decré, *A Generalization of the Riccati Recursion
  for Equality-Constrained Linear Quadratic Optimal Control*
  ([arXiv:2302.14836](https://arxiv.org/abs/2302.14836)), provide the
  rank-aware equality-constrained trajectory recursion used by FATROP.
- Quirynen, Houska, and Diehl, *Efficient Schur Complement Based Algorithms
  for Dynamic Optimization* / lifted collocation
  ([DOI:10.1007/s12532-017-0119-0](https://doi.org/10.1007/s12532-017-0119-0)),
  recover multiple-shooting structure by eliminating local collocation
  variables under a nonsingular local collocation Jacobian.
- Frey et al., *Multi-Phase Optimal Control Problems for Efficient Nonlinear
  Model Predictive Control With acados*
  ([arXiv:2408.07382](https://arxiv.org/abs/2408.07382)), support heterogeneous
  phase dimensions, but do not provide the one-copy interval-scope
  factorization described here.

No source in this audit was found that *implements and evaluates* arbitrary
overlapping one-copy interval variables together with implicit/rank-aware OCP
Riccati leaves and nonlinear direct collocation.  The PIPS papers make the
generic extension to several consecutive blocks plausible, so this is only a
narrow application/algorithm gap, not proof of a new algebraic principle.  A
formal review must additionally search the sparse multifrontal, factor-graph,
dynamic-programming, and optimal-estimation literature.

## 6. Implemented prototype and invariants

The implementation is split into three layers:

- `include/fatrop/ocp/interval_scoped_kkt_solver.hpp`;
- `include/fatrop/ocp/ocp_phase_condensing.hpp` and
  `include/fatrop/ocp/interval_scoped_ocp_kkt_solver.hpp`;
- `include/fatrop/ocp/interval_scoped_pd_solver.hpp`;
- `src/ocp/interval_scoped_kkt_solver.cpp`;
- the three corresponding tests under `unittest/ocp/`.

It currently provides:

- symbolic interval-overlap construction without an `O(B^2)` dense pattern;
- increasing-right-endpoint perfect elimination ordering;
- dense pivoting only inside each prescribed block;
- additive phase-clique and direct-block assembly;
- mixed-sign symmetric systems, not only Cholesky/SPD systems;
- two residual-correction iterations;
- retained trajectory and interval factors for a new right-hand side;
- explicit failing-block diagnostics;
- zero-dimensional block support;
- symbolic statistics for active width, front width, edges, and fill.

The OCP layer independently factors each explicit or implicit phase, compacts
identically zero response columns, assembles its condensed active clique, and
back-substitutes the trajectory step. The primal-dual layer performs exact
slack/complementarity elimination and reconstructs
`(dx,ds,dlambda,dz_L,dz_U)`. Direct scoped blocks supplied by the caller carry
their own barrier curvature.  Mehrotra's affine and corrector systems reuse
one trajectory/interval factorization: only the condensed right-hand side is
recomputed for the corrector.

The initial tests verify:

- zero symbolic fill and `maximum_front_dimension <= omega`;
- agreement with a monolithic dense `FullPivLU` reference for 1, 2, 4, and 8
  phases, with phase, interface, segment, and global blocks;
- agreement of the complete heterogeneous implicit-OCP KKT step with a dense
  reference for 1, 2, 4, and 8 phases and varying `(nx,nu,p)`, including a
  rank-deficient next-state Jacobian case;
- complete primal-dual barrier residuals below tolerance for 1, 2, and 4
  phases;
- equality of a retained-factor primal-dual RHS solve and an independent full
  refactorization for 1, 2, and 4 phases;
- equivalence of additive phase-clique assembly to monolithic assembly;
- correct rejection of couplings between disjoint scopes;
- correct reporting of a singular prescribed pivot;
- no regression in the existing phase-arrow kernel after sharing its
  dense-block LU implementation.

The complete project regression contains 165 tests and passes in Release. The
new reduced, implicit-OCP, primal-dual, and shared phase-arrow targets also
pass dedicated AddressSanitizer and UndefinedBehaviorSanitizer runs.

## 7. Preliminary falsification experiments

All timings below are Release medians on one pinned logical core of an AMD
Ryzen 9 9950X3D. The problem is a heterogeneous multiphase implicit OCP with
four nodes per phase. Every phase uses exactly the same Riccati condensation
kernel and the same nonzero response columns in both methods. The controlled
baseline then packs all scoped variables into one dense global Schur block;
it is deliberately more favourable than forming or scanning mission-wide zero
columns.

With phase, interface, width-four segment, and mission-global blocks of
dimensions `(2,1,2,2)`, the active width remains `omega=8`:

| Phases | Total scoped dimension `P` | Interval total, us | Packed-global total, us | Packed / interval | Complete KKT residual |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6 | `16.06` | `15.95` | `0.993` | `5.0e-16` |
| 8 | 29 | `179.28` | `178.69` | `0.997` | `9.4e-16` |
| 32 | 113 | `745.79` | `804.51` | `1.079` | `1.3e-15` |
| 64 | 225 | `1555.70` | `2023.76` | `1.301` | `8.8e-15` |
| 128 | 449 | `3942.23` | `7539.10` | `1.912` | `3.7e-15` |
| 256 | 897 | `8826.58` | `37718.34` | `4.273` | `6.0e-14` |

The small cases correctly falsify any universal speed claim: dense packing is
slightly faster through eight phases. At 256 phases, the reduced solves are
`89.5 us` and `28031 us`; the common trajectory work then limits the full-KKT
speedup to `4.27x`.

At 128 phases, increasing `(nx,nu)` and all static block sizes gives:

| `(nx,nu)` | `(phase,interface,segment,global)` dimensions | `P` | `omega` | Interval / packed, us | Packed / interval |
| --- | --- | ---: | ---: | ---: | ---: |
| `(2,1)` | `(1,1,1,1)` | 288 | 5 | `1502.52 / 2408.60` | `1.603` |
| `(4..6,2..3)` | `(2,1,2,2)` | 449 | 8 | `3944.40 / 7578.23` | `1.921` |
| `(8..10,4..5)` | `(4,2,4,4)` | 898 | 16 | `9972.67 / 38653.01` | `3.876` |
| `(12..14,6..7)` | `(8,4,8,8)` | 1796 | 32 | `24028.15 / 270122.94` | `11.242` |

All paired solutions agree within `5.4e-15`; the largest complete KKT
residual in this table is `2.1e-13`. The executable
`interval_scoped_ocp_benchmark` and script
`run_interval_scoped_ocp_sweep.sh` reproduce the experiment.

The reduced-only benchmark also exposes an implementation boundary. After
moving Schur-update lookup into symbolic analysis, retaining solve workspaces,
and applying block LU to matrix right-hand sides in place, the current kernel
is `1.7--2.1x` faster than Eigen SparseLU for `omega=8` and 64--1024 phases.
At 256 phases with overlapping segments, however, it crosses behind SparseLU
near `omega=38` (`1014 us` versus `949 us`) and loses by `2.05x` at
`omega=134` (`9997 us` versus `4882 us`). More cache-efficient dense front
updates are therefore still required for wide scopes; the complexity theorem
does not by itself guarantee a low constant.

These measurements compare linear solves, not nonlinear convergence and not
IPOPT+MA57. They establish algebraic equivalence and the predicted structural
crossover only.

## 8. Nonlinear multiphase direct collocation

The standalone nonlinear experiment uses the same interval-scoped Newton
interface without turning a static parameter into a copied state.  For phase
`f`, let `y_f` contain its complete Radau transcription and let `A_f` be the
set of scoped variables incident on that phase.  The generated NLP is

```text
minimize  sum_f phi_f(y_f, v_Af)
subject to c_f(y_f, v_Af) = 0,
           l_f <= h_f(y_f, v_Af) <= u_f,
                                      f = 0,...,F-1.
```

The constraints contain nonlinear `tanh` state dynamics, sinusoidal static
parameter dependence, implicit Radau equations of degree one or three, and
heterogeneous nonlinear phase resets.  Nonlinear two-sided path inequalities
act at every collocation stage and depend on state, control, and the scoped
static variables.  A phase interface is represented by one separator

```text
q_f in R^(nx_(f+1)),    scope(q_f) = [f,f+1],
q_f - R_f(x_f(T),v_Af) = 0,
x_(f+1)(0) - q_f = 0.
```

Consequently, no constraint contains trajectory variables from two phases.
The trajectory part of the Newton matrix stays block diagonal by phase, while
`q_f` and every phase-, interface-, segment-, or mission-scoped parameter is
stored exactly once in the interval graph.  This separator construction also
allows `nx_f != nx_(f+1)` without padding either trajectory.

The evaluator assembles the exact Lagrangian Hessian

```text
H_L = Hessian(sum_f phi_f)
      + sum_i lambda_i Hessian(c_i,h_i),
```

including the second derivatives of nonlinear dynamics, resets, and path
inequalities.  The structured solver and IPOPT adapter call the same
objective, constraint, Jacobian, bounds, and Hessian implementation.

The standalone solver is a primal-dual predictor-corrector method.  It keeps
strictly interior slacks, forms the affine step, computes `mu_aff`, applies
Mehrotra's second-order complementarity correction, and uses a
fraction-to-boundary residual line search.  The affine and corrector matrices
are identical, so every iteration performs one numeric factorization and one
retained-factor RHS solve.  The complete reconstructed direction is
`(dx,ds,dlambda,dz_L,dz_U)`.

Across the sweep below, the maximum normalized exact-Hessian directional
error is `9.71e-11`, the maximum full linear-system residual is `3.48e-10`,
and the maximum difference from a monolithic dense primal-dual KKT solve is
`2.59e-10`.  These checks include nonzero inequality multipliers and the
second-order corrector RHS.

### Matched IPOPT+MA57 sweep

Release medians below use seven repeats on pinned logical CPU 8.  All rows use
nonzero phase, interface, segment, and global static parameters with block
dimensions `(2,1,2,2)`, two inequalities per stage, and the interface
separator states.  `IPOPT / scoped` greater than one favours the proposed
factorization.

| Phases | Radau degree | `omega` | Scoped IPM, ms | IPOPT+MA57, ms | IPOPT / scoped | Iterations scoped / IPOPT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 6 | `0.445` | `1.137` | `2.553` | `6 / 8` |
| 2 | 1 | 13 | `2.141` | `3.142` | `1.468` | `7 / 11` |
| 4 | 1 | 17 | `5.169` | `6.946` | `1.344` | `8 / 12` |
| 8 | 1 | 20 | `14.302` | `15.773` | `1.103` | `9 / 12` |
| 1 | 3 | 6 | `1.153` | `1.545` | `1.340` | `6 / 8` |
| 2 | 3 | 13 | `5.205` | `5.216` | `1.002` | `8 / 11` |
| 4 | 3 | 17 | `10.543` | `11.967` | `1.135` | `8 / 12` |
| 8 | 3 | 20 | `26.357` | `29.332` | `1.113` | `9 / 13` |

Both methods converged in every row.  Over all eleven profiles, the largest
structured equality/bound violation, stationarity residual, and
complementarity product were `1.55e-9`, `4.91e-10`, and `5.06e-9`.  IPOPT's
largest bound/equality violation was `9.99e-9`.  At the common `1e-8`
termination tolerance, the largest componentwise primal and multiplier
distances were `1.98e-4` and `8.63e-5` near an active-set transition, while
the maximum objective difference was `9.84e-9`.  Tightening both solvers to
`1e-10` on the eight-phase Radau-3 case reduced primal/multiplier distance to
`2.08e-7 / 2.29e-7` and objective distance to `2.90e-11`.

Factor reuse is material rather than cosmetic.  On that eight-phase Radau-3
case, disabling retained-factor correction increased median KKT time from
`11.883` to `21.162 ms` and total time from `26.340` to `35.688 ms`.  The
default performs 9 factorizations plus 9 retained RHS solves instead of 18
factorizations.

The result has a clear counter-regime.  A four-phase Radau-3 problem with
`nx=(8,12,6,10)`, `nu=(4,6,3,5)`, static block dimensions `(6,4,8,12)`, and
four inequalities per stage has `omega=58`.  Scoped IPM took `80.023 ms`
versus `66.968 ms` for IPOPT+MA57 (`IPOPT/scoped=0.837`).  Wide dense fronts
therefore remain a real limitation, consistent with the reduced-kernel
crossover near `omega=38`.

An exact-Hessian ablation is also decisive.  On the eight-phase Radau-3 row,
using only the objective/Gauss--Newton Hessian required 35 structured SQP
iterations and IPOPT failed its 1500-iteration limit.  Supplying the analytic
constraint curvature reduced both solvers to five iterations.  Comparisons
that label this callback `exact` while omitting constraint curvature are not
methodologically valid.

The executable `nonlinear_interval_scoped_sqp_benchmark`, separate equality
and inequality CTest regressions, and
`run_nonlinear_interval_scoped_sweep.sh` reproduce these checks.  ASan+UBSan
passes the retained-RHS unit test and the nonlinear inequality regression.

## 9. Assumptions and failure modes

The result is conditional, not magic sparse pivoting:

- every scoped dependence must be representable by a contiguous phase
  interval; disconnected support either needs several blocks plus equality
  links or a more general chordal formulation;
- every prescribed block pivot must be nonsingular after primal/dual
  regularization; unrestricted cross-block pivoting can destroy the interval
  ordering;
- exact symmetry is assumed at the KKT level;
- the current implicit trajectory interface still uses square dynamics
  residual blocks; arbitrary rectangular index-varying DAEs are not yet
  covered;
- the nonlinear path is a standalone primal-dual research solver, not yet a
  selectable backend of FATROP's production `IpAlgorithm`; it does not yet
  inherit the production filter, restoration phase, inertia loop, warm-start
  policy, or difficult-start safeguards;
- retained predictor/corrector factors are valid only while the Hessian,
  Jacobian, barrier diagonal, regularization, and scope graph are unchanged;
  the public RHS-only API deliberately makes this precondition explicit.

## 10. Falsifiable experimental claims

The next implementation stages should try to disprove, not merely illustrate,
the following claims:

1. **Algebraic equivalence (supported at the linear layer):** the structured
   step matches a monolithic dense or sparse KKT step to a scaled residual
   tolerance across phase counts and dimensions `(n_x,n_u,n_p)`.
2. **Width scaling (supported, with a constant-factor caveat):** with bounded
   `omega`, reduced factor time and memory grow approximately linearly with
   phase count, whereas treating all scoped variables as global grows cubically
   in their total dimension.
3. **No hidden iteration penalty (supported on the current generated IPM
   matrix):** the exact-Hessian structured predictor-corrector converges in
   6--9 iterations versus IPOPT's 8--13 on the eight bounded-width rows.
   Difficult-start and production-restoration evidence is still missing.
4. **Robust rank handling (supported on a dense-reference regression):** local
   rank-deficient implicit dynamics do not force dense cross-phase fallback
   when the generalized phase recursion succeeds.
5. **Crossover honesty:** for small `P`, large `omega`, or dense true-global
   coupling, the generic sparse solver or the global-Schur implementation may
   remain faster; those regimes must be reported.

Until production integration, broader scaling curves, real application
benchmarks, and independent literature review are complete, the defensible
wording is “new candidate interval-scoped OCP/IPM algorithm”, not “proven new
scientific method”.
