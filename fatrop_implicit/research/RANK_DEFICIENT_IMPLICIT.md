# Rank-deficient implicit dynamics and inertia

This note separates three cases that are often conflated:

1. an explicit transition with a rectangular reset matrix;
2. an implicit residual with a singular next-state Jacobian;
3. a genuinely rectangular implicit residual with a different number of
   equations and incoming-state variables.

The first case is already handled by the heterogeneous-phase benchmark. This
note derives exact transformations for the other two.

## 1. Linearized implicit stage

Let

```text
y_k = [u_k; x_k],
F_k(y_k, x_(k+1), theta) = 0.
```

At one Newton step,

```text
A_k delta_y_k
+ J_k delta_x_(k+1)
+ E_k delta_theta
= -r_k,
```

with

```text
A_k in R^(m_k x ny_k),
J_k in R^(m_k x nx_(k+1)).
```

The ordinary Riccati transition requires

```text
delta_x_(k+1) =
    B_k delta_y_k
  + C_k delta_theta
  + b_k.
```

That direct normalization is possible only when `m_k = nx_(k+1)` and `J_k`
is nonsingular. A singular DAE Jacobian must not be inverted.

## 2. Universal lifting

Introduce one local copy of the incoming state,

```text
v_k in R^(nx_(k+1)),
```

and replace the implicit transition by

```text
A_k delta_y_k
+ J_k delta_v_k
+ E_k delta_theta
= -r_k,

delta_x_(k+1) - delta_v_k = 0.
```

The first equation is a stage-local equality. The second is an ordinary
explicit OCP transition. This construction is valid for arbitrary rectangular
`J_k`, including rank-deficient and zero matrices.

It is an exact reformulation:

- eliminating `v_k` with `v_k = x_(k+1)` recovers the original residual;
- the multiplier of the lifted implicit equality is the original dynamics
  multiplier;
- the copy multiplier enforces stationarity of `v_k` and disappears when the
  lifted KKT system is condensed.

The lifting adds `nx_(k+1)` primal variables and the same number of copy
constraints. It preserves the chain structure but increases every nonterminal
stage.

## 3. Exact cross-stage Hessians

An exact Newton Hessian can contain

```text
delta_y_k^T C_k delta_x_(k+1).
```

After lifting, this becomes

```text
delta_y_k^T C_k delta_v_k,
```

which is local to stage `k`. Thus the universal lifting supports an arbitrary
cross-stage block without any special Riccati kernel.

This property is important for nonlinear implicit dynamics. A Gauss--Newton
or block-diagonal approximation can set the block to zero, but an exact
Lagrangian Hessian generally cannot.

## 4. Rank-aware partial lifting

The compact alternative uses a rank-revealing factorization of `J_k`. Let

```text
rank(J_k) = r_k.
```

After row and column transformations, split the incoming coordinate into

```text
delta_x_(k+1) = T_k [delta_x_d; delta_x_f],

dim(delta_x_d) = r_k,
dim(delta_x_f) = nx_(k+1) - r_k.
```

The independent equations determine `delta_x_d`. The free coordinates
`delta_x_f` are moved to the controls of stage `k+1`. Equations not used to
determine `delta_x_d` become stage equalities. For square `J_k`, the current
implicit implementation therefore moves

```text
nx_(k+1) - r_k
```

state components and the same number of dynamics equations.

This transformation preserves the total number of primal variables and
constraints. It is a partial lifting: only the null-space coordinates are
exposed as controls.

Necessary rank conditions include:

- the complete constraint Jacobian must have full row rank;
- the equations transferred between stages must remain independent after
  propagation;
- each local reduced Hessian must be positive definite on the null space of
  the active equalities.

For a rectangular `J_k`, the same construction produces
`nx_(k+1)-r_k` free coordinates and `m_k-r_k` remaining equations. Supporting
those unequal counts requires an interface that stores `m_k` independently of
`nx_(k+1)`.

## 5. Inertia condition

For the equality-constrained base system

```text
K(delta) = [H + delta I  A^T],
           [A              0 ],
```

assume `A` has full row rank and let the columns of `Z` be an orthonormal basis
for `null(A)`. The KKT matrix has the desired inertia

```text
(number of primal variables,
 number of constraints,
 0)
```

if and only if

```text
Z^T (H + delta I) Z
```

is positive definite.

For a uniform primal shift and a requested margin `epsilon`,

```text
delta =
    max(0,
        epsilon
        - lambda_min(Z^T H Z)).
```

The benchmark computes this value independently with a dense SVD/eigensolve,
passes the resulting diagonal `D_x` to FATROP, and verifies the inertia of
both the base and global-parameter KKT matrices.

The global static variables add a second condition: after eliminating the OCP
block, their Schur complement must have the desired positive inertia as well.

## 6. What is implemented and verified

`rank_deficient_implicit_kkt_benchmark.cpp` constructs heterogeneous
multiphase problems with:

- different `nx_f` and `nu_f`;
- prescribed rank loss in every ordinary transition and phase linkage;
- one-copy global static parameters;
- optional negative curvature followed by inertia correction;
- an optional exact cross-stage Hessian.

It compares:

1. the compact rank-aware `ImplicitOcpType` preprocessing;
2. the universal lifted `OcpType` formulation;
3. a dense full KKT solve.

The compact and lifted paths are now both exact in the verified `FuFx = 0`
and nonzero-`FuFx` cases for square residual blocks. Focused regressions cover
full-rank and rank-deficient dynamics, with and without equality-dual
stabilization, and solve a different RHS after the base factorization.
A dense-reference validation sweep also passes all 324 combinations of
1/2/4/8 phases, heterogeneous state/control dimensions, 0/2/8 global
parameters, rank loss 0/1/2, and Gauss--Newton, exact-curvature, or
inertia-corrected Hessians.

The compact-recursion defect originally exposed by this benchmark was fixed
by carrying each transformed adjacent-stage Hessian stripe through every
backward step and including its affine-dynamics terms in the retained-RHS
backward and multiplier-recovery equations. Rank-deficient and stabilized
exact-curvature retained-RHS paths currently refactor through the same
structured recursion for correctness; cached/tiled reuse remains future work.

Other core bugs fixed while enabling repeated bordered solves include:

- the modified state/control/equality dimensions are restored from the
  original problem before every preprocessing pass;
- terminal preprocessing no longer reads stage `K` through `k+1`.

Active unit tests perform repeated full-rank and rank-deficient
right-hand-side solves and check normalized original-KKT residuals.

## 7. Research implication

Universal lifting is an exact baseline, not by itself a novelty claim.
The more interesting remaining target is a compact rank-aware recursion that
simultaneously supports:

- arbitrary rectangular implicit residuals;
- cached/tiled exact cross-stage Hessian responses;
- global and phase-local static parameters;
- phase-aware elimination;
- inertia-preserving pivoting and regularization.

Whether that combined algorithm is scientifically novel requires a dedicated
literature comparison. The current code establishes a reproducible algebraic
and numerical baseline for that comparison.
