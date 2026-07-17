# Parametric Riccati recursion

This note defines the first mathematical kernel for the global-parameter
extension. It deliberately separates the algebraic claim from the optimized
FATROP implementation.

## 1. Problem class

For stages `k = 0,...,N-1`, let

- `x_k in R^(nx)` be the interface state;
- `u_k in R^(nu)` collect controls and any stage-local variables already
  exposed to the recursion;
- `theta in R^p` be one global static vector;
- `lambda_k` be the multiplier of a stage equality.

Define the persistent coordinate

```text
q_k = [x_k; theta].
```

The stage problem is

```text
min  1/2 [u_k; q_k]^T [R_k  N_k  ] [u_k; q_k]
                         [N_k^T Q_k]
     + r_k^T u_k + s_k^T q_k + c_k

s.t. D_k u_k + C_k q_k + d_k = 0

     x_(k+1) = A_k x_k + B_k u_k + E_k theta + a_k.
```

The terminal value is

```text
V_N(q_N) = 1/2 q_N^T W_N q_N + w_N^T q_N + c_N.
```

An inter-phase reset is the same transition with phase-specific
`A_k, B_k, E_k, a_k`. Thus adjacent multiphase linkages are covered as long as
the next state can be written explicitly. Rectangular or rank-deficient
implicit linkages are intentionally deferred to the generalized recursion.

## 2. Inductive invariant

Assume the tail problem has the exact value function

```text
V_(k+1)(q_(k+1))
  = 1/2 q_(k+1)^T W_(k+1) q_(k+1)
    + w_(k+1)^T q_(k+1) + c_(k+1).
```

Introduce

```text
q_(k+1) = Abar_k q_k + Bbar_k u_k + abar_k,

Abar_k = [A_k E_k],   Bbar_k = [B_k],   abar_k = [a_k].
         [ 0   I  ]            [ 0 ]             [ 0 ]
```

After substituting the transition, define

```text
G_k = R_k + Bbar_k^T W_(k+1) Bbar_k,

L_k = N_k + Bbar_k^T W_(k+1) Abar_k,

Qbar_k = Q_k + Abar_k^T W_(k+1) Abar_k,

g_u,k = r_k
        + Bbar_k^T (W_(k+1) abar_k + w_(k+1)),

g_q,k = s_k
        + Abar_k^T (W_(k+1) abar_k + w_(k+1)).
```

The stage-local equality-constrained KKT matrix and its coupling to the
persistent coordinate are

```text
M_k = [G_k D_k^T],       K_k = [L_k],
      [D_k   0  ]              [C_k]

h_k = [g_u,k].
      [ d_k ]
```

Provided `M_k` is nonsingular,

```text
[u_k(q_k)     ] = -M_k^(-1) (K_k q_k + h_k).
[lambda_k(q_k)]
```

Substitution gives the exact backward recursion

```text
W_k = Qbar_k - K_k^T M_k^(-1) K_k,

w_k = g_q,k - K_k^T M_k^(-1) h_k,

c_k = cbar_k - 1/2 h_k^T M_k^(-1) h_k,
```

where

```text
cbar_k = c_stage,k + c_(k+1)
         + 1/2 abar_k^T W_(k+1) abar_k
         + w_(k+1)^T abar_k.
```

Partitioning the propagated Hessian exposes the parameter recursion:

```text
W_k = [P_k S_k],
      [S_k^T T_k].
```

`P_k` is the ordinary state Riccati block, `S_k` carries state-parameter
sensitivities, and `T_k` is the accumulated dense parameter Schur block.
All three are computed by the same local solve. No parameter copies
`theta_k` and no `p` separate horizon traversals are required.

## 3. Initial parameter solve and rollout

For fixed `x_0`, the global parameter is obtained from

```text
T_0 theta =
  -(S_0^T x_0 + t_0),
```

where `t_0` is the parameter part of `w_0`. One forward traversal then applies
the stored affine policies and transition maps.

The multiplier of

```text
x_(k+1) - A_k x_k - B_k u_k - E_k theta - a_k = 0
```

is recovered as

```text
pi_k = -d V_(k+1) / d x_(k+1).
```

## 4. Exactness

The recursion is block Gaussian elimination of the full equality-constrained
QP KKT system.

Induction proof:

1. `V_N` is the exact terminal objective.
2. Assume `V_(k+1)` equals the optimal tail value.
3. Substitution of the affine transition creates the quadratic coefficients
   `G_k, L_k, Qbar_k, g_u,k, g_q,k`.
4. Solving the local KKT system enforces stationarity and all local
   equalities.
5. Its Schur complement is exactly `W_k, w_k, c_k`.
6. Therefore `V_k` equals the optimal value from stage `k` onward.

By induction, solving the final `p x p` system and rolling forward produces
the same primal and dual solution as the full KKT solve.

## 5. Computational interpretation

The existing bordered prototype factors the OCP block once and then performs
one full `solve_rhs` traversal for every parameter column. The recursion above
instead treats the `p` parameter columns as a dense block inside each stage:

```text
one backward traversal
  -> local factorization
  -> blocked solves for [state columns | parameter columns | affine column]
  -> updates of P_k, S_k, T_k
one p x p terminal solve
one forward traversal.
```

This does not remove arithmetic dependence on `p`; it removes repeated horizon
control flow and exposes matrix-matrix kernels and cache reuse. Its practical
benefit must therefore be demonstrated against both:

- sequential bordered `solve_rhs`;
- naive augmentation `x_k <- [x_k; theta_k]`.

## 6. Current verified scope

`parametric_riccati_reference.cpp` implements the equations above and
independently assembles the full dense QP KKT system. It compares:

- primal variables;
- global parameters;
- path and dynamics multipliers;
- KKT residuals;
- objective values.

The generator includes multiple phases, phase-dependent dynamics, global
parameter coupling, and stage equality constraints.

Not covered by this first kernel:

- rectangular or singular implicit next-state/linkage Jacobians;
- different state dimensions on opposite sides of a linkage;
- phase-local parameter vectors;
- inequality/slack blocks and inertia correction;
- production interior-point integration.

Those require the generalized, rank-aware recursion rather than the
nonsingular local Schur complement used here.

The separate
[HETEROGENEOUS_PHASE_PARAMETERS.md](HETEROGENEOUS_PHASE_PARAMETERS.md)
and `heterogeneous_phase_kkt_benchmark.cpp` now cover different phase
dimensions and phase-local parameters through an exact dense-border reference.
They do not yet implement the phase-aware or rank-deficient recursion.

The separate
[NONLINEAR_GLOBAL_PARAMETER_SQP.md](NONLINEAR_GLOBAL_PARAMETER_SQP.md)
uses the same one-copy global border in a complete equality-constrained
nonlinear SQP loop.
