# Heterogeneous phases and phase-local parameters

This note defines the third research kernel: multiple phases with different
state and control dimensions, one global static vector, and independent
phase-local static vectors.

## 1. Variables and locality

For phase `f = 0,...,F-1`, let

```text
x_(f,k) in R^(nx_f),
u_(f,k) in R^(nu_f),
theta_g in R^(pg),
theta_f in R^(pf).
```

Collect every static variable once:

```text
theta = [theta_g; theta_0; ...; theta_(F-1)].
```

Ordinary stage functions in phase `f` depend only on

```text
eta_f = [theta_g; theta_f] = J_f theta,
```

where `J_f` is a column selector. Thus the original stage-to-parameter
Jacobian has nonzeros only in the global block and the block local to that
phase.

The linkage between adjacent phases may additionally depend on both local
parameter vectors:

```text
x_(f+1,0) =
    R_f x_(f,end)
  + E_(g,f) theta_g
  + E_(-,f) theta_f
  + E_(+,f) theta_(f+1)
  + d_f.
```

Here

```text
R_f in R^(nx_(f+1) x nx_f)
```

may be rectangular. The complete linkage Jacobian with respect to the
outgoing and incoming states is

```text
[R_f  -I_(nx_(f+1))].
```

It always has full row rank because of the incoming `-I` block, even when the
state dimension grows or shrinks between phases. This is the heterogeneous
explicit-linkage case. A genuinely rank-deficient implicit linkage, for which
no such full-rank incoming block exists, is a separate problem.

## 2. Exact dense-border reduction

After flattening the phases into one chain, let `w` contain all stage primal
variables and equality multipliers. The Newton system is

```text
[ K       B ] [delta_w    ] = -[r_w    ],
[ B^T  Htheta] [delta_theta]    [r_theta]
```

where:

- `K` is the ordinary heterogeneous OCP KKT matrix;
- columns of `B` encode objective, dynamics, collocation, and linkage
  derivatives with respect to the one-copy static variables;
- `Htheta` contains direct static-static Hessian terms.

Factor `K` once and define

```text
Y = K^(-1) B,
z = K^(-1) r_w.
```

The exact reduced system is

```text
S_theta delta_theta =
    -r_theta + B^T z,

S_theta = Htheta - B^T Y.
```

Back-substitution gives

```text
delta_w = -z - Y delta_theta.
```

The implementation uses FATROP's sign convention internally, but assembles
this same Schur complement. A dense pivoted-LU solve of the full KKT system is
the independent reference.

## 3. Why phase locality does not imply a sparse final Schur matrix

Before elimination, a column for `theta_f` touches only phase `f` and its two
adjacent linkages. Nevertheless,

```text
B^T K^(-1) B
```

is generally dense: changing a parameter in an early phase changes its
terminal state, which propagates through every later phase. Therefore treating

```text
p_total = pg + sum_f pf
```

as one dense border is exact but has dense work and storage in `p_total`,
including an `O(p_total^3)` final factorization.

The direct static Hessian used by the benchmark preserves the intended model
locality:

- global-global couplings are allowed;
- global-to-local couplings are allowed;
- local-local couplings are allowed inside one phase;
- unrelated local vectors have no direct Hessian coupling.

Any cross-phase fill in the reduced system is consequently created by
trajectory elimination rather than inserted artificially.

## 4. Phase-aware recursion suggested by the structure

The scalable target is not the full dense border. A phase-aware elimination
can retain, for each phase,

```text
(incoming boundary state, outgoing boundary state, theta_g, theta_f)
```

while eliminating its internal shooting or collocation variables. The
resulting reduced problem is a chain in the boundary states and
`theta_f` blocks, with one arrow corresponding to `theta_g`.

A second block elimination can remove the phase-local variables in chain
order. Only the global block must remain dense. For uniformly bounded phase
dimensions, the intended complexity is linear in the number of phases plus
the dense cost in `pg`, instead of a dense factorization in
`pg + sum_f pf`.

Establishing the exact recursion, its pivoting/inertia conditions, and a
performance crossover against the full dense border is a candidate research
contribution. The current benchmark is the correctness oracle for that future
algorithm.

## 5. Direct collocation representation

For Radau degree `d`, each nonterminal mesh stage in phase `f` has

```text
nu_f + d nx_f
```

local variables: the physical control and `d` collocation states. The `d nx_f`
collocation residuals are stage-local equalities. Only the mesh endpoint is
passed to the next stage. Phase linkages use the rectangular map above.

The benchmark covers shooting and Radau degrees 1--3. It therefore tests
heterogeneous state, control, local-parameter, and collocation block sizes in
the same FATROP recursion.

## 6. Verification

`heterogeneous_phase_kkt_benchmark.cpp` compares:

- sequential one-column applications of the factored OCP KKT matrix;
- one blocked multi-right-hand-side traversal;
- structured KKT residuals;
- a full dense KKT solve for problems of dimension at most 2500.

`run_heterogeneous_phase_sweep.sh` varies:

- phase count;
- every phase's `nx_f`, `nu_f`, and `pf`;
- the global dimension `pg`;
- shooting and Radau degrees 1--3.

This kernel does not yet include:

- rank-deficient implicit linkages or singular DAE mass matrices;
- cached blocked multi-RHS propagation for all stabilized/exact-curvature
  corners;
- a native nonlinear builder that retains phase-local parameter blocks and
  selects the phase-arrow primal-dual solver in ordinary and restoration
  iterations.

The reusable core now includes phase-aware local/interface elimination and a
full primal-dual wrapper for inequality slacks, bound-dual reconstruction, and
equality-dual stabilization. These are assembled-system capabilities; they are
not yet an end-to-end phase-aware nonlinear `IpAlgorithm`.

The separate
[NONLINEAR_GLOBAL_PARAMETER_SQP.md](NONLINEAR_GLOBAL_PARAMETER_SQP.md)
provides an end-to-end nonlinear SQP for one-copy global parameters, but it
does not yet combine that loop with phase-local parameter elimination.
