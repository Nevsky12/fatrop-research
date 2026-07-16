# Preliminary results: global parameters and direct collocation

Date: 2026-07-16

These are research measurements, not production performance claims. They
evaluate the exactness and cost of a dense global-parameter border around the
FATROP OCP KKT recursion.

## Environment and protocol

- FATROP branch: `research/global-parameter-collocation`
- FATROP base commit: `4587bef`
- CPU: AMD Ryzen 9 9950X3D
- Compiler: GCC 13.3.0
- CMake: 3.28.3
- IPOPT: 3.14.20 with MUMPS
- CUTEst: 2.7.1
- Build: `Release`, bundled BLASFEO
- Affinity: logical CPU 8
- Thread limits: `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
  `MKL_NUM_THREADS=1`, `BLIS_NUM_THREADS=1`

Bordered and naive KKT-kernel values are medians after an untimed warm-up. The
IPOPT columns in those tables time `OptimizeTNLP` after application
initialization; all quadratic cases converge in one iteration, but the timing
scope is broader than an isolated KKT factorization. End-to-end FATROP and
CUTEst/IPOPT values are medians of three independent process runs. Times are in
milliseconds.

## Correctness

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

## Main conclusion

The proposed one-copy global-parameter generalization is correct, but the
current implementation is not a universal improvement over state
augmentation.

- It is already useful for the tested full-rank implicit shooting problems.
- It loses for explicit-normalized direct collocation and small-state DTOC3
  because `p` separate Riccati right-hand-side sweeps cost more than one
  recursion with `p` additional state components.
- The next algorithmic experiment should be a fused or blocked multi-RHS
  Riccati recursion that propagates all parameter sensitivities in one
  cache-friendly sweep. Direct collocation should also be tested with
  stage-local static condensation before that sweep.

Only after implementing those two changes and integrating the border into the
complete nonlinear primal-dual loop would a broad performance or novelty claim
be justified.

## Known gaps

- no complete nonlinear FATROP solver using the new global border;
- no nonlinear implicit DAE collocation case;
- no inequalities or active-set/slack stress test;
- no nonzero `FuFx` validation;
- no memory or cache-counter measurements;
- no comparison against other structure-exploiting solvers such as acados,
  MOTO, or HPIPM on the same generated derivatives;
- one machine, one compiler, and MUMPS as IPOPT's linear solver.
