# Global-parameter and collocation research benchmarks

This directory contains experimental benchmarks for structures that are not
part of the public FATROP API. The current code answers two questions:

1. Can a small, one-copy vector of global static parameters be handled as a
   dense border around the OCP KKT system instead of copying it into every
   state?
2. Does the same construction remain valid and useful when an implicit ODE is
   transcribed with direct Radau collocation?

The short answer from the current experiments is:

- the bordered solve is algebraically correct;
- it is faster than naive state augmentation for the tested non-identity
  implicit dynamics;
- with an explicit-normalized Radau transcription, the present implementation
  is slower than a fair naive FATROP baseline because it performs one
  additional full Riccati right-hand-side sweep per global parameter;
- both structured variants are generally much faster than the one-step IPOPT
  runs in the quadratic benchmarks, but those columns have different timing
  scopes and end-to-end nonlinear-solver gains depend strongly on the problem.

See [RESULTS.md](RESULTS.md) for measurements and conclusions.

## Executables

| Executable | Purpose |
| --- | --- |
| `global_parameter_kkt_benchmark` | Synthetic implicit shooting, direct Radau collocation, and a DTOC3-derived quadratic problem; compares the bordered KKT solve, naive FATROP state augmentation, dense LU on small cases, and optionally IPOPT. |
| `dtoc_fatrop_benchmark` | End-to-end FATROP solves for exact DTOC1L, DTOC3, and DTOC6 formulations derived from CUTEst SIF files. |
| `cutest_probe` | Loads a decoded CUTEst problem and reports dimensions, derivative sparsity, and locality before and after an OCP stage ordering. |
| `cutest_ipopt_benchmark` | Generic CUTEst-to-IPOPT adapter with exact sparse first and second derivatives. |

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
3. applies the same factorization to the `p` columns of `B`;
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

The current implementation is linear-quadratic and uses a fixed step solely to
isolate the KKT structure. It does not yet exercise nonlinear implicit residual
evaluation or automatic differentiation.

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
- equality consensus between all naive parameter copies;
- agreement with the naive FATROP solve;
- agreement with a dense pivoted-LU solve when the total KKT dimension is at
  most 2500;
- agreement with IPOPT when `--ipopt` is enabled.

The bordered and naive KKT timings exclude problem generation and solver
construction. They include numerical normalization, factorization, solves, and
the dense parameter Schur complement. The reported component medians need not
sum exactly to the median total.

The IPOPT column measures `OptimizeTNLP` after `IpoptApplication`
initialization. The quadratic cases converge in one iteration, but this timing
still includes TNLP callbacks and IPOPT bookkeeping. It is therefore useful
context, not an isolated linear-solver timing directly equivalent to the
bordered and naive kernel measurements.

`dtoc_fatrop_benchmark` and `cutest_ipopt_benchmark` measure complete
optimization calls, but not process startup or SIF decoding.

## Current limitations

- The new border is a KKT-kernel prototype, not yet integrated into FATROP's
  complete nonlinear primal-dual iteration.
- Parameter columns are solved sequentially. There is no fused multi-RHS
  Riccati sweep, which is the main performance opportunity exposed by the
  collocation results.
- The direct-collocation test is a linear implicit ODE with Radau degree at
  most 3.
- The present suite has equality constraints only; inequality/slack behavior
  with a global border has not been benchmarked.
- Nonzero cross-stage Hessian blocks `FuFx` are not supported by the current
  full-rank normalization path. Earlier experiments on the implicit branch did
  not pass dense-residual validation, so the executable now rejects
  `--cross-hessian-scale` values other than zero. The reported results therefore
  use a stage-block-diagonal Hessian.
- No memory-usage comparison has been collected yet.
- Results from one CPU and one IPOPT linear solver are evidence, not a general
  performance theorem.
