#!/usr/bin/env bash

set -euo pipefail

research_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$research_dir/.." && pwd)"
build_dir="${BUILD_DIR:-$repo_dir/build-research}"
cutest_cases="${CUTEST_CASES:-$repo_dir/../research-deps/problems}"
bench_cpu="${BENCH_CPU-8}"

if [[ "${1-}" == "--help" ]]; then
    echo "Usage: BENCH_CPU=8 BUILD_DIR=... CUTEST_CASES=... $0"
    exit 0
fi
if (( $# != 0 )); then
    echo "Unexpected arguments; use --help" >&2
    exit 2
fi

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1

runner=()
if [[ -n "$bench_cpu" ]] && command -v taskset >/dev/null 2>&1; then
    runner=(taskset -c "$bench_cpu")
fi

run()
{
    "${runner[@]}" "$@"
}

kkt="$build_dir/research/global_parameter_kkt_benchmark"
fatrop="$build_dir/research/dtoc_fatrop_benchmark"
probe="$build_dir/research/cutest_probe"
cutest_ipopt="$build_dir/research/cutest_ipopt_benchmark"

run "$kkt" \
    --problem synthetic --stages 20 --nx 6 --nu 3 \
    --parameters 2 --repeats 11 --ipopt

run "$kkt" \
    --problem synthetic --stages 200 --nx 20 --nu 10 \
    --parameters 4 --repeats 15 --ipopt --no-dense-validation

run "$kkt" \
    --problem synthetic --stages 2000 --nx 20 --nu 10 \
    --parameters 4 --repeats 9 --ipopt --no-dense-validation

for parameters in 1 4 8; do
    run "$kkt" \
        --problem synthetic --stages 200 --nx 6 --nu 3 \
        --parameters "$parameters" --collocation-degree 3 \
        --repeats 15 --ipopt --no-dense-validation
done

run "$kkt" \
    --problem synthetic --stages 1961 --nx 6 --nu 3 \
    --parameters 4 --collocation-degree 3 \
    --repeats 9 --ipopt --no-dense-validation

run "$kkt" \
    --problem synthetic --stages 589 --nx 20 --nu 10 \
    --parameters 4 --collocation-degree 3 \
    --repeats 9 --ipopt --no-dense-validation

run "$kkt" \
    --problem synthetic --stages 236 --nx 50 --nu 25 \
    --parameters 4 --collocation-degree 3 \
    --repeats 9 --ipopt --no-dense-validation

for parameters in 0 1 4; do
    run "$kkt" \
        --problem dtoc3 --stages 30000 \
        --parameters "$parameters" --repeats 9 \
        --ipopt --no-dense-validation
done

for repeat in 1 2 3; do
    run "$fatrop" \
        --problem dtoc1l --stages 1000 --controls 5 --states 10
    run "$fatrop" \
        --problem dtoc1l --stages 6667 --controls 5 --states 10
    run "$fatrop" \
        --problem dtoc1l --stages 9091 --controls 1 --states 10
    run "$fatrop" --problem dtoc3 --stages 5000
    run "$fatrop" --problem dtoc3 --stages 30000
    run "$fatrop" --problem dtoc6 --stages 5001
    run "$fatrop" --problem dtoc6 --stages 50000
done

if [[ -x "$probe" && -x "$cutest_ipopt" && -d "$cutest_cases" ]]; then
    cases=(
        DTOC3_N5000
        DTOC3_N30000
        DTOC1L_N1000_X5_Y10
        DTOC1L_N6667_X5_Y10
        DTOC1L_N9091_X1_Y10
        DTOC6_N5001
        DTOC6_N50000
    )

    for case_name in "${cases[@]}"; do
        outsdif="$cutest_cases/$case_name/OUTSDIF.d"
        library="$cutest_cases/$case_name/lib${case_name}.so"
        if [[ ! -f "$outsdif" || ! -f "$library" ]]; then
            continue
        fi
        run "$probe" "$outsdif" "$library"
        for repeat in 1 2 3; do
            run "$cutest_ipopt" "$outsdif" "$library"
        done
    done
fi
