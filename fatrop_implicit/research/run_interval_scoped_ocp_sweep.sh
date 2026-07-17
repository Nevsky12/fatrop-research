#!/usr/bin/env bash

set -euo pipefail

research_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$research_dir/.." && pwd)"
build_dir="${BUILD_DIR:-$repo_dir/build-research}"
benchmark="$build_dir/research/interval_scoped_ocp_benchmark"
bench_cpu="${BENCH_CPU-8}"

read -r -a phase_values <<< "${PHASES_LIST:-1 2 4 8 16 32 64 128 256}"
read -r -a dimension_phase_values <<< "${DIMENSION_PHASES_LIST:-16 64 128}"
read -r -a dimension_cases <<< "${DIMENSION_CASES:-2:1:1:1:1:1 4:2:2:1:2:2 8:4:4:2:4:4 12:6:8:4:8:8}"
repeats="${REPEATS:-11}"
nodes="${NODES:-4}"

if [[ ! -x "$benchmark" ]]; then
    echo "Benchmark executable not found: $benchmark" >&2
    exit 1
fi

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1

runner=()
if [[ -n "$bench_cpu" ]] && command -v taskset >/dev/null 2>&1; then
    runner=(taskset -c "$bench_cpu")
fi

header_printed=0
run_case() {
    local sweep="$1"
    local phases="$2"
    local nx="$3"
    local nu="$4"
    local phase_p="$5"
    local interface_p="$6"
    local segment_p="$7"
    local global_p="$8"
    local nx_period="$9"
    local nu_period="${10}"
    local output header row

    output="$("${runner[@]}" "$benchmark" \
        --phases "$phases" \
        --nodes "$nodes" \
        --nx "$nx" \
        --nx-period "$nx_period" \
        --nu "$nu" \
        --nu-period "$nu_period" \
        --phase-parameters "$phase_p" \
        --interface-parameters "$interface_p" \
        --segment-parameters "$segment_p" \
        --segment-width 4 \
        --segment-stride 4 \
        --global-parameters "$global_p" \
        --repeats "$repeats")"
    header="$(printf '%s\n' "$output" | head -n 1)"
    row="$(printf '%s\n' "$output" | tail -n 1)"
    if (( header_printed == 0 )); then
        printf 'sweep,input_nx,input_nu,phase_p,interface_p,segment_p,global_p,%s\n' "$header"
        header_printed=1
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$sweep" "$nx" "$nu" "$phase_p" "$interface_p" \
        "$segment_p" "$global_p" "$row"
}

for phases in "${phase_values[@]}"; do
    run_case phase_scaling "$phases" 4 2 2 1 2 2 3 2
done

for phases in "${dimension_phase_values[@]}"; do
    for encoded in "${dimension_cases[@]}"; do
        IFS=: read -r nx nu phase_p interface_p segment_p global_p <<< "$encoded"
        run_case dimension_scaling "$phases" \
            "$nx" "$nu" "$phase_p" "$interface_p" "$segment_p" \
            "$global_p" 1 1
    done
done
