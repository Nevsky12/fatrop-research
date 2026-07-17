#!/usr/bin/env bash

set -euo pipefail

research_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$research_dir/.." && pwd)"
build_dir="${BUILD_DIR:-$repo_dir/build-research}"
benchmark="$build_dir/research/interval_scoped_kkt_benchmark"
bench_cpu="${BENCH_CPU-8}"

read -r -a phase_values <<< "${PHASES_LIST:-8 16 32 64 128 256 512}"
read -r -a segment_width_values <<< "${SEGMENT_WIDTH_LIST:-2 4 8}"
repeats="${REPEATS:-11}"
phase_dimension="${PHASE_DIMENSION:-2}"
interface_dimension="${INTERFACE_DIMENSION:-1}"
segment_dimension="${SEGMENT_DIMENSION:-2}"
global_dimension="${GLOBAL_DIMENSION:-2}"
dense_limit="${DENSE_LIMIT:-1200}"

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
for phases in "${phase_values[@]}"; do
    for width in "${segment_width_values[@]}"; do
        output="$("${runner[@]}" "$benchmark" \
            --phases "$phases" \
            --phase-dimension "$phase_dimension" \
            --interface-dimension "$interface_dimension" \
            --segment-dimension "$segment_dimension" \
            --segment-width "$width" \
            --segment-stride "$width" \
            --global-dimension "$global_dimension" \
            --repeats "$repeats" \
            --dense-limit "$dense_limit")"
        header="$(printf '%s\n' "$output" | head -n 1)"
        row="$(printf '%s\n' "$output" | tail -n 1)"
        if (( header_printed == 0 )); then
            printf 'segment_width,%s\n' "$header"
            header_printed=1
        fi
        printf '%s,%s\n' "$width" "$row"
    done
done
