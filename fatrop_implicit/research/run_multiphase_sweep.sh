#!/usr/bin/env bash

set -euo pipefail

research_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$research_dir/.." && pwd)"
build_dir="${BUILD_DIR:-$repo_dir/build-research}"
bench_cpu="${BENCH_CPU-8}"
kkt="$build_dir/research/global_parameter_kkt_benchmark"

repeats="${REPEATS:-5}"
fixed_total_stages="${FIXED_TOTAL_STAGES:-160}"
nodes_per_phase="${NODES_PER_PHASE:-30}"
collocation_degree="${COLLOCATION_DEGREE:-3}"
run_ipopt="${RUN_IPOPT:-0}"
run_native_ipm="${RUN_NATIVE_IPM:-0}"
run_full_solvers="${RUN_FULL_SOLVERS:-0}"
ipopt_linear_solver="${IPOPT_LINEAR_SOLVER:-mumps}"

read -r -a phase_values <<< "${PHASES_LIST:-1 2 4 8}"
read -r -a state_values <<< "${NX_LIST:-6 20}"
read -r -a control_values <<< "${NU_LIST:-3 10}"
read -r -a parameter_values <<< "${NP_LIST:-0 1 4 8}"
read -r -a transcription_values <<< "${TRANSCRIPTIONS:-shooting radau}"
read -r -a sweep_modes <<< "${SWEEP_MODES:-fixed scaling}"

if [[ "${1-}" == "--help" ]]; then
    echo "Usage: [environment overrides] $0 > multiphase.csv"
    echo
    echo "Environment:"
    echo "  PHASES_LIST='1 2 4 8'"
    echo "  NX_LIST='6 20'"
    echo "  NU_LIST='3 10'"
    echo "  NP_LIST='0 1 4 8'"
    echo "  TRANSCRIPTIONS='shooting radau'"
    echo "  SWEEP_MODES='fixed scaling'"
    echo "  FIXED_TOTAL_STAGES=160"
    echo "  NODES_PER_PHASE=30"
    echo "  COLLOCATION_DEGREE=3"
    echo "  REPEATS=5"
    echo "  RUN_IPOPT=0"
    echo "  RUN_NATIVE_IPM=0"
    echo "  RUN_FULL_SOLVERS=0 (enables native IPM and IPOPT)"
    echo "  IPOPT_LINEAR_SOLVER=mumps"
    echo "  BENCH_CPU=8"
    exit 0
fi
if (( $# != 0 )); then
    echo "Unexpected arguments; use --help" >&2
    exit 2
fi
if [[ ! -x "$kkt" ]]; then
    echo "Benchmark executable not found: $kkt" >&2
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
run_case()
{
    local mode="$1"
    local stages="$2"
    local phases="$3"
    local nx="$4"
    local nu="$5"
    local np="$6"
    local transcription="$7"

    local args=(
        --problem synthetic
        --stages "$stages"
        --phases "$phases"
        --nx "$nx"
        --nu "$nu"
        --parameters "$np"
        --repeats "$repeats"
        --no-dense-validation
    )
    if [[ "$transcription" == "radau" ]]; then
        args+=(--collocation-degree "$collocation_degree")
    elif [[ "$transcription" != "shooting" ]]; then
        echo "Unknown transcription: $transcription" >&2
        exit 2
    fi
    if [[ "$run_full_solvers" == "1" ]]; then
        args+=(--full-solvers --ipopt-linear-solver "$ipopt_linear_solver")
    else
        if [[ "$run_native_ipm" == "1" ]]; then
            args+=(--native-ipm)
        fi
        if [[ "$run_ipopt" == "1" ]]; then
            args+=(--ipopt --ipopt-linear-solver "$ipopt_linear_solver")
        fi
    fi

    echo \
        "mode=$mode stages=$stages phases=$phases nx=$nx nu=$nu np=$np transcription=$transcription" \
        >&2
    local output
    output="$("${runner[@]}" "$kkt" "${args[@]}")"
    local header
    local row
    header="$(printf '%s\n' "$output" | head -n 1)"
    row="$(printf '%s\n' "$output" | tail -n 1)"
    if (( header_printed == 0 )); then
        printf 'sweep_mode,%s\n' "$header"
        header_printed=1
    fi
    printf '%s,%s\n' "$mode" "$row"
}

for mode in "${sweep_modes[@]}"; do
    for phases in "${phase_values[@]}"; do
        case "$mode" in
        fixed)
            stages="$fixed_total_stages"
            ;;
        scaling)
            stages=$((nodes_per_phase * phases))
            ;;
        *)
            echo "Unknown sweep mode: $mode" >&2
            exit 2
            ;;
        esac
        if (( stages < 2 * phases )); then
            echo \
                "Skipping invalid case: stages=$stages must be at least 2*phases=$((2 * phases))" \
                >&2
            continue
        fi

        for nx in "${state_values[@]}"; do
            for nu in "${control_values[@]}"; do
                for np in "${parameter_values[@]}"; do
                    for transcription in "${transcription_values[@]}"; do
                        run_case \
                            "$mode" "$stages" "$phases" "$nx" "$nu" "$np" \
                            "$transcription"
                    done
                done
            done
        done
    done
done
