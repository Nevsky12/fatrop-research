#!/usr/bin/env bash

set -euo pipefail

research_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$research_dir/.." && pwd)"
build_dir="${BUILD_DIR:-$repo_dir/build-research}"
benchmark="$build_dir/research/nonlinear_global_parameter_sqp_benchmark"
bench_cpu="${BENCH_CPU-8}"

repeats="${REPEATS:-3}"
nodes_per_phase="${NODES_PER_PHASE:-4}"
max_iterations="${MAX_ITERATIONS:-80}"
tolerance="${TOLERANCE:-1e-8}"
dense_validation="${DENSE_VALIDATION:-0}"
run_ipopt="${RUN_IPOPT:-0}"
run_native_ipm="${RUN_NATIVE_IPM:-0}"
run_full_solvers="${RUN_FULL_SOLVERS:-0}"
ipopt_linear_solver="${IPOPT_LINEAR_SOLVER:-mumps}"

read -r -a phase_values <<< "${PHASES_LIST:-1 2 4 8}"
read -r -a parameter_values <<< "${PARAMETERS_LIST:-0 2 8}"
read -r -a patterns <<< "${PHASE_PATTERNS:-alternating ramp zigzag}"
read -r -a scales <<< "${DIMENSION_SCALES:-1 2}"
read -r -a transcriptions <<< "${TRANSCRIPTIONS:-radau1 radau3}"

if [[ "${1-}" == "--help" ]]; then
    echo "Usage: [environment overrides] $0 > nonlinear-sqp.csv"
    echo
    echo "Environment:"
    echo "  PHASES_LIST='1 2 4 8'"
    echo "  PARAMETERS_LIST='0 2 8'"
    echo "  PHASE_PATTERNS='alternating ramp zigzag'"
    echo "  DIMENSION_SCALES='1 2'"
    echo "  TRANSCRIPTIONS='radau1 radau3'"
    echo "  NODES_PER_PHASE=4"
    echo "  MAX_ITERATIONS=80"
    echo "  TOLERANCE=1e-8"
    echo "  REPEATS=3"
    echo "  DENSE_VALIDATION=0"
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

join_by_comma()
{
    local result=""
    local value
    for value in "$@"; do
        if [[ -n "$result" ]]; then
            result+=","
        fi
        result+="$value"
    done
    printf '%s' "$result"
}

make_phase_lists()
{
    local phases="$1"
    local pattern="$2"
    local scale="$3"
    local -a nodes=()
    local -a nx=()
    local -a nu=()
    local phase

    for ((phase = 0; phase < phases; ++phase)); do
        nodes+=("$((nodes_per_phase + phase % 2))")
        case "$pattern" in
        alternating)
            if (( phase % 2 == 0 )); then
                nx+=("$((4 * scale))")
                nu+=("$((2 * scale))")
            else
                nx+=("$((9 * scale))")
                nu+=("$((4 * scale))")
            fi
            ;;
        ramp)
            nx+=("$(((3 + (3 * phase) % 10) * scale))")
            nu+=("$(((1 + (2 * phase) % 5) * scale))")
            ;;
        zigzag)
            case "$((phase % 4))" in
            0)
                nx+=("$((12 * scale))")
                nu+=("$((6 * scale))")
                ;;
            1)
                nx+=("$((2 * scale))")
                nu+=(0)
                ;;
            2)
                nx+=("$((8 * scale))")
                nu+=("$((3 * scale))")
                ;;
            3)
                nx+=("$((3 * scale))")
                nu+=("$scale")
                ;;
            esac
            ;;
        *)
            echo "Unknown phase pattern: $pattern" >&2
            exit 2
            ;;
        esac
    done

    phase_nodes="$(join_by_comma "${nodes[@]}")"
    phase_nx="$(join_by_comma "${nx[@]}")"
    phase_nu="$(join_by_comma "${nu[@]}")"
}

header_printed=0
for phases in "${phase_values[@]}"; do
    for pattern in "${patterns[@]}"; do
        for scale in "${scales[@]}"; do
            make_phase_lists "$phases" "$pattern" "$scale"
            for parameters in "${parameter_values[@]}"; do
                for transcription in "${transcriptions[@]}"; do
                    case "$transcription" in
                    radau1)
                        degree=1
                        ;;
                    radau2)
                        degree=2
                        ;;
                    radau3)
                        degree=3
                        ;;
                    *)
                        echo "Unknown transcription: $transcription" >&2
                        exit 2
                        ;;
                    esac

                    args=(
                        --phase-nodes "$phase_nodes"
                        --phase-nx "$phase_nx"
                        --phase-nu "$phase_nu"
                        --parameters "$parameters"
                        --collocation-degree "$degree"
                        --max-iterations "$max_iterations"
                        --tolerance "$tolerance"
                        --repeats "$repeats"
                    )
                    if [[ "$dense_validation" != "1" ]]; then
                        args+=(--no-dense-validation)
                    fi
                    if [[ "$run_full_solvers" == "1" ]]; then
                        args+=(
                            --full-solvers
                            --ipopt-linear-solver "$ipopt_linear_solver"
                        )
                    else
                        if [[ "$run_native_ipm" == "1" ]]; then
                            args+=(--native-ipm)
                        fi
                        if [[ "$run_ipopt" == "1" ]]; then
                            args+=(
                                --ipopt
                                --ipopt-linear-solver "$ipopt_linear_solver"
                            )
                        fi
                    fi

                    echo \
                        "pattern=$pattern scale=$scale phases=$phases p=$parameters transcription=$transcription nx=$phase_nx nu=$phase_nu" \
                        >&2
                    output="$("${runner[@]}" "$benchmark" "${args[@]}")"
                    header="$(printf '%s\n' "$output" | head -n 1)"
                    row="$(printf '%s\n' "$output" | tail -n 1)"
                    if (( header_printed == 0 )); then
                        printf 'phase_pattern,dimension_scale,%s\n' "$header"
                        header_printed=1
                    fi
                    printf '%s,%s,%s\n' "$pattern" "$scale" "$row"
                done
            done
        done
    done
done
