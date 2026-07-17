#!/usr/bin/env bash

set -euo pipefail

research_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$research_dir/.." && pwd)"
build_dir="${BUILD_DIR:-$repo_dir/build-research}"
benchmark="$build_dir/research/rank_deficient_implicit_kkt_benchmark"
bench_cpu="${BENCH_CPU-8}"

repeats="${REPEATS:-7}"
nodes_per_phase="${NODES_PER_PHASE:-6}"
dense_validation="${DENSE_VALIDATION:-0}"

read -r -a phase_values <<< "${PHASES_LIST:-1 2 4 8}"
read -r -a parameter_values <<< "${PARAMETERS_LIST:-0 2 8}"
read -r -a deficiency_values <<< "${RANK_DEFICIENCY_LIST:-0 1 2}"
read -r -a patterns <<< "${PHASE_PATTERNS:-alternating ramp zigzag}"
read -r -a modes <<< "${MODES:-gauss_newton exact_hessian inertia}"

if [[ "${1-}" == "--help" ]]; then
    echo "Usage: [environment overrides] $0 > rank-deficient.csv"
    echo
    echo "Environment:"
    echo "  PHASES_LIST='1 2 4 8'"
    echo "  PARAMETERS_LIST='0 2 8'"
    echo "  RANK_DEFICIENCY_LIST='0 1 2'"
    echo "  PHASE_PATTERNS='alternating ramp zigzag'"
    echo "  MODES='gauss_newton exact_hessian inertia'"
    echo "  NODES_PER_PHASE=6"
    echo "  REPEATS=7"
    echo "  DENSE_VALIDATION=0"
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
    local -a nodes=()
    local -a nx=()
    local -a nu=()
    local phase

    for ((phase = 0; phase < phases; ++phase)); do
        nodes+=("$((nodes_per_phase + phase % 2))")
        case "$pattern" in
        alternating)
            if (( phase % 2 == 0 )); then
                nx+=(4)
                nu+=(2)
            else
                nx+=(9)
                nu+=(4)
            fi
            ;;
        ramp)
            nx+=("$((3 + (3 * phase) % 10))")
            nu+=("$((1 + (2 * phase) % 5))")
            ;;
        zigzag)
            case "$((phase % 4))" in
            0)
                nx+=(12)
                nu+=(6)
                ;;
            1)
                nx+=(2)
                nu+=(0)
                ;;
            2)
                nx+=(8)
                nu+=(3)
                ;;
            3)
                nx+=(3)
                nu+=(1)
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
        make_phase_lists "$phases" "$pattern"
        for parameters in "${parameter_values[@]}"; do
            for deficiency in "${deficiency_values[@]}"; do
                for mode in "${modes[@]}"; do
                    cross_hessian=0
                    negative_curvature=0
                    case "$mode" in
                    gauss_newton)
                        ;;
                    exact_hessian)
                        cross_hessian=0.01
                        ;;
                    inertia)
                        negative_curvature=4
                        ;;
                    *)
                        echo "Unknown mode: $mode" >&2
                        exit 2
                        ;;
                    esac

                    args=(
                        --phase-nodes "$phase_nodes"
                        --phase-nx "$phase_nx"
                        --phase-nu "$phase_nu"
                        --rank-deficiency "$deficiency"
                        --parameters "$parameters"
                        --cross-hessian-scale "$cross_hessian"
                        --negative-curvature "$negative_curvature"
                        --repeats "$repeats"
                    )
                    if [[ "$dense_validation" != "1" ]]; then
                        args+=(--no-dense-validation)
                    fi

                    echo \
                        "mode=$mode pattern=$pattern phases=$phases p=$parameters deficiency=$deficiency nx=$phase_nx nu=$phase_nu" \
                        >&2
                    output="$("${runner[@]}" "$benchmark" "${args[@]}")"
                    header="$(printf '%s\n' "$output" | head -n 1)"
                    row="$(printf '%s\n' "$output" | tail -n 1)"
                    if (( header_printed == 0 )); then
                        printf 'mode,phase_pattern,%s\n' "$header"
                        header_printed=1
                    fi
                    printf '%s,%s,%s\n' "$mode" "$pattern" "$row"
                done
            done
        done
    done
done
