#!/usr/bin/env bash

set -euo pipefail

research_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$research_dir/.." && pwd)"
build_dir="${BUILD_DIR:-$repo_dir/build-research}"
benchmark="$build_dir/research/heterogeneous_phase_kkt_benchmark"
bench_cpu="${BENCH_CPU-8}"

repeats="${REPEATS:-5}"
nodes_per_phase="${NODES_PER_PHASE:-12}"
dense_validation="${DENSE_VALIDATION:-0}"

read -r -a phase_values <<< "${PHASES_LIST:-1 2 4 8}"
read -r -a global_parameter_values <<< "${GLOBAL_PARAMETERS_LIST:-0 2 8}"
read -r -a patterns <<< "${PHASE_PATTERNS:-alternating ramp zigzag}"
read -r -a transcriptions <<< "${TRANSCRIPTIONS:-shooting radau1 radau2 radau3}"

if [[ "${1-}" == "--help" ]]; then
    echo "Usage: [environment overrides] $0 > heterogeneous.csv"
    echo
    echo "Environment:"
    echo "  PHASES_LIST='1 2 4 8'"
    echo "  GLOBAL_PARAMETERS_LIST='0 2 8'"
    echo "  PHASE_PATTERNS='alternating ramp zigzag'"
    echo "  TRANSCRIPTIONS='shooting radau1 radau2 radau3'"
    echo "  NODES_PER_PHASE=12"
    echo "  REPEATS=5"
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
    local -a parameters=()
    local phase

    for ((phase = 0; phase < phases; ++phase)); do
        nodes+=("$((nodes_per_phase + phase % 3))")
        case "$pattern" in
        alternating)
            if (( phase % 2 == 0 )); then
                nx+=(4)
                nu+=(2)
                parameters+=(1)
            else
                nx+=(10)
                nu+=(5)
                parameters+=(3)
            fi
            ;;
        ramp)
            nx+=("$((3 + (3 * phase) % 11))")
            nu+=("$((1 + (2 * phase) % 6))")
            parameters+=("$((phase % 4))")
            ;;
        zigzag)
            case "$((phase % 4))" in
            0)
                nx+=(12)
                nu+=(6)
                parameters+=(2)
                ;;
            1)
                nx+=(2)
                nu+=(0)
                parameters+=(0)
                ;;
            2)
                nx+=(8)
                nu+=(3)
                parameters+=(4)
                ;;
            3)
                nx+=(3)
                nu+=(1)
                parameters+=(1)
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
    phase_parameters="$(join_by_comma "${parameters[@]}")"
}

header_printed=0
for phases in "${phase_values[@]}"; do
    for pattern in "${patterns[@]}"; do
        make_phase_lists "$phases" "$pattern"
        for global_parameters in "${global_parameter_values[@]}"; do
            for transcription in "${transcriptions[@]}"; do
                degree=0
                case "$transcription" in
                shooting)
                    degree=0
                    ;;
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
                    --phase-parameters "$phase_parameters"
                    --global-parameters "$global_parameters"
                    --collocation-degree "$degree"
                    --repeats "$repeats"
                )
                if [[ "$dense_validation" != "1" ]]; then
                    args+=(--no-dense-validation)
                fi

                echo \
                    "pattern=$pattern phases=$phases global_p=$global_parameters transcription=$transcription nx=$phase_nx nu=$phase_nu local_p=$phase_parameters" \
                    >&2
                output="$("${runner[@]}" "$benchmark" "${args[@]}")"
                header="$(printf '%s\n' "$output" | head -n 1)"
                row="$(printf '%s\n' "$output" | tail -n 1)"
                if (( header_printed == 0 )); then
                    printf 'phase_pattern,%s\n' "$header"
                    header_printed=1
                fi
                printf '%s,%s\n' "$pattern" "$row"
            done
        done
    done
done
