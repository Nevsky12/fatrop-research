#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir=${1:-"${script_dir}/../build-research"}
binary="${build_dir}/research/nonlinear_interval_scoped_sqp_benchmark"

if [[ ! -x "${binary}" ]]; then
    echo "benchmark executable not found: ${binary}" >&2
    echo "configure with -DBUILD_RESEARCH_BENCHMARKS=ON and build it first" >&2
    exit 1
fi

repeats=${REPEATS:-5}
linear_solver=${IPOPT_LINEAR_SOLVER:-ma57}
cpu=${CPU:-}
path_inequalities=${PATH_INEQUALITIES:-2}

ipopt_arguments=(
    --ipopt
    --ipopt-linear-solver "${linear_solver}"
)
if [[ -n "${HSL_LIBRARY:-}" ]]; then
    ipopt_arguments+=(--ipopt-hsl-library "${HSL_LIBRARY}")
fi

first_case=1
run_case()
{
    local command=(
        "${binary}"
        --path-inequalities "${path_inequalities}"
        "$@"
        --repeats "${repeats}"
        "${ipopt_arguments[@]}"
    )
    if [[ -n "${cpu}" ]]; then
        command=(taskset -c "${cpu}" "${command[@]}")
    fi
    if (( first_case )); then
        "${command[@]}"
        first_case=0
    else
        "${command[@]}" | tail -n 1
    fi
}

# Scaling in phase count with deliberately heterogeneous state/control sizes.
# Static-parameter dimensions remain nonzero in every scaling case.
for degree in 1 3; do
    run_case \
        --phase-nodes 5 \
        --phase-nx 4 \
        --phase-nu 2 \
        --collocation-degree "${degree}"
    run_case \
        --phase-nodes 5,4 \
        --phase-nx 4,6 \
        --phase-nu 2,3 \
        --collocation-degree "${degree}"
    run_case \
        --phase-nodes 5,4,6,5 \
        --phase-nx 4,6,3,5 \
        --phase-nu 2,3,1,2 \
        --collocation-degree "${degree}"
    run_case \
        --phase-nodes 5,4,6,5,4,6,5,4 \
        --phase-nx 4,6,3,5,7,4,6,3 \
        --phase-nu 2,3,1,2,3,2,1,2 \
        --collocation-degree "${degree}"
done

# Orthogonal x/u/p profiles at four phases and Radau degree three.  The first
# case has no user static parameters (only interface separator states), the
# second has narrow scopes, and the third stresses a wide active separator.
run_case \
    --phase-nodes 4,5,4,5 \
    --phase-nx 2,3,2,4 \
    --phase-nu 1,1,2,1 \
    --phase-parameters 0 \
    --interface-parameters 0 \
    --segment-parameters 0 \
    --global-parameters 0 \
    --path-inequalities 1 \
    --collocation-degree 3

run_case \
    --phase-nodes 4,5,4,5 \
    --phase-nx 2,3,2,4 \
    --phase-nu 1,1,2,1 \
    --phase-parameters 1 \
    --interface-parameters 1 \
    --segment-parameters 1 \
    --segment-width 2 \
    --segment-stride 1 \
    --global-parameters 1 \
    --path-inequalities 2 \
    --collocation-degree 3

run_case \
    --phase-nodes 5,4,6,5 \
    --phase-nx 8,12,6,10 \
    --phase-nu 4,6,3,5 \
    --phase-parameters 6 \
    --interface-parameters 4 \
    --segment-parameters 8 \
    --segment-width 4 \
    --segment-stride 2 \
    --global-parameters 12 \
    --path-inequalities 4 \
    --collocation-degree 3
