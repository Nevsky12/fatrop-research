#!/bin/bash

# This script runs the walking gait optimization for both problem types.

# Go to the directory of the script
cd "$(dirname "$0")"

# --- Run for accelerated_ocp_type ---
echo "Running for accelerated_ocp_type"

# Modify config.yaml for accelerated_ocp_type
sed -i "s/problem_type: 'ocp_type'/problem_type: 'accelerated_ocp_type'/" config.yaml
./run_gait_tests_multiple_times.sh

# --- Run for ocp_type ---
echo "Running for ocp_type"

# Modify config.yaml for ocp_type
sed -i "s/problem_type: 'accelerated_ocp_type'/problem_type: 'ocp_type'/" config.yaml
./run_gait_tests_multiple_times.sh
