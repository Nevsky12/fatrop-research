#!/bin/bash

# Go to the directory of the script
cd "$(dirname "$0")"

# Modify config.yaml for first case
sed -i "s/save_opti_f: False/save_opti_f: True/" config.yaml
sed -i "s/load_opti_f: True/load_opti_f: False/" config.yaml
sed -i "s/perform_code_generation: False/perform_code_generation: False/" config.yaml
sed -i "s/load_solution: True/load_solution: False/" config.yaml
sed -i "s/visualize_solution: True/visualize_solution: False/" config.yaml
sed -i "s/file_name_appendix: ''/file_name_appendix: ''/" config.yaml

# Run the python script
python3 test_gait_shortcut_reformulated.py

# run a few more times by loading the opti_f 
sed -i "s/save_opti_f: True/save_opti_f: False/" config.yaml
sed -i "s/load_opti_f: False/load_opti_f: True/" config.yaml
for i in {1..5}
do
    sed -i "s/file_name_appendix: ''/file_name_appendix: '_run_$i'/" config.yaml
    sed -i "s/file_name_appendix: '_run_$((i-1))'/file_name_appendix: '_run_$i'/" config.yaml
    echo "Run $i with loaded opti_f"
    python3 test_gait_shortcut_reformulated.py
done

sed -i "s/file_name_appendix: '_run_5'/file_name_appendix: ''/" config.yaml
