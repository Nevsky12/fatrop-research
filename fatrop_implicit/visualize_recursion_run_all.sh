#!/bin/bash

SETTINGS_FILE="visualize_recursion_settings.json"
PYTHON_FILE="visualize_recursion.py"

# Check if python file exists
if [ ! -f "$PYTHON_FILE" ]; then
    echo "Error: $PYTHON_FILE not found."
    exit 1
fi

# Check if settings file exists
if [ ! -f "$SETTINGS_FILE" ]; then
    echo "Error: $SETTINGS_FILE not found."
    exit 1
fi

update_and_run() {
    local config_name=$1
    local explicit=$2
    local reformulated=$3
    local implicit=$4

    echo "-------------------------------------------"
    echo "Running configuration: $config_name"
    echo "-------------------------------------------"

    # Update only the three flags, leaving all other fields untouched
    python3 -c "
import json

with open('$SETTINGS_FILE', 'r') as f:
    settings = json.load(f)

settings['EXPLICIT']     = $explicit
settings['REFORMULATED'] = $reformulated
settings['IMPLICIT']     = $implicit

with open('$SETTINGS_FILE', 'w') as f:
    json.dump(settings, f, indent=4)
"

    echo "Updated $SETTINGS_FILE:"
    cat "$SETTINGS_FILE"
    echo ""

    # Run the python script
    python3 "$PYTHON_FILE"

    echo ""
}

# Configuration 1: EXPLICIT=true
update_and_run "EXPLICIT"     True  False False

# Configuration 2: REFORMULATED=true
update_and_run "REFORMULATED" False True  False

# Configuration 3: IMPLICIT=true
update_and_run "IMPLICIT"     False False True

echo "-------------------------------------------"
echo "All configurations completed."
echo "-------------------------------------------"