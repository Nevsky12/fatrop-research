from casadi import *

# load functions
folder = "casadi_funcs"
# opti_f = Function.load(f"{folder}/test_gait_shortcut_matlab.casadi")
# opti_f = Function.load(f"{folder}/test_gait_shortcut_python.casadi")
# opti_f = Function.load(f"{folder}/test_gait_shortcut_python_reformulated.casadi")
opti_f = Function.load(f"{folder}/test_gait_shortcut_python_reformulated_ocp_type.casadi")
opti_f()
opti_f = Function.load(f"{folder}/test_gait_shortcut_python_reformulated_accelerated_ocp_type.casadi")
opti_f()

