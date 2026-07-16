import casadi
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import example_robot_data as erd

# from unittest.implicit.quadruped.helper import *
import sys
sys.path.append("unittest/implicit/quadruped")
from helper import *
import subprocess
import shutil

manip = ContactManipulator()
expl, impl = manip.get_discrete_dynamics()
# manip.simulate_joint_configurations()
# manip.simulate()
manip.example_opti()

dt = 0.03
quad = QuadrupedDynamics(timestep=dt)

explicit_integrator, implicit_integrator = quad.create_discrete_dynamics()

functions = [explicit_integrator, implicit_integrator]
names = ["quadruped_explicit_integrator", "quadruped_implicit_integrator"]

for f, n in zip(functions, names):
    # f.generate(f"{n}.c")
    # subprocess.run(["gcc", "-fPIC", "-shared", f"{n}.c", "-o", f"{n}.so"], check=True)
    # shutil.move(f"{n}.c", f"build/unittest/{n}.c")
    # shutil.move(f"{n}.so", f"build/unittest/{n}.so")

    f.save(f"build/unittest/{n}.casadi")
