import numpy as np
from test_debug_helper import *
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "build_docker"))

from factorization_info import *
from preprocess_info import *


### Original problem ###
blocks = GetBlockMatrices(K, modified_nu, modified_nx, modified_ng_eq, RSQrqt, GuGx, FuFx, Gg_eqt, BAbt)
KKT, rhs = GetKKT(K, modified_nu, modified_nx, modified_ng_eq, blocks['R'], blocks['S'], blocks['Q'], blocks['Gu'], blocks['Gx'], blocks['Fu'], blocks['Fx'], blocks['Hu'], blocks['Hx'], blocks['B'], blocks['A'], blocks['r'], blocks['q'], blocks['h'], blocks['b'])
solution = np.linalg.solve(KKT, -rhs)
extracted_solution = extract_solultion(K, modified_nu, modified_nx, modified_ng_eq, solution)
# print_solution(extracted_solution)

solution2 = Solve(K, modified_nu, modified_nx, modified_ng_eq, blocks['R'], blocks['S'], blocks['Q'], blocks['Gu'], blocks['Gx'], blocks['Fu'], blocks['Fx'], blocks['Hu'], blocks['Hx'], blocks['B'], blocks['A'], blocks['r'], blocks['q'], blocks['h'], blocks['b'], Pl_r, Pr_r, L_r, U_r, Lmbd, rank_k_values, Hut)
# print_solution(solution2)

### check differences
# check controls
for k in range(K):
    n = np.linalg.norm(extracted_solution['u'][k] - solution2['u'][k])
    if n > 1e-4:
        print(f"Difference in control u[{k}]: {n}")
# check states
for k in range(K):
    n = np.linalg.norm(extracted_solution['x'][k] - solution2['x'][k])
    if n > 1e-4:
        print(f"Difference in state x[{k}]: {n}")
# check multipliers
for k in range(K):
    n = np.linalg.norm(extracted_solution['lambda'][k] - solution2['lambda'][k])
    if n > 1e-4:
        print(f"Difference in multiplier lambda[{k}]: {n}")
for k in range(K-1):
    n = np.linalg.norm(extracted_solution['pi'][k] - solution2['pi'][k])
    if n > 1e-4:
        print(f"Difference in multiplier pi[{k}]: {n}")

