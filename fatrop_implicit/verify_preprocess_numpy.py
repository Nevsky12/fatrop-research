import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "build_docker"))
from preprocess_info import *
from test_debug_helper import *

BAbt_expected, GuGx_expected, FuFx_expected, RSQrqt_expected, Gg_eqt_expected, Gg_ineqt_expected, Dl_list, Dr_list, Dl_inv_list, Dr_inv_list = get_expected_matrices(K, nu, nx, r, ng_eq, ng_ineq, modified_nu, modified_nx, modified_ng_eq, modified_ng_ineq, 
                          RSQrqt_original, GuGx_original, FuFx_original, Gg_eqt_original, Gg_ineqt_original, BAbt_original, D_x,
                          Pl, Pr, L, U, 
                          RSQrqt, GuGx, FuFx, Gg_eqt, Gg_ineqt, BAbt, Jt)

# post-processing code
primal_x_offset = 0
for k in range(K):
    if k > 0:
        ukxk = x[primal_x_offset:primal_x_offset+nu[k]+nx[k]]
        uk = ukxk[:nu[k]].copy()
        sk = ukxk[nu[k]:modified_nu[k]].copy()
        xk = ukxk[modified_nu[k]:].copy()
        x[primal_x_offset+nu[k]:primal_x_offset + nu[k] + r[k-1]] = xk
        x[primal_x_offset+nu[k]+r[k-1]:primal_x_offset+nu[k]+nx[k]] = sk
        x[primal_x_offset+nu[k]:primal_x_offset+nu[k]+nx[k]] = \
            Dr_inv_list[k-1] @ x[primal_x_offset+nu[k]:primal_x_offset+nu[k]+nx[k]] 
    
    primal_x_offset += nu[k] + nx[k]
x_expected = x

original_eq_path_offset = 0
original_eq_dyn_offset = 0
new_eq_path_offset = 0
new_eq_dyn_offset = 0
original_total_nb_eq_path = sum(modified_ng_eq)
original_total_nb_dyn_eq = sum(modified_nx[1:])
original_eq_path = eq_mult[:original_total_nb_eq_path].copy()
original_eq_dyn = eq_mult[original_total_nb_eq_path:original_total_nb_eq_path+original_total_nb_dyn_eq].copy()
new_total_nb_eq_path = sum(ng_eq)
new_total_nb_dyn_eq = sum(nx[1:])
new_eq_path = np.zeros((new_total_nb_eq_path,))
new_eq_dyn = np.zeros((new_total_nb_dyn_eq,))
for k in range(K-1):
    lmbd = original_eq_path[original_eq_path_offset:original_eq_path_offset+modified_ng_eq[k]].copy()
    pi = original_eq_dyn[original_eq_dyn_offset:original_eq_dyn_offset+modified_nx[k+1]].copy()

    true_pi = np.block([pi, lmbd[ng_eq[k]:]])
    lmbd = lmbd[:ng_eq[k]].copy()
    
    print(f"\n[{k}] eq_mult before: {true_pi}")
    true_pi = Dl_inv_list[k].T @ true_pi
    print(f"[{k}] eq_mult after: {true_pi}")

    new_eq_path[new_eq_path_offset:new_eq_path_offset+ng_eq[k]] = lmbd
    new_eq_dyn[new_eq_dyn_offset:new_eq_dyn_offset+nx[k+1]] = true_pi

    original_eq_path_offset += modified_ng_eq[k]
    original_eq_dyn_offset += modified_nx[k+1]
    new_eq_path_offset += ng_eq[k]
    new_eq_dyn_offset += nx[k+1]
new_eq_path[new_eq_path_offset:] = original_eq_path[original_eq_path_offset:].copy()

eq_mult_expected = np.block([new_eq_path, new_eq_dyn])

from final_solution import *
x = x[:,0]
eq_mult = eq_mult[:,0]

print(f"x error norm: {np.linalg.norm(x - x_expected)}")
print(f"eq_mult error norm: {np.linalg.norm(eq_mult - eq_mult_expected)}")

for i in range(len(x)):
    if abs(x[i] - x_expected[i]) > 1e-6:
        print(f"x[{i}] error: {x[i] - x_expected[i]}")

for i in range(eq_mult.shape[0]):
    if i == new_total_nb_eq_path:
        print(f"--- eq_mult[{i}] is the first multiplier for dynamics ---")
    if abs(eq_mult[i] - eq_mult_expected[i]) > 1e-5:
        # print(f"eq_mult[{i}] error: {eq_mult[i] - eq_mult_expected[i]}")
        print(f"eq_mult[{i}] error: {eq_mult[i]} - {eq_mult_expected[i]}")


### Solution to preprocessed problem
blocks = GetBlockMatrices(K, modified_nu, modified_nx, modified_ng_eq, RSQrqt, GuGx, FuFx, Gg_eqt, BAbt)
KKT, rhs = GetKKT(K, modified_nu, modified_nx, modified_ng_eq, blocks['R'], blocks['S'], blocks['Q'], blocks['Gu'], blocks['Gx'], blocks['Fu'], blocks['Fx'], blocks['Hu'], blocks['Hx'], blocks['B'], blocks['A'], blocks['r'], blocks['q'], blocks['h'], blocks['b'])
solution_vector = np.linalg.solve(KKT, -rhs)
solution = extract_solultion(K, modified_nu, modified_nx, modified_ng_eq, solution_vector)
# print_solution(solution)

### Solution to original problem
blocks = GetBlockMatrices(K, nu, nx, ng_eq, RSQrqt_original, GuGx_original, FuFx_original, Gg_eqt_original, BAbt_original)
KKT, rhs = GetKKT(K, nu, nx, ng_eq, blocks['R'], blocks['S'], blocks['Q'], blocks['Gu'], blocks['Gx'], blocks['Fu'], blocks['Fx'], blocks['Hu'], blocks['Hx'], blocks['B'], blocks['A'], blocks['r'], blocks['q'], blocks['h'], blocks['b'], Jt=Jt)
solution_vector = np.linalg.solve(KKT, -rhs)
solution = extract_solultion(K, nu, nx, ng_eq, solution_vector)
# print_solution(solution)