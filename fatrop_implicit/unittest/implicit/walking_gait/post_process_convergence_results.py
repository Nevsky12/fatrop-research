import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
from post_process_convergence_parser import parse_convergence_from_output

def latexify():
    params = {#'backend': 'ps',
              'axes.labelsize': 18,
              'axes.titlesize': 15,
              'legend.fontsize': 8,
              'xtick.labelsize': 11,
              'ytick.labelsize': 11,
              'text.usetex': True,
              'font.family': 'serif',
              'figure.figsize': [6, 3],
              'text.latex.preamble': r'\usepackage{bm}',
              }
 
    plt.rcParams.update(params)

def get_first_diff_idx(conv_ocp, conv_accelerated, key, rel_diff_tolerance):
    first_diff_idx = -1
    for i in range(len(conv_ocp['iteration'])):
        if first_diff_idx == -1 and abs(conv_ocp[key][i] - conv_accelerated[key][i]) > rel_diff_tolerance:
            first_diff_idx = i
    return first_diff_idx

latexify()

FOLDER = 'unittest/implicit/walking_gait/stored_solutions'
# FOLDER = 'stored_solutions'

# parse convergence data from solver output files
conv_ocp = parse_convergence_from_output(f'{FOLDER}/solver_output_ocp_type.txt')
conv_accelerated = parse_convergence_from_output(f'{FOLDER}/solver_output_accelerated_ocp_type.txt')

# Calculate ||f(xk)-f(x^*)||_2 for each method
# f(x^*) is the final objective value
f_opt_ocp = conv_ocp['obj'][-1]
f_opt_accelerated = conv_accelerated['obj'][-1]

obj_diff_ocp = np.array(conv_ocp['obj']) - f_opt_ocp
obj_diff_accelerated = np.array(conv_accelerated['obj']) - f_opt_accelerated

xlim = [0, max(conv_ocp['iteration'][-1], conv_accelerated['iteration'][-1]) + 5]

rel_diff_tolerance = 1e-10  # Tolerance for considering two values as different
first_diff_idx_primal_feasibility = get_first_diff_idx(conv_ocp, conv_accelerated, 'primal_feasibility', rel_diff_tolerance)
first_diff_idx_dual_feasibility = get_first_diff_idx(conv_ocp, conv_accelerated, 'dual_feasibility', rel_diff_tolerance)
first_diff_idx_objective = get_first_diff_idx(conv_ocp, conv_accelerated, 'obj', rel_diff_tolerance)

# Create separate figures for each KPI
# Figure 1: Primal Feasibility
fig1, ax1 = plt.subplots()
ax1.semilogy(conv_accelerated['iteration'], conv_accelerated['primal_feasibility'], color='grey', linestyle='-', label='Blocked LU', linewidth=2)
ax1.semilogy(conv_ocp['iteration'], conv_ocp['primal_feasibility'], 'k-', label='Normal LU', linewidth=2)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Primal Feasibility')
# ax1.set_title('Primal Feasibility Convergence')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', frameon=True)
ax1.axvline(conv_ocp['iteration'][first_diff_idx_primal_feasibility], color='r', linestyle='-', label='First Divergence', linewidth=1)
plt.xlim(xlim)
plt.tight_layout()
plt.savefig(f'{FOLDER}/../visualization_output/convergence_primal_feasibility.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Dual Feasibility
fig2, ax2 = plt.subplots()
ax2.semilogy(conv_accelerated['iteration'], conv_accelerated['dual_feasibility'], color='grey', linestyle='-', label='Blocked LU', linewidth=2)
ax2.semilogy(conv_ocp['iteration'], conv_ocp['dual_feasibility'], 'k-', label='Normal LU', linewidth=2)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Dual Feasibility')
# ax2.set_title('Dual Feasibility Convergence')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='best', frameon=True)
ax2.axvline(conv_ocp['iteration'][first_diff_idx_dual_feasibility], color='r', linestyle='-', label='First Divergence', linewidth=1)
plt.xlim(xlim)
plt.tight_layout()
plt.savefig(f'{FOLDER}/../visualization_output/convergence_dual_feasibility.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Objective Value Difference
fig3, ax3 = plt.subplots()
ax3.semilogy(conv_accelerated['iteration'], np.abs(obj_diff_accelerated), color='grey', linestyle='-', label='Blocked LU', linewidth=2)
ax3.semilogy(conv_ocp['iteration'], np.abs(obj_diff_ocp), 'k-', label='Normal LU', linewidth=2)
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Optimality gap')
# ax3.set_title('Objective Value Difference Convergence')
ax3.grid(True, alpha=0.3)
ax3.legend(loc='best', frameon=True)
ax3.axvline(conv_ocp['iteration'][first_diff_idx_objective], color='r', linestyle='-', label='First Divergence', linewidth=1)
plt.xlim(xlim)
plt.tight_layout()
plt.savefig(f'{FOLDER}/../visualization_output/convergence_objective_difference.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate relative difference in objective value
# Handle different number of iterations by using only the common length
min_len = min(len(conv_ocp['obj']), len(conv_accelerated['obj']))
obj_ocp_common = np.array(conv_ocp['obj'][:min_len])
obj_accelerated_common = np.array(conv_accelerated['obj'][:min_len])
iteration_common = conv_ocp['iteration'][:min_len]

# Relative difference: (obj_accelerated - obj_ocp) / obj_ocp
relative_diff = np.abs((obj_accelerated_common - obj_ocp_common) / obj_ocp_common)

# Figure 4: Relative Difference in Objective (from iteration 0)
fig4, ax4 = plt.subplots()
ax4.semilogy(iteration_common, 100 * relative_diff, 'k-', linewidth=2)
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Relative Difference (\%)')
ax4.grid(True, alpha=0.3)
ax4.axhline(0, color='grey', linestyle='--', linewidth=1)
plt.xlim(xlim)
plt.ylim(1e-12, 1e2)
plt.tight_layout()
plt.savefig(f'{FOLDER}/../visualization_output/convergence_relative_difference_full.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 5: Relative Difference in Objective (from last iteration)
obj_ocp_common = np.array(conv_ocp['obj'][-min_len:])
obj_accelerated_common = np.array(conv_accelerated['obj'][-min_len:])
relative_diff_from_last = np.abs((obj_accelerated_common - obj_ocp_common) / np.abs(obj_ocp_common[-1]))

fig5, ax5 = plt.subplots()
ax5.semilogy(iteration_common, 100 * relative_diff_from_last, 'k-', linewidth=2)
ax5.set_xlabel('Iteration')
ax5.set_ylabel('Relative Difference from Final (\%)')
ax5.grid(True, alpha=0.3)
ax5.axhline(0, color='grey', linestyle='--', linewidth=1)
plt.xlim(xlim)
plt.ylim(1e-12, 1e2)
plt.tight_layout()
plt.savefig(f'{FOLDER}/../visualization_output/convergence_relative_difference_from_final.png', dpi=300, bbox_inches='tight')
plt.close()

print("Convergence plots saved:")
print("  - convergence_primal_feasibility.png")
print("  - convergence_dual_feasibility.png")
print("  - convergence_objective_difference.png")
print("  - convergence_relative_difference_full.png")
print("  - convergence_relative_difference_from_final.png")

