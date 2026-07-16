import matplotlib.pyplot as plt
import numpy as np
import json

BOX_PLOTS = False

def latexify():
    params = {#'backend': 'ps',
              'axes.labelsize': 20,
              'axes.titlesize': 15,
              'legend.fontsize': 15,
              'xtick.labelsize': 15,
              'ytick.labelsize': 15,
              'text.usetex': True,
              'font.family': 'serif',
              'figure.figsize': [7,5],
              'text.latex.preamble': r'\usepackage{bm}',
              }
 
    plt.rcParams.update(params)

latexify()

def compute_flops(nx, nu, nxp):
    inverse = nxp*(nx + nu + 1)*(2*nxp - 1)
    factorization = (2/3)*nxp**3 - (1/2)*nxp**2 - (1/6)*nxp + \
                    (nx + nu + 1)*(nxp**2 - nxp) + \
                    (nx + nu + 1)*nxp**2
    return inverse, factorization

def compute_symbolic_flops(nnz_BA, nnz_J, nxp):
    inverse = nnz_BA + nnz_J + nxp**2
    factorization = nnz_BA + nnz_J
    return inverse, factorization

def approach_to_label(approach):
    if approach == 'us_inverse':
        return r'$\tilde{B} = -J^{-1}B$'
    elif approach == 'us_factorization':
        return r'$J\tilde{B} = -B$'
    
def get_theoretical_flops(n_trailers, sym_flop_scaling_factor):
    nx = 2 + n_trailers
    nu = 2
    nxp = nx

    flops_inverse = []
    flops_factorization = []
    sym_flops_inverse = []
    sym_flops_factorization = []
    for n in n_trailers:
        n_x = 2 + n + 1
        n_u = 2
        n_xp = n_x
        f_inv, f_fact = compute_flops(n_x, n_u, n_xp)
        flops_inverse.append(f_inv)
        flops_factorization.append(f_fact)

        nnz_BA = 6 + 2*n
        nnz_J = (n + 4)*(n+3)/2 - 1
        sf_inv, sf_fact = compute_symbolic_flops(nnz_BA, nnz_J, n_xp)
        scaling_factor = sym_flop_scaling_factor
        sym_flops_inverse.append(scaling_factor*sf_inv)
        sym_flops_factorization.append(scaling_factor*sf_fact)

    flops_inverse = np.array(flops_inverse)
    flops_factorization = np.array(flops_factorization)
    sym_flops_inverse = np.array(sym_flops_inverse)
    sym_flops_factorization = np.array(sym_flops_factorization)

    total_inverse = flops_inverse + sym_flops_inverse
    total_factorization = flops_factorization + sym_flops_factorization

    return total_inverse, total_factorization

def create_theoretical_scaling_figure():
    n_trailers = np.arange(0, 30)
    total_inverse, total_factorization = get_theoretical_flops(n_trailers, 100)

    plt.figure()
    plt.plot(n_trailers, total_factorization, label='Factorization')
    plt.plot(n_trailers, total_inverse, label='Inverse')
    # plt.yscale('log')
    plt.xlabel('Number of trailers')
    plt.ylabel('FLOPs')
    plt.title('Theoretical FLOPs for Jacobian Decomposition Approaches')
    plt.legend()
    plt.tight_layout()
    plt.savefig('unittest/implicit/figures/jacobian_decomposition_theoretical_flops.png', dpi=300)
    plt.close()

def create_figure(results):
    # create a bar plot for each approach
    x_key = 'n_trailers'
    x_vals = np.array(results[x_key])

    nb_approaches = len(results.keys()) - 1
    bar_width = 0.8 / nb_approaches
    plt.figure()

    for i, (approach, y_vals) in enumerate(results.items()):
        if approach == x_key:
            continue
        if BOX_PLOTS:
            plt.boxplot(y_vals, positions=x_vals + i * bar_width, widths=bar_width * 0.9, labels=[''] * len(x_vals))
        else:
            yy_vals = np.array([np.mean(y) for y in y_vals])
            plt.bar(x_vals + (i-1.5) * bar_width, yy_vals, width=bar_width, label=approach_to_label(approach))

    sym_flop_scaling = 130
    flop_overall_scaling = 1300
    total_inverse, total_factorization = get_theoretical_flops(x_vals, sym_flop_scaling)
    plt.plot(x_vals, total_factorization/flop_overall_scaling, 'k--', label=approach_to_label('us_factorization') + ' (theoretical)')
    plt.plot(x_vals, total_inverse/flop_overall_scaling, 'k:', label=approach_to_label('us_inverse') + ' (theoretical)')

    plt.xlabel('Number of trailers')
    plt.ylabel('Time (us)')
    # plt.title('Jacobian Decomposition Approaches Comparison')
    plt.ylim([0,260])
    plt.legend()
    plt.tight_layout()
    plt.savefig('unittest/implicit/figures/jacobian_decomposition_comparison.png', dpi=300)
    plt.close()

# load the file jacobian_decomposition.json
with open('build/unittest/jacobian_decomposition_results.json', 'r') as f:
    results = json.load(f)

create_theoretical_scaling_figure()
create_figure(results)