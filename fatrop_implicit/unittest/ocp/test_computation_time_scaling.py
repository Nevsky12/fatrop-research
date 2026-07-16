import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sympy as sp

import sys
sys.path.append('build/unittest/')
from scaling_test_results_explicit import *
from scaling_test_results_reformulation import *
from scaling_test_results_implicit import *

def latexify():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
    })
latexify()

def translate_label_names(label):
    if label == "nx":
        return "$n_x$"
    elif label == "nu":
        return "$n_u$"
    elif label == "ng":
        return "$n_{g,eq}$"
    elif label == "ng_ineq":
        return "$n_{g,ineq}$"
    else:
        return label

def preprocessing(nx, nu, nxp):
    return nxp*(nx + nu + 1)*(2*nxp - 1) + \
        (nx + nu)*(nx + nu + 1)*(4*nx - 1)

def postprocessing(nx, nu, nxp):
    return nxp*(2*nu - 1) + nxp*(2*nx - 1) + nxp*(2*nxp - 1)

def backwardrecursion(nx, nu, nxp, ngp, ngi, rho, gamma):
    # w = min(gamma, nu + nx + 1)
    w = gamma
    return nxp*(nu + nx + 1)*(2*nxp - 1) + \
        0.5*(nu + nx)*(nu + nx + 1)*(2*nxp - 1) + \
        (nu + nx + 1)*ngp*(2*nx - 1) + \
        0.5*(nu + nx)*(nu + nx + 1)*(2*ngi - 1) + \
        2*w**3/3 - 2*w/3 + \
        (nu - rho + nx + 1)*(nu + nx)*(2*rho - 1) + \
        0.5*(nu - rho + nx - 1)*(nu + nx - rho)*(2*rho - 1) + \
        (nu - rho)**3/3 + \
        0.5*(nx + 1)*nx*(2*(nu - rho) - 1)

def backwardrecursion_initial_stage(nx, nu, gamma, rho):
    # w = min(gamma, nu + nx + 1)
    w = gamma
    return 2*w**3/3 - 2*w/3 + \
        (nx - rho + 1)*nx*(2*rho - 1) + \
        0.5*(nx - rho)*(nx - rho + 1)*(2*rho - 1) + \
        (nx - rho)**3/3 - (nx - rho)/3

def forwardrecursion_initial_stage(nx, rho):
    return (nx - rho)*(2*rho - 1) + nx*(2*rho - 1)

def forwardrecursion(nx, nu, nxp, ngi, rho, rhop, gammap):
    return nx*(2*(nu - rho) - 1) + \
        (nx + nu - rho)*(2*rho - 1) + \
        (nu + nx)*(2*rho - 1) + \
        (nu + nx)*(2*ngi - 1) + \
        (nu + nx)*(2*nxp - 1) + \
        nxp*(2*nxp - 1) + \
        (gammap - rhop)*(2*nxp - 1)

def get_rough_flop_count(K, nx, nu, ng, **kwargs):
    implicit = kwargs.get('implicit', False)
    reformulated = kwargs.get('reformulated', False)
    add_constant_offset = kwargs.get('add_constant_offset', False)
    if implicit and reformulated:
        raise ValueError("Cannot use both implicit and reformulated options together.")
    
    ngi = kwargs.get('ngi', 0.5*ng)

    if not isinstance(nu, int) and not isinstance(nu, sp.core.symbol.Symbol):
        nu = nu.copy()
        nx = nx.copy()
        ng = ng.copy()

    if reformulated:
        ng += nx
        nu += nx

    flops = backwardrecursion(nx, nu, nx, ng, ngi, ng, ng) + \
            forwardrecursion(nx, nu, nx, ngi, ng, ng, ng)
    if implicit:
        flops  += preprocessing(nx, nu, nx) + \
                  postprocessing(nx, nu, nx)

    return K*flops + backwardrecursion_initial_stage(nx, nu, ng, ng) + \
              forwardrecursion_initial_stage(nx, ng) + 500000*add_constant_offset

def simplify_flops(expr, vars_to_keep=['nx', 'nu'], other_vars=None):
    # Define symbols to keep
    keep_syms = sp.symbols(' '.join(vars_to_keep))
    
    # Identify all symbols in the expression
    all_syms = expr.free_symbols
    
    # Determine symbols to treat as constant
    if other_vars is None:
        other_syms = all_syms - set(keep_syms)
    else:
        other_syms = sp.symbols(' '.join(other_vars))
    
    # Introduce a single constant to represent all other symbols
    C = sp.symbols('C')
    # subs_dict = {s: C for s in other_syms}
    subs_dict = {}
    
    # Substitute and collect terms in keep_syms
    simplified_expr = sp.collect(expr.subs(subs_dict), keep_syms)
    
    # Optional: expand for clarity
    simplified_expr = sp.expand(simplified_expr)
    
    return simplified_expr

def print_coeffs(expr, nx, nu):
    # Convert to polynomial in nx and nu only
    poly = sp.Poly(expr, nx, nu, domain='EX')  # domain='EX' keeps symbolic coeffs
    
    print("Monomial coefficients in nx, nu:")
    for monom, coeff in zip(poly.monoms(), poly.coeffs()):
        nx_pow, nu_pow = monom
        print(f"nx^{nx_pow} * nu^{nu_pow} : {coeff}")

def simplify_flop_count():
    nx, nu, ng, ngi, K = sp.symbols('nx nu ng ngi K')

    sym_flops_expl = get_rough_flop_count(K, nx, nu, ng, ngi=ngi, implicit=False, reformulated=False)
    sym_flops_expl = sp.simplify(sym_flops_expl)

    sym_flops_impl = get_rough_flop_count(K, nx, nu, ng, ngi=ngi, implicit=True, reformulated=False)
    sym_flops_impl = sp.simplify(sym_flops_impl)

    sym_flops_reform = get_rough_flop_count(K, nx, nu, ng, ngi=ngi, implicit=False, reformulated=True)
    sym_flops_reform = sp.simplify(sym_flops_reform)

    sym_rel_impl_reform = sp.simplify((sym_flops_impl - sym_flops_reform) / sym_flops_reform * 100)
    # print(sym_rel_impl_reform)
    print(simplify_flops(sym_rel_impl_reform))

    num, den = sp.fraction(sym_rel_impl_reform)
    print("Numerator:")
    print_coeffs(num, nx, nu)
    print("Denominator:")
    print_coeffs(den, nx, nu)

def visualize_3d():
    nx = np.arange(1, 15)
    nu = np.arange(1, 7)
    ng = np.arange(1, 5)

    print(f"nx: {nx}")
    print(f"nu: {nu}")
    print(f"ng: {ng}")

    X, Y, Z = np.meshgrid(nx, nu, ng, indexing='ij')
    print(f"X: {X.flatten()}")
    print(f"Y: {Y.flatten()}")
    print(f"Z: {Z.flatten()}")

    W = get_rough_flop_count(X, Y, Z)
    W_implicit = get_rough_flop_count(X, Y, Z, implicit=True)
    W_implicit_overhead = W_implicit - W
    W_reformulated = get_rough_flop_count(X, Y, Z, reformulated=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ms = 1
    ax.scatter(X.flatten(), Y.flatten(), W.flatten(), c='r', marker='o', s=ms, label='Explicit')
    ax.scatter(X.flatten(), Y.flatten(), W_implicit.flatten(), c='b', marker='^', s=ms, label='Implicit')
    # ax.scatter(X.flatten(), Y.flatten(), W_implicit_overhead.flatten(), c='b', marker='^', s=ms, label='Implicit Overhead')
    ax.scatter(X.flatten(), Y.flatten(), W_reformulated.flatten(), c='g', marker='s', s=ms, label='Reformulated')

    plot_polynomial_surface(ax, X.flatten(), Y.flatten(), W.flatten(), color='r', label='Explicit')
    plot_polynomial_surface(ax, X.flatten(), Y.flatten(), W_implicit.flatten(), color='b', label='Implicit')
    # plot_polynomial_surface(ax, X.flatten(), Y.flatten(), W_implicit_overhead.flatten(), color='b', label='Implicit Overhead')
    plot_polynomial_surface(ax, X.flatten(), Y.flatten(), W_reformulated.flatten(), color='g', label='Reformulated')

    ax.set_xlabel('nx')
    ax.set_ylabel('nu')
    ax.set_zlabel('Flops')

    ax.legend()
    
    plt.show()

def get_average_Z(X, Y, Z):
    # for all pairs [x, y], compute the average z value and return
    # X_unique, Y_unique, Z_avg

    unique_pairs = {}
    for x, y, z in zip(X, Y, Z):
        key = (x, y)
        if key not in unique_pairs:
            unique_pairs[key] = []
        unique_pairs[key].append(z)

    X_unique = []
    Y_unique = []
    Z_avg = []
    for (x, y), zs in unique_pairs.items():
        X_unique.append(x)
        Y_unique.append(y)
        Z_avg.append(np.mean(zs))

    return np.array(X_unique), np.array(Y_unique), np.array(Z_avg)

def get_median_Z(X, Y, Z):
    unique_pairs = {}
    for x, y, z in zip(X, Y, Z):
        key = (x, y)
        if key not in unique_pairs:
            unique_pairs[key] = []
        unique_pairs[key].append(z)

    X_unique = []
    Y_unique = []
    Z_median = []
    for (x, y), zs in unique_pairs.items():
        X_unique.append(x)
        Y_unique.append(y)
        Z_median.append(np.median(zs))

    return np.array(X_unique), np.array(Y_unique), np.array(Z_median)

def fit_polynomial_surface(X, Y, Z):
    from scipy.optimize import curve_fit

    MAX_DEGREE = 2

    if MAX_DEGREE == 3:
        def polynomial_surface(xy, x1, x2, x3, y1, y2, y3, c):
            x, y = xy
            return x1*x + x2*x**2 + x3*x**3 + y1*y + y2*y**2 + y3*y**3 + c
        
        # Fit the polynomial surface
        popt, _ = curve_fit(polynomial_surface, (X, Y), Z)

        print(f"Surface fit: t(nx, nu) = {popt[0]}*nx + {popt[1]}*nx^2 + {popt[2]}*nx^3 + "
            f"{popt[3]}*nu + {popt[4]}*nu^2 + {popt[5]}*nu^3 + {popt[6]}")
        
    elif MAX_DEGREE == 2:
        def polynomial_surface(xy, x1, x2, y1, y2, c):
            x, y = xy
            return x1*x + x2*x**2 + y1*y + y2*y**2 + c
        
        # Fit the polynomial surface
        popt, _ = curve_fit(polynomial_surface, (X, Y), Z)

        print(f"Surface fit: t(nx, nu) = {popt[0]}*nx + {popt[1]}*nx^2 + "
            f"{popt[2]}*nu + {popt[3]}*nu^2 + {popt[4]}")

    # get surface Z values on the grid
    X_grid, Y_grid = np.meshgrid(np.unique(X), np.unique(Y), indexing='ij')
    Z_grid = polynomial_surface((X_grid, Y_grid), *popt)

    return popt, X_grid, Y_grid, Z_grid

def plot_polynomial_surface(ax, X, Y, Z, **kwargs):
    popt, X_grid, Y_grid, Z_grid = fit_polynomial_surface(X, Y, Z)
    kwargs["label"] = None
    if "color" in kwargs.keys():
        ax.plot_surface(X_grid, Y_grid, Z_grid, color=kwargs.get("color", 'k'), label=kwargs.get("label", None))
    else:
        ax.plot_surface(X_grid, Y_grid, Z_grid, label=kwargs.get("label", None))

def visualize_experiments(X, Y, Z, **kwargs):
    if 'ax' not in kwargs.keys():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = kwargs['ax']

    color = kwargs.get('color', 'b')
    show = kwargs.get('show', True)
    show_surface = kwargs.get('show_surface', True)
    ms = kwargs.get('marker_size', 1)

    XX, YY, ZZ = get_median_Z(X, Y, Z)
    ax.scatter(XX, YY, ZZ, c=color, marker='^', s=ms)
    if show_surface:
        plot_polynomial_surface(ax, XX, YY, ZZ, color=color)


    ax.set_xlabel('nx')
    ax.set_ylabel('nu')
    ax.set_zlabel('microseconds')
    ax.set_zlim([0, 250])

    if show:
        plt.tight_layout()
        plt.show()

def visualize_experiments_vs_flop_count(X, Y, Z):

    XX, YY, ZZ = get_median_Z(X, Y, Z)
    flop_counts = get_rough_flop_count(XX, YY, XX/2)

    ms_per_flop = ZZ / flop_counts

    # plot ms_per_flop as a heatplot
    fig = plt.figure()
    plt.scatter(XX, YY, c=np.log(ms_per_flop), cmap='viridis', s=15)
    plt.colorbar(label='microseconds per flop')
    ax = plt.gca()

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ms = 1
    # ax.scatter(XX, YY, ms_per_flop, c='b', marker='^', s=ms)
    # # plot_polynomial_surface(ax, XX, YY, ms_per_flop, color='b')

    # print(ms_per_flop)

    ax.set_xlabel('nx')
    # ax.set_ylabel('nu')
    # ax.set_zlabel('microseconds per flop')

    plt.show()  

def plot_line_with_std(ax, x_vals, xx, time, color, alpha, x_vals_offsets=None, **kwargs):
    show_std = kwargs.get('show_std', True)
    lw = kwargs.get('linewidth', 2.0)
    marker = kwargs.get('marker', None)
    linestyle = kwargs.get('linestyle', '-')
    
    t_vals = 0*x_vals
    t_vals_min_std = 0*t_vals
    t_vals_plus_std = 0*t_vals

    if x_vals_offsets is None:
        x_vals_offsets = 0*x_vals

    MAX = 5000
    for i in range(len(x_vals)):
        mask = (xx == x_vals[i]) & (time < MAX)
        std = np.std(time[mask])
        # t_vals[i] = np.median(time[mask])
        t_vals[i] = np.mean(time[mask])
        t_vals_min_std[i] = t_vals[i] - std
        t_vals_plus_std[i] = t_vals[i] + std

    orig_t_vals = t_vals.copy()
    t_vals += x_vals_offsets
    t_vals_min_std += x_vals_offsets
    t_vals_plus_std += x_vals_offsets
    x_vals_offsets += orig_t_vals

    if show_std:
        ax.fill_between(x_vals, t_vals_min_std, t_vals_plus_std, color=color, alpha=0.1)
    ax.plot(x_vals, t_vals, color=color, linewidth=lw, marker=marker, linestyle=linestyle)

    return max(t_vals_plus_std)

def show_2d_plot(data, **kwargs):
    color = kwargs.get('color', 'b')
    show = kwargs.get('show', True)
    show_details = "timing_details" in kwargs and kwargs['timing_details'] is not None
    independent_var = kwargs.get('independent_var', 'nx')
    if independent_var not in ['K', 'nx', 'nu', 'ng', 'ng_ineq']:
        raise ValueError("independent_var must be in ['K', 'nx', 'nu', 'ng', 'ng_ineq']")

    if 'ax' in kwargs.keys():
        ax = kwargs['ax']
    else:
        plt.figure()
        ax = plt.gca()

    if independent_var == 'K':
        x_vals = np.unique(data['K'])
        xx = data['K']
    elif independent_var == 'nx':
        x_vals = np.unique(data['nx'])
        xx = data['nx']
    elif independent_var == 'nu':
        x_vals = np.unique(data['nu'])
        xx = data['nu']
    elif independent_var == 'ng':
        x_vals = np.unique(data['ng'])
        xx = data['ng']
    elif independent_var == 'ng_ineq':
        x_vals = np.unique(data['ng_ineq'])
        xx = data['ng_ineq']

    max_y_val = -1
    y = plot_line_with_std(ax, x_vals, xx, data['time'], color=color, alpha=0.1)
    max_y_val = max(max_y_val, y)

    if show_details:
        x_vals_offsets = 0*x_vals
        for detail_name, detail in kwargs['timing_details'].items():
            y = plot_line_with_std(ax, x_vals, xx, detail['time'], 
                                   color=detail['color'], alpha=0.1, 
                                   x_vals_offsets=x_vals_offsets, 
                                   show_std=False, linewidth=1.0,
                                   marker=detail.get('marker', None),
                                   linestyle=detail.get('linestyle', '-'))
            max_y_val = max(max_y_val, y)

    ax.set_xlabel(translate_label_names(independent_var))
    # ax.set_ylabel('t_comp [us]')
    ax.set_xlim([np.min(x_vals), np.max(x_vals)])
    curr_y_lim = ax.get_ylim()
    ax.set_ylim([0, max(curr_y_lim[1], 1.1*max_y_val)])

    ylog_vars = ['nx']#, 'nu', 'ng']
    if independent_var in ylog_vars:
        ax.set_yscale('log')
        ax.set_ylim([10, max(curr_y_lim[1], 1.1*max_y_val)])

    if show:
        plt.tight_layout()
        plt.show()

def show_2d_plot_all_cases(data_all_cases, **kwargs):
    if 'ax' in kwargs.keys():
        ax = kwargs['ax']
    else:
        plt.figure()
        ax = plt.gca()

    show = kwargs.get("show", False)
    iv = kwargs.get("independent_var", 'nx')
    implicit_details = kwargs.get("implicit_timings_details", None)

    show_2d_plot(data_all_cases['reformulation'], ax=ax, color='g', show=False, independent_var=iv)
    show_2d_plot(data_all_cases['implicit'], ax=ax, color='b', show=show, independent_var=iv, timing_details=implicit_details)
    show_2d_plot(data_all_cases['explicit'], ax=ax, color='r', show=False, independent_var=iv)

def show_flops_2d_plot(data, **kwargs):
    color = kwargs.get('color', 'b')
    show = kwargs.get('show', True)
    independent_var = kwargs.get('independent_var', 'nx')
    if independent_var not in ['K', 'nx', 'nu', 'ng', 'ng_ineq']:
        raise ValueError("independent_var must be in ['K', 'nx', 'nu', 'ng', 'ng_ineq']")

    if 'ax' in kwargs.keys():
        ax = kwargs['ax']
    else:
        plt.figure()
        ax = plt.gca()

    if independent_var == 'K':
        x_vals = np.unique(data['K'])
        xx = data['K']
    elif independent_var == 'nx':
        x_vals = np.unique(data['nx'])
        xx = data['nx']
    elif independent_var == 'nu':
        x_vals = np.unique(data['nu'])
        xx = data['nu']
    elif independent_var == 'ng':
        x_vals = np.unique(data['ng'])
        xx = data['ng']
    elif independent_var == 'ng_ineq':
        x_vals = np.unique(data['ng_ineq'])
        xx = data['ng_ineq']
    
    T = 0*xx
    t_vals = 0*x_vals
    t_vals_min_std = 0*t_vals
    t_vals_plus_std = 0*t_vals
    t_vals_offset = 0*x_vals
    t_vals_min_std_offset = 0*t_vals
    t_vals_plus_std_offset = 0*t_vals

    T = get_rough_flop_count(data['K'], data['nx'], data['nu'], data['ng'], 
                             implicit=kwargs.get("implicit", False), 
                             reformulated=kwargs.get("reformulated", False))
    T_offset = get_rough_flop_count(data['K'], data['nx'], data['nu'], data['ng'], 
                             implicit=kwargs.get("implicit", False), 
                             reformulated=kwargs.get("reformulated", False), 
                             add_constant_offset=True)

    for i in range(len(x_vals)):
        mask = (xx == x_vals[i])
        std = np.std(T[mask])
        # t_vals[i] = np.median(T[mask])
        t_vals[i] = np.mean(T[mask])
        t_vals_min_std[i] = t_vals[i] - std
        t_vals_plus_std[i] = t_vals[i] + std

        t_vals_offset[i] = np.mean(T_offset[mask])
        t_vals_min_std_offset[i] = t_vals_offset[i] - std
        t_vals_plus_std_offset[i] = t_vals_offset[i] + std

    ax.fill_between(x_vals, t_vals_min_std, t_vals_plus_std, color=color, alpha=0.1)
    # ax.fill_between(x_vals, t_vals_min_std_offset, t_vals_plus_std_offset, color=color, alpha=0.1)
    ax.plot(x_vals, t_vals, color=color)
    # ax.plot(x_vals, t_vals_offset, color=color, linestyle=':')

    ax.set_xlabel(translate_label_names(independent_var))
    # ax.set_ylabel('# flops')
    ax.set_xlim([np.min(x_vals), np.max(x_vals)])
    curr_y_lim = ax.get_ylim()
    ax.set_ylim([0, max(curr_y_lim[1], 1.1*np.max(t_vals_plus_std))])

    ylog_vars = ['nx']#, 'nu', 'ng']
    if independent_var in ylog_vars:
        ax.set_yscale('log')
        ax.set_ylim([1e5, max(curr_y_lim[1], 1.1*np.max(t_vals_plus_std))])

    if show:
        plt.tight_layout()
        plt.show()
    
def show_flops_2d_plot_all_cases(data_all_cases, **kwargs):
    if 'ax' in kwargs.keys():
        ax = kwargs['ax']
    else:
        plt.figure()
        ax = plt.gca()

    show = kwargs.get("show", False)
    iv = kwargs.get("independent_var", 'nx')

    show_flops_2d_plot(data_all_cases['explicit'], ax=ax, color='r', show=False, independent_var=iv)
    show_flops_2d_plot(data_all_cases['reformulation'], ax=ax, color='g', show=False, reformulated=True, independent_var=iv)
    show_flops_2d_plot(data_all_cases['implicit'], ax=ax, color='b', show=show, implicit=True, independent_var=iv)

def show_speedup_2d_plot(data, data_base, **kwargs):
    color = kwargs.get('color', 'b')
    show = kwargs.get('show', True)
    independent_var = kwargs.get('independent_var', 'nx')
    show_expected_speedup = kwargs.get('show_expected_speedup', False)
    implicit = kwargs.get('implicit', True)
    if independent_var not in ['K', 'nx', 'nu', 'ng', 'ng_ineq']:
        raise ValueError("independent_var must be in ['K', 'nx', 'nu', 'ng', 'ng_ineq']")

    if 'ax' in kwargs.keys():
        ax = kwargs['ax']
    else:
        plt.figure()
        ax = plt.gca()

    if independent_var == 'K':
        x_vals = np.unique(data['K'])
        xx = data['K']
        xx_base = data_base['K']
    elif independent_var == 'nx':
        x_vals = np.unique(data['nx'])
        xx = data['nx']
        xx_base = data_base['nx']
    elif independent_var == 'nu':
        x_vals = np.unique(data['nu'])
        xx = data['nu']
        xx_base = data_base['nu']
    elif independent_var == 'ng':
        x_vals = np.unique(data['ng'])
        xx = data['ng']
        xx_base = data_base['ng']
    elif independent_var == 'ng_ineq':
        x_vals = np.unique(data['ng_ineq'])
        xx = data['ng_ineq']
        xx_base = data_base['ng_ineq']

    t_vals = 0*x_vals
    ax.plot([x_vals[0], x_vals[-1]], [0, 0], color='k', linestyle='-', linewidth=0.5)
    if show_expected_speedup:
        t_vals_expected = 0*x_vals
        t_vals_expected_offset = 0*x_vals
        flops = get_rough_flop_count(data['K'], data['nx'], data['nu'], 
            data['ng'], ngi=data['ng_ineq'], implicit=implicit, reformulated=False)
        flops_base = get_rough_flop_count(data_base['K'], data_base['nx'],
            data_base['nu'], data_base['ng'], ngi=data_base['ng_ineq'],
            implicit=False, reformulated=True)
        flops_offset = get_rough_flop_count(data['K'], data['nx'], data['nu'], 
            data['ng'], ngi=data['ng_ineq'], implicit=implicit, 
            reformulated=False, add_constant_offset=True)
        flops_base_offset = get_rough_flop_count(data_base['K'], 
            data_base['nx'], data_base['nu'], data_base['ng'], 
            ngi=data_base['ng_ineq'], implicit=False, reformulated=True,
            add_constant_offset=True)
            
    MAX = 5000
    for i in range(len(x_vals)):
        mask = (xx == x_vals[i])
        mask_base = (xx_base == x_vals[i])
        # t_vals[i] = (np.median(data['time'][mask]) - np.median(data_base['time'][mask_base])) / np.median(data_base['time'][mask_base])*100
        t_vals[i] = (np.mean(data['time'][mask]) - np.mean(data_base['time'][mask_base])) / np.mean(data_base['time'][mask_base])*100

        if show_expected_speedup:
            # t_vals_expected[i] = (np.median(flops[mask]) - np.median(flops_base[mask_base])) / np.median(flops_base[mask_base]) * 100
            t_vals_expected[i] = (np.mean(flops[mask]) - np.mean(flops_base[mask_base])) / np.mean(flops_base[mask_base]) * 100
            # t_vals_expected_offset[i] = (np.mean(flops_offset[mask]) - np.mean(flops_base_offset[mask_base])) / np.mean(flops_base_offset[mask_base]) * 100

    lw = 2
    if show_expected_speedup:
        ax.plot(x_vals, t_vals_expected, '-', color=color, linewidth=lw, alpha=0.2)
        # ax.plot(x_vals, t_vals_expected_offset, ':', color=color, linewidth=lw, alpha=0.2)
    ax.plot(x_vals, t_vals, '-', color=color, linewidth=lw)

    # find nx closest to zero and highlight this crossover value
    if independent_var == 'nx' and not show_expected_speedup:
        idx = np.where(t_vals < 0)[0][0]
        last_pos_val = t_vals[idx-1]
        first_neg_val = t_vals[idx]
        ax.hlines(0, xmin=min(x_vals), xmax=max(x_vals), color='k', linestyle='-', linewidth=0.5)
        x_val = x_vals[idx-1] + (last_pos_val/(last_pos_val - first_neg_val))*(x_vals[idx] - x_vals[idx-1])
        ax.vlines(x_val, -100, 100, color='k', linestyle='-', linewidth=0.5)
        
        # add x_val to xticks
        xticks = ax.get_xticks().tolist()
        if x_val not in xticks:
            xticks.append(x_val)
            xticks.sort()
            ax.set_xticks(xticks)

    ax.set_xlabel(translate_label_names(independent_var))
    # ax.set_ylabel('speedup')
    ax.set_xlim([np.min(x_vals), np.max(x_vals)])

    ax.set_ylim([-90, 30])

    if show:
        plt.tight_layout()
        plt.show()

def show_speedup_2d_plot_all_cases(data_all_cases, **kwargs):
    if 'ax' in kwargs.keys():
        ax = kwargs['ax']
    else:
        plt.figure()
        ax = plt.gca()

    show = kwargs.get("show", False)
    iv = kwargs.get("independent_var", 'nx')

    show_speedup_2d_plot(data_all_cases['implicit'], data_all_cases['reformulation'],
                         ax=ax, color='b', show=show, implicit=True, independent_var=iv,
                         show_expected_speedup=kwargs.get('show_expected_speedup', False))

    show_speedup_2d_plot(data_all_cases['explicit'], data_all_cases['reformulation'],
                         ax=ax, color='r', show=show, implicit=False, independent_var=iv,
                         show_expected_speedup=kwargs.get('show_expected_speedup', False))

def show_speedup_heatmap(ax, data, data_base, var1, var2):
    x_vals = np.unique(data[var1])
    y_vals = np.unique(data[var2])

    xx, yy = np.meshgrid(x_vals, y_vals, indexing='ij')
    speedup = np.zeros(xx.shape)

    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            mask = (data[var1] == x_vals[i]) & (data[var2] == y_vals[j])
            mask_base = (data_base[var1] == x_vals[i]) & (data_base[var2] == y_vals[j])
            speedup[i, j] = (np.mean(data['time'][mask]) - np.mean(data_base['time'][mask_base])) / np.mean(data_base['time'][mask_base]) * 100

    # plot heatmap
    norm = mcolors.TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
    c = ax.pcolormesh(xx, yy, speedup, shading='auto', cmap='seismic', norm=norm)
    ax.set_xlabel(translate_label_names(var1))
    ax.set_ylabel(translate_label_names(var2))
    
    return c

def show_speedup_heatmap_all_plots(data, data_base, vars):
    fig, axs = plt.subplots(len(vars)-1, len(vars)-1, figsize=(9, 9))
    for i, var1 in enumerate(vars):
        for j, var2 in enumerate(vars):
            if i <= j:
                continue
            print(f"{var1} ({i}) vs {var2} ({j})")
            c = show_speedup_heatmap(axs[i-1, j], data, data_base, var1, var2)
            # axs[i, j].set_title(f"{translate_label_names(var1)} vs {translate_label_names(var2)}")
    
    for i in range(len(vars)-1):
        for j in range(i+1, len(vars)-1):
            axs[i,j].set_visible(False)

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.85, 0.3, 0.03, 0.6])  # [left, bottom, width, height]
    cbar = fig.colorbar(c, cax=cbar_ax, label=r"Relative reduction in $t_\mathrm{comp}$ [\%]")

    plt.tight_layout()

def add_legend_below(fig, timing_details=None):
    # create some space below the axes
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25)

    # get basic labels
    labels = ['Explicit', 'Implicit', 'Reformulated']
    colors = ['r', 'b', 'g']
    handles = [plt.Line2D([0], [0], color=color, lw=2) for color in colors]


    # add detailed handles if provided
    if timing_details is not None:
        # add empty entry
        labels.append('')
        handles.append(plt.Line2D([0], [0], color='none', lw=0))

        # add new entries
        for detail_name, detail in timing_details.items():
            labels.append(detail_name)
            handles.append(plt.Line2D([0], [0], color=detail['color'], lw=2, linestyle=detail.get('linestyle', '-'), marker=detail.get('marker', None)))

    # permute labels such that it is row-major instead of column-major
    nb_cols = 4
    nb_rows = int(np.ceil(len(labels) / nb_cols))
    perm = [(i % nb_rows)*nb_cols + i // nb_rows for i in range(len(labels))]
    handles = [handles[i] for i in perm]
    labels = [labels[i] for i in perm]    

    # add legend below the axes
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0), frameon=False)

if __name__ == "__main__":
    timing_details_implicit = {
        #"copy_rhs" : {"time": time_copying_rhs_implicit, "color": 'k'},
        "solve" : {
            "time": time_solve_implicit, 
            "color": 'k', 
            'linestyle' : '-'},
        "preprocess_jac" : {
            "time": time_preprocess_jac_implicit, 
            "color": 'k', 
            'linestyle' : '--'},
        "preprocess_hess" : {
            "time": time_preprocess_hess_implicit, 
            "color": 'k', 
            'linestyle' : '-.'},
        "postprocess" : {
            "time": time_postprocess_implicit, 
            "color": 'k', 
            'linestyle' : ':'},
    }

    data_explicit = {'K': K_explicit, 'nx': nx_explicit, 'nu': nu_explicit, 'ng': ng_explicit, 'ng_ineq': ng_ineq_explicit, 'time': time_explicit}
    data_reformulation = {'K': K_reformulation, 'nx': nx_reformulation, 'nu': nu_reformulation, 'ng': ng_reformulation, 'ng_ineq': ng_ineq_reformulation, 'time': time_reformulation}
    data_implicit = {'K': K_implicit, 'nx': nx_implicit, 'nu': nu_implicit, 'ng': ng_implicit, 'ng_ineq': ng_ineq_implicit, 'time': time_implicit}
    data_all_cases = {'explicit': data_explicit,
                      'reformulation': data_reformulation, 
                      'implicit': data_implicit}
    
    # simplify_flop_count()
    # exit()

    VISUALIZE_FLOPS = True
    VISUALIZE_EXPERIMENTS = True
    VISUALIZE_SPEEDUP = True
    VISUALIZE_EXPECTED_SPEEDUP = True
    VISUALIZE_2D_SPEEDUP = True

    ### visualize flop counts
    if VISUALIZE_FLOPS:
        print(f"Visualizing flop counts")
        fig, axs = plt.subplots(1, 5, figsize=(13, 4))
        fig.suptitle(r"\# FLOPs")
        show_flops_2d_plot_all_cases(data_all_cases, ax=axs[0], independent_var='K')
        show_flops_2d_plot_all_cases(data_all_cases, ax=axs[1], independent_var='nx')
        show_flops_2d_plot_all_cases(data_all_cases, ax=axs[2], independent_var='nu')
        show_flops_2d_plot_all_cases(data_all_cases, ax=axs[3], independent_var='ng')
        show_flops_2d_plot_all_cases(data_all_cases, ax=axs[4], independent_var='ng_ineq')
        add_legend_below(fig)
        # plt.savefig("unittest/ocp/figures/flop_count_scaling.png", dpi=300)
        plt.savefig("unittest/ocp/figures/flop_count_scaling.svg")
    
    ### simplified 2d figure of experimental results
    if VISUALIZE_EXPERIMENTS:
        print(f"Visualizing experiments")
        fig, axs = plt.subplots(1, 5, figsize=(13, 4))
        fig.suptitle(r"$t_\mathrm{comp}$ [$\mu s$]")
        show_2d_plot_all_cases(data_all_cases, ax=axs[0], independent_var='K', implicit_timings_details=timing_details_implicit)
        show_2d_plot_all_cases(data_all_cases, ax=axs[1], independent_var='nx', implicit_timings_details=timing_details_implicit)
        show_2d_plot_all_cases(data_all_cases, ax=axs[2], independent_var='nu', implicit_timings_details=timing_details_implicit)
        show_2d_plot_all_cases(data_all_cases, ax=axs[3], independent_var='ng', implicit_timings_details=timing_details_implicit)
        show_2d_plot_all_cases(data_all_cases, ax=axs[4], independent_var='ng_ineq', implicit_timings_details=timing_details_implicit)
        add_legend_below(fig, timing_details=timing_details_implicit)
        # plt.savefig("unittest/ocp/figures/t_comp_scaling.png", dpi=300)
        plt.savefig("unittest/ocp/figures/t_comp_count_scaling.svg")
    
    ### visualize speedup
    if VISUALIZE_SPEEDUP:
        print(f"Visualizing speedup")
        fig, axs = plt.subplots(1, 5, figsize=(13, 4))
        fig.suptitle(r"Relative reduction in $t_\mathrm{comp}$ [\%]")
        show_speedup_2d_plot_all_cases(data_all_cases, ax=axs[0], independent_var='K')
        show_speedup_2d_plot_all_cases(data_all_cases, ax=axs[1], independent_var='nx')
        show_speedup_2d_plot_all_cases(data_all_cases, ax=axs[2], independent_var='nu')
        show_speedup_2d_plot_all_cases(data_all_cases, ax=axs[3], independent_var='ng')
        show_speedup_2d_plot_all_cases(data_all_cases, ax=axs[4], independent_var='ng_ineq')
        # add_legend_below(fig)
        plt.tight_layout()
        # plt.savefig("unittest/ocp/figures/speedup_scaling.png", dpi=300)
        plt.savefig("unittest/ocp/figures/speedup_scaling.svg")

    ### visualize expected speedup
    if VISUALIZE_EXPECTED_SPEEDUP:
        print(f"Visualizing expected speedup")
        fig, axs = plt.subplots(1, 5, figsize=(13, 4))
        fig.suptitle(r"Relative reduction in computation time [\%]")
        show_speedup_2d_plot_all_cases(data_all_cases, show_expected_speedup=True, ax=axs[0], independent_var='K')
        show_speedup_2d_plot_all_cases(data_all_cases, show_expected_speedup=True, ax=axs[1], independent_var='nx')
        show_speedup_2d_plot_all_cases(data_all_cases, show_expected_speedup=True, ax=axs[2], independent_var='nu')
        show_speedup_2d_plot_all_cases(data_all_cases, show_expected_speedup=True, ax=axs[3], independent_var='ng')
        show_speedup_2d_plot_all_cases(data_all_cases, show_expected_speedup=True, ax=axs[4], independent_var='ng_ineq')
        # add_legend_below(fig)
        plt.tight_layout()
        # plt.savefig("unittest/ocp/figures/expected_speedup_scaling.png", dpi=300)
        plt.savefig("unittest/ocp/figures/expected_speedup_scaling.svg")

    if VISUALIZE_2D_SPEEDUP:
        print(f"Visualizing 2d speedup")
        show_speedup_heatmap_all_plots(data_implicit, data_reformulation, ['nx', 'nu', 'K', 'ng', 'ng_ineq'])
        # plt.savefig("unittest/ocp/figures/speedup_heatmap.png", dpi=300)
        plt.savefig("unittest/ocp/figures/speedup_heatmap.svg")

    plt.show()

    ### see how well flop count behaviour matches computation time
    # visualize_experiments_vs_flop_count(X_explicit, Y_explicit, Z_explicit)
    # visualize_experiments_vs_flop_count(X_reformulation, Y_reformulation, Z_reformulation)
    # visualize_experiments_vs_flop_count(X_implicit, Y_implicit, Z_implicit)

