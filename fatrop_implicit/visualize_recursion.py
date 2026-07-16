import numpy as np
from test_debug_helper import *
import sys
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import SymLogNorm
from matplotlib.colors import LinearSegmentedColormap

sys.path.append(os.path.join(os.path.dirname(__file__), "build_docker"))

import json
settings = json.load(open(os.path.join(os.path.dirname(__file__), "visualize_recursion_settings.json"), 'r'))

EXPLICIT = settings['EXPLICIT']
REFORMULATED = settings['REFORMULATED']
IMPLICIT = settings['IMPLICIT']

assert (EXPLICIT + REFORMULATED + IMPLICIT) == 1, "Exactly one of EXPLICIT, REFORMULATED, IMPLICIT must be True"

if EXPLICIT:
    from factorization_info_expl import *
    from preprocess_info_expl import *
elif REFORMULATED:
    from factorization_info_reform import *
    from preprocess_info_reform import *
elif IMPLICIT:
    from factorization_info_impl import *
    from preprocess_info_impl import *

# preprocessing
_, _, _, _, _, _, _, _, _, _, lss_pre = get_expected_matrices(K, nu, nx, r, ng_eq, ng_ineq, modified_nu, modified_nx, modified_ng_eq, modified_ng_ineq, 
                          RSQrqt_original, GuGx_original, FuFx_original, Gg_eqt_original, Gg_ineqt_original, BAbt_original, D_x,
                          Pl, Pr, L, U, 
                          RSQrqt, GuGx, FuFx, Gg_eqt, Gg_ineqt, BAbt, Jt, store_linear_systems=True)

# recursion
blocks = GetBlockMatrices(K, modified_nu, modified_nx, modified_ng_eq, RSQrqt, GuGx, FuFx, Gg_eqt, BAbt)
solution, lss = Solve(K, modified_nu, modified_nx, modified_ng_eq, blocks['R'], blocks['S'], blocks['Q'], blocks['Gu'], blocks['Gx'], blocks['Fu'], blocks['Fx'], blocks['Hu'], blocks['Hx'], blocks['B'], blocks['A'], blocks['r'], blocks['q'], blocks['h'], blocks['b'], Pl_r, Pr_r, L_r, U_r, Lmbd, rank_k_values, Hut, -1, store_linear_systems=True)


def update(frame_idx, lss_idxs, backward_pass, lss, lss_pre, max_dim, max_value, fig, ax, ax_sol, ax_rhs):
        backward_mode = backward_pass[frame_idx]
        i = lss_idxs[frame_idx]
        print(f"frame_idx: {frame_idx}, backward_mode: {backward_mode}")
        print(f"i = {i}")
        print(len(lss))
            
        if i < 0:
            # show pre-processing
            ls = lss_pre[-(i+1)]
        else:
            # show recursion
            ls = lss[i]

        curr_dim = ls['KKT'].shape[0]
        KKT = np.block([[np.zeros((max_dim-curr_dim, max_dim))],
                        [np.zeros((curr_dim, max_dim-curr_dim)), ls['KKT']]])
        rhs = np.block([[np.zeros((max_dim-curr_dim,1))],
                        [ls['rhs']]])
        try:
            solution = np.block([[np.zeros((max_dim-curr_dim,1))],
                            [np.linalg.solve(ls['KKT'], -ls['rhs']) if backward_mode else np.zeros((curr_dim,1))]])
        except:
            # print(ls['KKT'])
            for ki in range(ls['KKT'].shape[0]):
                row = ""
                for kj in range(ls['KKT'].shape[1]):
                    elem = ls['KKT'][ki, kj]
                    if elem == 0:
                        row += "."
                    else:
                        row += "x"
                print(row)
            removed_rows = []
            for row in range(curr_dim):
                if np.all(ls['KKT'][row,:] == 0):
                    removed_rows.append(row)
            new_KKT = np.delete(ls['KKT'], removed_rows, axis=0)
            new_KKT = np.delete(new_KKT, removed_rows, axis=1)
            new_rhs = np.delete(ls['rhs'], removed_rows, axis=0)
            for ki in range(new_KKT.shape[0]):
                row = ""
                for kj in range(new_KKT.shape[1]):
                    elem = new_KKT[ki, kj]
                    if elem == 0:
                        row += "."
                    else:
                        row += "x"
                print(row)
            new_solution = np.linalg.solve(new_KKT, -new_rhs)
            for removed_rows_idx, row in enumerate(removed_rows):
                new_solution = np.insert(new_solution, row, 0)
            new_solution = np.array([[elem] for elem in new_solution])
            solution = np.block([[np.zeros((max_dim-curr_dim,1))],
                            [new_solution]])
            
        # make rhs wider by adding repeating the column
        rhs = np.hstack([rhs, rhs])
        solution = np.hstack([solution, solution])

        if ax is not None:
            ax.clear(); 
        if ax_sol is not None:
            ax_sol.clear(); 
        if ax_rhs is not None:
            ax_rhs.clear()
        norm = SymLogNorm(linthresh=1e-3, vmin=-max_value, vmax=max_value)
        cmap = 'bwr'
        cmap = LinearSegmentedColormap.from_list('custom', ['maroon', 'white', 'royalblue'])
        
        if ax is not None:
            im = ax.imshow(KKT, cmap=cmap, norm=norm)
            ax.set_xticks([])
            ax.set_yticks([])

        if ax_sol is not None:
            im_sol = ax_sol.imshow(solution, cmap=cmap, norm=norm)
            ax_sol.set_xticks([])
            ax_sol.set_yticks([])

        if ax_rhs is not None:
            im_rhs = ax_rhs.imshow(rhs, cmap=cmap, norm=norm)
            ax_rhs.set_xticks([])
            ax_rhs.set_yticks([])

        if i < 0:
            fig.suptitle(f"Preprocessing" if not backward_mode else f"Postprocessing")
        else:
            fig.suptitle(f"Backward recursion" if not backward_mode else f"Forward recursion")

        return_tuple = []
        if ax is not None:
            return_tuple.append(im)
        if ax_sol is not None:
            return_tuple.append(im_sol)
        if ax_rhs is not None:
            return_tuple.append(im_rhs)
        return tuple(return_tuple)


def visualize_recursion(lss_pre, lss, K, nx, nu, ng_eq):
    # find the largest dimension and largest (absolute) value of the KKT matrix
    max_dim = 0
    max_value = 0
    for ls in lss:
        dim = ls['KKT'].shape[0]
        if dim > max_dim:
            max_dim = dim
        max_value = max(max_value, np.max(np.abs(ls['KKT'])))

    print(f"K = {K}")
    print(f"nx = {nx}")
    print(f"nu = {nu}")
    print(f"ng_eq = {ng_eq}")

    print(f"max dim found: {max_dim}")
    max_dim = sum(nx) + sum(nu) + sum(ng_eq) + sum(nx[1:])
    if not REFORMULATED:
        max_dim += 2*sum(nx[1:])
    print(f"max_dim computed: {max_dim}")
    
    max_value = 15

    # make animation of heatmap values
    fig, (ax, ax_sol, ax_rhs) = plt.subplots(1, 3, figsize=(10, 8), 
                           gridspec_kw={'width_ratios': [max_dim, 2, 2]})

    ### CREATE ANIMATION ###
    preprocess_range = range(-1, -len(lss_pre)-1, -1) 
    pause_range = [len(lss)-1 for _ in range(7)]
    end_pause_range = [0 for _ in range(7)]
    start_pause_range = [-1 for _ in range(10)]
    recursion_range = range(len(lss)-1, -1, -1)
    lss_idxs = start_pause_range + list(preprocess_range) + pause_range + list(recursion_range) + end_pause_range
    backward_pass = [False for _ in lss_idxs] + [True for _ in lss_idxs]

    lss_idxs_flipped = lss_idxs[::-1]
    lss_idxs = lss_idxs + lss_idxs_flipped
    ani = FuncAnimation(fig, update, frames=range(len(lss_idxs)), blit=True, repeat=False, fargs=(lss_idxs, backward_pass, lss, lss_pre, max_dim, max_value, fig, ax, ax_sol, ax_rhs))
    # ani.save("recursion_visualization.gif", writer='imagemagick', fps=15)

    # save as mp4
    appendix = "explicit" if EXPLICIT else "reformulated" if REFORMULATED else "implicit"
    # ani.save(f"recursion_visualization_{appendix}.mp4", writer='ffmpeg', fps=5, dpi=200)


    ### CREATE PNG SNAPSHOTS ###
    frames_to_save = [0]
    p = len(start_pause_range) + len(preprocess_range)
    frames_to_save.append(p-1)
    frames_to_save.append(p)
    p += len(pause_range) + len(recursion_range) 
    frames_to_save.append(p)
    p += 2*len(end_pause_range) 
    frames_to_save.append(p)
    p += len(recursion_range)
    frames_to_save.append(p)
    p += len(pause_range)
    frames_to_save.append(p)
    p += len(preprocess_range)
    frames_to_save.append(p-1)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for frame_idx in frames_to_save:
        update(frame_idx, lss_idxs, backward_pass, lss, lss_pre, max_dim, max_value, fig, ax, None, None)
        # remove the title
        plt.suptitle("")
        plt.tight_layout()
        # plt.savefig(f"recursion_visualization_frame_{appendix}_{frame_idx}.png", dpi=300)

    ### VISUALZIE HIGHLIGHTED LINEAR SYSTEMS ###
    if settings["create_highlighted_visualizations"]:
        val = 1.0e-3
        if settings["highlighted_options"]["show_only_dynamics"]:
            ax.clear()
            blocks = GetBlockMatrices(K, nu, nx, ng_eq, RSQrqt_original, GuGx_original, FuFx_original, Gg_eqt_original, BAbt_original)
            R = [np.where(r, val, 0) for r in blocks['R']]
            S = [np.where(s, val, 0) for s in blocks['S']]
            Q = [np.where(q, val, 0) for q in blocks['Q']]
            Gu = [np.where(gu, val, 0) for gu in blocks['Gu']]
            Gx = [np.where(gx, val, 0) for gx in blocks['Gx']]
            Fu = [np.where(fu, val, 0) for fu in blocks['Fu']]
            Fx = [np.where(fx, val, 0) for fx in blocks['Fx']]
            if REFORMULATED:
                Hu = [hu.copy() for hu in blocks['Hu']]
                Hx = [hx.copy() for hx in blocks['Hx']]
                # Hu = blocks['Hu']
                # Hx = blocks['Hx']

                for k in range(len(Hu)):
                    temp = nu[k]-(nx[k+1] if k < K-1 else 0)
                    temp2 = ng_eq[k]-(nx[k+1] if k < K-1 else 0)
                    Hu[k][:temp2, temp:] = val
                    Hx[k][:temp2, :] = val
            else:
                Hu = [np.where(hu, val, 0) for hu in blocks['Hu']]
                Hx = [np.where(hx, val, 0) for hx in blocks['Hx']]
            B = [b for b in blocks['B']]
            A = [a for a in blocks['A']]
            r = [val + 0*ri for ri in blocks['r']]
            q = [val + 0*qi for qi in blocks['q']]
            h = [val + 0*hi for hi in blocks['h']]
            b = [val + 0*bi for bi in blocks['b']]
            # print(blocks.keys())
            # Jt = [jt for jt in blocks['Jt']]
            KKT, rhs = GetKKT(K, nu, nx, ng_eq, R, S, Q, Gu, Gx, Fu, Fx, Hu, Hx, B, A, r, q, h, b, Jt)

            lss = [{"KKT": KKT, "rhs": rhs}]
            update(0, [0], [False], lss, lss_pre, max_dim, max_value, fig, ax, None, None)
            plt.suptitle("")
            plt.tight_layout()
            plt.savefig(f"recursion_visualization_highlighted_dynamics_{appendix}.png", dpi=300)
        
        if settings['highlighted_options']["show_LU_decomposition"]:
            ax.clear()
            blocks = GetBlockMatrices(K, nu, nx, ng_eq, RSQrqt_original, GuGx_original, FuFx_original, Gg_eqt_original, BAbt_original)
            R = [np.where(r, val, 0) for r in blocks['R']]
            S = [np.where(s, val, 0) for s in blocks['S']]
            Q = [np.where(q, val, 0) for q in blocks['Q']]
            Gu = [np.where(gu, val, 0) for gu in blocks['Gu']]
            Gx = [np.where(gx, val, 0) for gx in blocks['Gx']]
            Fu = [np.where(fu, val, 0) for fu in blocks['Fu']]
            Fx = [np.where(fx, val, 0) for fx in blocks['Fx']]
            Hx = [np.where(hx, val, 0) for hx in blocks['Hx']]
            B = [np.where(b, val, 0) for b in blocks['B']]
            A = [np.where(a, val, 0) for a in blocks['A']]
            r = [np.where(ri, val, 0) for ri in blocks['r']]
            q = [np.where(qi, val, 0) for qi in blocks['q']]
            h = [np.where(hi, val, 0) for hi in blocks['h']]
            b = [np.where(bi, val, 0) for bi in blocks['b']]
            Hu = [hu.copy() for hu in blocks['Hu']]
            if not IMPLICIT:
                Jt_copy = [np.where(Jt_i, val, 0) for Jt_i in Jt]
            else:
                Jt_copy = [jt for jt in Jt]
            KKT, rhs = GetKKT(K, nu, nx, ng_eq, R, S, Q, Gu, Gx, Fu, Fx, Hu, Hx, B, A, r, q, h, b, Jt_copy)

            lss = [{"KKT": KKT, "rhs": rhs}]
            update(0, [0], [False], lss, lss_pre, max_dim, max_value, fig, ax, None, None)
            plt.suptitle("")
            plt.tight_layout()
            plt.savefig(f"recursion_visualization_highlighted_LU_{appendix}.png", dpi=300)

        if settings['show_case_no_states_no_constraints']:
            ax.clear()
            K = 3
            nx = [0 for _ in range(K+1)]
            ng_eq = [0 for _ in range(K)]
            Jt_copy = [np.zeros((0,0)) for _ in range(K+1)]

            blocks = GetBlockMatrices(K, nu, nx, ng_eq, RSQrqt_original, GuGx_original, FuFx_original, Gg_eqt_original, BAbt_original)

            KKT, rhs = GetKKT(K, nu, nx, ng_eq, blocks['R'], blocks['S'], blocks['Q'], blocks['Gu'], blocks['Gx'], blocks['Fu'], blocks['Fx'], blocks['Hu'], blocks['Hx'], blocks['B'], blocks['A'], blocks['r'], blocks['q'], blocks['h'], blocks['b'], Jt=Jt_copy)

            lss = [{"KKT": KKT, "rhs": rhs}]
            update(0, [0], [False], lss, lss_pre, max_dim, max_value, fig, ax, None, None)
            plt.suptitle("")
            plt.tight_layout()
            plt.savefig(f"recursion_visualization_no_states_no_constraints_{appendix}.png", dpi=300)


visualize_recursion(lss_pre, lss, K, nx, nu, ng_eq)