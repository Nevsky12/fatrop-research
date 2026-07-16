import numpy as np
import matplotlib.pyplot as plt

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

def get_KKT(K, nx, nu, ng, **kwargs):
    # check input
    assert K > 0
    assert len(nx) == K + 1, "Length of nx must match K"
    assert len(nu) == K, "Length of nu must match K"
    assert len(ng) == K + 1, "Length of ng must match K"
    assert all(n >= 0 for n in nx), "All nx must be non-negative"
    assert all(n >= 0 for n in nu), "All nu must be non-negative"
    assert all(n >= 0 for n in ng), "All ng must be non-negative"
    nu.append(0)

    implicit = kwargs.get('implicit', False)
    reformulated = kwargs.get('reformulated', False)
    assert not (implicit and reformulated), "Cannot be both implicit and reformulated"

    n = sum(nx) + sum(nu) + sum(ng) + sum(nx[1:])
    n += sum(nx[1:]) + sum(nx[1:])

    if reformulated:
        for j in range(len(nx) - 1):
            nu[j] += nx[j+1]
            ng[j] += nx[j+1]

    KKT_dense = np.zeros((n, n))
    KKT_ones = np.zeros((n, n))
    KKT_extra = np.zeros((n, n))

    p = 0
    for k in range(K, -1, -1):
        KKT_dense[p:p+nx[k]+nu[k], p:p+nx[k]+nu[k]] = 2                 # objective (RSQ)
        KKT_dense[p+nx[k]+nu[k]:p+nx[k]+nu[k]+ng[k], p:p+nx[k]+nu[k]] = 2     # constraints
        KKT_dense[p:p+nx[k]+nu[k], p+nx[k]+nu[k]:p+nx[k]+nu[k]+ng[k]] = 2     # constraints

        if k > 0:
            if implicit:
                KKT_dense[p+nx[k]+nu[k]+ng[k]:p+nx[k]+nu[k]+ng[k]+nx[k], p+nu[k]:p+nu[k]+nx[k]] = 2   # J
                KKT_dense[p+nu[k]:p+nu[k]+nx[k], p+nx[k]+nu[k]+ng[k]:p+nx[k]+nu[k]+ng[k]+nx[k]] = 2   # J
                KKT_extra[p+nx[k]+nu[k]+ng[k]+nx[k]:p+nx[k]+nu[k]+ng[k]+nx[k]+nu[k-1]+nx[k-1], p+nu[k]:p+nu[k]+nx[k]] = 2   # Fu Fx
                KKT_extra[p+nu[k]:p+nu[k]+nx[k], p+nx[k]+nu[k]+ng[k]+nx[k]:p+nx[k]+nu[k]+ng[k]+nx[k]+nu[k-1]+nx[k-1]] = 2   # Fu Fx
            else:
                for j in range(nx[k]):
                    KKT_ones[p+nx[k]+nu[k]+ng[k]+j, p+nu[k]+j] = 1
                    KKT_ones[p+nu[k]+j, p+nx[k]+nu[k]+ng[k]+j] = 1

            p2 = p + nx[k] + nu[k] + ng[k] + nx[k]
            KKT_dense[p2:p2+nx[k-1]+nu[k-1], p2-nx[k]:p2] = 2   # AB
            KKT_dense[p2-nx[k]:p2, p2:p2+nx[k-1]+nu[k-1]] = 2   # AB

        p += nx[k] + nu[k] + ng[k] + nx[k]

    # count elements in KKT_dense
    nnz = np.count_nonzero(KKT_dense) + np.count_nonzero(KKT_extra) + np.count_nonzero(KKT_ones)
    nones = np.count_nonzero(KKT_ones)
    title = "Reformulated" if reformulated else "Implicit" if implicit else "Explicit"
    title += " (nonzeros: " + str(nnz) + ", of which 'ones': " + str(nones) + ")"

    nu.pop(-1)  # remove the last zero added for convenience
    return KKT_dense, KKT_ones, KKT_extra, title

def show_structure(ax, KKT_dense, KKT_ones, KKT_extra, title):
    ms = int(176/KKT_dense.shape[0])
    ax.spy(KKT_dense, markersize=ms, color='black')
    ax.spy(KKT_ones, markersize=ms, color='gray')
    ax.spy(KKT_extra, markersize=ms, color='steelblue')
    ax.set_title(title)

# define original problem dimensions
K = 4
nx = [4, 2, 3, 5, 2]
nu = [2, 1, 4, 2]
ng = [2, 3, 1, 0, 2]

# visualize
plt.figure()
show_structure(plt.gca(), *get_KKT(K, nx, nu, ng))
plt.savefig('unittest/ocp/figures/KKT_structure_explicit.png', dpi=300)

plt.figure()
show_structure(plt.gca(), *get_KKT(K, nx, nu, ng, implicit=True))
plt.savefig('unittest/ocp/figures/KKT_structure_implicit.png', dpi=300)

plt.figure()
show_structure(plt.gca(), *get_KKT(K, nx, nu, ng, reformulated=True))
plt.savefig('unittest/ocp/figures/KKT_structure_reformulated.png', dpi=300)

plt.show()