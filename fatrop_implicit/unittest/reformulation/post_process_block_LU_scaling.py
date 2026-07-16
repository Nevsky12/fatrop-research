import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def latexify():
    params = {#'backend': 'ps',
              'axes.labelsize': 14,
              'axes.titlesize': 15,
              'legend.fontsize': 10,
              'xtick.labelsize': 14,
              'ytick.labelsize': 14,
              'text.usetex': True,
              'font.family': 'serif',
              'figure.figsize': [4, 3],
              'text.latex.preamble': r'\usepackage{bm}',
              }
 
    plt.rcParams.update(params)

latexify()

class FlopCounter:
    def __init__(self):
        self.scaling_factor = 0.01

    def add_flops_to_data(self, df):
        df = self.add_full_flops_to_data(df)
        df = self.add_blocked_flops_to_data(df)
        return df
    
    def get_lu_flops(self, r, m, n):
        return 2*r*(m-1)*(n-1) + r*(m-1) + r*(r-1)*(2*r-1)/3 - r*(r-1)*(2*m+2*n-3)/2

    def add_full_flops_to_data(self, df):
        r = df['rank'] + np.minimum(df['nu'], df['ng']) # assume full rank of bottom left block
        m = df['ng'] + df['nx']
        n = df['nu'] + df['nx']

        df['flops_full'] = self.scaling_factor*self.get_lu_flops(r, m, n)
        return df
    
    def add_blocked_flops_to_data(self, df):
        # r1 = df['nx'] # assume full rank
        r1 = df['rank']
        r2 = np.minimum(df['nu'], df['ng']) # assume full rank

        m1 = df['nx']
        n1 = df['nx']
        m2 = df['ng']
        n2 = df['nu']

        min_m1_n1 = np.minimum(m1, n1)

        df['flops_blocked'] = self.scaling_factor*(
            self.get_lu_flops(r1, m1, n1) +
            self.get_lu_flops(r2, m2, n2) + 
            m2 * min_m1_n1**2
        )
        return df
        

def plot_scaling(df, **kwargs):
    # metrics = ['nu', 'nx', 'ng', 'rank']
    metrics = ['rank']

    for metric in metrics:
        plt.figure()

        if metric == 'rank':
            step = 0.1
            metric_unique_sorted = np.arange(0, 1, step)
            times_full = [df[(df['rank']/df['nx'] > val - step) & (df['rank']/df['nx'] <= val)]['time_full'].mean() for val in metric_unique_sorted[1:]]
            times_blocked = [df[(df['rank']/df['nx'] > val - step) & (df['rank']/df['nx'] <= val)]['time_blocked'].mean() for val in metric_unique_sorted[1:]]
            metric_unique_sorted = metric_unique_sorted[1:] - step/2
        else:
            metric_unique_sorted = np.sort(df[metric].unique())
            times_full = [df[df[metric] == val]['time_full'].mean() for val in metric_unique_sorted]
            times_blocked = [df[df[metric] == val]['time_blocked'].mean() for val in metric_unique_sorted]

        if kwargs.get('show_relative', False):
            plt.plot(metric_unique_sorted, (np.array(times_blocked) - np.array(times_full)) / np.array(times_full), color='blue')

            if kwargs.get('show_flops', False):
                flops_full = [df[df[metric] == val]['flops_full'].mean() for val in metric_unique_sorted]
                flops_blocked = [df[df[metric] == val]['flops_blocked'].mean() for val in metric_unique_sorted]
                plt.plot(metric_unique_sorted, (np.array(flops_blocked) - np.array(flops_full)) / np.array(flops_full), linestyle='dashed', color='blue')
        else:
            plt.plot(metric_unique_sorted, times_full, label='Full LU', color='green')
            plt.plot(metric_unique_sorted, times_blocked, label='Blocked LU', color='blue')

            if kwargs.get('show_flops', False):
                flops_full = [df[df[metric] == val]['flops_full'].mean() for val in metric_unique_sorted]
                flops_blocked = [df[df[metric] == val]['flops_blocked'].mean() for val in metric_unique_sorted]
                plt.plot(metric_unique_sorted, flops_full, label='Full LU Flops', linestyle='dashed', color='green')
                plt.plot(metric_unique_sorted, flops_blocked, label='Blocked LU Flops', linestyle='dashed', color='blue')

            if kwargs.get('show_std', False):
                times_full_std = [df[df[metric] == val]['time_full'].std() for val in metric_unique_sorted]
                times_blocked_std = [df[df[metric] == val]['time_blocked'].std() for val in metric_unique_sorted]
                plt.fill_between(metric_unique_sorted, np.array(times_full) - np.array(times_full_std), np.array(times_full) + np.array(times_full_std), color='green', alpha=0.2)
                plt.fill_between(metric_unique_sorted, np.array(times_blocked) - np.array(times_blocked_std), np.array(times_blocked) + np.array(times_blocked_std), color='blue', alpha=0.2)

        if metric == 'rank':
            plt.xlabel('rank of M1 / nx')        
        else:
            plt.xlabel(metric)
        if kwargs.get('show_relative', False):
            plt.ylabel('Relative time difference')
        else:
            plt.ylabel('Time (us)')
        plt.legend()
        plt.axhline(0, color='gray', linestyle='-')
        plt.xlim(min(metric_unique_sorted), max(metric_unique_sorted))
        # plt.show()
        plt.tight_layout()
        plt.savefig("unittest/reformulation/figures/scaling_" +
                    ("relative_" if kwargs.get("show_relative", False) else "") + metric + ".png", dpi=300)

def create_2d_plot(metric_x, metric_y, df, **kwargs):
    metric_x_unique_sorted = np.sort(df[metric_x].unique())
    min_x = metric_x_unique_sorted[0]
    if min_x > 0:
        metric_x_unique_sorted = np.concatenate((np.arange(0, min_x, 1), metric_x_unique_sorted))
    metric_y_unique_sorted = np.sort(df[metric_y].unique())
    
    Z = np.full((len(metric_x_unique_sorted), len(metric_y_unique_sorted)), np.nan)

    for ix, val_x in enumerate(metric_x_unique_sorted):
        for iy, val_y in enumerate(metric_y_unique_sorted):
            if kwargs.get('show_time_spent_LU', False):
                lu = df[(df[metric_x] == val_x) & (df[metric_y] == val_y)]['lu_reform'].mean()
                total = df[(df[metric_x] == val_x) & (df[metric_y] == val_y)]['t_reform'].mean()
                Z[ix, iy] = lu/total
            elif kwargs.get('show_flops', False):
                full = df[(df[metric_x] == val_x) & (df[metric_y] == val_y)]['flops_full'].mean()
                blocked = df[(df[metric_x] == val_x) & (df[metric_y] == val_y)]['flops_blocked'].mean()
                Z[ix,iy] = (blocked - full) / full
            else:
                full = df[(df[metric_x] == val_x) & (df[metric_y] == val_y)]['time_full'].mean()
                blocked = df[(df[metric_x] == val_x) & (df[metric_y] == val_y)]['time_blocked'].mean()
                Z[ix,iy] = (blocked - full) / full

    ax = kwargs.get("ax", None)
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()
    else:
        fig = ax.get_figure()

    if kwargs.get("simple_colorbar", False):
        cmap = 'gist_rainbow'
        norm = None
    else:
        level_jump = 0.1
        max_val = kwargs.get("max_val", None)
        if max_val is None:
            max_val = max(np.nanmax(Z), -np.nanmin(Z))
            max_val = min(1.1, max_val)
        levels = np.arange(level_jump*(np.nanmin(Z)//level_jump-1), level_jump*(np.nanmax(Z)//level_jump+1), level_jump)
        
        # cmap = 'bwr'
        # cmap = 'seismic'
        # cmap = 'Spectral'
        # cmap = 'RdBu'
        # cmap = 'jet'
        cmap = plt.cm.seismic

        # TwoSlopeNorm has weird scaling
        # norm = mcolors.TwoSlopeNorm(vmin=min(-0.01, np.nanmin(Z)), vcenter=0, vmax=max(np.nanmax(Z),0.01))
        
        # good option but continuous colorbar
        # norm = mcolors.TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)

        # discrete colorbar with good scaling
        if kwargs.get('show_time_spent_LU', False):
            levels = np.arange(0, 1, level_jump)
        else:
            levels = np.arange(level_jump*(-max_val//level_jump), level_jump*(max_val//level_jump+2), level_jump)
        norm = mcolors.BoundaryNorm(levels, ncolors=256)
    
    # color nan values in Z in grey
    cmap.set_bad(color='lightgrey')
    
    # cntr = ax.contourf(metric_x_unique_sorted, metric_y_unique_sorted, Z, levels=levels, cmap=cmap, norm=norm)
    # cbar = fig.colorbar(cntr, ax=ax, label='Relative difference')
    mesh = ax.pcolormesh(metric_x_unique_sorted, metric_y_unique_sorted, Z.T, cmap=cmap, norm=norm)
    if kwargs.get('show_time_spent_LU', False):
        cbar = fig.colorbar(mesh, ax=ax, label='Percentage of time\nspent in LU')
    else:
        cbar = fig.colorbar(mesh, ax=ax, label='Relative difference', )

    ax.set_xlabel(metric_x)
    ax.set_ylabel(metric_y)
    plt.tight_layout()
    # plt.show()
    if kwargs.get("save_fig", True):
        appendix = "lu_time_spent_" if kwargs.get("show_time_spent_LU", False) else ("flops_" if kwargs.get("show_flops", False) else "")
        plt.savefig("unittest/reformulation/figures/scaling_2d_" + 
                    appendix + 
                    metric_x + "_" + metric_y + ".png", dpi=300)

def plot_scaling_2d(df, **kwargs):
    metrics = ['nu', 'nx', 'ng', 'rank']

    # filter metrics not in df
    metrics = [metric for metric in metrics if metric in df.columns]

    for i in range(len(metrics)):
        for j in range(i):
            metric_x = metrics[i]
            metric_y = metrics[j]
            create_2d_plot(metric_x, metric_y, df, **kwargs)

def plot_sliced_scaling(df, **kwargs):
    ng_levels = np.concatenate([np.array([0, 1, 2]), np.arange(2, df['ng'].max()+1, 4)[1:]])
    print(ng_levels)
    nb_figures = len(ng_levels) - 1

    # make more or less square arrangement of subfigures
    ncols = int(np.ceil(np.sqrt(nb_figures)))
    nrows = int(np.ceil(nb_figures / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows))

    for fig_idx in range(nb_figures):
        ng_min = ng_levels[fig_idx]
        ng_max = ng_levels[fig_idx+1]

        df_slice = df[(df['ng'] >= ng_min) & (df['ng'] < ng_max)]
        ax = axes[fig_idx // ncols, fig_idx % ncols]
        create_2d_plot('nx', 'nu', df_slice, ax=ax, save_fig=False, max_val=1.2, **kwargs)
        ax.set_title(f'ng in [{ng_min}, {ng_max})')

    for i in range(nb_figures, nrows*ncols):
        fig.delaxes(axes[i // ncols, i % ncols])

    plt.tight_layout()
    # plt.show()
    plt.savefig("unittest/reformulation/figures/scaling_sliced.png", dpi=300)

def plot_rank_scaling(df, **kwargs):
    # rank_levels = [0, 0.2, 0.8, 1]
    # nb_figures = len(rank_levels) - 1

    # # make more or less square arrangement of subfigures
    # ncols = nb_figures
    # nrows = 1
    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols, 3*nrows))

    # for fig_idx in range(nb_figures):
    #     rank_min = rank_levels[fig_idx]
    #     rank_max = rank_levels[fig_idx+1]

    #     # filter df such that rank/nx is in [rank_min, rank_max)
    #     df_slice = df[(df['rank']/df['nx'] >= rank_min) & (df['rank']/df['nx'] < rank_max)]
    #     ax = axes[fig_idx]
    #     create_2d_plot('nx', 'nu', df_slice, ax=ax, save_fig=False, max_val=1.2, **kwargs)
    #     ax.set_xlim(df['nx'].min(), df['nx'].max())
    #     ax.set_title(f'rank/nx in [{rank_min}, {rank_max})')

    # rank < 0.5*nx, 0.5*nx > rank < nx, rank = nx
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(4*3, 3*1))

    # filter df such that rank/nx is in [rank_min, rank_max)
    df_slice = df[df['rank'] <= 0.5*df['nx']]
    ax = axes[0]
    create_2d_plot('nx', 'nu', df_slice, ax=ax, save_fig=False, max_val=1.2, **kwargs)
    ax.set_xlim(df['nx'].min(), df['nx'].max())
    ax.set_title(f'rank of M1 $<$ 0.5 nx')

    df_slice = df[(df['rank'] > 0.5*df['nx']) & (df['rank'] < df['nx'])]
    ax = axes[1]
    create_2d_plot('nx', 'nu', df_slice, ax=ax, save_fig=False, max_val=1.2, **kwargs)
    ax.set_xlim(df['nx'].min(), df['nx'].max())
    ax.set_title(f'0.5 nx $<$ rank of M1 $<$ nx')

    df_slice = df[df['rank'] == df['nx']]
    ax = axes[2]
    create_2d_plot('nx', 'nu', df_slice, ax=ax, save_fig=False, max_val=1.2, **kwargs)
    ax.set_xlim(df['nx'].min(), df['nx'].max())
    ax.set_title(f'rank of M1 = nx')
    
    plt.tight_layout()
    # plt.show()
    plt.savefig("unittest/reformulation/figures/scaling_sliced_rank.png", dpi=300)

def show_time_spent_LU(**kwargs):
    df = pd.read_csv('build_docker/random_benchmark_results_huge.csv')
    plot_scaling_2d(df, show_time_spent_LU=True)

# read the csv file (nu, nx, ng, time_full, time_blocked)
df = pd.read_csv('build_docker/blocked_lu_timings_general.csv')

# filter data such that ng <= nx + nu
df = df[df['ng'] <= df['nx'] + df['nu']]

# fc = FlopCounter()
# fc.add_flops_to_data(df)

# plot_scaling(df, show_flops=False)
# plot_scaling(df, show_flops=False, show_relative=True)
# plot_scaling_2d(df, simple_colorbar=False)
# plot_sliced_scaling(df)
plot_rank_scaling(df)
# show_time_spent_LU()



