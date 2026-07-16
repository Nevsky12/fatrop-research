import json
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os
import pandas as pd

def visualize_holonomic_result(data, **kwargs):
    dimension = data["generator_data"]["n"]
    control_level = data["generator_data"]["control_level"]
    problem_type = data["problem type"]
    solver = data["solver"]
    problem_name = data["generator_data"]["problem_name"]

    axs = kwargs.get('axs', None)
    if axs is None:
        fig, axs = plt.subplots(control_level+1, 1)

    tt = np.linspace(0, data["generator_data"]['K'] * data["generator_data"]['dt'], data["generator_data"]['K'])

    state_names = ['pos', 'vel', 'acc', 'jerk', 'snap']
    for i in range(control_level):
        for j in range(dimension):
            axs[i].plot(tt, [x[i*dimension + j] for x in data['states']], label=f'{state_names[i]} ({j+1})')
        axs[i].set_xlim([tt[0], tt[-1]])
        axs[i].set_ylabel(state_names[i])
        axs[i].legend()
    
    # plot control
    for j in range(dimension):
        axs[control_level].plot(tt[:-1], [u[j] for u in data['inputs']], label=f'u ({j+1})')
        axs[control_level].set_xlim([tt[0], tt[-1]])
        axs[control_level].set_ylabel('Control')
        axs[control_level].set_xlabel('Time')
        axs[control_level].legend()

    plt.suptitle(f"{problem_type} - {solver}")

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"unittest/implicit/figures/ocp_result_{problem_name}_dim{dimension}_ctl{control_level}_{problem_type}_{solver}.png", dpi=300)
    plt.close()

def visualize_trucktrailer_result(data, **kwargs):
    n = data["generator_data"]["n"]
    problem_type = data["problem type"]
    solver = data["solver"]
    problem_name = data["generator_data"]["problem_name"]
    L = 1.0
    M = 0.0

    axs = kwargs.get('axs', None)
    if axs is None:
        fig, axs = plt.subplots(2, 1, height_ratios=[3, 1])

    tt = np.linspace(0, data["generator_data"]['K'] * data["generator_data"]['dt'], data["generator_data"]['K'])

    th = np.array([data['states'][k][:-2] for k in range(data["generator_data"]['K'])]).transpose()
    xy = np.array([data['states'][k][-2:] for k in range(data["generator_data"]['K'])]).transpose()

    xx = [[] for _ in range(n+1)]   # xx[veh_idx][time_idx]
    yy = [[] for _ in range(n+1)]
    th = [[th[i,k] for k in range(data["generator_data"]['K'])] for i in range(n+1)]  # th[veh_idx][time_idx]

    for k in range(data["generator_data"]['K']):
        # position of last trailer
        x = xy[0, k]
        y = xy[1, k]

        xx[n].append(x)
        yy[n].append(y)

        for i in range(n-1, -1, -1):
            xx[i].append(xx[i+1][k] + L*np.cos(th[i+1][k]) + M*np.cos(th[i][k]))
            yy[i].append(yy[i+1][k] + L*np.sin(th[i+1][k]) + M*np.sin(th[i][k]))

    # plot trajectory of each trailer
    for i in range(n+1):
        axs[0].plot(xx[i], yy[i], label=f'vehicle {i}')
        step = 5
        axs[0].quiver(xx[i][::step], yy[i][::step], np.cos(th[i][::step]), np.sin(th[i][::step]), scale=20, width=0.003, zorder=5)
    axs[0].set_aspect('equal', 'box')
    axs[0].legend()
    
    # plot control
    axs[1].plot(tt[:-1], [u[0] for u in data['inputs']], label='v')
    axs[1].plot(tt[:-1], [u[1] for u in data['inputs']], label='w')
    axs[1].legend()

    plt.suptitle(f"{problem_type} - {solver}")

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"unittest/implicit/figures/ocp_result_{problem_name}_n_{n}_{problem_type}_{solver}.png", dpi=300)
    plt.close()

def visualize_planar_robot_result(data, **kwargs):
    n = data["generator_data"]["n"]
    problem_type = data["problem type"]
    solver = data["solver"]
    problem_name = data["generator_data"]["problem_name"]

    axs = kwargs.get('axs', None)
    if axs is None:
        fig, axs = plt.subplots(2, 1, height_ratios=[3, 1])

    tt = np.linspace(0, data["generator_data"]['K'] * data["generator_data"]['dt'], data["generator_data"]['K'])

    # get positions of joints
    angles = np.array([data['states'][k][:n] for k in range(data["generator_data"]['K'])]).transpose()
    print(data["generator_data"])
    link_length = data["generator_data"]["l"]
    xx = [[] for _ in range(n+1)]   # xx[joint_idx][time_idx]
    yy = [[] for _ in range(n+1)]
    for k in range(data["generator_data"]['K']):
        x, y = 0, 0
        xx[0].append(x)
        yy[0].append(y)
        current_angle = 0
        for i in range(n):
            # current_angle = angles[i,k]
            current_angle = data['states'][k][i]
            x -= link_length * np.sin(current_angle)
            y += link_length * np.cos(current_angle)
            xx[i+1].append(x)
            yy[i+1].append(y)

    start_color = np.array([1, 0, 0])
    end_color = np.array([0, 0, 1])
    step = 15
    for k in range(0, data["generator_data"]['K'], step):
        c = k/data["generator_data"]['K']
        color = start_color * (1 - c) + end_color * c
        axs[0].plot([xx[i][k] for i in range(n+1)], [yy[i][k] for i in range(n+1)], 'o-', color=color)

    axs[0].set_aspect('equal', 'box')
    axs[0].legend()
    
    # plot control
    for j in range(n):
        axs[1].plot(tt[:-1], [u[j] for u in data['inputs']], label='u'+str(j))
    axs[1].legend()

    plt.suptitle(f"{problem_type} - {solver}")

    plt.tight_layout()
    # plt.show()
    plt.savefig(f"unittest/implicit/figures/ocp_result_{problem_name}_n_{n}_{problem_type}_{solver}.png", dpi=300)
    plt.close()

def animate_planar_robot_result(data, **kwargs):
    n = data["generator_data"]["n"]
    problem_type = data["problem type"]
    solver = data["solver"]
    problem_name = data["generator_data"]["problem_name"]

    axs = kwargs.get('axs', None)
    if axs is None:
        fig, axs = plt.subplots(1, 1, height_ratios=[1])

    tt = np.linspace(0, data["generator_data"]['K'] * data["generator_data"]['dt'], data["generator_data"]['K'])

    # get positions of joints
    angles = np.array([data['states'][k][:n] for k in range(data["generator_data"]['K'])]).transpose()
    print(data["generator_data"])
    link_length = data["generator_data"]["l"]
    xx = [[] for _ in range(n+1)]   # xx[joint_idx][time_idx]
    yy = [[] for _ in range(n+1)]
    for k in range(data["generator_data"]['K']):
        x, y = 0, 0
        xx[0].append(x)
        yy[0].append(y)
        current_angle = 0
        for i in range(n):
            # current_angle = angles[i,k]
            current_angle = data['states'][k][i]
            x -= link_length * np.sin(current_angle)
            y += link_length * np.cos(current_angle)
            xx[i+1].append(x)
            yy[i+1].append(y)

    def update(frame):
        axs.clear()
        start_color = np.array([1, 0, 0])
        end_color = np.array([0, 0, 1])
        step = 15
        k = frame
        c = k/data["generator_data"]['K']
        color = start_color * (1 - c) + end_color * c
        plt.plot([xx[i][k] for i in range(n+1)], [yy[i][k] for i in range(n+1)], 'o-', color=color)
        axs.set_aspect('equal', 'box')
        axs.set_xlim([-(n+1)*link_length, (n+1)*link_length])
        axs.set_ylim([-(n+1)*link_length, (n+1)*link_length])
        axs.set_title(f'Time: {tt[frame]:.2f}s')

    sim_ms_per_frame = 100
    real_ms_per_frame = data["generator_data"]['dt'] * 1000
    frame_step = max(1, int(sim_ms_per_frame / real_ms_per_frame))
    ani = FuncAnimation(fig, update, frames=range(0, data["generator_data"]['K'], frame_step), interval=sim_ms_per_frame)
    ani.save(f"unittest/implicit/figures/ocp_result_{problem_name}_n_{n}_{problem_type}_{solver}_animation.gif", writer='imagemagick', fps=10)
    plt.close()

def visualize_performance(df):
    nx_vals = df['nx'].unique()
    nx_vals = np.sort(nx_vals)
    problem_name = df["problem_name"][0]
    print(f"Visualizing performance for problem: {problem_name}")

    # problem_types = df['problem type'].unique()
    problem_types = ['explicit', 'implicit', 'reformulated']

    colors = {'implicit': ['royalblue'],
              'explicit': ['firebrick'],
              'reformulated': ['midnightblue']}
    
    # show solver times
    plt.figure()
    for i, problem_type in enumerate(problem_types):
        times = []
        df_pt = df[df['problem type'] == problem_type]
        for nx in nx_vals:
            df_nx = df_pt[df_pt['nx'] == nx]
            times.append((df_nx['time_solver'] / df_nx['nb_iterations']).mean())
    
        times = np.array(times)

        bar_width = 0.2
        index = nx_vals + (i-1)*bar_width
        bb = np.zeros(len(nx_vals))
        plt.bar(index, times, bar_width, bottom=bb, label=f'{problem_type}', color=colors[problem_type][0])    
    plt.xlabel('Number of state variables (nx)')
    plt.ylabel('average solver time per iteration (s)')
    plt.title(problem_name)
    plt.xticks(nx_vals, nx_vals)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"unittest/implicit/figures/ocp_{problem_name}_performance_comparison_avg_solver_times.png", dpi=300)
    plt.close()

    # show function evaluation times
    plt.figure()
    for i, problem_type in enumerate(problem_types):
        times = []
        df_pt = df[df['problem type'] == problem_type]
        for nx in nx_vals:
            df_nx = df_pt[df_pt['nx'] == nx]
            times.append((df_nx['time_function_evaluation'] / df_nx['nb_iterations']).mean())
    
        times = np.array(times)

        bar_width = 0.2
        index = nx_vals + (i-1)*bar_width
        bb = np.zeros(len(nx_vals))
        plt.bar(index, times, bar_width, bottom=bb, label=f'{problem_type}', color=colors[problem_type][0])    
    plt.xlabel('Number of state variables (nx)')
    plt.ylabel('average func eval time per iteration (s)')
    plt.title(problem_name)
    plt.xticks(nx_vals, nx_vals)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"unittest/implicit/figures/ocp_{problem_name}_performance_comparison_avg_func_eval_times.png", dpi=300)
    plt.close()

    # show nb of iterations
    plt.figure()
    for i, problem_type in enumerate(problem_types):
        times = []
        df_pt = df[df['problem type'] == problem_type]
        for nx in nx_vals:
            df_nx = df_pt[df_pt['nx'] == nx]
            times.append((df_nx['nb_iterations']).mean())
    
        times = np.array(times)

        bar_width = 0.2
        index = nx_vals + (i-1)*bar_width
        bb = np.zeros(len(nx_vals))
        plt.bar(index, times, bar_width, bottom=bb, label=f'{problem_type}', color=colors[problem_type][0])    
    plt.xlabel('Number of state variables (nx)')
    plt.ylabel('nb of iterations')
    plt.title(problem_name)
    plt.xticks(nx_vals, nx_vals)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"unittest/implicit/figures/ocp_{problem_name}_performance_comparison_nb_iterations.png", dpi=300)
    plt.close()

    # show total time
    plt.figure()
    for i, problem_type in enumerate(problem_types):
        times = []
        df_pt = df[df['problem type'] == problem_type]
        for nx in nx_vals:
            df_nx = df_pt[df_pt['nx'] == nx]
            times.append((df_nx['time_total']).mean())
    
        times = np.array(times)

        bar_width = 0.2
        index = nx_vals + (i-1)*bar_width
        bb = np.zeros(len(nx_vals))
        plt.bar(index, times, bar_width, bottom=bb, label=f'{problem_type}', color=colors[problem_type][0])    
    plt.xlabel('Number of state variables (nx)')
    plt.ylabel('total time (s)')
    plt.title(problem_name)
    plt.xticks(nx_vals, nx_vals)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"unittest/implicit/figures/ocp_{problem_name}_performance_comparison_total_time.png", dpi=300)
    plt.close()

# def visualize_performance_different_problems(list_of_dfs):
#     problem_types = ['explicit', 'implicit', 'reformulated']
#     colors = {'implicit': ['royalblue'],
#               'explicit': ['firebrick'],
#               'reformulated': ['midnightblue']}
    
#     # for each performance metric, create a figure comparing averages of each df
#     performance_metrics = ['time_solver', 'time_function_evaluation', 'nb_iterations', 'time_total']
#     problem_names = [df["problem_name"].values[0] for df in list_of_dfs]

#     for metric in performance_metrics:
#         plt.figure()
#         for i, problem_type in enumerate(problem_types):
#             times = []
#             for df in list_of_dfs:
#                 df_pt = df[df['problem type'] == problem_type]
#                 times.append(df_pt[metric].mean())
        
#             times = np.array(times)

#             bar_width = 0.2
#             index = np.arange(len(list_of_dfs)) + (i-1)*bar_width
#             bb = np.zeros(len(list_of_dfs))
#             plt.bar(index, times, bar_width, bottom=bb, label=f'{problem_type}', color=colors[problem_type][0])
#         plt.ylabel(metric.replace('_', ' '))
#         plt.title(f'Performance comparison: {metric.replace("_", " ")}')
#         plt.xticks(np.arange(len(list_of_dfs)), problem_names)
#         plt.xlabel('Problem')
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig(f"unittest/implicit/figures/ocp_performance_comparison_{metric}.png", dpi=300)
#         plt.close()
    
def visualize_performance_different_problems(list_of_dfs):
    problem_types = ['explicit', 'implicit', 'reformulated']
    colors = {'implicit': ['royalblue'],
              'explicit': ['firebrick'],
              'reformulated': ['midnightblue']}
    
    # for each performance metric, create a figure comparing averages of each df
    performance_metrics = ['time_solver', 'time_function_evaluation', 'nb_iterations', 'time_total', 'compute_search_dir']
    ylabels = ['time [s]', 'time [s]', '', 'time [s]', 'time [s]']
    problem_names = [df["problem_name"].values[0] for df in list_of_dfs]

    for metric in performance_metrics:
        plt.figure(figsize=(2*len(problem_names), 5))
        for i, problem_name in enumerate(problem_names):
            plt.subplot(1, len(problem_names), i+1)

            times = []
            df = list_of_dfs[i]
            for j, problem_type in enumerate(problem_types):
                df_pt = df[df['problem type'] == problem_type]
                times.append(df_pt[metric].mean())
                plt.bar(-1+j, times[-1], color=colors[problem_type][0], label=problem_type)
        
            plt.xticks([])
            plt.title(problem_name)
            plt.gca().set_ylabel(ylabels[performance_metrics.index(metric)])

        plt.tight_layout()
        # put the legend of the last plot under the figure
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9)
        plt.figlegend(handles, labels, loc='lower center', ncol=len(problem_types), bbox_to_anchor=(0.5, 0.02))

        plt.suptitle(metric)
        plt.savefig(f"unittest/implicit/figures/ocp_performance_comparison_{metric}.png", dpi=300)
        plt.close()



def visualize_performance_quadruped(df):
    problem_name = df["problem_name"][0]
    print(f"Visualizing performance for problem: {problem_name}")

    # problem_types = df['problem type'].unique()
    problem_types = ['explicit', 'implicit', 'reformulated']

    colors = {'implicit': ['royalblue'],
              'explicit': ['firebrick'],
              'reformulated': ['midnightblue']}
    
    vx_vy_pairs = df[['v0x', 'v0y']].drop_duplicates().values
    nx_vals = np.arange(len(vx_vy_pairs))
    
    # show solver times
    plt.figure()
    for i, problem_type in enumerate(problem_types):
        times = []
        df_pt = df[df['problem type'] == problem_type]
        # for nx in nx_vals:
        #     df_nx = df_pt[df_pt['nx'] == nx]
        #     times.append((df_nx['time_solver'] / df_nx['nb_iterations']).mean())
        for (vx, vy) in vx_vy_pairs:
            df_v = df_pt[(df_pt['v0x'] == vx) & (df_pt['v0y'] == vy)]
            times.append((df_v['time_solver'] / df_v['nb_iterations']).mean())
    
        times = np.array(times)

        bar_width = 0.2
        index = nx_vals + (i-1)*bar_width
        bb = np.zeros(len(nx_vals))
        plt.bar(index, times, bar_width, bottom=bb, label=f'{problem_type}', color=colors[problem_type][0])    
    # plt.xlabel('Number of state variables (nx)')
    plt.ylabel('average solver time per iteration (s)')
    plt.title(problem_name)
    # plt.xticks(nx_vals, nx_vals)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"unittest/implicit/figures/ocp_{problem_name}_performance_comparison_avg_solver_times.png", dpi=300)
    plt.close()

    # show function evaluation times
    plt.figure()
    for i, problem_type in enumerate(problem_types):
        times = []
        df_pt = df[df['problem type'] == problem_type]
        # for nx in nx_vals:
        #     df_nx = df_pt[df_pt['nx'] == nx]
        #     times.append((df_nx['time_function_evaluation'] / df_nx['nb_iterations']).mean())
        for (vx, vy) in vx_vy_pairs:
            df_v = df_pt[(df_pt['v0x'] == vx) & (df_pt['v0y'] == vy)]
            times.append((df_v['time_function_evaluation'] / df_v['nb_iterations']).mean())
    
        times = np.array(times)

        bar_width = 0.2
        index = nx_vals + (i-1)*bar_width
        bb = np.zeros(len(nx_vals))
        plt.bar(index, times, bar_width, bottom=bb, label=f'{problem_type}', color=colors[problem_type][0])    
    # plt.xlabel('Number of state variables (nx)')
    plt.ylabel('average func eval time per iteration (s)')
    plt.title(problem_name)
    # plt.xticks(nx_vals, nx_vals)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"unittest/implicit/figures/ocp_{problem_name}_performance_comparison_avg_func_eval_times.png", dpi=300)
    plt.close()

    # show nb of iterations
    plt.figure()
    for i, problem_type in enumerate(problem_types):
        times = []
        df_pt = df[df['problem type'] == problem_type]
        # for nx in nx_vals:
        #     df_nx = df_pt[df_pt['nx'] == nx]
        #     times.append((df_nx['nb_iterations']).mean())
        for (vx, vy) in vx_vy_pairs:
            df_v = df_pt[(df_pt['v0x'] == vx) & (df_pt['v0y'] == vy)]
            times.append((df_v['nb_iterations']).mean())
    
        times = np.array(times)

        bar_width = 0.2
        index = nx_vals + (i-1)*bar_width
        bb = np.zeros(len(nx_vals))
        plt.bar(index, times, bar_width, bottom=bb, label=f'{problem_type}', color=colors[problem_type][0])    
    # plt.xlabel('Number of state variables (nx)')
    plt.ylabel('nb of iterations')
    plt.title(problem_name)
    # plt.xticks(nx_vals, nx_vals)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"unittest/implicit/figures/ocp_{problem_name}_performance_comparison_nb_iterations.png", dpi=300)
    plt.close()

    # show total time
    plt.figure()
    for i, problem_type in enumerate(problem_types):
        times = []
        df_pt = df[df['problem type'] == problem_type]
        # for nx in nx_vals:
        #     df_nx = df_pt[df_pt['nx'] == nx]
        #     times.append((df_nx['time_total']).mean())
        for (vx, vy) in vx_vy_pairs:
            df_v = df_pt[(df_pt['v0x'] == vx) & (df_pt['v0y'] == vy)]
            times.append((df_v['time_total']).mean())
    
        times = np.array(times)

        bar_width = 0.2
        index = nx_vals + (i-1)*bar_width
        bb = np.zeros(len(nx_vals))
        plt.bar(index, times, bar_width, bottom=bb, label=f'{problem_type}', color=colors[problem_type][0])    
    # plt.xlabel('Number of state variables (nx)')
    plt.ylabel('total time (s)')
    plt.title(problem_name)
    # plt.xticks(nx_vals, nx_vals)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"unittest/implicit/figures/ocp_{problem_name}_performance_comparison_total_time.png", dpi=300)
    plt.close()

def visualize_func_eval_breakdown(df, fig_name_appendix=""):
    # group by problem type
    # problem_types = df['problem_type'].unique()
    problem_types = ['explicit', 'implicit', 'reformulated']
    colors = ['firebrick', 'royalblue', 'midnightblue']
    df_per_type = [df[df['problem_type'] == pt] for pt in problem_types]

    cols_to_discard = ['problem_type', 'initialization', 'total', 'rest time']

    # get all other columns of the df
    other_columns = [col for col in df.columns if col not in cols_to_discard]
    
    plt.figure()
    bar_width = 0.3

    bar_xxs = np.arange(len(other_columns))*1.0
    # create some spaces
    bar_xxs[2:] += 1.0
    # bar_xxs[-2:] += 1.0

    for i, df_pt in enumerate(df_per_type):
        means = df_pt[other_columns].mean()
        stds = df_pt[other_columns].std()

        index = bar_xxs + (i-1)*bar_width

        bar_values = [means[k] for k in other_columns]
        bar_stds = [stds[k] for k in other_columns]
        try:
            label = f"{df_pt['problem_type'].values[0]}"
        except:
            label = None
        plt.bar(index, bar_values, bar_width, label=label, color=colors[i])
        # plt.bar(index, bar_values, bar_width, label=f"{df_pt['problem_type'].values[0]}", color=colors[i], yerr=bar_stds, capsize=5)

    plt.xlabel('Function evaluation components')
    plt.ylabel('Time (s)')
    plt.title('Function evaluation time breakdown')
    plt.xticks(bar_xxs, other_columns, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"unittest/implicit/figures/ocp_function_evaluation_time_breakdown_{fig_name_appendix}.png", dpi=300)
    plt.close()

def print_performance_table(df):
    # rows: nb iterations, t_func_avg, t_func_total, t_fatrop_avg, t_fatrop_total, t_total
    # columns: explicit, implicit, reformulated
    problem_types = ['explicit', 'implicit', 'reformulated']
    performance_data = {}

    for problem_type in problem_types:
        df_pt = df[df['problem type'] == problem_type]
        performance_data[problem_type] = [
            df_pt['nb_iterations'].mean(),
            df_pt['time_function_evaluation'].mean()/df_pt['nb_iterations'].mean(),
            df_pt['time_function_evaluation'].mean(),
            df_pt['time_solver'].mean()/df_pt['nb_iterations'].mean(),
            df_pt['time_solver'].mean(),
            df_pt['time_total'].mean()
        ]

    performance_df = pd.DataFrame(performance_data, index=[
        'nb iterations', 
        't_func_avg (s)', 
        't_func_total (s)', 
        't_fatrop_avg (s)', 
        't_fatrop_total (s)', 
        't_total (s)'
    ])

    print(performance_df.to_markdown())
    print("")

def print_holonomic_result_differences(df):
    problem_types = ['explicit', 'implicit', 'reformulated']
    results = {}
    baseline = 'implicit'

    for problem_type in problem_types:
        if problem_type == baseline:
            continue

        df_pt = df[df['problem type'] == problem_type]
        df_bl = df[df['problem type'] == baseline]

        avg_err_states = 0
        avg_err_inputs = 0
        nb_entries = 0
        
        # ensure same dimensions and control levels
        for (dim, ctl) in df_pt[['dimension', 'control level']].values:
            df_pt_dc = df_pt[(df_pt['dimension'] == dim) & (df_pt['control level'] == ctl)]
            df_bl_dc = df_bl[(df_bl['dimension'] == dim) & (df_bl['control level'] == ctl)]

            if df_pt_dc.empty or df_bl_dc.empty:
                continue

            # compare states and inputs
            states_bl = np.array(df_bl_dc['states'].tolist()).flatten()
            inputs_bl = np.array(df_bl_dc['inputs'].tolist()).flatten()

            if problem_type == 'reformulated':
                states_pt = np.array(df_pt_dc['states'].tolist()).flatten()
                nu = int(df_bl_dc['nu'].values[0])
                K = df_bl_dc['K'].values[0]
                inputs_pt = np.zeros((nu*(K-1)))

                for k in range(K-1):
                    inputs_pt[nu*k:nu*(k+1)] = np.array(df_pt_dc['inputs'].to_list())[:, k, :nu]
                inputs_pt = inputs_pt.flatten()
            else:
                states_pt = np.array(df_pt_dc['states'].tolist()).flatten()
                inputs_pt = np.array(df_pt_dc['inputs'].tolist()).flatten()

            states_diff = np.abs(states_bl - states_pt)
            inputs_diff = np.abs(inputs_bl - inputs_pt)

            avg_err_states += np.linalg.norm(states_diff)
            avg_err_inputs += np.linalg.norm(inputs_diff)
            nb_entries += 1

        if nb_entries > 0:
            results[problem_type] = {'state_diff_norm': avg_err_states/nb_entries, 'input_diff_norm': avg_err_inputs/nb_entries}
        else:
            results[problem_type] = {'state_diff_norm': -1, 'input_diff_norm': -1}

    df = pd.DataFrame.from_dict(results, orient='index', columns=['state_diff_norm', 'input_diff_norm'])
    print(df.to_markdown())
    print("")

def print_trucktrailer_result_differences(df):
    problem_types = ['explicit', 'implicit', 'reformulated']
    results = {}
    baseline = 'implicit'

    for problem_type in problem_types:
        if problem_type == baseline:
            continue

        df_pt = df[df['problem type'] == problem_type]
        df_bl = df[df['problem type'] == baseline]

        avg_err_states = 0
        avg_err_inputs = 0
        nb_entries = 0
        
        # ensure same dimensions and control levels
        for n in df_pt['n'].values:
            df_pt_dc = df_pt[df_pt['n'] == n]
            df_bl_dc = df_bl[df_bl['n'] == n]

            if df_pt_dc.empty or df_bl_dc.empty:
                continue

            # compare states and inputs
            states_bl = np.array(df_bl_dc['states'].tolist()).flatten()
            inputs_bl = np.array(df_bl_dc['inputs'].tolist()).flatten()

            if problem_type == 'reformulated':
                states_pt = np.array(df_pt_dc['states'].tolist()).flatten()
                nu = int(df_bl_dc['nu'].values[0])
                K = df_bl_dc['K'].values[0]
                inputs_pt = np.zeros((nu*(K-1)))

                for k in range(K-1):
                    inputs_pt[nu*k:nu*(k+1)] = np.array(df_pt_dc['inputs'].to_list())[:, k, :nu]
                inputs_pt = inputs_pt.flatten()
            else:
                states_pt = np.array(df_pt_dc['states'].tolist()).flatten()
                inputs_pt = np.array(df_pt_dc['inputs'].tolist()).flatten()

            states_diff = np.abs(states_bl - states_pt)
            inputs_diff = np.abs(inputs_bl - inputs_pt)

            avg_err_states += np.linalg.norm(states_diff)
            avg_err_inputs += np.linalg.norm(inputs_diff)
            nb_entries += 1

        if nb_entries > 0:
            results[problem_type] = {'state_diff_norm': avg_err_states/nb_entries, 'input_diff_norm': avg_err_inputs/nb_entries}
        else:
            results[problem_type] = {'state_diff_norm': -1, 'input_diff_norm': -1}

    df = pd.DataFrame.from_dict(results, orient='index', columns=['state_diff_norm', 'input_diff_norm'])
    print(df.to_markdown())
    print("")


if __name__ == "__main__":
    # find all files in build/unittest/ocp_results/
    dir_path = os.path.join(os.path.dirname(__file__), '../../../build/unittest/ocp_results')

    # store timing info in dataframe
    df_holonomic = pd.DataFrame(columns=['problem_name', 'problem type', 
                                         'solver', 'dimension', 
                                         'control level', 'K', 'nx', 'nu',
                                         'time_total', 
                                         'time_solver', 
                                         'time_function_evaluation', 
                                         'compute_search_dir',
                                         'nb_iterations',
                                         'states', 'inputs'])
    df_trucktrailer = pd.DataFrame(columns=['problem_name', 'problem type', 
                                            'solver', 'n',
                                            'K', 'nx', 'nu',
                                            'time_total', 
                                            'time_solver', 
                                            'time_function_evaluation', 
                                            'compute_search_dir',
                                            'nb_iterations',
                                            'states', 'inputs'])
    
    df_planarrobot = pd.DataFrame(columns=['problem_name', 'problem type',
                                            'solver', 'n',
                                            'K', 'nx', 'nu', 'l', 
                                            'time_total', 
                                            'time_solver', 
                                            'time_function_evaluation', 
                                            'compute_search_dir',
                                            'nb_iterations',
                                            'states', 'inputs'])
    
    df_quadruped = pd.DataFrame(columns=['problem_name', 'problem type',
                                          'solver', 'v0x', 'v0y',
                                          'K', 'nx', 'nu', 
                                          'time_total', 
                                          'time_solver', 
                                          'time_function_evaluation', 
                                          'compute_search_dir',
                                          'nb_iterations',
                                          'states', 'inputs'])

    df_batch_reactor = pd.DataFrame(columns=['problem_name', 'problem type',
                                             'solver', 'n',
                                             'K', 'nx', 'nu', 
                                             'time_total', 
                                             'time_solver', 
                                             'time_function_evaluation', 
                                             'compute_search_dir',
                                             'nb_iterations',
                                             'states', 'inputs'])
    
    df_solar_receiver_reactor = pd.DataFrame(columns=['problem_name', 'problem type',
                                                'solver',
                                                'K', 'nx', 'nu', 
                                                'time_total', 
                                                'time_solver', 
                                                'time_function_evaluation', 
                                                'compute_search_dir',
                                                'nb_iterations',
                                                'states', 'inputs'])

    all_jsons = []
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.json'):
            file_path = os.path.join(dir_path, file_name)
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                except:
                    continue
            all_jsons.append(data)
    
    for data in all_jsons:
        if data["generator_data"]["problem_name"] == "holonomic":
            df_holonomic.loc[len(df_holonomic)] = {
                'problem_name': data["generator_data"]["problem_name"],
                'problem type': data['problem type'],
                'solver': data['solver'],
                'dimension': data["generator_data"]["n"],
                'control level': data["generator_data"]["control_level"],
                'K': data["generator_data"]['K'],
                'nx': data["generator_data"]['nx'], 
                'nu': data["generator_data"]['nu'],
                'time_total': data["metadata"]["timing_statistics"]['total'],
                'time_solver': data["metadata"]["timing_statistics"]['fatrop'],
                'time_function_evaluation': data["metadata"]["timing_statistics"]['function evaluation'],
                'compute_search_dir': data["metadata"]["timing_statistics"]['compute search dir'],
                'nb_iterations': data["metadata"]['iterations'],
                'states': data['states'],
                'inputs': data['inputs']
            }
            
            # visualize_holonomic_result(data)
        
        elif data["generator_data"]["problem_name"] == "truck_trailer":
            df_trucktrailer.loc[len(df_trucktrailer)] = {
                'problem_name': data["generator_data"]["problem_name"],
                'problem type': data['problem type'],
                'solver': data['solver'],
                'n': data["generator_data"]["n"],
                'K': data["generator_data"]['K'], 
                'nx': data["generator_data"]['nx'], 
                'nu': data["generator_data"]['nu'],
                'time_total': data["metadata"]["timing_statistics"]['total'],
                'time_solver': data["metadata"]["timing_statistics"]['fatrop'],
                'time_function_evaluation': data["metadata"]["timing_statistics"]['function evaluation'],
                'compute_search_dir': data["metadata"]["timing_statistics"]['compute search dir'],
                'nb_iterations': data["metadata"]['iterations'],
                'states': data['states'],
                'inputs': data['inputs']
            }
        
            # visualize_trucktrailer_result(data)

        elif data["generator_data"]["problem_name"] == "planar_robot":
            df_planarrobot.loc[len(df_planarrobot)] = {
                'problem_name': data["generator_data"]["problem_name"],
                'problem type': data['problem type'],
                'solver': data['solver'],
                'n': data["generator_data"]["n"],
                'K': data["generator_data"]['K'], 
                'nx': data["generator_data"]['nx'], 
                'nu': data["generator_data"]['nu'],
                'l': data["generator_data"]['l'],
                'time_total': data["metadata"]["timing_statistics"]['total'],
                'time_solver': data["metadata"]["timing_statistics"]['fatrop'],
                'time_function_evaluation': data["metadata"]["timing_statistics"]['function evaluation'],
                'compute_search_dir': data["metadata"]["timing_statistics"]['compute search dir'],
                'nb_iterations': data["metadata"]['iterations'],
                'states': data['states'],
                'inputs': data['inputs']
            }

            # visualize_planar_robot_result(data)
            # animate_planar_robot_result(data)
        
        elif data["generator_data"]["problem_name"] == "quadruped":
            df_quadruped.loc[len(df_quadruped)] = {
                'problem_name': data["generator_data"]["problem_name"],
                'problem type': data['problem type'],
                'solver': data['solver'],
                'K': data["generator_data"]['K'], 
                'v0x': data["generator_data"]['v0x'],
                'v0y': data["generator_data"]['v0y'],
                'nx': data["generator_data"]['nx'], 
                'nu': data["generator_data"]['nu'],
                'time_total': data["metadata"]["timing_statistics"]['total'],
                'time_solver': data["metadata"]["timing_statistics"]['fatrop'],
                'time_function_evaluation': data["metadata"]["timing_statistics"]['function evaluation'],
                'compute_search_dir': data["metadata"]["timing_statistics"]['compute search dir'],
                'nb_iterations': data["metadata"]['iterations'],
                'states': data['states'],
                'inputs': data['inputs']
            }
            
            # visualize_quadruped_result(data)

        elif data["generator_data"]["problem_name"] == "batch_reactor":
            df_batch_reactor.loc[len(df_batch_reactor)] = {
                'problem_name': data["generator_data"]["problem_name"],
                'problem type': data['problem type'],
                'solver': data['solver'],
                'n': data["generator_data"]["n"],
                'K': data["generator_data"]['K'], 
                'nx': data["generator_data"]['nx'], 
                'nu': data["generator_data"]['nu'],
                'time_total': data["metadata"]["timing_statistics"]['total'],
                'time_solver': data["metadata"]["timing_statistics"]['fatrop'],
                'time_function_evaluation': data["metadata"]["timing_statistics"]['function evaluation'],
                'compute_search_dir': data["metadata"]["timing_statistics"]['compute search dir'],
                'nb_iterations': data["metadata"]['iterations'],
                'states': data['states'],
                'inputs': data['inputs']
            }

        elif data["generator_data"]["problem_name"] == "solar_receiver_reactor":
            # check if the entry just added is the explicit one
            if data['problem type'] == 'explicit':
                # timings are not reliable, so set to zero
                data['metadata']['timing_statistics'] = {
                    'total': 0.0,
                    'fatrop': 0.0,
                    'function evaluation': 0.0,
                    'compute search dir': 0.0
                }
                data['metadata']['iterations'] = 1e-8
            df_solar_receiver_reactor.loc[len(df_solar_receiver_reactor)] = {
                'problem_name': data["generator_data"]["problem_name"],
                'problem type': data['problem type'],
                'solver': data['solver'],
                'K': data["generator_data"]['K'], 
                'nx': data["generator_data"]['nx'], 
                'nu': data["generator_data"]['nu'],
                'time_total': data["metadata"]["timing_statistics"]['total'],
                'time_solver': data["metadata"]["timing_statistics"]['fatrop'],
                'time_function_evaluation': data["metadata"]["timing_statistics"]['function evaluation'],
                'compute_search_dir': data["metadata"]["timing_statistics"]['compute search dir'],
                'nb_iterations': data["metadata"]['iterations'],
                'states': data['states'],
                'inputs': data['inputs']
            }
        
        else:
            print(f"Unknown problem name: {data["generator_data"]['problem_name']}")

    
    df_func_eval_breakdown = pd.DataFrame(columns=
        ['problem_type', 'compute search dir', 'fatrop', 'initialization', 
         'eval objective', 'eval gradient', 'eval constraint violation',
          'eval jacobian', 'eval hessian', 'function evaluation','rest time', 
          'total'])
    df_func_eval_holonomic_breakdown = pd.DataFrame(columns=
        ['problem_type', 'compute search dir', 'fatrop', 'initialization', 
         'eval objective', 'eval gradient', 'eval constraint violation',
          'eval jacobian', 'eval hessian', 'function evaluation','rest time', 
          'total'])
    df_func_eval_truck_trailer_breakdown = pd.DataFrame(columns=
        ['problem_type', 'compute search dir', 'fatrop', 'initialization', 
         'eval objective', 'eval gradient', 'eval constraint violation',
          'eval jacobian', 'eval hessian', 'function evaluation','rest time', 
          'total'])
    df_func_eval_planar_robot_breakdown = pd.DataFrame(columns=
        ['problem_type', 'compute search dir', 'fatrop', 'initialization', 
         'eval objective', 'eval gradient', 'eval constraint violation',
          'eval jacobian', 'eval hessian', 'function evaluation','rest time', 
          'total'])
    df_func_eval_quadruped_breakdown = pd.DataFrame(columns=
        ['problem_type', 'compute search dir', 'fatrop', 'initialization', 
         'eval objective', 'eval gradient', 'eval constraint violation',
          'eval jacobian', 'eval hessian', 'function evaluation','rest time', 
          'total'])
    df_func_eval_batch_reactor_breakdown = pd.DataFrame(columns=
        ['problem_type', 'compute search dir', 'fatrop', 'initialization', 
         'eval objective', 'eval gradient', 'eval constraint violation',
          'eval jacobian', 'eval hessian', 'function evaluation','rest time', 
          'total'])
    df_func_eval_solar_receiver_reactor_breakdown = pd.DataFrame(columns=
        ['problem_type', 'compute search dir', 'fatrop', 'initialization', 
         'eval objective', 'eval gradient', 'eval constraint violation',
          'eval jacobian', 'eval hessian', 'function evaluation','rest time', 
          'total'])

    # get func eval data for each problem type (holonomic, truck trailer, planar robot, quadruped)
    for data in all_jsons:
        if data["generator_data"]["problem_name"] not in ["holonomic", "truck_trailer", "planar_robot", "quadruped", "batch_reactor", "solar_receiver_reactor"]:
            continue
        timing_stats = data["metadata"]["timing_statistics"]
        nb_iter = data["metadata"]['iterations']
        entry = {
            'problem_type': data['problem type'],
            'compute search dir': data["metadata"]["timing_statistics"].get('compute search dir', 0)/data["metadata"]['iterations'],
            'fatrop': data["metadata"]["timing_statistics"].get('fatrop', 0)/data["metadata"]['iterations'],
            'initialization': data["metadata"]["timing_statistics"].get('initialization', 0),
            'eval objective': data["metadata"]["timing_statistics"].get('eval objective', 0)/data["metadata"]['iterations'],
            'eval gradient': data["metadata"]["timing_statistics"].get('eval gradient', 0)/data["metadata"]['iterations'],
            'eval constraint violation': data["metadata"]["timing_statistics"].get('eval constraint violation', 0)/data["metadata"]['iterations'],
            'eval jacobian': data["metadata"]["timing_statistics"].get('eval jacobian', 0)/data["metadata"]['iterations'],
            'eval hessian': data["metadata"]["timing_statistics"].get('eval hessian', 0)/data["metadata"]['iterations'],
            'rest time': data["metadata"]["timing_statistics"].get('rest time', 0),
            'function evaluation': data["metadata"]["timing_statistics"].get('function evaluation', 0)/data["metadata"]['iterations'],
            'total': data["metadata"]["timing_statistics"].get('total', 0)/data["metadata"]['iterations']
        }
        if data["generator_data"]["problem_name"] == "holonomic":
            # check if we find the other two problem types with same dimension and control level
            dim = data["generator_data"]["n"]
            ctl = data["generator_data"]["control_level"]
            df_other = df_holonomic[(df_holonomic['dimension'] == dim) & (df_holonomic['control level'] == ctl)]
            if df_other.shape[0] < 3:
                continue
            df_func_eval_holonomic_breakdown.loc[len(df_func_eval_holonomic_breakdown)] = entry
        elif data["generator_data"]["problem_name"] == "truck_trailer":
            n = data["generator_data"]["n"]
            df_other = df_trucktrailer[df_trucktrailer['n'] == n]
            if df_other.shape[0] < 3:
                continue
            df_func_eval_truck_trailer_breakdown.loc[len(df_func_eval_truck_trailer_breakdown)] = entry
                
        elif data["generator_data"]["problem_name"] == "planar_robot":
            n = data["generator_data"]["n"]
            df_other = df_planarrobot[df_planarrobot['n'] == n]
            if df_other.shape[0] < 3:
                continue
            df_func_eval_planar_robot_breakdown.loc[len(df_func_eval_planar_robot_breakdown)] = entry

        elif data["generator_data"]["problem_name"] == "quadruped":
            df_func_eval_quadruped_breakdown.loc[len(df_func_eval_quadruped_breakdown)] = entry
            continue

        elif data["generator_data"]["problem_name"] == "batch_reactor":
            df_func_eval_batch_reactor_breakdown.loc[len(df_func_eval_batch_reactor_breakdown)] = entry
            continue

        elif data["generator_data"]["problem_name"] == "solar_receiver_reactor":
            df_func_eval_solar_receiver_reactor_breakdown.loc[len(df_func_eval_solar_receiver_reactor_breakdown)] = entry
            continue

        df_func_eval_breakdown.loc[len(df_func_eval_breakdown)] = entry
    
    # visualize_func_eval_breakdown(df_func_eval_breakdown)
    # visualize_func_eval_breakdown(df_func_eval_holonomic_breakdown, fig_name_appendix="holonomic")
    # visualize_func_eval_breakdown(df_func_eval_truck_trailer_breakdown, fig_name_appendix="truck_trailer")
    # visualize_func_eval_breakdown(df_func_eval_planar_robot_breakdown, fig_name_appendix="planar_robot")
    # visualize_func_eval_breakdown(df_func_eval_quadruped_breakdown, fig_name_appendix="quadruped")
    # visualize_func_eval_breakdown(df_func_eval_batch_reactor_breakdown, fig_name_appendix="batch_reactor")
    # visualize_func_eval_breakdown(df_func_eval_solar_receiver_reactor_breakdown, fig_name_appendix="solar_receiver_reactor")

    if (not df_holonomic.empty):
        visualize_performance(df_holonomic)
        print_performance_table(df_holonomic)
        print_holonomic_result_differences(df_holonomic)

    if (not df_trucktrailer.empty):
        visualize_performance(df_trucktrailer)
        print_performance_table(df_trucktrailer)
        print_trucktrailer_result_differences(df_trucktrailer)

    if (not df_planarrobot.empty):
        visualize_performance(df_planarrobot)
        print_performance_table(df_planarrobot)
        # print_trucktrailer_result_differences(df_planarrobot)

    if (not df_quadruped.empty):
        visualize_performance_quadruped(df_quadruped)
        print_performance_table(df_quadruped)

    if (not df_batch_reactor.empty):
        visualize_performance(df_batch_reactor)
        print_performance_table(df_batch_reactor)

    if (not df_solar_receiver_reactor.empty):
        visualize_performance(df_solar_receiver_reactor)
        print_performance_table(df_solar_receiver_reactor)

    visualize_performance_different_problems([df_trucktrailer, df_quadruped, df_batch_reactor, df_solar_receiver_reactor])
