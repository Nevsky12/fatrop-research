import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from helper import *
from metric_computer import *
from plot_preparator import *
from filter import *
from visualizer import *
settings = json.load(open('../post_process_settings.json', 'r'))


if __name__ == '__main__':
    if settings.get('latexify', True):
        latexify()
        
    # load data
    df = pd.DataFrame(get_data())
    df_lu = pd.DataFrame(get_lu_data())
            
    # define filters
    areas = np.linspace(0, max(df['m']*df['n']), 7)
    max_sqrt_areas = np.linspace(0, np.sqrt(max(df['m']*df['n'])), 7)
    
    max_sqrt_to_consider = 60
    nb_filters = 6
    filter_sqrt = [metric_range_filter(square_root_area_computer(), 
                                       max_sqrt_to_consider*i/nb_filters, 
                                       max_sqrt_to_consider*(i+1)/nb_filters, 
                                       integer=True)
                   for i in range(nb_filters)]
    
    # define x-axis functions
    x_funcs = [area_computer(relative=True)]
    y_funcs_abs = [plain_df_key('t_accel'), plain_df_key('t_reform')]
    y_func_lu_rel = rel_speedup_lu_computer(recursion_benchmark_data=True)
    
    # define colors for filters
    first_color = [0.7, 0.7, 0.7]
    last_color = [0.0, 0.0, 0.0]
    colors = [tuple(np.array(first_color) + (np.array(last_color) - np.array(first_color)) * i / (len(filter_sqrt)-1)) for i in range(len(filter_sqrt))]
        
    # relative difference in t_total for different sizes
    visualize_scaling_1d(df, x_funcs[0], filters=filter_sqrt, x_min_step=0.05, filter_colors=colors,
                         file_name=f'relative_improvement_relative_area_binned_sqrt_area.png', show_std=False, y_lim=(-0.3, 0.3))
    
    # relative difference in t_lu for different sizes
    visualize_scaling_1d(df, x_funcs[0], y_funcs=[y_func_lu_rel], filters=filter_sqrt, x_min_step=0.05, filter_colors=colors,
                         file_name=f'relative_improvement_relative_area_binned_sqrt_area_lu.png', show_std=False)
    
    # absolute t_lu
    visualize_scaling_1d(df, x_funcs[0], y_funcs=[plain_df_key('lu_accel'), plain_df_key('lu_reform')], x_min_step=0.05, 
                         filters=filter_sqrt, filter_colors=colors,
                         file_name=f'lu_relative_area_binned_sqrt_area.png', show_std=False)
    
    # absolute t_total
    visualize_scaling_1d(df, x_funcs[0], y_funcs=[plain_df_key('t_accel'), plain_df_key('t_reform')], x_min_step=0.05, 
                         filters=filter_sqrt, filter_colors=colors,
                         file_name=f't_total_relative_area_binned_sqrt_area.png', show_std=False)
    
    # lu relevanve for different sizes
    visualize_scaling_1d(df, x_funcs[0], y_funcs=[lu_relevance_computer(reformulated=True)], filters=filter_sqrt, x_min_step=0.05, filter_colors=colors, file_name=f'lu_relevance_relative_area_binned_sqrt_area.png', show_std=False)

    # realtive difference in t_other for different sizes
    visualize_scaling_1d(df, x_funcs[0], y_funcs=[relative_difference_other_time()], filters=filter_sqrt, x_min_step=0.05, filter_colors=colors,
                         file_name=f'relative_improvement_other_relative_area_binned_sqrt_area_lu.png', show_std=False)
    
    # absolute area
    visualize_scaling_1d(df, area_computer(relative=True), y_funcs=[area_computer(relative=False)], x_min_step=0.05, filter_colors=colors,
                         file_name=f'total_area_relative_area.png', show_std=False)
        
    x_funcs = [plain_df_key('nx'), plain_df_key('nu'), plain_df_key('ng')]
    for x_func in x_funcs:
        # visualize_scaling_1d(df, x_func)
        visualize_scaling_1d(df, x_func, y_funcs=[rel_speedup_lu_computer(recursion_benchmark_data=True)], title='Relative Speedup of LU vs Full Factorization', file_name=f'relative_speedup_lu_{x_func.name}_recursion.png')
    
    visualize_distribution(df, lu_relevance_computer(reformulated=True), bins=40, normalized=True, color='grey', show_median=True, x_lim=(0, 1), file_name='lu_relevance_distribution.png')
        

