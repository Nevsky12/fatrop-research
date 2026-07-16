import json
from helper import *
from metric_computer import *
from flop_data_generator import FlopDataGenerator
from filter import *
from visualizer import visualize_distribution, visualize_scaling_1d

settings = json.load(open('../post_process_settings.json', 'r'))

if __name__ == '__main__':
    if settings.get('latexify', True):
        latexify()
        
    fdg = FlopDataGenerator(nb_samples=10000)
    df = fdg.flop_data
    
    # define filters
    filter_sqrt = [plain_df_key_filter('total_size_sqrt', 10*i, 10*(i+1)) for i in range(6)]
    
    # define axis functions
    x_func = plain_df_key('relative_size')
    y_func = plain_df_key('rel_diff_flops')
    
    
    first_color = [0.7, 0.7, 0.7]
    last_color = [0.0, 0.0, 0.0]
    colors = [tuple(np.array(first_color) + (np.array(last_color) - np.array(first_color)) * i / (len(filter_sqrt)-1)) for i in range(len(filter_sqrt))]
    
    visualize_distribution(df, x_func, bins=100, normalized=True, color='grey', 
                           x_lim=(0,1), 
                           file_name='flop_relative_size_distribution.png')
    visualize_distribution(df, y_func, bins=100, normalized=True, color='grey', 
                           file_name='flop_rel_diff_flops_distribution.png')
    
    visualize_scaling_1d(df, x_func, y_funcs=[y_func], filters=filter_sqrt, 
                         x_min_step=0.05,
                         filter_colors=colors, 
                         title='Relative Difference in FLOPs vs Relative Size',
                         file_name=f'flop_rel_diff_relative_size_binned_sqrt_total_size.png', 
                         show_std=False)#, y_lim=(-0.3, 0.3))
    
    visualize_scaling_1d(df, x_func, 
                         y_funcs=[
                             plain_df_key_scaled_offset('flops_normal', offset=0, scale=1), 
                             plain_df_key_scaled_offset('flops_structure', offset=0, scale=1)], 
                         x_min_step=0.05, file_name=f'flop_flops_relative_size_binned.png', show_std=False)
    
    visualize_scaling_1d(df, x_func, 
                         y_funcs=[
                            flops_per_size('flops_normal'), 
                            flops_per_size('flops_structure')], 
                         x_min_step=0.05, file_name=f'flop_flops_per_size_relative_size_binned.png', show_std=False)
    
    visualize_scaling_1d(df, x_func, 
                         y_funcs=[flops_per_size('rho')], 
                         x_min_step=0.05, file_name=f'flop_rho_per_size_relative_size_binned.png', show_std=False)
    
    visualize_scaling_1d(df, x_func, 
                         y_funcs=[plain_df_key('total_size')], 
                         x_min_step=0.05, file_name=f'flop_total_size_relative_size_binned.png', show_std=False)