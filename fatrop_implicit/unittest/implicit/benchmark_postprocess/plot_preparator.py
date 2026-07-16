from metric_computer import rel_speedup_computer
from filter import default_filter
import numpy as np

class PlotPreparator:
    def __init__(self):
        self.x_metric_computer = None
        self.y_metric_computer = rel_speedup_computer() # default
        self.filter = default_filter()
        
    def set_x_metric_computer(self, x_metric_computer):
        self.x_metric_computer = x_metric_computer
        
    def set_y_metric_computer(self, y_metric_computer):
        self.y_metric_computer = y_metric_computer
        
    def set_filter(self, filter):
        self.filter = filter

    def prepare(self, df, **kwargs):
        assert self.x_metric_computer is not None, "x_metric_computer must be set before calling prepare()"
        assert self.y_metric_computer is not None, "y_metric_computer must be set before calling prepare()"
        assert self.filter is not None, "filter must be set before calling prepare()"
        
        # get all x-values
        x_values = self.x_metric_computer.compute_unique_sorted_values(df)
        if kwargs.get('use_integer_x', False):
            x_values = [int(x) for x in x_values]
            x_values = np.unique(x_values)
        elif kwargs.get('x_min_step') is not None:
            x_min = np.min(x_values)
            x_max = np.max(x_values)
            x_min_step = kwargs.get('x_min_step')
            x_values = np.arange(x_min, x_max + x_min_step, x_min_step)
        
        # for each x value, compute the mean and std of the corresponding y values
        y_values_mean = []
        y_values_std = []
        y_all = []
        y_masks = []
        df_filtered, _ = self.filter.filter_data(df)
        for i, x in enumerate(x_values):
            if kwargs.get('use_integer_x') or kwargs.get('x_min_step') is not None:
                df_filtered_value, mask = self.x_metric_computer.filter_df_range(df_filtered, x, x_values[i+1] if i+1 < len(x_values) else x)
            else:
                df_filtered_value, mask = self.x_metric_computer.filter_df(df_filtered, x)
                
            y_mean, y_std, y_all_i = self.y_metric_computer.evaluate(df_filtered_value)
            y_values_mean.append(y_mean)
            y_values_std.append(y_std)
            y_all.append(list(y_all_i))
            y_masks.append(mask)

        return x_values, y_values_mean, y_values_std, y_all, y_masks