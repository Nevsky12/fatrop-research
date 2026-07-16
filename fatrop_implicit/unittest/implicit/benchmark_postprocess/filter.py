import numpy as np

class Filter:
    def __init__(self, name):
        self.name = name       
        
    # returns true or false for each row in the dataframe, indicating whether the row should be included in the plot
    def filter_pass_condition(self, df):
        raise NotImplementedError
    
    # return a filtered data frame and the indices of the rows that pass the filter
    def filter_data(self, df):
        mask = self.filter_pass_condition(df)
        return df[mask], mask
    
    
class default_filter(Filter):
    def __init__(self):
        super().__init__('Default Filter')
        
    def filter_pass_condition(self, df):
        # return true for all rows of df
        return np.ones(len(df), dtype=bool)
    
class size_filter(Filter):
    def __init__(self, area_min, area_max):
        super().__init__(f'{area_min} $<$ area $<$ {area_max}')
        self.area_min = area_min
        self.area_max = area_max
        
    def filter_pass_condition(self, df):
        area = df['m'] * df['n']
        return (area >= self.area_min) & (area <= self.area_max)
    
class plain_df_key_filter(Filter):
    def __init__(self, key, value_min, value_max):
        super().__init__(f'{value_min:.2f} $<$ {key} $<$ {value_max:.2f}')
        self.key = key
        self.value_min = value_min
        self.value_max = value_max

    def filter_pass_condition(self, df):
        return (df[self.key] >= self.value_min) & (df[self.key] <= self.value_max)
    
class lu_relevance_filter(Filter):
    def __init__(self, threshold_min, threshold_max, reformulated=False):
        super().__init__(f'LU relevance $>$ {threshold_min:.2f} and $<$ {threshold_max:.2f}')
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.reformulated = reformulated

    def filter_pass_condition(self, df):
        if self.reformulated:
            relevance = df['lu_reform'] / df['t_reform']
        else:
            relevance = df['lu_accel'] / df['t_accel']
        return (relevance >= self.threshold_min) & (relevance <= self.threshold_max)
    
class metric_range_filter(Filter):
    def __init__(self, metric_computer, value_min, value_max, integer=False):
        super().__init__(
            f'{value_min:.2f} $<$ {metric_computer.name} $<$ {value_max:.2f}' if not integer else
            f'{int(value_min)} $<$ {metric_computer.name} $<$ {int(value_max)}'
        )
        self.metric_computer = metric_computer
        self.value_min = value_min
        self.value_max = value_max

    def filter_pass_condition(self, df):
        metric_values = self.metric_computer.compute_metric(df)
        return (metric_values >= self.value_min) & (metric_values <= self.value_max)