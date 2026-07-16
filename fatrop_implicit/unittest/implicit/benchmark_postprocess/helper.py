import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def translate_label(label):
    if label == 'K':
        return 'K'
    elif label == 'nx':
        return r'$n_x$'
    elif label == 'nu':
        return r'$n_u$'
    elif label == 'ng':
        return r'$n_g$'
    elif label == 'r':
        return r'$r$'
    elif label == 'ng_ineq':
        return r'$n_{g,ineq}$'
    else:
        raise ValueError(f'Unknown label: {label}')
    

def latexify():
    params = {#'backend': 'ps',
              'axes.labelsize': 18,
              'axes.titlesize': 15,
              'legend.fontsize': 8,
              'xtick.labelsize': 11,
              'ytick.labelsize': 11,
              'text.usetex': True,
              'font.family': 'serif',
              'figure.figsize': [6, 4],
              'text.latex.preamble': r'\usepackage{bm}',
              }
 
    plt.rcParams.update(params)
    
def get_data():   
    use_generalization = True
    if use_generalization:
        # file = '../../../build_docker/random_benchmark_results_generalized_20000.csv'
        # file = '../../../build_docker/random_benchmark_results_generalized_faked.csv'
        file = '../../../build_docker/random_benchmark_results_generalized_20000_standalone.csv'
    else:
        file = '../../../build_docker/random_benchmark_results_extended_20000.csv'
        
    df = pd.read_csv(file)
    
    if not use_generalization:
        df['nz'] = df['nx']
        df['nf'] = df['nx']
        df['nv'] = df['nx']
    else:
        df['nf'] = df['nv'] + (df['nx'] - df['nz'])
    
    # number of rows: ng + nv
    df['m'] = df['ng'] + df['nv']
    df['m_rel'] = df['ng'] / df['m']
    
    # number of cols: nu + nz
    df['n'] = df['nz'] + df['nu']
    df['n_rel'] = df['nu'] / df['n']

    print("Loaded data file")
    return df

def get_lu_data():
    file = '../../../build_docker/blocked_lu_timings_general.csv'
    df = pd.read_csv(file)
    df['nz'] = df['nx']
    df['nv'] = df['nx']
    df['nf'] = df['nv'] + (df['nx'] - df['nz'])
    df['m'] = df['ng'] + df['nv']
    df['m_rel'] = df['ng'] / df['m']
    df['n'] = df['nz'] + df['nu']
    df['n_rel'] = df['nu'] / df['n']
    print("Loaded LU data file")
    return df