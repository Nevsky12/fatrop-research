import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data():
    df = pd.read_csv('build_docker/lu_fact_complexity.csv')
    
    data = {
        'm': np.array(df['m']),
        'n': np.array(df['n']),
        't': np.array(df['time_ns']),
    }
    return data

def visualize_complexity(data):
    m = data['m']
    n = data['n']
    t = data['t']

    plt.figure()
    plt.scatter(m, n, c=t, cmap='coolwarm', marker='o')
    plt.colorbar(label='Time (seconds)')
    plt.xlabel('m')
    plt.ylabel('n')
    plt.title('LU Factorization Complexity')

    # expected complexity
    plt.figure()
    t_expected = m*n*n
    plt.scatter(m, n, c=t_expected, cmap='coolwarm', marker='o')
    plt.colorbar(label='Expected Time (seconds)')
    plt.xlabel('m')
    plt.ylabel('n')
    plt.title('Expected LU Factorization Complexity (m*n^2)')

    plt.show()

visualize_complexity(get_data())
