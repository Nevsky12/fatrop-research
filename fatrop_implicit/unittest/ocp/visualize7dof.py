import matplotlib.pyplot as plt
import numpy as np


# PASTE COMMAND LINE OUTPUT HERE
#=====================================================
avg_total_implicit = [29.404, 66.722, 138.704, 218.03, 378.442]
avg_total_reformulation = [27.21, 70.808, 171.852, 314.506, 551.398]
avg_preprocessing_jac = [3.042, 10.29, 27.164, 41.15, 79.282]
avg_preprocessing_hess = [6.14, 16.942, 33.56, 58.892, 103.898]
avg_solve = [14.306, 28.306, 56.084, 85.822, 144.314]
avg_postprocess = [3.538, 8.41, 18.38, 28.278, 46.446]
#=====================================================

components = {
    'Preprocess (Jac)': avg_preprocessing_jac,
    'Preprocess (Hess)': avg_preprocessing_hess,
    'Solve': avg_solve,
    'Postprocess': avg_postprocess,
    'Other': list(np.array(avg_total_implicit) - np.array(avg_preprocessing_jac) - np.array(avg_preprocessing_hess) - np.array(avg_solve) - np.array(avg_postprocess))
}
component_colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Calculate absolute times for stacked bar
implicit_times = [c for c in components.values()]
component_labels = list(components.keys())


# Plotting
fig, ax = plt.subplots()

# Bar positions
x = [i for i in range(len(avg_total_implicit))]
bar_width = 0.2
bar_offset = bar_width/2

# Plot stacked bar for implicit
for i in range(len(avg_total_implicit)):
    bottom = 0
    for time, label, c in zip(implicit_times, component_labels, component_colours):
        ax.bar(x[i]-bar_offset, time[i], bottom=bottom, label=label if i == 0 else None, width=bar_width, color=c)
        bottom += time[i]

    # Plot single bar for reformulation
    ax.bar(x[i]+bar_offset, avg_total_reformulation[i], color='gray', label='Reformulation' if i == 0 else None, width=bar_width)


# Formatting
ax.set_xticks(x)
ax.set_xticklabels([7*i for i in range(1, len(avg_total_implicit)+1)])
ax.set_xlabel('nx')
ax.set_ylabel('Time per instance')
ax.set_title('Time Comparison: Implicit vs Reformulation')
ax.legend(loc='upper left')

plt.tight_layout()
plt.show()