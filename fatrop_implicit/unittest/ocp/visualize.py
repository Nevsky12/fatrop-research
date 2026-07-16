import matplotlib.pyplot as plt

# PASTE COMMAND LINE OUTPUT HERE
#=====================================================
N = 500
preprocess_jac_rel = 0.154607
preprocess_hess_rel = 0.222358
solve_rel = 0.502208
postprocess_rel = 0.108901
other_rel = 0.0119255

total_time_implicit = 178860
total_time_reformulation = 257326
#=====================================================


# Input data
total_time_implicit = total_time_implicit / N
total_time_reformulation = total_time_reformulation / N

# Relative time components for implicit method
components = {
    'Preprocess (Jac)': preprocess_jac_rel,
    'Preprocess (Hess)': preprocess_hess_rel,
    'Solve': solve_rel,
    'Postprocess': postprocess_rel,
    'Other': other_rel
}

# Calculate absolute times for stacked bar
implicit_times = [total_time_implicit * rel for rel in components.values()]
component_labels = list(components.keys())

# Plotting
fig, ax = plt.subplots(figsize=(6, 6))

# Bar positions
x = [0, 1]  # 0 = implicit, 1 = reformulation

# Plot stacked bar for implicit
bottom = 0
for time, label in zip(implicit_times, component_labels):
    ax.bar(x[0], time, bottom=bottom, label=label)
    bottom += time

# Plot single bar for reformulation
ax.bar(x[1], total_time_reformulation, color='gray', label='Reformulation')

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(['Implicit', 'Reformulation'])
ax.set_ylabel('Time per instance')
ax.set_title('Time Comparison: Implicit vs Reformulation')
ax.legend(loc='center right')

plt.tight_layout()
plt.show()