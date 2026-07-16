import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import casadi as ca
from visualize_optimal_trajectory_helper import *

from visualize_robot_beam import show_robot_with_beam, show_robot_with_surface

### Configuration
scenario = "contact"
SOLUTION_FILE  = f"manipulator_path_follower_{scenario}_solution.json"
PATH_FUNC_FILE = f"path_func_{scenario}.casadi"
L              = 0.3
FOLDER         = "panda/"
FPS            = 25          # animation frames per second
PLAYBACK_SPEED = 1.0         # set < 1 to slow down, > 1 to speed up


### Load solution and necessary functions
with open(SOLUTION_FILE, "r") as f:
    sol = json.load(f)
path_func = ca.Function.load(PATH_FUNC_FILE)
if scenario == "contact":
    ee_rot_func = ca.Function.load(f"{FOLDER}panda_fkrot_ee.casadi")
    end_pos_func = ca.Function.load(f"{FOLDER}panda_fkpos_ee.casadi")
elif scenario == "beam":
    end_pos_func = ca.Function.load(f"{FOLDER}panda_beam_end_pos.casadi")


### Build the real-time knot vector from dp
dp  = np.array(sol["dp"]) if sol["dp"] is not None else T/N*np.ones([1, N])
dp_flat = dp.flatten()                     # length N
t_knots = np.concatenate([[0.0], np.cumsum(dp_flat)])   # length N+1
T_total = t_knots[-1]                      # should equal T
n_frames   = max(2, int(np.ceil(T_total * FPS / PLAYBACK_SPEED)))
t_uniform, uniform_ocp_vars = get_uniform_ocp_vars(sol['N'], t_knots, n_frames, sol, T_total, scenario, FOLDER)
desired_positions = get_desired_endpoint_positions(t_uniform, uniform_ocp_vars, path_func)
actual_tip = get_actual_endpoint_positions(t_uniform, uniform_ocp_vars, scenario, end_pos_func, L)



### True visualisations ###
marker = 'o'
markersize = 2.5

# plot joint positions
fig = plt.figure()
plt.plot(t_uniform, uniform_ocp_vars['q'].T, marker=marker, markersize=markersize)
plt.xlim(min(t_uniform), max(t_uniform))
plt.xlabel("t (s)")
plt.ylabel("joint positions (rad)")
plt.savefig("visualization_output/joint_positions.png", dpi=300)

# plot joint velocities
fig = plt.figure()
plt.plot(t_uniform, uniform_ocp_vars['qd'].T, marker=marker, markersize=markersize)
plt.xlim(min(t_uniform), max(t_uniform))
plt.xlabel("t (s)")
plt.ylabel("joint velocities (rad/s)")
plt.savefig("visualization_output/joint_velocities.png", dpi=300)

# plot torques
fig = plt.figure()
plt.plot(t_uniform, uniform_ocp_vars['tau'].T, marker=marker, markersize=markersize)
plt.xlim(min(t_uniform), max(t_uniform))
plt.xlabel("t (s)")
plt.ylabel("joint torques (Nm)")
plt.savefig("visualization_output/torques.png", dpi=300)

# plot normal force
if scenario == "contact":
    fig = plt.figure()
    plt.plot(t_uniform, uniform_ocp_vars['force'].T, marker=marker, markersize=markersize)
    plt.xlim(min(t_uniform), max(t_uniform))
    plt.xlabel("t (s)")
    plt.ylabel("Contact normal force (N)")
    plt.savefig("visualization_output/normal_force.png", dpi=300)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(desired_positions[0], desired_positions[1], desired_positions[2], 'g--', linewidth=1.5, label="Desired path")
    xx = actual_tip[0]; yy = actual_tip[1]; zz = actual_tip[2]
    ax.plot(xx, yy, zz)
    force = uniform_ocp_vars['force'] / 100
    ax.plot(xx, yy, zz + force)
    width = 0.
    X = np.vstack([xx-width, xx+width])
    Y = np.vstack([yy-width, yy+width])
    Z = np.vstack([zz, zz + force])
    ax.plot_surface(X, Y, Z, alpha=0.5)
    set_axes_equal(ax)
    plt.savefig("visualization_output/normal_force_3d.png", dpi=300)

# plot progress variable
fig = plt.figure()
plt.plot(t_uniform, uniform_ocp_vars['dp'].T)
plt.savefig("visualization_output/progress_variable", dpi=300)

# Animation
plt.close()
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection="3d")
interval_ms = int(1000 / FPS)
ani = animation.FuncAnimation(
    fig,
    update,
    frames=n_frames,
    interval=interval_ms,
    blit=False,
    repeat=True,
    fargs=(t_uniform, uniform_ocp_vars, desired_positions, actual_tip, sol, scenario)
)

plt.tight_layout()
ani.save("visualization_output/path_follower_animation.mp4", writer="ffmpeg", fps=FPS, dpi=150)
print("saved animation")
exit()