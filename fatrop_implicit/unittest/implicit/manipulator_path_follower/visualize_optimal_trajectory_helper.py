import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from visualize_robot_beam import *

def interpolate(t_knots, t_query, N, var):
    idx   = np.searchsorted(t_knots, t_query, side="right") - 1
    idx   = np.clip(idx, 0, N - 1)
    idx   = np.clip(idx, 0, var.shape[1]-2)
    t0, t1 = t_knots[idx], t_knots[idx + 1]
    alpha  = (t_query - t0) / (t1 - t0) if (t1 > t0) else 0.0
    alpha  = float(np.clip(alpha, 0.0, 1.0))
    return (1 - alpha)*var[:,idx] + alpha*var[:,idx+1]

def get_desired_endpoint_positions(t_uniform, uniform_ocp_vars, path_func):
    desired_positions = np.zeros((3, len(t_uniform)))
    for i, t in enumerate(t_uniform):
        desired_positions[:, i] = np.array(path_func(uniform_ocp_vars['p'][:,i])).flatten()
    return desired_positions

def get_actual_endpoint_positions(t_uniform, uniform_ocp_vars, scenario, end_pos_func, L=None):
    actual_tip = np.zeros((3, len(t_uniform)))
    for i, t in enumerate(t_uniform):
        if scenario == "contact":
            actual_tip[:, i] = np.array(end_pos_func(uniform_ocp_vars['q'][:,i])).flatten()  # th=0 for contact scenario
        else:
            actual_tip[:, i] = np.array(end_pos_func(uniform_ocp_vars['q'][:,i], uniform_ocp_vars['th_uniform'][:,i], L)).flatten()
    return actual_tip

def get_uniform_ocp_vars(N, t_knots, n_frames, sol, T_total, scenario, FOLDER):
    t_uniform = np.linspace(0.0, T_total, n_frames)
    uniform_ocp_vars = {
        'p':np.zeros((1, n_frames)),
        'dp':np.zeros((1, n_frames)),
        'q':np.zeros((7, n_frames)),
        'tau':np.zeros((7, n_frames))
    }

    sol_p = np.array(sol["p"]).reshape(1, N+1)
    sol_dp = np.array(sol["dp"]).reshape(1, N)
    sol_q = np.array(sol["q"]).reshape(7, N+1)
    sol_tau = np.array(sol["tau"]).reshape(7, N)

    if scenario == "beam":
        sol_th = np.array(sol["th"]).reshape(1, N+1)
        uniform_ocp_vars['th_uniform'] = np.zeros((1, n_frames))

    elif scenario == "contact":
        sol_qd = np.array(sol["qd"]).reshape(7, N+1)
        uniform_ocp_vars['qd'] = np.zeros((7, n_frames))
        contact_force_func = ca.Function.load(f"{FOLDER}panda_normal_force.casadi")
        uniform_ocp_vars['force'] = np.zeros((1, n_frames))

    for i, t in enumerate(t_uniform):
        uniform_ocp_vars['p'][:,i] = interpolate(t_knots, t, N, sol_p)
        uniform_ocp_vars['q'][:,i] = interpolate(t_knots, t, N, sol_q)
        if i < len(t_uniform) - 1:
            uniform_ocp_vars['dp'][:,i] = interpolate(t_knots, t, N, sol_dp)
            uniform_ocp_vars['tau'][:,i] = interpolate(t_knots, t, N, sol_tau)
        if scenario == "beam":
            uniform_ocp_vars['th_uniform'][:,i] = interpolate(t_knots, t, N, sol_th)
        elif scenario == "contact":
            uniform_ocp_vars['qd'][:,i] = interpolate(t_knots, t, N, sol_qd)
            uniform_ocp_vars['force'][:,i] = contact_force_func(
                uniform_ocp_vars['q'][:,i],
                uniform_ocp_vars['qd'][:,i],
                sol["floor_height"], sol['k'], sol['alpha'], sol['d']
            )
        
    return t_uniform, uniform_ocp_vars

def update(frame, t_uniform, uniform_ocp_vars, desired_positions, actual_tip, sol, scenario):
    ax = plt.gca()
    ax.cla()

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")

    # 3. Robot at interpolated configuration
    t_now = t_uniform[frame]
    if scenario == "beam":
        show_robot_with_beam(ax, uniform_ocp_vars['q'][:,frame], uniform_ocp_vars['th_uniform'][:,frame], L)
    else:
        show_robot_with_surface(ax, uniform_ocp_vars['q'][:,frame], sol["floor_height"])


    # 1. Full desired path (static)
    ax.plot(
        desired_positions[0], desired_positions[1], desired_positions[2],
        "g--", linewidth=1.5, label="Desired path", zorder=2,
    )

    # 2. Beam-tip trace up to current frame
    ax.plot(
        actual_tip[0, :frame + 1],
        actual_tip[1, :frame + 1],
        actual_tip[2, :frame + 1],
        "b-", linewidth=1.5, label="Beam tip trace", zorder=3,
    )

    # 4. Time stamp
    n_frames = len(t_uniform)
    ax.text2D(0.02, 0.95, f"t = {t_now:.2f} s  (frame {frame}/{n_frames - 1})",
              transform=ax.transAxes, fontsize=10)
    if scenario == "beam":
        # show theta value as well
        thk_degree = np.degrees(thk)
        ax.text2D(0.02, 0.90, f"theta = {thk_degree:.2f} deg",
              transform=ax.transAxes, fontsize=10)
    else:
        force = uniform_ocp_vars['force'][:,frame]
        ax.text2D(0.02, 0.90, f"contact force = {float(force[0]):.2f} N",
              transform=ax.transAxes, fontsize=10)
        # z_ax_dot_product = ca.mtimes(ca.transpose(ca.mtimes(ee_rot_func(qk), ca.vertcat(0, 0, 1))), ca.vertcat(0, 0, -1))
        # ax.text2D(0.02, 0.85, f"dot")

    ax.legend(loc="upper right", fontsize=8)

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])