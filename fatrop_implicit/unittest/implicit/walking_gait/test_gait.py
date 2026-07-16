"""
Walking Gait Optimization using Direct Collocation

Dependencies:
    - CasADi 3.7.1 or later
    - NumPy
    - Matplotlib

This script sets up and solves an optimal control problem (OCP) for simulating
human walking using direct collocation with Radau or Legendre collocation schemes.

The model can be either torque-driven (6 DOF) or muscle-driven (9 DOF).

Original MATLAB author: Lars D'Hondt (based on PredSim framework)
Converted to Python: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import casadi
from collocation_scheme import collocation_scheme
import os
import warnings

print(casadi.__version__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Choose solver: 'fatrop' or 'ipopt'
# solver = 'ipopt'  # Change to 'fatrop' if available
solver = 'fatrop'

# Mesh intervals
N_mesh = 64

# Order of Radau collocation scheme
n_coll = 3

# Musculoskeletal model parameters
# Number of coordinates:
#   6: pelvis_tx, pelvis_ty, hip_r, hip_l, knee_r, knee_l
#   9: pelvis_tilt, pelvis_tx, pelvis_ty, hip_r, hip_l, knee_r, knee_l,
#       ankle_r, ankle_l
n_coords = 9

# Leg joints are driven by ideal torque actuators.
# The model with 9 coordinates can also be muscle-driven.
torque_driven = False

# =============================================================================
# SETUP
# =============================================================================

# Impose walking at 1.2 m/s with 0.9 strides/s (average adult)
step_time = 1.0 / 0.9 / 2  # time horizon of 1 step
step_length = 1.2 * step_time  # forward distance covered in 1 step

# Timestep
dt = step_time / N_mesh

# Load function to evaluate system dynamics
# Input:
#   - coordinate positions
#   - coordinate velocities
#   - coordinate accelerations
#   - actuation controls
# Output:
#   - effort (sum of squared controls)
#   - limit torques (sum of squared) - range of motion via load in ligaments
#   - moment equilibrium error on each coordinate
#
if n_coords == 6:
    idx_q_fwd = 0  # index of forward position coordinate (0-indexed Python)
    n_act = 4
    casadi_func_file = 'f_sysdyn_6dof_0mus_12states.casadi'
elif n_coords == 9:
    idx_q_fwd = 1  # index of forward position coordinate (0-indexed Python)
    if torque_driven:
        n_act = 6
        casadi_func_file = 'f_sysdyn_9dof_0mus_18states.casadi'
    else:
        n_act = 18
        casadi_func_file = 'f_sysdyn_9dof_18mus_18states.casadi'

# Check if the casadi function file exists
if os.path.exists(casadi_func_file):
    f_sysdyn = casadi.Function.load(casadi_func_file)
    f_sysdyn.save("temp.casadi")
else:
    print(f"Warning: {casadi_func_file} not found. Skipping dynamics loading.")
    f_sysdyn = None

# Scale factors for optimisation variables
scale_qdots = 10  # velocities
scale_qddots = 100  # accelerations

# =============================================================================
# HELPER FUNCTIONS AND VARIABLES
# =============================================================================

# States on 1st mesh point
q_mesh_1_SX = casadi.SX.sym('q_mesh_1', n_coords - 1, 1)  # except forward position
qdot_mesh_1_SX = casadi.SX.sym('qdot_mesh_1', n_coords, 1)

# States on kth mesh point
q_mesh_k_SX = casadi.SX.sym('q_mesh_k', n_coords, 1)
qdot_mesh_k_SX = casadi.SX.sym('qdot_mesh_k', n_coords, 1)

if solver == 'fatrop':
    # State vector is augmented with states on 1st mesh point. This is
    # needed to formulate periodicity within the OCP format of fatrop.
    x_mesh_k_SX = casadi.vertcat(q_mesh_k_SX, qdot_mesh_k_SX,
                                  q_mesh_1_SX, qdot_mesh_1_SX)
else:
    x_mesh_k_SX = casadi.vertcat(q_mesh_k_SX, qdot_mesh_k_SX)

# States and state derivatives on collocation points
q_coll_k_SX = casadi.SX.sym('q_coll_k', n_coords, n_coll)
qdot_coll_k_SX = casadi.SX.sym('qdot_coll_k', n_coords, n_coll)
qddot_coll_k_SX = casadi.SX.sym('qddot_coll_k', n_coords, n_coll)

# Actuation control in mesh interval
act_mesh_k_SX = casadi.SX.sym('act_mesh_k', n_act, 1)

# "Controls" in mesh interval for the OCP format of fatrop
u_mesh_k_SX = casadi.vertcat(act_mesh_k_SX,
                              q_coll_k_SX.reshape((-1, 1)),
                              qdot_coll_k_SX.reshape((-1, 1)),
                              qddot_coll_k_SX.reshape((-1, 1)))
print(f"u_mesh_k_SX: {u_mesh_k_SX}")

nx = x_mesh_k_SX.shape[0]
nu = u_mesh_k_SX.shape[0]

# Reshape for collocation
q_k_SX = casadi.horzcat(q_mesh_k_SX, q_coll_k_SX)
qdot_k_SX = casadi.horzcat(qdot_mesh_k_SX, qdot_coll_k_SX)

# =============================================================================
# CREATE OCP FUNCTIONS FOR KTH MESH INTERVAL
# =============================================================================

tau_root, C, D, B = collocation_scheme(n_coll, 'radau')

# States on mesh point k+1
# Last Radau collocation point corresponds to next mesh point.
# States on 1st mesh point are simply passed on.
if solver == 'fatrop':
    x_mesh_kp1_SX = casadi.vertcat(
        q_k_SX[:, n_coll],  # q at next mesh point
        qdot_k_SX[:, n_coll],  # qdot at next mesh point
        q_mesh_1_SX,
        qdot_mesh_1_SX
    )
else:
    x_mesh_kp1_SX = casadi.vertcat(
        q_k_SX[:, n_coll],
        qdot_k_SX[:, n_coll]
    )

# Error on collocation equation - path constraint
err_coll_q_SX = q_k_SX @ C[:, 1:] - (qdot_coll_k_SX * scale_qdots) @ np.array([[dt]])
err_coll_qdot_SX = (qdot_k_SX * scale_qdots) @ C[:, 1:] - \
                   (qddot_coll_k_SX * scale_qddots) @ np.array([[dt]])

# Evaluate system dynamics function
# This function is from another project, hence there are dummy inputs and
# unused outputs.
if f_sysdyn is not None:
    [effort_coll_k_SX, limtorq_coll_k_SX, _, err_sysdyn_SX, _] = f_sysdyn(
        casadi.SX.sym('dummy1', 0, 1),  # placeholder
        q_coll_k_SX,
        qdot_coll_k_SX * scale_qdots,
        casadi.SX.sym('dummy2', 0, 1),  # placeholder
        casadi.SX.sym('dummy3', 0, 1),  # placeholder
        qddot_coll_k_SX * scale_qddots,
        casadi.SX.sym('dummy4', 0, 1),  # placeholder
        0, 0,
        act_mesh_k_SX,
        0, 0
    )
else:
    raise FileNotFoundError(f"CasADi function file {casadi_func_file} not found. Cannot evaluate system dynamics.")
    # Use dummy effort and torque constraints
    effort_coll_k_SX = casadi.SX.zeros(1, n_coll)
    limtorq_coll_k_SX = casadi.SX.zeros(1, n_coll)
    err_sysdyn_SX = casadi.SX.zeros(n_coords, n_coll)

# Integrate objective terms over mesh interval
effort_k_SX = effort_coll_k_SX @ B[1:] * dt
limtorq_k_SX = limtorq_coll_k_SX @ B[1:] * dt

# Create CasADi Functions
f_int = casadi.Function('f_int', [x_mesh_k_SX, u_mesh_k_SX], [x_mesh_kp1_SX])

f_path = casadi.Function('f_path', [x_mesh_k_SX, u_mesh_k_SX],
                         [casadi.vertcat(
                             err_coll_q_SX.reshape((-1, 1)),
                             err_coll_qdot_SX.reshape((-1, 1)) / scale_qdots,
                             err_sysdyn_SX.reshape((-1, 1)) / 100
                         )])

f_obj = casadi.Function('f_obj', [x_mesh_k_SX, u_mesh_k_SX],
                        [effort_k_SX, limtorq_k_SX])

# =============================================================================
# PERIODICITY AND SYMMETRY CONSTRAINTS
# =============================================================================
# Impose walking to be periodic and symmetric (with 180° phase shift).
# This is formulated as equality constraints on the states, with a mapping
# between left and right side.
#
# k = 1         k = N+1
# ------------------------
# pelvis_tx     pelvis_tx
# pelvis_ty     pelvis_ty
# hip_r         hip_l
# hip_l         hip_r
# knee_r        knee_l
# knee_l        knee_r
#
# The state giving the forward position of the floating base is excluded,
# because its initial and final value are prescribed.
#

if n_coords == 6:
    # MATLAB indices [2,4,3,6,5] converted to Python 0-indexed: [1,3,2,5,4]
    err_per_q_SX = q_mesh_1_SX - q_mesh_k_SX[[1, 3, 2, 5, 4], :]
    # MATLAB indices [1,2,4,3,6,5] converted to Python 0-indexed: [0,1,3,2,5,4]
    err_per_qdot_SX = qdot_mesh_1_SX - qdot_mesh_k_SX[[0, 1, 3, 2, 5, 4], :]
elif n_coords == 9:
    # MATLAB indices [1,3,5,4,7,6,9,8] converted to Python 0-indexed: [0,2,4,3,6,5,8,7]
    err_per_q_SX = q_mesh_1_SX - q_mesh_k_SX[[0, 2, 4, 3, 6, 5, 8, 7], :]
    # MATLAB indices [1,2,3,5,4,7,6,9,8] converted to Python 0-indexed: [0,1,2,4,3,6,5,8,7]
    err_per_qdot_SX = qdot_mesh_1_SX - qdot_mesh_k_SX[[0, 1, 2, 4, 3, 6, 5, 8, 7], :]

if solver == 'fatrop':
    # State vector is augmented with states on 1st mesh point.
    f_per = casadi.Function('f_per', [x_mesh_k_SX],
                            [casadi.vertcat(err_per_q_SX, err_per_qdot_SX)])
else:
    f_per = casadi.Function('f_per',
                            [casadi.vertcat(q_mesh_1_SX, qdot_mesh_1_SX), x_mesh_k_SX],
                            [casadi.vertcat(err_per_q_SX, err_per_qdot_SX)])

# =============================================================================
# VARIABLE BOUNDS
# =============================================================================

if n_coords == 6:
    q_lb = np.array([-0.1, 0.5, -50, -50, -120, -120]) * np.array([1, 1, np.pi/180, np.pi/180, np.pi/180, np.pi/180])
    q_ub = np.array([step_length + 0.1, 1.5, 90, 90, 10, 10]) * np.array([1, 1, np.pi/180, np.pi/180, np.pi/180, np.pi/180])
elif n_coords == 9:
    q_lb = np.array([-1, -0.1, 0.5, -50, -50, -120, -120, -50, -50]) * \
           np.array([1, 1, 1, np.pi/180, np.pi/180, np.pi/180, np.pi/180, np.pi/180, np.pi/180])
    q_ub = np.array([1, step_length + 0.1, 1.5, 90, 90, 10, 10, 50, 50]) * \
           np.array([1, 1, 1, np.pi/180, np.pi/180, np.pi/180, np.pi/180, np.pi/180, np.pi/180])

qdot_ub = 10 * (q_ub - q_lb) / scale_qdots
qddot_ub = 150 * qdot_ub / scale_qddots

if torque_driven:
    act_lb = -np.ones((n_act, 1))
    act_ub = np.ones((n_act, 1))
else:
    act_lb = 0.05 * np.ones((n_act, 1))
    act_ub = np.ones((n_act, 1))

# Build state bounds
# Python setdiff1d for indices
idx_excl = np.array([idx_q_fwd])
idx_not_excl = np.setdiff1d(np.arange(n_coords), idx_excl)

x_lb = np.concatenate([q_lb, -qdot_ub, q_lb[idx_not_excl], -qdot_ub])
x_ub = np.concatenate([q_ub, qdot_ub, q_ub[idx_not_excl], qdot_ub])

u_lb = np.concatenate([act_lb.flatten(),
                       np.tile(q_lb, n_coll),
                       np.tile(-qdot_ub, n_coll),
                       np.tile(-qddot_ub, n_coll)])

u_ub = np.concatenate([act_ub.flatten(),
                       np.tile(q_ub, n_coll),
                       np.tile(qdot_ub, n_coll),
                       np.tile(qddot_ub, n_coll)])

# =============================================================================
# INITIAL GUESS
# =============================================================================

x_init = np.arange(N_mesh * 2 + 1)
q_tx_ig = x_init / N_mesh * step_length
q_ty_ig = 0.9 * np.ones(N_mesh * 2 + 1)
q_ty_ig = q_ty_ig - 0.05 * np.cos(x_init / N_mesh / 2 * np.pi * 4)
q_hip_ig = 20 * np.pi / 180 * np.cos(x_init / N_mesh / 2 * np.pi * 2)
q_knee_ig = np.concatenate([-0.1 * np.ones(N_mesh + 1),
                            0.5 * (np.cos(x_init[:N_mesh] / N_mesh * np.pi * 2) - 1) - 0.1])

q_ig = np.zeros((n_coords, N_mesh + 1))
qdot_ig = np.zeros((n_coords, N_mesh + 1))

qdot_ig[idx_q_fwd, :] = 1.2 / scale_qdots

if n_coords == 6:
    q_ig[0, :] = q_tx_ig[:N_mesh + 1]
    q_ig[1, :] = q_ty_ig[:N_mesh + 1]
    q_ig[3, :] = q_hip_ig[:N_mesh + 1]
    q_ig[2, :] = q_hip_ig[N_mesh:]
    q_ig[5, :] = q_knee_ig[:N_mesh + 1]
    q_ig[4, :] = q_knee_ig[N_mesh:]

elif n_coords == 9:
    q_ig[1, :] = q_tx_ig[:N_mesh + 1]
    q_ig[2, :] = q_ty_ig[:N_mesh + 1]
    q_ig[4, :] = q_hip_ig[:N_mesh + 1]
    q_ig[3, :] = q_hip_ig[N_mesh:]
    q_ig[6, :] = q_knee_ig[:N_mesh + 1]
    q_ig[5, :] = q_knee_ig[N_mesh:]

x_ig = np.vstack([q_ig, qdot_ig,
                  np.tile(q_ig[idx_not_excl, 0:1], (1, N_mesh + 1)),
                  np.tile(qdot_ig[:, 0:1], (1, N_mesh + 1))])

u_ig = np.vstack([0.1 * np.ones((n_act, N_mesh)),
                  np.tile(q_ig[:, 1:], (n_coll, 1)),
                  np.tile(qdot_ig[:, 1:], (n_coll, 1)),
                  np.zeros((n_coords * n_coll, N_mesh))])

if solver != 'fatrop':
    x_lb = x_lb[:n_coords * 2]
    x_ub = x_ub[:n_coords * 2]
    x_ig = x_ig[:n_coords * 2, :]

# =============================================================================
# OCP SETUP
# =============================================================================
# Based on https://github.com/jgillis/fatrop_demo/blob/master/fatrop_opti.m

### printing some info

opti = casadi.Opti()

X = []
U = []
for k in range(N_mesh):
    X.append(opti.variable(nx))
    U.append(opti.variable(nu))
X.append(opti.variable(nx))

J_act = []
J_lim = []

for k in range(N_mesh):
    # Multiple shooting gap-closing constraint
    opti.subject_to(X[k + 1] - f_int(X[k], U[k]) == 0)

    # Initial constraints
    if k == 0:
        # Forward position starts at 0
        opti.subject_to(X[0][idx_q_fwd] == 0)
        if solver == 'fatrop':
            # States on kth mesh point are equal to states on 1st mesh point
            # because here k=1 (0-indexed)
            idx_keep_indices = np.setdiff1d(np.arange(n_coords * 2), [idx_q_fwd])
            opti.subject_to(X[0][idx_keep_indices] == X[0][n_coords * 2:])

    # Path constraint
    opti.subject_to(f_path(X[k], U[k]) == 0)

    # Bounds
    opti.subject_to(x_lb < X[k])
    opti.subject_to(X[k] < x_ub)
    opti.subject_to(u_lb < U[k])
    opti.subject_to(U[k] < u_ub)

    # Initial guess
    opti.set_initial(X[k], x_ig[:, k])
    opti.set_initial(U[k], u_ig[:, k])

    # Final constraints
    if k == N_mesh - 1:
        # Forward position is imposed step length
        opti.subject_to(X[N_mesh][idx_q_fwd] == step_length)
        if solver == 'fatrop':
            opti.subject_to(f_per(X[N_mesh]) == 0)
        else:
            # For non-FATROP, extract initial state from X[0]
            # f_per expects (q without forward coord, all qdots) and full state
            # Build the state vector manually
            x_0_state_list = [X[0][i] for i in idx_not_excl] + [X[0][n_coords + i] for i in range(n_coords)]
            x_0_state = casadi.vertcat(*x_0_state_list)
            opti.subject_to(f_per(x_0_state, X[N_mesh]) == 0)

        opti.set_initial(X[N_mesh], x_ig[:, N_mesh])

    # Objective
    J_act_k, J_lim_k = f_obj(X[k], U[k])
    J_act.append(J_act_k)
    J_lim.append(J_lim_k)

# Combine objective
obj = sum(J_act) * 1e2 + sum(J_lim) * 1e-1  # effort + limit torques

opti.minimize(obj)

# =============================================================================
# SOLVER SETUP AND SOLVE
# =============================================================================

# Solver options
options = {
    'expand': True,
    'detect_simple_bounds': True,
}

options[solver] = {
    'tol': 1e-4,
    'max_iter': 1
}

if solver == 'fatrop':
    options['fatrop']['mu_init'] = 0.1
    options['structure_detection'] = 'auto'
    options['debug'] = True
elif solver == 'ipopt':
    options['ipopt'] = {
        'nlp_scaling_method': 'none',
        'mu_strategy': 'adaptive',
    }

opti.solver(solver, options)

print(f"Solving with {solver}...")
try:
    sol = opti.solve()
except Exception as e:
    print(f"Solver failed with error: {e}")
    sol = opti.debug
    warnings.warn(str(e))

    

# =============================================================================
# EXTRACT AND VISUALIZE SOLUTION
# =============================================================================

X_sol = np.hstack([opti.debug.value(X[k]) for k in range(len(X))])
U_sol = np.hstack([opti.debug.value(U[k]) for k in range(len(U))])

q_mesh = X_sol[:n_coords, :]
qdot_mesh = X_sol[n_coords:n_coords * 2, :]

# Construct full cycle from solution for half cycle
if n_coords == 6:
    q_GC = np.hstack([q_mesh[:, :N_mesh], q_mesh[[0, 1, 3, 2, 5, 4], :]])
elif n_coords == 9:
    q_GC = np.hstack([q_mesh[:, :N_mesh], q_mesh[[0, 1, 2, 4, 3, 6, 5, 8, 7], :]])

q_GC[idx_q_fwd, N_mesh:] += step_length
act_GC = np.hstack([U_sol[n_act // 2:n_act, :],
                     U_sol[:n_act // 2, :]])

t_GC = np.linspace(0, step_time * 2, N_mesh * 2 + 1)

# Create plots
fig = plt.figure(figsize=(14, 10))

plot_idx = 1
coord_idx = 0

if n_coords == 9:
    ax = plt.subplot(3, 3, plot_idx)
    ax.plot(t_GC, q_GC[coord_idx, :] * 180 / np.pi)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('(°)')
    ax.set_title('torso angle')
    plot_idx += 1
    coord_idx += 1

# Forward position
ax = plt.subplot(3, 3, plot_idx)
ax.plot(t_GC, q_GC[coord_idx, :])
ax.set_xlabel('time (s)')
ax.set_ylabel('(m)')
ax.set_title('forward')
plot_idx += 1
coord_idx += 1

# Vertical position
ax = plt.subplot(3, 3, plot_idx)
ax.plot(t_GC, q_GC[coord_idx, :])
ax.set_xlabel('time (s)')
ax.set_ylabel('(m)')
ax.set_title('vertical')
plot_idx += 1
coord_idx += 2  # Skip velocity

# Hip angle
ax = plt.subplot(3, 3, plot_idx)
ax.plot(t_GC, q_GC[coord_idx, :] * 180 / np.pi)
ax.set_xlabel('time (s)')
ax.set_ylabel('(°)')
ax.set_title('hip')
plot_idx += 1
coord_idx += 2  # Skip velocity

# Knee angle
ax = plt.subplot(3, 3, plot_idx)
ax.plot(t_GC, q_GC[coord_idx, :] * 180 / np.pi)
ax.set_xlabel('time (s)')
ax.set_ylabel('(°)')
ax.set_title('knee')
plot_idx += 1
coord_idx += 2

if n_coords == 9:
    # Ankle angle
    ax = plt.subplot(3, 3, plot_idx)
    ax.plot(t_GC, q_GC[coord_idx, :] * 180 / np.pi)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('(°)')
    ax.set_title('ankle')
    plot_idx += 1

plt.tight_layout()
plt.savefig('walking_gait_solution.png', dpi=150)
print("Plot saved as 'walking_gait_solution.png'")
plt.show()

print("Optimization completed!")
