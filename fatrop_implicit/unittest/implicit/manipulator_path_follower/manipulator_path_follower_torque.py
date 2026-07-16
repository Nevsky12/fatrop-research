from casadi import *

functions = {"ur10_fk.casadi":None, "ur10_fd.casadi":None, "ur10_id.casadi":None, "ur10_J_fd.casadi":None, "ur10_J_id.casadi":None}

for f in functions.keys():
    try:
        func = Function.load(f)
    except:
        func = Function.load(f"unittest/implicit/manipulator_path_follower/{f}")

    functions[f] = func
    print(func)


def ee_pos(q, fk):
    return fk(q)[-1][:-1, -1]

# generate path to follow
def get_path(fk, n, N, T):
    q = DM.zeros(n, N+1)
    for i in range(n):
        q[i, :] = 1-cos(linspace(0, T*i, N+1))
    
    path = DM.zeros(3, N+1)
    for i in range(N+1):
        path[:, i] = ee_pos(q[:, i], fk)

    return path

def get_path_function(fk, n):
    t = MX.sym("t", 1)
    q = MX.zeros(n, 1)
    for i in range(n):
        # q[i] = 0.5*(1-cos(t*(i+1)))
        q[i] = 0.5*(1-cos(t*(0.5*i+1)))
        # q[i] = 0.1*t

    return Function("eval_path", [t], [ee_pos(q, fk)])

def get_equilibrium_tau(q):
    n = 6
    opti = Opti()
    tau_eq = opti.variable(n)
    opti.subject_to(tau_eq == functions['ur10_id.casadi'](q, DM.zeros(n), DM.zeros(n)))
    opti.minimize(sumsqr(tau_eq))
    opti.solver('ipopt', {}, {})
    sol = opti.solve()
    print("Equilibrium tau:", sol.value(tau_eq))
    return sol.value(tau_eq)

tau_eq = get_equilibrium_tau(DM.zeros(6))

# define params
opti = Opti()
N = 70
T = 3.0
dt = T/N
# tau_bound = 500
tau_bound = DM([330, 330, 150, 80, 80, 80])
path_tol = 0.05
control_penalty = 1.0e-4
n = 6

with_slack = 1
with_progress_variable = 1
with_accel_variables = 1

# define variables
q = []; qd = []; qdd = []; tau = []; s = []
p = []; dp = []
for k in range(N):
    # states
    q.append(opti.variable(n))
    qd.append(opti.variable(n))
    if with_progress_variable:
        p.append(opti.variable(1))
    # controls
        dp.append(opti.variable(1))

    if with_accel_variables:
        qdd.append(opti.variable(n))
    tau.append(opti.variable(n))
    if with_slack:
        s.append(opti.variable(3))
q.append(opti.variable(n))
qd.append(opti.variable(n))
p.append(opti.variable(1))
if with_accel_variables:
    qdd.append(opti.variable(n))



q = hcat(q)
qd = hcat(qd)
if with_accel_variables:
    qdd = hcat(qdd)
tau = hcat(tau)
if with_slack:
    s = hcat(s)
else:
    s = MX.zeros(3, N)
if with_progress_variable:
    p = hcat(p)
    dp = hcat(dp)
else:
    p = transpose(linspace(0, T, N+1))
    dp = MX.ones(1, N)

# initial state
opti.subject_to(q[:, 0] == 0)
opti.subject_to(qd[:, 0] == 0)
if with_progress_variable:
    opti.subject_to(p[:,0] == 0)

path_func = get_path_function(functions['ur10_fk.casadi'], n)
obj = 0.0
for k in range(N):
    # dymamics
    opti.subject_to(q[:,k+1] == q[:,k] + qd[:,k]*dp[:,k])
    if with_accel_variables:
        opti.subject_to(qd[:,k+1] == qd[:,k] + qdd[:,k]*dp[:,k])
    else:
        qdd = functions['ur10_fd.casadi'](q[:,k], qd[:,k], tau[:,k])
        opti.subject_to(qd[:,k+1] == qd[:,k] + qdd*dp[:,k])
    
    if with_progress_variable:
        opti.subject_to(p[:,k+1] == p[:,k] + dp[:,k])


    # slack constraints
    ee_pos_k = path_func(p[:,k])
    if with_slack:
        opti.subject_to(ee_pos(q[:,k], functions['ur10_fk.casadi']) - ee_pos_k - s[:,k] == 0)
    else:
        s[:,k] = ee_pos(q[:,k], functions['ur10_fk.casadi']) - ee_pos_k
    
    # true dynamics
    if with_accel_variables:
        opti.subject_to(tau[:,k] == functions['ur10_id.casadi'](q[:,k], qd[:,k], qdd[:,k]))

    # inequalities
    opti.subject_to(-1 <= (sumsqr(s[:,k]) <= path_tol**2))
    opti.subject_to(-tau_bound <= (tau[:,k] <= tau_bound))

    if with_progress_variable:
        opti.subject_to(0.2*dt <= (dp[:,k] <= 2.0*dt))
        # opti.subject_to(dp[:,k] == dt)

    obj += 0*sumsqr(s[:,k]) + control_penalty*sumsqr(tau[:,k]) + 1e-4*sumsqr(dp[:,k] - dt)
    if with_accel_variables:
        # obj += sumsqr(qdd[:,k])*dp[:,k]
        obj += sumsqr(qdd[:,k])

if with_progress_variable:
    opti.subject_to(p[-1,-1] == T)
    opti.set_initial(p, transpose(linspace(0, T, N+1)))
    opti.set_initial(dp, dt)

for k in range(N):
    opti.set_initial(tau[:,k], tau_eq)

opti.minimize(obj)
opti.solver('fatrop', {'structure_detection':'auto', 'debug':True}, {})
# opti.solver('ipopt', {}, {})

sol = opti.solve()

# t_sol = linspace(0, T, N+1)
t_sol = sol.value(p)
q_sol = sol.value(q)
qd_sol = sol.value(qd)
tau_sol = sol.value(tau)
s_sol = sol.value(s)

path = DM.zeros(3, N+1)
travelled_path = DM.zeros(3, N+1)
for i in range(N+1):
    path[:, i] = path_func(t_sol[i])
    travelled_path[:, i] = ee_pos(q_sol[:, i], functions['ur10_fk.casadi'])

import matplotlib.pyplot as plt

# show 3d travelled path
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(path[0, :], path[1, :], path[2, :], label='Desired Path')
ax.plot(travelled_path[0, :], travelled_path[1, :], travelled_path[2, :], label='Travelled Path')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Manipulator Path Follower')
ax.set_box_aspect([1,1,1])  # Equal aspect ratio
ax.legend()

# show 1d travelled coordinates
fig, axs = plt.subplots(4, 1)
for i in range(3):
    axs[i].plot(t_sol, transpose(path[i, :]), label="desired coordinate")
    axs[i].plot(t_sol, transpose(travelled_path[i, :]), label="travelled coordinate")
    axs[i].legend()

distance = [float(sumsqr(s_sol[:,k])) for k in range(N)]
axs[3].plot(t_sol[:-1], distance, label="distance to path")
axs[3].axhline(path_tol**2, color='r', linestyle='-', label="path tolerance")
axs[3].set_ylim(0, (path_tol**2)*1.1)
plt.tight_layout()

# show joint positions, velocities, and accelerations
fig, axs = plt.subplots(3, 1)
for i in range(n):
    axs[0].plot(t_sol, q_sol[i, :], label=f"joint {i} position")
    axs[1].plot(t_sol, qd_sol[i, :], label=f"joint {i} velocity")
    axs[2].plot(t_sol[:-1], tau_sol[i, :], label=f"joint {i} torque")
    axs[2].axhline(float(tau_bound[i]), color='r', linestyle='-', label=f"tau bound {i}")
    axs[2].axhline(float(-tau_bound[i]), color='r', linestyle='-')
    
# show progress variable over time
fig, ax = plt.subplots()
if with_progress_variable:
    ax.plot(linspace(0, T, len(t_sol)), t_sol, label="progress variable")
    ax.plot([0, T], [0, T], '--', label="ideal progress")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Progress Variable")
    ax.set_title("Progress Variable Over Time")
    ax.legend()

plt.show()

