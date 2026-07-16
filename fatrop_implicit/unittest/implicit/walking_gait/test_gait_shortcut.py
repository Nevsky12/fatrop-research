from casadi import *

# load functions
folder = "casadi_funcs/n_coll_3"
f_objk = Function.load(f"{folder}/f_objk.casadi")
f_g0 = Function.load(f"{folder}/f_g0.casadi")
f_gk = Function.load(f"{folder}/f_gk.casadi")
f_gK = Function.load(f"{folder}/f_gN.casadi")
f_gk_ineq = Function.load(f"{folder}/f_gk_ineq.casadi")
f_gap = Function.load(f"{folder}/f_gap.casadi")

f_lb = Function.load(f"{folder}/f_lb.casadi")
f_ub = Function.load(f"{folder}/f_ub.casadi")
f_x_init = Function.load(f"{folder}/f_init_x.casadi")
f_u_init = Function.load(f"{folder}/f_init_u.casadi")

# derive dimensions
xx_init = f_x_init()['x_init']
uu_init = f_u_init()['u_init']
lb = f_lb()['lb']
ub = f_ub()['ub']
nx = xx_init.shape[0]
nu = uu_init.shape[0]
N = xx_init.shape[1] - 1

# setup opti instance
opti = Opti()
xx = []
uu = []
for k in range(N):
    xx.append(opti.variable(nx, 1))
    uu.append(opti.variable(nu, 1))
xx.append(opti.variable(nx, 1))  # add state at mesh point N

obj = 0
for k in range(N):
    # gap-closing constraint
    opti.subject_to(xx[k+1] == f_gap(xx[k], uu[k]))
    
    # # equality constraint
    if k == 0:
        opti.subject_to(f_g0(xx[k], uu[k]) == 0)
    else:
        opti.subject_to(f_gk(xx[k], uu[k]) == 0)
        
    # inequality constraint
    opti.subject_to(lb <= (f_gk_ineq(xx[k], uu[k]) <= ub))
    
    # initial guess
    opti.set_initial(xx[k], xx_init[:, k])
    opti.set_initial(uu[k], uu_init[:, k])
    
    # final constraints
    if k == N-1:
        opti.subject_to(f_gK(xx[k+1]) == 0)
        opti.set_initial(xx[k+1], xx_init[:, k+1])
        
    # objective
    obj += f_objk(xx[k], uu[k])
    
opti.minimize(obj)

opti.solver('fatrop', 
            {'expand': True, 
             'detect_simple_bounds': True,
             'structure_detection': 'auto'},
            {'tol': 1e-4,
             'mu_init': 0.1})

# opti_f = opti.to_function('opti_f', [], [hcat(xx), hcat(uu)], [], ['xx', 'uu'])
# opti_f.save('casadi_funcs/test_gait_shortcut_python.casadi')
sol = opti.solve()

xx_sol = sol.value(hcat(xx))
uu_sol = sol.value(hcat(uu))
import matplotlib.pyplot as plt
n_coords = 9
q_mesh = xx_sol[:n_coords, :]
qdot_mesh = xx_sol[n_coords:2*n_coords, :]

step_time = 1.0 / 0.9 / 2.0
dt = step_time / N
step_length = 1.2 * step_time

plt.figure()
plt.plot(xx_sol.T)
plt.show()

# Construct full cycle from solution for half cycle
q_GC = np.hstack([q_mesh[:, :], q_mesh[[0, 2, 1, 4, 3, 6, 5, 8, 7],:]])
q_GC[1, N:] = q_GC[1, N:] + step_length

t_GC = np.linspace(0, 2*dt*N, 2*N+1)

fig, axs = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Walking Gait Simulation Results')

# Torso Angle
axs[0, 0].plot(t_GC, q_GC[0, :-1] * 180 / np.pi)
axs[0, 0].set_title('Torso Angle')
axs[0, 0].set_xlabel('Time (s)')
axs[0, 0].set_ylabel('Angle (°)')

# Forward Position
axs[0, 1].plot(t_GC, q_GC[1, :-1])
axs[0, 1].set_title('Forward Position')
axs[0, 1].set_xlabel('Time (s)')
axs[0, 1].set_ylabel('Position (m)')

# Vertical Position
axs[0, 2].plot(t_GC, q_GC[2, :-1])
axs[0, 2].set_title('Vertical Position')
axs[0, 2].set_xlabel('Time (s)')
axs[0, 2].set_ylabel('Position (m)')

# Hip Angle
axs[1, 0].plot(t_GC, q_GC[4, :-1] * 180 / np.pi)
axs[1, 0].set_title('Hip Angle')
axs[1, 0].set_xlabel('Time (s)')
axs[1, 0].set_ylabel('Angle (°)')

# Knee Angle
axs[1, 1].plot(t_GC, q_GC[6, :-1] * 180 / np.pi)
axs[1, 1].set_title('Knee Angle')
axs[1, 1].set_xlabel('Time (s)')
axs[1, 1].set_ylabel('Angle (°)')

# Ankle Angle
axs[1, 2].plot(t_GC, q_GC[8, :-1] * 180 / np.pi)
axs[1, 2].set_title('Ankle Angle')
axs[1, 2].set_xlabel('Time (s)')
axs[1, 2].set_ylabel('Angle (°)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
