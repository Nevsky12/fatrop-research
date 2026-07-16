from casadi import *
from collocation_scheme import collocation_scheme
from reformulator import Reformulator
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import yaml
import sys
import io

######################
### Set parameters ###
######################
config_file = "config.yaml"
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

print(config)

n_coll = config['n_coll']
introduce_periodicity_vars = config['introduce_periodicity_vars']
problem_type = config['problem_type']
save_opti_f = config['save_opti_f']
solve = config['solve']
store_solution = config['store_solution']
load_solution = config['load_solution']
visualize_solution = config['visualize_solution']

######################
### load functions ###
######################
folder = f"casadi_funcs/n_coll_{n_coll}"
f_objk_orig = Function.load(f"{folder}/f_objk.casadi")
f_g0_orig = Function.load(f"{folder}/f_g0.casadi")
f_gk_orig = Function.load(f"{folder}/f_gk.casadi")
f_gK_orig = Function.load(f"{folder}/f_gN.casadi")
f_gk_ineq_orig = Function.load(f"{folder}/f_gk_ineq.casadi")
f_gap_orig = Function.load(f"{folder}/f_gap.casadi")
f_lb_orig = Function.load(f"{folder}/f_lb.casadi")
f_ub_orig = Function.load(f"{folder}/f_ub.casadi")
f_x_init_orig = Function.load(f"{folder}/f_init_x.casadi")
f_u_init_orig = Function.load(f"{folder}/f_init_u.casadi")
f_g0_orig_unique = Function.load(f"{folder}/f_g0_unique.casadi")

# derive dimensions
xx_init = f_x_init_orig()['x_init']
uu_init = f_u_init_orig()['u_init']
lb = f_lb_orig()['lb']
ub = f_ub_orig()['ub']
nx = xx_init.shape[0]
nu = uu_init.shape[0]
N = xx_init.shape[1] - 1
xk = SX.sym('xk', nx)


#############################
### change ordering of uk ###
#############################
# from
# [act_mesh_k_SX, q_coll(1)_k_SX, ..., q_coll(3)_k_SX, qdot_coll(1)_k_SX, ..., qdot_coll(3)_k_SX, qddot_coll(1)_k_SX, ..., qddot_coll(3)_k_SX]
#                                      --------------                          ------------------                           
# to
# [act_mesh_k_SX, q_coll(1)_k_SX, ..., qddot_coll(1)_k_SX, q_coll(2)_k_SX, ..., qddot_coll(2)_k_SX, qddot_coll(3)_k_SX, q_coll(3)_k_SX, ...]
#                                                                                                                       --------------------
# such that zk variables are at the back
n_act_mesh = 18
n_coords = 9
dt = (1.0/0.9/2.0)/N
act_mesh = SX.sym('act_mesh_k', n_act_mesh)
q_coll = SX.sym('q_coll_k', n_coll * n_coords)
qdot_coll = SX.sym('qdot_coll_k', n_coll * n_coords)
qddot_coll = SX.sym('qddot_coll_k', n_coll * n_coords)
uk_orig = vertcat(act_mesh, q_coll, qdot_coll, qddot_coll)
assert uk_orig.shape[0] == nu

periodicity_vars = SX.sym('periodicity_vars', introduce_periodicity_vars*(n_coords * 2 - 1)) # all q and qdot variables except for the forward foot x position
zk = vertcat(q_coll[(n_coll-1)*n_coords:n_coll*n_coords], qdot_coll[(n_coll-1)*n_coords:n_coll*n_coords], periodicity_vars)
if introduce_periodicity_vars:
    assert zk.shape[0] == nx
    periodicity_constraint = periodicity_vars - xk[n_coords * 2:]
else:
    periodicity_constraint = SX.zeros(0,1)

uk = act_mesh
for i in range(n_coll-1):
    uk = vertcat(uk, q_coll[i*n_coords:(i+1)*n_coords], qdot_coll[i*n_coords:(i+1)*n_coords], qddot_coll[i*n_coords:(i+1)*n_coords])
uk = vertcat(uk, qddot_coll[(n_coll-1)*n_coords:n_coll*n_coords], zk) # keep the last q_coll and qdot_coll as zk variables to enforce periodicity
# uk  = vertcat(act_mesh, 
#               q_coll[0:n_coords], qdot_coll[0:n_coords], qddot_coll[0:n_coords],
#               q_coll[n_coords:2*n_coords], qdot_coll[n_coords:2*n_coords], qddot_coll[n_coords:2*n_coords],
#               qddot_coll[2*n_coords:3*n_coords], # accelerations are still controls
#               zk)
f_objk = Function('f_objk', [xk, uk], [f_objk_orig(xk, uk_orig)])
f_g0 = Function('f_g0', [xk, uk], [vertcat(f_g0_orig(xk, uk_orig), periodicity_constraint)])
f_gk = Function('f_gk', [xk, uk], [vertcat(f_gk_orig(xk, uk_orig), periodicity_constraint)])
f_gK = Function('f_gK', [xk], [f_gK_orig(xk)])
f_gk_ineq = Function('f_gk_ineq', [xk, uk], [f_gk_ineq_orig(xk, uk_orig)])
f_gap = Function('f_gap', [xk, uk], [zk if introduce_periodicity_vars else vertcat(zk, xk[n_coords*2:])])
f_g0_unique = Function('f_g0_unique', [xk, uk], [f_g0_orig_unique(xk, uk_orig)])

###############################
### Update path constraints ###
###############################
reformulator = Reformulator(n_coll, n_coords, n_act_mesh, dt, xk, uk, uk_orig, q_coll, qdot_coll, qddot_coll)
f_gk_reformulated = reformulator.get_f_gk_reformulated(f_gk)
f_g0_reformulated = Function('f_g0_reformulated', [xk, uk], [vertcat(f_g0_unique(xk, uk), f_gk_reformulated(xk, uk))])
# reformulator.show_gk_jacobian_structure_for_debugging(f_gk)
# reformulator.show_gk_jacobian_structure_for_debugging(f_gk_reformulated)

########################################
### update initial guess of controls ###
########################################
act_init = uu_init[:n_act_mesh, :]
q_coll_init = uu_init[n_act_mesh:n_act_mesh+n_coll*n_coords, :]
qdot_coll_init = uu_init[n_act_mesh+n_coll*n_coords:n_act_mesh+2*n_coll*n_coords, :]
qddot_coll_init = uu_init[n_act_mesh+2*n_coll*n_coords:n_act_mesh+3*n_coll*n_coords, :]

# actual controls
uu_init_reordered = act_init

# collocation points
for i in range(n_coll-1):
    uu_init_reordered = vertcat(uu_init_reordered, q_coll_init[i*n_coords:(i+1)*n_coords, :], qdot_coll_init[i*n_coords:(i+1)*n_coords, :], qddot_coll_init[i*n_coords:(i+1)*n_coords, :])

# last collocation point qddot and zk values
uu_init_reordered = vertcat(uu_init_reordered, 
                            qddot_coll_init[(n_coll-1)*n_coords:n_coll*n_coords, :], 
                            q_coll_init[(n_coll-1)*n_coords:n_coll*n_coords, :], 
                            qdot_coll_init[(n_coll-1)*n_coords:n_coll*n_coords, :])
if introduce_periodicity_vars:
    uu_init_reordered = vertcat(uu_init_reordered, xx_init[n_coords * 2:, :-1]) # periodicity vars initial guess

assert uu_init_reordered.shape[0] == nu + (2*n_coords - 1 if introduce_periodicity_vars else 0)

######################################
### Code generate casadi functions ###
######################################
# if config['perform_code_generation']:
#     from code_generator import get_code_generated_function
    
#     f_objk = get_code_generated_function(f_objk)
#     f_g0_reformulated = get_code_generated_function(f_g0_reformulated)
#     f_gk_reformulated = get_code_generated_function(f_gk_reformulated)
#     f_gK = get_code_generated_function(f_gK)
#     f_gk_ineq = get_code_generated_function(f_gk_ineq)
#     f_gap = get_code_generated_function(f_gap)


###########################
### setup opti instance ###
###########################
opti = Opti()
xx = []
uu = []
for k in range(N):
    xx.append(opti.variable(nx, 1))
    uu.append(opti.variable(nu + (2*n_coords - 1 if introduce_periodicity_vars else 0), 1))
xx.append(opti.variable(nx, 1))  # add state at mesh point N

obj = 0
for k in range(N):
    # gap-closing constraint
    opti.subject_to(xx[k+1] == f_gap(xx[k], uu[k]))
    
    # # equality constraint
    if k == 0:
        # opti.subject_to(f_g0(xx[k], uu[k]) == 0)
        opti.subject_to(f_g0_reformulated(xx[k], uu[k]) == 0)
    else:
        # opti.subject_to(f_gk(xx[k], uu[k]) == 0)
        opti.subject_to(f_gk_reformulated(xx[k], uu[k]) == 0)

    # inequality constraint
    opti.subject_to(lb <= (f_gk_ineq(xx[k], uu[k]) <= ub))
    
    # initial guess
    opti.set_initial(xx[k], xx_init[:, k])
    opti.set_initial(uu[k], uu_init_reordered[:, k])
    
    # final constraints
    if k == N-1:
        opti.subject_to(f_gK(xx[k+1]) == 0)
        opti.set_initial(xx[k+1], xx_init[:, k+1])
        
    # objective
    obj += f_objk(xx[k], uu[k])
    
opti.minimize(obj)

# Note: Fatrop does not support CasADi callbacks, so we will capture stdout during solving
convergence_stats = None

opts_casadi = {
    'expand': True, 
    'detect_simple_bounds': True,
    'structure_detection': 'auto'
}
opts_fatrop = {
    'tol': 1e-5,
    'mu_init': 0.1,
    'max_iter': 300,
    'problem_type': problem_type,
}
if problem_type == 'accelerated_ocp_type':
    opts_fatrop['linsol_nb_of_dynamics_constraints'] = reformulator.number_of_constraints_with_zk
    opts_fatrop['linsol_nb_of_zk_vars'] = zk.shape[0]

opti.solver('fatrop', opts_casadi, opts_fatrop)

if save_opti_f:
    print(f"creating opti to function object")
    opti_f = opti.to_function(
        f'test_gait_shortcut_python_reformulated_{problem_type}', 
        [], [hcat(xx), hcat(uu)], [], ['xx', 'uu'])
    
    if False and config['perform_code_generation']: ### disables since the problem is too big to be code generated
        from code_generator import get_code_generated_function
        print(f"code generating opti_f")
        opti_f_cg = get_code_generated_function(opti_f) # will save automatically
    else:
        print(f"saving opti to function object")
        opti_f.save(f'casadi_funcs/test_gait_shortcut_python_reformulated_{problem_type}.casadi')

if solve and not load_solution:
    print(f"solving")
    
    if config['load_opti_f']:
        opti_f = Function.load(f'casadi_funcs/test_gait_shortcut_python_reformulated_{problem_type}.casadi')
        
        if config['log_convergence']:
            solver_output_file = f'stored_solutions/solver_output_{problem_type}{config["file_name_appendix"]}.txt'
            with open(solver_output_file, 'w') as f:
                old_stdout = sys.stdout
                sys.stdout = f
                try:
                    xx_sol, uu_sol = opti_f()
                    stats = opti_f.stats()
                finally:
                    sys.stdout = old_stdout
        else:
            xx_sol, uu_sol = opti_f()
            stats = opti_f.stats()
        
    elif save_opti_f:
        if config['log_convergence']:
            solver_output_file = f'stored_solutions/solver_output_{problem_type}{config["file_name_appendix"]}.txt'
            with open(solver_output_file, 'w') as f:
                old_stdout = sys.stdout
                sys.stdout = f
                try:
                    xx_sol, uu_sol = opti_f()
                    stats = opti_f.stats()
                finally:
                    sys.stdout = old_stdout
        else:
            xx_sol, uu_sol = opti_f()
            stats = opti_f.stats()
    else:
        if config['log_convergence']:
            solver_output_file = f'stored_solutions/solver_output_{problem_type}{config["file_name_appendix"]}.txt'
            with open(solver_output_file, 'w') as f:
                old_stdout = sys.stdout
                sys.stdout = f
                try:
                    sol = opti.solve()
                    xx_sol = sol.value(hcat(xx))
                    uu_sol = sol.value(hcat(uu))
                    stats = sol.stats()
                finally:
                    sys.stdout = old_stdout
        else:
            sol = opti.solve()
            xx_sol = sol.value(hcat(xx))
            uu_sol = sol.value(hcat(uu))
            stats = sol.stats()
    
    if store_solution:
        with open(f'stored_solutions/solution_gait_shortcut_reformulated_{problem_type}{config["file_name_appendix"]}.pkl', 'wb') as f:
            pkl.dump({'xx_sol': xx_sol, 'uu_sol': uu_sol, 'stats': stats}, f)
    
    if config['log_convergence']:
        print(f"Solver output saved to: {solver_output_file}")
            
    nb_iterations = stats['fatrop']['iterations_count']
    print(f"Search dir computation time per iteration: {stats['fatrop']['compute_sd_time']/nb_iterations*1000:.4f} ms")
    print(f"other time per iteration: {stats['fatrop']['time_total']/nb_iterations*1000:.4f} ms")
    
    # Parse convergence data from solver output if needed
    if config['log_convergence']:
        from post_process_test_gait import load_and_parse_convergence
        convergence_pkl = f'stored_solutions/convergence_gait_shortcut_reformulated_{problem_type}{config["file_name_appendix"]}.pkl'
        convergence_stats = load_and_parse_convergence(solver_output_file, convergence_pkl)
            
if load_solution:
    print(f"loading solution")
    with open(f'stored_solutions/solution_gait_shortcut_reformulated_{problem_type}{config["file_name_appendix"]}.pkl', 'rb') as f:
        data = pkl.load(f)
        xx_sol = data['xx_sol']
        uu_sol = data['uu_sol']
        stats = data['stats']

if (solve or load_solution) and visualize_solution:   
    q_mesh = xx_sol[:n_coords, :]
    qdot_mesh = xx_sol[n_coords:2*n_coords, :]

    step_time = 1.0 / 0.9 / 2.0
    step_length = 1.2 * step_time
        
    # joints:
    # pelvis_tilt, pelvis_tx, pelvis_ty, hip_r, hip_l, knee_r, knee_l,
    # ankle_r, ankle_l
    # k = 0         k = N
    # ------------------------
    # pelvis_tx     pelvis_tx
    # pelvis_ty     pelvis_ty
    # hip_r         hip_l
    # hip_l         hip_r
    # knee_r        knee_l
    # knee_l        knee_r
    second_period_copy_idxs = [0, 1, 2, 4, 3, 6, 5, 8, 7] # swap left and right leg joints for the second half of the gait cycle
    
    # Construct full cycle from solution for half cycle
    q_GC = np.hstack([q_mesh[:, :], q_mesh[second_period_copy_idxs,1:]])
    q_GC[1, N+1:] = q_GC[1, N+1:] + step_length

    t_GC = np.linspace(0, 2*dt*N, 2*N+1)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Walking Gait Simulation Results')

    # Torso Angle
    axs[0, 0].plot(t_GC, q_GC[0, :] * 180 / np.pi)
    axs[0, 0].set_title('Torso Angle')
    axs[0, 0].set_xlabel('Time (s)')
    axs[0, 0].set_ylabel('Angle (°)')

    # Forward Position
    axs[0, 1].plot(t_GC, q_GC[1, :])
    axs[0, 1].set_title('Forward Position')
    axs[0, 1].set_xlabel('Time (s)')
    axs[0, 1].set_ylabel('Position (m)')

    # Vertical Position
    axs[0, 2].plot(t_GC, q_GC[2, :])
    axs[0, 2].set_title('Vertical Position')
    axs[0, 2].set_xlabel('Time (s)')
    axs[0, 2].set_ylabel('Position (m)')

    # Hip Angle
    axs[1, 0].plot(t_GC, q_GC[4, :] * 180 / np.pi)
    axs[1, 0].set_title('Hip Angle')
    axs[1, 0].set_xlabel('Time (s)')
    axs[1, 0].set_ylabel('Angle (°)')

    # Knee Angle
    axs[1, 1].plot(t_GC, q_GC[6, :] * 180 / np.pi)
    axs[1, 1].set_title('Knee Angle')
    axs[1, 1].set_xlabel('Time (s)')
    axs[1, 1].set_ylabel('Angle (°)')

    # Ankle Angle
    axs[1, 2].plot(t_GC, q_GC[8, :] * 180 / np.pi)
    axs[1, 2].set_title('Ankle Angle')
    axs[1, 2].set_xlabel('Time (s)')
    axs[1, 2].set_ylabel('Angle (°)')
    

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()

    def show_skeleton(ax, q):
        # segment lengths
        L_thigh = 0.44
        L_shank = 0.45

        # Pelvis
        pelvis_tx = q[1]
        pelvis_ty = q[2]
        pelvis_tilt = q[0]
        
        # Joint angles
        hip_r_angle = q[3]
        hip_l_angle = q[4]
        knee_r_angle = q[5]
        knee_l_angle = q[6]
        ankle_r_angle = q[7]
        ankle_l_angle = q[8]

        # Joint positions
        pelvis_pos = np.array([pelvis_tx, pelvis_ty])
        
        # Hip positions (relative to pelvis)
        hip_r_pos = pelvis_pos
        hip_l_pos = pelvis_pos
        
        # Knee positions
        knee_r_pos = hip_r_pos + L_thigh * np.array([np.sin(pelvis_tilt + hip_r_angle), -np.cos(pelvis_tilt + hip_r_angle)])
        knee_l_pos = hip_l_pos + L_thigh * np.array([np.sin(pelvis_tilt + hip_l_angle), -np.cos(pelvis_tilt + hip_l_angle)])
        
        # Ankle positions
        ankle_r_pos = knee_r_pos + L_shank * np.array([np.sin(pelvis_tilt + hip_r_angle + knee_r_angle), -np.cos(pelvis_tilt + hip_r_angle + knee_r_angle)])
        ankle_l_pos = knee_l_pos + L_shank * np.array([np.sin(pelvis_tilt + hip_l_angle + knee_l_angle), -np.cos(pelvis_tilt + hip_l_angle + knee_l_angle)])

        # Plot skeleton
        ax.plot([hip_r_pos[0], knee_r_pos[0]], [hip_r_pos[1], knee_r_pos[1]], 'r-') # Right thigh
        ax.plot([hip_l_pos[0], knee_l_pos[0]], [hip_l_pos[1], knee_l_pos[1]], 'b-') # Left thigh
        ax.plot([knee_r_pos[0], ankle_r_pos[0]], [knee_r_pos[1], ankle_r_pos[1]], 'r-') # Right shank
        ax.plot([knee_l_pos[0], ankle_l_pos[0]], [knee_l_pos[1], ankle_l_pos[1]], 'b-') # Left shank

        # Plot joints
        ax.plot(pelvis_pos[0], pelvis_pos[1], 'ko', markersize=10, label='Pelvis')
        ax.plot(knee_r_pos[0], knee_r_pos[1], 'ro', markersize=5, label='Right Knee')
        ax.plot(knee_l_pos[0], knee_l_pos[1], 'bo', markersize=5, label='Left Knee')
        ax.plot(ankle_r_pos[0], ankle_r_pos[1], 'ro', markersize=5, label='Right Ankle')
        ax.plot(ankle_l_pos[0], ankle_l_pos[1], 'bo', markersize=5, label='Left Ankle')
        
        # ax.legend()
        return pelvis_pos[0] # return forward position of the pelvis for setting x limits in animation
        
    # animate skeleton
    fig = plt.figure()
    ax = fig.add_subplot(111)
    def update(frame):
        ax.clear()
        x_pos = show_skeleton(ax, q_GC[:, frame])
        ax.set_xlim(x_pos - 0.5, x_pos + 1)
        ax.set_ylim(-0.5, 1.5)
        ax.axhline(0, color='k', linestyle='-', lw=2)
        # show small lines at distance half steplength to highlight motion
        for k in range(-5, 5):
            ax.plot([k*step_length/2, k*step_length/2], 
                    [0, -0.01], color='k', linestyle='-', alpha=1)
        
        ax.set_aspect('equal')
        ax.set_xlabel('Forward Position (m)')
        ax.set_ylabel('Vertical Position (m)')

    from matplotlib.animation import FuncAnimation
    nb_frames = q_GC.shape[1]
    fps = 1/dt
    ani = FuncAnimation(fig, update, frames=nb_frames, repeat=True)
    ani.save(f"visualization_output/walking_gait_skeleton_{problem_type}.gif", writer='pillow', fps=30)