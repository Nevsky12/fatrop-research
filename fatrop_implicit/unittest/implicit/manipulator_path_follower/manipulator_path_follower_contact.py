from casadi import *
import numpy as np

folder = 'panda/'
# folder = "unittest/implicit/manipulator_path_follower/"

### define helper functions
def get_all_functions():
    functions = {"fk":None, "fd":None, "fk_ee":None, "fkpos_ee":None, "fkrot_ee":None, "id":None, "fd_contact":None, "id_contact":None, "normal_force":None}
    for f in functions.keys():
        functions[f] = Function.load(f"{folder}panda_{f}.casadi")

    return functions

def get_full_equilibrium(functions, q0):
    opti = Opti()
    tau_eq = opti.variable(7)

    q_eq = DM(q0)
    qd_eq = DM.zeros(7)
    qdd_eq = DM.zeros(7)

    opti.subject_to(tau_eq == functions['id'](q_eq, qd_eq, qdd_eq))

    opti.minimize(sumsqr(tau_eq))
    opti.solver('ipopt', {}, {})
    sol = opti.solve()
    print("Equilibrium tau:", sol.value(tau_eq))
    return sol.value(tau_eq)

def get_path_function(functions, q0, floor_height):
    # t = MX.sym("t", 1)
    # q = MX.zeros(7, 1)
    # for i in range(7):
    #     q[i] = q0[i] + 0.1*(1-cos(t*(0.5*i+1)))
    #     # q[i] = q0[i] + (i == 0)*t*0.1

    # ee_pos = functions['fkpos_ee'](q)
    # ee_pos[2] = floor_height
    # return Function("eval_path", [t], [ee_pos])

    t = MX.sym("t", 1)
    circle_radius = 0.1
    ee_end_pos = functions['fkpos_ee'](q0)
    center = vertcat(ee_end_pos[0] + circle_radius, ee_end_pos[1], floor_height - 0.01)
    return Function("eval_path", [t], 
                    [center + vertcat(
                        -circle_radius*cos(2*np.pi*t/T), 
                        circle_radius*sin(4*np.pi*t/T), 
                        0)])



### Main
# define parameters
T = 1.5
functions = get_all_functions()
q0 = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4]).T
floor_height = functions['fkpos_ee'](q0)[2]-0.05

tau_eq = get_full_equilibrium(functions, q0)
path_func = get_path_function(functions, q0, floor_height)

opti = Opti()
N = 30
dt = T/N
floor_k = 1000
floor_alpha = 1000
floor_d = 10
tau_bound = [87, 87, 87, 87, 12, 12, 12]
fn0 = float(functions['normal_force'](q0, DM.zeros(7,1), floor_height, floor_k, floor_alpha, floor_d))
force_min = -1 - 1000
force_max = 200 + 1000
desired_force = 50
path_tol = 0.01
control_penalty = 1.0e-2
n = 7

with_slack = 1
with_progress_variable = 1
with_accel_variables = 0
with_implicit_integrator = 1
with_perpendicular_constraint = 0
with_force_variables = 0

# define variables
q = []; qd = []; qdd = []; tau = []; s = []
p = []; dp = []
fn = []
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
        s.append(opti.variable(3 + 1*with_perpendicular_constraint))
    if with_force_variables:
        fn.append(opti.variable(1))
q.append(opti.variable(n))
qd.append(opti.variable(n))
p.append(opti.variable(1))

q = hcat(q); qd = hcat(qd)
if with_accel_variables:
    qdd = hcat(qdd)
tau = hcat(tau)
if with_slack:
    s = hcat(s)
else:
    s = MX.zeros(3 + with_perpendicular_constraint, N)
if with_progress_variable:
    p = hcat(p)
    dp = hcat(dp)
else:
    p = transpose(linspace(0, T, N+1))
    dp = MX.ones(1, N)
if with_force_variables:
    fn = hcat(fn)
else:
    fn = MX.zeros(1,N)

# initial state
opti.subject_to(q[:, 0] == q0)
opti.subject_to(qd[:, 0] == 0)
if with_progress_variable:
    opti.subject_to(p[:,0] == 0)

obj = 0.0
for k in range(N):
    # dymamics
    rhs_idx = k if not with_implicit_integrator else k+1
    opti.subject_to(q[:,k+1] == q[:,k] + qd[:,rhs_idx]*dp[:,k])
    if with_accel_variables:
        opti.subject_to(qd[:,k+1] == qd[:,k] + qdd[:,k]*dp[:,k])
    else:
        qdd = functions['fd_contact'](q[:,rhs_idx], qd[:,rhs_idx], tau[:,k], floor_height, floor_k, floor_alpha, floor_d)
        opti.subject_to(qd[:,k+1] == qd[:,k] + qdd*dp[:,k])
    
    if with_progress_variable:
        opti.subject_to(p[:,k+1] == p[:,k] + dp[:,k])

    # slack constraints
    ee_pos_k = path_func(p[:,k])
    if with_slack:
        opti.subject_to(functions['fkpos_ee'](q[:,k]) - ee_pos_k - s[:3,k] == 0)
        if with_perpendicular_constraint:
            z_axes = functions['fkrot_ee'](q[:,k]) @ vertcat(0, 0, 1)
            opti.subject_to(s[3,k] == 1 - mtimes(transpose(z_axes), vertcat(0, 0, -1)))
    else:
        s[:3,k] = functions['fkpos_ee'](q[:,k]) - ee_pos_k
    
    # true dynamics
    if with_accel_variables:
        opti.subject_to(tau[:,k] == functions['id_contact'](q[:,rhs_idx], qd[:,rhs_idx], qdd[:,k], floor_height, floor_k, floor_alpha, floor_d))

    if with_force_variables:
        opti.subject_to(fn[:,k] == functions['normal_force'](q[:,k], qd[:,k], floor_height, floor_k, floor_alpha, floor_d))
    else:
        fn[:,k] = functions['normal_force'](q[:,k], qd[:,k], floor_height, floor_k, floor_alpha, floor_d)

    # inequalities
    # opti.subject_to(-1 <= (sumsqr(s[:,k]) <= path_tol**2))
    if with_perpendicular_constraint:
        opti.subject_to(-1 <= (sumsqr(s[3,k]) <= path_tol**2))
    for i in range(n):
        opti.subject_to(-tau_bound[i] <= (tau[i,k] <= tau_bound[i]))
    opti.subject_to(force_min <= (fn[:,k] <= force_max))

    if with_progress_variable:
        opti.subject_to(0.2*dt <= (dp[:,k] <= 2.0*dt))

    obj += 0*sumsqr(s[:,k]) + control_penalty*sumsqr(tau[:,k]) + 1e-4*sumsqr(dp[:,k] - dt) + 0*sumsqr(tau[:,k] - tau_eq)
    if with_accel_variables:
        obj += sumsqr(qdd[:,k])
    # obj += 1e1*sumsqr(fn[:,k] - desired_force)
    obj += 1e2*sumsqr(s[:3,k])

opti.subject_to(q[:,-1] == q0)
opti.subject_to(qd[:,-1] == 0)

if with_progress_variable:
    opti.subject_to(p[-1,-1] == T)
    opti.set_initial(p, transpose(linspace(0, T, N+1)))
    opti.set_initial(dp, dt)

for k in range(N):
    opti.set_initial(q[:,k], q0)
    opti.set_initial(tau[:,k], tau_eq)

opti.minimize(obj)
# opti.solver('fatrop', {'structure_detection':'auto', 'debug':True}, {'tolerance':1e-4})
opti.solver('ipopt', {}, {'tol':1e-4})

sol = opti.solve()

def DM_to_json_serializable(var):
    sol_val = sol.value(var)
    if isinstance(sol_val, np.ndarray):
        return sol_val.tolist()
    elif isinstance(sol_val, DM):
        return sol_val.full().tolist()

solution = {
    "q": DM_to_json_serializable(q),
    "qd": DM_to_json_serializable(qd),
    "qdd": DM_to_json_serializable(qdd) if with_accel_variables else None,
    "tau": DM_to_json_serializable(tau),
    "s": DM_to_json_serializable(s),
    "p": DM_to_json_serializable(p) if with_progress_variable else None,
    "dp": DM_to_json_serializable(dp) if with_progress_variable else None,
    "N": N,
    "T": T,
    "k": floor_k,
    "alpha": floor_alpha,
    "d": floor_d,
    "force_min": force_min,
    "force_max": force_max,
    "floor_height": float(floor_height),
    "implicit_integrator": with_implicit_integrator,
}
solution_filename = f"manipulator_path_follower_contact_solution.json"
import json
with open(solution_filename, 'w') as f:
    json.dump(solution, f, indent=4)

path_func.save(f"path_func_contact.casadi")