import casadi as ca
import numpy as np

# load foward kinematics
fk = ca.Function.load("panda_fk.casadi")
fkpos_ee = ca.Function.load("panda_fkpos_ee.casadi")

# load dynamics
fd = ca.Function.load("panda_fd.casadi")
id = ca.Function.load("panda_id.casadi")

# define symbols
q = ca.SX.sym("q", 7)
qd = ca.SX.sym("qd", 7)
qdd = ca.SX.sym("qdd", 7)
floor_height = ca.SX.sym("floor_height", 1)
k = ca.SX.sym("k", 1)
alpha = ca.SX.sym("alpha", 1)
d = ca.SX.sym("d", 1)

ee_pos = fk(q)[-1][:3, 3]
ee_pos_x = ee_pos[0]
ee_pos_y = ee_pos[1]
ee_pos_z = ee_pos[2]
ee_vel = ca.jacobian(fk(q)[-1][:3, 3], q) @ qd
ee_vel_z = ee_vel[2]

penetration_depth = floor_height - ee_pos_z + 0.0*ee_pos_y
penetration_velocity = -ee_vel_z
normal_force = k*ca.exp(alpha*penetration_depth) + d * penetration_velocity /(1 + ca.exp(-alpha*penetration_depth))
normal_force_func = ca.Function("normal_force_func", [q, qd, floor_height, k, alpha, d], [normal_force], 
                                    ["q", "qd", "floor_height", "k", "alpha", "d"], ["normal_force"])
normal_force_func.save("panda_normal_force.casadi")

# construct force jacobian of end effector
J = ca.jacobian(fkpos_ee(q), q)

# update dynamics with contact force tau = tau_original + J^T * [0, 0, normal_force]
inverse_dynamics_contact = ca.Function("inverse_dynamics_contact", 
                                    [q, qd, qdd, floor_height, k, alpha, d], 
                                    [id(q, qd, qdd) + J.T @ ca.vertcat(0, 0, normal_force)],
                                    ["q", "qd", "qdd", "floor_height", "k", "alpha", "d"],
                                    ["tau_contact"])
print("inverse_dynamics_contact:", inverse_dynamics_contact)
inverse_dynamics_contact.save("panda_id_contact.casadi")

tau = ca.SX.sym("tau", 7)
tau_extended = tau + J.T @ ca.vertcat(0, 0, normal_force)
# tau_extended = tau
foward_dynamics_contact = ca.Function("forward_dynamics_contact",
                                    [q, qd, tau, floor_height, k, alpha, d], 
                                    [fd(q, qd, tau_extended)],
                                    ["q", "qd", "tau", "floor_height", "k", "alpha", "d"],
                                    ["qdd_contact"])
print("forward_dynamics_contact:", foward_dynamics_contact)
foward_dynamics_contact.save("panda_fd_contact.casadi")