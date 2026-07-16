import casadi as ca
import numpy as np

# load foward kinematics
fk = ca.Function.load("panda_fk.casadi")

# define eval_vb
q = ca.SX.sym("q", 7)
dq = ca.SX.sym("dq", 7)
T = fk(q)[-1]
R = T[:3, :3]
p = T[:3, 3]

Jv = ca.jacobian(p, q)
v = ca.mtimes(Jv, dq)

def skew_to_vec(S):
    return ca.vertcat(S[2,1], S[0,2], S[1,0])

# Rdot = ca.jacobian(R, q) @ dq
Rdot = ca.jacobian(ca.reshape(R, 9, 1), q) @ dq
Rdot = ca.reshape(Rdot, 3, 3)
S_omega = R.T @ Rdot
omega = skew_to_vec(S_omega)

eval_vb = ca.Function("eval_vb", [q, dq], [ca.vertcat(v, omega)])
eval_vb.save("panda_eval_vb.casadi")

# define eval_ab
ddq = ca.SX.sym("ddq", 7)
# Jv_dot = ca.jacobian(Jv, q) @ dq
Jv_dot = ca.jacobian(ca.reshape(Jv, 21, 1), q) @ dq
Jv_dot = ca.reshape(Jv_dot, 3, 7)
v_dot = Jv @ ddq + Jv_dot @ dq

omega_dot = ca.jacobian(omega, q) @ dq + ca.jacobian(omega, dq) @ ddq
eval_ab = ca.Function("eval_ab", [q, dq, ddq], [ca.vertcat(v_dot, omega_dot)])
eval_ab.save("panda_eval_ab.casadi")

print("eval_vb:", eval_vb)
print("eval_ab:", eval_ab)


# now define the beam model
dpee = eval_vb(q, dq)
ddpee = eval_ab(q, dq, ddq)

dpee_v, dpee_w = dpee[:3], dpee[3:]
ddpee_v = ddpee[:3]
dwee = ddpee[3:]

g0 = 9.81
g = ca.vertcat(0, 0, -g0)

wn = ca.SX.sym("wn", 1)
zeta = ca.SX.sym("zeta", 1)
L = ca.SX.sym("L", 1)

theta = ca.SX.sym("theta", 1)
dtheta = ca.SX.sym("dtheta", 1)
S_w = ca.skew(dpee_w)
S_dw = ca.skew(dwee)

def rotz(q):
    c = np.cos(q); s = np.sin(q)
    result = ca.SX.zeros(3,3)
    result[0,0] = c; result[0,1] = -s
    result[1,0] = s; result[1,1] = c
    result[2,2] = 1.
    return result
def rotz_casadi(q):
    return ca.SX(rotz(q))

def drotz(q):
    c = np.cos(q); s = np.sin(q)
    result = ca.SX.zeros(3,3)
    result[0,0] = -s; result[0,1] = -c
    result[1,0] = c; result[1,1] = -s
    result[2,2] = 1.
    return result
    # return np.array(([-s, -c, 0.],
    #                  [c, -s, 0.],
    #                  [0., 0., 1.]))
def drotz_casadi(q):
    return ca.SX(drotz(q))

k = np.array([[1, 0, 0]]).T
ddtheta = (-2*zeta*wn*dtheta - wn**2*theta + 1/L*k.T @ drotz_casadi(theta).T @ R.T @ (g - ddpee_v)
          - k.T @ drotz_casadi(theta).T @ R.T @ S_w @ S_w @ R @ rotz_casadi(theta) @ k
          - k.T @ drotz_casadi(theta).T @ R.T @ S_dw @ R @ rotz_casadi(theta) @ k)

beam_model = ca.Function("beam_model", [q, dq, ddq, wn, zeta, L, theta, dtheta], [ddtheta], ["q", "dq", "ddq", "wn", "zeta", "L", "theta", "dtheta"], ["ddtheta"])
beam_model.save("panda_beam_model.casadi")

# beam and pos
beam_end_pos = p + R @ rotz_casadi(theta) @ k*L
beam_end_pos_func = ca.Function("beam_end_pos", [q, theta, L], [beam_end_pos], ["q", "theta", "L"], ["beam_end_pos"])
beam_end_pos_func.save("panda_beam_end_pos.casadi")