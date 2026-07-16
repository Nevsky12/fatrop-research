import casadi as ca
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import example_robot_data as erd

# -----------------------------
# 1) Load robot model
# -----------------------------
robot = erd.load("anymal")
model = robot.model
data = robot.data

nq = model.nq       # e.g., 19 (floating base + 12 joints)
nv = model.nv       # 18
na = nv - 6         # number of actuated joints

# Foot frames (linear contact)
foot_frames = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
foot_ids = [model.getFrameId(f) for f in foot_frames]
m = 3 * len(foot_ids)  # total contact constraints

# -----------------------------
# 2) CasADi symbolic variables
# -----------------------------
q = ca.SX.sym("q", nq)
v = ca.SX.sym("v", nv)
tau_j = ca.SX.sym("tau", na)          # joint torques
tau = ca.vertcat(ca.DM.zeros(6), tau_j)  # full nv torque vector

# -----------------------------
# 3) Symbolic dynamics function
# -----------------------------
def casadi_forward_dynamics(q_sym, v_sym, tau_sym):
    """
    Returns qdd, lambda (contact forces) symbolically in CasADi SX
    """
    # 1. Mass matrix
    M = pin.crba(model, data, q_sym)    # nv x nv
    M = ca.DM(M)

    # 2. Nonlinear effects
    h = pin.rnea(model, data, q_sym, v_sym, ca.DM.zeros(nv))
    h = ca.DM(h)

    # 3. Build stacked contact Jacobian Jc and Jc_dot_v
    J_list = []
    Jdotv_list = []
    pin.framesForwardKinematics(model, data, q_sym)
    pin.updateFramePlacements(model, data)

    for fid in foot_ids:
        J6 = pin.computeFrameJacobian(model, data, q_sym, fid, pin.LOCAL_WORLD_ALIGNED)
        J_lin = J6[:3, :]
        J_list.append(J_lin)

        # Jdot * v
        try:
            Jdot6 = pin.frameJacobianTimeVariation(model, data, q_sym, v_sym, fid, pin.LOCAL_WORLD_ALIGNED)
            Jdot_lin = Jdot6[:3, :]
        except Exception:
            data2 = model.createData()
            pin.computeJointJacobiansTimeVariation(model, data2, q_sym, v_sym)
            Jdot6 = pin.getFrameJacobianTimeVariation(model, data2, fid, pin.LOCAL_WORLD_ALIGNED)
            Jdot_lin = Jdot6[:3, :]
        Jdotv_list.append(Jdot_lin @ v_sym)

    Jc = ca.vertcat(*J_list)             # m x nv
    Jc_dot_v = ca.vertcat(*Jdotv_list)  # m x 1

    # 4. Solve constrained dynamics using Schur complement
    # M qdd + h = tau + Jc^T lambda
    # Jc qdd + Jdot v = 0

    # Schur complement: lambda = (Jc M^-1 Jc^T)^-1 * (Jc M^-1 (tau - h) + Jdotv)
    Minv_tau_h = ca.solve(M, tau_sym - h)       # nv x 1
    Minv_JcT = ca.solve(M, Jc.T)                # nv x m
    A = Jc @ Minv_JcT                            # m x m
    b = Jc @ Minv_tau_h + Jc_dot_v              # m x 1

    # Regularize for numerical stability
    reg = 1e-8
    A += reg * ca.DM_eye(m)

    lam = ca.solve(A, b)                         # contact forces
    qdd = ca.solve(M, tau_sym - h - Jc.T @ lam) # nv x 1

    return qdd, lam

# Symbolic CasADi function
qdd, lam = casadi_forward_dynamics(q, v, tau)
dyn_fun = ca.Function("dyn_fun", [q, v, tau_j], [qdd, lam])

# -----------------------------
# 4) Example: direct transcription OCP
# -----------------------------
# time horizon
T = 0.5     # seconds
N = 20      # timesteps
dt = T/N

# initial state
q0 = pin.neutral(model)
v0 = np.zeros(nv)

# optimization variables
w = []
w0 = []
lbw = []
ubw = []

g = []
lbg = []
ubg = []

xk = ca.vertcat(q0, v0)

for k in range(N):
    # 1. add control
    tauk = ca.SX.sym(f"tau_{k}", na)
    w.append(tauk)
    w0.extend([0]*na)
    lbw.extend([-50]*na)
    ubw.extend([50]*na)

    # 2. integrate dynamics (Euler)
    qk = xk[:nq]
    vk = xk[nq:]
    qdd_k, lam_k = dyn_fun(qk, vk, tauk)
    x_next = xk + dt * ca.vertcat(vk, qdd_k)

    # 3. enforce state continuity
    xk_next = ca.SX.sym(f"x_{k}", nq+nv)
    w.append(xk_next)
    w0.extend(list(q0)+[0]*nv)
    lbw.extend([-ca.inf]*(nq+nv))
    ubw.extend([ca.inf]*(nq+nv))

    g.append(xk_next - x_next)
    lbg.extend([0]*(nq+nv))
    ubg.extend([0]*(nq+nv))

    xk = xk_next

# -----------------------------
# 5) Define simple cost
# -----------------------------
J = 0
for k in range(N):
    tauk = w[2*k]  # tau alternates with xk_next in w list
    J += ca.sumsqr(tauk)  # minimize torque effort

# -----------------------------
# 6) Solve with IPOPT
# -----------------------------
prob = {'f': J, 'x': ca.vertcat(*w), 'g': ca.vertcat(*g)}
opts = {'ipopt.print_level':0, 'print_time':0}
solver = ca.nlpsol('solver', 'ipopt', prob, opts)

sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
w_opt = sol['x'].full().flatten()

# Extract solution
tau_opt = w_opt[::2*(na+nv)][:N]  # depends on stacking order
print("Optimized torques (first timestep):", tau_opt[:na])
