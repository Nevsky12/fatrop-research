import casadi
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import example_robot_data as erd
from pinocchio.visualize import MeshcatVisualizer

def make_quadruped():
    robot = erd.load("anymal")
    model = robot.model

    return model, robot.collision_model, robot.visual_model

def sigmoid(x):
    try:
        if x >= 0:
            return 1 / (1 + casadi.exp(-x))
        else:
            ex = casadi.exp(x)
            return ex / (1 + ex)
    except:
        return 1 / (1 + np.exp(-x))
        
def ground_reaction_force(p, v):
    # Ground stiffness and damping parameters
    k_ground = 80  # N/m
    d_ground = 100  # N·s/m
    alpha_k = 100
    alpha_d = 100

    penetration = -p[2]
    fz = k_ground * np.exp(alpha_k*penetration) - d_ground * sigmoid(alpha_d*penetration)*v[2]
    fx = -d_ground * sigmoid(alpha_d*penetration)*v[0]
    fy = -d_ground * sigmoid(alpha_d*penetration)*v[1]
    return np.array([fx, fy, fz])

class PinocchioCasadi:
    """Take a Pinocchio model, turn it into a Casadi model
    and define the appropriate graphs.
    """

    def __init__(self, model: pin.Model, timestep=0.05):
        self.model = model
        self.cmodel = cpin.Model(model)  # cast to CasADi model
        self.cdata = self.cmodel.createData()
        self.timestep = timestep
        self.create_dynamics()
        self.create_discrete_dynamics()

    def create_dynamics(self):
        """Create the acceleration expression and acceleration function."""
        nq = self.model.nq
        nu = 12 # HAA, HFE, KFE for 4 legs (LF, LH, RF, RH)
        nv = self.model.nv
        q = casadi.SX.sym("q", nq)
        v = casadi.SX.sym("v", nv)
        u = casadi.SX.sym("u", nu)
        dq_ = casadi.SX.sym("dq_", nv)
        self.u_node = u
        self.q_node = q
        self.v_node = v
        self.dq_ = dq_

        # actuation
        B = np.zeros((18, 12))
        B[6:,:] = np.eye(12)
        tau = B @ u

        # get foot positions and velocities
        cpin.forwardKinematics(self.cmodel, self.cdata, q, v)
        cpin.updateFramePlacements(self.cmodel, self.cdata)
        foot_names = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
        foot_ids = [self.model.getFrameId(name) for name in foot_names]

        tau_contacts = 0
        temp = 0
        for fid in foot_ids:
            # Foot position and velocity (in world frame)
            p = self.cdata.oMf[fid].translation
            v_frame = cpin.getFrameVelocity(self.cmodel, self.cdata, fid, pin.LOCAL_WORLD_ALIGNED)
            v_foot = v_frame.linear

            f_contact = ground_reaction_force(p, v_foot)

            # Foot Jacobian (LOCAL_WORLD_ALIGNED)
            J = cpin.computeFrameJacobian(self.cmodel, self.cdata, q, fid, pin.LOCAL_WORLD_ALIGNED)
            Jlin = J[:3, :]  # linear part
            tau_contacts += Jlin.T @ f_contact

        tau += tau_contacts

        a = cpin.aba(self.cmodel, self.cdata, q, v, tau)
        self.acc = a
        self.acc_func = casadi.Function("acc", [q, v, u], [a, temp], ["q", "v", "u"], ["a", "temp"])

    def create_discrete_dynamics(self):
        """
        Create the map `(q,v) -> (qnext, vnext)` using semi-implicit Euler integration.
        """
        q = self.q_node
        v = self.v_node
        u = self.u_node
        dq_ = self.dq_
        # q' = q + dq
        q_dq = cpin.integrate(self.cmodel, q, dq_)
        self.q_dq = q_dq
        # express acceleration using q' = q + dq
        a, temp = self.acc_func(q_dq, v, u)

        dt = self.timestep
        vnext = v + a * dt
        qnext = cpin.integrate(self.cmodel, self.q_dq, dt * vnext)

        self.dyn_qv_fn_ = casadi.Function(
            "discrete_dyn",
            [q, dq_, v, u],
            [qnext, vnext, temp],
            ["q", "dq_", "v", "u"],
            ["qnext", "vnext", "temp"],
        )

    def forward(self, x, u):
        nq = self.model.nq
        nv = self.model.nv
        q = x[:nq]
        v = x[nq:]
        dq_ = np.zeros(nv)
        qnext, vnext, temp = self.dyn_qv_fn_(q, dq_, v, u)
        # print("temp value:", temp)
        xnext = np.concatenate((qnext, vnext))
        return xnext

    def residual_fwd(self, x, u, xnext):
        nv = self.model.nv
        dq = np.zeros(nv)
        dqn = dq
        res = self.dyn_residual(x, u, xnext, dq, dqn)
        return res


class QuadrupedDynamics(PinocchioCasadi):
    def __init__(self, timestep=0.05):
        model, collision_model, visual_model = make_quadruped()
        self.collision_model = collision_model
        self.visual_model = visual_model
        super().__init__(model=model, timestep=timestep)


dt = 0.0002
quad = QuadrupedDynamics(timestep=dt)
model = quad.model

print(model)

q0 = model.referenceConfigurations["standing"]
q0 = pin.normalize(model, q0)
q0[2] += 0.5
v = np.zeros(model.nv)
u = np.zeros(1)
a0 = quad.acc_func(q0, v, u)

print("a0:", a0)

x0 = np.append(q0, v)
xnext = quad.forward(x0, u)


def integrate_no_control(x0, nsteps):
    states_ = [x0.copy()]
    for t in range(nsteps):
        u = np.zeros(12)
        u[0] = 0
        u[3 + 1] = 0
        u[6 + 2] = 0
        xnext = quad.forward(states_[t], u).ravel()
        states_.append(xnext)
    return states_


states_ = integrate_no_control(x0, nsteps=4000)
states_ = np.stack(states_).T

# print out all states
# print("states_:")
# for i in range(states_.shape[1]):
#     print(f"t={i*dt:.2f}s: q={states_[:model.nq, i]}, v={states_[model.nq:, i]}")

try:
    viz = MeshcatVisualizer(
        model=model,
        collision_model=quad.collision_model,
        visual_model=quad.visual_model,
    )

    viz.initViewer()
    viz.loadViewerModel("pinocchio")

    qs_ = states_[: model.nq, :].T
    viz.play(q_trajectory=qs_, dt=dt)
except ImportError as err:
    print(
        "Error while initializing the viewer. "
        "It seems you should install Python meshcat"
    )
    print(err)
    sys.exit(0)
