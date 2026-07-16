import casadi
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import example_robot_data as erd
from pinocchio.visualize import MeshcatVisualizer
import json

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
    # return np.array([0, 0, 0])
    # Ground stiffness and damping parameters
    k_ground = 200  # N/m
    d_ground = 600  # N·s/m
    alpha_k = 40
    alpha_d = 40
    # k_ground = 0
    # d_ground = 0
    # alpha_k = 0
    # alpha_d = 0

    penetration = -p[2]
    fx = -d_ground * sigmoid(alpha_d*penetration)*v[0]
    fy = -d_ground * sigmoid(alpha_d*penetration)*v[1]
    fz = k_ground * np.exp(alpha_k*penetration) - d_ground * sigmoid(alpha_d*penetration)*v[2]
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

    def get_explicit_integrator(self):
        q = self.q_node
        v = self.v_node
        u = self.u_node
        a, temp = self.acc_func(q, v, u)
        # a = casadi.vertcat(casadi.SX.zeros(2,1), -9.81, casadi.SX.zeros(self.model.nv-3,1))

        dt = self.timestep
        vnext = v + a * dt
        qnext = cpin.integrate(self.cmodel, q, dt * vnext)
        # qnext = q + casadi.vertcat(0, dt * vnext)

        x = casadi.vertcat(q, v)
        xnext = casadi.vertcat(qnext, vnext)

        explicit_integrator = casadi.Function(
            "quadruped_explicit_integrator",
            [u, x], [xnext]
        )       

        return explicit_integrator 

    def get_implicit_integrator(self):
        q = self.q_node
        v = self.v_node
        u = self.u_node
        qnext = casadi.SX.sym("qnext", self.model.nq)
        vnext = casadi.SX.sym("vnext", self.model.nv)
        xnext = casadi.vertcat(qnext, vnext)
        a_start, _ = self.acc_func(q, v, u)
        a_end, _ = self.acc_func(qnext, vnext, u)
        # a = 0.5 * (a_start + a_end)
        a = a_end

        dt = self.timestep
        x = casadi.vertcat(q, v)
        dyn_equations = \
            casadi.vertcat(cpin.integrate(self.cmodel, q, dt * vnext) - qnext,
                           v + a * dt - vnext)

        implicit_integrator = casadi.Function(
            "quadruped_implicit_integrator",
            [u, x, xnext], [dyn_equations]
        )        

        return implicit_integrator

    def create_discrete_dynamics(self):
        """
        Create the map `(q,v) -> (qnext, vnext)` using semi-implicit Euler integration.
        """
        q = self.q_node
        v = self.v_node
        u = self.u_node
        dq_ = self.dq_
        # q' = q + dq
        # q_dq = cpin.integrate(self.cmodel, q, dq_)
        q_dq = q
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

        explicit_integrator = self.get_explicit_integrator()
        implicit_integrator = self.get_implicit_integrator()

        return explicit_integrator, implicit_integrator

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

    def forward_casadi(self, x, u):
        nq = self.model.nq
        nv = self.model.nv
        q = x[:nq]
        v = x[nq:]
        dq_ = casadi.MX.zeros(nv)
        qnext, vnext, temp = self.dyn_qv_fn_(q, dq_, v, u)
        # print("temp value:", temp)
        xnext = casadi.vertcat(qnext, vnext)
        return xnext

    def residual_fwd(self, x, u, xnext):
        nv = self.model.nv
        dq = np.zeros(nv)
        dqn = dq
        res = self.dyn_residual(x, u, xnext, dq, dqn)
        return res

    def test_dynamics(self):
        q_test = [0, 0, 0.5292, 0, 0, 0, 1, -0.1, 0.7, -1, 0.1, 0.7, -1, -0.1, -0.7, -1, 0.1, -0.7, -1]
        v_test = np.zeros(self.model.nv)

        u = np.ones((12, 1))

        # evaluate forward
        x_test = np.concatenate((q_test, v_test))
        xnext = self.forward(x_test, u)

        print("xnext:", xnext)

class QuadrupedDynamics(PinocchioCasadi):
    def __init__(self, timestep=0.05):
        model, collision_model, visual_model = make_quadruped()
        self.collision_model = collision_model
        self.visual_model = visual_model
        super().__init__(model=model, timestep=timestep)




class ContactManipulator():
    def __init__(self, timestep=0.05):
        robot = erd.load("ur5")
        self.model = robot.model
        self.cmodel = cpin.Model(self.model)  # cast to CasADi model
        self.cdata = self.cmodel.createData()
        self.timestep = timestep
        self.create_dynamics()
        self.create_discrete_dynamics()

    def get_reaction_forces(self, p_ee, p_sphere, r_sphere, v_ee, v_sphere):
        # return casadi.SX.zeros(3)
        # extract positions and velocities
        px_ee = p_ee[0]; py_ee = p_ee[1]; pz_ee = p_ee[2]
        v_ee_norm = casadi.sqrt(v_ee[0]**2 + v_ee[1]**2 + v_ee[2]**2)
        px_sphere = p_sphere[0]
        py_sphere = p_sphere[1]
        pz_sphere = r_sphere

        # define contact force
        penetration = r_sphere - casadi.sqrt((px_ee - px_sphere)**2 + (py_ee - py_sphere)**2 + (pz_ee - pz_sphere)**2)
        penetration_velocity = -( (px_ee - px_sphere)*(v_ee[0] - v_sphere[0]) + (py_ee - py_sphere)*(v_ee[1] - v_sphere[1]) + (pz_ee - pz_sphere)*(v_ee[2] - v_sphere[2]) ) / casadi.sqrt((px_ee - px_sphere)**2 + (py_ee - py_sphere)**2 + (pz_ee - pz_sphere)**2)
        k_contact = 400  # N/m
        d_contact = 0
        alpha_k = 20
        alpha_d = 30
        f_contact_magnitude = k_contact * casadi.exp(alpha_k*penetration) - d_contact * sigmoid(alpha_d*penetration)*penetration_velocity

        # determine contact force vector
        f_on_ee = casadi.SX.zeros(3)
        f_on_ee[0] = f_contact_magnitude * (px_ee - px_sphere) / casadi.sqrt((px_ee - px_sphere)**2 + (py_ee - py_sphere)**2 + (pz_ee - pz_sphere)**2)
        f_on_ee[1] = f_contact_magnitude * (py_ee - py_sphere) / casadi.sqrt((px_ee - px_sphere)**2 + (py_ee - py_sphere)**2 + (pz_ee - pz_sphere)**2)
        f_on_ee[2] = f_contact_magnitude * (pz_ee - pz_sphere) / casadi.sqrt((px_ee - px_sphere)**2 + (py_ee - py_sphere)**2 + (pz_ee - pz_sphere)**2)

        return f_on_ee


    def create_dynamics(self):
        q = casadi.SX.sym("q", self.model.nq)
        v = casadi.SX.sym("v", self.model.nv)
        u = casadi.SX.sym("u", self.model.nv)

        # actuation
        tau = u

        # define spherical sliding object
        p_sphere = casadi.SX.sym("p_sphere", 2) # x, y position of the sphere on the table
        v_sphere = casadi.SX.sym("v_sphere", 2)
    
        r_sphere = 0.1
        m_sphere = 0.5

        # get end-effector position and velocity
        cpin.forwardKinematics(self.cmodel, self.cdata, q, v)
        cpin.updateFramePlacements(self.cmodel, self.cdata)
        ee_name = "ee_link"
        ee_id = self.model.getFrameId(ee_name)
        p_ee = self.cdata.oMf[ee_id].translation
        v_frame = cpin.getFrameVelocity(self.cmodel, self.cdata, ee_id, pin.LOCAL_WORLD_ALIGNED)
        v_ee = v_frame.linear
        J = cpin.computeFrameJacobian(self.cmodel, self.cdata, q, ee_id, pin.LOCAL_WORLD_ALIGNED)
        Jlin = J[:3, :]

        self.p_ee_func = casadi.Function("p_ee_func", [q, v], [p_ee], ["q", "v"], ["p_ee"])

        # contact forces
        f_on_ee = self.get_reaction_forces(p_ee, p_sphere, r_sphere, v_ee, casadi.vertcat(v_sphere, 0))
        tau_contact = Jlin.T @ f_on_ee
        tau += tau_contact

        # manipulator acceleration
        a_manipulator = cpin.aba(self.cmodel, self.cdata, q, v, tau)
        
        # sphere acceleration
        a_sphere = casadi.vertcat(-f_on_ee[0]/m_sphere, -f_on_ee[1]/m_sphere, 0)

        self.acc = casadi.Function("acc", [q, v, u, p_sphere, v_sphere], [a_manipulator, a_sphere], ["q", "v", "u", "p_sphere", "v_sphere"], ["a_manipulator", "a_sphere"])

    def create_explicit_dynamics(self):
        q = casadi.SX.sym("q", self.model.nq)
        v = casadi.SX.sym("v", self.model.nv)
        u = casadi.SX.sym("u", self.model.nv)
        p_sphere = casadi.SX.sym("p_sphere", 2)
        v_sphere = casadi.SX.sym("v_sphere", 2)

        a_manipulator, a_sphere = self.acc(q, v, u, p_sphere, v_sphere)
        dt = self.timestep

        vnext = v + a_manipulator * dt
        qnext = cpin.integrate(self.cmodel, q, dt * vnext)

        v_spherenext = v_sphere + a_sphere[:2] * dt
        p_spherenext = p_sphere + v_spherenext * dt

        x = casadi.vertcat(q, v, p_sphere, v_sphere)
        xnext = casadi.vertcat(qnext, vnext, p_spherenext, v_spherenext)

        self.explicit_integrator = casadi.Function(
            "manipulator_explicit_integrator",
            [u, x], [xnext]
        )
    
    def create_implicit_dynamics(self):
        q = casadi.SX.sym("q", self.model.nq)
        v = casadi.SX.sym("v", self.model.nv)
        u = casadi.SX.sym("u", self.model.nv)
        p_sphere = casadi.SX.sym("p_sphere", 2)
        v_sphere = casadi.SX.sym("v_sphere", 2)
        qnext = casadi.SX.sym("qnext", self.model.nq)
        vnext = casadi.SX.sym("vnext", self.model.nv)
        p_spherenext = casadi.SX.sym("p_spherenext", 2)
        v_spherenext = casadi.SX.sym("v_spherenext", 2)
        xnext = casadi.vertcat(qnext, vnext, p_spherenext, v_spherenext)

        # a_manipulator_start, a_sphere_start = self.acc(q, v, u, p_sphere, v_sphere)
        a_manipulator_end, a_sphere_end = self.acc(qnext, vnext, u, p_spherenext, v_spherenext)
        a_manipulator = a_manipulator_end
        a_sphere = a_sphere_end
        dt = self.timestep

        x = casadi.vertcat(q, v, p_sphere, v_sphere)
        dyn_equations = \
            casadi.vertcat(cpin.integrate(self.cmodel, q, dt * vnext) - qnext,
                           v + a_manipulator * dt - vnext,
                           p_sphere + v_spherenext * dt - p_spherenext,
                           v_sphere + a_sphere[:2] * dt - v_spherenext)
        self.implicit_integrator = casadi.Function(
            "manipulator_implicit_integrator",
            [u, x, xnext], [dyn_equations]
        )

    def create_discrete_dynamics(self):
        self.create_explicit_dynamics()
        self.create_implicit_dynamics()

    def get_discrete_dynamics(self):
        return self.explicit_integrator, self.implicit_integrator    
    
    def simulate(self):
        x0 = casadi.DM(self.model.nq + self.model.nv + 4, 1)
        x0[1] = -1.5
        x0[2] = 1.5
        print(x0.size)
        
        xx = [x0]
        for i in range(500):
            u = np.array([0, 0, 0, 0, 0, 0])
            x0 = self.explicit_integrator(u, x0)
            xx.append(x0)

        time = np.arange(len(xx)) * self.timestep

        # visualize
        robot = erd.load("ur5")
        viz = MeshcatVisualizer(
            model=self.model,
            collision_model=robot.collision_model,
            visual_model=robot.visual_model,
        )
        viz.initViewer(open=False)
        viz.loadViewerModel("pinocchio")

        qs_ = np.array(xx)[:, : self.model.nq]
        ps_ = np.array(xx)[:, self.model.nq + self.model.nv : self.model.nq + self.model.nv + 2][:,:,0]
        viz.play(q_trajectory=qs_, dt=self.timestep)

        # failed attempts

        # # add sphere visualization
        # # robot = erd.load("ur5")
        # # model = robot.model
        # # frame = pin.Frame("sphere_center", 0, 0, pin.SE3.Identity(), pin.FrameType.OP_FRAME)
        # # frame_id = model.addFrame(frame)

        # # # create sphere geometry
        # # sphere_geom = pin.GeometryObject("sphere", frame_id, pin.Sphere(0.05), pin.SE3.Identity(), np.array([0.1, 0.1, 0.1, 1.0]))        

        # import meshcat
        # vis = meshcat.Visualizer()
        # import time
        # # vis.open()

        # # visualize
        # robot = erd.load("ur5")
        # viz = MeshcatVisualizer(
        #     model=self.model,
        #     collision_model=robot.collision_model,
        #     visual_model=robot.visual_model,
        # )
        # viz.initViewer(open=False)
        # viz.loadViewerModel("pinocchio")

        # from meshcat.geometry import Sphere, MeshLambertMaterial
        # from meshcat.transformations import translation_matrix
        # sphere_radius = 0.05
        # sphere_geom = Sphere(sphere_radius)
        # sphere_material = MeshLambertMaterial(color=0xff0000)
        # vis["sphere"].set_object(sphere_geom, sphere_material)


        # qs_ = np.array(xx)[:, : self.model.nq]
        # ps_ = np.array(xx)[:, self.model.nq + self.model.nv : self.model.nq + self.model.nv + 2][:,:,0]
        # for i in range(len(qs_)):
        #     viz.display(qs_[i, :])
        #     # update sphere position
        #     p_sphere = ps_[i, :]
        #     sphere_pos = translation_matrix([p_sphere[0], p_sphere[1], sphere_radius])
        #     vis["sphere"].set_transform(sphere_pos)
        #     time.sleep(self.timestep)

        # # viz.play(q_trajectory=qs_, dt=self.timestep)

        # import meshcat
        # import meshcat.geometry as g
        # import meshcat.transformations as tf
        # import time
        # vis = meshcat.Visualizer()
        # robot = erd.load("ur5")
        # model = robot.model

        # for geom in robot.visual_model.geometryObjects:
        #     name = geom.name
        #     parent_id = geom.parentJoint
        #     vis[name].set_object(g.Box([0.05, 0.05, 0.05]))  # simple placeholder box

        # sphere_radius = 0.05
        # vis["sphere"].set_object(g.Sphere(sphere_radius), g.MeshLambertMaterial(color=0xff0000))
            
        # qs_ = np.array(xx)[:, : self.model.nq]
        # ps_ = np.array(xx)[:, self.model.nq + self.model.nv : self.model.nq + self.model.nv + 2][:,:,0]

        # for i in range(len(qs_)):
        #     # Compute link transform
        #     oMf = pin.forwardKinematics(robot.model, robot.data, qs_[i])
        #     for geom in robot.visual_model.geometryObjects:
        #         joint_id = geom.parentJoint
        #         T = oMf[joint_id]        
        #         vis[geom.name].set_transform(tf.translation_matrix(T.translation) @ tf.quaternion_matrix(T.rotation.flatten()))

        #     pos = ps_[i, :]
        #     sphere_pos = tf.translation_matrix([pos[0], pos[1], sphere_radius])
        #     vis["sphere"].set_transform(sphere_pos)

        #     time.sleep(self.timestep)

    def example_opti(self):
        N = 50
        opti = casadi.Opti()

        xx = opti.variable(self.model.nq + self.model.nv + 4, N + 1)
        uu = opti.variable(self.model.nv, N)

        x0 = casadi.DM(self.model.nq + self.model.nv + 4, 1)
        x0[1] = -1.5
        x0[2] = 2.5
        x0[-4:] = casadi.DM([1.0, 0, 0, 0])
        sphere_terminal = casadi.DM([1.5, -0.5])

        ### initial condition
        opti.subject_to(xx[:,0] == x0)

        ### stage-wise contributions
        for k in range(N):
            ## explicit
            # xnext = self.explicit_integrator(uu[:,k], xx[:,k])
            # opti.subject_to(xx[:,k+1] == xnext)

            ## implicit
            opti.subject_to(self.implicit_integrator(uu[:,k], xx[:,k], xx[:,k+1]) == 0)

        ### inequalities
            # joint limits
        q_min = self.model.lowerPositionLimit[:self.model.nq].tolist()
        q_max = self.model.upperPositionLimit[:self.model.nq].tolist()
        for k in range(N+1):
            opti.subject_to(opti.bounded(q_min, xx[:self.model.nq, k], q_max))

            # end-effector above the ground
        for k in range(N+1):
            # q = xx[:self.model.nq, k]
            # v = xx[self.model.nq:self.model.nq+self.model.nv, k]
            # cpin.forwardKinematics(self.cmodel, self.cdata, q, v)
            # cpin.updateFramePlacements(self.cmodel, self.cdata)
            # ee_name = "ee_link"
            # ee_id = self.model.getFrameId(ee_name)
            # p_ee = self.cdata.oMf[ee_id].translation
            # opti.subject_to(p_ee[2] >= 0.0)
            q = xx[:self.model.nq, k]
            v = xx[self.model.nq:self.model.nq+self.model.nv, k]
            p_ee = self.p_ee_func(q, v)
            opti.subject_to(p_ee[2] >= 0.0)

        ### objective
        opti.minimize(
            1 * 1e0 * casadi.sumsqr(xx - x0) + 
            0 * 1e0 * casadi.sumsqr(uu) + 
            1 * 1e1 * casadi.sumsqr(xx[self.model.nq:self.model.nq+self.model.nv, :]) +
            1 * 1e4 * casadi.sumsqr(xx[-4:-2, -1] - sphere_terminal)
        )

        ### initial guesses
        for k in range(N+1):
            opti.set_initial(xx[:,k], x0)
        u_init = casadi.DM([-3.53400489e-16, -1.87599750e+01,
                            -1.56838285e+01, -1.70856982e-12,
                            -1.36872302e-17,  1.95018676e-18])
        for k in range(N):
            opti.set_initial(uu[:,k], u_init)

        opti.solver("ipopt")
        sol = opti.solve()

        print("Optimal solution found:")
        print("States:", sol.value(xx))
        print("Controls:", sol.value(uu))

        print("controls at final time-step: ")
        print(sol.value(uu[:, -1]))

        # display solution
        # robot = erd.load("ur5")
        # viz = MeshcatVisualizer(
        #     model=self.model,
        #     collision_model=robot.collision_model,
        #     visual_model=robot.visual_model,
        # )
        # viz.initViewer(open=False)
        # viz.loadViewerModel("pinocchio")
        # qs_ = sol.value(xx[: self.model.nq, :]).T
        # import time
        # time.sleep(5)
        # viz.play(q_trajectory=qs_, dt=0.5*self.timestep)
        # time.sleep(5)


        sphere_positions = sol.value(xx[self.model.nq + self.model.nv : self.model.nq + self.model.nv + 2, :]).T
        ee_positions = []
        for k in range(N+1):
            q = sol.value(xx[:self.model.nq, k])
            v = sol.value(xx[self.model.nq:self.model.nq+self.model.nv, k])
            p_ee = self.p_ee_func(q, v)
            ee_positions.append(p_ee.full().flatten())

        # # create animation of sphere and end-effector positions
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlim([-1, 2])
        # ax.set_ylim([-2, 2])
        # ax.set_zlim([0, 2])
        # sphere_plot, = ax.plot([], [], [], 'ro', label='Sphere')
        # ee_plot, = ax.plot([], [], [], 'bo', label='End-Effector')
        # def update_plot(num):
        #     sphere_plot.set_data([sphere_positions[num, 0]], [sphere_positions[num, 1]])
        #     sphere_plot.set_3d_properties(0.1)
        #     ee_plot.set_data([ee_positions[num][0]], [ee_positions[num][1]])
        #     ee_plot.set_3d_properties([ee_positions[num][2]])
        #     return sphere_plot, ee_plot
        
        # from matplotlib.animation import FuncAnimation
        # ani = FuncAnimation(fig, update_plot, frames=N+1, interval=100)
        # plt.legend()
        # plt.show()
            
        # show 2d scatter of both trajectories (sphere solid, ee not filled) (with time as color)
        color_start = np.array([1, 0, 0])    # Red
        color_end = np.array([0, 0, 1])  # Blue
        import matplotlib.pyplot as plt
        plt.figure()
        for i in range(N+1):
            t = i / N
            color = (1 - t) * color_start + t * color_end
            plt.scatter(sphere_positions[i, 0], sphere_positions[i, 1], color=color, marker='o', label='Sphere' if i == 0 else "")
            plt.scatter(ee_positions[i][0], ee_positions[i][1], color=color, marker='x', label='End-Effector' if i == 0 else "")
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Sphere and End-Effector Trajectories')
        plt.legend()
        plt.axis('equal')
        plt.show()

    def simulate_joint_configurations(self):
        # let every joint rotate from 0 to -pi, back to 0, to pi, back to 0
        import time
        robot = erd.load("ur5")
        viz = MeshcatVisualizer(
            model=self.model,
            collision_model=robot.collision_model,
            visual_model=robot.visual_model,
        )

        viz.initViewer(open=False)
        viz.loadViewerModel("pinocchio")
        nq = self.model.nq
        qs_ = []
        for joint_idx in range(nq):
            n = 20
            for angle in np.linspace(0, -np.pi, n):
                q = np.zeros(nq)
                q[joint_idx] = angle
                qs_.append(q)
            for angle in np.linspace(-np.pi, 0, n):
                q = np.zeros(nq)
                q[joint_idx] = angle
                qs_.append(q)
            for angle in np.linspace(0, np.pi, n):
                q = np.zeros(nq)
                q[joint_idx] = angle
                qs_.append(q)
            for angle in np.linspace(np.pi, 0, n):
                q = np.zeros(nq)
                q[joint_idx] = angle
                qs_.append(q)
            
            # add some steady-state
            for _ in range(10):
                q = np.zeros(nq)
                q[joint_idx] = 0
                qs_.append(q)

        qs_ = np.array(qs_)
        viz.play(q_trajectory=qs_, dt=0.1)


def make_manipulator():
    robot = erd.load("ur5")
    model = robot.model

    return model, robot.collision_model, robot.visual_model