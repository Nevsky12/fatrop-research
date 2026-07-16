import casadi as ca
import numpy as np

def SetupModel(config):
    model = Pendulum3DModel(config)
    if config['stabilizer']['use']:
        model.add_stabilizer(gamma_1=config['stabilizer']['gamma_1'], 
                             gamma_2=config['stabilizer']['gamma_2'])
    if config['stiff_joints']['use']:
        model.set_stiff_joints(joint_stiffness=config['stiff_joints']['joint_stiffness'], 
                               joint_damping=config['stiff_joints']['joint_damping'])
    model.set_model()
    model.print_model_dimensions()
    q0 = model.get_init_vector(randomize=config['scenario']['initialization']['randomize'], 
                               at_rest=config['scenario']['initialization']['at_rest'])
    v0 = 0*q0
    return model, q0, v0

class Pendulum3DModel():
    def __init__(self, config):
        # define model parameters
        self.nb_pendulums = config['n']
        self.m = [config['m']] * self.nb_pendulums
        self.L = [config['L']] * self.nb_pendulums
        self.g = config['g']

        # define actuated joints
        self.actuated_joint_idxs = []
        for idx in config['scenario']['actuation_idxs']:
            true_idx = idx
            if true_idx < 0:
                true_idx = self.nb_pendulums + idx
            assert true_idx >= 0 and true_idx < self.nb_pendulums, f"Actuated joint index {idx} is out of bounds for number of pendulums {self.nb_pendulums}"
            
            if idx not in self.actuated_joint_idxs:
                self.actuated_joint_idxs.append(idx)

        # self.set_model()
        self.stabilizer = None
        self.with_stiff_joints = False

    def set_stiff_joints(self, joint_stiffness, joint_damping):
        self.with_stiff_joints = True
        self.joint_stiffness = joint_stiffness
        self.joint_damping = joint_damping

    def get_joint_pos(self, q, i):
        if i < 0:
            return ca.vertcat(ca.SX.zeros(2), self.L[0]*(-(i+1)))
        
        return q[3*i:3*i+3]
    
    def get_pendulum_vector(self, q, i):
        return self.get_joint_pos(q, i) - self.get_joint_pos(q, i-1)
    
    def get_normalized_pendulum_vector(self, q, i):
        L = self.L[i] if i >= 0 else self.L[0]
        return self.get_pendulum_vector(q, i) / L
    
    def phi(self, q, i):
        return ca.sumsqr(self.get_normalized_pendulum_vector(q, i) - self.get_normalized_pendulum_vector(q, i-1))
    
    def phi_dot(self, q, v, i):
        return ca.sumsqr(self.get_normalized_pendulum_vector(v, i) - self.get_normalized_pendulum_vector(v, i-1))

    def Rayleigh_dissipation(self, q, v):
        if not self.with_stiff_joints:
            return 0
        
        R = 0
        for i in range(self.nb_pendulums):
            R += 0.5 * self.joint_damping * self.phi_dot(q, v, i)
        return R        

    def get_damping_force(self, q, v, i):
        R = self.Rayleigh_dissipation(q, v)
        vi = self.get_joint_pos(v, i)
        return -ca.jacobian(R, vi).T

    def set_model(self):
        # define q and v
        q = ca.SX.sym("q", 3*self.nb_pendulums)
        v = ca.SX.sym("v", 3*self.nb_pendulums)
        z = ca.SX.sym("z", 1*self.nb_pendulums)
        F = ca.SX.sym("F", 3*len(self.actuated_joint_idxs))

        M = ca.SX.eye(3*self.nb_pendulums)
        for i in range(self.nb_pendulums):
            M[3*i:3*i+3, 3*i:3*i+3] = self.m[i] * ca.SX.eye(3)

        # define dynamics
        T = 0.5 * v.T @ M @ v
        V = 0
        c = ca.SX.zeros(0,1)
        external_forces = ca.SX.zeros(3*(self.nb_pendulums),1)
        for i, idx in enumerate(self.actuated_joint_idxs):
            external_forces[3*idx:3*idx+3] += F[3*i:3*i+3]

        for i in range(self.nb_pendulums):
            # gravity potential energy
            V += self.m[i] * self.g * self.get_joint_pos(q, i)[2]

            # joint stiffness
            if self.with_stiff_joints:
                V += 0.5 * self.joint_stiffness * self.phi(q, i)

                F_damp = self.get_damping_force(q, v, i)
                external_forces[3*i:3*i+3] += F_damp

            # pendulum length constraint
            c = ca.vertcat(c, ca.sumsqr(self.get_joint_pos(q, i) - self.get_joint_pos(q, i-1)) - self.L[i]**2)

        Jc = ca.jacobian(c, q)

        self.T = ca.Function("T", [q, v], [T], ["q", "v"], ["T"])
        self.V = ca.Function("V", [q], [V], ["q"], ["V"])
        self.E_total = ca.Function("E_total", [q, v], [T + V], ["q", "v"], ["E_total"])

        qdd = ca.inv(M) @ (external_forces + ca.jacobian(T - V, q).T - ca.transpose(Jc) @ z) # assuming Mdot is 0
        self.f = ca.Function("f", [q, v, z, F], [v, qdd], ["q", "v", "z", "F"], ["qdot", "vdot"])

        eq = Jc @ qdd + ca.jacobian(Jc @ v, q) @ v
        self.g_eq = ca.Function("g_eq", [q, v, z, F], [eq], ["q", "v", "z", "F"], ["c"])
        self.g_eq0 = ca.Function("g_eq0", [q, v], [ca.vertcat(c, Jc @ v)], ["q", "v"], ["c0"])

        self.f.save("casadi_functions/pendulum_3d_dynamics.casadi")
        self.g_eq.save("casadi_functions/pendulum_3d_g_eq.casadi")
        self.g_eq0.save("casadi_functions/pendulum_3d_g_eq0.casadi")

    def add_stabilizer(self, gamma_1, gamma_2):
        q = ca.SX.sym("q", 3*self.nb_pendulums)
        v = ca.SX.sym("v", 3*self.nb_pendulums)

        stabilizer = ca.SX.zeros(0,1)
        for i in range(self.nb_pendulums):
            # qi = q[3*i:3*i+3]
            # vi = v[3*i:3*i+3]
            # qi_prev = q[3*(i-1):3*(i-1)+3] if i > 0 else ca.SX.zeros(3)

            qi = self.get_joint_pos(q, i)
            vi = self.get_joint_pos(v, i)
            qi_prev = self.get_joint_pos(q, i-1)
            # if i == 0:
            #     qi_prev = ca.SX.zeros(3)

            stabilizer = ca.vertcat(stabilizer,
                                -gamma_1*qi.T @ vi - gamma_2*(ca.sumsqr(qi - qi_prev) - self.L[i]**2))

        self.stabilizer = ca.Function("stabilizer", [q, v], [-stabilizer], ["q", "v"], ["stabilizer"])
        self.stabilizer.save("casadi_functions/pendulum_3d_stabilizer.casadi")

    def simulate_step(self, q, v, F, dt, z_init):
        z = ca.SX.sym("z", self.nb_pendulums)
        qdot, vdot = self.f(q, v, z, F)
        g_eq = self.g_eq(q, v, z, F)

        # solve for z using g_eq = 0
        # solver = ca.rootfinder("solver", "newton", self.g_eq)
        # z_sol = solver(q=q, v=v, z=ca.SX.zeros(self.nb_pendulums), F=F)
        opti = ca.Opti()
        z_var = opti.variable(self.nb_pendulums)
        opti.minimize(ca.sumsqr(self.g_eq(q, v, z_var, F)))
        opti.set_initial(z_var, z_init)
        opti.solver('ipopt')
        sol = opti.solve()
        z_sol = sol.value(z_var)
        qdot = sol.value(qdot)
        vdot = sol.value(vdot)

        q_next = q + qdot * dt
        v_next = v + vdot * dt

        return q_next, v_next

    def print_model_dimensions(self):
        print(f"Number of pendulums: {self.nb_pendulums}")
        nx = self.f.sparsity_in(0).size1() + \
             self.f.sparsity_in(1).size1()
        nu = self.f.sparsity_in(2).size1() + \
             self.f.sparsity_in(3).size1()
        ng = self.g_eq.sparsity_out(0).size1()
        print(f"nx: {nx}\nnu: {nu}\nng: {ng}")

    def get_init_vector(self, randomize=False, at_rest=False):
        def get_rotation_matrix(theta_x, theta_y, theta_z):
            Rx = ca.vertcat(
                ca.horzcat(1, 0, 0),
                ca.horzcat(0, ca.cos(theta_x), -ca.sin(theta_x)),
                ca.horzcat(0, ca.sin(theta_x), ca.cos(theta_x))
            )
            Ry = ca.vertcat(
                ca.horzcat(ca.cos(theta_y), 0, ca.sin(theta_y)),
                ca.horzcat(0, 1, 0),
                ca.horzcat(-ca.sin(theta_y), 0, ca.cos(theta_y))
            )
            Rz = ca.vertcat(
                ca.horzcat(ca.cos(theta_z), -ca.sin(theta_z), 0),
                ca.horzcat(ca.sin(theta_z), ca.cos(theta_z), 0),
                ca.horzcat(0, 0, 1)
            )
            return Rz @ Ry @ Rx
                
        q0 = np.zeros(3*self.nb_pendulums)
        p = q0[:3]
        for i in range(self.nb_pendulums):
            if randomize:
                theta_x = np.random.uniform(-0.3, 0.3)
                theta_y = np.random.uniform(-0.3, 0.3)
                theta_z = np.random.uniform(-0.3, 0.3)
            elif at_rest:
                theta_x = 0.0
                theta_y = 0.0
                theta_z = 0.0
            else:
                theta_x = -0.2*i
                theta_y = -0.3
                theta_z = 0.0 if i == 0 else 0
            
            R = get_rotation_matrix(theta_x, theta_y, theta_z)

            p += R @ ca.vertcat(0, 0, -self.L[i])    
            q0[3*i:3*(i+1)] = p.full().flatten()

        return q0