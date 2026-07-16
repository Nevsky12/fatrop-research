from casadi import *
import numpy as np

class TrajectorySimulator():
    def __init__(self, model):
        self.model = model

    def simulate(self, q0, v0, F_sol, z_sol, T, N):
        dt = T/N
        q_sim = np.zeros((3*self.model.nb_pendulums, N+1))
        v_sim = np.zeros((3*self.model.nb_pendulums, N+1))
        q_sim[:, 0] = q0
        v_sim[:, 0] = v0

        for k in range(N):
            print(f"Simulating step {k}...")
            qk = q_sim[:, k]
            vk = v_sim[:, k]
            Fk = F_sol[:, k]
            zk = z_sol[:, k]

            # simulate one step using the dynamics function
            q_next, v_next = self.model.simulate_step(qk, vk, Fk, dt, zk)

            q_sim[:, k+1] = q_next
            v_sim[:, k+1] = v_next

        return q_sim, v_sim