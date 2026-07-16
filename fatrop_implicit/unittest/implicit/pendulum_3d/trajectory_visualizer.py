from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class TrajectoryVisualizer():
    def __init__(self, model, solver):
        self.model = model
        self.solver = solver

    def add_data(self, T, solution):
        self.T = T
        self.N = solution['q_sol'].shape[1] - 1
        self.t_sol = np.linspace(0, self.T, self.N+1)
        self.q_sol = solution['q_sol']
        self.v_sol = solution['v_sol']
        self.F_sol = solution['F_sol']
        self.z_sol = solution['z_sol']
        self.p_sol = solution['p_sol'] if self.solver.tracking else None

    def add_data_from_ocp_json_file(self, data):
        self.N = data['generator_data']['K'] - 1
        self.T = data['generator_data']['dt']*self.N
        uu_sol = np.array(data['inputs']).T
        xx_sol = np.array(data['states']).T
        self.q_sol = xx_sol[:3*self.model.nb_pendulums, :]
        self.v_sol = xx_sol[3*self.model.nb_pendulums:-1, :]
        self.F_sol = uu_sol[:3*self.model.nb_pendulums, :]
        self.z_sol = uu_sol[3*self.model.nb_pendulums:, :]
        self.t_sol = np.linspace(0, self.T, self.N+1)

    def visualize_all(self, appendix=""):
        self.file_name_appendix = appendix
        self.visualize_trajectory()
        self.visualize_forces()
        self.visualize_pendulum_length()
        self.visualize_z_vars()
        self.visualize_energy()
        self.animate_trajectory()
        
    def visualize_result(self, T, result, appendix=""):
        if not result['success']:
            print(f"Result ({appendix}) was not successful, skipping visualization.")
            return
    
        print(f"Visualizing result ({appendix})...")
        self.add_data(T=T, solution=result)
        self.visualize_all(appendix=appendix)

    def get_point(self, q, i, slice_min=0, slice_max=None):
        if slice_max is None:
            slice_max = q.shape[1]

        if i < 0:
            return np.zeros(slice_max - slice_min), np.zeros(slice_max - slice_min), np.zeros(slice_max - slice_min)

        return q[3*i, slice_min:slice_max], q[3*i+1, slice_min:slice_max], q[3*i+2, slice_min:slice_max]

    def visualize_trajectory(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # plot anchor point
        ax.scatter(0, 0, 0, color='k', label='Anchor Point')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-self.model.nb_pendulums-0.5, 0.5)

        # plot trajectory
        for i in range(self.model.nb_pendulums):
            plt.plot(*self.get_point(self.q_sol, i), marker='o', label=f'Pendulum {i}')

        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        plt.legend()

        plt.savefig(f"visualization_output/pendulum_trajectory_{self.file_name_appendix}.png", dpi=300)

    def show_pendulum_projection(self, ax, frame, plane_x, plane_y, plane_z):
        all_points = np.zeros((self.model.nb_pendulums + 1, 3))
        for i in range(-1, self.model.nb_pendulums):
            all_points[i+1, :] = np.array(self.get_point(self.q_sol, i, slice_min=frame, slice_max=frame+1)).T

        ax.plot(plane_x, all_points[:, 1], all_points[:,2], color='grey', marker='.', alpha=1.0)
        ax.plot(all_points[:, 0], plane_y, all_points[:,2], color='grey', marker='.', alpha=1.0)
        ax.plot(all_points[:, 0], all_points[:,1], plane_z, color='grey', marker='.', alpha=1.0)
        
    
    def animate_trajectory(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # compute reference path if it exists
        if self.solver.tracking:
            ref_path = np.zeros((3, self.N+1))
            for k in range(self.N+1):
                ref_path[:, k] = self.solver.path_func(self.p_sol[:,k], self.T, self.q_sol[:, 0]).full().flatten()

        # animation function
        def update(frame):
            ax.clear()
            ax.scatter(0, 0, 0, color='k', label='Anchor Point')
            for i in range(self.model.nb_pendulums):
                # show points
                point = self.get_point(self.q_sol, i, slice_min=frame, slice_max=frame+1)
                plt.plot(point[0], point[1], point[2], marker='.', color='r', label=f'Pendulum {i}')

                # show traced path
                if self.solver.tracking and i == self.solver.tracking_mass_index:
                    plt.plot(*self.get_point(self.q_sol, i, slice_min=0, slice_max=frame), color='k', alpha=0.5)

                # show reference path if it exists
                if self.solver.tracking:
                    plt.plot(ref_path[0, :frame], ref_path[1, :frame], ref_path[2, :frame], color='g', alpha=0.5, label='Reference Path')

                # show pendulum itself
                prev_point = self.get_point(self.q_sol, i-1, slice_min=frame, slice_max=frame+1)
                ax.plot([prev_point[0], point[0]], [prev_point[1], point[1]], [prev_point[2], point[2]], color='r')

            # show forces
            if frame < self.N:
                max_force = np.max(np.linalg.norm(self.F_sol, axis=0))
                max_force_vector_length = 1.0
                force_scale = max_force_vector_length / max_force if max_force > 0 else 1.0

                for i, idx in enumerate(self.model.actuated_joint_idxs):
                    point = self.get_point(self.q_sol, idx, slice_min=frame, slice_max=frame+1)
                    force_vector = self.F_sol[3*i:3*i+3, frame] * force_scale
                    ax.quiver(point[0], point[1], point[2], force_vector[0], force_vector[1], force_vector[2], color='b')#, normalize=True)

            xlim = (-1.5, 1.5)
            ylim = (-1.5, 1.5)
            zlim = (-np.sum(self.model.L)-0.5, 0.5)
            center = ((xlim[0]+xlim[1])/2, (ylim[0]+ylim[1])/2, (zlim[0]+zlim[1])/2)
            max_range = max(xlim[1]-xlim[0], ylim[1]-ylim[0], zlim[1]-zlim[0]) / 2
            
            xlim = (center[0] - max_range, center[0] + max_range)
            ylim = (center[1] - max_range, center[1] + max_range)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_zlim(-np.sum(self.model.L)-0.5, 0.5)

            # show pendulum projection on XZ, YZ and XY planes
            self.show_pendulum_projection(ax, frame, plane_x=xlim[0], plane_y=ylim[1], plane_z=-np.sum(self.model.L)-0.5)
            ax.set_xlabel('X Position (m)')
            ax.set_ylabel('Y Position (m)')
            ax.set_zlabel('Z Position (m)')
            # ax.legend(loc="upper right")

        interval_ms = self.T / self.N * 1000
        ani = animation.FuncAnimation(fig, update, frames=self.N+1, interval=interval_ms)
        ani.save(f"visualization_output/pendulum_trajectory_animation_{self.file_name_appendix}.mp4", writer='ffmpeg', fps=self.N/self.T, dpi=150)

    def visualize_forces(self):
        fig = plt.figure()
        plt.plot(self.t_sol[:-1], self.F_sol.T)
        plt.xlabel("Time (s)")
        plt.ylabel("Control Forces (N)")
        plt.title("Control Forces over Time")
        plt.legend(['Fx', 'Fy', 'Fz'])
        plt.savefig(f"visualization_output/control_forces_{self.file_name_appendix}.png", dpi=300)

    def visualize_pendulum_length(self):
        fig = plt.figure()
        lengths = np.zeros((self.model.nb_pendulums, self.N+1))
        for i in range(self.model.nb_pendulums):
            point = self.get_point(self.q_sol, i)
            prev_point = self.get_point(self.q_sol, i-1)
            lengths[i, :] = np.linalg.norm(np.array(point) - np.array(prev_point), axis=0)
            # plt.plot(self.t_sol, lengths.T)
            plt.plot(self.t_sol, lengths[i, :], label=f'Pendulum {i} Length')

        plt.xlabel("Time (s)")
        plt.ylabel("Pendulum Length (m)")
        plt.title("Pendulum Length over Time")
        plt.legend()
        plt.savefig(f"visualization_output/pendulum_length_{self.file_name_appendix}.png", dpi=300)

    def visualize_z_vars(self):
        fig = plt.figure()
        plt.plot(self.t_sol[:-1], self.z_sol.T)
        plt.xlabel("Time (s)")
        plt.ylabel("Slack Variables")
        plt.title("z Variables over Time")
        plt.savefig(f"visualization_output/z_variables_{self.file_name_appendix}.png", dpi=300)

    def visualize_energy(self):
        fig = plt.figure()
        T = np.zeros(self.N+1)
        V = np.zeros(self.N+1)
        E = np.zeros(self.N+1)
        for k in range(self.N+1):
            q_k = self.q_sol[:, k]
            v_k = self.v_sol[:, k]
            T[k] = self.model.T(q_k, v_k)
            V[k] = self.model.V(q_k)
            E[k] = self.model.E_total(q_k, v_k)

        plt.plot(self.t_sol, T, label='Kinetic Energy')
        plt.plot(self.t_sol, V, label='Potential Energy')
        plt.plot(self.t_sol, E, label='Total Energy')
        plt.xlabel("Time (s)")
        plt.ylabel("Energy (J)")
        plt.title("Energy over Time")
        plt.legend()
        plt.savefig(f"visualization_output/energy_{self.file_name_appendix}.png", dpi=300)

    def show(self):
        plt.show()

    def close(self):
        plt.close('all')
