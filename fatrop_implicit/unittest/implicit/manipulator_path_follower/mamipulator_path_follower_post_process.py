import json
import os
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca

### PARAMETERS ###
folder = "build_docker/ocp_results/"



### HELPER FUNCTIONS ###
def get_all_json(folder):
    files = []
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            with open(os.path.join(folder, filename), 'r') as f:
                data = json.load(f)
                if data['generator_data']['problem_name'] != "path_follower":
                    continue
                files.append(data)
    return files

def plot_trajectories_3d(files):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    fk = ca.Function.load("unittest/implicit/manipulator_path_follower/ur10_fk.casadi")

    for f in files:
        xx = np.array(f['states']).T
        travelled_path = np.zeros((3, xx.shape[1]))
        for i in range(xx.shape[1]):
            q = xx[:6, i]
            pos = fk(q)[-1][:3,-1]
            travelled_path[:, i] = np.array(pos).flatten()

        ax.plot(travelled_path[0, :], travelled_path[1, :], travelled_path[2, :], label=f['method_name'])
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Manipulator Path Follower')
    ax.set_box_aspect([1,1,1])  # Equal aspect ratio
    ax.legend()

def show_progress_variable(files):
    fig, ax = plt.subplots(2, 1)
    for f in files:
        xx = np.array(f['states']).T
        t_sol = xx[-1,:]
        ax[0].plot(np.linspace(0, t_sol[-1], len(t_sol)), t_sol, label=f['method_name'])

        uu = np.array(f['inputs']).T
        dt_sol = uu[12,:]
        ax[1].plot(np.linspace(0, t_sol[-1], len(dt_sol)), dt_sol, label=f['method_name'])
        
    ax[0].plot([0, t_sol[-1]], [0, t_sol[-1]], '--', label="ideal progress")
    
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Progress Variable")
    ax[0].set_title("Progress Variable Over Time")
    ax[0].legend()
    
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Progress Variable Speed")
    ax[1].set_title("Progress Variable Speed Over Time")
    ax[1].legend()
    
    plt.tight_layout()


### MAIN ###
files = get_all_json(folder)
plot_trajectories_3d(files)
show_progress_variable(files)
plt.show()