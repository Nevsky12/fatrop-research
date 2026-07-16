from trajectory_solver import TrajectorySolver
from trajectory_visualizer import TrajectoryVisualizer
from pendulum_3d_model import Pendulum3DModel
from trajectory_simulator import TrajectorySimulator
import os
import json
import yaml
import numpy as np

def get_ocp_files():
    folder = "../../../build_docker/ocp_results/"
    files = [f for f in os.listdir(folder) if "pendulum" in f]
    ocp_files = {}
    for f in files:
        if "expl" in f:
            ocp_files["expl"] = json.load(open(os.path.join(folder, f)))
        elif "impl" in f:
            ocp_files["impl"] = json.load(open(os.path.join(folder, f)))
        elif "accel" in f:
            ocp_files["accel"] = json.load(open(os.path.join(folder, f)))
        elif "rf" in f:
            ocp_files["rf"] = json.load(open(os.path.join(folder, f)))
        elif "reform" in f:
            ocp_files["reform"] = json.load(open(os.path.join(folder, f)))
    return ocp_files

def read_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config
config = read_config('config.yaml')

n = config['n']
model = Pendulum3DModel(nb_pendulums=n, m=[config['m']]*n, L=[config['L']]*n, g=9.81)
if config['stabilizer']['use']:
    model.add_stabilizer(gamma_1=config['stabilizer']['gamma_1'], 
                         gamma_2=config['stabilizer']['gamma_2'])
if config['stiff_joints']['use']:
    model.set_stiff_joints(joint_stiffness=config['stiff_joints']['joint_stiffness'], 
                           joint_damping=config['stiff_joints']['joint_damping'])
model.set_model()

# load data
ocp_files = get_ocp_files()

# initialize solver and visualizer
solver = TrajectorySolver(model)
if config['scenario']['tracking']:
    solver.set_path_tracking_scenario()
visualizer = TrajectoryVisualizer(model, solver)

# visualize
q_sol, v_sol, F_sol, z_sol = solver.extract_solution(
    np.array(ocp_files["expl"]["states"]).T, 
    np.array(ocp_files["expl"]["inputs"]).T
)
# visualizer.add_data_from_ocp_json_file(ocp_files["expl"])
visualizer.add_data(T=config['scenario']['T'], q_sol=q_sol, v_sol=v_sol, 
                    F_sol=F_sol, z_sol=z_sol)
visualizer.visualize_all(appendix="fatrop_expl")

# simulate trajectory
if config['simulate']:
    simulator = TrajectorySimulator(model)
    q_sim, v_sim = simulator.simulate(q0=q_sol[:, 0], v0=v_sol[:, 0], 
                                    F_sol=F_sol, z_sol=z_sol, T=config['scenario']['T'], 
                                    N=config['scenario']['N'])
    visualizer.add_data(T=config['scenario']['T'], q_sol=q_sim, v_sol=v_sim, 
                        F_sol=F_sol, z_sol=z_sol)
    visualizer.visualize_all(appendix="fatrop_expl_simulated")