from trajectory_solver import PrintAveragedTimes, PrintDifferences, TrajectorySolver, SolveExplicit, SolveImplicit, SolveReformulated, SolveAccelerated 
from trajectory_visualizer import TrajectoryVisualizer
from pendulum_3d_model import Pendulum3DModel, SetupModel
import numpy as np
import yaml
import pickle

### Read config file
def read_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config
config = read_config('config.yaml')

### Set up the model
model, q0, v0 = SetupModel(config)

### Set up the solver
solver = TrajectorySolver(model, config)
if config['scenario']['tracking']:
    solver.set_path_tracking_scenario()

T = config['scenario']['T']
N = config['scenario']['N']

### Solve methods
if config['caching']['load']:
    with open('solution.pkl', 'rb') as f:
        T, result_expl, result_reform, result_accel = pickle.load(f)
else:
    result_expl = SolveExplicit(solver, q0, v0, T, N, nb_runs=config['nb_runs'])
    # result_expl = None
    result_reform = SolveReformulated(solver, q0, v0, T, N, nb_runs=config['nb_runs'])
    result_accel = SolveAccelerated(solver, q0, v0, T, N, nb_runs=config['nb_runs'])
    # result_accel = None

### Store solutions
if config['caching']['store'] and not config['caching']['load']:
    with open('solution.pkl', 'wb') as f:
        pickle.dump([T, result_expl, result_reform, result_accel], f)

### Print averaged computation times
if result_expl is not None:
    PrintAveragedTimes(result_expl, "Explicit")
if result_reform is not None:
    PrintAveragedTimes(result_reform, "Reformulated")
if result_accel is not None:
    PrintAveragedTimes(result_accel, "Accelerated")
# TrajectorySolver.PrintAveragedTimes(result_impl, "Implicit")

PrintDifferences(result_reform, result_accel, "Reformulated", "Accelerated")

### Visualize
if config['visualize']:
    visualizer = TrajectoryVisualizer(model, solver)
    if result_expl is not None:
        visualizer.visualize_result(T, result_expl, appendix="explicit")
    if result_reform is not None:
        visualizer.visualize_result(T, result_reform, appendix="reformulated")
    if result_accel is not None:
        visualizer.visualize_result(T, result_accel, appendix="accelerated")
    # visualizer.visualize_result(T, result_impl, appendix="implicit")
