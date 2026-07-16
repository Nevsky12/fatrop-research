from asyncio import subprocess
from fnmatch import fnmatch

from casadi import *
import numpy as np
import json
import yaml

def SolveExplicit(solver, q0, v0, T, N, nb_runs=1):
    solver.implicit = False
    solver.reformulate = False
    solver.accelerated = False
    
    return SolveMultipleRuns(solver, q0, v0, T, N, nb_runs)

def SolveImplicit(solver, q0, v0, T, N, nb_runs=1):
    solver.implicit = True
    solver.reformulate = False
    solver.accelerated = False
    
    return SolveMultipleRuns(solver, q0, v0, T, N, nb_runs)

def SolveReformulated(solver, q0, v0, T, N, nb_runs=1):
    solver.implicit = True
    solver.reformulate = True
    solver.accelerated = False
    
    return SolveMultipleRuns(solver, q0, v0, T, N, nb_runs)    

def SolveAccelerated(solver, q0, v0, T, N, nb_runs=1):
    solver.implicit = True
    solver.reformulate = True
    solver.accelerated = True
    
    return SolveMultipleRuns(solver, q0, v0, T, N, nb_runs)

def SolveMultipleRuns(solver, q0, v0, T, N, nb_runs):
    t_total = []
    t_search_direction = []
    nb_iterations = []
    stats = []
    for run in range(nb_runs):
        print(f"Running solve, run {run+1}/{nb_runs}...")
        sol = solver.solve_baked_trajectory(q0=q0, v0=v0, T=T, N=N)
        try:
            t_total.append(solver.sol.stats()['fatrop']['time_total'])
            t_search_direction.append(solver.sol.stats()['fatrop']['compute_sd_time'])
            nb_iterations.append(solver.get_nb_iters())
            stats.append(solver.sol.stats())
        except:
            return None
    
    sol['t_total'] = t_total
    sol['t_search_direction'] = t_search_direction
    sol['nb_iterations'] = nb_iterations
    sol['stats'] = stats
    return sol

def PrintAveragedTimes(sol, method_name):
    print(f"Averaged computation times for {method_name}:")
    print(f"  Total time:                          {np.mean(sol['t_total']):.3f}")
    print(f"  Total time per iteration:            {np.mean(sol['t_total'])/np.mean(sol['nb_iterations']):.3f}")
    print(f"  Search direction time:               {np.mean(sol['t_search_direction']):.3f}")
    print(f"  Search direction time per iteration: {np.mean(sol['t_search_direction'])/np.mean(sol['nb_iterations']):.3f}")
    print(f"  Number of iterations:                {int(np.mean(sol['nb_iterations']))}")
    func_eval_times = [sum([v for k, v in stat['fatrop'].items() if fnmatch(k, 'eval_*_time')]) for stat in sol['stats']]
    func_eval_time = np.mean(func_eval_times)
    func_eval_calls = np.mean([sum([v for k, v in stat['fatrop'].items() if fnmatch(k, 'eval_*_calls')]) for stat in sol['stats']])
    print(f"  Function eval time:                  {func_eval_time:.3f} (over {func_eval_calls:.0f} calls)")
    print(f"  Function eval time per iteration:    {func_eval_time/func_eval_calls:.6f}")
    
    print(f"  Percentage of total time spent in search direction: {np.mean(sol['t_search_direction'])/np.mean(sol['t_total'])*100:.2f}%")
    print(f"  Percentage of fatrop time spent in search direction: {np.mean(sol['t_search_direction'])/(np.mean([sol['t_total']]) - func_eval_time)*100:.2f}%")
    
def PrintDifferences(sol1, sol2, name1, name2):
    print(f"Relative difference between {name1} and {name2}:")
    print(f"   Search direction time:               {(np.mean(sol2['t_search_direction']) - np.mean(sol1['t_search_direction']))/np.mean(sol1['t_search_direction'])*100:.4f}%")
    print(f"   Search direction time per iteration: {(np.mean(sol2['t_search_direction'])/np.mean(sol2['nb_iterations']) - np.mean(sol1['t_search_direction'])/np.mean(sol1['nb_iterations']))/(np.mean(sol1['t_search_direction'])/np.mean(sol1['nb_iterations']))*100:.2f}%")
    print(f"   Total time:                          {(np.mean(sol2['t_total']) - np.mean(sol1['t_total']))/np.mean(sol1['t_total'])*100:.2f}%")
    print(f"   Total time per iteration:            {(np.mean(sol2['t_total'])/np.mean(sol2['nb_iterations']) - np.mean(sol1['t_total'])/np.mean(sol1['nb_iterations']))/(np.mean(sol1['t_total'])/np.mean(sol1['nb_iterations']))*100:.2f}%")

class TrajectorySolver():
    def __init__(self, model, config, implicit=False):
        # load models
        self.model = model
        self.implicit = implicit
        self.reformulate = False
        self.accelerated = False

        # ocp params
        self.force_bounds = config['scenario']['force_bounds']
        
        # opti params
        self.opti_to_function = config['opti']['to_function']
        self.opti_code_generate = config['opti']['code_generate']

        # ocp scenario
        self.tracking = False
        self.tracking_mass_index = self.model.nb_pendulums-1
        
        # options
        self.store_casadi_functions = False
        self.introduce_zk_sparsily = True

    def set_path_tracking_scenario(self):
        self.tracking = True
        
        t = SX.sym("t")
        T = SX.sym("T")
        q0 = SX.sym("q0", 3*self.model.nb_pendulums)
        ee_pos = self.model.get_joint_pos(q0, self.tracking_mass_index)
        
        ### circle
        # R_sq = 0.01 + sumsqr(self.model.get_joint_pos(q0, self.tracking_mass_index)[:2])
        # center = vertcat(0, 0, self.model.get_joint_pos(q0, self.tracking_mass_index)[2])
        # phase_0 = atan2(self.model.get_joint_pos(q0, self.tracking_mass_index)[1], self.model.get_joint_pos(q0, self.tracking_mass_index)[0])

        # nb_circles = 0.5
        # ref_point = center + \
        #     vertcat(cos(2*np.pi*t/T*nb_circles + phase_0), 
        #                sin(2*np.pi*t/T*nb_circles + phase_0), 
        #                0)*sqrt(R_sq)
        
        ### lemniscate
        # R = 0.5
        # center = vertcat(0, 0, self.model.get_joint_pos(q0, self.tracking_mass_index)[2])
        # phase_0 = pi/2
        # nb_lemniscates = 0.5
        # s = sin(2*pi*t/T*nb_lemniscates + phase_0)
        # c = cos(2*pi*t/T*nb_lemniscates + phase_0)
        # x = R*c/(1+ s**2)
        # y = R*s*c/(1+ s**2)
        # ref_point = center + vertcat(x, y, 0)
        
        ### large swing
        R = sum(self.model.L)
        th1 = Function("th1", [t, T], [0.1*(1-cos(2*np.pi*t/T))], ['t', 'T'], ['th1'])
        ref_point = vertcat(R*sin(th1(t,T)), 0, -R*cos(th1(t,T)))
        
        self.path_func = Function("path_func", [t, T, q0], [ref_point], ['t', 'T', 'q0'], ['ref_point'])

    def get_q_rest(self):
        q_rest = np.zeros(3*self.model.nb_pendulums)
        q_rest[2] = -self.model.L[0]
        for i in range(1, self.model.nb_pendulums):
            q_rest[3*i + 2] = q_rest[3*(i-1) + 2] - self.model.L[i]
        return q_rest
    
    def extract_solution(self, xx, uu):
        q_sol = xx[:3*self.model.nb_pendulums, :]
        v_sol = xx[3*self.model.nb_pendulums:6*self.model.nb_pendulums, :]
        p_sol = xx[6*self.model.nb_pendulums:, :]
        F_sol = uu[3*self.tracking + 1:3*self.tracking + 1 + 3*len(self.model.actuated_joint_idxs), :]
        z_sol = uu[3*self.tracking:3*self.tracking + self.model.nb_pendulums, :]
        return q_sol, v_sol, F_sol, z_sol
    
    def bake(self, q0, v0, T, N):
        # xk
        q = SX.sym("q", 3*self.model.nb_pendulums)
        v = SX.sym("v", 3*self.model.nb_pendulums)
        p = SX.sym("p", 1*self.tracking)
        xk = vertcat(v, p, q)
        
        # xk_plus
        qp = SX.sym("qp", 3*self.model.nb_pendulums)
        vp = SX.sym("vp", 3*self.model.nb_pendulums)
        pp = SX.sym("pp", 1*self.tracking)
        xk_plus = vertcat(vp, pp, qp)

        # uk        
        s = SX.sym("s", 3*self.tracking)
        z = SX.sym("z", 1*self.model.nb_pendulums)
        F = SX.sym("F", 3*len(self.model.actuated_joint_idxs))
        dp = SX.sym("dp", 0*1*self.tracking)
        uk = vertcat(s, z, F, dp)
        uk_true = uk
        
        # reformulation trick
        if self.reformulate:
            if self.introduce_zk_sparsily:
                zk_true = SX.sym("zk_true", 3*self.model.nb_pendulums) # only keep velocity
                zk = vertcat(qp, pp, zk_true)
                uk = vertcat(uk, zk_true)
            else:
                zk = SX.sym("zk", xk.size1())
                zk_true = zk
                uk = vertcat(uk, zk)

        dt = T/N

        ### define functions
        self.funcs = {}

        # objective
        objk = 1e-5*sumsqr(F) + (1e2*sumsqr(s) if self.tracking else 0) + 0*1e-2*sumsqr(v)
        self.funcs['eval_objk'] = Function("eval_objk", [uk, xk], [objk], ['uk', 'xk'], ['objk'])

        objK = 0.0 if self.tracking else sumsqr(v) + 1e2*sumsqr(q - self.get_q_rest())
        self.funcs['eval_objK'] = Function("eval_objK", [xk], [objK], ['xk'], ['objK'])

        # equality constraints
        gk = self.model.g_eq(q, v, z, F) + (self.model.stabilizer(q, v) if self.model.stabilizer is not None else 0)
        if self.tracking:
            ref_point = self.path_func(p, T, q0)
            gk = vertcat(gk, s - (q[:3] - ref_point))
        self.funcs['eval_g0'] = Function("eval_g0", [uk, xk], [vertcat(vertcat(v - v0, p, q - q0), gk)], ['uk', 'xk'], ['g0'])
        self.funcs['eval_gk'] = Function("eval_gk", [uk, xk], [gk], ['uk', 'xk'], ['gk'])
        self.funcs['eval_gK'] = Function("eval_gK", [xk], [vertcat(p - T) if self.tracking else vertcat()], ['xk'], ['gK'])

        # inequality constraints
        self.funcs['eval_gk_ineq'] = Function("eval_gk_ineq", [uk, xk], [vertcat(F, dp)], ['uk', 'xk'], ['ineqk'])
        self.funcs['eval_gK_ineq'] = Function("eval_gK_ineq", [xk], [vertcat()], ['xk'], ['ineqK'])
        ub = vertcat(self.force_bounds*SX.ones(F.size1(), 1),
                     1.5*SX.ones(dp.size1(), 1))
        lb = vertcat(-self.force_bounds*SX.ones(F.size1(), 1),
                     0.5*SX.ones(dp.size1(), 1))
        self.funcs["lb"] = Function("lb", [], [lb], [], ['lb'])
        self.funcs["ub"] = Function("ub", [], [ub], [], ['ub'])
        self.funcs["lbK"] = Function("lbK", [], [vertcat()], [], ['lbK'])
        self.funcs["ubK"] = Function("ubK", [], [vertcat()], [], ['ubK'])

        # dynamics
        qdot, vdot = self.model.f(q, v, z, F)
        if self.tracking:
            rhs_expl = vertcat(v + vdot*dt, p + dt, q + v*dt)
            rhs_impl = vertcat(v + vdot*dt - vp, p + dt - pp, q + vp*dt - qp)
        else:
            rhs_expl = vertcat(v + vdot*dt, q + v*dt)
            rhs_impl = vertcat(v + vdot*dt - vp, q + vp*dt - qp)
        self.funcs['expl_dyn'] = Function("expl_dyn", [uk, xk], [rhs_expl], ['uk', 'xk'], ['xk_plus'])
        self.funcs['impl_dyn'] = Function("impl_dyn", [uk, xk, xk_plus], [rhs_impl], ['uk', 'xk', 'xk_plus'], ['dyn_res'])
        
        # initialization
        k = SX.sym("k")
        x_init = vertcat(v0, q0)
        u_init = 0*uk_true
        if self.tracking:
            x_init = vertcat(x_init, k*dt)
            u_init = vertcat(u_init[:-1,:], 1)
        self.funcs["x_init"] = Function("x_init", [k], [x_init], ['k'], ['x_init'])
        self.funcs["u_init"] = Function("u_init", [k], [u_init], ['k'], ['u_init'])
        
        # reformulated
        if self.reformulate:
            # move dynamics into constraints
            dyn_expr = self.funcs['impl_dyn'](uk, xk, zk) if not self.introduce_zk_sparsily else (v + vdot*dt - zk_true)
            self.funcs['eval_g0'] = Function("eval_g0", [uk, xk], 
                                             [vertcat(self.funcs['eval_g0'](uk, xk),
                                                      dyn_expr)], 
                                             ['uk', 'xk'], ['g0'])
            self.funcs['eval_gk'] = Function("eval_gk", [uk, xk],
                                             [vertcat(self.funcs['eval_gk'](uk, xk),
                                                      dyn_expr)],
                                             ['uk', 'xk'], ['gk'])
            
            # create new dynamics and update initialization
            if self.introduce_zk_sparsily:
                self.funcs['expl_dyn'] = Function("expl_dyn", [uk, xk], [vertcat(zk_true, q + zk_true*dt)], ['uk', 'xk'], ['zk'])
                zk_true_init = self.funcs['x_init'](k)[-3*self.model.nb_pendulums:,:]
                self.funcs["u_init"] = Function("u_init", [k], [vertcat(self.funcs['u_init'](k), zk_true_init)], ['k'], ['u_init'])
            else:
                self.funcs['expl_dyn'] = Function("expl_dyn", [uk, xk], [zk], ['uk', 'xk'], ['zk'])
                self.funcs["u_init"] = Function("u_init", [k], [vertcat(self.funcs['u_init'](k), self.funcs['x_init'](k))], ['k'], ['u_init'])
        

        # metadata
        data = {
            "nx": xk.size1(),
            "nu": uk.size1(),
            "N": N,
            "T": T,
            "dt": dt,
            "x0": vertcat(v0, 0, q0).full().flatten().tolist(),
            "lb": self.funcs["lb"]()["lb"].full().flatten().tolist(),
            "ub": self.funcs["ub"]()["ub"].full().flatten().tolist(),
            "lbK": self.funcs["lbK"]()["lbK"].full().flatten().tolist(),
            "ubK": self.funcs["ubK"]()["ubK"].full().flatten().tolist(),
            "u_init": [self.funcs["u_init"]()["u_init"].full().flatten().tolist() for _ in range(N)],
            "x_init": [self.funcs["x_init"](k).full().flatten().tolist() for k in range(N+1)]
        }

        if self.store_casadi_functions:
            folder = "casadi_functions/"
            build_folder = "../../../build_docker/casadi_functions/"
            with open(f"{folder}metadata.json", "w") as f:
                json.dump(data, f, indent=4)
            with open(f"{build_folder}metadata.json", "w") as f:
                json.dump(data, f, indent=4)

            for name, func in self.funcs.items():
                func.save(f"{folder}{name}.casadi")
                func.save(f"{build_folder}{name}.casadi")

    def solve_baked_trajectory(self, q0, v0, T, N):
        self.bake(q0, v0, T, N)

        if self.store_casadi_functions:
            eval_objk = Function.load("casadi_functions/eval_objk.casadi")
            eval_objK = Function.load("casadi_functions/eval_objK.casadi")
            eval_g0 = Function.load("casadi_functions/eval_g0.casadi")
            eval_gk = Function.load("casadi_functions/eval_gk.casadi")
            eval_gK = Function.load("casadi_functions/eval_gK.casadi")
            eval_gk_ineq = Function.load("casadi_functions/eval_gk_ineq.casadi")
            eval_gK_ineq = Function.load("casadi_functions/eval_gK_ineq.casadi")
            lb = Function.load("casadi_functions/lb.casadi")
            ub = Function.load("casadi_functions/ub.casadi")
            lbK = Function.load("casadi_functions/lbK.casadi")
            ubK = Function.load("casadi_functions/ubK.casadi")

            expl_dyn = Function.load("casadi_functions/expl_dyn.casadi")
            impl_dyn = Function.load("casadi_functions/impl_dyn.casadi")
            x_init = Function.load("casadi_functions/x_init.casadi")
            u_init = Function.load("casadi_functions/u_init.casadi")
        else:
            eval_objk = self.funcs['eval_objk']
            eval_objK = self.funcs['eval_objK']
            eval_g0 = self.funcs['eval_g0']
            eval_gk = self.funcs['eval_gk']
            eval_gK = self.funcs['eval_gK']
            eval_gk_ineq = self.funcs['eval_gk_ineq']
            eval_gK_ineq = self.funcs['eval_gK_ineq']
            lb = self.funcs["lb"]
            ub = self.funcs["ub"]
            lbK = self.funcs["lbK"]
            ubK = self.funcs["ubK"]

            expl_dyn = self.funcs['expl_dyn']
            impl_dyn = self.funcs['impl_dyn']
            x_init = self.funcs["x_init"]
            u_init = self.funcs["u_init"]
        
        nu = eval_objk.sparsity_in(0).size1()
        nx = eval_objk.sparsity_in(1).size1()

        opti = Opti()
        # xx = opti.variable(nx, N+1)
        # uu = opti.variable(nu, N)
        # define variables in a stage-wise way
        xx = []; uu = []
        for k in range(N):
            xx.append(opti.variable(nx))
            uu.append(opti.variable(nu))
        xx.append(opti.variable(nx))
        xx = hcat(xx)
        uu = hcat(uu)
        
        obj = 0
        for k in range(N):
            if self.implicit and not self.reformulate:
                opti.subject_to(impl_dyn(uu[:,k], xx[:,k], xx[:,k+1]) == 0)
            else:
                opti.subject_to(xx[:,k+1] == expl_dyn(uu[:,k], xx[:,k]))


            if  k == 0 and eval_g0.sparsity_out(0).size1() > 0:
                opti.subject_to(eval_g0(uu[:, 0], xx[:, 0]) == 0)
            if k > 0 and eval_gk.sparsity_out(0).size1() > 0:
                opti.subject_to(eval_gk(uu[:,k], xx[:,k]) == 0)
            if eval_gk_ineq.sparsity_out(0).size1() > 0:
                opti.subject_to(lb()['lb'] <= (eval_gk_ineq(uu[:,k], xx[:,k]) <= ub()['ub']))

            obj += eval_objk(uu[:,k], xx[:,k])

        if eval_gK.sparsity_out(0).size1() > 0:
            opti.subject_to(eval_gK(xx[:,N]) == 0)
        if eval_gK_ineq.sparsity_out(0).size1() > 0:
            opti.subject_to(lbK()['lbK'] <= (eval_gK_ineq(xx[:,N]) <= ubK()['ubK']))
        obj += eval_objK(xx[:,N])

        opti.minimize(obj)

        uu_init = []; xx_init = []
        for k in range(N):
            opti.set_initial(xx[:,k], x_init(k))
            opti.set_initial(uu[:,k], u_init(k))
            uu_init.append(u_init(k))
            xx_init.append(x_init(k))
        opti.set_initial(xx[:,N], x_init(N))

        if self.implicit and not self.reformulate:
            opti.solver('ipopt', {'error_on_fail': True}, {'max_iter':1, 'tol':1e-4})
        else:
            problem_type = 'ocp_type' if not self.accelerated else 'accelerated_ocp_type'
            fatrop_opts = {
                'max_iter': 400,
                'tol': 1e-4,
                'problem_type': problem_type,
            }            
            if self.accelerated:
                fatrop_opts['linsol_nb_of_dynamics_constraints'] = 3*self.model.nb_pendulums if self.introduce_zk_sparsily else nx
                fatrop_opts['linsol_nb_of_zk_vars'] = 3*self.model.nb_pendulums if self.introduce_zk_sparsily else nx
            opti.solver('fatrop', {'error_on_fail': True, 'structure_detection': 'auto', 'debug': False, 'expand': True}, fatrop_opts)
            
        try:
            if self.opti_to_function:
                opti_f = opti.to_function("ocp_solver", [], [uu, xx], [], ['uu', 'xx'])
                if self.opti_code_generate:
                    opti_f.generate("ocp_solver", {"with_header": True, "cpp": True})
                    print("Compiled C++ solver generated successfully.")
                    import subprocess
                    subprocess.run(["gcc", "-shared", "-fPIC", "-o", "ocp_solver.so", "ocp_solver.cpp"])
                    print("Shared library compiled successfully.")
                    opti_f = external("ocp_solver", "ocp_solver.so")
                    print("External function loaded successfully.")
                
                sol = opti_f()
                uu_sol = sol['uu']
                xx_sol = sol['xx']
            else:
                self.sol = opti.solve()
                xx_sol = self.sol.value(xx)
                uu_sol = self.sol.value(uu)
            
        except RuntimeError as e:
            print("Solver failed with error:", e)
            return {
                'success': False, 'nb_iter': self.get_nb_iters(), 'q_sol': None,
                'v_sol': None, 'F_sol': None, 'z_sol': None
            }

        # xx_sol = self.sol.value(xx)
        # uu_sol = self.sol.value(uu)
        result = {
            'success': True,
            'nb_iter': self.get_nb_iters(),
            'q_sol': xx_sol[3*self.model.nb_pendulums:6*self.model.nb_pendulums, :],
            'v_sol': xx_sol[:3*self.model.nb_pendulums, :],
            'p_sol': xx_sol[6*self.model.nb_pendulums:, :],
            'z_sol': uu_sol[3*self.tracking:3*self.tracking + self.model.nb_pendulums, :],
            'F_sol': uu_sol[3*self.tracking + 1*self.model.nb_pendulums:3*self.tracking + 1*self.model.nb_pendulums + 3*len(self.model.actuated_joint_idxs), :],
            
        }
        return result

    def solve_trajectory(self, q0, v0, T, N):
        assert len(q0) == 3*self.model.nb_pendulums, "Length of q0 must match 3 times the number of pendulums"
        assert len(v0) == 3*self.model.nb_pendulums, "Length of v0 must match 3 times the number of pendulums"

        opti = Opti()
        q = opti.variable(3*self.model.nb_pendulums, N+1)
        v = opti.variable(3*self.model.nb_pendulums, N+1)
        p = opti.variable(1*self.tracking, N+1)
        # dp = opti.variable(1, N)
        s = opti.variable(3*self.tracking, N+1)
        z = opti.variable(1*self.model.nb_pendulums, N)
        F = opti.variable(3*len(self.model.actuated_joint_idxs), N)

        dt = T/N

        # initial conditions
        # opti.subject_to(self.model.g_eq0(q[:, 0], v[:, 0]) == 0) --> enforced using initial conditions
        opti.subject_to(q[:, 0] == q0)
        opti.subject_to(v[:, 0] == v0)
        if self.tracking:
            opti.subject_to(p[:, 0] == 0)

        obj = 0.0
        for k in range(N):
            # dynamics constraints
            idx = k+1 if self.implicit else k
            qdot, vdot = self.model.f(q[:,k], v[:,k], z[:,k], F[:,k])
            opti.subject_to(q[:,k+1] == q[:,k] + v[:,idx]*dt)
            opti.subject_to(v[:,k+1] == v[:,k] + vdot*dt)
            if self.tracking:
                opti.subject_to(p[:,k+1] == p[:,k] + dt)

            # equalities
            eq = self.model.g_eq(q[:,k], v[:,k], z[:,k], F[:,k])
            if self.model.stabilizer is not None:
                eq += self.model.stabilizer(q[:,k], v[:,k])
            opti.subject_to(eq == 0)

            if self.tracking:
                ref_point = self.path_func(p[k], T, q0)
                opti.subject_to(s[:,k] == q[:3, k] - ref_point)

            # inequalities
            opti.subject_to(-self.force_bounds <= (F[:,k] <= self.force_bounds))

            # objective
            obj += 1e-2*sumsqr(F[:,k])
            if self.tracking:
                obj += 1e2*sumsqr(s[:,k])

        # terminal constraint
        if self.tracking:
            opti.subject_to(p[:, -1] == T)

        if not self.tracking:
            obj += sumsqr(v[:,N]) + 1e1*sumsqr(q[:,N] - self.get_q_rest())

        # objective
        opti.minimize(obj)

        opti.set_initial(q, np.tile(q0, (N+1, 1)).T)
        opti.set_initial(v, np.tile(v0, (N+1, 1)).T)
        if self.tracking:
            opti.set_initial(p, np.linspace(0, T, N+1))

        if self.implicit and not self.reformulate:
            opti.solver('ipopt', {}, {'max_iter':400, 'tol':1e-4})
        else:
            problem_type = 'ocp_type' if not self.accelerated else 'accelerated_ocp_type'
            opti.solver('fatrop', {'structure_detection':'auto', 'debug':False}, {'max_iter':400, 'tol':1e-4, 'problem_type': problem_type})
        try:
            self.sol = opti.solve()
        except RuntimeError as e:
            print("Solver failed with error:", e)
            return {
                'success': False, 'nb_iter': self.get_nb_iters(), 'q_sol': None,
                'v_sol': None, 'F_sol': None, 'z_sol': None
            }

        q_sol = self.sol.value(q)
        v_sol = self.sol.value(v)
        F_sol = self.sol.value(F)
        z_sol = self.sol.value(z)
        return {
            'success': True,
            'nb_iter': self.get_nb_iters(),
            'q_sol': q_sol,
            'v_sol': v_sol,
            'F_sol': F_sol,
            'z_sol': z_sol
        }

    def get_nb_iters(self):
        try:
            return self.sol.stats()['iter_count']
        except AttributeError:
            return None