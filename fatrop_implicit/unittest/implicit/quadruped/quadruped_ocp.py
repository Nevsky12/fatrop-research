import casadi
import numpy as np
import pinocchio as pin
import pinocchio.casadi as cpin
import example_robot_data as erd
from pinocchio.visualize import MeshcatVisualizer
import json

import sys
sys.path.append("unittest/implicit/quadruped")
from helper import *

def TestDymamics():
    quad = QuadrupedDynamics(timestep=0.02)
    q_test = casadi.DM(quad.model.nq, 1)
    v_test = casadi.DM(quad.model.nv, 1)
    u_test = casadi.DM(12, 1)
    for i in range(quad.model.nq):
        q_test[i] = 1 - 0.1*i + 0.02*i**2
    for i in range(quad.model.nv):
        v_test[i] = - 2 + 0.524*i**3
    for i in range(12):
        u_test[i] = 3.141592 * (0.5 - i)

    expl, impl = quad.create_discrete_dynamics()
    
    x_test = casadi.vertcat(q_test, v_test)
    x_next = expl(u_test, x_test)

    print("Testing dynamics:")
    print(x_next)

    print("Testing acc func:")
    a, temp = quad.acc_func(q_test, v_test, u_test)
    print(a)

def get_ocp_functions():
    K = 100 - 1
    nq = 3 + 4 + 12
    nx = 2*nq - 1
    nu = 12

    dt = 0.02
    uk_min = -50
    uk_max = 50

    push_vx = 0*1.5
    push_vy = 0*2.0

    base_pos = casadi.MX.sym("base_pos", 3)
    base_vel = casadi.MX.sym("base_vel", 3)
    base_quat = casadi.MX.sym("base_quat", 4)
    base_omega = casadi.MX.sym("base_omega", 3)
    leg_q = casadi.MX.sym("leg_q", 12)
    leq_qdot = casadi.MX.sym("leg_qdot", 12)
    q = casadi.vertcat(base_pos, base_quat, leg_q)
    v = casadi.vertcat(base_vel, base_omega, leq_qdot)

    base_pos_p = casadi.MX.sym("base_pos_p", 3)
    base_vel_p = casadi.MX.sym("base_vel_p", 3)
    base_quat_p = casadi.MX.sym("base_quat_p", 4)
    base_omega_p = casadi.MX.sym("base_omega_p", 3)
    leg_q_p = casadi.MX.sym("leg_q_p", 12)
    leg_v_p = casadi.MX.sym("leg_qdot_p", 12)
    q_p = casadi.vertcat(base_pos_p, base_quat_p, leg_q_p)
    v_p = casadi.vertcat(base_vel_p, base_omega_p, leg_v_p)

    xk = casadi.vertcat(q, v)
    uk = casadi.MX.sym("uk", nu)
    xkp = casadi.vertcat(q_p, v_p)

    standing_body_pos = [0, 0, 0.5292]
    original_body_pos = [0, 0, 0.5292]
    standing_body_quat = [0, 0, 0, 1]
    standing_leg_q = [-0.1, 0.7, -1, -0.1, -0.7, 1,
                       0.1, 0.7, -1,  0.1, -0.7, 1]
    start = [0.0 for _ in range(nx)]
    lb = [uk_min]*(nu)
    ub = [uk_max]*(nu)
    lb_K = []
    ub_K = []

    standing_body_pos[2] += 0.05
    standing_stance = standing_body_pos + standing_body_quat + standing_leg_q
    for i in range(nq): start[i] = standing_stance[i]
    start[nq] = push_vx
    start[nq+1] = push_vy

    x_init = []
    original_stance = original_body_pos + standing_body_quat + standing_leg_q
    for k in range(K+1):
        xx = original_stance.copy()
        for _ in range(nq-1):
            xx.append(0.0)
        xx[nq] = 0
        xx[nq+1] = 0
        x_init.append(xx)
    
    u_init = []
    for k in range(K):
        u_init.append([-4.27557097,   5.91086644,  17.19719031,  -3.27040327,  
                       -4.81348097, -12.73962232,   3.67687864,   4.22256038,  
                       10.28272374,   5.82689115,  -3.42997252, -14.4580922])
        
    eval_objk = casadi.Function("eval_objk", [uk, xk],
        [1e1*casadi.sumsqr(base_pos - original_body_pos) +
         1e3*casadi.sumsqr(base_quat - standing_body_quat) +
         1e1*casadi.sumsqr(v[:3,:]) + 
         1e0*casadi.sumsqr(v) + 
         0*(casadi.sumsqr(uk) + casadi.sumsqr(xk))])
    eval_objK = casadi.Function("eval_objK", [xk],
        [1e3*casadi.sumsqr(v) +
         1e3*(casadi.sumsqr(base_quat - standing_body_quat) +
              casadi.sumsqr(leg_q - standing_leg_q)) +
              0*casadi.sumsqr(xk)])
    
    eval_g0 = casadi.Function("eval_g0", [uk, xk], [xk - start])
    z = casadi.MX.zeros(0,1)
    eval_gk = casadi.Function("eval_gk", [uk, xk], [z])
    eval_gK = casadi.Function("eval_gK", [xk], [z])
    eval_gk_ineq = casadi.Function("eval_gk_ineq", [uk, xk], [uk])
    eval_gK_ineq = casadi.Function("eval_gK_ineq", [xk], [z])

    quad = QuadrupedDynamics(timestep=dt)
    expl, impl = quad.create_discrete_dynamics()

    return {
        "eval_objk": eval_objk,
        "eval_objK": eval_objK,
        "eval_g0": eval_g0,
        "eval_gk": eval_gk,
        "eval_gK": eval_gK,
        "eval_gk_ineq": eval_gk_ineq,
        "eval_gK_ineq": eval_gK_ineq,
        "start": start,
        "x_init": x_init,
        "u_init": u_init,
        "lb": lb,
        "ub": ub,
        "lb_K": lb_K,
        "ub_K": ub_K,
        "dt": dt,
        "K": K,
        "expl": expl,
    }

def SolveOcp3():
    opti = casadi.Opti()

    data = get_ocp_functions()

    dt = data["dt"]
    N = data["K"]

    quad = QuadrupedDynamics(timestep=dt)
    rhs = quad.acc_func

    qq_list = []
    vv_list = []
    uu_list = []
    for k in range(N+1):
        qq_list.append(opti.variable(quad.model.nq))
        vv_list.append(opti.variable(quad.model.nv))
        if k < N:
            uu_list.append(opti.variable(12))
    qq = casadi.horzcat(*qq_list)
    vv = casadi.horzcat(*vv_list)
    uu = casadi.horzcat(*uu_list)

    start = data["start"]
    q0 = np.array(start[:quad.model.nq])
    v0 = np.array(start[quad.model.nq:])

    # opti.subject_to(qq[:,0] == q0)
    # opti.subject_to(vv[:,0] == v0)
    opti.subject_to(data["eval_g0"](uu[:,0], casadi.vertcat(qq[:,0], vv[:,0])) == 0)

    for k in range(N):
        qk = qq[:,k]
        vk = vv[:,k]
        uk = uu[:,k]

        x = casadi.vertcat(qk, vk)
        # x_next = quad.forward_casadi(x, uk)
        x_next = data["expl"](uk, x)
        q_next = x_next[:quad.model.nq]
        v_next = x_next[quad.model.nq:]

        opti.subject_to(qq[:,k+1] == q_next)
        opti.subject_to(vv[:,k+1] == v_next)

        # torque limits
        # opti.subject_to(opti.bounded(-50, uk, 50))
        # opti.subject_to(uk <= 50)
        # opti.subject_to(uk >= -50)
        opti.subject_to(-50 <= (uk <= 50))
       
    obj = 0
    for k in range(N):
        qk = qq[:,k]
        vk = vv[:,k]
        uk = uu[:,k]
        obj += data["eval_objk"](uk, casadi.vertcat(qk, vk))
    qN = qq[:,N]
    vN = vv[:,N]
    obj += data["eval_objK"](casadi.vertcat(qN, vN))
    opti.minimize(obj)

    for k in range(N+1):
        opti.set_initial(qq[:,k], data["x_init"][0][:quad.model.nq])
    for k in range(N):
        opti.set_initial(uu[:,k], data["u_init"][0])

    p_opts = {"structure_detection":'auto'}
    s_opts = {"max_iter":1000, 'mu_init':0.1}
    opti.solver("fatrop", p_opts, s_opts)

    try:
        sol = opti.solve()

        q_opt = sol.value(qq)
        v_opt = sol.value(vv)
        u_opt = sol.value(uu)
        t_opt = np.arange(N+1)*dt
        return {"t": t_opt, "q": q_opt, "v": v_opt, "u": u_opt, "dt": dt, "N": N}, \
            {"model": quad.model, "visual_model": quad.visual_model, 
                "collision_model": quad.collision_model}
    except:
        return None, None

def SolveOcp2():
    opti = casadi.Opti()

    data = get_ocp_functions()

    dt = data["dt"]
    N = data["K"]

    quad = QuadrupedDynamics(timestep=dt)
    rhs = quad.acc_func

    qq = opti.variable(quad.model.nq, N+1)
    vv = opti.variable(quad.model.nv, N+1)
    uu = opti.variable(12, N)

    start = data["start"]
    q0 = np.array(start[:quad.model.nq])
    v0 = np.array(start[quad.model.nq:])

    opti.subject_to(qq[:,0] == q0)
    opti.subject_to(vv[:,0] == v0)

    for k in range(N):
        qk = qq[:,k]
        vk = vv[:,k]
        uk = uu[:,k]

        x = casadi.vertcat(qk, vk)
        x_next = quad.forward_casadi(x, uk)
        q_next = x_next[:quad.model.nq]
        v_next = x_next[quad.model.nq:]

        opti.subject_to(qq[:,k+1] == q_next)
        opti.subject_to(vv[:,k+1] == v_next)

        # torque limits
        opti.subject_to(opti.bounded(-50, uk, 50))
       
    obj = 0
    for k in range(N):
        qk = qq[:,k]
        vk = vv[:,k]
        uk = uu[:,k]
        obj += data["eval_objk"](uk, casadi.vertcat(qk, vk))
    qN = qq[:,N]
    vN = vv[:,N]
    obj += data["eval_objK"](casadi.vertcat(qN, vN))
    opti.minimize(obj)

    for k in range(N+1):
        opti.set_initial(qq[:,k], data["x_init"][0][:quad.model.nq])
    for k in range(N):
        opti.set_initial(uu[:,k], data["u_init"][0])

    p_opts = {"expand": True}
    s_opts = {"max_iter":1000}
    opti.solver("ipopt", p_opts, s_opts)

    try:
        sol = opti.solve()

        q_opt = sol.value(qq)
        v_opt = sol.value(vv)
        u_opt = sol.value(uu)
        t_opt = np.arange(N+1)*dt
        return {"t": t_opt, "q": q_opt, "v": v_opt, "u": u_opt, "dt": dt, "N": N}, \
            {"model": quad.model, "visual_model": quad.visual_model, 
                "collision_model": quad.collision_model}
    except:
        return None, None

def SolveOcp():
    opti = casadi.Opti()

    dt = 0.02
    N = 100

    quad = QuadrupedDynamics(timestep=dt)
    rhs = quad.acc_func

    qq = opti.variable(quad.model.nq, N+1)
    vv = opti.variable(quad.model.nv, N+1)
    uu = opti.variable(12, N)

    q0 = quad.model.referenceConfigurations["standing"]
    q0 = pin.normalize(quad.model, q0)
    q0[2] += 0.05
    print(f"q0: {q0}")
    print(get_ocp_functions()["x_init"][0])

    v0 = np.zeros(quad.model.nv)
    v0[0] = 1*1.5
    v0[1] = 1*2.0

    opti.subject_to(qq[:,0] == q0)
    opti.subject_to(vv[:,0] == v0)

    for k in range(N):
        qk = qq[:,k]
        vk = vv[:,k]
        uk = uu[:,k]

        x = casadi.vertcat(qk, vk)
        x_next = quad.forward_casadi(x, uk)
        q_next = x_next[:quad.model.nq]
        v_next = x_next[quad.model.nq:]

        opti.subject_to(qq[:,k+1] == q_next)
        opti.subject_to(vv[:,k+1] == v_next)

        # torque limits
        opti.subject_to(opti.bounded(-50, uk, 50))

    control_penalty = casadi.sumsqr(uu)
    terminal_standing_still_penalty = casadi.sumsqr(vv[:,N])
    n = 0
    reference_joint_stance = quad.model.referenceConfigurations["standing"][n:]
    reference_body_position = reference_joint_stance[0:3]
    reference_body_orientation = reference_joint_stance[3:7]
    reference_leg_joints = reference_joint_stance[7:]
    reference_body_position_penalty = casadi.sumsqr(qq[0:3,N] - reference_body_position.reshape(-1,1))
    reference_body_orientation_penalty = casadi.sumsqr(qq[3:7,N] - reference_body_orientation.reshape(-1,1))
    reference_leg_joints_penalty = casadi.sumsqr(qq[7:,N] - reference_leg_joints.reshape(-1,1))
    terminal_reference_penalty = casadi.sumsqr(qq[3:,N] - reference_joint_stance[3:].reshape(-1,1))

    movement_penalty = casadi.sumsqr(vv)
    body_movement_penalty = casadi.sumsqr(vv[:3,:])

    opti.minimize(#0*control_penalty + 
                  1e3*terminal_standing_still_penalty + 
                  1e1*reference_body_position_penalty +
                  1e3*reference_body_orientation_penalty + 
                  #0*reference_leg_joints_penalty + 
                  1e0*movement_penalty + 
                  1e1*body_movement_penalty +
                  1e3*terminal_reference_penalty)

    data = get_ocp_functions()
    opti.set_initial(qq, np.tile(q0.reshape(-1,1), (1,N+1)))
    for k in range(N):
        opti.set_initial(uu[:,k], 
                         [-4.27557097,   5.91086644,  17.19719031,  -3.27040327,  
                          -4.81348097, -12.73962232,   3.67687864,   4.22256038,  
                          10.28272374,   5.82689115,  -3.42997252, -14.4580922])

    p_opts = {"expand": True}
    s_opts = {"max_iter":1000}
    opti.solver("ipopt", p_opts, s_opts)

    try:
        sol = opti.solve()

        q_opt = sol.value(qq)
        v_opt = sol.value(vv)
        u_opt = sol.value(uu)
        t_opt = np.arange(N+1)*dt
        return {"t": t_opt, "q": q_opt, "v": v_opt, "u": u_opt, "dt": dt, "N": N}, \
            {"model": quad.model, "visual_model": quad.visual_model, 
                "collision_model": quad.collision_model}
    except:
        return None, None

def visualize_ocp_result(result, models):
    N = result["N"]
    q_opt = result["q"]
    v_opt = result["v"]
    states_ = []
    for k in range(N+1):
        xk = np.concatenate((q_opt[:,k], v_opt[:,k]))
        states_.append(xk)

        if k == 0:
            # copy the first state to have a longer video
            time_to_add = 2.0
            steps_to_add = int(np.ceil(time_to_add / result["dt"]))
            for _ in range(steps_to_add):
                states_.append(states_[0])
    
    time = result["t"]
    controls = result["u"]

    ##########################
    ### VISUALIZE CONTROLS ###
    ##########################
    import matplotlib.pyplot as plt
    
    hip_abduction_color = "tab:blue"
    hip_flexion_color = "tab:orange"
    knee_flexion_color = "tab:green"
    legs = ["LF", "RF", "LH", "RH"]

    print("Control dims:")
    print(controls.shape)

    plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.title(legs[i])
        plt.plot(time[:-1], controls[i*3+0,:], label="HAA", color=hip_abduction_color)
        plt.plot(time[:-1], controls[i*3+1,:], label="HFE", color=hip_flexion_color)
        plt.plot(time[:-1], controls[i*3+2,:], label="KFE", color=knee_flexion_color)
        plt.xlabel("Time [s]")
        plt.ylabel("Torque [Nm]")
        plt.xlim([time[0], time[-1]])

    # create space below for a single legend
    plt.subplots_adjust(bottom=0.15)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', ncol=3)
    plt.tight_layout()
    plt.savefig("unittest/implicit/quadruped/figures/quadruped_controls.png", dpi=300)
    # plt.show()

    print(f"controls in last time-step: {controls[:,-1]}")

    ##############################
    ### VISUALIZE FOOT HEIGHTS ###
    ##############################
    plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.title(legs[i])
        foot_names = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
        foot_id = models["model"].getFrameId(foot_names[i])
        heights = []
        for k in range(N+1):
            qk = np.array(q_opt[:,k])
            data = models["model"].createData()
            pin.forwardKinematics(models["model"], data, qk)
            pin.updateFramePlacements(models["model"], data)
            p = data.oMf[foot_id].translation
            heights.append(p[2])
        plt.plot(time, heights)
        plt.xlabel("Time [s]")
        plt.ylabel("Height [m]")
        plt.xlim([time[0], time[-1]])
    
    plt.tight_layout()
    plt.savefig("unittest/implicit/quadruped/figures/quadruped_foot_heights.png", dpi=300)
    # plt.show()

    ###################################
    ### VISUALIZE ROBOT BODY MOTION ###
    ###################################
    plt.figure()
    for i in range(3):
        plt.subplot(3,1,i+1)
        if i == 0:
            plt.title("Body Position")
            plt.ylabel("X [m]")
        elif i == 1:
            plt.ylabel("Y [m]")
        else:
            plt.ylabel("Z [m]")
            plt.xlabel("Time [s]")
        plt.plot(time, q_opt[i,:])
        plt.xlim([time[0], time[-1]])
    
    plt.tight_layout()
    plt.savefig("unittest/implicit/quadruped/figures/quadruped_body_position.png", dpi=300)
    # plt.show()
    
    #################################
    ### VISUALIZE REACTION FORCES ###
    #################################
    plt.figure()
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.title(legs[i])
        fx = []
        fy = []
        fz = []
        for k in range(N+1):
            qk = np.array(q_opt[:,k])
            vk = np.array(v_opt[:,k])
            data = models["model"].createData()
            pin.forwardKinematics(models["model"], data, qk, vk)
            pin.updateFramePlacements(models["model"], data)
            foot_names = ["LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"]
            foot_id = models["model"].getFrameId(foot_names[i])
            p = data.oMf[foot_id].translation
            v_frame = pin.getFrameVelocity(models["model"], data, foot_id, pin.LOCAL_WORLD_ALIGNED)
            v_foot = v_frame.linear
            f_contact = ground_reaction_force(p, v_foot)
            fx.append(f_contact[0])
            fy.append(f_contact[1])
            fz.append(f_contact[2])
        plt.plot(time, fx, label="fx")
        plt.plot(time, fy, label="fy")
        plt.plot(time, fz, label="fz")
        plt.xlabel("Time [s]")
        plt.ylabel("Force [N]")
        plt.xlim([time[0], time[-1]])
    
    # create space below for a single legend
    plt.subplots_adjust(bottom=0.15)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.figlegend(handles, labels, loc='lower center', ncol=3)
    plt.tight_layout()
    plt.savefig("unittest/implicit/quadruped/figures/quadruped_reaction_forces.png", dpi=300)
    # plt.show()
    plt.close()

    ##############################
    ### VISUALIZE ROBOT MOTION ###
    ##############################
    states_ = np.stack(states_).T
    viz = MeshcatVisualizer(
        model=models["model"],
        collision_model=models["collision_model"],
        visual_model=models["visual_model"],
    )
    qs_ = states_[: models["model"].nq, :].T
    viz.initViewer()
    viz.loadViewerModel("pinocchio")
    import time
    time.sleep(5.0)
    viz.play(q_trajectory=qs_, dt=results["dt"])

    time.sleep(N*results["dt"] + 5.0)
           

if __name__ == "__main__":
    TestDymamics()
    # quad = QuadrupedDynamics(timestep=0.02)
    # quad.test_dynamics()

    folder_name = "unittest/implicit/quadruped/json_files/"

    # load settings file
    with open(folder_name + "quadruped_ocp_settings.json", "r") as f:
        settings = json.load(f)

    if not settings["LOAD_OCP_SOLUTION"]:
        # results, models = SolveOcp()
        # results, models = SolveOcp2()
        results, models = SolveOcp3()

        if settings["STORE_OCP_SOLUTION"]:
            # save results to a json file
            with open(folder_name + "quadruped_ocp_solution.json", "w") as f:
                json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in results.items()}, f)

    else:
        # load results from a json file
        with open(folder_name + "quadruped_ocp_solution.json", "r") as f:
            results = json.load(f)
        results["q"] = np.array(results["q"])
        results["v"] = np.array(results["v"])
        results["u"] = np.array(results["u"])
        results["t"] = np.array(results["t"])
        from pinocchio.robot_wrapper import RobotWrapper
        robot = erd.load("anymal")
        models = {}
        models["model"] = robot.model
        models["collision_model"] = robot.collision_model
        models["visual_model"] = robot.visual_model

    visualize_ocp_result(results, models)
