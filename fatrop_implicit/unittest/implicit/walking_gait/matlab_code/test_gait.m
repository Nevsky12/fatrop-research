% Dependencies:
%   - https://github.com/KULeuvenNeuromechanics/PredSim/blob/master/OCP/CollocationScheme.m
%   - CasADi 3.7.1

clear
close all
clc

import casadi.*


solver = 'fatrop';
solver = 'ipopt';

% mesh intervals
N_mesh = 64;

% order of Radau collocation scheme
n_coll = 3;


% musculoskeletal model: dynamics generated from https://codeberg.org/Lars-DHondt/
%   SOC_walking_DHondt2026/src/branch/main/Functions/create_system_dynamics.m
%
% number of coordinates:
%   6: pelvis_tx, pelvis_ty, hip_r, hip_l, knee_r, knee_l
%   9: pelvis_tilt, pelvis_tx, pelvis_ty, hip_r, hip_l, knee_r, knee_l,
%       ankle_r, ankle_l
n_coords = 9;
% Leg joints are driven by ideal torque actuators.
% The model with 9 coordinates can also be muscle-driven.
torque_driven = 0;




%%

% Impose walking at 1.2 m/s with 0.9 strides/s (average adult)
step_time = 1/0.9/2; % time horizon of 1 step
step_length = 1.2*step_time; % forward distance covered in 1 step

% timestep
dt = step_time/N_mesh;


% load function to evaluate system dynamics
%   inputs:
%   - coordinate positions
%   - coordinate velocities
%   - coordinate accelerations
%   - actuation controls
%   outputs:
%   - effort (sum of squared controls)
%   - limit torques (sum of squared) - range of motion via load in ligaments
%   - moment equilibrium error on each coordinate
%
if n_coords == 6
    idx_q_fwd = 1; % index of forward position coordinate - for periodicity
    n_act = 4;
    f_sysdyn = casadi.Function.load('./f_sysdyn_6dof_0mus_12states.casadi');

elseif n_coords == 9
    idx_q_fwd = 2; % index of forward position coordinate - for periodicity
    if torque_driven
        n_act = 6;
        f_sysdyn = casadi.Function.load('./f_sysdyn_9dof_0mus_18states.casadi');

    else
        n_act = 18;
        f_sysdyn = casadi.Function.load('./f_sysdyn_9dof_18mus_18states.casadi');
    end
end

% scale factors for optimisation variables
scale_qdots = 10; % velocities
scale_qddots = 100; % accelerations


%% helper functions

%%% Variables

% states on 1st mesh point
q_mesh_1_SX = SX.sym('q_mesh_1',n_coords-1,1); % except forward position
qdot_mesh_1_SX = SX.sym('qdot_mesh_1',n_coords,1);

% states on kth mesh point
q_mesh_k_SX = SX.sym('q_mesh_k',n_coords,1);
qdot_mesh_k_SX = SX.sym('qdot_mesh_k',n_coords,1);


if strcmp(solver, 'fatrop')
    % State vector is augmented with states on 1st mesh point. This is
    % needed to formulate periodicity within the OCP format of fatrop.
    x_mesh_k_SX = [q_mesh_k_SX; qdot_mesh_k_SX;
                q_mesh_1_SX; qdot_mesh_1_SX];
else
    x_mesh_k_SX = [q_mesh_k_SX; qdot_mesh_k_SX];
end

% states and state derivatives on collocation points
q_coll_k_SX = SX.sym('q_coll_k',n_coords,n_coll);
qdot_coll_k_SX = SX.sym('qdot_coll_k',n_coords,n_coll);
qddot_coll_k_SX = SX.sym('qddot_coll_k',n_coords,n_coll);

% actuation control in mesh interval
act_mesh_k_SX = SX.sym('act_mesh_k',n_act,1);

% "controls" in mesh interval for the OCP format of fatrop
u_mesh_k_SX = [act_mesh_k_SX; 
    q_coll_k_SX(:); qdot_coll_k_SX(:); qddot_coll_k_SX(:)];

nx = length(x_mesh_k_SX);
nu = length(u_mesh_k_SX);

q_k_SX = [q_mesh_k_SX, q_coll_k_SX];
qdot_k_SX = [qdot_mesh_k_SX, qdot_coll_k_SX];

%%% Create OCP functions for kth mesh interval

[~,C,D,B] = CollocationScheme(n_coll,'radau');

% states on mesh point k+1
%   Last Radau collocation point corresponds to next mesh point.
%   States on 1st mesh point are simply passed on.
if strcmp(solver, 'fatrop')
    x_mesh_kp1_SX = [q_k_SX(:,n_coll+1); qdot_k_SX(:,n_coll+1);
        q_mesh_1_SX; qdot_mesh_1_SX];
else
    x_mesh_kp1_SX = [q_k_SX(:,n_coll+1); qdot_k_SX(:,n_coll+1)];
end


% error on collocation equation - path constraint
err_coll_q_SX = q_k_SX*C(:,2:end) - (qdot_coll_k_SX*scale_qdots)*dt;
err_coll_qdot_SX = (qdot_k_SX*scale_qdots)*C(:,2:end) ...
    - (qddot_coll_k_SX*scale_qddots)*dt;


% Evaluate system dynamics function.
% This function is from another project, hence there are dummy inputs and
% unused outputs.
[effort_coll_k_SX, limtorq_coll_k_SX, ~, err_sysdyn_SX, ~] =...
    f_sysdyn([], q_coll_k_SX, qdot_coll_k_SX*scale_qdots, [], ...
    [], qddot_coll_k_SX*scale_qddots, [], 0, 0,...
    act_mesh_k_SX, 0, 0);

% Integrate objective terms over mesh interval
effort_k_SX = effort_coll_k_SX*B(2:end)*dt;
limtorq_k_SX = limtorq_coll_k_SX*B(2:end)*dt;

%   Integrator
f_int = Function('f_int',{x_mesh_k_SX, u_mesh_k_SX},{x_mesh_kp1_SX});
%   Path constraints
f_path = Function('f_path',{x_mesh_k_SX, u_mesh_k_SX},...
    {[err_coll_q_SX; err_coll_qdot_SX/scale_qdots; err_sysdyn_SX/100]});
%   Objective
f_obj = Function('f_obj',{x_mesh_k_SX, u_mesh_k_SX},{effort_k_SX, limtorq_k_SX});

%%% Periodicity and symmetry constraints
% Impose walking to be periodic and symmetric (with 180° phase shift).
% This is formulated as equality constraints on the states, with a mapping
% between left and right side.
%
% k = 1         k = N+1
% ------------------------
% pelvis_tx     pelvis_tx
% pelvis_ty     pelvis_ty
% hip_r         hip_l
% hip_l         hip_r
% knee_r        knee_l
% knee_l        knee_r
%
% The state giving the forward position of the floating base is excluded,
% becaus its initial and final value are prescribed.
%
if n_coords == 6
    err_per_q_SX = q_mesh_1_SX - q_mesh_k_SX([2,4,3,6,5]);
    err_per_qdot_SX = qdot_mesh_1_SX - qdot_mesh_k_SX([1,2,4,3,6,5]);
elseif n_coords == 9
    err_per_q_SX = q_mesh_1_SX - q_mesh_k_SX([1,3,5,4,7,6,9,8]);
    err_per_qdot_SX = qdot_mesh_1_SX - qdot_mesh_k_SX([1,2,3,5,4,7,6,9,8]);
end

if strcmp(solver, 'fatrop')
    % State vector is augmented with states on 1st mesh point.
    f_per = Function('f_per',{x_mesh_k_SX},{[err_per_q_SX; err_per_qdot_SX]});
else
    f_per = Function('f_per',{[q_mesh_1_SX; qdot_mesh_1_SX], x_mesh_k_SX},...
        {[err_per_q_SX; err_per_qdot_SX]});
end

% remove all SX variables from workspace
clearvars("*_SX")

%% variable bounds

if n_coords == 6
    q_lb = [-0.1; 0.5; [-50;-50; -120;-120]*pi/180];
    q_ub = [step_length+0.1; 1.5; [90;90; 10;10]*pi/180];
elseif n_coords == 9
    q_lb = [-1; -0.1; 0.5; [-50;-50; -120;-120; -50;-50]*pi/180];
    q_ub = [1; step_length+0.1; 1.5; [90;90; 10;10; 50;50]*pi/180];
end

qdot_ub = 10*(q_ub-q_lb)/scale_qdots;
qddot_ub = 150*qdot_ub/scale_qddots;

if torque_driven
    act_lb = -ones(n_act,1);
    act_ub = ones(n_act,1);
else
    act_lb = 0.05*ones(n_act,1);
    act_ub = ones(n_act,1);
end

x_lb = [q_lb; -qdot_ub; q_lb(setdiff(1:n_coords, idx_q_fwd)); -qdot_ub];
x_ub = [q_ub; qdot_ub; q_ub(setdiff(1:n_coords, idx_q_fwd)); qdot_ub];

u_lb = [act_lb; repmat(q_lb,n_coll,1); repmat(-qdot_ub,n_coll,1); 
    repmat(-qddot_ub,n_coll,1)];
u_ub = [act_ub; repmat(q_ub,n_coll,1); repmat(qdot_ub,n_coll,1); 
    repmat(qddot_ub,n_coll,1)];

%% initial guess

x=0:N_mesh*2;
q_tx_ig = x/N_mesh*step_length;
q_ty_ig = 0.9*ones(1,N_mesh*2+1);
q_ty_ig = q_ty_ig - 0.05*cos(x/N_mesh/2*pi*4);
q_hip_ig = 20*pi/180 *cos(x/N_mesh/2*pi*2);
q_knee_ig = [-0.1*ones(1,N_mesh+1), 0.5*(cos((x(1:N_mesh)/N_mesh)*pi*2)-1)-0.1];

q_ig = zeros(n_coords,N_mesh+1);
qdot_ig = zeros(n_coords,N_mesh+1);

qdot_ig(idx_q_fwd,:) = 1.2 /scale_qdots;
if n_coords == 6
    q_ig(1,:) = q_tx_ig(1:N_mesh+1);
    q_ig(2,:) = q_ty_ig(1:N_mesh+1);
    q_ig(4,:) = q_hip_ig(1:N_mesh+1);
    q_ig(3,:) = q_hip_ig(N_mesh+1:end);
    q_ig(6,:) = q_knee_ig(1:N_mesh+1);
    q_ig(5,:) = q_knee_ig(N_mesh+1:end);

elseif n_coords == 9
    q_ig(2,:) = q_tx_ig(1:N_mesh+1);
    q_ig(3,:) = q_ty_ig(1:N_mesh+1);
    q_ig(5,:) = q_hip_ig(1:N_mesh+1);
    q_ig(4,:) = q_hip_ig(N_mesh+1:end);
    q_ig(7,:) = q_knee_ig(1:N_mesh+1);
    q_ig(6,:) = q_knee_ig(N_mesh+1:end);

end



x_ig = [q_ig; qdot_ig;
    repmat(q_ig(setdiff(1:n_coords, idx_q_fwd),1),1,N_mesh+1); 
    repmat(qdot_ig(:,1),1,N_mesh+1)];

u_ig = [0.1*ones(n_act,N_mesh); repmat(q_ig(:,2:end),n_coll,1);
    repmat(qdot_ig(:,2:end),n_coll,1); zeros(n_coords*n_coll,N_mesh)];

if ~strcmp(solver, 'fatrop')
    x_lb = x_lb(1:n_coords*2,:);
    x_ub = x_ub(1:n_coords*2,:);
    x_ig = x_ig(1:n_coords*2,:);
end


%% OCP setup
% Based on https://github.com/jgillis/fatrop_demo/blob/master/fatrop_opti.m

opti = Opti();

X = {};
U = {};
for k = 1:N_mesh
    X{end+1} = opti.variable(nx);
    U{end+1} = opti.variable(nu);
end
X{end+1} = opti.variable(nx);

J_act = {};
J_lim = {};
J_reg = {};

for k = 1:N_mesh

    % Multiple shooting gap-closing constraint
    opti.subject_to(X{k+1} - f_int(X{k}, U{k}) == 0);
    
    % Initial constraints
    if k == 1
        % forward position starts at 0
        opti.subject_to(X{1}(idx_q_fwd) == 0); 
        if strcmp(solver, 'fatrop')
            % states on kth mesh point are equal to states on 1st mesh
            % point - because here k=1
            opti.subject_to(X{1}(setdiff(1:n_coords*2, idx_q_fwd)) == ...
                X{1}(n_coords*2+1:end) );
        end 
    end

    % Path constraint
    opti.subject_to(f_path(X{k}, U{k}) == 0);
    % bounds
    opti.subject_to(x_lb < X{k} < x_ub)
    opti.subject_to(u_lb < U{k} < u_ub)

    % Initial guess
    opti.set_initial(X{k}, x_ig(:,k))
    opti.set_initial(U{k}, u_ig(:,k))

    % Final constraints
    if k == N_mesh
        % forward position is imposed step length
        opti.subject_to(X{N_mesh+1}(idx_q_fwd) == step_length);
        if strcmp(solver, 'fatrop')
            opti.subject_to(f_per(X{N_mesh+1}) == 0);
        else
            opti.subject_to(f_per(X{1}(setdiff(1:n_coords*2, idx_q_fwd)), X{N_mesh+1}) == 0);
        end

        opti.set_initial(X{N_mesh+1}, x_ig(:,N_mesh+1))
    end

    % Objective
    [J_act{k}, J_lim{k}] = f_obj(X{k}, U{k});

end

obj = sum([J_act{:}])*1e2 ...   % effort
    + sum([J_lim{:}])*1e-1;     % limit torques (i.e. range of motion)


opti.minimize(obj)


%%

% Solver options
options.expand = true;
options.detect_simple_bounds = true;

options.(solver).tol = 1e-4;
% options.(solver).max_iter = 0;

if strcmp(solver, 'fatrop')
    options.fatrop.mu_init = 0.1;
    options.structure_detection = 'auto';
    options.debug = true;
end

if strcmp(solver, 'ipopt')
    options.ipopt.nlp_scaling_method = 'none';
    options.ipopt.mu_strategy = 'adaptive';
%     options.ipopt.hessian_approximation = 'limited-memory';
%     options.ipopt.recalc_y_feas_tol = 1e-3;
end


opti.solver(solver, options);



diary(['test_2_',solver,'.txt'])

try
    sol = opti.solve_limited();
catch sol_err
    sol = opti.debug;
    warning(sol_err.message)
end

diary off


%%

X_sol = sol.value([X{:}]);
U_sol = sol.value([U{:}]);


q_mesh = X_sol(1:n_coords,:);
qdot_mesh = X_sol(n_coords+1:n_coords*2,:);

% Construct full cycle from solution for half cycle
if n_coords == 6
    q_GC = [q_mesh(:,1:N_mesh), q_mesh([1,2,4,3,6,5],:)];
elseif n_coords == 9
    q_GC = [q_mesh(:,1:N_mesh), q_mesh([1,2,3,5,4,7,6,9,8],:)];
end

q_GC(idx_q_fwd,N_mesh+1:end) = q_GC(idx_q_fwd,N_mesh+1:end) + step_length;
act_GC = [U_sol(n_act/2+1:n_act,:), U_sol(1:n_act/2,:)];

t_GC = linspace(0,step_time*2,N_mesh*2+1);

figure
tiledlayout('flow')

i=1;

if n_coords == 9
    nexttile
    hold on
    plot(t_GC,q_GC(i,:)*180/pi)
    xlabel('time (s)')
    ylabel('(°)')
    title('torso angle')
    i=i+1;
end

nexttile
hold on
plot(t_GC,q_GC(i,:))
xlabel('time (s)')
ylabel('(m)')
title('forward')

i=i+1;
nexttile
hold on
plot(t_GC,q_GC(i,:))
xlabel('time (s)')
ylabel('(m)')
title('vertical')

i=i+2;
nexttile
hold on
plot(t_GC,q_GC(i,:)*180/pi)
xlabel('time (s)')
ylabel('(°)')
title('hip')

i=i+2;
nexttile
hold on
plot(t_GC,q_GC(i,:)*180/pi)
xlabel('time (s)')
ylabel('(°)')
title('knee')

if n_coords == 9
    i=i+2;
    nexttile
    hold on
    plot(t_GC,q_GC(i,:)*180/pi)
    xlabel('time (s)')
    ylabel('(°)')
    title('ankle')
end


