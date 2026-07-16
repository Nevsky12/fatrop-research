% Load functions
folder = "casadi_funcs";
f_objk = casadi.Function.load(fullfile(folder, "f_objk.casadi"));
f_g0 = casadi.Function.load(fullfile(folder, "f_g0.casadi"));
f_gk = casadi.Function.load(fullfile(folder, "f_gk.casadi"));
f_gK = casadi.Function.load(fullfile(folder, "f_gN.casadi"));
f_gk_ineq = casadi.Function.load(fullfile(folder, "f_gk_ineq.casadi"));
f_gap = casadi.Function.load(fullfile(folder, "f_gap.casadi"));

f_lb = casadi.Function.load(fullfile(folder, "f_lb.casadi"));
f_ub = casadi.Function.load(fullfile(folder, "f_ub.casadi"));
f_x_init = casadi.Function.load(fullfile(folder, "f_init_x.casadi"));
f_u_init = casadi.Function.load(fullfile(folder, "f_init_u.casadi"));

% Derive dimensions
xx_init = f_x_init().x_init;
uu_init = f_u_init().u_init;
lb = f_lb().lb;
ub = f_ub().ub;
nx = size(xx_init, 1);
nu = size(uu_init, 1);
N = size(xx_init, 2) - 1;

% Setup opti instance
opti = casadi.Opti();
xx = {};
uu = {};
for k = 1:N
    xx{k} = opti.variable(nx, 1);
    uu{k} = opti.variable(nu, 1);
end
xx{N+1} = opti.variable(nx, 1);  % add state at mesh point N

obj = 0;
for k = 1:N
    % gap-closing constraint
    opti.subject_to(xx{k+1} == f_gap(xx{k}, uu{k}));
    
    % equality constraint
    if k == 1
        opti.subject_to(f_g0(xx{k}, uu{k}) == 0);
    else
        opti.subject_to(f_gk(xx{k}, uu{k}) == 0);
    end
    
    % inequality constraint
    opti.subject_to(lb <= f_gk_ineq(xx{k}, uu{k}) <= ub);
    
    % initial guess
    opti.set_initial(xx{k}, xx_init(:, k));
    opti.set_initial(uu{k}, uu_init(:, k));
    
    % final constraints
    if k == N - 1
        opti.subject_to(f_gK(xx{k+1}) == 0);
        opti.set_initial(xx{k+1}, xx_init(:, k+1));
    end
    
    % objective
    obj = obj + f_objk(xx{k}, uu{k});
end

opti.minimize(obj);

opti.solver('fatrop', ...
            struct('expand', true, ...
                   'detect_simple_bounds', true, ...
                   'structure_detection', 'auto'), ...
            struct('tol', 1e-4, ...
                   'mu_init', 0.1));

sol = opti.solve();
