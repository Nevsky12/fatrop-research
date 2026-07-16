#ifndef __MANIPULATOR_PATH_FOLLOWER_GENERATOR_HPP__
#define __MANIPULATOR_PATH_FOLLOWER_GENERATOR_HPP__

#include "fatrop/context/context.hpp"
#include "fatrop/context/generic.hpp"
#include "fatrop/ocp/ocp_abstract.hpp"
#include "../test_problem/implicit_test_problem.hpp"    
#include "../test_problem/explicit_test_problem.hpp"
#include "ocp_interface_generator.hpp"

#include <casadi/casadi.hpp>

using namespace casadi;
using namespace fatrop;

#define TORQUE_CONTROL

#ifndef TORQUE_CONTROL
class ManipulatorPathFollower : public InterfaceGenerator {
    public:
        // Constructor
        ManipulatorPathFollower(){
            // define initial guess
            x_init_ = std::vector<std::vector<double>>(K_+1, std::vector<double>(nx_, 0.0));
            u_init_ = std::vector<std::vector<double>>(K_, std::vector<double>(nu_, 0.0));
            if (with_progress_variable){
                for (int k = 0; k < K_+1; k++){
                    if (k < K_){
                        u_init_[k][n] = dt_;
                    }
                    x_init_[k][2*n] = k*dt_;
                }
            }

            // define path to follow
            eval_path_ = get_path_function();

            // define bounds
            for (int i = 0; i < n; i++){
                lb_[i] = -accel_bound_; ub_[i] = accel_bound_;
            }
            lb_[n] = -1; ub_[n] = path_tol_*path_tol_;
            if (with_progress_variable){
                lb_[n+1] = 0.5*dt_; ub_[n+1] = 1.5*dt_;
            }

            // define functions
            eval_objk_ = Function("eval_objk", {uk_, xk_}, {sumsqr(qdd_)});
            eval_objK_ = Function("eval_objK", {xk_}, {0});
            
            MX z = MX::zeros(0,1);
            MX s_temp = (with_slack) ? s_ : ee_pos(MXVector{q_})[0] - eval_path_(MXVector{p_})[0];
            eval_g0_ = Function("eval_g0", {uk_, xk_}, {xk_ - start_});
            eval_gk_ = Function("eval_gk", {uk_, xk_}, {with_slack ? ee_pos(MXVector{q_})[0] - eval_path_(MXVector{p_})[0] - s_ : z});
            if (with_progress_variable){
                eval_gK_ = Function("eval_gK", {xk_}, {p_ - T_});
                eval_gk_ineq_ = Function("eval_gk_ineq", {uk_, xk_}, {vertcat(uk_(Slice(0,n)), sumsqr(s_temp), uk_(n))});
            } else {
                eval_gK_ = Function("eval_gK", {xk_}, {z});
                eval_gk_ineq_ = Function("eval_gk_ineq", {uk_, xk_}, {vertcat(uk_(Slice(0,n)), sumsqr(s_temp))});
            }
            eval_gK_ineq_ = Function("eval_gK_ineq", {xk_}, {z});

            // define dynamics
            Function rhs = Function("rhs", {uk_, xk_}, {vertcat(qd_, uk_(Slice(0,n)), 1)});
            if (with_progress_variable){
                expl_dyn_ = Function("expl_dyn", {uk_, xk_}, {xk_ + dp_ * rhs(MXVector{uk_, xk_})[0]});
                impl_dyn_ = Function("impl_dyn", {uk_, xk_, xkp_}, {xkp_ - (xk_ + dp_ * rhs(MXVector{uk_, xkp_})[0])});
            } else {
                expl_dyn_ = Function("expl_dyn", {uk_, xk_}, {xk_ + dt_ * rhs(MXVector{uk_, xk_})[0]});
                impl_dyn_ = Function("impl_dyn", {uk_, xk_, xkp_}, {xkp_ - (xk_ + dt_ * rhs(MXVector{uk_, xkp_})[0])});
            }
        };

        Function get_path_function(){
            MX t = MX::sym("t");
            MX q = MX::zeros(n, 1);
            for (int i = 0; i < n; i++){
                q(i) = 0.5 * (1 - cos(t*(i+1)));
            }
            return Function("eval_path", {t}, {ee_pos(MXVector{q})[0]});
        }

        virtual ImplicitTestProblem PrepareImplicit(){
            return ImplicitTestProblem(K_, nx_, nu_, 
                    x_init_, u_init_, 
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk_, eval_objK_, eval_gk_, eval_g0_, eval_gK_, eval_gk_ineq_, eval_gK_ineq_,
                    impl_dyn_);
        }

        virtual ExplicitTestProblem PrepareRootFinder(){
            Function eval_dynamics_equation_implicit = Function("eval_dynamics_equation", {xkp_, uk_, xk_}, {impl_dyn_(MXVector{uk_, xk_, xkp_})[0]});
            Function rf = rootfinder("rf", "newton", eval_dynamics_equation_implicit);
            Function explicit_rootfinder = Function("explicit_rootfinder", {uk_, xk_}, {rf(MXVector{xk_, uk_, xk_})});

            return ExplicitTestProblem(K_, nx_, nu_, 
                    x_init_, u_init_, 
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk_, eval_objK_, eval_gk_, eval_g0_, eval_gK_, eval_gk_ineq_, eval_gK_ineq_,
                    explicit_rootfinder);
        }

        virtual ExplicitTestProblem PrepareExplicit(){
            return ExplicitTestProblem(
                    K_, nx_, nu_, 
                    x_init_, u_init_,
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk_, eval_objK_, eval_gk_, eval_g0_, eval_gK_,
                    eval_gk_ineq_, eval_gK_ineq_, expl_dyn_);
        }

        virtual ExplicitTestProblem PrepareReformulated(){
            MX zk = MX::sym("zk", nx_);
            MX uk_aug = vertcat(uk_, zk);
            MXVector ukxk = {uk_, xk_};

            Function eval_objk = Function("eval_objk", {uk_aug, xk_}, {eval_objk_(ukxk)[0]});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_aug, xk_}, {eval_gk_ineq_(ukxk)[0]});

            MXVector in = {uk_, xk_, zk};
            MXVector out = impl_dyn_(in);
            Function eval_g0 = Function("eval_g0", {uk_aug, xk_}, {vertcat(eval_g0_(ukxk)[0], out[0])});
            Function eval_gk = Function("eval_gk", {uk_aug, xk_}, {vertcat(eval_gk_(ukxk)[0], out[0])});

            Function eval_dynamics_equation_reformulated = Function("eval_dynamics_equation", {uk_aug, xk_}, {zk});

            std::vector<std::vector<double>> u_init(K_, std::vector<double>(nu_ + nx_, 0.0));
            for (int k = 0; k < K_-1; k++){
                for (int i = 0; i < nu_; i++){
                    u_init[k][i] = u_init_[k][i];
                }
                for (int i = 0; i < nx_; i++){
                    u_init[k][nu_ + i] = x_init_[k][i];
                }
            }

            return ExplicitTestProblem(
                    K_, nx_, nu_+nx_, 
                    x_init_, u_init,
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk, eval_objK_, eval_gk, eval_g0, eval_gK_,
                    eval_gk_ineq, eval_gK_ineq_,
                    eval_dynamics_equation_reformulated);
        }


        void SolveOptiInstance(){
            Opti opti = Opti();
            int N = K_-1;

            std::vector<MX> x_list = {};
            std::vector<MX> u_list = {};
            for (int k = 0; k < N + 1; k++){
                x_list.push_back(opti.variable(nx_));
                if (k < N){
                    u_list.push_back(opti.variable(nu_));
                }
            }
            MX xx = vertcat(x_list);
            MX uu = vertcat(u_list);

            if (eval_g0_.sparsity_out(0).size1() > 0){
                opti.subject_to(eval_g0_(MXVector{u_list[0], x_list[0]})[0] == DM::zeros(eval_g0_.sparsity_out(0).size1()));
            }

            MX obj = 0;        
            for (int k = 0; k < N; k++){
                MX xk = x_list[k];
                MX uk = u_list[k];

                MX x_next = expl_dyn_(MXVector{uk, xk})[0];
                opti.subject_to(x_list[k+1] == x_next);

                if (eval_gk_.sparsity_out(0).size1() > 0){
                    opti.subject_to(eval_gk_(MXVector{uk, xk})[0] == DM::zeros(eval_gk_.sparsity_out(0).size1()));
                }

                if (eval_gk_ineq_.sparsity_out(0).size1() > 0){
                    opti.subject_to(lb_ <= eval_gk_ineq_(MXVector{uk, xk})[0] <= ub_);
                }

                obj += eval_objk_(MXVector{uk, xk})[0];
            }

            if (eval_gK_.sparsity_out(0).size1() > 0){
                opti.subject_to(eval_gK_(MXVector{x_list[N]})[0] == DM::zeros(eval_gK_.sparsity_out(0).size1()));
            }
            if (eval_gK_ineq_.sparsity_out(0).size1() > 0){
                opti.subject_to(lb_K_ <= eval_gK_ineq_(MXVector{x_list[N]})[0] <= ub_K_);
            }

            obj += eval_objK_(MXVector{x_list[N]})[0];

            opti.minimize(obj);

            for (int k = 0; k < N+1; k++){
                for (int i = 0; i < nx_; i++){
                    opti.set_initial(x_list[k](i), x_init_[k][i]);
                }
                if (k < N){
                    for (int i = 0; i < nu_; i++){
                        opti.set_initial(u_list[k](i), u_init_[k][i]);
                    }
                }
            }

            // define options
            Dict casadi_opts;
            Dict solver_opts;
            // casadi_opts["structure_detection"] = "auto";
            // casadi_opts["fatrop.print_level"] = 12;
            // casadi_opts["fatrop.mu_init"] = 0.1;
            // solver_opts["max_iter"] = 100;
            // solver_opts["mu_init"] = 0.1;
            // solver_opts["print_level"] = 7;

            opti.solver("ipopt", casadi_opts, solver_opts);

            try{
                opti.solve();
            } catch (std::exception& e){
                std::cout << "Exception: " << e.what() << std::endl;
            }
        };

        virtual json GetJsonData(){
            json j;
            j["problem_name"] = "path_follower";
            j["K"] = K_;
            j["nx"] = nx_;
            j["nu"] = nu_;
            j["dt"] = dt_;
            j["uk_min"] = accel_bound_;
            j["start"] = start_;
            return j;
        }

        virtual std::string GetInterfaceName(){ return "path_follower";};
        virtual std::string GetFileNameAppendix(){ return "";};
        void PrintDimensions(){
            std::cout << "nx: " << nx_ << std::endl;
            std::cout << "nu: " << nu_ << std::endl;
            std::cout << "ng: " << eval_gk_.sparsity_out(0).size1() << std::endl;
        }

        // std::unique_ptr<PinocchioCasadi> pc_;
    private:
        int K_ = 70 + 1;
        double T_ = 3.0;
        double dt_ = T_ / (K_ - 1);
        double accel_bound_ = 3;
        double path_tol_ = 0.05;

        bool with_slack = true;
        bool with_progress_variable = true;

        int n = 6;
        int nx_ = 2*n + 1*with_progress_variable;
        int nu_ = n + 3*with_slack + 1*with_progress_variable;

        MX q_ = MX::sym("q", n);
        MX qd_ = MX::sym("qd", n);
        MX qdd_ = MX::sym("qdd", n);
        MX p_ = MX::sym("p", 1*with_progress_variable);
        MX dp_ = MX::sym("dp", 1*with_progress_variable);
        MX s_ = MX::sym("s", 3*with_slack);

        MX q_p_ = MX::sym("q_p", n);
        MX qd_p_ = MX::sym("qd_p", n);
        MX p_p_ = MX::sym("p_p", 1*with_progress_variable);
        
        MX xk_ = vertcat(q_, qd_, p_);
        MX uk_ = vertcat(qdd_, dp_, s_);
        MX xkp_ = vertcat(q_p_, qd_p_, p_p_);

        Function fk_ = Function::load("ur10_fk.casadi");
        Function ee_pos = Function("ee_pos", {q_}, {fk_(q_)[n](Slice(0,3), 3)});
        Function eval_path_;
        std::vector<double> start_ = std::vector<double>(nx_, 0.0);

        std::vector<double> lb_ = std::vector<double>(n + 1 + 1*with_progress_variable);
        std::vector<double> ub_ = std::vector<double>(n + 1 + 1*with_progress_variable);
        std::vector<double> lb_K_ = {};
        std::vector<double> ub_K_ = {};

        std::vector<std::vector<double>> x_init_;
        std::vector<std::vector<double>> u_init_;

        Function eval_objk_;
        Function eval_objK_;
        Function eval_g0_;
        Function eval_gk_;
        Function eval_gK_;
        Function eval_gk_ineq_;
        Function eval_gK_ineq_;

        Function expl_dyn_;
        Function impl_dyn_;
};
#else 
class ManipulatorPathFollower : public InterfaceGenerator {
    public:
        // Constructor
        ManipulatorPathFollower(){
            // define initial guess
            x_init_ = std::vector<std::vector<double>>(K_+1, std::vector<double>(nx_, 0.0));
            u_init_ = std::vector<std::vector<double>>(K_, std::vector<double>(nu_, 0.0));
            for (int k = 0; k < K_+1; k++){
                if (k < K_){
                    u_init_[k][2*n] = dt_;
                    for (int i = 0; i < n; i++){
                        u_init_[k][n+i] = tau_eq_[i];
                    }
                }
                x_init_[k][2*n] = k*dt_;
            }

            // define path to follow
            eval_path_ = get_path_function();

            // define bounds
            for (int i = 0; i < n; i++){
                lb_[i] = -tau_lim_[i]; ub_[i] = tau_lim_[i];
            }
            lb_[n] = -1; ub_[n] = path_tol_*path_tol_;
            lb_[n+1] = 0.5*dt_; ub_[n+1] = 1.5*dt_;

            // define functions
            eval_objk_ = Function("eval_objk", {uk_, xk_}, {sumsqr(qdd_) + 0*1.0e-4*sumsqr(dp_ - dt_)});
            eval_objK_ = Function("eval_objK", {xk_}, {0});
            
            MX z = MX::zeros(0,1);
            MX s_temp = (with_slack) ? s_ : ee_pos(MXVector{q_})[0] - eval_path_(MXVector{p_})[0];
            eval_g0_ = Function("eval_g0", {uk_, xk_}, {xk_ - start_});
            if (with_accel_variables){
                eval_gk_ = Function("eval_gk", {uk_, xk_}, {
                    vertcat(
                        with_slack ? ee_pos(MXVector{q_})[0] - eval_path_(MXVector{p_})[0] - s_ : z,
                        tau_ - id_(MXVector{q_, qd_, qdd_})[0]
                    )
                });
            } else {
                eval_gk_ = Function("eval_gk", {uk_, xk_}, {
                    vertcat(
                        with_slack ? ee_pos(MXVector{q_})[0] - eval_path_(MXVector{p_})[0] - s_ : z,
                        tau_ - id_(MXVector{q_, qd_, qdd_})[0]
                    )
                });
            }
            eval_gK_ = Function("eval_gK", {xk_}, {p_ - T_});
            eval_gk_ineq_ = Function("eval_gk_ineq", {uk_, xk_}, {vertcat(uk_(Slice(0,n)), sumsqr(s_temp), dp_)});
            eval_gK_ineq_ = Function("eval_gK_ineq", {xk_}, {z});

            // define dynamics
            Function rhs = Function("rhs", {uk_, xk_}, {vertcat(qd_, uk_(Slice(0,n)), 1)});
            expl_dyn_ = Function("expl_dyn", {uk_, xk_}, {xk_ + dp_ * rhs(MXVector{uk_, xk_})[0]});
            impl_dyn_ = Function("impl_dyn", {uk_, xk_, xkp_}, {xkp_ - (xk_ + dp_ * rhs(MXVector{uk_, xkp_})[0])});
        };

        Function get_path_function(){
            MX t = MX::sym("t");
            MX q = MX::zeros(n, 1);
            for (int i = 0; i < n; i++){
                // q(i) = 0.5 * (1 - cos(t*(i+1)));
                q(i) = 0.5 * (1 - cos(t*(0.5*i+1)));
                // q(i) = 0.1*t;
            }
            return Function("eval_path", {t}, {ee_pos(MXVector{q})[0]});
        }

        virtual ImplicitTestProblem PrepareImplicit(){
            return ImplicitTestProblem(K_, nx_, nu_, 
                    x_init_, u_init_, 
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk_, eval_objK_, eval_gk_, eval_g0_, eval_gK_, eval_gk_ineq_, eval_gK_ineq_,
                    impl_dyn_);
        }

        virtual ExplicitTestProblem PrepareRootFinder(){
            Function eval_dynamics_equation_implicit = Function("eval_dynamics_equation", {xkp_, uk_, xk_}, {impl_dyn_(MXVector{uk_, xk_, xkp_})[0]});
            Function rf = rootfinder("rf", "newton", eval_dynamics_equation_implicit);
            Function explicit_rootfinder = Function("explicit_rootfinder", {uk_, xk_}, {rf(MXVector{xk_, uk_, xk_})});

            return ExplicitTestProblem(K_, nx_, nu_, 
                    x_init_, u_init_, 
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk_, eval_objK_, eval_gk_, eval_g0_, eval_gK_, eval_gk_ineq_, eval_gK_ineq_,
                    explicit_rootfinder);
        }

        virtual ExplicitTestProblem PrepareExplicit(){
            return ExplicitTestProblem(
                    K_, nx_, nu_, 
                    x_init_, u_init_,
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk_, eval_objK_, eval_gk_, eval_g0_, eval_gK_,
                    eval_gk_ineq_, eval_gK_ineq_, expl_dyn_);
        }

        virtual ExplicitTestProblem PrepareReformulated(){
            MX zk = MX::sym("zk", nx_);
            MX uk_aug = vertcat(uk_, zk);
            MXVector ukxk = {uk_, xk_};

            Function eval_objk = Function("eval_objk", {uk_aug, xk_}, {eval_objk_(ukxk)[0]});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_aug, xk_}, {eval_gk_ineq_(ukxk)[0]});

            MXVector in = {uk_, xk_, zk};
            MXVector out = impl_dyn_(in);
            Function eval_g0 = Function("eval_g0", {uk_aug, xk_}, {vertcat(eval_g0_(ukxk)[0], out[0])});
            Function eval_gk = Function("eval_gk", {uk_aug, xk_}, {vertcat(eval_gk_(ukxk)[0], out[0])});

            Function eval_dynamics_equation_reformulated = Function("eval_dynamics_equation", {uk_aug, xk_}, {zk});

            std::vector<std::vector<double>> u_init(K_, std::vector<double>(nu_ + nx_, 0.0));
            for (int k = 0; k < K_-1; k++){
                for (int i = 0; i < nu_; i++){
                    u_init[k][i] = u_init_[k][i];
                }
                for (int i = 0; i < nx_; i++){
                    u_init[k][nu_ + i] = x_init_[k][i];
                }
            }

            return ExplicitTestProblem(
                    K_, nx_, nu_+nx_, 
                    x_init_, u_init,
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk, eval_objK_, eval_gk, eval_g0, eval_gK_,
                    eval_gk_ineq, eval_gK_ineq_,
                    eval_dynamics_equation_reformulated);
        }

        virtual json GetJsonData(){
            json j;
            j["problem_name"] = "path_follower";
            j["K"] = K_;
            j["nx"] = nx_;
            j["nu"] = nu_;
            j["dt"] = dt_;
            j["start"] = start_;
            return j;
        }

        virtual std::string GetInterfaceName(){ return "path_follower";};
        virtual std::string GetFileNameAppendix(){ return "";};
        void PrintDimensions(){
            std::cout << "nx: " << nx_ << std::endl;
            std::cout << "nu: " << nu_ << std::endl;
            std::cout << "ng: " << eval_gk_.sparsity_out(0).size1() << std::endl;
        }

        // std::unique_ptr<PinocchioCasadi> pc_;
    private:
        int K_ = 100 + 1;
        double T_ = 3.0;
        double dt_ = T_ / (K_ - 1);
        double path_tol_ = 0.05;

        bool with_slack = true;
        bool with_accel_variables = true;

        int n = 6;
        int nx_ = 2*n + 1;
        int nu_ = n + n*with_accel_variables + 1 + 3*with_slack;

        MX q_ = MX::sym("q", n);
        MX qd_ = MX::sym("qd", n);
        MX qdd_ = MX::sym("qdd", n);
        MX tau_ = MX::sym("tau", n*with_accel_variables);
        MX p_ = MX::sym("p", 1);
        MX dp_ = MX::sym("dp", 1);
        MX s_ = MX::sym("s", 3*with_slack);

        MX q_p_ = MX::sym("q_p", n);
        MX qd_p_ = MX::sym("qd_p", n);
        MX p_p_ = MX::sym("p_p", 1);
        
        MX xk_ = vertcat(q_, qd_, p_);
        MX xkp_ = vertcat(q_p_, qd_p_, p_p_);
        MX uk_ = vertcat(vertcat(qdd_, tau_), vertcat(dp_, s_));

        Function fk_ = Function::load("ur10_fk.casadi");
        Function fd_ = Function::load("ur10_fd.casadi");
        Function id_ = Function::load("ur10_id.casadi");

        Function ee_pos = Function("ee_pos", {q_}, {fk_(q_)[n](Slice(0,3), 3)});
        Function eval_path_;
        std::vector<double> tau_eq_ = {0, -120.8, -34, 0, 0, 0};
        std::vector<double> tau_lim_ = {330, 330, 150, 80, 80, 80};
        
        std::vector<double> start_ = std::vector<double>(nx_, 0.0);

        std::vector<double> lb_ = std::vector<double>(n + 1 + 1);
        std::vector<double> ub_ = std::vector<double>(n + 1 + 1);
        std::vector<double> lb_K_ = {};
        std::vector<double> ub_K_ = {};

        std::vector<std::vector<double>> x_init_;
        std::vector<std::vector<double>> u_init_;

        Function eval_objk_;
        Function eval_objK_;
        Function eval_g0_;
        Function eval_gk_;
        Function eval_gK_;
        Function eval_gk_ineq_;
        Function eval_gK_ineq_;

        Function expl_dyn_;
        Function impl_dyn_;
};
#endif

#endif
