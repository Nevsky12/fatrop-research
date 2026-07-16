#ifndef __N_LINK_GENERATOR_HPP__
#define __N_LINK_GENERATOR_HPP__

#include "fatrop/context/context.hpp"
#include "fatrop/context/generic.hpp"
#include "fatrop/ocp/ocp_abstract.hpp"
#include "../test_problem/implicit_test_problem.hpp"    
#include "../test_problem/explicit_test_problem.hpp"
#include "ocp_interface_generator.hpp"

#include <casadi/casadi.hpp>

using namespace casadi;
using namespace fatrop;

// see: "On 𝑛-Link Planar Revolute Robot: Motion Equations and New Properties"
class PlanarRobot : public InterfaceGenerator {
    public:
        // Constructor
        // n: number of links
        PlanarRobot(int n){
            // define parameters
            K_ = 500;
            dt_ = 0.01;

            n_ = n;

            nx_ = 2*n_;
            nu_ = n_;

            m_ = 1.0/n_;
            l_ = 1.0/n_;
            lc_ = l_/2;
            J = pow(1.0*3.0*(m_*l_), 3);

            // define start and endpoints
            start_ = std::vector<double>(nx_, 0.0);
            end_ = std::vector<double>(n_, 0.0);
            for (int i = 0; i < n_; i++){
                // start_[i] = (i > 0) ? 0.1 : 0.9*3.14;
                start_[i] = 3.14;
                end_[i] = 0;
            }

            // define variables
            th_ = MX::sym("theta", n_);     
            thd_ = MX::sym("theta_dot", n_);
            uk_ = MX::sym("uk", n);
            xk_ = vertcat(th_, thd_);
            thp_ = MX::sym("theta_plus", n_);
            thdp_ = MX::sym("theta_dot_plus", n_);
            xkp_ = vertcat(thp_, thdp_);

            // define objective
            // eval_objk_ = Function("eval_objk", {uk_, xk_}, {10*sumsqr(uk_)});
            eval_objk_ = Function("eval_objk", {uk_, xk_}, 
                {sumsqr(uk_) + 0.1*sumsqr(thd_(Slice(0,n_-1)) - thd_(Slice(1,n_))) + 0.1*sumsqr(th_(Slice(0,n_-1)) - th_(Slice(1,n_)))});
            eval_objK_ = Function("eval_objK", {xk_}, {0});
            eval_g0_ = Function("eval_g0", {uk_, xk_}, {xk_ - start_});
            eval_gk_ = Function("eval_gk", {uk_, xk_}, {MX::zeros(0,1)});
            eval_gk_ineq_ = Function("eval_gk_ineq", {uk_, xk_}, {vertcat(uk_, th_)});
            eval_gK_ = Function("eval_gK", {xk_}, {thd_});
            eval_gK_ineq_ = Function("eval_gK_ineq", {xk_}, {sumsqr(th_ - end_)});
            
            // define dynamics basics
            Rinv_ = DM::zeros(n_, n_);
            for (int i = 0; i < n_; i++){
                for (int j = 0; j < n_; j++){
                    if (j < i){
                        Rinv_(i,j) = -1;
                    } else if (j == i){
                        Rinv_(i,j) = 1;
                    } else {
                        Rinv_(i,j) = 0;
                    }
                }
            }
            set_M(); set_C(); set_G();

            // set bounds
            lb_ = std::vector<double>(nu_+n_, uk_min_);
            ub_ = std::vector<double>(nu_+n_, uk_max_);
            for (int i = 0; i < n_; i++){
                lb_[i] = -50;
                ub_[i] = 50;
            }
            lb_K_ = {0};
            ub_K_ = {1.0e-3};

            // set initialization
            x_init_ = std::vector<std::vector<double>>(K_+1, std::vector<double>(nx_, 0.0));
            u_init_ = std::vector<std::vector<double>>(K_, std::vector<double>(nu_, 0.0));
            for (int k = 0; k < K_+1; k++){
                for (int i = 0; i < n_; i++){
                    x_init_[k][i] = start_[i] + (end_[i] - start_[i]) * k / K_;
                }
            }
        };

        virtual ImplicitTestProblem PrepareImplicit(){           
            MX rhs = vertcat(thdp_, mtimes(inv(eval_M_(thp_)[0]), mtimes(Rinv_, uk_) - mtimes(eval_C_(xkp_)[0], thdp_) - eval_G_(thp_)[0]));
            Function eval_dynamics_equation_implicit = Function("eval_dynamics_equation", {uk_, xk_, xkp_}, {xk_ + dt_*rhs - xkp_});

            return ImplicitTestProblem(K_, nx_, nu_, 
                    x_init_, u_init_, 
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk_, eval_objK_, eval_gk_, eval_g0_, eval_gK_, eval_gk_ineq_, eval_gK_ineq_,
                    eval_dynamics_equation_implicit);
        }

        virtual ExplicitTestProblem PrepareRootFinder(){           
            MX rhs = vertcat(thdp_, mtimes(inv(eval_M_(thp_)[0]), mtimes(Rinv_, uk_) - mtimes(eval_C_(xkp_)[0], thdp_) - eval_G_(thp_)[0]));
            Function eval_dynamics_equation_implicit = Function("eval_dynamics_equation", {xkp_, uk_, xk_}, {xk_ + dt_*rhs - xkp_});
            Function rf = rootfinder("rf", "newton", eval_dynamics_equation_implicit);
            Function explicit_rootfinder = Function("explicit_rootfinder", {uk_, xk_}, {rf(MXVector{xk_, uk_, xk_})});

            return ExplicitTestProblem(K_, nx_, nu_, 
                    x_init_, u_init_, 
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk_, eval_objK_, eval_gk_, eval_g0_, eval_gK_, eval_gk_ineq_, eval_gK_ineq_,
                    explicit_rootfinder);
        }

        virtual ExplicitTestProblem PrepareExplicit(){           
            MX rhs = vertcat(thd_, mtimes(inv(eval_M_(th_)[0]), mtimes(Rinv_, uk_) - mtimes(eval_C_(xk_)[0], thd_) - eval_G_(th_)[0]));
            Function eval_dynamics_equation_explicit = Function("eval_dynamics_equation", {uk_, xk_}, {xk_ + dt_*rhs});

            return ExplicitTestProblem(
                    K_, nx_, nu_, 
                    x_init_, u_init_,
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk_, eval_objK_, eval_gk_, eval_g0_, eval_gK_,
                    eval_gk_ineq_, eval_gK_ineq_, eval_dynamics_equation_explicit);
        }

        virtual ExplicitTestProblem PrepareReformulated(){
            MX zk = MX::sym("zk", nx_);
            MX uk_aug = vertcat(uk_, zk);

            MXVector ukxk = {uk_, xk_};
            Function eval_objk = Function("eval_objk", {uk_aug, xk_}, {eval_objk_(ukxk)[0]});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_aug, xk_}, {eval_gk_ineq_(ukxk)[0]});
            Function eval_gK = Function("eval_gK", {xk_}, {eval_gK_(xk_)[0]});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {eval_gK_ineq_(xk_)[0]});
            
            MX rhs = vertcat(thd_, mtimes(inv(eval_M_(th_)[0]), mtimes(Rinv_, uk_) - mtimes(eval_C_(xk_)[0], thd_) - eval_G_(th_)[0]));

            Function eval_g0 = Function("eval_g0", {uk_aug, xk_}, {vertcat(eval_g0_(ukxk)[0], xk_ + dt_*rhs - zk)});
            Function eval_gk = Function("eval_gk", {uk_aug, xk_}, {vertcat(eval_gk_(ukxk)[0], xk_ + dt_*rhs - zk)});
            Function eval_dynamics_equation_reformulated = Function("eval_dynamics_equation", {uk_aug, xk_}, {zk});

            std::vector<std::vector<double>> u_init(K_-1, std::vector<double>(nu_ + nx_, 0.0));
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
                    eval_objk, eval_objK_, eval_gk, eval_g0, eval_gK,
                    eval_gk_ineq, eval_gK_ineq,
                    eval_dynamics_equation_reformulated);
        }

        void SolveOptiInstance(){
            MX rhs = vertcat(thd_, mtimes(inv(eval_M_(th_)[0]), mtimes(Rinv_, uk_) - mtimes(eval_C_(xk_)[0], thd_) - eval_G_(th_)[0]));
            Function eval_dynamics_equation_explicit = Function("eval_dynamics_equation", {uk_, xk_}, {xk_ + dt_*rhs});

            Opti opti = Opti();
            int N = K_-1;

            std::vector<MX> qq_list = {};
            std::vector<MX> vv_list = {};
            std::vector<MX> uu_list = {};
            for (int k = 0; k < N + 1; k++){
                qq_list.push_back(opti.variable(n_));
                vv_list.push_back(opti.variable(n_));
                if (k < N){
                    uu_list.push_back(opti.variable(n_));
                }
            }
            MX qq = vertcat(qq_list);
            MX vv = vertcat(vv_list);
            MX uu = vertcat(uu_list);

            if (eval_g0_.sparsity_out(0).size1() > 0){
                opti.subject_to(eval_g0_(MXVector{uu_list[0], vertcat(qq_list[0], vv_list[0])})[0] == DM::zeros(eval_g0_.sparsity_out(0).size1()));
            }
            
            for (int k = 0; k < N; k++){
                MX qk = qq_list[k];
                MX vk = vv_list[k];
                MX uk = uu_list[k];

                MX x = vertcat(qk, vk);
                MX x_next = eval_dynamics_equation_explicit(MXVector{uk, x})[0];
                MX q_next = x_next(Slice(0, n_));
                MX v_next = x_next(Slice(n_, nx_));

                opti.subject_to(qq_list[k+1] == q_next);
                opti.subject_to(vv_list[k+1] == v_next);

                if (eval_gk_ineq_.sparsity_out(0).size1() > 0){
                    opti.subject_to(
                        lb_ <= (eval_gk_ineq_(MXVector{uk, x})[0] <= ub_));
                }
            }

            MX obj = 0;
            for (int k = 0; k < N; k++){
                MX qk = qq_list[k];
                MX vk = vv_list[k];
                MX uk = uu_list[k];
                MX x = vertcat(qk, vk);
                obj += eval_objk_(MXVector{uk, x})[0];
            }
            MX qk = qq_list[N];
            MX vk = vv_list[N];
            MX x = vertcat(qk, vk);
            obj += eval_objK_(x)[0];
            opti.minimize(obj);

            if (eval_gK_ineq_.sparsity_out(0).size1() > 0){
                opti.subject_to(lb_K_ <= (eval_gK_ineq_(vertcat(qq_list[N], vv_list[N]))[0] <= ub_K_));
            }
            if (eval_gK_.sparsity_out(0).size1() > 0){
                opti.subject_to(eval_gK_(vertcat(qq_list[N], vv_list[N]))[0] == DM::zeros(eval_gK_.sparsity_out(0).size1()));
            }

            for (int k = 0; k < N+1; k++){
                for (int i = 0; i < n_; i++){
                    opti.set_initial(qq_list[k](i), x_init_[0][i]);
                }
                if (k < N){
                    for (int i = 0; i < nu_; i++){
                        opti.set_initial(uu_list[k](i), u_init_[0][i]);
                    }
                }
            }

            // define options
            Dict casadi_opts;
            Dict solver_opts;
            casadi_opts["structure_detection"] = "auto";
            casadi_opts["fatrop.print_level"] = 12;
            casadi_opts["fatrop.mu_init"] = 0.1;
            solver_opts["max_iter"] = 1000;

            opti.solver("fatrop", casadi_opts, solver_opts);

            opti.solve();
        }

        virtual json GetJsonData(){
            json j;
            j["problem_name"] = "planar_robot";
            j["K"] = K_;
            j["nx"] = nx_;
            j["nu"] = nu_;
            j["dt"] = dt_;
            j["n"] = n_;
            j["m"] = m_;
            j["l"] = l_;
            j["lc"] = lc_;
            j["J"] = J;
            j["uk_min"] = uk_min_;
            j["uk_max"] = uk_max_;
            j["start"] = start_;
            j["end"] = end_;
            return j;
        }

        virtual std::string GetInterfaceName(){ return "planar_robot";};
        virtual std::string GetFileNameAppendix(){return "nb_links_" + std::to_string(n_);};

    private:
        double get_alpha(int i, int j){
            if (i == j){
                double val = J + m_*lc_*lc_;
                for (int k = j+1; k <= n_; k++){
                    val += m_*l_*l_;
                }
                return val;
            } else {
                double val = m_*lc_*l_;
                for (int k = std::max(i,j)+1; k <= n_; k++){
                    val += m_*l_*l_;
                }
                return val;
            }
        }

        double get_beta(int i){
            if (i == n_){
                return m_*lc_*g;
            } else {
                double val = m_*lc_*g;
                for (int k = i+1; k <= n_; k++){
                    val += m_*l_*g;
                }
                return val;
            }
        }

        void set_M(){
            MX M = MX::zeros(n_, n_);
            for (int i = 0; i < n_; i++){
                for (int j = 0; j < n_; j++){
                    M(i,j) = get_alpha(i,j)*cos(th_(j)-th_(i));
                }
            }
            eval_M_ = Function("eval_M", {th_}, {M});
        }

        void set_C(){
            MX C = MX::zeros(n_, n_);
            for (int i = 0; i < n_; i++){
                for (int j = 0; j < n_; j++){
                    C(i,j) = -get_alpha(i, j)*sin(th_(j)-th_(i))*thd_(j);
                }
            }
            eval_C_ = Function("eval_C", {xk_}, {C});
        }

        void set_G(){
            MX G = MX::zeros(n_, 1);
            for (int i = 0; i < n_; i++){
                G(i) = -get_beta(i)*sin(th_(i));
            }
            eval_G_ = Function("eval_G", {th_}, {G});
        }

        int K_ = 100;
        double dt_ = 0.05;

        double m_;
        double l_;
        double lc_;
        double J;
        double g = 3;

        double uk_min_ = -10;
        double uk_max_ = 10;

        MX th_;
        MX thd_;
        MX uk_;
        MX xk_;
        MX thp_;
        MX thdp_;
        MX xkp_;
        DM Rinv_;

        Function eval_M_;
        Function eval_C_;
        Function eval_G_;

        Function eval_objk_;
        Function eval_objK_;
        Function eval_g0_;
        Function eval_gk_;
        Function eval_gk_ineq_;
        Function eval_gK_;
        Function eval_gK_ineq_;

        int n_;
        int nx_;
        int nu_;

        std::vector<double> start_;
        std::vector<double> end_;

        std::vector<double> lb_;
        std::vector<double> ub_;
        std::vector<double> lb_K_;
        std::vector<double> ub_K_;

        std::vector<std::vector<double>> x_init_;
        std::vector<std::vector<double>> u_init_;
};

#endif