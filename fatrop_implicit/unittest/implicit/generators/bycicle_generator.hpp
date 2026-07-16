#ifndef __BYCICLE_GENERATOR_HPP__
#define __BYCICLE_GENERATOR_HPP__

#include "fatrop/context/context.hpp"
#include "fatrop/context/generic.hpp"
#include "fatrop/ocp/ocp_abstract.hpp"
#include "../test_problem/implicit_test_problem.hpp"    
#include "../test_problem/explicit_test_problem.hpp"
#include "ocp_interface_generator.hpp"

#include <casadi/casadi.hpp>

using namespace casadi;
using namespace fatrop;

class BycicleGenerator : public InterfaceGenerator {
    public:
        // Constructor
        // min    u^2
        // s.t. x(0) = 0
        //      px(K) = 5, py(K) = 1
        //      |v| <= v_max
        //      |w| <= w_max
        //      x_dot = [w, v * cos(theta); v * sin(theta)]^T
        BycicleGenerator(){
            // define parameters
            K_ = 20;
            dt_ = 0.1;

            nx_ = 3;
            nu_ = 2;

            v_max_ = 1.0;
            w_max_ = 0.5;

            // define start and endpoints
            start_ = std::vector<double>(nx_, 0.0);
            end_ = std::vector<double>(2, 0.0);
            end_[0] = 5.0; end_[1] = 1.0;

            // define variables
            pxk_ = MX::sym("x", 1, 1);
            pyk_ = MX::sym("y", 1, 1);
            thk_ = MX::sym("theta", 1, 1);
            xk_ = vertcat(thk_, pxk_, pyk_);    
            vk_ = MX::sym("v", 1, 1);
            wk_ = MX::sym("w", 1, 1);
            uk_ = vertcat(wk_, vk_);

            pxkp_ = MX::sym("xp", 1, 1);
            pykp_ = MX::sym("yp", 1, 1);
            thkp_ = MX::sym("thetap", 1, 1);
            xkp_ = vertcat(thkp_, pxkp_, pykp_);  

            // set bounds
            lb_ = {-w_max_, -v_max_};
            ub_ = {w_max_, v_max_};
            lb_K_ = {};
            ub_K_ = {};

            // set initialization
            x_init_ = std::vector<std::vector<double>>(K_, std::vector<double>(nx_, 10.0));
            u_init_ = std::vector<std::vector<double>>(K_-1, std::vector<double>(nu_, 2.0));
            for (int k = 0; k < K_; k++){
                x_init_[k][1] = start_[0] + (end_[0]-start_[0])/(K_-1)*k;
                x_init_[k][2] = start_[1] + (end_[1]-start_[1])/(K_-1)*k;
            }
        };

        virtual ImplicitTestProblem PrepareImplicit(){
            Function eval_objk = Function("eval_objk", {uk_, xk_}, {sumsqr(uk_)});
            Function eval_objK = Function("eval_objK", {xk_}, {0});
            Function eval_gk = Function("eval_gk", {uk_, xk_}, {MX::zeros(0,1)});
            Function eval_g0 = Function("eval_g0", {uk_, xk_}, {xk_ - start_});
            // Function eval_gk_ineq = Function("eval_gk_ineq", {uk_, xk_}, {uk_});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_, xk_}, {zero_});
            Function eval_gK = Function("eval_gK", {xk_}, {vertcat(pxk_ - end_[0], pyk_ - end_[1])});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {MX::zeros(0,1)});
            
            MX rhs = vertcat(wk_, vk_*cos(thkp_), vk_*sin(thkp_));
            Function eval_dynamics_equation_implicit = Function("eval_dynamics_equation", {uk_, xk_, xkp_}, {xk_ + dt_*rhs - xkp_});

            return ImplicitTestProblem(K_, nx_, nu_, 
                    x_init_, u_init_, 
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk, eval_objK, eval_gk, eval_g0, eval_gK, eval_gk_ineq, eval_gK_ineq,
                    eval_dynamics_equation_implicit);
        }

        virtual ExplicitTestProblem PrepareRootFinder(){
            Function eval_objk = Function("eval_objk", {uk_, xk_}, {sumsqr(uk_)});
            Function eval_objK = Function("eval_objK", {xk_}, {0});
            Function eval_gk = Function("eval_gk", {uk_, xk_}, {MX::zeros(0,1)});
            Function eval_g0 = Function("eval_g0", {uk_, xk_}, {xk_ - start_});
            // Function eval_gk_ineq = Function("eval_gk_ineq", {uk_, xk_}, {uk_});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_, xk_}, {zero_});
            Function eval_gK = Function("eval_gK", {xk_}, {vertcat(pxk_ - end_[0], pyk_ - end_[1])});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {MX::zeros(0,1)});
            
            MX rhs = vertcat(wk_, vk_*cos(thkp_), vk_*sin(thkp_));
            Function eval_dynamics_equation_implicit = Function("eval_dynamics_equation", {xkp_, uk_, xk_}, {xk_ + dt_*rhs - xkp_});
            Function rf = rootfinder("rf", "newton", eval_dynamics_equation_implicit);
            Function explicit_rootfinder = Function("explicit_rootfinder", {uk_, xk_}, {rf(MXVector{xk_, uk_, xk_})});

            return ExplicitTestProblem(K_, nx_, nu_, 
                    x_init_, u_init_, 
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk, eval_objK, eval_gk, eval_g0, eval_gK, eval_gk_ineq, eval_gK_ineq,
                    explicit_rootfinder);
        }

        virtual ExplicitTestProblem PrepareExplicit(){
            Function eval_objk = Function("eval_objk", {uk_, xk_}, {sumsqr(uk_)});
            Function eval_objK = Function("eval_objK", {xk_}, {0});
            Function eval_gk = Function("eval_gk", {uk_, xk_}, {MX::zeros(0,1)});
            Function eval_g0 = Function("eval_g0", {uk_, xk_}, {xk_ - start_});
            // Function eval_gk_ineq = Function("eval_gk_ineq", {uk_, xk_}, {uk_});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_, xk_}, {zero_});
            Function eval_gK = Function("eval_gK", {xk_}, {vertcat(pxk_ - end_[0], pyk_ - end_[1])});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {MX::zeros(0,1)});
            
            MX rhs = vertcat(wk_, vk_*cos(thk_), vk_*sin(thk_));
            Function eval_dynamics_equation_explicit = Function("eval_dynamics_equation", {uk_, xk_}, {xk_ + dt_*rhs});

            return ExplicitTestProblem(
                    K_, nx_, nu_, 
                    x_init_, u_init_,
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk, eval_objK, eval_gk, eval_g0, eval_gK,
                    eval_gk_ineq, eval_gK_ineq, eval_dynamics_equation_explicit);
        }

        virtual ExplicitTestProblem PrepareReformulated(){
            MX zk = MX::sym("zk", nx_);
            MX uk_aug = vertcat(uk_, zk);

            Function eval_objk = Function("eval_objk", {uk_aug, xk_}, {sumsqr(uk_)});
            Function eval_objK = Function("eval_objK", {xk_}, {0});
            // Function eval_gk_ineq = Function("eval_gk_ineq", {uk_aug, xk_}, {uk_});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_aug, xk_}, {zero_});
            Function eval_gK = Function("eval_gK", {xk_}, {vertcat(pxk_ - end_[0], pyk_ - end_[1])});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {MX::zeros(0,1)});
            
            MX rhs = vertcat(wk_, vk_*cos(zk(0)), vk_*sin(zk(0)));
            Function eval_g0 = Function("eval_g0", {uk_aug, xk_}, {vertcat(xk_ - start_, xk_ + dt_*rhs - zk)});
            Function eval_gk = Function("eval_gk", {uk_aug, xk_}, {xk_ + dt_*rhs - zk});
            Function eval_dynamics_equation_reformulated = Function("eval_dynamics_equation", {uk_aug, xk_}, {zk});

            std::vector<std::vector<double>> u_init(K_ - 1, std::vector<double>(nu_ + nx_, 0.0));
            for (int k = 0; k < K_ - 1; k++){
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
                    eval_objk, eval_objK, eval_gk, eval_g0, eval_gK,
                    eval_gk_ineq, eval_gK_ineq,
                    eval_dynamics_equation_reformulated);
        }

        virtual json GetJsonData(){
            json j;
            j["problem_name"] = "bycicle_model";
            j["K"] = K_;
            j["nx"] = nx_;
            j["nu"] = nu_;
            j["dt"] = dt_;
            j["start"] = start_;
            j["end"] = end_;
            return j;
        }

        virtual std::string GetInterfaceName(){ return "bycicle";};
        virtual std::string GetFileNameAppendix(){return "";};

    private:
        int K_;
        int nx_;
        int nu_;
        double dt_;
        double v_max_;
        double w_max_;

        MX pxk_;
        MX pyk_;
        MX thk_;
        MX xk_;
        MX vk_;
        MX wk_;
        MX uk_;
        MX pxkp_;
        MX pykp_;
        MX thkp_;
        MX xkp_;
        MX zero_ = MX::zeros(0, 1); 

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