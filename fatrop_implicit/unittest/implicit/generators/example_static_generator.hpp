#ifndef __EXAMPLE_STATIC_GENERATOR_HPP__
#define __EXAMPLE_STATIC_GENERATOR_HPP__

#include "fatrop/context/context.hpp"
#include "fatrop/context/generic.hpp"
#include "fatrop/ocp/ocp_abstract.hpp"
#include "../test_problem/implicit_test_problem.hpp"    
#include "../test_problem/explicit_test_problem.hpp"
#include "ocp_interface_generator.hpp"

#include <casadi/casadi.hpp>

using namespace casadi;
using namespace fatrop;

class ExampleStaticGenerator : public InterfaceGenerator {
    public:
        // Constructor
        // min    fx^2 + fy^2
        // s.t. [x(0), y(0), vx(0), vy(0)]^T = 0
        //      [x(K), y(K), vx(K), vy(K)]^T = [1, 1, 0, 0]^T
        //      xk+1 = xk + dt*vxk
        //      yk+1 = yk + dt*vyk
        //      vxk+1 = vxk + dt*fx/m +0.5*dt*fy**2/m
        //      vyk+1 = vyk + dt*fy/m
        ExampleStaticGenerator(){
            // define parameters
            K_ = 100;
            dt_ = 0.05;
            m_ = 1.0;

            nx_ = 4;
            nu_ = 2;

            // define start and endpoints
            start_ = {0, 0, 0, 0};
            end_ = {1, 2, 3, 4};

            // define variables
            xk_ = MX::sym("xk", nx_);
            uk_ = MX::sym("uk", nu_);
            xkp_ = MX::sym("xkp", nx_);

            // set bounds
            lb_ = {-50.0, -100.0};
            ub_ = {50.0, 100.0};
            lb_K_ = {};
            ub_K_ = {};

            // set initialization
            x_init_ = std::vector<std::vector<double>>(K_, std::vector<double>(nx_, 0.0));
            u_init_ = std::vector<std::vector<double>>(K_-1, std::vector<double>(nu_, 0.0));
        };

        virtual ImplicitTestProblem PrepareImplicit(){
            Function eval_objk = Function("eval_objk", {uk_, xk_}, {sumsqr(uk_)});
            Function eval_objK = Function("eval_objK", {xk_}, {0});
            Function eval_gk = Function("eval_gk", {uk_, xk_}, {zero_});
            Function eval_g0 = Function("eval_g0", {uk_, xk_}, {xk_ - start_});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_, xk_}, {uk_});
            Function eval_gK = Function("eval_gK", {xk_}, {xk_ - end_});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {MX::zeros(0,1)});
            
            MX rhs = vertcat(
                vertcat(xkp_(2), xkp_(3)), vertcat(uk_(0)/m_ + 0.5*uk_(1)*uk_(1)/m_, uk_(1)/m_)
                // vertcat(xk_(2), xk_(3)), vertcat(uk_(0)/m_ + 0.5*uk_(1)*uk_(1)/m_, uk_(1)/m_)
            );
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
            Function eval_gk = Function("eval_gk", {uk_, xk_}, {zero_});
            Function eval_g0 = Function("eval_g0", {uk_, xk_}, {xk_ - start_});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_, xk_}, {uk_});
            Function eval_gK = Function("eval_gK", {xk_}, {xk_ - end_});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {MX::zeros(0,1)});
            
            MX rhs = vertcat(
                vertcat(xkp_(2), xkp_(3)), vertcat(uk_(0)/m_ + 0.5*uk_(1)*uk_(1)/m_, uk_(1)/m_)
                // vertcat(xk_(2), xk_(3)), vertcat(uk_(0)/m_ + 0.5*uk_(1)*uk_(1)/m_, uk_(1)/m_)
            );
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
            Function eval_gk = Function("eval_gk", {uk_, xk_}, {zero_});
            Function eval_g0 = Function("eval_g0", {uk_, xk_}, {xk_ - start_});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_, xk_}, {uk_});
            Function eval_gK = Function("eval_gK", {xk_}, {xk_ - end_});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {MX::zeros(0,1)});
            
            MX rhs = vertcat(
                vertcat(xk_(2), xk_(3)), vertcat(uk_(0)/m_ + 0.5*uk_(1)*uk_(1)/m_, uk_(1)/m_)
            );
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
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_aug, xk_}, {uk_});
            Function eval_gK = Function("eval_gK", {xk_}, {xk_ - end_});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {MX::zeros(0,1)});
            
            MX rhs = vertcat(
                vertcat(xk_(2), xk_(3)), vertcat(uk_(0)/m_ + 0.5*uk_(1)*uk_(1)/m_, uk_(1)/m_)
            );
            Function eval_g0 = Function("eval_g0", {uk_aug, xk_}, {vertcat(xk_ - start_, xk_ + dt_*rhs - zk)});
            Function eval_gk = Function("eval_gk", {uk_aug, xk_}, {xk_ + dt_*rhs - zk});
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

        virtual std::string GetInterfaceName(){ return "example_static";};
        virtual std::string GetFileNameAppendix(){return "";};

    private:
        int K_;
        int nx_;
        int nu_;
        double dt_;
        double m_;
        MX xk_;
        MX uk_;
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