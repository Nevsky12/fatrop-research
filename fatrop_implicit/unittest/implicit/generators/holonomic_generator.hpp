#ifndef __HOLONOMIC_GENERATOR_HPP__
#define __HOLONOMIC_GENERATOR_HPP__

#include "fatrop/context/context.hpp"
#include "fatrop/context/generic.hpp"
#include "fatrop/ocp/ocp_abstract.hpp"
#include "../test_problem/implicit_test_problem.hpp"    
#include "../test_problem/explicit_test_problem.hpp"
#include "ocp_interface_generator.hpp"

#include <casadi/casadi.hpp>

using namespace casadi;
using namespace fatrop;

class HolonomicInterfaceGenerator : public InterfaceGenerator {
    public:
        // Constructor
        // velocity control: level 1
        // acceleration control: level 2
        // ...
        HolonomicInterfaceGenerator(int n, int control_level){
            K_ = 100;
            dt_ = 0.05;
            uk_min_ = -10;
            uk_max_ = 10;

            n_ = n;
            control_level_ = control_level;

            nx_ = control_level_*n_;
            nu_ = n_;

            start_ = std::vector<double>(nx_, 0.0);
            end_ = std::vector<double>(nx_, 0.0);
            for (int i = 0; i < n_; i++){
                end_[i] = 1.0 + 0.2*i;
            }

            xk_ = MX::sym("xk", nx_);
            uk_ = MX::sym("uk", nu_);
            xkp_ = MX::sym("xkp", nx_);
        };

        virtual ImplicitTestProblem PrepareImplicit(){
            Function eval_objk = Function("eval_objk", {uk_, xk_}, {sumsqr(uk_)});
            Function eval_objK = Function("eval_objK", {xk_}, {0});
            Function eval_gk = Function("eval_gk", {uk_, xk_}, {MX::zeros(0,1)});
            Function eval_g0 = Function("eval_g0", {uk_, xk_}, {xk_ - start_});
            Function eval_gK = Function("eval_gK", {xk_}, {xk_ - end_});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_, xk_}, {uk_});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {MX::zeros(0,1)});
            
            MX temp_implicit = MX(nx_, 1);
            for (int i = 0; i < control_level_; ++i) {
                for (int j = 0; j < n_; ++j) {
                    MX der_implicit = (i < control_level_ - 1) ? xkp_((i+1)*n_ + j) : uk_(j);
                    temp_implicit(i*n_ + j) = xk_(i*n_ + j) + dt_*der_implicit - xkp_(i*n_ + j);
                }
            }
            Function eval_dynamics_equation_implicit = Function("eval_dynamics_equation", {uk_, xk_, xkp_}, {temp_implicit});

            return ImplicitTestProblem(K_, nx_, nu_, 
                    std::vector<std::vector<double>>(100, std::vector<double>(nx_, 0.0)), 
                    std::vector<std::vector<double>>(100, std::vector<double>(nu_, 0.0)), 
                    std::vector<double>(nu_, uk_min_), 
                    std::vector<double>(nu_, uk_max_), 
                    std::vector<double>(0,0), 
                    std::vector<double>(0,0),
                    eval_objk, eval_objK, eval_gk, eval_g0, eval_gK, eval_gk_ineq, eval_gK_ineq,
                    eval_dynamics_equation_implicit);
        }

        virtual ExplicitTestProblem PrepareRootFinder(){
            Function eval_objk = Function("eval_objk", {uk_, xk_}, {sumsqr(uk_)});
            Function eval_objK = Function("eval_objK", {xk_}, {0});
            Function eval_gk = Function("eval_gk", {uk_, xk_}, {MX::zeros(0,1)});
            Function eval_g0 = Function("eval_g0", {uk_, xk_}, {xk_ - start_});
            Function eval_gK = Function("eval_gK", {xk_}, {xk_ - end_});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_, xk_}, {uk_});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {MX::zeros(0,1)});
            
            MX temp_implicit = MX(nx_, 1);
            for (int i = 0; i < control_level_; ++i) {
                for (int j = 0; j < n_; ++j) {
                    MX der_implicit = (i < control_level_ - 1) ? xkp_((i+1)*n_ + j) : uk_(j);
                    temp_implicit(i*n_ + j) = xk_(i*n_ + j) + dt_*der_implicit - xkp_(i*n_ + j);
                }
            }
            Function eval_dynamics_equation_implicit = Function("eval_dynamics_equation", {xkp_, uk_, xk_}, {temp_implicit});
            Function rf = rootfinder("rf", "newton", eval_dynamics_equation_implicit);
            Function explicit_rootfinder = Function("explicit_rootfinder", {uk_, xk_}, {rf(MXVector{xk_, uk_, xk_})});

            return ExplicitTestProblem(K_, nx_, nu_, 
                    std::vector<std::vector<double>>(100, std::vector<double>(nx_, 0.0)), 
                    std::vector<std::vector<double>>(100, std::vector<double>(nu_, 0.0)), 
                    std::vector<double>(nu_, uk_min_), 
                    std::vector<double>(nu_, uk_max_), 
                    std::vector<double>(0,0), 
                    std::vector<double>(0,0),
                    eval_objk, eval_objK, eval_gk, eval_g0, eval_gK, eval_gk_ineq, eval_gK_ineq,
                    explicit_rootfinder);
        }

        virtual ExplicitTestProblem PrepareExplicit(){
            Function eval_objk = Function("eval_objk", {uk_, xk_}, {sumsqr(uk_)});
            Function eval_objK = Function("eval_objK", {xk_}, {0});
            Function eval_gk = Function("eval_gk", {uk_, xk_}, {MX::zeros(0,1)});
            Function eval_g0 = Function("eval_g0", {uk_, xk_}, {xk_ - start_});
            Function eval_gK = Function("eval_gK", {xk_}, {xk_ - end_});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_, xk_}, {uk_});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {MX::zeros(0,1)});
            
            MX temp_explicit = MX(nx_, 1);
            for (int i = 0; i < control_level_; ++i) {
                for (int j = 0; j < n_; ++j) {
                    MX der_explicit = (i < control_level_ - 1) ? xk_((i+1)*n_ + j) : uk_(j);
                    temp_explicit(i*n_ + j) = xk_(i*n_ + j) + dt_*der_explicit;
                }
            }
            Function eval_dynamics_equation_explicit = Function("eval_dynamics_equation", {uk_, xk_}, {temp_explicit});

            return ExplicitTestProblem(
                    K_, nx_, nu_, 
                    std::vector<std::vector<double>>(100, std::vector<double>(nx_, 0.0)), 
                    std::vector<std::vector<double>>(100, std::vector<double>(nu_, 0.0)), 
                    std::vector<double>(nu_, uk_min_),
                    std::vector<double>(nu_, uk_max_), 
                    std::vector<double>(0, 0),
                    std::vector<double>(0, 0),
                    eval_objk, eval_objK, eval_gk, eval_g0, eval_gK, eval_gk_ineq, eval_gK_ineq,
                    eval_dynamics_equation_explicit);
        }

        virtual ExplicitTestProblem PrepareReformulated(){
            MX zk = MX::sym("zk", nx_);
            MX uk_aug = vertcat(uk_, zk);

            Function eval_objk = Function("eval_objk", {uk_aug, xk_}, {sumsqr(uk_)});
            Function eval_objK = Function("eval_objK", {xk_}, {0});
            Function eval_gK = Function("eval_gK", {xk_}, {xk_ - end_});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_aug, xk_}, {uk_});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {MX::zeros(0,1)});
            
            MX temp_reformulated = MX(nx_, 1);
            for (int i = 0; i < control_level_; ++i) {
                for (int j = 0; j < n_; ++j) {
                    MX der_reformulated = (i < control_level_ - 1) ? zk((i+1)*n_ + j) : uk_(j);
                    temp_reformulated(i*n_ + j) = xk_(i*n_ + j) + dt_*der_reformulated - zk(i*n_ + j);
                }
            }
            Function eval_g0 = Function("eval_g0", {uk_aug, xk_}, {vertcat(xk_ - start_, temp_reformulated)});
            Function eval_gk = Function("eval_gk", {uk_aug, xk_}, {temp_reformulated});
            Function eval_dynamics_equation_reformulated = Function("eval_dynamics_equation", {uk_aug, xk_}, {zk});

            std::vector<std::vector<double>> u_init(K_-1, std::vector<double>(nu_ + nx_, 0.0));
            for (int k = 0; k < K_-1; k++){
                for (int i = 0; i < nu_; i++){
                    u_init[k][i] = 0;
                }
                for (int i = 0; i < nx_; i++){
                    u_init[k][nu_ + i] = 0;
                }
            }

            return ExplicitTestProblem(
                    K_, nx_, nu_+nx_, 
                    std::vector<std::vector<double>>(100, std::vector<double>(nx_, 0.0)), 
                    std::vector<std::vector<double>>(100, std::vector<double>(nu_ + nx_, 0.0)), 
                    std::vector<double>(nu_, uk_min_), 
                    std::vector<double>(nu_, uk_max_), 
                    std::vector<double>(0,0),
                    std::vector<double>(0,0),
                    eval_objk, eval_objK, eval_gk, eval_g0, eval_gK, eval_gk_ineq, eval_gK_ineq,
                    eval_dynamics_equation_reformulated);
        }

        virtual json GetJsonData(){
            json j;
            j["problem_name"] = "holonomic";
            j["K"] = K_;
            j["n"] = n_;
            j["control_level"] = control_level_;
            j["nx"] = nx_;
            j["nu"] = nu_;
            j["dt"] = dt_;
            j["uk_min"] = uk_min_;
            j["uk_max"] = uk_max_;
            j["start"] = start_;
            j["end"] = end_;
            return j;
        }

        virtual std::string GetInterfaceName(){ return "holonomic";};
        virtual std::string GetFileNameAppendix(){ return
            "nb_dimensions_" + std::to_string(n_) + 
            "_control_level_" + std::to_string(control_level_);
        };

    private:
        int K_;
        int n_;
        int control_level_;
        int nx_;
        int nu_;
        double dt_;
        double uk_min_;
        double uk_max_;

        MX xk_;
        MX uk_;
        MX xkp_;

        std::vector<double> start_;
        std::vector<double> end_;
};

#endif