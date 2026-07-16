#ifndef __TRUCK_TRAILER_GENERATOR_HPP__
#define __TRUCK_TRAILER_GENERATOR_HPP__

#include "fatrop/context/context.hpp"
#include "fatrop/context/generic.hpp"
#include "fatrop/ocp/ocp_abstract.hpp"
#include "../test_problem/implicit_test_problem.hpp"    
#include "../test_problem/explicit_test_problem.hpp"
#include "ocp_interface_generator.hpp"

#include <casadi/casadi.hpp>

using namespace casadi;
using namespace fatrop;

class TruckTrailerGenerator : public InterfaceGenerator {
    public:
        // Constructor
        // n: number of trailers (n=1 means one truck and one trailer)
        TruckTrailerGenerator(int n){
            // define parameters
            K_ = 20;
            dt_ = 0.1;

            n_ = n;

            nx_ = 2 + n_ + 1;
            nu_ = 2;

            L_ = 1;//0.5;
            M_ = 0;//0.05;

            uk_min_ = -10;
            uk_max_ = 10;

            // define start and endpoints
            start_ = std::vector<double>(nx_, 0.0);
            end_ = std::vector<double>(2, 0.0);
            end_[0] = 5.0; end_[1] = 1.0;

            // define variables
            th_ = MX::sym("theta", n_+1);           // th0, th1, ..., thn
            xk_ = vertcat(th_, MX::sym("x", 2));    
            uk_ = MX::sym("uk", nu_);               // vk, wk
            thp_ = MX::sym("th_plus", n_+1);
            xkp_ = vertcat(thp_, MX::sym("x_plus", 2));

            // set bounds
            lb_ = std::vector<double>(nu_+n_+1, uk_min_);
            ub_ = std::vector<double>(nu_+n_+1, uk_max_);
            for (int i = 0; i < n_ + 1; i++){
                lb_[i] = -3.14;
                ub_[i] = 3.14;
            }
            lb_K_ = {};
            ub_K_ = {};

            // set initialization
            x_init_ = std::vector<std::vector<double>>(K_+1, std::vector<double>(nx_, 0.0));
            u_init_ = std::vector<std::vector<double>>(K_, std::vector<double>(nu_, 0.0));
            for (int k = 0; k < K_+1; k++){
                x_init_[k][nx_-2] = start_[0] + (end_[0]-start_[0])/(K_)*k;
                x_init_[k][nx_-1] = start_[1] + (end_[1]-start_[1])/(K_)*k;
            }
        };

        virtual ImplicitTestProblem PrepareImplicit(){
            Function eval_objk = Function("eval_objk", {uk_, xk_}, {sumsqr(uk_)});
            Function eval_objK = Function("eval_objK", {xk_}, {0});
            Function eval_gk = Function("eval_gk", {uk_, xk_}, {MX::zeros(0,1)});
            Function eval_g0 = Function("eval_g0", {uk_, xk_}, {xk_ - start_});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_, xk_}, {vertcat(th_, uk_)});
            Function eval_gK = Function("eval_gK", {xk_}, {xk_(Slice(nx_-2, nx_, 1)) - end_});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {MX::zeros(0,1)});
            
            MX v = MX(n_+1, 1); v(0) = uk_(0);
            MX th_der = MX(n_+1, 1); th_der(0) = uk_(1);
            for (int i = 0; i < n_; i++){
                v(i+1) = v(i)*cos(thp_(i) - thp_(i+1)) + M_*th_der(i)*sin(thp_(i) - thp_(i+1));
                th_der(i+1) = v(i)*sin(thp_(i) - thp_(i+1))/L_ - M_*cos(thp_(i) - thp_(i+1))*th_der(i)/L_;
            }
            MX rhs = vertcat(th_der, v(n_)*cos(thp_(n_)), v(n_)*sin(thp_(n_)));
            Function eval_dynamics_equation_implicit = Function("eval_dynamics_equation", {uk_, xk_, xkp_}, {xk_ + dt_*rhs - xkp_});

            return ImplicitTestProblem(K_, nx_, nu_, 
                    x_init_, u_init_, 
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk, eval_objK, eval_gk, eval_g0, eval_gK, eval_gk_ineq, eval_gK_ineq,
                    eval_dynamics_equation_implicit);
        }

        ExplicitTestProblem PrepareRootFinder(){
            Function eval_objk = Function("eval_objk", {uk_, xk_}, {sumsqr(uk_)});
            Function eval_objK = Function("eval_objK", {xk_}, {0});
            Function eval_gk = Function("eval_gk", {uk_, xk_}, {MX::zeros(0,1)});
            Function eval_g0 = Function("eval_g0", {uk_, xk_}, {xk_ - start_});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_, xk_}, {vertcat(th_, uk_)});
            Function eval_gK = Function("eval_gK", {xk_}, {xk_(Slice(nx_-2, nx_, 1)) - end_});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {MX::zeros(0,1)});
            
            MX v = MX(n_+1, 1); v(0) = uk_(0);
            MX th_der = MX(n_+1, 1); th_der(0) = uk_(1);
            for (int i = 0; i < n_; i++){
                v(i+1) = v(i)*cos(thp_(i) - thp_(i+1)) + M_*th_der(i)*sin(thp_(i) - thp_(i+1));
                th_der(i+1) = v(i)*sin(thp_(i) - thp_(i+1))/L_ - M_*cos(thp_(i) - thp_(i+1))*th_der(i)/L_;
            }
            MX rhs = vertcat(th_der, v(n_)*cos(thp_(n_)), v(n_)*sin(thp_(n_)));
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
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_, xk_}, {vertcat(th_, uk_)});
            Function eval_gK = Function("eval_gK", {xk_}, {xk_(Slice(nx_-2, nx_, 1)) - end_});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {MX::zeros(0,1)});
            
            MX v = MX(n_+1, 1); v(0) = uk_(0);
            MX th_der = MX(n_+1, 1); th_der(0) = uk_(1);
            for (int i = 0; i < n_; i++){
                v(i+1) = v(i)*cos(th_(i) - th_(i+1)) + M_*th_der(i)*sin(th_(i) - th_(i+1));
                th_der(i+1) = v(i)*sin(th_(i) - th_(i+1))/L_ - M_*cos(th_(i) - th_(i+1))*th_der(i)/L_;
            }
            MX rhs = vertcat(th_der, v(n_)*cos(th_(n_)), v(n_)*sin(th_(n_)));
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
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_aug, xk_}, {vertcat(th_, uk_)});
            Function eval_gK = Function("eval_gK", {xk_}, {xk_(Slice(nx_-2, nx_, 1)) - end_});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {MX::zeros(0,1)});
            
            MX v = MX(n_+1, 1); v(0) = uk_(0);
            MX th_der = MX(n_+1, 1); th_der(0) = uk_(1);
            for (int i = 0; i < n_; i++){
                v(i+1) = v(i)*cos(zk(i) - zk(i+1)) + M_*th_der(i)*sin(zk(i) - zk(i+1));
                th_der(i+1) = v(i)*sin(zk(i) - zk(i+1))/L_ - M_*cos(zk(i) - zk(i+1))*th_der(i)/L_;
            }
            MX rhs = vertcat(th_der, v(n_)*cos(zk(n_)), v(n_)*sin(zk(n_)));

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
            j["problem_name"] = "truck_trailer";
            j["K"] = K_;
            j["nx"] = nx_;
            j["nu"] = nu_;
            j["dt"] = dt_;
            j["L"] = L_;
            j["M"] = M_;
            j["n"] = n_;
            j["uk_min"] = uk_min_;
            j["uk_max"] = uk_max_;
            j["start"] = start_;
            j["end"] = end_;
            return j;
        }

    virtual std::string GetInterfaceName(){ return "truck_trailer";};
    virtual std::string GetFileNameAppendix(){return "nb_trailers_" + std::to_string(n_);};

    private:
        int K_;
        int n_;
        double L_;
        double M_;
        int nx_;
        int nu_;
        double dt_;
        double uk_min_;
        double uk_max_;

        MX th_;
        MX xk_;
        MX uk_;
        MX thp_;
        MX xkp_;

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