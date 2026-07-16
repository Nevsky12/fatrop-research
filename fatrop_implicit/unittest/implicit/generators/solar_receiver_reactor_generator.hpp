#ifndef __SOLAR_RECEIVER_REACTOR_GENERATOR_HPP__
#define __SOLAR_RECEIVER_REACTOR_GENERATOR_HPP__

#include "fatrop/context/context.hpp"
#include "fatrop/context/generic.hpp"
#include "fatrop/ocp/ocp_abstract.hpp"
#include "../test_problem/implicit_test_problem.hpp"    
#include "../test_problem/explicit_test_problem.hpp"
#include "ocp_interface_generator.hpp"
#include <json/include/nlohmann/json.hpp>

#include <casadi/casadi.hpp>

using namespace casadi;
using namespace fatrop;
using json = nlohmann::json;

class SolarReceiverReactorGenerator : public InterfaceGenerator {
    public:
        // Constructor
        SolarReceiverReactorGenerator(int n){
            n_ = n;

            // load metadata
            std::ifstream file("../../unittest/implicit/generators/solar_receiver_reactor/meta_data.json");
            json meta_data;
            file >> meta_data;
            T_ = meta_data["T"];
            dt_ = meta_data["dt"];
            K_ = T_ / dt_;
            nx_ = meta_data["nx"];
            nu_ = meta_data["nu"];
            dt_ = meta_data["dt"];
            file.close();

            // load functions
            std::string folder = "../../unittest/implicit/generators/solar_receiver_reactor/";
            eval_objk_ = Function::load(folder + "eval_objk.casadi");
            eval_objK_ = Function::load(folder + "eval_objK.casadi");
            eval_g0_ = Function::load(folder + "eval_g0.casadi");
            eval_gk_ = Function::load(folder + "eval_gk.casadi");
            eval_gk_ineq_ = Function::load(folder + "eval_gk_ineq.casadi");
            eval_gK_ = Function::load(folder + "eval_gK.casadi");
            eval_gK_ineq_ = Function::load(folder + "eval_gK_ineq.casadi");
            rhs_ = Function::load(folder + "rhs.casadi");
            expl_dyn_ = Function::load(folder + "expl_dyn.casadi");
            expl_dyn_rk2_ = Function::load(folder + "expl_dyn_rk2.casadi");
            impl_dyn_ = Function::load(folder + "impl_dyn.casadi");
            impl_dyn_trap_ = Function::load(folder + "impl_dyn_trap.casadi");

            // set parameter value
            MX k = MX::sym("k", 1, 1);
            eval_p_ = Function("eval_p", {k}, {400 + 0*(k > K_/3.0)*(k < 2*K_/3.0)*100});

            // set initialization
            x_init_ = std::vector<std::vector<double>>(K_+1, std::vector<double>(nx_, 0.0));
            u_init_ = std::vector<std::vector<double>>(K_, std::vector<double>(nu_, 0.0));
            for (int k = 0; k < K_+1; k++){
                for (int i = 0; i < nx_; i++){ x_init_[k][i] = x0_[i];}
            }
            for (int k = 0; k < K_; k++){ u_init_[k][0] = u0_;}
        };

        virtual ImplicitTestProblem PrepareImplicit(){           
            return ImplicitTestProblem(K_, nx_, nu_, 
                    x_init_, u_init_, 
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk_, eval_objK_, eval_gk_, eval_g0_, eval_gK_, eval_gk_ineq_, eval_gK_ineq_,
                    impl_dyn_trap_, eval_p_);
        }

        virtual ExplicitTestProblem PrepareRootFinder(){           
            Function eval_dynamics_equation_implicit = Function("eval_dynamics_equation", {xkp_, uk_, xk_, p_}, {impl_dyn_trap_(MXVector{uk_, xk_, xkp_, p_})[0]});
            Function rf = rootfinder("rf", "newton", eval_dynamics_equation_implicit);
            Function explicit_rootfinder = Function("explicit_rootfinder", {uk_, xk_, p_}, {rf(MXVector{xk_, uk_, xk_, p_})});
            return ExplicitTestProblem(K_, nx_, nu_, 
                    x_init_, u_init_, 
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk_, eval_objK_, eval_gk_, eval_g0_, eval_gK_, eval_gk_ineq_, eval_gK_ineq_,
                    explicit_rootfinder, eval_p_);
        }

        virtual ExplicitTestProblem PrepareExplicit(){           
            return ExplicitTestProblem(
                    K_, nx_, nu_, 
                    x_init_, u_init_,
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk_, eval_objK_, eval_gk_, eval_g0_, eval_gK_,
                    eval_gk_ineq_, eval_gK_ineq_, expl_dyn_rk2_, eval_p_);
        }

        virtual ExplicitTestProblem PrepareReformulated(){
            MX zk = MX::sym("zk", nx_);
            MX uk_aug = vertcat(uk_, zk);

            MXVector ukxk = {uk_, xk_};
            MX zk_sum = 0; for (int i = 0; i < nx_; i++){ zk_sum += zk(i);}
            Function eval_objk = Function("eval_objk", {uk_aug, xk_}, {eval_objk_(ukxk)[0] + 1.0e-10*zk_sum});
            Function eval_gk_ineq = Function("eval_gk_ineq", {uk_aug, xk_}, {eval_gk_ineq_(ukxk)[0]});
            Function eval_gK = Function("eval_gK", {xk_}, {eval_gK_(xk_)[0]});
            Function eval_gK_ineq = Function("eval_gK_ineq", {xk_}, {eval_gK_ineq_(xk_)[0]});
            
            Function eval_g0 = Function("eval_g0", {uk_aug, xk_, p_}, {vertcat(eval_g0_(ukxk)[0], impl_dyn_trap_(MXVector{uk_, xk_, zk, p_})[0])});
            Function eval_gk = Function("eval_gk", {uk_aug, xk_, p_}, {vertcat(eval_gk_(ukxk)[0], impl_dyn_trap_(MXVector{uk_, xk_, zk, p_})[0])});
            Function eval_dynamics_equation_reformulated = Function("eval_dynamics_equation", {uk_aug, xk_}, {zk});

            std::vector<std::vector<double>> u_init(K_, std::vector<double>(nu_ + nx_, 0.0));
            for (int k = 0; k < K_; k++){
                for (int i = 0; i < nu_; i++){ u_init[k][i] = u_init_[k][i];}
                for (int i = 0; i < nx_; i++){ u_init[k][nu_ + i] = x_init_[k][i];}
            }

            return ExplicitTestProblem(
                    K_, nx_, nu_+nx_, 
                    x_init_, u_init,
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk, eval_objK_, eval_gk, eval_g0, eval_gK,
                    eval_gk_ineq, eval_gK_ineq,
                    eval_dynamics_equation_reformulated, eval_p_);
        }

        void SolveOptiInstance(std::string solver_name = "fatrop"){
            Opti opti = Opti();

            int N = K_ - 1;

            std::vector<MX> xx_list = {};
            std::vector<MX> uu_list = {};
            for (int k = 0; k < N + 1; k++){
                xx_list.push_back(opti.variable(nx_));
                if (k < N){
                    uu_list.push_back(opti.variable(nu_));
                }
            }
            MX xx = vertcat(xx_list);
            MX uu = vertcat(uu_list);

            if (eval_g0_.sparsity_out(0).size1() > 0){
                opti.subject_to(eval_g0_(MXVector{uu_list[0], xx_list[0]})[0] == DM::zeros(eval_g0_.sparsity_out(0).size1()));
            }

            for (int k = 0; k < N; k++){
                MX dni = eval_p_(DM(k))[0];
                MX xk = xx_list[k];
                MX uk = uu_list[k];

                MX x_next = expl_dyn_rk2_(MXVector{uk, xk, dni})[0];
                opti.subject_to(xx_list[k+1] == x_next);

                if (eval_gk_.sparsity_out(0).size1() > 0){
                    opti.subject_to(eval_gk_(MXVector{uk, xk})[0] == DM::zeros(eval_gk_.sparsity_out(0).size1()));
                }

                if (eval_gk_ineq_.sparsity_out(0).size1() > 0){
                    opti.subject_to(lb_ <= (eval_gk_ineq_(MXVector{uk, xk})[0] <= ub_));
                }
            }

            if (eval_gK_.sparsity_out(0).size1() > 0){
                opti.subject_to(eval_gK_(MXVector{xx_list[N]})[0] == DM::zeros(eval_gK_.sparsity_out(0).size1()));
            }

            if (eval_gK_ineq_.sparsity_out(0).size1() > 0){
                opti.subject_to(lb_K_ <= (eval_gK_ineq_(xx_list[N])[0] <= ub_K_));
            }

            MX obj = 0;
            for (int k = 0; k < N; k++){
                MX xk = xx_list[k];
                MX uk = uu_list[k];
                obj += eval_objk_(MXVector{uk, xk})[0];
            }
            MX xk = xx_list[N];
            obj += eval_objK_(xk)[0];
            opti.minimize(obj);

            for (int k = 0; k < N + 1; k++){
                for (int i = 0; i < nx_; i++){
                    opti.set_initial(xx_list[k](i), x_init_[0][i]);
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
            solver_opts["max_iter"] = 100;

            if (solver_name == "fatrop"){
                casadi_opts["structure_detection"] = "auto";
                casadi_opts["fatrop.mu_init"] = 0.1;
            } else if (solver_name == "ipopt"){
                solver_opts["linear_solver"] = "ma27";
                // solver_opts["linear_solver"] = "MUMPS";
            }
            opti.solver(solver_name, casadi_opts, solver_opts);
            try{
                opti.solve();
            } catch (const std::exception &e){
                std::cout << "Solver failed: " << e.what() << std::endl;
            }
        }

        virtual json GetJsonData(){
            json j;
            j["problem_name"] = "solar_receiver_reactor";
            j["n"] = n_;
            j["K"] = K_;
            j["nx"] = nx_;
            j["nu"] = nu_;
            j["dt"] = dt_;
            return j;
        }

        virtual std::string GetInterfaceName(){ return "solar_receiver_reactor";};
        virtual std::string GetFileNameAppendix(){return std::to_string(n_);};

    private:
        int K_;
        double T_;
        double dt_;

        int n_;

        int nx_ = 18;
        int nu_ = 1;
        
        MX xk_ = MX::sym("xk", nx_);
        MX uk_ = MX::sym("uk", nu_);
        MX xkp_ = MX::sym("xkp", nx_);
        MX p_ = MX::sym("p", 1, 1);
        
        Function eval_objk_;
        Function eval_objK_;

        Function eval_g0_;
        Function eval_gk_;
        Function eval_gk_ineq_;
        Function eval_gK_;
        Function eval_gK_ineq_;
        Function rhs_;
        Function expl_dyn_;
        Function expl_dyn_rk2_;
        Function impl_dyn_;
        Function impl_dyn_trap_;
        Function eval_p_;

        std::vector<double> x0_ = {6.79982544e-01, 3.42209761e-01, 
            9.44749463e-03, 2.80015274e-01, 5.75566459e-01, 8.66733442e-01, 
            4.00021820e-02, 8.22237799e-02, 1.23819063e-01, 3.11737855e+02, 
            3.11947172e+02, 3.12186673e+02, 5.39943204e+02, 5.41843654e+02, 
            5.44006506e+02, 5.27211235e+02, 5.30084664e+02, 5.37091843e+02};
        double u0_ = 0.5186015644624212;
        std::vector<double> lb_ = {0, 0, 0, 0, 0, 0, 0, 0, 0, 298, 298, 298, 450, 450, 450, 450, 450, 450};
        std::vector<double> ub_ = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000};
        std::vector<double> lb_K_ = {};
        std::vector<double> ub_K_ = {};

        std::vector<std::vector<double>> x_init_;
        std::vector<std::vector<double>> u_init_;
};

#endif