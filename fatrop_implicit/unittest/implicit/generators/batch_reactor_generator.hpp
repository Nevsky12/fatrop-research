#ifndef __BATCH_REACTOR_GENERATOR_HPP__
#define __BATCH_REACTOR_GENERATOR_HPP__

#include "fatrop/context/context.hpp"
#include "fatrop/context/generic.hpp"
#include "fatrop/ocp/ocp_abstract.hpp"
#include "../test_problem/implicit_test_problem.hpp"    
#include "../test_problem/explicit_test_problem.hpp"
#include "ocp_interface_generator.hpp"

#include <casadi/casadi.hpp>

using namespace casadi;
using namespace fatrop;

// see: "Towards practical optimal control of batch reactors"
class BatchReactorGenerator : public InterfaceGenerator {
    public:
        // Constructor
        BatchReactorGenerator(int n){
            n_ = n;
            dt_ = T_/K_;

            // define dynamics //
            MX ki = MX(5,1);
            for (int i = 0; i < 5; i++){
                ki(i) = k0_[i] * exp(-E_[i]/(R_*xk_(4)));
            }

            MX rho = 2139*xk_(0) + 760*xk_(1) + 560*xk_(2) + 1800*xk_(3);
            MX Cp = 1.16*xk_(0) + 0.827*(1 - xk_(0)) + (3.4*xk_(0) + 0.92*(1-xk_(0)))*(xk_(4)-298)*1.0e-3;
            double UA = 1000;

            rhs_ = Function("rhs", {uk_, xk_}, {
                vertcat(
                    -ki(0)*xk_(0) - (ki(2) + ki(3) + ki(4))*xk_(0)*xk_(1),
                    ki(0)*xk_(0) - ki(1)*xk_(1) + ki(2)*xk_(0)*xk_(1),
                    ki(1)*xk_(1) + ki(3)*xk_(0)*xk_(1),
                    ki(4)*xk_(0)*xk_(1),
                    (602.4*ki(0)*xk_(0) - 0.833*UA/rho*(xk_(4)-xk_(5))) / Cp,
                    uk_(0)*(873 - xk_(5)) + uk_(1)*(373 - xk_(5)) + 0.01357*UA*(xk_(4)-xk_(5))
                )
            });
            // rhs_ = Function("rhs", {uk_, xk_}, {MX::zeros(nx_,1)});


            // set initialization //
            x_init_ = std::vector<std::vector<double>>(K_+1, std::vector<double>(nx_, 0.0));
            u_init_ = std::vector<std::vector<double>>(K_, std::vector<double>(nu_, 0.0));
            for (int k = 0; k < K_+1; k++){
                for (int i = 0; i < nx_; i++){ x_init_[k][i] = start_[i];}
            }
            for (int k = 0; k < K_; k++){ u_init_[k][0] = 1; u_init_[k][1] = 0;}
        };

        virtual ImplicitTestProblem PrepareImplicit(){           
            Function eval_dynamics_equation_implicit = Function("eval_dynamics_equation", {uk_, xk_, xkp_}, {xk_ + dt_*rhs_(MXVector{uk_, xkp_})[0] - xkp_});

            return ImplicitTestProblem(K_, nx_, nu_, 
                    x_init_, u_init_, 
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk_, eval_objK_, eval_gk_, eval_g0_, eval_gK_, eval_gk_ineq_, eval_gK_ineq_,
                    eval_dynamics_equation_implicit);
        }

        virtual ExplicitTestProblem PrepareRootFinder(){
            Function eval_dynamics_equation_implicit = Function("eval_dynamics_equation", {xkp_, uk_, xk_}, {xk_ + dt_*rhs_(MXVector{uk_, xkp_})[0] - xkp_});
            Function rf = rootfinder("rf", "newton", eval_dynamics_equation_implicit);
            Function explicit_rootfinder = Function("explicit_rootfinder", {uk_, xk_}, {rf(MXVector{xk_, uk_, xk_})});

            return ExplicitTestProblem(
                    K_, nx_, nu_, 
                    x_init_, u_init_,
                    lb_, ub_, lb_K_, ub_K_,
                    eval_objk_, eval_objK_, eval_gk_, eval_g0_, eval_gK_,
                    eval_gk_ineq_, eval_gK_ineq_, explicit_rootfinder);
        }

        virtual ExplicitTestProblem PrepareExplicit(){           
            Function eval_dynamics_equation_explicit = Function("eval_dynamics_equation", {uk_, xk_}, {xk_ + dt_*rhs_(MXVector{uk_, xk_})[0]});

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
            
            Function eval_g0 = Function("eval_g0", {uk_aug, xk_}, {vertcat(eval_g0_(ukxk)[0], xk_ + dt_*rhs_(MXVector{uk_, zk})[0] - zk)});
            Function eval_gk = Function("eval_gk", {uk_aug, xk_}, {vertcat(eval_gk_(ukxk)[0], xk_ + dt_*rhs_(MXVector{uk_, zk})[0] - zk)});
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

        void SolveOptiInstance(std::string solver_name="fatrop"){
            // test RHS values
            std::vector<double> x_numeric = {};
            std::vector<double> u_numeric = {};
            for (int i = 0; i < nx_; i++){
                x_numeric.push_back(0.1*i + 0.5);
            }
            for (int i = 0; i < nu_; i++){
                u_numeric.push_back(0.5*i + 0.1);
            }
            DM rhs_val = rhs_(DMVector{DM(u_numeric), DM(x_numeric)})[0];
            std::cout << "RHS value test: " << rhs_val << std::endl;

            // load rhs function 
            Function rhs_python = Function::load("../../unittest/implicit/generators/bath_reactor_rhs.casadi");

            std::cout << "dt: " << dt_ << std::endl;

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
            // opti.subject_to(xx_list[0] == DM(start_));

            for (int k = 0; k < N; k++){
                MX xk = xx_list[k];
                MX uk = uu_list[k];

                MX x_next = xk + dt_*rhs_(MXVector{uk, xk})[0];
                // MX x_next = xk + dt_*rhs_(MXVector{uk, xx_list[k+1]})[0];
                opti.subject_to(xx_list[k+1] == x_next);
                // opti.subject_to(xx_list[k+1] == xx_list[k]);

                if (eval_gk_ineq_.sparsity_out(0).size1() > 0){
                    opti.subject_to(lb_ <= (eval_gk_ineq_(MXVector{uk, xk})[0] <= ub_));
                }
                // opti.subject_to(0 <= (xx_list[k](Slice(0,4)) <= 1));
                // opti.subject_to(698.15 <= (xx_list[k](4) <= 748.15));
                // opti.subject_to(0 <= (uk <= 5));
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
            // for (int k = 0; k < K_+1; k++){
            //     opti.set_initial(xx_list[k], DM({0.95, 0.0, 0.0, 0.0, 698.15, 698.15}));
            //     if (k < K_){
            //         opti.set_initial(uu_list[k](0), 1);
            //         opti.set_initial(uu_list[k](1), 0);
            //     }
            // }

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

                /*
                /// define functions ///
                MX opti_x = opti.x();
                MX opti_g = opti.g();
                MX lam_g = opti.lam_g();
                MX opti_f = opti.f();
                Function g_vals = Function("g_vals", {opti_x}, {opti_g});
                Function g_jac = Function("g_jac", {opti_x}, {jacobian(opti_g, opti_x)});
                Function hess_lag = Function("hes_lag", {opti_x, lam_g}, 
                    {hessian(opti_f + mtimes(transpose(lam_g), opti_g), opti_x)});
                std::cout << "g_jac: " << g_jac << std::endl;
                std::cout << "hess_lag: " << hess_lag << std::endl;
                std::ofstream file;

                /// define variable values ///
                DM x_numeric = DM(opti_x.size());
                // int x_numeric_ptr = 0;
                // for (int k = 0; k < N + 1; k++){
                //     for (int i = 0; i < nx_; i++){
                //         x_numeric(x_numeric_ptr) = 0*x_init_[0][i];
                //         x_numeric_ptr++;
                //     }

                //     if (k < N - 1){
                //         for (int i = 0; i < nu_; i++){
                //             x_numeric(x_numeric_ptr) = 0*u_init_[0][i];
                //             x_numeric_ptr++;
                //         }
                //     }
                // }
                DM lam_g_numeric = DM(lam_g.size());
                // for (int i = 0; i < lam_g.size1(); i++){ lam_g_numeric(i) = 0.2*i;}

                /// construct full hess functions ///
                // randomize x_numeric en lam_g_numeric
                for (int i = 0; i < x_numeric.size1(); i++){
                    x_numeric(i) = double(rand()) / double(RAND_MAX);
                }
                for (int i = 0; i < lam_g_numeric.size1(); i++){
                    lam_g_numeric(i) = double(rand()) / double(RAND_MAX);
                }
                // set all lambdas corresponding to dynamics equal to zero
                // int lam_ptr = eval_g0_.size1_out(0);
                // for (int k = 0; k < N; k++){
                //     for (int i = 0; i < nx_; i++){
                //         lam_g_numeric(lam_ptr + i) = 0.0;
                //     }
                //     lam_ptr += nx_ + eval_gk_.size1_out(0) + eval_gk_ineq_.size1_out(0);
                // }
                Function opti_full_hess = hess_lag;
                std::cout << "\npreparing explicit full hess\n" << std::endl;
                ExplicitTestProblem etp = PrepareExplicit();
                Function interface_full_hess = etp.build_full_hessian();
                std::cout << "\n\tDone!\n" << std::endl;
                std::cout << "opti_full_hess: " << opti_full_hess << std::endl;
                std::cout << "interface_full_hess: " << interface_full_hess << std::endl;
                DM opti_full_hess_numeric = opti_full_hess(DMVector{x_numeric, lam_g_numeric})[0];
                DM interface_full_hess_numeric = interface_full_hess(DMVector{x_numeric, lam_g_numeric})[0];


                /// check equality of full hess ///
                if (false){
                    std::cout << "checking equality: " << std::endl;
                    DM equality_mtx = opti_full_hess_numeric - interface_full_hess_numeric;
                    for (int i = 0; i < equality_mtx.size1(); i++){
                        for (int j = 0; j < equality_mtx.size2(); j++){
                            if (double(equality_mtx(i,j)) > 1.0e-16){
                                std::cout << "First non-equal entry at (" << i << "," << j << "): " 
                                        << opti_full_hess_numeric(i,j) << " != " 
                                        << interface_full_hess_numeric(i,j) 
                                        << " (" << equality_mtx(i,j) << ")" << std::endl;
                                // throw std::runtime_error("Matrices are not equal!");
                            }
                        }
                    }
                }


                /// write full hess to file ///
                file.open("opti_full_hess.txt");
                for (int i = 0; i < opti_full_hess_numeric.size1(); i++){
                    for (int j = 0; j < opti_full_hess_numeric.size2(); j++){
                        // if (opti_full_hess.sparsity_out(0).has_nz(i,j)){
                        if (double(opti_full_hess_numeric(i,j)) != 0.0){
                            file << opti_full_hess_numeric(i,j) << std::endl;
                        }
                    }
                }
                file.close();
                file.open("interface_full_hess.txt");
                for (int i = 0; i < interface_full_hess_numeric.size1(); i++){
                    for (int j = 0; j < interface_full_hess_numeric.size2(); j++){
                        // if (interface_full_hess.sparsity_out(0).has_nz(i,j)){
                        if (double(interface_full_hess_numeric(i,j)) != 0.0){
                            file << interface_full_hess_numeric(i,j) << std::endl;
                        }
                    }
                }
                file.close();

                if (nx_ < 10){
                    file.open("opti_full_hess_mtx.txt");
                    for (int i = 0; i < opti_full_hess_numeric.size1(); i++){
                        for (int j = 0; j < opti_full_hess_numeric.size2(); j++){
                            file << opti_full_hess_numeric(i,j) << " ";
                        }
                        file << std::endl;
                    }
                    file.close();

                    file.open("interface_full_hess_mtx.txt");
                    for (int i = 0; i < interface_full_hess_numeric.size1(); i++){
                        for (int j = 0; j < interface_full_hess_numeric.size2(); j++){
                            file << interface_full_hess_numeric(i,j) << " ";
                        }
                        file << std::endl;
                    }
                    file.close();
                }
                */


            } catch (std::exception& e){
                std::cout << "Exception: " << e.what() << std::endl;
            }
        }

        virtual json GetJsonData(){
            json j;
            j["problem_name"] = "batch_reactor";
            j["n"] = n_;
            j["K"] = K_;
            j["nx"] = nx_;
            j["nu"] = nu_;
            j["dt"] = dt_;
            j["uk_min"] = uk_min_;
            j["uk_max"] = uk_max_;
            j["start"] = start_;
            return j;
        }

        virtual std::string GetInterfaceName(){ return "batch_reactor";};
        virtual std::string GetFileNameAppendix(){return std::to_string(n_);};

    private:
        MX ki(int i, MX T){ return k0_[i] * exp(-E_[i]/(R_*T));}

        int K_ = 80;
        double T_ = 9;
        double dt_;

        int n_;

        double uk_min_ = 0;
        double uk_max_ = 5;

        int nx_ = 6;
        int nu_ = 2;
        
        MX xk_ = MX::sym("xk", nx_);
        MX uk_ = MX::sym("uk", nu_);
        MX xkp_ = MX::sym("xkp", nx_);
        
        MX z = MX::zeros(0,1);
        MX xk_term = 1*1.0e-10*sumsqr(xk_);
        Function eval_objk_ = Function("eval_objk", {uk_, xk_}, {1.0e-6 * (uk_(0)*uk_(0) * uk_(1)*uk_(1)) + xk_term});
        Function eval_objK_ = Function("eval_objK", {xk_}, {-xk_(1) + xk_term});

        std::vector<double> start_ = {0.95, 0.0, 0.0, 0.0, 698.15, 698.15};
        Function eval_g0_ = Function("eval_g0", {uk_, xk_}, {xk_ - start_});
        // Function eval_g0_ = Function("eval_g0", {uk_, xk_}, {z});
        Function eval_gk_ = Function("eval_gk", {uk_, xk_}, {z});
        Function eval_gk_ineq_ = Function("eval_gk_ineq", {uk_, xk_}, {vertcat(uk_, xk_(Slice(0,5)))});
        // Function eval_gk_ineq_ = Function("eval_gk_ineq", {uk_, xk_}, {z});
        Function eval_gK_ = Function("eval_gK", {xk_}, {MX::zeros(0,1)});
        Function eval_gK_ineq_ = Function("eval_gK_ineq", {xk_}, {xk_(Slice(0,5))});
        Function rhs_;

        std::vector<double> lb_ = {0, 0, 0, 0, 0, 0, 698.15};
        std::vector<double> ub_ = {5, 5, 1, 1, 1, 1, 748.15};
        std::vector<double> lb_K_ = {0, 0, 0, 0, 698.15};
        std::vector<double> ub_K_ = {1, 1, 1, 1, 748.15};

        std::vector<double> k0_ = {exp(8.86), exp(24.25), exp(23.67), exp(18.75), exp(20.7)};
        std::vector<double> E_ = {20300, 37400, 33800, 28200, 31000};
        double R_ = 1.9872;

        std::vector<std::vector<double>> x_init_;
        std::vector<std::vector<double>> u_init_;
};

#endif