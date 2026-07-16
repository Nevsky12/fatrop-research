#ifndef __EXPLICIT_TEST_PROBLEM_HPP__
#define __EXPLICIT_TEST_PROBLEM_HPP__

#include "fatrop/context/context.hpp"
#include "fatrop/context/generic.hpp"
#include "fatrop/ocp/ocp_abstract.hpp"

#include <casadi/casadi.hpp>

using namespace casadi;
using namespace fatrop;

/////////////////////////////////////
/////////////////////////////////////
////                             ////
////  EXPLICIT OCP TEST PROBLEM  ////
////                             ////
/////////////////////////////////////
/////////////////////////////////////
// Expected signatures:
// eval_objk(uk, xk) -> objk
// eval_objK(xk) -> objK
// eval_g0(uk, xk) -> g0
// eval_gk(uk, xk) -> gk
// eval_gK(xk) -> gK
// eval_gk_ineq(uk, xk) -> gk_ineq
// eval_gK_ineq(xk) -> gK_ineq
// eval_dynamics_equation(uk, xk, xkp) -> f(uk, xk)
class ExplicitTestProblem : public OcpAbstract{
    public:
        ExplicitTestProblem(Index K, const Index nx, const Index nu, 
            std::vector<std::vector<double>> x_init, 
            std::vector<std::vector<double>> u_init, 
            std::vector<double> g_ineq_lb, std::vector<double> g_ineq_ub,
            std::vector<double> g_ineq_K_lb, std::vector<double> g_ineq_K_ub, 
            Function eval_objk, Function eval_objK, Function eval_gk, Function eval_g0, 
            Function eval_gK, Function eval_gk_ineq, Function eval_gK_ineq,
            Function eval_dynamics_equation, Function eval_p = Function("eval_p", {MX::sym("k",1,1)}, {MX::zeros(0,0)})){
            K_ = K;
            nx_ = nx;
            nu_ = nu;
            x_init_ = x_init;
            u_init_ = u_init;
            g_ineq_lb_ = g_ineq_lb;
            g_ineq_ub_ = g_ineq_ub;
            g_ineq_K_lb_ = g_ineq_K_lb;
            g_ineq_K_ub_ = g_ineq_K_ub;

            MX xk = MX::sym("xk", nx_);
            MX uk = MX::sym("uk", nu_);
            MX xkp = MX::sym("xkp", nx_);
            MX p = MX::sym("p", 1, 1);
            MXVector ukxk = {uk, xk};
            MXVector ukxkxkp = {uk, xk, xkp};

            std::chrono::milliseconds ms = std::chrono::duration_cast< std::chrono::milliseconds >(
                std::chrono::system_clock::now().time_since_epoch()
            );
            long long timestamp = ms.count();
            std::string ts = "_" + std::to_string(timestamp);

            eval_objk_ = Function("eval_objk" + ts, {uk, xk}, eval_objk(ukxk));
            eval_objK_ = Function("eval_objK" + ts, {xk}, eval_objK({xk}));

            MXVector args_g = {uk, xk};
            if (eval_gk.n_in() == 3){ args_g.push_back(p); use_parameter_for_equality_constraints_ = true;}
            eval_gk_ = Function("eval_gk" + ts, args_g, eval_gk(args_g));
            eval_g0_ = Function("eval_g0" + ts, args_g, eval_g0(args_g));
            
            eval_gK_ = Function("eval_gK" + ts, {xk}, eval_gK({xk}));
            eval_gk_ineq_ = Function("eval_gk_ineq" + ts, {uk, xk}, eval_gk_ineq(ukxk));
            eval_gK_ineq_ = Function("eval_gK_ineq" + ts, {xk}, eval_gK_ineq({xk}));
            MX k = MX::sym("k", 1, 1);
            eval_p_ = Function("eval_p" + ts, {k}, eval_p({k}));

            // Initialize derived functions
            MX lam_dyn_k = MX::sym("lam_dyn_k", nx_);
            MX lam_eq_k = MX::sym("lam_eq_k", eval_gk_.sparsity_out(0).size1());
            MX lam_eq_0 = MX::sym("lam_eq_0", eval_g0_.sparsity_out(0).size1());
            MX lam_eq_K = MX::sym("lam_eq_K", eval_gK_.sparsity_out(0).size1());
            MX lam_ineq_k = MX::sym("lam_eq_k", eval_gk_ineq_.sparsity_out(0).size1());
            MX lam_ineq_K = MX::sym("lam_eq_K", eval_gK_ineq_.sparsity_out(0).size1());
            MX obj_scale = MX::sym("obj_scale", 1);

            grad_ = Function("grad"+ts, {uk, xk}, 
                {transpose(jacobian(eval_objk_(ukxk)[0], vertcat(uk, xk)))});
            grad_K_ = Function("gradK"+ts, {xk}, 
                {transpose(jacobian(eval_objK_(xk)[0], xk))});
            Ggt_ = Function("Ggt"+ts, args_g, {transpose(jacobian(eval_gk_(args_g)[0], vertcat(uk, xk)))});
            GgKt_ = Function("GgKt"+ts, {xk}, {transpose(jacobian(eval_gK_(xk)[0], xk))});
            Gg0t_ = Function("Gg0t"+ts, args_g, {transpose(jacobian(eval_g0_(args_g)[0], vertcat(uk, xk)))});
            Ggt_ineq_ = Function("Ggt_ineq"+ts, {uk, xk}, {transpose(jacobian(eval_gk_ineq_(ukxk)[0], vertcat(uk, xk)))});
            GgKt_ineq_ = Function("GgKt_ineq"+ts, {xk}, {transpose(jacobian(eval_gK_ineq_(xk)[0], xk))});

            // dynamics
            MXVector args = {uk, xk};
            if (eval_dynamics_equation.n_in() == 3){ args.push_back(p); use_parameter_for_dynamics_ = true;}
            eval_dynamics_equation_ = Function("eval_dynamics_equation" + ts, args, eval_dynamics_equation(args));
            BAbt_ = Function("BAbt"+ts, args, 
                {transpose(horzcat(
                    jacobian(eval_dynamics_equation_(args)[0], uk),      // B
                    jacobian(eval_dynamics_equation_(args)[0], xk)       // A
                ))});
            b_ = Function("b"+ts, {args}, {eval_dynamics_equation_(args)[0]});

            // construct lagrangian (containing uk, xk and potentially xkp)
            MX lagrangian_k = obj_scale*eval_objk_(ukxk)[0] + \
                mtimes(transpose(lam_dyn_k), b_(args)[0]) + \
                mtimes(transpose(lam_eq_k), eval_gk_(args_g)[0]) + \
                mtimes(transpose(lam_ineq_k), eval_gk_ineq_(ukxk)[0]);
            MX lagrangian_0 = obj_scale*eval_objk_(ukxk)[0] + \
                mtimes(transpose(lam_dyn_k), b_(args)[0]) + \
                mtimes(transpose(lam_eq_0), eval_g0_(args_g)[0]) + \
                mtimes(transpose(lam_ineq_k), eval_gk_ineq_(ukxk)[0]);
            MX lagrangian_K = obj_scale*eval_objK_(xk)[0] + \
                mtimes(transpose(lam_eq_K), eval_gK_(xk)[0]) + \
                mtimes(transpose(lam_ineq_K), eval_gK_ineq_(xk)[0]);
            if (use_parameter_for_dynamics_ || use_parameter_for_equality_constraints_){
                lag_hess_k_ = Function("lag_hess_k"+ts, {uk, xk, lam_dyn_k, lam_eq_k, lam_ineq_k, obj_scale, p}, 
                    {transpose(hessian(lagrangian_k, vertcat(uk, xk)))});               // RSQ[k+1]
                lag_hess_0_ = Function("lag_hess_0"+ts, {uk, xk, lam_dyn_k, lam_eq_0, lam_ineq_k, obj_scale, p}, 
                    {transpose(hessian(lagrangian_0, vertcat(uk, xk)))});               // RSQ[k+1]
            } else {
                lag_hess_k_ = Function("lag_hess_k"+ts, {uk, xk, lam_dyn_k, lam_eq_k, lam_ineq_k, obj_scale}, 
                    {transpose(hessian(lagrangian_k, vertcat(uk, xk)))});               // RSQ[k+1]
                lag_hess_0_ = Function("lag_hess_0"+ts, {uk, xk, lam_dyn_k, lam_eq_0, lam_ineq_k, obj_scale}, 
                    {transpose(hessian(lagrangian_0, vertcat(uk, xk)))});               // RSQ[k+1]
            }
            lag_hess_K_ = Function("lag_hess_K"+ts, {xk, lam_dyn_k, lam_eq_K, lam_ineq_K, obj_scale}, 
                {transpose(hessian(lagrangian_K, xk))});

            // update sparsities
            BAbt_sp_ = BAbt_.sparsity_out(0);
            lag_hess_k_sp_ = lag_hess_k_.sparsity_out(0);
            lag_hess_0_sp_ = lag_hess_0_.sparsity_out(0);
            lag_hess_K_sp_ = lag_hess_K_.sparsity_out(0);
            Ggt_sp_ = Ggt_.sparsity_out(0);
            GgKt_sp_ = GgKt_.sparsity_out(0);
            Gg0t_sp_ = Gg0t_.sparsity_out(0);
            Ggt_ineq_sp_ = Ggt_ineq_.sparsity_out(0);
            GgKt_ineq_sp_ = GgKt_ineq_.sparsity_out(0);

            CodeGenerateAll();
        };

        Function build_full_hessian(){
            MX full_hess = MX::zeros(nu_ * (K_-1) + nx_ * K_, nu_ * (K_-1) + nx_ * K_);
            // top-left: k == 0
            // variable sequence: [x0, u0, x1, u1 ... ] ! different than FATROP usually does this
            
            // define primal variables
            std::vector<MX> x_all = {};
            std::vector<MX> u_all = {};
            MX x;
            for (int k = 0; k < K_; k++){
                x_all.push_back(MX::sym("x_" + std::to_string(k), nx_));
                x = vertcat(x, x_all.back());
                if (k < K_-1){
                    u_all.push_back(MX::sym("u_" + std::to_string(k), nu_));
                    x = vertcat(x, u_all.back());
                }
            }

            // define lagrange multipliers
            std::vector<MX> lam_g_all = {};
            std::vector<MX> lam_g_ineq_all = {};
            std::vector<MX> lam_dyn_all = {};
            MX lam;
            int nb_eq_constraints = 0;
            int nb_ineq_constraints = 0;
            for (int k = 0; k < K_; k++){
                // equality constraints
                lam_g_all.push_back(MX::sym("lam_g_" + std::to_string(k), (k == 0) ? eval_g0_.sparsity_out(0).size1() : (k == K_-1) ? eval_gK_.sparsity_out(0).size1() : eval_gk_.sparsity_out(0).size1()));
                nb_eq_constraints += lam_g_all.back().size1();
                lam = vertcat(lam, lam_g_all.back());

                // dynamics constraints
                if (k < K_-1){
                    lam_dyn_all.push_back(MX::sym("lam_dyn_" + std::to_string(k), nx_));
                    lam = vertcat(lam, lam_dyn_all.back());
                    nb_eq_constraints += nx_;
                }

                // inequality constraints
                lam_g_ineq_all.push_back(MX::sym("lam_g_ineq_" + std::to_string(k), (k == K_-1) ? eval_gK_ineq_.sparsity_out(0).size1() : eval_gk_ineq_.sparsity_out(0).size1()));
                nb_ineq_constraints += lam_g_ineq_all.back().size1();
                lam = vertcat(lam, lam_g_ineq_all.back());
            }
            std::cout << "nb eq constraints: " << nb_eq_constraints << std::endl;
            std::cout << "nb ineq constraints: " << nb_ineq_constraints << std::endl;

            // evaluate all hessian contributions
            for (int k = 0; k < K_; k++){
                MX hess_k;
                if (k == 0){
                    MXVector in = {u_all[k], x_all[k], -lam_dyn_all[k], lam_g_all[k], lam_g_ineq_all[k], 1.0};
                    hess_k = lag_hess_0_(in)[0];
                } else if (k == K_-1){
                    MXVector in = {x_all[k], -lam_dyn_all[k-1], lam_g_all[k], lam_g_ineq_all[k], 1.0};
                    hess_k = lag_hess_K_(in)[0];
                } else {
                    MXVector in = {u_all[k], x_all[k], -lam_dyn_all[k], lam_g_all[k], lam_g_ineq_all[k], 1.0};
                    hess_k = lag_hess_k_(in)[0];
                }

                int nu = get_nu(k);
                int nx = get_nx(k);

                // place in full hessian
                Index row_start = k * (nx_ + nu_);
                Index col_start = row_start;

                // split the hessian in its parts
                MX hess_uu = hess_k(Slice(0, nu), Slice(0, nu));
                MX hess_ux = hess_k(Slice(0, nu), Slice(nu, nu + nx));
                MX hess_xu = hess_k(Slice(nu, nu + nx), Slice(0, nu));
                MX hess_xx = hess_k(Slice(nu, nu + nx), Slice(nu, nu + nx));

                // place in full hessian
                if (k < K_-1){
                    // full_hess(Slice(row_start, row_start + nu), Slice(col_start, col_start + nu)) = hess_uu;
                    // full_hess(Slice(row_start, row_start + nu), Slice(col_start + nu, col_start + nu + nx)) = hess_ux;
                    // full_hess(Slice(row_start + nu, row_start + nu + nx), Slice(col_start, col_start + nu)) = hess_xu;
                    // full_hess(Slice(row_start + nu, row_start + nu + nx), Slice(col_start + nu, col_start + nu + nx)) = hess_xx;
                    full_hess(Slice(row_start, row_start + nx), Slice(col_start, col_start + nx)) = hess_xx;
                    full_hess(Slice(row_start, row_start + nx), Slice(col_start + nx, col_start + nx + nu)) = hess_xu;
                    full_hess(Slice(row_start + nx, row_start + nx + nu), Slice(col_start, col_start + nx)) = hess_ux;
                    full_hess(Slice(row_start + nx, row_start + nx + nu), Slice(col_start + nx, col_start + nx + nu)) = hess_uu;
                } else {
                    full_hess(Slice(row_start, row_start + nx_), Slice(col_start, col_start + nx_)) = hess_xx;
                }
            }

            // wrap all in function
            Function full_hess_f = Function("full_hess_f", {x, lam}, {full_hess});
            return full_hess_f;
        };

        virtual Index get_nx(const Index k) const { return nx_;}
        virtual Index get_nu(const Index k) const { return (k == K_-1) ? 0 : nu_;}
        virtual Index get_ng(const Index k) const
        {
            if (k == 0) {
                return eval_g0_.sparsity_out(0).size1();
            } else if (k == K_ - 1) {
                return eval_gK_.sparsity_out(0).size1();
            } else {
                return eval_gk_.sparsity_out(0).size1();
            }
        };
        virtual Index get_ng_ineq(const Index k) const {
            if (k == K_ - 1) { return eval_gK_ineq_.sparsity_out(0).size1();}
            return eval_gk_ineq_.sparsity_out(0).size1();
        };
        virtual Index get_horizon_length() const { return K_; };
        virtual Index eval_BAbt(const Scalar *states_kp1, const Scalar *inputs_k,
                                const Scalar *states_k, MAT *res, const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            if (k == K_ - 1){
                throw std::runtime_error("Error in eval_BAbt: cannot evaluate BAbt at final stage");
            }
            std::vector<const double*> arg_in = {inputs_k, states_k};
            if (use_parameter_for_dynamics_){
                // evaluate parameter
                double p_val[1];
                double k_double = static_cast<double>(k);
                std::vector<const double*> arg_in_p = {&k_double};
                std::vector<double*> arg_out_p = {p_val};
                eval_p_(arg_in_p, arg_out_p);
                arg_in.push_back(p_val);
            }
            std::vector<double*> arg_out = {&scratch_[0]};
            BAbt_gc_(arg_in, arg_out);

            // store nonzeros in the matrix
            int scratch_ptr = 0;
            const casadi_int* c = BAbt_sp_.colind();
            for (int i = 0; i < BAbt_sp_.size2(); i++){
                for (int el = c[i]; el != c[i+1]; ++el){
                    blasfeo_matel_wrap(res, BAbt_sp_.row(el), i) = scratch_[scratch_ptr];
                    scratch_ptr++;
                }
            }

            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                blasfeo_print_dmat(res->m, res->n, res, 0, 0);
            }
            return 0;
        }
        virtual Index eval_RSQrqt(const Scalar *objective_scale, const Scalar *inputs_k,
                                    const Scalar *states_k,
                                    const Scalar *lam_dyn_k,
                                    const Scalar *lam_eq_k, 
                                    const Scalar *lam_eq_ineq_k, MAT *res,
                                    const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
                // auto start = std::chrono::high_resolution_clock::now();
            blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            Function lag_hess = (k == 0) ? lag_hess_0_gc_ : (k == K_ - 1) ? lag_hess_K_gc_ : lag_hess_k_gc_;
            Sparsity lag_hess_sp = (k == 0) ? lag_hess_0_sp_ : (k == K_ - 1) ? lag_hess_K_sp_ : lag_hess_k_sp_;
            std::vector<const double*> arg_in = (k == K_ - 1) ?
                std::vector<const double*>{states_k, lam_dyn_k, lam_eq_k, lam_eq_ineq_k, objective_scale} : 
                std::vector<const double*>{inputs_k, states_k, lam_dyn_k, lam_eq_k, lam_eq_ineq_k, objective_scale};
            if (k < K_ - 1 && (use_parameter_for_dynamics_ || use_parameter_for_equality_constraints_)){
                // evaluate parameter
                double p_val[1];
                double k_double = static_cast<double>(k);
                std::vector<const double*> arg_in_p = {&k_double};
                std::vector<double*> arg_out_p = {p_val};
                eval_p_(arg_in_p, arg_out_p);
                arg_in.push_back(p_val);
            }
            
            std::vector<double*> arg_out = {&scratch_[0]};
                // auto stop = std::chrono::high_resolution_clock::now();
                // us_other_ += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
                // start = std::chrono::high_resolution_clock::now();
            lag_hess(arg_in, arg_out);
                // stop = std::chrono::high_resolution_clock::now();
                // us_function_call_ += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
                // start = std::chrono::high_resolution_clock::now();
                        
            // store nonzeros in the matrix
            int scratch_ptr = 0;
            const casadi_int* c = lag_hess_sp.colind();
            for (int i = 0; i < lag_hess_sp.size2(); i++){
                for (int el = c[i]; el != c[i+1]; ++el){
                    blasfeo_matel_wrap(res, lag_hess_sp.row(el), i) = scratch_[scratch_ptr];
                    scratch_ptr++;
                }
            }

                // stop = std::chrono::high_resolution_clock::now();
                // us_store_result_ += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                // blasfeo_print_dmat(res->m, res->n, res, 0, 0);
                std::cout << "exiting function " << __func__ << std::endl;
            }
            return 0;
        };
        void get_hess_time_breakdown(int nb_iterations){
            double total = us_other_ + us_function_call_ + us_store_result_;
            if (total == 0) { return;}
            // std::cout << "Hessian computation time breakdown (microseconds):" << std::endl;
            std::cout << "\tfunction_call: " << us_function_call_/nb_iterations << " (" << (100.0*us_function_call_/total) << "%)" << std::endl;
            std::cout << "\tstore_result:  " << us_store_result_/nb_iterations  << " (" << (100.0*us_store_result_/total)  << "%)" << std::endl;
            std::cout << "\tother:         " << us_other_/nb_iterations         << " (" << (100.0*us_other_/total)         << "%)" << std::endl;
            std::cout << "\ttotal:         " << total/nb_iterations             << " (100%)"                        << std::endl;
        }
        virtual Index eval_Ggt(const Scalar *inputs_k, const Scalar *states_k, MAT *res,
                                const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            
            Function G = (k == 0) ? Gg0t_gc_ : (k == K_ - 1) ? GgKt_gc_ : Ggt_gc_;
            Sparsity G_sp = (k == 0) ? Gg0t_sp_ : (k == K_ - 1) ? GgKt_sp_ : Ggt_sp_;
            std::vector<const double*> arg_in = (k == K_ - 1) ? 
                std::vector<const double*>{states_k} : 
                std::vector<const double*>{inputs_k, states_k};
            if (k < K_ - 1 && use_parameter_for_equality_constraints_){
                // evaluate parameter
                double p_val[1];
                double k_double = static_cast<double>(k);
                std::vector<const double*> arg_in_p = {&k_double};
                std::vector<double*> arg_out_p = {p_val};
                eval_p_(arg_in_p, arg_out_p);
                arg_in.push_back(p_val);
            }
            
            std::vector<double*> arg_out = {&scratch_[0]};
            G(arg_in, arg_out);
           
            // store nonzeros in the matrix
            int scratch_ptr = 0;
            const casadi_int* c = G_sp.colind();
            for (int i = 0; i < G_sp.size2(); i++){
                for (int el = c[i]; el != c[i+1]; ++el){
                    blasfeo_matel_wrap(res, G_sp.row(el), i) = scratch_[scratch_ptr];
                    scratch_ptr++;
                }
            }
            
            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                blasfeo_print_dmat(res->m, res->n, res, 0, 0);
            }
            return 0;
        }
        virtual Index eval_Ggt_ineq(const Scalar *inputs_k, const Scalar *states_k, MAT *res,
                                    const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            Function Ggt_ineq = (k == K_ - 1) ? GgKt_ineq_gc_ : Ggt_ineq_gc_;
            Sparsity Ggt_ineq_sp = (k == K_ - 1) ? GgKt_ineq_sp_ : Ggt_ineq_sp_;
            std::vector<const double*> arg_in = (k == K_ - 1) ? 
                std::vector<const double*>{states_k} : 
                std::vector<const double*>{inputs_k, states_k};

            std::vector<double*> arg_out = {&scratch_[0]};
            Ggt_ineq(arg_in, arg_out);
            
            // store nonzeros in the matrix
            int scratch_ptr = 0;
            const casadi_int* c = Ggt_ineq_sp.colind();
            for (int i = 0; i < Ggt_ineq_sp.size2(); i++){
                for (int el = c[i]; el != c[i+1]; ++el){
                    blasfeo_matel_wrap(res, Ggt_ineq_sp.row(el), i) = scratch_[scratch_ptr];
                    scratch_ptr++;
                }
            }

            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                blasfeo_print_dmat(res->m, res->n, res, 0, 0);
            }
            return 0;
        };
        virtual Index eval_b(const Scalar *states_kp1, const Scalar *inputs_k,
                                const Scalar *states_k, Scalar *res, const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            std::vector<const double*> arg_in = {inputs_k, states_k};
            if (use_parameter_for_dynamics_){
                // evaluate parameter
                double p_val[1];
                double k_double = static_cast<double>(k);
                std::vector<const double*> arg_in_p = {&k_double};
                std::vector<double*> arg_out_p = {p_val};
                eval_p_(arg_in_p, arg_out_p);
                arg_in.push_back(p_val);
            }
            std::vector<double*> arg_out = {res};
            eval_dynamics_equation_gc_(arg_in, arg_out);
            for (int i = 0; i < get_nx(k+1); i++){
                res[i] -= states_kp1[i];
            }
            
            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                std::cout << "b: ";
                for (Index i = 0; i < get_nx(k+1); ++i) {
                    std::cout << res[i] << " ";
                }
                std::cout<<std::endl;
            }
            return 0;
        }

        virtual Index eval_g(const Scalar *inputs_k, const Scalar *states_k, Scalar *res,
                                const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            std::vector<const double*> arg_in = (k == K_ - 1) ? 
                std::vector<const double*>{states_k} : 
                std::vector<const double*>{inputs_k, states_k};
            if (k < K_ - 1 && use_parameter_for_equality_constraints_){
                // evaluate parameter
                double p_val[1];
                double k_double = static_cast<double>(k);
                std::vector<const double*> arg_in_p = {&k_double};
                std::vector<double*> arg_out_p = {p_val};
                eval_p_(arg_in_p, arg_out_p);
                arg_in.push_back(p_val);
            }

            std::vector<double*> arg_out = {res};
            if (k == 0){
                eval_g0_gc_(arg_in, arg_out);
            } else if (k == K_ - 1){
                eval_gK_gc_(arg_in, arg_out);
            } else {
                eval_gk_gc_(arg_in, arg_out);
            }

            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                std::cout << "g: ";
                for (Index i = 0; i < get_ng(k); ++i) {
                    std::cout << res[i] << " ";
                }
                std::cout<<std::endl;
            }
            return 0;
        };
        virtual Index eval_gineq(const Scalar *inputs_k, const Scalar *states_k, Scalar *res,
                                    const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            Function g_ineq = (k == K_ - 1) ? eval_gK_ineq_gc_ : eval_gk_ineq_gc_;
            std::vector<const double*> arg_in = (k == K_ - 1) ? 
                std::vector<const double*>{states_k} : 
                std::vector<const double*>{inputs_k, states_k};

            std::vector<double*> arg_out = {res};
            g_ineq(arg_in, arg_out);
            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                std::cout << "g_ineq: ";
                for (Index i = 0; i < g_ineq.sparsity_out(0).size1(); ++i) {
                    std::cout << res[i] << " ";
                }
                std::cout<<std::endl;
            }
            return 0;
        };
        virtual Index eval_rq(const Scalar *objective_scale, const Scalar *inputs_k,
                                const Scalar *states_k, Scalar *res, const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            Function grad = (k == K_ - 1) ? grad_K_gc_ : grad_gc_;
            std::vector<const double*> arg_in = (k == K_ - 1) ? 
                std::vector<const double*>{states_k} : 
                std::vector<const double*>{inputs_k, states_k};

            std::vector<double*> arg_out = {res};
            grad(arg_in, arg_out);

            for (Index i = 0; i < get_nu(k) + get_nx(k); ++i) {
                res[i] *= (*objective_scale);
            }

            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                std::cout << "grad: ";
                for (Index i = 0; i < get_nu(k) + get_nx(k); ++i) {
                    std::cout << res[i] << " ";
                }
                std::cout<<std::endl;
            }
            return 0;
        }
        virtual Index eval_L(const Scalar *objective_scale, const Scalar *inputs_k,
                                const Scalar *states_k, Scalar *res, const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            Function obj = (k == K_ - 1) ? eval_objK_gc_ : eval_objk_gc_;
            std::vector<const double*> arg_in = (k == K_ - 1) ? 
                std::vector<const double*>{states_k} :
                std::vector<const double*>{inputs_k, states_k};
            
            std::vector<double*> arg_out = {res};
            obj(arg_in, arg_out);
            res[0] *= (*objective_scale);        
            return 0;
        }
        virtual Index get_bounds(Scalar *lower, Scalar *upper, const Index k) const
        {
            if (k == K_ - 1){ 
                for (Index i = 0; i < eval_gK_ineq_.sparsity_out(0).size1(); ++i) {
                    lower[i] = g_ineq_K_lb_[i];
                    upper[i] = g_ineq_K_ub_[i];
                }    
                return 0;
            } else {
                for (Index i = 0; i < eval_gk_ineq_.sparsity_out(0).size1(); ++i) {
                    lower[i] = g_ineq_lb_[i];
                    upper[i] = g_ineq_ub_[i];
                }
                return 0;
            }
            return 0;
        }

        virtual Index get_initial_xk(Scalar *xk, const Index k) const
        {
            for (Index i = 0; i < get_nx(k); ++i) {
                xk[i] = x_init_[k][i];
            }
            return 0;
        };
        virtual Index get_initial_uk(Scalar *uk, const Index k) const
        {
            for (Index i = 0; i < get_nu(k); ++i) {
                uk[i] = u_init_[k][i];
            }
            return 0;
        };
        virtual ~ExplicitTestProblem() = default;  

        void set_debug_print(const bool debug_print){
            DEBUG_PRINT = debug_print;
        }

        void use_codegen(const bool use_codegen){
            if (use_codegen && !USE_CODEGEN_){
                USE_CODEGEN_ = use_codegen;
                CodeGenerateAll();
            } else {
                USE_CODEGEN_ = use_codegen;
            }
        }

    private:
        Function CodeGenerateFunction(Function &f){
            std::string name = f.name();
            f.generate(name);
            std::string compile_command = "gcc -fPIC -shared -O3 " + name + ".c -o " + name + ".so";
            // std::string compile_command = "gcc -fPIC -shared " + name + ".c -o " + name + ".so";
            // std::string compile_command = "gcc -fPIC -shared -march=native -ffast-math " + name + ".c -o " + name + ".so";
            int flag;
            try{
                flag = system(compile_command.c_str());
            } catch (const std::exception& e){
                return f;
            }
            if (flag != 0){
                // throw std::runtime_error("Error in CodeGenerateFunction: could not compile " + name + ".so");
                return f;
            }
            Function f_cg = external(name);

            // remove the generated files
            std::string remove_command = "rm " + name + ".c " + name + ".so";
            flag = system(remove_command.c_str());
            if (flag != 0){
                // throw std::runtime_error("Error in CodeGenerateFunction: could not remove generated files for " + name);
                return f;
            }
            return f_cg;
        }

        void CodeGenerateAll(){
            if (!USE_CODEGEN_){
                eval_objk_gc_ = eval_objk_; eval_objK_gc_ = eval_objK_;
                eval_gk_gc_ = eval_gk_; eval_g0_gc_ = eval_g0_;
                eval_gK_gc_ = eval_gK_; eval_gk_ineq_gc_ = eval_gk_ineq_;
                eval_gK_ineq_gc_ = eval_gK_ineq_;
                eval_dynamics_equation_gc_ = eval_dynamics_equation_;
                grad_gc_ = grad_; grad_K_gc_ = grad_K_; BAbt_gc_ = BAbt_;
                lag_hess_k_gc_ = lag_hess_k_; lag_hess_0_gc_ = lag_hess_0_;
                lag_hess_K_gc_ = lag_hess_K_; Ggt_gc_ = Ggt_; GgKt_gc_ = GgKt_;
                Gg0t_gc_ = Gg0t_; Ggt_ineq_gc_ = Ggt_ineq_;
                GgKt_ineq_gc_ = GgKt_ineq_;// b_gc_ = b_;
            } else {
                // print out progress percentage
                std::vector<Function*> f = {
                    &eval_objk_, &eval_objK_, &eval_gk_, &eval_g0_, &eval_gK_,
                    &eval_gk_ineq_, &eval_gK_ineq_, &eval_dynamics_equation_,
                    &grad_, &grad_K_, &BAbt_, &lag_hess_k_, &lag_hess_0_,
                    &lag_hess_K_, &Ggt_, &GgKt_, &Gg0t_,
                    &Ggt_ineq_, &GgKt_ineq_//, &b_
                };
                std::vector<Function*> f_gc = {
                    &eval_objk_gc_, &eval_objK_gc_, &eval_gk_gc_, &eval_g0_gc_, &eval_gK_gc_,
                    &eval_gk_ineq_gc_, &eval_gK_ineq_gc_, &eval_dynamics_equation_gc_,
                    &grad_gc_, &grad_K_gc_, &BAbt_gc_, &lag_hess_k_gc_, &lag_hess_0_gc_,
                    &lag_hess_K_gc_, &Ggt_gc_, &GgKt_gc_, &Gg0t_gc_,
                    &Ggt_ineq_gc_, &GgKt_ineq_gc_//, &b_gc_
                };
                for (size_t i = 0; i < f.size(); i++){
                    int progress = int(( (i+1) / (double) f.size() ) * 100.0);
                    std::cout << "code generating function " << (*f[i]).name() << " (" << progress << "%)                                          \r";
                    std::cout.flush();
                    // *f_gc[i] = CodeGenerateFunction(*f[i]);
                    // expand function
                    Function fsx = (*f[i]).expand();
                    *f_gc[i] = CodeGenerateFunction(fsx);
                }
                std::cout << std::endl;
            }
        }

        bool DEBUG_PRINT = false;
        bool USE_CODEGEN_ = false;

        // user-provided info
        Index K_;
        Index nx_;
        Index nu_;

        std::vector<std::vector<double>> x_init_;
        std::vector<std::vector<double>> u_init_;
        std::vector<double> g_ineq_lb_;
        std::vector<double> g_ineq_ub_;
        std::vector<double> g_ineq_K_lb_;
        std::vector<double> g_ineq_K_ub_;

        Function eval_objk_;
        Function eval_objK_;
        Function eval_gk_;
        Function eval_g0_;
        Function eval_gK_;
        Function eval_gk_ineq_;
        Function eval_gK_ineq_;
        Function eval_dynamics_equation_;
        Function eval_p_;
        bool use_parameter_for_dynamics_ = false;
        bool use_parameter_for_equality_constraints_ = false;

        // deduced info
        Function grad_;
        Function grad_K_;
        Function BAbt_;
        Function lag_hess_k_;
        Function lag_hess_0_;
        Function lag_hess_K_;
        Function Ggt_;
        Function GgKt_;
        Function Gg0t_;
        Function Ggt_ineq_;
        Function GgKt_ineq_;
        Function b_;

        // code generated
        Function eval_objk_gc_;
        Function eval_objK_gc_;
        Function eval_gk_gc_;
        Function eval_g0_gc_;
        Function eval_gK_gc_;
        Function eval_gk_ineq_gc_;
        Function eval_gK_ineq_gc_;
        Function eval_dynamics_equation_gc_;
        Function grad_gc_;
        Function grad_K_gc_;
        Function BAbt_gc_;
        Function lag_hess_k_gc_;
        Function lag_hess_0_gc_;
        Function lag_hess_K_gc_;
        Function Ggt_gc_;
        Function GgKt_gc_;
        Function Gg0t_gc_;
        Function Ggt_ineq_gc_;
        Function GgKt_ineq_gc_;

        Sparsity BAbt_sp_;
        Sparsity lag_hess_k_sp_;
        Sparsity lag_hess_0_sp_;
        Sparsity lag_hess_K_sp_;
        Sparsity Ggt_sp_;
        Sparsity GgKt_sp_;
        Sparsity Gg0t_sp_;
        Sparsity Ggt_ineq_sp_;
        Sparsity GgKt_ineq_sp_;

        double us_function_call_ = 0;
        double us_store_result_ = 0;
        double us_other_ = 0;

        // scratch space
        std::vector<double> scratch_ = std::vector<double>(100000, 0.0); // Adjust size as needed
        std::vector<double> scratch2_ = std::vector<double>(100000, 0.0); // Adjust size as needed
};  // EXPLICIT OCP TEST PROBLEM

#endif // __EXPLICIT_TEST_PROBLEM_HPP__