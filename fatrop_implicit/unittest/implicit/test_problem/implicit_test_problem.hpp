#ifndef FATROP_IMPLICIT_TEST_PROBLEM_HPP
#define FATROP_IMPLICIT_TEST_PROBLEM_HPP

#include "fatrop/context/context.hpp"
#include "fatrop/context/generic.hpp"
#include "fatrop/ocp/ocp_abstract.hpp"

#include <casadi/casadi.hpp>

using namespace casadi;
using namespace fatrop;

/////////////////////////////////////
/////////////////////////////////////
////                             ////
////  IMPLICIT OCP TEST PROBLEM  ////
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
// eval_dynamics_equation(uk, xk, xkp) -> f(uk, xk, xkp) == 0
class ImplicitTestProblem : public ImplicitOcpAbstract{
    public:
        ImplicitTestProblem(Index K, const Index nx, const Index nu, 
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
            eval_gk_ = Function("eval_gk" + ts, {uk, xk}, eval_gk(ukxk));
            eval_g0_ = Function("eval_g0" + ts, {uk, xk}, eval_g0(ukxk));
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
            Ggt_ = Function("Ggt"+ts, {uk, xk}, {transpose(jacobian(eval_gk_(ukxk)[0], vertcat(uk, xk)))});
            GgKt_ = Function("GgKt"+ts, {xk}, {transpose(jacobian(eval_gK_(xk)[0], xk))});
            Gg0t_ = Function("Gg0t"+ts, {uk, xk}, {transpose(jacobian(eval_g0_(ukxk)[0], vertcat(uk, xk)))});
            Ggt_ineq_ = Function("Ggt_ineq"+ts, {uk, xk}, {transpose(jacobian(eval_gk_ineq_(ukxk)[0], vertcat(uk, xk)))});
            GgKt_ineq_ = Function("GgKt_ineq"+ts, {xk}, {transpose(jacobian(eval_gK_ineq_(xk)[0], xk))});

            // dynamics
            MXVector args = {uk, xk, xkp};
            if (eval_dynamics_equation.n_in() == 4){ args.push_back(p); use_parameter_for_dynamics_ = true;}

            eval_dynamics_equation_ = Function("eval_dynamics_equation" + ts, args, eval_dynamics_equation(args));
            BAbt_ = Function("BAbt"+ts, args, 
                {transpose(horzcat(
                    jacobian(eval_dynamics_equation_(args)[0], uk),      // B
                    jacobian(eval_dynamics_equation_(args)[0], xk)       // A
                ))});
            BAJbt_ = Function("BAJbt"+ts, args, 
                {transpose(jacobian(eval_dynamics_equation_(args)[0], vertcat(uk, xk, xkp))),
                 inv(transpose(jacobian(eval_dynamics_equation_(args)[0], xkp)))});
            BAJbt_no_inv_ = Function("BAJbt_no_inv"+ts, args, 
                {transpose(jacobian(eval_dynamics_equation_(args)[0], vertcat(uk, xk, xkp)))});
            // b_ = Function("b"+ts, args, {eval_dynamics_equation_(args)[0]});
            Jt_ = Function("Jt"+ts, args, 
                {transpose(
                    jacobian(eval_dynamics_equation_(args)[0], xkp)
                )});
            Jt_inv_ = Function("Jt_inv"+ts, args,
                {inv(Jt_(args)[0])});
   
            // construct lagrangian (containing uk, xk and potentially xkp) (omitting dynamics)
            MX lagrangian_k = obj_scale*eval_objk_(ukxk)[0] + \
                mtimes(transpose(lam_eq_k), eval_gk_(ukxk)[0]) + \
                mtimes(transpose(lam_ineq_k), eval_gk_ineq_(ukxk)[0]);
            MX lagrangian_0 = obj_scale*eval_objk_(ukxk)[0] + \
                mtimes(transpose(lam_eq_0), eval_g0_(ukxk)[0]) + \
                mtimes(transpose(lam_ineq_k), eval_gk_ineq_(ukxk)[0]);
            MX lagrangian_K = obj_scale*eval_objK_(xk)[0] + \
                mtimes(transpose(lam_eq_K), eval_gK_(xk)[0]) + \
                mtimes(transpose(lam_ineq_K), eval_gK_ineq_(xk)[0]);
            lag_hess_k_ = Function("lag_hess_k"+ts, {uk, xk, xkp, lam_eq_k, lam_ineq_k, obj_scale}, 
                {transpose(hessian(lagrangian_k, vertcat(uk, xk)))});
            lag_hess_0_ = Function("lag_hess_0"+ts, {uk, xk, xkp, lam_eq_0, lam_ineq_k, obj_scale}, 
                {transpose(hessian(lagrangian_0, vertcat(uk, xk)))});
            lag_hess_K_ = Function("lag_hess_K"+ts, {xk, xkp, lam_eq_K, lam_ineq_K, obj_scale}, 
                {transpose(hessian(lagrangian_K, xk))});

            // dynamics contribution
            if (eval_dynamics_equation.n_in() == 3){
                dyn_hess_kp_ = Function("dyn_hess_kp"+ts, {uk, xk, xkp, lam_dyn_k}, 
                    {transpose(hessian(mtimes(transpose(lam_dyn_k), eval_dynamics_equation_(args)[0]), vertcat(xkp, uk, xk)))});
            } else {
                dyn_hess_kp_ = Function("dyn_hess_kp"+ts, {uk, xk, xkp, lam_dyn_k, p}, 
                    {transpose(hessian(mtimes(transpose(lam_dyn_k), eval_dynamics_equation_(args)[0]), vertcat(xkp, uk, xk)))});
            }

            // update sparsities
            BAbt_sp_ = BAbt_.sparsity_out(0);
            BAJbt_sp_ = BAJbt_.sparsity_out(0);
            lag_hess_k_sp_ = lag_hess_k_.sparsity_out(0);
            lag_hess_0_sp_ = lag_hess_0_.sparsity_out(0);
            lag_hess_K_sp_ = lag_hess_K_.sparsity_out(0);
            dyn_hess_kp_sp_ = dyn_hess_kp_.sparsity_out(0);
            Ggt_sp_ = Ggt_.sparsity_out(0);
            GgKt_sp_ = GgKt_.sparsity_out(0);
            Gg0t_sp_ = Gg0t_.sparsity_out(0);
            Ggt_ineq_sp_ = Ggt_ineq_.sparsity_out(0);
            GgKt_ineq_sp_ = GgKt_ineq_.sparsity_out(0);
            Jt_sp_ = Jt_.sparsity_out(0);
            Jt_inv_sp_ = Jt_inv_.sparsity_out(0);
            // FuFxt_sp_ = FuFxt_.sparsity_out(0);

            CodeGenerateAll();
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
        /*
        virtual Index eval_BAJbt(const Scalar *states_kp1, const Scalar *inputs_k,
                                 const Scalar *states_k, MAT *res, MAT *res_J, MAT *res_J_inv, const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            if (k == K_ - 1){
                throw std::runtime_error("Error in eval_BAbt: cannot evaluate BAbt at final stage");
            }
            blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            blasfeo_gese_wrap(res_J->m, res_J->n, 0.0, res_J, 0, 0);
            blasfeo_gese_wrap(res_J_inv->m, res_J_inv->n, 0.0, res_J_inv, 0, 0);
            std::vector<const double*> arg_in = {inputs_k, states_k, states_kp1};
            if (use_parameter_for_dynamics_){
                // evaluate parameter
                double p_val[1];
                double k_double = static_cast<double>(k);
                std::vector<const double*> arg_in_p = {&k_double};
                std::vector<double*> arg_out_p = {p_val};
                eval_p_(arg_in_p, arg_out_p);
                arg_in.push_back(p_val);
            }
            std::vector<double*> arg_out = {&scratch_[0], &scratch2_[0]};
            BAJbt_gc_(arg_in, arg_out);

            // store nonzeros in the matrix
            int scratch_ptr = 0;
            const casadi_int* c = BAJbt_sp_.colind();
            int r;
            for (int i = 0; i < BAJbt_sp_.size2(); i++){
                for (int el = c[i]; el != c[i+1]; ++el){
                    r = BAJbt_sp_.row(el);
                    if (r < get_nu(k) + get_nx(k)){
                        // contribution to BA
                        blasfeo_matel_wrap(res, r, i) = scratch_[scratch_ptr];
                    } else {
                        // contribution to J
                        blasfeo_matel_wrap(res_J, r - get_nu(k) - get_nx(k), i) = scratch_[scratch_ptr];
                    }
                    scratch_ptr++;
                }
            }

            // store nonzeros in the inverse of J
            int scratch2_ptr = 0;
            const casadi_int* c2 = Jt_inv_sp_.colind();
            for (int i = 0; i < Jt_inv_sp_.size2(); i++){
                for (int el = c2[i]; el != c2[i+1]; ++el){
                    blasfeo_matel_wrap(res_J_inv, Jt_inv_sp_.row(el), i) = scratch2_[scratch2_ptr];
                    scratch2_ptr++;
                }
            }
            

            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                blasfeo_print_dmat(res->m, res->n, res, 0, 0);
            }
            return 0;
        }
        */
        virtual Index eval_BAJbt(const Scalar *states_kp1, const Scalar *inputs_k,
                                 const Scalar *states_k, MAT *res, MAT *res_J, MAT *res_J_inv, const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            if (k == K_ - 1){
                throw std::runtime_error("Error in eval_BAbt: cannot evaluate BAbt at final stage");
            }
            blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            blasfeo_gese_wrap(res_J->m, res_J->n, 0.0, res_J, 0, 0);
            blasfeo_gese_wrap(res_J_inv->m, res_J_inv->n, 0.0, res_J_inv, 0, 0);
            std::vector<const double*> arg_in = {inputs_k, states_k, states_kp1};
            if (use_parameter_for_dynamics_){
                // evaluate parameter
                double p_val[1];
                double k_double = static_cast<double>(k);
                std::vector<const double*> arg_in_p = {&k_double};
                std::vector<double*> arg_out_p = {p_val};
                eval_p_(arg_in_p, arg_out_p);
                arg_in.push_back(p_val);
            }
            // std::vector<double*> arg_out = {&scratch_[0], &scratch2_[0]};
            // BAJbt_gc_(arg_in, arg_out);
            std::vector<double*> arg_out = {&scratch_[0]};
            BAJbt_no_inv_gc_(arg_in, arg_out);

            // store nonzeros in the matrix
            int scratch_ptr = 0;
            const casadi_int* c = BAJbt_sp_.colind();
            int r;
            for (int i = 0; i < BAJbt_sp_.size2(); i++){
                for (int el = c[i]; el != c[i+1]; ++el){
                    r = BAJbt_sp_.row(el);
                    if (r < get_nu(k) + get_nx(k)){
                        // contribution to BA
                        blasfeo_matel_wrap(res, r, i) = scratch_[scratch_ptr];
                    } else {
                        // contribution to J
                        blasfeo_matel_wrap(res_J, r - get_nu(k) - get_nx(k), i) = scratch_[scratch_ptr];
                    }
                    scratch_ptr++;
                }
            }

            /*
            // store nonzeros in the inverse of J
            int scratch2_ptr = 0;
            const casadi_int* c2 = Jt_inv_sp_.colind();
            for (int i = 0; i < Jt_inv_sp_.size2(); i++){
                for (int el = c2[i]; el != c2[i+1]; ++el){
                    blasfeo_matel_wrap(res_J_inv, Jt_inv_sp_.row(el), i) = scratch2_[scratch2_ptr];
                    scratch2_ptr++;
                }
            }
            */

            // compute the inverse of J
            /*
            // step 1: get LU factorization of Jt
            Index nx_next = get_nx(k+1);
            MatRealAllocated LU(nx_next, nx_next);
            MatRealAllocated I(nx_next, nx_next);
            for (Index i = 0; i < nx_next; ++i){
                blasfeo_matel_wrap(&I.mat(), i, i) = 1.0;
            }
            // MatRealAllocated res_J_inv_test(nx_next, nx_next);
            blasfeo_dgetrf_np(nx_next, nx_next, res_J, 0, 0, &LU.mat(), 0, 0);

            // step 2: compute inverse via solving with identity matrix
            MatRealAllocated Y(nx_next, nx_next);
            blasfeo_dtrsm_runn(nx_next, nx_next, 1.0, &LU.mat(), 0, 0, &I.mat(), 0, 0, &Y.mat(), 0, 0);
            blasfeo_dtrsm_rlnu(nx_next, nx_next, 1.0, &LU.mat(), 0, 0, &Y.mat(), 0, 0, res_J_inv, 0, 0);
            */

            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                blasfeo_print_dmat(res->m, res->n, res, 0, 0);
            }
            return 0;
        }

        virtual Index eval_BAJbt_no_inverse(const Scalar *states_kp1, const Scalar *inputs_k,
                                 const Scalar *states_k, MAT *res, MAT *res_J, const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            if (k == K_ - 1){
                throw std::runtime_error("Error in eval_BAbt: cannot evaluate BAbt at final stage");
            }
            blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);
            blasfeo_gese_wrap(res_J->m, res_J->n, 0.0, res_J, 0, 0);
            std::vector<const double*> arg_in = {inputs_k, states_k, states_kp1};
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
            std::cout << "calling function " << BAJbt_no_inv_gc_ << std::endl;
            BAJbt_no_inv_gc_(arg_in, arg_out);

            // store nonzeros in the matrix
            int scratch_ptr = 0;
            const casadi_int* c = BAJbt_sp_.colind();
            int r;
            for (int i = 0; i < BAJbt_sp_.size2(); i++){
                for (int el = c[i]; el != c[i+1]; ++el){
                    r = BAJbt_sp_.row(el);
                    if (r < get_nu(k) + get_nx(k)){
                        // contribution to BA
                        blasfeo_matel_wrap(res, r, i) = scratch_[scratch_ptr];
                    } else {
                        // contribution to J
                        blasfeo_matel_wrap(res_J, r - get_nu(k) - get_nx(k), i) = scratch_[scratch_ptr];
                    }
                    scratch_ptr++;
                }
            }

            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                blasfeo_print_dmat(res->m, res->n, res, 0, 0);
            }
            return 0;
        }
        /*
        virtual Index eval_RSQrqt_old(const Scalar *objective_scale, 
                                    const Scalar *inputs_km1,
                                    const Scalar *states_km1,
                                    const Scalar *inputs_k,
                                    const Scalar *states_k, 
                                    const Scalar *states_kp1,
                                    const Scalar *lam_dyn_k,
                                    const Scalar *lam_dyn_km1,
                                    const Scalar *lam_eq_k, 
                                    const Scalar *lam_eq_ineq_k, 
                                    MAT *res, const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            
            blasfeo_gese_wrap(res->m, res->n, 0.0, res, 0, 0);

            // contribution of obj(uk, xk), equality and inequality constraints
            // and f(uk, xk, xkp) = 0
            Function lag_hess = (k == 0) ? lag_hess_0_gc_ : (k == K_ - 1) ? lag_hess_K_gc_ : lag_hess_k_gc_;
            Sparsity lag_hess_sp = (k == 0) ? lag_hess_0_sp_ : (k == K_ - 1) ? lag_hess_K_sp_ : lag_hess_k_sp_;
            std::vector<const double*> arg_in = (k == K_ - 1) ? 
                std::vector<const double*>{states_k, states_kp1, lam_dyn_k, lam_eq_k, lam_eq_ineq_k, objective_scale}:
                std::vector<const double*>{inputs_k, states_k, states_kp1, lam_dyn_k, lam_eq_k, lam_eq_ineq_k, objective_scale};
            
            std::vector<double*> arg_out = {&scratch_[0]};
            lag_hess(arg_in, arg_out);

            // store nonzeros in the matrix
            int scratch_ptr = 0;
            for (int j = 0; j < lag_hess_sp.size2(); j++){
                for (int i = 0; i < lag_hess_sp.size1(); i++){
                    if (i > res->m || j > res->n){
                        throw std::runtime_error("Error in eval_RSQrqt: trying to write outside of matrix bounds");
                    }
                    if (lag_hess_sp.has_nz(i, j)) {
                        blasfeo_matel_wrap(res, i, j) = scratch_[scratch_ptr];
                        scratch_ptr++;
                    } else {
                        blasfeo_matel_wrap(res, i, j) = 0.0;
                    }
                }
            }

            // contribution of f(uk-1, xk-1, xk) = 0
            if (k > 0){
                std::vector<const double*> arg_in_dyn = {inputs_km1, states_km1, states_k, lam_dyn_km1};
                std::vector<double*> arg_out_dyn = {&scratch_[0]};
                dyn_hess_kp_gc_(arg_in_dyn, arg_out_dyn);

                // store nonzeros in the matrix
                int scratch_ptr_dyn = 0;
                for (int j = 0; j < dyn_hess_kp_sp_.size2(); j++){
                    for (int i = 0; i < dyn_hess_kp_sp_.size1(); i++){
                        if (i > get_nu(k) + res->m || j > get_nu(k) + res->n){
                            throw std::runtime_error("Error in eval_RSQrqt (2): trying to write outside of matrix bounds");
                        }
                        if (dyn_hess_kp_sp_.has_nz(i, j)) {
                            blasfeo_matel_wrap(res, get_nu(k) + i, get_nu(k) + j) += scratch_[scratch_ptr_dyn];
                            scratch_ptr_dyn++;
                        } else {
                            blasfeo_matel_wrap(res, get_nu(k) + i, get_nu(k) + j) += 0.0;
                        }
                    }
                }
            }

            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                blasfeo_print_dmat(res->m, res->n, res, 0, 0);
            }
            return 0;
        };
        */
        virtual Index eval_RSQrqt(const Scalar *objective_scale, 
                                    const Scalar *inputs_k,
                                    const Scalar *states_k, 
                                    const Scalar *states_kp1,
                                    const Scalar *lam_dyn_k,
                                    const Scalar *lam_eq_k, 
                                    const Scalar *lam_eq_ineq_k, 
                                    MAT *res, MAT *res_kp1, 
                                    MAT *res_FuFx, const Index k)
        {
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
                // auto start = std::chrono::high_resolution_clock::now();

            // reset all of res except res[get_nu(k):-1, get_nu(k):]
            blasfeo_gese_wrap(get_nu(k), res->n, 0.0, res, 0, 0);
            blasfeo_gese_wrap(res->m - get_nu(k), get_nu(k), 0.0, res, get_nu(k), 0);
            blasfeo_gese_wrap(1, res->n, 0.0, res, get_nu(k) + get_nx(k), 0);

            if (res_kp1){blasfeo_gese_wrap(res_kp1->m, res_kp1->n, 0.0, res_kp1, 0, 0);}
            if (res_FuFx){blasfeo_gese_wrap(res_FuFx->m, res_FuFx->n, 0.0, res_FuFx, 0, 0);}

            // contribution of obj(uk, xk), equality and inequality constraints
            Function lag_hess = (k == 0) ? lag_hess_0_gc_ : (k == K_ - 1) ? lag_hess_K_gc_ : lag_hess_k_gc_;
            Sparsity lag_hess_sp = (k == 0) ? lag_hess_0_sp_ : (k == K_ - 1) ? lag_hess_K_sp_ : lag_hess_k_sp_;
            std::vector<const double*> arg_in = (k == K_ - 1) ? 
                std::vector<const double*>{states_k, states_kp1, lam_eq_k, lam_eq_ineq_k, objective_scale}:
                std::vector<const double*>{inputs_k, states_k, states_kp1, lam_eq_k, lam_eq_ineq_k, objective_scale};
            
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
                    blasfeo_matel_wrap(res, lag_hess_sp.row(el), i) += scratch_[scratch_ptr];
                    scratch_ptr++;
                }
            }
                // stop = std::chrono::high_resolution_clock::now();
                // us_store_result_ += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

            // contribution of f(uk, xk, xkp1) = 0
            if (k < K_ - 1){
                    // start = std::chrono::high_resolution_clock::now();
                std::vector<const double*> arg_in_dyn = {inputs_k, states_k, states_kp1, lam_dyn_k};
                if (use_parameter_for_dynamics_){
                    // evaluate parameter
                    double p_val[1];
                    double k_double = static_cast<double>(k);
                    std::vector<const double*> arg_in_p = {&k_double};
                    std::vector<double*> arg_out_p = {p_val};
                    eval_p_(arg_in_p, arg_out_p);
                    arg_in_dyn.push_back(p_val);
                }
                std::vector<double*> arg_out_dyn = {&scratch_[0]};
                    // stop = std::chrono::high_resolution_clock::now();
                    // us_other_ += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
                    // start = std::chrono::high_resolution_clock::now();
                dyn_hess_kp_gc_(arg_in_dyn, arg_out_dyn);
                    
                    // stop = std::chrono::high_resolution_clock::now();
                    // us_function_call_ += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
                    // start = std::chrono::high_resolution_clock::now();

                // store nonzeros in the matrix   
                int scratch_ptr_dyn = 0;
                const casadi_int* c2 = dyn_hess_kp_sp_.colind();
                int r;
                for (int i = 0; i < dyn_hess_kp_sp_.size2(); i++){
                    for (int el = c2[i]; el != c2[i+1]; ++el){
                        r = dyn_hess_kp_sp_.row(el);
                        if (i < get_nx(k+1) && r < get_nx(k+1)){
                            // contribution to res_kp1
                            blasfeo_matel_wrap(res_kp1, r + get_nu(k+1), i + get_nu(k+1)) = scratch_[scratch_ptr_dyn];
                        } else if (i >= get_nx(k+1) && r >= get_nx(k+1)){
                            // contribution to res
                            blasfeo_matel_wrap(res, r - get_nx(k+1), i - get_nx(k+1)) += scratch_[scratch_ptr_dyn];
                        } else if (i < get_nx(k+1)){
                            // contribution to res_FuFxt
                            blasfeo_matel_wrap(res_FuFx, i, r - get_nx(k+1)) = scratch_[scratch_ptr_dyn];
                        } // otherwise discard contribution to res_FuFx
                        // blasfeo_matel_wrap(res, dyn_hess_kp_sp_.row(el), i) = scratch_[scratch_ptr_dyn];
                        scratch_ptr_dyn++;
                    }
                }
                

                    // stop = std::chrono::high_resolution_clock::now();
                    // us_store_result_ += std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
            }

            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                blasfeo_print_dmat(res->m, res->n, res, 0, 0);
            }
            return 0;
        };
        void get_hess_time_breakdown(int nb_iterations){
            double total = us_other_ + us_function_call_ + us_store_result_;
            if (total == 0){ return;}
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
            std::vector<const double*> arg_in = {inputs_k, states_k, states_kp1};
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
            
            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                std::cout << "b: ";
                for (Index i = 0; i < get_nx(k); ++i) {
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
        virtual ~ImplicitTestProblem() = default;

        /*
        virtual Index eval_Jt(const Scalar *states_kp1, const Scalar *inputs_k,
                                const Scalar *states_k, MAT *res, const Index k){
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            if (k == K_ - 1){
                throw std::runtime_error("[ocp_interface_generator]: Jt is not defined for the last stage.");
            }
            std::vector<const double*> arg_in = {inputs_k, states_k, states_kp1};
            std::vector<double*> arg_out = {&scratch_[0]};
            Jt_gc_(arg_in, arg_out);
            
            // store nonzeros in the matrix
            int scratch_ptr = 0;
            for (int j = 0; j < Jt_sp_.size2(); j++){
                for (int i = 0; i < Jt_sp_.size1(); i++){
                    if (i > res->m || j > res->n){
                        throw std::runtime_error("Error in eval_Jt: trying to write outside of matrix bounds");
                    }
                    if (Jt_.sparsity_out(0).has_nz(i, j)) {
                        blasfeo_matel_wrap(res, i, j) = scratch_[scratch_ptr];
                        scratch_ptr++;
                    } else {
                        blasfeo_matel_wrap(res, i, j) = 0.0;
                    }
                }
            }

            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                blasfeo_print_dmat(res->m, res->n, res, 0, 0);
            }
            return 0;
        };

        virtual Index eval_Jt_inv(const Scalar *states_kp1, const Scalar *inputs_k,
                                    const Scalar *states_k, MAT *res, const Index k){
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            if (k == K_ - 1){
                throw std::runtime_error("[ocp_interface_generator]: Jt_inv is not defined for the last stage.");
            }
            std::vector<const double*> arg_in = {inputs_k, states_k, states_kp1};
            std::vector<double*> arg_out = {&scratch_[0]};
            Jt_inv_gc_(arg_in, arg_out);
            
            // store nonzeros in the matrix
            int scratch_ptr = 0;
            for (int j = 0; j < Jt_inv_sp_.size2(); j++){
                for (int i = 0; i < Jt_inv_sp_.size1(); i++){
                    if (i > res->m || j > res->n){
                        throw std::runtime_error("Error in eval_Jt_inv: trying to write outside of matrix bounds");
                    }
                    if (Jt_inv_.sparsity_out(0).has_nz(i, j)) {
                        blasfeo_matel_wrap(res, i, j) = scratch_[scratch_ptr];
                        scratch_ptr++;
                    } else {
                        blasfeo_matel_wrap(res, i, j) = 0.0;
                    }
                }
            }

            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                blasfeo_print_dmat(res->m, res->n, res, 0, 0);
            }
            return 0;
        };

        virtual Index eval_FuFxt(const Scalar *inputs_k, const Scalar *states_k, 
                                 const Scalar *states_kp1, 
                                 const Scalar *lam_dyn_k,MAT *res, const Index k){
            if (DEBUG_PRINT){
                std::cout << "entering " << __func__ << " [" << k << "]" << std::endl;
            }
            if (k == K_ - 1){
                throw std::runtime_error("[ocp_interface_generator]: FuFxt is not defined for the last stage.");
            }
            std::vector<const double*> arg_in = {inputs_k, states_k, states_kp1, lam_dyn_k};
            std::vector<double*> arg_out = {&scratch_[0]};
            FuFxt_gc_(arg_in, arg_out);
            

            // store nonzeros in the matrix
            int scratch_ptr = 0;
            for (int j = 0; j < FuFxt_sp_.size2(); j++){
                for (int i = 0; i < FuFxt_sp_.size1(); i++){
                    if (i > res->m || j > res->n){
                        throw std::runtime_error("Error in eval_FuFxt: trying to write outside of matrix bounds");
                    }
                    if (FuFxt_sp_.has_nz(i, j)) {
                        blasfeo_matel_wrap(res, i, j) = scratch_[scratch_ptr];
                        scratch_ptr++;
                    } else {
                        blasfeo_matel_wrap(res, i, j) = 0.0;
                    }
                }
            }

            if (DEBUG_PRINT){
                std::cout << __func__ << " [" << k << "]" << std::endl;
                blasfeo_print_dmat(res->m, res->n, res, 0, 0);
            }
            return 0;
        };
        */

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
            int flag = system(compile_command.c_str());
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
        };

        void CodeGenerateAll(){
            if (!USE_CODEGEN_){
                eval_objk_gc_ = eval_objk_; eval_objK_gc_ = eval_objK_;
                eval_gk_gc_ = eval_gk_; eval_g0_gc_ = eval_g0_;
                eval_gK_gc_ = eval_gK_; eval_gk_ineq_gc_ = eval_gk_ineq_;
                eval_gK_ineq_gc_ = eval_gK_ineq_;
                eval_dynamics_equation_gc_ = eval_dynamics_equation_;
                grad_gc_ = grad_; grad_K_gc_ = grad_K_; /*BAbt_gc_ = BAbt_;*/
                /*BAJbt_gc_ = BAJbt_;*/ BAJbt_no_inv_gc_ = BAJbt_no_inv_;
                lag_hess_k_gc_ = lag_hess_k_; lag_hess_0_gc_ = lag_hess_0_;
                lag_hess_K_gc_ = lag_hess_K_; Ggt_gc_ = Ggt_; GgKt_gc_ = GgKt_;
                Gg0t_gc_ = Gg0t_; Ggt_ineq_gc_ = Ggt_ineq_;
                GgKt_ineq_gc_ = GgKt_ineq_; //b_gc_ = b_; Jt_gc_ = Jt_;
                //Jt_inv_gc_ = Jt_inv_; FuFxt_gc_ = FuFxt_;
                dyn_hess_kp_gc_ = dyn_hess_kp_;
            } else {
                // print out progress percentage
                std::vector<Function*> f = {
                    &eval_objk_, &eval_objK_, &eval_gk_, &eval_g0_, &eval_gK_,
                    &eval_gk_ineq_, &eval_gK_ineq_, &eval_dynamics_equation_,
                    &grad_, &grad_K_, /*&BAbt_, &BAJbt_,*/ &BAJbt_no_inv_, &lag_hess_k_, &lag_hess_0_,
                    &lag_hess_K_, &dyn_hess_kp_, &Ggt_, &GgKt_, &Gg0t_,
                    &Ggt_ineq_, &GgKt_ineq_//, &b_, &Jt_, &Jt_inv_, &FuFxt_
                };
                std::vector<Function*> f_gc = {
                    &eval_objk_gc_, &eval_objK_gc_, &eval_gk_gc_, &eval_g0_gc_, &eval_gK_gc_,
                    &eval_gk_ineq_gc_, &eval_gK_ineq_gc_, &eval_dynamics_equation_gc_,
                    &grad_gc_, &grad_K_gc_, /*&BAbt_gc_, &BAJbt_gc_,*/ &BAJbt_no_inv_gc_, &lag_hess_k_gc_, &lag_hess_0_gc_,
                    &lag_hess_K_gc_, &dyn_hess_kp_gc_, &Ggt_gc_, &GgKt_gc_, &Gg0t_gc_,
                    &Ggt_ineq_gc_, &GgKt_ineq_gc_//, &b_gc_, &Jt_gc_, &Jt_inv_gc_, &FuFxt_gc_
                };
                for (size_t i = 0; i < f.size(); i++){
                    int progress = int(( (i+1) / (double) f.size() ) * 100.0);
                    std::cout << "code generating function " << (*f[i]).name() << " (" << progress << "%)                                          \r";
                    std::cout.flush();
                    try{
                        // expand function
                        Function f_expanded = (*f[i]).expand();
                        *f_gc[i] = CodeGenerateFunction(f_expanded);
                    } catch (casadi::CasadiException& e){
                        *f_gc[i] = CodeGenerateFunction(*f[i]);
                    }
                }
                std::cout << std::endl;
            }
        };

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

        // deduced info
        Function grad_;
        Function grad_K_;
        Function BAbt_;
        Function BAJbt_;
        Function BAJbt_no_inv_;
        Function lag_hess_k_;
        Function lag_hess_0_;
        Function lag_hess_K_;
        Function dyn_hess_kp_;
        Function Ggt_;
        Function GgKt_;
        Function Gg0t_;
        Function Ggt_ineq_;
        Function GgKt_ineq_;
        // Function b_;
        Function Jt_;
        Function Jt_inv_;
        // Function FuFxt_;

        // code generated functions
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
        Function BAJbt_gc_;
        Function BAJbt_no_inv_gc_;
        Function lag_hess_k_gc_;
        Function lag_hess_0_gc_;
        Function lag_hess_K_gc_;
        Function dyn_hess_kp_gc_;
        Function Ggt_gc_;
        Function GgKt_gc_;
        Function Gg0t_gc_;
        Function Ggt_ineq_gc_;
        Function GgKt_ineq_gc_;
        // Function b_gc_;
        // Function Jt_gc_;
        // Function Jt_inv_gc_;
        // Function FuFxt_gc_;

        Sparsity BAbt_sp_;
        Sparsity BAJbt_sp_;
        Sparsity lag_hess_k_sp_;
        Sparsity lag_hess_0_sp_;
        Sparsity lag_hess_K_sp_;
        Sparsity dyn_hess_kp_sp_;
        Sparsity Ggt_sp_;
        Sparsity GgKt_sp_;
        Sparsity Gg0t_sp_;
        Sparsity Ggt_ineq_sp_;
        Sparsity GgKt_ineq_sp_;
        Sparsity Jt_sp_;
        Sparsity Jt_inv_sp_;
        // Sparsity FuFxt_sp_;

        double us_function_call_ = 0;
        double us_store_result_ = 0;
        double us_other_ = 0;

        // scratch space
        std::vector<double> scratch_ = std::vector<double>(100000, 0.0); // Adjust size as needed
        std::vector<double> scratch2_ = std::vector<double>(100000, 0.0); // Adjust size as needed
};  // IMPLICIT OCP TEST PROBLEM

#endif //FATROP_IMPLICIT_TEST_PROBLEM_HPP