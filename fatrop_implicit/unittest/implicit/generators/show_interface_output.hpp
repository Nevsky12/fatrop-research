#ifndef __SHOW_INTERFACE_OUTPUT_HPP__
#define __SHOW_INTERFACE_OUTPUT_HPP__

#include "fatrop/ocp/ocp_abstract.hpp"
#include "fatrop/linear_algebra/matrix.hpp"

#include <iostream>
#include <fstream>

using namespace fatrop;

void show_interface_output(OcpAbstractTpl<OcpAbstractDynamic>& interface, std::string file_name){
    // open the file
    std::cout << "Showing interface output to " << file_name << std::endl;
    std::ofstream file(file_name);
    
    int K = interface.get_horizon_length();


    for (int k = 0; k < K; k++)
    {
        file << "k = " << k << ": nx = " << interface.get_nx(k) << ", nu = " << interface.get_nu(k)
                  << ", ng = " << interface.get_ng(k) << ", ng_ineq = " << interface.get_ng_ineq(k)
                  << std::endl;
    }

    std::vector<VecRealAllocated> xk_all; xk_all.reserve(K);
    std::vector<VecRealAllocated> uk_all; uk_all.reserve(K-1);
    std::vector<VecRealAllocated> lam_dyn_all; lam_dyn_all.reserve(K-1);
    std::vector<VecRealAllocated> lam_eq_all; lam_eq_all.reserve(K);
    std::vector<VecRealAllocated> lam_ineq_all; lam_ineq_all.reserve(K);
    for (int k = 0; k < K; k++){
        xk_all.emplace_back(interface.get_nx(k));
        for (int i = 0; i < interface.get_nx(k); i++) xk_all[k](i) = 0.1*(i+1) + k;
        
        lam_eq_all.emplace_back(interface.get_ng(k));
        for (int i = 0; i < interface.get_ng(k); i++) lam_eq_all[k](i) = 0.3*(i+1) + k;
        
        lam_ineq_all.emplace_back(interface.get_ng_ineq(k));
        for (int i = 0; i < interface.get_ng_ineq(k); i++) lam_ineq_all[k](i) = 0.4*(i+1) + k;

        if (k < K-1){
            uk_all.emplace_back(interface.get_nu(k));
            for (int i = 0; i < interface.get_nu(k); i++) uk_all[k](i) = 0.2*(i+1) - k;

            lam_dyn_all.emplace_back(interface.get_nx(k));
            for (int i = 0; i < interface.get_nx(k); i++) lam_dyn_all[k](i) = 0.5*(i+1) - k;
        }
    }

    // eval BAbt
    for (int k = 0; k < K - 1; k++){
        // virtual Index eval_BAbt(const Scalar *states_kp1, const Scalar *inputs_k,
        //                 const Scalar *states_k, MAT *res, const Index k)
        MatRealAllocated res(interface.get_nx(k) + interface.get_nu(k) + 1, interface.get_nx(k+1));
        interface.eval_BAbt(xk_all[k].data(), uk_all[k].data(), xk_all[k+1].data(), &res.mat(), k);
        file << "BAbt[" << k << "] = \n" << res << std::endl;
    }

    // eval RSQrqt
    for (int k = 0; k < K; k++){
        // virtual Index eval_RSQrqt(const Scalar *objective_scale, const Scalar *inputs_k,
        //                     const Scalar *states_k,
        //                     const Scalar *lam_dyn_k,
        //                     const Scalar *lam_eq_k, 
        //                     const Scalar *lam_eq_ineq_k, MAT *res,
        //                     const Index k)
        MatRealAllocated res(interface.get_nx(k) + interface.get_nu(k) + 1, interface.get_nx(k) + interface.get_nu(k));
        Scalar objective_scale = 0.9;
        interface.eval_RSQrqt(&objective_scale, 
                       (k == K-1) ? nullptr : uk_all[k].data(), 
                       xk_all[k].data(), 
                       (k == K-1) ? nullptr : lam_dyn_all[k].data(), 
                       lam_eq_all[k].data(), 
                       lam_ineq_all[k].data(),
                       &res.mat(), k);
        file << "RSQrqt[" << k << "] = \n" << res << std::endl;
    }

    // eval Ggt
    for (int k = 0; k < K; k++){
        // virtual Index eval_Ggt(const Scalar *inputs_k, const Scalar *states_k, MAT *res,
        //                     const Index k)
        MatRealAllocated res(interface.get_nx(k) + interface.get_nu(k) + 1, interface.get_ng(k));
        interface.eval_Ggt((k == K-1) ? nullptr : uk_all[k].data(), xk_all[k].data(), &res.mat(), k);
        file << "Ggt[" << k << "] = \n" << res << std::endl;
    }

    // eval Ggt_ineq
    for (int k = 0; k < K; k++){
        // virtual Index eval_Ggt_ineq(const Scalar *inputs_k, const Scalar *states_k, MAT *res,
        //                         const Index k)
        MatRealAllocated res(interface.get_nx(k) + interface.get_nu(k) + 1, interface.get_ng_ineq(k));
        interface.eval_Ggt_ineq((k == K-1) ? nullptr : uk_all[k].data(), xk_all[k].data(), &res.mat(), k);
        file << "Ggt_ineq[" << k << "] = \n" << res << std::endl;
    }

    // eval b
    for (int k = 0; k < K - 1; k++){
        // virtual Index eval_b(const Scalar *states_kp1, const Scalar *inputs_k,
        //                     const Scalar *states_k, Scalar *res, const Index k)
        VecRealAllocated res(interface.get_nx(k+1));
        interface.eval_b(xk_all[k+1].data(), uk_all[k].data(), xk_all[k].data(), res.data(), k);
        file << "b[" << k << "] = \n" << res << std::endl;
    }

    // eval g
    for (int k = 0; k < K; k++){
        // virtual Index eval_g(const Scalar *inputs_k, const Scalar *states_k, Scalar *res,
        //                     const Index k)
        VecRealAllocated res(interface.get_ng(k));
        interface.eval_g((k == K-1) ? nullptr : uk_all[k].data(), xk_all[k].data(), res.data(), k);
        file << "g[" << k << "] = \n" << res << std::endl;
    }

    // eval g_ineq
    for (int k = 0; k < K; k++){
        // virtual Index eval_gineq(const Scalar *inputs_k, const Scalar *states_k, Scalar *res,
        //                         const Index k)
        VecRealAllocated res(interface.get_ng_ineq(k));
        interface.eval_gineq((k == K-1) ? nullptr : uk_all[k].data(), xk_all[k].data(), res.data(), k);
        file << "g_ineq[" << k << "] = \n" << res << std::endl;
    }

    // eval rq
    for (int k = 0; k < K; k++){
        // virtual Index eval_rq(const Scalar *objective_scale, const Scalar *inputs_k,
        //                     const Scalar *states_k, Scalar *res, const Index k)
        VecRealAllocated res(interface.get_nx(k) + interface.get_nu(k));
        Scalar objective_scale = 0.9;
        interface.eval_rq(&objective_scale, 
                   (k == K-1) ? nullptr : uk_all[k].data(), 
                   xk_all[k].data(), 
                   res.data(), k);
        file << "rq[" << k << "] = \n" << res << std::endl;
    }

    // eval L
    for (int k = 0; k < K; k++){
        // virtual Index eval_L(const Scalar *objective_scale, const Scalar *inputs_k,
        //                     const Scalar *states_k, Scalar *res, const Index k)
        Scalar res;
        Scalar objective_scale = 0.9;
        interface.eval_L(&objective_scale, 
                   (k == K-1) ? nullptr : uk_all[k].data(), 
                   xk_all[k].data(), 
                   &res, k);
        file << "L[" << k << "] = \n" << res << std::endl;
    }

    // eval bounds
    for (int k = 0; k < K; k++){
        // virtual Index get_bounds(Scalar *lower, Scalar *upper, const Index k) const
        VecRealAllocated lower(interface.get_ng_ineq(k));
        VecRealAllocated upper(interface.get_ng_ineq(k));
        interface.get_bounds(lower.data(), upper.data(), k);
        file << "bounds[" << k << "] = \n";
        file << "lower: ";
        for (int i = 0; i < interface.get_ng_ineq(k); i++) file << lower(i) << " ";
        file << "\nupper: ";
        for (int i = 0; i < interface.get_ng_ineq(k); i++) file << upper(i) << " ";
        file << std::endl;
    }

    // get initial xk
    for (int k = 0; k < K; k++){
        // virtual Index get_initial_xk(Scalar *xk, const Index k) const
        VecRealAllocated xk(interface.get_nx(k));
        interface.get_initial_xk(xk.data(), k);
        file << "initial_xk[" << k << "] = \n" << xk << std::endl;
    }

    // get initial uk
    for (int k = 0; k < K-1; k++){
        // virtual Index get_initial_uk(Scalar *uk, const Index k) const
        VecRealAllocated uk(interface.get_nu(k));
        interface.get_initial_uk(uk.data(), k);
        file << "initial_uk[" << k << "] = \n" << uk << std::endl;
    }
}




void show_implicit_interface_output(OcpAbstractTpl<ImplicitOcpAbstractDynamic>& interface, std::string file_name){
    // open the file
    std::cout << "Showing interface output to " << file_name << std::endl;
    std::ofstream file(file_name);
    
    int K = interface.get_horizon_length();


    for (int k = 0; k < K; k++)
    {
        file << "k = " << k << ": nx = " << interface.get_nx(k) << ", nu = " << interface.get_nu(k)
                  << ", ng = " << interface.get_ng(k) << ", ng_ineq = " << interface.get_ng_ineq(k)
                  << std::endl;
    }

    std::vector<VecRealAllocated> xk_all; xk_all.reserve(K);
    std::vector<VecRealAllocated> uk_all; uk_all.reserve(K-1);
    std::vector<VecRealAllocated> lam_dyn_all; lam_dyn_all.reserve(K-1);
    std::vector<VecRealAllocated> lam_eq_all; lam_eq_all.reserve(K);
    std::vector<VecRealAllocated> lam_ineq_all; lam_ineq_all.reserve(K);
    for (int k = 0; k < K; k++){
        xk_all.emplace_back(interface.get_nx(k));
        for (int i = 0; i < interface.get_nx(k); i++) xk_all[k](i) = 0.1*(i+1) + k;
        
        lam_eq_all.emplace_back(interface.get_ng(k));
        for (int i = 0; i < interface.get_ng(k); i++) lam_eq_all[k](i) = 0.3*(i+1) + k;
        
        lam_ineq_all.emplace_back(interface.get_ng_ineq(k));
        for (int i = 0; i < interface.get_ng_ineq(k); i++) lam_ineq_all[k](i) = 0.4*(i+1) + k;

        if (k < K-1){
            uk_all.emplace_back(interface.get_nu(k));
            for (int i = 0; i < interface.get_nu(k); i++) uk_all[k](i) = 0.2*(i+1) - k;

            lam_dyn_all.emplace_back(interface.get_nx(k));
            for (int i = 0; i < interface.get_nx(k); i++) lam_dyn_all[k](i) = 0.5*(i+1) - k;
        }
    }

    // eval BAbt
    /*
    for (int k = 0; k < K - 1; k++){
        // virtual Index eval_BAbt(const Scalar *states_kp1, const Scalar *inputs_k,
        //                 const Scalar *states_k, MAT *res, const Index k)
        MatRealAllocated res(interface.get_nx(k) + interface.get_nu(k) + 1, interface.get_nx(k+1));
        interface.eval_BAbt(xk_all[k].data(), uk_all[k].data(), xk_all[k+1].data(), &res.mat(), k);
        file << "BAbt[" << k << "] = \n" << res << std::endl;
    }
    */

    // eval RSQrqt
    std::vector<MatRealAllocated> RSQrqt_all; RSQrqt_all.reserve(K);
    for (int k = 0; k < K; k++){
        RSQrqt_all.emplace_back(interface.get_nx(k) + interface.get_nu(k) + 1, interface.get_nx(k) + interface.get_nu(k));
        // assign random values
        for (int i = 0; i < RSQrqt_all[k].m(); i++){
            for (int j = 0; j < RSQrqt_all[k].n(); j++){
                RSQrqt_all[k](i,j) = 0.01*(i+1) + 0.001*(j+1) + k;
            }
        }
    }
    std::vector<MatRealAllocated> FuFxt_all; FuFxt_all.reserve(K);
    for (int k = 0; k < K-1; k++){
        FuFxt_all.emplace_back(interface.get_nx(k) + interface.get_nu(k), interface.get_nx(k+1));
        // assign random values
        for (int i = 0; i < FuFxt_all[k].m(); i++){
            for (int j = 0; j < FuFxt_all[k].n(); j++){
                FuFxt_all[k](i,j) = 0.02*(i+1) + 0.002*(j+1) + k;
            }
        }
    }
    // OLD APPROACH
    /*
    for (int k = 0; k < K; k++){
        // virtual Index eval_RSQrqt(const Scalar *objective_scale, const Scalar *inputs_k,
        //                           const Scalar *states_k, const Scalar *states_kp1, 
        //                           const Scalar *lam_dyn_k,
        //                           const Scalar *lam_eq_k, const Scalar *lam_eq_ineq_k, MAT *res, MAT *res_next,
        //                           const Index k)
        Scalar objective_scale = 0.9;
        interface.eval_RSQrqt_old(&objective_scale, 
                       (k == 0) ? nullptr : uk_all[k-1].data(), 
                       (k == 0) ? nullptr : xk_all[k-1].data(), 
                       (k == K-1) ? nullptr : uk_all[k].data(), 
                       xk_all[k].data(), 
                       (k == K-1) ? nullptr : xk_all[k+1].data(),
                       (k == K-1) ? nullptr : lam_dyn_all[k].data(), 
                       (k == 0) ? nullptr : lam_dyn_all[k-1].data(),
                       lam_eq_all[k].data(), 
                       lam_ineq_all[k].data(),
                       &RSQrqt_all[k].mat(),
                       k);
        file << "RSQrqt[" << k << "] = \n" << RSQrqt_all[k] << std::endl;
    }
    */
    for (int k = 0; k < K; k++){
        Scalar objective_scale = 0.9;
        // std::cout << "calling eval_RSQRqt for k = " << k << std::endl;
        interface.eval_RSQrqt(&objective_scale, 
                       (k == K-1) ? nullptr : uk_all[k].data(), 
                       xk_all[k].data(), 
                       (k == K-1) ? nullptr : xk_all[k+1].data(),
                       (k == K-1) ? nullptr : lam_dyn_all[k].data(), 
                       lam_eq_all[k].data(), 
                       lam_ineq_all[k].data(),
                       &RSQrqt_all[k].mat(),
                       (k == K-1) ? nullptr : &RSQrqt_all[k+1].mat(),
                       (k == K-1) ? nullptr : &FuFxt_all[k].mat(),
                       k);
    }
    for (int k = 0; k < K; k++){
        file << "RSQrqt[" << k << "] = \n" << RSQrqt_all[k] << std::endl;
    }
    // }

    for (int k = 0; k < K-1; k++){
        file << "FuFxt[" << k << "] = \n" << FuFxt_all[k] << std::endl;
    }

    // eval Ggt
    for (int k = 0; k < K; k++){
        // virtual Index eval_Ggt(const Scalar *inputs_k, const Scalar *states_k, MAT *res,
        //                     const Index k)
        MatRealAllocated res(interface.get_nx(k) + interface.get_nu(k) + 1, interface.get_ng(k));
        interface.eval_Ggt((k == K-1) ? nullptr : uk_all[k].data(), xk_all[k].data(), &res.mat(), k);
        file << "Ggt[" << k << "] = \n" << res << std::endl;
    }

    // eval Ggt_ineq
    for (int k = 0; k < K; k++){
        // virtual Index eval_Ggt_ineq(const Scalar *inputs_k, const Scalar *states_k, MAT *res,
        //                         const Index k)
        MatRealAllocated res(interface.get_nx(k) + interface.get_nu(k) + 1, interface.get_ng_ineq(k));
        interface.eval_Ggt_ineq((k == K-1) ? nullptr : uk_all[k].data(), xk_all[k].data(), &res.mat(), k);
        file << "Ggt_ineq[" << k << "] = \n" << res << std::endl;
    }

    // eval b
    for (int k = 0; k < K - 1; k++){
        // virtual Index eval_b(const Scalar *states_kp1, const Scalar *inputs_k,
        //                     const Scalar *states_k, Scalar *res, const Index k)
        VecRealAllocated res(interface.get_nx(k+1));
        interface.eval_b(xk_all[k+1].data(), uk_all[k].data(), xk_all[k].data(), res.data(), k);
        file << "b[" << k << "] = \n" << res << std::endl;
    }

    // eval g
    for (int k = 0; k < K; k++){
        // virtual Index eval_g(const Scalar *inputs_k, const Scalar *states_k, Scalar *res,
        //                     const Index k)
        VecRealAllocated res(interface.get_ng(k));
        interface.eval_g((k == K-1) ? nullptr : uk_all[k].data(), xk_all[k].data(), res.data(), k);
        file << "g[" << k << "] = \n" << res << std::endl;
    }

    // eval g_ineq
    for (int k = 0; k < K; k++){
        // virtual Index eval_gineq(const Scalar *inputs_k, const Scalar *states_k, Scalar *res,
        //                         const Index k)
        VecRealAllocated res(interface.get_ng_ineq(k));
        interface.eval_gineq((k == K-1) ? nullptr : uk_all[k].data(), xk_all[k].data(), res.data(), k);
        file << "g_ineq[" << k << "] = \n" << res << std::endl;
    }

    // eval rq
    for (int k = 0; k < K; k++){
        // virtual Index eval_rq(const Scalar *objective_scale, const Scalar *inputs_k,
        //                     const Scalar *states_k, Scalar *res, const Index k)
        VecRealAllocated res(interface.get_nx(k) + interface.get_nu(k));
        Scalar objective_scale = 0.9;
        interface.eval_rq(&objective_scale, 
                   (k == K-1) ? nullptr : uk_all[k].data(), 
                   xk_all[k].data(), 
                   res.data(), k);
        file << "rq[" << k << "] = \n" << res << std::endl;
    }

    // eval L
    for (int k = 0; k < K; k++){
        // virtual Index eval_L(const Scalar *objective_scale, const Scalar *inputs_k,
        //                     const Scalar *states_k, Scalar *res, const Index k)
        Scalar res;
        Scalar objective_scale = 0.9;
        interface.eval_L(&objective_scale, 
                   (k == K-1) ? nullptr : uk_all[k].data(), 
                   xk_all[k].data(), 
                   &res, k);
        file << "L[" << k << "] = \n" << res << std::endl;
    }

    // eval bounds
    for (int k = 0; k < K; k++){
        // virtual Index get_bounds(Scalar *lower, Scalar *upper, const Index k) const
        VecRealAllocated lower(interface.get_ng_ineq(k));
        VecRealAllocated upper(interface.get_ng_ineq(k));
        interface.get_bounds(lower.data(), upper.data(), k);
        file << "bounds[" << k << "] = \n";
        file << "lower: ";
        for (int i = 0; i < interface.get_ng_ineq(k); i++) file << lower(i) << " ";
        file << "\nupper: ";
        for (int i = 0; i < interface.get_ng_ineq(k); i++) file << upper(i) << " ";
        file << std::endl;
    }

    // get initial xk
    for (int k = 0; k < K; k++){
        // virtual Index get_initial_xk(Scalar *xk, const Index k) const
        VecRealAllocated xk(interface.get_nx(k));
        interface.get_initial_xk(xk.data(), k);
        file << "initial_xk[" << k << "] = \n" << xk << std::endl;
    }

    // get initial uk
    for (int k = 0; k < K-1; k++){
        // virtual Index get_initial_uk(Scalar *uk, const Index k) const
        VecRealAllocated uk(interface.get_nu(k));
        interface.get_initial_uk(uk.data(), k);
        file << "initial_uk[" << k << "] = \n" << uk << std::endl;
    }

    /*
    // eval J - eval Jt
    for (int k = 0; k < K - 1; k++){
        // virtual Index eval_J(const Scalar *states_kp1, const Scalar *inputs_k,
        //                     const Scalar *states_k, MAT *res, const Index k)
        MatRealAllocated res(interface.get_nx(k+1), interface.get_nx(k+1));
        interface.eval_Jt(xk_all[k+1].data(), uk_all[k].data(), xk_all[k].data(), &res.mat(), k);
        file << "Jt[" << k << "] = \n" << res << std::endl;

        // virtual Index eval_Jt(const Scalar *states_kp1, const Scalar *inputs_k,
        //                     const Scalar *states_k, MAT *res, const Index k)
        MatRealAllocated res_inv(interface.get_nx(k+1), interface.get_nx(k+1));
        interface.eval_Jt_inv(xk_all[k+1].data(), uk_all[k].data(), xk_all[k].data(), &res_inv.mat(), k);
        file << "Jt_inv[" << k << "] = \n" << res_inv << std::endl;

        // check if J * Jt_inv = I
        MatRealAllocated res_check(interface.get_nx(k+1), interface.get_nx(k+1));
        blasfeo_dgemm_nn(res.m(), res.m(), res.m(), 1.0, 
                const_cast<MAT *>(&res.mat()), 0, 0, 
                const_cast<MAT *>(&res_inv.mat()), 0, 0, 0.0, 
                const_cast<MAT *>(&res_check.mat()), 0, 0,
                const_cast<MAT *>(&res_check.mat()), 0, 0);
        file << "Jt * Jt_inv[" << k << "] = \n" << res_check << std::endl;
    }
    */


    // eval FuFxt
    /*
    for (int k = 0; k < K - 1; k++){
        // virtual Index eval_FuFxt(const Scalar *states_kp1, const Scalar *inputs_k,
        //                     const Scalar *states_k, MAT *res, const Index k)
        MatRealAllocated res(interface.get_nx(k) + interface.get_nu(k) + 1, interface.get_nx(k) + interface.get_nu(k) + 1);
        interface.eval_FuFxt(uk_all[k].data(), xk_all[k].data(), xk_all[k+1].data(), lam_dyn_all[k].data(), &res.mat(), k);
        file << "FuFxt[" << k << "] = \n" << res << std::endl;
    }
    */
}

void print_jac_and_hess(OcpAbstractTpl<OcpAbstractDynamic>& interface){
    std::cout << "executing " << __func__ << std::endl;
    int K = interface.get_horizon_length();

    ////////////////////////////////
    /// set values for xk and uk ///
    ////////////////////////////////
    std::vector<VecRealAllocated> xk_all; xk_all.reserve(K);
    std::vector<VecRealAllocated> uk_all; uk_all.reserve(K-1);
    int var_count = 0;
    double s = 1;
    for (int k = 0; k < K; k++){
        xk_all.emplace_back(interface.get_nx(k));
        // for (int i = 0; i < interface.get_nx(k); i++) xk_all[k](i) = s*var_count + s*i;
        // var_count += interface.get_nx(k);
        interface.get_initial_xk(xk_all[k].data(), k);

        if (k < K-1){
            uk_all.emplace_back(interface.get_nu(k));
        //     for (int i = 0; i < interface.get_nu(k); i++) uk_all[k](i) = s*var_count + s*i;
        //     var_count += interface.get_nu(k);
            interface.get_initial_uk(uk_all[k].data(), k);
        }
    }
    std::vector<double> all_vars = {};
    for (int k = 0; k < K; k++){
        if (k < K-1){
            for (int i = 0; i < interface.get_nu(k); i++) all_vars.push_back(uk_all[k](i));
        }
        for (int i = 0; i < interface.get_nx(k); i++) all_vars.push_back(xk_all[k](i));
    }
    std::string vars_file_name = "interface_all_vars.txt";
    std::ofstream vars_file(vars_file_name);
    for (int i = 0; i < all_vars.size(); i++) vars_file << all_vars[i] << std::endl;
    vars_file.close();







    //////////////////////////////
    /// set values for lambdas ///
    //////////////////////////////
    std::vector<VecRealAllocated> lam_dyn_all; lam_dyn_all.reserve(K-1);
    std::vector<VecRealAllocated> lam_g_eq_all; lam_g_eq_all.reserve(K);
    std::vector<VecRealAllocated> lam_g_ineq_all; lam_g_ineq_all.reserve(K);    
    var_count = 0; s = 0.2;
    for (int k = 0; k < K; k++){
        lam_g_eq_all.emplace_back(interface.get_ng(k));
        for (int i = 0; i < interface.get_ng(k); i++) lam_g_eq_all[k](i) = (var_count + i)*s;
        var_count += interface.get_ng(k);

        lam_g_ineq_all.emplace_back(interface.get_ng_ineq(k));
        for (int i = 0; i < interface.get_ng_ineq(k); i++) lam_g_ineq_all[k](i) = (var_count + i)*s;
        var_count += interface.get_ng_ineq(k);

        if (k < K-1){
            lam_dyn_all.emplace_back(interface.get_nx(k+1));
            for (int i = 0; i < interface.get_nx(k+1); i++) lam_dyn_all[k](i) = -(var_count + i)*s;
            var_count += interface.get_nx(k);
        }

    }
    std::vector<double> all_lams = {};
    for (int k = 0; k < K; k++){
        if (k < K-1){
            for (int i = 0; i < interface.get_nx(k+1); i++) all_lams.push_back(lam_dyn_all[k](i));
        }
        for (int i = 0; i < interface.get_ng(k); i++) all_lams.push_back(lam_g_eq_all[k](i));
        for (int i = 0; i < interface.get_ng_ineq(k); i++) all_lams.push_back(lam_g_ineq_all[k](i));
    }
    std::string lams_file_name = "interface_all_lams.txt";
    std::ofstream lams_file(lams_file_name);
    for (int i = 0; i < all_lams.size(); i++) lams_file << all_lams[i] << std::endl;
    lams_file.close();

    std::vector<double> g_jac_nz = {};
    std::vector<double> hess_lag_nz = {};
    std::vector<double> g_vals = {};
    




    ///////////////////////////////
    /// evaluate g and jacobian ///
    ///////////////////////////////
    for (int k = 0; k < K; k++){
        int ng = interface.get_ng(k);
        int ng_ineq = interface.get_ng_ineq(k);
        int nx = interface.get_nx(k);
        int nu = interface.get_nu(k);

        // g
        if (ng > 0){
            MatRealAllocated res_g(nx + nu + 1, ng);
            interface.eval_Ggt((k == K-1) ? nullptr : uk_all[k].data(), xk_all[k].data(), &res_g.mat(), k);
            for (int i = 0; i < res_g.m(); i++){
                for (int j = 0; j < res_g.n(); j++){
                    if (res_g(i,j) != 0.0){
                        g_jac_nz.push_back(res_g(i,j));
                    }
                }
            }

            // also store the g values
            VecRealAllocated res_g_val(ng);
            interface.eval_g((k == K-1) ? nullptr : uk_all[k].data(), xk_all[k].data(), res_g_val.data(), k);
            for (int i = 0; i < res_g_val.m(); i++){
                g_vals.push_back(res_g_val(i));
            }
        }

        // g_ineq
        if (ng_ineq > 0){
            MatRealAllocated res_g_ineq(nx + nu + 1, ng_ineq);
            interface.eval_Ggt_ineq((k == K-1) ? nullptr : uk_all[k].data(), xk_all[k].data(), &res_g_ineq.mat(), k);
            for (int i = 0; i < res_g_ineq.m(); i++){
                for (int j = 0; j < res_g_ineq.n(); j++){
                    if (res_g_ineq(i,j) != 0.0){
                        g_jac_nz.push_back(res_g_ineq(i,j));
                    }
                }
            }

            // also store the g_ineq values
            VecRealAllocated res_g_ineq_val(ng_ineq);
            interface.eval_gineq((k == K-1) ? nullptr : uk_all[k].data(), xk_all[k].data(), res_g_ineq_val.data(), k);
            for (int i = 0; i < res_g_ineq_val.m(); i++){
                g_vals.push_back(res_g_ineq_val(i));
            }
        }

        // dynamics
        if (k < K - 1){
            MatRealAllocated res_dyn(nx + nu + 1, interface.get_nx(k+1));
            interface.eval_BAbt(xk_all[k+1].data(), uk_all[k].data(), xk_all[k].data(), &res_dyn.mat(), k);
            for (int i = 0; i < res_dyn.m(); i++){
                for (int j = 0; j < res_dyn.n(); j++){
                    if (res_dyn(i,j) != 0.0){
                        g_jac_nz.push_back(-res_dyn(i,j));
                    }
                }
            }
            for (int i = 0; i < interface.get_nx(k+1); i++){
                g_jac_nz.push_back(1.0); // identity part
            }

            // also store the dynamics values
            VecRealAllocated res_dyn_val(interface.get_nx(k+1));
            interface.eval_b(xk_all[k+1].data(), uk_all[k].data(), xk_all[k].data(), res_dyn_val.data(), k);
            for (int i = 0; i < res_dyn_val.m(); i++){
                g_vals.push_back(-res_dyn_val(i));
            }
        }
    }




    ////////////////////////
    /// evaluate hessian ///
    ////////////////////////
    for (int k = 0; k < K; k++){
        int ng = interface.get_ng(k);
        int ng_ineq = interface.get_ng_ineq(k);
        int nx = interface.get_nx(k);
        int nu = interface.get_nu(k);

        MatRealAllocated res_hess(nx + nu + 1, nx + nu);
        Scalar objective_scale = 1.0;
        interface.eval_RSQrqt(&objective_scale, 
                       (k == K-1) ? nullptr : uk_all[k].data(), 
                       xk_all[k].data(), 
                       (k == K-1) ? nullptr : lam_dyn_all[k].data(), 
                       lam_g_eq_all[k].data(), 
                       lam_g_ineq_all[k].data(),
                       &res_hess.mat(), k);
        // blasfeo_print_dmat(nx + nu + 1, nx + nu, &res_hess.mat(), 0, 0);
        for (int i = 0; i < res_hess.m(); i++){
            for (int j = 0; j < res_hess.n(); j++){
                // if (res_hess(i,j) != 0.0){
                    hess_lag_nz.push_back(res_hess(i,j));
                // }
            }
        }
        // hess_lag_nz.push_back(10000000);
    }





    //////////////////////
    /// write to files ///
    //////////////////////
    std::string g_vals_file_name = "interface_g_vals.txt";
    std::cout << "Writing g values to " << g_vals_file_name << std::endl;
    std::ofstream file_g(g_vals_file_name);
    for (auto v : g_vals) file_g << v << std::endl;
    file_g.close();

    std::string g_jac_file_name = "interface_g_jac.txt";
    std::cout << "Writing g_jac to " << g_jac_file_name << std::endl;
    std::ofstream file(g_jac_file_name);
    for (auto v : g_jac_nz) file << v << std::endl; 
    file.close();

    std::string hess_lag_file_name = "interface_hess_lag.txt";
    std::cout << "Writing hess_lag to " << hess_lag_file_name << std::endl;
    std::ofstream file2(hess_lag_file_name);
    for (auto v : hess_lag_nz) file2 << v << std::endl;
    file2.close();


}

#endif // __SHOW_INTERFACE_OUTPUT_HPP__