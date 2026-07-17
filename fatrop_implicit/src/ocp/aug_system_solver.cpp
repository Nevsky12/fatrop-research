//
// Copyright (C) 2024 Lander Vanroye, KU Leuven
//
// #define PROFILE             // define to enable detailed profiling
// #define IGNORE_EXTENSION    // define to use original fatrop algorithm
// #define IGNORE_JAC_HESS_PREPROCESS // define to ignore the pre- and postprocessing of jacobian and hessian

#define OFFSET_FREE_P2

#include "fatrop/ocp/aug_system_solver.hpp"
#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/problem_info.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
using namespace fatrop;


void PrintNpArray(MatRealAllocated const &A, std::string name, int m=-1, int n=-1, bool with_name=true, std::ostream& o = std::cout){
    if (m < 0){m = A.m();}
    if (n < 0){n = A.n();}

    if (with_name){
        o << name << " = np.array([\n\t";
    } else {
        o << "np.array([\n\t";
    }

    if (m == 0){ o << "[]";}
    else{
        for (int i = 0; i < m; i++){
            o << "[";
            for (int j = 0; j < n; j++){
                o << std::setw(10) << std::setprecision(10) << A(i,j);
                if (j < n - 1){ o << ",";}
                o << " ";
            }
            o << "],\n\t";        
        }
    }
    o << "])" << std::endl;
}

void PrintNpArray(VecRealAllocated const &v, std::string name, std::ostream& o = std::cout){
    o << name << " = np.transpose(np.array([[";
    for (int i = 0; i < v.m(); i++){
        o << v(i);
        if (i < v.m() - 1){ o << ",";}
        o << " ";
    }
    o << "]]))" << std::endl;
}

void WriteJacobianHessianToFile(const ProblemInfo &info, const Jacobian<OcpType> &jac, 
    const Hessian<OcpType> &hess, const VecRealView& f, const VecRealView& g, 
    const VecRealView& x, const VecRealView& mult,
    const std::string &filename_prefix)
{
    std::string file_folder = "";
    std::ofstream jac_file(file_folder + filename_prefix + "_jacobian.py");
    std::ofstream hess_file(file_folder + filename_prefix + "_hessian.py");
    std::ofstream f_file(file_folder + filename_prefix + "_f.py");
    std::ofstream g_file(file_folder + filename_prefix + "_g.py");
    std::ofstream x_file(file_folder + filename_prefix + "_x.py");
    std::ofstream mult_file(file_folder + filename_prefix + "_mult.py");

    if (!jac_file.is_open() || !hess_file.is_open() || !f_file.is_open() || !g_file.is_open() || !x_file.is_open() || !mult_file.is_open())
    {
        std::cerr << "Error opening file for writing Jacobian or Hessian." << std::endl;
        return;
    }
    jac_file << "import numpy as np" << std::endl;
    hess_file << "import numpy as np" << std::endl;
    f_file << "import numpy as np" << std::endl;
    g_file << "import numpy as np" << std::endl;
    x_file << "import numpy as np" << std::endl;
    mult_file << "import numpy as np" << std::endl;

    // Write Jacobian
    for (Index k = 0; k < info.dims.K; k++)
    {
        jac_file << "# Jacobian at stage " << k << ":\n";
        if (k < info.dims.K - 1){
            PrintNpArray(jac.BAbt[k], "BAbt_" + std::to_string(k), -1, -1, true, jac_file);
        }
        PrintNpArray(jac.Gg_eqt[k], "Gg_eqt_" + std::to_string(k), -1, -1, true, jac_file);
        PrintNpArray(jac.Gg_ineqt[k], "Gg_ineqt_" + std::to_string(k), -1, -1, true, jac_file);
        jac_file << "\n";
    }

    // Write Hessian
    for (Index k = 0; k < info.dims.K; k++)
    {
        hess_file << "# Hessian at stage " << k << ":\n";
        PrintNpArray(hess.RSQrqt[k], "RSQrqt_" + std::to_string(k), -1, -1, true, hess_file);
        hess_file << "\n";
    }

    // Write f, g, x and mult
    PrintNpArray(f, "f", f_file);
    PrintNpArray(g, "g", g_file);
    PrintNpArray(x, "x", x_file);
    PrintNpArray(mult, "mult", mult_file);

    jac_file.close();
    hess_file.close();
    f_file.close();
    g_file.close();
    x_file.close();
    mult_file.close();
}

void PrintNpArray(VecRealAllocated const &v, int offset, int length, std::string name){
    std::cout << name << " = np.transpose(np.array([[";
    for (int i = 0; i < length; i++){
        std::cout << v(offset + i);
        if (i < length - 1){ std::cout << ",";}
        std::cout << " ";
    }
    std::cout << "]]))" << std::endl;
}

bool check_reg(const Index m, MAT *sA, const Index ai, const Index aj)
{
    for (Index i = 0; i < m; i++)
    {
        if (blasfeo_matel_wrap(sA, ai + i, aj + i) < 1e-8)
            return false;
    }
    return true;
}

namespace
{
void solve_upper_unit_rectangular_batch(
    const Index leading_rows, const Index total_rows,
    const Index right_hand_sides, const MatRealView &factor,
    const Index factor_row, const Index factor_column,
    MatRealView &values, const Index value_row,
    const Index value_column)
{
    if (leading_rows < total_rows)
    {
        gemm_nn(
            leading_rows, right_hand_sides,
            total_rows - leading_rows, -1.0,
            factor, factor_row, factor_column + leading_rows,
            values, value_row + leading_rows, value_column,
            1.0, values, value_row, value_column,
            values, value_row, value_column);
    }
    if (leading_rows > 0)
    {
        trsm_lunu(
            leading_rows, right_hand_sides, 1.0,
            factor, factor_row, factor_column,
            values, value_row, value_column,
            values, value_row, value_column);
    }
}
} // namespace

AugSystemSolver<OcpType>::AugSystemSolver(const ProblemInfo &info)
{
    Index max_number_of_controls =
        *std::max_element(info.dims.number_of_controls.begin(), info.dims.number_of_controls.end());
    Index max_number_of_states =
        *std::max_element(info.dims.number_of_states.begin(), info.dims.number_of_states.end());
    Index max_number_of_variables = *std::max_element(info.number_of_stage_variables.begin(),
                                                      info.number_of_stage_variables.end());
    Index max_number_of_ineq_constraints = *std::max_element(
        info.dims.number_of_ineq_constraints.begin(), info.dims.number_of_ineq_constraints.end());
    Index max_number_of_eq_consttraints = *std::max_element(
        info.dims.number_of_eq_constraints.begin(), info.dims.number_of_eq_constraints.end());

    AL.emplace_back(max_number_of_variables + 1, max_number_of_variables);
    Ggt_stripe.emplace_back(max_number_of_variables + 1, max_number_of_variables);
    GgLt.emplace_back(max_number_of_variables + 1, max_number_of_variables);
    RSQrqt_hat.emplace_back(max_number_of_variables + 1, max_number_of_variables);
    Llt_shift.emplace_back(max_number_of_variables + 1, max_number_of_controls);
    GgIt_tilde.emplace_back(info.dims.number_of_states[0] + 1, info.dims.number_of_states[0]);
    GgLIt.emplace_back(info.dims.number_of_states[0] + 1, info.dims.number_of_states[0]);
    HhIt.emplace_back(info.dims.number_of_states[0] + 1, info.dims.number_of_states[0]);
    PpIt_hat.emplace_back(info.dims.number_of_states[0] + 1, info.dims.number_of_states[0]);
    LlIt.emplace_back(info.dims.number_of_states[0] + 1, info.dims.number_of_states[0]);
    Ggt_ineq_temp.emplace_back(max_number_of_variables + 1, max_number_of_ineq_constraints);

    Ppt.reserve(info.dims.K);
    Hh.reserve(info.dims.K);
    RSQrqt_tilde.reserve(info.dims.K);
    Ggt_tilde.reserve(info.dims.K);
    Llt.reserve(info.dims.K);
    for (Index k = 0; k < info.dims.K; k++)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        Index ng_eq = info.dims.number_of_eq_constraints[k];
        Ppt.emplace_back(nx + 1, nx);
        Hh.emplace_back(nx, nx + 1);
        RSQrqt_tilde.emplace_back(nu + nx + 1, nx + nu);
        Ggt_tilde.emplace_back(nu + nx + 1, nx + nu);
        Llt.emplace_back(nu + nx + 1, nu);
    }

    v_AL.emplace_back(max_number_of_variables);
    v_Ggt_stripe.emplace_back(max_number_of_variables);
    v_GgLt.emplace_back(max_number_of_variables);
    v_RSQrqt_hat.emplace_back(max_number_of_variables);
    v_Llt_shift.emplace_back(max_number_of_controls);
    v_GgIt_tilde.emplace_back(info.dims.number_of_states[0]);
    v_GgLIt.emplace_back(info.dims.number_of_states[0]);
    v_HhIt.emplace_back(info.dims.number_of_states[0]);
    v_PpIt_hat.emplace_back(info.dims.number_of_states[0]);
    v_LlIt.emplace_back(info.dims.number_of_states[0]);
    v_Ggt_ineq_temp.emplace_back(max_number_of_ineq_constraints);
    v_tmp.emplace_back(max_number_of_variables);

    v_Ppt.reserve(info.dims.K);
    v_Hh.reserve(info.dims.K);
    v_RSQrqt_tilde.reserve(info.dims.K);
    v_Ggt_tilde.reserve(info.dims.K);
    v_Llt.reserve(info.dims.K);

    for (Index k = 0; k < info.dims.K; k++)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        v_Ppt.emplace_back(nx);
        v_Hh.emplace_back(nx);
        v_RSQrqt_tilde.emplace_back(nu + nx);
        v_Ggt_tilde.emplace_back(nu + nx);
        v_Llt.emplace_back(nu + nx);
    }

    PlI.emplace_back(info.dims.number_of_states[0]);
    PrI.emplace_back(info.dims.number_of_states[0]);

    Pl.reserve(info.dims.K);
    Pr.reserve(info.dims.K);

    for (Index k = 0; k < info.dims.K; k++)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Pl.emplace_back(max_number_of_controls);
        Pr.emplace_back(max_number_of_controls);
    }

    gamma.resize(info.dims.K);
    rho.resize(info.dims.K);
};

void AugSystemSolver<OcpType>::register_options(OptionRegistry &registry)
{
    registry.register_option("linsol_it_ref", &AugSystemSolver<OcpType>::set_it_ref, this);
    registry.register_option("linsol_perturbed_mode", &AugSystemSolver<OcpType>::set_perturbed_mode, this);
    registry.register_option("linsol_perturbed_mode_param", &AugSystemSolver<OcpType>::set_perturbed_mode_param, this);
    registry.register_option("linsol_lu_fact_tol", &AugSystemSolver<OcpType>::set_lu_fact_tol, this);
    registry.register_option("linsol_diagnostic", &AugSystemSolver<OcpType>::set_diagnostic, this);
    registry.register_option("linsol_increased_accuracy", &AugSystemSolver<OcpType>::set_increased_accuracy, this);
}

LinsolReturnFlag AugSystemSolver<OcpType>::solve(const ProblemInfo &info,
                                           Jacobian<OcpType> &jacobian, Hessian<OcpType> &hessian,
                                           const VecRealView &D_x, const VecRealView &D_s,
                                           const VecRealView &f, const VecRealView &g,
                                           VecRealView &x, VecRealView &eq_mult)
{
    // std::cout << "solving here" << std::endl;
    // WriteJacobianHessianToFile(info, jacobian, hessian, f, g, x, eq_mult, "my_interface");
    // throw std::runtime_error("aborting");
    // std::cout << "dims:\n";
    // std::cout << "\tnx: "; for (auto e : info.dims.number_of_states) { std::cout << e << " ";}
    // std::cout << "\n\tnu: "; for (auto e : info.dims.number_of_controls) { std::cout << e << " ";}
    // std::cout << "\n\tng_eq: "; for (auto e : info.dims.number_of_eq_constraints) { std::cout << e << " ";}
    // std::cout << "\n\tng_ineq: "; for (auto e : info.dims.number_of_ineq_constraints) { std::cout << e << " ";}
    // std::cout << std::endl;
    MatRealView *RSQrq_hat_curr_p;
    Index rank_k;
    auto intermediate_start = std::chrono::high_resolution_clock::now();
    /////////////// recursion ///////////////
    for (Index k = info.dims.K - 1; k >= 0; --k)
    {
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offset_ineq_k = info.offsets_slack[k];
        const Index offset_u = info.offsets_primal_u[k];
        const Index offset_eq_path = info.offsets_g_eq_path[k];
        const Index offset_eq_slack = info.offsets_g_eq_slack[k];
        //////// SUBSDYN
        Index gamma_k;
        if (k == info.dims.K - 1)
        {
            gamma_k = ng;
            gamma[k] = gamma_k;
            rowin(ng, 1.0, g, offset_eq_path, jacobian.Gg_eqt[k], nu + nx, 0);
            gecp(nx + nu + 1, ng, jacobian.Gg_eqt[k], 0, 0, Ggt_stripe[0], 0, 0);
            rowin(nu + nx, 1.0, f, offset_u, hessian.RSQrqt[k], nu + nx, 0);
            gecp(nx + nu + 1, nu + nx, hessian.RSQrqt[k], 0, 0, RSQrqt_tilde[k], 0, 0);
        }
        else
        {
            const Index offset_eq_dyn = info.offsets_g_eq_dyn[k];
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index Hp1_size = gamma[k + 1] - rho[k + 1];
            if (Hp1_size + ng > nu + nx)
            return LinsolReturnFlag::NOFULL_RANK;
            gamma_k = Hp1_size + ng;
            // AL <- [BAb]^T_k P_kp1
            rowin(nxp1, 1.0, g, offset_eq_dyn, jacobian.BAbt[k], nu + nx, 0);
            gemm_nt(nu + nx + 1, nxp1, nxp1, 1.0, jacobian.BAbt[k], 0, 0, Ppt[k + 1], 0, 0, 0.0,
                    AL[0], 0, 0, AL[0], 0, 0);
            // AL[-1,:] <- AL[-1,:] + p_kp1^T
            gead(1, nxp1, 1.0, Ppt[k + 1], nxp1, 0, AL[0], nx + nu, 0);
            // RSQrqt_stripe <- AL[BA] + RSQrqt
            rowin(nu + nx, 1.0, f, offset_u, hessian.RSQrqt[k], nu + nx, 0);
            syrk_ln_mn(nu + nx + 1, nu + nx, nxp1, 1.0, AL[0], 0, 0, jacobian.BAbt[k], 0, 0, 1.0,
                       hessian.RSQrqt[k], 0, 0, RSQrqt_tilde[k], 0, 0);
            //// inequalities
            gamma[k] = gamma_k;
            // if ng[k]>0
            if (gamma_k > 0)
            {
                // if Gk nonempty
                if (ng > 0)
                {
                    // Ggt_stripe  <- Ggt_k
                    rowin(ng, 1.0, g, offset_eq_path, jacobian.Gg_eqt[k], nu + nx, 0);
                    gecp(nu + nx + 1, ng, jacobian.Gg_eqt[k], 0, 0, Ggt_stripe[0], 0, 0);
                }
                // if Hkp1 nonempty
                if (Hp1_size > 0)
                {
                    // Ggt_stripe <- [Ggt_k [BAb_k^T]H_kp1]
                    gemm_nt(nu + nx + 1, Hp1_size, nxp1, 1.0, jacobian.BAbt[k], 0, 0, Hh[k + 1], 0,
                            0, 0.0, Ggt_stripe[0], 0, ng, Ggt_stripe[0], 0, ng);
                    // Ggt_stripe[-1,ng:] <- Ggt_stripe[-1,ng:] + h_kp1^T
                    gead_transposed(1, Hp1_size, 1.0, Hh[k + 1], 0, nxp1, Ggt_stripe[0], nu + nx,
                                    ng);
                }
            }
            else
            {
                rho[k] = 0;
                rank_k = 0;
                RSQrq_hat_curr_p = &RSQrqt_tilde[k];
            }
        }
        // inequalities + inertia correction
        {
            if (ng_ineq > 0)
            {
                rowin(ng_ineq, 1.0, g, offset_eq_slack, jacobian.Gg_ineqt[k], nu + nx, 0);
                gecp(nu + nx + 1, ng_ineq, jacobian.Gg_ineqt[k], 0, 0, Ggt_ineq_temp[0], 0, 0);
                for (Index i = 0; i < ng_ineq; i++)
                {
                    Scalar scaling_factor = 1.0 / D_s(offset_ineq_k + i);
                    colsc(nu + nx + 1, scaling_factor, Ggt_ineq_temp[0], 0, i);
                }
                // add the penalty
                syrk_ln_mn(nu + nx + 1, nu + nx, ng_ineq, 1.0, Ggt_ineq_temp[0], 0, 0,
                           jacobian.Gg_ineqt[k], 0, 0, 1.0, RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k],
                           0, 0);
            }
            // inertia correction
            diaad(nu + nx, 1.0, D_x, offset_u, RSQrqt_tilde[k], 0, 0);
        }
        //////// TRANSFORM_AND_SUBSEQ
        {
            // symmetric transformation, done a little different than in paper, in order to fuse LA
            // operations LU_FACT_TRANSPOSE(Ggtstripe[:gamma_k, nu+nx+1], nu max) if(k==K-2)
            // blasfeo_print_dmat(1, gamma_k, Ggt_stripe[0], nu+nx, 0);
            auto start = std::chrono::steady_clock::now();
            // MatRealAllocated A_original(nu + nx + 1, gamma_k);
            // gecp(nu + nx + 1, gamma_k, Ggt_stripe[0], 0, 0, A_original, 0, 0);

            lu_fact_transposed(gamma_k, nu + nx + 1, nu, rank_k, Ggt_stripe[0], Pl[k], Pr[k],
                               lu_fact_tol);
            auto stop = std::chrono::steady_clock::now();
            duration_lu_factorization += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            // std::cout << duration_lu_factorization.count() << std::endl;

            // verify Tr^-T At Tl^-T
            /*
            MatRealAllocated At2(nu, gamma_k);
            gecp(nu, gamma_k, A_original, 0, 0, At2, 0, 0);

            // - apply Tl^-T
            Pl[k].apply_on_cols(rank_k, &At2.mat());
            blasfeo_dtrsm_runu(nu, rank_k, 1.0, &Ggt_stripe[0].mat(), 0, 0, 
                                &At2.mat(), 0, 0, &At2.mat(), 0, 0); // L1^-T to first rank_k cols
            blasfeo_dgemm_nn(nu, gamma_k-rank_k, rank_k, -1.0, &At2.mat(), 0, 0, 
                                &Ggt_stripe[0].mat(), 0, rank_k, 1.0, 
                                &At2.mat(), 0, rank_k, &At2.mat(), 0, rank_k); // - L2^-T * L1^-T to last rank_k cols
            blasfeo_dtrsm_rlnn(nu, rank_k, -1.0, &Ggt_stripe[0].mat(), 0, 0, &At2.mat(), 0, 0, &At2.mat(), 0, 0); // *U1^-T to first rank_k cols
            // - apply Tr^-T
            Pr[k].apply_on_rows(rank_k, &At2.mat());
            MatRealAllocated U1U2t(nu - rank_k, rank_k);
            gecp(nu - rank_k, rank_k, At2, rank_k, 0, U1U2t, 0, 0); // U2t in there
            if (nu - rank_k > 0){
            trsm_rlnn(nu-rank_k, rank_k, -1.0, At2, 0, 0, U1U2t, 0, 0, U1U2t, 0, 0); // U1^-T * U2t in there
            gemm_nn(nu - rank_k, gamma_k, rank_k, 1.0, U1U2t, 0, 0, At2, 0, 0, 1.0,
                    At2, 0, rank_k, At2, 0, rank_k); // - U2 * U1^-T * U2t in there
            }
            if (k == 0){
                std::cout << "At after applying Tl^-T and Tr^-T:\n"; blasfeo_print_dmat(nu, gamma_k, &At2.mat(), 0, 0);
            }
            */

            rho[k] = rank_k;
            if (gamma_k - rank_k > 0)
            {
                // transfer eq's to next stage
                if (gamma_k - rank_k > nx)
                    return LinsolReturnFlag::NOFULL_RANK;
                getr(nx + 1, gamma_k - rank_k, Ggt_stripe[0], nu, rank_k, Hh[k], 0, 0);
            }
            if (rank_k > 0)
            {
                // Ggt_tilde_k <- Ggt_stripe[rho_k:nu+nx+1, :rho] L-T (note that this is slightly
                // different from the implementation)
                trsm_rlnn(nu - rank_k + nx + 1, rank_k, -1.0, Ggt_stripe[0], 0, 0, Ggt_stripe[0],
                          rank_k, 0, Ggt_tilde[k], 0, 0);
                // the following command copies the top block matrix (LU) to the bottom because it
                // it needed later
                gecp(rank_k, gamma_k, Ggt_stripe[0], 0, 0, Ggt_tilde[k], nu - rank_k + nx + 1, 0);
                // permutations
                trtr_l(nu + nx, RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k], 0,
                       0); // copy lower part of RSQ to upper part
                Pr[k].apply_on_rows(rank_k, &RSQrqt_tilde[k].mat()); // TODO make use of symmetry
                Pr[k].apply_on_cols(rank_k, &RSQrqt_tilde[k].mat());
                // GL <- Ggt_tilde_k @ RSQ[:rho,:nu+nx] + RSQrqt[rho:nu+nx+1, rho:] (with
                // RSQ[:rho,:nu+nx] = RSQrqt[:nu+nx,:rho]^T) GEMM_NT(nu - rank_k + nx + 1, nu + nx,
                // rank_k, 1.0, Ggt_tilde[k], 0, 0, RSQrqt_tilde[k], 0, 0, 1.0, RSQrqt_tilde_p
                // + k, rank_k, 0, GgLt[0], 0, 0); split up because valgrind was giving invalid read
                // errors when C matrix has nonzero row offset GgLt[0].print();
                gecp(nu - rank_k + nx + 1, nu + nx, RSQrqt_tilde[k], rank_k, 0, GgLt[0], 0, 0);
                gemm_nt(nu - rank_k + nx + 1, nu + nx, rank_k, 1.0, Ggt_tilde[k], 0, 0,
                        RSQrqt_tilde[k], 0, 0, 1.0, GgLt[0], 0, 0, GgLt[0], 0, 0);
                // RSQrqt_hat = GgLt[nu-rank_k + nx +1, :rank_k] * G[:rank_k, :nu+nx] +
                // GgLt[rank_k:, :]  (with G[:rank_k,:nu+nx] = Gt[:nu+nx,:rank_k]^T)
                syrk_ln_mn(nu - rank_k + nx + 1, nu + nx - rank_k, rank_k, 1.0, GgLt[0], 0, 0,
                           Ggt_tilde[k], 0, 0, 1.0, GgLt[0], 0, rank_k, RSQrqt_hat[0], 0, 0);
                // GEMM_NT(nu - rank_k + nx + 1, nu + nx - rank_k, rank_k, 1.0, GgLt[0], 0, 0,
                // Ggt_tilde[k], 0, 0, 1.0, GgLt[0], 0, rank_k, RSQrqt_hat[0], 0, 0);
                RSQrq_hat_curr_p = &RSQrqt_hat[0];
            }
            else
            {
                RSQrq_hat_curr_p = &RSQrqt_tilde[k];
            }
        }
        //////// SCHUR
        {
            if (nu - rank_k > 0)
            {
                // DLlt_k = [chol(R_hatk); Llk@chol(R_hatk)^-T]
                potrf_l_mn(nu - rank_k + nx + 1, nu - rank_k, RSQrq_hat_curr_p[0], 0, 0, Llt[k], 0,
                           0);
                if (!check_reg(nu - rank_k, &Llt[k].mat(), 0, 0))
                    return LinsolReturnFlag::INDEFINITE;
                // Pp_k = Qq_hatk - L_k^T @ Ll_k
                // SYRK_LN_MN(nx+1, nx, nu-rank_k, -1.0,Llt_p+k, nu-rank_k,0, Llt_p+k,
                // nu-rank_k,0, 1.0, RSQrq_hat_curr[0], nu-rank_k, nu-rank_k,Pp+k,0,0); // feature
                // not implmented yet
                gecp(nx + 1, nu - rank_k, Llt[k], nu - rank_k, 0, Llt_shift[0], 0,
                     0); // needless operation because feature not implemented yet
                // SYRK_LN_MN(nx + 1, nx, nu - rank_k, -1.0, Llt_shift[0], 0, 0, Llt_shift[0], 0,
                // 0, 1.0, RSQrq_hat_curr[0], nu - rank_k, nu - rank_k, Ppt[k], 0, 0);
                gecp(nx + 1, nx, RSQrq_hat_curr_p[0], nu - rank_k, nu - rank_k, Ppt[k], 0, 0);
                syrk_ln_mn(nx + 1, nx, nu - rank_k, -1.0, Llt_shift[0], 0, 0, Llt_shift[0], 0, 0,
                           1.0, Ppt[k], 0, 0, Ppt[k], 0, 0);
                // next steps are for better accuracy
                if (increased_accuracy)
                {
                    // copy eta
                    getr(nu - rank_k, gamma_k - rank_k, Ggt_stripe[0], rank_k, rank_k,
                         Ggt_stripe[0], 0, 0);
                    // blasfeo_print_dmat(gamma_k-rank_k, nu-rank_k, Ggt_stripe[0], 0,0);
                    // eta L^-T
                    trsm_rltn(gamma_k - rank_k, nu - rank_k, 1.0, Llt[k], 0, 0, Ggt_stripe[0], 0, 0,
                              Ggt_stripe[0], 0, 0);
                    // ([S^T \\ r^T] L^-T) @ (L^-1 eta^T)
                    // (eta L^-T) @ ([S^T \\ r^T] L^-T)^T
                    gemm_nt(gamma_k - rank_k, nx + 1, nu - rank_k, -1.0, Ggt_stripe[0], 0, 0,
                            Llt[k], nu - rank_k, 0, 1.0, Hh[k], 0, 0, Hh[k], 0, 0);
                    // keep (L^-1 eta^T) for forward recursion
                    getr(gamma_k - rank_k, nu - rank_k, Ggt_stripe[0], 0, 0, Ggt_tilde[k], 0,
                         rank_k);
                }
            }
            else
            {
                gecp(nx + 1, nx, RSQrq_hat_curr_p[0], 0, 0, Ppt[k], 0, 0);
            }
            trtr_l(nx, Ppt[k], 0, 0, Ppt[k], 0, 0);
        }
    }
    auto intermediate_stop = std::chrono::high_resolution_clock::now();
    duration_backward_recursion += std::chrono::duration_cast<std::chrono::nanoseconds>(intermediate_stop - intermediate_start);
    intermediate_start = std::chrono::high_resolution_clock::now();

    rankI = 0;
    //////// FIRST_STAGE
    {
        const Index nx = info.dims.number_of_states[0];
        Index gamma_I = gamma[0] - rho[0];
        if (gamma_I > nx)
        {
            return LinsolReturnFlag::NOFULL_RANK;
        }
        if (gamma_I > 0)
        {
            getr(gamma_I, nx + 1, Hh[0], 0, 0, HhIt[0], 0, 0); // transposition may be avoided
            // HhIt[0].print();
            lu_fact_transposed(gamma_I, nx + 1, nx, rankI, HhIt[0], PlI[0], PrI[0], lu_fact_tol);
            if (rankI < gamma_I)
                return LinsolReturnFlag::NOFULL_RANK;
            // PpIt_tilde <- Ggt[rankI:nx+1, :rankI] L-T (note that this is slightly different from
            // the implementation)
            trsm_rlnn(nx - rankI + 1, rankI, -1.0, HhIt[0], 0, 0, HhIt[0], rankI, 0, GgIt_tilde[0],
                      0, 0);
            // permutations
            PrI[0].apply_on_rows(rankI, &Ppt[0].mat()); // TODO make use of symmetry
            PrI[0].apply_on_cols(rankI, &Ppt[0].mat());
            // // GL <- GgIt_tilde @ Pp[:rankI,:nx] + Ppt[rankI:nx+1, rankI:] (with Pp[:rankI,:nx] =
            // Ppt[:nx,:rankI]^T) GEMM_NT(nx - rankI + 1, nx, rankI, 1.0, GgIt_tilde[0], 0, 0,
            // Ppt[0], 0, 0, 1.0, Ppt[0], rankI, 0, GgLIt[0], 0, 0); split up because valgrind was
            // giving invalid read errors when C matrix has nonzero row offset
            gecp(nx - rankI + 1, nx, Ppt[0], rankI, 0, GgLIt[0], 0, 0);
            gemm_nt(nx - rankI + 1, nx, rankI, 1.0, GgIt_tilde[0], 0, 0, Ppt[0], 0, 0, 1.0,
                    GgLIt[0], 0, 0, GgLIt[0], 0, 0);
            // // RSQrqt_hat = GgLt[nu-rank_k + nx +1, :rank_k] * G[:rank_k, :nu+nx] + GgLt[rank_k:,
            // :]  (with G[:rank_k,:nu+nx] = Gt[:nu+nx,:rank_k]^T)
            syrk_ln_mn(nx - rankI + 1, nx - rankI, rankI, 1.0, GgLIt[0], 0, 0, GgIt_tilde[0], 0, 0,
                       1.0, GgLIt[0], 0, rankI, PpIt_hat[0], 0, 0);
            // TODO skipped if nx-rankI = 0
            potrf_l_mn(nx - rankI + 1, nx - rankI, PpIt_hat[0], 0, 0, LlIt[0], 0, 0);
            if (!check_reg(nx - rankI, &LlIt[0].mat(), 0, 0))
                return LinsolReturnFlag::INDEFINITE;
        }
        else
        {
            rankI = 0;
            potrf_l_mn(nx + 1, nx, Ppt[0], 0, 0, LlIt[0], 0, 0);
            if (!check_reg(nx, &LlIt[0].mat(), 0, 0))
                return LinsolReturnFlag::INDEFINITE;
        }
    }
    ////// FORWARD_SUBSTITUTION:
    // first stage
    {
        const Index nx = info.dims.number_of_states[0];
        const Index nu = info.dims.number_of_controls[0];
        const Index offs_u = info.offsets_primal_u[0];
        const Index offs_x = info.offsets_primal_x[0];
        const Index offs_g = info.offsets_g_eq_path[0];
        // calculate xIb
        rowex(nx - rankI, -1.0, LlIt[0], nx - rankI, 0, x, offs_x + rankI);
        // assume TRSV_LTN allows aliasing, this is the case in normal BLAS
        trsv_ltn(nx - rankI, LlIt[0], 0, 0, x, offs_x + rankI, x, offs_x + rankI);
        // calculate xIa
        rowex(rankI, 1.0, GgIt_tilde[0], nx - rankI, 0, x, offs_x);
        // assume aliasing is possible for last two elements
        gemv_t(nx - rankI, rankI, 1.0, GgIt_tilde[0], 0, 0, x, offs_x + rankI, 1.0, x, offs_x, x,
               offs_x);
        //// lag
        rowex(rankI, -1.0, Ppt[0], nx, 0, eq_mult, offs_g);
        // assume aliasing is possible for last two elements
        gemv_t(nx, rankI, -1.0, Ppt[0], 0, 0, x, offs_x, 1.0, eq_mult, offs_g, eq_mult, offs_g);

        // U^-T
        trsv_lnn(rankI, HhIt[0], 0, 0, eq_mult, offs_g, eq_mult, offs_g);
        // L^-T
        trsv_unu(rankI, rankI, HhIt[0], 0, 0, eq_mult, offs_g, eq_mult, offs_g);
        PlI[0].apply_inverse(rankI, &eq_mult.vec(), offs_g);
        PrI[0].apply_inverse(rankI, &x.vec(), offs_x);
    }
    intermediate_stop = std::chrono::high_resolution_clock::now();
    duration_initial_stage += std::chrono::duration_cast<std::chrono::nanoseconds>(intermediate_stop - intermediate_start);
    intermediate_start = std::chrono::high_resolution_clock::now();
    // std::cout << "------------------------------" << std::endl;
    // std::cout << "First stage:\n";
    // PrintNpArray(Ppt[0], "\tPpt");
    // PrintNpArray(Hh[0], "\tHh");
    // std::cout << "Solution to first stage:\n";
    // std::cout << "\tx0: " << x.block(info.dims.number_of_states[0], info.offsets_primal_x[0])<< std::endl;
    // std::cout << "\tu0: " << x.block(info.dims.number_of_controls[0], info.offsets_primal_u[0])<< std::endl;
    // std::cout << "\tnu0: " << eq_mult.block(gamma[0] - rho[0], info.offsets_g_eq_path[0])<< std::endl; 
    // std::cout << "------------------------------" << std::endl;
    // other stages
    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_x = info.offsets_primal_x[k];
        const Index rho_k = rho[k];
        const Index numrho_k = nu - rho_k;
        const Index offs_g_k = info.offsets_g_eq_path[k];
        const Index gammamrho_k = gamma[k] - rho[k];
        const Index gamma_k = gamma[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_eq_ineq = info.offsets_g_eq_slack[k];
        const Index offs_slack = info.offsets_slack[k];
        if (numrho_k > 0)
        {
            /// calculate ukb_tilde
            // -Lkxk - lk
            rowex(numrho_k, -1.0, Llt[k], numrho_k + nx, 0, x, offs + rho_k);
            if (increased_accuracy)
            {
                gemv_n(nu - rho_k, gamma_k - rho_k, -1.0, Ggt_tilde[k], 0, rho_k, eq_mult, offs_g_k,
                       1.0, x, offs + rho_k, x, offs + rho_k);
            }
            // assume aliasing of last two eliments is allowed
            gemv_t(nx, numrho_k, -1.0, Llt[k], numrho_k, 0, x, offs_x, 1.0, x, offs + rho_k, x,
                   offs + rho_k);
            trsv_ltn(numrho_k, Llt[k], 0, 0, x, offs + rho_k, x, offs + rho_k);
        }
        /// calcualate uka_tilde
        if (rho_k > 0)
        {
            rowex(rho_k, 1.0, Ggt_tilde[k], numrho_k + nx, 0, x, offs);
            gemv_t(nx + numrho_k, rho_k, 1.0, Ggt_tilde[k], 0, 0, x, offs + rho_k, 1.0, x, offs, x,
                   offs);
            // calculate lamda_tilde_k
            // copy vk to right location
            veccp(gammamrho_k, eq_mult, offs_g_k, v_tmp[0], 0);
            veccp(gammamrho_k, v_tmp[0], 0, eq_mult, offs_g_k + rho_k);
            rowex(rho_k, -1.0, RSQrqt_tilde[k], nu + nx, 0, eq_mult, offs_g_k);
            // assume aliasing of last two eliments is allowed
            gemv_t(nu + nx, rho_k, -1.0, RSQrqt_tilde[k], 0, 0, x, offs, 1.0, eq_mult, offs_g_k,
                   eq_mult, offs_g_k);
            // nu-rank_k+nx,0
            // needless copy because feature not implemented yet in trsv_lnn
            gecp(rho_k, gamma_k, Ggt_tilde[k], nu - rho_k + nx + 1, 0, AL[0], 0, 0);
            // U^-T
            trsv_lnn(rho_k, AL[0], 0, 0, eq_mult, offs_g_k, eq_mult, offs_g_k);
            // L^-T
            trsv_unu(rho_k, gamma_k, AL[0], 0, 0, eq_mult, offs_g_k, eq_mult, offs_g_k);
            Pl[k].apply_inverse(rho_k, &eq_mult.vec(), offs_g_k);
            Pr[k].apply_inverse(rho_k, &x.vec(), offs);
        }
        if (ng_ineq > 0)
        {
            gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_eq_ineq,
                   eq_mult, offs_eq_ineq);
            eq_mult.block(ng_ineq, offs_eq_ineq) =
                eq_mult.block(ng_ineq, offs_eq_ineq) / D_s.block(ng_ineq, offs_slack);
        }
        if (k != info.dims.K - 1)
        {
            const Index offs_dyn_eq_k = info.offsets_g_eq_dyn[k];
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index nup1 = info.dims.number_of_controls[k + 1];
            const Index offsp1 = info.offsets_primal_u[k + 1];
            const Index offsxp1 = info.offsets_primal_x[k + 1];
            const Index offs_g_kp1 = info.offsets_g_eq_path[k + 1];
            const Index gammamrho_kp1 = gamma[k + 1] - rho[k + 1];
            // calculate xkp1
            rowex(nxp1, 1.0, jacobian.BAbt[k], nu + nx, 0, x, offsxp1);
            gemv_t(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, x, offs, 1.0, x, offsxp1, x,
                   offsxp1);
            // calculate lam_dyn xp1
            rowex(nxp1, 1.0, Ppt[k + 1], nxp1, 0, eq_mult, offs_dyn_eq_k);
            gemv_t(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, x, offsxp1, 1.0, eq_mult, offs_dyn_eq_k,
                   eq_mult, offs_dyn_eq_k);
            gemv_t(gammamrho_kp1, nxp1, 1.0, Hh[k + 1], 0, 0, eq_mult, offs_g_kp1, 1.0, eq_mult,
                   offs_dyn_eq_k, eq_mult, offs_dyn_eq_k);
        }
    }
    intermediate_stop = std::chrono::high_resolution_clock::now();
    duration_forward_recursion += std::chrono::duration_cast<std::chrono::nanoseconds>(intermediate_stop - intermediate_start);

    // save solution to a file
    // WriteJacobianHessianToFile(info, jacobian, hessian, f, g, x, eq_mult, "my_interface");
    // throw std::runtime_error("aborting");

    return LinsolReturnFlag::SUCCESS;
}
LinsolReturnFlag AugSystemSolver<OcpType>::solve(const ProblemInfo &info,
                                           Jacobian<OcpType> &jacobian, Hessian<OcpType> &hessian,
                                           const VecRealView &D_x, const VecRealView &D_eq,
                                           const VecRealView &D_s, const VecRealView &f,
                                           const VecRealView &g, VecRealView &x,
                                           VecRealView &eq_mult)
{
    MatRealView *RSQrq_hat_curr_p;
    for (Index k = info.dims.K - 1; k >= 0; --k)
    {
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_ineq_k = info.offsets_slack[k];
        const Index offset_u = info.offsets_primal_u[k];
        const Index offset_eq_k = info.offsets_eq[k];
        const Index offset_g_eq_k = info.offsets_g_eq_path[k];
        const Index offset_g_ineq_k = info.offsets_g_eq_slack[k];
        // const fatrop_int offs_g_ineq_k = offs_g_ineq_p[k];
        //////// SUBSDYN
        if (k == info.dims.K - 1)
        {
            rowin(nu + nx, 1.0, f, offset_u, hessian.RSQrqt[k], nu + nx, 0);
            gecp(nx + nu + 1, nu + nx, hessian.RSQrqt[k], 0, 0, RSQrqt_tilde[k], 0, 0);
        }
        else
        {
            const Index offset_eq_dyn = info.offsets_g_eq_dyn[k];
            const Index nxp1 = info.dims.number_of_states[k + 1];
            // AL <- [BAb]^T_k P_kp1
            rowin(nxp1, 1.0, g, offset_eq_dyn, jacobian.BAbt[k], nu + nx, 0);
            gemm_nt(nu + nx + 1, nxp1, nxp1, 1.0, jacobian.BAbt[k], 0, 0, Ppt[k + 1], 0, 0, 0.0,
                    AL[0], 0, 0, AL[0], 0, 0);
            // AL[-1,:] <- AL[-1,:] + p_kp1^T
            gead(1, nxp1, 1.0, Ppt[k + 1], nxp1, 0, AL[0], nx + nu, 0);
            // RSQrqt_stripe <- AL[BA] + RSQrqt
            rowin(nu + nx, 1.0, f, offset_u, hessian.RSQrqt[k], nu + nx, 0);
            syrk_ln_mn(nu + nx + 1, nu + nx, nxp1, 1.0, AL[0], 0, 0, jacobian.BAbt[k], 0, 0, 1.0,
                       hessian.RSQrqt[k], 0, 0, RSQrqt_tilde[k], 0, 0);
        }
        // equality penalty
        {
            rowin(ng, 1.0, g, offset_g_eq_k, jacobian.Gg_eqt[k], nu + nx, 0);
            gecp(nu + nx + 1, ng, jacobian.Gg_eqt[k], 0, 0, Ggt_stripe[0], 0, 0);
            for (Index i = 0; i < ng; i++)
            {
                Scalar scaling_factor = 1.0 / D_eq(offset_eq_k + i);
                colsc(nu + nx + 1, scaling_factor, Ggt_stripe[0], 0, i);
            }
            // add the penalty
            syrk_ln_mn(nu + nx + 1, nu + nx, ng, 1.0, Ggt_stripe[0], 0, 0, jacobian.Gg_eqt[k], 0, 0,
                       1.0, RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k], 0, 0);
        }
        // inequalities + inertia correction
        {
            if (ng_ineq > 0)
            {
                rowin(ng_ineq, 1.0, g, offset_g_ineq_k, jacobian.Gg_ineqt[k], nu + nx, 0);
                gecp(nu + nx + 1, ng_ineq, jacobian.Gg_ineqt[k], 0, 0, Ggt_ineq_temp[0], 0, 0);
                for (Index i = 0; i < ng_ineq; i++)
                {
                    Scalar scaling_factor = 1.0 / D_s(offs_ineq_k + i);
                    colsc(nu + nx + 1, scaling_factor, Ggt_ineq_temp[0], 0, i);
                }
                // add the penalty
                syrk_ln_mn(nu + nx + 1, nu + nx, ng_ineq, 1.0, Ggt_ineq_temp[0], 0, 0,
                           jacobian.Gg_ineqt[k], 0, 0, 1.0, RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k],
                           0, 0);
            }
            // inertia correction
            diaad(nu + nx, 1.0, D_x, offset_u, RSQrqt_tilde[k], 0, 0);
        }

        //////// TRANSFORM_AND_SUBSEQ
        {
            RSQrq_hat_curr_p = &RSQrqt_tilde[k];
        }
        //////// SCHUR
        {
            // DLlt_k = [chol(R_hatk); Llk@chol(R_hatk)^-T]
            potrf_l_mn(nu + nx + 1, nu, *RSQrq_hat_curr_p, 0, 0, Llt[k], 0, 0);
            if (!check_reg(nu, &Llt[k].mat(), 0, 0))
                return LinsolReturnFlag::INDEFINITE;
            // Pp_k = Qq_hatk - L_k^T @ Ll_k
            // SYRK_LN_MN(nx+1, nx, nu-rank_k, -1.0,Llt_p+k, nu-rank_k,0, Llt_p+k, nu-rank_k,0, 1.0,
            // RSQrq_hat_curr_p, nu-rank_k, nu-rank_k,Pp+k,0,0); // feature not implmented yet
            gecp(nx + 1, nu, Llt[k], nu, 0, Llt_shift[0], 0,
                 0); // needless operation because feature not implemented yet
            syrk_ln_mn(nx + 1, nx, nu, -1.0, Llt_shift[0], 0, 0, Llt_shift[0], 0, 0, 1.0,
                       *RSQrq_hat_curr_p, nu, nu, Ppt[k], 0, 0);
        }
        trtr_l(nx, Ppt[k], 0, 0, Ppt[k], 0, 0);
    }
    //////// FIRST_STAGE
    {
        const Index nx = info.dims.number_of_states[0];
        {
            potrf_l_mn(nx + 1, nx, Ppt[0], 0, 0, LlIt[0], 0, 0);
            if (!check_reg(nx, &LlIt[0].mat(), 0, 0))
                return LinsolReturnFlag::INDEFINITE;
        }
    }
    ////// FORWARD_SUBSTITUTION:
    // first stage
    {
        const Index nx = info.dims.number_of_states[0];
        const Index nu = info.dims.number_of_controls[0];
        const Index offs_x = info.offsets_primal_x[0];
        // calculate xIb
        rowex(nx, -1.0, LlIt[0], nx, 0, x, offs_x);
        // assume TRSV_LTN allows aliasing, this is the case in normal BLAS
        trsv_ltn(nx, LlIt[0], 0, 0, x, offs_x, x, offs_x);
    }
    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_x = info.offsets_primal_x[k];
        rowex(nu, -1.0, Llt[k], nu + nx, 0, x, offs);
        gemv_t(nx, nu, -1.0, Llt[k], nu, 0, x, offs_x, 1.0, x, offs, x, offs);
        trsv_ltn(nu, Llt[k], 0, 0, x, offs, x, offs);
        if (k != info.dims.K - 1)
        {
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index nup1 = info.dims.number_of_controls[k + 1];
            const Index offs_x_p1 = info.offsets_primal_x[k + 1];
            const Index offs_dyn_eq_k = info.offsets_g_eq_dyn[k];
            // calculate xkp1
            rowex(nxp1, 1.0, jacobian.BAbt[k], nu + nx, 0, x, offs_x_p1);
            gemv_t(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, x, offs, 1.0, x, offs_x_p1, x,
                   offs_x_p1);
            // calculate lam_dyn xp1
            rowex(nxp1, 1.0, Ppt[k + 1], nxp1, 0, eq_mult, offs_dyn_eq_k);
            gemv_t(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, x, offs_x_p1, 1.0, eq_mult, offs_dyn_eq_k,
                   eq_mult, offs_dyn_eq_k);
        }
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index offs_g_eq_k = info.offsets_g_eq_path[k];
        const Index offs_eq_k = info.offsets_eq[k];
        if (ng > 0)
        {
            gemv_t(nu + nx, ng, 1.0, jacobian.Gg_eqt[k], 0, 0, x, offs, 1.0, g, offs_g_eq_k,
                   eq_mult, offs_g_eq_k);
            eq_mult.block(ng, offs_g_eq_k) =
                eq_mult.block(ng, offs_g_eq_k) / D_eq.block(ng, offs_eq_k);
        }
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_slack = info.offsets_slack[k];
        const Index offs_eq_ineq = info.offsets_g_eq_slack[k];
        if (ng_ineq > 0)
        {
            gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_eq_ineq,
                   eq_mult, offs_eq_ineq);
            eq_mult.block(ng_ineq, offs_eq_ineq) =
                eq_mult.block(ng_ineq, offs_eq_ineq) / D_s.block(ng_ineq, offs_slack);
        }
    }
    return LinsolReturnFlag::SUCCESS;
}

LinsolReturnFlag AugSystemSolver<OcpType>::solve_rhs(const ProblemInfo &info,
                                               const Jacobian<OcpType> &jacobian,
                                               const Hessian<OcpType> &hessian,
                                               const VecRealView &D_s, const VecRealView &f,
                                               const VecRealView &g, VecRealView &x,
                                               VecRealView &eq_mult)
{
    VecRealView *v_RSQrq_hat_curr_p;
    Index rank_k;
    /////////////// recursion ///////////////

    for (Index k = info.dims.K - 1; k >= 0; --k)
    {
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offset_ineq_k = info.offsets_slack[k];
        const Index offs_g_ineq_k = info.offsets_g_eq_slack[k];
        const Index offs_g_k = info.offsets_g_eq_path[k];
        const Index offs = info.offsets_primal_u[k];
        //         //////// SUBSDYN
        Index gamma_k;
        if (k == info.dims.K - 1)
        {
            gamma_k = ng;
            gamma[k] = gamma_k;
            veccp(ng, g, offs_g_k, v_Ggt_stripe[0], 0);
            veccp(nu + nx, f, offs, v_RSQrqt_tilde[k], 0);
        }
        else
        {
            const Index offs_dyn_k = info.offsets_g_eq_dyn[k];
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index Hp1_size = gamma[k + 1] - rho[k + 1];
            gamma_k = Hp1_size + ng;
            gemv_n(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, g, offs_dyn_k, 0.0, v_AL[0], 0, v_AL[0], 0);
            axpy(nxp1, 1.0, v_Ppt[k + 1], 0, v_AL[0], 0, v_AL[0], 0);
            gemv_n(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, v_AL[0], 0, 1.0, f, offs,
                   v_RSQrqt_tilde[k], 0);
            if (gamma_k > 0)
            {
                if (ng > 0)
                {
                    veccp(ng, g, offs_g_k, v_Ggt_stripe[0], 0);
                }
                if (Hp1_size > 0)
                {
                    gemv_n(Hp1_size, nxp1, 1.0, Hh[k + 1], 0, 0, g, offs_dyn_k, 0.0,
                           v_Ggt_stripe[0], ng, v_Ggt_stripe[0], ng);
                    axpy(Hp1_size, 1.0, v_Hh[k + 1], 0, v_Ggt_stripe[0], ng, v_Ggt_stripe[0], ng);
                }
            }
            else
            {
                rank_k = 0;
                v_RSQrq_hat_curr_p = &v_RSQrqt_tilde[k];
            }
        }
        if (ng_ineq > 0)
        {
            for (Index i = 0; i < ng_ineq; i++)
            {
                Scalar scaling_factor = D_s(offset_ineq_k + i);
                Scalar grad_barrier = g(offs_g_ineq_k + i);
                v_Ggt_ineq_temp[0](i) = grad_barrier / scaling_factor;
            }
            gemv_n(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, v_Ggt_ineq_temp[0], 0, 1.0,
                   v_RSQrqt_tilde[k], 0, v_RSQrqt_tilde[k], 0);
        }
        {
            rank_k = rho[k];
            gecp(rank_k, gamma_k, Ggt_tilde[k], nu - rank_k + nx + 1, 0, Ggt_stripe[0], 0, 0);
            Pl[k].apply(rank_k, &v_Ggt_stripe[0].vec(), 0);
            trsv_utu(rank_k, Ggt_stripe[0], 0, 0, v_Ggt_stripe[0], 0, v_Ggt_stripe[0], 0);
            gemv_t(rank_k, gamma_k - rank_k, -1.0, Ggt_stripe[0], 0, rank_k, v_Ggt_stripe[0], 0,
                   1.0, v_Ggt_stripe[0], rank_k, v_Ggt_stripe[0], rank_k);

            if (gamma_k - rank_k > 0)
            {
                veccp(gamma_k - rank_k, v_Ggt_stripe[0], rank_k, v_Hh[k], 0);
            }
            if (rank_k > 0)
            {
                veccpsc(rank_k, -1.0, v_Ggt_stripe[0], 0, v_Ggt_tilde[k], 0);
                trsv_ltn(rank_k, Ggt_stripe[0], 0, 0, v_Ggt_tilde[k], 0, v_Ggt_tilde[k], 0);
                Pr[k].apply(rank_k, &v_RSQrqt_tilde[k].vec(), 0);
                veccp(nu + nx, v_RSQrqt_tilde[k], 0, v_GgLt[0], 0);
                gemv_n(nu + nx, rank_k, 1.0, RSQrqt_tilde[k], 0, 0, v_Ggt_tilde[k], 0, 1.0,
                       v_GgLt[0], 0, v_GgLt[0], 0);
                gemv_n(nu + nx - rank_k, rank_k, 1.0, Ggt_tilde[k], 0, 0, v_GgLt[0], 0, 1.0,
                       v_GgLt[0], rank_k, v_RSQrqt_hat[0], 0);
                v_RSQrq_hat_curr_p = &v_RSQrqt_hat[0];
            }
            else
            {
                v_RSQrq_hat_curr_p = &v_RSQrqt_tilde[k];
            }
        }
        //         //////// SCHUR
        {
            if (nu - rank_k > 0)
            {
                trsv_lnn(nu - rank_k, Llt[k], 0, 0, *v_RSQrq_hat_curr_p, 0, v_Llt[k], 0);
                gecp(nx + 1, nu - rank_k, Llt[k], nu - rank_k, 0, Llt_shift[0], 0, 0);
                veccp(nu - rank_k, v_Llt[k], 0, v_Llt_shift[0], 0);
                veccp(nx, *v_RSQrq_hat_curr_p, nu - rank_k, v_Ppt[k], 0);
                gemv_n(nx, nu - rank_k, -1.0, Llt_shift[0], 0, 0, v_Llt_shift[0], 0, 1.0, v_Ppt[k],
                       0, v_Ppt[k], 0);
                if (increased_accuracy)
                {
                    gemv_t(nu - rank_k, gamma_k - rank_k, -1.0, Ggt_tilde[k], 0, rank_k, v_Llt[k],
                           0, 1.0, v_Hh[k], 0, v_Hh[k], 0);
                }
            }
            else
            {
                veccp(nx, *v_RSQrq_hat_curr_p, 0, v_Ppt[k], 0);
            }
        }
    }
    {
        const Index nx = info.dims.number_of_states[0];
        Index gamma_I = gamma[0] - rho[0];
        if (gamma_I > 0)
        {
            veccp(gamma_I, v_Hh[0], 0, v_HhIt[0], 0);
            PlI[0].apply(rankI, &v_HhIt[0].vec(), 0);
            trsv_utu(rankI, HhIt[0], 0, 0, v_HhIt[0], 0, v_HhIt[0], 0);
            gemv_t(rankI, gamma_I - rankI, -1.0, HhIt[0], 0, rankI, v_HhIt[0], 0, 1.0, v_HhIt[0],
                   rankI, v_HhIt[0], rankI);
            veccpsc(rankI, -1.0, v_HhIt[0], 0, v_GgIt_tilde[0], 0);
            trsv_ltn(rankI, HhIt[0], 0, 0, v_GgIt_tilde[0], 0, v_GgIt_tilde[0], 0);
            PrI[0].apply(rankI, &v_Ppt[0].vec(), 0);
            veccp(nx, v_Ppt[0], 0, v_GgLIt[0], 0);
            gemv_n(nx, rankI, 1.0, Ppt[0], 0, 0, v_GgIt_tilde[0], 0, 1.0, v_GgLIt[0], 0, v_GgLIt[0],
                   0);
            gemv_n(nx - rankI, rankI, 1.0, GgIt_tilde[0], 0, 0, v_GgLIt[0], 0, 1.0, v_GgLIt[0],
                   rankI, v_PpIt_hat[0], 0);
            trsv_lnn(nx - rankI, LlIt[0], 0, 0, v_PpIt_hat[0], 0, v_LlIt[0], 0);
        }
        else
        {
            trsv_lnn(nx, LlIt[0], 0, 0, v_Ppt[0], 0, v_LlIt[0], 0);
        }
    }
    {
        const Index nx = info.dims.number_of_states[0];
        const Index nu = info.dims.number_of_controls[0];
        const Index offs_u = info.offsets_primal_u[0];
        const Index offs_x = info.offsets_primal_x[0];
        const Index offs_g = info.offsets_g_eq_path[0];
        veccpsc(nx - rankI, -1.0, v_LlIt[0], 0, x, offs_x + rankI);
        trsv_ltn(nx - rankI, LlIt[0], 0, 0, x, offs_x + rankI, x, offs_x + rankI);
        veccp(rankI, v_GgIt_tilde[0], 0, x, offs_x);
        gemv_t(nx - rankI, rankI, 1.0, GgIt_tilde[0], 0, 0, x, offs_x + rankI, 1.0, x, offs_x, x,
               offs_x);
        veccpsc(rankI, -1.0, v_Ppt[0], 0, eq_mult, offs_g);
        gemv_t(nx, rankI, -1.0, Ppt[0], 0, 0, x, nu, 1.0, eq_mult, offs_g, eq_mult, offs_g);
        trsv_lnn(rankI, HhIt[0], 0, 0, eq_mult, offs_g, eq_mult, offs_g);
        trsv_unu(rankI, rankI, HhIt[0], 0, 0, eq_mult, offs_g, eq_mult, offs_g);
        PlI[0].apply_inverse(rankI, &eq_mult.vec(), offs_g);
        PrI[0].apply_inverse(rankI, &x.vec(), offs_x);
    }
    for (Index k = 0; k < info.dims.K; k++)
    {

        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_x = info.offsets_primal_x[k];
        const Index rho_k = rho[k];
        const Index numrho_k = nu - rho_k;
        const Index offs_g_k = info.offsets_g_eq_path[k];
        const Index gammamrho_k = gamma[k] - rho[k];
        const Index gamma_k = gamma[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_eq_ineq = info.offsets_g_eq_slack[k];
        const Index offs_slack = info.offsets_slack[k];
        if (numrho_k > 0)
        {
            veccpsc(numrho_k, -1.0, v_Llt[k], 0, x, offs + rho_k);
            if (increased_accuracy)
            {
                gemv_n(nu - rho_k, gamma_k - rho_k, -1.0, Ggt_tilde[k], 0, rho_k, eq_mult, offs_g_k,
                       1.0, x, offs + rho_k, x, offs + rho_k);
            }
            gemv_t(nx, numrho_k, -1.0, Llt[k], numrho_k, 0, x, offs_x, 1.0, x, offs + rho_k, x,
                   offs + rho_k);
            trsv_ltn(numrho_k, Llt[k], 0, 0, x, offs + rho_k, x, offs + rho_k);
        }
        //         /// calcualate uka_tilde
        if (rho_k > 0)
        {
            // ROWEX(rho_k, 1.0, Ggt_tilde[k], numrho_k + nx, 0, ux[0], offs);
            veccp(rho_k, v_Ggt_tilde[k], 0, x, offs);
            gemv_t(nx + numrho_k, rho_k, 1.0, Ggt_tilde[k], 0, 0, x, offs + rho_k, 1.0, x, offs, x,
                   offs);
            veccp(gammamrho_k, eq_mult, offs_g_k, v_tmp[0], 0);
            veccp(gammamrho_k, v_tmp[0], 0, eq_mult, offs_g_k + rho_k);
            veccpsc(rho_k, -1.0, v_RSQrqt_tilde[k], 0, eq_mult, offs_g_k);
            gemv_t(nu + nx, rho_k, -1.0, RSQrqt_tilde[k], 0, 0, x, offs, 1.0, eq_mult, offs_g_k,
                   eq_mult, offs_g_k);
            gecp(rho_k, gamma_k, Ggt_tilde[k], nu - rho_k + nx + 1, 0, AL[0], 0, 0);
            trsv_lnn(rho_k, AL[0], 0, 0, eq_mult, offs_g_k, eq_mult, offs_g_k);
            trsv_unu(rho_k, gamma_k, AL[0], 0, 0, eq_mult, offs_g_k, eq_mult, offs_g_k);
            Pl[k].apply_inverse(rho_k, &eq_mult.vec(), offs_g_k);
            Pr[k].apply_inverse(rho_k, &x.vec(), offs);
        }
        if (ng_ineq > 0)
        {
            gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_eq_ineq,
                   eq_mult, offs_eq_ineq);
            eq_mult.block(ng_ineq, offs_eq_ineq) =
                eq_mult.block(ng_ineq, offs_eq_ineq) / D_s.block(ng_ineq, offs_slack);
        }
        if (k != info.dims.K - 1)
        {
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index nup1 = info.dims.number_of_controls[k + 1];
            const Index offsp1 = info.offsets_primal_u[k + 1];
            const Index offsxp1 = info.offsets_primal_x[k + 1];
            const Index offs_g_kp1 = info.offsets_g_eq_path[k + 1];
            const Index offs_dyn_k = info.offsets_g_eq_dyn[k];
            const Index gammamrho_kp1 = gamma[k + 1] - rho[k + 1];
            veccp(nxp1, g, offs_dyn_k, x, offsxp1);
            gemv_t(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, x, offs, 1.0, x, offsxp1, x,
                   offsxp1);
            veccp(nxp1, v_Ppt[k + 1], 0, eq_mult, offs_dyn_k);
            gemv_t(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, x, offsxp1, 1.0, eq_mult, offs_dyn_k, eq_mult,
                   offs_dyn_k);
            gemv_t(gammamrho_kp1, nxp1, 1.0, Hh[k + 1], 0, 0, eq_mult, offs_g_kp1, 1.0, eq_mult,
                   offs_dyn_k, eq_mult, offs_dyn_k);
        }
    }
    return LinsolReturnFlag::SUCCESS;
}

LinsolReturnFlag AugSystemSolver<OcpType>::solve_rhs_batch(
    const ProblemInfo &info,
    const Jacobian<OcpType> &jacobian,
    const Hessian<OcpType> &hessian,
    const VecRealView &D_s,
    const MatRealView &f,
    const MatRealView &g,
    MatRealView &x,
    MatRealView &eq_mult)
{
    const Index right_hand_sides = f.n();
    fatrop_dbg_assert(f.m() == info.number_of_primal_variables);
    fatrop_dbg_assert(g.m() == info.number_of_eq_constraints);
    fatrop_dbg_assert(g.n() == right_hand_sides);
    fatrop_dbg_assert(x.m() == info.number_of_primal_variables);
    fatrop_dbg_assert(x.n() == right_hand_sides);
    fatrop_dbg_assert(eq_mult.m() == info.number_of_eq_constraints);
    fatrop_dbg_assert(eq_mult.n() == right_hand_sides);
    (void)hessian;

    if (right_hand_sides == 0)
        return LinsolReturnFlag::SUCCESS;

    const Index max_number_of_controls =
        *std::max_element(
            info.dims.number_of_controls.begin(),
            info.dims.number_of_controls.end());
    const Index max_number_of_states =
        *std::max_element(
            info.dims.number_of_states.begin(),
            info.dims.number_of_states.end());
    const Index max_number_of_variables =
        *std::max_element(
            info.number_of_stage_variables.begin(),
            info.number_of_stage_variables.end());
    const Index max_number_of_ineq_constraints =
        *std::max_element(
            info.dims.number_of_ineq_constraints.begin(),
            info.dims.number_of_ineq_constraints.end());

    std::vector<MatRealAllocated> batch_Ppt;
    std::vector<MatRealAllocated> batch_Hh;
    std::vector<MatRealAllocated> batch_RSQrqt_tilde;
    std::vector<MatRealAllocated> batch_Ggt_tilde;
    std::vector<MatRealAllocated> batch_Llt;
    batch_Ppt.reserve(info.dims.K);
    batch_Hh.reserve(info.dims.K);
    batch_RSQrqt_tilde.reserve(info.dims.K);
    batch_Ggt_tilde.reserve(info.dims.K);
    batch_Llt.reserve(info.dims.K);
    for (Index k = 0; k < info.dims.K; ++k)
    {
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        batch_Ppt.emplace_back(nx, right_hand_sides);
        batch_Hh.emplace_back(nx, right_hand_sides);
        batch_RSQrqt_tilde.emplace_back(
            nu + nx, right_hand_sides);
        batch_Ggt_tilde.emplace_back(
            nu + nx, right_hand_sides);
        batch_Llt.emplace_back(
            nu + nx, right_hand_sides);
    }

    MatRealAllocated batch_AL(
        max_number_of_variables, right_hand_sides);
    MatRealAllocated batch_Ggt_rhs(
        max_number_of_variables, right_hand_sides);
    MatRealAllocated batch_GgLt(
        max_number_of_variables, right_hand_sides);
    MatRealAllocated batch_RSQrqt_hat(
        max_number_of_variables, right_hand_sides);
    MatRealAllocated batch_GgIt_tilde(
        max_number_of_states, right_hand_sides);
    MatRealAllocated batch_GgLIt(
        max_number_of_states, right_hand_sides);
    MatRealAllocated batch_HhIt(
        max_number_of_states, right_hand_sides);
    MatRealAllocated batch_PpIt_hat(
        max_number_of_states, right_hand_sides);
    MatRealAllocated batch_LlIt(
        max_number_of_states, right_hand_sides);
    MatRealAllocated batch_ineq(
        max_number_of_ineq_constraints, right_hand_sides);
    MatRealAllocated batch_tmp(
        max_number_of_variables, right_hand_sides);
    x = 0.0;
    eq_mult = 0.0;

    for (Index k = info.dims.K; k-- > 0;)
    {
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq =
            info.dims.number_of_ineq_constraints[k];
        const Index offset_ineq_k = info.offsets_slack[k];
        const Index offs_g_ineq_k =
            info.offsets_g_eq_slack[k];
        const Index offs_g_k =
            info.offsets_g_eq_path[k];
        const Index offs = info.offsets_primal_u[k];

        Index gamma_k = 0;
        if (k == info.dims.K - 1)
        {
            gamma_k = ng;
            if (ng > 0)
            {
                gecp(
                    ng, right_hand_sides,
                    g, offs_g_k, 0,
                    batch_Ggt_rhs, 0, 0);
            }
            gecp(
                nu + nx, right_hand_sides,
                f, offs, 0,
                batch_RSQrqt_tilde[k], 0, 0);
        }
        else
        {
            const Index offs_dyn_k =
                info.offsets_g_eq_dyn[k];
            const Index nxp1 =
                info.dims.number_of_states[k + 1];
            const Index Hp1_size =
                gamma[k + 1] - rho[k + 1];
            gamma_k = Hp1_size + ng;

            gemm_nn(
                nxp1, right_hand_sides, nxp1, 1.0,
                Ppt[k + 1], 0, 0,
                g, offs_dyn_k, 0,
                0.0, batch_AL, 0, 0,
                batch_AL, 0, 0);
            gead(
                nxp1, right_hand_sides, 1.0,
                batch_Ppt[k + 1], 0, 0,
                batch_AL, 0, 0);
            gecp(
                nu + nx, right_hand_sides,
                f, offs, 0,
                batch_RSQrqt_tilde[k], 0, 0);
            gemm_nn(
                nu + nx, right_hand_sides, nxp1, 1.0,
                jacobian.BAbt[k], 0, 0,
                batch_AL, 0, 0,
                1.0, batch_RSQrqt_tilde[k], 0, 0,
                batch_RSQrqt_tilde[k], 0, 0);

            if (ng > 0)
            {
                gecp(
                    ng, right_hand_sides,
                    g, offs_g_k, 0,
                    batch_Ggt_rhs, 0, 0);
            }
            if (Hp1_size > 0)
            {
                gemm_nn(
                    Hp1_size, right_hand_sides, nxp1, 1.0,
                    Hh[k + 1], 0, 0,
                    g, offs_dyn_k, 0,
                    0.0, batch_Ggt_rhs, ng, 0,
                    batch_Ggt_rhs, ng, 0);
                gead(
                    Hp1_size, right_hand_sides, 1.0,
                    batch_Hh[k + 1], 0, 0,
                    batch_Ggt_rhs, ng, 0);
            }
        }

        if (ng_ineq > 0)
        {
            gecp(
                ng_ineq, right_hand_sides,
                g, offs_g_ineq_k, 0,
                batch_ineq, 0, 0);
            for (Index equation = 0;
                 equation < ng_ineq; ++equation)
            {
                const Scalar inverse_scale =
                    1.0 / D_s(offset_ineq_k + equation);
                for (Index column = 0;
                     column < right_hand_sides; ++column)
                {
                    batch_ineq(equation, column) *=
                        inverse_scale;
                }
            }
            gemm_nn(
                nu + nx, right_hand_sides, ng_ineq, 1.0,
                jacobian.Gg_ineqt[k], 0, 0,
                batch_ineq, 0, 0,
                1.0, batch_RSQrqt_tilde[k], 0, 0,
                batch_RSQrqt_tilde[k], 0, 0);
        }

        const Index rank_k = rho[k];
        if (rank_k > 0)
        {
            gecp(
                rank_k, gamma_k,
                Ggt_tilde[k], nu - rank_k + nx + 1, 0,
                Ggt_stripe[0], 0, 0);
            Pl[k].apply_on_rows(
                rank_k, &batch_Ggt_rhs.mat(), 0,
                right_hand_sides);
            trsm_lutu(
                rank_k, right_hand_sides, 1.0,
                Ggt_stripe[0], 0, 0,
                batch_Ggt_rhs, 0, 0,
                batch_Ggt_rhs, 0, 0);
        }
        if (gamma_k - rank_k > 0 && rank_k > 0)
        {
            gemm_tn(
                gamma_k - rank_k, right_hand_sides,
                rank_k, -1.0,
                Ggt_stripe[0], 0, rank_k,
                batch_Ggt_rhs, 0, 0,
                1.0, batch_Ggt_rhs, rank_k, 0,
                batch_Ggt_rhs, rank_k, 0);
        }
        if (gamma_k - rank_k > 0)
        {
            gecp(
                gamma_k - rank_k, right_hand_sides,
                batch_Ggt_rhs, rank_k, 0,
                batch_Hh[k], 0, 0);
        }

        MatRealView *batch_RSQrq_hat = nullptr;
        if (rank_k > 0)
        {
            gecp(
                rank_k, right_hand_sides,
                batch_Ggt_rhs, 0, 0,
                batch_Ggt_tilde[k], 0, 0);
            gesc(
                rank_k, right_hand_sides, -1.0,
                batch_Ggt_tilde[k], 0, 0);
            trsm_lltn(
                rank_k, right_hand_sides, 1.0,
                Ggt_stripe[0], 0, 0,
                batch_Ggt_tilde[k], 0, 0,
                batch_Ggt_tilde[k], 0, 0);

            Pr[k].apply_on_rows(
                rank_k, &batch_RSQrqt_tilde[k].mat(),
                0, right_hand_sides);
            gecp(
                nu + nx, right_hand_sides,
                batch_RSQrqt_tilde[k], 0, 0,
                batch_GgLt, 0, 0);
            gemm_nn(
                nu + nx, right_hand_sides, rank_k, 1.0,
                RSQrqt_tilde[k], 0, 0,
                batch_Ggt_tilde[k], 0, 0,
                1.0, batch_GgLt, 0, 0,
                batch_GgLt, 0, 0);
            gemm_nn(
                nu + nx - rank_k, right_hand_sides,
                rank_k, 1.0,
                Ggt_tilde[k], 0, 0,
                batch_GgLt, 0, 0,
                1.0, batch_GgLt, rank_k, 0,
                batch_RSQrqt_hat, 0, 0);
            batch_RSQrq_hat = &batch_RSQrqt_hat;
        }
        else
        {
            batch_RSQrq_hat =
                &batch_RSQrqt_tilde[k];
        }

        const Index reduced_controls = nu - rank_k;
        if (reduced_controls > 0)
        {
            trsm_llnn(
                reduced_controls, right_hand_sides, 1.0,
                Llt[k], 0, 0,
                *batch_RSQrq_hat, 0, 0,
                batch_Llt[k], 0, 0);
            gecp(
                nx, right_hand_sides,
                *batch_RSQrq_hat, reduced_controls, 0,
                batch_Ppt[k], 0, 0);
            gemm_nn(
                nx, right_hand_sides, reduced_controls,
                -1.0, Llt[k], reduced_controls, 0,
                batch_Llt[k], 0, 0,
                1.0, batch_Ppt[k], 0, 0,
                batch_Ppt[k], 0, 0);
            if (increased_accuracy &&
                gamma_k - rank_k > 0)
            {
                gemm_tn(
                    gamma_k - rank_k,
                    right_hand_sides,
                    reduced_controls, -1.0,
                    Ggt_tilde[k], 0, rank_k,
                    batch_Llt[k], 0, 0,
                    1.0, batch_Hh[k], 0, 0,
                    batch_Hh[k], 0, 0);
            }
        }
        else
        {
            gecp(
                nx, right_hand_sides,
                *batch_RSQrq_hat, 0, 0,
                batch_Ppt[k], 0, 0);
        }
    }

    const Index nx0 = info.dims.number_of_states[0];
    const Index gamma_I = gamma[0] - rho[0];
    if (gamma_I > 0)
    {
        gecp(
            gamma_I, right_hand_sides,
            batch_Hh[0], 0, 0,
            batch_HhIt, 0, 0);
        PlI[0].apply_on_rows(
            rankI, &batch_HhIt.mat(), 0,
            right_hand_sides);
        if (rankI > 0)
        {
            trsm_lutu(
                rankI, right_hand_sides, 1.0,
                HhIt[0], 0, 0,
                batch_HhIt, 0, 0,
                batch_HhIt, 0, 0);
        }
        if (gamma_I - rankI > 0 && rankI > 0)
        {
            gemm_tn(
                gamma_I - rankI, right_hand_sides,
                rankI, -1.0,
                HhIt[0], 0, rankI,
                batch_HhIt, 0, 0,
                1.0, batch_HhIt, rankI, 0,
                batch_HhIt, rankI, 0);
        }
        if (rankI > 0)
        {
            gecp(
                rankI, right_hand_sides,
                batch_HhIt, 0, 0,
                batch_GgIt_tilde, 0, 0);
            gesc(
                rankI, right_hand_sides, -1.0,
                batch_GgIt_tilde, 0, 0);
            trsm_lltn(
                rankI, right_hand_sides, 1.0,
                HhIt[0], 0, 0,
                batch_GgIt_tilde, 0, 0,
                batch_GgIt_tilde, 0, 0);
        }
        PrI[0].apply_on_rows(
            rankI, &batch_Ppt[0].mat(), 0,
            right_hand_sides);
        gecp(
            nx0, right_hand_sides,
            batch_Ppt[0], 0, 0,
            batch_GgLIt, 0, 0);
        if (rankI > 0)
        {
            gemm_nn(
                nx0, right_hand_sides, rankI, 1.0,
                Ppt[0], 0, 0,
                batch_GgIt_tilde, 0, 0,
                1.0, batch_GgLIt, 0, 0,
                batch_GgLIt, 0, 0);
        }
        if (nx0 - rankI > 0)
        {
            gemm_nn(
                nx0 - rankI, right_hand_sides,
                rankI, 1.0,
                GgIt_tilde[0], 0, 0,
                batch_GgLIt, 0, 0,
                1.0, batch_GgLIt, rankI, 0,
                batch_PpIt_hat, 0, 0);
            trsm_llnn(
                nx0 - rankI, right_hand_sides, 1.0,
                LlIt[0], 0, 0,
                batch_PpIt_hat, 0, 0,
                batch_LlIt, 0, 0);
        }
    }
    else
    {
        trsm_llnn(
            nx0, right_hand_sides, 1.0,
            LlIt[0], 0, 0,
            batch_Ppt[0], 0, 0,
            batch_LlIt, 0, 0);
    }

    const Index offs_x0 = info.offsets_primal_x[0];
    const Index offs_g0 = info.offsets_g_eq_path[0];
    if (nx0 - rankI > 0)
    {
        gecp(
            nx0 - rankI, right_hand_sides,
            batch_LlIt, 0, 0,
            x, offs_x0 + rankI, 0);
        gesc(
            nx0 - rankI, right_hand_sides, -1.0,
            x, offs_x0 + rankI, 0);
        trsm_lltn(
            nx0 - rankI, right_hand_sides, 1.0,
            LlIt[0], 0, 0,
            x, offs_x0 + rankI, 0,
            x, offs_x0 + rankI, 0);
    }
    if (rankI > 0)
    {
        gecp(
            rankI, right_hand_sides,
            batch_GgIt_tilde, 0, 0,
            x, offs_x0, 0);
        if (nx0 - rankI > 0)
        {
            gemm_tn(
                rankI, right_hand_sides,
                nx0 - rankI, 1.0,
                GgIt_tilde[0], 0, 0,
                x, offs_x0 + rankI, 0,
                1.0, x, offs_x0, 0,
                x, offs_x0, 0);
        }
        gecp(
            rankI, right_hand_sides,
            batch_Ppt[0], 0, 0,
            eq_mult, offs_g0, 0);
        gesc(
            rankI, right_hand_sides, -1.0,
            eq_mult, offs_g0, 0);
        gemm_tn(
            rankI, right_hand_sides, nx0, -1.0,
            Ppt[0], 0, 0,
            x, offs_x0, 0,
            1.0, eq_mult, offs_g0, 0,
            eq_mult, offs_g0, 0);
        trsm_llnn(
            rankI, right_hand_sides, 1.0,
            HhIt[0], 0, 0,
            eq_mult, offs_g0, 0,
            eq_mult, offs_g0, 0);
        trsm_lunu(
            rankI, right_hand_sides, 1.0,
            HhIt[0], 0, 0,
            eq_mult, offs_g0, 0,
            eq_mult, offs_g0, 0);
        PlI[0].apply_inverse_on_rows(
            rankI, &eq_mult.mat(), offs_g0);
        PrI[0].apply_inverse_on_rows(
            rankI, &x.mat(), offs_x0);
    }
    for (Index k = 0; k < info.dims.K; ++k)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_x = info.offsets_primal_x[k];
        const Index rho_k = rho[k];
        const Index reduced_controls = nu - rho_k;
        const Index offs_g_k =
            info.offsets_g_eq_path[k];
        const Index gamma_minus_rank =
            gamma[k] - rho[k];
        const Index gamma_k = gamma[k];
        const Index ng_ineq =
            info.dims.number_of_ineq_constraints[k];
        const Index offs_eq_ineq =
            info.offsets_g_eq_slack[k];
        const Index offs_slack = info.offsets_slack[k];

        if (reduced_controls > 0)
        {
            gecp(
                reduced_controls, right_hand_sides,
                batch_Llt[k], 0, 0,
                x, offs + rho_k, 0);
            gesc(
                reduced_controls, right_hand_sides, -1.0,
                x, offs + rho_k, 0);
            if (increased_accuracy &&
                gamma_minus_rank > 0)
            {
                gemm_nn(
                    reduced_controls, right_hand_sides,
                    gamma_minus_rank, -1.0,
                    Ggt_tilde[k], 0, rho_k,
                    eq_mult, offs_g_k, 0,
                    1.0, x, offs + rho_k, 0,
                    x, offs + rho_k, 0);
            }
            gemm_tn(
                reduced_controls, right_hand_sides,
                nx, -1.0,
                Llt[k], reduced_controls, 0,
                x, offs_x, 0,
                1.0, x, offs + rho_k, 0,
                x, offs + rho_k, 0);
            trsm_lltn(
                reduced_controls, right_hand_sides, 1.0,
                Llt[k], 0, 0,
                x, offs + rho_k, 0,
                x, offs + rho_k, 0);
        }

        if (rho_k > 0)
        {
            gecp(
                rho_k, right_hand_sides,
                batch_Ggt_tilde[k], 0, 0,
                x, offs, 0);
            gemm_tn(
                rho_k, right_hand_sides,
                nx + reduced_controls, 1.0,
                Ggt_tilde[k], 0, 0,
                x, offs + rho_k, 0,
                1.0, x, offs, 0,
                x, offs, 0);

            if (gamma_minus_rank > 0)
            {
                gecp(
                    gamma_minus_rank, right_hand_sides,
                    eq_mult, offs_g_k, 0,
                    batch_tmp, 0, 0);
                gecp(
                    gamma_minus_rank, right_hand_sides,
                    batch_tmp, 0, 0,
                    eq_mult, offs_g_k + rho_k, 0);
            }
            gecp(
                rho_k, right_hand_sides,
                batch_RSQrqt_tilde[k], 0, 0,
                eq_mult, offs_g_k, 0);
            gesc(
                rho_k, right_hand_sides, -1.0,
                eq_mult, offs_g_k, 0);
            gemm_tn(
                rho_k, right_hand_sides,
                nu + nx, -1.0,
                RSQrqt_tilde[k], 0, 0,
                x, offs, 0,
                1.0, eq_mult, offs_g_k, 0,
                eq_mult, offs_g_k, 0);
            gecp(
                rho_k, gamma_k,
                Ggt_tilde[k],
                nu - rho_k + nx + 1, 0,
                AL[0], 0, 0);
            trsm_llnn(
                rho_k, right_hand_sides, 1.0,
                AL[0], 0, 0,
                eq_mult, offs_g_k, 0,
                eq_mult, offs_g_k, 0);
            solve_upper_unit_rectangular_batch(
                rho_k, gamma_k, right_hand_sides,
                AL[0], 0, 0,
                eq_mult, offs_g_k, 0);
            Pl[k].apply_inverse_on_rows(
                rho_k, &eq_mult.mat(), offs_g_k);
            Pr[k].apply_inverse_on_rows(
                rho_k, &x.mat(), offs);
        }

        if (ng_ineq > 0)
        {
            gecp(
                ng_ineq, right_hand_sides,
                g, offs_eq_ineq, 0,
                eq_mult, offs_eq_ineq, 0);
            gemm_tn(
                ng_ineq, right_hand_sides,
                nu + nx, 1.0,
                jacobian.Gg_ineqt[k], 0, 0,
                x, offs, 0,
                1.0, eq_mult, offs_eq_ineq, 0,
                eq_mult, offs_eq_ineq, 0);
            for (Index equation = 0;
                 equation < ng_ineq; ++equation)
            {
                const Scalar inverse_scale =
                    1.0 / D_s(offs_slack + equation);
                for (Index column = 0;
                     column < right_hand_sides; ++column)
                {
                    eq_mult(
                        offs_eq_ineq + equation,
                        column) *= inverse_scale;
                }
            }
        }

        if (k != info.dims.K - 1)
        {
            const Index nxp1 =
                info.dims.number_of_states[k + 1];
            const Index offs_x_p1 =
                info.offsets_primal_x[k + 1];
            const Index offs_g_kp1 =
                info.offsets_g_eq_path[k + 1];
            const Index offs_dyn_k =
                info.offsets_g_eq_dyn[k];
            const Index gamma_minus_rank_kp1 =
                gamma[k + 1] - rho[k + 1];

            gecp(
                nxp1, right_hand_sides,
                g, offs_dyn_k, 0,
                x, offs_x_p1, 0);
            gemm_tn(
                nxp1, right_hand_sides,
                nu + nx, 1.0,
                jacobian.BAbt[k], 0, 0,
                x, offs, 0,
                1.0, x, offs_x_p1, 0,
                x, offs_x_p1, 0);
            gecp(
                nxp1, right_hand_sides,
                batch_Ppt[k + 1], 0, 0,
                eq_mult, offs_dyn_k, 0);
            gemm_tn(
                nxp1, right_hand_sides, nxp1, 1.0,
                Ppt[k + 1], 0, 0,
                x, offs_x_p1, 0,
                1.0, eq_mult, offs_dyn_k, 0,
                eq_mult, offs_dyn_k, 0);
            if (gamma_minus_rank_kp1 > 0)
            {
                gemm_tn(
                    nxp1, right_hand_sides,
                    gamma_minus_rank_kp1, 1.0,
                    Hh[k + 1], 0, 0,
                    eq_mult, offs_g_kp1, 0,
                    1.0, eq_mult, offs_dyn_k, 0,
                    eq_mult, offs_dyn_k, 0);
            }
        }
    }
    return LinsolReturnFlag::SUCCESS;
}

LinsolReturnFlag AugSystemSolver<OcpType>::solve_rhs(const ProblemInfo &info,
                                               const Jacobian<OcpType> &jacobian,
                                               const Hessian<OcpType> &hessian,
                                               const VecRealView &D_eq, const VecRealView &D_s,
                                               const VecRealView &f, const VecRealView &g,
                                               VecRealView &x, VecRealView &eq_mult)
{
    VecRealView *v_RSQrq_hat_curr_p;
    for (Index k = info.dims.K - 1; k >= 0; --k)
    {
        const Index offs_ux_k = info.offsets_primal_u[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_g_dyn = info.offsets_g_eq_dyn[k];
        const Index offs_g_eq = info.offsets_g_eq_path[k];
        const Index offs_ge_eq_ineq = info.offsets_g_eq_slack[k];
        //     //////// SUBSDYN
        if (k == info.dims.K - 1)
        {
            veccp(nu + nx, f, offs_ux_k, v_RSQrqt_tilde[k], 0);
        }
        else
        {
            const Index nxp1 = info.dims.number_of_states[k + 1];
            gemv_n(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, g, offs_g_dyn, 0.0, v_AL[0], 0, v_AL[0], 0);
            axpy(nxp1, 1.0, v_Ppt[k + 1], 0, v_AL[0], 0, v_AL[0], 0);
            gemv_n(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, v_AL[0], 0, 1.0, f, offs_ux_k,
                   v_RSQrqt_tilde[k], 0);
        }
        if (ng > 0)
        {
            const Index offs_eq_k = info.offsets_eq[k];
            for (Index i = 0; i < ng; i++)
            {
                Scalar scaling_factor = D_eq(offs_eq_k + i);
                v_Ggt_stripe[0](i) = g(offs_g_eq + i) / scaling_factor;
            }
            gemv_n(nu + nx, ng, 1.0, jacobian.Gg_eqt[k], 0, 0, v_Ggt_stripe[0], 0, 1.0,
                   v_RSQrqt_tilde[k], 0, v_RSQrqt_tilde[k], 0);
        }
        if (ng_ineq > 0)
        {
            const Index offs_ineq_k = info.offsets_slack[k];
            for (Index i = 0; i < ng_ineq; i++)
            {
                Scalar scaling_factor = D_s(offs_ineq_k + i);
                v_Ggt_ineq_temp[0](i) = g(offs_ge_eq_ineq + i) / scaling_factor;
            }
            gemv_n(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, v_Ggt_ineq_temp[0], 0, 1.0,
                   v_RSQrqt_tilde[k], 0, v_RSQrqt_tilde[k], 0);
        }
        {
            v_RSQrq_hat_curr_p = &v_RSQrqt_tilde[k];
        }
        {
            trsv_lnn(nu, Llt[k], 0, 0, *v_RSQrq_hat_curr_p, 0, v_Llt[k], 0);
            veccp(nu, v_Llt[k], 0, v_Llt_shift[0], 0);
            gemv_n(nx, nu, -1.0, Llt[k], nu, 0, v_Llt_shift[0], 0, 1.0, v_RSQrqt_tilde[k], nu,
                   v_Ppt[k], 0);
        }
    }
    {
        const Index nx = info.dims.number_of_states[0];
        {
            trsv_lnn(nx, LlIt[0], 0, 0, v_Ppt[0], 0, v_LlIt[0], 0);
        }
    }
    {
        const Index nx = info.dims.number_of_states[0];
        const Index nu = info.dims.number_of_controls[0];
        const Index offs_x = info.offsets_primal_x[0];
        veccpsc(nx, -1.0, v_LlIt[0], 0, x, offs_x);
        trsv_ltn(nx, LlIt[0], 0, 0, x, offs_x, x, offs_x);
    }
    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_x = info.offsets_primal_x[k];
        const Index offs_dyn_eq_k = info.offsets_g_eq_dyn[k];
        veccpsc(nu, -1.0, v_Llt[k], 0, x, offs);
        gemv_t(nx, nu, -1.0, Llt[k], nu, 0, x, offs_x, 1.0, x, offs, x, offs);
        trsv_ltn(nu, Llt[k], 0, 0, x, offs, x, offs);
        if (k != info.dims.K - 1)
        {
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index offsp1 = info.offsets_primal_u[k + 1];
            const Index offs_x_p1 = info.offsets_primal_x[k + 1];
            veccp(nxp1, g, offs_dyn_eq_k, x, offs_x_p1);
            gemv_t(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, x, offs, 1.0, x, offs_x_p1, x,
                   offs_x_p1);
            veccp(nxp1, v_Ppt[k + 1], 0, eq_mult, offs_dyn_eq_k);
            gemv_t(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, x, offs_x_p1, 1.0, eq_mult,
                   offs_dyn_eq_k, eq_mult, offs_dyn_eq_k);
        }
    }
    // // calculate lam_eq xk
    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_g_k = info.offsets_g_eq_path[k];
        const Index offs_eq = info.offsets_eq[k];
        if (ng > 0)
        {
            gemv_t(nu + nx, ng, 1.0, jacobian.Gg_eqt[k], 0, 0, x, offs, 1.0, g, offs_g_k,
                   eq_mult, offs_g_k);
            eq_mult.block(ng, offs_g_k) =
                eq_mult.block(ng, offs_g_k) / D_eq.block(ng, offs_eq);
        }
    }

    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_gineq_k = info.offsets_g_eq_slack[k];
        const Index offs_slack = info.offsets_slack[k];
        if (ng_ineq > 0)
        {
            gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_gineq_k,
                   eq_mult, offs_gineq_k);
            eq_mult.block(ng_ineq, offs_gineq_k) =
                eq_mult.block(ng_ineq, offs_gineq_k) / D_s.block(ng_ineq, offs_slack);
        }
    }
    return LinsolReturnFlag::SUCCESS;
}





















AugSystemSolver<AcceleratedOcpType>::AugSystemSolver(const ProblemInfo &info)
{
    Index max_number_of_controls =
        *std::max_element(info.dims.number_of_controls.begin(), info.dims.number_of_controls.end());
    Index max_number_of_states =
        *std::max_element(info.dims.number_of_states.begin(), info.dims.number_of_states.end());
    Index max_number_of_variables = *std::max_element(info.number_of_stage_variables.begin(),
                                                      info.number_of_stage_variables.end());
    Index max_number_of_ineq_constraints = *std::max_element(
        info.dims.number_of_ineq_constraints.begin(), info.dims.number_of_ineq_constraints.end());
    Index max_number_of_eq_consttraints = *std::max_element(
        info.dims.number_of_eq_constraints.begin(), info.dims.number_of_eq_constraints.end());

    AL.emplace_back(max_number_of_variables + 1, max_number_of_variables);
    Ggt_stripe.emplace_back(max_number_of_variables + 1, max_number_of_variables);
    GgLt.emplace_back(max_number_of_variables + 1, max_number_of_variables);
    RSQrqt_hat.emplace_back(max_number_of_variables + 1, max_number_of_variables);
    Llt_shift.emplace_back(max_number_of_variables + 1, max_number_of_controls);
    GgIt_tilde.emplace_back(info.dims.number_of_states[0] + 1, info.dims.number_of_states[0]);
    GgLIt.emplace_back(info.dims.number_of_states[0] + 1, info.dims.number_of_states[0]);
    HhIt.emplace_back(info.dims.number_of_states[0] + 1, info.dims.number_of_states[0]);
    PpIt_hat.emplace_back(info.dims.number_of_states[0] + 1, info.dims.number_of_states[0]);
    LlIt.emplace_back(info.dims.number_of_states[0] + 1, info.dims.number_of_states[0]);
    Ggt_ineq_temp.emplace_back(max_number_of_variables + 1, max_number_of_ineq_constraints);

    Ppt.reserve(info.dims.K);
    Hh.reserve(info.dims.K);
    RSQrqt_tilde.reserve(info.dims.K);
    Ggt_tilde.reserve(info.dims.K);
    Llt.reserve(info.dims.K);
    for (Index k = 0; k < info.dims.K; k++)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        Index ng_eq = info.dims.number_of_eq_constraints[k];
        Ppt.emplace_back(nx + 1, nx);
        Hh.emplace_back(nx, nx + 1);
        RSQrqt_tilde.emplace_back(nu + nx + 1, nx + nu);
        Ggt_tilde.emplace_back(nu + nx + 1, nx + nu);
        Llt.emplace_back(nu + nx + 1, nu);
    }

    v_AL.emplace_back(max_number_of_variables);
    v_Ggt_stripe.emplace_back(max_number_of_variables);
    v_GgLt.emplace_back(max_number_of_variables);
    v_RSQrqt_hat.emplace_back(max_number_of_variables);
    v_Llt_shift.emplace_back(max_number_of_controls);
    v_GgIt_tilde.emplace_back(info.dims.number_of_states[0]);
    v_GgLIt.emplace_back(info.dims.number_of_states[0]);
    v_HhIt.emplace_back(info.dims.number_of_states[0]);
    v_PpIt_hat.emplace_back(info.dims.number_of_states[0]);
    v_LlIt.emplace_back(info.dims.number_of_states[0]);
    v_Ggt_ineq_temp.emplace_back(max_number_of_ineq_constraints);
    v_tmp.emplace_back(max_number_of_variables);

    v_Ppt.reserve(info.dims.K);
    v_Hh.reserve(info.dims.K);
    v_RSQrqt_tilde.reserve(info.dims.K);
    v_Ggt_tilde.reserve(info.dims.K);
    v_Llt.reserve(info.dims.K);

    for (Index k = 0; k < info.dims.K; k++)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        v_Ppt.emplace_back(nx);
        v_Hh.emplace_back(nx);
        v_RSQrqt_tilde.emplace_back(nu + nx);
        v_Ggt_tilde.emplace_back(nu + nx);
        v_Llt.emplace_back(nu + nx);
    }

    PlI.emplace_back(info.dims.number_of_states[0]);
    PrI.emplace_back(info.dims.number_of_states[0]);

    Pl1.reserve(info.dims.K);
    Pl_rank.reserve(info.dims.K);
    Pl2.reserve(info.dims.K);
    Pr1.reserve(info.dims.K);
    Pr2.reserve(info.dims.K);
    scratch.emplace_back(max_number_of_variables, max_number_of_variables);

    for (Index k = 0; k < info.dims.K; k++)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng = info.dims.number_of_eq_constraints[k];
        // we know gamma <= nu + nx + 1
        // Pl1.emplace_back(2*max_number_of_eq_consttraints);
        // Pl_rank.emplace_back(2*max_number_of_eq_consttraints + nx);
        // Pl2.emplace_back(2*max_number_of_eq_consttraints);
        // Pr1.emplace_back(nu + nx + 1);
        // Pr2.emplace_back(nu + nx + 1);
        Pl1.emplace_back(nu + nx + 1);
        Pl_rank.emplace_back(nu + (nx + nu + 1) + 1);
        Pl2.emplace_back(nu + (nx + nu + 1) + 1);
        Pr1.emplace_back(nu + nx + 1);
        Pr2.emplace_back(nu + nx + 1);
    }

    gamma.resize(info.dims.K);
    rho.resize(info.dims.K);
    rho1.resize(info.dims.K);
    rho2.resize(info.dims.K);
};

void AugSystemSolver<AcceleratedOcpType>::register_options(OptionRegistry &registry)
{
    registry.register_option("linsol_it_ref", &AugSystemSolver<AcceleratedOcpType>::set_it_ref, this);
    registry.register_option("linsol_perturbed_mode", &AugSystemSolver<AcceleratedOcpType>::set_perturbed_mode, this);
    registry.register_option("linsol_perturbed_mode_param", &AugSystemSolver<AcceleratedOcpType>::set_perturbed_mode_param, this);
    registry.register_option("linsol_lu_fact_tol", &AugSystemSolver<AcceleratedOcpType>::set_lu_fact_tol, this);
    registry.register_option("linsol_diagnostic", &AugSystemSolver<AcceleratedOcpType>::set_diagnostic, this);
    registry.register_option("linsol_increased_accuracy", &AugSystemSolver<AcceleratedOcpType>::set_increased_accuracy, this);
    registry.register_option("linsol_nb_of_dynamics_constraints", &AugSystemSolver<AcceleratedOcpType>::set_nb_of_dynamics_constraints, this);
    registry.register_option("linsol_nb_of_zk_vars", &AugSystemSolver<AcceleratedOcpType>::set_nb_of_zk_vars, this);
}

LinsolReturnFlag AugSystemSolver<AcceleratedOcpType>::solve(const ProblemInfo &info,
                                           Jacobian<AcceleratedOcpType> &jacobian, Hessian<AcceleratedOcpType> &hessian,
                                           const VecRealView &D_x, const VecRealView &D_s,
                                           const VecRealView &f, const VecRealView &g,
                                           VecRealView &x, VecRealView &eq_mult)
{
    // std::cout << "solving AcceleratedOcpType!!" << std::endl;
    // for (Index k = info.dims.K - 1; k >= 0; --k){
    //     std::cout << "Gg_eqt[" << k << "]:\n" << jacobian.Gg_eqt[k] << "\n";
    // }
    MatRealView *RSQrq_hat_curr_p;
    Index rank_k;
    auto intermediate_start = std::chrono::high_resolution_clock::now();
    /////////////// recursion ///////////////
    for (Index k = info.dims.K - 1; k >= 0; --k)
    {
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offset_ineq_k = info.offsets_slack[k];
        const Index offset_u = info.offsets_primal_u[k];
        const Index offset_eq_path = info.offsets_g_eq_path[k];
        const Index offset_eq_slack = info.offsets_g_eq_slack[k];
        //////// SUBSDYN
        Index gamma_k;
        if (k == info.dims.K - 1)
        {
            gamma_k = ng;
            gamma[k] = gamma_k;
            rowin(ng, 1.0, g, offset_eq_path, jacobian.Gg_eqt[k], nu + nx, 0);
            gecp(nx + nu + 1, ng, jacobian.Gg_eqt[k], 0, 0, Ggt_stripe[0], 0, 0);
            rowin(nu + nx, 1.0, f, offset_u, hessian.RSQrqt[k], nu + nx, 0);
            gecp(nx + nu + 1, nu + nx, hessian.RSQrqt[k], 0, 0, RSQrqt_tilde[k], 0, 0);
        }
        else
        {
            const Index offset_eq_dyn = info.offsets_g_eq_dyn[k];
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index Hp1_size = gamma[k + 1] - rho[k + 1];
            if (Hp1_size + ng > nu + nx)
                return LinsolReturnFlag::NOFULL_RANK;
            gamma_k = Hp1_size + ng;
            // AL <- [BAb]^T_k P_kp1
            rowin(nxp1, 1.0, g, offset_eq_dyn, jacobian.BAbt[k], nu + nx, 0);
            gemm_nt(nu + nx + 1, nxp1, nxp1, 1.0, jacobian.BAbt[k], 0, 0, Ppt[k + 1], 0, 0, 0.0,
                    AL[0], 0, 0, AL[0], 0, 0);
            // AL[-1,:] <- AL[-1,:] + p_kp1^T
            gead(1, nxp1, 1.0, Ppt[k + 1], nxp1, 0, AL[0], nx + nu, 0);
            // RSQrqt_stripe <- AL[BA] + RSQrqt
            rowin(nu + nx, 1.0, f, offset_u, hessian.RSQrqt[k], nu + nx, 0);
            syrk_ln_mn(nu + nx + 1, nu + nx, nxp1, 1.0, AL[0], 0, 0, jacobian.BAbt[k], 0, 0, 1.0,
                       hessian.RSQrqt[k], 0, 0, RSQrqt_tilde[k], 0, 0);
            //// inequalities
            gamma[k] = gamma_k;
            // if ng[k]>0
            if (gamma_k > 0)
            {
                // if Gk nonempty
                if (ng > 0)
                {
                    // Ggt_stripe  <- Ggt_k
                    rowin(ng, 1.0, g, offset_eq_path, jacobian.Gg_eqt[k], nu + nx, 0);
                    gecp(nu + nx + 1, ng, jacobian.Gg_eqt[k], 0, 0, Ggt_stripe[0], 0, 0);
                }
                // if Hkp1 nonempty
                if (Hp1_size > 0)
                {
                    // Ggt_stripe <- [Ggt_k [BAb_k^T]H_kp1]
                    gemm_nt(nu + nx + 1, Hp1_size, nxp1, 1.0, jacobian.BAbt[k], 0, 0, Hh[k + 1], 0,
                            0, 0.0, Ggt_stripe[0], 0, ng, Ggt_stripe[0], 0, ng);
                    // Ggt_stripe[-1,ng:] <- Ggt_stripe[-1,ng:] + h_kp1^T
                    gead_transposed(1, Hp1_size, 1.0, Hh[k + 1], 0, nxp1, Ggt_stripe[0], nu + nx, ng);
                }
            }
            else
            {
                rho[k] = 0;
                rank_k = 0;
                RSQrq_hat_curr_p = &RSQrqt_tilde[k];
            }
        }
        // inequalities + inertia correction
        {
            if (ng_ineq > 0)
            {
                rowin(ng_ineq, 1.0, g, offset_eq_slack, jacobian.Gg_ineqt[k], nu + nx, 0);
                gecp(nu + nx + 1, ng_ineq, jacobian.Gg_ineqt[k], 0, 0, Ggt_ineq_temp[0], 0, 0);
                for (Index i = 0; i < ng_ineq; i++)
                {
                    Scalar scaling_factor = 1.0 / D_s(offset_ineq_k + i);
                    colsc(nu + nx + 1, scaling_factor, Ggt_ineq_temp[0], 0, i);
                }
                // add the penalty
                syrk_ln_mn(nu + nx + 1, nu + nx, ng_ineq, 1.0, Ggt_ineq_temp[0], 0, 0,
                           jacobian.Gg_ineqt[k], 0, 0, 1.0, RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k],
                           0, 0);
            }
            // inertia correction
            diaad(nu + nx, 1.0, D_x, offset_u, RSQrqt_tilde[k], 0, 0);
        }
        //////// TRANSFORM_AND_SUBSEQ
        {
            // symmetric transformation, done a little different than in paper, in order to fuse LA
            // operations LU_FACT_TRANSPOSE(Ggtstripe[:gamma_k, nu+nx+1], nu max) if(k==K-2)
            // blasfeo_print_dmat(1, gamma_k, Ggt_stripe[0], nu+nx, 0);
            auto start = std::chrono::steady_clock::now();
            // MatRealAllocated A_original(Ggt_stripe[0].m(), Ggt_stripe[0].n());
            // blasfeo_dgecp(Ggt_stripe[0].m(), Ggt_stripe[0].n(), &Ggt_stripe[0].mat(), 0, 0, &A_original.mat(), 0, 0);
            // std::cout << "Computing LU decomposition of:\n" << Ggt_stripe[0] << std::endl;
            if (k > 0 && k < info.dims.K - 1){
                int nx_next = info.dims.number_of_states[k+1];
                // 1. my approach
                /*
                MatRealAllocated temp(Ggt_stripe[0].m(), Ggt_stripe[0].n());
                gecp(Ggt_stripe[0].m(), Ggt_stripe[0].n(), Ggt_stripe[0], 0, 0, temp, 0, 0);
                fatrop_lu_fact_blocked_transposed(info.dims, k, &temp.mat(), true);
                std::cout << "LU decomposition avoiding additional perms:\n" << temp.block(nu+nx+1, gamma_k, 0, 0) << std::endl;
                // reset permutationmatrices
                for (Index i = 0; i < Pl1[k].size(); i++){Pl1[k][i] = i;}
                for (Index i = 0; i < Pl_rank[k].size(); i++){Pl_rank[k][i] = i;}
                for (Index i = 0; i < Pl2[k].size(); i++){Pl2[k][i] = i;}
                for (Index i = 0; i < Pr1[k].size(); i++){Pr1[k][i] = i;}
                for (Index i = 0; i < Pr2[k].size(); i++){Pr2[k][i] = i;}

                fatrop_lu_fact_blocked_transposed(info.dims, k, &Ggt_stripe[0].mat());
                std::cout << "LU decomposition with additional perms:\n" << Ggt_stripe[0].block(nu+nx+1, gamma_k, 0, 0) << std::endl;
                gead(Ggt_stripe[0].m(), Ggt_stripe[0].n(), -1, Ggt_stripe[0], 0, 0, temp, 0, 0);
                std::cout << "difference: \n" << temp.block(nu+nx+1, gamma_k, 0, 0) << std::endl;
                */

                // 1. Normal approach
                fatrop_lu_fact_blocked_transposed(info.dims, k, &Ggt_stripe[0].mat(), true);
                rank_k = rho[k];


                // 2. original approach
                // fatrop_lu_fact_transposed(gamma_k, nu + nx + 1, nu, 
                //         rank_k, &Ggt_stripe[0].mat(), Pl1[k], Pr1[k]);
                // rho1[k] = rank_k;

                // check if Tr^-T * A^T * Tl^-T = [-I, 0; 0, 0]
                /*
                MatRealAllocated At2(nu, gamma_k);
                gecp(nu, gamma_k, A_original, 0, 0, At2, 0, 0);
                // - apply Tl^-T
                apply_Pl_on_cols(Pl1[k], Pl_rank[k], Pl2[k], rho1[k], rho2[k], gamma_k, &At2.mat(), 0);
                blasfeo_dtrsm_runu(nu, rank_k, 1.0, &Ggt_stripe[0].mat(), 0, 0, 
                                  &At2.mat(), 0, 0, &At2.mat(), 0, 0); // L1^-T to first rank_k cols
                blasfeo_dgemm_nn(nu, gamma_k-rank_k, rank_k, -1.0, &At2.mat(), 0, 0, 
                                 &Ggt_stripe[0].mat(), 0, rank_k, 1.0, 
                                 &At2.mat(), 0, rank_k, &At2.mat(), 0, rank_k); // - L2^-T * L1^-T to last rank_k cols
                blasfeo_dtrsm_rlnn(nu, rank_k, -1.0, &Ggt_stripe[0].mat(), 0, 0, &At2.mat(), 0, 0, &At2.mat(), 0, 0); // *U1^-T to first rank_k cols
                // - apply Tr^-T
                apply_Pr_on_rows(Pr1[k], Pr2[k], rho1[k], rho2[k], &At2.mat());
                MatRealAllocated U1U2t(nu - rank_k, rank_k);
                gecp(nu - rank_k, rank_k, At2, rank_k, 0, U1U2t, 0, 0); // U2t in there
                if (nu - rank_k > 0){
                trsm_rlnn(nu-rank_k, rank_k, -1.0, At2, 0, 0, U1U2t, 0, 0, U1U2t, 0, 0); // U1^-T * U2t in there
                gemm_nn(nu - rank_k, gamma_k, rank_k, 1.0, U1U2t, 0, 0, At2, 0, 0, 1.0,
                        At2, 0, rank_k, At2, 0, rank_k); // - U2 * U1^-T * U2t in there
                }
                std::cout << "At after applying Tl^-T and Tr^-T:\n"; blasfeo_print_dmat(nu, gamma_k, &At2.mat(), 0, 0);
                */

                // bool verification = verify_blocked_lu_new(Ggt_stripe[0],
                //     A_original, Pl1[k], Pr1[k], rho1[k], Pl_rank[k], Pl2[k], 
                //     Pr2[k], rho2[k], info.dims.number_of_eq_constraints[k]-nx_next, 
                //     nu-nx_next, nx_next, gamma_k-info.dims.number_of_eq_constraints[k]);
                // // std::cout << "verification: " << verification << std::endl;
                // if (!verification){
                //     throw std::runtime_error("LU factorization verification failed");
                // }
            } else {
                fatrop_lu_fact_transposed(gamma_k, nu + nx + 1, nu, 
                        rho2[k], &Ggt_stripe[0].mat(), Pl2[k], Pr2[k]);
                rank_k = rho2[k];
                // bool verification = verify_blocked_lu_new(Ggt_stripe[0],
                //     A_original, Pl1[k], Pr1[k], rho1[k], Pl_rank[k], Pl2[k], 
                //     Pr2[k], rho2[k], gamma_k, nu, 0);
                // std::cout << "verification: " << verification << std::endl;
            }
            auto stop = std::chrono::steady_clock::now();
            duration_lu_factorization += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            // std::cout << duration_lu_factorization.count() << std::endl;

            rho[k] = rank_k;
            if (gamma_k - rank_k > 0)
            {
                // transfer eq's to next stage
                if (gamma_k - rank_k > nx)
                    return LinsolReturnFlag::NOFULL_RANK;
                getr(nx + 1, gamma_k - rank_k, Ggt_stripe[0], nu, rank_k, Hh[k], 0, 0);
                // std::cout << "copying over constraints:\n" << Hh[k] << "\n";
            }
            if (rank_k > 0)
            {
                // Ggt_tilde_k <- Ggt_stripe[rho_k:nu+nx+1, :rho] L-T (note that this is slightly
                // different from the implementation)
                trsm_rlnn(nu - rank_k + nx + 1, rank_k, -1.0, Ggt_stripe[0], 0, 0, Ggt_stripe[0],
                          rank_k, 0, Ggt_tilde[k], 0, 0);
                // the following command copies the top block matrix (LU) to the bottom because it
                // it needed later
                gecp(rank_k, gamma_k, Ggt_stripe[0], 0, 0, Ggt_tilde[k], nu - rank_k + nx + 1, 0);
                // permutations
                trtr_l(nu + nx, RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k], 0,
                       0); // copy lower part of RSQ to upper part
                
                // Pr[k].apply_on_rows(rank_k, &RSQrqt_tilde[k].mat()); // TODO make use of symmetry
                // Pr[k].apply_on_cols(rank_k, &RSQrqt_tilde[k].mat());
                apply_Pr_on_rows(Pr1[k], Pr2[k], rho1[k], rho2[k], &RSQrqt_tilde[k].mat());
                apply_Pr_on_cols(Pr1[k], Pr2[k], rho1[k], rho2[k], &RSQrqt_tilde[k].mat());

                // GL <- Ggt_tilde_k @ RSQ[:rho,:nu+nx] + RSQrqt[rho:nu+nx+1, rho:] (with
                // RSQ[:rho,:nu+nx] = RSQrqt[:nu+nx,:rho]^T) GEMM_NT(nu - rank_k + nx + 1, nu + nx,
                // rank_k, 1.0, Ggt_tilde[k], 0, 0, RSQrqt_tilde[k], 0, 0, 1.0, RSQrqt_tilde_p
                // + k, rank_k, 0, GgLt[0], 0, 0); split up because valgrind was giving invalid read
                // errors when C matrix has nonzero row offset GgLt[0].print();
                gecp(nu - rank_k + nx + 1, nu + nx, RSQrqt_tilde[k], rank_k, 0, GgLt[0], 0, 0);
                gemm_nt(nu - rank_k + nx + 1, nu + nx, rank_k, 1.0, Ggt_tilde[k], 0, 0,
                        RSQrqt_tilde[k], 0, 0, 1.0, GgLt[0], 0, 0, GgLt[0], 0, 0);
                // RSQrqt_hat = GgLt[nu-rank_k + nx +1, :rank_k] * G[:rank_k, :nu+nx] +
                // GgLt[rank_k:, :]  (with G[:rank_k,:nu+nx] = Gt[:nu+nx,:rank_k]^T)
                syrk_ln_mn(nu - rank_k + nx + 1, nu + nx - rank_k, rank_k, 1.0, GgLt[0], 0, 0,
                           Ggt_tilde[k], 0, 0, 1.0, GgLt[0], 0, rank_k, RSQrqt_hat[0], 0, 0);
                // GEMM_NT(nu - rank_k + nx + 1, nu + nx - rank_k, rank_k, 1.0, GgLt[0], 0, 0,
                // Ggt_tilde[k], 0, 0, 1.0, GgLt[0], 0, rank_k, RSQrqt_hat[0], 0, 0);
                RSQrq_hat_curr_p = &RSQrqt_hat[0];
            }
            else
            {
                RSQrq_hat_curr_p = &RSQrqt_tilde[k];
            }
        }
        //////// SCHUR
        {
            if (nu - rank_k > 0)
            {
                // DLlt_k = [chol(R_hatk); Llk@chol(R_hatk)^-T]
                potrf_l_mn(nu - rank_k + nx + 1, nu - rank_k, RSQrq_hat_curr_p[0], 0, 0, Llt[k], 0,
                           0);
                if (!check_reg(nu - rank_k, &Llt[k].mat(), 0, 0)){
                    return LinsolReturnFlag::INDEFINITE;
                }
                // Pp_k = Qq_hatk - L_k^T @ Ll_k
                // SYRK_LN_MN(nx+1, nx, nu-rank_k, -1.0,Llt_p+k, nu-rank_k,0, Llt_p+k,
                // nu-rank_k,0, 1.0, RSQrq_hat_curr[0], nu-rank_k, nu-rank_k,Pp+k,0,0); // feature
                // not implmented yet
                gecp(nx + 1, nu - rank_k, Llt[k], nu - rank_k, 0, Llt_shift[0], 0,
                     0); // needless operation because feature not implemented yet
                // SYRK_LN_MN(nx + 1, nx, nu - rank_k, -1.0, Llt_shift[0], 0, 0, Llt_shift[0], 0,
                // 0, 1.0, RSQrq_hat_curr[0], nu - rank_k, nu - rank_k, Ppt[k], 0, 0);
                gecp(nx + 1, nx, RSQrq_hat_curr_p[0], nu - rank_k, nu - rank_k, Ppt[k], 0, 0);
                syrk_ln_mn(nx + 1, nx, nu - rank_k, -1.0, Llt_shift[0], 0, 0, Llt_shift[0], 0, 0,
                           1.0, Ppt[k], 0, 0, Ppt[k], 0, 0);
                // next steps are for better accuracy
                if (increased_accuracy)
                {
                    // copy eta
                    getr(nu - rank_k, gamma_k - rank_k, Ggt_stripe[0], rank_k, rank_k,
                         Ggt_stripe[0], 0, 0);
                    // blasfeo_print_dmat(gamma_k-rank_k, nu-rank_k, Ggt_stripe[0], 0,0);
                    // eta L^-T
                    trsm_rltn(gamma_k - rank_k, nu - rank_k, 1.0, Llt[k], 0, 0, Ggt_stripe[0], 0, 0,
                              Ggt_stripe[0], 0, 0);
                    // ([S^T \\ r^T] L^-T) @ (L^-1 eta^T)
                    // (eta L^-T) @ ([S^T \\ r^T] L^-T)^T
                    gemm_nt(gamma_k - rank_k, nx + 1, nu - rank_k, -1.0, Ggt_stripe[0], 0, 0,
                            Llt[k], nu - rank_k, 0, 1.0, Hh[k], 0, 0, Hh[k], 0, 0);
                    // keep (L^-1 eta^T) for forward recursion
                    getr(gamma_k - rank_k, nu - rank_k, Ggt_stripe[0], 0, 0, Ggt_tilde[k], 0,
                         rank_k);
                }
            }
            else
            {
                gecp(nx + 1, nx, RSQrq_hat_curr_p[0], 0, 0, Ppt[k], 0, 0);
            }
            trtr_l(nx, Ppt[k], 0, 0, Ppt[k], 0, 0);
        }
    }
    auto intermediate_stop = std::chrono::high_resolution_clock::now();
    duration_backward_recursion += std::chrono::duration_cast<std::chrono::nanoseconds>(intermediate_stop - intermediate_start);
    intermediate_start = std::chrono::high_resolution_clock::now();

    rankI = 0;
    //////// FIRST_STAGE
    {
        const Index nx = info.dims.number_of_states[0];
        Index gamma_I = gamma[0] - rho[0];
        if (gamma_I > nx)
        {
            return LinsolReturnFlag::NOFULL_RANK;
        }
        if (gamma_I > 0)
        {
            getr(gamma_I, nx + 1, Hh[0], 0, 0, HhIt[0], 0, 0); // transposition may be avoided
            // HhIt[0].print();
            lu_fact_transposed(gamma_I, nx + 1, nx, rankI, HhIt[0], PlI[0], PrI[0], lu_fact_tol);
            if (rankI < gamma_I)
                return LinsolReturnFlag::NOFULL_RANK;
            // PpIt_tilde <- Ggt[rankI:nx+1, :rankI] L-T (note that this is slightly different from
            // the implementation)
            trsm_rlnn(nx - rankI + 1, rankI, -1.0, HhIt[0], 0, 0, HhIt[0], rankI, 0, GgIt_tilde[0],
                      0, 0);
            // permutations
            PrI[0].apply_on_rows(rankI, &Ppt[0].mat()); // TODO make use of symmetry
            PrI[0].apply_on_cols(rankI, &Ppt[0].mat());
            // // GL <- GgIt_tilde @ Pp[:rankI,:nx] + Ppt[rankI:nx+1, rankI:] (with Pp[:rankI,:nx] =
            // Ppt[:nx,:rankI]^T) GEMM_NT(nx - rankI + 1, nx, rankI, 1.0, GgIt_tilde[0], 0, 0,
            // Ppt[0], 0, 0, 1.0, Ppt[0], rankI, 0, GgLIt[0], 0, 0); split up because valgrind was
            // giving invalid read errors when C matrix has nonzero row offset
            gecp(nx - rankI + 1, nx, Ppt[0], rankI, 0, GgLIt[0], 0, 0);
            gemm_nt(nx - rankI + 1, nx, rankI, 1.0, GgIt_tilde[0], 0, 0, Ppt[0], 0, 0, 1.0,
                    GgLIt[0], 0, 0, GgLIt[0], 0, 0);
            // // RSQrqt_hat = GgLt[nu-rank_k + nx +1, :rank_k] * G[:rank_k, :nu+nx] + GgLt[rank_k:,
            // :]  (with G[:rank_k,:nu+nx] = Gt[:nu+nx,:rank_k]^T)
            syrk_ln_mn(nx - rankI + 1, nx - rankI, rankI, 1.0, GgLIt[0], 0, 0, GgIt_tilde[0], 0, 0,
                       1.0, GgLIt[0], 0, rankI, PpIt_hat[0], 0, 0);
            // TODO skipped if nx-rankI = 0
            potrf_l_mn(nx - rankI + 1, nx - rankI, PpIt_hat[0], 0, 0, LlIt[0], 0, 0);
            if (!check_reg(nx - rankI, &LlIt[0].mat(), 0, 0)){
                return LinsolReturnFlag::INDEFINITE;
            }
        }
        else
        {
            rankI = 0;
            potrf_l_mn(nx + 1, nx, Ppt[0], 0, 0, LlIt[0], 0, 0);
            if (!check_reg(nx, &LlIt[0].mat(), 0, 0)){
                return LinsolReturnFlag::INDEFINITE;
            }
        }
    }
    ////// FORWARD_SUBSTITUTION:
    // first stage
    {
        const Index nx = info.dims.number_of_states[0];
        const Index nu = info.dims.number_of_controls[0];
        const Index offs_u = info.offsets_primal_u[0];
        const Index offs_x = info.offsets_primal_x[0];
        const Index offs_g = info.offsets_g_eq_path[0];
        // calculate xIb
        rowex(nx - rankI, -1.0, LlIt[0], nx - rankI, 0, x, offs_x + rankI);
        // assume TRSV_LTN allows aliasing, this is the case in normal BLAS
        trsv_ltn(nx - rankI, LlIt[0], 0, 0, x, offs_x + rankI, x, offs_x + rankI);
        // calculate xIa
        rowex(rankI, 1.0, GgIt_tilde[0], nx - rankI, 0, x, offs_x);
        // assume aliasing is possible for last two elements
        gemv_t(nx - rankI, rankI, 1.0, GgIt_tilde[0], 0, 0, x, offs_x + rankI, 1.0, x, offs_x, x,
               offs_x);
        //// lag
        rowex(rankI, -1.0, Ppt[0], nx, 0, eq_mult, offs_g);
        // assume aliasing is possible for last two elements
        gemv_t(nx, rankI, -1.0, Ppt[0], 0, 0, x, offs_x, 1.0, eq_mult, offs_g, eq_mult, offs_g);

        // U^-T
        trsv_lnn(rankI, HhIt[0], 0, 0, eq_mult, offs_g, eq_mult, offs_g);
        // L^-T
        trsv_unu(rankI, rankI, HhIt[0], 0, 0, eq_mult, offs_g, eq_mult, offs_g);
        PlI[0].apply_inverse(rankI, &eq_mult.vec(), offs_g);
        PrI[0].apply_inverse(rankI, &x.vec(), offs_x);
    }
    intermediate_stop = std::chrono::high_resolution_clock::now();
    duration_initial_stage += std::chrono::duration_cast<std::chrono::nanoseconds>(intermediate_stop - intermediate_start);
    intermediate_start = std::chrono::high_resolution_clock::now();
    // std::cout << "------------------------------" << std::endl;
    // std::cout << "First stage:\n";
    // PrintNpArray(Ppt[0], "\tPpt");
    // PrintNpArray(Hh[0], "\tHh");
    // std::cout << "Solution to first stage:\n";
    // std::cout << "\tx0: " << x.block(info.dims.number_of_states[0], info.offsets_primal_x[0])<< std::endl;
    // std::cout << "\tu0: " << x.block(info.dims.number_of_controls[0], info.offsets_primal_u[0])<< std::endl;
    // std::cout << "\tnu0: " << eq_mult.block(gamma[0] - rho[0], info.offsets_g_eq_path[0])<< std::endl; 
    // std::cout << "------------------------------" << std::endl;
    // other stages
    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_x = info.offsets_primal_x[k];
        const Index rho_k = rho[k];
        const Index numrho_k = nu - rho_k;
        const Index offs_g_k = info.offsets_g_eq_path[k];
        const Index gammamrho_k = gamma[k] - rho[k];
        const Index gamma_k = gamma[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_eq_ineq = info.offsets_g_eq_slack[k];
        const Index offs_slack = info.offsets_slack[k];
        if (numrho_k > 0)
        {
            /// calculate ukb_tilde
            // -Lkxk - lk
            rowex(numrho_k, -1.0, Llt[k], numrho_k + nx, 0, x, offs + rho_k);
            if (increased_accuracy)
            {
                gemv_n(nu - rho_k, gamma_k - rho_k, -1.0, Ggt_tilde[k], 0, rho_k, eq_mult, offs_g_k,
                       1.0, x, offs + rho_k, x, offs + rho_k);
            }
            // assume aliasing of last two eliments is allowed
            gemv_t(nx, numrho_k, -1.0, Llt[k], numrho_k, 0, x, offs_x, 1.0, x, offs + rho_k, x,
                   offs + rho_k);
            trsv_ltn(numrho_k, Llt[k], 0, 0, x, offs + rho_k, x, offs + rho_k);
        }
        /// calcualate uka_tilde
        if (rho_k > 0)
        {
            rowex(rho_k, 1.0, Ggt_tilde[k], numrho_k + nx, 0, x, offs);
            gemv_t(nx + numrho_k, rho_k, 1.0, Ggt_tilde[k], 0, 0, x, offs + rho_k, 1.0, x, offs, x,
                   offs);
            // calculate lamda_tilde_k
            // copy vk to right location
            veccp(gammamrho_k, eq_mult, offs_g_k, v_tmp[0], 0);
            veccp(gammamrho_k, v_tmp[0], 0, eq_mult, offs_g_k + rho_k);
            rowex(rho_k, -1.0, RSQrqt_tilde[k], nu + nx, 0, eq_mult, offs_g_k);
            // assume aliasing of last two eliments is allowed
            gemv_t(nu + nx, rho_k, -1.0, RSQrqt_tilde[k], 0, 0, x, offs, 1.0, eq_mult, offs_g_k,
                   eq_mult, offs_g_k);
            // nu-rank_k+nx,0
            // needless copy because feature not implemented yet in trsv_lnn
            gecp(rho_k, gamma_k, Ggt_tilde[k], nu - rho_k + nx + 1, 0, AL[0], 0, 0);
            // U^-T
            trsv_lnn(rho_k, AL[0], 0, 0, eq_mult, offs_g_k, eq_mult, offs_g_k);
            // L^-T
            trsv_unu(rho_k, gamma_k, AL[0], 0, 0, eq_mult, offs_g_k, eq_mult, offs_g_k);
            
            // Pl[k].apply_inverse(rho_k, &eq_mult.vec(), offs_g_k);
            // Pr[k].apply_inverse(rho_k, &x.vec(), offs);
            apply_Pl_inverse(Pl1[k], Pl_rank[k], Pl2[k], rho1[k], rho2[k], gamma_k, &eq_mult.vec(), offs_g_k);
            apply_Pr_inverse(Pr1[k], Pr2[k], rho1[k], rho2[k], &x.vec(), offs);
        }
        if (ng_ineq > 0)
        {
            gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_eq_ineq,
                   eq_mult, offs_eq_ineq);
            eq_mult.block(ng_ineq, offs_eq_ineq) =
                eq_mult.block(ng_ineq, offs_eq_ineq) / D_s.block(ng_ineq, offs_slack);
        }
        if (k != info.dims.K - 1)
        {
            const Index offs_dyn_eq_k = info.offsets_g_eq_dyn[k];
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index nup1 = info.dims.number_of_controls[k + 1];
            const Index offsp1 = info.offsets_primal_u[k + 1];
            const Index offsxp1 = info.offsets_primal_x[k + 1];
            const Index offs_g_kp1 = info.offsets_g_eq_path[k + 1];
            const Index gammamrho_kp1 = gamma[k + 1] - rho[k + 1];
            // calculate xkp1
            rowex(nxp1, 1.0, jacobian.BAbt[k], nu + nx, 0, x, offsxp1);
            gemv_t(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, x, offs, 1.0, x, offsxp1, x,
                   offsxp1);
            // calculate lam_dyn xp1
            rowex(nxp1, 1.0, Ppt[k + 1], nxp1, 0, eq_mult, offs_dyn_eq_k);
            gemv_t(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, x, offsxp1, 1.0, eq_mult, offs_dyn_eq_k,
                   eq_mult, offs_dyn_eq_k);
            gemv_t(gammamrho_kp1, nxp1, 1.0, Hh[k + 1], 0, 0, eq_mult, offs_g_kp1, 1.0, eq_mult,
                   offs_dyn_eq_k, eq_mult, offs_dyn_eq_k);
        }
    }
    intermediate_stop = std::chrono::high_resolution_clock::now();
    duration_forward_recursion += std::chrono::duration_cast<std::chrono::nanoseconds>(intermediate_stop - intermediate_start);
    return LinsolReturnFlag::SUCCESS;
}
LinsolReturnFlag AugSystemSolver<AcceleratedOcpType>::solve(const ProblemInfo &info,
                                           Jacobian<AcceleratedOcpType> &jacobian, Hessian<AcceleratedOcpType> &hessian,
                                           const VecRealView &D_x, const VecRealView &D_eq,
                                           const VecRealView &D_s, const VecRealView &f,
                                           const VecRealView &g, VecRealView &x,
                                           VecRealView &eq_mult)
{
    MatRealView *RSQrq_hat_curr_p;
    for (Index k = info.dims.K - 1; k >= 0; --k)
    {
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_ineq_k = info.offsets_slack[k];
        const Index offset_u = info.offsets_primal_u[k];
        const Index offset_eq_k = info.offsets_eq[k];
        const Index offset_g_eq_k = info.offsets_g_eq_path[k];
        const Index offset_g_ineq_k = info.offsets_g_eq_slack[k];
        // const fatrop_int offs_g_ineq_k = offs_g_ineq_p[k];
        //////// SUBSDYN
        if (k == info.dims.K - 1)
        {
            rowin(nu + nx, 1.0, f, offset_u, hessian.RSQrqt[k], nu + nx, 0);
            gecp(nx + nu + 1, nu + nx, hessian.RSQrqt[k], 0, 0, RSQrqt_tilde[k], 0, 0);
        }
        else
        {
            const Index offset_eq_dyn = info.offsets_g_eq_dyn[k];
            const Index nxp1 = info.dims.number_of_states[k + 1];
            // AL <- [BAb]^T_k P_kp1
            rowin(nxp1, 1.0, g, offset_eq_dyn, jacobian.BAbt[k], nu + nx, 0);
            gemm_nt(nu + nx + 1, nxp1, nxp1, 1.0, jacobian.BAbt[k], 0, 0, Ppt[k + 1], 0, 0, 0.0,
                    AL[0], 0, 0, AL[0], 0, 0);
            // AL[-1,:] <- AL[-1,:] + p_kp1^T
            gead(1, nxp1, 1.0, Ppt[k + 1], nxp1, 0, AL[0], nx + nu, 0);
            // RSQrqt_stripe <- AL[BA] + RSQrqt
            rowin(nu + nx, 1.0, f, offset_u, hessian.RSQrqt[k], nu + nx, 0);
            syrk_ln_mn(nu + nx + 1, nu + nx, nxp1, 1.0, AL[0], 0, 0, jacobian.BAbt[k], 0, 0, 1.0,
                       hessian.RSQrqt[k], 0, 0, RSQrqt_tilde[k], 0, 0);
        }
        // equality penalty
        {
            rowin(ng, 1.0, g, offset_g_eq_k, jacobian.Gg_eqt[k], nu + nx, 0);
            gecp(nu + nx + 1, ng, jacobian.Gg_eqt[k], 0, 0, Ggt_stripe[0], 0, 0);
            for (Index i = 0; i < ng; i++)
            {
                Scalar scaling_factor = 1.0 / D_eq(offset_eq_k + i);
                colsc(nu + nx + 1, scaling_factor, Ggt_stripe[0], 0, i);
            }
            // add the penalty
            syrk_ln_mn(nu + nx + 1, nu + nx, ng, 1.0, Ggt_stripe[0], 0, 0, jacobian.Gg_eqt[k], 0, 0,
                       1.0, RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k], 0, 0);
        }
        // inequalities + inertia correction
        {
            if (ng_ineq > 0)
            {
                rowin(ng_ineq, 1.0, g, offset_g_ineq_k, jacobian.Gg_ineqt[k], nu + nx, 0);
                gecp(nu + nx + 1, ng_ineq, jacobian.Gg_ineqt[k], 0, 0, Ggt_ineq_temp[0], 0, 0);
                for (Index i = 0; i < ng_ineq; i++)
                {
                    Scalar scaling_factor = 1.0 / D_s(offs_ineq_k + i);
                    colsc(nu + nx + 1, scaling_factor, Ggt_ineq_temp[0], 0, i);
                }
                // add the penalty
                syrk_ln_mn(nu + nx + 1, nu + nx, ng_ineq, 1.0, Ggt_ineq_temp[0], 0, 0,
                           jacobian.Gg_ineqt[k], 0, 0, 1.0, RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k],
                           0, 0);
            }
            // inertia correction
            diaad(nu + nx, 1.0, D_x, offset_u, RSQrqt_tilde[k], 0, 0);
        }

        //////// TRANSFORM_AND_SUBSEQ
        {
            RSQrq_hat_curr_p = &RSQrqt_tilde[k];
        }
        //////// SCHUR
        {
            // DLlt_k = [chol(R_hatk); Llk@chol(R_hatk)^-T]
            potrf_l_mn(nu + nx + 1, nu, *RSQrq_hat_curr_p, 0, 0, Llt[k], 0, 0);
            if (!check_reg(nu, &Llt[k].mat(), 0, 0)){
                return LinsolReturnFlag::INDEFINITE;
            }
            // Pp_k = Qq_hatk - L_k^T @ Ll_k
            // SYRK_LN_MN(nx+1, nx, nu-rank_k, -1.0,Llt_p+k, nu-rank_k,0, Llt_p+k, nu-rank_k,0, 1.0,
            // RSQrq_hat_curr_p, nu-rank_k, nu-rank_k,Pp+k,0,0); // feature not implmented yet
            gecp(nx + 1, nu, Llt[k], nu, 0, Llt_shift[0], 0,
                 0); // needless operation because feature not implemented yet
            syrk_ln_mn(nx + 1, nx, nu, -1.0, Llt_shift[0], 0, 0, Llt_shift[0], 0, 0, 1.0,
                       *RSQrq_hat_curr_p, nu, nu, Ppt[k], 0, 0);
        }
        trtr_l(nx, Ppt[k], 0, 0, Ppt[k], 0, 0);
    }
    //////// FIRST_STAGE
    {
        const Index nx = info.dims.number_of_states[0];
        {
            potrf_l_mn(nx + 1, nx, Ppt[0], 0, 0, LlIt[0], 0, 0);
            if (!check_reg(nx, &LlIt[0].mat(), 0, 0)){
                return LinsolReturnFlag::INDEFINITE;
            }
        }
    }
    ////// FORWARD_SUBSTITUTION:
    // first stage
    {
        const Index nx = info.dims.number_of_states[0];
        const Index nu = info.dims.number_of_controls[0];
        const Index offs_x = info.offsets_primal_x[0];
        // calculate xIb
        rowex(nx, -1.0, LlIt[0], nx, 0, x, offs_x);
        // assume TRSV_LTN allows aliasing, this is the case in normal BLAS
        trsv_ltn(nx, LlIt[0], 0, 0, x, offs_x, x, offs_x);
    }
    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_x = info.offsets_primal_x[k];
        rowex(nu, -1.0, Llt[k], nu + nx, 0, x, offs);
        gemv_t(nx, nu, -1.0, Llt[k], nu, 0, x, offs_x, 1.0, x, offs, x, offs);
        trsv_ltn(nu, Llt[k], 0, 0, x, offs, x, offs);
        if (k != info.dims.K - 1)
        {
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index nup1 = info.dims.number_of_controls[k + 1];
            const Index offs_x_p1 = info.offsets_primal_x[k + 1];
            const Index offs_dyn_eq_k = info.offsets_g_eq_dyn[k];
            // calculate xkp1
            rowex(nxp1, 1.0, jacobian.BAbt[k], nu + nx, 0, x, offs_x_p1);
            gemv_t(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, x, offs, 1.0, x, offs_x_p1, x,
                   offs_x_p1);
            // calculate lam_dyn xp1
            rowex(nxp1, 1.0, Ppt[k + 1], nxp1, 0, eq_mult, offs_dyn_eq_k);
            gemv_t(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, x, offs_x_p1, 1.0, eq_mult, offs_dyn_eq_k,
                   eq_mult, offs_dyn_eq_k);
        }
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index offs_g_eq_k = info.offsets_g_eq_path[k];
        const Index offs_eq_k = info.offsets_eq[k];
        if (ng > 0)
        {
            gemv_t(nu + nx, ng, 1.0, jacobian.Gg_eqt[k], 0, 0, x, offs, 1.0, g, offs_g_eq_k,
                   eq_mult, offs_g_eq_k);
            eq_mult.block(ng, offs_g_eq_k) =
                eq_mult.block(ng, offs_g_eq_k) / D_eq.block(ng, offs_eq_k);
        }
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_slack = info.offsets_slack[k];
        const Index offs_eq_ineq = info.offsets_g_eq_slack[k];
        if (ng_ineq > 0)
        {
            gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_eq_ineq,
                   eq_mult, offs_eq_ineq);
            eq_mult.block(ng_ineq, offs_eq_ineq) =
                eq_mult.block(ng_ineq, offs_eq_ineq) / D_s.block(ng_ineq, offs_slack);
        }
    }
    return LinsolReturnFlag::SUCCESS;
}

LinsolReturnFlag AugSystemSolver<AcceleratedOcpType>::solve_rhs(const ProblemInfo &info,
                                               const Jacobian<AcceleratedOcpType> &jacobian,
                                               const Hessian<AcceleratedOcpType> &hessian,
                                               const VecRealView &D_s, const VecRealView &f,
                                               const VecRealView &g, VecRealView &x,
                                               VecRealView &eq_mult)
{
    VecRealView *v_RSQrq_hat_curr_p;
    Index rank_k;
    /////////////// recursion ///////////////

    for (Index k = info.dims.K - 1; k >= 0; --k)
    {
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offset_ineq_k = info.offsets_slack[k];
        const Index offs_g_ineq_k = info.offsets_g_eq_slack[k];
        const Index offs_g_k = info.offsets_g_eq_path[k];
        const Index offs = info.offsets_primal_u[k];
        //         //////// SUBSDYN
        Index gamma_k;
        if (k == info.dims.K - 1)
        {
            gamma_k = ng;
            gamma[k] = gamma_k;
            veccp(ng, g, offs_g_k, v_Ggt_stripe[0], 0);
            veccp(nu + nx, f, offs, v_RSQrqt_tilde[k], 0);
        }
        else
        {
            const Index offs_dyn_k = info.offsets_g_eq_dyn[k];
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index Hp1_size = gamma[k + 1] - rho[k + 1];
            gamma_k = Hp1_size + ng;
            gemv_n(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, g, offs_dyn_k, 0.0, v_AL[0], 0, v_AL[0], 0);
            axpy(nxp1, 1.0, v_Ppt[k + 1], 0, v_AL[0], 0, v_AL[0], 0);
            gemv_n(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, v_AL[0], 0, 1.0, f, offs,
                   v_RSQrqt_tilde[k], 0);
            if (gamma_k > 0)
            {
                if (ng > 0)
                {
                    veccp(ng, g, offs_g_k, v_Ggt_stripe[0], 0);
                }
                if (Hp1_size > 0)
                {
                    gemv_n(Hp1_size, nxp1, 1.0, Hh[k + 1], 0, 0, g, offs_dyn_k, 0.0,
                           v_Ggt_stripe[0], ng, v_Ggt_stripe[0], ng);
                    axpy(Hp1_size, 1.0, v_Hh[k + 1], 0, v_Ggt_stripe[0], ng, v_Ggt_stripe[0], ng);
                }
            }
            else
            {
                rank_k = 0;
                v_RSQrq_hat_curr_p = &v_RSQrqt_tilde[k];
            }
        }
        if (ng_ineq > 0)
        {
            for (Index i = 0; i < ng_ineq; i++)
            {
                Scalar scaling_factor = D_s(offset_ineq_k + i);
                Scalar grad_barrier = g(offs_g_ineq_k + i);
                v_Ggt_ineq_temp[0](i) = grad_barrier / scaling_factor;
            }
            gemv_n(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, v_Ggt_ineq_temp[0], 0, 1.0,
                   v_RSQrqt_tilde[k], 0, v_RSQrqt_tilde[k], 0);
        }
        {
            rank_k = rho[k];
            gecp(rank_k, gamma_k, Ggt_tilde[k], nu - rank_k + nx + 1, 0, Ggt_stripe[0], 0, 0);
            
            // Pl[k].apply(rank_k, &v_Ggt_stripe[0].vec(), 0);
            apply_Pl(Pl1[k], Pl_rank[k], Pl2[k], rho1[k], rho2[k], gamma_k, &v_Ggt_stripe[0].vec(), 0);
            
            trsv_utu(rank_k, Ggt_stripe[0], 0, 0, v_Ggt_stripe[0], 0, v_Ggt_stripe[0], 0);
            gemv_t(rank_k, gamma_k - rank_k, -1.0, Ggt_stripe[0], 0, rank_k, v_Ggt_stripe[0], 0,
                   1.0, v_Ggt_stripe[0], rank_k, v_Ggt_stripe[0], rank_k);

            if (gamma_k - rank_k > 0)
            {
                veccp(gamma_k - rank_k, v_Ggt_stripe[0], rank_k, v_Hh[k], 0);
            }
            if (rank_k > 0)
            {
                veccpsc(rank_k, -1.0, v_Ggt_stripe[0], 0, v_Ggt_tilde[k], 0);
                trsv_ltn(rank_k, Ggt_stripe[0], 0, 0, v_Ggt_tilde[k], 0, v_Ggt_tilde[k], 0);
                
                // Pr[k].apply(rank_k, &v_RSQrqt_tilde[k].vec(), 0);
                apply_Pr(Pr1[k], Pr2[k], rho1[k], rho2[k], &v_RSQrqt_tilde[k].vec(), 0);
                
                veccp(nu + nx, v_RSQrqt_tilde[k], 0, v_GgLt[0], 0);
                gemv_n(nu + nx, rank_k, 1.0, RSQrqt_tilde[k], 0, 0, v_Ggt_tilde[k], 0, 1.0,
                       v_GgLt[0], 0, v_GgLt[0], 0);
                gemv_n(nu + nx - rank_k, rank_k, 1.0, Ggt_tilde[k], 0, 0, v_GgLt[0], 0, 1.0,
                       v_GgLt[0], rank_k, v_RSQrqt_hat[0], 0);
                v_RSQrq_hat_curr_p = &v_RSQrqt_hat[0];
            }
            else
            {
                v_RSQrq_hat_curr_p = &v_RSQrqt_tilde[k];
            }
        }
        //         //////// SCHUR
        {
            if (nu - rank_k > 0)
            {
                trsv_lnn(nu - rank_k, Llt[k], 0, 0, *v_RSQrq_hat_curr_p, 0, v_Llt[k], 0);
                gecp(nx + 1, nu - rank_k, Llt[k], nu - rank_k, 0, Llt_shift[0], 0, 0);
                veccp(nu - rank_k, v_Llt[k], 0, v_Llt_shift[0], 0);
                veccp(nx, *v_RSQrq_hat_curr_p, nu - rank_k, v_Ppt[k], 0);
                gemv_n(nx, nu - rank_k, -1.0, Llt_shift[0], 0, 0, v_Llt_shift[0], 0, 1.0, v_Ppt[k],
                       0, v_Ppt[k], 0);
                if (increased_accuracy)
                {
                    gemv_t(nu - rank_k, gamma_k - rank_k, -1.0, Ggt_tilde[k], 0, rank_k, v_Llt[k],
                           0, 1.0, v_Hh[k], 0, v_Hh[k], 0);
                }
            }
            else
            {
                veccp(nx, *v_RSQrq_hat_curr_p, 0, v_Ppt[k], 0);
            }
        }
    }
    {
        const Index nx = info.dims.number_of_states[0];
        Index gamma_I = gamma[0] - rho[0];
        if (gamma_I > 0)
        {
            veccp(gamma_I, v_Hh[0], 0, v_HhIt[0], 0);
            PlI[0].apply(rankI, &v_HhIt[0].vec(), 0);
            trsv_utu(rankI, HhIt[0], 0, 0, v_HhIt[0], 0, v_HhIt[0], 0);
            gemv_t(rankI, gamma_I - rankI, -1.0, HhIt[0], 0, rankI, v_HhIt[0], 0, 1.0, v_HhIt[0],
                   rankI, v_HhIt[0], rankI);
            veccpsc(rankI, -1.0, v_HhIt[0], 0, v_GgIt_tilde[0], 0);
            trsv_ltn(rankI, HhIt[0], 0, 0, v_GgIt_tilde[0], 0, v_GgIt_tilde[0], 0);
            PrI[0].apply(rankI, &v_Ppt[0].vec(), 0);
            veccp(nx, v_Ppt[0], 0, v_GgLIt[0], 0);
            gemv_n(nx, rankI, 1.0, Ppt[0], 0, 0, v_GgIt_tilde[0], 0, 1.0, v_GgLIt[0], 0, v_GgLIt[0],
                   0);
            gemv_n(nx - rankI, rankI, 1.0, GgIt_tilde[0], 0, 0, v_GgLIt[0], 0, 1.0, v_GgLIt[0],
                   rankI, v_PpIt_hat[0], 0);
            trsv_lnn(nx - rankI, LlIt[0], 0, 0, v_PpIt_hat[0], 0, v_LlIt[0], 0);
        }
        else
        {
            trsv_lnn(nx, LlIt[0], 0, 0, v_Ppt[0], 0, v_LlIt[0], 0);
        }
    }
    {
        const Index nx = info.dims.number_of_states[0];
        const Index nu = info.dims.number_of_controls[0];
        const Index offs_u = info.offsets_primal_u[0];
        const Index offs_x = info.offsets_primal_x[0];
        const Index offs_g = info.offsets_g_eq_path[0];
        veccpsc(nx - rankI, -1.0, v_LlIt[0], 0, x, offs_x + rankI);
        trsv_ltn(nx - rankI, LlIt[0], 0, 0, x, offs_x + rankI, x, offs_x + rankI);
        veccp(rankI, v_GgIt_tilde[0], 0, x, offs_x);
        gemv_t(nx - rankI, rankI, 1.0, GgIt_tilde[0], 0, 0, x, offs_x + rankI, 1.0, x, offs_x, x,
               offs_x);
        veccpsc(rankI, -1.0, v_Ppt[0], 0, eq_mult, offs_g);
        gemv_t(nx, rankI, -1.0, Ppt[0], 0, 0, x, nu, 1.0, eq_mult, offs_g, eq_mult, offs_g);
        trsv_lnn(rankI, HhIt[0], 0, 0, eq_mult, offs_g, eq_mult, offs_g);
        trsv_unu(rankI, rankI, HhIt[0], 0, 0, eq_mult, offs_g, eq_mult, offs_g);
        PlI[0].apply_inverse(rankI, &eq_mult.vec(), offs_g);
        PrI[0].apply_inverse(rankI, &x.vec(), offs_x);
    }
    for (Index k = 0; k < info.dims.K; k++)
    {

        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_x = info.offsets_primal_x[k];
        const Index rho_k = rho[k];
        const Index numrho_k = nu - rho_k;
        const Index offs_g_k = info.offsets_g_eq_path[k];
        const Index gammamrho_k = gamma[k] - rho[k];
        const Index gamma_k = gamma[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_eq_ineq = info.offsets_g_eq_slack[k];
        const Index offs_slack = info.offsets_slack[k];
        if (numrho_k > 0)
        {
            veccpsc(numrho_k, -1.0, v_Llt[k], 0, x, offs + rho_k);
            if (increased_accuracy)
            {
                gemv_n(nu - rho_k, gamma_k - rho_k, -1.0, Ggt_tilde[k], 0, rho_k, eq_mult, offs_g_k,
                       1.0, x, offs + rho_k, x, offs + rho_k);
            }
            gemv_t(nx, numrho_k, -1.0, Llt[k], numrho_k, 0, x, offs_x, 1.0, x, offs + rho_k, x,
                   offs + rho_k);
            trsv_ltn(numrho_k, Llt[k], 0, 0, x, offs + rho_k, x, offs + rho_k);
        }
        //         /// calcualate uka_tilde
        if (rho_k > 0)
        {
            // ROWEX(rho_k, 1.0, Ggt_tilde[k], numrho_k + nx, 0, ux[0], offs);
            veccp(rho_k, v_Ggt_tilde[k], 0, x, offs);
            gemv_t(nx + numrho_k, rho_k, 1.0, Ggt_tilde[k], 0, 0, x, offs + rho_k, 1.0, x, offs, x,
                   offs);
            veccp(gammamrho_k, eq_mult, offs_g_k, v_tmp[0], 0);
            veccp(gammamrho_k, v_tmp[0], 0, eq_mult, offs_g_k + rho_k);
            veccpsc(rho_k, -1.0, v_RSQrqt_tilde[k], 0, eq_mult, offs_g_k);
            gemv_t(nu + nx, rho_k, -1.0, RSQrqt_tilde[k], 0, 0, x, offs, 1.0, eq_mult, offs_g_k,
                   eq_mult, offs_g_k);
            gecp(rho_k, gamma_k, Ggt_tilde[k], nu - rho_k + nx + 1, 0, AL[0], 0, 0);
            trsv_lnn(rho_k, AL[0], 0, 0, eq_mult, offs_g_k, eq_mult, offs_g_k);
            trsv_unu(rho_k, gamma_k, AL[0], 0, 0, eq_mult, offs_g_k, eq_mult, offs_g_k);

            // Pl[k].apply_inverse(rho_k, &eq_mult.vec(), offs_g_k);
            // Pr[k].apply_inverse(rho_k, &x.vec(), offs);
            apply_Pl_inverse(Pl1[k], Pl_rank[k], Pl2[k], rho1[k], rho2[k], gamma_k, &eq_mult.vec(), offs_g_k);
            apply_Pr_inverse(Pr1[k], Pr2[k], rho1[k], rho2[k], &x.vec(), offs);
        }
        if (ng_ineq > 0)
        {
            gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_eq_ineq,
                   eq_mult, offs_eq_ineq);
            eq_mult.block(ng_ineq, offs_eq_ineq) =
                eq_mult.block(ng_ineq, offs_eq_ineq) / D_s.block(ng_ineq, offs_slack);
        }
        if (k != info.dims.K - 1)
        {
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index nup1 = info.dims.number_of_controls[k + 1];
            const Index offsp1 = info.offsets_primal_u[k + 1];
            const Index offsxp1 = info.offsets_primal_x[k + 1];
            const Index offs_g_kp1 = info.offsets_g_eq_path[k + 1];
            const Index offs_dyn_k = info.offsets_g_eq_dyn[k];
            const Index gammamrho_kp1 = gamma[k + 1] - rho[k + 1];
            veccp(nxp1, g, offs_dyn_k, x, offsxp1);
            gemv_t(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, x, offs, 1.0, x, offsxp1, x,
                   offsxp1);
            veccp(nxp1, v_Ppt[k + 1], 0, eq_mult, offs_dyn_k);
            gemv_t(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, x, offsxp1, 1.0, eq_mult, offs_dyn_k, eq_mult,
                   offs_dyn_k);
            gemv_t(gammamrho_kp1, nxp1, 1.0, Hh[k + 1], 0, 0, eq_mult, offs_g_kp1, 1.0, eq_mult,
                   offs_dyn_k, eq_mult, offs_dyn_k);
        }
    }
    return LinsolReturnFlag::SUCCESS;
}
LinsolReturnFlag AugSystemSolver<AcceleratedOcpType>::solve_rhs(const ProblemInfo &info,
                                               const Jacobian<AcceleratedOcpType> &jacobian,
                                               const Hessian<AcceleratedOcpType> &hessian,
                                               const VecRealView &D_eq, const VecRealView &D_s,
                                               const VecRealView &f, const VecRealView &g,
                                               VecRealView &x, VecRealView &eq_mult)
{
    VecRealView *v_RSQrq_hat_curr_p;
    for (Index k = info.dims.K - 1; k >= 0; --k)
    {
        const Index offs_ux_k = info.offsets_primal_u[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_g_dyn = info.offsets_g_eq_dyn[k];
        const Index offs_g_eq = info.offsets_g_eq_path[k];
        const Index offs_ge_eq_ineq = info.offsets_g_eq_slack[k];
        //     //////// SUBSDYN
        if (k == info.dims.K - 1)
        {
            veccp(nu + nx, f, offs_ux_k, v_RSQrqt_tilde[k], 0);
        }
        else
        {
            const Index nxp1 = info.dims.number_of_states[k + 1];
            gemv_n(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, g, offs_g_dyn, 0.0, v_AL[0], 0, v_AL[0], 0);
            axpy(nxp1, 1.0, v_Ppt[k + 1], 0, v_AL[0], 0, v_AL[0], 0);
            gemv_n(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, v_AL[0], 0, 1.0, f, offs_ux_k,
                   v_RSQrqt_tilde[k], 0);
        }
        if (ng > 0)
        {
            const Index offs_eq_k = info.offsets_eq[k];
            for (Index i = 0; i < ng; i++)
            {
                Scalar scaling_factor = D_eq(offs_eq_k + i);
                v_Ggt_stripe[0](i) = g(offs_g_eq + i) / scaling_factor;
            }
            gemv_n(nu + nx, ng, 1.0, jacobian.Gg_eqt[k], 0, 0, v_Ggt_stripe[0], 0, 1.0,
                   v_RSQrqt_tilde[k], 0, v_RSQrqt_tilde[k], 0);
        }
        if (ng_ineq > 0)
        {
            const Index offs_ineq_k = info.offsets_slack[k];
            for (Index i = 0; i < ng_ineq; i++)
            {
                Scalar scaling_factor = D_s(offs_ineq_k + i);
                v_Ggt_ineq_temp[0](i) = g(offs_ge_eq_ineq + i) / scaling_factor;
            }
            gemv_n(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, v_Ggt_ineq_temp[0], 0, 1.0,
                   v_RSQrqt_tilde[k], 0, v_RSQrqt_tilde[k], 0);
        }
        {
            v_RSQrq_hat_curr_p = &v_RSQrqt_tilde[k];
        }
        {
            trsv_lnn(nu, Llt[k], 0, 0, *v_RSQrq_hat_curr_p, 0, v_Llt[k], 0);
            veccp(nu, v_Llt[k], 0, v_Llt_shift[0], 0);
            gemv_n(nx, nu, -1.0, Llt[k], nu, 0, v_Llt_shift[0], 0, 1.0, v_RSQrqt_tilde[k], nu,
                   v_Ppt[k], 0);
        }
    }
    {
        const Index nx = info.dims.number_of_states[0];
        {
            trsv_lnn(nx, LlIt[0], 0, 0, v_Ppt[0], 0, v_LlIt[0], 0);
        }
    }
    {
        const Index nx = info.dims.number_of_states[0];
        const Index nu = info.dims.number_of_controls[0];
        const Index offs_x = info.offsets_primal_x[0];
        veccpsc(nx, -1.0, v_LlIt[0], 0, x, offs_x);
        trsv_ltn(nx, LlIt[0], 0, 0, x, offs_x, x, offs_x);
    }
    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_x = info.offsets_primal_x[k];
        const Index offs_dyn_eq_k = info.offsets_g_eq_dyn[k];
        veccpsc(nu, -1.0, v_Llt[k], 0, x, offs);
        gemv_t(nx, nu, -1.0, Llt[k], nu, 0, x, offs_x, 1.0, x, offs, x, offs);
        trsv_ltn(nu, Llt[k], 0, 0, x, offs, x, offs);
        if (k != info.dims.K - 1)
        {
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index offsp1 = info.offsets_primal_u[k + 1];
            const Index offs_x_p1 = info.offsets_primal_x[k + 1];
            veccp(nxp1, g, offs_dyn_eq_k, x, offs_x_p1);
            gemv_t(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, x, offs, 1.0, x, offs_x_p1, x,
                   offs_x_p1);
            veccp(nxp1, v_Ppt[k + 1], 0, eq_mult, offs_dyn_eq_k);
            gemv_t(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, x, offs_x_p1, 1.0, eq_mult,
                   offs_dyn_eq_k, eq_mult, offs_dyn_eq_k);
        }
    }
    // // calculate lam_eq xk
    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_g_k = info.offsets_g_eq_path[k];
        const Index offs_eq = info.offsets_eq[k];
        if (ng > 0)
        {
            gemv_t(nu + nx, ng, 1.0, jacobian.Gg_eqt[k], 0, 0, x, offs, 1.0, g, offs_g_k,
                   eq_mult, offs_g_k);
            eq_mult.block(ng, offs_g_k) =
                eq_mult.block(ng, offs_g_k) / D_eq.block(ng, offs_eq);
        }
    }

    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_gineq_k = info.offsets_g_eq_slack[k];
        const Index offs_slack = info.offsets_slack[k];
        if (ng_ineq > 0)
        {
            gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_gineq_k,
                   eq_mult, offs_gineq_k);
            eq_mult.block(ng_ineq, offs_gineq_k) =
                eq_mult.block(ng_ineq, offs_gineq_k) / D_s.block(ng_ineq, offs_slack);
        }
    }
    return LinsolReturnFlag::SUCCESS;
}

void AugSystemSolver<AcceleratedOcpType>::TestPermutationFunctions(const ProblemInfo& info, int k){
    std::cout << "starting test of permutation functions..." << std::endl;
    // generate some permutation vectors
    Index m, n_max, n, r1, r2, r;
    if (k < 0){
        m = 9; // ng
        n_max = 7; // nu
        n = 12; // nu + nx + 1
        r1 = 3; // < min(ng, nu)
        r2 = 5; // < min(ng, nu) - r1
        r = r1 + r2;
    } else {
        m = gamma[k]; // ng
        n_max = info.dims.number_of_controls[k]; // nu
        n = info.dims.number_of_controls[k] + info.dims.number_of_states[k] + 1; // nu + nx + 1
        r1 = rho1[k]; // < min(ng, nu)
        r2 = rho2[k]; // < min(ng, nu) - r1
        r = r1 + r2;
    }
    PermutationMatrix Pl1_test(m), Pl_rank_test(m), Pl2_test(m), Pr1_test(n), Pr2_test(n);
    if (k < 0){
        for (Index i = 0; i < m; i++){
            if (i < r1){ Pl1_test[i] = (i + 3) % m;}
            Pl_rank_test[i] = (4*i + 5) % m;
            if (i < r2) { Pl2_test[i] = (i + 7) % m;}
        }

        for (Index i = 0; i < n; i++){
            if (i < r1) {Pr1_test[i] = (i + 2) % n;}
            if (i < r2) {Pr2_test[i] = (i + 4) % n;}
        }
    } else {
        for (Index i = 0; i < m; i++){
            Pl1_test[i] = Pl1[k][i];
            Pl_rank_test[i] = Pl_rank[k][i];
            Pl2_test[i] = Pl2[k][i];
        }

        for (Index i = 0; i < n; i++){
            Pr1_test[i] = Pr1[k][i];
            Pr2_test[i] = Pr2[k][i];
        }
    }

    // TEST 1: Pl application functions
    VecRealAllocated v(m);
    MatRealAllocated v_row(1, m);
    VecRealAllocated v1(m); 

    for (Index i = 0; i < m; i++){
        v(i) = i;
        v_row(0, i) = i;
        v1(i) = i;
    }
    Pl1_test.apply(r1, &v1.vec(), 0);
    // std::cout << "v1: " << v1 << std::endl;

    apply_Pl(Pl1_test, Pl_rank_test, Pl2_test, r1, r2, m, &v.vec(), 0);
    apply_Pl_on_cols(Pl1_test, Pl_rank_test, Pl2_test, r1, r2, m, &v_row.mat(), 0);
    // std::cout << "Pl1:     " << Pl1_test << std::endl;
    // std::cout << "Pl_rank: " << Pl_rank_test << std::endl;
    // std::cout << "Pl2:     " << Pl2_test << std::endl;
    // std::cout << "v: " << v << std::endl;
    // std::cout << "v_row: " << v_row << std::endl;
    for (Index i = 0; i < m; i++){
        if (std::abs(v(i) - v_row(0, i)) > 1e-10){
            std::cout << "Error in apply_Pl_on_cols at index " << i << ": " << v(i) << " vs " << v_row(0, i) << std::endl;
        }
    }
    apply_Pl_inverse(Pl1_test, Pl_rank_test, Pl2_test, r1, r2, m, &v.vec(), 0);
    for (Index i = 0; i < m; i++){
        if (std::abs(v(i) - i) > 1e-10){
            std::cout << "Error in apply_Pl_inverse at index " << i << ": " << v(i) << " vs " << i << std::endl;
        }
    }

    
    // TEST 2: Pr application functions
    VecRealAllocated w(n);
    MatRealAllocated w_row(1, n);
    MatRealAllocated w_col(n, 1);
    for (Index i = 0; i < n; i++){
        w(i) = i;
        w_row(0, i) = w(i);
        w_col(i, 0) = w(i);
    }

    apply_Pr(Pr1_test, Pr2_test, r1, r2, &w.vec(), 0);
    apply_Pr_on_rows(Pr1_test, Pr2_test, r1, r2, &w_col.mat());
    apply_Pr_on_cols(Pr1_test, Pr2_test, r1, r2, &w_row.mat());
    for (Index i = 0; i < n; i++){
        if (std::abs(w(i) - w_row(0, i)) > 1e-10){
            std::cout << "Error in apply_Pr_on_rows at index " << i << ": " << w(i) << " vs " << w_row(0, i) << std::endl;
        }
        if (std::abs(w(i) - w_col(i, 0)) > 1e-10){
            std::cout << "Error in apply_Pr_on_cols at index " << i << ": " << w(i) << " vs " << w_col(i, 0) << std::endl;
        }
    }
    apply_Pr_inverse(Pr1_test, Pr2_test, r1, r2, &w.vec(), 0);
    for (Index i = 0; i < n; i++){
        if (std::abs(w(i) - i) > 1e-10){
            std::cout << "Error in apply_Pr_inverse at index " << i << ": " << w(i) << " vs " << i << std::endl;
        }
    }
    std::cout << "Finished test of permutation functions." << std::endl;
}

void AugSystemSolver<AcceleratedOcpType>::fatrop_lu_fact_blocked_transposed(
        const ProblemDims& dims, const Index k, MAT *At, bool avoid_additional_perms){

    Index nu = dims.number_of_controls[k];
    Index nx = dims.number_of_states[k];
    Index ng = dims.number_of_eq_constraints[k];
    Index nx_next = (k < dims.K - 1) ? nb_of_zk_vars : 0;
    Index nf = (k < dims.K - 1) ? nb_of_dynamics_constraints : 0;
    if (nx_next < 0){ nx_next = dims.number_of_states[k + 1];}
    if (nf < 0){ nf = nx_next;}

    Index nu_true = nu - nx_next;
    Index ng_true = ng - nf;

    Index m = gamma[k];
    Index nc = m - ng;
    Index n = nu + nx + 1;
    Index n_max = nu;

    // lu of top-left block
    if (avoid_additional_perms){
        fatrop_lu_fact_transposed(ng_true, nu_true, nu_true, rho1[k], At, Pl1[k], 
                                Pr1[k], lu_fact_tol, n, m);
    } else {
        fatrop_lu_fact_transposed(ng_true, nu_true, nu_true, rho1[k], At, Pl1[k], 
                                 Pr1[k], lu_fact_tol);
    }
    for (int i = 0; i < m - ng_true; i++){Pl_rank[k][rho1[k] + i] = ng_true + i;}

    // permute rows of matrix
    if (avoid_additional_perms){
        Pl_rank[k].apply_on_cols(m, At, 0, 0, n);
    } else {
        Pl_rank[k].apply_on_cols(m, At, 0, 0, n_max);
    }

    // scaling bottom-left
    blasfeo_dgecp(nu_true, nf, At, 0, rho1[k], &scratch[0].mat(), 0, 0);   // M2 to B
    // NOTE: this operation can be performed in the lu decomposition by 
    // applying the permutation to the full matrix already
    if (!avoid_additional_perms){
        Pr1[k].apply_on_rows(rho1[k], &scratch[0].mat());
    }

    // compute K3 and K4
    blasfeo_dtrsm_llnn(rho1[k], nf, 1.0, At, 0, 0, &scratch[0].mat(), 0, 0, 
                       At, 0, rho1[k]); // K3 <-- M2_1 V1^-1
    blasfeo_dgemm_nn(n_max-nx_next-rho1[k], nf, rho1[k], -1.0, 
                     At, rho1[k], 0, At, 0, rho1[k], 0.0, 
                     At, rho1[k], rho1[k], At, rho1[k], rho1[k]); // K4 <-- M2_2 - K3*V2
    if (rho1[k] > 0){
        blasfeo_dgead(nu_true-rho1[k], nf, 1.0, &scratch[0].mat(), rho1[k], 0, At, rho1[k], rho1[k]);
    } else {
        blasfeo_dgecp(nu_true-rho1[k], nf, &scratch[0].mat(), rho1[k], 0, At, rho1[k], rho1[k]);
    }

    // second LU decomposition
    if (avoid_additional_perms){
        // NOTE: offset lu factorization seems to show bugs in blasfeo_dcolsw
        fatrop_lu_fact_transposed(nf + nc, nu-rho1[k], nu-rho1[k], rho2[k], 
                                At, rho1[k], rho1[k], Pl2[k], Pr2[k], 
                                lu_fact_tol, n, m);
    } else {
        #ifndef OFFSET_FREE_P2
            blasfeo_dgecp(nu-rho1[k], nf + nc, At, rho1[k], rho1[k], &scratch[0].mat(), 0, 0);
            fatrop_lu_fact_transposed(nf + nc, nu-rho1[k], nu-rho1[k], rho2[k], &scratch[0].mat(), Pl2[k], Pr2[k], lu_fact_tol);
            blasfeo_dgecp(nu-rho1[k], nf + nc, &scratch[0].mat(), 0, 0, At, rho1[k], rho1[k]);
            // NOTE: these operations can be performed in the lu decompositions by
            // applying those permutations to the full matrix already
            Pl2[k].apply_on_cols(rho2[k], At, 0, rho1[k], rho1[k]); // permute K3
            Pr2[k].apply_on_rows(rho2[k], At, rho1[k], rho1[k]); // permute V2
        #else
            blasfeo_dgecp(nu-rho1[k], nf + nc, At, rho1[k], rho1[k], &scratch[0].mat(), 0, 0);
            PermutationMatrix Pl2_temp(Pl2[k].size());
            PermutationMatrix Pr2_temp(Pr2[k].size());
            fatrop_lu_fact_transposed(nf + nc, nu-rho1[k], nu-rho1[k], rho2[k], &scratch[0].mat(), Pl2_temp, Pr2_temp, lu_fact_tol);
            for (int i = 0; i < rho2[k]; i++){
                Pl2[k][rho1[k] + i] = Pl2_temp[i] + rho1[k];
                Pr2[k][rho1[k] + i] = Pr2_temp[i] + rho1[k];
            }
            blasfeo_dgecp(nu-rho1[k], nf + nc, &scratch[0].mat(), 0, 0, At, rho1[k], rho1[k]);
            // NOTE: these operations can be performed in the lu decompositions by
            // applying those permutations to the full matrix already
            Pl2[k].apply_on_cols(rho1[k]+rho2[k], At, 0, 0, rho1[k]); // permute K3
            Pr2[k].apply_on_rows(rho1[k]+rho2[k], At, 0, rho1[k]); // permute V2
        #endif
        std::cout << "rho1: " << rho1[k] << ", rho2: " << rho2[k] << std::endl;
        std::cout << "Pl2:\n" << Pl2[k] << std::endl;
        std::cout << "Pr2:\n" << Pr2[k] << std::endl;
    }
    rho[k] = rho1[k] + rho2[k];

    // Fix bottom part
    // std::cout << "bottom part before:\n"; blasfeo_print_dmat(n-n_max, m, At, n_max, 0);
    if (n > n_max){    
        // NOTE: this operation can be performed in the lu decomposition by 
        // applying the permutation to the full matrix already
        if (!avoid_additional_perms){
            apply_Pl_on_cols(Pl1[k], Pl_rank[k], Pl2[k], rho1[k], rho2[k], m, At, n_max);
        }
        
        // M1 <- M1 * L1^-T
        blasfeo_dtrsm_runu(n-n_max, rho[k], 1.0, At, 0, 0, At, n_max, 0, At, n_max, 0);
        // M2 <- M2 - M1 * L2^T
        blasfeo_dgemm_nn(n-n_max, m-rho[k], rho[k], -1.0, At, n_max, 0, At, 0, rho[k], 1.0, 
                        At, n_max, rho[k], At, n_max, rho[k]);
    }
    // std::cout << "bottom part:\n"; blasfeo_print_dmat(n-n_max, m, At, n_max, 0);
}

void extract_L(const MatRealAllocated &LU, MatRealAllocated &L, int m, int n, int ai=0, int aj=0, int bi=0, int bj=0){
    for (int row = 0; row < m; row++){
        for (int col = 0; col < m; col++){
            if (row > col && col < m){
                L(bi+row,bj+col) = LU(ai+col,aj+row);
            } else if (row == col){
                L(bi+row,bj+col) = 1.0;
            } else {
                L(bi+row,bj+col) = 0.0;
            }
        }
    }
}

void extract_U(const MatRealAllocated &LU, MatRealAllocated &U, int m, int n, int ai=0, int aj=0, int bi=0, int bj=0){
    for (int row = 0; row < m; row++){
        for (int col = 0; col < n; col++){
            if (row <= col && row < LU.n()){
                U(bi+row,bj+col) = LU(ai+col,aj+row);
            } else {
                U(bi+row,bj+col) = 0.0;
            }
        }
    }
}
bool AugSystemSolver<AcceleratedOcpType>::verify_blocked_lu_new(const MatRealAllocated& LU, 
        const MatRealAllocated& A_original,
        PermutationMatrix& Pl1, PermutationMatrix& Pr1, int rank1,
        PermutationMatrix& Pl_rank,
        PermutationMatrix& Pl2, PermutationMatrix& Pr2, int rank2,
        int ng, int nu, int nx, int nc){
    MatRealAllocated A(A_original.m(), A_original.n());
    gecp(A_original.m(), A_original.n(), A_original, 0, 0, A, 0, 0);

    MatRealAllocated A_verification(A_original.m(), A_original.n());
    MatRealAllocated A_verification_T(A_original.n(), A_original.m());
    MatRealAllocated L(ng+nx+nc, ng+nx+nc);
    MatRealAllocated U(ng+nx+nc, nu+nx);
    extract_L(LU, L, ng+nx+nc, nu+nx, 0, 0);
    extract_U(LU, U, ng+nx+nc, nu+nx, 0, 0);

    // std::cout << "L:\n" << L.block(ng+nx+nc, ng+nx+nc, 0, 0) << std::endl;
    // std::cout << "U:\n" << U.block(ng+nx+nc, nu+nx, 0, 0) << std::endl;

    // compute L*U
    blasfeo_dgemm_nn(ng+nx+nc, nu+nx, ng+nx+nc, 1.0, &L.mat(), 0, 0, &U.mat(), 0, 0, 0.0, 
                     &A_verification.mat(), 0, 0, &A_verification.mat(), 0, 0);

    // apply permutations
    apply_Pl_on_cols(Pl1, Pl_rank, Pl2, rank1, rank2, ng+nx+nc, &A.mat(), 0);
    // std::cout << "A after Pl on cols:\n" << A.block(nu+nx, ng+nx, 0, 0) << std::endl;
    apply_Pr_on_rows(Pr1, Pr2, rank1, rank2, &A.mat());
    // std::cout << "A after Pr on rows:\n" << A.block(nu+nx, ng+nx, 0, 0) << std::endl;

    // transpose A_verification
    blasfeo_dgetr(ng+nx+nc, nu+nx, &A_verification.mat(), 0, 0, &A_verification_T.mat(), 0, 0);

    // check that A_copy and A_verification are close
    double max_diff = 0.0;
    for (int row = 0; row < nu+nx; row++){
        for (int col = 0; col < ng+nx+nc; col++){
            double diff = std::abs(A(row,col) - A_verification_T(row,col));
            if (diff > max_diff){
                max_diff = diff;
            }
        }
    }

    if (max_diff > 1e-4){
        std::cout << "\nBlocked LU factorization verification failed" << std::endl;
        // std::cout << "A is an " << (ng+nx) << "x" << (nu+nx) << " matrix" << std::endl;
        // std::cout << "Max difference: " << max_diff << std::endl;
        std::cout << "A_original:\n" << A_original.block(nu+nx, ng+nx+nc, 0, 0) << std::endl;
        std::cout << "A_copy:\n" << A.block(nu+nx, ng+nx+nc, 0, 0) << std::endl;
        std::cout << "A_verification:\n" << A_verification_T.block(nu+nx, ng+nx+nc, 0, 0) << std::endl;
        std::cout << "L:\n" << L << std::endl;
        std::cout << "U:\n" << U << std::endl;
        std::cout << "difference between Pl A Pr and L*U:" << std::endl;
        blasfeo_dgead(nu+nx, ng+nx+nc, -1.0, &A.mat(), 0, 0, &A_verification_T.mat(), 0, 0);
        std::cout << A_verification_T.block(nu+nx, ng+nx+nc, 0, 0) << std::endl;
        std::cout << "Pl1:    " << Pl1 << std::endl;
        std::cout << "Pl_rank:" << Pl_rank << std::endl;
        std::cout << "Pl2:    " << Pl2 << std::endl;
        std::cout << "Pr1:    " << Pr1 << std::endl;
        std::cout << "Pr2:    " << Pr2 << std::endl;
        std::cout << "r1:     " << rank1 << std::endl;
        std::cout << "r2:     " << rank2 << std::endl;
        return false;
    }

    return true;
}

void AugSystemSolver<AcceleratedOcpType>::apply_Pl_on_cols(
        PermutationMatrix& Pl1, PermutationMatrix& Pl_rank, PermutationMatrix& Pl2, 
        const Index r1, const Index r2, const Index m, MAT* A, const Index row_start){
    // if (row_start != 0){ throw std::runtime_error("Error in apply_Pl_on_cols: row_start is not supported yet");}

    Pl1.apply_on_cols(r1, A, row_start, 0, A->m-row_start);
    if (m > A->n){throw std::runtime_error("Error in apply_Pl: m is larger than vector size");}
    Pl_rank.apply_on_cols(m, A, row_start, 0, A->m-row_start);
    #ifndef OFFSET_FREE_P2
        Pl2.apply_on_cols(r2, A, row_start, r1, A->m-row_start);
    #else
        Pl2.apply_on_cols(r1+r2, A, row_start, 0, A->m-row_start);
    #endif
}
void AugSystemSolver<AcceleratedOcpType>::apply_Pl(PermutationMatrix& Pl1, 
        PermutationMatrix& Pl_rank, PermutationMatrix& Pl2, 
        const Index r1, const Index r2, const Index m, VEC *vec, const Index ai){
    Pl1.apply(r1, vec, ai);
    if (m > vec->m){throw std::runtime_error("Error in apply_Pl: m is larger than vector size");}
    Pl_rank.apply(m, vec, ai);
    #ifndef OFFSET_FREE_P2
        Pl2.apply(r2, vec, ai+r1);
    #else
        Pl2.apply(r1+r2, vec, ai);
    #endif
}
void AugSystemSolver<AcceleratedOcpType>::apply_Pl_inverse(PermutationMatrix& Pl1, 
        PermutationMatrix& Pl_rank, PermutationMatrix& Pl2, const Index r1, 
        const Index r2, const Index m, VEC *vec, const Index ai){
    #ifndef OFFSET_FREE_P2
        Pl2.apply_inverse(r2, vec, ai+r1);
    #else
        Pl2.apply_inverse(r1+r2, vec, ai);
    #endif
    if (m > vec->m){throw std::runtime_error("Error in apply_Pl: m is larger than vector size");}
    Pl_rank.apply_inverse(m, vec, ai);
    Pl1.apply_inverse(r1, vec, ai);
}

void AugSystemSolver<AcceleratedOcpType>::apply_Pr_on_rows(PermutationMatrix& Pr1, 
        PermutationMatrix& Pr2, const Index r1, const Index r2, MAT* A){
    Pr1.apply_on_rows(r1, A);
    #ifndef OFFSET_FREE_P2
        Pr2.apply_on_rows(r2, A, r1);
    #else
        Pr2.apply_on_rows(r1+r2, A, 0);
    #endif
}

void AugSystemSolver<AcceleratedOcpType>::apply_Pr_on_cols(PermutationMatrix& Pr1,
        PermutationMatrix& Pr2, const Index r1, const Index r2, MAT* A){
    Pr1.apply_on_cols(r1, A);
    #ifndef OFFSET_FREE_P2
        Pr2.apply_on_cols(r2, A, 0, r1, A->m);
    #else
        Pr2.apply_on_cols(r1+r2, A, 0, 0, A->m);
    #endif
}

void AugSystemSolver<AcceleratedOcpType>::apply_Pr(PermutationMatrix& Pr1, 
        PermutationMatrix& Pr2, const Index r1, const Index r2, VEC *vec, const Index ai){
    Pr1.apply(r1, vec, ai);
    #ifndef OFFSET_FREE_P2
        Pr2.apply(r2, vec, ai+r1);
    #else
        Pr2.apply(r1+r2, vec, ai);
    #endif
}

void AugSystemSolver<AcceleratedOcpType>::apply_Pr_inverse(PermutationMatrix& Pr1, 
        PermutationMatrix& Pr2, const Index r1, const Index r2, VEC *vec, const Index ai){
    #ifndef OFFSET_FREE_P2
        Pr2.apply_inverse(r2, vec, ai+r1);
    #else
        Pr2.apply_inverse(r1+r2, vec, ai);
    #endif
    Pr1.apply_inverse(r1, vec, ai);
}






















































MatRealAllocated PermutationVectorToMatrix(PermutationMatrix &P){
    MatRealAllocated result(P.size(), P.size());
    for (int i = 0; i < P.size(); i++){
        result(i, i) = 1.0;
    }
    P.apply_on_cols(P.size(), &result.mat());
    return result;
}

void PrintPreProcessNpInfo(const ProblemInfo &info, 
                           const ProblemInfo & modified_info,
                           const Hessian<ImplicitOcpType> &hess, 
                           const Jacobian<ImplicitOcpType> &jacobian,
                           const VecRealView &x, const VecRealView &eq_mult,
                           const VecRealView &D_x,
                           std::string file_name){
    // Print dimensions
    // std::ostream& o = std::cout;
    std::ofstream o(file_name);
    o << "import numpy as np\n";

    // o << "==============================================================\n";
    o << "K = " << info.dims.K << "\n";
    o << "nu = [";
    std::vector<int> nu = {};
    for (int k = 0; k < info.dims.K; k++){
        o << info.dims.number_of_controls[k];
        if (k < info.dims.K - 1){ o << ", ";}
        nu.push_back(info.dims.number_of_controls[k]);
    }
    o << "]\n";
    o << "nx = [";
    std::vector<int> nx = {};
    for (int k = 0; k < info.dims.K; k++){
        o << info.dims.number_of_states[k];
        if (k < info.dims.K - 1){ o << ", ";}
        nx.push_back(info.dims.number_of_states[k]);
    }
    o << "]\n";
    o << "ng_ineq = [";
    std::vector<int> ng_ineq = {};
    for (int k = 0; k < info.dims.K; k++){
        o << info.dims.number_of_ineq_constraints[k];
        if (k < info.dims.K - 1){ o << ", ";}
        ng_ineq.push_back(info.dims.number_of_ineq_constraints[k]);
    }
    o << "]\n";
    o << "ng_eq = [";
    std::vector<int> ng_eq = {};
    for (int k = 0; k < info.dims.K; k++){
        o << info.dims.number_of_eq_constraints[k];
        if (k < info.dims.K - 1){ o << ", ";}
        ng_eq.push_back(info.dims.number_of_eq_constraints[k]);
    }
    o << "]\n";
    o << "r = [";
    std::vector<int> r = {};
    for (int k = 0; k < info.dims.K - 1; k++){
        o << jacobian.J_ranks[k];
        if (k < info.dims.K - 2){ o << ", ";}
        r.push_back(jacobian.J_ranks[k]);
    }
    o << "]\n";
    
    o << "D_x = [";
    for (int k = 0; k < info.dims.K; k++){
        o << "[";
        int n = info.dims.number_of_controls[k] + info.dims.number_of_states[k];
        for (int i = 0; i < n; i++){
            o << D_x(info.offsets_primal_u[k] + i);
            if (i < n - 1){ o << ", ";}
        }
        o << "]";
        if (k < info.dims.K - 1){ o << ",\n";}
    }
    o << "]\n";

    // modified info
    o << "modified_K = " << modified_info.dims.K << "\n";
    o << "modified_nu = [";
    std::vector<int> modified_nu = {};
    for (int k = 0; k < modified_info.dims.K; k++){
        o << modified_info.dims.number_of_controls[k];
        if (k < modified_info.dims.K - 1){ o << ", ";}
        modified_nu.push_back(modified_info.dims.number_of_controls[k]);
    }
    o << "]\n";
    o << "modified_nx = [";
    std::vector<int> modified_nx = {};
    for (int k = 0; k < modified_info.dims.K; k++){
        o << modified_info.dims.number_of_states[k];
        if (k < modified_info.dims.K - 1){ o << ", ";}
        modified_nx.push_back(modified_info.dims.number_of_states[k]);
    }
    o << "]\n";
    o << "modified_ng_ineq = [";
    std::vector<int> modified_ng_ineq = {};
    for (int k = 0; k < modified_info.dims.K; k++){
        o << modified_info.dims.number_of_ineq_constraints[k];
        if (k < modified_info.dims.K - 1){ o << ", ";}
        modified_ng_ineq.push_back(modified_info.dims.number_of_ineq_constraints[k]);
    }
    o << "]\n";
    o << "modified_ng_eq = [";
    std::vector<int> modified_ng_eq = {};
    for (int k = 0; k < modified_info.dims.K; k++){
        o << modified_info.dims.number_of_eq_constraints[k];
        if (k < modified_info.dims.K - 1){ o << ", ";}
        modified_ng_eq.push_back(modified_info.dims.number_of_eq_constraints[k]);
    }
    o << "]\n";

    // hessian attributes
    o << "RSQrqt = [\n";
    for (int k = 0; k < info.dims.K; k++){
        PrintNpArray(hess.RSQrqt[k], "", modified_nu[k] + modified_nx[k] + 1, modified_nu[k] + modified_nx[k], false, o);
        if (k < info.dims.K - 1){ o << ",\n";}
    }
    o << "]\n";
    o << "RSQrqt_original = [\n";
    for (int k = 0; k < info.dims.K; k++){
        PrintNpArray(hess.RSQrqt_original[k], "", nu[k] + nx[k] + 1, nu[k] + nx[k], false, o);
        if (k < info.dims.K - 1){ o << ",\n";}
    }
    o << "]\n";
    o << "FuFx" << " = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        PrintNpArray(hess.FuFx[k], "", modified_nu[k] + modified_nx[k], modified_nx[k+1], false, o);
        if (k < info.dims.K - 2){ o << ",\n";}
    }
    o << "]\n";
    o << "FuFx_original" << " = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        PrintNpArray(hess.FuFx_original[k], "", nu[k] + nx[k], nx[k+1], false, o);
        if (k < info.dims.K - 2){ o << ",\n";}
    }
    o << "]\n";
    o << "GuGx" << " = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        PrintNpArray(hess.GuGx[k], "", modified_nu[k] + modified_nx[k], modified_nu[k+1], false, o);
        if (k < info.dims.K - 2){ o << ",\n";}
    }
    o << "]\n";
    o << "GuGx_original" << " = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        PrintNpArray(hess.GuGx_original[k], "", nu[k] + nx[k], nu[k+1], false, o);
        if (k < info.dims.K - 2){ o << ",\n";}
    }
    o << "]\n";

    // jacobian attributes
    o << "Gg_eqt = [\n";
    for (int k = 0; k < info.dims.K; k++){
        PrintNpArray(jacobian.Gg_eqt[k], "",  modified_nu[k] + modified_nx[k] + 1, modified_ng_eq[k], false, o);
        if (k < info.dims.K - 1){ o << ",\n";}
    }
    o << "]\n";
    o << "Gg_eqt_original = [\n";
    for (int k = 0; k < info.dims.K; k++){
        PrintNpArray(jacobian.Gg_eqt_original[k], "", nu[k] + nx[k] + 1, ng_eq[k], false, o);
        if (k < info.dims.K - 1){ o << ",\n";}
    }
    o << "]\n";
    o << "Gg_ineqt = [\n";
    for (int k = 0; k < info.dims.K; k++){
        PrintNpArray(jacobian.Gg_ineqt[k], "", modified_nu[k] + modified_nx[k] + 1, modified_ng_ineq[k], false, o);
        if (k < info.dims.K - 1){ o << ",\n";}
    }
    o << "]\n";
    o << "Gg_ineqt_original = [\n";
    for (int k = 0; k < info.dims.K; k++){
        PrintNpArray(jacobian.Gg_ineqt_original[k], "", nu[k] + nx[k] + 1, ng_ineq[k], false, o);
        if (k < info.dims.K - 1){ o << ",\n";}
    }
    o << "]\n";
    o << "BAbt = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        PrintNpArray(jacobian.BAbt[k], "", modified_nu[k] + modified_nx[k] + 1, modified_nx[k+1], false, o);
        if (k < info.dims.K - 2){ o << ",\n";}
    }
    o << "]\n";
    o << "BAbt_original = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        PrintNpArray(jacobian.BAbt_original[k], "", nu[k] + nx[k] + 1, nx[k+1], false, o);
        if (k < info.dims.K - 2){ o << ",\n";}
    }
    o << "]\n";

    std::vector<MatRealAllocated> Jt_LU;
    Jt_LU.reserve(info.dims.K - 1);
    std::vector<PermutationMatrix> Pl;
    Pl.reserve(info.dims.K - 1);
    std::vector<PermutationMatrix> Pr;
    Pr.reserve(info.dims.K - 1);
    
    o << "Jt = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        PrintNpArray(jacobian.Jt[k], "", nx[k+1], nx[k+1], false, o);
        if (k < info.dims.K - 2){ o << ",\n";}
        Jt_LU.push_back(jacobian.Jt[k]);
        Pl.push_back(PermutationMatrix(nx[k+1]));
        Pr.push_back(PermutationMatrix(nx[k+1]));
        int rho;
        lu_fact_transposed(nx[k+1], nx[k+1], nx[k+1], rho, Jt_LU[k], Pl[k], Pr[k]);
        if (rho != jacobian.J_ranks[k]){
            std::cerr << "Error: LU factorization rank does not match J_ranks." << std::endl;
        }
    }
    o << "]\n";

    // LU info
    o << "L = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        o << "np.array([";
        int nx_next = info.dims.number_of_states[k+1];
        if (nx_next == 0){ o << "[]";}
        for (int i = 0; i < nx_next; i++){
            o << "[";
            for (int j = 0; j < nx_next; j++){
                if (i > j){
                    // o << jacobian.Jt_LU[k](j,i);
                    o << std::setw(10) << std::setprecision(10) << Jt_LU[k](j,i);
                } else if (i == j){
                    o << 1.0;
                } else {
                    o << 0.0;
                }
                if (j < nx_next - 1){
                    o << ", ";
                }
            }
            o << "]";
            if (i < nx_next){
                o << ",\n";
            }
        }
        o << "])\n";
        if (k < info.dims.K - 2){ o << ",\n";}
    }
    o << "]\n";

    o << "U = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        o << "np.array([";
        int nx_next = info.dims.number_of_states[k+1];
        if (nx_next == 0){ o << "[]";}
        for (int i = 0; i < nx_next; i++){
            o << "[";
            for (int j = 0; j < nx_next; j++){
                if (i <= j){
                    // o << jacobian.Jt_LU[k](j,i);
                    o << std::setw(10) << std::setprecision(10) << Jt_LU[k](j,i);
                } else {
                    o << 0.0;
                }
                if (j < nx_next - 1){
                    o << ", ";
                }
            }
            o << "]";
            if (i < nx_next){
                o << ",\n";
            }
        }
        o << "])\n";
        if (k < info.dims.K - 2){ o << ",\n";}
    }
    o << "]\n";

    o << "Pl = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        PermutationMatrix Pl_copy = jacobian.Pl_pre[k];
        PrintNpArray(PermutationVectorToMatrix(Pl_copy), "", nx[k+1], nx[k+1], false, o);
        if (k < info.dims.K - 2){ o << ",\n";}
    }
    o << "]\n";
    o << "Pr = [\n";
    for (int k = 0; k < info.dims.K - 1; k++){
        PermutationMatrix Pr_copy = jacobian.Pr_pre[k];
        PrintNpArray(PermutationVectorToMatrix(Pr_copy), "", nx[k+1], nx[k+1], false, o);
        if (k < info.dims.K - 2){ o << ",\n";}
    }
    o << "]\n";

    o << "x = np.array([";
    for (int i = 0; i < info.number_of_primal_variables; i++){
        o << std::setw(10) << std::setprecision(10) << x(i);
        if (i < info.number_of_primal_variables - 1){
            o << ", ";
        }
    }
    o << "])\n";
    o << "eq_mult = np.array([";
    for (int i = 0; i < info.number_of_eq_constraints; i++){
        o << std::setw(10) << std::setprecision(10) << eq_mult(i);
        if (i < info.number_of_eq_constraints - 1){
            o << ", ";
        }
    }
    o << "])\n";
    // o << "==============================================================\n";
    o.close();
    std::cout << "wrote preprocess info to " << file_name << std::endl;
}

void PrintFactorizationInfo(const ProblemInfo &info,
                            const std::vector<Index>& rank_k_values,
                            std::vector<PermutationMatrix>& Pl,
                            std::vector<PermutationMatrix>& Pr,
                            std::vector<MatRealAllocated>& LU,
                            std::vector<Index>& gamma_k_values,
                            std::vector<MatRealAllocated>& Gg_eqt,
                            std::vector<MatRealAllocated>& Llt,
                            std::vector<MatRealAllocated>& R_shur,
                            std::string filename){
    // std::ostream& o = std::cout;
    std::ofstream o(filename);
    o << "import numpy as np\n";

    // o << "==============================================================\n";

    o << "rank_k_values = [";
    for (size_t i = 0; i < rank_k_values.size(); i++){
        o << rank_k_values[i];
        if (i < rank_k_values.size() - 1){
            o << ", ";
        }
    }
    o << "]\n";

    o << "Pl_r = [\n";
    for (size_t i = 0; i < Pl.size(); i++){
        PermutationMatrix Pl_copy = Pl[i];
        PrintNpArray(PermutationVectorToMatrix(Pl_copy), "", gamma_k_values[i], 
            gamma_k_values[i], false, o);
        if (i < Pl.size() - 1){ o << ",\n";}
    }
    o << "]\n";

    o << "Pr_r = [\n";
    for (size_t i = 0; i < Pr.size(); i++){
        PermutationMatrix Pr_copy = Pr[i];
        PrintNpArray(PermutationVectorToMatrix(Pr_copy), "", info.dims.number_of_controls[i], info.dims.number_of_controls[i], false, o);
        if (i < Pr.size() - 1){ o << ",\n";}
    }
    o << "]\n";

    // LU factorization
    o << "L_r = [\n";
    for (int k = 0; k < LU.size(); k++){
        int m = gamma_k_values[k];
        o << "np.array([";
        int n = info.dims.number_of_controls[k];
        if (m == 0){ o << "[]";}
        for (int i = 0; i < m; i++){
            o << "[";
            for (int j = 0; j < m; j++){
                if (i > j){
                    o << std::setw(10) << std::setprecision(10) << LU[k](j,i);
                } else if (i == j){
                    o << 1.0;
                } else {
                    o << 0.0;
                }
                if (j < m - 1){
                    o << ", ";
                }
            }
            o << "]";
            if (i < m){
                o << ",\n";
            }
        }
        o << "])\n";
        if (k < LU.size() - 1){ o << ",\n";}
    }
    o << "]\n";

    o << "U_r = [\n";
    for (int k = 0; k < LU.size(); k++){
        int m = gamma_k_values[k];
        o << "np.array([";
        int n = info.dims.number_of_controls[k];
        if (m == 0){ o << "[]";}
        for (int i = 0; i < m; i++){
            o << "[";
            for (int j = 0; j < n; j++){
                if (i <= j){
                    o << std::setw(10) << std::setprecision(10) << LU[k](j,i);
                } else {
                    o << 0.0;
                }
                if (j < n - 1){
                    o << ", ";
                }
            }
            o << "]";
            if (i < m){
                o << ",\n";
            }
        }
        o << "])\n";
        if (k < LU.size() - 1){ o << ",\n";}
    }
    o << "]\n";

    o << "Hut = [\n";
    for (size_t i = 0; i < Gg_eqt.size(); i++){
        PrintNpArray(Gg_eqt[i], "", info.dims.number_of_controls[i], gamma_k_values[i], false, o);
        if (i < Gg_eqt.size() - 1){ o << ",\n";}
    }
    o << "]\n";

    // shur step
    o << "Lmbd = [\n";
    for (size_t i = 0; i < Llt.size(); i++){
        int n = info.dims.number_of_controls[i] - rank_k_values[i];
        PrintNpArray(Llt[i], "", n, n, false, o);
        if (i < Llt.size() - 1){ o << ",\n";}
    }
    o << "]\n";

    o << "R_shur = [\n";
    for (size_t i = 0; i < R_shur.size(); i++){
        int m = info.dims.number_of_controls[i] - rank_k_values[i];
        int n = info.dims.number_of_controls[i] - rank_k_values[i];
        PrintNpArray(R_shur[i], "", m, n, false, o);
        if (i < R_shur.size() - 1){ o << ",\n";}
    }
    o << "]\n";
    // o << "==============================================================\n";
    o.close();
}


ModifiedAugSystemSolver::ModifiedAugSystemSolver(const ProblemInfo &info)
{
    std::vector<Index> number_of_controls = info.dims.number_of_controls;
    std::vector<Index> number_of_states = info.dims.number_of_states;
    std::vector<Index> number_of_ineq_constraints = info.dims.number_of_ineq_constraints;
    std::vector<Index> number_of_eq_constraints = info.dims.number_of_eq_constraints;

    for (Index i = 1; i < number_of_controls.size(); i++)
    {
        number_of_controls[i] += info.dims.number_of_states[i];
    }
    for (Index i = 0; i < number_of_eq_constraints.size()-1; i++)
    {
        // There can be no more constraints than number of controls and states.
        // Otherwise, the constraints can not be satisfied (assuming the 
        // jacobian is not degenerate)
        // The preprocessing should check the rank of the J-matrix to make
        // sure we don't end up in this case.
        number_of_eq_constraints[i] = std::min(
            number_of_eq_constraints[i] + info.dims.number_of_states[i+1],
            number_of_controls[i] + number_of_states[i]);
    }
    ProblemInfo new_info = ProblemInfo(ProblemDims(info.dims.K, number_of_controls, number_of_states, number_of_eq_constraints,
                       number_of_ineq_constraints));
    

    Index max_number_of_controls =
        *std::max_element(new_info.dims.number_of_controls.begin(), new_info.dims.number_of_controls.end());
    Index max_number_of_states =
        *std::max_element(new_info.dims.number_of_states.begin(), new_info.dims.number_of_states.end());
    Index max_number_of_variables = *std::max_element(new_info.number_of_stage_variables.begin(),
                                                      new_info.number_of_stage_variables.end());
    Index max_number_of_ineq_constraints = *std::max_element(
        new_info.dims.number_of_ineq_constraints.begin(), new_info.dims.number_of_ineq_constraints.end());
    Index max_number_of_eq_consttraints = *std::max_element(
        new_info.dims.number_of_eq_constraints.begin(), new_info.dims.number_of_eq_constraints.end());

    AL.emplace_back(max_number_of_variables + 1, max_number_of_variables);
    Ggt_stripe.emplace_back(max_number_of_variables + 1, max_number_of_variables);
    GgLt.emplace_back(max_number_of_variables + 1, max_number_of_variables);
    RSQrqt_hat.emplace_back(max_number_of_variables + 1, max_number_of_variables);
    Llt_shift.emplace_back(max_number_of_variables + 1, max_number_of_controls);
    GgIt_tilde.emplace_back(new_info.dims.number_of_states[0] + 1, new_info.dims.number_of_states[0]);
    GgLIt.emplace_back(new_info.dims.number_of_states[0] + 1, new_info.dims.number_of_states[0]);
    HhIt.emplace_back(new_info.dims.number_of_states[0] + 1, new_info.dims.number_of_states[0]);
    PpIt_hat.emplace_back(new_info.dims.number_of_states[0] + 1, new_info.dims.number_of_states[0]);
    LlIt.emplace_back(new_info.dims.number_of_states[0] + 1, new_info.dims.number_of_states[0]);
    Ggt_ineq_temp.emplace_back(max_number_of_variables + 1, max_number_of_ineq_constraints);

    FuFx_underbar.reserve(new_info.dims.K-1);
    GuGx_tilde.reserve(new_info.dims.K-1);
    GuGx_hat.reserve(new_info.dims.K-1);
    RSQrqt_underbar.reserve(new_info.dims.K);
    Ppt.reserve(new_info.dims.K);
    Hh.reserve(new_info.dims.K);
    RSQrqt_tilde.reserve(new_info.dims.K);
    Ggt_tilde.reserve(new_info.dims.K);
    Llt.reserve(new_info.dims.K);
    for (Index k = 0; k < new_info.dims.K; k++)
    {
        Index nu = new_info.dims.number_of_controls[k];
        Index nx = new_info.dims.number_of_states[k];
        Index ng_ineq = new_info.dims.number_of_ineq_constraints[k];
        Index ng_eq = new_info.dims.number_of_eq_constraints[k];
        Ppt.emplace_back(nx + 1, nx);
        Hh.emplace_back(nx, nx + 1);
        RSQrqt_underbar.emplace_back(nu + nx + 1, nx + nu);
        RSQrqt_tilde.emplace_back(nu + nx + 1, nx + nu);
        Ggt_tilde.emplace_back(nu + nx + 1, nx + nu);
        Llt.emplace_back(nu + nx + 1, nu);
        if (k < new_info.dims.K - 1){
            Index nu_next = new_info.dims.number_of_controls[k+1];
            Index nx_next = new_info.dims.number_of_states[k+1];
            FuFx_underbar.emplace_back(nu + nx, nx_next);
            GuGx_tilde.emplace_back(nu + nx, nu_next);
            GuGx_hat.emplace_back(nu + nx, nu_next);
        }
    }

    v_r_tilde.emplace_back(1, max_number_of_variables);
    v_AL.emplace_back(max_number_of_variables);
    v_Ggt_stripe.emplace_back(max_number_of_variables);
    v_GgLt.emplace_back(max_number_of_variables);
    v_RSQrqt_hat.emplace_back(max_number_of_variables);
    v_Llt_shift.emplace_back(max_number_of_controls);
    v_GgIt_tilde.emplace_back(new_info.dims.number_of_states[0]);
    v_GgLIt.emplace_back(new_info.dims.number_of_states[0]);
    v_HhIt.emplace_back(new_info.dims.number_of_states[0]);
    v_PpIt_hat.emplace_back(new_info.dims.number_of_states[0]);
    v_LlIt.emplace_back(new_info.dims.number_of_states[0]);
    v_Ggt_ineq_temp.emplace_back(max_number_of_ineq_constraints);
    v_tmp.emplace_back(max_number_of_variables);

    v_Ppt.reserve(new_info.dims.K);
    v_Hh.reserve(new_info.dims.K);
    v_RSQrqt_tilde.reserve(new_info.dims.K);
    v_Ggt_tilde.reserve(new_info.dims.K);
    v_Llt.reserve(new_info.dims.K);

    for (Index k = 0; k < new_info.dims.K; k++)
    {
        Index nu = new_info.dims.number_of_controls[k];
        Index nx = new_info.dims.number_of_states[k];
        v_Ppt.emplace_back(nx);
        v_Hh.emplace_back(nx);
        v_RSQrqt_tilde.emplace_back(nu + nx);
        v_Ggt_tilde.emplace_back(nu + nx);
        v_Llt.emplace_back(nu + nx);
    }

    PlI.emplace_back(new_info.dims.number_of_states[0]);
    PrI.emplace_back(new_info.dims.number_of_states[0]);

    Pl.reserve(new_info.dims.K);
    Pr.reserve(new_info.dims.K);

    for (Index k = 0; k < new_info.dims.K; k++)
    {
        Index nu = new_info.dims.number_of_controls[k];
        Index nx = new_info.dims.number_of_states[k];
        Pl.emplace_back(max_number_of_controls);
        Pr.emplace_back(max_number_of_controls);
    }

    gamma.resize(new_info.dims.K);
    rho.resize(new_info.dims.K);

    // for debugging
    if (write_factorization_file){
    rank_k_values = std::vector<Index>(info.dims.K);
    LU.reserve(info.dims.K);
    for (Index k = 0; k < info.dims.K; k++){
        LU.emplace_back(max_number_of_controls, max_number_of_eq_consttraints);
    }
    gamma_k_values = std::vector<Index>(info.dims.K);
    Ggt_eq.reserve(info.dims.K);
    for (Index k = 0; k < info.dims.K; k++){ 
        Ggt_eq.emplace_back(max_number_of_controls + max_number_of_states + 1, max_number_of_eq_consttraints);
    }
    R_shur.reserve(info.dims.K);
    for (Index k = 0; k < info.dims.K; k++){ 
        R_shur.emplace_back(max_number_of_states, max_number_of_states);
    }
    }
};

void ModifiedAugSystemSolver::register_options(OptionRegistry &registry)
{
    registry.register_option("linsol_it_ref", &ModifiedAugSystemSolver::set_it_ref, this);
    registry.register_option("linsol_perturbed_mode", &ModifiedAugSystemSolver::set_perturbed_mode, this);
    registry.register_option("linsol_perturbed_mode_param", &ModifiedAugSystemSolver::set_perturbed_mode_param, this);
    registry.register_option("linsol_lu_fact_tol", &ModifiedAugSystemSolver::set_lu_fact_tol, this);
    registry.register_option("linsol_diagnostic", &ModifiedAugSystemSolver::set_diagnostic, this);
    registry.register_option("linsol_increased_accuracy", &ModifiedAugSystemSolver::set_increased_accuracy, this);
}

LinsolReturnFlag ModifiedAugSystemSolver::solve(const ProblemInfo &info,
                                           Jacobian<ImplicitOcpType> &jacobian, Hessian<ImplicitOcpType> &hessian,
                                           const VecRealView &D_x, const VecRealView &D_s,
                                           const VecRealView &f, const VecRealView &g,
                                           VecRealView &x, VecRealView &eq_mult)
{
    #ifdef PROFILE 
    auto outer_start = std::chrono::high_resolution_clock::now(); 
    #endif
    MatRealView *RSQrq_hat_curr_p;
    Index rank_k;

    #ifdef PROFILE
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    #endif

    auto intermediate_start = std::chrono::high_resolution_clock::now();
    /////////////// recursion ///////////////
    for (Index k = info.dims.K - 1; k >= 0; --k)
    {
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offset_ineq_k = info.offsets_slack[k];
        const Index offset_u = info.offsets_primal_u[k];
        const Index offset_eq_path = info.offsets_g_eq_path[k];
        const Index offset_eq_slack = info.offsets_g_eq_slack[k];
        const Index nunxm1 = (k > 0) ? info.dims.number_of_controls[k-1] + info.dims.number_of_states[k-1] : 0;

        if (k == info.dims.K - 1){
            if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
            rowin(nu + nx, 1.0, f, offset_u, hessian.RSQrqt[k], nu + nx, 0);
            gecp(nx + nu + 1, nu + nx, hessian.RSQrqt[k], 0, 0, RSQrqt_underbar[k], 0, 0);
        }
        #ifdef PROFILE
        start = std::chrono::high_resolution_clock::now();
        #endif
        if (k > 0){
            if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
            gecp(nunxm1 + 1, nunxm1, hessian.RSQrqt[k-1], 0, 0, RSQrqt_underbar[k-1], 0, 0);
            // The adjacent-stage Hessian belongs to the transition entering
            // the current stage.  Refresh it at every backward step, just as
            // the local Hessian above.  Copying only the terminal transition
            // left all earlier full-rank FuFx blocks stale; rank-deficient
            // paths happened to overwrite some of them while forming GuGx.
            gecp(nunxm1, nx, hessian.FuFx[k-1], 0, 0,
                 FuFx_underbar[k-1], 0, 0);
        }
        #ifdef PROFILE
        stop = std::chrono::high_resolution_clock::now();
        duration_RSQrqt_copy += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
        #endif
        // std::cout << "\n\nk = " << k << std::endl;
        // PrintNpArray(RSQrqt_underbar[0], "RSQrqt_underbar[0]");
        // PrintNpArray(RSQrqt_underbar[1], "RSQrqt_underbar[1]");
        // std::cout << " === SUBSDYN === " << std::endl; 
        //////// SUBSDYN
        Index gamma_k;
        if (k == info.dims.K - 1)
        {
            gamma_k = ng;
            gamma[k] = gamma_k;
            rowin(ng, 1.0, g, offset_eq_path, jacobian.Gg_eqt[k], nu + nx, 0);
            gecp(nx + nu + 1, ng, jacobian.Gg_eqt[k], 0, 0, Ggt_stripe[0], 0, 0);
            if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
            gecp(nx + nu + 1, nu + nx, RSQrqt_underbar[k], 0, 0, RSQrqt_tilde[k], 0, 0);
        }
        else
        {
            const Index offset_eq_dyn = info.offsets_g_eq_dyn[k];
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index Hp1_size = gamma[k + 1] - rho[k + 1];
            if (Hp1_size + ng > nu + nx){
                if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                return LinsolReturnFlag::NOFULL_RANK;
            }
            gamma_k = Hp1_size + ng;
            // AL <- [BAb]^T_k P_kp1
            rowin(nxp1, 1.0, g, offset_eq_dyn, jacobian.BAbt[k], nu + nx, 0);
            gemm_nt(nu + nx + 1, nxp1, nxp1, 1.0, jacobian.BAbt[k], 0, 0, Ppt[k + 1], 0, 0, 0.0,
                    AL[0], 0, 0, AL[0], 0, 0);
            // AL[-1,:] <- AL[-1,:] + p_kp1^T
            gead(1, nxp1, 1.0, Ppt[k + 1], nxp1, 0, AL[0], nx + nu, 0);
            // RSQrqt_stripe <- AL[BA] + RSQrqt
            syrk_ln_mn(nu + nx + 1, nu + nx, nxp1, 1.0, AL[0], 0, 0, jacobian.BAbt[k], 0, 0, 1.0,
                       RSQrqt_underbar[k], 0, 0, RSQrqt_tilde[k], 0, 0);

            // Add second order dynamics contribution
            // std::cout << "--------------------------------------------" << std::endl;
            // std::cout << "test FuFx addition to RSQrqt" << std::endl;
            // PrintNpArray(jacobian.BAbt[k], "BAbt");
            // PrintNpArray(FuFx_underbar[k], "FuFx");
            // PrintNpArray(hessian.FuFx[k], "FuFx_hessian");
            // PrintNpArray(RSQrqt_tilde[k], "RSQrqt_tilde");
            // std::cout << "nu = " << nu << std::endl;
            // std::cout << "nx = " << nx << std::endl;
            // std::cout << "nx_next = " << nxp1 << std::endl;
            #ifdef PROFILE
            start = std::chrono::high_resolution_clock::now();
            #endif
            #ifndef IGNORE_EXTENSION
            gemm_nt(nu + nx + 1, nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, FuFx_underbar[k], 0, 0, 1.0,
                    RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k], 0, 0);
            // PrintNpArray(RSQrqt_tilde[k], "RSQrqt_intermediate");
            gemm_nt(nu + nx, nu + nx, nxp1, 1.0, FuFx_underbar[k], 0, 0, jacobian.BAbt[k], 0, 0, 1.0,
                    RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k], 0, 0);
            #endif
            #ifdef PROFILE
            stop = std::chrono::high_resolution_clock::now();
            duration_FuFx_addition += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            #endif
            // PrintNpArray(RSQrqt_tilde[k], "RSQrqt_tilde_after");
            // std::cout << "--------------------------------------------" << std::endl;

            //// inequalities
            gamma[k] = gamma_k;
            // if ng[k]>0
            if (gamma_k > 0)
            {
                // if Gk nonempty
                if (ng > 0)
                {
                    // Ggt_stripe  <- Ggt_k
                    rowin(ng, 1.0, g, offset_eq_path, jacobian.Gg_eqt[k], nu + nx, 0);
                    gecp(nu + nx + 1, ng, jacobian.Gg_eqt[k], 0, 0, Ggt_stripe[0], 0, 0);
                }
                // if Hkp1 nonempty
                if (Hp1_size > 0)
                {
                    // Ggt_stripe <- [Ggt_k [BAb_k^T]H_kp1]
                    gemm_nt(nu + nx + 1, Hp1_size, nxp1, 1.0, jacobian.BAbt[k], 0, 0, Hh[k + 1], 0,
                            0, 0.0, Ggt_stripe[0], 0, ng, Ggt_stripe[0], 0, ng);
                    // Ggt_stripe[-1,ng:] <- Ggt_stripe[-1,ng:] + h_kp1^T
                    gead_transposed(1, Hp1_size, 1.0, Hh[k + 1], 0, nxp1, Ggt_stripe[0], nu + nx,
                                    ng);
                }
            }
            else
            {
                rho[k] = 0;
                rank_k = 0;
                RSQrq_hat_curr_p = &RSQrqt_tilde[k];
            }
        }
        // inequalities + inertia correction
        {
            // We've already covered this in pre-processing
            // if (ng_ineq > 0)
            // {
            //     rowin(ng_ineq, 1.0, g, offset_eq_slack, jacobian.Gg_ineqt[k], nu + nx, 0);
            //     gecp(nu + nx + 1, ng_ineq, jacobian.Gg_ineqt[k], 0, 0, Ggt_ineq_temp[0], 0, 0);
            //     for (Index i = 0; i < ng_ineq; i++)
            //     {
            //         Scalar scaling_factor = 1.0 / D_s(offset_ineq_k + i);
            //         colsc(nu + nx + 1, scaling_factor, Ggt_ineq_temp[0], 0, i);
            //     }
            //     // add the penalty
            //     syrk_ln_mn(nu + nx + 1, nu + nx, ng_ineq, 1.0, Ggt_ineq_temp[0], 0, 0,
            //                jacobian.Gg_ineqt[k], 0, 0, 1.0, RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k],
            //                0, 0);
            // }
            // // inertia correction
            // diaad(nu + nx, 1.0, D_x, offset_u, RSQrqt_tilde[k], 0, 0);
        }
        // PrintNpArray(RSQrqt_underbar[0], "RSQrqt_underbar[0]");
        // PrintNpArray(RSQrqt_underbar[1], "RSQrqt_underbar[1]");
        // std::cout << " === TRANSFORM AND SUBSEQ === " << std::endl;
        //////// TRANSFORM_AND_SUBSEQ
        {
            // symmetric transformation, done a little different than in paper, in order to fuse LA
            // operations LU_FACT_TRANSPOSE(Ggtstripe[:gamma_k, nu+nx+1], nu max) if(k==K-2)
            // blasfeo_print_dmat(1, gamma_k, Ggt_stripe[0], nu+nx, 0);
            // std::cout << "computing factorization" << std::endl;
            // PrintNpArray(Ggt_stripe[0], "Ggt_stripe before factorization");
            // std::cout << "(gamma_k = " << gamma_k << ")" << std::endl;
            if (write_factorization_file){
                gamma_k_values[k] = gamma_k;
                gecp(nu, gamma_k, Ggt_stripe[0], 0, 0, Ggt_eq[k], 0, 0);
            }
            auto start = std::chrono::steady_clock::now();
            lu_fact_transposed(gamma_k, nu + nx + 1, nu, rank_k, Ggt_stripe[0], Pl[k], Pr[k], lu_fact_tol);
            auto stop = std::chrono::steady_clock::now();
            duration_lu_factorization += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            if (write_factorization_file){
                rank_k_values[k] = rank_k;
                gecp(nu, gamma_k, Ggt_stripe[0], 0, 0, LU[k], 0, 0);
            }
            // std::cout << "Pl: "; for (int i = 0; i < rank_k; i++){ std::cout << Pl[k][i] << " ";}std::cout << std::endl;
            // std::cout << "Pr: "; for (int i = 0; i < rank_k; i++){ std::cout << Pr[k][i] << " ";}std::cout << std::endl;
            // std::cout << "rank_" << k << " = " << rank_k << std::endl;
            
            rho[k] = rank_k;
            if (gamma_k - rank_k > 0)
            {
                // transfer eq's to next stage
                if (gamma_k - rank_k > nx){
                    if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                    return LinsolReturnFlag::NOFULL_RANK;
                }
                getr(nx + 1, gamma_k - rank_k, Ggt_stripe[0], nu, rank_k, Hh[k], 0, 0);
            }
            if (rank_k > 0)
            {
                // Ggt_tilde_k <- Ggt_stripe[rho_k:nu+nx+1, :rho] L-T (note that this is slightly
                // different from the implementation)
                trsm_rlnn(nu - rank_k + nx + 1, rank_k, -1.0, Ggt_stripe[0], 0, 0, Ggt_stripe[0],
                          rank_k, 0, Ggt_tilde[k], 0, 0);
                // the following command copies the top block matrix (LU) to the bottom because it
                // it needed later
                gecp(rank_k, gamma_k, Ggt_stripe[0], 0, 0, Ggt_tilde[k], nu - rank_k + nx + 1, 0);
                // permutations
                trtr_l(nu + nx, RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k], 0,
                       0); // copy lower part of RSQ to upper part
                Pr[k].apply_on_rows(rank_k, &RSQrqt_tilde[k].mat()); // TODO make use of symmetry
                Pr[k].apply_on_cols(rank_k, &RSQrqt_tilde[k].mat());
                // GL <- Ggt_tilde_k @ RSQ[:rho,:nu+nx] + RSQrqt[rho:nu+nx+1, rho:] (with
                // RSQ[:rho,:nu+nx] = RSQrqt[:nu+nx,:rho]^T) GEMM_NT(nu - rank_k + nx + 1, nu + nx,
                // rank_k, 1.0, Ggt_tilde[k], 0, 0, RSQrqt_tilde[k], 0, 0, 1.0, RSQrqt_tilde_p
                // + k, rank_k, 0, GgLt[0], 0, 0); split up because valgrind was giving invalid read
                // errors when C matrix has nonzero row offset GgLt[0].print();
                gecp(nu - rank_k + nx + 1, nu + nx, RSQrqt_tilde[k], rank_k, 0, GgLt[0], 0, 0);
                gemm_nt(nu - rank_k + nx + 1, nu + nx, rank_k, 1.0, Ggt_tilde[k], 0, 0,
                        RSQrqt_tilde[k], 0, 0, 1.0, GgLt[0], 0, 0, GgLt[0], 0, 0);
                // RSQrqt_hat = GgLt[nu-rank_k + nx +1, :rank_k] * G[:rank_k, :nu+nx] +
                // GgLt[rank_k:, :]  (with G[:rank_k,:nu+nx] = Gt[:nu+nx,:rank_k]^T)
                syrk_ln_mn(nu - rank_k + nx + 1, nu + nx - rank_k, rank_k, 1.0, GgLt[0], 0, 0,
                           Ggt_tilde[k], 0, 0, 1.0, GgLt[0], 0, rank_k, RSQrqt_hat[0], 0, 0);
                // GEMM_NT(nu - rank_k + nx + 1, nu + nx - rank_k, rank_k, 1.0, GgLt[0], 0, 0,
                // Ggt_tilde[k], 0, 0, 1.0, GgLt[0], 0, rank_k, RSQrqt_hat[0], 0, 0);
                RSQrq_hat_curr_p = &RSQrqt_hat[0];
            }
            else
            {
                RSQrq_hat_curr_p = &RSQrqt_tilde[k];
            }
            #ifdef PROFILE
            start = std::chrono::steady_clock::now();  
            #endif
            #ifndef IGNORE_EXTENSION
            if (k > 0 && jacobian.nb_new_controls[k-1] > 0){
                // GuGx_tilde = GuGx * T_r^-1
                gecp(nunxm1, nu, hessian.GuGx[k-1], 0, 0, GuGx_tilde[k-1], 0, 0);
                // PrintNpArray(GuGx_tilde[k-1], "GuGx_tilde before permutation");
                Pr[k].apply_on_cols(rank_k, &GuGx_tilde[k-1].mat());
                if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                gemm_nt(nunxm1, nu - rank_k, rank_k, 1.0, GuGx_tilde[k-1], 0, 0, Ggt_tilde[k], 0, 0,
                        1.0, GuGx_tilde[k-1], 0, rank_k, GuGx_tilde[k-1], 0, rank_k);
                
                // FuFx = FuFx + GuGx * [Hx_tilde; 0]
                gecp(nunxm1, nx, hessian.FuFx[k-1], 0, 0, FuFx_underbar[k-1], 0, 0);
                // std::cout << "-----------------------------------" << std::endl;
                // std::cout << "testing FuFx and RSQrqt updates" << std::endl;
                // PrintNpArray(GuGx_tilde[k-1], "GuGx_tilde");
                // PrintNpArray(FuFx_underbar[k-1], "FuFx_underbar_before");
                // PrintNpArray(Ggt_tilde[k], "Ggt_tilde");
                // std::cout << "nu = " << nu << std::endl;
                // std::cout << "rank_k = " << rank_k << std::endl;
                // std::cout << "nunxm1 = " << nunxm1 << std::endl;
                // std::cout << "nx = " << nx << std::endl;
                if (print_debug_lines) {std::cout << __LINE__ << std::endl;}            
                gemm_nt(nunxm1, nx, rank_k, 1.0, GuGx_tilde[k-1], 0, 0, 
                        Ggt_tilde[k], nu - rank_k, 0, 1.0, FuFx_underbar[k-1], 0, 0, FuFx_underbar[k-1], 0, 0);
                // PrintNpArray(FuFx_underbar[k-1], "FuFx_underbar_after");

                // [r^T q^T] = [r^T q^T] + [h_tilde^T, 0] * GuGx_tilde^T
                if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                // PrintNpArray(Ggt_tilde[k], "Ggt_tilde");
                // PrintNpArray(RSQrqt_underbar[k-1], "RSQrqt_underbar_before");
                gemm_nt(1, nunxm1, rank_k, 1.0,  Ggt_tilde[k], nu - rank_k + nx, 0, 
                        GuGx_tilde[k-1], 0, 0, 1.0, RSQrqt_underbar[k-1], nunxm1, 0, RSQrqt_underbar[k-1], nunxm1, 0);
                // PrintNpArray(RSQrqt_underbar[k-1], "RSQrqt_underbar_after");
                // std::cout << "-----------------------------------" << std::endl;
            }
            #endif
            #ifdef PROFILE
            stop = std::chrono::steady_clock::now();
            duration_GuGx_addition += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            #endif
        }
        // PrintNpArray(RSQrqt_underbar[0], "RSQrqt_underbar[0]");
        // PrintNpArray(RSQrqt_underbar[1], "RSQrqt_underbar[1]");
        // std::cout << " === SCHUR === " << std::endl;
        //////// SCHUR
        {
            if (nu - rank_k > 0)
            {
                // DLlt_k = [chol(R_hatk); Llk@chol(R_hatk)^-T]
                // PrintNpArray(RSQrq_hat_curr_p[0], "RSQrq_hat");
                if (write_factorization_file){
                    gecp(nu-rank_k, nu-rank_k, RSQrq_hat_curr_p[0], 0, 0, R_shur[k], 0, 0);
                }
                potrf_l_mn(nu - rank_k + nx + 1, nu - rank_k, RSQrq_hat_curr_p[0], 0, 0, Llt[k], 0,
                           0);
                // PrintNpArray(Llt[k], "shur[" + std::to_string(k) + "]");
                // PrintNpArray(RSQrq_hat_curr_p[0], "RSQrq_hat");
                // PrintNpArray(Llt[k], "Llt");
                // PrintNpArray(GuGx_tilde[k-1], "GuGx_tilde");
                if (!check_reg(nu - rank_k, &Llt[k].mat(), 0, 0))
                    return LinsolReturnFlag::INDEFINITE;
                // Pp_k = Qq_hatk - L_k^T @ Ll_k
                // SYRK_LN_MN(nx+1, nx, nu-rank_k, -1.0,Llt_p+k, nu-rank_k,0, Llt_p+k,
                // nu-rank_k,0, 1.0, RSQrq_hat_curr[0], nu-rank_k, nu-rank_k,Pp+k,0,0); // feature
                // not implmented yet
                gecp(nx + 1, nu - rank_k, Llt[k], nu - rank_k, 0, Llt_shift[0], 0,
                     0); // needless operation because feature not implemented yet
                // SYRK_LN_MN(nx + 1, nx, nu - rank_k, -1.0, Llt_shift[0], 0, 0, Llt_shift[0], 0,
                // 0, 1.0, RSQrq_hat_curr[0], nu - rank_k, nu - rank_k, Ppt[k], 0, 0);
                gecp(nx + 1, nx, RSQrq_hat_curr_p[0], nu - rank_k, nu - rank_k, Ppt[k], 0, 0);
                syrk_ln_mn(nx + 1, nx, nu - rank_k, -1.0, Llt_shift[0], 0, 0, Llt_shift[0], 0, 0,
                           1.0, Ppt[k], 0, 0, Ppt[k], 0, 0);
                // next steps are for better accuracy
                if (increased_accuracy)
                {
                    // copy eta
                    getr(nu - rank_k, gamma_k - rank_k, Ggt_stripe[0], rank_k, rank_k,
                         Ggt_stripe[0], 0, 0);
                    // blasfeo_print_dmat(gamma_k-rank_k, nu-rank_k, Ggt_stripe[0], 0,0);
                    // eta L^-T
                    trsm_rltn(gamma_k - rank_k, nu - rank_k, 1.0, Llt[k], 0, 0, Ggt_stripe[0], 0, 0,
                              Ggt_stripe[0], 0, 0);
                    // ([S^T \\ r^T] L^-T) @ (L^-1 eta^T)
                    // (eta L^-T) @ ([S^T \\ r^T] L^-T)^T
                    gemm_nt(gamma_k - rank_k, nx + 1, nu - rank_k, -1.0, Ggt_stripe[0], 0, 0,
                            Llt[k], nu - rank_k, 0, 1.0, Hh[k], 0, 0, Hh[k], 0, 0);
                    // keep (L^-1 eta^T) for forward recursion
                    getr(gamma_k - rank_k, nu - rank_k, Ggt_stripe[0], 0, 0, Ggt_tilde[k], 0,
                         rank_k);
                }
                #ifdef PROFILE
                start = std::chrono::high_resolution_clock::now();
                #endif
                #ifndef IGNORE_EXTENSION
                if (k > 0 && jacobian.nb_new_controls[k-1] > 0){
                    // GuGx_hat = GuGx * L^-1
                    if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                    gecp(nunxm1, nu, GuGx_tilde[k-1], 0, 0, GuGx_hat[k-1], 0, 0);
                    trsm_rltn(nunxm1, nu - rank_k, 1.0, Llt[k], 0, 0, GuGx_hat[k-1], 0, rank_k, GuGx_hat[k-1], 0, rank_k);

                    // RSQrqt = RSQrqt - GuGx^T L^-T L^-1 GuGx
                    if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                    // std::cout << "#-----------------------------------" << std::endl;
                    // std::cout << "#testing RSQrqt_bar update (also copy RSQrqt_hat, Llt and GuGxtilde definition)" << std::endl;
                    // std::cout << "nx = " << nx << std::endl;
                    // std::cout << "nu = " << nu << std::endl;
                    // std::cout << "nunxm1 = " << nunxm1 << std::endl;
                    // std::cout << "rank_k = " << rank_k << std::endl;
                    // PrintNpArray(RSQrqt_underbar[k-1], "RSQrqt_underbar_before");
                    // PrintNpArray(RSQrqt_tilde[k], "RSQrqt_tilde");
                    gemm_nt(nunxm1, nunxm1, nu - rank_k, -1.0, GuGx_hat[k-1], 0, rank_k, GuGx_hat[k-1], 0, rank_k, 
                            1.0, RSQrqt_underbar[k-1], 0, 0, RSQrqt_underbar[k-1], 0, 0);
                    if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                    trsm_rltn(1, nu - rank_k, 1.0, Llt[k], 0, 0, RSQrq_hat_curr_p[0], nu - rank_k + nx, 0, v_r_tilde[0], 0, 0);
                    // PrintNpArray(v_r_tilde[0], "v_r_tilde");
                    // PrintNpArray(hessian.GuGx[k-1], "GuGx");
                    // PrintNpArray(GuGx_tilde[k-1], "GuGx_tilde");
                    // PrintNpArray(GuGx_hat[k-1], "GuGx_hat");
                    // PrintNpArray(RSQrqt_underbar[k-1], "RSQrqt_underbar_intermediate");
                    // MatRealAllocated temp(1, nunxm1);
                    // gemm_nt(1, nunxm1, nu - rank_k, -1.0, v_r_tilde[0], 0, 0, GuGx_hat[k-1], 0, rank_k, 
                    //         1.0, temp, 0, 0, temp, 0, 0);
                    // PrintNpArray(temp, "temp");
                    gemm_nt(1, nunxm1, nu - rank_k, -1.0, v_r_tilde[0], 0, 0, GuGx_hat[k-1], 0, rank_k,
                            1.0, RSQrqt_underbar[k-1], nunxm1, 0, RSQrqt_underbar[k-1], nunxm1, 0);
                    // PrintNpArray(RSQrqt_underbar[k-1], "RSQrqt_underbar_after");

                    // FuFx = FuFx - GuGx_hat * L
                    // PrintNpArray(FuFx_underbar[k-1], "FuFx_underbar_before");
                    // PrintNpArray(Llt[k], "Llt");
                    if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                    gemm_nt(nunxm1, nx, nu - rank_k, -1.0, GuGx_hat[k-1], 0, rank_k, Llt[k], nu - rank_k, 0,
                            1.0, FuFx_underbar[k-1], 0, 0, FuFx_underbar[k-1], 0, 0);
                    // PrintNpArray(FuFx_underbar[k-1], "FuFx_underbar_after");
                    // std::cout << "-----------------------------------" << std::endl;
                }
                #endif
                #ifdef PROFILE
                stop = std::chrono::high_resolution_clock::now();
                duration_GuGx_hat_addition += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
                #endif
            }
            else
            {
                gecp(nx + 1, nx, RSQrq_hat_curr_p[0], 0, 0, Ppt[k], 0, 0);
            }
            trtr_l(nx, Ppt[k], 0, 0, Ppt[k], 0, 0);
        }
        // PrintNpArray(RSQrqt_underbar[0], "RSQrqt_underbar[0]");
        // PrintNpArray(RSQrqt_underbar[1], "RSQrqt_underbar[1]");
    }
    // std::cout << " === FIRST STAGE === " << std::endl;
    if (print_initial_stage){
        PrintNpArray(Ppt[0], "Ppt");
        PrintNpArray(Hh[0], "Hh");
    }
    auto intermediate_stop = std::chrono::high_resolution_clock::now();
    duration_backward_recursion += std::chrono::duration_cast<std::chrono::nanoseconds>(intermediate_stop - intermediate_start);
    intermediate_start = std::chrono::high_resolution_clock::now();

    rankI = 0;
    //////// FIRST_STAGE
    {
        const Index nx = info.dims.number_of_states[0];
        Index gamma_I = gamma[0] - rho[0];
        if (gamma_I > nx)
        {
            if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
            return LinsolReturnFlag::NOFULL_RANK;
        }
        if (gamma_I > 0)
        {
            getr(gamma_I, nx + 1, Hh[0], 0, 0, HhIt[0], 0, 0); // transposition may be avoided
            // HhIt[0].print();
            lu_fact_transposed(gamma_I, nx + 1, nx, rankI, HhIt[0], PlI[0], PrI[0], lu_fact_tol);
            if (rankI < gamma_I){
                if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                return LinsolReturnFlag::NOFULL_RANK;
            }
            // PpIt_tilde <- Ggt[rankI:nx+1, :rankI] L-T (note that this is slightly different from
            // the implementation)
            trsm_rlnn(nx - rankI + 1, rankI, -1.0, HhIt[0], 0, 0, HhIt[0], rankI, 0, GgIt_tilde[0],
                      0, 0);
            // permutations
            PrI[0].apply_on_rows(rankI, &Ppt[0].mat()); // TODO make use of symmetry
            PrI[0].apply_on_cols(rankI, &Ppt[0].mat());
            // // GL <- GgIt_tilde @ Pp[:rankI,:nx] + Ppt[rankI:nx+1, rankI:] (with Pp[:rankI,:nx] =
            // Ppt[:nx,:rankI]^T) GEMM_NT(nx - rankI + 1, nx, rankI, 1.0, GgIt_tilde[0], 0, 0,
            // Ppt[0], 0, 0, 1.0, Ppt[0], rankI, 0, GgLIt[0], 0, 0); split up because valgrind was
            // giving invalid read errors when C matrix has nonzero row offset
            gecp(nx - rankI + 1, nx, Ppt[0], rankI, 0, GgLIt[0], 0, 0);
            gemm_nt(nx - rankI + 1, nx, rankI, 1.0, GgIt_tilde[0], 0, 0, Ppt[0], 0, 0, 1.0,
                    GgLIt[0], 0, 0, GgLIt[0], 0, 0);
            // // RSQrqt_hat = GgLt[nu-rank_k + nx +1, :rank_k] * G[:rank_k, :nu+nx] + GgLt[rank_k:,
            // :]  (with G[:rank_k,:nu+nx] = Gt[:nu+nx,:rank_k]^T)
            syrk_ln_mn(nx - rankI + 1, nx - rankI, rankI, 1.0, GgLIt[0], 0, 0, GgIt_tilde[0], 0, 0,
                       1.0, GgLIt[0], 0, rankI, PpIt_hat[0], 0, 0);
            // TODO skipped if nx-rankI = 0
            potrf_l_mn(nx - rankI + 1, nx - rankI, PpIt_hat[0], 0, 0, LlIt[0], 0, 0);
            if (!check_reg(nx - rankI, &LlIt[0].mat(), 0, 0))
                return LinsolReturnFlag::INDEFINITE;
        }
        else
        {
            rankI = 0;
            potrf_l_mn(nx + 1, nx, Ppt[0], 0, 0, LlIt[0], 0, 0);
            if (!check_reg(nx, &LlIt[0].mat(), 0, 0))
                return LinsolReturnFlag::INDEFINITE;
        }
    }
    // std::cout << " === FORWARD FIRST STAGE === " << std::endl;
    ////// FORWARD_SUBSTITUTION:
    // first stage
    {
        const Index nx = info.dims.number_of_states[0];
        const Index nu = info.dims.number_of_controls[0];
        const Index offs_u = info.offsets_primal_u[0];
        const Index offs_x = info.offsets_primal_x[0];
        const Index offs_g = info.offsets_g_eq_path[0];
        // calculate xIb
        rowex(nx - rankI, -1.0, LlIt[0], nx - rankI, 0, x, offs_x + rankI);
        // assume TRSV_LTN allows aliasing, this is the case in normal BLAS
        trsv_ltn(nx - rankI, LlIt[0], 0, 0, x, offs_x + rankI, x, offs_x + rankI);
        // calculate xIa
        rowex(rankI, 1.0, GgIt_tilde[0], nx - rankI, 0, x, offs_x);
        // assume aliasing is possible for last two elements
        gemv_t(nx - rankI, rankI, 1.0, GgIt_tilde[0], 0, 0, x, offs_x + rankI, 1.0, x, offs_x, x,
               offs_x);
        //// lag
        rowex(rankI, -1.0, Ppt[0], nx, 0, eq_mult, offs_g);
        // assume aliasing is possible for last two elements
        gemv_t(nx, rankI, -1.0, Ppt[0], 0, 0, x, offs_x, 1.0, eq_mult, offs_g, eq_mult, offs_g);

        // U^-T
        trsv_lnn(rankI, HhIt[0], 0, 0, eq_mult, offs_g, eq_mult, offs_g);
        // L^-T
        trsv_unu(rankI, rankI, HhIt[0], 0, 0, eq_mult, offs_g, eq_mult, offs_g);
        PlI[0].apply_inverse(rankI, &eq_mult.vec(), offs_g);
        PrI[0].apply_inverse(rankI, &x.vec(), offs_x);
    }
    // other stages
    // if (print_initial_stage){
    //     PrintNpArray(x, "x_first_stage");
    //     PrintNpArray(eq_mult, "eq_mult_first_stage");
    // }
    intermediate_stop = std::chrono::high_resolution_clock::now();
    duration_initial_stage += std::chrono::duration_cast<std::chrono::nanoseconds>(intermediate_stop - intermediate_start);
    intermediate_start = std::chrono::high_resolution_clock::now();
    // std::cout << "x initial solution:\n" << x << std::endl;
    // std::cout << "eq_mult initial solution:\n" << eq_mult << std::endl;
    // std::cout << " === FORWARD OTHER STAGES === " << std::endl;
    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_x = info.offsets_primal_x[k];
        const Index rho_k = rho[k];
        const Index numrho_k = nu - rho_k;
        const Index offs_g_k = info.offsets_g_eq_path[k];
        const Index gammamrho_k = gamma[k] - rho[k];
        const Index gamma_k = gamma[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_eq_ineq = info.offsets_g_eq_slack[k];
        const Index offs_slack = info.offsets_slack[k];
        if (numrho_k > 0)
        {
            /// calculate ukb_tilde
            // -Lkxk - lk
            rowex(numrho_k, -1.0, Llt[k], numrho_k + nx, 0, x, offs + rho_k);
            if (increased_accuracy)
            {
                gemv_n(nu - rho_k, gamma_k - rho_k, -1.0, Ggt_tilde[k], 0, rho_k, eq_mult, offs_g_k,
                       1.0, x, offs + rho_k, x, offs + rho_k);
            }
            // assume aliasing of last two eliments is allowed
            gemv_t(nx, numrho_k, -1.0, Llt[k], numrho_k, 0, x, offs_x, 1.0, x, offs + rho_k, x,
                   offs + rho_k);
            trsv_ltn(numrho_k, Llt[k], 0, 0, x, offs + rho_k, x, offs + rho_k);

            #ifdef PROFILE
            start = std::chrono::high_resolution_clock::now();
            #endif
            #ifndef IGNORE_EXTENSION
            // + GuGxt [uk-1, xk-1]
            if (k > 0 && jacobian.nb_new_controls[k-1] > 0){
                const Index nunxm1 = info.dims.number_of_controls[k-1] + info.dims.number_of_states[k-1];
                // std::cout << "----------------------------------------" << std::endl;
                // std::cout << "test ukb_tilde update" << std::endl;
                // PrintNpArray(hessian.GuGx[k-1], "GuGx");
                // PrintNpArray(GuGx_hat[k-1], "GuGx_hat");
                // PrintNpArray(Llt[k], "Llt");
                // std::cout << "nunxm1 = " << nunxm1 << std::endl;
                // std::cout << "nu = " << nu << std::endl;
                // std::cout << "rho_k = " << rho_k << std::endl;
                // PrintNpArray(x, info.offsets_primal_u[k-1], nunxm1, "ukxk");
                // PrintNpArray(x, info.offsets_primal_u[k], nu, "uk_before");
                trsm_rlnn(nunxm1, numrho_k, 1.0, Llt[k], 0, 0, GuGx_hat[k-1], 0, rho_k, GuGx_hat[k-1], 0, rho_k);
                // PrintNpArray(GuGx_hat[k-1], "GuGx_hat_intermediate");
                gemv_t(nunxm1, nu - rho_k, -1.0, GuGx_hat[k-1], 0, rho_k, x, info.offsets_primal_u[k-1], 
                       1.0, x, offs + rho_k, x, offs + rho_k);
                // PrintNpArray(x, info.offsets_primal_u[k], nu, "uk_after");
                // std::cout << "----------------------------------------" << std::endl;
            }
            #endif
            #ifdef PROFILE
            stop = std::chrono::high_resolution_clock::now();
            duration_ukb_tilde_addition += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            #endif
        }
        /// calcualate uka_tilde
        if (rho_k > 0)
        {
            rowex(rho_k, 1.0, Ggt_tilde[k], numrho_k + nx, 0, x, offs);
            gemv_t(nx + numrho_k, rho_k, 1.0, Ggt_tilde[k], 0, 0, x, offs + rho_k, 1.0, x, offs, x,
                   offs);
            // calculate lamda_tilde_k
            // copy vk to right location
            veccp(gammamrho_k, eq_mult, offs_g_k, v_tmp[0], 0);
            veccp(gammamrho_k, v_tmp[0], 0, eq_mult, offs_g_k + rho_k);
            rowex(rho_k, -1.0, RSQrqt_tilde[k], nu + nx, 0, eq_mult, offs_g_k);
            // assume aliasing of last two eliments is allowed
            gemv_t(nu + nx, rho_k, -1.0, RSQrqt_tilde[k], 0, 0, x, offs, 1.0, eq_mult, offs_g_k,
                   eq_mult, offs_g_k);

            #ifdef PROFILE
            start = std::chrono::high_resolution_clock::now();
            #endif
            #ifndef IGNORE_EXTENSION
            if (k > 0 && jacobian.nb_new_controls[k-1] > 0){
                const Index nunxm1 = info.dims.number_of_controls[k-1] + info.dims.number_of_states[k-1];
                // std::cout << "eq_mult before:\n" << eq_mult << std::endl;
                // PrintNpArray(GuGx_tilde[k-1], "GuGx_tilde");
                // PrintNpArray(x, info.offsets_primal_u[k-1], nunxm1, "ukxk");
                gemv_t(nunxm1, rho_k, -1.0, GuGx_tilde[k-1], 0, 0, x, info.offsets_primal_u[k-1], 1.0, 
                       eq_mult, offs_g_k, eq_mult, offs_g_k);
                // std::cout << "eq_mult after:\n" << eq_mult << std::endl;
            }
            #endif
            #ifdef PROFILE
            stop = std::chrono::high_resolution_clock::now();
            duration_lambdatilde_addition += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            #endif

            // nu-rank_k+nx,0
            // needless copy because feature not implemented yet in trsv_lnn
            gecp(rho_k, gamma_k, Ggt_tilde[k], nu - rho_k + nx + 1, 0, AL[0], 0, 0);
            // U^-T
            trsv_lnn(rho_k, AL[0], 0, 0, eq_mult, offs_g_k, eq_mult, offs_g_k);
            // L^-T
            trsv_unu(rho_k, gamma_k, AL[0], 0, 0, eq_mult, offs_g_k, eq_mult, offs_g_k);
            Pl[k].apply_inverse(rho_k, &eq_mult.vec(), offs_g_k);
            Pr[k].apply_inverse(rho_k, &x.vec(), offs);
            // std::cout << "eq_mult after after:\n" << eq_mult << std::endl;
        }
        // we've already covered this in pre-processing
        // VecRealAllocated eq_mult_copy(eq_mult);
        // for (int i = 0; i < eq_mult.m(); i++){ eq_mult_copy(i) = eq_mult(i);}
        // if (ng_ineq > 0)
        // {
        //     gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_eq_ineq,
        //            eq_mult, offs_eq_ineq);
        //     eq_mult.block(ng_ineq, offs_eq_ineq) =
        //         eq_mult.block(ng_ineq, offs_eq_ineq) / D_s.block(ng_ineq, offs_slack);
        //     // if (k == 6){
        //     // PrintNpArray(jacobian.Gg_ineqt[k], "\nGg_ineqt[" + std::to_string(k) + "]", nu + nx + 1, ng_ineq);
        //     // PrintNpArray(D_s, offs_slack, ng_ineq, "D_s[" + std::to_string(k) + "]");
        //     // PrintNpArray(g, offs_eq_ineq, ng_ineq, "g[" + std::to_string(k) + "]");
        //     // PrintNpArray(x, offs, nu + nx, "x[" + std::to_string(k) + "]");
        //     // PrintNpArray(eq_mult_copy, offs_eq_ineq, ng_ineq, "[" + std::to_string(k) + "] eq_mult before ineq regularization");
        //     // std::cout << "nu: " << nu << " nx: " << nx << " ng_ineq: " << ng_ineq << std::endl;
        //     // std::cout << "offs: " << offs << " offs_eq_ineq: " << offs_eq_ineq << " offs_slack: " << offs_slack << std::endl;
        //     // gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_eq_ineq,
        //     //        eq_mult_copy, offs_eq_ineq);
        //     // eq_mult_copy.block(ng_ineq, offs_eq_ineq) =
        //     //     eq_mult_copy.block(ng_ineq, offs_eq_ineq) / D_s.block(ng_ineq, offs_slack);
        //     // PrintNpArray(eq_mult_copy, offs_eq_ineq, ng_ineq, "[" + std::to_string(k) + "] eq_mult after ineq regularization");
        //     // }
        // }
        if (k != info.dims.K - 1)
        {
            const Index offs_dyn_eq_k = info.offsets_g_eq_dyn[k];
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index nup1 = info.dims.number_of_controls[k + 1];
            const Index offsp1 = info.offsets_primal_u[k + 1];
            const Index offsxp1 = info.offsets_primal_x[k + 1];
            const Index offs_g_kp1 = info.offsets_g_eq_path[k + 1];
            const Index gammamrho_kp1 = gamma[k + 1] - rho[k + 1];
            // calculate xkp1
            rowex(nxp1, 1.0, jacobian.BAbt[k], nu + nx, 0, x, offsxp1);
            gemv_t(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, x, offs, 1.0, x, offsxp1, x,
                   offsxp1);
            // calculate lam_dyn xp1
            rowex(nxp1, 1.0, Ppt[k + 1], nxp1, 0, eq_mult, offs_dyn_eq_k);
            gemv_t(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, x, offsxp1, 1.0, eq_mult, offs_dyn_eq_k,
                   eq_mult, offs_dyn_eq_k);
            gemv_t(gammamrho_kp1, nxp1, 1.0, Hh[k + 1], 0, 0, eq_mult, offs_g_kp1, 1.0, eq_mult,
                   offs_dyn_eq_k, eq_mult, offs_dyn_eq_k);
            
            // std::cout << "----------------------------------------" << std::endl;
            // std::cout << "test pi update" << std::endl;
            // PrintNpArray(eq_mult, offs_dyn_eq_k, nxp1, "eq_mult");
            // PrintNpArray(hessian.FuFx[k], "FuFx");
            // PrintNpArray(x, offs, nu + nx, "x");
            #ifdef PROFILE
            start = std::chrono::high_resolution_clock::now();
            #endif
            #ifndef IGNORE_EXTENSION
            gemv_t(nu + nx, nxp1, 1.0, FuFx_underbar[k], 0, 0, x, offs, 1.0, 
                   eq_mult, offs_dyn_eq_k, eq_mult, offs_dyn_eq_k);
            #endif
            #ifdef PROFILE
            stop = std::chrono::high_resolution_clock::now();
            duration_FuFx_addition_forward += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
            #endif
            // PrintNpArray(eq_mult, offs_dyn_eq_k, nxp1, "eq_mult");
            // std::cout << "----------------------------------------" << std::endl;
        }
    }

    intermediate_stop = std::chrono::high_resolution_clock::now();
    duration_forward_recursion += std::chrono::duration_cast<std::chrono::nanoseconds>(intermediate_stop - intermediate_start);

    if (write_factorization_file){
        PrintFactorizationInfo(info, rank_k_values, Pl, Pr, LU, gamma_k_values, Ggt_eq, Llt, R_shur, factorization_file_name);
    }

    #ifdef PROFILE
    auto outer_stop = std::chrono::high_resolution_clock::now(); 
    duration_inner_solve = std::chrono::duration_cast<std::chrono::nanoseconds>(outer_stop - outer_start);
    #endif

    return LinsolReturnFlag::SUCCESS;
}
LinsolReturnFlag ModifiedAugSystemSolver::solve(const ProblemInfo &info,
                                           Jacobian<ImplicitOcpType> &jacobian, Hessian<ImplicitOcpType> &hessian,
                                           const VecRealView &D_x, const VecRealView &D_eq,
                                           const VecRealView &D_s, const VecRealView &f,
                                           const VecRealView &g, VecRealView &x,
                                           VecRealView &eq_mult)
{
    MatRealView *RSQrq_hat_curr_p;
    // This stabilized path has no per-stage equality transform that refreshes
    // its working Hessian stripes. Initialize them once so eliminating the
    // controls of stage k can update the stripe consumed at stage k-1.
    for (Index stage = 0; stage < info.dims.K; ++stage)
    {
        const Index local = info.dims.number_of_controls[stage]
                          + info.dims.number_of_states[stage];
        gecp(local + 1, local, hessian.RSQrqt[stage], 0, 0,
             RSQrqt_underbar[stage], 0, 0);
        if (stage + 1 < info.dims.K)
        {
            const Index next_states =
                info.dims.number_of_states[stage + 1];
            gecp(local, next_states, hessian.FuFx[stage], 0, 0,
                 FuFx_underbar[stage], 0, 0);
        }
    }
    for (Index k = info.dims.K - 1; k >= 0; --k)
    {
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_ineq_k = info.offsets_slack[k];
        const Index offset_u = info.offsets_primal_u[k];
        const Index offset_eq_k = info.offsets_eq[k];
        const Index offset_g_eq_k = info.offsets_g_eq_path[k];
        const Index offset_g_ineq_k = info.offsets_g_eq_slack[k];
        // const fatrop_int offs_g_ineq_k = offs_g_ineq_p[k];
        //////// SUBSDYN
        if (k == info.dims.K - 1)
        {
            rowin(nu + nx, 1.0, f, offset_u, RSQrqt_underbar[k], nu + nx, 0);
            gecp(nx + nu + 1, nu + nx, RSQrqt_underbar[k], 0, 0, RSQrqt_tilde[k], 0, 0);
        }
        else
        {
            const Index offset_eq_dyn = info.offsets_g_eq_dyn[k];
            const Index nxp1 = info.dims.number_of_states[k + 1];
            // AL <- [BAb]^T_k P_kp1
            rowin(nxp1, 1.0, g, offset_eq_dyn, jacobian.BAbt[k], nu + nx, 0);
            gemm_nt(nu + nx + 1, nxp1, nxp1, 1.0, jacobian.BAbt[k], 0, 0, Ppt[k + 1], 0, 0, 0.0,
                    AL[0], 0, 0, AL[0], 0, 0);
            // AL[-1,:] <- AL[-1,:] + p_kp1^T
            gead(1, nxp1, 1.0, Ppt[k + 1], nxp1, 0, AL[0], nx + nu, 0);
            // RSQrqt_stripe <- AL[BA] + RSQrqt
            rowin(nu + nx, 1.0, f, offset_u, RSQrqt_underbar[k], nu + nx, 0);
            syrk_ln_mn(nu + nx + 1, nu + nx, nxp1, 1.0, AL[0], 0, 0, jacobian.BAbt[k], 0, 0, 1.0,
                       RSQrqt_underbar[k], 0, 0, RSQrqt_tilde[k], 0, 0);

            // Add second order dynamics contribution                
            gemm_nt(nu + nx + 1, nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, FuFx_underbar[k], 0, 0, 1.0,
                    RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k], 0, 0);
            gemm_nt(nu + nx, nu + nx, nxp1, 1.0, FuFx_underbar[k], 0, 0, jacobian.BAbt[k], 0, 0, 1.0,
                    RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k], 0, 0);
        }
        // Covered in pre-processing
        // // equality penalty
        // {
        //     rowin(ng, 1.0, g, offset_g_eq_k, jacobian.Gg_eqt[k], nu + nx, 0);
        //     gecp(nu + nx + 1, ng, jacobian.Gg_eqt[k], 0, 0, Ggt_stripe[0], 0, 0);
        //     for (Index i = 0; i < ng; i++)
        //     {
        //         Scalar scaling_factor = 1.0 / D_eq(offset_eq_k + i);
        //         colsc(nu + nx + 1, scaling_factor, Ggt_stripe[0], 0, i);
        //     }
        //     // add the penalty
        //     syrk_ln_mn(nu + nx + 1, nu + nx, ng, 1.0, Ggt_stripe[0], 0, 0, jacobian.Gg_eqt[k], 0, 0,
        //                1.0, RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k], 0, 0);
        // }
        // // inequalities + inertia correction
        // {
        //     if (ng_ineq > 0)
        //     {
        //         rowin(ng_ineq, 1.0, g, offset_g_ineq_k, jacobian.Gg_ineqt[k], nu + nx, 0);
        //         gecp(nu + nx + 1, ng_ineq, jacobian.Gg_ineqt[k], 0, 0, Ggt_ineq_temp[0], 0, 0);
        //         for (Index i = 0; i < ng_ineq; i++)
        //         {
        //             Scalar scaling_factor = 1.0 / D_s(offs_ineq_k + i);
        //             colsc(nu + nx + 1, scaling_factor, Ggt_ineq_temp[0], 0, i);
        //         }
        //         // add the penalty
        //         syrk_ln_mn(nu + nx + 1, nu + nx, ng_ineq, 1.0, Ggt_ineq_temp[0], 0, 0,
        //                    jacobian.Gg_ineqt[k], 0, 0, 1.0, RSQrqt_tilde[k], 0, 0, RSQrqt_tilde[k],
        //                    0, 0);
        //     }
        //     // inertia correction
        //     diaad(nu + nx, 1.0, D_x, offset_u, RSQrqt_tilde[k], 0, 0);
        // }

        //////// TRANSFORM_AND_SUBSEQ
        {
            RSQrq_hat_curr_p = &RSQrqt_tilde[k];
        }
        //////// SCHUR
        {
            // DLlt_k = [chol(R_hatk); Llk@chol(R_hatk)^-T]
            potrf_l_mn(nu + nx + 1, nu, *RSQrq_hat_curr_p, 0, 0, Llt[k], 0, 0);
            if (!check_reg(nu, &Llt[k].mat(), 0, 0))
                return LinsolReturnFlag::INDEFINITE;
            // Pp_k = Qq_hatk - L_k^T @ Ll_k
            // SYRK_LN_MN(nx+1, nx, nu-rank_k, -1.0,Llt_p+k, nu-rank_k,0, Llt_p+k, nu-rank_k,0, 1.0,
            // RSQrq_hat_curr_p, nu-rank_k, nu-rank_k,Pp+k,0,0); // feature not implmented yet
            gecp(nx + 1, nu, Llt[k], nu, 0, Llt_shift[0], 0,
                 0); // needless operation because feature not implemented yet
            syrk_ln_mn(nx + 1, nx, nu, -1.0, Llt_shift[0], 0, 0, Llt_shift[0], 0, 0, 1.0,
                       *RSQrq_hat_curr_p, nu, nu, Ppt[k], 0, 0);

            if (k > 0){
                // GuGx_hat = GuGx * L^-1
                if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                const Index nunxm1 = info.dims.number_of_controls[k-1] + info.dims.number_of_states[k-1];
                gecp(nunxm1, nu, hessian.GuGx[k-1], 0, 0, GuGx_hat[k-1], 0, 0);
                trsm_rltn(nunxm1, nu, 1.0, Llt[k], 0, 0, GuGx_hat[k-1], 0, 0, GuGx_hat[k-1], 0, 0);

                // RSQrqt = RSQrqt - GuGx^T L^-T L^-1 GuGx
                if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                gemm_nt(nunxm1, nunxm1, nu, -1.0, GuGx_hat[k-1], 0, 0, GuGx_hat[k-1], 0, 0, 
                        1.0, RSQrqt_underbar[k-1], 0, 0, RSQrqt_underbar[k-1], 0, 0);
                if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                trsm_rltn(1, nu, 1.0, Llt[k], 0, 0, RSQrq_hat_curr_p[0], nu + nx, 0, v_r_tilde[0], 0, 0);
                gemm_nt(1, nunxm1, nu, -1.0, v_r_tilde[0], 0, 0, GuGx_hat[k-1], 0, 0,
                        1.0, RSQrqt_underbar[k-1], nunxm1, 0, RSQrqt_underbar[k-1], nunxm1, 0);

                // FuFx = FuFx - GuGx_hat * L
                if (print_debug_lines) {std::cout << __LINE__ << std::endl;}
                gemm_nt(nunxm1, nx, nu, -1.0, GuGx_hat[k-1], 0, 0, Llt[k], nu, 0,
                        1.0, FuFx_underbar[k-1], 0, 0, FuFx_underbar[k-1], 0, 0);
            }

        }
        trtr_l(nx, Ppt[k], 0, 0, Ppt[k], 0, 0);
    }
    //////// FIRST_STAGE
    {
        const Index nx = info.dims.number_of_states[0];
        {
            potrf_l_mn(nx + 1, nx, Ppt[0], 0, 0, LlIt[0], 0, 0);
            if (!check_reg(nx, &LlIt[0].mat(), 0, 0))
                return LinsolReturnFlag::INDEFINITE;
        }
    }
    ////// FORWARD_SUBSTITUTION:
    // first stage
    {
        const Index nx = info.dims.number_of_states[0];
        const Index nu = info.dims.number_of_controls[0];
        const Index offs_x = info.offsets_primal_x[0];
        // calculate xIb
        rowex(nx, -1.0, LlIt[0], nx, 0, x, offs_x);
        // assume TRSV_LTN allows aliasing, this is the case in normal BLAS
        trsv_ltn(nx, LlIt[0], 0, 0, x, offs_x, x, offs_x);
    }
    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_x = info.offsets_primal_x[k];
        rowex(nu, -1.0, Llt[k], nu + nx, 0, x, offs);
        gemv_t(nx, nu, -1.0, Llt[k], nu, 0, x, offs_x, 1.0, x, offs, x, offs);
        trsv_ltn(nu, Llt[k], 0, 0, x, offs, x, offs);
        // + GuGxt [uk-1, xk-1]
        if (k > 0){
            const Index nunxm1 = info.dims.number_of_controls[k-1] + info.dims.number_of_states[k-1];
            trsm_rlnn(nunxm1, nu, 1.0, Llt[k], 0, 0, GuGx_hat[k-1], 0, 0, GuGx_hat[k-1], 0, 0);
            gemv_t(nunxm1, nu, -1.0, GuGx_hat[k-1], 0, 0, x, info.offsets_primal_u[k-1], 
                    1.0, x, offs, x, offs);
        }
        if (k != info.dims.K - 1)
        {
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index nup1 = info.dims.number_of_controls[k + 1];
            const Index offs_x_p1 = info.offsets_primal_x[k + 1];
            const Index offs_dyn_eq_k = info.offsets_g_eq_dyn[k];
            // calculate xkp1
            rowex(nxp1, 1.0, jacobian.BAbt[k], nu + nx, 0, x, offs_x_p1);
            gemv_t(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, x, offs, 1.0, x, offs_x_p1, x,
                   offs_x_p1);
            // calculate lam_dyn xp1
            rowex(nxp1, 1.0, Ppt[k + 1], nxp1, 0, eq_mult, offs_dyn_eq_k);
            gemv_t(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, x, offs_x_p1, 1.0, eq_mult, offs_dyn_eq_k,
                   eq_mult, offs_dyn_eq_k);
            
            gemv_t(nu + nx, nxp1, 1.0, FuFx_underbar[k], 0, 0, x, offs, 1.0, 
                   eq_mult, offs_dyn_eq_k, eq_mult, offs_dyn_eq_k);
        }
        // Covered in post-processing
        // const Index ng = info.dims.number_of_eq_constraints[k];
        // const Index offs_g_eq_k = info.offsets_g_eq_path[k];
        // const Index offs_eq_k = info.offsets_eq[k];
        // if (ng > 0)
        // {
        //     gemv_t(nu + nx, ng, 1.0, jacobian.Gg_eqt[k], 0, 0, x, offs, 1.0, g, offs_g_eq_k,
        //            eq_mult, offs_g_eq_k);
        //     eq_mult.block(ng, offs_g_eq_k) =
        //         eq_mult.block(ng, offs_g_eq_k) / D_eq.block(ng, offs_eq_k);
        // }
        // const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        // const Index offs_slack = info.offsets_slack[k];
        // const Index offs_eq_ineq = info.offsets_g_eq_slack[k];
        // if (ng_ineq > 0)
        // {
        //     gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_eq_ineq,
        //            eq_mult, offs_eq_ineq);
        //     eq_mult.block(ng_ineq, offs_eq_ineq) =
        //         eq_mult.block(ng_ineq, offs_eq_ineq) / D_s.block(ng_ineq, offs_slack);
        // }
    }
    return LinsolReturnFlag::SUCCESS;
}

LinsolReturnFlag ModifiedAugSystemSolver::solve_rhs(const ProblemInfo &info,
                                               const Jacobian<ImplicitOcpType> &jacobian,
                                               const Hessian<ImplicitOcpType> &hessian,
                                               const VecRealView &D_s, const VecRealView &f,
                                               const VecRealView &g, VecRealView &x,
                                               VecRealView &eq_mult)
{
    VecRealView *v_RSQrq_hat_curr_p;
    Index rank_k;
    /////////////// recursion ///////////////

    for (Index k = info.dims.K - 1; k >= 0; --k)
    {
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offset_ineq_k = info.offsets_slack[k];
        const Index offs_g_ineq_k = info.offsets_g_eq_slack[k];
        const Index offs_g_k = info.offsets_g_eq_path[k];
        const Index offs = info.offsets_primal_u[k];
        //         //////// SUBSDYN
        Index gamma_k;
        if (k == info.dims.K - 1)
        {
            gamma_k = ng;
            gamma[k] = gamma_k;
            veccp(ng, g, offs_g_k, v_Ggt_stripe[0], 0);
            veccp(nu + nx, f, offs, v_RSQrqt_tilde[k], 0);
        }
        else
        {
            const Index offs_dyn_k = info.offsets_g_eq_dyn[k];
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index Hp1_size = gamma[k + 1] - rho[k + 1];
            gamma_k = Hp1_size + ng;
            gemv_n(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, g, offs_dyn_k, 0.0, v_AL[0], 0, v_AL[0], 0);
            axpy(nxp1, 1.0, v_Ppt[k + 1], 0, v_AL[0], 0, v_AL[0], 0);
            gemv_n(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, v_AL[0], 0, 1.0, f, offs,
                   v_RSQrqt_tilde[k], 0);
            // For a retained factorization, the affine dynamics RHS is new.
            // Substitution into z[k]^T FuFx[k] x[k+1] adds FuFx[k] g[k]
            // to the current linear term.
            gemv_n(nu + nx, nxp1, 1.0, FuFx_underbar[k], 0, 0,
                   g, offs_dyn_k, 1.0, v_RSQrqt_tilde[k], 0,
                   v_RSQrqt_tilde[k], 0);
            if (gamma_k > 0)
            {
                if (ng > 0)
                {
                    veccp(ng, g, offs_g_k, v_Ggt_stripe[0], 0);
                }
                if (Hp1_size > 0)
                {
                    gemv_n(Hp1_size, nxp1, 1.0, Hh[k + 1], 0, 0, g, offs_dyn_k, 0.0,
                           v_Ggt_stripe[0], ng, v_Ggt_stripe[0], ng);
                    axpy(Hp1_size, 1.0, v_Hh[k + 1], 0, v_Ggt_stripe[0], ng, v_Ggt_stripe[0], ng);
                }
            }
            else
            {
                rank_k = 0;
                v_RSQrq_hat_curr_p = &v_RSQrqt_tilde[k];
            }
        }
        if (ng_ineq > 0)
        {
            for (Index i = 0; i < ng_ineq; i++)
            {
                Scalar scaling_factor = D_s(offset_ineq_k + i);
                Scalar grad_barrier = g(offs_g_ineq_k + i);
                v_Ggt_ineq_temp[0](i) = grad_barrier / scaling_factor;
            }
            gemv_n(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, v_Ggt_ineq_temp[0], 0, 1.0,
                   v_RSQrqt_tilde[k], 0, v_RSQrqt_tilde[k], 0);
        }
        {
            rank_k = rho[k];
            gecp(rank_k, gamma_k, Ggt_tilde[k], nu - rank_k + nx + 1, 0, Ggt_stripe[0], 0, 0);
            Pl[k].apply(rank_k, &v_Ggt_stripe[0].vec(), 0);
            trsv_utu(rank_k, Ggt_stripe[0], 0, 0, v_Ggt_stripe[0], 0, v_Ggt_stripe[0], 0);
            gemv_t(rank_k, gamma_k - rank_k, -1.0, Ggt_stripe[0], 0, rank_k, v_Ggt_stripe[0], 0,
                   1.0, v_Ggt_stripe[0], rank_k, v_Ggt_stripe[0], rank_k);

            if (gamma_k - rank_k > 0)
            {
                veccp(gamma_k - rank_k, v_Ggt_stripe[0], rank_k, v_Hh[k], 0);
            }
            if (rank_k > 0)
            {
                veccpsc(rank_k, -1.0, v_Ggt_stripe[0], 0, v_Ggt_tilde[k], 0);
                trsv_ltn(rank_k, Ggt_stripe[0], 0, 0, v_Ggt_tilde[k], 0, v_Ggt_tilde[k], 0);
                Pr[k].apply(rank_k, &v_RSQrqt_tilde[k].vec(), 0);
                veccp(nu + nx, v_RSQrqt_tilde[k], 0, v_GgLt[0], 0);
                gemv_n(nu + nx, rank_k, 1.0, RSQrqt_tilde[k], 0, 0, v_Ggt_tilde[k], 0, 1.0,
                       v_GgLt[0], 0, v_GgLt[0], 0);
                gemv_n(nu + nx - rank_k, rank_k, 1.0, Ggt_tilde[k], 0, 0, v_GgLt[0], 0, 1.0,
                       v_GgLt[0], rank_k, v_RSQrqt_hat[0], 0);
                v_RSQrq_hat_curr_p = &v_RSQrqt_hat[0];
            }
            else
            {
                v_RSQrq_hat_curr_p = &v_RSQrqt_tilde[k];
            }
        }
        //         //////// SCHUR
        {
            if (nu - rank_k > 0)
            {
                trsv_lnn(nu - rank_k, Llt[k], 0, 0, *v_RSQrq_hat_curr_p, 0, v_Llt[k], 0);
                gecp(nx + 1, nu - rank_k, Llt[k], nu - rank_k, 0, Llt_shift[0], 0, 0);
                veccp(nu - rank_k, v_Llt[k], 0, v_Llt_shift[0], 0);
                veccp(nx, *v_RSQrq_hat_curr_p, nu - rank_k, v_Ppt[k], 0);
                gemv_n(nx, nu - rank_k, -1.0, Llt_shift[0], 0, 0, v_Llt_shift[0], 0, 1.0, v_Ppt[k],
                       0, v_Ppt[k], 0);
                if (increased_accuracy)
                {
                    gemv_t(nu - rank_k, gamma_k - rank_k, -1.0, Ggt_tilde[k], 0, rank_k, v_Llt[k],
                           0, 1.0, v_Hh[k], 0, v_Hh[k], 0);
                }
            }
            else
            {
                veccp(nx, *v_RSQrq_hat_curr_p, 0, v_Ppt[k], 0);
            }
        }
    }
    {
        const Index nx = info.dims.number_of_states[0];
        Index gamma_I = gamma[0] - rho[0];
        if (gamma_I > 0)
        {
            veccp(gamma_I, v_Hh[0], 0, v_HhIt[0], 0);
            PlI[0].apply(rankI, &v_HhIt[0].vec(), 0);
            trsv_utu(rankI, HhIt[0], 0, 0, v_HhIt[0], 0, v_HhIt[0], 0);
            gemv_t(rankI, gamma_I - rankI, -1.0, HhIt[0], 0, rankI, v_HhIt[0], 0, 1.0, v_HhIt[0],
                   rankI, v_HhIt[0], rankI);
            veccpsc(rankI, -1.0, v_HhIt[0], 0, v_GgIt_tilde[0], 0);
            trsv_ltn(rankI, HhIt[0], 0, 0, v_GgIt_tilde[0], 0, v_GgIt_tilde[0], 0);
            PrI[0].apply(rankI, &v_Ppt[0].vec(), 0);
            veccp(nx, v_Ppt[0], 0, v_GgLIt[0], 0);
            gemv_n(nx, rankI, 1.0, Ppt[0], 0, 0, v_GgIt_tilde[0], 0, 1.0, v_GgLIt[0], 0, v_GgLIt[0],
                   0);
            gemv_n(nx - rankI, rankI, 1.0, GgIt_tilde[0], 0, 0, v_GgLIt[0], 0, 1.0, v_GgLIt[0],
                   rankI, v_PpIt_hat[0], 0);
            trsv_lnn(nx - rankI, LlIt[0], 0, 0, v_PpIt_hat[0], 0, v_LlIt[0], 0);
        }
        else
        {
            trsv_lnn(nx, LlIt[0], 0, 0, v_Ppt[0], 0, v_LlIt[0], 0);
        }
    }
    {
        const Index nx = info.dims.number_of_states[0];
        const Index nu = info.dims.number_of_controls[0];
        const Index offs_u = info.offsets_primal_u[0];
        const Index offs_x = info.offsets_primal_x[0];
        const Index offs_g = info.offsets_g_eq_path[0];
        veccpsc(nx - rankI, -1.0, v_LlIt[0], 0, x, offs_x + rankI);
        trsv_ltn(nx - rankI, LlIt[0], 0, 0, x, offs_x + rankI, x, offs_x + rankI);
        veccp(rankI, v_GgIt_tilde[0], 0, x, offs_x);
        gemv_t(nx - rankI, rankI, 1.0, GgIt_tilde[0], 0, 0, x, offs_x + rankI, 1.0, x, offs_x, x,
               offs_x);
        veccpsc(rankI, -1.0, v_Ppt[0], 0, eq_mult, offs_g);
        gemv_t(nx, rankI, -1.0, Ppt[0], 0, 0, x, nu, 1.0, eq_mult, offs_g, eq_mult, offs_g);
        trsv_lnn(rankI, HhIt[0], 0, 0, eq_mult, offs_g, eq_mult, offs_g);
        trsv_unu(rankI, rankI, HhIt[0], 0, 0, eq_mult, offs_g, eq_mult, offs_g);
        PlI[0].apply_inverse(rankI, &eq_mult.vec(), offs_g);
        PrI[0].apply_inverse(rankI, &x.vec(), offs_x);
    }
    for (Index k = 0; k < info.dims.K; k++)
    {

        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_x = info.offsets_primal_x[k];
        const Index rho_k = rho[k];
        const Index numrho_k = nu - rho_k;
        const Index offs_g_k = info.offsets_g_eq_path[k];
        const Index gammamrho_k = gamma[k] - rho[k];
        const Index gamma_k = gamma[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_eq_ineq = info.offsets_g_eq_slack[k];
        const Index offs_slack = info.offsets_slack[k];
        if (numrho_k > 0)
        {
            veccpsc(numrho_k, -1.0, v_Llt[k], 0, x, offs + rho_k);
            if (increased_accuracy)
            {
                gemv_n(nu - rho_k, gamma_k - rho_k, -1.0, Ggt_tilde[k], 0, rho_k, eq_mult, offs_g_k,
                       1.0, x, offs + rho_k, x, offs + rho_k);
            }
            gemv_t(nx, numrho_k, -1.0, Llt[k], numrho_k, 0, x, offs_x, 1.0, x, offs + rho_k, x,
                   offs + rho_k);
            trsv_ltn(numrho_k, Llt[k], 0, 0, x, offs + rho_k, x, offs + rho_k);
        }
        //         /// calcualate uka_tilde
        if (rho_k > 0)
        {
            // ROWEX(rho_k, 1.0, Ggt_tilde[k], numrho_k + nx, 0, ux[0], offs);
            veccp(rho_k, v_Ggt_tilde[k], 0, x, offs);
            gemv_t(nx + numrho_k, rho_k, 1.0, Ggt_tilde[k], 0, 0, x, offs + rho_k, 1.0, x, offs, x,
                   offs);
            veccp(gammamrho_k, eq_mult, offs_g_k, v_tmp[0], 0);
            veccp(gammamrho_k, v_tmp[0], 0, eq_mult, offs_g_k + rho_k);
            veccpsc(rho_k, -1.0, v_RSQrqt_tilde[k], 0, eq_mult, offs_g_k);
            gemv_t(nu + nx, rho_k, -1.0, RSQrqt_tilde[k], 0, 0, x, offs, 1.0, eq_mult, offs_g_k,
                   eq_mult, offs_g_k);
            gecp(rho_k, gamma_k, Ggt_tilde[k], nu - rho_k + nx + 1, 0, AL[0], 0, 0);
            trsv_lnn(rho_k, AL[0], 0, 0, eq_mult, offs_g_k, eq_mult, offs_g_k);
            trsv_unu(rho_k, gamma_k, AL[0], 0, 0, eq_mult, offs_g_k, eq_mult, offs_g_k);
            Pl[k].apply_inverse(rho_k, &eq_mult.vec(), offs_g_k);
            Pr[k].apply_inverse(rho_k, &x.vec(), offs);
        }
        if (ng_ineq > 0)
        {
            gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_eq_ineq,
                   eq_mult, offs_eq_ineq);
            eq_mult.block(ng_ineq, offs_eq_ineq) =
                eq_mult.block(ng_ineq, offs_eq_ineq) / D_s.block(ng_ineq, offs_slack);
        }
        if (k != info.dims.K - 1)
        {
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index nup1 = info.dims.number_of_controls[k + 1];
            const Index offsp1 = info.offsets_primal_u[k + 1];
            const Index offsxp1 = info.offsets_primal_x[k + 1];
            const Index offs_g_kp1 = info.offsets_g_eq_path[k + 1];
            const Index offs_dyn_k = info.offsets_g_eq_dyn[k];
            const Index gammamrho_kp1 = gamma[k + 1] - rho[k + 1];
            veccp(nxp1, g, offs_dyn_k, x, offsxp1);
            gemv_t(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, x, offs, 1.0, x, offsxp1, x,
                   offsxp1);
            veccp(nxp1, v_Ppt[k + 1], 0, eq_mult, offs_dyn_k);
            gemv_t(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, x, offsxp1, 1.0, eq_mult, offs_dyn_k, eq_mult,
                   offs_dyn_k);
            gemv_t(gammamrho_kp1, nxp1, 1.0, Hh[k + 1], 0, 0, eq_mult, offs_g_kp1, 1.0, eq_mult,
                   offs_dyn_k, eq_mult, offs_dyn_k);
            gemv_t(nu + nx, nxp1, 1.0, FuFx_underbar[k], 0, 0,
                   x, offs, 1.0, eq_mult, offs_dyn_k, eq_mult,
                   offs_dyn_k);
        }
    }
    return LinsolReturnFlag::SUCCESS;
}
LinsolReturnFlag ModifiedAugSystemSolver::solve_rhs(const ProblemInfo &info,
                                               const Jacobian<ImplicitOcpType> &jacobian,
                                               const Hessian<ImplicitOcpType> &hessian,
                                               const VecRealView &D_eq, const VecRealView &D_s,
                                               const VecRealView &f, const VecRealView &g,
                                               VecRealView &x, VecRealView &eq_mult)
{
    VecRealView *v_RSQrq_hat_curr_p;
    for (Index k = info.dims.K - 1; k >= 0; --k)
    {
        const Index offs_ux_k = info.offsets_primal_u[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs_g_dyn = info.offsets_g_eq_dyn[k];
        const Index offs_g_eq = info.offsets_g_eq_path[k];
        const Index offs_ge_eq_ineq = info.offsets_g_eq_slack[k];
        //     //////// SUBSDYN
        if (k == info.dims.K - 1)
        {
            veccp(nu + nx, f, offs_ux_k, v_RSQrqt_tilde[k], 0);
        }
        else
        {
            const Index nxp1 = info.dims.number_of_states[k + 1];
            gemv_n(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, g, offs_g_dyn, 0.0, v_AL[0], 0, v_AL[0], 0);
            axpy(nxp1, 1.0, v_Ppt[k + 1], 0, v_AL[0], 0, v_AL[0], 0);
            gemv_n(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, v_AL[0], 0, 1.0, f, offs_ux_k,
                   v_RSQrqt_tilde[k], 0);
            gemv_n(nu + nx, nxp1, 1.0, FuFx_underbar[k], 0, 0,
                   g, offs_g_dyn, 1.0, v_RSQrqt_tilde[k], 0,
                   v_RSQrqt_tilde[k], 0);
        }
        if (ng > 0)
        {
            const Index offs_eq_k = info.offsets_eq[k];
            for (Index i = 0; i < ng; i++)
            {
                Scalar scaling_factor = D_eq(offs_eq_k + i);
                v_Ggt_stripe[0](i) = g(offs_g_eq + i) / scaling_factor;
            }
            gemv_n(nu + nx, ng, 1.0, jacobian.Gg_eqt[k], 0, 0, v_Ggt_stripe[0], 0, 1.0,
                   v_RSQrqt_tilde[k], 0, v_RSQrqt_tilde[k], 0);
        }
        if (ng_ineq > 0)
        {
            const Index offs_ineq_k = info.offsets_slack[k];
            for (Index i = 0; i < ng_ineq; i++)
            {
                Scalar scaling_factor = D_s(offs_ineq_k + i);
                v_Ggt_ineq_temp[0](i) = g(offs_ge_eq_ineq + i) / scaling_factor;
            }
            gemv_n(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, v_Ggt_ineq_temp[0], 0, 1.0,
                   v_RSQrqt_tilde[k], 0, v_RSQrqt_tilde[k], 0);
        }
        {
            v_RSQrq_hat_curr_p = &v_RSQrqt_tilde[k];
        }
        {
            trsv_lnn(nu, Llt[k], 0, 0, *v_RSQrq_hat_curr_p, 0, v_Llt[k], 0);
            veccp(nu, v_Llt[k], 0, v_Llt_shift[0], 0);
            gemv_n(nx, nu, -1.0, Llt[k], nu, 0, v_Llt_shift[0], 0, 1.0, v_RSQrqt_tilde[k], nu,
                   v_Ppt[k], 0);
        }
    }
    {
        const Index nx = info.dims.number_of_states[0];
        {
            trsv_lnn(nx, LlIt[0], 0, 0, v_Ppt[0], 0, v_LlIt[0], 0);
        }
    }
    {
        const Index nx = info.dims.number_of_states[0];
        const Index nu = info.dims.number_of_controls[0];
        const Index offs_x = info.offsets_primal_x[0];
        veccpsc(nx, -1.0, v_LlIt[0], 0, x, offs_x);
        trsv_ltn(nx, LlIt[0], 0, 0, x, offs_x, x, offs_x);
    }
    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_x = info.offsets_primal_x[k];
        const Index offs_dyn_eq_k = info.offsets_g_eq_dyn[k];
        veccpsc(nu, -1.0, v_Llt[k], 0, x, offs);
        gemv_t(nx, nu, -1.0, Llt[k], nu, 0, x, offs_x, 1.0, x, offs, x, offs);
        trsv_ltn(nu, Llt[k], 0, 0, x, offs, x, offs);
        if (k != info.dims.K - 1)
        {
            const Index nxp1 = info.dims.number_of_states[k + 1];
            const Index offsp1 = info.offsets_primal_u[k + 1];
            const Index offs_x_p1 = info.offsets_primal_x[k + 1];
            veccp(nxp1, g, offs_dyn_eq_k, x, offs_x_p1);
            gemv_t(nu + nx, nxp1, 1.0, jacobian.BAbt[k], 0, 0, x, offs, 1.0, x, offs_x_p1, x,
                   offs_x_p1);
            veccp(nxp1, v_Ppt[k + 1], 0, eq_mult, offs_dyn_eq_k);
            gemv_t(nxp1, nxp1, 1.0, Ppt[k + 1], 0, 0, x, offs_x_p1, 1.0, eq_mult,
                   offs_dyn_eq_k, eq_mult, offs_dyn_eq_k);
        }
    }
    // // calculate lam_eq xk
    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_g_k = info.offsets_g_eq_path[k];
        const Index offs_eq = info.offsets_eq[k];
        if (ng > 0)
        {
            gemv_t(nu + nx, ng, 1.0, jacobian.Gg_eqt[k], 0, 0, x, offs, 1.0, g, offs_g_k,
                   eq_mult, offs_g_k);
            eq_mult.block(ng, offs_g_k) =
                eq_mult.block(ng, offs_g_k) / D_eq.block(ng, offs_eq);
        }
    }

    for (Index k = 0; k < info.dims.K; k++)
    {
        const Index nx = info.dims.number_of_states[k];
        const Index nu = info.dims.number_of_controls[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index offs = info.offsets_primal_u[k];
        const Index offs_gineq_k = info.offsets_g_eq_slack[k];
        const Index offs_slack = info.offsets_slack[k];
        if (ng_ineq > 0)
        {
            gemv_t(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, x, offs, 1.0, g, offs_gineq_k,
                   eq_mult, offs_gineq_k);
            eq_mult.block(ng_ineq, offs_gineq_k) =
                eq_mult.block(ng_ineq, offs_gineq_k) / D_s.block(ng_ineq, offs_slack);
        }
    }
    return LinsolReturnFlag::SUCCESS;
}

void ModifiedAugSystemSolver::set_performance_mode(bool set){
    if (set){
        print_debug_lines = false;
        print_initial_stage = false;
        write_factorization_file = false;
    }
}







































































void PrintLuInfo(MatRealAllocated const &JBAbt, MatRealAllocated const &Jt, 
                 PermutationMatrix &Pl, PermutationMatrix &Pr){

        int nx_next = JBAbt.n();
        std::cout << "JBAbt: " << std::endl << JBAbt << std::endl;

        PrintNpArray(Jt, "Jt");

        std::cout << "Pl: "; for (int i = 0; i < nx_next; i++){std::cout << Pl[i] << "  ";} std::cout << std::endl;
        std::cout << "Pr: "; for (int i = 0; i < nx_next; i++){std::cout << Pr[i] << "  ";} std::cout << std::endl;

        // print L and U matrix
        std::cout << "L = np.array([\n";
        for (int i = 0; i < nx_next; i++){
            std::cout << "\t[";
            for (int j = 0; j < nx_next; j++){
                if (i > j){
                    std::cout << JBAbt(j,i);
                } else if (i == j){
                    std::cout << 1.0;
                } else {
                    std::cout << 0.0;
                }
                if (j < nx_next - 1){
                    std::cout << ", ";
                }
            }
            std::cout << "]";
            if (i < nx_next){
                std::cout << ",\n";
            }
        }
        std::cout << "\n])" << std::endl;
        std::cout << "U = np.array([\n";
        for (int i = 0; i < nx_next; i++){
            std::cout << "\t[";
            for (int j = 0; j < nx_next; j++){
                if (i <= j){
                    std::cout << JBAbt(j,i);
                } else {
                    std::cout << 0.0;
                }
                if (j < nx_next - 1){
                    std::cout << ", ";
                }
            }
            std::cout << "]";
            if (i < nx_next){
                std::cout << ",\n";
            }
        }
        std::cout << "\n])" << std::endl;
}

MatRealAllocated GetKKT(const ProblemInfo &info,
                        Jacobian<ImplicitOcpType> &jacobian,
                        Hessian<ImplicitOcpType> &hessian,
                        bool identity_jacobian=false){
    int nb_primal = info.number_of_primal_variables;
    int nb_eq = info.number_of_eq_constraints;
    MatRealAllocated full_kkt_matrix = MatRealAllocated(
        nb_primal + nb_eq, nb_primal + nb_eq);
    MatRealAllocated full_matrix_jacobian = MatRealAllocated(nb_eq, nb_primal);
    MatRealAllocated full_matrix_hessian = MatRealAllocated(nb_primal, nb_primal);

    for (int k = 0; k < info.dims.K-1; k++){
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index offs_ux = info.offsets_primal_u[k];
        Index offs_x_next = info.offsets_primal_x[k + 1];
        Index offs_u_next = info.offsets_primal_u[k + 1];
        Index nx_next = info.dims.number_of_states[k + 1];
        Index nu_next = info.dims.number_of_controls[k + 1];
        Index offs_eq_dyn = info.offsets_g_eq_dyn[k];
        full_matrix_jacobian.block(nx_next, nu + nx, offs_eq_dyn, offs_ux) =
            transpose(jacobian.BAbt[k].block(nu + nx, nx_next, 0, 0));
        full_matrix_hessian.block(nx_next, nu + nx, offs_x_next, offs_ux) = 
            transpose(hessian.FuFx[k]);
        full_matrix_hessian.block(nu + nx, nx_next, offs_ux, offs_x_next) =
            hessian.FuFx[k];
        full_matrix_hessian.block(nu_next, nu + nx, offs_u_next, offs_ux) = 
            transpose(hessian.GuGx[k]);
        full_matrix_hessian.block(nu + nx, nu_next, offs_ux, offs_u_next) =
            hessian.GuGx[k];
        if (identity_jacobian){
            for (int i = 0; i < nx_next; i++){
                full_matrix_jacobian(offs_eq_dyn + i, offs_x_next + i) = -1;    
            }
        } else {
            full_matrix_jacobian.block(nx_next, nx_next, offs_eq_dyn, offs_x_next) = 
                transpose(jacobian.Jt[k]);
        }
    }

    // equality path equality constraints
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng = info.dims.number_of_eq_constraints[k];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_g_eq = info.offsets_g_eq_path[k];
        full_matrix_jacobian.block(ng, nu + nx, offset_g_eq, offset_ux) =
            transpose(jacobian.Gg_eqt[k].block(nu + nx, ng, 0, 0));
    }
    // inequality path constraints
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_g_ineq = info.offsets_g_eq_slack[k];
        full_matrix_jacobian.block(ng_ineq, nu + nx, offset_g_ineq, offset_ux) =
            transpose(jacobian.Gg_ineqt[k].block(nu + nx, ng_ineq, 0, 0));
    }
    // populate the full matrix
    for (Index k = 0; k < info.dims.K; k++)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index offs_ux = info.offsets_primal_u[k];
        full_matrix_hessian.block(nu + nx, nu + nx, offs_ux, offs_ux) =
            hessian.RSQrqt[k].block(nu + nx, nu + nx, 0, 0);
    }
    // set up the full KKT matrix
    full_kkt_matrix.block(info.number_of_primal_variables, info.number_of_primal_variables, 0,
                            0) = full_matrix_hessian;
    full_kkt_matrix.block(info.number_of_primal_variables, info.number_of_eq_constraints, 0,
                            info.number_of_primal_variables) = transpose(full_matrix_jacobian);
    full_kkt_matrix.block(info.number_of_eq_constraints, info.number_of_primal_variables,
                            info.number_of_primal_variables, 0) = full_matrix_jacobian;
    // PrintNpArray(full_kkt_matrix, "KKT");

    return full_kkt_matrix;
}

void VerifyIntermediateSolution(const ProblemInfo &info,
                                Jacobian<ImplicitOcpType> &jacobian,
                                Hessian<ImplicitOcpType> &hessian,
                                VecRealView &x, VecRealView& mult,
                                VecRealView &f, VecRealView g){
    VecRealAllocated solution_grad = VecRealAllocated(info.number_of_primal_variables);
    VecRealAllocated solution_g = VecRealAllocated(info.number_of_eq_constraints);

    hessian.apply_on_right(info, x, 0.0, solution_grad, solution_grad);
    jacobian.transpose_apply_on_right(info, mult, 1.0, solution_grad, solution_grad, true);

    jacobian.apply_on_right(info, x, 0.0, solution_g, solution_g, true);

    double max_diff_grad = 0.0;
    for (int i = 0; i < info.number_of_primal_variables; i++){
        std::cout << std::setw(10) << std::setprecision(5) << solution_grad(i) << "\t-\t" << -f(i) << std::endl;
        max_diff_grad = std::max(max_diff_grad, std::abs(solution_grad(i) + f(i)));
    }
    std::cout << "MAX DIFF GRAD: " << max_diff_grad << std::endl;

    std::cout << "------------" << std::endl;
    double max_diff_g = 0.0;
    for (int i = 0; i < info.number_of_eq_constraints; i++){
        std::cout << std::setw(10) << std::setprecision(5) << solution_g(i) << "\t-\t" << -g(i) << std::endl;
        max_diff_g = std::max(max_diff_g, std::abs(solution_g(i) + g(i)));
    }
    std::cout << "MAX DIFF G: " << max_diff_g << std::endl;
}




AugSystemSolver<ImplicitOcpType>::AugSystemSolver(const ProblemInfo &info)
    : ModifiedAugSystemSolver(info),
      identity_dynamics_solver_(
          std::make_unique<AugSystemSolver<OcpType>>(info))
{
    // Initialize additional members specific to ImplicitOcpType if needed
    int max_nx = *std::max_element(info.dims.number_of_states.begin(), info.dims.number_of_states.end());
    int max_nu = *std::max_element(info.dims.number_of_controls.begin(), info.dims.number_of_controls.end());
    int max_ng = *std::max_element(info.dims.number_of_eq_constraints.begin(), info.dims.number_of_eq_constraints.end());
    
    number_of_states = info.dims.number_of_states;
    number_of_controls = info.dims.number_of_controls;
    number_of_eq_constraints = info.dims.number_of_eq_constraints;
    number_of_ineq_constraints = info.dims.number_of_ineq_constraints;
    
    f_copy.emplace_back(info.number_of_primal_variables);
    g_copy.emplace_back(info.number_of_eq_constraints);
    g_original_copy.emplace_back(info.number_of_eq_constraints);
    D_x_copy.emplace_back(info.number_of_primal_variables);
    D_s_copy.emplace_back(info.number_of_slack_variables);
    D_eq_copy.emplace_back(info.number_of_g_eq_path);
    x_copy.emplace_back(info.number_of_primal_variables);
    eq_mult_copy.emplace_back(info.number_of_eq_constraints + info.number_of_slack_variables);
    batch_primal.emplace_back(info.number_of_primal_variables);
    batch_multipliers.emplace_back(info.number_of_eq_constraints);

    int dim = std::max(max_nu + max_nx + 1, max_ng);
    // Implicit Hessian cross blocks reserve eight extra columns for the
    // rank-aware state-to-input transformation.  TreatStatesAsInputs copies
    // the allocated row stripe, including that capacity, so the scratch
    // matrix must carry the same column padding as well as row padding.
    scratch = std::make_unique<MatRealAllocated>(dim + 8, dim + 8);
    JBAbt.emplace_back(dim + max_nx, max_nx);
    JBAbt_modified.emplace_back(dim + max_nx, max_nx);
};

bool AugSystemSolver<ImplicitOcpType>::CanUseIdentityDynamicsFastPath(
    const ProblemInfo &info,
    const Jacobian<ImplicitOcpType> &jacobian,
    const Hessian<ImplicitOcpType> &hessian) const
{
    for (Index stage = 0; stage + 1 < info.dims.K; ++stage)
    {
        const Index local = info.dims.number_of_controls[stage]
                          + info.dims.number_of_states[stage];
        const Index next_states =
            info.dims.number_of_states[stage + 1];
        const Index next_controls =
            info.dims.number_of_controls[stage + 1];
        for (Index row = 0; row < next_states; ++row)
        for (Index column = 0; column < next_states; ++column)
        {
            const Scalar expected = row == column ? -1.0 : 0.0;
            if (jacobian.Jt[stage](row, column) != expected)
                return false;
        }
        for (Index row = 0; row < local; ++row)
        {
            for (Index column = 0; column < next_states; ++column)
                if (hessian.FuFx[stage](row, column) != 0.0)
                    return false;
            for (Index column = 0; column < next_controls; ++column)
                if (hessian.GuGx[stage](row, column) != 0.0)
                    return false;
        }
    }
    return true;
}

void AugSystemSolver<ImplicitOcpType>::MarkFullRankIdentityDynamics(
    const ProblemInfo &info,
    Jacobian<ImplicitOcpType> &jacobian) const
{
    for (Index stage = 0; stage + 1 < info.dims.K; ++stage)
    {
        jacobian.J_ranks[stage] =
            info.dims.number_of_states[stage + 1];
        jacobian.nb_new_controls[stage] = 0;
    }
}

namespace
{
bool CanUseNormalizedExplicitFastPath(
    const ProblemInfo &original_info,
    const ProblemInfo &modified_info,
    const Jacobian<ImplicitOcpType> &jacobian,
    const Hessian<ImplicitOcpType> &hessian)
{
    if (original_info.number_of_primal_variables
        != modified_info.number_of_primal_variables)
        return false;
    for (Index stage = 0; stage + 1 < original_info.dims.K; ++stage)
    {
        if (jacobian.J_ranks[stage]
            != original_info.dims.number_of_states[stage + 1])
            return false;
        const Index local = modified_info.dims.number_of_controls[stage]
                          + modified_info.dims.number_of_states[stage];
        const Index next_states =
            modified_info.dims.number_of_states[stage + 1];
        const Index next_controls =
            modified_info.dims.number_of_controls[stage + 1];
        for (Index row = 0; row < local; ++row)
        {
            for (Index column = 0; column < next_states; ++column)
                if (hessian.FuFx[stage](row, column) != 0.)
                    return false;
            for (Index column = 0; column < next_controls; ++column)
                if (hessian.GuGx[stage](row, column) != 0.)
                    return false;
        }
    }
    return true;
}

bool HasRankDeficientDynamics(
    const ProblemInfo &info,
    const Jacobian<ImplicitOcpType> &jacobian)
{
    for (Index stage = 0; stage + 1 < info.dims.K; ++stage)
        if (jacobian.J_ranks[stage]
            < info.dims.number_of_states[stage + 1])
            return true;
    return false;
}

bool HasAdjacentCurvature(
    const ProblemInfo &info,
    const Hessian<ImplicitOcpType> &hessian)
{
    bool adjacent_curvature = false;
    for (Index stage = 0; stage + 1 < info.dims.K; ++stage)
    {
        const Index local = info.dims.number_of_controls[stage]
                          + info.dims.number_of_states[stage];
        const Index next_states = info.dims.number_of_states[stage + 1];
        const Index next_controls = info.dims.number_of_controls[stage + 1];
        for (Index row = 0; row < local; ++row)
        {
            for (Index column = 0; column < next_states; ++column)
                adjacent_curvature = adjacent_curvature
                    || hessian.FuFx[stage](row, column) != 0.0;
            for (Index column = 0; column < next_controls; ++column)
                adjacent_curvature = adjacent_curvature
                    || hessian.GuGx[stage](row, column) != 0.0;
        }
    }
    return adjacent_curvature;
}

bool NeedsRankDeficientAdjacentCurvatureRefactor(
    const ProblemInfo &info,
    const Jacobian<ImplicitOcpType> &jacobian,
    const Hessian<ImplicitOcpType> &hessian)
{
    return HasRankDeficientDynamics(info, jacobian)
        && HasAdjacentCurvature(info, hessian);
}

ProblemInfo MakeNormalizedExplicitInfo(
    const ProblemInfo &modified_info, const bool stabilized)
{
    std::vector<Index> equalities = stabilized
        ? std::vector<Index>(modified_info.dims.K, 0)
        : modified_info.dims.number_of_eq_constraints;
    std::vector<Index> inequalities(modified_info.dims.K, 0);
    return ProblemInfo(ProblemDims(
        modified_info.dims.K,
        modified_info.dims.number_of_controls,
        modified_info.dims.number_of_states,
        equalities, inequalities, 0,
        modified_info.dims.number_of_stage_border_variables));
}

void PackNormalizedExplicitConstraintRhs(
    const ProblemInfo &modified_info,
    const ProblemInfo &normalized_info,
    const bool stabilized,
    const VecRealView &source,
    VecRealView &destination)
{
    destination = 0.;
    for (Index stage = 0; stage < modified_info.dims.K; ++stage)
    {
        if (!stabilized)
        {
            const Index equalities =
                modified_info.dims.number_of_eq_constraints[stage];
            veccp(
                equalities, source,
                modified_info.offsets_g_eq_path[stage],
                destination,
                normalized_info.offsets_g_eq_path[stage]);
        }
        if (stage + 1 < modified_info.dims.K)
        {
            const Index dynamics =
                modified_info.dims.number_of_states[stage + 1];
            veccp(
                dynamics, source,
                modified_info.offsets_g_eq_dyn[stage],
                destination,
                normalized_info.offsets_g_eq_dyn[stage]);
        }
    }
}

void UnpackNormalizedExplicitMultipliers(
    const ProblemInfo &modified_info,
    const ProblemInfo &normalized_info,
    const bool stabilized,
    const VecRealView &source,
    VecRealView &destination)
{
    destination = 0.;
    for (Index stage = 0; stage < modified_info.dims.K; ++stage)
    {
        if (!stabilized)
        {
            const Index equalities =
                modified_info.dims.number_of_eq_constraints[stage];
            veccp(
                equalities, source,
                normalized_info.offsets_g_eq_path[stage],
                destination,
                modified_info.offsets_g_eq_path[stage]);
        }
        if (stage + 1 < modified_info.dims.K)
        {
            const Index dynamics =
                modified_info.dims.number_of_states[stage + 1];
            veccp(
                dynamics, source,
                normalized_info.offsets_g_eq_dyn[stage],
                destination,
                modified_info.offsets_g_eq_dyn[stage]);
        }
    }
}
} // namespace

void AugSystemSolver<ImplicitOcpType>::register_options(
    OptionRegistry &registry)
{
    registry.register_option(
        "linsol_it_ref",
        &AugSystemSolver<ImplicitOcpType>::set_it_ref, this);
    registry.register_option(
        "linsol_perturbed_mode",
        &AugSystemSolver<ImplicitOcpType>::set_perturbed_mode, this);
    registry.register_option(
        "linsol_perturbed_mode_param",
        &AugSystemSolver<ImplicitOcpType>::set_perturbed_mode_param, this);
    registry.register_option(
        "linsol_lu_fact_tol",
        &AugSystemSolver<ImplicitOcpType>::set_lu_fact_tol, this);
    registry.register_option(
        "linsol_diagnostic",
        &AugSystemSolver<ImplicitOcpType>::set_diagnostic, this);
    registry.register_option(
        "linsol_increased_accuracy",
        &AugSystemSolver<ImplicitOcpType>::set_increased_accuracy, this);
}

LinsolReturnFlag
AugSystemSolver<ImplicitOcpType>::SolveStabilizedRankDeficient(
    const ProblemInfo &original_info,
    const ProblemInfo &modified_info,
    Jacobian<ImplicitOcpType> &jacobian,
    Hessian<ImplicitOcpType> &hessian,
    const VecRealView &D_x,
    const VecRealView &D_s,
    const VecRealView &f,
    const VecRealView &g,
    VecRealView &x,
    VecRealView &eq_mult)
{
    // PreProcess has already eliminated the regularized original path
    // multipliers into H and f. Rank-deficient dynamics additionally created
    // hard path rows. Solve only those promoted rows as equalities; treating
    // them with D_eq would change the original KKT system.
    std::vector<Index> hard_equalities(
        static_cast<std::size_t>(modified_info.dims.K), 0);
    for (Index stage = 0; stage + 1 < modified_info.dims.K; ++stage)
        hard_equalities[stage] = jacobian.nb_new_controls[stage];

    const ProblemDims hard_dims(
        modified_info.dims.K,
        modified_info.dims.number_of_controls,
        modified_info.dims.number_of_states,
        hard_equalities,
        modified_info.dims.number_of_ineq_constraints);
    const ProblemInfo hard_info(hard_dims);
    VecRealAllocated hard_g(hard_info.number_of_eq_constraints);
    VecRealAllocated hard_multipliers(hard_info.number_of_eq_constraints);
    hard_g = 0.0;
    hard_multipliers = 0.0;

    for (Index stage = 0; stage < hard_info.dims.K; ++stage)
    {
        const Index local = hard_info.dims.number_of_controls[stage]
                          + hard_info.dims.number_of_states[stage];
        const Index original_equalities =
            original_info.dims.number_of_eq_constraints[stage];
        const Index hard_count = hard_equalities[stage];
        if (hard_count > 0)
        {
            // Columns are [original regularized rows, promoted hard rows].
            // Move only the latter to the front for the hard-only view.
            gecp(local + 1, hard_count, jacobian.Gg_eqt[stage],
                 0, original_equalities, *scratch, 0, 0);
            gecp(local + 1, hard_count, *scratch, 0, 0,
                 jacobian.Gg_eqt[stage], 0, 0);
            veccp(hard_count, g,
                  modified_info.offsets_g_eq_path[stage]
                      + original_equalities,
                  hard_g, hard_info.offsets_g_eq_path[stage]);
        }

        if (stage + 1 < hard_info.dims.K)
        {
            const Index dynamics = hard_info.dims.number_of_states[stage + 1];
            veccp(dynamics, g, modified_info.offsets_g_eq_dyn[stage],
                  hard_g, hard_info.offsets_g_eq_dyn[stage]);
        }
        const Index inequalities =
            hard_info.dims.number_of_ineq_constraints[stage];
        veccp(inequalities, g, modified_info.offsets_g_eq_slack[stage],
              hard_g, hard_info.offsets_g_eq_slack[stage]);
    }

    const LinsolReturnFlag status = ModifiedAugSystemSolver::solve(
        hard_info, jacobian, hessian, D_x, D_s, f, hard_g,
        x, hard_multipliers);
    if (status != LinsolReturnFlag::SUCCESS)
        return status;

    eq_mult = 0.0;
    for (Index stage = 0; stage < hard_info.dims.K; ++stage)
    {
        const Index original_equalities =
            original_info.dims.number_of_eq_constraints[stage];
        const Index hard_count = hard_equalities[stage];
        veccp(hard_count, hard_multipliers,
              hard_info.offsets_g_eq_path[stage], eq_mult,
              modified_info.offsets_g_eq_path[stage]
                  + original_equalities);
        if (stage + 1 < hard_info.dims.K)
        {
            const Index dynamics = hard_info.dims.number_of_states[stage + 1];
            veccp(dynamics, hard_multipliers,
                  hard_info.offsets_g_eq_dyn[stage], eq_mult,
                  modified_info.offsets_g_eq_dyn[stage]);
        }
        const Index inequalities =
            hard_info.dims.number_of_ineq_constraints[stage];
        veccp(inequalities, hard_multipliers,
              hard_info.offsets_g_eq_slack[stage], eq_mult,
              modified_info.offsets_g_eq_slack[stage]);
    }
    return status;
}

LinsolReturnFlag AugSystemSolver<ImplicitOcpType>::solve(const ProblemInfo &info,
                                           Jacobian<ImplicitOcpType> &jacobian, Hessian<ImplicitOcpType> &hessian,
                                           const VecRealView &D_x, const VecRealView &D_s,
                                           const VecRealView &f, const VecRealView &g,
                                           VecRealView &x, VecRealView &eq_mult)
{
    identity_factorization_ready_ = false;
    normalized_explicit_factorization_ready_ = false;
    normalized_explicit_stabilized_ = false;
    if (CanUseIdentityDynamicsFastPath(info, jacobian, hessian))
    {
        MarkFullRankIdentityDynamics(info, jacobian);
        const LinsolReturnFlag status = identity_dynamics_solver_->solve(
            info, static_cast<Jacobian<OcpType> &>(jacobian),
            static_cast<Hessian<OcpType> &>(hessian),
            D_x, D_s, f, g, x, eq_mult);
        identity_factorization_ready_ =
            status == LinsolReturnFlag::SUCCESS;
        return status;
    }
    if (print_debug) {std::cout << "AugSystemSolver<ImplicitOcpType> solve start" << std::endl;}
    // copy the rhs since they are altered during preprocessing and are needed for checking the solution
    auto start = std::chrono::high_resolution_clock::now();
    #ifdef PROFILE
    #endif
    veccp(info.number_of_primal_variables, f, 0, f_copy[0], 0);
    veccp(info.number_of_eq_constraints, g, 0, g_copy[0], 0);
    veccp(info.number_of_primal_variables, D_x, 0, D_x_copy[0], 0);
    veccp(info.number_of_slack_variables, D_s, 0, D_s_copy[0], 0);

    #ifdef PROFILE
    auto stop = std::chrono::high_resolution_clock::now();
    duration_copying_rhs = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    // start = std::chrono::high_resolution_clock::now();
    #endif

    ProblemInfo modified_info = PreProcess(info, jacobian, hessian, f_copy[0], g_copy[0], &(D_x_copy[0]), nullptr, &(D_s_copy[0]));
    auto stop = std::chrono::high_resolution_clock::now();
    duration_preprocess = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

    start = std::chrono::high_resolution_clock::now();
    LinsolReturnFlag flag;
    if (CanUseNormalizedExplicitFastPath(
            info, modified_info, jacobian, hessian))
    {
        const ProblemInfo normalized_info =
            MakeNormalizedExplicitInfo(modified_info, false);
        VecRealView normalized_g(
            g_original_copy[0],
            normalized_info.number_of_eq_constraints, 0);
        VecRealView normalized_multipliers(
            batch_multipliers[0],
            normalized_info.number_of_eq_constraints, 0);
        PackNormalizedExplicitConstraintRhs(
            modified_info, normalized_info, false,
            g_copy[0], normalized_g);
        D_x_copy[0] = 0.;
        normalized_multipliers = 0.;
        flag = identity_dynamics_solver_->solve(
            normalized_info,
            static_cast<Jacobian<OcpType> &>(jacobian),
            static_cast<Hessian<OcpType> &>(hessian),
            D_x_copy[0], D_s_copy[0], f_copy[0], normalized_g,
            x, normalized_multipliers);
        UnpackNormalizedExplicitMultipliers(
            modified_info, normalized_info, false,
            normalized_multipliers, eq_mult);
        normalized_explicit_factorization_ready_ =
            flag == LinsolReturnFlag::SUCCESS;
    }
    else
    {
        flag = ModifiedAugSystemSolver::solve(
            modified_info, jacobian, hessian, D_x_copy[0],
            D_s_copy[0], f_copy[0], g_copy[0], x, eq_mult);
    }
    auto end = std::chrono::high_resolution_clock::now();
    duration_solve = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();
    if (write_preprocessing_file){
        PrintPreProcessNpInfo(info, modified_info, hessian, jacobian, x, eq_mult, D_x, preprocessing_file_name);
    }

    if (print_preprocessed_system){
        std::cout << "KKT matrix:" << std::endl;
        PrintNpArray(GetKKT(modified_info, jacobian, hessian, true), "KKT");
        VecRealAllocated full_rhs = VecRealAllocated(modified_info.number_of_primal_variables + modified_info.number_of_eq_constraints);
        for (Index i = 0; i < info.number_of_primal_variables; ++i){full_rhs(i) = (f_copy[0])(i) + (D_x_copy[0])(i)*x(i);}
        for (Index i = 0; i < info.number_of_eq_constraints; ++i){full_rhs(info.number_of_primal_variables + i) = (g_copy[0])(i);}
        for (Index i = 0; i < info.number_of_slack_variables; ++i){
            full_rhs(info.number_of_primal_variables + info.offset_g_eq_slack + i) -= D_s(i) * eq_mult(info.offset_g_eq_slack + i);
        }
        PrintNpArray(full_rhs, "rhs");
        std::cout << "obtained x:" << std::endl << x << std::endl;
        std::cout << "obtained eq_mult:" << std::endl << eq_mult << std::endl;
    }
    if (verify_preprocessed_solution){
        VerifyIntermediateSolution(modified_info, jacobian, hessian, x, eq_mult, f_copy[0], g_copy[0]);    
    }
    #ifdef PROFILE
    stop = std::chrono::high_resolution_clock::now();
    duration_printing_preprocessed = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

    start = std::chrono::high_resolution_clock::now();
    #endif
    PostProcess(info, modified_info, jacobian, hessian, x, eq_mult, &D_s, nullptr, g);
    stop = std::chrono::high_resolution_clock::now();
    duration_postprocess = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

    if (print_final_solution){
        std::string file = "final_solution.py";
        std::ofstream o(file);
        o << "import numpy as np\n\n";
        MatRealAllocated x_mat(info.number_of_primal_variables, 1);
        for (int i = 0; i < info.number_of_primal_variables; i++){
            x_mat(i,0) = x(i);
        }
        PrintNpArray(x_mat, "x", info.number_of_primal_variables, 1, true, o);
        MatRealAllocated eq_mult_mat(info.number_of_eq_constraints, 1);
        for (int i = 0; i < info.number_of_eq_constraints; i++){
            eq_mult_mat(i,0) = eq_mult(i);
        }
        PrintNpArray(eq_mult_mat, "eq_mult", info.number_of_eq_constraints, 1, true, o);
    }
    if (print_debug) {std::cout << "AugSystemSolver<ImplicitOcpType> solve end" << std::endl;}
    return flag;
}
LinsolReturnFlag AugSystemSolver<ImplicitOcpType>::solve(const ProblemInfo &info,
                                           Jacobian<ImplicitOcpType> &jacobian, Hessian<ImplicitOcpType> &hessian,
                                           const VecRealView &D_x, const VecRealView &D_eq,
                                           const VecRealView &D_s, const VecRealView &f,
                                           const VecRealView &g, VecRealView &x,
                                           VecRealView &eq_mult)
{
    identity_factorization_ready_ = false;
    normalized_explicit_factorization_ready_ = false;
    normalized_explicit_stabilized_ = false;
    if (CanUseIdentityDynamicsFastPath(info, jacobian, hessian))
    {
        MarkFullRankIdentityDynamics(info, jacobian);
        const LinsolReturnFlag status = identity_dynamics_solver_->solve(
            info, static_cast<Jacobian<OcpType> &>(jacobian),
            static_cast<Hessian<OcpType> &>(hessian),
            D_x, D_eq, D_s, f, g, x, eq_mult);
        identity_factorization_ready_ =
            status == LinsolReturnFlag::SUCCESS;
        return status;
    }
    if (print_debug) {std::cout << "AugSystemSolver<ImplicitOcpType> solve start" << std::endl;}
    // copy the rhs since they are altered during preprocessing and are needed for checking the solution
    #ifdef PROFILE
    auto start = std::chrono::high_resolution_clock::now();
    #endif
    veccp(info.number_of_primal_variables, f, 0, f_copy[0], 0);
    veccp(info.number_of_eq_constraints, g, 0, g_copy[0], 0);
    veccp(info.number_of_primal_variables, D_x, 0, D_x_copy[0], 0);
    veccp(info.number_of_g_eq_path, D_eq, 0, D_eq_copy[0], 0);
    veccp(info.number_of_slack_variables, D_s, 0, D_s_copy[0], 0);

    #ifdef PROFILE
    auto stop = std::chrono::high_resolution_clock::now();
    duration_copying_rhs = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    #endif

    // return LinsolReturnFlag::SUCCESS;
    ProblemInfo modified_info = PreProcess(info, jacobian, hessian, f_copy[0], g_copy[0], &D_x_copy[0], &D_eq_copy[0], &D_s_copy[0]);
    #ifdef PROFILE
    start = std::chrono::high_resolution_clock::now();
    #endif
    // Path-equality regularization is condensed by the implicit
    // preprocessing and changes the inertia seen by the Riccati recursion.
    // Removing those rows from a normalized explicit view is therefore not
    // equivalent to the stabilized modified recursion.  Keep this case on
    // the verified implementation; only the unstabilized full-rank system
    // may use the normalized-explicit fast path.
    const bool rank_deficient =
        HasRankDeficientDynamics(info, jacobian);
    LinsolReturnFlag flag = rank_deficient
        ? SolveStabilizedRankDeficient(
            info, modified_info, jacobian, hessian, D_x_copy[0],
            D_s_copy[0], f_copy[0], g_copy[0], x, eq_mult)
        : ModifiedAugSystemSolver::solve(
            modified_info, jacobian, hessian, D_x_copy[0],
            D_eq_copy[0], D_s_copy[0], f_copy[0], g_copy[0],
            x, eq_mult);
    #ifdef PROFILE
    auto end = std::chrono::high_resolution_clock::now();
    duration_solve = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    #endif
    PostProcess(info, modified_info, jacobian, hessian, x, eq_mult, &D_s, &D_eq, g);
    if (print_debug) {std::cout << "AugSystemSolver<ImplicitOcpType> solve end" << std::endl;}
    return flag;
}

LinsolReturnFlag AugSystemSolver<ImplicitOcpType>::solve_rhs(const ProblemInfo &info,
                                               Jacobian<ImplicitOcpType> &jacobian,
                                               Hessian<ImplicitOcpType> &hessian,
                                               const VecRealView &D_s, const VecRealView &f,
                                               const VecRealView &g, VecRealView &x,
                                               VecRealView &eq_mult)
{
    if (identity_factorization_ready_)
        return identity_dynamics_solver_->solve_rhs(
            info, static_cast<const Jacobian<OcpType> &>(jacobian),
            static_cast<const Hessian<OcpType> &>(hessian),
            D_s, f, g, x, eq_mult);
    if (normalized_explicit_factorization_ready_)
    {
        fatrop_assert_msg(
            !normalized_explicit_stabilized_,
            "An unstabilized RHS cannot reuse a stabilized normalized factorization.");
        veccp(info.number_of_primal_variables, f, 0, f_copy[0], 0);
        veccp(info.number_of_eq_constraints, g, 0, g_copy[0], 0);
        veccp(info.number_of_slack_variables, D_s, 0, D_s_copy[0], 0);
        const ProblemInfo modified_info = PreProcess(
            info, jacobian, hessian, f_copy[0], g_copy[0],
            nullptr, nullptr, &D_s_copy[0]);
        const ProblemInfo normalized_info =
            MakeNormalizedExplicitInfo(modified_info, false);
        VecRealView normalized_g(
            g_original_copy[0], normalized_info.number_of_eq_constraints, 0);
        VecRealView normalized_multipliers(
            batch_multipliers[0], normalized_info.number_of_eq_constraints, 0);
        PackNormalizedExplicitConstraintRhs(
            modified_info, normalized_info, false, g_copy[0], normalized_g);
        normalized_multipliers = 0.;
        const LinsolReturnFlag flag = identity_dynamics_solver_->solve_rhs(
            normalized_info,
            static_cast<const Jacobian<OcpType> &>(jacobian),
            static_cast<const Hessian<OcpType> &>(hessian),
            D_s_copy[0], f_copy[0], normalized_g, x,
            normalized_multipliers);
        UnpackNormalizedExplicitMultipliers(
            modified_info, normalized_info, false,
            normalized_multipliers, eq_mult);
        PostProcess(
            info, modified_info, jacobian, hessian, x, eq_mult,
            &D_s, nullptr, g);
        return flag;
    }
    const bool refactor_rank_deficient_curvature =
        NeedsRankDeficientAdjacentCurvatureRefactor(
            info, jacobian, hessian);
    #ifdef PROFILE
    auto start = std::chrono::high_resolution_clock::now();
    #endif
    veccp(info.number_of_primal_variables, f, 0, f_copy[0], 0);
    veccp(info.number_of_eq_constraints, g, 0, g_copy[0], 0);
    veccp(info.number_of_slack_variables, D_s, 0, D_s_copy[0], 0);
    #ifdef PROFILE
    auto stop = std::chrono::high_resolution_clock::now();
    duration_copying_rhs = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    #endif

    // return LinsolReturnFlag::SUCCESS;
    ProblemInfo modified_info = PreProcess(
        info, jacobian, hessian, f_copy[0], g_copy[0],
        refactor_rank_deficient_curvature ? &D_x_copy[0] : nullptr,
        nullptr, &D_s_copy[0]);
    #ifdef PROFILE
    start = std::chrono::high_resolution_clock::now();
    #endif
    // Rank-deficient preprocessing promotes null-space states to controls,
    // which creates GuGx coupling.  Its retained-RHS back-substitution does
    // not yet cache both triangular GuGx response forms.  Refactor through
    // the same O(N) structured recursion for correctness in that corner;
    // full-rank exact-curvature systems retain factor reuse.
    LinsolReturnFlag flag = refactor_rank_deficient_curvature
        ? ModifiedAugSystemSolver::solve(
            modified_info, jacobian, hessian, D_x_copy[0],
            D_s_copy[0], f_copy[0], g_copy[0], x, eq_mult)
        : ModifiedAugSystemSolver::solve_rhs(
            modified_info, jacobian, hessian, D_s_copy[0],
            f_copy[0], g_copy[0], x, eq_mult);
    #ifdef PROFILE
    auto end = std::chrono::high_resolution_clock::now();
    duration_solve = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    #endif
    PostProcess(info, modified_info, jacobian, hessian, x, eq_mult, &D_s, nullptr, g);
    return flag;
}
LinsolReturnFlag AugSystemSolver<ImplicitOcpType>::solve_rhs(const ProblemInfo &info,
                                               Jacobian<ImplicitOcpType> &jacobian,
                                               Hessian<ImplicitOcpType> &hessian,
                                               const VecRealView &D_eq, const VecRealView &D_s,
                                               const VecRealView &f, const VecRealView &g,
                                               VecRealView &x, VecRealView &eq_mult)
{
    if (identity_factorization_ready_)
        return identity_dynamics_solver_->solve_rhs(
            info, static_cast<const Jacobian<OcpType> &>(jacobian),
            static_cast<const Hessian<OcpType> &>(hessian),
            D_eq, D_s, f, g, x, eq_mult);
    fatrop_assert_msg(
        !normalized_explicit_factorization_ready_,
        "A stabilized RHS cannot reuse an unstabilized normalized factorization.");
    const bool rank_deficient =
        HasRankDeficientDynamics(info, jacobian);
    const bool adjacent_curvature = HasAdjacentCurvature(info, hessian);
    // Promoted null-space dynamics rows are hard equalities, whereas D_eq
    // regularizes only the original path rows. Exact adjacent curvature also
    // needs the factor-and-solve multiplier back-substitution. Both cases use
    // a fresh O(N) structured factorization for a retained RHS.
    const bool refactor = rank_deficient || adjacent_curvature;
    #ifdef PROFILE
    auto start = std::chrono::high_resolution_clock::now();
    #endif
    veccp(info.number_of_primal_variables, f, 0, f_copy[0], 0);
    veccp(info.number_of_eq_constraints, g, 0, g_copy[0], 0);
    veccp(info.number_of_g_eq_path, D_eq, 0, D_eq_copy[0], 0);
    veccp(info.number_of_slack_variables, D_s, 0, D_s_copy[0], 0);

    #ifdef PROFILE
    auto stop = std::chrono::high_resolution_clock::now();
    duration_copying_rhs = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    #endif

    // return LinsolReturnFlag::SUCCESS;
    ProblemInfo modified_info = PreProcess(
        info, jacobian, hessian, f_copy[0], g_copy[0],
        refactor ? &D_x_copy[0] : nullptr,
        &D_eq_copy[0], &D_s_copy[0]);

    // PreProcess has already condensed the regularized path-equality and
    // inequality rows into the primal right-hand side.  The reused-RHS
    // recursion normally performs that condensation itself; leaving these
    // entries in g_copy therefore adds A^T D^-1 g a second time (and, for
    // equalities, uses the already scaled Jacobian once more).  Keep dynamics
    // residuals intact, but suppress the local rows that were condensed
    // above.  PostProcess receives the original g and reconstructs their
    // multipliers from A x + g.
    for (Index k = 0; k < info.dims.K; ++k)
    {
        vecse(
            info.dims.number_of_eq_constraints[k], 0.0,
            g_copy[0], modified_info.offsets_g_eq_path[k]);
        vecse(
            info.dims.number_of_ineq_constraints[k], 0.0,
            g_copy[0], modified_info.offsets_g_eq_slack[k]);
    }
    #ifdef PROFILE
    start = std::chrono::high_resolution_clock::now();
    #endif
    LinsolReturnFlag flag = rank_deficient
        ? SolveStabilizedRankDeficient(
            info, modified_info, jacobian, hessian, D_x_copy[0],
            D_s_copy[0], f_copy[0], g_copy[0], x, eq_mult)
        : adjacent_curvature
        ? ModifiedAugSystemSolver::solve(
            modified_info, jacobian, hessian, D_x_copy[0],
            D_eq_copy[0], D_s_copy[0], f_copy[0], g_copy[0],
            x, eq_mult)
        : ModifiedAugSystemSolver::solve_rhs(
            modified_info, jacobian, hessian, D_eq_copy[0],
            D_s_copy[0], f_copy[0], g_copy[0], x, eq_mult);
    #ifdef PROFILE
    auto end = std::chrono::high_resolution_clock::now();
    duration_solve = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    #endif
    PostProcess(info, modified_info, jacobian, hessian, x, eq_mult, &D_s, &D_eq, g);
    return flag;
}

void AugSystemSolver<ImplicitOcpType>::PrepareFullRankRhs(
    const ProblemInfo &info,
    Jacobian<ImplicitOcpType> &jacobian,
    const VecRealView *D_eq,
    const VecRealView &D_s,
    const MatRealView &f,
    const MatRealView &g,
    const Index column)
{
    for (Index row = 0; row < info.number_of_primal_variables; ++row)
        f_copy[0](row) = f(row, column);
    for (Index row = 0; row < info.number_of_eq_constraints; ++row)
    {
        const Scalar value = g(row, column);
        g_copy[0](row) = value;
        g_original_copy[0](row) = value;
    }

    // Eliminate the regularized local duals in the original coordinates:
    //
    //   [ H  A^T ] [dx] = -[f]
    //   [ A  -D  ] [dl]   [g]
    //
    // gives (H + A^T D^-1 A) dx = -(f + A^T D^-1 g).
    // Coefficient preprocessing has already formed the left-hand side; this
    // is its right-hand-side counterpart for the current batch column.
    for (Index k = 0; k < info.dims.K; ++k)
    {
        const Index nu = info.dims.number_of_controls[k];
        const Index nx = info.dims.number_of_states[k];
        const Index ng = info.dims.number_of_eq_constraints[k];
        const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        const Index primal_offset = info.offsets_primal_u[k];
        const Index equality_offset = info.offsets_g_eq_path[k];
        const Index equality_diagonal_offset = info.offsets_eq[k];
        const Index inequality_offset = info.offsets_g_eq_slack[k];
        const Index slack_offset = info.offsets_slack[k];

        for (Index variable = 0; variable < nu + nx; ++variable)
        {
            Scalar value = f_copy[0](primal_offset + variable);
            for (Index equation = 0;
                 D_eq != nullptr && equation < ng; ++equation)
            {
                value +=
                    jacobian.Gg_eqt_original[k](variable, equation)
                  * g(equality_offset + equation, column)
                  / (*D_eq)(equality_diagonal_offset + equation);
            }
            for (Index equation = 0; equation < ng_ineq; ++equation)
            {
                value +=
                    jacobian.Gg_ineqt_original[k](variable, equation)
                  * g(inequality_offset + equation, column)
                  / D_s(slack_offset + equation);
            }
            f_copy[0](primal_offset + variable) = value;
        }

        if (D_eq != nullptr)
            vecse(ng, 0.0, g_copy[0], equality_offset);
        vecse(ng_ineq, 0.0, g_copy[0], inequality_offset);
    }

    for (Index k = 0; k + 1 < info.dims.K; ++k)
    {
        const Index nx_next = info.dims.number_of_states[k + 1];
        const Index dynamics_offset = info.offsets_g_eq_dyn[k];

        // Apply the cached full-pivoted factorization of dF/dx[k+1] to
        // the dynamics residual, exactly as coefficient PreProcess applies
        // it to the last row of [J^T; B^T; g^T].
        for (Index state = 0; state < nx_next; ++state)
            (*scratch)(0, state) = g(dynamics_offset + state, column);
        jacobian.Pl_pre[k].apply_on_cols(
            nx_next, &scratch->mat(), 0, 0, 1);
        trsm_runu(
            1, nx_next, 1.0,
            jacobian.Jt_LU[k], 0, 0,
            *scratch, 0, 0,
            *scratch, 0, 0);
        trsm_rlnn(
            1, nx_next, -1.0,
            jacobian.Jt_LU[k], 0, 0,
            *scratch, 0, 0,
            *scratch, 0, 0);
        for (Index state = 0; state < nx_next; ++state)
            g_copy[0](dynamics_offset + state) = (*scratch)(0, state);

        // The same right permutation is a coordinate permutation of the
        // next-stage state block.  Controls retain their original order.
        jacobian.Pr_pre[k].apply(
            nx_next, &f_copy[0].vec(),
            info.offsets_primal_x[k + 1]);
    }
}

LinsolReturnFlag AugSystemSolver<ImplicitOcpType>::solve_rhs_batch(
    const ProblemInfo &info,
    Jacobian<ImplicitOcpType> &jacobian,
    Hessian<ImplicitOcpType> &hessian,
    const VecRealView &D_s,
    const MatRealView &f,
    const MatRealView &g,
    MatRealView &x,
    MatRealView &eq_mult)
{
    return SolveFullRankRhsBatch(
        info, jacobian, hessian, nullptr, D_s,
        f, g, x, eq_mult);
}

LinsolReturnFlag AugSystemSolver<ImplicitOcpType>::solve_rhs_batch(
    const ProblemInfo &info,
    Jacobian<ImplicitOcpType> &jacobian,
    Hessian<ImplicitOcpType> &hessian,
    const VecRealView &D_eq,
    const VecRealView &D_s,
    const MatRealView &f,
    const MatRealView &g,
    MatRealView &x,
    MatRealView &eq_mult)
{
    return SolveFullRankRhsBatch(
        info, jacobian, hessian, &D_eq, D_s,
        f, g, x, eq_mult);
}

LinsolReturnFlag AugSystemSolver<ImplicitOcpType>::SolveFullRankRhsBatch(
    const ProblemInfo &info,
    Jacobian<ImplicitOcpType> &jacobian,
    Hessian<ImplicitOcpType> &hessian,
    const VecRealView *D_eq,
    const VecRealView &D_s,
    const MatRealView &f,
    const MatRealView &g,
    MatRealView &x,
    MatRealView &eq_mult)
{
    fatrop_assert_msg(
        f.m() == info.number_of_primal_variables
        && g.m() == info.number_of_eq_constraints
        && x.m() == info.number_of_primal_variables
        && eq_mult.m() == info.number_of_eq_constraints
        && f.n() == g.n()
        && f.n() == x.n()
        && f.n() == eq_mult.n(),
        "The implicit batch right-hand-side dimensions are inconsistent.");
    fatrop_assert_msg(
        (!D_eq || D_eq->m() == info.number_of_g_eq_path)
        && D_s.m() == info.number_of_slack_variables,
        "The implicit batch regularization dimensions are inconsistent.");
    if (f.n() == 0)
        return LinsolReturnFlag::SUCCESS;

    if (identity_factorization_ready_)
    {
        if (!D_eq)
            return identity_dynamics_solver_->solve_rhs_batch(
                info,
                static_cast<const Jacobian<OcpType> &>(jacobian),
                static_cast<const Hessian<OcpType> &>(hessian),
                D_s, f, g, x, eq_mult);

        for (Index column = 0; column < f.n(); ++column)
        {
            for (Index row = 0;
                 row < info.number_of_primal_variables; ++row)
                f_copy[0](row) = f(row, column);
            for (Index row = 0;
                 row < info.number_of_eq_constraints; ++row)
                g_copy[0](row) = g(row, column);
            batch_primal[0] = 0.0;
            batch_multipliers[0] = 0.0;
            const LinsolReturnFlag status =
                identity_dynamics_solver_->solve_rhs(
                    info,
                    static_cast<const Jacobian<OcpType> &>(jacobian),
                    static_cast<const Hessian<OcpType> &>(hessian),
                    *D_eq, D_s, f_copy[0], g_copy[0],
                    batch_primal[0], batch_multipliers[0]);
            if (status != LinsolReturnFlag::SUCCESS)
                return status;
            for (Index row = 0;
                 row < info.number_of_primal_variables; ++row)
                x(row, column) = batch_primal[0](row);
            for (Index row = 0;
                 row < info.number_of_eq_constraints; ++row)
                eq_mult(row, column) = batch_multipliers[0](row);
        }
        return LinsolReturnFlag::SUCCESS;
    }

    // The base factorization records the ranks and permutations.  The batch
    // fast path deliberately excludes rank-deficient transitions: their
    // moved-state/moved-equation RHS map is more general and remains on the
    // independently verified scalar path.
    for (Index k = 0; k + 1 < info.dims.K; ++k)
    {
        if (jacobian.J_ranks[k]
            != info.dims.number_of_states[k + 1])
            return LinsolReturnFlag::NOFULL_RANK;
    }

    for (Index row = 0; row < info.number_of_primal_variables; ++row)
        f_copy[0](row) = f(row, 0);
    for (Index row = 0; row < info.number_of_eq_constraints; ++row)
        g_copy[0](row) = g(row, 0);
    if (D_eq)
        veccp(
            info.number_of_g_eq_path, *D_eq, 0,
            D_eq_copy[0], 0);
    veccp(info.number_of_slack_variables, D_s, 0, D_s_copy[0], 0);

    const auto reset = [&]()
    {
        jacobian.ResetPreProcess(info);
        hessian.ResetPreProcess(info, jacobian);
    };

    try
    {
        // Recreate the preprocessed coefficient view once.  The Riccati
        // factors themselves are retained from the preceding base solve.
        // Keep preprocessing inside the cleanup scope because it modifies
        // the Jacobian and Hessian incrementally and may throw midway.
        const ProblemInfo modified_info = PreProcess(
            info, jacobian, hessian,
            f_copy[0], g_copy[0],
            nullptr, D_eq ? &D_eq_copy[0] : nullptr,
            &D_s_copy[0]);

        if (normalized_explicit_factorization_ready_)
        {
            fatrop_assert_msg(
                D_eq == nullptr && !normalized_explicit_stabilized_,
                "Only an unstabilized normalized-explicit factorization "
                "can serve the implicit batch fast path.");
            const ProblemInfo normalized_info =
                MakeNormalizedExplicitInfo(modified_info, false);
            VecRealView packed_g(
                g_original_copy[0],
                normalized_info.number_of_eq_constraints, 0);

            // A blocked traversal allocates and initializes matrix-valued
            // stage workspaces.  For one to three columns that setup costs
            // more than reusing the scalar explicit factors, while the
            // implicit coefficient preprocessing can still be shared by all
            // columns.  Switch to GEMM-oriented traversal once the border is
            // wide enough to amortize its workspace.
            constexpr Index blocked_rhs_threshold = 4;
            if (f.n() < blocked_rhs_threshold)
            {
                VecRealView normalized_multipliers(
                    g_copy[0],
                    normalized_info.number_of_eq_constraints, 0);
                for (Index column = 0; column < f.n(); ++column)
                {
                    PrepareFullRankRhs(
                        info, jacobian, nullptr, D_s,
                        f, g, column);
                    PackNormalizedExplicitConstraintRhs(
                        modified_info, normalized_info, false,
                        g_copy[0], packed_g);
                    batch_primal[0] = 0.;
                    normalized_multipliers = 0.;
                    const LinsolReturnFlag status =
                        identity_dynamics_solver_->solve_rhs(
                            normalized_info,
                            static_cast<const Jacobian<OcpType> &>(jacobian),
                            static_cast<const Hessian<OcpType> &>(hessian),
                            D_s_copy[0], f_copy[0], packed_g,
                            batch_primal[0], normalized_multipliers);
                    if (status != LinsolReturnFlag::SUCCESS)
                    {
                        reset();
                        return status;
                    }
                    UnpackNormalizedExplicitMultipliers(
                        modified_info, normalized_info, false,
                        normalized_multipliers, batch_multipliers[0]);
                    for (Index row = 0;
                         row < info.number_of_eq_constraints; ++row)
                        g_original_copy[0](row) = g(row, column);
                    PostProcess(
                        info, modified_info, jacobian, hessian,
                        batch_primal[0], batch_multipliers[0],
                        &D_s, nullptr, g_original_copy[0], false);
                    for (Index row = 0;
                         row < info.number_of_primal_variables; ++row)
                        x(row, column) = batch_primal[0](row);
                    for (Index row = 0;
                         row < info.number_of_eq_constraints; ++row)
                        eq_mult(row, column) = batch_multipliers[0](row);
                }
                reset();
                return LinsolReturnFlag::SUCCESS;
            }

            MatRealAllocated normalized_f(
                info.number_of_primal_variables, f.n());
            MatRealAllocated normalized_g(
                normalized_info.number_of_eq_constraints, f.n());
            MatRealAllocated normalized_x(
                info.number_of_primal_variables, f.n());
            MatRealAllocated normalized_multipliers(
                normalized_info.number_of_eq_constraints, f.n());

            // Apply the cached implicit-coordinate transformation once per
            // column, then solve all parameter-response columns in a single
            // blocked explicit Riccati traversal.
            for (Index column = 0; column < f.n(); ++column)
            {
                PrepareFullRankRhs(
                    info, jacobian, nullptr, D_s,
                    f, g, column);
                PackNormalizedExplicitConstraintRhs(
                    modified_info, normalized_info, false,
                    g_copy[0], packed_g);
                for (Index row = 0;
                     row < info.number_of_primal_variables; ++row)
                    normalized_f(row, column) = f_copy[0](row);
                for (Index row = 0;
                     row < normalized_info.number_of_eq_constraints; ++row)
                    normalized_g(row, column) = packed_g(row);
            }

            const LinsolReturnFlag status =
                identity_dynamics_solver_->solve_rhs_batch(
                    normalized_info,
                    static_cast<const Jacobian<OcpType> &>(jacobian),
                    static_cast<const Hessian<OcpType> &>(hessian),
                    D_s_copy[0], normalized_f, normalized_g,
                    normalized_x, normalized_multipliers);
            if (status != LinsolReturnFlag::SUCCESS)
            {
                reset();
                return status;
            }

            VecRealView packed_multipliers(
                g_copy[0],
                normalized_info.number_of_eq_constraints, 0);
            for (Index column = 0; column < f.n(); ++column)
            {
                for (Index row = 0;
                     row < info.number_of_primal_variables; ++row)
                    batch_primal[0](row) = normalized_x(row, column);
                for (Index row = 0;
                     row < normalized_info.number_of_eq_constraints; ++row)
                    packed_multipliers(row) =
                        normalized_multipliers(row, column);
                UnpackNormalizedExplicitMultipliers(
                    modified_info, normalized_info, false,
                    packed_multipliers, batch_multipliers[0]);
                for (Index row = 0;
                     row < info.number_of_eq_constraints; ++row)
                    g_original_copy[0](row) = g(row, column);
                PostProcess(
                    info, modified_info, jacobian, hessian,
                    batch_primal[0], batch_multipliers[0],
                    &D_s, nullptr, g_original_copy[0], false);
                for (Index row = 0;
                     row < info.number_of_primal_variables; ++row)
                    x(row, column) = batch_primal[0](row);
                for (Index row = 0;
                     row < info.number_of_eq_constraints; ++row)
                    eq_mult(row, column) = batch_multipliers[0](row);
            }
            reset();
            return LinsolReturnFlag::SUCCESS;
        }

        for (Index column = 0; column < f.n(); ++column)
        {
            PrepareFullRankRhs(
                info, jacobian, D_eq, D_s,
                f, g, column);
            batch_primal[0] = 0.0;
            batch_multipliers[0] = 0.0;
            const LinsolReturnFlag status = D_eq
                ? ModifiedAugSystemSolver::solve_rhs(
                    modified_info, jacobian, hessian,
                    D_eq_copy[0], D_s_copy[0],
                    f_copy[0], g_copy[0],
                    batch_primal[0], batch_multipliers[0])
                : ModifiedAugSystemSolver::solve_rhs(
                    modified_info, jacobian, hessian,
                    D_s_copy[0], f_copy[0], g_copy[0],
                    batch_primal[0], batch_multipliers[0]);
            if (status != LinsolReturnFlag::SUCCESS)
            {
                reset();
                return status;
            }
            PostProcess(
                info, modified_info, jacobian, hessian,
                batch_primal[0], batch_multipliers[0],
                &D_s, D_eq, g_original_copy[0], false);
            for (Index row = 0;
                 row < info.number_of_primal_variables; ++row)
                x(row, column) = batch_primal[0](row);
            for (Index row = 0;
                 row < info.number_of_eq_constraints; ++row)
                eq_mult(row, column) = batch_multipliers[0](row);
        }
    }
    catch (...)
    {
        reset();
        throw;
    }
    reset();
    return LinsolReturnFlag::SUCCESS;
}

void AugSystemSolver<ImplicitOcpType>::set_performance_mode(bool set){
    if (set){
        print_debug = false;
        print_preprocessed_system = false;
        write_preprocessing_file = false;
        verify_preprocessed_solution = false;
        print_final_solution = false;
        ModifiedAugSystemSolver::set_performance_mode(true);
    }
}

ProblemInfo AugSystemSolver<ImplicitOcpType>::PreProcess(const ProblemInfo &info,
                                                  Jacobian<ImplicitOcpType> &jacobian,
                                                  Hessian<ImplicitOcpType> &hessian,
                                                  VecRealView &f,
                                                  VecRealView &g,
                                                  VecRealView* D_x,
                                                  VecRealView* D_eq,
                                                  VecRealView* D_s)
{
    if (print_debug){ std::cout << "AugSystemSolver<ImplicitOcpType>::PreProcess start" << std::endl;}

    // PreProcess changes the stage partition in place by moving incoming
    // null-space state components to controls and the corresponding
    // dynamics equations to path equalities. Every base or right-hand-side
    // solve must start from the original partition.
    number_of_states = info.dims.number_of_states;
    number_of_controls = info.dims.number_of_controls;
    number_of_eq_constraints =
        info.dims.number_of_eq_constraints;
    number_of_ineq_constraints =
        info.dims.number_of_ineq_constraints;

    // GENERAL VERSION
    #ifdef PROFILE
    auto start = std::chrono::high_resolution_clock::now();
    #endif
    #ifndef IGNORE_JAC_HESS_PREPROCESS
    jacobian.PreProcess(info, f, g);
    #endif
    if (print_debug){ std::cout << "AugSystemSolver<ImplicitOcpType>::jacobian::PreProcess done" << std::endl;}
    #ifdef PROFILE
    auto stop = std::chrono::high_resolution_clock::now();
    duration_preprocess_jac = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

    start = std::chrono::high_resolution_clock::now();
    #endif
    #ifndef IGNORE_JAC_HESS_PREPROCESS
    hessian.PreProcess(info, jacobian, f, g);
    #endif
    if (print_debug){ std::cout << "AugSystemSolver<ImplicitOcpType>::hessian::PreProcess done" << std::endl;}
    #ifdef PROFILE
    stop = std::chrono::high_resolution_clock::now();
    duration_preprocess_hess = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

    start = std::chrono::high_resolution_clock::now();
    #endif
    int K = info.dims.K;
    bool USE_NEW_REGULARIZATION_TREATMENT = true;
    if (USE_NEW_REGULARIZATION_TREATMENT){
    // Deal with regularization terms
    for (int k = 0; k < K; ++k){
        const Index nx = number_of_states[k];
        const Index nu = number_of_controls[k];
        const Index ng = number_of_eq_constraints[k];
        const Index ng_ineq = number_of_ineq_constraints[k];
        const Index offset_eq_k = info.offsets_g_eq_path[k];
        const Index offs_ineq_k = info.offsets_slack[k];
        const Index offset_g_ineq_k = info.offsets_g_eq_slack[k];
        const Index offset_u = info.offsets_primal_u[k];

        trtr_l(nu + nx, hessian.RSQrqt[k], 0, 0, hessian.RSQrqt[k], 0, 0); // copy lower part of RSQ to upper part
        if (D_eq != nullptr && ng > 0)
        // equality penalty
        {
            for (Index i = 0; i < ng; i++)
            {
                Scalar scaling_factor = 1.0 / (*D_eq)(offset_eq_k + i);
                colsc(nu + nx + 1, scaling_factor, jacobian.Gg_eqt[k], 0, i);
            }
            // add the penalty to hessian
            syrk_ln_mn(nu + nx + 1, nu + nx, ng, 1.0, jacobian.Gg_eqt[k], 0, 0, jacobian.Gg_eqt_original[k], 0, 0,
                       1.0, hessian.RSQrqt[k], 0, 0, hessian.RSQrqt[k], 0, 0);
            // add the penalty to rhs
            gemv_n(nu + nx, ng, 1.0, jacobian.Gg_eqt[k], 0, 0, g, offset_eq_k, 1.0, f, offset_u, f, offset_u);

            // vecse(ng, 1.0, *D_eq, offset_eq_k);
        }
        if (D_s != nullptr && ng_ineq > 0)
        // inequalities + inertia correction
        {
            if (ng_ineq > 0)
            {
                for (Index i = 0; i < ng_ineq; i++)
                {
                    Scalar scaling_factor = 1.0 / (*D_s)(offs_ineq_k + i);
                    colsc(nu + nx + 1, scaling_factor, jacobian.Gg_ineqt[k], 0, i);
                }
                // add the penalty to hessian
                syrk_ln_mn(nu + nx + 1, nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0,
                           jacobian.Gg_ineqt_original[k], 0, 0, 1.0, hessian.RSQrqt[k], 0, 0, hessian.RSQrqt[k],
                           0, 0);
                // add the penalty to rhs
                // TODO: is this needed? 
                gemv_n(nu + nx, ng_ineq, 1.0, jacobian.Gg_ineqt[k], 0, 0, g, offset_g_ineq_k, 1.0, f, offset_u, f, offset_u);

                // vecse(ng_ineq, 1.0, *D_s, offs_ineq_k);
            }
        }
        if (D_x != nullptr)
        {
        // inertia correction
        diaad(nu + nx, 1.0, *D_x, offset_u, hessian.RSQrqt[k], 0, 0);
        // vecse(nu + nx, 0.0, *D_x, offset_u);
        }
    }
    }
    #ifdef PROFILE
    stop = std::chrono::high_resolution_clock::now();
    duration_preprocess_regularization = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    #endif

    // Pre-process 
    #ifdef PROFILE
    start = std::chrono::high_resolution_clock::now();
    #endif
    for (int k = 0; k < K-1; ++k){
        int nx = number_of_states[k];
        int nx_next = number_of_states[k + 1];
        int nu = number_of_controls[k];
        int nu_next = number_of_controls[k + 1];

        // construct JABbt-matrix
        int rank;
        #ifdef PROFILE
        auto inner_start = std::chrono::high_resolution_clock::now();
        #endif
        gecp(nx_next, nx_next, jacobian.Jt[k], 0, 0, JBAbt[0], 0, 0);
        gecp(nu + nx + 1, nx_next, jacobian.BAbt[k], 0, 0, JBAbt[0], nx_next, 0);
        gecp(nx_next + nu + nx + 1, nx_next, JBAbt[0], 0, 0, JBAbt_modified[0], 0, 0);
        #ifdef PROFILE
        auto inner_stop = std::chrono::high_resolution_clock::now();
        duration_decomp_copies += std::chrono::duration_cast<std::chrono::nanoseconds>(inner_stop - inner_start);
        #endif

        // decompose J matrix
        auto inner_start = std::chrono::high_resolution_clock::now();
        lu_fact_transposed(nx_next, nx_next + nu + nx + 1, nx_next, rank, JBAbt[0], jacobian.Pl_pre[k], jacobian.Pr_pre[k], lu_fact_tol);
        gecp(rank, rank, JBAbt[0], 0, 0, jacobian.U1t[k], 0, 0);
        // PrintLuInfo(JBAbt, jacobian.Jt[k], jacobian.Pl_pre[k], jacobian.Pr_pre[k]);
        // rank = nx_next;
        auto inner_stop = std::chrono::high_resolution_clock::now();
        duration_decomp_decomp += std::chrono::duration_cast<std::chrono::nanoseconds>(inner_stop - inner_start);
        inner_start = std::chrono::high_resolution_clock::now();
        if (nu + nx < info.dims.number_of_eq_constraints[k] + nx_next - rank){
            // there will be more constraints at this stage than can be
            // satisfied using the constrols and the states.
            // likely, the problem is ill-defined
            throw std::runtime_error("The problem seems to be ill-defined "
                "since the number of constraints at stage " + std::to_string(k) + 
                " (" + std::to_string(info.dims.number_of_eq_constraints[k]) + " + " + std::to_string(nx_next-rank) + ") "
                "exceeds the number of controls and states (" + std::to_string(nu) + " + " + std::to_string(nx) + ") "
                "that can be used to satisfy them.");
        }
        // extend permutation matrix to include nu_next part
        PermutationMatrix Pr_extended = PermutationMatrix(nu_next + rank);
        for (int i = 0; i < rank; i++){ Pr_extended[nu_next + i] = nu_next + jacobian.Pr_pre[k][i];}

        // Modify dynamics jacobian
        jacobian.Pl_pre[k].apply_on_cols(rank, &(JBAbt_modified[0]).mat());                                              // * P_l
        trsm_runu(nx_next + nu + nx + 1, nx_next, 1.0, JBAbt[0], 0, 0, JBAbt_modified[0], 0, 0, JBAbt_modified[0], 0, 0);    // * L^-1
        trsm_rlnn(nx_next + nu + nx + 1, rank, -1.0, JBAbt[0], 0, 0, JBAbt_modified[0], 0, 0, JBAbt_modified[0], 0, 0);      // * U^-1
        gecp(nu + nx + 1, nx_next, JBAbt_modified[0], nx_next, 0, jacobian.BAbt[k], 0, 0);
        jacobian.Pr_pre[k].apply_on_rows(rank, &(JBAbt_modified[0]).mat());                                              // * P_r
        gecp(nx_next-rank, rank, JBAbt_modified[0], rank, 0, jacobian.U1U2t[k], 0, 0);

        // other hessian contribution
        trtr_l(nu_next + nx_next, hessian.RSQrqt[k+1], 0, 0, hessian.RSQrqt[k+1], 0, 0);
        if (!USE_NEW_REGULARIZATION_TREATMENT){
        if (D_x != nullptr){
            // consider regularization already here
            diaad(nu_next + nx_next, 1.0, *D_x, info.offsets_primal_u[k+1], hessian.RSQrqt[k+1], 0, 0);
            // make sure to not consider them later on again
            vecsc(nu_next + nx_next, 0.0, *D_x, info.offsets_primal_u[k+1]);

            // don't skip k == 0
            if (k == 0){
                diaad(nu + nx, 1.0, *D_x, info.offsets_primal_u[k], hessian.RSQrqt[k], 0, 0);
                vecsc(nu + nx, 0.0, *D_x, info.offsets_primal_u[k]);
            }
        }
        }
        #ifdef PROFILE
        inner_stop = std::chrono::high_resolution_clock::now();
        duration_decomp_scale1 += std::chrono::duration_cast<std::chrono::nanoseconds>(inner_stop - inner_start);
        inner_start = std::chrono::high_resolution_clock::now();
        #endif

        // right-multiply right part with Dr^-1
        Pr_extended.apply_on_rows(nu_next + rank, &hessian.RSQrqt[k+1].mat());
        gemm_nn(nx_next - rank, nu_next + nx_next, rank, 1.0, jacobian.U1U2t[k], 0, 0, hessian.RSQrqt[k+1], nu_next, 0, 1.0, 
                hessian.RSQrqt[k+1], nu_next + rank, 0, hessian.RSQrqt[k+1], nu_next + rank, 0);
        // left-multiply bottom part with Dr^-T
        Pr_extended.apply_on_cols(nu_next + rank, &hessian.RSQrqt[k+1].mat());
        gemm_nt(nu_next + nx_next + 1, nx_next - rank, rank, 1.0, hessian.RSQrqt[k+1], 0, nu_next, jacobian.U1U2t[k], 0, 0, 1.0, 
                hessian.RSQrqt[k+1], 0, nu_next + rank, hessian.RSQrqt[k+1], 0, nu_next + rank);


        // hessian contribution of dynamics
        jacobian.Pr_pre[k].apply_on_cols(rank, &hessian.FuFx[k].mat());
        gemm_nt(nu + nx, nx_next - rank, rank, 1.0, hessian.FuFx[k], 0, 0, jacobian.U1U2t[k], 0, 0, 1.0, hessian.FuFx[k], 0, rank, hessian.FuFx[k], 0, rank);
            
        // right multiply with Dr^-1
        if (k < K - 2){
            int nx_next_next = number_of_states[k + 2];
            // dynamics jacobian
            Pr_extended.apply_on_rows(nu_next + rank, &jacobian.BAbt[k+1].mat());
            gemm_nn(nx_next - rank, nx_next_next, rank, 1.0, jacobian.U1U2t[k], 0, 0, jacobian.BAbt[k+1], nu_next, 0, 1.0, 
                    jacobian.BAbt[k+1], nu_next + rank, 0, jacobian.BAbt[k+1], nu_next + rank, 0);
            // dynamics hessian
            Pr_extended.apply_on_rows(nu_next + rank, &hessian.FuFx[k+1].mat());
            gemm_nn(nx_next - rank, nx_next_next, rank, 1.0, jacobian.U1U2t[k], 0, 0, hessian.FuFx[k+1], nu_next, 0, 1.0, hessian.FuFx[k+1], nu_next + rank, 0, hessian.FuFx[k+1], nu_next + rank, 0);
        }

        // equality constraints
        Pr_extended.apply_on_rows(nu_next + rank, &jacobian.Gg_eqt[k+1].mat());
        gemm_nn(nx_next - rank, info.dims.number_of_eq_constraints[k+1], rank, 1.0, jacobian.U1U2t[k], 0, 0, jacobian.Gg_eqt[k+1], nu_next, 0, 1.0, jacobian.Gg_eqt[k+1], nu_next + rank, 0, jacobian.Gg_eqt[k+1], nu_next + rank, 0);
        // inequality constraints
        // Pr_extended.apply_on_rows(nu_next + rank, &jacobian.Gg_ineqt[k+1].mat());
        // gemm_nn(nx_next - rank, info.dims.number_of_ineq_constraints[k+1], rank, 1.0, jacobian.U1U2t[k], 0, 0, jacobian.Gg_ineqt[k+1], nu_next, 0, 1.0, jacobian.Gg_ineqt[k+1], nu_next + rank, 0, jacobian.Gg_ineqt[k+1], nu_next + rank, 0);
        #ifdef PROFILE
        inner_stop = std::chrono::high_resolution_clock::now();
        duration_decomp_scale2 += std::chrono::duration_cast<std::chrono::nanoseconds>(inner_stop - inner_start);

        inner_start = std::chrono::high_resolution_clock::now();
        #endif
        // Move undefined states to controls
        if (rank < nx_next){
            gecp(nu + nx, nx_next - rank, hessian.FuFx[k], 0, rank, hessian.GuGx[k], 0, nu_next);
            TreatStatesAsInputs(nu_next, nx_next, rank, hessian.RSQrqt[k+1], true);
            TreatStatesAsInputs(nu_next, nx_next, rank, hessian.RSQrqt[k+1]);
            TreatStatesAsInputs(nu_next, nx_next, rank, jacobian.Gg_eqt[k+1], true);
            // TreatStatesAsInputs(nu_next, nx_next, rank, jacobian.Gg_ineqt[k+1], true);
            if (k < K - 2){
                TreatStatesAsInputs(nu_next, nx_next, rank, jacobian.BAbt[k+1], true);
                TreatStatesAsInputs(nu_next, nx_next, rank, hessian.FuFx[k+1], true);
            }

            // treat some of the dynamics constraints as path constraints
            gecp(nu + nx + 1, nx_next - rank, jacobian.BAbt[k], 0, rank, jacobian.Gg_eqt[k], 0, info.dims.number_of_eq_constraints[k]);

            // update dimensions
            int sk = nx_next - rank;
            number_of_controls[k + 1] += sk;
            number_of_eq_constraints[k] += sk;
            number_of_states[k + 1] = rank;
        }
        #ifdef PROFILE
        inner_stop = std::chrono::high_resolution_clock::now();
        duration_decomp_permutation += std::chrono::duration_cast<std::chrono::nanoseconds>(inner_stop - inner_start);
        #endif

        // store info
        #ifdef PROFILE
        inner_start = std::chrono::high_resolution_clock::now();
        #endif
        jacobian.J_ranks[k] = rank;
        jacobian.nb_new_controls[k] = nx_next - rank;
        // jacobian.Jt_LU[k] = JBAbt[0];
        gecp(nx_next, nx_next, JBAbt[0], 0, 0, jacobian.Jt_LU[k], 0, 0);
        #ifdef PROFILE
        inner_stop = std::chrono::high_resolution_clock::now();
        duration_decomp_store += std::chrono::duration_cast<std::chrono::nanoseconds>(inner_stop - inner_start);
        #endif
    }
    #ifdef PROFILE
    stop = std::chrono::high_resolution_clock::now();
    duration_preprocess_decomposition += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);

    start = std::chrono::high_resolution_clock::now();
    #endif
    ProblemInfo modified_info(ProblemDims(
        K, number_of_controls, number_of_states,
        number_of_eq_constraints, number_of_ineq_constraints,
        0,
        info.dims.number_of_stage_border_variables));
    #ifdef PROFILE
    stop = std::chrono::high_resolution_clock::now();
    duration_preprocess_info += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    #endif

    // modify right-hand side
    #ifdef PROFILE
    start = std::chrono::high_resolution_clock::now();
    #endif
    for (int k = 0; k < K; ++k){
        int nx = modified_info.dims.number_of_states[k];
        int nu = modified_info.dims.number_of_controls[k];

        // modify f
        // for (int i = 0; i < nu + nx; i++){
        //     f(modified_info.offsets_primal_u[k] + i) = hessian.RSQrqt[k](nu + nx, i);
        // }
        rowex(nu + nx, 1.0, hessian.RSQrqt[k], nu + nx, 0, f, modified_info.offsets_primal_u[k]);

        // modify g
        // for (int i = 0; i < modified_info.dims.number_of_eq_constraints[k]; i++){
        //     g(modified_info.offsets_g_eq_path[k] + i) = jacobian.Gg_eqt[k](nu + nx, i);
        // }
        rowex(modified_info.dims.number_of_eq_constraints[k], 1.0, jacobian.Gg_eqt[k], nu + nx, 0, g, modified_info.offsets_g_eq_path[k]);

        if (k < K - 1){
            int nx_next = modified_info.dims.number_of_states[k + 1];
            // for (int i = 0; i < nx_next; i++){
            //     g(modified_info.offsets_g_eq_dyn[k] + i) = jacobian.BAbt[k](nu + nx, i);
            // }
            rowex(nx_next, 1.0, jacobian.BAbt[k], nu + nx, 0, g, modified_info.offsets_g_eq_dyn[k]); 
        }
    }
    #ifdef PROFILE
    stop = std::chrono::high_resolution_clock::now();
    duration_preprocess_modify_rhs = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    #endif

    if (print_debug){ std::cout << "AugSystemSolver<ImplicitOcpType>::PreProcess done" << std::endl;}   
    return modified_info;
}

void AugSystemSolver<ImplicitOcpType>::PostProcess(const ProblemInfo &info,
                                                   const ProblemInfo &modified_info,
                                                   Jacobian<ImplicitOcpType> &jacobian,
                                                   Hessian<ImplicitOcpType> &hessian,
                                                   VecRealView &x, VecRealView &eq_mult,
                                                   const VecRealView* D_s, 
                                                   const VecRealView* D_eq,
                                                   const VecRealView &g,
                                                   const bool reset_matrices){
    // GENERAL VERSION
    if (print_debug){ std::cout << "AugSystemSolver<ImplicitOcpType>::PostProcess start" << std::endl;}
    // VecRealAllocated x_copy(x.m());
    // for (int i = 0; i < x.m(); i++){
    //     x_copy(i) = x(i);
    // }
    // VecRealAllocated eq_mult_copy(eq_mult.m());
    // for (int i = 0; i < eq_mult.m(); i++){
    //     eq_mult_copy(i) = eq_mult(i);
    // }
    #ifdef PROFILE
    auto start = std::chrono::high_resolution_clock::now();
    #endif
    veccp(x.m(), x, 0, x_copy[0], 0);
    veccp(eq_mult.m(), eq_mult, 0, eq_mult_copy[0], 0);
    #ifdef PROFILE
    auto stop = std::chrono::high_resolution_clock::now();
    duration_post_rearrange_solution += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    #endif

    for (int k = 0; k < info.dims.K; ++k){
        #ifdef PROFILE
        start = std::chrono::high_resolution_clock::now();
        #endif
        Index nu = info.dims.number_of_controls[k];
        Index nu_mod = modified_info.dims.number_of_controls[k];
        Index s = (k < info.dims.K - 1) ? info.dims.number_of_states[k + 1] - jacobian.J_ranks[k] : 0;
        Index s_states = (k > 0) ? info.dims.number_of_states[k] - jacobian.J_ranks[k-1] : 0;

        // controls (plain copy)
        // for (int i = 0; i < nu; i++){
        //     x(info.offsets_primal_u[k] + i) = (x_copy[0])(modified_info.offsets_primal_u[k] + i);
        // }
        veccp(nu, x_copy[0], modified_info.offsets_primal_u[k], x, info.offsets_primal_u[k]);

        // states (copy existing states, and append additional states treated as controls)
        if (k > 0){
            // for (int i = 0; i < jacobian.J_ranks[k-1]; i++){
            //     x(info.offsets_primal_x[k] + i) = (x_copy[0])(modified_info.offsets_primal_x[k] + i);
            // }
            veccp(jacobian.J_ranks[k-1], x_copy[0], modified_info.offsets_primal_x[k], x, info.offsets_primal_x[k]);
            // for (int i = 0; i < s_states; i++){
            //     x(info.offsets_primal_x[k] + jacobian.J_ranks[k-1] + i) = (x_copy[0])(modified_info.offsets_primal_u[k] + nu + i);
            // }
            veccp(s_states, x_copy[0], modified_info.offsets_primal_u[k] + nu, x, info.offsets_primal_x[k] + jacobian.J_ranks[k-1]);
        } else {
            // for (int i = 0; i < info.dims.number_of_states[k]; i++){
            //     x(info.offsets_primal_x[k] + i) = (x_copy[0])(modified_info.offsets_primal_x[k] + i);
            // }
            veccp(info.dims.number_of_states[k], x_copy[0], modified_info.offsets_primal_x[k], x, info.offsets_primal_x[k]);
        }
        if (k < info.dims.K - 1){
            // dynamics (copy existing dynamics, and append additional path constraints)
            // for (int i = 0; i < jacobian.J_ranks[k]; i++){
            //     eq_mult(info.offsets_g_eq_dyn[k] + i) = (eq_mult_copy[0])(modified_info.offsets_g_eq_dyn[k] + i);
            // }
            veccp(jacobian.J_ranks[k], eq_mult_copy[0], modified_info.offsets_g_eq_dyn[k], eq_mult, info.offsets_g_eq_dyn[k]);
            // for (int i = 0; i < s; i++){
            //     eq_mult(info.offsets_g_eq_dyn[k] + jacobian.J_ranks[k] + i) = 
            //         (eq_mult_copy[0])(modified_info.offsets_g_eq_path[k] + info.dims.number_of_eq_constraints[k] + i);
            // }
            veccp(s, eq_mult_copy[0], modified_info.offsets_g_eq_path[k] + info.dims.number_of_eq_constraints[k], eq_mult, info.offsets_g_eq_dyn[k] + jacobian.J_ranks[k]);
        }

        // equality path constraints (plain copy)
        // for (int i = 0; i < info.dims.number_of_eq_constraints[k]; i++){
        //     eq_mult(info.offsets_g_eq_path[k] + i) = (eq_mult_copy[0])(modified_info.offsets_g_eq_path[k] + i);
        // }
        veccp(info.dims.number_of_eq_constraints[k], eq_mult_copy[0], modified_info.offsets_g_eq_path[k], eq_mult, info.offsets_g_eq_path[k]);

        // inequality path constraints (plain copy)
        // for (int i = 0; i < info.dims.number_of_ineq_constraints[k]; i++){
        //     eq_mult(info.offsets_g_eq_slack[k] + i) = (eq_mult_copy[0])(modified_info.offsets_g_eq_slack[k] + i);
        // }
        veccp(info.dims.number_of_ineq_constraints[k], eq_mult_copy[0], modified_info.offsets_g_eq_slack[k], eq_mult, info.offsets_g_eq_slack[k]);

        #ifdef PROFILE
        stop = std::chrono::high_resolution_clock::now();
        duration_post_rearrange_solution += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
        start = std::chrono::high_resolution_clock::now();
        #endif
        
        // scale states and dynamics multipliers
        if (k > 0){
            // if (k < info.dims.K){
            gemv_t(info.dims.number_of_states[k] - jacobian.J_ranks[k-1], info.dims.number_of_states[k], 1.0, 
                jacobian.U1U2t[k-1], 0, 0, 
                x, info.offsets_primal_x[k] + jacobian.J_ranks[k-1], 1.0, 
                x, info.offsets_primal_x[k], 
                x, info.offsets_primal_x[k]);
            jacobian.Pr_pre[k-1].apply_inverse(jacobian.J_ranks[k-1], &x.vec(), info.offsets_primal_x[k]);
            // } 

            // U^-T * 
            // PrintNpArray(eq_mult, info.offsets_g_eq_dyn[k-1], info.dims.number_of_states[k], "\n[" + std::to_string(k) + "] eq_mult before");
            trsv_lnn(jacobian.J_ranks[k-1], jacobian.U1t[k-1], 0, 0, eq_mult, info.offsets_g_eq_dyn[k-1], eq_mult, info.offsets_g_eq_dyn[k-1]);
            vecsc(jacobian.J_ranks[k-1], -1.0, eq_mult, info.offsets_g_eq_dyn[k-1]);
            // L^-T *
            trsv_unu(info.dims.number_of_states[k], info.dims.number_of_states[k], jacobian.Jt_LU[k-1], 0, 0, eq_mult, info.offsets_g_eq_dyn[k-1], eq_mult, info.offsets_g_eq_dyn[k-1]);
            // Pl * 
            jacobian.Pl_pre[k-1].apply_inverse(jacobian.J_ranks[k-1], &eq_mult.vec(), info.offsets_g_eq_dyn[k-1]);
            // std::cout << "eq_mult after:\n" << eq_mult << std::endl;
            // PrintNpArray(eq_mult, info.offsets_g_eq_dyn[k-1], info.dims.number_of_states[k], "[" + std::to_string(k) + "] eq_mult after");
        }
        #ifdef PROFILE
        stop = std::chrono::high_resolution_clock::now();
        duration_post_scale_solution += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
        #endif
    }

    #ifdef PROFILE
    start = std::chrono::high_resolution_clock::now();
    #endif
    #ifndef IGNORE_JAC_HESS_PREPROCESS
    if (reset_matrices)
        jacobian.ResetPreProcess(info);
    #endif
    #ifdef PROFILE
    stop = std::chrono::high_resolution_clock::now();
    duration_post_reset_jacobian_pre += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    start = std::chrono::high_resolution_clock::now();
    #endif
    #ifndef IGNORE_JAC_HESS_PREPROCESS
    if (reset_matrices)
        hessian.ResetPreProcess(info, jacobian);
    #endif
    #ifdef PROFILE
    stop = std::chrono::high_resolution_clock::now();
    duration_post_reset_hessian_pre += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    start = std::chrono::high_resolution_clock::now();
    #endif

    // Consider constraint regularizations
    if (D_s != nullptr){
        for (int k = 0; k < info.dims.K; ++k){
            const Index nu = info.dims.number_of_controls[k];
            const Index nx = info.dims.number_of_states[k];
            const Index ng_ineq = info.dims.number_of_ineq_constraints[k];
            const Index offs = info.offsets_primal_u[k];
            const Index offs_eq_ineq = info.offsets_g_eq_slack[k];
            const Index offs_slack = info.offsets_slack[k];
            // PrintNpArray(jacobian.Gg_ineqt[k], "\nGg_ineqt[" + std::to_string(k) + "]", nu + nx + 1, ng_ineq);
            // PrintNpArray(*D_s, offs_slack, ng_ineq, "D_s[" + std::to_string(k) + "]");
            // PrintNpArray(g, offs_eq_ineq, ng_ineq, "g[" + std::to_string(k) + "]");
            // PrintNpArray(x, offs, nu + nx, "x[" + std::to_string(k) + "]");
            // PrintNpArray(eq_mult, offs_eq_ineq, ng_ineq, "[" + std::to_string(k) + "] eq_mult before ineq regularization");
            // std::cout << "nu: " << nu << " nx: " << nx << " ng_ineq: " << ng_ineq << std::endl;
            // std::cout << "offs: " << offs << " offs_eq_ineq: " << offs_eq_ineq << " offs_slack: " << offs_slack << std::endl;
            const MatRealView &Gg_ineqt =
                reset_matrices
                ? static_cast<const MatRealView &>(jacobian.Gg_ineqt[k])
                : static_cast<const MatRealView &>(jacobian.Gg_ineqt_original[k]);
            gemv_t(nu + nx, ng_ineq, 1.0, Gg_ineqt, 0, 0, x, offs, 1.0, g, offs_eq_ineq,
                   eq_mult, offs_eq_ineq);
            eq_mult.block(ng_ineq, offs_eq_ineq) =
                eq_mult.block(ng_ineq, offs_eq_ineq) / (*D_s).block(ng_ineq, offs_slack);
            // PrintNpArray(eq_mult, offs_eq_ineq, ng_ineq, "[" + std::to_string(k) + "] eq_mult after ineq regularization");
        }
    }
    if (D_eq != nullptr){
        for (int k = 0; k < info.dims.K; ++k){
            const Index nu = info.dims.number_of_controls[k];
            const Index nx = info.dims.number_of_states[k];
            const Index offs = info.offsets_primal_u[k];
            const Index ng = info.dims.number_of_eq_constraints[k];
            const Index offs_g_eq_k = info.offsets_g_eq_path[k];
            const Index offs_eq_k = info.offsets_eq[k];            
            if (ng > 0)
            {
                const MatRealView &Gg_eqt =
                    reset_matrices
                    ? static_cast<const MatRealView &>(jacobian.Gg_eqt[k])
                    : static_cast<const MatRealView &>(jacobian.Gg_eqt_original[k]);
                gemv_t(nu + nx, ng, 1.0, Gg_eqt, 0, 0, x, offs, 1.0, g, offs_g_eq_k,
                    eq_mult, offs_g_eq_k);
                eq_mult.block(ng, offs_g_eq_k) =
                    eq_mult.block(ng, offs_g_eq_k) / (*D_eq).block(ng, offs_eq_k);
            }
        }
    }
    #ifdef PROFILE
    stop = std::chrono::high_resolution_clock::now();
    duration_post_regularization += std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    #endif
    if (print_debug){ std::cout << "AugSystemSolver<ImplicitOcpType>::PostProcess done" << std::endl;}
}

void AugSystemSolver<ImplicitOcpType>::TreatStatesAsInputs(Index nu_next, Index nx_next, Index rank, MatRealAllocated& A, bool rows){
    // std::cout << "Matrix before shifting states to inputs:" << std::endl;
    // std::cout << A << std::endl;
    if (!rows){
        // copy nu_next + nx_next columns to scratch
        gecp(A.m(), nu_next + nx_next, A, 0, 0, *scratch, 0, 0);

        // put last nx_next - rank columns in front
        gecp(A.m(), nx_next - rank, *scratch, 0, nu_next + rank, A, 0, nu_next);

        // shift back the first rank columns
        gecp(A.m(), rank, *scratch, 0, nu_next, A, 0, nu_next + nx_next - rank);

        // insert copied columns
        // gecp(A.m(), nx_next - rank, *scratch, 0, 0, A, 0, nu_next);
    } else {
        // copy nu_next + nx_next rows to scratch
        gecp(nu_next + nx_next, A.n(), A, 0, 0, *scratch, 0, 0);

        // put last nx_next - rank rows in front
        gecp(nx_next - rank, A.n(), *scratch, nu_next + rank, 0, A, nu_next, 0);

        // shift back the first rank rows
        gecp(rank, A.n(), *scratch, nu_next, 0, A, nu_next + nx_next - rank, 0);

        // insert copied rows
        // gecp(nx_next - rank, A.n(), *scratch, 0, 0, A, 0, 0);
    }
    // std::cout << "Matrix after shifting states to inputs:" << std::endl;
    // std::cout << A << std::endl;
}
