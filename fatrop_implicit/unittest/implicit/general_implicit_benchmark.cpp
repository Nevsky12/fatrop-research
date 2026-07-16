#include "../random_matrix.hpp"
#include "fatrop/context/context.hpp"
#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/ocp/aug_system_solver.hpp"
#include "fatrop/ocp/dims.hpp" // inherit
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/problem_info.hpp" //inherit
#include "fatrop/ocp/type.hpp"
#include <gtest/gtest.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <random>
#include <fstream>

static const bool USE_GENERALIZATION = true;

using namespace fatrop;

class RandomBenchmarkTest : public ::testing::Test
{
// protected:
public:
    bool full_rank = true;
    bool constant_dimensions = true;
    bool use_generalization = USE_GENERALIZATION;

    int K;
    std::vector<Index> nx;
    std::vector<Index> r;
    std::vector<Index> nu;
    std::vector<Index> ng;
    std::vector<Index> ng_ineq;
    
    std::vector<Index> nv;
    std::vector<Index> nz;

    /// Explicit case ///
    std::optional<ProblemDims> dims_expl;
    std::optional<ProblemInfo> info_expl;
    std::optional<Jacobian<OcpType>> jacobian_expl;
    std::optional<Hessian<OcpType>> hessian_expl;
    std::optional<MatRealAllocated> full_matrix_jacobian_expl;
    std::optional<MatRealAllocated> full_matrix_hessian_expl;
    std::optional<VecRealAllocated> x_expl;
    std::optional<VecRealAllocated> mult_expl;
    std::optional<VecRealAllocated> rhs_x_expl;
    std::optional<VecRealAllocated> rhs_g_expl;
    std::optional<VecRealAllocated> D_x_expl;
    std::optional<VecRealAllocated> D_s_expl;
    std::optional<VecRealAllocated> D_eq_expl;
    std::optional<MatRealAllocated> full_kkt_matrix_expl;
    std::optional<AugSystemSolver<OcpType>> solver_expl;

    /// Implicit case ///
    std::optional<ProblemDims> dims_impl;
    std::optional<ProblemInfo> info_impl;
    std::optional<Jacobian<ImplicitOcpType>> jacobian_impl;
    std::optional<Hessian<ImplicitOcpType>> hessian_impl;
    std::optional<MatRealAllocated> full_matrix_jacobian_impl;
    std::optional<MatRealAllocated> full_matrix_hessian_impl;
    std::optional<VecRealAllocated> x_impl;
    std::optional<VecRealAllocated> mult_impl;
    std::optional<VecRealAllocated> rhs_x_impl;
    std::optional<VecRealAllocated> rhs_g_impl;
    std::optional<VecRealAllocated> D_x_impl;
    std::optional<VecRealAllocated> D_s_impl;
    std::optional<VecRealAllocated> D_eq_impl;
    std::optional<MatRealAllocated> full_kkt_matrix_impl;
    std::optional<AugSystemSolver<ImplicitOcpType>> solver_impl;

    /// Reformulated case ///
    std::optional<ProblemDims> dims_reform;
    std::optional<ProblemInfo> info_reform;
    std::optional<Jacobian<OcpType>> jacobian_reform;
    std::optional<Hessian<OcpType>> hessian_reform;
    std::optional<MatRealAllocated> full_matrix_jacobian_reform;
    std::optional<MatRealAllocated> full_matrix_hessian_reform;
    std::optional<VecRealAllocated> x_reform;
    std::optional<VecRealAllocated> mult_reform;
    std::optional<VecRealAllocated> rhs_x_reform;
    std::optional<VecRealAllocated> rhs_g_reform;
    std::optional<VecRealAllocated> D_x_reform;
    std::optional<VecRealAllocated> D_s_reform;
    std::optional<VecRealAllocated> D_eq_reform;
    std::optional<MatRealAllocated> full_kkt_matrix_reform;
    std::optional<AugSystemSolver<OcpType>> solver_reform;

    /// Accelerated case ///
    std::optional<ProblemDims> dims_accel;
    std::optional<ProblemInfo> info_accel;
    // std::optional<Jacobian<AcceleratedOcpType>> jacobian_accel;
    // std::optional<Hessian<AcceleratedOcpType>> hessian_accel;
    std::optional<Jacobian<OcpType>> jacobian_accel;
    std::optional<Hessian<OcpType>> hessian_accel;
    std::optional<MatRealAllocated> full_matrix_jacobian_accel;
    std::optional<MatRealAllocated> full_matrix_hessian_accel;
    std::optional<VecRealAllocated> x_accel;
    std::optional<VecRealAllocated> mult_accel;
    std::optional<VecRealAllocated> rhs_x_accel;
    std::optional<VecRealAllocated> rhs_g_accel;
    std::optional<VecRealAllocated> D_x_accel;
    std::optional<VecRealAllocated> D_s_accel;
    std::optional<VecRealAllocated> D_eq_accel;
    std::optional<MatRealAllocated> full_kkt_matrix_accel;
    // std::optional<AugSystemSolver<AcceleratedOcpType>> solver_accel;
    std::optional<AugSystemSolver<OcpType>> solver_accel;

    std::vector<int> RandomVector(int size, int min_val, int max_val)
    {
        std::vector<int> vec(size);
        if (size == 0){ return vec;}

        vec[0] = rand() % (max_val - min_val + 1) + min_val;
        for (int i = 0; i < size; ++i){
            if (constant_dimensions){
                vec[i] = vec[0];
            } else {
                vec[i] = rand() % (max_val - min_val + 1) + min_val;
            }
        }
        return vec;
    }

    void ClearOptionals(){
        solver_expl.reset();
        hessian_expl.reset();
        jacobian_expl.reset();
        info_expl.reset();
        dims_expl.reset();
        full_matrix_jacobian_expl.reset();
        full_matrix_hessian_expl.reset();
        x_expl.reset();
        mult_expl.reset();
        rhs_x_expl.reset();
        rhs_g_expl.reset();
        D_x_expl.reset();
        D_s_expl.reset();
        D_eq_expl.reset();
        full_kkt_matrix_expl.reset();

        if (!use_generalization){
            solver_impl.reset();
            hessian_impl.reset();
            jacobian_impl.reset();
            info_impl.reset();
            dims_impl.reset();
            full_matrix_jacobian_impl.reset();
            full_matrix_hessian_impl.reset();
            x_impl.reset();
            mult_impl.reset();
            rhs_x_impl.reset();
            rhs_g_impl.reset();
            D_x_impl.reset();
            D_s_impl.reset();
            D_eq_impl.reset();
            full_kkt_matrix_impl.reset();
        }
            
        solver_reform.reset();
        hessian_reform.reset();
        jacobian_reform.reset();
        info_reform.reset();
        dims_reform.reset();
        full_matrix_jacobian_reform.reset();
        full_matrix_hessian_reform.reset();
        x_reform.reset();
        mult_reform.reset();
        rhs_x_reform.reset();
        rhs_g_reform.reset();
        D_x_reform.reset();
        D_s_reform.reset();
        D_eq_reform.reset();
        full_kkt_matrix_reform.reset();

        solver_accel.reset();
        hessian_accel.reset();
        jacobian_accel.reset();
        info_accel.reset();
        dims_accel.reset();
        full_matrix_jacobian_accel.reset();
        full_matrix_hessian_accel.reset();
        x_accel.reset();
        mult_accel.reset();
        rhs_x_accel.reset();
        rhs_g_accel.reset();
        D_x_accel.reset();
        D_s_accel.reset();
        D_eq_accel.reset();
        full_kkt_matrix_accel.reset();
    }

    void GetRandomDimensions()
    {
        ClearOptionals();
        int max_val = 40;
        K = rand() % 10 + 2; // Random K between 2 and 21
        nx = RandomVector(K, 0, max_val);
        if (full_rank){
            r = nx;
        } else {
            r = std::vector<Index>(K, 100);
            for (int k = 0; k < K; ++k){ 
                while (r[k] > nx[k]){ r[k] = rand() % (nx[k]+1);}
            }
        }
        nu = RandomVector(K, 0, max_val);
        ng = RandomVector(K, 0, max_val);
        for (int k = 0; k < K; ++k){
            bool okay = false;
            while (!okay){
                int max_allowed_ng = nx[k] + nu[k];
                if (k < K-1){ max_allowed_ng -= (nx[k+1] - r[k+1]);}
                if (ng[k] <= max_allowed_ng){
                    okay = true;
                } else {
                    // randomize both the nb of constraints and the nb of controls
                    if (constant_dimensions){
                        ng = RandomVector(K, 0, max_val);
                        nu = RandomVector(K, 0, max_val);
                    } else {
                        ng[k] = rand() % (max_val + 1);
                        nu[k] = rand() % (max_val + 1);
                    }
                }
            }
        }
        ng_ineq = RandomVector(K, 0, max_val);

        // K = 1;
        // nx = {1};
        // nu = {1};
        // ng = {1};
        // ng_ineq = {0};

        // std::cout << "K: " << K << std::endl;
        // std::cout << "nx:      "; for (auto val : nx){ std::cout << std::setw(4) << val << " ";} std::cout << std::endl;
        // std::cout << "r:       "; for (auto val : r){ std::cout << std::setw(4) << val << " ";} std::cout << std::endl;
        // std::cout << "nu:      "; for (auto val : nu){ std::cout << std::setw(4) << val << " ";} std::cout << std::endl;
        // std::cout << "ng:      "; for (auto val : ng){ std::cout << std::setw(4) << val << " ";} std::cout << std::endl;
        // std::cout << "ng_ineq: "; for (auto val : ng_ineq){ std::cout << std::setw(4) << val << " ";} std::cout << std::endl;

    }

    int GetRandomGeneralizedDimensions(){
        if (!constant_dimensions){ throw std::runtime_error("Invalid settings for Generalized Dimensions");}
        int max_val = 30;
        K = rand() % 10 + 2;
        nz = RandomVector(K, 0, max_val);
        nz[K-1] = 0;
        std::vector<Index> ng_true = RandomVector(K, 0, max_val);
        std::vector<Index> nu_true = RandomVector(K, 0, max_val);
        nx = RandomVector(K, nz[0], nz[0] + max_val);
        r = nx;
        ng_ineq = RandomVector(K, 0, max_val);
        nv = RandomVector(K, 0, max_val);

        nu = nu_true;
        ng = ng_true;

        for (int i = 0; i < K - 1; ++i){
            if (nx[i] + nu_true[i] + nx[i+1] < nv[i] + (nx[i] - nz[i]) ||
                nx[i] + nu_true[i] + nz[i] < ng_true[i] + nv[i] ||
                nx[i+1] < nz[i]){
            // try again
            return 1 + GetRandomGeneralizedDimensions();
            }
        }
        if (ng_true[K-1] > nx[K-1] + nu_true[K-1]){
            // try again
            return 1 + GetRandomGeneralizedDimensions();
        }

        return 1;
    }

    void AllocateSolvers(){
        AllocateExplicitSolver();
        if (!use_generalization){
            AllocateImplicitSolver();
        }
        AllocateReformulatedSolver();
        AllocateAcceleratedSolver();
        // std::cout << "KKT size expl:   " << full_kkt_matrix_expl.value().m() << " x " << full_kkt_matrix_expl.value().n() << std::endl;
        // std::cout << "KKT size impl:   " << full_kkt_matrix_impl.value().m() << " x " << full_kkt_matrix_impl.value().n() << std::endl;
        // std::cout << "KKT size reform: " << full_kkt_matrix_reform.value().m() << " x " << full_kkt_matrix_reform.value().n() << std::endl;
    }

    void AllocateExplicitSolver(){
        dims_expl.emplace(ProblemDims{K, nu, nx, ng, ng_ineq});
        info_expl.emplace(ProblemInfo(dims_expl.value()));
        jacobian_expl.emplace(Jacobian<OcpType>(dims_expl.value()));
        full_matrix_jacobian_expl.emplace(info_expl->number_of_eq_constraints, info_expl->number_of_primal_variables);
        hessian_expl.emplace(Hessian<OcpType>(dims_expl.value()));
        full_matrix_hessian_expl.emplace(info_expl->number_of_primal_variables, info_expl->number_of_primal_variables);
        x_expl.emplace(info_expl->number_of_primal_variables);
        mult_expl.emplace(info_expl->number_of_eq_constraints);
        rhs_x_expl.emplace(info_expl->number_of_primal_variables);
        rhs_g_expl.emplace(info_expl->number_of_eq_constraints);
        D_x_expl.emplace(info_expl->number_of_primal_variables);
        D_s_expl.emplace(info_expl->number_of_slack_variables);
        D_eq_expl.emplace(info_expl->number_of_g_eq_path);
        full_kkt_matrix_expl.emplace(info_expl->number_of_primal_variables + info_expl->number_of_eq_constraints,
                             info_expl->number_of_primal_variables + info_expl->number_of_eq_constraints);
        solver_expl.emplace(AugSystemSolver<OcpType>(info_expl.value()));
    }

    void AllocateImplicitSolver(){
        if (use_generalization){ throw std::runtime_error("Generalization not supported for implicit solver"); }
        dims_impl.emplace(ProblemDims{K, nu, nx, ng, ng_ineq});
        info_impl.emplace(ProblemInfo(dims_impl.value()));
        jacobian_impl.emplace(Jacobian<ImplicitOcpType>(dims_impl.value()));
        full_matrix_jacobian_impl.emplace(info_impl->number_of_eq_constraints, info_impl->number_of_primal_variables);
        hessian_impl.emplace(Hessian<ImplicitOcpType>(dims_impl.value()));
        full_matrix_hessian_impl.emplace(info_impl->number_of_primal_variables, info_impl->number_of_primal_variables);
        x_impl.emplace(info_impl->number_of_primal_variables);
        mult_impl.emplace(info_impl->number_of_eq_constraints);
        rhs_x_impl.emplace(info_impl->number_of_primal_variables);
        rhs_g_impl.emplace(info_impl->number_of_eq_constraints);
        D_x_impl.emplace(info_impl->number_of_primal_variables);
        D_s_impl.emplace(info_impl->number_of_slack_variables);
        D_eq_impl.emplace(info_impl->number_of_g_eq_path);
        full_kkt_matrix_impl.emplace(info_impl->number_of_primal_variables + info_impl->number_of_eq_constraints,
                             info_impl->number_of_primal_variables + info_impl->number_of_eq_constraints);
        solver_impl.emplace(AugSystemSolver<ImplicitOcpType>(info_impl.value()));
    }

    void AllocateReformulatedSolver(){
        std::vector<Index> nu_reform = nu;
        std::vector<Index> ng_reform = ng;
        for (int k = 0; k < K-1; ++k){
            if (use_generalization){
                nu_reform[k] += nz[k];
                ng_reform[k] += nv[k];
            } else {
                nu_reform[k] += nx[k+1];
                ng_reform[k] += nx[k+1];
            }
        }

        dims_reform.emplace(ProblemDims{K, nu_reform, nx, ng_reform, ng_ineq});
        info_reform.emplace(ProblemInfo(dims_reform.value()));
        jacobian_reform.emplace(Jacobian<OcpType>(dims_reform.value()));
        full_matrix_jacobian_reform.emplace(info_reform->number_of_eq_constraints, info_reform->number_of_primal_variables);
        hessian_reform.emplace(Hessian<OcpType>(dims_reform.value()));
        full_matrix_hessian_reform.emplace(info_reform->number_of_primal_variables, info_reform->number_of_primal_variables);
        x_reform.emplace(info_reform->number_of_primal_variables);
        mult_reform.emplace(info_reform->number_of_eq_constraints);
        rhs_x_reform.emplace(info_reform->number_of_primal_variables);
        rhs_g_reform.emplace(info_reform->number_of_eq_constraints);
        D_x_reform.emplace(info_reform->number_of_primal_variables);
        D_s_reform.emplace(info_reform->number_of_slack_variables);
        D_eq_reform.emplace(info_reform->number_of_g_eq_path);
        full_kkt_matrix_reform.emplace(info_reform->number_of_primal_variables + info_reform->number_of_eq_constraints,
                             info_reform->number_of_primal_variables + info_reform->number_of_eq_constraints);
        solver_reform.emplace(AugSystemSolver<OcpType>(info_reform.value()));
    }

    void AllocateAcceleratedSolver(){
        // std::vector<Index> nu_accel = nu;
        // std::vector<Index> ng_accel = ng;
        // for (int k = 0; k < K-1; ++k){
        //     if (use_generalization){
        //         nu_accel[k] += nz[k];
        //         ng_accel[k] += nv[k];
        //     } else {
        //         nu_accel[k] += nx[k+1];
        //         ng_accel[k] += nx[k+1];
        //     }
        // }

        // dims_accel.emplace(ProblemDims{K, nu_accel, nx, ng_accel, ng_ineq});
        // info_accel.emplace(ProblemInfo(dims_accel.value()));
        // // jacobian_accel.emplace(Jacobian<AcceleratedOcpType>(dims_accel.value()));
        // jacobian_accel.emplace(Jacobian<OcpType>(dims_accel.value()));
        // full_matrix_jacobian_accel.emplace(info_accel->number_of_eq_constraints, info_accel->number_of_primal_variables);
        // // hessian_accel.emplace(Hessian<AcceleratedOcpType>(dims_accel.value()));
        // hessian_accel.emplace(Hessian<OcpType>(dims_accel.value()));
        // full_matrix_hessian_accel.emplace(info_accel->number_of_primal_variables, info_accel->number_of_primal_variables);
        // x_accel.emplace(info_accel->number_of_primal_variables);
        // mult_accel.emplace(info_accel->number_of_eq_constraints);
        // rhs_x_accel.emplace(info_accel->number_of_primal_variables);
        // rhs_g_accel.emplace(info_accel->number_of_eq_constraints);
        // D_x_accel.emplace(info_accel->number_of_primal_variables);
        // D_s_accel.emplace(info_accel->number_of_slack_variables);
        // D_eq_accel.emplace(info_accel->number_of_g_eq_path);
        // full_kkt_matrix_accel.emplace(info_accel->number_of_primal_variables + info_accel->number_of_eq_constraints,
        //                      info_accel->number_of_primal_variables + info_accel->number_of_eq_constraints);
        // // solver_accel.emplace(AugSystemSolver<AcceleratedOcpType>(info_accel.value()));
        // solver_accel.emplace(AugSystemSolver<OcpType>(info_accel.value()));

        std::vector<Index> nu_accel = nu;
        std::vector<Index> ng_accel = ng;
        for (int k = 0; k < K-1; ++k){
            if (use_generalization){
                nu_accel[k] += nz[k];
                ng_accel[k] += nv[k];
            } else {
                nu_accel[k] += nx[k+1];
                ng_accel[k] += nx[k+1];
            }
        }

        dims_accel.emplace(ProblemDims{K, nu_accel, nx, ng_accel, ng_ineq});
        info_accel.emplace(ProblemInfo(dims_accel.value()));
        jacobian_accel.emplace(Jacobian<OcpType>(dims_accel.value()));
        full_matrix_jacobian_accel.emplace(info_accel->number_of_eq_constraints, info_accel->number_of_primal_variables);
        hessian_accel.emplace(Hessian<OcpType>(dims_accel.value()));
        full_matrix_hessian_accel.emplace(info_accel->number_of_primal_variables, info_accel->number_of_primal_variables);
        x_accel.emplace(info_accel->number_of_primal_variables);
        mult_accel.emplace(info_accel->number_of_eq_constraints);
        rhs_x_accel.emplace(info_accel->number_of_primal_variables);
        rhs_g_accel.emplace(info_accel->number_of_eq_constraints);
        D_x_accel.emplace(info_accel->number_of_primal_variables);
        D_s_accel.emplace(info_accel->number_of_slack_variables);
        D_eq_accel.emplace(info_accel->number_of_g_eq_path);
        full_kkt_matrix_accel.emplace(info_accel->number_of_primal_variables + info_accel->number_of_eq_constraints,
                             info_accel->number_of_primal_variables + info_accel->number_of_eq_constraints);
        solver_accel.emplace(AugSystemSolver<OcpType>(info_accel.value()));
    }

    void FillExplicitSolver(){
        x_expl = 0;
        full_matrix_jacobian_expl.value() = 0.;
        full_matrix_hessian_expl.value() = 0.;

        for (Index k = 0; k < info_expl.value().dims.K; ++k)
        {
            Index nu = info_expl.value().dims.number_of_controls[k];
            Index nx = info_expl.value().dims.number_of_states[k];
            Index offs_eq_dyn = info_expl.value().offsets_g_eq_dyn[k];
            Index offs_ux = info_expl.value().offsets_primal_u[k];
            Index offset_g_eq = info_expl.value().offsets_g_eq_path[k];
            Index offset_g_ineq = info_expl.value().offsets_g_eq_slack[k];
            Index ng = info_expl.value().dims.number_of_eq_constraints[k];
            Index ng_ineq = info_expl.value().dims.number_of_ineq_constraints[k];
            if (k < info_expl.value().dims.K - 1)
            {
                Index nx_next = info_expl.value().dims.number_of_states[k + 1];
                Index offs_x_next = info_expl.value().offsets_primal_x[k + 1];
                jacobian_expl.value().BAbt[k].block(nu + nx, nx_next, 0, 0) =
                    ::test::random_matrix(nu + nx, nx_next);
                full_matrix_jacobian_expl.value().block(nx_next, nu + nx, offs_eq_dyn, offs_ux) =
                transpose(jacobian_expl.value().BAbt[k].block(nu + nx, nx_next, 0, 0));
            }
            jacobian_expl.value().Gg_eqt[k].block(nu + nx, info_expl.value().dims.number_of_eq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info_expl.value().dims.number_of_eq_constraints[k]);
            full_matrix_jacobian_expl.value().block(ng, nu + nx, offset_g_eq, offs_ux) =
                transpose(jacobian_expl.value().Gg_eqt[k].block(nu + nx, ng, 0, 0));

            jacobian_expl.value().Gg_ineqt[k].block(nu + nx, info_expl.value().dims.number_of_ineq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info_expl.value().dims.number_of_ineq_constraints[k]);
            full_matrix_jacobian_expl.value().block(ng_ineq, nu + nx, offset_g_ineq, offs_ux) =
                transpose(jacobian_expl.value().Gg_ineqt[k].block(nu + nx, ng_ineq, 0, 0));

            hessian_expl.value().RSQrqt[k].block(nu + nx, nu + nx, 0, 0) = ::test::random_spd_matrix(nu + nx);
            full_matrix_hessian_expl.value().block(nu + nx, nu + nx, offs_ux, offs_ux) =
                hessian_expl.value().RSQrqt[k].block(nu + nx, nu + nx, 0, 0);
        }
        
        // set up the full KKT matrix
        full_kkt_matrix_expl.value().block(info_expl.value().number_of_primal_variables, info_expl.value().number_of_primal_variables, 0,
                              0) = full_matrix_hessian_expl.value();
        full_kkt_matrix_expl.value().block(info_expl.value().number_of_primal_variables, info_expl.value().number_of_eq_constraints, 0,
                              info_expl.value().number_of_primal_variables) = transpose(full_matrix_jacobian_expl.value());
        full_kkt_matrix_expl.value().block(info_expl.value().number_of_eq_constraints, info_expl.value().number_of_primal_variables,
                              info_expl.value().number_of_primal_variables, 0) = full_matrix_jacobian_expl.value();

        // fill the x vector with random values
        for (Index i = 0; i < info_expl.value().number_of_primal_variables; ++i){
            rhs_x_expl.value()(i) = 10 + 1.0 * i; D_x_expl.value()(i) = 1.0 * (i + 0.1);
        }
        // fill the mult vector with random values
        for (Index i = 0; i < info_expl.value().number_of_eq_constraints; ++i){
            rhs_g_expl.value()(i) = 1.0 * i;
        }
        for (Index i = 0; i < info_expl.value().number_of_g_eq_path; ++i){
            D_eq_expl.value()(i) = 10.0 * (i + 1);
        }
        for (Index i = 0; i < info_expl.value().number_of_slack_variables; ++i){
            D_s_expl.value()(i) =  1.0 + 0*10.0 * (i + 0.1);
        }
    }

    void FillImplicitSolver(){
        if (use_generalization){
            // the implicit algorithm assumes nx_next dynamics constraints, which is not guaranteed in the generalized case
            // --> implicit recursion should be modified to allow for an arbitrary number of dynamics constraints
            throw std::runtime_error("Implicit solver cannot be used when using the generalization of dimensions.");
            return;
        }

        x_impl = 0;
        full_matrix_jacobian_impl.value() = 0.;
        full_matrix_hessian_impl.value() = 0.;

        for (Index k = 0; k < info_impl.value().dims.K; ++k)
        {
            Index nu = info_impl.value().dims.number_of_controls[k];
            Index nx = info_impl.value().dims.number_of_states[k];
            Index offs_eq_dyn = info_impl.value().offsets_g_eq_dyn[k];
            Index offs_ux = info_impl.value().offsets_primal_u[k];
            Index offset_g_eq = info_impl.value().offsets_g_eq_path[k];
            Index offset_g_ineq = info_impl.value().offsets_g_eq_slack[k];
            Index ng = info_impl.value().dims.number_of_eq_constraints[k];
            Index ng_ineq = info_impl.value().dims.number_of_ineq_constraints[k];
            if (k < info_impl.value().dims.K - 1)
            {
                Index nx_next = info_impl.value().dims.number_of_states[k + 1];
                Index offs_x_next = info_impl.value().offsets_primal_x[k + 1];
                jacobian_impl.value().BAbt[k].block(nu + nx, nx_next, 0, 0) =
                    ::test::random_matrix(nu + nx, nx_next);
                full_matrix_jacobian_impl.value().block(nx_next, nu + nx, offs_eq_dyn, offs_ux) =
                transpose(jacobian_impl.value().BAbt[k].block(nu + nx, nx_next, 0, 0));

                jacobian_impl.value().Jt[k].block(nx_next, nx_next, 0, 0) =
                ::test::random_degenerate_matrix(nx_next, r[k+1]);
                full_matrix_jacobian_impl.value().block(nx_next, nx_next, offs_eq_dyn, offs_x_next) = 
                    transpose(jacobian_impl.value().Jt[k]);

                hessian_impl.value().FuFx[k].block(nx + nu, nx_next, 0, 0) =
                    ::test::random_matrix(nx + nu, nx_next);
                full_matrix_hessian_impl.value().block(nx_next, nu + nx, offs_x_next, offs_ux) = 
                    transpose(hessian_impl.value().FuFx[k]);
                full_matrix_hessian_impl.value().block(nu + nx, nx_next, offs_ux, offs_x_next) =
                    hessian_impl.value().FuFx[k];
            }
            jacobian_impl.value().Gg_eqt[k].block(nu + nx, info_impl.value().dims.number_of_eq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info_impl.value().dims.number_of_eq_constraints[k]);
            full_matrix_jacobian_impl.value().block(ng, nu + nx, offset_g_eq, offs_ux) =
                transpose(jacobian_impl.value().Gg_eqt[k].block(nu + nx, ng, 0, 0));

            jacobian_impl.value().Gg_ineqt[k].block(nu + nx, info_impl.value().dims.number_of_ineq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info_impl.value().dims.number_of_ineq_constraints[k]);
            full_matrix_jacobian_impl.value().block(ng_ineq, nu + nx, offset_g_ineq, offs_ux) =
                transpose(jacobian_impl.value().Gg_ineqt[k].block(nu + nx, ng_ineq, 0, 0));

            hessian_impl.value().RSQrqt[k].block(nu + nx, nu + nx, 0, 0) = ::test::random_spd_matrix(nu + nx);
            full_matrix_hessian_impl.value().block(nu + nx, nu + nx, offs_ux, offs_ux) =
                hessian_impl.value().RSQrqt[k].block(nu + nx, nu + nx, 0, 0);
        }
        
        // set up the full KKT matrix
        full_kkt_matrix_impl.value().block(info_impl.value().number_of_primal_variables, info_impl.value().number_of_primal_variables, 0,
                              0) = full_matrix_hessian_impl.value();
        full_kkt_matrix_impl.value().block(info_impl.value().number_of_primal_variables, info_impl.value().number_of_eq_constraints, 0,
                              info_impl.value().number_of_primal_variables) = transpose(full_matrix_jacobian_impl.value());
        full_kkt_matrix_impl.value().block(info_impl.value().number_of_eq_constraints, info_impl.value().number_of_primal_variables,
                              info_impl.value().number_of_primal_variables, 0) = full_matrix_jacobian_impl.value();

        // fill the x vector with random values
        for (Index i = 0; i < info_impl.value().number_of_primal_variables; ++i){
            rhs_x_impl.value()(i) = 10 + 1.0 * i; D_x_impl.value()(i) = 1.0 * (i + 0.1);
        }
        // fill the mult vector with random values
        for (Index i = 0; i < info_impl.value().number_of_eq_constraints; ++i){
            rhs_g_impl.value()(i) = 1.0 * i;
        }
        for (Index i = 0; i < info_impl.value().number_of_g_eq_path; ++i){
            D_eq_impl.value()(i) = 10.0 * (i + 1);
        }
        for (Index i = 0; i < info_impl.value().number_of_slack_variables; ++i){
            D_s_impl.value()(i) =  1.0 + 0*10.0 * (i + 0.1);
        }
    }

    void FillReformulatedSolver(){
        x_reform = 0;
        full_matrix_jacobian_reform.value() = 0.;
        full_matrix_hessian_reform.value() = 0.;

        for (Index k = 0; k < info_reform.value().dims.K; ++k)
        {
            Index nu = info_reform.value().dims.number_of_controls[k];
            Index nx = info_reform.value().dims.number_of_states[k];
            Index offs_eq_dyn = info_reform.value().offsets_g_eq_dyn[k];
            Index offs_ux = info_reform.value().offsets_primal_u[k];
            Index offset_g_eq = info_reform.value().offsets_g_eq_path[k];
            Index offset_g_ineq = info_reform.value().offsets_g_eq_slack[k];
            Index ng = info_reform.value().dims.number_of_eq_constraints[k];
            Index ng_ineq = info_reform.value().dims.number_of_ineq_constraints[k];
            Index nu_true;
            Index ng_true;
            if (use_generalization){
                nu_true = (k < info_reform.value().dims.K - 1) ? nu - nz[k] : nu;
                ng_true = (k < info_reform.value().dims.K - 1) ? ng - nv[k] : ng;
            } else {
                nu_true = (k < info_reform.value().dims.K - 1) ? nu - info_reform.value().dims.number_of_states[k+1] : nu;
                ng_true = (k < info_reform.value().dims.K - 1) ? ng - info_reform.value().dims.number_of_states[k+1] : ng;
            }
            if (k < info_reform.value().dims.K - 1)
            {
                Index nx_next = info_reform.value().dims.number_of_states[k + 1];
                Index offs_x_next = info_reform.value().offsets_primal_x[k + 1];
                if (use_generalization){
                    // Bottom part of B
                    jacobian_reform.value().BAbt[k].block(nu_true, nx_next - nz[k], 0, nz[k]) =
                        ::test::random_matrix(nu_true, nx_next - nz[k]);
                    // Bottom part of A and b
                    jacobian_reform.value().BAbt[k].block(nx + 1, nx_next - nz[k], nu, nz[k]) =
                        ::test::random_matrix(nx + 1, nx_next - nz[k]);
                    // identity part in B
                    jacobian_reform.value().BAbt[k].block(nz[k], nz[k], nu_true, 0) =
                        ::test::identity_matrix(nz[k], -1);
                } else {
                    jacobian_reform.value().BAbt[k].block(nx_next, nx_next, nu_true, 0) =
                        ::test::identity_matrix(nx_next, -1);
                }
                full_matrix_jacobian_reform.value().block(nx_next, nu + nx, offs_eq_dyn, offs_ux) =
                    transpose(jacobian_reform.value().BAbt[k].block(nu + nx, nx_next, 0, 0));
            }
            jacobian_reform.value().Gg_eqt[k].block(nu + nx, info_reform.value().dims.number_of_eq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info_reform.value().dims.number_of_eq_constraints[k]);
            jacobian_reform.value().Gg_eqt[k].block(nu - nu_true, ng_true, nu_true, 0) =
                ::test::empty_matrix(nu - nu_true, ng_true);
            full_matrix_jacobian_reform.value().block(ng, nu + nx, offset_g_eq, offs_ux) =
                transpose(jacobian_reform.value().Gg_eqt[k].block(nu + nx, ng, 0, 0));

            jacobian_reform.value().Gg_ineqt[k].block(nu + nx, info_reform.value().dims.number_of_ineq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info_reform.value().dims.number_of_ineq_constraints[k]);
            jacobian_reform.value().Gg_ineqt[k].block(nu - nu_true, ng_ineq, nu_true, 0) =
                ::test::empty_matrix(nu - nu_true, ng_ineq);
            full_matrix_jacobian_reform.value().block(ng_ineq, nu + nx, offset_g_ineq, offs_ux) =
                transpose(jacobian_reform.value().Gg_ineqt[k].block(nu + nx, ng_ineq, 0, 0));

            hessian_reform.value().RSQrqt[k].block(nu + nx, nu + nx, 0, 0) = ::test::random_spd_matrix(nu + nx);
            // hessian_reform.value().RSQrqt[k].block(nu - nu_true, nu + nx, nu_true, 0) = ::test::empty_matrix(nu_true, nu + nx);
            // hessian_reform.value().RSQrqt[k].block(nu + nx, nu - nu_true, 0, nu_true) = ::test::empty_matrix(nu + nx, nu_true);
            full_matrix_hessian_reform.value().block(nu + nx, nu + nx, offs_ux, offs_ux) =
                hessian_reform.value().RSQrqt[k].block(nu + nx, nu + nx, 0, 0);
        }
        
        // set up the full KKT matrix
        full_kkt_matrix_reform.value().block(info_reform.value().number_of_primal_variables, info_reform.value().number_of_primal_variables, 0,
                              0) = full_matrix_hessian_reform.value();
        full_kkt_matrix_reform.value().block(info_reform.value().number_of_primal_variables, info_reform.value().number_of_eq_constraints, 0,
                              info_reform.value().number_of_primal_variables) = transpose(full_matrix_jacobian_reform.value());
        full_kkt_matrix_reform.value().block(info_reform.value().number_of_eq_constraints, info_reform.value().number_of_primal_variables,
                              info_reform.value().number_of_primal_variables, 0) = full_matrix_jacobian_reform.value();

        // fill the x vector with random values
        for (Index i = 0; i < info_reform.value().number_of_primal_variables; ++i){
            rhs_x_reform.value()(i) = 10 + 1.0 * i; D_x_reform.value()(i) = 1.0 * (i + 0.1);
        }
        // fill the mult vector with random values
        for (Index i = 0; i < info_reform.value().number_of_eq_constraints; ++i){
            rhs_g_reform.value()(i) = 1.0 * i;
        }
        for (Index i = 0; i < info_reform.value().number_of_g_eq_path; ++i){
            D_eq_reform.value()(i) = 10.0 * (i + 1);
        }
        for (Index i = 0; i < info_reform.value().number_of_slack_variables; ++i){
            D_s_reform.value()(i) =  1.0 + 0*10.0 * (i + 0.1);
        }
    }

    void FillAcceleratedSolver(){
        // x_accel = 0;
        // full_matrix_jacobian_accel.value() = 0.;
        // full_matrix_hessian_accel.value() = 0.;

        // for (Index k = 0; k < info_accel.value().dims.K; ++k)
        // {
        //     Index nu = info_accel.value().dims.number_of_controls[k];
        //     Index nx = info_accel.value().dims.number_of_states[k];
        //     Index offs_eq_dyn = info_accel.value().offsets_g_eq_dyn[k];
        //     Index offs_ux = info_accel.value().offsets_primal_u[k];
        //     Index offset_g_eq = info_accel.value().offsets_g_eq_path[k];
        //     Index offset_g_ineq = info_accel.value().offsets_g_eq_slack[k];
        //     Index ng = info_accel.value().dims.number_of_eq_constraints[k];
        //     Index ng_ineq = info_accel.value().dims.number_of_ineq_constraints[k];
        //     Index nu_true;
        //     Index ng_true;
        //     if (use_generalization){
        //         nu_true = (k < info_accel.value().dims.K - 1) ? nu - nz[k] : nu;
        //         ng_true = (k < info_accel.value().dims.K - 1) ? ng - nv[k] : ng;
        //     } else {
        //         nu_true = (k < info_accel.value().dims.K - 1) ? nu - info_accel.value().dims.number_of_states[k+1] : nu;
        //         ng_true = (k < info_accel.value().dims.K - 1) ? ng - info_accel.value().dims.number_of_states[k+1] : ng;
        //     }
        //     if (k < info_accel.value().dims.K - 1)
        //     {
        //         Index nx_next = info_accel.value().dims.number_of_states[k + 1];
        //         Index offs_x_next = info_accel.value().offsets_primal_x[k + 1];
        //         if (use_generalization){
        //             // Bottom part of B
        //             jacobian_reform.value().BAbt[k].block(nu_true, nx_next - nz[k], 0, nz[k]) =
        //                 ::test::random_matrix(nu_true, nx_next - nz[k]);
        //             // Bottom part of A and b
        //             jacobian_reform.value().BAbt[k].block(nx + 1, nx_next - nz[k], nu, nz[k]) =
        //                 ::test::random_matrix(nx + 1, nx_next - nz[k]);
        //             // identity part in B
        //             jacobian_reform.value().BAbt[k].block(nz[k], nz[k], nu_true, 0) =
        //                 ::test::identity_matrix(nz[k], -1);
        //         } else {
        //             jacobian_reform.value().BAbt[k].block(nx_next, nx_next, nu_true, 0) =
        //                 ::test::identity_matrix(nx_next, -1);
        //         }
        //         full_matrix_jacobian_accel.value().block(nx_next, nu + nx, offs_eq_dyn, offs_ux) =
        //             transpose(jacobian_accel.value().BAbt[k].block(nu + nx, nx_next, 0, 0));
        //     }
        //     jacobian_accel.value().Gg_eqt[k].block(nu + nx, info_accel.value().dims.number_of_eq_constraints[k], 0, 0) =
        //         ::test::random_matrix(nu + nx, info_accel.value().dims.number_of_eq_constraints[k]);
        //     jacobian_accel.value().Gg_eqt[k].block(nu - nu_true, ng_true, nu_true, 0) =
        //         ::test::empty_matrix(nu - nu_true, ng_true);
        //     full_matrix_jacobian_accel.value().block(ng, nu + nx, offset_g_eq, offs_ux) =
        //         transpose(jacobian_accel.value().Gg_eqt[k].block(nu + nx, ng, 0, 0));

        //     jacobian_accel.value().Gg_ineqt[k].block(nu + nx, info_accel.value().dims.number_of_ineq_constraints[k], 0, 0) =
        //         ::test::random_matrix(nu + nx, info_accel.value().dims.number_of_ineq_constraints[k]);
        //     jacobian_accel.value().Gg_ineqt[k].block(nu - nu_true, ng_ineq, nu_true, 0) =
        //         ::test::empty_matrix(nu - nu_true, ng_ineq);
        //     full_matrix_jacobian_accel.value().block(ng_ineq, nu + nx, offset_g_ineq, offs_ux) =
        //         transpose(jacobian_accel.value().Gg_ineqt[k].block(nu + nx, ng_ineq, 0, 0));

        //     hessian_accel.value().RSQrqt[k].block(nu + nx, nu + nx, 0, 0) = ::test::random_spd_matrix(nu + nx);
        //     // hessian_accel.value().RSQrqt[k].block(nu - nu_true, nu + nx, nu_true, 0) = ::test::empty_matrix(nu_true, nu + nx);
        //     // hessian_accel.value().RSQrqt[k].block(nu + nx, nu - nu_true, 0, nu_true) = ::test::empty_matrix(nu + nx, nu_true);
        //     full_matrix_hessian_accel.value().block(nu + nx, nu + nx, offs_ux, offs_ux) =
        //         hessian_accel.value().RSQrqt[k].block(nu + nx, nu + nx, 0, 0);
        // }
        
        // // set up the full KKT matrix
        // full_kkt_matrix_accel.value().block(info_accel.value().number_of_primal_variables, info_accel.value().number_of_primal_variables, 0,
        //                       0) = full_matrix_hessian_accel.value();
        // full_kkt_matrix_accel.value().block(info_accel.value().number_of_primal_variables, info_accel.value().number_of_eq_constraints, 0,
        //                       info_accel.value().number_of_primal_variables) = transpose(full_matrix_jacobian_accel.value());
        // full_kkt_matrix_accel.value().block(info_accel.value().number_of_eq_constraints, info_accel.value().number_of_primal_variables,
        //                       info_accel.value().number_of_primal_variables, 0) = full_matrix_jacobian_accel.value();

        // // fill the x vector with random values
        // for (Index i = 0; i < info_accel.value().number_of_primal_variables; ++i){
        //     rhs_x_accel.value()(i) = 10 + 1.0 * i; D_x_accel.value()(i) = 1.0 * (i + 0.1);
        // }
        // // fill the mult vector with random values
        // for (Index i = 0; i < info_accel.value().number_of_eq_constraints; ++i){
        //     rhs_g_accel.value()(i) = 1.0 * i;
        // }
        // for (Index i = 0; i < info_accel.value().number_of_g_eq_path; ++i){
        //     D_eq_accel.value()(i) = 10.0 * (i + 1);
        // }
        // for (Index i = 0; i < info_accel.value().number_of_slack_variables; ++i){
        //     D_s_accel.value()(i) =  1.0 + 0*10.0 * (i + 0.1);
        // }

        // // pass on options
        // // if (use_generalization){
        // //     solver_accel.value().set_nb_of_dynamics_constraints(nv[0]);
        // //     solver_accel.value().set_nb_of_zk_vars(nz[0]);
        // // }


        x_accel = 0;
        full_matrix_jacobian_accel.value() = 0.;
        full_matrix_hessian_accel.value() = 0.;

        for (Index k = 0; k < info_accel.value().dims.K; ++k)
        {
            Index nu = info_accel.value().dims.number_of_controls[k];
            Index nx = info_accel.value().dims.number_of_states[k];
            Index offs_eq_dyn = info_accel.value().offsets_g_eq_dyn[k];
            Index offs_ux = info_accel.value().offsets_primal_u[k];
            Index offset_g_eq = info_accel.value().offsets_g_eq_path[k];
            Index offset_g_ineq = info_accel.value().offsets_g_eq_slack[k];
            Index ng = info_accel.value().dims.number_of_eq_constraints[k];
            Index ng_ineq = info_accel.value().dims.number_of_ineq_constraints[k];
            Index nu_true;
            Index ng_true;
            if (use_generalization){
                nu_true = (k < info_accel.value().dims.K - 1) ? nu - nz[k] : nu;
                ng_true = (k < info_accel.value().dims.K - 1) ? ng - nv[k] : ng;
            } else {
                nu_true = (k < info_accel.value().dims.K - 1) ? nu - info_accel.value().dims.number_of_states[k+1] : nu;
                ng_true = (k < info_accel.value().dims.K - 1) ? ng - info_accel.value().dims.number_of_states[k+1] : ng;
            }
            if (k < info_accel.value().dims.K - 1)
            {
                Index nx_next = info_accel.value().dims.number_of_states[k + 1];
                Index offs_x_next = info_accel.value().offsets_primal_x[k + 1];
                if (use_generalization){
                    // Bottom part of B
                    jacobian_accel.value().BAbt[k].block(nu_true, nx_next - nz[k], 0, nz[k]) =
                        ::test::random_matrix(nu_true, nx_next - nz[k]);
                    // Bottom part of A and b
                    jacobian_accel.value().BAbt[k].block(nx + 1, nx_next - nz[k], nu, nz[k]) =
                        ::test::random_matrix(nx + 1, nx_next - nz[k]);
                    // identity part in B
                    jacobian_accel.value().BAbt[k].block(nz[k], nz[k], nu_true, 0) =
                        ::test::identity_matrix(nz[k], -1);
                } else {
                    jacobian_accel.value().BAbt[k].block(nx_next, nx_next, nu_true, 0) =
                        ::test::identity_matrix(nx_next, -1);
                }
                full_matrix_jacobian_accel.value().block(nx_next, nu + nx, offs_eq_dyn, offs_ux) =
                    transpose(jacobian_accel.value().BAbt[k].block(nu + nx, nx_next, 0, 0));
            }
            jacobian_accel.value().Gg_eqt[k].block(nu + nx, info_accel.value().dims.number_of_eq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info_accel.value().dims.number_of_eq_constraints[k]);
            jacobian_accel.value().Gg_eqt[k].block(nu - nu_true, ng_true, nu_true, 0) =
                ::test::empty_matrix(nu - nu_true, ng_true);
            full_matrix_jacobian_accel.value().block(ng, nu + nx, offset_g_eq, offs_ux) =
                transpose(jacobian_accel.value().Gg_eqt[k].block(nu + nx, ng, 0, 0));

            jacobian_accel.value().Gg_ineqt[k].block(nu + nx, info_accel.value().dims.number_of_ineq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info_accel.value().dims.number_of_ineq_constraints[k]);
            jacobian_accel.value().Gg_ineqt[k].block(nu - nu_true, ng_ineq, nu_true, 0) =
                ::test::empty_matrix(nu - nu_true, ng_ineq);
            full_matrix_jacobian_accel.value().block(ng_ineq, nu + nx, offset_g_ineq, offs_ux) =
                transpose(jacobian_accel.value().Gg_ineqt[k].block(nu + nx, ng_ineq, 0, 0));

            hessian_accel.value().RSQrqt[k].block(nu + nx, nu + nx, 0, 0) = ::test::random_spd_matrix(nu + nx);
            // hessian_accel.value().RSQrqt[k].block(nu - nu_true, nu + nx, nu_true, 0) = ::test::empty_matrix(nu_true, nu + nx);
            // hessian_accel.value().RSQrqt[k].block(nu + nx, nu - nu_true, 0, nu_true) = ::test::empty_matrix(nu + nx, nu_true);
            full_matrix_hessian_accel.value().block(nu + nx, nu + nx, offs_ux, offs_ux) =
                hessian_accel.value().RSQrqt[k].block(nu + nx, nu + nx, 0, 0);
        }
        
        // set up the full KKT matrix
        full_kkt_matrix_accel.value().block(info_accel.value().number_of_primal_variables, info_accel.value().number_of_primal_variables, 0,
                              0) = full_matrix_hessian_accel.value();
        full_kkt_matrix_accel.value().block(info_accel.value().number_of_primal_variables, info_accel.value().number_of_eq_constraints, 0,
                              info_accel.value().number_of_primal_variables) = transpose(full_matrix_jacobian_accel.value());
        full_kkt_matrix_accel.value().block(info_accel.value().number_of_eq_constraints, info_accel.value().number_of_primal_variables,
                              info_accel.value().number_of_primal_variables, 0) = full_matrix_jacobian_accel.value();

        // fill the x vector with random values
        for (Index i = 0; i < info_accel.value().number_of_primal_variables; ++i){
            rhs_x_accel.value()(i) = 10 + 1.0 * i; D_x_accel.value()(i) = 1.0 * (i + 0.1);
        }
        // fill the mult vector with random values
        for (Index i = 0; i < info_accel.value().number_of_eq_constraints; ++i){
            rhs_g_accel.value()(i) = 1.0 * i;
        }
        for (Index i = 0; i < info_accel.value().number_of_g_eq_path; ++i){
            D_eq_accel.value()(i) = 10.0 * (i + 1);
        }
        for (Index i = 0; i < info_accel.value().number_of_slack_variables; ++i){
            D_s_accel.value()(i) =  1.0 + 0*10.0 * (i + 0.1);
        }
    }

    void Randomize(){
        // std::cout << "randomizing dimensions" << std::endl;
        if (use_generalization){
            int nb_tries = GetRandomGeneralizedDimensions();
            // std::cout << "nb_tries: " << nb_tries << std::endl;
        } else {
            GetRandomDimensions();
        }

        // std::cout << "allocating solvers" << std::endl;
        AllocateSolvers();

        // std::cout << "filling solvers" << std::endl;
        // std::cout << "\texplicit:" << std::endl;
        FillExplicitSolver();
        if (!use_generalization){
            // std::cout << "\timplicit:" << std::endl;
            FillImplicitSolver();
            solver_impl.value().set_performance_mode(true);
        }
        // std::cout << "\treformulated:" << std::endl;
        FillReformulatedSolver();
        // std::cout << "\taccelerated:" << std::endl;
        FillAcceleratedSolver();
        // std::cout << "done" << std::endl;

    }

    void SetUp()
    {
        Randomize();
    };
};

void PrintLine(std::string name, long int val, long int total, bool show_absolute=true){
    // append name with spaces to align with the longest name
    int max_name_length = 30;
    name += ": ";
    if (name.length() < max_name_length){ name += std::string(max_name_length - name.length(), ' ');}
    std::cout << "  - " << name << std::setw(4) << std::setprecision(3) << 100.0 * val / total << " %";
    if (show_absolute){
        std::cout << " (" << val << ")";
    }
    std::cout << std::endl;
}

void PrintBreakdown(const std::map<std::string, long int>& timings, 
                    const std::string breakdown_key,
                    const std::map<std::string, std::vector<std::string>>& breakdown,
                    const std::map<std::string, std::vector<std::string>>& breakdown_totals){
    std::cout << "Breakdown of " << breakdown_key << ":" << std::endl;
    std::map<std::string, std::string> key_translations = {
        {"total_ns_expl", "explicit"},
        {"total_ns_impl", "implicit"},
        {"total_ns_reform", "reformulated"},
        {"total_ns_impl_solve", "implicit solve"},
        {"total_ns_impl_preprocess", "implicit preprocess"},
        {"total_ns_impl_postprocess", "implicit postprocess"},
        {"total_pre_jac", "jacobian preprocessing"},
        {"total_pre_hess", "hessian preprocessing"},
        {"total_pre_reg", "regularization"},
        {"total_pre_decomp", "decomposition"},
        {"total_pre_info", "info"},
        {"total_pre_rhs", "rhs"},
        {"total_decomp_copies", "copies"},
        {"total_decomp_decomp", "decomp"},
        {"total_decomp_scale1", "scale1"},
        {"total_decomp_scale2", "scale2"},
        {"total_decomp_permutation", "permutation"},
        {"total_decomp_store", "store"},
        {"total_lu_reform", "reformulated lu"},
        {"total_lu_impl", "implicit lu"},
        {"total_ns_impl_solve_inner", "implicit solve (inner)"},
        {"total_solve_RSQrqt_copy", "copy RSQrqt"},
        {"total_solve_FuFx_addition", "FuFx addition"},
        {"total_solve_GuGx_addition", "GuGx addition"},
        {"total_solve_GuGx_hat_addition", "GuGx hat addition"},
        {"total_solve_ukb_tilde_addition", "ukb tilde addition"},
        {"total_solve_lambdatilde_addition", "lambdatilde addition"},
        {"total_solve_FuFx_addition_forward", "FuFx addition forward"},
    };

    for (const auto& sub_key : breakdown.at(breakdown_key)){
        std::string key_translation = key_translations.count(sub_key) ? key_translations[sub_key] : sub_key;
        PrintLine(key_translation, timings.at(sub_key), timings.at(breakdown_totals.at(breakdown_key)[0]), true);
    }
}

TEST_F(RandomBenchmarkTest, Test)
{
    std::cout << "running test" << std::endl;
    int nb_runs = 2000;
    int nb_runs_completed = 0;
    bool write_csv = true;

    bool skip_impl = USE_GENERALIZATION;

    std::map<std::string, long int> timings = {
        {"total_ns_expl", 0},
        {"total_ns_impl", 0},
        {"total_ns_reform", 0},
        {"total_ns_accel", 0},
        {"total_ns_impl_solve", 0},
        {"total_ns_impl_preprocess", 0},
        {"total_ns_impl_postprocess", 0},
        {"total_pre_jac", 0},
        {"total_pre_hess", 0},
        {"total_pre_reg", 0},
        {"total_pre_decomp", 0},
        {"total_pre_info", 0},
        {"total_pre_rhs", 0},
        {"total_decomp_copies", 0},
        {"total_decomp_decomp", 0},
        {"total_decomp_scale1", 0},
        {"total_decomp_scale2", 0},
        {"total_decomp_permutation", 0},
        {"total_decomp_store", 0},
        {"total_lu_reform", 0},
        {"total_ns_accel", 0},
        {"total_lu_impl", 0},
        {"total_ns_impl_solve_inner", 0},
        {"total_solve_RSQrqt_copy", 0},
        {"total_solve_FuFx_addition", 0},
        {"total_solve_GuGx_addition", 0},
        {"total_solve_GuGx_hat_addition", 0},
        {"total_solve_ukb_tilde_addition", 0},
        {"total_solve_lambdatilde_addition", 0},
        {"total_solve_FuFx_addition_forward", 0},
        {"total_expl_backward", 0},
        {"total_impl_backward", 0},
        {"total_reform_backward", 0},
        {"total_accel_backward", 0},
        {"total_expl_initial", 0},
        {"total_impl_initial", 0},
        {"total_reform_initial", 0},
        {"total_accel_initial", 0},
        {"total_expl_forward", 0},
        {"total_impl_forward", 0},
        {"total_reform_forward", 0},
        {"total_accel_forward", 0},
        {"total_post_rearrange_solution", 0},
        {"total_post_scale_solution", 0},
        {"total_post_reset_jacobian_pre", 0},
        {"total_post_reset_hessian_pre", 0},
        {"total_post_regularization", 0}
    };

    std::ofstream f;    
    if (write_csv){
        f.open("random_benchmark_results_generalized_faked.csv");
        f << "K,nu,nx,r,ng,ng_ineq,";
        if (skip_impl){ f << "nz,nv,";}
        f << "t_expl,t_expl_backward,t_expl_solve,t_expl_forward,";
        if (!skip_impl){
            f << "t_impl,t_impl_backward,t_impl_solve,t_impl_forward,t_impl_pre,t_impl_solve,t_impl_post,";
        }
        f << "t_reform,t_reform_backward,t_reform_solve,t_reform_forward,";
        f << "t_accel,t_accel_backward,t_accel_solve,t_accel_forward,";
        f << "lu_expl,";
        if (!skip_impl){ f << "lu_impl,";}
        f << "lu_reform,lu_accel";
        if (!skip_impl){ f << ",impl_decomp"; }
        f << "\n";
    }
    
    int nb_consecutive_failures = 0;
    std::array<int, 4> order = {0, 1, 2, 3};

    auto start_impl = std::chrono::steady_clock::now();
    auto stop_impl = std::chrono::steady_clock::now();
    auto start_expl = std::chrono::steady_clock::now();
    auto stop_expl = std::chrono::steady_clock::now();
    auto start_reform = std::chrono::steady_clock::now();
    auto stop_reform = std::chrono::steady_clock::now();
    auto start_accel = std::chrono::steady_clock::now();
    auto stop_accel = std::chrono::steady_clock::now();
    Index ret_impl, ret_expl, ret_reform, ret_accel;
    std::mt19937 rng(std::random_device{}());

    auto benchmark_start_time = std::chrono::steady_clock::now();
    while (nb_runs_completed < nb_runs){
        // std::cout << "Run " << nb_runs_completed + 1 << "/" << nb_runs << std::endl;
        // overwrite current status
        std::cout << "progress: " << std::setw(5) << std::setprecision(2) << 100.0 * nb_runs_completed / nb_runs << " %";
        auto current_time = std::chrono::steady_clock::now();
        // compute estimated time remaining
        long int elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time - benchmark_start_time).count();
        long int estimated_total_ns = elapsed_ns * nb_runs / (nb_runs_completed + 1);
        long int estimated_remaining_ns = estimated_total_ns - elapsed_ns;
        long int estimated_remaining_min = std::chrono::duration_cast<std::chrono::minutes>(std::chrono::nanoseconds(estimated_remaining_ns)).count();
        long int estimated_remaining_seconds = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::nanoseconds(estimated_remaining_ns)).count() % 60;
        std::cout << " (estimated remaining time: " << estimated_remaining_min << " min " << estimated_remaining_seconds << " sec)";
        if (nb_consecutive_failures > 0){ std::cout << " (+" << nb_consecutive_failures << " fails)"; }

        std::cout << "\r" << std::flush;
        Randomize();

        // std::cout << "explicit:\n" << full_kkt_matrix_expl.value() << std::endl;
        // std::cout << "implicit:\n" << full_kkt_matrix_impl.value() << std::endl;
        // std::cout << "reformulated:\n" << full_kkt_matrix_reform.value() << std::endl;
        // std::cout << std::endl;

        int nb_averaging_runs = 20;
        long int ns_expl = 0;
        long int ns_reform = 0;
        long int ns_accel = 0;
        long int ns_impl = 0;

        for (int averaging_run_counter = 0; averaging_run_counter < nb_averaging_runs; ++averaging_run_counter){
        std::shuffle(order.begin(), order.end(), rng);

        try{
        for (int idx : order){
            if (idx == 0 && !skip_impl){
                // implicit //
                // std::cout << "Running implicit solver..." << std::endl;
                start_impl = std::chrono::steady_clock::now();
                ret_impl = solver_impl.value().solve(info_impl.value(), jacobian_impl.value(), hessian_impl.value(), D_x_impl.value(), D_s_impl.value(), rhs_x_impl.value(), rhs_g_impl.value(), x_impl.value(), mult_impl.value());
                stop_impl = std::chrono::steady_clock::now();
                ns_impl += std::chrono::duration_cast<std::chrono::nanoseconds>(stop_impl - start_impl).count();
            } else if (idx == 1) {
                // explicit //
                // std::cout << "Running explicit solver..." << std::endl;
                start_expl = std::chrono::steady_clock::now();
                ret_expl = solver_expl.value().solve(info_expl.value(), jacobian_expl.value(), hessian_expl.value(), D_x_expl.value(), D_s_expl.value(), rhs_x_expl.value(), rhs_g_expl.value(), x_expl.value(), mult_expl.value());
                stop_expl = std::chrono::steady_clock::now();
                ns_expl += std::chrono::duration_cast<std::chrono::nanoseconds>(stop_expl - start_expl).count();
            } else if (idx == 2) {
                // accelerated
                // std::cout << "Running accelerated solver..." << std::endl;
                start_accel = std::chrono::steady_clock::now();
                ret_accel = solver_accel.value().solve(info_accel.value(), jacobian_accel.value(), hessian_accel.value(), D_x_accel.value(), D_s_accel.value(), rhs_x_accel.value(), rhs_g_accel.value(), x_accel.value(), mult_accel.value());
                stop_accel = std::chrono::steady_clock::now();
                ns_accel += std::chrono::duration_cast<std::chrono::nanoseconds>(stop_accel - start_accel).count();
                // std::cout << "acclerated solver returned with flag " << ret_accel << std::endl;
            } else {
                // reformulated //
                // std::cout << "Running reformulated solver..." << std::endl;
                start_reform = std::chrono::steady_clock::now();
                ret_reform = solver_reform.value().solve(info_reform.value(), jacobian_reform.value(), hessian_reform.value(), D_x_reform.value(), D_s_reform.value(), rhs_x_reform.value(), rhs_g_reform.value(), x_reform.value(), mult_reform.value());
                stop_reform = std::chrono::steady_clock::now();
                ns_reform += std::chrono::duration_cast<std::chrono::nanoseconds>(stop_reform - start_reform).count();
            }
            // std::cout << "Done." << std::endl;
        }
        } catch (const std::exception& e){
            std::cout << "Exception caught: " << e.what() << std::endl;
            nb_consecutive_failures++;
            continue;
        }
        }

        if (ret_expl != LinsolReturnFlag::SUCCESS || 
                ret_reform != LinsolReturnFlag::SUCCESS || 
                (ret_impl != LinsolReturnFlag::SUCCESS && !skip_impl) ||
                ret_accel != LinsolReturnFlag::SUCCESS){
            // std::cout << "\n" << ret_expl << " " << ret_reform << " " << ret_accel << std::endl;
            nb_consecutive_failures++;
            continue;
        }

        timings["total_ns_expl"] += ns_expl;
        timings["total_ns_reform"] += ns_reform;
        timings["total_ns_accel"] += ns_accel;
        long int ns_impl_preprocess = 0;
        long int ns_impl_solve = 0;
        long int ns_impl_postprocess = 0;

        long int ns_decomp_decomp = 0;
        long int ns_lu_impl = 0;
        if (!skip_impl){
            ns_impl_preprocess = solver_impl.value().duration_preprocess.count();
            ns_impl_solve = solver_impl.value().duration_solve.count();
            ns_impl_postprocess = solver_impl.value().duration_postprocess.count();
            timings["total_ns_impl"] += ns_impl;
            timings["total_ns_impl_preprocess"] += ns_impl_preprocess;
            timings["total_ns_impl_solve"] += ns_impl_solve;
            timings["total_ns_impl_postprocess"] += ns_impl_postprocess;
            timings["total_ns_impl_solve_inner"] += solver_impl.value().duration_inner_solve.count();
            
            timings["total_pre_jac"] += solver_impl.value().duration_preprocess_jac.count();
            timings["total_pre_hess"] += solver_impl.value().duration_preprocess_hess.count();
            timings["total_pre_reg"] += solver_impl.value().duration_preprocess_regularization.count();
            timings["total_pre_decomp"] += solver_impl.value().duration_preprocess_decomposition.count();
            timings["total_pre_info"] += solver_impl.value().duration_preprocess_info.count();
            timings["total_pre_rhs"] += solver_impl.value().duration_preprocess_modify_rhs.count();
            
            timings["total_decomp_copies"] += solver_impl.value().duration_decomp_copies.count();
            ns_decomp_decomp = solver_impl.value().duration_decomp_decomp.count();
            timings["total_decomp_decomp"] += ns_decomp_decomp;
            timings["total_decomp_scale1"] += solver_impl.value().duration_decomp_scale1.count();
            timings["total_decomp_scale2"] += solver_impl.value().duration_decomp_scale2.count();
            timings["total_decomp_permutation"] += solver_impl.value().duration_decomp_permutation.count();
            timings["total_decomp_store"] += solver_impl.value().duration_decomp_store.count();
            
            timings["total_solve_RSQrqt_copy"] += solver_impl.value().duration_RSQrqt_copy.count();
            timings["total_solve_FuFx_addition"] += solver_impl.value().duration_FuFx_addition.count();
            timings["total_solve_GuGx_addition"] += solver_impl.value().duration_GuGx_addition.count();
            timings["total_solve_GuGx_hat_addition"] += solver_impl.value().duration_GuGx_hat_addition.count();
            timings["total_solve_ukb_tilde_addition"] += solver_impl.value().duration_ukb_tilde_addition.count();
            timings["total_solve_lambdatilde_addition"] += solver_impl.value().duration_lambdatilde_addition.count();
            timings["total_solve_FuFx_addition_forward"] += solver_impl.value().duration_FuFx_addition_forward.count();
            
            ns_lu_impl = solver_impl.value().duration_lu_factorization.count() + solver_impl.value().duration_decomp_decomp.count();
            timings["ns_lu_impl"] = ns_lu_impl;
            timings["total_lu_impl"] += solver_impl.value().duration_lu_factorization.count() + solver_impl.value().duration_decomp_decomp.count();
            timings["total_lu_reform"] += solver_reform.value().duration_lu_factorization.count();
            timings["total_lu_accel"] += solver_accel.value().duration_lu_factorization.count();

            timings["total_impl_backward"] += solver_impl.value().duration_backward_recursion.count();
            timings["total_impl_initial"] += solver_impl.value().duration_initial_stage.count();
            timings["total_impl_forward"] += solver_impl.value().duration_forward_recursion.count();

            timings["total_post_rearrange_solution"] += solver_impl.value().duration_post_rearrange_solution.count();
            timings["total_post_scale_solution"] += solver_impl.value().duration_post_scale_solution.count();
            timings["total_post_reset_jacobian_pre"] += solver_impl.value().duration_post_reset_jacobian_pre.count();
            timings["total_post_reset_hessian_pre"] += solver_impl.value().duration_post_reset_hessian_pre.count();
            timings["total_post_regularization"] += solver_impl.value().duration_post_regularization.count();
        }
        
        long int ns_lu_expl = solver_expl.value().duration_lu_factorization.count();
        long int ns_lu_reform = solver_reform.value().duration_lu_factorization.count();
        long int ns_lu_accel = solver_accel.value().duration_lu_factorization.count();
        timings["ns_lu_expl"] = ns_lu_expl;
        timings["ns_lu_reform"] = ns_lu_reform;
        timings["ns_lu_accel"] = ns_lu_accel;
        

        timings["total_expl_backward"] += solver_expl.value().duration_backward_recursion.count();
        timings["total_reform_backward"] += solver_reform.value().duration_backward_recursion.count();
        timings["total_accel_backward"] += solver_accel.value().duration_backward_recursion.count();
        timings["total_expl_initial"] += solver_expl.value().duration_initial_stage.count();
        timings["total_reform_initial"] += solver_reform.value().duration_initial_stage.count();
        timings["total_accel_initial"] += solver_accel.value().duration_initial_stage.count();
        timings["total_expl_forward"] += solver_expl.value().duration_forward_recursion.count();
        timings["total_reform_forward"] += solver_reform.value().duration_forward_recursion.count();
        timings["total_accel_forward"] += solver_accel.value().duration_forward_recursion.count();

        nb_runs_completed++;
        nb_consecutive_failures = 0;

        if (write_csv){
            // parameters
            f << K << "," << nu[0] << "," << nx[0] << "," << r[0] << "," << ng[0] << "," << ng_ineq[0] << ",";
            if (skip_impl){ f << nz[0] << "," << nv[0] << ",";}

            // timings expl
            f << ns_expl << "," << solver_expl.value().duration_backward_recursion.count() << ",";
            f << solver_expl.value().duration_initial_stage.count() << ",";
            f << solver_expl.value().duration_forward_recursion.count() << ",";

            // timings impl
            if (!skip_impl){
                f << ns_impl << "," << solver_impl.value().duration_backward_recursion.count() << ",";
                f << solver_impl.value().duration_initial_stage.count() << ",";
                f << solver_impl.value().duration_forward_recursion.count() << ",";
                f << ns_impl_preprocess << "," << ns_impl_solve << "," << ns_impl_postprocess << ",";
            }

            // timings reform
            f << ns_reform << "," << solver_reform.value().duration_backward_recursion.count() << ",";
            f << solver_reform.value().duration_initial_stage.count() << ",";
            f << solver_reform.value().duration_forward_recursion.count() << ",";

            // timings accel
            f << ns_accel << "," << solver_accel.value().duration_backward_recursion.count() << ",";
            f << solver_accel.value().duration_initial_stage.count() << ",";
            f << solver_accel.value().duration_forward_recursion.count() << ",";

            // timings lu
            f << ns_lu_expl << ",";
            if (!skip_impl){ f << ns_lu_impl << ",";}
            f << ns_lu_reform << "," << ns_lu_accel;
            if (!skip_impl){ f << "," << ns_decomp_decomp;}
            f << "\n";
        }       
    }

    std::cout << "Average time explicit:     " << timings["total_ns_expl"] / nb_runs_completed << " ns (";
    std::cout << timings["total_expl_backward"] / nb_runs_completed << " - ";
    std::cout << timings["total_expl_initial"] / nb_runs_completed << " - ";
    std::cout << timings["total_expl_forward"] / nb_runs_completed << ")" << std::endl;
    std::cout << "Average time reformulated: " << timings["total_ns_reform"] / nb_runs_completed << " ns (";
    std::cout << timings["total_reform_backward"] / nb_runs_completed << " - ";
    std::cout << timings["total_reform_initial"] / nb_runs_completed << " - ";
    std::cout << timings["total_reform_forward"] / nb_runs_completed << ")" << std::endl;
    std::cout << "Average time accelerated:  " << timings["total_ns_accel"] / nb_runs_completed << " ns (";
    std::cout << timings["total_accel_backward"] / nb_runs_completed << " - ";
    std::cout << timings["total_accel_initial"] / nb_runs_completed << " - ";
    std::cout << timings["total_accel_forward"] / nb_runs_completed << ")" << std::endl;
    if (!skip_impl){
        std::cout << "Average time implicit:     " << timings["total_ns_impl"] / nb_runs_completed << " ns (";
        std::cout << timings["total_impl_backward"] / nb_runs_completed << " - ";
        std::cout << timings["total_impl_initial"] / nb_runs_completed << " - ";
        std::cout << timings["total_impl_forward"] / nb_runs_completed << ")  (";
        std::cout << timings["total_ns_impl_preprocess"] / nb_runs_completed << " - ";
        std::cout << timings["total_ns_impl_solve"] / nb_runs_completed << " - ";
        std::cout << timings["total_ns_impl_postprocess"] / nb_runs_completed << ")" << std::endl;
    }
    
    std::map<std::string, std::vector<std::string>> breakdowns = {
        {"LU factorization", {"total_lu_impl", "total_lu_reform", "total_lu_accel"}},
        {"implicit preprocess", {"total_pre_jac", "total_pre_hess", "total_pre_reg", "total_pre_decomp", "total_pre_info", "total_pre_rhs"}},
        {"implicit preprocess decomp", {"total_decomp_copies", "total_decomp_decomp", "total_decomp_scale1", "total_decomp_scale2", "total_decomp_permutation", "total_decomp_store"}},
        {"implicit solve modifications", {"total_solve_RSQrqt_copy", "total_solve_FuFx_addition", "total_solve_GuGx_addition", "total_solve_GuGx_hat_addition", "total_solve_ukb_tilde_addition", "total_solve_lambdatilde_addition", "total_solve_FuFx_addition_forward"}},
        {"implicit postprocess", {"total_post_rearrange_solution", "total_post_scale_solution", "total_post_reset_jacobian_pre", "total_post_reset_hessian_pre", "total_post_regularization"}}
    };
    std::map<std::string, std::vector<std::string>> breakdown_totals = {
        {"LU factorization", {"total_ns_impl", "total_ns_reform", "total_ns_accel"}},
        {"implicit preprocess", {"total_ns_impl_preprocess"}},
        {"implicit preprocess decomp", {"total_pre_decomp"}},
        {"implicit solve modifications", {"total_ns_impl_solve_inner"}},
        {"implicit postprocess", {"total_ns_impl_postprocess"}}
    };

    std::cout << "Time spent in LU factorization: " << std::endl;
    if (!skip_impl){
        PrintLine("Implicit", timings["total_lu_impl"], timings["total_ns_impl"]);
    }
    PrintLine("Reformulated", timings["total_lu_reform"], timings["total_ns_reform"]);
    PrintLine("Accelerated", timings["total_lu_accel"], timings["total_ns_accel"]);
    
    if (!skip_impl){
        PrintBreakdown(timings, "implicit preprocess", breakdowns, breakdown_totals);
        PrintBreakdown(timings, "implicit preprocess decomp", breakdowns, breakdown_totals);
        PrintBreakdown(timings, "implicit solve modifications", breakdowns, breakdown_totals);
        PrintBreakdown(timings, "implicit postprocess", breakdowns, breakdown_totals);

        // sort keys of timings by decreasing value
        std::vector<std::string> keys_to_sort = breakdowns["implicit preprocess"];
        keys_to_sort.insert(keys_to_sort.end(), 
                            breakdowns["implicit preprocess decomp"].begin(), 
                            breakdowns["implicit preprocess decomp"].end());
        keys_to_sort.insert(keys_to_sort.end(), 
                            breakdowns["implicit solve modifications"].begin(), 
                            breakdowns["implicit solve modifications"].end());
        keys_to_sort.insert(keys_to_sort.end(), 
                            breakdowns["implicit postprocess"].begin(), 
                            breakdowns["implicit postprocess"].end());
        std::vector<std::pair<std::string, long int>> sorted_timings(timings.begin(), timings.end());
        std::sort(sorted_timings.begin(), sorted_timings.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        std::cout << "\nSorted time components:" << std::endl;
        int i = 0;
        int nb_printed = 0;
        while (nb_printed < 100 && i < sorted_timings.size()) {
            if (std::find(keys_to_sort.begin(), keys_to_sort.end(), sorted_timings[i].first) != keys_to_sort.end()){
                PrintLine(sorted_timings[i].first, sorted_timings[i].second, timings["total_ns_impl"], true);
                nb_printed++;
            }
            i++;
        }   

        long int jacobian_pre_post = 
            timings["total_pre_jac"] + timings["total_post_reset_jacobian_pre"];
        long int hessian_pre_post =
            timings["total_pre_hess"] + timings["total_post_reset_hessian_pre"];
        std::cout << "\nJacobian and Hessian pre- and postprocessing overhead:" << std::endl;
        PrintLine("jacobian", jacobian_pre_post, timings["total_ns_impl"], true);
        PrintLine("hessian", hessian_pre_post, timings["total_ns_impl"], true);
    }
}