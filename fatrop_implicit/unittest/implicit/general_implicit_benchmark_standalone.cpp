#include "../random_matrix.hpp"
#include "fatrop/context/context.hpp"
#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/ocp/aug_system_solver.hpp"
#include "fatrop/ocp/dims.hpp" // inherit
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/problem_info.hpp" //inherit
#include "fatrop/ocp/type.hpp"
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <random>
#include <fstream>
#include <algorithm>
#include <array>
#include <optional>

static const bool USE_GENERALIZATION = true;

using namespace fatrop;

class BenchmarkManager
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
    std::optional<Jacobian<AcceleratedOcpType>> jacobian_accel;
    std::optional<Hessian<AcceleratedOcpType>> hessian_accel;
    // std::optional<Jacobian<OcpType>> jacobian_accel;
    // std::optional<Hessian<OcpType>> hessian_accel;
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
    std::optional<AugSystemSolver<AcceleratedOcpType>> solver_accel;
    // std::optional<AugSystemSolver<OcpType>> solver_accel;

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
        AllocateReformulatedSolver();
        AllocateAcceleratedSolver();
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
        jacobian_accel.emplace(Jacobian<AcceleratedOcpType>(dims_accel.value()));
        // jacobian_accel.emplace(Jacobian<OcpType>(dims_accel.value()));
        full_matrix_jacobian_accel.emplace(info_accel->number_of_eq_constraints, info_accel->number_of_primal_variables);
        hessian_accel.emplace(Hessian<AcceleratedOcpType>(dims_accel.value()));
        // hessian_accel.emplace(Hessian<OcpType>(dims_accel.value()));
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
        solver_accel.emplace(AugSystemSolver<AcceleratedOcpType>(info_accel.value()));
        // solver_accel.emplace(AugSystemSolver<OcpType>(info_accel.value()));
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

        // pass on options
        if (use_generalization){
            solver_accel.value().set_nb_of_dynamics_constraints(nv[0]);
            solver_accel.value().set_nb_of_zk_vars(nz[0]);
        }
    }

    void Randomize(){
        // std::cout << "randomizing dimensions" << std::endl;
        int nb_tries = GetRandomGeneralizedDimensions();
        // std::cout << "nb_tries: " << nb_tries << std::endl;

        // std::cout << "allocating solvers" << std::endl;
        AllocateSolvers();

        // std::cout << "filling solvers" << std::endl;
        // std::cout << "\texplicit:" << std::endl;
        FillExplicitSolver();
        // std::cout << "\treformulated:" << std::endl;
        FillReformulatedSolver();
        // std::cout << "\taccelerated:" << std::endl;
        FillAcceleratedSolver();
        // std::cout << "done" << std::endl;

    }
};

int main()
{
    BenchmarkManager bm = BenchmarkManager();
    std::cout << "running test" << std::endl;
    int nb_runs = 20000;
    int nb_runs_completed = 0;
    bool write_csv = true;

    bool skip_impl = USE_GENERALIZATION;

    std::map<std::string, long int> timings = {
        {"total_ns_expl", 0},
        {"total_ns_reform", 0},
        {"total_ns_accel", 0},
        {"total_lu_reform", 0},
        {"total_ns_accel", 0},
        {"total_expl_backward", 0},
        {"total_reform_backward", 0},
        {"total_accel_backward", 0},
        {"total_expl_initial", 0},
        {"total_reform_initial", 0},
        {"total_accel_initial", 0},
        {"total_expl_forward", 0},
        {"total_reform_forward", 0},
        {"total_accel_forward", 0},
    };

    std::ofstream f;    
    if (write_csv){
        f.open("random_benchmark_results_generalized_20000_standalone.csv");
        f << "K,nu,nx,r,ng,ng_ineq,";
        f << "nz,nv,";
        f << "t_expl,t_expl_backward,t_expl_solve,t_expl_forward,";
        f << "t_reform,t_reform_backward,t_reform_solve,t_reform_forward,";
        f << "t_accel,t_accel_backward,t_accel_solve,t_accel_forward,";
        f << "lu_expl,";
        f << "lu_reform,lu_accel";
        f << "\n";
    }
    
    int nb_consecutive_failures = 0;
    std::array<int, 3> order = {0, 1, 2};

    auto start_expl = std::chrono::steady_clock::now();
    auto stop_expl = std::chrono::steady_clock::now();
    auto start_reform = std::chrono::steady_clock::now();
    auto stop_reform = std::chrono::steady_clock::now();
    auto start_accel = std::chrono::steady_clock::now();
    auto stop_accel = std::chrono::steady_clock::now();
    Index ret_expl, ret_reform, ret_accel;
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
        bm.Randomize();

        int nb_averaging_runs = 10;
        long int ns_expl = 0;
        long int ns_reform = 0;
        long int ns_accel = 0;

        for (int averaging_run_counter = 0; averaging_run_counter < nb_averaging_runs; ++averaging_run_counter){
        std::shuffle(order.begin(), order.end(), rng);

        try{
        // std::cout << std::endl;
        for (int idx : order){
             if (idx == 0) {
                // explicit //
                // std::cout << "Running explicit solver..." << std::endl;
                start_expl = std::chrono::steady_clock::now();
                ret_expl = bm.solver_expl.value().solve(bm.info_expl.value(), bm.jacobian_expl.value(), bm.hessian_expl.value(), bm.D_x_expl.value(), bm.D_s_expl.value(), bm.rhs_x_expl.value(), bm.rhs_g_expl.value(), bm.x_expl.value(), bm.mult_expl.value());
                stop_expl = std::chrono::steady_clock::now();
                ns_expl += std::chrono::duration_cast<std::chrono::nanoseconds>(stop_expl - start_expl).count();
            } else if (idx == 1) {
                // accelerated
                // std::cout << "Running accelerated solver..." << std::endl;
                start_accel = std::chrono::steady_clock::now();
                ret_accel = bm.solver_accel.value().solve(bm.info_accel.value(), bm.jacobian_accel.value(), bm.hessian_accel.value(), bm.D_x_accel.value(), bm.D_s_accel.value(), bm.rhs_x_accel.value(), bm.rhs_g_accel.value(), bm.x_accel.value(), bm.mult_accel.value());
                stop_accel = std::chrono::steady_clock::now();
                ns_accel += std::chrono::duration_cast<std::chrono::nanoseconds>(stop_accel - start_accel).count();
                // std::cout << "acclerated solver returned with flag " << ret_accel << std::endl;
            } else {
                // reformulated //
                // std::cout << "Running reformulated solver..." << std::endl;
                start_reform = std::chrono::steady_clock::now();
                ret_reform = bm.solver_reform.value().solve(bm.info_reform.value(), bm.jacobian_reform.value(), bm.hessian_reform.value(), bm.D_x_reform.value(), bm.D_s_reform.value(), bm.rhs_x_reform.value(), bm.rhs_g_reform.value(), bm.x_reform.value(), bm.mult_reform.value());
                stop_reform = std::chrono::steady_clock::now();
                ns_reform += std::chrono::duration_cast<std::chrono::nanoseconds>(stop_reform - start_reform).count();
            }
            // std::cout << "Done." << std::endl;
        }
        // std::cout << "Ran all solvers." << std::endl;
        } catch (const std::exception& e){
            std::cout << "Exception caught: " << e.what() << std::endl;
            nb_consecutive_failures++;
            continue;
        }
        }

        if (ret_expl != LinsolReturnFlag::SUCCESS || 
                ret_reform != LinsolReturnFlag::SUCCESS || 
                ret_accel != LinsolReturnFlag::SUCCESS){
            // std::cout << "\n" << ret_expl << " " << ret_reform << " " << ret_accel << std::endl;
            nb_consecutive_failures++;
            continue;
        }

        // skip first few runs
        if (nb_runs_completed < 10){
            nb_runs_completed++;
            continue;
        }

        timings["total_ns_expl"] += ns_expl;
        timings["total_ns_reform"] += ns_reform;
        timings["total_ns_accel"] += ns_accel;
        
        long int ns_lu_expl = bm.solver_expl.value().duration_lu_factorization.count();
        long int ns_lu_reform = bm.solver_reform.value().duration_lu_factorization.count();
        long int ns_lu_accel = bm.solver_accel.value().duration_lu_factorization.count();
        timings["ns_lu_expl"] = ns_lu_expl;
        timings["ns_lu_reform"] = ns_lu_reform;
        timings["ns_lu_accel"] = ns_lu_accel;
        

        timings["total_expl_backward"] += bm.solver_expl.value().duration_backward_recursion.count();
        timings["total_reform_backward"] += bm.solver_reform.value().duration_backward_recursion.count();
        timings["total_accel_backward"] += bm.solver_accel.value().duration_backward_recursion.count();
        timings["total_expl_initial"] += bm.solver_expl.value().duration_initial_stage.count();
        timings["total_reform_initial"] += bm.solver_reform.value().duration_initial_stage.count();
        timings["total_accel_initial"] += bm.solver_accel.value().duration_initial_stage.count();
        timings["total_expl_forward"] += bm.solver_expl.value().duration_forward_recursion.count();
        timings["total_reform_forward"] += bm.solver_reform.value().duration_forward_recursion.count();
        timings["total_accel_forward"] += bm.solver_accel.value().duration_forward_recursion.count();

        nb_runs_completed++;
        nb_consecutive_failures = 0;

        if (write_csv){
            // parameters
            f << bm.K << "," << bm.nu[0] << "," << bm.nx[0] << "," << bm.r[0] << "," << bm.ng[0] << "," << bm.ng_ineq[0] << ",";
            f << bm.nz[0] << "," << bm.nv[0] << ",";

            // timings expl
            f << ns_expl << "," << bm.solver_expl.value().duration_backward_recursion.count() << ",";
            f << bm.solver_expl.value().duration_initial_stage.count() << ",";
            f << bm.solver_expl.value().duration_forward_recursion.count() << ",";

            // timings reform
            f << ns_reform << "," << bm.solver_reform.value().duration_backward_recursion.count() << ",";
            f << bm.solver_reform.value().duration_initial_stage.count() << ",";
            f << bm.solver_reform.value().duration_forward_recursion.count() << ",";

            // timings accel
            f << ns_accel << "," << bm.solver_accel.value().duration_backward_recursion.count() << ",";
            f << bm.solver_accel.value().duration_initial_stage.count() << ",";
            f << bm.solver_accel.value().duration_forward_recursion.count() << ",";

            // timings lu
            f << ns_lu_expl << ",";
            f << ns_lu_reform << "," << ns_lu_accel;
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
}
