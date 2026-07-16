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
#include <optional>

using namespace fatrop;

class RandomVisualizer
{
// protected:
public:
    bool full_rank = true;
    bool constant_dimensions = true;

    int K;
    std::vector<Index> nx;
    std::vector<Index> r;
    std::vector<Index> nu;
    std::vector<Index> ng;
    std::vector<Index> ng_ineq;

    /// Explicit case ///
    std::optional<ProblemDims> dims_expl;
    std::optional<ProblemInfo> info_expl;
    std::optional<Jacobian<ImplicitOcpType>> jacobian_expl;
    std::optional<Hessian<ImplicitOcpType>> hessian_expl;
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
    std::optional<AugSystemSolver<ImplicitOcpType>> solver_expl;

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
    std::optional<Jacobian<ImplicitOcpType>> jacobian_reform;
    std::optional<Hessian<ImplicitOcpType>> hessian_reform;
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
    std::optional<AugSystemSolver<ImplicitOcpType>> solver_reform;

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
    }

    void SetDimensions(int K_val, int nx_val, int nu_val, int ng_val, int ng_ineq_val)
    {
        ClearOptionals();
        K = K_val;
        nx = std::vector<Index>(K, nx_val);
        r = RandomVector(K, 0, nx_val);
        nu = std::vector<Index>(K, nu_val);
        ng = std::vector<Index>(K, ng_val);
        nx = std::vector<Index>(K, nx_val);
        ng_ineq = std::vector<Index>(K, ng_ineq_val);
        std::cout << "nu.size() == " << nu.size() << std::endl;
        std::cout << "nx.size() == " << nx.size() << std::endl;
        std::cout << "ng.size() == " << ng.size() << std::endl;
        std::cout << "ng_ineq.size() == " << ng_ineq.size() << std::endl;
        std::cout << "K = " << K << std::endl;
    }

    void AllocateSolvers(){
        AllocateExplicitSolver();
        AllocateImplicitSolver();
        AllocateReformulatedSolver();
    }

    void AllocateExplicitSolver(){
        dims_expl.emplace(ProblemDims{K, nu, nx, ng, ng_ineq});
        info_expl.emplace(ProblemInfo(dims_expl.value()));
        jacobian_expl.emplace(Jacobian<ImplicitOcpType>(dims_expl.value()));
        full_matrix_jacobian_expl =
            MatRealAllocated(info_expl->number_of_eq_constraints, info_expl->number_of_primal_variables);
        hessian_expl.emplace(Hessian<ImplicitOcpType>(dims_expl.value()));
        full_matrix_hessian_expl =
            MatRealAllocated(info_expl->number_of_primal_variables, info_expl->number_of_primal_variables);
        x_expl = VecRealAllocated(info_expl->number_of_primal_variables);
        mult_expl = VecRealAllocated(info_expl->number_of_eq_constraints);
        rhs_x_expl = VecRealAllocated(info_expl->number_of_primal_variables);
        rhs_g_expl = VecRealAllocated(info_expl->number_of_eq_constraints);
        D_x_expl = VecRealAllocated(info_expl->number_of_primal_variables);
        D_s_expl = VecRealAllocated(info_expl->number_of_slack_variables);
        D_eq_expl = VecRealAllocated(info_expl->number_of_g_eq_path);
        full_kkt_matrix_expl =
            MatRealAllocated(info_expl->number_of_primal_variables + info_expl->number_of_eq_constraints,
                             info_expl->number_of_primal_variables + info_expl->number_of_eq_constraints);
        solver_expl.emplace(AugSystemSolver<ImplicitOcpType>(info_expl.value()));
    }

    void AllocateImplicitSolver(){
        dims_impl.emplace(ProblemDims{K, nu, nx, ng, ng_ineq});
        info_impl.emplace(ProblemInfo(dims_impl.value()));
        jacobian_impl.emplace(Jacobian<ImplicitOcpType>(dims_impl.value()));
        full_matrix_jacobian_impl =
            MatRealAllocated(info_impl->number_of_eq_constraints, info_impl->number_of_primal_variables);
        hessian_impl.emplace(Hessian<ImplicitOcpType>(dims_impl.value()));
        full_matrix_hessian_impl =
            MatRealAllocated(info_impl->number_of_primal_variables, info_impl->number_of_primal_variables);
        x_impl = VecRealAllocated(info_impl->number_of_primal_variables);
        mult_impl = VecRealAllocated(info_impl->number_of_eq_constraints);
        rhs_x_impl = VecRealAllocated(info_impl->number_of_primal_variables);
        rhs_g_impl = VecRealAllocated(info_impl->number_of_eq_constraints);
        D_x_impl = VecRealAllocated(info_impl->number_of_primal_variables);
        D_s_impl = VecRealAllocated(info_impl->number_of_slack_variables);
        D_eq_impl = VecRealAllocated(info_impl->number_of_g_eq_path);
        full_kkt_matrix_impl =
            MatRealAllocated(info_impl->number_of_primal_variables + info_impl->number_of_eq_constraints,
                             info_impl->number_of_primal_variables + info_impl->number_of_eq_constraints);
        solver_impl.emplace(AugSystemSolver<ImplicitOcpType>(info_impl.value()));
    }

    void AllocateReformulatedSolver(){
        std::vector<Index> nu_reform = nu;
        std::vector<Index> ng_reform = ng;
        for (int k = 0; k < K-1; ++k){
            nu_reform[k] += nx[k+1];
            ng_reform[k] += nx[k+1];
        }

        dims_reform.emplace(ProblemDims{K, nu_reform, nx, ng_reform, ng_ineq});
        info_reform.emplace(ProblemInfo(dims_reform.value()));
        jacobian_reform.emplace(Jacobian<ImplicitOcpType>(dims_reform.value()));
        full_matrix_jacobian_reform =
            MatRealAllocated(info_reform->number_of_eq_constraints, info_reform->number_of_primal_variables);
        hessian_reform.emplace(Hessian<ImplicitOcpType>(dims_reform.value()));
        full_matrix_hessian_reform =
            MatRealAllocated(info_reform->number_of_primal_variables, info_reform->number_of_primal_variables);
        x_reform = VecRealAllocated(info_reform->number_of_primal_variables);
        mult_reform = VecRealAllocated(info_reform->number_of_eq_constraints);
        rhs_x_reform = VecRealAllocated(info_reform->number_of_primal_variables);
        rhs_g_reform = VecRealAllocated(info_reform->number_of_eq_constraints);
        D_x_reform = VecRealAllocated(info_reform->number_of_primal_variables);
        D_s_reform = VecRealAllocated(info_reform->number_of_slack_variables);
        D_eq_reform = VecRealAllocated(info_reform->number_of_g_eq_path);
        full_kkt_matrix_reform =
            MatRealAllocated(info_reform->number_of_primal_variables + info_reform->number_of_eq_constraints,
                             info_reform->number_of_primal_variables + info_reform->number_of_eq_constraints);
        solver_reform.emplace(AugSystemSolver<ImplicitOcpType>(info_reform.value()));
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

                jacobian_expl.value().Jt[k].block(nx_next, nx_next, 0, 0) =
                ::test::identity_matrix(nx_next, -1);
                full_matrix_jacobian_expl.value().block(nx_next, nx_next, offs_eq_dyn, offs_x_next) = 
                    transpose(jacobian_expl.value().Jt[k]);

                hessian_expl.value().FuFx[k].block(nx + nu, nx_next, 0, 0) =
                    ::test::empty_matrix(nx + nu, nx_next);
                full_matrix_hessian_expl.value().block(nx_next, nu + nx, offs_x_next, offs_ux) = 
                    transpose(hessian_expl.value().FuFx[k]);
                full_matrix_hessian_expl.value().block(nu + nx, nx_next, offs_ux, offs_x_next) =
                    hessian_expl.value().FuFx[k];
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
            rhs_x_expl.value()(i) = 1.0 * i; D_x_expl.value()(i) = 1.0 * (i + 0.1);
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
                
                // jacobian_impl.value().Jt[k].block(nx_next, nx_next, 0, 0) =
                // ::test::identity_matrix(nx_next, -1);
                // full_matrix_jacobian_impl.value().block(nx_next, nx_next, offs_eq_dyn, offs_x_next) = 
                //     transpose(jacobian_impl.value().Jt[k]);

                // hessian_impl.value().FuFx[k].block(nx + nu, nx_next, 0, 0) =
                //     ::test::empty_matrix(nx + nu, nx_next);
                // full_matrix_hessian_impl.value().block(nx_next, nu + nx, offs_x_next, offs_ux) = 
                //     transpose(hessian_impl.value().FuFx[k]);
                // full_matrix_hessian_impl.value().block(nu + nx, nx_next, offs_ux, offs_x_next) =
                //     hessian_impl.value().FuFx[k];
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
            rhs_x_impl.value()(i) = 1.0 * i; D_x_impl.value()(i) = 1.0 * (i + 0.1);
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
            Index nu_true = (k < info_reform.value().dims.K - 1) ? nu - info_reform.value().dims.number_of_states[k+1] : nu;
            Index ng_true = (k < info_reform.value().dims.K - 1) ? ng - info_reform.value().dims.number_of_states[k+1] : ng;
            if (k < info_reform.value().dims.K - 1)
            {
                Index nx_next = info_reform.value().dims.number_of_states[k + 1];
                Index offs_x_next = info_reform.value().offsets_primal_x[k + 1];
                // jacobian_reform.value().BAbt[k].block(nx_next, nx_next, nu_true, 0) =
                //     ::test::identity_matrix(nx_next, 1);
                jacobian_reform.value().BAbt[k].block(nx_next, nx_next, 0, 0) =
                    ::test::identity_matrix(nx_next, 1);
                full_matrix_jacobian_reform.value().block(nx_next, nu + nx, offs_eq_dyn, offs_ux) =
                    transpose(jacobian_reform.value().BAbt[k].block(nu + nx, nx_next, 0, 0));

                jacobian_reform.value().Jt[k].block(nx_next, nx_next, 0, 0) =
                ::test::identity_matrix(nx_next, -1);
                full_matrix_jacobian_reform.value().block(nx_next, nx_next, offs_eq_dyn, offs_x_next) = 
                    transpose(jacobian_reform.value().Jt[k]);

                hessian_reform.value().FuFx[k].block(nx + nu, nx_next, 0, 0) =
                    ::test::empty_matrix(nx + nu, nx_next);
                full_matrix_hessian_reform.value().block(nx_next, nu + nx, offs_x_next, offs_ux) = 
                    transpose(hessian_reform.value().FuFx[k]);
                full_matrix_hessian_reform.value().block(nu + nx, nx_next, offs_ux, offs_x_next) =
                    hessian_reform.value().FuFx[k];
            }
            jacobian_reform.value().Gg_eqt[k].block(nu + nx, info_reform.value().dims.number_of_eq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info_reform.value().dims.number_of_eq_constraints[k]);
            // jacobian_reform.value().Gg_eqt[k].block(nu - nu_true, ng_true, nu_true, 0) =
            //     ::test::empty_matrix(nu - nu_true, ng_true);
            jacobian_reform.value().Gg_eqt[k].block(nu - nu_true, ng_true, 0, 0) =
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
            hessian_reform.value().RSQrqt[k].block(nu - nu_true, nu + nx, 0, 0) = ::test::empty_matrix(nu_true, nu + nx);
            hessian_reform.value().RSQrqt[k].block(nu + nx, nu - nu_true, 0, 0) = ::test::empty_matrix(nu + nx, nu_true);
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
            rhs_x_reform.value()(i) = 1.0 * i; D_x_reform.value()(i) = 1.0 * (i + 0.1);
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
        for (int k = 0; k < info_reform.value().dims.K - 1; ++k){
            Index nuk = info_reform.value().dims.number_of_controls[k];
            Index nxk = info_reform.value().dims.number_of_states[k+1];
            Index ng = info_reform.value().dims.number_of_eq_constraints[k];
            Index nu_true = nuk - nxk;
            Index ng_true = ng - nxk;
            for (Index i = nu_true; i < nuk; ++i){
                rhs_x_reform.value()(info_reform.value().offsets_primal_u[k] + i) = 0;
                D_x_reform.value()(info_reform.value().offsets_primal_u[k] + i) = 0;
                rhs_g_reform.value()(info_reform.value().offsets_g_eq_path[k] + ng_true) = 0;
            }
        }   
    }

    void Randomize(int K_val, int nx_val, int nu_val, int ng_val, int ng_ineq_val){
        SetDimensions(K_val, nx_val, nu_val, ng_val, ng_ineq_val);
        AllocateSolvers();
        FillExplicitSolver();
        FillImplicitSolver();
        FillReformulatedSolver();
    }
};



int main(){
    RandomVisualizer v;
    bool success = false;

    while (!success){
        std::cout << "randomizing" << std::endl;
        v.Randomize(4, 4, 4, 3, 0);
        std::cout << "randomized!" << std::endl;
        v.solver_expl.value().set_factorization_file_name("factorization_info_expl.py");
        v.solver_expl.value().set_preprocessing_file_name("preprocess_info_expl.py");
        v.solver_reform.value().set_factorization_file_name("factorization_info_reform.py");
        v.solver_reform.value().set_preprocessing_file_name("preprocess_info_reform.py");
        v.solver_impl.value().set_factorization_file_name("factorization_info_impl.py");
        v.solver_impl.value().set_preprocessing_file_name("preprocess_info_impl.py");
        std::cout << "here" << std::endl;

        // explicit //
        std::cout << __LINE__ << std::endl;
        Index ret_expl = v.solver_expl.value().solve(
            v.info_expl.value(), v.jacobian_expl.value(), v.hessian_expl.value(), 
            v.D_x_expl.value(), v.D_s_expl.value(), v.rhs_x_expl.value(), 
            v.rhs_g_expl.value(), v.x_expl.value(), v.mult_expl.value());

        // reformulated //
        std::cout << __LINE__ << std::endl;
        Index ret_reform = v.solver_reform.value().solve(
            v.info_reform.value(), v.jacobian_reform.value(), v.hessian_reform.value(), 
            v.D_x_reform.value(), v.D_s_reform.value(), v.rhs_x_reform.value(), 
            v.rhs_g_reform.value(), v.x_reform.value(), v.mult_reform.value());

        // implicit //
        std::cout << __LINE__ << std::endl;
        std::cout << v.full_kkt_matrix_impl.value() << std::endl;
        Index ret_impl = 0;
        try{
            std::cout << __LINE__ << std::endl;
        Index ret_impl = v.solver_impl.value().solve(
            v.info_impl.value(), v.jacobian_impl.value(), v.hessian_impl.value(), 
            v.D_x_impl.value(), v.D_s_impl.value(), v.rhs_x_impl.value(), 
            v.rhs_g_impl.value(), v.x_impl.value(), v.mult_impl.value());
        } catch (std::exception &e){
            std::cout << "Exception caught during solve!" << std::endl;
            continue;
        }
        std::cout << __LINE__ << std::endl;

        if (ret_expl != LinsolReturnFlag::SUCCESS || 
                ret_reform != LinsolReturnFlag::SUCCESS || 
                ret_impl != LinsolReturnFlag::SUCCESS){
            continue;
        }
        std::cout << __LINE__ << std::endl;
        success = true;        
    }
    std::cout << __LINE__ << std::endl;
}