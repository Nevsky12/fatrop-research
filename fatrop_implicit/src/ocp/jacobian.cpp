//
// Copyright (c) Lander Vanroye, KU Leuven
//
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/common/exception.hpp"
#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/ocp/dims.hpp"
#include "fatrop/ocp/problem_info.hpp"

#include <chrono>
#include <algorithm>
#include <numeric>

using namespace fatrop;

Jacobian<OcpType>::Jacobian(const ProblemDims &dims)
    : global_parameter_jacobian(
          std::accumulate(dims.number_of_states.begin() + 1,
                          dims.number_of_states.end(), 0)
        + std::accumulate(dims.number_of_eq_constraints.begin(),
                          dims.number_of_eq_constraints.end(), 0)
        + std::accumulate(dims.number_of_ineq_constraints.begin(),
                          dims.number_of_ineq_constraints.end(), 0),
          std::max<Index>(dims.number_of_global_parameters, 1))
{
    // reserve memory for the Jacobian matrices
    BAbt.reserve(dims.K - 1);
    Gg_eqt.reserve(dims.K);
    Gg_ineqt.reserve(dims.K);
    // allocate memory for the Jacobian matrices
    for (int k = 0; k < dims.K - 1; ++k)
        BAbt.emplace_back(dims.number_of_states[k] + dims.number_of_controls[k] + 1,
                          dims.number_of_states[k + 1]);
    for (int k = 0; k < dims.K; ++k)
    {
        Gg_eqt.emplace_back(dims.number_of_states[k] + dims.number_of_controls[k] + 1,
                            dims.number_of_eq_constraints[k]);
    }
    for (int k = 0; k < dims.K; ++k)
        Gg_ineqt.emplace_back(dims.number_of_states[k] + dims.number_of_controls[k] + 1,
                              dims.number_of_ineq_constraints[k]);
};
void Jacobian<OcpType>::apply_on_right(const OcpInfo &info, const VecRealView &x, Scalar alpha,
                                       const VecRealView &y, VecRealView &out) const
{
    out = alpha * y;
    // dynamics constraints BA*ux - x_next
    for (Index k = 0; k < info.dims.K - 1; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index nx_next = info.dims.number_of_states[k + 1];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_x_next = info.offsets_primal_x[k + 1];
        Index offset_dyn_eq = info.offsets_g_eq_dyn[k];
        // apply out[offs:offs+nx] =  BAbt.T @ x[offs:offs+nu+nx] - x_next[offs:offs+nx]
        gemv_t(nu + nx, nx_next, 1.0, BAbt[k], 0, 0, x, offset_ux, 1.0, out, offset_dyn_eq, out,
               offset_dyn_eq);
        axpy(nx_next, -1.0, x, offset_x_next, out, offset_dyn_eq, out, offset_dyn_eq);
    }
    // equality path constraints
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng = info.dims.number_of_eq_constraints[k];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_g_eq = info.offsets_g_eq_path[k];
        // apply out[offs:offs+ng] =  Gg_eqt.T @ x[offs:offs+nu+nx]
        gemv_t(nu + nx, ng, 1.0, Gg_eqt[k], 0, 0, x, offset_ux, 1.0, out, offset_g_eq, out,
               offset_g_eq);
    }
    // slack equality path constraints Gg_ineqt @ x
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_g_ineq = info.offsets_g_eq_slack[k];
        // apply out[offs:offs+ng_ineq] =  Gg_ineqt.T @ x[offs:offs+nu+nx]
        gemv_t(nu + nx, ng_ineq, 1.0, Gg_ineqt[k], 0, 0, x, offset_ux, 1.0, out, offset_g_ineq, out,
               offset_g_ineq);
    }
    if (info.number_of_global_parameters > 0)
        gemv_n(info.number_of_eq_constraints,
               info.number_of_global_parameters, 1.0,
               global_parameter_jacobian, 0, 0, x,
               info.offset_primal_global, 1.0, out, 0, out, 0);
};
void Jacobian<OcpType>::transpose_apply_on_right(const OcpInfo &info, const VecRealView &mult_eq,
                                                 Scalar alpha, const VecRealView &y,
                                                 VecRealView &out) const
{
    // set the output to zero, we will add the contributions
    // dynamics constraints'contributions [BA.T ; 0; -I] @ mult_eq
    out = alpha * y;
    for (Index k = 0; k < info.dims.K - 1; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index offs_ux = info.offsets_primal_u[k];
        Index offs_x_next = info.offsets_primal_x[k + 1];
        Index nx_next = info.dims.number_of_states[k + 1];
        Index offset_g_dyn = info.offsets_g_eq_dyn[k];
        // apply out[offs_ux:offs_ux + nu + nx] +=  BAbt @ mult_eq[offs_g_dyn:offs_g_dyn + nx_next]
        gemv_n(nu + nx, nx_next, 1.0, BAbt[k], 0, 0, mult_eq, offset_g_dyn, 1.0, out, offs_ux, out,
               offs_ux);
        // apply out[offs_ux:offs_ux + nu + nx] -= mult_eq[offs_f_dyn:offs_g_dyn + nx_next]
        axpy(nx_next, -1.0, mult_eq, offset_g_dyn, out, offs_x_next, out, offs_x_next);
    };
    // equality path constraints' contributions Gg_eqt @ mult_eq
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng = info.dims.number_of_eq_constraints[k];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_g_eq = info.offsets_g_eq_path[k];
        // apply out[offs:offs+nu+nx] +=  Gg_eqt @ mult_eq[offs:offs+ng]
        gemv_n(nu + nx, ng, 1.0, Gg_eqt[k], 0, 0, mult_eq, offset_g_eq, 1.0, out, offset_ux, out,
               offset_ux);
    }
    // inequality path constraints' contributions Gg_ineqt @ mult_eq
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_g_ineq = info.offsets_g_eq_slack[k];
        // apply out[offs:offs+nu+nx] +=  Gg_ineqt @ mult_eq[offs:offs+ng_ineq]
        gemv_n(nu + nx, ng_ineq, 1.0, Gg_ineqt[k], 0, 0, mult_eq, offset_g_ineq, 1.0, out,
               offset_ux, out, offset_ux);
    }
    if (info.number_of_global_parameters > 0)
        gemv_t(info.number_of_eq_constraints,
               info.number_of_global_parameters, 1.0,
               global_parameter_jacobian, 0, 0, mult_eq, 0,
               1.0, out, info.offset_primal_global,
               out, info.offset_primal_global);
}
void Jacobian<OcpType>::get_rhs(const OcpInfo &info, VecRealView &rhs) const
{
    // dynamics constraints' right-hand side
    for (Index k = 0; k < info.dims.K - 1; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index nx_next = info.dims.number_of_states[k + 1];
        Index offset_eq_dyn = info.offsets_g_eq_dyn[k];
        // the rhs is the last row of the BAbt[k] matrix
        rowex(nx_next, 1.0, BAbt[k], nu + nx, 0, rhs, offset_eq_dyn);
    }
    // equality path constraints' right-hand side
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng = info.dims.number_of_eq_constraints[k];
        Index offset_eq_path = info.offsets_g_eq_path[k];
        // the rhs is the last row of the Gg_eqt[k] matrix
        rowex(ng, 1.0, Gg_eqt[k], nu + nx, 0, rhs, offset_eq_path);
    }
    // inequality path constraints' right-hand side
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        Index offset_eq_ineq = info.offsets_g_eq_slack[k];
        // the rhs is the last row of the Gg_ineqt[k] matrix
        rowex(ng_ineq, 1.0, Gg_ineqt[k], nu + nx, 0, rhs, offset_eq_ineq);
    }
};
void Jacobian<OcpType>::set_rhs(const OcpInfo &info, const VecRealView &rhs)
{
    // dynamics constraints' right-hand side
    for (Index k = 0; k < info.dims.K - 1; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index nx_next = info.dims.number_of_states[k + 1];
        Index offset_eq_dyn = info.offsets_g_eq_dyn[k];
        // the rhs is the last row of the BAbt[k] matrix
        rowin(nx_next, 1.0, rhs, offset_eq_dyn, BAbt[k], nu + nx, 0);
    }
    // equality path constraints' right-hand side
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng = info.dims.number_of_eq_constraints[k];
        Index offset_eq_path = info.offsets_g_eq_path[k];
        // the rhs is the last row of the Gg_eqt[k] matrix
        rowin(ng, 1.0, rhs, offset_eq_path, Gg_eqt[k], nu + nx, 0);
    }
    // inequality path constraints' right-hand side
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        Index offset_eq_ineq = info.offsets_g_eq_slack[k];
        // the rhs is the last row of the Gg_ineqt[k] matrix
        rowin(ng_ineq, 1.0, rhs, offset_eq_ineq, Gg_ineqt[k], nu + nx, 0);
    }
};

// make printable
namespace fatrop
{
    std::ostream &operator<<(std::ostream &os, const Jacobian<OcpType> &jac)
    {
        os << "Jacobian<OcpType> object with horizon length " << jac.Gg_eqt.size();
        for (int k = 0; k < jac.Gg_eqt.size(); ++k)
        {
            os << "\n ----- Stage " << k << ": -----\n";
            os << "Gg_eq:\n" << transpose(jac.Gg_eqt[k]) << "\n";
            os << "Gg_ineq:\n" << transpose(jac.Gg_ineqt[k]) << "\n";
            if (k < jac.BAbt.size())
                os << "BAb:\n" << transpose(jac.BAbt[k]) << "\n";
        }
        return os;
    }
}


















//////////////////////////////
// ImplicitOcpType-specific //
//////////////////////////////

void Jacobian<ImplicitOcpType>::PreProcess(const ProblemInfo &info,
                                           VecRealView &f,
                                           VecRealView &g){
    // GENERAL CASE
    for (int k = 0; k < info.dims.K - 1; ++k){
        // consider provided right-hand-side as b column
        rowin(info.dims.number_of_states[k + 1], 1.0,
              g, info.offsets_g_eq_dyn[k], 
              BAbt[k], info.dims.number_of_states[k] + 
              info.dims.number_of_controls[k], 0);
        // BAbt_original[k] = BAbt[k];
        gecp(info.dims.number_of_states[k] + info.dims.number_of_controls[k] + 1,
             info.dims.number_of_states[k + 1], BAbt[k], 0, 0, BAbt_original[k], 0, 0);

    }
    for (int k = 0; k < info.dims.K; ++k){
        // equalities
        rowin(info.dims.number_of_eq_constraints[k], 1.0, 
              g, info.offsets_g_eq_path[k], 
              Gg_eqt[k], info.dims.number_of_states[k] + 
              info.dims.number_of_controls[k], 0);
        // Gg_eqt_original[k] = Gg_eqt[k];
        gecp(info.dims.number_of_states[k] + info.dims.number_of_controls[k] + 1,
             info.dims.number_of_eq_constraints[k], Gg_eqt[k], 0, 0, Gg_eqt_original[k], 0, 0);

        // inequalities
        rowin(info.dims.number_of_ineq_constraints[k], 1.0, 
              g, info.offsets_g_eq_slack[k], 
              Gg_ineqt[k], info.dims.number_of_states[k] + 
              info.dims.number_of_controls[k], 0);
        // Gg_ineqt_original[k] = Gg_ineqt[k];
        gecp(info.dims.number_of_states[k] + info.dims.number_of_controls[k] + 1,
             info.dims.number_of_ineq_constraints[k], Gg_ineqt[k], 0, 0, Gg_ineqt_original[k], 0, 0);
    }
    return;

    dgemm_time = 0.0;
    // Make sure to store the current BAbt into BAbt_original before modifying BAbt matrices
    for (int k = 0; k < info.dims.K - 1; ++k){
        BAbt_original[k] = BAbt[k];                                                                 // 25% of the jacobian preprocessing time

        // AugSystemSolver<OcpType> overwrites the entries corresponding to
        // the vector b. We do the same here
        // this is necessary to make sure we have the correct BAbt matrix
        // when calling apply_on_right
        rowin(info.dims.number_of_states[k + 1], 1.0,                                               //  3% of the jacobian preprocessing time
              g, info.offsets_g_eq_dyn[k], 
              BAbt_original[k], info.dims.number_of_states[k] + 
                info.dims.number_of_controls[k], 0);
    }   

    // Compute BAbt = BAbt_original * Jt^-1
    VecRealAllocated g_copy(info.number_of_eq_constraints);                                         // 10% of the jacobian preprocessing time
    for (int i = 0; i < info.number_of_eq_constraints; i++){
        g_copy(i) = g(i);
    }    
    if (!ASSUME_INVERSE_GIVEN){
        // Jt_inv has to be prepared first
        PrepareInverseOfJ(info);
    }

    for (int k = 0; k < info.dims.K - 1; ++k){                                                      // 75% of the jacobian preprocessing time
        Index nx_next = info.dims.number_of_states[k + 1];
        Index nx = info.dims.number_of_states[k];
        Index nu = info.dims.number_of_controls[k];

        blasfeo_dgemm_nn(nx + nu + 1, nx_next, nx_next, -1.0, 
                            &BAbt_original[k].mat(), 0, 0, &Jt_inv[k].mat(), 0, 0, 0.0,
                            &BAbt[k].mat(), 0, 0, &BAbt[k].mat(), 0, 0);

        // apply transformation to rhs also, since AugSystemSolver<OcpType> 
        // overwrites the entries corresponding to the vector b in BAbt
        // by considering g (b <-- -J^-1 @ b)
        blasfeo_dgemv_t(nx_next, nx_next, -1.0,
                        &Jt_inv[k].mat(), 0, 0, 
                        &g_copy.vec(), info.offsets_g_eq_dyn[k], 0.0,
                        &g_copy.vec(), info.offsets_g_eq_dyn[k],
                        &g.vec(), info.offsets_g_eq_dyn[k]);

    } 
}

void Jacobian<ImplicitOcpType>::ResetPreProcess(const ProblemInfo &info){
    for (int k = 0; k < info.dims.K - 1; ++k){
        // std::cout << "BAbt[" << k << "]:" << std::endl << BAbt[k] << std::endl;
        // std::cout << "BAbt_original[" << k << "]:" << std::endl << BAbt_original[k] << std::endl;
        // BAbt[k] = BAbt_original[k];
        gecp(info.dims.number_of_states[k] + info.dims.number_of_controls[k] + 1,
             info.dims.number_of_states[k + 1], BAbt_original[k], 0, 0, BAbt[k], 0, 0);
    }   
    for (int k = 0; k < info.dims.K; ++k){
        // std::cout << "Gg_eqt[" << k << "]:" << std::endl << Gg_eqt[k] << std::endl;
        // std::cout << "Gg_eqt_original[" << k << "]:" << std::endl << Gg_eqt_original[k] << std::endl;
        // Gg_eqt[k] = Gg_eqt_original[k];
        // Gg_ineqt[k] = Gg_ineqt_original[k];
        gecp(info.dims.number_of_states[k] + info.dims.number_of_controls[k] + 1,
             info.dims.number_of_eq_constraints[k], Gg_eqt_original[k], 0, 0, Gg_eqt[k], 0, 0);
        gecp(info.dims.number_of_states[k] + info.dims.number_of_controls[k] + 1,
             info.dims.number_of_ineq_constraints[k], Gg_ineqt_original[k], 0, 0, Gg_ineqt[k], 0, 0);    
    }
}

void Jacobian<ImplicitOcpType>::PrepareInverseOfJ(const ProblemInfo &info){
    if (ASSUME_INVERSE_GIVEN){
        return;
    } else {
        // throw std::runtime_error("Jacobian<ImplicitOcpType>::PrepareInverseOfJ: "
        //                          "ASSUME_INVERSE_GIVEN == false is not implemented yet.");
        for (int k = 0; k < info.dims.K - 1; ++k){
            Index nx_next = info.dims.number_of_states[k + 1];
            MatRealAllocated I(nx_next, nx_next);
            MatRealAllocated Y(nx_next, nx_next);
            for (Index i = 0; i < nx_next; ++i){
                blasfeo_matel_wrap(&I.mat(), i, i) = 1.0;
            }
            // MatRealAllocated res_J_inv_test(nx_next, nx_next);
            blasfeo_dgetrf_np(nx_next, nx_next, &Jt[k].mat(), 0, 0, &Jt_LU[k].mat(), 0, 0);

            // step 2: compute inverse via solving with identity matrix
            blasfeo_dtrsm_runn(nx_next, nx_next, 1.0, &Jt_LU[k].mat(), 0, 0, &I.mat(), 0, 0, &Y.mat(), 0, 0);
            blasfeo_dtrsm_rlnu(nx_next, nx_next, 1.0, &Jt_LU[k].mat(), 0, 0, &Y.mat(), 0, 0, &Jt_inv[k].mat(), 0, 0);
        }
    }
}

void Jacobian<ImplicitOcpType>::apply_on_right(const OcpInfo& info, const VecRealView& x, Scalar alpha, const VecRealView& y, VecRealView& out, bool ignore_Jt) const{
    if (print_debug){ std::cout << "Jacobian<ImplicitOcpType>::apply_on_right" << std::endl;}
    out = alpha * y;
    // dynamics constraints BA*ux - x_next
    for (Index k = 0; k < info.dims.K - 1; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index nx_next = info.dims.number_of_states[k + 1];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_x_next = info.offsets_primal_x[k + 1];
        Index offset_dyn_eq = info.offsets_g_eq_dyn[k];
        // apply out[offs:offs+nx] =  BAbt.T @ x[offs:offs+nu+nx] + Jt @ x_next[offs:offs+nx]
        gemv_t(nu + nx, nx_next, 1.0, BAbt[k], 0, 0, x, offset_ux, 1.0, out, offset_dyn_eq, out,
               offset_dyn_eq);
        MatRealAllocated Jt_curr = Jt[k];
        if (ignore_Jt){
            for (int i = 0; i < nx_next; i++){
                for (int j = 0; j < nx_next; j++){
                    if (i != j || i > J_ranks[k] || j > J_ranks[k]){
                        Jt_curr(i, j) = 0.0;
                    } else {
                        Jt_curr(i, j) = -1.0;
                    }
                }
            }
        }
        // axpy(nx_next, -1.0, x, offset_x_next, out, offset_dyn_eq, out, offset_dyn_eq);
        gemv_t(nx_next, nx_next, 1.0, Jt_curr, 0, 0, x, offset_x_next, 1.0, out,
               offset_dyn_eq, out, offset_dyn_eq);
    }
    // equality path constraints
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng = info.dims.number_of_eq_constraints[k];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_g_eq = info.offsets_g_eq_path[k];
        // apply out[offs:offs+ng] =  Gg_eqt.T @ x[offs:offs+nu+nx]
        gemv_t(nu + nx, ng, 1.0, Gg_eqt[k], 0, 0, x, offset_ux, 1.0, out, offset_g_eq, out,
               offset_g_eq);
    }
    // slack equality path constraints Gg_ineqt @ x
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_g_ineq = info.offsets_g_eq_slack[k];
        // apply out[offs:offs+ng_ineq] =  Gg_ineqt.T @ x[offs:offs+nu+nx]
        gemv_t(nu + nx, ng_ineq, 1.0, Gg_ineqt[k], 0, 0, x, offset_ux, 1.0, out, offset_g_ineq, out,
               offset_g_ineq);
    }
    if (info.number_of_global_parameters > 0)
        gemv_n(info.number_of_eq_constraints,
               info.number_of_global_parameters, 1.0,
               global_parameter_jacobian, 0, 0, x,
               info.offset_primal_global, 1.0, out, 0, out, 0);
    if (print_debug){ std::cout << "Jacobian<ImplicitOcpType>::apply_on_right done" << std::endl;}
};

void Jacobian<ImplicitOcpType>::transpose_apply_on_right(const OcpInfo &info, const VecRealView &mult_eq,
                                                 Scalar alpha, const VecRealView &y,
                                                 VecRealView &out, bool ignore_Jt) const
{
    if (print_debug){ std::cout << "Jacobian<ImplicitOcpType>::transpose_apply_on_right" << std::endl;}
    // set the output to zero, we will add the contributions
    // dynamics constraints'contributions [BA.T ; 0; -I] @ mult_eq
    out = alpha * y;
    for (Index k = 0; k < info.dims.K - 1; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index offs_ux = info.offsets_primal_u[k];
        Index offs_x_next = info.offsets_primal_x[k + 1];
        Index nx_next = info.dims.number_of_states[k + 1];
        Index offset_g_dyn = info.offsets_g_eq_dyn[k];
        // apply out[offs_ux:offs_ux + nu + nx] +=  BAbt @ mult_eq[offs_g_dyn:offs_g_dyn + nx_next]
        gemv_n(nu + nx, nx_next, 1.0, BAbt[k], 0, 0, mult_eq, offset_g_dyn, 1.0, out, offs_ux, out,
               offs_ux);
        MatRealAllocated Jt_curr = Jt[k];
        if (ignore_Jt){
            for (int i = 0; i < nx_next; i++){
                for (int j = 0; j < nx_next; j++){
                    if (i != j || i > J_ranks[k] || j > J_ranks[k]){
                        Jt_curr(i, j) = 0.0;
                    } else {
                        Jt_curr(i, j) = -1.0;
                    }
                }
            }
        }
        // axpy(nx_next, -1.0, mult_eq, offset_g_dyn, out, offs_x_next, out, offs_x_next);
        gemv_n(nx_next, nx_next, 1.0, Jt_curr, 0, 0, mult_eq, offset_g_dyn, 1.0, out,
               offs_x_next, out, offs_x_next);
    };
    // equality path constraints' contributions Gg_eqt @ mult_eq
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng = info.dims.number_of_eq_constraints[k];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_g_eq = info.offsets_g_eq_path[k];
        // apply out[offs:offs+nu+nx] +=  Gg_eqt @ mult_eq[offs:offs+ng]
        gemv_n(nu + nx, ng, 1.0, Gg_eqt[k], 0, 0, mult_eq, offset_g_eq, 1.0, out, offset_ux, out,
               offset_ux);
    }
    // inequality path constraints' contributions Gg_ineqt @ mult_eq
    for (Index k = 0; k < info.dims.K; ++k)
    {
        Index nu = info.dims.number_of_controls[k];
        Index nx = info.dims.number_of_states[k];
        Index ng_ineq = info.dims.number_of_ineq_constraints[k];
        Index offset_ux = info.offsets_primal_u[k];
        Index offset_g_ineq = info.offsets_g_eq_slack[k];
        // apply out[offs:offs+nu+nx] +=  Gg_ineqt @ mult_eq[offs:offs+ng_ineq]
        gemv_n(nu + nx, ng_ineq, 1.0, Gg_ineqt[k], 0, 0, mult_eq, offset_g_ineq, 1.0, out,
               offset_ux, out, offset_ux);
    }
    if (info.number_of_global_parameters > 0)
        gemv_t(info.number_of_eq_constraints,
               info.number_of_global_parameters, 1.0,
               global_parameter_jacobian, 0, 0, mult_eq, 0,
               1.0, out, info.offset_primal_global,
               out, info.offset_primal_global);
    if (print_debug){ std::cout << "Jacobian<ImplicitOcpType>::transpose_apply_on_right done" << std::endl;}
}
