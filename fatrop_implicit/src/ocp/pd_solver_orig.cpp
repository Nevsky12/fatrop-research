//
// Copyright (c) 2024 Lander Vanroye, KU Leuven
//

#include "fatrop/ocp/pd_solver_orig.hpp"
#include "fatrop/linear_algebra/linear_solver.hxx"
#include "fatrop/linear_algebra/vector.hpp"
#include "fatrop/ocp/aug_system_solver.hpp"
#include "fatrop/ocp/problem_info.hpp"
#include <algorithm>
#include <type_traits>
using namespace fatrop;

// instantiate the template class
template class fatrop::PdSystemType<OcpType>;
template class fatrop::PdSystemType<ImplicitOcpType>;
template class fatrop::PdSystemType<AcceleratedOcpType>;

template class fatrop::LinearSolver<PdSolverOrig<OcpType>, PdSystemType<OcpType>>;
template class fatrop::LinearSolver<PdSolverOrig<ImplicitOcpType>, PdSystemType<ImplicitOcpType>>;
template class fatrop::LinearSolver<PdSolverOrig<AcceleratedOcpType>, PdSystemType<AcceleratedOcpType>>;

template <typename ProblemType>
PdSolverOrig<ProblemType>::PdSolverOrig(const ProblemInfo &info,
                                    const std::shared_ptr<AugSystemSolver<ProblemType>> &aug_system_solver)
    : LinearSolver<PdSolverOrig<ProblemType>, PdSystemType<ProblemType>>(
          LinearSystem<PdSystemType<ProblemType>>::m(info)),
      sigma_inverse_(info.number_of_slack_variables), ss_(info.number_of_slack_variables),
      g_ii_(info.number_of_slack_variables), D_ii_(info.number_of_slack_variables),
      gg_(info.number_of_eq_constraints), x_aug_(info.number_of_primal_variables),
      mult_aug_(info.number_of_eq_constraints),
      parameter_hessian_regularized_(
          std::max<Index>(info.number_of_global_parameters, 1),
          std::max<Index>(info.number_of_global_parameters, 1)),
      aug_system_solver_(aug_system_solver)
{
    if constexpr (!std::is_same_v<ProblemType, AcceleratedOcpType>)
    {
        if (info.number_of_global_parameters > 0)
            border_solver_ = std::make_unique<BorderSolver>(
                info, info.number_of_global_parameters, aug_system_solver_);
    }
}

template <typename ProblemType>
void PdSolverOrig<ProblemType>::assemble_regularized_parameter_hessian(
    LinearSystem<PdSystemType<ProblemType>> &ls)
{
    const Index np = ls.info_.number_of_global_parameters;
    parameter_hessian_regularized_ = 0.0;
    for (Index row = 0; row < np; ++row)
    {
        for (Index column = 0; column < np; ++column)
            parameter_hessian_regularized_(row, column) =
                ls.hess_.global_parameter_hessian(row, column);
        parameter_hessian_regularized_(row, row) +=
            ls.D_x_(ls.info_.offset_primal_global + row);
    }
}

template <typename ProblemType>
LinsolReturnFlag PdSolverOrig<ProblemType>::solve_augmented_system(
    LinearSystem<PdSystemType<ProblemType>> &ls)
{
    const Index np = ls.info_.number_of_global_parameters;
    if (np == 0)
    {
        if (ls.De_is_zero_)
            return aug_system_solver_->solve(
                ls.info_, ls.jac_, ls.hess_, ls.D_x_, D_ii_,
                ls.rhs_f_x_, gg_, x_aug_, mult_aug_);
        return aug_system_solver_->solve(
            ls.info_, ls.jac_, ls.hess_, ls.D_x_,
            ls.D_e_.block(
                ls.info_.number_of_g_eq_path,
                ls.info_.offset_g_eq_path),
            D_ii_, ls.rhs_f_x_, gg_, x_aug_, mult_aug_);
    }

    if constexpr (std::is_same_v<ProblemType, AcceleratedOcpType>)
    {
        return LinsolReturnFlag::NOFULL_RANK;
    }
    else
    {
        assemble_regularized_parameter_hessian(ls);
        VecRealView trajectory_diagonal = ls.D_x_.block(
            ls.info_.number_of_trajectory_variables, ls.info_.offset_primal);
        VecRealView trajectory_rhs = ls.rhs_f_x_.block(
            ls.info_.number_of_trajectory_variables, ls.info_.offset_primal);
        VecRealView parameter_rhs = ls.rhs_f_x_.block(
            np, ls.info_.offset_primal_global);
        VecRealView trajectory_solution = x_aug_.block(
            ls.info_.number_of_trajectory_variables, ls.info_.offset_primal);
        VecRealView parameter_solution = x_aug_.block(
            np, ls.info_.offset_primal_global);
        MatRealView cross_hessian =
            ls.hess_.global_parameter_cross_hessian.block(
                ls.info_.number_of_trajectory_variables, np, 0, 0);
        MatRealView parameter_jacobian =
            ls.jac_.global_parameter_jacobian.block(
                ls.info_.number_of_eq_constraints, np, 0, 0);
        MatRealView parameter_hessian =
            parameter_hessian_regularized_.block(np, np, 0, 0);

        if (ls.De_is_zero_)
            return border_solver_->solve(
                ls.info_, ls.jac_, ls.hess_, trajectory_diagonal, D_ii_,
                cross_hessian, parameter_jacobian, parameter_hessian,
                trajectory_rhs, gg_, parameter_rhs,
                trajectory_solution, mult_aug_, parameter_solution);

        if constexpr (std::is_same_v<ProblemType, ImplicitOcpType>)
        {
            VecRealView path_equality_diagonal = ls.D_e_.block(
                ls.info_.number_of_g_eq_path,
                ls.info_.offset_g_eq_path);
            return border_solver_->solve(
                ls.info_, ls.jac_, ls.hess_, trajectory_diagonal,
                path_equality_diagonal, D_ii_, cross_hessian, parameter_jacobian,
                parameter_hessian, trajectory_rhs, gg_, parameter_rhs,
                trajectory_solution, mult_aug_, parameter_solution);
        }
        else
        {
            VecRealView path_equality_diagonal = ls.D_e_.block(
                ls.info_.number_of_g_eq_path,
                ls.info_.offset_g_eq_path);
            return border_solver_->solve(
                ls.info_, ls.jac_, ls.hess_, trajectory_diagonal,
                path_equality_diagonal, D_ii_, cross_hessian,
                parameter_jacobian, parameter_hessian,
                trajectory_rhs, gg_, parameter_rhs,
                trajectory_solution, mult_aug_, parameter_solution);
        }
    }
}

template <typename ProblemType>
LinsolReturnFlag PdSolverOrig<ProblemType>::solve_augmented_rhs(
    LinearSystem<PdSystemType<ProblemType>> &ls)
{
    const Index np = ls.info_.number_of_global_parameters;
    if (np == 0)
    {
        if (ls.De_is_zero_)
            return aug_system_solver_->solve_rhs(
                ls.info_, ls.jac_, ls.hess_, D_ii_,
                ls.rhs_f_x_, gg_, x_aug_, mult_aug_);
        return aug_system_solver_->solve_rhs(
            ls.info_, ls.jac_, ls.hess_,
            ls.D_e_.block(
                ls.info_.number_of_g_eq_path,
                ls.info_.offset_g_eq_path),
            D_ii_, ls.rhs_f_x_, gg_, x_aug_, mult_aug_);
    }

    if constexpr (std::is_same_v<ProblemType, AcceleratedOcpType>)
    {
        return LinsolReturnFlag::NOFULL_RANK;
    }
    else
    {
        VecRealView trajectory_rhs = ls.rhs_f_x_.block(
            ls.info_.number_of_trajectory_variables, ls.info_.offset_primal);
        VecRealView parameter_rhs = ls.rhs_f_x_.block(
            np, ls.info_.offset_primal_global);
        VecRealView trajectory_solution = x_aug_.block(
            ls.info_.number_of_trajectory_variables, ls.info_.offset_primal);
        VecRealView parameter_solution = x_aug_.block(
            np, ls.info_.offset_primal_global);
        MatRealView cross_hessian =
            ls.hess_.global_parameter_cross_hessian.block(
                ls.info_.number_of_trajectory_variables, np, 0, 0);
        MatRealView parameter_jacobian =
            ls.jac_.global_parameter_jacobian.block(
                ls.info_.number_of_eq_constraints, np, 0, 0);

        if (ls.De_is_zero_)
            return border_solver_->solve_rhs(
                ls.info_, ls.jac_, ls.hess_, D_ii_,
                cross_hessian, parameter_jacobian,
                trajectory_rhs, gg_, parameter_rhs,
                trajectory_solution, mult_aug_, parameter_solution);

        if constexpr (std::is_same_v<ProblemType, ImplicitOcpType>)
        {
            VecRealView path_equality_diagonal = ls.D_e_.block(
                ls.info_.number_of_g_eq_path,
                ls.info_.offset_g_eq_path);
            return border_solver_->solve_rhs(
                ls.info_, ls.jac_, ls.hess_, path_equality_diagonal, D_ii_,
                cross_hessian, parameter_jacobian,
                trajectory_rhs, gg_, parameter_rhs,
                trajectory_solution, mult_aug_, parameter_solution);
        }
        else
        {
            VecRealView path_equality_diagonal = ls.D_e_.block(
                ls.info_.number_of_g_eq_path,
                ls.info_.offset_g_eq_path);
            return border_solver_->solve_rhs(
                ls.info_, ls.jac_, ls.hess_, path_equality_diagonal, D_ii_,
                cross_hessian, parameter_jacobian,
                trajectory_rhs, gg_, parameter_rhs,
                trajectory_solution, mult_aug_, parameter_solution);
        }
    }
}

template <typename ProblemType>
void PdSolverOrig<ProblemType>::reduce(LinearSystem<PdSystemType<ProblemType>> &ls)
{
    VecRealView gi =
        ls.rhs_g_.block(ls.info_.number_of_slack_variables, ls.info_.offset_g_eq_slack);
    sigma_inverse_ =
        1. / (ls.D_x_.block(ls.info_.number_of_slack_variables, ls.info_.offset_slack) +
              1. / ls.Sl_i_ * ls.Zl_i_ + 1. / ls.Su_i_ * ls.Zu_i_);
    ss_ = ls.rhs_f_s_ + 1. / ls.Sl_i_ * ls.rhs_cl_ - 1. / ls.Su_i_ * ls.rhs_cu_;
    D_ii_ = sigma_inverse_ + ls.D_e_.block(ls.info_.number_of_g_eq_slack, ls.info_.offset_g_eq_slack);
    g_ii_ = gi + sigma_inverse_ * ss_;

    gg_.block(ls.info_.number_of_g_eq_path, ls.info_.offset_g_eq_path) =
        ls.rhs_g_.block(ls.info_.number_of_g_eq_path, ls.info_.offset_g_eq_path);
    gg_.block(ls.info_.number_of_g_eq_dyn, ls.info_.offset_g_eq_dyn) =
        ls.rhs_g_.block(ls.info_.number_of_g_eq_dyn, ls.info_.offset_g_eq_dyn);
    gg_.block(ls.info_.number_of_g_eq_slack, ls.info_.offset_g_eq_slack) = g_ii_;
}
template <typename ProblemType>
void PdSolverOrig<ProblemType>::dereduce(LinearSystem<PdSystemType<ProblemType>> &ls, VecRealView &x)
{
    // set x
    x.block(ls.info_.number_of_primal_variables, ls.info_.pd_orig_offset_primal) = x_aug_;
    // set mult
    x.block(ls.info_.number_of_eq_constraints, ls.info_.pd_orig_offset_mult) = mult_aug_;
    // set s
    VecRealView mult_i =
        mult_aug_.block(ls.info_.number_of_slack_variables, ls.info_.offset_g_eq_slack);
    x.block(ls.info_.number_of_slack_variables, ls.info_.pd_orig_offset_slack) =
        sigma_inverse_ * (mult_i - ss_);
    // set zl and zu
    x.block(ls.info_.number_of_slack_variables, ls.info_.pd_orig_offset_zl) =
        1. / ls.Sl_i_ *
        (-ls.rhs_cl_ -
         ls.Zl_i_ * x.block(ls.info_.number_of_slack_variables, ls.info_.pd_orig_offset_slack));
    x.block(ls.info_.number_of_slack_variables, ls.info_.pd_orig_offset_zu) =
        1. / ls.Su_i_ *
        (-ls.rhs_cu_ +
         ls.Zu_i_ * x.block(ls.info_.number_of_slack_variables, ls.info_.pd_orig_offset_slack));
}
template <typename ProblemType>
LinsolReturnFlag PdSolverOrig<ProblemType>::solve_once_impl(LinearSystem<PdSystemType<ProblemType>> &ls,
                                                        VecRealView &x)
{
    //    [ H + D_x    0        A_e^T  A_d^T  A_i^T    0     0  ] [ x   ] = [ -f_x ]
    //    [    0     D_x          0      0    -I      -I     I  ] [ s   ] = [ -f_s ]
    //    [ A_e       0        -D_e      0     0       0     0  ] [ λ_e ] = [ -g_e ]
    //    [ A_d       0          0       0     0       0     0  ] [ λ_d ] = [ -g_d ]
    //    [ A_i      -I          0       0   -D_i      0     0  ] [ λ_i ] = [ -g_i ]
    //    [   0     Zl_i         0       0     0     Sl_i    0  ] [ zl  ] = [ -cl  ]
    //    [   0    -Zu_i         0       0     0      0    Su_i ] [ zu  ] = [ -cu  ]

    // Step 1: Eliminate \(zl\) and \(zu\)
    //
    // From the last equation:
    //     \( zl = Sl_i^{-1} (-cl - Zl_i s) \)
    //     \( zu = Su_i^{-1} (-cu + Zu_i s) \)

    //    [ H + D_x    0        A_e^T  A_d^T  A_i^T ] [ x   ] = [ -f_x ]
    //    [    0     \Sigma      0       0    -I    ] [ s   ] = [ -ss  ]
    //    [ A_e       0        -D_e      0     0    ] [ λ_e ] = [ -g_e ]
    //    [ A_d       0          0       0     0    ] [ λ_d ] = [ -g_d ]
    //    [ A_i      -I          0       0    -D_i  ] [ λ_i ] = [ -g_i ]
    //    with \Sigma = D_x + Sl{-1} Zl + Su{-1} Zu
    //          ss = f_s  + Sl_i^{-1} cl - Su_i^{-1} cu
    // Step 2: Eliminate
    //    \(s\) = \Sigma{-1} (λ_i - ss)

    //    [ H + D_x    A_e^T  A_d^T  A_i^T  ] [ x   ] = [ -f_x ]
    //    [ A_e       -D_e      0     0     ] [ λ_e ] = [ -g_e ]
    //    [ A_d         0       0     0     ] [ λ_d ] = [ -g_d ]
    //    [ A_i         0       0     -D_ii ] [ λ_i ] = [ -g_ii]
    //    with D_ii = \Sigma^{-1} + D_i
    //         g_ii =  g_i + \Sigma{-1} ss
    //  This system is in the Augmented system form and can be solved by AugSystemSolver<OcpType>
    // call aug_system_solver to solve the system
    reduce(ls);
    const LinsolReturnFlag ret = solve_augmented_system(ls);
    dereduce(ls, x);
    return ret;
}
template <typename ProblemType>
void PdSolverOrig<ProblemType>::solve_rhs_impl(LinearSystem<PdSystemType<ProblemType>> &ls, VecRealView &x)
{
    reduce(ls);
    (void)solve_augmented_rhs(ls);
    dereduce(ls, x);
}

template class fatrop::PdSolverOrig<OcpType>;
template class fatrop::PdSolverOrig<ImplicitOcpType>;
template class fatrop::PdSolverOrig<AcceleratedOcpType>;
