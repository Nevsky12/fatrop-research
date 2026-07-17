//
// Copyright (c) 2026
//

#ifndef __fatrop_ocp_global_parameter_kkt_solver_hpp__
#define __fatrop_ocp_global_parameter_kkt_solver_hpp__

#include "fatrop/common/exception.hpp"
#include "fatrop/context/context.hpp"
#include "fatrop/linear_algebra/linear_solver_return_flags.hpp"
#include "fatrop/linear_algebra/blasfeo_operations.hpp"
#include "fatrop/linear_algebra/matrix.hpp"
#include "fatrop/linear_algebra/vector.hpp"
#include "fatrop/ocp/aug_system_solver.hpp"
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/problem_info.hpp"
#include "fatrop/ocp/type.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace fatrop
{
    enum class BorderedKktFailureLocation
    {
        None,
        TrajectoryFactorization,
        TrajectoryResponse,
        BorderFactorization,
        BorderSolve,
        NonfiniteSolution
    };

    /**
     * @brief Solves an OCP KKT system bordered by one-copy global variables.
     *
     * The system is
     *
     * \f[
     * \begin{bmatrix}
     * K       & E \\
     * E^\top  & H_{pp}
     * \end{bmatrix}
     * \begin{bmatrix}
     * y \\ p
     * \end{bmatrix}
     * =
     * -
     * \begin{bmatrix}
     * r \\ r_p
     * \end{bmatrix},
     * \qquad
     * y =
     * \begin{bmatrix}
     * x \\ \lambda
     * \end{bmatrix},
     * \qquad
     * E =
     * \begin{bmatrix}
     * H_{xp} \\ J_p
     * \end{bmatrix}.
     * \f]
     *
     * The trajectory block K is factored once by the ordinary FATROP
     * recursion. Explicit dynamics without equality regularization propagate
     * all columns of E in one blocked right-hand-side traversal. Full-rank
     * implicit dynamics, with or without equality-dual stabilization, also
     * preprocess the trajectory factors once and reuse them for the complete
     * response batch. Rank-deficient implicit systems reuse the retained
     * factorization one response at a time. Only the dense parameter Schur complement
     *
     * \f[
     * S_p = H_{pp} + E^\top K^{-1}(-E)
     * \f]
     *
     * is solved with dense pivoted LU. The global variables are therefore
     * stored once instead of being copied into every stage state.
     *
     * This class is the reusable linear-algebra kernel. Barrier contributions
     * from parameter bounds, restoration terms, and inertia correction of the
     * parameter Schur complement must be assembled by the caller.
     */
    template <typename ProblemType>
    class GlobalParameterKktSolverTpl
    {
    public:
        static_assert(
            std::is_same_v<ProblemType, OcpType>
            || std::is_same_v<ProblemType, ImplicitOcpType>,
            "GlobalParameterKktSolverTpl supports OcpType and ImplicitOcpType.");

        GlobalParameterKktSolverTpl(
            const ProblemInfo &info, const Index number_of_parameters,
            std::shared_ptr<AugSystemSolver<ProblemType>> trajectory_solver = nullptr)
            : number_of_parameters_(number_of_parameters),
              number_of_primal_variables_(info.number_of_trajectory_variables),
              number_of_eq_constraints_(info.number_of_eq_constraints),
              number_of_slack_variables_(info.number_of_slack_variables),
              stages_(info.dims.K),
              controls_(info.dims.number_of_controls),
              states_(info.dims.number_of_states),
              equalities_(info.dims.number_of_eq_constraints),
              inequalities_(info.dims.number_of_ineq_constraints),
              trajectory_dims_(
                  info.dims.K, controls_, states_, equalities_, inequalities_),
              trajectory_info_(trajectory_dims_),
              trajectory_solver_(
                  trajectory_solver
                  ? std::move(trajectory_solver)
                  : std::make_shared<AugSystemSolver<ProblemType>>(
                        trajectory_info_)),
              base_primal_(number_of_primal_variables_),
              base_multipliers_(number_of_eq_constraints_),
              response_primal_(
                  number_of_primal_variables_,
                  std::max<Index>(number_of_parameters_, 1)),
              response_multipliers_(
                  number_of_eq_constraints_,
                  std::max<Index>(number_of_parameters_, 1)),
              column_rhs_primal_(number_of_primal_variables_),
              column_rhs_constraints_(number_of_eq_constraints_),
              column_primal_(number_of_primal_variables_),
              column_multipliers_(number_of_eq_constraints_),
              schur_(
                  std::max<Index>(number_of_parameters_, 1),
                  std::max<Index>(number_of_parameters_, 1)),
              schur_factor_(
                  std::max<Index>(number_of_parameters_, 1),
                  std::max<Index>(number_of_parameters_, 1)),
              schur_rhs_(std::max<Index>(number_of_parameters_, 1)),
              schur_solution_(std::max<Index>(number_of_parameters_, 1)),
              schur_residual_(std::max<Index>(number_of_parameters_, 1)),
              schur_correction_(std::max<Index>(number_of_parameters_, 1)),
              pivots_(static_cast<std::size_t>(
                  std::max<Index>(number_of_parameters_, 1)))
        {
            fatrop_assert_msg(
                number_of_parameters >= 0,
                "The global parameter dimension must be non-negative.");
        }

        Index number_of_parameters() const noexcept
        {
            return number_of_parameters_;
        }

        AugSystemSolver<ProblemType> &trajectory_solver() noexcept
        {
            return *trajectory_solver_;
        }

        const AugSystemSolver<ProblemType> &trajectory_solver() const noexcept
        {
            return *trajectory_solver_;
        }

        void set_schur_pivot_tolerance(const Scalar tolerance)
        {
            fatrop_assert_msg(
                std::isfinite(tolerance) && tolerance >= 0.0,
                "The parameter Schur pivot tolerance must be finite and non-negative.");
            schur_pivot_tolerance_ = tolerance;
        }

        Scalar schur_pivot_tolerance() const noexcept
        {
            return schur_pivot_tolerance_;
        }

        BorderedKktFailureLocation last_failure_location() const noexcept
        {
            return last_failure_location_;
        }

        /**
         * @brief Factor and solve the bordered KKT system.
         *
         * The FATROP right-hand-side convention is retained: the returned
         * solution satisfies the KKT matrix times the solution equal to the
         * negative of `(rhs_primal, rhs_constraints, rhs_parameters)`.
         *
         * @param cross_hessian H_xp, with shape n_primal x n_parameters.
         * @param parameter_jacobian J_p, with shape n_equalities x n_parameters.
         * @param parameter_hessian H_pp, with shape n_parameters x n_parameters.
         */
        LinsolReturnFlag solve(
            const ProblemInfo &info,
            Jacobian<ProblemType> &jacobian,
            Hessian<ProblemType> &hessian,
            const VecRealView &D_x,
            const VecRealView &D_s,
            const MatRealView &cross_hessian,
            const MatRealView &parameter_jacobian,
            const MatRealView &parameter_hessian,
            const VecRealView &rhs_primal,
            const VecRealView &rhs_constraints,
            const VecRealView &rhs_parameters,
            VecRealView &primal,
            VecRealView &multipliers,
            VecRealView &parameters)
        {
            return solve_impl(
                info, jacobian, hessian, D_x, nullptr, D_s,
                cross_hessian, parameter_jacobian, parameter_hessian,
                rhs_primal, rhs_constraints, rhs_parameters,
                primal, multipliers, parameters);
        }

        /**
         * @brief Stabilized bordered solve with a diagonal equality-dual block.
         *
         * `D_eq` is passed to the FATROP trajectory recursion as its equality
         * regularization. This is useful when the trajectory-only KKT block is
         * rank deficient although the complete bordered system is regular.
         * In an SQP method the stabilization changes the step, not the target
         * KKT point, because it vanishes from the residual at a zero step.
         */
        LinsolReturnFlag solve(
            const ProblemInfo &info,
            Jacobian<ProblemType> &jacobian,
            Hessian<ProblemType> &hessian,
            const VecRealView &D_x,
            const VecRealView &D_eq,
            const VecRealView &D_s,
            const MatRealView &cross_hessian,
            const MatRealView &parameter_jacobian,
            const MatRealView &parameter_hessian,
            const VecRealView &rhs_primal,
            const VecRealView &rhs_constraints,
            const VecRealView &rhs_parameters,
            VecRealView &primal,
            VecRealView &multipliers,
            VecRealView &parameters)
        {
            return solve_impl(
                info, jacobian, hessian, D_x, &D_eq, D_s,
                cross_hessian, parameter_jacobian, parameter_hessian,
                rhs_primal, rhs_constraints, rhs_parameters,
                primal, multipliers, parameters);
        }

        /**
         * @brief Solve a new right-hand side using the retained trajectory
         * and parameter-Schur factorizations.
         */
        LinsolReturnFlag solve_rhs(
            const ProblemInfo &info,
            Jacobian<ProblemType> &jacobian,
            Hessian<ProblemType> &hessian,
            const VecRealView &D_s,
            const MatRealView &cross_hessian,
            const MatRealView &parameter_jacobian,
            const VecRealView &rhs_primal,
            const VecRealView &rhs_constraints,
            const VecRealView &rhs_parameters,
            VecRealView &primal,
            VecRealView &multipliers,
            VecRealView &parameters)
        {
            return solve_rhs_impl(
                info, jacobian, hessian, nullptr, D_s,
                cross_hessian, parameter_jacobian,
                rhs_primal, rhs_constraints, rhs_parameters,
                primal, multipliers, parameters);
        }

        /** Retained-factor solve with equality-dual stabilization. */
        LinsolReturnFlag solve_rhs(
            const ProblemInfo &info,
            Jacobian<ProblemType> &jacobian,
            Hessian<ProblemType> &hessian,
            const VecRealView &D_eq,
            const VecRealView &D_s,
            const MatRealView &cross_hessian,
            const MatRealView &parameter_jacobian,
            const VecRealView &rhs_primal,
            const VecRealView &rhs_constraints,
            const VecRealView &rhs_parameters,
            VecRealView &primal,
            VecRealView &multipliers,
            VecRealView &parameters)
        {
            return solve_rhs_impl(
                info, jacobian, hessian, &D_eq, D_s,
                cross_hessian, parameter_jacobian,
                rhs_primal, rhs_constraints, rhs_parameters,
                primal, multipliers, parameters);
        }

    private:
        LinsolReturnFlag solve_impl(
            const ProblemInfo &info,
            Jacobian<ProblemType> &jacobian,
            Hessian<ProblemType> &hessian,
            const VecRealView &D_x,
            const VecRealView *D_eq,
            const VecRealView &D_s,
            const MatRealView &cross_hessian,
            const MatRealView &parameter_jacobian,
            const MatRealView &parameter_hessian,
            const VecRealView &rhs_primal,
            const VecRealView &rhs_constraints,
            const VecRealView &rhs_parameters,
            VecRealView &primal,
            VecRealView &multipliers,
            VecRealView &parameters)
        {
            validate_dimensions(
                info, D_x, D_eq, D_s, cross_hessian, parameter_jacobian,
                parameter_hessian, rhs_primal, rhs_constraints,
                rhs_parameters, primal, multipliers, parameters);
            last_failure_location_ =
                BorderedKktFailureLocation::None;
            factorization_ready_ = false;

            base_primal_ = 0.0;
            base_multipliers_ = 0.0;
            const LinsolReturnFlag factor_status =
                D_eq
                ? trajectory_solver_->solve(
                    trajectory_info_, jacobian, hessian, D_x, *D_eq, D_s,
                    rhs_primal, rhs_constraints,
                    base_primal_, base_multipliers_)
                : trajectory_solver_->solve(
                    trajectory_info_, jacobian, hessian, D_x, D_s,
                    rhs_primal, rhs_constraints,
                    base_primal_, base_multipliers_);
            if (factor_status != LinsolReturnFlag::SUCCESS)
            {
                last_failure_location_ =
                    BorderedKktFailureLocation::TrajectoryFactorization;
                return factor_status;
            }

            if (number_of_parameters_ == 0)
            {
                primal = base_primal_;
                multipliers = base_multipliers_;
                if (finite_solution(primal, multipliers, parameters))
                {
                    factorization_ready_ = true;
                    return LinsolReturnFlag::SUCCESS;
                }
                last_failure_location_ =
                    BorderedKktFailureLocation::NonfiniteSolution;
                return LinsolReturnFlag::NAN_SOLUTION;
            }

            MatRealView response_primal = response_primal_.block(
                number_of_primal_variables_, number_of_parameters_, 0, 0);
            MatRealView response_multipliers = response_multipliers_.block(
                number_of_eq_constraints_, number_of_parameters_, 0, 0);
            response_primal = 0.0;
            response_multipliers = 0.0;

            const LinsolReturnFlag response_status =
                solve_parameter_responses(
                    info, jacobian, hessian, D_x, D_eq, D_s,
                    cross_hessian, parameter_jacobian,
                    response_primal, response_multipliers);
            if (response_status != LinsolReturnFlag::SUCCESS)
            {
                last_failure_location_ =
                    BorderedKktFailureLocation::TrajectoryResponse;
                return response_status;
            }

            assemble_schur(
                cross_hessian, parameter_jacobian, parameter_hessian,
                response_primal, response_multipliers);
            assemble_schur_rhs(
                cross_hessian, parameter_jacobian, rhs_parameters);

            if (!factor_schur())
            {
                last_failure_location_ =
                    BorderedKktFailureLocation::BorderFactorization;
                return LinsolReturnFlag::NOFULL_RANK;
            }
            if (!solve_factored_schur(schur_rhs_, schur_solution_))
            {
                last_failure_location_ =
                    BorderedKktFailureLocation::BorderSolve;
                return LinsolReturnFlag::NAN_SOLUTION;
            }
            if (!refine_schur_solution())
            {
                last_failure_location_ =
                    BorderedKktFailureLocation::BorderSolve;
                return LinsolReturnFlag::NAN_SOLUTION;
            }

            gemv_n(
                number_of_primal_variables_, number_of_parameters_,
                1.0, response_primal, 0, 0,
                schur_solution_, 0,
                1.0, base_primal_, 0,
                primal, 0);
            gemv_n(
                number_of_eq_constraints_, number_of_parameters_,
                1.0, response_multipliers, 0, 0,
                schur_solution_, 0,
                1.0, base_multipliers_, 0,
                multipliers, 0);
            veccp(
                number_of_parameters_, schur_solution_, 0,
                parameters, 0);

            if (finite_solution(primal, multipliers, parameters))
            {
                factorization_ready_ = true;
                return LinsolReturnFlag::SUCCESS;
            }
            last_failure_location_ =
                BorderedKktFailureLocation::NonfiniteSolution;
            return LinsolReturnFlag::NAN_SOLUTION;
        }

        LinsolReturnFlag solve_rhs_impl(
            const ProblemInfo &info,
            Jacobian<ProblemType> &jacobian,
            Hessian<ProblemType> &hessian,
            const VecRealView *D_eq,
            const VecRealView &D_s,
            const MatRealView &cross_hessian,
            const MatRealView &parameter_jacobian,
            const VecRealView &rhs_primal,
            const VecRealView &rhs_constraints,
            const VecRealView &rhs_parameters,
            VecRealView &primal,
            VecRealView &multipliers,
            VecRealView &parameters)
        {
            validate_rhs_dimensions(
                info, D_eq, D_s, cross_hessian, parameter_jacobian,
                rhs_primal, rhs_constraints, rhs_parameters,
                primal, multipliers, parameters);
            fatrop_assert_msg(
                factorization_ready_,
                "A bordered right-hand-side solve requires a successful factorization first.");
            last_failure_location_ = BorderedKktFailureLocation::None;

            base_primal_ = 0.0;
            base_multipliers_ = 0.0;
            const LinsolReturnFlag trajectory_status =
                D_eq
                ? trajectory_solver_->solve_rhs(
                    trajectory_info_, jacobian, hessian, *D_eq, D_s,
                    rhs_primal, rhs_constraints,
                    base_primal_, base_multipliers_)
                : trajectory_solver_->solve_rhs(
                    trajectory_info_, jacobian, hessian, D_s,
                    rhs_primal, rhs_constraints,
                    base_primal_, base_multipliers_);
            if (trajectory_status != LinsolReturnFlag::SUCCESS)
            {
                last_failure_location_ =
                    BorderedKktFailureLocation::TrajectoryResponse;
                return trajectory_status;
            }

            if (number_of_parameters_ == 0)
            {
                primal = base_primal_;
                multipliers = base_multipliers_;
                return finite_solution(primal, multipliers, parameters)
                    ? LinsolReturnFlag::SUCCESS
                    : LinsolReturnFlag::NAN_SOLUTION;
            }

            MatRealView response_primal = response_primal_.block(
                number_of_primal_variables_, number_of_parameters_, 0, 0);
            MatRealView response_multipliers = response_multipliers_.block(
                number_of_eq_constraints_, number_of_parameters_, 0, 0);
            assemble_schur_rhs(
                cross_hessian, parameter_jacobian, rhs_parameters);
            if (!solve_factored_schur(schur_rhs_, schur_solution_)
                || !refine_schur_solution())
            {
                last_failure_location_ =
                    BorderedKktFailureLocation::BorderSolve;
                return LinsolReturnFlag::NAN_SOLUTION;
            }

            gemv_n(
                number_of_primal_variables_, number_of_parameters_,
                1.0, response_primal, 0, 0,
                schur_solution_, 0,
                1.0, base_primal_, 0,
                primal, 0);
            gemv_n(
                number_of_eq_constraints_, number_of_parameters_,
                1.0, response_multipliers, 0, 0,
                schur_solution_, 0,
                1.0, base_multipliers_, 0,
                multipliers, 0);
            veccp(number_of_parameters_, schur_solution_, 0, parameters, 0);
            if (finite_solution(primal, multipliers, parameters))
                return LinsolReturnFlag::SUCCESS;
            last_failure_location_ =
                BorderedKktFailureLocation::NonfiniteSolution;
            return LinsolReturnFlag::NAN_SOLUTION;
        }

        LinsolReturnFlag solve_parameter_responses(
            const ProblemInfo &info,
            Jacobian<ProblemType> &jacobian,
            Hessian<ProblemType> &hessian,
            const VecRealView &D_x,
            const VecRealView *D_eq,
            const VecRealView &D_s,
            const MatRealView &cross_hessian,
            const MatRealView &parameter_jacobian,
            MatRealView &response_primal,
            MatRealView &response_multipliers)
        {
            if constexpr (std::is_same_v<ProblemType, OcpType>)
            {
                if (!D_eq)
                    return trajectory_solver_->solve_rhs_batch(
                        trajectory_info_, jacobian, hessian, D_s,
                        cross_hessian, parameter_jacobian,
                        response_primal, response_multipliers);
            }
            else
            {
                bool full_rank_dynamics = true;
                for (Index stage = 0; stage + 1 < info.dims.K; ++stage)
                {
                    full_rank_dynamics = full_rank_dynamics
                        && jacobian.J_ranks[stage]
                           == info.dims.number_of_states[stage + 1];
                }
                if (full_rank_dynamics)
                {
                    if (D_eq)
                        return trajectory_solver_->solve_rhs_batch(
                            trajectory_info_, jacobian, hessian,
                            *D_eq, D_s,
                            cross_hessian, parameter_jacobian,
                            response_primal, response_multipliers);
                    return trajectory_solver_->solve_rhs_batch(
                        trajectory_info_, jacobian, hessian, D_s,
                        cross_hessian, parameter_jacobian,
                        response_primal, response_multipliers);
                }
            }

            // Rank-deficient implicit transitions use the verified scalar
            // reused-RHS operation. Both full-rank implicit variants above
            // have a cached batched preprocessing path.
            for (Index parameter = 0;
                 parameter < number_of_parameters_; ++parameter)
            {
                for (Index row = 0;
                     row < number_of_primal_variables_; ++row)
                    column_rhs_primal_(row) =
                        cross_hessian(row, parameter);
                for (Index row = 0;
                     row < number_of_eq_constraints_; ++row)
                    column_rhs_constraints_(row) =
                        parameter_jacobian(row, parameter);
                column_primal_ = 0.0;
                column_multipliers_ = 0.0;
                const LinsolReturnFlag status =
                    D_eq
                    ? trajectory_solver_->solve_rhs(
                        trajectory_info_, jacobian, hessian, *D_eq, D_s,
                        column_rhs_primal_,
                        column_rhs_constraints_,
                        column_primal_,
                        column_multipliers_)
                    : trajectory_solver_->solve_rhs(
                        trajectory_info_, jacobian, hessian, D_s,
                        column_rhs_primal_,
                        column_rhs_constraints_,
                        column_primal_,
                        column_multipliers_);
                if (status != LinsolReturnFlag::SUCCESS)
                    return status;

                for (Index row = 0;
                     row < number_of_primal_variables_; ++row)
                    response_primal(row, parameter) =
                        column_primal_(row);
                for (Index row = 0;
                     row < number_of_eq_constraints_; ++row)
                    response_multipliers(row, parameter) =
                        column_multipliers_(row);
            }
            return LinsolReturnFlag::SUCCESS;
        }

        void validate_dimensions(
            const ProblemInfo &info,
            const VecRealView &D_x,
            const VecRealView *D_eq,
            const VecRealView &D_s,
            const MatRealView &cross_hessian,
            const MatRealView &parameter_jacobian,
            const MatRealView &parameter_hessian,
            const VecRealView &rhs_primal,
            const VecRealView &rhs_constraints,
            const VecRealView &rhs_parameters,
            const VecRealView &primal,
            const VecRealView &multipliers,
            const VecRealView &parameters) const
        {
            fatrop_assert_msg(
                info.dims.K == stages_
                && info.dims.number_of_controls == controls_
                && info.dims.number_of_states == states_
                && info.dims.number_of_eq_constraints == equalities_
                && info.dims.number_of_ineq_constraints == inequalities_,
                "The bordered solver received a different OCP stage partition.");
            fatrop_assert_msg(
                info.number_of_trajectory_variables == number_of_primal_variables_
                && info.number_of_eq_constraints == number_of_eq_constraints_
                && info.number_of_slack_variables == number_of_slack_variables_,
                "The bordered solver received inconsistent aggregate dimensions.");
            fatrop_assert_msg(
                D_x.m() == number_of_primal_variables_
                && rhs_primal.m() == number_of_primal_variables_
                && primal.m() == number_of_primal_variables_,
                "The bordered solver primal vector dimensions are inconsistent.");
            fatrop_assert_msg(
                D_s.m() == number_of_slack_variables_,
                "The bordered solver slack diagonal dimension is inconsistent.");
            if constexpr (std::is_same_v<ProblemType, ImplicitOcpType>)
            {
                fatrop_assert_msg(
                    !D_eq || D_eq->m() == info.number_of_g_eq_path,
                    "The implicit bordered solver requires the path-equality diagonal.");
            }
            else
            {
                fatrop_assert_msg(
                    !D_eq || D_eq->m() == info.number_of_g_eq_path,
                    "The explicit bordered solver requires the path-equality diagonal.");
            }
            fatrop_assert_msg(
                rhs_constraints.m() == number_of_eq_constraints_
                && multipliers.m() == number_of_eq_constraints_,
                "The bordered solver equality vector dimensions are inconsistent.");
            fatrop_assert_msg(
                cross_hessian.m() == number_of_primal_variables_
                && cross_hessian.n() == number_of_parameters_,
                "H_xp has an inconsistent shape.");
            fatrop_assert_msg(
                parameter_jacobian.m() == number_of_eq_constraints_
                && parameter_jacobian.n() == number_of_parameters_,
                "J_p has an inconsistent shape.");
            fatrop_assert_msg(
                parameter_hessian.m() == number_of_parameters_
                && parameter_hessian.n() == number_of_parameters_,
                "H_pp has an inconsistent shape.");
            fatrop_assert_msg(
                rhs_parameters.m() == number_of_parameters_
                && parameters.m() == number_of_parameters_,
                "The global parameter vector dimensions are inconsistent.");
        }

        void validate_rhs_dimensions(
            const ProblemInfo &info,
            const VecRealView *D_eq,
            const VecRealView &D_s,
            const MatRealView &cross_hessian,
            const MatRealView &parameter_jacobian,
            const VecRealView &rhs_primal,
            const VecRealView &rhs_constraints,
            const VecRealView &rhs_parameters,
            const VecRealView &primal,
            const VecRealView &multipliers,
            const VecRealView &parameters) const
        {
            fatrop_assert_msg(
                info.dims.K == stages_
                && info.dims.number_of_controls == controls_
                && info.dims.number_of_states == states_
                && info.dims.number_of_eq_constraints == equalities_
                && info.dims.number_of_ineq_constraints == inequalities_,
                "The bordered RHS solver received a different OCP partition.");
            fatrop_assert_msg(
                info.number_of_trajectory_variables == number_of_primal_variables_
                && info.number_of_eq_constraints == number_of_eq_constraints_
                && info.number_of_slack_variables == number_of_slack_variables_,
                "The bordered RHS solver received inconsistent aggregate dimensions.");
            fatrop_assert_msg(
                D_s.m() == number_of_slack_variables_
                && rhs_primal.m() == number_of_primal_variables_
                && primal.m() == number_of_primal_variables_
                && rhs_constraints.m() == number_of_eq_constraints_
                && multipliers.m() == number_of_eq_constraints_
                && rhs_parameters.m() == number_of_parameters_
                && parameters.m() == number_of_parameters_,
                "The bordered RHS vector dimensions are inconsistent.");
            fatrop_assert_msg(
                cross_hessian.m() == number_of_primal_variables_
                && cross_hessian.n() == number_of_parameters_
                && parameter_jacobian.m() == number_of_eq_constraints_
                && parameter_jacobian.n() == number_of_parameters_,
                "The bordered RHS matrix dimensions are inconsistent.");
            if constexpr (std::is_same_v<ProblemType, ImplicitOcpType>)
                fatrop_assert_msg(
                    !D_eq || D_eq->m() == info.number_of_g_eq_path,
                    "The implicit bordered RHS solver requires the path-equality diagonal.");
            else
                fatrop_assert_msg(
                    !D_eq || D_eq->m() == info.number_of_g_eq_path,
                    "The explicit bordered RHS solver requires the path-equality diagonal.");
        }

        void assemble_schur(
            const MatRealView &cross_hessian,
            const MatRealView &parameter_jacobian,
            const MatRealView &parameter_hessian,
            const MatRealView &response_primal,
            const MatRealView &response_multipliers)
        {
            gecp(
                number_of_parameters_, number_of_parameters_,
                parameter_hessian, 0, 0,
                schur_, 0, 0);
            gemm_tn(
                number_of_parameters_, number_of_parameters_,
                number_of_primal_variables_,
                1.0, cross_hessian, 0, 0,
                response_primal, 0, 0,
                1.0, schur_, 0, 0,
                schur_, 0, 0);
            gemm_tn(
                number_of_parameters_, number_of_parameters_,
                number_of_eq_constraints_,
                1.0, parameter_jacobian, 0, 0,
                response_multipliers, 0, 0,
                1.0, schur_, 0, 0,
                schur_, 0, 0);
        }

        void assemble_schur_rhs(
            const MatRealView &cross_hessian,
            const MatRealView &parameter_jacobian,
            const VecRealView &rhs_parameters)
        {
            veccpsc(
                number_of_parameters_, -1.0,
                rhs_parameters, 0,
                schur_rhs_, 0);
            gemv_t(
                number_of_primal_variables_, number_of_parameters_,
                -1.0, cross_hessian, 0, 0,
                base_primal_, 0,
                1.0, schur_rhs_, 0,
                schur_rhs_, 0);
            gemv_t(
                number_of_eq_constraints_, number_of_parameters_,
                -1.0, parameter_jacobian, 0, 0,
                base_multipliers_, 0,
                1.0, schur_rhs_, 0,
                schur_rhs_, 0);
        }

        bool factor_schur()
        {
            Scalar scale = 0.0;
            for (Index row = 0; row < number_of_parameters_; ++row)
            {
                for (Index column = 0;
                     column < number_of_parameters_; ++column)
                {
                    const Scalar value = schur_(row, column);
                    if (!std::isfinite(value))
                        return false;
                    schur_factor_(row, column) = value;
                    scale = std::max(scale, std::abs(value));
                }
            }
            const Scalar threshold =
                schur_pivot_tolerance_ * std::max<Scalar>(scale, 1.0);

            for (Index column = 0;
                 column < number_of_parameters_; ++column)
            {
                Index pivot = column;
                Scalar pivot_size =
                    std::abs(schur_factor_(column, column));
                for (Index row = column + 1;
                     row < number_of_parameters_; ++row)
                {
                    const Scalar candidate =
                        std::abs(schur_factor_(row, column));
                    if (candidate > pivot_size)
                    {
                        pivot = row;
                        pivot_size = candidate;
                    }
                }
                if (!std::isfinite(pivot_size) || pivot_size <= threshold)
                    return false;
                pivots_[static_cast<std::size_t>(column)] = pivot;
                if (pivot != column)
                {
                    for (Index j = 0;
                         j < number_of_parameters_; ++j)
                    {
                        std::swap(
                            schur_factor_(column, j),
                            schur_factor_(pivot, j));
                    }
                }

                const Scalar diagonal =
                    schur_factor_(column, column);
                for (Index row = column + 1;
                     row < number_of_parameters_; ++row)
                {
                    schur_factor_(row, column) /= diagonal;
                    const Scalar multiplier =
                        schur_factor_(row, column);
                    for (Index j = column + 1;
                         j < number_of_parameters_; ++j)
                    {
                        schur_factor_(row, j) -=
                            multiplier * schur_factor_(column, j);
                    }
                }
            }
            return true;
        }

        bool solve_factored_schur(
            const VecRealView &rhs,
            VecRealView &solution) const
        {
            for (Index row = 0; row < number_of_parameters_; ++row)
                solution(row) = rhs(row);

            for (Index row = 0; row < number_of_parameters_; ++row)
            {
                const Index pivot =
                    pivots_[static_cast<std::size_t>(row)];
                if (pivot != row)
                    std::swap(solution(row), solution(pivot));
            }

            for (Index row = 0; row < number_of_parameters_; ++row)
            {
                Scalar value = solution(row);
                for (Index column = 0; column < row; ++column)
                    value -= schur_factor_(row, column) * solution(column);
                solution(row) = value;
            }

            for (Index row = number_of_parameters_; row-- > 0;)
            {
                Scalar value = solution(row);
                for (Index column = row + 1;
                     column < number_of_parameters_; ++column)
                {
                    value -= schur_factor_(row, column)
                           * solution(column);
                }
                const Scalar diagonal = schur_factor_(row, row);
                if (!std::isfinite(diagonal) || diagonal == 0.0)
                    return false;
                solution(row) = value / diagonal;
                if (!std::isfinite(solution(row)))
                    return false;
            }
            return true;
        }

        bool refine_schur_solution()
        {
            for (Index iteration = 0; iteration < 2; ++iteration)
            {
                Scalar residual_norm = 0.0;
                Scalar rhs_norm = 0.0;
                for (Index row = 0;
                     row < number_of_parameters_; ++row)
                {
                    Scalar value = schur_rhs_(row);
                    for (Index column = 0;
                         column < number_of_parameters_; ++column)
                    {
                        value -= schur_(row, column)
                               * schur_solution_(column);
                    }
                    schur_residual_(row) = value;
                    residual_norm =
                        std::max(residual_norm, std::abs(value));
                    rhs_norm = std::max(
                        rhs_norm, std::abs(schur_rhs_(row)));
                }
                if (!std::isfinite(residual_norm))
                    return false;
                if (residual_norm <=
                    32.0 * std::numeric_limits<Scalar>::epsilon()
                    * std::max<Scalar>(rhs_norm, 1.0))
                {
                    return true;
                }
                if (!solve_factored_schur(
                        schur_residual_, schur_correction_))
                    return false;
                for (Index parameter = 0;
                     parameter < number_of_parameters_; ++parameter)
                {
                    schur_solution_(parameter) +=
                        schur_correction_(parameter);
                }
            }
            return schur_solution_.is_finite();
        }

        static bool finite_solution(
            const VecRealView &primal,
            const VecRealView &multipliers,
            const VecRealView &parameters)
        {
            return primal.is_finite()
                && multipliers.is_finite()
                && parameters.is_finite();
        }

        const Index number_of_parameters_;
        const Index number_of_primal_variables_;
        const Index number_of_eq_constraints_;
        const Index number_of_slack_variables_;
        const int stages_;
        const std::vector<Index> controls_;
        const std::vector<Index> states_;
        const std::vector<Index> equalities_;
        const std::vector<Index> inequalities_;
        const ProblemDims trajectory_dims_;
        const ProblemInfo trajectory_info_;

        std::shared_ptr<AugSystemSolver<ProblemType>> trajectory_solver_;
        VecRealAllocated base_primal_;
        VecRealAllocated base_multipliers_;
        MatRealAllocated response_primal_;
        MatRealAllocated response_multipliers_;
        VecRealAllocated column_rhs_primal_;
        VecRealAllocated column_rhs_constraints_;
        VecRealAllocated column_primal_;
        VecRealAllocated column_multipliers_;
        MatRealAllocated schur_;
        MatRealAllocated schur_factor_;
        VecRealAllocated schur_rhs_;
        VecRealAllocated schur_solution_;
        VecRealAllocated schur_residual_;
        VecRealAllocated schur_correction_;
        std::vector<Index> pivots_;
        Scalar schur_pivot_tolerance_ = 1e-12;
        BorderedKktFailureLocation last_failure_location_ =
            BorderedKktFailureLocation::None;
        bool factorization_ready_ = false;
    };

    using GlobalParameterKktSolver =
        GlobalParameterKktSolverTpl<OcpType>;
    using ImplicitGlobalParameterKktSolver =
        GlobalParameterKktSolverTpl<ImplicitOcpType>;

} // namespace fatrop

#endif // __fatrop_ocp_global_parameter_kkt_solver_hpp__
