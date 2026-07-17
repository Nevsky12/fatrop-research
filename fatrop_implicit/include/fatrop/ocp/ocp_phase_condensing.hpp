//
// Copyright (c) 2026
//

#ifndef __fatrop_ocp_ocp_phase_condensing_hpp__
#define __fatrop_ocp_ocp_phase_condensing_hpp__

#include "fatrop/common/exception.hpp"
#include "fatrop/context/context.hpp"
#include "fatrop/linear_algebra/blasfeo_operations.hpp"
#include "fatrop/linear_algebra/linear_solver_return_flags.hpp"
#include "fatrop/linear_algebra/matrix.hpp"
#include "fatrop/linear_algebra/vector.hpp"
#include "fatrop/ocp/aug_system_solver.hpp"
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/problem_info.hpp"
#include "fatrop/ocp/type.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <type_traits>
#include <vector>

namespace fatrop
{
    /** Non-owning KKT data for one independently condensable OCP phase. */
    template <typename ProblemType>
    struct OcpPhaseCondensingViewTpl
    {
        OcpPhaseCondensingViewTpl(
            const ProblemInfo &info_value,
            Jacobian<ProblemType> &jacobian_value,
            Hessian<ProblemType> &hessian_value,
            const VecRealView &D_x_value,
            const VecRealView &D_eq_value,
            const VecRealView &D_s_value,
            const MatRealView &cross_hessian_value,
            const MatRealView &border_jacobian_value,
            const VecRealView &rhs_primal_value,
            const VecRealView &rhs_constraints_value,
            const VecRealView &primal_value,
            const VecRealView &multipliers_value,
            const bool equality_dual_diagonal_is_zero_value = false)
            : info(info_value),
              jacobian(jacobian_value),
              hessian(hessian_value),
              D_x(D_x_value),
              D_eq(D_eq_value),
              D_s(D_s_value),
              cross_hessian(cross_hessian_value),
              border_jacobian(border_jacobian_value),
              rhs_primal(rhs_primal_value),
              rhs_constraints(rhs_constraints_value),
              primal(primal_value),
              multipliers(multipliers_value),
              equality_dual_diagonal_is_zero(
                  equality_dual_diagonal_is_zero_value)
        {
        }

        const ProblemInfo &info;
        Jacobian<ProblemType> &jacobian;
        Hessian<ProblemType> &hessian;
        VecRealView D_x;
        VecRealView D_eq;
        VecRealView D_s;
        MatRealView cross_hessian;
        MatRealView border_jacobian;
        VecRealView rhs_primal;
        VecRealView rhs_constraints;
        VecRealView primal;
        VecRealView multipliers;
        bool equality_dual_diagonal_is_zero = false;
    };

    struct OcpPhaseCondensingStatistics
    {
        Index incident_response_columns = 0;
        Index active_response_columns = 0;
        double factorization_milliseconds = 0.0;
        double response_milliseconds = 0.0;
        double condensation_milliseconds = 0.0;
    };

    enum class OcpPhaseCondensingFailureLocation
    {
        None,
        TrajectoryFactorization,
        TrajectoryResponse
    };

    /**
     * @brief Condense one explicit or implicit FATROP trajectory KKT block.
     *
     * For
     *
     *     [K E; E^T H] [y;v] = -[r;r_v],
     *
     * the kernel retains the FATROP factorization of K, computes
     * `base = -K^{-1}r`, `response = -K^{-1}E`, and exposes
     * `E^T response` plus `E^T base`.  A higher-level reduced solver owns H,
     * r_v, the scope graph, and the final solve for v.
     */
    template <typename ProblemType>
    class OcpPhaseCondensingKernelTpl
    {
    public:
        static_assert(
            std::is_same_v<ProblemType, OcpType>
            || std::is_same_v<ProblemType, ImplicitOcpType>,
            "OcpPhaseCondensingKernelTpl supports explicit and implicit OCPs.");

        OcpPhaseCondensingKernelTpl(
            const ProblemInfo &info,
            const Index incident_columns)
            : columns_(incident_columns),
              solver_(std::make_unique<AugSystemSolver<ProblemType>>(info)),
              base_primal_(info.number_of_primal_variables),
              base_multipliers_(info.number_of_eq_constraints),
              response_primal_(
                  info.number_of_primal_variables,
                  std::max<Index>(columns_, 1)),
              response_multipliers_(
                  info.number_of_eq_constraints,
                  std::max<Index>(columns_, 1)),
              compact_rhs_primal_(
                  info.number_of_primal_variables,
                  std::max<Index>(columns_, 1)),
              compact_rhs_constraints_(
                  info.number_of_eq_constraints,
                  std::max<Index>(columns_, 1)),
              compact_response_primal_(
                  info.number_of_primal_variables,
                  std::max<Index>(columns_, 1)),
              compact_response_multipliers_(
                  info.number_of_eq_constraints,
                  std::max<Index>(columns_, 1)),
              compact_condensed_(
                  std::max<Index>(columns_, 1),
                  std::max<Index>(columns_, 1)),
              compact_condensed_rhs_(std::max<Index>(columns_, 1)),
              condensed_(
                  std::max<Index>(columns_, 1),
                  std::max<Index>(columns_, 1)),
              condensed_rhs_(std::max<Index>(columns_, 1)),
              column_rhs_primal_(info.number_of_primal_variables),
              column_rhs_constraints_(info.number_of_eq_constraints),
              column_primal_(info.number_of_primal_variables),
              column_multipliers_(info.number_of_eq_constraints),
              residual_primal_(info.number_of_primal_variables),
              residual_constraints_(info.number_of_eq_constraints),
              correction_primal_(info.number_of_primal_variables),
              correction_multipliers_(info.number_of_eq_constraints),
              hessian_scratch_(info.number_of_primal_variables),
              jacobian_scratch_(info.number_of_eq_constraints),
              jacobian_transpose_product_(info.number_of_primal_variables),
              active_column_position_(
                  static_cast<std::size_t>(columns_), -1)
        {
            fatrop_assert_msg(
                columns_ >= 0,
                "A phase-condensing incident dimension must be non-negative.");
            active_columns_.reserve(static_cast<std::size_t>(columns_));
        }

        Index incident_columns() const noexcept
        {
            return columns_;
        }

        AugSystemSolver<ProblemType> &trajectory_solver() noexcept
        {
            return *solver_;
        }

        const OcpPhaseCondensingStatistics &last_statistics() const noexcept
        {
            return statistics_;
        }

        OcpPhaseCondensingFailureLocation
        last_failure_location() const noexcept
        {
            return failure_location_;
        }

        MatRealView condensed_matrix()
        {
            return condensed_.block(columns_, columns_, 0, 0);
        }

        VecRealView condensed_rhs()
        {
            return condensed_rhs_.block(columns_, 0);
        }

        LinsolReturnFlag factor_and_condense(
            const OcpPhaseCondensingViewTpl<ProblemType> &phase)
        {
            validate(phase);
            factorization_ready_ = false;
            statistics_ = {};
            statistics_.incident_response_columns = columns_;
            failure_location_ = OcpPhaseCondensingFailureLocation::None;
            base_primal_ = 0.0;
            base_multipliers_ = 0.0;
            condensed_matrix() = 0.0;
            condensed_rhs() = 0.0;

            const auto factor_start = Clock::now();
            const LinsolReturnFlag factor_status =
                phase.equality_dual_diagonal_is_zero
                ? solver_->solve(
                    phase.info, phase.jacobian, phase.hessian,
                    phase.D_x, phase.D_s,
                    phase.rhs_primal, phase.rhs_constraints,
                    base_primal_, base_multipliers_)
                : solver_->solve(
                    phase.info, phase.jacobian, phase.hessian,
                    phase.D_x, phase.D_eq, phase.D_s,
                    phase.rhs_primal, phase.rhs_constraints,
                    base_primal_, base_multipliers_);
            if (factor_status != LinsolReturnFlag::SUCCESS)
            {
                statistics_.factorization_milliseconds =
                    elapsed_milliseconds(factor_start);
                failure_location_ =
                    OcpPhaseCondensingFailureLocation::TrajectoryFactorization;
                return factor_status;
            }
            const LinsolReturnFlag base_refinement = refine(
                phase,
                phase.rhs_primal,
                phase.rhs_constraints,
                base_primal_,
                base_multipliers_);
            statistics_.factorization_milliseconds =
                elapsed_milliseconds(factor_start);
            if (base_refinement != LinsolReturnFlag::SUCCESS)
            {
                failure_location_ =
                    OcpPhaseCondensingFailureLocation::TrajectoryResponse;
                return base_refinement;
            }

            const auto response_start = Clock::now();
            const LinsolReturnFlag response_status = solve_responses(phase);
            statistics_.response_milliseconds =
                elapsed_milliseconds(response_start);
            if (response_status != LinsolReturnFlag::SUCCESS)
            {
                failure_location_ =
                    OcpPhaseCondensingFailureLocation::TrajectoryResponse;
                return response_status;
            }

            const auto condensation_start = Clock::now();
            assemble_condensation(phase);
            statistics_.condensation_milliseconds =
                elapsed_milliseconds(condensation_start);
            factorization_ready_ = true;
            return LinsolReturnFlag::SUCCESS;
        }

        /**
         * Re-condense a new phase RHS using the trajectory factorization and
         * border responses retained by `factor_and_condense()`.
         */
        LinsolReturnFlag condense_rhs(
            const OcpPhaseCondensingViewTpl<ProblemType> &phase)
        {
            validate(phase);
            fatrop_assert_msg(
                factorization_ready_,
                "condense_rhs requires a retained phase factorization.");
            statistics_ = {};
            statistics_.incident_response_columns = columns_;
            statistics_.active_response_columns =
                static_cast<Index>(active_columns_.size());
            failure_location_ = OcpPhaseCondensingFailureLocation::None;
            base_primal_ = 0.0;
            base_multipliers_ = 0.0;
            condensed_rhs() = 0.0;

            const auto response_start = Clock::now();
            const LinsolReturnFlag status =
                phase.equality_dual_diagonal_is_zero
                ? solver_->solve_rhs(
                    phase.info, phase.jacobian, phase.hessian,
                    phase.D_s,
                    phase.rhs_primal, phase.rhs_constraints,
                    base_primal_, base_multipliers_)
                : solver_->solve_rhs(
                    phase.info, phase.jacobian, phase.hessian,
                    phase.D_eq, phase.D_s,
                    phase.rhs_primal, phase.rhs_constraints,
                    base_primal_, base_multipliers_);
            if (status != LinsolReturnFlag::SUCCESS)
            {
                statistics_.response_milliseconds =
                    elapsed_milliseconds(response_start);
                failure_location_ =
                    OcpPhaseCondensingFailureLocation::TrajectoryResponse;
                return status;
            }
            const LinsolReturnFlag refinement = refine(
                phase,
                phase.rhs_primal,
                phase.rhs_constraints,
                base_primal_, base_multipliers_);
            statistics_.response_milliseconds =
                elapsed_milliseconds(response_start);
            if (refinement != LinsolReturnFlag::SUCCESS)
            {
                failure_location_ =
                    OcpPhaseCondensingFailureLocation::TrajectoryResponse;
                return refinement;
            }

            const auto condensation_start = Clock::now();
            assemble_condensation(phase);
            statistics_.condensation_milliseconds =
                elapsed_milliseconds(condensation_start);
            return LinsolReturnFlag::SUCCESS;
        }

        void back_substitute(
            const OcpPhaseCondensingViewTpl<ProblemType> &phase,
            const VecRealView &scoped_solution) const
        {
            fatrop_assert_msg(
                scoped_solution.m() == columns_,
                "A phase-condensing scoped solution has a wrong size.");
            for (Index row = 0;
                 row < phase.info.number_of_primal_variables; ++row)
            {
                Scalar value = base_primal_(row);
                for (Index column = 0; column < columns_; ++column)
                {
                    value += response_primal_(row, column)
                           * scoped_solution(column);
                }
                phase.primal(row) = value;
            }
            for (Index row = 0;
                 row < phase.info.number_of_eq_constraints; ++row)
            {
                Scalar value = base_multipliers_(row);
                for (Index column = 0; column < columns_; ++column)
                {
                    value += response_multipliers_(row, column)
                           * scoped_solution(column);
                }
                phase.multipliers(row) = value;
            }
        }

    private:
        using Clock = std::chrono::steady_clock;

        static double elapsed_milliseconds(
            const Clock::time_point start) noexcept
        {
            return std::chrono::duration<double, std::milli>(
                Clock::now() - start).count();
        }

        void validate(
            const OcpPhaseCondensingViewTpl<ProblemType> &phase) const
        {
            fatrop_assert_msg(
                phase.cross_hessian.m()
                    == phase.info.number_of_primal_variables
                && phase.cross_hessian.n() == columns_,
                "A phase-condensing cross Hessian has a wrong shape.");
            fatrop_assert_msg(
                phase.border_jacobian.m()
                    == phase.info.number_of_eq_constraints
                && phase.border_jacobian.n() == columns_,
                "A phase-condensing border Jacobian has a wrong shape.");
            fatrop_assert_msg(
                phase.D_x.m() == phase.info.number_of_primal_variables
                && phase.D_s.m() == phase.info.number_of_slack_variables,
                "A phase-condensing regularization vector has a wrong size.");
            fatrop_assert_msg(
                phase.equality_dual_diagonal_is_zero
                ? (phase.D_eq.m() == 0
                   || phase.D_eq.m()
                      == phase.info.number_of_g_eq_path)
                : phase.D_eq.m() == phase.info.number_of_g_eq_path,
                "A phase-condensing equality-dual diagonal has a wrong size.");
            fatrop_assert_msg(
                phase.rhs_primal.m()
                    == phase.info.number_of_primal_variables
                && phase.primal.m()
                    == phase.info.number_of_primal_variables
                && phase.rhs_constraints.m()
                    == phase.info.number_of_eq_constraints
                && phase.multipliers.m()
                    == phase.info.number_of_eq_constraints,
                "A phase-condensing KKT vector has a wrong size.");
        }

        LinsolReturnFlag solve_responses(
            const OcpPhaseCondensingViewTpl<ProblemType> &phase)
        {
            MatRealView response_primal = response_primal_.block(
                phase.info.number_of_primal_variables,
                columns_, 0, 0);
            MatRealView response_multipliers = response_multipliers_.block(
                phase.info.number_of_eq_constraints,
                columns_, 0, 0);
            response_primal = 0.0;
            response_multipliers = 0.0;
            active_columns_.clear();
            std::fill(
                active_column_position_.begin(),
                active_column_position_.end(), -1);

            for (Index column = 0; column < columns_; ++column)
            {
                bool active = false;
                for (Index row = 0;
                     row < phase.info.number_of_primal_variables; ++row)
                    active = active
                        || phase.cross_hessian(row, column) != 0.0;
                for (Index row = 0;
                     row < phase.info.number_of_eq_constraints; ++row)
                    active = active
                        || phase.border_jacobian(row, column) != 0.0;
                if (active)
                {
                    active_column_position_[
                        static_cast<std::size_t>(column)] =
                        static_cast<Index>(active_columns_.size());
                    active_columns_.push_back(column);
                }
            }

            const Index active_count =
                static_cast<Index>(active_columns_.size());
            statistics_.active_response_columns = active_count;
            if (active_count == 0)
                return LinsolReturnFlag::SUCCESS;

            MatRealView compact_rhs_primal = compact_rhs_primal_.block(
                phase.info.number_of_primal_variables,
                active_count, 0, 0);
            MatRealView compact_rhs_constraints =
                compact_rhs_constraints_.block(
                    phase.info.number_of_eq_constraints,
                    active_count, 0, 0);
            MatRealView compact_response_primal =
                compact_response_primal_.block(
                    phase.info.number_of_primal_variables,
                    active_count, 0, 0);
            MatRealView compact_response_multipliers =
                compact_response_multipliers_.block(
                    phase.info.number_of_eq_constraints,
                    active_count, 0, 0);
            compact_response_primal = 0.0;
            compact_response_multipliers = 0.0;
            for (Index compact = 0; compact < active_count; ++compact)
            {
                const Index source = active_columns_[
                    static_cast<std::size_t>(compact)];
                for (Index row = 0;
                     row < phase.info.number_of_primal_variables; ++row)
                    compact_rhs_primal(row, compact) =
                        phase.cross_hessian(row, source);
                for (Index row = 0;
                     row < phase.info.number_of_eq_constraints; ++row)
                    compact_rhs_constraints(row, compact) =
                        phase.border_jacobian(row, source);
            }

            bool use_batch = false;
            if constexpr (std::is_same_v<ProblemType, ImplicitOcpType>)
            {
                use_batch = true;
                for (Index stage = 0;
                     stage + 1 < phase.info.dims.K; ++stage)
                {
                    use_batch = use_batch
                        && phase.jacobian.J_ranks[stage]
                           == phase.info.dims.number_of_states[stage + 1];
                }
            }

            if (use_batch)
            {
                const LinsolReturnFlag status =
                    phase.equality_dual_diagonal_is_zero
                    ? solver_->solve_rhs_batch(
                        phase.info, phase.jacobian, phase.hessian,
                        phase.D_s,
                        compact_rhs_primal,
                        compact_rhs_constraints,
                        compact_response_primal,
                        compact_response_multipliers)
                    : solver_->solve_rhs_batch(
                        phase.info, phase.jacobian, phase.hessian,
                        phase.D_eq, phase.D_s,
                        compact_rhs_primal,
                        compact_rhs_constraints,
                        compact_response_primal,
                        compact_response_multipliers);
                if (status != LinsolReturnFlag::SUCCESS)
                    return status;

                for (Index compact = 0;
                     compact < active_count; ++compact)
                {
                    for (Index row = 0;
                         row < phase.info.number_of_primal_variables; ++row)
                    {
                        column_rhs_primal_(row) =
                            compact_rhs_primal(row, compact);
                        column_primal_(row) =
                            compact_response_primal(row, compact);
                    }
                    for (Index row = 0;
                         row < phase.info.number_of_eq_constraints; ++row)
                    {
                        column_rhs_constraints_(row) =
                            compact_rhs_constraints(row, compact);
                        column_multipliers_(row) =
                            compact_response_multipliers(row, compact);
                    }
                    const LinsolReturnFlag refinement = refine(
                        phase,
                        column_rhs_primal_,
                        column_rhs_constraints_,
                        column_primal_,
                        column_multipliers_);
                    if (refinement != LinsolReturnFlag::SUCCESS)
                        return refinement;
                    const Index destination = active_columns_[
                        static_cast<std::size_t>(compact)];
                    for (Index row = 0;
                         row < phase.info.number_of_primal_variables; ++row)
                    {
                        response_primal(row, destination) =
                            column_primal_(row);
                        compact_response_primal(row, compact) =
                            column_primal_(row);
                    }
                    for (Index row = 0;
                         row < phase.info.number_of_eq_constraints; ++row)
                    {
                        response_multipliers(row, destination) =
                            column_multipliers_(row);
                        compact_response_multipliers(row, compact) =
                            column_multipliers_(row);
                    }
                }
                return LinsolReturnFlag::SUCCESS;
            }

            for (Index compact = 0; compact < active_count; ++compact)
            {
                const Index column = active_columns_[
                    static_cast<std::size_t>(compact)];
                for (Index row = 0;
                     row < phase.info.number_of_primal_variables; ++row)
                    column_rhs_primal_(row) =
                        phase.cross_hessian(row, column);
                for (Index row = 0;
                     row < phase.info.number_of_eq_constraints; ++row)
                    column_rhs_constraints_(row) =
                        phase.border_jacobian(row, column);
                column_primal_ = 0.0;
                column_multipliers_ = 0.0;
                const LinsolReturnFlag status =
                    phase.equality_dual_diagonal_is_zero
                    ? solver_->solve_rhs(
                        phase.info, phase.jacobian, phase.hessian,
                        phase.D_s,
                        column_rhs_primal_,
                        column_rhs_constraints_,
                        column_primal_, column_multipliers_)
                    : solver_->solve_rhs(
                        phase.info, phase.jacobian, phase.hessian,
                        phase.D_eq, phase.D_s,
                        column_rhs_primal_,
                        column_rhs_constraints_,
                        column_primal_, column_multipliers_);
                if (status != LinsolReturnFlag::SUCCESS)
                    return status;
                const LinsolReturnFlag refinement = refine(
                    phase,
                    column_rhs_primal_,
                    column_rhs_constraints_,
                    column_primal_,
                    column_multipliers_);
                if (refinement != LinsolReturnFlag::SUCCESS)
                    return refinement;
                for (Index row = 0;
                     row < phase.info.number_of_primal_variables; ++row)
                {
                    response_primal(row, column) = column_primal_(row);
                    compact_response_primal(row, compact) =
                        column_primal_(row);
                }
                for (Index row = 0;
                     row < phase.info.number_of_eq_constraints; ++row)
                {
                    response_multipliers(row, column) =
                        column_multipliers_(row);
                    compact_response_multipliers(row, compact) =
                        column_multipliers_(row);
                }
            }
            return LinsolReturnFlag::SUCCESS;
        }

        LinsolReturnFlag refine(
            const OcpPhaseCondensingViewTpl<ProblemType> &phase,
            const VecRealView &rhs_primal,
            const VecRealView &rhs_constraints,
            VecRealView &primal,
            VecRealView &multipliers)
        {
            constexpr Index maximum_refinements = 3;
            constexpr Scalar backward_tolerance = 5e-12;
            for (Index iteration = 0;
                 iteration < maximum_refinements; ++iteration)
            {
                residual_primal_ = 0.0;
                residual_constraints_ = 0.0;
                hessian_scratch_ = 0.0;
                jacobian_scratch_ = 0.0;
                jacobian_transpose_product_ = 0.0;

                phase.hessian.apply_on_right(
                    phase.info, primal, 0.0,
                    hessian_scratch_, residual_primal_);
                phase.jacobian.transpose_apply_on_right(
                    phase.info, multipliers, 0.0,
                    hessian_scratch_, jacobian_transpose_product_);
                residual_primal_ =
                    residual_primal_
                    + phase.D_x * primal
                    + jacobian_transpose_product_
                    + rhs_primal;

                phase.jacobian.apply_on_right(
                    phase.info, primal, 0.0,
                    jacobian_scratch_, residual_constraints_);
                residual_constraints_ =
                    residual_constraints_ + rhs_constraints;
                if (!phase.equality_dual_diagonal_is_zero
                    && phase.info.number_of_g_eq_path > 0)
                {
                    residual_constraints_.block(
                        phase.info.number_of_g_eq_path,
                        phase.info.offset_g_eq_path) =
                        residual_constraints_.block(
                            phase.info.number_of_g_eq_path,
                            phase.info.offset_g_eq_path)
                        - phase.D_eq
                          * multipliers.block(
                              phase.info.number_of_g_eq_path,
                              phase.info.offset_g_eq_path);
                }
                if (phase.info.number_of_slack_variables > 0)
                {
                    residual_constraints_.block(
                        phase.info.number_of_slack_variables,
                        phase.info.offset_g_eq_slack) =
                        residual_constraints_.block(
                            phase.info.number_of_slack_variables,
                            phase.info.offset_g_eq_slack)
                        - phase.D_s
                          * multipliers.block(
                              phase.info.number_of_slack_variables,
                              phase.info.offset_g_eq_slack);
                }

                const Scalar residual_norm = std::max(
                    norm_inf(residual_primal_),
                    norm_inf(residual_constraints_));
                const Scalar scale =
                    1.0
                    + norm_inf(rhs_primal)
                    + norm_inf(rhs_constraints)
                    + norm_inf(primal)
                    + norm_inf(multipliers);
                if (residual_norm <= backward_tolerance * scale)
                    return LinsolReturnFlag::SUCCESS;

                correction_primal_ = 0.0;
                correction_multipliers_ = 0.0;
                const LinsolReturnFlag status =
                    phase.equality_dual_diagonal_is_zero
                    ? solver_->solve_rhs(
                        phase.info, phase.jacobian, phase.hessian,
                        phase.D_s,
                        residual_primal_, residual_constraints_,
                        correction_primal_, correction_multipliers_)
                    : solver_->solve_rhs(
                        phase.info, phase.jacobian, phase.hessian,
                        phase.D_eq, phase.D_s,
                        residual_primal_, residual_constraints_,
                        correction_primal_, correction_multipliers_);
                if (status != LinsolReturnFlag::SUCCESS)
                    return status;
                primal = primal + correction_primal_;
                multipliers = multipliers + correction_multipliers_;
            }
            return LinsolReturnFlag::SUCCESS;
        }

        void assemble_condensation(
            const OcpPhaseCondensingViewTpl<ProblemType> &phase)
        {
            MatRealView full_condensed = condensed_matrix();
            VecRealView full_rhs = condensed_rhs();
            full_condensed = 0.0;
            full_rhs = 0.0;
            const Index active_count =
                static_cast<Index>(active_columns_.size());
            if (active_count == 0)
                return;

            MatRealView compact_rhs_primal = compact_rhs_primal_.block(
                phase.info.number_of_primal_variables,
                active_count, 0, 0);
            MatRealView compact_rhs_constraints =
                compact_rhs_constraints_.block(
                    phase.info.number_of_eq_constraints,
                    active_count, 0, 0);
            MatRealView compact_response_primal =
                compact_response_primal_.block(
                    phase.info.number_of_primal_variables,
                    active_count, 0, 0);
            MatRealView compact_response_multipliers =
                compact_response_multipliers_.block(
                    phase.info.number_of_eq_constraints,
                    active_count, 0, 0);
            MatRealView compact_condensed = compact_condensed_.block(
                active_count, active_count, 0, 0);
            VecRealView compact_rhs = compact_condensed_rhs_.block(
                active_count, 0);
            compact_condensed = 0.0;
            compact_rhs = 0.0;

            if (phase.info.number_of_primal_variables > 0)
            {
                gemm_tn(
                    active_count, active_count,
                    phase.info.number_of_primal_variables,
                    1.0, compact_rhs_primal, 0, 0,
                    compact_response_primal, 0, 0,
                    1.0, compact_condensed, 0, 0,
                    compact_condensed, 0, 0);
                gemv_t(
                    phase.info.number_of_primal_variables,
                    active_count,
                    1.0, compact_rhs_primal, 0, 0,
                    base_primal_, 0,
                    1.0, compact_rhs, 0,
                    compact_rhs, 0);
            }
            if (phase.info.number_of_eq_constraints > 0)
            {
                gemm_tn(
                    active_count, active_count,
                    phase.info.number_of_eq_constraints,
                    1.0, compact_rhs_constraints, 0, 0,
                    compact_response_multipliers, 0, 0,
                    1.0, compact_condensed, 0, 0,
                    compact_condensed, 0, 0);
                gemv_t(
                    phase.info.number_of_eq_constraints,
                    active_count,
                    1.0, compact_rhs_constraints, 0, 0,
                    base_multipliers_, 0,
                    1.0, compact_rhs, 0,
                    compact_rhs, 0);
            }

            for (Index row = 0; row < active_count; ++row)
            {
                const Index destination_row = active_columns_[
                    static_cast<std::size_t>(row)];
                full_rhs(destination_row) = compact_rhs(row);
                for (Index column = 0; column < active_count; ++column)
                {
                    const Index destination_column = active_columns_[
                        static_cast<std::size_t>(column)];
                    full_condensed(destination_row, destination_column) =
                        compact_condensed(row, column);
                }
            }
        }

        Index columns_ = 0;
        std::unique_ptr<AugSystemSolver<ProblemType>> solver_;
        VecRealAllocated base_primal_;
        VecRealAllocated base_multipliers_;
        MatRealAllocated response_primal_;
        MatRealAllocated response_multipliers_;
        MatRealAllocated compact_rhs_primal_;
        MatRealAllocated compact_rhs_constraints_;
        MatRealAllocated compact_response_primal_;
        MatRealAllocated compact_response_multipliers_;
        MatRealAllocated compact_condensed_;
        VecRealAllocated compact_condensed_rhs_;
        MatRealAllocated condensed_;
        VecRealAllocated condensed_rhs_;
        VecRealAllocated column_rhs_primal_;
        VecRealAllocated column_rhs_constraints_;
        VecRealAllocated column_primal_;
        VecRealAllocated column_multipliers_;
        VecRealAllocated residual_primal_;
        VecRealAllocated residual_constraints_;
        VecRealAllocated correction_primal_;
        VecRealAllocated correction_multipliers_;
        VecRealAllocated hessian_scratch_;
        VecRealAllocated jacobian_scratch_;
        VecRealAllocated jacobian_transpose_product_;
        std::vector<Index> active_column_position_;
        std::vector<Index> active_columns_;
        OcpPhaseCondensingStatistics statistics_;
        OcpPhaseCondensingFailureLocation failure_location_ =
            OcpPhaseCondensingFailureLocation::None;
        bool factorization_ready_ = false;
    };

    using OcpPhaseCondensingView =
        OcpPhaseCondensingViewTpl<OcpType>;
    using ImplicitOcpPhaseCondensingView =
        OcpPhaseCondensingViewTpl<ImplicitOcpType>;
    using OcpPhaseCondensingKernel =
        OcpPhaseCondensingKernelTpl<OcpType>;
    using ImplicitOcpPhaseCondensingKernel =
        OcpPhaseCondensingKernelTpl<ImplicitOcpType>;
} // namespace fatrop

#endif // __fatrop_ocp_ocp_phase_condensing_hpp__
