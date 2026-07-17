//
// Copyright (c) 2026
//

#ifndef __fatrop_ocp_phase_arrow_ocp_kkt_solver_hpp__
#define __fatrop_ocp_phase_arrow_ocp_kkt_solver_hpp__

#include "fatrop/common/exception.hpp"
#include "fatrop/context/context.hpp"
#include "fatrop/linear_algebra/blasfeo_operations.hpp"
#include "fatrop/linear_algebra/linear_solver_return_flags.hpp"
#include "fatrop/linear_algebra/matrix.hpp"
#include "fatrop/linear_algebra/vector.hpp"
#include "fatrop/ocp/aug_system_solver.hpp"
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/phase_arrow_kkt_solver.hpp"
#include "fatrop/ocp/problem_info.hpp"
#include "fatrop/ocp/type.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace fatrop
{
    struct PhaseArrowOcpKktStatistics
    {
        Index incident_response_columns = 0;
        Index active_response_columns = 0;
        double trajectory_factorization_milliseconds = 0.0;
        double trajectory_response_milliseconds = 0.0;
        double condensation_milliseconds = 0.0;
        double reduced_solve_milliseconds = 0.0;
        double back_substitution_milliseconds = 0.0;
    };

    enum class PhaseArrowOcpKktFailureLocation
    {
        None,
        TrajectoryFactorization,
        TrajectoryResponse,
        ReducedPhaseArrow,
        NonfiniteSolution
    };

    /**
     * @brief Non-owning data for one independently condensable OCP phase.
     *
     * The columns of `cross_hessian` and `border_jacobian` are ordered as
     *
     *     [ q_(f-1) | q_f | g ],
     *
     * omitting q_(f-1) for the first phase.  This incidence is what makes the
     * condensed phase system block tridiagonal instead of globally dense.
     */
    template <typename ProblemType>
    struct PhaseArrowOcpPhaseViewTpl
    {
        PhaseArrowOcpPhaseViewTpl(
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
        {}

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

    /**
     * @brief Two-level OCP KKT solver for phase-local static variables.
     *
     * Every trajectory block K_f is factored independently by FATROP.  Its
     * base solution and responses to [q_(f-1), q_f, g] are condensed into a
     * heterogeneous phase-arrow system, solved by PhaseArrowKktSolver, and
     * then back-substituted.  This is algebraically equivalent to solving the
     * complete KKT matrix, while avoiding a dense Schur block containing all
     * phase-local variables at once.
     */
    template <typename ProblemType>
    class PhaseArrowOcpKktSolverTpl
    {
    public:
        static_assert(
            std::is_same_v<ProblemType, OcpType>
            || std::is_same_v<ProblemType, ImplicitOcpType>,
            "PhaseArrowOcpKktSolverTpl supports OcpType and ImplicitOcpType.");

        PhaseArrowOcpKktSolverTpl(
            const std::vector<const ProblemInfo *> &phase_info,
            std::vector<Index> phase_block_sizes,
            const Index number_of_global_variables)
            : phase_block_sizes_(std::move(phase_block_sizes)),
              number_of_global_variables_(number_of_global_variables),
              reduced_solver_(
                  phase_block_sizes_, number_of_global_variables_)
        {
            fatrop_assert_msg(
                phase_info.size() == phase_block_sizes_.size(),
                "The OCP phase count and reduced phase-block count differ.");
            workspaces_.reserve(phase_info.size());
            trajectory_solvers_.reserve(phase_info.size());
            reduced_diagonal_.reserve(phase_info.size());
            reduced_arrow_.reserve(phase_info.size());
            reduced_rhs_phase_.reserve(phase_info.size());
            for (std::size_t phase = 0;
                 phase < phase_info.size(); ++phase)
            {
                fatrop_assert_msg(
                    phase_info[phase] != nullptr,
                    "A phase-arrow OCP ProblemInfo pointer is null.");
                const Index left = phase == 0
                    ? 0 : phase_block_sizes_[phase - 1];
                const Index columns = left
                    + phase_block_sizes_[phase]
                    + number_of_global_variables_;
                trajectory_solvers_.push_back(
                    std::make_unique<AugSystemSolver<ProblemType>>(
                        *phase_info[phase]));
                workspaces_.emplace_back(*phase_info[phase], columns);
                reduced_diagonal_.emplace_back(
                    phase_block_sizes_[phase],
                    phase_block_sizes_[phase]);
                reduced_arrow_.emplace_back(
                    phase_block_sizes_[phase],
                    number_of_global_variables_);
                reduced_rhs_phase_.emplace_back(
                    phase_block_sizes_[phase]);
            }
            reduced_coupling_.reserve(
                phase_info.empty() ? 0 : phase_info.size() - 1);
            for (std::size_t phase = 0;
                 phase + 1 < phase_info.size(); ++phase)
            {
                reduced_coupling_.emplace_back(
                    phase_block_sizes_[phase],
                    phase_block_sizes_[phase + 1]);
            }
            const Index global_storage =
                std::max<Index>(number_of_global_variables_, 1);
            reduced_global_storage_ = std::make_unique<MatRealAllocated>(
                global_storage, global_storage);
            reduced_rhs_global_storage_ =
                std::make_unique<VecRealAllocated>(global_storage);
        }

        PhaseArrowKktSolver &reduced_solver() noexcept
        {
            return reduced_solver_;
        }

        const PhaseArrowKktSolver &reduced_solver() const noexcept
        {
            return reduced_solver_;
        }

        AugSystemSolver<ProblemType> &trajectory_solver(
            const Index phase)
        {
            fatrop_assert_msg(
                phase >= 0
                && phase < static_cast<Index>(trajectory_solvers_.size()),
                "The phase-arrow trajectory solver index is out of range.");
            return *trajectory_solvers_[static_cast<std::size_t>(phase)];
        }

        PhaseArrowOcpKktFailureLocation last_failure_location() const noexcept
        {
            return last_failure_location_;
        }

        Index last_failure_phase() const noexcept
        {
            return last_failure_phase_;
        }

        const PhaseArrowOcpKktStatistics &last_statistics() const noexcept
        {
            return last_statistics_;
        }

        /**
         * @brief Solve the complete two-level KKT system.
         *
         * The FATROP residual convention is used: all `rhs_*` inputs are KKT
         * residuals and the returned step solves A step = -rhs.
         */
        LinsolReturnFlag solve(
            const std::vector<PhaseArrowOcpPhaseViewTpl<ProblemType>> &phases,
            const std::vector<MatRealView> &phase_diagonal,
            const std::vector<MatRealView> &phase_coupling,
            const std::vector<MatRealView> &global_arrow,
            const MatRealView &global_hessian,
            const std::vector<VecRealView> &rhs_phase,
            const VecRealView &rhs_global,
            const std::vector<VecRealView> &phase_solution,
            const VecRealView &global_solution)
        {
            validate_dimensions(
                phases, phase_diagonal, phase_coupling,
                global_arrow, global_hessian,
                rhs_phase, rhs_global,
                phase_solution, global_solution);
            last_failure_location_ = PhaseArrowOcpKktFailureLocation::None;
            last_failure_phase_ = -1;
            last_statistics_ = {};

            for (std::size_t phase = 0; phase < phases.size(); ++phase)
            {
                PhaseWorkspace &workspace = workspaces_[phase];
                workspace.base_primal = 0.0;
                workspace.base_multipliers = 0.0;
                const auto &current = phases[phase];
                const auto factorization_start = Clock::now();
                const LinsolReturnFlag status =
                    current.equality_dual_diagonal_is_zero
                    ? trajectory_solvers_[phase]->solve(
                        current.info,
                        current.jacobian,
                        current.hessian,
                        current.D_x,
                        current.D_s,
                        current.rhs_primal,
                        current.rhs_constraints,
                        workspace.base_primal,
                        workspace.base_multipliers)
                    : trajectory_solvers_[phase]->solve(
                        current.info,
                        current.jacobian,
                        current.hessian,
                        current.D_x,
                        current.D_eq,
                        current.D_s,
                        current.rhs_primal,
                        current.rhs_constraints,
                        workspace.base_primal,
                        workspace.base_multipliers);
                if (status != LinsolReturnFlag::SUCCESS)
                {
                    last_statistics_.trajectory_factorization_milliseconds +=
                        elapsed_milliseconds(factorization_start);
                    last_failure_location_ =
                        PhaseArrowOcpKktFailureLocation::TrajectoryFactorization;
                    last_failure_phase_ = static_cast<Index>(phase);
                    return status;
                }
                const LinsolReturnFlag refinement_status =
                    refine_trajectory_solution(
                        phase, current,
                        current.rhs_primal,
                        current.rhs_constraints,
                        workspace.base_primal,
                        workspace.base_multipliers);
                if (refinement_status != LinsolReturnFlag::SUCCESS)
                {
                    last_statistics_.trajectory_factorization_milliseconds +=
                        elapsed_milliseconds(factorization_start);
                    last_failure_location_ =
                        PhaseArrowOcpKktFailureLocation::TrajectoryResponse;
                    last_failure_phase_ = static_cast<Index>(phase);
                    return refinement_status;
                }
                // The base solve is not complete until refinement has reached
                // the backward-error target.  Keep that work in the reported
                // trajectory factorization cost instead of silently omitting
                // it from the phase timing.
                last_statistics_.trajectory_factorization_milliseconds +=
                    elapsed_milliseconds(factorization_start);
                const auto response_start = Clock::now();
                const LinsolReturnFlag response_status =
                    solve_responses(phase, current);
                last_statistics_.trajectory_response_milliseconds +=
                    elapsed_milliseconds(response_start);
                if (response_status != LinsolReturnFlag::SUCCESS)
                {
                    last_failure_location_ =
                        PhaseArrowOcpKktFailureLocation::TrajectoryResponse;
                    last_failure_phase_ = static_cast<Index>(phase);
                    return response_status;
                }
            }

            const auto condensation_start = Clock::now();
            for (std::size_t phase = 0; phase < phases.size(); ++phase)
                assemble_compact_condensation(phase, phases[phase]);
            initialize_reduced_system(
                phase_diagonal, phase_coupling,
                global_arrow, global_hessian,
                rhs_phase, rhs_global);
            for (std::size_t phase = 0; phase < phases.size(); ++phase)
                add_phase_condensation(phase, phases[phase]);
            last_statistics_.condensation_milliseconds +=
                elapsed_milliseconds(condensation_start);

            std::vector<MatRealView> reduced_diagonal_views;
            std::vector<MatRealView> reduced_coupling_views;
            std::vector<MatRealView> reduced_arrow_views;
            std::vector<VecRealView> reduced_rhs_views;
            reduced_diagonal_views.reserve(reduced_diagonal_.size());
            reduced_coupling_views.reserve(reduced_coupling_.size());
            reduced_arrow_views.reserve(reduced_arrow_.size());
            reduced_rhs_views.reserve(reduced_rhs_phase_.size());
            for (MatRealAllocated &block : reduced_diagonal_)
                reduced_diagonal_views.push_back(
                    block.block(block.m(), block.n(), 0, 0));
            for (MatRealAllocated &block : reduced_coupling_)
                reduced_coupling_views.push_back(
                    block.block(block.m(), block.n(), 0, 0));
            for (MatRealAllocated &block : reduced_arrow_)
                reduced_arrow_views.push_back(
                    block.block(block.m(), block.n(), 0, 0));
            for (VecRealAllocated &block : reduced_rhs_phase_)
                reduced_rhs_views.push_back(block.block(block.m(), 0));

            MatRealView reduced_global =
                reduced_global_storage_->block(
                    number_of_global_variables_,
                    number_of_global_variables_, 0, 0);
            VecRealView reduced_rhs_global =
                reduced_rhs_global_storage_->block(
                    number_of_global_variables_, 0);
            const auto reduced_start = Clock::now();
            const LinsolReturnFlag reduced_status =
                reduced_solver_.solve(
                    reduced_diagonal_views,
                    reduced_coupling_views,
                    reduced_arrow_views,
                    reduced_global,
                    reduced_rhs_views,
                    reduced_rhs_global,
                    phase_solution,
                    global_solution);
            last_statistics_.reduced_solve_milliseconds +=
                elapsed_milliseconds(reduced_start);
            if (reduced_status != LinsolReturnFlag::SUCCESS)
            {
                last_failure_location_ =
                    PhaseArrowOcpKktFailureLocation::ReducedPhaseArrow;
                last_failure_phase_ =
                    reduced_solver_.last_failure_phase();
                return reduced_status;
            }

            const auto back_substitution_start = Clock::now();
            for (std::size_t phase = 0; phase < phases.size(); ++phase)
                back_substitute(phase, phases[phase], phase_solution, global_solution);
            last_statistics_.back_substitution_milliseconds +=
                elapsed_milliseconds(back_substitution_start);
            for (const auto &phase : phases)
            {
                if (!phase.primal.is_finite()
                || !phase.multipliers.is_finite())
                {
                    last_failure_location_ =
                        PhaseArrowOcpKktFailureLocation::NonfiniteSolution;
                    return LinsolReturnFlag::NAN_SOLUTION;
                }
            }
            return LinsolReturnFlag::SUCCESS;
        }

    private:
        using Clock = std::chrono::steady_clock;

        static double elapsed_milliseconds(
            const Clock::time_point start) noexcept
        {
            return std::chrono::duration<double, std::milli>(
                Clock::now() - start).count();
        }

        struct PhaseWorkspace
        {
            PhaseWorkspace(
                const ProblemInfo &info,
                const Index columns_value)
                : columns(columns_value),
                  base_primal(info.number_of_primal_variables),
                  base_multipliers(info.number_of_eq_constraints),
                  response_primal(
                      info.number_of_primal_variables,
                      std::max<Index>(columns, 1)),
                  response_multipliers(
                      info.number_of_eq_constraints,
                      std::max<Index>(columns, 1)),
                  compact_rhs_primal(
                      info.number_of_primal_variables,
                      std::max<Index>(columns, 1)),
                  compact_rhs_constraints(
                      info.number_of_eq_constraints,
                      std::max<Index>(columns, 1)),
                  compact_response_primal(
                      info.number_of_primal_variables,
                      std::max<Index>(columns, 1)),
                  compact_response_multipliers(
                      info.number_of_eq_constraints,
                      std::max<Index>(columns, 1)),
                  compact_condensed(
                      std::max<Index>(columns, 1),
                      std::max<Index>(columns, 1)),
                  compact_condensed_rhs(
                      std::max<Index>(columns, 1)),
                  column_rhs_primal(info.number_of_primal_variables),
                  column_rhs_constraints(info.number_of_eq_constraints),
                  column_primal(info.number_of_primal_variables),
                  column_multipliers(info.number_of_eq_constraints),
                  residual_primal(info.number_of_primal_variables),
                  residual_constraints(info.number_of_eq_constraints),
                  correction_primal(info.number_of_primal_variables),
                  correction_multipliers(info.number_of_eq_constraints),
                  hessian_scratch(info.number_of_primal_variables),
                  jacobian_scratch(info.number_of_eq_constraints),
                  jacobian_transpose_product(
                      info.number_of_primal_variables),
                  active_column_mask(
                      static_cast<std::size_t>(columns), 0),
                  active_column_position(
                      static_cast<std::size_t>(columns), -1)
            {
                active_columns.reserve(
                    static_cast<std::size_t>(columns));
            }

            Index columns = 0;
            VecRealAllocated base_primal;
            VecRealAllocated base_multipliers;
            MatRealAllocated response_primal;
            MatRealAllocated response_multipliers;
            MatRealAllocated compact_rhs_primal;
            MatRealAllocated compact_rhs_constraints;
            MatRealAllocated compact_response_primal;
            MatRealAllocated compact_response_multipliers;
            MatRealAllocated compact_condensed;
            VecRealAllocated compact_condensed_rhs;
            VecRealAllocated column_rhs_primal;
            VecRealAllocated column_rhs_constraints;
            VecRealAllocated column_primal;
            VecRealAllocated column_multipliers;
            VecRealAllocated residual_primal;
            VecRealAllocated residual_constraints;
            VecRealAllocated correction_primal;
            VecRealAllocated correction_multipliers;
            VecRealAllocated hessian_scratch;
            VecRealAllocated jacobian_scratch;
            VecRealAllocated jacobian_transpose_product;
            std::vector<unsigned char> active_column_mask;
            std::vector<Index> active_column_position;
            std::vector<Index> active_columns;
        };

        void validate_dimensions(
            const std::vector<PhaseArrowOcpPhaseViewTpl<ProblemType>> &phases,
            const std::vector<MatRealView> &phase_diagonal,
            const std::vector<MatRealView> &phase_coupling,
            const std::vector<MatRealView> &global_arrow,
            const MatRealView &global_hessian,
            const std::vector<VecRealView> &rhs_phase,
            const VecRealView &rhs_global,
            const std::vector<VecRealView> &phase_solution,
            const VecRealView &global_solution) const
        {
            const std::size_t count = phase_block_sizes_.size();
            fatrop_assert_msg(
                phases.size() == count
                && phase_diagonal.size() == count
                && global_arrow.size() == count
                && rhs_phase.size() == count
                && phase_solution.size() == count,
                "The phase-arrow OCP phase cardinalities are inconsistent.");
            fatrop_assert_msg(
                phase_coupling.size() + 1 == count,
                "The phase-arrow OCP chain has an inconsistent coupling count.");
            for (std::size_t phase = 0; phase < count; ++phase)
            {
                const auto &current = phases[phase];
                const Index left = phase == 0
                    ? 0 : phase_block_sizes_[phase - 1];
                const Index incident = left
                    + phase_block_sizes_[phase]
                    + number_of_global_variables_;
                fatrop_assert_msg(
                    current.cross_hessian.m()
                        == current.info.number_of_primal_variables
                    && current.cross_hessian.n() == incident,
                    "A phase trajectory cross Hessian has an inconsistent shape.");
                fatrop_assert_msg(
                    current.border_jacobian.m()
                        == current.info.number_of_eq_constraints
                    && current.border_jacobian.n() == incident,
                    "A phase trajectory border Jacobian has an inconsistent shape.");
                fatrop_assert_msg(
                    current.D_x.m()
                        == current.info.number_of_primal_variables
                    && (current.equality_dual_diagonal_is_zero
                        ? (current.D_eq.m() == 0
                           || current.D_eq.m()
                              == current.info.number_of_g_eq_path)
                        : current.D_eq.m()
                          == current.info.number_of_g_eq_path)
                    && current.D_s.m()
                        == current.info.number_of_slack_variables,
                    "A phase trajectory regularization vector has an inconsistent size.");
                fatrop_assert_msg(
                    current.rhs_primal.m()
                        == current.info.number_of_primal_variables
                    && current.primal.m()
                        == current.info.number_of_primal_variables
                    && current.rhs_constraints.m()
                        == current.info.number_of_eq_constraints
                    && current.multipliers.m()
                        == current.info.number_of_eq_constraints,
                    "A phase trajectory KKT vector has an inconsistent size.");
                const Index size = phase_block_sizes_[phase];
                fatrop_assert_msg(
                    phase_diagonal[phase].m() == size
                    && phase_diagonal[phase].n() == size
                    && global_arrow[phase].m() == size
                    && global_arrow[phase].n()
                       == number_of_global_variables_
                    && rhs_phase[phase].m() == size
                    && phase_solution[phase].m() == size,
                    "A reduced phase-arrow block has an inconsistent shape.");
                if (phase + 1 < count)
                {
                    fatrop_assert_msg(
                        phase_coupling[phase].m() == size
                        && phase_coupling[phase].n()
                           == phase_block_sizes_[phase + 1],
                        "A reduced adjacent phase coupling has an inconsistent shape.");
                }
            }
            fatrop_assert_msg(
                global_hessian.m() == number_of_global_variables_
                && global_hessian.n() == number_of_global_variables_
                && rhs_global.m() == number_of_global_variables_
                && global_solution.m() == number_of_global_variables_,
                "The reduced global arrow block has an inconsistent shape.");
        }

        LinsolReturnFlag solve_responses(
            const std::size_t phase,
            const PhaseArrowOcpPhaseViewTpl<ProblemType> &current)
        {
            PhaseWorkspace &workspace = workspaces_[phase];
            last_statistics_.incident_response_columns += workspace.columns;
            if (workspace.columns == 0)
                return LinsolReturnFlag::SUCCESS;
            MatRealView response_primal = workspace.response_primal.block(
                current.info.number_of_primal_variables,
                workspace.columns, 0, 0);
            MatRealView response_multipliers =
                workspace.response_multipliers.block(
                    current.info.number_of_eq_constraints,
                    workspace.columns, 0, 0);
            response_primal = 0.0;
            response_multipliers = 0.0;

            std::fill(
                workspace.active_column_mask.begin(),
                workspace.active_column_mask.end(), 0);
            std::fill(
                workspace.active_column_position.begin(),
                workspace.active_column_position.end(), -1);
            workspace.active_columns.clear();
            for (Index column = 0;
                 column < workspace.columns; ++column)
            {
                bool active = false;
                for (Index row = 0;
                     row < current.info.number_of_primal_variables; ++row)
                    active = active
                        || current.cross_hessian(row, column) != 0.0;
                for (Index row = 0;
                     row < current.info.number_of_eq_constraints; ++row)
                    active = active
                        || current.border_jacobian(row, column) != 0.0;
                if (active)
                {
                    const Index position = static_cast<Index>(
                        workspace.active_columns.size());
                    workspace.active_column_mask[
                        static_cast<std::size_t>(column)] = 1;
                    workspace.active_column_position[
                        static_cast<std::size_t>(column)] = position;
                    workspace.active_columns.push_back(column);
                }
            }
            if (workspace.active_columns.empty())
                return LinsolReturnFlag::SUCCESS;

            const Index active_count =
                static_cast<Index>(workspace.active_columns.size());
            last_statistics_.active_response_columns += active_count;
            MatRealView compact_rhs_primal =
                workspace.compact_rhs_primal.block(
                    current.info.number_of_primal_variables,
                    active_count, 0, 0);
            MatRealView compact_rhs_constraints =
                workspace.compact_rhs_constraints.block(
                    current.info.number_of_eq_constraints,
                    active_count, 0, 0);
            MatRealView compact_response_primal =
                workspace.compact_response_primal.block(
                    current.info.number_of_primal_variables,
                    active_count, 0, 0);
            MatRealView compact_response_multipliers =
                workspace.compact_response_multipliers.block(
                    current.info.number_of_eq_constraints,
                    active_count, 0, 0);
            compact_response_primal = 0.0;
            compact_response_multipliers = 0.0;
            for (Index compact = 0; compact < active_count; ++compact)
            {
                const Index source =
                    workspace.active_columns[
                        static_cast<std::size_t>(compact)];
                for (Index row = 0;
                     row < current.info.number_of_primal_variables; ++row)
                    compact_rhs_primal(row, compact) =
                        current.cross_hessian(row, source);
                for (Index row = 0;
                     row < current.info.number_of_eq_constraints; ++row)
                    compact_rhs_constraints(row, compact) =
                        current.border_jacobian(row, source);
            }

            if constexpr (std::is_same_v<ProblemType, ImplicitOcpType>)
            {
                bool full_rank = true;
                for (Index stage = 0;
                     stage + 1 < current.info.dims.K; ++stage)
                {
                    full_rank = full_rank
                        && current.jacobian.J_ranks[stage]
                           == current.info.dims.number_of_states[stage + 1];
                }
                if (full_rank)
                {
                    const LinsolReturnFlag status =
                        current.equality_dual_diagonal_is_zero
                        ? trajectory_solvers_[phase]->solve_rhs_batch(
                            current.info,
                            current.jacobian,
                            current.hessian,
                            current.D_s,
                            compact_rhs_primal,
                            compact_rhs_constraints,
                            compact_response_primal,
                            compact_response_multipliers)
                        : trajectory_solvers_[phase]->solve_rhs_batch(
                            current.info,
                            current.jacobian,
                            current.hessian,
                            current.D_eq,
                            current.D_s,
                            compact_rhs_primal,
                            compact_rhs_constraints,
                            compact_response_primal,
                            compact_response_multipliers);
                    if (status != LinsolReturnFlag::SUCCESS)
                        return status;
                    for (Index compact = 0;
                         compact < active_count; ++compact)
                    {
                        const Index destination = workspace.active_columns[
                            static_cast<std::size_t>(compact)];
                        for (Index row = 0;
                             row < current.info.number_of_primal_variables; ++row)
                        {
                            workspace.column_rhs_primal(row) =
                                compact_rhs_primal(row, compact);
                            workspace.column_primal(row) =
                                compact_response_primal(row, compact);
                        }
                        for (Index row = 0;
                             row < current.info.number_of_eq_constraints; ++row)
                        {
                            workspace.column_rhs_constraints(row) =
                                compact_rhs_constraints(row, compact);
                            workspace.column_multipliers(row) =
                                compact_response_multipliers(row, compact);
                        }
                        const LinsolReturnFlag refinement_status =
                            refine_trajectory_solution(
                                phase, current,
                                workspace.column_rhs_primal,
                                workspace.column_rhs_constraints,
                                workspace.column_primal,
                                workspace.column_multipliers);
                        if (refinement_status != LinsolReturnFlag::SUCCESS)
                            return refinement_status;
                        for (Index row = 0;
                             row < current.info.number_of_primal_variables; ++row)
                        {
                            response_primal(row, destination) =
                                workspace.column_primal(row);
                            compact_response_primal(row, compact) =
                                workspace.column_primal(row);
                        }
                        for (Index row = 0;
                             row < current.info.number_of_eq_constraints; ++row)
                        {
                            response_multipliers(row, destination) =
                                workspace.column_multipliers(row);
                            compact_response_multipliers(row, compact) =
                                workspace.column_multipliers(row);
                        }
                    }
                    return LinsolReturnFlag::SUCCESS;
                }
            }

            for (Index compact = 0;
                 compact < active_count; ++compact)
            {
                const Index column = workspace.active_columns[
                    static_cast<std::size_t>(compact)];
                for (Index row = 0;
                     row < current.info.number_of_primal_variables; ++row)
                    workspace.column_rhs_primal(row) =
                        current.cross_hessian(row, column);
                for (Index row = 0;
                     row < current.info.number_of_eq_constraints; ++row)
                    workspace.column_rhs_constraints(row) =
                        current.border_jacobian(row, column);
                workspace.column_primal = 0.0;
                workspace.column_multipliers = 0.0;
                const LinsolReturnFlag status =
                    current.equality_dual_diagonal_is_zero
                    ? trajectory_solvers_[phase]->solve_rhs(
                        current.info,
                        current.jacobian,
                        current.hessian,
                        current.D_s,
                        workspace.column_rhs_primal,
                        workspace.column_rhs_constraints,
                        workspace.column_primal,
                        workspace.column_multipliers)
                    : trajectory_solvers_[phase]->solve_rhs(
                        current.info,
                        current.jacobian,
                        current.hessian,
                        current.D_eq,
                        current.D_s,
                        workspace.column_rhs_primal,
                        workspace.column_rhs_constraints,
                        workspace.column_primal,
                        workspace.column_multipliers);
                if (status != LinsolReturnFlag::SUCCESS)
                    return status;
                const LinsolReturnFlag refinement_status =
                    refine_trajectory_solution(
                        phase, current,
                        workspace.column_rhs_primal,
                        workspace.column_rhs_constraints,
                        workspace.column_primal,
                        workspace.column_multipliers);
                if (refinement_status != LinsolReturnFlag::SUCCESS)
                    return refinement_status;
                for (Index row = 0;
                     row < current.info.number_of_primal_variables; ++row)
                {
                    response_primal(row, column) =
                        workspace.column_primal(row);
                    compact_response_primal(row, compact) =
                        workspace.column_primal(row);
                }
                for (Index row = 0;
                     row < current.info.number_of_eq_constraints; ++row)
                {
                    response_multipliers(row, column) =
                        workspace.column_multipliers(row);
                    compact_response_multipliers(row, compact) =
                        workspace.column_multipliers(row);
                }
            }
            return LinsolReturnFlag::SUCCESS;
        }

        LinsolReturnFlag refine_trajectory_solution(
            const std::size_t phase,
            const PhaseArrowOcpPhaseViewTpl<ProblemType> &current,
            const VecRealView &rhs_primal,
            const VecRealView &rhs_constraints,
            VecRealView &primal,
            VecRealView &multipliers)
        {
            PhaseWorkspace &workspace = workspaces_[phase];
            constexpr Index maximum_refinements = 3;
            constexpr Scalar backward_tolerance = 5e-12;
            for (Index iteration = 0;
                 iteration < maximum_refinements; ++iteration)
            {
                workspace.residual_primal = 0.0;
                workspace.residual_constraints = 0.0;
                workspace.hessian_scratch = 0.0;
                workspace.jacobian_scratch = 0.0;
                workspace.jacobian_transpose_product = 0.0;

                current.hessian.apply_on_right(
                    current.info, primal, 0.0,
                    workspace.hessian_scratch,
                    workspace.residual_primal);
                current.jacobian.transpose_apply_on_right(
                    current.info, multipliers, 0.0,
                    workspace.hessian_scratch,
                    workspace.jacobian_transpose_product);
                workspace.residual_primal =
                    workspace.residual_primal
                    + current.D_x * primal
                    + workspace.jacobian_transpose_product
                    + rhs_primal;

                current.jacobian.apply_on_right(
                    current.info, primal, 0.0,
                    workspace.jacobian_scratch,
                    workspace.residual_constraints);
                workspace.residual_constraints =
                    workspace.residual_constraints + rhs_constraints;
                if (!current.equality_dual_diagonal_is_zero
                    && current.info.number_of_g_eq_path > 0)
                {
                    workspace.residual_constraints.block(
                        current.info.number_of_g_eq_path,
                        current.info.offset_g_eq_path) =
                        workspace.residual_constraints.block(
                            current.info.number_of_g_eq_path,
                            current.info.offset_g_eq_path)
                        - current.D_eq
                          * multipliers.block(
                              current.info.number_of_g_eq_path,
                              current.info.offset_g_eq_path);
                }
                if (current.info.number_of_slack_variables > 0)
                {
                    workspace.residual_constraints.block(
                        current.info.number_of_slack_variables,
                        current.info.offset_g_eq_slack) =
                        workspace.residual_constraints.block(
                            current.info.number_of_slack_variables,
                            current.info.offset_g_eq_slack)
                        - current.D_s
                          * multipliers.block(
                              current.info.number_of_slack_variables,
                              current.info.offset_g_eq_slack);
                }

                const Scalar residual_norm = std::max(
                    norm_inf(workspace.residual_primal),
                    norm_inf(workspace.residual_constraints));
                const Scalar scale =
                    1.0
                    + norm_inf(rhs_primal)
                    + norm_inf(rhs_constraints)
                    + norm_inf(primal)
                    + norm_inf(multipliers);
                if (residual_norm <= backward_tolerance * scale)
                    return LinsolReturnFlag::SUCCESS;

                workspace.correction_primal = 0.0;
                workspace.correction_multipliers = 0.0;
                const LinsolReturnFlag status =
                    current.equality_dual_diagonal_is_zero
                    ? trajectory_solvers_[phase]->solve_rhs(
                        current.info,
                        current.jacobian,
                        current.hessian,
                        current.D_s,
                        workspace.residual_primal,
                        workspace.residual_constraints,
                        workspace.correction_primal,
                        workspace.correction_multipliers)
                    : trajectory_solvers_[phase]->solve_rhs(
                        current.info,
                        current.jacobian,
                        current.hessian,
                        current.D_eq,
                        current.D_s,
                        workspace.residual_primal,
                        workspace.residual_constraints,
                        workspace.correction_primal,
                        workspace.correction_multipliers);
                if (status != LinsolReturnFlag::SUCCESS)
                    return status;
                primal = primal + workspace.correction_primal;
                multipliers = multipliers + workspace.correction_multipliers;
            }
            return LinsolReturnFlag::SUCCESS;
        }

        void assemble_compact_condensation(
            const std::size_t phase,
            const PhaseArrowOcpPhaseViewTpl<ProblemType> &current)
        {
            PhaseWorkspace &workspace = workspaces_[phase];
            const Index active_count = static_cast<Index>(
                workspace.active_columns.size());
            if (active_count == 0)
                return;

            MatRealView compact_rhs_primal =
                workspace.compact_rhs_primal.block(
                    current.info.number_of_primal_variables,
                    active_count, 0, 0);
            MatRealView compact_rhs_constraints =
                workspace.compact_rhs_constraints.block(
                    current.info.number_of_eq_constraints,
                    active_count, 0, 0);
            MatRealView compact_response_primal =
                workspace.compact_response_primal.block(
                    current.info.number_of_primal_variables,
                    active_count, 0, 0);
            MatRealView compact_response_multipliers =
                workspace.compact_response_multipliers.block(
                    current.info.number_of_eq_constraints,
                    active_count, 0, 0);
            MatRealView compact_condensed =
                workspace.compact_condensed.block(
                    active_count, active_count, 0, 0);
            VecRealView compact_condensed_rhs =
                workspace.compact_condensed_rhs.block(
                    active_count, 0);
            compact_condensed = 0.0;
            compact_condensed_rhs = 0.0;

            if (current.info.number_of_primal_variables > 0)
            {
                gemm_tn(
                    active_count, active_count,
                    current.info.number_of_primal_variables,
                    1.0, compact_rhs_primal, 0, 0,
                    compact_response_primal, 0, 0,
                    1.0, compact_condensed, 0, 0,
                    compact_condensed, 0, 0);
                gemv_t(
                    current.info.number_of_primal_variables,
                    active_count,
                    1.0, compact_rhs_primal, 0, 0,
                    workspace.base_primal, 0,
                    1.0, compact_condensed_rhs, 0,
                    compact_condensed_rhs, 0);
            }
            if (current.info.number_of_eq_constraints > 0)
            {
                gemm_tn(
                    active_count, active_count,
                    current.info.number_of_eq_constraints,
                    1.0, compact_rhs_constraints, 0, 0,
                    compact_response_multipliers, 0, 0,
                    1.0, compact_condensed, 0, 0,
                    compact_condensed, 0, 0);
                gemv_t(
                    current.info.number_of_eq_constraints,
                    active_count,
                    1.0, compact_rhs_constraints, 0, 0,
                    workspace.base_multipliers, 0,
                    1.0, compact_condensed_rhs, 0,
                    compact_condensed_rhs, 0);
            }
        }

        void initialize_reduced_system(
            const std::vector<MatRealView> &phase_diagonal,
            const std::vector<MatRealView> &phase_coupling,
            const std::vector<MatRealView> &global_arrow,
            const MatRealView &global_hessian,
            const std::vector<VecRealView> &rhs_phase,
            const VecRealView &rhs_global)
        {
            for (std::size_t phase = 0;
                 phase < phase_block_sizes_.size(); ++phase)
            {
                reduced_diagonal_[phase] = phase_diagonal[phase];
                reduced_arrow_[phase] = global_arrow[phase];
                for (Index row = 0;
                     row < phase_block_sizes_[phase]; ++row)
                {
                    reduced_rhs_phase_[phase](row) =
                        -rhs_phase[phase](row);
                }
            }
            for (std::size_t phase = 0;
                 phase < reduced_coupling_.size(); ++phase)
                reduced_coupling_[phase] = phase_coupling[phase];
            MatRealView reduced_global =
                reduced_global_storage_->block(
                    number_of_global_variables_,
                    number_of_global_variables_, 0, 0);
            reduced_global = global_hessian;
            for (Index row = 0;
                 row < number_of_global_variables_; ++row)
                (*reduced_rhs_global_storage_)(row) = -rhs_global(row);
        }

        Scalar condensed_entry(
            const std::size_t phase,
            const PhaseArrowOcpPhaseViewTpl<ProblemType> &current,
            const Index row,
            const Index column) const
        {
            const PhaseWorkspace &workspace = workspaces_[phase];
            const Index compact_row = workspace.active_column_position[
                static_cast<std::size_t>(row)];
            const Index compact_column = workspace.active_column_position[
                static_cast<std::size_t>(column)];
            if (compact_row < 0 || compact_column < 0)
                return 0.0;
            return workspace.compact_condensed(
                compact_row, compact_column);
        }

        Scalar condensed_rhs_entry(
            const std::size_t phase,
            const PhaseArrowOcpPhaseViewTpl<ProblemType> &current,
            const Index column) const
        {
            const PhaseWorkspace &workspace = workspaces_[phase];
            const Index compact_column = workspace.active_column_position[
                static_cast<std::size_t>(column)];
            if (compact_column < 0)
                return 0.0;
            return workspace.compact_condensed_rhs(compact_column);
        }

        void add_phase_condensation(
            const std::size_t phase,
            const PhaseArrowOcpPhaseViewTpl<ProblemType> &current)
        {
            const Index left_size = phase == 0
                ? 0 : phase_block_sizes_[phase - 1];
            const Index right_size = phase_block_sizes_[phase];
            const Index right_offset = left_size;
            const Index global_offset = left_size + right_size;

            if (phase > 0)
            {
                for (Index row = 0; row < left_size; ++row)
                {
                    reduced_rhs_phase_[phase - 1](row) -=
                        condensed_rhs_entry(phase, current, row);
                    for (Index column = 0; column < left_size; ++column)
                    {
                        reduced_diagonal_[phase - 1](row, column) +=
                            condensed_entry(phase, current, row, column);
                    }
                    for (Index column = 0; column < right_size; ++column)
                    {
                        reduced_coupling_[phase - 1](row, column) +=
                            condensed_entry(
                                phase, current,
                                row, right_offset + column);
                    }
                    for (Index column = 0;
                         column < number_of_global_variables_; ++column)
                    {
                        reduced_arrow_[phase - 1](row, column) +=
                            condensed_entry(
                                phase, current,
                                row, global_offset + column);
                    }
                }
            }

            for (Index row = 0; row < right_size; ++row)
            {
                const Index incident_row = right_offset + row;
                reduced_rhs_phase_[phase](row) -=
                    condensed_rhs_entry(phase, current, incident_row);
                for (Index column = 0; column < right_size; ++column)
                {
                    reduced_diagonal_[phase](row, column) +=
                        condensed_entry(
                            phase, current,
                            incident_row, right_offset + column);
                }
                for (Index column = 0;
                     column < number_of_global_variables_; ++column)
                {
                    reduced_arrow_[phase](row, column) +=
                        condensed_entry(
                            phase, current,
                            incident_row, global_offset + column);
                }
            }

            MatRealView reduced_global =
                reduced_global_storage_->block(
                    number_of_global_variables_,
                    number_of_global_variables_, 0, 0);
            for (Index row = 0;
                 row < number_of_global_variables_; ++row)
            {
                const Index incident_row = global_offset + row;
                (*reduced_rhs_global_storage_)(row) -=
                    condensed_rhs_entry(phase, current, incident_row);
                for (Index column = 0;
                     column < number_of_global_variables_; ++column)
                {
                    reduced_global(row, column) +=
                        condensed_entry(
                            phase, current,
                            incident_row, global_offset + column);
                }
            }
        }

        void back_substitute(
            const std::size_t phase,
            const PhaseArrowOcpPhaseViewTpl<ProblemType> &current,
            const std::vector<VecRealView> &phase_solution,
            const VecRealView &global_solution)
        {
            const PhaseWorkspace &workspace = workspaces_[phase];
            const Index left_size = phase == 0
                ? 0 : phase_block_sizes_[phase - 1];
            const Index right_size = phase_block_sizes_[phase];
            const Index global_offset = left_size + right_size;

            for (Index row = 0;
                 row < current.info.number_of_primal_variables; ++row)
            {
                Scalar value = workspace.base_primal(row);
                for (Index column = 0; column < left_size; ++column)
                    value += workspace.response_primal(row, column)
                           * phase_solution[phase - 1](column);
                for (Index column = 0; column < right_size; ++column)
                    value += workspace.response_primal(
                                 row, left_size + column)
                           * phase_solution[phase](column);
                for (Index column = 0;
                     column < number_of_global_variables_; ++column)
                    value += workspace.response_primal(
                                 row, global_offset + column)
                           * global_solution(column);
                current.primal(row) = value;
            }
            for (Index row = 0;
                 row < current.info.number_of_eq_constraints; ++row)
            {
                Scalar value = workspace.base_multipliers(row);
                for (Index column = 0; column < left_size; ++column)
                    value += workspace.response_multipliers(row, column)
                           * phase_solution[phase - 1](column);
                for (Index column = 0; column < right_size; ++column)
                    value += workspace.response_multipliers(
                                 row, left_size + column)
                           * phase_solution[phase](column);
                for (Index column = 0;
                     column < number_of_global_variables_; ++column)
                    value += workspace.response_multipliers(
                                 row, global_offset + column)
                           * global_solution(column);
                current.multipliers(row) = value;
            }
        }

        std::vector<Index> phase_block_sizes_;
        Index number_of_global_variables_ = 0;
        std::vector<std::unique_ptr<AugSystemSolver<ProblemType>>>
            trajectory_solvers_;
        std::vector<PhaseWorkspace> workspaces_;
        PhaseArrowKktSolver reduced_solver_;
        std::vector<MatRealAllocated> reduced_diagonal_;
        std::vector<MatRealAllocated> reduced_coupling_;
        std::vector<MatRealAllocated> reduced_arrow_;
        std::unique_ptr<MatRealAllocated> reduced_global_storage_;
        std::vector<VecRealAllocated> reduced_rhs_phase_;
        std::unique_ptr<VecRealAllocated> reduced_rhs_global_storage_;
        PhaseArrowOcpKktFailureLocation last_failure_location_ =
            PhaseArrowOcpKktFailureLocation::None;
        Index last_failure_phase_ = -1;
        PhaseArrowOcpKktStatistics last_statistics_;
    };

    using PhaseArrowOcpPhaseView =
        PhaseArrowOcpPhaseViewTpl<OcpType>;
    using ImplicitPhaseArrowOcpPhaseView =
        PhaseArrowOcpPhaseViewTpl<ImplicitOcpType>;
    using PhaseArrowOcpKktSolver =
        PhaseArrowOcpKktSolverTpl<OcpType>;
    using ImplicitPhaseArrowOcpKktSolver =
        PhaseArrowOcpKktSolverTpl<ImplicitOcpType>;

} // namespace fatrop

#endif // __fatrop_ocp_phase_arrow_ocp_kkt_solver_hpp__
