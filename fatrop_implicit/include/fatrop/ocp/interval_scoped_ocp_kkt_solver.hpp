//
// Copyright (c) 2026
//

#ifndef __fatrop_ocp_interval_scoped_ocp_kkt_solver_hpp__
#define __fatrop_ocp_interval_scoped_ocp_kkt_solver_hpp__

#include "fatrop/common/exception.hpp"
#include "fatrop/linear_algebra/linear_solver_return_flags.hpp"
#include "fatrop/ocp/interval_scoped_kkt_solver.hpp"
#include "fatrop/ocp/ocp_phase_condensing.hpp"

#include <chrono>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace fatrop
{
    struct IntervalScopedOcpKktStatistics
    {
        Index incident_response_columns = 0;
        Index active_response_columns = 0;
        double trajectory_factorization_milliseconds = 0.0;
        double trajectory_response_milliseconds = 0.0;
        double condensation_milliseconds = 0.0;
        double reduced_solve_milliseconds = 0.0;
        double back_substitution_milliseconds = 0.0;
    };

    enum class IntervalScopedOcpKktFailureLocation
    {
        None,
        TrajectoryFactorization,
        TrajectoryResponse,
        ReducedIntervalSystem,
        NonfiniteSolution
    };

    /**
     * @brief Two-level FATROP KKT solver with arbitrary contiguous phase
     *        scopes for one-copy reduced variables.
     *
     * Every phase trajectory is factored independently.  The columns of a
     * phase view are ordered by `active_blocks(phase)`, with every block's
     * scalar entries contiguous.  Phase Schur contributions are assembled in
     * the fill-free interval reduced solver, then trajectory steps are
     * recovered by back substitution.
     */
    template <typename ProblemType>
    class IntervalScopedOcpKktSolverTpl
    {
    public:
        static_assert(
            std::is_same_v<ProblemType, OcpType>
            || std::is_same_v<ProblemType, ImplicitOcpType>,
            "IntervalScopedOcpKktSolverTpl supports explicit and implicit OCPs.");

        IntervalScopedOcpKktSolverTpl(
            const std::vector<const ProblemInfo *> &phase_info,
            std::vector<IntervalScope> scopes)
            : scopes_(std::move(scopes)),
              reduced_solver_(
                  static_cast<Index>(phase_info.size()), scopes_)
        {
            fatrop_assert_msg(
                !phase_info.empty(),
                "An interval-scoped OCP needs at least one phase.");
            active_blocks_.resize(phase_info.size());
            kernels_.reserve(phase_info.size());
            scoped_solution_.reserve(phase_info.size());
            for (std::size_t phase = 0;
                 phase < phase_info.size(); ++phase)
            {
                fatrop_assert_msg(
                    phase_info[phase] != nullptr,
                    "An interval-scoped OCP ProblemInfo pointer is null.");
                Index columns = 0;
                for (Index block = 0;
                     block < static_cast<Index>(scopes_.size()); ++block)
                {
                    const IntervalScope &scope = scopes_[
                        static_cast<std::size_t>(block)];
                    if (scope.first_phase <= static_cast<Index>(phase)
                        && static_cast<Index>(phase) <= scope.last_phase)
                    {
                        active_blocks_[phase].push_back(block);
                        columns += scope.dimension;
                    }
                }
                kernels_.push_back(std::make_unique<
                    OcpPhaseCondensingKernelTpl<ProblemType>>(
                        *phase_info[phase], columns));
                scoped_solution_.emplace_back(std::max<Index>(columns, 1));
            }
        }

        Index number_of_phases() const noexcept
        {
            return static_cast<Index>(active_blocks_.size());
        }

        const std::vector<Index> &active_blocks(
            const Index phase) const
        {
            fatrop_assert_msg(
                phase >= 0
                && phase < static_cast<Index>(active_blocks_.size()),
                "An interval-scoped active-phase index is out of range.");
            return active_blocks_[static_cast<std::size_t>(phase)];
        }

        IntervalScopedKktSolver &reduced_solver() noexcept
        {
            return reduced_solver_;
        }

        const IntervalScopedKktSolver &reduced_solver() const noexcept
        {
            return reduced_solver_;
        }

        AugSystemSolver<ProblemType> &trajectory_solver(
            const Index phase)
        {
            fatrop_assert_msg(
                phase >= 0
                && phase < static_cast<Index>(kernels_.size()),
                "An interval-scoped trajectory solver index is out of range.");
            return kernels_[static_cast<std::size_t>(phase)]
                ->trajectory_solver();
        }

        IntervalScopedOcpKktFailureLocation
        last_failure_location() const noexcept
        {
            return last_failure_location_;
        }

        Index last_failure_phase() const noexcept
        {
            return last_failure_phase_;
        }

        Index last_failure_block() const noexcept
        {
            return last_failure_block_;
        }

        const IntervalScopedOcpKktStatistics &
        last_statistics() const noexcept
        {
            return last_statistics_;
        }

        /**
         * Solve the complete KKT system using FATROP's residual convention:
         * the returned step solves `A step = -rhs`.
         *
         * `edge_blocks` follows `reduced_solver().edges()` exactly.
         */
        LinsolReturnFlag solve(
            const std::vector<OcpPhaseCondensingViewTpl<ProblemType>> &phases,
            const std::vector<MatRealView> &block_diagonal,
            const std::vector<MatRealView> &edge_blocks,
            const std::vector<VecRealView> &rhs_blocks,
            const std::vector<VecRealView> &block_solution)
        {
            validate_dimensions(
                phases, block_diagonal, edge_blocks,
                rhs_blocks, block_solution);
            last_failure_location_ =
                IntervalScopedOcpKktFailureLocation::None;
            last_failure_phase_ = -1;
            last_failure_block_ = -1;
            last_statistics_ = {};
            factorization_ready_ = false;

            for (std::size_t phase = 0;
                 phase < phases.size(); ++phase)
            {
                const LinsolReturnFlag status =
                    kernels_[phase]->factor_and_condense(phases[phase]);
                const OcpPhaseCondensingStatistics &statistics =
                    kernels_[phase]->last_statistics();
                last_statistics_.incident_response_columns +=
                    statistics.incident_response_columns;
                last_statistics_.active_response_columns +=
                    statistics.active_response_columns;
                last_statistics_.trajectory_factorization_milliseconds +=
                    statistics.factorization_milliseconds;
                last_statistics_.trajectory_response_milliseconds +=
                    statistics.response_milliseconds;
                last_statistics_.condensation_milliseconds +=
                    statistics.condensation_milliseconds;
                if (status != LinsolReturnFlag::SUCCESS)
                {
                    last_failure_phase_ = static_cast<Index>(phase);
                    last_failure_location_ =
                        kernels_[phase]->last_failure_location()
                            == OcpPhaseCondensingFailureLocation::
                                TrajectoryFactorization
                        ? IntervalScopedOcpKktFailureLocation::
                            TrajectoryFactorization
                        : IntervalScopedOcpKktFailureLocation::
                            TrajectoryResponse;
                    return status;
                }
            }

            reduced_solver_.clear_numeric();
            for (Index block = 0;
                 block < static_cast<Index>(scopes_.size()); ++block)
            {
                reduced_solver_.add_matrix_block(
                    block, block,
                    block_diagonal[static_cast<std::size_t>(block)]);
                reduced_solver_.add_rhs_block(
                    block,
                    rhs_blocks[static_cast<std::size_t>(block)],
                    -1.0);
            }
            const std::vector<IntervalKktEdge> &edges =
                reduced_solver_.edges();
            for (std::size_t edge = 0; edge < edges.size(); ++edge)
            {
                reduced_solver_.add_matrix_block(
                    edges[edge].first_block,
                    edges[edge].second_block,
                    edge_blocks[edge]);
            }
            for (std::size_t phase = 0;
                 phase < phases.size(); ++phase)
            {
                reduced_solver_.add_phase_contribution(
                    static_cast<Index>(phase),
                    active_blocks_[phase],
                    kernels_[phase]->condensed_matrix(),
                    kernels_[phase]->condensed_rhs(),
                    1.0, -1.0);
            }

            const auto reduced_start = Clock::now();
            const LinsolReturnFlag reduced_status =
                reduced_solver_.factor_and_solve(block_solution);
            last_statistics_.reduced_solve_milliseconds =
                elapsed_milliseconds(reduced_start);
            if (reduced_status != LinsolReturnFlag::SUCCESS)
            {
                last_failure_location_ =
                    IntervalScopedOcpKktFailureLocation::
                        ReducedIntervalSystem;
                last_failure_block_ =
                    reduced_solver_.last_failure_block();
                return reduced_status;
            }
            factorization_ready_ = true;

            const auto back_start = Clock::now();
            for (std::size_t phase = 0;
                 phase < phases.size(); ++phase)
            {
                Index offset = 0;
                VecRealAllocated &phase_solution = scoped_solution_[phase];
                for (const Index block : active_blocks_[phase])
                {
                    const Index dimension = scopes_[
                        static_cast<std::size_t>(block)].dimension;
                    for (Index row = 0; row < dimension; ++row)
                    {
                        phase_solution(offset + row) =
                            block_solution[static_cast<std::size_t>(block)](row);
                    }
                    offset += dimension;
                }
                kernels_[phase]->back_substitute(
                    phases[phase],
                    phase_solution.block(offset, 0));
            }
            last_statistics_.back_substitution_milliseconds =
                elapsed_milliseconds(back_start);

            for (const auto &phase : phases)
            {
                if (!phase.primal.is_finite()
                    || !phase.multipliers.is_finite())
                {
                    last_failure_location_ =
                        IntervalScopedOcpKktFailureLocation::
                            NonfiniteSolution;
                    factorization_ready_ = false;
                    return LinsolReturnFlag::NAN_SOLUTION;
                }
            }
            return LinsolReturnFlag::SUCCESS;
        }

        /**
         * Solve a new RHS with the trajectory and interval factors retained
         * by the preceding `solve()`.  Matrix, regularization, and border
         * blocks must be unchanged.
         */
        LinsolReturnFlag solve_rhs(
            const std::vector<OcpPhaseCondensingViewTpl<ProblemType>> &phases,
            const std::vector<VecRealView> &rhs_blocks,
            const std::vector<VecRealView> &block_solution)
        {
            fatrop_assert_msg(
                factorization_ready_,
                "Interval-scoped solve_rhs requires a retained factorization.");
            validate_rhs_dimensions(phases, rhs_blocks, block_solution);
            last_failure_location_ =
                IntervalScopedOcpKktFailureLocation::None;
            last_failure_phase_ = -1;
            last_failure_block_ = -1;
            last_statistics_ = {};

            for (std::size_t phase = 0;
                 phase < phases.size(); ++phase)
            {
                const LinsolReturnFlag status =
                    kernels_[phase]->condense_rhs(phases[phase]);
                const OcpPhaseCondensingStatistics &statistics =
                    kernels_[phase]->last_statistics();
                last_statistics_.incident_response_columns +=
                    statistics.incident_response_columns;
                last_statistics_.active_response_columns +=
                    statistics.active_response_columns;
                last_statistics_.trajectory_response_milliseconds +=
                    statistics.response_milliseconds;
                last_statistics_.condensation_milliseconds +=
                    statistics.condensation_milliseconds;
                if (status != LinsolReturnFlag::SUCCESS)
                {
                    last_failure_phase_ = static_cast<Index>(phase);
                    last_failure_location_ =
                        IntervalScopedOcpKktFailureLocation::
                            TrajectoryResponse;
                    factorization_ready_ = false;
                    return status;
                }
            }

            reduced_solver_.clear_rhs();
            for (Index block = 0;
                 block < static_cast<Index>(scopes_.size()); ++block)
                reduced_solver_.add_rhs_block(
                    block,
                    rhs_blocks[static_cast<std::size_t>(block)],
                    -1.0);
            for (std::size_t phase = 0;
                 phase < phases.size(); ++phase)
            {
                Index offset = 0;
                VecRealView condensed_rhs =
                    kernels_[phase]->condensed_rhs();
                for (const Index block : active_blocks_[phase])
                {
                    const Index dimension = scopes_[
                        static_cast<std::size_t>(block)].dimension;
                    reduced_solver_.add_rhs_block(
                        block,
                        condensed_rhs.block(dimension, offset),
                        -1.0);
                    offset += dimension;
                }
            }

            const auto reduced_start = Clock::now();
            const LinsolReturnFlag reduced_status =
                reduced_solver_.solve(block_solution);
            last_statistics_.reduced_solve_milliseconds =
                elapsed_milliseconds(reduced_start);
            if (reduced_status != LinsolReturnFlag::SUCCESS)
            {
                last_failure_location_ =
                    IntervalScopedOcpKktFailureLocation::
                        ReducedIntervalSystem;
                last_failure_block_ =
                    reduced_solver_.last_failure_block();
                factorization_ready_ = false;
                return reduced_status;
            }

            const auto back_start = Clock::now();
            for (std::size_t phase = 0;
                 phase < phases.size(); ++phase)
            {
                Index offset = 0;
                VecRealAllocated &phase_solution = scoped_solution_[phase];
                for (const Index block : active_blocks_[phase])
                {
                    const Index dimension = scopes_[
                        static_cast<std::size_t>(block)].dimension;
                    for (Index row = 0; row < dimension; ++row)
                        phase_solution(offset + row) =
                            block_solution[
                                static_cast<std::size_t>(block)](row);
                    offset += dimension;
                }
                kernels_[phase]->back_substitute(
                    phases[phase], phase_solution.block(offset, 0));
            }
            last_statistics_.back_substitution_milliseconds =
                elapsed_milliseconds(back_start);
            for (const auto &phase : phases)
            {
                if (!phase.primal.is_finite()
                    || !phase.multipliers.is_finite())
                {
                    last_failure_location_ =
                        IntervalScopedOcpKktFailureLocation::
                            NonfiniteSolution;
                    factorization_ready_ = false;
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

        void validate_dimensions(
            const std::vector<OcpPhaseCondensingViewTpl<ProblemType>> &phases,
            const std::vector<MatRealView> &block_diagonal,
            const std::vector<MatRealView> &edge_blocks,
            const std::vector<VecRealView> &rhs_blocks,
            const std::vector<VecRealView> &block_solution) const
        {
            fatrop_assert_msg(
                phases.size() == active_blocks_.size(),
                "The interval-scoped OCP phase count is inconsistent.");
            fatrop_assert_msg(
                block_diagonal.size() == scopes_.size()
                && rhs_blocks.size() == scopes_.size()
                && block_solution.size() == scopes_.size(),
                "The interval-scoped OCP block count is inconsistent.");
            fatrop_assert_msg(
                edge_blocks.size() == reduced_solver_.edges().size(),
                "The interval-scoped OCP edge count is inconsistent.");

            for (std::size_t phase = 0;
                 phase < phases.size(); ++phase)
            {
                Index columns = 0;
                for (const Index block : active_blocks_[phase])
                    columns += scopes_[static_cast<std::size_t>(block)].dimension;
                fatrop_assert_msg(
                    phases[phase].cross_hessian.n() == columns
                    && phases[phase].border_jacobian.n() == columns,
                    "An interval-scoped phase incidence has a wrong width.");
            }
            for (std::size_t block = 0;
                 block < scopes_.size(); ++block)
            {
                const Index dimension = scopes_[block].dimension;
                fatrop_assert_msg(
                    block_diagonal[block].m() == dimension
                    && block_diagonal[block].n() == dimension
                    && rhs_blocks[block].m() == dimension
                    && block_solution[block].m() == dimension,
                    "An interval-scoped direct block has a wrong shape.");
            }
            const std::vector<IntervalKktEdge> &edges =
                reduced_solver_.edges();
            for (std::size_t edge = 0; edge < edges.size(); ++edge)
            {
                fatrop_assert_msg(
                    edge_blocks[edge].m() == scopes_[static_cast<std::size_t>(
                        edges[edge].first_block)].dimension
                    && edge_blocks[edge].n() == scopes_[static_cast<std::size_t>(
                        edges[edge].second_block)].dimension,
                    "An interval-scoped direct edge has a wrong shape.");
            }
        }

        void validate_rhs_dimensions(
            const std::vector<OcpPhaseCondensingViewTpl<ProblemType>> &phases,
            const std::vector<VecRealView> &rhs_blocks,
            const std::vector<VecRealView> &block_solution) const
        {
            fatrop_assert_msg(
                phases.size() == active_blocks_.size(),
                "The interval-scoped OCP phase count is inconsistent.");
            fatrop_assert_msg(
                rhs_blocks.size() == scopes_.size()
                && block_solution.size() == scopes_.size(),
                "The interval-scoped OCP RHS block count is inconsistent.");
            for (std::size_t block = 0; block < scopes_.size(); ++block)
            {
                const Index dimension = scopes_[block].dimension;
                fatrop_assert_msg(
                    rhs_blocks[block].m() == dimension
                    && block_solution[block].m() == dimension,
                    "An interval-scoped OCP RHS block has a wrong size.");
            }
        }

        std::vector<IntervalScope> scopes_;
        std::vector<std::vector<Index>> active_blocks_;
        std::vector<std::unique_ptr<
            OcpPhaseCondensingKernelTpl<ProblemType>>> kernels_;
        std::vector<VecRealAllocated> scoped_solution_;
        IntervalScopedKktSolver reduced_solver_;
        IntervalScopedOcpKktFailureLocation last_failure_location_ =
            IntervalScopedOcpKktFailureLocation::None;
        Index last_failure_phase_ = -1;
        Index last_failure_block_ = -1;
        IntervalScopedOcpKktStatistics last_statistics_;
        bool factorization_ready_ = false;
    };

    using IntervalScopedOcpKktSolver =
        IntervalScopedOcpKktSolverTpl<OcpType>;
    using IntervalScopedImplicitOcpKktSolver =
        IntervalScopedOcpKktSolverTpl<ImplicitOcpType>;
} // namespace fatrop

#endif // __fatrop_ocp_interval_scoped_ocp_kkt_solver_hpp__
