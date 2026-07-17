//
// Copyright (c) 2026
//

#ifndef __fatrop_ocp_phase_arrow_pd_solver_hpp__
#define __fatrop_ocp_phase_arrow_pd_solver_hpp__

#include "fatrop/common/exception.hpp"
#include "fatrop/context/context.hpp"
#include "fatrop/linear_algebra/vector.hpp"
#include "fatrop/ocp/phase_arrow_ocp_kkt_solver.hpp"

#include <algorithm>
#include <type_traits>
#include <utility>
#include <vector>

namespace fatrop
{
    /**
     * @brief Non-owning primal-dual data for one independently condensable
     * phase.
     *
     * The phase `ProblemInfo` contains trajectory variables only. Phase-local
     * interface/static variables and mission-global variables are supplied to
     * `PhaseArrowPdSolverTpl::solve` through the reduced phase-arrow blocks.
     * Their derivatives in this phase use the same incident-column ordering
     * as `PhaseArrowOcpPhaseViewTpl`:
     *
     *     [q_(f-1) | q_f | g].
     *
     * Bounds on an incident static variable can be represented without
     * destroying this structure by assigning an identity inequality row to
     * one phase and placing its derivative in `border_jacobian`.
     */
    template <typename ProblemType>
    struct PhaseArrowPdPhaseViewTpl
    {
        PhaseArrowPdPhaseViewTpl(
            const ProblemInfo &info_value,
            Jacobian<ProblemType> &jacobian_value,
            Hessian<ProblemType> &hessian_value,
            const VecRealView &D_x_value,
            const bool equality_dual_diagonal_is_zero_value,
            const VecRealView &D_e_value,
            const VecRealView &slack_lower_distance_value,
            const VecRealView &slack_upper_distance_value,
            const VecRealView &slack_lower_dual_value,
            const VecRealView &slack_upper_dual_value,
            const MatRealView &cross_hessian_value,
            const MatRealView &border_jacobian_value,
            const VecRealView &rhs_primal_value,
            const VecRealView &rhs_slack_value,
            const VecRealView &rhs_constraints_value,
            const VecRealView &rhs_lower_complementarity_value,
            const VecRealView &rhs_upper_complementarity_value,
            const VecRealView &solution_value)
            : info(info_value),
              jacobian(jacobian_value),
              hessian(hessian_value),
              D_x(D_x_value),
              equality_dual_diagonal_is_zero(
                  equality_dual_diagonal_is_zero_value),
              D_e(D_e_value),
              slack_lower_distance(slack_lower_distance_value),
              slack_upper_distance(slack_upper_distance_value),
              slack_lower_dual(slack_lower_dual_value),
              slack_upper_dual(slack_upper_dual_value),
              cross_hessian(cross_hessian_value),
              border_jacobian(border_jacobian_value),
              rhs_primal(rhs_primal_value),
              rhs_slack(rhs_slack_value),
              rhs_constraints(rhs_constraints_value),
              rhs_lower_complementarity(
                  rhs_lower_complementarity_value),
              rhs_upper_complementarity(
                  rhs_upper_complementarity_value),
              solution(solution_value)
        {}

        const ProblemInfo &info;
        Jacobian<ProblemType> &jacobian;
        Hessian<ProblemType> &hessian;
        VecRealView D_x;
        bool equality_dual_diagonal_is_zero = false;
        VecRealView D_e;
        VecRealView slack_lower_distance;
        VecRealView slack_upper_distance;
        VecRealView slack_lower_dual;
        VecRealView slack_upper_dual;
        MatRealView cross_hessian;
        MatRealView border_jacobian;
        VecRealView rhs_primal;
        VecRealView rhs_slack;
        VecRealView rhs_constraints;
        VecRealView rhs_lower_complementarity;
        VecRealView rhs_upper_complementarity;
        VecRealView solution;
    };

    /**
     * @brief Full primal-dual reduction around the two-level phase-arrow OCP
     * solver.
     *
     * For every phase this class eliminates inequality bound-dual steps and
     * slack steps exactly as `PdSolverOrig`, passes the resulting augmented
     * trajectory system to `PhaseArrowOcpKktSolverTpl`, and reconstructs the
     * complete phase step `(x,s,lambda,zL,zU)`. The reduced `q_f` and `g`
     * blocks are therefore used inside a real barrier Newton system rather
     * than only by an equality-constrained KKT micro-kernel.
     *
     * The supplied reduced diagonal, coupling, arrow, Hessian, and RHS blocks
     * must already contain objective curvature, primal regularization, and any
     * barrier contribution belonging directly to the reduced variables.
     */
    template <typename ProblemType>
    class PhaseArrowPdSolverTpl
    {
    public:
        static_assert(
            std::is_same_v<ProblemType, OcpType>
            || std::is_same_v<ProblemType, ImplicitOcpType>,
            "PhaseArrowPdSolverTpl supports OcpType and ImplicitOcpType.");

        PhaseArrowPdSolverTpl(
            const std::vector<const ProblemInfo *> &phase_info,
            std::vector<Index> phase_block_sizes,
            const Index number_of_global_variables)
            : phase_solver_(
                  phase_info, std::move(phase_block_sizes),
                  number_of_global_variables)
        {
            workspaces_.reserve(phase_info.size());
            for (const ProblemInfo *info : phase_info)
            {
                fatrop_assert_msg(
                    info != nullptr,
                    "A phase-arrow primal-dual ProblemInfo pointer is null.");
                workspaces_.emplace_back(*info);
            }
        }

        PhaseArrowOcpKktSolverTpl<ProblemType> &phase_solver() noexcept
        {
            return phase_solver_;
        }

        const PhaseArrowOcpKktSolverTpl<ProblemType> &phase_solver() const noexcept
        {
            return phase_solver_;
        }

        LinsolReturnFlag solve(
            const std::vector<PhaseArrowPdPhaseViewTpl<ProblemType>> &phases,
            const std::vector<MatRealView> &phase_diagonal,
            const std::vector<MatRealView> &phase_coupling,
            const std::vector<MatRealView> &global_arrow,
            const MatRealView &global_hessian,
            const std::vector<VecRealView> &rhs_phase,
            const VecRealView &rhs_global,
            const std::vector<VecRealView> &phase_solution,
            const VecRealView &global_solution)
        {
            fatrop_assert_msg(
                phases.size() == workspaces_.size(),
                "The phase-arrow primal-dual phase count is inconsistent.");

            std::vector<PhaseArrowOcpPhaseViewTpl<ProblemType>>
                augmented_phases;
            augmented_phases.reserve(phases.size());
            for (std::size_t phase = 0; phase < phases.size(); ++phase)
            {
                validate_phase(phases[phase]);
                reduce_phase(phases[phase], workspaces_[phase]);
                PhaseWorkspace &workspace = workspaces_[phase];
                const ProblemInfo &info = phases[phase].info;
                augmented_phases.emplace_back(
                    info,
                    phases[phase].jacobian,
                    phases[phase].hessian,
                    phases[phase].D_x.block(
                        info.number_of_primal_variables, info.offset_primal),
                    phases[phase].D_e.block(
                        info.number_of_g_eq_path, info.offset_g_eq_path),
                    workspace.inequality_diagonal.block(
                        info.number_of_slack_variables, 0),
                    phases[phase].cross_hessian,
                    phases[phase].border_jacobian,
                    phases[phase].rhs_primal,
                    workspace.augmented_constraints.block(
                        info.number_of_eq_constraints, 0),
                    workspace.primal.block(
                        info.number_of_primal_variables, 0),
                    workspace.multipliers.block(
                        info.number_of_eq_constraints, 0),
                    phases[phase].equality_dual_diagonal_is_zero);
            }

            const LinsolReturnFlag status = phase_solver_.solve(
                augmented_phases,
                phase_diagonal,
                phase_coupling,
                global_arrow,
                global_hessian,
                rhs_phase,
                rhs_global,
                phase_solution,
                global_solution);
            if (status != LinsolReturnFlag::SUCCESS)
                return status;

            for (std::size_t phase = 0; phase < phases.size(); ++phase)
                recover_phase(phases[phase], workspaces_[phase]);
            return LinsolReturnFlag::SUCCESS;
        }

    private:
        struct PhaseWorkspace
        {
            explicit PhaseWorkspace(const ProblemInfo &info)
                : sigma_inverse(info.number_of_slack_variables),
                  slack_rhs(info.number_of_slack_variables),
                  inequality_rhs(info.number_of_slack_variables),
                  inequality_diagonal(info.number_of_slack_variables),
                  augmented_constraints(info.number_of_eq_constraints),
                  primal(info.number_of_primal_variables),
                  multipliers(info.number_of_eq_constraints)
            {}

            VecRealAllocated sigma_inverse;
            VecRealAllocated slack_rhs;
            VecRealAllocated inequality_rhs;
            VecRealAllocated inequality_diagonal;
            VecRealAllocated augmented_constraints;
            VecRealAllocated primal;
            VecRealAllocated multipliers;
        };

        static Index solution_dimension(const ProblemInfo &info)
        {
            return info.pd_orig_offset_zu + info.number_of_slack_variables;
        }

        static void validate_phase(
            const PhaseArrowPdPhaseViewTpl<ProblemType> &phase)
        {
            const ProblemInfo &info = phase.info;
            const Index slack = info.number_of_slack_variables;
            fatrop_assert_msg(
                phase.D_x.m()
                    == info.number_of_primal_variables + slack
                && phase.D_e.m() == info.number_of_eq_constraints
                && phase.slack_lower_distance.m() == slack
                && phase.slack_upper_distance.m() == slack
                && phase.slack_lower_dual.m() == slack
                && phase.slack_upper_dual.m() == slack,
                "A phase-arrow primal-dual diagonal has an inconsistent size.");
            fatrop_assert_msg(
                phase.rhs_primal.m() == info.number_of_primal_variables
                && phase.rhs_slack.m() == slack
                && phase.rhs_constraints.m()
                    == info.number_of_eq_constraints
                && phase.rhs_lower_complementarity.m() == slack
                && phase.rhs_upper_complementarity.m() == slack
                && phase.solution.m() == solution_dimension(info),
                "A phase-arrow primal-dual vector has an inconsistent size.");
        }

        static void reduce_phase(
            const PhaseArrowPdPhaseViewTpl<ProblemType> &phase,
            PhaseWorkspace &workspace)
        {
            const ProblemInfo &info = phase.info;
            const Index slack = info.number_of_slack_variables;
            if (slack > 0)
            {
                workspace.sigma_inverse =
                    1.0 /
                    (phase.D_x.block(slack, info.offset_slack)
                     + 1.0 / phase.slack_lower_distance
                         * phase.slack_lower_dual
                     + 1.0 / phase.slack_upper_distance
                         * phase.slack_upper_dual);
                workspace.slack_rhs =
                    phase.rhs_slack
                    + 1.0 / phase.slack_lower_distance
                        * phase.rhs_lower_complementarity
                    - 1.0 / phase.slack_upper_distance
                        * phase.rhs_upper_complementarity;
                workspace.inequality_diagonal =
                    workspace.sigma_inverse
                    + phase.D_e.block(
                        slack, info.offset_g_eq_slack);
                workspace.inequality_rhs =
                    phase.rhs_constraints.block(
                        slack, info.offset_g_eq_slack)
                    + workspace.sigma_inverse * workspace.slack_rhs;
            }

            workspace.augmented_constraints.block(
                info.number_of_g_eq_path, info.offset_g_eq_path) =
                phase.rhs_constraints.block(
                    info.number_of_g_eq_path, info.offset_g_eq_path);
            workspace.augmented_constraints.block(
                info.number_of_g_eq_dyn, info.offset_g_eq_dyn) =
                phase.rhs_constraints.block(
                    info.number_of_g_eq_dyn, info.offset_g_eq_dyn);
            workspace.augmented_constraints.block(
                info.number_of_g_eq_slack, info.offset_g_eq_slack) =
                workspace.inequality_rhs;
            workspace.primal = 0.0;
            workspace.multipliers = 0.0;
        }

        static void recover_phase(
            const PhaseArrowPdPhaseViewTpl<ProblemType> &phase,
            const PhaseWorkspace &workspace)
        {
            const ProblemInfo &info = phase.info;
            const Index slack = info.number_of_slack_variables;
            phase.solution.block(
                info.number_of_primal_variables,
                info.pd_orig_offset_primal) = workspace.primal;
            phase.solution.block(
                info.number_of_eq_constraints,
                info.pd_orig_offset_mult) = workspace.multipliers;
            if (slack == 0)
                return;

            VecRealView slack_step = phase.solution.block(
                slack, info.pd_orig_offset_slack);
            const VecRealView inequality_multipliers =
                workspace.multipliers.block(
                    slack, info.offset_g_eq_slack);
            slack_step = workspace.sigma_inverse
                * (inequality_multipliers - workspace.slack_rhs);
            phase.solution.block(slack, info.pd_orig_offset_zl) =
                1.0 / phase.slack_lower_distance
                * (-phase.rhs_lower_complementarity
                   - phase.slack_lower_dual * slack_step);
            phase.solution.block(slack, info.pd_orig_offset_zu) =
                1.0 / phase.slack_upper_distance
                * (-phase.rhs_upper_complementarity
                   + phase.slack_upper_dual * slack_step);
        }

        PhaseArrowOcpKktSolverTpl<ProblemType> phase_solver_;
        std::vector<PhaseWorkspace> workspaces_;
    };

    using PhaseArrowPdPhaseView =
        PhaseArrowPdPhaseViewTpl<OcpType>;
    using ImplicitPhaseArrowPdPhaseView =
        PhaseArrowPdPhaseViewTpl<ImplicitOcpType>;
    using PhaseArrowPdSolver =
        PhaseArrowPdSolverTpl<OcpType>;
    using ImplicitPhaseArrowPdSolver =
        PhaseArrowPdSolverTpl<ImplicitOcpType>;

} // namespace fatrop

#endif // __fatrop_ocp_phase_arrow_pd_solver_hpp__
