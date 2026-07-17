//
// Copyright (c) 2026
//

#include "fatrop/ocp/phase_arrow_pd_solver.hpp"
#include "fatrop/ocp/pd_system_orig.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

using namespace fatrop;

namespace
{
    struct PhaseData
    {
        PhaseData(
            const Index phase_value,
            const Index nx_value,
            const Index nu_value,
            const Index incident_columns,
            const bool stabilized_value)
            : phase(phase_value),
              stabilized(stabilized_value),
              dims(
                  2,
                  std::vector<Index>{nu_value, 0},
                  std::vector<Index>{nx_value, nx_value},
                  std::vector<Index>{1, 0},
                  std::vector<Index>{1, 0}),
              info(dims),
              jacobian(dims),
              hessian(dims),
              D_x(
                  info.number_of_primal_variables
                  + info.number_of_slack_variables),
              D_e(info.number_of_eq_constraints),
              slack_lower_distance(info.number_of_slack_variables),
              slack_upper_distance(info.number_of_slack_variables),
              slack_lower_dual(info.number_of_slack_variables),
              slack_upper_dual(info.number_of_slack_variables),
              cross_hessian(
                  info.number_of_primal_variables, incident_columns),
              border_jacobian(
                  info.number_of_eq_constraints, incident_columns),
              rhs_primal(info.number_of_primal_variables),
              rhs_slack(info.number_of_slack_variables),
              rhs_constraints(info.number_of_eq_constraints),
              rhs_lower_complementarity(info.number_of_slack_variables),
              rhs_upper_complementarity(info.number_of_slack_variables),
              solution(
                  info.pd_orig_offset_zu
                  + info.number_of_slack_variables)
        {
            jacobian.BAbt[0] = 0.0;
            jacobian.Jt[0] = 0.0;
            jacobian.Jt_inv[0] = 0.0;
            jacobian.Gg_eqt[0] = 0.0;
            jacobian.Gg_eqt[1] = 0.0;
            jacobian.Gg_ineqt[0] = 0.0;
            jacobian.Gg_ineqt[1] = 0.0;
            hessian.set_zero();

            const Index current_variables = nu_value + nx_value;
            for (Index row = 0; row < current_variables; ++row)
            {
                for (Index state = 0; state < nx_value; ++state)
                {
                    jacobian.BAbt[0](row, state) =
                        0.025
                        * (1 + ((row + 2 * state + phase) % 5))
                        - 0.045;
                    hessian.FuFx[0](row, state) =
                        0.006
                        * (1 + ((2 * row + state + phase) % 4))
                        - 0.012;
                }
                jacobian.Gg_eqt[0](row, 0) =
                    0.04
                    + 0.01 * ((row + phase) % 4);
                jacobian.Gg_ineqt[0](row, 0) =
                    -0.035
                    + 0.012 * ((2 * row + phase) % 5);
            }
            for (Index state = 0; state < nx_value; ++state)
            {
                const Scalar diagonal =
                    1.15 + 0.04 * phase + 0.03 * state;
                jacobian.Jt[0](state, state) = diagonal;
                jacobian.Jt_inv[0](state, state) = 1.0 / diagonal;
            }

            for (Index stage = 0; stage < dims.K; ++stage)
            {
                const Index variables =
                    dims.number_of_controls[stage]
                    + dims.number_of_states[stage];
                for (Index row = 0; row < variables; ++row)
                for (Index column = 0; column < variables; ++column)
                {
                    hessian.RSQrqt[stage](row, column) =
                        row == column
                        ? 4.5 + 0.25 * phase + 0.15 * stage
                              + 0.08 * row
                        : 0.015
                          / (1.0 + std::abs(row - column));
                }
            }

            for (Index row = 0; row < D_x.m(); ++row)
                D_x(row) = 0.18 + 0.015 * ((row + phase) % 4);
            D_e = 0.0;
            if (stabilized)
                D_e(info.offsets_g_eq_path[0]) =
                    0.055 + 0.005 * phase;
            D_e(info.offsets_g_eq_slack[0]) =
                0.035 + 0.004 * phase;

            slack_lower_distance(0) = 1.2 + 0.1 * phase;
            slack_upper_distance(0) = 1.5 + 0.08 * phase;
            slack_lower_dual(0) = 0.75 + 0.05 * phase;
            slack_upper_dual(0) = 0.65 + 0.04 * phase;
            rhs_slack(0) = -0.09 + 0.025 * phase;
            rhs_lower_complementarity(0) = 0.13 + 0.02 * phase;
            rhs_upper_complementarity(0) = -0.08 + 0.015 * phase;

            for (Index row = 0; row < rhs_primal.m(); ++row)
                rhs_primal(row) =
                    0.07 * (row + 1)
                    - 0.025 * ((row + phase) % 3);
            for (Index row = 0; row < rhs_constraints.m(); ++row)
                rhs_constraints(row) =
                    -0.045 * (row + 1)
                    + 0.018 * ((row + phase) % 4);

            cross_hessian = 0.0;
            border_jacobian = 0.0;
            for (Index row = 0; row < cross_hessian.m(); ++row)
            for (Index column = 0; column < incident_columns; ++column)
                cross_hessian(row, column) =
                    0.008
                    * (1 + ((row + 2 * column + phase) % 6))
                    - 0.021;
            for (Index row = 0; row < border_jacobian.m(); ++row)
            for (Index column = 0; column < incident_columns; ++column)
                border_jacobian(row, column) =
                    0.007
                    * (1 + ((2 * row + column + phase) % 5))
                    - 0.016;
            solution = 0.0;
        }

        Index phase;
        bool stabilized;
        ProblemDims dims;
        ProblemInfo info;
        Jacobian<ImplicitOcpType> jacobian;
        Hessian<ImplicitOcpType> hessian;
        VecRealAllocated D_x;
        VecRealAllocated D_e;
        VecRealAllocated slack_lower_distance;
        VecRealAllocated slack_upper_distance;
        VecRealAllocated slack_lower_dual;
        VecRealAllocated slack_upper_dual;
        MatRealAllocated cross_hessian;
        MatRealAllocated border_jacobian;
        VecRealAllocated rhs_primal;
        VecRealAllocated rhs_slack;
        VecRealAllocated rhs_constraints;
        VecRealAllocated rhs_lower_complementarity;
        VecRealAllocated rhs_upper_complementarity;
        VecRealAllocated solution;
    };

    std::vector<MatRealView> matrix_views(
        std::vector<MatRealAllocated> &storage)
    {
        std::vector<MatRealView> result;
        result.reserve(storage.size());
        for (MatRealAllocated &matrix : storage)
            result.push_back(matrix.block(matrix.m(), matrix.n(), 0, 0));
        return result;
    }

    std::vector<VecRealView> vector_views(
        std::vector<VecRealAllocated> &storage)
    {
        std::vector<VecRealView> result;
        result.reserve(storage.size());
        for (VecRealAllocated &vector : storage)
            result.push_back(vector.block(vector.m(), 0));
        return result;
    }

    Scalar normalized_backward_error(
        const VecRealView &residual,
        const VecRealView &applied,
        const VecRealView &rhs)
    {
        return norm_inf(residual)
            / (1.0 + norm_inf(applied) + norm_inf(rhs));
    }

    void run_case(const Index phase_count, const Index global_size)
    {
        std::vector<Index> phase_sizes;
        phase_sizes.reserve(static_cast<std::size_t>(phase_count));
        for (Index phase = 0; phase < phase_count; ++phase)
            phase_sizes.push_back(1 + phase % 3);

        std::vector<std::unique_ptr<PhaseData>> phases;
        std::vector<const ProblemInfo *> phase_info;
        phases.reserve(static_cast<std::size_t>(phase_count));
        phase_info.reserve(static_cast<std::size_t>(phase_count));
        for (Index phase = 0; phase < phase_count; ++phase)
        {
            const Index left = phase == 0 ? 0 : phase_sizes[phase - 1];
            const Index incident =
                left + phase_sizes[phase] + global_size;
            phases.push_back(std::make_unique<PhaseData>(
                phase, 1 + phase % 3, 1 + (2 * phase) % 3,
                incident, phase % 2 == 0));
            phase_info.push_back(&phases.back()->info);
        }

        std::vector<MatRealAllocated> diagonal;
        std::vector<MatRealAllocated> coupling;
        std::vector<MatRealAllocated> arrow;
        std::vector<VecRealAllocated> rhs_phase;
        std::vector<VecRealAllocated> solution_phase;
        for (Index phase = 0; phase < phase_count; ++phase)
        {
            const Index size = phase_sizes[phase];
            diagonal.emplace_back(size, size);
            arrow.emplace_back(size, global_size);
            rhs_phase.emplace_back(size);
            solution_phase.emplace_back(size);
            diagonal.back() = 0.0;
            arrow.back() = 0.0;
            solution_phase.back() = 0.0;
            for (Index row = 0; row < size; ++row)
            {
                for (Index column = 0; column < size; ++column)
                    diagonal.back()(row, column) =
                        row == column
                        ? 5.2 + 0.25 * phase + 0.12 * row
                        : 0.018
                          / (1.0 + std::abs(row - column));
                rhs_phase.back()(row) =
                    0.11 * (row + 1) - 0.035 * phase;
                for (Index global = 0; global < global_size; ++global)
                    arrow.back()(row, global) =
                        0.012
                        * (1 + ((row + global + phase) % 4))
                        - 0.022;
            }
        }
        for (Index phase = 0; phase + 1 < phase_count; ++phase)
        {
            coupling.emplace_back(
                phase_sizes[phase], phase_sizes[phase + 1]);
            for (Index row = 0; row < coupling.back().m(); ++row)
            for (Index column = 0; column < coupling.back().n(); ++column)
                coupling.back()(row, column) =
                    0.009
                    * (1 + ((row + 2 * column + phase) % 5))
                    - 0.017;
        }

        const Index global_storage_size = std::max<Index>(global_size, 1);
        MatRealAllocated global_hessian_storage(
            global_storage_size, global_storage_size);
        VecRealAllocated rhs_global_storage(global_storage_size);
        VecRealAllocated solution_global_storage(global_storage_size);
        global_hessian_storage = 0.0;
        rhs_global_storage = 0.0;
        solution_global_storage = 0.0;
        for (Index row = 0; row < global_size; ++row)
        {
            for (Index column = 0; column < global_size; ++column)
                global_hessian_storage(row, column) =
                    row == column
                    ? 6.0 + 0.2 * row
                    : 0.025 / (1.0 + std::abs(row - column));
            rhs_global_storage(row) = 0.14 * (row + 1) - 0.03;
        }

        std::vector<ImplicitPhaseArrowPdPhaseView> phase_views;
        phase_views.reserve(phases.size());
        for (auto &phase : phases)
        {
            phase_views.emplace_back(
                phase->info,
                phase->jacobian,
                phase->hessian,
                phase->D_x.block(phase->D_x.m(), 0),
                !phase->stabilized,
                phase->D_e.block(phase->D_e.m(), 0),
                phase->slack_lower_distance.block(
                    phase->slack_lower_distance.m(), 0),
                phase->slack_upper_distance.block(
                    phase->slack_upper_distance.m(), 0),
                phase->slack_lower_dual.block(
                    phase->slack_lower_dual.m(), 0),
                phase->slack_upper_dual.block(
                    phase->slack_upper_dual.m(), 0),
                phase->cross_hessian.block(
                    phase->cross_hessian.m(),
                    phase->cross_hessian.n(), 0, 0),
                phase->border_jacobian.block(
                    phase->border_jacobian.m(),
                    phase->border_jacobian.n(), 0, 0),
                phase->rhs_primal.block(phase->rhs_primal.m(), 0),
                phase->rhs_slack.block(phase->rhs_slack.m(), 0),
                phase->rhs_constraints.block(
                    phase->rhs_constraints.m(), 0),
                phase->rhs_lower_complementarity.block(
                    phase->rhs_lower_complementarity.m(), 0),
                phase->rhs_upper_complementarity.block(
                    phase->rhs_upper_complementarity.m(), 0),
                phase->solution.block(phase->solution.m(), 0));
        }

        std::vector<MatRealView> diagonal_views = matrix_views(diagonal);
        std::vector<MatRealView> coupling_views = matrix_views(coupling);
        std::vector<MatRealView> arrow_views = matrix_views(arrow);
        std::vector<VecRealView> rhs_phase_views = vector_views(rhs_phase);
        std::vector<VecRealView> solution_phase_views =
            vector_views(solution_phase);
        MatRealView global_hessian = global_hessian_storage.block(
            global_size, global_size, 0, 0);
        VecRealView rhs_global = rhs_global_storage.block(global_size, 0);
        VecRealView solution_global =
            solution_global_storage.block(global_size, 0);

        ImplicitPhaseArrowPdSolver solver(
            phase_info, phase_sizes, global_size);
        ASSERT_EQ(
            solver.solve(
                phase_views,
                diagonal_views,
                coupling_views,
                arrow_views,
                global_hessian,
                rhs_phase_views,
                rhs_global,
                solution_phase_views,
                solution_global),
            LinsolReturnFlag::SUCCESS);

        for (Index phase = 0; phase < phase_count; ++phase)
        {
            const PhaseData &data = *phases[phase];
            const Index variables =
                data.dims.number_of_controls[0]
                + data.dims.number_of_states[0];
            for (Index row = 0; row < variables; ++row)
            for (Index state = 0;
                 state < data.dims.number_of_states[1]; ++state)
            {
                const Scalar expected =
                    0.006
                    * (1 + ((2 * row + state + phase) % 4))
                    - 0.012;
                EXPECT_NEAR(data.hessian.FuFx[0](row, state), expected, 2e-12)
                    << "phase=" << phase
                    << ", row=" << row
                    << ", state=" << state;
            }
        }

        std::vector<VecRealAllocated> reduced_residual;
        reduced_residual.reserve(static_cast<std::size_t>(phase_count));
        for (Index phase = 0; phase < phase_count; ++phase)
        {
            reduced_residual.emplace_back(phase_sizes[phase]);
            reduced_residual.back() = rhs_phase[phase];
            for (Index row = 0; row < phase_sizes[phase]; ++row)
            for (Index column = 0; column < phase_sizes[phase]; ++column)
                reduced_residual.back()(row) +=
                    diagonal[phase](row, column)
                    * solution_phase[phase](column);
            for (Index row = 0; row < phase_sizes[phase]; ++row)
            for (Index global = 0; global < global_size; ++global)
                reduced_residual.back()(row) +=
                    arrow[phase](row, global)
                    * solution_global_storage(global);
        }
        for (Index phase = 0; phase + 1 < phase_count; ++phase)
        {
            for (Index row = 0; row < phase_sizes[phase]; ++row)
            for (Index column = 0; column < phase_sizes[phase + 1]; ++column)
            {
                const Scalar value = coupling[phase](row, column);
                reduced_residual[phase](row) +=
                    value * solution_phase[phase + 1](column);
                reduced_residual[phase + 1](column) +=
                    value * solution_phase[phase](row);
            }
        }
        VecRealAllocated global_residual(global_storage_size);
        global_residual = 0.0;
        for (Index row = 0; row < global_size; ++row)
        {
            global_residual(row) = rhs_global_storage(row);
            for (Index column = 0; column < global_size; ++column)
                global_residual(row) +=
                    global_hessian_storage(row, column)
                    * solution_global_storage(column);
            for (Index phase = 0; phase < phase_count; ++phase)
            for (Index local = 0; local < phase_sizes[phase]; ++local)
                global_residual(row) +=
                    arrow[phase](local, row)
                    * solution_phase[phase](local);
        }

        for (Index phase = 0; phase < phase_count; ++phase)
        {
            PhaseData &data = *phases[phase];
            LinearSystem<PdSystemType<ImplicitOcpType>> local_system(
                data.info,
                data.jacobian,
                data.hessian,
                data.D_x,
                !data.stabilized,
                data.D_e,
                data.slack_lower_distance,
                data.slack_upper_distance,
                data.slack_lower_dual,
                data.slack_upper_dual,
                data.rhs_primal,
                data.rhs_slack,
                data.rhs_constraints,
                data.rhs_lower_complementarity,
                data.rhs_upper_complementarity);
            VecRealAllocated local_rhs(local_system.m());
            VecRealAllocated local_applied(local_system.m());
            VecRealAllocated local_residual(local_system.m());
            VecRealAllocated local_zero(local_system.m());
            local_zero = 0.0;
            local_system.get_rhs(local_rhs);
            local_system.apply_on_right(
                data.solution, 0.0, local_zero, local_applied);
            local_residual = local_applied + local_rhs;

            const Index left = phase == 0 ? 0 : phase_sizes[phase - 1];
            const Index right = phase_sizes[phase];
            const Index incident_count = left + right + global_size;
            VecRealAllocated incident(std::max<Index>(incident_count, 1));
            for (Index row = 0; row < left; ++row)
                incident(row) = solution_phase[phase - 1](row);
            for (Index row = 0; row < right; ++row)
                incident(left + row) = solution_phase[phase](row);
            for (Index row = 0; row < global_size; ++row)
                incident(left + right + row) = solution_global_storage(row);

            for (Index row = 0;
                 row < data.info.number_of_primal_variables; ++row)
            for (Index column = 0; column < incident_count; ++column)
                local_residual(
                    data.info.pd_orig_offset_primal + row) +=
                    data.cross_hessian(row, column) * incident(column);
            for (Index row = 0;
                 row < data.info.number_of_primal_variables; ++row)
            for (Index column = 0; column < incident_count; ++column)
                local_applied(
                    data.info.pd_orig_offset_primal + row) +=
                    data.cross_hessian(row, column) * incident(column);
            for (Index row = 0;
                 row < data.info.number_of_eq_constraints; ++row)
            for (Index column = 0; column < incident_count; ++column)
                local_residual(
                    data.info.pd_orig_offset_mult + row) +=
                    data.border_jacobian(row, column) * incident(column);
            for (Index row = 0;
                 row < data.info.number_of_eq_constraints; ++row)
            for (Index column = 0; column < incident_count; ++column)
                local_applied(
                    data.info.pd_orig_offset_mult + row) +=
                    data.border_jacobian(row, column) * incident(column);

            VecRealView trajectory_step = data.solution.block(
                data.info.number_of_primal_variables,
                data.info.pd_orig_offset_primal);
            VecRealView multiplier_step = data.solution.block(
                data.info.number_of_eq_constraints,
                data.info.pd_orig_offset_mult);
            for (Index column = 0; column < incident_count; ++column)
            {
                Scalar contribution = 0.0;
                for (Index row = 0;
                     row < data.info.number_of_primal_variables; ++row)
                    contribution +=
                        data.cross_hessian(row, column)
                        * trajectory_step(row);
                for (Index row = 0;
                     row < data.info.number_of_eq_constraints; ++row)
                    contribution +=
                        data.border_jacobian(row, column)
                        * multiplier_step(row);

                if (column < left)
                    reduced_residual[phase - 1](column) += contribution;
                else if (column < left + right)
                    reduced_residual[phase](column - left) += contribution;
                else
                    global_residual(column - left - right) += contribution;
            }

            Index largest_row = 0;
            for (Index row = 1; row < local_residual.m(); ++row)
            {
                if (std::abs(local_residual(row))
                    > std::abs(local_residual(largest_row)))
                    largest_row = row;
            }
            EXPECT_LT(
                normalized_backward_error(
                    local_residual, local_applied, local_rhs), 2e-9)
                << "phase=" << phase
                << ", row=" << largest_row
                << ", value=" << local_residual(largest_row);
        }

        for (Index phase = 0; phase < phase_count; ++phase)
        {
            VecRealAllocated applied(phase_sizes[phase]);
            applied = reduced_residual[phase] - rhs_phase[phase];
            EXPECT_LT(
                normalized_backward_error(
                    reduced_residual[phase], applied, rhs_phase[phase]),
                2e-9)
                << "phase=" << phase;
        }
        VecRealView global_residual_view =
            global_residual.block(global_size, 0);
        VecRealAllocated global_applied(global_storage_size);
        global_applied = 0.0;
        for (Index row = 0; row < global_size; ++row)
            global_applied(row) =
                global_residual(row) - rhs_global_storage(row);
        VecRealView global_applied_view =
            global_applied.block(global_size, 0);
        EXPECT_LT(
            normalized_backward_error(
                global_residual_view, global_applied_view, rhs_global),
            2e-9);
    }
}

TEST(ImplicitPhaseArrowPdSolverTest,
     SolvesFullBarrierSystemAcrossPhaseAndParameterDimensions)
{
    for (const Index phases : {1, 2, 4})
    for (const Index global : {0, 2})
    {
        SCOPED_TRACE(
            "phases=" + std::to_string(phases)
            + ", global=" + std::to_string(global));
        run_case(phases, global);
    }
}
