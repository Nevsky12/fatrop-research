//
// Copyright (c) 2026
//

#include "fatrop/ocp/interval_scoped_pd_solver.hpp"
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
            const Index nx,
            const Index nu,
            const Index incident_columns,
            const bool stabilized_value)
            : phase(phase_value),
              stabilized(stabilized_value),
              dims(
                  2,
                  std::vector<Index>{nu, 0},
                  std::vector<Index>{nx, nx},
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

            const Index current_variables = nu + nx;
            for (Index row = 0; row < current_variables; ++row)
            {
                for (Index state = 0; state < nx; ++state)
                {
                    jacobian.BAbt[0](row, state) =
                        0.025 * (1 + ((row + 2 * state + phase) % 5))
                      - 0.045;
                    hessian.FuFx[0](row, state) =
                        0.006 * (1 + ((2 * row + state + phase) % 4))
                      - 0.012;
                }
                jacobian.Gg_eqt[0](row, 0) =
                    0.04 + 0.01 * ((row + phase) % 4);
                jacobian.Gg_ineqt[0](row, 0) =
                    -0.035 + 0.012 * ((2 * row + phase) % 5);
            }
            for (Index state = 0; state < nx; ++state)
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
                {
                    for (Index column = 0;
                         column < variables; ++column)
                    {
                        hessian.RSQrqt[stage](row, column) =
                            row == column
                            ? 4.5 + 0.25 * phase + 0.15 * stage + 0.08 * row
                            : 0.015 / (1.0 + std::abs(row - column));
                    }
                }
            }

            for (Index row = 0; row < D_x.m(); ++row)
                D_x(row) = 0.18 + 0.015 * ((row + phase) % 4);
            D_e = 0.0;
            if (stabilized)
                D_e(info.offsets_g_eq_path[0]) = 0.055 + 0.005 * phase;
            D_e(info.offsets_g_eq_slack[0]) = 0.035 + 0.004 * phase;

            slack_lower_distance(0) = 1.2 + 0.1 * phase;
            slack_upper_distance(0) = 1.5 + 0.08 * phase;
            slack_lower_dual(0) = 0.75 + 0.05 * phase;
            slack_upper_dual(0) = 0.65 + 0.04 * phase;
            rhs_slack(0) = -0.09 + 0.025 * phase;
            rhs_lower_complementarity(0) = 0.13 + 0.02 * phase;
            rhs_upper_complementarity(0) = -0.08 + 0.015 * phase;
            for (Index row = 0; row < rhs_primal.m(); ++row)
                rhs_primal(row) =
                    0.07 * (row + 1) - 0.025 * ((row + phase) % 3);
            for (Index row = 0; row < rhs_constraints.m(); ++row)
                rhs_constraints(row) =
                    -0.045 * (row + 1) + 0.018 * ((row + phase) % 4);

            cross_hessian = 0.0;
            border_jacobian = 0.0;
            for (Index row = 0; row < cross_hessian.m(); ++row)
            for (Index column = 0; column < incident_columns; ++column)
                cross_hessian(row, column) =
                    0.008 * (1 + ((row + 2 * column + phase) % 6))
                  - 0.021;
            for (Index row = 0; row < border_jacobian.m(); ++row)
            for (Index column = 0; column < incident_columns; ++column)
                border_jacobian(row, column) =
                    0.007 * (1 + ((2 * row + column + phase) % 5))
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

    std::vector<IntervalScope> make_scopes(const Index phases)
    {
        std::vector<IntervalScope> scopes;
        for (Index phase = 0; phase < phases; ++phase)
            scopes.push_back({1 + phase % 2, phase, phase});
        for (Index phase = 0; phase + 1 < phases; ++phase)
            scopes.push_back({1, phase, phase + 1});
        if (phases >= 3)
            scopes.push_back({2, 0, 2});
        scopes.push_back({2, 0, phases - 1});
        return scopes;
    }

    std::vector<Index> active_blocks(
        const std::vector<IntervalScope> &scopes,
        const Index phase)
    {
        std::vector<Index> result;
        for (Index block = 0;
             block < static_cast<Index>(scopes.size()); ++block)
        {
            const IntervalScope &scope = scopes[
                static_cast<std::size_t>(block)];
            if (scope.first_phase <= phase && phase <= scope.last_phase)
                result.push_back(block);
        }
        return result;
    }

    Scalar normalized_error(
        const VecRealView &residual,
        const VecRealView &applied,
        const VecRealView &rhs)
    {
        return norm_inf(residual)
            / (1.0 + norm_inf(applied) + norm_inf(rhs));
    }

    void run_case(const Index phase_count)
    {
        const std::vector<IntervalScope> scopes = make_scopes(phase_count);
        std::vector<std::unique_ptr<PhaseData>> phases;
        std::vector<const ProblemInfo *> phase_info;
        for (Index phase = 0; phase < phase_count; ++phase)
        {
            Index incident = 0;
            for (const Index block : active_blocks(scopes, phase))
                incident += scopes[static_cast<std::size_t>(block)].dimension;
            phases.push_back(std::make_unique<PhaseData>(
                phase, 1 + phase % 3, 1 + (2 * phase) % 3,
                incident, phase % 2 == 0));
            phase_info.push_back(&phases.back()->info);
        }

        IntervalScopedImplicitPdSolver solver(phase_info, scopes);
        solver.phase_solver().reduced_solver().set_pivot_tolerance(1e-14);
        for (Index phase = 0; phase < phase_count; ++phase)
            solver.phase_solver().trajectory_solver(phase)
                .set_lu_fact_tol(1e-12);

        std::vector<MatRealAllocated> diagonal;
        std::vector<VecRealAllocated> rhs_blocks;
        std::vector<VecRealAllocated> solutions;
        diagonal.reserve(scopes.size());
        rhs_blocks.reserve(scopes.size());
        solutions.reserve(scopes.size());
        for (std::size_t block = 0; block < scopes.size(); ++block)
        {
            const Index dimension = scopes[block].dimension;
            diagonal.emplace_back(dimension, dimension);
            rhs_blocks.emplace_back(dimension);
            solutions.emplace_back(dimension);
            diagonal.back() = 0.0;
            solutions.back() = 0.0;
            for (Index row = 0; row < dimension; ++row)
            {
                for (Index column = 0; column < dimension; ++column)
                    diagonal.back()(row, column) = row == column
                        ? 6.0 + 0.25 * block + 0.1 * row
                        : 0.018 / (1.0 + std::abs(row - column));
                rhs_blocks.back()(row) =
                    0.12 * (row + 1) - 0.025 * block;
            }
        }

        const std::vector<IntervalKktEdge> &edges =
            solver.phase_solver().reduced_solver().edges();
        std::vector<MatRealAllocated> edge_storage;
        edge_storage.reserve(edges.size());
        for (const IntervalKktEdge edge : edges)
        {
            const Index rows = scopes[static_cast<std::size_t>(
                edge.first_block)].dimension;
            const Index columns = scopes[static_cast<std::size_t>(
                edge.second_block)].dimension;
            edge_storage.emplace_back(rows, columns);
            for (Index row = 0; row < rows; ++row)
            for (Index column = 0; column < columns; ++column)
                edge_storage.back()(row, column) =
                    0.009
                    * (1 + ((row + 2 * column
                        + edge.first_block + edge.second_block) % 5))
                    - 0.017;
        }

        std::vector<IntervalScopedImplicitPdPhaseView> phase_views;
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

        std::vector<MatRealView> diagonal_views;
        std::vector<MatRealView> edge_views;
        std::vector<VecRealView> rhs_views;
        std::vector<VecRealView> solution_views;
        for (std::size_t block = 0; block < scopes.size(); ++block)
        {
            const Index dimension = scopes[block].dimension;
            diagonal_views.push_back(
                diagonal[block].block(dimension, dimension, 0, 0));
            rhs_views.push_back(rhs_blocks[block].block(dimension, 0));
            solution_views.push_back(solutions[block].block(dimension, 0));
        }
        for (std::size_t edge = 0; edge < edges.size(); ++edge)
        {
            edge_views.push_back(edge_storage[edge].block(
                scopes[static_cast<std::size_t>(
                    edges[edge].first_block)].dimension,
                scopes[static_cast<std::size_t>(
                    edges[edge].second_block)].dimension,
                0, 0));
        }

        ASSERT_EQ(
            solver.solve(
                phase_views,
                diagonal_views,
                edge_views,
                rhs_views,
                solution_views),
            LinsolReturnFlag::SUCCESS);

        std::vector<VecRealAllocated> reduced_residual;
        std::vector<VecRealAllocated> reduced_applied;
        reduced_residual.reserve(scopes.size());
        reduced_applied.reserve(scopes.size());
        for (std::size_t block = 0; block < scopes.size(); ++block)
        {
            const Index dimension = scopes[block].dimension;
            reduced_residual.emplace_back(dimension);
            reduced_applied.emplace_back(dimension);
            reduced_residual.back() = rhs_blocks[block];
            reduced_applied.back() = 0.0;
            for (Index row = 0; row < dimension; ++row)
            for (Index column = 0; column < dimension; ++column)
            {
                const Scalar value =
                    diagonal[block](row, column) * solutions[block](column);
                reduced_residual.back()(row) += value;
                reduced_applied.back()(row) += value;
            }
        }
        for (std::size_t edge_index = 0;
             edge_index < edges.size(); ++edge_index)
        {
            const IntervalKktEdge edge = edges[edge_index];
            const Index rows = scopes[static_cast<std::size_t>(
                edge.first_block)].dimension;
            const Index columns = scopes[static_cast<std::size_t>(
                edge.second_block)].dimension;
            for (Index row = 0; row < rows; ++row)
            for (Index column = 0; column < columns; ++column)
            {
                const Scalar value = edge_storage[edge_index](row, column);
                const Scalar first_update =
                    value * solutions[static_cast<std::size_t>(
                        edge.second_block)](column);
                const Scalar second_update =
                    value * solutions[static_cast<std::size_t>(
                        edge.first_block)](row);
                reduced_residual[static_cast<std::size_t>(
                    edge.first_block)](row) += first_update;
                reduced_applied[static_cast<std::size_t>(
                    edge.first_block)](row) += first_update;
                reduced_residual[static_cast<std::size_t>(
                    edge.second_block)](column) += second_update;
                reduced_applied[static_cast<std::size_t>(
                    edge.second_block)](column) += second_update;
            }
        }

        for (Index phase = 0; phase < phase_count; ++phase)
        {
            PhaseData &data = *phases[static_cast<std::size_t>(phase)];
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
            VecRealAllocated zero(local_system.m());
            zero = 0.0;
            local_system.get_rhs(local_rhs);
            local_system.apply_on_right(
                data.solution, 0.0, zero, local_applied);
            local_residual = local_applied + local_rhs;

            Index incident_dimension = 0;
            for (const Index block : active_blocks(scopes, phase))
                incident_dimension += scopes[static_cast<std::size_t>(
                    block)].dimension;
            VecRealAllocated incident(std::max<Index>(incident_dimension, 1));
            Index incident_offset = 0;
            for (const Index block : active_blocks(scopes, phase))
            {
                const Index dimension = scopes[
                    static_cast<std::size_t>(block)].dimension;
                for (Index row = 0; row < dimension; ++row)
                    incident(incident_offset + row) =
                        solutions[static_cast<std::size_t>(block)](row);
                incident_offset += dimension;
            }

            for (Index row = 0;
                 row < data.info.number_of_primal_variables; ++row)
            for (Index column = 0; column < incident_dimension; ++column)
            {
                const Scalar update =
                    data.cross_hessian(row, column) * incident(column);
                local_residual(data.info.pd_orig_offset_primal + row) += update;
                local_applied(data.info.pd_orig_offset_primal + row) += update;
            }
            for (Index row = 0;
                 row < data.info.number_of_eq_constraints; ++row)
            for (Index column = 0; column < incident_dimension; ++column)
            {
                const Scalar update =
                    data.border_jacobian(row, column) * incident(column);
                local_residual(data.info.pd_orig_offset_mult + row) += update;
                local_applied(data.info.pd_orig_offset_mult + row) += update;
            }

            VecRealView trajectory_step = data.solution.block(
                data.info.number_of_primal_variables,
                data.info.pd_orig_offset_primal);
            VecRealView multiplier_step = data.solution.block(
                data.info.number_of_eq_constraints,
                data.info.pd_orig_offset_mult);
            incident_offset = 0;
            for (const Index block : active_blocks(scopes, phase))
            {
                const Index dimension = scopes[
                    static_cast<std::size_t>(block)].dimension;
                for (Index column = 0; column < dimension; ++column)
                {
                    Scalar contribution = 0.0;
                    for (Index row = 0;
                         row < data.info.number_of_primal_variables; ++row)
                        contribution += data.cross_hessian(
                            row, incident_offset + column)
                            * trajectory_step(row);
                    for (Index row = 0;
                         row < data.info.number_of_eq_constraints; ++row)
                        contribution += data.border_jacobian(
                            row, incident_offset + column)
                            * multiplier_step(row);
                    reduced_residual[static_cast<std::size_t>(block)](column)
                        += contribution;
                    reduced_applied[static_cast<std::size_t>(block)](column)
                        += contribution;
                }
                incident_offset += dimension;
            }

            EXPECT_LT(
                normalized_error(
                    local_residual, local_applied, local_rhs),
                3e-9)
                << "phase=" << phase;
        }

        for (std::size_t block = 0; block < scopes.size(); ++block)
        {
            EXPECT_LT(
                normalized_error(
                    reduced_residual[block],
                    reduced_applied[block],
                    rhs_blocks[block]),
                3e-9)
                << "block=" << block;
        }

        // Mehrotra's affine and corrector systems share their matrix.  Verify
        // that the retained-factor RHS path is numerically identical to a
        // complete refactorization for a genuinely different residual.
        for (auto &phase : phases)
        {
            for (Index row = 0; row < phase->rhs_primal.m(); ++row)
                phase->rhs_primal(row) += 0.013 * (row + 1);
            for (Index row = 0; row < phase->rhs_slack.m(); ++row)
            {
                phase->rhs_slack(row) -= 0.009 * (row + 1);
                phase->rhs_lower_complementarity(row) +=
                    0.007 * (row + 1);
                phase->rhs_upper_complementarity(row) -=
                    0.006 * (row + 1);
            }
            for (Index row = 0; row < phase->rhs_constraints.m(); ++row)
                phase->rhs_constraints(row) += 0.005 * (row + 1);
        }
        for (std::size_t block = 0; block < rhs_blocks.size(); ++block)
        for (Index row = 0; row < rhs_blocks[block].m(); ++row)
            rhs_blocks[block](row) -=
                0.011 * (row + 1) + 0.003 * block;

        ASSERT_EQ(
            solver.solve_rhs(
                phase_views, rhs_views, solution_views),
            LinsolReturnFlag::SUCCESS);
        std::vector<std::vector<Scalar>> reused_phase_solution;
        reused_phase_solution.reserve(phases.size());
        for (const auto &phase : phases)
        {
            reused_phase_solution.emplace_back(
                static_cast<std::size_t>(phase->solution.m()));
            for (Index row = 0; row < phase->solution.m(); ++row)
                reused_phase_solution.back()[static_cast<std::size_t>(row)] =
                    phase->solution(row);
        }
        std::vector<std::vector<Scalar>> reused_scoped_solution;
        reused_scoped_solution.reserve(solutions.size());
        for (const VecRealAllocated &solution : solutions)
        {
            reused_scoped_solution.emplace_back(
                static_cast<std::size_t>(solution.m()));
            for (Index row = 0; row < solution.m(); ++row)
                reused_scoped_solution.back()[static_cast<std::size_t>(row)] =
                    solution(row);
        }

        IntervalScopedImplicitPdSolver fresh_solver(phase_info, scopes);
        fresh_solver.phase_solver().reduced_solver()
            .set_pivot_tolerance(1e-14);
        for (Index phase = 0; phase < phase_count; ++phase)
            fresh_solver.phase_solver().trajectory_solver(phase)
                .set_lu_fact_tol(1e-12);
        std::vector<VecRealAllocated> fresh_solutions;
        std::vector<VecRealView> fresh_solution_views;
        fresh_solutions.reserve(scopes.size());
        fresh_solution_views.reserve(scopes.size());
        for (const IntervalScope &scope : scopes)
        {
            fresh_solutions.emplace_back(scope.dimension);
            fresh_solutions.back() = 0.0;
            fresh_solution_views.push_back(
                fresh_solutions.back().block(scope.dimension, 0));
        }
        ASSERT_EQ(
            fresh_solver.solve(
                phase_views,
                diagonal_views,
                edge_views,
                rhs_views,
                fresh_solution_views),
            LinsolReturnFlag::SUCCESS);
        for (std::size_t phase = 0; phase < phases.size(); ++phase)
        for (Index row = 0; row < phases[phase]->solution.m(); ++row)
            EXPECT_NEAR(
                phases[phase]->solution(row),
                reused_phase_solution[phase][static_cast<std::size_t>(row)],
                3e-9);
        for (std::size_t block = 0; block < scopes.size(); ++block)
        for (Index row = 0; row < fresh_solutions[block].m(); ++row)
            EXPECT_NEAR(
                fresh_solutions[block](row),
                reused_scoped_solution[block][static_cast<std::size_t>(row)],
                3e-9);
    }
} // namespace

TEST(IntervalScopedImplicitPdSolverTest,
     SolvesFullBarrierSystemAcrossPhaseAndScopeDimensions)
{
    for (const Index phases : {Index{1}, Index{2}, Index{4}})
    {
        SCOPED_TRACE("phases=" + std::to_string(phases));
        run_case(phases);
    }
}
