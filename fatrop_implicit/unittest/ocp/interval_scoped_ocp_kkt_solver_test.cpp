//
// Copyright (c) 2026
//

#include "fatrop/ocp/interval_scoped_ocp_kkt_solver.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

using namespace fatrop;

namespace
{
    struct ImplicitPhaseData
    {
        ImplicitPhaseData(
            const Index phase_value,
            const Index nx_value,
            const Index nu_value,
            const Index incident_columns)
            : phase(phase_value),
              nx(nx_value),
              nu(nu_value),
              dims(
                  3,
                  std::vector<Index>{nu, nu, 0},
                  std::vector<Index>{nx, nx, nx},
                  std::vector<Index>{1, 0, 1},
                  std::vector<Index>{0, 0, 0}),
              info(dims),
              jacobian(dims),
              hessian(dims),
              D_x(info.number_of_primal_variables),
              D_eq(info.number_of_g_eq_path),
              D_s(info.number_of_slack_variables),
              rhs_primal(info.number_of_primal_variables),
              rhs_constraints(info.number_of_eq_constraints),
              cross_hessian(
                  info.number_of_primal_variables,
                  incident_columns),
              border_jacobian(
                  info.number_of_eq_constraints,
                  incident_columns),
              primal(info.number_of_primal_variables),
              multipliers(info.number_of_eq_constraints),
              dense_k(Eigen::MatrixXd::Zero(
                  info.number_of_primal_variables
                    + info.number_of_eq_constraints,
                  info.number_of_primal_variables
                    + info.number_of_eq_constraints)),
              dense_b(Eigen::MatrixXd::Zero(
                  info.number_of_primal_variables
                    + info.number_of_eq_constraints,
                  incident_columns))
        {
            const Index primal_count = info.number_of_primal_variables;
            const Index constraint_count = info.number_of_eq_constraints;
            for (Index stage = 0; stage < dims.K; ++stage)
            {
                jacobian.Gg_eqt[stage] = 0.0;
                const Index variables =
                    dims.number_of_controls[stage]
                  + dims.number_of_states[stage];
                const Index offset = info.offsets_primal_u[stage];
                hessian.RSQrqt[stage] = 0.0;
                for (Index row = 0; row < variables; ++row)
                {
                    for (Index column = 0;
                         column < variables; ++column)
                    {
                        const Scalar value = row == column
                            ? 4.0 + 0.3 * phase + 0.2 * stage + 0.1 * row
                            : 0.018 / (1.0 + std::abs(row - column));
                        hessian.RSQrqt[stage](row, column) = value;
                        dense_k(offset + row, offset + column) = value;
                    }
                    D_x(offset + row) =
                        0.16 + 0.01 * ((offset + row + phase) % 4);
                    dense_k(offset + row, offset + row) +=
                        D_x(offset + row);
                    rhs_primal(offset + row) =
                        0.06 * (1 + offset + row)
                      - 0.025 * ((row + phase) % 3);
                }

                if (dims.number_of_eq_constraints[stage] > 0)
                {
                    const Index path_row = info.offsets_g_eq_path[stage];
                    for (Index variable = 0;
                         variable < variables; ++variable)
                    {
                        const Scalar value =
                            0.07
                          + 0.015 * ((variable + stage + phase) % 4);
                        jacobian.Gg_eqt[stage](variable, 0) = value;
                        dense_k(offset + variable,
                                primal_count + path_row) = value;
                        dense_k(primal_count + path_row,
                                offset + variable) = value;
                    }
                }

                if (stage + 1 == dims.K)
                    continue;
                jacobian.BAbt[stage] = 0.0;
                jacobian.Jt[stage] = 0.0;
                hessian.FuFx[stage] = 0.0;
                hessian.GuGx[stage] = 0.0;
                const Index dynamics_offset =
                    info.offsets_g_eq_dyn[stage];
                const Index next_state_offset =
                    info.offsets_primal_x[stage + 1];
                for (Index variable = 0;
                     variable < variables; ++variable)
                {
                    for (Index equation = 0; equation < nx; ++equation)
                    {
                        const Scalar value =
                            0.025
                          * (1 + ((variable + 2 * equation
                                  + stage + phase) % 6))
                          - 0.055;
                        jacobian.BAbt[stage](variable, equation) = value;
                        dense_k(offset + variable,
                                primal_count + dynamics_offset + equation) =
                            value;
                        dense_k(primal_count + dynamics_offset + equation,
                                offset + variable) = value;
                    }
                }
                for (Index state = 0; state < nx; ++state)
                {
                    for (Index equation = 0; equation < nx; ++equation)
                    {
                        Scalar value = state == equation
                            ? 1.15 + 0.05 * stage + 0.02 * state
                            : 0.08 / (1.0 + std::abs(state - equation));
                        if (stage == 0 && nx > 1)
                        {
                            if (state == 0 && equation == 0) value = 0.12;
                            if (state == 0 && equation == 1) value = 1.08;
                            if (state == 1 && equation == 0) value = 1.02;
                        }
                        jacobian.Jt[stage](state, equation) = value;
                        dense_k(next_state_offset + state,
                                primal_count + dynamics_offset + equation) =
                            value;
                        dense_k(primal_count + dynamics_offset + equation,
                                next_state_offset + state) = value;
                    }
                }
            }

            D_eq = 0.0;
            for (Index row = 0; row < info.number_of_g_eq_path; ++row)
                D_eq(row) = 0.035 + 0.004 * ((row + phase) % 5);
            for (Index row = 0; row < constraint_count; ++row)
            {
                rhs_constraints(row) =
                    -0.045 * (row + 1)
                  + 0.018 * ((row + phase) % 4);
            }
            for (Index stage = 0; stage < dims.K; ++stage)
            {
                for (Index equation = 0;
                     equation < dims.number_of_eq_constraints[stage];
                     ++equation)
                {
                    const Index row =
                        info.offsets_g_eq_path[stage] + equation;
                    const Index diagonal =
                        info.offsets_eq[stage] + equation;
                    dense_k(primal_count + row, primal_count + row) =
                        -D_eq(diagonal);
                }
            }

            D_s = 0.0;
            cross_hessian = 0.0;
            border_jacobian = 0.0;
            primal = 0.0;
            multipliers = 0.0;
            for (Index row = 0; row < primal_count; ++row)
            {
                for (Index column = 0;
                     column < incident_columns; ++column)
                {
                    const Scalar value =
                        0.011 * (1 + ((row + 2 * column + phase) % 7))
                      - 0.028;
                    cross_hessian(row, column) = value;
                    dense_b(row, column) = value;
                }
            }
            for (Index row = 0; row < constraint_count; ++row)
            {
                for (Index column = 0;
                     column < incident_columns; ++column)
                {
                    const Scalar value =
                        0.009 * (1 + ((2 * row + column + phase) % 6))
                      - 0.020;
                    border_jacobian(row, column) = value;
                    dense_b(primal_count + row, column) = value;
                }
            }
        }

        void make_rank_deficient()
        {
            fatrop_assert_msg(
                nx >= 2,
                "The rank-deficient test phase needs at least two states.");
            const Index primal_count = info.number_of_primal_variables;
            for (Index stage = 0; stage + 1 < dims.K; ++stage)
            {
                const Index variables =
                    dims.number_of_controls[stage]
                  + dims.number_of_states[stage];
                const Index current_offset = info.offsets_primal_u[stage];
                const Index next_offset = info.offsets_primal_x[stage + 1];
                const Index dynamics_offset = info.offsets_g_eq_dyn[stage];
                jacobian.BAbt[stage] = 0.0;
                jacobian.Jt[stage] = 0.0;
                for (Index equation = 0; equation < nx; ++equation)
                {
                    for (Index variable = 0;
                         variable < variables; ++variable)
                    {
                        dense_k(
                            current_offset + variable,
                            primal_count + dynamics_offset + equation) = 0.0;
                        dense_k(
                            primal_count + dynamics_offset + equation,
                            current_offset + variable) = 0.0;
                    }
                    for (Index state = 0; state < nx; ++state)
                    {
                        dense_k(
                            next_offset + state,
                            primal_count + dynamics_offset + equation) = 0.0;
                        dense_k(
                            primal_count + dynamics_offset + equation,
                            next_offset + state) = 0.0;
                    }
                }

                for (Index equation = 0; equation + 1 < nx; ++equation)
                {
                    const Scalar value = -1.0 - 0.05 * equation;
                    jacobian.Jt[stage](equation, equation) = value;
                    dense_k(
                        next_offset + equation,
                        primal_count + dynamics_offset + equation) = value;
                    dense_k(
                        primal_count + dynamics_offset + equation,
                        next_offset + equation) = value;
                }
                const Index missing_equation = nx - 1;
                jacobian.BAbt[stage](0, missing_equation) = 1.0;
                dense_k(
                    current_offset,
                    primal_count + dynamics_offset + missing_equation) = 1.0;
                dense_k(
                    primal_count + dynamics_offset + missing_equation,
                    current_offset) = 1.0;
                for (Index equation = 0; equation + 1 < nx; ++equation)
                {
                    jacobian.BAbt[stage](0, equation) =
                        0.12 + 0.02 * equation;
                    dense_k(
                        current_offset,
                        primal_count + dynamics_offset + equation) =
                        jacobian.BAbt[stage](0, equation);
                    dense_k(
                        primal_count + dynamics_offset + equation,
                        current_offset) =
                        jacobian.BAbt[stage](0, equation);
                }
            }
        }

        Index phase;
        Index nx;
        Index nu;
        ProblemDims dims;
        ProblemInfo info;
        Jacobian<ImplicitOcpType> jacobian;
        Hessian<ImplicitOcpType> hessian;
        VecRealAllocated D_x;
        VecRealAllocated D_eq;
        VecRealAllocated D_s;
        VecRealAllocated rhs_primal;
        VecRealAllocated rhs_constraints;
        MatRealAllocated cross_hessian;
        MatRealAllocated border_jacobian;
        VecRealAllocated primal;
        VecRealAllocated multipliers;
        Eigen::MatrixXd dense_k;
        Eigen::MatrixXd dense_b;
    };

    std::vector<IntervalScope> make_scopes(
        const Index phases,
        const Index global_dimension)
    {
        std::vector<IntervalScope> scopes;
        for (Index phase = 0; phase < phases; ++phase)
            scopes.push_back({1 + (phase % 3), phase, phase});
        for (Index phase = 0; phase + 1 < phases; ++phase)
            scopes.push_back({1 + (phase % 2), phase, phase + 1});
        for (Index first = 0; first + 2 < phases; first += 2)
            scopes.push_back({2, first, first + 2});
        if (global_dimension > 0)
            scopes.push_back({global_dimension, 0, phases - 1});
        return scopes;
    }

    std::vector<Index> block_offsets(
        const std::vector<IntervalScope> &scopes)
    {
        std::vector<Index> result(scopes.size(), 0);
        for (std::size_t block = 1; block < scopes.size(); ++block)
            result[block] = result[block - 1]
                          + scopes[block - 1].dimension;
        return result;
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

    void run_case(
        const std::vector<Index> &nx,
        const std::vector<Index> &nu,
        const Index global_dimension,
        const bool rank_deficient = false)
    {
        ASSERT_EQ(nx.size(), nu.size());
        ASSERT_FALSE(nx.empty());
        const Index phase_count = static_cast<Index>(nx.size());
        const std::vector<IntervalScope> scopes =
            make_scopes(phase_count, global_dimension);
        const std::vector<Index> scoped_offsets = block_offsets(scopes);
        Index scoped_dimension = 0;
        for (const IntervalScope &scope : scopes)
            scoped_dimension += scope.dimension;

        std::vector<std::unique_ptr<ImplicitPhaseData>> phases;
        std::vector<const ProblemInfo *> phase_info;
        phases.reserve(static_cast<std::size_t>(phase_count));
        phase_info.reserve(static_cast<std::size_t>(phase_count));
        for (Index phase = 0; phase < phase_count; ++phase)
        {
            Index incident = 0;
            for (const Index block : active_blocks(scopes, phase))
                incident += scopes[static_cast<std::size_t>(block)].dimension;
            phases.push_back(std::make_unique<ImplicitPhaseData>(
                phase,
                nx[static_cast<std::size_t>(phase)],
                nu[static_cast<std::size_t>(phase)],
                incident));
            if (rank_deficient)
                phases.back()->make_rank_deficient();
            phase_info.push_back(&phases.back()->info);
        }

        IntervalScopedImplicitOcpKktSolver solver(phase_info, scopes);
        solver.reduced_solver().set_pivot_tolerance(1e-14);
        for (Index phase = 0; phase < phase_count; ++phase)
        {
            solver.trajectory_solver(phase).set_lu_fact_tol(1e-12);
            EXPECT_EQ(
                solver.active_blocks(phase),
                active_blocks(scopes, phase));
        }

        std::vector<MatRealAllocated> diagonal;
        std::vector<VecRealAllocated> rhs_scoped;
        std::vector<VecRealAllocated> solution_scoped;
        diagonal.reserve(scopes.size());
        rhs_scoped.reserve(scopes.size());
        solution_scoped.reserve(scopes.size());
        for (std::size_t block = 0; block < scopes.size(); ++block)
        {
            const Index dimension = scopes[block].dimension;
            const Index allocation = std::max<Index>(dimension, 1);
            diagonal.emplace_back(allocation, allocation);
            rhs_scoped.emplace_back(allocation);
            solution_scoped.emplace_back(allocation);
            diagonal.back() = 0.0;
            rhs_scoped.back() = 0.0;
            solution_scoped.back() = 0.0;
            for (Index row = 0; row < dimension; ++row)
            {
                for (Index column = 0; column < dimension; ++column)
                {
                    diagonal.back()(row, column) = row == column
                        ? ((block + static_cast<std::size_t>(row)) % 3 == 2
                            ? -5.4 - 0.2 * block
                            : 6.0 + 0.3 * block + 0.1 * row)
                        : 0.017 / (1.0 + std::abs(row - column));
                }
                rhs_scoped.back()(row) =
                    0.11 * (1 + scoped_offsets[block] + row)
                  - 0.04 * ((block + row) % 3);
            }
        }

        std::vector<MatRealAllocated> edge_storage;
        edge_storage.reserve(solver.reduced_solver().edges().size());
        for (const IntervalKktEdge edge : solver.reduced_solver().edges())
        {
            const Index rows = scopes[static_cast<std::size_t>(
                edge.first_block)].dimension;
            const Index columns = scopes[static_cast<std::size_t>(
                edge.second_block)].dimension;
            edge_storage.emplace_back(
                std::max<Index>(rows, 1),
                std::max<Index>(columns, 1));
            edge_storage.back() = 0.0;
            for (Index row = 0; row < rows; ++row)
            {
                for (Index column = 0; column < columns; ++column)
                {
                    edge_storage.back()(row, column) =
                        0.007
                      * (1 + ((row + 2 * column
                          + edge.first_block + edge.second_block) % 5))
                      - 0.013;
                }
            }
        }

        Index trajectory_dimension = 0;
        std::vector<Index> trajectory_offsets(
            static_cast<std::size_t>(phase_count), 0);
        for (Index phase = 0; phase < phase_count; ++phase)
        {
            trajectory_offsets[static_cast<std::size_t>(phase)] =
                trajectory_dimension;
            trajectory_dimension += static_cast<Index>(
                phases[static_cast<std::size_t>(phase)]->dense_k.rows());
        }
        const Index total = trajectory_dimension + scoped_dimension;
        Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(total, total);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(total);

        for (Index phase = 0; phase < phase_count; ++phase)
        {
            const ImplicitPhaseData &current =
                *phases[static_cast<std::size_t>(phase)];
            const Index y_size = static_cast<Index>(current.dense_k.rows());
            const Index y_offset = trajectory_offsets[
                static_cast<std::size_t>(phase)];
            dense.block(y_offset, y_offset, y_size, y_size) = current.dense_k;
            for (Index row = 0;
                 row < current.info.number_of_primal_variables; ++row)
                rhs(y_offset + row) = -current.rhs_primal(row);
            for (Index row = 0;
                 row < current.info.number_of_eq_constraints; ++row)
            {
                rhs(y_offset + current.info.number_of_primal_variables + row) =
                    -current.rhs_constraints(row);
            }

            Index incident_offset = 0;
            for (const Index block : active_blocks(scopes, phase))
            {
                const Index dimension = scopes[
                    static_cast<std::size_t>(block)].dimension;
                for (Index column = 0; column < dimension; ++column)
                {
                    const Index full_column = trajectory_dimension
                        + scoped_offsets[static_cast<std::size_t>(block)]
                        + column;
                    for (Index row = 0; row < y_size; ++row)
                    {
                        const Scalar value = current.dense_b(
                            row, incident_offset + column);
                        dense(y_offset + row, full_column) = value;
                        dense(full_column, y_offset + row) = value;
                    }
                }
                incident_offset += dimension;
            }
        }

        for (std::size_t block = 0; block < scopes.size(); ++block)
        {
            const Index dimension = scopes[block].dimension;
            const Index offset = trajectory_dimension + scoped_offsets[block];
            for (Index row = 0; row < dimension; ++row)
            {
                rhs(offset + row) = -rhs_scoped[block](row);
                for (Index column = 0; column < dimension; ++column)
                    dense(offset + row, offset + column) =
                        diagonal[block](row, column);
            }
        }
        const std::vector<IntervalKktEdge> &edges =
            solver.reduced_solver().edges();
        for (std::size_t edge_index = 0;
             edge_index < edges.size(); ++edge_index)
        {
            const IntervalKktEdge edge = edges[edge_index];
            const Index rows = scopes[static_cast<std::size_t>(
                edge.first_block)].dimension;
            const Index columns = scopes[static_cast<std::size_t>(
                edge.second_block)].dimension;
            const Index row_offset = trajectory_dimension
                + scoped_offsets[static_cast<std::size_t>(edge.first_block)];
            const Index column_offset = trajectory_dimension
                + scoped_offsets[static_cast<std::size_t>(edge.second_block)];
            for (Index row = 0; row < rows; ++row)
            {
                for (Index column = 0; column < columns; ++column)
                {
                    const Scalar value =
                        edge_storage[edge_index](row, column);
                    dense(row_offset + row, column_offset + column) = value;
                    dense(column_offset + column, row_offset + row) = value;
                }
            }
        }

        const Eigen::FullPivLU<Eigen::MatrixXd> dense_factor(dense);
        ASSERT_EQ(dense_factor.rank(), total);
        const Eigen::VectorXd reference = dense_factor.solve(rhs);
        ASSERT_LT(
            (dense * reference - rhs).lpNorm<Eigen::Infinity>(),
            4e-11);

        std::vector<ImplicitOcpPhaseCondensingView> phase_views;
        phase_views.reserve(static_cast<std::size_t>(phase_count));
        for (auto &phase : phases)
        {
            phase_views.emplace_back(
                phase->info, phase->jacobian, phase->hessian,
                phase->D_x.block(phase->D_x.m(), 0),
                phase->D_eq.block(phase->D_eq.m(), 0),
                phase->D_s.block(phase->D_s.m(), 0),
                phase->cross_hessian.block(
                    phase->cross_hessian.m(),
                    phase->cross_hessian.n(), 0, 0),
                phase->border_jacobian.block(
                    phase->border_jacobian.m(),
                    phase->border_jacobian.n(), 0, 0),
                phase->rhs_primal.block(phase->rhs_primal.m(), 0),
                phase->rhs_constraints.block(
                    phase->rhs_constraints.m(), 0),
                phase->primal.block(phase->primal.m(), 0),
                phase->multipliers.block(phase->multipliers.m(), 0));
        }

        std::vector<MatRealView> diagonal_views;
        std::vector<VecRealView> rhs_views;
        std::vector<VecRealView> solution_views;
        diagonal_views.reserve(scopes.size());
        rhs_views.reserve(scopes.size());
        solution_views.reserve(scopes.size());
        for (std::size_t block = 0; block < scopes.size(); ++block)
        {
            const Index dimension = scopes[block].dimension;
            diagonal_views.push_back(diagonal[block].block(
                dimension, dimension, 0, 0));
            rhs_views.push_back(rhs_scoped[block].block(dimension, 0));
            solution_views.push_back(
                solution_scoped[block].block(dimension, 0));
        }
        std::vector<MatRealView> edge_views;
        edge_views.reserve(edges.size());
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
        EXPECT_EQ(
            solver.last_failure_location(),
            IntervalScopedOcpKktFailureLocation::None);
        if (rank_deficient)
        {
            for (const auto &phase : phases)
                for (const Index rank : phase->jacobian.J_ranks)
                    EXPECT_EQ(rank, phase->nx - 1);
        }

        Eigen::VectorXd computed(total);
        for (Index phase = 0; phase < phase_count; ++phase)
        {
            const ImplicitPhaseData &current =
                *phases[static_cast<std::size_t>(phase)];
            const Index offset = trajectory_offsets[
                static_cast<std::size_t>(phase)];
            for (Index row = 0;
                 row < current.info.number_of_primal_variables; ++row)
                computed(offset + row) = current.primal(row);
            for (Index row = 0;
                 row < current.info.number_of_eq_constraints; ++row)
            {
                computed(offset + current.info.number_of_primal_variables + row)
                    = current.multipliers(row);
            }
        }
        for (std::size_t block = 0; block < scopes.size(); ++block)
        {
            for (Index row = 0; row < scopes[block].dimension; ++row)
            {
                computed(
                    trajectory_dimension + scoped_offsets[block] + row) =
                    solution_scoped[block](row);
            }
        }
        EXPECT_LT(
            (computed - reference).lpNorm<Eigen::Infinity>(),
            3e-8);
        EXPECT_LT(
            (dense * computed - rhs).lpNorm<Eigen::Infinity>(),
            3e-8);
    }
} // namespace

TEST(IntervalScopedImplicitOcpKktSolverTest,
     MatchesCompleteDenseKktAcrossPhasesAndDimensions)
{
    run_case({1}, {0}, 0);
    run_case({2, 3}, {1, 2}, 2);
    run_case({1, 3, 2, 4}, {0, 2, 1, 3}, 2);
    run_case(
        {2, 3, 1, 4, 2, 3, 2, 4},
        {1, 2, 0, 3, 1, 2, 1, 3},
        3);
    run_case({2, 2, 2, 2}, {1, 1, 1, 1}, 2, true);
}
