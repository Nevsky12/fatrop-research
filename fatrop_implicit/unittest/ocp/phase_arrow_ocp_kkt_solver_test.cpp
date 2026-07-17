//
// Copyright (c) 2026
//

#include "fatrop/ocp/phase_arrow_ocp_kkt_solver.hpp"

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
            jacobian.Gg_eqt[0] = 0.0;
            jacobian.Gg_eqt[1] = 0.0;
            jacobian.Gg_eqt[2] = 0.0;
            for (Index stage = 0; stage < dims.K; ++stage)
            {
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
                        const Scalar value =
                            row == column
                            ? 4.0 + 0.3 * phase + 0.2 * stage
                                  + 0.1 * row
                            : 0.018
                              / (1.0 + std::abs(row - column));
                        hessian.RSQrqt[stage](row, column) = value;
                        dense_k(offset + row, offset + column) = value;
                    }
                }
                for (Index row = 0; row < variables; ++row)
                {
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
                    const Index path_row =
                        info.offsets_g_eq_path[stage];
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
                            // Force a nontrivial pivot without losing rank.
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
            {
                D_eq(row) = 0.035 + 0.004 * ((row + phase) % 5);
            }
            for (Index row = 0; row < constraint_count; ++row)
            {
                rhs_constraints(row) =
                    -0.045 * (row + 1)
                  + 0.018 * ((row + phase) % 4);
            }
            // FATROP's stabilized OCP recursion damps stage-local equality
            // duals. Dynamics multipliers remain in the Riccati chain and
            // therefore have no -D_eq diagonal in the original KKT matrix.
            for (Index stage = 0; stage < dims.K; ++stage)
            for (Index equation = 0;
                 equation < dims.number_of_eq_constraints[stage]; ++equation)
            {
                const Index row =
                    info.offsets_g_eq_path[stage] + equation;
                const Index diagonal = info.offsets_eq[stage] + equation;
                dense_k(primal_count + row, primal_count + row) =
                    -D_eq(diagonal);
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

    void run_case(
        const std::vector<Index> &nx,
        const std::vector<Index> &nu,
        const std::vector<Index> &phase_sizes,
        const Index global_size)
    {
        const std::size_t phase_count = phase_sizes.size();
        ASSERT_EQ(nx.size(), phase_count);
        ASSERT_EQ(nu.size(), phase_count);
        ASSERT_FALSE(phase_sizes.empty());

        std::vector<std::unique_ptr<ImplicitPhaseData>> phases;
        std::vector<const ProblemInfo *> phase_info;
        phases.reserve(phase_count);
        phase_info.reserve(phase_count);
        for (std::size_t phase = 0; phase < phase_count; ++phase)
        {
            const Index left = phase == 0 ? 0 : phase_sizes[phase - 1];
            const Index incident = left + phase_sizes[phase] + global_size;
            phases.push_back(std::make_unique<ImplicitPhaseData>(
                static_cast<Index>(phase), nx[phase], nu[phase], incident));
            phase_info.push_back(&phases.back()->info);
        }

        std::vector<MatRealAllocated> diagonal;
        std::vector<MatRealAllocated> coupling;
        std::vector<MatRealAllocated> arrow;
        std::vector<VecRealAllocated> rhs_phase;
        std::vector<VecRealAllocated> solution_phase;
        diagonal.reserve(phase_count);
        coupling.reserve(phase_count - 1);
        arrow.reserve(phase_count);
        rhs_phase.reserve(phase_count);
        solution_phase.reserve(phase_count);
        for (std::size_t phase = 0; phase < phase_count; ++phase)
        {
            const Index size = phase_sizes[phase];
            diagonal.emplace_back(size, size);
            arrow.emplace_back(size, global_size);
            rhs_phase.emplace_back(size);
            solution_phase.emplace_back(size);
            arrow.back() = 0.0;
            solution_phase.back() = 0.0;
            for (Index row = 0; row < size; ++row)
            {
                for (Index column = 0; column < size; ++column)
                {
                    diagonal.back()(row, column) =
                        row == column
                        ? (row + 1 == size && phase + 1 < phase_count
                            ? -2.8 - 0.1 * phase
                            : 3.6 + 0.2 * phase + 0.1 * row)
                        : 0.021 / (1.0 + std::abs(row - column));
                }
                rhs_phase.back()(row) =
                    0.14 * (row + 1) - 0.035 * phase;
                for (Index global = 0; global < global_size; ++global)
                    arrow.back()(row, global) =
                        0.013 * (1 + ((row + global + phase) % 5))
                      - 0.024;
            }
        }
        for (std::size_t phase = 0; phase + 1 < phase_count; ++phase)
        {
            coupling.emplace_back(phase_sizes[phase], phase_sizes[phase + 1]);
            for (Index row = 0; row < phase_sizes[phase]; ++row)
            for (Index column = 0; column < phase_sizes[phase + 1]; ++column)
                coupling.back()(row, column) =
                    0.008 * (1 + ((row + 2 * column + phase) % 5))
                  - 0.014;
        }

        const Index global_storage_size = std::max<Index>(global_size, 1);
        MatRealAllocated global_storage(
            global_storage_size, global_storage_size);
        VecRealAllocated rhs_global_storage(global_storage_size);
        VecRealAllocated solution_global_storage(global_storage_size);
        global_storage = 0.0;
        rhs_global_storage = 0.0;
        solution_global_storage = 0.0;
        for (Index row = 0; row < global_size; ++row)
        {
            for (Index column = 0; column < global_size; ++column)
                global_storage(row, column) =
                    row == column
                    ? 4.5 + 0.3 * row
                    : 0.025 / (1.0 + std::abs(row - column));
            rhs_global_storage(row) = 0.19 * (row + 1) - 0.05;
        }

        Index trajectory_dimension = 0;
        for (auto const &phase : phases)
            trajectory_dimension +=
                phase->info.number_of_primal_variables
              + phase->info.number_of_eq_constraints;
        Index reduced_dimension = 0;
        std::vector<Index> q_offsets(phase_count, 0);
        for (std::size_t phase = 0; phase < phase_count; ++phase)
        {
            if (phase > 0)
                q_offsets[phase] = q_offsets[phase - 1]
                                 + phase_sizes[phase - 1];
            reduced_dimension += phase_sizes[phase];
        }
        const Index total = trajectory_dimension
                          + reduced_dimension + global_size;
        Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(total, total);
        Eigen::VectorXd dense_rhs(total);
        Index trajectory_offset = 0;
        for (std::size_t phase = 0; phase < phase_count; ++phase)
        {
            ImplicitPhaseData const &current = *phases[phase];
            const Index y_size = static_cast<Index>(current.dense_k.rows());
            dense.block(
                trajectory_offset, trajectory_offset,
                y_size, y_size) = current.dense_k;
            for (Index row = 0;
                 row < current.info.number_of_primal_variables; ++row)
                dense_rhs(trajectory_offset + row) =
                    -current.rhs_primal(row);
            for (Index row = 0;
                 row < current.info.number_of_eq_constraints; ++row)
                dense_rhs(
                    trajectory_offset
                  + current.info.number_of_primal_variables + row) =
                    -current.rhs_constraints(row);

            const Index left = phase == 0 ? 0 : phase_sizes[phase - 1];
            const Index right = phase_sizes[phase];
            for (Index incident = 0;
                 incident < current.dense_b.cols(); ++incident)
            {
                Index reduced_column = 0;
                if (incident < left)
                    reduced_column = q_offsets[phase - 1] + incident;
                else if (incident < left + right)
                    reduced_column = q_offsets[phase] + incident - left;
                else
                    reduced_column = reduced_dimension
                                   + incident - left - right;
                const Index full_column =
                    trajectory_dimension + reduced_column;
                for (Index row = 0; row < y_size; ++row)
                {
                    dense(trajectory_offset + row, full_column) =
                        current.dense_b(row, incident);
                    dense(full_column, trajectory_offset + row) =
                        current.dense_b(row, incident);
                }
            }
            trajectory_offset += y_size;
        }

        const Index q_base = trajectory_dimension;
        for (std::size_t phase = 0; phase < phase_count; ++phase)
        {
            const Index size = phase_sizes[phase];
            for (Index row = 0; row < size; ++row)
            {
                dense_rhs(q_base + q_offsets[phase] + row) =
                    -rhs_phase[phase](row);
                for (Index column = 0; column < size; ++column)
                    dense(q_base + q_offsets[phase] + row,
                          q_base + q_offsets[phase] + column) =
                        diagonal[phase](row, column);
                for (Index global = 0; global < global_size; ++global)
                {
                    const Index global_column =
                        q_base + reduced_dimension + global;
                    dense(q_base + q_offsets[phase] + row,
                          global_column) = arrow[phase](row, global);
                    dense(global_column,
                          q_base + q_offsets[phase] + row) =
                        arrow[phase](row, global);
                }
            }
            if (phase + 1 < phase_count)
            {
                for (Index row = 0; row < size; ++row)
                for (Index column = 0;
                     column < phase_sizes[phase + 1]; ++column)
                {
                    const Scalar value = coupling[phase](row, column);
                    dense(q_base + q_offsets[phase] + row,
                          q_base + q_offsets[phase + 1] + column) = value;
                    dense(q_base + q_offsets[phase + 1] + column,
                          q_base + q_offsets[phase] + row) = value;
                }
            }
        }
        for (Index row = 0; row < global_size; ++row)
        {
            dense_rhs(q_base + reduced_dimension + row) =
                -rhs_global_storage(row);
            for (Index column = 0; column < global_size; ++column)
                dense(q_base + reduced_dimension + row,
                      q_base + reduced_dimension + column) =
                    global_storage(row, column);
        }

        const Eigen::FullPivLU<Eigen::MatrixXd> dense_factor(dense);
        ASSERT_EQ(dense_factor.rank(), total);
        const Eigen::VectorXd reference = dense_factor.solve(dense_rhs);
        ASSERT_LT(
            (dense * reference - dense_rhs).lpNorm<Eigen::Infinity>(),
            3e-11);

        std::vector<ImplicitPhaseArrowOcpPhaseView> phase_views;
        phase_views.reserve(phase_count);
        for (auto &phase : phases)
        {
            phase_views.emplace_back(
                phase->info, phase->jacobian, phase->hessian,
                phase->D_x.block(phase->D_x.m(), 0),
                phase->D_eq.block(phase->D_eq.m(), 0),
                phase->D_s.block(phase->D_s.m(), 0),
                phase->cross_hessian.block(
                    phase->cross_hessian.m(), phase->cross_hessian.n(), 0, 0),
                phase->border_jacobian.block(
                    phase->border_jacobian.m(), phase->border_jacobian.n(), 0, 0),
                phase->rhs_primal.block(phase->rhs_primal.m(), 0),
                phase->rhs_constraints.block(phase->rhs_constraints.m(), 0),
                phase->primal.block(phase->primal.m(), 0),
                phase->multipliers.block(phase->multipliers.m(), 0));
        }
        std::vector<MatRealView> diagonal_views = matrix_views(diagonal);
        std::vector<MatRealView> coupling_views = matrix_views(coupling);
        std::vector<MatRealView> arrow_views = matrix_views(arrow);
        std::vector<VecRealView> rhs_views = vector_views(rhs_phase);
        std::vector<VecRealView> solution_views = vector_views(solution_phase);
        MatRealView global_view = global_storage.block(
            global_size, global_size, 0, 0);
        VecRealView rhs_global_view = rhs_global_storage.block(global_size, 0);
        VecRealView solution_global_view =
            solution_global_storage.block(global_size, 0);

        ImplicitPhaseArrowOcpKktSolver solver(
            phase_info, phase_sizes, global_size);
        for (Index phase = 0;
             phase < static_cast<Index>(phase_count); ++phase)
            solver.trajectory_solver(phase).set_lu_fact_tol(1e-12);
        solver.reduced_solver().set_pivot_tolerance(1e-14);
        ASSERT_EQ(
            solver.solve(
                phase_views,
                diagonal_views, coupling_views, arrow_views,
                global_view, rhs_views, rhs_global_view,
                solution_views, solution_global_view),
            LinsolReturnFlag::SUCCESS);
        EXPECT_EQ(
            solver.last_failure_location(),
            PhaseArrowOcpKktFailureLocation::None);

        Eigen::VectorXd computed(total);
        trajectory_offset = 0;
        for (auto const &phase : phases)
        {
            for (Index row = 0;
                 row < phase->info.number_of_primal_variables; ++row)
                computed(trajectory_offset + row) = phase->primal(row);
            for (Index row = 0;
                 row < phase->info.number_of_eq_constraints; ++row)
                computed(
                    trajectory_offset
                  + phase->info.number_of_primal_variables + row) =
                    phase->multipliers(row);
            trajectory_offset +=
                phase->info.number_of_primal_variables
              + phase->info.number_of_eq_constraints;
        }
        for (std::size_t phase = 0; phase < phase_count; ++phase)
        for (Index row = 0; row < phase_sizes[phase]; ++row)
            computed(q_base + q_offsets[phase] + row) =
                solution_phase[phase](row);
        for (Index row = 0; row < global_size; ++row)
            computed(q_base + reduced_dimension + row) =
                solution_global_storage(row);

        EXPECT_LT(
            (computed - reference).lpNorm<Eigen::Infinity>(),
            2e-8);
        EXPECT_LT(
            (dense * computed - dense_rhs).lpNorm<Eigen::Infinity>(),
            2e-8);
    }
}

TEST(ImplicitPhaseArrowOcpKktSolverTest,
     MatchesCompleteDenseKktAcrossHeterogeneousPhaseDimensions)
{
    run_case({2}, {1}, {3}, 0);
    run_case({2, 3}, {1, 2}, {3, 5}, 2);
    run_case(
        {1, 3, 2, 4},
        {0, 2, 1, 3},
        {2, 5, 3, 6},
        2);
    run_case(
        {2, 3, 1, 4, 2, 3, 2, 4},
        {1, 2, 0, 3, 1, 2, 1, 3},
        {3, 5, 2, 6, 4, 5, 3, 7},
        3);
}
