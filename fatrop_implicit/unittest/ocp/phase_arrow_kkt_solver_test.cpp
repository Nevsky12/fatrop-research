//
// Copyright (c) 2026
//

#include "fatrop/ocp/phase_arrow_kkt_solver.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace fatrop;

namespace
{
    struct Case
    {
        std::vector<Index> phase_sizes;
        Index global_size = 0;
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

    void run_dense_reference_case(const Case &config)
    {
        ASSERT_FALSE(config.phase_sizes.empty());
        Index phase_dimension = 0;
        for (const Index size : config.phase_sizes)
            phase_dimension += size;
        const Index total = phase_dimension + config.global_size;

        std::vector<MatRealAllocated> diagonal;
        std::vector<MatRealAllocated> coupling;
        std::vector<MatRealAllocated> arrow;
        std::vector<VecRealAllocated> rhs_phase;
        std::vector<VecRealAllocated> solution_phase;
        diagonal.reserve(config.phase_sizes.size());
        coupling.reserve(config.phase_sizes.size() - 1);
        arrow.reserve(config.phase_sizes.size());
        rhs_phase.reserve(config.phase_sizes.size());
        solution_phase.reserve(config.phase_sizes.size());

        Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(total, total);
        Eigen::VectorXd rhs(total);
        std::vector<Index> offsets(config.phase_sizes.size(), 0);
        for (std::size_t phase = 1;
             phase < config.phase_sizes.size(); ++phase)
        {
            offsets[phase] = offsets[phase - 1]
                           + config.phase_sizes[phase - 1];
        }

        for (std::size_t phase = 0;
             phase < config.phase_sizes.size(); ++phase)
        {
            const Index size = config.phase_sizes[phase];
            diagonal.emplace_back(size, size);
            arrow.emplace_back(size, config.global_size);
            rhs_phase.emplace_back(size);
            solution_phase.emplace_back(size);
            diagonal.back() = 0.0;
            arrow.back() = 0.0;
            solution_phase.back() = 0.0;

            for (Index row = 0; row < size; ++row)
            {
                for (Index column = 0; column < size; ++column)
                {
                    Scalar value = 0.0;
                    if (row == column)
                    {
                        // Alternate primal-like positive and
                        // linkage-dual-like negative directions.
                        value = (row % 3 == 2 ? -1.0 : 1.0)
                              * (3.5 + 0.2 * phase + 0.1 * row);
                    }
                    else
                    {
                        value = 0.025
                              * (1 + ((row + column + phase) % 4));
                    }
                    diagonal.back()(row, column) = value;
                    dense(offsets[phase] + row,
                          offsets[phase] + column) = value;
                }
                rhs_phase.back()(row) =
                    0.17 * (1 + offsets[phase] + row)
                  - 0.09 * ((row + phase) % 3);
                rhs(offsets[phase] + row) = rhs_phase.back()(row);
                for (Index global = 0;
                     global < config.global_size; ++global)
                {
                    const Scalar value =
                        0.018 * (1 + ((2 * row + global + phase) % 5))
                      - 0.035;
                    arrow.back()(row, global) = value;
                    dense(offsets[phase] + row,
                          phase_dimension + global) = value;
                    dense(phase_dimension + global,
                          offsets[phase] + row) = value;
                }
            }
        }

        for (std::size_t phase = 0;
             phase + 1 < config.phase_sizes.size(); ++phase)
        {
            const Index rows = config.phase_sizes[phase];
            const Index columns = config.phase_sizes[phase + 1];
            coupling.emplace_back(rows, columns);
            for (Index row = 0; row < rows; ++row)
            {
                for (Index column = 0; column < columns; ++column)
                {
                    const Scalar value =
                        0.012 * (1 + ((row + 2 * column + phase) % 6))
                      - 0.025;
                    coupling.back()(row, column) = value;
                    dense(offsets[phase] + row,
                          offsets[phase + 1] + column) = value;
                    dense(offsets[phase + 1] + column,
                          offsets[phase] + row) = value;
                }
            }
        }

        const Index global_storage_size =
            std::max<Index>(config.global_size, 1);
        MatRealAllocated global_storage(
            global_storage_size, global_storage_size);
        VecRealAllocated rhs_global_storage(global_storage_size);
        VecRealAllocated solution_global_storage(global_storage_size);
        global_storage = 0.0;
        rhs_global_storage = 0.0;
        solution_global_storage = 0.0;
        for (Index row = 0; row < config.global_size; ++row)
        {
            for (Index column = 0;
                 column < config.global_size; ++column)
            {
                const Scalar value =
                    row == column
                    ? (row % 2 == 0 ? 4.2 + 0.3 * row
                                    : -3.8 - 0.2 * row)
                    : 0.04 / (1.0 + std::abs(row - column));
                global_storage(row, column) = value;
                dense(phase_dimension + row,
                      phase_dimension + column) = value;
            }
            rhs_global_storage(row) = 0.21 * (row + 1) - 0.08;
            rhs(phase_dimension + row) = rhs_global_storage(row);
        }

        const Eigen::FullPivLU<Eigen::MatrixXd> reference_factor(dense);
        ASSERT_EQ(reference_factor.rank(), total);
        const Eigen::VectorXd reference = reference_factor.solve(rhs);
        ASSERT_LT(
            (dense * reference - rhs).lpNorm<Eigen::Infinity>(),
            2e-12);

        std::vector<MatRealView> diagonal_view = matrix_views(diagonal);
        std::vector<MatRealView> coupling_view = matrix_views(coupling);
        std::vector<MatRealView> arrow_view = matrix_views(arrow);
        std::vector<VecRealView> rhs_view = vector_views(rhs_phase);
        std::vector<VecRealView> solution_view = vector_views(solution_phase);
        MatRealView global_view = global_storage.block(
            config.global_size, config.global_size, 0, 0);
        VecRealView rhs_global_view = rhs_global_storage.block(
            config.global_size, 0);
        VecRealView solution_global_view = solution_global_storage.block(
            config.global_size, 0);

        PhaseArrowKktSolver solver(
            config.phase_sizes, config.global_size);
        solver.set_pivot_tolerance(1e-14);
        ASSERT_EQ(
            solver.solve(
                diagonal_view, coupling_view, arrow_view,
                global_view, rhs_view, rhs_global_view,
                solution_view, solution_global_view),
            LinsolReturnFlag::SUCCESS);
        EXPECT_EQ(
            solver.last_failure_location(),
            PhaseArrowKktFailureLocation::None);

        Eigen::VectorXd computed(total);
        for (std::size_t phase = 0;
             phase < config.phase_sizes.size(); ++phase)
        {
            for (Index row = 0; row < config.phase_sizes[phase]; ++row)
                computed(offsets[phase] + row) = solution_phase[phase](row);
        }
        for (Index row = 0; row < config.global_size; ++row)
            computed(phase_dimension + row) = solution_global_storage(row);

        EXPECT_LT(
            (computed - reference).lpNorm<Eigen::Infinity>(),
            3e-11);
        EXPECT_LT(
            (dense * computed - rhs).lpNorm<Eigen::Infinity>(),
            3e-11);
    }
}

TEST(PhaseArrowKktSolverTest,
     MatchesDenseReferenceAcrossPhaseAndGlobalDimensions)
{
    const std::vector<Case> cases{
        {{4}, 0},
        {{3, 5}, 2},
        {{2, 5, 3, 6}, 1},
        {{3, 4, 6, 2, 5, 3, 7, 4}, 3}
    };
    for (const Case &config : cases)
    {
        SCOPED_TRACE(
            "phases=" + std::to_string(config.phase_sizes.size())
          + ", global=" + std::to_string(config.global_size));
        run_dense_reference_case(config);
    }
}

TEST(PhaseArrowKktSolverTest, ReportsSingularPhaseBlock)
{
    MatRealAllocated diagonal_storage(2, 2);
    MatRealAllocated arrow_storage(2, 1);
    MatRealAllocated global_storage(1, 1);
    VecRealAllocated rhs_phase_storage(2);
    VecRealAllocated rhs_global_storage(1);
    VecRealAllocated solution_phase_storage(2);
    VecRealAllocated solution_global_storage(1);
    diagonal_storage = 0.0;
    arrow_storage = 0.0;
    global_storage = 1.0;
    rhs_phase_storage = 1.0;
    rhs_global_storage = 0.0;

    PhaseArrowKktSolver solver({2}, 1);
    EXPECT_EQ(
        solver.solve(
            {diagonal_storage.block(2, 2, 0, 0)}, {},
            {arrow_storage.block(2, 1, 0, 0)},
            global_storage.block(1, 1, 0, 0),
            {rhs_phase_storage.block(2, 0)},
            rhs_global_storage.block(1, 0),
            {solution_phase_storage.block(2, 0)},
            solution_global_storage.block(1, 0)),
        LinsolReturnFlag::NOFULL_RANK);
    EXPECT_EQ(
        solver.last_failure_location(),
        PhaseArrowKktFailureLocation::PhaseBlockFactorization);
    EXPECT_EQ(solver.last_failure_phase(), 0);
}

TEST(PhaseArrowKktSolverTest, ReportsSingularGlobalSchurComplement)
{
    MatRealAllocated diagonal_storage(2, 2);
    MatRealAllocated arrow_storage(2, 1);
    MatRealAllocated global_storage(1, 1);
    VecRealAllocated rhs_phase_storage(2);
    VecRealAllocated rhs_global_storage(1);
    VecRealAllocated solution_phase_storage(2);
    VecRealAllocated solution_global_storage(1);
    diagonal_storage = 0.0;
    diagonal_storage(0, 0) = 1.0;
    diagonal_storage(1, 1) = 2.0;
    arrow_storage = 0.0;
    global_storage = 0.0;
    rhs_phase_storage = 1.0;
    rhs_global_storage = 1.0;

    PhaseArrowKktSolver solver({2}, 1);
    EXPECT_EQ(
        solver.solve(
            {diagonal_storage.block(2, 2, 0, 0)}, {},
            {arrow_storage.block(2, 1, 0, 0)},
            global_storage.block(1, 1, 0, 0),
            {rhs_phase_storage.block(2, 0)},
            rhs_global_storage.block(1, 0),
            {solution_phase_storage.block(2, 0)},
            solution_global_storage.block(1, 0)),
        LinsolReturnFlag::NOFULL_RANK);
    EXPECT_EQ(
        solver.last_failure_location(),
        PhaseArrowKktFailureLocation::GlobalFactorization);
    EXPECT_EQ(solver.last_failure_phase(), -1);
}
