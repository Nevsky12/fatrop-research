//
// Copyright (c) 2026
//

#include "fatrop/ocp/interval_scoped_kkt_solver.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace fatrop;

namespace
{
    bool overlap(const IntervalScope &first, const IntervalScope &second)
    {
        return std::max(first.first_phase, second.first_phase)
            <= std::min(first.last_phase, second.last_phase);
    }

    std::vector<Index> offsets(const std::vector<IntervalScope> &scopes)
    {
        std::vector<Index> result(scopes.size(), 0);
        for (std::size_t block = 1; block < scopes.size(); ++block)
        {
            result[block] = result[block - 1]
                          + scopes[block - 1].dimension;
        }
        return result;
    }

    Index total_dimension(const std::vector<IntervalScope> &scopes)
    {
        Index result = 0;
        for (const IntervalScope &scope : scopes)
            result += scope.dimension;
        return result;
    }

    std::vector<VecRealView> views(
        std::vector<VecRealAllocated> &storage,
        const std::vector<IntervalScope> &scopes)
    {
        std::vector<VecRealView> result;
        result.reserve(storage.size());
        for (std::size_t block = 0; block < storage.size(); ++block)
        {
            result.push_back(storage[block].block(
                scopes[block].dimension, 0));
        }
        return result;
    }

    std::vector<IntervalScope> make_scopes(
        const Index phases,
        const Index global_dimension)
    {
        std::vector<IntervalScope> scopes;
        for (Index phase = 0; phase < phases; ++phase)
        {
            scopes.push_back({1 + (phase % 3), phase, phase});
        }
        for (Index phase = 0; phase + 1 < phases; ++phase)
        {
            scopes.push_back({1 + (phase % 2), phase, phase + 1});
        }
        for (Index first = 0; first < phases; first += 2)
        {
            scopes.push_back({1, first, std::min(phases - 1, first + 2)});
        }
        if (global_dimension > 0)
            scopes.push_back({global_dimension, 0, phases - 1});
        return scopes;
    }

    void run_direct_dense_case(
        const Index phases,
        const Index global_dimension)
    {
        const std::vector<IntervalScope> scopes =
            make_scopes(phases, global_dimension);
        const std::vector<Index> block_offsets = offsets(scopes);
        const Index total = total_dimension(scopes);

        IntervalScopedKktSolver solver(phases, scopes);
        solver.set_pivot_tolerance(1e-14);

        Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(total, total);
        Eigen::VectorXd dense_rhs(total);
        std::vector<MatRealAllocated> diagonal_storage;
        std::vector<VecRealAllocated> rhs_storage;
        std::vector<VecRealAllocated> solution_storage;
        diagonal_storage.reserve(scopes.size());
        rhs_storage.reserve(scopes.size());
        solution_storage.reserve(scopes.size());

        for (std::size_t block = 0; block < scopes.size(); ++block)
        {
            const Index dimension = scopes[block].dimension;
            const Index allocation = std::max<Index>(dimension, 1);
            diagonal_storage.emplace_back(allocation, allocation);
            rhs_storage.emplace_back(allocation);
            solution_storage.emplace_back(allocation);
            diagonal_storage.back() = 0.0;
            rhs_storage.back() = 0.0;
            solution_storage.back() = 0.0;

            for (Index row = 0; row < dimension; ++row)
            {
                for (Index column = 0; column < dimension; ++column)
                {
                    const Scalar value = row == column
                        ? ((block + static_cast<std::size_t>(row)) % 2 == 0
                            ? 20.0 + 0.4 * block + 0.2 * row
                            : -19.0 - 0.3 * block - 0.2 * row)
                        : 0.015 * (1 + ((row + column + block) % 4));
                    diagonal_storage.back()(row, column) = value;
                    dense(block_offsets[block] + row,
                          block_offsets[block] + column) = value;
                }
                const Scalar value =
                    0.13 * (1 + block_offsets[block] + row)
                  - 0.07 * ((block + row) % 3);
                rhs_storage.back()(row) = value;
                dense_rhs(block_offsets[block] + row) = value;
            }
            solver.add_matrix_block(
                static_cast<Index>(block),
                static_cast<Index>(block),
                diagonal_storage.back().block(
                    dimension, dimension, 0, 0));
            solver.add_rhs_block(
                static_cast<Index>(block),
                rhs_storage.back().block(dimension, 0));
        }

        std::vector<MatRealAllocated> edge_storage;
        edge_storage.reserve(solver.edges().size());
        for (const IntervalKktEdge edge : solver.edges())
        {
            const Index rows = scopes[
                static_cast<std::size_t>(edge.first_block)].dimension;
            const Index columns = scopes[
                static_cast<std::size_t>(edge.second_block)].dimension;
            edge_storage.emplace_back(
                std::max<Index>(rows, 1),
                std::max<Index>(columns, 1));
            edge_storage.back() = 0.0;
            for (Index row = 0; row < rows; ++row)
            {
                for (Index column = 0; column < columns; ++column)
                {
                    const Scalar value =
                        0.009 * (1 + ((3 * row + 2 * column
                            + edge.first_block + edge.second_block) % 7))
                      - 0.027;
                    edge_storage.back()(row, column) = value;
                    dense(
                        block_offsets[static_cast<std::size_t>(
                            edge.first_block)] + row,
                        block_offsets[static_cast<std::size_t>(
                            edge.second_block)] + column) = value;
                    dense(
                        block_offsets[static_cast<std::size_t>(
                            edge.second_block)] + column,
                        block_offsets[static_cast<std::size_t>(
                            edge.first_block)] + row) = value;
                }
            }
            solver.add_matrix_block(
                edge.first_block,
                edge.second_block,
                edge_storage.back().block(rows, columns, 0, 0));
        }

        const Eigen::FullPivLU<Eigen::MatrixXd> reference_factor(dense);
        ASSERT_EQ(reference_factor.rank(), total);
        const Eigen::VectorXd reference =
            reference_factor.solve(dense_rhs);
        ASSERT_LT(
            (dense * reference - dense_rhs).lpNorm<Eigen::Infinity>(),
            3e-12);

        std::vector<VecRealView> solution_views =
            views(solution_storage, scopes);
        ASSERT_EQ(
            solver.factor_and_solve(solution_views),
            LinsolReturnFlag::SUCCESS);

        Eigen::VectorXd computed(total);
        for (std::size_t block = 0; block < scopes.size(); ++block)
        {
            for (Index row = 0; row < scopes[block].dimension; ++row)
            {
                computed(block_offsets[block] + row) =
                    solution_storage[block](row);
            }
        }
        EXPECT_LT(
            (computed - reference).lpNorm<Eigen::Infinity>(),
            4e-11);
        EXPECT_LT(
            (dense * computed - dense_rhs).lpNorm<Eigen::Infinity>(),
            4e-11);
    }
} // namespace

TEST(IntervalScopedKktSolverTest,
     SymbolicOrderingHasNoFillAndFrontIsBoundedByActiveWidth)
{
    const std::vector<IntervalScope> scopes{
        {2, 0, 0},
        {1, 0, 1},
        {3, 1, 3},
        {2, 2, 2},
        {1, 3, 5},
        {2, 0, 5},
        {0, 5, 5}
    };
    IntervalScopedKktSolver solver(6, scopes);

    EXPECT_EQ(
        solver.elimination_order(),
        (std::vector<Index>{0, 1, 3, 2, 5, 4, 6}));
    const IntervalKktSymbolicStats stats = solver.symbolic_stats();
    EXPECT_EQ(stats.number_of_blocks, 7);
    EXPECT_EQ(stats.maximum_active_dimension, 7);
    EXPECT_EQ(stats.maximum_front_dimension, 7);
    EXPECT_EQ(stats.symbolic_fill_edges, 0);

    Index expected_edges = 0;
    for (std::size_t first = 0; first < scopes.size(); ++first)
        for (std::size_t second = first + 1;
             second < scopes.size(); ++second)
            expected_edges += overlap(scopes[first], scopes[second]);
    EXPECT_EQ(stats.number_of_edges, expected_edges);
}

TEST(IntervalScopedKktSolverTest,
     MatchesDenseReferenceAcrossPhaseAndScopeDimensions)
{
    for (const Index phases : {Index{1}, Index{2}, Index{4}, Index{8}})
    {
        for (const Index global_dimension : {Index{0}, Index{2}})
        {
            SCOPED_TRACE(
                "phases=" + std::to_string(phases)
              + ", global=" + std::to_string(global_dimension));
            run_direct_dense_case(phases, global_dimension);
        }
    }
}

TEST(IntervalScopedKktSolverTest,
     PhaseCliqueAssemblyMatchesMonolithicDenseSystem)
{
    const Index phases = 5;
    const std::vector<IntervalScope> scopes{
        {2, 0, 0}, {1, 1, 1}, {2, 2, 2}, {1, 3, 3}, {2, 4, 4},
        {1, 0, 1}, {2, 1, 2}, {1, 2, 3}, {2, 3, 4},
        {2, 1, 3}, {3, 0, 4}
    };
    const std::vector<Index> block_offsets = offsets(scopes);
    const Index total = total_dimension(scopes);
    IntervalScopedKktSolver solver(phases, scopes);
    solver.set_pivot_tolerance(1e-14);

    Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(total, total);
    Eigen::VectorXd dense_rhs = Eigen::VectorXd::Zero(total);
    std::vector<MatRealAllocated> phase_matrices;
    std::vector<VecRealAllocated> phase_rhs;
    phase_matrices.reserve(phases);
    phase_rhs.reserve(phases);

    for (Index phase = 0; phase < phases; ++phase)
    {
        std::vector<Index> active;
        for (Index block = static_cast<Index>(scopes.size());
             block-- > 0;)
        {
            if (scopes[static_cast<std::size_t>(block)].first_phase <= phase
                && phase <= scopes[static_cast<std::size_t>(block)].last_phase)
                active.push_back(block);
        }
        Index clique_dimension = 0;
        std::vector<Index> clique_offsets(active.size(), 0);
        for (std::size_t item = 0; item < active.size(); ++item)
        {
            clique_offsets[item] = clique_dimension;
            clique_dimension += scopes[
                static_cast<std::size_t>(active[item])].dimension;
        }

        phase_matrices.emplace_back(clique_dimension, clique_dimension);
        phase_rhs.emplace_back(clique_dimension);
        phase_matrices.back() = 0.0;
        phase_rhs.back() = 0.0;
        for (std::size_t row_item = 0;
             row_item < active.size(); ++row_item)
        {
            const Index row_block = active[row_item];
            const Index row_dimension = scopes[
                static_cast<std::size_t>(row_block)].dimension;
            for (Index row = 0; row < row_dimension; ++row)
            {
                const Scalar rhs_value =
                    0.04 * (1 + phase + row_block + row);
                phase_rhs.back()(clique_offsets[row_item] + row) = rhs_value;
                dense_rhs(
                    block_offsets[static_cast<std::size_t>(row_block)] + row)
                    += rhs_value;
            }
            for (std::size_t column_item = row_item;
                 column_item < active.size(); ++column_item)
            {
                const Index column_block = active[column_item];
                const Index column_dimension = scopes[
                    static_cast<std::size_t>(column_block)].dimension;
                for (Index row = 0; row < row_dimension; ++row)
                {
                    for (Index column = 0;
                         column < column_dimension; ++column)
                    {
                        Scalar value = 0.0;
                        if (row_item == column_item && row == column)
                        {
                            value = ((row_block + row) % 2 == 0 ? 1.0 : -1.0)
                                  * (8.0 + 0.3 * phase + 0.2 * row_block);
                        }
                        else if (row_item == column_item)
                        {
                            value = 0.012
                                  * (1 + ((row + column + phase) % 3));
                        }
                        else
                        {
                            value = 0.006
                                  * (1 + ((row + 2 * column + row_block
                                      + column_block + phase) % 5))
                                  - 0.012;
                        }
                        phase_matrices.back()(
                            clique_offsets[row_item] + row,
                            clique_offsets[column_item] + column) = value;
                        phase_matrices.back()(
                            clique_offsets[column_item] + column,
                            clique_offsets[row_item] + row) = value;
                        dense(
                            block_offsets[static_cast<std::size_t>(row_block)]
                                + row,
                            block_offsets[static_cast<std::size_t>(column_block)]
                                + column) += value;
                        if (row_item != column_item)
                        {
                            dense(
                                block_offsets[static_cast<std::size_t>(
                                    column_block)] + column,
                                block_offsets[static_cast<std::size_t>(
                                    row_block)] + row) += value;
                        }
                    }
                }
            }
        }

        solver.add_phase_contribution(
            phase,
            active,
            phase_matrices.back().block(
                clique_dimension, clique_dimension, 0, 0),
            phase_rhs.back().block(clique_dimension, 0));
    }

    const Eigen::FullPivLU<Eigen::MatrixXd> reference_factor(dense);
    ASSERT_EQ(reference_factor.rank(), total);
    const Eigen::VectorXd reference = reference_factor.solve(dense_rhs);

    std::vector<VecRealAllocated> solution_storage;
    solution_storage.reserve(scopes.size());
    for (const IntervalScope &scope : scopes)
    {
        solution_storage.emplace_back(std::max<Index>(scope.dimension, 1));
        solution_storage.back() = 0.0;
    }
    std::vector<VecRealView> solution_views =
        views(solution_storage, scopes);
    ASSERT_EQ(
        solver.factor_and_solve(solution_views),
        LinsolReturnFlag::SUCCESS);

    Eigen::VectorXd computed(total);
    for (std::size_t block = 0; block < scopes.size(); ++block)
        for (Index row = 0; row < scopes[block].dimension; ++row)
            computed(block_offsets[block] + row) = solution_storage[block](row);
    EXPECT_LT(
        (computed - reference).lpNorm<Eigen::Infinity>(),
        5e-11);
    EXPECT_LT(
        (dense * computed - dense_rhs).lpNorm<Eigen::Infinity>(),
        5e-11);
}

TEST(IntervalScopedKktSolverTest, ReportsThePrescribedSingularBlock)
{
    IntervalScopedKktSolver solver(
        2, {{1, 0, 0}, {1, 0, 1}});
    MatRealAllocated first_diagonal(1, 1);
    MatRealAllocated second_diagonal(1, 1);
    first_diagonal = 0.0;
    second_diagonal = 1.0;
    solver.add_matrix_block(
        0, 0, first_diagonal.block(1, 1, 0, 0));
    solver.add_matrix_block(
        1, 1, second_diagonal.block(1, 1, 0, 0));

    EXPECT_EQ(solver.factorize(), LinsolReturnFlag::NOFULL_RANK);
    EXPECT_EQ(
        solver.last_failure_location(),
        IntervalKktFailureLocation::BlockFactorization);
    EXPECT_EQ(solver.last_failure_block(), 0);
}

TEST(IntervalScopedKktSolverTest, RejectsCouplingOfDisjointScopes)
{
    IntervalScopedKktSolver solver(
        2, {{1, 0, 0}, {1, 1, 1}});
    MatRealAllocated coupling(1, 1);
    coupling = 1.0;
    EXPECT_THROW(
        solver.add_matrix_block(
            0, 1, coupling.block(1, 1, 0, 0)),
        FatropException);
}
