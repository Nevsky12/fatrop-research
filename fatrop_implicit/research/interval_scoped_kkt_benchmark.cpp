#include "fatrop/ocp/interval_scoped_kkt_solver.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using fatrop::Index;
using fatrop::IntervalKktEdge;
using fatrop::IntervalScope;
using fatrop::IntervalScopedKktSolver;
using fatrop::LinsolReturnFlag;
using fatrop::MatRealAllocated;
using fatrop::MatRealView;
using fatrop::Scalar;
using fatrop::VecRealAllocated;
using fatrop::VecRealView;

namespace
{
    struct Config
    {
        Index phases = 64;
        Index phase_dimension = 2;
        Index interface_dimension = 1;
        Index segment_dimension = 2;
        Index segment_width = 4;
        Index segment_stride = 4;
        Index global_dimension = 2;
        Index repeats = 9;
        Index dense_limit = 1200;
    };

    Config parse_arguments(int argc, char **argv)
    {
        Config config;
        for (int argument = 1; argument < argc; ++argument)
        {
            const std::string option(argv[argument]);
            const auto next = [&](const std::string &name)
            {
                if (argument + 1 >= argc)
                    throw std::runtime_error(name + " requires a value");
                return static_cast<Index>(std::stol(argv[++argument]));
            };
            if (option == "--phases")
                config.phases = next(option);
            else if (option == "--phase-dimension")
                config.phase_dimension = next(option);
            else if (option == "--interface-dimension")
                config.interface_dimension = next(option);
            else if (option == "--segment-dimension")
                config.segment_dimension = next(option);
            else if (option == "--segment-width")
                config.segment_width = next(option);
            else if (option == "--segment-stride")
                config.segment_stride = next(option);
            else if (option == "--global-dimension")
                config.global_dimension = next(option);
            else if (option == "--repeats")
                config.repeats = next(option);
            else if (option == "--dense-limit")
                config.dense_limit = next(option);
            else if (option == "--help")
            {
                std::cout
                    << "Usage: interval_scoped_kkt_benchmark [options]\n"
                    << "  --phases N\n"
                    << "  --phase-dimension N\n"
                    << "  --interface-dimension N\n"
                    << "  --segment-dimension N\n"
                    << "  --segment-width N\n"
                    << "  --segment-stride N\n"
                    << "  --global-dimension N\n"
                    << "  --repeats N\n"
                    << "  --dense-limit N\n";
                std::exit(0);
            }
            else
                throw std::runtime_error("Unknown option: " + option);
        }
        if (config.phases < 1 || config.phase_dimension < 0
            || config.interface_dimension < 0
            || config.segment_dimension < 0
            || config.segment_width < 1 || config.segment_stride < 1
            || config.global_dimension < 0 || config.repeats < 1
            || config.dense_limit < 0)
            throw std::runtime_error("Invalid benchmark configuration");
        return config;
    }

    std::vector<IntervalScope> make_scopes(const Config &config)
    {
        std::vector<IntervalScope> scopes;
        if (config.phase_dimension > 0)
        {
            for (Index phase = 0; phase < config.phases; ++phase)
                scopes.push_back(
                    {config.phase_dimension, phase, phase});
        }
        if (config.interface_dimension > 0)
        {
            for (Index phase = 0; phase + 1 < config.phases; ++phase)
                scopes.push_back(
                    {config.interface_dimension, phase, phase + 1});
        }
        if (config.segment_dimension > 0)
        {
            for (Index first = 0; first < config.phases;
                 first += config.segment_stride)
            {
                const Index last = std::min(
                    config.phases - 1,
                    first + config.segment_width - 1);
                scopes.push_back(
                    {config.segment_dimension, first, last});
            }
        }
        if (config.global_dimension > 0)
            scopes.push_back(
                {config.global_dimension, 0, config.phases - 1});
        if (scopes.empty())
            throw std::runtime_error("The benchmark has no scoped variables");
        return scopes;
    }

    double median(std::vector<double> values)
    {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }

    template <typename Function>
    std::pair<double, bool> time_repeated(
        const Index repeats,
        Function &&function)
    {
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(repeats));
        bool success = true;
        for (Index repeat = 0; repeat < repeats; ++repeat)
        {
            const auto start = std::chrono::steady_clock::now();
            success = function() && success;
            samples.push_back(
                std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - start).count());
        }
        return {median(std::move(samples)), success};
    }
} // namespace

int main(int argc, char **argv)
try
{
    const Config config = parse_arguments(argc, argv);
    const std::vector<IntervalScope> scopes = make_scopes(config);
    std::vector<Index> offsets(scopes.size(), 0);
    Index total_dimension = 0;
    for (std::size_t block = 0; block < scopes.size(); ++block)
    {
        offsets[block] = total_dimension;
        total_dimension += scopes[block].dimension;
    }

    IntervalScopedKktSolver solver(config.phases, scopes);
    solver.set_pivot_tolerance(1e-14);
    std::vector<MatRealAllocated> diagonal;
    std::vector<VecRealAllocated> rhs;
    std::vector<VecRealAllocated> solution;
    diagonal.reserve(scopes.size());
    rhs.reserve(scopes.size());
    solution.reserve(scopes.size());

    std::vector<Scalar> row_sum(
        static_cast<std::size_t>(total_dimension), 0.0);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(
        static_cast<std::size_t>(
            4 * total_dimension
            + 16 * solver.symbolic_stats().number_of_edges));

    std::vector<MatRealAllocated> edge_storage;
    edge_storage.reserve(solver.edges().size());
    for (const IntervalKktEdge edge : solver.edges())
    {
        const Index rows = scopes[static_cast<std::size_t>(
            edge.first_block)].dimension;
        const Index columns = scopes[static_cast<std::size_t>(
            edge.second_block)].dimension;
        edge_storage.emplace_back(rows, columns);
        for (Index row = 0; row < rows; ++row)
        for (Index column = 0; column < columns; ++column)
        {
            const Scalar value =
                0.006 * (1 + ((3 * row + 2 * column
                    + edge.first_block + edge.second_block) % 7))
              - 0.021;
            edge_storage.back()(row, column) = value;
            const Index global_row = offsets[static_cast<std::size_t>(
                edge.first_block)] + row;
            const Index global_column = offsets[static_cast<std::size_t>(
                edge.second_block)] + column;
            row_sum[static_cast<std::size_t>(global_row)] += std::abs(value);
            row_sum[static_cast<std::size_t>(global_column)] += std::abs(value);
            triplets.emplace_back(global_row, global_column, value);
            triplets.emplace_back(global_column, global_row, value);
        }
    }

    Eigen::VectorXd eigen_rhs(total_dimension);
    for (std::size_t block = 0; block < scopes.size(); ++block)
    {
        const Index dimension = scopes[block].dimension;
        diagonal.emplace_back(dimension, dimension);
        rhs.emplace_back(dimension);
        solution.emplace_back(dimension);
        diagonal.back() = 0.0;
        solution.back() = 0.0;
        for (Index row = 0; row < dimension; ++row)
        {
            for (Index column = row + 1; column < dimension; ++column)
            {
                const Scalar value =
                    0.008 * (1 + ((row + column + block) % 4));
                diagonal.back()(row, column) = value;
                diagonal.back()(column, row) = value;
                row_sum[static_cast<std::size_t>(offsets[block] + row)]
                    += std::abs(value);
                row_sum[static_cast<std::size_t>(offsets[block] + column)]
                    += std::abs(value);
                triplets.emplace_back(
                    offsets[block] + row,
                    offsets[block] + column, value);
                triplets.emplace_back(
                    offsets[block] + column,
                    offsets[block] + row, value);
            }
        }
        for (Index row = 0; row < dimension; ++row)
        {
            const Index global_row = offsets[block] + row;
            const Scalar sign = (block + static_cast<std::size_t>(row)) % 4 == 3
                ? -1.0 : 1.0;
            const Scalar value = sign
                * (2.0 + row_sum[static_cast<std::size_t>(global_row)]);
            diagonal.back()(row, row) = value;
            triplets.emplace_back(global_row, global_row, value);
            rhs.back()(row) =
                0.03 * (1 + global_row) - 0.07 * ((block + row) % 3);
            eigen_rhs(global_row) = rhs.back()(row);
        }
        solver.add_matrix_block(
            static_cast<Index>(block),
            static_cast<Index>(block),
            diagonal.back().block(dimension, dimension, 0, 0));
        solver.add_rhs_block(
            static_cast<Index>(block),
            rhs.back().block(dimension, 0));
    }
    for (std::size_t edge = 0;
         edge < solver.edges().size(); ++edge)
    {
        const IntervalKktEdge endpoints = solver.edges()[edge];
        solver.add_matrix_block(
            endpoints.first_block,
            endpoints.second_block,
            edge_storage[edge].block(
                scopes[static_cast<std::size_t>(
                    endpoints.first_block)].dimension,
                scopes[static_cast<std::size_t>(
                    endpoints.second_block)].dimension,
                0, 0));
    }

    std::vector<VecRealView> solution_views;
    solution_views.reserve(solution.size());
    for (std::size_t block = 0; block < solution.size(); ++block)
        solution_views.push_back(solution[block].block(scopes[block].dimension, 0));

    Eigen::SparseMatrix<double> sparse(total_dimension, total_dimension);
    sparse.setFromTriplets(triplets.begin(), triplets.end());
    sparse.makeCompressed();
    Eigen::SparseLU<Eigen::SparseMatrix<double>> sparse_lu;
    sparse_lu.analyzePattern(sparse);
    Eigen::VectorXd sparse_solution(total_dimension);

    solver.factor_and_solve(solution_views);
    sparse_lu.factorize(sparse);
    sparse_solution = sparse_lu.solve(eigen_rhs);

    const auto interval_timing = time_repeated(
        config.repeats,
        [&]()
        {
            return solver.factor_and_solve(solution_views)
                == LinsolReturnFlag::SUCCESS;
        });
    const auto sparse_timing = time_repeated(
        config.repeats,
        [&]()
        {
            sparse_lu.factorize(sparse);
            if (sparse_lu.info() != Eigen::Success)
                return false;
            sparse_solution = sparse_lu.solve(eigen_rhs);
            return sparse_lu.info() == Eigen::Success;
        });

    double interval_residual = 0.0;
    Eigen::VectorXd interval_solution(total_dimension);
    for (std::size_t block = 0; block < scopes.size(); ++block)
    for (Index row = 0; row < scopes[block].dimension; ++row)
        interval_solution(offsets[block] + row) = solution[block](row);
    interval_residual =
        (sparse * interval_solution - eigen_rhs).lpNorm<Eigen::Infinity>();
    const double sparse_residual =
        (sparse * sparse_solution - eigen_rhs).lpNorm<Eigen::Infinity>();

    double dense_microseconds =
        std::numeric_limits<double>::quiet_NaN();
    double dense_residual =
        std::numeric_limits<double>::quiet_NaN();
    bool dense_success = true;
    if (total_dimension <= config.dense_limit)
    {
        const Eigen::MatrixXd dense = Eigen::MatrixXd(sparse);
        Eigen::PartialPivLU<Eigen::MatrixXd> dense_lu;
        Eigen::VectorXd dense_solution(total_dimension);
        const auto dense_timing = time_repeated(
            config.repeats,
            [&]()
            {
                dense_lu.compute(dense);
                dense_solution = dense_lu.solve(eigen_rhs);
                return dense_solution.allFinite();
            });
        dense_microseconds = dense_timing.first;
        dense_success = dense_timing.second;
        dense_residual =
            (dense * dense_solution - eigen_rhs)
                .lpNorm<Eigen::Infinity>();
    }

    const auto stats = solver.symbolic_stats();
    std::cout
        << "phases,blocks,total_dimension,omega,edges,scalar_nnz,"
           "interval_us,eigen_sparse_lu_us,dense_lu_us,"
           "interval_residual,sparse_residual,dense_residual,success\n";
    std::cout
        << config.phases << ','
        << scopes.size() << ','
        << total_dimension << ','
        << stats.maximum_active_dimension << ','
        << stats.number_of_edges << ','
        << sparse.nonZeros() << ','
        << std::setprecision(12)
        << interval_timing.first << ','
        << sparse_timing.first << ','
        << dense_microseconds << ','
        << interval_residual << ','
        << sparse_residual << ','
        << dense_residual << ','
        << (interval_timing.second && sparse_timing.second && dense_success
            ? 1 : 0)
        << '\n';
    return 0;
}
catch (const std::exception &error)
{
    std::cerr << "interval_scoped_kkt_benchmark: "
              << error.what() << '\n';
    return 1;
}
