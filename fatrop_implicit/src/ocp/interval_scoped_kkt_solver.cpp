//
// Copyright (c) 2026
//

#include "fatrop/ocp/interval_scoped_kkt_solver.hpp"

#include "fatrop/common/exception.hpp"
#include "fatrop/linear_algebra/dense_block.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <utility>

namespace fatrop
{
    namespace
    {
        bool intervals_overlap(
            const IntervalScope &first,
            const IntervalScope &second)
        {
            return std::max(first.first_phase, second.first_phase)
                <= std::min(first.last_phase, second.last_phase);
        }

        std::pair<Index, Index> canonical_pair(Index first, Index second)
        {
            if (second < first)
                std::swap(first, second);
            return {first, second};
        }
    } // namespace

    struct IntervalScopedKktSolver::Impl
    {
        struct EdgeStorage
        {
            IntervalKktEdge public_edge;
            Index first_block = 0;
            Index second_block = 0;
            Index first_position = 0;
            Index second_position = 0;
            detail::DenseBlock original;
            detail::DenseBlock modified;
            detail::DenseBlock response;
        };

        struct SchurUpdate
        {
            std::size_t left_edge = 0;
            std::size_t right_edge = 0;
            bool diagonal_target = false;
            Index target_block = -1;
            std::size_t target_edge = 0;
        };

        Impl(
            const Index number_of_phases_value,
            std::vector<IntervalScope> scopes_value)
            : number_of_phases(number_of_phases_value),
              block_scopes(std::move(scopes_value))
        {
            fatrop_assert_msg(
                number_of_phases >= 1,
                "An interval-scoped system needs at least one phase.");
            for (const IntervalScope &scope : block_scopes)
            {
                fatrop_assert_msg(
                    scope.dimension >= 0,
                    "An interval-scoped block dimension must be non-negative.");
                fatrop_assert_msg(
                    scope.first_phase >= 0
                    && scope.first_phase <= scope.last_phase
                    && scope.last_phase < number_of_phases,
                    "An interval-scoped block has an invalid phase interval.");
            }

            build_symbolic_factorization();
            allocate_numeric_storage();
        }

        void build_symbolic_factorization()
        {
            const Index blocks = static_cast<Index>(block_scopes.size());
            order.resize(static_cast<std::size_t>(blocks));
            std::iota(order.begin(), order.end(), Index{0});
            std::sort(
                order.begin(), order.end(),
                [this](const Index first, const Index second)
                {
                    const IntervalScope &a =
                        block_scopes[static_cast<std::size_t>(first)];
                    const IntervalScope &b =
                        block_scopes[static_cast<std::size_t>(second)];
                    if (a.last_phase != b.last_phase)
                        return a.last_phase < b.last_phase;
                    if (a.first_phase != b.first_phase)
                        return a.first_phase < b.first_phase;
                    return first < second;
                });

            position.assign(static_cast<std::size_t>(blocks), 0);
            for (Index item = 0; item < blocks; ++item)
            {
                position[static_cast<std::size_t>(
                    order[static_cast<std::size_t>(item)])] = item;
            }

            // A start-point sweep constructs exactly the interval-overlap
            // edges in O(B log B + E), apart from compact active-set erasure.
            std::vector<Index> start_order(static_cast<std::size_t>(blocks));
            std::iota(start_order.begin(), start_order.end(), Index{0});
            std::sort(
                start_order.begin(), start_order.end(),
                [this](const Index first, const Index second)
                {
                    const IntervalScope &a =
                        block_scopes[static_cast<std::size_t>(first)];
                    const IntervalScope &b =
                        block_scopes[static_cast<std::size_t>(second)];
                    if (a.first_phase != b.first_phase)
                        return a.first_phase < b.first_phase;
                    if (a.last_phase != b.last_phase)
                        return a.last_phase < b.last_phase;
                    return first < second;
                });

            std::vector<Index> active;
            std::vector<std::pair<Index, Index>> edge_pairs;
            for (const Index block : start_order)
            {
                const Index left = block_scopes[
                    static_cast<std::size_t>(block)].first_phase;
                active.erase(
                    std::remove_if(
                        active.begin(), active.end(),
                        [this, left](const Index candidate)
                        {
                            return block_scopes[
                                static_cast<std::size_t>(candidate)]
                                    .last_phase < left;
                        }),
                    active.end());
                for (const Index candidate : active)
                {
                    edge_pairs.push_back(
                        canonical_pair(candidate, block));
                }
                active.push_back(block);
            }
            std::sort(edge_pairs.begin(), edge_pairs.end());

            public_edges.reserve(edge_pairs.size());
            edge_storage.reserve(edge_pairs.size());
            edge_lookup.clear();
            for (const std::pair<Index, Index> pair : edge_pairs)
            {
                fatrop_assert_msg(
                    intervals_overlap(
                        block_scopes[static_cast<std::size_t>(pair.first)],
                        block_scopes[static_cast<std::size_t>(pair.second)]),
                    "The interval sweep generated a non-overlap edge.");

                IntervalKktEdge public_edge{pair.first, pair.second};
                public_edges.push_back(public_edge);

                EdgeStorage edge;
                edge.public_edge = public_edge;
                if (position[static_cast<std::size_t>(pair.first)]
                    < position[static_cast<std::size_t>(pair.second)])
                {
                    edge.first_block = pair.first;
                    edge.second_block = pair.second;
                }
                else
                {
                    edge.first_block = pair.second;
                    edge.second_block = pair.first;
                }
                edge.first_position = position[
                    static_cast<std::size_t>(edge.first_block)];
                edge.second_position = position[
                    static_cast<std::size_t>(edge.second_block)];
                const Index first_dimension = block_scopes[
                    static_cast<std::size_t>(edge.first_block)].dimension;
                const Index second_dimension = block_scopes[
                    static_cast<std::size_t>(edge.second_block)].dimension;
                edge.original.reset(first_dimension, second_dimension);
                edge.modified.reset(first_dimension, second_dimension);
                edge.response.reset(first_dimension, second_dimension);

                const std::size_t edge_index = edge_storage.size();
                edge_storage.push_back(std::move(edge));
                edge_lookup.emplace(pair, edge_index);
            }

            later_edges.assign(
                static_cast<std::size_t>(blocks), {});
            for (std::size_t edge_index = 0;
                 edge_index < edge_storage.size(); ++edge_index)
            {
                const EdgeStorage &edge = edge_storage[edge_index];
                later_edges[static_cast<std::size_t>(edge.first_position)]
                    .push_back(edge_index);
            }
            for (std::vector<std::size_t> &incident : later_edges)
            {
                std::sort(
                    incident.begin(), incident.end(),
                    [this](const std::size_t first,
                           const std::size_t second)
                    {
                        return edge_storage[first].second_position
                             < edge_storage[second].second_position;
                    });
            }

            // Verify the perfect-elimination invariant explicitly.  This is
            // cheap relative to numeric factorization and turns an accidental
            // scope/pattern mismatch into a construction error, not corruption.
            for (const std::vector<std::size_t> &incident : later_edges)
            {
                for (std::size_t first = 0;
                     first < incident.size(); ++first)
                {
                    for (std::size_t second = first + 1;
                         second < incident.size(); ++second)
                    {
                        const Index a =
                            edge_storage[incident[first]].second_block;
                        const Index b =
                            edge_storage[incident[second]].second_block;
                        fatrop_assert_msg(
                            edge_lookup.count(canonical_pair(a, b)) == 1,
                            "Increasing-right-endpoint order is not a perfect "
                            "elimination order for the supplied scopes.");
                    }
                }
            }

            // Resolve every numeric Schur-update destination once.  The
            // interval graph is fixed, so map lookups inside every subsequent
            // factorization are pure overhead and obscure the symbolic /
            // numeric separation.
            schur_updates.assign(
                static_cast<std::size_t>(blocks), {});
            for (Index order_position = 0;
                 order_position < blocks; ++order_position)
            {
                const std::vector<std::size_t> &incident = later_edges[
                    static_cast<std::size_t>(order_position)];
                std::vector<SchurUpdate> &updates = schur_updates[
                    static_cast<std::size_t>(order_position)];
                updates.reserve(
                    incident.size() * (incident.size() + 1) / 2);
                for (std::size_t first = 0;
                     first < incident.size(); ++first)
                {
                    for (std::size_t second = first;
                         second < incident.size(); ++second)
                    {
                        SchurUpdate update;
                        update.left_edge = incident[first];
                        update.right_edge = incident[second];
                        if (first == second)
                        {
                            update.diagonal_target = true;
                            update.target_block = edge_storage[
                                incident[first]].second_block;
                        }
                        else
                        {
                            update.target_edge = find_edge(
                                edge_storage[incident[first]].second_block,
                                edge_storage[incident[second]].second_block);
                        }
                        updates.push_back(update);
                    }
                }
            }

            stats.number_of_blocks = blocks;
            stats.number_of_edges =
                static_cast<Index>(edge_storage.size());
            stats.symbolic_fill_edges = 0;

            std::vector<Index> dimension_events(
                static_cast<std::size_t>(number_of_phases + 1), 0);
            for (const IntervalScope &scope : block_scopes)
            {
                dimension_events[static_cast<std::size_t>(
                    scope.first_phase)] += scope.dimension;
                dimension_events[static_cast<std::size_t>(
                    scope.last_phase + 1)] -= scope.dimension;
            }
            Index active_dimension = 0;
            for (Index phase = 0; phase < number_of_phases; ++phase)
            {
                active_dimension += dimension_events[
                    static_cast<std::size_t>(phase)];
                stats.maximum_active_dimension = std::max(
                    stats.maximum_active_dimension, active_dimension);
            }

            for (Index order_position = 0;
                 order_position < blocks; ++order_position)
            {
                const Index block = order[
                    static_cast<std::size_t>(order_position)];
                Index front_dimension = block_scopes[
                    static_cast<std::size_t>(block)].dimension;
                for (const std::size_t edge_index : later_edges[
                         static_cast<std::size_t>(order_position)])
                {
                    front_dimension += block_scopes[static_cast<std::size_t>(
                        edge_storage[edge_index].second_block)].dimension;
                }
                stats.maximum_front_dimension = std::max(
                    stats.maximum_front_dimension, front_dimension);
            }
            fatrop_assert_msg(
                stats.maximum_front_dimension
                    <= stats.maximum_active_dimension,
                "An interval elimination front exceeds the active width.");
        }

        void allocate_numeric_storage()
        {
            const std::size_t blocks = block_scopes.size();
            diagonal.reserve(blocks);
            modified_diagonal.reserve(blocks);
            factors.reserve(blocks);
            rhs.reserve(blocks);
            modified_rhs.reserve(blocks);
            solution.reserve(blocks);
            residual.reserve(blocks);
            correction.reserve(blocks);
            base_solution.reserve(blocks);
            for (const IntervalScope &scope : block_scopes)
            {
                diagonal.emplace_back(scope.dimension, scope.dimension);
                modified_diagonal.emplace_back(
                    scope.dimension, scope.dimension);
                factors.emplace_back(scope.dimension);
                rhs.emplace_back(
                    static_cast<std::size_t>(scope.dimension), 0.0);
                modified_rhs.emplace_back(
                    static_cast<std::size_t>(scope.dimension), 0.0);
                solution.emplace_back(
                    static_cast<std::size_t>(scope.dimension), 0.0);
                residual.emplace_back(
                    static_cast<std::size_t>(scope.dimension), 0.0);
                correction.emplace_back(
                    static_cast<std::size_t>(scope.dimension), 0.0);
                base_solution.emplace_back(
                    static_cast<std::size_t>(scope.dimension), 0.0);
            }
            seen_stamp.assign(blocks, 0);
            phase_offsets.reserve(blocks);
        }

        std::size_t find_edge(const Index first, const Index second) const
        {
            const auto found = edge_lookup.find(
                canonical_pair(first, second));
            fatrop_assert_msg(
                found != edge_lookup.end(),
                "A matrix contribution couples disjoint phase scopes.");
            return found->second;
        }

        void clear_numeric()
        {
            for (detail::DenseBlock &block : diagonal)
                block.set_zero();
            for (EdgeStorage &edge : edge_storage)
                edge.original.set_zero();
            for (std::vector<Scalar> &block_rhs : rhs)
                std::fill(block_rhs.begin(), block_rhs.end(), 0.0);
            is_factored = false;
            last_failure = IntervalKktFailureLocation::None;
            last_failure_block = -1;
        }

        void clear_rhs()
        {
            for (std::vector<Scalar> &block_rhs : rhs)
                std::fill(block_rhs.begin(), block_rhs.end(), 0.0);
            last_failure = IntervalKktFailureLocation::None;
            last_failure_block = -1;
        }

        void add_matrix_block(
            const Index row_block,
            const Index column_block,
            const MatRealView &contribution,
            const Scalar scale)
        {
            fatrop_assert_msg(
                std::isfinite(scale),
                "An interval-scoped matrix scale must be finite.");
            const Index blocks = static_cast<Index>(block_scopes.size());
            fatrop_assert_msg(
                row_block >= 0 && row_block < blocks
                && column_block >= 0 && column_block < blocks,
                "An interval-scoped matrix block index is out of range.");
            const Index row_dimension = block_scopes[
                static_cast<std::size_t>(row_block)].dimension;
            const Index column_dimension = block_scopes[
                static_cast<std::size_t>(column_block)].dimension;
            fatrop_assert_msg(
                contribution.m() == row_dimension
                && contribution.n() == column_dimension,
                "An interval-scoped matrix contribution has a wrong shape.");

            if (row_block == column_block)
            {
                detail::DenseBlock &target = diagonal[
                    static_cast<std::size_t>(row_block)];
                for (Index row = 0; row < row_dimension; ++row)
                {
                    for (Index column = 0;
                         column < column_dimension; ++column)
                    {
                        const Scalar value = contribution(row, column);
                        fatrop_assert_msg(
                            std::isfinite(value),
                            "A matrix contribution contains a nonfinite value.");
                        target(row, column) += scale * value;
                    }
                }
            }
            else
            {
                EdgeStorage &edge = edge_storage[
                    find_edge(row_block, column_block)];
                const bool input_is_elimination_orientation =
                    row_block == edge.first_block;
                for (Index row = 0; row < edge.original.rows; ++row)
                {
                    for (Index column = 0;
                         column < edge.original.columns; ++column)
                    {
                        const Scalar value = input_is_elimination_orientation
                            ? contribution(row, column)
                            : contribution(column, row);
                        fatrop_assert_msg(
                            std::isfinite(value),
                            "A matrix contribution contains a nonfinite value.");
                        edge.original(row, column) += scale * value;
                    }
                }
            }
            is_factored = false;
        }

        void add_rhs_block(
            const Index block,
            const VecRealView &contribution,
            const Scalar scale)
        {
            fatrop_assert_msg(
                std::isfinite(scale),
                "An interval-scoped RHS scale must be finite.");
            fatrop_assert_msg(
                block >= 0
                && block < static_cast<Index>(block_scopes.size()),
                "An interval-scoped RHS block index is out of range.");
            const Index dimension = block_scopes[
                static_cast<std::size_t>(block)].dimension;
            fatrop_assert_msg(
                contribution.m() == dimension,
                "An interval-scoped RHS contribution has a wrong size.");
            for (Index row = 0; row < dimension; ++row)
            {
                const Scalar value = contribution(row);
                fatrop_assert_msg(
                    std::isfinite(value),
                    "An RHS contribution contains a nonfinite value.");
                rhs[static_cast<std::size_t>(block)]
                   [static_cast<std::size_t>(row)] += scale * value;
            }
        }

        LinsolReturnFlag factorize()
        {
            last_failure = IntervalKktFailureLocation::None;
            last_failure_block = -1;
            is_factored = false;

            modified_diagonal = diagonal;
            for (EdgeStorage &edge : edge_storage)
            {
                edge.modified = edge.original;
                edge.response.set_zero();
            }

            for (Index order_position = 0;
                 order_position < static_cast<Index>(order.size());
                 ++order_position)
            {
                const Index block = order[
                    static_cast<std::size_t>(order_position)];
                if (!factors[static_cast<std::size_t>(block)].factor(
                        modified_diagonal[static_cast<std::size_t>(block)],
                        pivot_tolerance))
                {
                    last_failure =
                        IntervalKktFailureLocation::BlockFactorization;
                    last_failure_block = block;
                    return LinsolReturnFlag::NOFULL_RANK;
                }

                const std::vector<std::size_t> &incident = later_edges[
                    static_cast<std::size_t>(order_position)];
                for (const std::size_t edge_index : incident)
                {
                    EdgeStorage &edge = edge_storage[edge_index];
                    if (!factors[static_cast<std::size_t>(block)].solve(
                            edge.modified, edge.response))
                    {
                        last_failure = IntervalKktFailureLocation::Solve;
                        last_failure_block = block;
                        return LinsolReturnFlag::NAN_SOLUTION;
                    }
                }

                for (const SchurUpdate &update : schur_updates[
                         static_cast<std::size_t>(order_position)])
                {
                    const EdgeStorage &left =
                        edge_storage[update.left_edge];
                    const Index left_dimension = left.modified.columns;
                    const EdgeStorage &right =
                        edge_storage[update.right_edge];
                    const Index right_dimension =
                        right.response.columns;
                    detail::DenseBlock *target = nullptr;
                    if (update.diagonal_target)
                    {
                        target = &modified_diagonal[
                            static_cast<std::size_t>(update.target_block)];
                    }
                    else
                    {
                        target = &edge_storage[
                            update.target_edge].modified;
                    }

                    for (Index row = 0;
                         row < left_dimension; ++row)
                    {
                        for (Index column = 0;
                             column < right_dimension; ++column)
                        {
                            Scalar value = 0.0;
                            for (Index inner = 0;
                                 inner < left.modified.rows; ++inner)
                            {
                                value += left.modified(inner, row)
                                       * right.response(inner, column);
                            }
                            (*target)(row, column) -= value;
                        }
                    }
                }
            }

            is_factored = true;
            return LinsolReturnFlag::SUCCESS;
        }

        bool solve_factored(
            const std::vector<std::vector<Scalar>> &input_rhs,
            std::vector<std::vector<Scalar>> &output_solution)
        {
            modified_rhs = input_rhs;

            for (Index order_position = 0;
                 order_position < static_cast<Index>(order.size());
                 ++order_position)
            {
                const Index block = order[
                    static_cast<std::size_t>(order_position)];
                if (!factors[static_cast<std::size_t>(block)].solve(
                        modified_rhs[static_cast<std::size_t>(block)],
                        base_solution[static_cast<std::size_t>(block)]))
                    return false;

                for (const std::size_t edge_index : later_edges[
                         static_cast<std::size_t>(order_position)])
                {
                    const EdgeStorage &edge = edge_storage[edge_index];
                    std::vector<Scalar> &next_rhs = modified_rhs[
                        static_cast<std::size_t>(edge.second_block)];
                    for (Index row = 0;
                         row < edge.modified.columns; ++row)
                    {
                        Scalar update = 0.0;
                        for (Index inner = 0;
                             inner < edge.modified.rows; ++inner)
                        {
                            update += edge.modified(inner, row)
                                    * base_solution[
                                        static_cast<std::size_t>(block)]
                                      [static_cast<std::size_t>(inner)];
                        }
                        next_rhs[static_cast<std::size_t>(row)] -= update;
                    }
                }
            }

            output_solution = base_solution;
            for (Index reverse = static_cast<Index>(order.size());
                 reverse-- > 0;)
            {
                const Index block = order[static_cast<std::size_t>(reverse)];
                std::vector<Scalar> &current = output_solution[
                    static_cast<std::size_t>(block)];
                for (const std::size_t edge_index : later_edges[
                         static_cast<std::size_t>(reverse)])
                {
                    const EdgeStorage &edge = edge_storage[edge_index];
                    const std::vector<Scalar> &next = output_solution[
                        static_cast<std::size_t>(edge.second_block)];
                    for (Index row = 0; row < edge.response.rows; ++row)
                    {
                        Scalar update = 0.0;
                        for (Index column = 0;
                             column < edge.response.columns; ++column)
                        {
                            update += edge.response(row, column)
                                    * next[static_cast<std::size_t>(column)];
                        }
                        current[static_cast<std::size_t>(row)] -= update;
                    }
                }
            }
            return finite(output_solution);
        }

        void add_phase_contribution(
            const Index phase,
            const std::vector<Index> &active_blocks,
            const MatRealView &matrix,
            const VecRealView &phase_rhs,
            const Scalar matrix_scale,
            const Scalar rhs_scale)
        {
            fatrop_assert_msg(
                std::isfinite(matrix_scale) && std::isfinite(rhs_scale),
                "Interval-scoped phase contribution scales must be finite.");
            fatrop_assert_msg(
                phase >= 0 && phase < number_of_phases,
                "An interval-scoped phase contribution has an invalid phase.");

            if (seen_generation == std::numeric_limits<Index>::max())
            {
                std::fill(seen_stamp.begin(), seen_stamp.end(), 0);
                seen_generation = 1;
            }
            else
            {
                ++seen_generation;
            }
            phase_offsets.resize(active_blocks.size());
            Index total_dimension = 0;
            for (std::size_t item = 0;
                 item < active_blocks.size(); ++item)
            {
                const Index block = active_blocks[item];
                fatrop_assert_msg(
                    block >= 0
                    && block < static_cast<Index>(block_scopes.size()),
                    "An active interval-scoped block index is out of range.");
                fatrop_assert_msg(
                    seen_stamp[static_cast<std::size_t>(block)]
                        != seen_generation,
                    "An active interval-scoped block is listed twice.");
                seen_stamp[static_cast<std::size_t>(block)] =
                    seen_generation;
                const IntervalScope &scope = block_scopes[
                    static_cast<std::size_t>(block)];
                fatrop_assert_msg(
                    scope.first_phase <= phase && phase <= scope.last_phase,
                    "A phase contribution lists a block outside its scope.");
                phase_offsets[item] = total_dimension;
                total_dimension += scope.dimension;
            }
            fatrop_assert_msg(
                matrix.m() == total_dimension
                && matrix.n() == total_dimension
                && phase_rhs.m() == total_dimension,
                "An interval-scoped phase clique has inconsistent dimensions.");

            for (std::size_t row_item = 0;
                 row_item < active_blocks.size(); ++row_item)
            {
                const Index row_block = active_blocks[row_item];
                const Index row_dimension = block_scopes[
                    static_cast<std::size_t>(row_block)].dimension;
                add_rhs_block(
                    row_block,
                    phase_rhs.block(
                        row_dimension, phase_offsets[row_item]),
                    rhs_scale);
                for (std::size_t column_item = row_item;
                     column_item < active_blocks.size(); ++column_item)
                {
                    const Index column_block = active_blocks[column_item];
                    const Index column_dimension = block_scopes[
                        static_cast<std::size_t>(column_block)].dimension;
                    add_matrix_block(
                        row_block,
                        column_block,
                        matrix.block(
                            row_dimension,
                            column_dimension,
                            phase_offsets[row_item],
                            phase_offsets[column_item]),
                        matrix_scale);
                }
            }
        }

        static bool finite(
            const std::vector<std::vector<Scalar>> &blocks)
        {
            for (const std::vector<Scalar> &block : blocks)
                for (const Scalar value : block)
                    if (!std::isfinite(value))
                        return false;
            return true;
        }

        void form_residual(
            const std::vector<std::vector<Scalar>> &candidate,
            std::vector<std::vector<Scalar>> &output_residual,
            Scalar &residual_norm,
            Scalar &rhs_norm) const
        {
            output_residual = rhs;
            residual_norm = 0.0;
            rhs_norm = 0.0;

            for (std::size_t block = 0;
                 block < block_scopes.size(); ++block)
            {
                const Index dimension = block_scopes[block].dimension;
                for (Index row = 0; row < dimension; ++row)
                {
                    rhs_norm = std::max(
                        rhs_norm,
                        std::abs(rhs[block][static_cast<std::size_t>(row)]));
                    Scalar update = 0.0;
                    for (Index column = 0;
                         column < dimension; ++column)
                    {
                        update += diagonal[block](row, column)
                                * candidate[block][
                                    static_cast<std::size_t>(column)];
                    }
                    output_residual[block][static_cast<std::size_t>(row)]
                        -= update;
                }
            }

            for (const EdgeStorage &edge : edge_storage)
            {
                std::vector<Scalar> &first_residual = output_residual[
                    static_cast<std::size_t>(edge.first_block)];
                std::vector<Scalar> &second_residual = output_residual[
                    static_cast<std::size_t>(edge.second_block)];
                const std::vector<Scalar> &first_solution = candidate[
                    static_cast<std::size_t>(edge.first_block)];
                const std::vector<Scalar> &second_solution = candidate[
                    static_cast<std::size_t>(edge.second_block)];
                for (Index row = 0; row < edge.original.rows; ++row)
                {
                    Scalar update = 0.0;
                    for (Index column = 0;
                         column < edge.original.columns; ++column)
                    {
                        update += edge.original(row, column)
                                * second_solution[
                                    static_cast<std::size_t>(column)];
                    }
                    first_residual[static_cast<std::size_t>(row)] -= update;
                }
                for (Index row = 0; row < edge.original.columns; ++row)
                {
                    Scalar update = 0.0;
                    for (Index column = 0;
                         column < edge.original.rows; ++column)
                    {
                        update += edge.original(column, row)
                                * first_solution[
                                    static_cast<std::size_t>(column)];
                    }
                    second_residual[static_cast<std::size_t>(row)] -= update;
                }
            }

            for (const std::vector<Scalar> &block : output_residual)
                for (const Scalar value : block)
                    residual_norm = std::max(residual_norm, std::abs(value));
        }

        LinsolReturnFlag solve(
            const std::vector<VecRealView> &block_solution)
        {
            fatrop_assert_msg(
                is_factored,
                "IntervalScopedKktSolver::solve requires factorize first.");
            fatrop_assert_msg(
                block_solution.size() == block_scopes.size(),
                "The number of interval-scoped output blocks is inconsistent.");
            for (std::size_t block = 0;
                 block < block_scopes.size(); ++block)
            {
                fatrop_assert_msg(
                    block_solution[block].m()
                        == block_scopes[block].dimension,
                    "An interval-scoped output block has a wrong size.");
            }

            last_failure = IntervalKktFailureLocation::None;
            last_failure_block = -1;
            if (!solve_factored(rhs, solution))
            {
                last_failure = IntervalKktFailureLocation::Solve;
                return LinsolReturnFlag::NAN_SOLUTION;
            }

            for (Index refinement = 0; refinement < 2; ++refinement)
            {
                Scalar residual_norm = 0.0;
                Scalar rhs_norm = 0.0;
                form_residual(
                    solution, residual,
                    residual_norm, rhs_norm);
                if (!std::isfinite(residual_norm))
                {
                    last_failure =
                        IntervalKktFailureLocation::NonfiniteSolution;
                    return LinsolReturnFlag::NAN_SOLUTION;
                }
                if (residual_norm
                    <= 64.0 * std::numeric_limits<Scalar>::epsilon()
                     * std::max<Scalar>(rhs_norm, 1.0))
                    break;

                if (!solve_factored(residual, correction))
                {
                    last_failure = IntervalKktFailureLocation::Solve;
                    return LinsolReturnFlag::NAN_SOLUTION;
                }
                for (std::size_t block = 0;
                     block < solution.size(); ++block)
                {
                    for (std::size_t row = 0;
                         row < solution[block].size(); ++row)
                    {
                        solution[block][row] += correction[block][row];
                    }
                }
            }

            if (!finite(solution))
            {
                last_failure =
                    IntervalKktFailureLocation::NonfiniteSolution;
                return LinsolReturnFlag::NAN_SOLUTION;
            }
            for (std::size_t block = 0;
                 block < solution.size(); ++block)
            {
                for (Index row = 0;
                     row < block_scopes[block].dimension; ++row)
                {
                    block_solution[block](row) =
                        solution[block][static_cast<std::size_t>(row)];
                }
            }
            return LinsolReturnFlag::SUCCESS;
        }

        Index number_of_phases = 0;
        std::vector<IntervalScope> block_scopes;
        std::vector<Index> order;
        std::vector<Index> position;
        std::vector<IntervalKktEdge> public_edges;
        std::vector<EdgeStorage> edge_storage;
        std::map<std::pair<Index, Index>, std::size_t> edge_lookup;
        std::vector<std::vector<std::size_t>> later_edges;
        std::vector<std::vector<SchurUpdate>> schur_updates;
        IntervalKktSymbolicStats stats;

        std::vector<detail::DenseBlock> diagonal;
        std::vector<detail::DenseBlock> modified_diagonal;
        std::vector<detail::DenseBlockLu> factors;
        std::vector<std::vector<Scalar>> rhs;
        std::vector<std::vector<Scalar>> modified_rhs;
        std::vector<std::vector<Scalar>> solution;
        std::vector<std::vector<Scalar>> residual;
        std::vector<std::vector<Scalar>> correction;
        std::vector<std::vector<Scalar>> base_solution;
        std::vector<Index> seen_stamp;
        std::vector<Index> phase_offsets;
        Index seen_generation = 0;

        Scalar pivot_tolerance = 0.0;
        bool is_factored = false;
        IntervalKktFailureLocation last_failure =
            IntervalKktFailureLocation::None;
        Index last_failure_block = -1;
    };

    IntervalScopedKktSolver::IntervalScopedKktSolver(
        const Index number_of_phases,
        std::vector<IntervalScope> scopes)
        : impl_(std::make_unique<Impl>(
              number_of_phases, std::move(scopes)))
    {
    }

    IntervalScopedKktSolver::~IntervalScopedKktSolver() = default;
    IntervalScopedKktSolver::IntervalScopedKktSolver(
        IntervalScopedKktSolver &&) noexcept = default;
    IntervalScopedKktSolver &IntervalScopedKktSolver::operator=(
        IntervalScopedKktSolver &&) noexcept = default;

    Index IntervalScopedKktSolver::number_of_phases() const noexcept
    {
        return impl_->number_of_phases;
    }

    Index IntervalScopedKktSolver::number_of_blocks() const noexcept
    {
        return static_cast<Index>(impl_->block_scopes.size());
    }

    const std::vector<IntervalScope> &
    IntervalScopedKktSolver::scopes() const noexcept
    {
        return impl_->block_scopes;
    }

    const std::vector<Index> &
    IntervalScopedKktSolver::elimination_order() const noexcept
    {
        return impl_->order;
    }

    const std::vector<IntervalKktEdge> &
    IntervalScopedKktSolver::edges() const noexcept
    {
        return impl_->public_edges;
    }

    IntervalKktSymbolicStats
    IntervalScopedKktSolver::symbolic_stats() const noexcept
    {
        return impl_->stats;
    }

    void IntervalScopedKktSolver::set_pivot_tolerance(
        const Scalar tolerance)
    {
        fatrop_assert_msg(
            std::isfinite(tolerance) && tolerance >= 0.0,
            "The interval-scoped pivot tolerance must be finite and "
            "non-negative.");
        impl_->pivot_tolerance = tolerance;
    }

    Scalar IntervalScopedKktSolver::pivot_tolerance() const noexcept
    {
        return impl_->pivot_tolerance;
    }

    IntervalKktFailureLocation
    IntervalScopedKktSolver::last_failure_location() const noexcept
    {
        return impl_->last_failure;
    }

    Index IntervalScopedKktSolver::last_failure_block() const noexcept
    {
        return impl_->last_failure_block;
    }

    void IntervalScopedKktSolver::clear_numeric()
    {
        impl_->clear_numeric();
    }

    void IntervalScopedKktSolver::clear_rhs()
    {
        impl_->clear_rhs();
    }

    void IntervalScopedKktSolver::add_matrix_block(
        const Index row_block,
        const Index column_block,
        const MatRealView &contribution,
        const Scalar scale)
    {
        impl_->add_matrix_block(
            row_block, column_block, contribution, scale);
    }

    void IntervalScopedKktSolver::add_rhs_block(
        const Index block,
        const VecRealView &contribution,
        const Scalar scale)
    {
        impl_->add_rhs_block(block, contribution, scale);
    }

    void IntervalScopedKktSolver::add_phase_contribution(
        const Index phase,
        const std::vector<Index> &active_blocks,
        const MatRealView &matrix,
        const VecRealView &rhs,
        const Scalar matrix_scale,
        const Scalar rhs_scale)
    {
        impl_->add_phase_contribution(
            phase, active_blocks, matrix, rhs,
            matrix_scale, rhs_scale);
    }

    LinsolReturnFlag IntervalScopedKktSolver::factorize()
    {
        return impl_->factorize();
    }

    LinsolReturnFlag IntervalScopedKktSolver::solve(
        const std::vector<VecRealView> &block_solution)
    {
        return impl_->solve(block_solution);
    }

    LinsolReturnFlag IntervalScopedKktSolver::factor_and_solve(
        const std::vector<VecRealView> &block_solution)
    {
        const LinsolReturnFlag factor_flag = factorize();
        if (factor_flag != LinsolReturnFlag::SUCCESS)
            return factor_flag;
        return solve(block_solution);
    }
} // namespace fatrop
