//
// Copyright (c) 2026
//

#ifndef __fatrop_linear_algebra_dense_block_hpp__
#define __fatrop_linear_algebra_dense_block_hpp__

#include "fatrop/context/context.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace fatrop
{
    namespace detail
    {
        /**
         * Small row-major dense block used by the structure-exploiting
         * factorizations.  Keeping this storage independent of BLASFEO makes
         * symbolic block algorithms easy to test; BLASFEO remains the storage
         * used at the public solver boundary.
         */
        struct DenseBlock
        {
            DenseBlock() = default;

            DenseBlock(const Index rows_value, const Index columns_value)
            {
                reset(rows_value, columns_value);
            }

            void reset(const Index rows_value, const Index columns_value)
            {
                rows = rows_value;
                columns = columns_value;
                values.assign(
                    static_cast<std::size_t>(rows * columns), 0.0);
            }

            void set_zero()
            {
                std::fill(values.begin(), values.end(), 0.0);
            }

            Scalar &operator()(const Index row, const Index column)
            {
                return values[static_cast<std::size_t>(
                    row * columns + column)];
            }

            Scalar operator()(const Index row, const Index column) const
            {
                return values[static_cast<std::size_t>(
                    row * columns + column)];
            }

            Index rows = 0;
            Index columns = 0;
            std::vector<Scalar> values;
        };

        /**
         * Dense LU with partial row pivoting restricted to one symbolic
         * block.  Restricting pivots to a block is what preserves the outer
         * Riccati/interval sparsity; callers are responsible for regularizing
         * the KKT system until the prescribed block pivots are nonsingular.
         */
        class DenseBlockLu
        {
        public:
            DenseBlockLu() = default;

            explicit DenseBlockLu(const Index size_value)
            {
                reset(size_value);
            }

            void reset(const Index size_value)
            {
                size_ = size_value;
                factor_.reset(size_, size_);
                pivots_.assign(static_cast<std::size_t>(size_), 0);
            }

            bool factor(
                const DenseBlock &matrix,
                const Scalar relative_tolerance)
            {
                if (matrix.rows != size_ || matrix.columns != size_)
                    return false;

                Scalar scale = 0.0;
                for (Index row = 0; row < size_; ++row)
                {
                    for (Index column = 0; column < size_; ++column)
                    {
                        const Scalar value = matrix(row, column);
                        if (!std::isfinite(value))
                            return false;
                        factor_(row, column) = value;
                        scale = std::max(scale, std::abs(value));
                    }
                }

                const Scalar threshold =
                    relative_tolerance * std::max<Scalar>(scale, 1.0);
                for (Index column = 0; column < size_; ++column)
                {
                    Index pivot = column;
                    Scalar pivot_size =
                        std::abs(factor_(column, column));
                    for (Index row = column + 1; row < size_; ++row)
                    {
                        const Scalar candidate =
                            std::abs(factor_(row, column));
                        if (candidate > pivot_size)
                        {
                            pivot = row;
                            pivot_size = candidate;
                        }
                    }
                    if (!std::isfinite(pivot_size)
                        || pivot_size <= threshold)
                        return false;

                    pivots_[static_cast<std::size_t>(column)] = pivot;
                    if (pivot != column)
                    {
                        for (Index entry = 0; entry < size_; ++entry)
                        {
                            std::swap(
                                factor_(column, entry),
                                factor_(pivot, entry));
                        }
                    }

                    const Scalar diagonal = factor_(column, column);
                    for (Index row = column + 1; row < size_; ++row)
                    {
                        factor_(row, column) /= diagonal;
                        const Scalar multiplier = factor_(row, column);
                        for (Index entry = column + 1;
                             entry < size_; ++entry)
                        {
                            factor_(row, entry) -=
                                multiplier * factor_(column, entry);
                        }
                    }
                }
                return true;
            }

            bool solve(
                const std::vector<Scalar> &rhs,
                std::vector<Scalar> &solution) const
            {
                if (rhs.size() != static_cast<std::size_t>(size_))
                    return false;

                solution = rhs;
                for (Index row = 0; row < size_; ++row)
                {
                    const Index pivot =
                        pivots_[static_cast<std::size_t>(row)];
                    if (pivot != row)
                    {
                        std::swap(
                            solution[static_cast<std::size_t>(row)],
                            solution[static_cast<std::size_t>(pivot)]);
                    }
                }
                for (Index row = 0; row < size_; ++row)
                {
                    Scalar value = solution[static_cast<std::size_t>(row)];
                    for (Index column = 0; column < row; ++column)
                    {
                        value -= factor_(row, column)
                               * solution[static_cast<std::size_t>(column)];
                    }
                    solution[static_cast<std::size_t>(row)] = value;
                }
                for (Index row = size_; row-- > 0;)
                {
                    Scalar value = solution[static_cast<std::size_t>(row)];
                    for (Index column = row + 1;
                         column < size_; ++column)
                    {
                        value -= factor_(row, column)
                               * solution[static_cast<std::size_t>(column)];
                    }
                    const Scalar diagonal = factor_(row, row);
                    if (!std::isfinite(diagonal) || diagonal == 0.0)
                        return false;
                    value /= diagonal;
                    if (!std::isfinite(value))
                        return false;
                    solution[static_cast<std::size_t>(row)] = value;
                }
                return true;
            }

            bool solve(
                const DenseBlock &rhs,
                DenseBlock &solution) const
            {
                if (rhs.rows != size_)
                    return false;

                // Apply the retained row pivots and triangular solves to all
                // right-hand sides together.  Besides improving locality for
                // Riccati response blocks, this avoids two temporary vectors
                // and one allocation per scalar column.
                solution.rows = size_;
                solution.columns = rhs.columns;
                solution.values.resize(static_cast<std::size_t>(
                    size_ * rhs.columns));
                for (Index row = 0; row < size_; ++row)
                {
                    for (Index column = 0;
                         column < rhs.columns; ++column)
                    {
                        const Scalar value = rhs(row, column);
                        if (!std::isfinite(value))
                            return false;
                        solution(row, column) = value;
                    }
                }
                for (Index row = 0; row < size_; ++row)
                {
                    const Index pivot =
                        pivots_[static_cast<std::size_t>(row)];
                    if (pivot != row)
                    {
                        for (Index column = 0;
                             column < rhs.columns; ++column)
                        {
                            std::swap(
                                solution(row, column),
                                solution(pivot, column));
                        }
                    }
                }
                for (Index row = 0; row < size_; ++row)
                {
                    for (Index column = 0;
                         column < rhs.columns; ++column)
                    {
                        Scalar value = solution(row, column);
                        for (Index inner = 0; inner < row; ++inner)
                            value -= factor_(row, inner)
                                   * solution(inner, column);
                        solution(row, column) = value;
                    }
                }
                for (Index row = size_; row-- > 0;)
                {
                    const Scalar diagonal = factor_(row, row);
                    if (!std::isfinite(diagonal) || diagonal == 0.0)
                        return false;
                    for (Index column = 0;
                         column < rhs.columns; ++column)
                    {
                        Scalar value = solution(row, column);
                        for (Index inner = row + 1;
                             inner < size_; ++inner)
                            value -= factor_(row, inner)
                                   * solution(inner, column);
                        value /= diagonal;
                        if (!std::isfinite(value))
                            return false;
                        solution(row, column) = value;
                    }
                }
                return true;
            }

        private:
            Index size_ = 0;
            DenseBlock factor_;
            std::vector<Index> pivots_;
        };
    } // namespace detail
} // namespace fatrop

#endif // __fatrop_linear_algebra_dense_block_hpp__
