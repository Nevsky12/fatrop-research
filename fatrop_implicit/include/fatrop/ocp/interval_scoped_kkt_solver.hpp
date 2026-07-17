//
// Copyright (c) 2026
//

#ifndef __fatrop_ocp_interval_scoped_kkt_solver_hpp__
#define __fatrop_ocp_interval_scoped_kkt_solver_hpp__

#include "fatrop/context/context.hpp"
#include "fatrop/linear_algebra/linear_solver_return_flags.hpp"
#include "fatrop/linear_algebra/matrix.hpp"
#include "fatrop/linear_algebra/vector.hpp"

#include <memory>
#include <vector>

namespace fatrop
{
    /**
     * A one-copy variable block and the closed interval of phases in which it
     * may occur.  Typical scopes are [f,f] for a phase parameter, [f,f+1]
     * for an interface quantity, and [0,F-1] for a mission-global parameter.
     */
    struct IntervalScope
    {
        Index dimension = 0;
        Index first_phase = 0;
        Index last_phase = 0;
    };

    /** A structural off-diagonal block in original block numbering. */
    struct IntervalKktEdge
    {
        Index first_block = 0;
        Index second_block = 0;
    };

    struct IntervalKktSymbolicStats
    {
        Index number_of_blocks = 0;
        Index number_of_edges = 0;
        Index maximum_active_dimension = 0;
        Index maximum_front_dimension = 0;
        Index symbolic_fill_edges = 0;
    };

    enum class IntervalKktFailureLocation
    {
        None,
        BlockFactorization,
        Solve,
        NonfiniteSolution
    };

    /**
     * @brief Block factorization for symmetric KKT systems whose variable
     *        blocks have contiguous phase support.
     *
     * Two blocks are structurally adjacent exactly when their phase intervals
     * overlap.  Ordering blocks by increasing right endpoint is a perfect
     * elimination ordering of this interval graph.  Consequently, prescribed
     * block Gaussian elimination introduces no structural fill and its fronts
     * are bounded by the maximum simultaneously active scoped dimension.
     *
     * The class owns numeric assembly as well as the symbolic factorization.
     * `add_phase_contribution()` is the intended trajectory-condensation
     * boundary: a phase Riccati solve contributes a dense clique only among
     * the one-copy blocks active in that phase.
     *
     * The numeric matrix is treated as symmetric.  Off-diagonal input is
     * stored once and its transpose is used for the lower triangle.  Pivoting
     * is restricted to each diagonal block; an outer primal-dual algorithm
     * must therefore regularize the KKT system until these prescribed pivots
     * are nonsingular.
     */
    class IntervalScopedKktSolver
    {
    public:
        IntervalScopedKktSolver(
            Index number_of_phases,
            std::vector<IntervalScope> scopes);
        ~IntervalScopedKktSolver();

        IntervalScopedKktSolver(IntervalScopedKktSolver &&) noexcept;
        IntervalScopedKktSolver &operator=(
            IntervalScopedKktSolver &&) noexcept;

        IntervalScopedKktSolver(const IntervalScopedKktSolver &) = delete;
        IntervalScopedKktSolver &operator=(
            const IntervalScopedKktSolver &) = delete;

        Index number_of_phases() const noexcept;
        Index number_of_blocks() const noexcept;
        const std::vector<IntervalScope> &scopes() const noexcept;
        const std::vector<Index> &elimination_order() const noexcept;
        const std::vector<IntervalKktEdge> &edges() const noexcept;
        IntervalKktSymbolicStats symbolic_stats() const noexcept;

        void set_pivot_tolerance(Scalar tolerance);
        Scalar pivot_tolerance() const noexcept;

        IntervalKktFailureLocation last_failure_location() const noexcept;
        Index last_failure_block() const noexcept;

        /** Clear all matrix and right-hand-side contributions. */
        void clear_numeric();

        /** Clear only the RHS and retain the current matrix factorization. */
        void clear_rhs();

        /** Add a dense block in original block numbering. */
        void add_matrix_block(
            Index row_block,
            Index column_block,
            const MatRealView &contribution,
            Scalar scale = 1.0);

        /** Add a right-hand-side contribution in original block numbering. */
        void add_rhs_block(
            Index block,
            const VecRealView &contribution,
            Scalar scale = 1.0);

        /**
         * Add one condensed phase clique.  `active_blocks` defines the block
         * order in `matrix` and `rhs`; every listed scope must contain phase.
         */
        void add_phase_contribution(
            Index phase,
            const std::vector<Index> &active_blocks,
            const MatRealView &matrix,
            const VecRealView &rhs,
            Scalar matrix_scale = 1.0,
            Scalar rhs_scale = 1.0);

        /** Factor the currently assembled matrix. */
        LinsolReturnFlag factorize();

        /**
         * Solve for the currently assembled RHS using the current
         * factorization, including two residual-correction steps.
         */
        LinsolReturnFlag solve(
            const std::vector<VecRealView> &block_solution);

        /** Factor and solve the currently assembled system. */
        LinsolReturnFlag factor_and_solve(
            const std::vector<VecRealView> &block_solution);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace fatrop

#endif // __fatrop_ocp_interval_scoped_kkt_solver_hpp__
