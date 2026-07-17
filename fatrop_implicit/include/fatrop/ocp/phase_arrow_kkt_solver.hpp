//
// Copyright (c) 2026
//

#ifndef __fatrop_ocp_phase_arrow_kkt_solver_hpp__
#define __fatrop_ocp_phase_arrow_kkt_solver_hpp__

#include "fatrop/common/exception.hpp"
#include "fatrop/context/context.hpp"
#include "fatrop/linear_algebra/dense_block.hpp"
#include "fatrop/linear_algebra/linear_solver_return_flags.hpp"
#include "fatrop/linear_algebra/matrix.hpp"
#include "fatrop/linear_algebra/vector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace fatrop
{
    /**
     * @brief Location of the last phase-arrow factorization failure.
     */
    enum class PhaseArrowKktFailureLocation
    {
        None,
        PhaseBlockFactorization,
        GlobalFactorization,
        Solve,
        NonfiniteSolution
    };

    /**
     * @brief Solve a heterogeneous block-tridiagonal KKT system with a
     *        genuinely global dense arrow.
     *
     * The reduced system is
     *
     * \f[
     * \begin{bmatrix}
     * D_0 & C_0 &       &       & E_0 \\
     * C_0^T & D_1 & C_1 &       & E_1 \\
     *       & \ddots & \ddots & \ddots & \vdots \\
     *       &        & C_{F-2}^T & D_{F-1} & E_{F-1} \\
     * E_0^T & E_1^T & \cdots & E_{F-1}^T & G
     * \end{bmatrix}
     * \begin{bmatrix}q_0\\q_1\\\vdots\\q_{F-1}\\g\end{bmatrix}
     * =
     * \begin{bmatrix}b_0\\b_1\\\vdots\\b_{F-1}\\b_g\end{bmatrix}.
     * \f]
     *
     * In the MOCP application, q_f contains the variables that should die at
     * phase f (phase-local static parameters and, except at the last phase,
     * the outgoing linkage multiplier).  Eliminating a phase trajectory then
     * contributes only to D_{f-1}, D_f, and C_{f-1}.  Only mission-global
     * variables remain in g.
     *
     * The phase chain is factored by a heterogeneous block Thomas recursion.
     * The complete response to g is propagated as a matrix and only the final
     * global Schur complement is factored densely.  Hence the cubic work is a
     * sum over phase block sizes rather than the cube of their sum.
     *
     * This class solves the algebraic reduced system.  Per-phase trajectory
     * condensation and back-substitution are deliberately separate so the
     * same kernel can be used by explicit and implicit OCP recursions.
     */
    class PhaseArrowKktSolver
    {
    public:
        PhaseArrowKktSolver(
            std::vector<Index> phase_block_sizes,
            const Index number_of_global_variables)
            : phase_block_sizes_(std::move(phase_block_sizes)),
              number_of_global_variables_(number_of_global_variables)
        {
            fatrop_assert_msg(
                !phase_block_sizes_.empty(),
                "A phase-arrow system must contain at least one phase block.");
            fatrop_assert_msg(
                number_of_global_variables_ >= 0,
                "The phase-arrow global dimension must be non-negative.");
            for (const Index size : phase_block_sizes_)
            {
                fatrop_assert_msg(
                    size >= 0,
                    "Phase-arrow block dimensions must be non-negative.");
                phase_factors_.emplace_back(size);
                modified_diagonal_.emplace_back(size, size);
                modified_arrow_.emplace_back(
                    size, number_of_global_variables_);
                arrow_response_.emplace_back(
                    size, number_of_global_variables_);
                modified_rhs_.emplace_back(
                    static_cast<std::size_t>(size), 0.0);
                base_solution_.emplace_back(
                    static_cast<std::size_t>(size), 0.0);
                solution_.emplace_back(
                    static_cast<std::size_t>(size), 0.0);
                residual_.emplace_back(
                    static_cast<std::size_t>(size), 0.0);
                correction_.emplace_back(
                    static_cast<std::size_t>(size), 0.0);
            }
            global_schur_.reset(
                number_of_global_variables_,
                number_of_global_variables_);
            global_factor_.reset(number_of_global_variables_);
            global_solution_.assign(
                static_cast<std::size_t>(number_of_global_variables_), 0.0);
            global_residual_.assign(
                static_cast<std::size_t>(number_of_global_variables_), 0.0);
            global_correction_.assign(
                static_cast<std::size_t>(number_of_global_variables_), 0.0);
        }

        Index number_of_phases() const noexcept
        {
            return static_cast<Index>(phase_block_sizes_.size());
        }

        Index number_of_global_variables() const noexcept
        {
            return number_of_global_variables_;
        }

        std::vector<Index> const &phase_block_sizes() const noexcept
        {
            return phase_block_sizes_;
        }

        void set_pivot_tolerance(const Scalar tolerance)
        {
            fatrop_assert_msg(
                std::isfinite(tolerance) && tolerance >= 0.0,
                "The phase-arrow pivot tolerance must be finite and non-negative.");
            pivot_tolerance_ = tolerance;
        }

        Scalar pivot_tolerance() const noexcept
        {
            return pivot_tolerance_;
        }

        PhaseArrowKktFailureLocation last_failure_location() const noexcept
        {
            return last_failure_location_;
        }

        Index last_failure_phase() const noexcept
        {
            return last_failure_phase_;
        }

        /**
         * @brief Factor and solve A x = b.
         *
         * `diagonal[f]` is D_f, `coupling[f]` is C_f, and `arrow[f]`
         * is E_f.  The input matrices are not modified.  Zero-dimensional
         * phase or global blocks are accepted when their incident matrix
         * dimensions are also zero.
         */
        LinsolReturnFlag solve(
            const std::vector<MatRealView> &diagonal,
            const std::vector<MatRealView> &coupling,
            const std::vector<MatRealView> &arrow,
            const MatRealView &global,
            const std::vector<VecRealView> &rhs_phase,
            const VecRealView &rhs_global,
            const std::vector<VecRealView> &phase_solution,
            const VecRealView &global_solution)
        {
            validate_dimensions(
                diagonal, coupling, arrow, global,
                rhs_phase, rhs_global,
                phase_solution, global_solution);
            last_failure_location_ = PhaseArrowKktFailureLocation::None;
            last_failure_phase_ = -1;

            if (!factor_chain(diagonal, coupling, arrow, global))
                return LinsolReturnFlag::NOFULL_RANK;
            if (!solve_factored(
                    coupling, arrow, rhs_phase, rhs_global,
                    solution_, global_solution_))
            {
                last_failure_location_ = PhaseArrowKktFailureLocation::Solve;
                return LinsolReturnFlag::NAN_SOLUTION;
            }

            // Two residual-correction steps are inexpensive at phase level
            // and materially improve mixed primal/linkage-dual scaling.
            for (Index iteration = 0; iteration < 2; ++iteration)
            {
                Scalar residual_norm = 0.0;
                Scalar rhs_norm = 0.0;
                form_residual(
                    diagonal, coupling, arrow, global,
                    rhs_phase, rhs_global,
                    solution_, global_solution_,
                    residual_, global_residual_,
                    residual_norm, rhs_norm);
                if (!std::isfinite(residual_norm))
                {
                    last_failure_location_ =
                        PhaseArrowKktFailureLocation::NonfiniteSolution;
                    return LinsolReturnFlag::NAN_SOLUTION;
                }
                if (residual_norm <=
                    64.0 * std::numeric_limits<Scalar>::epsilon()
                    * std::max<Scalar>(rhs_norm, 1.0))
                    break;

                if (!solve_factored_dense_rhs(
                        coupling, arrow,
                        residual_, global_residual_,
                        correction_, global_correction_))
                {
                    last_failure_location_ =
                        PhaseArrowKktFailureLocation::Solve;
                    return LinsolReturnFlag::NAN_SOLUTION;
                }
                for (std::size_t phase = 0;
                     phase < solution_.size(); ++phase)
                {
                    for (Index row = 0;
                         row < phase_block_sizes_[phase]; ++row)
                    {
                        solution_[phase][static_cast<std::size_t>(row)] +=
                            correction_[phase][static_cast<std::size_t>(row)];
                    }
                }
                for (Index row = 0;
                     row < number_of_global_variables_; ++row)
                {
                    global_solution_[static_cast<std::size_t>(row)] +=
                        global_correction_[static_cast<std::size_t>(row)];
                }
            }

            if (!finite_solution(solution_, global_solution_))
            {
                last_failure_location_ =
                    PhaseArrowKktFailureLocation::NonfiniteSolution;
                return LinsolReturnFlag::NAN_SOLUTION;
            }
            for (std::size_t phase = 0;
                 phase < phase_solution.size(); ++phase)
            {
                for (Index row = 0;
                     row < phase_block_sizes_[phase]; ++row)
                {
                    phase_solution[phase](row) =
                        solution_[phase][static_cast<std::size_t>(row)];
                }
            }
            for (Index row = 0;
                 row < number_of_global_variables_; ++row)
            {
                global_solution(row) =
                    global_solution_[static_cast<std::size_t>(row)];
            }
            return LinsolReturnFlag::SUCCESS;
        }

    private:
        using DenseMatrix = detail::DenseBlock;
        using DenseLu = detail::DenseBlockLu;

        void validate_dimensions(
            const std::vector<MatRealView> &diagonal,
            const std::vector<MatRealView> &coupling,
            const std::vector<MatRealView> &arrow,
            const MatRealView &global,
            const std::vector<VecRealView> &rhs_phase,
            const VecRealView &rhs_global,
            const std::vector<VecRealView> &phase_solution,
            const VecRealView &global_solution) const
        {
            const std::size_t phases = phase_block_sizes_.size();
            fatrop_assert_msg(
                diagonal.size() == phases
                && arrow.size() == phases
                && rhs_phase.size() == phases
                && phase_solution.size() == phases,
                "The phase-arrow phase-vector cardinalities are inconsistent.");
            fatrop_assert_msg(
                coupling.size() + 1 == phases,
                "The phase-arrow chain must have one fewer coupling than blocks.");
            for (std::size_t phase = 0; phase < phases; ++phase)
            {
                const Index size = phase_block_sizes_[phase];
                fatrop_assert_msg(
                    diagonal[phase].m() == size
                    && diagonal[phase].n() == size,
                    "A phase-arrow diagonal block has an inconsistent shape.");
                fatrop_assert_msg(
                    arrow[phase].m() == size
                    && arrow[phase].n()
                       == number_of_global_variables_,
                    "A phase-arrow global coupling has an inconsistent shape.");
                fatrop_assert_msg(
                    rhs_phase[phase].m() == size
                    && phase_solution[phase].m() == size,
                    "A phase-arrow phase vector has an inconsistent size.");
                if (phase + 1 < phases)
                {
                    fatrop_assert_msg(
                        coupling[phase].m() == size
                        && coupling[phase].n()
                           == phase_block_sizes_[phase + 1],
                        "A phase-arrow adjacent coupling has an inconsistent shape.");
                }
            }
            fatrop_assert_msg(
                global.m() == number_of_global_variables_
                && global.n() == number_of_global_variables_,
                "The phase-arrow global block has an inconsistent shape.");
            fatrop_assert_msg(
                rhs_global.m() == number_of_global_variables_
                && global_solution.m()
                   == number_of_global_variables_,
                "The phase-arrow global vectors have inconsistent sizes.");
        }

        static void copy_matrix(
            const MatRealView &source,
            DenseMatrix &destination)
        {
            destination.reset(source.m(), source.n());
            for (Index row = 0; row < source.m(); ++row)
                for (Index column = 0; column < source.n(); ++column)
                    destination(row, column) = source(row, column);
        }

        bool factor_chain(
            const std::vector<MatRealView> &diagonal,
            const std::vector<MatRealView> &coupling,
            const std::vector<MatRealView> &arrow,
            const MatRealView &global)
        {
            copy_matrix(diagonal.front(), modified_diagonal_.front());
            copy_matrix(arrow.front(), modified_arrow_.front());
            if (!phase_factors_.front().factor(
                    modified_diagonal_.front(), pivot_tolerance_))
            {
                last_failure_location_ =
                    PhaseArrowKktFailureLocation::PhaseBlockFactorization;
                last_failure_phase_ = 0;
                return false;
            }

            for (std::size_t phase = 1;
                 phase < phase_block_sizes_.size(); ++phase)
            {
                const MatRealView &previous_coupling =
                    coupling[phase - 1];
                DenseMatrix coupling_dense;
                copy_matrix(previous_coupling, coupling_dense);
                DenseMatrix coupling_response;
                if (!phase_factors_[phase - 1].solve(
                        coupling_dense, coupling_response))
                    return false;

                copy_matrix(diagonal[phase], modified_diagonal_[phase]);
                const Index previous_size =
                    phase_block_sizes_[phase - 1];
                const Index size = phase_block_sizes_[phase];
                for (Index row = 0; row < size; ++row)
                {
                    for (Index column = 0; column < size; ++column)
                    {
                        Scalar update = 0.0;
                        for (Index inner = 0;
                             inner < previous_size; ++inner)
                        {
                            update += previous_coupling(inner, row)
                                    * coupling_response(inner, column);
                        }
                        modified_diagonal_[phase](row, column) -= update;
                    }
                }

                DenseMatrix arrow_response;
                if (!phase_factors_[phase - 1].solve(
                        modified_arrow_[phase - 1],
                        arrow_response))
                    return false;
                copy_matrix(arrow[phase], modified_arrow_[phase]);
                for (Index row = 0; row < size; ++row)
                {
                    for (Index column = 0;
                         column < number_of_global_variables_; ++column)
                    {
                        Scalar update = 0.0;
                        for (Index inner = 0;
                             inner < previous_size; ++inner)
                        {
                            update += previous_coupling(inner, row)
                                    * arrow_response(inner, column);
                        }
                        modified_arrow_[phase](row, column) -= update;
                    }
                }

                if (!phase_factors_[phase].factor(
                        modified_diagonal_[phase], pivot_tolerance_))
                {
                    last_failure_location_ =
                        PhaseArrowKktFailureLocation::PhaseBlockFactorization;
                    last_failure_phase_ = static_cast<Index>(phase);
                    return false;
                }
            }

            // Backward matrix solve R = T^{-1}(-E).
            for (std::size_t reverse = phase_block_sizes_.size();
                 reverse-- > 0;)
            {
                const Index size = phase_block_sizes_[reverse];
                DenseMatrix rhs(size, number_of_global_variables_);
                for (Index row = 0; row < size; ++row)
                {
                    for (Index column = 0;
                         column < number_of_global_variables_; ++column)
                    {
                        Scalar value =
                            -modified_arrow_[reverse](row, column);
                        if (reverse + 1 < phase_block_sizes_.size())
                        {
                            const MatRealView &next_coupling =
                                coupling[reverse];
                            for (Index inner = 0;
                                 inner < phase_block_sizes_[reverse + 1];
                                 ++inner)
                            {
                                value -= next_coupling(row, inner)
                                       * arrow_response_[reverse + 1](
                                             inner, column);
                            }
                        }
                        rhs(row, column) = value;
                    }
                }
                if (!phase_factors_[reverse].solve(
                        rhs, arrow_response_[reverse]))
                    return false;
            }

            copy_matrix(global, global_schur_);
            for (std::size_t phase = 0;
                 phase < phase_block_sizes_.size(); ++phase)
            {
                const Index size = phase_block_sizes_[phase];
                for (Index row = 0;
                     row < number_of_global_variables_; ++row)
                {
                    for (Index column = 0;
                         column < number_of_global_variables_; ++column)
                    {
                        Scalar update = 0.0;
                        for (Index inner = 0; inner < size; ++inner)
                        {
                            update += arrow[phase](inner, row)
                                    * arrow_response_[phase](inner, column);
                        }
                        global_schur_(row, column) += update;
                    }
                }
            }
            if (!global_factor_.factor(
                    global_schur_, pivot_tolerance_))
            {
                last_failure_location_ =
                    PhaseArrowKktFailureLocation::GlobalFactorization;
                last_failure_phase_ = -1;
                return false;
            }
            return true;
        }

        bool solve_factored(
            const std::vector<MatRealView> &coupling,
            const std::vector<MatRealView> &arrow,
            const std::vector<VecRealView> &rhs_phase,
            const VecRealView &rhs_global,
            std::vector<std::vector<Scalar>> &phase_solution,
            std::vector<Scalar> &global_solution)
        {
            std::vector<std::vector<Scalar>> dense_rhs;
            dense_rhs.reserve(rhs_phase.size());
            for (std::size_t phase = 0; phase < rhs_phase.size(); ++phase)
            {
                dense_rhs.emplace_back(
                    static_cast<std::size_t>(phase_block_sizes_[phase]),
                    0.0);
                for (Index row = 0;
                     row < phase_block_sizes_[phase]; ++row)
                {
                    dense_rhs.back()[static_cast<std::size_t>(row)] =
                        rhs_phase[phase](row);
                }
            }
            std::vector<Scalar> dense_global_rhs(
                static_cast<std::size_t>(number_of_global_variables_),
                0.0);
            for (Index row = 0;
                 row < number_of_global_variables_; ++row)
                dense_global_rhs[static_cast<std::size_t>(row)] =
                    rhs_global(row);
            return solve_factored_dense_rhs(
                coupling, arrow,
                dense_rhs, dense_global_rhs,
                phase_solution, global_solution);
        }

        bool solve_factored_dense_rhs(
            const std::vector<MatRealView> &coupling,
            const std::vector<MatRealView> &arrow,
            const std::vector<std::vector<Scalar>> &rhs_phase,
            const std::vector<Scalar> &rhs_global,
            std::vector<std::vector<Scalar>> &phase_solution,
            std::vector<Scalar> &global_solution)
        {
            modified_rhs_.front() = rhs_phase.front();
            for (std::size_t phase = 1;
                 phase < phase_block_sizes_.size(); ++phase)
            {
                std::vector<Scalar> previous_response;
                if (!phase_factors_[phase - 1].solve(
                        modified_rhs_[phase - 1],
                        previous_response))
                    return false;
                const MatRealView &previous_coupling =
                    coupling[phase - 1];
                const Index size = phase_block_sizes_[phase];
                const Index previous_size =
                    phase_block_sizes_[phase - 1];
                modified_rhs_[phase] = rhs_phase[phase];
                for (Index row = 0; row < size; ++row)
                {
                    Scalar update = 0.0;
                    for (Index inner = 0;
                         inner < previous_size; ++inner)
                    {
                        update += previous_coupling(inner, row)
                                * previous_response[
                                      static_cast<std::size_t>(inner)];
                    }
                    modified_rhs_[phase][static_cast<std::size_t>(row)] -=
                        update;
                }
            }

            for (std::size_t reverse = phase_block_sizes_.size();
                 reverse-- > 0;)
            {
                std::vector<Scalar> rhs = modified_rhs_[reverse];
                if (reverse + 1 < phase_block_sizes_.size())
                {
                    const MatRealView &next_coupling = coupling[reverse];
                    for (Index row = 0;
                         row < phase_block_sizes_[reverse]; ++row)
                    {
                        Scalar update = 0.0;
                        for (Index inner = 0;
                             inner < phase_block_sizes_[reverse + 1];
                             ++inner)
                        {
                            update += next_coupling(row, inner)
                                    * base_solution_[reverse + 1][
                                          static_cast<std::size_t>(inner)];
                        }
                        rhs[static_cast<std::size_t>(row)] -= update;
                    }
                }
                if (!phase_factors_[reverse].solve(
                        rhs, base_solution_[reverse]))
                    return false;
            }

            std::vector<Scalar> reduced_global_rhs = rhs_global;
            for (std::size_t phase = 0;
                 phase < phase_block_sizes_.size(); ++phase)
            {
                for (Index row = 0;
                     row < number_of_global_variables_; ++row)
                {
                    Scalar update = 0.0;
                    for (Index inner = 0;
                         inner < phase_block_sizes_[phase]; ++inner)
                    {
                        update += arrow[phase](inner, row)
                                * base_solution_[phase][
                                      static_cast<std::size_t>(inner)];
                    }
                    reduced_global_rhs[static_cast<std::size_t>(row)] -=
                        update;
                }
            }
            if (!global_factor_.solve(
                    reduced_global_rhs, global_solution))
                return false;

            for (std::size_t phase = 0;
                 phase < phase_block_sizes_.size(); ++phase)
            {
                phase_solution[phase] = base_solution_[phase];
                for (Index row = 0;
                     row < phase_block_sizes_[phase]; ++row)
                {
                    Scalar update = 0.0;
                    for (Index column = 0;
                         column < number_of_global_variables_; ++column)
                    {
                        update += arrow_response_[phase](row, column)
                                * global_solution[
                                      static_cast<std::size_t>(column)];
                    }
                    phase_solution[phase][static_cast<std::size_t>(row)] +=
                        update;
                }
            }
            return true;
        }

        static void form_residual(
            const std::vector<MatRealView> &diagonal,
            const std::vector<MatRealView> &coupling,
            const std::vector<MatRealView> &arrow,
            const MatRealView &global,
            const std::vector<VecRealView> &rhs_phase,
            const VecRealView &rhs_global,
            const std::vector<std::vector<Scalar>> &phase_solution,
            const std::vector<Scalar> &global_solution,
            std::vector<std::vector<Scalar>> &phase_residual,
            std::vector<Scalar> &global_residual,
            Scalar &residual_norm,
            Scalar &rhs_norm)
        {
            residual_norm = 0.0;
            rhs_norm = 0.0;
            for (std::size_t phase = 0;
                 phase < phase_solution.size(); ++phase)
            {
                const Index size = diagonal[phase].m();
                for (Index row = 0; row < size; ++row)
                {
                    Scalar value = rhs_phase[phase](row);
                    rhs_norm = std::max(rhs_norm, std::abs(value));
                    for (Index column = 0; column < size; ++column)
                    {
                        value -= diagonal[phase](row, column)
                               * phase_solution[phase][
                                     static_cast<std::size_t>(column)];
                    }
                    if (phase > 0)
                    {
                        for (Index column = 0;
                             column < coupling[phase - 1].m(); ++column)
                        {
                            value -= coupling[phase - 1](column, row)
                                   * phase_solution[phase - 1][
                                         static_cast<std::size_t>(column)];
                        }
                    }
                    if (phase + 1 < phase_solution.size())
                    {
                        for (Index column = 0;
                             column < coupling[phase].n(); ++column)
                        {
                            value -= coupling[phase](row, column)
                                   * phase_solution[phase + 1][
                                         static_cast<std::size_t>(column)];
                        }
                    }
                    for (Index column = 0;
                         column < global.m(); ++column)
                    {
                        value -= arrow[phase](row, column)
                               * global_solution[
                                     static_cast<std::size_t>(column)];
                    }
                    phase_residual[phase][static_cast<std::size_t>(row)] =
                        value;
                    residual_norm =
                        std::max(residual_norm, std::abs(value));
                }
            }

            for (Index row = 0; row < global.m(); ++row)
            {
                Scalar value = rhs_global(row);
                rhs_norm = std::max(rhs_norm, std::abs(value));
                for (Index column = 0; column < global.n(); ++column)
                {
                    value -= global(row, column)
                           * global_solution[
                                 static_cast<std::size_t>(column)];
                }
                for (std::size_t phase = 0;
                     phase < phase_solution.size(); ++phase)
                {
                    for (Index inner = 0;
                         inner < arrow[phase].m(); ++inner)
                    {
                        value -= arrow[phase](inner, row)
                               * phase_solution[phase][
                                     static_cast<std::size_t>(inner)];
                    }
                }
                global_residual[static_cast<std::size_t>(row)] = value;
                residual_norm =
                    std::max(residual_norm, std::abs(value));
            }
        }

        static bool finite_solution(
            const std::vector<std::vector<Scalar>> &phase_solution,
            const std::vector<Scalar> &global_solution)
        {
            for (const auto &phase : phase_solution)
                for (const Scalar value : phase)
                    if (!std::isfinite(value)) return false;
            for (const Scalar value : global_solution)
                if (!std::isfinite(value)) return false;
            return true;
        }

        std::vector<Index> phase_block_sizes_;
        Index number_of_global_variables_ = 0;
        Scalar pivot_tolerance_ = 0.0;
        PhaseArrowKktFailureLocation last_failure_location_ =
            PhaseArrowKktFailureLocation::None;
        Index last_failure_phase_ = -1;

        std::vector<DenseLu> phase_factors_;
        std::vector<DenseMatrix> modified_diagonal_;
        std::vector<DenseMatrix> modified_arrow_;
        std::vector<DenseMatrix> arrow_response_;
        DenseMatrix global_schur_;
        DenseLu global_factor_;

        std::vector<std::vector<Scalar>> modified_rhs_;
        std::vector<std::vector<Scalar>> base_solution_;
        std::vector<std::vector<Scalar>> solution_;
        std::vector<Scalar> global_solution_;
        std::vector<std::vector<Scalar>> residual_;
        std::vector<Scalar> global_residual_;
        std::vector<std::vector<Scalar>> correction_;
        std::vector<Scalar> global_correction_;
    };

} // namespace fatrop

#endif // __fatrop_ocp_phase_arrow_kkt_solver_hpp__
