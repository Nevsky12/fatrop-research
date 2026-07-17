//
// Copyright (c) 2026
//

#ifndef FATROP_RESEARCH_INTERVAL_SCOPED_COLLOCATION_PROBLEM_HPP
#define FATROP_RESEARCH_INTERVAL_SCOPED_COLLOCATION_PROBLEM_HPP

#include "fatrop/context/context.hpp"
#include "fatrop/ocp/interval_scoped_kkt_solver.hpp"

#include <Eigen/Dense>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace fatrop::research
{
    struct IntervalScopedCollocationConfig
    {
        std::vector<Index> phase_nodes{5, 4, 6};
        std::vector<Index> phase_states{4, 7, 3};
        std::vector<Index> phase_controls{2, 3, 1};
        Index phase_parameter_dimension = 2;
        Index interface_parameter_dimension = 1;
        Index segment_parameter_dimension = 2;
        Index segment_width = 3;
        Index segment_stride = 3;
        Index global_parameter_dimension = 2;
        Index path_inequality_dimension = 0;
        Scalar inequality_lower_bound = -0.045;
        Scalar inequality_upper_bound = 0.045;
        Index collocation_degree = 3;
        Index max_iterations = 80;
        Index repeats = 3;
        Scalar tolerance = 1e-8;
        Scalar regularization = 1e-8;
        Scalar dual_regularization = 1e-5;
        Scalar fraction_to_boundary = 0.995;
        bool reuse_predictor_corrector_factorization = true;
        bool dense_step_validation = true;
        bool run_ipopt = false;
        std::string ipopt_linear_solver = "ma57";
        std::string ipopt_hsl_library;
    };

    enum class ScopedVariableKind
    {
        PhaseParameter,
        BoundarySeparator,
        InterfaceParameter,
        SegmentParameter,
        GlobalParameter
    };

    struct ScopedVariableBlock
    {
        IntervalScope scope;
        ScopedVariableKind kind = ScopedVariableKind::PhaseParameter;
        Index owner = -1;
    };

    struct ScopedNlpValues
    {
        Scalar objective = 0.0;
        Scalar constraint_inf = 0.0;
        Scalar dual_inf = 0.0;
    };

    struct ScopedNewtonDirection
    {
        Eigen::VectorXd primal;
        Eigen::VectorXd multipliers;
        Scalar residual_inf = 0.0;
        Scalar dense_difference = 0.0;
        double elapsed_ms = 0.0;
        double reduced_ms = 0.0;
        bool success = false;
    };

    struct ScopedDerivativeCheck
    {
        Scalar objective_error = 0.0;
        Scalar constraint_error = 0.0;
        Scalar lagrangian_hessian_error = 0.0;
    };

    struct ScopedSqpResult
    {
        Eigen::VectorXd primal;
        Eigen::VectorXd multipliers;
        Scalar objective = 0.0;
        Scalar constraint_inf = 0.0;
        Scalar dual_inf = 0.0;
        Scalar linear_residual = 0.0;
        Scalar dense_step_difference = 0.0;
        double elapsed_ms = 0.0;
        double kkt_ms = 0.0;
        Index iterations = 0;
        Index line_search_evaluations = 0;
        bool converged = false;
    };

    /** Result of the interval-scoped primal-dual predictor-corrector solve. */
    struct ScopedIpmResult
    {
        Eigen::VectorXd primal;
        Eigen::VectorXd multipliers;
        Eigen::VectorXd slack;
        Eigen::VectorXd lower_dual;
        Eigen::VectorXd upper_dual;
        Scalar objective = 0.0;
        Scalar constraint_inf = 0.0;
        Scalar dual_inf = 0.0;
        Scalar complementarity_inf = 0.0;
        Scalar linear_residual = 0.0;
        Scalar dense_step_difference = 0.0;
        double elapsed_ms = 0.0;
        double kkt_ms = 0.0;
        Index iterations = 0;
        Index line_search_evaluations = 0;
        Index factorizations = 0;
        Index retained_rhs_solves = 0;
        bool converged = false;
    };

    /**
     * Nonlinear heterogeneous multiphase Radau problem used to falsify the
     * interval-scoped Riccati hypothesis end to end.
     *
     * The public NLP ordering is `[y_0,...,y_(F-1),v]`, where each `y_f` is a
     * complete direct-collocation phase and every scoped block in `v` occurs
     * once.  A boundary separator `q_f` is shared by phases `f` and `f+1`:
     * the left phase imposes `q_f=R_f(x_end,v)`, while the right phase imposes
     * `x_start=q_f`.  Thus phase trajectory KKT blocks remain independent.
     *
     * The class owns both the analytic NLP evaluator used by IPOPT and the
     * structured SQP Newton assembly used by FATROP, ensuring that comparisons
     * cannot silently use different models or derivatives.
     */
    class IntervalScopedCollocationProblem
    {
    public:
        explicit IntervalScopedCollocationProblem(
            IntervalScopedCollocationConfig config);
        ~IntervalScopedCollocationProblem();

        IntervalScopedCollocationProblem(
            IntervalScopedCollocationProblem &&) noexcept;
        IntervalScopedCollocationProblem &operator=(
            IntervalScopedCollocationProblem &&) noexcept;

        IntervalScopedCollocationProblem(
            const IntervalScopedCollocationProblem &) = delete;
        IntervalScopedCollocationProblem &operator=(
            const IntervalScopedCollocationProblem &) = delete;

        const IntervalScopedCollocationConfig &config() const noexcept;
        Index number_of_phases() const noexcept;
        Index number_of_variables() const noexcept;
        Index number_of_constraints() const noexcept;
        Index number_of_equalities() const noexcept;
        Index number_of_inequalities() const noexcept;
        Index number_of_scoped_variables() const noexcept;
        Index number_of_trajectory_variables() const noexcept;
        const Eigen::VectorXd &constraint_lower_bounds() const noexcept;
        const Eigen::VectorXd &constraint_upper_bounds() const noexcept;
        const std::vector<ScopedVariableBlock> &blocks() const noexcept;
        const std::vector<IntervalScope> &scopes() const noexcept;
        IntervalKktSymbolicStats symbolic_stats() const noexcept;

        Eigen::VectorXd initial_primal() const;
        Eigen::VectorXd initial_multipliers() const;

        /**
         * Evaluate values, analytic first derivatives, and the exact
         * Lagrangian Hessian
         *
         *   objective_factor * Hessian(f)
         *     + sum_i multipliers(i) * Hessian(g_i).
         *
         * The objective and constraint values themselves are never scaled.
         */
        ScopedNlpValues evaluate(
            const Eigen::VectorXd &primal,
            const Eigen::VectorXd &multipliers,
            Scalar objective_factor = 1.0);

        Scalar objective() const noexcept;
        const Eigen::VectorXd &objective_gradient() const noexcept;
        const Eigen::VectorXd &constraints() const noexcept;
        const Eigen::VectorXd &lagrangian_gradient() const noexcept;

        /** Fixed C-style sparse patterns and values from the last evaluation. */
        const std::vector<std::pair<Index, Index>> &
        jacobian_pattern() const noexcept;
        const std::vector<Scalar> &jacobian_values() const noexcept;
        const std::vector<std::pair<Index, Index>> &
        hessian_pattern() const noexcept;
        const std::vector<Scalar> &hessian_values() const noexcept;

        ScopedNewtonDirection solve_direction(
            const Eigen::VectorXd &primal,
            const Eigen::VectorXd &multipliers,
            bool validate_with_dense);
        ScopedDerivativeCheck check_derivatives(
            const Eigen::VectorXd &primal,
            Scalar step = 1e-6);
        ScopedSqpResult solve_sqp();
        ScopedIpmResult solve_ipm();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace fatrop::research

#endif
