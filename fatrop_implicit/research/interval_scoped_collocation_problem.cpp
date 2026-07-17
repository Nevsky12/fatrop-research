//
// Copyright (c) 2026
//

#include "interval_scoped_collocation_problem.hpp"

#include "fatrop/ocp/interval_scoped_ocp_kkt_solver.hpp"
#include "fatrop/ocp/interval_scoped_pd_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace fatrop::research
{
    namespace
    {
        using Clock = std::chrono::steady_clock;

        Scalar max_abs(const Eigen::VectorXd &value)
        {
            return value.size() == 0
                ? 0.0 : value.cwiseAbs().maxCoeff();
        }

        Scalar sech_squared(const Scalar value)
        {
            const Scalar cosine = std::cosh(value);
            return 1.0 / (cosine * cosine);
        }

        Scalar deterministic_coefficient(
            const Index first,
            const Index second,
            const Index phase,
            const Scalar scale)
        {
            return scale * std::sin(
                0.17 * static_cast<Scalar>(first + 1)
              + 0.29 * static_cast<Scalar>(second + 1)
              + 0.11 * static_cast<Scalar>(phase + 1));
        }

        struct RadauCoefficients
        {
            explicit RadauCoefficients(const Index degree)
                : tau(static_cast<std::size_t>(degree + 1)),
                  C(degree + 1, degree + 1),
                  D(degree + 1)
            {
                tau[0] = 0.0;
                if (degree == 1)
                    tau[1] = 1.0;
                else if (degree == 2)
                {
                    tau[1] = 1.0 / 3.0;
                    tau[2] = 1.0;
                }
                else if (degree == 3)
                {
                    tau[1] = 0.15505102572168219018;
                    tau[2] = 0.64494897427831780982;
                    tau[3] = 1.0;
                }
                else
                    throw std::runtime_error(
                        "Radau degree must be 1, 2, or 3");

                C.setZero();
                D.setZero();
                for (Index basis = 0; basis <= degree; ++basis)
                {
                    std::vector<Scalar> polynomial{1.0};
                    for (Index other = 0; other <= degree; ++other)
                    {
                        if (other == basis)
                            continue;
                        const Scalar denominator =
                            tau[static_cast<std::size_t>(basis)]
                          - tau[static_cast<std::size_t>(other)];
                        std::vector<Scalar> product(
                            polynomial.size() + 1, 0.0);
                        for (std::size_t power = 0;
                             power < polynomial.size(); ++power)
                        {
                            product[power] -= polynomial[power]
                                * tau[static_cast<std::size_t>(other)]
                                / denominator;
                            product[power + 1] +=
                                polynomial[power] / denominator;
                        }
                        polynomial = std::move(product);
                    }
                    for (Index point = 0; point <= degree; ++point)
                    {
                        const Scalar evaluation =
                            tau[static_cast<std::size_t>(point)];
                        Scalar derivative = 0.0;
                        for (std::size_t power = 1;
                             power < polynomial.size(); ++power)
                        {
                            derivative += static_cast<Scalar>(power)
                                * polynomial[power]
                                * std::pow(
                                    evaluation,
                                    static_cast<int>(power - 1));
                        }
                        C(basis, point) = derivative;
                    }
                    for (const Scalar coefficient : polynomial)
                        D(basis) += coefficient;
                }
            }

            std::vector<Scalar> tau;
            Eigen::MatrixXd C;
            Eigen::VectorXd D;
        };

        ProblemDims make_phase_dims(
            const IntervalScopedCollocationConfig &config,
            const Index phase,
            const Index incident_border_dimension)
        {
            const Index phases = static_cast<Index>(
                config.phase_nodes.size());
            const Index nodes = config.phase_nodes[
                static_cast<std::size_t>(phase)];
            const Index nx = config.phase_states[
                static_cast<std::size_t>(phase)];
            const Index nu = config.phase_controls[
                static_cast<std::size_t>(phase)];
            std::vector<Index> controls(
                static_cast<std::size_t>(nodes), 0);
            std::vector<Index> states(
                static_cast<std::size_t>(nodes), nx);
            std::vector<Index> equalities(
                static_cast<std::size_t>(nodes), 0);
            std::vector<Index> inequalities(
                static_cast<std::size_t>(nodes),
                config.path_inequality_dimension);
            std::vector<Index> border(
                static_cast<std::size_t>(nodes),
                incident_border_dimension);
            for (Index stage = 0; stage + 1 < nodes; ++stage)
            {
                controls[static_cast<std::size_t>(stage)] =
                    nu + config.collocation_degree * nx;
                equalities[static_cast<std::size_t>(stage)] =
                    config.collocation_degree * nx
                  + (stage == 0 ? nx : 0);
            }
            if (phase + 1 < phases)
            {
                equalities.back() = config.phase_states[
                    static_cast<std::size_t>(phase + 1)];
            }
            return ProblemDims(
                nodes, std::move(controls), std::move(states),
                std::move(equalities), std::move(inequalities), 0,
                std::move(border));
        }

        struct PhaseData
        {
            PhaseData(
                const IntervalScopedCollocationConfig &config,
                const Index phase_value,
                std::vector<Index> global_columns_value,
                const Index total_scoped_variables,
                const Index left_separator_block_value,
                const Index right_separator_block_value,
                const std::vector<Index> &block_offsets)
                : phase(phase_value),
                  nx(config.phase_states[
                      static_cast<std::size_t>(phase)]),
                  nu(config.phase_controls[
                      static_cast<std::size_t>(phase)]),
                  nodes(config.phase_nodes[
                      static_cast<std::size_t>(phase)]),
                  degree(config.collocation_degree),
                  global_columns(std::move(global_columns_value)),
                  global_to_local(
                      static_cast<std::size_t>(total_scoped_variables), -1),
                  left_separator_block(left_separator_block_value),
                  right_separator_block(right_separator_block_value),
                  dims(make_phase_dims(
                      config, phase,
                      static_cast<Index>(global_columns.size()))),
                  info(dims),
                  jacobian(dims),
                  hessian(dims),
                  D_x(info.number_of_primal_variables),
                  D_eq(info.number_of_g_eq_path),
                  D_s(info.number_of_slack_variables),
                  pd_D_x(
                      info.number_of_primal_variables
                      + info.number_of_slack_variables),
                  pd_D_e(info.number_of_eq_constraints),
                  slack_lower_distance(info.number_of_slack_variables),
                  slack_upper_distance(info.number_of_slack_variables),
                  slack_lower_dual(info.number_of_slack_variables),
                  slack_upper_dual(info.number_of_slack_variables),
                  rhs_primal(info.number_of_primal_variables),
                  rhs_slack(info.number_of_slack_variables),
                  rhs_constraints(info.number_of_eq_constraints),
                  rhs_lower_complementarity(
                      info.number_of_slack_variables),
                  rhs_upper_complementarity(
                      info.number_of_slack_variables),
                  cross_hessian(
                      info.number_of_primal_variables,
                      static_cast<Index>(global_columns.size())),
                  border_jacobian(
                      info.number_of_eq_constraints,
                      static_cast<Index>(global_columns.size())),
                  primal_step(info.number_of_primal_variables),
                  multiplier_step(info.number_of_eq_constraints),
                  pd_solution(
                      info.pd_orig_offset_zu
                      + info.number_of_slack_variables),
                  objective_gradient_y(
                      Eigen::VectorXd::Zero(
                          info.number_of_primal_variables)),
                  objective_gradient_v(
                      Eigen::VectorXd::Zero(
                          static_cast<Eigen::Index>(global_columns.size()))),
                  lagrangian_gradient_y(objective_gradient_y),
                  lagrangian_gradient_v(objective_gradient_v),
                  constraints(Eigen::VectorXd::Zero(
                      info.number_of_eq_constraints)),
                  jacobian_y(Eigen::MatrixXd::Zero(
                      info.number_of_eq_constraints,
                      info.number_of_primal_variables)),
                  jacobian_v(Eigen::MatrixXd::Zero(
                      info.number_of_eq_constraints,
                      static_cast<Eigen::Index>(global_columns.size()))),
                  hessian_y(Eigen::MatrixXd::Zero(
                      info.number_of_primal_variables,
                      info.number_of_primal_variables)),
                  hessian_yv(Eigen::MatrixXd::Zero(
                      info.number_of_primal_variables,
                      static_cast<Eigen::Index>(global_columns.size()))),
                  hessian_vv(Eigen::MatrixXd::Zero(
                      static_cast<Eigen::Index>(global_columns.size()),
                      static_cast<Eigen::Index>(global_columns.size())))
            {
                for (Index local = 0;
                     local < static_cast<Index>(global_columns.size());
                     ++local)
                {
                    global_to_local[static_cast<std::size_t>(
                        global_columns[static_cast<std::size_t>(local)])] =
                        local;
                }
                if (left_separator_block >= 0)
                    left_separator_offset = block_offsets[
                        static_cast<std::size_t>(left_separator_block)];
                if (right_separator_block >= 0)
                    right_separator_offset = block_offsets[
                        static_cast<std::size_t>(right_separator_block)];
                D_x = config.regularization;
                D_eq = config.dual_regularization;
                D_s = 0.0;
                pd_D_x = 0.0;
                pd_D_x.block(
                    info.number_of_primal_variables,
                    info.offset_primal) = config.regularization;
                pd_D_e = 0.0;
                pd_D_e.block(
                    info.number_of_g_eq_path,
                    info.offset_g_eq_path) = config.dual_regularization;
                slack_lower_distance = 1.0;
                slack_upper_distance = 1.0;
                slack_lower_dual = 1.0;
                slack_upper_dual = 1.0;
                primal_step = 0.0;
                multiplier_step = 0.0;
                rhs_slack = 0.0;
                rhs_lower_complementarity = 0.0;
                rhs_upper_complementarity = 0.0;
                pd_solution = 0.0;
            }

            Index local_static(const Index global) const
            {
                if (global < 0
                    || global >= static_cast<Index>(global_to_local.size()))
                    return -1;
                return global_to_local[static_cast<std::size_t>(global)];
            }

            Index phase = 0;
            Index nx = 0;
            Index nu = 0;
            Index nodes = 0;
            Index degree = 0;
            std::vector<Index> global_columns;
            std::vector<Index> global_to_local;
            Index left_separator_block = -1;
            Index right_separator_block = -1;
            Index left_separator_offset = -1;
            Index right_separator_offset = -1;
            Index global_slack_offset = 0;

            ProblemDims dims;
            ProblemInfo info;
            Jacobian<ImplicitOcpType> jacobian;
            Hessian<ImplicitOcpType> hessian;
            VecRealAllocated D_x;
            VecRealAllocated D_eq;
            VecRealAllocated D_s;
            VecRealAllocated pd_D_x;
            VecRealAllocated pd_D_e;
            VecRealAllocated slack_lower_distance;
            VecRealAllocated slack_upper_distance;
            VecRealAllocated slack_lower_dual;
            VecRealAllocated slack_upper_dual;
            VecRealAllocated rhs_primal;
            VecRealAllocated rhs_slack;
            VecRealAllocated rhs_constraints;
            VecRealAllocated rhs_lower_complementarity;
            VecRealAllocated rhs_upper_complementarity;
            MatRealAllocated cross_hessian;
            MatRealAllocated border_jacobian;
            VecRealAllocated primal_step;
            VecRealAllocated multiplier_step;
            VecRealAllocated pd_solution;

            Scalar objective = 0.0;
            Eigen::VectorXd objective_gradient_y;
            Eigen::VectorXd objective_gradient_v;
            Eigen::VectorXd lagrangian_gradient_y;
            Eigen::VectorXd lagrangian_gradient_v;
            Eigen::VectorXd constraints;
            Eigen::MatrixXd jacobian_y;
            Eigen::MatrixXd jacobian_v;
            Eigen::MatrixXd hessian_y;
            Eigen::MatrixXd hessian_yv;
            Eigen::MatrixXd hessian_vv;
        };

        struct JacobianSource
        {
            Index phase = 0;
            Index row = 0;
            Index column = 0;
            bool scoped_column = false;
        };

        enum class HessianSourceKind
        {
            Phase,
            Cross,
            Scoped
        };

        struct HessianSource
        {
            HessianSourceKind kind = HessianSourceKind::Phase;
            Index phase = 0;
            Index row = 0;
            Index column = 0;
        };

        struct ScopedPdState
        {
            Eigen::VectorXd primal;
            Eigen::VectorXd multipliers;
            Eigen::VectorXd slack;
            Eigen::VectorXd lower_dual;
            Eigen::VectorXd upper_dual;
        };

        struct ScopedPdDirection
        {
            Eigen::VectorXd primal;
            Eigen::VectorXd multipliers;
            Eigen::VectorXd slack;
            Eigen::VectorXd lower_dual;
            Eigen::VectorXd upper_dual;
            Scalar residual_inf = 0.0;
            Scalar dense_difference =
                std::numeric_limits<Scalar>::quiet_NaN();
            double elapsed_ms = 0.0;
            double reduced_ms = 0.0;
            bool success = false;
        };

        struct ScopedPdMetrics
        {
            Scalar primal_inf = 0.0;
            Scalar dual_inf = 0.0;
            Scalar complementarity_inf = 0.0;
            Scalar residual_inf = 0.0;
            Scalar average_complementarity = 0.0;
            bool finite = true;
        };
    } // namespace

    struct IntervalScopedCollocationProblem::Impl
    {
        explicit Impl(IntervalScopedCollocationConfig config_value)
            : config(std::move(config_value)),
              coefficients(config.collocation_degree)
        {
            validate_config();
            build_scopes();
            build_phases();
            build_solver();
            primal_initial = make_initial_primal();
            multiplier_initial = Eigen::VectorXd::Zero(total_constraints);
            evaluate(primal_initial, multiplier_initial);
            build_sparse_patterns();
            refresh_sparse_values();
        }

        void validate_config() const
        {
            const std::size_t phases = config.phase_nodes.size();
            if (phases == 0
                || config.phase_states.size() != phases
                || config.phase_controls.size() != phases)
                throw std::runtime_error(
                    "Phase node/state/control lists must have equal nonzero size");
            for (std::size_t phase = 0; phase < phases; ++phase)
            {
                if (config.phase_nodes[phase] < 2
                    || config.phase_states[phase] < 1
                    || config.phase_controls[phase] < 0)
                    throw std::runtime_error(
                        "Each phase needs nodes>=2, nx>=1, and nu>=0");
            }
            if (config.collocation_degree < 1
                || config.collocation_degree > 3
                || config.phase_parameter_dimension < 0
                || config.interface_parameter_dimension < 0
                || config.segment_parameter_dimension < 0
                || config.segment_width < 1
                || config.segment_stride < 1
                || config.global_parameter_dimension < 0
                || config.path_inequality_dimension < 0
                || !(config.inequality_lower_bound
                     < config.inequality_upper_bound)
                || config.max_iterations < 1 || config.repeats < 1
                || config.tolerance <= 0.0
                || config.regularization < 0.0
                || config.dual_regularization <= 0.0
                || config.fraction_to_boundary <= 0.0
                || config.fraction_to_boundary >= 1.0)
                throw std::runtime_error(
                    "Invalid scoped collocation configuration");
        }

        Index append_block(
            const IntervalScope scope,
            const ScopedVariableKind kind,
            const Index owner)
        {
            const Index index = static_cast<Index>(blocks.size());
            blocks.push_back({scope, kind, owner});
            scopes.push_back(scope);
            return index;
        }

        void build_scopes()
        {
            const Index phases = static_cast<Index>(
                config.phase_nodes.size());
            phase_parameter_blocks.assign(
                static_cast<std::size_t>(phases), -1);
            left_separator_blocks.assign(
                static_cast<std::size_t>(phases), -1);
            right_separator_blocks.assign(
                static_cast<std::size_t>(phases), -1);

            if (config.phase_parameter_dimension > 0)
            {
                for (Index phase = 0; phase < phases; ++phase)
                {
                    phase_parameter_blocks[static_cast<std::size_t>(phase)] =
                        append_block(
                            {config.phase_parameter_dimension, phase, phase},
                            ScopedVariableKind::PhaseParameter,
                            phase);
                }
            }
            for (Index phase = 0; phase + 1 < phases; ++phase)
            {
                const Index separator = append_block(
                    {config.phase_states[static_cast<std::size_t>(phase + 1)],
                     phase, phase + 1},
                    ScopedVariableKind::BoundarySeparator,
                    phase);
                right_separator_blocks[static_cast<std::size_t>(phase)] =
                    separator;
                left_separator_blocks[static_cast<std::size_t>(phase + 1)] =
                    separator;
            }
            if (config.interface_parameter_dimension > 0)
            {
                for (Index phase = 0; phase + 1 < phases; ++phase)
                    append_block(
                        {config.interface_parameter_dimension,
                         phase, phase + 1},
                        ScopedVariableKind::InterfaceParameter,
                        phase);
            }
            if (config.segment_parameter_dimension > 0)
            {
                for (Index first = 0; first < phases;
                     first += config.segment_stride)
                {
                    append_block(
                        {config.segment_parameter_dimension,
                         first,
                         std::min(
                             phases - 1,
                             first + config.segment_width - 1)},
                        ScopedVariableKind::SegmentParameter,
                        first);
                }
            }
            if (config.global_parameter_dimension > 0)
                append_block(
                    {config.global_parameter_dimension, 0, phases - 1},
                    ScopedVariableKind::GlobalParameter, 0);
            if (blocks.empty())
                throw std::runtime_error(
                    "The scoped collocation problem has no scoped variables");

            block_offsets.resize(blocks.size(), 0);
            for (std::size_t block = 1; block < blocks.size(); ++block)
                block_offsets[block] = block_offsets[block - 1]
                    + scopes[block - 1].dimension;
            total_scoped = 0;
            for (const IntervalScope &scope : scopes)
                total_scoped += scope.dimension;
            scalar_to_block.resize(
                static_cast<std::size_t>(total_scoped), -1);
            scalar_to_local.resize(
                static_cast<std::size_t>(total_scoped), -1);
            for (Index block = 0;
                 block < static_cast<Index>(blocks.size()); ++block)
            {
                for (Index local = 0;
                     local < scopes[static_cast<std::size_t>(block)].dimension;
                     ++local)
                {
                    const Index scalar = block_offsets[
                        static_cast<std::size_t>(block)] + local;
                    scalar_to_block[static_cast<std::size_t>(scalar)] = block;
                    scalar_to_local[static_cast<std::size_t>(scalar)] = local;
                }
            }

            phase_active_blocks.resize(static_cast<std::size_t>(phases));
            phase_columns.resize(static_cast<std::size_t>(phases));
            for (Index phase = 0; phase < phases; ++phase)
            {
                for (Index block = 0;
                     block < static_cast<Index>(scopes.size()); ++block)
                {
                    const IntervalScope &scope = scopes[
                        static_cast<std::size_t>(block)];
                    if (scope.first_phase <= phase
                        && phase <= scope.last_phase)
                    {
                        phase_active_blocks[static_cast<std::size_t>(phase)]
                            .push_back(block);
                        for (Index local = 0; local < scope.dimension; ++local)
                            phase_columns[static_cast<std::size_t>(phase)]
                                .push_back(block_offsets[
                                    static_cast<std::size_t>(block)] + local);
                    }
                }
            }
        }

        void build_phases()
        {
            const Index phase_count = static_cast<Index>(
                config.phase_nodes.size());
            phase_variable_offsets.resize(
                static_cast<std::size_t>(phase_count), 0);
            phase_constraint_offsets.resize(
                static_cast<std::size_t>(phase_count), 0);
            phases.reserve(static_cast<std::size_t>(phase_count));
            total_trajectory = 0;
            total_constraints = 0;
            total_inequalities = 0;
            for (Index phase = 0; phase < phase_count; ++phase)
            {
                phase_variable_offsets[static_cast<std::size_t>(phase)] =
                    total_trajectory;
                phase_constraint_offsets[static_cast<std::size_t>(phase)] =
                    total_constraints;
                phases.push_back(std::make_unique<PhaseData>(
                    config, phase,
                    phase_columns[static_cast<std::size_t>(phase)],
                    total_scoped,
                    left_separator_blocks[static_cast<std::size_t>(phase)],
                    right_separator_blocks[static_cast<std::size_t>(phase)],
                    block_offsets));
                phases.back()->global_slack_offset = total_inequalities;
                total_trajectory +=
                    phases.back()->info.number_of_primal_variables;
                total_constraints +=
                    phases.back()->info.number_of_eq_constraints;
                total_inequalities +=
                    phases.back()->info.number_of_slack_variables;
            }
            total_equalities = total_constraints - total_inequalities;
            total_variables = total_trajectory + total_scoped;
            objective_gradient_global = Eigen::VectorXd::Zero(total_variables);
            constraints_global = Eigen::VectorXd::Zero(total_constraints);
            lagrangian_gradient_global =
                Eigen::VectorXd::Zero(total_variables);
            scoped_objective_gradient = Eigen::VectorXd::Zero(total_scoped);
            scoped_lagrangian_gradient = Eigen::VectorXd::Zero(total_scoped);
            scoped_hessian = Eigen::MatrixXd::Zero(
                total_scoped, total_scoped);
            constraint_lower = Eigen::VectorXd::Zero(total_constraints);
            constraint_upper = Eigen::VectorXd::Zero(total_constraints);
            for (std::size_t phase_index = 0;
                 phase_index < phases.size(); ++phase_index)
            {
                const PhaseData &phase = *phases[phase_index];
                const Index constraint_offset =
                    phase_constraint_offsets[phase_index];
                for (Index stage = 0; stage < phase.nodes; ++stage)
                for (Index row = 0;
                     row < phase.dims.number_of_ineq_constraints[stage];
                     ++row)
                {
                    const Index global = constraint_offset
                        + phase.info.offsets_g_eq_slack[stage] + row;
                    constraint_lower(global) =
                        config.inequality_lower_bound;
                    constraint_upper(global) =
                        config.inequality_upper_bound;
                }
            }
        }

        void build_solver()
        {
            std::vector<const ProblemInfo *> phase_info;
            phase_info.reserve(phases.size());
            for (const auto &phase : phases)
                phase_info.push_back(&phase->info);
            solver = std::make_unique<IntervalScopedImplicitOcpKktSolver>(
                phase_info, scopes);
            pd_solver = std::make_unique<IntervalScopedImplicitPdSolver>(
                phase_info, scopes);
            solver->reduced_solver().set_pivot_tolerance(1e-14);
            pd_solver->phase_solver().reduced_solver()
                .set_pivot_tolerance(1e-14);
            for (Index phase = 0;
                 phase < static_cast<Index>(phases.size()); ++phase)
            {
                solver->trajectory_solver(phase).set_lu_fact_tol(1e-12);
                pd_solver->phase_solver().trajectory_solver(phase)
                    .set_lu_fact_tol(1e-12);
            }

            diagonal_storage.reserve(scopes.size());
            rhs_storage.reserve(scopes.size());
            scoped_solution_storage.reserve(scopes.size());
            for (const IntervalScope &scope : scopes)
            {
                diagonal_storage.emplace_back(
                    scope.dimension, scope.dimension);
                rhs_storage.emplace_back(scope.dimension);
                scoped_solution_storage.emplace_back(scope.dimension);
            }
            const auto &edges = solver->reduced_solver().edges();
            edge_storage.reserve(edges.size());
            for (const IntervalKktEdge edge : edges)
            {
                edge_storage.emplace_back(
                    scopes[static_cast<std::size_t>(
                        edge.first_block)].dimension,
                    scopes[static_cast<std::size_t>(
                        edge.second_block)].dimension);
            }

            for (std::size_t block = 0; block < scopes.size(); ++block)
            {
                diagonal_views.push_back(diagonal_storage[block].block(
                    scopes[block].dimension,
                    scopes[block].dimension, 0, 0));
                rhs_views.push_back(rhs_storage[block].block(
                    scopes[block].dimension, 0));
                scoped_solution_views.push_back(
                    scoped_solution_storage[block].block(
                        scopes[block].dimension, 0));
            }
            for (std::size_t edge = 0; edge < edges.size(); ++edge)
            {
                edge_views.push_back(edge_storage[edge].block(
                    scopes[static_cast<std::size_t>(
                        edges[edge].first_block)].dimension,
                    scopes[static_cast<std::size_t>(
                        edges[edge].second_block)].dimension,
                    0, 0));
            }

            phase_views.reserve(phases.size());
            pd_phase_views.reserve(phases.size());
            for (auto &phase : phases)
            {
                phase_views.emplace_back(
                    phase->info,
                    phase->jacobian,
                    phase->hessian,
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
                    phase->primal_step.block(phase->primal_step.m(), 0),
                    phase->multiplier_step.block(
                        phase->multiplier_step.m(), 0),
                    false);
                pd_phase_views.emplace_back(
                    phase->info,
                    phase->jacobian,
                    phase->hessian,
                    phase->pd_D_x.block(phase->pd_D_x.m(), 0),
                    false,
                    phase->pd_D_e.block(phase->pd_D_e.m(), 0),
                    phase->slack_lower_distance.block(
                        phase->slack_lower_distance.m(), 0),
                    phase->slack_upper_distance.block(
                        phase->slack_upper_distance.m(), 0),
                    phase->slack_lower_dual.block(
                        phase->slack_lower_dual.m(), 0),
                    phase->slack_upper_dual.block(
                        phase->slack_upper_dual.m(), 0),
                    phase->cross_hessian.block(
                        phase->cross_hessian.m(),
                        phase->cross_hessian.n(), 0, 0),
                    phase->border_jacobian.block(
                        phase->border_jacobian.m(),
                        phase->border_jacobian.n(), 0, 0),
                    phase->rhs_primal.block(phase->rhs_primal.m(), 0),
                    phase->rhs_slack.block(phase->rhs_slack.m(), 0),
                    phase->rhs_constraints.block(
                        phase->rhs_constraints.m(), 0),
                    phase->rhs_lower_complementarity.block(
                        phase->rhs_lower_complementarity.m(), 0),
                    phase->rhs_upper_complementarity.block(
                        phase->rhs_upper_complementarity.m(), 0),
                    phase->pd_solution.block(phase->pd_solution.m(), 0));
            }
        }

        Eigen::VectorXd make_initial_primal() const
        {
            Eigen::VectorXd result = Eigen::VectorXd::Zero(total_variables);
            for (std::size_t phase_index = 0;
                 phase_index < phases.size(); ++phase_index)
            {
                const PhaseData &phase = *phases[phase_index];
                const Index global_offset = phase_variable_offsets[phase_index];
                for (Index stage = 0; stage < phase.nodes; ++stage)
                {
                    const Index state_offset =
                        global_offset + phase.info.offsets_primal_x[stage];
                    for (Index state = 0; state < phase.nx; ++state)
                    {
                        result(state_offset + state) =
                            0.025 * std::sin(
                                0.19 * static_cast<Scalar>(
                                    1 + phase.phase + stage + state));
                    }
                    if (stage + 1 == phase.nodes)
                        continue;
                    const Index control_offset =
                        global_offset + phase.info.offsets_primal_u[stage];
                    for (Index control = 0; control < phase.nu; ++control)
                        result(control_offset + control) =
                            0.015 * std::cos(
                                0.23 * static_cast<Scalar>(
                                    1 + phase.phase + stage + control));
                    for (Index point = 1; point <= phase.degree; ++point)
                    for (Index state = 0; state < phase.nx; ++state)
                    {
                        result(
                            control_offset + phase.nu
                            + (point - 1) * phase.nx + state) =
                            result(state_offset + state);
                    }
                }
            }
            for (Index scalar = 0; scalar < total_scoped; ++scalar)
            {
                result(total_trajectory + scalar) =
                    0.01 * std::sin(
                        0.13 * static_cast<Scalar>(scalar + 1));
            }
            return result;
        }

        void add_state_objective(
            PhaseData &phase,
            const Eigen::VectorXd &y,
            const Eigen::VectorXd &local_v,
            const Index stage,
            const Index state)
        {
            const Index y_index = phase.info.offsets_primal_x[stage] + state;
            const Scalar target = 0.08 * std::sin(
                0.21 * static_cast<Scalar>(
                    1 + phase.phase + 2 * stage + state));
            const Scalar weight = 1.0 + 0.05 * phase.phase
                + 0.02 * stage;
            Scalar residual = y(y_index) - target;
            std::vector<Scalar> coupling(
                phase.global_columns.size(), 0.0);
            for (Index column = 0;
                 column < static_cast<Index>(phase.global_columns.size());
                 ++column)
            {
                coupling[static_cast<std::size_t>(column)] =
                    deterministic_coefficient(
                        state + stage,
                        phase.global_columns[static_cast<std::size_t>(column)],
                        phase.phase, 0.018);
                residual -= coupling[static_cast<std::size_t>(column)]
                          * local_v(column);
            }
            phase.objective += 0.5 * weight * residual * residual;
            phase.objective_gradient_y(y_index) += weight * residual;
            phase.hessian_y(y_index, y_index) += weight;
            for (Index column = 0;
                 column < static_cast<Index>(phase.global_columns.size());
                 ++column)
            {
                const Scalar coefficient =
                    coupling[static_cast<std::size_t>(column)];
                phase.objective_gradient_v(column) -=
                    weight * coefficient * residual;
                phase.hessian_yv(y_index, column) -=
                    weight * coefficient;
                for (Index other = 0;
                     other < static_cast<Index>(phase.global_columns.size());
                     ++other)
                {
                    phase.hessian_vv(column, other) +=
                        weight * coefficient
                        * coupling[static_cast<std::size_t>(other)];
                }
            }
        }

        void evaluate_phase(
            PhaseData &phase,
            const Eigen::VectorXd &y,
            const Eigen::VectorXd &local_v,
            const Eigen::VectorXd &lambda,
            const Scalar objective_factor)
        {
            phase.objective = 0.0;
            phase.objective_gradient_y.setZero();
            phase.objective_gradient_v.setZero();
            phase.constraints.setZero();
            phase.jacobian_y.setZero();
            phase.jacobian_v.setZero();
            phase.hessian_y.setZero();
            phase.hessian_yv.setZero();
            phase.hessian_vv.setZero();

            for (Index stage = 0; stage < phase.nodes; ++stage)
            {
                for (Index state = 0; state < phase.nx; ++state)
                    add_state_objective(
                        phase, y, local_v, stage, state);
                if (stage + 1 == phase.nodes)
                    continue;
                const Index local_offset = phase.info.offsets_primal_u[stage];
                const Index state_offset = phase.info.offsets_primal_x[stage];
                for (Index control = 0; control < phase.nu; ++control)
                {
                    const Index index = local_offset + control;
                    const Scalar target = 0.035 * std::cos(
                        0.16 * static_cast<Scalar>(
                            1 + phase.phase + stage + control));
                    const Scalar residual = y(index) - target;
                    const Scalar weight = 0.35 + 0.02 * phase.phase;
                    phase.objective += 0.5 * weight * residual * residual;
                    phase.objective_gradient_y(index) += weight * residual;
                    phase.hessian_y(index, index) += weight;
                }
                for (Index point = 1; point <= phase.degree; ++point)
                for (Index state = 0; state < phase.nx; ++state)
                {
                    const Index collocation = local_offset + phase.nu
                        + (point - 1) * phase.nx + state;
                    const Index state_index = state_offset + state;
                    const Scalar residual =
                        y(collocation) - y(state_index);
                    constexpr Scalar weight = 0.12;
                    phase.objective += 0.5 * weight * residual * residual;
                    phase.objective_gradient_y(collocation) +=
                        weight * residual;
                    phase.objective_gradient_y(state_index) -=
                        weight * residual;
                    phase.hessian_y(collocation, collocation) += weight;
                    phase.hessian_y(state_index, state_index) += weight;
                    phase.hessian_y(collocation, state_index) -= weight;
                    phase.hessian_y(state_index, collocation) -= weight;
                }
            }

            const Scalar static_weight = 0.025
                / std::max<Scalar>(1.0, phase.nodes);
            for (Index column = 0;
                 column < static_cast<Index>(phase.global_columns.size());
                 ++column)
            {
                const Index global = phase.global_columns[
                    static_cast<std::size_t>(column)];
                const Scalar target = 0.03 * std::cos(
                    0.09 * static_cast<Scalar>(global + 1));
                const Scalar residual = local_v(column) - target;
                phase.objective +=
                    0.5 * static_weight * residual * residual;
                phase.objective_gradient_v(column) +=
                    static_weight * residual;
                phase.hessian_vv(column, column) += static_weight;
            }

            // Everything accumulated above is the exact Hessian of the
            // quadratic objective.  Scale it before adding the nonlinear
            // constraint contribution so the same evaluator obeys IPOPT's
            // objective_factor contract and FATROP can use objective_factor=1.
            phase.hessian_y *= objective_factor;
            phase.hessian_yv *= objective_factor;
            phase.hessian_vv *= objective_factor;

            const Scalar step_size =
                (0.65 + 0.04 * (phase.phase % 5))
                / static_cast<Scalar>(phase.nodes - 1);
            for (Index stage = 0; stage + 1 < phase.nodes; ++stage)
            {
                const Index local_offset = phase.info.offsets_primal_u[stage];
                const Index state_offset = phase.info.offsets_primal_x[stage];
                const Index next_offset = phase.info.offsets_primal_x[stage + 1];
                const Index path_offset = phase.info.offsets_g_eq_path[stage];
                const Index dynamics_offset = phase.info.offsets_g_eq_dyn[stage];
                for (Index point = 1; point <= phase.degree; ++point)
                for (Index state = 0; state < phase.nx; ++state)
                {
                    const Index equation = path_offset
                        + (point - 1) * phase.nx + state;
                    Scalar value = coefficients.C(0, point)
                        * y(state_offset + state);
                    phase.jacobian_y(
                        equation, state_offset + state) +=
                        coefficients.C(0, point);
                    for (Index basis = 1;
                         basis <= phase.degree; ++basis)
                    {
                        const Index collocation = local_offset + phase.nu
                            + (basis - 1) * phase.nx + state;
                        value += coefficients.C(basis, point)
                               * y(collocation);
                        phase.jacobian_y(equation, collocation) +=
                            coefficients.C(basis, point);
                    }
                    const Index current_collocation = local_offset + phase.nu
                        + (point - 1) * phase.nx + state;
                    const Scalar state_value = y(current_collocation);
                    const Scalar state_coefficient =
                        -0.28 - 0.015 * (state + phase.phase);
                    const Scalar state_sech_squared =
                        sech_squared(state_value);
                    Scalar dynamics =
                        state_coefficient * std::tanh(state_value);
                    phase.jacobian_y(equation, current_collocation) -=
                        step_size * state_coefficient
                        * state_sech_squared;
                    phase.hessian_y(
                        current_collocation, current_collocation) +=
                        lambda(equation) * 2.0 * step_size
                        * state_coefficient * std::tanh(state_value)
                        * state_sech_squared;
                    for (Index control = 0; control < phase.nu; ++control)
                    {
                        const Scalar coefficient = deterministic_coefficient(
                            state, control, phase.phase, 0.12);
                        dynamics += coefficient
                                  * y(local_offset + control);
                        phase.jacobian_y(
                            equation, local_offset + control) -=
                            step_size * coefficient;
                    }
                    for (Index column = 0;
                         column < static_cast<Index>(
                             phase.global_columns.size()); ++column)
                    {
                        const Scalar coefficient = deterministic_coefficient(
                            state + point,
                            phase.global_columns[
                                static_cast<std::size_t>(column)],
                            phase.phase, 0.035);
                        dynamics += coefficient * std::sin(local_v(column));
                        phase.jacobian_v(equation, column) -=
                            step_size * coefficient * std::cos(local_v(column));
                        phase.hessian_vv(column, column) +=
                            lambda(equation) * step_size * coefficient
                            * std::sin(local_v(column));
                    }
                    phase.constraints(equation) =
                        value - step_size * dynamics;
                }

                for (Index state = 0; state < phase.nx; ++state)
                {
                    const Index equation = dynamics_offset + state;
                    Scalar value = y(next_offset + state)
                        - coefficients.D(0) * y(state_offset + state);
                    phase.jacobian_y(equation, next_offset + state) = 1.0;
                    phase.jacobian_y(equation, state_offset + state) =
                        -coefficients.D(0);
                    for (Index basis = 1;
                         basis <= phase.degree; ++basis)
                    {
                        const Index collocation = local_offset + phase.nu
                            + (basis - 1) * phase.nx + state;
                        value -= coefficients.D(basis) * y(collocation);
                        phase.jacobian_y(equation, collocation) =
                            -coefficients.D(basis);
                    }
                    phase.constraints(equation) = value;
                }
            }

            const Index initial_path = phase.info.offsets_g_eq_path[0]
                + phase.degree * phase.nx;
            const Index initial_state = phase.info.offsets_primal_x[0];
            for (Index state = 0; state < phase.nx; ++state)
            {
                const Index equation = initial_path + state;
                phase.jacobian_y(equation, initial_state + state) = 1.0;
                if (phase.phase == 0)
                {
                    const Scalar target = 0.06 * std::sin(
                        0.27 * static_cast<Scalar>(state + 1));
                    phase.constraints(equation) =
                        y(initial_state + state) - target;
                }
                else
                {
                    const Index global = phase.left_separator_offset + state;
                    const Index local = phase.local_static(global);
                    if (local < 0)
                        throw std::runtime_error(
                            "Missing left boundary separator incidence");
                    phase.constraints(equation) =
                        y(initial_state + state) - local_v(local);
                    phase.jacobian_v(equation, local) = -1.0;
                }
            }

            if (phase.phase + 1
                < static_cast<Index>(config.phase_nodes.size()))
            {
                const Index target_nx = config.phase_states[
                    static_cast<std::size_t>(phase.phase + 1)];
                const Index terminal = phase.nodes - 1;
                const Index terminal_state =
                    phase.info.offsets_primal_x[terminal];
                const Index terminal_path =
                    phase.info.offsets_g_eq_path[terminal];
                for (Index equation_local = 0;
                     equation_local < target_nx; ++equation_local)
                {
                    const Index equation = terminal_path + equation_local;
                    const Index source = equation_local % phase.nx;
                    const Scalar source_value = y(terminal_state + source);
                    const Scalar source_sech_squared =
                        sech_squared(source_value);
                    Scalar reset = 0.72 * std::tanh(source_value)
                        + 0.02 * static_cast<Scalar>(phase.phase + 1);
                    phase.jacobian_y(equation, terminal_state + source) -=
                        0.72 * source_sech_squared;
                    phase.hessian_y(
                        terminal_state + source,
                        terminal_state + source) +=
                        lambda(equation) * 1.44
                        * std::tanh(source_value)
                        * source_sech_squared;
                    for (Index state = 0; state < phase.nx; ++state)
                    {
                        reset += 0.04 / static_cast<Scalar>(phase.nx)
                               * y(terminal_state + state);
                        phase.jacobian_y(equation, terminal_state + state) -=
                            0.04 / static_cast<Scalar>(phase.nx);
                    }
                    const Index separator_global =
                        phase.right_separator_offset + equation_local;
                    const Index separator_local =
                        phase.local_static(separator_global);
                    if (separator_local < 0)
                        throw std::runtime_error(
                            "Missing right boundary separator incidence");
                    for (Index column = 0;
                         column < static_cast<Index>(
                             phase.global_columns.size()); ++column)
                    {
                        if (column == separator_local)
                            continue;
                        const Scalar coefficient = deterministic_coefficient(
                            equation_local,
                            phase.global_columns[
                                static_cast<std::size_t>(column)],
                            phase.phase, 0.012);
                        reset += coefficient * std::sin(local_v(column));
                        phase.jacobian_v(equation, column) -=
                            coefficient * std::cos(local_v(column));
                        phase.hessian_vv(column, column) +=
                            lambda(equation) * coefficient
                            * std::sin(local_v(column));
                    }
                    phase.constraints(equation) =
                        local_v(separator_local) - reset;
                    phase.jacobian_v(equation, separator_local) += 1.0;
                }
            }

            // Two-sided nonlinear path constraints.  Their values occupy the
            // ProblemInfo slack-equation rows, while the evaluator exposes
            // the raw h(y,v); the primal-dual solver later forms h-s=0.
            for (Index stage = 0; stage < phase.nodes; ++stage)
            for (Index local_row = 0;
                 local_row
                    < phase.dims.number_of_ineq_constraints[stage];
                 ++local_row)
            {
                const Index equation =
                    phase.info.offsets_g_eq_slack[stage] + local_row;
                const Index state =
                    (local_row + stage + phase.phase) % phase.nx;
                const Index state_index =
                    phase.info.offsets_primal_x[stage] + state;
                const Scalar state_value = y(state_index);
                const Scalar state_sech_squared =
                    sech_squared(state_value);
                Scalar value = 0.65 * std::tanh(state_value)
                    + 0.015 * std::sin(
                        0.31 * static_cast<Scalar>(
                            1 + phase.phase + stage + 2 * local_row));
                phase.jacobian_y(equation, state_index) +=
                    0.65 * state_sech_squared;
                phase.hessian_y(state_index, state_index) +=
                    lambda(equation) * (-1.3)
                    * std::tanh(state_value) * state_sech_squared;

                if (stage + 1 < phase.nodes && phase.nu > 0)
                {
                    const Index control =
                        (local_row + 2 * stage + phase.phase) % phase.nu;
                    const Index control_index =
                        phase.info.offsets_primal_u[stage] + control;
                    value += 0.25 * y(control_index);
                    phase.jacobian_y(equation, control_index) += 0.25;
                }

                const Scalar normalization = 1.0
                    / std::sqrt(std::max<Scalar>(
                        1.0,
                        static_cast<Scalar>(phase.global_columns.size())));
                for (Index column = 0;
                     column < static_cast<Index>(
                         phase.global_columns.size()); ++column)
                {
                    const Scalar coefficient = normalization
                        * deterministic_coefficient(
                            local_row + 3 * stage,
                            phase.global_columns[
                                static_cast<std::size_t>(column)],
                            phase.phase, 0.055);
                    value += coefficient * std::sin(local_v(column));
                    phase.jacobian_v(equation, column) +=
                        coefficient * std::cos(local_v(column));
                    phase.hessian_vv(column, column) -=
                        lambda(equation) * coefficient
                        * std::sin(local_v(column));
                }
                phase.constraints(equation) = value;
            }

            phase.lagrangian_gradient_y =
                objective_factor * phase.objective_gradient_y
              + phase.jacobian_y.transpose() * lambda;
            phase.lagrangian_gradient_v =
                objective_factor * phase.objective_gradient_v
              + phase.jacobian_v.transpose() * lambda;
            populate_fatrop_blocks(phase);
        }

        void populate_fatrop_blocks(PhaseData &phase)
        {
            phase.hessian.set_zero();
            for (auto &block : phase.hessian.FuFx)
                block = 0.0;
            for (auto &block : phase.hessian.GuGx)
                block = 0.0;
            for (auto &block : phase.jacobian.BAbt)
                block = 0.0;
            for (auto &block : phase.jacobian.Jt)
                block = 0.0;
            for (auto &block : phase.jacobian.Jt_inv)
                block = 0.0;
            for (auto &block : phase.jacobian.Gg_eqt)
                block = 0.0;
            for (auto &block : phase.jacobian.Gg_ineqt)
                block = 0.0;
            for (Index stage = 0; stage < phase.nodes; ++stage)
            {
                const Index local_dimension =
                    phase.dims.number_of_controls[stage]
                  + phase.dims.number_of_states[stage];
                const Index local_offset = phase.info.offsets_primal_u[stage];
                for (Index row = 0; row < local_dimension; ++row)
                for (Index column = 0;
                     column < local_dimension; ++column)
                {
                    phase.hessian.RSQrqt[stage](row, column) =
                        phase.hessian_y(
                            local_offset + row,
                            local_offset + column);
                }
                const Index path_offset = phase.info.offsets_g_eq_path[stage];
                for (Index variable = 0;
                     variable < local_dimension; ++variable)
                for (Index equation = 0;
                     equation < phase.dims.number_of_eq_constraints[stage];
                     ++equation)
                {
                    phase.jacobian.Gg_eqt[stage](variable, equation) =
                        phase.jacobian_y(
                            path_offset + equation,
                            local_offset + variable);
                }
                const Index inequality_offset =
                    phase.info.offsets_g_eq_slack[stage];
                for (Index variable = 0;
                     variable < local_dimension; ++variable)
                for (Index equation = 0;
                     equation
                        < phase.dims.number_of_ineq_constraints[stage];
                     ++equation)
                {
                    phase.jacobian.Gg_ineqt[stage](variable, equation) =
                        phase.jacobian_y(
                            inequality_offset + equation,
                            local_offset + variable);
                }
                if (stage + 1 == phase.nodes)
                    continue;
                const Index dynamics_offset =
                    phase.info.offsets_g_eq_dyn[stage];
                const Index next_offset =
                    phase.info.offsets_primal_x[stage + 1];
                const Index next_nx = phase.dims.number_of_states[stage + 1];
                for (Index variable = 0;
                     variable < local_dimension; ++variable)
                for (Index equation = 0; equation < next_nx; ++equation)
                {
                    phase.jacobian.BAbt[stage](variable, equation) =
                        phase.jacobian_y(
                            dynamics_offset + equation,
                            local_offset + variable);
                }
                for (Index state = 0; state < next_nx; ++state)
                for (Index equation = 0; equation < next_nx; ++equation)
                {
                    phase.jacobian.Jt[stage](state, equation) =
                        phase.jacobian_y(
                            dynamics_offset + equation,
                            next_offset + state);
                    phase.jacobian.Jt_inv[stage](state, equation) =
                        state == equation ? 1.0 : 0.0;
                }
            }
            for (Index row = 0;
                 row < phase.info.number_of_primal_variables; ++row)
            {
                phase.rhs_primal(row) =
                    phase.lagrangian_gradient_y(row);
                for (Index column = 0;
                     column < static_cast<Index>(
                         phase.global_columns.size()); ++column)
                    phase.cross_hessian(row, column) =
                        phase.hessian_yv(row, column);
            }
            for (Index row = 0;
                 row < phase.info.number_of_eq_constraints; ++row)
            {
                phase.rhs_constraints(row) = phase.constraints(row);
                for (Index column = 0;
                     column < static_cast<Index>(
                         phase.global_columns.size()); ++column)
                    phase.border_jacobian(row, column) =
                        phase.jacobian_v(row, column);
            }
        }

        ScopedNlpValues evaluate(
            const Eigen::VectorXd &primal,
            const Eigen::VectorXd &multipliers,
            const Scalar objective_factor = 1.0)
        {
            if (primal.size() != total_variables
                || multipliers.size() != total_constraints)
                throw std::runtime_error(
                    "Scoped NLP primal or multiplier dimension mismatch");
            objective_value = 0.0;
            objective_gradient_global.setZero();
            constraints_global.setZero();
            lagrangian_gradient_global.setZero();
            scoped_objective_gradient.setZero();
            scoped_lagrangian_gradient.setZero();
            scoped_hessian.setZero();
            const Eigen::VectorXd scoped =
                primal.tail(total_scoped);
            for (std::size_t phase_index = 0;
                 phase_index < phases.size(); ++phase_index)
            {
                PhaseData &phase = *phases[phase_index];
                const Index y_offset = phase_variable_offsets[phase_index];
                const Index c_offset = phase_constraint_offsets[phase_index];
                const Eigen::VectorXd y = primal.segment(
                    y_offset, phase.info.number_of_primal_variables);
                const Eigen::VectorXd lambda = multipliers.segment(
                    c_offset, phase.info.number_of_eq_constraints);
                Eigen::VectorXd local_v(
                    static_cast<Eigen::Index>(phase.global_columns.size()));
                for (std::size_t column = 0;
                     column < phase.global_columns.size(); ++column)
                    local_v(static_cast<Eigen::Index>(column)) =
                        scoped(phase.global_columns[column]);
                evaluate_phase(
                    phase, y, local_v, lambda, objective_factor);
                objective_value += phase.objective;
                objective_gradient_global.segment(
                    y_offset,
                    phase.info.number_of_primal_variables) =
                    phase.objective_gradient_y;
                lagrangian_gradient_global.segment(
                    y_offset,
                    phase.info.number_of_primal_variables) =
                    phase.lagrangian_gradient_y;
                constraints_global.segment(
                    c_offset,
                    phase.info.number_of_eq_constraints) =
                    phase.constraints;
                for (Index local = 0;
                     local < static_cast<Index>(
                         phase.global_columns.size()); ++local)
                {
                    const Index global = phase.global_columns[
                        static_cast<std::size_t>(local)];
                    scoped_objective_gradient(global) +=
                        phase.objective_gradient_v(local);
                    scoped_lagrangian_gradient(global) +=
                        phase.lagrangian_gradient_v(local);
                    for (Index other = 0;
                         other < static_cast<Index>(
                             phase.global_columns.size()); ++other)
                    {
                        scoped_hessian(
                            global,
                            phase.global_columns[
                                static_cast<std::size_t>(other)]) +=
                            phase.hessian_vv(local, other);
                    }
                }
            }
            objective_gradient_global.tail(total_scoped) =
                scoped_objective_gradient;
            lagrangian_gradient_global.tail(total_scoped) =
                scoped_lagrangian_gradient;
            refresh_sparse_values();
            Scalar constraint_violation = 0.0;
            for (Index row = 0; row < total_constraints; ++row)
            {
                constraint_violation = std::max(
                    constraint_violation,
                    std::max<Scalar>(
                        0.0,
                        std::max(
                            constraint_lower(row) - constraints_global(row),
                            constraints_global(row) - constraint_upper(row))));
            }
            return {
                objective_value,
                constraint_violation,
                max_abs(lagrangian_gradient_global)};
        }

        void build_sparse_patterns()
        {
            constexpr Scalar structural_tolerance = 1e-15;
            jacobian_pattern.clear();
            jacobian_sources.clear();
            hessian_pattern.clear();
            hessian_sources.clear();
            for (std::size_t phase_index = 0;
                 phase_index < phases.size(); ++phase_index)
            {
                const PhaseData &phase = *phases[phase_index];
                const Index row_offset = phase_constraint_offsets[phase_index];
                const Index column_offset = phase_variable_offsets[phase_index];
                for (Index row = 0; row < phase.jacobian_y.rows(); ++row)
                for (Index column = 0;
                     column < phase.jacobian_y.cols(); ++column)
                {
                    if (std::abs(phase.jacobian_y(row, column))
                        <= structural_tolerance)
                        continue;
                    jacobian_pattern.emplace_back(
                        row_offset + row, column_offset + column);
                    jacobian_sources.push_back(
                        {static_cast<Index>(phase_index),
                         row, column, false});
                }
                for (Index row = 0; row < phase.jacobian_v.rows(); ++row)
                for (Index column = 0;
                     column < phase.jacobian_v.cols(); ++column)
                {
                    if (std::abs(phase.jacobian_v(row, column))
                        <= structural_tolerance)
                        continue;
                    jacobian_pattern.emplace_back(
                        row_offset + row,
                        total_trajectory + phase.global_columns[
                            static_cast<std::size_t>(column)]);
                    jacobian_sources.push_back(
                        {static_cast<Index>(phase_index),
                         row, column, true});
                }

                for (Index row = 0; row < phase.hessian_y.rows(); ++row)
                for (Index column = 0; column <= row; ++column)
                {
                    if (std::abs(phase.hessian_y(row, column))
                        <= structural_tolerance)
                        continue;
                    hessian_pattern.emplace_back(
                        column_offset + row, column_offset + column);
                    hessian_sources.push_back(
                        {HessianSourceKind::Phase,
                         static_cast<Index>(phase_index), row, column});
                }
                for (Index row = 0; row < phase.hessian_yv.rows(); ++row)
                for (Index column = 0;
                     column < phase.hessian_yv.cols(); ++column)
                {
                    if (std::abs(phase.hessian_yv(row, column))
                        <= structural_tolerance)
                        continue;
                    hessian_pattern.emplace_back(
                        total_trajectory + phase.global_columns[
                            static_cast<std::size_t>(column)],
                        column_offset + row);
                    hessian_sources.push_back(
                        {HessianSourceKind::Cross,
                         static_cast<Index>(phase_index), row, column});
                }
            }
            for (Index row = 0; row < total_scoped; ++row)
            for (Index column = 0; column <= row; ++column)
            {
                if (std::abs(scoped_hessian(row, column))
                    <= structural_tolerance)
                    continue;
                hessian_pattern.emplace_back(
                    total_trajectory + row,
                    total_trajectory + column);
                hessian_sources.push_back(
                    {HessianSourceKind::Scoped, -1, row, column});
            }
            jacobian_values.resize(jacobian_pattern.size(), 0.0);
            hessian_values.resize(hessian_pattern.size(), 0.0);
        }

        void refresh_sparse_values()
        {
            if (jacobian_values.size() != jacobian_sources.size())
                return;
            for (std::size_t entry = 0;
                 entry < jacobian_sources.size(); ++entry)
            {
                const JacobianSource &source = jacobian_sources[entry];
                const PhaseData &phase = *phases[
                    static_cast<std::size_t>(source.phase)];
                jacobian_values[entry] = source.scoped_column
                    ? phase.jacobian_v(source.row, source.column)
                    : phase.jacobian_y(source.row, source.column);
            }
            for (std::size_t entry = 0;
                 entry < hessian_sources.size(); ++entry)
            {
                const HessianSource &source = hessian_sources[entry];
                if (source.kind == HessianSourceKind::Scoped)
                    hessian_values[entry] =
                        scoped_hessian(source.row, source.column);
                else
                {
                    const PhaseData &phase = *phases[
                        static_cast<std::size_t>(source.phase)];
                    hessian_values[entry] =
                        source.kind == HessianSourceKind::Phase
                        ? phase.hessian_y(source.row, source.column)
                        : phase.hessian_yv(source.row, source.column);
                }
            }
        }

        void assemble_direct_system()
        {
            for (std::size_t block = 0; block < scopes.size(); ++block)
            {
                diagonal_storage[block] = 0.0;
                rhs_storage[block] = 0.0;
                scoped_solution_storage[block] = 0.0;
                const Index dimension = scopes[block].dimension;
                const Index offset = block_offsets[block];
                for (Index row = 0; row < dimension; ++row)
                {
                    rhs_storage[block](row) =
                        scoped_lagrangian_gradient(offset + row);
                    for (Index column = 0; column < dimension; ++column)
                        diagonal_storage[block](row, column) =
                            scoped_hessian(
                                offset + row, offset + column)
                            + (row == column
                                ? config.regularization : 0.0);
                }
            }
            const auto &edges = solver->reduced_solver().edges();
            for (std::size_t edge_index = 0;
                 edge_index < edges.size(); ++edge_index)
            {
                const IntervalKktEdge edge = edges[edge_index];
                const Index first_offset = block_offsets[
                    static_cast<std::size_t>(edge.first_block)];
                const Index second_offset = block_offsets[
                    static_cast<std::size_t>(edge.second_block)];
                const Index rows = scopes[static_cast<std::size_t>(
                    edge.first_block)].dimension;
                const Index columns = scopes[static_cast<std::size_t>(
                    edge.second_block)].dimension;
                for (Index row = 0; row < rows; ++row)
                for (Index column = 0; column < columns; ++column)
                    edge_storage[edge_index](row, column) =
                        scoped_hessian(
                            first_offset + row,
                            second_offset + column);
            }
        }

        Eigen::VectorXd collect_scoped_step() const
        {
            Eigen::VectorXd result(total_scoped);
            for (std::size_t block = 0; block < scopes.size(); ++block)
            for (Index row = 0; row < scopes[block].dimension; ++row)
                result(block_offsets[block] + row) =
                    scoped_solution_storage[block](row);
            return result;
        }

        Eigen::VectorXd dual_regularization(
            const PhaseData &phase) const
        {
            Eigen::VectorXd result = Eigen::VectorXd::Zero(
                phase.info.number_of_eq_constraints);
            for (Index stage = 0; stage < phase.nodes; ++stage)
            for (Index equation = 0;
                 equation < phase.dims.number_of_eq_constraints[stage];
                 ++equation)
            {
                result(
                    phase.info.offsets_g_eq_path[stage] + equation) =
                    phase.D_eq(
                        phase.info.offsets_eq[stage] + equation);
            }
            return result;
        }

        Scalar direction_residual(
            const Eigen::VectorXd &direction,
            const Eigen::VectorXd &multiplier_direction) const
        {
            const Eigen::VectorXd scoped_direction =
                direction.tail(total_scoped);
            Eigen::VectorXd static_residual =
                scoped_lagrangian_gradient
              + (scoped_hessian
                 + config.regularization
                    * Eigen::MatrixXd::Identity(total_scoped, total_scoped))
                    * scoped_direction;
            Scalar residual = 0.0;
            for (std::size_t phase_index = 0;
                 phase_index < phases.size(); ++phase_index)
            {
                const PhaseData &phase = *phases[phase_index];
                const Eigen::VectorXd dy = direction.segment(
                    phase_variable_offsets[phase_index],
                    phase.info.number_of_primal_variables);
                const Eigen::VectorXd dlambda =
                    multiplier_direction.segment(
                        phase_constraint_offsets[phase_index],
                        phase.info.number_of_eq_constraints);
                Eigen::VectorXd dv(
                    static_cast<Eigen::Index>(phase.global_columns.size()));
                for (std::size_t column = 0;
                     column < phase.global_columns.size(); ++column)
                    dv(static_cast<Eigen::Index>(column)) =
                        scoped_direction(phase.global_columns[column]);
                const Eigen::VectorXd primal_residual =
                    (phase.hessian_y
                     + config.regularization
                        * Eigen::MatrixXd::Identity(
                            phase.hessian_y.rows(),
                            phase.hessian_y.cols())) * dy
                  + phase.hessian_yv * dv
                  + phase.jacobian_y.transpose() * dlambda
                  + phase.lagrangian_gradient_y;
                const Eigen::VectorXd constraint_residual =
                    phase.jacobian_y * dy
                  + phase.jacobian_v * dv
                  + phase.constraints
                  - dual_regularization(phase).cwiseProduct(dlambda);
                residual = std::max(
                    residual,
                    std::max(
                        max_abs(primal_residual),
                        max_abs(constraint_residual)));
                const Eigen::VectorXd contribution =
                    phase.hessian_yv.transpose() * dy
                  + phase.jacobian_v.transpose() * dlambda;
                for (std::size_t column = 0;
                     column < phase.global_columns.size(); ++column)
                    static_residual(phase.global_columns[column]) +=
                        contribution(static_cast<Eigen::Index>(column));
            }
            return std::max(residual, max_abs(static_residual));
        }

        Scalar dense_step_difference(
            const Eigen::VectorXd &direction,
            const Eigen::VectorXd &multiplier_direction) const
        {
            const Index total = total_variables + total_constraints;
            if (total > 2400)
                return std::numeric_limits<Scalar>::quiet_NaN();
            Eigen::MatrixXd kkt = Eigen::MatrixXd::Zero(total, total);
            Eigen::VectorXd rhs(total);
            rhs.head(total_variables) = -lagrangian_gradient_global;
            rhs.tail(total_constraints) = -constraints_global;
            kkt.topLeftCorner(total_variables, total_variables).diagonal()
                .array() += config.regularization;
            kkt.block(
                total_trajectory, total_trajectory,
                total_scoped, total_scoped) += scoped_hessian;
            for (std::size_t phase_index = 0;
                 phase_index < phases.size(); ++phase_index)
            {
                const PhaseData &phase = *phases[phase_index];
                const Index y_offset = phase_variable_offsets[phase_index];
                const Index c_offset = phase_constraint_offsets[phase_index];
                kkt.block(
                    y_offset, y_offset,
                    phase.hessian_y.rows(),
                    phase.hessian_y.cols()) += phase.hessian_y;
                kkt.block(
                    total_variables + c_offset,
                    y_offset,
                    phase.jacobian_y.rows(),
                    phase.jacobian_y.cols()) = phase.jacobian_y;
                kkt.block(
                    y_offset,
                    total_variables + c_offset,
                    phase.jacobian_y.cols(),
                    phase.jacobian_y.rows()) =
                    phase.jacobian_y.transpose();
                const Eigen::VectorXd dual_diagonal =
                    dual_regularization(phase);
                kkt.block(
                    total_variables + c_offset,
                    total_variables + c_offset,
                    dual_diagonal.size(), dual_diagonal.size()).diagonal()
                    -= dual_diagonal;
                for (Index local = 0;
                     local < static_cast<Index>(
                         phase.global_columns.size()); ++local)
                {
                    const Index global = total_trajectory
                        + phase.global_columns[
                            static_cast<std::size_t>(local)];
                    kkt.block(
                        y_offset, global,
                        phase.hessian_yv.rows(), 1) =
                        phase.hessian_yv.col(local);
                    kkt.block(
                        global, y_offset,
                        1, phase.hessian_yv.rows()) =
                        phase.hessian_yv.col(local).transpose();
                    kkt.block(
                        total_variables + c_offset,
                        global,
                        phase.jacobian_v.rows(), 1) =
                        phase.jacobian_v.col(local);
                    kkt.block(
                        global,
                        total_variables + c_offset,
                        1, phase.jacobian_v.rows()) =
                        phase.jacobian_v.col(local).transpose();
                }
            }
            const Eigen::FullPivLU<Eigen::MatrixXd> factor(kkt);
            if (factor.rank() != total)
                return std::numeric_limits<Scalar>::infinity();
            const Eigen::VectorXd reference = factor.solve(rhs);
            Eigen::VectorXd computed(total);
            computed.head(total_variables) = direction;
            computed.tail(total_constraints) = multiplier_direction;
            return max_abs(computed - reference);
        }

        ScopedNewtonDirection solve_direction(
            const Eigen::VectorXd &primal,
            const Eigen::VectorXd &multipliers,
            const bool validate_dense)
        {
            if (total_inequalities != 0)
                throw std::runtime_error(
                    "solve_direction is equality-SQP only; use solve_ipm "
                    "when path inequalities are present");
            evaluate(primal, multipliers);
            assemble_direct_system();
            const auto start = Clock::now();
            const LinsolReturnFlag status = solver->solve(
                phase_views,
                diagonal_views,
                edge_views,
                rhs_views,
                scoped_solution_views);
            const double elapsed = std::chrono::duration<double, std::milli>(
                Clock::now() - start).count();
            ScopedNewtonDirection result;
            result.primal = Eigen::VectorXd::Zero(total_variables);
            result.multipliers = Eigen::VectorXd::Zero(total_constraints);
            result.elapsed_ms = elapsed;
            result.reduced_ms =
                solver->last_statistics().reduced_solve_milliseconds;
            result.success = status == LinsolReturnFlag::SUCCESS;
            if (!result.success)
                return result;
            for (std::size_t phase_index = 0;
                 phase_index < phases.size(); ++phase_index)
            {
                const PhaseData &phase = *phases[phase_index];
                for (Index row = 0;
                     row < phase.info.number_of_primal_variables; ++row)
                    result.primal(
                        phase_variable_offsets[phase_index] + row) =
                        phase.primal_step(row);
                for (Index row = 0;
                     row < phase.info.number_of_eq_constraints; ++row)
                    result.multipliers(
                        phase_constraint_offsets[phase_index] + row) =
                        phase.multiplier_step(row);
            }
            result.primal.tail(total_scoped) = collect_scoped_step();
            result.residual_inf = direction_residual(
                result.primal, result.multipliers);
            result.dense_difference = validate_dense
                ? dense_step_difference(
                    result.primal, result.multipliers)
                : std::numeric_limits<Scalar>::quiet_NaN();
            return result;
        }

        ScopedDerivativeCheck check_derivatives(
            const Eigen::VectorXd &primal,
            const Scalar step)
        {
            Eigen::VectorXd direction(primal.size());
            for (Index row = 0; row < direction.size(); ++row)
                direction(row) = std::sin(
                    0.37 * static_cast<Scalar>(row + 1));
            direction.normalize();
            const Eigen::VectorXd zero =
                Eigen::VectorXd::Zero(total_constraints);
            evaluate(primal, zero);
            const Scalar analytic_objective =
                objective_gradient_global.dot(direction);
            const Eigen::VectorXd analytic_constraints = [&]()
            {
                Eigen::VectorXd result(total_constraints);
                for (std::size_t phase_index = 0;
                     phase_index < phases.size(); ++phase_index)
                {
                    const PhaseData &phase = *phases[phase_index];
                    const Eigen::VectorXd dy = direction.segment(
                        phase_variable_offsets[phase_index],
                        phase.info.number_of_primal_variables);
                    Eigen::VectorXd dv(
                        static_cast<Eigen::Index>(
                            phase.global_columns.size()));
                    for (std::size_t column = 0;
                         column < phase.global_columns.size(); ++column)
                        dv(static_cast<Eigen::Index>(column)) =
                            direction(
                                total_trajectory
                                + phase.global_columns[column]);
                    result.segment(
                        phase_constraint_offsets[phase_index],
                        phase.info.number_of_eq_constraints) =
                        phase.jacobian_y * dy + phase.jacobian_v * dv;
                }
                return result;
            }();
            const ScopedNlpValues plus = evaluate(
                primal + step * direction, zero);
            const Eigen::VectorXd constraints_plus = constraints_global;
            const ScopedNlpValues minus = evaluate(
                primal - step * direction, zero);
            const Eigen::VectorXd constraints_minus = constraints_global;
            const Scalar finite_objective =
                (plus.objective - minus.objective) / (2.0 * step);
            const Eigen::VectorXd finite_constraints =
                (constraints_plus - constraints_minus) / (2.0 * step);

            Eigen::VectorXd multipliers(total_constraints);
            for (Index row = 0; row < multipliers.size(); ++row)
                multipliers(row) = 0.07 * std::cos(
                    0.23 * static_cast<Scalar>(row + 1));
            constexpr Scalar objective_factor = 0.43;
            evaluate(primal, multipliers, objective_factor);
            Eigen::VectorXd analytic_hessian_direction =
                Eigen::VectorXd::Zero(total_variables);
            for (std::size_t entry = 0;
                 entry < hessian_pattern.size(); ++entry)
            {
                const Index row = hessian_pattern[entry].first;
                const Index column = hessian_pattern[entry].second;
                const Scalar value = hessian_values[entry];
                analytic_hessian_direction(row) +=
                    value * direction(column);
                if (row != column)
                    analytic_hessian_direction(column) +=
                        value * direction(row);
            }
            evaluate(
                primal + step * direction,
                multipliers,
                objective_factor);
            const Eigen::VectorXd lagrangian_gradient_plus =
                lagrangian_gradient_global;
            evaluate(
                primal - step * direction,
                multipliers,
                objective_factor);
            const Eigen::VectorXd lagrangian_gradient_minus =
                lagrangian_gradient_global;
            const Eigen::VectorXd finite_hessian_direction =
                (lagrangian_gradient_plus - lagrangian_gradient_minus)
                / (2.0 * step);
            evaluate(primal, zero);
            return {
                std::abs(analytic_objective - finite_objective)
                    / (1.0 + std::abs(analytic_objective)
                       + std::abs(finite_objective)),
                max_abs(analytic_constraints - finite_constraints)
                    / (1.0 + max_abs(analytic_constraints)
                       + max_abs(finite_constraints)),
                max_abs(
                    analytic_hessian_direction
                    - finite_hessian_direction)
                    / (1.0 + max_abs(analytic_hessian_direction)
                       + max_abs(finite_hessian_direction))};
        }

        ScopedPdState make_initial_pd_state()
        {
            ScopedPdState state;
            state.primal = primal_initial;
            state.multipliers = Eigen::VectorXd::Zero(total_constraints);
            state.slack = Eigen::VectorXd::Zero(total_inequalities);
            state.lower_dual = Eigen::VectorXd::Zero(total_inequalities);
            state.upper_dual = Eigen::VectorXd::Zero(total_inequalities);
            evaluate(state.primal, state.multipliers);
            const Scalar initial_mu = std::max<Scalar>(
                1e-2, 10.0 * config.tolerance);
            for (std::size_t phase_index = 0;
                 phase_index < phases.size(); ++phase_index)
            {
                const PhaseData &phase = *phases[phase_index];
                const Index constraint_offset =
                    phase_constraint_offsets[phase_index];
                for (Index local = 0;
                     local < phase.info.number_of_slack_variables; ++local)
                {
                    const Index global_slack =
                        phase.global_slack_offset + local;
                    const Index local_constraint =
                        phase.info.offset_g_eq_slack + local;
                    const Index global_constraint =
                        constraint_offset + local_constraint;
                    const Scalar lower = constraint_lower(global_constraint);
                    const Scalar upper = constraint_upper(global_constraint);
                    const Scalar margin = 0.1 * (upper - lower);
                    state.slack(global_slack) = std::clamp(
                        phase.constraints(local_constraint),
                        lower + margin,
                        upper - margin);
                    const Scalar lower_distance =
                        state.slack(global_slack) - lower;
                    const Scalar upper_distance =
                        upper - state.slack(global_slack);
                    state.lower_dual(global_slack) =
                        initial_mu / lower_distance;
                    state.upper_dual(global_slack) =
                        initial_mu / upper_distance;
                    state.multipliers(global_constraint) =
                        state.upper_dual(global_slack)
                      - state.lower_dual(global_slack);
                }
            }
            return state;
        }

        void prepare_pd_system(
            const ScopedPdState &state,
            const Scalar target_mu,
            const Eigen::VectorXd &lower_correction,
            const Eigen::VectorXd &upper_correction)
        {
            if (state.primal.size() != total_variables
                || state.multipliers.size() != total_constraints
                || state.slack.size() != total_inequalities
                || state.lower_dual.size() != total_inequalities
                || state.upper_dual.size() != total_inequalities
                || lower_correction.size() != total_inequalities
                || upper_correction.size() != total_inequalities)
                throw std::runtime_error(
                    "Interval-scoped primal-dual state dimension mismatch");
            evaluate(state.primal, state.multipliers);
            for (std::size_t phase_index = 0;
                 phase_index < phases.size(); ++phase_index)
            {
                PhaseData &phase = *phases[phase_index];
                const Index constraint_offset =
                    phase_constraint_offsets[phase_index];
                phase.pd_solution = 0.0;
                for (Index local = 0;
                     local < phase.info.number_of_slack_variables; ++local)
                {
                    const Index global_slack =
                        phase.global_slack_offset + local;
                    const Index local_constraint =
                        phase.info.offset_g_eq_slack + local;
                    const Index global_constraint =
                        constraint_offset + local_constraint;
                    const Scalar slack = state.slack(global_slack);
                    const Scalar lower_distance =
                        slack - constraint_lower(global_constraint);
                    const Scalar upper_distance =
                        constraint_upper(global_constraint) - slack;
                    if (!(lower_distance > 0.0)
                        || !(upper_distance > 0.0)
                        || !(state.lower_dual(global_slack) > 0.0)
                        || !(state.upper_dual(global_slack) > 0.0))
                        throw std::runtime_error(
                            "Primal-dual iterate left the strict interior");
                    phase.slack_lower_distance(local) = lower_distance;
                    phase.slack_upper_distance(local) = upper_distance;
                    phase.slack_lower_dual(local) =
                        state.lower_dual(global_slack);
                    phase.slack_upper_dual(local) =
                        state.upper_dual(global_slack);
                    phase.rhs_slack(local) =
                        -state.multipliers(global_constraint)
                        - state.lower_dual(global_slack)
                        + state.upper_dual(global_slack);
                    phase.rhs_constraints(local_constraint) =
                        phase.constraints(local_constraint) - slack;
                    phase.rhs_lower_complementarity(local) =
                        lower_distance * state.lower_dual(global_slack)
                        - target_mu + lower_correction(global_slack);
                    phase.rhs_upper_complementarity(local) =
                        upper_distance * state.upper_dual(global_slack)
                        - target_mu + upper_correction(global_slack);
                }
            }
            assemble_direct_system();
        }

        ScopedPdDirection collect_pd_direction() const
        {
            ScopedPdDirection direction;
            direction.primal = Eigen::VectorXd::Zero(total_variables);
            direction.multipliers =
                Eigen::VectorXd::Zero(total_constraints);
            direction.slack = Eigen::VectorXd::Zero(total_inequalities);
            direction.lower_dual =
                Eigen::VectorXd::Zero(total_inequalities);
            direction.upper_dual =
                Eigen::VectorXd::Zero(total_inequalities);
            for (std::size_t phase_index = 0;
                 phase_index < phases.size(); ++phase_index)
            {
                const PhaseData &phase = *phases[phase_index];
                const Index variable_offset =
                    phase_variable_offsets[phase_index];
                const Index constraint_offset =
                    phase_constraint_offsets[phase_index];
                for (Index row = 0;
                     row < phase.info.number_of_primal_variables; ++row)
                    direction.primal(variable_offset + row) =
                        phase.pd_solution(
                            phase.info.pd_orig_offset_primal + row);
                for (Index row = 0;
                     row < phase.info.number_of_eq_constraints; ++row)
                    direction.multipliers(constraint_offset + row) =
                        phase.pd_solution(
                            phase.info.pd_orig_offset_mult + row);
                for (Index local = 0;
                     local < phase.info.number_of_slack_variables; ++local)
                {
                    const Index global =
                        phase.global_slack_offset + local;
                    direction.slack(global) = phase.pd_solution(
                        phase.info.pd_orig_offset_slack + local);
                    direction.lower_dual(global) = phase.pd_solution(
                        phase.info.pd_orig_offset_zl + local);
                    direction.upper_dual(global) = phase.pd_solution(
                        phase.info.pd_orig_offset_zu + local);
                }
            }
            direction.primal.tail(total_scoped) = collect_scoped_step();
            return direction;
        }

        Scalar pd_direction_residual(
            const ScopedPdDirection &direction) const
        {
            const Eigen::VectorXd scoped_direction =
                direction.primal.tail(total_scoped);
            Eigen::VectorXd scoped_residual =
                scoped_lagrangian_gradient
              + (scoped_hessian
                 + config.regularization * Eigen::MatrixXd::Identity(
                     total_scoped, total_scoped)) * scoped_direction;
            Scalar residual = 0.0;
            for (std::size_t phase_index = 0;
                 phase_index < phases.size(); ++phase_index)
            {
                const PhaseData &phase = *phases[phase_index];
                const Index variable_offset =
                    phase_variable_offsets[phase_index];
                const Index constraint_offset =
                    phase_constraint_offsets[phase_index];
                const Eigen::VectorXd dy = direction.primal.segment(
                    variable_offset,
                    phase.info.number_of_primal_variables);
                const Eigen::VectorXd dlambda =
                    direction.multipliers.segment(
                        constraint_offset,
                        phase.info.number_of_eq_constraints);
                Eigen::VectorXd dv(
                    static_cast<Eigen::Index>(phase.global_columns.size()));
                for (std::size_t column = 0;
                     column < phase.global_columns.size(); ++column)
                    dv(static_cast<Eigen::Index>(column)) =
                        scoped_direction(phase.global_columns[column]);

                Eigen::VectorXd primal_residual =
                    phase.hessian_y * dy
                  + phase.hessian_yv * dv
                  + phase.jacobian_y.transpose() * dlambda
                  + phase.lagrangian_gradient_y;
                for (Index row = 0;
                     row < phase.info.number_of_primal_variables; ++row)
                    primal_residual(row) +=
                        phase.pd_D_x(phase.info.offset_primal + row)
                        * dy(row);
                Eigen::VectorXd constraint_residual =
                    phase.jacobian_y * dy
                  + phase.jacobian_v * dv
                  + phase.constraints;
                for (Index row = 0;
                     row < phase.info.number_of_eq_constraints; ++row)
                {
                    constraint_residual(row) =
                        phase.rhs_constraints(row)
                      + phase.jacobian_y.row(row).dot(dy)
                      + phase.jacobian_v.row(row).dot(dv)
                      - phase.pd_D_e(row) * dlambda(row);
                }
                for (Index local = 0;
                     local < phase.info.number_of_slack_variables; ++local)
                    constraint_residual(
                        phase.info.offset_g_eq_slack + local) -=
                        direction.slack(
                            phase.global_slack_offset + local);
                residual = std::max(
                    residual,
                    std::max(
                        max_abs(primal_residual),
                        max_abs(constraint_residual)));

                const Eigen::VectorXd scoped_contribution =
                    phase.hessian_yv.transpose() * dy
                  + phase.jacobian_v.transpose() * dlambda;
                for (std::size_t column = 0;
                     column < phase.global_columns.size(); ++column)
                    scoped_residual(phase.global_columns[column]) +=
                        scoped_contribution(
                            static_cast<Eigen::Index>(column));

                for (Index local = 0;
                     local < phase.info.number_of_slack_variables; ++local)
                {
                    const Index global =
                        phase.global_slack_offset + local;
                    const Index local_constraint =
                        phase.info.offset_g_eq_slack + local;
                    const Scalar ds = direction.slack(global);
                    const Scalar dzl = direction.lower_dual(global);
                    const Scalar dzu = direction.upper_dual(global);
                    const Scalar slack_residual =
                        phase.rhs_slack(local)
                      + phase.pd_D_x(
                            phase.info.offset_slack + local) * ds
                      - dlambda(local_constraint) - dzl + dzu;
                    const Scalar lower_residual =
                        phase.rhs_lower_complementarity(local)
                      + phase.slack_lower_dual(local) * ds
                      + phase.slack_lower_distance(local) * dzl;
                    const Scalar upper_residual =
                        phase.rhs_upper_complementarity(local)
                      - phase.slack_upper_dual(local) * ds
                      + phase.slack_upper_distance(local) * dzu;
                    residual = std::max(
                        residual,
                        std::max(
                            std::abs(slack_residual),
                            std::max(
                                std::abs(lower_residual),
                                std::abs(upper_residual))));
                }
            }
            return std::max(residual, max_abs(scoped_residual));
        }

        Scalar dense_pd_step_difference(
            const ScopedPdDirection &direction) const
        {
            const Index dimension = total_variables
                + total_inequalities + total_constraints
                + 2 * total_inequalities;
            if (dimension > 1800)
                return std::numeric_limits<Scalar>::quiet_NaN();
            const Index slack_offset = total_variables;
            const Index multiplier_offset =
                slack_offset + total_inequalities;
            const Index lower_dual_offset =
                multiplier_offset + total_constraints;
            const Index upper_dual_offset =
                lower_dual_offset + total_inequalities;
            Eigen::MatrixXd matrix =
                Eigen::MatrixXd::Zero(dimension, dimension);
            Eigen::VectorXd rhs = Eigen::VectorXd::Zero(dimension);
            rhs.head(total_variables) = -lagrangian_gradient_global;
            matrix.block(
                total_trajectory, total_trajectory,
                total_scoped, total_scoped) += scoped_hessian;
            matrix.block(
                total_trajectory, total_trajectory,
                total_scoped, total_scoped).diagonal().array() +=
                config.regularization;

            for (std::size_t phase_index = 0;
                 phase_index < phases.size(); ++phase_index)
            {
                const PhaseData &phase = *phases[phase_index];
                const Index variable_offset =
                    phase_variable_offsets[phase_index];
                const Index constraint_offset =
                    phase_constraint_offsets[phase_index];
                const Index lambda_offset =
                    multiplier_offset + constraint_offset;
                matrix.block(
                    variable_offset, variable_offset,
                    phase.hessian_y.rows(), phase.hessian_y.cols()) +=
                    phase.hessian_y;
                for (Index row = 0;
                     row < phase.info.number_of_primal_variables; ++row)
                    matrix(variable_offset + row, variable_offset + row) +=
                        phase.pd_D_x(phase.info.offset_primal + row);
                matrix.block(
                    lambda_offset, variable_offset,
                    phase.jacobian_y.rows(), phase.jacobian_y.cols()) =
                    phase.jacobian_y;
                matrix.block(
                    variable_offset, lambda_offset,
                    phase.jacobian_y.cols(), phase.jacobian_y.rows()) =
                    phase.jacobian_y.transpose();
                for (Index row = 0;
                     row < phase.info.number_of_eq_constraints; ++row)
                {
                    matrix(lambda_offset + row, lambda_offset + row) -=
                        phase.pd_D_e(row);
                    rhs(lambda_offset + row) =
                        -phase.rhs_constraints(row);
                }
                for (Index local = 0;
                     local < static_cast<Index>(
                         phase.global_columns.size()); ++local)
                {
                    const Index global = total_trajectory
                        + phase.global_columns[
                            static_cast<std::size_t>(local)];
                    matrix.block(
                        variable_offset, global,
                        phase.hessian_yv.rows(), 1) =
                        phase.hessian_yv.col(local);
                    matrix.block(
                        global, variable_offset,
                        1, phase.hessian_yv.rows()) =
                        phase.hessian_yv.col(local).transpose();
                    matrix.block(
                        lambda_offset, global,
                        phase.jacobian_v.rows(), 1) =
                        phase.jacobian_v.col(local);
                    matrix.block(
                        global, lambda_offset,
                        1, phase.jacobian_v.rows()) =
                        phase.jacobian_v.col(local).transpose();
                }
                for (Index local = 0;
                     local < phase.info.number_of_slack_variables; ++local)
                {
                    const Index global =
                        phase.global_slack_offset + local;
                    const Index s = slack_offset + global;
                    const Index zl = lower_dual_offset + global;
                    const Index zu = upper_dual_offset + global;
                    const Index local_constraint =
                        phase.info.offset_g_eq_slack + local;
                    const Index lambda = lambda_offset + local_constraint;
                    matrix(s, s) += phase.pd_D_x(
                        phase.info.offset_slack + local);
                    matrix(s, lambda) = -1.0;
                    matrix(s, zl) = -1.0;
                    matrix(s, zu) = 1.0;
                    matrix(lambda, s) = -1.0;
                    matrix(zl, s) = phase.slack_lower_dual(local);
                    matrix(zl, zl) = phase.slack_lower_distance(local);
                    matrix(zu, s) = -phase.slack_upper_dual(local);
                    matrix(zu, zu) = phase.slack_upper_distance(local);
                    rhs(s) = -phase.rhs_slack(local);
                    rhs(zl) =
                        -phase.rhs_lower_complementarity(local);
                    rhs(zu) =
                        -phase.rhs_upper_complementarity(local);
                }
            }
            const Eigen::FullPivLU<Eigen::MatrixXd> factor(matrix);
            if (factor.rank() != dimension)
                return std::numeric_limits<Scalar>::infinity();
            const Eigen::VectorXd reference = factor.solve(rhs);
            Eigen::VectorXd computed(dimension);
            computed.head(total_variables) = direction.primal;
            computed.segment(slack_offset, total_inequalities) =
                direction.slack;
            computed.segment(multiplier_offset, total_constraints) =
                direction.multipliers;
            computed.segment(lower_dual_offset, total_inequalities) =
                direction.lower_dual;
            computed.segment(upper_dual_offset, total_inequalities) =
                direction.upper_dual;
            return max_abs(computed - reference);
        }

        ScopedPdDirection solve_pd_direction(
            const ScopedPdState &state,
            const Scalar target_mu,
            const Eigen::VectorXd &lower_correction,
            const Eigen::VectorXd &upper_correction,
            const bool reuse_factorization,
            const bool validate_dense)
        {
            prepare_pd_system(
                state, target_mu,
                lower_correction, upper_correction);
            const auto start = Clock::now();
            const LinsolReturnFlag status = reuse_factorization
                ? pd_solver->solve_rhs(
                    pd_phase_views,
                    rhs_views,
                    scoped_solution_views)
                : pd_solver->solve(
                    pd_phase_views,
                    diagonal_views,
                    edge_views,
                    rhs_views,
                    scoped_solution_views);
            const double elapsed = std::chrono::duration<double, std::milli>(
                Clock::now() - start).count();
            ScopedPdDirection direction = collect_pd_direction();
            direction.elapsed_ms = elapsed;
            direction.reduced_ms = pd_solver->phase_solver()
                .last_statistics().reduced_solve_milliseconds;
            direction.success = status == LinsolReturnFlag::SUCCESS;
            if (!direction.success)
                return direction;
            direction.residual_inf = pd_direction_residual(direction);
            direction.dense_difference = validate_dense
                ? dense_pd_step_difference(direction)
                : std::numeric_limits<Scalar>::quiet_NaN();
            return direction;
        }

        ScopedPdMetrics pd_metrics(
            const ScopedPdState &state,
            const Scalar target_mu)
        {
            const ScopedNlpValues values = evaluate(
                state.primal, state.multipliers);
            ScopedPdMetrics metrics;
            metrics.primal_inf = values.constraint_inf;
            metrics.dual_inf = values.dual_inf;
            Scalar complementarity_sum = 0.0;
            for (std::size_t phase_index = 0;
                 phase_index < phases.size(); ++phase_index)
            {
                const PhaseData &phase = *phases[phase_index];
                const Index constraint_offset =
                    phase_constraint_offsets[phase_index];
                for (Index row = 0;
                     row < phase.info.offset_g_eq_slack; ++row)
                    metrics.primal_inf = std::max(
                        metrics.primal_inf,
                        std::abs(phase.constraints(row)));
                for (Index local = 0;
                     local < phase.info.number_of_slack_variables; ++local)
                {
                    const Index global =
                        phase.global_slack_offset + local;
                    const Index local_constraint =
                        phase.info.offset_g_eq_slack + local;
                    const Index global_constraint =
                        constraint_offset + local_constraint;
                    const Scalar lower_distance = state.slack(global)
                        - constraint_lower(global_constraint);
                    const Scalar upper_distance =
                        constraint_upper(global_constraint)
                        - state.slack(global);
                    const Scalar lower_product =
                        lower_distance * state.lower_dual(global);
                    const Scalar upper_product =
                        upper_distance * state.upper_dual(global);
                    const Scalar primal_residual =
                        phase.constraints(local_constraint)
                        - state.slack(global);
                    const Scalar slack_dual_residual =
                        -state.multipliers(global_constraint)
                        - state.lower_dual(global)
                        + state.upper_dual(global);
                    metrics.primal_inf = std::max(
                        metrics.primal_inf,
                        std::abs(primal_residual));
                    metrics.dual_inf = std::max(
                        metrics.dual_inf,
                        std::abs(slack_dual_residual));
                    metrics.complementarity_inf = std::max(
                        metrics.complementarity_inf,
                        std::max(lower_product, upper_product));
                    metrics.residual_inf = std::max(
                        metrics.residual_inf,
                        std::max(
                            std::abs(lower_product - target_mu),
                            std::abs(upper_product - target_mu)));
                    complementarity_sum +=
                        lower_product + upper_product;
                    metrics.finite = metrics.finite
                        && std::isfinite(lower_distance)
                        && std::isfinite(upper_distance)
                        && lower_distance > 0.0
                        && upper_distance > 0.0
                        && state.lower_dual(global) > 0.0
                        && state.upper_dual(global) > 0.0;
                }
            }
            metrics.residual_inf = std::max(
                metrics.residual_inf,
                std::max(metrics.primal_inf, metrics.dual_inf));
            metrics.average_complementarity = total_inequalities > 0
                ? complementarity_sum
                    / (2.0 * static_cast<Scalar>(total_inequalities))
                : 0.0;
            metrics.finite = metrics.finite
                && state.primal.allFinite()
                && state.multipliers.allFinite()
                && state.slack.allFinite()
                && state.lower_dual.allFinite()
                && state.upper_dual.allFinite()
                && std::isfinite(metrics.residual_inf);
            return metrics;
        }

        Scalar maximum_pd_step(
            const ScopedPdState &state,
            const ScopedPdDirection &direction,
            const Scalar fraction) const
        {
            Scalar maximum = std::numeric_limits<Scalar>::infinity();
            for (std::size_t phase_index = 0;
                 phase_index < phases.size(); ++phase_index)
            {
                const PhaseData &phase = *phases[phase_index];
                const Index constraint_offset =
                    phase_constraint_offsets[phase_index];
                for (Index local = 0;
                     local < phase.info.number_of_slack_variables; ++local)
                {
                    const Index global =
                        phase.global_slack_offset + local;
                    const Index global_constraint = constraint_offset
                        + phase.info.offset_g_eq_slack + local;
                    const Scalar ds = direction.slack(global);
                    if (ds < 0.0)
                        maximum = std::min(
                            maximum,
                            -(state.slack(global)
                              - constraint_lower(global_constraint)) / ds);
                    else if (ds > 0.0)
                        maximum = std::min(
                            maximum,
                            (constraint_upper(global_constraint)
                              - state.slack(global)) / ds);
                    if (direction.lower_dual(global) < 0.0)
                        maximum = std::min(
                            maximum,
                            -state.lower_dual(global)
                            / direction.lower_dual(global));
                    if (direction.upper_dual(global) < 0.0)
                        maximum = std::min(
                            maximum,
                            -state.upper_dual(global)
                            / direction.upper_dual(global));
                }
            }
            if (!std::isfinite(maximum))
                return 1.0;
            return std::max<Scalar>(
                0.0, std::min<Scalar>(1.0, fraction * maximum));
        }

        ScopedPdState trial_pd_state(
            const ScopedPdState &state,
            const ScopedPdDirection &direction,
            const Scalar alpha) const
        {
            return {
                state.primal + alpha * direction.primal,
                state.multipliers + alpha * direction.multipliers,
                state.slack + alpha * direction.slack,
                state.lower_dual + alpha * direction.lower_dual,
                state.upper_dual + alpha * direction.upper_dual};
        }

        Scalar affine_complementarity(
            const ScopedPdState &state,
            const ScopedPdDirection &direction,
            const Scalar alpha) const
        {
            Scalar sum = 0.0;
            for (std::size_t phase_index = 0;
                 phase_index < phases.size(); ++phase_index)
            {
                const PhaseData &phase = *phases[phase_index];
                const Index constraint_offset =
                    phase_constraint_offsets[phase_index];
                for (Index local = 0;
                     local < phase.info.number_of_slack_variables; ++local)
                {
                    const Index global =
                        phase.global_slack_offset + local;
                    const Index global_constraint = constraint_offset
                        + phase.info.offset_g_eq_slack + local;
                    const Scalar slack = state.slack(global)
                        + alpha * direction.slack(global);
                    sum += (slack - constraint_lower(global_constraint))
                        * (state.lower_dual(global)
                           + alpha * direction.lower_dual(global));
                    sum += (constraint_upper(global_constraint) - slack)
                        * (state.upper_dual(global)
                           + alpha * direction.upper_dual(global));
                }
            }
            return sum
                / (2.0 * static_cast<Scalar>(total_inequalities));
        }

        ScopedIpmResult solve_ipm()
        {
            if (total_inequalities == 0)
            {
                const ScopedSqpResult sqp = solve_sqp();
                ScopedIpmResult result;
                result.primal = sqp.primal;
                result.multipliers = sqp.multipliers;
                result.slack.resize(0);
                result.lower_dual.resize(0);
                result.upper_dual.resize(0);
                result.objective = sqp.objective;
                result.constraint_inf = sqp.constraint_inf;
                result.dual_inf = sqp.dual_inf;
                result.linear_residual = sqp.linear_residual;
                result.dense_step_difference = sqp.dense_step_difference;
                result.elapsed_ms = sqp.elapsed_ms;
                result.kkt_ms = sqp.kkt_ms;
                result.iterations = sqp.iterations;
                result.line_search_evaluations =
                    sqp.line_search_evaluations;
                result.factorizations = sqp.iterations;
                result.converged = sqp.converged;
                return result;
            }

            ScopedIpmResult result;
            ScopedPdState state = make_initial_pd_state();
            const Eigen::VectorXd zero_correction =
                Eigen::VectorXd::Zero(total_inequalities);
            const auto start = Clock::now();
            for (Index iteration = 0;
                 iteration < config.max_iterations; ++iteration)
            {
                const ScopedPdMetrics current = pd_metrics(state, 0.0);
                if (current.finite
                    && std::max(
                        current.primal_inf,
                        std::max(
                            current.dual_inf,
                            current.complementarity_inf))
                        <= config.tolerance)
                {
                    result.converged = true;
                    break;
                }
                if (!current.finite
                    || !(current.average_complementarity > 0.0))
                    break;

                const ScopedPdDirection affine = solve_pd_direction(
                    state, 0.0,
                    zero_correction, zero_correction,
                    false, false);
                ++result.factorizations;
                result.kkt_ms += affine.elapsed_ms;
                result.linear_residual = std::max(
                    result.linear_residual, affine.residual_inf);
                if (!affine.success)
                    break;
                const Scalar affine_alpha = maximum_pd_step(
                    state, affine, 1.0);
                const Scalar mu_affine = std::max<Scalar>(
                    0.0,
                    affine_complementarity(
                        state, affine, affine_alpha));
                const Scalar ratio = std::clamp(
                    mu_affine / current.average_complementarity,
                    Scalar{0.0}, Scalar{1.0});
                const Scalar sigma = ratio * ratio * ratio;
                const Scalar target_mu =
                    sigma * current.average_complementarity;
                const Eigen::VectorXd lower_correction =
                    affine.slack.cwiseProduct(affine.lower_dual);
                const Eigen::VectorXd upper_correction =
                    -affine.slack.cwiseProduct(affine.upper_dual);
                const ScopedPdDirection direction = solve_pd_direction(
                    state, target_mu,
                    lower_correction, upper_correction,
                    config.reuse_predictor_corrector_factorization,
                    config.dense_step_validation && iteration == 0);
                if (config.reuse_predictor_corrector_factorization)
                    ++result.retained_rhs_solves;
                else
                    ++result.factorizations;
                result.kkt_ms += direction.elapsed_ms;
                result.linear_residual = std::max(
                    result.linear_residual, direction.residual_inf);
                if (iteration == 0)
                    result.dense_step_difference =
                        direction.dense_difference;
                if (!direction.success)
                    break;

                Scalar alpha = maximum_pd_step(
                    state, direction, config.fraction_to_boundary);
                const ScopedPdMetrics current_target =
                    pd_metrics(state, target_mu);
                bool accepted = false;
                for (Index line_search = 0;
                     line_search < 32; ++line_search)
                {
                    const ScopedPdState trial = trial_pd_state(
                        state, direction, alpha);
                    const ScopedPdMetrics trial_metrics =
                        pd_metrics(trial, target_mu);
                    ++result.line_search_evaluations;
                    if (trial_metrics.finite
                        && trial_metrics.residual_inf
                            <= (1.0 - 1e-4 * alpha)
                                * current_target.residual_inf)
                    {
                        state = trial;
                        accepted = true;
                        break;
                    }
                    alpha *= 0.5;
                }
                if (!accepted)
                    break;
                result.iterations = iteration + 1;
            }

            const ScopedPdMetrics final_metrics = pd_metrics(state, 0.0);
            const ScopedNlpValues final_values = evaluate(
                state.primal, state.multipliers);
            result.primal = state.primal;
            result.multipliers = state.multipliers;
            result.slack = state.slack;
            result.lower_dual = state.lower_dual;
            result.upper_dual = state.upper_dual;
            result.objective = final_values.objective;
            result.constraint_inf = std::max(
                final_values.constraint_inf,
                final_metrics.primal_inf);
            result.dual_inf = final_metrics.dual_inf;
            result.complementarity_inf =
                final_metrics.complementarity_inf;
            result.converged = result.converged
                || (final_metrics.finite
                    && std::max(
                        final_metrics.primal_inf,
                        std::max(
                            final_metrics.dual_inf,
                            final_metrics.complementarity_inf))
                        <= config.tolerance);
            result.elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - start).count();
            return result;
        }

        ScopedSqpResult solve_sqp()
        {
            if (total_inequalities != 0)
                throw std::runtime_error(
                    "solve_sqp is equality-only; use solve_ipm for "
                    "path inequalities");
            ScopedSqpResult result;
            result.primal = primal_initial;
            result.multipliers = multiplier_initial;
            const auto start = Clock::now();
            for (Index iteration = 0;
                 iteration < config.max_iterations; ++iteration)
            {
                const ScopedNlpValues current = evaluate(
                    result.primal, result.multipliers);
                if (std::max(current.constraint_inf, current.dual_inf)
                    <= config.tolerance)
                {
                    result.converged = true;
                    break;
                }
                ScopedNewtonDirection direction = solve_direction(
                    result.primal,
                    result.multipliers,
                    config.dense_step_validation && iteration == 0);
                if (!direction.success)
                    break;
                result.kkt_ms += direction.elapsed_ms;
                result.linear_residual = std::max(
                    result.linear_residual, direction.residual_inf);
                if (iteration == 0)
                    result.dense_step_difference =
                        direction.dense_difference;

                const Scalar penalty = std::max<Scalar>(
                    10.0,
                    2.0 * max_abs(
                        result.multipliers + direction.multipliers));
                const Scalar current_merit = current.objective
                    + penalty * constraints_global.lpNorm<1>();
                Scalar directional_derivative =
                    objective_gradient_global.dot(direction.primal)
                    - penalty * constraints_global.lpNorm<1>();
                if (!(directional_derivative < 0.0))
                    directional_derivative =
                        -0.5 * direction.primal.squaredNorm()
                        - penalty * constraints_global.lpNorm<1>();

                Scalar alpha = 1.0;
                bool accepted = false;
                for (Index line_search = 0; line_search < 28; ++line_search)
                {
                    const Eigen::VectorXd trial =
                        result.primal + alpha * direction.primal;
                    const ScopedNlpValues trial_values = evaluate(
                        trial, result.multipliers);
                    ++result.line_search_evaluations;
                    const Scalar trial_merit = trial_values.objective
                        + penalty * constraints_global.lpNorm<1>();
                    if (std::isfinite(trial_merit)
                        && trial_merit <= current_merit
                            + 1e-4 * alpha * directional_derivative)
                    {
                        result.primal = trial;
                        result.multipliers +=
                            alpha * direction.multipliers;
                        accepted = true;
                        break;
                    }
                    alpha *= 0.5;
                }
                if (!accepted)
                    break;
                result.iterations = iteration + 1;
            }
            const ScopedNlpValues final = evaluate(
                result.primal, result.multipliers);
            result.objective = final.objective;
            result.constraint_inf = final.constraint_inf;
            result.dual_inf = final.dual_inf;
            result.converged = result.converged
                || std::max(final.constraint_inf, final.dual_inf)
                    <= config.tolerance;
            result.elapsed_ms =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - start).count();
            return result;
        }

        IntervalScopedCollocationConfig config;
        RadauCoefficients coefficients;
        std::vector<ScopedVariableBlock> blocks;
        std::vector<IntervalScope> scopes;
        std::vector<Index> block_offsets;
        std::vector<Index> scalar_to_block;
        std::vector<Index> scalar_to_local;
        std::vector<Index> phase_parameter_blocks;
        std::vector<Index> left_separator_blocks;
        std::vector<Index> right_separator_blocks;
        std::vector<std::vector<Index>> phase_active_blocks;
        std::vector<std::vector<Index>> phase_columns;
        Index total_scoped = 0;
        Index total_trajectory = 0;
        Index total_variables = 0;
        Index total_constraints = 0;
        Index total_equalities = 0;
        Index total_inequalities = 0;
        std::vector<Index> phase_variable_offsets;
        std::vector<Index> phase_constraint_offsets;
        std::vector<std::unique_ptr<PhaseData>> phases;
        std::unique_ptr<IntervalScopedImplicitOcpKktSolver> solver;
        std::unique_ptr<IntervalScopedImplicitPdSolver> pd_solver;

        std::vector<MatRealAllocated> diagonal_storage;
        std::vector<MatRealAllocated> edge_storage;
        std::vector<VecRealAllocated> rhs_storage;
        std::vector<VecRealAllocated> scoped_solution_storage;
        std::vector<MatRealView> diagonal_views;
        std::vector<MatRealView> edge_views;
        std::vector<VecRealView> rhs_views;
        std::vector<VecRealView> scoped_solution_views;
        std::vector<ImplicitOcpPhaseCondensingView> phase_views;
        std::vector<IntervalScopedImplicitPdPhaseView> pd_phase_views;

        Eigen::VectorXd primal_initial;
        Eigen::VectorXd multiplier_initial;
        Scalar objective_value = 0.0;
        Eigen::VectorXd objective_gradient_global;
        Eigen::VectorXd constraints_global;
        Eigen::VectorXd constraint_lower;
        Eigen::VectorXd constraint_upper;
        Eigen::VectorXd lagrangian_gradient_global;
        Eigen::VectorXd scoped_objective_gradient;
        Eigen::VectorXd scoped_lagrangian_gradient;
        Eigen::MatrixXd scoped_hessian;

        std::vector<std::pair<Index, Index>> jacobian_pattern;
        std::vector<JacobianSource> jacobian_sources;
        std::vector<Scalar> jacobian_values;
        std::vector<std::pair<Index, Index>> hessian_pattern;
        std::vector<HessianSource> hessian_sources;
        std::vector<Scalar> hessian_values;
    };

    IntervalScopedCollocationProblem::IntervalScopedCollocationProblem(
        IntervalScopedCollocationConfig config)
        : impl_(std::make_unique<Impl>(std::move(config)))
    {
    }

    IntervalScopedCollocationProblem::~IntervalScopedCollocationProblem() =
        default;
    IntervalScopedCollocationProblem::IntervalScopedCollocationProblem(
        IntervalScopedCollocationProblem &&) noexcept = default;
    IntervalScopedCollocationProblem &
    IntervalScopedCollocationProblem::operator=(
        IntervalScopedCollocationProblem &&) noexcept = default;

    const IntervalScopedCollocationConfig &
    IntervalScopedCollocationProblem::config() const noexcept
    {
        return impl_->config;
    }

    Index IntervalScopedCollocationProblem::number_of_phases() const noexcept
    {
        return static_cast<Index>(impl_->phases.size());
    }

    Index IntervalScopedCollocationProblem::number_of_variables() const noexcept
    {
        return impl_->total_variables;
    }

    Index IntervalScopedCollocationProblem::number_of_constraints() const noexcept
    {
        return impl_->total_constraints;
    }

    Index IntervalScopedCollocationProblem::number_of_equalities() const noexcept
    {
        return impl_->total_equalities;
    }

    Index IntervalScopedCollocationProblem::number_of_inequalities() const noexcept
    {
        return impl_->total_inequalities;
    }

    Index IntervalScopedCollocationProblem::number_of_scoped_variables() const noexcept
    {
        return impl_->total_scoped;
    }

    Index IntervalScopedCollocationProblem::number_of_trajectory_variables() const noexcept
    {
        return impl_->total_trajectory;
    }

    const Eigen::VectorXd &
    IntervalScopedCollocationProblem::constraint_lower_bounds() const noexcept
    {
        return impl_->constraint_lower;
    }

    const Eigen::VectorXd &
    IntervalScopedCollocationProblem::constraint_upper_bounds() const noexcept
    {
        return impl_->constraint_upper;
    }

    const std::vector<ScopedVariableBlock> &
    IntervalScopedCollocationProblem::blocks() const noexcept
    {
        return impl_->blocks;
    }

    const std::vector<IntervalScope> &
    IntervalScopedCollocationProblem::scopes() const noexcept
    {
        return impl_->scopes;
    }

    IntervalKktSymbolicStats
    IntervalScopedCollocationProblem::symbolic_stats() const noexcept
    {
        return impl_->solver->reduced_solver().symbolic_stats();
    }

    Eigen::VectorXd IntervalScopedCollocationProblem::initial_primal() const
    {
        return impl_->primal_initial;
    }

    Eigen::VectorXd IntervalScopedCollocationProblem::initial_multipliers() const
    {
        return impl_->multiplier_initial;
    }

    ScopedNlpValues IntervalScopedCollocationProblem::evaluate(
        const Eigen::VectorXd &primal,
        const Eigen::VectorXd &multipliers,
        const Scalar objective_factor)
    {
        return impl_->evaluate(
            primal, multipliers, objective_factor);
    }

    Scalar IntervalScopedCollocationProblem::objective() const noexcept
    {
        return impl_->objective_value;
    }

    const Eigen::VectorXd &
    IntervalScopedCollocationProblem::objective_gradient() const noexcept
    {
        return impl_->objective_gradient_global;
    }

    const Eigen::VectorXd &
    IntervalScopedCollocationProblem::constraints() const noexcept
    {
        return impl_->constraints_global;
    }

    const Eigen::VectorXd &
    IntervalScopedCollocationProblem::lagrangian_gradient() const noexcept
    {
        return impl_->lagrangian_gradient_global;
    }

    const std::vector<std::pair<Index, Index>> &
    IntervalScopedCollocationProblem::jacobian_pattern() const noexcept
    {
        return impl_->jacobian_pattern;
    }

    const std::vector<Scalar> &
    IntervalScopedCollocationProblem::jacobian_values() const noexcept
    {
        return impl_->jacobian_values;
    }

    const std::vector<std::pair<Index, Index>> &
    IntervalScopedCollocationProblem::hessian_pattern() const noexcept
    {
        return impl_->hessian_pattern;
    }

    const std::vector<Scalar> &
    IntervalScopedCollocationProblem::hessian_values() const noexcept
    {
        return impl_->hessian_values;
    }

    ScopedNewtonDirection
    IntervalScopedCollocationProblem::solve_direction(
        const Eigen::VectorXd &primal,
        const Eigen::VectorXd &multipliers,
        const bool validate_with_dense)
    {
        return impl_->solve_direction(
            primal, multipliers, validate_with_dense);
    }

    ScopedDerivativeCheck
    IntervalScopedCollocationProblem::check_derivatives(
        const Eigen::VectorXd &primal,
        const Scalar step)
    {
        return impl_->check_derivatives(primal, step);
    }

    ScopedSqpResult IntervalScopedCollocationProblem::solve_sqp()
    {
        return impl_->solve_sqp();
    }

    ScopedIpmResult IntervalScopedCollocationProblem::solve_ipm()
    {
        return impl_->solve_ipm();
    }
} // namespace fatrop::research
