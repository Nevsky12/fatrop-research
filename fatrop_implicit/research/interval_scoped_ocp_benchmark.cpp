#include "fatrop/ocp/interval_scoped_ocp_kkt_solver.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace fatrop;

namespace
{
    using Clock = std::chrono::steady_clock;

    struct Config
    {
        Index phases = 32;
        Index nodes = 4;
        Index nx = 4;
        Index nx_period = 3;
        Index nu = 2;
        Index nu_period = 2;
        Index phase_parameters = 2;
        Index interface_parameters = 1;
        Index segment_parameters = 2;
        Index segment_width = 4;
        Index segment_stride = 4;
        Index global_parameters = 2;
        Index repeats = 9;
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
            else if (option == "--nodes")
                config.nodes = next(option);
            else if (option == "--nx")
                config.nx = next(option);
            else if (option == "--nx-period")
                config.nx_period = next(option);
            else if (option == "--nu")
                config.nu = next(option);
            else if (option == "--nu-period")
                config.nu_period = next(option);
            else if (option == "--phase-parameters")
                config.phase_parameters = next(option);
            else if (option == "--interface-parameters")
                config.interface_parameters = next(option);
            else if (option == "--segment-parameters")
                config.segment_parameters = next(option);
            else if (option == "--segment-width")
                config.segment_width = next(option);
            else if (option == "--segment-stride")
                config.segment_stride = next(option);
            else if (option == "--global-parameters")
                config.global_parameters = next(option);
            else if (option == "--repeats")
                config.repeats = next(option);
            else if (option == "--help")
            {
                std::cout
                    << "Usage: interval_scoped_ocp_benchmark [options]\n"
                    << "  --phases N\n"
                    << "  --nodes N\n"
                    << "  --nx N\n"
                    << "  --nx-period N\n"
                    << "  --nu N\n"
                    << "  --nu-period N\n"
                    << "  --phase-parameters N\n"
                    << "  --interface-parameters N\n"
                    << "  --segment-parameters N\n"
                    << "  --segment-width N\n"
                    << "  --segment-stride N\n"
                    << "  --global-parameters N\n"
                    << "  --repeats N\n";
                std::exit(0);
            }
            else
                throw std::runtime_error("Unknown option: " + option);
        }

        if (config.phases < 1 || config.nodes < 2 || config.nx < 1
            || config.nx_period < 1 || config.nu < 0
            || config.nu_period < 1 || config.phase_parameters < 0
            || config.interface_parameters < 0
            || config.segment_parameters < 0 || config.segment_width < 1
            || config.segment_stride < 1 || config.global_parameters < 0
            || config.repeats < 1)
            throw std::runtime_error("Invalid benchmark configuration");
        if (config.phase_parameters == 0
            && config.interface_parameters == 0
            && config.segment_parameters == 0
            && config.global_parameters == 0)
            throw std::runtime_error(
                "At least one scoped parameter block is required");
        return config;
    }

    std::vector<Index> controls(const Index nodes, const Index nu)
    {
        std::vector<Index> result(static_cast<std::size_t>(nodes), nu);
        result.back() = 0;
        return result;
    }

    std::vector<Index> states(const Index nodes, const Index nx)
    {
        return std::vector<Index>(static_cast<std::size_t>(nodes), nx);
    }

    std::vector<Index> path_equalities(const Index nodes)
    {
        std::vector<Index> result(static_cast<std::size_t>(nodes), 0);
        result.front() = 1;
        if (nodes > 2)
            result.back() = 1;
        return result;
    }

    std::vector<Index> no_inequalities(const Index nodes)
    {
        return std::vector<Index>(static_cast<std::size_t>(nodes), 0);
    }

    std::vector<IntervalScope> make_scopes(const Config &config)
    {
        std::vector<IntervalScope> scopes;
        if (config.phase_parameters > 0)
        {
            for (Index phase = 0; phase < config.phases; ++phase)
                scopes.push_back(
                    {config.phase_parameters, phase, phase});
        }
        if (config.interface_parameters > 0)
        {
            for (Index phase = 0; phase + 1 < config.phases; ++phase)
                scopes.push_back(
                    {config.interface_parameters, phase, phase + 1});
        }
        if (config.segment_parameters > 0)
        {
            for (Index first = 0; first < config.phases;
                 first += config.segment_stride)
            {
                scopes.push_back(
                    {config.segment_parameters,
                     first,
                     std::min(
                         config.phases - 1,
                         first + config.segment_width - 1)});
            }
        }
        if (config.global_parameters > 0)
            scopes.push_back(
                {config.global_parameters, 0, config.phases - 1});
        return scopes;
    }

    std::vector<Index> block_offsets(
        const std::vector<IntervalScope> &scopes)
    {
        std::vector<Index> offsets(scopes.size(), 0);
        for (std::size_t block = 1; block < scopes.size(); ++block)
            offsets[block] = offsets[block - 1]
                           + scopes[block - 1].dimension;
        return offsets;
    }

    std::vector<Index> phase_columns(
        const std::vector<IntervalScope> &scopes,
        const std::vector<Index> &offsets,
        const Index phase)
    {
        std::vector<Index> columns;
        for (std::size_t block = 0; block < scopes.size(); ++block)
        {
            if (scopes[block].first_phase <= phase
                && phase <= scopes[block].last_phase)
            {
                for (Index column = 0;
                     column < scopes[block].dimension; ++column)
                    columns.push_back(offsets[block] + column);
            }
        }
        return columns;
    }

    Scalar cross_value(
        const Index phase, const Index row, const Index global_column,
        const Scalar scale)
    {
        return scale * std::sin(
            0.31 * static_cast<Scalar>(1 + phase)
          + 0.17 * static_cast<Scalar>(1 + row)
          + 0.13 * static_cast<Scalar>(1 + global_column));
    }

    struct ImplicitPhaseData
    {
        ImplicitPhaseData(
            const Index phase_value,
            const Index nodes,
            const Index nx_value,
            const Index nu_value,
            std::vector<Index> global_columns_value)
            : phase(phase_value),
              nx(nx_value),
              nu(nu_value),
              global_columns(std::move(global_columns_value)),
              dims(
                  nodes,
                  controls(nodes, nu),
                  states(nodes, nx),
                  path_equalities(nodes),
                  no_inequalities(nodes)),
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
                  static_cast<Index>(global_columns.size())),
              border_jacobian(
                  info.number_of_eq_constraints,
                  static_cast<Index>(global_columns.size())),
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
                  static_cast<Eigen::Index>(global_columns.size()))),
              dense_residual(Eigen::VectorXd::Zero(
                  info.number_of_primal_variables
                    + info.number_of_eq_constraints))
        {
            const Index primal_count = info.number_of_primal_variables;
            const Index constraint_count = info.number_of_eq_constraints;
            for (Index stage = 0; stage < dims.K; ++stage)
            {
                jacobian.Gg_eqt[stage] = 0.0;
                const Index variables =
                    dims.number_of_controls[stage]
                  + dims.number_of_states[stage];
                const Index offset = info.offsets_primal_u[stage];
                hessian.RSQrqt[stage] = 0.0;
                for (Index row = 0; row < variables; ++row)
                {
                    for (Index column = 0; column < variables; ++column)
                    {
                        const Scalar value = row == column
                            ? 3.8 + 0.04 * phase + 0.08 * stage
                                  + 0.015 * row
                            : 0.012
                                / (1.0 + std::abs(row - column));
                        hessian.RSQrqt[stage](row, column) = value;
                        dense_k(offset + row, offset + column) = value;
                    }
                    D_x(offset + row) =
                        0.14 + 0.006 * ((offset + row + phase) % 5);
                    dense_k(offset + row, offset + row) +=
                        D_x(offset + row);
                    rhs_primal(offset + row) =
                        0.025 * (1 + offset + row)
                      - 0.011 * ((row + phase) % 4);
                    dense_residual(offset + row) =
                        rhs_primal(offset + row);
                }

                if (dims.number_of_eq_constraints[stage] > 0)
                {
                    const Index path_row = info.offsets_g_eq_path[stage];
                    for (Index variable = 0;
                         variable < variables; ++variable)
                    {
                        const Scalar value =
                            0.045
                          + 0.007 * ((variable + stage + phase) % 5);
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
                            0.014
                          * (1 + ((variable + 2 * equation
                                  + stage + phase) % 7))
                          - 0.034;
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
                        const Scalar value = state == equation
                            ? 1.08 + 0.015 * stage + 0.003 * state
                            : 0.025
                                / (1.0 + std::abs(state - equation));
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
            for (Index row = 0; row < constraint_count; ++row)
            {
                rhs_constraints(row) =
                    -0.021 * (1 + row)
                  + 0.009 * ((row + phase) % 5);
                dense_residual(primal_count + row) =
                    rhs_constraints(row);
            }
            for (Index stage = 0; stage < dims.K; ++stage)
            {
                for (Index equation = 0;
                     equation < dims.number_of_eq_constraints[stage];
                     ++equation)
                {
                    const Index row =
                        info.offsets_g_eq_path[stage] + equation;
                    const Index diagonal =
                        info.offsets_eq[stage] + equation;
                    D_eq(diagonal) =
                        0.026 + 0.003 * ((row + phase) % 4);
                    dense_k(primal_count + row, primal_count + row) =
                        -D_eq(diagonal);
                }
            }

            D_s = 0.0;
            cross_hessian = 0.0;
            border_jacobian = 0.0;
            primal = 0.0;
            multipliers = 0.0;
            for (Index row = 0; row < primal_count; ++row)
            {
                for (Index column = 0;
                     column < static_cast<Index>(global_columns.size());
                     ++column)
                {
                    const Scalar value = cross_value(
                        phase, row,
                        global_columns[static_cast<std::size_t>(column)],
                        0.006);
                    cross_hessian(row, column) = value;
                    dense_b(row, column) = value;
                }
            }
            for (Index row = 0; row < constraint_count; ++row)
            {
                for (Index column = 0;
                     column < static_cast<Index>(global_columns.size());
                     ++column)
                {
                    const Scalar value = cross_value(
                        phase, primal_count + row,
                        global_columns[static_cast<std::size_t>(column)],
                        0.0045);
                    border_jacobian(row, column) = value;
                    dense_b(primal_count + row, column) = value;
                }
            }
        }

        Index phase;
        Index nx;
        Index nu;
        std::vector<Index> global_columns;
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
        Eigen::VectorXd dense_residual;
    };

    std::vector<ImplicitOcpPhaseCondensingView> make_views(
        std::vector<std::unique_ptr<ImplicitPhaseData>> &phases)
    {
        std::vector<ImplicitOcpPhaseCondensingView> views;
        views.reserve(phases.size());
        for (auto &phase : phases)
        {
            views.emplace_back(
                phase->info, phase->jacobian, phase->hessian,
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
                phase->primal.block(phase->primal.m(), 0),
                phase->multipliers.block(phase->multipliers.m(), 0));
        }
        return views;
    }

    struct PackedBaselineStatistics
    {
        Index incident_response_columns = 0;
        Index active_response_columns = 0;
        double trajectory_milliseconds = 0.0;
        double assembly_milliseconds = 0.0;
        double reduced_milliseconds = 0.0;
        double back_substitution_milliseconds = 0.0;
    };

    /**
     * Controlled baseline: use the identical phase condensation kernels, but
     * pack every scoped scalar into one dense global Schur block.  This is the
     * best-case version of the naive one-global-parameter-vector extension:
     * inactive zero response columns are never formed or scanned.
     */
    class PackedDenseOcpBaseline
    {
    public:
        PackedDenseOcpBaseline(
            const std::vector<const ProblemInfo *> &phase_info,
            std::vector<std::vector<Index>> local_to_global,
            const Index total_dimension)
            : local_to_global_(std::move(local_to_global)),
              total_dimension_(total_dimension),
              reduced_solver_(
                  static_cast<Index>(phase_info.size()),
                  std::vector<IntervalScope>{
                      {total_dimension, 0,
                       static_cast<Index>(phase_info.size()) - 1}}),
              packed_matrix_(total_dimension, total_dimension),
              packed_rhs_(total_dimension),
              packed_solution_(total_dimension)
        {
            kernels_.reserve(phase_info.size());
            local_solution_.reserve(phase_info.size());
            for (std::size_t phase = 0; phase < phase_info.size(); ++phase)
            {
                kernels_.push_back(
                    std::make_unique<ImplicitOcpPhaseCondensingKernel>(
                        *phase_info[phase],
                        static_cast<Index>(local_to_global_[phase].size())));
                local_solution_.emplace_back(
                    std::max<Index>(
                        static_cast<Index>(local_to_global_[phase].size()),
                        1));
            }
            reduced_solver_.set_pivot_tolerance(1e-14);
        }

        LinsolReturnFlag solve(
            const std::vector<ImplicitOcpPhaseCondensingView> &phases,
            const MatRealView &direct_matrix,
            const VecRealView &direct_rhs)
        {
            statistics_ = {};
            for (std::size_t phase = 0; phase < phases.size(); ++phase)
            {
                const auto start = Clock::now();
                const LinsolReturnFlag status =
                    kernels_[phase]->factor_and_condense(phases[phase]);
                statistics_.trajectory_milliseconds +=
                    elapsed_milliseconds(start);
                const auto &phase_statistics =
                    kernels_[phase]->last_statistics();
                statistics_.incident_response_columns +=
                    phase_statistics.incident_response_columns;
                statistics_.active_response_columns +=
                    phase_statistics.active_response_columns;
                if (status != LinsolReturnFlag::SUCCESS)
                    return status;
            }

            const auto assembly_start = Clock::now();
            for (Index row = 0; row < total_dimension_; ++row)
            {
                packed_rhs_(row) = -direct_rhs(row);
                for (Index column = 0;
                     column < total_dimension_; ++column)
                    packed_matrix_(row, column) =
                        direct_matrix(row, column);
            }
            for (std::size_t phase = 0; phase < phases.size(); ++phase)
            {
                const MatRealView condensed =
                    kernels_[phase]->condensed_matrix();
                const VecRealView condensed_rhs =
                    kernels_[phase]->condensed_rhs();
                const auto &mapping = local_to_global_[phase];
                for (Index row = 0;
                     row < static_cast<Index>(mapping.size()); ++row)
                {
                    const Index global_row =
                        mapping[static_cast<std::size_t>(row)];
                    packed_rhs_(global_row) -= condensed_rhs(row);
                    for (Index column = 0;
                         column < static_cast<Index>(mapping.size());
                         ++column)
                    {
                        packed_matrix_(
                            global_row,
                            mapping[static_cast<std::size_t>(column)])
                            += condensed(row, column);
                    }
                }
            }
            reduced_solver_.clear_numeric();
            reduced_solver_.add_matrix_block(
                0, 0,
                packed_matrix_.block(
                    total_dimension_, total_dimension_, 0, 0));
            reduced_solver_.add_rhs_block(
                0, packed_rhs_.block(total_dimension_, 0));
            statistics_.assembly_milliseconds =
                elapsed_milliseconds(assembly_start);

            const auto reduced_start = Clock::now();
            std::vector<VecRealView> solution_views{
                packed_solution_.block(total_dimension_, 0)};
            const LinsolReturnFlag status =
                reduced_solver_.factor_and_solve(solution_views);
            statistics_.reduced_milliseconds =
                elapsed_milliseconds(reduced_start);
            if (status != LinsolReturnFlag::SUCCESS)
                return status;

            const auto back_start = Clock::now();
            for (std::size_t phase = 0; phase < phases.size(); ++phase)
            {
                const auto &mapping = local_to_global_[phase];
                VecRealAllocated &local = local_solution_[phase];
                for (Index column = 0;
                     column < static_cast<Index>(mapping.size()); ++column)
                    local(column) = packed_solution_(
                        mapping[static_cast<std::size_t>(column)]);
                kernels_[phase]->back_substitute(
                    phases[phase],
                    local.block(static_cast<Index>(mapping.size()), 0));
            }
            statistics_.back_substitution_milliseconds =
                elapsed_milliseconds(back_start);
            return LinsolReturnFlag::SUCCESS;
        }

        void set_trajectory_lu_tolerance(const Scalar tolerance)
        {
            for (auto &kernel : kernels_)
                kernel->trajectory_solver().set_lu_fact_tol(tolerance);
        }

        Scalar solution(const Index row) const
        {
            return packed_solution_(row);
        }

        const PackedBaselineStatistics &statistics() const noexcept
        {
            return statistics_;
        }

    private:
        static double elapsed_milliseconds(
            const Clock::time_point start) noexcept
        {
            return std::chrono::duration<double, std::milli>(
                Clock::now() - start).count();
        }

        std::vector<std::vector<Index>> local_to_global_;
        Index total_dimension_ = 0;
        std::vector<std::unique_ptr<ImplicitOcpPhaseCondensingKernel>> kernels_;
        std::vector<VecRealAllocated> local_solution_;
        IntervalScopedKktSolver reduced_solver_;
        MatRealAllocated packed_matrix_;
        VecRealAllocated packed_rhs_;
        VecRealAllocated packed_solution_;
        PackedBaselineStatistics statistics_;
    };

    double median(std::vector<double> values)
    {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }

    template <typename Function>
    std::pair<double, bool> timed_call(Function &&function)
    {
        const auto start = Clock::now();
        const bool success = function();
        return {
            std::chrono::duration<double, std::micro>(
                Clock::now() - start).count(),
            success};
    }

    Eigen::VectorXd scoped_solution(
        const std::vector<IntervalScope> &scopes,
        const std::vector<Index> &offsets,
        const std::vector<VecRealAllocated> &solution)
    {
        Index total = 0;
        for (const auto &scope : scopes)
            total += scope.dimension;
        Eigen::VectorXd result(total);
        for (std::size_t block = 0; block < scopes.size(); ++block)
        {
            for (Index row = 0; row < scopes[block].dimension; ++row)
                result(offsets[block] + row) = solution[block](row);
        }
        return result;
    }

    double complete_kkt_residual(
        const std::vector<std::unique_ptr<ImplicitPhaseData>> &phases,
        const Eigen::VectorXd &parameters,
        const MatRealAllocated &direct_matrix,
        const VecRealAllocated &direct_rhs)
    {
        Eigen::VectorXd parameter_residual(parameters.size());
        for (Eigen::Index row = 0; row < parameters.size(); ++row)
        {
            Scalar value = direct_rhs(static_cast<Index>(row));
            for (Eigen::Index column = 0;
                 column < parameters.size(); ++column)
                value += direct_matrix(
                    static_cast<Index>(row),
                    static_cast<Index>(column)) * parameters(column);
            parameter_residual(row) = value;
        }

        double residual = 0.0;
        for (const auto &phase : phases)
        {
            Eigen::VectorXd trajectory(phase->dense_k.rows());
            for (Index row = 0;
                 row < phase->info.number_of_primal_variables; ++row)
                trajectory(row) = phase->primal(row);
            for (Index row = 0;
                 row < phase->info.number_of_eq_constraints; ++row)
            {
                trajectory(
                    phase->info.number_of_primal_variables + row) =
                    phase->multipliers(row);
            }

            Eigen::VectorXd local_parameters(
                static_cast<Eigen::Index>(phase->global_columns.size()));
            for (std::size_t column = 0;
                 column < phase->global_columns.size(); ++column)
                local_parameters(static_cast<Eigen::Index>(column)) =
                    parameters(phase->global_columns[column]);
            const Eigen::VectorXd local_residual =
                phase->dense_k * trajectory
              + phase->dense_b * local_parameters
              + phase->dense_residual;
            residual = std::max(
                residual,
                local_residual.lpNorm<Eigen::Infinity>());
            const Eigen::VectorXd contribution =
                phase->dense_b.transpose() * trajectory;
            for (std::size_t column = 0;
                 column < phase->global_columns.size(); ++column)
            {
                parameter_residual(phase->global_columns[column]) +=
                    contribution(static_cast<Eigen::Index>(column));
            }
        }
        return std::max(
            residual,
            parameter_residual.lpNorm<Eigen::Infinity>());
    }

    double trajectory_difference(
        const std::vector<std::unique_ptr<ImplicitPhaseData>> &first,
        const std::vector<std::unique_ptr<ImplicitPhaseData>> &second)
    {
        double difference = 0.0;
        for (std::size_t phase = 0; phase < first.size(); ++phase)
        {
            for (Index row = 0;
                 row < first[phase]->info.number_of_primal_variables; ++row)
                difference = std::max(
                    difference,
                    std::abs(
                        first[phase]->primal(row)
                      - second[phase]->primal(row)));
            for (Index row = 0;
                 row < first[phase]->info.number_of_eq_constraints; ++row)
                difference = std::max(
                    difference,
                    std::abs(
                        first[phase]->multipliers(row)
                      - second[phase]->multipliers(row)));
        }
        return difference;
    }
} // namespace

int main(int argc, char **argv)
try
{
    const Config config = parse_arguments(argc, argv);
    const std::vector<IntervalScope> scopes = make_scopes(config);
    const std::vector<Index> offsets = block_offsets(scopes);
    Index parameter_dimension = 0;
    for (const auto &scope : scopes)
        parameter_dimension += scope.dimension;

    std::vector<std::vector<Index>> local_to_global;
    local_to_global.reserve(static_cast<std::size_t>(config.phases));
    for (Index phase = 0; phase < config.phases; ++phase)
        local_to_global.push_back(
            phase_columns(scopes, offsets, phase));

    std::vector<std::unique_ptr<ImplicitPhaseData>> structured_phases;
    std::vector<std::unique_ptr<ImplicitPhaseData>> packed_phases;
    std::vector<const ProblemInfo *> structured_info;
    std::vector<const ProblemInfo *> packed_info;
    Index trajectory_dimension = 0;
    Index nx_min = std::numeric_limits<Index>::max();
    Index nx_max = 0;
    Index nu_min = std::numeric_limits<Index>::max();
    Index nu_max = 0;
    for (Index phase = 0; phase < config.phases; ++phase)
    {
        const Index nx = config.nx + phase % config.nx_period;
        const Index nu = config.nu + phase % config.nu_period;
        structured_phases.push_back(std::make_unique<ImplicitPhaseData>(
            phase, config.nodes, nx, nu,
            local_to_global[static_cast<std::size_t>(phase)]));
        packed_phases.push_back(std::make_unique<ImplicitPhaseData>(
            phase, config.nodes, nx, nu,
            local_to_global[static_cast<std::size_t>(phase)]));
        structured_info.push_back(&structured_phases.back()->info);
        packed_info.push_back(&packed_phases.back()->info);
        trajectory_dimension +=
            structured_phases.back()->info.number_of_primal_variables
          + structured_phases.back()->info.number_of_eq_constraints;
        nx_min = std::min(nx_min, nx);
        nx_max = std::max(nx_max, nx);
        nu_min = std::min(nu_min, nu);
        nu_max = std::max(nu_max, nu);
    }

    IntervalScopedImplicitOcpKktSolver structured_solver(
        structured_info, scopes);
    structured_solver.reduced_solver().set_pivot_tolerance(1e-14);
    for (Index phase = 0; phase < config.phases; ++phase)
        structured_solver.trajectory_solver(phase).set_lu_fact_tol(1e-12);
    PackedDenseOcpBaseline packed_solver(
        packed_info, local_to_global, parameter_dimension);
    packed_solver.set_trajectory_lu_tolerance(1e-12);

    std::vector<MatRealAllocated> diagonal;
    std::vector<VecRealAllocated> rhs;
    std::vector<VecRealAllocated> solution;
    diagonal.reserve(scopes.size());
    rhs.reserve(scopes.size());
    solution.reserve(scopes.size());
    MatRealAllocated direct_matrix(parameter_dimension, parameter_dimension);
    VecRealAllocated direct_rhs(parameter_dimension);
    direct_matrix = 0.0;
    direct_rhs = 0.0;
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
            const Index global_row = offsets[block] + row;
            rhs.back()(row) =
                0.018 * (1 + global_row)
              - 0.007 * ((static_cast<Index>(block) + row) % 5);
            direct_rhs(global_row) = rhs.back()(row);
            for (Index column = 0; column < dimension; ++column)
            {
                const Scalar value = row == column
                    ? 11.0 + 0.015 * global_row
                    : 0.003 / (1.0 + std::abs(row - column));
                diagonal.back()(row, column) = value;
                direct_matrix(
                    global_row,
                    offsets[block] + column) = value;
            }
        }
    }

    std::vector<MatRealAllocated> edge_storage;
    const auto &edges = structured_solver.reduced_solver().edges();
    edge_storage.reserve(edges.size());
    for (const IntervalKktEdge edge : edges)
    {
        const Index rows = scopes[static_cast<std::size_t>(
            edge.first_block)].dimension;
        const Index columns = scopes[static_cast<std::size_t>(
            edge.second_block)].dimension;
        edge_storage.emplace_back(rows, columns);
        for (Index row = 0; row < rows; ++row)
        {
            for (Index column = 0; column < columns; ++column)
            {
                const Scalar value = cross_value(
                    edge.first_block + edge.second_block,
                    row,
                    offsets[static_cast<std::size_t>(edge.second_block)]
                        + column,
                    0.0015);
                edge_storage.back()(row, column) = value;
                const Index global_row =
                    offsets[static_cast<std::size_t>(edge.first_block)] + row;
                const Index global_column =
                    offsets[static_cast<std::size_t>(edge.second_block)]
                    + column;
                direct_matrix(global_row, global_column) = value;
                direct_matrix(global_column, global_row) = value;
            }
        }
    }

    std::vector<MatRealView> diagonal_views;
    std::vector<VecRealView> rhs_views;
    std::vector<VecRealView> solution_views;
    for (std::size_t block = 0; block < scopes.size(); ++block)
    {
        const Index dimension = scopes[block].dimension;
        diagonal_views.push_back(
            diagonal[block].block(dimension, dimension, 0, 0));
        rhs_views.push_back(rhs[block].block(dimension, 0));
        solution_views.push_back(solution[block].block(dimension, 0));
    }
    std::vector<MatRealView> edge_views;
    edge_views.reserve(edges.size());
    for (std::size_t edge = 0; edge < edges.size(); ++edge)
    {
        edge_views.push_back(edge_storage[edge].block(
            scopes[static_cast<std::size_t>(
                edges[edge].first_block)].dimension,
            scopes[static_cast<std::size_t>(
                edges[edge].second_block)].dimension,
            0, 0));
    }

    std::vector<ImplicitOcpPhaseCondensingView> structured_views =
        make_views(structured_phases);
    std::vector<ImplicitOcpPhaseCondensingView> packed_views =
        make_views(packed_phases);
    const MatRealView direct_matrix_view = direct_matrix.block(
        parameter_dimension, parameter_dimension, 0, 0);
    const VecRealView direct_rhs_view = direct_rhs.block(
        parameter_dimension, 0);

    const auto run_structured = [&]()
    {
        return structured_solver.solve(
            structured_views,
            diagonal_views,
            edge_views,
            rhs_views,
            solution_views) == LinsolReturnFlag::SUCCESS;
    };
    const auto run_packed = [&]()
    {
        return packed_solver.solve(
            packed_views,
            direct_matrix_view,
            direct_rhs_view) == LinsolReturnFlag::SUCCESS;
    };

    if (!run_structured() || !run_packed()
        || !run_structured() || !run_packed())
        throw std::runtime_error("A warm-up solve failed");

    std::vector<double> structured_samples;
    std::vector<double> packed_samples;
    structured_samples.reserve(static_cast<std::size_t>(config.repeats));
    packed_samples.reserve(static_cast<std::size_t>(config.repeats));
    bool success = true;
    for (Index repeat = 0; repeat < config.repeats; ++repeat)
    {
        if (repeat % 2 == 0)
        {
            const auto structured = timed_call(run_structured);
            const auto packed = timed_call(run_packed);
            structured_samples.push_back(structured.first);
            packed_samples.push_back(packed.first);
            success = success && structured.second && packed.second;
        }
        else
        {
            const auto packed = timed_call(run_packed);
            const auto structured = timed_call(run_structured);
            structured_samples.push_back(structured.first);
            packed_samples.push_back(packed.first);
            success = success && structured.second && packed.second;
        }
    }

    const Eigen::VectorXd interval_parameters = scoped_solution(
        scopes, offsets, solution);
    Eigen::VectorXd packed_parameters(parameter_dimension);
    for (Index row = 0; row < parameter_dimension; ++row)
        packed_parameters(row) = packed_solver.solution(row);
    const double parameter_difference =
        (interval_parameters - packed_parameters)
            .lpNorm<Eigen::Infinity>();
    const double full_solution_difference = std::max(
        parameter_difference,
        trajectory_difference(structured_phases, packed_phases));
    const double residual = complete_kkt_residual(
        structured_phases,
        interval_parameters,
        direct_matrix,
        direct_rhs);

    const double structured_microseconds =
        median(std::move(structured_samples));
    const double packed_microseconds =
        median(std::move(packed_samples));
    const auto symbolic = structured_solver.reduced_solver().symbolic_stats();
    const auto &structured_statistics =
        structured_solver.last_statistics();
    const auto &packed_statistics = packed_solver.statistics();
    success = success
        && std::isfinite(full_solution_difference)
        && std::isfinite(residual)
        && full_solution_difference < 2e-7
        && residual < 2e-7;

    std::cout
        << "phases,nodes,nx_min,nx_max,nu_min,nu_max,blocks,"
           "parameter_dimension,trajectory_dimension,omega,edges,"
           "active_response_columns,interval_us,packed_global_us,speedup,"
           "interval_reduced_us,packed_reduced_us,packed_assembly_us,"
           "solution_difference,kkt_residual,success\n";
    std::cout
        << config.phases << ','
        << config.nodes << ','
        << nx_min << ','
        << nx_max << ','
        << nu_min << ','
        << nu_max << ','
        << scopes.size() << ','
        << parameter_dimension << ','
        << trajectory_dimension << ','
        << symbolic.maximum_active_dimension << ','
        << symbolic.number_of_edges << ','
        << structured_statistics.active_response_columns << ','
        << std::setprecision(12)
        << structured_microseconds << ','
        << packed_microseconds << ','
        << packed_microseconds / structured_microseconds << ','
        << 1000.0 * structured_statistics.reduced_solve_milliseconds << ','
        << 1000.0 * packed_statistics.reduced_milliseconds << ','
        << 1000.0 * packed_statistics.assembly_milliseconds << ','
        << full_solution_difference << ','
        << residual << ','
        << (success ? 1 : 0)
        << '\n';
    return success ? 0 : 2;
}
catch (const std::exception &error)
{
    std::cerr << "interval_scoped_ocp_benchmark: "
              << error.what() << '\n';
    return 1;
}
