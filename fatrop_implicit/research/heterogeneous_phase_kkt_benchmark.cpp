#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/ocp/aug_system_solver.hpp"
#include "fatrop/ocp/dims.hpp"
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/problem_info.hpp"
#include "fatrop/ocp/type.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using fatrop::AugSystemSolver;
using fatrop::Hessian;
using fatrop::Index;
using fatrop::Jacobian;
using fatrop::LinsolReturnFlag;
using fatrop::MatRealAllocated;
using fatrop::OcpType;
using fatrop::ProblemDims;
using fatrop::ProblemInfo;
using fatrop::Scalar;
using fatrop::VecRealAllocated;

namespace
{

struct Config
{
    std::vector<Index> phase_nodes{8, 7, 9};
    std::vector<Index> phase_nx{4, 7, 3};
    std::vector<Index> phase_nu{2, 3, 1};
    std::vector<Index> phase_parameters{1, 2, 1};
    Index global_parameters = 2;
    Index collocation_degree = 0;
    Index repeats = 5;
    bool dense_validation = true;
};

std::vector<Index> parse_index_list(
    const std::string &text, const std::string &option)
{
    std::vector<Index> values;
    std::size_t begin = 0;
    while (begin <= text.size())
    {
        const std::size_t end = text.find_first_of(",:", begin);
        const std::string token =
            text.substr(begin, end == std::string::npos
                                   ? std::string::npos
                                   : end - begin);
        if (token.empty())
            throw std::runtime_error(option + " contains an empty entry");
        const long value = std::stol(token);
        if (value < 0 ||
            value > std::numeric_limits<Index>::max())
            throw std::runtime_error(option + " contains an invalid value");
        values.push_back(static_cast<Index>(value));
        if (end == std::string::npos)
            break;
        begin = end + 1;
    }
    return values;
}

std::string encode_list(const std::vector<Index> &values)
{
    std::string encoded;
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (i > 0)
            encoded += ':';
        encoded += std::to_string(values[i]);
    }
    return encoded;
}

Config parse_arguments(int argc, char **argv)
{
    Config config;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument(argv[i]);
        const auto next = [&](const std::string &option) {
            if (i + 1 >= argc)
                throw std::runtime_error(option + " requires a value");
            return std::string(argv[++i]);
        };
        const auto next_integer = [&](const std::string &option) {
            return static_cast<Index>(std::stol(next(option)));
        };

        if (argument == "--phase-nodes")
            config.phase_nodes =
                parse_index_list(next(argument), argument);
        else if (argument == "--phase-nx")
            config.phase_nx =
                parse_index_list(next(argument), argument);
        else if (argument == "--phase-nu")
            config.phase_nu =
                parse_index_list(next(argument), argument);
        else if (argument == "--phase-parameters")
            config.phase_parameters =
                parse_index_list(next(argument), argument);
        else if (argument == "--global-parameters")
            config.global_parameters = next_integer(argument);
        else if (argument == "--collocation-degree")
            config.collocation_degree = next_integer(argument);
        else if (argument == "--repeats")
            config.repeats = next_integer(argument);
        else if (argument == "--no-dense-validation")
            config.dense_validation = false;
        else if (argument == "--help")
        {
            std::cout
                << "Usage: heterogeneous_phase_kkt_benchmark [options]\n"
                << "  --phase-nodes LIST       e.g. 8,7,9\n"
                << "  --phase-nx LIST          e.g. 4,7,3\n"
                << "  --phase-nu LIST          e.g. 2,3,1\n"
                << "  --phase-parameters LIST  e.g. 1,2,1\n"
                << "  --global-parameters N\n"
                << "  --collocation-degree {0|1|2|3}\n"
                << "  --repeats N\n"
                << "  --no-dense-validation\n";
            std::exit(0);
        }
        else
            throw std::runtime_error("Unknown option: " + argument);
    }

    const std::size_t phases = config.phase_nodes.size();
    if (phases == 0 ||
        config.phase_nx.size() != phases ||
        config.phase_nu.size() != phases ||
        config.phase_parameters.size() != phases)
    {
        throw std::runtime_error(
            "All phase lists must have the same nonzero length");
    }
    if (config.global_parameters < 0 || config.repeats < 1 ||
        config.collocation_degree < 0 ||
        config.collocation_degree > 3)
    {
        throw std::runtime_error("Invalid scalar configuration value");
    }
    for (std::size_t phase = 0; phase < phases; ++phase)
    {
        if (config.phase_nodes[phase] < 2 ||
            config.phase_nx[phase] < 1 ||
            config.phase_nu[phase] < 0 ||
            config.phase_parameters[phase] < 0)
        {
            throw std::runtime_error(
                "Each phase needs nodes>=2, nx>=1, nu>=0, p>=0");
        }
    }
    return config;
}

struct PhaseLayout
{
    explicit PhaseLayout(const Config &config)
        : phase_first(config.phase_nodes.size()),
          phase_last(config.phase_nodes.size()),
          local_parameter_offset(config.phase_nodes.size())
    {
        Index stage = 0;
        Index parameter_offset = config.global_parameters;
        for (Index phase = 0;
             phase < static_cast<Index>(config.phase_nodes.size());
             ++phase)
        {
            phase_first[phase] = stage;
            local_parameter_offset[phase] = parameter_offset;
            for (Index local = 0;
                 local < config.phase_nodes[phase]; ++local)
            {
                stage_phase.push_back(phase);
                stage_local.push_back(local);
                ++stage;
            }
            phase_last[phase] = stage - 1;
            parameter_offset += config.phase_parameters[phase];
        }
        stages = stage;
        parameters = parameter_offset;
    }

    bool is_phase_end(Index stage) const
    {
        return phase_last[stage_phase[stage]] == stage;
    }

    bool is_linkage(Index transition) const
    {
        return stage_phase[transition] != stage_phase[transition + 1];
    }

    Index stages = 0;
    Index parameters = 0;
    std::vector<Index> stage_phase;
    std::vector<Index> stage_local;
    std::vector<Index> phase_first;
    std::vector<Index> phase_last;
    std::vector<Index> local_parameter_offset;
};

struct CollocationCoefficients
{
    explicit CollocationCoefficients(Index degree)
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
            std::vector<double> polynomial{1.0};
            for (Index other = 0; other <= degree; ++other)
            {
                if (other == basis)
                    continue;
                const double denominator =
                    tau[static_cast<std::size_t>(basis)] -
                    tau[static_cast<std::size_t>(other)];
                std::vector<double> product(
                    polynomial.size() + 1, 0.0);
                for (std::size_t power = 0;
                     power < polynomial.size(); ++power)
                {
                    product[power] -=
                        polynomial[power] *
                        tau[static_cast<std::size_t>(other)] /
                        denominator;
                    product[power + 1] +=
                        polynomial[power] / denominator;
                }
                polynomial = std::move(product);
            }
            for (Index point = 0; point <= degree; ++point)
            {
                const double evaluation =
                    tau[static_cast<std::size_t>(point)];
                double derivative = 0.0;
                for (std::size_t power = 1;
                     power < polynomial.size(); ++power)
                {
                    derivative +=
                        static_cast<double>(power) *
                        polynomial[power] *
                        std::pow(
                            evaluation,
                            static_cast<int>(power - 1));
                }
                C(basis, point) = derivative;
            }
            for (double coefficient : polynomial)
                D(basis) += coefficient;
        }
    }

    std::vector<double> tau;
    Eigen::MatrixXd C;
    Eigen::VectorXd D;
};

double deterministic_value(Index i, Index j, double scale)
{
    return scale *
           std::sin(
               0.173 * static_cast<double>(i + 1) +
               0.317 * static_cast<double>(j + 1));
}

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 0)
        return 0.5 * (values[middle - 1] + values[middle]);
    return values[middle];
}

double max_abs(const Eigen::VectorXd &value)
{
    return value.size() == 0
               ? 0.0
               : value.cwiseAbs().maxCoeff();
}

Eigen::VectorXd to_eigen(const VecRealAllocated &value)
{
    Eigen::VectorXd result(value.m());
    for (Index i = 0; i < value.m(); ++i)
        result(i) = value(i);
    return result;
}

struct Problem
{
    explicit Problem(const Config &config)
        : config(config),
          layout(config),
          controls(static_cast<std::size_t>(layout.stages)),
          states(static_cast<std::size_t>(layout.stages)),
          equalities(static_cast<std::size_t>(layout.stages)),
          inequalities(static_cast<std::size_t>(layout.stages), 0),
          dims(make_dims()),
          info(dims),
          jacobian(dims),
          hessian(dims),
          D_x(info.number_of_primal_variables),
          D_s(info.number_of_slack_variables),
          rhs_x(info.number_of_primal_variables),
          rhs_g(info.number_of_eq_constraints),
          border_primal(
              info.number_of_primal_variables, layout.parameters),
          border_constraints(
              info.number_of_eq_constraints, layout.parameters),
          border_hessian(layout.parameters, layout.parameters),
          rhs_parameters(layout.parameters)
    {
        D_x = 0.0;
        D_s = 0.0;
        rhs_x = 0.0;
        rhs_g = 0.0;
        border_primal.setZero();
        border_constraints.setZero();
        border_hessian.setZero();
        rhs_parameters.setZero();

        populate_objective();
        if (config.collocation_degree > 0)
            populate_collocation();
        else
            populate_shooting();
        populate_parameter_objective();
    }

    ProblemDims make_dims()
    {
        for (Index stage = 0; stage < layout.stages; ++stage)
        {
            const Index phase = layout.stage_phase[stage];
            const Index nx = config.phase_nx[phase];
            states[stage] = nx;
            if (layout.is_phase_end(stage))
            {
                controls[stage] = 0;
                equalities[stage] = 0;
            }
            else
            {
                controls[stage] =
                    config.phase_nu[phase] +
                    config.collocation_degree * nx;
                equalities[stage] =
                    config.collocation_degree * nx;
            }
        }
        return ProblemDims(
            layout.stages, controls, states,
            equalities, inequalities);
    }

    std::vector<Index> active_parameters(Index phase) const
    {
        std::vector<Index> active;
        active.reserve(
            static_cast<std::size_t>(
                config.global_parameters +
                config.phase_parameters[phase]));
        for (Index parameter = 0;
             parameter < config.global_parameters; ++parameter)
            active.push_back(parameter);
        const Index offset = layout.local_parameter_offset[phase];
        for (Index parameter = 0;
             parameter < config.phase_parameters[phase]; ++parameter)
            active.push_back(offset + parameter);
        return active;
    }

    void populate_objective()
    {
        for (Index stage = 0; stage < layout.stages; ++stage)
        {
            const Index phase = layout.stage_phase[stage];
            const Index local_variables =
                controls[stage] + states[stage];
            hessian.RSQrqt[stage] = 0.0;
            const auto active = active_parameters(phase);
            for (Index row = 0; row < local_variables; ++row)
            {
                hessian.RSQrqt[stage](row, row) =
                    1.4 +
                    0.025 * static_cast<double>(
                                (row + 3 * stage + phase) % 13);
                if (row + 1 < local_variables)
                {
                    const double coupling =
                        0.012 /
                        (1.0 + static_cast<double>(row));
                    hessian.RSQrqt[stage](row, row + 1) =
                        coupling;
                    hessian.RSQrqt[stage](row + 1, row) =
                        coupling;
                }
                const Index global_row =
                    info.offsets_primal_u[stage] + row;
                rhs_x(global_row) =
                    deterministic_value(global_row, phase, 0.12);
                for (Index parameter : active)
                {
                    border_primal(global_row, parameter) =
                        deterministic_value(
                            global_row, parameter + phase, 0.007);
                }
            }
        }
    }

    void populate_linkage(Index stage)
    {
        const Index phase = layout.stage_phase[stage];
        const Index next_phase = phase + 1;
        const Index nx = config.phase_nx[phase];
        const Index nx_next = config.phase_nx[next_phase];
        jacobian.BAbt[stage] = 0.0;

        for (Index row = 0; row < nx; ++row)
        {
            for (Index equation = 0;
                 equation < nx_next; ++equation)
            {
                const double diagonal =
                    row == equation
                        ? 0.82 +
                              0.015 * static_cast<double>(
                                          (phase + row) % 5)
                        : 0.0;
                const double dense_tail =
                    0.018 *
                    std::cos(
                        0.11 * static_cast<double>(
                                   (row + 1) * (equation + 2)) +
                        0.07 * static_cast<double>(phase + 1));
                jacobian.BAbt[stage](row, equation) =
                    diagonal + dense_tail;
            }
        }

        const Index constraint_offset =
            info.offsets_g_eq_dyn[stage];
        for (Index equation = 0; equation < nx_next; ++equation)
        {
            const Index constraint =
                constraint_offset + equation;
            rhs_g(constraint) =
                deterministic_value(
                    constraint, phase + 9, 0.045);
            for (Index parameter = 0;
                 parameter < config.global_parameters; ++parameter)
            {
                border_constraints(constraint, parameter) =
                    deterministic_value(
                        equation + phase, parameter, 0.028);
            }
            for (Index local = 0;
                 local < config.phase_parameters[phase]; ++local)
            {
                const Index parameter =
                    layout.local_parameter_offset[phase] + local;
                border_constraints(constraint, parameter) =
                    deterministic_value(
                        equation, local + phase, 0.035);
            }
            for (Index local = 0;
                 local < config.phase_parameters[next_phase]; ++local)
            {
                const Index parameter =
                    layout.local_parameter_offset[next_phase] + local;
                border_constraints(constraint, parameter) =
                    deterministic_value(
                        equation + 3, local + next_phase, -0.031);
            }
        }
    }

    void populate_shooting()
    {
        for (Index stage = 0;
             stage + 1 < layout.stages; ++stage)
        {
            if (layout.is_linkage(stage))
            {
                populate_linkage(stage);
                continue;
            }
            const Index phase = layout.stage_phase[stage];
            const Index nx = config.phase_nx[phase];
            const Index nu = config.phase_nu[phase];
            jacobian.BAbt[stage] = 0.0;
            for (Index equation = 0; equation < nx; ++equation)
            {
                for (Index control = 0; control < nu; ++control)
                {
                    jacobian.BAbt[stage](control, equation) =
                        0.045 *
                        std::cos(
                            0.13 * static_cast<double>(
                                       (control + 1) *
                                       (equation + 2)) +
                            0.09 * static_cast<double>(phase));
                }
                for (Index state = 0; state < nx; ++state)
                {
                    const double diagonal =
                        state == equation
                            ? 0.90 -
                                  0.012 * static_cast<double>(phase % 4)
                            : 0.0;
                    const double neighbour =
                        std::abs(state - equation) == 1
                            ? 0.017
                            : 0.0;
                    jacobian.BAbt[stage](nu + state, equation) =
                        diagonal + neighbour;
                }
                const Index constraint =
                    info.offsets_g_eq_dyn[stage] + equation;
                rhs_g(constraint) =
                    deterministic_value(constraint, phase, 0.07);
                for (Index parameter :
                     active_parameters(phase))
                {
                    border_constraints(constraint, parameter) =
                        deterministic_value(
                            constraint, parameter + phase, 0.021);
                }
            }
        }
    }

    void populate_collocation()
    {
        const CollocationCoefficients coefficients(
            config.collocation_degree);
        for (Index stage = 0;
             stage + 1 < layout.stages; ++stage)
        {
            if (layout.is_linkage(stage))
            {
                populate_linkage(stage);
                continue;
            }

            const Index phase = layout.stage_phase[stage];
            const Index nx = config.phase_nx[phase];
            const Index nu = config.phase_nu[phase];
            const Index local_nu = controls[stage];
            const double step =
                0.035 *
                (1.0 + 0.10 * static_cast<double>(phase % 5));
            Eigen::MatrixXd mass(nx, nx);
            Eigen::MatrixXd system(nx, nx);
            Eigen::MatrixXd input(nx, nu);
            mass.setZero();
            system.setZero();
            input.setZero();
            for (Index row = 0; row < nx; ++row)
            {
                for (Index column = 0; column < nx; ++column)
                {
                    mass(row, column) =
                        row == column
                            ? 1.0 +
                                  0.02 * static_cast<double>(
                                             (row + phase) % 7)
                            : (std::abs(row - column) == 1
                                   ? 0.015
                                   : 0.0);
                    system(row, column) =
                        row == column
                            ? -0.40 -
                                  0.014 * static_cast<double>(
                                              (row + 2 * phase) % 9)
                            : (std::abs(row - column) == 1
                                   ? 0.048
                                   : 0.0);
                }
                for (Index control = 0; control < nu; ++control)
                {
                    input(row, control) =
                        0.11 *
                        std::cos(
                            0.12 * static_cast<double>(
                                       (row + 1) * (control + 2)) +
                            0.08 * static_cast<double>(phase));
                }
            }

            jacobian.BAbt[stage] = 0.0;
            jacobian.Gg_eqt[stage] = 0.0;
            for (Index state = 0; state < nx; ++state)
            {
                for (Index basis = 0;
                     basis <= config.collocation_degree; ++basis)
                {
                    const Index variable =
                        basis == 0
                            ? local_nu + state
                            : nu + (basis - 1) * nx + state;
                    jacobian.BAbt[stage](variable, state) =
                        coefficients.D(basis);
                }
            }

            const auto active = active_parameters(phase);
            for (Index point = 1;
                 point <= config.collocation_degree; ++point)
            {
                for (Index equation = 0; equation < nx; ++equation)
                {
                    const Index local_constraint =
                        (point - 1) * nx + equation;
                    const Index global_constraint =
                        info.offsets_g_eq_path[stage] +
                        local_constraint;
                    for (Index control = 0;
                         control < nu; ++control)
                    {
                        jacobian.Gg_eqt[stage](
                            control, local_constraint) =
                            -step * input(equation, control);
                    }
                    for (Index basis = 0;
                         basis <= config.collocation_degree; ++basis)
                    {
                        for (Index state = 0; state < nx; ++state)
                        {
                            const Index variable =
                                basis == 0
                                    ? local_nu + state
                                    : nu + (basis - 1) * nx + state;
                            double coefficient =
                                coefficients.C(basis, point) *
                                mass(equation, state);
                            if (basis == point)
                                coefficient -=
                                    step * system(equation, state);
                            jacobian.Gg_eqt[stage](
                                variable, local_constraint) =
                                coefficient;
                        }
                    }
                    for (Index parameter : active)
                    {
                        border_constraints(
                            global_constraint, parameter) =
                            -step *
                            deterministic_value(
                                equation + phase,
                                parameter + point, 0.075);
                    }
                    rhs_g(global_constraint) =
                        -step *
                        deterministic_value(
                            stage * nx + equation,
                            point + phase, 0.032);
                }
            }

            const Index dynamics_offset =
                info.offsets_g_eq_dyn[stage];
            for (Index state = 0; state < nx; ++state)
            {
                rhs_g(dynamics_offset + state) =
                    deterministic_value(
                        dynamics_offset + state, phase, 0.014);
            }
        }
    }

    void populate_parameter_objective()
    {
        const double horizon_scale =
            static_cast<double>(layout.stages);
        for (Index parameter = 0;
             parameter < layout.parameters; ++parameter)
        {
            rhs_parameters(parameter) =
                deterministic_value(
                    parameter, layout.stages, 0.11);
            border_hessian(parameter, parameter) =
                2.0 + 0.08 * horizon_scale +
                0.04 * static_cast<double>(parameter);
        }

        for (Index row = 0;
             row < config.global_parameters; ++row)
        {
            for (Index column = 0;
                 column < config.global_parameters; ++column)
            {
                if (row != column)
                {
                    border_hessian(row, column) =
                        0.006 /
                        (1.0 + std::abs(row - column));
                }
            }
        }

        for (Index phase = 0;
             phase < static_cast<Index>(
                         config.phase_nodes.size());
             ++phase)
        {
            const Index offset =
                layout.local_parameter_offset[phase];
            const Index count =
                config.phase_parameters[phase];
            for (Index local = 0; local < count; ++local)
            {
                const Index parameter = offset + local;
                for (Index global = 0;
                     global < config.global_parameters; ++global)
                {
                    const double coupling =
                        0.005 /
                        (1.0 +
                         static_cast<double>(
                             global + local + phase));
                    border_hessian(global, parameter) =
                        coupling;
                    border_hessian(parameter, global) =
                        coupling;
                }
                for (Index other = 0;
                     other < count; ++other)
                {
                    if (local != other)
                    {
                        border_hessian(
                            parameter, offset + other) =
                            0.006 /
                            (1.0 +
                             std::abs(local - other));
                    }
                }
            }
        }
    }

    Config config;
    PhaseLayout layout;
    std::vector<Index> controls;
    std::vector<Index> states;
    std::vector<Index> equalities;
    std::vector<Index> inequalities;
    ProblemDims dims;
    ProblemInfo info;
    Jacobian<OcpType> jacobian;
    Hessian<OcpType> hessian;
    VecRealAllocated D_x;
    VecRealAllocated D_s;
    VecRealAllocated rhs_x;
    VecRealAllocated rhs_g;
    Eigen::MatrixXd border_primal;
    Eigen::MatrixXd border_constraints;
    Eigen::MatrixXd border_hessian;
    Eigen::VectorXd rhs_parameters;
};

struct Result
{
    Eigen::VectorXd primal;
    Eigen::VectorXd multipliers;
    Eigen::VectorXd parameters;
    double elapsed_ms = 0.0;
    double factor_ms = 0.0;
    double parameter_rhs_ms = 0.0;
};

Result solve_bordered(
    Problem &problem, AugSystemSolver<OcpType> &solver,
    bool blocked)
{
    VecRealAllocated primal(
        problem.info.number_of_primal_variables);
    VecRealAllocated multipliers(
        problem.info.number_of_eq_constraints);
    primal = 0.0;
    multipliers = 0.0;

    const auto start = std::chrono::steady_clock::now();
    const auto factor_status = solver.solve(
        problem.info, problem.jacobian, problem.hessian,
        problem.D_x, problem.D_s,
        problem.rhs_x, problem.rhs_g,
        primal, multipliers);
    if (factor_status != LinsolReturnFlag::SUCCESS)
        throw std::runtime_error("FATROP factorization failed");
    const auto after_factor = std::chrono::steady_clock::now();

    Result result;
    const Eigen::VectorXd base_primal = to_eigen(primal);
    const Eigen::VectorXd base_multipliers = to_eigen(multipliers);
    const Index parameters = problem.layout.parameters;
    if (parameters == 0)
    {
        result.primal = base_primal;
        result.multipliers = base_multipliers;
        result.parameters.resize(0);
        const auto stop = std::chrono::steady_clock::now();
        result.elapsed_ms =
            std::chrono::duration<double, std::milli>(
                stop - start)
                .count();
        result.factor_ms =
            std::chrono::duration<double, std::milli>(
                after_factor - start)
                .count();
        return result;
    }

    Eigen::MatrixXd response_primal(
        problem.info.number_of_primal_variables, parameters);
    Eigen::MatrixXd response_multipliers(
        problem.info.number_of_eq_constraints, parameters);
    if (blocked)
    {
        MatRealAllocated batch_f(
            problem.info.number_of_primal_variables, parameters);
        MatRealAllocated batch_g(
            problem.info.number_of_eq_constraints, parameters);
        MatRealAllocated batch_x(
            problem.info.number_of_primal_variables, parameters);
        MatRealAllocated batch_multipliers(
            problem.info.number_of_eq_constraints, parameters);
        for (Index column = 0; column < parameters; ++column)
        {
            for (Index row = 0;
                 row < problem.info.number_of_primal_variables; ++row)
                batch_f(row, column) =
                    problem.border_primal(row, column);
            for (Index row = 0;
                 row < problem.info.number_of_eq_constraints; ++row)
                batch_g(row, column) =
                    problem.border_constraints(row, column);
        }
        batch_x = 0.0;
        batch_multipliers = 0.0;
        const auto status = solver.solve_rhs_batch(
            problem.info, problem.jacobian, problem.hessian,
            problem.D_s, batch_f, batch_g,
            batch_x, batch_multipliers);
        if (status != LinsolReturnFlag::SUCCESS)
            throw std::runtime_error("Blocked RHS solve failed");
        for (Index column = 0; column < parameters; ++column)
        {
            for (Index row = 0;
                 row < problem.info.number_of_primal_variables; ++row)
                response_primal(row, column) =
                    batch_x(row, column);
            for (Index row = 0;
                 row < problem.info.number_of_eq_constraints; ++row)
                response_multipliers(row, column) =
                    batch_multipliers(row, column);
        }
    }
    else
    {
        VecRealAllocated column_f(
            problem.info.number_of_primal_variables);
        VecRealAllocated column_g(
            problem.info.number_of_eq_constraints);
        VecRealAllocated column_x(
            problem.info.number_of_primal_variables);
        VecRealAllocated column_multipliers(
            problem.info.number_of_eq_constraints);
        for (Index column = 0; column < parameters; ++column)
        {
            for (Index row = 0;
                 row < problem.info.number_of_primal_variables; ++row)
                column_f(row) =
                    problem.border_primal(row, column);
            for (Index row = 0;
                 row < problem.info.number_of_eq_constraints; ++row)
                column_g(row) =
                    problem.border_constraints(row, column);
            column_x = 0.0;
            column_multipliers = 0.0;
            const auto status = solver.solve_rhs(
                problem.info, problem.jacobian, problem.hessian,
                problem.D_s, column_f, column_g,
                column_x, column_multipliers);
            if (status != LinsolReturnFlag::SUCCESS)
                throw std::runtime_error("Sequential RHS solve failed");
            response_primal.col(column) = to_eigen(column_x);
            response_multipliers.col(column) =
                to_eigen(column_multipliers);
        }
    }
    const auto after_parameter_rhs =
        std::chrono::steady_clock::now();

    const Eigen::MatrixXd schur =
        problem.border_hessian +
        problem.border_primal.transpose() * response_primal +
        problem.border_constraints.transpose() *
            response_multipliers;
    const Eigen::VectorXd schur_rhs =
        -problem.rhs_parameters -
        problem.border_primal.transpose() * base_primal -
        problem.border_constraints.transpose() *
            base_multipliers;
    const Eigen::LDLT<Eigen::MatrixXd> ldlt(schur);
    if (ldlt.info() != Eigen::Success)
        throw std::runtime_error("Static-parameter Schur factorization failed");
    result.parameters = ldlt.solve(schur_rhs);
    if (ldlt.info() != Eigen::Success)
        throw std::runtime_error("Static-parameter Schur solve failed");
    result.primal =
        base_primal + response_primal * result.parameters;
    result.multipliers =
        base_multipliers +
        response_multipliers * result.parameters;

    const auto stop = std::chrono::steady_clock::now();
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(
            stop - start)
            .count();
    result.factor_ms =
        std::chrono::duration<double, std::milli>(
            after_factor - start)
            .count();
    result.parameter_rhs_ms =
        std::chrono::duration<double, std::milli>(
            after_parameter_rhs - after_factor)
            .count();
    return result;
}

double structured_residual(
    const Problem &problem, const Result &result)
{
    Eigen::VectorXd stationarity(
        problem.info.number_of_primal_variables);
    Eigen::VectorXd constraints(
        problem.info.number_of_eq_constraints);
    for (Index row = 0;
         row < problem.info.number_of_primal_variables; ++row)
        stationarity(row) = problem.rhs_x(row);
    for (Index row = 0;
         row < problem.info.number_of_eq_constraints; ++row)
        constraints(row) = problem.rhs_g(row);
    stationarity +=
        problem.border_primal * result.parameters;
    constraints +=
        problem.border_constraints * result.parameters;
    Eigen::VectorXd parameter_stationarity =
        problem.rhs_parameters +
        problem.border_hessian * result.parameters +
        problem.border_primal.transpose() * result.primal +
        problem.border_constraints.transpose() *
            result.multipliers;

    for (Index stage = 0;
         stage < problem.layout.stages; ++stage)
    {
        const Index variables =
            problem.controls[stage] + problem.states[stage];
        const Index variable_offset =
            problem.info.offsets_primal_u[stage];
        for (Index row = 0; row < variables; ++row)
        {
            for (Index column = 0; column < variables; ++column)
            {
                stationarity(variable_offset + row) +=
                    problem.hessian.RSQrqt[stage](row, column) *
                    result.primal(variable_offset + column);
            }
        }

        const Index path_constraints =
            problem.equalities[stage];
        const Index path_offset =
            problem.info.offsets_g_eq_path[stage];
        for (Index equation = 0;
             equation < path_constraints; ++equation)
        {
            const double multiplier =
                result.multipliers(path_offset + equation);
            for (Index variable = 0;
                 variable < variables; ++variable)
            {
                const double derivative =
                    problem.jacobian.Gg_eqt[stage](
                        variable, equation);
                constraints(path_offset + equation) +=
                    derivative *
                    result.primal(variable_offset + variable);
                stationarity(variable_offset + variable) +=
                    derivative * multiplier;
            }
        }

        if (stage + 1 < problem.layout.stages)
        {
            const Index nx_next =
                problem.states[stage + 1];
            const Index dynamics_offset =
                problem.info.offsets_g_eq_dyn[stage];
            const Index next_state_offset =
                problem.info.offsets_primal_x[stage + 1];
            for (Index equation = 0;
                 equation < nx_next; ++equation)
            {
                const double multiplier =
                    result.multipliers(
                        dynamics_offset + equation);
                for (Index variable = 0;
                     variable < variables; ++variable)
                {
                    const double derivative =
                        problem.jacobian.BAbt[stage](
                            variable, equation);
                    constraints(dynamics_offset + equation) +=
                        derivative *
                        result.primal(variable_offset + variable);
                    stationarity(variable_offset + variable) +=
                        derivative * multiplier;
                }
                constraints(dynamics_offset + equation) -=
                    result.primal(next_state_offset + equation);
                stationarity(next_state_offset + equation) -=
                    multiplier;
            }
        }
    }

    return std::max(
        {max_abs(stationarity),
         max_abs(constraints),
         max_abs(parameter_stationarity)});
}

Eigen::VectorXd dense_reference(const Problem &problem)
{
    const Index primal = problem.info.number_of_primal_variables;
    const Index constraints = problem.info.number_of_eq_constraints;
    const Index parameters = problem.layout.parameters;
    const Index dimension = primal + constraints + parameters;
    Eigen::MatrixXd kkt =
        Eigen::MatrixXd::Zero(dimension, dimension);
    Eigen::VectorXd rhs(dimension);
    for (Index row = 0; row < primal; ++row)
        rhs(row) = problem.rhs_x(row);
    for (Index row = 0; row < constraints; ++row)
        rhs(primal + row) = problem.rhs_g(row);
    rhs.tail(parameters) = problem.rhs_parameters;

    for (Index stage = 0;
         stage < problem.layout.stages; ++stage)
    {
        const Index variables =
            problem.controls[stage] + problem.states[stage];
        const Index variable_offset =
            problem.info.offsets_primal_u[stage];
        for (Index row = 0; row < variables; ++row)
        {
            for (Index column = 0; column < variables; ++column)
            {
                kkt(variable_offset + row,
                    variable_offset + column) =
                    problem.hessian.RSQrqt[stage](row, column);
            }
        }

        const Index path_constraints =
            problem.equalities[stage];
        const Index path_offset =
            problem.info.offsets_g_eq_path[stage];
        for (Index equation = 0;
             equation < path_constraints; ++equation)
        {
            const Index kkt_constraint =
                primal + path_offset + equation;
            for (Index variable = 0;
                 variable < variables; ++variable)
            {
                const double derivative =
                    problem.jacobian.Gg_eqt[stage](
                        variable, equation);
                kkt(kkt_constraint,
                    variable_offset + variable) = derivative;
                kkt(variable_offset + variable,
                    kkt_constraint) = derivative;
            }
        }

        if (stage + 1 < problem.layout.stages)
        {
            const Index nx_next =
                problem.states[stage + 1];
            const Index dynamics_offset =
                problem.info.offsets_g_eq_dyn[stage];
            const Index next_state_offset =
                problem.info.offsets_primal_x[stage + 1];
            for (Index equation = 0;
                 equation < nx_next; ++equation)
            {
                const Index kkt_constraint =
                    primal + dynamics_offset + equation;
                for (Index variable = 0;
                     variable < variables; ++variable)
                {
                    const double derivative =
                        problem.jacobian.BAbt[stage](
                            variable, equation);
                    kkt(kkt_constraint,
                        variable_offset + variable) = derivative;
                    kkt(variable_offset + variable,
                        kkt_constraint) = derivative;
                }
                kkt(kkt_constraint,
                    next_state_offset + equation) = -1.0;
                kkt(next_state_offset + equation,
                    kkt_constraint) = -1.0;
            }
        }
    }

    if (parameters > 0)
    {
        kkt.block(0, primal + constraints, primal, parameters) =
            problem.border_primal;
        kkt.block(primal + constraints, 0, parameters, primal) =
            problem.border_primal.transpose();
        kkt.block(
            primal, primal + constraints,
            constraints, parameters) =
            problem.border_constraints;
        kkt.block(
            primal + constraints, primal,
            parameters, constraints) =
            problem.border_constraints.transpose();
        kkt.bottomRightCorner(parameters, parameters) =
            problem.border_hessian;
    }

    const Eigen::FullPivLU<Eigen::MatrixXd> lu(kkt);
    if (!lu.isInvertible())
        throw std::runtime_error("Dense KKT reference is singular");
    return lu.solve(-rhs);
}

double dense_reference_difference(
    const Problem &problem, const Result &result)
{
    const Index dimension =
        problem.info.number_of_primal_variables +
        problem.info.number_of_eq_constraints +
        problem.layout.parameters;
    if (!problem.config.dense_validation || dimension > 2500)
        return std::numeric_limits<double>::quiet_NaN();
    const Eigen::VectorXd reference = dense_reference(problem);
    Eigen::VectorXd candidate(dimension);
    candidate.head(problem.info.number_of_primal_variables) =
        result.primal;
    candidate.segment(
        problem.info.number_of_primal_variables,
        problem.info.number_of_eq_constraints) =
        result.multipliers;
    candidate.tail(problem.layout.parameters) =
        result.parameters;
    return max_abs(candidate - reference);
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const Config config = parse_arguments(argc, argv);
        Problem problem(config);
        AugSystemSolver<OcpType> sequential_solver(problem.info);
        AugSystemSolver<OcpType> blocked_solver(problem.info);

        (void)solve_bordered(
            problem, sequential_solver, false);
        (void)solve_bordered(
            problem, blocked_solver, true);

        std::vector<double> sequential_times;
        std::vector<double> sequential_rhs_times;
        std::vector<double> blocked_times;
        std::vector<double> blocked_rhs_times;
        Result sequential_result;
        Result blocked_result;
        for (Index repeat = 0; repeat < config.repeats; ++repeat)
        {
            sequential_result =
                solve_bordered(
                    problem, sequential_solver, false);
            blocked_result =
                solve_bordered(
                    problem, blocked_solver, true);
            sequential_times.push_back(
                sequential_result.elapsed_ms);
            sequential_rhs_times.push_back(
                sequential_result.parameter_rhs_ms);
            blocked_times.push_back(
                blocked_result.elapsed_ms);
            blocked_rhs_times.push_back(
                blocked_result.parameter_rhs_ms);
        }

        const double sequential_residual =
            structured_residual(problem, sequential_result);
        const double blocked_residual =
            structured_residual(problem, blocked_result);
        const double primal_difference =
            max_abs(
                sequential_result.primal -
                blocked_result.primal);
        const double multiplier_difference =
            max_abs(
                sequential_result.multipliers -
                blocked_result.multipliers);
        const double parameter_difference =
            max_abs(
                sequential_result.parameters -
                blocked_result.parameters);
        const double dense_difference =
            dense_reference_difference(
                problem, sequential_result);

        const double sequential_ms =
            median(sequential_times);
        const double sequential_rhs_ms =
            median(sequential_rhs_times);
        const double blocked_ms =
            median(blocked_times);
        const double blocked_rhs_ms =
            median(blocked_rhs_times);

        std::cout << std::setprecision(12);
        std::cout
            << "transcription,collocation_degree,phases,stages,"
               "phase_nodes,phase_nx,phase_nu,global_parameters,"
               "phase_parameters,total_parameters,base_primal,"
               "base_constraints,sequential_ms,sequential_rhs_ms,"
               "blocked_ms,blocked_rhs_ms,total_speedup,rhs_speedup,"
               "sequential_residual,blocked_residual,"
               "blocked_primal_difference,"
               "blocked_multiplier_difference,"
               "blocked_parameter_difference,dense_reference_difference\n";
        std::cout
            << (config.collocation_degree > 0
                    ? "radau"
                    : "shooting")
            << ',' << config.collocation_degree
            << ',' << config.phase_nodes.size()
            << ',' << problem.layout.stages
            << ',' << encode_list(config.phase_nodes)
            << ',' << encode_list(config.phase_nx)
            << ',' << encode_list(config.phase_nu)
            << ',' << config.global_parameters
            << ',' << encode_list(config.phase_parameters)
            << ',' << problem.layout.parameters
            << ',' << problem.info.number_of_primal_variables
            << ',' << problem.info.number_of_eq_constraints
            << ',' << sequential_ms
            << ',' << sequential_rhs_ms
            << ',' << blocked_ms
            << ',' << blocked_rhs_ms
            << ',' << sequential_ms / blocked_ms
            << ','
            << (blocked_rhs_ms > 0.0
                    ? sequential_rhs_ms / blocked_rhs_ms
                    : 1.0)
            << ',' << sequential_residual
            << ',' << blocked_residual
            << ',' << primal_difference
            << ',' << multiplier_difference
            << ',' << parameter_difference
            << ',' << dense_difference << '\n';

        constexpr double tolerance = 5e-7;
        if (sequential_residual > tolerance ||
            blocked_residual > tolerance ||
            primal_difference > tolerance ||
            multiplier_difference > tolerance ||
            parameter_difference > tolerance ||
            (std::isfinite(dense_difference) &&
             dense_difference > tolerance))
        {
            std::cerr
                << "Validation failed; tolerance="
                << tolerance << '\n';
            return 2;
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "heterogeneous_phase_kkt_benchmark: "
            << error.what() << '\n';
        return 1;
    }
}
