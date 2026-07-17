#include "fatrop/common/options.hpp"
#include "fatrop/ip_algorithm/ip_alg_builder.hpp"
#include "fatrop/ip_algorithm/ip_algorithm.hpp"
#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/nlp/nlp.hpp"
#include "fatrop/ocp/aug_system_solver.hpp"
#include "fatrop/ocp/dims.hpp"
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/problem_info.hpp"
#include "fatrop/ocp/type.hpp"

#include <Eigen/Dense>
#include <coin-or/IpIpoptApplication.hpp>
#include <coin-or/IpIpoptData.hpp>
#include <coin-or/IpTNLP.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using fatrop::AugSystemSolver;
using fatrop::Hessian;
using fatrop::ImplicitOcpType;
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
    std::vector<Index> phase_nodes{5, 4, 6};
    std::vector<Index> phase_nx{4, 7, 3};
    std::vector<Index> phase_nu{2, 3, 1};
    Index parameters = 3;
    Index collocation_degree = 3;
    Index max_iterations = 80;
    Index repeats = 3;
    double tolerance = 1e-8;
    double regularization = 1e-8;
    bool dense_validation = true;
    bool run_ipopt = false;
    bool run_native_ipm = false;
    std::string ipopt_linear_solver = "mumps";
};

std::vector<Index> parse_index_list(const std::string &text)
{
    std::vector<Index> result;
    std::size_t begin = 0;
    while (begin <= text.size())
    {
        const std::size_t end = text.find(',', begin);
        const std::string token =
            text.substr(begin, end == std::string::npos
                                   ? std::string::npos
                                   : end - begin);
        if (token.empty())
            throw std::runtime_error("Empty integer-list entry");
        result.push_back(static_cast<Index>(std::stoll(token)));
        if (end == std::string::npos)
            break;
        begin = end + 1;
    }
    return result;
}

std::string join(const std::vector<Index> &values)
{
    std::string result;
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (i != 0)
            result += ':';
        result += std::to_string(values[i]);
    }
    return result;
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
    return value.size() == 0 ? 0.0 : value.cwiseAbs().maxCoeff();
}

double sech_squared(double value)
{
    const double cosine = std::cosh(value);
    return 1.0 / (cosine * cosine);
}

double coefficient(Index row, Index column, Index phase, double scale)
{
    return scale *
           std::sin(0.17 * static_cast<double>(row + 1) +
                    0.29 * static_cast<double>(column + 1) +
                    0.11 * static_cast<double>(phase + 1));
}

void assign(VecRealAllocated &target, const Eigen::VectorXd &source)
{
    for (Index i = 0; i < source.size(); ++i)
        target(i) = source(i);
}

Eigen::VectorXd to_eigen(const VecRealAllocated &source)
{
    Eigen::VectorXd result(source.m());
    for (Index i = 0; i < result.size(); ++i)
        result(i) = source(i);
    return result;
}

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
        {
            throw std::runtime_error(
                "Radau collocation degree must be 1, 2, or 3");
        }

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

            double endpoint = 0.0;
            for (double polynomial_coefficient : polynomial)
                endpoint += polynomial_coefficient;
            D(basis) = endpoint;
        }
    }

    std::vector<double> tau;
    Eigen::MatrixXd C;
    Eigen::VectorXd D;
};

enum class StageKind
{
    Interval,
    Linkage,
    Terminal
};

struct Stage
{
    Index phase = 0;
    Index local_index = 0;
    StageKind kind = StageKind::Interval;
};

struct Layout
{
    explicit Layout(const Config &config)
    {
        const Index phases =
            static_cast<Index>(config.phase_nodes.size());
        for (Index phase = 0; phase < phases; ++phase)
        {
            const Index nodes =
                config.phase_nodes[static_cast<std::size_t>(phase)];
            const Index nx =
                config.phase_nx[static_cast<std::size_t>(phase)];
            const Index nu =
                config.phase_nu[static_cast<std::size_t>(phase)];
            for (Index local = 0; local < nodes - 1; ++local)
            {
                stages.push_back(
                    Stage{phase, local, StageKind::Interval});
                controls.push_back(
                    nu + config.collocation_degree * nx);
                states.push_back(nx);
                equalities.push_back(
                    config.collocation_degree * nx +
                    (phase == 0 && local == 0 ? nx : 0));
                inequalities.push_back(0);
            }
            stages.push_back(
                Stage{
                    phase,
                    nodes - 1,
                    phase + 1 < phases
                        ? StageKind::Linkage
                        : StageKind::Terminal});
            controls.push_back(0);
            states.push_back(nx);
            equalities.push_back(0);
            inequalities.push_back(0);
        }
    }

    std::vector<Stage> stages;
    std::vector<Index> controls;
    std::vector<Index> states;
    std::vector<Index> equalities;
    std::vector<Index> inequalities;
};

struct Values
{
    double objective = 0.0;
    Eigen::VectorXd constraints;
};

class NonlinearProblem
{
public:
    explicit NonlinearProblem(const Config &config)
        : config(config),
          coefficients(config.collocation_degree),
          layout(config),
          dims(
              static_cast<Index>(layout.stages.size()),
              layout.controls,
              layout.states,
              layout.equalities,
              layout.inequalities),
          info(dims),
          jacobian(dims),
          hessian(dims),
          D_x(info.number_of_primal_variables),
          D_s(info.number_of_slack_variables),
          rhs_x(info.number_of_primal_variables),
          rhs_g(info.number_of_eq_constraints),
          border_primal(
              info.number_of_primal_variables,
              config.parameters),
          border_constraints(
              info.number_of_eq_constraints,
              config.parameters),
          border_hessian(
              config.parameters,
              config.parameters),
          rhs_parameters(config.parameters),
          gradient_w(info.number_of_primal_variables),
          gradient_parameters(config.parameters),
          constraint_values(info.number_of_eq_constraints)
    {
        D_s = 0.0;
        set_regularization(config.regularization);
    }

    void set_regularization(double value)
    {
        regularization = value;
        for (Index i = 0;
             i < info.number_of_primal_variables; ++i)
            D_x(i) = value;
    }

    Eigen::VectorXd initial_primal() const
    {
        Eigen::VectorXd result =
            Eigen::VectorXd::Zero(
                info.number_of_primal_variables);
        for (Index stage = 0; stage < dims.K; ++stage)
        {
            const Index local_dim =
                layout.controls[static_cast<std::size_t>(stage)] +
                layout.states[static_cast<std::size_t>(stage)];
            const Index offset = info.offsets_primal_u[stage];
            for (Index local = 0; local < local_dim; ++local)
            {
                result(offset + local) =
                    0.03 *
                    std::sin(
                        0.13 * static_cast<double>(
                                   offset + local + 1));
            }
        }
        return result;
    }

    Eigen::VectorXd initial_parameters() const
    {
        return Eigen::VectorXd::Zero(config.parameters);
    }

    Values values(
        const Eigen::VectorXd &primal,
        const Eigen::VectorXd &parameters)
    {
        return evaluate(primal, parameters, false);
    }

    Values linearize(
        const Eigen::VectorXd &primal,
        const Eigen::VectorXd &parameters,
        const Eigen::VectorXd &multipliers)
    {
        Values result = evaluate(primal, parameters, true);
        rhs_x = 0.0;
        for (Index i = 0;
             i < info.number_of_primal_variables; ++i)
            rhs_x(i) = gradient_w(i);
        rhs_g = 0.0;
        for (Index i = 0;
             i < info.number_of_eq_constraints; ++i)
            rhs_g(i) = constraint_values(i);

        VecRealAllocated multiplier_fatrop(
            info.number_of_eq_constraints);
        VecRealAllocated stationarity(
            info.number_of_primal_variables);
        assign(multiplier_fatrop, multipliers);
        stationarity = 0.0;
        jacobian.transpose_apply_on_right(
            info,
            multiplier_fatrop,
            0.0,
            stationarity,
            stationarity);
        for (Index i = 0;
             i < info.number_of_primal_variables; ++i)
            rhs_x(i) += stationarity(i);

        rhs_parameters =
            gradient_parameters +
            border_constraints.transpose() * multipliers;
        return result;
    }

    Eigen::MatrixXd dense_constraint_jacobian() const
    {
        Eigen::MatrixXd result =
            Eigen::MatrixXd::Zero(
                info.number_of_eq_constraints,
                info.number_of_primal_variables +
                    config.parameters);
        for (Index stage = 0; stage < dims.K; ++stage)
        {
            const Index local_dim =
                layout.controls[static_cast<std::size_t>(stage)] +
                layout.states[static_cast<std::size_t>(stage)];
            const Index local_constraints =
                layout.equalities[
                    static_cast<std::size_t>(stage)];
            const Index variable_offset =
                info.offsets_primal_u[stage];
            const Index constraint_offset =
                info.offsets_g_eq_path[stage];
            for (Index equation = 0;
                 equation < local_constraints; ++equation)
            {
                for (Index variable = 0;
                     variable < local_dim; ++variable)
                {
                    result(
                        constraint_offset + equation,
                        variable_offset + variable) =
                        jacobian.Gg_eqt[stage](
                            variable, equation);
                }
            }
        }
        for (Index stage = 0; stage + 1 < dims.K; ++stage)
        {
            const Index local_dim =
                layout.controls[static_cast<std::size_t>(stage)] +
                layout.states[static_cast<std::size_t>(stage)];
            const Index next_nx =
                layout.states[
                    static_cast<std::size_t>(stage + 1)];
            const Index variable_offset =
                info.offsets_primal_u[stage];
            const Index next_state_offset =
                info.offsets_primal_x[stage + 1];
            const Index constraint_offset =
                info.offsets_g_eq_dyn[stage];
            for (Index equation = 0;
                 equation < next_nx; ++equation)
            {
                for (Index variable = 0;
                     variable < local_dim; ++variable)
                {
                    result(
                        constraint_offset + equation,
                        variable_offset + variable) =
                        jacobian.BAbt[stage](
                            variable, equation);
                }
                result(
                    constraint_offset + equation,
                    next_state_offset + equation) = -1.0;
            }
        }
        result.rightCols(config.parameters) =
            border_constraints;
        return result;
    }

    Eigen::MatrixXd dense_primal_hessian() const
    {
        Eigen::MatrixXd result =
            Eigen::MatrixXd::Zero(
                info.number_of_primal_variables,
                info.number_of_primal_variables);
        for (Index stage = 0; stage < dims.K; ++stage)
        {
            const Index local_dim =
                layout.controls[static_cast<std::size_t>(stage)] +
                layout.states[static_cast<std::size_t>(stage)];
            const Index offset = info.offsets_primal_u[stage];
            for (Index row = 0; row < local_dim; ++row)
            {
                for (Index column = 0;
                     column < local_dim; ++column)
                {
                    result(offset + row, offset + column) =
                        hessian.RSQrqt[stage](row, column);
                }
            }
        }
        result.diagonal().array() += regularization;
        return result;
    }

    Config config;
    CollocationCoefficients coefficients;
    Layout layout;
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
    Eigen::VectorXd gradient_w;
    Eigen::VectorXd gradient_parameters;
    Eigen::VectorXd constraint_values;
    double regularization = 0.0;

private:
    double local_value(
        const Eigen::VectorXd &primal,
        Index stage,
        Index basis,
        Index state) const
    {
        const Stage &metadata =
            layout.stages[static_cast<std::size_t>(stage)];
        const Index phase = metadata.phase;
        const Index nx =
            config.phase_nx[static_cast<std::size_t>(phase)];
        const Index nu =
            config.phase_nu[static_cast<std::size_t>(phase)];
        if (basis == 0)
            return primal(info.offsets_primal_x[stage] + state);
        return primal(
            info.offsets_primal_u[stage] + nu +
            (basis - 1) * nx + state);
    }

    Index local_variable(
        Index stage,
        Index basis,
        Index state) const
    {
        const Stage &metadata =
            layout.stages[static_cast<std::size_t>(stage)];
        const Index phase = metadata.phase;
        const Index nx =
            config.phase_nx[static_cast<std::size_t>(phase)];
        const Index nu =
            config.phase_nu[static_cast<std::size_t>(phase)];
        if (basis == 0)
        {
            return layout.controls[
                       static_cast<std::size_t>(stage)] +
                   state;
        }
        return nu + (basis - 1) * nx + state;
    }

    Values evaluate(
        const Eigen::VectorXd &primal,
        const Eigen::VectorXd &parameters,
        bool derivatives)
    {
        Values result;
        result.constraints =
            Eigen::VectorXd::Zero(
                info.number_of_eq_constraints);

        if (derivatives)
        {
            gradient_w.setZero();
            gradient_parameters.setZero();
            border_primal.setZero();
            border_constraints.setZero();
            border_hessian.setZero();
            hessian.set_zero();
            for (Index stage = 0; stage < dims.K; ++stage)
            {
                jacobian.Gg_eqt[stage] = 0.0;
                if (stage + 1 < dims.K)
                    jacobian.BAbt[stage] = 0.0;
            }
        }

        const double stage_scale =
            1.0 / static_cast<double>(dims.K);
        const double parameter_scale =
            config.parameters > 0
                ? 1.0 /
                      std::sqrt(
                          static_cast<double>(config.parameters))
                : 0.0;

        for (Index stage = 0; stage < dims.K; ++stage)
        {
            const Stage &metadata =
                layout.stages[static_cast<std::size_t>(stage)];
            const Index local_controls =
                layout.controls[static_cast<std::size_t>(stage)];
            const Index nx =
                layout.states[static_cast<std::size_t>(stage)];
            const Index local_dim = local_controls + nx;
            const Index offset = info.offsets_primal_u[stage];
            const Index physical_nu =
                config.phase_nu[
                    static_cast<std::size_t>(metadata.phase)];
            for (Index local = 0; local < local_dim; ++local)
            {
                double base_weight = 1.0;
                if (metadata.kind == StageKind::Interval &&
                    local < physical_nu)
                    base_weight = 0.18;
                else if (metadata.kind != StageKind::Interval)
                    base_weight = 1.8;
                const double square_root_weight =
                    std::sqrt(base_weight * stage_scale);
                const double target =
                    0.12 *
                    std::sin(
                        0.09 * static_cast<double>(
                                   offset + local + 1) +
                        0.21 * static_cast<double>(
                                   metadata.phase + 1));
                double parameter_prediction = 0.0;
                Eigen::VectorXd parameter_jacobian =
                    Eigen::VectorXd::Zero(config.parameters);
                for (Index parameter = 0;
                     parameter < config.parameters; ++parameter)
                {
                    const double sensitivity =
                        coefficient(
                            offset + local,
                            parameter,
                            metadata.phase,
                            0.035 * parameter_scale);
                    parameter_prediction +=
                        sensitivity *
                        std::tanh(parameters(parameter));
                    parameter_jacobian(parameter) =
                        -square_root_weight *
                        sensitivity *
                        sech_squared(parameters(parameter));
                }
                const double residual =
                    square_root_weight *
                    (primal(offset + local) -
                     target -
                     parameter_prediction);
                result.objective +=
                    0.5 * residual * residual;
                if (derivatives)
                {
                    gradient_w(offset + local) +=
                        square_root_weight * residual;
                    gradient_parameters +=
                        parameter_jacobian * residual;
                    hessian.RSQrqt[stage](local, local) +=
                        square_root_weight *
                        square_root_weight;
                    border_primal.row(offset + local) +=
                        square_root_weight *
                        parameter_jacobian.transpose();
                    border_hessian.noalias() +=
                        parameter_jacobian *
                        parameter_jacobian.transpose();
                }
            }
        }

        for (Index parameter = 0;
             parameter < config.parameters; ++parameter)
        {
            const double target =
                0.18 *
                std::sin(
                    0.37 * static_cast<double>(parameter + 1));
            const double residual =
                std::sqrt(0.8) *
                (parameters(parameter) - target);
            result.objective += 0.5 * residual * residual;
            if (derivatives)
            {
                gradient_parameters(parameter) +=
                    std::sqrt(0.8) * residual;
                border_hessian(parameter, parameter) += 0.8;
            }
        }

        for (Index stage = 0; stage < dims.K; ++stage)
        {
            const Stage &metadata =
                layout.stages[static_cast<std::size_t>(stage)];
            if (metadata.kind == StageKind::Interval)
            {
                evaluate_interval(
                    stage,
                    primal,
                    parameters,
                    derivatives,
                    result.constraints);
            }
            else if (metadata.kind == StageKind::Linkage)
            {
                evaluate_linkage(
                    stage,
                    primal,
                    parameters,
                    derivatives,
                    result.constraints);
            }
        }

        if (derivatives)
            constraint_values = result.constraints;
        return result;
    }

    void evaluate_interval(
        Index stage,
        const Eigen::VectorXd &primal,
        const Eigen::VectorXd &parameters,
        bool derivatives,
        Eigen::VectorXd &constraints)
    {
        const Stage &metadata =
            layout.stages[static_cast<std::size_t>(stage)];
        const Index phase = metadata.phase;
        const Index nx =
            config.phase_nx[static_cast<std::size_t>(phase)];
        const Index nu =
            config.phase_nu[static_cast<std::size_t>(phase)];
        const Index local_controls =
            layout.controls[static_cast<std::size_t>(stage)];
        const Index path_offset =
            info.offsets_g_eq_path[stage];
        const bool initial = phase == 0 &&
                             metadata.local_index == 0;
        const Index collocation_path_offset =
            initial ? nx : 0;
        const double step =
            (0.45 + 0.04 * static_cast<double>(phase % 5)) /
            static_cast<double>(
                config.phase_nodes[
                    static_cast<std::size_t>(phase)] -
                1);

        if (initial)
        {
            for (Index state = 0; state < nx; ++state)
            {
                double target =
                    0.16 *
                    std::sin(
                        0.31 * static_cast<double>(state + 1));
                for (Index parameter = 0;
                     parameter < config.parameters; ++parameter)
                {
                    target +=
                        coefficient(
                            state,
                            parameter,
                            phase,
                            0.025) *
                        std::sin(parameters(parameter));
                }
                const Index global_constraint =
                    path_offset + state;
                constraints(global_constraint) =
                    primal(
                        info.offsets_primal_x[stage] + state) -
                    target;
                if (derivatives)
                {
                    jacobian.Gg_eqt[stage](
                        local_controls + state,
                        state) = 1.0;
                    for (Index parameter = 0;
                         parameter < config.parameters; ++parameter)
                    {
                        border_constraints(
                            global_constraint,
                            parameter) =
                            -coefficient(
                                state,
                                parameter,
                                phase,
                                0.025) *
                            std::cos(parameters(parameter));
                    }
                }
            }
        }

        for (Index point = 1;
             point <= config.collocation_degree; ++point)
        {
            Eigen::VectorXd state(nx);
            Eigen::VectorXd polynomial_derivative(nx);
            for (Index equation = 0; equation < nx; ++equation)
            {
                state(equation) =
                    local_value(
                        primal, stage, point, equation);
                polynomial_derivative(equation) = 0.0;
                for (Index basis = 0;
                     basis <= config.collocation_degree; ++basis)
                {
                    polynomial_derivative(equation) +=
                        coefficients.C(basis, point) *
                        local_value(
                            primal,
                            stage,
                            basis,
                            equation);
                }
            }

            for (Index equation = 0;
                 equation < nx; ++equation)
            {
                const Index neighbour =
                    nx > 1 ? (equation + 1) % nx : equation;
                double mass =
                    1.0 +
                    0.02 * std::sin(state(equation));
                for (Index parameter = 0;
                     parameter < config.parameters; ++parameter)
                {
                    mass +=
                        coefficient(
                            equation,
                            parameter,
                            phase,
                            0.004) *
                        std::tanh(parameters(parameter));
                }

                const double decay =
                    0.52 +
                    0.025 * static_cast<double>(
                                (phase + equation) % 7);
                double dynamics =
                    -decay * state(equation) +
                    0.05 * std::sin(state(equation)) +
                    0.02 *
                        std::sin(
                            0.23 * static_cast<double>(
                                       stage + equation + 1));
                if (nx > 1)
                {
                    dynamics +=
                        0.025 * std::tanh(state(neighbour));
                }
                for (Index control = 0; control < nu; ++control)
                {
                    dynamics +=
                        coefficient(
                            equation,
                            control,
                            phase,
                            0.09) *
                        std::tanh(
                            primal(
                                info.offsets_primal_u[stage] +
                                control));
                }
                for (Index parameter = 0;
                     parameter < config.parameters; ++parameter)
                {
                    dynamics +=
                        coefficient(
                            equation,
                            parameter,
                            phase,
                            0.065) *
                        std::sin(parameters(parameter));
                }

                const Index local_constraint =
                    collocation_path_offset +
                    (point - 1) * nx +
                    equation;
                const Index global_constraint =
                    path_offset + local_constraint;
                constraints(global_constraint) =
                    mass *
                        polynomial_derivative(equation) -
                    step * dynamics;

                if (!derivatives)
                    continue;

                for (Index control = 0;
                     control < nu; ++control)
                {
                    jacobian.Gg_eqt[stage](
                        control,
                        local_constraint) =
                        -step *
                        coefficient(
                            equation,
                            control,
                            phase,
                            0.09) *
                        sech_squared(
                            primal(
                                info.offsets_primal_u[stage] +
                                control));
                }

                for (Index basis = 0;
                     basis <= config.collocation_degree; ++basis)
                {
                    for (Index state_index = 0;
                         state_index < nx; ++state_index)
                    {
                        double derivative = 0.0;
                        if (state_index == equation)
                        {
                            derivative +=
                                mass *
                                coefficients.C(basis, point);
                        }
                        if (basis == point)
                        {
                            if (state_index == equation)
                            {
                                derivative +=
                                    0.02 *
                                    std::cos(state(equation)) *
                                    polynomial_derivative(equation);
                                derivative -=
                                    step *
                                    (-decay +
                                     0.05 *
                                         std::cos(
                                             state(equation)));
                            }
                            if (nx > 1 &&
                                state_index == neighbour)
                            {
                                derivative -=
                                    step * 0.025 *
                                    sech_squared(
                                        state(neighbour));
                            }
                        }
                        jacobian.Gg_eqt[stage](
                            local_variable(
                                stage,
                                basis,
                                state_index),
                            local_constraint) = derivative;
                    }
                }

                for (Index parameter = 0;
                     parameter < config.parameters; ++parameter)
                {
                    const double mass_derivative =
                        coefficient(
                            equation,
                            parameter,
                            phase,
                            0.004) *
                        sech_squared(parameters(parameter));
                    const double dynamics_derivative =
                        coefficient(
                            equation,
                            parameter,
                            phase,
                            0.065) *
                        std::cos(parameters(parameter));
                    border_constraints(
                        global_constraint,
                        parameter) =
                        mass_derivative *
                            polynomial_derivative(equation) -
                        step * dynamics_derivative;
                }
            }
        }

        const Index next_state_offset =
            info.offsets_primal_x[stage + 1];
        const Index dynamics_offset =
            info.offsets_g_eq_dyn[stage];
        for (Index state = 0; state < nx; ++state)
        {
            double endpoint = 0.0;
            for (Index basis = 0;
                 basis <= config.collocation_degree; ++basis)
            {
                endpoint +=
                    coefficients.D(basis) *
                    local_value(
                        primal,
                        stage,
                        basis,
                        state);
                if (derivatives)
                {
                    jacobian.BAbt[stage](
                        local_variable(
                            stage,
                            basis,
                            state),
                        state) =
                        coefficients.D(basis);
                }
            }
            constraints(dynamics_offset + state) =
                endpoint -
                primal(next_state_offset + state);
        }
    }

    void evaluate_linkage(
        Index stage,
        const Eigen::VectorXd &primal,
        const Eigen::VectorXd &parameters,
        bool derivatives,
        Eigen::VectorXd &constraints)
    {
        const Stage &metadata =
            layout.stages[static_cast<std::size_t>(stage)];
        const Index phase = metadata.phase;
        const Index nx =
            config.phase_nx[static_cast<std::size_t>(phase)];
        const Index next_nx =
            config.phase_nx[
                static_cast<std::size_t>(phase + 1)];
        const Index state_offset =
            info.offsets_primal_x[stage];
        const Index next_state_offset =
            info.offsets_primal_x[stage + 1];
        const Index dynamics_offset =
            info.offsets_g_eq_dyn[stage];

        for (Index equation = 0;
             equation < next_nx; ++equation)
        {
            const Index main = equation % nx;
            const Index neighbour =
                nx > 1 ? (main + 1) % nx : main;
            double reset =
                0.72 *
                    std::tanh(
                        primal(state_offset + main)) +
                0.035 *
                    std::sin(
                        0.19 * static_cast<double>(
                                   phase + equation + 1));
            if (nx > 1)
                reset +=
                    0.08 * primal(state_offset + neighbour);
            for (Index parameter = 0;
                 parameter < config.parameters; ++parameter)
            {
                reset +=
                    coefficient(
                        equation,
                        parameter,
                        phase,
                        0.055) *
                    std::sin(parameters(parameter));
            }
            const Index global_constraint =
                dynamics_offset + equation;
            constraints(global_constraint) =
                reset -
                primal(next_state_offset + equation);

            if (!derivatives)
                continue;
            jacobian.BAbt[stage](main, equation) +=
                0.72 *
                sech_squared(primal(state_offset + main));
            if (nx > 1)
                jacobian.BAbt[stage](neighbour, equation) +=
                    0.08;
            for (Index parameter = 0;
                 parameter < config.parameters; ++parameter)
            {
                border_constraints(
                    global_constraint,
                    parameter) =
                    coefficient(
                        equation,
                        parameter,
                        phase,
                        0.055) *
                    std::cos(parameters(parameter));
            }
        }
    }
};

class NativeNonlinearNlp final
    : public fatrop::Nlp<ImplicitOcpType>
{
public:
    explicit NativeNonlinearNlp(NonlinearProblem &problem)
        : problem_(problem),
          dims_(
              problem.dims.K, problem.layout.controls,
              problem.layout.states, problem.layout.equalities,
              problem.layout.inequalities, problem.config.parameters),
          nlp_dims_(
              problem.info.number_of_primal_variables
                  + problem.config.parameters,
              problem.info.number_of_eq_constraints, 0),
          trajectory_(problem.info.number_of_primal_variables),
          parameters_(problem.config.parameters),
          multipliers_(problem.info.number_of_eq_constraints)
    {
    }

    const fatrop::NlpDims &nlp_dims() const override
    {
        return nlp_dims_;
    }

    const ProblemDims &problem_dims() const override
    {
        return dims_;
    }

    Index eval_lag_hess(
        const ProblemInfo &info, Scalar objective_scale,
        const fatrop::VecRealView &primal_x,
        const fatrop::VecRealView &primal_s,
        const fatrop::VecRealView &lam,
        Hessian<ImplicitOcpType> &hess) override
    {
        (void)primal_s;
        load_primal(info, primal_x);
        for (Index row = 0; row < multipliers_.size(); ++row)
            multipliers_(row) = lam(row);
        problem_.linearize(trajectory_, parameters_, multipliers_);

        hess.set_zero();
        for (auto &block : hess.FuFx)
            block = 0.0;
        for (auto &block : hess.GuGx)
            block = 0.0;
        for (Index stage = 0; stage < problem_.dims.K; ++stage)
        {
            const Index local = problem_.layout.controls[stage]
                              + problem_.layout.states[stage];
            for (Index row = 0; row < local; ++row)
            for (Index column = 0; column < local; ++column)
            {
                hess.RSQrqt[stage](row, column) = objective_scale
                    * problem_.hessian.RSQrqt[stage](row, column);
            }
        }
        for (Index row = 0;
             row < problem_.info.number_of_primal_variables; ++row)
        for (Index parameter = 0;
             parameter < problem_.config.parameters; ++parameter)
        {
            hess.global_parameter_cross_hessian(row, parameter) =
                objective_scale * problem_.border_primal(row, parameter);
        }
        for (Index row = 0; row < problem_.config.parameters; ++row)
        for (Index column = 0; column < problem_.config.parameters; ++column)
        {
            hess.global_parameter_hessian(row, column) =
                objective_scale * problem_.border_hessian(row, column);
        }
        return 0;
    }

    Index eval_constr_jac(
        const ProblemInfo &info,
        const fatrop::VecRealView &primal_x,
        const fatrop::VecRealView &primal_s,
        Jacobian<ImplicitOcpType> &jac) override
    {
        (void)primal_s;
        load_primal(info, primal_x);
        multipliers_.setZero();
        problem_.linearize(trajectory_, parameters_, multipliers_);

        jac.global_parameter_jacobian = 0.0;
        for (Index stage = 0; stage < problem_.dims.K; ++stage)
        {
            jac.Gg_eqt[stage] = 0.0;
            jac.Gg_ineqt[stage] = 0.0;
            const Index local = problem_.layout.controls[stage]
                              + problem_.layout.states[stage];
            const Index local_equalities = problem_.layout.equalities[stage];
            for (Index row = 0; row < local; ++row)
            for (Index equation = 0;
                 equation < local_equalities; ++equation)
            {
                jac.Gg_eqt[stage](row, equation) =
                    problem_.jacobian.Gg_eqt[stage](row, equation);
            }
            const Index equality_offset =
                problem_.info.offsets_g_eq_path[stage];
            for (Index equation = 0;
                 equation < local_equalities; ++equation)
            for (Index parameter = 0;
                 parameter < problem_.config.parameters; ++parameter)
            {
                jac.global_parameter_jacobian(
                    info.offsets_g_eq_path[stage] + equation, parameter) =
                    problem_.border_constraints(
                        equality_offset + equation, parameter);
            }

            if (stage + 1 == problem_.dims.K)
                continue;
            jac.BAbt[stage] = 0.0;
            jac.Jt[stage] = 0.0;
            jac.Jt_inv[stage] = 0.0;
            const Index next_nx = problem_.layout.states[stage + 1];
            for (Index row = 0; row < local; ++row)
            for (Index equation = 0; equation < next_nx; ++equation)
            {
                jac.BAbt[stage](row, equation) =
                    problem_.jacobian.BAbt[stage](row, equation);
            }
            for (Index state = 0; state < next_nx; ++state)
            {
                jac.Jt[stage](state, state) = -1.0;
                jac.Jt_inv[stage](state, state) = -1.0;
            }
            const Index dynamics_offset =
                problem_.info.offsets_g_eq_dyn[stage];
            for (Index equation = 0; equation < next_nx; ++equation)
            for (Index parameter = 0;
                 parameter < problem_.config.parameters; ++parameter)
            {
                jac.global_parameter_jacobian(
                    info.offsets_g_eq_dyn[stage] + equation, parameter) =
                    problem_.border_constraints(
                        dynamics_offset + equation, parameter);
            }
        }
        return 0;
    }

    Index eval_constraint_violation(
        const ProblemInfo &info,
        const fatrop::VecRealView &primal_x,
        const fatrop::VecRealView &primal_s,
        fatrop::VecRealView &res) override
    {
        (void)primal_s;
        load_primal(info, primal_x);
        const Values values = problem_.values(trajectory_, parameters_);
        for (Index row = 0; row < values.constraints.size(); ++row)
            res(row) = values.constraints(row);
        return 0;
    }

    Index eval_objective_gradient(
        const ProblemInfo &info, Scalar objective_scale,
        const fatrop::VecRealView &primal_x,
        const fatrop::VecRealView &primal_s,
        fatrop::VecRealView &grad_x,
        fatrop::VecRealView &grad_s) override
    {
        (void)primal_s;
        load_primal(info, primal_x);
        multipliers_.setZero();
        problem_.linearize(trajectory_, parameters_, multipliers_);
        grad_x = 0.0;
        grad_s = 0.0;
        for (Index row = 0;
             row < problem_.info.number_of_primal_variables; ++row)
            grad_x(row) = objective_scale * problem_.gradient_w(row);
        for (Index parameter = 0;
             parameter < problem_.config.parameters; ++parameter)
        {
            grad_x(info.offset_primal_global + parameter) =
                objective_scale * problem_.gradient_parameters(parameter);
        }
        return 0;
    }

    Index eval_objective(
        const ProblemInfo &info, Scalar objective_scale,
        const fatrop::VecRealView &primal_x,
        const fatrop::VecRealView &primal_s,
        Scalar &res) override
    {
        (void)primal_s;
        load_primal(info, primal_x);
        res = objective_scale
            * problem_.values(trajectory_, parameters_).objective;
        return 0;
    }

    Index get_bounds(
        const ProblemInfo &, fatrop::VecRealView &lower_bounds,
        fatrop::VecRealView &upper_bounds) override
    {
        lower_bounds = 0.0;
        upper_bounds = 0.0;
        return 0;
    }

    Index get_initial_primal(
        const ProblemInfo &info,
        fatrop::VecRealView &primal_x) override
    {
        trajectory_ = problem_.initial_primal();
        parameters_ = problem_.initial_parameters();
        for (Index row = 0; row < trajectory_.size(); ++row)
            primal_x(row) = trajectory_(row);
        for (Index parameter = 0; parameter < parameters_.size(); ++parameter)
            primal_x(info.offset_primal_global + parameter) =
                parameters_(parameter);
        return 0;
    }

    void get_primal_damping(
        const ProblemInfo &, fatrop::VecRealView &damping) override
    {
        damping = 0.0;
    }

    void apply_jacobian_s_transpose(
        const ProblemInfo &, const fatrop::VecRealView &,
        Scalar alpha, const fatrop::VecRealView &y,
        fatrop::VecRealView &out) override
    {
        out = alpha * y;
    }

private:
    void load_primal(
        const ProblemInfo &info,
        const fatrop::VecRealView &primal_x)
    {
        for (Index row = 0; row < trajectory_.size(); ++row)
            trajectory_(row) = primal_x(row);
        for (Index parameter = 0; parameter < parameters_.size(); ++parameter)
        {
            parameters_(parameter) =
                primal_x(info.offset_primal_global + parameter);
        }
    }

    NonlinearProblem &problem_;
    ProblemDims dims_;
    fatrop::NlpDims nlp_dims_;
    Eigen::VectorXd trajectory_;
    Eigen::VectorXd parameters_;
    Eigen::VectorXd multipliers_;
};

struct Direction
{
    Eigen::VectorXd primal;
    Eigen::VectorXd multipliers;
    Eigen::VectorXd parameters;
    double elapsed_ms = 0.0;
    double factor_ms = 0.0;
    double parameter_rhs_ms = 0.0;
    double schur_ms = 0.0;
    double residual = 0.0;
};

double direction_residual(
    NonlinearProblem &problem,
    const Direction &direction)
{
    VecRealAllocated primal(
        problem.info.number_of_primal_variables);
    VecRealAllocated multipliers(
        problem.info.number_of_eq_constraints);
    VecRealAllocated hessian_product(
        problem.info.number_of_primal_variables);
    VecRealAllocated jacobian_transpose_product(
        problem.info.number_of_primal_variables);
    VecRealAllocated jacobian_product(
        problem.info.number_of_eq_constraints);
    assign(primal, direction.primal);
    assign(multipliers, direction.multipliers);
    hessian_product = 0.0;
    jacobian_transpose_product = 0.0;
    jacobian_product = 0.0;
    problem.hessian.apply_on_right(
        problem.info,
        primal,
        0.0,
        hessian_product,
        hessian_product);
    problem.jacobian.transpose_apply_on_right(
        problem.info,
        multipliers,
        0.0,
        jacobian_transpose_product,
        jacobian_transpose_product);
    problem.jacobian.apply_on_right(
        problem.info,
        primal,
        0.0,
        jacobian_product,
        jacobian_product);

    Eigen::VectorXd stationarity =
        to_eigen(problem.rhs_x) +
        to_eigen(hessian_product) +
        to_eigen(jacobian_transpose_product) +
        problem.border_primal * direction.parameters;
    for (Index i = 0;
         i < problem.info.number_of_primal_variables; ++i)
    {
        stationarity(i) +=
            problem.D_x(i) * direction.primal(i);
    }
    const Eigen::VectorXd constraints =
        to_eigen(problem.rhs_g) +
        to_eigen(jacobian_product) +
        problem.border_constraints * direction.parameters;
    const Eigen::VectorXd parameter_stationarity =
        problem.rhs_parameters +
        problem.border_primal.transpose() *
            direction.primal +
        problem.border_constraints.transpose() *
            direction.multipliers +
        problem.border_hessian * direction.parameters;
    return std::max(
        {max_abs(stationarity),
         max_abs(constraints),
         max_abs(parameter_stationarity)});
}

Direction solve_direction(
    NonlinearProblem &problem,
    AugSystemSolver<OcpType> &solver)
{
    Direction result;
    result.primal.resize(
        problem.info.number_of_primal_variables);
    result.multipliers.resize(
        problem.info.number_of_eq_constraints);
    result.parameters.resize(problem.config.parameters);

    VecRealAllocated primal(
        problem.info.number_of_primal_variables);
    VecRealAllocated multipliers(
        problem.info.number_of_eq_constraints);
    primal = 0.0;
    multipliers = 0.0;

    const auto start = std::chrono::steady_clock::now();
    const auto factor_status =
        solver.solve(
            problem.info,
            problem.jacobian,
            problem.hessian,
            problem.D_x,
            problem.D_s,
            problem.rhs_x,
            problem.rhs_g,
            primal,
            multipliers);
    const auto after_factor =
        std::chrono::steady_clock::now();
    if (factor_status != LinsolReturnFlag::SUCCESS)
        throw std::runtime_error(
            "Nonlinear SQP FATROP factorization failed");

    const Eigen::VectorXd base_primal = to_eigen(primal);
    const Eigen::VectorXd base_multipliers =
        to_eigen(multipliers);

    if (problem.config.parameters == 0)
    {
        result.primal = base_primal;
        result.multipliers = base_multipliers;
        result.parameters.resize(0);
        const auto stop = std::chrono::steady_clock::now();
        result.elapsed_ms =
            std::chrono::duration<double, std::milli>(
                stop - start)
                .count();
        result.factor_ms = result.elapsed_ms;
        result.residual =
            direction_residual(problem, result);
        return result;
    }

    MatRealAllocated batch_primal_rhs(
        problem.info.number_of_primal_variables,
        problem.config.parameters);
    MatRealAllocated batch_constraint_rhs(
        problem.info.number_of_eq_constraints,
        problem.config.parameters);
    MatRealAllocated batch_primal(
        problem.info.number_of_primal_variables,
        problem.config.parameters);
    MatRealAllocated batch_multipliers(
        problem.info.number_of_eq_constraints,
        problem.config.parameters);
    for (Index column = 0;
         column < problem.config.parameters; ++column)
    {
        for (Index row = 0;
             row < problem.info.number_of_primal_variables; ++row)
        {
            batch_primal_rhs(row, column) =
                problem.border_primal(row, column);
        }
        for (Index row = 0;
             row < problem.info.number_of_eq_constraints; ++row)
        {
            batch_constraint_rhs(row, column) =
                problem.border_constraints(row, column);
        }
    }
    batch_primal = 0.0;
    batch_multipliers = 0.0;
    const auto batch_status =
        solver.solve_rhs_batch(
            problem.info,
            problem.jacobian,
            problem.hessian,
            problem.D_s,
            batch_primal_rhs,
            batch_constraint_rhs,
            batch_primal,
            batch_multipliers);
    if (batch_status != LinsolReturnFlag::SUCCESS)
        throw std::runtime_error(
            "Nonlinear SQP blocked parameter solve failed");
    const auto after_parameter_rhs =
        std::chrono::steady_clock::now();

    Eigen::MatrixXd response_primal(
        problem.info.number_of_primal_variables,
        problem.config.parameters);
    Eigen::MatrixXd response_multipliers(
        problem.info.number_of_eq_constraints,
        problem.config.parameters);
    for (Index column = 0;
         column < problem.config.parameters; ++column)
    {
        for (Index row = 0;
             row < problem.info.number_of_primal_variables; ++row)
            response_primal(row, column) =
                batch_primal(row, column);
        for (Index row = 0;
             row < problem.info.number_of_eq_constraints; ++row)
            response_multipliers(row, column) =
                batch_multipliers(row, column);
    }

    const Eigen::MatrixXd schur =
        problem.border_hessian +
        problem.border_primal.transpose() *
            response_primal +
        problem.border_constraints.transpose() *
            response_multipliers;
    const Eigen::VectorXd schur_rhs =
        -problem.rhs_parameters -
        problem.border_primal.transpose() * base_primal -
        problem.border_constraints.transpose() *
            base_multipliers;
    const Eigen::LDLT<Eigen::MatrixXd> ldlt(schur);
    if (ldlt.info() != Eigen::Success)
        throw std::runtime_error(
            "Nonlinear SQP parameter Schur factorization failed");
    result.parameters = ldlt.solve(schur_rhs);
    if (ldlt.info() != Eigen::Success ||
        !result.parameters.allFinite())
    {
        throw std::runtime_error(
            "Nonlinear SQP parameter Schur solve failed");
    }
    result.primal =
        base_primal +
        response_primal * result.parameters;
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
    result.schur_ms =
        std::chrono::duration<double, std::milli>(
            stop - after_parameter_rhs)
            .count();
    result.residual =
        direction_residual(problem, result);
    return result;
}

double dense_direction_difference(
    NonlinearProblem &problem,
    const Direction &direction)
{
    const Index n =
        problem.info.number_of_primal_variables;
    const Index m =
        problem.info.number_of_eq_constraints;
    const Index p = problem.config.parameters;
    const Index total = n + m + p;
    if (!problem.config.dense_validation || total > 2500)
        return std::numeric_limits<double>::quiet_NaN();

    const Eigen::MatrixXd A =
        problem.dense_constraint_jacobian()
            .leftCols(n);
    Eigen::MatrixXd matrix =
        Eigen::MatrixXd::Zero(total, total);
    matrix.block(0, 0, n, n) =
        problem.dense_primal_hessian();
    matrix.block(0, n, n, m) = A.transpose();
    matrix.block(n, 0, m, n) = A;
    if (p > 0)
    {
        matrix.block(0, n + m, n, p) =
            problem.border_primal;
        matrix.block(n + m, 0, p, n) =
            problem.border_primal.transpose();
        matrix.block(n, n + m, m, p) =
            problem.border_constraints;
        matrix.block(n + m, n, p, m) =
            problem.border_constraints.transpose();
        matrix.block(n + m, n + m, p, p) =
            problem.border_hessian;
    }

    Eigen::VectorXd rhs(total);
    rhs.segment(0, n) = to_eigen(problem.rhs_x);
    rhs.segment(n, m) = to_eigen(problem.rhs_g);
    if (p > 0)
        rhs.tail(p) = problem.rhs_parameters;
    const Eigen::VectorXd reference =
        matrix.fullPivLu().solve(-rhs);
    Eigen::VectorXd structured(total);
    structured.segment(0, n) = direction.primal;
    structured.segment(n, m) = direction.multipliers;
    if (p > 0)
        structured.tail(p) = direction.parameters;
    return max_abs(reference - structured);
}

struct SqpResult
{
    Eigen::VectorXd primal;
    Eigen::VectorXd parameters;
    Eigen::VectorXd multipliers;
    Index iterations = 0;
    Index line_search_evaluations = 0;
    bool converged = false;
    double elapsed_ms = 0.0;
    double kkt_ms = 0.0;
    double parameter_rhs_ms = 0.0;
    double objective = 0.0;
    double primal_inf = 0.0;
    double dual_inf = 0.0;
    double linear_residual = 0.0;
    double dense_step_difference =
        std::numeric_limits<double>::quiet_NaN();
};

struct DerivativeCheck
{
    double objective_error =
        std::numeric_limits<double>::quiet_NaN();
    double constraint_error =
        std::numeric_limits<double>::quiet_NaN();
};

DerivativeCheck check_derivatives(NonlinearProblem &problem)
{
    DerivativeCheck result;
    if (!problem.config.dense_validation)
        return result;
    const Eigen::VectorXd primal =
        problem.initial_primal();
    const Eigen::VectorXd parameters =
        problem.initial_parameters();
    problem.linearize(
        primal,
        parameters,
        Eigen::VectorXd::Zero(
            problem.info.number_of_eq_constraints));

    Eigen::VectorXd direction(
        primal.size() + parameters.size());
    for (Index i = 0; i < direction.size(); ++i)
    {
        direction(i) =
            std::sin(
                0.23 * static_cast<double>(i + 1));
    }
    direction.normalize();
    const Eigen::VectorXd primal_direction =
        direction.head(primal.size());
    const Eigen::VectorXd parameter_direction =
        direction.tail(parameters.size());
    constexpr double step = 1e-6;
    const Values plus =
        problem.values(
            primal + step * primal_direction,
            parameters + step * parameter_direction);
    const Values minus =
        problem.values(
            primal - step * primal_direction,
            parameters - step * parameter_direction);
    const double objective_finite_difference =
        (plus.objective - minus.objective) /
        (2.0 * step);
    const double objective_analytic =
        problem.gradient_w.dot(primal_direction) +
        problem.gradient_parameters.dot(
            parameter_direction);
    result.objective_error =
        std::abs(
            objective_finite_difference -
            objective_analytic);

    const Eigen::VectorXd constraint_finite_difference =
        (plus.constraints - minus.constraints) /
        (2.0 * step);
    const Eigen::VectorXd constraint_analytic =
        problem.dense_constraint_jacobian() *
        direction;
    result.constraint_error =
        max_abs(
            constraint_finite_difference -
            constraint_analytic);
    return result;
}

SqpResult solve_sqp(
    NonlinearProblem &problem,
    AugSystemSolver<OcpType> &solver)
{
    SqpResult result;
    result.primal = problem.initial_primal();
    result.parameters = problem.initial_parameters();
    result.multipliers =
        Eigen::VectorXd::Zero(
            problem.info.number_of_eq_constraints);
    double penalty = 10.0;
    const auto start = std::chrono::steady_clock::now();

    for (Index iteration = 0;
         iteration < problem.config.max_iterations;
         ++iteration)
    {
        const Values current =
            problem.linearize(
                result.primal,
                result.parameters,
                result.multipliers);
        const double primal_inf =
            max_abs(current.constraints);
        const double dual_inf =
            std::max(
                max_abs(to_eigen(problem.rhs_x)),
                max_abs(problem.rhs_parameters));
        result.iterations = iteration;
        if (std::max(primal_inf, dual_inf) <=
            problem.config.tolerance)
        {
            result.converged = true;
            break;
        }

        const Direction direction =
            solve_direction(problem, solver);
        result.kkt_ms += direction.elapsed_ms;
        result.parameter_rhs_ms +=
            direction.parameter_rhs_ms;
        result.linear_residual =
            std::max(
                result.linear_residual,
                direction.residual);
        if (iteration == 0)
        {
            result.dense_step_difference =
                dense_direction_difference(
                    problem, direction);
        }

        penalty = std::max(
            penalty,
            1.0 +
                max_abs(
                    result.multipliers +
                    direction.multipliers));
        const double current_merit =
            current.objective +
            penalty *
                current.constraints.lpNorm<1>();
        double directional_derivative =
            problem.gradient_w.dot(direction.primal) +
            problem.gradient_parameters.dot(
                direction.parameters) -
            penalty *
                current.constraints.lpNorm<1>();
        if (!(directional_derivative < 0.0))
        {
            directional_derivative =
                -0.5 *
                    (direction.primal.squaredNorm() +
                     direction.parameters.squaredNorm()) -
                penalty *
                    current.constraints.lpNorm<1>();
        }

        double alpha = 1.0;
        bool accepted = false;
        for (Index line_search = 0;
             line_search < 24; ++line_search)
        {
            const Eigen::VectorXd trial_primal =
                result.primal +
                alpha * direction.primal;
            const Eigen::VectorXd trial_parameters =
                result.parameters +
                alpha * direction.parameters;
            const Values trial =
                problem.values(
                    trial_primal,
                    trial_parameters);
            ++result.line_search_evaluations;
            const double trial_merit =
                trial.objective +
                penalty *
                    trial.constraints.lpNorm<1>();
            if (std::isfinite(trial_merit) &&
                trial_merit <=
                    current_merit +
                        1e-4 * alpha *
                            directional_derivative)
            {
                result.primal = trial_primal;
                result.parameters = trial_parameters;
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

    const Values final_values =
        problem.linearize(
            result.primal,
            result.parameters,
            result.multipliers);
    result.objective = final_values.objective;
    result.primal_inf =
        max_abs(final_values.constraints);
    result.dual_inf =
        std::max(
            max_abs(to_eigen(problem.rhs_x)),
            max_abs(problem.rhs_parameters));
    if (std::max(result.primal_inf, result.dual_inf) <=
        problem.config.tolerance)
        result.converged = true;
    const auto stop = std::chrono::steady_clock::now();
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(
            stop - start)
            .count();
    return result;
}

struct NativeIpmResult
{
    Eigen::VectorXd primal;
    Eigen::VectorXd parameters;
    Eigen::VectorXd multipliers;
    double objective = std::numeric_limits<double>::quiet_NaN();
    double elapsed_ms = std::numeric_limits<double>::quiet_NaN();
    double constraint_inf = std::numeric_limits<double>::quiet_NaN();
    double dual_inf = std::numeric_limits<double>::quiet_NaN();
    int status = -1;
    int iterations = -1;
    bool converged = false;
};

NativeIpmResult solve_native_ipm(NonlinearProblem &problem)
{
    auto nlp = std::make_shared<NativeNonlinearNlp>(problem);
    fatrop::OptionRegistry options;
    fatrop::IpAlgBuilder<ImplicitOcpType> builder(nlp);
    auto algorithm = builder.with_options_registry(&options).build();
    options.set_option("print_level", 0);
    options.set_option("max_iter", problem.config.max_iterations * 3);
    options.set_option("tolerance", problem.config.tolerance);

    NativeIpmResult result;
    std::vector<double> elapsed;
    elapsed.reserve(static_cast<std::size_t>(problem.config.repeats));
    for (Index repeat = 0; repeat < problem.config.repeats; ++repeat)
    {
        const auto start = std::chrono::steady_clock::now();
        const fatrop::IpSolverReturnFlag status = algorithm->optimize();
        const auto stop = std::chrono::steady_clock::now();
        elapsed.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count());
        result.status = static_cast<int>(status);
        result.iterations = static_cast<int>(algorithm->iteration_count());
        if (status != fatrop::IpSolverReturnFlag::Success
            && status != fatrop::IpSolverReturnFlag::StopAtAcceptablePoint)
            break;
    }
    result.elapsed_ms = median(elapsed);

    const ProblemInfo &info = algorithm->info();
    const auto &solution = algorithm->solution_primal();
    const auto &multipliers = algorithm->solution_dual();
    result.primal.resize(problem.info.number_of_primal_variables);
    result.parameters.resize(problem.config.parameters);
    result.multipliers.resize(problem.info.number_of_eq_constraints);
    for (Index row = 0; row < result.primal.size(); ++row)
        result.primal(row) = solution(row);
    for (Index parameter = 0; parameter < result.parameters.size(); ++parameter)
    {
        result.parameters(parameter) =
            solution(info.offset_primal_global + parameter);
    }
    for (Index row = 0; row < result.multipliers.size(); ++row)
        result.multipliers(row) = multipliers(row);

    const Values final_values = problem.linearize(
        result.primal, result.parameters, result.multipliers);
    result.objective = final_values.objective;
    result.constraint_inf = max_abs(final_values.constraints);
    result.dual_inf = std::max(
        max_abs(to_eigen(problem.rhs_x)),
        max_abs(problem.rhs_parameters));
    result.converged =
        (result.status
             == static_cast<int>(fatrop::IpSolverReturnFlag::Success)
         || result.status == static_cast<int>(
             fatrop::IpSolverReturnFlag::StopAtAcceptablePoint))
        && result.constraint_inf <= 10.0 * problem.config.tolerance
        && result.dual_inf <= 10.0 * problem.config.tolerance;
    return result;
}

struct IpoptResult
{
    Eigen::VectorXd primal;
    Eigen::VectorXd parameters;
    Eigen::VectorXd multipliers;
    double objective = std::numeric_limits<double>::quiet_NaN();
    double elapsed_ms = std::numeric_limits<double>::quiet_NaN();
    double constraint_inf =
        std::numeric_limits<double>::quiet_NaN();
    int status = -1;
    int iterations = -1;
    Index jacobian_nonzeros = 0;
};

class NonlinearTnlp final : public Ipopt::TNLP
{
public:
    NonlinearTnlp(
        NonlinearProblem &problem,
        IpoptResult &result)
        : problem_(problem), result_(result)
    {
        const Eigen::VectorXd primal =
            problem_.initial_primal();
        const Eigen::VectorXd parameters =
            problem_.initial_parameters();
        problem_.linearize(
            primal,
            parameters,
            Eigen::VectorXd::Zero(
                problem_.info.number_of_eq_constraints));
        const Eigen::MatrixXd jacobian =
            problem_.dense_constraint_jacobian();
        for (Index row = 0; row < jacobian.rows(); ++row)
        {
            for (Index column = 0;
                 column < jacobian.cols(); ++column)
            {
                if (std::abs(jacobian(row, column)) > 1e-16)
                    jacobian_pattern_.emplace_back(row, column);
            }
        }
        result_.jacobian_nonzeros =
            static_cast<Index>(jacobian_pattern_.size());

        const Index parameter_offset =
            problem_.info.number_of_primal_variables;
        for (Index variable = 0; variable < parameter_offset; ++variable)
            hessian_pattern_.emplace_back(variable, variable);
        for (Index parameter = 0;
             parameter < problem_.config.parameters; ++parameter)
        {
            for (Index variable = 0; variable < parameter_offset; ++variable)
            {
                hessian_pattern_.emplace_back(
                    parameter_offset + parameter, variable);
            }
            for (Index other = 0; other <= parameter; ++other)
            {
                hessian_pattern_.emplace_back(
                    parameter_offset + parameter,
                    parameter_offset + other);
            }
        }
    }

    bool get_nlp_info(
        Ipopt::Index &n,
        Ipopt::Index &m,
        Ipopt::Index &nnz_jac_g,
        Ipopt::Index &nnz_h_lag,
        IndexStyleEnum &index_style) override
    {
        n = static_cast<Ipopt::Index>(
            problem_.info.number_of_primal_variables +
            problem_.config.parameters);
        m = static_cast<Ipopt::Index>(
            problem_.info.number_of_eq_constraints);
        nnz_jac_g =
            static_cast<Ipopt::Index>(
                jacobian_pattern_.size());
        nnz_h_lag =
            static_cast<Ipopt::Index>(hessian_pattern_.size());
        index_style = C_STYLE;
        return true;
    }

    bool get_bounds_info(
        Ipopt::Index n,
        Ipopt::Number *x_l,
        Ipopt::Number *x_u,
        Ipopt::Index m,
        Ipopt::Number *g_l,
        Ipopt::Number *g_u) override
    {
        for (Ipopt::Index i = 0; i < n; ++i)
        {
            x_l[i] = -2e19;
            x_u[i] = 2e19;
        }
        for (Ipopt::Index i = 0; i < m; ++i)
        {
            g_l[i] = 0.0;
            g_u[i] = 0.0;
        }
        return true;
    }

    bool get_starting_point(
        Ipopt::Index n,
        bool init_x,
        Ipopt::Number *x,
        bool init_z,
        Ipopt::Number *,
        Ipopt::Number *,
        Ipopt::Index,
        bool init_lambda,
        Ipopt::Number *) override
    {
        if (!init_x || init_z || init_lambda)
            return false;
        const Eigen::VectorXd primal =
            problem_.initial_primal();
        const Eigen::VectorXd parameters =
            problem_.initial_parameters();
        for (Index i = 0; i < primal.size(); ++i)
            x[i] = primal(i);
        for (Index i = 0; i < parameters.size(); ++i)
            x[primal.size() + i] = parameters(i);
        return static_cast<Index>(n) ==
               primal.size() + parameters.size();
    }

    bool eval_f(
        Ipopt::Index,
        const Ipopt::Number *x,
        bool,
        Ipopt::Number &objective) override
    {
        Eigen::VectorXd primal;
        Eigen::VectorXd parameters;
        split(x, primal, parameters);
        objective =
            problem_.values(primal, parameters).objective;
        return std::isfinite(objective);
    }

    bool eval_grad_f(
        Ipopt::Index,
        const Ipopt::Number *x,
        bool,
        Ipopt::Number *gradient) override
    {
        Eigen::VectorXd primal;
        Eigen::VectorXd parameters;
        split(x, primal, parameters);
        problem_.linearize(
            primal,
            parameters,
            Eigen::VectorXd::Zero(
                problem_.info.number_of_eq_constraints));
        for (Index i = 0; i < primal.size(); ++i)
            gradient[i] = problem_.gradient_w(i);
        for (Index i = 0; i < parameters.size(); ++i)
            gradient[primal.size() + i] =
                problem_.gradient_parameters(i);
        return true;
    }

    bool eval_g(
        Ipopt::Index,
        const Ipopt::Number *x,
        bool,
        Ipopt::Index,
        Ipopt::Number *constraints) override
    {
        Eigen::VectorXd primal;
        Eigen::VectorXd parameters;
        split(x, primal, parameters);
        const Values values =
            problem_.values(primal, parameters);
        for (Index i = 0;
             i < values.constraints.size(); ++i)
            constraints[i] = values.constraints(i);
        return true;
    }

    bool eval_jac_g(
        Ipopt::Index,
        const Ipopt::Number *x,
        bool,
        Ipopt::Index m,
        Ipopt::Index nele_jac,
        Ipopt::Index *i_row,
        Ipopt::Index *j_col,
        Ipopt::Number *values) override
    {
        if (nele_jac !=
            static_cast<Ipopt::Index>(
                jacobian_pattern_.size()))
            return false;
        if (values == nullptr)
        {
            for (std::size_t cursor = 0;
                 cursor < jacobian_pattern_.size();
                 ++cursor)
            {
                i_row[cursor] =
                    static_cast<Ipopt::Index>(
                        jacobian_pattern_[cursor].first);
                j_col[cursor] =
                    static_cast<Ipopt::Index>(
                        jacobian_pattern_[cursor].second);
            }
            return true;
        }

        Eigen::VectorXd primal;
        Eigen::VectorXd parameters;
        split(x, primal, parameters);
        problem_.linearize(
            primal,
            parameters,
            Eigen::VectorXd::Zero(
                problem_.info.number_of_eq_constraints));
        const Eigen::MatrixXd jacobian =
            problem_.dense_constraint_jacobian();
        for (std::size_t cursor = 0;
             cursor < jacobian_pattern_.size();
             ++cursor)
        {
            values[cursor] =
                jacobian(
                    jacobian_pattern_[cursor].first,
                    jacobian_pattern_[cursor].second);
        }
        return true;
    }

    bool eval_h(
        Ipopt::Index,
        const Ipopt::Number *x,
        bool,
        Ipopt::Number obj_factor,
        Ipopt::Index m,
        const Ipopt::Number *lambda,
        bool,
        Ipopt::Index nele_hess,
        Ipopt::Index *i_row,
        Ipopt::Index *j_col,
        Ipopt::Number *values) override
    {
        if (nele_hess
            != static_cast<Ipopt::Index>(hessian_pattern_.size()))
            return false;
        if (values == nullptr)
        {
            for (std::size_t cursor = 0;
                 cursor < hessian_pattern_.size(); ++cursor)
            {
                i_row[cursor] = static_cast<Ipopt::Index>(
                    hessian_pattern_[cursor].first);
                j_col[cursor] = static_cast<Ipopt::Index>(
                    hessian_pattern_[cursor].second);
            }
            return true;
        }

        Eigen::VectorXd primal;
        Eigen::VectorXd parameters;
        split(x, primal, parameters);
        Eigen::VectorXd multipliers(m);
        for (Index row = 0; row < m; ++row)
            multipliers(row) = lambda[row];
        problem_.linearize(primal, parameters, multipliers);

        std::size_t cursor = 0;
        for (Index stage = 0; stage < problem_.dims.K; ++stage)
        {
            const Index local = problem_.layout.controls[stage]
                              + problem_.layout.states[stage];
            for (Index variable = 0; variable < local; ++variable)
            {
                values[cursor++] = obj_factor
                    * problem_.hessian.RSQrqt[stage](variable, variable);
            }
        }
        for (Index parameter = 0;
             parameter < problem_.config.parameters; ++parameter)
        {
            for (Index variable = 0;
                 variable < problem_.info.number_of_primal_variables;
                 ++variable)
            {
                values[cursor++] = obj_factor
                    * problem_.border_primal(variable, parameter);
            }
            for (Index other = 0; other <= parameter; ++other)
            {
                values[cursor++] = obj_factor
                    * problem_.border_hessian(parameter, other);
            }
        }
        if (cursor != hessian_pattern_.size())
            return false;
        return true;
    }

    void finalize_solution(
        Ipopt::SolverReturn status,
        Ipopt::Index,
        const Ipopt::Number *x,
        const Ipopt::Number *,
        const Ipopt::Number *,
        Ipopt::Index m,
        const Ipopt::Number *,
        const Ipopt::Number *lambda,
        Ipopt::Number objective,
        const Ipopt::IpoptData *ip_data,
        Ipopt::IpoptCalculatedQuantities *) override
    {
        result_.status = static_cast<int>(status);
        result_.objective = objective;
        result_.iterations =
            ip_data == nullptr
                ? -1
                : static_cast<int>(ip_data->iter_count());
        result_.primal.resize(
            problem_.info.number_of_primal_variables);
        result_.parameters.resize(
            problem_.config.parameters);
        result_.multipliers.resize(m);
        for (Index i = 0;
             i < result_.primal.size(); ++i)
            result_.primal(i) = x[i];
        for (Index i = 0;
             i < result_.parameters.size(); ++i)
        {
            result_.parameters(i) =
                x[result_.primal.size() + i];
        }
        for (Index i = 0; i < m; ++i)
            result_.multipliers(i) = lambda[i];
        result_.constraint_inf =
            max_abs(
                problem_
                    .values(
                        result_.primal,
                        result_.parameters)
                    .constraints);
    }

private:
    void split(
        const Ipopt::Number *x,
        Eigen::VectorXd &primal,
        Eigen::VectorXd &parameters) const
    {
        primal.resize(
            problem_.info.number_of_primal_variables);
        parameters.resize(problem_.config.parameters);
        for (Index i = 0; i < primal.size(); ++i)
            primal(i) = x[i];
        for (Index i = 0; i < parameters.size(); ++i)
            parameters(i) = x[primal.size() + i];
    }

    NonlinearProblem &problem_;
    IpoptResult &result_;
    std::vector<std::pair<Index, Index>>
        jacobian_pattern_;
    std::vector<std::pair<Index, Index>>
        hessian_pattern_;
};

IpoptResult solve_ipopt(NonlinearProblem &problem)
{
    IpoptResult result;
    Ipopt::SmartPtr<Ipopt::IpoptApplication> application =
        IpoptApplicationFactory();
    application->Options()->SetIntegerValue(
        "print_level", 0);
    application->Options()->SetStringValue(
        "sb", "yes");
    application->Options()->SetStringValue(
        "hessian_approximation", "exact");
    application->Options()->SetStringValue(
        "linear_solver", problem.config.ipopt_linear_solver);
    application->Options()->SetNumericValue(
        "tol", problem.config.tolerance);
    application->Options()->SetIntegerValue(
        "max_iter",
        static_cast<int>(problem.config.max_iterations * 3));
    application->Options()->SetStringValue(
        "mu_strategy", "adaptive");
    const auto initialization = application->Initialize();
    if (initialization != Ipopt::Solve_Succeeded)
    {
        result.status = static_cast<int>(initialization);
        return result;
    }

    Ipopt::SmartPtr<NonlinearTnlp> raw_problem =
        new NonlinearTnlp(problem, result);
    Ipopt::SmartPtr<Ipopt::TNLP> tnlp = raw_problem;
    std::vector<double> elapsed;
    elapsed.reserve(static_cast<std::size_t>(problem.config.repeats));
    Ipopt::ApplicationReturnStatus status = Ipopt::Internal_Error;
    for (Index repeat = 0; repeat < problem.config.repeats; ++repeat)
    {
        const auto start = std::chrono::steady_clock::now();
        status = application->OptimizeTNLP(tnlp);
        const auto stop = std::chrono::steady_clock::now();
        elapsed.push_back(
            std::chrono::duration<double, std::milli>(stop - start).count());
        if (status != Ipopt::Solve_Succeeded
            && status != Ipopt::Solved_To_Acceptable_Level)
            break;
    }
    result.elapsed_ms = median(elapsed);
    if (result.status == -1)
        result.status = static_cast<int>(status);
    return result;
}

Config parse_arguments(int argc, char **argv)
{
    Config config;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument(argv[i]);
        auto next_string = [&](const char *name) {
            if (i + 1 >= argc)
            {
                throw std::runtime_error(
                    std::string("Missing value for ") + name);
            }
            return std::string(argv[++i]);
        };
        auto next_integer = [&](const char *name) {
            return static_cast<Index>(
                std::stoll(next_string(name)));
        };
        auto next_double = [&](const char *name) {
            return std::stod(next_string(name));
        };

        if (argument == "--phase-nodes")
            config.phase_nodes =
                parse_index_list(
                    next_string("--phase-nodes"));
        else if (argument == "--phase-nx")
            config.phase_nx =
                parse_index_list(
                    next_string("--phase-nx"));
        else if (argument == "--phase-nu")
            config.phase_nu =
                parse_index_list(
                    next_string("--phase-nu"));
        else if (argument == "--parameters")
            config.parameters =
                next_integer("--parameters");
        else if (argument == "--collocation-degree")
            config.collocation_degree =
                next_integer("--collocation-degree");
        else if (argument == "--max-iterations")
            config.max_iterations =
                next_integer("--max-iterations");
        else if (argument == "--repeats")
            config.repeats =
                next_integer("--repeats");
        else if (argument == "--tolerance")
            config.tolerance =
                next_double("--tolerance");
        else if (argument == "--regularization")
            config.regularization =
                next_double("--regularization");
        else if (argument == "--no-dense-validation")
            config.dense_validation = false;
        else if (argument == "--ipopt")
            config.run_ipopt = true;
        else if (argument == "--native-ipm")
            config.run_native_ipm = true;
        else if (argument == "--full-solvers")
        {
            config.run_native_ipm = true;
            config.run_ipopt = true;
        }
        else if (argument == "--ipopt-linear-solver")
            config.ipopt_linear_solver =
                next_string("--ipopt-linear-solver");
        else if (argument == "--help")
        {
            std::cout
                << "Usage: nonlinear_global_parameter_sqp_benchmark "
                   "[options]\n"
                << "  --phase-nodes n0,n1,...\n"
                << "  --phase-nx nx0,nx1,...\n"
                << "  --phase-nu nu0,nu1,...\n"
                << "  --parameters p\n"
                << "  --collocation-degree 1|2|3\n"
                << "  --max-iterations n\n"
                << "  --repeats n\n"
                << "  --tolerance value\n"
                << "  --regularization value\n"
                << "  --no-dense-validation\n"
                << "  --ipopt\n"
                << "  --native-ipm\n"
                << "  --full-solvers\n"
                << "  --ipopt-linear-solver {mumps|ma57|...}\n";
            std::exit(0);
        }
        else
        {
            throw std::runtime_error(
                "Unknown argument: " + argument);
        }
    }

    const std::size_t phases = config.phase_nodes.size();
    if (phases == 0 ||
        config.phase_nx.size() != phases ||
        config.phase_nu.size() != phases)
    {
        throw std::runtime_error(
            "phase-nodes, phase-nx, and phase-nu must "
            "have the same nonzero length");
    }
    for (std::size_t phase = 0; phase < phases; ++phase)
    {
        if (config.phase_nodes[phase] < 2)
            throw std::runtime_error(
                "Every phase must contain at least two nodes");
        if (config.phase_nx[phase] <= 0)
            throw std::runtime_error(
                "Every phase must have positive nx");
        if (config.phase_nu[phase] < 0)
            throw std::runtime_error(
                "Every phase must have nonnegative nu");
    }
    if (config.parameters < 0 ||
        config.max_iterations <= 0 ||
        config.repeats <= 0 ||
        config.tolerance <= 0.0 ||
        config.regularization < 0.0)
    {
        throw std::runtime_error(
            "Invalid parameter, iteration, tolerance, "
            "repeat, or regularization value");
    }
    if (config.collocation_degree < 1 ||
        config.collocation_degree > 3)
    {
        throw std::runtime_error(
            "Collocation degree must be 1, 2, or 3");
    }
    return config;
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const Config config = parse_arguments(argc, argv);
        NonlinearProblem problem(config);
        AugSystemSolver<OcpType> solver(problem.info);

        const DerivativeCheck derivative_check =
            check_derivatives(problem);
        const SqpResult validation_result =
            solve_sqp(problem, solver);
        const bool dense_validation =
            problem.config.dense_validation;
        problem.config.dense_validation = false;
        std::vector<double> elapsed_times;
        std::vector<double> kkt_times;
        std::vector<double> parameter_rhs_times;
        SqpResult sqp_result;
        for (Index repeat = 0;
             repeat < config.repeats; ++repeat)
        {
            sqp_result = solve_sqp(problem, solver);
            elapsed_times.push_back(
                sqp_result.elapsed_ms);
            kkt_times.push_back(sqp_result.kkt_ms);
            parameter_rhs_times.push_back(
                sqp_result.parameter_rhs_ms);
        }
        sqp_result.elapsed_ms = median(elapsed_times);
        sqp_result.kkt_ms = median(kkt_times);
        sqp_result.parameter_rhs_ms =
            median(parameter_rhs_times);
        sqp_result.dense_step_difference =
            validation_result.dense_step_difference;
        problem.config.dense_validation =
            dense_validation;

        NativeIpmResult native_ipm_result;
        if (config.run_native_ipm)
            native_ipm_result = solve_native_ipm(problem);

        IpoptResult ipopt_result;
        if (config.run_ipopt)
            ipopt_result = solve_ipopt(problem);

        const double native_primal_difference =
            config.run_native_ipm
                    && native_ipm_result.primal.size()
                        == sqp_result.primal.size()
                ? max_abs(native_ipm_result.primal - sqp_result.primal)
                : std::numeric_limits<double>::quiet_NaN();
        const double native_parameter_difference =
            config.run_native_ipm
                    && native_ipm_result.parameters.size()
                        == sqp_result.parameters.size()
                ? max_abs(
                      native_ipm_result.parameters - sqp_result.parameters)
                : std::numeric_limits<double>::quiet_NaN();

        const double ipopt_primal_difference =
            config.run_ipopt &&
                    ipopt_result.primal.size() ==
                        sqp_result.primal.size()
                ? max_abs(
                      ipopt_result.primal -
                      sqp_result.primal)
                : std::numeric_limits<double>::quiet_NaN();
        const double ipopt_parameter_difference =
            config.run_ipopt &&
                    ipopt_result.parameters.size() ==
                        sqp_result.parameters.size()
                ? max_abs(
                      ipopt_result.parameters -
                      sqp_result.parameters)
                : std::numeric_limits<double>::quiet_NaN();

        std::cout
            << "phases,stages,phase_nodes,phase_nx,phase_nu,"
               "collocation_degree,parameters,primal,constraints,"
               "sqp_converged,sqp_iterations,sqp_ms,sqp_kkt_ms,"
               "sqp_parameter_rhs_ms,line_search_evaluations,"
               "objective,primal_inf,dual_inf,"
               "linear_step_residual,dense_step_difference,"
               "objective_derivative_error,"
               "constraint_derivative_error,"
               "native_ipm_ms,native_ipm_status,native_ipm_iterations,"
               "native_ipm_objective,native_ipm_constraint_inf,"
               "native_ipm_dual_inf,native_ipm_primal_difference,"
               "native_ipm_parameter_difference,"
               "ipopt_ms,ipopt_status,ipopt_iterations,"
               "ipopt_jacobian_nonzeros,"
               "ipopt_objective,ipopt_constraint_inf,"
               "ipopt_primal_difference,"
               "ipopt_parameter_difference,"
               "ipopt_over_native_ipm\n";
        std::cout
            << std::setprecision(12)
            << config.phase_nodes.size() << ','
            << problem.dims.K << ','
            << join(config.phase_nodes) << ','
            << join(config.phase_nx) << ','
            << join(config.phase_nu) << ','
            << config.collocation_degree << ','
            << config.parameters << ','
            << problem.info.number_of_primal_variables << ','
            << problem.info.number_of_eq_constraints << ','
            << (sqp_result.converged ? 1 : 0) << ','
            << sqp_result.iterations << ','
            << sqp_result.elapsed_ms << ','
            << sqp_result.kkt_ms << ','
            << sqp_result.parameter_rhs_ms << ','
            << sqp_result.line_search_evaluations << ','
            << sqp_result.objective << ','
            << sqp_result.primal_inf << ','
            << sqp_result.dual_inf << ','
            << sqp_result.linear_residual << ','
            << sqp_result.dense_step_difference << ','
            << derivative_check.objective_error << ','
            << derivative_check.constraint_error << ','
            << native_ipm_result.elapsed_ms << ','
            << native_ipm_result.status << ','
            << native_ipm_result.iterations << ','
            << native_ipm_result.objective << ','
            << native_ipm_result.constraint_inf << ','
            << native_ipm_result.dual_inf << ','
            << native_primal_difference << ','
            << native_parameter_difference << ','
            << ipopt_result.elapsed_ms << ','
            << ipopt_result.status << ','
            << ipopt_result.iterations << ','
            << ipopt_result.jacobian_nonzeros << ','
            << ipopt_result.objective << ','
            << ipopt_result.constraint_inf << ','
            << ipopt_primal_difference << ','
            << ipopt_parameter_difference << ','
            << (config.run_native_ipm && config.run_ipopt
                    ? ipopt_result.elapsed_ms
                        / native_ipm_result.elapsed_ms
                    : std::numeric_limits<double>::quiet_NaN())
            << '\n';
        const bool native_ok =
            !config.run_native_ipm || native_ipm_result.converged;
        const bool ipopt_ok =
            !config.run_ipopt
            || ipopt_result.status == static_cast<int>(Ipopt::SUCCESS)
            || ipopt_result.status
                == static_cast<int>(Ipopt::STOP_AT_ACCEPTABLE_POINT);
        return sqp_result.converged && native_ok && ipopt_ok ? 0 : 2;
    }
    catch (const std::exception &exception)
    {
        std::cerr << "error: " << exception.what() << '\n';
        return 1;
    }
}
