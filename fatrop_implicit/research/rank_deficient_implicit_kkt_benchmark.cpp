#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/ocp/aug_system_solver.hpp"
#include "fatrop/ocp/dims.hpp"
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/problem_info.hpp"
#include "fatrop/ocp/type.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>

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
using fatrop::ImplicitOcpType;
using fatrop::Index;
using fatrop::Jacobian;
using fatrop::LinsolReturnFlag;
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
    Index rank_deficiency = 2;
    Index parameters = 3;
    Index repeats = 5;
    double cross_hessian_scale = 0.0;
    double negative_curvature = 0.0;
    double primal_regularization = 0.0;
    double inertia_margin = 0.1;
    double rank_tolerance = 1e-10;
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
            text.substr(
                begin,
                end == std::string::npos
                    ? std::string::npos
                    : end - begin);
        if (token.empty())
            throw std::runtime_error(
                option + " contains an empty entry");
        const long value = std::stol(token);
        if (value < 0 ||
            value > std::numeric_limits<Index>::max())
            throw std::runtime_error(
                option + " contains an invalid value");
        values.push_back(static_cast<Index>(value));
        if (end == std::string::npos)
            break;
        begin = end + 1;
    }
    return values;
}

std::string encode_list(const std::vector<Index> &values)
{
    std::string result;
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        if (i > 0)
            result += ':';
        result += std::to_string(values[i]);
    }
    return result;
}

Config parse_arguments(int argc, char **argv)
{
    Config config;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument(argv[i]);
        const auto next = [&](const std::string &option) {
            if (i + 1 >= argc)
                throw std::runtime_error(
                    option + " requires a value");
            return std::string(argv[++i]);
        };
        const auto next_integer = [&](const std::string &option) {
            return static_cast<Index>(
                std::stol(next(option)));
        };
        const auto next_double = [&](const std::string &option) {
            return std::stod(next(option));
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
        else if (argument == "--rank-deficiency")
            config.rank_deficiency = next_integer(argument);
        else if (argument == "--parameters")
            config.parameters = next_integer(argument);
        else if (argument == "--repeats")
            config.repeats = next_integer(argument);
        else if (argument == "--cross-hessian-scale")
            config.cross_hessian_scale =
                next_double(argument);
        else if (argument == "--negative-curvature")
            config.negative_curvature =
                next_double(argument);
        else if (argument == "--primal-regularization")
            config.primal_regularization =
                next_double(argument);
        else if (argument == "--inertia-margin")
            config.inertia_margin = next_double(argument);
        else if (argument == "--rank-tolerance")
            config.rank_tolerance = next_double(argument);
        else if (argument == "--no-dense-validation")
            config.dense_validation = false;
        else if (argument == "--help")
        {
            std::cout
                << "Usage: rank_deficient_implicit_kkt_benchmark"
                   " [options]\n"
                << "  --phase-nodes LIST\n"
                << "  --phase-nx LIST\n"
                << "  --phase-nu LIST\n"
                << "  --rank-deficiency N\n"
                << "  --parameters N\n"
                << "  --cross-hessian-scale VALUE\n"
                << "  --negative-curvature VALUE\n"
                << "  --primal-regularization VALUE\n"
                << "  --inertia-margin VALUE\n"
                << "  --rank-tolerance VALUE\n"
                << "  --repeats N\n"
                << "  --no-dense-validation\n";
            std::exit(0);
        }
        else
            throw std::runtime_error(
                "Unknown option: " + argument);
    }

    const std::size_t phases = config.phase_nodes.size();
    if (phases == 0 ||
        config.phase_nx.size() != phases ||
        config.phase_nu.size() != phases)
    {
        throw std::runtime_error(
            "All phase lists must have the same nonzero length");
    }
    if (config.rank_deficiency < 0 ||
        config.parameters < 0 ||
        config.repeats < 1 ||
        config.cross_hessian_scale < 0.0 ||
        config.primal_regularization < 0.0 ||
        config.inertia_margin <= 0.0 ||
        config.rank_tolerance <= 0.0)
    {
        throw std::runtime_error(
            "Invalid scalar configuration value");
    }
    for (std::size_t phase = 0; phase < phases; ++phase)
    {
        if (config.phase_nodes[phase] < 2 ||
            config.phase_nx[phase] < 1 ||
            config.phase_nu[phase] < 0)
        {
            throw std::runtime_error(
                "Each phase needs nodes>=2, nx>=1, nu>=0");
        }
    }
    return config;
}

struct Layout
{
    explicit Layout(const Config &config)
    {
        Index stage = 0;
        for (Index phase = 0;
             phase <
             static_cast<Index>(config.phase_nodes.size());
             ++phase)
        {
            for (Index local = 0;
                 local < config.phase_nodes[phase];
                 ++local)
            {
                stage_phase.push_back(phase);
                stage_local.push_back(local);
                ++stage;
            }
        }
        stages = stage;
    }

    bool is_linkage(Index transition) const
    {
        return stage_phase[transition] !=
               stage_phase[transition + 1];
    }

    Index stages = 0;
    std::vector<Index> stage_phase;
    std::vector<Index> stage_local;
};

double deterministic_value(
    Index row, Index column, double scale)
{
    return scale *
           (std::sin(
                0.17 * static_cast<double>(row + 1) +
                0.11 * static_cast<double>(column + 2)) +
            0.55 *
                std::cos(
                    0.07 * static_cast<double>(
                               (row + 2) * (column + 1))));
}

double max_abs(const Eigen::VectorXd &value)
{
    return value.size() == 0
               ? 0.0
               : value.cwiseAbs().maxCoeff();
}

double median(std::vector<double> values)
{
    if (values.empty())
        return 0.0;
    const std::size_t middle = values.size() / 2;
    std::nth_element(
        values.begin(), values.begin() + middle, values.end());
    if (values.size() % 2 == 1)
        return values[middle];
    const double upper = values[middle];
    std::nth_element(
        values.begin(), values.begin() + middle - 1,
        values.begin() + middle);
    return 0.5 * (values[middle - 1] + upper);
}

Eigen::VectorXd to_eigen(const VecRealAllocated &value)
{
    Eigen::VectorXd result(value.m());
    for (Index i = 0; i < value.m(); ++i)
        result(i) = value(i);
    return result;
}

struct Inertia
{
    Index positive = 0;
    Index negative = 0;
    Index zero = 0;
};

Inertia inertia(
    const Eigen::MatrixXd &matrix, double tolerance)
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
        matrix, Eigen::EigenvaluesOnly);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error(
            "Dense inertia eigensolve failed");
    Inertia result;
    for (double eigenvalue : solver.eigenvalues())
    {
        if (eigenvalue > tolerance)
            ++result.positive;
        else if (eigenvalue < -tolerance)
            ++result.negative;
        else
            ++result.zero;
    }
    return result;
}

struct Problem
{
    explicit Problem(const Config &config)
        : config(config),
          layout(config),
          controls(static_cast<std::size_t>(layout.stages)),
          states(static_cast<std::size_t>(layout.stages)),
          equalities(
              static_cast<std::size_t>(layout.stages), 0),
          inequalities(
              static_cast<std::size_t>(layout.stages), 0),
          dims(make_dims()),
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
              config.parameters, config.parameters),
          rhs_parameters(config.parameters),
          dense_hessian(
              Eigen::MatrixXd::Zero(
                  info.number_of_primal_variables,
                  info.number_of_primal_variables)),
          dense_jacobian(
              Eigen::MatrixXd::Zero(
                  info.number_of_eq_constraints,
                  info.number_of_primal_variables))
    {
        D_x = 0.0;
        D_s = 0.0;
        rhs_x = 0.0;
        rhs_g = 0.0;
        border_primal.setZero();
        border_constraints.setZero();
        border_hessian.setZero();
        rhs_parameters.setZero();

        populate_hessian();
        populate_dynamics();
        populate_parameter_terms();
        analyze_and_set_regularization();
    }

    ProblemDims make_dims()
    {
        for (Index stage = 0;
             stage < layout.stages; ++stage)
        {
            const Index phase = layout.stage_phase[stage];
            states[stage] = config.phase_nx[phase];
            controls[stage] =
                stage + 1 < layout.stages
                    ? config.phase_nu[phase]
                    : 0;
        }
        return ProblemDims(
            layout.stages, controls, states,
            equalities, inequalities);
    }

    void populate_hessian()
    {
        for (Index stage = 0;
             stage < layout.stages; ++stage)
        {
            const Index phase = layout.stage_phase[stage];
            const Index variables =
                controls[stage] + states[stage];
            const Index offset =
                info.offsets_primal_u[stage];
            hessian.RSQrqt[stage] = 0.0;
            for (Index row = 0; row < variables; ++row)
            {
                const double diagonal =
                    3.0 +
                    0.04 *
                        static_cast<double>(
                            (row + 2 * stage + phase) % 11) -
                    config.negative_curvature;
                hessian.RSQrqt[stage](row, row) =
                    diagonal;
                dense_hessian(
                    offset + row, offset + row) =
                    diagonal;
                rhs_x(offset + row) =
                    deterministic_value(
                        offset + row, phase, 0.09);
                for (Index parameter = 0;
                     parameter < config.parameters;
                     ++parameter)
                {
                    border_primal(
                        offset + row, parameter) =
                        deterministic_value(
                            offset + row,
                            parameter + 3 * phase,
                            0.004);
                }
                if (row + 1 < variables)
                {
                    const double coupling =
                        0.015 /
                        (1.0 + static_cast<double>(row));
                    hessian.RSQrqt[stage](
                        row, row + 1) = coupling;
                    hessian.RSQrqt[stage](
                        row + 1, row) = coupling;
                    dense_hessian(
                        offset + row,
                        offset + row + 1) = coupling;
                    dense_hessian(
                        offset + row + 1,
                        offset + row) = coupling;
                }
            }
        }

        for (Index stage = 0;
             stage + 1 < layout.stages; ++stage)
        {
            const Index current_variables =
                controls[stage] + states[stage];
            const Index next_states = states[stage + 1];
            const Index current_offset =
                info.offsets_primal_u[stage];
            const Index next_offset =
                info.offsets_primal_x[stage + 1];
            hessian.FuFx[stage] = 0.0;
            for (Index row = 0;
                 row < current_variables; ++row)
            {
                for (Index column = 0;
                     column < next_states; ++column)
                {
                    const double value =
                        deterministic_value(
                            row + 5 * stage,
                            column + 2,
                            config.cross_hessian_scale);
                    hessian.FuFx[stage](
                        row, column) = value;
                    dense_hessian(
                        current_offset + row,
                        next_offset + column) = value;
                    dense_hessian(
                        next_offset + column,
                        current_offset + row) = value;
                }
            }
        }
    }

    void populate_dynamics()
    {
        expected_ranks.reserve(
            static_cast<std::size_t>(layout.stages - 1));
        for (Index stage = 0;
             stage + 1 < layout.stages; ++stage)
        {
            const Index phase = layout.stage_phase[stage];
            const Index current_variables =
                controls[stage] + states[stage];
            const Index next_states = states[stage + 1];
            const Index missing =
                std::min(
                    config.rank_deficiency, next_states);
            const Index rank = next_states - missing;
            if (missing > current_variables)
            {
                throw std::runtime_error(
                    "Rank deficiency leaves more free incoming "
                    "components than current controls and states");
            }
            expected_ranks.push_back(rank);
            jacobian.BAbt[stage] = 0.0;
            jacobian.Jt[stage] = 0.0;

            std::vector<Index> constraint_order(
                static_cast<std::size_t>(next_states));
            std::vector<Index> state_order(
                static_cast<std::size_t>(next_states));
            std::iota(
                constraint_order.begin(),
                constraint_order.end(), 0);
            std::iota(
                state_order.begin(), state_order.end(), 0);
            if (next_states > 1)
            {
                const Index shift =
                    (stage + 1) % next_states;
                std::rotate(
                    constraint_order.begin(),
                    constraint_order.begin() + shift,
                    constraint_order.end());
                std::reverse(
                    constraint_order.begin(),
                    constraint_order.end());
                std::rotate(
                    state_order.begin(),
                    state_order.begin() +
                        ((2 * stage + 1) % next_states),
                    state_order.end());
            }

            std::vector<bool> constrained_by_next_state(
                static_cast<std::size_t>(next_states), false);
            for (Index pivot = 0; pivot < rank; ++pivot)
            {
                const Index equation =
                    constraint_order[pivot];
                const Index state = state_order[pivot];
                const double value =
                    -1.0 -
                    0.03 * static_cast<double>(pivot);
                jacobian.Jt[stage](
                    state, equation) = value;
                constrained_by_next_state[equation] = true;
            }

            Index missing_index = 0;
            for (Index equation = 0;
                 equation < next_states; ++equation)
            {
                for (Index variable = 0;
                     variable < current_variables; ++variable)
                {
                    jacobian.BAbt[stage](
                        variable, equation) =
                        deterministic_value(
                            variable + 3 * stage,
                            equation + phase,
                            layout.is_linkage(stage)
                                ? 0.018
                                : 0.012);
                }
                if (!constrained_by_next_state[equation])
                {
                    jacobian.BAbt[stage](
                        missing_index, equation) +=
                        1.15 +
                        0.02 *
                            static_cast<double>(missing_index);
                    ++missing_index;
                }

                const Index constraint =
                    info.offsets_g_eq_dyn[stage] + equation;
                rhs_g(constraint) =
                    deterministic_value(
                        constraint, phase + 7, 0.05);
                for (Index parameter = 0;
                     parameter < config.parameters;
                     ++parameter)
                {
                    border_constraints(
                        constraint, parameter) =
                        deterministic_value(
                            constraint + stage,
                            parameter + phase,
                            layout.is_linkage(stage)
                                ? 0.026
                                : 0.019);
                }
            }

            const Index current_offset =
                info.offsets_primal_u[stage];
            const Index next_offset =
                info.offsets_primal_x[stage + 1];
            const Index constraint_offset =
                info.offsets_g_eq_dyn[stage];
            for (Index equation = 0;
                 equation < next_states; ++equation)
            {
                for (Index variable = 0;
                     variable < current_variables; ++variable)
                {
                    dense_jacobian(
                        constraint_offset + equation,
                        current_offset + variable) =
                        jacobian.BAbt[stage](
                            variable, equation);
                }
                for (Index state = 0;
                     state < next_states; ++state)
                {
                    dense_jacobian(
                        constraint_offset + equation,
                        next_offset + state) =
                        jacobian.Jt[stage](
                            state, equation);
                }
            }
        }
    }

    void populate_parameter_terms()
    {
        for (Index parameter = 0;
             parameter < config.parameters; ++parameter)
        {
            rhs_parameters(parameter) =
                deterministic_value(
                    parameter, layout.stages, 0.08);
            for (Index other = 0;
                 other < config.parameters; ++other)
            {
                border_hessian(parameter, other) =
                    parameter == other
                        ? 5.0 +
                              0.12 *
                                  static_cast<double>(
                                      layout.stages + parameter)
                        : 0.004 /
                              (1.0 +
                               std::abs(parameter - other));
            }
        }
    }

    void analyze_and_set_regularization()
    {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            dense_jacobian,
            Eigen::ComputeFullU | Eigen::ComputeFullV);
        svd.setThreshold(config.rank_tolerance);
        jacobian_rank = static_cast<Index>(svd.rank());
        if (jacobian_rank !=
            info.number_of_eq_constraints)
        {
            throw std::runtime_error(
                "Generated full constraint Jacobian is rank "
                "deficient");
        }

        const Index nullity =
            info.number_of_primal_variables - jacobian_rank;
        if (nullity > 0)
        {
            const Eigen::MatrixXd nullspace =
                svd.matrixV().rightCols(nullity);
            const Eigen::MatrixXd reduced =
                nullspace.transpose() *
                dense_hessian * nullspace;
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>
                eigensolver(
                    reduced, Eigen::EigenvaluesOnly);
            if (eigensolver.info() != Eigen::Success)
                throw std::runtime_error(
                    "Reduced-Hessian eigensolve failed");
            reduced_min_before =
                eigensolver.eigenvalues().minCoeff();
        }
        else
        {
            reduced_min_before =
                std::numeric_limits<double>::infinity();
        }

        regularization =
            config.primal_regularization;
        if (std::isfinite(reduced_min_before))
        {
            regularization =
                std::max(
                    regularization,
                    config.inertia_margin -
                        reduced_min_before);
        }
        regularization =
            std::max(0.0, regularization);
        reduced_min_after =
            std::isfinite(reduced_min_before)
                ? reduced_min_before + regularization
                : reduced_min_before;
        for (Index variable = 0;
             variable < info.number_of_primal_variables;
             ++variable)
            D_x(variable) = regularization;
    }

    Eigen::MatrixXd regularized_hessian() const
    {
        return dense_hessian +
               regularization *
                   Eigen::MatrixXd::Identity(
                       dense_hessian.rows(),
                       dense_hessian.cols());
    }

    Eigen::MatrixXd base_kkt() const
    {
        const Index primal =
            info.number_of_primal_variables;
        const Index constraints =
            info.number_of_eq_constraints;
        Eigen::MatrixXd matrix =
            Eigen::MatrixXd::Zero(
                primal + constraints,
                primal + constraints);
        matrix.topLeftCorner(primal, primal) =
            regularized_hessian();
        matrix.topRightCorner(primal, constraints) =
            dense_jacobian.transpose();
        matrix.bottomLeftCorner(constraints, primal) =
            dense_jacobian;
        return matrix;
    }

    Eigen::MatrixXd full_kkt() const
    {
        const Index primal =
            info.number_of_primal_variables;
        const Index constraints =
            info.number_of_eq_constraints;
        const Index parameters = config.parameters;
        Eigen::MatrixXd matrix =
            Eigen::MatrixXd::Zero(
                primal + constraints + parameters,
                primal + constraints + parameters);
        matrix.topLeftCorner(
            primal + constraints,
            primal + constraints) = base_kkt();
        if (parameters > 0)
        {
            matrix.block(
                0, primal + constraints,
                primal, parameters) =
                border_primal;
            matrix.block(
                primal + constraints, 0,
                parameters, primal) =
                border_primal.transpose();
            matrix.block(
                primal, primal + constraints,
                constraints, parameters) =
                border_constraints;
            matrix.block(
                primal + constraints, primal,
                parameters, constraints) =
                border_constraints.transpose();
            matrix.bottomRightCorner(
                parameters, parameters) =
                border_hessian;
        }
        return matrix;
    }

    Config config;
    Layout layout;
    std::vector<Index> controls;
    std::vector<Index> states;
    std::vector<Index> equalities;
    std::vector<Index> inequalities;
    ProblemDims dims;
    ProblemInfo info;
    Jacobian<ImplicitOcpType> jacobian;
    Hessian<ImplicitOcpType> hessian;
    VecRealAllocated D_x;
    VecRealAllocated D_s;
    VecRealAllocated rhs_x;
    VecRealAllocated rhs_g;
    Eigen::MatrixXd border_primal;
    Eigen::MatrixXd border_constraints;
    Eigen::MatrixXd border_hessian;
    Eigen::VectorXd rhs_parameters;
    Eigen::MatrixXd dense_hessian;
    Eigen::MatrixXd dense_jacobian;
    std::vector<Index> expected_ranks;
    Index jacobian_rank = 0;
    double reduced_min_before = 0.0;
    double reduced_min_after = 0.0;
    double regularization = 0.0;
};

struct Result
{
    Eigen::VectorXd primal;
    Eigen::VectorXd multipliers;
    Eigen::VectorXd parameters;
    double elapsed_ms = 0.0;
    double parameter_rhs_ms = 0.0;
    std::vector<Index> detected_ranks;
};

double result_scale(const Result &result)
{
    return std::max(
        {max_abs(result.primal),
         max_abs(result.multipliers),
         max_abs(result.parameters)});
}

struct LiftedProblem
{
    explicit LiftedProblem(const Problem &source)
        : source(source),
          controls(
              static_cast<std::size_t>(
                  source.layout.stages)),
          states(source.states),
          equalities(
              static_cast<std::size_t>(
                  source.layout.stages), 0),
          inequalities(
              static_cast<std::size_t>(
                  source.layout.stages), 0),
          dims(make_dims()),
          info(dims),
          jacobian(dims),
          hessian(dims),
          D_x(info.number_of_primal_variables),
          D_s(info.number_of_slack_variables),
          rhs_x(info.number_of_primal_variables),
          rhs_g(info.number_of_eq_constraints),
          border_primal(
              info.number_of_primal_variables,
              source.config.parameters),
          border_constraints(
              info.number_of_eq_constraints,
              source.config.parameters),
          border_hessian(source.border_hessian),
          rhs_parameters(source.rhs_parameters)
    {
        D_x = 0.0;
        D_s = 0.0;
        rhs_x = 0.0;
        rhs_g = 0.0;
        border_primal.setZero();
        border_constraints.setZero();
        populate();
    }

    ProblemDims make_dims()
    {
        for (Index stage = 0;
             stage < source.layout.stages; ++stage)
        {
            const Index copies =
                stage + 1 < source.layout.stages
                    ? source.states[stage + 1]
                    : 0;
            controls[stage] =
                source.controls[stage] + copies;
            if (stage + 1 < source.layout.stages)
                equalities[stage] = copies;
        }
        return ProblemDims(
            source.layout.stages,
            controls,
            states,
            equalities,
            inequalities);
    }

    Index local_copy_count(Index stage) const
    {
        return stage + 1 < source.layout.stages
                   ? source.states[stage + 1]
                   : 0;
    }

    Index lifted_local_index(
        Index stage, Index original_local) const
    {
        const Index original_controls =
            source.controls[stage];
        if (original_local < original_controls)
            return original_local;
        return original_controls +
               local_copy_count(stage) +
               original_local - original_controls;
    }

    void populate()
    {
        for (Index stage = 0;
             stage < source.layout.stages; ++stage)
        {
            hessian.RSQrqt[stage] = 0.0;
            jacobian.Gg_eqt[stage] = 0.0;
            const Index original_variables =
                source.controls[stage] +
                source.states[stage];
            const Index original_offset =
                source.info.offsets_primal_u[stage];
            const Index lifted_offset =
                info.offsets_primal_u[stage];
            for (Index row = 0;
                 row < original_variables; ++row)
            {
                const Index lifted_row =
                    lifted_local_index(stage, row);
                rhs_x(lifted_offset + lifted_row) =
                    source.rhs_x(original_offset + row);
                D_x(lifted_offset + lifted_row) =
                    source.D_x(original_offset + row);
                for (Index parameter = 0;
                     parameter <
                     source.config.parameters;
                     ++parameter)
                {
                    border_primal(
                        lifted_offset + lifted_row,
                        parameter) =
                        source.border_primal(
                            original_offset + row,
                            parameter);
                }
                for (Index column = 0;
                     column < original_variables;
                     ++column)
                {
                    const Index lifted_column =
                        lifted_local_index(
                            stage, column);
                    hessian.RSQrqt[stage](
                        lifted_row, lifted_column) =
                        source.hessian.RSQrqt[stage](
                            row, column);
                }
            }

            if (stage + 1 >= source.layout.stages)
                continue;
            const Index original_controls =
                source.controls[stage];
            const Index next_states =
                source.states[stage + 1];
            const Index path_offset =
                info.offsets_g_eq_path[stage];
            const Index source_constraint_offset =
                source.info.offsets_g_eq_dyn[stage];

            for (Index equation = 0;
                 equation < next_states; ++equation)
            {
                rhs_g(path_offset + equation) =
                    source.rhs_g(
                        source_constraint_offset +
                        equation);
                for (Index parameter = 0;
                     parameter <
                     source.config.parameters;
                     ++parameter)
                {
                    border_constraints(
                        path_offset + equation,
                        parameter) =
                        source.border_constraints(
                            source_constraint_offset +
                                equation,
                            parameter);
                }
                for (Index variable = 0;
                     variable < original_variables;
                     ++variable)
                {
                    jacobian.Gg_eqt[stage](
                        lifted_local_index(
                            stage, variable),
                        equation) =
                        source.jacobian.BAbt[stage](
                            variable, equation);
                }
                for (Index state = 0;
                     state < next_states; ++state)
                {
                    jacobian.Gg_eqt[stage](
                        original_controls + state,
                        equation) =
                        source.jacobian.Jt[stage](
                            state, equation);
                }
            }

            jacobian.BAbt[stage] = 0.0;
            for (Index state = 0;
                 state < next_states; ++state)
            {
                jacobian.BAbt[stage](
                    original_controls + state,
                    state) = 1.0;
            }

            for (Index row = 0;
                 row < original_variables; ++row)
            {
                const Index lifted_row =
                    lifted_local_index(stage, row);
                for (Index state = 0;
                     state < next_states; ++state)
                {
                    const Index copy_column =
                        original_controls + state;
                    const double value =
                        source.hessian.FuFx[stage](
                            row, state);
                    hessian.RSQrqt[stage](
                        lifted_row, copy_column) +=
                        value;
                    hessian.RSQrqt[stage](
                        copy_column, lifted_row) +=
                        value;
                }
            }
        }
    }

    Result map_result(
        const Eigen::VectorXd &lifted_primal,
        const Eigen::VectorXd &lifted_multipliers,
        const Eigen::VectorXd &parameters) const
    {
        Result result;
        result.primal.resize(
            source.info.number_of_primal_variables);
        result.multipliers.resize(
            source.info.number_of_eq_constraints);
        result.parameters = parameters;
        for (Index stage = 0;
             stage < source.layout.stages; ++stage)
        {
            const Index original_variables =
                source.controls[stage] +
                source.states[stage];
            for (Index variable = 0;
                 variable < original_variables;
                 ++variable)
            {
                result.primal(
                    source.info.offsets_primal_u[stage] +
                    variable) =
                    lifted_primal(
                        info.offsets_primal_u[stage] +
                        lifted_local_index(
                            stage, variable));
            }
            if (stage + 1 < source.layout.stages)
            {
                const Index constraints =
                    source.states[stage + 1];
                for (Index equation = 0;
                     equation < constraints;
                     ++equation)
                {
                    result.multipliers(
                        source.info.offsets_g_eq_dyn[
                            stage] +
                        equation) =
                        lifted_multipliers(
                            info.offsets_g_eq_path[stage] +
                            equation);
                }
            }
        }
        return result;
    }

    const Problem &source;
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

Result solve_bordered(
    Problem &problem,
    AugSystemSolver<ImplicitOcpType> &solver)
{
    VecRealAllocated primal(
        problem.info.number_of_primal_variables);
    VecRealAllocated multipliers(
        problem.info.number_of_eq_constraints);
    primal = 0.0;
    multipliers = 0.0;

    const auto start =
        std::chrono::steady_clock::now();
    const auto factor_status = solver.solve(
        problem.info,
        problem.jacobian,
        problem.hessian,
        problem.D_x,
        problem.D_s,
        problem.rhs_x,
        problem.rhs_g,
        primal,
        multipliers);
    if (factor_status != LinsolReturnFlag::SUCCESS)
    {
        throw std::runtime_error(
            "Implicit FATROP factorization failed with "
            "status " +
            std::to_string(
                static_cast<int>(factor_status)));
    }

    Result result;
    result.detected_ranks =
        problem.jacobian.J_ranks;
    const Eigen::VectorXd base_primal =
        to_eigen(primal);
    const Eigen::VectorXd base_multipliers =
        to_eigen(multipliers);
    if (problem.config.parameters == 0)
    {
        result.primal = base_primal;
        result.multipliers = base_multipliers;
        result.parameters.resize(0);
        result.elapsed_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start)
                .count();
        return result;
    }

    Eigen::MatrixXd response_primal(
        problem.info.number_of_primal_variables,
        problem.config.parameters);
    Eigen::MatrixXd response_multipliers(
        problem.info.number_of_eq_constraints,
        problem.config.parameters);
    VecRealAllocated column_f(
        problem.info.number_of_primal_variables);
    VecRealAllocated column_g(
        problem.info.number_of_eq_constraints);
    VecRealAllocated column_primal(
        problem.info.number_of_primal_variables);
    VecRealAllocated column_multipliers(
        problem.info.number_of_eq_constraints);

    const auto before_parameter_rhs =
        std::chrono::steady_clock::now();
    for (Index parameter = 0;
         parameter < problem.config.parameters;
         ++parameter)
    {
        for (Index row = 0;
             row <
             problem.info.number_of_primal_variables;
             ++row)
            column_f(row) =
                problem.border_primal(row, parameter);
        for (Index row = 0;
             row <
             problem.info.number_of_eq_constraints;
             ++row)
            column_g(row) =
                problem.border_constraints(row, parameter);
        column_primal = 0.0;
        column_multipliers = 0.0;
        const auto status = solver.solve_rhs(
            problem.info,
            problem.jacobian,
            problem.hessian,
            problem.D_s,
            column_f,
            column_g,
            column_primal,
            column_multipliers);
        if (status != LinsolReturnFlag::SUCCESS)
        {
            throw std::runtime_error(
                "Implicit parameter RHS failed with status " +
                std::to_string(static_cast<int>(status)));
        }
        response_primal.col(parameter) =
            to_eigen(column_primal);
        response_multipliers.col(parameter) =
            to_eigen(column_multipliers);
        if (problem.jacobian.J_ranks !=
            result.detected_ranks)
        {
            throw std::runtime_error(
                "Rank pattern changed between RHS solves");
        }
    }
    const auto after_parameter_rhs =
        std::chrono::steady_clock::now();

    const Eigen::MatrixXd schur =
        problem.border_hessian +
        problem.border_primal.transpose() *
            response_primal +
        problem.border_constraints.transpose() *
            response_multipliers;
    const Eigen::VectorXd schur_rhs =
        -problem.rhs_parameters -
        problem.border_primal.transpose() *
            base_primal -
        problem.border_constraints.transpose() *
            base_multipliers;
    const Eigen::LDLT<Eigen::MatrixXd> ldlt(schur);
    if (ldlt.info() != Eigen::Success ||
        !ldlt.isPositive())
        throw std::runtime_error(
            "Static-parameter Schur factorization failed");
    result.parameters = ldlt.solve(schur_rhs);
    if (ldlt.info() != Eigen::Success)
        throw std::runtime_error(
            "Static-parameter Schur solve failed");
    result.primal =
        base_primal +
        response_primal * result.parameters;
    result.multipliers =
        base_multipliers +
        response_multipliers * result.parameters;
    const auto stop =
        std::chrono::steady_clock::now();
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(
            stop - start)
            .count();
    result.parameter_rhs_ms =
        std::chrono::duration<double, std::milli>(
            after_parameter_rhs -
            before_parameter_rhs)
            .count();
    return result;
}

Result solve_lifted_bordered(
    LiftedProblem &problem,
    AugSystemSolver<OcpType> &solver)
{
    VecRealAllocated primal(
        problem.info.number_of_primal_variables);
    VecRealAllocated multipliers(
        problem.info.number_of_eq_constraints);
    primal = 0.0;
    multipliers = 0.0;
    const auto start =
        std::chrono::steady_clock::now();
    const auto factor_status = solver.solve(
        problem.info,
        problem.jacobian,
        problem.hessian,
        problem.D_x,
        problem.D_s,
        problem.rhs_x,
        problem.rhs_g,
        primal,
        multipliers);
    if (factor_status != LinsolReturnFlag::SUCCESS)
    {
        throw std::runtime_error(
            "Lifted FATROP factorization failed with "
            "status " +
            std::to_string(
                static_cast<int>(factor_status)));
    }

    const Eigen::VectorXd base_primal =
        to_eigen(primal);
    const Eigen::VectorXd base_multipliers =
        to_eigen(multipliers);
    if (problem.source.config.parameters == 0)
    {
        Result result =
            problem.map_result(
                base_primal,
                base_multipliers,
                Eigen::VectorXd(0));
        result.elapsed_ms =
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start)
                .count();
        return result;
    }

    const Index parameters =
        problem.source.config.parameters;
    Eigen::MatrixXd response_primal(
        problem.info.number_of_primal_variables,
        parameters);
    Eigen::MatrixXd response_multipliers(
        problem.info.number_of_eq_constraints,
        parameters);
    VecRealAllocated column_f(
        problem.info.number_of_primal_variables);
    VecRealAllocated column_g(
        problem.info.number_of_eq_constraints);
    VecRealAllocated column_primal(
        problem.info.number_of_primal_variables);
    VecRealAllocated column_multipliers(
        problem.info.number_of_eq_constraints);
    const auto before_parameter_rhs =
        std::chrono::steady_clock::now();
    for (Index parameter = 0;
         parameter < parameters; ++parameter)
    {
        for (Index row = 0;
             row <
             problem.info.number_of_primal_variables;
             ++row)
            column_f(row) =
                problem.border_primal(row, parameter);
        for (Index row = 0;
             row <
             problem.info.number_of_eq_constraints;
             ++row)
            column_g(row) =
                problem.border_constraints(
                    row, parameter);
        column_primal = 0.0;
        column_multipliers = 0.0;
        const auto status = solver.solve_rhs(
            problem.info,
            problem.jacobian,
            problem.hessian,
            problem.D_s,
            column_f,
            column_g,
            column_primal,
            column_multipliers);
        if (status != LinsolReturnFlag::SUCCESS)
        {
            throw std::runtime_error(
                "Lifted parameter RHS failed with status " +
                std::to_string(static_cast<int>(status)));
        }
        response_primal.col(parameter) =
            to_eigen(column_primal);
        response_multipliers.col(parameter) =
            to_eigen(column_multipliers);
    }
    const auto after_parameter_rhs =
        std::chrono::steady_clock::now();

    const Eigen::MatrixXd schur =
        problem.border_hessian +
        problem.border_primal.transpose() *
            response_primal +
        problem.border_constraints.transpose() *
            response_multipliers;
    const Eigen::VectorXd schur_rhs =
        -problem.rhs_parameters -
        problem.border_primal.transpose() *
            base_primal -
        problem.border_constraints.transpose() *
            base_multipliers;
    const Eigen::LDLT<Eigen::MatrixXd> ldlt(schur);
    if (ldlt.info() != Eigen::Success ||
        !ldlt.isPositive())
        throw std::runtime_error(
            "Lifted static-parameter Schur "
            "factorization failed");
    const Eigen::VectorXd parameters_solution =
        ldlt.solve(schur_rhs);
    if (ldlt.info() != Eigen::Success)
        throw std::runtime_error(
            "Lifted static-parameter Schur solve failed");
    const Eigen::VectorXd final_primal =
        base_primal +
        response_primal * parameters_solution;
    const Eigen::VectorXd final_multipliers =
        base_multipliers +
        response_multipliers * parameters_solution;
    Result result =
        problem.map_result(
            final_primal,
            final_multipliers,
            parameters_solution);
    const auto stop =
        std::chrono::steady_clock::now();
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(
            stop - start)
            .count();
    result.parameter_rhs_ms =
        std::chrono::duration<double, std::milli>(
            after_parameter_rhs -
            before_parameter_rhs)
            .count();
    return result;
}

double residual(
    const Problem &problem, const Result &result)
{
    const Eigen::VectorXd stationarity =
        problem.regularized_hessian() * result.primal +
        problem.dense_jacobian.transpose() *
            result.multipliers +
        problem.border_primal * result.parameters +
        to_eigen(problem.rhs_x);
    const Eigen::VectorXd constraints =
        problem.dense_jacobian * result.primal +
        problem.border_constraints * result.parameters +
        to_eigen(problem.rhs_g);
    const Eigen::VectorXd parameter_stationarity =
        problem.border_primal.transpose() *
            result.primal +
        problem.border_constraints.transpose() *
            result.multipliers +
        problem.border_hessian * result.parameters +
        problem.rhs_parameters;
    return std::max(
        {max_abs(stationarity),
         max_abs(constraints),
         max_abs(parameter_stationarity)});
}

double operator_residual(
    const Problem &problem, const Result &result)
{
    VecRealAllocated primal(
        problem.info.number_of_primal_variables);
    VecRealAllocated multipliers(
        problem.info.number_of_eq_constraints);
    VecRealAllocated stationarity(
        problem.info.number_of_primal_variables);
    VecRealAllocated constraints(
        problem.info.number_of_eq_constraints);
    VecRealAllocated scratch_primal(
        problem.info.number_of_primal_variables);
    VecRealAllocated scratch_constraints(
        problem.info.number_of_eq_constraints);
    for (Index row = 0;
         row < problem.info.number_of_primal_variables;
         ++row)
        primal(row) = result.primal(row);
    for (Index row = 0;
         row < problem.info.number_of_eq_constraints;
         ++row)
        multipliers(row) = result.multipliers(row);

    stationarity = 0.0;
    constraints = 0.0;
    scratch_primal = 0.0;
    scratch_constraints = 0.0;
    problem.hessian.apply_on_right(
        problem.info, primal, 0.0,
        scratch_primal, stationarity);
    stationarity =
        stationarity + problem.D_x * primal +
        problem.rhs_x;
    problem.jacobian.transpose_apply_on_right(
        problem.info, multipliers, 1.0,
        stationarity, stationarity);
    problem.jacobian.apply_on_right(
        problem.info, primal, 0.0,
        scratch_constraints, constraints);
    constraints = constraints + problem.rhs_g;
    for (Index row = 0;
         row < problem.info.number_of_primal_variables;
         ++row)
    {
        stationarity(row) +=
            problem.border_primal.row(row).dot(
                result.parameters);
    }
    for (Index row = 0;
         row < problem.info.number_of_eq_constraints;
         ++row)
    {
        constraints(row) +=
            problem.border_constraints.row(row).dot(
                result.parameters);
    }
    return std::max(
        max_abs(to_eigen(stationarity)),
        max_abs(to_eigen(constraints)));
}

double dense_difference(
    const Problem &problem, const Result &result)
{
    const Index primal =
        problem.info.number_of_primal_variables;
    const Index constraints =
        problem.info.number_of_eq_constraints;
    const Index parameters =
        problem.config.parameters;
    const Index dimension =
        primal + constraints + parameters;
    if (!problem.config.dense_validation ||
        dimension > 2500)
        return std::numeric_limits<double>::quiet_NaN();

    Eigen::VectorXd rhs(dimension);
    rhs.head(primal) = to_eigen(problem.rhs_x);
    rhs.segment(primal, constraints) =
        to_eigen(problem.rhs_g);
    rhs.tail(parameters) =
        problem.rhs_parameters;
    const Eigen::FullPivLU<Eigen::MatrixXd> lu(
        problem.full_kkt());
    if (!lu.isInvertible())
        throw std::runtime_error(
            "Dense full KKT reference is singular");
    const Eigen::VectorXd reference =
        lu.solve(-rhs);
    Eigen::VectorXd candidate(dimension);
    candidate.head(primal) = result.primal;
    candidate.segment(primal, constraints) =
        result.multipliers;
    candidate.tail(parameters) =
        result.parameters;
    return max_abs(candidate - reference);
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const Config config =
            parse_arguments(argc, argv);
        Problem problem(config);
        AugSystemSolver<ImplicitOcpType> solver(
            problem.info);
        solver.set_lu_fact_tol(config.rank_tolerance);
        solver.set_performance_mode(true);
        LiftedProblem lifted_problem(problem);
        AugSystemSolver<OcpType> lifted_solver(
            lifted_problem.info);
        lifted_solver.set_lu_fact_tol(
            config.rank_tolerance);

        (void)solve_bordered(problem, solver);
        (void)solve_lifted_bordered(
            lifted_problem, lifted_solver);
        std::vector<double> rank_aware_times;
        std::vector<double> rank_aware_rhs_times;
        std::vector<double> lifted_times;
        std::vector<double> lifted_rhs_times;
        Result rank_aware_result;
        Result lifted_result;
        for (Index repeat = 0;
             repeat < config.repeats; ++repeat)
        {
            rank_aware_result =
                solve_bordered(problem, solver);
            lifted_result =
                solve_lifted_bordered(
                    lifted_problem, lifted_solver);
            rank_aware_times.push_back(
                rank_aware_result.elapsed_ms);
            rank_aware_rhs_times.push_back(
                rank_aware_result.parameter_rhs_ms);
            lifted_times.push_back(
                lifted_result.elapsed_ms);
            lifted_rhs_times.push_back(
                lifted_result.parameter_rhs_ms);
        }

        if (rank_aware_result.detected_ranks !=
            problem.expected_ranks)
        {
            throw std::runtime_error(
                "Detected transition ranks do not match "
                "the generator");
        }

        const double rank_aware_residual =
            residual(problem, rank_aware_result);
        const double rank_aware_operator_residual =
            operator_residual(
                problem, rank_aware_result);
        const double rank_aware_reference_difference =
            dense_difference(
                problem, rank_aware_result);
        const double lifted_residual =
            residual(problem, lifted_result);
        const double lifted_operator_residual =
            operator_residual(problem, lifted_result);
        const double lifted_reference_difference =
            dense_difference(problem, lifted_result);
        const double solution_difference =
            std::max(
                {max_abs(
                     rank_aware_result.primal -
                     lifted_result.primal),
                 max_abs(
                     rank_aware_result.multipliers -
                     lifted_result.multipliers),
                 max_abs(
                     rank_aware_result.parameters -
                     lifted_result.parameters)});
        const double rank_aware_scale =
            result_scale(rank_aware_result);
        const double lifted_scale =
            result_scale(lifted_result);
        const double rank_aware_normalized_reference_difference =
            rank_aware_reference_difference /
            (1.0 + rank_aware_scale);
        const double lifted_normalized_reference_difference =
            lifted_reference_difference /
            (1.0 + lifted_scale);
        const double normalized_solution_difference =
            solution_difference /
            (1.0 + std::max(rank_aware_scale, lifted_scale));
        const Inertia base_inertia =
            inertia(
                problem.base_kkt(),
                10.0 * config.rank_tolerance);
        const Inertia full_inertia =
            inertia(
                problem.full_kkt(),
                10.0 * config.rank_tolerance);

        std::cout << std::setprecision(12);
        std::cout
            << "phases,stages,phase_nodes,phase_nx,phase_nu,"
               "rank_deficiency,expected_ranks,detected_ranks,"
               "parameters,primal,constraints,jacobian_rank,"
               "cross_hessian_scale,negative_curvature,"
               "reduced_min_before,regularization,"
               "reduced_min_after,base_inertia_positive,"
               "base_inertia_negative,base_inertia_zero,"
               "full_inertia_positive,full_inertia_negative,"
               "full_inertia_zero,lifted_primal,"
               "lifted_constraints,rank_aware_ms,"
               "rank_aware_parameter_rhs_ms,lifted_ms,"
               "lifted_parameter_rhs_ms,"
               "lifted_over_rank_aware,"
               "rank_aware_kkt_residual,"
               "rank_aware_operator_residual,"
               "rank_aware_dense_difference,"
               "lifted_kkt_residual,"
               "lifted_operator_residual,"
               "lifted_dense_difference,"
               "rank_aware_lifted_difference,"
               "rank_aware_normalized_dense_difference,"
               "lifted_normalized_dense_difference,"
               "rank_aware_lifted_normalized_difference\n";
        std::cout
            << config.phase_nodes.size()
            << ',' << problem.layout.stages
            << ',' << encode_list(config.phase_nodes)
            << ',' << encode_list(config.phase_nx)
            << ',' << encode_list(config.phase_nu)
            << ',' << config.rank_deficiency
            << ',' << encode_list(problem.expected_ranks)
            << ','
            << encode_list(
                   rank_aware_result.detected_ranks)
            << ',' << config.parameters
            << ','
            << problem.info.number_of_primal_variables
            << ','
            << problem.info.number_of_eq_constraints
            << ',' << problem.jacobian_rank
            << ',' << config.cross_hessian_scale
            << ',' << config.negative_curvature
            << ',' << problem.reduced_min_before
            << ',' << problem.regularization
            << ',' << problem.reduced_min_after
            << ',' << base_inertia.positive
            << ',' << base_inertia.negative
            << ',' << base_inertia.zero
            << ',' << full_inertia.positive
            << ',' << full_inertia.negative
            << ',' << full_inertia.zero
            << ','
            << lifted_problem.info
                   .number_of_primal_variables
            << ','
            << lifted_problem.info
                   .number_of_eq_constraints
            << ',' << median(rank_aware_times)
            << ',' << median(rank_aware_rhs_times)
            << ',' << median(lifted_times)
            << ',' << median(lifted_rhs_times)
            << ','
            << median(lifted_times) /
                   median(rank_aware_times)
            << ',' << rank_aware_residual
            << ',' << rank_aware_operator_residual
            << ',' << rank_aware_reference_difference
            << ',' << lifted_residual
            << ',' << lifted_operator_residual
            << ',' << lifted_reference_difference
            << ',' << solution_difference
            << ',' << rank_aware_normalized_reference_difference
            << ',' << lifted_normalized_reference_difference
            << ',' << normalized_solution_difference
            << '\n';

        constexpr double tolerance = 5e-7;
        const Index expected_base_positive =
            problem.info.number_of_primal_variables;
        const Index expected_negative =
            problem.info.number_of_eq_constraints;
        const Index expected_full_positive =
            expected_base_positive + config.parameters;
        if (rank_aware_residual > tolerance ||
            rank_aware_operator_residual > tolerance ||
            (std::isfinite(
                 rank_aware_normalized_reference_difference) &&
             rank_aware_normalized_reference_difference > tolerance) ||
            normalized_solution_difference > tolerance ||
            lifted_residual > tolerance ||
            lifted_operator_residual > tolerance ||
            (std::isfinite(lifted_normalized_reference_difference) &&
             lifted_normalized_reference_difference > tolerance) ||
            problem.reduced_min_after <
                0.5 * config.inertia_margin ||
            base_inertia.positive !=
                expected_base_positive ||
            base_inertia.negative != expected_negative ||
            base_inertia.zero != 0 ||
            full_inertia.positive !=
                expected_full_positive ||
            full_inertia.negative != expected_negative ||
            full_inertia.zero != 0)
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
            << "rank_deficient_implicit_kkt_benchmark: "
            << error.what() << '\n';
        return 1;
    }
}
