#include "fatrop/linear_algebra/linear_algebra.hpp"
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
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
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
    std::string problem = "synthetic";
    Index stages = 20;
    Index nx = 6;
    Index nu = 3;
    Index parameters = 2;
    Index repeats = 5;
    Index collocation_degree = 0;
    double cross_hessian_scale = 0.0;
    bool identity_next_jacobian = false;
    bool dense_validation = true;
    bool run_ipopt = false;
};

std::vector<Index> constant_vector(Index size, Index value)
{
    return std::vector<Index>(static_cast<std::size_t>(size), value);
}

std::vector<Index> controls_vector(const Config &config)
{
    const Index local_controls =
        config.nu + config.collocation_degree * config.nx;
    auto controls = constant_vector(config.stages, local_controls);
    controls.back() = 0;
    return controls;
}

std::vector<Index> equalities_vector(const Config &config)
{
    if (config.problem == "dtoc3")
    {
        auto equalities = constant_vector(config.stages, 0);
        equalities.front() = 2;
        return equalities;
    }
    auto equalities = constant_vector(
        config.stages, config.collocation_degree * config.nx);
    equalities.back() = 0;
    return equalities;
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
            throw std::runtime_error(
                "Radau collocation degree must be 1, 2, or 3");

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
                std::vector<double> product(polynomial.size() + 1, 0.0);
                for (std::size_t power = 0; power < polynomial.size();
                     ++power)
                {
                    product[power] -=
                        polynomial[power] *
                        tau[static_cast<std::size_t>(other)] / denominator;
                    product[power + 1] += polynomial[power] / denominator;
                }
                polynomial = std::move(product);
            }

            for (Index point = 0; point <= degree; ++point)
            {
                const double evaluation =
                    tau[static_cast<std::size_t>(point)];
                double derivative = 0.0;
                for (std::size_t power = 1; power < polynomial.size();
                     ++power)
                {
                    derivative += static_cast<double>(power) *
                                  polynomial[power] *
                                  std::pow(evaluation,
                                           static_cast<int>(power - 1));
                }
                C(basis, point) = derivative;
            }

            double endpoint = 0.0;
            for (double coefficient : polynomial)
                endpoint += coefficient;
            D(basis) = endpoint;
        }
    }

    std::vector<double> tau;
    Eigen::MatrixXd C;
    Eigen::VectorXd D;
};

double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 0)
        return 0.5 * (values[middle - 1] + values[middle]);
    return values[middle];
}

double max_abs(const Eigen::VectorXd &vector)
{
    if (vector.size() == 0)
        return 0.0;
    return vector.cwiseAbs().maxCoeff();
}

double deterministic_value(Index i, Index j, double scale)
{
    return scale * std::sin(0.17 * static_cast<double>(i + 1) +
                            0.31 * static_cast<double>(j + 1));
}

struct PhysicalProblem
{
    explicit PhysicalProblem(const Config &config)
        : config(config),
          controls(controls_vector(config)),
          states(constant_vector(config.stages, config.nx)),
          equalities(equalities_vector(config)),
          inequalities(constant_vector(config.stages, 0)),
          dims(config.stages, controls, states, equalities, inequalities),
          info(dims),
          jacobian(dims),
          hessian(dims),
          D_x(info.number_of_primal_variables),
          D_s(info.number_of_slack_variables),
          rhs_x(info.number_of_primal_variables),
          rhs_g(info.number_of_eq_constraints),
          border_primal(info.number_of_primal_variables, config.parameters),
          border_constraints(info.number_of_eq_constraints, config.parameters),
          border_hessian(config.parameters, config.parameters),
          rhs_parameters(config.parameters)
    {
        D_x = 0.0;
        D_s = 0.0;
        rhs_x = 0.0;
        rhs_g = 0.0;
        border_primal.setZero();
        border_constraints.setZero();
        border_hessian.setZero();
        rhs_parameters.setZero();

        if (config.problem == "dtoc3")
        {
            const double objective_scale =
                1.0 /
                (2.0 * static_cast<double>(config.stages));
            for (Index k = 0; k < config.stages; ++k)
            {
                const Index local_nu = controls[k];
                const Index local_dim = local_nu + config.nx;
                hessian.RSQrqt[k] = 0.0;
                if (k < config.stages - 1)
                    hessian.RSQrqt[k](0, 0) =
                        12.0 * objective_scale;
                if (k > 0)
                {
                    hessian.RSQrqt[k](local_nu, local_nu) =
                        4.0 * objective_scale;
                    hessian.RSQrqt[k](local_nu + 1, local_nu + 1) =
                        2.0 * objective_scale;
                }
                for (Index i = 0; i < local_dim; ++i)
                {
                    const Index global_i =
                        info.offsets_primal_u[k] + i;
                    for (Index p = 0; p < config.parameters; ++p)
                    {
                        border_primal(global_i, p) =
                            deterministic_value(global_i, p, 1e-5);
                    }
                }
            }
        }
        else
        {
            for (Index k = 0; k < config.stages; ++k)
            {
                const Index local_nu = controls[k];
                const Index local_dim = local_nu + config.nx;
                hessian.RSQrqt[k] = 0.0;

                for (Index i = 0; i < local_dim; ++i)
                {
                    hessian.RSQrqt[k](i, i) =
                        1.5 +
                        0.03 * static_cast<double>((i + 3 * k) % 11);
                    if (i + 1 < local_dim)
                    {
                        const double coupling =
                            0.015 / (1.0 + static_cast<double>(i));
                        hessian.RSQrqt[k](i, i + 1) = coupling;
                        hessian.RSQrqt[k](i + 1, i) = coupling;
                    }

                    const Index global_i =
                        info.offsets_primal_u[k] + i;
                    rhs_x(global_i) =
                        deterministic_value(global_i, 0, 0.2);
                    for (Index p = 0; p < config.parameters; ++p)
                    {
                        border_primal(global_i, p) =
                            deterministic_value(global_i, p, 0.008);
                    }
                }
            }
        }

        if (config.problem == "dtoc3")
        {
            const double step =
                1.0 / static_cast<double>(config.stages);
            jacobian.Gg_eqt[0] = 0.0;
            for (Index state = 0; state < config.nx; ++state)
            {
                jacobian.Gg_eqt[0](
                    controls[0] + state, state) = 1.0;
                rhs_g(info.offsets_g_eq_path[0] + state) =
                    state == 0 ? -15.0 : -5.0;
            }

            for (Index k = 0; k < config.stages - 1; ++k)
            {
                const Index local_nu = controls[k];
                jacobian.BAbt[k] = 0.0;
                jacobian.Jt[k] = 0.0;
                hessian.FuFx[k] = 0.0;

                jacobian.BAbt[k](local_nu, 0) = 1.0;
                jacobian.BAbt[k](local_nu + 1, 0) = step;
                jacobian.BAbt[k](0, 1) = step;
                jacobian.BAbt[k](local_nu, 1) = -step;
                jacobian.BAbt[k](local_nu + 1, 1) = 1.0;
                jacobian.Jt[k](0, 0) = -1.0;
                jacobian.Jt[k](1, 1) = -1.0;

                for (Index equation = 0; equation < config.nx;
                     ++equation)
                {
                    const Index constraint =
                        info.offsets_g_eq_dyn[k] + equation;
                    for (Index p = 0; p < config.parameters; ++p)
                    {
                        border_constraints(constraint, p) =
                            0.02 * std::cos(
                                       0.09 * static_cast<double>(
                                                  (equation + 1) *
                                                  (p + 2)) +
                                       0.01 * static_cast<double>(k));
                    }
                }
            }
        }
        else if (config.collocation_degree > 0)
        {
            const CollocationCoefficients coefficients(
                config.collocation_degree);
            const double step = 0.04;
            Eigen::MatrixXd mass(config.nx, config.nx);
            Eigen::MatrixXd system(config.nx, config.nx);
            Eigen::MatrixXd input(config.nx, config.nu);
            Eigen::MatrixXd parameter(config.nx, config.parameters);
            mass.setZero();
            system.setZero();
            input.setZero();
            parameter.setZero();
            for (Index row = 0; row < config.nx; ++row)
            {
                for (Index column = 0; column < config.nx; ++column)
                {
                    mass(row, column) =
                        row == column
                            ? 1.0 + 0.025 * static_cast<double>(row % 7)
                            : (std::abs(row - column) == 1 ? 0.018 : 0.0);
                    system(row, column) =
                        row == column
                            ? -0.45 - 0.015 * static_cast<double>(row % 9)
                            : (std::abs(row - column) == 1 ? 0.055 : 0.0);
                }
                for (Index i = 0; i < config.nu; ++i)
                    input(row, i) =
                        0.12 * std::cos(0.13 * static_cast<double>(
                                                   (row + 1) * (i + 2)));
                for (Index p = 0; p < config.parameters; ++p)
                    parameter(row, p) =
                        0.09 * std::sin(0.11 * static_cast<double>(
                                                   (row + 2) * (p + 1)));
            }

            for (Index k = 0; k < config.stages - 1; ++k)
            {
                const Index local_nu = controls[k];
                jacobian.BAbt[k] = 0.0;
                jacobian.Jt[k] = 0.0;
                jacobian.Gg_eqt[k] = 0.0;
                hessian.FuFx[k] = 0.0;

                for (Index state = 0; state < config.nx; ++state)
                {
                    jacobian.Jt[k](state, state) = -1.0;
                    for (Index basis = 0;
                         basis <= config.collocation_degree; ++basis)
                    {
                        const Index variable_row =
                            basis == 0
                                ? local_nu + state
                                : config.nu + (basis - 1) * config.nx + state;
                        jacobian.BAbt[k](variable_row, state) =
                            coefficients.D(basis);
                    }
                }

                for (Index point = 1;
                     point <= config.collocation_degree; ++point)
                {
                    for (Index equation = 0; equation < config.nx;
                         ++equation)
                    {
                        const Index local_constraint =
                            (point - 1) * config.nx + equation;
                        const Index global_constraint =
                            info.offsets_g_eq_path[k] + local_constraint;

                        for (Index control = 0; control < config.nu;
                             ++control)
                        {
                            jacobian.Gg_eqt[k](control, local_constraint) =
                                -step * input(equation, control);
                        }
                        for (Index basis = 0;
                             basis <= config.collocation_degree; ++basis)
                        {
                            for (Index state = 0; state < config.nx;
                                 ++state)
                            {
                                const Index variable_row =
                                    basis == 0
                                        ? local_nu + state
                                        : config.nu +
                                              (basis - 1) * config.nx +
                                              state;
                                double coefficient =
                                    coefficients.C(basis, point) *
                                    mass(equation, state);
                                if (basis == point)
                                    coefficient -=
                                        step * system(equation, state);
                                jacobian.Gg_eqt[k](variable_row,
                                                  local_constraint) =
                                    coefficient;
                            }
                        }
                        for (Index p = 0; p < config.parameters; ++p)
                        {
                            border_constraints(global_constraint, p) =
                                -step * parameter(equation, p);
                        }
                        rhs_g(global_constraint) =
                            -step * deterministic_value(
                                        k * config.nx + equation, point,
                                        0.035);
                    }
                }

                const Index dynamics_offset = info.offsets_g_eq_dyn[k];
                for (Index state = 0; state < config.nx; ++state)
                    rhs_g(dynamics_offset + state) =
                        deterministic_value(dynamics_offset + state, 1,
                                            0.015);
            }
        }
        else
        {
            for (Index k = 0; k < config.stages - 1; ++k)
            {
                const Index local_nu = controls[k];
                jacobian.BAbt[k] = 0.0;
                jacobian.Jt[k] = 0.0;
                hessian.FuFx[k] = 0.0;

                for (Index j = 0; j < config.nx; ++j)
                {
                    for (Index i = 0; i < local_nu; ++i)
                    {
                        jacobian.BAbt[k](i, j) =
                            0.04 * std::cos(
                                       0.11 * static_cast<double>(
                                                  (i + 1) * (j + 2)));
                    }

                    for (Index i = 0; i < config.nx; ++i)
                    {
                        const double diagonal = i == j ? 0.92 : 0.0;
                        const double neighbour =
                            std::abs(i - j) == 1 ? 0.015 : 0.0;
                        jacobian.BAbt[k](local_nu + i, j) =
                            diagonal + neighbour;
                    }
                    for (Index i = 0; i < config.nx; ++i)
                    {
                        const double diagonal =
                            i == j
                                ? (config.identity_next_jacobian
                                       ? -1.0
                                       : -1.08 -
                                             0.002 * static_cast<double>(
                                                         (j + k) % 7))
                                : 0.0;
                        const double neighbour =
                            !config.identity_next_jacobian &&
                                    std::abs(i - j) == 1
                                ? 0.012 *
                                      std::cos(0.07 * static_cast<double>(
                                                          k + i + j + 1))
                                : 0.0;
                        jacobian.Jt[k](i, j) = diagonal + neighbour;
                    }

                    const Index constraint_i =
                        info.offsets_g_eq_dyn[k] + j;
                    rhs_g(constraint_i) =
                        deterministic_value(constraint_i, 1, 0.1);
                    for (Index p = 0; p < config.parameters; ++p)
                    {
                        border_constraints(constraint_i, p) =
                            0.025 *
                            std::cos(0.13 * static_cast<double>(
                                                (j + 1) * (p + 2)) +
                                     0.03 * static_cast<double>(k));
                    }
                }

                for (Index i = 0; i < local_nu + config.nx; ++i)
                {
                    for (Index j = 0; j < config.nx; ++j)
                    {
                        hessian.FuFx[k](i, j) =
                            deterministic_value(
                                i + k * (local_nu + config.nx), j,
                                config.cross_hessian_scale);
                    }
                }
            }
        }

        const double horizon_scale = static_cast<double>(config.stages);
        for (Index p = 0; p < config.parameters; ++p)
        {
            rhs_parameters(p) = deterministic_value(p, config.stages, 0.15);
            for (Index q = 0; q < config.parameters; ++q)
            {
                border_hessian(p, q) =
                    p == q ? 1.0 + 0.12 * horizon_scale + 0.1 * p
                           : 0.01 / (1.0 + std::abs(p - q));
            }
        }
    }

    Config config;
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
};

struct NaiveProblem
{
    explicit NaiveProblem(const PhysicalProblem &physical)
        : config(physical.config),
          controls(physical.controls),
          states(constant_vector(config.stages, config.nx + config.parameters)),
          equalities(physical.equalities),
          inequalities(constant_vector(config.stages, 0)),
          dims(config.stages, controls, states, equalities, inequalities),
          info(dims),
          jacobian(dims),
          hessian(dims),
          D_x(info.number_of_primal_variables),
          D_s(info.number_of_slack_variables),
          rhs_x(info.number_of_primal_variables),
          rhs_g(info.number_of_eq_constraints)
    {
        D_x = 0.0;
        D_s = 0.0;
        rhs_x = 0.0;
        rhs_g = 0.0;

        for (Index k = 0; k < config.stages; ++k)
        {
            const Index local_nu = controls[k];
            const Index physical_dim = local_nu + config.nx;
            const Index augmented_dim = physical_dim + config.parameters;
            hessian.RSQrqt[k] = 0.0;

            for (Index i = 0; i < physical_dim; ++i)
            {
                for (Index j = 0; j < physical_dim; ++j)
                    hessian.RSQrqt[k](i, j) = physical.hessian.RSQrqt[k](i, j);

                const Index physical_global_i = physical.info.offsets_primal_u[k] + i;
                const Index augmented_global_i = info.offsets_primal_u[k] + i;
                rhs_x(augmented_global_i) = physical.rhs_x(physical_global_i);

                for (Index p = 0; p < config.parameters; ++p)
                {
                    const double cross = physical.border_primal(physical_global_i, p);
                    hessian.RSQrqt[k](i, physical_dim + p) = cross;
                    hessian.RSQrqt[k](physical_dim + p, i) = cross;
                }
            }

            for (Index p = 0; p < config.parameters; ++p)
            {
                const Index theta_i = info.offsets_primal_x[k] + config.nx + p;
                rhs_x(theta_i) =
                    physical.rhs_parameters(p) / static_cast<double>(config.stages);
                for (Index q = 0; q < config.parameters; ++q)
                {
                    hessian.RSQrqt[k](physical_dim + p, physical_dim + q) =
                        physical.border_hessian(p, q) /
                        static_cast<double>(config.stages);
                }
            }

            (void)augmented_dim;
        }

        for (Index k = 0; k < config.stages; ++k)
        {
            jacobian.Gg_eqt[k] = 0.0;
            const Index physical_local_dim = controls[k] + config.nx;
            for (Index constraint = 0; constraint < equalities[k];
                 ++constraint)
            {
                const Index physical_constraint =
                    physical.info.offsets_g_eq_path[k] + constraint;
                const Index augmented_constraint =
                    info.offsets_g_eq_path[k] + constraint;
                rhs_g(augmented_constraint) =
                    physical.rhs_g(physical_constraint);
                for (Index variable = 0; variable < physical_local_dim;
                     ++variable)
                {
                    jacobian.Gg_eqt[k](variable, constraint) =
                        physical.jacobian.Gg_eqt[k](variable, constraint);
                }
                for (Index p = 0; p < config.parameters; ++p)
                {
                    jacobian.Gg_eqt[k](physical_local_dim + p,
                                      constraint) =
                        physical.border_constraints(physical_constraint, p);
                }
            }
        }

        for (Index k = 0; k < config.stages - 1; ++k)
        {
            const Index local_nu = controls[k];
            const Index physical_local_dim = local_nu + config.nx;
            jacobian.BAbt[k] = 0.0;
            jacobian.Jt[k] = 0.0;
            hessian.FuFx[k] = 0.0;

            for (Index i = 0; i < physical_local_dim; ++i)
            {
                for (Index j = 0; j < config.nx; ++j)
                {
                    jacobian.BAbt[k](i, j) = physical.jacobian.BAbt[k](i, j);
                    hessian.FuFx[k](i, j) = physical.hessian.FuFx[k](i, j);
                }
            }

            for (Index p = 0; p < config.parameters; ++p)
            {
                const Index theta_row = physical_local_dim + p;
                for (Index j = 0; j < config.nx; ++j)
                {
                    const Index physical_constraint =
                        physical.info.offsets_g_eq_dyn[k] + j;
                    jacobian.BAbt[k](theta_row, j) =
                        physical.border_constraints(physical_constraint, p);
                }
                jacobian.BAbt[k](theta_row, config.nx + p) = 1.0;
            }

            for (Index i = 0; i < config.nx; ++i)
            {
                for (Index j = 0; j < config.nx; ++j)
                    jacobian.Jt[k](i, j) = physical.jacobian.Jt[k](i, j);
            }
            for (Index p = 0; p < config.parameters; ++p)
                jacobian.Jt[k](config.nx + p, config.nx + p) = -1.0;

            for (Index j = 0; j < config.nx; ++j)
            {
                const Index physical_constraint =
                    physical.info.offsets_g_eq_dyn[k] + j;
                const Index augmented_constraint = info.offsets_g_eq_dyn[k] + j;
                rhs_g(augmented_constraint) = physical.rhs_g(physical_constraint);
            }
        }
    }

    Config config;
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
};

struct ExplicitNaiveProblem
{
    explicit ExplicitNaiveProblem(const NaiveProblem &naive)
        : config(naive.config),
          controls(naive.controls),
          states(naive.states),
          equalities(naive.equalities),
          inequalities(naive.inequalities),
          dims(config.stages, controls, states, equalities, inequalities),
          info(dims),
          jacobian(dims),
          hessian(dims),
          D_x(info.number_of_primal_variables),
          D_s(info.number_of_slack_variables),
          rhs_x(info.number_of_primal_variables),
          rhs_g(info.number_of_eq_constraints),
          multiplier_back_transform(
              static_cast<std::size_t>(config.stages - 1))
    {
        D_x = 0.0;
        D_s = 0.0;
        for (Index i = 0; i < info.number_of_primal_variables; ++i)
            rhs_x(i) = naive.rhs_x(i);

        for (Index k = 0; k < config.stages; ++k)
        {
            const Index local_dim = controls[k] + states[k];
            hessian.RSQrqt[k] = 0.0;
            jacobian.Gg_eqt[k] = 0.0;
            for (Index i = 0; i < local_dim; ++i)
            {
                for (Index j = 0; j < local_dim; ++j)
                    hessian.RSQrqt[k](i, j) =
                        naive.hessian.RSQrqt[k](i, j);
            }
            for (Index constraint = 0; constraint < equalities[k];
                 ++constraint)
            {
                const Index global_constraint =
                    info.offsets_g_eq_path[k] + constraint;
                rhs_g(global_constraint) =
                    naive.rhs_g(global_constraint);
                for (Index variable = 0; variable < local_dim; ++variable)
                {
                    jacobian.Gg_eqt[k](variable, constraint) =
                        naive.jacobian.Gg_eqt[k](variable, constraint);
                }
            }
        }

        for (Index k = 0; k < config.stages - 1; ++k)
        {
            const Index local_dim = controls[k] + states[k];
            const Index next_nx = states[k + 1];
            Eigen::MatrixXd jt(next_nx, next_nx);
            Eigen::MatrixXd right_hand_sides(next_nx,
                                              local_dim + 1);
            for (Index i = 0; i < next_nx; ++i)
            {
                for (Index j = 0; j < next_nx; ++j)
                    jt(i, j) = naive.jacobian.Jt[k](i, j);
                for (Index j = 0; j < local_dim; ++j)
                    right_hand_sides(i, j) =
                        naive.jacobian.BAbt[k](j, i);
                right_hand_sides(i, local_dim) =
                    naive.rhs_g(naive.info.offsets_g_eq_dyn[k] + i);
            }

            const Eigen::MatrixXd transformed =
                -jt.transpose().partialPivLu().solve(
                    right_hand_sides);
            jacobian.BAbt[k] = 0.0;
            for (Index variable = 0; variable < local_dim; ++variable)
            {
                for (Index equation = 0; equation < next_nx; ++equation)
                    jacobian.BAbt[k](variable, equation) =
                        transformed(equation, variable);
            }
            for (Index equation = 0; equation < next_nx; ++equation)
            {
                rhs_g(info.offsets_g_eq_dyn[k] + equation) =
                    transformed(equation, local_dim);
            }
            multiplier_back_transform[static_cast<std::size_t>(k)] =
                -jt.partialPivLu().solve(
                    Eigen::MatrixXd::Identity(next_nx, next_nx));
        }
    }

    Eigen::VectorXd original_multipliers(
        const Eigen::VectorXd &normalized_multipliers) const
    {
        Eigen::VectorXd result = normalized_multipliers;
        for (Index k = 0; k < config.stages - 1; ++k)
        {
            const Index next_nx = states[k + 1];
            result.segment(info.offsets_g_eq_dyn[k], next_nx) =
                multiplier_back_transform[static_cast<std::size_t>(k)] *
                normalized_multipliers.segment(
                    info.offsets_g_eq_dyn[k], next_nx);
        }
        return result;
    }

    Config config;
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
    std::vector<Eigen::MatrixXd> multiplier_back_transform;
};

struct NormalizedProblem
{
    explicit NormalizedProblem(const PhysicalProblem &physical)
        : config(physical.config),
          controls(physical.controls),
          states(physical.states),
          equalities(physical.equalities),
          inequalities(physical.inequalities),
          dims(config.stages, controls, states, equalities, inequalities),
          info(dims),
          jacobian(dims),
          hessian(dims),
          D_x(info.number_of_primal_variables),
          D_s(info.number_of_slack_variables),
          rhs_x(info.number_of_primal_variables),
          rhs_g(info.number_of_eq_constraints),
          border_primal(info.number_of_primal_variables, config.parameters),
          border_constraints(info.number_of_eq_constraints, config.parameters),
          border_hessian(config.parameters, config.parameters),
          rhs_parameters(config.parameters),
          multiplier_back_transform(
              static_cast<std::size_t>(config.stages - 1))
    {
        D_x = 0.0;
        D_s = 0.0;
        refresh(physical);
    }

    void refresh(const PhysicalProblem &physical)
    {
        if (physical.config.cross_hessian_scale != 0.0)
            throw std::runtime_error(
                "Full-rank normalization benchmark currently requires FuFx=0");

        for (Index i = 0; i < info.number_of_primal_variables; ++i)
            rhs_x(i) = physical.rhs_x(i);
        border_primal = physical.border_primal;
        border_hessian = physical.border_hessian;
        rhs_parameters = physical.rhs_parameters;
        border_constraints.setZero();

        for (Index k = 0; k < config.stages; ++k)
        {
            const Index local_dim = controls[k] + config.nx;
            hessian.RSQrqt[k] = 0.0;
            jacobian.Gg_eqt[k] = 0.0;
            for (Index i = 0; i < local_dim; ++i)
            {
                for (Index j = 0; j < local_dim; ++j)
                    hessian.RSQrqt[k](i, j) =
                        physical.hessian.RSQrqt[k](i, j);
            }
            for (Index constraint = 0; constraint < equalities[k];
                 ++constraint)
            {
                const Index global_constraint =
                    info.offsets_g_eq_path[k] + constraint;
                rhs_g(global_constraint) =
                    physical.rhs_g(global_constraint);
                for (Index variable = 0; variable < local_dim; ++variable)
                {
                    jacobian.Gg_eqt[k](variable, constraint) =
                        physical.jacobian.Gg_eqt[k](variable, constraint);
                }
                for (Index p = 0; p < config.parameters; ++p)
                {
                    border_constraints(global_constraint, p) =
                        physical.border_constraints(global_constraint, p);
                }
            }
        }

        for (Index k = 0; k < config.stages - 1; ++k)
        {
            const Index local_dim = controls[k] + config.nx;
            Eigen::MatrixXd jt(config.nx, config.nx);
            Eigen::MatrixXd right_hand_sides(
                config.nx, local_dim + 1 + config.parameters);
            for (Index i = 0; i < config.nx; ++i)
            {
                for (Index j = 0; j < config.nx; ++j)
                    jt(i, j) = physical.jacobian.Jt[k](i, j);
                for (Index j = 0; j < local_dim; ++j)
                    right_hand_sides(i, j) =
                        physical.jacobian.BAbt[k](j, i);
                right_hand_sides(i, local_dim) =
                    physical.rhs_g(physical.info.offsets_g_eq_dyn[k] + i);
                for (Index p = 0; p < config.parameters; ++p)
                {
                    right_hand_sides(i, local_dim + 1 + p) =
                        physical.border_constraints(
                            physical.info.offsets_g_eq_dyn[k] + i, p);
                }
            }

            const Eigen::PartialPivLU<Eigen::MatrixXd> lu(jt.transpose());
            const Eigen::MatrixXd transformed = -lu.solve(right_hand_sides);
            jacobian.BAbt[k] = 0.0;
            for (Index i = 0; i < local_dim; ++i)
            {
                for (Index j = 0; j < config.nx; ++j)
                    jacobian.BAbt[k](i, j) = transformed(j, i);
            }

            for (Index i = 0; i < config.nx; ++i)
            {
                rhs_g(info.offsets_g_eq_dyn[k] + i) =
                    transformed(i, local_dim);
                for (Index p = 0; p < config.parameters; ++p)
                {
                    border_constraints(info.offsets_g_eq_dyn[k] + i, p) =
                        transformed(i, local_dim + 1 + p);
                }
            }

            multiplier_back_transform[static_cast<std::size_t>(k)] =
                -jt.partialPivLu().solve(
                    Eigen::MatrixXd::Identity(config.nx, config.nx));
        }
    }

    Eigen::VectorXd original_multipliers(
        const Eigen::VectorXd &normalized_multipliers) const
    {
        Eigen::VectorXd result = normalized_multipliers;
        for (Index k = 0; k < config.stages - 1; ++k)
        {
            result.segment(info.offsets_g_eq_dyn[k], config.nx) =
                multiplier_back_transform[static_cast<std::size_t>(k)] *
                normalized_multipliers.segment(info.offsets_g_eq_dyn[k],
                                               config.nx);
        }
        return result;
    }

    Config config;
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
    std::vector<Eigen::MatrixXd> multiplier_back_transform;
};

struct BorderedResult
{
    Eigen::VectorXd primal;
    Eigen::VectorXd multipliers;
    Eigen::VectorXd parameters;
    double elapsed_ms = 0.0;
    double normalization_ms = 0.0;
    double factor_ms = 0.0;
    double parameter_rhs_ms = 0.0;
    double schur_ms = 0.0;
};

struct NaiveResult
{
    Eigen::VectorXd primal;
    Eigen::VectorXd multipliers;
    Eigen::VectorXd parameters;
    double parameter_consensus = 0.0;
    double elapsed_ms = 0.0;
};

struct IpoptResult
{
    Eigen::VectorXd primal;
    Eigen::VectorXd multipliers;
    Eigen::VectorXd parameters;
    double elapsed_ms = std::numeric_limits<double>::quiet_NaN();
    int status = -1;
    int iterations = -1;
};

Eigen::VectorXd to_eigen(const VecRealAllocated &vector);

class GlobalParameterTnlp final : public Ipopt::TNLP
{
public:
    explicit GlobalParameterTnlp(const PhysicalProblem &problem)
        : problem_(problem),
          solution_(problem.info.number_of_primal_variables +
                    problem.config.parameters),
          multipliers_(problem.info.number_of_eq_constraints)
    {
        solution_.setZero();
        multipliers_.setZero();
    }

    bool get_nlp_info(Ipopt::Index &n, Ipopt::Index &m,
                      Ipopt::Index &nnz_jac_g, Ipopt::Index &nnz_h_lag,
                      IndexStyleEnum &index_style) override
    {
        n = static_cast<Ipopt::Index>(
            problem_.info.number_of_primal_variables +
            problem_.config.parameters);
        m = static_cast<Ipopt::Index>(problem_.info.number_of_eq_constraints);
        nnz_jac_g = count_jacobian_nonzeros();
        nnz_h_lag = count_hessian_nonzeros();
        index_style = TNLP::C_STYLE;
        return true;
    }

    bool get_bounds_info(Ipopt::Index n, Ipopt::Number *x_l,
                         Ipopt::Number *x_u, Ipopt::Index m,
                         Ipopt::Number *g_l,
                         Ipopt::Number *g_u) override
    {
        for (Ipopt::Index i = 0; i < n; ++i)
        {
            x_l[i] = -1e19;
            x_u[i] = 1e19;
        }
        for (Ipopt::Index i = 0; i < m; ++i)
        {
            g_l[i] = 0.0;
            g_u[i] = 0.0;
        }
        return true;
    }

    bool get_starting_point(Ipopt::Index n, bool init_x,
                            Ipopt::Number *x, bool init_z,
                            Ipopt::Number *, Ipopt::Number *,
                            Ipopt::Index, bool init_lambda,
                            Ipopt::Number *) override
    {
        if (!init_x || init_z || init_lambda)
            return false;
        std::fill(x, x + n, 0.0);
        return true;
    }

    bool eval_f(Ipopt::Index, const Ipopt::Number *x, bool,
                Ipopt::Number &obj_value) override
    {
        const Index npr = problem_.info.number_of_primal_variables;
        Eigen::Map<const Eigen::VectorXd> primal(x, npr);
        Eigen::Map<const Eigen::VectorXd> parameters(
            x + npr, problem_.config.parameters);
        obj_value = 0.0;
        for (Index k = 0; k < problem_.config.stages; ++k)
        {
            const Index local_dim =
                problem_.controls[k] + problem_.config.nx;
            const Index offset = problem_.info.offsets_primal_u[k];
            for (Index i = 0; i < local_dim; ++i)
            {
                obj_value += problem_.rhs_x(offset + i) * primal(offset + i);
                for (Index j = 0; j < local_dim; ++j)
                {
                    obj_value +=
                        0.5 * primal(offset + i) *
                        problem_.hessian.RSQrqt[k](i, j) *
                        primal(offset + j);
                }
            }
        }
        obj_value += primal.dot(problem_.border_primal * parameters);
        obj_value +=
            0.5 * parameters.dot(problem_.border_hessian * parameters);
        obj_value += parameters.dot(problem_.rhs_parameters);
        return true;
    }

    bool eval_grad_f(Ipopt::Index, const Ipopt::Number *x, bool,
                     Ipopt::Number *grad_f) override
    {
        const Index npr = problem_.info.number_of_primal_variables;
        Eigen::Map<const Eigen::VectorXd> primal(x, npr);
        Eigen::Map<const Eigen::VectorXd> parameters(
            x + npr, problem_.config.parameters);
        Eigen::Map<Eigen::VectorXd> gradient_primal(grad_f, npr);
        Eigen::Map<Eigen::VectorXd> gradient_parameters(
            grad_f + npr, problem_.config.parameters);
        gradient_primal = to_eigen(problem_.rhs_x) +
                          problem_.border_primal * parameters;
        for (Index k = 0; k < problem_.config.stages; ++k)
        {
            const Index local_dim =
                problem_.controls[k] + problem_.config.nx;
            const Index offset = problem_.info.offsets_primal_u[k];
            for (Index i = 0; i < local_dim; ++i)
            {
                for (Index j = 0; j < local_dim; ++j)
                {
                    gradient_primal(offset + i) +=
                        problem_.hessian.RSQrqt[k](i, j) *
                        primal(offset + j);
                }
            }
        }
        gradient_parameters =
            problem_.rhs_parameters +
            problem_.border_primal.transpose() * primal +
            problem_.border_hessian * parameters;
        return true;
    }

    bool eval_g(Ipopt::Index, const Ipopt::Number *x, bool,
                Ipopt::Index, Ipopt::Number *g) override
    {
        const Index npr = problem_.info.number_of_primal_variables;
        Eigen::Map<const Eigen::VectorXd> primal(x, npr);
        Eigen::Map<const Eigen::VectorXd> parameters(
            x + npr, problem_.config.parameters);
        for (Index k = 0; k < problem_.config.stages; ++k)
        {
            const Index local_dim =
                problem_.controls[k] + problem_.config.nx;
            const Index local_offset =
                problem_.info.offsets_primal_u[k];
            const Index constraint_offset =
                problem_.info.offsets_g_eq_path[k];
            for (Index row = 0; row < problem_.equalities[k]; ++row)
            {
                double value =
                    problem_.rhs_g(constraint_offset + row);
                for (Index i = 0; i < local_dim; ++i)
                    value += problem_.jacobian.Gg_eqt[k](i, row) *
                             primal(local_offset + i);
                for (Index p = 0; p < problem_.config.parameters; ++p)
                    value +=
                        problem_.border_constraints(constraint_offset + row, p) *
                        parameters(p);
                g[constraint_offset + row] = value;
            }
        }
        for (Index k = 0; k < problem_.config.stages - 1; ++k)
        {
            const Index local_dim =
                problem_.controls[k] + problem_.config.nx;
            const Index local_offset = problem_.info.offsets_primal_u[k];
            const Index next_offset = problem_.info.offsets_primal_x[k + 1];
            const Index constraint_offset =
                problem_.info.offsets_g_eq_dyn[k];
            for (Index row = 0; row < problem_.config.nx; ++row)
            {
                double value = problem_.rhs_g(constraint_offset + row);
                for (Index i = 0; i < local_dim; ++i)
                    value += problem_.jacobian.BAbt[k](i, row) *
                             primal(local_offset + i);
                for (Index i = 0; i < problem_.config.nx; ++i)
                    value += problem_.jacobian.Jt[k](i, row) *
                             primal(next_offset + i);
                for (Index p = 0; p < problem_.config.parameters; ++p)
                    value +=
                        problem_.border_constraints(constraint_offset + row, p) *
                        parameters(p);
                g[constraint_offset + row] = value;
            }
        }
        return true;
    }

    bool eval_jac_g(Ipopt::Index, const Ipopt::Number *, bool,
                    Ipopt::Index, Ipopt::Index,
                    Ipopt::Index *i_row, Ipopt::Index *j_col,
                    Ipopt::Number *values) override
    {
        Ipopt::Index cursor = 0;
        const Index parameter_offset =
            problem_.info.number_of_primal_variables;
        for (Index k = 0; k < problem_.config.stages; ++k)
        {
            const Index local_dim =
                problem_.controls[k] + problem_.config.nx;
            const Index local_offset =
                problem_.info.offsets_primal_u[k];
            const Index constraint_offset =
                problem_.info.offsets_g_eq_path[k];
            for (Index row = 0; row < problem_.equalities[k]; ++row)
            {
                for (Index i = 0; i < local_dim; ++i)
                {
                    const double coefficient =
                        problem_.jacobian.Gg_eqt[k](i, row);
                    append_sparse_entry(
                        constraint_offset + row, local_offset + i,
                        coefficient, coefficient, cursor, i_row, j_col,
                        values);
                }
                for (Index p = 0; p < problem_.config.parameters; ++p)
                {
                    const double coefficient =
                        problem_.border_constraints(
                            constraint_offset + row, p);
                    append_sparse_entry(
                        constraint_offset + row, parameter_offset + p,
                        coefficient, coefficient, cursor, i_row, j_col,
                        values);
                }
            }
        }
        for (Index k = 0; k < problem_.config.stages - 1; ++k)
        {
            const Index local_dim =
                problem_.controls[k] + problem_.config.nx;
            const Index local_offset = problem_.info.offsets_primal_u[k];
            const Index next_offset = problem_.info.offsets_primal_x[k + 1];
            const Index constraint_offset =
                problem_.info.offsets_g_eq_dyn[k];
            for (Index row = 0; row < problem_.config.nx; ++row)
            {
                for (Index i = 0; i < local_dim; ++i)
                    append_sparse_entry(constraint_offset + row,
                                        local_offset + i,
                                        problem_.jacobian.BAbt[k](i, row),
                                        problem_.jacobian.BAbt[k](i, row),
                                        cursor, i_row, j_col, values);
                for (Index i = 0; i < problem_.config.nx; ++i)
                    append_sparse_entry(constraint_offset + row,
                                        next_offset + i,
                                        problem_.jacobian.Jt[k](i, row),
                                        problem_.jacobian.Jt[k](i, row),
                                        cursor, i_row, j_col, values);
                for (Index p = 0; p < problem_.config.parameters; ++p)
                    append_sparse_entry(
                        constraint_offset + row, parameter_offset + p,
                        problem_.border_constraints(constraint_offset + row, p),
                        problem_.border_constraints(constraint_offset + row, p),
                        cursor, i_row, j_col, values);
            }
        }
        return true;
    }

    bool eval_h(Ipopt::Index, const Ipopt::Number *, bool,
                Ipopt::Number obj_factor, Ipopt::Index,
                const Ipopt::Number *, bool, Ipopt::Index,
                Ipopt::Index *i_row, Ipopt::Index *j_col,
                Ipopt::Number *values) override
    {
        Ipopt::Index cursor = 0;
        const Index parameter_offset =
            problem_.info.number_of_primal_variables;
        for (Index k = 0; k < problem_.config.stages; ++k)
        {
            const Index local_dim =
                problem_.controls[k] + problem_.config.nx;
            const Index offset = problem_.info.offsets_primal_u[k];
            for (Index i = 0; i < local_dim; ++i)
            {
                for (Index j = 0; j <= i; ++j)
                {
                    const double coefficient =
                        problem_.hessian.RSQrqt[k](i, j);
                    append_sparse_entry(
                        offset + i, offset + j, coefficient,
                        obj_factor * coefficient,
                        cursor, i_row, j_col, values);
                }
            }
        }
        for (Index p = 0; p < problem_.config.parameters; ++p)
        {
            for (Index i = 0;
                 i < problem_.info.number_of_primal_variables; ++i)
            {
                const double coefficient = problem_.border_primal(i, p);
                append_sparse_entry(
                    parameter_offset + p, i, coefficient,
                    obj_factor * coefficient, cursor,
                    i_row, j_col, values);
            }
            for (Index q = 0; q <= p; ++q)
            {
                const double coefficient =
                    problem_.border_hessian(p, q);
                append_sparse_entry(
                    parameter_offset + p, parameter_offset + q, coefficient,
                    obj_factor * coefficient, cursor,
                    i_row, j_col, values);
            }
        }
        return true;
    }

    void finalize_solution(
        Ipopt::SolverReturn, Ipopt::Index n, const Ipopt::Number *x,
        const Ipopt::Number *, const Ipopt::Number *, Ipopt::Index m,
        const Ipopt::Number *, const Ipopt::Number *lambda,
        Ipopt::Number, const Ipopt::IpoptData *ip_data,
        Ipopt::IpoptCalculatedQuantities *) override
    {
        solution_ = Eigen::Map<const Eigen::VectorXd>(x, n);
        multipliers_ = Eigen::Map<const Eigen::VectorXd>(lambda, m);
        if (ip_data != nullptr)
            iterations_ = static_cast<int>(ip_data->iter_count());
    }

    const Eigen::VectorXd &solution() const { return solution_; }
    const Eigen::VectorXd &multipliers() const { return multipliers_; }
    int iterations() const { return iterations_; }

private:
    static bool structurally_nonzero(double value)
    {
        return std::abs(value) > 1e-18;
    }

    static void append_sparse_entry(
        Index row, Index column, double structural_value, double value,
        Ipopt::Index &cursor, Ipopt::Index *i_row,
        Ipopt::Index *j_col, Ipopt::Number *values)
    {
        if (!structurally_nonzero(structural_value))
            return;
        if (values == nullptr)
        {
            i_row[cursor] = static_cast<Ipopt::Index>(row);
            j_col[cursor] = static_cast<Ipopt::Index>(column);
        }
        else
            values[cursor] = value;
        ++cursor;
    }

    Ipopt::Index count_jacobian_nonzeros() const
    {
        Ipopt::Index count = 0;
        for (Index k = 0; k < problem_.config.stages; ++k)
        {
            const Index local_dim =
                problem_.controls[k] + problem_.config.nx;
            const Index constraint_offset =
                problem_.info.offsets_g_eq_path[k];
            for (Index row = 0; row < problem_.equalities[k]; ++row)
            {
                for (Index i = 0; i < local_dim; ++i)
                    count += structurally_nonzero(
                        problem_.jacobian.Gg_eqt[k](i, row));
                for (Index p = 0; p < problem_.config.parameters; ++p)
                    count += structurally_nonzero(
                        problem_.border_constraints(
                            constraint_offset + row, p));
            }
        }
        for (Index k = 0; k < problem_.config.stages - 1; ++k)
        {
            const Index local_dim =
                problem_.controls[k] + problem_.config.nx;
            const Index constraint_offset =
                problem_.info.offsets_g_eq_dyn[k];
            for (Index row = 0; row < problem_.config.nx; ++row)
            {
                for (Index i = 0; i < local_dim; ++i)
                    count += structurally_nonzero(
                        problem_.jacobian.BAbt[k](i, row));
                for (Index i = 0; i < problem_.config.nx; ++i)
                    count += structurally_nonzero(
                        problem_.jacobian.Jt[k](i, row));
                for (Index p = 0; p < problem_.config.parameters; ++p)
                    count += structurally_nonzero(
                        problem_.border_constraints(constraint_offset + row, p));
            }
        }
        return count;
    }

    Ipopt::Index count_hessian_nonzeros() const
    {
        Ipopt::Index count = 0;
        for (Index k = 0; k < problem_.config.stages; ++k)
        {
            const Index local_dim =
                problem_.controls[k] + problem_.config.nx;
            for (Index i = 0; i < local_dim; ++i)
                for (Index j = 0; j <= i; ++j)
                    count += structurally_nonzero(
                        problem_.hessian.RSQrqt[k](i, j));
        }
        for (Index p = 0; p < problem_.config.parameters; ++p)
        {
            for (Index i = 0;
                 i < problem_.info.number_of_primal_variables; ++i)
                count += structurally_nonzero(
                    problem_.border_primal(i, p));
            for (Index q = 0; q <= p; ++q)
                count += structurally_nonzero(
                    problem_.border_hessian(p, q));
        }
        return count;
    }

    const PhysicalProblem &problem_;
    Eigen::VectorXd solution_;
    Eigen::VectorXd multipliers_;
    int iterations_ = -1;
};

Eigen::VectorXd to_eigen(const VecRealAllocated &vector)
{
    Eigen::VectorXd result(vector.m());
    for (Index i = 0; i < vector.m(); ++i)
        result(i) = vector(i);
    return result;
}

IpoptResult solve_ipopt(const PhysicalProblem &problem)
{
    Ipopt::SmartPtr<Ipopt::IpoptApplication> application =
        IpoptApplicationFactory();
    application->Options()->SetIntegerValue("print_level", 0);
    application->Options()->SetStringValue("sb", "yes");
    application->Options()->SetStringValue("linear_solver", "mumps");
    application->Options()->SetStringValue("hessian_approximation", "exact");
    application->Options()->SetStringValue("nlp_scaling_method", "none");
    application->Options()->SetNumericValue("tol", 1e-10);
    application->Options()->SetNumericValue("constr_viol_tol", 1e-10);
    application->Options()->SetNumericValue("dual_inf_tol", 1e-10);
    application->Options()->SetIntegerValue("max_iter", 50);

    const Ipopt::ApplicationReturnStatus initialization =
        application->Initialize();
    if (initialization != Ipopt::Solve_Succeeded)
        throw std::runtime_error("IPOPT initialization failed");

    auto *raw_problem = new GlobalParameterTnlp(problem);
    Ipopt::SmartPtr<Ipopt::TNLP> tnlp = raw_problem;
    const auto start = std::chrono::steady_clock::now();
    const Ipopt::ApplicationReturnStatus status =
        application->OptimizeTNLP(tnlp);
    const auto stop = std::chrono::steady_clock::now();

    IpoptResult result;
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    result.status = static_cast<int>(status);
    result.iterations = raw_problem->iterations();
    const Index npr = problem.info.number_of_primal_variables;
    result.primal = raw_problem->solution().head(npr);
    result.parameters =
        raw_problem->solution().tail(problem.config.parameters);
    result.multipliers = raw_problem->multipliers();
    return result;
}

void assign(VecRealAllocated &target, const Eigen::VectorXd &source)
{
    if (target.m() != source.size())
        throw std::runtime_error("Vector dimension mismatch");
    for (Index i = 0; i < target.m(); ++i)
        target(i) = source(i);
}

BorderedResult solve_normalized_bordered(
    const PhysicalProblem &physical, NormalizedProblem &problem,
    AugSystemSolver<OcpType> &solver)
{
    VecRealAllocated primal(problem.info.number_of_primal_variables);
    VecRealAllocated multipliers(problem.info.number_of_eq_constraints);
    VecRealAllocated column_primal_rhs(
        problem.info.number_of_primal_variables);
    VecRealAllocated column_constraint_rhs(
        problem.info.number_of_eq_constraints);
    VecRealAllocated column_primal(
        problem.info.number_of_primal_variables);
    VecRealAllocated column_multipliers(
        problem.info.number_of_eq_constraints);
    Eigen::VectorXd base_primal(
        problem.info.number_of_primal_variables);
    Eigen::VectorXd base_multipliers(
        problem.info.number_of_eq_constraints);
    Eigen::MatrixXd response_primal(
        problem.info.number_of_primal_variables,
        problem.config.parameters);
    Eigen::MatrixXd response_multipliers(
        problem.info.number_of_eq_constraints,
        problem.config.parameters);
    Eigen::MatrixXd schur(problem.config.parameters,
                          problem.config.parameters);
    Eigen::VectorXd schur_rhs(problem.config.parameters);
    BorderedResult result;
    result.primal.resize(problem.info.number_of_primal_variables);
    result.multipliers.resize(problem.info.number_of_eq_constraints);
    result.parameters.resize(problem.config.parameters);
    primal = 0.0;
    multipliers = 0.0;

    const auto start = std::chrono::steady_clock::now();
    if (problem.config.problem == "synthetic" &&
        problem.config.collocation_degree == 0)
        problem.refresh(physical);
    const auto after_normalization = std::chrono::steady_clock::now();
    const auto factor_status =
        solver.solve(problem.info, problem.jacobian, problem.hessian, problem.D_x,
                     problem.D_s, problem.rhs_x, problem.rhs_g, primal, multipliers);
    if (factor_status != LinsolReturnFlag::SUCCESS)
        throw std::runtime_error("Normalized bordered factorization failed");

    base_primal = to_eigen(primal);
    base_multipliers = to_eigen(multipliers);
    const auto after_factor = std::chrono::steady_clock::now();
    if (problem.config.parameters == 0)
    {
        result.primal = base_primal;
        result.multipliers =
            problem.original_multipliers(base_multipliers);
        const auto stop = std::chrono::steady_clock::now();
        result.elapsed_ms =
            std::chrono::duration<double, std::milli>(stop - start)
                .count();
        result.normalization_ms =
            std::chrono::duration<double, std::milli>(
                after_normalization - start)
                .count();
        result.factor_ms =
            std::chrono::duration<double, std::milli>(
                after_factor - after_normalization)
                .count();
        result.parameter_rhs_ms = 0.0;
        result.schur_ms =
            std::chrono::duration<double, std::milli>(
                stop - after_factor)
                .count();
        return result;
    }
    for (Index p = 0; p < problem.config.parameters; ++p)
    {
        for (Index i = 0; i < problem.info.number_of_primal_variables; ++i)
            column_primal_rhs(i) = problem.border_primal(i, p);
        for (Index i = 0; i < problem.info.number_of_eq_constraints; ++i)
            column_constraint_rhs(i) = problem.border_constraints(i, p);

        column_primal = 0.0;
        column_multipliers = 0.0;
        const auto status =
            solver.solve_rhs(problem.info, problem.jacobian, problem.hessian,
                             problem.D_s, column_primal_rhs,
                             column_constraint_rhs, column_primal,
                             column_multipliers);
        if (status != LinsolReturnFlag::SUCCESS)
            throw std::runtime_error(
                "Normalized bordered right-hand-side solve failed");
        response_primal.col(p) = to_eigen(column_primal);
        response_multipliers.col(p) = to_eigen(column_multipliers);
    }
    const auto after_parameter_rhs = std::chrono::steady_clock::now();

    schur =
        problem.border_hessian +
        problem.border_primal.transpose() * response_primal +
        problem.border_constraints.transpose() * response_multipliers;
    schur_rhs =
        -problem.rhs_parameters -
        problem.border_primal.transpose() * base_primal -
        problem.border_constraints.transpose() * base_multipliers;
    const Eigen::LDLT<Eigen::MatrixXd> ldlt(schur);
    if (ldlt.info() != Eigen::Success)
        throw std::runtime_error(
            "Normalized dense parameter Schur factorization failed");
    result.parameters = ldlt.solve(schur_rhs);
    if (ldlt.info() != Eigen::Success)
        throw std::runtime_error(
            "Normalized dense parameter Schur solve failed");

    result.primal =
        base_primal + response_primal * result.parameters;
    const Eigen::VectorXd normalized_multipliers =
        base_multipliers + response_multipliers * result.parameters;
    result.multipliers =
        problem.original_multipliers(normalized_multipliers);
    const auto stop = std::chrono::steady_clock::now();
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();
    result.normalization_ms =
        std::chrono::duration<double, std::milli>(
            after_normalization - start)
            .count();
    result.factor_ms =
        std::chrono::duration<double, std::milli>(
            after_factor - after_normalization)
            .count();
    result.parameter_rhs_ms =
        std::chrono::duration<double, std::milli>(
            after_parameter_rhs - after_factor)
            .count();
    result.schur_ms =
        std::chrono::duration<double, std::milli>(
            stop - after_parameter_rhs)
            .count();
    return result;
}

NaiveResult solve_naive(NaiveProblem &problem,
                        AugSystemSolver<ImplicitOcpType> &solver)
{
    VecRealAllocated primal(problem.info.number_of_primal_variables);
    VecRealAllocated multipliers(problem.info.number_of_eq_constraints);
    primal = 0.0;
    multipliers = 0.0;

    const auto start = std::chrono::steady_clock::now();
    const auto status =
        solver.solve(problem.info, problem.jacobian, problem.hessian, problem.D_x,
                     problem.D_s, problem.rhs_x, problem.rhs_g, primal, multipliers);
    const auto stop = std::chrono::steady_clock::now();
    if (status != LinsolReturnFlag::SUCCESS)
        throw std::runtime_error("Naive augmented-state factorization failed");

    NaiveResult result;
    result.primal = to_eigen(primal);
    result.multipliers = to_eigen(multipliers);
    result.parameters = Eigen::VectorXd::Zero(problem.config.parameters);
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();

    for (Index p = 0; p < problem.config.parameters; ++p)
    {
        double mean = 0.0;
        for (Index k = 0; k < problem.config.stages; ++k)
            mean += primal(problem.info.offsets_primal_x[k] + problem.config.nx + p);
        mean /= static_cast<double>(problem.config.stages);
        result.parameters(p) = mean;

        for (Index k = 0; k < problem.config.stages; ++k)
        {
            result.parameter_consensus =
                std::max(result.parameter_consensus,
                         std::abs(primal(problem.info.offsets_primal_x[k] +
                                         problem.config.nx + p) -
                                  mean));
        }
    }
    return result;
}

NaiveResult solve_explicit_naive(
    ExplicitNaiveProblem &problem,
    AugSystemSolver<OcpType> &solver)
{
    VecRealAllocated primal(problem.info.number_of_primal_variables);
    VecRealAllocated normalized_multipliers(
        problem.info.number_of_eq_constraints);
    primal = 0.0;
    normalized_multipliers = 0.0;

    const auto start = std::chrono::steady_clock::now();
    const auto status =
        solver.solve(problem.info, problem.jacobian, problem.hessian,
                     problem.D_x, problem.D_s, problem.rhs_x,
                     problem.rhs_g, primal, normalized_multipliers);
    const auto stop = std::chrono::steady_clock::now();
    if (status != LinsolReturnFlag::SUCCESS)
        throw std::runtime_error(
            "Explicit naive augmented-state factorization failed");

    NaiveResult result;
    result.primal = to_eigen(primal);
    result.multipliers = problem.original_multipliers(
        to_eigen(normalized_multipliers));
    result.parameters =
        Eigen::VectorXd::Zero(problem.config.parameters);
    result.elapsed_ms =
        std::chrono::duration<double, std::milli>(stop - start)
            .count();

    for (Index p = 0; p < problem.config.parameters; ++p)
    {
        double mean = 0.0;
        for (Index k = 0; k < problem.config.stages; ++k)
        {
            mean += primal(problem.info.offsets_primal_x[k] +
                           problem.config.nx + p);
        }
        mean /= static_cast<double>(problem.config.stages);
        result.parameters(p) = mean;
        for (Index k = 0; k < problem.config.stages; ++k)
        {
            result.parameter_consensus = std::max(
                result.parameter_consensus,
                std::abs(
                    primal(problem.info.offsets_primal_x[k] +
                           problem.config.nx + p) -
                    mean));
        }
    }
    return result;
}

double implicit_kkt_residual(const ProblemInfo &info,
                             const Jacobian<ImplicitOcpType> &jacobian,
                             const Hessian<ImplicitOcpType> &hessian,
                             const VecRealAllocated &rhs_x,
                             const VecRealAllocated &rhs_g,
                             const Eigen::VectorXd &primal,
                             const Eigen::VectorXd &multipliers)
{
    VecRealAllocated primal_fatrop(info.number_of_primal_variables);
    VecRealAllocated multipliers_fatrop(info.number_of_eq_constraints);
    VecRealAllocated hessian_product(info.number_of_primal_variables);
    VecRealAllocated jacobian_transpose_product(info.number_of_primal_variables);
    VecRealAllocated jacobian_product(info.number_of_eq_constraints);
    assign(primal_fatrop, primal);
    assign(multipliers_fatrop, multipliers);
    hessian_product = 0.0;
    jacobian_transpose_product = 0.0;
    jacobian_product = 0.0;

    hessian.apply_on_right(info, primal_fatrop, 0.0, hessian_product,
                           hessian_product);
    jacobian.transpose_apply_on_right(info, multipliers_fatrop, 0.0,
                                      jacobian_transpose_product,
                                      jacobian_transpose_product);
    jacobian.apply_on_right(info, primal_fatrop, 0.0, jacobian_product,
                            jacobian_product);

    Eigen::VectorXd primal_residual =
        to_eigen(rhs_x) + to_eigen(hessian_product) +
        to_eigen(jacobian_transpose_product);
    Eigen::VectorXd constraint_residual =
        to_eigen(rhs_g) + to_eigen(jacobian_product);

    return std::max(max_abs(primal_residual), max_abs(constraint_residual));
}

double bordered_residual(const PhysicalProblem &problem,
                         const BorderedResult &result)
{
    VecRealAllocated primal_fatrop(problem.info.number_of_primal_variables);
    VecRealAllocated multipliers_fatrop(problem.info.number_of_eq_constraints);
    VecRealAllocated hessian_product(problem.info.number_of_primal_variables);
    VecRealAllocated jacobian_transpose_product(
        problem.info.number_of_primal_variables);
    VecRealAllocated jacobian_product(problem.info.number_of_eq_constraints);
    assign(primal_fatrop, result.primal);
    assign(multipliers_fatrop, result.multipliers);
    hessian_product = 0.0;
    jacobian_transpose_product = 0.0;
    jacobian_product = 0.0;
    problem.hessian.apply_on_right(problem.info, primal_fatrop, 0.0,
                                   hessian_product, hessian_product);
    problem.jacobian.transpose_apply_on_right(
        problem.info, multipliers_fatrop, 0.0, jacobian_transpose_product,
        jacobian_transpose_product);
    problem.jacobian.apply_on_right(problem.info, primal_fatrop, 0.0,
                                    jacobian_product, jacobian_product);

    Eigen::VectorXd primal_residual =
        to_eigen(problem.rhs_x) + to_eigen(hessian_product) +
        to_eigen(jacobian_transpose_product) +
        problem.border_primal * result.parameters;
    Eigen::VectorXd constraint_residual =
        to_eigen(problem.rhs_g) + to_eigen(jacobian_product) +
        problem.border_constraints * result.parameters;

    const Eigen::VectorXd parameter_residual =
        problem.border_primal.transpose() * result.primal +
        problem.border_constraints.transpose() * result.multipliers +
        problem.border_hessian * result.parameters +
        problem.rhs_parameters;

    return std::max({max_abs(primal_residual), max_abs(constraint_residual),
                     max_abs(parameter_residual)});
}

double objective_value(const PhysicalProblem &problem,
                       const BorderedResult &result)
{
    double value = result.primal.dot(to_eigen(problem.rhs_x));
    for (Index k = 0; k < problem.config.stages; ++k)
    {
        const Index local_dim =
            problem.controls[k] + problem.config.nx;
        const Index offset = problem.info.offsets_primal_u[k];
        for (Index i = 0; i < local_dim; ++i)
        {
            for (Index j = 0; j < local_dim; ++j)
            {
                value +=
                    0.5 * result.primal(offset + i) *
                    problem.hessian.RSQrqt[k](i, j) *
                    result.primal(offset + j);
            }
        }
    }
    value +=
        result.primal.dot(problem.border_primal * result.parameters);
    value += 0.5 * result.parameters.dot(
                       problem.border_hessian * result.parameters);
    value += result.parameters.dot(problem.rhs_parameters);
    return value;
}

Eigen::VectorXd extract_naive_physical_primal(const PhysicalProblem &physical,
                                              const NaiveProblem &naive,
                                              const NaiveResult &result)
{
    Eigen::VectorXd extracted(physical.info.number_of_primal_variables);
    for (Index k = 0; k < physical.config.stages; ++k)
    {
        const Index local_dim = physical.controls[k] + physical.config.nx;
        for (Index i = 0; i < local_dim; ++i)
            extracted(physical.info.offsets_primal_u[k] + i) =
                result.primal(naive.info.offsets_primal_u[k] + i);
    }
    return extracted;
}

Eigen::VectorXd extract_naive_physical_multipliers(const PhysicalProblem &physical,
                                                   const NaiveProblem &naive,
                                                   const NaiveResult &result)
{
    Eigen::VectorXd extracted =
        Eigen::VectorXd::Zero(physical.info.number_of_eq_constraints);
    for (Index k = 0; k < physical.config.stages; ++k)
    {
        for (Index i = 0; i < physical.equalities[k]; ++i)
        {
            extracted(physical.info.offsets_g_eq_path[k] + i) =
                result.multipliers(naive.info.offsets_g_eq_path[k] + i);
        }
    }
    for (Index k = 0; k < physical.config.stages - 1; ++k)
    {
        for (Index i = 0; i < physical.config.nx; ++i)
            extracted(physical.info.offsets_g_eq_dyn[k] + i) =
                result.multipliers(naive.info.offsets_g_eq_dyn[k] + i);
    }
    return extracted;
}

double dense_reference_error(const PhysicalProblem &problem,
                             const BorderedResult &result)
{
    const Index physical_dim =
        problem.info.number_of_primal_variables + problem.info.number_of_eq_constraints;
    const Index total_dim = physical_dim + problem.config.parameters;
    if (total_dim > 2500)
        return std::numeric_limits<double>::quiet_NaN();

    Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(total_dim, total_dim);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(total_dim);

    VecRealAllocated basis(problem.info.number_of_primal_variables);
    VecRealAllocated hessian_column(problem.info.number_of_primal_variables);
    VecRealAllocated jacobian_column(problem.info.number_of_eq_constraints);
    for (Index column = 0; column < problem.info.number_of_primal_variables;
         ++column)
    {
        basis = 0.0;
        basis(column) = 1.0;
        hessian_column = 0.0;
        jacobian_column = 0.0;
        problem.hessian.apply_on_right(problem.info, basis, 0.0, hessian_column,
                                       hessian_column);
        problem.jacobian.apply_on_right(problem.info, basis, 0.0,
                                        jacobian_column, jacobian_column);
        for (Index row = 0; row < problem.info.number_of_primal_variables;
             ++row)
            matrix(row, column) = hessian_column(row);
        for (Index row = 0; row < problem.info.number_of_eq_constraints; ++row)
        {
            matrix(problem.info.number_of_primal_variables + row, column) =
                jacobian_column(row);
            matrix(column, problem.info.number_of_primal_variables + row) =
                jacobian_column(row);
        }
    }

    const Index parameter_offset = physical_dim;
    matrix.block(0, parameter_offset, problem.info.number_of_primal_variables,
                 problem.config.parameters) = problem.border_primal;
    matrix.block(parameter_offset, 0, problem.config.parameters,
                 problem.info.number_of_primal_variables) =
        problem.border_primal.transpose();
    matrix.block(problem.info.number_of_primal_variables, parameter_offset,
                 problem.info.number_of_eq_constraints, problem.config.parameters) =
        problem.border_constraints;
    matrix.block(parameter_offset, problem.info.number_of_primal_variables,
                 problem.config.parameters, problem.info.number_of_eq_constraints) =
        problem.border_constraints.transpose();
    matrix.block(parameter_offset, parameter_offset, problem.config.parameters,
                 problem.config.parameters) = problem.border_hessian;

    rhs.segment(0, problem.info.number_of_primal_variables) =
        to_eigen(problem.rhs_x);
    rhs.segment(problem.info.number_of_primal_variables,
                problem.info.number_of_eq_constraints) =
        to_eigen(problem.rhs_g);
    rhs.tail(problem.config.parameters) = problem.rhs_parameters;

    const Eigen::VectorXd reference = matrix.fullPivLu().solve(-rhs);
    Eigen::VectorXd structured(total_dim);
    structured.segment(0, problem.info.number_of_primal_variables) = result.primal;
    structured.segment(problem.info.number_of_primal_variables,
                       problem.info.number_of_eq_constraints) = result.multipliers;
    structured.tail(problem.config.parameters) = result.parameters;
    return max_abs(reference - structured);
}

Config parse_arguments(int argc, char **argv)
{
    Config config;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument(argv[i]);
        auto next_integer = [&](const char *name) {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string("Missing value for ") + name);
            return static_cast<Index>(std::stoll(argv[++i]));
        };
        auto next_double = [&](const char *name) {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string("Missing value for ") + name);
            return std::stod(argv[++i]);
        };

        if (argument == "--stages")
            config.stages = next_integer("--stages");
        else if (argument == "--problem")
        {
            if (i + 1 >= argc)
                throw std::runtime_error(
                    "Missing value for --problem");
            config.problem = argv[++i];
        }
        else if (argument == "--nx")
            config.nx = next_integer("--nx");
        else if (argument == "--nu")
            config.nu = next_integer("--nu");
        else if (argument == "--parameters")
            config.parameters = next_integer("--parameters");
        else if (argument == "--repeats")
            config.repeats = next_integer("--repeats");
        else if (argument == "--collocation-degree")
            config.collocation_degree =
                next_integer("--collocation-degree");
        else if (argument == "--cross-hessian-scale")
            config.cross_hessian_scale =
                next_double("--cross-hessian-scale");
        else if (argument == "--identity-next-jacobian")
            config.identity_next_jacobian = true;
        else if (argument == "--ipopt")
            config.run_ipopt = true;
        else if (argument == "--no-dense-validation")
            config.dense_validation = false;
        else if (argument == "--help")
        {
            std::cout
                << "Usage: global_parameter_kkt_benchmark [options]\n"
                << "  --problem {synthetic|dtoc3}\n"
                << "  --stages N\n"
                << "  --nx N\n"
                << "  --nu N\n"
                << "  --parameters N\n"
                << "  --repeats N\n"
                << "  --collocation-degree {1|2|3}\n"
                << "  --cross-hessian-scale VALUE (currently must be 0)\n"
                << "  --identity-next-jacobian\n"
                << "  --ipopt\n"
                << "  --no-dense-validation\n";
            std::exit(0);
        }
        else
            throw std::runtime_error("Unknown argument: " + argument);
    }

    if (config.problem != "synthetic" && config.problem != "dtoc3")
        throw std::runtime_error("Unknown benchmark problem");
    if (config.problem == "dtoc3")
    {
        config.nx = 2;
        config.nu = 1;
        config.collocation_degree = 0;
        config.identity_next_jacobian = true;
    }
    if (config.stages < 2 || config.nx < 1 || config.nu < 0 ||
        config.parameters < 0 || config.repeats < 1 ||
        config.cross_hessian_scale < 0.0 ||
        (config.collocation_degree != 0 &&
         config.collocation_degree != 1 &&
         config.collocation_degree != 2 &&
         config.collocation_degree != 3))
        throw std::runtime_error("Invalid non-positive benchmark dimension");
    if (config.collocation_degree > 0 &&
        config.cross_hessian_scale != 0.0)
        throw std::runtime_error(
            "Collocation benchmark currently requires zero FuFx");
    return config;
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const Config config = parse_arguments(argc, argv);
        PhysicalProblem physical(config);
        NormalizedProblem normalized(physical);
        NaiveProblem naive(physical);
        AugSystemSolver<OcpType> bordered_solver(normalized.info);
        const bool use_explicit_naive =
            config.collocation_degree > 0 ||
            config.problem == "dtoc3";
        std::unique_ptr<ExplicitNaiveProblem> explicit_naive;
        std::unique_ptr<AugSystemSolver<OcpType>>
            explicit_naive_solver;
        std::unique_ptr<AugSystemSolver<ImplicitOcpType>>
            implicit_naive_solver;
        if (use_explicit_naive)
        {
            explicit_naive =
                std::make_unique<ExplicitNaiveProblem>(naive);
            explicit_naive_solver =
                std::make_unique<AugSystemSolver<OcpType>>(
                    explicit_naive->info);
        }
        else
        {
            implicit_naive_solver =
                std::make_unique<
                    AugSystemSolver<ImplicitOcpType>>(naive.info);
            implicit_naive_solver->set_performance_mode(true);
        }

        // Untimed warm-up includes lazy code/data initialization.
        (void)solve_normalized_bordered(physical, normalized, bordered_solver);
        if (use_explicit_naive)
            (void)solve_explicit_naive(
                *explicit_naive, *explicit_naive_solver);
        else
            (void)solve_naive(naive, *implicit_naive_solver);

        std::vector<double> bordered_times;
        std::vector<double> normalization_times;
        std::vector<double> factor_times;
        std::vector<double> parameter_rhs_times;
        std::vector<double> schur_times;
        std::vector<double> naive_times;
        BorderedResult bordered_result;
        NaiveResult naive_result;
        for (Index repeat = 0; repeat < config.repeats; ++repeat)
        {
            bordered_result =
                solve_normalized_bordered(physical, normalized, bordered_solver);
            if (use_explicit_naive)
                naive_result = solve_explicit_naive(
                    *explicit_naive, *explicit_naive_solver);
            else
                naive_result =
                    solve_naive(naive, *implicit_naive_solver);
            bordered_times.push_back(bordered_result.elapsed_ms);
            normalization_times.push_back(
                bordered_result.normalization_ms);
            factor_times.push_back(bordered_result.factor_ms);
            parameter_rhs_times.push_back(
                bordered_result.parameter_rhs_ms);
            schur_times.push_back(bordered_result.schur_ms);
            naive_times.push_back(naive_result.elapsed_ms);
        }

        const Eigen::VectorXd naive_physical_primal =
            extract_naive_physical_primal(physical, naive, naive_result);
        const Eigen::VectorXd naive_physical_multipliers =
            extract_naive_physical_multipliers(physical, naive, naive_result);
        const double primal_difference =
            max_abs(bordered_result.primal - naive_physical_primal);
        const double multiplier_difference =
            max_abs(bordered_result.multipliers - naive_physical_multipliers);
        const double parameter_difference =
            max_abs(bordered_result.parameters - naive_result.parameters);
        const double bordered_kkt_residual =
            bordered_residual(physical, bordered_result);
        const double bordered_objective =
            objective_value(physical, bordered_result);
        const double naive_kkt_residual =
            implicit_kkt_residual(naive.info, naive.jacobian, naive.hessian,
                                  naive.rhs_x, naive.rhs_g, naive_result.primal,
                                  naive_result.multipliers);
        const double dense_error =
            config.dense_validation
                ? dense_reference_error(physical, bordered_result)
                : std::numeric_limits<double>::quiet_NaN();

        const double bordered_ms = median(bordered_times);
        const double normalization_ms = median(normalization_times);
        const double factor_ms = median(factor_times);
        const double parameter_rhs_ms = median(parameter_rhs_times);
        const double schur_ms = median(schur_times);
        const double naive_ms = median(naive_times);
        IpoptResult ipopt_result;
        double ipopt_primal_difference =
            std::numeric_limits<double>::quiet_NaN();
        double ipopt_multiplier_difference =
            std::numeric_limits<double>::quiet_NaN();
        double ipopt_parameter_difference =
            std::numeric_limits<double>::quiet_NaN();
        if (config.run_ipopt)
        {
            ipopt_result = solve_ipopt(physical);
            ipopt_primal_difference =
                max_abs(bordered_result.primal - ipopt_result.primal);
            ipopt_multiplier_difference =
                max_abs(bordered_result.multipliers -
                        ipopt_result.multipliers);
            ipopt_parameter_difference =
                max_abs(bordered_result.parameters -
                        ipopt_result.parameters);
        }

        std::cout << std::setprecision(12);
        std::cout
            << "problem,transcription,collocation_degree,stages,nx,nu,"
               "parameters,"
               "base_primal,base_constraints,"
               "naive_primal,naive_constraints,bordered_ms,"
               "normalization_ms,factor_ms,parameter_rhs_ms,schur_ms,"
               "naive_ms,"
               "naive_over_bordered,bordered_objective,"
               "bordered_residual,naive_residual,"
               "primal_difference,multiplier_difference,parameter_difference,"
               "parameter_consensus,dense_reference_difference,"
               "ipopt_ms,ipopt_status,ipopt_iterations,"
               "ipopt_primal_difference,ipopt_multiplier_difference,"
               "ipopt_parameter_difference\n";
        std::cout
            << config.problem << ','
            << (config.collocation_degree > 0 ? "radau" : "shooting")
            << ',' << config.collocation_degree << ',' << config.stages
            << ',' << config.nx << ',' << config.nu << ','
            << config.parameters << ',' << physical.info.number_of_primal_variables
            << ',' << physical.info.number_of_eq_constraints << ','
            << naive.info.number_of_primal_variables << ','
            << naive.info.number_of_eq_constraints << ',' << bordered_ms << ','
            << normalization_ms << ',' << factor_ms << ','
            << parameter_rhs_ms << ',' << schur_ms << ','
            << naive_ms << ',' << naive_ms / bordered_ms << ','
            << bordered_objective << ',' << bordered_kkt_residual << ','
            << naive_kkt_residual << ','
            << primal_difference << ',' << multiplier_difference << ','
            << parameter_difference << ',' << naive_result.parameter_consensus
            << ',' << dense_error << ',' << ipopt_result.elapsed_ms << ','
            << ipopt_result.status << ',' << ipopt_result.iterations << ','
            << ipopt_primal_difference << ','
            << ipopt_multiplier_difference << ','
            << ipopt_parameter_difference << '\n';

        const double tolerance = 5e-7;
        const bool ipopt_status_ok =
            !config.run_ipopt ||
            ipopt_result.status ==
                static_cast<int>(Ipopt::Solve_Succeeded) ||
            ipopt_result.status ==
                static_cast<int>(Ipopt::Solved_To_Acceptable_Level);
        const bool ipopt_solution_ok =
            !config.run_ipopt ||
            (ipopt_primal_difference <= tolerance &&
             ipopt_multiplier_difference <= tolerance &&
             ipopt_parameter_difference <= tolerance);
        if (bordered_kkt_residual > tolerance || naive_kkt_residual > tolerance ||
            primal_difference > tolerance || multiplier_difference > tolerance ||
            parameter_difference > tolerance ||
            naive_result.parameter_consensus > tolerance ||
            (std::isfinite(dense_error) && dense_error > tolerance) ||
            !ipopt_status_ok || !ipopt_solution_ok)
        {
            std::cerr << "Validation failed; tolerance=" << tolerance << '\n';
            return 2;
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "global_parameter_kkt_benchmark: " << error.what() << '\n';
        return 1;
    }
}
