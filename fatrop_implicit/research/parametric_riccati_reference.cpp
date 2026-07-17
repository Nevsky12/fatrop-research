#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

using Eigen::Index;
using Eigen::MatrixXd;
using Eigen::VectorXd;

struct Config
{
    Index intervals = 12;
    Index phases = 3;
    Index nx = 4;
    Index nu = 3;
    Index parameters = 2;
    Index equalities = 1;
};

double deterministic_value(Index i, Index j, double scale)
{
    return scale *
           (std::sin(0.17 * static_cast<double>(i + 1) +
                     0.31 * static_cast<double>(j + 1)) +
            0.5 * std::cos(0.23 * static_cast<double>(i + 2) -
                           0.11 * static_cast<double>(j + 3)));
}

MatrixXd deterministic_spd(Index dimension, Index seed, double diagonal)
{
    MatrixXd factor(dimension, dimension);
    for (Index row = 0; row < dimension; ++row)
    {
        for (Index column = 0; column < dimension; ++column)
        {
            factor(row, column) =
                deterministic_value(row + 3 * seed, column + seed, 0.18);
        }
    }
    MatrixXd result =
        factor.transpose() * factor /
        std::max<double>(1.0, static_cast<double>(dimension));
    result.diagonal().array() += diagonal;
    return result;
}

Index phase_of_interval(const Config &config, Index interval)
{
    return std::min<Index>(
        config.phases - 1,
        interval * config.phases / config.intervals);
}

bool is_phase_linkage(const Config &config, Index interval)
{
    if (interval + 1 >= config.intervals)
        return false;
    return phase_of_interval(config, interval) !=
           phase_of_interval(config, interval + 1);
}

struct Stage
{
    MatrixXd R;
    MatrixXd N;
    MatrixXd Q;
    VectorXd r;
    VectorXd s;
    double constant = 0.0;

    MatrixXd A;
    MatrixXd B;
    MatrixXd E;
    VectorXd a;

    MatrixXd D;
    MatrixXd C;
    VectorXd d;
};

struct Problem
{
    Config config;
    VectorXd initial_state;
    std::vector<Stage> stages;
    MatrixXd terminal_W;
    VectorXd terminal_w;
    double terminal_constant = 0.0;
};

Problem generate_problem(const Config &config)
{
    const Index nq = config.nx + config.parameters;
    Problem problem;
    problem.config = config;
    problem.initial_state.resize(config.nx);
    for (Index state = 0; state < config.nx; ++state)
    {
        problem.initial_state(state) =
            deterministic_value(state, config.phases, 0.2);
    }

    problem.stages.reserve(static_cast<std::size_t>(config.intervals));
    for (Index k = 0; k < config.intervals; ++k)
    {
        Stage stage;
        const MatrixXd stage_hessian =
            deterministic_spd(config.nu + nq, k + 1, 0.4);
        stage.R = stage_hessian.topLeftCorner(config.nu, config.nu);
        stage.N =
            stage_hessian.block(0, config.nu, config.nu, nq);
        stage.Q =
            stage_hessian.bottomRightCorner(nq, nq);
        stage.r.resize(config.nu);
        stage.s.resize(nq);
        for (Index i = 0; i < config.nu; ++i)
            stage.r(i) = deterministic_value(k, i, 0.08);
        for (Index i = 0; i < nq; ++i)
            stage.s(i) = deterministic_value(k + 2, i, 0.06);
        stage.constant = deterministic_value(k, 7, 0.03);

        const Index phase = phase_of_interval(config, k);
        const bool linkage = is_phase_linkage(config, k);
        stage.A = MatrixXd::Zero(config.nx, config.nx);
        stage.B.resize(config.nx, config.nu);
        stage.E.resize(config.nx, config.parameters);
        stage.a.resize(config.nx);
        for (Index row = 0; row < config.nx; ++row)
        {
            for (Index column = 0; column < config.nx; ++column)
            {
                const double diagonal =
                    row == column
                        ? (linkage ? 0.58 : 0.82) +
                              0.01 * static_cast<double>(
                                         (phase + row) % 5)
                        : 0.0;
                const double neighbour =
                    std::abs(row - column) == 1
                        ? (linkage ? 0.035 : 0.018) *
                              std::cos(0.13 * static_cast<double>(
                                                  row + column + phase + 1))
                        : 0.0;
                stage.A(row, column) = diagonal + neighbour;
            }
            for (Index control = 0; control < config.nu; ++control)
            {
                stage.B(row, control) =
                    (linkage ? 0.025 : 0.09) *
                    std::cos(0.12 * static_cast<double>(
                                        (row + 1) * (control + 2)) +
                             0.08 * static_cast<double>(phase));
            }
            for (Index parameter = 0;
                 parameter < config.parameters; ++parameter)
            {
                stage.E(row, parameter) =
                    (linkage ? 0.075 : 0.045) *
                    std::sin(0.14 * static_cast<double>(
                                        (row + 2) * (parameter + 1)) +
                             0.09 * static_cast<double>(phase));
            }
            stage.a(row) =
                deterministic_value(k * config.nx + row, phase, 0.04);
        }

        stage.D = MatrixXd::Zero(config.equalities, config.nu);
        stage.C.resize(config.equalities, nq);
        stage.d.resize(config.equalities);
        for (Index equation = 0; equation < config.equalities;
             ++equation)
        {
            for (Index control = 0; control < config.nu; ++control)
            {
                stage.D(equation, control) =
                    control == equation
                        ? 1.0
                        : deterministic_value(
                              equation + k, control, 0.035);
            }
            for (Index column = 0; column < nq; ++column)
            {
                stage.C(equation, column) =
                    deterministic_value(
                        equation + 2 * k, column + phase, 0.055);
            }
            stage.d(equation) =
                deterministic_value(k, equation + 4, 0.025);
        }
        problem.stages.push_back(std::move(stage));
    }

    problem.terminal_W =
        deterministic_spd(nq, config.intervals + 11, 0.7);
    problem.terminal_w.resize(nq);
    for (Index i = 0; i < nq; ++i)
    {
        problem.terminal_w(i) =
            deterministic_value(config.intervals, i, 0.09);
    }
    problem.terminal_constant =
        deterministic_value(config.intervals, 19, 0.05);
    return problem;
}

struct ValueFunction
{
    MatrixXd W;
    VectorXd w;
    double constant = 0.0;
};

struct Policy
{
    MatrixXd primal_feedback;
    VectorXd primal_feedforward;
    MatrixXd multiplier_feedback;
    VectorXd multiplier_feedforward;
};

struct RiccatiResult
{
    std::vector<ValueFunction> values;
    std::vector<Policy> policies;
    std::vector<VectorXd> states;
    std::vector<VectorXd> controls;
    std::vector<VectorXd> path_multipliers;
    std::vector<VectorXd> dynamics_multipliers;
    VectorXd parameters;
    double objective = 0.0;
    double local_solve_residual = 0.0;
};

RiccatiResult solve_parametric_riccati(const Problem &problem)
{
    const Config &config = problem.config;
    const Index nq = config.nx + config.parameters;
    const Index local_kkt_dim = config.nu + config.equalities;

    RiccatiResult result;
    result.values.resize(
        static_cast<std::size_t>(config.intervals + 1));
    result.policies.resize(
        static_cast<std::size_t>(config.intervals));
    result.values.back() = {
        problem.terminal_W,
        problem.terminal_w,
        problem.terminal_constant};

    for (Index k = config.intervals; k-- > 0;)
    {
        const Stage &stage =
            problem.stages[static_cast<std::size_t>(k)];
        const ValueFunction &next =
            result.values[static_cast<std::size_t>(k + 1)];

        MatrixXd transition_q =
            MatrixXd::Zero(nq, nq);
        transition_q.topLeftCorner(config.nx, config.nx) =
            stage.A;
        if (config.parameters > 0)
        {
            transition_q.topRightCorner(
                config.nx, config.parameters) = stage.E;
            transition_q.bottomRightCorner(
                config.parameters, config.parameters)
                .setIdentity();
        }
        MatrixXd transition_u =
            MatrixXd::Zero(nq, config.nu);
        transition_u.topRows(config.nx) = stage.B;
        VectorXd transition_constant = VectorXd::Zero(nq);
        transition_constant.head(config.nx) = stage.a;

        const VectorXd shifted_next_gradient =
            next.W * transition_constant + next.w;
        const MatrixXd reduced_control_hessian =
            stage.R +
            transition_u.transpose() * next.W * transition_u;
        const MatrixXd control_state_cross =
            stage.N +
            transition_u.transpose() * next.W * transition_q;
        const MatrixXd propagated_hessian =
            stage.Q +
            transition_q.transpose() * next.W * transition_q;
        const VectorXd propagated_control_gradient =
            stage.r +
            transition_u.transpose() * shifted_next_gradient;
        const VectorXd propagated_gradient =
            stage.s +
            transition_q.transpose() * shifted_next_gradient;
        const double propagated_constant =
            stage.constant + next.constant +
            0.5 * transition_constant.dot(
                      next.W * transition_constant) +
            next.w.dot(transition_constant);

        MatrixXd local_kkt =
            MatrixXd::Zero(local_kkt_dim, local_kkt_dim);
        local_kkt.topLeftCorner(config.nu, config.nu) =
            reduced_control_hessian;
        if (config.equalities > 0)
        {
            local_kkt.topRightCorner(
                config.nu, config.equalities) =
                stage.D.transpose();
            local_kkt.bottomLeftCorner(
                config.equalities, config.nu) = stage.D;
        }

        MatrixXd coupling(local_kkt_dim, nq);
        coupling.topRows(config.nu) = control_state_cross;
        if (config.equalities > 0)
            coupling.bottomRows(config.equalities) = stage.C;
        VectorXd affine(local_kkt_dim);
        affine.head(config.nu) = propagated_control_gradient;
        if (config.equalities > 0)
            affine.tail(config.equalities) = stage.d;

        Eigen::FullPivLU<MatrixXd> local_lu(local_kkt);
        if (local_lu.rank() != local_kkt_dim)
            throw std::runtime_error(
                "Generated local equality-constrained KKT block is singular");

        MatrixXd right_hand_sides(
            local_kkt_dim, nq + 1);
        right_hand_sides.leftCols(nq) = coupling;
        right_hand_sides.rightCols(1) = affine;
        const MatrixXd response =
            -local_lu.solve(right_hand_sides);
        result.local_solve_residual = std::max(
            result.local_solve_residual,
            (local_kkt * response + right_hand_sides)
                .cwiseAbs()
                .maxCoeff());

        Policy policy;
        policy.primal_feedback =
            response.topRows(config.nu).leftCols(nq);
        policy.primal_feedforward =
            response.topRows(config.nu).rightCols(1);
        policy.multiplier_feedback =
            response.bottomRows(config.equalities).leftCols(nq);
        policy.multiplier_feedforward =
            response.bottomRows(config.equalities).rightCols(1);
        result.policies[static_cast<std::size_t>(k)] =
            std::move(policy);

        ValueFunction current;
        current.W =
            propagated_hessian +
            coupling.transpose() * response.leftCols(nq);
        current.W =
            0.5 * (current.W + current.W.transpose());
        current.w =
            propagated_gradient +
            coupling.transpose() * response.rightCols(1);
        current.constant =
            propagated_constant +
            0.5 * affine.dot(response.col(nq));
        result.values[static_cast<std::size_t>(k)] =
            std::move(current);
    }

    const ValueFunction &initial = result.values.front();
    result.parameters = VectorXd::Zero(config.parameters);
    if (config.parameters > 0)
    {
        const MatrixXd parameter_hessian =
            initial.W.bottomRightCorner(
                config.parameters, config.parameters);
        const VectorXd parameter_gradient =
            initial.w.tail(config.parameters) +
            initial.W.bottomLeftCorner(
                config.parameters, config.nx) *
                problem.initial_state;
        Eigen::LDLT<MatrixXd> parameter_ldlt(parameter_hessian);
        if (parameter_ldlt.info() != Eigen::Success)
            throw std::runtime_error(
                "Condensed parameter Hessian factorization failed");
        result.parameters =
            parameter_ldlt.solve(-parameter_gradient);
        if (parameter_ldlt.info() != Eigen::Success)
            throw std::runtime_error(
                "Condensed parameter solve failed");
    }

    result.states.resize(
        static_cast<std::size_t>(config.intervals + 1));
    result.controls.resize(
        static_cast<std::size_t>(config.intervals));
    result.path_multipliers.resize(
        static_cast<std::size_t>(config.intervals));
    result.dynamics_multipliers.resize(
        static_cast<std::size_t>(config.intervals));
    result.states.front() = problem.initial_state;
    for (Index k = 0; k < config.intervals; ++k)
    {
        VectorXd q(nq);
        q.head(config.nx) =
            result.states[static_cast<std::size_t>(k)];
        if (config.parameters > 0)
            q.tail(config.parameters) = result.parameters;
        const Policy &policy =
            result.policies[static_cast<std::size_t>(k)];
        result.controls[static_cast<std::size_t>(k)] =
            policy.primal_feedback * q +
            policy.primal_feedforward;
        result.path_multipliers[static_cast<std::size_t>(k)] =
            policy.multiplier_feedback * q +
            policy.multiplier_feedforward;

        const Stage &stage =
            problem.stages[static_cast<std::size_t>(k)];
        result.states[static_cast<std::size_t>(k + 1)] =
            stage.A * result.states[static_cast<std::size_t>(k)] +
            stage.B * result.controls[static_cast<std::size_t>(k)] +
            stage.E * result.parameters + stage.a;

        VectorXd next_q(nq);
        next_q.head(config.nx) =
            result.states[static_cast<std::size_t>(k + 1)];
        if (config.parameters > 0)
            next_q.tail(config.parameters) = result.parameters;
        const ValueFunction &next =
            result.values[static_cast<std::size_t>(k + 1)];
        result.dynamics_multipliers[static_cast<std::size_t>(k)] =
            -(next.W * next_q + next.w).head(config.nx);
    }

    VectorXd initial_q(nq);
    initial_q.head(config.nx) = problem.initial_state;
    if (config.parameters > 0)
        initial_q.tail(config.parameters) = result.parameters;
    result.objective =
        0.5 * initial_q.dot(initial.W * initial_q) +
        initial.w.dot(initial_q) + initial.constant;
    return result;
}

struct DenseQp
{
    MatrixXd hessian;
    VectorXd gradient;
    MatrixXd jacobian;
    VectorXd constraint_constant;
    double objective_constant = 0.0;
};

Index control_offset(const Config &config, Index interval)
{
    return interval * config.nu;
}

Index state_offset(const Config &config, Index state_node)
{
    if (state_node < 1 || state_node > config.intervals)
        throw std::runtime_error("Only x_1 through x_N are dense variables");
    return config.intervals * config.nu +
           (state_node - 1) * config.nx;
}

Index parameter_offset(const Config &config)
{
    return config.intervals * (config.nu + config.nx);
}

DenseQp assemble_dense_qp(const Problem &problem)
{
    const Config &config = problem.config;
    const Index nq = config.nx + config.parameters;
    const Index variables =
        config.intervals * (config.nu + config.nx) +
        config.parameters;
    const Index constraints =
        config.intervals * (config.nx + config.equalities);

    DenseQp dense{
        MatrixXd::Zero(variables, variables),
        VectorXd::Zero(variables),
        MatrixXd::Zero(constraints, variables),
        VectorXd::Zero(constraints),
        0.0};

    for (Index k = 0; k < config.intervals; ++k)
    {
        const Stage &stage =
            problem.stages[static_cast<std::size_t>(k)];
        MatrixXd local_hessian(config.nu + nq, config.nu + nq);
        local_hessian.topLeftCorner(config.nu, config.nu) = stage.R;
        local_hessian.topRightCorner(config.nu, nq) = stage.N;
        local_hessian.bottomLeftCorner(nq, config.nu) =
            stage.N.transpose();
        local_hessian.bottomRightCorner(nq, nq) = stage.Q;
        VectorXd local_gradient(config.nu + nq);
        local_gradient.head(config.nu) = stage.r;
        local_gradient.tail(nq) = stage.s;

        MatrixXd map =
            MatrixXd::Zero(config.nu + nq, variables);
        VectorXd affine = VectorXd::Zero(config.nu + nq);
        map.block(0, control_offset(config, k),
                  config.nu, config.nu)
            .setIdentity();
        if (k == 0)
            affine.segment(config.nu, config.nx) =
                problem.initial_state;
        else
        {
            map.block(config.nu, state_offset(config, k),
                      config.nx, config.nx)
                .setIdentity();
        }
        if (config.parameters > 0)
        {
            map.block(config.nu + config.nx,
                      parameter_offset(config),
                      config.parameters, config.parameters)
                .setIdentity();
        }

        dense.hessian.noalias() +=
            map.transpose() * local_hessian * map;
        dense.gradient.noalias() +=
            map.transpose() *
            (local_hessian * affine + local_gradient);
        dense.objective_constant +=
            stage.constant +
            0.5 * affine.dot(local_hessian * affine) +
            local_gradient.dot(affine);

        const Index dynamics_row = k * config.nx;
        dense.jacobian.block(
            dynamics_row, control_offset(config, k),
            config.nx, config.nu) = -stage.B;
        if (k == 0)
        {
            dense.constraint_constant.segment(
                dynamics_row, config.nx) =
                -stage.A * problem.initial_state - stage.a;
        }
        else
        {
            dense.jacobian.block(
                dynamics_row, state_offset(config, k),
                config.nx, config.nx) = -stage.A;
            dense.constraint_constant.segment(
                dynamics_row, config.nx) = -stage.a;
        }
        dense.jacobian.block(
            dynamics_row, state_offset(config, k + 1),
            config.nx, config.nx)
            .setIdentity();
        if (config.parameters > 0)
        {
            dense.jacobian.block(
                dynamics_row, parameter_offset(config),
                config.nx, config.parameters) = -stage.E;
        }

        const Index path_row =
            config.intervals * config.nx +
            k * config.equalities;
        if (config.equalities > 0)
        {
            dense.jacobian.block(
                path_row, control_offset(config, k),
                config.equalities, config.nu) = stage.D;
            if (k == 0)
            {
                dense.constraint_constant.segment(
                    path_row, config.equalities) =
                    stage.C.leftCols(config.nx) *
                        problem.initial_state +
                    stage.d;
            }
            else
            {
                dense.jacobian.block(
                    path_row, state_offset(config, k),
                    config.equalities, config.nx) =
                    stage.C.leftCols(config.nx);
                dense.constraint_constant.segment(
                    path_row, config.equalities) = stage.d;
            }
            if (config.parameters > 0)
            {
                dense.jacobian.block(
                    path_row, parameter_offset(config),
                    config.equalities, config.parameters) =
                    stage.C.rightCols(config.parameters);
            }
        }
    }

    MatrixXd terminal_map =
        MatrixXd::Zero(
            config.nx + config.parameters,
            variables);
    terminal_map.block(
        0, state_offset(config, config.intervals),
        config.nx, config.nx)
        .setIdentity();
    if (config.parameters > 0)
    {
        terminal_map.block(
            config.nx, parameter_offset(config),
            config.parameters, config.parameters)
            .setIdentity();
    }
    dense.hessian.noalias() +=
        terminal_map.transpose() *
        problem.terminal_W * terminal_map;
    dense.gradient.noalias() +=
        terminal_map.transpose() * problem.terminal_w;
    dense.objective_constant += problem.terminal_constant;
    dense.hessian =
        0.5 * (dense.hessian + dense.hessian.transpose());
    return dense;
}

VectorXd pack_primal(const Problem &problem,
                     const RiccatiResult &result)
{
    const Config &config = problem.config;
    VectorXd primal(
        config.intervals * (config.nu + config.nx) +
        config.parameters);
    for (Index k = 0; k < config.intervals; ++k)
    {
        primal.segment(control_offset(config, k), config.nu) =
            result.controls[static_cast<std::size_t>(k)];
        primal.segment(state_offset(config, k + 1), config.nx) =
            result.states[static_cast<std::size_t>(k + 1)];
    }
    if (config.parameters > 0)
    {
        primal.tail(config.parameters) = result.parameters;
    }
    return primal;
}

VectorXd pack_multipliers(const Problem &problem,
                          const RiccatiResult &result)
{
    const Config &config = problem.config;
    VectorXd multipliers(
        config.intervals * (config.nx + config.equalities));
    for (Index k = 0; k < config.intervals; ++k)
    {
        multipliers.segment(k * config.nx, config.nx) =
            result.dynamics_multipliers[static_cast<std::size_t>(k)];
        if (config.equalities > 0)
        {
            multipliers.segment(
                config.intervals * config.nx +
                    k * config.equalities,
                config.equalities) =
                result.path_multipliers[static_cast<std::size_t>(k)];
        }
    }
    return multipliers;
}

double max_abs(const VectorXd &vector)
{
    if (vector.size() == 0)
        return 0.0;
    return vector.cwiseAbs().maxCoeff();
}

struct Validation
{
    double primal_difference = 0.0;
    double multiplier_difference = 0.0;
    double riccati_kkt_residual = 0.0;
    double dense_kkt_residual = 0.0;
    double objective_difference = 0.0;
    double minimum_parameter_eigenvalue =
        std::numeric_limits<double>::quiet_NaN();
};

Validation validate(const Problem &problem,
                    const RiccatiResult &riccati)
{
    const DenseQp dense = assemble_dense_qp(problem);
    const Index variables = dense.hessian.rows();
    const Index constraints = dense.jacobian.rows();
    MatrixXd kkt =
        MatrixXd::Zero(
            variables + constraints,
            variables + constraints);
    kkt.topLeftCorner(variables, variables) = dense.hessian;
    kkt.topRightCorner(variables, constraints) =
        dense.jacobian.transpose();
    kkt.bottomLeftCorner(constraints, variables) =
        dense.jacobian;
    VectorXd rhs(variables + constraints);
    rhs.head(variables) = dense.gradient;
    rhs.tail(constraints) = dense.constraint_constant;

    Eigen::FullPivLU<MatrixXd> dense_lu(kkt);
    if (dense_lu.rank() != kkt.rows())
        throw std::runtime_error("Assembled dense KKT system is singular");
    const VectorXd dense_solution = dense_lu.solve(-rhs);
    const VectorXd riccati_primal = pack_primal(problem, riccati);
    const VectorXd riccati_multipliers =
        pack_multipliers(problem, riccati);
    VectorXd riccati_solution(variables + constraints);
    riccati_solution.head(variables) = riccati_primal;
    riccati_solution.tail(constraints) = riccati_multipliers;

    const double dense_objective =
        0.5 *
            dense_solution.head(variables).dot(
                dense.hessian * dense_solution.head(variables)) +
        dense.gradient.dot(dense_solution.head(variables)) +
        dense.objective_constant;

    Validation validation;
    validation.primal_difference =
        max_abs(riccati_primal - dense_solution.head(variables));
    validation.multiplier_difference =
        max_abs(
            riccati_multipliers -
            dense_solution.tail(constraints));
    validation.riccati_kkt_residual =
        max_abs(kkt * riccati_solution + rhs);
    validation.dense_kkt_residual =
        max_abs(kkt * dense_solution + rhs);
    validation.objective_difference =
        std::abs(riccati.objective - dense_objective);
    if (problem.config.parameters > 0)
    {
        const MatrixXd parameter_hessian =
            riccati.values.front().W.bottomRightCorner(
                problem.config.parameters,
                problem.config.parameters);
        validation.minimum_parameter_eigenvalue =
            Eigen::SelfAdjointEigenSolver<MatrixXd>(
                parameter_hessian)
                .eigenvalues()
                .minCoeff();
    }
    return validation;
}

Config parse_arguments(int argc, char **argv)
{
    Config config;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument(argv[i]);
        auto next_integer = [&](const char *name) {
            if (i + 1 >= argc)
            {
                throw std::runtime_error(
                    std::string("Missing value for ") + name);
            }
            return static_cast<Index>(std::stoll(argv[++i]));
        };

        if (argument == "--intervals")
            config.intervals = next_integer("--intervals");
        else if (argument == "--phases")
            config.phases = next_integer("--phases");
        else if (argument == "--nx")
            config.nx = next_integer("--nx");
        else if (argument == "--nu")
            config.nu = next_integer("--nu");
        else if (argument == "--parameters")
            config.parameters = next_integer("--parameters");
        else if (argument == "--equalities")
            config.equalities = next_integer("--equalities");
        else if (argument == "--help")
        {
            std::cout
                << "Usage: parametric_riccati_reference [options]\n"
                << "  --intervals N\n"
                << "  --phases N\n"
                << "  --nx N\n"
                << "  --nu N\n"
                << "  --parameters N\n"
                << "  --equalities N (must not exceed nu)\n";
            std::exit(0);
        }
        else
            throw std::runtime_error(
                "Unknown argument: " + argument);
    }

    if (config.intervals < 1 || config.phases < 1 ||
        config.phases > config.intervals ||
        config.nx < 1 || config.nu < 1 ||
        config.parameters < 0 || config.equalities < 0 ||
        config.equalities > config.nu)
    {
        throw std::runtime_error("Invalid reference problem dimensions");
    }
    return config;
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const Config config = parse_arguments(argc, argv);
        const Problem problem = generate_problem(config);
        const RiccatiResult riccati =
            solve_parametric_riccati(problem);
        const Validation validation =
            validate(problem, riccati);

        std::cout << std::setprecision(12);
        std::cout
            << "intervals,phases,linkages,nx,nu,parameters,equalities,"
               "local_solve_residual,riccati_kkt_residual,"
               "dense_kkt_residual,primal_difference,"
               "multiplier_difference,objective_difference,"
               "minimum_parameter_eigenvalue\n";
        std::cout
            << config.intervals << ',' << config.phases << ','
            << config.phases - 1 << ',' << config.nx << ','
            << config.nu << ',' << config.parameters << ','
            << config.equalities << ','
            << riccati.local_solve_residual << ','
            << validation.riccati_kkt_residual << ','
            << validation.dense_kkt_residual << ','
            << validation.primal_difference << ','
            << validation.multiplier_difference << ','
            << validation.objective_difference << ','
            << validation.minimum_parameter_eigenvalue << '\n';

        const double tolerance = 2e-9;
        if (riccati.local_solve_residual > tolerance ||
            validation.riccati_kkt_residual > tolerance ||
            validation.dense_kkt_residual > tolerance ||
            validation.primal_difference > tolerance ||
            validation.multiplier_difference > tolerance ||
            validation.objective_difference > tolerance)
        {
            std::cerr
                << "Parametric Riccati validation failed; tolerance="
                << tolerance << '\n';
            return 2;
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr
            << "parametric_riccati_reference: "
            << error.what() << '\n';
        return 1;
    }
}
