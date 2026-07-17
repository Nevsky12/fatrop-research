//
// Copyright (c) 2026
//

#include "fatrop/ocp/global_parameter_kkt_solver.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

using namespace fatrop;

namespace
{
    struct Case
    {
        std::vector<Index> nx;
        std::vector<Index> nu;
        Index parameters;
    };

    Eigen::MatrixXd dense_jacobian(
        const ProblemInfo &info,
        const Jacobian<OcpType> &jacobian)
    {
        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(
            info.number_of_eq_constraints,
            info.number_of_primal_variables);
        for (Index stage = 0; stage < info.dims.K - 1; ++stage)
        {
            const Index variables =
                info.dims.number_of_controls[stage]
              + info.dims.number_of_states[stage];
            const Index next_states =
                info.dims.number_of_states[stage + 1];
            const Index variable_offset =
                info.offsets_primal_u[stage];
            const Index next_state_offset =
                info.offsets_primal_x[stage + 1];
            const Index row_offset =
                info.offsets_g_eq_dyn[stage];
            for (Index row = 0; row < next_states; ++row)
            {
                for (Index column = 0; column < variables; ++column)
                {
                    result(row_offset + row, variable_offset + column) =
                        jacobian.BAbt[stage](column, row);
                }
                result(row_offset + row, next_state_offset + row) = -1.0;
            }
        }
        return result;
    }

    Eigen::MatrixXd dense_hessian(
        const ProblemInfo &info,
        const Hessian<OcpType> &hessian,
        const VecRealView &D_x)
    {
        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(
            info.number_of_primal_variables,
            info.number_of_primal_variables);
        for (Index stage = 0; stage < info.dims.K; ++stage)
        {
            const Index variables =
                info.dims.number_of_controls[stage]
              + info.dims.number_of_states[stage];
            const Index offset = info.offsets_primal_u[stage];
            for (Index row = 0; row < variables; ++row)
            {
                for (Index column = 0; column < variables; ++column)
                {
                    result(offset + row, offset + column) =
                        hessian.RSQrqt[stage](row, column);
                }
            }
        }
        for (Index variable = 0;
             variable < info.number_of_primal_variables; ++variable)
        {
            result(variable, variable) += D_x(variable);
        }
        return result;
    }

    void run_dense_reference_case(const Case &config)
    {
        const Index stages = static_cast<Index>(config.nx.size());
        ASSERT_EQ(config.nu.size(), config.nx.size());
        std::vector<Index> equalities(
            static_cast<std::size_t>(stages), 0);
        std::vector<Index> inequalities(
            static_cast<std::size_t>(stages), 0);
        ProblemDims dims(
            stages, config.nu, config.nx,
            equalities, inequalities);
        ProblemInfo info(dims);
        Jacobian<OcpType> jacobian(dims);
        Hessian<OcpType> hessian(dims);

        for (Index stage = 0; stage < stages; ++stage)
        {
            const Index variables =
                config.nu[static_cast<std::size_t>(stage)]
              + config.nx[static_cast<std::size_t>(stage)];
            hessian.RSQrqt[stage] = 0.0;
            for (Index row = 0; row < variables; ++row)
            {
                for (Index column = 0; column < variables; ++column)
                {
                    if (row == column)
                    {
                        hessian.RSQrqt[stage](row, column) =
                            3.0 + 0.2 * stage + 0.1 * row;
                    }
                    else
                    {
                        hessian.RSQrqt[stage](row, column) =
                            0.015 / (1.0 + std::abs(row - column));
                    }
                }
            }
            if (stage + 1 < stages)
            {
                const Index next_states =
                    config.nx[static_cast<std::size_t>(stage + 1)];
                jacobian.BAbt[stage] = 0.0;
                for (Index row = 0; row < variables; ++row)
                {
                    for (Index column = 0;
                         column < next_states; ++column)
                    {
                        jacobian.BAbt[stage](row, column) =
                            0.04 * (1 + ((row + 2 * column + stage) % 7))
                          - 0.11;
                    }
                }
            }
        }

        VecRealAllocated D_x(info.number_of_primal_variables);
        VecRealAllocated D_s(info.number_of_slack_variables);
        VecRealAllocated rhs_primal(info.number_of_primal_variables);
        VecRealAllocated rhs_constraints(info.number_of_eq_constraints);
        VecRealAllocated rhs_parameters(config.parameters);
        D_s = 0.0;
        for (Index row = 0;
             row < info.number_of_primal_variables; ++row)
        {
            D_x(row) = 0.2 + 0.01 * (row % 5);
            rhs_primal(row) =
                0.07 * (row + 1) - 0.15 * (row % 3);
        }
        for (Index row = 0;
             row < info.number_of_eq_constraints; ++row)
        {
            rhs_constraints(row) =
                -0.03 * (row + 1) + 0.08 * (row % 4);
        }
        for (Index parameter = 0;
             parameter < config.parameters; ++parameter)
        {
            rhs_parameters(parameter) =
                0.12 * (parameter + 1) - 0.05 * (parameter % 2);
        }

        MatRealAllocated cross_hessian(
            info.number_of_primal_variables, config.parameters);
        MatRealAllocated parameter_jacobian(
            info.number_of_eq_constraints, config.parameters);
        MatRealAllocated parameter_hessian(
            config.parameters, config.parameters);
        for (Index row = 0;
             row < info.number_of_primal_variables; ++row)
        {
            for (Index parameter = 0;
                 parameter < config.parameters; ++parameter)
            {
                cross_hessian(row, parameter) =
                    0.025 * (1 + ((row + 3 * parameter) % 5)) - 0.06;
            }
        }
        for (Index row = 0;
             row < info.number_of_eq_constraints; ++row)
        {
            for (Index parameter = 0;
                 parameter < config.parameters; ++parameter)
            {
                parameter_jacobian(row, parameter) =
                    0.02 * (1 + ((2 * row + parameter) % 7)) - 0.07;
            }
        }
        for (Index row = 0; row < config.parameters; ++row)
        {
            for (Index column = 0;
                 column < config.parameters; ++column)
            {
                parameter_hessian(row, column) =
                    row == column
                    ? 4.0 + 0.3 * row
                    : 0.04 / (1.0 + std::abs(row - column));
            }
        }

        VecRealAllocated primal(info.number_of_primal_variables);
        VecRealAllocated multipliers(info.number_of_eq_constraints);
        VecRealAllocated parameters(config.parameters);
        primal = 0.0;
        multipliers = 0.0;
        parameters = 0.0;

        GlobalParameterKktSolver solver(info, config.parameters);
        const LinsolReturnFlag status = solver.solve(
            info, jacobian, hessian, D_x, D_s,
            cross_hessian, parameter_jacobian, parameter_hessian,
            rhs_primal, rhs_constraints, rhs_parameters,
            primal, multipliers, parameters);
        ASSERT_EQ(status, LinsolReturnFlag::SUCCESS);
        EXPECT_EQ(
            solver.last_failure_location(),
            BorderedKktFailureLocation::None);

        const Index trajectory_variables =
            info.number_of_primal_variables;
        const Index trajectory_constraints =
            info.number_of_eq_constraints;
        const Index total =
            trajectory_variables + trajectory_constraints
          + config.parameters;
        Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(total, total);
        const Eigen::MatrixXd H =
            dense_hessian(info, hessian, D_x);
        const Eigen::MatrixXd A =
            dense_jacobian(info, jacobian);
        dense.block(
            0, 0, trajectory_variables, trajectory_variables) = H;
        dense.block(
            0, trajectory_variables,
            trajectory_variables, trajectory_constraints) =
            A.transpose();
        dense.block(
            trajectory_variables, 0,
            trajectory_constraints, trajectory_variables) = A;

        for (Index row = 0; row < trajectory_variables; ++row)
        {
            for (Index parameter = 0;
                 parameter < config.parameters; ++parameter)
            {
                dense(row,
                      trajectory_variables + trajectory_constraints
                      + parameter) =
                    cross_hessian(row, parameter);
                dense(trajectory_variables + trajectory_constraints
                      + parameter, row) =
                    cross_hessian(row, parameter);
            }
        }
        for (Index row = 0; row < trajectory_constraints; ++row)
        {
            for (Index parameter = 0;
                 parameter < config.parameters; ++parameter)
            {
                dense(trajectory_variables + row,
                      trajectory_variables + trajectory_constraints
                      + parameter) =
                    parameter_jacobian(row, parameter);
                dense(trajectory_variables + trajectory_constraints
                      + parameter, trajectory_variables + row) =
                    parameter_jacobian(row, parameter);
            }
        }
        for (Index row = 0; row < config.parameters; ++row)
        {
            for (Index column = 0;
                 column < config.parameters; ++column)
            {
                dense(
                    trajectory_variables + trajectory_constraints + row,
                    trajectory_variables + trajectory_constraints + column) =
                    parameter_hessian(row, column);
            }
        }

        Eigen::VectorXd dense_rhs(total);
        for (Index row = 0; row < trajectory_variables; ++row)
            dense_rhs(row) = -rhs_primal(row);
        for (Index row = 0; row < trajectory_constraints; ++row)
            dense_rhs(trajectory_variables + row) =
                -rhs_constraints(row);
        for (Index row = 0; row < config.parameters; ++row)
        {
            dense_rhs(
                trajectory_variables + trajectory_constraints + row) =
                -rhs_parameters(row);
        }
        const Eigen::FullPivLU<Eigen::MatrixXd> lu(dense);
        ASSERT_EQ(lu.rank(), total);
        const Eigen::VectorXd reference = lu.solve(dense_rhs);
        ASSERT_LT(
            (dense * reference - dense_rhs).lpNorm<Eigen::Infinity>(),
            2e-11);

        Scalar error = 0.0;
        for (Index row = 0; row < trajectory_variables; ++row)
            error = std::max(error, std::abs(primal(row) - reference(row)));
        for (Index row = 0; row < trajectory_constraints; ++row)
        {
            error = std::max(
                error,
                std::abs(
                    multipliers(row)
                  - reference(trajectory_variables + row)));
        }
        for (Index row = 0; row < config.parameters; ++row)
        {
            error = std::max(
                error,
                std::abs(
                    parameters(row)
                  - reference(
                        trajectory_variables
                      + trajectory_constraints + row)));
        }
        EXPECT_LT(error, 2e-9);

        Eigen::VectorXd computed(total);
        for (Index row = 0; row < trajectory_variables; ++row)
            computed(row) = primal(row);
        for (Index row = 0; row < trajectory_constraints; ++row)
            computed(trajectory_variables + row) = multipliers(row);
        for (Index row = 0; row < config.parameters; ++row)
        {
            computed(
                trajectory_variables + trajectory_constraints + row) =
                parameters(row);
        }
        EXPECT_LT(
            (dense * computed - dense_rhs).lpNorm<Eigen::Infinity>(),
            2e-9);

        // A second right-hand side must reuse both the trajectory and the
        // parameter-Schur factorizations without changing the KKT matrix.
        for (Index row = 0; row < trajectory_variables; ++row)
            rhs_primal(row) = -0.09 * (row + 1) + 0.04 * (row % 2);
        for (Index row = 0; row < trajectory_constraints; ++row)
            rhs_constraints(row) = 0.05 * (row + 1) - 0.02 * (row % 3);
        for (Index row = 0; row < config.parameters; ++row)
            rhs_parameters(row) = -0.13 * (row + 1) + 0.03 * (row % 2);

        ASSERT_EQ(
            solver.solve_rhs(
                info, jacobian, hessian, D_s,
                cross_hessian, parameter_jacobian,
                rhs_primal, rhs_constraints, rhs_parameters,
                primal, multipliers, parameters),
            LinsolReturnFlag::SUCCESS);
        for (Index row = 0; row < trajectory_variables; ++row)
            dense_rhs(row) = -rhs_primal(row);
        for (Index row = 0; row < trajectory_constraints; ++row)
            dense_rhs(trajectory_variables + row) = -rhs_constraints(row);
        for (Index row = 0; row < config.parameters; ++row)
            dense_rhs(
                trajectory_variables + trajectory_constraints + row) =
                -rhs_parameters(row);
        for (Index row = 0; row < trajectory_variables; ++row)
            computed(row) = primal(row);
        for (Index row = 0; row < trajectory_constraints; ++row)
            computed(trajectory_variables + row) = multipliers(row);
        for (Index row = 0; row < config.parameters; ++row)
            computed(
                trajectory_variables + trajectory_constraints + row) =
                parameters(row);
        EXPECT_LT(
            (computed - lu.solve(dense_rhs)).lpNorm<Eigen::Infinity>(),
            2e-9);
        EXPECT_LT(
            (dense * computed - dense_rhs).lpNorm<Eigen::Infinity>(),
            2e-9);
    }
}

TEST(GlobalParameterKktSolverTest, MatchesDenseReferenceAcrossDimensions)
{
    const std::vector<Case> cases{
        { {2, 2}, {1, 0}, 1 },
        { {2, 3, 1, 4, 2}, {2, 1, 3, 1, 0}, 3 },
        { {1, 4, 2, 3, 5, 2, 4, 2},
          {3, 1, 2, 2, 1, 3, 1, 0}, 8 }
    };
    for (const Case &config : cases)
    {
        SCOPED_TRACE(
            "stages=" + std::to_string(config.nx.size())
          + ", parameters=" + std::to_string(config.parameters));
        run_dense_reference_case(config);
    }
}

TEST(GlobalParameterKktSolverTest, ReportsSingularParameterSchurComplement)
{
    const std::vector<Index> nx{1, 1};
    const std::vector<Index> nu{1, 0};
    const std::vector<Index> ng{0, 0};
    const std::vector<Index> ng_ineq{0, 0};
    const ProblemDims dims(2, nu, nx, ng, ng_ineq);
    const ProblemInfo info(dims);
    Jacobian<OcpType> jacobian(dims);
    Hessian<OcpType> hessian(dims);
    hessian.RSQrqt[0] = 0.0;
    hessian.RSQrqt[1] = 0.0;
    hessian.RSQrqt[0](0, 0) = 2.0;
    hessian.RSQrqt[0](1, 1) = 2.0;
    hessian.RSQrqt[1](0, 0) = 2.0;
    jacobian.BAbt[0] = 0.0;

    VecRealAllocated D_x(info.number_of_primal_variables);
    VecRealAllocated D_s(info.number_of_slack_variables);
    VecRealAllocated rhs_primal(info.number_of_primal_variables);
    VecRealAllocated rhs_constraints(info.number_of_eq_constraints);
    VecRealAllocated rhs_parameters(2);
    VecRealAllocated primal(info.number_of_primal_variables);
    VecRealAllocated multipliers(info.number_of_eq_constraints);
    VecRealAllocated parameters(2);
    MatRealAllocated cross_hessian(
        info.number_of_primal_variables, 2);
    MatRealAllocated parameter_jacobian(
        info.number_of_eq_constraints, 2);
    MatRealAllocated parameter_hessian(2, 2);

    D_x = 0.1;
    D_s = 0.0;
    rhs_primal = 0.0;
    rhs_constraints = 0.0;
    rhs_parameters = 1.0;
    cross_hessian = 0.0;
    parameter_jacobian = 0.0;
    parameter_hessian = 0.0;
    primal = 17.0;
    multipliers = 19.0;
    parameters = 23.0;

    GlobalParameterKktSolver solver(info, 2);
    EXPECT_EQ(
        solver.solve(
            info, jacobian, hessian, D_x, D_s,
            cross_hessian, parameter_jacobian, parameter_hessian,
            rhs_primal, rhs_constraints, rhs_parameters,
            primal, multipliers, parameters),
        LinsolReturnFlag::NOFULL_RANK);
    EXPECT_EQ(
        solver.last_failure_location(),
        BorderedKktFailureLocation::BorderFactorization);

    for (Index row = 0; row < primal.m(); ++row)
        EXPECT_DOUBLE_EQ(primal(row), 17.0);
    for (Index row = 0; row < multipliers.m(); ++row)
        EXPECT_DOUBLE_EQ(multipliers(row), 19.0);
    for (Index row = 0; row < parameters.m(); ++row)
        EXPECT_DOUBLE_EQ(parameters(row), 23.0);
}

TEST(ImplicitGlobalParameterKktSolverTest,
     MatchesDenseReferenceWithRankDeficientNextStateJacobian)
{
    constexpr Index stages = 5;
    constexpr Index state_dimension = 2;
    constexpr Index parameters_count = 3;
    const std::vector<Index> nx(
        static_cast<std::size_t>(stages), state_dimension);
    std::vector<Index> nu(
        static_cast<std::size_t>(stages), 1);
    nu.back() = 0;
    const std::vector<Index> ng(
        static_cast<std::size_t>(stages), 0);
    const std::vector<Index> ng_ineq(
        static_cast<std::size_t>(stages), 0);
    const ProblemDims dims(stages, nu, nx, ng, ng_ineq);
    const ProblemInfo info(dims);
    Jacobian<ImplicitOcpType> jacobian(dims);
    Hessian<ImplicitOcpType> hessian(dims);

    Eigen::MatrixXd dense_hessian = Eigen::MatrixXd::Zero(
        info.number_of_primal_variables,
        info.number_of_primal_variables);
    Eigen::MatrixXd dense_jacobian = Eigen::MatrixXd::Zero(
        info.number_of_eq_constraints,
        info.number_of_primal_variables);

    for (Index stage = 0; stage < stages; ++stage)
    {
        const Index variables = nu[stage] + nx[stage];
        const Index offset = info.offsets_primal_u[stage];
        hessian.RSQrqt[stage] = 0.0;
        for (Index row = 0; row < variables; ++row)
        {
            for (Index column = 0; column < variables; ++column)
            {
                const Scalar value =
                    row == column
                    ? 3.0 + 0.2 * stage + 0.1 * row
                    : 0.02 / (1.0 + std::abs(row - column));
                hessian.RSQrqt[stage](row, column) = value;
                dense_hessian(offset + row, offset + column) = value;
            }
        }

        if (stage + 1 == stages)
            continue;

        jacobian.BAbt[stage] = 0.0;
        jacobian.Jt[stage] = 0.0;
        hessian.FuFx[stage] = 0.0;
        hessian.GuGx[stage] = 0.0;

        // The first residual constrains the next physical state. The
        // second has no next-state derivative and instead uses the current
        // control. This is the collocation-relevant case where a nodal
        // control is carried as a state but remains free at the next node.
        jacobian.Jt[stage](0, 0) = -1.0;
        jacobian.BAbt[stage](nu[stage], 0) =
            0.85 + 0.01 * stage;
        jacobian.BAbt[stage](0, 1) = 1.2 + 0.03 * stage;
        jacobian.BAbt[stage](nu[stage] + 1, 1) = 0.15;

        const Index constraint_offset =
            info.offsets_g_eq_dyn[stage];
        const Index next_state_offset =
            info.offsets_primal_x[stage + 1];
        for (Index equation = 0;
             equation < state_dimension; ++equation)
        {
            for (Index variable = 0; variable < variables; ++variable)
            {
                dense_jacobian(
                    constraint_offset + equation,
                    offset + variable) =
                    jacobian.BAbt[stage](variable, equation);
            }
            for (Index state = 0; state < state_dimension; ++state)
            {
                dense_jacobian(
                    constraint_offset + equation,
                    next_state_offset + state) =
                    jacobian.Jt[stage](state, equation);
            }
        }

        for (Index row = 0; row < variables; ++row)
        {
            for (Index state = 0; state < state_dimension; ++state)
            {
                // A separated implicit residual such as trapezoidal
                // collocation has no mixed second derivative between the
                // two endpoint blocks.
                const Scalar value = 0.0;
                hessian.FuFx[stage](row, state) = value;
                dense_hessian(
                    offset + row,
                    next_state_offset + state) = value;
                dense_hessian(
                    next_state_offset + state,
                    offset + row) = value;
            }
        }
    }

    VecRealAllocated D_x(info.number_of_primal_variables);
    VecRealAllocated D_s(info.number_of_slack_variables);
    VecRealAllocated rhs_primal(info.number_of_primal_variables);
    VecRealAllocated rhs_constraints(info.number_of_eq_constraints);
    VecRealAllocated rhs_parameters(parameters_count);
    for (Index row = 0;
         row < info.number_of_primal_variables; ++row)
    {
        D_x(row) = 0.3 + 0.01 * (row % 3);
        dense_hessian(row, row) += D_x(row);
        rhs_primal(row) =
            0.09 * (row + 1) - 0.04 * (row % 2);
    }
    D_s = 0.0;
    for (Index row = 0;
         row < info.number_of_eq_constraints; ++row)
    {
        rhs_constraints(row) =
            -0.06 * (row + 1) + 0.02 * (row % 3);
    }
    for (Index parameter = 0;
         parameter < parameters_count; ++parameter)
    {
        rhs_parameters(parameter) =
            0.13 * (parameter + 1);
    }

    MatRealAllocated cross_hessian(
        info.number_of_primal_variables, parameters_count);
    MatRealAllocated parameter_jacobian(
        info.number_of_eq_constraints, parameters_count);
    MatRealAllocated parameter_hessian(
        parameters_count, parameters_count);
    for (Index row = 0;
         row < info.number_of_primal_variables; ++row)
    {
        for (Index parameter = 0;
             parameter < parameters_count; ++parameter)
        {
            cross_hessian(row, parameter) =
                0.007 * (1 + ((row + 2 * parameter) % 5)) - 0.015;
        }
    }
    for (Index row = 0;
         row < info.number_of_eq_constraints; ++row)
    {
        for (Index parameter = 0;
             parameter < parameters_count; ++parameter)
        {
            parameter_jacobian(row, parameter) =
                0.009 * (1 + ((2 * row + parameter) % 4)) - 0.012;
        }
    }
    for (Index row = 0; row < parameters_count; ++row)
    {
        for (Index column = 0;
             column < parameters_count; ++column)
        {
            parameter_hessian(row, column) =
                row == column
                ? 5.0 + 0.2 * row
                : 0.015 / (1.0 + std::abs(row - column));
        }
    }

    const Index primal_count = info.number_of_primal_variables;
    const Index constraint_count = info.number_of_eq_constraints;
    const Index total =
        primal_count + constraint_count + parameters_count;
    Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(total, total);
    dense.topLeftCorner(primal_count, primal_count) =
        dense_hessian;
    dense.block(
        0, primal_count, primal_count, constraint_count) =
        dense_jacobian.transpose();
    dense.block(
        primal_count, 0, constraint_count, primal_count) =
        dense_jacobian;
    for (Index row = 0; row < primal_count; ++row)
    {
        for (Index parameter = 0;
             parameter < parameters_count; ++parameter)
        {
            dense(row, primal_count + constraint_count + parameter) =
                cross_hessian(row, parameter);
            dense(primal_count + constraint_count + parameter, row) =
                cross_hessian(row, parameter);
        }
    }
    for (Index row = 0; row < constraint_count; ++row)
    {
        for (Index parameter = 0;
             parameter < parameters_count; ++parameter)
        {
            dense(
                primal_count + row,
                primal_count + constraint_count + parameter) =
                parameter_jacobian(row, parameter);
            dense(
                primal_count + constraint_count + parameter,
                primal_count + row) =
                parameter_jacobian(row, parameter);
        }
    }
    for (Index row = 0; row < parameters_count; ++row)
    {
        for (Index column = 0;
             column < parameters_count; ++column)
        {
            dense(
                primal_count + constraint_count + row,
                primal_count + constraint_count + column) =
                parameter_hessian(row, column);
        }
    }

    Eigen::VectorXd dense_rhs(total);
    for (Index row = 0; row < primal_count; ++row)
        dense_rhs(row) = -rhs_primal(row);
    for (Index row = 0; row < constraint_count; ++row)
        dense_rhs(primal_count + row) = -rhs_constraints(row);
    for (Index row = 0; row < parameters_count; ++row)
    {
        dense_rhs(primal_count + constraint_count + row) =
            -rhs_parameters(row);
    }
    const Eigen::FullPivLU<Eigen::MatrixXd> reference_factor(dense);
    ASSERT_EQ(reference_factor.rank(), total);
    const Eigen::VectorXd reference =
        reference_factor.solve(dense_rhs);

    VecRealAllocated primal(primal_count);
    VecRealAllocated multipliers(constraint_count);
    VecRealAllocated parameters(parameters_count);
    primal = 0.0;
    multipliers = 0.0;
    parameters = 0.0;
    ImplicitGlobalParameterKktSolver solver(
        info, parameters_count);
    solver.trajectory_solver().set_lu_fact_tol(1e-10);
    ASSERT_EQ(
        solver.solve(
            info, jacobian, hessian, D_x, D_s,
            cross_hessian, parameter_jacobian, parameter_hessian,
            rhs_primal, rhs_constraints, rhs_parameters,
            primal, multipliers, parameters),
        LinsolReturnFlag::SUCCESS);

    ASSERT_EQ(jacobian.J_ranks.size(),
              static_cast<std::size_t>(stages - 1));
    for (Index const rank : jacobian.J_ranks)
        EXPECT_EQ(rank, 1);

    Eigen::VectorXd computed(total);
    for (Index row = 0; row < primal_count; ++row)
        computed(row) = primal(row);
    for (Index row = 0; row < constraint_count; ++row)
        computed(primal_count + row) = multipliers(row);
    for (Index row = 0; row < parameters_count; ++row)
    {
        computed(primal_count + constraint_count + row) =
            parameters(row);
    }
    EXPECT_LT(
        (computed - reference).lpNorm<Eigen::Infinity>(),
        2e-8);
    EXPECT_LT(
        (dense * computed - dense_rhs).lpNorm<Eigen::Infinity>(),
        2e-8);
}

TEST(ImplicitGlobalParameterKktSolverTest,
     EqualityStabilizationHandlesSingularTrajectoryBorder)
{
    const std::vector<Index> nx{2};
    const std::vector<Index> nu{0};
    const std::vector<Index> ng{2};
    const std::vector<Index> ng_ineq{0};
    const ProblemDims dims(1, nu, nx, ng, ng_ineq);
    const ProblemInfo info(dims);
    Jacobian<ImplicitOcpType> jacobian(dims);
    Hessian<ImplicitOcpType> hessian(dims);

    jacobian.Gg_eqt[0] = 0.0;
    // With the border held fixed these two rows are duplicates, so the
    // ordinary trajectory KKT block is singular.
    jacobian.Gg_eqt[0](0, 0) = 1.0;
    jacobian.Gg_eqt[0](0, 1) = 1.0;
    hessian.RSQrqt[0] = 0.0;
    hessian.RSQrqt[0](0, 0) = 2.0;
    hessian.RSQrqt[0](1, 1) = 1.7;

    VecRealAllocated D_x(info.number_of_primal_variables);
    VecRealAllocated D_eq(info.number_of_eq_constraints);
    VecRealAllocated D_s(info.number_of_slack_variables);
    D_x(0) = 0.2;
    D_x(1) = 0.1;
    D_eq(0) = 0.05;
    D_eq(1) = 0.08;
    D_s = 0.0;

    MatRealAllocated cross_hessian(
        info.number_of_primal_variables, 1);
    MatRealAllocated parameter_jacobian(
        info.number_of_eq_constraints, 1);
    MatRealAllocated parameter_hessian(1, 1);
    cross_hessian(0, 0) = 0.3;
    cross_hessian(1, 0) = -0.2;
    parameter_jacobian(0, 0) = 0.0;
    parameter_jacobian(1, 0) = 1.0;
    parameter_hessian(0, 0) = 1.5;

    VecRealAllocated rhs_primal(info.number_of_primal_variables);
    VecRealAllocated rhs_constraints(info.number_of_eq_constraints);
    VecRealAllocated rhs_parameters(1);
    rhs_primal(0) = 0.7;
    rhs_primal(1) = -0.1;
    rhs_constraints(0) = -0.2;
    rhs_constraints(1) = 0.4;
    rhs_parameters(0) = -0.6;

    VecRealAllocated primal(info.number_of_primal_variables);
    VecRealAllocated multipliers(info.number_of_eq_constraints);
    VecRealAllocated parameters(1);
    primal = 0.0;
    multipliers = 0.0;
    parameters = 0.0;

    Eigen::Matrix4d trajectory_dense = Eigen::Matrix4d::Zero();
    trajectory_dense(0, 0) = 2.0 + D_x(0);
    trajectory_dense(1, 1) = 1.7 + D_x(1);
    trajectory_dense(0, 2) = trajectory_dense(2, 0) = 1.0;
    trajectory_dense(0, 3) = trajectory_dense(3, 0) = 1.0;
    trajectory_dense(2, 2) = -D_eq(0);
    trajectory_dense(3, 3) = -D_eq(1);

    Jacobian<ImplicitOcpType> direct_jacobian(dims);
    Hessian<ImplicitOcpType> direct_hessian(dims);
    direct_jacobian.Gg_eqt[0] = 0.0;
    direct_jacobian.Gg_eqt[0](0, 0) = 1.0;
    direct_jacobian.Gg_eqt[0](0, 1) = 1.0;
    direct_hessian.RSQrqt[0] = 0.0;
    direct_hessian.RSQrqt[0](0, 0) = 2.0;
    direct_hessian.RSQrqt[0](1, 1) = 1.7;
    VecRealAllocated direct_primal(info.number_of_primal_variables);
    VecRealAllocated direct_multipliers(info.number_of_eq_constraints);
    direct_primal = 0.0;
    direct_multipliers = 0.0;
    AugSystemSolver<ImplicitOcpType> trajectory_solver(info);
    trajectory_solver.set_lu_fact_tol(1e-12);
    ASSERT_EQ(
        trajectory_solver.solve(
            info, direct_jacobian, direct_hessian,
            D_x, D_eq, D_s, rhs_primal, rhs_constraints,
            direct_primal, direct_multipliers),
        LinsolReturnFlag::SUCCESS);
    Eigen::Vector4d direct_base;
    direct_base <<
        direct_primal(0), direct_primal(1),
        direct_multipliers(0), direct_multipliers(1);
    Eigen::Vector4d trajectory_rhs;
    trajectory_rhs <<
        -rhs_primal(0), -rhs_primal(1),
        -rhs_constraints(0), -rhs_constraints(1);
    ASSERT_LT(
        (trajectory_dense * direct_base - trajectory_rhs)
            .lpNorm<Eigen::Infinity>(),
        2e-10);

    ImplicitGlobalParameterKktSolver solver(info, 1);
    solver.trajectory_solver().set_lu_fact_tol(1e-12);
    ASSERT_EQ(
        solver.solve(
            info, jacobian, hessian, D_x, D_eq, D_s,
            cross_hessian, parameter_jacobian, parameter_hessian,
            rhs_primal, rhs_constraints, rhs_parameters,
            primal, multipliers, parameters),
        LinsolReturnFlag::SUCCESS);

    Eigen::Matrix<double, 5, 5> dense =
        Eigen::Matrix<double, 5, 5>::Zero();
    dense(0, 0) = 2.0 + D_x(0);
    dense(1, 1) = 1.7 + D_x(1);
    dense(0, 2) = dense(2, 0) = 1.0;
    dense(0, 3) = dense(3, 0) = 1.0;
    dense(0, 4) = dense(4, 0) = cross_hessian(0, 0);
    dense(1, 4) = dense(4, 1) = cross_hessian(1, 0);
    dense(2, 2) = -D_eq(0);
    dense(3, 3) = -D_eq(1);
    dense(3, 4) = dense(4, 3) = parameter_jacobian(1, 0);
    dense(4, 4) = parameter_hessian(0, 0);

    Eigen::Matrix<double, 5, 1> dense_rhs;
    dense_rhs <<
        -rhs_primal(0),
        -rhs_primal(1),
        -rhs_constraints(0),
        -rhs_constraints(1),
        -rhs_parameters(0);
    const Eigen::Matrix<double, 5, 1> reference =
        dense.fullPivLu().solve(dense_rhs);
    Eigen::Matrix<double, 5, 1> computed;
    computed <<
        primal(0), primal(1),
        multipliers(0), multipliers(1), parameters(0);

    EXPECT_LT((computed - reference).lpNorm<Eigen::Infinity>(), 2e-10)
        << "computed=" << computed.transpose()
        << "\nreference=" << reference.transpose();
    EXPECT_LT(
        (dense * computed - dense_rhs).lpNorm<Eigen::Infinity>(),
        2e-10)
        << "residual=" << (dense * computed - dense_rhs).transpose();
}

TEST(ImplicitGlobalParameterKktSolverTest,
     StabilizedFullRankBatchMatchesDenseMultistageKkt)
{
    constexpr Index stages = 3;
    constexpr Index parameters_count = 4;
    const std::vector<Index> nx{2, 2, 2};
    const std::vector<Index> nu{1, 1, 0};
    const std::vector<Index> ng{1, 1, 1};
    const std::vector<Index> ng_ineq{1, 1, 1};
    const ProblemDims dims(stages, nu, nx, ng, ng_ineq);
    const ProblemInfo info(dims);
    Jacobian<ImplicitOcpType> jacobian(dims);
    Hessian<ImplicitOcpType> hessian(dims);

    const Index primal_count = info.number_of_primal_variables;
    const Index constraint_count = info.number_of_eq_constraints;
    Eigen::MatrixXd dense_hessian =
        Eigen::MatrixXd::Zero(primal_count, primal_count);
    Eigen::MatrixXd dense_jacobian =
        Eigen::MatrixXd::Zero(constraint_count, primal_count);

    for (Index stage = 0; stage < stages; ++stage)
    {
        const Index variables = nu[stage] + nx[stage];
        const Index offset = info.offsets_primal_u[stage];
        hessian.RSQrqt[stage] = 0.0;
        jacobian.Gg_eqt[stage] = 0.0;
        jacobian.Gg_ineqt[stage] = 0.0;
        for (Index row = 0; row < variables; ++row)
        {
            for (Index column = 0; column < variables; ++column)
            {
                const Scalar value =
                    row == column
                    ? 4.0 + 0.2 * stage + 0.1 * row
                    : 0.025 / (1.0 + std::abs(row - column));
                hessian.RSQrqt[stage](row, column) = value;
                dense_hessian(offset + row, offset + column) = value;
            }
            const Scalar equality_value =
                0.12 + 0.03 * ((row + stage) % 4);
            const Scalar inequality_value =
                -0.08 + 0.025 * ((2 * row + stage) % 5);
            jacobian.Gg_eqt[stage](row, 0) = equality_value;
            jacobian.Gg_ineqt[stage](row, 0) = inequality_value;
            dense_jacobian(
                info.offsets_g_eq_path[stage], offset + row) =
                equality_value;
            dense_jacobian(
                info.offsets_g_eq_slack[stage], offset + row) =
                inequality_value;
        }

        if (stage + 1 == stages)
            continue;
        jacobian.BAbt[stage] = 0.0;
        jacobian.Jt[stage] = 0.0;
        hessian.FuFx[stage] = 0.0;
        hessian.GuGx[stage] = 0.0;
        for (Index variable = 0; variable < variables; ++variable)
        for (Index equation = 0; equation < nx[stage + 1]; ++equation)
        {
            jacobian.BAbt[stage](variable, equation) =
                0.04 * (1 + ((variable + 2 * equation + stage) % 5))
              - 0.09;
        }
        if (stage == 0)
        {
            // A deliberately pivoted, non-diagonal next-state block.
            jacobian.Jt[stage](0, 0) = 0.10;
            jacobian.Jt[stage](0, 1) = 1.20;
            jacobian.Jt[stage](1, 0) = 1.10;
            jacobian.Jt[stage](1, 1) = 0.20;
        }
        else
        {
            jacobian.Jt[stage](0, 0) = 1.30;
            jacobian.Jt[stage](0, 1) = 0.40;
            jacobian.Jt[stage](1, 0) = 0.20;
            jacobian.Jt[stage](1, 1) = 0.90;
        }

        const Index dynamics_offset =
            info.offsets_g_eq_dyn[stage];
        const Index next_state_offset =
            info.offsets_primal_x[stage + 1];
        for (Index equation = 0;
             equation < nx[stage + 1]; ++equation)
        {
            for (Index variable = 0; variable < variables; ++variable)
            {
                dense_jacobian(
                    dynamics_offset + equation,
                    offset + variable) =
                    jacobian.BAbt[stage](variable, equation);
            }
            for (Index state = 0; state < nx[stage + 1]; ++state)
            {
                dense_jacobian(
                    dynamics_offset + equation,
                    next_state_offset + state) =
                    jacobian.Jt[stage](state, equation);
            }
        }
    }

    VecRealAllocated D_x(primal_count);
    VecRealAllocated D_eq(info.number_of_g_eq_path);
    VecRealAllocated D_s(info.number_of_slack_variables);
    VecRealAllocated rhs_primal(primal_count);
    VecRealAllocated rhs_constraints(constraint_count);
    VecRealAllocated rhs_parameters(parameters_count);
    D_eq = 0.07;
    for (Index row = 0; row < primal_count; ++row)
    {
        D_x(row) = 0.20 + 0.01 * (row % 3);
        dense_hessian(row, row) += D_x(row);
        rhs_primal(row) =
            0.08 * (row + 1) - 0.03 * (row % 2);
    }
    for (Index row = 0; row < constraint_count; ++row)
        rhs_constraints(row) =
            -0.035 * (row + 1) + 0.02 * (row % 3);
    for (Index stage = 0; stage < stages; ++stage)
    {
        D_eq(info.offsets_eq[stage]) = 0.045 + 0.01 * stage;
        D_s(info.offsets_slack[stage]) = 0.09 + 0.015 * stage;
    }
    for (Index parameter = 0;
         parameter < parameters_count; ++parameter)
        rhs_parameters(parameter) = 0.11 * (parameter + 1);

    MatRealAllocated cross_hessian(primal_count, parameters_count);
    MatRealAllocated parameter_jacobian(
        constraint_count, parameters_count);
    MatRealAllocated parameter_hessian(
        parameters_count, parameters_count);
    for (Index row = 0; row < primal_count; ++row)
    for (Index parameter = 0;
         parameter < parameters_count; ++parameter)
        cross_hessian(row, parameter) =
            0.012 * (1 + ((row + 2 * parameter) % 5)) - 0.025;
    for (Index row = 0; row < constraint_count; ++row)
    for (Index parameter = 0;
         parameter < parameters_count; ++parameter)
        parameter_jacobian(row, parameter) =
            0.01 * (1 + ((2 * row + parameter) % 6)) - 0.022;
    for (Index row = 0; row < parameters_count; ++row)
    for (Index column = 0; column < parameters_count; ++column)
        parameter_hessian(row, column) =
            row == column
            ? 3.5 + 0.2 * row
            : 0.018 / (1.0 + std::abs(row - column));

    const Index total =
        primal_count + constraint_count + parameters_count;
    Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(total, total);
    dense.topLeftCorner(primal_count, primal_count) = dense_hessian;
    dense.block(0, primal_count, primal_count, constraint_count) =
        dense_jacobian.transpose();
    dense.block(primal_count, 0, constraint_count, primal_count) =
        dense_jacobian;
    for (Index stage = 0; stage < stages; ++stage)
    {
        dense(
            primal_count + info.offsets_g_eq_path[stage],
            primal_count + info.offsets_g_eq_path[stage]) =
            -D_eq(info.offsets_eq[stage]);
        dense(
            primal_count + info.offsets_g_eq_slack[stage],
            primal_count + info.offsets_g_eq_slack[stage]) =
            -D_s(info.offsets_slack[stage]);
    }
    for (Index row = 0; row < primal_count; ++row)
    for (Index parameter = 0;
         parameter < parameters_count; ++parameter)
    {
        dense(row, primal_count + constraint_count + parameter) =
            cross_hessian(row, parameter);
        dense(primal_count + constraint_count + parameter, row) =
            cross_hessian(row, parameter);
    }
    for (Index row = 0; row < constraint_count; ++row)
    for (Index parameter = 0;
         parameter < parameters_count; ++parameter)
    {
        dense(
            primal_count + row,
            primal_count + constraint_count + parameter) =
            parameter_jacobian(row, parameter);
        dense(
            primal_count + constraint_count + parameter,
            primal_count + row) =
            parameter_jacobian(row, parameter);
    }
    for (Index row = 0; row < parameters_count; ++row)
    for (Index column = 0; column < parameters_count; ++column)
        dense(
            primal_count + constraint_count + row,
            primal_count + constraint_count + column) =
            parameter_hessian(row, column);

    Eigen::VectorXd dense_rhs(total);
    for (Index row = 0; row < primal_count; ++row)
        dense_rhs(row) = -rhs_primal(row);
    for (Index row = 0; row < constraint_count; ++row)
        dense_rhs(primal_count + row) = -rhs_constraints(row);
    for (Index row = 0; row < parameters_count; ++row)
        dense_rhs(primal_count + constraint_count + row) =
            -rhs_parameters(row);
    const Eigen::FullPivLU<Eigen::MatrixXd> reference_factor(dense);
    ASSERT_EQ(reference_factor.rank(), total);
    const Eigen::VectorXd reference =
        reference_factor.solve(dense_rhs);

    VecRealAllocated primal(primal_count);
    VecRealAllocated multipliers(constraint_count);
    VecRealAllocated parameters(parameters_count);
    primal = 0.0;
    multipliers = 0.0;
    parameters = 0.0;
    ImplicitGlobalParameterKktSolver solver(info, parameters_count);
    solver.trajectory_solver().set_lu_fact_tol(1e-12);
    ASSERT_EQ(
        solver.solve(
            info, jacobian, hessian,
            D_x, D_eq, D_s,
            cross_hessian, parameter_jacobian, parameter_hessian,
            rhs_primal, rhs_constraints, rhs_parameters,
            primal, multipliers, parameters),
        LinsolReturnFlag::SUCCESS);
    for (Index const rank : jacobian.J_ranks)
        EXPECT_EQ(rank, 2);

    Eigen::VectorXd computed(total);
    for (Index row = 0; row < primal_count; ++row)
        computed(row) = primal(row);
    for (Index row = 0; row < constraint_count; ++row)
        computed(primal_count + row) = multipliers(row);
    for (Index row = 0; row < parameters_count; ++row)
        computed(primal_count + constraint_count + row) = parameters(row);
    EXPECT_LT(
        (computed - reference).lpNorm<Eigen::Infinity>(), 3e-8)
        << "computed=" << computed.transpose()
        << "\nreference=" << reference.transpose();
    EXPECT_LT(
        (dense * computed - dense_rhs).lpNorm<Eigen::Infinity>(), 3e-8)
        << "residual=" << (dense * computed - dense_rhs).transpose();

    for (Index row = 0; row < primal_count; ++row)
        rhs_primal(row) = -0.045 * (row + 1) + 0.015 * (row % 3);
    for (Index row = 0; row < constraint_count; ++row)
        rhs_constraints(row) = 0.025 * (row + 1) - 0.01 * (row % 2);
    for (Index row = 0; row < parameters_count; ++row)
        rhs_parameters(row) = -0.09 * (row + 1);
    ASSERT_EQ(
        solver.solve_rhs(
            info, jacobian, hessian, D_eq, D_s,
            cross_hessian, parameter_jacobian,
            rhs_primal, rhs_constraints, rhs_parameters,
            primal, multipliers, parameters),
        LinsolReturnFlag::SUCCESS);
    for (Index row = 0; row < primal_count; ++row)
    {
        dense_rhs(row) = -rhs_primal(row);
        computed(row) = primal(row);
    }
    for (Index row = 0; row < constraint_count; ++row)
    {
        dense_rhs(primal_count + row) = -rhs_constraints(row);
        computed(primal_count + row) = multipliers(row);
    }
    for (Index row = 0; row < parameters_count; ++row)
    {
        dense_rhs(primal_count + constraint_count + row) =
            -rhs_parameters(row);
        computed(primal_count + constraint_count + row) = parameters(row);
    }
    EXPECT_LT(
        (computed - reference_factor.solve(dense_rhs))
            .lpNorm<Eigen::Infinity>(),
        3e-8);
    EXPECT_LT(
        (dense * computed - dense_rhs).lpNorm<Eigen::Infinity>(), 3e-8);
}
