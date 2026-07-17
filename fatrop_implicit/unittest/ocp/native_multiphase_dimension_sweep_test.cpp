#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/ocp/aug_system_solver.hpp"
#include "fatrop/ocp/dims.hpp"
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/pd_solver_orig.hpp"
#include "fatrop/ocp/pd_system_orig.hpp"
#include "fatrop/ocp/problem_info.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

using namespace fatrop;

namespace
{
    void run_case(
        Index phases, Index nx, Index nu, Index np,
        Index collocation_degree)
    {
        constexpr Index nodes_per_phase = 3;
        const Index stages = phases * nodes_per_phase;
        std::vector<Index> states(stages, nx);
        std::vector<Index> controls(
            stages, nu + collocation_degree * nx);
        std::vector<Index> equalities(
            stages, collocation_degree * nx);
        std::vector<Index> inequalities(stages, 0);
        for (Index stage = nodes_per_phase - 1;
             stage < stages; stage += nodes_per_phase)
        {
            controls[stage] = 0;
            equalities[stage] = 0;
        }

        const ProblemDims dims(
            stages, controls, states, equalities, inequalities, np);
        const ProblemInfo info(dims);
        Jacobian<ImplicitOcpType> jacobian(dims);
        Hessian<ImplicitOcpType> hessian(dims);
        jacobian.global_parameter_jacobian = 0.0;
        hessian.set_zero();

        for (Index stage = 0; stage < stages; ++stage)
        {
            jacobian.Gg_eqt[stage] = 0.0;
            jacobian.Gg_ineqt[stage] = 0.0;
            const Index local_variables = controls[stage] + nx;
            for (Index equation = 0;
                 equation < equalities[stage]; ++equation)
            {
                // Collocation states are local stage variables placed after
                // the physical controls.  Their residual block is square and
                // receives a small dependence on the mesh state and p.
                jacobian.Gg_eqt[stage](nu + equation, equation) = 1.0;
                jacobian.Gg_eqt[stage](controls[stage] + equation % nx, equation)
                    = -0.1 / (1.0 + equation / nx);
                const Index global_equation =
                    info.offsets_g_eq_path[stage] + equation;
                for (Index parameter = 0; parameter < np; ++parameter)
                    jacobian.global_parameter_jacobian(
                        global_equation, parameter) =
                        0.002 * (1 + ((equation + parameter) % 3));
            }
            for (Index row = 0; row < local_variables; ++row)
            {
                hessian.RSQrqt[stage](row, row) =
                    2.0 + 0.03 * stage + 0.02 * row;
                if (row + 1 < local_variables)
                {
                    hessian.RSQrqt[stage](row, row + 1) = 0.01;
                    hessian.RSQrqt[stage](row + 1, row) = 0.01;
                }
                const Index global_row = info.offsets_primal_u[stage] + row;
                for (Index parameter = 0; parameter < np; ++parameter)
                    hessian.global_parameter_cross_hessian(
                        global_row, parameter) =
                        1e-3 * (1 + ((global_row + parameter) % 5));
            }

            if (stage + 1 == stages)
                continue;
            jacobian.BAbt[stage] = 0.0;
            jacobian.Jt[stage] = 0.0;
            jacobian.Jt_inv[stage] = 0.0;
            hessian.FuFx[stage] = 0.0;
            hessian.GuGx[stage] = 0.0;

            const bool phase_link =
                stage % nodes_per_phase == nodes_per_phase - 1;
            for (Index state = 0; state < nx; ++state)
            {
                jacobian.Jt[stage](state, state) = -1.0;
                jacobian.Jt_inv[stage](state, state) = -1.0;
                const Index state_row = controls[stage] + state;
                jacobian.BAbt[stage](state_row, state) =
                    phase_link ? 0.8 : 1.0;
                for (Index control = 0;
                     control < controls[stage]; ++control)
                    jacobian.BAbt[stage](control, state) =
                        0.02 * (1 + ((control + state + stage) % 3));

                const Index equation = info.offsets_g_eq_dyn[stage] + state;
                for (Index parameter = 0; parameter < np; ++parameter)
                    jacobian.global_parameter_jacobian(
                        equation, parameter) =
                        (phase_link ? 0.015 : 0.005)
                        * (1 + ((state + parameter) % 4));
            }
        }
        for (Index row = 0; row < np; ++row)
        {
            hessian.global_parameter_hessian(row, row) = 3.0 + 0.1 * row;
            if (row + 1 < np)
            {
                hessian.global_parameter_hessian(row, row + 1) = 0.02;
                hessian.global_parameter_hessian(row + 1, row) = 0.02;
            }
        }

        VecRealAllocated diagonal(info.number_of_primal_variables);
        VecRealAllocated equality_diagonal(info.number_of_eq_constraints);
        VecRealAllocated rhs_primal(info.number_of_primal_variables);
        VecRealAllocated rhs_constraints(info.number_of_eq_constraints);
        diagonal = 0.15;
        equality_diagonal = 0.0;
        for (Index row = 0; row < rhs_primal.m(); ++row)
            rhs_primal(row) = 0.01 * (1 + (row % 11));
        for (Index row = 0; row < rhs_constraints.m(); ++row)
            rhs_constraints(row) = -0.008 * (1 + (row % 7));

        VecRealAllocated empty(info.number_of_slack_variables);
        VecRealAllocated rhs_slack(info.number_of_slack_variables);
        VecRealAllocated rhs_lower(info.number_of_slack_variables);
        VecRealAllocated rhs_upper(info.number_of_slack_variables);
        auto trajectory_solver =
            std::make_shared<AugSystemSolver<ImplicitOcpType>>(info);
        trajectory_solver->set_lu_fact_tol(1e-12);
        PdSolverOrig<ImplicitOcpType> solver(info, trajectory_solver);
        LinearSystem<PdSystemType<ImplicitOcpType>> system(
            info, jacobian, hessian, diagonal, true, equality_diagonal,
            empty, empty, empty, empty, rhs_primal, rhs_slack,
            rhs_constraints, rhs_lower, rhs_upper);

        VecRealAllocated rhs(system.m());
        VecRealAllocated solution(system.m());
        VecRealAllocated residual(system.m());
        system.get_rhs(rhs);
        ASSERT_EQ(solver.solve_once(system, solution), LinsolReturnFlag::SUCCESS);
        system.apply_on_right(solution, 1.0, rhs, residual);
        EXPECT_LT(norm_inf(residual), 2e-8);
    }
}

TEST(NativeMultiphaseDimensionSweep, VariesPhasesStateControlAndParameterSizes)
{
    for (const Index phases : {1, 2, 4, 8})
        for (const Index nx : {1, 3, 6})
            for (const Index nu : {1, 2})
                for (const Index np : {0, 1, 3, 6})
                {
                    SCOPED_TRACE(
                        "phases=" + std::to_string(phases)
                        + " nx=" + std::to_string(nx)
                        + " nu=" + std::to_string(nu)
                        + " np=" + std::to_string(np));
                    run_case(phases, nx, nu, np, 0);
                }
}

TEST(NativeMultiphaseDimensionSweep, CoversImplicitDirectCollocationDegrees)
{
    for (const Index phases : {1, 2, 4})
        for (const Index nx : {1, 3})
            for (const Index nu : {1, 2})
                for (const Index np : {0, 1, 3})
                    for (const Index degree : {1, 2, 3})
                    {
                        SCOPED_TRACE(
                            "phases=" + std::to_string(phases)
                            + " nx=" + std::to_string(nx)
                            + " nu=" + std::to_string(nu)
                            + " np=" + std::to_string(np)
                            + " degree=" + std::to_string(degree));
                        run_case(phases, nx, nu, np, degree);
                    }
}
