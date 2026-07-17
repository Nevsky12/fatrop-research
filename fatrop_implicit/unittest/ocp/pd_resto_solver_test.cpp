
//
// Copyright (c) Lander Vanroye, KU Leuven
//
#include "../random_matrix.hpp"
#include "fatrop/common/exception.hpp"
#include "fatrop/context/context.hpp"
#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/ocp/aug_system_solver.hpp"
#include "fatrop/ocp/dims.hpp"
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/pd_solver_orig.hpp"
#include "fatrop/ocp/pd_solver_resto.hpp"
#include "fatrop/ocp/pd_system_orig.hpp"
#include "fatrop/ocp/pd_system_resto.hpp"
#include "fatrop/ocp/problem_info.hpp"
#include "fatrop/ocp/type.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace fatrop;

class PdTest : public ::testing::Test
{
protected:
    // // Create OcpDims object
    int K = 10;                                                   // Number of stages
    std::vector<Index> nx = {20, 10, 10, 10, 10, 2, 0, 1, 10, 5}; // State dimensions for each stage
    std::vector<Index> nu = {1, 4, 2, 10, 1, 30, 4, 1, 10, 0};    // Input dimensions for each stage
    // std::vector<Index> ng = std::vector<Index>(K, 0);      // Equality constraints for each stage
    // std::vector<Index> ng_ineq = std::vector<Index>(K, 1); // Inequality constraints for each
    // stage
    std::vector<Index> ng = {9, 3, 4, 3, 4, 0, 1, 0, 1, 5}; // Equality constraints for each

    // std::vector<Index> ng = {9, 3, 4, 3, 4, 0, 1, 0, 1, 5}; // Equality constraints for each
    std::vector<Index> ng_ineq = {0, 5, 10, 0,  0,
                                  0, 0, 0,  10, 0}; // Inequality constraints for each stage
    // Create OcpDims object
    // int K = 3;                                                   // Number of stages
    // std::vector<Index> nx = {2, 2, 0}; // State dimensions for each stage
    // std::vector<Index> nu = {2, 2, 2};    // Input dimensions for each stage
    // // std::vector<Index> ng = std::vector<Index>(K, 0);      // Equality constraints for each stage
    // // std::vector<Index> ng_ineq = std::vector<Index>(K, 1); // Inequality constraints for each
    // // stage
    // std::vector<Index> ng = {2, 0, 0}; // Equality constraints for each

    // // std::vector<Index> ng = {9, 3, 4, 3, 4, 0, 1, 0, 1, 5}; // Equality constraints for each
    // std::vector<Index> ng_ineq = {3, 0, 0};

    ProblemDims dims{K, nu, nx, ng, ng_ineq};

    ProblemInfo info{dims};
    // Create Jacobian object
    Jacobian<OcpType> jacobian{dims};
    MatRealAllocated full_matrix_jacobian =
        MatRealAllocated(info.number_of_eq_constraints, info.number_of_primal_variables);
    Hessian<OcpType> hessian{dims};
    MatRealAllocated full_matrix_hessian =
        MatRealAllocated(info.number_of_primal_variables, info.number_of_primal_variables);
    VecRealAllocated x = VecRealAllocated(info.number_of_primal_variables);
    VecRealAllocated sl = VecRealAllocated(info.number_of_slack_variables_resto);
    VecRealAllocated su = VecRealAllocated(info.number_of_slack_variables_resto);
    VecRealAllocated zl = VecRealAllocated(info.number_of_slack_variables_resto);
    VecRealAllocated zu = VecRealAllocated(info.number_of_slack_variables_resto);
    VecRealAllocated rhs_cl = VecRealAllocated(info.number_of_slack_variables_resto);
    VecRealAllocated rhs_cu = VecRealAllocated(info.number_of_slack_variables_resto);
    VecRealAllocated mult = VecRealAllocated(info.number_of_eq_constraints);
    VecRealAllocated rhs_x = VecRealAllocated(info.number_of_primal_variables);
    VecRealAllocated rhs_g = VecRealAllocated(info.number_of_eq_constraints);
    VecRealAllocated rhs_s = VecRealAllocated(info.number_of_slack_variables_resto);
    VecRealAllocated D_x =
        VecRealAllocated(info.number_of_primal_variables + info.number_of_slack_variables_resto);
    VecRealAllocated D_eq = VecRealAllocated(info.number_of_eq_constraints);
    std::shared_ptr<AugSystemSolver<OcpType>> solver =
        std::make_shared<AugSystemSolver<OcpType>>(info);
    std::shared_ptr<PdSolverOrig<OcpType>> pd_solver_sp =
        std::make_shared<PdSolverOrig<OcpType>>(info, solver);
    PdSolverOrig<OcpType> &pd_solver = *pd_solver_sp;
    PdSolverResto<OcpType> pd_solver_resto = PdSolverResto<OcpType>(info, pd_solver_sp);

    void SetUp()
    {
        x = 0;
        // fill the jacobian with random values
        for (Index k = 0; k < info.dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            if (k < info.dims.K - 1)
            {
                Index nx_next = info.dims.number_of_states[k + 1];
                jacobian.BAbt[k].block(nu + nx, nx_next, 0, 0) =
                    ::test::random_matrix(nu + nx, nx_next);
            }
            jacobian.Gg_eqt[k].block(nu + nx, info.dims.number_of_eq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info.dims.number_of_eq_constraints[k]);
            jacobian.Gg_ineqt[k].block(nu + nx, info.dims.number_of_ineq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info.dims.number_of_ineq_constraints[k]);
        }
        // fill the Hessian with random values
        for (Index k = 0; k < dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            hessian.RSQrqt[k].block(nu + nx, nu + nx, 0, 0) = ::test::random_spd_matrix(nu + nx);
            hessian.RSQrqt[k].diagonal() = hessian.RSQrqt[k].diagonal()+ 1.0;
        }
        // fill the x vector with random values
        for (Index i = 0; i < info.number_of_primal_variables; ++i)
        {
            rhs_x(i) = 1.0 * i;
            D_x = 1.0 * (i + 0.1);
        }
        // fill the mult vector with random values
        for (Index i = 0; i < info.number_of_eq_constraints; ++i)
        {
            rhs_g(i) = 1.0 * i;
            D_eq(i) = 1e-8 * (i + 1);
        }

        for (Index i = 0; i < info.number_of_slack_variables_resto; ++i)
        {
            sl(i) = 1. + 0.1 * i;
            su(i) = 1. + 0.2 * i;
            zl(i) = 1. + 0.3 * i;
            zu(i) = 1. + 0.4 * i;
            rhs_cl(i) = 1. + 0.5 * i;
            rhs_cu(i) = 1. + 0.6 * i;
            rhs_s(i) = 1. + 0.7 * i; 
        }
        sl.block(info.number_of_eq_constraints, info.offset_p).block(info.number_of_g_eq_dyn, info.offset_g_eq_dyn) = 1.;
        sl.block(info.number_of_eq_constraints, info.offset_n).block(info.number_of_g_eq_dyn, info.offset_g_eq_dyn) = 1.;
        zl.block(info.number_of_eq_constraints, info.offset_p).block(info.number_of_g_eq_dyn, info.offset_g_eq_dyn) = 0.;
        zl.block(info.number_of_eq_constraints, info.offset_n).block(info.number_of_g_eq_dyn, info.offset_g_eq_dyn) = 0.;
        rhs_cl.block(info.number_of_eq_constraints, info.offset_p).block(info.number_of_g_eq_dyn, info.offset_g_eq_dyn) = 0;
        rhs_cl.block(info.number_of_eq_constraints, info.offset_n).block(info.number_of_g_eq_dyn, info.offset_g_eq_dyn) = 0;
        rhs_s.block(info.number_of_eq_constraints, info.offset_p).block(info.number_of_g_eq_dyn, info.offset_g_eq_dyn) = 0;
        rhs_s.block(info.number_of_eq_constraints, info.offset_n).block(info.number_of_g_eq_dyn, info.offset_g_eq_dyn) = 0;
        // and set all upper bound related variables 
        su.block(info.number_of_eq_constraints, info.offset_n) = 1.;
        su.block(info.number_of_eq_constraints, info.offset_p) = 1.;
        zu.block(info.number_of_eq_constraints, info.offset_n) = 0.;
        zu.block(info.number_of_eq_constraints, info.offset_p) = 0.;
        rhs_cu.block(info.number_of_eq_constraints, info.offset_n) = 0.;
        rhs_cu.block(info.number_of_eq_constraints, info.offset_p) = 0.;

        // 
        // sl = 1.;
        // su = 1.;
        // zl = 0.;
        // zu = 0.;
        // rhs_cl = 0.;
        // rhs_cu = 0.;
        // rhs_g = 0.;
        // rhs_x = 0.;
        // rhs_s = 0.;
        // D_eq = 0.;
        // D_x = 0.;

    };
};

TEST_F(PdTest, TestSolveNoItRef)
{
    // Test the raw solve (no iterative refinement) to verify the reduction is correct
    LinearSystem<PdSystemResto<OcpType>> ls(info, jacobian, hessian, D_x, false, D_eq, sl, su, zl,
                                            zu, rhs_x, rhs_s, rhs_g, rhs_cl, rhs_cu);
    VecRealAllocated x_full(ls.m());
    VecRealAllocated rhs_save(ls.m());
    VecRealAllocated tmp(ls.m());
    ls.get_rhs(rhs_save);
    // Use solve_once directly (bypasses iterative refinement)
    LinsolReturnFlag ret = pd_solver_resto.solve_once(ls, x_full);
    EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);
    ls.set_rhs(rhs_save);
    ls.apply_on_right(x_full, 1.0, rhs_save, tmp);
    EXPECT_NEAR(norm_inf(tmp), 0, 1e-6);
}

TEST_F(PdTest, TestSolve)
{
    LinearSystem<PdSystemResto<OcpType>> ls(info, jacobian, hessian, D_x, false, D_eq, sl, su, zl,
                                            zu, rhs_x, rhs_s, rhs_g, rhs_cl, rhs_cu);
    VecRealAllocated x_full(ls.m());
    VecRealAllocated rhs_save(ls.m());
    VecRealAllocated tmp(ls.m());
    ls.get_rhs(rhs_save);
    LinsolReturnFlag ret = pd_solver_resto.solve_in_place(ls);
    ls.get_rhs(x_full);
    EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);
    ls.set_rhs(rhs_save);
    ls.apply_on_right(x_full, 1.0, rhs_save, tmp);
    EXPECT_NEAR(norm_inf(tmp), 0, 1e-6);
}

TEST(PdRestoGlobalParameters, NativeBorderIsPreservedByRestorationReduction)
{
    const ProblemDims dims(
        1, std::vector<Index>{0}, std::vector<Index>{1},
        std::vector<Index>{1}, std::vector<Index>{0}, 1);
    const ProblemInfo info(dims);
    Jacobian<OcpType> jacobian(dims);
    Hessian<OcpType> hessian(dims);
    jacobian.Gg_eqt[0] = 0.0;
    jacobian.Gg_eqt[0](0, 0) = 1.0;
    jacobian.global_parameter_jacobian = 0.0;
    jacobian.global_parameter_jacobian(info.offsets_g_eq_path[0], 0) = 0.5;
    hessian.set_zero();
    hessian.RSQrqt[0](0, 0) = 2.0;
    hessian.global_parameter_cross_hessian(0, 0) = 0.1;
    hessian.global_parameter_hessian(0, 0) = 3.0;

    VecRealAllocated diagonal(
        info.number_of_primal_variables + info.number_of_slack_variables_resto);
    VecRealAllocated equality_diagonal(info.number_of_eq_constraints);
    VecRealAllocated rhs_primal(info.number_of_primal_variables);
    VecRealAllocated rhs_slack(info.number_of_slack_variables_resto);
    VecRealAllocated rhs_constraints(info.number_of_eq_constraints);
    VecRealAllocated rhs_lower(info.number_of_slack_variables_resto);
    VecRealAllocated rhs_upper(info.number_of_slack_variables_resto);
    VecRealAllocated lower_distance(info.number_of_slack_variables_resto);
    VecRealAllocated upper_distance(info.number_of_slack_variables_resto);
    VecRealAllocated lower_dual(info.number_of_slack_variables_resto);
    VecRealAllocated upper_dual(info.number_of_slack_variables_resto);

    diagonal = 0.2;
    equality_diagonal = 0.05;
    rhs_primal(0) = 0.3;
    rhs_primal(1) = -0.2;
    rhs_constraints(0) = 0.4;
    rhs_slack(0) = -0.1;
    rhs_slack(1) = 0.15;
    rhs_lower(0) = 0.07;
    rhs_lower(1) = -0.04;
    rhs_upper = 0.0;
    lower_distance = 1.0;
    upper_distance = 1.0;
    lower_dual = 1.0;
    upper_dual = 0.0;

    auto augmented_solver = std::make_shared<AugSystemSolver<OcpType>>(info);
    auto original_solver =
        std::make_shared<PdSolverOrig<OcpType>>(info, augmented_solver);
    PdSolverResto<OcpType> restoration_solver(info, original_solver);
    LinearSystem<PdSystemResto<OcpType>> system(
        info, jacobian, hessian, diagonal, false, equality_diagonal,
        lower_distance, upper_distance, lower_dual, upper_dual,
        rhs_primal, rhs_slack, rhs_constraints, rhs_lower, rhs_upper);

    VecRealAllocated rhs(system.m());
    VecRealAllocated solution(system.m());
    VecRealAllocated residual(system.m());
    system.get_rhs(rhs);
    ASSERT_EQ(
        restoration_solver.solve_once(system, solution),
        LinsolReturnFlag::SUCCESS);
    system.apply_on_right(solution, 1.0, rhs, residual);
    EXPECT_LT(norm_inf(residual), 1e-10);
}

// TEST_F(PdTest, TestSolveDegen)
// {
//     LinearSystem<PdSystemType<OcpType>> ls(info, jacobian, hessian, D_x, true, D_eq, sl, su, zl,
//     zu,
//                                            rhs_x, rhs_s, rhs_g, rhs_cl, rhs_cu);
//     VecRealAllocated x_full(ls.m());
//     VecRealAllocated rhs_save(ls.m());
//     VecRealAllocated tmp(ls.m());
//     ls.get_rhs(rhs_save);
//     LinsolReturnFlag ret = pd_solver.solve_in_place(ls);
//     ls.get_rhs(x_full);
//     EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);
//     ls.set_rhs(rhs_save);
//     ls.apply_on_right(x_full, 1.0, rhs_save, tmp);
//     EXPECT_NEAR(norm_inf(tmp), 0, 1e-6);
// }

// TEST_F(ImplicitPdTest, TestImplicitSolve)
// {
//     LinearSystem<PdSystemResto<ImplicitOcpType>> ls(info, jacobian, hessian, D_x, false, D_eq, sl, su, zl,
//                                             zu, rhs_x, rhs_s, rhs_g, rhs_cl, rhs_cu);
//     VecRealAllocated x_full(ls.m());
//     VecRealAllocated rhs_save(ls.m());
//     VecRealAllocated tmp(ls.m());
//     ls.get_rhs(rhs_save);
//     LinsolReturnFlag ret = pd_solver_resto.solve_in_place(ls);
//     ls.get_rhs(x_full);
//     EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);
//     ls.set_rhs(rhs_save);
//     ls.apply_on_right(x_full, 1.0, rhs_save, tmp);
//     EXPECT_NEAR(norm_inf(tmp), 0, 1e-6);
// }

// TEST_F(ImplicitPdTest, TestImplicitSolveDegen)
// {
//     LinearSystem<PdSystemType<ImplicitOcpType>> ls(info, jacobian, hessian, D_x, true, D_eq, sl, su, zl,
//     zu,
//                                            rhs_x, rhs_s, rhs_g, rhs_cl, rhs_cu);
//     VecRealAllocated x_full(ls.m());
//     VecRealAllocated rhs_save(ls.m());
//     VecRealAllocated tmp(ls.m());
//     ls.get_rhs(rhs_save);
//     LinsolReturnFlag ret = pd_solver.solve_in_place(ls);
//     ls.get_rhs(x_full);
//     EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);
//     ls.set_rhs(rhs_save);
//     ls.apply_on_right(x_full, 1.0, rhs_save, tmp);
//     EXPECT_NEAR(norm_inf(tmp), 0, 1e-6);
// }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
