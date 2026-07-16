//
// Copyright (c) Lander Vanroye, KU Leuven
//
#include "../random_matrix.hpp"
#include "fatrop/context/context.hpp"
#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/ocp/dims.hpp"
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/problem_info.hpp"
#include "fatrop/ocp/type.hpp"
#include <gtest/gtest.h>
#include <vector>

using namespace fatrop;

MatRealAllocated get_inverse(const MatRealView &A)
{
    fatrop_dbg_assert(A.m() == A.n() && "Matrix must be square for inversion");
    MatRealAllocated A_inv(A.m(), A.m());
    MatRealAllocated LU(A.m(), A.m());
    blasfeo_dgetrf_np(A.m(), A.m(), const_cast<MAT *>(&A.mat()), 0, 0, &LU.mat(), 0, 0);

    // Solve the system LU * X = I, where I is the identity matrix
    MatRealAllocated I = ::test::identity_matrix(A.m());

    // (1) solve L Y = I
    blasfeo_dtrsm_llnu(A.m(), A.m(), 1.0, &LU.mat(), 0, 0, &I.mat(), 0, 0, &A_inv.mat(), 0, 0);
    // (2) solve U X = Y
    blasfeo_dtrsm_lunn(A.m(), A.m(), 1.0, &LU.mat(), 0, 0, &A_inv.mat(), 0, 0, &A_inv.mat(), 0, 0);

    // std::cout << "Inverse of A: \n" << A << "\nis given by \n"
    //           << A_inv << std::endl;

    // check if A_inv contains any NaN or Inf values
    for (Index i = 0; i < A_inv.m(); ++i)
    {
        for (Index j = 0; j < A_inv.n(); ++j)
        {
            if (std::isnan(A_inv(i, j)) || std::isinf(A_inv(i, j)))
            {
                throw std::runtime_error("Inverse contains NaN or Inf values");
            }
        }
    }

    // check result
    MatRealAllocated I_check = ::test::identity_matrix(A.m());
    blasfeo_dgemm_nn(A.m(), A.m(), A.m(), 1.0, 
                     const_cast<MAT *>(&A.mat()), 0, 0, 
                     const_cast<MAT *>(&A_inv.mat()), 0, 0, 0.0, 
                     const_cast<MAT *>(&I_check.mat()), 0, 0,
                     const_cast<MAT *>(&I_check.mat()), 0, 0);
    // std::cout << "identity check: \n"
    //           << I_check << std::endl;

    return A_inv;
}

TEST(JacobianTest, ConstructorTest)
{
    // Create OcpDims object
    int K = 5;                                       // Number of stages
    std::vector<Index> nx = {2, 2, 2, 2, 2};      // State dimensions for each stage
    std::vector<Index> nu = {1, 1, 1, 1, 1};         // Input dimensions for each stage
    std::vector<Index> ng = {1, 0, 0, 0, 2};      // Equality constraints for each stage
    std::vector<Index> ng_ineq = {1, 0, 2, 0, 0}; // Inequality constraints for each stage

    ProblemDims dims(K, nx, nu, ng, ng_ineq);

    // Check if Jacobian object can be constructed without throwing an exception
    EXPECT_NO_THROW({ Jacobian<OcpType> jacobian(dims); });
}

TEST(HessianTest, ConstructorTest)
{
    // Create OcpDims object
    int K = 5;                                       // Number of stages
    std::vector<Index> nx = {2, 2, 2, 2, 2};      // State dimensions for each stage
    std::vector<Index> nu = {1, 1, 1, 1, 1};         // Input dimensions for each stage
    std::vector<Index> ng = {1, 0, 0, 0, 2};      // Equality constraints for each stage
    std::vector<Index> ng_ineq = {1, 0, 2, 0, 0}; // Inequality constraints for each stage

    ProblemDims dims(K, nx, nu, ng, ng_ineq);

    // Check if Hessian object can be constructed without throwing an exception
    EXPECT_NO_THROW({ Hessian<OcpType> hessian(dims); });
}

// TEST(JacobianTest, AssertionViolationTest) {
//     // Create OcpDims object with invalid dimensions
//     int K = 3;  // Number of stages
//     std::vector<Index> nx = {2, 2, 2, 2};  // State dimensions for each stage
//     std::vector<Index> nu = {1, 1, 1};     // Input dimensions for each stage
//     std::vector<Index> ng = {4, 3, 3, 2};  // Equality constraints for each stage (violates
//     assertion) std::vector<Index> ng_ineq = {0, 0, 0, 0};  // Inequality constraints for each
//     stage

//     OcpDims dims(K, nx, nu, ng, ng_ineq);

//     // Check if Jacobian constructor throws an exception due to assertion violation
//     EXPECT_ANY_THROW({
//         Jacobian<OcpType> jacobian(dims);
//     });  // Assuming fatrop_assert throws a std::runtime_error
// }

class JacobianTestOperations : public ::testing::Test
{
protected:
    // Create OcpDims object
    int K = 5;                                    // Number of stages
    std::vector<Index> nx = {2, 2, 2, 2, 2};      // State dimensions for each stage
    std::vector<Index> nu = {1, 1, 4, 1, 0};      // Input dimensions for each stage
    std::vector<Index> ng = {1, 0, 5, 0, 2};      // Equality constraints for each stage
    std::vector<Index> ng_ineq = {1, 0, 3, 0, 0}; // Inequality constraints for each stage

    ProblemDims dims{K, nu, nx, ng, ng_ineq};

    ProblemInfo info{dims};
    // Create Jacobian object
    Jacobian<OcpType> jacobian{dims};
    MatRealAllocated full_matrix =
        MatRealAllocated(info.number_of_eq_constraints, info.number_of_primal_variables);
    VecRealAllocated x = VecRealAllocated(info.number_of_primal_variables);
    VecRealAllocated mult = VecRealAllocated(info.number_of_eq_constraints);
    void SetUp()
    {
        // fill the jacobian with random values
        for (Index k = 0; k < info.dims.K - 1; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index nx_next = info.dims.number_of_states[k + 1];
            jacobian.BAbt[k] = 1.0 * (k + 1);
            jacobian.BAbt[k].diagonal() = 2.0 * (k + 1);
            jacobian.Gg_eqt[k] = 3.0 * (k + 1);
            jacobian.Gg_eqt[k].diagonal() = 4.0 * (k + 1);
            jacobian.Gg_ineqt[k] = 5.0 * (k + 1);
            jacobian.Gg_ineqt[k].diagonal() = 6.0 * (k + 1);
        }
        // fill the x vector with random values
        for (Index i = 0; i < info.number_of_primal_variables; ++i)
        {
            x(i) = 1.0 * i;
        }
        // fill the mult vector with random values
        for (Index i = 0; i < info.number_of_eq_constraints; ++i)
        {
            mult(i) = 1.0 * i;
        }
        // add dynamics constraints
        for (Index k = 0; k < info.dims.K - 1; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index offs_ux = info.offsets_primal_u[k];
            Index offs_x_next = info.offsets_primal_x[k + 1];
            Index nx_next = info.dims.number_of_states[k + 1];
            Index offs_eq_dyn = info.offsets_g_eq_dyn[k];
            full_matrix.block(nx_next, nu + nx, offs_eq_dyn, offs_ux) =
                transpose(jacobian.BAbt[k].block(nu + nx, nx_next, 0, 0));
            full_matrix.block(nx_next, nx_next, offs_eq_dyn, offs_x_next).diagonal() = -1.0;
        }
        // equality path equality constraints
        for (Index k = 0; k < info.dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index ng = info.dims.number_of_eq_constraints[k];
            Index offset_ux = info.offsets_primal_u[k];
            Index offset_g_eq = info.offsets_g_eq_path[k];
            full_matrix.block(ng, nu + nx, offset_g_eq, offset_ux) =
                transpose(jacobian.Gg_eqt[k].block(nu + nx, ng, 0, 0));
        }
        // inequality path constraints
        for (Index k = 0; k < info.dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index ng_ineq = info.dims.number_of_ineq_constraints[k];
            Index offset_ux = info.offsets_primal_u[k];
            Index offset_g_ineq = info.offsets_g_eq_slack[k];
            full_matrix.block(ng_ineq, nu + nx, offset_g_ineq, offset_ux) =
                transpose(jacobian.Gg_ineqt[k].block(nu + nx, ng_ineq, 0, 0));
        }
    };
};

TEST_F(JacobianTestOperations, TestInOut)
{
    // put mult as rhs
    jacobian.set_rhs(info, mult);
    VecRealAllocated out = VecRealAllocated(info.number_of_eq_constraints);
    // get mult back from rhs -> out
    jacobian.get_rhs(info, out);
    for (Index i = 0; i < info.number_of_eq_constraints; ++i)
    {
        EXPECT_NEAR(out(i), mult(i), 1e-10);
    }
}
// check Jacobian if full matrix @ x == jacobian.apply_on_right(x)
TEST_F(JacobianTestOperations, ApplyOnRightTestt)
{
    VecRealAllocated out = VecRealAllocated(info.number_of_eq_constraints);
    jacobian.apply_on_right(info, x, 0.0, out, out);
    VecRealAllocated full_matrix_x(info.number_of_eq_constraints);
    gemv_n(info.number_of_eq_constraints, info.number_of_primal_variables, 1.0, full_matrix, 0, 0,
           x, 0, 0.0, full_matrix_x, 0, full_matrix_x, 0);
    for (Index i = 0; i < info.number_of_eq_constraints; ++i)
    {
        EXPECT_NEAR(out(i), full_matrix_x(i), 1e-10);
    }
}
// check Jacobian if full matrix.T @ mult == jacobian.transpose_apply_on_right(mult)
TEST_F(JacobianTestOperations, TransposeApplyOnRightTest)
{
    VecRealAllocated out = VecRealAllocated(info.number_of_primal_variables);
    jacobian.transpose_apply_on_right(info, mult, 0.0, out, out);
    VecRealAllocated full_matrix_t_mult(info.number_of_primal_variables);
    gemv_t(info.number_of_eq_constraints, info.number_of_primal_variables, 1.0, full_matrix, 0, 0,
           mult, 0, 0.0, full_matrix_t_mult, 0, full_matrix_t_mult, 0);
    for (Index i = 0; i < info.number_of_primal_variables; ++i)
    {
        EXPECT_NEAR(out(i), full_matrix_t_mult(i), 1e-10);
    }
}

class HessianTestOperations : public ::testing::Test
{
protected:
    // Create OcpDims object
    int K = 5;                                    // Number of stages
    std::vector<Index> nx = {2, 2, 2, 2, 2};      // State dimensions for each stage
    std::vector<Index> nu = {1, 1, 4, 1, 0};      // Input dimensions for each stage
    std::vector<Index> ng = {1, 0, 5, 0, 2};      // Equality constraints for each stage
    std::vector<Index> ng_ineq = {1, 0, 3, 0, 0}; // Inequality constraints for each stage

    ProblemDims dims{K, nu, nx, ng, ng_ineq};

    ProblemInfo info{dims};
    // Create Hessian object
    Hessian<OcpType> hessian{dims};
    MatRealAllocated full_matrix =
        MatRealAllocated(info.number_of_primal_variables, info.number_of_primal_variables);
    VecRealAllocated x = VecRealAllocated(info.number_of_primal_variables);
    void SetUp()
    {
        full_matrix = 0.;
        // fill the Hessian with random values
        for (Index k = 0; k < dims.K; ++k)
        {
            hessian.RSQrqt[k].diagonal() = 2.0 * (k + 1);
            hessian.RSQrqt[k](0, 1) = 1e-2;
            hessian.RSQrqt[k](1, 0) = 1e-2;
        }
        // fill the x vector with random values
        for (Index i = 0; i < info.number_of_primal_variables; ++i)
        {
            x(i) = 1.0 * i;
        }
        // populate the full matrix
        for (Index k = 0; k < dims.K; k++)
        {
            Index nu = dims.number_of_controls[k];
            Index nx = dims.number_of_states[k];
            Index offs_ux = info.offsets_primal_u[k];
            full_matrix.block(nu + nx, nu + nx, offs_ux, offs_ux) =
                hessian.RSQrqt[k].block(nu + nx, nu + nx, 0, 0);
        }
    };
};
TEST_F(HessianTestOperations, TestInOut)
{
    // put x as rhs
    hessian.set_rhs(info, x);
    VecRealAllocated out = VecRealAllocated(info.number_of_eq_constraints);
    // get x back from rhs -> out
    hessian.get_rhs(info, out);
    for (Index i = 0; i < info.number_of_primal_variables; ++i)
    {
        EXPECT_NEAR(out(i), x(i), 1e-10);
    }
}
// check Hessian if full matrix @ x == hessian.apply_on_right(x)
TEST_F(HessianTestOperations, ApplyOnRightTest)
{
    VecRealAllocated out = VecRealAllocated(info.number_of_primal_variables);
    hessian.apply_on_right(info, x, 0.0, out, out);
    VecRealAllocated full_matrix_x(info.number_of_primal_variables);
    gemv_n(info.number_of_primal_variables, info.number_of_primal_variables, 1.0, full_matrix, 0, 0,
           x, 0, 0.0, full_matrix_x, 0, full_matrix_x, 0);
    for (Index i = 0; i < info.number_of_primal_variables; ++i)
    {
        EXPECT_NEAR(out(i), full_matrix_x(i), 1e-10);
    }
}









/////////////////////////
/// IMPLICIT OCP CODE ///
/////////////////////////
TEST(ImplicitJacobianTest, ImplicitConstructorTest)
{
    // Create OcpDims object
    int K = 5;                                       // Number of stages
    std::vector<Index> nx = {2, 2, 2, 2, 2};      // State dimensions for each stage
    std::vector<Index> nu = {1, 1, 1, 1, 1};         // Input dimensions for each stage
    std::vector<Index> ng = {1, 0, 0, 0, 2};      // Equality constraints for each stage
    std::vector<Index> ng_ineq = {1, 0, 2, 0, 0}; // Inequality constraints for each stage

    ProblemDims dims(K, nx, nu, ng, ng_ineq);

    // Check if Jacobian object can be constructed without throwing an exception
    EXPECT_NO_THROW({ Jacobian<ImplicitOcpType> jacobian(dims); });
}

TEST(ImplicitHessianTest, ImplicitConstructorTest)
{
    // Create OcpDims object
    int K = 5;                                       // Number of stages
    std::vector<Index> nx = {2, 2, 2, 2, 2};      // State dimensions for each stage
    std::vector<Index> nu = {1, 1, 1, 1, 1};         // Input dimensions for each stage
    std::vector<Index> ng = {1, 0, 0, 0, 2};      // Equality constraints for each stage
    std::vector<Index> ng_ineq = {1, 0, 2, 0, 0}; // Inequality constraints for each stage

    ProblemDims dims(K, nx, nu, ng, ng_ineq);

    // Check if Hessian object can be constructed without throwing an exception
    EXPECT_NO_THROW({ Hessian<ImplicitOcpType> hessian(dims); });
}

// TEST(ImplicitJacobianTest, ImplicitAssertionViolationTest) {
//     // Create OcpDims object with invalid dimensions
//     int K = 3;  // Number of stages
//     std::vector<Index> nx = {2, 2, 2, 2};  // State dimensions for each stage
//     std::vector<Index> nu = {1, 1, 1};     // Input dimensions for each stage
//     std::vector<Index> ng = {4, 3, 3, 2};  // Equality constraints for each stage (violates
//     assertion) std::vector<Index> ng_ineq = {0, 0, 0, 0};  // Inequality constraints for each
//     stage

//     OcpDims dims(K, nx, nu, ng, ng_ineq);

//     // Check if Jacobian constructor throws an exception due to assertion violation
//     EXPECT_ANY_THROW({
//         Jacobian<ImplicitOcpType> jacobian(dims);
//     });  // Assuming fatrop_assert throws a std::runtime_error
// }

class ImplicitJacobianTestOperations : public ::testing::Test
{
protected:
    // Create OcpDims object
    int K = 5;                                    // Number of stages
    std::vector<Index> nx = {2, 2, 2, 2, 2};      // State dimensions for each stage
    std::vector<Index> nu = {1, 1, 4, 1, 0};      // Input dimensions for each stage
    std::vector<Index> ng = {1, 0, 5, 0, 2};      // Equality constraints for each stage
    std::vector<Index> ng_ineq = {1, 0, 3, 0, 0}; // Inequality constraints for each stage

    ProblemDims dims{K, nu, nx, ng, ng_ineq};

    ProblemInfo info{dims};
    // Create Jacobian object
    Jacobian<ImplicitOcpType> jacobian{dims};
    MatRealAllocated full_matrix =
        MatRealAllocated(info.number_of_eq_constraints, info.number_of_primal_variables);
    VecRealAllocated x = VecRealAllocated(info.number_of_primal_variables);
    VecRealAllocated mult = VecRealAllocated(info.number_of_eq_constraints);
    void SetUp()
    {
        // fill the jacobian with random values
        for (Index k = 0; k < info.dims.K - 1; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index nx_next = info.dims.number_of_states[k + 1];
            jacobian.BAbt[k] = 1.0 * (k + 1);
            jacobian.BAbt[k].diagonal() = 2.0 * (k + 1);
            jacobian.Gg_eqt[k] = 3.0 * (k + 1);
            jacobian.Gg_eqt[k].diagonal() = 4.0 * (k + 1);
            jacobian.Gg_ineqt[k] = 5.0 * (k + 1);
            jacobian.Gg_ineqt[k].diagonal() = 6.0 * (k + 1);
            
            jacobian.Jt[k] = 7.0 * (k + 1);
            jacobian.Jt[k].diagonal() = 8.0 * (k + 1);
            jacobian.Jt_inv[k] = get_inverse(jacobian.Jt[k]);
        }
        // fill the x vector with random values
        for (Index i = 0; i < info.number_of_primal_variables; ++i)
        {
            x(i) = 1.0 * i;
        }
        // fill the mult vector with random values
        for (Index i = 0; i < info.number_of_eq_constraints; ++i)
        {
            mult(i) = 1.0 * i;
        }
        // add dynamics constraints
        for (Index k = 0; k < info.dims.K - 1; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index offs_ux = info.offsets_primal_u[k];
            Index offs_x_next = info.offsets_primal_x[k + 1];
            Index nx_next = info.dims.number_of_states[k + 1];
            Index offs_eq_dyn = info.offsets_g_eq_dyn[k];
            full_matrix.block(nx_next, nu + nx, offs_eq_dyn, offs_ux) =
                transpose(jacobian.BAbt[k].block(nu + nx, nx_next, 0, 0));
            // full_matrix.block(nx_next, nx_next, offs_eq_dyn, offs_x_next).diagonal() = -1.0;
            full_matrix.block(nx_next, nx_next, offs_eq_dyn, offs_x_next) = transpose(jacobian.Jt[k]);
        }
        // equality path equality constraints
        for (Index k = 0; k < info.dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index ng = info.dims.number_of_eq_constraints[k];
            Index offset_ux = info.offsets_primal_u[k];
            Index offset_g_eq = info.offsets_g_eq_path[k];
            full_matrix.block(ng, nu + nx, offset_g_eq, offset_ux) =
                transpose(jacobian.Gg_eqt[k].block(nu + nx, ng, 0, 0));
        }
        // inequality path constraints
        for (Index k = 0; k < info.dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index ng_ineq = info.dims.number_of_ineq_constraints[k];
            Index offset_ux = info.offsets_primal_u[k];
            Index offset_g_ineq = info.offsets_g_eq_slack[k];
            full_matrix.block(ng_ineq, nu + nx, offset_g_ineq, offset_ux) =
                transpose(jacobian.Gg_ineqt[k].block(nu + nx, ng_ineq, 0, 0));
        }
    };
};

TEST_F(ImplicitJacobianTestOperations, ImplicitTestInOut)
{
    // put mult as rhs
    jacobian.set_rhs(info, mult);
    VecRealAllocated out = VecRealAllocated(info.number_of_eq_constraints);
    // get mult back from rhs -> out
    jacobian.get_rhs(info, out);
    for (Index i = 0; i < info.number_of_eq_constraints; ++i)
    {
        EXPECT_NEAR(out(i), mult(i), 1e-10);
    }
}
// check Jacobian if full matrix @ x == jacobian.apply_on_right(x)
TEST_F(ImplicitJacobianTestOperations, ImplicitApplyOnRightTestt)
{
    VecRealAllocated out = VecRealAllocated(info.number_of_eq_constraints);
    jacobian.apply_on_right(info, x, 0.0, out, out);
    VecRealAllocated full_matrix_x(info.number_of_eq_constraints);
    gemv_n(info.number_of_eq_constraints, info.number_of_primal_variables, 1.0, full_matrix, 0, 0,
           x, 0, 0.0, full_matrix_x, 0, full_matrix_x, 0);
    for (Index i = 0; i < info.number_of_eq_constraints; ++i)
    {
        EXPECT_NEAR(out(i), full_matrix_x(i), 1e-10);
    }
}
// check Jacobian if full matrix.T @ mult == jacobian.transpose_apply_on_right(mult)
TEST_F(ImplicitJacobianTestOperations, ImplicitTransposeApplyOnRightTest)
{
    VecRealAllocated out = VecRealAllocated(info.number_of_primal_variables);
    jacobian.transpose_apply_on_right(info, mult, 0.0, out, out);
    VecRealAllocated full_matrix_t_mult(info.number_of_primal_variables);
    gemv_t(info.number_of_eq_constraints, info.number_of_primal_variables, 1.0, full_matrix, 0, 0,
           mult, 0, 0.0, full_matrix_t_mult, 0, full_matrix_t_mult, 0);
    for (Index i = 0; i < info.number_of_primal_variables; ++i)
    {
        EXPECT_NEAR(out(i), full_matrix_t_mult(i), 1e-10);
    }
}

class ImplicitHessianTestOperations : public ::testing::Test
{
protected:
    // Create OcpDims object
    int K = 5;                                    // Number of stages
    std::vector<Index> nx = {2, 2, 2, 2, 2};      // State dimensions for each stage
    std::vector<Index> nu = {1, 1, 4, 1, 0};      // Input dimensions for each stage
    std::vector<Index> ng = {1, 0, 5, 0, 2};      // Equality constraints for each stage
    std::vector<Index> ng_ineq = {1, 0, 3, 0, 0}; // Inequality constraints for each stage

    ProblemDims dims{K, nu, nx, ng, ng_ineq};

    ProblemInfo info{dims};
    // Create Hessian object
    Hessian<ImplicitOcpType> hessian{dims};
    MatRealAllocated full_matrix =
        MatRealAllocated(info.number_of_primal_variables, info.number_of_primal_variables);
    VecRealAllocated x = VecRealAllocated(info.number_of_primal_variables);
    void SetUp()
    {
        full_matrix = 0.;
        // fill the Hessian with random values
        for (Index k = 0; k < dims.K-1; ++k)
        {
            hessian.RSQrqt[k].diagonal() = 2.0 * (k + 1);
            hessian.RSQrqt[k](0, 1) = 1e-2;
            hessian.RSQrqt[k](1, 0) = 1e-2;
            hessian.FuFx[k] = ::test::random_matrix(
                dims.number_of_states[k + 1],
                dims.number_of_states[k] + dims.number_of_controls[k]);
        }
        hessian.RSQrqt[dims.K-1].diagonal() = 2.0 * (dims.K);
            hessian.RSQrqt[dims.K-1](0, 1) = 1e-2;
            hessian.RSQrqt[dims.K-1](1, 0) = 1e-2;

        // fill the x vector with random values
        for (Index i = 0; i < info.number_of_primal_variables; ++i)
        {
            x(i) = 1.0 * i;
        }
        // populate the full matrix
        for (Index k = 0; k < dims.K - 1; k++)
        {
            Index nu = dims.number_of_controls[k];
            Index nx = dims.number_of_states[k];
            Index nx_next = dims.number_of_states[k + 1];
            Index offs_ux = info.offsets_primal_u[k];
            Index offs_x_next = info.offsets_primal_x[k + 1];
            full_matrix.block(nu + nx, nu + nx, offs_ux, offs_ux) =
                hessian.RSQrqt[k].block(nu + nx, nu + nx, 0, 0);
            full_matrix.block(nx_next, nu + nx, offs_x_next, offs_ux) =
                hessian.FuFx[k];
            full_matrix.block(nu + nx, nx_next, offs_ux, offs_x_next) =
                transpose(hessian.FuFx[k]);
        }
        Index nu = dims.number_of_controls[dims.K - 1];
        Index nx = dims.number_of_states[dims.K - 1];
        Index offs_ux = info.offsets_primal_u[dims.K - 1];
        full_matrix.block(nu + nx, nu + nx, offs_ux, offs_ux) =
            hessian.RSQrqt[dims.K - 1].block(nu + nx, nu + nx, 0, 0);
    };
};
TEST_F(ImplicitHessianTestOperations, ImplicitTestInOut)
{
    // put x as rhs
    hessian.set_rhs(info, x);
    VecRealAllocated out = VecRealAllocated(info.number_of_eq_constraints);
    // get x back from rhs -> out
    hessian.get_rhs(info, out);
    for (Index i = 0; i < info.number_of_primal_variables; ++i)
    {
        EXPECT_NEAR(out(i), x(i), 1e-10);
    }
}
// check Hessian if full matrix @ x == hessian.apply_on_right(x)
TEST_F(ImplicitHessianTestOperations, ImplicitApplyOnRightTest)
{
    VecRealAllocated out = VecRealAllocated(info.number_of_primal_variables);
    hessian.apply_on_right(info, x, 0.0, out, out);
    VecRealAllocated full_matrix_x(info.number_of_primal_variables);
    gemv_n(info.number_of_primal_variables, info.number_of_primal_variables, 1.0, full_matrix, 0, 0,
           x, 0, 0.0, full_matrix_x, 0, full_matrix_x, 0);
    for (Index i = 0; i < info.number_of_primal_variables; ++i)
    {
        EXPECT_NEAR(out(i), full_matrix_x(i), 1e-10);
    }
}
