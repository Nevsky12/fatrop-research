#include "fatrop/linear_algebra/blasfeo_wrapper.hpp"
#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/parametric_implicit_nlp_ocp.hpp"
#include "fatrop/ocp/problem_info.hpp"

#include <gtest/gtest.h>

#include <memory>

using namespace fatrop;

namespace
{
    class TwoStageParametricProblem final
        : public ParametricImplicitOcpAbstract
    {
    public:
        Index get_nx(Index) const override { return 1; }
        Index get_nu(Index k) const override { return k == 0 ? 1 : 0; }
        Index get_ng(Index k) const override { return k == 0 ? 1 : 0; }
        Index get_ng_ineq(Index k) const override { return k == 0 ? 1 : 0; }
        Index get_np() const override { return 2; }
        Index get_horizon_length() const override { return 2; }

        Index eval_BAJbt(
            const Scalar *, const Scalar *, const Scalar *, const Scalar *,
            MAT *ba, MAT *jt, MAT *jt_inv, Index) override
        {
            blasfeo_gese_wrap(ba->m, ba->n, 0.0, ba, 0, 0);
            blasfeo_gese_wrap(jt->m, jt->n, 0.0, jt, 0, 0);
            blasfeo_gese_wrap(jt_inv->m, jt_inv->n, 0.0, jt_inv, 0, 0);
            blasfeo_matel_wrap(ba, 0, 0) = 2.0;
            blasfeo_matel_wrap(ba, 1, 0) = 3.0;
            blasfeo_matel_wrap(jt, 0, 0) = 4.0;
            blasfeo_matel_wrap(jt_inv, 0, 0) = 0.25;
            return 0;
        }

        Index eval_dynamics_parameter_jacobian(
            const Scalar *, const Scalar *, const Scalar *, const Scalar *,
            MAT *res, Index) override
        {
            blasfeo_matel_wrap(res, 0, 0) = 5.0;
            blasfeo_matel_wrap(res, 0, 1) = 6.0;
            return 0;
        }

        Index eval_RSQrqt(
            const Scalar *scale, const Scalar *, const Scalar *, const Scalar *,
            const Scalar *, const Scalar *, const Scalar *, const Scalar *,
            MAT *res, MAT *, MAT *, Index k) override
        {
            if (k == 0)
            {
                blasfeo_matel_wrap(res, 0, 0) += 2.0 * *scale;
                blasfeo_matel_wrap(res, 1, 1) += 4.0 * *scale;
            }
            else
                blasfeo_matel_wrap(res, 0, 0) += 8.0 * *scale;
            return 0;
        }

        Index eval_parameter_hessian(
            const Scalar *scale, const Scalar *, const Scalar *, const Scalar *,
            const Scalar *, const Scalar *, const Scalar *, const Scalar *,
            MAT *ux_p, MAT *, MAT *pp, Index k) override
        {
            if (k == 0)
            {
                blasfeo_matel_wrap(ux_p, 0, 0) = *scale;
                blasfeo_matel_wrap(ux_p, 1, 1) = *scale;
                blasfeo_matel_wrap(pp, 0, 0) = 2.0 * *scale;
                blasfeo_matel_wrap(pp, 1, 1) = 6.0 * *scale;
            }
            else
            {
                blasfeo_matel_wrap(ux_p, 0, 0) = 2.0 * *scale;
                blasfeo_matel_wrap(pp, 0, 0) = 4.0 * *scale;
                blasfeo_matel_wrap(pp, 0, 1) = *scale;
                blasfeo_matel_wrap(pp, 1, 0) = *scale;
            }
            return 0;
        }

        Index eval_Ggt(
            const Scalar *, const Scalar *, const Scalar *, MAT *res,
            Index) override
        {
            blasfeo_matel_wrap(res, 0, 0) = 7.0;
            blasfeo_matel_wrap(res, 1, 0) = 8.0;
            return 0;
        }

        Index eval_Ggt_ineq(
            const Scalar *, const Scalar *, const Scalar *, MAT *res,
            Index) override
        {
            blasfeo_matel_wrap(res, 0, 0) = 11.0;
            blasfeo_matel_wrap(res, 1, 0) = 12.0;
            return 0;
        }

        Index eval_equality_parameter_jacobian(
            const Scalar *, const Scalar *, const Scalar *, MAT *res,
            Index) override
        {
            blasfeo_matel_wrap(res, 0, 0) = 9.0;
            blasfeo_matel_wrap(res, 0, 1) = 10.0;
            return 0;
        }

        Index eval_inequality_parameter_jacobian(
            const Scalar *, const Scalar *, const Scalar *, MAT *res,
            Index) override
        {
            blasfeo_matel_wrap(res, 0, 0) = 13.0;
            blasfeo_matel_wrap(res, 0, 1) = 14.0;
            return 0;
        }

        Index eval_b(
            const Scalar *xp1, const Scalar *u, const Scalar *x,
            const Scalar *p, Scalar *res, Index) override
        {
            res[0] = 2.0 * u[0] + 3.0 * x[0] + 4.0 * xp1[0]
                   + 5.0 * p[0] + 6.0 * p[1];
            return 0;
        }

        Index eval_g(
            const Scalar *u, const Scalar *x, const Scalar *p,
            Scalar *res, Index) override
        {
            res[0] = 7.0 * u[0] + 8.0 * x[0]
                   + 9.0 * p[0] + 10.0 * p[1];
            return 0;
        }

        Index eval_gineq(
            const Scalar *u, const Scalar *x, const Scalar *p,
            Scalar *res, Index) override
        {
            res[0] = 11.0 * u[0] + 12.0 * x[0]
                   + 13.0 * p[0] + 14.0 * p[1];
            return 0;
        }

        Index eval_rq(
            const Scalar *scale, const Scalar *u, const Scalar *x,
            const Scalar *p, Scalar *res, Index k) override
        {
            if (k == 0)
            {
                res[0] = *scale * (2.0 * u[0] + p[0]);
                res[1] = *scale * (4.0 * x[0] + p[1]);
            }
            else
                res[0] = *scale * (8.0 * x[0] + 2.0 * p[0]);
            return 0;
        }

        Index eval_rp(
            const Scalar *scale, const Scalar *u, const Scalar *x,
            const Scalar *p, Scalar *res, Index k) override
        {
            if (k == 0)
            {
                res[0] = *scale * (2.0 * p[0] + u[0]);
                res[1] = *scale * (6.0 * p[1] + x[0]);
            }
            else
            {
                res[0] = *scale * (4.0 * p[0] + p[1] + 2.0 * x[0]);
                res[1] = *scale * p[0];
            }
            return 0;
        }

        Index eval_L(
            const Scalar *scale, const Scalar *u, const Scalar *x,
            const Scalar *p, Scalar *res, Index k) override
        {
            if (k == 0)
                *res = *scale * (u[0] * u[0] + 2.0 * x[0] * x[0]
                    + p[0] * p[0] + 3.0 * p[1] * p[1]
                    + u[0] * p[0] + x[0] * p[1]);
            else
                *res = *scale * (4.0 * x[0] * x[0]
                    + 2.0 * p[0] * p[0] + p[0] * p[1]
                    + 2.0 * x[0] * p[0]);
            return 0;
        }

        Index get_bounds(Scalar *lower, Scalar *upper, Index) const override
        {
            lower[0] = -3.0;
            upper[0] = 3.0;
            return 0;
        }

        bool has_stage_variable_bounds(Index) const override
        {
            return true;
        }

        Index get_stage_variable_bounds(
            Scalar *lower, Scalar *upper, Index k) const override
        {
            if (k == 0)
            {
                lower[0] = -0.5;
                upper[0] = 0.5;
                lower[1] = -0.75;
                upper[1] = 0.75;
            }
            else
            {
                lower[0] = -1.25;
                upper[0] = 1.25;
            }
            return 0;
        }

        Index get_initial_xk(Scalar *x, Index k) const override
        {
            x[0] = k == 0 ? 0.2 : 0.3;
            return 0;
        }

        Index get_initial_uk(Scalar *u, Index k) const override
        {
            if (k == 0)
                u[0] = 0.1;
            return 0;
        }

        Index get_initial_parameters(Scalar *p) const override
        {
            p[0] = 0.4;
            p[1] = 0.5;
            return 0;
        }

        bool has_parameter_bounds() const override { return true; }

        Index get_parameter_bounds(
            Scalar *lower, Scalar *upper) const override
        {
            lower[0] = -1.0;
            lower[1] = -2.0;
            upper[0] = 1.0;
            upper[1] = 2.0;
            return 0;
        }
    };

    struct ParametricAdapterFixture : ::testing::Test
    {
        std::shared_ptr<TwoStageParametricProblem> problem =
            std::make_shared<TwoStageParametricProblem>();
        ParametricImplicitNlpOcp nlp{problem};
        ProblemInfo info{nlp.problem_dims()};
        VecRealAllocated primal{info.number_of_primal_variables};
        VecRealAllocated slack{info.number_of_slack_variables};
        VecRealAllocated multipliers{info.number_of_eq_constraints};

        void SetUp() override
        {
            nlp.get_initial_primal(info, primal);
            slack = 0.0;
            multipliers = 0.0;
        }
    };
}

TEST_F(ParametricAdapterFixture, AppendsOneCopyParametersAndParameterBounds)
{
    EXPECT_EQ(info.number_of_trajectory_variables, 3);
    EXPECT_EQ(info.number_of_global_parameters, 2);
    EXPECT_EQ(info.number_of_primal_variables, 5);
    EXPECT_EQ(info.number_of_slack_variables, 6);
    EXPECT_EQ(info.number_of_eq_constraints, 8);
    EXPECT_EQ(nlp.nlp_dims().number_of_variables, 5);
    EXPECT_EQ(primal(0), 0.1);
    EXPECT_EQ(primal(1), 0.2);
    EXPECT_EQ(primal(2), 0.3);
    EXPECT_EQ(primal(3), 0.4);
    EXPECT_EQ(primal(4), 0.5);

    VecRealAllocated lower(info.number_of_slack_variables);
    VecRealAllocated upper(info.number_of_slack_variables);
    ASSERT_EQ(nlp.get_bounds(info, lower, upper), 0);
    EXPECT_EQ(lower(0), -3.0);
    EXPECT_EQ(upper(0), 3.0);
    EXPECT_EQ(lower(1), -0.5);
    EXPECT_EQ(upper(1), 0.5);
    EXPECT_EQ(lower(2), -0.75);
    EXPECT_EQ(upper(2), 0.75);
    EXPECT_EQ(lower(3), -1.25);
    EXPECT_EQ(upper(3), 1.25);
    EXPECT_EQ(lower(4), -1.0);
    EXPECT_EQ(lower(5), -2.0);
    EXPECT_EQ(upper(4), 1.0);
    EXPECT_EQ(upper(5), 2.0);
}

TEST_F(ParametricAdapterFixture, AssemblesValuesAndGlobalJacobians)
{
    Jacobian<ImplicitOcpType> jac(nlp.problem_dims());
    ASSERT_EQ(nlp.eval_constr_jac(info, primal, slack, jac), 0);

    EXPECT_EQ(jac.global_parameter_jacobian(info.offsets_g_eq_dyn[0], 0), 5.0);
    EXPECT_EQ(jac.global_parameter_jacobian(info.offsets_g_eq_dyn[0], 1), 6.0);
    EXPECT_EQ(jac.global_parameter_jacobian(info.offsets_g_eq_path[0], 0), 9.0);
    EXPECT_EQ(jac.global_parameter_jacobian(info.offsets_g_eq_path[0], 1), 10.0);
    EXPECT_EQ(jac.global_parameter_jacobian(info.offsets_g_eq_slack[0], 0), 13.0);
    EXPECT_EQ(jac.global_parameter_jacobian(info.offsets_g_eq_slack[0], 1), 14.0);
    EXPECT_EQ(jac.Gg_ineqt[0](0, 1), 1.0);
    EXPECT_EQ(jac.Gg_ineqt[0](1, 2), 1.0);
    EXPECT_EQ(jac.Gg_ineqt[1](0, 0), 1.0);
    const Index parameter_bound_offset = info.offsets_g_eq_slack[1] + 1;
    EXPECT_EQ(jac.global_parameter_jacobian(parameter_bound_offset, 0), 1.0);
    EXPECT_EQ(jac.global_parameter_jacobian(parameter_bound_offset + 1, 1), 1.0);

    VecRealAllocated residual(info.number_of_eq_constraints);
    ASSERT_EQ(nlp.eval_constraint_violation(info, primal, slack, residual), 0);
    EXPECT_NEAR(residual(info.offsets_g_eq_path[0]), 10.9, 1e-14);
    EXPECT_NEAR(residual(info.offsets_g_eq_dyn[0]), 7.0, 1e-14);
    EXPECT_NEAR(residual(info.offsets_g_eq_slack[0]), 15.7, 1e-14);
    EXPECT_NEAR(residual(info.offsets_g_eq_slack[0] + 1), 0.1, 1e-14);
    EXPECT_NEAR(residual(info.offsets_g_eq_slack[0] + 2), 0.2, 1e-14);
    EXPECT_NEAR(residual(info.offsets_g_eq_slack[1]), 0.3, 1e-14);
    EXPECT_NEAR(residual(parameter_bound_offset), 0.4, 1e-14);
    EXPECT_NEAR(residual(parameter_bound_offset + 1), 0.5, 1e-14);
}

TEST_F(ParametricAdapterFixture, AssemblesObjectiveGradientAndHessianBorder)
{
    VecRealAllocated gradient(info.number_of_primal_variables);
    VecRealAllocated slack_gradient(info.number_of_slack_variables);
    ASSERT_EQ(nlp.eval_objective_gradient(
        info, 1.0, primal, slack, gradient, slack_gradient), 0);
    EXPECT_NEAR(gradient(0), 0.6, 1e-14);
    EXPECT_NEAR(gradient(1), 1.3, 1e-14);
    EXPECT_NEAR(gradient(2), 3.2, 1e-14);
    EXPECT_NEAR(gradient(3), 3.6, 1e-14);
    EXPECT_NEAR(gradient(4), 3.6, 1e-14);

    Scalar objective = 0.0;
    ASSERT_EQ(nlp.eval_objective(info, 1.0, primal, slack, objective), 0);
    EXPECT_NEAR(objective, 2.26, 1e-14);

    Hessian<ImplicitOcpType> hessian(nlp.problem_dims());
    ASSERT_EQ(nlp.eval_lag_hess(
        info, 1.0, primal, slack, multipliers, hessian), 0);
    EXPECT_EQ(hessian.global_parameter_cross_hessian(0, 0), 1.0);
    EXPECT_EQ(hessian.global_parameter_cross_hessian(1, 1), 1.0);
    EXPECT_EQ(hessian.global_parameter_cross_hessian(2, 0), 2.0);
    EXPECT_EQ(hessian.global_parameter_hessian(0, 0), 6.0);
    EXPECT_EQ(hessian.global_parameter_hessian(0, 1), 1.0);
    EXPECT_EQ(hessian.global_parameter_hessian(1, 0), 1.0);
    EXPECT_EQ(hessian.global_parameter_hessian(1, 1), 6.0);
}
