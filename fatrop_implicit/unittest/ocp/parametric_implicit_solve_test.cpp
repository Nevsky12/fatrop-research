#include "fatrop/common/options.hpp"
#include "fatrop/ip_algorithm/ip_alg_builder.hpp"
#include "fatrop/ip_algorithm/ip_algorithm.hpp"
#include "fatrop/linear_algebra/blasfeo_wrapper.hpp"
#include "fatrop/ocp/parametric_implicit_nlp_ocp.hpp"
#include "fatrop/ocp/problem_info.hpp"

#include <gtest/gtest.h>

#include <memory>

using namespace fatrop;

namespace
{
    class SharedParameterTransferProblem final
        : public ParametricImplicitOcpAbstract
    {
    public:
        Index get_nx(Index) const override { return 1; }
        Index get_nu(Index stage) const override { return stage == 0 ? 1 : 0; }
        Index get_ng(Index) const override { return 1; }
        Index get_ng_ineq(Index) const override { return 0; }
        Index get_np() const override { return 1; }
        Index get_horizon_length() const override { return 2; }

        Index eval_BAJbt(
            const Scalar *, const Scalar *, const Scalar *, const Scalar *,
            MAT *ba, MAT *jt, MAT *jt_inv, Index) override
        {
            blasfeo_gese_wrap(ba->m, ba->n, 0.0, ba, 0, 0);
            blasfeo_gese_wrap(jt->m, jt->n, 0.0, jt, 0, 0);
            blasfeo_gese_wrap(jt_inv->m, jt_inv->n, 0.0, jt_inv, 0, 0);
            // x_1 - x_0 - u_0 - p = 0
            blasfeo_matel_wrap(ba, 0, 0) = -1.0;
            blasfeo_matel_wrap(ba, 1, 0) = -1.0;
            blasfeo_matel_wrap(jt, 0, 0) = 1.0;
            blasfeo_matel_wrap(jt_inv, 0, 0) = 1.0;
            return 0;
        }

        Index eval_dynamics_parameter_jacobian(
            const Scalar *, const Scalar *, const Scalar *, const Scalar *,
            MAT *res, Index) override
        {
            blasfeo_matel_wrap(res, 0, 0) = -1.0;
            return 0;
        }

        Index eval_RSQrqt(
            const Scalar *scale, const Scalar *, const Scalar *, const Scalar *,
            const Scalar *, const Scalar *, const Scalar *, const Scalar *,
            MAT *res, MAT *, MAT *, Index stage) override
        {
            if (stage == 0)
                blasfeo_matel_wrap(res, 0, 0) += *scale;
            return 0;
        }

        Index eval_parameter_hessian(
            const Scalar *scale, const Scalar *, const Scalar *, const Scalar *,
            const Scalar *, const Scalar *, const Scalar *, const Scalar *,
            MAT *, MAT *, MAT *pp, Index stage) override
        {
            if (stage == 0)
                blasfeo_matel_wrap(pp, 0, 0) = *scale;
            return 0;
        }

        Index eval_Ggt(
            const Scalar *, const Scalar *, const Scalar *, MAT *res,
            Index stage) override
        {
            if (stage == 0)
                blasfeo_matel_wrap(res, 1, 0) = 1.0;
            else
                blasfeo_matel_wrap(res, 0, 0) = 1.0;
            return 0;
        }

        Index eval_Ggt_ineq(
            const Scalar *, const Scalar *, const Scalar *, MAT *, Index) override
        {
            return 0;
        }

        Index eval_equality_parameter_jacobian(
            const Scalar *, const Scalar *, const Scalar *, MAT *, Index) override
        {
            return 0;
        }

        Index eval_inequality_parameter_jacobian(
            const Scalar *, const Scalar *, const Scalar *, MAT *, Index) override
        {
            return 0;
        }

        Index eval_b(
            const Scalar *xp1, const Scalar *u, const Scalar *x,
            const Scalar *p, Scalar *res, Index) override
        {
            res[0] = xp1[0] - x[0] - u[0] - p[0];
            return 0;
        }

        Index eval_g(
            const Scalar *, const Scalar *x, const Scalar *,
            Scalar *res, Index stage) override
        {
            res[0] = stage == 0 ? x[0] : x[0] - 1.0;
            return 0;
        }

        Index eval_gineq(
            const Scalar *, const Scalar *, const Scalar *, Scalar *, Index) override
        {
            return 0;
        }

        Index eval_rq(
            const Scalar *scale, const Scalar *u, const Scalar *,
            const Scalar *, Scalar *res, Index stage) override
        {
            if (stage == 0)
            {
                res[0] = *scale * u[0];
                res[1] = 0.0;
            }
            else
                res[0] = 0.0;
            return 0;
        }

        Index eval_rp(
            const Scalar *scale, const Scalar *, const Scalar *,
            const Scalar *p, Scalar *res, Index stage) override
        {
            res[0] = stage == 0 ? *scale * p[0] : 0.0;
            return 0;
        }

        Index eval_L(
            const Scalar *scale, const Scalar *u, const Scalar *,
            const Scalar *p, Scalar *res, Index stage) override
        {
            *res = stage == 0
                ? 0.5 * *scale * (u[0] * u[0] + p[0] * p[0])
                : 0.0;
            return 0;
        }

        Index get_bounds(Scalar *, Scalar *, Index) const override { return 0; }

        bool has_stage_variable_bounds(Index stage) const override
        {
            return stage == 0;
        }

        Index get_stage_variable_bounds(
            Scalar *lower, Scalar *upper, Index stage) const override
        {
            if (stage == 0)
            {
                // [u_0, x_0].  The active control upper bound moves the
                // unconstrained split (u,p)=(0.5,0.5) to (0.4,0.6).
                lower[0] = -2.0;
                upper[0] = 0.4;
                lower[1] = -2.0;
                upper[1] = 2.0;
            }
            return 0;
        }

        Index get_initial_xk(Scalar *x, Index stage) const override
        {
            x[0] = stage == 0 ? 0.0 : 0.4;
            return 0;
        }

        Index get_initial_uk(Scalar *u, Index stage) const override
        {
            if (stage == 0)
                u[0] = 0.2;
            return 0;
        }

        Index get_initial_parameters(Scalar *p) const override
        {
            p[0] = 0.2;
            return 0;
        }

        bool has_parameter_bounds() const override { return true; }

        Index get_parameter_bounds(
            Scalar *lower, Scalar *upper) const override
        {
            lower[0] = -2.0;
            upper[0] = 2.0;
            return 0;
        }
    };

    class ZeroParameterEqualityProblem final
        : public ParametricImplicitOcpAbstract
    {
    public:
        Index get_nx(Index) const override { return 1; }
        Index get_nu(Index) const override { return 0; }
        Index get_ng(Index) const override { return 1; }
        Index get_ng_ineq(Index) const override { return 0; }
        Index get_np() const override { return 0; }
        Index get_horizon_length() const override { return 1; }

        Index eval_BAJbt(
            const Scalar *, const Scalar *, const Scalar *, const Scalar *,
            MAT *, MAT *, MAT *, Index) override
        {
            return 0;
        }

        Index eval_dynamics_parameter_jacobian(
            const Scalar *, const Scalar *, const Scalar *, const Scalar *,
            MAT *, Index) override
        {
            return 91;
        }

        Index eval_RSQrqt(
            const Scalar *scale, const Scalar *, const Scalar *, const Scalar *,
            const Scalar *, const Scalar *, const Scalar *, const Scalar *,
            MAT *res, MAT *, MAT *, Index) override
        {
            blasfeo_matel_wrap(res, 0, 0) += *scale;
            return 0;
        }

        Index eval_parameter_hessian(
            const Scalar *, const Scalar *, const Scalar *, const Scalar *,
            const Scalar *, const Scalar *, const Scalar *, const Scalar *,
            MAT *, MAT *, MAT *, Index) override
        {
            return 91;
        }

        Index eval_Ggt(
            const Scalar *, const Scalar *, const Scalar *, MAT *res,
            Index) override
        {
            blasfeo_matel_wrap(res, 0, 0) = 1.0;
            return 0;
        }

        Index eval_Ggt_ineq(
            const Scalar *, const Scalar *, const Scalar *, MAT *,
            Index) override
        {
            return 0;
        }

        Index eval_equality_parameter_jacobian(
            const Scalar *, const Scalar *, const Scalar *, MAT *,
            Index) override
        {
            return 91;
        }

        Index eval_inequality_parameter_jacobian(
            const Scalar *, const Scalar *, const Scalar *, MAT *,
            Index) override
        {
            return 91;
        }

        Index eval_b(
            const Scalar *, const Scalar *, const Scalar *, const Scalar *,
            Scalar *, Index) override
        {
            return 0;
        }

        Index eval_g(
            const Scalar *, const Scalar *x, const Scalar *, Scalar *res,
            Index) override
        {
            res[0] = x[0] - 1.0;
            return 0;
        }

        Index eval_gineq(
            const Scalar *, const Scalar *, const Scalar *, Scalar *,
            Index) override
        {
            return 0;
        }

        Index eval_rq(
            const Scalar *scale, const Scalar *, const Scalar *x,
            const Scalar *, Scalar *res, Index) override
        {
            res[0] = *scale * x[0];
            return 0;
        }

        Index eval_rp(
            const Scalar *, const Scalar *, const Scalar *, const Scalar *,
            Scalar *, Index) override
        {
            return 91;
        }

        Index eval_L(
            const Scalar *scale, const Scalar *, const Scalar *x,
            const Scalar *, Scalar *res, Index) override
        {
            *res = 0.5 * *scale * x[0] * x[0];
            return 0;
        }

        Index get_bounds(Scalar *, Scalar *, Index) const override
        {
            return 0;
        }

        Index get_initial_xk(Scalar *x, Index) const override
        {
            x[0] = 0.0;
            return 0;
        }

        Index get_initial_uk(Scalar *, Index) const override
        {
            return 0;
        }

        Index get_initial_parameters(Scalar *) const override
        {
            return 91;
        }
    };
}

TEST(ParametricImplicitSolve, OptimizesOneCopyParameterEndToEnd)
{
    auto nlp = std::make_shared<ParametricImplicitNlpOcp>(
        std::make_shared<SharedParameterTransferProblem>());
    OptionRegistry options;
    IpAlgBuilder<ImplicitOcpType> builder(nlp);
    auto algorithm = builder.with_options_registry(&options).build();
    options.set_option("print_level", 0);
    options.set_option("max_iter", 100);
    options.set_option("tolerance", 1e-9);

    const IpSolverReturnFlag status = algorithm->optimize();
    ASSERT_TRUE(
        status == IpSolverReturnFlag::Success
        || status == IpSolverReturnFlag::StopAtAcceptablePoint);

    const ProblemInfo info(nlp->problem_dims());
    const VecRealView &solution = algorithm->solution_primal();
    EXPECT_NEAR(solution(info.offsets_primal_u[0]), 0.4, 2e-6);
    EXPECT_NEAR(solution(info.offsets_primal_x[0]), 0.0, 1e-7);
    EXPECT_NEAR(solution(info.offsets_primal_x[1]), 1.0, 1e-7);
    EXPECT_NEAR(solution(info.offset_primal_global), 0.6, 2e-6);
}

TEST(ParametricImplicitSolve, ReusesCompletePrimalDualSolutionWithGlobalParameter)
{
    auto nlp = std::make_shared<ParametricImplicitNlpOcp>(
        std::make_shared<SharedParameterTransferProblem>());
    OptionRegistry options;
    IpAlgBuilder<ImplicitOcpType> builder(nlp);
    auto algorithm = builder.with_options_registry(&options).build();
    options.set_option("print_level", 0);
    options.set_option("max_iter", 100);
    options.set_option("tolerance", 1e-9);

    const IpSolverReturnFlag cold_status = algorithm->optimize();
    ASSERT_TRUE(cold_status == IpSolverReturnFlag::Success
                || cold_status == IpSolverReturnFlag::StopAtAcceptablePoint);
    const Index cold_iterations = algorithm->iteration_count();
    const Index parameter_offset = algorithm->info().offset_primal_global;
    const Scalar cold_parameter = algorithm->solution_primal()(parameter_offset);

    algorithm->set_warm_start_from_solution();
    ASSERT_TRUE(algorithm->has_warm_start());
    const IpSolverReturnFlag warm_status = algorithm->optimize();
    ASSERT_TRUE(warm_status == IpSolverReturnFlag::Success
                || warm_status == IpSolverReturnFlag::StopAtAcceptablePoint);
    EXPECT_LE(algorithm->iteration_count(), cold_iterations);
    EXPECT_NEAR(algorithm->solution_primal()(parameter_offset), cold_parameter, 1e-10);
    EXPECT_NEAR(algorithm->solution_primal()(parameter_offset), 0.6, 2e-6);

    algorithm->clear_warm_start();
    EXPECT_FALSE(algorithm->has_warm_start());
}

TEST(ParametricImplicitSolve, UsesTheSameNativeApiWithoutStaticParameters)
{
    auto nlp = std::make_shared<ParametricImplicitNlpOcp>(
        std::make_shared<ZeroParameterEqualityProblem>());
    EXPECT_EQ(nlp->problem_dims().number_of_global_parameters, 0);
    OptionRegistry options;
    IpAlgBuilder<ImplicitOcpType> builder(nlp);
    auto algorithm = builder.with_options_registry(&options).build();
    options.set_option("print_level", 0);
    options.set_option("max_iter", 50);
    options.set_option("tolerance", 1e-10);

    const IpSolverReturnFlag status = algorithm->optimize();
    ASSERT_TRUE(
        status == IpSolverReturnFlag::Success
        || status == IpSolverReturnFlag::StopAtAcceptablePoint);
    EXPECT_EQ(algorithm->info().number_of_global_parameters, 0);
    EXPECT_NEAR(algorithm->solution_primal()(0), 1.0, 1e-9);
}
