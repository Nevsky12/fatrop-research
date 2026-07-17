#include "../ocp/ocp_test_probem.hpp"
#include "fatrop/ip_algorithm/ip_data.hpp"
#include "fatrop/ip_algorithm/ip_eq_mult_initializer.hpp"
#include "fatrop/ip_algorithm/ip_initializer.hpp"
#include "fatrop/ocp/aug_system_solver.hpp"
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/nlp_ocp.hpp"
#include "fatrop/ocp/pd_solver_orig.hpp"
#include "fatrop/ocp/type.hpp"
#include <gtest/gtest.h>
#include <limits>

namespace fatrop
{
    namespace test
    {

        class IpInitializerTest : public ::testing::Test
        {
        protected:
            IpInitializerTest()
                : problem(std::make_shared<OcpTestProblem>()),
                  nlp(std::make_shared<NlpOcp>(problem)), info(nlp->problem_dims()),
                  ipdata(std::make_shared<IpData<OcpType>>(nlp)),
                  D_x(info.number_of_primal_variables + info.number_of_slack_variables),
                  D_eq(info.number_of_eq_constraints),
                  aug_solver(std::make_shared<AugSystemSolver<OcpType>>(info)),
                  linear_solver(std::make_shared<PdSolverOrig<OcpType>>(info, aug_solver)),
                  eq_mult_initializer(
                      std::make_shared<IpEqMultInitializer<OcpType>>(ipdata, linear_solver)),
                  initializer(std::make_shared<IpInitializer<OcpType>>(ipdata, eq_mult_initializer))
            {
                ipdata->current_iterate().set_dual_bounds_l(
                    VecRealScalar(ipdata->current_iterate().dual_bounds_l().m(), 1));
                ipdata->current_iterate().set_dual_bounds_u(
                    VecRealScalar(ipdata->current_iterate().dual_bounds_u().m(), 1));
            }

            std::shared_ptr<OcpTestProblem> problem;
            std::shared_ptr<NlpOcp> nlp;
            ProblemInfo info;
            std::shared_ptr<IpData<OcpType>> ipdata;
            VecRealAllocated D_x, D_eq;
            std::shared_ptr<AugSystemSolver<OcpType>> aug_solver;
            std::shared_ptr<PdSolverOrig<OcpType>> linear_solver;
            std::shared_ptr<IpEqMultInitializer<OcpType>> eq_mult_initializer;
            std::shared_ptr<IpInitializer<OcpType>> initializer;
        };

        TEST_F(IpInitializerTest, EqMultInitializerTest)
        {
            // Check that the equality multiplier initialization doesn't throw an exception
            EXPECT_NO_THROW({ initializer->initialize(); })
                << "Equality multiplier initialization should not throw an exception";
        }

        TEST_F(IpInitializerTest, BoundPushTestLow)
        {

            auto &current_iterate = ipdata->current_iterate();
            auto &primal_s = current_iterate.primal_s();
            auto &primal_x = current_iterate.primal_x();
            current_iterate.set_primal_x(VecRealScalar(primal_x.m(), -100.));
            Scalar norm_l1_cv = norm_l1(current_iterate.constr_viol_ineq());

            // Initialize the problem
            initializer->initialize();

            const auto &lower_bounds = current_iterate.lower_bounds();
            const auto &upper_bounds = current_iterate.upper_bounds();

            for (int i = 0; i < primal_s.m(); ++i)
            {
                if (current_iterate.lower_bounded()[i])
                {
                    EXPECT_GT(primal_s(i), lower_bounds(i))
                        << "Slack should be pushed away from lower bound at index " << i;
                }
                if (current_iterate.upper_bounded()[i])
                {
                    EXPECT_LT(primal_s(i), upper_bounds(i))
                        << "Slack should be pushed away from upper bound at index " << i;
                }
            }
            EXPECT_LT(norm_l1(current_iterate.constr_viol_ineq()), norm_l1_cv)
                << "Norm of inequality constraint violation should decrease";
        }

        TEST_F(IpInitializerTest, BoundPushTestHigh)
        {

            auto &current_iterate = ipdata->current_iterate();
            auto &primal_s = current_iterate.primal_s();
            auto &primal_x = current_iterate.primal_x();
            current_iterate.set_primal_x(VecRealScalar(primal_x.m(), 100.));
            Scalar norm_l1_cv = norm_l1(current_iterate.constr_viol_ineq());

            // Initialize the problem
            initializer->initialize();

            const auto &lower_bounds = current_iterate.lower_bounds();
            const auto &upper_bounds = current_iterate.upper_bounds();

            for (int i = 0; i < primal_s.m(); ++i)
            {
                if (current_iterate.lower_bounded()[i])
                {
                    EXPECT_GT(primal_s(i), lower_bounds(i))
                        << "Slack should be pushed away from lower bound at index " << i;
                }
                if (current_iterate.upper_bounded()[i])
                {
                    EXPECT_LT(primal_s(i), upper_bounds(i))
                        << "Slack should be pushed away from upper bound at index " << i;
                }
            }
            EXPECT_LT(norm_l1(current_iterate.constr_viol_ineq()), norm_l1_cv)
                << "Norm of inequality constraint violation should decrease";
        }

        TEST_F(IpInitializerTest, CompleteWarmStartIsCopiedAndRestored)
        {
            initializer->initialize();
            const auto &iterate = ipdata->current_iterate();
            VecRealAllocated primal_x(iterate.primal_x().m());
            VecRealAllocated primal_s(iterate.primal_s());
            VecRealAllocated dual_eq(iterate.dual_eq().m());
            VecRealAllocated dual_l(iterate.dual_bounds_l().m());
            VecRealAllocated dual_u(iterate.dual_bounds_u().m());

            for (Index i = 0; i < primal_x.m(); ++i)
                primal_x(i) = 0.125 + 0.01 * i;
            for (Index i = 0; i < dual_eq.m(); ++i)
                dual_eq(i) = -0.25 + 0.02 * i;
            for (Index i = 0; i < dual_l.m(); ++i)
            {
                dual_l(i) = 2. + 0.1 * i;
                dual_u(i) = 3. + 0.1 * i;
            }
            constexpr Scalar mu = 2.5e-2;
            ipdata->set_warm_start(primal_x, primal_s, dual_eq, dual_l, dual_u, mu);

            // Verify that the input may be destroyed or modified before initialize().
            primal_x = -99.;
            primal_s = -99.;
            dual_eq = -99.;
            dual_l = -99.;
            dual_u = -99.;
            ipdata->reset();
            initializer->initialize();

            const auto &warm = ipdata->current_iterate();
            for (Index i = 0; i < warm.primal_x().m(); ++i)
                EXPECT_DOUBLE_EQ(warm.primal_x()(i), 0.125 + 0.01 * i);
            for (Index i = 0; i < warm.dual_eq().m(); ++i)
                EXPECT_DOUBLE_EQ(warm.dual_eq()(i), -0.25 + 0.02 * i);
            for (Index i = 0; i < warm.dual_bounds_l().m(); ++i)
            {
                EXPECT_DOUBLE_EQ(warm.dual_bounds_l()(i),
                                 warm.lower_bounded()[i] ? 2. + 0.1 * i : 0.);
                EXPECT_DOUBLE_EQ(warm.dual_bounds_u()(i),
                                 warm.upper_bounded()[i] ? 3. + 0.1 * i : 0.);
            }
            EXPECT_DOUBLE_EQ(warm.mu(), mu);
            EXPECT_DOUBLE_EQ(ipdata->trial_iterate().mu(), mu);
        }

        TEST_F(IpInitializerTest, WarmStartRepairsBoundaryAndNonpositiveBoundDuals)
        {
            initializer->initialize();
            const auto &iterate = ipdata->current_iterate();
            VecRealAllocated primal_x(iterate.primal_x());
            VecRealAllocated primal_s(iterate.primal_s());
            VecRealAllocated dual_eq(iterate.dual_eq());
            VecRealAllocated dual_l(iterate.dual_bounds_l().m());
            VecRealAllocated dual_u(iterate.dual_bounds_u().m());
            dual_l = -1.;
            dual_u = 0.;
            for (Index i = 0; i < primal_s.m(); ++i)
            {
                if (iterate.lower_bounded()[i])
                    primal_s(i) = iterate.lower_bounds()(i);
                else if (iterate.upper_bounded()[i])
                    primal_s(i) = iterate.upper_bounds()(i);
            }

            ipdata->set_warm_start(primal_x, primal_s, dual_eq, dual_l, dual_u, 0.05);
            ipdata->reset();
            initializer->initialize();
            const auto &warm = ipdata->current_iterate();
            for (Index i = 0; i < warm.primal_s().m(); ++i)
            {
                if (warm.lower_bounded()[i])
                {
                    EXPECT_GT(warm.primal_s()(i), warm.lower_bounds()(i));
                    EXPECT_GT(warm.dual_bounds_l()(i), 0.);
                }
                else
                    EXPECT_DOUBLE_EQ(warm.dual_bounds_l()(i), 0.);
                if (warm.upper_bounded()[i])
                {
                    EXPECT_LT(warm.primal_s()(i), warm.upper_bounds()(i));
                    EXPECT_GT(warm.dual_bounds_u()(i), 0.);
                }
                else
                    EXPECT_DOUBLE_EQ(warm.dual_bounds_u()(i), 0.);
            }
        }

        TEST_F(IpInitializerTest, WarmStartRejectsInvalidDimensionAndNonfiniteData)
        {
            const auto &iterate = ipdata->current_iterate();
            VecRealAllocated primal_x(iterate.primal_x().m());
            VecRealAllocated bad_primal_x(iterate.primal_x().m() + 1);
            VecRealAllocated primal_s(iterate.primal_s().m());
            VecRealAllocated dual_eq(iterate.dual_eq().m());
            VecRealAllocated dual_l(iterate.dual_bounds_l().m());
            VecRealAllocated dual_u(iterate.dual_bounds_u().m());

            EXPECT_THROW(ipdata->set_warm_start(bad_primal_x, primal_s, dual_eq, dual_l,
                                                dual_u, 0.1),
                         FatropException);
            if (primal_x.m() > 0)
                primal_x(0) = std::numeric_limits<Scalar>::quiet_NaN();
            EXPECT_THROW(ipdata->set_warm_start(primal_x, primal_s, dual_eq, dual_l,
                                                dual_u, 0.1),
                         FatropException);
            primal_x = 0.;
            EXPECT_THROW(ipdata->set_warm_start(primal_x, primal_s, dual_eq, dual_l,
                                                dual_u, 0.),
                         FatropException);
        }


        class ImplicitIpInitializerTest : public ::testing::Test
        {
        protected:
            ImplicitIpInitializerTest()
                : problem(std::make_shared<ImplicitOcpTestProblem>()),
                  nlp(std::make_shared<ImplicitNlpOcp>(problem)), info(nlp->problem_dims()),
                  ipdata(std::make_shared<IpData<ImplicitOcpType>>(nlp)),
                  D_x(info.number_of_primal_variables + info.number_of_slack_variables),
                  D_eq(info.number_of_eq_constraints),
                  aug_solver(std::make_shared<AugSystemSolver<ImplicitOcpType>>(info)),
                  linear_solver(std::make_shared<PdSolverOrig<ImplicitOcpType>>(info, aug_solver)),
                  eq_mult_initializer(
                      std::make_shared<IpEqMultInitializer<ImplicitOcpType>>(ipdata, linear_solver)),
                  initializer(std::make_shared<IpInitializer<ImplicitOcpType>>(ipdata, eq_mult_initializer))
            {
                ipdata->current_iterate().set_dual_bounds_l(
                    VecRealScalar(ipdata->current_iterate().dual_bounds_l().m(), 1));
                ipdata->current_iterate().set_dual_bounds_u(
                    VecRealScalar(ipdata->current_iterate().dual_bounds_u().m(), 1));
            }

            std::shared_ptr<ImplicitOcpTestProblem> problem;
            std::shared_ptr<ImplicitNlpOcp> nlp;
            ProblemInfo info;
            std::shared_ptr<IpData<ImplicitOcpType>> ipdata;
            VecRealAllocated D_x, D_eq;
            std::shared_ptr<AugSystemSolver<ImplicitOcpType>> aug_solver;
            std::shared_ptr<PdSolverOrig<ImplicitOcpType>> linear_solver;
            std::shared_ptr<IpEqMultInitializer<ImplicitOcpType>> eq_mult_initializer;
            std::shared_ptr<IpInitializer<ImplicitOcpType>> initializer;
        };

        TEST_F(ImplicitIpInitializerTest, ImplicitEqMultInitializerTest)
        {
            // Check that the equality multiplier initialization doesn't throw an exception
            EXPECT_NO_THROW({ initializer->initialize(); })
                << "Equality multiplier initialization should not throw an exception";
        }

        TEST_F(ImplicitIpInitializerTest, ImplicitBoundPushTestLow)
        {

            auto &current_iterate = ipdata->current_iterate();
            auto &primal_s = current_iterate.primal_s();
            auto &primal_x = current_iterate.primal_x();
            current_iterate.set_primal_x(VecRealScalar(primal_x.m(), -100.));
            Scalar norm_l1_cv = norm_l1(current_iterate.constr_viol_ineq());

            // Initialize the problem
            initializer->initialize();

            const auto &lower_bounds = current_iterate.lower_bounds();
            const auto &upper_bounds = current_iterate.upper_bounds();

            for (int i = 0; i < primal_s.m(); ++i)
            {
                if (current_iterate.lower_bounded()[i])
                {
                    EXPECT_GT(primal_s(i), lower_bounds(i))
                        << "Slack should be pushed away from lower bound at index " << i;
                }
                if (current_iterate.upper_bounded()[i])
                {
                    EXPECT_LT(primal_s(i), upper_bounds(i))
                        << "Slack should be pushed away from upper bound at index " << i;
                }
            }
            EXPECT_LT(norm_l1(current_iterate.constr_viol_ineq()), norm_l1_cv)
                << "Norm of inequality constraint violation should decrease";
        }

        TEST_F(ImplicitIpInitializerTest, ImplicitBoundPushTestHigh)
        {

            auto &current_iterate = ipdata->current_iterate();
            auto &primal_s = current_iterate.primal_s();
            auto &primal_x = current_iterate.primal_x();
            current_iterate.set_primal_x(VecRealScalar(primal_x.m(), 100.));
            Scalar norm_l1_cv = norm_l1(current_iterate.constr_viol_ineq());

            // Initialize the problem
            initializer->initialize();

            const auto &lower_bounds = current_iterate.lower_bounds();
            const auto &upper_bounds = current_iterate.upper_bounds();

            for (int i = 0; i < primal_s.m(); ++i)
            {
                if (current_iterate.lower_bounded()[i])
                {
                    EXPECT_GT(primal_s(i), lower_bounds(i))
                        << "Slack should be pushed away from lower bound at index " << i;
                }
                if (current_iterate.upper_bounded()[i])
                {
                    EXPECT_LT(primal_s(i), upper_bounds(i))
                        << "Slack should be pushed away from upper bound at index " << i;
                }
            }
            EXPECT_LT(norm_l1(current_iterate.constr_viol_ineq()), norm_l1_cv)
                << "Norm of inequality constraint violation should decrease";
        }

    } // namespace fatrop::test
} // namespace fatrop::test

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
