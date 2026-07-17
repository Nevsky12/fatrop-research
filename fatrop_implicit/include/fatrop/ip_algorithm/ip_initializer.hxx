//
// Copyright (C) 2024 Lander Vanroye, KU Leuven
//

#ifndef __fatrop_ip_algorithm_ip_initializer_hxx__
#define __fatrop_ip_algorithm_ip_initializer_hxx__
#include "fatrop/common/options.hpp"
#include "fatrop/ip_algorithm/ip_data.hpp"
#include "fatrop/ip_algorithm/ip_eq_mult_initializer.hpp"
#include "fatrop/nlp/nlp.hpp"
#include "fatrop/ip_algorithm/ip_initializer.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
namespace fatrop
{

    template <typename ProblemType>
    IpInitializer<ProblemType>::IpInitializer(const IpDataSp ipdata,
                                              const IpEqMultInitializerSp &eq_mult_initializer)
        : ipdata_(ipdata), eq_mult_initializer_(eq_mult_initializer),
          primal_buff_(ipdata->current_iterate().primal_x().m()),
          slack_buff_(ipdata->current_iterate().primal_s().m())
    {
    }

    template <typename ProblemType> void IpInitializer<ProblemType>::initialize()
    {
        if (ipdata_->has_warm_start())
        {
            initialize_warm_start();
            return;
        }
        // get primal initialization from the interface
        ipdata_->get_nlp()->get_initial_primal(ipdata_->current_iterate().info(), primal_buff_);
        ipdata_->current_iterate().set_primal_x(primal_buff_);
        const Index m = ipdata_->current_iterate().primal_s().m();
        const std::vector<bool> &lower_bounded = ipdata_->current_iterate().lower_bounded();
        const std::vector<bool> &upper_bounded = ipdata_->current_iterate().upper_bounded();
        // set z to 1. if bounded and 0. otherwise
        initialize_slacks();
        ipdata_->current_iterate().set_dual_bounds_l(
            if_else(lower_bounded, VecRealScalar(m, 1.0), VecRealScalar(m, 0.0)));
        ipdata_->current_iterate().set_dual_bounds_u(
            if_else(upper_bounded, VecRealScalar(m, 1.0), VecRealScalar(m, 0.0)));
        eq_mult_initializer_->initialize_eq_mult();
    }
    template <typename ProblemType> void IpInitializer<ProblemType>::initialize_warm_start()
    {
        fatrop_assert_msg(std::isfinite(warm_start_bound_push_)
                              && std::isfinite(warm_start_bound_frac_)
                              && std::isfinite(warm_start_mult_bound_push_)
                              && warm_start_bound_push_ >= 0.
                              && warm_start_bound_frac_ >= 0.
                              && warm_start_mult_bound_push_ > 0.,
                          "Warm-start push options must be nonnegative and the multiplier push "
                          "must be positive.");
        IpIterate<ProblemType> &iterate = ipdata_->current_iterate();
        iterate.set_primal_x(ipdata_->warm_start_primal_x());
        project_slacks(ipdata_->warm_start_primal_s(), warm_start_bound_push_,
                       warm_start_bound_frac_);

        const Index m = iterate.primal_s().m();
        const VecRealScalar zero(m, 0.);
        const VecRealScalar dual_push(m, warm_start_mult_bound_push_);
        iterate.set_dual_bounds_l(
            if_else(iterate.lower_bounded(),
                    max(ipdata_->warm_start_dual_bounds_l(), dual_push), zero));
        iterate.set_dual_bounds_u(
            if_else(iterate.upper_bounded(),
                    max(ipdata_->warm_start_dual_bounds_u(), dual_push), zero));
        iterate.set_dual_eq(ipdata_->warm_start_dual_eq());
        iterate.set_mu(ipdata_->warm_start_mu());
        ipdata_->trial_iterate().set_mu(ipdata_->warm_start_mu());
    }
    template <typename ProblemType> void IpInitializer<ProblemType>::initialize_slacks()
    {
        // set slack variables to zero
        ipdata_->current_iterate().set_primal_s(
            VecRealScalar(ipdata_->current_iterate().primal_s().m(), 0.0));
        // fatrop_assert_msg(ipdata_->current_iterate().primal_s().is_zero(),
        //                   "Slack variables must be zero at initialization");
        const VecRealView viol_s = ipdata_->current_iterate().constr_viol_ineq();
        slack_buff_ = 0.;
        slack_buff_.block(viol_s.m(), 0) = viol_s;
        project_slacks(slack_buff_, bound_push, bound_frac);
    }

    template <typename ProblemType>
    void IpInitializer<ProblemType>::project_slacks(const VecRealView &input,
                                                    Scalar push, Scalar fraction)
    {
        IpIterate<ProblemType> &iterate = ipdata_->current_iterate();
        fatrop_assert_msg(input.m() == iterate.primal_s().m(),
                          "Slack projection received a vector with the wrong dimension.");
        fatrop_assert_msg(input.is_finite(), "Initial slack values must be finite.");
        fatrop_assert_msg(std::isfinite(push) && std::isfinite(fraction)
                              && push >= 0. && fraction >= 0.,
                          "Slack push options must be finite and nonnegative.");

        const VecRealView &lower = iterate.lower_bounds();
        const VecRealView &upper = iterate.upper_bounds();
        const std::vector<bool> &has_lower = iterate.lower_bounded();
        const std::vector<bool> &has_upper = iterate.upper_bounded();
        const Scalar safe_fraction = std::min(fraction, Scalar(0.499999999));
        for (Index i = 0; i < input.m(); ++i)
        {
            Scalar value = input(i);
            if (has_lower[i] && has_upper[i])
            {
                const Scalar width = upper(i) - lower(i);
                fatrop_assert_msg(std::isfinite(width) && width > 0.,
                                  "A double-bounded slack must have positive bound width.");
                const Scalar left_push =
                    std::min(push * std::max(Scalar(1.), std::abs(lower(i))),
                             safe_fraction * width);
                const Scalar right_push =
                    std::min(push * std::max(Scalar(1.), std::abs(upper(i))),
                             safe_fraction * width);
                value = std::max(value, lower(i) + left_push);
                value = std::min(value, upper(i) - right_push);
                if (!(value > lower(i)))
                    value = std::nextafter(lower(i), upper(i));
                if (!(value < upper(i)))
                    value = std::nextafter(upper(i), lower(i));
            }
            else if (has_lower[i])
            {
                value = std::max(
                    value, lower(i) + push * std::max(Scalar(1.), std::abs(lower(i))));
                if (!(value > lower(i)))
                    value = std::nextafter(lower(i), std::numeric_limits<Scalar>::infinity());
            }
            else if (has_upper[i])
            {
                value = std::min(
                    value, upper(i) - push * std::max(Scalar(1.), std::abs(upper(i))));
                if (!(value < upper(i)))
                    value = std::nextafter(upper(i), -std::numeric_limits<Scalar>::infinity());
            }
            slack_buff_(i) = value;
        }
        iterate.set_primal_s(slack_buff_);
    }

    template <typename ProblemType> void IpInitializer<ProblemType>::reset()
    {
        // Empty implementation
    }
    template <typename ProblemType>
    void IpInitializer<ProblemType>::register_options(OptionRegistry &registry)
    {
        registry.register_option("bound_push", &IpInitializer::set_bound_push, this);
        registry.register_option("bound_frac", &IpInitializer::set_bound_frac, this);
        registry.register_option("warm_start_bound_push",
                                 &IpInitializer::set_warm_start_bound_push, this);
        registry.register_option("warm_start_bound_frac",
                                 &IpInitializer::set_warm_start_bound_frac, this);
        registry.register_option("warm_start_mult_bound_push",
                                 &IpInitializer::set_warm_start_mult_bound_push, this);
    }

} // namespace fatrop

#endif // __fatrop_ip_algorithm_ip_initializer_hxx__
