//
// Copyright (C) 2024 Lander Vanroye, KU Leuven
//

#ifndef __fatrop_ip_algorithm_ip_data_hxx__
#define __fatrop_ip_algorithm_ip_data_hxx__
#include "fatrop/common/options.hpp"
#include "ip_data.hpp"
#include <cmath>

namespace fatrop
{

    template <typename ProblemType>
    IpData<ProblemType>::IpData(const NlpSp &nlp)
        : nlp_(nlp), info_(nlp->problem_dims()), iterate_data_{*this, *this, *this},
          current_iterate_(&iterate_data_[0]), trial_iterate_(&iterate_data_[1]),
          stored_iterate_(&iterate_data_[2]),
          warm_start_primal_x_(nlp->nlp_dims().number_of_variables),
          warm_start_primal_s_(nlp->nlp_dims().number_of_ineq_constraints),
          warm_start_dual_eq_(nlp->nlp_dims().number_of_eq_constraints),
          warm_start_dual_bounds_l_(nlp->nlp_dims().number_of_ineq_constraints),
          warm_start_dual_bounds_u_(nlp->nlp_dims().number_of_ineq_constraints),
          hessian_data_{nlp->problem_dims(), nlp->problem_dims(), nlp->problem_dims()},
          jacobian_data_{nlp->problem_dims(), nlp->problem_dims(), nlp->problem_dims()},
          lower_bounds_(nlp->nlp_dims().number_of_ineq_constraints),
          upper_bounds_(nlp->nlp_dims().number_of_ineq_constraints),
          lower_bounded_(nlp->nlp_dims().number_of_ineq_constraints),
          upper_bounded_(nlp->nlp_dims().number_of_ineq_constraints),
          single_lower_bounded_(nlp->nlp_dims().number_of_ineq_constraints),
          single_upper_bounded_(nlp->nlp_dims().number_of_ineq_constraints)
    {
        reset();
    }

    template <typename ProblemType> void IpData<ProblemType>::reset(bool is_resto /* = false*/)
    {
        // set the bounds
        get_nlp()->get_bounds(info_, lower_bounds_, upper_bounds_);
        // set the bound flags
        number_of_bounds_ = 0;
        for (Index i = 0; i < lower_bounds_.m(); i++)
        {
            lower_bounded_[i] = !std::isinf(lower_bounds_(i));
            upper_bounded_[i] = !std::isinf(upper_bounds_(i));
            if (lower_bounded_[i])
                number_of_bounds_++;
            if (upper_bounded_[i])
                number_of_bounds_++;
            bool single_bounded = lower_bounded_[i] ^ upper_bounded_[i];
            single_lower_bounded_[i] = single_bounded && lower_bounded_[i];
            single_upper_bounded_[i] = single_bounded && upper_bounded_[i];
        }
        tiny_step_flag_ = false;
        if(!is_resto) iteration_number_ = 0;
        // reset associated iterates
        for (auto &iterate : iterate_data_)
        {
            iterate.reset();
        }
        jacobian_curr_ = &jacobian_data_[0];
        hessian_curr_ = &hessian_data_[0];
        hessian_trial_ = &hessian_data_[1];
        jacobian_trial_ = &jacobian_data_[1];
        jacobian_stored_ = &jacobian_data_[2];
        hessian_stored_ = &hessian_data_[2];
        timing_statistics().reset();
        current_iterate().set_hessian(hessian_curr_);
        current_iterate().set_jacobian(jacobian_curr_);
        trial_iterate().set_hessian(hessian_trial_);
        trial_iterate().set_jacobian(jacobian_trial_);
        stored_iterate().set_hessian(nullptr);
        stored_iterate().set_jacobian(nullptr);
        stored_iterate_is_valid_ = false;
    }
    template <typename ProblemType> void IpData<ProblemType>::accept_trial_iterate()
    {
        // switch trial and current iterate
        std::swap(current_iterate_, trial_iterate_);
        std::swap(hessian_curr_, hessian_trial_);
        std::swap(jacobian_curr_, jacobian_trial_);
        trial_iterate_->reset_evaluated_quantities();
        // set the hessian and jacobian pointers
        current_iterate().set_hessian(hessian_curr_);
        current_iterate().set_jacobian(jacobian_curr_);
        trial_iterate().set_hessian(hessian_trial_);
        trial_iterate().set_jacobian(jacobian_trial_);
    }

    template <typename ProblemType> void IpData<ProblemType>::store_current_iterate()
    {
        // fatrop_assert_msg(!stored_iterate_is_valid_,
        //                   "Trying to store iterate but there is already an iterate stored.");
        // copy current iterate to stored iterate
        stored_iterate() = current_iterate();
        // set the hess and jac pointers of the current iterate to nullptr such that they are not
        // overwritten for certainty
        current_iterate().set_hessian(nullptr);
        current_iterate().set_jacobian(nullptr);
        // swap jacobian and hessian pointers
        std::swap(hessian_curr_, hessian_stored_);
        std::swap(jacobian_curr_, jacobian_stored_);
        stored_iterate_is_valid_ = true;
    }

    template <typename ProblemType> void IpData<ProblemType>::restore_current_iterate()
    {
        fatrop_assert_msg(stored_iterate_is_valid_,
                          "Trying to restore iterate but stored iterate is not valid.");
        // swap current and stored iterate
        std::swap(current_iterate_, stored_iterate_);
        // also swap the hessian and jacobian pointers
        std::swap(hessian_curr_, hessian_stored_);
        std::swap(jacobian_curr_, jacobian_stored_);
        stored_iterate_is_valid_ = false;
    }

    template <typename ProblemType>
    void IpData<ProblemType>::register_options(OptionRegistry &registry)
    {
        registry.register_option("tolerance", &IpData::set_tolerance, this);
    }

    template <typename ProblemType>
    void IpData<ProblemType>::set_warm_start(const VecRealView &primal_x,
                                             const VecRealView &primal_s,
                                             const VecRealView &dual_eq,
                                             const VecRealView &dual_bounds_l,
                                             const VecRealView &dual_bounds_u, Scalar mu)
    {
        const NlpDims &dims = nlp_->nlp_dims();
        fatrop_assert_msg(primal_x.m() == dims.number_of_variables,
                          "Warm-start primal vector has the wrong dimension.");
        fatrop_assert_msg(primal_s.m() == dims.number_of_ineq_constraints,
                          "Warm-start slack vector has the wrong dimension.");
        fatrop_assert_msg(dual_eq.m() == dims.number_of_eq_constraints,
                          "Warm-start equality-dual vector has the wrong dimension.");
        fatrop_assert_msg(dual_bounds_l.m() == dims.number_of_ineq_constraints,
                          "Warm-start lower-bound-dual vector has the wrong dimension.");
        fatrop_assert_msg(dual_bounds_u.m() == dims.number_of_ineq_constraints,
                          "Warm-start upper-bound-dual vector has the wrong dimension.");
        fatrop_assert_msg(primal_x.is_finite() && primal_s.is_finite()
                              && dual_eq.is_finite() && dual_bounds_l.is_finite()
                              && dual_bounds_u.is_finite(),
                          "Warm-start vectors must contain only finite values.");
        fatrop_assert_msg(std::isfinite(mu) && mu > 0.,
                          "Warm-start barrier parameter must be finite and positive.");

        warm_start_primal_x_ = primal_x;
        warm_start_primal_s_ = primal_s;
        warm_start_dual_eq_ = dual_eq;
        warm_start_dual_bounds_l_ = dual_bounds_l;
        warm_start_dual_bounds_u_ = dual_bounds_u;
        warm_start_mu_ = mu;
        warm_start_available_ = true;
    }

    template <typename ProblemType>
    void IpData<ProblemType>::set_warm_start_from_current_iterate()
    {
        const Iterate &iterate = current_iterate();
        set_warm_start(iterate.primal_x(), iterate.primal_s(), iterate.dual_eq(),
                       iterate.dual_bounds_l(), iterate.dual_bounds_u(), iterate.mu());
    }

} // namespace fatrop

#endif // __fatrop_ip_algorithm_ip_data_hxx__
