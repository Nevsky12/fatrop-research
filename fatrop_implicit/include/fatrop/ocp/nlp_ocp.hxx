//
// Copyright (C) 2024 Lander Vanroye, KU Leuven
//

#ifndef __fatrop_ocp_nlp_ocp_hxx__
#define __fatrop_ocp_nlp_ocp_hxx__

#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/nlp_ocp.hpp"
#include "fatrop/ocp/problem_info.hpp"

namespace fatrop
{
    namespace internal
    {
        template <typename OcpAbstractTag> struct NlpOcpAuxiliary
        {
            static ProblemDims get_ocp_dims(const OcpAbstractTpl<OcpAbstractTag> &ocp)
            {
                const Index K = ocp.get_horizon_length();
                std::vector<Index> nu(K), nx(K), ng(K), ng_ineq(K);
                for (Index k = 0; k < K; k++)
                {
                    nu[k] = ocp.get_nu(k);
                    nx[k] = ocp.get_nx(k);
                    ng[k] = ocp.get_ng(k);
                    ng_ineq[k] = ocp.get_ng_ineq(k);
                }
                return ProblemDims(K, nu, nx, ng, ng_ineq);
            }
            static NlpDims get_nlp_dims(const ProblemDims &ocp_dims)
            {
                Index number_of_variables = 0;
                Index number_of_eq_constraints = 0;
                Index number_of_ineq_constraints = 0;
                for (Index k = 0; k < ocp_dims.K; k++)
                {
                    number_of_variables +=
                        ocp_dims.number_of_controls[k] + ocp_dims.number_of_states[k];
                    number_of_eq_constraints += ocp_dims.number_of_eq_constraints[k] +
                                                ocp_dims.number_of_ineq_constraints[k];
                    if (k != ocp_dims.K - 1)
                    {
                        number_of_eq_constraints += ocp_dims.number_of_states[k + 1];
                    }
                    number_of_ineq_constraints += ocp_dims.number_of_ineq_constraints[k];
                }
                return NlpDims(number_of_variables, number_of_eq_constraints,
                               number_of_ineq_constraints);
            }
        };
    }

    template <typename OcpAbstractTag, typename ProblemType>
    NlpOcpTpl<OcpAbstractTag, ProblemType>::NlpOcpTpl(const OcpAbstractSp &ocp)
        : ocp_(ocp),
          ocp_dims_(fatrop::internal::NlpOcpAuxiliary<OcpAbstractTag>::get_ocp_dims(*ocp)),
          nlp_dims_(fatrop::internal::NlpOcpAuxiliary<OcpAbstractTag>::get_nlp_dims(ocp_dims_))
    {
    }

    template <typename OcpAbstractTag, typename ProblemType>
    Index NlpOcpTpl<OcpAbstractTag, ProblemType>::eval_lag_hess(const OcpInfo &info,
                                                   const Scalar objective_scale,
                                                   const VecRealView &primal_x,
                                                   const VecRealView &primal_s,
                                                   const VecRealView &lam, Hessian<ProblemType> &hess)
    {
        const Scalar *primal_x_p = primal_x.data();
        const Scalar *primal_s_p = primal_s.data();
        const Scalar *lam_p = lam.data();
        for (Index k = 0; k < info.dims.K; k++)
        {
            const Scalar *inputs_k = primal_x_p + info.offsets_primal_u[k];
            const Scalar *states_k = primal_x_p + info.offsets_primal_x[k];
            const Scalar *lam_dyn_k =
                (k != info.dims.K - 1) ? lam_p + info.offsets_g_eq_dyn[k] : nullptr;
            const Scalar *lam_eq_k = lam_p + info.offsets_g_eq_path[k];
            const Scalar *lam_eq_ineq_k = lam_p + info.offsets_g_eq_slack[k];

            if constexpr (std::is_same_v<ProblemType, ImplicitOcpType>)
            {
                // OLD APPROACH
                /*
                const Scalar *inputs_km1 = (k > 0) ? primal_x_p + info.offsets_primal_u[k - 1] : nullptr;
                const Scalar *states_km1 = (k > 0) ? primal_x_p + info.offsets_primal_x[k - 1] : nullptr;
                const Scalar *states_kp1 = (k < info.dims.K - 1) ? primal_x_p + info.offsets_primal_x[k + 1] : nullptr;
                const Scalar *lam_dyn_km1 = (k > 0) ? lam_p + info.offsets_g_eq_dyn[k - 1] : nullptr;

                // Evaluate lagrangian hessian (requires states_kp1 and also 
                // influences hessian at next stage)
                // ocp_->eval_RSQrqt(&objective_scale, inputs_k, states_k, 
                //                   states_kp1, lam_dyn_k, lam_eq_k, 
                //                   lam_eq_ineq_k, &hess.RSQrqt[k].mat(), 
                //                   hess_next, k);
                ocp_->eval_RSQrqt_old(&objective_scale, inputs_km1, states_km1,
                                  inputs_k, states_k, states_kp1, lam_dyn_k, 
                                  lam_dyn_km1, lam_eq_k, lam_eq_ineq_k, 
                                  &hess.RSQrqt[k].mat(), k);

                // Evaluate FuFxt terms
                if (k < info.dims.K - 1) {
                    ocp_->eval_FuFxt(inputs_k, states_k, states_kp1, lam_dyn_k,
                                     &hess.FuFxt[k].mat(), k);
                }
                */

                // NEW APPROACH
                const Scalar *states_kp1 = (k < info.dims.K - 1) ? primal_x_p + info.offsets_primal_x[k + 1] : nullptr;
                ocp_->eval_RSQrqt(&objective_scale, inputs_k, states_k, states_kp1,
                                  lam_dyn_k, lam_eq_k, lam_eq_ineq_k,
                                  &hess.RSQrqt[k].mat(), 
                                  (k < info.dims.K - 1) ? &hess.RSQrqt[k+1].mat() : nullptr,
                                  (k < info.dims.K - 1) ? &hess.FuFx[k].mat() : nullptr,
                                  k);
            } else {
                ocp_->eval_RSQrqt(&objective_scale, inputs_k, states_k, lam_dyn_k, lam_eq_k,
                                lam_eq_ineq_k, &hess.RSQrqt[k].mat(), k);
            }
        }
        return 0;
    }

    template <typename OcpAbstractTag, typename ProblemType>
    Index
    NlpOcpTpl<OcpAbstractTag, ProblemType>::eval_constr_jac(const OcpInfo &info, const VecRealView &primal_x,
                                               const VecRealView &primal_s, Jacobian<ProblemType> &jac)
    {
        const Scalar *primal_x_p = primal_x.data();
        for (Index k = 0; k < info.dims.K; k++)
        {
            const Scalar *inputs_k = primal_x_p + info.offsets_primal_u[k];
            const Scalar *states_k = primal_x_p + info.offsets_primal_x[k];
            ocp_->eval_Ggt(inputs_k, states_k, &jac.Gg_eqt[k].mat(), k);
            ocp_->eval_Ggt_ineq(inputs_k, states_k, &jac.Gg_ineqt[k].mat(), k);
            if (k != info.dims.K - 1)
            {
                const Scalar *states_kp1 = primal_x_p + info.offsets_primal_x[k + 1];

                // OLD APPROACH
                /*
                ocp_->eval_BAbt(states_kp1, inputs_k, states_k, &jac.BAbt[k].mat(), k);

                if constexpr (std::is_same_v<ProblemType, ImplicitOcpType>)
                {
                    // For Implicit OCP, we need to evaluate the Jt part
                    ocp_->eval_Jt(states_kp1, inputs_k, states_k, &jac.Jt[k].mat(), k);
                    ocp_->eval_Jt_inv(states_kp1, inputs_k, states_k, &jac.Jt_inv[k].mat(), k);
                }
                */

                if constexpr (std::is_same_v<ProblemType, ImplicitOcpType>)
                {
                    // For Implicit OCP, we need to evaluate the BAJbt part
                    ocp_->eval_BAJbt(states_kp1, inputs_k, states_k, &jac.BAbt[k].mat(), 
                                     &jac.Jt[k].mat(), &jac.Jt_inv[k].mat(), k);
                } else {
                    ocp_->eval_BAbt(states_kp1, inputs_k, states_k, &jac.BAbt[k].mat(), k);
                }
            }
        }
        return 0;
    }

    template <typename OcpAbstractTag, typename ProblemType>
    Index NlpOcpTpl<OcpAbstractTag, ProblemType>::eval_constraint_violation(const OcpInfo &info,
                                                               const VecRealView &primal_x,
                                                               const VecRealView &primal_s,
                                                               VecRealView &res)
    {
        const Scalar *primal_x_p = primal_x.data();
        Scalar *res_p = res.data();
        for (Index k = 0; k < info.dims.K; k++)
        {
            const Scalar *inputs_k = primal_x_p + info.offsets_primal_u[k];
            const Scalar *states_k = primal_x_p + info.offsets_primal_x[k];
            ocp_->eval_g(inputs_k, states_k, res_p + info.offsets_g_eq_path[k], k);
            ocp_->eval_gineq(inputs_k, states_k, res_p + info.offsets_g_eq_slack[k], k);
            if (k != info.dims.K - 1)
            {
                const Scalar *states_kp1 = primal_x_p + info.offsets_primal_x[k + 1];
                ocp_->eval_b(states_kp1, inputs_k, states_k, res_p + info.offsets_g_eq_dyn[k], k);
            }
        }
        // add -s to the slack constraints
        res.block(info.number_of_g_eq_slack, info.offset_g_eq_slack) =
            res.block(info.number_of_g_eq_slack, info.offset_g_eq_slack) -
            primal_s.block(info.number_of_g_eq_slack, 0);
        return 0;
    }

    template <typename OcpAbstractTag, typename ProblemType>
    Index NlpOcpTpl<OcpAbstractTag, ProblemType>::eval_objective_gradient(
        const OcpInfo &info, const Scalar objective_scale, const VecRealView &primal_x,
        const VecRealView &primal_s, VecRealView &grad_x, VecRealView &grad_s)
    {
        grad_s.block(info.number_of_g_eq_slack, 0) = 0;
        const Scalar *primal_x_p = primal_x.data();
        Scalar *grad_x_p = grad_x.data();
        for (Index k = 0; k < info.dims.K; k++)
        {
            const Scalar *inputs_k = primal_x_p + info.offsets_primal_u[k];
            const Scalar *states_k = primal_x_p + info.offsets_primal_x[k];
            ocp_->eval_rq(&objective_scale, inputs_k, states_k, grad_x_p + info.offsets_primal_u[k],
                          k);
        }
        return 0;
    }

    template <typename OcpAbstractTag, typename ProblemType>
    Index NlpOcpTpl<OcpAbstractTag, ProblemType>::eval_objective(const OcpInfo &info,
                                                    const Scalar objective_scale,
                                                    const VecRealView &primal_x,
                                                    const VecRealView &primal_s, Scalar &res)
    {
        res = 0;
        const Scalar *primal_x_p = primal_x.data();
        for (Index k = 0; k < info.dims.K; k++)
        {
            Scalar ret = 0;
            const Scalar *inputs_k = primal_x_p + info.offsets_primal_u[k];
            const Scalar *states_k = primal_x_p + info.offsets_primal_x[k];
            ocp_->eval_L(&objective_scale, inputs_k, states_k, &ret, k);
            res += ret;
        }
        return 0;
    }
    template <typename OcpAbstractTag, typename ProblemType>
    Index NlpOcpTpl<OcpAbstractTag, ProblemType>::get_bounds(const OcpInfo &info, VecRealView &lower_bounds,
                                                VecRealView &upper_bounds)
    {
        if (info.number_of_slack_variables == 0)
            return 0;
        Scalar *lower_bounds_p = lower_bounds.data();
        Scalar *upper_bounds_p = upper_bounds.data();
        for (Index k = 0; k < info.dims.K; k++)
        {
            Scalar *lower_bounds_k = lower_bounds_p + info.offsets_slack[k];
            Scalar *upper_bounds_k = upper_bounds_p + info.offsets_slack[k];
            ocp_->get_bounds(lower_bounds_k, upper_bounds_k, k);
        }
        return 0;
    }

    template <typename OcpAbstractTag, typename ProblemType>
    Index NlpOcpTpl<OcpAbstractTag, ProblemType>::get_initial_primal(const ProblemInfo &info,
                                                        VecRealView &primal_x)
    {
        Scalar *primal_x_ptr = primal_x.data();
        for (Index k = 0; k < info.dims.K; k++)
        {
            ocp_->get_initial_uk(primal_x_ptr + info.offsets_primal_u[k], k);
            ocp_->get_initial_xk(primal_x_ptr + info.offsets_primal_x[k], k);
        }
        return 0;
    }
    template <typename OcpAbstractTag, typename ProblemType>
    void NlpOcpTpl<OcpAbstractTag, ProblemType>::get_primal_damping(const ProblemInfo &info,
                                                       VecRealView &damping)
    {
        damping = 0;
    }
    template <typename OcpAbstractTag, typename ProblemType>
    void NlpOcpTpl<OcpAbstractTag, ProblemType>::apply_jacobian_s_transpose(const ProblemInfo &info,
                                                               const VecRealView &multipliers,
                                                               const Scalar alpha,
                                                               const VecRealView &y,
                                                               VecRealView &out)
    {
        out = alpha * y;
        out.block(info.number_of_slack_variables, 0) =
            out.block(info.number_of_slack_variables, 0) -
            multipliers.block(info.number_of_slack_variables, info.offset_g_eq_slack);
    }
}

#endif //__fatrop_ocp_nlp_ocp_hxx__
