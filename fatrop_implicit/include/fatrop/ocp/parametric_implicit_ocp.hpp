//
// Copyright (c) 2026
//

#ifndef __fatrop_ocp_parametric_implicit_ocp_hpp__
#define __fatrop_ocp_parametric_implicit_ocp_hpp__

#include "fatrop/context/context.hpp"
#include "fatrop/linear_algebra/fwd.hpp"

namespace fatrop
{
    /**
     * @brief User interface for an implicit OCP with optional one-copy global parameters.
     *
     * The primal vector is ordered as
     * `[u_0, x_0, ..., u_{K-1}, x_{K-1}, p]`.  The parameter vector `p`
     * is stored once and is visible to every stage. `get_np()` may return zero,
     * allowing parameter-free and parametric models to share one API. A
     * transition residual may
     * be a general implicit relation
     *
     *     b_k(x_{k+1}, u_k, x_k, p) = 0.
     *
     * Stage Jacobians use the same transposed BLASFEO convention as
     * ImplicitOcpAbstract.  Parameter Jacobians use the conventional
     * constraint-by-parameter orientation.
     */
    class ParametricImplicitOcpAbstract
    {
    public:
        virtual Index get_nx(Index k) const = 0;
        virtual Index get_nu(Index k) const = 0;
        virtual Index get_ng(Index k) const = 0;
        virtual Index get_ng_ineq(Index k) const = 0;
        virtual Index get_np() const = 0;
        virtual Index get_horizon_length() const = 0;

        virtual Index eval_BAJbt(
            const Scalar *states_kp1, const Scalar *inputs_k,
            const Scalar *states_k, const Scalar *parameters,
            MAT *res_BAt, MAT *res_Jt, MAT *res_Jt_inv, Index k) = 0;

        /** Fill db_k/dp, shape nx[k+1] by np. */
        virtual Index eval_dynamics_parameter_jacobian(
            const Scalar *states_kp1, const Scalar *inputs_k,
            const Scalar *states_k, const Scalar *parameters,
            MAT *res, Index k) = 0;

        virtual Index eval_RSQrqt(
            const Scalar *objective_scale, const Scalar *inputs_k,
            const Scalar *states_k, const Scalar *states_kp1,
            const Scalar *parameters, const Scalar *lam_dyn_k,
            const Scalar *lam_eq_k, const Scalar *lam_eq_ineq_k,
            MAT *res, MAT *res_kp1, MAT *res_FuFxt, Index k) = 0;

        /**
         * Fill the parameter blocks of the stage Lagrangian Hessian.
         * `res_ux_p` has shape (nu[k]+nx[k]) by np, `res_xnext_p`
         * has shape nx[k+1] by np (and is null at the terminal stage), and
         * `res_pp` has shape np by np.  Each call returns this stage's
         * contribution; FATROP accumulates the blocks across the horizon.
         */
        virtual Index eval_parameter_hessian(
            const Scalar *objective_scale, const Scalar *inputs_k,
            const Scalar *states_k, const Scalar *states_kp1,
            const Scalar *parameters, const Scalar *lam_dyn_k,
            const Scalar *lam_eq_k, const Scalar *lam_eq_ineq_k,
            MAT *res_ux_p, MAT *res_xnext_p, MAT *res_pp,
            Index k) = 0;

        virtual Index eval_Ggt(
            const Scalar *inputs_k, const Scalar *states_k,
            const Scalar *parameters, MAT *res, Index k) = 0;
        virtual Index eval_Ggt_ineq(
            const Scalar *inputs_k, const Scalar *states_k,
            const Scalar *parameters, MAT *res, Index k) = 0;

        /** Fill dg_k/dp, shape ng[k] by np. */
        virtual Index eval_equality_parameter_jacobian(
            const Scalar *inputs_k, const Scalar *states_k,
            const Scalar *parameters, MAT *res, Index k) = 0;
        /** Fill dg_ineq,k/dp, shape ng_ineq[k] by np. */
        virtual Index eval_inequality_parameter_jacobian(
            const Scalar *inputs_k, const Scalar *states_k,
            const Scalar *parameters, MAT *res, Index k) = 0;

        virtual Index eval_b(
            const Scalar *states_kp1, const Scalar *inputs_k,
            const Scalar *states_k, const Scalar *parameters,
            Scalar *res, Index k) = 0;
        virtual Index eval_g(
            const Scalar *inputs_k, const Scalar *states_k,
            const Scalar *parameters, Scalar *res, Index k) = 0;
        virtual Index eval_gineq(
            const Scalar *inputs_k, const Scalar *states_k,
            const Scalar *parameters, Scalar *res, Index k) = 0;

        virtual Index eval_rq(
            const Scalar *objective_scale, const Scalar *inputs_k,
            const Scalar *states_k, const Scalar *parameters,
            Scalar *res, Index k) = 0;
        /** Fill this stage's objective gradient with respect to p. */
        virtual Index eval_rp(
            const Scalar *objective_scale, const Scalar *inputs_k,
            const Scalar *states_k, const Scalar *parameters,
            Scalar *res, Index k) = 0;
        virtual Index eval_L(
            const Scalar *objective_scale, const Scalar *inputs_k,
            const Scalar *states_k, const Scalar *parameters,
            Scalar *res, Index k) = 0;

        virtual Index get_bounds(
            Scalar *lower, Scalar *upper, Index k) const = 0;

        /**
         * Optional box bounds for the local decision vector `[u_k, x_k]`.
         * The adapter represents them as stage-local identity inequalities,
         * so they preserve the OCP chain and participate in the ordinary
         * primal-dual barrier and restoration machinery.
         */
        virtual bool has_stage_variable_bounds(Index k) const
        {
            (void)k;
            return false;
        }
        virtual Index get_stage_variable_bounds(
            Scalar *lower, Scalar *upper, Index k) const
        {
            (void)lower;
            (void)upper;
            (void)k;
            return 0;
        }

        virtual Index get_initial_xk(Scalar *xk, Index k) const = 0;
        virtual Index get_initial_uk(Scalar *uk, Index k) const = 0;
        virtual Index get_initial_parameters(Scalar *parameters) const = 0;

        /**
         * Parameter box bounds are represented internally as one terminal
         * vector inequality, preserving the OCP-plus-border structure.
         */
        virtual bool has_parameter_bounds() const { return false; }
        virtual Index get_parameter_bounds(
            Scalar *lower, Scalar *upper) const
        {
            (void)lower;
            (void)upper;
            return 0;
        }

        virtual ~ParametricImplicitOcpAbstract() = default;
    };
} // namespace fatrop

#endif // __fatrop_ocp_parametric_implicit_ocp_hpp__
