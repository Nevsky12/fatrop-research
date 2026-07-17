//
// Copyright (c) 2026
//

#include "fatrop/ocp/parametric_implicit_nlp_ocp.hpp"

#include "fatrop/common/exception.hpp"
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/problem_info.hpp"

#include <algorithm>
#include <numeric>
#include <utility>

namespace fatrop
{
    namespace
    {
        const ParametricImplicitOcpAbstract &require_ocp(
            const std::shared_ptr<ParametricImplicitOcpAbstract> &ocp)
        {
            fatrop_assert_msg(ocp != nullptr, "The parametric OCP must not be null.");
            return *ocp;
        }
    }

    ProblemDims ParametricImplicitNlpOcp::make_problem_dims(
        const ParametricImplicitOcpAbstract &ocp)
    {
        const Index stages = ocp.get_horizon_length();
        fatrop_assert_msg(stages > 0, "The OCP horizon must be positive.");
        fatrop_assert_msg(
            ocp.get_np() >= 0,
            "The number of global parameters must be non-negative.");

        std::vector<Index> controls(stages);
        std::vector<Index> states(stages);
        std::vector<Index> equalities(stages);
        std::vector<Index> inequalities(stages);
        for (Index stage = 0; stage < stages; ++stage)
        {
            controls[stage] = ocp.get_nu(stage);
            states[stage] = ocp.get_nx(stage);
            equalities[stage] = ocp.get_ng(stage);
            inequalities[stage] = ocp.get_ng_ineq(stage);
            if (ocp.has_stage_variable_bounds(stage))
                inequalities[stage] += controls[stage] + states[stage];
        }
        if (ocp.has_parameter_bounds())
            inequalities.back() += ocp.get_np();

        return ProblemDims(
            stages, std::move(controls), std::move(states),
            std::move(equalities), std::move(inequalities), ocp.get_np());
    }

    NlpDims ParametricImplicitNlpOcp::make_nlp_dims(
        const ProblemDims &dims)
    {
        Index variables = dims.number_of_global_parameters;
        Index equalities = 0;
        Index inequalities = 0;
        for (Index stage = 0; stage < dims.K; ++stage)
        {
            variables += dims.number_of_controls[stage]
                       + dims.number_of_states[stage];
            equalities += dims.number_of_eq_constraints[stage]
                        + dims.number_of_ineq_constraints[stage];
            inequalities += dims.number_of_ineq_constraints[stage];
            if (stage + 1 < dims.K)
                equalities += dims.number_of_states[stage + 1];
        }
        return NlpDims(variables, equalities, inequalities);
    }

    ParametricImplicitNlpOcp::ParametricImplicitNlpOcp(
        std::shared_ptr<ParametricImplicitOcpAbstract> ocp)
        : ocp_(std::move(ocp)),
          ocp_dims_(make_problem_dims(require_ocp(ocp_))),
          nlp_dims_(make_nlp_dims(ocp_dims_)),
          parameter_bounds_(
              ocp_dims_.number_of_global_parameters > 0
              && ocp_->has_parameter_bounds()),
          stage_variable_bounds_(
              static_cast<std::size_t>(ocp_dims_.K), false),
          stage_parameter_gradient_(std::max<Index>(
              ocp_dims_.number_of_global_parameters, 1))
    {
        const Index parameters = ocp_dims_.number_of_global_parameters;
        const Index parameter_storage = std::max<Index>(parameters, 1);
        dynamics_parameter_jacobian_.reserve(ocp_dims_.K - 1);
        equality_parameter_jacobian_.reserve(ocp_dims_.K);
        inequality_parameter_jacobian_.reserve(ocp_dims_.K);
        stage_parameter_cross_hessian_.reserve(ocp_dims_.K);
        next_state_parameter_cross_hessian_.reserve(ocp_dims_.K - 1);
        stage_parameter_hessian_.reserve(ocp_dims_.K);

        for (Index stage = 0; stage < ocp_dims_.K; ++stage)
        {
            stage_variable_bounds_[static_cast<std::size_t>(stage)] =
                ocp_->has_stage_variable_bounds(stage);
            const Index stage_variables =
                ocp_dims_.number_of_controls[stage]
                + ocp_dims_.number_of_states[stage];
            equality_parameter_jacobian_.emplace_back(
                std::max<Index>(ocp_->get_ng(stage), 1), parameter_storage);
            inequality_parameter_jacobian_.emplace_back(
                std::max<Index>(ocp_->get_ng_ineq(stage), 1), parameter_storage);
            stage_parameter_cross_hessian_.emplace_back(
                std::max<Index>(stage_variables, 1), parameter_storage);
            stage_parameter_hessian_.emplace_back(
                parameter_storage, parameter_storage);
            if (stage + 1 < ocp_dims_.K)
            {
                const Index next_states = ocp_dims_.number_of_states[stage + 1];
                dynamics_parameter_jacobian_.emplace_back(
                    std::max<Index>(next_states, 1), parameter_storage);
                next_state_parameter_cross_hessian_.emplace_back(
                    std::max<Index>(next_states, 1), parameter_storage);
            }
        }
    }

    Index ParametricImplicitNlpOcp::user_inequalities(Index stage) const
    {
        return ocp_->get_ng_ineq(stage);
    }

    Index ParametricImplicitNlpOcp::stage_variable_bounds(Index stage) const
    {
        return stage_variable_bounds_[static_cast<std::size_t>(stage)]
            ? ocp_dims_.number_of_controls[stage]
              + ocp_dims_.number_of_states[stage]
            : 0;
    }

    Index ParametricImplicitNlpOcp::parameter_bound_offset(
        const ProblemInfo &info) const
    {
        const Index terminal = info.dims.K - 1;
        return info.offsets_g_eq_slack[terminal]
             + user_inequalities(terminal)
             + stage_variable_bounds(terminal);
    }

    Index ParametricImplicitNlpOcp::eval_lag_hess(
        const ProblemInfo &info, Scalar objective_scale,
        const VecRealView &primal_x, const VecRealView &primal_s,
        const VecRealView &lam, Hessian<ImplicitOcpType> &hess)
    {
        (void)primal_s;
        hess.set_zero();
        for (auto &block : hess.FuFx)
            block = 0.0;
        for (auto &block : hess.GuGx)
            block = 0.0;
        const Scalar *const primal = primal_x.data();
        const Scalar *const multipliers = lam.data();
        const Scalar *const parameters = primal + info.offset_primal_global;
        const Index np = info.number_of_global_parameters;

        for (Index stage = 0; stage < info.dims.K; ++stage)
        {
            const Scalar *const inputs = primal + info.offsets_primal_u[stage];
            const Scalar *const states = primal + info.offsets_primal_x[stage];
            const Scalar *const next_states = stage + 1 < info.dims.K
                ? primal + info.offsets_primal_x[stage + 1] : nullptr;
            const Scalar *const lam_dyn = stage + 1 < info.dims.K
                ? multipliers + info.offsets_g_eq_dyn[stage] : nullptr;
            const Scalar *const lam_eq =
                multipliers + info.offsets_g_eq_path[stage];
            const Scalar *const lam_ineq =
                multipliers + info.offsets_g_eq_slack[stage];

            Index status = ocp_->eval_RSQrqt(
                &objective_scale, inputs, states, next_states, parameters,
                lam_dyn, lam_eq, lam_ineq, &hess.RSQrqt[stage].mat(),
                stage + 1 < info.dims.K ? &hess.RSQrqt[stage + 1].mat() : nullptr,
                stage + 1 < info.dims.K ? &hess.FuFx[stage].mat() : nullptr,
                stage);
            if (status != 0)
                return status;

            if (np > 0)
            {
                MatRealAllocated &stage_cross =
                    stage_parameter_cross_hessian_[stage];
                MatRealAllocated &stage_pp = stage_parameter_hessian_[stage];
                stage_cross = 0.0;
                stage_pp = 0.0;
                MAT *next_cross = nullptr;
                if (stage + 1 < info.dims.K)
                {
                    next_state_parameter_cross_hessian_[stage] = 0.0;
                    next_cross =
                        &next_state_parameter_cross_hessian_[stage].mat();
                }
                status = ocp_->eval_parameter_hessian(
                    &objective_scale, inputs, states, next_states, parameters,
                    lam_dyn, lam_eq, lam_ineq, &stage_cross.mat(), next_cross,
                    &stage_pp.mat(), stage);
                if (status != 0)
                    return status;

                const Index stage_variables =
                    info.dims.number_of_controls[stage]
                    + info.dims.number_of_states[stage];
                for (Index row = 0; row < stage_variables; ++row)
                    for (Index parameter = 0; parameter < np; ++parameter)
                        hess.global_parameter_cross_hessian(
                            info.offsets_primal_u[stage] + row, parameter)
                            += stage_cross(row, parameter);
                if (stage + 1 < info.dims.K)
                {
                    const Index next_nx = info.dims.number_of_states[stage + 1];
                    const MatRealAllocated &cross_next =
                        next_state_parameter_cross_hessian_[stage];
                    for (Index row = 0; row < next_nx; ++row)
                        for (Index parameter = 0; parameter < np; ++parameter)
                            hess.global_parameter_cross_hessian(
                                info.offsets_primal_x[stage + 1] + row, parameter)
                                += cross_next(row, parameter);
                }
                for (Index row = 0; row < np; ++row)
                    for (Index column = 0; column < np; ++column)
                        hess.global_parameter_hessian(row, column)
                            += stage_pp(row, column);
            }
        }
        return 0;
    }

    Index ParametricImplicitNlpOcp::eval_constr_jac(
        const ProblemInfo &info, const VecRealView &primal_x,
        const VecRealView &primal_s, Jacobian<ImplicitOcpType> &jac)
    {
        (void)primal_s;
        jac.global_parameter_jacobian = 0.0;
        const Scalar *const primal = primal_x.data();
        const Scalar *const parameters = primal + info.offset_primal_global;
        const Index np = info.number_of_global_parameters;

        for (Index stage = 0; stage < info.dims.K; ++stage)
        {
            jac.Gg_eqt[stage] = 0.0;
            jac.Gg_ineqt[stage] = 0.0;
            const Scalar *const inputs = primal + info.offsets_primal_u[stage];
            const Scalar *const states = primal + info.offsets_primal_x[stage];

            const Index ng = info.dims.number_of_eq_constraints[stage];
            if (ng > 0)
            {
                Index status = ocp_->eval_Ggt(
                    inputs, states, parameters, &jac.Gg_eqt[stage].mat(), stage);
                if (status != 0)
                    return status;
                if (np > 0)
                {
                    equality_parameter_jacobian_[stage] = 0.0;
                    status = ocp_->eval_equality_parameter_jacobian(
                        inputs, states, parameters,
                        &equality_parameter_jacobian_[stage].mat(), stage);
                    if (status != 0)
                        return status;
                    for (Index row = 0; row < ng; ++row)
                        for (Index parameter = 0; parameter < np; ++parameter)
                            jac.global_parameter_jacobian(
                                info.offsets_g_eq_path[stage] + row, parameter)
                                = equality_parameter_jacobian_[stage](row, parameter);
                }
            }

            const Index ng_ineq = user_inequalities(stage);
            if (ng_ineq > 0)
            {
                Index status = ocp_->eval_Ggt_ineq(
                    inputs, states, parameters,
                    &jac.Gg_ineqt[stage].mat(), stage);
                if (status != 0)
                    return status;
                if (np > 0)
                {
                    inequality_parameter_jacobian_[stage] = 0.0;
                    status = ocp_->eval_inequality_parameter_jacobian(
                        inputs, states, parameters,
                        &inequality_parameter_jacobian_[stage].mat(), stage);
                    if (status != 0)
                        return status;
                    for (Index row = 0; row < ng_ineq; ++row)
                        for (Index parameter = 0; parameter < np; ++parameter)
                            jac.global_parameter_jacobian(
                                info.offsets_g_eq_slack[stage] + row, parameter)
                                = inequality_parameter_jacobian_[stage](row, parameter);
                }
            }

            const Index variable_bounds = stage_variable_bounds(stage);
            for (Index variable = 0; variable < variable_bounds; ++variable)
            {
                jac.Gg_ineqt[stage](
                    variable, ng_ineq + variable) = 1.0;
            }

            if (stage + 1 < info.dims.K)
            {
                jac.BAbt[stage] = 0.0;
                jac.Jt[stage] = 0.0;
                jac.Jt_inv[stage] = 0.0;
                const Scalar *const next_states =
                    primal + info.offsets_primal_x[stage + 1];
                Index status = ocp_->eval_BAJbt(
                    next_states, inputs, states, parameters,
                    &jac.BAbt[stage].mat(), &jac.Jt[stage].mat(),
                    &jac.Jt_inv[stage].mat(), stage);
                if (status != 0)
                    return status;
                if (np > 0)
                {
                    dynamics_parameter_jacobian_[stage] = 0.0;
                    status = ocp_->eval_dynamics_parameter_jacobian(
                        next_states, inputs, states, parameters,
                        &dynamics_parameter_jacobian_[stage].mat(), stage);
                    if (status != 0)
                        return status;
                    const Index next_nx = info.dims.number_of_states[stage + 1];
                    for (Index row = 0; row < next_nx; ++row)
                        for (Index parameter = 0; parameter < np; ++parameter)
                            jac.global_parameter_jacobian(
                                info.offsets_g_eq_dyn[stage] + row, parameter)
                                = dynamics_parameter_jacobian_[stage](row, parameter);
                }
            }
        }

        if (parameter_bounds_)
        {
            const Index offset = parameter_bound_offset(info);
            for (Index parameter = 0; parameter < np; ++parameter)
                jac.global_parameter_jacobian(offset + parameter, parameter) = 1.0;
        }
        return 0;
    }

    Index ParametricImplicitNlpOcp::eval_constraint_violation(
        const ProblemInfo &info, const VecRealView &primal_x,
        const VecRealView &primal_s, VecRealView &res)
    {
        res = 0.0;
        const Scalar *const primal = primal_x.data();
        const Scalar *const parameters = primal + info.offset_primal_global;
        for (Index stage = 0; stage < info.dims.K; ++stage)
        {
            const Scalar *const inputs = primal + info.offsets_primal_u[stage];
            const Scalar *const states = primal + info.offsets_primal_x[stage];
            if (info.dims.number_of_eq_constraints[stage] > 0)
            {
                const Index status = ocp_->eval_g(
                    inputs, states, parameters,
                    res.data() + info.offsets_g_eq_path[stage], stage);
                if (status != 0)
                    return status;
            }
            if (user_inequalities(stage) > 0)
            {
                const Index status = ocp_->eval_gineq(
                    inputs, states, parameters,
                    res.data() + info.offsets_g_eq_slack[stage], stage);
                if (status != 0)
                    return status;
            }
            const Index variable_bounds = stage_variable_bounds(stage);
            const Index variable_offset = info.offsets_g_eq_slack[stage]
                                        + user_inequalities(stage);
            for (Index variable = 0; variable < variable_bounds; ++variable)
                res(variable_offset + variable) = inputs[variable];
            if (stage + 1 < info.dims.K)
            {
                const Scalar *const next_states =
                    primal + info.offsets_primal_x[stage + 1];
                const Index status = ocp_->eval_b(
                    next_states, inputs, states, parameters,
                    res.data() + info.offsets_g_eq_dyn[stage], stage);
                if (status != 0)
                    return status;
            }
        }
        if (parameter_bounds_)
        {
            const Index offset = parameter_bound_offset(info);
            for (Index parameter = 0;
                 parameter < info.number_of_global_parameters; ++parameter)
                res(offset + parameter) = parameters[parameter];
        }
        res.block(info.number_of_g_eq_slack, info.offset_g_eq_slack) =
            res.block(info.number_of_g_eq_slack, info.offset_g_eq_slack)
            - primal_s.block(info.number_of_g_eq_slack, 0);
        return 0;
    }

    Index ParametricImplicitNlpOcp::eval_objective_gradient(
        const ProblemInfo &info, Scalar objective_scale,
        const VecRealView &primal_x, const VecRealView &primal_s,
        VecRealView &grad_x, VecRealView &grad_s)
    {
        (void)primal_s;
        grad_x = 0.0;
        grad_s = 0.0;
        const Scalar *const primal = primal_x.data();
        const Scalar *const parameters = primal + info.offset_primal_global;
        for (Index stage = 0; stage < info.dims.K; ++stage)
        {
            const Scalar *const inputs = primal + info.offsets_primal_u[stage];
            const Scalar *const states = primal + info.offsets_primal_x[stage];
            Index status = ocp_->eval_rq(
                &objective_scale, inputs, states, parameters,
                grad_x.data() + info.offsets_primal_u[stage], stage);
            if (status != 0)
                return status;
            if (info.number_of_global_parameters > 0)
            {
                stage_parameter_gradient_ = 0.0;
                status = ocp_->eval_rp(
                    &objective_scale, inputs, states, parameters,
                    stage_parameter_gradient_.data(), stage);
                if (status != 0)
                    return status;
                for (Index parameter = 0;
                     parameter < info.number_of_global_parameters; ++parameter)
                    grad_x(info.offset_primal_global + parameter)
                        += stage_parameter_gradient_(parameter);
            }
        }
        return 0;
    }

    Index ParametricImplicitNlpOcp::eval_objective(
        const ProblemInfo &info, Scalar objective_scale,
        const VecRealView &primal_x, const VecRealView &primal_s,
        Scalar &res)
    {
        (void)primal_s;
        res = 0.0;
        const Scalar *const primal = primal_x.data();
        const Scalar *const parameters = primal + info.offset_primal_global;
        for (Index stage = 0; stage < info.dims.K; ++stage)
        {
            Scalar stage_value = 0.0;
            const Index status = ocp_->eval_L(
                &objective_scale,
                primal + info.offsets_primal_u[stage],
                primal + info.offsets_primal_x[stage], parameters,
                &stage_value, stage);
            if (status != 0)
                return status;
            res += stage_value;
        }
        return 0;
    }

    Index ParametricImplicitNlpOcp::get_bounds(
        const ProblemInfo &info, VecRealView &lower_bounds,
        VecRealView &upper_bounds)
    {
        for (Index stage = 0; stage < info.dims.K; ++stage)
        {
            const Index user_bounds = user_inequalities(stage);
            if (user_bounds > 0)
            {
                const Index status = ocp_->get_bounds(
                    lower_bounds.data() + info.offsets_slack[stage],
                    upper_bounds.data() + info.offsets_slack[stage], stage);
                if (status != 0)
                    return status;
            }
            if (stage_variable_bounds(stage) > 0)
            {
                const Index offset = info.offsets_slack[stage] + user_bounds;
                const Index status = ocp_->get_stage_variable_bounds(
                    lower_bounds.data() + offset,
                    upper_bounds.data() + offset, stage);
                if (status != 0)
                    return status;
            }
        }
        if (parameter_bounds_)
        {
            const Index terminal = info.dims.K - 1;
            const Index offset = info.offsets_slack[terminal]
                               + user_inequalities(terminal)
                               + stage_variable_bounds(terminal);
            return ocp_->get_parameter_bounds(
                lower_bounds.data() + offset, upper_bounds.data() + offset);
        }
        return 0;
    }

    Index ParametricImplicitNlpOcp::get_initial_primal(
        const ProblemInfo &info, VecRealView &primal_x)
    {
        primal_x = 0.0;
        for (Index stage = 0; stage < info.dims.K; ++stage)
        {
            Index status = ocp_->get_initial_uk(
                primal_x.data() + info.offsets_primal_u[stage], stage);
            if (status != 0)
                return status;
            status = ocp_->get_initial_xk(
                primal_x.data() + info.offsets_primal_x[stage], stage);
            if (status != 0)
                return status;
        }
        if (info.number_of_global_parameters > 0)
            return ocp_->get_initial_parameters(
                primal_x.data() + info.offset_primal_global);
        return 0;
    }

    void ParametricImplicitNlpOcp::get_primal_damping(
        const ProblemInfo &info, VecRealView &damping)
    {
        (void)info;
        damping = 0.0;
    }

    void ParametricImplicitNlpOcp::apply_jacobian_s_transpose(
        const ProblemInfo &info, const VecRealView &multipliers,
        Scalar alpha, const VecRealView &y, VecRealView &out)
    {
        out = alpha * y;
        out.block(info.number_of_slack_variables, 0) =
            out.block(info.number_of_slack_variables, 0)
            - multipliers.block(
                info.number_of_slack_variables, info.offset_g_eq_slack);
    }
} // namespace fatrop
