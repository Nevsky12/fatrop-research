//
// Copyright (c) 2026
//

#ifndef __fatrop_ocp_parametric_implicit_nlp_ocp_hpp__
#define __fatrop_ocp_parametric_implicit_nlp_ocp_hpp__

#include "fatrop/linear_algebra/matrix.hpp"
#include "fatrop/nlp/dims.hpp"
#include "fatrop/nlp/nlp.hpp"
#include "fatrop/ocp/dims.hpp"
#include "fatrop/ocp/parametric_implicit_ocp.hpp"
#include "fatrop/ocp/type.hpp"

#include <memory>
#include <vector>

namespace fatrop
{
    /**
     * @brief Adapts ParametricImplicitOcpAbstract to FATROP's native NLP API.
     *
     * Both `np == 0` and one-copy `np > 0` problems are accepted.
     */
    class ParametricImplicitNlpOcp final : public Nlp<ImplicitOcpType>
    {
    public:
        explicit ParametricImplicitNlpOcp(
            std::shared_ptr<ParametricImplicitOcpAbstract> ocp);

        const NlpDims &nlp_dims() const override { return nlp_dims_; }
        const ProblemDims &problem_dims() const override { return ocp_dims_; }

        Index eval_lag_hess(
            const ProblemInfo &info, Scalar objective_scale,
            const VecRealView &primal_x, const VecRealView &primal_s,
            const VecRealView &lam, Hessian<ImplicitOcpType> &hess) override;
        Index eval_constr_jac(
            const ProblemInfo &info, const VecRealView &primal_x,
            const VecRealView &primal_s,
            Jacobian<ImplicitOcpType> &jac) override;
        Index eval_constraint_violation(
            const ProblemInfo &info, const VecRealView &primal_x,
            const VecRealView &primal_s, VecRealView &res) override;
        Index eval_objective_gradient(
            const ProblemInfo &info, Scalar objective_scale,
            const VecRealView &primal_x, const VecRealView &primal_s,
            VecRealView &grad_x, VecRealView &grad_s) override;
        Index eval_objective(
            const ProblemInfo &info, Scalar objective_scale,
            const VecRealView &primal_x, const VecRealView &primal_s,
            Scalar &res) override;
        Index get_bounds(
            const ProblemInfo &info, VecRealView &lower_bounds,
            VecRealView &upper_bounds) override;
        Index get_initial_primal(
            const ProblemInfo &info, VecRealView &primal_x) override;
        void get_primal_damping(
            const ProblemInfo &info, VecRealView &damping) override;
        void apply_jacobian_s_transpose(
            const ProblemInfo &info, const VecRealView &multipliers,
            Scalar alpha, const VecRealView &y, VecRealView &out) override;

    private:
        static ProblemDims make_problem_dims(
            const ParametricImplicitOcpAbstract &ocp);
        static NlpDims make_nlp_dims(const ProblemDims &dims);
        Index user_inequalities(Index stage) const;
        Index stage_variable_bounds(Index stage) const;
        Index parameter_bound_offset(const ProblemInfo &info) const;

        std::shared_ptr<ParametricImplicitOcpAbstract> ocp_;
        ProblemDims ocp_dims_;
        NlpDims nlp_dims_;
        bool parameter_bounds_;
        std::vector<bool> stage_variable_bounds_;

        std::vector<MatRealAllocated> dynamics_parameter_jacobian_;
        std::vector<MatRealAllocated> equality_parameter_jacobian_;
        std::vector<MatRealAllocated> inequality_parameter_jacobian_;
        std::vector<MatRealAllocated> stage_parameter_cross_hessian_;
        std::vector<MatRealAllocated> next_state_parameter_cross_hessian_;
        std::vector<MatRealAllocated> stage_parameter_hessian_;
        VecRealAllocated stage_parameter_gradient_;
    };
} // namespace fatrop

#endif // __fatrop_ocp_parametric_implicit_nlp_ocp_hpp__
