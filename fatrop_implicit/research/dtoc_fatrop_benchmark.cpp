#include "fatrop/fatrop.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

using fatrop::Index;
using fatrop::IpAlgBuilder;
using fatrop::IpAlgorithm;
using fatrop::IpSolverReturnFlag;
using fatrop::NlpOcp;
using fatrop::OcpAbstract;
using fatrop::OcpType;
using fatrop::OptionRegistry;
using fatrop::Scalar;
using fatrop::blasfeo_gese_wrap;
using fatrop::blasfeo_matel_wrap;

namespace
{

void clear_matrix(MAT *matrix)
{
    blasfeo_gese_wrap(
        matrix->m, matrix->n, 0.0, matrix, 0, 0);
}

class Dtoc1lProblem final : public OcpAbstract
{
public:
    Dtoc1lProblem(Index stages, Index controls, Index states)
        : stages_(stages), controls_(controls), states_(states)
    {
        if (stages_ < 2 || controls_ < 1 || states_ < 2)
            throw std::runtime_error("Invalid DTOC1L dimensions");
    }

    Index get_nx(Index) const override { return states_; }
    Index get_nu(Index k) const override
    {
        return k == stages_ - 1 ? 0 : controls_;
    }
    Index get_ng(Index k) const override
    {
        return k == 0 ? states_ : 0;
    }
    Index get_ng_ineq(Index) const override { return 0; }
    Index get_horizon_length() const override { return stages_; }

    Index eval_BAbt(const Scalar *, const Scalar *, const Scalar *,
                    MAT *result, Index) override
    {
        clear_matrix(result);
        for (Index state = 0; state < states_; ++state)
        {
            for (Index control = 0; control < controls_; ++control)
            {
                blasfeo_matel_wrap(result, control, state) =
                    static_cast<double>(state - control) /
                    static_cast<double>(controls_ + states_);
            }
            blasfeo_matel_wrap(result, controls_ + state, state) =
                0.5;
            if (state > 0)
                blasfeo_matel_wrap(
                    result, controls_ + state - 1, state) = -0.25;
            if (state + 1 < states_)
                blasfeo_matel_wrap(
                    result, controls_ + state + 1, state) = 0.25;
        }
        return 0;
    }

    Index eval_RSQrqt(
        const Scalar *objective_scale, const Scalar *inputs,
        const Scalar *states, const Scalar *, const Scalar *,
        const Scalar *, MAT *result, Index k) override
    {
        clear_matrix(result);
        const Index nu = get_nu(k);
        for (Index control = 0; control < nu; ++control)
        {
            const double difference = inputs[control] - 0.5;
            blasfeo_matel_wrap(result, control, control) =
                objective_scale[0] * 12.0 * difference * difference;
        }
        for (Index state = 0; state < states_; ++state)
        {
            const double difference = states[state] - 0.25;
            blasfeo_matel_wrap(
                result, nu + state, nu + state) =
                objective_scale[0] * 12.0 * difference * difference;
        }
        return 0;
    }

    Index eval_Ggt(const Scalar *, const Scalar *, MAT *result,
                   Index k) override
    {
        clear_matrix(result);
        if (k == 0)
        {
            for (Index state = 0; state < states_; ++state)
            {
                blasfeo_matel_wrap(
                    result, controls_ + state, state) = 1.0;
            }
        }
        return 0;
    }

    Index eval_Ggt_ineq(const Scalar *, const Scalar *, MAT *,
                        Index) override
    {
        return 0;
    }

    Index eval_b(const Scalar *states_next, const Scalar *inputs,
                 const Scalar *states, Scalar *result,
                 Index) override
    {
        for (Index state = 0; state < states_; ++state)
        {
            double value = -states_next[state] + 0.5 * states[state];
            if (state > 0)
                value -= 0.25 * states[state - 1];
            if (state + 1 < states_)
                value += 0.25 * states[state + 1];
            for (Index control = 0; control < controls_; ++control)
            {
                value +=
                    static_cast<double>(state - control) /
                    static_cast<double>(controls_ + states_) *
                    inputs[control];
            }
            result[state] = value;
        }
        return 0;
    }

    Index eval_g(const Scalar *, const Scalar *states, Scalar *result,
                 Index k) override
    {
        if (k == 0)
        {
            for (Index state = 0; state < states_; ++state)
                result[state] = states[state];
        }
        return 0;
    }

    Index eval_gineq(const Scalar *, const Scalar *, Scalar *,
                     Index) override
    {
        return 0;
    }

    Index eval_rq(const Scalar *objective_scale, const Scalar *inputs,
                  const Scalar *states, Scalar *result,
                  Index k) override
    {
        const Index nu = get_nu(k);
        for (Index control = 0; control < nu; ++control)
        {
            const double difference = inputs[control] - 0.5;
            result[control] =
                objective_scale[0] * 4.0 * difference * difference *
                difference;
        }
        for (Index state = 0; state < states_; ++state)
        {
            const double difference = states[state] - 0.25;
            result[nu + state] =
                objective_scale[0] * 4.0 * difference * difference *
                difference;
        }
        return 0;
    }

    Index eval_L(const Scalar *objective_scale, const Scalar *inputs,
                 const Scalar *states, Scalar *result,
                 Index k) override
    {
        double value = 0.0;
        const Index nu = get_nu(k);
        for (Index control = 0; control < nu; ++control)
        {
            const double difference = inputs[control] - 0.5;
            value += difference * difference * difference * difference;
        }
        for (Index state = 0; state < states_; ++state)
        {
            const double difference = states[state] - 0.25;
            value += difference * difference * difference * difference;
        }
        result[0] = objective_scale[0] * value;
        return 0;
    }

    Index get_bounds(Scalar *, Scalar *, Index) const override
    {
        return 0;
    }

    Index get_initial_xk(Scalar *states, Index) const override
    {
        std::fill(states, states + states_, 0.0);
        return 0;
    }

    Index get_initial_uk(Scalar *inputs, Index k) const override
    {
        std::fill(inputs, inputs + get_nu(k), 0.0);
        return 0;
    }

private:
    Index stages_;
    Index controls_;
    Index states_;
};

class Dtoc3Problem final : public OcpAbstract
{
public:
    explicit Dtoc3Problem(Index stages)
        : stages_(stages),
          inverse_horizon_(1.0 / static_cast<double>(stages))
    {
        if (stages_ < 2)
            throw std::runtime_error("Invalid DTOC3 horizon");
    }

    Index get_nx(Index) const override { return 2; }
    Index get_nu(Index k) const override
    {
        return k == stages_ - 1 ? 0 : 1;
    }
    Index get_ng(Index k) const override { return k == 0 ? 2 : 0; }
    Index get_ng_ineq(Index) const override { return 0; }
    Index get_horizon_length() const override { return stages_; }

    Index eval_BAbt(const Scalar *, const Scalar *, const Scalar *,
                    MAT *result, Index) override
    {
        clear_matrix(result);
        const double step = inverse_horizon_;
        blasfeo_matel_wrap(result, 1, 0) = 1.0;
        blasfeo_matel_wrap(result, 2, 0) = step;
        blasfeo_matel_wrap(result, 0, 1) = step;
        blasfeo_matel_wrap(result, 1, 1) = -step;
        blasfeo_matel_wrap(result, 2, 1) = 1.0;
        return 0;
    }

    Index eval_RSQrqt(
        const Scalar *objective_scale, const Scalar *, const Scalar *,
        const Scalar *, const Scalar *, const Scalar *, MAT *result,
        Index k) override
    {
        clear_matrix(result);
        const Index nu = get_nu(k);
        if (nu == 1)
        {
            blasfeo_matel_wrap(result, 0, 0) =
                objective_scale[0] * 6.0 * inverse_horizon_;
        }
        if (k > 0)
        {
            blasfeo_matel_wrap(result, nu, nu) =
                objective_scale[0] * 2.0 * inverse_horizon_;
            blasfeo_matel_wrap(result, nu + 1, nu + 1) =
                objective_scale[0] * inverse_horizon_;
        }
        return 0;
    }

    Index eval_Ggt(const Scalar *, const Scalar *, MAT *result,
                   Index k) override
    {
        clear_matrix(result);
        if (k == 0)
        {
            blasfeo_matel_wrap(result, 1, 0) = 1.0;
            blasfeo_matel_wrap(result, 2, 1) = 1.0;
        }
        return 0;
    }

    Index eval_Ggt_ineq(const Scalar *, const Scalar *, MAT *,
                        Index) override
    {
        return 0;
    }

    Index eval_b(const Scalar *states_next, const Scalar *inputs,
                 const Scalar *states, Scalar *result, Index) override
    {
        const double step = inverse_horizon_;
        result[0] =
            -states_next[0] + states[0] + step * states[1];
        result[1] =
            -states_next[1] + states[1] -
            step * states[0] + step * inputs[0];
        return 0;
    }

    Index eval_g(const Scalar *, const Scalar *states, Scalar *result,
                 Index k) override
    {
        if (k == 0)
        {
            result[0] = states[0] - 15.0;
            result[1] = states[1] - 5.0;
        }
        return 0;
    }

    Index eval_gineq(const Scalar *, const Scalar *, Scalar *,
                     Index) override
    {
        return 0;
    }

    Index eval_rq(const Scalar *objective_scale, const Scalar *inputs,
                  const Scalar *states, Scalar *result, Index k) override
    {
        const Index nu = get_nu(k);
        if (nu == 1)
        {
            result[0] =
                objective_scale[0] * 6.0 * inverse_horizon_ * inputs[0];
        }
        result[nu] =
            k > 0
                ? objective_scale[0] * 2.0 * inverse_horizon_ * states[0]
                : 0.0;
        result[nu + 1] =
            k > 0
                ? objective_scale[0] * inverse_horizon_ * states[1]
                : 0.0;
        return 0;
    }

    Index eval_L(const Scalar *objective_scale, const Scalar *inputs,
                 const Scalar *states, Scalar *result, Index k) override
    {
        double value = 0.0;
        if (k < stages_ - 1)
            value += 3.0 * inverse_horizon_ * inputs[0] * inputs[0];
        if (k > 0)
        {
            value += inverse_horizon_ * states[0] * states[0];
            value +=
                0.5 * inverse_horizon_ * states[1] * states[1];
        }
        result[0] = objective_scale[0] * value;
        return 0;
    }

    Index get_bounds(Scalar *, Scalar *, Index) const override
    {
        return 0;
    }

    Index get_initial_xk(Scalar *states, Index k) const override
    {
        states[0] = k == 0 ? 15.0 : 0.0;
        states[1] = k == 0 ? 5.0 : 0.0;
        return 0;
    }

    Index get_initial_uk(Scalar *inputs, Index k) const override
    {
        if (k < stages_ - 1)
            inputs[0] = 0.0;
        return 0;
    }

private:
    Index stages_;
    double inverse_horizon_;
};

class Dtoc6Problem final : public OcpAbstract
{
public:
    explicit Dtoc6Problem(Index stages) : stages_(stages)
    {
        if (stages_ < 2)
            throw std::runtime_error("Invalid DTOC6 horizon");
    }

    Index get_nx(Index) const override { return 1; }
    Index get_nu(Index k) const override
    {
        return k == stages_ - 1 ? 0 : 1;
    }
    Index get_ng(Index k) const override { return k == 0 ? 1 : 0; }
    Index get_ng_ineq(Index) const override { return 0; }
    Index get_horizon_length() const override { return stages_; }

    Index eval_BAbt(const Scalar *, const Scalar *inputs,
                    const Scalar *, MAT *result, Index) override
    {
        clear_matrix(result);
        blasfeo_matel_wrap(result, 0, 0) = std::exp(inputs[0]);
        blasfeo_matel_wrap(result, 1, 0) = 1.0;
        return 0;
    }

    Index eval_RSQrqt(
        const Scalar *objective_scale, const Scalar *inputs,
        const Scalar *states, const Scalar *lambda_dynamics,
        const Scalar *, const Scalar *, MAT *result,
        Index k) override
    {
        clear_matrix(result);
        if (k == stages_ - 1)
            return 0;
        const double exponential = std::exp(inputs[0]);
        const double residual = states[0] + exponential;
        blasfeo_matel_wrap(result, 0, 0) =
            objective_scale[0] *
                (1.0 + exponential * exponential +
                 residual * exponential) +
            lambda_dynamics[0] * exponential;
        blasfeo_matel_wrap(result, 0, 1) =
            objective_scale[0] * exponential;
        blasfeo_matel_wrap(result, 1, 0) =
            objective_scale[0] * exponential;
        blasfeo_matel_wrap(result, 1, 1) =
            objective_scale[0];
        return 0;
    }

    Index eval_Ggt(const Scalar *, const Scalar *, MAT *result,
                   Index k) override
    {
        clear_matrix(result);
        if (k == 0)
            blasfeo_matel_wrap(result, 1, 0) = 1.0;
        return 0;
    }

    Index eval_Ggt_ineq(const Scalar *, const Scalar *, MAT *,
                        Index) override
    {
        return 0;
    }

    Index eval_b(const Scalar *state_next, const Scalar *input,
                 const Scalar *state, Scalar *result,
                 Index) override
    {
        result[0] =
            -state_next[0] + state[0] + std::exp(input[0]);
        return 0;
    }

    Index eval_g(const Scalar *, const Scalar *state, Scalar *result,
                 Index k) override
    {
        if (k == 0)
            result[0] = state[0];
        return 0;
    }

    Index eval_gineq(const Scalar *, const Scalar *, Scalar *,
                     Index) override
    {
        return 0;
    }

    Index eval_rq(const Scalar *objective_scale, const Scalar *input,
                  const Scalar *state, Scalar *result,
                  Index k) override
    {
        if (k == stages_ - 1)
        {
            result[0] = 0.0;
            return 0;
        }
        const double exponential = std::exp(input[0]);
        const double residual = state[0] + exponential;
        result[0] =
            objective_scale[0] *
            (input[0] + residual * exponential);
        result[1] = objective_scale[0] * residual;
        return 0;
    }

    Index eval_L(const Scalar *objective_scale, const Scalar *input,
                 const Scalar *state, Scalar *result,
                 Index k) override
    {
        if (k == stages_ - 1)
        {
            result[0] = 0.0;
            return 0;
        }
        const double residual = state[0] + std::exp(input[0]);
        result[0] =
            objective_scale[0] *
            0.5 *
            (input[0] * input[0] + residual * residual);
        return 0;
    }

    Index get_bounds(Scalar *, Scalar *, Index) const override
    {
        return 0;
    }

    Index get_initial_xk(Scalar *state, Index) const override
    {
        state[0] = 0.0;
        return 0;
    }

    Index get_initial_uk(Scalar *input, Index k) const override
    {
        if (k < stages_ - 1)
            input[0] = 0.0;
        return 0;
    }

private:
    Index stages_;
};

double maximum_absolute(const fatrop::VecRealView &vector)
{
    double result = 0.0;
    for (Index i = 0; i < vector.m(); ++i)
        result = std::max(result, std::abs(vector(i)));
    return result;
}

} // namespace

int main(int argc, char **argv)
{
    std::string problem_name = "dtoc1l";
    Index stages = 1000;
    Index controls = 5;
    Index states = 10;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument(argv[i]);
        auto next_integer = [&](const char *name) {
            if (i + 1 >= argc)
                throw std::runtime_error(
                    std::string("Missing value for ") + name);
            return static_cast<Index>(std::stoll(argv[++i]));
        };
        if (argument == "--problem")
        {
            if (i + 1 >= argc)
                throw std::runtime_error(
                    "Missing value for --problem");
            problem_name = argv[++i];
        }
        else if (argument == "--stages")
            stages = next_integer("--stages");
        else if (argument == "--controls")
            controls = next_integer("--controls");
        else if (argument == "--states")
            states = next_integer("--states");
        else if (argument == "--help")
        {
            std::cout
                << "Usage: dtoc_fatrop_benchmark [options]\n"
                << "  --problem {dtoc1l|dtoc3|dtoc6}\n"
                << "  --stages N\n"
                << "  --controls N  (DTOC1L)\n"
                << "  --states N    (DTOC1L)\n";
            return 0;
        }
        else
            throw std::runtime_error("Unknown argument: " + argument);
    }

    try
    {
        std::shared_ptr<OcpAbstract> ocp;
        if (problem_name == "dtoc1l")
            ocp = std::make_shared<Dtoc1lProblem>(
                stages, controls, states);
        else if (problem_name == "dtoc3")
        {
            controls = 1;
            states = 2;
            ocp = std::make_shared<Dtoc3Problem>(stages);
        }
        else if (problem_name == "dtoc6")
        {
            controls = 1;
            states = 1;
            ocp = std::make_shared<Dtoc6Problem>(stages);
        }
        else
            throw std::runtime_error("Unknown DTOC problem");

        auto nlp = std::make_shared<NlpOcp>(ocp);
        OptionRegistry options;
        IpAlgBuilder<OcpType> builder(nlp);
        std::shared_ptr<IpAlgorithm<OcpType>> algorithm =
            builder.with_options_registry(&options).build();
        options.set_option("print_level", 0);
        options.set_option("max_iter", static_cast<Index>(500));
        options.set_option("tolerance", 1e-8);
        options.set_option("constr_viol_tol", 1e-8);

        const auto start = std::chrono::steady_clock::now();
        const IpSolverReturnFlag status = algorithm->optimize();
        const auto stop = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(stop - start)
                .count();

        auto data = builder.get_ipdata();
        auto &iterate = data->current_iterate();
        const double objective = iterate.obj_value();
        const double constraint_violation =
            maximum_absolute(iterate.constr_viol());

        const Index primal_variables =
            stages * states + (stages - 1) * controls;
        const Index constraints_count =
            (stages - 1) * states + states;
        std::cout << std::setprecision(12);
        std::cout
            << "problem,stages,controls,states,primal_variables,"
               "constraints,fatrop_ms,status,iterations,objective,"
               "max_constraint_violation\n";
        std::cout << problem_name << ',' << stages << ',' << controls
                  << ',' << states << ',' << primal_variables << ','
                  << constraints_count << ',' << elapsed_ms << ','
                  << static_cast<int>(status) << ','
                  << data->iteration_number() << ',' << objective << ','
                  << constraint_violation << '\n';
        return status == IpSolverReturnFlag::Success ? 0 : 2;
    }
    catch (const std::exception &error)
    {
        std::cerr << "dtoc_fatrop_benchmark: " << error.what() << '\n';
        return 1;
    }
}
