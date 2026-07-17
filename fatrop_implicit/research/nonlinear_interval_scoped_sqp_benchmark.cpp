#include "interval_scoped_collocation_problem.hpp"

#include <coin-or/IpIpoptApplication.hpp>
#include <coin-or/IpIpoptData.hpp>
#include <coin-or/IpTNLP.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using fatrop::Index;
using fatrop::Scalar;
using fatrop::research::IntervalScopedCollocationConfig;
using fatrop::research::IntervalScopedCollocationProblem;
using fatrop::research::ScopedDerivativeCheck;
using fatrop::research::ScopedIpmResult;

namespace
{
    struct IpoptResult
    {
        Eigen::VectorXd primal;
        Eigen::VectorXd multipliers;
        double elapsed_ms = std::numeric_limits<double>::quiet_NaN();
        double objective = std::numeric_limits<double>::quiet_NaN();
        double constraint_inf = std::numeric_limits<double>::quiet_NaN();
        double dual_inf = std::numeric_limits<double>::quiet_NaN();
        int status = -1;
        int iterations = -1;
        bool converged = false;
    };

    std::vector<Index> parse_list(const std::string &text)
    {
        std::vector<Index> result;
        std::size_t begin = 0;
        while (begin <= text.size())
        {
            const std::size_t end = text.find(',', begin);
            const std::string token = text.substr(
                begin,
                end == std::string::npos
                    ? std::string::npos : end - begin);
            if (token.empty())
                throw std::runtime_error("Empty list entry");
            result.push_back(static_cast<Index>(std::stoll(token)));
            if (end == std::string::npos)
                break;
            begin = end + 1;
        }
        return result;
    }

    std::string encode(const std::vector<Index> &values)
    {
        std::string result;
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            if (index > 0)
                result += ':';
            result += std::to_string(values[index]);
        }
        return result;
    }

    double median(std::vector<double> values)
    {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }

    IntervalScopedCollocationConfig parse_arguments(
        int argc, char **argv)
    {
        IntervalScopedCollocationConfig config;
        for (int argument = 1; argument < argc; ++argument)
        {
            const std::string option(argv[argument]);
            const auto next_string = [&](const std::string &name)
            {
                if (argument + 1 >= argc)
                    throw std::runtime_error(name + " requires a value");
                return std::string(argv[++argument]);
            };
            const auto next_index = [&](const std::string &name)
            {
                return static_cast<Index>(std::stoll(next_string(name)));
            };
            const auto next_scalar = [&](const std::string &name)
            {
                return std::stod(next_string(name));
            };

            if (option == "--phase-nodes")
                config.phase_nodes = parse_list(next_string(option));
            else if (option == "--phase-nx")
                config.phase_states = parse_list(next_string(option));
            else if (option == "--phase-nu")
                config.phase_controls = parse_list(next_string(option));
            else if (option == "--phase-parameters")
                config.phase_parameter_dimension = next_index(option);
            else if (option == "--interface-parameters")
                config.interface_parameter_dimension = next_index(option);
            else if (option == "--segment-parameters")
                config.segment_parameter_dimension = next_index(option);
            else if (option == "--segment-width")
                config.segment_width = next_index(option);
            else if (option == "--segment-stride")
                config.segment_stride = next_index(option);
            else if (option == "--global-parameters")
                config.global_parameter_dimension = next_index(option);
            else if (option == "--path-inequalities")
                config.path_inequality_dimension = next_index(option);
            else if (option == "--inequality-lower")
                config.inequality_lower_bound = next_scalar(option);
            else if (option == "--inequality-upper")
                config.inequality_upper_bound = next_scalar(option);
            else if (option == "--collocation-degree")
                config.collocation_degree = next_index(option);
            else if (option == "--max-iterations")
                config.max_iterations = next_index(option);
            else if (option == "--repeats")
                config.repeats = next_index(option);
            else if (option == "--tolerance")
                config.tolerance = next_scalar(option);
            else if (option == "--regularization")
                config.regularization = next_scalar(option);
            else if (option == "--dual-regularization")
                config.dual_regularization = next_scalar(option);
            else if (option == "--fraction-to-boundary")
                config.fraction_to_boundary = next_scalar(option);
            else if (option == "--no-factor-reuse")
                config.reuse_predictor_corrector_factorization = false;
            else if (option == "--no-dense-validation")
                config.dense_step_validation = false;
            else if (option == "--ipopt")
                config.run_ipopt = true;
            else if (option == "--ipopt-linear-solver")
                config.ipopt_linear_solver = next_string(option);
            else if (option == "--ipopt-hsl-library")
                config.ipopt_hsl_library = next_string(option);
            else if (option == "--help")
            {
                std::cout
                    << "Usage: nonlinear_interval_scoped_sqp_benchmark [options]\n"
                    << "  --phase-nodes n0,n1,...\n"
                    << "  --phase-nx nx0,nx1,...\n"
                    << "  --phase-nu nu0,nu1,...\n"
                    << "  --phase-parameters N\n"
                    << "  --interface-parameters N\n"
                    << "  --segment-parameters N\n"
                    << "  --segment-width N\n"
                    << "  --segment-stride N\n"
                    << "  --global-parameters N\n"
                    << "  --path-inequalities N\n"
                    << "  --inequality-lower VALUE\n"
                    << "  --inequality-upper VALUE\n"
                    << "  --collocation-degree 1|2|3\n"
                    << "  --max-iterations N\n"
                    << "  --repeats N\n"
                    << "  --tolerance VALUE\n"
                    << "  --regularization VALUE\n"
                    << "  --dual-regularization VALUE\n"
                    << "  --fraction-to-boundary VALUE\n"
                    << "  --no-factor-reuse\n"
                    << "  --no-dense-validation\n"
                    << "  --ipopt\n"
                    << "  --ipopt-linear-solver {ma57|mumps|...}\n"
                    << "  --ipopt-hsl-library /path/to/libhsl.so\n";
                std::exit(0);
            }
            else
                throw std::runtime_error("Unknown option: " + option);
        }
        return config;
    }

    class ScopedCollocationTnlp final : public Ipopt::TNLP
    {
    public:
        ScopedCollocationTnlp(
            IntervalScopedCollocationProblem &problem,
            IpoptResult &result)
            : problem_(problem), result_(result),
              zero_multipliers_(Eigen::VectorXd::Zero(
                  problem.number_of_constraints()))
        {
        }

        bool get_nlp_info(
            Ipopt::Index &n,
            Ipopt::Index &m,
            Ipopt::Index &nnz_jac_g,
            Ipopt::Index &nnz_h_lag,
            IndexStyleEnum &index_style) override
        {
            n = static_cast<Ipopt::Index>(problem_.number_of_variables());
            m = static_cast<Ipopt::Index>(problem_.number_of_constraints());
            nnz_jac_g = static_cast<Ipopt::Index>(
                problem_.jacobian_pattern().size());
            nnz_h_lag = static_cast<Ipopt::Index>(
                problem_.hessian_pattern().size());
            index_style = TNLP::C_STYLE;
            return true;
        }

        bool get_bounds_info(
            Ipopt::Index n,
            Ipopt::Number *x_l,
            Ipopt::Number *x_u,
            Ipopt::Index m,
            Ipopt::Number *g_l,
            Ipopt::Number *g_u) override
        {
            for (Ipopt::Index variable = 0; variable < n; ++variable)
            {
                x_l[variable] = -1e19;
                x_u[variable] = 1e19;
            }
            for (Ipopt::Index constraint = 0; constraint < m; ++constraint)
            {
                g_l[constraint] =
                    problem_.constraint_lower_bounds()(constraint);
                g_u[constraint] =
                    problem_.constraint_upper_bounds()(constraint);
            }
            return true;
        }

        bool get_starting_point(
            Ipopt::Index n,
            bool init_x,
            Ipopt::Number *x,
            bool init_z,
            Ipopt::Number *,
            Ipopt::Number *,
            Ipopt::Index,
            bool init_lambda,
            Ipopt::Number *) override
        {
            if (!init_x || init_z || init_lambda)
                return false;
            const Eigen::VectorXd initial = problem_.initial_primal();
            if (initial.size() != n)
                return false;
            std::copy(initial.data(), initial.data() + n, x);
            return true;
        }

        bool eval_f(
            Ipopt::Index n,
            const Ipopt::Number *x,
            bool,
            Ipopt::Number &objective) override
        {
            Eigen::Map<const Eigen::VectorXd> primal(x, n);
            objective = problem_.evaluate(
                primal, zero_multipliers_).objective;
            return std::isfinite(objective);
        }

        bool eval_grad_f(
            Ipopt::Index n,
            const Ipopt::Number *x,
            bool,
            Ipopt::Number *gradient) override
        {
            Eigen::Map<const Eigen::VectorXd> primal(x, n);
            problem_.evaluate(primal, zero_multipliers_);
            Eigen::Map<Eigen::VectorXd>(gradient, n) =
                problem_.objective_gradient();
            return problem_.objective_gradient().allFinite();
        }

        bool eval_g(
            Ipopt::Index n,
            const Ipopt::Number *x,
            bool,
            Ipopt::Index m,
            Ipopt::Number *constraints) override
        {
            Eigen::Map<const Eigen::VectorXd> primal(x, n);
            problem_.evaluate(primal, zero_multipliers_);
            Eigen::Map<Eigen::VectorXd>(constraints, m) =
                problem_.constraints();
            return problem_.constraints().allFinite();
        }

        bool eval_jac_g(
            Ipopt::Index n,
            const Ipopt::Number *x,
            bool,
            Ipopt::Index,
            Ipopt::Index nele_jac,
            Ipopt::Index *i_row,
            Ipopt::Index *j_col,
            Ipopt::Number *values) override
        {
            const auto &pattern = problem_.jacobian_pattern();
            if (nele_jac != static_cast<Ipopt::Index>(pattern.size()))
                return false;
            if (values == nullptr)
            {
                for (std::size_t entry = 0; entry < pattern.size(); ++entry)
                {
                    i_row[entry] = static_cast<Ipopt::Index>(
                        pattern[entry].first);
                    j_col[entry] = static_cast<Ipopt::Index>(
                        pattern[entry].second);
                }
                return true;
            }
            Eigen::Map<const Eigen::VectorXd> primal(x, n);
            problem_.evaluate(primal, zero_multipliers_);
            const auto &current = problem_.jacobian_values();
            std::copy(current.begin(), current.end(), values);
            return true;
        }

        bool eval_h(
            Ipopt::Index n,
            const Ipopt::Number *x,
            bool,
            Ipopt::Number objective_factor,
            Ipopt::Index m,
            const Ipopt::Number *lambda,
            bool,
            Ipopt::Index nele_hess,
            Ipopt::Index *i_row,
            Ipopt::Index *j_col,
            Ipopt::Number *values) override
        {
            const auto &pattern = problem_.hessian_pattern();
            if (nele_hess != static_cast<Ipopt::Index>(pattern.size()))
                return false;
            if (values == nullptr)
            {
                for (std::size_t entry = 0; entry < pattern.size(); ++entry)
                {
                    i_row[entry] = static_cast<Ipopt::Index>(
                        pattern[entry].first);
                    j_col[entry] = static_cast<Ipopt::Index>(
                        pattern[entry].second);
                }
                return true;
            }
            Eigen::Map<const Eigen::VectorXd> primal(x, n);
            Eigen::Map<const Eigen::VectorXd> multipliers(lambda, m);
            problem_.evaluate(
                primal, multipliers, objective_factor);
            const auto &current = problem_.hessian_values();
            std::copy(current.begin(), current.end(), values);
            return true;
        }

        void finalize_solution(
            Ipopt::SolverReturn status,
            Ipopt::Index n,
            const Ipopt::Number *x,
            const Ipopt::Number *,
            const Ipopt::Number *,
            Ipopt::Index m,
            const Ipopt::Number *,
            const Ipopt::Number *lambda,
            Ipopt::Number objective,
            const Ipopt::IpoptData *ip_data,
            Ipopt::IpoptCalculatedQuantities *) override
        {
            result_.status = static_cast<int>(status);
            result_.primal = Eigen::Map<const Eigen::VectorXd>(x, n);
            result_.multipliers =
                Eigen::Map<const Eigen::VectorXd>(lambda, m);
            result_.objective = objective;
            if (ip_data != nullptr)
                result_.iterations = static_cast<int>(ip_data->iter_count());
        }

    private:
        IntervalScopedCollocationProblem &problem_;
        IpoptResult &result_;
        Eigen::VectorXd zero_multipliers_;
    };

    IpoptResult solve_ipopt(
        IntervalScopedCollocationProblem &problem)
    {
        IpoptResult result;
        Ipopt::SmartPtr<Ipopt::IpoptApplication> application =
            IpoptApplicationFactory();
        application->Options()->SetIntegerValue("print_level", 0);
        application->Options()->SetStringValue("sb", "yes");
        application->Options()->SetStringValue(
            "linear_solver", problem.config().ipopt_linear_solver);
        if (!problem.config().ipopt_hsl_library.empty())
        {
            application->Options()->SetStringValue(
                "hsllib", problem.config().ipopt_hsl_library);
        }
        application->Options()->SetStringValue(
            "hessian_approximation", "exact");
        application->Options()->SetStringValue(
            "nlp_scaling_method", "none");
        application->Options()->SetStringValue("mu_strategy", "adaptive");
        application->Options()->SetNumericValue(
            "tol", problem.config().tolerance);
        application->Options()->SetNumericValue(
            "constr_viol_tol", problem.config().tolerance);
        application->Options()->SetNumericValue(
            "dual_inf_tol", problem.config().tolerance);
        application->Options()->SetIntegerValue(
            "max_iter", static_cast<int>(problem.config().max_iterations * 3));
        const auto initialization = application->Initialize();
        if (initialization != Ipopt::Solve_Succeeded)
        {
            result.status = static_cast<int>(initialization);
            return result;
        }

        Ipopt::SmartPtr<ScopedCollocationTnlp> raw =
            new ScopedCollocationTnlp(problem, result);
        Ipopt::SmartPtr<Ipopt::TNLP> tnlp = raw;
        std::vector<double> samples;
        Ipopt::ApplicationReturnStatus status = Ipopt::Internal_Error;
        for (Index repeat = 0; repeat < problem.config().repeats; ++repeat)
        {
            const auto start = std::chrono::steady_clock::now();
            status = application->OptimizeTNLP(tnlp);
            samples.push_back(
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count());
            if (status != Ipopt::Solve_Succeeded
                && status != Ipopt::Solved_To_Acceptable_Level)
                break;
        }
        result.elapsed_ms = median(std::move(samples));
        if (result.status == -1)
            result.status = static_cast<int>(status);
        if (result.primal.size() == problem.number_of_variables()
            && result.multipliers.size() == problem.number_of_constraints())
        {
            const auto values = problem.evaluate(
                result.primal, result.multipliers);
            result.objective = values.objective;
            result.constraint_inf = values.constraint_inf;
            result.dual_inf = values.dual_inf;
            result.converged =
                (status == Ipopt::Solve_Succeeded
                 || status == Ipopt::Solved_To_Acceptable_Level)
                && values.constraint_inf
                    <= 10.0 * problem.config().tolerance
                && values.dual_inf
                    <= 10.0 * problem.config().tolerance;
        }
        return result;
    }
} // namespace

int main(int argc, char **argv)
try
{
    IntervalScopedCollocationConfig config =
        parse_arguments(argc, argv);
    IntervalScopedCollocationProblem validation_problem(config);
    const ScopedDerivativeCheck derivative_check =
        validation_problem.check_derivatives(
            validation_problem.initial_primal());
    const ScopedIpmResult validation = validation_problem.solve_ipm();

    config.dense_step_validation = false;
    std::vector<double> total_samples;
    std::vector<double> kkt_samples;
    ScopedIpmResult result;
    for (Index repeat = 0; repeat < config.repeats; ++repeat)
    {
        IntervalScopedCollocationProblem problem(config);
        result = problem.solve_ipm();
        total_samples.push_back(result.elapsed_ms);
        kkt_samples.push_back(result.kkt_ms);
    }
    result.elapsed_ms = median(std::move(total_samples));
    result.kkt_ms = median(std::move(kkt_samples));
    result.dense_step_difference = validation.dense_step_difference;
    const auto symbolic = validation_problem.symbolic_stats();
    IpoptResult ipopt_result;
    if (config.run_ipopt)
    {
        IntervalScopedCollocationProblem ipopt_problem(config);
        ipopt_result = solve_ipopt(ipopt_problem);
    }
    const double ipopt_primal_difference =
        config.run_ipopt
        && ipopt_result.primal.size() == result.primal.size()
        ? (ipopt_result.primal - result.primal)
            .lpNorm<Eigen::Infinity>()
        : std::numeric_limits<double>::quiet_NaN();
    const double ipopt_multiplier_difference =
        config.run_ipopt
        && ipopt_result.multipliers.size() == result.multipliers.size()
        ? (ipopt_result.multipliers - result.multipliers)
            .lpNorm<Eigen::Infinity>()
        : std::numeric_limits<double>::quiet_NaN();
    const double ipopt_objective_difference =
        config.run_ipopt
        && std::isfinite(ipopt_result.objective)
        ? std::abs(ipopt_result.objective - result.objective)
        : std::numeric_limits<double>::quiet_NaN();
    // Near an active-set transition, componentwise primal/dual distance is
    // only O(sqrt(KKT tolerance)); objective and independent KKT residuals
    // remain the primary equivalence gates.
    const Scalar primal_comparison_tolerance =
        validation_problem.number_of_inequalities() > 0
        ? std::max<Scalar>(
            1e-6, 3.0 * std::sqrt(config.tolerance))
        : std::max<Scalar>(1e-6, 100.0 * config.tolerance);
    const Scalar multiplier_comparison_tolerance =
        validation_problem.number_of_inequalities() > 0
        ? std::max<Scalar>(
            1e-5, 3.0 * std::sqrt(config.tolerance))
        : std::max<Scalar>(1e-5, 1000.0 * config.tolerance);
    const Scalar objective_comparison_tolerance =
        std::max<Scalar>(1e-8, 10.0 * config.tolerance)
        * (1.0 + std::abs(result.objective));
    const bool success = validation.converged && result.converged
        && derivative_check.objective_error < 2e-7
        && derivative_check.constraint_error < 2e-7
        && derivative_check.lagrangian_hessian_error < 2e-7
        && result.constraint_inf <= 10.0 * config.tolerance
        && result.dual_inf <= 10.0 * config.tolerance
        && result.complementarity_inf <= 10.0 * config.tolerance
        && result.linear_residual < 2e-7
        && (std::isnan(result.dense_step_difference)
            || result.dense_step_difference < 2e-7)
        && (!config.run_ipopt
            || (ipopt_result.converged
                && ipopt_primal_difference
                    <= primal_comparison_tolerance
                && ipopt_multiplier_difference
                    <= multiplier_comparison_tolerance
                && ipopt_objective_difference
                    <= objective_comparison_tolerance));

    std::cout
        << "phases,phase_nodes,phase_nx,phase_nu,degree,path_ineq,phase_p,"
           "interface_p,segment_p,segment_width,segment_stride,global_p,"
           "hessian_model,predictor_corrector_reuse,blocks,"
           "scoped_variables,trajectory_variables,equalities,inequalities,"
           "constraints,omega,edges,"
           "converged,iterations,total_ms,kkt_ms,line_search_evaluations,"
           "factorizations,retained_rhs_solves,"
           "objective,constraint_inf,dual_inf,complementarity_inf,"
           "linear_residual,"
           "dense_step_difference,objective_derivative_error,"
           "constraint_derivative_error,lagrangian_hessian_error,"
           "ipopt_solver,ipopt_converged,"
           "ipopt_status,ipopt_iterations,ipopt_ms,ipopt_objective,"
           "ipopt_constraint_inf,ipopt_dual_inf,ipopt_primal_difference,"
           "ipopt_multiplier_difference,ipopt_objective_difference,"
           "ipopt_over_structured,success\n";
    std::cout
        << config.phase_nodes.size() << ','
        << encode(config.phase_nodes) << ','
        << encode(config.phase_states) << ','
        << encode(config.phase_controls) << ','
        << config.collocation_degree << ','
        << config.path_inequality_dimension << ','
        << config.phase_parameter_dimension << ','
        << config.interface_parameter_dimension << ','
        << config.segment_parameter_dimension << ','
        << config.segment_width << ','
        << config.segment_stride << ','
        << config.global_parameter_dimension << ','
        << "exact_lagrangian" << ','
        << (config.reuse_predictor_corrector_factorization ? 1 : 0) << ','
        << validation_problem.blocks().size() << ','
        << validation_problem.number_of_scoped_variables() << ','
        << validation_problem.number_of_trajectory_variables() << ','
        << validation_problem.number_of_equalities() << ','
        << validation_problem.number_of_inequalities() << ','
        << validation_problem.number_of_constraints() << ','
        << symbolic.maximum_active_dimension << ','
        << symbolic.number_of_edges << ','
        << (result.converged ? 1 : 0) << ','
        << result.iterations << ','
        << std::setprecision(12)
        << result.elapsed_ms << ','
        << result.kkt_ms << ','
        << result.line_search_evaluations << ','
        << result.factorizations << ','
        << result.retained_rhs_solves << ','
        << result.objective << ','
        << result.constraint_inf << ','
        << result.dual_inf << ','
        << result.complementarity_inf << ','
        << result.linear_residual << ','
        << result.dense_step_difference << ','
        << derivative_check.objective_error << ','
        << derivative_check.constraint_error << ','
        << derivative_check.lagrangian_hessian_error << ','
        << config.ipopt_linear_solver << ','
        << (ipopt_result.converged ? 1 : 0) << ','
        << ipopt_result.status << ','
        << ipopt_result.iterations << ','
        << ipopt_result.elapsed_ms << ','
        << ipopt_result.objective << ','
        << ipopt_result.constraint_inf << ','
        << ipopt_result.dual_inf << ','
        << ipopt_primal_difference << ','
        << ipopt_multiplier_difference << ','
        << ipopt_objective_difference << ','
        << (config.run_ipopt
            ? ipopt_result.elapsed_ms / result.elapsed_ms
            : std::numeric_limits<double>::quiet_NaN()) << ','
        << (success ? 1 : 0) << '\n';
    return success ? 0 : 2;
}
catch (const std::exception &error)
{
    std::cerr << "nonlinear_interval_scoped_sqp_benchmark: "
              << error.what() << '\n';
    return 1;
}
