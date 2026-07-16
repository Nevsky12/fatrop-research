extern "C"
{
#include <cutest_c.h>
#include <cutest_routines.h>
}

#include <coin-or/IpIpoptApplication.hpp>
#include <coin-or/IpIpoptData.hpp>
#include <coin-or/IpTNLP.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

void check(ipc_ status, const char *operation)
{
    if (status != 0)
        throw std::runtime_error(std::string(operation) +
                                 " failed with CUTEst status " +
                                 std::to_string(status));
}

std::string trim_fortran_string(const char *value, std::size_t length)
{
    std::string result(value, length);
    while (!result.empty() &&
           (result.back() == ' ' || result.back() == '\0'))
        result.pop_back();
    return result;
}

class CutestProblem
{
public:
    CutestProblem(const std::string &outsdif,
                  const std::string &library)
    {
        CUTEST_load_routines_c_r(library.c_str());
        loaded_ = true;
        FORTRAN_open_c_r(&input_, outsdif.c_str(), &status_);
        check(status_, "FORTRAN_open");
        opened_ = true;

        CUTEST_cdimen_c_r(&status_, &input_, &n_, &m_);
        check(status_, "CUTEST_cdimen");

        x0_.resize(static_cast<std::size_t>(n_));
        lower_x_.resize(static_cast<std::size_t>(n_));
        upper_x_.resize(static_cast<std::size_t>(n_));
        initial_multipliers_.resize(static_cast<std::size_t>(m_));
        lower_c_.resize(static_cast<std::size_t>(m_));
        upper_c_.resize(static_cast<std::size_t>(m_));
        equation_ = std::make_unique<logical[]>(
            static_cast<std::size_t>(m_));
        linear_ = std::make_unique<logical[]>(
            static_cast<std::size_t>(m_));

        CUTEST_csetup_c_r(
            &status_, &input_, &output_, &buffer_, &n_, &m_, x0_.data(),
            lower_x_.data(), upper_x_.data(),
            initial_multipliers_.data(), lower_c_.data(), upper_c_.data(),
            equation_.get(), linear_.get(), &ordering_, &ordering_,
            &ordering_);
        check(status_, "CUTEST_csetup");
        setup_ = true;

        char raw_name[FSTRING_LEN + 1] = {};
        CUTEST_probname_c_r(&status_, raw_name);
        check(status_, "CUTEST_probname");
        name_ = trim_fortran_string(raw_name, FSTRING_LEN);

        char raw_classification[FCSTRING_LEN + 1] = {};
        CUTEST_classification_c_r(
            &status_, &input_, raw_classification);
        check(status_, "CUTEST_classification");
        classification_ =
            trim_fortran_string(raw_classification, FCSTRING_LEN);

        CUTEST_cdimsj_c_r(&status_, &jacobian_capacity_);
        check(status_, "CUTEST_cdimsj");
        jacobian_values_.resize(
            static_cast<std::size_t>(jacobian_capacity_));
        jacobian_columns_.resize(
            static_cast<std::size_t>(jacobian_capacity_));
        jacobian_rows_.resize(
            static_cast<std::size_t>(jacobian_capacity_));
        CUTEST_csjp_c_r(
            &status_, &jacobian_nonzeros_, &jacobian_capacity_,
            jacobian_columns_.data(), jacobian_rows_.data());
        check(status_, "CUTEST_csjp");
        jacobian_capacity_ = jacobian_nonzeros_;
        jacobian_values_.resize(
            static_cast<std::size_t>(jacobian_nonzeros_));
        jacobian_columns_.resize(
            static_cast<std::size_t>(jacobian_nonzeros_));
        jacobian_rows_.resize(
            static_cast<std::size_t>(jacobian_nonzeros_));

        CUTEST_cdimsh_c_r(&status_, &hessian_capacity_);
        check(status_, "CUTEST_cdimsh");
        hessian_values_.resize(
            static_cast<std::size_t>(hessian_capacity_));
        hessian_rows_.resize(
            static_cast<std::size_t>(hessian_capacity_));
        hessian_columns_.resize(
            static_cast<std::size_t>(hessian_capacity_));
        CUTEST_cshp_c_r(
            &status_, &n_, &hessian_nonzeros_, &hessian_capacity_,
            hessian_rows_.data(), hessian_columns_.data());
        check(status_, "CUTEST_cshp");
        hessian_capacity_ = hessian_nonzeros_;
        hessian_values_.resize(
            static_cast<std::size_t>(hessian_nonzeros_));
        hessian_rows_.resize(
            static_cast<std::size_t>(hessian_nonzeros_));
        hessian_columns_.resize(
            static_cast<std::size_t>(hessian_nonzeros_));

        final_x_ = x0_;
        final_constraints_.resize(static_cast<std::size_t>(m_));
        gradient_scratch_.resize(static_cast<std::size_t>(n_));
    }

    ~CutestProblem()
    {
        if (setup_)
            CUTEST_cterminate_c_r(&status_);
        if (opened_)
            FORTRAN_close_c_r(&input_, &status_);
        if (loaded_)
            CUTEST_unload_routines_c_r();
    }

    CutestProblem(const CutestProblem &) = delete;
    CutestProblem &operator=(const CutestProblem &) = delete;

    ipc_ n() const { return n_; }
    ipc_ m() const { return m_; }
    ipc_ jacobian_nonzeros() const { return jacobian_nonzeros_; }
    ipc_ hessian_nonzeros() const { return hessian_nonzeros_; }
    const std::string &name() const { return name_; }
    const std::string &classification() const
    {
        return classification_;
    }
    const std::vector<rpc_> &x0() const { return x0_; }
    const std::vector<rpc_> &lower_x() const { return lower_x_; }
    const std::vector<rpc_> &upper_x() const { return upper_x_; }
    const std::vector<rpc_> &lower_c() const { return lower_c_; }
    const std::vector<rpc_> &upper_c() const { return upper_c_; }
    const std::vector<ipc_> &jacobian_rows() const
    {
        return jacobian_rows_;
    }
    const std::vector<ipc_> &jacobian_columns() const
    {
        return jacobian_columns_;
    }
    const std::vector<ipc_> &hessian_rows() const
    {
        return hessian_rows_;
    }
    const std::vector<ipc_> &hessian_columns() const
    {
        return hessian_columns_;
    }

    double objective(const double *x)
    {
        rpc_ value = 0.0;
        logical gradient = false;
        CUTEST_cofg_c_r(
            &status_, &n_, x, &value, gradient_scratch_.data(),
            &gradient);
        check(status_, "CUTEST_cofg(objective)");
        return value;
    }

    void gradient(const double *x, double *gradient_values)
    {
        rpc_ value = 0.0;
        logical gradient = true;
        CUTEST_cofg_c_r(
            &status_, &n_, x, &value, gradient_values, &gradient);
        check(status_, "CUTEST_cofg(gradient)");
    }

    void constraints(const double *x, double *values)
    {
        CUTEST_ccf_c_r(&status_, &n_, &m_, x, values);
        check(status_, "CUTEST_ccf");
    }

    void jacobian(const double *x, double *values)
    {
        ipc_ nonzeros = 0;
        CUTEST_csj_c_r(
            &status_, &n_, x, &nonzeros, &jacobian_capacity_, values,
            jacobian_columns_.data(), jacobian_rows_.data());
        check(status_, "CUTEST_csj");
        if (nonzeros != jacobian_nonzeros_)
            throw std::runtime_error(
                "CUTEst Jacobian sparsity changed during optimization");
    }

    void hessian(const double *x, double objective_factor,
                 const double *multipliers, double *values)
    {
        ipc_ nonzeros = 0;
        CUTEST_cshj_c_r(
            &status_, &n_, &m_, x, &objective_factor, multipliers,
            &nonzeros, &hessian_capacity_, values, hessian_rows_.data(),
            hessian_columns_.data());
        check(status_, "CUTEST_cshj");
        if (nonzeros != hessian_nonzeros_)
            throw std::runtime_error(
                "CUTEst Hessian sparsity changed during optimization");
    }

    void store_final(const double *x, double objective,
                     int iterations)
    {
        std::copy(x, x + n_, final_x_.begin());
        final_objective_ = objective;
        iterations_ = iterations;
        constraints(final_x_.data(), final_constraints_.data());
        max_constraint_violation_ = 0.0;
        for (ipc_ i = 0; i < m_; ++i)
        {
            const double value =
                final_constraints_[static_cast<std::size_t>(i)];
            const double violation = std::max(
                {lower_c_[static_cast<std::size_t>(i)] - value,
                 value - upper_c_[static_cast<std::size_t>(i)], 0.0});
            max_constraint_violation_ =
                std::max(max_constraint_violation_, violation);
        }
    }

    double final_objective() const { return final_objective_; }
    double max_constraint_violation() const
    {
        return max_constraint_violation_;
    }
    int iterations() const { return iterations_; }

private:
    ipc_ status_ = 0;
    ipc_ input_ = 55;
    ipc_ output_ = 6;
    ipc_ buffer_ = 77;
    ipc_ ordering_ = 1;
    ipc_ n_ = 0;
    ipc_ m_ = 0;
    ipc_ jacobian_capacity_ = 0;
    ipc_ jacobian_nonzeros_ = 0;
    ipc_ hessian_capacity_ = 0;
    ipc_ hessian_nonzeros_ = 0;
    bool loaded_ = false;
    bool opened_ = false;
    bool setup_ = false;
    std::string name_;
    std::string classification_;
    std::vector<rpc_> x0_;
    std::vector<rpc_> lower_x_;
    std::vector<rpc_> upper_x_;
    std::vector<rpc_> initial_multipliers_;
    std::vector<rpc_> lower_c_;
    std::vector<rpc_> upper_c_;
    std::unique_ptr<logical[]> equation_;
    std::unique_ptr<logical[]> linear_;
    std::vector<rpc_> jacobian_values_;
    std::vector<ipc_> jacobian_columns_;
    std::vector<ipc_> jacobian_rows_;
    std::vector<rpc_> hessian_values_;
    std::vector<ipc_> hessian_rows_;
    std::vector<ipc_> hessian_columns_;
    std::vector<rpc_> final_x_;
    std::vector<rpc_> final_constraints_;
    std::vector<rpc_> gradient_scratch_;
    double final_objective_ = 0.0;
    double max_constraint_violation_ = 0.0;
    int iterations_ = -1;
};

class CutestTnlp final : public Ipopt::TNLP
{
public:
    explicit CutestTnlp(CutestProblem &problem) : problem_(problem) {}

    bool get_nlp_info(Ipopt::Index &n, Ipopt::Index &m,
                      Ipopt::Index &nnz_jac_g,
                      Ipopt::Index &nnz_h_lag,
                      IndexStyleEnum &index_style) override
    {
        n = problem_.n();
        m = problem_.m();
        nnz_jac_g = problem_.jacobian_nonzeros();
        nnz_h_lag = problem_.hessian_nonzeros();
        index_style = TNLP::C_STYLE;
        return true;
    }

    bool get_bounds_info(Ipopt::Index n, Ipopt::Number *x_l,
                         Ipopt::Number *x_u, Ipopt::Index m,
                         Ipopt::Number *g_l,
                         Ipopt::Number *g_u) override
    {
        std::copy_n(problem_.lower_x().data(), n, x_l);
        std::copy_n(problem_.upper_x().data(), n, x_u);
        std::copy_n(problem_.lower_c().data(), m, g_l);
        std::copy_n(problem_.upper_c().data(), m, g_u);
        return true;
    }

    bool get_starting_point(
        Ipopt::Index n, bool init_x, Ipopt::Number *x, bool init_z,
        Ipopt::Number *, Ipopt::Number *, Ipopt::Index,
        bool init_lambda, Ipopt::Number *) override
    {
        if (!init_x || init_z || init_lambda)
            return false;
        std::copy_n(problem_.x0().data(), n, x);
        return true;
    }

    bool eval_f(Ipopt::Index, const Ipopt::Number *x, bool,
                Ipopt::Number &obj_value) override
    {
        obj_value = problem_.objective(x);
        return true;
    }

    bool eval_grad_f(Ipopt::Index, const Ipopt::Number *x, bool,
                     Ipopt::Number *grad_f) override
    {
        problem_.gradient(x, grad_f);
        return true;
    }

    bool eval_g(Ipopt::Index, const Ipopt::Number *x, bool,
                Ipopt::Index, Ipopt::Number *g) override
    {
        problem_.constraints(x, g);
        return true;
    }

    bool eval_jac_g(
        Ipopt::Index, const Ipopt::Number *x, bool, Ipopt::Index,
        Ipopt::Index nele_jac, Ipopt::Index *i_row,
        Ipopt::Index *j_col, Ipopt::Number *values) override
    {
        if (nele_jac != problem_.jacobian_nonzeros())
            return false;
        if (values == nullptr)
        {
            std::copy(problem_.jacobian_rows().begin(),
                      problem_.jacobian_rows().end(), i_row);
            std::copy(problem_.jacobian_columns().begin(),
                      problem_.jacobian_columns().end(), j_col);
        }
        else
            problem_.jacobian(x, values);
        return true;
    }

    bool eval_h(
        Ipopt::Index, const Ipopt::Number *x, bool,
        Ipopt::Number obj_factor, Ipopt::Index,
        const Ipopt::Number *lambda, bool, Ipopt::Index nele_hess,
        Ipopt::Index *i_row, Ipopt::Index *j_col,
        Ipopt::Number *values) override
    {
        if (nele_hess != problem_.hessian_nonzeros())
            return false;
        if (values == nullptr)
        {
            for (Ipopt::Index entry = 0; entry < nele_hess; ++entry)
            {
                const Ipopt::Index row =
                    problem_.hessian_rows()[static_cast<std::size_t>(
                        entry)];
                const Ipopt::Index column =
                    problem_.hessian_columns()[static_cast<std::size_t>(
                        entry)];
                i_row[entry] = std::max(row, column);
                j_col[entry] = std::min(row, column);
            }
        }
        else
            problem_.hessian(x, obj_factor, lambda, values);
        return true;
    }

    void finalize_solution(
        Ipopt::SolverReturn, Ipopt::Index, const Ipopt::Number *x,
        const Ipopt::Number *, const Ipopt::Number *, Ipopt::Index,
        const Ipopt::Number *, const Ipopt::Number *,
        Ipopt::Number obj_value, const Ipopt::IpoptData *ip_data,
        Ipopt::IpoptCalculatedQuantities *) override
    {
        problem_.store_final(
            x, obj_value,
            ip_data == nullptr
                ? -1
                : static_cast<int>(ip_data->iter_count()));
    }

private:
    CutestProblem &problem_;
};

} // namespace

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        std::cerr
            << "Usage: cutest_ipopt_benchmark OUTSDIF.d libsif.so\n";
        return 1;
    }

    try
    {
        CutestProblem problem(argv[1], argv[2]);
        Ipopt::SmartPtr<Ipopt::IpoptApplication> application =
            IpoptApplicationFactory();
        application->Options()->SetIntegerValue("print_level", 0);
        application->Options()->SetStringValue("sb", "yes");
        application->Options()->SetStringValue("linear_solver", "mumps");
        application->Options()->SetStringValue(
            "hessian_approximation", "exact");
        application->Options()->SetNumericValue("tol", 1e-8);
        application->Options()->SetNumericValue(
            "constr_viol_tol", 1e-8);
        application->Options()->SetIntegerValue("max_iter", 500);
        const auto initialization = application->Initialize();
        if (initialization != Ipopt::Solve_Succeeded)
            throw std::runtime_error("IPOPT initialization failed");

        Ipopt::SmartPtr<Ipopt::TNLP> tnlp =
            new CutestTnlp(problem);
        const auto start = std::chrono::steady_clock::now();
        const Ipopt::ApplicationReturnStatus status =
            application->OptimizeTNLP(tnlp);
        const auto stop = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(stop - start)
                .count();

        std::cout << std::setprecision(12);
        std::cout
            << "problem,classification,n,m,jacobian_nonzeros,"
               "hessian_nonzeros,ipopt_ms,status,iterations,"
               "objective,max_constraint_violation\n";
        std::cout << problem.name() << ',' << problem.classification()
                  << ',' << problem.n() << ',' << problem.m() << ','
                  << problem.jacobian_nonzeros() << ','
                  << problem.hessian_nonzeros() << ',' << elapsed_ms
                  << ',' << static_cast<int>(status) << ','
                  << problem.iterations() << ','
                  << problem.final_objective() << ','
                  << problem.max_constraint_violation() << '\n';

        const bool success =
            status == Ipopt::Solve_Succeeded ||
            status == Ipopt::Solved_To_Acceptable_Level;
        return success ? 0 : 2;
    }
    catch (const std::exception &error)
    {
        std::cerr << "cutest_ipopt_benchmark: " << error.what() << '\n';
        return 3;
    }
}
