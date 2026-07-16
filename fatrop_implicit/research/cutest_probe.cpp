extern "C"
{
#include <cutest_c.h>
#include <cutest_routines.h>
}

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
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

double max_abs(const std::vector<rpc_> &values)
{
    double result = 0.0;
    for (const double value : values)
        result = std::max(result, std::abs(value));
    return result;
}

std::string trim_name(const char *value)
{
    std::string result(value, FSTRING_LEN);
    while (!result.empty() &&
           (result.back() == ' ' || result.back() == '\0'))
        result.pop_back();
    return result;
}

bool parse_dtoc_variable(const std::string &name, ipc_ &stage,
                         ipc_ &kind, ipc_ &component)
{
    if (name.size() < 2 ||
        (name.front() != 'X' && name.front() != 'Y'))
        return false;
    try
    {
        const std::size_t open = name.find('(');
        const std::size_t start =
            open == std::string::npos ? 1 : open + 1;
        const std::size_t comma = name.find(',', start);
        const std::size_t close =
            open == std::string::npos ? name.size()
                                      : name.find(')', start);
        if (close == std::string::npos)
            return false;
        const std::size_t first_end =
            comma == std::string::npos ? close : comma;
        stage = static_cast<ipc_>(std::stoi(
            name.substr(start, first_end - start)));
        component =
            comma == std::string::npos
                ? 0
                : static_cast<ipc_>(
                      std::stoi(name.substr(comma + 1,
                                            close - comma - 1)));
        kind = name.front() == 'X' ? 0 : 1;
        return true;
    }
    catch (const std::exception &)
    {
        return false;
    }
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 3 && argc != 4)
    {
        std::cerr
            << "Usage: cutest_probe OUTSDIF.d libsif.so [--show-names]\n";
        return 1;
    }

    const std::string outsdif = argv[1];
    const std::string library = argv[2];
    ipc_ status = 0;
    ipc_ input = 55;
    ipc_ output = 6;
    ipc_ buffer = 77;
    ipc_ ordering = 1;
    bool loaded = false;
    bool opened = false;
    bool setup = false;

    try
    {
        CUTEST_load_routines_c_r(library.c_str());
        loaded = true;
        FORTRAN_open_c_r(&input, outsdif.c_str(), &status);
        check(status, "FORTRAN_open");
        opened = true;

        ipc_ n = 0;
        ipc_ m = 0;
        CUTEST_cdimen_c_r(&status, &input, &n, &m);
        check(status, "CUTEST_cdimen");

        std::vector<rpc_> x(static_cast<std::size_t>(n));
        std::vector<rpc_> lower_x(static_cast<std::size_t>(n));
        std::vector<rpc_> upper_x(static_cast<std::size_t>(n));
        std::vector<rpc_> multipliers(static_cast<std::size_t>(m));
        std::vector<rpc_> lower_c(static_cast<std::size_t>(m));
        std::vector<rpc_> upper_c(static_cast<std::size_t>(m));
        auto equation = std::make_unique<logical[]>(
            static_cast<std::size_t>(m));
        auto linear = std::make_unique<logical[]>(
            static_cast<std::size_t>(m));

        CUTEST_csetup_c_r(
            &status, &input, &output, &buffer, &n, &m, x.data(),
            lower_x.data(), upper_x.data(), multipliers.data(),
            lower_c.data(), upper_c.data(), equation.get(), linear.get(),
            &ordering, &ordering, &ordering);
        check(status, "CUTEST_csetup");
        setup = true;

        std::vector<rpc_> constraints(static_cast<std::size_t>(m));
        rpc_ objective = 0.0;
        CUTEST_cfn_c_r(&status, &n, &m, x.data(), &objective,
                       constraints.data());
        check(status, "CUTEST_cfn");

        ipc_ jacobian_capacity = 0;
        CUTEST_cdimsj_c_r(&status, &jacobian_capacity);
        check(status, "CUTEST_cdimsj");
        std::vector<rpc_> jacobian_values(
            static_cast<std::size_t>(jacobian_capacity));
        std::vector<ipc_> jacobian_variables(
            static_cast<std::size_t>(jacobian_capacity));
        std::vector<ipc_> jacobian_constraints(
            static_cast<std::size_t>(jacobian_capacity));
        ipc_ jacobian_nonzeros = 0;
        CUTEST_csj_c_r(
            &status, &n, x.data(), &jacobian_nonzeros,
            &jacobian_capacity, jacobian_values.data(),
            jacobian_variables.data(), jacobian_constraints.data());
        check(status, "CUTEST_csj");

        ipc_ hessian_capacity = 0;
        CUTEST_cdimsh_c_r(&status, &hessian_capacity);
        check(status, "CUTEST_cdimsh");
        std::vector<rpc_> hessian_values(
            static_cast<std::size_t>(hessian_capacity));
        std::vector<ipc_> hessian_rows(
            static_cast<std::size_t>(hessian_capacity));
        std::vector<ipc_> hessian_columns(
            static_cast<std::size_t>(hessian_capacity));
        ipc_ hessian_nonzeros = 0;
        CUTEST_csh_c_r(
            &status, &n, &m, x.data(), multipliers.data(),
            &hessian_nonzeros, &hessian_capacity, hessian_values.data(),
            hessian_rows.data(), hessian_columns.data());
        check(status, "CUTEST_csh");

        ipc_ nonlinear_objective = 0;
        ipc_ nonlinear_constraints = 0;
        ipc_ equality_constraints = 0;
        ipc_ linear_constraints = 0;
        CUTEST_cstats_c_r(
            &status, &nonlinear_objective, &nonlinear_constraints,
            &equality_constraints, &linear_constraints);
        check(status, "CUTEST_cstats");

        char classification[FCSTRING_LEN + 1] = {};
        CUTEST_classification_c_r(&status, &input, classification);
        check(status, "CUTEST_classification");
        classification[FCSTRING_LEN] = '\0';

        char problem_name[FSTRING_LEN + 1] = {};
        std::vector<char> variable_names(
            static_cast<std::size_t>(n) * (FSTRING_LEN + 1));
        std::vector<char> constraint_names(
            static_cast<std::size_t>(m) * (FSTRING_LEN + 1));
        CUTEST_cnames_c_r(
            &status, &n, &m, problem_name, variable_names.data(),
            constraint_names.data());
        check(status, "CUTEST_cnames");

        std::vector<ipc_> reordered_position(
            static_cast<std::size_t>(n), -1);
        std::vector<std::tuple<ipc_, ipc_, ipc_, ipc_>> order;
        order.reserve(static_cast<std::size_t>(n));
        bool dtoc_names = true;
        for (ipc_ variable = 0; variable < n; ++variable)
        {
            const char *raw_name =
                variable_names.data() +
                static_cast<std::size_t>(variable) *
                    (FSTRING_LEN + 1);
            ipc_ stage = 0;
            ipc_ kind = 0;
            ipc_ component = 0;
            const std::string name = trim_name(raw_name);
            if (argc == 4)
                std::cerr << variable << ": " << name << '\n';
            if (!parse_dtoc_variable(name, stage, kind,
                                     component))
            {
                dtoc_names = false;
                continue;
            }
            order.emplace_back(stage, kind, component, variable);
        }
        if (dtoc_names)
        {
            std::sort(order.begin(), order.end());
            for (ipc_ position = 0; position < n; ++position)
            {
                const ipc_ original =
                    std::get<3>(order[static_cast<std::size_t>(position)]);
                reordered_position[static_cast<std::size_t>(original)] =
                    position;
            }
        }

        std::vector<ipc_> minimum_column(
            static_cast<std::size_t>(m),
            std::numeric_limits<ipc_>::max());
        std::vector<ipc_> maximum_column(static_cast<std::size_t>(m), -1);
        for (ipc_ entry = 0; entry < jacobian_nonzeros; ++entry)
        {
            const ipc_ constraint = jacobian_constraints[entry];
            const ipc_ variable = jacobian_variables[entry];
            if (constraint < 0 || constraint >= m)
                continue;
            minimum_column[static_cast<std::size_t>(constraint)] =
                std::min(
                    minimum_column[static_cast<std::size_t>(constraint)],
                    variable);
            maximum_column[static_cast<std::size_t>(constraint)] =
                std::max(
                    maximum_column[static_cast<std::size_t>(constraint)],
                    variable);
        }
        ipc_ maximum_constraint_span = 0;
        ipc_ maximum_reordered_constraint_span = 0;
        for (ipc_ constraint = 0; constraint < m; ++constraint)
        {
            if (maximum_column[static_cast<std::size_t>(constraint)] >= 0)
                maximum_constraint_span = std::max(
                    maximum_constraint_span,
                    maximum_column[static_cast<std::size_t>(constraint)] -
                        minimum_column[static_cast<std::size_t>(constraint)] +
                        1);
        }
        if (dtoc_names)
        {
            std::fill(minimum_column.begin(), minimum_column.end(),
                      std::numeric_limits<ipc_>::max());
            std::fill(maximum_column.begin(), maximum_column.end(), -1);
            for (ipc_ entry = 0; entry < jacobian_nonzeros; ++entry)
            {
                const ipc_ constraint = jacobian_constraints[entry];
                const ipc_ original = jacobian_variables[entry];
                if (constraint < 0 || constraint >= m || original < 0 ||
                    original >= n)
                    continue;
                const ipc_ variable =
                    reordered_position[static_cast<std::size_t>(original)];
                minimum_column[static_cast<std::size_t>(constraint)] =
                    std::min(
                        minimum_column[static_cast<std::size_t>(constraint)],
                        variable);
                maximum_column[static_cast<std::size_t>(constraint)] =
                    std::max(
                        maximum_column[static_cast<std::size_t>(constraint)],
                        variable);
            }
            for (ipc_ constraint = 0; constraint < m; ++constraint)
            {
                if (maximum_column[static_cast<std::size_t>(constraint)] >=
                    0)
                    maximum_reordered_constraint_span = std::max(
                        maximum_reordered_constraint_span,
                        maximum_column[static_cast<std::size_t>(constraint)] -
                            minimum_column[static_cast<std::size_t>(
                                constraint)] +
                            1);
            }
        }

        std::cout
            << "classification,n,m,jacobian_nonzeros,hessian_nonzeros,"
               "nonlinear_objective_variables,"
               "nonlinear_constraint_variables,equality_constraints,"
               "linear_constraints,initial_objective,"
               "initial_constraint_max_abs,max_constraint_column_span,"
               "stage_reordered_constraint_span\n";
        std::cout << classification << ',' << n << ',' << m << ','
                  << jacobian_nonzeros << ',' << hessian_nonzeros << ','
                  << nonlinear_objective << ',' << nonlinear_constraints
                  << ',' << equality_constraints << ','
                  << linear_constraints << ',' << objective << ','
                  << max_abs(constraints) << ','
                  << maximum_constraint_span << ','
                  << (dtoc_names
                          ? maximum_reordered_constraint_span
                          : -1)
                  << '\n';

        CUTEST_cterminate_c_r(&status);
        check(status, "CUTEST_cterminate");
        setup = false;
        FORTRAN_close_c_r(&input, &status);
        check(status, "FORTRAN_close");
        opened = false;
        CUTEST_unload_routines_c_r();
        loaded = false;
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "cutest_probe: " << error.what() << '\n';
        if (setup)
            CUTEST_cterminate_c_r(&status);
        if (opened)
            FORTRAN_close_c_r(&input, &status);
        if (loaded)
            CUTEST_unload_routines_c_r();
        return 2;
    }
}
