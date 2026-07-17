#include "fatrop/ocp/dims.hpp"
#include "fatrop/common/exception.hpp"
using namespace fatrop;

namespace
{
std::vector<Index> normalize_stage_border(
    const int K, const std::vector<Index> &border)
{
  return border.empty()
      ? std::vector<Index>(static_cast<std::size_t>(K), 0)
      : border;
}
}

ProblemDims::ProblemDims(int K, const std::vector<Index> &nu, const std::vector<Index> &nx,
                 const std::vector<Index> &ng, const std::vector<Index> &ng_ineq,
                 Index number_of_global_parameters)
    : ProblemDims(
          K, nu, nx, ng, ng_ineq, number_of_global_parameters,
          std::vector<Index>{})
{
}

ProblemDims::ProblemDims(int K, const std::vector<Index> &nu, const std::vector<Index> &nx,
                 const std::vector<Index> &ng, const std::vector<Index> &ng_ineq,
                 Index number_of_global_parameters,
                 const std::vector<Index> &number_of_stage_border_variables)
    : K(K), number_of_controls(nu), number_of_states(nx), number_of_eq_constraints(ng),
      number_of_ineq_constraints(ng_ineq),
      number_of_global_parameters(number_of_global_parameters),
      number_of_stage_border_variables(normalize_stage_border(
          K, number_of_stage_border_variables))
{
  check_problem_dimensions();
}

ProblemDims::ProblemDims(int K, std::vector<Index> &&nu, std::vector<Index> &&nx, std::vector<Index> &&ng,
                 std::vector<Index> &&ng_ineq, Index number_of_global_parameters)
    : ProblemDims(
          K, std::move(nu), std::move(nx), std::move(ng),
          std::move(ng_ineq), number_of_global_parameters,
          std::vector<Index>{})
{
}

ProblemDims::ProblemDims(int K, std::vector<Index> &&nu, std::vector<Index> &&nx, std::vector<Index> &&ng,
                 std::vector<Index> &&ng_ineq, Index number_of_global_parameters,
                 std::vector<Index> &&number_of_stage_border_variables)
    : K(K), number_of_controls(std::move(nu)), number_of_states(std::move(nx)),
      number_of_eq_constraints(std::move(ng)), number_of_ineq_constraints(std::move(ng_ineq)),
      number_of_global_parameters(number_of_global_parameters),
      number_of_stage_border_variables(normalize_stage_border(
          K, number_of_stage_border_variables))
{
  check_problem_dimensions();
}

void ProblemDims::check_problem_dimensions() const {
    fatrop_assert_msg(K > 0, "The OCP horizon length must be positive.");
    fatrop_assert_msg(number_of_global_parameters >= 0,
                      "The number of global parameters must be non-negative.");
    // check if the number of controls, states, and constraints are of the correct size
    fatrop_assert_msg(number_of_controls.size() == K, "The number of controls is not of size K.");
    fatrop_assert_msg(number_of_states.size() == K, "The number of states is not of size K.");
    fatrop_assert_msg(number_of_eq_constraints.size() == K, "The number of equality constraints is not of size K.");
    fatrop_assert_msg(number_of_ineq_constraints.size() == K, "The number of inequality constraints is not of size K.");
    fatrop_assert_msg(number_of_stage_border_variables.size() == K,
                      "The number of stage-border dimensions is not of size K.");
    // iterate over every time step and do some checks
    for (Index i = 0; i < K; i++) {
        fatrop_assert_msg(number_of_controls[i] >= 0, "The number of controls must be non-negative.");
        fatrop_assert_msg(number_of_states[i] >= 0, "The number of states must be non-negative.");
        fatrop_assert_msg(number_of_eq_constraints[i] >= 0, "The number of equality constraints must be non-negative.");
        fatrop_assert_msg(number_of_ineq_constraints[i] >= 0, "The number of inequality constraints must be non-negative.");
        fatrop_assert_msg(number_of_stage_border_variables[i] >= 0,
                          "The number of stage-border variables must be non-negative.");
        fatrop_assert_msg(number_of_eq_constraints[i] <= number_of_states[i]
                          + number_of_controls[i] + number_of_global_parameters
                          + number_of_stage_border_variables[i],
                          "The number of eq constraints (" + 
                          std::to_string(number_of_eq_constraints[i]) + 
                          ") exceeds the number of local and global variables (" +
                          std::to_string(number_of_states[i] + number_of_controls[i]
                                         + number_of_global_parameters
                                         + number_of_stage_border_variables[i]) + ").");
    }
}
