//
// Copyright (c) 2024 Lander Vanroye, KU Leuven
//

#ifndef __fatrop_fatrop_hpp__
#define __fatrop_fatrop_hpp__

#include "fatrop/common/options.hpp"
#include "fatrop/common/timing.hpp"
#include "fatrop/ip_algorithm/ip_alg_builder.hpp"
#include "fatrop/ip_algorithm/ip_algorithm.hpp"
#include "fatrop/ip_algorithm/ip_data.hpp"
#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/ocp/global_parameter_kkt_solver.hpp"
#include "fatrop/ocp/interval_scoped_kkt_solver.hpp"
#include "fatrop/ocp/interval_scoped_ocp_kkt_solver.hpp"
#include "fatrop/ocp/interval_scoped_pd_solver.hpp"
#include "fatrop/ocp/phase_arrow_kkt_solver.hpp"
#include "fatrop/ocp/phase_arrow_ocp_kkt_solver.hpp"
#include "fatrop/ocp/phase_arrow_pd_solver.hpp"
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/nlp_ocp.hpp"
#include "fatrop/ocp/ocp_abstract.hpp"
#include "fatrop/ocp/parametric_implicit_nlp_ocp.hpp"
#include "fatrop/ocp/parametric_implicit_ocp.hpp"

#endif // __fatrop_fatrop_hpp__
