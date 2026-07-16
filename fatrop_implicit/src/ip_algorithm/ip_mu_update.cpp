#include "fatrop/ip_algorithm/ip_mu_update.hxx"
#include "fatrop/ocp/type.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/problem_info.hpp"


template class fatrop::IpMonotoneMuUpdate<fatrop::OcpType>;
template class fatrop::IpMonotoneMuUpdate<fatrop::ImplicitOcpType>;
template class fatrop::IpMonotoneMuUpdate<fatrop::AcceleratedOcpType>;