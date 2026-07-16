#include "fatrop/ocp/nlp_ocp.hxx"
using namespace fatrop;
// explicit template instantiation
template class fatrop::NlpOcpTpl<OcpAbstractDynamic, OcpType>; 
template class fatrop::NlpOcpTpl<ImplicitOcpAbstractDynamic, ImplicitOcpType>; 
template class fatrop::NlpOcpTpl<OcpAbstractDynamic, AcceleratedOcpType>;