#ifndef __OCP_INTERFACE_GENERATOR_HPP__
#define __OCP_INTERFACE_GENERATOR_HPP__

#include "fatrop/context/context.hpp"
#include "fatrop/context/generic.hpp"
#include "fatrop/ocp/ocp_abstract.hpp"
#include "../test_problem/implicit_test_problem.hpp"    
#include "../test_problem/explicit_test_problem.hpp"
#include <nlohmann/json.hpp>

#include <casadi/casadi.hpp>

using namespace casadi;
using namespace fatrop;
using json = nlohmann::json;

// define abstract template for interface generator
class InterfaceGenerator{
    public:
        virtual ImplicitTestProblem PrepareImplicit() = 0;
        virtual ExplicitTestProblem PrepareRootFinder() = 0;
        virtual ExplicitTestProblem PrepareExplicit() = 0;
        virtual ExplicitTestProblem PrepareReformulated() = 0;
        virtual ~InterfaceGenerator() = default;
        virtual json GetJsonData() = 0;
        virtual std::string GetInterfaceName() = 0;
        virtual std::string GetFileNameAppendix() = 0;
};

#endif
