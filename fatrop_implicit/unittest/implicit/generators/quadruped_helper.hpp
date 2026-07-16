#ifndef __QUADRUPED_HELPER__
#define __QUADRUPED_HELPER__

/*
// quadruped_dynamics.cpp
#include <iostream>
#include <vector>
#include <string>

#include <casadi/casadi.hpp>

#include "pch.hpp"


// If your Pinocchio exposes the integrate function template:
// #include <pinocchio/algorithm/liegroup.hpp>

using namespace casadi;
// using namespace pinocchio;

// Helper: build a CasADi SX 3-vector ground reaction force using same form as Python code
SX ground_reaction_force_SX(const casadi::SX& p, const casadi::SX& v);

class PinocchioCasadi
{
public:
    // types templated on casadi::SX:
    typedef double Scalar;
    typedef SX ADScalar;
    typedef pinocchio::ModelTpl<Scalar> Model;
    typedef Model::Data Data;
    typedef pinocchio::ModelTpl<ADScalar> ADModel;
    typedef ADModel::Data ADData;

    Model model;
    std::unique_ptr<Data> data;
    ADModel cmodel;
    ADData cdata;

    typedef ADModel::ConfigVectorType ConfigVectorAD;
    typedef ADModel::TangentVectorType TangentVectorAD;
 
    double timestep;

    Function acc_func;     // casadi Function returning acceleration
    Function discrete_fn;  // explicit discrete integrator
    std::vector<std::string> foot_names = {"LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"};
    std::vector<int> foot_frame_ids;

    PinocchioCasadi(double dt=0.05);

    void create_dynamics();
    void create_discrete_dynamics();

    // Simple numeric forward using CasADi numeric evaluation (example)
    std::vector<double> forward_numeric(const std::vector<double>& x, const std::vector<double>& u);
    void SimulateFalling();
};

// int main(int argc, char** argv)
// {
//     if (argc < 2) {
//         std::cerr << "Usage: ./quadruped_dynamics <path_to_robot.urdf>\n";
//         return 1;
//     }

//     std::string urdf = argv[1];
//     PinocchioCasadi pc(urdf, 0.05);

//     std::cout << "Created Pinocchio-CasADi dynamics objects.\n";
//     std::cout << "acc_func name: " << pc.acc_func.name() << "\n";
//     std::cout << "discrete_fn name: " << pc.discrete_fn.name() << "\n";
//     return 0;
// }
*/
#endif