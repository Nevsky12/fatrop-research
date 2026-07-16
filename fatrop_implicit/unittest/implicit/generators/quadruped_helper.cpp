/*
// quadruped_dynamics.cpp
#include <iostream>
#include <vector>
#include <string>

// json
#include <json/single_include/nlohmann/json.hpp>

#include <casadi/casadi.hpp>

#include "quadruped_helper.hpp"

// If your Pinocchio exposes the integrate function template:
// #include <pinocchio/algorithm/liegroup.hpp>

using namespace casadi;
// using namespace pinocchio;

// Helper: build a CasADi SX 3-vector ground reaction force using same form as Python code
SX ground_reaction_force_SX(const casadi::SX& p, const casadi::SX& v)
{
    return SX::zeros(3,1);
    // p and v are 3x1 SX vectors (world frame)
    double k_ground = 200.0;   // N/m
    double d_ground = 600.0;   // N*s/m
    double alpha_k = 40.0;
    double alpha_d = 40.0;

    casadi::SX penetration = -p(2); // -pz

    auto sigmoid = [&](const casadi::SX& x) -> casadi::SX {
        // numerically safe sigmoid (same branch logic as python)
        // Here we just use casadi's exp; branchless for simplicity
        return 1.0 / (1.0 + casadi::SX::exp(-x));
    };

    casadi::SX s_d = sigmoid(alpha_d * penetration);
    casadi::SX fx = -d_ground * s_d * v(0);
    casadi::SX fy = -d_ground * s_d * v(1);
    casadi::SX fz = casadi::SX(k_ground) * casadi::SX::exp(alpha_k * penetration) - casadi::SX(d_ground) * s_d * v(2);

    casadi::SX f = casadi::SX::vertcat({fx, fy, fz});
    return f;
};


PinocchioCasadi::PinocchioCasadi(double dt) : timestep(dt)
{
    // pinocchio::buildModels::humanoidRandom(model);
    std::string urdf_path = "/home/u0143705/miniconda3/envs/pinocchio/share/example-robot-data/robots/anymal_b_simple_description/robots/anymal.urdf";
    // pinocchio::urdf::buildModel(urdf_path, model);
    pinocchio::urdf::buildModel(urdf_path, pinocchio::JointModelFreeFlyer(), model);

    cmodel = model.cast<ADScalar>();
    cdata = ADData(cmodel);

    // find foot frame ids (adjust names to your URDF)
    foot_frame_ids.clear();
    for (const auto &name : foot_names) {
        if (model.existFrame(name)) {
            foot_frame_ids.push_back(model.getFrameId(name));
        } else {
            std::cerr << "Warning: frame " << name << " not found in model\n";
        }
    }

    create_dynamics();
    create_discrete_dynamics();
};

void PinocchioCasadi::create_dynamics()
{
    int nq = model.nq;
    int nv = model.nv;
    int nu = 12; // actuation dimension

    SX q = SX::sym("q", nq);
    SX v = SX::sym("v", nv);
    SX u = SX::sym("u", nu);

    Eigen::MatrixXd B = Eigen::MatrixXd::Zero(model.nv, nu);
    // In Python B[6:, :] = eye(12)
    for (int i = 0; i < nu; ++i) {
        B(6 + i, i) = 1.0;
    }

    // Convert B to casadi SX dense matrix for multiplication
    SX Bsx = SX::zeros(nv, nu);
    for (int i = 0; i < nv; ++i)
        for (int j = 0; j < nu; ++j)
            Bsx(i,j) = B(i,j);

    // tau = B * u
    SX tau = mtimes(Bsx, u);

    ConfigVectorAD q_ad(nq);
    q_ad = Eigen::Map<ConfigVectorAD>(static_cast<std::vector<ADScalar>>(q).data(), nq, 1);
    TangentVectorAD v_ad(nv);
    v_ad = Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADScalar>>(v).data(), nv, 1);

    // Forward Kinematics and frame placements
    pinocchio::forwardKinematics(cmodel, cdata, q_ad, v_ad);
    pinocchio::updateFramePlacements(cmodel, cdata);
    
        // Now loop over feet, compute contact forces and add tau_contacts
    SX tau_contacts = SX::zeros(nv);
    for (int fid : foot_frame_ids)
    {
        // obtain position and velocity of feet
        // Frame placement translation (3x1)
        // cdata.oMf[fid].translation is of type SE3Tpl<Scalar>::Translation
        auto& placement = cdata.oMf[fid];
        // SX px = placement.translation[0];
        auto t = placement.translation();
        SX px = t(0);
        SX py = t(1);
        SX pz = t(2);
        SX p = SX::vertcat({px, py, pz});

        // Frame velocity -> DataTemplates store velocity as MotionTpl<Scalar>
        pinocchio::MotionTpl<ADScalar> v_frame = pinocchio::getFrameVelocity<ADScalar>(cmodel, cdata, fid, pinocchio::LOCAL_WORLD_ALIGNED);
        SX vx = v_frame.linear()(0);
        SX vy = v_frame.linear()(1);
        SX vz = v_frame.linear()(2);
        SX v = SX::vertcat({vx, vy, vz});

        // contact force as casadi SX expression
        SX f_contact = ground_reaction_force_SX(p, v);

        // Compute frame Jacobian (3 x nv)
        // We'll compute full Jacobian and take top 3 rows
        Eigen::Matrix<ADScalar, Eigen::Dynamic, Eigen::Dynamic> J_full(6, nv);
        // fill J_full with SX(0)
        for (int i = 0; i < J_full.rows(); ++i)
            for (int j = 0; j < J_full.cols(); ++j)
                J_full(i,j) = SX(0);
        
        pinocchio::computeFrameJacobian<ADScalar>(cmodel, cdata, q_ad, fid, pinocchio::LOCAL_WORLD_ALIGNED, J_full);

        // Keep only top 3 rows (linear)
        // Convert to casadi SX matrix:
        
        SX Jlin = SX::zeros(3, nv);
        for (int i = 0; i < 3; ++i){
            for (int j = 0; j < nv; ++j){
                Jlin(i,j) = J_full(i,j);
            }
        }
        
        // tau_contacts += Jlin^T * f_contact
        SX tau_ct = mtimes(transpose(Jlin), f_contact);
        tau_contacts = tau_contacts + tau_ct;
    }

    tau = tau + tau_contacts;
    
    // Compute ABA: a = aba(cmodel, cdata, q, v, tau)
    // For this we must call templated aba function. It expects q and v in a container.
    TangentVectorAD tau_ad(nv);
    tau_ad = Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADScalar>>(tau).data(), nv, 1);

    // re-use q_vec and v_vec (they were created above)
    pinocchio::aba(cmodel, cdata, q_ad, v_ad, tau_ad);
    
    SX a_ad(model.nv, 1);
    for (int i=0;i<nv;++i) a_ad(i) = cdata.ddq[i];

    // Create CasADi function: acc(q, v, u) -> (a, dummy)
    std::vector<SX> inputs = {q, v, u};
    std::vector<SX> outputs = {a_ad};

    acc_func = Function("acc", {q, v, u}, {a_ad}, {"q", "v", "u"}, {"a"});
}

void PinocchioCasadi::create_discrete_dynamics()
{
    // Build discrete integrator function using semi-implicit Euler
    int nq = model.nq;
    int nv = model.nv;
    int nu = 12;

    // define input vectors
    SX q = SX::sym("q", nq);
    SX v = SX::sym("v", nv);
    SX u = SX::sym("u", nu);
    SXVector qvu = {q, v, u};
    
    // compute accelerations
    SX a = acc_func(qvu)[0];

    SX vnext = v + a * timestep;

    // convert sx to tangentvectors
    ConfigVectorAD q_ad(nq);
    q_ad = Eigen::Map<ConfigVectorAD>(static_cast<std::vector<ADScalar>>(q).data(), nq, 1);
    TangentVectorAD v_ad(nv);
    v_ad = Eigen::Map<TangentVectorAD>(static_cast<std::vector<ADScalar>>(vnext).data(), nv, 1);

    ConfigVectorAD qnext = pinocchio::integrate(cmodel, q_ad, timestep * v_ad);

    SX qnext_sx = SX(nq, 1);
    for (int i=0;i<nq;++i) qnext_sx(i) = qnext(i);
    
    discrete_fn = Function("discrete_dyn", {q, v, u}, {qnext_sx, vnext}, {"q","v","u"}, {"qnext","vnext"});
    discrete_fn.save("discrete_in.casadi"); // save from an executable that does not contain casadi
    Function::load("discrete_in.casadi");   // load from an executable that only contains casadi
}

// Simple numeric forward using CasADi numeric evaluation (example)
std::vector<double> PinocchioCasadi::forward_numeric(const std::vector<double>& x, const std::vector<double>& u)
{
    int nq = model.nq;
    int nv = model.nv;
    
    DM q = DM::zeros(nq);
    DM v = DM::zeros(nv);
    for (int i=0;i<nq;++i) q(i) = x[i];
    for (int i=0;i<nv;++i) v(i) = x[nq + i];
    DM u_dm = DM::zeros(12);
    for (int i=0;i<12;++i) u_dm(i) = u[i];
    std::vector<DM> out = discrete_fn(DMVector{q, v, u_dm});
    std::vector<double> x_out;
    for (int i=0;i<nq;++i) x_out.push_back(double(out[0](i)));
    for (int i=0;i<nv;++i) x_out.push_back(double(out[1](i)));

    return x_out;
}

void PinocchioCasadi::SimulateFalling(){
    int nb_steps = 100;

    std::vector<double> x(model.nq + model.nv, 0.0);
    // standing pose
    std::vector<double> q_standing = {
        0, 0, 0.4792, 0, 0, 0, 1, -0.1, 0.7, -1, -0.1, -0.7, 1, 0.1, 0.7, -1, 0.1, -0.7, 1
    };
    for (int i=0;i<q_standing.size();++i) x[i] = q_standing[i];
    x[2] += 1; // lift up a bit

    // std::cout << "let's go" << std::endl;
    std::vector<double> u0(12, 0.0);
    std::vector<std::vector<double>> xx;
    xx.push_back(x);

    for (int i=0;i<nb_steps;++i){
        // std::cout << "caling forward_numeric" << std::endl;
        x = forward_numeric(x, u0);
        // std::cout << "\tstoring result" << std::endl;
        xx.push_back(x);
    }

    nlohmann::json j;
    j["states"] = xx;
    // write to file
    std::string json_file_name = "quadruped_falling_cpp.json";
    std::ofstream json_file(json_file_name);
    json_file << j.dump(4);
    json_file.close();
    
    // write results to file
    std::string file_name = "quadruped_falling_cpp.txt";
    std::ofstream file(file_name);
    for (const auto& xi : xx){
        for (double vi : xi) file << vi << " ";
        file << "\n";
    }
    file.close();
}


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