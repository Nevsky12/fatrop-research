#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"

#include "fatrop/common/timing.hpp"
#include "fatrop/common/options.hpp"
#include "fatrop/ip_algorithm/ip_alg_builder.hpp"
#include "fatrop/ip_algorithm/ip_algorithm.hpp"
#include "fatrop/ip_algorithm/ip_data.hpp"
#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/ocp/nlp_ocp.hpp"
#include "fatrop/ocp/ocp_abstract.hpp"
#include "generators/ocp_interface_generator.hpp"
#include "generators/holonomic_generator.hpp"
#include "generators/truck_trailer_generator.hpp"
#include "generators/bycicle_generator.hpp"
#include "generators/example_static_generator.hpp"
#include "generators/show_interface_output.hpp"
#include "generators/n_link_planar_robot.hpp"
#include "generators/quadruped_generator.hpp"

#include "json/single_include/nlohmann/json.hpp"

#include <fstream>
#include <tuple>
#include <ctime>

using namespace fatrop;
using json = nlohmann::json;

std::pair<double, double> PerformTest(ImplicitTestProblem& tp_impl){
    // prepare
    srand((unsigned) time(NULL));
    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    double us_approach_1 = 0.0;
    double us_approach_2 = 0.0;

    int nx = tp_impl.get_nx(0);
    int nu = tp_impl.get_nu(0);
    int nx_next = tp_impl.get_nx(1);

    VecRealAllocated xk(nx);
    VecRealAllocated uk(nu);
    VecRealAllocated xkp(nx_next);
    for (int i = 0; i < xk.m(); i++) xk(i) = ((double) rand() / RAND_MAX);
    for (int i = 0; i < uk.m(); i++) uk(i) = ((double) rand() / RAND_MAX);
    for (int i = 0; i < xkp.m(); i++) xkp(i) = ((double) rand() / RAND_MAX);
    std::cout << "xk: " << xk << std::endl;
    std::cout << "uk: " << uk << std::endl;
    std::cout << "xkp: " << xkp << std::endl;

    // approach 1: request J_inv, J, A, B from interface and perform multiplication
    //-------------------------------------------------------------------------
    MatRealAllocated BAbt1(nx + nu + 1, nx_next);
    MatRealAllocated BAbt_tilde1(nx + nu + 1, nx_next);
    MatRealAllocated Jt1(nx_next, nx_next);
    MatRealAllocated Jt_inv1(nx_next, nx_next);
    
    start = std::chrono::high_resolution_clock::now();
    tp_impl.eval_BAJbt(xkp.data(), uk.data(), xk.data(), &BAbt1.mat(), 
                       &Jt1.mat(), &Jt_inv1.mat(), 0);
    blasfeo_dgemm_nn(nx + nu + 1, nx_next, nx_next, 1.0, 
                    &BAbt1.mat(), 0, 0, &Jt_inv1.mat(), 0, 0, 0.0,
                    &BAbt_tilde1.mat(), 0, 0, &BAbt_tilde1.mat(), 0, 0);
    stop = std::chrono::high_resolution_clock::now();
    us_approach_1 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

    // approach 2: request J, A, B from interface and solve linear systems
    //-------------------------------------------------------------------------
    MatRealAllocated BAbt2(nx + nu + 1, nx_next);
    MatRealAllocated BAbt_tilde2(nx + nu + 1, nx_next);
    MatRealAllocated Jt2(nx_next, nx_next);
    MatRealAllocated LU(nx_next, nx_next);
    MatRealAllocated Y(nx + nu + 1, nx_next);
    
    start = std::chrono::high_resolution_clock::now();
    tp_impl.eval_BAJbt_no_inverse(xkp.data(), uk.data(), xk.data(), 
                                  &BAbt2.mat(), &Jt2.mat(), 0);
    // solve linear system Jt * X = BAbt2^T   
    blasfeo_dgetrf_np(nx_next, nx_next, &Jt2.mat(), 0, 0, &LU.mat(), 0, 0);
    blasfeo_dtrsm_runn(BAbt2.m(), BAbt2.n(), 1.0, &LU.mat(), 0, 0, &BAbt2.mat(), 0, 0, &Y.mat(), 0, 0);   
    blasfeo_dtrsm_rlnu(BAbt2.m(), BAbt2.n(), 1.0, &LU.mat(), 0, 0, &Y.mat(), 0, 0, &BAbt_tilde2.mat(), 0, 0);
    
    stop = std::chrono::high_resolution_clock::now();
    us_approach_2 = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();

    // print timings
    std::cout << "Time approach 1 (with inverse):    " << us_approach_1 << " us" << std::endl;
    std::cout << "Time approach 2 (without inverse): " << us_approach_2 << " us" << std::endl;

    // check result
    MatRealAllocated diff(nx + nu + 1, nx_next);
    for (int i = 0; i < diff.m(); i++) for (int j = 0; j < diff.n(); j++) diff(i,j) = 0.0;
    blasfeo_dgead(nx + nu + 1, nx_next, 1.0, &BAbt_tilde1.mat(), 0, 0, &diff.mat(), 0, 0);
    blasfeo_dgead(nx + nu + 1, nx_next, -1.0, &BAbt_tilde2.mat(), 0, 0, &diff.mat(), 0, 0);
    // std::cout << "difference between two approaches:" << std::endl;
    // blasfeo_print_dmat(diff.m(), diff.n(), &diff.mat(), 0, 0);

    return std::make_pair(us_approach_1, us_approach_2);
}

int normal_main(int argc, char **argv){
    std::unique_ptr<InterfaceGenerator> generator;

    if (std::string(argv[1]) == "truck_trailer"){
        int n_trailers = 1;
        if (argc > 2){ n_trailers = std::stoi(argv[2]);}
        generator = std::make_unique<TruckTrailerGenerator>(n_trailers);
    } else if (std::string(argv[1]) == "holonomic"){
        int n = 3;
        int control_level = 2;

        if (argc > 2){ n = std::stoi(argv[2]);}
        if (argc > 3){ control_level = std::stoi(argv[3]);}

        generator = std::make_unique<HolonomicInterfaceGenerator>(n, control_level);
    } else if (std::string(argv[1]) == "planar_robot"){
        int n_links = 3;
        if (argc > 2){ n_links = std::stoi(argv[2]);}
        generator = std::make_unique<PlanarRobot>(n_links);
    } else if (std::string(argv[1]) == "quadruped"){
        double vx = 0.0;
        double vy = 0.0;
        if (argc > 2){ vx = std::stod(argv[2]);}
        if (argc > 3){ vy = std::stod(argv[3]);}
        generator = std::make_unique<QuadrupedGenerator>(vx, vy);
    } else {
        std::cout << "Unknown problem name. Available: truck_trailer, holonomic, planar_robot, quadruped" << std::endl;
        return 0;
    }

    int nb_runs = 1000;
    ImplicitTestProblem tp_impl = generator->PrepareImplicit();
    std::vector<double> us_approach_1(nb_runs, 0);
    std::vector<double> us_approach_2(nb_runs, 0);
    for (int i = 0; i < nb_runs; i++){
        std::pair<double, double> result = PerformTest(tp_impl);
        us_approach_1[i] = result.first;
        us_approach_2[i] = result.second;
    }

    double avg_1 = 0.0;
    double avg_2 = 0.0;
    for (int i = 0; i < nb_runs; i++){
        avg_1 += us_approach_1[i];
        avg_2 += us_approach_2[i];
    }
    avg_1 /= nb_runs;
    avg_2 /= nb_runs;

    std::cout << "Average time approach 1 (with inverse):    " << avg_1 << " us" << std::endl;
    std::cout << "Average time approach 2 (without inverse): " << avg_2 << " us" << std::endl;

    return 0;
}

int main(int argc, char **argv){
    if (argc  < 2){
        std::cout << "Please provide the following arguments to this executable:" << std::endl;
        std::cout << "\tproblem name (truck_trailer, holonomic, planar_robot)" << std::endl;
        return 0;
    }
    std::unique_ptr<InterfaceGenerator> generator;

    if (std::string(argv[1]) == "quadruped"){
        return normal_main(argc, argv);
    }

    std::vector<int> n_trailers_list(31);
    for (int i = 0; i < 31; i++) n_trailers_list[i] = i;

    for (int n_trailers : n_trailers_list){
        std::cout << "generating problem with " << n_trailers << " trailers" << std::endl;
        generator = std::make_unique<TruckTrailerGenerator>(n_trailers);
        ImplicitTestProblem tp_impl = generator->PrepareImplicit();
    }
    return 0;

    if (std::string(argv[1]) == "all"){
        // setup tests
        int nb_runs = 1000;
        std::vector<int> n_trailers_list(31);
        for (int i = 0; i < 31; i++) n_trailers_list[i] = i;

        // setup containers for results
        std::vector<std::vector<double>> us_inverse(n_trailers_list.size(),
            std::vector<double>(nb_runs, 0.0));
        std::vector<std::vector<double>> us_factorization(n_trailers_list.size(),
            std::vector<double>(nb_runs, 0.0));

        for (int n_trailers : n_trailers_list){
            std::cout << "generating problem with " << n_trailers << " trailers" << std::endl;
            generator = std::make_unique<TruckTrailerGenerator>(n_trailers);
            ImplicitTestProblem tp_impl = generator->PrepareImplicit();

            for (int i = 0; i < nb_runs; i++){
                std::pair<double, double> result = PerformTest(tp_impl);
                us_inverse[n_trailers][i] = result.first;
                us_factorization[n_trailers][i] = result.second;
            }

            // save results to json
            json j;
            j["n_trailers"] = n_trailers_list;
            j["us_inverse"] = us_inverse;
            j["us_factorization"] = us_factorization;
            std::ofstream o("jacobian_decomposition_results.json");
            o << std::setw(4) << j << std::endl;
        }
        
        return 0;
    } else {
        return normal_main(argc, argv);
    }
}