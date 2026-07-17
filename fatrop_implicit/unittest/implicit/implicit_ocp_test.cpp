
//
// Copyright (c) Lander Vanroye, KU Leuven
//
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
#include "generators/batch_reactor_generator.hpp"
#include "generators/solar_receiver_reactor_generator.hpp"
#include "generators/manipulator_path_follower_generator.hpp"
#if __has_include("generators/pendulum_generator.hpp")
#include "generators/pendulum_generator.hpp"
#define FATROP_HAS_LEGACY_PENDULUM_GENERATOR 1
#else
#define FATROP_HAS_LEGACY_PENDULUM_GENERATOR 0
#endif

#include <nlohmann/json.hpp>

#include <fstream>
#include <tuple>

using namespace fatrop;
using json = nlohmann::json;

template <typename T>
double get_average(const std::vector<T>& v){
    double sum = 0.0;
    for (const auto& val : v) {sum += val;}
    return sum / v.size();
}

template <typename ProblemType>
json add_json_data(std::shared_ptr<IpData<ProblemType>> data, std::string problem_type)
{
    json my_json;
    my_json["metadata"] = json::object();
    my_json["metadata"]["timing_statistics"] = json::object();
    my_json["metadata"]["timing_statistics"]["initialization"] = data->timing_statistics().initialization.elapsed();
    my_json["metadata"]["timing_statistics"]["compute search dir"] = data->timing_statistics().compute_search_dir.elapsed();
    my_json["metadata"]["timing_statistics"]["eval objective"] = data->timing_statistics().eval_objective.elapsed();
    my_json["metadata"]["timing_statistics"]["eval gradient"] = data->timing_statistics().eval_gradient.elapsed();
    my_json["metadata"]["timing_statistics"]["eval constraint violation"] = data->timing_statistics().eval_constraint_violation.elapsed();
    my_json["metadata"]["timing_statistics"]["eval hessian"] = data->timing_statistics().eval_hessian.elapsed();
    my_json["metadata"]["timing_statistics"]["eval jacobian"] = data->timing_statistics().eval_jacobian.elapsed();
    my_json["metadata"]["timing_statistics"]["rest time"] = data->timing_statistics().compute_rest_time();
    my_json["metadata"]["timing_statistics"]["function evaluation"] = data->timing_statistics().compute_function_evaluation();
    my_json["metadata"]["timing_statistics"]["fatrop"] = data->timing_statistics().compute_fatrop();
    my_json["metadata"]["timing_statistics"]["total"] = data->timing_statistics().full_algorithm.elapsed();
    my_json["metadata"]["iterations"] = data->iteration_number();

    my_json["problem type"] = problem_type;
    my_json["solver"] = "FATROP";

    my_json["states"] = json::array();
    for (int k = 0; k < data->info().dims.K; k++)
    {
        json state = json::array();
        for (int i = 0; i < data->info().dims.number_of_states[k]; i++)
        {
            state.push_back(data->current_iterate().primal_x()(data->info().offsets_primal_x[k] + i));
        }
        my_json["states"].push_back(state);
    }
    my_json["inputs"] = json::array();
    for (int k = 0; k < data->info().dims.K-1; k++)
    {
        json input = json::array();
        for (int i = 0; i < data->info().dims.number_of_controls[k]; i++)
        {
            input.push_back(data->current_iterate().primal_x()(data->info().offsets_primal_u[k] + i));
        }
        my_json["inputs"].push_back(input);
    }

    return my_json;
};

// void PrintStatistics(std::vector<bool> solved_flags, std::vector<json> results){
//     std::cout << "\n\n==================== Summary ====================" << std::endl;
//     std::cout << "Number of iterations:\n";
//     for (size_t i = 0; i < results.size(); i++){
//         if (solved_flags[i]){
//             std::cout << "\t" << std::setw(30) << results[i]["method_name"] << ": " << results[i]["metadata"]["iterations"] << std::endl;
//         }
//     }
//     std::cout << "Total computation time (s) - Time per iteration (s):\n";
//     for (size_t i = 0; i < results.size(); i++){
//         if (solved_flags[i]){
//             std::cout << "\t" << std::setw(30) << results[i]["method_name"] << ": ";
//             std::cout << std::setw(30) << std::setprecision(2) << results[i]["metadata"]["timing_statistics"]["total"];
//             double time_per_iter = double(results[i]["metadata"]["timing_statistics"]["total"]) / int(results[i]["metadata"]["iterations"]);
//             std::cout << " - " << std::setw(15) << std::setprecision(2) << time_per_iter << std::endl;
//         }
//     }
//     std::cout << "Function evaluation time (s) - Time per iteration (s):\n";
//     for (size_t i = 0; i < results.size(); i++){
//         if (solved_flags[i]){
//             std::cout << "\t" << std::setw(30) << results[i]["method_name"] << ": ";
//             std::cout << std::setw(30) << std::setprecision(2) << results[i]["metadata"]["timing_statistics"]["function evaluation"];
//             double time_per_iter = double(results[i]["metadata"]["timing_statistics"]["function evaluation"]) / int(results[i]["metadata"]["iterations"]);
//             std::cout << " - " << std::setw(15) << std::setprecision(2) << time_per_iter << std::endl;
//         }
//     }
//     std::cout << "Fatrop time (s) - Time per iteration (s):\n";
//     for (size_t i = 0; i < results.size(); i++){
//         if (solved_flags[i]){
//             std::cout << "\t" << std::setw(30) << results[i]["method_name"] << ": ";
//             std::cout << std::setw(30) << std::setprecision(2) << results[i]["metadata"]["timing_statistics"]["fatrop"];
//             double time_per_iter = double(results[i]["metadata"]["timing_statistics"]["fatrop"]) / int(results[i]["metadata"]["iterations"]);
//             std::cout << " - " << std::setw(15) << std::setprecision(2) << time_per_iter << std::endl;
//         }
//     }
// }
void PrintStatistics(std::vector<bool> solved_flags, std::vector<json> results) {
    // Collect only solved methods
    std::vector<size_t> solved_idx;
    for (size_t i = 0; i < results.size(); i++)
        if (solved_flags[i]) solved_idx.push_back(i);

    if (solved_idx.empty()) {
        std::cout << "\n\nNo solved methods to display.\n";
        return;
    }

    // --- Column width configuration ---
    const int ROW_LABEL_W = 40;   // width of the leftmost label column
    const int COL_W       = 16;   // width of each method column
    const int PRECISION   =  4;   // decimal places for floats

    // Helper: format a double as a fixed-precision string
    auto fmt = [&](double v) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(PRECISION) << v;
        return oss.str();
    };

    // Helper: center a string within a field of given width
    auto center = [](const std::string& s, int w) {
        int pad = w - (int)s.size();
        int lpad = pad / 2, rpad = pad - lpad;
        return std::string(lpad, ' ') + s + std::string(rpad, ' ');
    };

    int total_width = ROW_LABEL_W + (int)solved_idx.size() * (COL_W + 1) + 1;
    std::string separator(total_width, '-');

    std::cout << "\n\n==================== Summary ====================\n";

    // --- Header row ---
    std::cout << std::string(ROW_LABEL_W, ' ') << "|";
    for (size_t i : solved_idx) {
        std::string name = results[i]["method_name"];
        // Truncate if too long
        if ((int)name.size() > COL_W - 1) name = name.substr(0, COL_W - 1);
        std::cout << center(name, COL_W) << "|";
    }
    std::cout << "\n" << separator << "\n";

    // --- Row definitions ---
    // Each row: { label, lambda returning cell string for result index i }
    using RowFn = std::function<std::string(size_t)>;
    std::vector<std::pair<std::string, RowFn>> rows = {
        {
            "Iterations",
            [&](size_t i) { return std::to_string(int(results[i]["metadata"]["iterations"])); }
        },
        {
            "Total time (s)",
            [&](size_t i) { return fmt(double(results[i]["metadata"]["timing_statistics"]["total"])); }
        },
        {
            "Func eval time (s)",
            [&](size_t i) { return fmt(double(results[i]["metadata"]["timing_statistics"]["function evaluation"])); }
        },
        {
            "Fatrop time (s)",
            [&](size_t i) { return fmt(double(results[i]["metadata"]["timing_statistics"]["fatrop"])); }
        },
        {
            "Search dir time (s)",
            [&](size_t i) { return fmt(double(results[i]["metadata"]["timing_statistics"]["compute search dir"])); }
        },
        {
            "Total time / iter (ms)",
            [&](size_t i) {
                double t = double(results[i]["metadata"]["timing_statistics"]["total"]);
                int   it = int  (results[i]["metadata"]["iterations"]);
                return fmt(1000 * t / it);
            }
        },
        {
            "Func eval time / iter (ms)",
            [&](size_t i) {
                double t = double(results[i]["metadata"]["timing_statistics"]["function evaluation"]);
                int   it = int  (results[i]["metadata"]["iterations"]);
                return fmt(1000 * t / it);
            }
        },
        {
            "Fatrop time / iter (ms)",
            [&](size_t i) {
                double t = double(results[i]["metadata"]["timing_statistics"]["fatrop"]);
                int   it = int  (results[i]["metadata"]["iterations"]);
                return fmt(1000 * t / it);
            }
        },
        {
            "Search dir time / iter (ms)",
            [&](size_t i) {
                double t = double(results[i]["metadata"]["timing_statistics"]["compute search dir"]);
                int   it = int  (results[i]["metadata"]["iterations"]);
                return fmt(1000 * t / it);
            }
        }
    };

    // --- Data rows ---
    for (auto& [label, fn] : rows) {
        std::cout << std::left << std::setw(ROW_LABEL_W) << label << "|";
        for (size_t i : solved_idx) {
            std::cout << center(fn(i), COL_W) << "|";
        }
        std::cout << "\n";
    }

    std::cout << separator << "\n";
}

void SolveProblem(std::unique_ptr<InterfaceGenerator> &generator){
    bool STORE_SOLUTION = true;
    bool USE_CODEGEN = false;
    std::cout << "preparing implicit" << std::endl;
    auto tp_impl = std::make_shared<ImplicitTestProblem>(generator->PrepareImplicit());
    std::cout << "preparing root finder" << std::endl;
    auto tp_rf = std::make_shared<ExplicitTestProblem>(generator->PrepareRootFinder());
    std::cout << "preparing reformulated" << std::endl;
    auto tp_reform = std::make_shared<ExplicitTestProblem>(generator->PrepareReformulated());
    std::cout << "preparing explicit" << std::endl;
    auto tp_expl = std::make_shared<ExplicitTestProblem>(generator->PrepareExplicit());
    std::cout << "preparing accelerated" << std::endl;
    auto tp_accel = std::make_shared<ExplicitTestProblem>(generator->PrepareReformulated());
    tp_impl->use_codegen(USE_CODEGEN);
    // tp_rf->use_codegen(USE_CODEGEN);
    tp_reform->use_codegen(USE_CODEGEN);
    tp_expl->use_codegen(USE_CODEGEN);
    tp_accel->use_codegen(USE_CODEGEN);
    ImplicitTestProblem tp_interface_impl = *tp_impl;
    ExplicitTestProblem tp_interface_rf = *tp_rf;
    ExplicitTestProblem tp_interface_expl = *tp_expl;
    ExplicitTestProblem tp_interface_reform = *tp_reform;
    ExplicitTestProblem tp_interface_accel = *tp_accel;
    // show_implicit_interface_output(tp_interface_impl, "output_interface_implicit.txt");
    // show_interface_output(tp_interface_rf, "output_interface_rootfinder.txt");
    // show_interface_output(tp_interface_expl, "output_interface_explicit.txt");
    // show_interface_output(tp_interface_reform, "output_interface_reformulated.txt");

    std::cout << "Generated test problems" << std::endl;
    std::string gen_type = generator->GetInterfaceName();
    std::string file_name_appendix = generator->GetFileNameAppendix();

    OptionRegistry options;
    IpAlgBuilder<ImplicitOcpType> builder_impl(std::make_shared<ImplicitNlpOcp>(tp_impl));
    IpAlgBuilder<OcpType> builder_rf(std::make_shared<NlpOcp>(tp_rf));
    IpAlgBuilder<OcpType> builder_expl(std::make_shared<NlpOcp>(tp_expl));
    IpAlgBuilder<OcpType> builder_reform(std::make_shared<NlpOcp>(tp_reform));
    IpAlgBuilder<AcceleratedOcpType> builder_accel(std::make_shared<AcceleratedNlpOcp>(tp_accel));    

    std::shared_ptr<IpAlgorithm<ImplicitOcpType>> ipalg_impl = builder_impl.with_options_registry(&options).build();
    std::shared_ptr<IpAlgorithm<OcpType>> ipalg_rf = builder_rf.with_options_registry(&options).build();
    std::shared_ptr<IpAlgorithm<OcpType>> ipalg_expl = builder_expl.with_options_registry(&options).build();
    std::shared_ptr<IpAlgorithm<OcpType>> ipalg_reform = builder_reform.with_options_registry(&options).build();
    std::shared_ptr<IpAlgorithm<AcceleratedOcpType>> ipalg_accel = builder_accel.with_options_registry(&options).build();
    // options.set_option("mu_init", 100.0);
    options.set_option("max_iter", 400);
    // options.set_option("print_level", 0);
    options.set_option("tolerance", 1e-4);
    std::cout << "built ip algorithms" << std::endl;

    json result_expl, result_reform, result_impl, result_rf, result_accel;
    bool solved_expl = false, solved_reform = false, solved_impl = false, solved_rf = false, solved_accel = false;
    
    bool skip_expl = false;
    bool skip_reform = true;
    bool skip_impl = true;
    bool skip_rf = true;
    bool skip_accel = true;

    // EXPLICIT
    if (!skip_expl){
    try{
        std::cout << "solving explicit test problem" << std::endl;
        Timer timer_expl; timer_expl.start();
        IpSolverReturnFlag ret_expl = ipalg_expl->optimize();
        std::cout << "Elapsed time: " << timer_expl.stop() << std::endl;
        auto data_expl = builder_expl.get_ipdata();
        result_expl = add_json_data(data_expl, "explicit");
        result_expl["method_name"] = "explicit";
        result_expl["generator_data"] = generator->GetJsonData();
        if (STORE_SOLUTION){
            std::ofstream file2("ocp_results/ocp_result_explicit_" + gen_type + "_" + file_name_appendix + ".json");
            if (file2.is_open())
            {
                file2 << result_expl.dump(4);
                file2.close();
            }
        }
        solved_expl = ret_expl == IpSolverReturnFlag::Success;
    } catch (std::exception& e){
        std::cout << "Exception caught during explicit solve: " << e.what() << std::endl;
    }
    }

    // IMPLICIT
    if (!skip_impl){
    try{
        std::cout << "solving implicit test problem" << std::endl;
        Timer timer_impl; timer_impl.start();
        IpSolverReturnFlag ret_impl = ipalg_impl->optimize();
        std::cout << "Elapsed time: " << timer_impl.stop() << std::endl;
        auto data_impl = builder_impl.get_ipdata();
        result_impl = add_json_data(data_impl, "implicit");
        result_impl["method_name"] = "implicit";
        result_impl["generator_data"] = generator->GetJsonData();
        std::ofstream file("ocp_results/ocp_result_implicit_" + gen_type + "_" + file_name_appendix + ".json");
        if (STORE_SOLUTION){
            if (file.is_open())
            {
                file << result_impl.dump(4);
                file.close();
            }
        }
        solved_impl = ret_impl == IpSolverReturnFlag::Success;
    } catch (std::exception& e){
        std::cout << "Exception caught during implicit solve: " << e.what() << std::endl;
    }
    }

    // REFORMULATED
    if (!skip_reform){
    try{
        std::cout << "solving reformulated test problem" << std::endl;
        Timer timer_reform; timer_reform.start();
        IpSolverReturnFlag ret_reform = ipalg_reform->optimize();
        std::cout << "Elapsed time: " << timer_reform.stop() << std::endl;
        auto data_reform = builder_reform.get_ipdata();
        result_reform = add_json_data(data_reform, "reformulated");
        result_reform["method_name"] = "reformulated";
        result_reform["generator_data"] = generator->GetJsonData();
        std::ofstream file3("ocp_results/ocp_result_reformulated_" + gen_type + "_" + file_name_appendix + ".json");
        if (STORE_SOLUTION){
            if (file3.is_open())
            {
                file3 << result_reform.dump(4);
                file3.close();
            }
        }
        solved_reform = ret_reform == IpSolverReturnFlag::Success;
    } catch (std::exception& e){
        std::cout << "Exception caught during reformulated solve: " << e.what() << std::endl;
    }
    }
    
    // ROOTFINDER
    if (!skip_rf){
    try{
        std::cout << "solving rootfinder test problem" << std::endl;
        Timer timer_rf; timer_rf.start();
        IpSolverReturnFlag ret_rf = ipalg_rf->optimize();
        std::cout << "Elapsed time: " << timer_rf.stop() << std::endl;
        auto data_rf = builder_rf.get_ipdata();
        result_rf = add_json_data(data_rf, "rootfinder");
        result_rf["method_name"] = "rootfinder";
        result_rf["generator_data"] = generator->GetJsonData();
        std::ofstream file4("ocp_results/ocp_result_rootfinder_" + gen_type + "_" + file_name_appendix + ".json");
        if (STORE_SOLUTION){
            if (file4.is_open())
            {
                file4 << result_rf.dump(4);
                file4.close();
            }
        }
        solved_rf = ret_rf == IpSolverReturnFlag::Success;
    } catch (std::exception& e){
        std::cout << "Exception caught during rootfinder solve: " << e.what() << std::endl;
    }
    }

    // Accelerated
    if (!skip_accel){
    try{
        std::cout << "solving accelerated test problem" << std::endl;
        Timer timer_accel; timer_accel.start();
        IpSolverReturnFlag ret_accel = ipalg_accel->optimize();
        std::cout << "Elapsed time: " << timer_accel.stop() << std::endl;
        auto data_accel = builder_accel.get_ipdata();
        result_accel = add_json_data(data_accel, "accelerated");
        result_accel["method_name"] = "accelerated";
        result_accel["generator_data"] = generator->GetJsonData();
        std::ofstream file5("ocp_results/ocp_result_accelerated_" + gen_type + "_" + file_name_appendix + ".json");
        if (STORE_SOLUTION){
            if (file5.is_open())
            {
                file5 << result_accel.dump(4);
                file5.close();
            }
        }
        solved_accel = ret_accel == IpSolverReturnFlag::Success;
    } catch (std::exception& e){
        std::cout << "Exception caught during accelerated solve: " << e.what() << std::endl;
    }
    }
    PrintStatistics({solved_expl, solved_reform, solved_accel, solved_impl, solved_rf}, {result_expl, result_reform, result_accel, result_impl, result_rf});
}

void SolveSingleProblemTruckTrailer(int n_trailers){
    std::unique_ptr<InterfaceGenerator> generator = 
        std::make_unique<TruckTrailerGenerator>(n_trailers);
    SolveProblem(generator);
}
void SolveAllTruckTrailer(){
    for (int n_trailers = 0; n_trailers <= 10; n_trailers++){
        SolveSingleProblemTruckTrailer(n_trailers);
    }
}

void SolveSingleProblemBycicle(){
    std::unique_ptr<InterfaceGenerator> generator = 
        std::make_unique<BycicleGenerator>();
    SolveProblem(generator);
}

void SolveSingleProblemExampleStatic(){
    std::unique_ptr<InterfaceGenerator> generator = 
        std::make_unique<ExampleStaticGenerator>();
    SolveProblem(generator);
}

void SolveSingleProblemHolonomic(int n, int control_level){
    std::unique_ptr<InterfaceGenerator> generator = 
        std::make_unique<HolonomicInterfaceGenerator>(n, control_level);
    SolveProblem(generator);
}
void SolveAllHolonomic(){
    for (int n = 1; n <= 7; n++){
        for (int control_level = n == 1 ? 2 : 1; control_level <= 4; control_level++){
            SolveSingleProblemHolonomic(n, control_level);
        }
    }
}

void SolveSingleProblemPlanarRobot(int n_links){
    std::unique_ptr<InterfaceGenerator> generator = 
        std::make_unique<PlanarRobot>(n_links);
    SolveProblem(generator);
}
void SolveAllPlanarRobot(){
    for (int n_links = 1; n_links <= 5; n_links++){
        SolveSingleProblemPlanarRobot(n_links);
    }
}

void SolveSingleProblemQuadruped(double vx, double vy){
    std::unique_ptr<InterfaceGenerator> generator = 
        std::make_unique<QuadrupedGenerator>(vx, vy);
    SolveProblem(generator);
}
void SolveAllQuadruped(){
    double vel = 2.0;
    std::vector<double> angles = {0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75};
    for (double angle : angles){
        double vx = vel*cos(angle*3.1415);
        double vy = vel*sin(angle*3.1415);
        SolveSingleProblemQuadruped(vx, vy);
    }
}

void SolveSingeBatchReactor(int n){
    std::unique_ptr<InterfaceGenerator> generator = 
        std::make_unique<BatchReactorGenerator>(n);
    SolveProblem(generator);
}
void SolveAllBatchReactor(){
    for (int n = 0; n <= 10; n++){
        SolveSingeBatchReactor(n);
    }
}

void SolveSingeSolarReceiverReactor(int n){
    std::unique_ptr<InterfaceGenerator> generator = 
        std::make_unique<SolarReceiverReactorGenerator>(n);
    SolveProblem(generator);
}
void SolveAllSolarReceiverReactor(){
    for (int n = 0; n <= 10; n++){
        SolveSingeSolarReceiverReactor(n);
    }
}

int main(int argc, char **argv)
{
    // // create a directory ocp_results
    // int temp = system("mkdir -p ocp_results");

    // SolarReceiverReactorGenerator srrg = SolarReceiverReactorGenerator(0);
    // srrg.SolveOptiInstance("ipopt");
    // srrg.SolveOptiInstance("fatrop");
    // TruckTrailerGenerator ttg = TruckTrailerGenerator(2);
    // auto tp =  std::make_shared<ExplicitTestProblem>(ttg.PrepareRootFinder());
    // IpAlgBuilder<OcpType> builder_expl(std::make_shared<NlpOcp>(tp));
    // OptionRegistry options;
    // std::shared_ptr<IpAlgorithm<OcpType>> ipalg_expl = builder_expl.with_options_registry(&options).build();
    // ipalg_expl->optimize();
    
    
    // std::unique_ptr<InterfaceGenerator> temp = std::make_unique<TruckTrailerGenerator>(2);
    // try{
    //     // ExplicitTestProblem tp = temp->PrepareExplicit();
    //     // print_jac_and_hess(tp);
    //     SolveProblem(temp);
    // } catch (std::exception& e){
    //     std::cout << "Exception caught during batch reactor solve: " << e.what() << std::endl;
    // }
    // return 0;

    // ManipulatorPathFollower pf;
    // pf.PrintDimensions();
    // pf.SolveOptiInstance();
    // return 0;

    // PendulumGenerator pg;
    // pg.SolveOptiInstance();
    // return 0;


    if (argc < 3){
        std::cout << "Please provide the following arguments to this executable:" << std::endl;
        std::cout << "\t\"single\" or \"all\"" << std::endl;
        std::cout << "\tproblem name (truck_trailer, bycicle, example_static, holonomic, planar_robot, quadruped, batch_reactor, solar_receiver_reactor)" << std::endl;
        std::cout << "in case of a single problem, also provide the parameters desired (optionally)" << std::endl;
        return 0;
    }

    if (std::string(argv[1]) == "single"){
        std::unique_ptr<InterfaceGenerator> generator;

        if (std::string(argv[2]) == "truck_trailer"){
            int n_trailers = 1;
            if (argc > 3){ n_trailers = std::stoi(argv[3]);}
            generator = std::make_unique<TruckTrailerGenerator>(n_trailers);

        } else if (std::string(argv[2]) == "bycicle"){
            generator = std::make_unique<BycicleGenerator>();
        } else if (std::string(argv[2]) == "example_static"){
            generator = std::make_unique<ExampleStaticGenerator>();
        } else if (std::string(argv[2]) == "holonomic"){
            int n = 3;
            int control_level = 2;

            if (argc > 3){ n = std::stoi(argv[3]);}
            if (argc > 4){ control_level = std::stoi(argv[4]);}

            generator = std::make_unique<HolonomicInterfaceGenerator>(n, control_level);
        } else if (std::string(argv[2]) == "planar_robot"){
            int n_links = 3;
            if (argc > 3){ n_links = std::stoi(argv[3]);}
            generator = std::make_unique<PlanarRobot>(n_links);
        } else if (std::string(argv[2]) == "quadruped"){
            double vx = 0.0;
            double vy = 0.0;
            if (argc > 3){ vx = std::stod(argv[3]);}
            if (argc > 4){ vy = std::stod(argv[4]);}
            generator = std::make_unique<QuadrupedGenerator>(vx, vy);
        } else if (std::string(argv[2]) == "batch_reactor"){
            int n = 0;
            if (argc > 3){ n = std::stoi(argv[3]);}
            generator = std::make_unique<BatchReactorGenerator>(n);
        } else if (std::string(argv[2]) == "solar_receiver_reactor"){
            int n = 0;
            if (argc > 3){ n = std::stoi(argv[3]);}
            generator = std::make_unique<SolarReceiverReactorGenerator>(n);
        } else if (std::string(argv[2]) == "path_follower"){
            generator = std::make_unique<ManipulatorPathFollower>();
        } else if (std::string(argv[2]) == "pendulum"){
#if FATROP_HAS_LEGACY_PENDULUM_GENERATOR
            generator = std::make_unique<PendulumGenerator>();
#else
            std::cerr << "The legacy pendulum generator is not present in this source tree."
                      << std::endl;
            return 1;
#endif
        } else {
            std::cout << "Second argument should be either \"truck_trailer\", \"bycicle\", \"example_static\", \"holonomic\", \"planar_robot\", \"quadruped\", \"batch_reactor\" \"solar_receiver_reactor\" or \"path_follower\" when first argument is \"single\"" << std::endl;
            return 0;
        }

        SolveProblem(generator);

    } else if (std::string(argv[1]) == "all"){
        if (std::string(argv[2]) == "truck_trailer"){
            SolveAllTruckTrailer();
        } else if (std::string(argv[2]) == "holonomic"){
            SolveAllHolonomic();
        } else if (std::string(argv[2]) == "planar_robot"){
            SolveAllPlanarRobot();
        } else if (std::string(argv[2]) == "quadruped"){
            SolveAllQuadruped();
        } else if (std::string(argv[2]) == "batch_reactor"){
            SolveAllBatchReactor();
        } else if (std::string(argv[2]) == "solar_receiver_reactor"){
            SolveAllSolarReceiverReactor();
        } else {
            std::cout << "Second argument should be either \"truck_trailer\", \"holonomic\". \"planar_robot\" or \"quadruped\" when first argument is \"all\"" << std::endl;
            return 0;
        }

    } else {
        std::cout << "First argument should be either \"single\" or \"all\"" << std::endl;
        return 0;
    }

    return 0;
}
