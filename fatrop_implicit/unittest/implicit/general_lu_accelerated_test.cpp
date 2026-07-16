#include "../random_matrix.hpp"
#include "fatrop/context/context.hpp"
#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/ocp/aug_system_solver.hpp"
#include "fatrop/ocp/dims.hpp" // inherit
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/problem_info.hpp" //inherit
#include "fatrop/ocp/type.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <random>

using namespace fatrop;

class AcceleratedAugSystemSolverTest : public ::testing::Test
{
// protected:
public:
    bool ignore_reformulated_structure = false;

    // Create OcpDims object
    int K;
    std::vector<Index> nx;
    std::vector<Index> r;
    std::vector<Index> nu;
    std::vector<Index> ng;
    std::vector<Index> ng_ineq;
    std::optional<ProblemDims> dims;
    std::optional<ProblemInfo> info;
    std::optional<Jacobian<OcpType>> jacobian_reference;
    std::optional<Hessian<OcpType>> hessian_reference;
    std::optional<Jacobian<AcceleratedOcpType>> jacobian;
    std::optional<Hessian<AcceleratedOcpType>> hessian;
    std::optional<MatRealAllocated> full_matrix_jacobian;
    std::optional<MatRealAllocated> full_matrix_hessian;
    std::optional<VecRealAllocated> x;
    std::optional<VecRealAllocated> mult;
    std::optional<VecRealAllocated> rhs_x;
    std::optional<VecRealAllocated> rhs_g;
    std::optional<VecRealAllocated> D_x;
    std::optional<VecRealAllocated> D_s;
    std::optional<VecRealAllocated> D_eq;
    std::optional<MatRealAllocated> full_kkt_matrix;
    std::optional<AugSystemSolver<OcpType>> solver_reference;
    std::optional<AugSystemSolver<AcceleratedOcpType>> solver;

    std::vector<int> RandomVector(int size, int min_val, int max_val)
    {
        std::vector<int> vec(size);
        for (int i = 0; i < size; ++i){
            vec[i] = rand() % (max_val - min_val + 1) + min_val;
        }
        return vec;
    }

    void ClearOptionals(){
        std::cout << "clearing optionals" << std::endl;
        solver_reference.reset();
        solver.reset();
        hessian_reference.reset();
        jacobian_reference.reset();
        hessian.reset();
        jacobian.reset();
        info.reset();
        dims.reset();
        full_matrix_jacobian.reset();
        full_matrix_hessian.reset();
        x.reset();
        mult.reset();
        rhs_x.reset();
        rhs_g.reset();
        D_x.reset();
        D_s.reset();
        D_eq.reset();
        full_kkt_matrix.reset();
        std::cout << "done" << std::endl;
    }

    void GetRandomDimensions()
    {
        ClearOptionals();
        int max_val = 10;
        K = rand() % max_val + 2; // Random K between 2 and 21
        // K = 3;
        nx = RandomVector(K, 0, max_val);
        r = std::vector<Index>(K, 100);
        for (int k = 0; k < K; ++k){ 
            while (r[k] > nx[k]){ r[k] = rand() % (nx[k]+1);}
        }
        nu = RandomVector(K, 0, max_val);
        ng = RandomVector(K, 0, max_val);
        for (int k = 0; k < K; ++k){
            bool okay = false;
            while (!okay){
                int max_allowed_ng = nx[k] + nu[k];
                if (k < K-1){ max_allowed_ng -= (nx[k+1] - r[k+1]);}
                if (ng[k] <= max_allowed_ng){
                    okay = true;
                } else {
                    // randomize both the nb of constraints and the nb of controls
                    ng[k] = rand() % (max_val + 1);
                    nu[k] = rand() % (max_val + 1);
                }
            }
        }
        ng_ineq = RandomVector(K, 0, max_val);

        // K = 2;
        // nx = {2, 2};
        // r = {2, 2};
        // nu = {2, 2};
        // ng = {2, 2};
        // std::cout << "nx[0]: " << nx[0] << std::endl;
        // nu[0] += nx[0];
        // nx[0] = 0;
        // int swap_val = 1;
        // nu[0] += swap_val; nx[0] -= swap_val;

        // int change_val = 1;
        // nx[0] -= change_val;
        // ng[0] -= change_val;

        // ng[K-1] -= 1;

        // reformulation
        for (int k = 0; k < K-1; ++k){
            nu[k] += nx[k+1];
            ng[k] += nx[k+1];
        }

        // print dimensions
        // std::cout << "int K = " << K << ";" << std::endl;
        // std::cout << "std::vector<Index> nx = {"; 
        // for (int i = 0; i < K; ++i){ std::cout << nx[i] << (i < K-1 ? ", " : "};\n");} 
        // std::cout << "std::vector<Index> r = {";
        // for (int i = 0; i < K; ++i){ std::cout << r[i] << (i < K-1 ? ", " : "};\n");}
        // std::cout << "std::vector<Index> nu = {";
        // for (int i = 0; i < K; ++i){ std::cout << nu[i] << (i < K-1 ? ", " : "};\n");}
        // std::cout << "std::vector<Index> ng = {";
        // for (int i = 0; i < K; ++i){ std::cout << ng[i] << (i < K-1 ? ", " : "};\n");}
        // std::cout << "std::vector<Index> ng_ineq = {";
        // for (int i = 0; i < K; ++i){ std::cout << ng_ineq[i] << (i < K-1 ? ", " : "};\n");}

        dims.emplace(ProblemDims{K, nu, nx, ng, ng_ineq});
        info.emplace(ProblemInfo(dims.value()));
        jacobian_reference.emplace(Jacobian<OcpType>(dims.value()));
        full_matrix_jacobian =
            MatRealAllocated(info->number_of_eq_constraints, info->number_of_primal_variables);
        hessian_reference.emplace(Hessian<OcpType>(dims.value()));
        jacobian.emplace(Jacobian<AcceleratedOcpType>(dims.value()));
        full_matrix_jacobian =
            MatRealAllocated(info->number_of_eq_constraints, info->number_of_primal_variables);
        hessian.emplace(Hessian<AcceleratedOcpType>(dims.value()));
        full_matrix_hessian =
            MatRealAllocated(info->number_of_primal_variables, info->number_of_primal_variables);
        x = VecRealAllocated(info->number_of_primal_variables);
        mult = VecRealAllocated(info->number_of_eq_constraints);
        rhs_x = VecRealAllocated(info->number_of_primal_variables);
        rhs_g = VecRealAllocated(info->number_of_eq_constraints);
        D_x = VecRealAllocated(info->number_of_primal_variables);
        D_s = VecRealAllocated(info->number_of_slack_variables);
        D_eq = VecRealAllocated(info->number_of_g_eq_path);
        full_kkt_matrix =
            MatRealAllocated(info->number_of_primal_variables + info->number_of_eq_constraints,
                             info->number_of_primal_variables + info->number_of_eq_constraints);
        solver_reference.emplace(AugSystemSolver<OcpType>(info.value()));
        solver.emplace(AugSystemSolver<AcceleratedOcpType>(info.value()));
    }

    void Randomize(){
        GetRandomDimensions();
        x = 0;
        full_matrix_jacobian.value() = 0.;
        full_matrix_hessian.value() = 0.;

        // fill the jacobian with random values
        for (Index k = 0; k < info.value().dims.K; ++k)
        {
            Index nu = info.value().dims.number_of_controls[k];
            Index nx = info.value().dims.number_of_states[k];
            Index offs_eq_dyn = info.value().offsets_g_eq_dyn[k];
            Index offs_ux = info.value().offsets_primal_u[k];
            Index offset_g_eq = info.value().offsets_g_eq_path[k];
            Index offset_g_ineq = info.value().offsets_g_eq_slack[k];
            Index ng = info.value().dims.number_of_eq_constraints[k];
            Index ng_ineq = info.value().dims.number_of_ineq_constraints[k];
            if (k < info.value().dims.K - 1)
            {
                Index nx_next = info.value().dims.number_of_states[k + 1];
                Index offs_x_next = info.value().offsets_primal_x[k + 1];
                if (ignore_reformulated_structure){
                    jacobian.value().BAbt[k].block(nu + nx, nx_next, 0, 0) =
                        ::test::random_matrix(nu + nx, nx_next);
                } else {
                    jacobian.value().BAbt[k].block(nu + nx, nx_next, 0, 0) =
                        ::test::empty_matrix(nu + nx, nx_next);
                    jacobian.value().BAbt[k].block(nx_next, nx_next, nu-nx_next, 0) = 
                        ::test::identity_matrix(nx_next, nx_next);
                }
                full_matrix_jacobian.value().block(nx_next, nu + nx, offs_eq_dyn, offs_ux) =
                transpose(jacobian.value().BAbt[k].block(nu + nx, nx_next, 0, 0));
            }
            jacobian.value().Gg_eqt[k].block(nu + nx, ng, 0, 0) =
                ::test::random_matrix(nu + nx, ng);
            if (!ignore_reformulated_structure && k < info.value().dims.K - 1){
                Index nx_next = info.value().dims.number_of_states[k + 1];
                jacobian.value().Gg_eqt[k].block(nx_next, ng-nx_next, nu-nx_next, 0) =
                    ::test::empty_matrix(nx_next, ng-nx_next);
            }
            
            full_matrix_jacobian.value().block(ng, nu + nx, offset_g_eq, offs_ux) =
                transpose(jacobian.value().Gg_eqt[k].block(nu + nx, ng, 0, 0));

            jacobian.value().Gg_ineqt[k].block(nu + nx, info.value().dims.number_of_ineq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info.value().dims.number_of_ineq_constraints[k]);
            full_matrix_jacobian.value().block(ng_ineq, nu + nx, offset_g_ineq, offs_ux) =
                transpose(jacobian.value().Gg_ineqt[k].block(nu + nx, ng_ineq, 0, 0));

        }
        // fill the Hessian with random values
        for (Index k = 0; k < dims.value().K; ++k)
        {
            Index nu = info.value().dims.number_of_controls[k];
            Index nx = info.value().dims.number_of_states[k];
            Index offs_ux = info.value().offsets_primal_u[k];
            hessian.value().RSQrqt[k].block(nu + nx, nu + nx, 0, 0) = ::test::random_spd_matrix(nu + nx);
            full_matrix_hessian.value().block(nu + nx, nu + nx, offs_ux, offs_ux) =
                hessian.value().RSQrqt[k].block(nu + nx, nu + nx, 0, 0);
        }

        // set up the full KKT matrix
        full_kkt_matrix.value().block(info.value().number_of_primal_variables, info.value().number_of_primal_variables, 0,
                              0) = full_matrix_hessian.value();
        full_kkt_matrix.value().block(info.value().number_of_primal_variables, info.value().number_of_eq_constraints, 0,
                              info.value().number_of_primal_variables) = transpose(full_matrix_jacobian.value());
        full_kkt_matrix.value().block(info.value().number_of_eq_constraints, info.value().number_of_primal_variables,
                              info.value().number_of_primal_variables, 0) = full_matrix_jacobian.value();

        // fill the x vector with random values
        for (Index i = 0; i < info.value().number_of_primal_variables; ++i)
        {
            rhs_x.value()(i) = 1.0 * i;
            D_x.value()(i) = 10.0 * (i + 10.1);
        }
        // fill the mult vector with random values
        for (Index i = 0; i < info.value().number_of_eq_constraints; ++i)
        {
            rhs_g.value()(i) = 10.0 * (i + 10);
        }

        for (Index i = 0; i < info.value().number_of_g_eq_path; ++i)
        {
            D_eq.value()(i) = 10.0 * (i + 10);
        }
        for (Index i = 0; i < info.value().number_of_slack_variables; ++i)
        {
            D_s.value()(i) =  1.0 + 10.0 * (i + 0.1);
        }
    }

    void SetUp()
    {
        int seed = time(0);
        // int seed = 1776841101; //--> failure case for max_val = 10
        // int seed = 1776846089; //--> failure case for max_val = 10
        // int seed = 1776846641; //--> significant failure for max_val = 10 --> only in absolute error (not in relative)
        srand(seed);
        std::cout << "int seed = " << seed << ";" << std::endl;
        Randomize();
    };
};


void CheckSolution(const ProblemInfo &info,
                   const Jacobian<OcpType> &jacobian,
                   const Hessian<OcpType> &hessian,
                   const VecRealView &D_x,
                   const VecRealView &D_s,
                   const VecRealView &rhs_x,
                   const VecRealView &rhs_g,
                   const VecRealView &x,
                   const VecRealView &mult)
{
    VecRealAllocated jac_x(info.number_of_eq_constraints);
    jacobian.apply_on_right(info, x, 0.0, jac_x, jac_x);
    VecRealAllocated rhs_gg(info.number_of_eq_constraints);
    rhs_gg = 0.;
    rhs_gg = rhs_gg + rhs_g + jac_x;
    rhs_gg.block(info.number_of_slack_variables, info.offset_g_eq_slack) =
        rhs_gg.block(info.number_of_slack_variables, info.offset_g_eq_slack) -
        D_s * mult.block(info.number_of_slack_variables, info.offset_g_eq_slack);
    VecRealAllocated grad(info.number_of_primal_variables);
    VecRealAllocated tmp(info.number_of_primal_variables);
    grad = 0;
    hessian.apply_on_right(info, x, 0.0, tmp, tmp);
    grad = grad + tmp + D_x * x;
    jacobian.transpose_apply_on_right(info, mult, 0.0, tmp, tmp);
    grad = grad + tmp;
    grad = grad + rhs_x;
    double max_rhs_gg = 0.;
    for (Index i = 0; i < info.number_of_eq_constraints; ++i){
        max_rhs_gg = std::max(max_rhs_gg, std::abs(rhs_gg(i)));
    }
    EXPECT_NEAR(max_rhs_gg, 0, 1e-5);
    double max_grad = 0.;
    for (Index i = 0; i < info.number_of_primal_variables; ++i){
        max_grad = std::max(max_grad, std::abs(grad(i)));
    }
    EXPECT_NEAR(max_grad, 0, 1e-5);
    // std::cout << "grad: " << grad << std::endl;
}

void PrintFullKKT(const ProblemInfo &info,
                   const MatRealView &full_kkt_matrix,
                   const VecRealView &rhs_x,
                   const VecRealView &rhs_g,
                   const VecRealView &D_x,
                   const VecRealView &D_s,
                   const VecRealView &x,
                   const VecRealView &mult,
                   std::ostream& out = std::cout){
    out << "KKT = np.array([\n";
    for (Index i = 0; i < full_kkt_matrix.m(); i++){
        out << "\t[";
        for (Index j = 0; j < full_kkt_matrix.n(); j++){
            out << std::setw(9) << std::setprecision(6) << full_kkt_matrix(i,j);
            if (j < full_kkt_matrix.n() - 1){
                out << ", ";
            }
        }
        out << "]";
        if (i < full_kkt_matrix.m() - 1){
            out << ",\n";
        }
    }
    out << "\n])" << std::endl;

    VecRealAllocated full_rhs = VecRealAllocated(info.number_of_primal_variables + info.number_of_eq_constraints);
    for (Index i = 0; i < info.number_of_primal_variables; ++i){full_rhs(i) = rhs_x(i) + D_x(i)*x(i);}
    for (Index i = 0; i < info.number_of_eq_constraints; ++i){full_rhs(info.number_of_primal_variables + i) = rhs_g(i);}
    for (Index i = 0; i < info.number_of_slack_variables; ++i){
        full_rhs(info.number_of_primal_variables + info.offset_g_eq_slack + i) -= D_s(i) * mult(info.offset_g_eq_slack + i);
    }
    out << "rhs = np.array([\n";
    for (Index i = 0; i < full_rhs.m(); i++){
        out << "\t" << full_rhs(i);
        if (i < full_rhs.m() - 1){
            out << ",";
        }
    }
    out << "\n])" << std::endl;

    // print obtained solution //
    out << "Obtained solution x:" << std::endl << x << std::endl;
    std::cout << info.offsets_primal_x[0] << " " << info.offsets_primal_u[0] << " " << info.offsets_primal_x[1] << " " << info.offsets_primal_u[1] << std::endl;
    out << "Obtained solution mult:" << std::endl << mult << std::endl;
}
void PrintKKTSparsity(const MatRealView &full_kkt_matrix, std::ostream& out = std::cout){
    for (Index i = 0; i < full_kkt_matrix.m(); i++){
        for (Index j = 0; j < full_kkt_matrix.n(); j++){
            if (std::abs(full_kkt_matrix(i,j)) > 1e-8){
                out << "X";
            } else {
                out << ".";
            }
        }
        out << "\n";
    }
    out << std::endl;
}


TEST_F(AcceleratedAugSystemSolverTest, TestRandomSolve)
{
    // int seed = time(0);
    // srand(seed);
    // std::cout << "int seed = " << seed << ";" << std::endl;
    double overall_max_diff = 0;
    double overall_max_diff_rel = 0;
    for (int test_counter = 0; test_counter < 1; ++test_counter){
        std::cout << "\n" << std::endl;
        std::cout << "==============================" << std::endl;
        std::cout << "Test iteration: " << test_counter << std::endl;
        std::cout << "==============================" << std::endl;
        this->Randomize();
        std::cout << "--------------------- Solving accelerated solver ---------------------" << std::endl;
        Index ret = solver.value().solve(info.value(), 
            jacobian.value(), hessian.value(), D_x.value(),
            D_s.value(), rhs_x.value(), rhs_g.value(), 
            x.value(), mult.value());
        std::cout << "Solver return flag: " << ret << std::endl;
        // EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);
        // solver.value().TestPermutationFunctions(info.value(), 0);
        std::cout << "-------------------------------  Done. -------------------------------" << std::endl;
        
        VecRealAllocated x_reference(info.value().number_of_primal_variables);
        VecRealAllocated mult_reference(info.value().number_of_eq_constraints);
        std::cout << "----------------------- Solving original solver ----------------------" << std::endl;
        Index ret_reference = solver_reference.value().solve(info.value(), 
            jacobian.value(), hessian.value(), D_x.value(),
            D_s.value(), rhs_x.value(), rhs_g.value(), 
            x_reference, mult_reference);
        // EXPECT_EQ(ret_reference, LinsolReturnFlag::SUCCESS);
        std::cout << "Solver return flag: " << ret_reference << std::endl;
        std::cout << "-------------------------------  Done. -------------------------------" << std::endl;
        
        EXPECT_EQ(ret, ret_reference);
        
        double max_diff_x = 0.;
        double max_diff_x_rel = 0.;
        int max_diff_x_idx = -1;
        int max_diff_x_rel_idx = -1;
        for (Index i = 0; i < info.value().number_of_primal_variables; ++i){
            double diff = std::abs(x.value()(i) - x_reference(i));
            max_diff_x = std::max(max_diff_x, diff);
            max_diff_x_idx = (max_diff_x == diff) ? i : max_diff_x_idx;
            max_diff_x_rel = std::max(max_diff_x_rel, diff / std::max(1e-8, std::abs(x_reference(i))));
            max_diff_x_rel_idx = (max_diff_x_rel == diff / std::max(1e-8, std::abs(x_reference(i)))) ? i : max_diff_x_rel_idx;
        }
        double max_diff_mult = 0.;
        double max_diff_mult_rel = 0.;
        int max_diff_mult_idx = -1;
        int max_diff_mult_rel_idx = -1;
        for (Index i = 0; i < info.value().number_of_eq_constraints; ++i){
            double diff = std::abs(mult.value()(i) - mult_reference(i));
            max_diff_mult = std::max(max_diff_mult, diff);
            max_diff_mult_idx = (max_diff_mult == diff) ? i : max_diff_mult_idx;
            max_diff_mult_rel = std::max(max_diff_mult_rel, diff / std::max(1e-8, std::abs(mult_reference(i))));
            max_diff_mult_rel_idx = (max_diff_mult_rel == diff / std::max(1e-8, std::abs(mult_reference(i)))) ? i : max_diff_mult_rel_idx;
        }
        overall_max_diff = std::max(overall_max_diff, std::max(max_diff_x, max_diff_mult));
        overall_max_diff_rel = std::max(overall_max_diff_rel, std::max(max_diff_x_rel, max_diff_mult_rel));
        // std::cout << "my x:        " << x.value() << std::endl;
        // std::cout << "reference x: " << x_reference << std::endl;
        // std::cout << "\nmy mult:        " << mult.value() << std::endl;
        // std::cout << "reference mult: " << mult_reference << std::endl;
        std::cout << "max rel diff x: " << x.value()(max_diff_x_rel_idx) << " vs " << x_reference(max_diff_x_rel_idx) << " - diff: " << max_diff_x_rel << std::endl;
        std::cout << "max rel diff mult: " << mult.value()(max_diff_mult_rel_idx) << " vs " << mult_reference(max_diff_mult_rel_idx) << " - diff: " << max_diff_mult_rel << std::endl;

        // print max diff info
        // if (max_diff_mult > 1e-5){
        //     // print out all mult differences
        //     for (Index i = 0; i < info.value().number_of_eq_constraints; ++i){
        //         double diff = std::abs(mult.value()(i) - mult_reference(i));
        //         double rel_diff = diff / std::max(1e-8, std::abs(mult_reference(i)));
        //         if (diff > 1e-5){
        //             std::cout << "mult difference at index " << std::setw(3) << i << ": " << std::setw(15) << diff << " - " << std::setw(15) << rel_diff << std::endl;
        //         }
        //     }

        //     // print sorted values of mult
        //     std::vector<double> mult_sorted(mult.value().data(), mult.value().data() + mult.value().m());
        //     std::vector<double> mult_reference_sorted(mult_reference.data(), mult_reference.data() + mult_reference.m());
        //     std::sort(mult_sorted.begin(), mult_sorted.end());
        //     std::sort(mult_reference_sorted.begin(), mult_reference_sorted.end());

        //     for (int i = 0; i < mult_sorted.size(); ++i){
        //         std::cout << std::setw(15) << mult_sorted[i] << " " << mult_reference_sorted[i] << std::endl;
        //     }
        // }

        // EXPECT_NEAR(max_diff_x, 0, 1e-5);
        // EXPECT_NEAR(max_diff_mult, 0, 1e-5);
        EXPECT_NEAR(max_diff_x_rel, 0, 1e-7);
        EXPECT_NEAR(max_diff_mult_rel, 0, 1e-7);

        // Solution checking
        std::cout << "checking my solution: " << std::endl;
        CheckSolution(info.value(), jacobian.value(), 
            hessian.value(), D_x.value(), D_s.value(), 
            rhs_x.value(), rhs_g.value(), x.value(),
            mult.value());
        std::cout << "done checking\n" << std::endl;
        std::cout << "checking reference solution: " << std::endl;
        CheckSolution(info.value(), jacobian.value(), 
            hessian.value(), D_x.value(), D_s.value(), 
            rhs_x.value(), rhs_g.value(), x_reference,
            mult_reference);
        std::cout << "done checking\n" << std::endl;

        /*
        std::cout << "Pl_rank modified: ";
        bool modified = false;
        for (int i = 0; i < solver.value().Pl_rank[0].size(); i++){
            if (solver.value().Pl_rank[0][i] != i){
                modified = true;
                break;
            }
        }



        std::cout << (modified ? "yes" : "no") << std::endl;
        if (true || modified){
            std::cout << "Pl_rank: " << solver.value().Pl_rank[0] << std::endl;
            std::cout << "rank: " << solver.value().rho1[0] << std::endl;
            std::cout << "nx: " << info.value().dims.number_of_states[1] << std::endl;
            std::cout << "gamma: " << solver.value().gamma[0] << std::endl;
            std::cout << "r1 == ng_true: " << (solver.value().rho1[0] == solver.value().gamma[0] - info.value().dims.number_of_states[1]) << std::endl;
            std::cout << "r1 == nu_true: " << (solver.value().rho1[0] == info.value().dims.number_of_controls[0] - info.value().dims.number_of_states[1]) << std::endl;

            // append to file: modified,rank,nx,gamma,r1_eq_ng_true,r1_eq_nu_true
            std::ofstream file;
            file.open("failure_cases.csv", std::ios_base::app);    
            file << max_diff_mult << "," 
                 << max_diff_x << ","
                 << (modified ? "1" : "0") << "," 
                 << solver.value().rho1[0] << "," 
                 << info.value().dims.number_of_states[1] << "," 
                 << solver.value().gamma[0] << "," 
                 << (solver.value().rho1[0] == solver.value().gamma[0] - info.value().dims.number_of_states[1]) << "," 
                 << (solver.value().rho1[0] == info.value().dims.number_of_controls[0] - info.value().dims.number_of_states[1]) << "\n";
            file.close();
        }
        */
    }
    std::cout << "\noverall max diff:   " << overall_max_diff << std::endl;
    std::cout << "overall max diff rel: " << overall_max_diff_rel << std::endl;
}