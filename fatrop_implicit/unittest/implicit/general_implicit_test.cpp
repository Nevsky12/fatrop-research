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

class GeneralImplicitAugSystemSolverTest : public ::testing::Test
{
// protected:
public:
    bool J_matrix_is_idendity = false;
    bool J_matrix_full_rank = false;
    bool no_second_order_effects = false;

    // Create OcpDims object
    // int K = 10;                                                   // Number of stages
    // std::vector<Index> nx = {20, 10, 10, 10, 10, 2, 0, 1, 10, 5}; // State dimensions for each stage
    // std::vector<Index> r =  {20, 5, 2, 10, 9, 1, 0, 1, 6, 1};
    // std::vector<Index> nu = {1, 4, 2, 10, 1, 30, 4, 5, 10, 2};    // Input dimensions for each stage
    // std::vector<Index> ng = {9, 3, 4, 3, 4, 2, 1, 0, 1, 5}; // Equality constraints for each stage
    // std::vector<Index> ng_ineq = {0, 0*5, 0*10, 0*4, 0, 0, 0, 0, 0*10, 0}; // Inequality constraints for each stage
    // int K = 12;
    // std::vector<Index> nx = {4, 5, 17, 0, 9, 6, 0, 19, 9, 16, 15, 12};
    // std::vector<Index> r =  {4, 5, 9, 0, 3, 5, 0, 2, 2, 14, 11, 8};
    // std::vector<Index> nu = {10, 10, 20, 15, 9, 9, 20, 13, 15, 17, 12, 15};
    // std::vector<Index> ng = {0, 4, 20, 6, 5, 12, 2, 19, 17, 0, 15, 18};
    // std::vector<Index> ng_ineq = {19, 18, 8, 7, 0, 15, 10, 18, 20, 2, 6, 10};
    // int K = 3;
    // std::vector<Index> nx = {2, 3, 2};
    // std::vector<Index> r =  {0, 2, 1};
    // std::vector<Index> nu = {2, 2, 1};
    // std::vector<Index> ng = {2, 2, 1};
    // std::vector<Index> ng_ineq = {0, 0, 0};
    // int K = 2;
    // std::vector<Index> nx = {2, 3};
    // std::vector<Index> r =  {0, 2};
    // std::vector<Index> nu = {2, 2};
    // std::vector<Index> ng = {2, 2};
    // std::vector<Index> ng_ineq = {0, 0};
    // int K = 6;
    // std::vector<Index> nx = {3, 10, 3, 6, 6, 3};
    // std::vector<Index> r = {0, 5, 3, 3, 1, 2};
    // std::vector<Index> nu = {7, 3, 0, 10, 6, 9};
    // std::vector<Index> ng = {4, 6, 0, 5, 4, 5};
    // std::vector<Index> ng_ineq = {0, 0, 0, 0, 0, 0};
    // int K = 11;
    // std::vector<Index> nx = {6, 0, 2, 2, 5, 6, 7, 10, 2, 7, 5};
    // std::vector<Index> r = {6, 0, 1, 1, 2, 2, 7, 1, 2, 7, 0};
    // std::vector<Index> nu = {4, 5, 7, 7, 8, 1, 4, 10, 4, 8, 9};
    // std::vector<Index> ng = {4, 0, 1, 5, 5, 1, 1, 1, 4, 3, 6};
    // std::vector<Index> ng_ineq = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int K = 2;
    std::vector<Index> nx = {1, 2};
    std::vector<Index> r = {0, 1};
    std::vector<Index> nu = {0, 0};
    std::vector<Index> ng = {0, 0};
    std::vector<Index> ng_ineq = {0, 1};

    ProblemDims dims{K, nu, nx, ng, ng_ineq};

    ProblemInfo info{dims};
    // Create Jacobian object
    Jacobian<ImplicitOcpType> jacobian{dims};
    MatRealAllocated full_matrix_jacobian =
        MatRealAllocated(info.number_of_eq_constraints, info.number_of_primal_variables);
    Hessian<ImplicitOcpType> hessian{dims};
    MatRealAllocated full_matrix_hessian =
        MatRealAllocated(info.number_of_primal_variables, info.number_of_primal_variables);
    VecRealAllocated x = VecRealAllocated(info.number_of_primal_variables);
    VecRealAllocated mult = VecRealAllocated(info.number_of_eq_constraints);
    VecRealAllocated rhs_x = VecRealAllocated(info.number_of_primal_variables);
    VecRealAllocated rhs_g = VecRealAllocated(info.number_of_eq_constraints);
    VecRealAllocated D_x = VecRealAllocated(info.number_of_primal_variables);
    VecRealAllocated D_s = VecRealAllocated(info.number_of_slack_variables);
    VecRealAllocated D_eq = VecRealAllocated(info.number_of_g_eq_path);
    MatRealAllocated full_kkt_matrix =
        MatRealAllocated(info.number_of_primal_variables + info.number_of_eq_constraints,
                         info.number_of_primal_variables + info.number_of_eq_constraints);
    AugSystemSolver<ImplicitOcpType> solver = AugSystemSolver<ImplicitOcpType>(info);

    void SetUp()
    {
        int seed = time(0);
        std::cout << "seed: " << seed << std::endl;
        srand(seed);
        x = 0;
        full_matrix_jacobian = 0.;
        full_matrix_hessian = 0.;

        if (J_matrix_full_rank){ r = nx;}

        // fill the jacobian with random values
        for (Index k = 0; k < info.dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            if (k < info.dims.K - 1)
            {
                Index nx_next = info.dims.number_of_states[k + 1];
                jacobian.BAbt[k].block(nu + nx, nx_next, 0, 0) =
                    ::test::random_matrix(nu + nx, nx_next);

                if (J_matrix_is_idendity){
                    jacobian.Jt[k].block(r[k+1], r[k+1], 0, 0) =
                        ::test::identity_matrix(r[k+1], -1);
                } else {
                    jacobian.Jt[k].block(nx_next, nx_next, 0, 0) =
                        ::test::random_degenerate_matrix(nx_next, r[k+1]);
                }
            }
            jacobian.Gg_eqt[k].block(nu + nx, info.dims.number_of_eq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info.dims.number_of_eq_constraints[k]);
            jacobian.Gg_ineqt[k].block(nu + nx, info.dims.number_of_ineq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info.dims.number_of_ineq_constraints[k]);
        }
        // fill the Hessian with random values
        for (Index k = 0; k < dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            hessian.RSQrqt[k].block(nu + nx, nu + nx, 0, 0) = ::test::random_spd_matrix(nu + nx);
        }
        // add dynamics constraints
        for (Index k = 0; k < info.dims.K - 1; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index offs_ux = info.offsets_primal_u[k];
            Index offs_x_next = info.offsets_primal_x[k + 1];
            Index nx_next = info.dims.number_of_states[k + 1];
            Index offs_eq_dyn = info.offsets_g_eq_dyn[k];
            full_matrix_jacobian.block(nx_next, nu + nx, offs_eq_dyn, offs_ux) =
                transpose(jacobian.BAbt[k].block(nu + nx, nx_next, 0, 0));
            full_matrix_jacobian.block(nx_next, nx_next, offs_eq_dyn, offs_x_next) = 
                transpose(jacobian.Jt[k]);
            hessian.FuFx[k].block(nx + nu, nx_next, 0, 0) =
                ::test::random_matrix(nx + nu, nx_next);
            if (no_second_order_effects){
                hessian.FuFx[k].block(nx + nu, nx_next, 0, 0) =
                    ::test::empty_matrix(nx + nu, nx_next);
            } else {
                hessian.FuFx[k].block(nx + nu, nx_next, 0, 0) =
                    ::test::random_matrix(nx + nu, nx_next, 0.0, 0.1);
            }
            full_matrix_hessian.block(nx_next, nu + nx, offs_x_next, offs_ux) = 
                transpose(hessian.FuFx[k]);
            full_matrix_hessian.block(nu + nx, nx_next, offs_ux, offs_x_next) =
                hessian.FuFx[k];
        }
        // equality path equality constraints
        for (Index k = 0; k < info.dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index ng = info.dims.number_of_eq_constraints[k];
            Index offset_ux = info.offsets_primal_u[k];
            Index offset_g_eq = info.offsets_g_eq_path[k];
            full_matrix_jacobian.block(ng, nu + nx, offset_g_eq, offset_ux) =
                transpose(jacobian.Gg_eqt[k].block(nu + nx, ng, 0, 0));
        }
        // inequality path constraints
        for (Index k = 0; k < info.dims.K; ++k)
        {
            Index nu = info.dims.number_of_controls[k];
            Index nx = info.dims.number_of_states[k];
            Index ng_ineq = info.dims.number_of_ineq_constraints[k];
            Index offset_ux = info.offsets_primal_u[k];
            Index offset_g_ineq = info.offsets_g_eq_slack[k];
            full_matrix_jacobian.block(ng_ineq, nu + nx, offset_g_ineq, offset_ux) =
                transpose(jacobian.Gg_ineqt[k].block(nu + nx, ng_ineq, 0, 0));
        }
        // populate the full matrix
        for (Index k = 0; k < dims.K; k++)
        {
            Index nu = dims.number_of_controls[k];
            Index nx = dims.number_of_states[k];
            Index offs_ux = info.offsets_primal_u[k];
            full_matrix_hessian.block(nu + nx, nu + nx, offs_ux, offs_ux) =
                hessian.RSQrqt[k].block(nu + nx, nu + nx, 0, 0);
        }
        // set up the full KKT matrix
        full_kkt_matrix.block(info.number_of_primal_variables, info.number_of_primal_variables, 0,
                              0) = full_matrix_hessian;
        full_kkt_matrix.block(info.number_of_primal_variables, info.number_of_eq_constraints, 0,
                              info.number_of_primal_variables) = transpose(full_matrix_jacobian);
        full_kkt_matrix.block(info.number_of_eq_constraints, info.number_of_primal_variables,
                              info.number_of_primal_variables, 0) = full_matrix_jacobian;

        // fill the x vector with random values
        for (Index i = 0; i < info.number_of_primal_variables; ++i)
        {
            rhs_x(i) = 1.0 * i;
            D_x(i) = 0*1.0 * (i + 0.1);
        }
        // fill the mult vector with random values
        for (Index i = 0; i < info.number_of_eq_constraints; ++i)
        {
            rhs_g(i) = 1.0 * i;
        }

        for (Index i = 0; i < info.number_of_g_eq_path; ++i)
        {
            D_eq(i) = 0 * 10.0 * (i + 1);
        }
        for (Index i = 0; i < info.number_of_slack_variables; ++i)
        {
            D_s(i) =  2 + 0*10.0 * (i + 0.1);
        }

        // Compute LU factorization to check the rank of the constraint jacobian
        PermutationMatrix Pl(info.number_of_eq_constraints);
        PermutationMatrix Pr(info.number_of_primal_variables);
        MatRealAllocated At(full_matrix_jacobian.n(), full_matrix_jacobian.m());
        for (int i = 0; i < full_matrix_jacobian.m(); i++){
            for (int j = 0; j < full_matrix_jacobian.n(); j++){
                At(j,i) = full_matrix_jacobian(i,j);
            }
        }
        Index rank;
        fatrop_lu_fact_transposed(At.n(), At.m(), At.m(), rank, &At.mat(), Pl, Pr, 1e-5);
        std::cout << "jacobian dimensions: " << full_matrix_jacobian.m() << " x " << full_matrix_jacobian.n() << std::endl;
        std::cout << "rank: " << rank << "(" << info.number_of_eq_constraints << " x " << info.number_of_primal_variables << ")" << std::endl;
    };
};

template <typename T>
class RandomAugSystemSolverTest : public ::testing::Test
{
// protected:
public:
    bool full_rank = false;
    bool no_second_order_effects = false;

    // Create OcpDims object
    int K;
    std::vector<Index> nx;
    std::vector<Index> r;
    std::vector<Index> nu;
    std::vector<Index> ng;
    std::vector<Index> ng_ineq;
    std::optional<ProblemDims> dims;
    std::optional<ProblemInfo> info;
    std::optional<Jacobian<T>> jacobian;
    std::optional<Hessian<T>> hessian;
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
    std::optional<AugSystemSolver<T>> solver;

    std::vector<int> RandomVector(int size, int min_val, int max_val)
    {
        std::vector<int> vec(size);
        for (int i = 0; i < size; ++i){
            vec[i] = rand() % (max_val - min_val + 1) + min_val;
        }
        return vec;
    }

    void ClearOptionals(){
        solver.reset();
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
    }

    void GetRandomDimensions()
    {
        ClearOptionals();
        int max_val = 10;
        K = rand() % max_val + 2; // Random K between 2 and 21
        nx = RandomVector(K, 0, max_val);
        if (full_rank){
            r = nx;
        } else {
            r = std::vector<Index>(K, 100);
            for (int k = 0; k < K; ++k){ 
                while (r[k] > nx[k]){ r[k] = rand() % (nx[k]+1);}
            }
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
        ng_ineq = RandomVector(K, 0, 0*max_val);

        // print dimensions
        std::cout << "int K = " << K << ";" << std::endl;
        std::cout << "std::vector<Index> nx = {"; 
        for (int i = 0; i < K; ++i){ std::cout << nx[i] << (i < K-1 ? ", " : "};\n");} 
        std::cout << "std::vector<Index> r = {";
        for (int i = 0; i < K; ++i){ std::cout << r[i] << (i < K-1 ? ", " : "};\n");}
        std::cout << "std::vector<Index> nu = {";
        for (int i = 0; i < K; ++i){ std::cout << nu[i] << (i < K-1 ? ", " : "};\n");}
        std::cout << "std::vector<Index> ng = {";
        for (int i = 0; i < K; ++i){ std::cout << ng[i] << (i < K-1 ? ", " : "};\n");}
        std::cout << "std::vector<Index> ng_ineq = {";
        for (int i = 0; i < K; ++i){ std::cout << ng_ineq[i] << (i < K-1 ? ", " : "};\n");}

        dims.emplace(ProblemDims{K, nu, nx, ng, ng_ineq});
        info.emplace(ProblemInfo(dims.value()));
        jacobian.emplace(Jacobian<T>(dims.value()));
        full_matrix_jacobian =
            MatRealAllocated(info->number_of_eq_constraints, info->number_of_primal_variables);
        hessian.emplace(Hessian<T>(dims.value()));
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
        solver.emplace(AugSystemSolver<T>(info.value()));
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
                jacobian.value().BAbt[k].block(nu + nx, nx_next, 0, 0) =
                    ::test::random_matrix(nu + nx, nx_next);
                full_matrix_jacobian.value().block(nx_next, nu + nx, offs_eq_dyn, offs_ux) =
                transpose(jacobian.value().BAbt[k].block(nu + nx, nx_next, 0, 0));

                // if T is implicitOcpType:
                if constexpr (std::is_same_v<T, ImplicitOcpType>){
                    jacobian.value().Jt[k].block(nx_next, nx_next, 0, 0) =
                    // ::test::random_matrix(r[k+1], r[k+1]);
                    ::test::random_degenerate_matrix(nx_next, r[k+1]);
                    full_matrix_jacobian.value().block(nx_next, nx_next, offs_eq_dyn, offs_x_next) = 
                        transpose(jacobian.value().Jt[k]);
                }
            }
            jacobian.value().Gg_eqt[k].block(nu + nx, info.value().dims.number_of_eq_constraints[k], 0, 0) =
                ::test::random_matrix(nu + nx, info.value().dims.number_of_eq_constraints[k]);
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

            if (k < info.value().dims.K - 1 && std::is_same_v<T, ImplicitOcpType>){
                Index nx_next = info.value().dims.number_of_states[k + 1];
                Index offs_x_next = info.value().offsets_primal_x[k + 1];
                Index offs_ux = info.value().offsets_primal_u[k];

                if constexpr (std::is_same_v<T, ImplicitOcpType>){
                    if (no_second_order_effects){
                        hessian.value().FuFx[k].block(nx + nu, nx_next, 0, 0) =
                            ::test::empty_matrix(nx + nu, nx_next);
                    } else {
                        hessian.value().FuFx[k].block(nx + nu, nx_next, 0, 0) =
                            ::test::random_matrix(nx + nu, nx_next);
                    }
                    full_matrix_hessian.value().block(nx_next, nu + nx, offs_x_next, offs_ux) = 
                        transpose(hessian.value().FuFx[k]);
                    full_matrix_hessian.value().block(nu + nx, nx_next, offs_ux, offs_x_next) =
                        hessian.value().FuFx[k];
                }
            }
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
            D_x.value()(i) = 1.0 * (i + 0.1);
        }
        // fill the mult vector with random values
        for (Index i = 0; i < info.value().number_of_eq_constraints; ++i)
        {
            rhs_g.value()(i) = 1.0 * i;
        }

        for (Index i = 0; i < info.value().number_of_g_eq_path; ++i)
        {
            D_eq.value()(i) = 10.0 * (i + 1);
        }
        for (Index i = 0; i < info.value().number_of_slack_variables; ++i)
        {
            D_s.value()(i) =  1.0 + 0*10.0 * (i + 0.1);
        }
    }

    void SetUp()
    {
        // Individual tests choose a deterministic seed and construct the
        // problem they actually solve.
    };
};


template <typename T>
void CheckSolution(const ProblemInfo &info,
                   const Jacobian<T> &jacobian,
                   const Hessian<T> &hessian,
                   const VecRealView &D_x,
                   const VecRealView &D_s,
                   const VecRealView &rhs_x,
                   const VecRealView &rhs_g,
                   const VecRealView &x,
                   const VecRealView &mult,
                   const VecRealView *D_eq = nullptr)
{
    VecRealAllocated jac_x(info.number_of_eq_constraints);
    jacobian.apply_on_right(info, x, 0.0, jac_x, jac_x);
    VecRealAllocated rhs_gg(info.number_of_eq_constraints);
    rhs_gg = rhs_g + jac_x;
    VecRealAllocated regularized_dual(info.number_of_slack_variables);
    regularized_dual = 0.0;
    if (info.number_of_slack_variables > 0)
    {
        regularized_dual =
            D_s * mult.block(
                info.number_of_slack_variables,
                info.offset_g_eq_slack);
        rhs_gg.block(
            info.number_of_slack_variables,
            info.offset_g_eq_slack) =
            rhs_gg.block(
                info.number_of_slack_variables,
                info.offset_g_eq_slack) - regularized_dual;
    }
    VecRealAllocated regularized_path_dual(info.number_of_g_eq_path);
    regularized_path_dual = 0.0;
    if (D_eq != nullptr && info.number_of_g_eq_path > 0)
    {
        regularized_path_dual =
            (*D_eq) * mult.block(
                info.number_of_g_eq_path,
                info.offset_g_eq_path);
        rhs_gg.block(
            info.number_of_g_eq_path,
            info.offset_g_eq_path) =
            rhs_gg.block(
                info.number_of_g_eq_path,
                info.offset_g_eq_path) - regularized_path_dual;
    }

    VecRealAllocated hess_x(info.number_of_primal_variables);
    VecRealAllocated jac_t_mult(info.number_of_primal_variables);
    VecRealAllocated grad(info.number_of_primal_variables);
    hessian.apply_on_right(info, x, 0.0, hess_x, hess_x);
    jacobian.transpose_apply_on_right(
        info, mult, 0.0, jac_t_mult, jac_t_mult);
    grad = hess_x + D_x * x + jac_t_mult + rhs_x;

    double max_rhs_gg = 0.;
    double constraint_scale = 1.;
    for (Index i = 0; i < info.number_of_eq_constraints; ++i){
        max_rhs_gg = std::max(max_rhs_gg, std::abs(rhs_gg(i)));
        constraint_scale = std::max(
            constraint_scale,
            std::max(std::abs(rhs_g(i)), std::abs(jac_x(i))));
    }
    for (Index i = 0; i < info.number_of_slack_variables; ++i)
        constraint_scale = std::max(
            constraint_scale, std::abs(regularized_dual(i)));
    for (Index i = 0; i < info.number_of_g_eq_path; ++i)
        constraint_scale = std::max(
            constraint_scale, std::abs(regularized_path_dual(i)));

    double max_grad = 0.;
    double gradient_scale = 1.;
    for (Index i = 0; i < info.number_of_primal_variables; ++i){
        max_grad = std::max(max_grad, std::abs(grad(i)));
        gradient_scale = std::max(
            gradient_scale,
            std::max(
                std::max(std::abs(hess_x(i)), std::abs(D_x(i) * x(i))),
                std::max(std::abs(jac_t_mult(i)), std::abs(rhs_x(i)))));
    }
    EXPECT_LE(max_rhs_gg / constraint_scale, 1e-6);
    EXPECT_LE(max_grad / gradient_scale, 1e-6);
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


TEST_F(GeneralImplicitAugSystemSolverTest, TestSolve)
{
    // IMPLICIT OCP VERSION //
    Index ret = solver.solve(info, jacobian, hessian, D_x, D_s, rhs_x, rhs_g, x, mult);
    EXPECT_EQ(ret, LinsolReturnFlag::SUCCESS);

    // print the full KKT matrix and rhs
    bool print_full_kkt = true;
    if (print_full_kkt){ 
        PrintFullKKT(info, full_kkt_matrix, rhs_x, rhs_g, D_x, D_s, x, mult);
    }
    
    // Solution checking
    CheckSolution(info, jacobian, hessian, D_x, D_s, rhs_x, rhs_g, x, mult);

    // print eq_mult offsets
    std::cout << "eq_mult offsets: ";
    for (Index k = 0; k < info.dims.K; ++k){
        std::cout << info.offsets_g_eq_path[k] << " (" << info.dims.number_of_eq_constraints[k] << ") - ";
    }
    std::cout << std::endl;
    std::cout << "dynamics constraints offsets:";
    for (Index k = 0; k < info.dims.K; ++k){
        std::cout << info.offsets_g_eq_dyn[k] << " (" << info.dims.number_of_states[k+1] << ") - ";
    }
    std::cout << std::endl;
}

template <typename ProblemType>
void ExerciseRandomSuccessCase(
    RandomAugSystemSolverTest<ProblemType> &fixture,
    const unsigned seed,
    const bool stabilized = false)
{
    std::cout << "unsigned seed = " << seed << ";" << std::endl;
    srand(seed);

    // Local dimension inequalities are necessary but not sufficient for a
    // randomly assembled global Jacobian to have full row rank. Likewise,
    // arbitrary inter-stage Hessian blocks need not yield a positive reduced
    // Hessian. Those are valid solver classifications, not failed solves.
    // Resample until this success-path test obtains a well-posed instance.
    constexpr int max_attempts = 32;
    for (int test_counter = 0; test_counter < max_attempts; ++test_counter){
        std::cout << "\n" << std::endl;
        std::cout << "==============================" << std::endl;
        std::cout << "Test iteration: " << test_counter << "  ("
                  << (std::is_same_v<ProblemType, ImplicitOcpType>
                      ? "Implicit" : "Normal") << ")" << std::endl;
        std::cout << "==============================" << std::endl;
        fixture.Randomize();
        if (stabilized)
            for (Index i = 0; i < fixture.D_eq->m(); ++i)
                fixture.D_eq.value()(i) = 0.5 + 0.1 * (i + 1);
        Index ret = stabilized
            ? fixture.solver.value().solve(
                fixture.info.value(), fixture.jacobian.value(),
                fixture.hessian.value(), fixture.D_x.value(),
                fixture.D_eq.value(), fixture.D_s.value(),
                fixture.rhs_x.value(), fixture.rhs_g.value(),
                fixture.x.value(), fixture.mult.value())
            : fixture.solver.value().solve(
                fixture.info.value(), fixture.jacobian.value(),
                fixture.hessian.value(), fixture.D_x.value(),
                fixture.D_s.value(), fixture.rhs_x.value(),
                fixture.rhs_g.value(), fixture.x.value(),
                fixture.mult.value());

        if (ret == LinsolReturnFlag::NOFULL_RANK
            || ret == LinsolReturnFlag::INDEFINITE)
            continue;
        ASSERT_EQ(ret, LinsolReturnFlag::SUCCESS);
    
        // Solution checking
        CheckSolution(fixture.info.value(), fixture.jacobian.value(),
            fixture.hessian.value(), fixture.D_x.value(), fixture.D_s.value(),
            fixture.rhs_x.value(), fixture.rhs_g.value(), fixture.x.value(),
            fixture.mult.value(),
            stabilized ? &fixture.D_eq.value() : nullptr);

        // Reuse the retained factorization with a genuinely different right
        // hand side.  Exact inter-stage curvature also enters the multiplier
        // back-substitution, so checking only the factor-and-solve path would
        // miss stale FuFx/GuGx response data.
        VecRealAllocated repeated_rhs_x(
            fixture.info->number_of_primal_variables);
        VecRealAllocated repeated_rhs_g(
            fixture.info->number_of_eq_constraints);
        for (Index i = 0; i < repeated_rhs_x.m(); ++i)
            repeated_rhs_x(i) = 0.125 * (i + 1) - 0.75;
        for (Index i = 0; i < repeated_rhs_g.m(); ++i)
            repeated_rhs_g(i) = -0.2 * (i + 1) + 0.5;
        fixture.x.value() = 0.0;
        fixture.mult.value() = 0.0;
        ret = stabilized
            ? fixture.solver->solve_rhs(
                fixture.info.value(), fixture.jacobian.value(),
                fixture.hessian.value(), fixture.D_eq.value(),
                fixture.D_s.value(), repeated_rhs_x, repeated_rhs_g,
                fixture.x.value(), fixture.mult.value())
            : fixture.solver->solve_rhs(
                fixture.info.value(), fixture.jacobian.value(),
                fixture.hessian.value(), fixture.D_s.value(), repeated_rhs_x,
                repeated_rhs_g, fixture.x.value(), fixture.mult.value());
        ASSERT_EQ(ret, LinsolReturnFlag::SUCCESS);
        CheckSolution(
            fixture.info.value(), fixture.jacobian.value(),
            fixture.hessian.value(), fixture.D_x.value(),
            fixture.D_s.value(), repeated_rhs_x, repeated_rhs_g,
            fixture.x.value(), fixture.mult.value(),
            stabilized ? &fixture.D_eq.value() : nullptr);
        return;
    }
    FAIL() << "Could not generate a full-rank, positive-reduced-Hessian "
              "random OCP in " << max_attempts << " attempts (seed "
           << seed << ").";
}

using SolverTypes = ::testing::Types<OcpType, ImplicitOcpType>;
TYPED_TEST_SUITE(RandomAugSystemSolverTest, SolverTypes);
TYPED_TEST(RandomAugSystemSolverTest, TestRandomFullRankSolve)
{
    this->full_rank = true;
    // Direct-collocation continuity and defect coupling is affine between
    // neighbouring stages, so the inter-stage exact-Hessian block is zero.
    // Nonzero blocks are tracked separately below as a readiness blocker.
    this->no_second_order_effects = true;
    const unsigned seed = std::is_same_v<TypeParam, ImplicitOcpType>
        ? 20260717U : 20260718U;
    ExerciseRandomSuccessCase(*this, seed);
}

class RankDeficientImplicitAugSystemSolverTest
    : public RandomAugSystemSolverTest<ImplicitOcpType>
{
};

class ImplicitInterstageHessianRegressionTest
    : public RandomAugSystemSolverTest<ImplicitOcpType>
{
};

TEST_F(RankDeficientImplicitAugSystemSolverTest,
       SolvesWithoutInterstageSecondOrderTerms)
{
    full_rank = false;
    no_second_order_effects = true;
    ExerciseRandomSuccessCase(*this, 20260720U);
}

TEST_F(RankDeficientImplicitAugSystemSolverTest,
       ExactInterstageHessianPreservesStationarity)
{
    full_rank = false;
    no_second_order_effects = false;
    for (const unsigned seed : {20260717U, 20260723U, 20260724U})
        ExerciseRandomSuccessCase(*this, seed);
}

TEST_F(ImplicitInterstageHessianRegressionTest,
       FullRankExactInterstageHessianPreservesStationarity)
{
    full_rank = true;
    no_second_order_effects = false;
    for (const unsigned seed : {20260717U, 20260725U, 20260726U})
        ExerciseRandomSuccessCase(*this, seed);
}

TEST_F(RankDeficientImplicitAugSystemSolverTest,
       StabilizedExactInterstageHessianPreservesStationarity)
{
    full_rank = false;
    no_second_order_effects = false;
    for (const unsigned seed : {20260721U, 20260727U, 20260728U})
        ExerciseRandomSuccessCase(*this, seed, true);
}

TEST_F(ImplicitInterstageHessianRegressionTest,
       StabilizedFullRankExactInterstageHessianPreservesStationarity)
{
    full_rank = true;
    no_second_order_effects = false;
    for (const unsigned seed : {20260722U, 20260729U, 20260730U})
        ExerciseRandomSuccessCase(*this, seed, true);
}
