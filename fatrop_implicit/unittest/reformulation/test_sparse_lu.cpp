#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <fstream>

#include <casadi/casadi.hpp>

#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/linear_algebra/lu_factorization.hpp"
#include "../random_matrix.hpp"

using namespace fatrop;

// This is the verification function from test_block_LU_scaling.cpp
bool verify_lu(const MatRealAllocated &LU, const MatRealAllocated &A_original, MatRealAllocated &A_verification, 
               MatRealAllocated &A_verification_T, 
               MatRealAllocated &L, MatRealAllocated &U, 
               PermutationMatrix &Pl, PermutationMatrix &Pr, int rank, 
               int m, int n, int ai=0, int aj=0){
    MatRealAllocated A(A_original.m(), A_original.n());
    gecp(A_original.m(), A_original.n(), A_original, 0, 0, A, 0, 0);

    blasfeo_dgese(A_verification.m(), A_verification.n(), 0, &A_verification.mat(), 0, 0);
    blasfeo_dgese(A_verification_T.m(), A_verification_T.n(), 0, &A_verification_T.mat(), 0, 0);
    blasfeo_dgese(L.m(), L.n(), 0, &L.mat(), 0, 0);
    blasfeo_dgese(U.m(), U.n(), 0, &U.mat(), 0, 0);

    if (ai != 0 || aj != 0){
        gecp(n, m, A, aj, ai, A, 0, 0);
    }

    // extract L and U
    for (int row = 0; row < m; row++){
        for (int col = 0; col < m; col++){
            if (row > col && col < n){
                L(row,col) = LU(col,row);
            } else if (row == col){
                L(row,col) = 1.0;
            } else {
                L(row,col) = 0.0;
            }
        }
    }
    for (int row = 0; row < m; row++){
        for (int col = 0; col < n; col++){
            if (row <= col){
                U(row,col) = LU(col,row);
            } else {
                U(row,col) = 0.0;
            }
        }
    }

    // compute L*U
    blasfeo_dgemm_nn(m, n, m, 1.0, &L.mat(), 0, 0, &U.mat(), 0, 0, 0.0, 
                     &A_verification.mat(), 0, 0, &A_verification.mat(), 0, 0);

    // apply permutations
    Pl.apply_on_cols(rank, &A.mat());
    Pr.apply_on_rows(rank, &A.mat());

    // transpose A_verification
    blasfeo_dgetr(m, n, &A_verification.mat(), 0, 0, &A_verification_T.mat(), 0, 0);  

    // check that A_copy and A_verification are close
    double max_diff = 0.0;
    for (int row = 0; row < n; row++){
        for (int col = 0; col < m; col++){
            double diff = std::abs(A(row,col) - A_verification_T(row,col));
            if (diff > max_diff){
                max_diff = diff;
            }
        }
    }
    if (max_diff > 1e-4){
        std::cout << "\nLU factorization verification failed" << std::endl;
        std::cout << "A is an " << (m) << "x" << (n) << " matrix" << std::endl;
        std::cout << "Max difference: " << max_diff << std::endl;
        return false;
    }
    return true;
}


int main() {
    srand(time(0));
    int nb_runs = 1;
    std::vector<double> time_dense, time_sparse;
    time_dense.reserve(nb_runs);
    time_sparse.reserve(nb_runs);

    std::cout << "Running sparse LU test..." << std::endl;

    for (int i = 0; i < nb_runs; ++i) {
        // 1. Generate random matrix with casadi
        int m = rand() % 50 + 10; // rows
        int n = rand() % 50 + 10; // cols

        casadi::SX x = casadi::SX::sym("x", n);
        
        // Create a random sparse expression
        casadi::SX fun_expr = casadi::SX::zeros(m, 1);
        for(int k=0; k<m; ++k) {
            int non_zeros = rand() % std::min(5, n) + 1;
            for(int l=0; l<non_zeros; ++l) {
                fun_expr(k) += x(rand() % n) * ((double)rand() / RAND_MAX);
            }
        }

        casadi::Function f("f", {x}, {fun_expr});
        casadi::Function jf("jf", {x}, {jacobian(fun_expr, x)});

        // Generate random input
        std::vector<double> x_in(n);
        for(int k=0; k<n; ++k) x_in[k] = (double)rand() / RAND_MAX;

        // Evaluate jacobian
        std::vector<casadi::DM> arg = {x_in};
        std::vector<casadi::DM> res = jf(arg);
        casadi::DM jac_dm = res[0];

        // Get sparsity pattern
        casadi::Sparsity sp = jac_dm.get_sparsity();
        std::vector<long long int> sp_row = sp.get_row();
        std::vector<long long int> sp_col = sp.get_col();
        const casadi_int* sp_colind = sp.colind();
        std::cout << "jac_dm: \n" << jac_dm << std::endl;
        std::cout << "row: \n" << sp_row << std::endl;
        std::cout << "col: \n" << sp_col << std::endl;
        std::cout << "colind: \n" << sp.colind() << std::endl;
        int nnz = sp.nnz();

        // Create fatrop matrix
        MatRealAllocated A(n, m);
        
        // Casadi DM is column-major, fatrop is column-major
        // jac_dm is m x n. A is n x m (transposed storage)
        std::vector<double> jac_dense = jac_dm.get_elements();
        for(int c=0; c<n; ++c) {
            for(int r=0; r<m; ++r) {
                A(c, r) = jac_dm(r,c).get_elements()[0];
            }
        }

        // Prepare for factorizations
        MatRealAllocated A_dense(n,m);
        gecp(n, m, A, 0, 0, A_dense, 0, 0);
        MatRealAllocated A_sparse(n,m);
        gecp(n, m, A, 0, 0, A_sparse, 0, 0);

        PermutationMatrix Pl_dense(m), Pr_dense(n);
        PermutationMatrix Pl_sparse(m), Pr_sparse(n);
        Index rank_dense, rank_sparse;

        // Time dense factorization
        auto start = std::chrono::high_resolution_clock::now();
        fatrop_lu_fact_transposed(m, n, n, rank_dense, &A_dense.mat(), Pl_dense, Pr_dense);
        auto end = std::chrono::high_resolution_clock::now();
        time_dense.push_back(std::chrono::duration<double, std::micro>(end - start).count());

        // Time sparse factorization
        start = std::chrono::high_resolution_clock::now();
        // fatrop_lu_fact_transposed_sparse(m, n, n, rank_sparse, &A_sparse.mat(), Pl_sparse, Pr_sparse, sp_colind, sp_row, nnz);
        end = std::chrono::high_resolution_clock::now();
        time_sparse.push_back(std::chrono::duration<double, std::micro>(end - start).count());

        // Verification
        MatRealAllocated A_orig_T(m, n);
        blasfeo_dgetr(n, m, &A.mat(), 0, 0, &A_orig_T.mat(), 0, 0);

        MatRealAllocated A_verif(m, n), A_verif_T(n, m), L(m, m), U(m, n);
        if (!verify_lu(A_sparse, A_orig_T, A_verif, A_verif_T, L, U, Pl_sparse, Pr_sparse, rank_sparse, m, n)) {
            std::cerr << "Verification failed for sparse LU at run " << i << std::endl;
            return 1;
        }
    }

    double total_dense_time = 0;
    for(double t : time_dense) total_dense_time += t;
    double total_sparse_time = 0;
    for(double t : time_sparse) total_sparse_time += t;

    std::cout << "\n--- Timings ---" << std::endl;
    std::cout << "Average time for dense LU: " << total_dense_time / nb_runs << " us" << std::endl;
    std::cout << "Average time for sparse LU: " << total_sparse_time / nb_runs << " us" << std::endl;

    return 0;
}