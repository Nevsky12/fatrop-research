#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <fstream>

#include "fatrop/linear_algebra/linear_algebra.hpp"
#include "fatrop/linear_algebra/lu_factorization.hpp"
#include "../random_matrix.hpp"

using namespace fatrop;

void apply_Pl_on_cols(PermutationMatrix& Pl1, PermutationMatrix& Pl_rank, PermutationMatrix& Pl2, 
                   const Index r1, const Index r2, const Index m, MAT* A){
    // m: nx + ng
    Pl1.apply_on_cols(r1, A);
    Pl_rank.apply_on_cols(m, A);
    Pl2.apply_on_cols(r2, A, 0, r1, A->m);
}

void apply_Pl_on_cols(PermutationMatrix& Pl1, PermutationMatrix& Pl_rank, PermutationMatrix& Pl2, 
                   const Index r1, const Index r2, const Index m, MAT* A, const Index row_start){
    Pl1.apply_on_cols(r1, A, row_start, 0, A->n-row_start);
    Pl_rank.apply_on_cols(m, A, row_start, 0, A->n-row_start);
    Pl2.apply_on_cols(r2, A, row_start, r1, A->n-row_start);
}

void apply_Pl(PermutationMatrix& Pl1, PermutationMatrix& Pl_rank, PermutationMatrix& Pl2, 
              const Index r1, const Index r2, const Index m, VEC *vec, const Index ai){
    Pl1.apply(r1, vec, ai);
    Pl_rank.apply(m, vec, ai);
    Pl2.apply(r2, vec, ai+r1);
}

void apply_Pl_inverse(PermutationMatrix& Pl1, PermutationMatrix& Pl_rank, PermutationMatrix& Pl2, 
                      const Index r1, const Index r2, const Index m, VEC *vec, const Index ai){
    Pl2.apply_inverse(r2, vec, ai+r1);
    Pl_rank.apply_inverse(m, vec, ai);
    Pl1.apply_inverse(r1, vec, ai);
}

void apply_Pr_on_rows(PermutationMatrix& Pr1, PermutationMatrix& Pr2, 
                      const Index r1, const Index r2, MAT* A){
    Pr1.apply_on_rows(r1, A);
    Pr2.apply_on_rows(r2, A, r1);
}

void apply_Pr_on_cols(PermutationMatrix& Pr1, PermutationMatrix& Pr2, 
                      const Index r1, const Index r2, MAT* A){
    Pr1.apply_on_cols(r1, A);
    Pr2.apply_on_cols(r2, A, 0, r1, A->m);
}

void apply_Pr(PermutationMatrix& Pr1, PermutationMatrix& Pr2, 
              const Index r1, const Index r2, VEC *vec, const Index ai){
    Pr1.apply(r1, vec, ai);
    Pr2.apply(r2, vec, ai+r1);
}

void apply_Pr_inverse(PermutationMatrix& Pr1, PermutationMatrix& Pr2, 
                      const Index r1, const Index r2, VEC *vec, const Index ai){
    Pr2.apply_inverse(r2, vec, ai+r1);
    Pr1.apply_inverse(r1, vec, ai);
}

void extract_L(const MatRealAllocated &LU, MatRealAllocated &L, int m, int n, int ai=0, int aj=0, int bi=0, int bj=0){
    for (int row = 0; row < m; row++){
        for (int col = 0; col < m; col++){
            if (row > col && col < n){
                L(bi+row,bj+col) = LU(ai+col,aj+row);
            } else if (row == col){
                L(bi+row,bj+col) = 1.0;
            } else {
                L(bi+row,bj+col) = 0.0;
            }
        }
    }
}

void extract_U(const MatRealAllocated &LU, MatRealAllocated &U, int m, int n, int ai=0, int aj=0, int bi=0, int bj=0){
    for (int row = 0; row < m; row++){
        for (int col = 0; col < n; col++){
            if (row <= col /*&& row < ng[i]+nx[i]*/){
                U(bi+row,bj+col) = LU(ai+col,aj+row);
            } else {
                U(bi+row,bj+col) = 0.0;
            }
        }
    }
}

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
        // gecp(m, n, LU, ai, aj, LU, 0, 0);
        gecp(n, m, A, aj, ai, A, 0, 0);
    }

    // extract L and U
    extract_L(LU, L, m, n, ai, aj);
    extract_U(LU, U, m, n, ai, aj);

    // compute L*U
    blasfeo_dgemm_nn(m, n, n, 1.0, &L.mat(), 0, 0, &U.mat(), 0, 0, 0.0, 
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
        std::cout << "Pl: " << Pl << std::endl;
        std::cout << "Pr: " << Pr << std::endl;
        std::cout << "A_copy:\n" << LU << std::endl;
        std::cout << "A_verification:\n" << A_verification_T << std::endl;
        std::cout << "L:\n" << L << std::endl;
        std::cout << "U:\n" << U << std::endl;
        return false;
    }
    return true;
}

bool verify_blocked_lu_new(const MatRealAllocated& LU, 
        const MatRealAllocated& A_original, MatRealAllocated& A_verification, 
        MatRealAllocated& A_verification_T, MatRealAllocated& L, MatRealAllocated& U, 
        PermutationMatrix& Pl1, PermutationMatrix& Pr1, int rank1,
        PermutationMatrix& Pl_rank,
        PermutationMatrix& Pl2, PermutationMatrix& Pr2, int rank2,
        int ng, int nu, int nx, int nc=0){
    MatRealAllocated A(A_original.m(), A_original.n());
    gecp(A_original.m(), A_original.n(), A_original, 0, 0, A, 0, 0);

    blasfeo_dgese(A_verification.m(), A_verification.n(), 0, &A_verification.mat(), 0, 0);
    blasfeo_dgese(A_verification_T.m(), A_verification_T.n(), 0, &A_verification_T.mat(), 0, 0);
    blasfeo_dgese(L.m(), L.n(), 0, &L.mat(), 0, 0);
    blasfeo_dgese(U.m(), U.n(), 0, &U.mat(), 0, 0);

    extract_L(LU, L, ng+nx+nc, ng+nx+nc, 0, 0);
    extract_U(LU, U, ng+nx+nc, nu+nx, 0, 0);

    // compute L*U
    blasfeo_dgemm_nn(ng+nx+nc, nu+nx, ng+nx+nc, 1.0, &L.mat(), 0, 0, &U.mat(), 0, 0, 0.0, 
                     &A_verification.mat(), 0, 0, &A_verification.mat(), 0, 0);

    // apply permutations
    apply_Pl_on_cols(Pl1, Pl_rank, Pl2, rank1, rank2, ng+nx+nc, &A.mat());
    apply_Pr_on_rows(Pr1, Pr2, rank1, rank2, &A.mat());

    // transpose A_verification
    blasfeo_dgetr(ng+nx+nc, nu+nx, &A_verification.mat(), 0, 0, &A_verification_T.mat(), 0, 0);

    // check that A_copy and A_verification are close
    double max_diff = 0.0;
    for (int row = 0; row < nu+nx; row++){
        for (int col = 0; col < ng+nx+nc; col++){
            double diff = std::abs(A(row,col) - A_verification_T(row,col));
            if (diff > max_diff){
                max_diff = diff;
            }
        }
    }

    if (max_diff > 1e-4){
        int m = ng + nx + nc;
        int n = nu + nx;
        std::cout << "\nBlocked LU factorization verification failed" << std::endl;
        std::cout << "A is an " << (ng+nx+nc) << "x" << (nu+nx) << " matrix" << std::endl;
        std::cout << "Max difference: " << max_diff << std::endl;
        std::cout << "A_original:\n" << A_original.block(n,m,0,0) << std::endl;
        std::cout << "A_copy:\n" << A.block(n,m,0,0) << std::endl;
        std::cout << "A_verification:\n" << A_verification_T.block(n,m,0,0) << std::endl;
        std::cout << "L:\n" << L.block(m,m,0,0) << std::endl;
        std::cout << "U:\n" << U.block(m,n,0,0) << std::endl;
        std::cout << "rank: " << rank1 << " + " << rank2 << std::endl;
        return false;
    }

    return true;
}

void fatrop_lu_fact_blocked_transposed(const Index m, const Index n, 
        const Index n1, const Index n_max, Index &r1, Index &r2, Index &r, MAT *At,
        PermutationMatrix &Pl1, PermutationMatrix &Pr1, PermutationMatrix &Pl_rank,
        PermutationMatrix &Pl2, PermutationMatrix &Pr2, double tol,
        MAT *scratch){
    // nx: n1
    // nu: n - n1
    // ng: m - n1

    // lu of top-left block
    fatrop_lu_fact_transposed(m-n1, n-n1, n-n1, r1, At, Pl1, Pr1, tol);
    for (int k = 0; k < n1; k++){Pl_rank[r1 + k] = m-n1 + k;}

    // permute rows of matrix
    Pl_rank.apply_on_cols(m, At);

    // scaling bottom-left
    blasfeo_dgecp(n-n1, n1, At, 0, r1, scratch, 0, 0);   // M2 to B
    Pr1.apply_on_rows(r1, scratch);

    // compute K3 and K4
    blasfeo_dtrsm_llnn(r1, n1, 1.0, At, 0, 0, scratch, 0, 0, At, 0, r1);

    // K4 was ng x nx - rho, dus in transpose nx-rho x ng (n1-r1) x m-n1
    // K4 is nu nx x nu-rho, dus in transpose nu-rho x nx (n-n1-r1) x n1
    blasfeo_dgemm_nn(n-n1-r1, n1, r1, -1.0, At, r1, 0, At, 0, r1, 0.0, 
                        At, r1, r1, At, r1, r1);
    if (r1 > 0){
        blasfeo_dgead(n-n1-r1, n1, 1.0, scratch, r1, 0, At, r1, r1);
    } else {
        blasfeo_dgecp(n-n1-r1, n1, scratch, r1, 0, At, r1, r1);
    }

    // second LU decomposition
    fatrop_lu_fact_transposed(n1, n-r1, n-r1, r2, At, r1, r1, Pl2, Pr2, tol);
    r = r1 + r2;

    Pl2.apply_on_cols(r2, At, 0, r1, r1); // permute K3
    Pr2.apply_on_rows(r2, At, r1, r1); // permute V2

    // Fix bottom part
    if (n > n_max){
        apply_Pl_on_cols(Pl1, Pl_rank, Pl2, r1, r2, n-n_max, At, n_max);
        blasfeo_dtrsm_runu(n-n_max, r, 1.0, At, 0, 0, At, n_max, 0, At, n_max, 0);

        for (int i = 0; i < r; i++){
            for (int j = r; j < m; j++){
                double Lji = blasfeo_matel_wrap(At, i, j);
                blasfeo_gead_wrap(n-n_max, 1, -Lji, At, n_max, i, At, n_max, j);
            }
        }
    }
}
void fatrop_lu_fact_blocked_with_carry_transposed(const Index m, const Index n, 
        const Index n1, const Index nc, const Index n_max, Index &r1, Index &r2, Index &r, MAT *At,
        PermutationMatrix &Pl1, PermutationMatrix &Pr1, PermutationMatrix &Pl_rank,
        PermutationMatrix &Pl2, PermutationMatrix &Pr2, double tol,
        MAT *scratch){
    // nx: n1
    // nu: n - n1
    // ng: m - n1

    int nu = n_max;
    int nx = n - 1 - nu;
    int ng = m - nc;
    int nx_next = n1;

    int nu_true = nu - nx_next;
    int ng_true = ng - nx_next;

    // lu of top-left block
    fatrop_lu_fact_transposed(ng_true, nu_true, nu_true, r1, At, Pl1, Pr1, tol);
    for (int k = 0; k < m-ng_true; k++){Pl_rank[r1 + k] = ng_true + k;}

    // permute rows of matrix
    Pl_rank.apply_on_cols(m, At, 0, 0, n_max);

    // scaling bottom-left
    blasfeo_dgecp(nu_true, nx_next, At, 0, r1, scratch, 0, 0);   // M2 to B
    Pr1.apply_on_rows(r1, scratch);

    // compute K3 and K4
    blasfeo_dtrsm_llnn(r1, nx_next, 1.0, At, 0, 0, scratch, 0, 0, At, 0, r1);

    // K4 was ng x nx - rho, dus in transpose nx-rho x ng (n1-r1) x m-n1
    // K4 is nu nx x nu-rho, dus in transpose nu-rho x nx (n-n1-r1) x n1
    blasfeo_dgemm_nn(n_max-nx_next-r1, nx_next, r1, -1.0, At, r1, 0, At, 0, r1, 0.0, 
                        At, r1, r1, At, r1, r1);
    if (r1 > 0){
        blasfeo_dgead(nu_true-r1, n1, 1.0, scratch, r1, 0, At, r1, r1);
    } else {
        blasfeo_dgecp(nu_true-r1, n1, scratch, r1, 0, At, r1, r1);
    }

    // second LU decomposition
    fatrop_lu_fact_transposed(m-ng_true, nu-r1, n-r1, r2, At, r1, r1, Pl2, Pr2, tol);
    r = r1 + r2;

    Pl2.apply_on_cols(r2, At, 0, r1, r1); // permute K3
    Pr2.apply_on_rows(r2, At, r1, r1); // permute V2

    // Fix bottom part
    if (n > n_max){
        apply_Pl_on_cols(Pl1, Pl_rank, Pl2, r1, r2, n-n_max, At, n_max);
        blasfeo_dtrsm_runu(n-n_max, r, 1.0, At, 0, 0, At, n_max, 0, At, n_max, 0);
        blasfeo_dgemm_nn(n-n_max, m-r, r, -1.0, At, n_max, 0, At, 0, r, 1.0, 
                         At, n_max, r, At, n_max, r);
    }
}

void write_np_matrix(const MatRealAllocated& M, int m, int n, std::string name){
    std::ostream& o = std::cout;

    o << name << " = np.array([\n";
    for (int i = 0; i < m; i++){
        o << "    [";
        for (int j = 0; j < n; j++){
            o << std::setprecision(12) << M(i,j);
            if (j < n-1){
                o << ", ";
            }
        }
        o << "]";
        if (i < m-1){
            o << ",\n";
        } else {
            o << "\n";
        }
    } 
    o << "])\n";
    o << std::flush;    
}

void test_fatrop_lu_fact_transposed_bottom_block(int N = 1000000){
    // Test fatrop_lu_fact_transposed bottom block
    // set random seed
    srand(time(0));
    int nb_successes = 0;
    int nb_degenerate_cases = 0;
    double last_progress_value_shown = 0;
    double progress_show_step = 1;
    for (int nb = 0; nb < N; nb++){
        double progress = (double)nb / N * 100.0;
        if (progress - last_progress_value_shown >= progress_show_step){
            last_progress_value_shown = progress;
            std::cout << "Testing fatrop_lu_fact_transposed bottom block... " << std::fixed << std::setprecision(2) << progress << "%\r" << std::flush;
        }

        // get random matrices
        int m = rand() % 10 + 1;
        int n_max = rand() % 10 + 1;
        int n = rand() % 15 + 1;
        n = std::max(n, n_max);
        int rank = rand() % (std::min(m, n_max)+1);
        MatRealAllocated At = ::test::random_degenerate_matrix(n, m, rank);
        MatRealAllocated At_copy(n, m);
        gecp(n, m, At, 0, 0, At_copy, 0, 0);

        /// compute normal LU ///
        PermutationMatrix Pl(m);
        PermutationMatrix Pr(n);
        int r;
        fatrop_lu_fact_transposed(m, n, n_max, r, &At.mat(), Pl, Pr, 1e-5);
        if (r < std::min(m, n_max)){ nb_degenerate_cases++;}
        /////////////////////////

        /// compute LU on top part and fix bottom part ///
        PermutationMatrix Pl2(m);
        PermutationMatrix Pr2(n);
        int r2;
        fatrop_lu_fact_transposed(m, n_max, n_max, r2, &At_copy.mat(), Pl2, Pr2, 1e-5);

        /*
        Pl2.apply_on_cols(r2, &At_copy.mat(), n_max, 0, n-n_max);
        blasfeo_dtrsm_runu(n-n_max, r2, 1.0, &At_copy.mat(), 0, 0, &At_copy.mat(), n_max, 0, &At_copy.mat(), n_max, 0);

        for (int i = 0; i < r2; i++){
            for (int j = r2; j < m; j++){
                double Lji = blasfeo_matel_wrap(&At_copy.mat(), i, j);
                blasfeo_gead_wrap(n-n_max, 1, -Lji, &At_copy.mat(), n_max, i, &At_copy.mat(), n_max, j);
            }
        }
        */

        // permute columns
        Pl2.apply_on_cols(r2, &At_copy.mat(), n_max, 0, n-n_max);
        // M1 <- M1 * L1^-T
        blasfeo_dtrsm_runu(n-n_max, r2, 1.0, &At_copy.mat(), 0, 0, &At_copy.mat(), n_max, 0, &At_copy.mat(), n_max, 0);

        // M2 <- M2 + M1 * L2
        blasfeo_dgemm_nn(n-n_max, m-r2, r2, -1.0, &At_copy.mat(), n_max, 0, &At_copy.mat(), 0, r2, 1.0, 
                        &At_copy.mat(), n_max, r2, &At_copy.mat(), n_max, r2);

        /////////////////////////

        bool success = true;
        for (int ai = 0; ai < n; ai++){
            for (int aj = 0; aj < m; aj++){
                double diff = std::abs(At(ai,aj) - At_copy(ai,aj));
                if (diff > 1e-4){
                    success = false;
                }
            }
        }
        if (success) { nb_successes++; }
        if (!success){
            // std::cout << "m: " << m << ", n_max: " << n_max << ", n: " << n << ", rank: " << r << std::endl;
            // std::cout << "expected:\n" << At.block(n-n_max, m, n_max, 0) << std::endl;
            // std::cout << "actual:\n" << At_copy.block(n-n_max, m, n_max, 0) << std::endl;
            // std::cout << "expected:\n" << At << std::endl;
            // std::cout << "actual:\n" << At_copy << std::endl;
            blasfeo_dgead(n, m, -1.0, &At.mat(), 0, 0, &At_copy.mat(), 0, 0);
            // std::cout << "difference:\n" << At_copy << std::endl;
        }    
    }
    std::cout << std::endl;

    std::cout << "success rate:    " << (double)nb_successes / N * 100.0 << "%" << std::endl;
    std::cout << "degenerate rate: " << (double)nb_degenerate_cases / N * 100.0 << "%" << N << std::endl;
}

int main(){
    // test_fatrop_lu_fact_transposed_bottom_block();
    // return 0;

    // setup random dimensions
    int nb_batches = 1;//5;
    int nb_runs = 100000;//1000000;
    bool verify = true;
    bool write_csv = false;
    bool add_carry_over = true;

    std::string file_name = "blocked_lu_timings_general.csv";
    std::ofstream csv_file;
    if (write_csv){
        csv_file.open(file_name);
        csv_file << "nu,nx,ng,rank,time_full,time_blocked\n";
    }

    for (int current_batch_number = 0; current_batch_number < nb_batches; current_batch_number++){

    std::vector<int> nu(nb_runs);
    std::vector<int> nx(nb_runs);
    std::vector<int> ng(nb_runs);
    std::vector<int> rank(nb_runs);
    std::vector<int> nc(nb_runs);
    int max_val = 20;
    for (int i = 0; i < nb_runs; ++i) {
        nu[i] = rand() % (max_val+1); // Random number of control inputs between 1 and 100
        nx[i] = rand() % (max_val+1); // Random number of states between 1 and 100
        ng[i] = rand() % (max_val+1); // Random number of constraints between 1 and 100
        // rank[i] = rand() % (nx[i]+1); // Random rank between 0 and nx[i]
        rank[i] = rand() % (std::min(ng[i], nu[i])+1); // Random rank between 0 and nx[i]
        nc[i] = rand() % (max_val+1); // Random carry-over between 0 and 20
        if (!add_carry_over){ nc[i] = 0;}
    }

    // allocate random matrices
    std::vector<MatRealAllocated> A_full;
    std::vector<PermutationMatrix> Pl_full;
    std::vector<PermutationMatrix> Pr_full;
    std::vector<MatRealAllocated> A_blocked;
    std::vector<PermutationMatrix> Pl_blocked1;
    std::vector<PermutationMatrix> Pr_blocked1;
    std::vector<PermutationMatrix> Pl_rank;
    std::vector<PermutationMatrix> Pl_blocked2;
    std::vector<PermutationMatrix> Pr_blocked2;
    std::vector<PermutationMatrix> Pl_blocked_total;
    std::vector<PermutationMatrix> Pr_blocked_total;
    MatRealAllocated A_copy(3*max_val, 3*max_val);
    MatRealAllocated A_verification(3*max_val, 3*max_val);
    MatRealAllocated A_verification_T(3*max_val, 3*max_val);
    MatRealAllocated L(3*max_val, 3*max_val);
    MatRealAllocated U(3*max_val, 3*max_val);
    MatRealAllocated B(3*max_val, 3*max_val);
    MatRealAllocated B_tilde(3*max_val, 3*max_val);

    A_full.reserve(nb_runs);
    Pl_full.reserve(nb_runs);
    Pr_full.reserve(nb_runs);
    A_blocked.reserve(nb_runs);
    Pl_blocked1.reserve(nb_runs);
    Pr_blocked1.reserve(nb_runs);
    Pl_rank.reserve(nb_runs);
    Pl_blocked2.reserve(nb_runs);
    Pr_blocked2.reserve(nb_runs);
    Pl_blocked_total.reserve(nb_runs);
    Pr_blocked_total.reserve(nb_runs);
    for (int k = 0; k < nb_runs; k++){
        double progress = (double)(k+1) / nb_runs * 100.0;
        std::cout << "allocating ... " << std::setw(9) << std::setprecision(3) << progress << "%\r" << std::flush;
        int m = ng[k] + nx[k] + nc[k];
        int n = nu[k] + nx[k];
        MatRealAllocated top_left_block = ::test::random_degenerate_matrix(nu[k], ng[k], rank[k]);
        A_full.emplace_back(n, m);
        A_blocked.emplace_back(n, m);

        // top-left (M1)
        // std::cout << __LINE__ << std::endl;
        A_full[k].block(nu[k], ng[k], 0, 0) = top_left_block;
        // std::cout << __LINE__ << std::endl;
        A_blocked[k].block(nu[k], ng[k], 0, 0) = top_left_block;
        // std::cout << __LINE__ << std::endl;

        // M2 and M3
        A_full[k].block(nu[k]+nx[k], nx[k], 0, ng[k]) = ::test::random_matrix(nu[k] + nx[k], nx[k]);
        // std::cout << __LINE__ << std::endl;
        A_blocked[k].block(nu[k]+nx[k], nx[k], 0, ng[k]) = A_full[k].block(nu[k]+nx[k], nx[k], 0, ng[k]);
        // std::cout << __LINE__ << std::endl;

        // M4
        A_full[k].block(nx[k], nc[k], nu[k], ng[k]+nx[k]) = ::test::random_matrix(nx[k], nc[k]);
        // std::cout << __LINE__ << std::endl;
        A_blocked[k].block(nx[k], nc[k], nu[k], ng[k]+nx[k]) = A_full[k].block(nx[k], nc[k], nu[k], ng[k]+nx[k]);
        // std::cout << __LINE__ << std::endl;

        // for (int i = 0; i < n; i++){
        //     for (int j = 0; j < m; j++){
        //         if (i < nu[k] && j < ng[k]){
        //             A_full[k](i,j) = top_left_block(i,j);
        //             A_blocked[k](i,j) = top_left_block(i,j);
        //         } else {
        //             if (j >= ng[k]){ // transposed
        //                 A_full[k](i,j) = rand() / double(RAND_MAX);
        //                 A_blocked[k](i,j) = A_full[k](i,j);
        //             } else {
        //                 A_blocked[k](i,j) = 0.0; // zero out the botton-left block (in transpose)
        //                 A_full[k](i,j) = 0.0;
        //             }
        //         }
        //     }
        // }
        Pl_full.emplace_back(m);
        Pr_full.emplace_back(n);
        Pl_blocked1.emplace_back(m);
        Pr_blocked1.emplace_back(n);
        Pl_rank.emplace_back(m);
        Pl_blocked2.emplace_back(m);
        Pr_blocked2.emplace_back(n);
        Pl_blocked_total.emplace_back(m);
        Pr_blocked_total.emplace_back(n);
    }
    std::cout << std::flush << std::endl;
    std::cout << "allocations done" << std::endl;

    std::vector<double> time_full(nb_runs);
    std::vector<double> time_blocked(nb_runs);

    // perform full lu timings
    int r;
    for (int i = 0; i < nb_runs; ++i) {
        double progress = (double)(i+1) / nb_runs * 100.0;
        std::cout << "running full LU ... " << std::setw(9) << std::setprecision(3) << progress << "%\r" << std::flush;

        // store current matrix for verification later
        blasfeo_dgecp(nu[i]+nx[i], ng[i]+nx[i]+nc[i], &A_full[i].mat(), 0, 0, &A_copy.mat(), 0, 0);

        // perform lu
        auto start = std::chrono::high_resolution_clock::now();
        fatrop_lu_fact_transposed(ng[i]+nx[i]+nc[i], nu[i]+nx[i], nu[i]+nx[i], r, &A_full[i].mat(), Pl_full[i], Pr_full[i], 1e-5);

        auto end = std::chrono::high_resolution_clock::now();
        time_full[i] = std::chrono::duration<double, std::micro>(end - start).count();

        // verify
        if (verify && !verify_lu(A_full[i], A_copy, A_verification, A_verification_T, L, U, Pl_full[i], Pr_full[i], r, ng[i]+nx[i]+nc[i], nu[i]+nx[i])){
            std::cout << "Verification failed for full LU at run " << i+1 << std::endl;
            return -1;
        }
    }
    std::cout << "\nfull LU done" << std::endl;

    // perform blocked lu timings
    int r1 = 0; 
    int r2 = 0;
    int r_tot;
    for (int i = 0; i < nb_runs; ++i) {
        double progress = (double)(i+1) / nb_runs * 100.0;
        std::cout << "running blocked LU ... " << std::setw(9) << std::setprecision(3) << progress << "%\r" << std::flush;
        
        blasfeo_dgese(A_copy.m(), A_copy.n(), 0, &A_copy.mat(), 0, 0);
        blasfeo_dgecp(nu[i]+nx[i], ng[i]+nx[i]+nc[i], &A_blocked[i].mat(), 0, 0, &A_copy.mat(), 0, 0);

        auto start = std::chrono::high_resolution_clock::now();
        // fatrop_lu_fact_blocked_transposed(ng[i]+nx[i], nu[i]+nx[i], nx[i], nu[i]+nx[i], r1, r2, r_tot, &A_blocked[i].mat(), 
        //     Pl_blocked1[i], Pr_blocked1[i], Pl_rank[i], Pl_blocked2[i], Pr_blocked2[i], 1e-5, 
        //     &B_tilde.mat());
        fatrop_lu_fact_blocked_with_carry_transposed(ng[i]+nx[i]+nc[i], 
            nu[i]+nx[i], nx[i], nc[i], nu[i]+nx[i], r1, r2, r_tot, &A_blocked[i].mat(), 
            Pl_blocked1[i], Pr_blocked1[i], Pl_rank[i], Pl_blocked2[i], Pr_blocked2[i], 1e-5, 
            &B_tilde.mat());
        auto end = std::chrono::high_resolution_clock::now();
        time_blocked[i] = std::chrono::duration<double, std::micro>(end - start).count();

        if (verify){
            bool check_block_full = verify_blocked_lu_new(A_blocked[i], A_copy, 
                                                    A_verification, A_verification_T, L, U, 
                                                    Pl_blocked1[i], Pr_blocked1[i], r1,
                                                    Pl_rank[i],  
                                                    Pl_blocked2[i], Pr_blocked2[i], r2, ng[i], nu[i], nx[i], nc[i]);
            if (!check_block_full){
                std::cout << "Verification failed for blocked LU (full) at run " << i+1 << std::endl;
                return -1;
            }
        }        
    }
    std::cout << "\nblocked LU done" << std::endl;

    if (write_csv){
        // write csv file
        for (int i = 0; i < nb_runs; ++i) {
            double progress = (double)(i+1) / nb_runs * 100.0;
            std::cout << "writing csv ... " << std::setw(9) << std::setprecision(3) << progress << "%\r" << std::flush;
            csv_file << nu[i] << "," << nx[i] << "," << ng[i] << "," << rank[i] << "," << time_full[i] << "," << time_blocked[i] << "\n";
        }
        std::cout << "\ncsv writing done" << std::endl;
    }

    std::cout << "average time for full LU:    " << std::accumulate(time_full.begin(), time_full.end(), 0.0) / nb_runs << " us" << std::endl;
    std::cout << "average time for blocked LU: " << std::accumulate(time_blocked.begin(), time_blocked.end(), 0.0) / nb_runs << " us" << std::endl;
    }
}