//
// Copyright (c) Lander Vanroye, KU Leuven
//

/**
 * @file lu_factorization.hpp
 * @brief Defines the PermutationMatrix class for LU factorization operations.
 *
 * This file also implements various linear algebra operations required by the FATROP algorithm
 * that are not currently available in the BLASFEO library.
 */

#ifndef __fatrop_linear_algebra_lu_factorization_hpp__
#define __fatrop_linear_algebra_lu_factorization_hpp__

#include "fatrop/common/exception.hpp"
#include "fatrop/context/context.hpp"
#include "fwd.hpp"
#include <vector>
#include <ostream>

namespace fatrop
{
    /**
     * @class PermutationMatrix
     * @brief Represents a permutation matrix for use in LU factorization.
     *
     * This class provides methods to apply row and column permutations
     * to matrices and vectors, which is essential for LU factorization
     * algorithms to maintain numerical stability and efficiency.
     */
    class PermutationMatrix
    {
    public:
        /**
         * @brief Constructs a PermutationMatrix of the given size.
         * @param size The size of the permutation matrix (number of rows/columns).
         */
        PermutationMatrix(const Index size);

        /**
         * @brief Applies the permutation to the rows of the given matrix.
         * @param kmax The number of permutations to be performed.
         * @param mat Pointer to the matrix to be permuted.
         */
        void apply_on_rows(const Index kmax, MAT *mat);
        void apply_on_rows(const Index kmax, MAT *mat, const Index ai);
        void apply_on_rows(const Index kmax, MAT *mat, const Index ai, const Index m);
        void apply_inverse_on_rows(const Index kmax, MAT *mat, const Index ai);

        /**
         * @brief Applies the permutation to the columns of the given matrix.
         * @param kmax The number of permutations to be performed.
         * @param mat Pointer to the matrix to be permuted.
         */
        void apply_on_cols(const Index kmax, MAT *mat);
        // void apply_on_cols(const Index kmax, MAT *mat, const Index aj);
        // void apply_on_cols(const Index kmax, MAT *mat, const Index aj, const Index n);
        // void apply_on_cols(const Index kmax, MAT *mat, const Index aj, const Index n, const Index row_start);
        void apply_on_cols(const Index kmax, MAT *mat, const Index ai, const Index aj, const Index nb_rows);
        void apply_inverse_on_cols(const Index kmax, MAT *mat);
        void apply_inverse_on_cols(const Index kmax, MAT *mat, const Index aj);

        /**
         * @brief Applies the permutation to the given vector.
         * @param kmax The number of permutations to be performed.
         * @param vec Pointer to the vector to be permuted.
         * @param ai The starting index for applying the permutation.
         */
        void apply(const Index kmax, VEC *vec, const Index ai);

        /**
         * @brief Applies the inverse of the permutation to the given vector.
         * @param kmax The number of permutations to be performed.
         * @param vec Pointer to the vector to be permuted.
         * @param ai The starting index for applying the inverse permutation.
         */
        void apply_inverse(const Index kmax, VEC *vec, const Index ai);

        int &operator[](const Index i)
        {
            fatrop_dbg_assert(i >= 0 && i < size_);
            return permutation_vector_[i];
        }

        friend std::ostream& operator<<(std::ostream& os, const PermutationMatrix& p)
        {
            os << "PermutationMatrix of size " << p.size() << ": [";
            std::vector<int> v = p.get_permutation_vector();
            for (Index i = 0; i < p.size(); ++i)
            {
                os << v[i];
                if (i < p.size() - 1)
                    os << ", ";
            }
            os << "]";
            return os;
        }

        Index size() const{ return size_;}
        std::vector<int> get_permutation_vector() const { return permutation_vector_; }

    private:
        const Index size_;                    ///< The size of the permutation matrix.
        std::vector<int> permutation_vector_; ///< The internal representation of the permutation.
    };

    /**
     * @brief Computes LU factorization of a matrix stored in transposed form.
     *
     * Performs LU factorization on a matrix \( A \), where the input \( At \) is \( A^T \)
     * (transposed). Results (\( L \) and \( U \)) are also stored in transposed form. Indices and
     * dimensions refer to the original matrix \( A \).
     *
     * @param m        [in]  Rows of the original matrix \( A \).
     * @param n        [in]  Columns of the original matrix \( A \).
     * @param n_max    [in]  Maximum column storage capacity.
     * @param rank     [out] Effective rank of \( A \).
     * @param At       [in]  Transposed input matrix \( A^T \).
     * @param Pl_p     [out] Left permutation matrix \( P_L \).
     * @param Pr_p     [out] Right permutation matrix \( P_R \).
     * @param tol      [in]  Tolerance for rank determination (default \( 1 \times 10^{-5} \)).
     *
     * @note Ensure \( At \), \( Pl_p \), and \( Pr_p \) are allocated before use.
     */
    void fatrop_lu_fact_transposed(const Index m, const Index n, const Index n_max, Index &rank, MAT *At,
                            PermutationMatrix &Pl, PermutationMatrix &Pr, double tol = 1e-5,
                            Index nb_row_perm = -1, Index nb_col_perm = -1);
    void fatrop_lu_fact_transposed(const Index m, const Index n, const Index n_max, Index &rank, MAT *At,
                            const Index ai, const Index aj, 
                            PermutationMatrix &Pl, PermutationMatrix &Pr, double tol = 1e-5,
                            Index nb_row_perm = -1, Index nb_col_perm = -1);

    void fatrop_lu_fact_transposed_sparse(
        const Index m, const Index n, const Index n_max, Index& rank, MAT* At,
        PermutationMatrix& Pl, PermutationMatrix& Pr, const std::vector<long long int>& sp_colind,
        const std::vector<long long int>& sp_row, const int nnz, double tol = 1e-5,
        Index nb_row_perm = -1, Index nb_col_perm = -1);

    /**
     * @brief Performs an addition operation with a transposed matrix.
     *
     * @param m Number of rows in the result matrix.
     * @param n Number of columns in the result matrix.
     * @param alpha Scalar multiplier for the addition.
     * @param sA Source matrix A.
     * @param offs_ai Row offset for matrix A.
     * @param offs_aj Column offset for matrix A.
     * @param sB Destination matrix B.
     * @param offs_bi Row offset for matrix B.
     * @param offs_bj Column offset for matrix B.
     */
    void fatrop_gead_transposed(Index m, Index n, Scalar alpha, MAT *sA,
                                Index offs_ai, Index offs_aj, MAT *sB,
                                Index offs_bi, Index offs_bj);

    /**
     * @brief Solves a triangular system of linear equations (upper triangular, not transposed, unit
     * diagonal).
     *
     */
    void fatrop_trsv_unu(const Index m, const Index n, MAT *sA, const Index ai,
                          const Index aj, VEC *sx, const Index xi, VEC *sz,
                          const Index zi);

    /**
     * @brief Solves a triangular system of linear equations (upper triangular, transposed, unit
     * diagonal).
     *
     */
    void fatrop_trsv_utu(const Index m, MAT *sA, const Index ai, const Index aj,
                          VEC *sx, const Index xi, VEC *sz, const Index zi);

} // namespace fatrop

#endif // __fatrop_linear_algebra_lu_factorization_hpp__
