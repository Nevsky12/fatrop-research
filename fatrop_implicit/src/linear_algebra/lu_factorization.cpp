//
// Copyright (c) Lander Vanroye, KU Leuven
//

#include "fatrop/common/exception.hpp"
#include "fatrop/linear_algebra/linear_algebra.hpp"
#include <algorithm>
#include <utility>
#include <vector>

namespace fatrop
{
    PermutationMatrix::PermutationMatrix(const Index size) : size_(size), permutation_vector_(size)
    {
        for (Index i = 0; i < size; i++)
        {
            permutation_vector_[i] = i;
        }
    };
    void PermutationMatrix::apply_on_rows(const Index kmax, MAT *mat)
    {
        fatrop_dbg_assert(kmax <= size_);
        ROWPE(kmax, permutation_vector_.data(), mat);
    };
    void PermutationMatrix::apply_on_rows(const Index kmax, MAT *mat, const Index ai)
    {
        fatrop_dbg_assert(kmax <= size_);
        fatrop_dbg_assert(ai >= 0);
        for (Index i = 0; i < kmax; i++)
        {
            ROWSW(mat->n, mat, ai + i, 0, mat, ai + permutation_vector_[i], 0);
        }
    };
    void PermutationMatrix::apply_on_rows(const Index kmax, MAT *mat, const Index ai, const Index m)
    {
        fatrop_dbg_assert(kmax <= size_);
        fatrop_dbg_assert(ai >= 0);
        fatrop_dbg_assert(m >= 0);
        for (Index i = 0; i < kmax; i++)
        {
            ROWSW(m, mat, ai + i, 0, mat, ai + permutation_vector_[i], 0);
        }
    };
    void PermutationMatrix::apply_inverse_on_rows(const Index kmax, MAT *mat, const Index ai)
    {
        fatrop_dbg_assert(kmax <= size_);
        fatrop_dbg_assert(ai >= 0);
        for (Index i = 0; i < kmax; i++)
        {
            ROWSW(mat->n, mat, ai + permutation_vector_[i], 0, mat, ai + i, 0);
        }
    };
    void PermutationMatrix::apply_on_cols(const Index kmax, MAT *mat)
    {
        fatrop_dbg_assert(kmax <= size_);
        COLPE(kmax, permutation_vector_.data(), mat);
    };
    // void PermutationMatrix::apply_on_cols(const Index kmax, MAT *mat, const Index aj)
    // {
    //     fatrop_dbg_assert(kmax <= size_);
    //     fatrop_dbg_assert(aj >= 0);
    //     for (Index i = 0; i < kmax; i++)
    //     {
    //         COLSW(mat->n, mat, 0, aj + i, mat, 0, aj + permutation_vector_[i]);
    //     }
    // }
    // void PermutationMatrix::apply_on_cols(const Index kmax, MAT *mat, const Index aj, const Index n)
    // {
    //     fatrop_dbg_assert(kmax <= size_);
    //     fatrop_dbg_assert(aj >= 0);
    //     fatrop_dbg_assert(n >= 0);
    //     for (Index i = 0; i < kmax; i++)
    //     {
    //         COLSW(n, mat, 0, aj + i, mat, 0, aj + permutation_vector_[i]);
    //     }
    // }
    // void PermutationMatrix::apply_on_cols(const Index kmax, MAT *mat, const Index aj, const Index n, const Index row_start)
    // {
    //     fatrop_dbg_assert(kmax <= size_);
    //     fatrop_dbg_assert(aj >= 0);
    //     fatrop_dbg_assert(n >= 0);
    //     for (Index i = 0; i < kmax; i++)
    //     {
    //         COLSW(n, mat, row_start, aj + i, mat, row_start, aj + permutation_vector_[i]);
    //     }
    // }
    void PermutationMatrix::apply_on_cols(const Index kmax, MAT *mat, 
                                          const Index ai, const Index aj, const Index nb_rows)
    {
        fatrop_dbg_assert(kmax <= size_);
        fatrop_dbg_assert(aj >= 0);
        fatrop_dbg_assert(ai >= 0);
        fatrop_dbg_assert(nb_rows >= 0);
        for (Index i = 0; i < kmax; i++)
        {
            COLSW(nb_rows, mat, ai, aj + i, mat, ai, aj + permutation_vector_[i]);
        }
    }
    void PermutationMatrix::apply_inverse_on_cols(const Index kmax, MAT *mat)
    {
        fatrop_dbg_assert(kmax <= size_);
        COLPEI(kmax, permutation_vector_.data(), mat);
    };
    void PermutationMatrix::apply_inverse_on_cols(const Index kmax, MAT *mat, const Index aj)
    {
        fatrop_dbg_assert(kmax <= size_);
        fatrop_dbg_assert(aj >= 0);
        for (Index i = 0; i < kmax; i++)
        {
            COLSW(mat->m, mat, 0, aj + permutation_vector_[i], mat, 0, aj + i);
        }
    };
    void PermutationMatrix::apply(const Index kmax, VEC *vec, const Index ai)
    {
        fatrop_dbg_assert(kmax <= size_);
        fatrop_dbg_assert(ai >= 0);
        VECPE(kmax, permutation_vector_.data(), vec, ai);
    };
    void PermutationMatrix::apply_inverse(const Index kmax, VEC *vec, const Index ai)
    {
        fatrop_dbg_assert(kmax <= size_);
        fatrop_dbg_assert(ai >= 0);
        VECPEI(kmax, permutation_vector_.data(), vec, ai);
    };

    /**
     * \brief Finds the maximum element (by absolute value) in a BLASFEO matrix within a submatrix.
     *
     * This function searches for the maximum absolute value element in a submatrix of size (m, n)
     * of the BLASFEO matrix, starting from the position (ai, aj). It returns the indices of the
     * maximum element as a pair (row, column).
     */
    std::pair<Index, Index> max_el(const Index m, const Index n, const MAT *matr, const Index ai,
                                   const Index aj)
    {
        // Initialize result indices to the starting position
        std::pair<Index, Index> max_indices{ai, aj};

        // Set the initial maximum value to the absolute value at the starting position
        double max_value = abs(blasfeo_matel_wrap(matr, ai, aj));

        // Iterate over the submatrix to find the maximum element
        for (Index col = aj; col < n; ++col)
        {
            for (Index row = ai; row < m; ++row)
            {
                double current_value = abs(blasfeo_matel_wrap(matr, row, col));
                if (current_value > max_value)
                {
                    max_value = current_value;
                    max_indices.first = row;
                    max_indices.second = col;
                }
            }
        }
        return max_indices;
    }
    void fatrop_lu_fact_transposed(const Index m, const Index n, const Index n_max, Index &rank, MAT *At,
                            PermutationMatrix &Pl, PermutationMatrix &Pr, double tol, 
                            Index nb_row_perm, Index nb_col_perm)
    {
        if (nb_row_perm < 0){ nb_row_perm = n;}
        if (nb_col_perm < 0){ nb_col_perm = m;}
        // std::cout << "computing LU of matrix" << std::endl;
        // blasfeo_print_dmat(n, m, At, 0, 0);
        fatrop_dbg_assert(m >= 0 && "m must be non-negative");
        fatrop_dbg_assert(n >= 0 && "n must be non-negative");
        fatrop_dbg_assert(n_max >= 0 && "n_max must be non-negative");
        fatrop_dbg_assert(tol >= 0 && "tolerance must be non-negative");
        At->use_dA = 0;
        Index minmn = std::min(m, n_max);
        Index j = 0;
        for (Index i = 0; i < minmn; i++)
        {
            std::pair<Index, Index> max_curr = max_el(n_max, m, At, i, i);
            if (abs(blasfeo_matel_wrap(At, max_curr.first, max_curr.second)) < tol)
            {
                break;
            }
            // switch rows
            COLSW(nb_row_perm, At, 0, i, At, 0, max_curr.second);
            // save in permutation vector
            Pl[i] = max_curr.second;
            // switch cols
            ROWSW(nb_col_perm, At, i, 0, At, max_curr.first, 0);
            // save in permutation vector
            Pr[i] = max_curr.first;
            for (Index j = i + 1; j < m; j++)
            {
                double Lji = blasfeo_matel_wrap(At, i, j) / blasfeo_matel_wrap(At, i, i);
                blasfeo_matel_wrap(At, i, j) = Lji;
                blasfeo_gead_wrap(n - (i + 1), 1, -Lji, At, i + 1, i, At, i + 1, j);
            }
            j = i + 1;
        }
        rank = j;
    }
    void fatrop_lu_fact_transposed(const Index m, const Index n, const Index n_max, Index &rank, MAT *At,
                            const Index ai, const Index aj,
                            PermutationMatrix &Pl, PermutationMatrix &Pr, double tol,
                            Index nb_row_perm, Index nb_col_perm)
    {
        if (nb_row_perm < 0){ nb_row_perm = n;}
        if (nb_col_perm < 0){ nb_col_perm = m;}
        // std::cout << "computing LU of matrix" << std::endl;
        // blasfeo_print_dmat(n, m, At, ai, aj);
        fatrop_dbg_assert(m >= 0 && "m must be non-negative");
        fatrop_dbg_assert(n >= 0 && "n must be non-negative");
        fatrop_dbg_assert(n_max >= 0 && "n_max must be non-negative");
        fatrop_dbg_assert(tol >= 0 && "tolerance must be non-negative");
        At->use_dA = 0;
        Index minmn = std::min(m, n_max);
        Index j = 0;
        for (Index i = 0; i < minmn; i++)
        {
            std::pair<Index, Index> max_curr = max_el(n_max + aj, m + ai, At, ai+i, aj+i);
            if (abs(blasfeo_matel_wrap(At, max_curr.first, max_curr.second)) < tol)
            {
                break;
            }

            // switch rows
            COLSW(nb_row_perm, At, 0*ai, aj+i, At, 0*ai, max_curr.second);
            // save in permutation vector
            // Pl[i] = max_curr.second - aj;
            Pl[aj + i] = max_curr.second;
            // switch cols
            ROWSW(nb_col_perm, At, ai+i, 0*aj, At, max_curr.first, 0*aj);
            // save in permutation vector
            // Pr[i] = max_curr.first - ai;
            Pr[ai + i] = max_curr.first;
            for (Index j = i + 1; j < m; j++)
            {
                double Lji = blasfeo_matel_wrap(At, ai+i, aj+j) / blasfeo_matel_wrap(At, ai+i, aj+i);
                blasfeo_matel_wrap(At, ai+i, aj+j) = Lji;
                blasfeo_gead_wrap(n - (i + 1), 1, -Lji, At, ai+i + 1, aj+i, At, ai+i + 1, aj+j);
            }
            j = i + 1;
        }
        rank = j;
    }
    // B <= B + alpha*A^T (B is mxn)
    void fatrop_gead_transposed(Index m, Index n, Scalar alpha, MAT *sA, Index offs_ai,
                                Index offs_aj, MAT *sB, Index offs_bi, Index offs_bj)
    {
        fatrop_dbg_assert(m >= 0 && "m must be non-negative");
        fatrop_dbg_assert(n >= 0 && "n must be non-negative");
        fatrop_dbg_assert(offs_ai >= 0 && offs_aj >= 0 && "offsets must be non-negative");
        fatrop_dbg_assert(offs_bi >= 0 && offs_bj >= 0 && "offsets must be non-negative");
        for (Index bj = 0; bj < n; bj++)
        {
            for (Index bi = 0; bi < m; bi++)
            {
                blasfeo_matel_wrap(sB, offs_bi + bi, offs_bj + bj) +=
                    alpha * blasfeo_matel_wrap(sA, offs_ai + bj, offs_aj + bi);
            }
        }
    }
    void fatrop_trsv_unu(const Index m, const Index n, MAT *sA, const Index ai, const Index aj,
                          VEC *sx, const Index xi, VEC *sz, const Index zi)
    {
        fatrop_dbg_assert(m >= 0 && "m must be non-negative");
        fatrop_dbg_assert(n >= 0 && "n must be non-negative");
        fatrop_dbg_assert(ai >= 0 && aj >= 0 && "matrix indices must be non-negative");
        fatrop_dbg_assert(xi >= 0 && zi >= 0 && "vector indices must be non-negative");
        fatrop_dbg_assert(m <= n && "m must be less than or equal to n");
        for (Index i = m; i < n; i++)
        {
            blasfeo_vecel_wrap(sz, zi + i) = blasfeo_vecel_wrap(sx, xi + i);
        }
        for (Index i = m - 1; i >= 0; i--)
        {
            Scalar res = blasfeo_vecel_wrap(sx, xi + i);
            for (Index j = i + 1; j < n; j++)
            {
                res -= blasfeo_matel_wrap(sA, ai + i, aj + j) * blasfeo_vecel_wrap(sz, zi + j);
            }
            blasfeo_vecel_wrap(sz, zi + i) = res;
        }
    }

    void fatrop_trsv_utu(const Index m, MAT *sA, const Index ai, const Index aj, VEC *sx,
                          const Index xi, VEC *sz, const Index zi)
    {
        fatrop_dbg_assert(m >= 0 && "m must be non-negative");
        fatrop_dbg_assert(ai >= 0 && aj >= 0 && "matrix indices must be non-negative");
        fatrop_dbg_assert(xi >= 0 && zi >= 0 && "vector indices must be non-negative");
        for (Index i = 0; i < m; i++)
        {
            Scalar res = blasfeo_vecel_wrap(sx, xi + i);
            for (Index j = 0; j < i; j++)
            {
                res -= blasfeo_matel_wrap(sA, ai + j, aj + i) * blasfeo_vecel_wrap(sz, zi + j);
            }
            blasfeo_vecel_wrap(sz, zi + i) = res;
        }
    }

    // This is a helper function that returns a list of non-zero columns for a given
    // row
    std::vector<int> get_nnz_cols_for_row(const long long int* sp_colind, 
                                          const long long int* sp_row,
                                          const int nnz, const int row) {
        std::vector<int> res;
        for (int i = 0; i < nnz; ++i) {
            if (sp_row[i] == row) {
                res.push_back(sp_colind[i]);
            }
        }
        return res;
    }

    void fatrop_lu_fact_transposed_sparse(
        const Index m, const Index n, const Index n_max, Index& rank, MAT* At,
        PermutationMatrix& Pl, PermutationMatrix& Pr, const std::vector<long long int>& sp_col,
        const std::vector<long long int>& sp_row, const int nnz, double tol, Index nb_row_perm,
        Index nb_col_perm) {
    if (nb_row_perm < 0) {
        nb_row_perm = n;
    }
    if (nb_col_perm < 0) {
        nb_col_perm = m;
    }

    // LU factorization with full pivoting
    Index i = 0;
    for (; i < std::min(m, n_max); ++i) {
        // Find the maximum element in the remaining submatrix
        double max_el_val = 0.0;
        Index max_el_row = -1;
        Index max_el_col = -1;

        for (int k = 0; k < nnz; ++k) {
            int r = sp_row[k];
            int c = sp_col[k];
            if (r >= i && c >= i) {
                double el_val = std::abs(blasfeo_matel_wrap(At, c, r));
                if (el_val > max_el_val) {
                    max_el_val = el_val;
                    max_el_row = c;
                    max_el_col = r;
                }
            }
        }

        // Check for rank deficiency
        if (max_el_val < tol) {
            break;
        }

        // Perform row pivoting
        if (max_el_col != i && i < nb_row_perm) {
            // blasfeo_swap_cols_wrap(n, At, i, max_el_col);
            // Pl.swap(i, max_el_col);
            COLSW(nb_row_perm, At, 0, i, At, 0, max_el_col);
            Pl[i] = max_el_col;
        }

        // Perform column pivoting
        if (max_el_row != i && i < nb_col_perm) {
            // blasfeo_swap_rows_wrap(m, At, i, max_el_row);
            // Pr.swap(i, max_el_row);
            ROWSW(nb_col_perm, At, 0, i, At, 0, max_el_row);
            Pr[i] = max_el_row;
        }

        // Perform the LU update
        for (int k = 0; k < nnz; ++k) {
            if (sp_row[k] == i && sp_col[k] > i) {
                int j = sp_col[k];
                blasfeo_matel_wrap(At, i, j) /= blasfeo_matel_wrap(At, i, i);
                for (int l = 0; l < nnz; ++l) {
                    if (sp_row[l] > i && sp_col[l] == j) {
                        int row_to_update = sp_row[l];
                        blasfeo_matel_wrap(At, row_to_update, j) -=
                            blasfeo_matel_wrap(At, i, j) *
                            blasfeo_matel_wrap(At, row_to_update, i);
                    }
                }
            }
        }
    }
    rank = i;
}
}
