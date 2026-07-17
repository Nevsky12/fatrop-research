//
// Copyright (C) 2024 Lander Vanroye, KU Leuven
//

#ifndef __fatrop_ocp_jacobian_hpp__
#define __fatrop_ocp_jacobian_hpp__

#include "fatrop/linear_algebra/lu_factorization.hpp"
#include "fatrop/nlp/jacobian.hpp"
#include "fatrop/linear_algebra/fwd.hpp"
#include "fatrop/ocp/fwd.hpp"
#include "fatrop/linear_algebra/matrix.hpp"
#include "fatrop/ocp/dims.hpp"
#include "fatrop/ocp/problem_info.hpp"
#include <vector>

/**
 * @file jacobian.hpp
 * @brief Defines the specialized Jacobian structure for Optimal Control Problems (OCPs).
 *
 * This file contains the specialization of the Jacobian structure template
 * for Optimal Control Problems (OCPs). It represents the constraint Jacobian
 * of the Karush-Kuhn-Tucker (KKT) system.
 *
 */

namespace fatrop
{
    typedef ProblemInfo OcpInfo;
    /**
     * @brief Specialization of the Jacobian structure for Optimal Control Problems.
     *
     * This specialization of the Jacobian structure is designed specifically
     * for Optimal Control Problems (OCPs). It represents the constraint Jacobian
     * of the Karush-Kuhn-Tucker (KKT) system for OCPs, taking into account their
     * specific structure and requirements.
     *
     * The Jacobian is structured to efficiently handle the block-sparse nature
     * of OCPs, which typically involve dynamics constraints, path constraints,
     * and terminal constraints over multiple time steps.
     */
    template <> struct Jacobian<OcpType>
    {
        /**
         * @brief Construct a new Jacobian object for an OCP.
         *
         * @param dims The dimensions of the OCP, used to allocate appropriate memory.
         */
        Jacobian(const ProblemDims &dims);

        /**
         * @brief Constraint Jacobian of the dynamics.
         *
         * The Jacobian is represented by a transposed matrix with an additional row
         * for the right-hand side. This structure allows for efficient simultaneous
         * factorization and solve in the Riccati recursion.
         *
         * Matrix dimensions: (nu[k] + nx[k] + 1) x nx[k+1]
         * Where:
         *   nu[k]: number of control inputs at time step k
         *   nx[k]: number of states at time step k
         *   nx[k+1]: number of states at time step k+1
         *  BAbt[:nu, :] is reserved for control Jacobian blocks, while BAbt[nu:nu+nx, :] is
         * reserved for state Jacobian blocks. BAbt[-1, :] is reserved for the right-hand side.
         */
        std::vector<MatRealAllocated> BAbt;

        /**
         * @brief Constraint Jacobian of path equality constraints. Similar to BAbt, it also has
         * an additional row for the right-hand side.
         *
         * Matrix dimensions: (nu[k] + nx[k]) x ng[k]
         * Where:
         *   nx[k]: number of states at time step k
         *   nu[k]: number of control inputs at time step k
         *   ng[k]: number of equality constraints at time step k
         *  Gg_eqt[:nu, :] is reserved for control Jacobian blocks, while Gg_eqt[nu:nu+nx, :] is
         * reserved for state Jacobian blocks. Gg_eqt[-1, :] is reserved for the right-hand side.
         */
        std::vector<MatRealAllocated> Gg_eqt;

        /**
         * @brief Constraint Jacobian of path inequality constraints. Similar to BAbt, it also
         * has an additional row for the right-hand side.
         *
         * Matrix dimensions: (nu[k] + nx[k] + 1) x ng_ineq[k]
         * Where:
         *   nx[k]: number of states at time step k
         *   nu[k]: number of control inputs at time step k
         *   ng_ineq[k]: number of inequality constraints at time step k
         * Gg_ineqt[:nu, :] is reserved for control Jacobian blocks, while Gg_ineqt[nu:nu+nx, :] is
         * reserved for state Jacobian blocks. Gg_ineqt[-1, :] is reserved for the right-hand side.
         */
        std::vector<MatRealAllocated> Gg_ineqt;

        /** Complete constraint Jacobian with respect to the one-copy global
         * parameter vector. Shape: number_of_eq_constraints x np. */
        MatRealAllocated global_parameter_jacobian;

        // out <- alpha*y + Jacobian * x
        void apply_on_right(const OcpInfo& info, const VecRealView &x, Scalar alpha, const VecRealView& y, VecRealView &out) const;
        void transpose_apply_on_right(const OcpInfo& info, const VecRealView &mult_eq, Scalar alpha, const VecRealView& y, VecRealView &out) const;
        void get_rhs(const OcpInfo& info, VecRealView &rhs) const;
        void set_rhs(const OcpInfo& info, const VecRealView &rhs);
        // make printable 
        friend std::ostream &operator<<(std::ostream &os, const Jacobian &jac);
    };
    template <> struct Jacobian<AcceleratedOcpType> : public Jacobian<OcpType> 
    {
        using Jacobian<OcpType>::Jacobian; // inherit constructor
    };

    template<>
    struct Jacobian<ImplicitOcpType> : public Jacobian<OcpType>
    {
        Jacobian(const ProblemDims &dims)
            : Jacobian<OcpType>(dims)
        {
            // store the original BAbt matrices (+ overwrite original ones)
            BAbt = {};
            BAbt.reserve(dims.K - 1);
            BAbt_original.reserve(dims.K);
            for (int k = 0; k < dims.K - 1; ++k){
                BAbt.emplace_back(dims.number_of_controls[k] + dims.number_of_states[k] + 1 + 0*8,
                                  dims.number_of_states[k + 1]);
                BAbt_original.emplace_back(dims.number_of_controls[k] + dims.number_of_states[k] + 1 + 0*8,
                                  dims.number_of_states[k + 1]);
            }

            // store additional dynamics jacobian
            Jt.reserve(dims.K - 1);
            Jt_LU.reserve(dims.K - 1);
            Jt_inv.reserve(dims.K - 1);
            rho.reserve(dims.K - 1);
            J_ranks.reserve(dims.K - 1);
            nb_new_controls.reserve(dims.K - 1);
            Pl_pre.reserve(dims.K - 1);
            Pr_pre.reserve(dims.K - 1);
            U1U2t.reserve(dims.K - 1);
            U1t.reserve(dims.K - 1);
            for (int k = 0; k < dims.K - 1; ++k)
            {
                Jt.emplace_back(dims.number_of_states[k + 1], dims.number_of_states[k + 1]);
                Jt_LU.emplace_back(dims.number_of_states[k + 1], dims.number_of_states[k + 1]);
                Jt_inv.emplace_back(dims.number_of_states[k + 1], dims.number_of_states[k + 1]);
                Pl_pre.emplace_back(dims.number_of_states[k + 1]);
                Pr_pre.emplace_back(dims.number_of_states[k + 1]);
                U1U2t.emplace_back(dims.number_of_states[k + 1] + 0*8, dims.number_of_states[k + 1] + 0*8);
                U1t.emplace_back(dims.number_of_states[k + 1], dims.number_of_states[k + 1]);
            }
            J_ranks = std::vector<Index>(dims.K - 1, 0);

            // enlarge Gg_eqt (since states from the previous time-step could be added)
            Gg_eqt = {};
            Gg_eqt.reserve(dims.K);
            for (int k = 0; k < dims.K; ++k){
                Gg_eqt.emplace_back(dims.number_of_controls[k] + dims.number_of_states[k] + 1 + 0*8,
                                  dims.number_of_eq_constraints[k] + ((k < dims.K-1) ? dims.number_of_states[k + 1] : 0));
            }

            // store eq constraint jacobian
            Gg_eqt_original.reserve(dims.K);
            Gg_ineqt = {};
            Gg_ineqt.reserve(dims.K);
            Gg_ineqt_original.reserve(dims.K);
            for (int k = 0; k < dims.K; ++k){
                Gg_eqt_original.emplace_back(dims.number_of_controls[k] + dims.number_of_states[k] + 1 + 0*8,
                                  dims.number_of_eq_constraints[k] + ((k < dims.K-1) ? dims.number_of_states[k + 1] : 0));
                Gg_ineqt.emplace_back(dims.number_of_controls[k] + dims.number_of_states[k] + 1 + 0*8,
                                               dims.number_of_ineq_constraints[k]);
                Gg_ineqt_original.emplace_back(dims.number_of_controls[k] + dims.number_of_states[k] + 1 + 0*8,
                                               dims.number_of_ineq_constraints[k]);
            }
        }
        
        /**
         * @brief Preprocess the Jacobian structure, by computing BAbt as
         * J^-1 * BAbt_original
         *
         * This function is called to prepare the Jacobian structure for use.
         * It can be used to allocate memory, initialize matrices, or perform
         * any other necessary setup before the Jacobian is used in computations.
         */
        void PreProcess(const ProblemInfo &info, VecRealView &f, 
                        VecRealView &g);
        void ResetPreProcess(const ProblemInfo &info);

        void PrepareInverseOfJ(const ProblemInfo &info);

        // Overloading (to account for changed structure)
        void apply_on_right(const OcpInfo& info, const VecRealView &x, Scalar alpha, const VecRealView& y, VecRealView &out, bool ignore_Jt=false) const;
        void transpose_apply_on_right(const OcpInfo& info, const VecRealView &mult_eq, Scalar alpha, const VecRealView& y, VecRealView &out, bool ignore_Jt=false) const;

        /**
         * @brief Jacobian of the dynamics with respect to x[k+1].
         *
         * Matrix dimensions: nx[k+1] x nx[k+1]
         * Where:
         *   nx[k+1]: number of states at time step k+1
         */
        std::vector<MatRealAllocated> Jt;
        std::vector<PermutationMatrix> Pl_pre;
        std::vector<PermutationMatrix> Pr_pre;
        std::vector<MatRealAllocated> U1U2t; // -(U1^-1 * U2)^T
        std::vector<MatRealAllocated> U1t;    // U1^T
        std::vector<Index> J_ranks;
        std::vector<Index> nb_new_controls;
        std::vector<MatRealAllocated> Jt_LU;

        std::vector<MatRealAllocated> Jt_inv; // remove
        bool ASSUME_INVERSE_GIVEN = false;

        // printing
        friend std::ostream &operator<<(std::ostream &os, const Jacobian<ImplicitOcpType> &jac)
        {
            os << "Jacobian<ImplicitOcpType>:" << std::endl;
            os << "BAbt:" << std::endl;
            for (const auto &mat : jac.BAbt)
            {
                os << mat << std::endl;
            }
            os << "Gg_eqt:" << std::endl;
            for (const auto &mat : jac.Gg_eqt)
            {
                os << mat << std::endl;
            }
            os << "Gg_ineqt:" << std::endl;
            for (const auto &mat : jac.Gg_ineqt)
            {
                os << mat << std::endl;
            }
            return os;
        }
        double dgemm_time = 0;
        std::vector<int> rho;
        std::vector<MatRealAllocated> BAbt_original;
        std::vector<MatRealAllocated> Gg_eqt_original;
        std::vector<MatRealAllocated> Gg_ineqt_original;
    private:

        bool print_debug = false;
    };
} // namespace fatrop

#endif //__fatrop_ocp_jacobian_hpp__
