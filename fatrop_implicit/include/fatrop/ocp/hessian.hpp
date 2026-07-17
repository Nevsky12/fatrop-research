//
// Copyright (C) 2024 Lander Vanroye, KU Leuven
//

#ifndef __fatrop_ocp_hessian_hpp__
#define __fatrop_ocp_hessian_hpp__

#include "fatrop/nlp/hessian.hpp"
#include "fatrop/linear_algebra/fwd.hpp"
#include "fatrop/ocp/fwd.hpp"
#include "fatrop/linear_algebra/matrix.hpp"
#include "fatrop/ocp/dims.hpp"

#include <vector>
#include <chrono>
#include <map>

/**
 * @file hessian.hpp
 * @brief Defines the specialized Hessian structure for Optimal Control Problems (OCPs).
 *
 * This file contains the specialization of the hessian structure template
 * for Optimal Control Problems (OCPs). It represents the constraint Hessian
 * of the Karush-Kuhn-Tucker (KKT) system.
 *
 */

namespace fatrop
{
    typedef ProblemInfo OcpInfo;
    /**
     * @brief Specialization of the Hessian structure for Optimal Control Problems.
     *
     * This specialization of the Hessian structure is designed specifically
     * for Optimal Control Problems (OCPs). It represents the Hessian
     * of the Karush-Kuhn-Tucker (KKT) system for OCPs, taking into account their
     * specific structure and requirements.
     *
     * The Hessian is structured to efficiently handle the block-sparse nature
     * of OCPs, which typically involve dynamics constraints, path constraints,
     * and terminal constraints over multiple time steps.
     */
    template <> struct Hessian<OcpType>
    {
        /**
         * @brief Construct a new Hessian object for an OCP.
         *
         * @param dims The dimensions of the OCP, used to allocate appropriate memory.
         */
        Hessian(const ProblemDims &dims);

        /**
         * @brief Constraint Hessian of the dynamics.
         *
         * The Hessian is represented by a transposed matrix with an additional row
         * for the right-hand side. This structure allows for efficient simultaneous
         * factorization and solve in the Riccati recursion.
         *
         * Matrix dimensions: (nu[k] + nx[k] + 1) x (nu[k] + nx[k] + 1)
         * Where:
         *   nu[k]: number of control inputs at time step k
         *   nx[k]: number of states at time step k
         *   nx[k+1]: number of states at time step k+1
         *   RSQrqt[:nu, :nu] is reserved for control-control Hessian blocks,
         *   RSQrqt[nu:nu+nx, nu:nu+nx] is reserved for state-state Hessian blocks,
         *   RSQrqt[:nu, nu:nu+nx] = RSQrqt[nu:nu+nx, :nu]^T is reserved for control-state "skew"
         *   RSQrqt[-1, :] is reserved for the right-hand side.
         */
        std::vector<MatRealAllocated> RSQrqt;
        /// H_xp with trajectory variables in rows and global parameters in columns.
        MatRealAllocated global_parameter_cross_hessian;
        /// H_pp for the one-copy global parameter vector.
        MatRealAllocated global_parameter_hessian;
        /// Parameter part of the embedded Hessian right-hand side.
        VecRealAllocated global_parameter_rhs;
        void apply_on_right(const OcpInfo& info, const VecRealView& x, Scalar alpha, const VecRealView& y, VecRealView& out) const;
        void get_rhs(const OcpInfo& info, VecRealView& out) const;
        void set_rhs(const OcpInfo& info, const VecRealView& in);
        void set_zero();
        // make printable
        friend std::ostream& operator<<(std::ostream& os, const Hessian<OcpType>& hess);
    };
    template <> struct Hessian<AcceleratedOcpType> : public Hessian<OcpType> {
        using Hessian<OcpType>::Hessian; // inherit constructor
    };

    /**
     * @brief Specialization of the Hessian structure for Implicit Optimal Control Problems.
     *
     * This specialization of the Hessian structure is designed specifically
     * for Implicit Optimal Control Problems (OCPs). It represents the Hessian
     * of the Karush-Kuhn-Tucker (KKT) system for Implicit OCPs, taking into account their
     * specific structure and requirements.
     *
     * The Hessian is structured to efficiently handle the block-sparse nature
     * of Implicit OCPs, which typically involve dynamics constraints, path constraints,
     * and terminal constraints over multiple time steps.
     */
    template <>
    struct Hessian<ImplicitOcpType> : public Hessian<OcpType>
    {
        Hessian() = delete;
        Hessian(const ProblemDims &dims)
            : Hessian<OcpType>(dims)
        {
            // store the original RSQrqt matrices (+ overwrite original ones)
            RSQrqt = {};
            RSQrqt.reserve(dims.K);
            RSQrqt_original.reserve(dims.K);
            for (int k = 0; k < dims.K; ++k)
            {
                RSQrqt.emplace_back(1*(dims.number_of_controls[k] + dims.number_of_states[k] + 1) + 8,
                                    1*(dims.number_of_controls[k] + dims.number_of_states[k]));
                RSQrqt_original.emplace_back(1*(dims.number_of_controls[k] + dims.number_of_states[k] + 1) + 8,
                                             1*(dims.number_of_controls[k] + dims.number_of_states[k]));
            }

            // store additional dynamics hessians
            FuFx.reserve(dims.K - 1);
            GuGx.reserve(dims.K - 1);
            FuFx_original.reserve(dims.K - 1);
            GuGx_original.reserve(dims.K - 1);
            for (int k = 0; k < dims.K - 1; ++k)
            {
                FuFx.emplace_back( dims.number_of_controls[k] + dims.number_of_states[k] + 8,
                                  dims.number_of_states[k + 1] + 8);
                FuFx_original.emplace_back(dims.number_of_controls[k] + dims.number_of_states[k] + 8,
                                           dims.number_of_states[k + 1] + 8);
                GuGx.emplace_back(dims.number_of_controls[k] + dims.number_of_states[k] + 8,
                                  dims.number_of_controls[k + 1] + dims.number_of_states[k + 1] + 8);
                GuGx_original.emplace_back(dims.number_of_controls[k] + dims.number_of_states[k] + 8,
                                           dims.number_of_controls[k + 1] + dims.number_of_states[k + 1] + 8);
            }
        }

        Hessian(const Hessian &other) = default;
        Hessian &operator=(const Hessian &other) = default;

        Hessian(Hessian<ImplicitOcpType> &&other)
            : Hessian<OcpType>(std::move(other)),
              FuFx(std::move(other.FuFx)),
              FuFx_original(std::move(other.FuFx_original)),
              GuGx(std::move(other.GuGx)),
              GuGx_original(std::move(other.GuGx_original)),
              RSQrqt_original(std::move(other.RSQrqt_original)),
              duration_copy_RSQrqt(other.duration_copy_RSQrqt),
              duration_modifying_RSQrqt(other.duration_modifying_RSQrqt) {
            if (RSQrqt[0].mat().m == 0 && RSQrqt[0].mat().n == 0){
                throw std::runtime_error("Hessian<ImplicitOcpType>)other): RSQrqt matrix has zero size.");
            }
        };
        Hessian &operator=(Hessian &&other){
            if (this != &other){
                Hessian<OcpType>::operator=(std::move(other));
                FuFx = std::move(other.FuFx);
                FuFx_original = std::move(other.FuFx_original);
                GuGx = std::move(other.GuGx);
                GuGx_original = std::move(other.GuGx_original);
                RSQrqt_original = std::move(other.RSQrqt_original);
                duration_copy_RSQrqt = other.duration_copy_RSQrqt;
                duration_modifying_RSQrqt = other.duration_modifying_RSQrqt;
            }
            if (RSQrqt[0].mat().m == 0 && RSQrqt[0].mat().n == 0){
                throw std::runtime_error("Hessian<ImplicitOcpType>::operator=: RSQrqt matrix has zero size.");
            }
            return *this;
        }

        void PreProcess(const ProblemInfo &info, Jacobian<ImplicitOcpType> &jacobian,
                        VecRealView &f, VecRealView &g);
        void ResetPreProcess(const ProblemInfo &info, const Jacobian<ImplicitOcpType> &jacobian);

        // Overloading (to account for changed structure)
        void apply_on_right(const OcpInfo& info, const VecRealView& x, Scalar alpha, const VecRealView& y, VecRealView& out) const;

        // dimensions nu[k] + nx[k] x nx[k+1]
        std::vector<MatRealAllocated> FuFx;
        std::vector<MatRealAllocated> GuGx; // contains preprocessed info
        std::vector<MatRealAllocated> RSQrqt_original;
        std::vector<MatRealAllocated> FuFx_original;
        std::vector<MatRealAllocated> GuGx_original;

        // printing
        friend std::ostream& operator<<(std::ostream& os, const Hessian<ImplicitOcpType>& hess)
        {
            os << "Hessian<ImplicitOcpType>:" << std::endl;
            os << "RSQrqt:" << std::endl;
            for (const auto &mat : hess.RSQrqt)
            {
                os << mat << std::endl;
            }
            os << "FuFxt:" << std::endl;
            for (const auto &mat : hess.FuFx)
            {
                os << mat << std::endl;
            }
            return os;
        }

        std::chrono::microseconds duration_copy_RSQrqt = std::chrono::microseconds(0);
        std::chrono::microseconds duration_modifying_RSQrqt = std::chrono::microseconds(0);

    private:

        bool print_debug = false;
    };
} // namespace fatrop

#endif //__fatrop_ocp_jacobian_hpp__
