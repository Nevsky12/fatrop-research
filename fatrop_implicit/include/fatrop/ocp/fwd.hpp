//
// Copyright (c) 2024 Lander Vanroye, KU Leuven
//

#ifndef __fatrop_ocp_fwd_hpp__
#define __fatrop_ocp_fwd_hpp__

namespace fatrop
{
    class OcpType;
    class ImplicitOcpType;
    class AcceleratedOcpType;
    struct ProblemDims;
    struct ProblemInfo;
    template <typename T> struct Jacobian;
    template <> struct Jacobian<OcpType>;
    template <> struct Jacobian<ImplicitOcpType>;
    template <> struct Jacobian<AcceleratedOcpType>;
    template <typename T> struct Hessian;
    template <> struct Hessian<OcpType>;
    template <> struct Hessian<ImplicitOcpType>;
    template <> struct Hessian<AcceleratedOcpType>;
    template <typename T> struct PdSolverOrig;
    template <typename ProblemType> class AugSystemSolver;
    template <> class AugSystemSolver<OcpType>;
    template <> class AugSystemSolver<ImplicitOcpType>;
    template <> class AugSystemSolver<AcceleratedOcpType>;
    template <typename ProblemType> class PdSystemOrig;
    template <typename ProblemType> class PdSystemResto;
    template <typename T> struct PdSolverResto;
} // namespace fatrop

#endif // __fatrop_ocp_fwd_hpp__