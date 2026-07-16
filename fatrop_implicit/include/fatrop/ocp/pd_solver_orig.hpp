//
// Copyright (C) 2024 Lander Vanroye, KU Leuven
//

#ifndef __fatrop_ocp_pd__solver_orig_hpp__
#define __fatrop_ocp_pd__solver_orig_hpp__
// Primal-Dual System (PD System)

#include "fatrop/linear_algebra/fwd.hpp"
#include "fatrop/ip_algorithm/pd_solver_orig.hpp"
#include "fatrop/ocp/fwd.hpp"
#include "fatrop/ocp/pd_system_orig.hpp"
#include "fatrop/ocp/type.hpp"
#include <memory>

namespace fatrop
{

    template <typename ProblemType>
    class PdSolverOrig : public LinearSolver<PdSolverOrig<ProblemType>, PdSystemType<ProblemType>>
    {
    public:
        PdSolverOrig(const ProblemInfo& info, const std::shared_ptr<AugSystemSolver<ProblemType>>& aug_system_solver);
        LinsolReturnFlag solve_once_impl(LinearSystem<PdSystemType<ProblemType>> &ls, VecRealView &x);
        void reduce(LinearSystem<PdSystemType<ProblemType>> &ls);
        void dereduce(LinearSystem<PdSystemType<ProblemType>> &ls, VecRealView &x);
        void solve_rhs_impl(LinearSystem<PdSystemType<ProblemType>> &ls, VecRealView &x);

    private:
        VecRealAllocated sigma_inverse_;
        VecRealAllocated ss_;
        VecRealAllocated g_ii_;
        VecRealAllocated D_ii_;
        VecRealAllocated gg_;
        VecRealAllocated x_aug_;
        VecRealAllocated mult_aug_;
        std::shared_ptr<AugSystemSolver<ProblemType>> aug_system_solver_;
    };

} // namespace fatrop

#endif //__fatrop_ocp_pd_solver_orig_hpp__
