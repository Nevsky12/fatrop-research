//
// Copyright (C) 2024 Lander Vanroye, KU Leuven
//

#ifndef __fatrop_ocp_aug_system_solver_hpp__
#define __fatrop_ocp_aug_system_solver_hpp__
#include "fatrop/context/context.hpp"
#include "fatrop/nlp/aug_system_solver.hpp"
#include "fatrop/linear_algebra/lu_factorization.hpp"
#include "fatrop/linear_algebra/matrix.hpp"
#include "fatrop/ocp/fwd.hpp"
#include "fatrop/linear_algebra/linear_solver_return_flags.hpp"
#include "fatrop/common/options.hpp"
#include <vector>
#include <chrono>
#include <memory>

namespace fatrop
{
    /**
     * @class AugSystemSolver<OcpType>
     * @brief Solves a system of equations for optimal control problems with augmented system
     * structure.
     *
     * This solver handles systems of the following form:
     *
     * \f[
     * \begin{bmatrix}
     *     H + D_x & A_e^T &  A_d^T &  A_i^T \\
     *     A_e     & -D_e  &  0     &  0     \\
     *     A_d     & 0     &  0     &  0     \\
     *     A_i     & 0     &  0     &  -D_i
     * \end{bmatrix}
     * \begin{bmatrix}
     *     x \\
     *     \lambda_e \\
     *     \lambda_d \\
     *     \lambda_i
     * \end{bmatrix} =
     * -\begin{bmatrix}
     *     f \\
     *     g_e \\
     *     g_d \\
     *     g_i
     * \end{bmatrix}
     * \f]
     *
     * where:
     * - \f$ H \f$ is the Lagrangian Hessian matrix.
     * - \f$ A_e \f$ is the Jacobian matrix of equality constraints.
     * - \f$ A_d \f$ is the Jacobian matrix of dynamics constraints.
     * - \f$ A_i \f$ is the Jacobian matrix of inequality constraints.
     * - \f$ D_x \f$, \f$ D_e \f$, and \f$ D_i \f$ are diagonal regularization matrices for primal variables,
     *   equality constraints, and inequality constraints, respectively.
     * - \f$ x \f$ represents the primal variables.
     * - \f$ \lambda_e \f$, \f$ \lambda_d \f$, and \f$ \lambda_i \f$ are the Lagrange multipliers for equality,
     *   dynamics, and inequality constraints, respectively.
     * - \f$ f, g_e, g_d, g_i \f$ are the corresponding residual vectors.
     *
     * The solver uses various numerical techniques, including LU factorization and iterative refinement,
     * to efficiently solve this system while handling potential numerical issues.
     */
    template<>
    class AugSystemSolver<OcpType>
    {
    public:
        /**
         * @brief Constructs an AugSystemSolver<OcpType> object.
         * @param info Problem information for the optimal control problem.
         */
        AugSystemSolver(const ProblemInfo &info);

        /**
         * @brief Solves the augmented system without path equality constraint regularization.
         * @param info Problem information.
         * @param jacobian Jacobian of the constraints.
         * @param hessian Hessian of the Lagrangian.
         * @param D_x Diagonal regularization for primal variables.
         * @param D_s Diagonal regularization for slack variables.
         * @param f Gradient of the objective function.
         * @param g Constraint residuals.
         * @param x [out] Solution vector for primal variables.
         * @param eq_mult [out] Solution vector for equality constraint multipliers.
         * @return Status flag indicating the outcome of the solve operation.
         */
        virtual LinsolReturnFlag solve(const ProblemInfo &info, Jacobian<OcpType> &jacobian,
                                       Hessian<OcpType> &hessian, const VecRealView &D_x,
                                       const VecRealView &D_s, const VecRealView &f, const VecRealView &g,
                                       VecRealView &x, VecRealView &eq_mult);

        /**
         * @brief Solves the augmented system with path equality constraint regularization.
         * @param info Problem information.
         * @param jacobian Jacobian of the constraints.
         * @param hessian Hessian of the Lagrangian.
         * @param D_x Diagonal regularization for primal variables.
         * @param D_eq Diagonal regularization for equality constraints.
         * @param D_s Diagonal regularization for slack variables.
         * @param f Gradient of the objective function.
         * @param g Constraint residuals.
         * @param x [out] Solution vector for primal variables.
         * @param eq_mult [out] Solution vector for equality constraint multipliers.
         * @return Status flag indicating the outcome of the solve operation.
         */
        virtual LinsolReturnFlag solve(const ProblemInfo &info, Jacobian<OcpType> &jacobian,
                                       Hessian<OcpType> &hessian, const VecRealView &D_x,
                                       const VecRealView &D_eq, const VecRealView &D_s,
                                       const VecRealView &f, const VecRealView &g, VecRealView &x,
                                       VecRealView &eq_mult);

        /**
         * @brief Solves the system for a new right-hand side without path equality constraint regularization.
         * @param info Problem information.
         * @param jacobian Jacobian of the constraints.
         * @param hessian Hessian of the Lagrangian.
         * @param D_s Diagonal regularization for slack variables.
         * @param f Gradient of the objective function.
         * @param g Constraint residuals.
         * @param x [out] Solution vector for primal variables.
         * @param eq_mult [out] Solution vector for equality constraint multipliers.
         * @return Status flag indicating the outcome of the solve operation.
         */
        virtual LinsolReturnFlag solve_rhs(const ProblemInfo &info,
                                           const Jacobian<OcpType> &jacobian,
                                           const Hessian<OcpType> &hessian, const VecRealView &D_s,
                                           const VecRealView &f, const VecRealView &g, VecRealView &x,
                                           VecRealView &eq_mult);

        /**
         * @brief Solves the system for a new right-hand side with path equality constraint regularization.
         * @param info Problem information.
         * @param jacobian Jacobian of the constraints.
         * @param hessian Hessian of the Lagrangian.
         * @param D_eq Diagonal regularization for equality constraints.
         * @param D_s Diagonal regularization for slack variables.
         * @param f Gradient of the objective function.
         * @param g Constraint residuals.
         * @param x [out] Solution vector for primal variables.
         * @param eq_mult [out] Solution vector for equality constraint multipliers.
         * @return Status flag indicating the outcome of the solve operation.
         */
        virtual LinsolReturnFlag solve_rhs(const ProblemInfo &info,
                                           const Jacobian<OcpType> &jacobian,
                                           const Hessian<OcpType> &hessian, const VecRealView &D_eq,
                                           const VecRealView &D_s, const VecRealView &f,
                                           const VecRealView &g, VecRealView &x, VecRealView &eq_mult);

        /**
         * @brief Registers the solver options with the provided options registry.
         * @param registry The options registry to register with.
         */
        void register_options(OptionRegistry &registry);

        // Option setters
        void set_it_ref(const bool &value) { it_ref = value; }
        void set_perturbed_mode(const bool &value) { perturbed_mode = value; }
        void set_perturbed_mode_param(const double &value) { perturbed_mode_param = value; }
        void set_lu_fact_tol(const Scalar &value) { lu_fact_tol = value; }
        void set_diagnostic(const bool &value) { diagnostic = value; }
        void set_increased_accuracy(const bool &value) { increased_accuracy = value; }

        std::chrono::nanoseconds duration_lu_factorization = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_backward_recursion = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_initial_stage = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_forward_recursion = std::chrono::nanoseconds(0);
    private:
        // temporaries, pre-allocated during construction to avoid allocation during
        // optimization
        std::vector<MatRealAllocated> Ppt;
        std::vector<MatRealAllocated> Hh;
        std::vector<MatRealAllocated> AL;
        std::vector<MatRealAllocated> RSQrqt_tilde;
        std::vector<MatRealAllocated> Ggt_stripe;
        std::vector<MatRealAllocated> Ggt_tilde;
        std::vector<MatRealAllocated> GgLt;
        std::vector<MatRealAllocated> RSQrqt_hat;
        std::vector<MatRealAllocated> Llt;
        std::vector<MatRealAllocated> Llt_shift;
        std::vector<MatRealAllocated> GgIt_tilde;
        std::vector<MatRealAllocated> GgLIt;
        std::vector<MatRealAllocated> HhIt;
        std::vector<MatRealAllocated> PpIt_hat;
        std::vector<MatRealAllocated> LlIt;
        std::vector<MatRealAllocated> Ggt_ineq_temp;
        std::vector<VecRealAllocated> v_Ppt;
        std::vector<VecRealAllocated> v_Hh;
        std::vector<VecRealAllocated> v_AL;
        std::vector<VecRealAllocated> v_RSQrqt_tilde;
        std::vector<VecRealAllocated> v_Ggt_stripe;
        std::vector<VecRealAllocated> v_Ggt_tilde;
        std::vector<VecRealAllocated> v_GgLt;
        std::vector<VecRealAllocated> v_RSQrqt_hat;
        std::vector<VecRealAllocated> v_Llt;
        std::vector<VecRealAllocated> v_Llt_shift;
        std::vector<VecRealAllocated> v_GgIt_tilde;
        std::vector<VecRealAllocated> v_GgLIt;
        std::vector<VecRealAllocated> v_HhIt;
        std::vector<VecRealAllocated> v_PpIt_hat;
        std::vector<VecRealAllocated> v_LlIt;
        std::vector<VecRealAllocated> v_Ggt_ineq_temp;
        std::vector<VecRealAllocated> v_tmp;
        std::vector<PermutationMatrix> Pl;
        std::vector<PermutationMatrix> Pr;
        std::vector<PermutationMatrix> PlI;
        std::vector<PermutationMatrix> PrI;
        std::vector<Index> gamma;
        std::vector<Index> rho;
        Index rankI = 0;
        bool it_ref = true;
        bool perturbed_mode = false;
        double perturbed_mode_param = 1e-6;
        Scalar lu_fact_tol = 1e-5;
        bool diagnostic = false;
        bool increased_accuracy = true;
    };



    template<>
    class AugSystemSolver<AcceleratedOcpType>
    {
    public:
        /**
         * @brief Constructs an AugSystemSolver<OcpType> object.
         * @param info Problem information for the optimal control problem.
         */
        AugSystemSolver<AcceleratedOcpType>(const ProblemInfo &info);

        /**
         * @brief Solves the augmented system without path equality constraint regularization.
         * @param info Problem information.
         * @param jacobian Jacobian of the constraints.
         * @param hessian Hessian of the Lagrangian.
         * @param D_x Diagonal regularization for primal variables.
         * @param D_s Diagonal regularization for slack variables.
         * @param f Gradient of the objective function.
         * @param g Constraint residuals.
         * @param x [out] Solution vector for primal variables.
         * @param eq_mult [out] Solution vector for equality constraint multipliers.
         * @return Status flag indicating the outcome of the solve operation.
         */
        virtual LinsolReturnFlag solve(const ProblemInfo &info, Jacobian<AcceleratedOcpType> &jacobian,
                                       Hessian<AcceleratedOcpType> &hessian, const VecRealView &D_x,
                                       const VecRealView &D_s, const VecRealView &f, const VecRealView &g,
                                       VecRealView &x, VecRealView &eq_mult);

        /**
         * @brief Solves the augmented system with path equality constraint regularization.
         * @param info Problem information.
         * @param jacobian Jacobian of the constraints.
         * @param hessian Hessian of the Lagrangian.
         * @param D_x Diagonal regularization for primal variables.
         * @param D_eq Diagonal regularization for equality constraints.
         * @param D_s Diagonal regularization for slack variables.
         * @param f Gradient of the objective function.
         * @param g Constraint residuals.
         * @param x [out] Solution vector for primal variables.
         * @param eq_mult [out] Solution vector for equality constraint multipliers.
         * @return Status flag indicating the outcome of the solve operation.
         */
        virtual LinsolReturnFlag solve(const ProblemInfo &info, Jacobian<AcceleratedOcpType> &jacobian,
                                       Hessian<AcceleratedOcpType> &hessian, const VecRealView &D_x,
                                       const VecRealView &D_eq, const VecRealView &D_s,
                                       const VecRealView &f, const VecRealView &g, VecRealView &x,
                                       VecRealView &eq_mult);

        /**
         * @brief Solves the system for a new right-hand side without path equality constraint regularization.
         * @param info Problem information.
         * @param jacobian Jacobian of the constraints.
         * @param hessian Hessian of the Lagrangian.
         * @param D_s Diagonal regularization for slack variables.
         * @param f Gradient of the objective function.
         * @param g Constraint residuals.
         * @param x [out] Solution vector for primal variables.
         * @param eq_mult [out] Solution vector for equality constraint multipliers.
         * @return Status flag indicating the outcome of the solve operation.
         */
        virtual LinsolReturnFlag solve_rhs(const ProblemInfo &info,
                                           const Jacobian<AcceleratedOcpType> &jacobian,
                                           const Hessian<AcceleratedOcpType> &hessian, const VecRealView &D_s,
                                           const VecRealView &f, const VecRealView &g, VecRealView &x,
                                           VecRealView &eq_mult);

        /**
         * @brief Solves the system for a new right-hand side with path equality constraint regularization.
         * @param info Problem information.
         * @param jacobian Jacobian of the constraints.
         * @param hessian Hessian of the Lagrangian.
         * @param D_eq Diagonal regularization for equality constraints.
         * @param D_s Diagonal regularization for slack variables.
         * @param f Gradient of the objective function.
         * @param g Constraint residuals.
         * @param x [out] Solution vector for primal variables.
         * @param eq_mult [out] Solution vector for equality constraint multipliers.
         * @return Status flag indicating the outcome of the solve operation.
         */
        virtual LinsolReturnFlag solve_rhs(const ProblemInfo &info,
                                           const Jacobian<AcceleratedOcpType> &jacobian,
                                           const Hessian<AcceleratedOcpType> &hessian, const VecRealView &D_eq,
                                           const VecRealView &D_s, const VecRealView &f,
                                           const VecRealView &g, VecRealView &x, VecRealView &eq_mult);

        void TestPermutationFunctions(const ProblemInfo& info, int k=-1);

        /**
         * @brief Registers the solver options with the provided options registry.
         * @param registry The options registry to register with.
         */
        void register_options(OptionRegistry &registry);

        // Option setters
        void set_it_ref(const bool &value) { it_ref = value; }
        void set_perturbed_mode(const bool &value) { perturbed_mode = value; }
        void set_perturbed_mode_param(const double &value) { perturbed_mode_param = value; }
        void set_lu_fact_tol(const Scalar &value) { lu_fact_tol = value; }
        void set_diagnostic(const bool &value) { diagnostic = value; }
        void set_increased_accuracy(const bool &value) { increased_accuracy = value; }
        void set_nb_of_dynamics_constraints(const Index &value) { nb_of_dynamics_constraints = value; }
        void set_nb_of_zk_vars(const Index &value) { nb_of_zk_vars = value; }

        std::chrono::nanoseconds duration_lu_factorization = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_backward_recursion = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_initial_stage = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_forward_recursion = std::chrono::nanoseconds(0);
        std::vector<PermutationMatrix> Pl_rank;
    // private:
        void fatrop_lu_fact_blocked_transposed(const ProblemDims& dims, const Index k, MAT *At, bool avoid_additional_perms=false);
        bool verify_blocked_lu_new(const MatRealAllocated& LU, 
            const MatRealAllocated& A_original,
            PermutationMatrix& Pl1, PermutationMatrix& Pr1, int rank1,
            PermutationMatrix& Pl_rank,
            PermutationMatrix& Pl2, PermutationMatrix& Pr2, int rank2,
            int ng, int nu, int nx, int nc=0);

        void apply_Pl_on_cols(
            PermutationMatrix& Pl1, PermutationMatrix& Pl_rank, PermutationMatrix& Pl2, 
            const Index r1, const Index r2, const Index m, MAT* A, const Index row_start);
        void apply_Pl(
            PermutationMatrix& Pl1, PermutationMatrix& Pl_rank, PermutationMatrix& Pl2, 
            const Index r1, const Index r2, const Index m, VEC *vec, const Index ai);
        void apply_Pl_inverse(
            PermutationMatrix& Pl1, PermutationMatrix& Pl_rank, PermutationMatrix& Pl2, 
            const Index r1, const Index r2, const Index m, VEC *vec, const Index ai);
        void apply_Pr_on_rows(
            PermutationMatrix& Pr1, PermutationMatrix& Pr2, const Index r1, const Index r2, MAT* A);
        void apply_Pr_on_cols(
            PermutationMatrix& Pr1, PermutationMatrix& Pr2, const Index r1, const Index r2, MAT* A);
        void apply_Pr(
            PermutationMatrix& Pr1, PermutationMatrix& Pr2, const Index r1, const Index r2, VEC *vec, const Index ai);
        void apply_Pr_inverse(
            PermutationMatrix& Pr1, PermutationMatrix& Pr2, const Index r1, const Index r2, VEC *vec, const Index ai);

        // temporaries, pre-allocated during construction to avoid allocation during
        // optimization
        std::vector<MatRealAllocated> Ppt;
        std::vector<MatRealAllocated> Hh;
        std::vector<MatRealAllocated> AL;
        std::vector<MatRealAllocated> RSQrqt_tilde;
        std::vector<MatRealAllocated> Ggt_stripe;
        std::vector<MatRealAllocated> Ggt_tilde;
        std::vector<MatRealAllocated> GgLt;
        std::vector<MatRealAllocated> RSQrqt_hat;
        std::vector<MatRealAllocated> Llt;
        std::vector<MatRealAllocated> Llt_shift;
        std::vector<MatRealAllocated> GgIt_tilde;
        std::vector<MatRealAllocated> GgLIt;
        std::vector<MatRealAllocated> HhIt;
        std::vector<MatRealAllocated> PpIt_hat;
        std::vector<MatRealAllocated> LlIt;
        std::vector<MatRealAllocated> Ggt_ineq_temp;
        std::vector<VecRealAllocated> v_Ppt;
        std::vector<VecRealAllocated> v_Hh;
        std::vector<VecRealAllocated> v_AL;
        std::vector<VecRealAllocated> v_RSQrqt_tilde;
        std::vector<VecRealAllocated> v_Ggt_stripe;
        std::vector<VecRealAllocated> v_Ggt_tilde;
        std::vector<VecRealAllocated> v_GgLt;
        std::vector<VecRealAllocated> v_RSQrqt_hat;
        std::vector<VecRealAllocated> v_Llt;
        std::vector<VecRealAllocated> v_Llt_shift;
        std::vector<VecRealAllocated> v_GgIt_tilde;
        std::vector<VecRealAllocated> v_GgLIt;
        std::vector<VecRealAllocated> v_HhIt;
        std::vector<VecRealAllocated> v_PpIt_hat;
        std::vector<VecRealAllocated> v_LlIt;
        std::vector<VecRealAllocated> v_Ggt_ineq_temp;
        std::vector<VecRealAllocated> v_tmp;
        
        std::vector<PermutationMatrix> Pl1;
        std::vector<PermutationMatrix> Pl2;
        std::vector<PermutationMatrix> Pr1;
        std::vector<PermutationMatrix> Pr2;
        std::vector<MatRealAllocated> scratch;
        
        std::vector<PermutationMatrix> PlI;
        std::vector<PermutationMatrix> PrI;
        std::vector<Index> gamma;
        std::vector<Index> rho;
        std::vector<Index> rho1;
        std::vector<Index> rho2;
        Index rankI = 0;
        bool it_ref = true;
        bool perturbed_mode = false;
        double perturbed_mode_param = 1e-6;
        Scalar it_ref_acc = 1e-8;
        Scalar lu_fact_tol = 1e-5;
        bool diagnostic = false;
        bool increased_accuracy = true;
        Index nb_of_dynamics_constraints = -1; // default value -1, nx will be used in this case
        Index nb_of_zk_vars = -1; // default value -1, nx will be used in this case

        private:
    };


    class ModifiedAugSystemSolver
    {
    public:
        ModifiedAugSystemSolver(const ProblemInfo &info);

        /**
         * @brief Solves the augmented system without path equality constraint regularization.
         * @param info Problem information.
         * @param jacobian Jacobian of the constraints.
         * @param hessian Hessian of the Lagrangian.
         * @param D_x Diagonal regularization for primal variables.
         * @param D_s Diagonal regularization for slack variables.
         * @param f Gradient of the objective function.
         * @param g Constraint residuals.
         * @param x [out] Solution vector for primal variables.
         * @param eq_mult [out] Solution vector for equality constraint multipliers.
         * @return Status flag indicating the outcome of the solve operation.
         */
        virtual LinsolReturnFlag solve(const ProblemInfo &info, Jacobian<ImplicitOcpType> &jacobian,
                                       Hessian<ImplicitOcpType> &hessian, const VecRealView &D_x,
                                       const VecRealView &D_s, const VecRealView &f, const VecRealView &g,
                                       VecRealView &x, VecRealView &eq_mult);

        /**
         * @brief Solves the augmented system with path equality constraint regularization.
         * @param info Problem information.
         * @param jacobian Jacobian of the constraints.
         * @param hessian Hessian of the Lagrangian.
         * @param D_x Diagonal regularization for primal variables.
         * @param D_eq Diagonal regularization for equality constraints.
         * @param D_s Diagonal regularization for slack variables.
         * @param f Gradient of the objective function.
         * @param g Constraint residuals.
         * @param x [out] Solution vector for primal variables.
         * @param eq_mult [out] Solution vector for equality constraint multipliers.
         * @return Status flag indicating the outcome of the solve operation.
         */
        virtual LinsolReturnFlag solve(const ProblemInfo &info, Jacobian<ImplicitOcpType> &jacobian,
                                       Hessian<ImplicitOcpType> &hessian, const VecRealView &D_x,
                                       const VecRealView &D_eq, const VecRealView &D_s,
                                       const VecRealView &f, const VecRealView &g, VecRealView &x,
                                       VecRealView &eq_mult);

        /**
         * @brief Solves the system for a new right-hand side without path equality constraint regularization.
         * @param info Problem information.
         * @param jacobian Jacobian of the constraints.
         * @param hessian Hessian of the Lagrangian.
         * @param D_s Diagonal regularization for slack variables.
         * @param f Gradient of the objective function.
         * @param g Constraint residuals.
         * @param x [out] Solution vector for primal variables.
         * @param eq_mult [out] Solution vector for equality constraint multipliers.
         * @return Status flag indicating the outcome of the solve operation.
         */
        virtual LinsolReturnFlag solve_rhs(const ProblemInfo &info,
                                           const Jacobian<ImplicitOcpType> &jacobian,
                                           const Hessian<ImplicitOcpType> &hessian, const VecRealView &D_s,
                                           const VecRealView &f, const VecRealView &g, VecRealView &x,
                                           VecRealView &eq_mult);

        /**
         * @brief Solves the system for a new right-hand side with path equality constraint regularization.
         * @param info Problem information.
         * @param jacobian Jacobian of the constraints.
         * @param hessian Hessian of the Lagrangian.
         * @param D_eq Diagonal regularization for equality constraints.
         * @param D_s Diagonal regularization for slack variables.
         * @param f Gradient of the objective function.
         * @param g Constraint residuals.
         * @param x [out] Solution vector for primal variables.
         * @param eq_mult [out] Solution vector for equality constraint multipliers.
         * @return Status flag indicating the outcome of the solve operation.
         */
        virtual LinsolReturnFlag solve_rhs(const ProblemInfo &info,
                                           const Jacobian<ImplicitOcpType> &jacobian,
                                           const Hessian<ImplicitOcpType> &hessian, const VecRealView &D_eq,
                                           const VecRealView &D_s, const VecRealView &f,
                                           const VecRealView &g, VecRealView &x, VecRealView &eq_mult);

        void set_performance_mode(bool set);
        void set_factorization_file_name(const std::string &name){factorization_file_name = name;};

        /**
         * @brief Registers the solver options with the provided options registry.
         * @param registry The options registry to register with.
         */
        void register_options(OptionRegistry &registry);

        // Option setters
        void set_it_ref(const bool &value) { it_ref = value; }
        void set_perturbed_mode(const bool &value) { perturbed_mode = value; }
        void set_perturbed_mode_param(const double &value) { perturbed_mode_param = value; }
        void set_lu_fact_tol(const Scalar &value) { lu_fact_tol = value; }
        void set_diagnostic(const bool &value) { diagnostic = value; }
        void set_increased_accuracy(const bool &value) { increased_accuracy = value; }


        std::chrono::nanoseconds duration_lu_factorization = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_RSQrqt_copy = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_FuFx_addition = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_GuGx_addition = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_GuGx_hat_addition = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_ukb_tilde_addition = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_lambdatilde_addition = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_FuFx_addition_forward = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_inner_solve = std::chrono::nanoseconds(0);

        std::chrono::nanoseconds duration_backward_recursion = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_initial_stage = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_forward_recursion = std::chrono::nanoseconds(0);

    private:
        // temporaries, pre-allocated during construction to avoid allocation during
        // optimization
        std::vector<MatRealAllocated> FuFx_underbar;
        std::vector<MatRealAllocated> GuGx_tilde;
        std::vector<MatRealAllocated> GuGx_hat;
        std::vector<MatRealAllocated> RSQrqt_underbar;
        std::vector<MatRealAllocated> v_r_tilde;

        std::vector<MatRealAllocated> Ppt;
        std::vector<MatRealAllocated> Hh;
        std::vector<MatRealAllocated> AL;
        std::vector<MatRealAllocated> RSQrqt_tilde;
        std::vector<MatRealAllocated> Ggt_stripe;
        std::vector<MatRealAllocated> Ggt_tilde;
        std::vector<MatRealAllocated> GgLt;
        std::vector<MatRealAllocated> RSQrqt_hat;
        std::vector<MatRealAllocated> Llt;
        std::vector<MatRealAllocated> Llt_shift;
        std::vector<MatRealAllocated> GgIt_tilde;
        std::vector<MatRealAllocated> GgLIt;
        std::vector<MatRealAllocated> HhIt;
        std::vector<MatRealAllocated> PpIt_hat;
        std::vector<MatRealAllocated> LlIt;
        std::vector<MatRealAllocated> Ggt_ineq_temp;
        std::vector<VecRealAllocated> v_Ppt;
        std::vector<VecRealAllocated> v_Hh;
        std::vector<VecRealAllocated> v_AL;
        std::vector<VecRealAllocated> v_RSQrqt_tilde;
        std::vector<VecRealAllocated> v_Ggt_stripe;
        std::vector<VecRealAllocated> v_Ggt_tilde;
        std::vector<VecRealAllocated> v_GgLt;
        std::vector<VecRealAllocated> v_RSQrqt_hat;
        std::vector<VecRealAllocated> v_Llt;
        std::vector<VecRealAllocated> v_Llt_shift;
        std::vector<VecRealAllocated> v_GgIt_tilde;
        std::vector<VecRealAllocated> v_GgLIt;
        std::vector<VecRealAllocated> v_HhIt;
        std::vector<VecRealAllocated> v_PpIt_hat;
        std::vector<VecRealAllocated> v_LlIt;
        std::vector<VecRealAllocated> v_Ggt_ineq_temp;
        std::vector<VecRealAllocated> v_tmp;
        std::vector<PermutationMatrix> Pl;
        std::vector<PermutationMatrix> Pr;
        std::vector<PermutationMatrix> PlI;
        std::vector<PermutationMatrix> PrI;
        std::vector<Index> gamma;
        std::vector<Index> rho;
        Index rankI = 0;
        bool it_ref = true;
        bool perturbed_mode = false;
        double perturbed_mode_param = 1e-6;
        Scalar it_ref_acc = 1e-8;
        Scalar lu_fact_tol = 1e-5;
        bool diagnostic = false;
        bool increased_accuracy = true;

        bool print_debug_lines = false;
        bool print_initial_stage = false;
        bool write_factorization_file = false;
        std::string factorization_file_name = "factorization_info.py";

        // for debugging
        std::vector<Index> rank_k_values;
        std::vector<MatRealAllocated> LU;
        std::vector<Index> gamma_k_values;
        std::vector<MatRealAllocated> Ggt_eq;
        std::vector<MatRealAllocated> R_shur;
    };

    // define ImplicitOcpType augmented system solver
    template<>
    class AugSystemSolver<ImplicitOcpType> : public ModifiedAugSystemSolver
    {
    public:
        /**
         * @brief Constructs an AugSystemSolver<ImplicitOcpType> object.
         * @param info Problem information for the optimal control problem.
         */
        AugSystemSolver(const ProblemInfo &info);

        /**
         * @brief Solves the augmented system without path equality constraint regularization.
         * @param info Problem information.
         * @param jacobian Jacobian of the constraints.
         * @param hessian Hessian of the Lagrangian.
         * @param D_x Diagonal regularization for primal variables.
         * @param D_s Diagonal regularization for slack variables.
         * @param f Gradient of the objective function.
         * @param g Constraint residuals.
         * @param x [out] Solution vector for primal variables.
         * @param eq_mult [out] Solution vector for equality constraint multipliers.
         * @return Status flag indicating the outcome of the solve operation.
         */
        virtual LinsolReturnFlag solve(const ProblemInfo &info, Jacobian<ImplicitOcpType> &jacobian,
                                       Hessian<ImplicitOcpType> &hessian, const VecRealView &D_x,
                                       const VecRealView &D_s, const VecRealView &f, const VecRealView &g,
                                       VecRealView &x, VecRealView &eq_mult);

        /**
         * @brief Solves the augmented system with path equality constraint regularization.
         * @param info Problem information.
         * @param jacobian Jacobian of the constraints.
         * @param hessian Hessian of the Lagrangian.
         * @param D_x Diagonal regularization for primal variables.
         * @param D_eq Diagonal regularization for equality constraints.
         * @param D_s Diagonal regularization for slack variables.
         * @param f Gradient of the objective function.
         * @param g Constraint residuals.
         * @param x [out] Solution vector for primal variables.
         * @param eq_mult [out] Solution vector for equality constraint multipliers.
         * @return Status flag indicating the outcome of the solve operation.
         */
        virtual LinsolReturnFlag solve(const ProblemInfo &info, Jacobian<ImplicitOcpType> &jacobian,
                                       Hessian<ImplicitOcpType> &hessian, const VecRealView &D_x,
                                       const VecRealView &D_eq, const VecRealView &D_s,
                                       const VecRealView &f, const VecRealView &g, VecRealView &x,
                                       VecRealView &eq_mult);

        /**
         * @brief Solves the system for a new right-hand side without path equality constraint regularization.
         * @param info Problem information.
         * @param jacobian Jacobian of the constraints.
         * @param hessian Hessian of the Lagrangian.
         * @param D_s Diagonal regularization for slack variables.
         * @param f Gradient of the objective function.
         * @param g Constraint residuals.
         * @param x [out] Solution vector for primal variables.
         * @param eq_mult [out] Solution vector for equality constraint multipliers.
         * @return Status flag indicating the outcome of the solve operation.
         */
        virtual LinsolReturnFlag solve_rhs(const ProblemInfo &info,
                                           Jacobian<ImplicitOcpType> &jacobian,
                                           Hessian<ImplicitOcpType> &hessian, const VecRealView &D_s,
                                           const VecRealView &f, const VecRealView &g, VecRealView &x,
                                           VecRealView &eq_mult);

        /**
         * @brief Solves the system for a new right-hand side with path equality constraint regularization.
         * @param info Problem information.
         * @param jacobian Jacobian of the constraints.
         * @param hessian Hessian of the Lagrangian.
         * @param D_eq Diagonal regularization for equality constraints.
         * @param D_s Diagonal regularization for slack variables.
         * @param f Gradient of the objective function.
         * @param g Constraint residuals.
         * @param x [out] Solution vector for primal variables.
         * @param eq_mult [out] Solution vector for equality constraint multipliers.
         * @return Status flag indicating the outcome of the solve operation.
         */
        virtual LinsolReturnFlag solve_rhs(const ProblemInfo &info,
                                           Jacobian<ImplicitOcpType> &jacobian,
                                           Hessian<ImplicitOcpType> &hessian, const VecRealView &D_eq,
                                           const VecRealView &D_s, const VecRealView &f,
                                           const VecRealView &g, VecRealView &x, VecRealView &eq_mult);


        void set_performance_mode(bool set);
        void set_preprocessing_file_name(const std::string &name){preprocessing_file_name = name;};

        std::chrono::nanoseconds duration_preprocess = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_preprocess_jac = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_preprocess_hess = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_preprocess_regularization = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_preprocess_decomposition = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_printing_preprocessed = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_decomp_copies = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_decomp_decomp = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_decomp_scale1 = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_decomp_scale2 = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_decomp_permutation = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_decomp_store = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_preprocess_info = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_preprocess_modify_rhs = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_solve = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_postprocess = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_copying_rhs = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_post_rearrange_solution = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_post_scale_solution = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_post_reset_jacobian_pre = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_post_reset_hessian_pre = std::chrono::nanoseconds(0);
        std::chrono::nanoseconds duration_post_regularization = std::chrono::nanoseconds(0);
    private:
        ProblemInfo PreProcess(const ProblemInfo &info, 
                        Jacobian<ImplicitOcpType> &jacobian,
                        Hessian<ImplicitOcpType> &hessian,
                        VecRealView &f, VecRealView &g,
                        VecRealView* D_x, VecRealView* D_eq, VecRealView* D_s);

        void PostProcess(const ProblemInfo &info, 
                         const ProblemInfo &modified_info,
                         Jacobian<ImplicitOcpType> &jacobian,
                         Hessian<ImplicitOcpType> &hessian,
                         VecRealView &x, VecRealView &eq_mult,
                         const VecRealView* D_s, const VecRealView* D_eq,
                         const VecRealView &g);

        // take last (nx_next - rank) columns (or rows) and insert them after
        // the first nu_next columns (or rows) of A, while shifting all
        // remaining columns (or rows) to the end of A
        void TreatStatesAsInputs(Index nu_next, Index nx_next, Index rank, 
                                 MatRealAllocated& A, bool rows=false);

        bool print_debug = false;
        bool print_preprocessed_system = false;
        bool write_preprocessing_file = false;
        std::string preprocessing_file_name = "preprocess_info..py";
        bool verify_preprocessed_solution = false;
        bool print_final_solution = false;

        double lu_fact_tol = 1e-5;

        // memory allocations
        std::vector<Index> number_of_states;
        std::vector<Index> number_of_controls;
        std::vector<Index> number_of_eq_constraints;
        std::vector<Index> number_of_ineq_constraints;
        std::vector<VecRealAllocated> f_copy;
        std::vector<VecRealAllocated> g_copy;
        std::vector<VecRealAllocated> D_x_copy;
        std::vector<VecRealAllocated> D_s_copy;
        std::vector<VecRealAllocated> D_eq_copy;
        std::vector<VecRealAllocated> x_copy;
        std::vector<VecRealAllocated> eq_mult_copy;

        std::unique_ptr<MatRealAllocated> scratch = std::make_unique<MatRealAllocated>(0,0);
        std::vector<MatRealAllocated> JBAbt;
        std::vector<MatRealAllocated> JBAbt_modified;
        // std::vector<PermutationMatrix> Pr_extended;
    };

} // namespace fatrop

#endif //__fatrop_ocp_aug_system_solver_hpp__
