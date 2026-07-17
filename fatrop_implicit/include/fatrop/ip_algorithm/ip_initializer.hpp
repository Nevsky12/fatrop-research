//
// Copyright (C) 2024 Lander Vanroye, KU Leuven
//

#ifndef __fatrop_ip_algorithm_ip_initializer_hpp__
#define __fatrop_ip_algorithm_ip_initializer_hpp__
#include "fatrop/context/context.hpp"
#include "fatrop/ip_algorithm/fwd.hpp"
#include "fatrop/common/fwd.hpp"
#include "fatrop/linear_algebra/vector.hpp"
#include <memory>
namespace fatrop
{
    /**
     * @brief Base class for initializers in interior point algorithms.
     */
    class IpInitializerBase
    {
    public:
        /**
         * @brief Initialize the algorithm's starting point.
         */
        virtual void initialize() = 0;
        virtual void register_options(OptionRegistry& registry) = 0;

        /**
         * @brief Reset the initializer to its initial state.
         */
        virtual void reset() = 0;

    protected:
        virtual ~IpInitializerBase() = default;
    };

    /**
     * @brief Concrete implementation of initializer for a specific problem type.
     * 
     * @tparam ProblemType The type of optimization problem being solved.
     */
    template <typename ProblemType> class IpInitializer : public IpInitializerBase
    {
        typedef std::shared_ptr<IpEqMultInitializerBase> IpEqMultInitializerSp;
        typedef std::shared_ptr<IpData<ProblemType>> IpDataSp;

    public:
        /**
         * @brief Construct a new IpInitializer object.
         * 
         * @param ipdata Shared pointer to the interior point algorithm data.
         * @param eq_mult_initializer Shared pointer to the equality multiplier initializer.
         */
        IpInitializer(const IpDataSp ipdata, const IpEqMultInitializerSp &eq_mult_initializer);

        void initialize() override;
        void reset() override;

    private:
        /**
         * @brief Initialize the slack variables.
         */
        void initialize_slacks();

        /// Initialize all primal-dual quantities from IpData's owned warm-start snapshot.
        void initialize_warm_start();

        /// Project a supplied slack vector strictly inside the relaxed bounds.
        void project_slacks(const VecRealView &input, Scalar push, Scalar fraction);

        IpDataSp ipdata_;                      ///< Interior point algorithm data
        IpEqMultInitializerSp eq_mult_initializer_; ///< Equality multiplier initializer
        Scalar bound_push = 1e-2;              ///< Bound push parameter (kappa_1)
        Scalar bound_frac = 1e-2;              ///< Bound fraction parameter (kappa_2)
        Scalar warm_start_bound_push_ = 1e-8;
        Scalar warm_start_bound_frac_ = 1e-8;
        Scalar warm_start_mult_bound_push_ = 1e-8;
        VecRealAllocated primal_buff_;
        VecRealAllocated slack_buff_;

    public:
        // Setter methods for options
        void set_bound_push(const Scalar& value) { bound_push = value; }
        void set_bound_frac(const Scalar& value) { bound_frac = value; }
        void set_warm_start_bound_push(const Scalar &value)
        {
            warm_start_bound_push_ = value;
        }
        void set_warm_start_bound_frac(const Scalar &value)
        {
            warm_start_bound_frac_ = value;
        }
        void set_warm_start_mult_bound_push(const Scalar &value)
        {
            warm_start_mult_bound_push_ = value;
        }

        // Register options
        void register_options(OptionRegistry& registry);
    };

} // namespace fatrop

#endif // __fatrop_ip_algorithm_ip_initializer_hpp__
