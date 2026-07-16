/**
 *
 *
 * This interface is used to be compatible with the C interface of the fatrop v0 library. In
 * particular this interface is used by CasADi.
 *
 *
 *
 */

#include "fatrop/common/options.hpp"
#include "fatrop/common/printing.hpp"
#include "fatrop/context/context.hpp"
#include "fatrop/ip_algorithm/ip_alg_builder.hpp"
#include "fatrop/ip_algorithm/ip_algorithm.hpp"
#include "fatrop/nlp/nlp.hpp"
#include "fatrop/ocp/hessian.hpp"
#include "fatrop/ocp/jacobian.hpp"
#include "fatrop/ocp/problem_info.hpp"
#include "fatrop/ocp/type.hpp"
#include "fatrop/ip_algorithm/ip_data.hpp"
#include <algorithm>
#include <limits>
#include <memory>

#include "fatrop/ocp/OCPCInterfaceInternal.hpp"
namespace fatrop
{

    struct PendingOption
    {
        std::string name;
        enum class Type { Double, Int, Bool, String } type;
        union {
            double d;
            int i;
            bool b;
        };
        std::string s; // can't be in union
    };

    class OcpSolverDriverBase;

    struct FatropOcpCSolver
    {
        OcpSolverDriverBase *driver;
        
        // store driver args
        FatropOcpCInterface *ocp_interface;
        FatropOcpCWrite write;
        FatropOcpCFlush flush;

        std::string problem_type = "unknown";
        std::vector<PendingOption> pending_options; // options can only be added once ocp type is known, so we store them here until then


        void set_problem_type(const std::string &problem_type);
    };

#define FATROP_OCP_SOLVER_IMPLEMENTATION
    namespace // nameless namespace
    {
        struct FatropOcpCAuxiliary
        {
            static ProblemDims get_ocp_dims(const FatropOcpCInterface &ocp)
            {
                const Index K = ocp.get_horizon_length(ocp.user_data);
                std::vector<Index> nu(K), nx(K), ng(K), ng_ineq(K);
                for (Index k = 0; k < K; k++)
                {
                    nu[k] = ocp.get_nu(k, ocp.user_data);
                    nx[k] = ocp.get_nx(k, ocp.user_data);
                    ng[k] = ocp.get_ng(k, ocp.user_data);
                    ng_ineq[k] = ocp.get_ng_ineq(k, ocp.user_data);
                }
                return ProblemDims(K, nu, nx, ng, ng_ineq);
            }
            static NlpDims get_nlp_dims(const ProblemDims &ocp_dims)
            {
                Index number_of_variables = 0;
                Index number_of_eq_constraints = 0;
                Index number_of_ineq_constraints = 0;
                for (Index k = 0; k < ocp_dims.K; k++)
                {
                    number_of_variables +=
                        ocp_dims.number_of_controls[k] + ocp_dims.number_of_states[k];
                    number_of_eq_constraints += ocp_dims.number_of_eq_constraints[k] +
                                                ocp_dims.number_of_ineq_constraints[k];
                    if (k != ocp_dims.K - 1)
                    {
                        number_of_eq_constraints += ocp_dims.number_of_states[k + 1];
                    }
                    number_of_ineq_constraints += ocp_dims.number_of_ineq_constraints[k];
                }
                return NlpDims(number_of_variables, number_of_eq_constraints,
                               number_of_ineq_constraints);
            }
        };

    }

    template <typename ProblemType>
    FatropOcpCMapping<ProblemType>::FatropOcpCMapping(FatropOcpCInterface *ocp)
        : ocp(ocp), ocp_dims_(FatropOcpCAuxiliary::get_ocp_dims(*ocp)),
          nlp_dims_(FatropOcpCAuxiliary::get_nlp_dims(ocp_dims_)), K_(ocp_dims_.K),
          matrix_buffer_{std::vector<MAT *>(K_), std::vector<MAT *>(K_), std::vector<MAT *>(K_)}
    {
        // check if no parameters are used by the ocp, because this is not supported anymore
        if (ocp->get_n_global_params && ocp->get_n_global_params(ocp->user_data) > 0)
        {
            throw std::runtime_error("Parameters are not supported anymore in the C interface");
        }
        for (Index k = 0; k < ocp_dims_.K; k++)
        {
            if (ocp->get_n_stage_params && ocp->get_n_stage_params(k, ocp->user_data) > 0)
            {
                throw std::runtime_error("Parameters are not supported anymore in the C interface");
            }
        }
    }

    template <typename ProblemType>
    const NlpDims &FatropOcpCMapping<ProblemType>::nlp_dims() const { return nlp_dims_; };
    template <typename ProblemType>
    const ProblemDims &FatropOcpCMapping<ProblemType>::problem_dims() const { return ocp_dims_; };
    template <typename ProblemType>
    Index FatropOcpCMapping<ProblemType>::eval_lag_hess(const ProblemInfo &info,
                                           const Scalar objective_scale,
                                           const VecRealView &primal_x, const VecRealView &primal_s,
                                           const VecRealView &lam, Hessian<ProblemType> &hess)
    {
        // take the matrices from hess and put them in the buffer
        std::vector<MAT *> &RSQrqt_buff = matrix_buffer_[0];
        for (Index k = 0; k < K_; k++)
        {
            RSQrqt_buff[k] = &hess.RSQrqt[k].mat();
        }
        // get the double pointers for the vector views
        const Scalar *primal_x_ptr = primal_x.data();
        const Scalar *primal_s_ptr = primal_s.data();
        const Scalar *lam_ptr = lam.data();
        // call the C interface
        int ret = ocp->full_eval_lag_hess(objective_scale, primal_x_ptr, lam_ptr, nullptr, nullptr,
                                          RSQrqt_buff[0], &s, ocp->user_data);
        if (ret == 2)
            return 0;
        if (ret == 0)
        {
            for (Index k = 0; k < info.dims.K; k++)
            {
                const Scalar *inputs_k = primal_x_ptr + info.offsets_primal_u[k];
                const Scalar *states_k = primal_x_ptr + info.offsets_primal_x[k];
                const Scalar *lam_dyn_k =
                    (k != info.dims.K - 1) ? lam_ptr + info.offsets_g_eq_dyn[k] : nullptr;
                const Scalar *lam_eq_k = lam_ptr + info.offsets_g_eq_path[k];
                const Scalar *lam_eq_ineq_k = lam_ptr + info.offsets_g_eq_slack[k];
                if (ocp->eval_RSQrqt)
                    ocp->eval_RSQrqt(&objective_scale, inputs_k, states_k, lam_dyn_k, lam_eq_k,
                                     lam_eq_ineq_k, nullptr, nullptr, RSQrqt_buff[k], k,
                                     ocp->user_data);
            }
        }
        return 0;
    }
    template <typename ProblemType>
    Index FatropOcpCMapping<ProblemType>::eval_constr_jac(const ProblemInfo &info,
                                             const VecRealView &primal_x,
                                             const VecRealView &primal_s, Jacobian<ProblemType> &jac)
    {
        // take the matrices from jac and put them in the buffer
        std::vector<MAT *> &BAbt_buff = matrix_buffer_[0];
        std::vector<MAT *> &Gg_eqt_buff = matrix_buffer_[1];
        std::vector<MAT *> &Gg_ineqt_buff = matrix_buffer_[2];
        // get the double pointers for the vector views
        const Scalar *primal_x_ptr = primal_x.data();
        for (Index k = 0; k < K_; k++)
        {
            BAbt_buff[k] = &jac.BAbt[k].mat();
            Gg_eqt_buff[k] = &jac.Gg_eqt[k].mat();
            Gg_ineqt_buff[k] = &jac.Gg_ineqt[k].mat();
        }
        // call the C interface
        int ret = ocp->full_eval_constr_jac(primal_x_ptr, nullptr, nullptr, BAbt_buff[0],
                                            Gg_eqt_buff[0], Gg_ineqt_buff[0], &s, ocp->user_data);
        if (ret == 2)
            return 0;
        if (ret == 0)
        {
            for (Index k = 0; k < info.dims.K; k++)
            {
                const Scalar *inputs_k = primal_x_ptr + info.offsets_primal_u[k];
                const Scalar *states_k = primal_x_ptr + info.offsets_primal_x[k];
                if (ocp->eval_Ggt)
                    ocp->eval_Ggt(inputs_k, states_k, nullptr, nullptr, Gg_eqt_buff[k], k,
                                  ocp->user_data);

                if (ocp->eval_Ggt_ineq)
                    ocp->eval_Ggt_ineq(inputs_k, states_k, nullptr, nullptr, Gg_ineqt_buff[k], k,
                                       ocp->user_data);
                if (k != info.dims.K - 1)
                {
                    const Scalar *states_kp1 = primal_x_ptr + info.offsets_primal_x[k + 1];
                    if (ocp->eval_BAbt)
                        ocp->eval_BAbt(states_kp1, inputs_k, states_k, nullptr, nullptr,
                                       BAbt_buff[k], k, ocp->user_data);
                }
            }
        }
        return 0;
    }
    template <typename ProblemType>
    Index FatropOcpCMapping<ProblemType>::eval_constraint_violation(const ProblemInfo &info,
                                                       const VecRealView &primal_x,
                                                       const VecRealView &primal_s,
                                                       VecRealView &res)
    {
        // get the double pointers for the vector views
        const Scalar *primal_x_ptr = primal_x.data();
        Scalar *res_ptr = res.data();
        // call the C interface
        int ret =
            ocp->full_eval_contr_viol(primal_x_ptr, nullptr, nullptr, res_ptr, &s, ocp->user_data);
        if (ret == 2)
            return 0;
        if (ret == 0)
        {
            Scalar *res_p = res.data();
            for (Index k = 0; k < info.dims.K; k++)
            {
                const Scalar *inputs_k = primal_x_ptr + info.offsets_primal_u[k];
                const Scalar *states_k = primal_x_ptr + info.offsets_primal_x[k];
                if (ocp->eval_g)
                    ocp->eval_g(inputs_k, states_k, nullptr, nullptr,
                                res_ptr + info.offsets_g_eq_path[k], k, ocp->user_data);

                if (ocp->eval_gineq)
                    ocp->eval_gineq(inputs_k, states_k, nullptr, nullptr,
                                    res_ptr + info.offsets_g_eq_slack[k], k, ocp->user_data);

                if (k != info.dims.K - 1)
                {
                    const Scalar *states_kp1 = primal_x_ptr + info.offsets_primal_x[k + 1];
                    if (ocp->eval_b)
                        ocp->eval_b(states_kp1, inputs_k, states_k, nullptr, nullptr,
                                    res_ptr + info.offsets_g_eq_dyn[k], k, ocp->user_data);
                }
            }
        }
        // add -s to the slack constraints
        res.block(info.number_of_g_eq_slack, info.offset_g_eq_slack) =
            res.block(info.number_of_g_eq_slack, info.offset_g_eq_slack) -
            primal_s.block(info.number_of_g_eq_slack, 0);
        return 0;
    }
    template <typename ProblemType>
    Index FatropOcpCMapping<ProblemType>::eval_objective_gradient(const ProblemInfo &info,
                                                                     const Scalar objective_scale,
                                                                     const VecRealView &primal_x,
                                                                     const VecRealView &primal_s,
                                                                     VecRealView &grad_x, VecRealView &grad_s)
    {
        // get the double pointers for the vector views
        const Scalar *primal_x_ptr = primal_x.data();
        Scalar *grad_x_ptr = grad_x.data();
        // set grad_s to zero
        grad_s = 0.0;
        // call the C interface
        int ret = ocp->full_eval_obj_grad(objective_scale, primal_x.data(), nullptr, nullptr,
                                          grad_x.data(), &s, ocp->user_data);
        if (ret == 2)
            return 0;
        if (ret == 0)
        {
            for (Index k = 0; k < info.dims.K; k++)
            {
                const Scalar *inputs_k = primal_x_ptr + info.offsets_primal_u[k];
                const Scalar *states_k = primal_x_ptr + info.offsets_primal_x[k];
                // ocp_->eval_rq(&objective_scale, inputs_k, states_k,
                //               grad_x_p + info.offsets_primal_u[k], k);
                if (ocp->eval_rq)
                    return ocp->eval_rq(&objective_scale, inputs_k, states_k, nullptr, nullptr,
                                        grad_x_ptr + info.offsets_primal_u[k], k, ocp->user_data);
            }
        }
        return 0;
    }
    template <typename ProblemType>
    Index FatropOcpCMapping<ProblemType>::eval_objective(const ProblemInfo &info,
                                            const Scalar objective_scale,
                                            const VecRealView &primal_x,
                                            const VecRealView &primal_s, Scalar &res)
    {
        // get the double pointers for the vector views
        const Scalar *primal_x_ptr = primal_x.data();
        // call the C interface
        int ret = ocp->full_eval_obj(objective_scale, primal_x.data(), nullptr, nullptr, &res, &s,
                                     ocp->user_data);
        if (ret == 2)
            return 0;
        if (ret == 0)
        {
            res = 0;
            for (Index k = 0; k < info.dims.K; k++)
            {
                Scalar ret = 0;
                const Scalar *inputs_k = primal_x_ptr + info.offsets_primal_u[k];
                const Scalar *states_k = primal_x_ptr + info.offsets_primal_x[k];
                // ocp_->eval_L(&objective_scale, inputs_k, states_k, &ret, k);
                if (ocp->eval_L)
                    return ocp->eval_L(&objective_scale, inputs_k, states_k, nullptr, nullptr, &ret,
                                       k, ocp->user_data);

                res += ret;
            }
        }
        return 0;
    }
    template <typename ProblemType>
    Index FatropOcpCMapping<ProblemType>::get_bounds(const ProblemInfo &info, VecRealView &lower_bounds,
                                                     VecRealView &upper_bounds)
    {
        if (info.number_of_slack_variables == 0)
            return 0;
        Scalar *lower_bounds_p = lower_bounds.data();
        Scalar *upper_bounds_p = upper_bounds.data();
        for (Index k = 0; k < info.dims.K; k++)
        {
            Scalar *lower_bounds_k = lower_bounds_p + info.offsets_slack[k];
            Scalar *upper_bounds_k = upper_bounds_p + info.offsets_slack[k];
            ocp->get_bounds(lower_bounds_k, upper_bounds_k, k, ocp->user_data);
        }
        return 0;
    }
    template <typename ProblemType>
    Index FatropOcpCMapping<ProblemType>::get_initial_primal(const ProblemInfo &info,
                                                             VecRealView &primal_x)
    {
        Scalar *primal_x_ptr = primal_x.data();
        for (Index k = 0; k < info.dims.K; k++)
        {
            if (ocp->get_initial_uk)
                ocp->get_initial_uk(primal_x_ptr + info.offsets_primal_u[k], k, ocp->user_data);
            if (ocp->get_initial_xk)
                ocp->get_initial_xk(primal_x_ptr + info.offsets_primal_x[k], k, ocp->user_data);
        }
        return 0;
    }
    template <typename ProblemType>
    void FatropOcpCMapping<ProblemType>::get_primal_damping(const ProblemInfo &info,
                                                            VecRealView &damping)
    {
        damping = 0.0;
    }
    template <typename ProblemType>
    void FatropOcpCMapping<ProblemType>::apply_jacobian_s_transpose(const ProblemInfo &info,
                                                                    const VecRealView &multipliers,
                                                                    const Scalar alpha, const VecRealView &y,
                                                                    VecRealView &out)
    {
        out = alpha * y;
        out.block(info.number_of_slack_variables, 0) =
            out.block(info.number_of_slack_variables, 0) -
            multipliers.block(info.number_of_slack_variables, info.offset_g_eq_slack);
    }

    // Stream buffer for std::cout like printing
    class FatropOcpCStreambuf : public std::streambuf
    {
    public:
        FatropOcpCStreambuf(FatropOcpCWrite write_cb, FatropOcpCFlush flush_cb)
            : write(write_cb), flush(flush_cb)
        {
        }

    protected:
        int_type overflow(int_type ch) override
        {
            if (ch != traits_type::eof())
            {
                char c = static_cast<char>(ch);
                write(&c, 1);
            }
            return ch;
        }
        std::streamsize xsputn(const char *s, std::streamsize num) override
        {
            // Delegate to write
            // write uses 'int' instead of 'std::streamsize' so extra logic is needed
            int max_chunk = std::numeric_limits<int>::max();
            std::streamsize written = 0;
            while (num > 0)
            {
                int chunk = static_cast<int>(std::min<std::streamsize>(num, max_chunk));
                write(s + written, chunk);
                written += chunk;
                num -= chunk;
            }
            return written;
        }
        int sync() override
        {
            if (flush)
                flush();
            return 0;
        }

    private:
        FatropOcpCWrite write;
        FatropOcpCFlush flush;
    };

    class FatropOcpCStream : public std::ostream
    {
    protected:
        FatropOcpCStreambuf buf;

    public:
        FatropOcpCStream(FatropOcpCWrite write_cb, FatropOcpCFlush flush_cb)
            : std::ostream(&buf), buf(write_cb, flush_cb)
        {
        }
    };

    struct OcpSolverDriverBase
    {
        // constructor
        OcpSolverDriverBase(FatropOcpCInterface *ocp_interface, FatropOcpCWrite write,
                            FatropOcpCFlush flush)
            : stream(write, flush)
        {
        }

        virtual fatrop_int solve() = 0;
        virtual ~OcpSolverDriverBase() = default;

        virtual const VecRealView &solution_primal() const = 0;
        virtual const VecRealView &solution_dual() const = 0;

        virtual IpTimingStatistics &timing_statistics() const = 0;
        virtual Index iteration_number() const = 0;

        virtual FatropOcpCDims& s() const = 0;


        OptionRegistry options;
        FatropOcpCStream stream;
        FatropOcpCStats stats;
        IpSolverReturnFlag flag;
    };

    template <typename ProblemType>
    class OcpSolverDriver : public OcpSolverDriverBase
    {
    public:
        OcpSolverDriver(FatropOcpCInterface *ocp_interface, FatropOcpCWrite write,
                        FatropOcpCFlush flush)
            : OcpSolverDriverBase(ocp_interface, write, flush), m(std::make_shared<FatropOcpCMapping<ProblemType>>(ocp_interface))
        {
            // set the stream
            if (write != 0)
            {
                OutputStreamManager::set_stream(std::make_unique<FatropOcpCStream>(write, flush));
            }
            IpAlgBuilder<ProblemType> builder(m);
            algo = builder.with_options_registry(&options).build();
            ip_data = builder.get_ipdata();
            m->s.nx = algo->info().dims.number_of_states.data();
            m->s.nu = algo->info().dims.number_of_controls.data();
            m->s.ng = algo->info().dims.number_of_eq_constraints.data();
            m->s.ng_ineq = algo->info().dims.number_of_ineq_constraints.data();
            m->s.K = algo->info().dims.K;
            m->s.ux_offs = algo->info().offsets_primal_u.data();
            m->s.g_offs = algo->info().offsets_g_eq_path.data();
            m->s.dyn_offs = algo->info().offsets_dyn.data();
            m->s.dyn_eq_offs = algo->info().offsets_g_eq_dyn.data();
            m->s.g_ineq_offs = algo->info().offsets_g_eq_slack.data();
            m->s.max_nu = *std::max_element(algo->info().dims.number_of_controls.begin(),
                                            algo->info().dims.number_of_controls.end());
            m->s.max_nx = *std::max_element(algo->info().dims.number_of_states.begin(),
                                            algo->info().dims.number_of_states.end());
            m->s.max_ng = *std::max_element(algo->info().dims.number_of_eq_constraints.begin(),
                                            algo->info().dims.number_of_eq_constraints.end());
            m->s.max_ngineq =
                *std::max_element(algo->info().dims.number_of_ineq_constraints.begin(),
                                  algo->info().dims.number_of_ineq_constraints.end());
            m->s.n_ineqs = algo->info().number_of_g_eq_slack;
        }
        fatrop_int solve()
        {
            flag = algo->optimize();
            if (flag == IpSolverReturnFlag::Success)
            {
                PRINT_ITERATIONS << ip_data->timing_statistics();
                return 0;
            }
            return 1;
        }

        const VecRealView &solution_primal() const override { return algo->solution_primal(); }
        const VecRealView &solution_dual() const override { return algo->solution_dual(); }
        IpTimingStatistics &timing_statistics() const override { return ip_data->timing_statistics(); }
        Index iteration_number() const override { return ip_data->iteration_number(); }
        FatropOcpCDims& s() const override { return m->s; }

        // std::shared_ptr<FatropPrinter> printer() { return app.printer_; }
        std::shared_ptr<FatropOcpCMapping<ProblemType>> m;
        std::shared_ptr<IpAlgorithm<ProblemType>> algo;
        std::shared_ptr<IpData<ProblemType>> ip_data;
    };

    FatropOcpCSolver *fatrop_ocp_c_create(FatropOcpCInterface *ocp_interface, FatropOcpCWrite write,
                                          FatropOcpCFlush flush)
    {
        FatropOcpCSolver *ret = new FatropOcpCSolver();
        // skip construction of driver since we do not know yet which ocp type is used
        // instead, store the arguments
        ret->ocp_interface = ocp_interface;
        ret->write = write;
        ret->flush = flush;

        // ret->driver = new OcpSolverDriver<OcpType>(ocp_interface, write, flush);
        // ret->driver = new OcpSolverDriver<AcceleratedOcpType>(ocp_interface, write, flush);
        return ret;
    }

    int fatrop_ocp_c_set_option_double(FatropOcpCSolver *s, const char *name, double val)
    {
        if (!s->driver){
            // driver is not yet constructed, so we store the option for later
            s->pending_options.push_back(PendingOption{std::string(name), PendingOption::Type::Double, .d = val});
            return 0;
        }
        // Backwards compatibility with v0.0.4
        if (std::string(name)=="tol") {
            s->driver->options.set_option<double>("tolerance", val);
            return 0;
        }
        s->driver->options.set_option<double>(name, val);
        return 0;
    }

    int fatrop_ocp_c_set_option_bool(FatropOcpCSolver *s, const char *name, int val)
    {
        if (!s->driver){
            // driver is not yet constructed, so we store the option for later
            s->pending_options.push_back(PendingOption{std::string(name), PendingOption::Type::Bool, .b = static_cast<bool>(val)});
            return 0;
        }
        s->driver->options.set_option<bool>(name, val);
        return 0;
    }

    int fatrop_ocp_c_set_option_int(FatropOcpCSolver *s, const char *name, int val)
    {
        if (!s->driver){
            // driver is not yet constructed, so we store the option for later
            s->pending_options.push_back(PendingOption{std::string(name), PendingOption::Type::Int, .i = val});
            return 0;
        }
        s->driver->options.set_option<int>(name, val);
        return 0;
    }

    int fatrop_ocp_c_set_option_string(FatropOcpCSolver *s, const char *name, const char *val)
    {
        if (std::string(name)=="problem_type") {
            // we can set the driver
            if (s->driver) {
                std::cerr << "Error: problem type seems to be set multiple times" << std::endl;
                return -1;
            }
            s->set_problem_type(std::string(val));
            return 0;
        }
        if (!s->driver){
            // driver is not yet constructed, so we store the option for later
            s->pending_options.push_back(PendingOption{std::string(name), PendingOption::Type::String, .s = std::string(val)});
            return 0;
        }
        s->driver->options.set_option<std::string>(name, std::string(val));
        return 0;
    }

    const blasfeo_dvec *fatrop_ocp_c_get_primal(FatropOcpCSolver *s)
    {
        // todo implement
        return static_cast<const blasfeo_dvec *>(&s->driver->solution_primal().vec());
    }

    const blasfeo_dvec *fatrop_ocp_c_get_dual(FatropOcpCSolver *s)
    {
        // todo implement
        return static_cast<const blasfeo_dvec *>(&s->driver->solution_dual().vec());
    }

    int fatrop_ocp_c_solve(FatropOcpCSolver *s)
    {
        if (!s->driver){
            // driver has not yet been constructed, do it now
            s->set_problem_type("ocp_type");
        }
        try
        {
            return s->driver->solve();
        }
        catch (std::exception &e)
        {
            // todo implement
            PRINT_ITERATIONS << "Uncaught Exception: " << e.what() << std::endl;
            return -1;
        }
    }

    void fatrop_ocp_c_destroy(FatropOcpCSolver *s)
    {
        delete s->driver;
        delete s;
    }

    int fatrop_ocp_c_option_type(const char *name)
    {
        // todo implement
        std::string n = name;
        if (n == "acceptable_iter")
            return 1;
        if (n == "compl_inf_tol")
            return 0;
        if (n == "mu_superlinear_decrease_power")
            return 0;
        if (n == "bound_frac")
            return 0;
        if (n == "barrier_tol_factor")
            return 0;
        if (n == "lam_max")
            return 0;
        if (n == "max_soft_resto_iters")
            return 1;
        if (n == "soft_rest_pd_error_reduction_factor")
            return 0;
        if (n == "alpha_red_factor")
            return 0;
        if (n == "watchdog_trial_iter_max")
            return 1;
        if (n == "max_iter")
            return 1;
        if (n == "print_level")
            return 1;
        if (n == "watchdog_shortened_iter_trigger")
            return 1;
        if (n == "obj_max_incr")
            return 0;
        if (n == "theta_min")
            return 0;
        if (n == "s_phi")
            return 0;
        if (n == "tau_min")
            return 0;
        if (n == "max_soc")
            return 1;
        if (n == "kappa_c")
            return 0;
        if (n == "kappa_wplus")
            return 0;
        if (n == "constr_mult_reset_treshold")
            return 0;
        if (n == "kappa_wplusem")
            return 0;
        if (n == "mu_linear_decrease_factor")
            return 0;
        if (n == "resto_failure_feasibility_treshold")
            return 0;
        if (n == "s_theta")
            return 0;
        if (n == "kappa_wmin")
            return 0;
        if (n == "bound_push")
            return 0;
        if (n == "delta_c_stripe")
            return 0;
        if (n == "delta_wmin")
            return 0;
        if (n == "kappa_soc")
            return 0;
        if (n == "max_filter_resets")
            return 1;
        if (n == "bound_mult_reset_treshold")
            return 0;
        if (n == "tolerance" || n == "tol")
            return 0;
        if (n == "theta_max")
            return 0;
        if (n == "delta")
            return 0;
        if (n == "delta_w0")
            return 0;
        if (n == "theta_min_fact")
            return 0;
        if (n == "tiny_step_tol")
            return 0;
        if (n == "constr_viol_tol")
            return 0;
        if (n == "tiny_step_y_tol")
            return 0;
        if (n == "mu_init")
            return 0;
        if (n == "eta_phi")
            return 0;
        if (n == "alpha_min_frac")
            return 0;
        if (n == "tol_acceptable")
            return 0;
        if (n == "mu_allow_fast_monotone_decrease")
            return 2;
        if (n == "gamma_theta")
            return 0;
        if (n == "gamma_phi")
            return 0;
        if (n == "theta_max_fact")
            return 0;
        if (n == "filter_reset_trigger")
            return 1;
        if (n == "linsol_it_ref")
            return 2;
        if (n == "linsol_perturbed_mode")
            return 2;
        if (n == "linsol_perturbed_mode_param")
            return 0;
        if (n == "linsol_lu_fact_tol")
            return 0;
        if (n == "linsol_diagnostic")
            return 2;
        if (n == "linsol_increased_accuracy")
            return 2;
        if (n == "linsol_nb_of_dynamics_constraints")
            return 1;
        if (n == "linsol_nb_of_zk_vars")
            return 1;
        if (n == "problem_type")
            return 3;
        return -1;
    }

    const struct FatropOcpCDims *fatrop_ocp_c_get_dims(struct FatropOcpCSolver *s)
    {
        return &s->driver->s();
    }

    const FatropOcpCStats *fatrop_ocp_c_get_stats(struct FatropOcpCSolver *s)
    {
        FatropOcpCStats *stats = &s->driver->stats;

        stats->compute_sd_time = s->driver->timing_statistics().compute_search_dir.elapsed();
        stats->duinf_time = 0.;
        stats->eval_hess_time = s->driver->timing_statistics().eval_hessian.elapsed();
        stats->eval_jac_time = s->driver->timing_statistics().eval_jacobian.elapsed();
        stats->eval_cv_time = s->driver->timing_statistics().eval_constraint_violation.elapsed();
        stats->eval_grad_time = s->driver->timing_statistics().eval_gradient.elapsed();
        stats->eval_obj_time = s->driver->timing_statistics().eval_objective.elapsed();
        stats->initialization_time = s->driver->timing_statistics().initialization.elapsed();
        stats->time_total = s->driver->timing_statistics().full_algorithm.elapsed();
        stats->eval_hess_count = 0.;
        stats->eval_jac_count = 0.;
        stats->eval_cv_count = 0.;
        stats->eval_grad_count = 0.;
        stats->eval_obj_count = 0.;
        stats->iterations_count = s->driver->iteration_number();
        stats->return_flag = int(s->driver->flag);

        return stats;
    }

        inline void FatropOcpCSolver::set_problem_type(const std::string &problem_type){
        this->problem_type = problem_type;
        std::cout << "Creating OcpSolver Driver for problem type: " << problem_type << std::endl;
        if (problem_type == "ocp_type"){
            this->driver = new OcpSolverDriver<OcpType>(ocp_interface, write, flush);
        } else if (problem_type == "accelerated_ocp_type"){
            this->driver = new OcpSolverDriver<AcceleratedOcpType>(ocp_interface, write, flush);
        } else {
            throw std::runtime_error("Unknown problem type: " + problem_type);
        }
        // set the pending options
        for (const auto &option : pending_options){
            switch (option.type){
                case PendingOption::Type::Double:
                    fatrop_ocp_c_set_option_double(this, option.name.c_str(), option.d);
                    break;
                case PendingOption::Type::Int:
                    fatrop_ocp_c_set_option_int(this, option.name.c_str(), option.i);
                    break;
                case PendingOption::Type::Bool:
                    fatrop_ocp_c_set_option_bool(this, option.name.c_str(), option.b);
                    break;
                case PendingOption::Type::String:
                    fatrop_ocp_c_set_option_string(this, option.name.c_str(), option.s.c_str());
                    break;
            }
        }
        // clear pending options
        pending_options.clear();
    };

    template class FatropOcpCMapping<OcpType>;
    template class FatropOcpCMapping<AcceleratedOcpType>;
    template class OcpSolverDriver<OcpType>;
    template class OcpSolverDriver<AcceleratedOcpType>;

}
