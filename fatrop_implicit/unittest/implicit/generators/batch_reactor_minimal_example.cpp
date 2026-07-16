#include <casadi/casadi.hpp>

using namespace casadi;

// minimal example of batch_reactor_generator::SolveOptiInstance()
MX ki(int i, double k0, double E, double R, MX T){ return k0 * exp(-E/(R*T));}

int main(){
        // specify OCP parameters //
        int K = 10;
        double T = 1;
        double dt;

        int nx = 6;
        int nu = 2;

        std::vector<double> x0 = {0.95, 0.0, 0.0, 0.0, 698.15, 698.15};

        // chemical properties
        std::vector<double> k0 = {exp(8.86), exp(24.25), exp(23.67), exp(18.75), exp(20.7)};
        std::vector<double> E = {20300, 37400, 33800, 28200, 31000};
        double R = 1.9872;
        double UA = 1000;

        /// Define ODE ///
        MX xk(nx, 1);
        MX uk(nu, 1);
        MX ki = MX(5,1);
        for (int i = 0; i < 5; i++){
            ki(i) = k0[i] * exp(-E[i]/(R*xk(4)));
        }
        MX rho = 2139*xk(0) + 760*xk(1) + 560*xk(2) + 1800*xk(3);
        MX Cp = 1.16*xk(0) + 0.827*(1 - xk(0)) + (3.4*xk(0) + 0.92*(1-xk(0)))*(xk(4)-298)*1.0e-3;
        // Function rhs = Function("rhs", {uk, xk}, {
        //     vertcat(
        //         -ki(0)*xk(0) - (ki(2) + ki(3) + ki(4))*xk(0)*xk(1),
        //         ki(0)*xk(0) - ki(1)*xk(1) + ki(2)*xk(0)*xk(1),
        //         ki(1)*xk(1) + ki(3)*xk(0)*xk(1),
        //         ki(4)*xk(0)*xk(1),
        //         (602.4*ki(0)*xk(0) - 0.833*UA/rho*(xk(4)-xk(5))) / Cp,
        //         uk(0)*(873 - xk(5)) + uk(1)*(373 - xk(5)) + 0.01357*UA*(xk(4)-xk(5))
        //     )
        // });
        Function rhs = Function("rhs", {uk, xk}, {MX::zeros(nx,1)});


        /// Define states and controls ///
        Opti opti = Opti();
        std::vector<MX> xx_list = {};
        std::vector<MX> uu_list = {};
        for (int k = 0; k < K + 1; k++){
            xx_list.push_back(opti.variable(nx));
            if (k < K){
                uu_list.push_back(opti.variable(nu));
            }
        }
        MX xx = vertcat(xx_list);
        MX uu = vertcat(uu_list);


        /// Initial constraint ///
        opti.subject_to(xx_list[0] == x0);

        /// Dynamics and path constraints ///
        for (int k = 0; k < K; k++){
            MX xk = xx_list[k];
            MX uk = uu_list[k];

            MX x_next = xk + dt*rhs(MXVector{uk, xk})[0];
            opti.subject_to(xx_list[k+1] == x_next);

            opti.subject_to(DM({0, 0, 0, 0, 0, 0, 698.15}) <= (vertcat(uk, xk(Slice(0,5))) <= DM({5, 5, 1, 1, 1, 1, 748.15})));
        }

        MX obj = 0;
        for (int k = 0; k < K; k++){
            MX xk = xx_list[k];
            MX uk = uu_list[k];
            obj += 1e-6*(uk(0)*uk(0) * uk(1)*uk(1));
        }
        xk = xx_list[K];
        obj += -xk(1);
        opti.minimize(obj);

        for (int k = 0; k < K+1; k++){
            opti.set_initial(xx_list[k], x0);
            if (k < K){
                opti.set_initial(uu_list[k](0), 1);
                opti.set_initial(uu_list[k](1), 0);
            }
        }

        // define options
        Dict casadi_opts;
        Dict solver_opts;
        solver_opts["max_iter"] = 200;
        std::string solver_name = "fatrop";
        // std::string solver_name = "ipopt";

        if (solver_name == "fatrop"){
            casadi_opts["structure_detection"] = "auto";
            casadi_opts["fatrop.mu_init"] = 0.1;
        } else if (solver_name == "ipopt"){
            solver_opts["linear_solver"] = "ma27";
        }
        opti.solver(solver_name, casadi_opts, solver_opts);

        try{
            opti.solve();
        } catch (std::exception& e){
            std::cout << "Exception: " << e.what() << std::endl;
        }
}