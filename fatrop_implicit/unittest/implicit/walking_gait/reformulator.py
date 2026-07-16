from casadi import *
from collocation_scheme import collocation_scheme
import matplotlib.pyplot as plt

class Reformulator:
    def __init__(self, n_coll, n_coords, n_act_mesh, dt, xk, uk, uk_orig, qcoll, qdotcoll, qddotcoll):
        [tau, C, D, B] = collocation_scheme(n_coll, 'radau')
        self.C = C[:,1:]
        # assert n_coll == 3
        assert n_coords == 9
        assert n_act_mesh == 18
        self.n_coll = n_coll
        self.n_coords = n_coords
        self.dt = dt
        self.xk = xk
        self.uk = uk
        self.uk_orig = uk_orig
        self.n_act_mesh = n_act_mesh
        
        self.keep_orginal_constraint_order = False
        self.substitute_q = False
        self.substitute_qdot = False
        self.eliminate_in_dynamics_equations = False
        
        self.q_mesh = self.xk[:self.n_coords]
        self.qdot_mesh = self.xk[self.n_coords:2*self.n_coords]
        
        self.q_coll = qcoll
        self.qdot_coll = qdotcoll
        self.qddot_coll = qddotcoll
        
        self.scale_qdots = 10
        self.scale_qddots = 100
        
        self.reconstruct_err_coll_general()
        
    def show_gk_jacobian_structure_for_debugging(self, f_gk):
        fig = plt.figure()
        # show jacobian
        plt.spy(jacobian(f_gk(self.xk, self.uk), vertcat(self.xk, self.uk)).sparsity())
        
        alpha = 0.2
        zorder = 1
        
        # for illustrations, highlight different block structures
        n_rows = f_gk(self.xk, self.uk).shape[0]
        col_ptr = 0
        # show states
        plt.gca().add_patch(plt.Rectangle((col_ptr, 0), 2*self.n_coords, n_rows, color='red', lw=0, alpha=alpha, zorder=zorder, label='[q_mesh, qdot_mesh]'))
        col_ptr += 2*self.n_coords
        
        plt.gca().add_patch(plt.Rectangle((col_ptr, 0), 2*self.n_coords - 1, n_rows, color='darkred', lw=0, alpha=alpha, zorder=zorder, label='[q0_mesh, q0dot_mesh]'))
        col_ptr += 2*self.n_coords - 1
        
        # show controls
        plt.gca().add_patch(plt.Rectangle((col_ptr, 0), self.n_act_mesh, n_rows, color='blue', lw=0, alpha=alpha, zorder=zorder, label='act_mesh'))
        col_ptr += self.n_act_mesh
        c = 'blue'
        for i in range(self.n_coll-1):
            plt.gca().add_patch(plt.Rectangle((col_ptr, 0), 3*self.n_coords, n_rows, color=c, lw=0, alpha=alpha, zorder=zorder, label='[q_coll, qdot_coll, qddot_coll]'))
            col_ptr += 3*self.n_coords
            c = 'darkblue' if c == 'blue' else 'blue'
        plt.gca().add_patch(plt.Rectangle((col_ptr, 0), self.n_coords, n_rows, color='blue', lw=0, alpha=alpha, zorder=zorder, label='[qddot_coll]'))
        col_ptr += self.n_coords
        
        # show zk variables
        plt.gca().add_patch(plt.Rectangle((col_ptr, 0), 2*self.n_coords, n_rows, color='green', lw=0, alpha=alpha, zorder=zorder, label='[qk, qdotk]'))
        col_ptr += 2*self.n_coords
        
        plt.gca().add_patch(plt.Rectangle((col_ptr, 0), 2*self.n_coords - 1, n_rows, color='darkgreen', lw=0, alpha=alpha, zorder=zorder, label='[q0_mesh, q0dot_mesh]'))
        col_ptr += 2*self.n_coords - 1
        
        # plt.legend()
        plt.savefig(f'jacobian_structure_{f_gk.name()}.png', dpi=300)
        plt.close()
        

    def reconstruct_err_coll_general(self):
        if not self.substitute_q and not self.substitute_qdot:
            err_coll_q_func = Function.load(f'casadi_funcs/n_coll_{self.n_coll}/err_coll_q_func.casadi')
            err_coll_qdot_func = Function.load(f'casadi_funcs/n_coll_{self.n_coll}/err_coll_qdot_func.casadi')
            self.err_coll = Function('err_coll', [self.xk, self.uk], [vertcat(err_coll_q_func(self.xk, self.uk_orig), err_coll_qdot_func(self.xk, self.uk_orig)/self.scale_qdots)])
            self.uk_subs = Function('uk_subs', [self.xk, self.uk], [self.uk])
            return
        
        q_coll_list = [self.q_coll[i*self.n_coords:(i+1)*self.n_coords] for i in range(self.n_coll)]
        qdot_coll_list = [self.qdot_coll[i*self.n_coords:(i+1)*self.n_coords] for i in range(self.n_coll)]
        qddot_coll_list = [self.qddot_coll[i*self.n_coords:(i+1)*self.n_coords] for i in range(self.n_coll)]

        scale_qdots = self.scale_qdots
        scale_qddots = self.scale_qddots

        # For the last collocation point, we might substitute it.
        # This is based on the logic in the original reconstruct_err_coll
        q_last_coll = q_coll_list[-1]
        qdot_last_coll = qdot_coll_list[-1]

        ### Perform substitutions
        if self.substitute_qdot:
            sum_C_qdot = scale_qdots*self.C[0,0]*self.qdot_mesh
            for i in range(self.n_coll - 1):
                sum_C_qdot += scale_qdots*self.C[i+1, 0] * qdot_coll_list[i]

            qdot_last_coll_subs = -1.0/(scale_qdots*self.C[self.n_coll, 0]) * (sum_C_qdot - scale_qddots*self.dt*qddot_coll_list[0])
            qdot_last_coll = qdot_last_coll_subs
            
        if self.substitute_q:           
            sum_C_q = self.C[0,0]*self.q_mesh
            for i in range(self.n_coll - 1):
                sum_C_q += self.C[i+1, 0] * q_coll_list[i]
            
            q_last_coll_subs = -1.0/self.C[self.n_coll, 0] * (sum_C_q - scale_qdots*self.dt*(qdot_coll_list[0] if self.n_coll > 1 else qdot_last_coll))
            q_last_coll = q_last_coll_subs
        


        ### Write down equations for q_coll
        err_coll_q_list = []
        for i in range(self.n_coll):
            expr = self.C[0, i] * self.q_mesh
            for j in range(self.n_coll - 1):
                expr += self.C[j+1, i] * q_coll_list[j]

            # don't use substitution in first equation (would lead to 0 = 0)
            expr += self.C[self.n_coll, i] * (q_last_coll if i > 0 else q_coll_list[-1])
            
            # make sure to use substitution of qdot here as well if needed
            if i == self.n_coll - 1:
                expr -= scale_qdots * self.dt * qdot_last_coll
            else:
                expr -= scale_qdots * self.dt * qdot_coll_list[i]
            err_coll_q_list.append(expr)
        err_coll_q_reconstructed = vertcat(*err_coll_q_list)

        ### Write down equations for qdot_coll
        err_coll_qdot_list = []
        for i in range(self.n_coll):
            expr = scale_qdots * self.C[0, i] * self.qdot_mesh
            for j in range(self.n_coll - 1):
                expr += scale_qdots * self.C[j+1, i] * qdot_coll_list[j]

            # don't use substitution in first equation (would lead to 0 = 0)
            expr += scale_qdots * self.C[self.n_coll, i] * (qdot_last_coll if i > 0 else qdot_coll_list[-1])
            expr -= scale_qddots * self.dt * qddot_coll_list[i]
            err_coll_qdot_list.append(expr)
        err_coll_qdot_reconstructed = vertcat(*err_coll_qdot_list)

        self.err_coll = Function('err_coll', [self.xk, self.uk], [vertcat(err_coll_q_reconstructed, err_coll_qdot_reconstructed/self.scale_qdots)])
        
        uk_substituted_list = [self.uk[:self.n_act_mesh]]
        for i in range(self.n_coll - 1):
            uk_substituted_list.append(q_coll_list[i])
            uk_substituted_list.append(qdot_coll_list[i])
            uk_substituted_list.append(qddot_coll_list[i])
        
        # Add the last collocation variables
        uk_substituted_list.append(qddot_coll_list[-1])
        uk_substituted_list.append(q_last_coll)
        uk_substituted_list.append(qdot_last_coll)
        
        # Add rest of uk
        start_idx = self.n_act_mesh + self.n_coll*self.n_coords*3
        uk_substituted_list.append(self.uk[start_idx:])
        
        uk_substituted = vertcat(*uk_substituted_list)
        
        self.uk_subs = Function('uk_subs', [self.xk, self.uk], [uk_substituted])
        
        ### Test
        xk_test = [1 + i*0.1 for i in range(self.xk.shape[0])]
        uk_test = [1 + i*0.1 for i in range(self.uk.shape[0])]
        temp = self.err_coll(xk_test, self.uk_subs(xk_test, uk_test)).full()
        # assert max(abs(temp)) < 1e-10, "Error in reconstruct_err_coll_general"
                        
    def get_f_gk_reformulated(self, f_gk):
        if self.eliminate_in_dynamics_equations:
            err_sysdyn_rewritten = f_gk(self.xk, self.uk_subs(self.xk, self.uk))[self.n_coll*self.n_coords*2:self.n_coll*self.n_coords*3]
        else:
            err_sysdyn_rewritten = f_gk(self.xk, self.uk)[self.n_coll*self.n_coords*2:self.n_coll*self.n_coords*3]
        err_coll = self.err_coll(self.xk, self.uk)
        subs_q_coll_eqs = err_coll[:self.n_coords]
        other_q_coll_eqs = err_coll[self.n_coords:self.n_coll*self.n_coords]
        subs_qdot_coll_eqs = err_coll[self.n_coll*self.n_coords:self.n_coll*self.n_coords + self.n_coords]
        other_qdot_coll_eqs = err_coll[self.n_coll*self.n_coords + self.n_coords:]
                
        if self.keep_orginal_constraint_order:
            err_coll_arranged = vertcat(
                subs_q_coll_eqs,            # original order
                other_q_coll_eqs,
                subs_qdot_coll_eqs,
                other_qdot_coll_eqs,
            )
        else:
            err_coll_arranged = vertcat(
                other_q_coll_eqs,           # desired order
                other_qdot_coll_eqs,
                subs_q_coll_eqs,
                subs_qdot_coll_eqs
            )
        
        
        periodic_constraints = f_gk(self.xk, self.uk)[self.n_coll*self.n_coords*3:]
        if self.keep_orginal_constraint_order:
            output = vertcat(err_coll_arranged,        # desired order
                             err_sysdyn_rewritten, 
                             periodic_constraints)
        else:
            output = vertcat(err_sysdyn_rewritten,     # desired order
                             err_coll_arranged,
                             periodic_constraints)
            
        self.f_gk_reformulated = Function('f_gk_reformulated', [self.xk, self.uk], [output])
        
        nb_constraints_without_zk = self.n_coords*(self.n_coll - 1)
        if self.substitute_q:
            nb_constraints_without_zk += self.n_coords*(self.n_coll - 1)
        if self.substitute_qdot:
            nb_constraints_without_zk += self.n_coords*(self.n_coll - 1)
        if self.eliminate_in_dynamics_equations and self.substitute_q and self.substitute_qdot:
            nb_constraints_without_zk += self.n_coords
            
        print(f"Number of constraints without zk: {nb_constraints_without_zk}")
        print(f"Number of constraints in f_gk_reformulated: {self.f_gk_reformulated(self.xk, self.uk).shape[0]}")
        self.number_of_constraints_with_zk = self.f_gk_reformulated(self.xk, self.uk).shape[0] - nb_constraints_without_zk
        print(f"Number of constraints with zk: {self.number_of_constraints_with_zk}")
        return self.f_gk_reformulated
