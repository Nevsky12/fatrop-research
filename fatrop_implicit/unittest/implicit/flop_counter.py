import numpy as np
import matplotlib.pyplot as plt

class FlopCounter:
    def __init__(self):
        self.implicit = False
        self.rank_mode = 'full' 
        self.ignore_FuFx_GuGx = False  
        self.ignore_regularization = False

        self.reset_flops()

    def reset_flops(self):
        self.flops = 0
        self.flops_preprocess = 0
        self.flops_backward_recursion = 0
        self.flops_initial_stage = 0
        self.flops_forward_substitution = 0
        self.flops_postprocess = 0

    def set_dimensions(self, K, nu, nx, r, ng, ng_ineq):
        self.K = K
        if isinstance(nu, int) or isinstance(nu, np.int64):
            nu = [nu for _ in range(K)]
        if isinstance(nx, int) or isinstance(nx, np.int64):
            nx = [nx for _ in range(K)]
        if isinstance(r, int) or isinstance(r, np.int64):
            r = [r for _ in range(K-1)]
        if isinstance(ng, int) or isinstance(ng, np.int64):
            ng = [ng for _ in range(K)]
        if isinstance(ng_ineq, int) or isinstance(ng_ineq, np.int64):
            ng_ineq = [ng_ineq for _ in range(K)]
        self.nu = nu
        self.nx = nx
        self.r = r
        self.ng = ng
        self.ng_ineq = ng_ineq

        self.rho = [0 for _ in range(K)]
        self.gamma = [0 for _ in range(K)]   

    def get_all_flops(self, data):
        impl = []
        expl = []
        reform = []
        for i in range(len(data['K'])):
            self.set_dimensions(data['K'][i], data['nu'][i], data['nx'][i], 
                                data['r'][i], data['ng'][i], data['ng_ineq'][i])
            self.get_flop_implicit(); impl.append(self.flops)
            self.get_flop_explicit(); expl.append(self.flops)
            self.get_flop_reformulated(); reform.append(self.flops)

            # print progress
            progress = (i + 1) / len(data['K']) * 100
            print(f"Progress: {progress:3.2f}%", end='\r')

        return np.array(impl), np.array(expl), np.array(reform)


    def get_flop_explicit(self):
        self.implicit = False
        self.reset_flops()
        self.backward_recursion()
        self.first_stage()
        self.forward_substitution()

        return self.flops

    def get_flop_implicit(self):
        self.implicit = True
        self.reset_flops()
        self.preprocess()
        self.backward_recursion()
        self.first_stage()
        self.forward_substitution()
        self.postprocess()

        return self.flops

    def get_flop_reformulated(self):
        for k in range(self.K-1):
            self.nu[k] += self.nx[k+1]
            self.ng[k] += self.nx[k+1]

        flops = self.get_flop_explicit()

        for k in range(self.K-1):
            self.nu[k] -= self.nx[k+1]
            self.ng[k] -= self.nx[k+1]
        
        return self.flops

    def get_detailed_flops(self):
        return {
            'preprocess': self.flops_preprocess,
            'backward_recursion': self.flops_backward_recursion,
            'initial_stage': self.flops_initial_stage,
            'forward_substitution': self.flops_forward_substitution,
            'postprocess': self.flops_postprocess
        }

    def set_constraint_jacobian_rank_mode(self, mode):
        if mode != 'full' and mode != 'random':
            raise ValueError(f"Unknown constraint jacobian rank mode: {mode}")
        self.rank_mode = mode

    def backward_recursion(self):
        flops_at_start = self.flops
        Hkp_rows = self.ng[self.K-1]
        gamma = Hkp_rows
        self.gamma_I = Hkp_rows

        self.flops_backward_recurion_implicit_specific = 0
        regularization_flops = 0
        for k in range(self.K-1, -1, -1):
            nu = self.nu[k]
            nx = self.nx[k]
            ng = self.ng[k]
            ng_ineq = self.ng_ineq[k]
            nunxm1 = self.nu[k-1] + self.nx[k-1] if k > 0 else 0
            nxp1 = self.nx[k+1] if k < self.K-1 else 0

            if k < self.K-1:
                self.gemm(nu + nx + 1, nxp1, nxp1)
                self.gead(1, nxp1)
                self.syrk(nu + nx + 1, nu + nx, nxp1)

                # second order dynamics contribution
                f = self.flops
                if self.implicit and not self.ignore_FuFx_GuGx and nxp1 > 0:
                    self.gemm(nu + nx + 1, nu + nx, nxp1)
                    self.gemm(nu + nx, nu + nx, nxp1)
                    self.flops_backward_recurion_implicit_specific += self.flops - f

                gamma = self.ng[k] + Hkp_rows
                if gamma > 0 and Hkp_rows > 0:
                    self.gemm(nu + nx + 1, Hkp_rows, nxp1)
                    self.gead(1, Hkp_rows)

            if not self.implicit and not self.ignore_regularization:
                f = self.flops
                if self.ng_ineq[k] > 0:
                    for i in range(self.ng_ineq[k]):
                        self.colsc(nu + nx + 1)
                        self.flops += nu + nx + 1
                    self.syrk(nu + nx + 1, nu + nx, self.ng_ineq[k])

                self.flops += self.nu[k] + self.nx[k]
                regularization_flops += self.flops - f

            # decomposition of constraint jacobian
            rank = min(gamma, nu + nx) if self.rank_mode == 'full' else 0 if min(gamma, nu + nx) == 0 else np.random.randint(0, min(gamma, nu + nx))
            self.lu_fact(gamma, nu + nx + 1, rank)
            if rank > 0:
                self.trsm(nu - rank + nx + 1, rank)
                self.gemm(nu - rank + nx + 1, nu + nx, rank)
                self.syrk(nu - rank + nx + 1, nu + nx - rank, rank)

            if self.implicit and k > 0 and not self.ignore_FuFx_GuGx and self.r[k-1] < self.nx[k]:
                f = self.flops
                self.gemm(nunxm1, nu - rank, rank)
                self.gemm(nunxm1, nx, rank)
                self.gemm(1, nunxm1, rank)
                self.flops_backward_recurion_implicit_specific += self.flops - f

            # shur complement step
            if nu - rank > 0:
                self.potrf(nu - rank, nu - rank)
                self.syrk(nx + 1, nx, nu - rank)
                # skip increased accuracy steps
                if self.implicit and not self.ignore_FuFx_GuGx and k > 0 and self.r[k-1] < self.nx[k]:
                    f = self.flops
                    self.trsm(nunxm1, nu - rank)
                    self.gemm(nunxm1, nunxm1, nu - rank)
                    self.trsm(1, nu - rank)
                    self.gemm(1, nunxm1, nu-rank)
                    self.gemm(nunxm1, nx, nu - rank)
                    self.flops_backward_recurion_implicit_specific += self.flops - f

            self.gamma[k] = gamma
            self.rho[k] = rank
            self.gamma_I = gamma - rank

        self.flops_backward_recursion = self.flops - flops_at_start

    def first_stage(self):
        flops_at_start = self.flops

        nx = self.nx[0]
        self.rank_I = min(self.gamma_I, nx + 1) if self.rank_mode == 'full' else 0 if min(self.gamma_I, nx+1) == 0 else np.random.randint(0, min(self.gamma_I, nx + 1))
        if self.gamma_I > 0:
            self.lu_fact(self.gamma_I, nx + 1, self.rank_I)

            self.trsm(nx - self.rank_I + 1, self.rank_I)
            self.gemm(nx - self.rank_I + 1, nx, self.rank_I)
            self.syrk(nx - self.rank_I + 1, nx - self.rank_I, self.rank_I)
            self.potrf(nx - self.rank_I + 1, nx - self.rank_I)
        else:
            self.potrf(nx + 1, nx)

        self.trsv(nx - self.rank_I)
        self.gemv(nx - self.rank_I, self.rank_I)
        self.gemv(nx, self.rank_I)
        self.trsv(self.rank_I)
        self.trsv(self.rank_I)

        self.flops_initial_stage = self.flops - flops_at_start

    def forward_substitution(self):
        flops_at_start = self.flops
        regularization_flops = 0

        self.flops_forward_substitution_implicit_specific = 0
        for k in range(self.K):
            nx = self.nx[k]
            nu = self.nu[k]
            rho = self.rho[k]
            gammamrho = self.gamma[k] - rho
            gamma = self.gamma[k]
            numrho = nu - rho

            if numrho > 0:
                # skip increased accuracy
                self.gemv(nx, numrho)
                self.trsv(numrho)

                if self.implicit and k > 0 and self.r[k-1] < self.nx[k]:
                    f = self.flops
                    nunxm1 = self.nu[k-1] + self.nx[k-1]
                    self.trsm(nunxm1, numrho)
                    self.gemv(nunxm1, numrho)
                    self.flops_forward_substitution_implicit_specific += self.flops - f

            if rho > 0:
                self.gemv(nx + numrho, rho)
                self.gemv(nu + nx, rho)
                if self.implicit and not self.ignore_FuFx_GuGx and k > 0 and self.r[k-1] < self.nx[k]:
                    f = self.flops
                    self.gemv(self.nu[k-1] + self.nx[k-1], rho)
                    self.flops_forward_substitution_implicit_specific += self.flops - f
                self.trsv(rho)
                self.trsv(rho)

            # regularization steps
            f = self.flops
            if not self.implicit and not self.ignore_regularization:
                if self.ng_ineq[k] > 0:
                    self.gemv(nu + nx, self.ng_ineq[k])
                    self.flops += self.ng_ineq[k]
            regularization_flops += self.flops - f

            if k != self.K - 1:
                nxp1 = self.nx[k+1]
                nup1 = self.nu[k+1]
                gammamrhop1 = self.gamma[k+1] - self.rho[k+1]

                self.gemv(nu + nx, nxp1)
                self.gemv(nxp1, nxp1)
                self.gemv(gammamrho, nxp1)
                if self.implicit and not self.ignore_FuFx_GuGx:
                    f = self.flops
                    self.gemv(nu + nx, nxp1)
                    self.flops_forward_substitution_implicit_specific += self.flops - f

        self.flops_forward_substitution = self.flops - flops_at_start


    def preprocess(self):
        flops_at_start = self.flops

        regularization_flops = 0

        if not self.implicit:
            self.flops += 0
            self.flops_preprocess = self.flops - flops_at_start
            return
        
        # regularization
        if not self.ignore_regularization:
            for k in range(self.K):
                f = self.flops
                # D_s
                if self.ng_ineq[k] > 0:
                    for i in range(self.ng_ineq[k]):
                        self.colsc(self.nu[k] + self.nx[k] + 1)
                        self.flops += self.nu[k] + self.nx[k] + 1
                    self.syrk(self.nu[k] + self.nx[k] + 1, self.nu[k] + self.nx[k], self.ng_ineq[k])
                    self.gemv(self.nu[k] + self.nx[k], self.ng_ineq[k]) # REDUNDANT!

                # D_x
                self.flops += self.nu[k] + self.nx[k]
                regularization_flops += self.flops - f

        # Preprocess loop
        for k in range(self.K-1):
            nx = self.nx[k]
            nu = self.nu[k]
            nx_next = self.nx[k+1]
            nu_next = self.nu[k+1]
            r = self.r[k]

            # decomposition
            self.lu_fact(nx_next, nx_next + nu + nx + 1, r)
            self.trsm(nx_next + nu + nx + 1, nx_next)
            self.trsm(nx_next + nu + nx + 1, r)

            # RSQrqt
            self.gemm(nx_next - r, nu_next + nx_next, r)
            self.gemm(nu_next + nx_next + 1, nx_next - r, r)
            
            # FuFx
            if not self.ignore_FuFx_GuGx:
                self.gemm(nu + nx, nx_next - r, r)

            if (k < self.K-2):
                nx_next_next = self.nx[k+2]
                self.gemm(nx_next - r, nx_next_next, r)
                if not self.ignore_FuFx_GuGx:
                    self.gemm(nx_next - r, nx_next_next, r)

            # equality constraints
            self.gemm(nx_next - r, self.ng[k+1], r)

        # modify rhs
        #   only memory operations, no flops
        self.flops_preprocess = self.flops - flops_at_start

    def postprocess(self):
        flops_at_start = self.flops
        if not self.implicit:
            self.flops += 0
            self.flops_postprocess = self.flops - flops_at_start
            return
        
        # postprocess loop
        for k in range(self.K):
            # copies

            if k > 0:
                self.gemv(self.nx[k] - self.r[k-1], self.nx[k])

                self.trsv(self.r[k-1])
                self.trsv(self.nx[k])
            
        # constraint regularization (inequalities)
        if not self.ignore_regularization:
            for k in range(self.K):
                self.gemv(self.nu[k] + self.nx[k], self.ng_ineq[k])
                self.flops += self.ng_ineq[k]

        self.flops_postprocess = self.flops - flops_at_start
        
    def colsc(self, n):
        self.flops += n

    def gead(self, m, n):
        self.flops += 2 * m * n

    def gemm(self, m, n, k):
        self.flops += m * n * (2 * k + 2)
    
    def gemv(self, m, n):
        if n == 0:
            return
        self.gemm(m, 1, n)

    def trsm(self, m, n):
        self.flops += m * n**2
    
    def trsv(self, m):
        self.trsm(m, 1)

    def potrf(self, m, n):
        self.flops += (m-n)*n**2 + n*(n-1)*(n+1)/6 + (n-1)*(n-2)*(n+1)/6 + n*(n-1)/2 + n*(n-1)/2 + n

    def syrk(self, m, n, k):
        self.flops += (2*k+2)*n*(2*m-n+1)/2

    def lu_fact(self, m, n, r):
        self.flops += 2*r*(m-1)*(n-1) + r*(m-1) + r*(r-1)*(2*r-1)/3 - r*(2*m+2*n-3)*(r-1)/2

    def lu_fact_block(self, m1, n1, r1, m2, n2, r2):
        self.lu_fact(m1, n1, r1)
        self.lu_fact(m2, n2, r2)
        self.flops += m2*min(m1,n1)**2

class FlopCountAnalyzer():
    def __init__(self):
        self.fc = FlopCounter()

    def NoStatesNoConstraintsCase(self):
        self.fc.set_dimensions(K=10, nu=10, nx=0, r=0, ng=0, ng_ineq=0)

        flop_impl = self.fc.get_flop_implicit()
        flop_impl_detailed = self.fc.get_detailed_flops()
        flop_expl = self.fc.get_flop_explicit()
        flop_expl_detailed = self.fc.get_detailed_flops()
        flop_reform = self.fc.get_flop_reformulated()
        flop_reform_detailed = self.fc.get_detailed_flops()

        print(f"Implicit:\n{flop_impl}\n{flop_impl_detailed}")
        print(f"Reformulated:\n{flop_reform}\n{flop_reform_detailed}")
        print(f"Explicit:\n{flop_expl}\n{flop_expl_detailed}")

    def show_relative_flop_counts(self):
        data = {
            'K': [], 'nu': [], 'nx': [], 'r': [], 'ng': [], 'ng_ineq': []
        }
        max_val = 10
        step = 2
        for k in range(1, max_val, step):
            for nu in range(1, max_val, step):
                for nx in range(1, max_val, step):
                    for ng in range(0, max_val, step):
                        for ng_ineq in range(0, max_val, step):
                            if ng <= nu + nx:
                                data['K'].append(k)
                                data['nu'].append(nu)
                                data['nx'].append(nx)
                                data['r'].append(nx)
                                data['ng'].append(ng)
                                data['ng_ineq'].append(ng_ineq)
        print(len(data['K']))
        print(f"data prepared")
        impl_flop, expl_flop, reform_flop = self.fc.get_all_flops(data)
        print(f"Flop counts computed")

        metrics = ['K', 'nx', 'nu', 'ng', 'ng_ineq']

        color_explicit = 'r'
        color_implicit = 'b'
        color_reformulated = 'g'

        fig, axs = plt.subplots(2, 3)
        for i, metric in enumerate(metrics):
            ax = axs[i//3, i%3]
            unique_metric = np.unique(data[metric])
            unique_sorted_metric = np.sort(unique_metric)

            # for each method, compute mean and std values for every metric value
            impl_flop_means = []; reform_flop_means = []; expl_flop_means = []

            for val in unique_sorted_metric:
                mask = data[metric] == val
                impl_flop_means.append(np.mean(impl_flop[mask]))
                reform_flop_means.append(np.mean(reform_flop[mask]))
                expl_flop_means.append(np.mean(expl_flop[mask]))

            relative_flop = (np.array(impl_flop_means) - np.array(reform_flop_means)) / np.array(reform_flop_means)
            relative_flop_expl = (np.array(expl_flop_means) - np.array(reform_flop_means)) / np.array(reform_flop_means)
            ax.plot(unique_sorted_metric, relative_flop, '-', alpha=0.5, label='FLOP estimate', color=color_implicit)
            ax.plot(unique_sorted_metric, relative_flop_expl, '-', alpha=0.5, label='FLOP estimate', color=color_explicit)

            ax.axhline(0, color='k', linestyle='-')
            ax.set_ylabel('Relative\ndifference')

            ax.set_xlabel(metric)
            # plt.title(f'Scaling of Times with {metric}')
            ax.grid()
            plt.tight_layout()

        # remove the last axis (bottom right) and put a legend there
        axs[1, 2].axis('off')
        handles, labels = [], []
        for ax in axs.flatten():
            h, l = ax.get_legend_handles_labels()
            handles.extend(h)
            labels.extend(l)
        unique_labels = []
        unique_handles = []
        for h, l in zip(handles, labels):
            if l not in unique_labels:
                unique_labels.append(l)
                unique_handles.append(h)
        axs[1, 2].legend(unique_handles, unique_labels, loc='center')
        
        plt.show()

    def show_flop_decomposition(self):
        self.fc.set_dimensions(K=10, nu=10, nx=10, r=10, ng=10, ng_ineq=10)
        flop_impl = self.fc.get_flop_implicit()
        flop_impl_detailed = self.fc.get_detailed_flops()

        flops = [
            flop_impl_detailed['preprocess'],
            flop_impl_detailed['backward_recursion'] - self.fc.flops_backward_recurion_implicit_specific,
            self.fc.flops_backward_recurion_implicit_specific,
            flop_impl_detailed['initial_stage'],
            flop_impl_detailed['forward_substitution'] - self.fc.flops_forward_substitution_implicit_specific,
            self.fc.flops_forward_substitution_implicit_specific,
            flop_impl_detailed['postprocess']
        ]
        names = [
            'Preprocess',
            'Backward Recursion',
            'Backward Recursion (extra)',
            'Initial Stage',
            'Forward Substitution',
            'Forward Substitution (extra)',
            'Postprocess'
        ]

        plt.figure(figsize=(10, 6))
        # make pie chart
        plt.pie(flops, labels=names, autopct='%1.1f%%')
        plt.title('FLOP decomposition for implicit method')
        plt.show()

    def block_LU_scaling(self):
        # vary ng, nx, nu
        data = {
            'nu': [], 'nx': [], 'ng': [], 'flops_full': [], 'flops_block': []
        }

        nb_runs = 100000
        for i in range(nb_runs):
            nu = np.random.randint(0, 20)
            nx = np.random.randint(0, 20)
            ng = np.random.randint(0, 20)

            self.fc.reset_flops()
            self.fc.lu_fact(ng + nx, nu + nx, min(ng + nx, nu + nx))
            flops_full = self.fc.flops

            self.fc.reset_flops()
            self.fc.lu_fact_block(ng, nu, min(ng, nu), nx, nx, nx)
            flops_block = self.fc.flops

            data['nu'].append(nu)
            data['nx'].append(nx)
            data['ng'].append(ng)
            data['flops_full'].append(flops_full)
            data['flops_block'].append(flops_block)

        # 1d visuals
        metrics = ['nu', 'nx', 'ng']
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            unique_metric = np.unique(data[metric])
            unique_sorted_metric = np.sort(unique_metric)

            flops_full_means = []
            flops_block_means = []
            for val in unique_sorted_metric:
                mask = np.array(data[metric]) == val
                flops_full_means.append(np.mean(np.array(data['flops_full'])[mask]))
                flops_block_means.append(np.mean(np.array(data['flops_block'])[mask]))

            plt.plot(unique_sorted_metric, flops_full_means, label='Full LU')
            plt.plot(unique_sorted_metric, flops_block_means, label='Block LU')
            plt.xlabel(metric)
            plt.ylabel('FLOPs')
            plt.title(f'FLOP scaling with {metric}')
            plt.legend()
            plt.grid()
        plt.show()

        # 2d visuals
        for i in range(len(metrics)):
            for j in range(i+1,len(metrics)):
                metric_x = metrics[i]
                metric_y = metrics[j]

                plt.figure(figsize=(10, 6))
                unique_metric_x = np.unique(data[metric_x])
                unique_metric_y = np.unique(data[metric_y])
                flops_full_means = np.zeros((len(unique_metric_x), len(unique_metric_y)))
                flops_block_means = np.zeros((len(unique_metric_x), len(unique_metric_y)))
                for ix, val_x in enumerate(unique_metric_x):
                    for iy, val_y in enumerate(unique_metric_y):
                        mask = (np.array(data[metric_x]) == val_x) & (np.array(data[metric_y]) == val_y)
                        flops_full_means[ix, iy] = np.mean(np.array(data['flops_full'])[mask])
                        flops_block_means[ix, iy] = np.mean(np.array(data['flops_block'])[mask])
                X, Y = np.meshgrid(unique_metric_y, unique_metric_x)
                plt.contourf(X, Y, (flops_block_means - flops_full_means)/flops_full_means, levels=20, cmap='viridis')
                plt.colorbar(label='FLOPs (Full LU)')
                plt.xlabel(metric_y)
                plt.ylabel(metric_x)
                plt.title(f'FLOP scaling with {metric_x} and {metric_y} (Full LU)')
                plt.grid()
        plt.show()




if __name__ == "__main__":
    analyzer = FlopCountAnalyzer()
    # analyzer.NoStatesNoConstraintsCase()

    # analyzer.fc.ignore_FuFx_GuGx = True
    # analyzer.fc.ignore_regularization = False
    # analyzer.show_relative_flop_counts()

    # analyzer.show_flop_decomposition()

    analyzer.block_LU_scaling()


