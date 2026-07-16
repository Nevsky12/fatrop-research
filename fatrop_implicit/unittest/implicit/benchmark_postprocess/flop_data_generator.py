import numpy as np
import pandas as pd

class FlopDataGenerator:
    def __init__(self, nb_samples=100000):
        self.nb_samples = nb_samples
        self.full_rank = False
        self.ignore_nx_part = False
        self.fixed_size = False
        self.flop_data = self.generate()
        
    def random_int(self, low, high):
        return np.random.randint(low, high) if high > low else low

    def sample(self):
        max_val = 50
        nu = np.zeros((self.nb_samples,), dtype=int)
        nx = np.zeros((self.nb_samples,), dtype=int)
        nz = np.zeros((self.nb_samples,), dtype=int)
        nh = np.zeros((self.nb_samples,), dtype=int)
        nf = np.zeros((self.nb_samples,), dtype=int)
        rho1 = np.zeros((self.nb_samples,), dtype=int)
        rho2 = np.zeros((self.nb_samples,), dtype=int)
        rho = np.zeros((self.nb_samples,), dtype=int)
        
        sample_idx = 0
        while sample_idx < self.nb_samples:
            print(f"Sampling... ({sample_idx/self.nb_samples*100:.2f}%)", end='\r')
            nu_sample = self.random_int(0, max_val)
            nx_sample = self.random_int(0, max_val)
            nz_sample = self.random_int(0, nx_sample)
            nh_sample = self.random_int(0, nu_sample + nx_sample)
            nf_sample = self.random_int(0, max_val)
            
            if self.fixed_size:
                nz_sample = max_val - nu_sample
                nf_sample = max_val - nh_sample
            
            if nh_sample > nu_sample + nx_sample or nf_sample + nh_sample > nu_sample + nz_sample + nx_sample:
                continue
            
            rho1_sample = self.random_int(0, min(nh_sample, nu_sample)) if not self.full_rank else min(nh_sample, nu_sample)
            rho2_sample = self.random_int(0, min(nf_sample, nu_sample + nz_sample - rho1_sample)) if not self.full_rank else min(nf_sample, nu_sample + nz_sample - rho1_sample)
            rho_sample = rho1_sample + rho2_sample
            
            nu[sample_idx] = nu_sample
            nx[sample_idx] = nx_sample
            nz[sample_idx] = nz_sample
            nh[sample_idx] = nh_sample
            nf[sample_idx] = nf_sample
            rho1[sample_idx] = rho1_sample
            rho2[sample_idx] = rho2_sample
            rho[sample_idx] = rho_sample
            
            sample_idx += 1
        print(f"Generated {self.nb_samples} samples.")
            
        total_size = (nu + nz) * (nh + nf)
        relative_size = nz*nh / total_size
        
        if self.ignore_nx_part:
            nx = np.zeros_like(nx)
            
        return nu, nx, nz, nh, nf, rho1, rho2, rho, relative_size, total_size
        
    def generate(self):
        nu, nx, nz, nh, nf, rho1, rho2, rho, relative_size, total_size = self.sample()
        
        print(f"Computing flops for {self.nb_samples} samples...", end='\r')
        flops_normal = self.LU_flops(nh + nf, nu + nz + nx, nu + nz, rho)
        print(f"Computed flops for {self.nb_samples} samples.")
        print(f"Computing structured flops for {self.nb_samples} samples...", end='\r')        
        flops_structure = self.LU_flops_structure(nx, nu, nz, nh, nf, rho1, rho2)
        print(f"Computed structured flops for {self.nb_samples} samples.")
        rel_diff_flops = (flops_structure - flops_normal) / (flops_normal + 1e-10)
        rel_diff_flops = np.clip(rel_diff_flops, -100, 100)
       
        # return panda dataframe
        return pd.DataFrame({
            'nu': nu,
            'nz': nz,
            'nh': nh,
            'nf': nf,
            'rho1': rho1,
            'rho2': rho2,
            'rho': rho,
            'nx': nx,
            'relative_size': relative_size,
            'total_size': total_size,
            'total_size_sqrt': np.sqrt(total_size),
            'flops_normal': flops_normal,
            'flops_structure': flops_structure,
            'rel_diff_flops': rel_diff_flops,
            'm': nh + nf,
            'n': nu + nz
        })
        
    def LU_flops(self, m, n, n_max, rho):       
        flops = 0*rho
        for k in range(len(rho)):
            for i in range(rho[k]):
                flops[k] = flops[k] + 2*(m[k] - i) * (n_max[k] - i) + (m[k] - i - 1) + 2*(m[k] - i - 1) * (n[k] - i - 1)           
        return flops
    
    def LU_flops_structure(self, nx, nu, nz, nh, nf, rho1, rho2):
        flops = 0
        flops += self.LU_flops(nh, nu, nu, rho1)
        flops += rho1**2 * nf + 2*(nu - rho1) * rho1 * nf
        flops += self.LU_flops(nf, nu + nz - rho1, nu + nz - rho1, rho2)
        flops += nx*(rho1 + rho2)**2 + 2*nx*(nf + nh - rho1 - rho2)*(rho1 + rho2)
        
        # flops = self.LU_flops(nh+nf, nu+nz+nx, nu+nz, rho1+rho2)
            
        return flops