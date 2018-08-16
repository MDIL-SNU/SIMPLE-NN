import numpy as np

class L_BFGS(object):
    def __init__(self, num_memlimit=50):
        """
        self.num_memlimit:  
        self.s           : list of s_k = x_(k+1) - x_k (size <= self.num_memlimit)
        self.dg          : list of dg_k = g_(k+1) - g_k (size <= self.num_memlimit)
        self.rho         : list of rho_k = 1/matmul(dg_k^T, s_k)
        """
        self.num_memlimit = num_memlimit
        self.s   = list()
        self.dg  = list()
        self.rho = list()

    def find_direction(self, g_k):
        """
        g_k: gradients of kth step
        s: list of s_k = x_(k+1) - x_k (size <= m)
        dg: list of y_k = g_(k+1) - g_k (size <= m)
        """
        q = np.copy(g_k)
        alpha = np.zeros([len(self.dg)])
        beta  = np.zeros([len(self.dg)])
 
        for i in np.arange(len(self.dg))[::-1]:
            alpha[i] = self.rho[i] * np.matmul(self.s[i].transpose(), q)
            q -= alpha[i]*self.dg[i]
 
        def is_pos_def(x):
            return np.all(np.linalg.eigvals(x) > 0)
 
        #H_0_k = np.matmul(y[-1], s[-1].transpose()) / np.matmul(y[-1].transpose(), y[-1])
        H_0_k = np.identity(q.shape[0])
        #print(is_pos_def(H_0_k))
        z = np.matmul(H_0_k, q)
 
        for i in range(len(self.dg)):
            beta[i] = self.rho[i] * np.matmul(self.dg[i].transpose(), z)
            z += self.s[i] * (alpha[i] - beta[i])
 
        return z

    def initialize_line_search(self, c_1=10.e-4, c_2=0.9):
        self.mu = 0.
        self.nu = float('inf')
        self.step = 1.
        self.c_1 = c_1
        self.c_2 = c_2
        self.ls_idx = 0

    def wolfe_line_search_iter(self, zero_vals, alpha_vals, z):
        tag_rerun = True

        step_grad_zero  = np.sum(zero_vals[0]*-z)
        step_grad_alpha = np.sum(alpha_vals[0]*-z)

        if alpha_vals[1] > zero_vals[1] + self.step*self.c_1*step_grad_zero: # Armijo fails
            self.nu = self.step
        elif step_grad_alpha < self.c_2*step_grad_zero: # Weak Wolfe condition fails
            self.mu = self.step
        else:
            self.ls_idx = 0
            tag_rerun = False
            return tag_rerun

        if self.nu < float('inf'):
            self.step = (self.mu + self.nu) / 2.
        else:
            self.step = 2. * self.step

        self.ls_idx += 1
        if self.ls_idx > 100:
            self.ls_idx = 0
            tag_rerun = False
        
        return tag_rerun

    def update_lists(self, dg, z):
        self.s.append(-z*self.step)
        self.dg.append(dg)
        self.rho.append(1./np.matmul(self.dg[-1].transpose(), self.s[-1]))

        if len(self.s) > self.num_memlimit:
            self.s.pop(0)
            self.dg.pop(0)
            self.rho.pop(0)
