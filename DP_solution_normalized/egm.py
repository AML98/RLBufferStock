import numpy as np
from numba import njit, prange

# consav
from consav import linear_interp # for 1D interpolation

# local modules
from . import utility

@njit(parallel=True)
def solve_bellman(t, sol, par):

    c_now = sol.c[t]
    q_post = sol.q

    m_temp = np.zeros(par.Na + 1)
    c_temp = np.zeros(par.Na + 1)

    for ia in prange(par.Na):
        # a. consumption from inverse marginal utility
        c_temp[ia+1] = utility.inv_marg_func(q_post[ia], par)
        # b. implied m from the budget identity m = a + c
        m_temp[ia+1] = par.grid_a[ia] + c_temp[ia+1]

    linear_interp.interp_1d_vec_mon_noprep(m_temp, c_temp, par.grid_m, c_now)
