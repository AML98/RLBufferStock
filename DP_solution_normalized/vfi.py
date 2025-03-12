import numpy as np
from numba import njit, prange

# consav
from consav import linear_interp  # for 1D interpolation
from consav import golden_section_search

from . import utility

############################################
# a. define the objective function for VFI #
############################################
@njit
def obj_bellman(c_choice, m, v_next, par):

    # 1. end-of-period assets
    a = m - c_choice

    # 2. expected future value
    w = 0.0
    for ishock in range(par.Nshocks):
        
        # i. shocks
        psi = par.psi[ishock]
        psi_w = par.psi_w[ishock]
        xi = par.xi[ishock]
        xi_w = par.xi_w[ishock]

        weight = psi_w * xi_w

        # ii. next-period normalized m
        m_plus = (par.R / psi) * a + xi

        # iii. interpolate next-period value
        v_plus = linear_interp.interp_1d(par.grid_m, v_next, m_plus)

        # iv. accumulate expected discounted value
        w += weight * par.beta * v_plus
    
    # 3. instantaneous utility
    util = utility.func(c_choice, par)

    # 4. total (current + discounted future)
    value_of_choice = util + w

    return -value_of_choice


#########################################
# b. solve the bellman equation via VFI #
#########################################
@njit(parallel=True)
def solve_bellman(t, sol, par):

    # unpack
    c_now = sol.c[t]
    v_now = sol.v[t]
    v_next = sol.v[t+1]

    tol = par.tol

    for im in prange(par.Nm):

        # 1. current normalized cash-on-hand
        m_val = par.grid_m[im]

        # 2. set search bounds for consumption
        c_low = np.fmin(m_val/2,1e-8)  
        c_high = m_val

        # 3. find the *c* that maximizes total value
        c_opt = golden_section_search.optimizer(
            obj_bellman,
            c_low,
            c_high,
            args=(m_val, v_next, par),
            tol=tol
        )

        # 4. store optimal consumption
        c_now[im] = c_opt

        # 5. compute the optimal value
        v_now[im] = -obj_bellman(c_opt, m_val, v_next, par)
