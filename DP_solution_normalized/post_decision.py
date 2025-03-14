# post_decision.py

import numpy as np
from numba import njit, prange

# consav
from consav import linear_interp

# local modules
from . import utility

@njit(parallel=True)
def compute_wq(t, sol, par, compute_w=False, compute_q=False):

    w_a = sol.w
    q_a = sol.q

    # next-period value and consumption
    v_next = sol.v[t+1]
    c_next = sol.c[t+1]

    # grids
    grid_a = par.grid_a
    Nm = par.Nm
    Na = par.Na

    # initialize post-decision arrays
    if compute_w:
        for ia in range(Na):
            w_a[ia] = 0.0
    if compute_q:
        for ia in range(Na):
            q_a[ia] = 0.0

    # Loop over post-decision states
    for ia in prange(Na):
        a_val = grid_a[ia]
        
        w_temp = 0.0
        q_temp = 0.0

        # Loop over shocks
        for ishock in range(par.Nshocks):

            psi = par.psi[ishock]
            xi = par.xi[ishock]
            psi_w = par.psi_w[ishock]
            xi_w = par.xi_w[ishock]

            # Probability weight
            weight = psi_w * xi_w

            # Next-period state: m_{t+1} = (R/psi)*a + xi
            m_plus = (par.R/psi)*a_val + xi

            # Interpolate v_{t+1}(m_plus) and c_{t+1}(m_plus) in 1D
            v_plus = linear_interp.interp_1d(par.grid_m, v_next, m_plus)
            c_plus = linear_interp.interp_1d(par.grid_m, c_next, m_plus)

            # accumulate
            if compute_w:
                # w(a) = E[ beta * v_{t+1}(m_{t+1}) ]
                w_temp += weight * par.beta * v_plus
            if compute_q:
                # q(a) = E[ R * beta * u'(c_{t+1}) ]
                marg_u_plus = utility.marg_func(c_plus, par)
                q_temp += weight * par.R * par.beta * marg_u_plus

        # store
        if compute_w:
            w_a[ia] = w_temp
        if compute_q:
            q_a[ia] = q_temp
