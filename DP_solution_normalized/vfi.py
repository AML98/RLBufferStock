import numpy as np
from numba import njit, prange

# consav
from consav import linear_interp  # for 1D interpolation
from consav import golden_section_search

from . import utility

##############################################
# a. define the objective function for VFI
##############################################
@njit
def obj_bellman(c_choice, m, v_next, par):
    """
    Evaluate the *negative* of the Bellman integrand (because we'll 
    'minimize' via golden_section_search).

    c_choice : consumption in current period (normalized)
    m        : normalized cash-on-hand in current period
    v_next   : next-period value function, v_{t+1}(m'), shape (Nm,)
    par      : parameters
    """

    # 1. End-of-period assets (normalized)
    a = m - c_choice  # must be >= 0 if no borrowing

    # 2. Expected future value
    w = 0.0
    for ishock in range(par.Nshocks):
        
        # i. shocks
        psi = par.psi[ishock]
        psi_w = par.psi_w[ishock]
        xi = par.xi[ishock]
        xi_w = par.xi_w[ishock]

        weight = psi_w * xi_w

        # ii. Next-period normalized m
        #     m_plus = (R / psi)*a + xi
        m_plus = (par.R / psi) * a + xi

        # iii. Interpolate next-period value, v_{t+1}(m_plus)
        v_plus = linear_interp.interp_1d(par.grid_m, v_next, m_plus)

        # iv. Accumulate expected discounted value
        w += weight * par.beta * v_plus
    
    # 3. Instantaneous utility
    util = utility.func(c_choice, par)

    # 4. Total (current + discounted future)
    value_of_choice = util + w

    # We return the *negative* because golden_section_search finds the minimum
    return -value_of_choice


#########################################
# b. solve the bellman equation via VFI
#########################################
@njit(parallel=True)
def solve_bellman(t, sol, par):
    """
    Solve the bellman equation using a direct value-function iteration approach
    with golden-section search for the consumption choice.
    
    sol.c[t,im] and sol.v[t,im] store the solution for each m-gridpoint.
    """

    # Unpack references
    c_now = sol.c[t]  # shape (Nm,)
    v_now = sol.v[t]  # shape (Nm,)
    v_next = sol.v[t+1]  # shape (Nm,)  next-period value function

    # For golden section search
    tol = par.tol

    # Loop over m-grid
    for im in prange(par.Nm):

        # 1. Current normalized cash-on-hand
        m_val = par.grid_m[im]

        # 2. Set search bounds for consumption
        #    - Typically c in [0, m_val]
        #    - If you want to avoid c=0, do c_low = some small positive number
        c_low = np.fmin(m_val/2,1e-8)  
        c_high = m_val  # no borrowing constraint => can't consume more than m

        # 3. Find the *c* that maximizes total value
        c_opt = golden_section_search.optimizer(
            obj_bellman,
            c_low,
            c_high,
            args=(m_val, v_next, par),
            tol=tol
        )

        # 4. Store optimal consumption
        c_now[im] = c_opt

        # 5. Compute the optimal value
        v_now[im] = -obj_bellman(c_opt, m_val, v_next, par)
