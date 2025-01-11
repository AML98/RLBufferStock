import numpy as np
from numba import njit, prange

# consav
from consav import linear_interp # for 1D interpolation

# local modules
from . import utility

@njit(parallel=True)
def solve_bellman(t, sol, par):
    """
    Solve the Bellman equation using the endogenous grid method (EGM)
    for the *normalized* model.
    
    We assume:
      - sol.q has shape (Na,), representing the post-decision marginal value
      - sol.c[t] has shape (Nm,), the consumption function on grid_m
      - par.grid_a has length Na
      - par.grid_m has length Nm
    """

    # 1. Unpack solution arrays
    c_now = sol.c[t]   # shape (Nm,)
    q_post = sol.q     # shape (Na,) post-decision marginal utilities

    # 2. Temporary containers for the "reverse" mapping
    #    We'll create a plus-one size to include a boundary point at 0
    #    if desired (not strictly necessary).
    m_temp = np.zeros(par.Na + 1)
    c_temp = np.zeros(par.Na + 1)

    # 3. Invert the Euler equation to find c(a) and then compute m(a) = a + c(a)
    #    Loop over each post-decision asset index ia
    for ia in prange(par.Na):
        # a) consumption from inverse marginal utility
        c_temp[ia+1] = utility.inv_marg_func(q_post[ia], par)
        # b) implied 'm' from the budget identity m = a + c
        m_temp[ia+1] = par.grid_a[ia] + c_temp[ia+1]

    # (Optional) set m_temp[0] = 0 and c_temp[0] = 0 if you want a boundary
    # m_temp[0] = 0.0
    # c_temp[0] = 0.0

    # 4. Interpolate c_temp onto the common grid_m
    #    This gives c_now(m) for each point in par.grid_m.
    #    Use a monotone (sorted) interpolation:
    linear_interp.interp_1d_vec_mon_noprep(m_temp, c_temp, par.grid_m, c_now)
