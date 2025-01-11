# last_period.py

from numba import njit, prange
from . import utility

@njit(parallel=True)
def solve(t, sol, par):
    """
    Solve the problem in the last period (normalized version).

    Key idea: in the last period, the agent just consumes all cash on hand,
    i.e., c_t = m_t (since there's no future).
    """

    v = sol.v[t]  # shape (Nm,)
    c = sol.c[t]  # shape (Nm,)

    for im in prange(par.Nm):  # parallel loop over the m-grid
        m_val = par.grid_m[im]

        # a) consume everything
        c[im] = m_val

        # b) utility
        v[im] = utility.func(c[im], par)
