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

    v = sol.v[t]
    c = sol.c[t]

    for im in prange(par.Nm):
        m_val = par.grid_m[im]
        c[im] = m_val
        v[im] = utility.func(c[im], par)
