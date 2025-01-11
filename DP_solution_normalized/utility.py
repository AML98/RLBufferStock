from numba import njit
import numpy as np

@njit
def func(c,par):
    if par.rho == 1.0:
        return np.log(c + 1e-8)
    return c**(1-par.rho)/(1-par.rho)

@njit
def marg_func(c,par):
    return c**(-par.rho)

@njit
def inv_marg_func(q,par):
    return q**(-1/par.rho)