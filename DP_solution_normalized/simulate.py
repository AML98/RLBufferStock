import numpy as np
from numba import njit, prange

 # consav
from consav import linear_interp # for linear interpolation

from . import utility

@njit(parallel=True)
def lifecycle(sim,sol,par):
    """ simulate full life-cycle """

    # unpack (to help numba optimize)
    p = sim.p
    m = sim.m
    c = sim.c
    a = sim.a
    u = sim.u
    v = sim.v
    
    for t in range(par.simT):
        for i in prange(par.simN): # in parallel
            
            # a. beginning of period states
            if t == 0:
                # p[t,i] = 10
                m[t,i] = 1
            else:
                # p[t,i] = sim.psi[t,i]*p[t-1,i]
                m[t,i] = par.R*a[t-1,i]/sim.psi[t,i] + sim.xi[t,i]

            # b. choices
            c[t,i] = linear_interp.interp_1d(par.grid_m,sol.c[t],m[t,i])
            a[t,i] = m[t,i]-c[t,i]
            u[t,i] = utility.func(c[t,i],par)

    for i in range(par.simN):
        v[i] = np.sum((par.beta**np.arange(par.T))*u[:,i])
            