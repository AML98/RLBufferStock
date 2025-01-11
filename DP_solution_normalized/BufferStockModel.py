# -*- coding: utf-8 -*-
"""BufferStockModel (Normalized Version)

Solves the Deaton-Carroll buffer-stock consumption model in normalized form:
    m_t = M_t / p_t
    a_t = A_t / p_t
with either:
A. vfi
B. nvfi
C. egm
"""

##############
# 1. imports #
##############

import time
import numpy as np

# consav package
from consav import ModelClass, jit
from consav.grids import nonlinspace
from consav.quadrature import create_PT_shocks
from consav.misc import elapsed

# local modules
from . import utility
from . import last_period
from . import post_decision
from . import vfi
from . import nvfi
from . import egm
from . import simulate
from . import figs

############
# 2. model #
############

class BufferStockModelClass(ModelClass):
    
    #########
    # setup #
    #########
    
    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = []
        
        # b. other attributes
        self.other_attrs = []
        
        # c. savefolder
        self.savefolder = 'saved'
        
        # d. list not-floats for safe type inference
        # NOTE: Removed 'Np' because we no longer have a p-grid
        self.not_floats = [
            'T','Npsi','Nxi','Nm','Na',
            'do_print','do_simple_w','simT','simN','sim_seed',
            'cppthreads','Nshocks'
        ]

        # e. cpp
        self.cpp_filename = 'cppfuncs/egm.cpp'
        self.cpp_options = {'compiler':'vs'}
        
    def setup(self):
        """ set baseline parameters """   

        par = self.par

        # a. solution method
        par.solmethod = 'nvfi'
        
        # b. horizon
        par.T = 20
        
        # c. preferences
        par.beta = 0.96
        par.rho = 2.0

        # d. returns and income
        par.R = 1.03
        par.sigma_psi = 0.1
        par.Npsi = 6
        par.sigma_xi = 0.1
        par.Nxi = 6
        par.pi = 0.1
        par.mu = 0.5
        
        # e. grids (removed Np; we only keep Nm and Na)
        par.Nm = 600
        par.Na = 800

        # f. misc
        par.tol = 1e-8
        par.do_print = True
        par.do_simple_w = False
        par.cppthreads = 1

        # g. simulation
        par.simT = par.T
        par.simN = 10000
        par.sim_seed = 1999
        
    def allocate(self):
        """ allocate model, i.e. create grids and allocate solution and simulation arrays """

        self.create_grids()
        self.solve_prep()
        self.simulate_prep()

    def create_grids(self):
        """ construct grids for states and shocks """

        par = self.par

        # a. normalized states
        #    we interpret grid_m as the possible values of m_t ( = M_t / p_t )
        par.grid_m = nonlinspace(1e-6,20,par.Nm,1.1)

        # b. post-decision states (still length Na)
        #    similarly interpret grid_a as a_t ( = A_t / p_t )
        par.grid_a = nonlinspace(1e-6,20,par.Na,1.1)
        
        # c. shocks
        #    same transitory+permanent shock structure, but eventually used in normalized transitions
        shocks = create_PT_shocks(
            par.sigma_psi,par.Npsi,par.sigma_xi,par.Nxi,
            par.pi,par.mu)
        par.psi,par.psi_w,par.xi,par.xi_w,par.Nshocks = shocks

        # d. set seed
        np.random.seed(self.par.sim_seed)

    def checksum(self):
        """ print checksum """
        # c is now shape (T,Nm), so we might do e.g.:
        return np.mean(self.sol.c[0])

    #########
    # solve #
    #########

    def solve_prep(self):
        """ allocate memory for solution """

        par = self.par
        sol = self.sol

        # c(t,m) and v(t,m) are now 2D with shape (T, Nm) [no dimension for p]
        sol.c = np.nan*np.ones((par.T, par.Nm))
        sol.v = np.nan*np.zeros((par.T, par.Nm))

        # post-decision objects w(a) and q(a) are now 1D with shape (Na)
        sol.w = np.nan*np.zeros(par.Na)
        sol.q = np.nan*np.zeros(par.Na)

    def solve(self):
        """ solve the model using solmethod """

        with jit(self) as model: 

            par = model.par
            sol = model.sol

            # backwards induction
            for t in reversed(range(par.T)):
                
                t0 = time.time()
                
                # a. last period
                if t == par.T-1:
                    
                    # we will need a last_period.solve() that 
                    # sets c(t,m) = m, for instance (or some variation).
                    last_period.solve(t,sol,par)

                # b. all other periods
                else:
                    
                    # i. compute post-decision functions
                    t0_w = time.time()

                    compute_w, compute_q = False, False
                    if par.solmethod in ['nvfi']: 
                        compute_w = True
                    elif par.solmethod in ['egm']: 
                        compute_q = True

                    if compute_w or compute_q:

                        if par.do_simple_w:
                            post_decision.compute_wq_simple(t,sol,par,compute_w=compute_w,compute_q=compute_q)
                        else:
                            post_decision.compute_wq(t,sol,par,compute_w=compute_w,compute_q=compute_q)

                    t1_w = time.time()

                    # ii. solve bellman equation
                    if par.solmethod == 'vfi':
                        vfi.solve_bellman(t,sol,par)
                    elif par.solmethod == 'nvfi':
                        nvfi.solve_bellman(t,sol,par)
                    elif par.solmethod == 'egm':
                        egm.solve_bellman(t,sol,par)
                    else:
                        raise ValueError(f'unknown solution method, {par.solmethod}')

                # c. print
                if par.do_print:
                    msg = f' t = {t} solved in {elapsed(t0)}'
                    if t < par.T-1:
                        msg += f' (w: {elapsed(t0_w,t1_w)})'
                    print(msg)

    def solve_cpp(self):
        """ solve the model using egm in C++ """

        par = self.par
        sol = self.sol

        t0 = time.time()
       
        if par.solmethod in ['egm']:
            self.cpp.solve(par,sol)
        else:
            raise ValueError(f'unknown cpp solution method, {par.solmethod}')            
        
        t1 = time.time()

        return t0,t1

    ############
    # simulate #
    ############

    def simulate_prep(self):
        """ allocate memory for simulation """

        par = self.par
        sim = self.sim

        # In normalized form, we store (m_t, a_t, c_t) for each sim
        sim.m = np.nan*np.zeros((par.simT,par.simN))
        sim.c = np.nan*np.zeros((par.simT,par.simN))
        sim.a = np.nan*np.zeros((par.simT,par.simN))

        # If you also track p_t in simulation, add sim.p as well:
        sim.p = np.nan*np.zeros((par.simT,par.simN))

        # b. draw random shocks
        sim.psi = np.ones((par.simT,par.simN))
        sim.xi = np.ones((par.simT,par.simN))

        sim.u = np.nan*np.zeros((par.simT, par.simN))
        sim.v = np.nan*np.zeros(par.simN)

    def simulate(self):
        """ simulate model """

        with jit(self) as model: 

            par = model.par
            sol = model.sol
            sim = model.sim
            
            t0 = time.time()

            # a. pick shocks
            I = np.random.choice(par.Nshocks,
                size=(par.simT,par.simN), 
                p=par.psi_w*par.xi_w)

            sim.psi[:] = par.psi[I]
            sim.xi[:] = par.xi[I]

            # b. run your custom simulate.lifecycle or simulate.infinite, 
            #    but be sure it updates and uses normalized states.
            simulate.lifecycle(sim,sol,par)

        if par.do_print:
            print(f'model simulated in {elapsed(t0)}')

    ########
    # figs #
    ########

    def consumption_function(self,t=0):
        # now c[t,:] is purely c(m) in normalized form
        figs.consumption_function(self,t)

    def consumption_function_interact(self):
        figs.consumption_function_interact(self)
          
    def lifecycle(self):
        figs.lifecycle(self)

    def plot_multiple_consumption_functions(self,t_list):
        figs.plot_multiple_consumption_functions(self,t_list)
