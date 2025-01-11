import matplotlib.pyplot as plt
import numpy as np

nice_blue = 'blue'

# -------------------
# - Fig 1: Baseline -
# -------------------

dp_fig1 = {
    'label':'DP household',
    'history':None,
    'color':'red',
    'linestyle':'-',
    'linewidth':2.0,
    'linestart':0,
    'alpha':1.0,
    'cf_start':0
}

rl_fig1 = {
    'label':'RL household',
    'history':None,
    'color':nice_blue,
    'linestyle':'-',
    'linewidth':2.0,
    'linestart':0,
    'alpha':1.0,
    'cf_start':0
}

# -------------------
# - Fig 2: DP vs RL -
# -------------------

rl_shock_fig2 = {
    'label':'RL household (incl. shock)',
    'history':None,
    'color':nice_blue,
    'linestyle':'-',
    'linewidth':2.0,
    'linestart':0,
    'alpha':1.0,
    'cf_start':None
}

rl_fig2 = {
    'label':'RL household (baseline)',
    'history':None,
    'color':nice_blue,
    'linestyle':':',
    'linewidth':2.0,
    'linestart':0,
    'alpha':1.0,
    'cf_start':None
}

dp_shock_fig2 = {
    'label':'DP household (incl. shock)',
    'history':None,
    'color':'red',
    'linestyle':'-',
    'linewidth':2.0,
    'linestart':0,
    'alpha':1.0,
    'cf_start':None
}

dp_fig2 = {
    'label':'DP household (baseline)',
    'history':None,
    'color':'red',
    'linestyle':':',
    'linewidth':2.0,
    'linestart':0,
    'alpha':1.0,
    'cf_start':None
}