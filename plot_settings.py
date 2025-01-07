import matplotlib.pyplot as plt
import numpy as np

nice_blue = 'blue'

# -------------------
# - Fig 0: Baseline -
# -------------------

dp_fig0 = {
    'label':'DP agent',
    'history':None,
    'color':'red',
    'linestyle':'-',
    'linewidth':2.0,
    'linestart':0,
    'alpha':1.0,
    'cf_start':0
}

td3_fig0 = {
    'label':None,
    'history':None,
    'color':nice_blue,
    'linestyle':'-',
    'linewidth':2.0,
    'linestart':0,
    'alpha':1.0,
    'cf_start':0
}

# ---------------------------------
# - Fig 1: Effect of model params -
# ---------------------------------

linestyles_fig1 = ['-','--',':']

td3_fig1 = {
    'label':None,
    'history':None,
    'color':nice_blue,
    'linestyle':None,
    'linewidth':2.0,
    'linestart':0,
    'alpha':1.0,
    'cf_start':None
}

# -------------------
# - Fig 2: DP vs RL -
# -------------------

td3_shock_fig2 = {
    'label':'RL agent',
    'history':None,
    'color':nice_blue,
    'linestyle':'-',
    'linewidth':2.0,
    'linestart':0,
    'alpha':1.0,
    'cf_start':None
}

td3_fig2 = {
    'label':'RL agent (no shock)',
    'history':None,
    'color':nice_blue,
    'linestyle':':',
    'linewidth':1.5,
    'linestart':0,
    'alpha':0.75,
    'cf_start':None
}

dp_shock_fig2 = {
    'label':'DP agent',
    'history':None,
    'color':'red',
    'linestyle':'-',
    'linewidth':2.0,
    'linestart':0,
    'alpha':1.0,
    'cf_start':None
}

dp_fig2 = {
    'label':'DP agent (no shock)',
    'history':None,
    'color':'red',
    'linestyle':':',
    'linewidth':1.5,
    'linestart':0,
    'alpha':0.75,
    'cf_start':None
}

# ---------------------------------
# - Fig 3: Effect of hyper params -
# ---------------------------------

colors_fig3 = [nice_blue, 'green', 'black']

td3_shock_fig3 = {
    'label':None,
    'history':None,
    'color':None,
    'linestyle':'-',
    'linewidth':2.0,
    'linestart':0,
    'alpha':1.0,
    'cf_start':None
}

dp_shock_fig3 = {
    'label':'DP agent',
    'history':None,
    'color':'red',
    'linestyle':'-',
    'linewidth':2.0,
    'linestart':0,
    'alpha':1.0,
    'cf_start':None
}