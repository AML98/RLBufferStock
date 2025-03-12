nice_blue = 'midnightblue'
nice_red = 'firebrick'

# -------------------
# - Fig 1: Baseline -
# -------------------

dp_fig1 = {
    'label':'DP household',
    'history':None,
    'color':nice_red,
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
    'linestyle':'--',
    'linewidth':2.0,
    'linestart':0,
    'alpha':1.0,
    'cf_start':0
}

rl_fig1_ = {
    'label':'RL household',
    'history':None,
    'color':nice_blue,
    'linestyle':'--',
    'linewidth':2.0,
    'linestart':0,
    'alpha':1.0,
    'cf_start':0
}

# -------------------
# - Fig 2: DP vs RL -
# -------------------

rl_shock_fig2 = {
    'label':'RL household',
    'history':None,
    'color':nice_blue,
    'linestyle':'--',
    'linewidth':2.0,
    'linestart':0,
    'alpha':1.0,
    'cf_start':0
}

rl_fig2 = {
    'label':'RL household (baseline)',
    'history':None,
    'color':nice_blue,
    'linestyle':':',
    'linewidth':2.0,
    'linestart':0,
    'alpha':0.8,
    'cf_start':None
}

dp_shock_fig2 = {
    'label':'DP household',
    'history':None,
    'color':nice_red,
    'linestyle':'-',
    'linewidth':2.0,
    'linestart':0,
    'alpha':1.0,
    'cf_start':0
}

dp_fig2 = {
    'label':'DP household (baseline)',
    'history':None,
    'color':nice_red,
    'linestyle':':',
    'linewidth':2.0,
    'linestart':0,
    'alpha':0.8,
    'cf_start':None
}

# ---------------
# - Policy plot -
# ---------------

dp_policy_fig = {
    'agent':None,
    'label':'DP household',
    'color':nice_red,
    'linestyle':'-',
    'linewidth':2.0,
    'alpha':1.0
}

rl_policy_fig = {
    'agent':None,
    'label':'RL household',
    'color':nice_blue,
    'linestyle':'--',
    'linewidth':2.0,
    'alpha':1.0
}

dp_policy_base_fig = {
    'agent':None,
    'label':'DP household (baseline)',
    'color':nice_red,
    'linestyle':':',
    'linewidth':2.0,
    'alpha':0.8
}

# -----------------
# - Learning plot -
# -----------------

rl1_learning_fig = {
    'a_loss':None,
    'c_loss':None,
    'value':None,
    'm_dist':None,
    'euler_error':None,
    'label':None,
    'color':nice_blue,
    'linewidth':2.0,
}

rl2_learning_fig = {
    'a_loss':None,
    'c_loss':None,
    'value':None,
    'm_dist':None,
    'euler_error':None,
    'label':None,
    'color':'green',
    'linewidth':2.0,
}

rl3_learning_fig = {
    'a_loss':None,
    'c_loss':None,
    'value':None,
    'm_dist':None,
    'euler_error':None,
    'label':None,
    'color':'darkorange',
    'linewidth':2.0,
}

dp_learning_fig = {
    'value':None,
    'label':'DP value',
    'color':nice_red,
    'linestyle':'--',
    'linewidth':2.0,
    'alpha':2.0
}

# LAST PLOT ...

hyp_dp_policy = {
    'agent':None,
    'label':'DP',
    'color':nice_red,
    'linestyle':'-',
    'linewidth':2.0,
    'alpha':1.0
}

hyp_rl1_policy = {
    'agent':None,
    'label':None,
    'color':nice_blue,
    'linestyle':'--',
    'linewidth':2.0,
    'alpha':1.0
}

hyp_rl2_policy = {
    'agent':None,
    'label':None,
    'color':'green',
    'linestyle':'--',
    'linewidth':2.0,
    'alpha':1.0
}

hyp_rl3_policy = {
    'agent':None,
    'label':None,
    'color':'darkorange',
    'linestyle':'--',
    'linewidth':2.0,
    'alpha':1.0
}

hyp_dp = {
    'label':'DP',
    'history':None,
    'color':nice_red,
    'linestyle':'-',
    'linewidth':2.0,
    'linestart':0,
    'alpha':1.0,
    'cf_start':None
}

hyp_rl1 = {
    'label':None,
    'history':None,
    'color':nice_blue,
    'linestyle':'--',
    'linewidth':2.0,
    'linestart':0,
    'alpha':1.0,
    'cf_start':None
}

hyp_rl2 = {
    'label':None,
    'history':None,
    'color':'green',
    'linestyle':'--',
    'linewidth':2.0,
    'linestart':0,
    'alpha':1.0,
    'cf_start':None
}

hyp_rl3 = {
    'label':None,
    'history':None,
    'color':'darkorange',
    'linestyle':'--',
    'linewidth':2.0,
    'linestart':0,
    'alpha':1.0,
    'cf_start':None
}