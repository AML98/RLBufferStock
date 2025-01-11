import numpy as np
import matplotlib.pyplot as plt
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
plt.rcParams.update({
    "axes.grid" : True, 
    "grid.color": "black", 
    "grid.alpha":"0.25", 
    "grid.linestyle": ":"
})
plt.rcParams.update({'font.size': 14})

import ipywidgets as widgets

def consumption_function(model, t):
    """
    Plot consumption c(t,m) as a function of m (1D).
    """

    # unpack
    par = model.par
    sol = model.sol

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    # plot c(t,m) vs. m
    ax.plot(par.grid_m, sol.c[t,:], lw=2, label=f"t = {t}")

    # details
    ax.set_xlabel('m')
    ax.set_ylabel('c')
    ax.set_title(f'Consumption function at t={t}')
    ax.grid(True)
    ax.legend()
    plt.show()

def consumption_function_interact(model):
    """
    Interactive widget to pick t and see the corresponding consumption function.
    """

    widgets.interact(
        consumption_function,
        model=widgets.fixed(model),
        t=widgets.Dropdown(
            description='t', 
            options=list(range(model.par.T)), 
            value=0
        )
    )

def lifecycle(model):
    """
    Plot average paths of (m_t, c_t, a_t) over the lifecycle (or finite T horizon).
    """

    # unpack
    par = model.par
    sim = model.sim

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    simvarlist = [
        ('m','$m_t$'),
        ('c','$c_t$'),
        ('a','$a_t$')
    ]

    # time index on x-axis
    age = np.arange(par.T)

    # plot average across agents
    for simvar,simvarlatex in simvarlist:
        simdata = getattr(sim,simvar)  # shape (T, simN)
        ax.plot(age, np.mean(simdata,axis=1), lw=2, label=simvarlatex)

    ax.set_xlabel('time')
    ax.set_title('Average simulated paths')
    ax.grid(True)
    ax.legend()
    plt.show()
    
def plot_multiple_consumption_functions(model, t_list):
    """
    Plot consumption c(t,m) for multiple t values in *one* figure.
    """

    # Unpack
    par = model.par
    sol = model.sol

    fig, ax = plt.subplots(figsize=(6,4))

    for t in t_list:
        ax.plot(par.grid_m, sol.c[t,:], label=f"t = {t}")

    ax.set_xlabel("m")
    ax.set_ylabel("c")
    ax.set_title("Consumption function for multiple t")
    ax.grid(True)
    ax.legend()
    plt.show()
