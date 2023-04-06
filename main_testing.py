# imports 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch as T
from hyperparams import *
from envs import *
from utils import *
from main_training import DDPG
from scipy import stats
np.seterr(divide='ignore', invalid='ignore')
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Palatino"
})

env_params, algo_params = init_params()
env = Environment(env_params)
rews, learned_pol = DDPG(algo_params, env)
(fig1, ax1), (fig2, ax2), (fig3, axs) = plt.subplots(1, 1, sharey=True), plt.subplots(1, 1, sharey=True), plt.subplots(1, len(env.spaces["t_space"]), figsize=(10, 2), sharey=True)  

# cumulative rewards

# ax1.plot(T.arange(algo_params["num_eps"]), rews)
# ax1.set(xlabel='Epochs', ylabel='Cumulative reward', title='Cumulative Reward per Epoch; Learned ANN Policy')    
# fig1.savefig('Avg.pdf')

    
# plot optimal policies through time
test_states = T.cartesian_prod(T.from_numpy(env.spaces["t_space"]).float(),\
              T.from_numpy(env.spaces["s_space"]).float(), T.from_numpy(env.spaces["q_space"]).float())

for k in range(len(env.spaces["t_space"])):
    
    plotted_pol = T.zeros((len(env.spaces["s_space"]), len(env.spaces["q_space"])))
    period_k_test_states = test_states[test_states[:,0] == env.spaces["t_space"][k]]
    opt_acts = learned_pol(period_k_test_states) 
    
    for i in range(len(env.spaces["s_space"])):
        for j in range(len(env.spaces["q_space"])):
            plotted_pol[i][j] = opt_acts[i*(len(env.spaces["q_space"])) + j]

    plotted_pol = T.flip(plotted_pol, dims=[0])
    mappable = axs[k].imshow(plotted_pol.detach().numpy(),
                    interpolation='none',
                    cmap=cmap,
                    extent=[np.min(env.spaces["q_space"]),
                            np.max(env.spaces["q_space"]),
                            np.min(env.spaces["s_space"]),
                            np.max(env.spaces["s_space"])],
                    aspect='auto',
                    vmin=env.params["min_q"],
                    vmax=env.params["max_q"])
    axs[k].set(title=f'Learned; Period: {k}', xlabel="Inventory", ylabel="Price")
    plt.colorbar(mappable, ax=axs[k])
    plt.tight_layout()

fig3.suptitle('Temporal Evolution of Learned ANN Policies')
fig3.savefig('Pols.pdf')
plt.show()
############################### 