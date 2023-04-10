# imports 
import numpy as np
import copy
import torch as T
import pdb
from models import *
from envs import *
from torch.nn import MSELoss 
from torch.distributions.uniform import Uniform
from tqdm import tqdm
from hyperparams import *
from utils import *
import matplotlib.pyplot as plt

# For reproducibility 
T.manual_seed(54321)

# Anamoly Detection 
T.autograd.set_detect_anomaly(True)

def plot_current_policy(env: Environment, actor_main: PolicyANN, episode_num: int):
    """Plot the current policy aafter given number of episodes (episode_num)

    Args:
        env (Environment): environment
        actor_main (PolicyANN): main actor network
        episode_num (int): training step to plot at
    """
    fig, axs = plt.subplots(1, len(env.spaces["t_space"]), figsize=(10, 2), sharey=True)
    curr_learned_pol = actor_main
    
    # plot optimal policies through time
    test_states = T.cartesian_prod(T.from_numpy(env.spaces["t_space"]).float(),\
                  T.from_numpy(env.spaces["s_space"]).float(), T.from_numpy(env.spaces["q_space"]).float())

    for k in range(len(env.spaces["t_space"])):
        
        period_k_test_states = test_states[test_states[:,0] == env.spaces["t_space"][k]]
        opt_acts = curr_learned_pol(period_k_test_states) 
        plotted_pol = T.zeros((len(env.spaces["s_space"]), len(env.spaces["q_space"])))
        
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

    fig.suptitle(f'Temporal Evolution of Learned ANN Policies; Episode No.: {episode_num}')
    fig.savefig(f'Learned_Policy_Episode_No_{episode_num}.pdf')
    
    
    
def update_critic_main(targets: T.tensor, trans_curr_states: T.tensor, trans_acts: T.tensor, \
                           critic_main: Q_ANN):
    """
    Update main critic network using labels produced by target network
    and minimizing the average squared Bellman-loss. 
    
    Args:
        targets (T.tensor): Regression labels from target network
        trans_new_states (T.tensor): Training data for main critic 
        acts (T.tensor): Training data for main critic 
        critic_main (Q_ANN): main critic network
        num_epochs (int): number of parameter updates 
    """
     
    input = critic_main(T.cat((trans_curr_states, trans_acts), -1))
    critic_main.optimizer.zero_grad()
    loss_function = MSELoss()
    loss = loss_function(targets, input)
    loss.backward() 
    critic_main.optimizer.step()
        
    if critic_main.scheduler.get_last_lr()[0] >= 5e-4:
        critic_main.scheduler.step()
    
    
    
def update_actor_main(critic_main: Q_ANN, actor_main: PolicyANN, curr_states: T.tensor):
    """ 
    Update main actor network via. Monte-Carlo estimate of objective function.
    
    Args:
        critic_main (Q_ANN):  main critic network
        actor_main (PolicyANN): main actor network
        curr_states (T.tensor): current states from mini-batch of transitions
    """
    # freeze main critic network params
    for param in critic_main.parameters():
        param.requires_grad = False
    
    actions = actor_main(curr_states)
    actor_main.optimizer.zero_grad()
    loss = T.mean(critic_main(T.cat((curr_states, actions), -1)))
    loss.backward()
    actor_main.optimizer.step()
    if actor_main.scheduler.get_last_lr()[0] >= 5e-4:
        actor_main.scheduler.step()

    # un-freeze main critic netwotk params
    for param in critic_main.parameters():
        param.requires_grad = True


def update_target_net(main, target, tau):
    """
    Updates target networks as a convex combination of main
    network weights.

    Args:
        main (PolicyANN or Q_ANN): main actor or main critic network
        target (PolicyANN or Q_ANN): target actor or target critic network
        tau (float): convex combination param (tau << 1)
    """
    with T.no_grad():
        for p, q in zip(main.parameters(), target.parameters()):
            q.data = tau*p.data + (1-tau)*q.data
        

def DDPG(algo_params: dict, env: Environment):
    """Generate, and train, DDPG model on the environment specifications
    provided in 'env' and executed with the hyperparams specified in 'algoparams'.

    Args:
        algo_params (dict): Dictionary containing training hyperparams
        env (Environment): Environment object containing environment/problem params
                           (price dynamics, max and min inventory levels etc.)
    Returns:
        actor_main (PolicyANN): Trained actor network; learned ANN policy.
        cum_rews (T.tensor): 
    """
    # initialize main and target networks
    actor_main = PolicyANN(input_size=3, hidden_size=algo_params["hidden_size_actor"], \
                           n_layers=algo_params["num_layers_actor"], env=env, learn_rate=algo_params["lr_actor"])
    critic_main = Q_ANN(input_size=4, hidden_size=algo_params["hidden_size_critic"], \
                        n_layers=algo_params["num_layers_critic"], env=env, learn_rate=algo_params["lr_critic"])
    actor_target = copy.deepcopy(actor_main)
    critic_target = copy.deepcopy(critic_main)
    
    # freeze all target networks w.r.t. optimizers
    for p, q in zip(actor_target.parameters(), critic_target.parameters()):
        p.requires_grad = False
        q.requires_grad = False 
    
    # initialize replay buffer and exploratory noise process
    replay_buffer = ReplayBuffer(algo_params) 
    noise_process = Noise(sigma=algo_params["noise_vol"]) 
    
    # uniform dist. over action space for exploration
    unif_act = Uniform(T.tensor([env.params["min_q"]]).float(), T.tensor([env.params["max_q"]]).float())
    cum_rews = []
    
    for m in tqdm(range(algo_params["num_eps"])):
        
        curr_state = env.random_reset()
        noise_process.reset()
        rews = []
        
        # main training loop
        for _ in range(env.params["Ndt"]):
            
            # If sufficiently many episodes elapsed, take action according to policy with exploratory noise,
            # otherwise sample uniformly to encourage aggressive exploration early 
            if m > algo_params["start_steps"]:
                action = T.clamp(actor_main(curr_state) + noise_process.noise(), env.params["min_q"], env.params["max_q"])  
            else: 
                action = unif_act.sample()
            
            # get new state and reward
            new_state, rew = env.step(curr_state, action) 
            rews.append(rew.detach())

            # add transition to buffer (all inputs get detached in the add method)
            replay_buffer.add(curr_state, action, rew, new_state)  
            
            # update state
            curr_state = new_state

            # once buffer sufficiently full, begin updates:
            if m >= algo_params["explore_eps"] and m % algo_params["update_every"] == 0:
                
                for _ in range(env.params["Ndt"]*algo_params["update_every"]):
                    # sample mini-batch of transitions 
                    trans_curr_states, trans_acts, trans_rews, trans_new_states = replay_buffer.sample()
                    
                    # compute targets
                    targets = trans_rews + critic_target(T.cat((trans_new_states, actor_target(trans_new_states).detach()), -1)).detach()
                    
                    # update main networks   
                    update_critic_main(targets, trans_curr_states, trans_acts, critic_main)     
                    update_actor_main(critic_main, actor_main, trans_curr_states)
                    
                    # update target networks 
                    update_target_net(critic_main, critic_target, algo_params["tau"])
                    update_target_net(actor_main, actor_target, algo_params["tau"])
            
        if m % algo_params["save_freq"] == 0:
            plot_current_policy(env, actor_main, episode_num=m)
            
        cum_rews.append(T.sum(T.stack(rews))) 
        
    return T.stack(cum_rews), actor_main