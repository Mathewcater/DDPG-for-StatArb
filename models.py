"""
Models

Policy: two fully-connected ANNs (main and target)
Q-function: two fully-connected ANNs (main and target)

Helper Objects

Replay buffer
OU exploratory noise 

"""
# imports

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import silu, tanh
from torch.distributions.categorical import Categorical
import pdb
from envs import Environment
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable



# build a fully-connected neural net for the policy
class PolicyANN(nn.Module):
    # constructor
    def __init__(self, input_size: int, hidden_size: int, n_layers: int, env: Environment,
                    learn_rate=0.001, step_size=50, gamma=0.95):
        super(PolicyANN, self).__init__()
        
        self.input_size = input_size # number of inputs
        self.hidden_size = hidden_size # number of hidden nodes
        self.output_size = 1 # number of outputs
        self.n_layers = n_layers # number of layers
        self.env = env # environment (for normalisation purposes)

        # build all layers
        self.layer_in = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.n_layers-1)])
        self.layer_out = nn.Linear(self.hidden_size, self.output_size)

        # initializers for weights and biases
        nn.init.normal_(self.layer_in.weight, mean=0, std=1/np.sqrt(input_size)/2)
        nn.init.constant_(self.layer_in.bias, 0)
        for layer in self.hidden_layers:
            nn.init.normal_(layer.weight, mean=0, std=1/np.sqrt(input_size)/2)
            nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.layer_out.weight, mean=0, std=1/np.sqrt(input_size)/2)
        nn.init.constant_(self.layer_out.bias, 0)
        
        # batch normalization
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(self.hidden_size) for _ in range(self.n_layers-1)])
        self.instance_norm = nn.InstanceNorm1d(self.hidden_size) # for batch size of 1

        # optimizer and scheduler
        self.optimizer = optim.AdamW(self.parameters(), lr=learn_rate, maximize=True) 
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    

    # forward propagation
    def forward(self, x):
        # normalize features with environment parameters
        y = x.clone()
        y[...,0] /= self.env.params["Ndt"] # time
        y[...,1] = (y[...,1] - self.env.params["theta"]) / 0.5 # price of the asset
        y[...,2] /= self.env.params["max_q"] # inventory
            
        # output of input layer 
        action = silu(self.layer_in(y))
    
        for layer in self.hidden_layers:
            action = silu(layer(action))
            
        # activate at final layer to ensure outputs in desired range
        action = self.env.params["max_q"]*T.tanh(self.layer_out(action))
        return action


# build a fully-connected neural net for the policy
class Q_ANN(nn.Module):
    # constructor
    def __init__(self, input_size, hidden_size, n_layers, env,
                    learn_rate=0.001, step_size=50, gamma=0.95):
        super(Q_ANN, self).__init__()
        
        self.input_size = input_size # number of inputs
        self.hidden_size = hidden_size # number of hidden nodes
        self.output_size = 1 # number of outputs
        self.n_layers = n_layers # number of layers
        self.env = env # environment (for normalisation purposes)

        # build all layers
       
        self.layer_in = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.n_layers-1)])
        self.layer_out = nn.Linear(self.hidden_size, self.output_size)

        # initializers for weights and biases
        nn.init.normal_(self.layer_in.weight, mean=0, std=1/np.sqrt(input_size)/2)
        nn.init.constant_(self.layer_in.bias, 0)
        for layer in self.hidden_layers:
            nn.init.normal_(layer.weight, mean=0, std=1/np.sqrt(input_size)/2)
            nn.init.constant_(layer.bias, 0)
        nn.init.normal_(self.layer_out.weight, mean=0, std=1/np.sqrt(input_size)/2)
        nn.init.constant_(self.layer_out.bias, 0)

        # batch normalization
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(self.hidden_size) for _ in range(self.n_layers-1)])
        self.instance_norm = nn.InstanceNorm1d(self.hidden_size) # for batch size of 1
        
        # optimizer and scheduler
        self.optimizer = optim.Adam(self.parameters(), lr=learn_rate) 
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
      
    
    def forward(self, x):
        # normalize features with environment parameters
        y = x.clone()
        y[...,0] /= self.env.params["Ndt"] # time
        y[...,1] = (y[...,1] - self.env.params["theta"]) / 0.5 # price of the asset
        y[...,2] /= self.env.params["max_q"] # inventory
        y[...,3] /= self.env.params["max_q"] # inventory
                        
        # output of input layer 
        Q_val = silu(self.layer_in(y))
        
        for i, layer in enumerate(self.hidden_layers):
            # apply batch normalization, handling batch size of 1
            if x.shape[0] == 1:
                Q_val = silu(self.instance_norm(layer(Q_val)))
            else:
                Q_val = silu(self.batch_norms[i](layer(Q_val)))
                
        Q_val = self.layer_out(Q_val)
        
        return Q_val
    
class OUNoise:
    """Ornstein-Uhlenbeck exploratory noise.
    """
    def __init__(self, action_dim=1, mu=0.0, kappa=4.25, sigma=2.5):
        self.action_dim = action_dim
        self.mu = mu
        self.kappa = kappa
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        self.state = T.ones(self.action_dim) * self.mu
        
    def noise(self):
        x = self.state
        dx = self.kappa * (self.mu - x) + self.sigma * T.randn(len(x))
        self.state = x + dx
        return self.state
        
        
class ReplayBuffer:
    """Replay buffer object that allows transitions to be stored and 
    mini-batches of transitions to be sampled from.
    """
    def __init__(self, algo_params: dict):
        self.algo_params = algo_params
        self.buffer_batch_size = algo_params["buffer_batch_size"]
        self.buffer_capacity = algo_params["buffer_capacity"]
        self.buffer = []
        
    def add(self, curr_state, action, rew, new_state):
        """Add the observed transition to the replay buffer
        """
        # if buffer full, then deque a transition
        if len(self.buffer) == self.buffer_capacity:
            self.buffer = self.buffer[1:] 
            
        # add new transition
        self.buffer.append(T.cat((curr_state.detach(), action.detach(), rew.detach(), new_state.detach())))
        
    def sample(self):
        """Sample a mini-batch of transitions from the replay buffer
        """
        # if buffer smaller than desired batch size, sample entire buffer
        if len(self.buffer) < self.buffer_batch_size:
            transitions = T.stack(self.buffer)
            
        # otherwise, uniformly sample mini-batch of transitions
        else:
            probs = (T.ones(len(self.buffer))/len(self.buffer)).repeat((self.buffer_batch_size, 1)) 
            dist = Categorical(probs)
            transitions = (T.stack(self.buffer))[dist.sample()] 
            
        trans_curr_states = transitions[:,:3]
        trans_acts = transitions[:,3].unsqueeze(dim=-1)
        trans_rews = transitions[:,4].unsqueeze(dim=-1)
        trans_new_states = transitions[:,5:]
        
        return trans_curr_states, trans_acts, trans_rews, trans_new_states     
    

def plot_noise(num_steps, noise: OUNoise):
    """Plot the OU noise (visualizer).
    """
    fig, ax = plt.subplots(1,1)
    stats = []
    for _ in range(num_steps):
        noise.noise()
        stats.append(noise.state)
    
    ax.plot(T.arange(num_steps), T.stack(stats))
    fig.savefig('OU Noise')
    
if __name__ == '__main__':
    noise = OUNoise()
    plot_noise(num_steps=50, noise=noise)
    