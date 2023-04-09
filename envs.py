"""
Statistical Arbitrage Environment
"""
# imports
import numpy as np
import torch as T
from hyperparams import *
from torch.distributions.uniform import Uniform
import pdb


class Environment():
    # constructor
    def __init__(self, params: dict):
        # parameters and spaces
        self.params = params
        self.spaces = {'t_space' : np.arange(params["Ndt"]), # time space / periods 
                      's_space' : np.linspace(params["theta"]-2*params["sigma"]/np.sqrt(2*params["kappa"]),
                                  params["theta"]+2*params["sigma"]/np.sqrt(2*params["kappa"]), 51), # price space
                      'q_space' : np.linspace(params["min_q"], params["max_q"], 51), # inventory space
                      'u_space' : np.linspace(params["min_u"], params["max_u"], 21)} # action space

    # initialization of the environment with its true initial state
    def reset(self, Nsims=1):
        t0 = T.zeros(Nsims)
        x0 = T.normal(self.params["theta"],
                        self.params["sigma"]/np.sqrt(2*self.params["kappa"]),
                        size=(Nsims,))
        x0 = T.minimum(T.maximum(x0, min(self.spaces["s_space"])*T.ones(1)), max(self.spaces["s_space"])*T.ones(1)) # ensures x0 in valid range
        qn1 = T.zeros(Nsims)

        return T.tensor([t0, x0, qn1])

        # initialization of the environment with its random initial state
    def random_reset(self, Nsims=1):
        t0 = T.zeros(Nsims)
        x0 = T.normal(self.params["theta"],
                        self.params["sigma"]/np.sqrt(2*self.params["kappa"]),
                        size=(Nsims,))
        x0 = T.minimum(T.maximum(x0, min(self.spaces["s_space"])*T.ones(1)), max(self.spaces["s_space"])*T.ones(1)) # ensures x0 in valid range
        qn1 = Uniform(T.tensor([self.params["min_q"]]).float(), T.tensor([self.params["max_q"]]).float()).sample()


        return T.tensor([t0, x0, qn1])   
    
    # simulation engine
    def step(self, curr_state, action):
        
        # time modification -- step forward
        time_t, x_t, q_tm1 = curr_state[0], curr_state[1], curr_state[2] 
        q_t = action  
        
        time_tp1 = time_t + 1

        # price modification -- OU process
        sizes = q_tm1.shape
        dt = self.params["T"]/self.params["Ndt"]
        eta = self.params["sigma"] * \
                np.sqrt((1 - np.exp(-2*self.params["kappa"]*dt)) / (2*self.params["kappa"]))
        x_tp1 = self.params["theta"] + \
                (x_t-self.params["theta"]) * np.exp(-self.params["kappa"]*dt) + \
                eta*T.randn(sizes, device=self.device)
        
        # reward -- change of book value of shares with transaction costs (attempting to recover bang-bang control)
        reward_t = q_t*(x_tp1 - x_t) - (self.params["phi"]*T.pow(q_t - q_tm1, 2))
        reward = reward_t
        new_state = T.tensor([time_tp1, x_tp1, q_t])
        
        return new_state, reward