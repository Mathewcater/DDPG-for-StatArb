"""
Hyperparameters
Initialization of all hyperparameters
"""


# initialize parameters for the environment and algorithm
def init_params():
    # parameters for the model
    env_params = {'kappa' : 4.5, # kappa of the OU process
              'sigma' : 0.9, # standard deviation of the OU process
              'theta' : 1.0, # mean-reversion level of the OU process
              'phi' : 0.005, # transaction costs
              'psi' : 0.0, # terminal penalty on the inventory
              'T' : 1, # trading horizon
              'Ndt' : 5, # number of periods
              'min_q' : -10, # minimum value for the inventory
              'max_q' : 10, # maximum value for the inventory
              'min_u' : -10, # minimum value for the rebalances
              'max_u' : 10} # maximum value for the rebalances

    # parameters for the algorithm
    
    algo_params = {'num_eps' : 1_000, # number of episodes of the whole algorithm
                'tau' : 0.005, # target network convex comb. update parameter 
                'buffer_batch_size': 750, # size of minibatches sampled from replay buffer
                'buffer_capacity': 5000, # max size of replay buffer
                'critic_main_num_epochs': 1_750, # number of epochs (gred. descent steps) considered in main critic updates
                'main_update_num_eps': 5, # number of epochs elapsed until main critic is updated
                'target_update_num_eps': 100, # number of epochs elapsed until the target networks are updated
                'lr_actor' : 0.001, # learning rate of the actor networks (main and target)
                'lr_critic' : 0.004, # learning rate of the critic networks (main and target) 
                'hidden_size_actor' : 16, # number of hidden nodes in the neural net associated with the actor
                'hidden_size_critic' : 50, # number of hidden nodes in the neural net associated with the critic
                'num_layers_actor' : 5, # number of layers in the neural net associated with actor
                'num_layers_critic' : 10, # number of layers in the neural net associated with critic
                } 

    return env_params, algo_params

