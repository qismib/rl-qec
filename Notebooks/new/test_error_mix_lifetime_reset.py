# import delle librerie 
import numpy as np
import matplotlib.pyplot as plt
import torch

from Function_Library import *
from Environments_error_mix_lifetime_reset import *
from Agent_Torch import *

import copy
import os
import time

fixed_configs = {"d": 5,              
                "use_Y": True,
                "volume_depth": 5,
                "train_freq": 1,
                "print_freq": 100,      
                "batch_size": 64,       
                "buffer_size": 50000,
                "rolling_average_length": 500,  
                "max_timesteps": 1000000,
                "stopping_patience": 5000,   
                "c_layers": [[64,3,2],[32,2,1],[32,2,1]],
                "ff_layers": [[512,0.2]],
                "dueling": True,
                "masked_greedy": True,
                "static_decoder": True}

variable_configs = {"p_phys": 0.003,
                    "p_meas": 0.003,
                    "success_threshold": 1100,      
                    "learning_starts": 1000,             
                    "learning_rate": 0.0001,        
                    "exploration_fraction": 75000,  
                    "max_eps": 1.0,
                    "target_network_update_freq":5000, 
                    "gamma": 0.99,
                    "final_eps": 0.02}

all_configs = {}
for key in fixed_configs.keys():
    all_configs[key] = fixed_configs[key]
for key in variable_configs.keys():
    all_configs[key] = variable_configs[key]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# 2 - ambiente 
env = Surface_Code_Environment_Multi_Decoding_Cycles(
    d=all_configs["d"],
    p_phys=all_configs["p_phys"],
    p_meas=all_configs["p_meas"],
    volume_depth=all_configs["volume_depth"],
    use_Y=fixed_configs['use_Y']
)

# 3 - agente e suoi componenti
input_shape = env.observation_space.shape
num_actions = env.num_actions

# reti neurali --> nuovo dueling
policy_net = ConvNetRL(all_configs["c_layers"], all_configs["ff_layers"], input_shape, num_actions, dueling=all_configs["dueling"]).to(device)

#----------------------

BEST_MODEL_PATH = r'runs/DDQN_d5_p0.003_20260223-154154/final_policy_net_ciao.pth'
policy_net.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
policy_net.eval() 

policy = LinearAnnealedEpsGreedyPolicy(
    value_max=all_configs["max_eps"], 
    value_min=all_configs["final_eps"], 
    nb_steps=all_configs["exploration_fraction"],
    masked_greedy=all_configs["masked_greedy"]
)



# test

TRAINING_LOG_PATH = r'runs/DDQN_d5_p0.003_20260223-154154/training_history.json'
N_episode = 1500
run = TRAINING_LOG_PATH.split('\\')[0]
# test_avg_reward, test_std_reward, test_rewards, test_avg_lifetime, test_std_lifetime, test_lifetimes = test_agent_performance(env, policy_net, policy, N_episode, device)
test_path = test_agent_performance(env, policy_net, policy, N_episode, device, log=True, run_name=run)