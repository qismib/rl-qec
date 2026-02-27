# import delle librerie 
import numpy as np
import matplotlib.pyplot as plt
import torch

from Function_Library import *
from Environments_no_referee_mixed_noise import *
from Agent_Torch_no_referee_mixed_noise import *

import copy
import os
import time

fixed_configs = {"d": 7,              
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

noise_cfg = {
    # uncorrelated
    "p_x": 0.002,
    "p_y": 0.0005,
    "p_z": 0.002,

    # correlated nearest-neighbor
    "p_corr_xx": 0.0004,
    "p_corr_zz": 0.0004,
    "p_corr_xz": 0.0001,
    "p_corr_zx": 0.0001,
    "p_corr_yy": 0.00005,

    "include_diagonal_pairs": False,
    "correlated_first": True,
}



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
    use_Y=fixed_configs['use_Y'],
    noise_cfg=noise_cfg
)

# 3 - agente e suoi componenti
input_shape = env.observation_space.shape
num_actions = env.num_actions

# reti neurali --> nuovo dueling
policy_net = ConvNetRL(all_configs["c_layers"], all_configs["ff_layers"], input_shape, num_actions, dueling=all_configs["dueling"]).to(device)
target_net = copy.deepcopy(policy_net).to(device)
target_net.eval()

# ottimizzatore
optimizer = optim.Adam(policy_net.parameters(), lr=all_configs["learning_rate"])

# memoria e policy
memory = ReplayBuffer(all_configs["buffer_size"])
policy = LinearAnnealedEpsGreedyPolicy(
    value_max=all_configs["max_eps"], 
    value_min=all_configs["final_eps"], 
    nb_steps=all_configs["exploration_fraction"],
    masked_greedy=all_configs["masked_greedy"]
)

# agente DDQN
agent = PyTorchDDQNAgent(policy_net, target_net, optimizer, memory, all_configs, device)

print("Inizializzazione di tutti i componenti completata.")
print(f"Policy Net input shape: {input_shape}, Output: {num_actions}")


timeS = time.time()
trainV2(agent, env, policy, policy_net, memory, all_configs, device)
timeF = time.time()
dT = timeF-timeS
print(f'durata {dT}s')


