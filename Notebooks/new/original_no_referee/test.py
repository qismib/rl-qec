# import delle librerie completo
import numpy as np
import matplotlib.pyplot as plt
import torch


from Function_Library import *
from Environments_no_referee_1 import *
from Agent_Torch_no_referee_1 import *

import json
import copy
import os
import time
import gc


fixed_configs = {"d": 5,
                "error_model": "X",                 
                "use_Y": False,
                "volume_depth": 5,
                "train_freq": 1,
                "print_freq": 100,      
                "batch_size": 64,       
                "buffer_size": 50000,
                "rolling_average_length": 500,  
                "max_timesteps": 1000000,
                "stopping_patience": 10000,   
                "c_layers": [[64,3,2],[32,2,1],[32,2,1]],
                "ff_layers": [[512,0.2]],
                "dueling": True,
                "masked_greedy": False}

variable_configs = {"p_phys": 0.003,            
                    "p_meas": 0.003,            
                    "success_threshold": 1100,       # necessaria per early stopping intelligente. impostata come almeno la vita media di un faulty qubit (1/p_err)
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

env = Surface_Code_Environment_Multi_Decoding_Cycles(
    d=all_configs["d"],
    p_phys=all_configs["p_phys"],
    p_meas=all_configs["p_meas"],
    error_model=all_configs["error_model"],
    volume_depth=all_configs["volume_depth"]
)

# 3 - agente e suoi componenti
input_shape = env.observation_space.shape
num_actions = env.num_actions

# reti neurali --> nuovo dueling
policy_net = ConvNetRL(all_configs["c_layers"], all_configs["ff_layers"], input_shape, num_actions, dueling=all_configs["dueling"]).to(device)

policy = LinearAnnealedEpsGreedyPolicy(
    value_max=all_configs["max_eps"], 
    value_min=all_configs["final_eps"], 
    nb_steps=all_configs["exploration_fraction"],
    masked_greedy=all_configs["masked_greedy"]
)

# estrazione log di allenamento 
#TRAINING_LOG_PATH = r'runs/DDQN_d5_p0.003_20260302-111330_rl1/training_history.json'
TRAINING_LOG_PATH = r'runs/DDQN_d5_p0.003_20260302-111341_rl2/training_history.json'

training_history = load_history(TRAINING_LOG_PATH)
# estraggo da history tutte le liste di log ottenuti con trainV2 : 
step = np.array([log_data['step'] for log_data in training_history])
episode = np.array([log_data['episode'] for log_data in training_history])
episode_step_count = np.array([log_data['episode_step_count'] for log_data in training_history])
reward = np.array([log_data['episode_reward'] for log_data in training_history])
avg_reward  = np.array([log_data['reward_mean'] for log_data in training_history])
std_reward = np.array([log_data['reward_std'] for log_data in training_history])
lifetime = np.array([log_data['episode_lifetime'] for log_data in training_history])
avg_lifetime = np.array( [log_data['lifetime_mean'] for log_data in training_history])
std_lifetime = np.array([log_data['lifetime_std'] for log_data in training_history])
loss = np.array([log_data['loss'] for log_data in training_history])
eps = np.array([log_data['epsilon'] for log_data in training_history])
run = TRAINING_LOG_PATH.split('/')[1]
print('----------------------------------------------------------')
print(f'numero episodi totali : {episode[-1]}')
print(f'numero step totali : {step[-1]}')
print(f"best avg lifetime (rolling lenght {all_configs['rolling_average_length']}) : {np.max(avg_lifetime)}")
print('----------------------------------------------------------')

# grafici
# lifetime vs episode
fig, ax = plt.subplots()
fig.suptitle(run)
# ax.errorbar(episode, avg_lifetime, yerr=std_lifetime, label='avg lifetime rolling 500 ep', c='b', ecolor='lightgray', elinewidth=1, capsize=0)
ax.plot(episode, lifetime, label='lifetime', alpha= 0.5,c='skyblue')
ax.plot(episode, avg_lifetime, label='avg lifetime rolling 500 ep', c='b')
ax.set_title('Lifetime vs Episode')
ax.set_xlabel('Episode')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.legend()
ax.grid()
plt.show()

# avg_lifetime vs timestep
fig, ax = plt.subplots()
fig.suptitle(run)
ax.plot(step, avg_lifetime, c='b', label='avg lifetime rolling 500 ep')
ax.set_title('Avg lifetime rolling 500 vs timesteps')
ax.set_xlabel('timestep')
ax.set_ylabel('lifetime')
ax.legend()
ax.grid()
plt.show()

# reward vs episode
fig, ax = plt.subplots()
fig.suptitle(run)
# ax.errorbar(episode, avg_reward, yerr=std_reward, label='avg reward rolling 500 ep', c='r', ecolor='lightgray', elinewidth=1, capsize=0)
ax.plot(episode, reward, label='reward', alpha= 0.5,c='coral')
ax.plot(episode, avg_reward, label='avg reward rolling 500 ep', c='r')
ax.set_title('Reward vs Episode')
ax.set_xlabel('Episode')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.legend()
ax.grid()
plt.show()

# step count episode vs episode
fig, ax = plt.subplots()
fig.suptitle(run)
ax.plot(episode, episode_step_count, label='step')
ax.set_title('Step per episode vs Episode')
ax.set_xlabel('Episode')
ax.set_ylabel('Step per episode')
ax.legend()
ax.grid()
plt.show()

# loss vs episode
fig, ax = plt.subplots()
fig.suptitle(run)
ax.plot(episode, loss, label='loss')
ax.set_title('Loss vs Episode')
ax.set_xlabel('Episode')
ax.set_ylabel('Loss')
ax.legend()
ax.grid()
plt.show()

# summary
fig, ax = plt.subplots()
fig.suptitle(run)
ax.plot(episode, eps, label='eps')
ax.plot(episode, avg_lifetime/np.max(avg_lifetime), label ='avg lifetime norm')
# ax.plot(episode, loss/np.max(loss), alpha=0.3, label='loss')
ax.set_title('Summary')
ax.set_xlabel('Episode')
ax.legend()
ax.grid()
plt.show()

BEST_MODEL_PATH = r'runs/DDQN_d5_p0.003_20260302-111330_rl1/checkpoints/best_policy_net.pth'
policy_net.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
policy_net.eval() 

# test
N_episode = 1000
# test_avg_reward, test_std_reward, test_rewards, test_avg_lifetime, test_std_lifetime, test_lifetimes = test_agent_performance(env, policy_net, policy, N_episode, device)
test_path = test_agent_performance(env, policy_net, policy, N_episode, device, log=True, run_name=run)

