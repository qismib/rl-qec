# import delle librerie completo
import numpy as np
import matplotlib.pyplot as plt
import torch


from Function_Library import *
from Environments_old import Surface_Code_Environment_Multi_Decoding_Cycles
from Agent_Torch import *
'''from Environments_mixed_noise import *
from Agent_Torch_mixed_noise import *'''

import json
import copy
import os
import time
import gc

import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class SmallGridCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        c, h, w = observation_space.shape  # (6, 11, 11)

        self.cnn = nn.Sequential(
            nn.Conv2d(c, 64, kernel_size=3, stride=1, padding=1),  # 11x11 -> 11x11
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # 11x11 -> 6x6
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1), # 6x6 -> 3x3
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.zeros(1, c, h, w)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))




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
                "stopping_patience": 1000,   
                "c_layers": [[64,3,2],[32,2,1],[32,2,1]],
                "ff_layers": [[512,0.2]],
                "dueling": True,
                "masked_greedy": False,
                "static_decoder": True}

variable_configs = {"p_phys": 0.003,
                    "p_meas": 0.003,
                    "success_threshold": 10000,      
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

# 1 - decoder statico 
'''static_decoder_path = os.path.join(os.getcwd(),"referee_decoders/static_decoder_d5_X_p5.pth")
static_decoder = PyTorchStaticDecoder() 
static_decoder.load_state_dict(torch.load(static_decoder_path)) 
static_decoder.eval()
print('static_decoder caricato!')'''

# 2 - ambiente 
'''env = SurfaceCodeGymWrapper(
    d=all_configs["d"],
    p_phys=all_configs["p_phys"],
    p_meas=all_configs["p_meas"],
    volume_depth=all_configs["volume_depth"],
)
print(env.error_model)

# 3 - agente e suoi componenti
input_shape = env.observation_space.shape
num_actions = env.num_actions

# reti neurali 
policy_net = ConvNetRL(all_configs["c_layers"], all_configs["ff_layers"], input_shape, num_actions, dueling=all_configs["dueling"]).to(device)
target_net = copy.deepcopy(policy_net).to(device)
target_net.eval()

# inizializzo l'Optimizer 
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
print(f'durata {dT}s')'''

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import os

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

import time

class DDQNStyleLoggerCallback(BaseCallback):
    def __init__(self, print_freq_episodes=100, rolling_window=500, verbose=1):
        super().__init__(verbose)
        self.print_freq_episodes = print_freq_episodes
        self.rolling_window = rolling_window

        self.episode_count = 0
        self.all_rewards = []
        self.all_lifetimes = []
        self.all_ep_steps = []

        self.best_avg_reward = -np.inf
        self.best_avg_lifetime = -np.inf

        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is None or infos is None:
            return True

        for i, done in enumerate(dones):
            if not done:
                continue

            info = infos[i] if i < len(infos) else {}
            ep_reward = info.get("episode_reward", None)
            ep_steps = info.get("episode_steps", None)
            ep_lifetime = info.get("episode_lifetime", None)

            # se manca qualcosa, salta (ma con il wrapper sopra NON dovrebbe mancare)
            if ep_reward is None or ep_steps is None or ep_lifetime is None:
                continue

            self.episode_count += 1
            self.all_rewards.append(float(ep_reward))
            self.all_ep_steps.append(int(ep_steps))
            self.all_lifetimes.append(int(ep_lifetime))

            # scegli finestra: tutti finché <500, poi ultimi 500
            if self.episode_count < self.rolling_window:
                r_slice = self.all_rewards
                l_slice = self.all_lifetimes
            else:
                r_slice = self.all_rewards[-self.rolling_window:]
                l_slice = self.all_lifetimes[-self.rolling_window:]

            current_reward_avg = float(np.mean(r_slice))
            current_reward_std = float(np.std(r_slice))
            current_lifetime_avg = float(np.mean(l_slice))
            current_lifetime_std = float(np.std(l_slice))

            # aggiorna best
            if current_reward_avg > self.best_avg_reward:
                self.best_avg_reward = current_reward_avg
            if current_lifetime_avg > self.best_avg_lifetime:
                self.best_avg_lifetime = current_lifetime_avg

            # stampa ogni 100 episodi
            if self.episode_count % self.print_freq_episodes == 0:
                time_elapsed = time.time() - self.start_time
                total_steps = self.num_timesteps  # timesteps SB3 (uguale concetto di step_count)

                window_name = f"All Ep" if self.episode_count < self.rolling_window else f"Last {self.rolling_window} Ep"

                print(f"""
--------------------------------
Episode: {self.episode_count}
Step: {total_steps}
Episode Step : {ep_steps}
Episode Reward : {ep_reward}
Avg Reward ({window_name}): {current_reward_avg:.2f} +- {current_reward_std:.2f}
Best Avg Reward: {self.best_avg_reward:.2f}
Episode Lifetime : {ep_lifetime}
Avg Lifetime ({window_name}): {current_lifetime_avg:.2f} +- {current_lifetime_std:.2f}
Best Avg Lifetime : {self.best_avg_lifetime:.2f}
Total Training Time: {time_elapsed:.2f}s
--------------------------------
""")

        return True


class AvgLifetimeCallback(BaseCallback):
    def __init__(self, print_every=100, window=500, verbose=1):
        super().__init__(verbose)
        self.print_every = print_every
        self.window = window
        self.episode_count = 0
        self.lifetimes = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None:
            return True

        for i, done in enumerate(dones):
            if done:
                self.episode_count += 1
                lt = None
                if infos is not None and i < len(infos):
                    lt = infos[i].get("episode_lifetime", None)
                if lt is not None:
                    self.lifetimes.append(lt)

                if self.episode_count % self.print_every == 0:
                    if len(self.lifetimes) == 0:
                        print(f"[Ep {self.episode_count}] lifetime non disponibile")
                        continue

                    if self.episode_count <= self.window:
                        avg_lt = float(np.mean(self.lifetimes))  # tutti
                        print(f"[Ep {self.episode_count}] Avg lifetime (All {len(self.lifetimes)}): {avg_lt:.2f}")
                    else:
                        avg_lt = float(np.mean(self.lifetimes[-self.window:]))  # ultimi 500
                        print(f"[Ep {self.episode_count}] Avg lifetime (Last {self.window}): {avg_lt:.2f}")

        return True

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class AvgLifetimeAndRewardCallback(BaseCallback):
    def __init__(self, print_every=100, window_after=500, window_size=500, verbose=1):
        super().__init__(verbose)
        self.print_every = print_every
        self.window_after = window_after
        self.window_size = window_size

        self.episode_count = 0
        self.lifetimes = []
        self.ep_rewards = []

        # per accumulare reward mentre l'episodio è in corso (per ogni env del vec env)
        self._running_rewards = None

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")   # shape (n_envs,)
        dones   = self.locals.get("dones")     # shape (n_envs,)
        infos   = self.locals.get("infos")     # list of dict, len=n_envs

        if dones is None or rewards is None:
            return True

        n_envs = len(dones)
        if self._running_rewards is None:
            self._running_rewards = np.zeros(n_envs, dtype=np.float64)

        # accumula reward step-by-step
        self._running_rewards += np.array(rewards, dtype=np.float64)

        for i, done in enumerate(dones):
            if done:
                self.episode_count += 1

                # chiudi reward episodio i
                ep_r = float(self._running_rewards[i])
                self.ep_rewards.append(ep_r)
                self._running_rewards[i] = 0.0

                # lifetime da info
                lt = None
                if infos is not None and i < len(infos):
                    lt = infos[i].get("episode_lifetime", None)
                if lt is not None:
                    self.lifetimes.append(float(lt))

                # stampa ogni N episodi
                if self.episode_count % self.print_every == 0:
                    if self.episode_count < self.window_after:
                        r_avg = float(np.mean(self.ep_rewards))
                        lt_avg = float(np.mean(self.lifetimes)) if len(self.lifetimes) > 0 else float("nan")
                        print(f"[EP {self.episode_count}] Avg reward (ALL): {r_avg:.2f} | Avg lifetime (ALL): {lt_avg:.2f}")
                    else:
                        r_avg = float(np.mean(self.ep_rewards[-self.window_size:]))
                        lt_avg = float(np.mean(self.lifetimes[-self.window_size:])) if len(self.lifetimes) >= self.window_size else float(np.mean(self.lifetimes))
                        print(f"[EP {self.episode_count}] Avg reward (LAST {self.window_size}): {r_avg:.2f} | "
                              f"Avg lifetime (LAST {self.window_size}): {lt_avg:.2f}")

        return True



import torch
from sb3_contrib import MaskablePPO

static_decoder_path = os.path.join(os.getcwd(),"referee_decoders/static_decoder_d5_X_p5.pth")
static_decoder = PyTorchStaticDecoder() 
static_decoder.load_state_dict(torch.load(static_decoder_path)) 
static_decoder.eval()
print('static_decoder caricato!')

# 1) crea env core identico a prima
core_env = Surface_Code_Environment_Multi_Decoding_Cycles(
    d=5,
    p_phys=0.003,
    p_meas=0.003,
    error_model="X",        # o "DP"
    use_Y=False,
    volume_depth=5,
    static_decoder=static_decoder
)

# 2) wrappa in gym env + mask
env = SurfaceCodeGymWrapper(core_env)


'''env = Surface_Code_Environment_Multi_Decoding_Cycles(
    d=all_configs["d"],
    p_phys=all_configs["p_phys"],
    p_meas=all_configs["p_meas"],
    error_model=all_configs["error_model"],
    volume_depth=all_configs["volume_depth"],
    static_decoder=static_decoder 
)'''


import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ConvNetRLFeatures(BaseFeaturesExtractor):
    """
    Replica la parte feature di ConvNetRL (Agent_Torch):
    conv stack definito da c_layers + fc stack definito da ff_layers.
    """
    def __init__(self, observation_space, c_layers, ff_layers):
        # features_dim = ultimo numero di neuroni dell'ultimo ff layer (es. 512)
        features_dim = ff_layers[-1][0]
        super().__init__(observation_space, features_dim)

        in_channels = observation_space.shape[0]

        convs = []
        for out_ch, k, s in c_layers:
            convs.append(nn.Conv2d(in_channels, out_ch, kernel_size=k, stride=s, padding=0))
            convs.append(nn.ReLU())
            in_channels = out_ch
        self.conv = nn.Sequential(*convs)

        # calcolo n_flatten esattamente come in ConvNetRL
        with th.no_grad():
            sample = th.zeros(1, *observation_space.shape)
            n_flatten = self.conv(sample).view(1, -1).shape[1]

        # FC + dropout come ff_layers
        fc_blocks = []
        in_dim = n_flatten
        for neurons, drop in ff_layers:
            fc_blocks.append(nn.Linear(in_dim, neurons))
            fc_blocks.append(nn.ReLU())
            fc_blocks.append(nn.Dropout(drop))
            in_dim = neurons
        self.fc = nn.Sequential(*fc_blocks)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        x = self.conv(obs)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
# 3) crea PPO (cnn policy va bene visto che obs è (C,H,W))
'''policy_kwargs = dict(
    features_extractor_class=SmallGridCNN,
    features_extractor_kwargs=dict(features_dim=256),
    normalize_images=False,   # ok lasciarlo
)'''

c_layers = [[64,3,2],[32,2,1],[32,2,1]]
ff_layers = [[512,0.2]]

policy_kwargs = dict(
    features_extractor_class=ConvNetRLFeatures,
    features_extractor_kwargs=dict(c_layers=c_layers, ff_layers=ff_layers),
    normalize_images=False,

    # IMPORTANTISSIMO se vuoi che sopra le feature ci siano *solo* le head lineari
    # (cioè niente MLP extra aggiuntivo)
    net_arch=dict(pi=[], vf=[]),
)

model = MaskablePPO(
    policy="CnnPolicy",
    env=env,
    verbose=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
    n_steps=2048,        # rollout length
    batch_size=64,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.05,
    policy_kwargs=policy_kwargs
)

callback = DDQNStyleLoggerCallback(
    print_freq_episodes=100,
    rolling_window=500,
    verbose=1
)

callback = AvgLifetimeCallback(
    verbose=1
)
callback = AvgLifetimeAndRewardCallback(print_every=100, window_after=500, window_size=500)

# 4) train
model.learn(total_timesteps=1000000000, callback=callback)


# 5) save
model.save("maskableppo_surfacecode_d5")
