# questo script contiene tutto il codice per l'agente custom in pytorch

# -------------import delle librerie----------------------------------------------------------------------------

import numpy as np
import random
import torch
from torch import nn, optim
import torch.nn.functional as F
import gymnasium as gym

from Function_Library import *
from Environments_no_referee_mixed_noise import *

import copy
import sys
import os
import time
import traceback

# -----------------------------------------------------------------------------------------------------
# default device : utile se si dimentica di inizializzarlo nello script
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- default settings -------------------------------------------------------------------
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
                "masked_greedy": False}

variable_configs = {"p_phys": 0.001,
                    "p_meas": 0.001,
                    "success_threshold": 1000,
                    "learning_starts": 1000,
                    "learning_rate": 0.0001,
                    "max_eps": 1.0,
                    "final_eps": 0.02,
                    "exploration_fraction": 75000,
                    "max_psi": 1.0,
                    "final_psi": 0.1,
                    "ppr_fraction": 50000, 
                    "target_network_update_freq": 5000,
                    "gamma": 0.99}

default_settings = {}
for key in fixed_configs.keys():
    default_settings[key] = fixed_configs[key]
for key in variable_configs.keys():
    default_settings[key] = variable_configs[key]

# (1)------------ replay buffer --> sostituisce SequentialMemory di keras-rl ----------------------------------------

class ReplayBuffer:
    """
    Memoria per le transizioni (s, a, r, s', terminated) con una capacità limitata.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        # lista semplice per le transizioni 
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, terminated):
        """salva una transizione nella memoria"""
        # se il buffer non è pieno, aggiungo, altrimenti sovrascrivo la posizione più vecchia
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        # salvo lo stato come array  --> sarà convertito in pytorch al campionamento
        self.buffer[self.position] = (state, action, reward, next_state, terminated)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """campiona un batch casuale di transizioni e le converte in tensori pytorch"""
        batch = random.sample(self.buffer, batch_size)
        
        # converto la lista di tuple (transizioni) in liste di elementi
        states, actions, rewards, next_states, terminateds = zip(*batch)
        
        # conversione in tensori pytorch
        state_tensor = torch.from_numpy(np.array(states)).float()
        action_tensor = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_state_tensor = torch.from_numpy(np.array(next_states)).float()
        # nb: gymnasium usa 'terminated' (booleano) che mappo a float per il calcolo
        terminated_tensor = torch.tensor(terminateds, dtype=torch.float32).unsqueeze(1)
        
        return state_tensor, action_tensor, reward_tensor, next_state_tensor, terminated_tensor

    def __len__(self):
        return len(self.buffer)


# (2) ------------ LinearAnnealedEpsGreedy --> Sostituzione di LinearAnnealedPolicy e EpsGreedyQPolicy ----------------------------------------------------------------

class LinearAnnealedEpsGreedyPolicy:
    """gestisce la logica di esplorazione epsilon-greedy con decay lineare"""
    def __init__(self, value_max, value_min, nb_steps, masked_greedy):
        self.value_max = value_max     # max_eps (1.0)
        self.value_min = value_min     # final_eps (0.02)
        self.nb_steps = nb_steps       # exploration_fraction (75000)
        self.masked_greedy = masked_greedy
        self.current_epsilon = value_max
        self.slope = (value_min - value_max) / nb_steps

    def update_epsilon(self, step_count):
        """aggiorna epsilon linearmente"""
        if step_count < self.nb_steps:
            self.current_epsilon = self.value_max + self.slope * step_count
        else:
            self.current_epsilon = self.value_min
    
    def select_action(self, q_values: torch.Tensor, legal_actions: set, is_training: bool, step_count: int):
        """sceglie un'azione in base alla policy corrente"""
        
        if is_training:
            self.update_epsilon(step_count) # nb: step_count deve essere passato esternamente
            epsilon = self.current_epsilon
        else:
            # "policy" per il test --> equivalente di GreedyQPolicy con o senza masking
            epsilon = 0.0  
        
        num_actions = q_values.size(-1)
        
        if is_training and np.random.rand() < epsilon:
            # esplorazione: selezione casuale tra azioni legali
            return random.choice(list(legal_actions))
        else:
            # sfruttamento: selezione greedy (con masking se richiesto)
            
            if self.masked_greedy or not is_training: 
                # applica il masking se richiesto o in test
                mask = torch.ones(num_actions, dtype=torch.bool).to(q_values.device)
                illegal_actions = [a for a in range(num_actions) if a not in legal_actions]
                mask[illegal_actions] = False
                
                # imposto i Q-value illegali a un valore molto basso prima di argmax
                masked_q_values = q_values.masked_fill(~mask, -1e9)
                return masked_q_values.argmax().item()
            else:
                # senza masking
                return q_values.argmax().item()

# nb: test_policy sarà un'istanza di questa classe con is_training=False --> utile nella funzione test_agent_performance

# (3) ------------------ ConvNetRl --> sostituisce build_convolutional_nn (custom model) di keras ------------------------------------------------------------

# Classe che definisce la rete neurale per l'agente, gestisce l'eventuale dueling
class ConvNetRL(nn.Module):
    """classe che definisce la rete neurale convoluzionale per l'agente DDQN"""
    def __init__(self, cc_layers, ff_layers, input_shape, num_actions, dueling=True):
        super(ConvNetRL, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        in_channels = input_shape[0] 
        for filt, ksize, stride in cc_layers:
            self.conv_layers.append(nn.Conv2d(in_channels, filt, kernel_size=ksize, stride=stride))
            in_channels = filt
        
        self.flatten = nn.Flatten()
        
        # calcolo della dimensione di input fully connected generato dalla cc precedente
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = dummy_input
            for conv in self.conv_layers:
                x = F.relu(conv(x))
            fc_input_dim = x.view(1, -1).shape[1] 

        # fully connected + dropout
        self.fc_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        in_dim = fc_input_dim
        for neurons, dropout_rate in ff_layers:
            self.fc_layers.append(nn.Linear(in_dim, neurons))
            self.dropouts.append(nn.Dropout(dropout_rate))
            in_dim = neurons
        
        # introduco la configurazione dueling vs standard
        # dueling calcola invece che direttamente i valori per le azioni, calcola in 2 "output" stato e bontà azione, da cui poi ricava Q(s,a)
        self.dueling = dueling
        
        if self.dueling:
            # flusso vantaggio -- azione A(s, a): output size = num_actions
            self.advantage_layer = nn.Linear(in_dim, num_actions)
            # flusso valore --  stato V(s): output size = 1
            self.value_layer = nn.Linear(in_dim, 1)
        else:
            # flusso standard : output size = num_actions
            self.out = nn.Linear(in_dim, num_actions)

    def forward(self, x):
        # strati convoluzionali e flatten
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        x = self.flatten(x)
        
        # strati fully connected
        for fc, drop in zip(self.fc_layers, self.dropouts):
            x = F.relu(fc(x))
            x = drop(x)
            
        # calcolo del Q-value --> differente per dueling vs standard
        if self.dueling:
            value = self.value_layer(x)       # V(s)
            advantage = self.advantage_layer(x) # A(s,a)
            
            # ricombinazione dueling: Q = V(s) + [A(s,a) - mean(A(s,a))]
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return q_values
        else:
            # standard DQN/DDQN
            return self.out(x)


# ------------------------ classe che implementa il modello delle reti neurali per gli static decoder portati a pytorch -----------------------------------------

# ho preso il decoder originale del repository, studiato la sua struttura nn, replicata questa classe torch e importato i pesi dal file .h5
# ho testato la compatibilià su una simulazione dell'env per le due reti nn 
class PyTorchStaticDecoder(nn.Module):
    """
    classe PyTorch statica che replica l'architettura della nn static decoder X err di keras fornito nel repository.
    usa i nomi delle chiavi (layers.0, layers.1, ...) attesi dal file .pth salvato.
    """
    def __init__(self):
        super().__init__()
        
        # definisco la lista dei layer lineari per replicare la struttura salvata
        self.layers = nn.ModuleList([
            nn.Linear(36, 1000),  # layers.0
            nn.Linear(1000, 500), # layers.1
            nn.Linear(500, 250),  # layers.2
            nn.Linear(250, 50),   # layers.3
            nn.Linear(50, 2)      # layers.4
        ])
        
    def forward(self, x):
        # layer 1 (ReLU)
        x = F.relu(self.layers[0](x))
        # layer 2 (ReLU)
        x = F.relu(self.layers[1](x))
        # layer 3 (ReLU)
        x = F.relu(self.layers[2](x))
        # layer 4 (ReLU)
        x = F.relu(self.layers[3](x))
        # layer 5 (Output - Logit)
        x = self.layers[4](x)
        return x
        
    def predict(self, x_np):
        self.eval() 
        device = next(self.parameters()).device 
        x = torch.from_numpy(x_np).float().to(device)
        if x.dim() == 1: x = x.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            
        return probs.cpu().numpy()

# versione per il decoder DP (4 output invece di 2)
class PyTorchStaticDecoderDP(nn.Module):
    """
    classe PyTorch statica che replica l'architettura della nn static decoder DP err di keras fornito nel repository.
    usa i nomi delle chiavi (layers.0, layers.1, ...) attesi dal file .pth salvato.
    """
    def __init__(self):
        super().__init__()
        
        # definisco la lista dei layer lineari per replicare la struttura salvata
        self.layers = nn.ModuleList([
            nn.Linear(36, 1000),  # layers.0
            nn.Linear(1000, 500), # layers.1
            nn.Linear(500, 250),  # layers.2
            nn.Linear(250, 50),   # layers.3
            nn.Linear(50, 4)      # layers.4
        ])
        
    def forward(self, x):
        # layer 1 (ReLU)
        x = F.relu(self.layers[0](x))
        # layer 2 (ReLU)
        x = F.relu(self.layers[1](x))
        # layer 3 (ReLU)
        x = F.relu(self.layers[2](x))
        # layer 4 (ReLU)
        x = F.relu(self.layers[3](x))
        # layer 5 (Output - Logit)
        x = self.layers[4](x)
        return x
        
    def predict(self, x_np):
        self.eval() 
        device = next(self.parameters()).device 
        x = torch.from_numpy(x_np).float().to(device)
        if x.dim() == 1: x = x.unsqueeze(0)
        
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            
        return probs.cpu().numpy()


# (4)------------------ PyTorchDDQNAgent --> sostituisce di DQNAgent di keras ---------------------------

# DDQN: modifica solo il metodo di calcolo di Q_target: 
# invece di utilizzare la stessa rete per azione e valore nel calcolo del target, sceglie l'azione ottimale da rete policy e la calcola valore con target
# disaccopiare questi due processi porta minor bias dovuto alla sovrastima di Q

class PyTorchDDQNAgent:
    def __init__(self, policy_net, target_net, optimizer, memory, all_configs, device=DEFAULT_DEVICE):

        self.policy_net = policy_net
        self.target_net = target_net
        self.device = device 
        self.optimizer = optimizer
        self.memory = memory
        self.criterion = nn.SmoothL1Loss()     # --> suggerita dal paper invece di nn.MSELoss()
        
        self.gamma = all_configs["gamma"]
        self.batch_size = all_configs["batch_size"]
        self.learning_starts = all_configs["learning_starts"]
        self.target_update_freq = all_configs["target_network_update_freq"]
               
    def learn(self):
        """
        esegue un singolo passo di ottimizzazione DDQN.
        restituisce la loss come float, o None se non si è appreso nulla (buffer non pieno o ancora in riscaldamento)
        """
        if len(self.memory) < self.learning_starts or len(self.memory) < self.batch_size:
            return None # ancora in fase di riscaldamento o buffer non pieno

        # campionamento dal buffer di replay 
        state_batch, action_batch, reward_batch, next_state_batch, terminated_batch = self.memory.sample(self.batch_size)
        
        # sposta i batch sul dispositivo (CPU/GPU)
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        terminated_batch = terminated_batch.to(self.device)

        # calcolo Q(s, a) corrente (valori stimati)
        # policy_net(state_batch) restituisce tutti i Q-values --> gather seleziona solo Q(s,a)
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # calcolo di max(Q_target(s', a')) (valori target)
        # target_net.max(1)[0] ottiene il Q-value massimo per lo stato successivo
        with torch.no_grad():
            # logica DDQN:
            # seleziono l'azione ottimale a' usando la policy_net (principale)
            # .argmax(1) restituisce l'indice dell'azione (a')
            # .unsqueeze(1) necessario per rendere compatibile con gather
            actions_from_policy_net = self.policy_net(next_state_batch).argmax(1).unsqueeze(1)
            
            # valuto il Q-value di quell'azione a' usando la target_net
            # .gather(1, actions_from_policy_net) estrae il Q(s', a') dalla target_net
            next_q_values = self.target_net(next_state_batch).gather(1, actions_from_policy_net)

            # calcolo del target Q-value           
            # imposto il target a 0 per gli stati terminati (terminated_batch = 1 se terminato)
            target_q_values = reward_batch + self.gamma * next_q_values * (1 - terminated_batch)

        # calcolo della loss secondo criterion scelto
        loss = self.criterion(current_q_values, target_q_values)

        # backward pass e aggiornamento
        self.optimizer.zero_grad()
        loss.backward()
        # clipping del gradiente --> buona pratica in RL per evitare esplosioni
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

    def update_target_net(self):
        """aggiorna i pesi della rete target."""
        self.target_net.load_state_dict(self.policy_net.state_dict())


# (5) ----------------- funzioni di training e test dell'agente ------------------------------------------------------------------------------- 

# funzione di training dell'agente DDQN: 

# esisteva una versione V1 ma con un piccolo errore logico: non resettava env, reward per i primi "rolling_average_length" 
# --> giustificata la iniziale vita media esagerata di alcune sessioni di training
# questa versione V2 corregge questo errore

def trainV2(agent, env, policy, policy_net, memory, all_configs=default_settings, device=DEFAULT_DEVICE, save_buffer = False):
    '''
    funzione che allena l'agente. prevede monitoraggio a terminale, salvataggio di log, checkpoint
    e condizione di early stopping basata su raggiungimento di threshold e non-miglioramento di avg_lifetime
    Args:
        agent (PyTorchDDQNAgent): l'agente da allenare
        env (gym.Env): l'ambiente di simulazione
        policy (LinearAnnealedEpsGreedyPolicy): la policy di esplorazione epsilon
        policy_net (nn.Module): la rete neurale principale dell'agente
        memory (ReplayBuffer): il buffer di replay
        all_configs (dict): dizionario con tutte le configurazioni
        device (torch.device): dispositivo su cui eseguire i calcoli (CPU o GPU)
        save_buffer : variabile booleana per stabilire se salvare o no il buffer alla fine del training
    Returns:
        best_model_path (str): il percorso del miglior modello salvato
    '''
    # inizializzazione dell'env 
    current_state, info = env.reset() 

    # variabili generali di tracking, metriche e contatori
    episode_reward = 0
    episode_lifetime = 0      # --> inizializzo anche la qubit lifetime, da estrarre per ogni episodio da env tramite env.lifetime
    episode_step_count = 0  
    episode_count = 0
    step_count = 0
    loss = np.nan 
    # nb: la logica di conteggio lifetime è contenuta nella classe env, non è necessario 
    # quindi aggiornarla per ogni step(), basta salvarla da env alla fine dell'episodio
    all_lifetimes = []           
    best_avg_lifetime = -float('inf')
    all_rewards = []
    best_avg_reward = -float('inf')
    patience_counter = 0

    # creo il percorso di logging, checkpoint e salvataggio buffer
    logging_directory, logging_path, best_model_path = setup_logging_and_checkpointing(all_configs)
    final_model_path = os.path.join(logging_directory, "final_policy_net.pth") 
    memory_path = os.path.join(logging_directory, "memory.pkl")

    # misura temporale
    start_time = time.time()

    # variabile per tracciare se l'addestramento è stato completato per maxtimesteps o per early stopping
    training_completed = False

    try:
        print(f"Inizio Addestramento su {all_configs['max_timesteps']} passi. Log in: {logging_directory}")

        for step_count in range(1, all_configs["max_timesteps"] + 1):

            # selezione dell'azione (epsilon-greedy)
            state_tensor = torch.from_numpy(current_state).float().to(device).unsqueeze(0)
            q_values = policy_net(state_tensor).squeeze(0) 

            action = policy.select_action(
                q_values=q_values, 
                legal_actions=env.legal_actions, 
                is_training=True,
                step_count=step_count
            )

            # interazione con l'env 
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated # condizione di fine episodio adattata a gymnasium

            # aggiorno gli step per episodio
            episode_step_count += 1

            # memorizzo la transizione
            memory.push(current_state, action, reward, next_state, terminated) 
            current_state = next_state
            episode_reward += reward

            # aggiornamento dell'agente : avviene se ho superato il warmup e sono in train freq
            if step_count >= all_configs["learning_starts"] and step_count % all_configs["train_freq"] == 0:
                new_loss = agent.learn()
                if new_loss is not None: # aggiorno loss solo se è avvenuto l'apprendimento --> in learn è contenuta la logica di warm up
                    loss = new_loss
                else:
                    print(f"DEBUG: learn() restituito None a step {step_count}. Len memory: {len(memory)}")

            # aggiornamento del target_net --> se sono in freq di update
            if step_count % all_configs["target_network_update_freq"] == 0:
                agent.update_target_net()

            # inizio della gestione di fine episodio 
            if done:
                episode_count += 1
                episode_lifetime = env.lifetime
                all_lifetimes.append(episode_lifetime)
                all_rewards.append(episode_reward)

                # logging :
                # calcolo la media mobile solo dopo aver raccolto abbastanza dati
                if len(all_rewards) >= all_configs["rolling_average_length"]:
                    current_reward_avg = np.mean(all_rewards[-all_configs["rolling_average_length"]:])
                    current_reward_std = np.std(all_rewards[-all_configs["rolling_average_length"]:])
                    current_loss_log = loss if not np.isnan(loss) else None
                    current_loss_print = f"{loss:.4f}" if not np.isnan(loss) else "N/A"
                    current_lifetime_avg = np.mean(all_lifetimes[-all_configs["rolling_average_length"]:])
                    current_lifetime_std = np.std(all_lifetimes[-all_configs["rolling_average_length"]:])
                else : # altrimenti uso quelli che ho
                    current_reward_avg = np.mean(all_rewards)
                    current_reward_std = np.std(all_rewards)
                    current_loss_log = loss if not np.isnan(loss) else None
                    current_loss_print = f"{loss:.4f}" if not np.isnan(loss) else "N/A"
                    current_lifetime_avg = np.mean(all_lifetimes)
                    current_lifetime_std = np.std(all_lifetimes)

                # preparo i dati di log
                log_data = {
                    "step": step_count,
                    "episode": episode_count,
                    "episode_reward" : episode_reward,
                    "episode_step_count" : episode_step_count,
                    "reward_mean": current_reward_avg,
                    "reward_std": current_reward_std,
                    "episode_lifetime" : episode_lifetime,
                    "lifetime_mean" : current_lifetime_avg,
                    "lifetime_std" : current_lifetime_std,
                    "loss": current_loss_log,
                    "epsilon": policy.current_epsilon 
                }
                # eseguo il log
                update_json_history(logging_path, log_data) 

                # stampo a console se sono in print freq
                if episode_count % all_configs["print_freq"] == 0:
                    time_elapsed = time.time() - start_time # aggiornamento per il tempo
                    # stringa di stampa dettagliata
                    output = f"""
--------------------------------
Episode: {episode_count}
Step: {step_count}/{all_configs["max_timesteps"]}
Episode Step : {episode_step_count}
Episode Reward : {episode_reward}
Avg Reward (Last {all_configs["rolling_average_length"]} Ep): {current_reward_avg:.2f}
Best Avg Reward: {best_avg_reward:.2f}
Episode Lifetime : {episode_lifetime}
Avg Lifetime (Last {all_configs["rolling_average_length"]} Ep): {current_lifetime_avg:.2f}
Best Avg Lifetime : {best_avg_lifetime:.2f}
Success threshold : {all_configs["success_threshold"]}
Patience Counter: {patience_counter}/{all_configs["stopping_patience"]}
Metrics: Loss: {current_loss_print}, Epsilon: {policy.current_epsilon:.4f}
Total Training Time: {time_elapsed:.2f}s
--------------------------------
                                  """ 
                    print(output)
                # aggiorno la miglior reward:
                if current_reward_avg > best_avg_reward :
                    best_avg_reward = current_reward_avg
                # early stopping e checkpoint
                if current_lifetime_avg > best_avg_lifetime:
                    # salvo il nuovo best model --> checkpoint
                    torch.save(policy_net.state_dict(), best_model_path)
                    if best_avg_lifetime != -float('inf'):
                        print(f"*** Nuovo Best Model salvato! Avg Lifetime: {current_lifetime_avg:.2f} ***")
                    # aggiorno la miglior lifetime
                    best_avg_lifetime = current_lifetime_avg
                    # resetto il contatore pazienza
                    patience_counter = 0
                # incremento il contatore pazienza se non miglioro e ho superato la soglia di successo
                elif best_avg_lifetime >= all_configs['success_threshold']:
                    patience_counter += 1
                    if patience_counter >= all_configs["stopping_patience"]:
                        print(f"--- Stopping early. Pazienza esaurita. Best Avg Lifetime: {best_avg_lifetime:.2f} ---")
                        training_completed = False # il train è finito per early stopping
                        # --> early stopping: uso 'break' per uscire dal loop 'for'
                        break  

                # reset per il nuovo episodio --> sotto condizione di done=True
                current_state, info = env.reset()
                episode_reward = 0
                episode_step_count = 0

        # se esco da ciclo for senza break significa che ho completato per maxtimesteps
        training_completed = True
    except Exception as e:
        # gestione di eventuali eccezioni critiche durante l'addestramento
        print(f"ERRORE CRITICO DURANTE IL TRAINING: {e}")    
        traceback.print_exc() # stampo la traccia completa dell'errore
        training_completed = False
    finally:
        # questo blocco viene eseguito sempre (dopo break, dopo eccezione, o a fine loop)
        if training_completed:
            print("Addestramento completato: Max Timesteps raggiunto.")

        # salvataggio del risultato finale
        torch.save(policy_net.state_dict(), final_model_path)
        print(f"Ultimo modello salvato in: {final_model_path}")

        # salvataggio del ReplayBuffer
        if save_buffer:
            save_replay_buffer(memory, memory_path)
        
        print(f'Fine addestramento. Log in: {logging_directory}')
        
        return best_model_path if best_avg_lifetime > -float('inf') else None


# funzione di test dell'agente post training:

def test_agent_performance(env, policy_net, policy, test_episodes=101, device=DEFAULT_DEVICE, log=False, run_name=None):
    """
    Esegue il testing deterministico (greedy) dell'agente DDQN su un numero specificato di episodi.
    Se log == True, salva il log del test in un apposito file json, altrimenti restituisce
     media e lista valori (utile per test veritieri su molti più episodi)
    Durante il test, l'agente seleziona sempre l'azione con il valore Q più alto (epsilon=0).
    Args:
        env: L'istanza dell'ambiente.
        policy_net: L'istanza della rete neurale allenata in mode .eval().
        policy: L'istanza della politica di esplorazione, usata con is_training=False.
        test_episodes (int): Numero di episodi da eseguire per il test.
        device (torch.device): Dispositivo (CPU/GPU) su cui eseguire il calcolo.
        log : parametro booleano per stabilire se loggare il test o no
        run_name : str contenente il nome della run relativa alla policy testata es: DDQN_d5_...
    Returns:
        avg_reward (float): Ricompensa media ottenuta durante il test.
        std_reward (float): Deviazione standard della ricompensa.
        test_rewards (list): Lista delle ricompense ottenute in ogni episodio di test.
        avg_lifetime (float): Lifetime media ottenuta durante il test.
        std_lifetime (float): Deviazione standard della lifetime.
        test_lifetimes (list): Lista delle lifetimes ottenute in ogni episodio di test.
        or
        log_path : percorso al file di log
    """
    if log :
        if run_name == None:
            print("Errore: per loggare il test è necessario inserire la run relativa alla policy testata.")
            return None 
        else :
            log_directory = 'test_runs'
            os.makedirs(log_directory, exist_ok=True)
            log_file = 'test_policy_'+ run_name +'.json'
            log_path = os.path.join(log_directory, log_file)

    policy_net.to(device)
    test_rewards = []
    test_lifetimes = []
    
    print(f"\n--- Inizio Test Deterministico (Episodi: {test_episodes}) ---")

    for episode in range(1, test_episodes + 1):

        current_state, info = env.reset()
        episode_reward = 0
        done = False
        ep_step = 0
        
        while not done:
            ep_step += 1
            # preparo lo stato per l'input della rete neurale
            state_tensor = torch.from_numpy(current_state).float().to(device).unsqueeze(0)
            
            with torch.no_grad(): # nessun calcolo di gradiente durante il test
                q_values = policy_net(state_tensor).squeeze(0)
            
            # selezione dell'azione deterministica (greedy)
            action = policy.select_action(
                q_values=q_values, 
                legal_actions=env.legal_actions, # usa azioni legali se implementate, solo se masked_greedy=True
                is_training=False, 
                step_count=None      # valore casuale, tanto in questo caso non serve
            )

            # interazione con env
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            current_state = next_state
            episode_reward += reward
        
        test_rewards.append(episode_reward)
        test_lifetimes.append(env.lifetime)
        
        if not log:
            print(f'tested episode : {episode}, steps : {ep_step}, reward : {episode_reward}, lifetime : {env.lifetime}')
        else:
            log_data = {
                "episode" : episode,
                "step" : ep_step,
                "reward" : episode_reward,
                "lifetime" : env.lifetime
            }
            update_json_history(log_path, log_data)
            # stampo con meno frequenza --> mi aspetto un numero molto maggiore di episodi testati
            if episode % 10 == 0 :
                print(f'tested episode : {episode}, steps : {ep_step}, reward : {episode_reward}, lifetime : {env.lifetime}')

    avg_reward = np.mean(test_rewards)
    std_reward = np.std(test_rewards)
    avg_lifetime = np.mean(test_lifetimes)
    std_lifetime = np.std(test_lifetimes)

    print("\n--- Risultati del Test ---")
    print(f"Ricompensa Media su {test_episodes} episodi: {avg_reward:.2f} +- {std_reward:.2f}")
    print(f"Lifetime Media su {test_episodes} episodi: {avg_lifetime:.2f} +- {std_lifetime:.2f}")
    
    if log :
        return log_path
    else:
        return avg_reward, std_reward, test_rewards, avg_lifetime, std_lifetime, test_lifetimes


# (6) -------------------------------- implementazione di Probabilistic Policy Reuse (PPR) -------------------------------------------------

# funzioni ausiliarie per PPR
def linear_update_psi(step_count, all_configs):
    """aggiorna il valore di psi (parametro che gestisce la dinamica di reuse) con decadimento lineare"""
   
    slope = (all_configs["final_psi"] - all_configs["max_psi"]) / all_configs["ppr_fraction"]
    if step_count < all_configs["ppr_fraction"]:
        current_psi = all_configs["max_psi"] + slope * step_count
    else :
        current_psi = all_configs["final_psi"]
    return current_psi


# funzione head di training con PPR :

def train_with_ppr(agent, env, policy, policy_net, memory, library, all_configs=default_settings, tau_increment=0.0001, device=DEFAULT_DEVICE, save_buffer = False):
    '''
    funzione che allena l'agente tramite PPR: all'inizio di ogni episodio estrae una policy dalla libreria secondo la distribuzione softmax
    basata sulle rewards comulative ottenute negli episodi precedenti. Pi-exploration: durante l'episodio, con probabilità psi sceglie l'azione greedy della policy (se k>0)
    estratta, altrimenti segue la normale logica eps-greedy della rete corrente. psi decresce linearmente da max_psi a min_psi in una frazione del training.
    implementa monitoraggio a terminale, salvataggio di log, checkpoint e condizione di early stopping basata su raggiungimento di threshold e non-miglioramento di avg_lifetime
    Args:
        agent: istanza di PyTorchDDQNAgent
        env: istanza dell'ambiente 
        policy: istanza di LinearAnnealedEpsGreedyPolicy
        policy_net: istanza della rete neurale in allenamento
        memory: istanza di ReplayBuffer
        all_configs: dizionario di configurazione --> deve contenere anche i parametri per PPR (max_psi, final_psi e ppr_fraction)
        library: lista di modelli di policy pre-addestrate 
        tau_increment: parametro di incremento per episode per la temperatura tau (stabilisce nel tempo quanto rapidamente il meccanismo di estrazione della policy tramite softmax diventi greedy) 
        device: dispositivo (CPU/GPU) su cui eseguire il calcolo
        save_buffer : variabile booleana per stabilire se salvare o no il buffer alla fine del training
    Returns:
        best_model_path: percorso del miglior modello salvato durante l'addestramento
    '''
     # inizializzo le variabili della logica PPR:
    current_psi = all_configs["max_psi"]
    current_tau = 0.0
    tau_increment = tau_increment
    # array di policy passate --> è un array di modelli statici
    policy_library = library
    # array delle rewards comulative per policy reuse e contatore di utilizzo delle policy 
    W = np.zeros(len(policy_library) + 1) # +1 per la policy corrente L0
    U = np.zeros(len(W))
    # inizializzo le variabili generali, logging e checkpointing:
    episode_reward = 0
    episode_lifetime = 0
    episode_step_count = 0
    episode_count = 0
    step_count = 0
    loss = np.nan
    all_lifetimes = []
    best_avg_lifetime = -float('inf')
    all_rewards = []
    best_avg_reward = -float('inf')
    patience_counter = 0
    # contatore azioni reuse
    reuse_action_count = 0

    # variabile per tracciare fine per early stopping
    training_completed = False

    # creo percorso di logging e checkpointing
    logging_directory, logging_path, best_model_path = setup_logging_and_checkpointing_ppr(all_configs)
    final_model_path = os.path.join(logging_directory, "final_policy_net.pth")
    memory_path = os.path.join(logging_directory, "memory.pkl") # --> salvataggio del replay buffer : in questo caso non dovrebbe servire, però lo implemento per ora

    # misura temporale
    start_time = time.time()

    # inizialmente imposto done=True per forzare la scelta della policy al primo step
    done = True
    
    try : 
        print(f"Inizio Addestramento con PPR su {all_configs['max_timesteps']} passi. Log in: {logging_directory}")
        for step_count in range(1, all_configs["max_timesteps"] +1):
        
            # se l'episodio è terminato o non è ancora iniziato, devo scegliere una policy 
            if done:
                current_state, info = env.reset()
                episode_reward = 0
                episode_step_count = 0
                reuse_action_count = 0

                # calcolo le probabilità softomax P
                P = softmax(W, current_tau) 
                # scelgo l'indice k (0 per L0, k>0 per Lk)
                k = np.random.choice(len(W), p=P) 
                # inizializzo la policy di reuse
                # nb: l'indice adeguato in policy_library è k-1 perchè L0 non è contenuta
                Li = policy_library[k-1] if k > 0 else None
                done = False

            # preparo lo stato per essere input di una rete neurale
            state_tensor = torch.from_numpy(current_state).float().to(device).unsqueeze(0)

            # scelta dell'azione da eseguire secondo logica pi-exploration con la policy Li di questo episodio:
            action = None
            # nb: invece che dividere la scelta per eps-greedy nel caso in cui k==0 oppure reuse fallisce (rand < psi),
            # estraggo subito rand (se k!=0) e poi nel caso prenderò eps-greedy
            if k > 0:
                if np.random.rand() < current_psi:
                    # seleziono l'azione greedy dalla policy Li
                    with torch.no_grad(): 
                        q_values_Li = Li(state_tensor).squeeze(0)

                    action = q_values_Li.argmax().item()
                    
                    reuse_action_count += 1

                    # aggiorno psi ed eps per questo step in cui ho reuse
                    current_psi = linear_update_psi(step_count, all_configs)
                    policy.update_epsilon(step_count)         

            # se non è ancora stata scelta un'azione o se k==0 (policy corrente)
            if action is None:
                # seleziono l'azione secondo eps-greedy dalla policy corrente
                q_values = policy_net(state_tensor).squeeze(0) 

                # nb: la selezione dell'azione e l'aggiornamento di epsilon sono gestiti internamente alla classe policy
                action = policy.select_action(
                    q_values=q_values, 
                    legal_actions=env.legal_actions, 
                    is_training=True,
                    step_count=step_count
                    )
                # aggiorno solo psi --> eps è aggiornato in "select_action" 
                current_psi = linear_update_psi(step_count, all_configs)


            # interazione con l'env: eseguo l'azione        
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # aggiorno gli step per episodio 
            episode_step_count += 1

            # memorizzo la transizione nel replay buffer 
            memory.push(current_state, action, reward, next_state, terminated)

            # aggiorno le variabili di stato
            current_state = next_state
            episode_reward += reward

            # apprendimento ed ottimizzazione dell'agente        
            if step_count >= all_configs["learning_starts"] and step_count % all_configs["train_freq"] == 0:
                new_loss = agent.learn()
                if new_loss is not None: # Aggiorna loss solo se è avvenuto l'apprendimento --> in learn è contenuta la logica di warm up
                    loss = new_loss
                else:
                    print(f"DEBUG: learn() restituito None a step {step_count}. Len memory: {len(memory)}")

            if step_count % all_configs["target_network_update_freq"] == 0:
                agent.update_target_net()

            # gestisco fine episodio
            if done:
                # aggiorno le rewards cumulative (W) e l'utilizzo (U) della policy estratta k
                W[k] = (W[k] * U[k] + episode_reward) / (U[k] + 1)
                U[k] += 1
                # aggiorno la temperatura per la prossima estrazione di policy
                current_tau += tau_increment

                # calcolo dei parametri di logging, logging, checkpointing e early stopping
                episode_count += 1
                episode_lifetime = env.lifetime
                all_lifetimes.append(episode_lifetime)
                all_rewards.append(episode_reward)

                # calcolo metriche di logging solo se ho abbastanza dati:
                if len(all_rewards) >= all_configs["rolling_average_length"]:
                    current_reward_avg = np.mean(all_rewards[-all_configs["rolling_average_length"]:])
                    current_reward_std = np.std(all_rewards[-all_configs["rolling_average_length"]:])
                    current_loss_log = loss if not np.isnan(loss) else None
                    current_loss_print = f"{loss:.4f}" if not np.isnan(loss) else "N/A"
                    current_lifetime_avg = np.mean(all_lifetimes[-all_configs["rolling_average_length"]:])
                    current_lifetime_std = np.std(all_lifetimes[-all_configs["rolling_average_length"]:])
                else : # altrimenti uso quelli che ho come posso
                    current_reward_avg = np.mean(all_rewards)
                    current_reward_std = np.std(all_rewards)
                    current_loss_log = loss if not np.isnan(loss) else None
                    current_loss_print = f"{loss:.4f}" if not np.isnan(loss) else "N/A"
                    current_lifetime_avg = np.mean(all_lifetimes)
                    current_lifetime_std = np.std(all_lifetimes)

                # log dei dati
                log_data = {
                    "step": step_count,
                    "episode": episode_count,
                    "policy_index_k": k,
                    "episode_step_count": episode_step_count,
                    "reuse_actions": reuse_action_count,
                    "episode_reward" : episode_reward,
                    "reward_mean": current_reward_avg,
                    "reward_std": current_reward_std,
                    "episode_lifetime" : episode_lifetime,
                    "lifetime_mean" : current_lifetime_avg,
                    "lifetime_std" : current_lifetime_std,
                    "loss": current_loss_log,
                    "epsilon": policy.current_epsilon,
                    "psi": current_psi,
                }
                update_json_history(logging_path, log_data)

                if episode_count % all_configs["print_freq"] == 0:
                    time_elapsed = time.time() - start_time # aggiornamento per il tempo
                    # stringa di stampa dettagliata
                    output = f"""
---------------------------------
Step : {step_count}/{all_configs["max_timesteps"]}
Episode : {episode_count}
Policy Index k : {k}
Episode Step : {episode_step_count} with {reuse_action_count} "reuse" actions
Episode Reward : {episode_reward}
Avg Reward (Last {all_configs["rolling_average_length"]} Ep) : {current_reward_avg:.2f}
Best Avg Reward : {best_avg_reward:.2f}
Episode Lifetime : {episode_lifetime}
Avg Lifetime (Last {all_configs["rolling_average_length"]} Ep) : {current_lifetime_avg:.2f}
Best Avg Lifetime : {best_avg_lifetime:.2f}
Success threshold : {all_configs["success_threshold"]}
Patience Counter : {patience_counter}/{all_configs["stopping_patience"]}
Metrics : Loss: {current_loss_print}, Epsilon: {policy.current_epsilon:.4f}, Psi: {current_psi:.4f}
Total Training Time : {time_elapsed:.2f}s
---------------------------------
                              """
                    print(output)
                # aggiorno la miglior reward solo nel momento in cui la media mobile è vera e non fittizia
                if current_reward_avg > best_avg_reward and len(all_rewards) >= all_configs["rolling_average_length"]:
                    best_avg_reward = current_reward_avg
                # early stopping e checkpointing (nel momento in cui la media mobile è vera e non fittizia) 
                if current_lifetime_avg > best_avg_lifetime and len(all_lifetimes) >= all_configs["rolling_average_length"]:
                    # salvo il nuovo best model --> checkpoint
                    torch.save(policy_net.state_dict(), best_model_path)
                    if best_avg_lifetime != -float('inf'):
                        print(f"*** Nuovo Best Model salvato! Avg Lifetime: {current_lifetime_avg:.2f} ***")
                    # aggiorno i nuovi record
                    best_avg_lifetime = current_lifetime_avg
                    # resetto il contatore pazienza
                    patience_counter = 0
                elif best_avg_lifetime >= all_configs['success_threshold']:
                    patience_counter += 1
                    # early stopping
                    if patience_counter >= all_configs["stopping_patience"]:
                        print(f"--- Stopping early. Pazienza esaurita. Best Avg Lifetime: {best_avg_lifetime:.2f} ---")
                        training_completed = False
                        break
        
        # se esco da ciclo senza break, ho completato l'allenamento per maxtimesteps
        training_completed = True
    except Exception as e:
        # gestione di eventuali eccezioni critiche durante l'addestramento
        print(f"ERRORE CRITICO DURANTE IL TRAINING: {e}")    
        traceback.print_exc() # Stampa la traccia completa dell'errore
        training_completed = False
    finally:
        # questo blocco viene eseguito sempre (dopo break, dopo eccezione, o a fine loop)
        if training_completed:
            print("Addestramento completato: Max Timesteps raggiunto.")
        # salvataggio del risultato finale (raggiunto per max timesteps o per early stopping)
        torch.save(policy_net.state_dict(), final_model_path)
        print(f"Ultimo modello salvato in: {final_model_path}")
        
        # salvataggio del ReplayBuffer
        if save_buffer:
            save_replay_buffer(memory, memory_path)
        
        print(f'Fine addestramento. Log in: {logging_directory}')

        return best_model_path if best_avg_lifetime > -float('inf') else None