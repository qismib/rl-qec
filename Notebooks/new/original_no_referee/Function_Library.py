# ------------ Helper Functions -------------------------------------------------------------------------
#
# This file provides all necessary helper functions. 
#
# ----- (0) Imports --------------------------------------------------------------------------------------

import random
import numpy as np
import json
import copy
import sys
import os
import datetime
import time
import pickle
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- (1) Functions -------------------------------------------------------------------------------------
# funzioni utilizzate nello script env

def generateSurfaceCodeLattice(d):
    """"
    This function generates a distance d square surface code lattice. in particular, the function returns 
    an array which, for each physical qubit, details the code-stabilizers supported on that qubit. To be more
    precise:
    
     - qubits[i,j,:,:] is a 4x3 array describing all the stabilizers supported on physical qubit(i,j)
           -> for the surface code geometry each qubit can support up to 4 stabilizers
     - qubits[i,j,k,:] is a 3-vector describing the k'th stabilizer supported on physical qubit(i,j)
           -> qubits[i,j,k,:] = [x_lattice_address, y_lattice_address, I or X or Y or Z]
    
    :param: d: The lattice width and height (or, equivalently, for the surface code, the code distance)
    :return: qubits: np.array listing and describing the code-stabilizers supported on each qubit
    """
    
    if np.mod(d,2) != 1:
        raise Exception("for the surface code d must be odd!")

    # creazione dell'array multidim --> gestione delle coord dello stabilizer e del tipo    
    qubits = [ [ [
                   [ x, y, ((x+y)%2)*2+1],
                   [ x, y+1, ((x+y+1)%2)*2+1],
                   [ x+1, y, ((x+1+y)%2)*2+1],
                   [ x+1, y+1, ((x+1+y+1)%2)*2+1]
                ] for y in range(d)] for x in range(d)]
    qubits = np.array(qubits)
    # gestione delle condizioni al contorno della griglia 
    for x in range(d):
        for y in range(d):
            for k in range(4):
                if (qubits[x,y,k,0] == 0 and qubits[x,y,k,1]%2 == 0):
                    qubits[x,y,k,2] = 0
                if (qubits[x,y,k,0] == d and qubits[x,y,k,1]%2 == 1):
                    qubits[x,y,k,2] = 0
                    
                if (qubits[x,y,k,1] == 0 and qubits[x,y,k,0]%2 == 1):
                    qubits[x,y,k,2] = 0
                if (qubits[x,y,k,1] == d and qubits[x,y,k,0]%2 == 0):
                    qubits[x,y,k,2] = 0
    return qubits


def multiplyPaulis(a,b):
    """"
    A simple helper function for multiplying Pauli Matrices. Returns ab.
    :param: a: an int in [0,1,2,3] representing [I,X,Y,Z]
    :param: b: an int in [0,1,2,3] representing [I,X,Y,Z]
    """
    
    out = [[0,1,2,3],[1,0,3,2],[2,3,0,1],[3,2,1,0]]
    return out[int(a)][int(b)]


# 2) Error generation

def generate_error(d,p_phys,error_model):
    """"
    This function generates an error configuration, via a single application of the specified error channel, on a square dxd lattice.
    
    :param: d: The code distance/lattice width and height (for surface/toric codes)
    :param: p_phys: The physical error rate.
    :param: error_model: A string in ["X", "DP", "IIDXZ"] indicating the desired error model.
    :return: error: The error configuration
    """
    
    if error_model == "X":
        return generate_X_error(d,p_phys)
    elif error_model == "DP":
        return generate_DP_error(d,p_phys)
    elif error_model == "IIDXZ":
        return generate_IIDXZ_error(d,p_phys)
        

def generate_DP_error(d,p_phys):
    """"
    This function generates an error configuration, via a single application of the depolarizing noise channel, on a square dxd lattice.
    
    :param: d: The code distance/lattice width and height (for surface/toric codes)
    :param: p_phys: The physical error rate.
    :return: error: The error configuration
    """

    error = np.zeros((d,d),int) 
    for i in range(d): 
        for j in range(d):
            p = 0
            if np.random.rand() < p_phys:
                p = np.random.randint(1,4)
                error[i,j] = p
                
    return error

def generate_X_error(d,p_phys):
    """"
    This function generates an error configuration, via a single application of the bitflip noise channel, on a square dxd lattice.
    
    :param: d: The code distance/lattice width and height (for surface/toric codes)
    :param: p_phys: The physical error rate.
    :return: error: The error configuration
    """
    
    
    error = np.zeros((d,d),int) 
    for i in range(d): 
        for j in range(d):
            p = 0
            if np.random.rand() < p_phys:
                error[i,j] = 1
    
    return error
                
def generate_IIDXZ_error(d,p_phys):
    """"
    This function generates an error configuration, via a single application of the IIDXZ noise channel, on a square dxd lattice.
    
    :param: d: The code distance/lattice width and height (for surface/toric codes)
    :param: p_phys: The physical error rate.
    :return: error: The error configuration
    """
    
    error = np.zeros((d,d),int)
    for i in range(d):
        for j in range(d):
            X_err = False
            Z_err = False
            p = 0
            if np.random.rand() < p_phys:
                X_err = True
                p = 1
            if np.random.rand() < p_phys:
                Z_err = True
                p = 3
            if X_err and Z_err:
                p = 2

            error[i,j] = p
    
    return error

def generate_surface_code_syndrome_NoFT_efficient(error,qubits):
    """"
    This function generates the syndrome (violated stabilizers) corresponding to the input error configuration, 
    for the surface code.
    
    :param: error: An error configuration on a square lattice
    :param: qubits: The qubit configuration
    :return: syndrome: The syndrome corresponding to input error
    """
    
    d = np.shape(error)[0]
    syndrome = np.zeros((d+1,d+1),int)

    for i in range(d): 
        for j in range(d):
            if error[i,j] != 0:
                for k in range(qubits.shape[2]):
                    if qubits[i,j,k,2] != error[i,j] and qubits[i,j,k,2] != 0:
                        a = qubits[i,j,k,0]
                        b = qubits[i,j,k,1]
                        syndrome[a,b] = 1 - syndrome[a,b]
                        
    return syndrome

def generate_faulty_syndrome(true_syndrome, p_measurement_error):
    """"
    This function takes in a true syndrome, and generates a faulty syndrome according to some
    given probability of measurement errors.
    
    :param: true_syndrome: The original perfect measurement syndrome
    :return: p_measurement_error: The probability of measurement error per stabilizer
    :return: faulty_syndrome: The faulty syndrome
    """
    
    faulty_syndrome = np.zeros(np.shape(true_syndrome),int)

    # First we take care of the "bulk stabilizers"
    for row in range(1, true_syndrome.shape[0]-1):
        for col in range(1,true_syndrome.shape[1]-1):
            if np.random.rand() < p_measurement_error:
                faulty_syndrome[row,col] = 1 - true_syndrome[row,col]
            else:
                faulty_syndrome[row,col] = true_syndrome[row,col]

    # Now we take care of the boundary stabilizers
    row = 0
    for col in [2*x +1 for x in range(int(true_syndrome.shape[0]/2 - 1))]:
        if np.random.rand() < p_measurement_error:
                faulty_syndrome[row,col] = 1 - true_syndrome[row,col]
        else:
            faulty_syndrome[row,col] = true_syndrome[row,col]
    row = true_syndrome.shape[0] - 1
    for col in [2*x + 2 for x in range(int(true_syndrome.shape[0]/2 - 1))]:
        if np.random.rand() < p_measurement_error:
                faulty_syndrome[row,col] = 1 - true_syndrome[row,col]
        else:
            faulty_syndrome[row,col] = true_syndrome[row,col]

    col = 0
    for row in [2*x + 2 for x in range(int(true_syndrome.shape[0]/2 - 1))]:
        if np.random.rand() < p_measurement_error:
                faulty_syndrome[row,col] = 1 - true_syndrome[row,col]
        else:
            faulty_syndrome[row,col] = true_syndrome[row,col]
    col = true_syndrome.shape[0] - 1
    for row in [2*x +1 for x in range(int(true_syndrome.shape[0]/2 - 1))]:
        if np.random.rand() < p_measurement_error:
                faulty_syndrome[row,col] = 1 - true_syndrome[row,col]
        else:
            faulty_syndrome[row,col] = true_syndrome[row,col]
  
    return faulty_syndrome


def obtain_new_error_configuration(old_configuration,new_gates):
    """"
    This function generates a new error configuration out of an old configuration and a new configuration,
     which might arise either from errors or corrections.
    
    :param: old_configuration: An error configuration on a square lattice
    :param: new_gates: An error configuration on a square lattice
    :return: new_configuration: The resulting error configuration
    """
    
    new_configuration = np.zeros(np.shape(old_configuration))
    for row in range(new_configuration.shape[0]):
        for col in range(new_configuration.shape[1]):
            new_configuration[row,col] = multiplyPaulis(new_gates[row,col], old_configuration[row,col])
            
    return new_configuration

def index_to_move(d,move_index,error_model,use_Y=True):
    """"
    Given an integer index corresponding to a Pauli flip on a physical data qubit, this
    function generates the lattice representation of the move.
    
    :param: d: The code distance
    :param: move_index: The integer representation of the Pauli Flip
    :param: error_model: A string in ["X", "DP", "IIDXZ"] indicating the desired error model.
    :param: use_Y: a boolean indicating whether or not Y flips are allowed
    :return: new_move: A lattice representation of the desired move.
    """

    new_move = np.zeros((d,d))
    
    if error_model == "X":
        if move_index < (d**2):

            move_type = 1
            new_index = move_index
            row = new_index/d
            col = new_index%d

            new_move[int(row),int(col)] = move_type

    elif error_model == "DP":
        if use_Y:
            if move_index < (d**2)*3:
                move_type = int(move_index/d**2) + 1
                new_index = move_index - (move_type - 1)*d**2

                row = new_index/d
                col = new_index%d

                new_move[int(row),int(col)] = move_type
        else:
            if move_index < (d**2)*2:
                move_type = int(move_index/d**2) + 1

                new_index = move_index - (move_type - 1)*d**2

                row = new_index/d
                col = new_index%d

                if move_type == 2:
                    move_type = 3

                new_move[int(row),int(col)] = move_type

    else:
        print("Error model you have specified is not currently supported")

    return new_move

def generate_one_hot_labels_surface_code(error,err_model):
    """"

    This function generates the homology class label, in a one-hot encoding, for a given perfect syndrome, to use as the target label
    for a feed forward neural network homology class predicting decoder.
!
    :param: error: An error configuration on a square lattice
    :param: err_model: A string in ["IIDXZ","DP","X"]
    :return: training_label: The one-encoded training label
    """
    
    d = error.shape[0]
    
    X = 0
    Z = 0
        
    for x in range(d):
        if error[x,0] == 1 or error[x,0] == 2:
            X = 1 - X
    for y in range(d):
        if error[0,y] == 3 or error[0,y] == 2:
            Z = 1 - Z
            
    if err_model in ["IIDXZ","DP"]:
        training_label = np.zeros(4,int)                       
    else:
        training_label = np.zeros(2,int)

    training_label[X + 2*Z] = 1
    
    return training_label

# ----------------------------------------------------------------------------------------------------------------------------
# -------------- funzioni che ho aggiunto io ---------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------


def update_json_history(filepath, new_log_data):
    """
    funzione per aggiornare il file JSON con i dati di training: 
    carica i dati JSON esistenti, aggiunge il nuovo record e salva.
    Args:
        filepath (str): Il percorso completo del file JSON (es: 'percorso/log_nome.json').
        new_log_data (dict): Un dizionario con i nuovi dati da aggiungere.
    """
    # sostituisce i valori NaN (da np.nan) con None per la serializzazione JSON
    cleaned_log_data = {k: (None if isinstance(v, float) and np.isnan(v) else v) 
                        for k, v in new_log_data.items()}

    # carica i dati esistenti
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    else:
        history = []
        
    # aggiungo il nuovo record
    history.append(cleaned_log_data)
    
    # salvo i dati aggiornati
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=4)


def setup_logging_and_checkpointing(all_configs, base_dir="runs", rl_number=None):
    """
    funzione per creare la struttura di directory per il logging e i checkpoint.
    """
    # creo il nome della run basato sull'orario e sui parametri
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if rl_number == None:
        run_name = f"DDQN_d{all_configs['d']}_p{all_configs['p_phys']}_{now}"
    else:
        run_name = f"DDQN_d{all_configs['d']}_p{all_configs['p_phys']}_{now}_rl{rl_number}"
    
    # creo la directory principale dell'esperimento
    logging_directory = os.path.join(base_dir, run_name)
    os.makedirs(logging_directory, exist_ok=True)
    
    # creo il percorso per il Log JSON
    logging_path = os.path.join(logging_directory, "training_history.json")
    
    # creo la directory e il percorso per salvare il best model --> checkpoint
    checkpoint_directory = os.path.join(logging_directory, "checkpoints")
    os.makedirs(checkpoint_directory, exist_ok=True)
    best_model_path = os.path.join(checkpoint_directory, "best_policy_net.pth")
    
    # salvo il dict di configurazione dell'esperimento in un file JSON
    with open(os.path.join(logging_directory, "config.json"), 'w') as f:
        json.dump(all_configs, f, indent=4)
        
    return logging_directory, logging_path, best_model_path


# NB: assicurarsi che la classe ReplayBuffer sia definita prima di usare queste funzioni
def save_replay_buffer(buffer, filepath):
    """
    salva l'oggetto ReplayBuffer su disco in formato binario (pickle), utile per caricare lo stato del buffer
    in un esperimento successivo.
    Args:
        buffer (ReplayBuffer): L'istanza del buffer da salvare.
        filepath (str): Il percorso completo del file (es: 'percorso/buffer_nome.pkl').
    """
    # creo la directory se non esiste
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
    try:
        with open(filepath, 'wb') as f:
            # 'wb' sta per write binary
            pickle.dump(buffer, f)
        print(f"ReplayBuffer salvato con successo in: {filepath}")
    except Exception as e:
        print(f"Errore durante il salvataggio del buffer: {e}")


def load_replay_buffer(filepath):
    """
    carica l'oggetto ReplayBuffer dal disco in formato binario (pickle).
    Args:
        filepath (str): Il percorso completo del file da caricare.
    Returns:
        ReplayBuffer: L'istanza del buffer ricaricata, oppure None se fallisce.
    """
    if not os.path.exists(filepath):
        print(f"Errore: File non trovato a: {filepath}")
        return None      
    try:
        with open(filepath, 'rb') as f:
            # 'rb' sta per read binary
            buffer = pickle.load(f)
        print(f"ReplayBuffer caricato con successo da: {filepath}")
        return buffer
    except Exception as e:
        print(f"Errore durante il caricamento del buffer: {e}")
        return None


def make_library(model_paths, policy_net, device):
    """
    Carica una lista di modelli pre-addestrati da file e li restituisce come una lista di policy.
    Args:
        model_paths (list of str): Lista di percorsi ai file dei modelli (.pth).
        policy_net: L'istanza della rete neurale per ottenere la struttura del modello.
        device (torch.device): Dispositivo (CPU/GPU) su cui caricare i modelli.
    Returns:
        list: Lista di policy caricate con i pesi dai file.
    """
    library = []
    for path in model_paths:
        model = copy.deepcopy(policy_net)  # copio la struttura della rete corrente
        model.load_state_dict(torch.load(path, map_location=device))  
        model.eval()  
        # per sicurezza disabilito anche il calcolo dei gradienti per i modelli caricati
        for param in model.parameters():
            param.requires_grad = False
        library.append(model)
    return library


# nb: per il logging posso riutilizzare la stessa funzione di prima, tanto non dipende dalla struttura dei dati
def setup_logging_and_checkpointing_ppr(all_configs, base_dir="runs"):
    """
    funzione per creare la struttura di directory per il logging e i checkpoint in caso di PPR.
    è simile a setup_logging_and_checkpointing ma con nome differente per la run.
    """
    # creo il nome della run basato sull'orario e sui parametri --> aggiungo _PPR
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"DDQN_PPR_d{all_configs['d']}_p{all_configs['p_phys']}_{now}"
    
    # creo la directory principale dell'esperimento
    logging_directory = os.path.join(base_dir, run_name)
    os.makedirs(logging_directory, exist_ok=True)
    
    # creo il percorso per il Log JSON
    logging_path = os.path.join(logging_directory, "training_history.json")
    
    # creo la directory e il percorso per salvare il best model --> checkpoint
    checkpoint_directory = os.path.join(logging_directory, "checkpoints")
    os.makedirs(checkpoint_directory, exist_ok=True)
    best_model_path = os.path.join(checkpoint_directory, "best_policy_net.pth")
    
    # salvo il dict di configurazione dell'esperimento in un file JSON
    with open(os.path.join(logging_directory, "config.json"), 'w') as f:
        json.dump(all_configs, f, indent=4)
        
    return logging_directory, logging_path, best_model_path


def softmax(W, tau):
    """Calcola la distribuzione softmax con temperatura tau, in modo stabile."""
    Z = W*tau
    # trovo il massimo e lo sottraggo per prevenire overflow
    max_Z = np.max(Z)
    exp_Z = np.exp(Z - max_Z)
    
    # normalizzo per ottenere la distribuzione di probabilità 
    sum_exp_Z = np.sum(exp_Z)
    
    # per sicurezza, gestisco preventivamente il caso di estremo underflow 
    if sum_exp_Z == 0:
        return np.ones_like(W) / len(W)
        
    return exp_Z / sum_exp_Z

# -------------- funzioni utili in analisi dati -------------------------------------------------------------------------------

def load_history(TRAINING_LOG_PATH):
    '''
    funzione utile per salvare il contenuto dei file json con history e log
    '''
    try:
        with open(TRAINING_LOG_PATH, 'r', encoding='utf-8') as file:
            # La funzione json.load() legge il file e lo converte
            training_history = json.load(file)
        print(f"Contenuto di {TRAINING_LOG_PATH} estratto con successo:")     
        return training_history
    except FileNotFoundError:
        print(f"Errore: Il file '{TRAINING_LOG_PATH}' non è stato trovato.")
        return None
    except json.JSONDecodeError:
        print(f"Errore: Il file '{TRAINING_LOG_PATH}' contiene JSON non valido.")
        return None
    

def rolling_avg(dati: np.ndarray, rolling_length: int, partial_fill = False):
    """
    funzione per calcolare la media mobile a posteriori data una lunghezza arbitraria.
    args:
        dati: array contentente i valori da elaborare
        rolling_length : lunghezza della finestra mobile
        partial_fill : parametro booleano per stabilire se riempire i primi valori con Nan oppure con medie su finestre parziali
    return:
        rolling_avg : media mobile, con i primi rolling_length-1 valori nan
    """
    pesi = np.ones(rolling_length) / rolling_length

    # np.convolve è un modo efficiente per calcolare somme mobili
    # 'valid' assicura che il risultato sia calcolato solo dove la finestra si sovrappone all'array input
    true_avg = np.convolve(dati, pesi, mode='valid')
    
    # true_avg è più corto, riempio i primi N-1 elementi mancanti o con medie parziali o con nan
    if partial_fill:
        partial_avg = []
        for i in range(1, rolling_length):
            partial_avg.append(np.mean(dati[:i]))
        initial_value = np.array(partial_avg)
    else: 
        initial_value = np.full(rolling_length - 1, np.nan)
    
    rolling_avg = np.concatenate((initial_value, true_avg))
    
    return rolling_avg