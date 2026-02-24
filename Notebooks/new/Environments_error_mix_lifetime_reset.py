#----- (0) Imports ---------------------------------------------------------------------------------------------------------------

from Function_Library import *
import gymnasium as gym
import copy
from itertools import product, starmap
import numpy as np

#---------- (1) --------------------------------------------------------------------------------------------------------------------------------------

class Surface_Code_Environment_Multi_Decoding_Cycles():
    """
    A surface code environment for obtaining decoding agents in the fault-tolerant setting.
    In particular:

        - The visible state consists of a syndrome volume + completed action volume
        - The agent can perform single qubit Pauli flips on physical data qubits
        -   - an error volume is introduced if the agent does the identity
            - an error volume is introduced if the agent repeats the same move twice
        - a plus 1 reward is given for every action that results in the code being in the ground state space.
        - The terminal state criterion is satisfied for every hidden state that cannot be decoded by the referee/static decoder

    Note, this environment can:
        - cater for error_model in {"X","DP"}
        - cater for faulty syndromes - i.e. p_meas > 0

    Also, this environment provides all methods as required by an openAi gym class. In particular:
        - reset
        - step


    Attributes
    ----------

    :param: d: The code distance
    :param: p_phys: The physical error probability on a single physical data qubit
    :param: p_meas: The measurement error probability on a single syndrome bit
    :param: error_model: A string in ["X, DP"]
    :param: use_Y: A boolean indicating whether the environment accepts Y Pauli flips as actions
    :param: volume_depth: The number of sequential syndrome measurements performed when generating a new syndrome volume.
    :param: static_decoder: A homology class predicting decoder for perfect syndromes.

    """


    def __init__(self, d=5, p_phys=0.01, p_meas=0.01, use_Y=True, volume_depth=3, static_decoder=None):

        self.d = d
        self.p_phys = p_phys
        self.p_meas= p_meas
        self.error_model = np.random.choice(["X", "DP"], p=[0.5, 0.5])
        self.use_Y = use_Y
        self.volume_depth = volume_depth
        self.static_decoder = static_decoder

        self.n_action_layers = 0
        '''if error_model == "X":
            self.num_actions = d**2 + 1
            self.n_action_layers = 1
        elif error_model == "DP":
            if use_Y:
                self.num_actions = 3*d**2 + 1
                self.n_action_layers = 3
            else:
                self.num_actions = 2*d**2 + 1 
                self.n_action_layers = 2
        else:
            print("specified error model not currently supported!")'''
        
        self.num_actions = 3*d**2 + 1
        self.n_action_layers = 3

        self.identity_index = self.num_actions -1
        self.identity_indicator = self.generate_identity_indicator(self.d)

        self.qubits = generateSurfaceCodeLattice(self.d)
        self.qubit_stabilizers = self.get_stabilizer_list(self.qubits, self.d)  
        self.qubit_neighbours = self.get_qubit_neighbour_list(self.d) 
        self.completed_actions = np.zeros(self.num_actions, int)
        
    
        self.observation_space=gym.spaces.Box(low=0,high=1,
                                      shape=(self.volume_depth+self.n_action_layers,
                                        2*self.d+1, 
                                        2*self.d+1),
                                      dtype=np.float32)   # modifica da np.uint8 a np.float32 per compatibilità con pytorch
        
        self.action_space = gym.spaces.Discrete(self.num_actions)

        self.hidden_state = np.zeros((self.d, self.d), int)
        self.current_true_syndrome = np.zeros((self.d+1, self.d+1), int)
        self.summed_syndrome_volume = None         
        self.board_state = np.zeros((self.volume_depth + self.n_action_layers, 2 * self.d + 1, 2 * self.d + 1),int)

        self.completed_actions = np.zeros(self.num_actions, int)
        self.acted_on_qubits = set()
        self.legal_actions = set()
        self.done = False       # con la nuova versione di gym diventa "terminated" e si aggiunge "truncated", anche se non usato in questo environment
        self.lifetime = 0

        self.multi_cycle = True

        self.X_count = 0
        self.DP_count = 0

        self.success_count = 0

        self.max_round_steps = 1000
        self.round_steps = 0
        self.rounds = 0
        self.max_episode_steps = 1000
        self.end_round_num_anyons = 0

    def reset(self):       
        """
        Resetting of the environment introduces a new non-trivial syndrome volume.

        :return: self.board_state: The new reset visible state of the environment = syndrome volume + blank action history volume
        """
        # sarebbe : terminated = False, truncated = False, 
        self.done = False
        self.lifetime = 0

        self.round_steps = 0
        self.episode_steps = 0

        self.error_model = np.random.choice(["X", "DP"], p=[0.5, 0.5])
        if self.error_model == "X":
            self.X_count += 1
        else:
            self.DP_count += 1
        
        # Create the initial error - we wait until there is a non-trivial syndrome. BUT, the lifetime is still updated!
        self.initialize_state()

        self.lifetime = 0

        # Update the legal moves available to us
        self.reset_legal_moves()

        # aggiungo info, necessario per compatibilità con gymnasium rispetto a gym
        info = {}

        # mi assicuro che il tipo di dato sia float32, necessario per compatibilità con pytorch
        return self.board_state.astype(np.float32), info


    def step(self, action):
        """
        Given an action, this method executes the logic of the environment.

        :param: action: The action, given as an integer, supplied by some agent.
        :return: self.board_state: The new visible state = syndrome volume + action history volume
        :return: reward: The reward for the action
        :return: terminated: terminal flag (logical failure)
        :return: truncated: Always False here (unless you later add a cap)
        :return: info: diagnostics dict (empty here)
        """

        self.end_round_num_anyons = 0

        self.round_steps += 1
        self.episode_steps += 1
        done_identity = False
        if action == self.identity_index or int(self.completed_actions[action]) == 1:
            done_identity = True

        # 1) Apply the action to the hidden state
        action_lattice = index_to_move(self.d, action, self.error_model, self.use_Y)
        self.hidden_state = obtain_new_error_configuration(self.hidden_state, action_lattice)

        # 2) Compute TRUE syndrome from hidden state (IMPORTANT: must be recomputed here)
        self.current_true_syndrome = generate_surface_code_syndrome_NoFT_efficient(self.hidden_state, self.qubits)
        num_anyons = int(np.sum(self.current_true_syndrome))

        # 3) Logical class label (simulation oracle)
        correct_label = generate_one_hot_labels_surface_code(self.hidden_state, self.error_model)
        logical_class = int(np.argmax(correct_label))  # 0=trivial, !=0 = logical (per tua funzione)

        # 4) Reward
        reward = 0.0
        if logical_class == 0 and num_anyons == 0:
            reward = 1.0
            self.success_count += 1

        # 5) Termination: logical failure (only when syndrome is empty -> closed/codespace)
        #    This avoids terminating while the error is still "open" (anyons present).
        #if (num_anyons == 0) and (logical_class != 0):
        if num_anyons == 0 and logical_class != 0:
            self.done = True
        else:
            self.done = False

        # 6) If necessary, apply multiple errors and obtain a NEW syndrome volume
        #    (this is where lifetime advances)

        if (self.max_round_steps is not None) and (not self.done) and (self.round_steps >= self.max_round_steps):
            done_identity = True
            self.round_steps = 0


        if done_identity:
            self.round_steps = 0

            self.rounds += 1   

            self.end_round_num_anyons = num_anyons 


            

            trivial_volume = True
            while trivial_volume:
                self.summed_syndrome_volume = np.zeros((self.d + 1, self.d + 1), int)
                faulty_syndromes = []

                for j in range(self.volume_depth):
                    error = generate_error(self.d, self.p_phys, self.error_model)

                    if int(np.sum(error) != 0):
                        self.hidden_state = obtain_new_error_configuration(self.hidden_state, error)
                        # update true syndrome after applying physical error
                        self.current_true_syndrome = generate_surface_code_syndrome_NoFT_efficient(self.hidden_state, self.qubits)

                    current_faulty_syndrome = generate_faulty_syndrome(self.current_true_syndrome, self.p_meas)
                    faulty_syndromes.append(current_faulty_syndrome)
                    self.summed_syndrome_volume += current_faulty_syndrome
                    self.lifetime += 1

                if int(np.sum(self.summed_syndrome_volume)) != 0:
                    trivial_volume = False

            # update board state with the measured (faulty) syndromes
            for j in range(self.volume_depth):
                self.board_state[j, :, :] = self.padding_syndrome(faulty_syndromes[j])

            # reset the legal moves
            self.reset_legal_moves()

            # clear action history layers
            self.board_state[self.volume_depth:, :, :] = np.zeros(
                (self.n_action_layers, 2 * self.d + 1, 2 * self.d + 1), int
            )

        else:
            # Update the completed actions and legal moves
            self.completed_actions[action] = int(not (self.completed_actions[action]))

            if action != self.identity_index:
                acted_qubit = action % (self.d ** 2)

                if acted_qubit not in self.acted_on_qubits:
                    self.acted_on_qubits.add(acted_qubit)
                    allowed_layers = self.get_allowed_action_layers()
                    for neighbour in self.qubit_neighbours[acted_qubit]:
                        for j in allowed_layers:
                            self.legal_actions.add(neighbour + j * self.d**2)

            # update the board state to reflect the action taken (action history layers)
            for k in range(self.n_action_layers):
                self.board_state[self.volume_depth + k, :, :] = self.padding_actions(
                    self.completed_actions[k * self.d ** 2:(k + 1) * self.d ** 2]
                )

        info = {}
        truncated = False

        if self.episode_steps > self.max_episode_steps:
            truncated = True
            self.episode_steps = 0
        
        return self.board_state.astype(np.float32), reward, self.done, truncated, info

    def initialize_state(self):
        """
        Generate an initial non-trivial syndrome volume
        """
        # sarebbe : terminated = False, truncated = False
        self.done = False
        self.hidden_state = np.zeros((self.d, self.d), int)
        self.current_true_syndrome = np.zeros((self.d+1, self.d+1), int) 
        self.board_state = np.zeros((self.volume_depth + self.n_action_layers, 2 * self.d + 1, 2 * self.d + 1),int)
        
        trivial_volume = True
        while trivial_volume:
            self.summed_syndrome_volume = np.zeros((self.d + 1, self.d + 1), int)
            faulty_syndromes = []
            for j in range(self.volume_depth):
                error = generate_error(self.d, self.p_phys, self.error_model)
                if int(np.sum(error)) != 0:
                    self.hidden_state = obtain_new_error_configuration(self.hidden_state, error)
                    self.current_true_syndrome = generate_surface_code_syndrome_NoFT_efficient(self.hidden_state, self.qubits)
                current_faulty_syndrome = generate_faulty_syndrome(self.current_true_syndrome, self.p_meas)
                faulty_syndromes.append(current_faulty_syndrome)
                self.summed_syndrome_volume += current_faulty_syndrome
                self.lifetime += 1

            if int(np.sum(self.summed_syndrome_volume)) != 0:
                trivial_volume = False

        # update the board state to reflect the measured syndromes
        for j in range(self.volume_depth):
            self.board_state[j, :, :] = self.padding_syndrome(faulty_syndromes[j])


    def get_allowed_action_layers(self):
        """
        Restituisce gli indici dei layer di azione ammessi per il noise model corrente.
        Assunzione mapping layer:
            j = 0 -> X
            j = 1 -> Y
            j = 2 -> Z
        """
        if self.error_model == "X":
            # per bit-flip noise, permetto solo correzioni X
            return [0]

        elif self.error_model == "DP":
            # per depolarizing, permetto X/Z (e Y opzionalmente)
            if self.use_Y:
                return [0, 1, 2]
            else:
                return [0, 2]

        # fallback prudente
        return [0]


    def reset_legal_moves(self):
        """
        Reset the legal moves (dipendenti dal noise model corrente).
        """
        self.completed_actions = np.zeros(self.num_actions, int)
        self.acted_on_qubits = set()
        self.legal_actions = set()

        legal_qubits = set()
        for qubit_number in range(self.d**2):
            # qubit adiacenti a sindromi violate
            if self.is_adjacent_to_syndrome(qubit_number):
                legal_qubits.add(qubit_number)

        # identity sempre legale
        self.legal_actions.add(self.identity_index)

        # abilito solo i layer coerenti con il noise model corrente
        allowed_layers = self.get_allowed_action_layers()
        for j in allowed_layers:
            base = j * self.d**2
            for legal_qubit in legal_qubits:
                self.legal_actions.add(legal_qubit + base)



    def is_adjacent_to_syndrome(self, qubit_number):
        """
        Determine whether a qubit is adjacent to a violated stabilizer
        """

        for stabilizer in self.qubit_stabilizers[qubit_number]:
            if self.summed_syndrome_volume[stabilizer] != 0:
                return True

        return False

    def padding_syndrome(self, syndrome_in):
        """
        Pad a syndrome into the required embedding
        """

        syndrome_out = np.zeros((2*self.d+1, 2*self.d+1),int)
        
        for x in range( 2*self.d+1 ):
            for y in range( 2*self.d+1 ):

                #label the boundaries and corners
                if x==0 or x== 2*self.d:
                    if y%2 == 1:
                        syndrome_out[x,y] = 1

                if y==0 or y== 2*self.d:
                    if x%2 == 1:
                        syndrome_out[x,y] = 1

                if x%2 == 0 and y%2 == 0: 
                    # copy in the syndrome
                    syndrome_out[ x, y ] = syndrome_in[ int(x/2), int(y/2) ]
                elif x%2 == 1 and y%2 == 1:
                    if (x+y)%4 == 0:
                        #label the stabilizers
                        syndrome_out[x,y] = 1
        return syndrome_out
        
    def padding_actions(self,actions_in):
        """
        Pad an action history for a single type of Pauli flip into the required embedding.
        """
        actions_out = np.zeros( ( 2*self.d+1, 2*self.d+1 ),int )

        for action_index, action_taken in enumerate( actions_in ):
            if action_taken:
                row = int( action_index / self.d )
                col = int( action_index % self.d )

                actions_out[ int( 2*row+1 ), int( 2*col+1 ) ] = 1

        return actions_out

    def indicate_identity(self, board_state):
        """
        Pad the action history to indicate that an identity has been performed.
        """

        for k in range(self.n_action_layers):
            board_state[self.volume_depth + k,:, :] = board_state[self.volume_depth + k,:, :] + self.identity_indicator

        return board_state

    def get_qubit_stabilizer_list(self, qubits, qubit):
        """"
        Given a qubit specification [qubit_row, qubit_column], this function returns the list of non-trivial stabilizer locations adjacent to that qubit
        """

        qubit_stabilizers = []
        row = qubit[0]
        column = qubit[1]
        for j in range(4):
            if qubits[row,column,j,:][2] != 0:     # i.e. if there is a non-trivial stabilizer at that site
                qubit_stabilizers.append(tuple(qubits[row,column,j,:][:2]))
        return qubit_stabilizers  

    def get_stabilizer_list(self, qubits, d):
        """"
        Given a lattice, this function outputs a list of non-trivial stabilizers adjacent to each qubit in the lattice, indexed row-wise starting from top left
        """
        stabilizer_list = []
        for qubit_row in range(self.d):
            for qubit_column in range(self.d):
                stabilizer_list.append(self.get_qubit_stabilizer_list(qubits,[qubit_row,qubit_column]))
        return stabilizer_list

    def get_qubit_neighbour_list(self, d):
        """"
        Given a lattice, this function provides a list of the neighbouring qubits for each physical qubit.
        """

        count = 0
        qubit_dict = {}
        qubit_neighbours = []
        for row in range(d):
            for col in range(d):
                qubit_dict[str(tuple([row,col]))] = count
                cells = starmap(lambda a,b: (row+a, col+b), product((0,-1,+1), (0,-1,+1)))
                qubit_neighbours.append(list(cells)[1:])
                count +=1
            
        neighbour_list = []
        for qubit in range(d**2):
            neighbours = []
            for neighbour in qubit_neighbours[qubit]:
                if str(neighbour) in qubit_dict.keys():
                    neighbours.append(qubit_dict[str(neighbour)])
            neighbour_list.append(neighbours)

        return neighbour_list

    def generate_identity_indicator(self, d):
        """"
        A simple helper function to generate the array that will be added to the action history to indicate that an identity has been performed.
        """

        identity_indicator = np.ones((2 *d + 1, 2 *d + 1),int)
        for j in range(d):
            row = 2*j + 1
            for k in range(d):
                col = 2*k + 1
                identity_indicator[row,col] = 0
        return identity_indicator
    


'''import gymnasium as gym
import numpy as np

class SurfaceCodeGymWrapper(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, core_env):
        super().__init__()
        self.core = core_env
        self.observation_space = core_env.observation_space
        self.action_space = core_env.action_space

        # inizializzazione sicura contatori
        self._ep_reward = 0.0
        self._ep_steps = 0

    @property
    def lifetime(self):
        return self.core.lifetime

    def _ensure_episode_counters(self):
        # nel dubbio: se per qualche motivo non esistono, creali
        if not hasattr(self, "_ep_reward"):
            self._ep_reward = 0.0
        if not hasattr(self, "_ep_steps"):
            self._ep_steps = 0

    def reset(self, *, seed=None, options=None):
        self._ensure_episode_counters()
        obs, info = self.core.reset()
        self._ep_reward = 0.0
        self._ep_steps = 0
        return obs, info

    def step(self, action):
        self._ensure_episode_counters()
        lifetime_before = self.lifetime

        obs, reward, terminated, truncated, info = self.core.step(action)

        self._ep_reward += float(reward)
        self._ep_steps += 1

        done = bool(terminated or truncated)
        if info is None:
            info = {}

        # info step-level
        info["lifetime"] = int(self.core.lifetime)

        # info episode-level SOLO a fine episodio

        lifetime_after = self.lifetime
        lifetime_increment = lifetime_after - lifetime_before

        # shaping: paga per ogni ciclo aggiunto
        #reward += 1.0 * lifetime_increment   # oppure alpha * lifetime_increment
    
        if done:
            info["episode_reward"] = float(self._ep_reward)
            info["episode_steps"] = int(self._ep_steps)
            info["episode_lifetime"] = int(self.core.lifetime)

        return obs, reward, terminated, truncated, info

    def action_masks(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[list(self.core.legal_actions)] = True
        return mask
'''