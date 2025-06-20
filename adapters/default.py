from typing import Dict, List
import pandas as pd
import torch
from torch import Tensor

# StateAdapter includes static methods for adapters
from elsciRL.encoders.poss_state_encoded import StateEncoder
from gymnasium.spaces import Discrete

class Adapter:
    _cached_state_idx: Dict[str, int] = dict()

    def __init__(self, setup_info:dict={}) -> None:
        # Creates list of all possible x+y states and indexes them
        self.x_range = [0,1,2,3,4,5]
        self.y_range = [0,1,2,3,4,5]
        possible_states = [str(x)+'_'+str(y) for x in self.x_range for y in self.y_range]
        
        self.encoder = StateEncoder(len(possible_states))
        self.observation_space = Discrete(len(possible_states))
            
    def adapter(self, state:str, legal_moves:list = None, episode_action_history:list = None, encode:bool = True, indexed: bool = False) -> Tensor:
        """ id number for each state as numeric representation """
        
        # Encode to Tensor for agents
        if encode:
            state_encoded = self.encoder.encode(state=state)
        else:
            state_encoded = state

        if (indexed):
            state_indexed = list()
            for sent in state:
                if (sent not in Adapter._cached_state_idx):
                    Adapter._cached_state_idx[sent] = len(Adapter._cached_state_idx)
                state_indexed.append(Adapter._cached_state_idx[sent])

            state_encoded = torch.tensor(state_indexed)

        return state_encoded