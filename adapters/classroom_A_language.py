from typing import Dict, List
import pandas as pd
import numpy as np
import torch
from torch import Tensor

# StateAdapter includes static methods for adapters
from elsciRL.encoders.sentence_transformer_MiniLM_L6v2 import LanguageEncoder
from gymnasium.spaces import Box

# Language data 
#from adapters.language_data.student_features import data as StudentFeatures

class Adapter:
    _cached_state_idx: Dict[str, int] = dict()

    def __init__(self, setup_info:dict={}) -> None:
        self.student_features = {
            '4_1': {'type': 'student','hair_colour': 'black','hair_style': 'medium curly','upperclothing_type': 'hoodie','upperclothing_colour': 'black',
                    'lowerclothing_type': 'denim jeans','lowerclothing_colour': 'black','piercings': 'ear', 'gender': 'female'},
            '3_1': {'type': 'student','hair_colour': 'black','hair_style': 'long straight','upperclothing_type': 'hoodie','upperclothing_colour': 'black',
                    'lowerclothing_type': 'denim jeans','lowerclothing_colour': 'black','piercings': 'ear', 'gender':'male'},
            '2_1': {'type': 'student','hair_colour': 'black','hair_style': 'long medium','upperclothing_type': 'hoodie','upperclothing_colour': 'black',
                    'lowerclothing_type': 'denim jeans','lowerclothing_colour': 'black','piercings': 'face','gender': 'female'},
            '1_1': {'type': 'student','hair_colour': 'brown','hair_style': 'medium straight','upperclothing_type': 'jacket','upperclothing_colour': 'blue',
                    'lowerclothing_type': 'denim jeans','lowerclothing_colour': 'black','piercings': 'no','gender': 'male'},
            '1_2': {'type': 'student','hair_colour': 'blonde','hair_style': 'long straight','upperclothing_type': 'hoodie','upperclothing_colour': 'red',
                    'lowerclothing_type': 'denim jeans','lowerclothing_colour': 'black','piercings': 'ear','gender': 'female'},
            '1_3': {'type': 'student','hair_colour': 'brown','hair_style': 'short','upperclothing_type': 'shirt','upperclothing_colour': 'blue',
                    'lowerclothing_type': 'chinos','lowerclothing_colour': 'brown','piercings': 'no','gender': 'male'},
            '2_3': {'type': 'student','hair_colour': 'blonde','hair_style': 'long straight','upperclothing_type': 'cardigan','upperclothing_colour': 'blue',
                    'lowerclothing_type': 'chinos','lowerclothing_colour': 'brown','piercings': 'no','gender':'female'},
            '3_3': {'type': 'teacher','hair_colour': 'brown','hair_style': 'short','upperclothing_type': 'jacket','upperclothing_colour': 'brown',
                    'lowerclothing_type': 'chinos','lowerclothing_colour': 'blue','piercings': 'no', 'gender': 'male'},
            '3_2': {'type': 'student','hair_colour': 'pink','hair_style': 'mohawk','upperclothing_type': 'studded denim jacket','upperclothing_colour': 'blue',
                    'lowerclothing_type': 'denim jeans','lowerclothing_colour': 'black','piercings': 'face', 'gender': 'male'},
            '4_3': {'type': 'recycling','hair_colour': 'NA','hair_style': 'NA','upperclothing_type': 'NA','upperclothing_colour': 'NA',
                    'lowerclothing_type': 'NA','lowerclothing_colour': 'NA','piercings':'NA','gender':'NA'},
            '4_2': {'type': 'trash','hair_colour': 'NA','hair_style': 'NA','upperclothing_type': 'NA','upperclothing_colour': 'NA',
                    'lowerclothing_type': 'NA','lowerclothing_colour': 'NA','piercings':'NA','gender':'NA'},
        }
        self.encoder = LanguageEncoder()
        self.observation_space = Box(low=-1, high=1, shape=(1,384), dtype=np.float32)
    
    def adapter(self, state:str, legal_moves:list = None, episode_action_history:list = None, encode:bool = True, indexed: bool = False) -> Tensor:
        """ Use Language description for every student for current grid position """
        student_features = self.student_features[state]
            
        stud_type = student_features['type']
        stud_hair_colour = student_features['hair_colour']
        stud_hair_style = student_features['hair_style']
        stud_upperclothing_type = student_features['upperclothing_type']
        stud_upperclothing_colour = student_features['upperclothing_colour']
        stud_lowerclothing_type = student_features['lowerclothing_type']
        stud_lowerclothing_colour = student_features['lowerclothing_colour']
        stud_piercings = student_features['piercings']
        stud_gender = student_features['gender']
        


        # Covert numeric dict to a list of strings describing player positions
        if stud_type == 'trash':
            full_str = 'This is a trash can.'
        elif stud_type == 'recycling':
            full_str = 'This is a recycling bin.'
        elif stud_type == 'teacher':
            full_str = ('The ' + stud_gender + ' teacher that has ' + stud_hair_style + ' ' + stud_hair_colour + ' hair and is wearing a ' + 
                        stud_upperclothing_colour + ' ' + stud_upperclothing_type + ', ' + stud_lowerclothing_colour + ' ' + stud_lowerclothing_type + 
                        ' and has ' + stud_piercings + ' piercings.')
        else:
            full_str = ('The ' + stud_gender + ' student that has ' + stud_hair_style + ' ' + stud_hair_colour + ' hair and is wearing a ' + 
                        stud_upperclothing_colour + ' ' + stud_upperclothing_type + ', ' + stud_lowerclothing_colour + ' ' + stud_lowerclothing_type + 
                        ' and has ' + stud_piercings + ' piercings.')

        # Encode to Tensor for agents
        if encode:
            state_encoded = self.encoder.encode(state=full_str)
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