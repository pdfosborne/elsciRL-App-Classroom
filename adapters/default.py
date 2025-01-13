from typing import Dict, List
import pandas as pd
import torch
from torch import Tensor

# StateAdapter includes static methods for adapters
from elsciRL.encoders.sentence_transformer_MiniLM_L6v2 import LanguageEncoder

# Language data 
from adapters.language_data.student_features import data as StudentFeatures

class ClassroomALanguage:
    _cached_state_idx: Dict[str, int] = dict()

    def __init__(self, setup_info:dict={}) -> None:
        self.student_features = StudentFeatures
        self.encoder = LanguageEncoder()
    
    def adapter(self, state:str, legal_moves:list = None, episode_action_history:list = None, encode:bool = True, indexed: bool = False) -> Tensor:
        """ Use Language name for every piece name for current board position """
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
        state:List[str] = []
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
        state.append(full_str) 
        # Encode to Tensor for agents
        if encode:
            state_encoded = self.encoder.encode(state=state)
        else:
            state_encoded = state

        if (indexed):
            state_indexed = list()
            for sent in state:
                if (sent not in ClassroomALanguage._cached_state_idx):
                    ClassroomALanguage._cached_state_idx[sent] = len(ClassroomALanguage._cached_state_idx)
                state_indexed.append(ClassroomALanguage._cached_state_idx[sent])

            state_encoded = torch.tensor(state_indexed)

        return state_encoded