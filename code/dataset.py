import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Tuple

class AnesthesiaDataset(Dataset):
    """
    PyTorch Dataset for Behavioral Cloning
    
    Expert demonstrations를 (state, action) 쌍으로 제공
    """
    
    def __init__(self, states, actions, orig_states):
        """
        Args:
            states: [B, N, state_dim] numpy array
            actions: [B, N, action_dim] numpy array
        """
        self.states = torch.FloatTensor(states)
        self.orig_states = torch.FloatTensor(orig_states)
        self.actions = torch.FloatTensor(actions)
        
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.orig_states[idx]

