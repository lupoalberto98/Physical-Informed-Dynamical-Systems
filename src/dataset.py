import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.integrate import odeint

### Costume dataset class
class DynSysDataset(Dataset):
    """
    Create a Dataset with the dynamics of a generic dynamical systems starting from state0
    """
    def __init__(self, state0, f, dt, steps, seq_len, discard, transform=None):
        """
        Args:
        state0 is the initial point (given as 1d array)
        f is the generative function
        dt is the time step
        steps are the number of points to be taken
        seq_len is the length of the sequences in which the lond sequence is divided
        discard are the number of points to be discarded at the beginning of the dynamics
        """
        # Parameters
        self.state0 = state0
        self.f = f
        self.dt = dt
        self.steps = steps
        self.seq_len = seq_len
        self.discard = discard
        self.transform = transform
        self.num_sequences = int(steps/seq_len)
        
        # Generate the dynamics and discard first entries
        self.time = np.arange(0.0, (steps+discard)*dt, dt)
        self.dataset = odeint(self.f, self.state0, self.time)
        self.time = self.time[discard:]
        self.dataset = self.dataset[discard:]
        
        # Divide in into tensor of size = (num_sequences, len_seq, feature_dim)
        self.data = np.reshape(self.dataset[:self.num_sequences*seq_len,:], (self.num_sequences, seq_len, len(state0)))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Apply transforms
        if self.transform is not None:
            self.data = self.transform(self.data)
            
        return self.data[idx]