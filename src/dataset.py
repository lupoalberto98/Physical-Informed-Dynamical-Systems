import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.integrate import odeint
from torchdiffeq import odeint_adjoint as odeint
import torchsde


### Costume dataset class
class DynSysDataset(Dataset):
    """
    Create a Dataset with the dynamics of a generic dynamical systems starting from state0
    """
    def __init__(self, state0, model, dt, steps, seq_len, discard, sigma=None, include_time=False, convolution=False, transform=None):
        """
        Args:
        state0 is the initial point (given as 1d array)
        f is the generative function
        dt is the time step
        steps are the number of points to be taken
        seq_len is the length of the sequences in which the lond sequence is divided
        discard are the number of points to be discarded at the beginning of the dynamics
        if include_time True, put time in the input vector in the last position
        """
        # Parameters
        self.state0 = state0
        self.model = model
        self.dt = dt
        self.steps = steps
        self.seq_len = seq_len
        self.discard = discard
        self.sigma = sigma # standard deviation of gaussian noise for stochasti integrator
        self.include_time = include_time
        self.transform = transform
        self.num_sequences = int(steps/seq_len)
        
        # Generate the dynamics and discard first entries
        self.time = torch.arange(0.0, (steps+discard)*self.dt, self.dt)
        if sigma is None:
            self.dataset = odeint(self.model, self.state0, self.time)
        else:
            model.sigma = sigma
            model.dt = dt
            self.dataset = torchsde.sdeint(self.model, self.state0.unsqueeze(0), self.time, method='euler') 
            self.dataset = self.dataset.squeeze()
            
        self.time = self.time[discard:]
        self.dataset = self.dataset[discard:]
        
        # Include time dimension
        if self.include_time:
            self.dataset = torch.cat((self.dataset, torch.tensor([self.dt]*steps).unsqueeze(-1)), dim=-1)
        
        # Divide in into tensor of size = (num_sequences, len_seq, feature_dim)
        self.data = torch.reshape(self.dataset[:self.num_sequences*seq_len,:], (self.num_sequences, seq_len, len(self.dataset[0,:])))
        self.data = self.data.type(torch.float32)
        
        # Unsqueeze dimension for convolution 
        if convolution:
            self.data = self.data.unsqueeze(1)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Apply transforms
        if self.transform is not None:
            self.data = self.transform(self.data)
            
        return self.data[idx,...]