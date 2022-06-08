import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from scipy.integrate import odeint
from torchdiffeq import odeint_adjoint as odeint
import torchsde
import pandas as pd

class GenerateDynSystem():
    """
    Generate data from an autonomous dynamical system
    """
    def __init__(self, state0, model, dt, steps, discard, filename, sigma=None, include_time=False):
        """
        Args:
        state0 is the initial point (given as 1d array)
        f is the generative function
        dt is the time step
        steps are the number of points to be taken
        discard are the number of points to be discarded at the beginning of the dynamics
        sigma is standard deviation, if None, data are generated with Euler-Majorana integrator
        if include_time True, put time in the input vector in the last position
        Return a pandas dataframe, shape (steps, state0.size)
        """
        # Parameters
        self.state0 = state0
        self.dt = dt
        self.steps = steps
        self.discard = discard
        self.filename = filename
        self.sigma = sigma 
        self.include_time = include_time
        model.sigma = sigma
        model.dt = dt
        self.model = model
        
    def __call__(self):
        # Generate the dynamics and discard first entries
        self.time = torch.arange(0.0, (self.steps+self.discard)*self.dt, self.dt)
        # Add possible noise
        if self.sigma is None:
            self.dataset = odeint(self.model, self.state0, self.time)
        else:
            self.dataset = torchsde.sdeint(self.model, self.state0.unsqueeze(0), self.time, method='euler') 
            self.dataset = self.dataset.squeeze()
            
        self.time = self.time[self.discard:]
        self.dataset = self.dataset[self.discard:]
        
        # Include time dimension
        if self.include_time:
            self.dataset = torch.cat((self.dataset, torch.tensor([self.dt]*steps).unsqueeze(-1)), dim=-1)
            
        # Convert into pandas dataframe and save
        df =  pd.DataFrame(self.dataset)
        df.to_pickle(self.filename)
        
        return self.dataset, self.time
    
### Costume dataset class
class DynSysDataset(Dataset):
    """
    Create a Dataset by dividing into datas sequences of fixed length
    """
    def __init__(self, filename, seq_len, convolution=False, transform=None):
        super().__init__()
        # Parameters
        self.filename = filename
        self.seq_len = seq_len
        self.convolution = convolution
        self.transform = transform
        
        
        # Load pandas dataframe from filename and convert to torch tensor
        self.dataset = pd.read_pickle(filename)
        self.dataset = torch.tensor(self.dataset.values, dtype=torch.float32, requires_grad=True)
        length = self.dataset.shape[0]
        self.num_sequences = int(length/seq_len)
        
        # Divide in into tensor of size = (num_sequences, len_seq, feature_dim)
        self.data = torch.reshape(self.dataset[:self.num_sequences*seq_len,:], (self.num_sequences, seq_len, len(self.dataset[0,:])))
        self.data = self.data.type(torch.float32)
        
        # Unsqueeze dimension for convolution 
        if self.convolution:
            self.data = self.data.unsqueeze(1)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Apply transforms
        if self.transform is not None:
            self.data = self.transform(self.data)
            
        return self.data[idx,...]