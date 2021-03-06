"""
@author : Alberto Bassi
"""

#!/usr/bin/env python3
import torch
import numpy as np
from scipy.integrate import odeint
import torch.nn as nn
import copy
from typing import Union
import math
    
    
# Loss classes
class Eul():
    """
    Compute forward pass with Euler scheme
    Args:
    dt is the time step
    if time_included, time is carried as last element of input vector
    """
    def __init__(self, dt, model):
        self.dt = dt
        self.model = model # callable object o nn.Module subclass

    def __call__(self, state):
        """
        If state and next_state are the ground truth, then model is the NN ansatz.
        Else, if next_state is the NN forwarded, then model should be the ground truth
        """
        t = 10 # required by system solvers
        # Propagate
        df = self.model(t, state)*self.dt
        
        return df
    
class RK4():
    """
    Implement a 4th order Runge-Kutta method to compute loss
    """
    def __init__(self, dt, model):
        self.dt = dt
        self.model = model # callable object o nn.Module subclass
        
    def __call__(self, state):
        """
        If state and next_state are the ground truth, then model is the NN ansatz.
        Else, if next_state is the NN forwarded, then model should be the ground truth
        """
        # Propagate
        t = 10
        k1 = self.model(t, state)
        k2 = self.model(t, state+self.dt*k1/2.)
        k3 = self.model(t, state+self.dt*k2/2.)
        k4 = self.model(t, state+self.dt*k3)
        df = self.dt*(k1+2*k2+2*k3+k4)/6.0
                                   
        return df

    
class ENLoss(nn.Module):
    def __init__(self, dt):
        super(ENLoss, self).__init__()
        self.dt = dt
    
    def forward(self, state, next_state, args):
        args_loss = 0
        if args is not None:
            E = torch.sum(state**2/2., dim=-1)
            E_next = torch.sum(next_state**2/2., dim=-1)
            E_tilde = torch.matmul(state**2/2., args[0])
            M_tilde = torch.matmul(state, args[1])
            args_loss += torch.mean((E_next - E + 2*E_tilde*self.dt-M_tilde*self.dt)**2) 
            
        return args_loss


class CeDLoss(nn.Module):
    """
    Compute the physical informed loss of a dynamical system with central derivative
    Args:
    dt is the time step
    field is the field of the dyanamical system (true model)
    """
    def __init__(self,  dt, field):
        super(CeDLoss, self).__init__()
        self.dt = dt
        self.field = field
        
    def forward(self, state, next_state, next_next_state):
        # Compute numerical derivative
        rhs_der = next_next_state - state
        # Compute derivative
        der = self.field(next_state)
        # Compute loss of derivative function
        dyn_loss = torch.mean((rhs_der - 2*self.dt*der)**2)
        # Compute initial condition loss
        ic_loss = torch.mean((next_state[:,0,:]-state[:,1,:])**2)          
        # Compute total physical informed loss
        pi_loss = dyn_loss + ic_loss      
                
        return pi_loss
    
        
    

### R2 score

def R2Score(net_states, true_states):
    """
    Compute R2 score of a generate dynamics with respect target outputs.
    Inputs are np.array of size = (seq_len, dim)
    """
    mean = np.mean(net_states, axis=0)
    mean = np.expand_dims(mean, axis=0)
    num = np.sum((true_states - net_states)**2)
    den = np.sum((true_states - mean)**2)
    
    return 1. - num/den

### Varitaional Autoencoder Functions
class Sampler(nn.Module):
    """
    Sample according to a standard Gaussian with mean = 0 and std = 1 and
    implement reparametrization trick.
    --------
    Returns:
    Torch tensor 
    """
    def __init__(self):
        super(Sampler, self).__init__()
        
    def forward(self, mean, log_var): 
        randn = torch.randn_like(log_var)
        return mean + randn*torch.exp(0.5*log_var)


class nKLDivLoss(nn.Module):
    """
    Compute Kulback-Libler divergence loss for a gaussian variational autoencoder
    """
    def __init__(self,):
        super(nKLDivLoss, self).__init__()
        
    def forward(self,  mean, log_var):
        return torch.sum(0.5*torch.sum(torch.exp(log_var) + mean**2 - 1 - log_var, dim=1), dim = 0)
   

