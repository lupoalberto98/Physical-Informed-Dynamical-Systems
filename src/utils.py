#!/usr/bin/env python3
import torch
import numpy as np
from scipy.integrate import odeint
import torch.nn as nn


# Define a function to create the initial condition
class Initializer():
    def __init__(self, dt, batch_size, len_seq, f):
        self.dt = dt
        self.batch_size = batch_size
        self.len_seq = len_seq
        self.f = f
        
    def __call__(self):
        states_seq = []
        for batch in range(self.batch_size):
            # Uniformly choose initial condition
            state0 = [1.0,1.0,1.0]
            
            # Genrate vector of time steps
            t = np.arange(0.0, dt*self.len_seq, dt)
            
            # Integrate initial dynamics
            states = odeint(f, state0, t)
            
            # Append
            states_seq.append(states)
        
        # Convert to numpy array first
        states_seq = np.array(states_seq)
        # Convert to torch tensor
        states_seq = torch.tensor(states_seq,dtype=torch.float, requires_grad=True)
        
        return states_seq
        


# Define the vector field for Lorenz 63 system, torch tensor
class L63_field():
    def __init__(self,  rho, sigma, beta):
        self.rho=rho
        self.sigma=sigma
        self.beta=beta
        self.dim = 3
        
    def __call__(self,state):
        field = torch.clone(state)
        field[...,0] = self.sigma*torch.sub(state[...,1],state[...,0])
        field[...,1] = torch.sub(state[...,0]*torch.sub(self.rho,state[...,2]),state[...,1])
        field[...,2] = torch.sub(state[...,0]*state[...,1],self.beta*state[...,2])

        return field

# Physical informed loss classes
class EuDLoss(nn.Module):
    """
    Compute the physical informed loss for a generic dynamical system with Euler scheme
    Args:
    dt is the time step
    field is the field of the dyanamical system
    if time_included, time is carried as last element of input vector
    """
    def __init__(self, dt, field, include_time=False):
        super(EuDLoss, self).__init__()
        self.dt = dt
        self.field = field
        self.include_time = include_time
  

    def forward(self, state, next_state):
        # Compute derivative term
        der = self.field(state)
        # Compute derivative
        rhs_der = next_state - state
        # Compute loss of the derivative function
        if self.include_time:
            # Compute time differences and expand
            dt = next_state[:,:, -1] - state[:,:, -1]
            dt = dt.unsqueeze(-1)
            # Compute loss
            dyn_loss = torch.mean((rhs_der[:,:,:self.field.dim]-der[:,:,:self.field.dim]*dt)**2)
        else:
            dyn_loss = torch.mean((rhs_der[:,:,:self.field.dim]-der[:,:,:self.field.dim]*self.dt)**2)       
        # Compute initial condition loss
        ic_loss = torch.mean((next_state[:,0,:]-state[:,1,:])**2)           
        # Compute total physical informed loss
        pi_loss = dyn_loss + ic_loss      
           
        return pi_loss

class CeDLoss(nn.Module):
    """
    Compute the physical informed loss of a dynamical system with central derivative
    Args:
    dt is the time step
    field is the field of the dyanamical system
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
   

