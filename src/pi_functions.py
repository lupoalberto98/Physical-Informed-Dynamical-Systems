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
        
    def __call__(self,state):
        field = torch.clone(state)
        field[...,0] = self.sigma*torch.sub(state[...,1],state[...,0])
        field[...,1] = torch.sub(state[...,0]*torch.sub(self.rho,state[...,2]),state[...,1])
        field[...,2] = torch.sub(state[...,0]*state[...,1],self.beta*state[...,2])

        return field

# Physical informed loss 
class PILoss(nn.Module):
    """
    Compute the physical informed loss for a generic dynamical system
    Args:
    dt is the time step
    field is the field of the dyanamical system
    annealing is the annealing strategy used to weight data driven (dd) loss
    """
    def __init__(self, dt, field, annealing):
        super(PILoss, self).__init__()
        self.dt = dt
        self.field = field
        self.annealing = annealing

    def forward(self, state, next_state, labels, num_epoch):
        # Compute derivative term
        der_term = self.field(state)
        # Compute rhs
        rhs = state + self.dt*der_term
        # Compute loss of the derivative function
        pi_loss = torch.mean((next_state-rhs)**2)
        
        # Compute initial condition loss
        ic_loss = 0 #torch.mean((next_state[:,1,:]-state[:,0,:])**2)
    
        # Compute data driven loss
        dd_loss = nn.MSELoss()(next_state,labels)
        
        # Compute total loss
        loss = pi_loss + ic_loss + self.annealing[num_epoch]*dd_loss
            
        return loss

