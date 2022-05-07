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

# Physical informed loss classes
class EuDLoss(nn.Module):
    """
    Compute the physical informed loss for a generic dynamical system with Euler scheme
    Args:
    dt is the time step
    field is the field of the dyanamical system
    dd_loss_fn is the loss function of data driven loss
    dd_mask is the data driven loss mask
    pi_mask is the physical informed loss mask
    """
    def __init__(self, dt, field, dd_loss_fn, dd_mask, pi_mask):
        super(EuDLoss, self).__init__()
        self.dt = dt
        self.field = field
        self.dd_loss_fn = dd_loss_fn
        self.dd_mask = dd_mask # Wights for data driven loss
        self.pi_mask = pi_mask # Weights for physical informed loss

    def forward(self, state, next_state, labels, num_epoch):
        # Compute derivative term
        der = self.field(state)
        # Compute derivative
        rhs_der = next_state - state
        # Compute loss of the derivative function
        dyn_loss = torch.mean((rhs_der-der*self.dt)**2)       
        # Compute initial condition loss
        ic_loss = torch.mean((next_state[:,0,:]-state[:,1,:])**2)    
        # Compute data driven loss
        dd_loss = self.dd_loss_fn(next_state,labels)       
        # Compute total physical informed loss
        pi_loss = dyn_loss + ic_loss      
        # Compute total loss
        loss = pi_loss*self.pi_mask[num_epoch] + dd_loss*self.dd_mask[num_epoch]          
        return loss

class CeDLoss(nn.Module):
    """
    Compute the physical informed loss of a dynamical system with central derivative
    Args:
    dt is the time step
    field is the field of the dyanamical system
    dd_loss_fn is the loss function of data driven loss
    dd_mask is the data driven loss mask
    pi_mask is the physical informed loss mask
    """
    def __init__(self,  dt, field, dd_loss_fn, dd_mask, pi_mask):
        super(CeDLoss, self).__init__()
        self.dt = dt
        self.field = field
        self.dd_loss_fn = dd_loss_fn
        self.dd_mask = dd_mask # Wights for data driven loss
        self.pi_mask = pi_mask # Weights for physical informed loss
        
    def forward(self, state, next_state, next_next_state, labels, num_epoch):
        # Compute numerical derivative
        rhs_der = next_next_state - state
        # Compute derivative
        der = self.field(next_state)
        # Compute loss of derivative function
        dyn_loss = torch.mean((rhs_der - 2*self.dt*der)**2)
        # Compute initial condition loss
        ic_loss = torch.mean((next_state[:,0,:]-state[:,1,:])**2)    
        # Compute data driven loss
        dd_loss = self.dd_loss_fn(next_state,labels)       
        # Compute total physical informed loss
        pi_loss = dyn_loss + ic_loss      
        # Compute total loss
        loss = pi_loss*self.pi_mask[num_epoch] + dd_loss*self.dd_mask[num_epoch]          
        return loss