#!/usr/bin/env python3
import torch
import numpy as np
from scipy.integrate import odeint
import torch.nn as nn
import copy
from typing import Union
import math



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
        


# Define the vector field for Lorenz 63 system
class Lorenz63(nn.Module):
    def __init__(self, params, sigma=1., dt=0.01):
        super(Lorenz63, self).__init__()
        self.register_parameter(name="params", param=nn.Parameter(params)) # tensor of size = (3) the same dimension of state, containing parameters (rho, sigma, beta)
        self.sigma = sigma # Standard deviation of gaussian noise
        self.dt = dt
        self.dim = 3
        
        # Torchsde parameters
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        
    def forward(self, t, state): 
        field = torch.clone(state)
        field[...,0] = self.params[...,1]*(state[...,1] - state[...,0])
        field[...,1] = state[...,0]*(self.params[...,0] - state[...,2]) - state[...,1]
        field[...,2] = state[...,0]*state[...,1] - self.params[...,2]*state[...,2]
        
        return field
    
    ### Functions for torchsde stochastic differential equation solver
    def f(self, t, state):
        field = torch.clone(state)
        field[...,0] = self.params[...,1]*(state[...,1] - state[...,0])
        field[...,1] = state[...,0]*(self.params[...,0] - state[...,2]) - state[...,1]
        field[...,2] = state[...,0]*state[...,1] - self.params[...,2]*state[...,2]
        
        return field
    
    def g(self, t, state):
        field = torch.ones_like(state)
        return self.sigma/math.sqrt(2*self.dt)*field
    
    ### Jacobian 
    def jacobian(self, state, t):
        """
        Compute the jacobian at a given state of the trajectory
        """
        x, y, z = state
        jac = np.zeros((self.dim,self.dim)) 
        # Compute derivatives
        jac[0][0] =  -self.params[1]
        jac[0][1] = self.params[1]
        jac[1][0] = self.params[0] - z
        jac[1][1] = -1.
        jac[1][2] = - x
        jac[2][0] = y
        jac[2][1] = x
        jac[2][2] = -self.params[2]

        return jac

# Vector field of Rossel system
class Roessler76(nn.Module):
    def __init__(self, params, noise=None):
        super(Roessler76, self).__init__()
        self.register_parameter(name="params", param=nn.Parameter(params))
        self.noise = noise
        self.dim = 3
        
    def forward(self, t, state):
        field = torch.clone(state)
        field[...,0] = - state[...,1] - state[...,2]
        field[...,1] = state[...,0] + self.params[0]*state[...,1]
        field[...,2] = self.params[1] + state[...,2]*(state[...,0] - self.params[2])
            
        return field

    def jacobian(self, state, t):
        x, y, z = state
        jac = np.zeros((self.dim,self.dim)) 
        jac[0][1] = -1.
        jac[0][2] = -1.
        jac[1][0] = 1.
        jac[1][1] = self.params[0]
        jac[2][0] = z
        jac[2][2] = x
        
        return jac
    
    
# Lorenz96 model (perturbed)
class Lorenz96(nn.Module):
    def __init__(self, dim, params, noise=None):
        """
        n_dim is the number of nodes
        force is the force, list of size n_dim
        args is a tensor of small perturbation (noise) in force and 
        """
        super(Lorenz96, self).__init__()
        self.register_parameter(name="params", param=nn.Parameter(params))
        self.dim = dim
        self.noise = noise
    
    def forward(self, t, state):
        field = torch.clone(state)
        # Standard Lorenz96
        for i in range(self.dim):
            field[...,i] = self.params[0,i]*(state[...,(i+1)%self.dim]-state[...,i-2])*state[...,i-1]-self.params[1,i]*state[...,i] + self.params[2,i]
       
        return field
    

    
# Loss classes
class Eul(nn.Module):
    """
    Compute forward pass with Euler scheme
    Args:
    dt is the time step
    if time_included, time is carried as last element of input vector
    """
    def __init__(self, dt, model):
        super(Eul, self).__init__()
        self.dt = dt
        self.model = model

    def forward(self, state):
        """
        If state and next_state are the ground truth, then model is the NN ansatz.
        Else, if next_state is the NN forwarded, then model should be the ground truth
        """
        t = 10
        # Propagate
        df = self.model(t, state)*self.dt
        
        return df
    
class RK4(nn.Module):
    """
    Implement a 4th order Runge-Kutta method to compute loss
    """
    def __init__(self, dt, model):
        super(RK4, self).__init__()
        self.dt = dt
        self.model = model
        
    def forward(self, state):
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
   

