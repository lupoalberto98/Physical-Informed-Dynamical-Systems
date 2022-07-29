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



# Define the vector field for Lorenz 63 system
class Lorenz63(nn.Module):
    def __init__(self, params, sigma=None, dt=0.002):
        super(Lorenz63, self).__init__()
        #self.register_parameter(name="params", param=nn.Parameter(params))
        self.params = params # tensor of size = (3) the same dimension of state, containing parameters (rho, sigma, beta)
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
    def __init__(self, params, sigma=None, dt=0.002):
        super(Roessler76, self).__init__()
        self.params = params
        self.sigma = sigma
        self.dt = dt
        self.dim = 3
        
        # Torchsde parameters
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        
    def forward(self, t, state):
        field = torch.clone(state)
        field[...,0] =  -state[...,1] - state[...,2]
        field[...,1] = state[...,0] + self.params[0]*state[...,1]
        field[...,2] = self.params[1] + state[...,2]*(state[...,0] - self.params[2])
            
        return field
    
    ### Functions for torchsde stochastic differential equation solver
    def f(self, t, state):
        field = torch.clone(state)
        field[...,0] =  -state[...,1] - state[...,2]
        field[...,1] = state[...,0] + self.params[0]*state[...,1]
        field[...,2] = self.params[1] + state[...,2]*(state[...,0] - self.params[2])
            
        return field

    def g(self, t, state):
        field = torch.ones_like(state)
        return self.sigma/math.sqrt(2*self.dt)*field
    
    def jacobian(self, state, t):
        x, y, z = state
        jac = np.zeros((self.dim,self.dim)) 
        jac[0][1] = -1.
        jac[0][2] = -1.
        jac[1][0] = 1.
        jac[1][1] = self.params[0]
        jac[2][0] = z
        jac[2][2] = x - self.params[2]
        
        return jac
    
    
# Lorenz96 model (perturbed)
class Lorenz96(nn.Module):
    def __init__(self, dim, params, sigma=None, dt=0.01):
        """
        n_dim is the number of nodes
        force is the force, list of size n_dim
        args is a tensor of small perturbation (noise) in force and 
        """
        super(Lorenz96, self).__init__()
        self.params = params
        self.dim = dim
        self.sigma = sigma
        self.dt = dt
    
        # Torchsde parameters
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        
    def forward(self, t, state):
        field = torch.clone(state)
        # Standard Lorenz96
        for i in range(self.dim):
            field[...,i] = self.params[0,i]*(state[...,(i+1)%self.dim]-state[...,i-2])*state[...,i-1]-self.params[1,i]*state[...,i] + self.params[2,i]
       
        return field
    
    ### Functions for torchsde stochastic differential equation solver
    def f(self, t, state):
        field = torch.clone(state)
        # Standard Lorenz96
        for i in range(self.dim):
            field[...,i] = self.params[0,i]*(state[...,(i+1)%self.dim]-state[...,i-2])*state[...,i-1]-self.params[1,i]*state[...,i] + self.params[2,i]
       
        return field

    def g(self, t, state):
        field = torch.ones_like(state)
        return self.sigma/math.sqrt(2*self.dt)*field