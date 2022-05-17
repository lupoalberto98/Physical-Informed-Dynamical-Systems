#!/usr/bin/env python3
import torch
import numpy as np
from scipy.integrate import odeint
import torch.nn as nn
import copy
from typing import Union




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
class Lorenz63():
    def __init__(self,  rho, sigma, beta):
        self.rho=rho
        self.sigma=sigma
        self.beta=beta
        self.dim = 3
        
        
    def __call__(self, state, t):
        x, y, z = state
        return (self.sigma*(y-x), x*(self.rho-z)-y, x*y-self.beta*z)
    
    def field(self, state): 
        field = torch.clone(state)
        field[...,0] = self.sigma*(state[...,1] - state[...,0])
        field[...,1] = state[...,0]*(self.rho - state[...,2]) - state[...,1]
        field[...,2] = state[...,0]*state[...,1] - self.beta*state[...,2]

        return field
    
    def jacobian(self, state, t):
        """
        Compute the jacobian at a given state of the trajectory
        """
        x, y, z = state
        jac = np.zeros((self.dim,self.dim)) 
        # Compute derivatives
        jac[0][0] =  -self.sigma
        jac[0][1] = self.sigma
        jac[1][0] = self.rho - z
        jac[1][1] = -1.
        jac[1][2] = - x
        jac[2][0] = y
        jac[2][1] = x
        jac[2][2] = -self.beta

        return jac

# Vector field of Rossel system
class Roessler():
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        self.dim = 3
        
    
    def __call__(self, state, t):
        x, y, z = state
        return (-y-z, x+self.a*y, self.b + z*(x-self.c))
    
    
    def field(self, state):
        field = torch.clone(state)
        field[...,0] = - state[...,1] - state[...,2]
        field[...,1] = state[...,0] + self.a*state[...,1]
        field[...,2] = self.b + state[...,2]*(state[...,0] - self.c)
        
        return field

    def jacobian(self, state, t):
        x, y, z = state
        jac = np.zeros((self.dim,self.dim)) 
        jac[0][1] = -1.
        jac[0][2] = -1.
        jac[1][0] = 1.
        jac[1][1] = self.a
        jac[2][0] = z
        jac[2][2] = x
        
        return jac
    
    
# Lorenz96 model (perturbed)
class Lorenz96(nn.Module):
    def __init__(self, args):
        """
        n_dim is the number of nodes
        force is the force, list of size n_dim
        a,b,c are small perturbation, lists of size n_dim
        """
        super(Lorenz96, self).__init__()
        self.register_parameter(name="args", param=nn.Parameter(args))
        self.dim = len(self.args[0])
        
       
    
    def forward(self, state, t):
        field = []
        for i in range(self.dim):
            i_up = (i+1)%self.dim
            i_down = (i-1)%self.dim
            i_down2 = (i-2)%self.dim
            der = (state[i_up] - state[i_down2])*state[i_down] - self.args.detach().numpy()[0,i]*state[i] + self.args.detach().numpy()[1,i]
            field.append(der)
         
        return field
    
    
    def field(self, state):
        field = torch.clone(state)
        for i in range(self.dim):
            i_up = (i+1)%self.dim
            i_down = (i-1)%self.dim
            i_down2 = (i-2)%self.dim
            field[...,i] = (state[...,i_up]-state[...,i_down2])*state[...,i_down]-self.args[0][i]*state[...,i] + self.args[1][i]
         
        return field
    
   
    
# Loss classes
class EuDLoss(nn.Module):
    """
    Compute the physical informed loss for a generic dynamical system with Euler scheme
    Args:
    dt is the time step
    field is the field of the dyanamical system
    if time_included, time is carried as last element of input vector
    """
    def __init__(self, dt, model, include_time=False):
        super(EuDLoss, self).__init__()
        self.dt = dt
        self.model = model
        self.include_time = include_time
  

    def forward(self, state, next_state):
        dim = state.shape[-1]
        # Compute derivative term
        der = self.model.field(state)
        # Compute derivative
        rhs_der = next_state - state
        # Compute loss of the derivative function
        if self.include_time:
            # Compute time differences and expand
            dt = state[:,:, -1]
            dt = dt.unsqueeze(-1)
            # Compute loss
            dyn_loss = torch.mean((rhs_der[:,:,:dim-1]-der[:,:,:dim-1]*dt)**2)
        else:
            dyn_loss = torch.mean((rhs_der[:,:,:dim]-der[:,:,:dim]*self.dt)**2)   
            
        # Compute initial condition loss
        ic_loss = torch.mean((next_state[:,0,:]-state[:,1,:])**2)           
       
        # Compute total physical informed loss
        pi_loss = dyn_loss + ic_loss 
        
        return pi_loss
    
class RK4Loss(nn.Module):
    """
    Implement a 4th order Runge-Kutta method to compute loss
    """
    def __init__(self, dt, model):
        super(RK4Loss, self).__init__()
        self.dt = dt
        self.model = model
        
    def forward(self, state, next_state):
        # Propagate
        k1 = self.model.field(state)
        k2 = self.model.field(state+self.dt*k1/2.)
        k3 = self.model.field(state+self.dt*k2/2.)
        k4 = self.model.field(state+self.dt*k3)
        # Compute Rk4 loss
        rk4_loss = torch.mean((6*(next_state-state)-self.dt*(k1+2*k2+2*k3+k4))**2)
        # Compute initial condition loss
        ic_loss = torch.mean((next_state[:,0,:]-state[:,1,:])**2)    
        
        return rk4_loss + ic_loss

    
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
   

