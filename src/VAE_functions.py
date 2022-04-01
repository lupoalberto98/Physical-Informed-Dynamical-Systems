import torch
import torch.nn as nn
import numpy as np



def Sampler(mean, log_var):
   """
   Sample according to a standard Gaussian with mean = 0 and std = 1 and
    implement reparametrization trick.
   --------
   Returns:
   Torch tensor 
   """
   randn = torch.randn_like(log_var)
   return mean + randn*torch.exp(0.5*log_var)


def nKLDivLoss(mean, log_var):
   """
   Compute Kulback-Libler divergence loss for a gaussian variational autoencoder
   """
   
   return torch.sum(0.5*torch.sum(torch.exp(log_var) + mean**2 - 1 - log_var, dim=1), dim = 0)
   

