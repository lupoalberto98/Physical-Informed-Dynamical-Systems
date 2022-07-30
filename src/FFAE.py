"""
@author : Alberto Bassi
"""

#!/usr/bin/env python3
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from utils import Sampler, nKLDivLoss
import numpy as np
import math
import warnings
import utils
from FFNet import FFNet
import models

class FFAE(pl.LightningModule):
    def __init__(self,
                 n_inputs,
                 n_outputs,
                 encoded_space_dim,
                 hidden_layers, 
                 true_system,
                 drop_p = 0.3, 
                 lr = 0.001,
                 dt = 0.002,
                 enc_space_reg = "PI",
                 weight_decay = 0, 
                 beta = 1, 
                 reconstruct = True):
        """
        Initialize a typical feedforward network with different hidden layers
        The input is typically a mnist image, given as a torch tensor of size = (1,784),
        or a sequence, torch.tensor of size (1, seq_length, 3)
        Args:
            n_inputs : input features
            n_outputs : output features
            hidden_layers : list of sizes of the hidden layers
            true_system : ground thruth hypothesis class
            drop_p : dropout probability
            lr : learning rate
            dt : time discretization
            feedforward_steps : number of prediction steps
            enc_space_reg : regularzation of encoded space
            weight_decay : l2 regularization constant
            beta : weight for regularization loss
            reconstruct : if True reconstruct input, else predict next step
        """
        super().__init__()
        # Parameters
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.encoded_space_dim = encoded_space_dim
        self.hidden_layers = hidden_layers
        self.num_hidden_layers = len(self.hidden_layers)
        self.true_system = true_system
        self.system_name = true_system.__class__.__name__ #string specifing the system being used
        self.system_dim = true_system.dim # dimension of the system
        self.num_param = len(true_system.params) # number of true parameters
        self.drop_p = drop_p
        self.lr = lr
        self.dt = dt
        self.enc_space_reg = enc_space_reg
        self.weight_decay = weight_decay
        self.beta = beta
        self.reconstruct = reconstruct 
        
        # Network architecture
        self.encoder = FFNet(self.n_inputs, self.encoded_space_dim, self.hidden_layers, system=self.true_system, 
                             true_system=self.true_system, drop_p=self.drop_p)
        
        self.hidden_layers.reverse()
        self.decoder = FFNet(self.encoded_space_dim, self.n_outputs, self.hidden_layers, 
                             system=self.true_system,true_system=self.true_system, drop_p=self.drop_p)
        ### Checkings 
        if self.enc_space_reg == None:
            warnings.warn("No encoded space regualarization used")
        else:
            if self.encoded_space_dim < self.num_param:
                raise ValueError("Encoded space dimension too small to be regularized with PI loss")
                
        print("Feedforward Autoencoder initialized")
        
    def forward(self, t, x):
        enc = self.encoder(t, x)
        rec = self.decoder(t, enc)
        return enc, rec
    
    def training_step(self, batch, batch_idx):
        """
        Input is dataloader output, a tensor of size (batch_size, seq_length, feature_dim) 
        that must be flattened in last dimensions to get a tensor of size (batch_size, seq_length*feature_dim)
        """
        ### Prepare network input and labels and flatten last dimension 
        state  = batch[:, :-1, :]
        state_flat = torch.flatten(state, start_dim=1)
        labels = batch[:, 1:,:]
        labels_flat = torch.flatten(labels, start_dim=1)
        
        # Set reg loss to zero
        reg_loss = 0
        # Forward step
        enc_state, rec_state = self.forward(batch_idx, state_flat)
        # Compute reconstruction loss
        if self.reconstruct:
            rec_loss = nn.MSELoss()(rec_state, state_flat)
        else:
            rec_loss = nn.MSELoss()(rec_state, labels_flat)
        
        # Compute regularization loss
        if self.enc_space_reg == "PI": # self.feedfrward_steps should be 1 here
            # Initialize the system with parameters as the first entries of encoded batch
            system = getattr(models, self.system_name)(params=enc_state[:,:self.num_param].unsqueeze(1))
            # Initialize the physical informed method to compute loss
            method = utils.RK4(self.dt, model=system)
            # Compute differential and error
            df = method(state)
            # Simple regularization loss
            reg_loss += nn.MSELoss()(state+df, labels)*self.beta
            # Noise regularized loss
            self.log("train_reg_loss", reg_loss, prog_bar=True)
        if self.enc_space_reg == "TRUE":
            reg_loss += nn.MSELoss()(enc_state[:,:self.num_param],self.true_system.params.unsqueeze(0))*self.beta
            self.log("train_reg_loss", reg_loss, prog_bar=True)
        
        # Logging to TensorBoard by default
        train_loss = rec_loss + reg_loss
        self.log("train_loss", train_loss, prog_bar=True)
    
        return train_loss
        
    def validation_step(self, batch, batch_idx):
        """
        Input is dataloader output, a tensor of size (batch_size, seq_length, feature_dim) 
        that must be flattened in last dimensions to get a tensor of size (batch_size, seq_length*feature_dim)
        """
        ### Prepare network input and labels and flatten last dimension 
        state  = batch[:, :-1, :]
        state_flat = torch.flatten(state, start_dim=1)
        labels = batch[:, 1:,:]
        labels_flat = torch.flatten(labels, start_dim=1)
        
        # Set reg loss to zero
        reg_loss = 0
        # Forward step
        enc_state, rec_state = self.forward(batch_idx, state_flat)
        # Compute reconstruction loss
        if self.reconstruct:
            rec_loss = nn.MSELoss()(rec_state, state_flat)
        else:
            rec_loss = nn.MSELoss()(rec_state, labels_flat)
        
        # Compute regularization loss
        if self.enc_space_reg == "PI": # self.feedfrward_steps should be 1 here
            # Initialize the system with parameters as the first entries of encoded batch
            system = getattr(models, self.system_name)(params=enc_state[:,:self.num_param].unsqueeze(1))
            # Initialize the physical informed method to compute loss
            method = utils.RK4(self.dt, model=system)
            # Compute differential and error
            df = method(state)
            # Simple regularization loss
            reg_loss += nn.MSELoss()(state+df, labels)*self.beta
            # Noise regularized loss
            self.log("val_reg_loss", reg_loss, prog_bar=True)
        if self.enc_space_reg == "TRUE":
            reg_loss += nn.MSELoss()(enc_state[:,:self.num_param],self.true_system.params.unsqueeze(0))*self.beta
            self.log("val_reg_loss", reg_loss, prog_bar=True)
            
        # Compute reconstruction loss
        val_loss = rec_loss + reg_loss
        # Logging to TensorBoard by default
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("epoch_num", self.current_epoch,prog_bar=True)
        
        return val_loss
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr)
        return optimizer
    
    def num_timesteps(self, time):
        """Returns the number of timesteps required to pass time time.
        Raises an error if timestep value does not divide length time.
        """
        num_timesteps = time / self.dt
        if not num_timesteps.is_integer():
            raise Exception
        return int(num_timesteps)
    