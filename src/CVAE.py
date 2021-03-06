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
from LSTM import LSTM
from FFNet import FFNet
import utils
import warnings


### Convolutional Variational Autoencoder
class CVEncoder(pl.LightningModule):
    
    def __init__(self, in_channels, out_channels, kernel_sizes, padding=(0,0), encoded_space_dim=2, 
                 drop_p=0.1, act=nn.ReLU, seq_len=100, feedforward_steps=1):
        """
        Convolutional Encoder layer, with 2 convolution layers
        """
        super().__init__()
      
        # Retrieve parameters
        self.in_channels = in_channels #tuple of int, input channels for convolutional layers
        self.out_channels = out_channels #tuple of int, of output channels 
        self.kernel_sizes = kernel_sizes #tuple of tuples of int kernel size, single integer or tuple itself
        self.padding = padding
        self.encoded_space_dim = encoded_space_dim
        self.drop_p = drop_p
        self.act = act
        self.seq_len = seq_len
        self.feedforward_steps = feedforward_steps
       
        """
        Numerical values as example:
        in_channels = (1, 4)
        out_channels = (4, 4)
        kernel_sizes = ((5, 3), (5,1))
        padding = (0,0)
        input = (20, 1, 500, 3) # (N, C, H, W)
        Output of the first layer:
        cl1 = (20, 16, 467, 1)
        cl2 = (20, 32, 434, 1)
        """
        
        ### Network architecture
        # First convolutional layer (2d convolutional layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(self.in_channels[0], self.out_channels[0], self.kernel_sizes[0], padding=self.padding[0]), 
            nn.BatchNorm2d(self.out_channels[0]),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            nn.MaxPool2d((2,1), return_indices=True)
        )
        
        # Second convolution layer
        self.second_conv = nn.Sequential(
            nn.Conv2d(self.in_channels[1], self.out_channels[1], self.kernel_sizes[1], padding=self.padding[1]), 
            nn.BatchNorm2d(self.out_channels[1]),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            nn.MaxPool2d((2,1), return_indices=True)
        )
        
        # Third convolutional layer
        self.third_conv = nn.Sequential(
            nn.Conv2d(self.in_channels[2], self.out_channels[2], self.kernel_sizes[2], padding=self.padding[2]), 
            nn.BatchNorm2d(self.out_channels[2]),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            nn.MaxPool2d((2,1), return_indices=True)
        )
        
        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        # Liner dimension after 2 convolutional layers
        self.lin_dim = int((((self.seq_len-self.feedforward_steps-self.kernel_sizes[0][0]+1)/2+1-self.kernel_sizes[1][0])/2+1-self.kernel_sizes[2][0])/2)
        
        # Linear parameter encoder
        self.encoder_param = nn.Sequential(
            # First linear layer
            nn.Linear(self.out_channels[2]*self.lin_dim, 256),
            nn.BatchNorm1d(256),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            # Second linear layer
            nn.Linear(256, self.encoded_space_dim)
        )
        
        # Encoder mean
        self.encoder_mean = nn.Sequential(
            # First linear layer
            nn.Linear(self.out_channels[2]*self.lin_dim, 256),
            nn.BatchNorm1d(256),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            # Second linear layer
            nn.Linear(256, 1)
        )
        
         # Encoder mean
        self.encoder_logvar = nn.Sequential(
            # First linear layer
            nn.Linear(self.out_channels[2]*self.lin_dim, 256),
            nn.BatchNorm1d(256),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            # Second linear layer
            nn.Linear(256, 1)
        )
        
            
        
    def forward(self, x):
        # Apply first convolutional layer
        x, indeces_1 = self.first_conv(x)
        # Apply second convolutional layer
        x, indeces_2 = self.second_conv(x)
        # Apply third conv layer
        x, indeces_3 = self.third_conv(x)
        # Flatten 
        x = self.flatten(x)
        # Apply linear encoder layer
        param = self.encoder_param(x)
        mean = self.encoder_mean(x)
        logvar = self.encoder_logvar(x)
        return param, mean, logvar, indeces_1, indeces_2, indeces_3
    

class CVDecoder(pl.LightningModule):
    
    def __init__(self, in_channels, out_channels, kernel_sizes, padding=(0,0), encoded_space_dim=2, drop_p=0.1, act=nn.ReLU, seq_len=100, feedforward_steps=1):
        
        super().__init__()
        
        # Retrieve parameters
        self.in_channels = in_channels #tuple of int, input channels for convolutional layers
        self.out_channels = out_channels #tuple of int, of output channels 
        self.kernel_sizes = kernel_sizes #tuple of tuples of int kernel size, single integer or tuple itself
        self.encoded_space_dim = encoded_space_dim
        self.drop_p = drop_p
        self.act = act
        self.seq_len = seq_len
        self.feedforward_steps = feedforward_steps
        
        ### Network architecture
        # Linear dimension
        self.lin_dim = int((((self.seq_len-self.feedforward_steps-self.kernel_sizes[0][0]+1)/2+1-self.kernel_sizes[1][0])/2+1-self.kernel_sizes[2][0])/2)
        
        # Linear decoder
        self.decoder_param = nn.Sequential(
            # First linear layer
            nn.Linear(self.encoded_space_dim, 256),
            nn.Dropout(self.drop_p, inplace = False),
            self.act(inplace = False),
            nn.BatchNorm1d(256),
            # Second linear layer
            nn.Linear(256, self.lin_dim*self.out_channels[2]),
        )
        
        # Linear decoder
        self.decoder_noise = nn.Sequential(
            # First linear layer
            nn.Linear(1, 256),
            nn.Dropout(self.drop_p, inplace = False),
            self.act(inplace = False),
            nn.BatchNorm1d(256),
            # Second linear layer
            nn.Linear(256, self.lin_dim*self.out_channels[2]),
        )
        
        # Unflatten layer
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(self.out_channels[2],self.lin_dim, 1))
        
        # Unpooling layer
        self.unpool = nn.MaxUnpool2d((2,1))
        
        self.first_deconv = nn.Sequential(
            # First transposed convolution
            nn.Dropout(self.drop_p, inplace = False),
            self.act(inplace = True),
            nn.BatchNorm2d(self.out_channels[2]),
            nn.ConvTranspose2d(self.out_channels[2], self.in_channels[2], self.kernel_sizes[2])    
        )
        
        self.second_deconv = nn.Sequential(
            nn.Dropout(self.drop_p, inplace = False),
            self.act(inplace = True),
            nn.BatchNorm2d(self.out_channels[1]),
            nn.ConvTranspose2d(self.out_channels[1], self.in_channels[1], self.kernel_sizes[1]) 
        )
        
        self.third_deconv = nn.Sequential(
            nn.Dropout(self.drop_p, inplace = False),
            self.act(inplace = True),
            nn.BatchNorm2d(self.out_channels[0]),
            nn.ConvTranspose2d(self.out_channels[0], self.in_channels[0], self.kernel_sizes[0]) 
        )
        
    def forward(self, param, noise, indeces_1, indeces_2, indeces_3):
        
        # Apply linear decoder
        dec_param = self.decoder_param(param)     
        dec_noise = self.decoder_noise(noise)
        # Unflatte layer
        x = dec_param + dec_noise
        x = self.unflatten(x)   
        # Apply first unpooling layer
        x = self.unpool(x,  indeces_3)
        # Apply first deconvolutional layer
        x = self.first_deconv(x)    
        # Apply second unpooling layer
        x = self.unpool(x, indeces_2)
        # Apply second deconvolutional layer
        x = self.second_deconv(x)   
        # Apply third unpooling layer
        x = self.unpool(x, indeces_1)
        # Apply third deconvolutional layer
        x = self.third_deconv(x)
        return x
        
### Symmetric convolutional variational autoencoder
class CVAE(pl.LightningModule):
    
    def __init__(self, in_channels, out_channels, kernel_sizes, 
           padding=(0,0),  encoded_space_dim=1, act=nn.ReLU, drop_p=0.1, seq_len=100, feedforward_steps=1, lr=0.001, dt=0.01, system_name="Lorenz63", system_dim=3, num_param=3, enc_space_reg="PI", beta=1.0):
        
        super().__init__()
        self.encoder = CVEncoder(in_channels, out_channels, kernel_sizes, padding, encoded_space_dim, 
                 drop_p, act, seq_len, feedforward_steps)
        self.decoder = CVDecoder(in_channels, out_channels, kernel_sizes, padding, encoded_space_dim, drop_p, act, seq_len, feedforward_steps)
        
        self.seq_len = seq_len
        self.feedforward_steps = feedforward_steps
        self.lr = lr
        self.dt = dt
        self.encoded_space_dim = encoded_space_dim
        self.system_name = system_name #string specifing the system being used
        self.system_dim = system_dim
        self.num_param = num_param
        self.enc_space_reg = enc_space_reg # string specifing ho to compute loss
        # Weights for regularization losses
        self.beta = beta
        
        ### Checkings 
        if self.enc_space_reg == "PI":
            if self.encoded_space_dim < self.num_param:
                raise ValueError("Encoded space dimension too small to be regularized with PI loss")
        else:
            warnings.warn("No encoded space regualarization used")
        
    def forward(self, x):
        # Encode data and keep track of indexes
        param, mean, logvar, indeces_1, indeces_2, indeces_3 = self.encoder(x)
        # Sampling
        noise = Sampler()(mean, logvar)
        # Decode data
        rec = self.decoder(param, noise, indeces_1, indeces_2, indeces_3)
        return (param, noise, rec)

    def training_step(self, batch, batch_idx):
        # Unsqueeze batch
        batch = batch.unsqueeze(1)
        ### Prepare network input and labels 
        state  = batch[:,:, :self.seq_len-self.feedforward_steps, :]
        labels = batch[:,:, self.feedforward_steps:, :]
        
        # Set reg loss to zero
        reg_loss = 0
        rec_loss = 0
        # Forward step
        enc_state, noise, rec_state = self.forward(state)
        
        # Compute regularization loss
        if self.enc_space_reg is not None: # self.feedfrward_steps should be 1 here
            # Initialize the system with parameters as the first entries of encoded batch
            system = getattr(utils, self.system_name)(params=enc_state[:,:self.num_param].unsqueeze(1).unsqueeze(1))
            # Initialize the physical informed method to compute loss
            method = utils.RK4(self.dt, model=system)
            # Compute differential and error
            df = method(state)
            # Simple regularization loss
            reg_loss += nn.MSELoss()(state+df+noise.unsqueeze(1).unsqueeze(1), labels)*self.beta
            # Noise regularized loss
            self.log("train_reg_loss", reg_loss, prog_bar=True)
            # Compute reconstruction loss
            rec_loss = nn.MSELoss()(rec_state+noise.unsqueeze(1).unsqueeze(1), state)
        else:
            # Compute reconstruction loss
            rec_loss = nn.MSELoss()(rec_state, state)
        
        # Logging to TensorBoard by default
        train_loss = rec_loss + reg_loss
        self.log("train_loss", train_loss, prog_bar=True)
    
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        # Unsqueeze batch
        batch = batch.unsqueeze(1)
        ### Prepare network input and labels 
        state  = batch[:,:, :self.seq_len-self.feedforward_steps, :]
        labels = batch[:,:, self.feedforward_steps:, :]
        # Set reg loss to zero
        reg_loss = 0
        rec_loss = 0
        # Forward step
        enc_state, noise, rec_state = self.forward(state)
        
        
        # Compute regularization loss
        if self.enc_space_reg is not None: # self.feedfrward_steps should be 1 here
            # Initialize the system with parameters as the first entries of encoded batch
            system = getattr(utils, self.system_name)(params=enc_state[:,:self.num_param].unsqueeze(1).unsqueeze(1))
            # Initialize the physical informed method to compute loss
            method = utils.RK4(self.dt, model=system)
            # Compute differential and error
            df = method(state)
            # Simple regularization loss
            reg_loss += nn.MSELoss()(state+df+noise.unsqueeze(1).unsqueeze(1), labels)*self.beta
            # Noise regularized loss
            self.log("val_reg_loss", reg_loss, prog_bar=True)
            # Compute reconstruction loss
            rec_loss = nn.MSELoss()(rec_state+noise.unsqueeze(1).unsqueeze(1), state)
        else:
            # Compute reconstruction loss
            rec_loss = nn.MSELoss()(rec_state, state)
            
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