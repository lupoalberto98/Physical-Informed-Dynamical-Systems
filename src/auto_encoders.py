#!/usr/bin/env python3
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from utils import Sampler, nKLDivLoss
import numpy as np
import math
from models import FFNet

### Convolutional Autoencoder
class ConvEncoder(pl.LightningModule):
    
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
            nn.Dropout(self.drop_p, inplace = False)
        )
        
        # Second convolution layer
        self.second_conv = nn.Sequential(
            nn.Conv2d(self.in_channels[1], self.out_channels[1], self.kernel_sizes[1], padding=self.padding[1]), 
            nn.BatchNorm2d(self.out_channels[1]),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False)
        )
        
        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        
        # Linear encoder
        self.encoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(self.out_channels[1]*(self.seq_len-self.feedforward_steps+2-self.kernel_sizes[0][0]-self.kernel_sizes[1][0]), 256),
            nn.BatchNorm1d(256),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            # Second linear layer
            nn.Linear(256, self.encoded_space_dim)
        )

        
    def forward(self, x):
        # Apply first convolutional layer
        x = self.first_conv(x)
        # Apply second convolutional layer
        x = self.second_conv(x)
        # Flatten 
        x = self.flatten(x)
        # Apply linear encoder layer
        x = self.encoder_lin(x)
        return x
    


    
class ConvDecoder(pl.LightningModule):
    
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
        # Linear decoder
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(self.encoded_space_dim, 256),
            nn.Dropout(self.drop_p, inplace = False),
            self.act(inplace = False),
            nn.BatchNorm1d(256),
            # Second linear layer
            nn.Linear(256, self.out_channels[1]*(self.seq_len-self.feedforward_steps+2-self.kernel_sizes[0][0]-self.kernel_sizes[1][0]))
        )
        
        # Unflatten layer
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(self.out_channels[1],self.seq_len-self.feedforward_steps+2-self.kernel_sizes[0][0]-self.kernel_sizes[1][0] , 1))
        
        self.first_deconv = nn.Sequential(
            # First transposed convolution
            nn.Dropout(self.drop_p, inplace = False),
            self.act(inplace = True),
            nn.BatchNorm2d(self.out_channels[1]),
            nn.ConvTranspose2d(self.out_channels[1], self.in_channels[1], self.kernel_sizes[1])    
        )
        
        self.second_deconv = nn.Sequential(
            nn.Dropout(self.drop_p, inplace = False),
            self.act(inplace = True),
            nn.BatchNorm2d(self.out_channels[0]),
            nn.ConvTranspose2d(self.out_channels[0], self.in_channels[0], self.kernel_sizes[0]) 
        )
        
    def forward(self, x):
        
        # Apply linear decoder
        x = self.decoder_lin(x)        
        # Unflatte layer
        x = self.unflatten(x)         
        # Apply first deconvolutional layer
        x = self.first_deconv(x)          
        # Apply second deconvolutional layer
        x = self.second_deconv(x)        
        return x
        

class ConvAE(pl.LightningModule):
    
    def __init__(self, in_channels, out_channels, kernel_sizes, 
           padding=(0,0),  encoded_space_dim=1, act=nn.ReLU, drop_p=0.1, seq_len=100, feedforward_steps=1, loss_fn=nn.MSELoss, lr=0.001):
        
        super().__init__()
        self.encoder = ConvEncoder(in_channels, out_channels, kernel_sizes, padding, encoded_space_dim, 
                 drop_p, act, seq_len, feedforward_steps)
        self.decoder = ConvDecoder(in_channels, out_channels, kernel_sizes, padding, encoded_space_dim, drop_p, act, seq_len, feedforward_steps)
        
        self.seq_len = seq_len
        self.feedforward_steps = feedforward_steps
        self.loss_fn = loss_fn
        self.lr = lr
        
        
    def forward(self, x):
        # Encode data and keep track of indexes
        enc = self.encoder(x)
        # Decode data
        rec = self.decoder(enc)
        return (enc, rec)

    def training_step(self, batch, batch_idx):
        ### Prepare network input and labels 
        state  = batch[:,:, :self.seq_len-self.feedforward_steps, :]
        labels = batch[:,:, self.feedforward_steps:, :]
        
        # Forward step
        enc_state, rec_state = self.forward(state)
        # Compute reconstruction loss
        train_loss = self.loss_fn(rec_state, labels)
        # Logging to TensorBoard by default
        self.log("train_loss", train_loss, prog_bar=True)
        
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        ### Prepare network input and labels 
        state  = batch[:,:, :self.seq_len-self.feedforward_steps, :]
        labels = batch[:,:, self.feedforward_steps:, :]
        
        # Forward step
        enc_state, rec_state = self.forward(state)
        # Compute reconstruction loss
        val_loss = self.loss_fn(rec_state, labels)
        # Logging to TensorBoard by default
        self.log("val_loss", val_loss, prog_bar=True)
        
        return val_loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr)
        return optimizer

### Convolutional Variational Autoencoder
class VConvEncoder(pl.LightningModule):

    def __init__(self, in_channels, out_channels, kernel_sizes, padding=(0,0), encoded_space_dim=2, 
                 drop_p=0.1, act=nn.ReLU, seq_len=100):
    
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
        
        ### Network architecture
        # First convolutional layer (2d convolutional layer
        self.first_conv = nn.Sequential(
            nn.Conv2d(self.in_channels[0], self.out_channels[0], self.kernel_sizes[0], padding=self.padding[0]), 
            nn.BatchNorm2d(self.out_channels[0]),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False)
        )
        
        # Second convolution layer
        self.second_conv = nn.Sequential(
            nn.Conv2d(self.in_channels[1], self.out_channels[1], self.kernel_sizes[1], padding=self.padding[1]), 
            nn.BatchNorm2d(self.out_channels[1]),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False)
        )
        
        # Flatten layer
        self.flatten = nn.Flatten(start_dim=1)

        ### Linear section

        # Encode mean
        self.encoder_mean = nn.Sequential(
            # First linear layer
            nn.Linear(self.out_channels[1]*(self.seq_len+2-self.kernel_sizes[0][0]-self.kernel_sizes[1][0]), 256),
            nn.BatchNorm1d(256),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            # Second linear layer
            nn.Linear(256, self.encoded_space_dim)
        )
        
        # Encode logvar
        self.encoder_logvar = nn.Sequential(
            # First linear layer
            nn.Linear(self.out_channels[1]*(self.seq_len+2-self.kernel_sizes[0][0]-self.kernel_sizes[1][0]), 256),
            nn.BatchNorm1d(256),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            # Second linear layer
            nn.Linear(256, self.encoded_space_dim)
        )
        
    def forward(self, x):
        # Apply first convolutional layer
        x = self.first_conv(x)
        # Apply second convolutional layer
        x = self.second_conv(x)
        # Flatten 
        x = self.flatten(x)
        ## Apply linear layers
        # Encode mean
        mean = self.encoder_mean(x)
        # Encode log_var
        log_var = self.encoder_logvar(x)
        
        return (mean, log_var)
	
class ConvVAE(pl.LightningModule):
    
    def __init__(self, in_channels, out_channels, kernel_sizes, 
           padding=(0,0),  encoded_space_dim=1, act=nn.ReLU, drop_p=0.3, seq_len=100,
           sampler=Sampler(), loss_fn=nn.MSELoss, lr=0.001, beta=0.0001):
    
        super().__init__()
        
        self.encoder = VConvEncoder(in_channels, out_channels, kernel_sizes, padding, encoded_space_dim, 
                 drop_p, act, seq_len)
        self.decoder = ConvDecoder(in_channels, out_channels, kernel_sizes, encoded_space_dim, drop_p, act, seq_len)
        
        self.sampler = sampler
        self.lr = lr
        self.loss_fn = loss_fn
        self.beta = beta
    
        
    def forward(self, x):
    
        ### Encode data       
        mean, log_var = self.encoder(x)
        # Sampling
        x = self.sampler(mean, log_var)
        ### Decode data       
        rec = self.decoder(x)
        
        return (rec, mean, log_var)
    
    def training_step(self, batch, batch_idx):
        # Forward step
        rec_batch, mean, log_var = self.forward(batch)
        # Compute reconstruction loss
        rec_loss = self.loss_fn(rec_batch, batch)
        # KL Loss
        kl_loss = nKLDivLoss()(mean, log_var)
        # Total train loss
        train_loss = rec_loss + self.beta*kl_loss
        # Logging to TensorBoard by default
        self.log("train_loss", train_loss, prog_bar=True)
        
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        # Forward step
        rec_batch, mean, log_var = self.forward(batch)
        # Compute reconstruction loss
        rec_loss = self.loss_fn(rec_batch, batch)
        # KL Loss
        kl_loss = nKLDivLoss()(mean, log_var)
        # Total train loss
        val_loss = rec_loss + self.beta*kl_loss
        # Logging to TensorBoard by default
        self.log("val_loss", val_loss, prog_bar=True)
        
        return val_loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr)
        return optimizer
    
        
### Variational feedforward Autoencoder   
    
class VarFFEnc(pl.LightningModule):
    """
    This class implement a general VARiational Feed Forward Encoder (VarFFEnc)
    """
    def __init__(self, params):
        # Initialize parent class
        super().__init__()
        
        # Retrieve parameters
        self.layers_sizes = params["layers_sizes"]
        self.num_layers = len(self.layers_sizes)
        self.act = params["act"]
        self.drop_p = params["drop_p"]
        self.encoded_space_dim = params["encoded_space_dim"]
        
        ### Network architecture
        # Feed forward part
        self.ff_net = FFNet(params)
        
   
        # Encode mean
        self.encoder_mean = nn.Sequential(
            # First linear layer
            nn.Linear(self.layers_sizes[-1], 16),
            nn.Dropout(self.drop_p, inplace = False),
            self.act,
            # Second linear layer
            nn.Linear(16, self.encoded_space_dim)
        )
        
        
        # Encode log_var
        self.encoder_logvar = nn.Sequential(
            # First linear layer
            nn.Linear(self.layers_sizes[-1], 16),
            nn.Dropout(self.drop_p, inplace = False),
            self.act,
            # Second linear layer
            nn.Linear(16, self.encoded_space_dim)
        )
        
        print("Encoder initialized")
        
    def forward(self, x):
        # Feedforward part
        x = self.ff_net(x)
        
        # Dropout and activation
        x = nn.Dropout(self.drop_p, inplace = False)(x)
        x = self.act(x)
            
        # Encode mean
        mean = self.encoder_mean(x)
        
        # Encode logvar
        logvar = self.encoder_logvar(x)
 
        return mean, logvar


class VarFFAE(pl.LightningModule):
    """
    Implementation a general feed forward auto encoder
    """
    def __init__(self, params):
        # Initialize parent class
        super().__init__()
        
        # Retrieve parameters
        self.layers_sizes = params["layers_sizes"]
        self.num_layers = len(self.layers_sizes)
        self.act = params["act"]
        self.drop_p = params["drop_p"]
        self.encoded_space_dim = params["encoded_space_dim"]
        self.loss_fn = params["loss_fn"]
        self.lr = params["lr"]
        
        ### Network architecture
        self.encoder = VarFFEnc(params)
        
        # Reverse layers sizes for decoder part
        params["layers_sizes"].reverse()
        self.decoder = nn.Sequential(
            nn.Linear(self.encoded_space_dim, 16),
            nn.Dropout(self.drop_p, inplace = False),
            self.act,
            nn.Linear(16, self.layers_sizes[0]),
            nn.Dropout(self.drop_p, inplace = False),
            self.act,
            FFNet(params)
        )
        
        print("Autoencoder Initialized")
        
    def forward(self, x):
        # Encode data
        mean, log_var = self.encoder(x)
        
        # Sample data
        x = Sampler()(mean, log_var)
        
        # Decode data
        x = self.decoder(x)
        
        return x, mean, log_var
        
    def training_step(self, batch, batch_idx):
        # Forward step
        rec_batch, mean, log_var = self.forward(batch)
        # Compute reconstruction loss
        rec_loss = self.loss_fn(rec_batch, batch)
        # KL Loss
        kl_loss = nKLDivLoss()(mean, log_var)
        # Total train loss
        train_loss = rec_loss + kl_loss
        # Logging to TensorBoard by default
        self.log("train_loss", train_loss, prog_bar=True)
        
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        # Forward step
        rec_batch, mean, log_var = self.forward(batch)
        # Compute reconstruction loss
        rec_loss = self.loss_fn(rec_batch, batch)
        # KL Loss
        kl_loss = nKLDivLoss()(mean, log_var)
        # Total train loss
        val_loss = rec_loss + kl_loss
        # Logging to TensorBoard by default
        self.log("val_loss", val_loss, prog_bar=True)
        
        return val_loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr)
        return optimizer
    
     
      
      
      
      
      
 
