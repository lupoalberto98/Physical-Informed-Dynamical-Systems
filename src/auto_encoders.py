#!/usr/bin/env python3
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from utils import Sampler, nKLDivLoss
import numpy as np
import math
from models import FFNet, LSTM
import utils

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
        
        # Linear encoder
        self.encoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(self.out_channels[2]*self.lin_dim, 256),
            nn.BatchNorm1d(256),
            self.act(inplace = True),
            nn.Dropout(self.drop_p, inplace = False),
            # Second linear layer
            nn.Linear(256, self.encoded_space_dim)
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
        x = self.encoder_lin(x)
        return x, indeces_1, indeces_2, indeces_3
    


    
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
        # Linear dimension
        self.lin_dim = int((((self.seq_len-self.feedforward_steps-self.kernel_sizes[0][0]+1)/2+1-self.kernel_sizes[1][0])/2+1-self.kernel_sizes[2][0])/2)
        
        # Linear decoder
        self.decoder_lin = nn.Sequential(
            # First linear layer
            nn.Linear(self.encoded_space_dim, 256),
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
        
    def forward(self, x, indeces_1, indeces_2, indeces_3):
        
        # Apply linear decoder
        x = self.decoder_lin(x)        
        # Unflatte layer
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
        
### Symmetric convolutional autoencoder
class ConvAE(pl.LightningModule):
    
    def __init__(self, in_channels, out_channels, kernel_sizes, 
           padding=(0,0),  encoded_space_dim=1, act=nn.ReLU, drop_p=0.1, seq_len=100, feedforward_steps=1, lr=0.001, dt=0.01, system_name="Lorenz63", system_dim=3, num_param=3, enc_space_reg="RK4", beta1=1.0, beta2=1.0):
        
        super().__init__()
        self.encoder = ConvEncoder(in_channels, out_channels, kernel_sizes, padding, encoded_space_dim, 
                 drop_p, act, seq_len, feedforward_steps)
        self.decoder = ConvDecoder(in_channels, out_channels, kernel_sizes, padding, encoded_space_dim, drop_p, act, seq_len, feedforward_steps)
        
        self.seq_len = seq_len
        self.feedforward_steps = feedforward_steps
        self.lr = lr
        self.dt = dt
        self.system_name = system_name #string specifing the system being used
        self.system_dim = system_dim
        self.num_param = num_param
        self.enc_space_reg = enc_space_reg # string specifing ho to compute loss
        # Weights for regularization losses
        self.beta1 = beta1
        self.beta2 = beta2
        
        
    def forward(self, x):
        # Encode data and keep track of indexes
        enc, indeces_1, indeces_2, indeces_3 = self.encoder(x)
        # Decode data
        rec = self.decoder(enc, indeces_1, indeces_2, indeces_3)
        return (enc, rec)

    def training_step(self, batch, batch_idx):
        ### Prepare network input and labels 
        state  = batch[:,:, :self.seq_len-self.feedforward_steps, :]
        labels = batch[:,:, self.feedforward_steps:, :]
        
        # Set reg loss to zero
        reg_loss = 0
        # Forward step
        enc_state, rec_state = self.forward(state)
        # Compute reconstruction loss
        rec_loss = nn.MSELoss()(rec_state, state)
        # Compute regularization loss
        if self.enc_space_reg is not None: # self.feedfrward_steps should be 1 here
            # Initialize the system with parameters as the first entries of encoded batch
            system = getattr(utils, self.system_name)(params=enc_state[:,:self.num_param].unsqueeze(1).unsqueeze(1))
            # Initialize the physical informed method to compute loss
            method = getattr(utils, self.enc_space_reg)(self.dt, model=system)
            # Compute differential and error
            df = method(state)
            reg_loss += nn.MSELoss()(state+df, labels)*self.beta1
            self.log("train_reg_loss", reg_loss, prog_bar=True)
        
        # Logging to TensorBoard by default
        train_loss = rec_loss + reg_loss
        self.log("train_loss", train_loss, prog_bar=True)
    
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        ### Prepare network input and labels 
        state  = batch[:,:, :self.seq_len-self.feedforward_steps, :]
        labels = batch[:,:, self.feedforward_steps:, :]
        # Set reg loss to zero
        reg_loss = 0
        # Forward step
        enc_state, rec_state = self.forward(state)
        # Compute reconstruction loss
        rec_loss = nn.MSELoss()(rec_state, state)
        # Compute regularization loss
        if self.enc_space_reg is not None: # self.feedfrward_steps should be 1 here
            system = getattr(utils, self.system_name)(params=enc_state[:,:self.num_param].unsqueeze(1).unsqueeze(1))
            method = getattr(utils, self.enc_space_reg)(self.dt, model=system)
            df = method(state)
            reg_loss += nn.MSELoss()(state+df, labels)*self.beta1
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
    

### Convolutional Encoder + LSTM Decoder

class ConvLSTMAE(pl.LightningModule):
    """
    Autoencoder with a convolutional encoder and a LSTM decoder
    """
    def __init__(self,in_channels, out_channels, kernel_sizes, 
           padding=(0,0),  encoded_space_dim=1, lstm_hidden_units=100, bidirectional=False,layers_num=2, act=nn.ReLU, drop_p=0.1, seq_len=100, feedforward_steps=1, lr=0.001, dt=0.01, system_name="Lorenz63", system_dim=3, num_param=3, enc_space_reg="RK4", beta1=1.0, beta2=1.0):
        
        super().__init__()
        
        # Parameters
        self.seq_len = seq_len
        self.feedforward_steps = feedforward_steps
        self.lr = lr
        self.dt = dt
        self.system_name = system_name #string specifing the system being used
        self.system_dim = system_dim
        self.num_param = num_param
        self.enc_space_reg = enc_space_reg # string specifing ho to compute loss
        # Weights for regularization losses
        self.beta1 = beta1
        self.beta2 = beta2
        
        # Encoder
        self.encoder = ConvEncoder(in_channels, out_channels, kernel_sizes, padding, encoded_space_dim, 
                 drop_p, act, seq_len, feedforward_steps)
            
        ### LSTM decoder
        self.lstm = nn.LSTM(input_size=encoded_space_dim, 
                           hidden_size=lstm_hidden_units,
                           num_layers=layers_num,
                           dropout=drop_p,
                           batch_first=True,
                          bidirectional=bidirectional)
        # Set D parameter for birectionality
        D=1
        if bidirectional:
            D=2
        self.out = nn.Linear(D*lstm_hidden_units, system_dim)
        
        
        print("Network initialized")

        
    def forward(self, x):
        # Encode data and keep track of indexes
        enc, indeces_1, indeces_2, indeces_3 = self.encoder(x)
        # Replicate enc along 1 dimension
        # Decode data
        hidd_rec, rnn = self.lstm(enc.unsqueeze(1).repeat(1, self.seq_len-self.feedforward_steps,1 ))
        # Fully connected output layer
        rec = self.out(hidd_rec)
        # Reinsert channel dimension
        rec = rec.unsqueeze(1)

        return (enc, rec)
        
    def training_step(self, batch, batch_idx):
        ### Prepare network input and labels 
        state  = batch[:,:, :self.seq_len-self.feedforward_steps, :]
        labels = batch[:,:, self.feedforward_steps:, :]
        
        # Set reg loss to zero
        reg_loss = 0
        # Forward step
        enc_state, rec_state = self.forward(state)
        # Compute reconstruction loss
        rec_loss = nn.MSELoss()(rec_state, state)
        # Compute regularization loss
        if self.enc_space_reg is not None: # self.feedfrward_steps should be 1 here
            # Initialize the system with parameters as the first entries of encoded batch
            system = getattr(utils, self.system_name)(params=enc_state[:,:self.num_param].unsqueeze(1).unsqueeze(1))
            # Initialize the physical informed method to compute loss
            method = getattr(utils, self.enc_space_reg)(self.dt, model=system)
            # Compute differential and error
            df = method(state)
            reg_loss += nn.MSELoss()(state+df, labels)*self.beta1
            self.log("train_reg_loss", reg_loss, prog_bar=True)
            
        # Logging to TensorBoard by default
        train_loss = rec_loss + reg_loss
        self.log("train_loss", train_loss, prog_bar=True)
    
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        ### Prepare network input and labels 
        state  = batch[:,:, :self.seq_len-self.feedforward_steps, :]
        labels = batch[:,:, self.feedforward_steps:, :]
        # Set reg loss to zero
        reg_loss = 0
        # Forward step
        enc_state, rec_state = self.forward(state)
        # Compute reconstruction loss
        rec_loss = nn.MSELoss()(rec_state, state)
        # Compute regularization loss
        if self.enc_space_reg is not None: # self.feedfrward_steps should be 1 here
            system = getattr(utils, self.system_name)(params=enc_state[:,:self.num_param].unsqueeze(1).unsqueeze(1))
            method = getattr(utils, self.enc_space_reg)(self.dt, model=system)
            df = method(state)
            reg_loss += nn.MSELoss()(state+df, labels)*self.beta1
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
    
     
      
      
      
      
      
 