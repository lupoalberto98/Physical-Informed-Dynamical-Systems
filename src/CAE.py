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
import models

### Convolutional Autoencoder
class ConvEncoder(pl.LightningModule):
    
    def __init__(self,
                 in_channels, 
                 out_channels,
                 kernel_sizes,
                 padding = (0,0),
                 encoded_space_dim = 2, 
                 drop_p = 0.1,
                 act = nn.ReLU,
                 seq_len = 100,
                 feedforward_steps = 1):
        """
        Convolutional Network with three convolutional layer and two dense
        Args:
            in_channels : inputs channels
            out_channels : output channels
            kernel_sizes : kernel sizes
            padding : padding added to edges
            encoded_space_dim : dimension of encoded space
            drop_p : dropout probability
            act : activation function
            seq_len : length of input sequences 
            feedforward_steps : prediction steps
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
    
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 kernel_sizes, 
                 padding = (0,0), 
                 encoded_space_dim = 2, 
                 drop_p = 0.1, 
                 act = nn.ReLU, 
                 seq_len = 100, 
                 feedforward_steps = 1):
        """
        Convolutional Decoder with three convolutional layer and two dense
        Args:
            in_channels : inputs channels
            out_channels : output channels
            kernel_sizes : kernel sizes
            padding : padding added to edges
            encoded_space_dim : dimension of encoded space
            drop_p : dropout probability
            act : activation function
            seq_len : length of input sequences 
            feedforward_steps : prediction steps
        """
        
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
    
    def __init__(self,
                 in_channels, 
                 out_channels, 
                 kernel_sizes, 
                 true_system,
                 padding=(0,0), 
                 encoded_space_dim=10, 
                 act=nn.ReLU,
                 drop_p=0.1,
                 seq_len=100, 
                 feedforward_steps=1,
                 lr=0.001, 
                 dt=0.01,
                 enc_space_reg=None, 
                 beta=1.0, 
                 lr_scheduler_name="ExponentialLR", 
                 gamma=1.0, 
                 reconstruct=True):
        """
        Convolutional Symmetric Autoencoder
        Args:
            in_channels : inputs channels
            out_channels : output channels
            kernel_sizes : kernel sizes
            true_system : ground true system hypothesis
            padding : padding added to edges
            encoded_space_dim : dimension of encoded space
            drop_p : dropout probability
            act : activation function
            seq_len : length of input sequences 
            feedforward_steps : prediction steps
            lr : learning rate
            dt : time discretization step
            enc_space_reg : regularization used for encoded space (PI, TRUE)
            beta : weight for regulariaztion loss
            lr_scheduler_name : name of lr scheduler
            gamma : decay of lr scheduler
            reconstruct : of True reconstruct input, else predict next step
        """
        
        super().__init__()
        self.encoder = ConvEncoder(in_channels, out_channels, kernel_sizes, padding, encoded_space_dim, 
                 drop_p, act, seq_len, feedforward_steps)
        self.decoder = ConvDecoder(in_channels, out_channels, kernel_sizes, padding, encoded_space_dim, drop_p, act, seq_len, feedforward_steps)
        
        self.seq_len = seq_len
        self.feedforward_steps = feedforward_steps # how many predictive steps
        self.lr = lr
        self.dt = dt
        self.encoded_space_dim = encoded_space_dim
        self.true_system = true_system
        self.system_name = true_system.__class__.__name__ #string specifing the system being used
        self.system_dim = true_system.dim
        self.num_param = len(true_system.params)
        self.enc_space_reg = enc_space_reg # string specifing how to compute regularization loss
        self.beta = beta # Weights for regularization losses
        self.lr_scheduler_name = lr_scheduler_name
        self.gamma = gamma
        self.reconstruct = reconstruct # if True reconstruct input, else predict next step
        
        ### Checkings 
        if self.enc_space_reg == None:
            warnings.warn("No encoded space regualarization used")
        else:
            if self.encoded_space_dim < self.num_param:
                raise ValueError("Encoded space dimension too small to be regularized with PI loss")
        
        print("Convolutional Autoencoder initialized")
       
    def forward(self, t, x):
        # Encode data and keep track of indexes
        enc, indeces_1, indeces_2, indeces_3 = self.encoder(x)
        # Decode data
        rec = self.decoder(enc, indeces_1, indeces_2, indeces_3)
        return enc, rec

    def training_step(self, batch, batch_idx):
        # Unsqueeze batch
        batch = batch.unsqueeze(1)
        ### Prepare network input and labels 
        state  = batch[:,:, :self.seq_len-self.feedforward_steps, :]
        labels = batch[:,:, self.feedforward_steps:, :] #if feedforard step=0 is equal to state
        
        # Set reg loss to zero
        reg_loss = 0
        # Forward step
        enc_state, rec_state = self.forward(batch_idx, state)
        # Compute reconstruction loss
        if self.reconstruct:
            rec_loss = nn.MSELoss()(rec_state, state)
        else:
            rec_loss = nn.MSELoss()(rec_state, labels)
        
        # Compute regularization loss
        if self.enc_space_reg == "PI": # self.feedfrward_steps should be 1 here
            # Initialize the system with parameters as the first entries of encoded batch
            system = getattr(models, self.system_name)(params=enc_state[:,:self.num_param].unsqueeze(1).unsqueeze(1))
            # Initialize the physical informed method to compute loss
            method = utils.RK4(self.dt, model=system)
            # Compute differential and error
            df = method(state)
            # Simple regularization loss
            reg_loss += nn.MSELoss()(state+df, labels)*self.beta
            # Noise regularized loss
            self.log("train_reg_loss", reg_loss, prog_bar=True)
        if self.enc_space_reg == "TRUE":
            reg_loss += nn.MSELoss()(enc_state[:,:self.num_param],self.true_system.params)*self.beta
            self.log("train_reg_loss", reg_loss, prog_bar=True)
        
        # Logging to TensorBoard by default
        train_loss = rec_loss + reg_loss
        self.log("train_loss", train_loss, prog_bar=True)
        
        # Lr scheduler
        self.lr_scheduler.step()
    
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        # Unsqueeze batch
        batch = batch.unsqueeze(1)
        ### Prepare network input and labels 
        state  = batch[:,:, :self.seq_len-self.feedforward_steps, :]
        labels = batch[:,:, self.feedforward_steps:, :]
        
        # Set reg loss to zero
        reg_loss = 0
        # Forward step
        enc_state, rec_state = self.forward(batch_idx, state)
        # Compute reconstruction loss
        if self.reconstruct:
            rec_loss = nn.MSELoss()(rec_state, state)
        else:
            rec_loss = nn.MSELoss()(rec_state, labels)
        
        
        # Compute regularization loss
        if self.enc_space_reg == "PI": # self.feedfrward_steps should be 1 here
            # Initialize the system with parameters as the first entries of encoded batch
            system = getattr(models, self.system_name)(params=enc_state[:,:self.num_param].unsqueeze(1).unsqueeze(1))
            # Initialize the physical informed method to compute loss
            method = utils.RK4(self.dt, model=system)
            # Compute differential and error
            df = method(state)
            # Simple regularization loss
            reg_loss += nn.MSELoss()(state+df, labels)*self.beta
            # Noise regularized loss
            self.log("val_reg_loss", reg_loss, prog_bar=True)
        if self.enc_space_reg == "TRUE":
            reg_loss += nn.MSELoss()(enc_state[:,:self.num_param],self.true_system.params)*self.beta
            self.log("val_reg_loss", reg_loss, prog_bar=True)
            
        # Compute reconstruction loss
        val_loss = rec_loss + reg_loss
        # Logging to TensorBoard by default
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("epoch_num", self.current_epoch,prog_bar=True)
        
        return val_loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr)
        self.lr_scheduler = getattr(optim.lr_scheduler, self.lr_scheduler_name)(optimizer, self.gamma)
        return optimizer
    
    def num_timesteps(self, time):
        """Returns the number of timesteps required to pass time time.
        Args:
            time : total time elapsed, to be divided by dt to get num_timesteps
        """
        num_timesteps = time / self.dt
        if not num_timesteps.is_integer():
            raise Exception
        return int(num_timesteps)
    

    def predict(self, time, inputs, input_is_looped=True):
        """
        Generate a trajectory of prediction_steps lenght starting from inputs
        Args:
            time : total time elapsed, to be divided by dt to get num_timesteps
            inputs : dataset to be compared, torch.tensor of size (num_points, dim)
            input_is_looped: if True, the output is fed into the network as an input at the next step
        """
        prediction_steps = self.num_timesteps(time)
        net_states = [inputs[0].detach().cpu().numpy().tolist()] # first element is first of inputs
        state = inputs[:self.seq_len-1,:].unsqueeze(0).unsqueeze(0) # define first state
        self.eval()
        
        # run an interation and save first elements
        with torch.no_grad(): 
            _, state = self(0, state)
            for i in range(self.seq_len-1):
                net_states.append(state[0,0,i].detach().cpu().numpy().tolist())
                
        
        # run all the other iterations
        for i in range(prediction_steps-self.seq_len):
            with torch.no_grad(): 
                if input_is_looped:
                    _, state = self(0, state)
                    net_states.append(state[0,0,-1,:].detach().cpu().numpy().tolist())
                else:
                    _, state = self(0, inputs[i:i+self.seq_len-1,:].unsqueeze(0).unsqueeze(0))
                    net_states.append(state[0,0,-1,:].detach().cpu().numpy().tolist())
                
        return torch.tensor(np.array(net_states))