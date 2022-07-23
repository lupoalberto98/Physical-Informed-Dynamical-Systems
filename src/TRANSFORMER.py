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

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)
        
        # Info
        self.dropout = nn.Dropout(dropout_p)
        
        
        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1) # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 1).float() * (-math.log(10000.0)) / dim_model) # 1000^(2i/dim_model)
        
        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)[:,0::2]
        
        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)[:,1::2]
        
        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding",pos_encoding)
        
    def forward(self, token_embedding):
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:,:token_embedding.shape[1], :])


class Transformer(pl.LightningModule):
    " Tranformer for sequence generation"
    def __init__(self, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, system, true_system, dropout=0.1,
                activation="ReLU", lr=0.001, l1=0.0, dt=0.002, method_name="RK4",
                 use_pi_loss=True, apply_src_mask=False, apply_tgt_mask=True):
        # Initialize parent class
        super().__init__()
        # Retrieve parameters
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.system = system
        self.true_system = true_system
        self.dropout = dropout
        self.activation = getattr(nn, activation)()
        self.lr = lr
        self.l1 = l1 # L1 regularization weight
        self.dt = dt
        self.src_mask = None
        self.tgt_mask = None
        self.apply_src_mask = apply_src_mask
        self.apply_tgt_mask = apply_tgt_mask
        
        # Loss
        self.method_name = method_name
        self.use_pi_loss = use_pi_loss
        if self.use_pi_loss:
            # Physical informed loss
            self.method = getattr(utils, self.method_name)(self.dt, model=system.forward)
        else:
            # Dat driven loss
            self.method = getattr(utils, self.method_name)(self.dt, model=self.forward)
            
        # Positional encoder
        self.positional_encoder = PositionalEncoding(
            dim_model=self.d_model, dropout_p=self.dropout, max_len=5000
        )
        # Transoformer
        self.transformer = nn.Transformer(d_model=self.d_model, nhead=self.nhead, num_encoder_layers=self.num_encoder_layers, 
                                          num_decoder_layers=self.num_decoder_layers, dim_feedforward=self.dim_feedforward, 
                                          dropout=self.dropout, activation=self.activation, batch_first=True)
        # Define output layer
        self.output = nn.Linear(self.d_model, self.d_model)
        print("Transformer initialized")
        
    def get_tgt_mask(self, size):
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        return mask
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Positional encoding
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)
        # Transformer step
        out = self.transformer(src, tgt, src_mask, tgt_mask)
        # Output layer
        out = self.output(out)
        return out
    
    
    def training_step(self, batch, batch_idx):
        ### Prepare network input and labels and first net_out for curriculum learning
        state  = batch[:, :-1, :]
        labels = batch[:, 1:, :]
        # Apply masks
        if self.apply_tgt_mask:
            self.tgt_mask = self.get_tgt_mask(size=state.shape[1]).to(self.device)
        if self.apply_src_mask:
            self.src_mask = self.get_tgt_mask(size=state.shape[1]).to(self.device)
            
        # Forward pass
        next_state = self.forward(state, labels, self.src_mask, self.tgt_mask)
        # Compute loss
        train_loss = nn.MSELoss()(state, next_state)
        # Logging to TensorBoard by default
        self.log("train_loss", train_loss, prog_bar=True)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        ### Prepare network input and labels and first net_out for curriculum learning
        state  = batch[:, :-1, :]
        labels = batch[:, 1:, :]
        # Apply masks
        if self.apply_tgt_mask:
            self.tgt_mask = self.get_tgt_mask(size=state.shape[1]).to(self.device)
        if self.apply_src_mask:
            self.src_mask = self.get_tgt_mask(size=state.shape[1]).to(self.device)
        # Forward pass
        next_state = self.forward(state, labels, self.src_mask, self.tgt_mask)
        # Compute loss
        val_loss = nn.MSELoss()(state, next_state)
        # Logging to TensorBoard by default
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr)
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