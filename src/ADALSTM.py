#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch.optim as optim
from utils import Sampler, nKLDivLoss
import numpy as np
import math
from models import FFNet, LSTM
import utils
import warnings
from models import LSTM
from dataset import DynSysDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping 
from callbacks import MetricsCallback

class multi_rate_sampler():
    """
    Divide an input sequence in subsequence of given length and given time step
    return a list of dataloaders
    """
    
    def __init__(self, filename, tau, length, dt, batch_size=20, shuffle=True):
        """
        tau is a list of int containing the multiples of time steps
        length is a int list containing the sequence length
        """
        self.filename = filename
        self.tau = tau
        self.length = length
        self.dt = dt
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if len(self.tau)!=len(self.length):
            raise ValueError("Length of 'tau' list must be equal to length of 'length' list")
        
    def __call__(self):
        """
        batch is the input raw sequence, torch.tensor of size (batch_size, raw_seq_length, num_features)
        output is a list of tensor, containing at position i the sequence of size (batch_size, self.length[i]/tau[i], num_features)
        """
        out_dataloaders = []
        for i in range(len(self.tau)):
            states = DynSysDataset(self.filename, seq_len=self.length[i], tau=self.tau[i], dt=self.dt)
            dataloader = DataLoader(states, batch_size=self.batch_size, num_workers=0, shuffle=self.shuffle)
            out_dataloaders.append(dataloader)
            
        return out_dataloaders
    
"""       
def train_epoch(lstm, out, device, dataloader, optimizer):
    
    # Set train mode
    lstm.train()
    out.train()
    # Iterate over dataloader
    for batch in dataloader:
        # Move to device and divide in state and labels
        batch = batch.to(device)
        state = batch[:,:-1,:]
        labels = batch[:,1:,:]
        # Forward pass
        next_state = lstm(state)
        next_state = out(next_state)
        # Copu
"""
 
def main(stack_lstm, train_mrs, val_mrs, max_num_epochs=10):
    num_lstm = len(stack_lstm)
    processes = []
    for rank in range(num_lstm):
        # Define the callbacks
        metrics_callback =  MetricsCallback()
        early_stopping = EarlyStopping(monitor="val_loss", patience = 20, mode="min")
        trainer = pl.Trainer(max_epochs=max_num_epochs,callbacks=[metrics_callback, early_stopping], accelerator="auto", log_every_n_steps=1)
        model = stack_lstm[rank]
        model.set_output(False)
        p = mp.Process(target=trainer.fit, args=(model, train_mrs[rank], val_mrs[rank]))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
class AdaLSTM(pl.LightningModule):
    """
    Implementation of n parallel LSTM to analyse sequences with adaptive time step and sequence length
    """
    def __init__(self, num_lstm, mrs, input_size, hidden_units, layers_num=2, drop_p=0.1,
                 lr=0.001, return_rnn=False, bidirectional=False, train_out=True):
        
        super().__init__()
        
        # Retrieve parameters
        self.num_lstm = num_lstm # number of independent lstms
        self.mrs = mrs # multi rate sampler
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.layers_num = layers_num
        self.drop_p = drop_p
        self.lr = lr
        self.return_rnn = return_rnn
        self.bidirectional = bidirectional
        self.train_out = train_out
           
        
        # Define n independent lstms
        stack_lstm = []
        for i in range(self.num_lstm):
            stack_lstm.append(nn.LSTM(input_size=self.input_size, 
                           hidden_size=self.hidden_units,
                           num_layers=self.layers_num,
                           dropout=self.drop_p,
                           batch_first=True,
                          bidirectional=self.bidirectional))
            
        self.stack_lstm = nn.ModuleList(stack_lstm)
        
        # Define n independent output layers
        stack_out = []
        for i in range(self.num_lstm):
            stack_out.append(nn.Linear(self.hidden_units, self.input_size))
        
        self.stack_out = nn.ModuleList(stack_out)
        
        # Attention weights
        self.register_parameter(name="attention", param=nn.Parameter(0.01*torch.randn(self.num_lstm)))
        
        # Linear neuron
        self.linear = nn.Linear(self.hidden_units, self.input_size)
        
        print("Network initialized")
    
    def set_train_out(self, train_out):
        self.train_out = train_out
    
    def set_return_rnn(self, return_rnn):
        self.return_rnn = return_rnn
        
    def forward(self, x, state=[None]*10000):
        """
        x is a list of num_lstm tensors of shape (B, S_i, F),
        where B is the batch size, S_i is seq_length of lstm i and F the number of features
        out is just the next element of the sequence, tensor of shape (B, F)
        """
        
        if self.train_out:
            out = []
            for i in range(self.num_lstm):
                lstm_i = self.stack_lstm[i]
                if self.return_rnn:
                    out_i, state[i] = lstm_i(x[i], state[i])
                else:
                    out_i, _ = lstm_i(x[i], state[i])
                    
                out_i = self.stack_out[i](out_i)
                out.append(out_i[:,-1,:]) 
        else:
            out = 0
            for i in range(self.num_lstm):
                lstm_i = self.stack_lstm[i]
                if self.return_rnn:
                    out_i, state[i] = lstm_i(x[i], state[i])
                else:
                    out_i, _ = lstm_i(x[i], state[i])
                    
                out_i = out_i*nn.Sigmoid()(self.attention[i])
                out += out_i[:,-1,:]

            out = self.linear(out)
        
        if self.return_rnn:
            return out, state
        else:
            return out

    def training_step(self, batch, batch_idx):
        # Define state and labels
        state = self.mrs(batch[:,:-1,:])
        labels = batch[:,-1,:]
        
        # Forward
        next_state = self.forward(state)
        
        # Train with or without linear and attention layers
        if self.train_out:
            train_loss = 0
            for i in range(self.num_lstm):
                train_loss += nn.MSELoss()(labels, next_state[i])
            
        else:
            train_loss = nn.MSELoss()(labels, next_state)
            
        # Logging to TensorBoard by default
        self.log("train_loss", train_loss, prog_bar=True)
        
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        # Define state and labels
        state = self.mrs(batch[:,:-1,:])
        labels = batch[:,-1,:]
        
        # Forward
        next_state = self.forward(state)
        
        # Train with or without linear and attention layers
        if self.train_out:
            val_loss = 0
            for i in range(self.num_lstm):
                val_loss += nn.MSELoss()(labels, next_state[i])
            
        else:
            val_loss = nn.MSELoss()(labels, next_state)
            
        # Logging to TensorBoard by default
        self.log("val_loss", val_loss, prog_bar=True)
        
        return val_loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr)
        return optimizer
            
        
                    