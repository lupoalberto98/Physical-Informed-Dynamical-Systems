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
from LSTM import LSTM
from FFNet import FFNet
import utils
import warnings
from dataset import DynSysDataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping 
from callbacks import MetricsCallback

class multi_rate_sampler():
    """
    Divide an input sequence in subsequence of given length and given time step
    return a list of dataloaders
    """
    
    def __init__(self, tau, length, dt, batch_size=20):
        """
        tau is a list of int containing the multiples of time steps
        length is a int list containing the sequence length
        """
        self.tau = tau
        self.length = length
        self.dt = dt
        self.batch_size = batch_size
        
        # Check lengths
        if len(self.tau)!=len(self.length):
            raise ValueError("Length of 'tau' list must be equal to length of 'length' list")
            
        self.num_lstm = len(self.tau) # =len(self.length)
        
    def divide_dataloader(self, filename, num_workers=0, shuffle=False):
        """
        batch is the input raw sequence, torch.tensor of size (batch_size, raw_seq_length, num_features)
        output is a list of tensor, containing at position i the sequence of size (batch_size, self.length[i]/tau[i], num_features)
        usefule for pretraining
        """
        out_dataloaders = []
        for i in range(self.num_lstm):
            states = DynSysDataset(filename, seq_len=self.length[i], tau=self.tau[i], dt=self.dt)
            dataloader = DataLoader(states, batch_size=self.batch_size, num_workers=num_workers, shuffle=shuffle)
            out_dataloaders.append(dataloader)
        
        return out_dataloaders
    
    def divide_batch(self, batch):
        """
        Divide a batch in a list of batches of size (batch_size, length[i], F), 
        each of which sample at differetn tau*dt timesteps
        useful for learning the entire model
        """
        
        batch_list = []
        for i in range(self.num_lstm):
            batch_list.append(batch[:,0::self.tau[i],:])
                              
        return batch_list
        
    
def pretrain(stack_lstm, train_mrs, val_mrs, patience=100, max_num_epochs=10):
    """
    Train independently num_lstm lstm with output layer
    stack_lstm is a disctionary containing all the lstms
    """
    num_lstm = len(stack_lstm)
    train_loss_logs = []
    val_loss_logs = []
    for i in range(num_lstm):
        print("### Train LSTM "+str(i+1)+" ###")
        # Call the model
        model = stack_lstm["LSTM"+str(i+1)]
        # Define the callbacks
        metrics_callback =  MetricsCallback()
        early_stopping = EarlyStopping(monitor="val_loss", patience = patience, mode="min")
        # Train
        trainer = pl.Trainer(max_epochs=max_num_epochs, callbacks=[metrics_callback, early_stopping], 
                             accelerator="auto", log_every_n_steps=1, enable_model_summary=False)
        trainer.fit(model=model, train_dataloaders=train_mrs[i], val_dataloaders=val_mrs[i])
        # Append losses
        train_loss_logs.append(metrics_callback.train_loss_log)
        val_loss_logs.append(metrics_callback.val_loss_log)
        
    return train_loss_logs, val_loss_logs
                             

class AdaLSTM(pl.LightningModule):
    """
    Implementation of n parallel LSTM to analyse sequences with adaptive time step and sequence length
    """
    def __init__(self, stack_lstm, mrs, input_size, hidden_units, lr=0.001, return_rnn=False):
        
        super().__init__()
        
        # Retrieve parameters
        self.num_lstm = len(stack_lstm)
        self.mrs = mrs
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.lr = lr
        self.return_rnn = return_rnn
        self.layers_num_list = [stack_lstm["LSTM"+str(i+1)].layers_num for i in range(self.num_lstm)]
        
        
        # Set train_out false and return_rrn false to train
        for i in range(self.num_lstm):
            stack_lstm["LSTM"+str(i+1)].train_out = False
            stack_lstm["LSTM"+str(i+1)].return_rnn = self.return_rnn

        # Define a single module
        self.stack_lstm = nn.ModuleDict(stack_lstm)
                
        # Attention weights
        self.register_parameter(name="attention", param=nn.Parameter(0.01*torch.randn(self.num_lstm)))
        
        # Linear neuron
        self.linear = nn.Linear(self.hidden_units, self.input_size)
        
        print("AdaLSTM initialized")
   
    def set_return_rnn(self, value):
        self.return_rnn = value
        for i in range(self.num_lstm):
            self.stack_lstm["LSTM"+str(i+1)].return_rnn = value

        
    def forward(self,t, x, rnn_state=[None]*10000):
        """
        x is a list of num_lstm tensors of shape (B, S_i, F),
        where B is the batch size, S_i is seq_length of lstm i and F the number of features
        out is just the next element of the sequence, tensor of shape (B, F)
        """
        out = 0
        for i in range(self.num_lstm):
            lstm_i = self.stack_lstm["LSTM"+str(i+1)]
            if self.return_rnn:
                out_i, rnn_state[i] = lstm_i(t, x[i], rnn_state[i])
            else:
                out_i = lstm_i(t, x[i], rnn_state[i])
            
            out_i = out_i*self.attention[i]
            out += out_i[:,-1,:]

        # Apply linear neuron
        out = self.linear(out)
        
        if self.return_rnn:
            return out, rnn_state
        else:
            return out

    def training_step(self, batch, batch_idx):
        # Define state and labels
        state = self.mrs.divide_batch(batch[:,:-1,:])
        labels = batch[:,-1,:]
        
        # Forward
        next_state = self.forward(batch_idx, state)
        # Compute loss
        train_loss = nn.MSELoss()(labels, next_state)
            
        # Logging to TensorBoard by default
        self.log("train_loss", train_loss, prog_bar=True)
        
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        # Define state and labels
        state = self.mrs.divide_batch(batch[:,:-1,:])
        labels = batch[:,-1,:]
        
        # Forward
        next_state = self.forward(batch_idx, state)
        # Compute loss
        val_loss = nn.MSELoss()(labels, next_state)
            
        # Logging to TensorBoard by default
        self.log("val_loss", val_loss, prog_bar=True)
        
        return val_loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr)
        return optimizer
            
        
                    