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



class LSTM(pl.LightningModule):  
    def __init__(self, input_size, hidden_units, layers_num, system, true_system, drop_p=0.1,
                 lr=0.001, dt=0.01, method_name="RK4", use_pi_loss=False, return_rnn=False, perturbation=None, bidirectional=False, train_out=True, l1=0.0, weight_decay=0.0):
        # Call the parent init function 
        super().__init__()
        # Retrieve parameters
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.layers_num = layers_num
        self.drop_p = drop_p
        self.lr = lr
        self.dt = dt
        self.system = system
        self.true_system = true_system
        self.perturbation = perturbation
        self.train_out = train_out 
        self.l1 = l1 # L1 regularization weight
        self.weight_decay = weight_decay
        
        # Define propagation methods (either use physics informed or data driven loss)
        self.method_name = method_name
        self.use_pi_loss = use_pi_loss
        if self.use_pi_loss:
            # Physical informed loss
            self.method = getattr(utils, self.method_name)(self.dt, model=system)
        else:
            # Dat driven loss
            self.method = getattr(utils, self.method_name)(self.dt, model=self)
            
        
        # Add system parameters to computational graph
        #self.register_parameter(name="params", param=nn.Parameter(self.system.params))
                                                     
        # Set output mode
        self.return_rnn = return_rnn
        
        # Define recurrent layers
        self.rnn = nn.LSTM(input_size=self.input_size, 
                           hidden_size=self.hidden_units,
                           num_layers=self.layers_num,
                           dropout=self.drop_p,
                           batch_first=True,
                          bidirectional=bidirectional)
        # Define output layer
        self.out = nn.Linear(self.hidden_units, self.input_size)
        print("LSTM initialized")
        
        

        
    def forward(self,t, x, rnn_state=None):
        # LSTM
        x, rnn_state = self.rnn(x, rnn_state)
        # Linear layer
        if self.train_out:
            x = self.out(x)
        # Remember to return also the RNN state, you will need it to generate data
        if self.return_rnn:
            return x, rnn_state
        else:
            return x
    
    def training_step(self, batch, batch_idx):
        ### Prepare network input and labels and first net_out for curriculum learning
        state  = batch[:, :-1, :]
        labels = batch[:, 1:,:]
        
        if self.use_pi_loss:
            ## Physical informed loss
            # Forward
            next_state = self.forward(batch_idx, state) # Propagated through network
            df = self.method(state) # Differential computeed with true model
            train_loss = nn.MSELoss()(state+df, next_state) + self.l1*sum(p.abs().sum() for p in self.parameters())
        else: 
            ## Data driven loss + possible perturbation
            # Forward
            self.method = getattr(utils, self.method_name)(self.dt, model=self.forward)
            df = self.method(state) # Differential computed propagating network
            if self.perturbation is None:
                train_loss = nn.MSELoss()(state+df, labels) + self.l1*sum(p.abs().sum() for p in self.parameters())
            else:
                train_loss = nn.MSELoss()(state+df+self.perturbation*self.dt, labels) + self.l1*sum(p.abs().sum() for p in self.parameters())
      
        # Compute loss between true and learned parameters
        params_loss = np.mean((self.system.params.detach().cpu().numpy()-self.true_system.params.detach().cpu().numpy())**2)
        self.log("params_loss", params_loss, prog_bar=True)
        
        # Logging to TensorBoard by default
        self.log("train_loss", train_loss, prog_bar=True)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        ### Prepare network input and labels and first net_out for curriculum learning
        state  = batch[:, :-1, :]
        labels = batch[:,1:,:]
        
        if self.use_pi_loss:
            ## Physical informed loss
            # Forward
            next_state = self.forward(batch_idx, state) # Propagated through network
            df = self.method(state) # Differential computeed with true model
            val_loss = nn.MSELoss()(state+df, next_state) + self.l1*sum(p.abs().sum() for p in self.parameters())
        else:
            ## Data driven loss + possible perturbation
            self.method = getattr(utils, self.method_name)(self.dt, model=self.forward)
            df = self.method(state) # Differential computed propagating network
            if self.perturbation is None:
                val_loss = nn.MSELoss()(state+df, labels) + self.l1*sum(p.abs().sum() for p in self.parameters())
            else:
                val_loss = nn.MSELoss()(state+df+self.perturbation*self.dt, labels) + self.l1*sum(p.abs().sum() for p in self.parameters())
                
        # Logging to TensorBoard by default
        self.log("val_loss", val_loss, logger=True, on_epoch=True, prog_bar=True)
        self.log("epoch_num", self.current_epoch,prog_bar=True)
        return val_loss
                        
                        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay=self.weight_decay)
        return optimizer
        
    def set_output(self, return_rnn):
        self.return_rnn = return_rnn
    
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
        state = inputs[0].unsqueeze(0).unsqueeze(0)
        rnn_state = (torch.zeros(self.layers_num, 1,self.hidden_units), torch.zeros(self.layers_num, 1,self.hidden_units))
        net_states = []
        self.eval()
        self.set_output(True)
        
        for i in range(prediction_steps):
            with torch.no_grad():
                net_states.append(state[-1].squeeze().numpy())
                if input_is_looped:
                    if self.use_pi_loss:
                        state, rnn_state = self(i, state, rnn_state)
                    else:
                        f, rnn_state = self(i, state, rnn_state)
                        state = state + self.dt*f
                else:
                    if self.use_pi_loss:
                        state, _ = self(i, inputs[i].unsqueeze(0).unsqueeze(0))
                    else:
                        f, _ = self(i, inputs[i].unsqueeze(0).unsqueeze(0))
                        state = state + self.dt*f
                        
        return torch.tensor(np.array(net_states))


        
