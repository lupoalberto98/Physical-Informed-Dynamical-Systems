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


class FFNet(pl.LightningModule):

    def __init__(self, seq_len, n_inputs, n_outputs, hidden_layers, system, true_system, drop_p=0.3, lr=0.001, dt=0.002, 
                method_name="RK4", activation="ReLU", use_pi_loss=False, int_mode=True, l1=0, weight_decay=0):
        """
        Initialize a typical feedforward network with different hidden layers
        The input is typically a mnist image, given as a torch tensor of size = (1,784),
        or a sequence, torch.tensor of size (1, seq_length, 3)
        ----------
        Args:
        n_inputs = input features
        n_outputs = output features
        hidden_layers = list of sizes of the hidden layers
        drop_p = dropout probability
        lr = learning rate
        """
        super().__init__()
        # Parameters
        self.seq_len = seq_len
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_layers = hidden_layers
        self.num_hidden_layers = len(self.hidden_layers)
        self.drop_p = drop_p
        self.lr = lr
        self.dt = dt
        self.activation = getattr(nn, activation)()
        self.system = system
        self.true_system = true_system
        self.l1 = l1 # L1 regularization weight
        self.weight_decay = weight_decay
        
        # Define propagation methods (either use physics informed or data driven loss)
        self.method_name = method_name
        self.use_pi_loss = use_pi_loss
        self.int_mode = int_mode # if True, net is integrator, else it is the derivative function
        if self.use_pi_loss:
            # Physical informed loss
            if self.int_mode is False:
                warnings.warn("Set integrator mode True")
                self.int_mode = True
                
            self.method = getattr(utils, self.method_name)(self.dt, model=system)
        else:
            # Data driven loss
            if self.int_mode is False:
                self.method = getattr(utils, self.method_name)(self.dt, model=self)
        
        ### Network architecture
        layers = []
        
        # input layer
        layers.append(nn.Linear((self.seq_len-1)*self.n_inputs, self.hidden_layers[0]))
        layers.append(nn.Dropout(self.drop_p, inplace = False))
        layers.append(self.activation)
        
        # hidden layers
        for l in range(self.num_hidden_layers-1):
            layers.append(nn.Linear(self.hidden_layers[l], self.hidden_layers[l+1]))
            layers.append(nn.Dropout(self.drop_p, inplace = False))
            layers.append(self.activation)
        
        # output layer
        layers.append(nn.Linear(self.hidden_layers[-1], (self.seq_len-1)*self.n_outputs))
        
        self.layers = nn.ModuleList(layers)
                          
        print("Feedforward Network initialized")
                  

    def forward(self, t, x):
        """
        Input tensor of size (batch_size, features)
        """
        for l in range(len(self.layers)):
            x = self.layers[l](x)

        return x

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
        
        if self.use_pi_loss:
            ## Physical informed loss
            # Forward
            next_state_flat = self.forward(batch_idx, state_flat) # Propagated through network (already flat)
            #next_state = torch.reshape(next_state_flat, (-1,self.seq_len-1, self.n_inputs))
            df = self.method(state) # Differential computed with true model
            df_flat = torch.flatten(df, start_dim=1)
            train_loss = nn.MSELoss()(state_flat+df_flat, next_state_flat) + self.l1*sum(p.abs().sum() for p in self.parameters())
        else: 
            ## Data driven loss 
            # Forward
            if self.int_mode:
                # act as integrator
                next_state_flat = self.forward(batch_idx, state_flat)
                train_loss = nn.MSELoss()(next_state_flat, labels_flat) + self.l1*sum(p.abs().sum() for p in self.parameters())
            else:
                df_flat = self.method(state_flat) # Differential computed propagating network
                train_loss = nn.MSELoss()(state_flat+df_flat, labels_flat) + self.l1*sum(p.abs().sum() for p in self.parameters())
           
      
        # Compute loss between true and learned parameters
        params_loss = np.mean((self.system.params.detach().cpu().numpy()-self.true_system.params.detach().cpu().numpy())**2)
        self.log("params_loss", params_loss, prog_bar=True)
        
        # Logging to TensorBoard by default
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
        
        if self.use_pi_loss:
            ## Physical informed loss
            # Forward
            next_state_flat = self.forward(batch_idx, state_flat) # Propagated through network (already flat)
            #next_state = torch.reshape(next_state_flat, (-1,self.seq_len-1, self.n_inputs))
            df = self.method(state) # Differential computed with true model
            df_flat = torch.flatten(df, start_dim=1)
            val_loss = nn.MSELoss()(state_flat+df_flat, next_state_flat) + self.l1*sum(p.abs().sum() for p in self.parameters())
        else:
            ## Data driven loss 
            # Forward
            if self.int_mode:
                # act as integrator
                next_state_flat = self.forward(batch_idx, state_flat)
                val_loss = nn.MSELoss()(next_state_flat, labels_flat) + self.l1*sum(p.abs().sum() for p in self.parameters())
            else:
                df_flat = self.method(state_flat) # Differential computed propagating network
                val_loss = nn.MSELoss()(state_flat+df_flat, labels_flat) + self.l1*sum(p.abs().sum() for p in self.parameters())
            
                      
        # Logging to TensorBoard by default
        self.log("val_loss", val_loss, logger=True, on_epoch=True, prog_bar=True)
        self.log("epoch_num", self.current_epoch,prog_bar=True)
        return val_loss
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr, weight_decay=self.weight_decay)
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
        state = torch.flatten(inputs[:self.seq_len-1,:]).unsqueeze(0) # define first state
        self.eval()
        # run an interation and save first elements
        with torch.no_grad():
            if self.int_mode:
                state = self(0, state)
            else:
                state = state+self.dt*self(0, state)
            for i in range(self.seq_len-1):
                net_states.append(torch.reshape(state, (self.seq_len-1, self.n_inputs))[i].detach().cpu().numpy().tolist())
                
        # run all the other iterations
        for i in range(prediction_steps-self.seq_len):
            with torch.no_grad():
                if input_is_looped:
                    if self.int_mode:
                        state = self(0, state)
                    else:
                        state = state + self.dt*self(0, state)
                    net_states.append(state[0, -self.n_inputs::].detach().cpu().numpy().tolist())
                else:
                    if self.int_mode:
                        state = self(0, torch.flatten(inputs[i:i+self.seq_len-1,:]).unsqueeze(0))
                    else:
                        state = torch.flatten(inputs[i:i+self.seq_len-1,:]).unsqueeze(0) + self.dt*self(0, torch.flatten(inputs[i:i+self.seq_len-1,:]).unsqueeze(0))
                    net_states.append(state[0, -self.n_inputs::].detach().cpu().numpy().tolist())
                    
        return torch.tensor(np.array(net_states))
        
