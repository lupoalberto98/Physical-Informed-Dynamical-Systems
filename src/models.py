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


        

    
class FFNet(pl.LightningModule):

    def __init__(self, seq_len, n_inputs, n_outputs, hidden_layers, system, true_system, drop_p=0.3, lr=0.001, dt=0.002, 
                method_name="RK4", activation="ReLU", use_pi_loss=False, l1=0, weight_decay=0):
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
        if self.use_pi_loss:
            # Physical informed loss
            self.method = getattr(utils, self.method_name)(self.dt, model=system.forward)
        else:
            # Dat driven loss
            self.method = getattr(utils, self.method_name)(self.dt, model=self.forward)
        
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
            self.method = getattr(utils, self.method_name)(self.dt, model=self.forward)
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
            self.method = getattr(utils, self.method_name)(self.dt, model=self.forward)
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
            if self.use_pi_loss:
                state = self(0, state)
            else:
                state = state+self.dt*self(0, state)
            for i in range(self.seq_len-1):
                net_states.append(torch.reshape(state, (self.seq_len-1, self.n_inputs))[i].detach().cpu().numpy().tolist())
                
        # run all the other iterations
        for i in range(prediction_steps-self.seq_len):
            with torch.no_grad():
                if input_is_looped:
                    if self.use_pi_loss:
                        state = self(0, state)
                    else:
                        state = state + self.dt*self(0, state)
                    net_states.append(state[0, -self.n_inputs::].detach().cpu().numpy().tolist())
                else:
                    if self.use_pi_loss:
                        state = self(0, torch.flatten(inputs[i:i+self.seq_len-1,:]).unsqueeze(0))
                    else:
                        state = torch.flatten(inputs[i:i+self.seq_len-1,:]).unsqueeze(0) + self.dt*self(0, torch.flatten(inputs[i:i+self.seq_len-1,:]).unsqueeze(0))
                    net_states.append(state[0, -self.n_inputs::].detach().cpu().numpy().tolist())
                    
        return torch.tensor(np.array(net_states))
        
        
"""    
#def VarAE_LSTM(nn.Module):
#    def __init__():
        
class ESN(object):
    "Implementation of a Echo State Network"
    def __init__(self, params):
        self.in_dim = params["input_dimension"]
        self.h_dim = params["hidden_dimension"]
        self.out_dim = params["out_dimension"]
        self.beta = params["beta"]
        self.spectral_radius = params["spectral_radius"]
        self.sparsity = 1. - 2*params["edges"]/(self.h_dim*(self.h_dim-1))
        
        
        
        # Initialize W_in and W_r 
        self.W_in = np.random.rand(self.h_dim, self.in_dim)*2 - 1.0
        self.W_r = np.random.rand(self.h_dim, self.h_dim)*2 - 1.0
        
        # delete the fraction of connections given by (self.sparsity):
        self.W_in[np.random.rand(*self.W_in.shape) < self.sparsity] = 0.
        
        # Rescale to match spectral radius
        radius = np.max(np.abs(np.linalg.eigvals(self.W_r)))
        self.W_r = self.W_r * self.spectral_radius / radius
        
        self.h = [np.zeros((self.h_dim,))]
        self.P = np.zeros((self.out_dim, self.h_dim))
        self.t_train = 0
        
        
    def fit(self, x):
        # Compute hidden state for training vector, shape [len_seq, n]
        dim_x = x.shape
        for t in range(dim_x[0]):
            self.h.append(np.tanh(np.dot(self.W_in, x[t]) + np.dot(self.W_r, self.h[-1])))
            self.t_train = self.t_train + 1
        ### Temporal loop
        num = np.zeros((self.out_dim, self.h_dim))
        den = np.zeros((self.out_dim, self.h_dim))
        ones_vec = np.ones((self.out_dim))
        
        for t in range(1, dim_x[0]):
            # Copmute h tilde, first half equal to h, second to h^2
            h_tilde = self.h[t]
            h_tilde[int(self.h_dim/2):] = h_tilde[int(self.h_dim/2):]*h_tilde[int(self.h_dim/2):]
            # Compute numerator
            num = num + np.tensordot(x[t], h_tilde, axes=0)
            mat = np.tensordot(ones_vec, h_tilde,axes=0)
            den = den + np.matmul(mat, np.diag(h_tilde))
            
        # Add bias
        den = den + self.beta*np.ones((self.out_dim, self.h_dim))
        
        # Compute output matrix P
        self.P = num/den
                
    def predict(self, x, continuation = True):
        dim_x = x.shape
        # Reinitializate hidden state
        if continuation is not True:
            self.h = [np.zeros((self.h_dim,))]
            self.t_train = 0
            
        # Initialize state
        y = [x[0]]
        h = self.h[-1]
        for t in range(dim_x[0]-1):
            h = np.tanh(np.dot(self.W_in, y[-1]) + np.dot(self.W_r, h))
            y.append(np.matmul(self.P,h))
        
        return y
        

        ### Prepare network input and labels and first net_out for curriculum learning
        state  = batch[:, :-self.feedforward_steps, :]
        seq_len = batch.shape[1]
        
        
        # Initialize loss
        pi_loss = 0 # physical informed  loss
        dd_loss = 0 # data driven loss
        
        # Compute physical informed loss
        for step in range(self.feedforward_steps):
            true_state = batch[:,step:-self.feedforward_steps+step,:]
            labels = batch[:,1+step:seq_len-self.feedforward_steps+step+1,:]
            if self.curriculum_learning is None:
                # Forward
                next_state, _ = self.forward(state)
                # Loss update
                if self.pi_loss_fn is not None:
                    pi_loss += self.pi_loss_fn(state, next_state)
                if self.dd_loss_fn is not None:
                    dd_loss += self.dd_loss_fn(next_state, labels)
                state = next_state.detach()
            else:
                if self.feedforward_steps==1:
                    # Forward
                    next_state, _ = self.forward(state)
                    # Loss update
                    if self.pi_loss_fn is not None:
                        pi_loss += self.pi_loss_fn(state, next_state)
                    if self.dd_loss_fn is not None:
                        dd_loss += self.dd_loss_fn(next_state, labels)
                    state = next_state.detach()
                else:
                    shape = state.shape
                    forget = torch.rand(shape[1]).unsqueeze(1).unsqueeze(0)
                    forget[forget>self.curriculum_learning[self.current_epoch]] = 1.
                    forget[forget<self.curriculum_learning[self.current_epoch]] = 0.
                    # Update state and detach
                    state = next_state*forget + true_state*(1.-forget)
                    state = state.detach()
                    # Forward
                    next_state, _ = self.forward(state)
                    # Loss update
                    if self.pi_loss_fn is not None:
                        pi_loss += self.pi_loss_fn(state, next_state)
                    if self.dd_loss_fn is not None:
                        dd_loss += self.dd_loss_fn(next_state, labels)
                    state = next_state.detach()
                    
        # Total loss
        if self.annealing is None:
            val_loss = pi_loss + dd_loss
        else:
            val_loss = pi_loss + dd_loss*self.annealing[self.current_epoch]
        
"""