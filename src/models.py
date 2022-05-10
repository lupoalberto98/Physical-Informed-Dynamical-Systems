#!/usr/bin/env python3
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.optim as optim
from utils import Sampler, nKLDivLoss
import numpy as np
import math


class LSTM(pl.LightningModule):  
    def __init__(self, params):
        # Call the parent init function 
        super().__init__()
        # Retrieve parameters
        self.input_size = params["input_size"]
        self.hidden_units = params["hidden_units"]
        self.layers_num = params["layers_num"]
        self.drop_p = params["drop_p"]
        self.loss_fn = params["loss_fn"]
        self.lr = params["lr"]
        self.feedforward_steps = params["feedforward_steps"]
        self.curriculum_learning = params["curriculum_learning"]
        
        # Define recurrent layers
        self.rnn = nn.LSTM(input_size=self.input_size, 
                           hidden_size=self.hidden_units,
                           num_layers=self.layers_num,
                           dropout=self.drop_p,
                           batch_first=True)
        # Define output layer
        self.out = nn.Linear(self.hidden_units, self.input_size)
        print("Network initialized")
        
    def forward(self, x, state=None):
        # LSTM
        x, rnn_state = self.rnn(x, state)
        # Linear layer
        x = self.out(x)
        # Remember to return also the RNN state, you will need it to generate data
        return x, rnn_state
    
    def training_step(self, batch, batch_idx):
        ### Prepare network input and labels and first net_out for curriculum learning
        state  = batch[:, :-self.feedforward_steps, :]
        labels = batch[:, self.feedforward_steps:, :]
        
        # Initialize loss
        train_loss = 0
        for step in range(self.feedforward_steps):
            # Forward pass
            next_state, _ = self.forward(state)
            train_loss += self.loss_fn(state, next_state)
            state = next_state.detach()
            
        """
        # Curriculum learning startegy
        if self.curriculum_learning is not None:
            shape = state.shape
            forget = torch.rand(shape[1]).unsqueeze(1).unsqueeze(0)
            forget[forget>self.curriculum_learning[self.current_epoch]] = 1.
            forget[forget<self.curriculum_learning[self.current_epoch]] = 0.
            # Update state and detach
            state = torch.cat((state[:,0,:].unsqueeze(1), next_state[:,:-1,:]), dim=1)*forget + state*(1.-forget)
            state = state.detach()
            # Forward
            next_state, _ = self.forward(state)
           
        # Compute loss
        if self.loss_fn.__class__.__name__ == "EuDLoss":
            train_loss = self.loss_fn(state, next_state)
        elif self.loss_fn.__class__.__name__ == "CeDLoss":
            # Trim batches
            state = state[:,:-1,:]
            next_state = next_state[:,:-1,:]
            labels = labels[:,1:,:]
            # Forward again
            next_next_state, _ = self.forward(next_state)
            # Compute loss
            train_loss = self.loss_fn(state, next_state, next_next_state)
        else:
            raise NameError(self.loss_fn.__class__.__name__)
        """
        
        # Logging to TensorBoard by default
        self.log("train_loss", train_loss, prog_bar=True)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        ### Prepare network input and labels and first net_out for curriculum learning
        state  = batch[:, :-self.feedforward_steps, :]
        labels = batch[:, self.feedforward_steps:, :]
        
        # Initialize loss
        val_loss = 0
        for step in range(self.feedforward_steps):
            # Forward pass
            next_state, _ = self.forward(state)
            val_loss += self.loss_fn(state, next_state)
            state = next_state.detach()
        """
        # Curriculum learning startegy
        if self.curriculum_learning is not None:
            shape = state.shape
            forget = torch.rand(shape[1]).unsqueeze(1).unsqueeze(0)
            forget[forget>self.curriculum_learning[self.current_epoch]] = 1.
            forget[forget<self.curriculum_learning[self.current_epoch]] = 0.
            # Update state and detach
            state = torch.cat((state[:,0,:].unsqueeze(1), next_state[:,:-1,:]), dim=1)*forget + state*(1.-forget)
            state = state.detach()
            # Forward
            next_state, _ = self.forward(state)
        # Compute loss
        if self.loss_fn.__class__.__name__ == "EuDLoss":
            val_loss = self.loss_fn(state, next_state)
        elif self.loss_fn.__class__.__name__ == "CeDLoss":
            # Trim batches
            state = state[:,:-1,:]
            next_state = next_state[:,:-1,:]
            labels = labels[:,1:,:]
            # Forward again
            next_next_state, _ = self.forward(next_state)
            # Compute loss
            val_loss = self.loss_fn(state, next_state, next_next_state)
        else:
            raise NameError(self.loss_fn.__class__.__name__)
        """
        # Logging to TensorBoard by default
        self.log("val_loss", val_loss, logger=True, on_epoch=True, prog_bar=True)
        self.log("epoch_num", self.current_epoch,prog_bar=True)
        return val_loss
                        
                        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr)
        return optimizer
        

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
    def __init__(self, params):
        # Initialize parent class
        super().__init__()
        # Retrieve parameters
        self.d_model = params["d_model"]
        self.nhead = params["nhead"]
        self.num_encoder_layers = params["num_encoder_layers"]
        self.num_decoder_layers = params["num_decoder_layers"]
        self.dim_feedforward = params["dim_feedforward"]
        self.dropout = params["dropout"]
        self.activation = params["activation"]
        self.loss_fn = params["loss_fn"]
        self.lr = params["lr"]
        self.src_mask = None
        self.tgt_mask = None
        self.apply_src_mask = params["apply_src_mask"]
        self.apply_tgt_mask = params["apply_tgt_mask"]
        
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
        print("Network initialized")
        
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
        train_loss = self.loss_fn(state, next_state)
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
        val_loss = self.loss_fn(state, next_state)
        # Logging to TensorBoard by default
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.lr)
        return optimizer
    
class FFNet(pl.LightningModule):

    def __init__(self, params):
        """
        Initialize a typical feedforward network with different hidden layers
        The input is typically a mnist image, given as a torch tensor of size = (1,784)
         ----------
        Parameters:
        layers_sizes = list of sizes of the hidden layers, the first is the visible layer and the last is the output layer

        """

        super().__init__()

        # Parameters
        self.layers_sizes = params["layers_sizes"]
        self.num_layers = len(self.layers_sizes)
        self.act = params["act"]
        self.drop_p = params["drop_p"]
        
        # Network architecture
        layers = []
        for l in range(self.num_layers-2):
            layers.append(nn.Linear(in_features = self.layers_sizes[l], out_features = self.layers_sizes[l+1]))
            layers.append(nn.Dropout(self.drop_p, inplace = False))
            layers.append(self.act)
        
        layers.append(nn.Linear(in_features = self.layers_sizes[self.num_layers-2], out_features = self.layers_sizes[self.num_layers-1]))
        
        self.layers = nn.ModuleList(layers)
                          
        print("Feedforward network initialized")
                  

    def forward(self, x):

        for l in range(len(self.layers)):
            layer = self.layers[l]
            x = layer(x)
 
        return x


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
        