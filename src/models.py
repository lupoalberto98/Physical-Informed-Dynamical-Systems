#!/usr/bin/env python3
import torch
import torch.nn as nn
import pytorch_lightning as pl
from VAE_functions import Sampler
import numpy as np

class LSTM(nn.Module):  
    def __init__(self, input_size, hidden_units, layers_num, drop_p=0):
        # Call the parent init function 
        super().__init__()
        # Define recurrent layers
        self.rnn = nn.LSTM(input_size=input_size, 
                           hidden_size=hidden_units,
                           num_layers=layers_num,
                           dropout=drop_p,
                           batch_first=True)
        # Define output layer
        self.out = nn.Linear(hidden_units, input_size)
        print("Network initialized")
        
    def forward(self, x, state=None):
        # LSTM
        x, rnn_state = self.rnn(x, state)
        # Linear layer
        x = self.out(x)
        # Remember to return also the RNN state, you will need it to generate data
        return x, rnn_state
    

class FFNet(nn.Module):

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
                          
        print("Network initialized")
                  

    def forward(self, x):

        for l in range(len(self.layers)):
            layer = self.layers[l]
            x = layer(x)
 
        return x


class VarFFEnc(nn.Module):
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
        
        print("Network initialized")
        
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


class VarFFAE(nn.Module):
    """
    Implement a general feed forward auto encoder
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
    def forward(self, x):
        # Encode data
        mean, logvar = self.encoder(x)
        
        # Sample data
        x = Sampler(mean, logvar)
        
        # Decode data
        x = self.decoder(x)
        
        return x
        
       
        
#def VarAE_LSTM(nn.Module):
#    def __init__():
        
class ESN(object):
    "Implementation of a Echo State Network"
    def __init__(self, params):
        self.in_dim = params["input_dimension"]
        self.h_dim = params["hidden_dimension"]
        self.out_dim = params["out_dimension"]
        self.beta = params["beta"]
        self.a = params["a"]
        self.b = params["b"]
        
        # Initialize W_in and W_r and hidden state
        self.W_in = np.random.rand(self.h_dim, self.in_dim)*2*self.a - np.ones((self.h_dim, self.in_dim))*self.a
        self.W_r = np.random.rand(self.h_dim, self.h_dim)*2*self.b - np.ones((self.h_dim, self.h_dim))*self.b
        self.h = [np.zeros((self.h_dim,))]
        self.P = np.zeros((self.out_dim, self.h_dim))
        
        
    def optimize(self, x):
        # Compute hidden state for training vector, shape [len_seq, n]
        dim_x = x.shape
        for t in range(dim_x[0]):
            self.h.append(np.tanh(np.dot(self.W_in, x[t]) + np.dot(self.W_r, self.h[-1])))
        for i in range(self.out_dim):
            for j in range(self.h_dim):
                num = 0
                den = 0
                for t in range(dim_x[0]-1):
                    num += x[t+1][i]*self.h[t+1][j]
                    den += self.h[t+1][j]**2
                den += self.beta
                self.P[i][j] = num/den
                
    def forward(self, x):
        dim_x = x.shape
        # Reinitializate hidden state
        self.h = [np.zeros((self.h_dim,))]
        for t in range(dim_x[0]):
            self.h.append(np.tanh(np.dot(self.W_in, x[t]) + np.dot(self.W_r, self.h[-1])))
        
        y = [x[0]]
        for t in range(dim_x[0]-1):
            y.append(np.dot(self.P,self.h[t+1]))
        
        return y
        