#!/usr/bin/env python3
import torch
import torch.nn as nn


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