from unicodedata import bidirectional
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler

class ArgoLSTM(nn.Module):
    def __init__(self, batch_size, input_size=2, hidden_size=50, output_size=2, num_layers=2, dropout=0.2, bidirectional=False):
        super(ArgoLSTM, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.relu = nn.ELU()
        self.D = 1
        if self.bidirectional == True:
            self.D = 2
        
        # LSTM input: [N, 50, 2]
        # Output: [2, N, 60]
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers, \
                            batch_first=True, 
                            dropout=dropout, 
                            bidirectional=bidirectional
                        ) # outputs (batch_size, length, hidden size)
        self.linear = nn.Linear(hidden_size * self.D, output_size)

    def forward(self, x, hidden):
        out, (h, c) = self.lstm(x, hidden)
        out = self.linear(out)
        out = self.relu(out)
        return out, (h, c)

    def init_Hidden(self, device):
        # (layer, batch, hidden)

        h = torch.zeros(self.num_layers * self.D, self.batch_size, self.hidden_size).to(device)
        c = torch.zeros(self.num_layers * self.D, self.batch_size, self.hidden_size).to(device)
        return h, c
