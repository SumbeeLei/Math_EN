import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb
import torch.nn.functional as F

class BaseRNN(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, input_dropout_p, dropout_p, \
                          n_layers, rnn_cell_name):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        self.rnn_cell_name = rnn_cell_name
        if rnn_cell_name.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell_name.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell_name))
        self.dropout_p = dropout_p

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

