import torch.nn as nn

from .baseRNN import BaseRNN

import pdb

class EncoderRNN(BaseRNN):
    def __init__(self, vocab_size, embed_model=None, emb_size=100, hidden_size=128, \
                 input_dropout_p=0, dropout_p=0, n_layers=1, bidirectional=False, \
                 rnn_cell=None, rnn_cell_name='gru', variable_lengths=True):
        super(EncoderRNN, self).__init__(vocab_size, emb_size, hidden_size,
              input_dropout_p, dropout_p, n_layers, rnn_cell_name)
        self.variable_lengths = variable_lengths
        self.bidirectional = bidirectional
        if embed_model == None:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        else:
            self.embedding = embed_model
        if rnn_cell == None:
            self.rnn = self.rnn_cell(emb_size, hidden_size, n_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=dropout_p)
        else:
            self.rnn = rnn_cell

    def forward(self, input_var, input_lengths=None):
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)
        #pdb.set_trace()
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return output, hidden



