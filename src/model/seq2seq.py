# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class Seq2seq(nn.Module):
    
    def __init__(self, encoder=None, decoder=None, decoder_function=F.log_softmax):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_function = decoder_function

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, target_variable=None, template_flag=True,\
                teacher_forcing_ratio=0, mode=0, use_rule=False, use_cuda=False, \
                vocab_dict = None, vocab_list = None, class_dict=None, class_list=None):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)

        encoder_hidden = self.process_gap_encoder_decoder(encoder_hidden, mode)

        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              template_flag = template_flag,
                              function=self.decoder_function,
                              teacher_forcing_ratio=teacher_forcing_ratio,
                              use_rule = use_rule,
                              use_cuda = use_cuda,
                              vocab_dict = vocab_dict,
                              vocab_list = vocab_list,
                              class_dict = class_dict,
                              class_list = class_list)

        return result

    def process_gap_encoder_decoder(self, encoder_hidden, mode):
        '''
        要么层数相同， 要么encoder是n层，decoder是1层
        '''
        if mode == 0:
            ''' lstm -> lstm '''
            encoder_hidden = self._init_state(encoder_hidden)
        elif mode == 1:
            ''' gru -> gru '''
            encoder_hidden = self._init_state(encoder_hidden)
        elif mode == 2:
            ''' gru -> lstm '''
            encoder_hidden = (encoder_hidden, encoder_hidden)
            encoder_hidden = self._init_state(encoder_hidden)
        elif mode == 3:
            ''' lstm -> gru '''
            encoder_hidden = encoder_hidden[0]
            encoder_hidden = self._init_state(encoder_hidden)
        return encoder_hidden

    def _init_state(self, encoder_hidden):
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        if self.encoder.bidirectional:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h


