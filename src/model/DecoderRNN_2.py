#encoding: utf-8
import random

import numpy as np
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from .baseRNN import BaseRNN

class DecoderRNN_2(BaseRNN):
    def __init__(self, vocab_size, class_size, embed_model=None, emb_size=100, hidden_size=128, \
                 n_layers=1, rnn_cell = None, rnn_cell_name='lstm', \
                 sos_id=1, eos_id=0, input_dropout_p=0, dropout_p=0): #use_attention=False):
        super(DecoderRNN_2, self).__init__(vocab_size, emb_size, hidden_size,
              input_dropout_p, dropout_p,
              n_layers, rnn_cell_name)
        self.vocab_size = vocab_size
        self.class_size = class_size
        self.sos_id = sos_id
        self.eos_id = eos_id

        if embed_model == None:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        else:
            self.embedding = embed_model

        if rnn_cell == None:
            self.rnn = self.rnn_cell(emb_size, hidden_size, n_layers, \
                                 batch_first=True, dropout=dropout_p)
        else:
            self.rnn = rnn_cell

        self.out = nn.Linear(self.hidden_size, self.class_size)

    #def _init_state(self, encoder_hidden, op_type):
        

    def forward_step(self, input_var, hidden, function):
        '''
        normal forward, step by step or all steps together
        '''
        
        if len(input_var.size()) == 1:
            input_var = torch.unsqueeze(input_var,1)
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        predicted_softmax = function(self.out(\
                            output.contiguous().view(-1, self.hidden_size)))\
                            .view(batch_size, output_size, -1)
        return predicted_softmax, hidden

    def decode(self, step, step_output):
        '''
        step_output: batch x classes , prob_log
        symbols: batch x 1
        '''
        symbols = step_output.topk(1)[1] 
        return symbols

    def decode_rule(self, step, sequence_symbols_list, step_output):
        symbols = self.rule_filter(sequence_symbols_list, step_output)
        return symbols
        
    def forward_normal_teacher_1(self, decoder_inputs, decoder_init_hidden, function):
        '''
        decoder_input: batch x seq_lengths x indices( sub last(-1), add first(sos_id))
        decoder_init_hidden: processed considering encoder layers, bi 
            lstm : h_0 (num_layers * num_directions, batch, hidden_size)
                   c_0 (num_layers * num_directions, batch, hidden_size)
            gru  : 
        decoder_outputs: batch x seq_lengths x classes,  probility_log
            lstm : h_n (num_layers * num_directions, batch, hidden_size)
                   c_n (num_layers * num_directions, batch, hidden_size)
            gru  :
        decoder_hidden: layers x batch x hidden_size 
        '''
        decoder_outputs, decoder_hidden = self.forward_step(\
                          decoder_inputs, decoder_init_hidden, function=function)
        decoder_outputs_list = []
        sequence_symbols_list = []
        for di in range(decoder_outputs.size(1)):
            step_output = decoder_outputs[:, di, :]
            symbols = self.decode(di, step_output)
            decoder_outputs_list.append(step_output)
            sequence_symbols_list.append(symbols)
        return decoder_outputs_list, decoder_hidden, sequence_symbols_list

    def forward_normal_teacher(self, decoder_inputs, decoder_init_hidden, function):
        decoder_outputs_list = []
        sequence_symbols_list = []
        decoder_hidden = decoder_init_hidden
        seq_len = decoder_inputs.size(1)
        for di in range(seq_len):
            decoder_input = decoder_inputs[:, di]
            #deocder_input = torch.unsqueeze(decoder_input, 1)
            #print '1', deocder_input.size()
            decoder_output, decoder_hidden = self.forward_step(\
                decoder_input, decoder_hidden, function=function)
            step_output = decoder_output.squeeze(1)
            if self.use_rule == False:
                symbols = self.decode(di, step_output)
            else:
                symbols = self.decode_rule(di, sequence_symbols_list, step_output)
            decoder_outputs_list.append(step_output)
            sequence_symbols_list.append(symbols)
        return decoder_outputs_list, decoder_hidden, sequence_symbols_list

    def symbol_norm(self, symbols):
        symbols = symbols.view(-1).data.cpu().numpy() 
        new_symbols = []
        for idx in symbols:
            #print idx, 
            #print self.class_list[idx],
            #pdb.set_trace()
            #print self.vocab_dict[self.class_list[idx]]
            new_symbols.append(self.vocab_dict[self.class_list[idx]])
        new_symbols = Variable(torch.LongTensor(new_symbols)) 
        #print new_symbols
        new_symbols = torch.unsqueeze(new_symbols, 1)
        if self.use_cuda:
            new_symbols = new_symbols.cuda()
        return new_symbols


    def forward_normal_no_teacher(self, decoder_input, decoder_init_hidden, max_length,  function):
        '''
        decoder_input: batch x 1
        decoder_output: batch x 1 x classes,  probility_log
        '''
        decoder_outputs_list = []
        sequence_symbols_list = []
        decoder_hidden = decoder_init_hidden
        for di in range(max_length):
            decoder_output, decoder_hidden = self.forward_step(\
                           decoder_input, decoder_hidden, function=function)
            step_output = decoder_output.squeeze(1)
            #print step_output.size()
            if self.use_rule == False:
                symbols = self.decode(di, step_output)
            else:
                symbols = self.decode_rule(di, sequence_symbols_list, step_output) 
            decoder_input = self.symbol_norm(symbols)
            decoder_outputs_list.append(step_output)
            sequence_symbols_list.append(symbols)
        #print sequence_symbols_list
        return decoder_outputs_list, decoder_hidden, sequence_symbols_list


    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None, template_flag=True,\
                function=F.log_softmax, teacher_forcing_ratio=0, use_rule=False, use_cuda=False, \
                vocab_dict = None, vocab_list = None, class_dict = None, class_list = None):
        '''
        使用rule的时候，teacher_forcing_rattio = 0
        '''
        self.use_rule = use_rule
        self.use_cuda = use_cuda
        self.class_dict = class_dict
        self.class_list = class_list
        self.vocab_dict = vocab_dict
        self.vocab_list = vocab_list

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        #pdb.set_trace()
        batch_size = encoder_outputs.size(0)
        #batch_size = inputs.size(0)

        pad_var = torch.LongTensor([self.sos_id]*batch_size) # marker

        pad_var = Variable(pad_var.view(batch_size, 1))#.cuda() # marker
        if self.use_cuda:
            pad_var = pad_var.cuda()

        decoder_init_hidden = encoder_hidden

        if template_flag == False:
            max_length = 40
        else:
            max_length = inputs.size(1)

        #inputs = torch.cat((pad_var, inputs), 1) # careful concate  batch x (seq_len+1)
        #inputs = inputs[:, :-1] # batch x seq_len

        if use_teacher_forcing:
            ''' all steps together'''
            inputs = torch.cat((pad_var, inputs), 1) # careful concate  batch x (seq_len+1)
            inputs = inputs[:, :-1] # batch x seq_len
            decoder_inputs = inputs 
            return self.forward_normal_teacher(decoder_inputs, decoder_init_hidden, function)
        else:
            #decoder_input = inputs[:,0].unsqueeze(1) # batch x 1
            decoder_input = pad_var#.unsqueeze(1) # batch x 1
            #pdb.set_trace()
            return self.forward_normal_no_teacher(decoder_input, decoder_init_hidden, \
                                                  max_length, function)


    def rule(self, symbol):
        filters = []
        if self.class_list[symbol] in ['+', '-', '*', '/']:
            filters.append(self.class_dict['+'])
            filters.append(self.class_dict['-'])
            filters.append(self.class_dict['*']) 
            filters.append(self.class_dict['/']) 
            filters.append(self.class_dict[')']) 
            filters.append(self.class_dict['=']) 
        elif self.class_list[symbol] == '=':
            filters.append(self.class_dict['+']) 
            filters.append(self.class_dict['-']) 
            filters.append(self.class_dict['*']) 
            filters.append(self.class_dict['/']) 
            filters.append(self.class_dict['=']) 
            filters.append(self.class_dict[')']) 
        elif self.class_list[symbol] == '(':
            filters.append(self.class_dict['(']) 
            filters.append(self.class_dict[')']) 
            filters.append(self.class_dict['+']) 
            filters.append(self.class_dict['-']) 
            filters.append(self.class_dict['*']) 
            filters.append(self.class_dict['/']) 
            filters.append(self.class_dict['=']) 
        elif self.class_list[symbol] == ')':
            filters.append(self.class_dict['('])
            filters.append(self.class_dict[')'])
            for k,v in self.class_dict.items():
                if 'temp' in k:
                    filters.append(v)
        elif 'temp' in self.class_list[symbol]:
            filters.append(self.class_dict['('])
            filters.append(self.class_dict['=']) 
        return np.array(filters)

    def filter_op(self):
        filters = []
        filters.append(self.class_dict['+']) 
        filters.append(self.class_dict['-']) 
        filters.append(self.class_dict['*']) 
        filters.append(self.class_dict['/']) 
        filters.append(self.class_dict['^']) 
        return np.array(filters)

    def filter_END(self):
        filters = []
        filters.append(self.class_dict['END_token']) 
        return np.array(filters)
        

    def rule_filter(self, sequence_symbols_list, current):
        '''
        32*28
        '''
        op_list = ['+','-','*','/','^']
        cur_out = current.cpu().data.numpy()
        #print len(sequence_symbols_list)
        #pdb.set_trace()
        cur_symbols = []
        if sequence_symbols_list == [] or len(sequence_symbols_list) <= 1:
            #filters = self.filter_op()
            filters = np.append(self.filter_op(), self.filter_END())
            for i in range(cur_out.shape[0]):
                cur_out[i][filters] = -float('inf')
                cur_symbols.append(np.argmax(cur_out[i]))
        else:
            for i in range(sequence_symbols_list[0].size(0)):
                num_var = 0
                num_op = 0
                for j in range(len(sequence_symbols_list)):
                    symbol = sequence_symbols_list[j][i].cpu().data[0]
                    if self.class_list[symbol] in op_list:
                        num_op += 1
                    elif 'temp' in self.class_list[symbol] or self.class_list[symbol] in ['1', 'PI']:
                        num_var += 1
                if num_var >= num_op + 2:
                    filters = self.filter_END()
                    cur_out[i][filters] = -float('inf')
                elif num_var == num_op + 1:
                    filters = self.filter_op() 
                    cur_out[i][filters] = -float('inf')
                cur_symbols.append(np.argmax(cur_out[i]))
        cur_symbols = Variable(torch.LongTensor(cur_symbols))
        cur_symbols = torch.unsqueeze(cur_symbols, 1)
        cur_symbols = cur_symbols.cuda()
        return cur_symbols

