

class DecoderRNN(BaseRNN):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, class_size, embed_model=None, emb_size=100, hidden_size=128, \
                 n_layers=1, rnn_cell_name='lstm', bidirectional=False, sos_id=1, eos_id=0,
                 input_dropout_p=0, dropout_p=0, use_attention=False):
        super(DecoderRNN, self).__init__(vocab_size, emb_size, hidden_size,
              input_dropout_p, dropout_p,
              n_layers, rnn_cell_name)
        self.bidirectional_encoder = bidirectional
        self.vocab_size = vocab_size
        self.class_size = class_size
        self.use_attention = use_attention
        self.sos_id = sos_id
        self.eos_id = eos_id

        if embed_model == None:
            self.embedding = nn.Embedding(vocab_size, emb_size)
        else:
            self.embedding = embed_model

        self.rnn = self.rnn_cell(emb_size, hidden_size, n_layers, \
                                 batch_first=True, dropout=dropout_p)

        if use_attention == 1:
            self.attention = Attention_f(self.hidden_size)
        elif use_attention == 2:
            self.attention = Attention_b(self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.class_size)

    def _init_state(self, encoder_hidden, op_type):
        

    def forward_one_step(self, input_var, hidden, encoder_outputs, function):
        '''
        normal forward, step by step
        '''
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        predicted_softmax = function(self.out(\
                            output.contiguous().view(-1, self.hidden_size)))\
                            .view(batch_size, output_size, -1)
        return predicted_softmax, hidden

    def forward_normal_teacher(self):
        pass

    def forward_normal_no_teacher(self):
        pass

    def forward(self):
        pass
                        

        



