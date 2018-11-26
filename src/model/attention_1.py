import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention_1(nn.Module):
    def __init__(self, dim):
        super(Attention_1, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, output, context):
        '''
        output: decoder,  (batch, 1, hiddem_dim2)
        context: from encoder, (batch, n, hidden_dim1)
        actually, dim2 == dim1, otherwise cannot do matrix multiplication 
        '''
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (b, o, dim) * (b, dim, i) -> (b, o, i)
        attn = torch.bmm(output, context.transpose(1,2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (b, o, i) * (b, i, dim) -> (b, o, dim)
        mix = torch.bmm(attn, context)

        combined = torch.cat((mix, output), dim=2)

        output = F.tanh(self.linear_out(combined.view(-1, 2*hidden_size)))\
                            .view(batch_size, -1, hidden_size)

        # output: (b, o, dim)
        # attn  : (b, o, i)
        return output, attn

