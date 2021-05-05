import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, n_hidden_enc, n_hidden_dec):
        super().__init__()

        self.h_hidden_enc = n_hidden_enc
        self.h_hidden_dec = n_hidden_dec

        self.W = nn.Linear(n_hidden_enc + n_hidden_dec, n_hidden_dec, bias=False)
        self.V = nn.Parameter(torch.rand(n_hidden_dec))

    def forward(self, hidden_dec, last_layer_enc, attention_mask):
        '''
            PARAMS:
                hidden_dec:     [b, n_hidden_dec]
                last_layer_enc: [b, seq_len, n_hidden_enc * 2]

            RETURN:
                att_weights:    [b, src_seq_len]
        '''

        batch_size = last_layer_enc.size(0)
        src_seq_len = last_layer_enc.size(1)

        hidden_dec = hidden_dec.unsqueeze(1).repeat(1, src_seq_len, 1)  # [b, src_seq_len, n_hidden_dec]

        tanh_W_s_h = torch.tanh(
            self.W(torch.cat((hidden_dec, last_layer_enc), dim=-1)))  # [b, src_seq_len, n_hidden_dec]
        tanh_W_s_h = tanh_W_s_h.permute(0, 2, 1)  # [b, n_hidde_dec, seq_len]

        V = self.V.repeat(batch_size, 1).unsqueeze(1)  # [b, 1, n_hidden_dec]
        e = torch.bmm(V, tanh_W_s_h).squeeze(1)  # [b, seq_len]
        e = attention_mask + e

        att_weights = F.softmax(e, dim=1)  # [b, src_seq_len]

        return att_weights
