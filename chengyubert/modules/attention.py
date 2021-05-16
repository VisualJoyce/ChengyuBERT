import torch
import torch.nn as nn
import torch.nn.functional as F

from chengyubert.modules.utils import masked_softmax


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


class ContrastiveCoAttention(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.affinity_linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, L, I, mask_L, mask_I):
        # L = idiom_states
        # I = encoded_context_masked
        # I = encoded_context_masked

        # idiom_length = (gather_index > 0).sum(1)
        # idiom_mask = sequence_mask(idiom_length)

        AI = self.affinity_linear(I)

        # co attention
        L_T = torch.transpose(L, 1, 2)  # B x l x m + 1
        Z = torch.bmm(AI, L_T)  # L = B x n + 1 x m + 1

        # row max
        A_L_ = masked_softmax(Z.max(dim=1)[0], mask=mask_L)  # B x n + 1 x m + 1
        C_L = torch.einsum('bn,bnd->bd', [A_L_, L])

        # col max
        A_I_ = masked_softmax(Z.max(dim=2)[0], mask=mask_I)  # B x n + 1 x m + 1
        C_I = torch.einsum('bn,bnd->bd', [A_I_, I])
        return C_L, C_I
