from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class ChengyuBertEmb(BertPreTrainedModel):
    def __init__(self, config, len_idiom_vocab, model_name='bert-emb-300'):
        super().__init__(config)
        self.model_name = model_name
        chengyu_emb_dim = int(model_name.split('-')[-1])
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.idiom_embedding = nn.Embedding(len_idiom_vocab, chengyu_emb_dim)
        self.project_linear = nn.Linear(config.hidden_size, chengyu_emb_dim)
        self.register_buffer('enlarged_candidates', torch.arange(len_idiom_vocab))
        self.init_weights()

    def vocab(self, over_states):
        idiom_embeddings = self.idiom_embedding(self.enlarged_candidates)
        c_mo_logits = torch.einsum('bd,nd->bn', [over_states, idiom_embeddings])  # (b, 256, 10)
        return c_mo_logits

    def forward(self, input_ids, token_type_ids, attention_mask, positions, option_ids,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        encoded_outputs = self.bert(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    inputs_embeds=inputs_embeds)
        encoded_layer = encoded_outputs[0]
        blank_states = encoded_layer[[i for i in range(len(positions))], positions]  # [batch, hidden_state]
        over_logits = self.vocab(self.project_linear(blank_states))

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            target = torch.gather(option_ids, dim=1, index=targets.unsqueeze(1))
            over_loss = loss_fct(over_logits, target.squeeze(1))
            return over_loss
        else:
            cond_logits = torch.gather(over_logits, dim=1, index=option_ids)
            return cond_logits, over_logits
