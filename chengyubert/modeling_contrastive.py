from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

from chengyubert.loss import ContrastiveLoss


class ContrastiveChengyuBERTIdiomEmbedding(nn.Module):

    def __init__(self, config, len_idiom_vocab):
        super().__init__()
        self.embedding = nn.Embedding(len_idiom_vocab, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, idiom_ids):
        embeddings = self.embedding(idiom_ids)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class ContrastiveChengyuBERTForPretrain(BertPreTrainedModel):
    def __init__(self, config, len_idiom_vocab, model_name='bert-dual'):
        super().__init__(config)
        self.model_name = model_name
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.idiom_embedding = ContrastiveChengyuBERTIdiomEmbedding(config, len_idiom_vocab)

        # register preset variables as buffer
        # So that, in testing , we can use buffer variables.
        self.register_buffer('enlarged_candidates', torch.arange(len_idiom_vocab))

        # projection MLP
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)

        self.init_weights()

    def vocab(self, over_states):
        idiom_embeddings = self.idiom_embedding(self.enlarged_candidates)
        return torch.einsum('bd,nd->bn', [over_states, self.projection(idiom_embeddings)])  # (b, 256, 10)

    def forward(self, input_ids, token_type_ids, attention_mask, positions, option_ids,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None, contrastive=False):
        encoded_outputs = self.bert(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)
        encoded_layer = encoded_outputs[0]

        blank_states = encoded_layer[[i for i in range(len(positions))], positions]  # [batch, hidden_state]

        over_logits = self.vocab(blank_states)

        if contrastive:
            if compute_loss:
                loss_fct = ContrastiveLoss()
                targets_embeddings = self.idiom_embedding(targets)
                return loss_fct(blank_states, self.projection(targets_embeddings))
            else:
                cond_logits = torch.gather(over_logits, dim=1, index=option_ids)
                return cond_logits, over_logits
        else:
            if compute_loss:
                loss_fct = nn.CrossEntropyLoss()
                return loss_fct(over_logits, targets)
            else:
                cond_logits = torch.gather(over_logits, dim=1, index=option_ids)
                return cond_logits, over_logits


class ContrastiveChengyuBERTForClozeChoice(BertPreTrainedModel):
    def __init__(self, config, len_idiom_vocab, model_name='bert-dual'):
        super().__init__(config)
        self.model_name = model_name
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.idiom_embedding = ContrastiveChengyuBERTIdiomEmbedding(config, len_idiom_vocab)

        # register preset variables as buffer
        # So that, in testing , we can use buffer variables.
        self.register_buffer('enlarged_candidates', torch.arange(len_idiom_vocab))

        # projection MLP
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)

        self.init_weights()

    def vocab(self, over_states):
        idiom_embeddings = self.idiom_embedding(self.enlarged_candidates)
        return torch.einsum('bd,nd->bn', [over_states, self.projection(idiom_embeddings)])  # (b, 256, 10)

    def forward(self, input_ids, token_type_ids, attention_mask, positions, option_ids,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        encoded_outputs = self.bert(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    inputs_embeds=inputs_embeds)
        encoded_layer = encoded_outputs[0]

        blank_states = encoded_layer[[i for i in range(len(positions))], positions]  # [batch, hidden_state]
        cls_states = encoded_layer[:, 0]
        over_states = self.over_linear(torch.cat([blank_states,
                                                  cls_states,
                                                  blank_states * cls_states,
                                                  blank_states - cls_states], dim=-1))

        if option_ids is None and options_embeds is None:
            raise ValueError('Either option_ids or options_embeds should be given.')
        elif options_embeds is not None:
            facial_state, meaning_state = options_embeds
        else:
            facial_state = self.idiom_facial_embedding(option_ids)  # (b, 10, 768)
            meaning_state = self.idiom_meaning_embedding(option_ids)  # (b, 10, 768)

        over_logits = self.vocab(over_states)
        # cond_logits = torch.gather(over_logits, dim=1, index=option_ids)

        mo_logits = torch.einsum('bld,bnd->bln', [encoded_layer, meaning_state])  # (b, 256, 10)
        c_mo_logits, _ = torch.max(mo_logits, dim=1)
        # over_states = cls_states

        c_fo_logits = torch.einsum('bd,bnd->bn', [over_states, facial_state])  # (b, 10)
        logits = c_mo_logits + c_fo_logits

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, targets)
            target = torch.gather(option_ids, dim=1, index=targets.unsqueeze(1))
            over_loss = loss_fct(over_logits, target.squeeze(1))
            return loss, over_loss
        else:
            return logits, over_logits
