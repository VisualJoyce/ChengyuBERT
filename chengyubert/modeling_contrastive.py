from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

from chengyubert.loss import ContrastiveLoss


class ChengyuBERTContrastive(BertPreTrainedModel):
    def __init__(self, config, len_idiom_vocab, model_name='bert-contrastive'):
        super().__init__(config)
        self.model_name = model_name
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.over_linear = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.idiom_embedding = nn.Embedding(len_idiom_vocab, config.hidden_size)

        # register preset variables as buffer
        # So that, in testing , we can use buffer variables.
        self.register_buffer('enlarged_candidates', torch.arange(len_idiom_vocab))

        # projection MLP
        self.projection = nn.Sequential(nn.Linear(config.hidden_size, 512, bias=False),
                                        nn.LayerNorm(512),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(512, 100, bias=True))

        self.init_weights()

    def vocab(self, over_states):
        idiom_embeddings = self.idiom_embedding(self.enlarged_candidates)
        return torch.einsum('bd,nd->bn', [over_states, idiom_embeddings])  # (b, 256, 10)

    def forward(self, input_ids, token_type_ids, attention_mask, positions, option_ids,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None, contrastive=False):
        encoded_outputs = self.bert(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)
        encoded_layer = encoded_outputs[0]

        cls_states = encoded_layer[:, 0]
        blank_states = encoded_layer[[i for i in range(len(positions))], positions]  # [batch, hidden_state]
        over_states = self.over_linear(torch.cat([blank_states,
                                                  cls_states,
                                                  blank_states * cls_states,
                                                  blank_states - cls_states], dim=-1))

        over_logits = self.vocab(over_states)

        if compute_loss:
            target = torch.gather(option_ids, dim=1, index=targets.unsqueeze(1))

            contrastive_loss_fct = ContrastiveLoss(tau=0.5)
            targets_embeddings = self.idiom_embedding(target.squeeze(1))
            closs = contrastive_loss_fct(self.projection(over_states), self.projection(targets_embeddings))

            loss_fct = nn.CrossEntropyLoss()
            over_loss = loss_fct(over_logits, target.squeeze(1))
            return over_loss + closs, over_logits
        else:
            cond_logits = torch.gather(over_logits, dim=1, index=option_ids)
            return cond_logits, over_logits


class BertContrastiveSingle(BertPreTrainedModel):
    def __init__(self, config, len_idiom_vocab, model_name='bertsingle'):
        super().__init__(config)
        self.model_name = model_name
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        emb_hidden_size = config.hidden_size
        self.idiom_embedding = nn.Embedding(len_idiom_vocab, emb_hidden_size)
        self.register_buffer('enlarged_candidates', torch.arange(len_idiom_vocab))

        # projection MLP
        self.projection = nn.Sequential(nn.Linear(config.hidden_size, 512, bias=False),
                                        nn.LayerNorm(512),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(512, 100, bias=True))

        self.init_weights()

    def vocab(self, blank_states):
        idiom_embeddings = self.idiom_embedding(self.enlarged_candidates)
        return torch.einsum('bd,nd->bn', [blank_states, idiom_embeddings])  # (b, 256, 10)

    def forward(self, input_ids, token_type_ids, attention_mask, positions, option_ids,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        # batch_size, sequence_num, length = input_ids.shape
        encoded_outputs = self.bert(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    inputs_embeds=inputs_embeds)
        encoded_layer = encoded_outputs[0]

        encoded_context = encoded_layer
        blank_states = encoded_context[[i for i in range(len(positions))], positions]  # [batch, hidden_state]
        # cls_states = encoded_layer[:, 0]

        if option_ids is None and options_embeds is None:
            raise ValueError('Either option_ids or options_embeds should be given.')
        elif options_embeds is not None:
            encoded_options = options_embeds
        else:
            encoded_options = self.idiom_embedding(option_ids)  # (b, 10, 768)

        over_logits = self.vocab(blank_states)
        # cond_logits = torch.gather(over_logits, dim=1, index=option_ids)

        mo_logits = torch.einsum('bld,bnd->bln', [encoded_context, encoded_options])  # (b, 256, 10)
        c_mo_logits, _ = torch.max(mo_logits, dim=1)
        # over_states = cls_states

        logits = c_mo_logits

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, targets)

            target = torch.gather(option_ids, dim=1, index=targets.unsqueeze(1))

            contrastive_loss_fct = ContrastiveLoss(tau=0.5)
            targets_embeddings = self.idiom_embedding(target.squeeze(1))
            closs = contrastive_loss_fct(self.projection(blank_states), self.projection(targets_embeddings))

            over_loss = loss_fct(over_logits, target.squeeze(1))
            return loss + over_loss + closs, over_logits
        else:
            return logits, over_logits


class BertContrastiveDual(BertPreTrainedModel):
    def __init__(self, config, len_idiom_vocab, model_name='bertdual'):
        super().__init__(config)
        self.model_name = model_name
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        emb_hidden_size = config.hidden_size
        self.idiom_facial_embedding = nn.Embedding(len_idiom_vocab, emb_hidden_size)
        self.idiom_meaning_embedding = nn.Embedding(len_idiom_vocab, emb_hidden_size)

        self.init_weights()

    def vocab(self, blank_states):
        c_fo_logits = torch.einsum('bd,nd->bn', [blank_states, self.idiom_facial_embedding.weight])  # (b, 256, 10)
        c_mo_logits = torch.einsum('bd,nd->bn', [blank_states, self.idiom_meaning_embedding.weight])  # (b, 256, 10)
        return c_mo_logits + c_fo_logits

    def forward(self, input_ids, token_type_ids, attention_mask, positions, option_ids,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        # batch_size, sequence_num, length = input_ids.shape
        encoded_outputs = self.bert(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    inputs_embeds=inputs_embeds)
        encoded_layer = encoded_outputs[0]

        encoded_context = encoded_layer
        blank_states = encoded_context[[i for i in range(len(positions))], positions]  # [batch, hidden_state]
        # cls_states = encoded_layer[:, 0]

        if option_ids is None and options_embeds is None:
            raise ValueError('Either option_ids or options_embeds should be given.')
        elif options_embeds is not None:
            facial_state, meaning_state = options_embeds
        else:
            facial_state = self.idiom_facial_embedding(option_ids)  # (b, 10, 768)
            meaning_state = self.idiom_meaning_embedding(option_ids)  # (b, 10, 768)

        over_logits = self.vocab(blank_states)
        # cond_logits = torch.gather(over_logits, dim=1, index=option_ids)

        mo_logits = torch.einsum('bld,bnd->bln', [encoded_context, meaning_state])  # (b, 256, 10)
        c_mo_logits, _ = torch.max(mo_logits, dim=1)
        # over_states = cls_states

        c_fo_logits = torch.einsum('bd,bnd->bn', [blank_states, facial_state])  # (b, 10)
        logits = c_mo_logits + c_fo_logits

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, targets)

            target = torch.gather(option_ids, dim=1, index=targets.unsqueeze(1))
            contrastive_loss_fct = ContrastiveLoss(tau=0.5)
            targets_embeddings = self.idiom_embedding(target.squeeze(1))
            closs = contrastive_loss_fct(self.projection(over_states), self.projection(targets_embeddings))

            over_loss = loss_fct(over_logits, target.squeeze(1))
            return loss, over_loss
        else:
            return logits, over_logits
