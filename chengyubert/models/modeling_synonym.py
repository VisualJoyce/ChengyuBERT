from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

from chengyubert.models import register_model


@register_model('chengyubert-emb')
class ChengyuBertEmb(BertPreTrainedModel):
    def __init__(self, config, opts):
        super().__init__(config)
        self.model_name = opts.model
        chengyu_emb_dim = int(self.model_name.split('-')[-1])
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.idiom_embedding = nn.Embedding(opts.len_idiom_vocab, chengyu_emb_dim)
        self.project_linear = nn.Linear(config.hidden_size, chengyu_emb_dim)
        self.register_buffer('enlarged_candidates', torch.arange(opts.len_idiom_vocab))
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


@register_model('chengyubert-ns')
class ChengyuBertSGNS(BertPreTrainedModel):
    def __init__(self, config, opts):
        super().__init__(config)
        assert opts.model.startswith(('chengyubert-ns-mask',
                                      'chengyubert-ns-cls-mask',
                                      'chengyubert-ns-element-wise',
                                      ))
        self.model_name = opts.model
        chengyu_emb_dim = int(self.model_name.split('-')[-1])

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.idiom_embedding = nn.Embedding(opts.len_idiom_vocab, chengyu_emb_dim)
        self.LayerNorm = nn.LayerNorm(chengyu_emb_dim, eps=config.layer_norm_eps)
        if self.model_name.startswith('chengyubert-ns-mask'):
            self.project_linear = nn.Linear(config.hidden_size, chengyu_emb_dim)
        elif self.model_name.startswith('chengyubert-ns-cls-mask'):
            self.project_linear = nn.Linear(config.hidden_size * 2, chengyu_emb_dim)
        else:
            self.project_linear = nn.Linear(config.hidden_size * 4, chengyu_emb_dim)
        self.register_buffer('enlarged_candidates', torch.arange(opts.len_idiom_vocab))
        self.init_weights()

    def project(self, cls_states, blank_states):
        if self.model_name.startswith('chengyubert-ns-mask'):
            return self.project_linear(blank_states)
        elif self.model_name.startswith('chengyubert-ns-cls-mask'):
            return self.project_linear(torch.cat([blank_states, cls_states], dim=-1))
        else:
            return self.project_linear(torch.cat([blank_states,
                                                  cls_states,
                                                  blank_states * cls_states,
                                                  blank_states - cls_states], dim=-1))

    def vocab(self, over_states):
        idiom_embeddings = self.LayerNorm(self.idiom_embedding(self.enlarged_candidates))
        c_mo_logits = torch.einsum('bd,nd->bn', [over_states, idiom_embeddings])  # (b, 256, 10)
        return c_mo_logits

    def forward(self, input_ids, token_type_ids, attention_mask, positions, option_ids,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        encoded_outputs = self.bert(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    inputs_embeds=inputs_embeds)
        encoded_layer = encoded_outputs[0]
        cls_states = encoded_layer[:, 0]
        blank_states = encoded_layer[[i for i in range(len(positions))], positions]  # [batch, hidden_state]
        emb_u = self.project(cls_states, blank_states)

        if compute_loss:
            target = torch.gather(option_ids, dim=1, index=targets.unsqueeze(1))
            emb_v = self.LayerNorm(self.idiom_embedding(target.squeeze(1)))

            bs, num = option_ids.size()

            negative_samples = torch.masked_select(option_ids,
                                                   option_ids != target.repeat([1, num])).view(bs, -1)
            emb_neg_v = self.LayerNorm(self.idiom_embedding(negative_samples))

            score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
            # score = torch.clamp(score, max=10, min=-10)
            score = -nn.functional.logsigmoid(score)

            neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
            # neg_score = torch.clamp(neg_score, max=10, min=-10)
            neg_score = -torch.sum(nn.functional.logsigmoid(-neg_score), dim=1)

            return torch.mean(score + neg_score)
        else:
            over_logits = self.vocab(emb_u)
            cond_logits = torch.gather(over_logits, dim=1, index=option_ids)
            return cond_logits, over_logits
