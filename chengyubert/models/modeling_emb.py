from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

from chengyubert.optim.loss import ContrastiveLoss


class ChengyuBertEmb(BertPreTrainedModel):
    def __init__(self, config, len_idiom_vocab, model_name='chengyubert-emb-300'):
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


class ChengyuBertSGNS(BertPreTrainedModel):
    def __init__(self, config, len_idiom_vocab, model_name):
        super().__init__(config)
        assert model_name.startswith(('chengyubert-mask-ns', 'chengyubert-cls-ns'))
        self.model_name = model_name
        chengyu_emb_dim = int(model_name.split('-')[-1])

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.idiom_embedding = nn.Embedding(len_idiom_vocab, chengyu_emb_dim)
        self.LayerNorm = nn.LayerNorm(chengyu_emb_dim, eps=config.layer_norm_eps)
        if self.model_name.startswith('chengyubert-mask-ns'):
            self.project_linear = nn.Linear(config.hidden_size, chengyu_emb_dim)
        else:
            self.project_linear = nn.Linear(config.hidden_size * 4, chengyu_emb_dim)
        self.register_buffer('enlarged_candidates', torch.arange(len_idiom_vocab))
        self.init_weights()

    def project(self, cls_states, blank_states):
        if self.model_name.startswith('chengyubert-mask-ns'):
            return self.project_linear(blank_states)
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


class ChengyuBertContrastive(BertPreTrainedModel):
    def __init__(self, config, len_idiom_vocab, model_name='bertsingle'):
        super().__init__(config)
        assert model_name.startswith(('chengyubert-mask-contrastive', 'chengyubert-cls-contrastive'))
        self.model_name = model_name
        chengyu_emb_dim = int(model_name.split('-')[-2])
        contrastive_dim = int(model_name.split('-')[-1])

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.idiom_embedding = nn.Embedding(len_idiom_vocab, chengyu_emb_dim)
        self.LayerNorm = nn.LayerNorm(chengyu_emb_dim, eps=config.layer_norm_eps)
        self.register_buffer('enlarged_candidates', torch.arange(len_idiom_vocab))

        # projection MLP
        if self.model_name.startswith('chengyubert-mask-contrastive'):
            self.project_linear = nn.Linear(config.hidden_size, chengyu_emb_dim)
        else:
            self.project_linear = nn.Linear(config.hidden_size * 4, chengyu_emb_dim)

        # projection MLP
        self.projection = nn.Sequential(nn.Linear(chengyu_emb_dim, chengyu_emb_dim, bias=False),
                                        nn.LayerNorm(chengyu_emb_dim),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(chengyu_emb_dim, contrastive_dim, bias=True))

        self.init_weights()

    def project(self, cls_states, blank_states):
        if self.model_name.startswith('chengyubert-mask-contrastive'):
            return self.project_linear(blank_states)
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
        # batch_size, sequence_num, length = input_ids.shape
        encoded_outputs = self.bert(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    inputs_embeds=inputs_embeds)
        encoded_layer = encoded_outputs[0]

        encoded_context = encoded_layer
        blank_states = encoded_context[[i for i in range(len(positions))], positions]  # [batch, hidden_state]
        cls_states = encoded_layer[:, 0]

        emb_u = self.project(cls_states, blank_states)

        if compute_loss:
            target_ids = torch.gather(option_ids, dim=1, index=targets.unsqueeze(1)).squeeze(1)
            emb_v = self.LayerNorm(self.idiom_embedding(target_ids))  # (b, 768)
            contrastive_loss_fct = ContrastiveLoss(tau=1)
            return contrastive_loss_fct(self.projection(emb_u), self.projection(emb_v))
        else:
            over_logits = self.vocab(emb_u)
            cond_logits = torch.gather(over_logits, dim=1, index=option_ids)
            return cond_logits, over_logits
