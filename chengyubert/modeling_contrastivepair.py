from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

from chengyubert.loss import ContrastiveLoss


class BertContrastivePairSingle(BertPreTrainedModel):
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
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None, contrastive=True):
        # batch_size, sequence_num, length = input_ids.shape
        encoded_outputs = self.bert(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    inputs_embeds=inputs_embeds)
        encoded_layer = encoded_outputs[0]

        encoded_context = encoded_layer
        blank_states = encoded_context[[i for i in range(len(positions))], positions]  # [batch, hidden_state]
        # cls_states = encoded_layer[:, 0]

        augmentation_0, augmentation_1 = blank_states.view(-1, 2, 10).chunk(2, dim=1)

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
            closs = contrastive_loss_fct(self.projection(augmentation_0),
                                         self.projection(augmentation_1))

            over_loss = loss_fct(over_logits, target.squeeze(1))
            return loss + over_loss + closs, over_logits
        else:
            return logits, over_logits