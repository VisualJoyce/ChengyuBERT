from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

from chengyubert.models import register_model


@register_model('chengyubert-2stage-stage1')
class ChengyuBertTwoStagePretrain(BertPreTrainedModel):
    def __init__(self, config, len_idiom_vocab, model_name='chengyubert-2stage'):
        super().__init__(config)
        self.model_name = model_name
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.over_linear = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.idiom_embedding = nn.Embedding(len_idiom_vocab, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.register_buffer('enlarged_candidates', torch.arange(len_idiom_vocab))
        self.init_weights()

    def vocab(self, blank_states):
        idiom_embeddings = self.LayerNorm(self.idiom_embedding(self.enlarged_candidates))
        return torch.einsum('bd,nd->bn', [blank_states, idiom_embeddings])  # (b, 256, 10)

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

        over_logits = self.vocab(over_states)

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            target = torch.gather(option_ids, dim=1, index=targets.unsqueeze(1))
            over_loss = loss_fct(over_logits, target.squeeze(1))
            return over_loss
        else:
            cond_logits = torch.gather(over_logits, dim=1, index=option_ids)
            return cond_logits, over_logits


@register_model('chengyubert-2stage-stage1-mask')
class ChengyuBertTwoStageMaskPretrain(BertPreTrainedModel):
    def __init__(self, config, len_idiom_vocab, model_name):
        super().__init__(config)
        self.model_name = model_name
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.idiom_embedding = nn.Embedding(len_idiom_vocab, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.register_buffer('enlarged_candidates', torch.arange(len_idiom_vocab))
        self.init_weights()

    def vocab(self, blank_states):
        idiom_embeddings = self.LayerNorm(self.idiom_embedding(self.enlarged_candidates))
        return torch.einsum('bd,nd->bn', [blank_states, idiom_embeddings])  # (b, 256, 10)

    def forward(self, input_ids, token_type_ids, attention_mask, positions, option_ids,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        encoded_outputs = self.bert(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    inputs_embeds=inputs_embeds)
        encoded_layer = encoded_outputs[0]
        blank_states = encoded_layer[[i for i in range(len(positions))], positions]  # [batch, hidden_state]

        over_logits = self.vocab(blank_states)

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            target = torch.gather(option_ids, dim=1, index=targets.unsqueeze(1))
            over_loss = loss_fct(over_logits, target.squeeze(1))
            return over_loss
        else:
            cond_logits = torch.gather(over_logits, dim=1, index=option_ids)
            return cond_logits, over_logits


@register_model('chengyubert-2stage-stage1-cls')
class ChengyuBertTwoStageCLSPretrain(BertPreTrainedModel):
    def __init__(self, config, len_idiom_vocab, model_name='chengyubert-2stage'):
        super().__init__(config)
        self.model_name = model_name
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.over_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.idiom_embedding = nn.Embedding(len_idiom_vocab, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.register_buffer('enlarged_candidates', torch.arange(len_idiom_vocab))
        self.init_weights()

    def vocab(self, blank_states):
        idiom_embeddings = self.LayerNorm(self.idiom_embedding(self.enlarged_candidates))
        return torch.einsum('bd,nd->bn', [blank_states, idiom_embeddings])  # (b, 256, 10)

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
                                                  cls_states], dim=-1))

        over_logits = self.vocab(over_states)

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            target = torch.gather(option_ids, dim=1, index=targets.unsqueeze(1))
            over_loss = loss_fct(over_logits, target.squeeze(1))
            return over_loss
        else:
            cond_logits = torch.gather(over_logits, dim=1, index=option_ids)
            return cond_logits, over_logits


@register_model('chengyubert-2stage-stage2')
class ChengyuBertTwoStageFinetune(BertPreTrainedModel):

    def __init__(self, config, len_idiom_vocab, model_name):
        super().__init__(config)
        self.model_name = model_name
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.over_linear = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.register_buffer('enlarged_candidates', torch.arange(len_idiom_vocab))
        self.idiom_embedding = nn.Embedding(len_idiom_vocab, config.hidden_size)
        self.init_weights()

    def vocab(self, blank_states):
        idiom_embeddings = self.idiom_embedding(self.enlarged_candidates)
        return torch.einsum('bd,nd->bn', [blank_states, idiom_embeddings])  # (b, 256, 10)

    def forward(self, input_ids, token_type_ids, attention_mask, positions, option_ids,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        encoded_outputs = self.bert(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    inputs_embeds=inputs_embeds)
        encoded_layer = encoded_outputs[0]
        blank_states = encoded_layer[[i for i in range(len(positions))], positions]  # [batch, hidden_state]
        cls_states = encoded_layer[:, 0]

        if option_ids is None and options_embeds is None:
            raise ValueError('Either option_ids or options_embeds should be given.')
        elif options_embeds is not None:
            encoded_options = options_embeds
        else:
            encoded_options = self.idiom_embedding(option_ids)

        over_states = self.over_linear(torch.cat([blank_states,
                                                  cls_states,
                                                  blank_states * cls_states,
                                                  blank_states - cls_states], dim=-1))

        over_logits = self.vocab(over_states)
        cond_logits = torch.gather(over_logits, dim=1, index=option_ids)

        # encoded_context = encoded_layer
        # mo_logits = torch.einsum('bld,bnd->bln', [encoded_context, encoded_options])  # (b, 256, 10)
        # logits, _ = torch.max(mo_logits, dim=1)

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(cond_logits, targets)
            target = torch.gather(option_ids, dim=1, index=targets.unsqueeze(1))
            over_loss = loss_fct(over_logits, target.squeeze(1))
            return loss, over_loss
        else:
            return cond_logits, over_logits, cond_logits


@register_model('chengyubert-2stage-stage2-mask')
class ChengyuBertTwoStageMaskFinetune(BertPreTrainedModel):

    def __init__(self, config, len_idiom_vocab, model_name):
        super().__init__(config)
        self.model_name = model_name
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer('enlarged_candidates', torch.arange(len_idiom_vocab))
        self.idiom_embedding = nn.Embedding(len_idiom_vocab, config.hidden_size)
        self.init_weights()

    def vocab(self, blank_states):
        idiom_embeddings = self.idiom_embedding(self.enlarged_candidates)
        return torch.einsum('bd,nd->bn', [blank_states, idiom_embeddings])  # (b, 256, 10)

    def forward(self, input_ids, token_type_ids, attention_mask, positions, option_ids,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        encoded_outputs = self.bert(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    inputs_embeds=inputs_embeds)
        encoded_layer = encoded_outputs[0]
        blank_states = encoded_layer[[i for i in range(len(positions))], positions]  # [batch, hidden_state]
        cls_states = encoded_layer[:, 0]

        if option_ids is None and options_embeds is None:
            raise ValueError('Either option_ids or options_embeds should be given.')
        elif options_embeds is not None:
            encoded_options = options_embeds
        else:
            encoded_options = self.idiom_embedding(option_ids)

        over_logits = self.vocab(blank_states)
        cond_logits = torch.gather(over_logits, dim=1, index=option_ids)

        # encoded_context = encoded_layer
        # mo_logits = torch.einsum('bld,bnd->bln', [encoded_context, encoded_options])  # (b, 256, 10)
        # logits, _ = torch.max(mo_logits, dim=1)

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(cond_logits, targets)
            target = torch.gather(option_ids, dim=1, index=targets.unsqueeze(1))
            over_loss = loss_fct(over_logits, target.squeeze(1))
            return loss, over_loss
        else:
            return cond_logits, over_logits, cond_logits


@register_model('chengyubert-2stage-stage2-cls')
class ChengyuBertTwoStageCLSFinetune(BertPreTrainedModel):

    def __init__(self, config, len_idiom_vocab, model_name):
        super().__init__(config)
        self.model_name = model_name
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.over_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.register_buffer('enlarged_candidates', torch.arange(len_idiom_vocab))
        self.idiom_embedding = nn.Embedding(len_idiom_vocab, config.hidden_size)
        self.init_weights()

    def vocab(self, blank_states):
        idiom_embeddings = self.idiom_embedding(self.enlarged_candidates)
        return torch.einsum('bd,nd->bn', [blank_states, idiom_embeddings])  # (b, 256, 10)

    def forward(self, input_ids, token_type_ids, attention_mask, positions, option_ids,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        encoded_outputs = self.bert(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    inputs_embeds=inputs_embeds)
        encoded_layer = encoded_outputs[0]
        blank_states = encoded_layer[[i for i in range(len(positions))], positions]  # [batch, hidden_state]
        cls_states = encoded_layer[:, 0]

        if option_ids is None and options_embeds is None:
            raise ValueError('Either option_ids or options_embeds should be given.')
        elif options_embeds is not None:
            encoded_options = options_embeds
        else:
            encoded_options = self.idiom_embedding(option_ids)

        over_states = self.over_linear(torch.cat([blank_states,
                                                  cls_states], dim=-1))

        over_logits = self.vocab(over_states)
        cond_logits = torch.gather(over_logits, dim=1, index=option_ids)

        # encoded_context = encoded_layer
        # mo_logits = torch.einsum('bld,bnd->bln', [encoded_context, encoded_options])  # (b, 256, 10)
        # logits, _ = torch.max(mo_logits, dim=1)

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(cond_logits, targets)
            target = torch.gather(option_ids, dim=1, index=targets.unsqueeze(1))
            over_loss = loss_fct(over_logits, target.squeeze(1))
            return loss, over_loss
        else:
            return cond_logits, over_logits, cond_logits


@register_model('chengyubert-2stage-stage2-window')
class ChengyuBertTwoStageWindow(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config, len_idiom_vocab, model_name):
        super().__init__(config)
        self.model_name = model_name
        self.window_size = int(self.model_name.split('-')[-1])
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.over_linear = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.register_buffer('enlarged_candidates', torch.arange(len_idiom_vocab))
        self.idiom_embedding = nn.Embedding(len_idiom_vocab, config.hidden_size)
        self.init_weights()

    def vocab(self, over_states):
        idiom_embeddings = self.idiom_embedding(self.enlarged_candidates)
        c_mo_logits = torch.einsum('bd,nd->bn', [over_states, idiom_embeddings])  # (b, 256, 10)
        return c_mo_logits

    def forward(self, input_ids, token_type_ids, attention_mask, positions, option_ids,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        batch_size, length = input_ids.size()
        encoded_outputs = self.bert(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    inputs_embeds=inputs_embeds)
        encoded_layer = encoded_outputs[0]
        blank_states = encoded_layer[[i for i in range(len(positions))], positions]  # [batch, hidden_state]
        cls_states = encoded_layer[:, 0]

        if option_ids is None and options_embeds is None:
            raise ValueError('Either option_ids or options_embeds should be given.')
        elif options_embeds is not None:
            encoded_options = options_embeds
        else:
            encoded_options = self.idiom_embedding(option_ids)

        over_states = self.over_linear(torch.cat([blank_states,
                                                  cls_states,
                                                  blank_states * cls_states,
                                                  blank_states - cls_states], dim=-1))

        over_logits = self.vocab(over_states)

        encoded_context = encoded_layer
        mo_logits = torch.einsum('bld,bnd->bln', [encoded_context, encoded_options])  # (b, 256, 10)

        if self.window_size > length:
            logits, _ = torch.max(mo_logits, dim=1)
        elif self.window_size == 0:
            new_logits = []
            for i, p in enumerate(positions):
                new_logits.append(mo_logits[i, p])
            logits = torch.stack(new_logits, dim=0)
        else:
            window_size = self.window_size
            new_logits = []
            for i, p in enumerate(positions):
                if p >= window_size and p + window_size >= length:
                    new_logits.append(torch.max(mo_logits[i, p - window_size:], dim=0)[0])
                elif p >= window_size and p + window_size < length:
                    new_logits.append(torch.max(mo_logits[i, (p - window_size): (p + window_size) + 1], dim=0)[0])
                elif p < window_size:
                    new_logits.append(torch.max(mo_logits[i, : (p + window_size) + 1], dim=0)[0])
            logits = torch.stack(new_logits, dim=0)

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, targets)
            target = torch.gather(option_ids, dim=1, index=targets.unsqueeze(1))
            over_loss = loss_fct(over_logits, target.squeeze(1))
            return loss, over_loss
        else:
            cond_logits = torch.gather(over_logits, dim=1, index=option_ids)
            return logits, over_logits, cond_logits


@register_model('chengyubert-2stage-stage2-mask-window')
class ChengyuBertTwoStageMaskWindow(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """

    def __init__(self, config, len_idiom_vocab, model_name):
        super().__init__(config)
        self.model_name = model_name
        self.window_size = int(self.model_name.split('-')[-1])
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.register_buffer('enlarged_candidates', torch.arange(len_idiom_vocab))
        self.idiom_embedding = nn.Embedding(len_idiom_vocab, config.hidden_size)
        self.init_weights()

    def vocab(self, over_states):
        idiom_embeddings = self.idiom_embedding(self.enlarged_candidates)
        c_mo_logits = torch.einsum('bd,nd->bn', [over_states, idiom_embeddings])  # (b, 256, 10)
        return c_mo_logits

    def forward(self, input_ids, token_type_ids, attention_mask, positions, option_ids,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        batch_size, length = input_ids.size()
        encoded_outputs = self.bert(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    inputs_embeds=inputs_embeds)
        encoded_layer = encoded_outputs[0]
        blank_states = encoded_layer[[i for i in range(len(positions))], positions]  # [batch, hidden_state]

        if option_ids is None and options_embeds is None:
            raise ValueError('Either option_ids or options_embeds should be given.')
        elif options_embeds is not None:
            encoded_options = options_embeds
        else:
            encoded_options = self.idiom_embedding(option_ids)

        over_logits = self.vocab(blank_states)

        encoded_context = encoded_layer
        mo_logits = torch.einsum('bld,bnd->bln', [encoded_context, encoded_options])  # (b, 256, 10)

        if self.window_size > length:
            logits, _ = torch.max(mo_logits, dim=1)
        elif self.window_size == 0:
            new_logits = []
            for i, p in enumerate(positions):
                new_logits.append(mo_logits[i, p])
            logits = torch.stack(new_logits, dim=0)
        else:
            window_size = self.window_size
            new_logits = []
            for i, p in enumerate(positions):
                if p >= window_size and p + window_size >= length:
                    new_logits.append(torch.max(mo_logits[i, p - window_size:], dim=0)[0])
                elif p >= window_size and p + window_size < length:
                    new_logits.append(torch.max(mo_logits[i, (p - window_size): (p + window_size) + 1], dim=0)[0])
                elif p < window_size:
                    new_logits.append(torch.max(mo_logits[i, : (p + window_size) + 1], dim=0)[0])
            logits = torch.stack(new_logits, dim=0)

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, targets)
            target = torch.gather(option_ids, dim=1, index=targets.unsqueeze(1))
            over_loss = loss_fct(over_logits, target.squeeze(1))
            return loss, over_loss
        else:
            cond_logits = torch.gather(over_logits, dim=1, index=option_ids)
            return logits, over_logits, cond_logits
