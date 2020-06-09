from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class ChengyuBert(BertPreTrainedModel):
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

    def __init__(self, config, len_idiom_vocab, model_name='chengyubert'):
        super().__init__(config)
        self.model_name = model_name
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.over_linear = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.idiom_embedding = nn.Embedding(len_idiom_vocab, config.hidden_size)
        self.init_weights()

    def vocab(self, over_states):
        c_mo_logits = torch.einsum('bd,nd->bn', [self.over_linear(over_states),
                                                 self.idiom_embedding.weight.half() if self.use_fp16 else self.idiom_embedding.weight])  # (b, 256, 10)
        return c_mo_logits

    def forward(self, input_ids, attention_mask, segment_ids, extra_ids, positions, option_ids):
        batch_size, sequence_num, length = input_ids.shape
        extra_ids = {k: extra_ids[k].view(-1, length) for k in extra_ids}
        encoded_outputs = self.bert(input_ids.view(-1, length),
                                    token_type_ids=segment_ids.view(-1, length),
                                    attention_mask=attention_mask.view(-1, length))
        encoded_layer = encoded_outputs[0]
        blank_states = encoded_layer[[i for i in range(len(positions))], positions]  # [batch, hidden_state]
        cls_states = encoded_layer[:, 0]
        encoded_options = self.idiom_embedding(option_ids)

        sentiment_states = torch.cat([blank_states,
                                      cls_states,
                                      blank_states * cls_states,
                                      blank_states - cls_states], dim=-1)

        over_logits = self.vocab(sentiment_states)
        cond_logits = torch.gather(over_logits, dim=1, index=option_ids)

        extra_logits = {}
        for k in self.used_paras:
            extra_logits[k] = getattr(self, 'idiom_{}'.format(k))(sentiment_states)

        if self.use_sequence:
            encoded_context = encoded_layer
            mo_logits = torch.einsum('bld,bnd->bln', [encoded_context, encoded_options])  # (b, 256, 10)
            logits, _ = torch.max(mo_logits, dim=1)
            return logits + cond_logits, over_logits, cond_logits, None, extra_logits
        else:
            return cond_logits, over_logits, cond_logits, None, extra_logits


class BertForClozeChoice(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """

    def __init__(self, config, args):
        super().__init__(config)
        self.use_kld = args.use_kld

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids, option_ids, token_type_ids, attention_mask, positions, idiom_ids,
                labels=None):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=None,
                            head_mask=None)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        return reshaped_logits, None, None, None


class BertIdiomEmotion(BertPreTrainedModel):
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

    def __init__(self, config, args):
        super(BertIdiomEmotion, self).__init__(config)
        self.use_fp16 = args.fp16
        self.use_sentiment2 = args.use_sentiment2
        self.use_sentiment7 = args.use_sentiment7
        self.use_sentiment21 = args.use_sentiment21
        self.use_polarity = args.use_polarity

        self.bert = BertModel(config, args)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.used_paras = []
        for k, d in args.sentiments:
            if hasattr(args, 'use_{}'.format(k)) and getattr(args, 'use_{}'.format(k)):
                setattr(self, 'idiom_{}'.format(k), nn.Linear(config.hidden_size, d))
                self.used_paras.append(k)
        self.init_weights()

    def forward(self, input_ids, attention_mask, segment_ids, idioms):
        encoded_outputs = self.bert(input_ids,
                                    token_type_ids=segment_ids,
                                    attention_mask=attention_mask,
                                    extra_embeddings=None)
        encoded_layer = encoded_outputs[0]
        cls_states = encoded_layer[:, 0]
        extra_logits = {}
        for k in self.used_paras:
            extra_logits[k] = getattr(self, 'idiom_{}'.format(k))(cls_states)
        return extra_logits


class IBertIdiomEmotion(BertPreTrainedModel):
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

    def __init__(self, config, args):
        super(IBertIdiomEmotion, self).__init__(config)
        self.use_fp16 = args.fp16
        self.use_sentiment2 = args.use_sentiment2
        self.use_sentiment7 = args.use_sentiment7
        self.use_sentiment21 = args.use_sentiment21
        self.use_polarity = args.use_polarity

        self.bert = BertModel(config, args)
        self.idiom_embedding = nn.Embedding(args.len_idiom_vocab, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.init_weights()

    def forward(self, input_ids, attention_mask, segment_ids, idiom_ids):
        encoded_outputs = self.bert(input_ids,
                                    token_type_ids=segment_ids,
                                    attention_mask=attention_mask,
                                    extra_embeddings=None)
        encoded_layer = encoded_outputs[0]
        cls_states = encoded_layer[:, 0]
        idiom_states = self.idiom_embedding(idiom_ids)

        sentiment_states = torch.cat([idiom_states,
                                      cls_states,
                                      idiom_states * cls_states,
                                      idiom_states - cls_states], dim=-1)

        extra_logits = {}
        for k in self.used_paras:
            extra_logits[k] = getattr(self, 'idiom_{}'.format(k))(sentiment_states)
        return extra_logits


class IdiomEmbeddingEmotion(BertPreTrainedModel):

    def __init__(self, config, args):
        super().__init__(config)
        self.bert = BertModelPlus(config, args)
        self.idiom_embedding = nn.Embedding(args.len_idiom_vocab, config.hidden_size)
        self.static_idiom_embedding = nn.Embedding.from_pretrained(self.idiom_embedding.weight, freeze=True)
        self.used_paras = []
        for k, d in args.sentiments:
            if hasattr(args, 'use_{}'.format(k)) and getattr(args, 'use_{}'.format(k)):
                setattr(self, 'idiom_{}'.format(k), nn.Linear(config.hidden_size, d))
                self.used_paras.append(k)
        self.init_weights()

    def forward(self, input_ids, attention_mask, segment_ids, idiom_ids):
        cls_states = self.static_idiom_embedding(idiom_ids)
        extra_logits = {}
        for k in self.used_paras:
            extra_logits[k] = getattr(self, 'idiom_{}'.format(k))(cls_states)
        return extra_logits
