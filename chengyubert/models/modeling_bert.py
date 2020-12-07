from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

from chengyubert.models import register_model


@register_model('chengyubert-cloze')
class ChengyuBertForClozeChid(BertPreTrainedModel):
    """BERT model for multiple choice tasks.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_choices`: the number of classes for the classifier. Default = 2.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `option_ids`: a torch.LongTensor of shape [batch_size, num_choices]
        `positions`: a torch.LongTensor of shape [batch_size]
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].
    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].
    """

    def __init__(self, config, len_idiom_vocab, model_name='bertchid'):
        super().__init__(config)
        self.model_name = model_name
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.idiom_embedding = nn.Embedding(len_idiom_vocab, config.hidden_size)
        self.init_weights()

    def vocab(self, blank_states):
        return torch.einsum('bd,nd->bn', [blank_states, self.idiom_embedding.weight])  # (b, 256, 10)

    def forward(self, input_ids, token_type_ids, attention_mask, positions, option_ids,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):

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

        multiply_result = torch.einsum('abc,ac->abc', encoded_options, blank_states)

        over_logits = self.vocab(blank_states)

        pooled_output = self.dropout(multiply_result)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(len(positions), -1)

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, targets)
            target = torch.gather(option_ids, dim=1, index=targets.unsqueeze(1))
            over_loss = loss_fct(over_logits, target.squeeze(1))
            return loss, over_loss
        else:
            return reshaped_logits, over_logits


@register_model('chengyubert-choice')
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

        over_states = torch.cat([idiom_states,
                                 cls_states,
                                 idiom_states * cls_states,
                                 idiom_states - cls_states], dim=-1)

        extra_logits = {}
        for k in self.used_paras:
            extra_logits[k] = getattr(self, 'idiom_{}'.format(k))(over_states)
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


class BertForPretrain(BertPreTrainedModel):
    def __init__(self, config, len_idiom_vocab, model_name='bert-dual'):
        super().__init__(config)
        self.model_name = model_name
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.idiom_embedding = nn.Embedding(len_idiom_vocab, config.hidden_size)
        self.init_weights()

    def vocab(self, over_states):
        c_mo_logits = torch.einsum('bd,nd->bn', [over_states, self.idiom_embedding.weight])  # (b, 256, 10)
        return c_mo_logits

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