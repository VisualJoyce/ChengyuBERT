from typing import Any, Tuple, Optional

import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel

from chengyubert.models import register_model
from chengyubert.modules.attention import ContrastiveCoAttention
from chengyubert.modules.utils import WeightNormClassifier, LatentComposition, sequence_mask
from chengyubert.optim.loss import FocalLoss


class CaloClassifier(nn.Module):

    def __init__(self, hidden_size, hidden_dropout_prob):
        super().__init__()
        # Emotion-7 Predictor
        self.fine_emotion_classifier = WeightNormClassifier(hidden_size,
                                                            21,
                                                            hidden_size,
                                                            hidden_dropout_prob)

        # Sentiment Predictor
        self.sentiment_classifier = WeightNormClassifier(hidden_size,
                                                         4,
                                                         hidden_size,
                                                         hidden_dropout_prob)

    def forward(self, emotion_state) -> Tuple[Any, Any]:
        # affection prediction
        fine_emotion_logits = self.fine_emotion_classifier(emotion_state)
        # coarse_emotion_logits = self.coarse_emotion_classifier(emotion_state)
        sentiment_logits = self.sentiment_classifier(emotion_state)
        return fine_emotion_logits, sentiment_logits


class SlideClassifier(nn.Module):

    def __init__(self, hidden_size, hidden_dropout_prob):
        super().__init__()
        self.sentiment_classifier = WeightNormClassifier(hidden_size,
                                                         3,
                                                         hidden_size,
                                                         hidden_dropout_prob)

    def forward(self, emotion_state) -> None:
        # slide prediction
        return self.sentiment_classifier(emotion_state)


classifiers = {
    'calo': CaloClassifier,
    'slide': SlideClassifier
}


class CaloLoss(nn.Module):

    def __init__(self, use_focal, weights):
        super().__init__()
        fine_emotion_weights, sentiment_weights = weights
        if use_focal:
            self.loss_fct = FocalLoss()
        else:
            self.fine_emotion_loss_fct = nn.CrossEntropyLoss(weight=fine_emotion_weights, reduction='none')
            self.sentiment_loss_fct = nn.CrossEntropyLoss(weight=sentiment_weights, reduction='none')

    def forward(self, logits, targets) -> Tuple[Optional[Any], Any, Any]:
        over_logits, (fine_emotion_logits, sentiment_logits) = logits
        if over_logits is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            over_loss = loss_fct(over_logits, targets[:, 0])
        else:
            over_loss = None
        fine_emotion_loss = self.fine_emotion_loss_fct(fine_emotion_logits, targets[:, 2])
        sentiment_loss = self.sentiment_loss_fct(sentiment_logits, targets[:, 3])
        return over_loss, (fine_emotion_loss, sentiment_loss)


class SlideLoss(nn.Module):

    def __init__(self, use_focal, weights):
        super().__init__()
        if use_focal:
            self.loss_fct = FocalLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss(weight=weights, reduction='none')

    def forward(self, logits, targets) -> Tuple[Any, Any]:
        over_logits, sentiment_logits = logits
        if over_logits is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            over_loss = loss_fct(over_logits, targets[:, 0])
        else:
            over_loss = None
        sentiment_emotion_loss = self.loss_fct(sentiment_logits, targets[:, 1])
        return over_loss, sentiment_emotion_loss


loss_calculators = {
    'calo': CaloLoss,
    'slide': SlideLoss
}


@register_model('affection-max-pooling')
class AffectionMaxPooling(BertPreTrainedModel):

    def __init__(self, config, opts):
        super().__init__(config)
        self.project = opts.project
        self.model_name = opts.model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.channel1_linear = nn.Linear(config.hidden_size, config.hidden_size)

        self.classifier = classifiers[self.project](config.hidden_size, config.hidden_dropout_prob)
        if self.project == 'calo':
            self.loss_fct = loss_calculators['calo'](opts.use_focal, (opts.fine_emotion_weights,
                                                                      opts.sentiment_weights))
        else:
            self.loss_fct = loss_calculators['slide'](opts.use_focal, opts.weights)
        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask, positions, option_ids, gather_index,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        # n, batch_size, seq_len = input_ids.size()
        encoded_outputs = self.bert(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)
        encoded_context = encoded_outputs[0]

        # idiom_length = (gather_index > 0).sum(1)
        gather_index = gather_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(input_ids)
        idiom_states = torch.gather(encoded_context, dim=1, index=gather_index)

        composed_states, _ = idiom_states.max(dim=1)

        emotion_state = self.channel1_linear(composed_states).tanh()

        # affection prediction
        logits = self.classifier(emotion_state)

        if compute_loss:
            _, losses = self.loss_fct([None, logits], targets)
            return None, None, None, losses
        else:
            return None, None, None, logits


@register_model('affection-max-pooling-masked')
class AffectionMaxPoolingMasked(BertPreTrainedModel):

    def __init__(self, config, opts):
        super().__init__(config)
        self.model_name = opts.model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.channel1_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.channel2_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.compose_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = classifiers[self.project](config.hidden_size, config.hidden_dropout_prob)
        if self.project == 'calo':
            self.loss_fct = loss_calculators['calo'](opts.use_focal, (opts.fine_emotion_weights,
                                                                      opts.sentiment_weights))
        else:
            self.loss_fct = loss_calculators['slide'](opts.use_focal, opts.weights)

        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask, positions, option_ids, gather_index,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        n, batch_size, seq_len = input_ids.size()
        encoded_outputs = self.bert(input_ids.view(n * batch_size, seq_len),
                                    token_type_ids=token_type_ids.view(n * batch_size, seq_len),
                                    attention_mask=attention_mask.view(n * batch_size, seq_len))
        encoded_context = encoded_outputs[0].view(n, batch_size, seq_len, -1)[0]
        encoded_context_masked = encoded_outputs[0].view(n, batch_size, seq_len, -1)[1]

        gather_index, gather_index_masked = gather_index
        idiom_length = (gather_index > 0).sum(1)

        gather_index_unsqueezed = gather_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(input_ids)
        idiom_states = torch.gather(encoded_context, dim=1, index=gather_index_unsqueezed)

        gather_index_masked_unsqueezed = gather_index_masked.unsqueeze(-1).expand(-1, -1,
                                                                                  self.config.hidden_size).type_as(
            input_ids)
        idiom_states_masked = torch.gather(encoded_context_masked, dim=1, index=gather_index_masked_unsqueezed)

        composed_states, _ = idiom_states.max(dim=1)
        composed_states_masked, _ = idiom_states_masked.max(dim=1)

        channel1 = self.channel1_linear(composed_states).tanh()
        channel2 = self.channel2_linear(composed_states_masked).tanh()

        # affection prediction
        emotion_state = self.compose_linear(torch.cat([channel1, channel2], dim=-1)).tanh()

        # affection prediction
        logits = self.classifier(emotion_state)

        if compute_loss:
            _, losses = self.loss_fct([None, logits], targets)
            return None, None, None, losses
        else:
            return None, None, None, logits


@register_model('affection-max-pooling-masked-latent-idiom')
class AffectionMaxPoolingMaskedLatentIdiom(BertPreTrainedModel):

    def __init__(self, config, opts):
        super().__init__(config)
        self.model_name = opts.model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if opts.enlarged_candidates is not None:
            self.register_buffer('enlarged_candidates', torch.tensor(opts.enlarged_candidates, dtype=torch.long))
        else:
            self.register_buffer('enlarged_candidates', torch.arange(opts.len_idiom_vocab))

        print(self.enlarged_candidates.size())

        self.idiom_embedding = nn.Embedding(opts.len_idiom_vocab, config.hidden_size)

        self.channel1_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.channel2_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.compose_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = classifiers[self.project](config.hidden_size, config.hidden_dropout_prob)
        if self.project == 'calo':
            self.loss_fct = loss_calculators['calo'](opts.use_focal, (opts.fine_emotion_weights,
                                                                      opts.sentiment_weights))
        else:
            self.loss_fct = loss_calculators['slide'](opts.use_focal, opts.weights)
        self.init_weights()

    def vocab(self, blank_states):
        idiom_embeddings = self.idiom_embedding(self.enlarged_candidates)
        logits = torch.einsum('bd,nd->bn', [blank_states, idiom_embeddings])  # (b, 256, 10)
        state = torch.einsum('bn,nd->bd', [logits.softmax(dim=-1), idiom_embeddings])  # (b, 256, 10)
        return logits, state

    def forward(self, input_ids, token_type_ids, attention_mask, positions, gather_index, option_ids=None,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        n, batch_size, seq_len = input_ids.size()
        encoded_outputs = self.bert(input_ids.view(n * batch_size, seq_len),
                                    token_type_ids=token_type_ids.view(n * batch_size, seq_len),
                                    attention_mask=attention_mask.view(n * batch_size, seq_len))
        encoded_context = encoded_outputs[0].view(n, batch_size, seq_len, -1)[0]
        encoded_context_masked = encoded_outputs[0].view(n, batch_size, seq_len, -1)[1]

        gather_index, gather_index_masked = gather_index
        gather_index = gather_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(input_ids)
        idiom_states = torch.gather(encoded_context, dim=1, index=gather_index)
        gather_index_masked_unsqueezed = gather_index_masked.unsqueeze(-1).expand(-1, -1,
                                                                                  self.config.hidden_size).type_as(
            input_ids)
        idiom_states_masked = torch.gather(encoded_context_masked, dim=1, index=gather_index_masked_unsqueezed)

        composed_states, _ = idiom_states.max(dim=1)
        composed_states_masked, _ = idiom_states_masked.max(dim=1)

        over_logits, idiom_attn_state = self.vocab(composed_states_masked)

        channel1 = self.channel1_linear(composed_states).tanh()
        channel2 = self.channel2_linear(torch.cat([composed_states_masked, idiom_attn_state], dim=-1)).tanh()

        # affection prediction
        emotion_state = self.compose_linear(torch.cat([channel1, channel2], dim=-1)).tanh()

        fine_emotion_logits = self.fine_emotion_classifier(emotion_state)
        # coarse_emotion_logits = self.coarse_emotion_classifier(emotion_state)
        sentiment_logits = self.sentiment_classifier(emotion_state)

        # affection prediction
        logits = self.classifier(emotion_state)

        if compute_loss:
            over_loss, losses = self.loss_fct([over_logits, logits], targets)
            return None, over_loss, None, losses
        else:
            return None, over_logits, None, logits


@register_model('affection-max-pooling-masked-latent-idiom-with-gate')
class AffectionMaxPoolingMaskedLatentIdiomWithGate(BertPreTrainedModel):

    def __init__(self, config, opts):
        super().__init__(config)
        self.model_name = opts.model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if opts.enlarged_candidates is not None:
            self.register_buffer('enlarged_candidates', torch.tensor(opts.enlarged_candidates, dtype=torch.long))
        else:
            self.register_buffer('enlarged_candidates', torch.arange(opts.len_idiom_vocab))

        print(self.enlarged_candidates.size())

        self.idiom_embedding = nn.Embedding(opts.len_idiom_vocab, config.hidden_size)

        self.channel1_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.channel2_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.compose_linear = nn.Linear(config.hidden_size, config.hidden_size)

        self.register_parameter(name='g',
                                param=torch.nn.Parameter(torch.ones(config.hidden_size) / config.hidden_size))

        self.classifier = classifiers[self.project](config.hidden_size, config.hidden_dropout_prob)
        if self.project == 'calo':
            self.loss_fct = loss_calculators['calo'](opts.use_focal, (opts.fine_emotion_weights,
                                                                      opts.sentiment_weights))
        else:
            self.loss_fct = loss_calculators['slide'](opts.use_focal, opts.weights)
        self.init_weights()

    def vocab(self, blank_states):
        idiom_embeddings = self.idiom_embedding(self.enlarged_candidates)
        logits = torch.einsum('bd,nd->bn', [blank_states, idiom_embeddings])  # (b, 256, 10)
        state = torch.einsum('bn,nd->bd', [logits.softmax(dim=-1), idiom_embeddings])  # (b, 256, 10)
        return logits, state

    def forward(self, input_ids, token_type_ids, attention_mask, positions, gather_index, option_ids=None,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        n, batch_size, seq_len = input_ids.size()
        encoded_outputs = self.bert(input_ids.view(n * batch_size, seq_len),
                                    token_type_ids=token_type_ids.view(n * batch_size, seq_len),
                                    attention_mask=attention_mask.view(n * batch_size, seq_len))
        encoded_context = encoded_outputs[0].view(n, batch_size, seq_len, -1)[0]
        encoded_context_masked = encoded_outputs[0].view(n, batch_size, seq_len, -1)[1]

        gather_index, gather_index_masked = gather_index
        gather_index = gather_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(input_ids)
        idiom_states = torch.gather(encoded_context, dim=1, index=gather_index)
        gather_index_masked_unsqueezed = gather_index_masked.unsqueeze(-1).expand(-1, -1,
                                                                                  self.config.hidden_size).type_as(
            input_ids)
        idiom_states_masked = torch.gather(encoded_context_masked, dim=1, index=gather_index_masked_unsqueezed)

        composed_states, _ = idiom_states.max(dim=1)
        composed_states_masked, _ = idiom_states_masked.max(dim=1)

        over_logits, idiom_attn_state = self.vocab(composed_states_masked)

        channel1 = self.channel1_linear(composed_states).tanh()
        channel2 = self.channel2_linear(torch.cat([composed_states_masked, idiom_attn_state], dim=-1)).tanh()

        gate = torch.sigmoid(self.g * channel1)
        s = gate * channel1 + (1 - gate) * channel2

        # affection prediction
        emotion_state = self.compose_linear(s).tanh()

        # affection prediction
        logits = self.classifier(emotion_state)

        if compute_loss:
            over_loss, losses = self.loss_fct([over_logits, logits], targets)
            return None, over_loss, None, losses
        else:
            return None, over_logits, None, logits


@register_model('affection-compose')
class AffectionCompose(BertPreTrainedModel):

    def __init__(self, config, opts):
        super().__init__(config)
        self.model_name = opts.model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.idiom_compose = LatentComposition(config.hidden_size)
        self.channel1_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = classifiers[self.project](config.hidden_size, config.hidden_dropout_prob)
        if self.project == 'calo':
            self.loss_fct = loss_calculators['calo'](opts.use_focal, (opts.fine_emotion_weights,
                                                                      opts.sentiment_weights))
        else:
            self.loss_fct = loss_calculators['slide'](opts.use_focal, opts.weights)
        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask, positions, option_ids, gather_index,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        # n, batch_size, seq_len = input_ids.size()
        encoded_outputs = self.bert(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)
        encoded_context = encoded_outputs[0]

        idiom_length = (gather_index > 0).sum(1)
        gather_index = gather_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(input_ids)
        idiom_states = torch.gather(encoded_context, dim=1, index=gather_index)

        composed_states, _, select_masks = self.idiom_compose(idiom_states, idiom_length)

        emotion_state = self.channel1_linear(composed_states).tanh()

        # affection prediction
        logits = self.classifier(emotion_state)

        if compute_loss:
            _, losses = self.loss_fct([None, logits], targets)
            return None, None, select_masks, losses
        else:
            return None, None, select_masks, logits


@register_model('affection-compose-masked')
class AffectionComposeMasked(BertPreTrainedModel):

    def __init__(self, config, opts):
        super().__init__(config)
        self.model_name = opts.model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.idiom_compose = LatentComposition(config.hidden_size)

        self.channel1_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.channel2_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.compose_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = classifiers[self.project](config.hidden_size, config.hidden_dropout_prob)
        if self.project == 'calo':
            self.loss_fct = loss_calculators['calo'](opts.use_focal, (opts.fine_emotion_weights,
                                                                      opts.sentiment_weights))
        else:
            self.loss_fct = loss_calculators['slide'](opts.use_focal, opts.weights)
        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask, positions, option_ids, gather_index,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        n, batch_size, seq_len = input_ids.size()
        encoded_outputs = self.bert(input_ids.view(n * batch_size, seq_len),
                                    token_type_ids=token_type_ids.view(n * batch_size, seq_len),
                                    attention_mask=attention_mask.view(n * batch_size, seq_len))
        encoded_context = encoded_outputs[0].view(n, batch_size, seq_len, -1)[0]
        encoded_context_masked = encoded_outputs[0].view(n, batch_size, seq_len, -1)[1]

        gather_index, gather_index_masked = gather_index
        idiom_length = (gather_index > 0).sum(1)

        gather_index = gather_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(input_ids)
        idiom_states = torch.gather(encoded_context, dim=1, index=gather_index)

        gather_index_masked = gather_index_masked.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(
            input_ids)
        idiom_states_masked = torch.gather(encoded_context_masked, dim=1, index=gather_index_masked)

        composed_states, _, select_masks = self.idiom_compose(idiom_states, idiom_length)
        # composed_states_masked, _, select_masks_masked = self.idiom_compose(idiom_states_masked, idiom_length)
        composed_states_masked, _ = idiom_states_masked.max(dim=1)

        channel1 = self.channel1_linear(composed_states).tanh()
        channel2 = self.channel2_linear(composed_states_masked).tanh()

        emotion_state = self.compose_linear(torch.cat([channel1, channel2], dim=-1)).tanh()

        # affection prediction
        logits = self.classifier(emotion_state)

        if compute_loss:
            _, losses = self.loss_fct([None, logits], targets)
            return None, None, select_masks, losses
        else:
            return None, None, select_masks, logits


@register_model('affection-compose-masked-latent-idiom')
class AffectionComposeMaskedLatentIdiom(BertPreTrainedModel):

    def __init__(self, config, opts):
        super().__init__(config)
        self.model_name = opts.model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if opts.enlarged_candidates is not None:
            self.register_buffer('enlarged_candidates', torch.tensor(opts.enlarged_candidates, dtype=torch.long))
        else:
            self.register_buffer('enlarged_candidates', torch.arange(opts.len_idiom_vocab))

        print(self.enlarged_candidates.size())

        self.idiom_embedding = nn.Embedding(opts.len_idiom_vocab, config.hidden_size)

        self.idiom_compose = LatentComposition(config.hidden_size)

        self.channel1_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.channel2_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.compose_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = classifiers[self.project](config.hidden_size, config.hidden_dropout_prob)
        if self.project == 'calo':
            self.loss_fct = loss_calculators['calo'](opts.use_focal, (opts.fine_emotion_weights,
                                                                      opts.sentiment_weights))
        else:
            self.loss_fct = loss_calculators['slide'](opts.use_focal, opts.weights)
        self.init_weights()

    def vocab(self, blank_states):
        idiom_embeddings = self.idiom_embedding(self.enlarged_candidates)
        logits = torch.einsum('bd,nd->bn', [blank_states, idiom_embeddings])  # (b, 256, 10)
        state = torch.einsum('bn,nd->bd', [logits.softmax(dim=-1), idiom_embeddings])  # (b, 256, 10)
        return logits, state

    def forward(self, input_ids, token_type_ids, attention_mask, positions, gather_index, option_ids=None,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        n, batch_size, seq_len = input_ids.size()
        encoded_outputs = self.bert(input_ids.view(n * batch_size, seq_len),
                                    token_type_ids=token_type_ids.view(n * batch_size, seq_len),
                                    attention_mask=attention_mask.view(n * batch_size, seq_len))
        encoded_context = encoded_outputs[0].view(n, batch_size, seq_len, -1)[0]
        encoded_context_masked = encoded_outputs[0].view(n, batch_size, seq_len, -1)[1]

        gather_index, gather_index_masked = gather_index
        idiom_length = (gather_index > 0).sum(1)

        gather_index = gather_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(input_ids)
        idiom_states = torch.gather(encoded_context, dim=1, index=gather_index)

        gather_index_masked = gather_index_masked.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(
            input_ids)
        idiom_states_masked = torch.gather(encoded_context_masked, dim=1, index=gather_index_masked)
        # idiom_states = encoded_context[[i for i in range(len(positions))], positions]  # [batch, hidden_state]

        composed_states, _, select_masks = self.idiom_compose(idiom_states, idiom_length)
        composed_states_masked, _ = idiom_states_masked.max(dim=1)

        over_logits, idiom_attn_state = self.vocab(composed_states_masked)

        channel1 = self.channel1_linear(composed_states).tanh()
        channel2 = self.channel2_linear(torch.cat([composed_states_masked, idiom_attn_state], dim=-1)).tanh()

        emotion_state = self.compose_linear(torch.cat([channel1, channel2], dim=-1)).tanh()

        # affection prediction
        logits = self.classifier(emotion_state)

        if compute_loss:
            over_loss, losses = self.loss_fct([over_logits, logits], targets)
            return None, over_loss, select_masks, losses
        else:
            return None, over_logits, select_masks, logits


@register_model('affection-compose-masked-latent-idiom-with-gate')
class AffectionComposeMaskedLatentIdiomWithGate(BertPreTrainedModel):

    def __init__(self, config, opts):
        super().__init__(config)
        self.use_leaf_rnn = True
        self.intra_attention = False
        self.gumbel_temperature = 1
        self.bidirectional = True

        self.model_name = opts.model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if opts.enlarged_candidates is not None:
            self.register_buffer('enlarged_candidates', torch.tensor(opts.enlarged_candidates, dtype=torch.long))
        else:
            self.register_buffer('enlarged_candidates', torch.arange(opts.len_idiom_vocab))

        print(self.enlarged_candidates.size())

        self.idiom_embedding = nn.Embedding(opts.len_idiom_vocab, config.hidden_size)

        self.idiom_compose = LatentComposition(config.hidden_size)

        self.channel1_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.channel2_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.compose_linear = nn.Linear(config.hidden_size, config.hidden_size)

        self.register_parameter(name='g',
                                param=torch.nn.Parameter(torch.ones(config.hidden_size) / config.hidden_size))

        self.classifier = classifiers[self.project](config.hidden_size, config.hidden_dropout_prob)
        if self.project == 'calo':
            self.loss_fct = loss_calculators['calo'](opts.use_focal, (opts.fine_emotion_weights,
                                                                      opts.sentiment_weights))
        else:
            self.loss_fct = loss_calculators['slide'](opts.use_focal, opts.weights)
        self.init_weights()

    def vocab(self, blank_states):
        idiom_embeddings = self.idiom_embedding(self.enlarged_candidates)
        logits = torch.einsum('bd,nd->bn', [blank_states, idiom_embeddings])  # (b, 256, 10)
        state = torch.einsum('bn,nd->bd', [logits.softmax(dim=-1), idiom_embeddings])  # (b, 256, 10)
        return logits, state

    def forward(self, input_ids, token_type_ids, attention_mask, positions, gather_index, option_ids=None,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        n, batch_size, seq_len = input_ids.size()
        encoded_outputs = self.bert(input_ids.view(n * batch_size, seq_len),
                                    token_type_ids=token_type_ids.view(n * batch_size, seq_len),
                                    attention_mask=attention_mask.view(n * batch_size, seq_len))
        encoded_context = encoded_outputs[0].view(n, batch_size, seq_len, -1)[0]
        encoded_context_masked = encoded_outputs[0].view(n, batch_size, seq_len, -1)[1]

        gather_index, gather_index_masked = gather_index
        idiom_length = (gather_index > 0).sum(1)

        gather_index = gather_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(input_ids)
        idiom_states = torch.gather(encoded_context, dim=1, index=gather_index)

        gather_index_masked = gather_index_masked.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(
            input_ids)
        idiom_states_masked = torch.gather(encoded_context_masked, dim=1, index=gather_index_masked)
        # idiom_states = encoded_context[[i for i in range(len(positions))], positions]  # [batch, hidden_state]

        composed_states, _, select_masks = self.idiom_compose(idiom_states, idiom_length)
        composed_states_masked, _ = idiom_states_masked.max(dim=1)

        over_logits, idiom_attn_state = self.vocab(composed_states_masked)

        channel1 = self.channel1_linear(composed_states).tanh()
        channel2 = self.channel2_linear(torch.cat([composed_states_masked, idiom_attn_state], dim=-1)).tanh()

        gate = torch.sigmoid(self.g * channel1)
        s = gate * channel1 + (1 - gate) * channel2

        # affection prediction
        emotion_state = self.compose_linear(s).tanh()

        # affection prediction
        logits = self.classifier(emotion_state)

        if compute_loss:
            over_loss, losses = self.loss_fct([over_logits, logits], targets)
            return None, over_loss, select_masks, losses
        else:
            return None, over_logits, select_masks, logits


@register_model('affection-coattention-masked')
class AffectionCoAttentionMasked(BertPreTrainedModel):

    def __init__(self, config, opts):
        super().__init__(config)
        self.model_name = opts.model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if opts.enlarged_candidates is not None:
            self.register_buffer('enlarged_candidates', torch.tensor(opts.enlarged_candidates, dtype=torch.long))
        else:
            self.register_buffer('enlarged_candidates', torch.arange(opts.len_idiom_vocab))

        print(self.enlarged_candidates.size())

        self.coattention = ContrastiveCoAttention(config.hidden_size)

        self.channel1_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.channel2_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.compose_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = classifiers[self.project](config.hidden_size, config.hidden_dropout_prob)
        if self.project == 'calo':
            self.loss_fct = loss_calculators['calo'](opts.use_focal, (opts.fine_emotion_weights,
                                                                      opts.sentiment_weights))
        else:
            self.loss_fct = loss_calculators['slide'](opts.use_focal, opts.weights)
        self.init_weights()

    def forward(self, input_ids, token_type_ids, attention_mask, positions, gather_index, option_ids=None,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        n, batch_size, seq_len = input_ids.size()
        encoded_outputs = self.bert(input_ids.view(n * batch_size, seq_len),
                                    token_type_ids=token_type_ids.view(n * batch_size, seq_len),
                                    attention_mask=attention_mask.view(n * batch_size, seq_len))
        encoded_context = encoded_outputs[0].view(n, batch_size, seq_len, -1)[0]
        encoded_context_masked = encoded_outputs[0].view(n, batch_size, seq_len, -1)[1]

        gather_index, gather_index_masked = gather_index

        gather_index_unsqueezed = gather_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(input_ids)
        idiom_states = torch.gather(encoded_context, dim=1, index=gather_index_unsqueezed)

        gather_index_masked_unsqueezed = gather_index_masked.unsqueeze(-1).expand(-1, -1,
                                                                                  self.config.hidden_size).type_as(
            input_ids)
        idiom_states_masked = torch.gather(encoded_context_masked, dim=1, index=gather_index_masked_unsqueezed)

        # composed_states_masked, _ = idiom_states_masked.max(dim=1)

        L = idiom_states
        I = idiom_states_masked

        idiom_length = (gather_index > 0).sum(1)
        idiom_mask = sequence_mask(idiom_length)

        C_L, C_I = self.coattention(L, I, idiom_mask, idiom_mask)

        channel1 = self.channel1_linear(C_L).tanh()
        channel2 = self.channel2_linear(C_I).tanh()

        # slide prediction
        emotion_state = self.compose_linear(torch.cat([channel1, channel2], dim=-1)).tanh()

        # affection prediction
        logits = self.classifier(emotion_state)

        if compute_loss:
            _, losses = self.loss_fct([None, logits], targets)
            return None, None, None, losses
        else:
            return None, None, None, logits


@register_model('affection-coattention-masked-latent-idiom')
class AffectionCoAttentionMaskedLatentIdiom(BertPreTrainedModel):

    def __init__(self, config, opts):
        super().__init__(config)
        self.model_name = opts.model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if opts.enlarged_candidates is not None:
            self.register_buffer('enlarged_candidates', torch.tensor(opts.enlarged_candidates, dtype=torch.long))
        else:
            self.register_buffer('enlarged_candidates', torch.arange(opts.len_idiom_vocab))

        print(self.enlarged_candidates.size())

        self.idiom_embedding = nn.Embedding(opts.len_idiom_vocab, config.hidden_size)

        self.coattention = ContrastiveCoAttention(config.hidden_size)

        self.channel1_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.channel2_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.compose_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = classifiers[self.project](config.hidden_size, config.hidden_dropout_prob)
        if self.project == 'calo':
            self.loss_fct = loss_calculators['calo'](opts.use_focal, (opts.fine_emotion_weights,
                                                                      opts.sentiment_weights))
        else:
            self.loss_fct = loss_calculators['slide'](opts.use_focal, opts.weights)
        self.init_weights()

    def vocab(self, blank_states):
        idiom_embeddings = self.idiom_embedding(self.enlarged_candidates)
        logits = torch.einsum('bd,nd->bn', [blank_states, idiom_embeddings])  # (b, 256, 10)
        state = torch.einsum('bn,nd->bd', [logits.softmax(dim=-1), idiom_embeddings])  # (b, 256, 10)
        return logits, state

    def forward(self, input_ids, token_type_ids, attention_mask, positions, gather_index, option_ids=None,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        n, batch_size, seq_len = input_ids.size()
        encoded_outputs = self.bert(input_ids.view(n * batch_size, seq_len),
                                    token_type_ids=token_type_ids.view(n * batch_size, seq_len),
                                    attention_mask=attention_mask.view(n * batch_size, seq_len))
        encoded_context = encoded_outputs[0].view(n, batch_size, seq_len, -1)[0]
        encoded_context_masked = encoded_outputs[0].view(n, batch_size, seq_len, -1)[1]

        gather_index, gather_index_masked = gather_index

        gather_index_unsqueezed = gather_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(input_ids)
        idiom_states = torch.gather(encoded_context, dim=1, index=gather_index_unsqueezed)

        gather_index_masked_unsqueezed = gather_index_masked.unsqueeze(-1).expand(-1, -1,
                                                                                  self.config.hidden_size).type_as(
            input_ids)
        idiom_states_masked = torch.gather(encoded_context_masked, dim=1, index=gather_index_masked_unsqueezed)

        # composed_states_masked, _ = idiom_states_masked.max(dim=1)

        L = idiom_states
        I = idiom_states_masked

        idiom_length = (gather_index > 0).sum(1)
        idiom_mask = sequence_mask(idiom_length)

        C_L, C_I = self.coattention(L, I, idiom_mask, idiom_mask)

        over_logits, idiom_attn_state = self.vocab(C_I)

        channel1 = self.channel1_linear(C_L).tanh()
        channel2 = self.channel2_linear(torch.cat([C_I, idiom_attn_state], dim=-1)).tanh()

        # slide prediction
        emotion_state = self.compose_linear(torch.cat([channel1, channel2], dim=-1)).tanh()

        # affection prediction
        logits = self.classifier(emotion_state)

        if compute_loss:
            over_loss, losses = self.loss_fct([over_logits, logits], targets)
            return None, over_loss, None, losses
        else:
            return None, over_logits, None, logits


@register_model('affection-coattention-masked-latent-idiom-with-gate')
class AffectionCoAttentionMaskedLatentIdiomWithGate(BertPreTrainedModel):

    def __init__(self, config, opts):
        super().__init__(config)
        self.model_name = opts.model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if opts.enlarged_candidates is not None:
            self.register_buffer('enlarged_candidates', torch.tensor(opts.enlarged_candidates, dtype=torch.long))
        else:
            self.register_buffer('enlarged_candidates', torch.arange(opts.len_idiom_vocab))

        print(self.enlarged_candidates.size())

        self.idiom_embedding = nn.Embedding(opts.len_idiom_vocab, config.hidden_size)

        self.coattention = ContrastiveCoAttention(config.hidden_size)

        self.channel1_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.channel2_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.compose_linear = nn.Linear(config.hidden_size, config.hidden_size)

        self.register_parameter(name='g',
                                param=torch.nn.Parameter(torch.ones(config.hidden_size) / config.hidden_size))

        self.classifier = classifiers[self.project](config.hidden_size, config.hidden_dropout_prob)
        if self.project == 'calo':
            self.loss_fct = loss_calculators['calo'](opts.use_focal, (opts.fine_emotion_weights,
                                                                      opts.sentiment_weights))
        else:
            self.loss_fct = loss_calculators['slide'](opts.use_focal, opts.weights)
        self.init_weights()

    def vocab(self, blank_states):
        idiom_embeddings = self.idiom_embedding(self.enlarged_candidates)
        logits = torch.einsum('bd,nd->bn', [blank_states, idiom_embeddings])  # (b, 256, 10)
        state = torch.einsum('bn,nd->bd', [logits.softmax(dim=-1), idiom_embeddings])  # (b, 256, 10)
        return logits, state

    def forward(self, input_ids, token_type_ids, attention_mask, positions, gather_index, option_ids=None,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        n, batch_size, seq_len = input_ids.size()
        encoded_outputs = self.bert(input_ids.view(n * batch_size, seq_len),
                                    token_type_ids=token_type_ids.view(n * batch_size, seq_len),
                                    attention_mask=attention_mask.view(n * batch_size, seq_len))
        encoded_context = encoded_outputs[0].view(n, batch_size, seq_len, -1)[0]
        encoded_context_masked = encoded_outputs[0].view(n, batch_size, seq_len, -1)[1]

        gather_index, gather_index_masked = gather_index

        gather_index_unsqueezed = gather_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(input_ids)
        idiom_states = torch.gather(encoded_context, dim=1, index=gather_index_unsqueezed)

        gather_index_masked_unsqueezed = gather_index_masked.unsqueeze(-1).expand(-1, -1,
                                                                                  self.config.hidden_size).type_as(
            input_ids)
        idiom_states_masked = torch.gather(encoded_context_masked, dim=1, index=gather_index_masked_unsqueezed)

        # composed_states_masked, _ = idiom_states_masked.max(dim=1)

        L = idiom_states
        I = idiom_states_masked

        idiom_length = (gather_index > 0).sum(1)
        idiom_mask = sequence_mask(idiom_length)

        C_L, C_I = self.coattention(L, I, idiom_mask, idiom_mask)

        over_logits, idiom_attn_state = self.vocab(C_I)

        channel1 = self.channel1_linear(C_L).tanh()
        channel2 = self.channel2_linear(torch.cat([C_I, idiom_attn_state], dim=-1)).tanh()

        gate = torch.sigmoid(self.g * channel1)
        s = gate * channel1 + (1 - gate) * channel2

        # affection prediction
        emotion_state = self.compose_linear(s).tanh()

        # affection prediction
        logits = self.classifier(emotion_state)

        if compute_loss:
            over_loss, losses = self.loss_fct([over_logits, logits], targets)
            return None, over_loss, None, losses
        else:
            return None, over_logits, None, logits


@register_model('affection-coattention-masked-full-latent-idiom')
class AffectionCoAttentionMaskedFullLatentIdiom(BertPreTrainedModel):

    def __init__(self, config, opts):
        super().__init__(config)
        self.model_name = opts.model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if opts.enlarged_candidates is not None:
            self.register_buffer('enlarged_candidates', torch.tensor(opts.enlarged_candidates, dtype=torch.long))
        else:
            self.register_buffer('enlarged_candidates', torch.arange(opts.len_idiom_vocab))

        print(self.enlarged_candidates.size())

        self.idiom_embedding = nn.Embedding(opts.len_idiom_vocab, config.hidden_size)

        # self.idiom_compose = LatentComposition(config.hidden_size)
        self.coattention = ContrastiveCoAttention(config.hidden_size)

        self.channel1_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.channel2_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.compose_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)

        self.classifier = classifiers[self.project](config.hidden_size, config.hidden_dropout_prob)
        if self.project == 'calo':
            self.loss_fct = loss_calculators['calo'](opts.use_focal, (opts.fine_emotion_weights,
                                                                      opts.sentiment_weights))
        else:
            self.loss_fct = loss_calculators['slide'](opts.use_focal, opts.weights)
        self.init_weights()

    def vocab(self, blank_states):
        idiom_embeddings = self.idiom_embedding(self.enlarged_candidates)
        logits = torch.einsum('bd,nd->bn', [blank_states, idiom_embeddings])  # (b, 256, 10)
        state = torch.einsum('bn,nd->bd', [logits.softmax(dim=-1), idiom_embeddings])  # (b, 256, 10)
        return logits, state

    def forward(self, input_ids, token_type_ids, attention_mask, positions, gather_index, option_ids=None,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        n, batch_size, seq_len = input_ids.size()
        encoded_outputs = self.bert(input_ids.view(n * batch_size, seq_len),
                                    token_type_ids=token_type_ids.view(n * batch_size, seq_len),
                                    attention_mask=attention_mask.view(n * batch_size, seq_len))
        encoded_context = encoded_outputs[0].view(n, batch_size, seq_len, -1)[0]
        encoded_context_masked = encoded_outputs[0].view(n, batch_size, seq_len, -1)[1]

        gather_index, gather_index_masked = gather_index

        gather_index_unsqueezed = gather_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(input_ids)
        idiom_states = torch.gather(encoded_context, dim=1, index=gather_index_unsqueezed)

        gather_index_masked = gather_index_masked.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(
            input_ids)
        idiom_states_masked = torch.gather(encoded_context_masked, dim=1, index=gather_index_masked)
        # idiom_states = encoded_context[[i for i in range(len(positions))], positions]  # [batch, hidden_state]

        # composed_states, _, select_masks = self.idiom_compose(idiom_states, idiom_length)
        composed_states_masked, _ = idiom_states_masked.max(dim=1)

        L = idiom_states
        mask_L = torch.gather(attention_mask[0], dim=1, index=gather_index)
        I = encoded_context_masked
        mask_I = attention_mask[1]

        C_L, C_I = self.coattention(L, I, mask_L, mask_I)

        over_logits, idiom_attn_state = self.vocab(C_I)

        channel1 = self.channel1_linear(C_L).tanh()
        channel2 = self.channel2_linear(torch.cat([C_I, idiom_attn_state], dim=-1)).tanh()

        # slide prediction
        emotion_state = self.compose_linear(torch.cat([channel1, channel2], dim=-1)).tanh()

        # affection prediction
        logits = self.classifier(emotion_state)

        if compute_loss:
            over_loss, losses = self.loss_fct([over_logits, logits], targets)
            return None, over_loss, None, losses
        else:
            return None, over_logits, None, logits
