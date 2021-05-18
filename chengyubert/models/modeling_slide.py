import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel

from chengyubert.models import register_model
from chengyubert.modules.attention import ContrastiveCoAttention
from chengyubert.modules.utils import WeightNormClassifier, LatentComposition, sequence_mask
from chengyubert.optim.loss import FocalLoss


@register_model('slide-max-pooling')
class SlideMaxPooling(BertPreTrainedModel):

    def __init__(self, config, opts):
        super().__init__(config)
        self.use_leaf_rnn = True
        self.intra_attention = False
        self.gumbel_temperature = 1
        self.bidirectional = True

        self.model_name = opts.model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.channel1_linear = nn.Linear(config.hidden_size, config.hidden_size)
        emotion_hidden_size = config.hidden_size

        # Sentiment Predictor
        self.sentiment_classifier = WeightNormClassifier(emotion_hidden_size,
                                                         3,
                                                         emotion_hidden_size,
                                                         config.hidden_dropout_prob)

        if opts.use_focal:
            self.loss_fct = FocalLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss(weight=opts.weight, reduction='none')
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

        # composed_states, _, select_masks = self.idiom_compose(idiom_states, idiom_length)
        composed_states, _ = idiom_states.max(dim=1)

        emotion_state = self.channel1_linear(composed_states).tanh()

        # slide prediction
        sentiment_logits = self.sentiment_classifier(emotion_state)

        if compute_loss:
            sentiment_emotion_loss = self.loss_fct(sentiment_logits, targets[:, 1])
            return None, None, None, sentiment_emotion_loss
        else:
            return None, None, None, sentiment_logits


@register_model('slide-max-pooling-masked')
class SlideMaxPoolingMasked(BertPreTrainedModel):

    def __init__(self, config, opts):
        super().__init__(config)
        self.model_name = opts.model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.channel1_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.channel2_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.compose_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)

        emotion_hidden_size = config.hidden_size
        # Sentiment Predictor
        self.sentiment_classifier = WeightNormClassifier(emotion_hidden_size,
                                                         3,
                                                         emotion_hidden_size,
                                                         config.hidden_dropout_prob)

        if opts.use_focal:
            self.loss_fct = FocalLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss(weight=opts.weight, reduction='none')
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

        composed_states, _ = idiom_states.max(dim=1)
        composed_states_masked, _ = idiom_states_masked.max(dim=1)

        channel1 = self.channel1_linear(composed_states).tanh()
        channel2 = self.channel2_linear(composed_states_masked).tanh()

        # affection prediction
        emotion_state = self.compose_linear(torch.cat([channel1, channel2], dim=-1)).tanh()

        # slide prediction
        sentiment_logits = self.sentiment_classifier(emotion_state)

        if compute_loss:
            sentiment_emotion_loss = self.loss_fct(sentiment_logits, targets[:, 1])
            return None, None, None, sentiment_emotion_loss
        else:
            return None, None, None, sentiment_logits


@register_model('slide-max-pooling-masked-latent-idiom')
class SlideMaxPoolingMaskedLatentIdiom(BertPreTrainedModel):

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

        # Sentiment Predictor
        self.sentiment_classifier = WeightNormClassifier(config.hidden_size,
                                                         3,
                                                         config.hidden_size,
                                                         config.hidden_dropout_prob)

        if opts.use_focal:
            self.loss_fct = FocalLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss(weight=opts.weight, reduction='none')
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

        composed_states, _ = idiom_states.max(dim=1)
        composed_states_masked, _ = idiom_states_masked.max(dim=1)

        over_logits, idiom_attn_state = self.vocab(composed_states_masked)

        channel1 = self.channel1_linear(composed_states).tanh()
        channel2 = self.channel2_linear(torch.cat([composed_states_masked, idiom_attn_state], dim=-1)).tanh()

        # affection prediction
        emotion_state = self.compose_linear(torch.cat([channel1, channel2], dim=-1)).tanh()

        sentiment_logits = self.sentiment_classifier(emotion_state)

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            over_loss = loss_fct(over_logits, targets[:, 0])
            # loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            # over_loss = loss_fct(over_logits,
            #                      nn.functional.one_hot(targets[:, 0],
            #                                            num_classes=self.enlarged_candidates.size(0)).type_as(
            #                          over_logits))
            # over_loss = over_loss.sum(dim=-1)
            sentiment_emotion_loss = self.loss_fct(sentiment_logits, targets[:, 1])
            return (None, over_loss, None, sentiment_emotion_loss)
        else:
            return (None, over_logits, None, sentiment_logits)


@register_model('slide-max-pooling-masked-latent-idiom-with-gate')
class SlideMaxPoolingMaskedLatentIdiomWithGate(BertPreTrainedModel):

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
        # Idiom Predictor
        # Sentiment Predictor
        self.sentiment_classifier = WeightNormClassifier(config.hidden_size,
                                                         3,
                                                         config.hidden_size,
                                                         config.hidden_dropout_prob)

        if opts.use_focal:
            self.loss_fct = FocalLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss(weight=opts.weight, reduction='none')
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

        gather_index_masked = gather_index_masked.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(
            input_ids)
        idiom_states_masked = torch.gather(encoded_context_masked, dim=1, index=gather_index_masked)
        # idiom_states = encoded_context[[i for i in range(len(positions))], positions]  # [batch, hidden_state]

        composed_states, _ = idiom_states.max(dim=1)
        composed_states_masked, _ = idiom_states_masked.max(dim=1)

        over_logits, idiom_attn_state = self.vocab(composed_states_masked)

        channel1 = self.channel1_linear(composed_states).tanh()
        channel2 = self.channel2_linear(torch.cat([composed_states_masked, idiom_attn_state], dim=-1)).tanh()

        gate = torch.sigmoid(self.g * channel1)
        s = gate * channel1 + (1 - gate) * channel2

        # affection prediction
        emotion_state = self.compose_linear(s).tanh()
        sentiment_logits = self.sentiment_classifier(emotion_state)

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            over_loss = loss_fct(over_logits, targets[:, 0])
            # loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            # over_loss = loss_fct(over_logits,
            #                      nn.functional.one_hot(targets[:, 0],
            #                                            num_classes=self.enlarged_candidates.size(0)).type_as(
            #                          over_logits))
            # over_loss = over_loss.sum(dim=-1)
            sentiment_emotion_loss = self.loss_fct(sentiment_logits, targets[:, 1])
            return (None, over_loss, None, sentiment_emotion_loss)
        else:
            return (None, over_logits, None, sentiment_logits)


@register_model('slide-compose')
class SlideCompose(BertPreTrainedModel):

    def __init__(self, config, opts):
        super().__init__(config)
        self.model_name = opts.model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.idiom_compose = LatentComposition(config.hidden_size)
        self.channel1_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)

        emotion_hidden_size = config.hidden_size

        # Sentiment Predictor
        self.sentiment_classifier = WeightNormClassifier(emotion_hidden_size,
                                                         3,
                                                         emotion_hidden_size,
                                                         config.hidden_dropout_prob)

        if opts.use_focal:
            self.loss_fct = FocalLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss(weight=opts.weight, reduction='none')
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

        # slide prediction
        sentiment_logits = self.sentiment_classifier(emotion_state)

        if compute_loss:
            sentiment_emotion_loss = self.loss_fct(sentiment_logits, targets[:, 1])
            return None, None, select_masks, sentiment_emotion_loss
        else:
            return None, None, select_masks, sentiment_logits


@register_model('slide-compose-masked')
class SlideComposeMasked(BertPreTrainedModel):

    def __init__(self, config, opts):
        super().__init__(config)
        self.model_name = opts.model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.idiom_compose = LatentComposition(config.hidden_size)

        self.channel1_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.channel2_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.compose_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)

        emotion_hidden_size = config.hidden_size

        # Sentiment Predictor
        self.sentiment_classifier = WeightNormClassifier(emotion_hidden_size,
                                                         3,
                                                         emotion_hidden_size,
                                                         config.hidden_dropout_prob)

        if opts.use_focal:
            self.loss_fct = FocalLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss(weight=opts.weight, reduction='none')
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

        # slide prediction
        sentiment_logits = self.sentiment_classifier(emotion_state)

        if compute_loss:
            sentiment_emotion_loss = self.loss_fct(sentiment_logits, targets[:, 1])
            return None, None, select_masks, sentiment_emotion_loss
        else:
            return None, None, select_masks, sentiment_logits


@register_model('slide-compose-masked-latent-idiom')
class SlideComposeMaskedLatentIdiom(BertPreTrainedModel):

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

        # Idiom Predictor
        # Sentiment Predictor
        self.sentiment_classifier = WeightNormClassifier(config.hidden_size,
                                                         3,
                                                         config.hidden_size,
                                                         config.hidden_dropout_prob)
        if opts.use_focal:
            self.loss_fct = FocalLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss(weight=opts.weight, reduction='none')
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

        sentiment_logits = self.sentiment_classifier(emotion_state)

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            over_loss = loss_fct(over_logits, targets[:, 0])
            # loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            # over_loss = loss_fct(over_logits,
            #                      nn.functional.one_hot(targets[:, 0],
            #                                            num_classes=self.enlarged_candidates.size(0)).type_as(
            #                          over_logits))
            # over_loss = over_loss.sum(dim=-1)
            sentiment_emotion_loss = self.loss_fct(sentiment_logits, targets[:, 1])
            return (None, over_loss, select_masks, sentiment_emotion_loss)
        else:
            return (None, over_logits, select_masks, sentiment_logits)


@register_model('slide-compose-masked-latent-idiom-with-gate')
class SlideComposeMaskedLatentIdiomWithGate(BertPreTrainedModel):

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

        # Idiom Predictor
        # Sentiment Predictor
        self.sentiment_classifier = WeightNormClassifier(config.hidden_size,
                                                         3,
                                                         config.hidden_size,
                                                         config.hidden_dropout_prob)
        if opts.use_focal:
            self.loss_fct = FocalLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss(weight=opts.weight, reduction='none')
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

        sentiment_logits = self.sentiment_classifier(emotion_state)

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            over_loss = loss_fct(over_logits, targets[:, 0])
            # loss_fct = nn.BCEWithLogitsLoss(reduction='none')
            # over_loss = loss_fct(over_logits,
            #                      nn.functional.one_hot(targets[:, 0],
            #                                            num_classes=self.enlarged_candidates.size(0)).type_as(
            #                          over_logits))
            # over_loss = over_loss.sum(dim=-1)
            sentiment_emotion_loss = self.loss_fct(sentiment_logits, targets[:, 1])
            return (None, over_loss, select_masks, sentiment_emotion_loss)
        else:
            return (None, over_logits, select_masks, sentiment_logits)


@register_model('slide-coattention-masked')
class SlideCoAttentionMasked(BertPreTrainedModel):

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

        # Sentiment Predictor
        self.sentiment_classifier = WeightNormClassifier(config.hidden_size,
                                                         3,
                                                         config.hidden_size,
                                                         config.hidden_dropout_prob)

        if opts.use_focal:
            self.loss_fct = FocalLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss(weight=opts.weight, reduction='none')
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

        sentiment_logits = self.sentiment_classifier(emotion_state)

        if compute_loss:
            sentiment_emotion_loss = self.loss_fct(sentiment_logits, targets[:, 1])
            return (None, None, None, sentiment_emotion_loss)
        else:
            return (None, None, None, sentiment_logits)


@register_model('slide-coattention-masked-latent-idiom')
class SlideCoAttentionMaskedLatentIdiom(BertPreTrainedModel):

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

        # Sentiment Predictor
        self.sentiment_classifier = WeightNormClassifier(config.hidden_size,
                                                         3,
                                                         config.hidden_size,
                                                         config.hidden_dropout_prob)

        if opts.use_focal:
            self.loss_fct = FocalLoss()
        else:
            self.loss_fct = nn.CrossEntropyLoss(weight=opts.weight, reduction='none')
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

        sentiment_logits = self.sentiment_classifier(emotion_state)

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            over_loss = loss_fct(over_logits, targets[:, 0])
            sentiment_emotion_loss = self.loss_fct(sentiment_logits, targets[:, 1])
            return (None, over_loss, None, sentiment_emotion_loss)
        else:
            return (None, over_logits, None, sentiment_logits)

# @register_model('chengyubert-slide-latent-idiom-masked-coattention-full')
# class ChengyuBertSlideLatentIdiomMaskedCoAttentionFull(BertPreTrainedModel):
#
#     def __init__(self, config, opts):
#         super().__init__(config)
#         self.model_name = opts.model
#         self.bert = BertModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#
#         if opts.enlarged_candidates is not None:
#             self.register_buffer('enlarged_candidates', torch.tensor(opts.enlarged_candidates, dtype=torch.long))
#         else:
#             self.register_buffer('enlarged_candidates', torch.arange(opts.len_idiom_vocab))
#
#         print(self.enlarged_candidates.size())
#
#         self.idiom_embedding = nn.Embedding(opts.len_idiom_vocab, config.hidden_size)
#
#         # self.idiom_compose = LatentComposition(config.hidden_size)
#         self.coattention = ContrastiveCoAttention(config.hidden_size)
#         self.compose_linear = nn.Linear(config.hidden_size * 3, config.hidden_size)
#         # self.compose_linear = nn.Sequential(nn.Linear(4 * config.hidden_size, config.hidden_size),
#         #                                     nn.SELU(),
#         #                                     nn.Dropout(p=0.1),
#         #                                     nn.Linear(config.hidden_size, config.hidden_size),
#         #                                     nn.SELU())
#
#         # Idiom Predictor
#         # Sentiment Predictor
#         self.sentiment_classifier = WeightNormClassifier(config.hidden_size,
#                                                          3,
#                                                          config.hidden_size,
#                                                          config.hidden_dropout_prob)
#
#         if opts.use_focal:
#             self.loss_fct = FocalLoss()
#         else:
#             self.loss_fct = nn.CrossEntropyLoss(weight=opts.weight, reduction='none')
#         self.init_weights()
#
#     def vocab(self, blank_states):
#         idiom_embeddings = self.idiom_embedding(self.enlarged_candidates)
#         logits = torch.einsum('bd,nd->bn', [blank_states, idiom_embeddings])  # (b, 256, 10)
#         state = torch.einsum('bn,nd->bd', [logits.softmax(dim=-1), idiom_embeddings])  # (b, 256, 10)
#         return logits, state
#
#     def forward(self, input_ids, token_type_ids, attention_mask, positions, gather_index, option_ids=None,
#                 inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
#         n, batch_size, seq_len = input_ids.size()
#         encoded_outputs = self.bert(input_ids.view(n * batch_size, seq_len),
#                                     token_type_ids=token_type_ids.view(n * batch_size, seq_len),
#                                     attention_mask=attention_mask.view(n * batch_size, seq_len))
#         encoded_context = encoded_outputs[0].view(n, batch_size, seq_len, -1)[0]
#         encoded_context_masked = encoded_outputs[0].view(n, batch_size, seq_len, -1)[1]
#
#         gather_index, gather_index_masked = gather_index
#
#         gather_index_unsqueezed = gather_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(input_ids)
#         idiom_states = torch.gather(encoded_context, dim=1, index=gather_index_unsqueezed)
#
#         gather_index_masked = gather_index_masked.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(
#             input_ids)
#         idiom_states_masked = torch.gather(encoded_context_masked, dim=1, index=gather_index_masked)
#         # idiom_states = encoded_context[[i for i in range(len(positions))], positions]  # [batch, hidden_state]
#
#         # composed_states, _, select_masks = self.idiom_compose(idiom_states, idiom_length)
#         composed_states_masked, _ = idiom_states_masked.max(dim=1)
#         over_logits, idiom_attn_state = self.vocab(composed_states_masked)
#
#         L = idiom_states
#         mask_L = torch.gather(attention_mask[0], dim=1, index=gather_index)
#         I = encoded_context_masked
#         mask_I = attention_mask[1]
#
#         C_L, C_I = self.coattention(L, I, mask_L, mask_I)
#
#         # slide prediction
#         emotion_state = self.compose_linear(torch.cat([C_L, C_I,
#                                                        idiom_attn_state], dim=-1)).tanh()
#
#         sentiment_logits = self.sentiment_classifier(emotion_state)
#
#         if compute_loss:
#             loss_fct = nn.CrossEntropyLoss(reduction='none')
#             over_loss = loss_fct(over_logits, targets[:, 0])
#             sentiment_emotion_loss = self.loss_fct(sentiment_logits, targets[:, 1])
#             return (None, over_loss, None, sentiment_emotion_loss)
#         else:
#             return (None, over_logits, None, sentiment_logits)
#
#
