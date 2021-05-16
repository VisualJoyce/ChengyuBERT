import torch
from torch import nn
from torch.nn.utils import weight_norm
from transformers import BertModel, BertPreTrainedModel

from chengyubert.models import register_model
from chengyubert.optim.loss import FocalLoss


def convert_to_one_hot(indices, num_classes):
    """
    Args:
        indices (tensor): A vector containing indices,
            whose size is (batch_size,).
        num_classes (tensor): The number of classes, which would be
            the second dimension of the resulting one-hot matrix.

    Returns:
        result: The one-hot matrix of size (batch_size, num_classes).
    """

    batch_size = indices.size(0)
    indices = indices.unsqueeze(1)
    one_hot = indices.new_zeros(batch_size, num_classes).scatter_(1, indices, 1)
    return one_hot


def masked_softmax(logits, mask=None):
    # eps = 1e-20
    # probs = torch.softmax(logits, dim=1)
    # if mask is not None:
    #     mask = mask.float()
    #     probs = probs * mask + eps
    #     probs = probs / probs.sum(1, keepdim=True)
    logits += (1.0 - mask.type_as(logits)) * -10000.0
    return logits.softmax(dim=-1)


def greedy_select(logits, mask=None):
    probs = masked_softmax(logits=logits, mask=mask)
    one_hot = convert_to_one_hot(indices=probs.max(1)[1],
                                 num_classes=logits.size(1))
    return one_hot


def st_gumbel_softmax(logits, temperature=1.0, mask=None):
    """
    Return the result of Straight-Through Gumbel-Softmax Estimation.
    It approximates the discrete sampling via Gumbel-Softmax trick
    and applies the biased ST estimator.
    In the forward propagation, it emits the discrete one-hot result,
    and in the backward propagation it approximates the categorical
    distribution via smooth Gumbel-Softmax distribution.

    Args:
        logits (tensor): A un-normalized probability values,
            which has the size (batch_size, num_classes)
        temperature (float): A temperature parameter. The higher
            the value is, the smoother the distribution is.
        mask (tensor, optional): If given, it masks the softmax
            so that indices of '0' mask values are not selected.
            The size is (batch_size, num_classes).

    Returns:
        y: The sampled output, which has the property explained above.
    """

    eps = 1e-10
    u = logits.data.new(*logits.size()).uniform_()
    gumbel_noise = -torch.log(-torch.log(u + eps) + eps)
    y = logits + gumbel_noise
    y = masked_softmax(logits=y / temperature, mask=mask)
    y_argmax = y.max(1)[1]
    y_hard = convert_to_one_hot(indices=y_argmax, num_classes=y.size(1)).float()
    y = (y_hard - y).detach() + y
    return y


def sequence_mask(sequence_length, max_length=None):
    if max_length is None:
        max_length = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_length).long().type_as(sequence_length)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_length)
    seq_range_expand = seq_range_expand.to(sequence_length)
    seq_length_expand = sequence_length.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand


def reverse_padded_sequence(inputs, lengths, batch_first=False):
    """Reverses sequences according to their lengths.
    Inputs should have size ``T x B x *`` if ``batch_first`` is False, or
    ``B x T x *`` if True. T is the length of the longest sequence (or larger),
    B is the batch size, and * is any number of dimensions (including 0).
    Arguments:
        inputs (tensor): padded batch of variable length sequences.
        lengths (list[int]): list of sequence lengths
        batch_first (bool, optional): if True, inputs should be B x T x *.
    Returns:
        A tensor with the same size as inputs, but with each sequence
        reversed according to its length.
    """

    if not batch_first:
        inputs = inputs.transpose(0, 1)
    if inputs.size(0) != len(lengths):
        raise ValueError('inputs incompatible with lengths.')
    reversed_indices = [list(range(inputs.size(1)))
                        for _ in range(inputs.size(0))]
    for i, length in enumerate(lengths):
        if length > 0:
            reversed_indices[i][:length] = reversed_indices[i][length - 1::-1]
    reversed_indices = (torch.LongTensor(reversed_indices).unsqueeze(2)
                        .expand_as(inputs))
    # reversed_indices = reversed_indices.to(inputs)
    reversed_inputs = torch.gather(inputs, 1, reversed_indices.type_as(lengths))
    if not batch_first:
        reversed_inputs = reversed_inputs.transpose(0, 1)
    return reversed_inputs


class BinaryTreeLSTMLayer(nn.Module):

    def __init__(self, hidden_dim):
        super(BinaryTreeLSTMLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.comp_linear = nn.Linear(in_features=2 * hidden_dim,
                                     out_features=5 * hidden_dim)

    def forward(self, l=None, r=None):
        """
        Args:
            l: A (h_l, c_l) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
            r: A (h_r, c_r) tuple, where each value has the size
                (batch_size, max_length, hidden_dim).
        Returns:
            h, c: The hidden and cell state of the composed parent,
                each of which has the size
                (batch_size, max_length - 1, hidden_dim).
        """
        hl, cl = l
        hr, cr = r
        hlr_cat = torch.cat([hl, hr], dim=2)
        treelstm_vector = self.comp_linear(hlr_cat)
        i, fl, fr, u, o = treelstm_vector.chunk(chunks=5, dim=2)
        c = (cl * (fl + 1).sigmoid() + cr * (fr + 1).sigmoid()
             + u.tanh() * i.sigmoid())
        h = o.sigmoid() * c.tanh()
        return h, c


class LatentComposition(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.use_leaf_rnn = True
        self.intra_attention = False
        self.gumbel_temperature = 1
        self.bidirectional = True
        self.hidden_size = hidden_size

        assert not (self.bidirectional and not self.use_leaf_rnn)

        word_dim = hidden_size
        hidden_dim = hidden_size
        if self.use_leaf_rnn:
            self.leaf_rnn_cell = nn.LSTMCell(
                input_size=word_dim, hidden_size=hidden_dim)
            if self.bidirectional:
                self.leaf_rnn_cell_bw = nn.LSTMCell(
                    input_size=word_dim, hidden_size=hidden_dim)
        else:
            self.word_linear = nn.Linear(in_features=word_dim,
                                         out_features=2 * hidden_dim)
        if self.bidirectional:
            self.treelstm_layer = BinaryTreeLSTMLayer(2 * hidden_dim)
            # self.comp_query = nn.Parameter(torch.FloatTensor(2 * hidden_dim))
            self.comp_query_linear = nn.Linear(hidden_dim * 2, 1, bias=False)
        else:
            self.treelstm_layer = BinaryTreeLSTMLayer(hidden_dim)
            # self.comp_query = nn.Parameter(torch.FloatTensor(hidden_dim))
            self.comp_query_linear = nn.Linear(hidden_dim, 1, bias=False)

    @staticmethod
    def update_state(old_state, new_state, done_mask):
        old_h, old_c = old_state
        new_h, new_c = new_state
        done_mask = done_mask.type_as(old_h).unsqueeze(1).unsqueeze(2)
        h = done_mask * new_h + (1 - done_mask) * old_h[:, :-1, :]
        c = done_mask * new_c + (1 - done_mask) * old_c[:, :-1, :]
        return h, c

    def select_composition(self, old_state, new_state, mask):
        new_h, new_c = new_state
        old_h, old_c = old_state
        old_h_left, old_h_right = old_h[:, :-1, :], old_h[:, 1:, :]
        old_c_left, old_c_right = old_c[:, :-1, :], old_c[:, 1:, :]
        # comp_weights = (self.comp_query * new_h).sum(-1)
        # comp_weights = comp_weights / math.sqrt(self.config.hidden_size)
        comp_weights = self.comp_query_linear(new_h).sum(-1).squeeze(dim=-1)
        if self.training:
            # select_mask = st_gumbel_softmax(
            #     logits=comp_weights, temperature=self.gumbel_temperature,
            #     mask=mask)
            select_mask = torch.nn.functional.gumbel_softmax(logits=comp_weights,
                                                             tau=self.gumbel_temperature,
                                                             hard=True)
        else:
            select_mask = greedy_select(logits=comp_weights, mask=mask)

        select_mask = select_mask.type_as(old_h)
        select_mask_expand = select_mask.unsqueeze(2).expand_as(new_h)
        select_mask_cumsum = select_mask.cumsum(1)

        left_mask = 1 - select_mask_cumsum
        left_mask_expand = left_mask.unsqueeze(2).expand_as(old_h_left)
        right_mask = select_mask_cumsum - select_mask
        right_mask_expand = right_mask.unsqueeze(2).expand_as(old_h_right)

        new_h = (select_mask_expand * new_h
                 + left_mask_expand * old_h_left
                 + right_mask_expand * old_h_right)
        new_c = (select_mask_expand * new_c
                 + left_mask_expand * old_c_left
                 + right_mask_expand * old_c_right)

        selected_h = (select_mask_expand * new_h).sum(1)
        return new_h, new_c, select_mask, selected_h

    def forward(self, idiom_hidden, idiom_length):
        max_depth = idiom_hidden.size(1)
        length_mask = sequence_mask(sequence_length=idiom_length,
                                    max_length=max_depth)
        select_masks = []

        if self.use_leaf_rnn:
            hs = []
            cs = []
            batch_size, max_length, _ = idiom_hidden.size()
            zero_state = idiom_hidden.data.new_zeros(batch_size, self.hidden_size)
            # input.data.new_zeros(batch_size, self.config.hidden_size)
            h_prev = c_prev = zero_state
            for i in range(max_length):
                h, c = self.leaf_rnn_cell(input=idiom_hidden[:, i, :], hx=(h_prev, c_prev))
                hs.append(h)
                cs.append(c)
                h_prev = h
                c_prev = c
            hs = torch.stack(hs, dim=1)
            cs = torch.stack(cs, dim=1)

            if self.bidirectional:
                hs_bw = []
                cs_bw = []
                h_bw_prev = c_bw_prev = zero_state
                # lengths_list = list(length.data)
                input_bw = reverse_padded_sequence(
                    inputs=idiom_hidden, lengths=idiom_length, batch_first=True)
                for i in range(max_length):
                    h_bw, c_bw = self.leaf_rnn_cell_bw(
                        input=input_bw[:, i, :], hx=(h_bw_prev, c_bw_prev))
                    hs_bw.append(h_bw)
                    cs_bw.append(c_bw)
                    h_bw_prev = h_bw
                    c_bw_prev = c_bw
                hs_bw = torch.stack(hs_bw, dim=1)
                cs_bw = torch.stack(cs_bw, dim=1)
                hs_bw = reverse_padded_sequence(
                    inputs=hs_bw, lengths=idiom_length, batch_first=True)
                cs_bw = reverse_padded_sequence(
                    inputs=cs_bw, lengths=idiom_length, batch_first=True)
                hs = torch.cat([hs, hs_bw], dim=2)
                cs = torch.cat([cs, cs_bw], dim=2)
            state = (hs, cs)
        else:
            state = self.word_linear(idiom_hidden)
            state = state.chunk(chunks=2, dim=2)

        nodes = []
        if self.intra_attention:
            nodes.append(state[0])

        for i in range(max_depth - 1):
            h, c = state
            l = (h[:, :-1, :], c[:, :-1, :])
            r = (h[:, 1:, :], c[:, 1:, :])
            new_state = self.treelstm_layer(l=l, r=r)
            if i < max_depth - 2:
                # We don't need to greedily select the composition in the
                # last iteration, since it has only one option left.
                new_h, new_c, select_mask, selected_h = self.select_composition(
                    old_state=state, new_state=new_state,
                    mask=length_mask[:, i + 1:])
                new_state = (new_h, new_c)
                select_masks.append(select_mask)
                if self.intra_attention:
                    nodes.append(selected_h)
            done_mask = length_mask[:, i + 1]
            state = self.update_state(old_state=state, new_state=new_state,
                                      done_mask=done_mask)
            if self.intra_attention and i >= max_depth - 2:
                nodes.append(state[0])

        h, c = state
        if self.intra_attention:
            att_mask = torch.cat([length_mask, length_mask[:, 1:]], dim=1)
            att_mask = att_mask.type_as(h)
            # nodes: (batch_size, num_tree_nodes, hidden_dim)
            nodes = torch.cat(nodes, dim=1)
            att_mask_expand = att_mask.unsqueeze(2).expand_as(nodes)
            nodes = nodes * att_mask_expand
            # nodes_mean: (batch_size, hidden_dim, 1)
            nodes_mean = nodes.mean(1).squeeze(1).unsqueeze(2)
            # att_weights: (batch_size, num_tree_nodes)
            att_weights = torch.bmm(nodes, nodes_mean).squeeze(2)
            att_weights = masked_softmax(logits=att_weights, mask=att_mask)
            # att_weights_expand: (batch_size, num_tree_nodes, hidden_dim)
            att_weights_expand = att_weights.unsqueeze(2).expand_as(nodes)
            # h: (batch_size, 1, 2 * hidden_dim)
            h = (att_weights_expand * nodes).sum(1)
        assert h.size(1) == 1 and c.size(1) == 1
        return h.squeeze(1), c.squeeze(1), select_masks


class GatedTanh(nn.Module):
    """
    From: https://arxiv.org/pdf/1707.07998.pdf
    nonlinear_layer (f_a) : x\in R^m => y \in R^n
    \tilda{y} = tanh(Wx + b)
    g = sigmoid(W'x + b')
    y = \tilda(y) \circ g
    input: (N, *, in_dim)
    output: (N, *, out_dim)
    """

    def __init__(self, in_dim, out_dim):
        super(GatedTanh, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.gate_fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        y_tilda = torch.tanh(self.fc(x))
        gated = torch.sigmoid(self.gate_fc(x))

        # Element wise multiplication
        y = y_tilda * gated

        return y


class WeightNormClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, dropout):
        super(WeightNormClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hidden_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hidden_dim, out_dim), dim=None),
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits


@register_model('chengyubert-slide-max-pooling')
class ChengyuBertSlideMaxPooling(BertPreTrainedModel):

    def __init__(self, config, opts):
        super().__init__(config)
        self.use_leaf_rnn = True
        self.intra_attention = False
        self.gumbel_temperature = 1
        self.bidirectional = True

        self.model_name = opts.model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.idiom_compose = LatentComposition(config.hidden_size)
        self.compose_linear = nn.Linear(config.hidden_size, config.hidden_size)
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

        emotion_state = self.compose_linear(composed_states).tanh()

        # slide prediction
        sentiment_logits = self.sentiment_classifier(emotion_state)

        if compute_loss:
            sentiment_emotion_loss = self.loss_fct(sentiment_logits, targets[:, 1])
            return None, None, None, sentiment_emotion_loss
        else:
            return None, None, None, sentiment_logits


@register_model('chengyubert-slide-compose-only')
class ChengyuBertSlideComposeOnly(BertPreTrainedModel):

    def __init__(self, config, opts):
        super().__init__(config)
        self.use_leaf_rnn = True
        self.intra_attention = False
        self.gumbel_temperature = 1
        self.bidirectional = True

        self.model_name = opts.model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.idiom_compose = LatentComposition(config.hidden_size)
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
        # n, batch_size, seq_len = input_ids.size()
        encoded_outputs = self.bert(input_ids,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)
        encoded_context = encoded_outputs[0]

        idiom_length = (gather_index > 0).sum(1)
        gather_index = gather_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(input_ids)
        idiom_states = torch.gather(encoded_context, dim=1, index=gather_index)

        composed_states, _, select_masks = self.idiom_compose(idiom_states, idiom_length)

        emotion_state = self.compose_linear(composed_states).tanh()

        # slide prediction
        sentiment_logits = self.sentiment_classifier(emotion_state)

        if compute_loss:
            sentiment_emotion_loss = self.loss_fct(sentiment_logits, targets[:, 1])
            return None, None, select_masks, sentiment_emotion_loss
        else:
            return None, None, select_masks, sentiment_logits


@register_model('chengyubert-slide-compose-only-masked')
class ChengyuBertSlideComposeOnlyMasked(BertPreTrainedModel):

    def __init__(self, config, opts):
        super().__init__(config)
        self.use_leaf_rnn = True
        self.intra_attention = False
        self.gumbel_temperature = 1
        self.bidirectional = True

        self.model_name = opts.model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.idiom_compose = LatentComposition(config.hidden_size)
        self.compose_linear = nn.Linear(config.hidden_size * 3, config.hidden_size)
        emotion_hidden_size = config.hidden_size
        # self.compose_linear = nn.Sequential(nn.Linear(3 * config.hidden_size, config.hidden_size),
        #                                     nn.SELU(),
        #                                     nn.Dropout(p=0.1),
        #                                     nn.Linear(config.hidden_size, emotion_hidden_size),
        #                                     nn.SELU())

        # self.idiomacity_gate = GatedTanh(3 * config.hidden_size, config.hidden_size)

        # Idiom Predictor
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
        emotion_state = self.compose_linear(torch.cat([composed_states, composed_states_masked], dim=-1)).tanh()
        # emotion_state = self.idiomacity_gate(torch.cat([composed_states, composed_states_masked], dim=-1))

        # slide prediction
        sentiment_logits = self.sentiment_classifier(emotion_state)

        if compute_loss:
            sentiment_emotion_loss = self.loss_fct(sentiment_logits, targets[:, 1])
            return None, None, select_masks, sentiment_emotion_loss
        else:
            return None, None, select_masks, sentiment_logits


@register_model('chengyubert-slide-latent-emotion-masked')
class ChengyuBertSlideLatentEmotionMasked(BertPreTrainedModel):

    def __init__(self, config, opts):
        super().__init__(config)
        self.use_leaf_rnn = True
        self.intra_attention = False
        self.gumbel_temperature = 1
        self.bidirectional = True

        self.model_name = opts.model
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        emotion_hidden_size = config.hidden_size
        self.register_buffer('fine_emotions', torch.arange(21))
        self.emotion_embedding = nn.Embedding(21, emotion_hidden_size)
        # self.emotion_layernorm = nn.LayerNorm(emotion_hidden_size, eps=config.layer_norm_eps)

        self.idiom_compose = LatentComposition(config.hidden_size)
        self.compose_linear = nn.Linear(config.hidden_size * 3, emotion_hidden_size)
        # self.compose_linear = nn.Sequential(nn.Linear(3 * config.hidden_size, config.hidden_size),
        #                                      nn.SELU(),
        #                                      nn.Dropout(p=0.1),
        #                                      nn.Linear(config.hidden_size, emotion_hidden_size),
        #                                      nn.SELU())

        # Idiom Predictor
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

    def emotion(self, blank_states):
        # emotion_embeddings = self.emotion_layernorm(self.emotion_embedding(self.fine_emotions))
        emotion_embeddings = self.emotion_embedding(self.fine_emotions)
        logits = torch.einsum('bd,nd->bn', [blank_states, emotion_embeddings])  # (b, 256, 10)
        state = torch.einsum('bn,nd->bd', [logits.softmax(dim=-1), emotion_embeddings])  # (b, 256, 10)
        return logits, state

    def forward(self, input_ids, token_type_ids, attention_mask, positions, option_ids, gather_index,
                inputs_embeds=None, options_embeds=None, compute_loss=False, targets=None):
        n, batch_size, seq_len = input_ids.size()
        encoded_outputs = self.bert(input_ids.view(n * batch_size, seq_len),
                                    token_type_ids=token_type_ids.view(n * batch_size, seq_len),
                                    attention_mask=attention_mask.view(n * batch_size, seq_len))
        encoded_context = encoded_outputs[0].view(n, batch_size, seq_len, -1)[0]
        encoded_context_masked = encoded_outputs[0].view(n, batch_size, seq_len, -1)[1]

        idiom_length = (gather_index > 0).sum(1)
        gather_index = gather_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(input_ids)
        idiom_states = torch.gather(encoded_context, dim=1, index=gather_index)
        idiom_states_masked = torch.gather(encoded_context_masked, dim=1, index=gather_index)
        # idiom_states = encoded_context[[i for i in range(len(positions))], positions]  # [batch, hidden_state]

        composed_states, _, select_masks = self.idiom_compose(idiom_states, idiom_length)
        composed_states_masked, _ = idiom_states_masked.max(dim=1)
        emotion_state = self.compose_linear(torch.cat([composed_states, composed_states_masked], dim=-1))

        # slide prediction
        sentiment_logits = self.sentiment_classifier(emotion_state)

        if compute_loss:
            sentiment_emotion_loss = self.loss_fct(sentiment_logits, targets[:, 1])
            return (None, None, select_masks, sentiment_emotion_loss)
        else:
            return (None, None, select_masks, sentiment_logits)


@register_model('chengyubert-slide-latent-idiom-masked')
class ChengyuBertSlideLatentIdiomMasked(BertPreTrainedModel):

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

        self.compose_linear = nn.Linear(config.hidden_size * 3, config.hidden_size)
        # self.compose_linear = nn.Sequential(nn.Linear(3 * config.hidden_size, config.hidden_size),
        #                                     nn.SELU(),
        #                                     nn.Dropout(p=0.1),
        #                                     nn.Linear(config.hidden_size, config.hidden_size),
        #                                     nn.SELU())

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

        composed_states, _ = idiom_states.max(dim=1)
        composed_states_masked, _ = idiom_states_masked.max(dim=1)

        over_logits, idiom_attn_state = self.vocab(composed_states_masked)

        # slide prediction
        emotion_state = self.compose_linear(torch.cat([composed_states,
                                                       idiom_attn_state,
                                                       composed_states_masked], dim=-1)).tanh()

        sentiment_logits = self.sentiment_classifier(emotion_state)

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            over_loss = loss_fct(over_logits, targets[:, 0])
            sentiment_emotion_loss = self.loss_fct(sentiment_logits, targets[:, 1])
            return (None, over_loss, None, sentiment_emotion_loss)
        else:
            return (None, over_logits, None, sentiment_logits)


@register_model('chengyubert-slide-latent-idiom-masked-coattention')
class ChengyuBertSlideLatentIdiomMaskedCoAttention(BertPreTrainedModel):

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

        # self.idiom_compose = LatentComposition(config.hidden_size)
        self.compose_linear = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.affinity_linear = nn.Linear(config.hidden_size, config.hidden_size)
        # self.compose_linear = nn.Sequential(nn.Linear(4 * config.hidden_size, config.hidden_size),
        #                                     nn.SELU(),
        #                                     nn.Dropout(p=0.1),
        #                                     nn.Linear(config.hidden_size, config.hidden_size),
        #                                     nn.SELU())

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

        # composed_states, _, select_masks = self.idiom_compose(idiom_states, idiom_length)
        composed_states_masked, _ = idiom_states_masked.max(dim=1)
        over_logits, idiom_attn_state = self.vocab(composed_states_masked)

        L = idiom_states
        I = idiom_states_masked
        # I = encoded_context_masked

        idiom_mask = sequence_mask(idiom_length)

        AI = self.affinity_linear(I)

        # co attention
        L_T = torch.transpose(L, 1, 2)  # B x l x m + 1
        Z = torch.bmm(AI, L_T)  # L = B x n + 1 x m + 1

        # col max
        A_I_ = masked_softmax(Z.max(dim=2)[0], mask=idiom_mask)  # B x n + 1 x m + 1
        C_I = torch.einsum('bn,bnd->bd', [A_I_, I])

        # row max
        A_L_ = masked_softmax(Z.max(dim=1)[0], mask=idiom_mask)  # B x n + 1 x m + 1
        C_L = torch.einsum('bn,bnd->bd', [A_L_, L])

        # slide prediction
        emotion_state = self.compose_linear(torch.cat([C_L, C_I,
                                                       idiom_attn_state], dim=-1)).tanh()

        sentiment_logits = self.sentiment_classifier(emotion_state)

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            over_loss = loss_fct(over_logits, targets[:, 0])
            sentiment_emotion_loss = self.loss_fct(sentiment_logits, targets[:, 1])
            return (None, over_loss, None, sentiment_emotion_loss)
        else:
            return (None, over_logits, None, sentiment_logits)


@register_model('chengyubert-slide-latent-idiom-masked-coattention-full')
class ChengyuBertSlideLatentIdiomMaskedCoAttentionFull(BertPreTrainedModel):

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

        # self.idiom_compose = LatentComposition(config.hidden_size)
        self.compose_linear = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.affinity_linear = nn.Linear(config.hidden_size, config.hidden_size)
        # self.compose_linear = nn.Sequential(nn.Linear(4 * config.hidden_size, config.hidden_size),
        #                                     nn.SELU(),
        #                                     nn.Dropout(p=0.1),
        #                                     nn.Linear(config.hidden_size, config.hidden_size),
        #                                     nn.SELU())

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

        gather_index_unsqueezed = gather_index.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(input_ids)
        idiom_states = torch.gather(encoded_context, dim=1, index=gather_index_unsqueezed)

        gather_index_masked = gather_index_masked.unsqueeze(-1).expand(-1, -1, self.config.hidden_size).type_as(
            input_ids)
        idiom_states_masked = torch.gather(encoded_context_masked, dim=1, index=gather_index_masked)
        # idiom_states = encoded_context[[i for i in range(len(positions))], positions]  # [batch, hidden_state]

        # composed_states, _, select_masks = self.idiom_compose(idiom_states, idiom_length)
        composed_states_masked, _ = idiom_states_masked.max(dim=1)
        over_logits, idiom_attn_state = self.vocab(composed_states_masked)

        L = idiom_states
        I = encoded_context_masked
        # I = encoded_context_masked

        # idiom_length = (gather_index > 0).sum(1)
        # idiom_mask = sequence_mask(idiom_length)

        AI = self.affinity_linear(I)

        # co attention
        L_T = torch.transpose(L, 1, 2)  # B x l x m + 1
        Z = torch.bmm(AI, L_T)  # L = B x n + 1 x m + 1

        # col max
        A_I_ = masked_softmax(Z.max(dim=2)[0], mask=attention_mask[1])  # B x n + 1 x m + 1
        C_I = torch.einsum('bn,bnd->bd', [A_I_, I])

        # row max
        A_L_ = masked_softmax(Z.max(dim=1)[0], mask=torch.gather(attention_mask[0], dim=1,
                                                                 index=gather_index))  # B x n + 1 x m + 1
        C_L = torch.einsum('bn,bnd->bd', [A_L_, L])

        # slide prediction
        emotion_state = self.compose_linear(torch.cat([C_L, C_I,
                                                       idiom_attn_state], dim=-1)).tanh()

        sentiment_logits = self.sentiment_classifier(emotion_state)

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            over_loss = loss_fct(over_logits, targets[:, 0])
            sentiment_emotion_loss = self.loss_fct(sentiment_logits, targets[:, 1])
            return (None, over_loss, None, sentiment_emotion_loss)
        else:
            return (None, over_logits, None, sentiment_logits)


@register_model('chengyubert-slide-compose-latent-idiom-masked')
class ChengyuBertSlideComposeLatentIdiomMasked(BertPreTrainedModel):

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
        self.compose_linear = nn.Linear(config.hidden_size * 4, config.hidden_size)
        # self.compose_linear = nn.Sequential(nn.Linear(4 * config.hidden_size, config.hidden_size),
        #                                     nn.SELU(),
        #                                     nn.Dropout(p=0.1),
        #                                     nn.Linear(config.hidden_size, config.hidden_size),
        #                                     nn.SELU())

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

        # slide prediction
        emotion_state = self.compose_linear(torch.cat([composed_states,
                                                       idiom_attn_state,
                                                       composed_states_masked], dim=-1)).tanh()

        sentiment_logits = self.sentiment_classifier(emotion_state)

        if compute_loss:
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            over_loss = loss_fct(over_logits, targets[:, 0])
            sentiment_emotion_loss = self.loss_fct(sentiment_logits, targets[:, 1])
            return (None, over_loss, select_masks, sentiment_emotion_loss)
        else:
            return (None, over_logits, select_masks, sentiment_logits)
