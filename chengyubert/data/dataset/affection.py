import json
import operator
import random
from functools import reduce

import torch
from more_itertools import unzip
from torch.nn.utils.rnn import pad_sequence

from chengyubert.data import ChengyuLmdb, calo_process
from chengyubert.data.dataset import register_dataset


# @register_dataset('chengyu-affection')
# class ChengyuCALODataset(ChengyuLmdb):
#     def __init__(self, split, max_txt_len, opts):
#         super().__init__(split, max_txt_len, opts)
#         self.calo_vocab = calo_process(self.chengyu_vocab, self.config.calo_file)
#
#         self.reverse_index = {}
#         for k, v in json.load(open(f'{self.db_dir}/reverse_index.json')).items():
#             k = int(k)
#             if k < opts.len_idiom_vocab:
#                 if split == 'train' and k not in self.calo_vocab:
#                     continue
#                 self.reverse_index[k] = v
#
#         self.allowed = set()
#         [self.allowed.update(v) for _, v in self.reverse_index.items()]
#
#         self.idiom_input_ids = self.tokenize_idioms()
#         self.lens, self.ids, self.st_ed = self.get_ids_and_lens()
#
#     def __len__(self):
#         return len(self.ids)
#
#     def tokenize_idioms(self):
#         idiom_ids = {}
#         for idiom, idiom_id in self.chengyu_vocab.items():
#             tokens = self.tokenizer.tokenize(idiom)
#             input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
#             idiom_ids[idiom_id] = input_ids
#         return idiom_ids
#
#     def get_ids_and_lens(self):
#         lens = []
#         ids = []
#         st_ed = []
#         for id_, len_ in self.id2len.items():
#             if id_ not in self.allowed:
#                 continue
#             example = self.db[id_]
#             position = example['position']
#             input_ids = example['input_ids']
#             half_length = self.max_txt_len // 2
#             if position < half_length:  # cut at tail
#                 st = 0
#                 ed = min(len(input_ids) + 1, self.max_txt_len - 2)
#             elif len(input_ids) - position < half_length:  # cut at head
#                 ed = len(input_ids)
#                 st = max(0, ed - (self.max_txt_len - 2))
#             else:  # cut at both sides
#                 st = position - (half_length - 2)
#                 ed = position + half_length
#
#             assert ed - st <= self.max_txt_len - 2
#             st_ed.append((st, ed))
#             lens.append(ed - st + 1)
#             ids.append(id_)
#         return lens, ids, st_ed
#
#     def __getitem__(self, i):
#         """
#         [[txt, img1],
#          [txt, img2]]
#         """
#         id_ = self.ids[i]
#         st, ed = self.st_ed[i]
#         example = self.db[id_]
#         options = example['options']
#         # target = example['target']
#         idiom = example['idiom']
#         if len(options) == 0:
#             options = random.sample(self.idiom_ids, k=7)
#             if idiom not in options:
#                 options[-1] = idiom
#             random.shuffle(options)
#         #     target = options.index(idiom)
#
#         context_ids = example['input_ids'][st: ed]
#         idiom_start = context_ids.index(self.tokenizer.mask_token_id)
#         idiom_input_ids = self.idiom_input_ids[idiom]
#         idiom_len = len(idiom_input_ids)
#
#         target = idiom
#         if self.split == 'train':
#             # We choose to randomly mask idiom input
#             k = random.randint(0, idiom_len)
#             for i in random.sample(range(0, idiom_len), k=k):
#                 idiom_input_ids[i] = self.tokenizer.mask_token_id
#
#             if k <= 2:
#                 target = -100
#
#         if idiom in self.calo_vocab:
#             affections = self.calo_vocab[idiom]
#             target = [
#                 target,
#                 affections['coarse_emotion'],
#                 affections['fine_emotion'],
#                 affections['sentiment'],
#                 affections['strength'],
#             ]
#         else:
#             target = [
#                 target, -100, -100, -100, 0
#             ]
#
#         input_ids = reduce(operator.add, [
#             [self.tokenizer.cls_token_id],
#             context_ids[:idiom_start],
#             idiom_input_ids,
#             context_ids[idiom_start + 1:],
#             [self.tokenizer.sep_token_id]])
#         assert len(input_ids) <= self.max_txt_len + idiom_len
#
#         position = idiom_start + 1
#
#         token_type_ids = [0] * len(input_ids)
#         attention_mask = [1] * len(input_ids)
#
#         input_ids = torch.tensor(input_ids)
#         token_type_ids = torch.tensor(token_type_ids)
#         attention_mask = torch.tensor(attention_mask)
#         return input_ids, token_type_ids, attention_mask, position, idiom_len, options, target
#
#     @staticmethod
#     def collate_fn(inputs):
#         (input_ids, token_type_ids, attention_mask, positions, widths, options, targets) = map(list, unzip(inputs))
#
#         input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
#         token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
#         attn_masks = pad_sequence(attention_mask, batch_first=True, padding_value=0)
#
#         width_max = max(widths)
#         gather_index = torch.arange(0, width_max, dtype=torch.long).unsqueeze(0).repeat(len(inputs), 1).clone()
#         for i, (p, w) in enumerate(zip(positions, widths)):
#             gather_index.data[i, :w] = torch.arange(p, p + w, dtype=torch.long).data
#
#         batch = {'input_ids': input_ids,
#                  'token_type_ids': token_type_ids,
#                  'attention_mask': attn_masks,
#                  'gather_index': gather_index,
#                  'positions': torch.tensor(positions).long(),
#                  'option_ids': torch.tensor(options).long(),
#                  'targets': torch.tensor(targets).long()}
#         return batch
#
#
# @register_dataset('chengyu-affection-eval')
# class ChengyuCALOEvalDataset(ChengyuCALODataset):
#     def __getitem__(self, i):
#         qid = self.ids[i]
#         tensors = super().__getitem__(i)
#         return (qid, *tensors)
#
#     @staticmethod
#     def collate_fn(inputs):
#         qids, batch = [], []
#         for id_, *tensors in inputs:
#             qids.append(id_)
#             batch.append(tensors)
#         batch = ChengyuCALODataset.collate_fn(batch)
#         batch['qids'] = qids
#         return batch
#
#
# @register_dataset('chengyu-affection-mask')
# class ChengyuCALOMaskDataset(ChengyuLmdb):
#     def __init__(self, split, max_txt_len, opts):
#         super().__init__(split, max_txt_len, opts)
#         self.calo_vocab = calo_process(self.chengyu_vocab, self.config.calo_file)
#
#         self.reverse_index = {}
#         for k, v in json.load(open(f'{self.db_dir}/reverse_index.json')).items():
#             k = int(k)
#             if k < opts.len_idiom_vocab:
#                 if split == 'train' and k not in self.calo_vocab:
#                     continue
#                 self.reverse_index[k] = v
#
#         self.allowed = set()
#         [self.allowed.update(v) for _, v in self.reverse_index.items()]
#
#         self.idiom_input_ids = self.tokenize_idioms()
#         self.lens, self.ids, self.st_ed = self.get_ids_and_lens()
#
#     def __len__(self):
#         return len(self.ids)
#
#     def tokenize_idioms(self):
#         idiom_ids = {}
#         for idiom, idiom_id in self.chengyu_vocab.items():
#             tokens = self.tokenizer.tokenize(idiom)
#             input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
#             idiom_ids[idiom_id] = input_ids
#         return idiom_ids
#
#     def get_ids_and_lens(self):
#         lens = []
#         ids = []
#         st_ed = []
#         for id_, len_ in self.id2len.items():
#             if id_ not in self.allowed:
#                 continue
#             example = self.db[id_]
#             position = example['position']
#             input_ids = example['input_ids']
#             half_length = self.max_txt_len // 2
#             if position < half_length:  # cut at tail
#                 st = 0
#                 ed = min(len(input_ids) + 1, self.max_txt_len - 2)
#             elif len(input_ids) - position < half_length:  # cut at head
#                 ed = len(input_ids)
#                 st = max(0, ed - (self.max_txt_len - 2))
#             else:  # cut at both sides
#                 st = position - (half_length - 2)
#                 ed = position + half_length
#
#             assert ed - st <= self.max_txt_len - 2
#             st_ed.append((st, ed))
#             lens.append(ed - st + 1)
#             ids.append(id_)
#         return lens, ids, st_ed
#
#     def __getitem__(self, i):
#         id_ = self.ids[i]
#         st, ed = self.st_ed[i]
#         example = self.db[id_]
#         options = example['options']
#         idiom = example['idiom']
#         if len(options) == 0:
#             options = random.sample(self.idiom_ids, k=7)
#             if idiom not in options:
#                 options[-1] = idiom
#             random.shuffle(options)
#         #     target = options.index(idiom)
#
#         context_ids = example['input_ids'][st: ed]
#         idiom_start = context_ids.index(self.tokenizer.mask_token_id)
#         idiom_input_ids = self.idiom_input_ids[idiom]
#         idiom_len = len(idiom_input_ids)
#         target = idiom
#
#         idiom_masked_input_ids = [self.tokenizer.mask_token_id] * idiom_len
#
#         if idiom in self.calo_vocab:
#             affections = self.calo_vocab[idiom]
#             target = [
#                 target,
#                 affections['coarse_emotion'],
#                 affections['fine_emotion'],
#                 affections['sentiment'],
#                 affections['strength'],
#             ]
#         else:
#             target = [
#                 target, -100, -100, -100, 0
#             ]
#
#         input_ids = reduce(operator.add, [
#             [self.tokenizer.cls_token_id],
#             context_ids[:idiom_start],
#             idiom_input_ids,
#             context_ids[idiom_start + 1:],
#             [self.tokenizer.sep_token_id]])
#         input_masked_ids = reduce(operator.add, [
#             [self.tokenizer.cls_token_id],
#             context_ids[:idiom_start],
#             idiom_masked_input_ids,
#             context_ids[idiom_start + 1:],
#             [self.tokenizer.sep_token_id]])
#         assert len(input_ids) <= self.max_txt_len + idiom_len
#
#         position = idiom_start + 1
#
#         token_type_ids = [0] * len(input_ids)
#         attention_mask = [1] * len(input_ids)
#
#         input_ids = torch.tensor(input_ids)
#         input_masked_ids = torch.tensor(input_masked_ids)
#         token_type_ids = torch.tensor(token_type_ids)
#         attention_mask = torch.tensor(attention_mask)
#         return (input_ids, input_masked_ids), token_type_ids, attention_mask, position, idiom_len, options, target
#
#     @staticmethod
#     def collate_fn(inputs):
#         (input_ids_tuple, token_type_ids, attention_mask, positions, widths, options, targets) = map(list,
#                                                                                                      unzip(inputs))
#
#         input_ids = pad_sequence([item[0] for item in input_ids_tuple], batch_first=True, padding_value=0)
#         input_masked_ids = pad_sequence([item[1] for item in input_ids_tuple], batch_first=True, padding_value=0)
#         token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
#         attn_masks = pad_sequence(attention_mask, batch_first=True, padding_value=0)
#
#         width_max = max(widths)
#         gather_index = torch.arange(0, width_max, dtype=torch.long).unsqueeze(0).repeat(len(inputs), 1).clone()
#         for i, (p, w) in enumerate(zip(positions, widths)):
#             gather_index.data[i, :w] = torch.arange(p, p + w, dtype=torch.long).data
#
#         batch = {'input_ids': torch.stack([input_ids, input_masked_ids]),
#                  'token_type_ids': torch.stack([token_type_ids, token_type_ids]),
#                  'attention_mask': torch.stack([attn_masks, attn_masks]),
#                  'gather_index': gather_index,
#                  'positions': torch.tensor(positions).long(),
#                  'option_ids': torch.tensor(options).long(),
#                  'targets': torch.tensor(targets).long()}
#         return batch
#
#
# @register_dataset('chengyu-affection-mask-eval')
# class ChengyuCALOMaskEvalDataset(ChengyuCALOMaskDataset):
#     def __getitem__(self, i):
#         qid = self.ids[i]
#         tensors = super().__getitem__(i)
#         return (qid, *tensors)
#
#     @staticmethod
#     def collate_fn(inputs):
#         qids, batch = [], []
#         for id_, *tensors in inputs:
#             qids.append(id_)
#             batch.append(tensors)
#         batch = ChengyuCALOMaskDataset.collate_fn(batch)
#         batch['qids'] = qids
#         return batch


@register_dataset('chengyu-affection-compose-only')
class ChengyuCALOComposeOnlyDataset(ChengyuLmdb):
    def __init__(self, split, max_txt_len, opts):
        super().__init__(split, max_txt_len, opts)
        # load labelled idioms for the split
        with open(f'{self.db_dir}/{"dev" if split == "val" else split}.json') as f:
            self.filtered = json.load(f)
        self.calo_vocab = calo_process(self.chengyu_vocab, self.config.calo_file)
        self.allowed = self.get_allowed_examples(split, opts)

        self.idiom_input_ids = self.tokenize_idioms()
        self.lens, self.ids, self.st_ed = self.get_ids_and_lens()

        if split == 'train':
            self.enlarged_candidates = [i for i in range(opts.len_idiom_vocab) if i in self.filtered]
            opts.enlarged_candidates = self.enlarged_candidates
        else:
            self.enlarged_candidates = opts.enlarged_candidates

    def get_allowed_examples(self, split, opts):
        """
        For training dataset:
        1) we can choose whether to add unlabelled data
        2) we can choose whether to add labelled but not in training
        :param opts:
        :return:
        """
        reverse_index = {}
        for k, v in json.load(open(f'{self.db_dir}/reverse_index.json')).items():
            k = int(k)
            if k < opts.len_idiom_vocab:
                if k in self.filtered:
                    reverse_index[k] = v

        allowed = set()
        [allowed.update(v) for _, v in reverse_index.items()]
        return allowed

    def __len__(self):
        return len(self.ids)

    def tokenize_idioms(self):
        idiom_ids = {}
        for idiom, idiom_id in self.chengyu_vocab.items():
            tokens = self.tokenizer.tokenize(idiom)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            idiom_ids[idiom_id] = input_ids
        return idiom_ids

    def get_ids_and_lens(self):
        lens = []
        ids = []
        st_ed = []
        for id_, len_ in self.id2len.items():
            if id_ not in self.allowed:
                continue

            example = self.db[id_]
            position = example['position']
            input_ids = example['input_ids']
            half_length = self.max_txt_len // 2
            if position < half_length:  # cut at tail
                st = 0
                ed = min(len(input_ids) + 1, self.max_txt_len - 2)
            elif len(input_ids) - position < half_length:  # cut at head
                ed = len(input_ids)
                st = max(0, ed - (self.max_txt_len - 2))
            else:  # cut at both sides
                st = position - (half_length - 2)
                ed = position + half_length

            assert ed - st <= self.max_txt_len - 2
            st_ed.append((st, ed))
            lens.append(ed - st + 1)
            ids.append(id_)
        return lens, ids, st_ed

    def _decide_target(self, idiom):
        if idiom in self.enlarged_candidates:
            idx = self.enlarged_candidates.index(idiom)
        else:
            idx = -100
        target = [
            idx, -100, -100, -100, 0
        ]
        if idiom in self.calo_vocab and idiom in self.filtered:
            affections = self.calo_vocab[idiom][0]
            target = [
                idx,
                affections['coarse_emotion'],
                affections['fine_emotion'],
                affections['sentiment'],
                affections['strength'],
            ]
        return target

    def __getitem__(self, i):
        id_ = self.ids[i]
        st, ed = self.st_ed[i]
        example = self.db[id_]
        options = example['options']
        idiom = example['idiom']
        if len(options) == 0:
            options = random.sample(self.idiom_ids, k=7)
            if idiom not in options:
                options[-1] = idiom
            random.shuffle(options)
        #     target = options.index(idiom)

        context_ids = example['input_ids'][st: ed]
        idiom_start = context_ids.index(self.tokenizer.mask_token_id)
        idiom_input_ids = self.idiom_input_ids[idiom]
        idiom_len = len(idiom_input_ids)
        # target = idiom

        target = self._decide_target(idiom)

        input_ids = reduce(operator.add, [
            [self.tokenizer.cls_token_id],
            context_ids[:idiom_start],
            idiom_input_ids,
            context_ids[idiom_start + 1:],
            [self.tokenizer.sep_token_id]])
        assert len(input_ids) <= self.max_txt_len + idiom_len

        position = idiom_start + 1

        token_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)

        input_ids = torch.tensor(input_ids)
        token_type_ids = torch.tensor(token_type_ids)
        attention_mask = torch.tensor(attention_mask)
        return input_ids, token_type_ids, attention_mask, position, idiom_len, options, target

    @staticmethod
    def collate_fn(inputs):
        (input_ids, token_type_ids, attention_mask, positions, widths, options, targets) = map(list,
                                                                                               unzip(inputs))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        attn_masks = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        width_max = max(widths)
        gather_index = torch.arange(0, width_max, dtype=torch.long).unsqueeze(0).repeat(len(inputs), 1).clone()
        for i, (p, w) in enumerate(zip(positions, widths)):
            gather_index.data[i, :w] = torch.arange(p, p + w, dtype=torch.long).data

        batch = {'input_ids': input_ids,
                 'token_type_ids': token_type_ids,
                 'attention_mask': attn_masks,
                 'gather_index': gather_index,
                 'positions': torch.tensor(positions).long(),
                 'option_ids': torch.tensor(options).long(),
                 'targets': torch.tensor(targets).long()}
        return batch


@register_dataset('chengyu-affection-compose-only-eval')
class ChengyuCALOComposeOnlyEvalDataset(ChengyuCALOComposeOnlyDataset):
    def __getitem__(self, i):
        qid = self.ids[i]
        tensors = super().__getitem__(i)
        return (qid, *tensors)

    @staticmethod
    def collate_fn(inputs):
        qids, batch = [], []
        for id_, *tensors in inputs:
            qids.append(id_)
            batch.append(tensors)
        batch = ChengyuCALOComposeOnlyDataset.collate_fn(batch)
        batch['qids'] = qids
        return batch


@register_dataset('chengyu-affection-compose-only-masked')
class ChengyuCALOComposeOnlyMaskedDataset(ChengyuLmdb):
    def __init__(self, split, max_txt_len, opts):
        super().__init__(split, max_txt_len, opts)
        with open(f'{self.db_dir}/{"dev" if split == "val" else split}.json') as f:
            self.filtered = json.load(f)
        with open(f'{self.db_dir}/unlabelled.json') as f:
            self.unlabeled = json.load(f)
        self.use_unlabeled = opts.use_unlabeled
        self.calo_vocab = calo_process(self.chengyu_vocab, self.config.calo_file)
        self.allowed, self.reverse_index = self.get_allowed_examples(split, opts)

        self.idiom_input_ids = self.tokenize_idioms()
        self.lens, self.ids, self.st_ed = self.get_ids_and_lens()

        if split == 'train':
            if not self.use_unlabeled:
                self.enlarged_candidates = [i for i in range(opts.len_idiom_vocab) if i in self.filtered]
            else:
                # all the idioms without CALO labels are all considered
                self.enlarged_candidates = [i for i in range(opts.len_idiom_vocab) if i in self.reverse_index]
            opts.enlarged_candidates = self.enlarged_candidates
        else:
            self.enlarged_candidates = opts.enlarged_candidates

    def get_allowed_examples(self, split, opts):
        """
        For training dataset:
        1) we can choose whether to add unlabelled data
        2) we can choose whether to add labelled but not in training
        :param opts:
        :return:
        """
        reverse_index = {}
        for k, v in json.load(open(f'{self.db_dir}/reverse_index.json')).items():
            k = int(k)
            if k < opts.len_idiom_vocab:
                if split == 'train':
                    if self.use_unlabeled:
                        if k in self.filtered or k in self.unlabeled:
                            reverse_index[k] = v
                    else:
                        if k in self.filtered:
                            reverse_index[k] = v
                else:
                    if k in self.filtered:
                        reverse_index[k] = v

        allowed = set()
        [allowed.update(v) for _, v in reverse_index.items()]
        return allowed, reverse_index

    def __len__(self):
        return len(self.ids)

    def tokenize_idioms(self):
        idiom_ids = {}
        for idiom, idiom_id in self.chengyu_vocab.items():
            tokens = self.tokenizer.tokenize(idiom)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            idiom_ids[idiom_id] = input_ids
        return idiom_ids

    def get_ids_and_lens(self):
        lens = []
        ids = []
        st_ed = []
        for id_, len_ in self.id2len.items():
            if id_ not in self.allowed:  # not in allowed idioms
                continue

            example = self.db[id_]
            position = example['position']
            input_ids = example['input_ids']
            half_length = self.max_txt_len // 2
            if position < half_length:  # cut at tail
                st = 0
                ed = min(len(input_ids) + 1, self.max_txt_len - 2)
            elif len(input_ids) - position < half_length:  # cut at head
                ed = len(input_ids)
                st = max(0, ed - (self.max_txt_len - 2))
            else:  # cut at both sides
                st = position - (half_length - 2)
                ed = position + half_length

            assert ed - st <= self.max_txt_len - 2
            st_ed.append((st, ed))
            lens.append(ed - st + 1)
            ids.append(id_)
        return lens, ids, st_ed

    def _decide_target(self, idiom):
        if idiom in self.enlarged_candidates:
            idx = self.enlarged_candidates.index(idiom)
        else:
            idx = -100
        target = [
            idx, -100, -100, -100, 0
        ]
        if idiom in self.calo_vocab and idiom in self.filtered:
            affections = self.calo_vocab[idiom][0]
            target = [
                idx,
                affections['coarse_emotion'],
                affections['fine_emotion'],
                affections['sentiment'],
                affections['strength'],
            ]
        return target

    def __getitem__(self, i):
        id_ = self.ids[i]
        st, ed = self.st_ed[i]
        example = self.db[id_]
        options = example['options']
        idiom = example['idiom']
        if len(options) == 0:
            options = random.sample(self.idiom_ids, k=7)
            if idiom not in options:
                options[-1] = idiom
            random.shuffle(options)
        #     target = options.index(idiom)

        context_ids = example['input_ids'][st: ed]
        idiom_start = context_ids.index(self.tokenizer.mask_token_id)
        idiom_input_ids = self.idiom_input_ids[idiom]
        idiom_len = len(idiom_input_ids)
        # target = idiom

        idiom_masked_input_ids = [self.tokenizer.mask_token_id] * idiom_len
        target = self._decide_target(idiom)

        input_ids = reduce(operator.add, [
            [self.tokenizer.cls_token_id],
            context_ids[:idiom_start],
            idiom_input_ids,
            context_ids[idiom_start + 1:],
            [self.tokenizer.sep_token_id]])
        input_masked_ids = reduce(operator.add, [
            [self.tokenizer.cls_token_id],
            context_ids[:idiom_start],
            idiom_masked_input_ids,
            context_ids[idiom_start + 1:],
            [self.tokenizer.sep_token_id]])
        assert len(input_ids) <= self.max_txt_len + idiom_len

        position = idiom_start + 1

        token_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)

        input_ids = torch.tensor(input_ids)
        input_masked_ids = torch.tensor(input_masked_ids)
        token_type_ids = torch.tensor(token_type_ids)
        attention_mask = torch.tensor(attention_mask)
        return (input_ids, input_masked_ids), token_type_ids, attention_mask, position, idiom_len, options, target

    @staticmethod
    def collate_fn(inputs):
        (input_ids_tuple, token_type_ids, attention_mask, positions, widths, options, targets) = map(list,
                                                                                                     unzip(inputs))

        input_ids = pad_sequence([item[0] for item in input_ids_tuple], batch_first=True, padding_value=0)
        input_masked_ids = pad_sequence([item[1] for item in input_ids_tuple], batch_first=True, padding_value=0)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        attn_masks = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        width_max = max(widths)
        gather_index = torch.arange(0, width_max, dtype=torch.long).unsqueeze(0).repeat(len(inputs), 1).clone()
        for i, (p, w) in enumerate(zip(positions, widths)):
            gather_index.data[i, :w] = torch.arange(p, p + w, dtype=torch.long).data

        batch = {'input_ids': torch.stack([input_ids, input_masked_ids]),
                 'token_type_ids': torch.stack([token_type_ids, token_type_ids]),
                 'attention_mask': torch.stack([attn_masks, attn_masks]),
                 'gather_index': gather_index,
                 'positions': torch.tensor(positions).long(),
                 'option_ids': torch.tensor(options).long(),
                 'targets': torch.tensor(targets).long()}
        return batch


@register_dataset('chengyu-affection-compose-only-masked-eval')
class ChengyuCALOComposeOnlyMaskedEvalDataset(ChengyuCALOComposeOnlyMaskedDataset):
    def __getitem__(self, i):
        qid = self.ids[i]
        tensors = super().__getitem__(i)
        return (qid, *tensors)

    @staticmethod
    def collate_fn(inputs):
        qids, batch = [], []
        for id_, *tensors in inputs:
            qids.append(id_)
            batch.append(tensors)
        batch = ChengyuCALOComposeOnlyMaskedDataset.collate_fn(batch)
        batch['qids'] = qids
        return batch