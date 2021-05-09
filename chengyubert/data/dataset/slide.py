import json
import operator
import random
from abc import abstractmethod
from functools import reduce

import torch
from more_itertools import unzip
from torch.nn.utils.rnn import pad_sequence

from chengyubert.data import IdiomsLmdb
from chengyubert.data.dataset import register_dataset


class ChengyuSlideDataset(IdiomsLmdb):
    def __init__(self, split, max_txt_len, opts):
        super().__init__(split, max_txt_len, opts)
        # load labelled idioms for the split
        with open(f'{self.db_dir}/{"dev" if split == "val" else split}.json') as f:
            self.filtered = json.load(f)
        with open(f'{self.db_dir}/span_idiom_mapping.json') as f:
            self.span_idiom_mapping = json.load(f)
        self.unlabeled = []
        self.allowed, self.reverse_index = self.get_allowed_examples(split, opts)

        self.idiom_input_ids = self.tokenize_idioms()
        self.lens, self.ids, self.st_ed = self.get_ids_and_lens()

    @abstractmethod
    def get_allowed_examples(self, split, opts):
        raise NotImplementedError

    def __len__(self):
        return len(self.ids)

    def tokenize_idioms(self):
        idiom_ids = {}
        for k in self.allowed:
            idiom = self.span_idiom_mapping[k]
            tokens = self.tokenizer.tokenize(idiom)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            idiom_ids[k] = input_ids
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

    def _decide_target(self, idiom, idx):
        target = [idx, -100]
        if idiom in self.sentiments and idiom in self.filtered:
            sentiment = self.sentiments[idiom]
            target = [idx, sentiment]
        return target


@register_dataset('chengyu-slide-compose-only')
class ChengyuSlideComposeOnlyDataset(ChengyuSlideDataset):
    def __init__(self, split, max_txt_len, opts):
        super().__init__(split, max_txt_len, opts)

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
        return allowed, reverse_index

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
        idiom_input_ids = self.idiom_input_ids[id_]
        idiom_len = len(idiom_input_ids)
        # target = idiom

        idx = -100 if idiom not in self.enlarged_candidates else self.enlarged_candidates.index(idiom)
        target = self._decide_target(idiom, idx)

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


@register_dataset('chengyu-slide-compose-only-eval')
class ChengyuSlideComposeOnlyEvalDataset(ChengyuSlideComposeOnlyDataset):
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
        batch = ChengyuSlideComposeOnlyDataset.collate_fn(batch)
        batch['qids'] = qids
        return batch


@register_dataset('chengyu-slide-compose-only-masked')
class ChengyuSlideComposeOnlyMaskedDataset(ChengyuSlideDataset):
    def __init__(self, split, max_txt_len, opts):
        super().__init__(split, max_txt_len, opts)
        self.allowed, self.reverse_index = self.get_allowed_examples(split, opts)

        if split == 'train':
            if not opts.use_unlabeled:
                self.enlarged_candidates = [i for i in range(opts.len_idiom_vocab) if i in self.filtered]
            else:
                # all the idioms without Slide labels are all considered
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
                    if opts.use_unlabeled:
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

        idx = -100 if idiom not in self.enlarged_candidates else self.enlarged_candidates.index(idiom)
        target = self._decide_target(idiom, idx)

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


@register_dataset('chengyu-slide-compose-only-masked-eval')
class ChengyuSlideComposeOnlyMaskedEvalDataset(ChengyuSlideComposeOnlyMaskedDataset):
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
        batch = ChengyuSlideComposeOnlyMaskedDataset.collate_fn(batch)
        batch['qids'] = qids
        return batch
