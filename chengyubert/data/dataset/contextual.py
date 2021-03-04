import json
import random

import torch
from more_itertools import unzip
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from chengyubert.data import TxtTokLmdb, chengyu_process
from chengyubert.data.dataset import register_dataset


@register_dataset('chengyu-contextual')
class ChengyuContextualDataset(TxtTokLmdb):
    def __init__(self, db_dir, max_txt_len, opts, split):
        super().__init__(db_dir, max_txt_len)
        self.config = opts
        self.chengyu_vocab = chengyu_process(len_idiom_vocab=opts.len_idiom_vocab, annotation_dir='/annotations')
        self.id2idiom = {v: k for k, v in self.chengyu_vocab.items()}
        self.idiom_ids = list(range(opts.len_idiom_vocab))
        self.tokenizer = AutoTokenizer.from_pretrained(opts.pretrained_model_name_or_path)
        self.reverse_index = {int(k): v for k, v in json.load(open(f'{db_dir}/reverse_index.json')).items() if
                              int(k) < opts.len_idiom_vocab}
        self.allowed = set()
        [self.allowed.update(v) for _, v in self.reverse_index.items()]
        self.idiom_input_ids = self.tokenize_idioms()
        self.lens, self.ids, self.st_ed = self.get_ids_and_lens()
        self.window_size = opts.window_size

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

    def __getitem__(self, i):
        """
        [[txt, img1],
         [txt, img2]]
        """
        id_ = self.ids[i]
        st, ed = self.st_ed[i]
        example = self.db[id_]
        options = example['options']
        target = example['target']
        idiom = example['idiom']
        if len(options) == 0:
            options = random.sample(self.idiom_ids, k=7)
            if idiom not in options:
                options[-1] = idiom
            random.shuffle(options)
            target = options.index(idiom)

        context_ids = example['input_ids'][st: ed]
        idiom_start = context_ids.index(self.tokenizer.mask_token_id)
        idiom_input_ids = self.idiom_input_ids[idiom]
        width = len(idiom_input_ids)
        input_ids = [self.tokenizer.cls_token_id] + context_ids[:idiom_start] + idiom_input_ids + context_ids[
                                                                                                  idiom_start + 1:] + [
                        self.tokenizer.sep_token_id]
        assert len(input_ids) <= self.max_txt_len + width

        input_len = len(input_ids)
        position = idiom_start + 1
        left_start = position - self.window_size if self.window_size < position else 1
        left_end = position
        right_start = position + width
        right_end = right_start + self.window_size if right_start + self.window_size < input_len - 1 else input_len

        boundary_pairs = ((left_start, left_end), (right_start, right_end))

        token_type_ids = [0] * input_len
        attention_mask = [1] * input_len

        input_ids = torch.tensor(input_ids)
        token_type_ids = torch.tensor(token_type_ids)
        attention_mask = torch.tensor(attention_mask)
        return input_ids, token_type_ids, attention_mask, position, width, boundary_pairs, options, target

    @staticmethod
    def collate_fn(inputs):
        (input_ids, token_type_ids, attention_mask,
         positions, widths, boundary_pairs,
         options, targets) = map(list, unzip(inputs))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        attn_masks = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        width_max = max(widths)
        context_width_max = max(
            [left_end - left_start + right_end - right_start for ((left_start, left_end), (right_start, right_end)) in
             boundary_pairs])
        gather_index = torch.arange(0, width_max, dtype=torch.long).unsqueeze(0).repeat(len(inputs), 1).clone()
        context_gather_index = torch.arange(0, context_width_max, dtype=torch.long).unsqueeze(0).repeat(len(inputs),
                                                                                                        1).clone()
        for i, (p, w, ((left_start, left_end), (right_start, right_end))) in enumerate(
                zip(positions, widths, boundary_pairs)):
            gather_index.data[i, :w] = torch.arange(p, p + w, dtype=torch.long).data

            cw = left_end - left_start + right_end - right_start
            context_gather_index.data[i, :cw] = torch.cat([torch.arange(left_start, left_end, dtype=torch.long),
                                                           torch.arange(right_start, right_end, dtype=torch.long)]).data

        batch = {'input_ids': input_ids,
                 'token_type_ids': token_type_ids,
                 'attention_mask': attn_masks,
                 'gather_index': gather_index,
                 'context_gather_index': context_gather_index,
                 'positions': torch.tensor(positions).long(),
                 'option_ids': torch.tensor(options).long(),
                 'targets': torch.tensor(targets).long()}
        return batch


@register_dataset('chengyu-contextual-eval')
class ChengyuContextualEvalDataset(ChengyuContextualDataset):
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
        batch = ChengyuContextualDataset.collate_fn(batch)
        batch['qids'] = qids
        return batch
