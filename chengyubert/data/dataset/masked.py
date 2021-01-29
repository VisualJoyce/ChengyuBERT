import json
import random

import torch
from more_itertools import unzip
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from chengyubert.data import TxtTokLmdb, chengyu_process
from chengyubert.data.dataset import register_dataset


@register_dataset('chengyu-masked')
class ChengyuMaskedDataset(TxtTokLmdb):
    def __init__(self, db_dir, max_txt_len, opts):
        super().__init__(db_dir, max_txt_len)
        self.config = opts
        self.chengyu_vocab = chengyu_process(len_idiom_vocab=opts.len_idiom_vocab, annotation_dir='/annotations')
        self.idiom_ids = list(range(opts.len_idiom_vocab))
        self.tokenizer = AutoTokenizer.from_pretrained(opts.pretrained_model_name_or_path)
        self.reverse_index = {int(k): v for k, v in json.load(open(f'{db_dir}/reverse_index.json')).items() if
                              int(k) < opts.len_idiom_vocab}
        self.allowed = set()
        [self.allowed.update(v) for _, v in self.reverse_index.items()]
        self.lens, self.ids, self.st_ed = self.get_ids_and_lens()

    def __len__(self):
        return len(self.ids)

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
        if len(options) == 0:
            idiom = example['idiom']
            options = random.sample(self.idiom_ids, k=7)
            if idiom not in options:
                options[-1] = idiom
            random.shuffle(options)
            target = options.index(idiom)

        context_ids = example['input_ids'][st: ed]
        if hasattr(self.config, 'structured') and self.config.structured:
            idiom_start = context_ids.index(self.tokenizer.mask_token_id)
            for _ in range(3):
                context_ids.insert(idiom_start, self.tokenizer.mask_token_id)

            if len(context_ids) > self.max_txt_len - 2:
                half_length = self.max_txt_len // 2
                pop_length = len(context_ids) - (self.max_txt_len - 2)
                if idiom_start > half_length:
                    context_ids = context_ids[pop_length:]
                else:
                    context_ids = context_ids[:-pop_length]

        input_ids = [self.tokenizer.cls_token_id] + context_ids + [self.tokenizer.sep_token_id]
        assert len(input_ids) <= self.max_txt_len

        position = input_ids.index(self.tokenizer.mask_token_id)
        token_type_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)

        input_ids = torch.tensor(input_ids)
        token_type_ids = torch.tensor(token_type_ids)
        attention_mask = torch.tensor(attention_mask)
        return input_ids, token_type_ids, attention_mask, position, options, target

    @staticmethod
    def collate_fn(inputs):
        (input_ids, token_type_ids, attention_mask, positions, options, targets) = map(list, unzip(inputs))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        attn_masks = pad_sequence(attention_mask, batch_first=True, padding_value=0)

        lengths = attn_masks.sum(-1).long()
        width = 5
        span = 2 * width + 4
        gather_index = torch.arange(0, span, dtype=torch.long).unsqueeze(0).repeat(len(inputs), 1).clone()
        for i, (p, l) in enumerate(zip(positions, lengths)):
            if p <= width:
                left, right = 1, 1 + span
            elif p + 4 + width >= l:
                left, right = l - span, l
            else:
                left, right = p - width, p + 4 + width
            gather_index.data[i, :] = torch.arange(left, right, dtype=torch.long).data

        batch = {'input_ids': input_ids,
                 'token_type_ids': token_type_ids,
                 'attention_mask': attn_masks,
                 'gather_index': gather_index,
                 'positions': torch.tensor(positions).long(),
                 'option_ids': torch.tensor(options).long(),
                 'targets': torch.tensor(targets).long()}
        return batch


@register_dataset('chengyu-masked-eval')
class ChengyuMaskedEvalDataset(ChengyuMaskedDataset):
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
        batch = ChengyuMaskedDataset.collate_fn(batch)
        batch['qids'] = qids
        return batch


def judge(pred_file, answer_file):
    if isinstance(pred_file, str):
        pred = open(pred_file).readlines()
    else:
        pred = pred_file.readlines()

    ans = open(answer_file).readlines()
    assert len(pred) == len(ans)

    ans_dict = {}
    for line in ans:
        line = line.strip().split(',')
        ans_dict[line[0]] = int(line[1])

    pred_dict = {}
    for line in pred:
        line = line.strip().split(',')
        pred_dict[line[0]] = int(line[1])

    cnt = 0
    acc = 0
    for key in ans_dict:
        assert key in pred_dict
        cnt += 1
        if ans_dict[key] == pred_dict[key]:
            acc += 1

    return acc / cnt


def judge_by_idiom(pred_file, id2idiom, out_file=None):
    if isinstance(pred_file, str):
        pred = open(pred_file).readlines()
    else:
        pred = pred_file.readlines()

    cnt = 0
    mrr_total = 0
    mrr_dict = {}
    for line in pred:
        qid, ans, mrr, target = line.strip().split(',')
        target = int(target)
        mrr_dict.setdefault(target, [])
        mrr_dict[target].append(float(mrr))
        cnt += 1
        mrr_total += float(mrr)

    if out_file is not None:
        with open(out_file, 'w') as f:
            for idiom_id, mrrs in mrr_dict.items():
                if len(mrrs) > 0:
                    mean_mrr = sum(mrrs) / len(mrrs)
                    if idiom_id % 1000 == 0:
                        print(id2idiom[idiom_id], mean_mrr)
                else:
                    mean_mrr = 0
                f.write(f'{id2idiom[idiom_id]},{mean_mrr:.3f}\n')
    return mrr_total / cnt
