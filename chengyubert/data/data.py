import json
import os
import sys
from contextlib import contextmanager

import lmdb
import msgpack
import pandas as pda
import torch
from lz4.frame import compress, decompress
from more_itertools import unzip
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

os.environ['TOKENIZERS_PARALLELISM'] = "false"


@contextmanager
def open_lmdb(db_dir, readonly=False):
    db = TxtLmdb(db_dir, readonly)
    try:
        yield db
    finally:
        del db


class TxtLmdb(object):
    def __init__(self, db_dir, readonly=True):
        self.readonly = readonly
        if readonly:
            # training
            self.env = lmdb.open(db_dir,
                                 readonly=True, create=False, lock=False,
                                 readahead=not _check_distributed())
            self.txn = self.env.begin(buffers=True)
            self.write_cnt = None
        else:
            # prepro
            self.env = lmdb.open(db_dir, readonly=False, create=True,
                                 map_size=4 * 1024 ** 4)
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0

    def __del__(self):
        if self.write_cnt:
            self.txn.commit()
        self.env.close()

    def __getitem__(self, key):
        return msgpack.loads(decompress(self.txn.get(key.encode('utf-8'))),
                             raw=False)

    def __setitem__(self, key, value):
        # NOTE: not thread safe
        if self.readonly:
            raise ValueError('readonly text DB')
        ret = self.txn.put(key.encode('utf-8'),
                           compress(msgpack.dumps(value, use_bin_type=True)))
        self.write_cnt += 1
        if self.write_cnt % 1000 == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)
            self.write_cnt = 0
        return ret


def _check_distributed():
    # try:
    #     dist = hvd.size() != hvd.local_size
    # except ValueError:
    #     # not using horovod
    #     dist = False
    # return dist
    return False


class TxtTokLmdb(object):
    def __init__(self, db_dir, max_txt_len=60):
        self.id2len = json.load(open(f'{db_dir}/id2len.json'))
        self.max_txt_len = max_txt_len
        self.db_dir = db_dir
        self.db = TxtLmdb(db_dir, readonly=True)

    def __getitem__(self, id_):
        txt_dump = self.db[id_]
        return txt_dump


def chengyu_process(len_idiom_vocab=sys.maxsize, annotation_dir='/annotations'):
    """
    Load Chengyu to vocab with explanation
    :return:
    """
    # load ChID idioms to first 3848
    chengyu_vocab = {each: i for i, each in enumerate(eval(open(f'{annotation_dir}/idiomList.txt').readline()))}

    # load Xinhua idioms
    chengyu_pretrain = pda.read_csv(f"{annotation_dir}/idioms_pretrain.json", sep='\t')
    chengyu_pretrain.fillna('', inplace=True)

    # read explanation for each idiom
    explanation = {}
    count = {}
    for item in chengyu_pretrain.itertuples():
        each = item.idiom
        count[each] = item.num
        explanation[each] = item.explanation

        # add extra Chengyu to vocab
        if each not in chengyu_vocab:
            chengyu_vocab[each] = len(chengyu_vocab)

    chengyu_vocab = {k: v for k, v in chengyu_vocab.items() if v < len_idiom_vocab}

    print("Total idioms: {}".format(len(chengyu_vocab)))

    return chengyu_vocab


class ChengyuDataset(TxtTokLmdb):
    def __init__(self, db_dir, max_txt_len, opts):
        super().__init__(db_dir, max_txt_len)
        self.config = opts
        self.chengyu_vocab = chengyu_process(len_idiom_vocab=opts.len_idiom_vocab, annotation_dir='/annotations')
        with open('/annotations/synonyms/chengyu_synonyms_recall_filter.json') as f:
            self.chengyu_synonyms_dict = json.load(f)
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


def chengyu_collate(inputs):
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


class ChengyuEvalDataset(ChengyuDataset):
    def __getitem__(self, i):
        qid = self.ids[i]
        tensors = super().__getitem__(i)
        return (qid, *tensors)


def chengyu_eval_collate(inputs):
    qids, batch = [], []
    for id_, *tensors in inputs:
        qids.append(id_)
        batch.append(tensors)
    batch = chengyu_collate(batch)
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
