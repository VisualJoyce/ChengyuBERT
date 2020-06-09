import json
from contextlib import contextmanager

import lmdb
import msgpack
import torch
from lz4.frame import compress, decompress
from more_itertools import unzip
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer


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


class ChengyuDataset(TxtTokLmdb):
    def __init__(self, db_dir, max_txt_len, opts):
        super().__init__(db_dir, max_txt_len)
        self.config = opts
        self.lens, self.ids = self.get_ids_and_lens()
        self.tokenizer = BertTokenizer.from_pretrained('/pretrain/wwm_ext')

    def __len__(self):
        return len(self.ids)

    def get_ids_and_lens(self):
        lens = []
        ids = []
        for id_, len_ in self.id2len.items():
            lens.append(len_)
            ids.append(id_)
        return lens, ids

    def __getitem__(self, i):
        """
        [[txt, img1],
         [txt, img2]]
        """
        id_ = self.ids[i]
        example = self.db[id_]
        position = example['position']
        options = example['options']
        target = example['target']
        input_ids = example['input_ids']

        half_length = self.max_txt_len // 2
        if position < half_length:  # cut at tail
            st = 0
            ed = min(len(input_ids), self.max_txt_len - 2)
        elif len(input_ids) - position < half_length:  # cut at head
            ed = len(input_ids)
            st = max(0, ed - (self.max_txt_len - 2))
        else:  # cut at both sides
            st = position + 2 - half_length
            ed = position + half_length

        position = input_ids[st:ed + 1].index(self.tokenizer.mask_token_id) + 1
        inputs = self.tokenizer.prepare_for_model(input_ids[st: ed],
                                                  pair_ids=None,
                                                  max_length=self.max_txt_len,
                                                  add_special_tokens=True,
                                                  return_tensors='pt',
                                                  truncation_strategy='do_not_truncate')
        input_ids, token_type_ids = inputs["input_ids"][0], inputs["token_type_ids"][0]
        attention_mask = [1] * input_ids.size(0)
        attention_mask = torch.tensor(attention_mask)
        return input_ids, token_type_ids, attention_mask, position, options, target


def chengyu_collate(inputs):
    (input_ids, token_type_ids, attention_mask, positions, options, targets) = map(list, unzip(inputs))

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    attn_masks = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    batch = {'input_ids': input_ids,
             'token_type_ids': token_type_ids,
             'attention_mask': attn_masks,
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
