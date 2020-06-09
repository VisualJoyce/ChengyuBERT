import json
from contextlib import contextmanager

import lmdb
import msgpack
import torch
from lz4.frame import compress, decompress
from more_itertools import unzip
from torch.nn.utils.rnn import pad_sequence


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


def id2len_func(max_txt_len, id2len):
    if max_txt_len == -1:
        return id2len
    else:
        return {
            id_: len_
            for id_, len_ in id2len.items()
            if len_ <= max_txt_len
        }


class TxtTokLmdb(object):
    def __init__(self, db_dir, max_txt_len=60):
        id2len = json.load(open(f'{db_dir}/id2len.json'))
        self.id2len = id2len_func(max_txt_len, id2len)
        self.db_dir = db_dir
        self.db = TxtLmdb(db_dir, readonly=True)

    def __getitem__(self, id_):
        txt_dump = self.db[id_]
        return txt_dump


class ChengyuDataset(TxtTokLmdb):
    def __init__(self, db_dir, max_txt_len, opts):
        super().__init__(db_dir, max_txt_len)
        self.config = opts
        txt_lens, self.ids = self.get_ids_and_lens()
        self.lens = [tl[1] for tl, id_ in zip(txt_lens, self.ids)]

    def get_ids_and_lens(self):
        lens = []
        ids = []
        for id_, len_ in self.db.id2len.items():
            lens.append(len_)
            ids.append(id_)
        return lens, ids

    def __getitem__(self, i):
        """
        [[txt, img1],
         [txt, img2]]
        """
        example = super().__getitem__(i)
        target = example[self.config.target]

        # text input
        input_ids = [self.txt_db.cls_] + example['input_2_ids'] + [self.txt_db.sep]
        input_ids = torch.tensor(input_ids)

        attn_masks = [1] * input_ids.size(0)
        attn_masks = torch.tensor(attn_masks)

        return (input_ids, input_concreteness,
                img_feat, img_pos_feat, img_depth_feat,
                region_masks, attn_masks,
                target, decode_ids, decode_attn_masks, target_ids)


def chengyu_collate(inputs):
    (input_ids, input_concreteness,
     img_feats, img_pos_feats, img_depth_feats,
     region_masks, attn_masks,
     targets, decode_ids, decode_attn_masks, target_ids) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0).expand_as(input_ids).clone()

    if all([item is not None for item in decode_ids]):
        decode_ids = pad_sequence(decode_ids, batch_first=True, padding_value=0)
        decode_position_ids = torch.arange(0, decode_ids.size(1), dtype=torch.long
                                           ).unsqueeze(0).expand_as(decode_ids).clone()
        decode_attn_masks = pad_sequence(decode_attn_masks, batch_first=True, padding_value=0)
        target_ids = pad_sequence(target_ids, batch_first=True, padding_value=-100)
    else:
        decode_ids, decode_position_ids, decode_attn_masks, target_ids = None, None, None, None

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    if all([item is not None for item in input_concreteness]):
        # inputs_concreteness = pad_tensors(input_concreteness, txt_lens)
        inputs_concreteness = pad_sequence(input_concreteness, batch_first=True, padding_value=0)
    else:
        inputs_concreteness = None

    bs, max_tl = input_ids.size()

    region_masks = pad_sequence(region_masks, batch_first=True, padding_value=0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    out_size = attn_masks.size(1)

    if max_tl < out_size:
        gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)
    else:
        gather_index = None

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'decode_ids': decode_ids,
             'decode_position_ids': decode_position_ids,
             'decode_attn_masks': decode_attn_masks,
             'target_ids': target_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'region_masks': region_masks,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'img_type_ids': None,
             'inputs_concreteness': inputs_concreteness,
             'targets': torch.tensor(targets).long()}
    batch = default_none_dict(batch)
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
