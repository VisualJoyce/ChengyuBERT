import json
import os
import sys
from contextlib import contextmanager

import lmdb
import msgpack
import pandas as pda
from lz4.frame import compress, decompress
from torch.utils.data import DataLoader

from chengyubert.data.loader import PrefetchLoader
from chengyubert.data.sampler import DistributedTokenBucketSampler, ContrastiveSampler, ContrastivePairSampler
from chengyubert.utils.const import BUCKET_SIZE

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


calo_mapping = {
    'sentiment': {'中性': 0, '褒义': 1, '贬义': 2, '褒贬兼有': 3},
    'emotion': {
        'PA': {'main': '乐', 'sub': '快乐', 'id': 0, 'main_id': 0, 'binary': 1},
        'PE': {'main': '乐', 'sub': '安心', 'id': 1, 'main_id': 0, 'binary': 1},
        'PD': {'main': '好', 'sub': '尊敬', 'id': 2, 'main_id': 1, 'binary': 1},
        'PH': {'main': '好', 'sub': '赞扬', 'id': 3, 'main_id': 1, 'binary': 1},
        'PG': {'main': '好', 'sub': '相信', 'id': 4, 'main_id': 1, 'binary': 1},
        'PB': {'main': '好', 'sub': '喜爱', 'id': 5, 'main_id': 1, 'binary': 1},
        'PK': {'main': '好', 'sub': '祝愿', 'id': 6, 'main_id': 1, 'binary': 1},
        'NA': {'main': '怒', 'sub': '愤怒', 'id': 7, 'main_id': 2, 'binary': 2},
        'NB': {'main': '哀', 'sub': '悲伤', 'id': 8, 'main_id': 3, 'binary': 2},
        'NJ': {'main': '哀', 'sub': '失望', 'id': 9, 'main_id': 3, 'binary': 2},
        'NH': {'main': '哀', 'sub': '疚', 'id': 10, 'main_id': 3, 'binary': 2},
        'PF': {'main': '哀', 'sub': '思', 'id': 11, 'main_id': 3, 'binary': 2},
        'NI': {'main': '惧', 'sub': '慌', 'id': 12, 'main_id': 4, 'binary': 2},
        'NC': {'main': '惧', 'sub': '恐惧', 'id': 13, 'main_id': 4, 'binary': 2},
        'NG': {'main': '惧', 'sub': '羞', 'id': 14, 'main_id': 4, 'binary': 2},
        'NE': {'main': '恶', 'sub': '烦闷', 'id': 15, 'main_id': 5, 'binary': 2},
        'ND': {'main': '恶', 'sub': '憎恶', 'id': 16, 'main_id': 5, 'binary': 2},
        'NN': {'main': '恶', 'sub': '贬责', 'id': 17, 'main_id': 5, 'binary': 2},
        'NK': {'main': '恶', 'sub': '妒忌', 'id': 18, 'main_id': 5, 'binary': 2},
        'NL': {'main': '恶', 'sub': '怀疑', 'id': 19, 'main_id': 5, 'binary': 2},
        'PC': {'main': '惊', 'sub': '惊奇', 'id': 20, 'main_id': 6, 'binary': 1}
    }
}


def calo_process(chengyu_vocab, calo_file):
    # calo_file = '情感词汇本体.xlsx'
    df_sentiment = pda.read_excel(calo_file, keep_default_na=False)
    df_sentiment['情感分类'] = df_sentiment['情感分类'].str.strip()
    df_sentiment = df_sentiment[df_sentiment['词语'].isin(chengyu_vocab.keys())]

    calo_vocab = {}
    for item in df_sentiment.itertuples():
        w = getattr(item, '词语')
        k = getattr(item, '情感分类')

        v = chengyu_vocab[w]
        calo_vocab[v] = {
            'coarse_emotion': calo_mapping['emotion'][k]['main_id'],
            'fine_emotion': calo_mapping['emotion'][k]['id'],
            'sentiment': calo_mapping['sentiment'][getattr(item, '极性')],
            'strength': getattr(item, '强度')
        }
    return calo_vocab


def create_dataloader(txt_path, batch_size, split, dset_cls, opts):
    dset = dset_cls(txt_path, opts.max_txt_len, opts, split)
    sampler = DistributedTokenBucketSampler(
        opts.size, opts.rank, dset.lens,
        bucket_size=BUCKET_SIZE, batch_size=batch_size, droplast='train' == split)
    loader = DataLoader(dset, batch_sampler=sampler,
                        num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                        collate_fn=dset_cls.collate_fn)
    return PrefetchLoader(loader)


def create_contrastive_dataloader(txt_path, batch_size, is_train, dset_cls, opts):
    dset = dset_cls(txt_path, opts.max_txt_len, opts)
    sampler = ContrastiveSampler(opts.size, opts.rank, dset.lens, dset.ids, batch_size, dset.reverse_index,
                                 droplast=is_train)
    loader = DataLoader(dset, batch_sampler=sampler,
                        num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                        collate_fn=dset_cls.collate_fn)
    return PrefetchLoader(loader)


def create_contrastive_pair_dataloader(txt_path, batch_size, is_train,
                                       dset_cls, collate_fn, opts):
    dset = dset_cls(txt_path, opts.max_txt_len, opts)
    sampler = ContrastivePairSampler(opts.size, opts.rank, dset.lens, dset.ids, batch_size, dset.reverse_index,
                                     droplast=is_train)
    loader = DataLoader(dset, batch_sampler=sampler,
                        num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                        collate_fn=collate_fn)
    return PrefetchLoader(loader)


def create_dataloaders(LOGGER, DatasetCls, EvalDatasetCls, opts, splits=None):
    if splits is None:
        splits = []
        for k in dir(opts):
            if k.endswith('_txt_db'):
                splits.append(k.replace('_txt_db', ''))

    print(splits)

    dataloaders = {}
    for split in splits:
        txt_db = getattr(opts, f'{split}_txt_db')
        LOGGER.info(f"Loading {split} Dataset {txt_db}")
        batch_size = getattr(opts, f'{split}_batch_size') if split in ['pretrain', 'train'] else opts.val_batch_size
        dataset_cls = DatasetCls if split in ['pretrain', 'train'] else EvalDatasetCls
        # create_dataloader_fn = create_dataloader_fn_dict.get(split, create_dataloader)
        dataloaders[split] = create_dataloader(txt_db, batch_size, split, dataset_cls, opts)
    return splits, dataloaders
