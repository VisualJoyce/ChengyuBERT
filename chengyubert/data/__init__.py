import json
import os
import sys
from contextlib import contextmanager

import lmdb
import msgpack
import pandas as pda
from lz4.frame import compress, decompress
from more_itertools import chunked
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from chengyubert.data.loader import PrefetchLoader
from chengyubert.data.sampler import DistributedTokenBucketSampler, ContrastiveSampler, ContrastivePairSampler
from chengyubert.utils.const import BUCKET_SIZE
from chengyubert.utils.logger import LOGGER

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


def intermediate_dir(pretrained_model_name_or_path):
    model_name = pretrained_model_name_or_path
    if model_name.startswith('/pretrained'):
        # user defined pretrained model
        return model_name.replace('/pretrained', '')
    else:
        return model_name


class ChengyuLmdb(object):
    def __init__(self, split, max_txt_len, opts):
        self.split = split
        self.max_txt_len = max_txt_len
        self.config = opts

        # The txt db used for training should be generated by the its tokenizor accordingly, therefore, we insert
        # intermediate directories to indicate the pretrained model name or path
        self.db_dir = os.path.join('/txt',
                                   intermediate_dir(self.config.pretrained_model_name_or_path),
                                   getattr(opts, f'{split}_txt_db'))
        LOGGER.info(f"Loading {split} Dataset {self.db_dir}")

        self.id2len = json.load(open(f'{self.db_dir}/id2len.json'))
        self.db = TxtLmdb(self.db_dir, readonly=True)
        self.chengyu_vocab = chengyu_process(len_idiom_vocab=opts.len_idiom_vocab, annotation_dir='/annotations')

        self.id2idiom = {v: k for k, v in self.chengyu_vocab.items()}
        self.idiom_ids = list(range(opts.len_idiom_vocab))
        self.tokenizer = AutoTokenizer.from_pretrained(opts.pretrained_model_name_or_path)

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

calo_inverse_mapping = {
    'sentiment': {v: k for k, v in calo_mapping['sentiment'].items()},
    'coarse_emotion': {v['main_id']: v['main'] for k, v in calo_mapping['emotion'].items()},
    'fine_emotion': {v['id']: (v['main'], k, v['sub']) for k, v in calo_mapping['emotion'].items()}
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
        calo_vocab.setdefault(v, [])
        calo_vocab[v].append({
            'coarse_emotion': calo_mapping['emotion'][k]['main_id'],
            'fine_emotion': calo_mapping['emotion'][k]['id'],
            'sentiment': getattr(item, '极性'),
            'strength': getattr(item, '强度')
        })
    return calo_vocab


idioms_mapping = {
    'sentiment': {'中性': 0, '褒义': 1, '贬义': 2},
}
idioms_inverse_mapping = {
    'sentiment': {v: k for k, v in idioms_mapping['sentiment'].items()},
}


def get_form(idiom):
    tmp = idiom.lower()
    for a, b in [
        (" someone's ", ' ones '),
        (' someones ', ' ones '),
        (' your ', ' ones '),
        (' my ', ' ones '),
        (' his ', ' ones '),
        (' them ', ' someone '),
        (",", ' '),
        ("'", ' '),
        ('-', ' '),
        (' ', '')
    ]:
        tmp = tmp.replace(a, b)
    return tmp


def idioms_process(len_idiom_vocab=sys.maxsize, annotation_dir='/annotations'):
    """
    Load idioms to vocab with explanation
    :return:
    """
    sentiment_mapping = {
        'neutral': '中性',
        'positive': '褒义',
        'negative': '贬义',
        # '褒贬兼有': 3
    }
    df_sentiment = pda.read_csv(f"{annotation_dir}/slide/idiomLexicon.tsv", sep="\t")
    df_sentiment = df_sentiment[df_sentiment['Maj. Label'] != 'inappropriate']
    df_sentiment = df_sentiment.assign(label=df_sentiment['Maj. Label'].map(sentiment_mapping))

    df_idioms_580 = pda.read_csv(f'{annotation_dir}/slide/idioms_580.csv')
    idioment = df_idioms_580.idiom.tolist()

    idioms_extra = []
    for _, idiom, explanation in chunked(open(f'{annotation_dir}/slide/idioms_dataset_2432').read().split('\n'), 3):
        idioms_extra.append(idiom)

    idioms_vocab = {}
    idioms_forms = {}

    idioms_ids_range = {}
    for dataset, idioms in [('slide', df_sentiment.Idiom.tolist()),
                            ('idioment', idioment),
                            ('idioms2432', idioms_extra)]:
        start = len(idioms_vocab)
        for idiom in idioms:
            idx = idioms_vocab.get(idiom)
            if not idx:
                form = get_form(idiom)
                if form in idioms_forms and idiom not in idioms_forms[form]:
                    idioms_forms[form].append(idiom)
                else:
                    idioms_vocab[idiom] = len(idioms_vocab)
                    idioms_forms.setdefault(form, [])
                    idioms_forms[form].append(idiom)
        idioms_ids_range[dataset] = {
            'start': start,
            'end': len(idioms_vocab)
        }

    with open(f'{annotation_dir}/slide/idiom_span_mapping.json') as f:
        idiom_span_mapping = json.load(f)

    idioms_vocab = {k: v for k, v in idioms_vocab.items() if v < len_idiom_vocab}
    print(idioms_ids_range)
    print("Total idioms: {}".format(len(idioms_vocab)))

    sentiment_vocab = {}
    for item in df_sentiment.itertuples():
        sentiment = getattr(item, 'label')
        idiom = idiom_span_mapping[item.Idiom]
        sentiment_vocab[idioms_vocab[idiom]] = calo_mapping['sentiment'][sentiment]
    return idioms_vocab, sentiment_vocab, idiom_span_mapping


class IdiomsLmdb(object):
    def __init__(self, split, max_txt_len, opts):
        self.split = split
        self.max_txt_len = max_txt_len
        self.config = opts

        # The txt db used for training should be generated by the its tokenizor accordingly, therefore, we insert
        # intermediate directories to indicate the pretrained model name or path
        self.db_dir = os.path.join('/txt',
                                   intermediate_dir(self.config.pretrained_model_name_or_path),
                                   getattr(opts, f'{split}_txt_db'))
        LOGGER.info(f"Loading {split} Dataset {self.db_dir}")

        self.id2len = json.load(open(f'{self.db_dir}/id2len.json'))
        self.db = TxtLmdb(self.db_dir, readonly=True)
        self.vocab, self.sentiments, self.mapping = idioms_process(len_idiom_vocab=opts.len_idiom_vocab,
                                                                   annotation_dir='/annotations')

        self.id2idiom = {v: k for k, v in self.vocab.items()}
        self.idiom_ids = list(range(opts.len_idiom_vocab))
        self.tokenizer = AutoTokenizer.from_pretrained(opts.pretrained_model_name_or_path)

    def __getitem__(self, id_):
        txt_dump = self.db[id_]
        return txt_dump


def create_dataloader(batch_size, split, dset_cls, opts):
    dset = dset_cls(split, opts.max_txt_len, opts)
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


def create_dataloaders(dataset_cls, eval_dataset_cls, opts, splits=None):
    if splits is None:
        splits = []
        for k in dir(opts):
            if k.endswith('_txt_db'):
                splits.append(k.replace('_txt_db', ''))

    print(splits)

    dataloaders = {}
    for split in ['train'] + [s for s in splits if s != 'train']:
        # txt_db = getattr(opts, f'{split}_txt_db')
        batch_size = getattr(opts, f'{split}_batch_size') if split == 'train' else opts.val_batch_size
        DatasetCls = dataset_cls if split == 'train' else eval_dataset_cls
        # create_dataloader_fn = create_dataloader_fn_dict.get(split, create_dataloader)
        LOGGER.info(f"Loading {split} Dataset using {DatasetCls}")
        dataloaders[split] = create_dataloader(batch_size, split, DatasetCls, opts)
    return splits, dataloaders
