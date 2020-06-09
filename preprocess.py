import argparse
import json
import math
import os
import random
import re

import pandas as pda
from cytoolz import curry
from tqdm import tqdm
from transformers import BertTokenizer

from chengyubert.data import open_lmdb


class Example(object):
    def __init__(self,
                 idx,
                 tag,
                 context,
                 options,
                 label=None):
        self.idx = idx
        self.tag = tag
        self.context = context
        self.options = options
        self.label = label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "idx: %s" % (self.idx)
        s += "tag: %s" % (self.tag)
        s += ", context: %s" % (self.context)
        s += ", options: [%s]" % (", ".join(self.options))
        if self.label is not None:
            s += ", answer: %s" % self.options[self.label]
        return s


def chengyu_process():
    """
    Load Chengyu to vocab with explanation
    :return:
    """
    # load ChID idioms to first 3848
    chengyu_vocab = {each: i for i, each in enumerate(eval(open('/txt/idiomList.txt').readline()))}

    # load Xinhua idioms
    chengyu_pretrain = pda.read_csv("/txt/idioms_pretrain.json", sep='\t')
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

    print("Total idioms: {}".format(len(chengyu_vocab)))

    return chengyu_vocab


def create_cct_dataset(candidates_num):
    """
    Create a random Chengyu Cloze Test with a specific candidates number
    :param candidates_num: Candidate set size
    :return:
    """
    df = pda.read_csv('/txt/chengyu/chengyu_sentence.txt', header=None, names=['ground_truth', 'context'])
    with open('/txt/chengyu/chengyu_sentence.txt') as g:
        with open(self.data_file, 'w') as f:
            for l in g:
                item = l.strip().split(',')
                tmp_context = ','.join(item[1:])
                label = item[0]
                options = df.ground_truth.sample(candidates_num - 1).to_list() + [label]
                random.shuffle(options)
                if "～" in tmp_context:
                    tmp_context = tmp_context.replace("～", "#idiom#")
                else:
                    tmp_context = tmp_context.replace("#{}#".format(label), "#idiom#")

                f.write(json.dumps({
                    "groundTruth": [label],
                    "candidates": [options],
                    "content": tmp_context,
                    "realCount": 1
                }, ensure_ascii=False))
                f.write('\n')


def tokenize(tokenizer, example):
    tag = example.tag
    parts = re.split(tag, example.context)
    assert len(parts) == 2
    before_part = tokenizer.tokenize(parts[0]) if len(parts[0]) > 0 else []
    after_part = tokenizer.tokenize(parts[1]) if len(parts[1]) > 0 else []

    tokens = before_part + [tokenizer.mask_token] + after_part
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    position = input_ids.index(tokenizer.mask_token_id)
    return input_ids, position


class ChidOfficialParser(object):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    splits = ['pretrain', 'train', 'dev', 'test', 'ran', 'sim', 'out', 'chengyu', 'chengyu4']

    # splits = ['chengyu', 'chengyu4']

    def __init__(self, split, vocab):
        self.split = split
        self.vocab = vocab
        self.data_dir = '/txt/official'

    @property
    def answer_file(self):
        return os.path.join(self.data_dir, '{}_answer.csv'.format(self.split))

    @property
    def data_file(self):
        if self.split in ['pretrain', 'train', 'dev', 'test']:
            return os.path.join(self.data_dir, '{}_data.txt'.format(self.split))
        elif self.split == 'out':
            return os.path.join(self.data_dir, 'test_out_data.txt')
        elif self.split == 'sim':
            return os.path.join(self.data_dir, 'test_data_sim.txt')
        elif self.split == 'ran':
            return os.path.join(self.data_dir, 'test_data_ord.txt')
        elif self.split in ['chengyu', 'chengyu4']:
            return os.path.join(self.data_dir, 'chengyu.txt')

    def read_examples(self):
        idioms = list(self.vocab.keys())

        with tqdm(total=os.path.getsize(self.data_file), desc=self.split,
                  bar_format="{desc}: {percentage:.3f}%|{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            with open(self.data_file, mode='rb') as f:
                for idx, data_str in enumerate(f):
                    pbar.update(len(data_str))
                    data = eval(data_str.decode('utf8'))
                    context = data['content']
                    if 'candidates' not in data:
                        k_i = data['groundTruth'][0]
                        if k_i not in idioms and not self.config.use_xinhua:
                            continue

                        candidates = random.sample(idioms, 6) + [k_i]
                        random.shuffle(candidates)
                        data['candidates'] = [candidates]

                    for i, (tag, label, options) in enumerate(
                            zip(re.finditer("#idiom#", context), data['groundTruth'], data['candidates'])):
                        tmp_context = context
                        ind = options.index(label)
                        new_tag = idx * 20 + i
                        tag_str = "#idiom%06d#" % new_tag

                        tmp_context = "".join((tmp_context[:tag.start(0)], tag_str, tmp_context[tag.end(0):]))
                        tmp_context = tmp_context.replace("#idiom#", "[UNK]")

                        yield Example(
                            idx=new_tag,
                            tag=tag_str,
                            context=tmp_context,
                            options=options,
                            label=ind
                        )


class ChidCompetitionDataset(object):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    splits = ['train', 'dev', 'test', 'out']

    def __init__(self, split, tokenizer, config):
        self.split = split
        self.data_dir = '/txt/competition'

    @property
    def answer_file(self):
        return os.path.join(self.data_dir, '{}_answer.csv'.format(self.split))

    @property
    def data_file(self):
        return os.path.join(self.data_dir, '{}.txt'.format(self.split))

    def read_examples(self):
        ans_dict = {}
        with open(self.answer_file, 'r') as f:
            for ll in f:
                k, v = ll.strip().split(',')
                ans_dict[k] = int(v)

        with tqdm(total=os.path.getsize(self.data_file), desc=self.split,
                  bar_format="{desc}: {percentage:.3f}%|{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            with open(self.data_file, mode='rb') as f:
                for idx, data_str in enumerate(f):
                    pbar.update(len(data_str))
                    data = eval(data_str.decode('utf8'))
                    options = data['candidates']
                    for context in data['content']:
                        tags = re.findall("#idiom\d+#", context)
                        for tag in tags:
                            tmp_context = context
                            for other_tag in tags:
                                if other_tag != tag:
                                    tmp_context = tmp_context.replace(other_tag, "[UNK]")
                            yield Example(
                                idx=idx,
                                tag=tag,
                                context=tmp_context,
                                options=options,
                                label=ans_dict[tag]
                            )


def process_chid(opts, db, tokenizer):
    source, split = opts.annotation.split('_')

    vocab = chengyu_process()

    if source == 'official':
        assert split in ['pretrain', 'train', 'dev', 'test', 'ran', 'sim', 'out', 'chengyu', 'chengyu4']
        parser = ChidOfficialParser(split, vocab)
    else:
        assert split in ['train', 'dev', 'test', 'out']
        parser = ChidCompetitionDataset(split, vocab)

    def parse_example(example):
        input_ids, position = tokenize(tokenizer, example)

        id_ = example.tag
        id2len[id_] = len(input_ids)
        db[id_] = {
            'input_ids': input_ids,
            'position': position,
            'target': example.label,
            'options': example.options
        }

    id2len = {}
    ans_dict = {}
    for ex in parser.read_examples():
        parse_example(ex)
        ans_dict[ex.tag] = ex.label

    assert len(id2len) == len(ans_dict)

    with open(parser.answer_file, 'w') as f:
        for k, v in ans_dict.items():
            f.write('{},{}\n'.format(k, v))

    return id2len


def main(opts):
    print(opts)
    # train_db_dir = os.path.join(os.path.dirname(opts.output), f'{source}_{split}.db')
    # meta = vars(opts)
    # meta['tokenizer'] = opts.toker
    tokenizer = BertTokenizer(vocab_file='/pretrain/wwm_ext/vocab.txt')

    open_db = curry(open_lmdb, '/output', readonly=False)
    with open_db() as db:
        id2lens = process_chid(opts, db, tokenizer)

    with open(f'{opts.output}/id2len.json', 'w') as f:
        json.dump(id2lens, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', required=True,
                        help='annotation JSON')
    args = parser.parse_args()
    main(args)
