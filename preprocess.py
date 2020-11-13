import argparse
import json
import os
import random
import re
from abc import abstractmethod

import pandas as pda
from cytoolz import curry
from tqdm import tqdm
from transformers import AutoTokenizer

from chengyubert.data import open_lmdb
from chengyubert.data.data import chengyu_process
from chengyubert.utils.misc import parse_with_config


class Example(object):
    def __init__(self,
                 idx,
                 tag,
                 context,
                 idiom,
                 options,
                 label=None):
        self.idx = idx
        self.tag = tag
        self.context = context
        self.idiom = idiom
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
        if self.label is not None and self.options:
            s += ", answer: %s" % self.options[self.label]
        return s


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


class ChidParser(object):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, split, vocab, annotation_dir='/annotation'):
        self.split = split
        self.vocab = vocab
        self.annotation_dir = annotation_dir

    @property
    @abstractmethod
    def data_dir(self):
        pass

    @property
    @abstractmethod
    def data_file(self):
        pass

    @property
    def answer_file(self):
        return os.path.join(self.data_dir, '{}_answer.csv'.format(self.split))

    def read_examples(self):
        with tqdm(total=os.path.getsize(self.data_file), desc=self.split,
                  bar_format="{desc}: {percentage:.3f}%|{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
            with open(self.data_file, mode='rb') as f:
                for idx, data_str in enumerate(f):
                    pbar.update(len(data_str))
                    data = eval(data_str.decode('utf8'))
                    context = data['content']
                    for i, (tag, idiom) in enumerate(zip(re.finditer("#idiom#", context), data['groundTruth'])):
                        new_tag = idx * 20 + i
                        tag_str = "#idiom%06d#" % new_tag

                        tmp_context = context
                        tmp_context = "".join((tmp_context[:tag.start(0)], tag_str, tmp_context[tag.end(0):]))
                        tmp_context = tmp_context.replace("#idiom#", "[UNK]")

                        if 'candidates' not in data:
                            options, label = [], -1
                        else:
                            options = data['candidates'][i]
                            if len(options) != 7:
                                print(data)
                                assert len(options) == 7
                            label = options.index(idiom)

                        yield Example(
                            idx=new_tag,
                            tag=tag_str,
                            context=tmp_context,
                            idiom=idiom,
                            options=options,
                            label=label
                        )


class ChidBalancedParser(ChidParser):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    splits = ['train', 'test']

    def __init__(self, split, vocab, annotation_dir='/annotations'):
        self.split = split
        self.vocab = vocab
        self.annotation_dir = annotation_dir

    @property
    def data_dir(self):
        return f'{self.annotation_dir}/balanced'

    @property
    def data_file(self):
        return os.path.join(self.data_dir, 'balanced{}_{}.txt'.format(len(self.vocab), self.split))


class ChidBalancedFixParser(ChidParser):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    splits = ['train', 'test']
    fix_dict = {'一笑百媚': '一笑百媚生',
                '不可终日': '惶惶不可终日',
                '不识泰山': '有眼不识泰山',
                '吹灰之力': '不费吹灰之力',
                '来者居上': '后来者居上',
                '死而后已': '鞠躬尽瘁，死而后已',
                '语不惊人': '语不惊人死不休',
                '雨后春笋': '如雨后春笋般'}

    def __init__(self, split, vocab, annotation_dir='/annotations'):
        self.split = split
        self.vocab = {}
        for k, v in vocab.items():
            k = self.fix_dict.get(k, k)
            self.vocab[k] = v

        self.annotation_dir = annotation_dir

    @property
    def data_dir(self):
        return f'{self.annotation_dir}/balanced'

    @property
    def data_file(self):
        return os.path.join(self.data_dir, 'balanced{}fix_{}.txt'.format(len(self.vocab), self.split))


class ChidExternalParser(ChidParser):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    splits = ['pretrain', 'cct7', 'cct4']

    def __init__(self, split, vocab, annotation_dir='/annotations'):
        self.split = split
        self.vocab = vocab
        self.annotation_dir = annotation_dir

    @property
    def data_dir(self):
        return f'{self.annotation_dir}/external'

    @property
    def data_file(self):
        if self.split in ['pretrain']:
            return os.path.join(self.data_dir, '{}_data.txt'.format(self.split))
        elif self.split in ['cct7', 'cct4']:
            return os.path.join(self.data_dir, 'chengyu.txt')


class ChidOfficialParser(ChidParser):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    splits = ['train', 'dev', 'test', 'ran', 'sim', 'out']

    def __init__(self, split, vocab, annotation_dir='/annotations'):
        self.split = split
        self.vocab = vocab
        self.annotation_dir = annotation_dir

    @property
    def data_dir(self):
        return f'/{self.annotation_dir}/official'

    @property
    def data_file(self):
        if self.split in ['train', 'dev', 'test']:
            return os.path.join(self.data_dir, '{}_data.txt'.format(self.split))
        elif self.split == 'out':
            return os.path.join(self.data_dir, 'test_out_data.txt')
        elif self.split == 'sim':
            return os.path.join(self.data_dir, 'test_data_sim.txt')
        elif self.split == 'ran':
            return os.path.join(self.data_dir, 'test_data_ord.txt')


class ChidCompetitionDataset(object):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    splits = ['train', 'dev', 'test', 'out']

    def __init__(self, split, vocab, annotation_dir='/annotations'):
        self.split = split
        self.vocab = vocab
        self.annotation_dir = annotation_dir
        self.data_dir = f'{self.annotation_dir}/competition'

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
                                idiom=options[ans_dict[tag]],
                                options=options,
                                label=ans_dict[tag]
                            )


def process_chid(opts, db, tokenizer):
    source, split = opts.annotation.split('_')

    vocab = chengyu_process(len_idiom_vocab=opts.len_idiom_vocab, annotation_dir='/annotations')

    if source == 'official':
        assert split in ['train', 'dev', 'test', 'ran', 'sim', 'out']
        parser = ChidOfficialParser(split, vocab)
    elif source == 'external':
        assert split in ['pretrain', 'cct7', 'cct4']
        parser = ChidExternalParser(split, vocab)
    elif source == 'balanced':
        assert split in ['train', 'val']
        parser = ChidBalancedParser(split, vocab)
    elif source == 'balancedfix':
        assert split in ['train', 'val']
        parser = ChidBalancedFixParser(split, vocab)
    else:
        assert split in ['train', 'dev', 'test', 'out']
        parser = ChidCompetitionDataset(split, vocab)

    def parse_example(example):
        input_ids, position = tokenize(tokenizer, example)
        return {
            'input_ids': input_ids,
            'position': position,
            'idiom': parser.vocab[example.idiom],
            'target': example.label,
            'options': [parser.vocab[o] for o in example.options]
        }

    id2len = {}
    ans_dict = {}
    id2eid = {}
    reverse_index = {}
    for ex in parser.read_examples():
        exa = parse_example(ex)

        db[ex.tag] = exa
        id2len[ex.tag] = len(exa['input_ids'])
        ans_dict[ex.tag] = ex.label
        id2eid[ex.tag] = ex.idx

        idiom_id = parser.vocab[ex.idiom]
        reverse_index.setdefault(idiom_id, [])
        reverse_index[idiom_id].append(ex.tag)

    assert len(id2len) == len(ans_dict)

    with open(f'{opts.output}/answer.csv', 'w') as f:
        for k, v in ans_dict.items():
            f.write('{},{}\n'.format(k, v))

    with open(f'{opts.output}/id2eid.json', 'w') as f:
        json.dump(id2eid, f)

    with open(f'{opts.output}/reverse_index.json', 'w') as f:
        json.dump(reverse_index, f)

    return id2len


def main(opts):
    print(opts)
    dataset, split = opts.annotation.split('_')
    if split == 'dev':
        txt_db = 'val_txt_db'
    elif split == 'pretrain':
        txt_db = 'train_txt_db'
    else:
        txt_db = f'{split}_txt_db'
    opts.output = getattr(opts, txt_db)
    # train_db_dir = os.path.join(os.path.dirname(opts.output), f'{source}_{split}.db')
    # meta = vars(opts)
    # meta['tokenizer'] = opts.toker
    tokenizer = AutoTokenizer.from_pretrained(opts.pretrained_model_name_or_path, use_fast=True)

    open_db = curry(open_lmdb, opts.output, readonly=False)
    with open_db() as db:
        id2lens = process_chid(opts, db, tokenizer)

    with open(f'{opts.output}/id2len.json', 'w') as f:
        json.dump(id2lens, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', required=True,
                        help='annotation JSON')
    parser.add_argument('--config', help='JSON config files')
    args = parse_with_config(parser)
    main(args)
