import bz2
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

"""
This file deals with reading pretrained word vectors from online resources and extracting representations from BERT.
"""


def read_vectors_from_bz2(path):
    with bz2.BZ2File(path) as f:
        for line_count, line in enumerate(f):
            try:
                yield line_count, line.decode().rstrip()
            except:
                print(line)


def read_vectors_from_txt(path):
    with tqdm(total=os.path.getsize(path),
              bar_format="{desc}: {percentage:.3f}%|{bar}| {n:.2f}/{total_fmt} [{elapsed}<{remaining}]") as pbar:
        with open(path, 'rb') as f:
            for line_count, line in enumerate(f):
                pbar.update(len(line))
                yield line_count, line.decode().rstrip()


def read_vectors(path, top_k=0):  # read top n word vectors, i.e. top is 10000
    lines_num, dim = 0, 0
    vectors = {}
    iw = []

    if path.endswith('.bz2'):
        itr = read_vectors_from_bz2(path)
    else:
        itr = read_vectors_from_txt(path)

    with tqdm(total=100) as pbar:
        for i, line in itr:
            if i == 0:
                lines_num, dim = list(map(int, line.rstrip().split()))
                pbar.reset(total=lines_num)
            else:
                tokens = line.rstrip().split()
                if len(tokens) == dim + 1:
                    word = tokens[0]
                    if word not in vectors:
                        vectors[word] = np.asarray([float(x) for x in tokens[1:]])
                        iw.append(word)
                elif len(tokens) > dim + 1:
                    word = ' '.join(tokens[:len(tokens) - dim])
                    if word not in vectors:
                        vectors[word] = np.asarray([float(x) for x in tokens[len(tokens) - dim:]])
                        iw.append(word)
                    print(word)
                else:
                    print(f"Skip a line of all spaces! {tokens[:2]}")

                if top_k != 0 and i > top_k:
                    break
                pbar.update(1)

    wi = {w: i for i, w in enumerate(iw)}
    return vectors, iw, wi, dim


def get_bert_embedding(model, tokenizer, text):
    input_dict = tokenizer(text, return_tensors="pt")
    input_ids = input_dict['input_ids']
    token_type_ids = input_dict['token_type_ids']
    attention_mask = input_dict['attention_mask']

    with torch.no_grad():
        encoded_layers = model(
            input_ids.cuda(),
            attention_mask=attention_mask.cuda(),
            token_type_ids=token_type_ids.cuda()
        )
        #         return encoded_layers[0][:, 0].squeeze().cpu().numpy()
        return encoded_layers[0][:, 1:-1].mean(1).squeeze().cpu().numpy()


def load_embeddings(chengyu_vocab, emb_path, embedding_type, wi):
    if embedding_type in ['word', 'char', 'bigram', 'bigram-char']:
        embeddings, _, _, dim = read_vectors(f'{emb_path}/Literature/sgns.literature.{embedding_type}.bz2')
    else:
        if embedding_type == 'ERNIE':
            tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0", use_fast=True)
            model = AutoModel.from_pretrained("nghuyong/ernie-1.0").cuda()
            embeddings = {k: get_bert_embedding(model, tokenizer, k) for k, v in chengyu_vocab.items() if k in wi}
        elif embedding_type == 'BERT':
            tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext', use_fast=True)
            model = AutoModel.from_pretrained('hfl/chinese-bert-wwm-ext').cuda()
            embeddings = {k: get_bert_embedding(model, tokenizer, k) for k, v in chengyu_vocab.items() if k in wi}
        else:
            embeddings = None
            assert ValueError("Unsupported embedding type!")
    return embeddings
