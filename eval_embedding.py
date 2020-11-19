import bz2
import json
import os
import sys
from itertools import chain

import numpy as np
import pandas as pda
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


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


def read_vectors(path, topn=0):  # read top n word vectors, i.e. top is 10000
    lines_num, dim = 0, 0
    vectors = {}
    iw = []
    wi = {}

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

                if topn != 0 and i > topn:
                    break
                pbar.update(1)

    wi = {w: i for i, w in enumerate(iw)}
    return vectors, iw, wi, dim


def construct_synonyms(annotations_dir):
    with open(f'{annotations_dir}/synonyms/chengyu_synonyms_recall.json') as f:
        crawl_synonyms_dict = json.load(f)

    with open(f'{annotations_dir}/synonyms/chengyu_synonyms_recall_baike.json') as f:
        baike_synonyms_dict = json.load(f)

    with open(f'{annotations_dir}/synonyms/chengyu_synonyms_recall_baike_extra.json') as f:
        baike_synonyms_extra = json.load(f)

    chengyu_synonyms = {
        0: {},
        1: {},
        2: {},
        3: {},
        4: {}
    }

    chengyu_filtered = {
        0: {},
        1: {},
        2: {},
        3: {},
        4: {}
    }

    for item, v in chain(crawl_synonyms_dict.items(), baike_synonyms_dict.items(), baike_synonyms_extra.items()):
        for vv in v:
            overlapping = len(set(item).intersection(set(vv)))
            for k in chengyu_synonyms:
                if overlapping <= k:
                    chengyu_synonyms[k].setdefault(item, [])
                    if vv not in chengyu_synonyms[k][item] and vv:
                        if vv != item:
                            chengyu_synonyms[k][item].append(vv)
                else:
                    chengyu_filtered[k].setdefault(item, [])
                    if vv not in chengyu_filtered[k][item] and vv:
                        if vv != item:
                            chengyu_filtered[k][item].append(vv)

    to_del = []
    for k in chengyu_synonyms:
        for v in chengyu_synonyms[k]:
            if not chengyu_synonyms[k][v]:
                to_del.append([k, v])
    for k, v in to_del:
        del chengyu_synonyms[k][v]

    with open(f'{annotations_dir}/synonyms/chengyu_antonyms.json') as f:
        chengyu_antonyms_dict = json.load(f)

    chengyu_antonyms = {}
    for k, v in chengyu_antonyms_dict.items():
        for vv in v:
            if vv.strip():
                chengyu_antonyms.setdefault(k, [])
                chengyu_antonyms[k].append(vv)
    return chengyu_synonyms, chengyu_filtered, chengyu_antonyms


def evaluate_embeddings_recall(iw, wi, vectors_np, chengyu_synonyms_dict, chengyu_antonyms_dict, cached):
    cnt = 0
    recall_at_k_cosine = {}
    antonyms_at_k_cosine = {}
    recall_at_k_norm = {}
    antonyms_at_k_norm = {}
    set_recall_at_k_cosine = {}
    set_recall_at_k_norm = {}

    mrr_cosine = 0
    mrr_norm = 0

    k_list = [1, 3, 5, 10, 50, 100]
    k_max = max(k_list)

    total = len(chengyu_synonyms_dict)
    top_k = {}
    for w, wl in tqdm(chengyu_synonyms_dict.items()):
        wl_len = len([x for x in wl if x in wi])
        if w in wi and wl_len > 0:
            cnt += 1

            neighbourhood_cosine = []
            neighbourhood_norm = []

            for x in [w] + wl:
                if x in embeddings:
                    if x not in cached:
                        cosine_distances = (1 - cosine_similarity(embeddings[x].reshape(1, -1), vectors)[0]).argsort()
                        norm_distances = np.linalg.norm(vectors - embeddings[x], axis=1).argsort()
                        cached[x] = (cosine_distances, norm_distances)
                    else:
                        cosine_distances, norm_distances = cached[x]

                    neighbourhood_cosine.append([idx for idx in cosine_distances if iw[idx] in chengyu_vocab][:k_max])
                    neighbourhood_norm.append([idx for idx in norm_distances if iw[idx] in chengyu_vocab][:k_max])

            for k in k_list:
                set_recall_at_k_cosine.setdefault(k, 0)
                set_recall_at_k_cosine[k] += len(set.intersection(*[set(ns[:k]) for ns in neighbourhood_cosine])) / len(
                    set.union(*[set(ns[:k]) for ns in neighbourhood_cosine]))

                set_recall_at_k_norm.setdefault(k, 0)
                set_recall_at_k_norm[k] += len(set.intersection(*[set(ns[:k]) for ns in neighbourhood_norm])) / len(
                    set.union(*[set(ns[:k]) for ns in neighbourhood_norm]))
            if w not in cached:
                w_emb = vectors_np[wi[w]].reshape(1, -1)
                cosine_distances = (1 - cosine_similarity(w_emb, vectors_np)[0]).argsort().tolist()
                norm_distances = np.linalg.norm(vectors_np - w_emb, axis=1).argsort().tolist()
                cached[w] = (cosine_distances, norm_distances)
            else:
                cosine_distances, norm_distances = cached[w]

            cids = [idx for idx in cosine_distances if iw[idx] in chengyu_vocab]
            nids = [idx for idx in norm_distances if iw[idx] in chengyu_vocab]
            for k in k_list:
                top_ids = cids[1:k + 1]
                recall_at_k_cosine.setdefault(k, 0)
                recall_at_k_cosine[k] += sum([1 for idx in top_ids if iw[idx] in wl]) / wl_len

                antonyms_at_k_cosine.setdefault(k, 0)
                antonyms_at_k_cosine[k] += sum([1 for idx in top_ids if iw[idx] in chengyu_antonyms_dict.get(w, [])])

                top_ids = nids[1:k + 1]
                recall_at_k_norm.setdefault(k, 0)
                recall_at_k_norm[k] += sum([1 for idx in top_ids if iw[idx] in wl]) / wl_len

                #                 for idx in top_ids:
                #                     if iw[idx] in chengyu_antonyms.get(w, []):
                #                         print(w, iw[idx])
                antonyms_at_k_norm.setdefault(k, 0)
                antonyms_at_k_norm[k] += sum([1 for idx in top_ids if iw[idx] in chengyu_antonyms_dict.get(w, [])])

            mrr_cosine += sum([1 / (cids.index(wi[x])) for x in wl if x in wi]) / wl_len
            mrr_norm += sum([1 / (nids.index(wi[x])) for x in wl if x in wi]) / wl_len
            top_k[w] = ([x for x in wl if x in wi],
                        [iw[idx] for idx in top_ids if iw[idx] in wl],
                        [iw[idx] for idx in top_ids])
    print(cnt, ' word pairs appered in the training dictionary , total word pairs ', total)
    print(recall_at_k_cosine)
    print(recall_at_k_norm)
    print(antonyms_at_k_cosine)
    for k in recall_at_k_cosine:
        recall_at_k_cosine[k] /= cnt
        recall_at_k_norm[k] /= cnt
        set_recall_at_k_cosine[k] /= cnt
        set_recall_at_k_norm[k] /= cnt

    print('\t'.join(chain([format(recall_at_k_cosine[k], "0.6f") for k in k_list],
                          [format(antonyms_at_k_cosine[k], "d") for k in k_list],
                          [format(recall_at_k_norm[k], "0.6f") for k in k_list],
                          [format(antonyms_at_k_norm[k], "d") for k in k_list],
                          [format(set_recall_at_k_cosine[k], "0.6f") for k in k_list],
                          [format(set_recall_at_k_norm[k], "0.6f") for k in k_list],
                          [format(mrr, "0.6f") for mrr in [mrr_cosine / cnt, mrr_norm / cnt]]
                          )))
    print(recall_at_k_cosine)
    print(recall_at_k_norm)
    print(set_recall_at_k_cosine)
    print(set_recall_at_k_norm)
    return cnt, total, recall_at_k_cosine, recall_at_k_norm, top_k


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--corpus",
                        default='Literature', type=str,
                        help="json file for model architecture")
    parser.add_argument("--subtype",
                        default='literature', type=str,
                        help="json file for model architecture")
    parser.add_argument("--model_path",
                        default=None, type=str,
                        help="json file for model architecture")

    args = parser.parse_args()

    chengyu_vocab = chengyu_process(annotation_dir='./data/annotations')
    synonyms, filtered, antonyms = construct_synonyms(annotations_dir='./data/annotations')

    emb_path = "./data/pretrained/Chinese-Word-Vectors"

    if args.model_path is None:
        for t in ['word', 'char', 'bigram', 'bigram-char']:
            print(f"^^^^^^^^^^^^^^^^^^^^^^^^{t}^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            embeddings, _, _, dim = read_vectors(f'{emb_path}/embeddings/{args.corpus}/sgns.{args.subtype}.{t}.bz2')

            i2w = [k for k in embeddings if k in chengyu_vocab]
            w2i = {w: i for i, w in enumerate(i2w)}
            vectors = np.array([embeddings[i2w[i]] for i in range(len(i2w))])

            cached = {}
            for k in synonyms:
                print(f'==========================={t}-{k}===============================')
                evaluate_embeddings_recall(i2w, w2i, vectors, synonyms[k], antonyms, cached)
    else:
        _, iw, wi, dim = read_vectors(f'{emb_path}/embeddings/{args.corpus}/sgns.{args.subtype}.word.bz2')
        model = torch.load(args.model_path)
        embeddings_np = model['idiom_embedding.weight'].cpu().numpy()
        embeddings = {k: embeddings_np[v] for k, v in chengyu_vocab.items() if k in wi}

        i2w = [k for k in embeddings if k in chengyu_vocab]
        w2i = {w: i for i, w in enumerate(i2w)}
        vectors = np.array([embeddings[i2w[i]] for i in range(len(i2w))])

        cached = {}
        for k in synonyms:
            print(f'==========================={args.model_path}-{k}===============================')
            evaluate_embeddings_recall(i2w, w2i, vectors, synonyms[k], antonyms, cached)
