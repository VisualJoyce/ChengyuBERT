import numpy as np
import pandas as pda
import torch

from chengyubert.data import chengyu_process
from chengyubert.data.embeddings import load_embeddings, read_vectors
from chengyubert.data.evaluation import evaluate_embeddings

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
    # synonyms, filtered, antonyms = construct_synonyms_paper(annotations_dir='./data/annotations')
    df_synonyms = pda.read_csv('./data/annotations/synonyms/idiom_synonyms.tsv', sep='\t')
    df_antonyms = pda.read_csv('./data/annotations/synonyms/idiom_antonyms.tsv', sep='\t')

    df_synonyms = df_synonyms[
        (df_synonyms.query_id < len(chengyu_vocab)) & (df_synonyms.synonym_id < len(chengyu_vocab))]
    df_antonyms = df_antonyms[
        (df_antonyms.query_id < len(chengyu_vocab)) & (df_antonyms.antonym_id < len(chengyu_vocab))]

    chengyu_synonyms = {}
    for item in df_synonyms.itertuples():
        key = '>=2' if item.overlapping > 1 else '<=1'
        chengyu_synonyms.setdefault(key, {})
        query = getattr(item, 'query')
        chengyu_synonyms[key].setdefault(query, [])
        chengyu_synonyms[key][query].append(item.synonym)

    chengyu_antonyms = {}
    for item in df_antonyms.itertuples():
        key = '>=2' if item.overlapping > 1 else '<=1'
        chengyu_antonyms.setdefault(key, {})
        query = getattr(item, 'query')
        chengyu_antonyms[key].setdefault(query, [])
        chengyu_antonyms[key][query].append(item.antonym)

    emb_path = "./data/pretrained/Chinese-Word-Vectors/embeddings"
    _, iw, wi, dim = read_vectors(f'{emb_path}/{args.corpus}/sgns.{args.subtype}.word.bz2')

    if args.model_path is None:
        for t in ['word', 'char', 'bigram', 'bigram-char', 'BERT', 'ERNIE']:
            embeddings = load_embeddings(chengyu_vocab, emb_path, t, wi)

            i2w = [k for k in embeddings if k in chengyu_vocab]
            w2i = {w: i for i, w in enumerate(i2w)}
            vectors_np = np.array([embeddings[i2w[i]] for i in range(len(i2w))])

            cached = {}
            for k in chengyu_synonyms:
                print(f'==========================={t}-{k}===============================')
                evaluate_embeddings(i2w, w2i, vectors_np, chengyu_synonyms[k], chengyu_antonyms[k], cached)
    else:
        model = torch.load(args.model_path)
        embeddings_np = model['idiom_embedding.weight'].cpu().numpy()
        embeddings = {k: embeddings_np[v] for k, v in chengyu_vocab.items() if k in wi}

        i2w = [k for k in embeddings if k in chengyu_vocab]
        w2i = {w: i for i, w in enumerate(i2w)}
        vectors = np.array([embeddings[i2w[i]] for i in range(len(i2w))])

        cached = {}
        for k in chengyu_synonyms:
            print(f'==========================={k}===============================')
            evaluate_embeddings(i2w, w2i, vectors, chengyu_synonyms[k], chengyu_antonyms[k], cached)
