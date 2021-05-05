from itertools import chain

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


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


def evaluate_embeddings(iw, wi, vectors_np,
                        chengyu_synonyms_dict,
                        chengyu_antonyms_dict,
                        cached
                        ):
    cnt = 0
    recall_at_k_cosine = {}
    antonyms_at_k_cosine = {}
    recall_at_k_norm = {}
    antonyms_at_k_norm = {}
    set_recall_at_k_cosine = {}
    set_recall_at_k_norm = {}

    set_recall_at_delta_norm = {}
    set_recall_at_cosine_norm = {}

    mrr_cosine = 0
    mrr_norm = 0

    k_list = [1, 3, 5, 10]

    delta_list = [0.5, 1, 2, 3, 5]
    cosine_delta_list = list([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    total = len(chengyu_synonyms_dict)
    top_k = {}
    for w, wll in tqdm(chengyu_synonyms_dict.items()):
        wl_new = [x for x in wll if x in wi]
        if w in wi and wl_new:
            cnt += 1

            neighbourhood_cosine = []
            neighbourhood_norm = []
            neighbourhood_delta_cosine = {}
            neighbourhood_delta_norm = {}

            for x in [w] + wl_new:
                embedding_x = vectors_np[wi[x]]
                if x not in cached:
                    cosine_distances = (1 - cosine_similarity(embedding_x.reshape(1, -1), vectors_np)[0])
                    norm_distances = np.linalg.norm(vectors_np - embedding_x, axis=1)
                    cached[x] = (
                        cosine_distances.argsort(), norm_distances.argsort(), cosine_distances, norm_distances)

                cosine_distances_sort, norm_distances_sort, cosine_distances, norm_distances = cached[x]

                neighbourhood_cosine.append([idx for idx in cosine_distances_sort])
                neighbourhood_norm.append([idx for idx in norm_distances_sort])

                for delta in delta_list:
                    neighbourhood_delta_norm.setdefault(delta, [])
                    neighbourhood_delta_norm[delta].append(
                        [idx.item() for idx in np.argwhere(norm_distances <= delta)])

                for delta in cosine_delta_list:
                    neighbourhood_delta_cosine.setdefault(delta, [])
                    neighbourhood_delta_cosine[delta].append(
                        [idx.item() for idx in np.argwhere(cosine_distances <= delta)])

            for k in k_list:
                set_recall_at_k_cosine.setdefault(k, 0)
                set_recall_at_k_cosine[k] += len(set.intersection(*[set(ns[:k]) for ns in neighbourhood_cosine])) / len(
                    set.union(*[set(ns[:k]) for ns in neighbourhood_cosine]))

                set_recall_at_k_norm.setdefault(k, 0)
                set_recall_at_k_norm[k] += len(set.intersection(*[set(ns[:k]) for ns in neighbourhood_norm])) / len(
                    set.union(*[set(ns[:k]) for ns in neighbourhood_norm]))

            for delta in delta_list:
                set_recall_at_delta_norm.setdefault(delta, 0)
                set_recall_at_delta_norm[delta] += len(
                    set.intersection(*[set(ns) for ns in neighbourhood_delta_norm[delta]])) / len(
                    set.union(*[set(ns) for ns in neighbourhood_delta_norm[delta]]))

            for delta in cosine_delta_list:
                set_recall_at_cosine_norm.setdefault(delta, 0)
                set_recall_at_cosine_norm[delta] += len(
                    set.intersection(*[set(ns) for ns in neighbourhood_delta_cosine[delta]])) / len(
                    set.union(*[set(ns) for ns in neighbourhood_delta_cosine[delta]]))

            cosine_distances_sort, norm_distances_sort, cosine_distances, norm_distances = cached[w]

            cids = [idx for idx in cosine_distances_sort if w != iw[idx]]
            nids = [idx for idx in norm_distances_sort if w != iw[idx]]

            for k in k_list:
                top_ids = cids[:k]
                recall_at_k_cosine.setdefault(k, 0)
                recall_at_k_cosine[k] += sum([1 for idx in top_ids if iw[idx] in wl_new]) / len(wl_new)

                antonyms_at_k_cosine.setdefault(k, 0)
                antonyms_at_k_cosine[k] += sum([1 for idx in top_ids if iw[idx] in chengyu_antonyms_dict.get(w, [])])

                top_ids = nids[:k]
                recall_at_k_norm.setdefault(k, 0)
                recall_at_k_norm[k] += sum([1 for idx in top_ids if iw[idx] in wl_new]) / len(wl_new)

                antonyms_at_k_norm.setdefault(k, 0)
                antonyms_at_k_norm[k] += sum([1 for idx in top_ids if iw[idx] in chengyu_antonyms_dict.get(w, [])])

            mrr_cosine += sum([1 / (1 + cids.index(wi[x])) for x in wl_new if x in wi]) / len(wl_new)
            mrr_norm += sum([1 / (1 + nids.index(wi[x])) for x in wl_new if x in wi]) / len(wl_new)

    print(cnt, ' entries appeared in the training dictionary , total word pairs ', total)
    print(recall_at_k_cosine)
    print(recall_at_k_norm)
    print(antonyms_at_k_cosine)
    for k in recall_at_k_cosine:
        recall_at_k_cosine[k] /= cnt
        recall_at_k_norm[k] /= cnt
        set_recall_at_k_cosine[k] /= cnt
        set_recall_at_k_norm[k] /= cnt

    for delta in delta_list:
        set_recall_at_delta_norm[delta] /= cnt
    for delta in cosine_delta_list:
        set_recall_at_cosine_norm[delta] /= cnt

    print('\t'.join(chain([format(recall_at_k_cosine[k], "0.6f") for k in k_list],
                          [format(antonyms_at_k_cosine[k], "d") for k in k_list],
                          [format(recall_at_k_norm[k], "0.6f") for k in k_list],
                          [format(antonyms_at_k_norm[k], "d") for k in k_list],
                          [format(set_recall_at_delta_norm[k], "0.6f") for k in delta_list],
                          [format(set_recall_at_k_cosine[k], "0.6f") for k in k_list],
                          [format(set_recall_at_k_norm[k], "0.6f") for k in k_list],
                          [format(mrr, "0.6f") for mrr in [mrr_cosine / cnt, mrr_norm / cnt]],
                          [format(set_recall_at_cosine_norm[k], "0.6f") for k in cosine_delta_list],
                          )))
    print(recall_at_k_cosine)
    print(recall_at_k_norm)
    print(set_recall_at_k_cosine)
    print(set_recall_at_k_norm)
    print(set_recall_at_delta_norm)
    return cnt, total, recall_at_k_cosine, recall_at_k_norm, top_k
