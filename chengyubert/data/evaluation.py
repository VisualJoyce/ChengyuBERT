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
