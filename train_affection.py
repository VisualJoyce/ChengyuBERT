"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""
import argparse
import glob
import os
import re
import shutil
from collections import Counter
from os.path import exists, join
from pprint import pprint
from time import time

import numpy as np
import pandas as pd
import torch
from horovod import torch as hvd
# from apex import amp
from nltk import Tree
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from chengyubert.data import create_dataloaders, calo_inverse_mapping, intermediate_dir, idioms_inverse_mapping
from chengyubert.data.dataset import DATA_REGISTRY
from chengyubert.data.evaluation import judge
from chengyubert.models import build_model
from chengyubert.optim import get_lr_sched
from chengyubert.optim.misc import build_optimizer
from chengyubert.utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                                           broadcast_tensors)
from chengyubert.utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from chengyubert.utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
from chengyubert.utils.save import ModelSaver, save_training_meta
from chengyubert.utils.tree import TreePrettyPrinter


def train(model, dataloaders, opts):
    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    scaler = GradScaler()

    global_step = 0
    if opts.rank == 0:
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps, desc=opts.model)
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
        os.makedirs(join(opts.output_dir, 'results'), exist_ok=True)  # store val predictions
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    LOGGER.info(f"***** Running training with {opts.n_gpu} GPUs *****")
    LOGGER.info("  Num examples = %d", len(dataloaders['train'].dataset))
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)

    running_loss = RunningMeter('loss')
    model.train()
    n_examples = 0
    n_epoch = 0
    best_ckpt = 0
    best_eval = 0
    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()
    while True:
        for step, batch in enumerate(dataloaders['train']):
            targets = batch['targets']
            n_examples += targets.size(0)

            with autocast():
                (_, over_loss,
                 select_masks,
                 # coarse_emotion_loss,
                 losses) = model(**batch, compute_loss=True)
                if over_loss is not None:
                    if opts.project == 'calo':
                        fine_emotion_loss, sentiment_emotion_loss = losses
                        loss = (over_loss + fine_emotion_loss + sentiment_emotion_loss).mean()
                    else:
                        sentiment_emotion_loss = losses
                        loss = (over_loss + sentiment_emotion_loss).mean()
                else:
                    if opts.project == 'calo':
                        fine_emotion_loss, sentiment_emotion_loss = losses
                        loss = (fine_emotion_loss + sentiment_emotion_loss).mean()
                    else:
                        sentiment_emotion_loss = losses
                        loss = sentiment_emotion_loss.mean()

            delay_unscale = (step + 1) % opts.gradient_accumulation_steps != 0
            scaler.scale(loss).backward()
            if not delay_unscale:
                # gather gradients from every processes
                # do this before unscaling to make sure every process uses
                # the same gradient scale
                grads = [p.grad.data for p in model.parameters()
                         if p.requires_grad and p.grad is not None]
                all_reduce_and_rescale_tensors(grads, float(1))

            running_loss(loss.item())

            if (step + 1) % opts.gradient_accumulation_steps == 0:
                global_step += 1

                # learning rate scheduling
                lr_this_step = get_lr_sched(global_step, opts)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                losses = all_gather_list(running_loss)
                running_loss = RunningMeter(
                    'loss', sum(l.val for l in losses) / len(losses))
                TB_LOGGER.add_scalar('loss', running_loss.val, global_step)
                TB_LOGGER.step()

                # update model params
                if opts.grad_norm != -1:
                    # Unscales the gradients of optimizer's assigned params in-place
                    scaler.unscale_(optimizer)
                    grad_norm = clip_grad_norm_(model.parameters(), opts.grad_norm)
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)

                # scaler.step() first unscales gradients of the optimizer's params.
                # If gradients don't contain infs/NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()
                optimizer.zero_grad()
                pbar.update(1)

                if global_step % 100 == 0:
                    # monitor training throughput
                    tot_ex = sum(all_gather_list(n_examples))
                    ex_per_sec = int(tot_ex / (time() - start))
                    LOGGER.info(f'{opts.model}: {n_epoch}-{global_step}: '
                                f'{tot_ex} examples trained at '
                                f'{ex_per_sec} ex/s \n'
                                f'over loss: {over_loss.mean() if over_loss is not None else over_loss} \n'
                                # f'coarse emotion loss: {coarse_emotion_loss.mean()} \n'
                                # f'fine emotion loss: {fine_emotion_loss.mean()} \n'
                                # f'sentiment loss: {sentiment_emotion_loss.mean()} \n'
                                f'best_acc-{best_eval * 100:.2f}')
                    TB_LOGGER.add_scalar('perf/ex_per_s',
                                         ex_per_sec, global_step)

                if global_step % opts.valid_steps == 0:
                    log = evaluation(model,
                                     dict(filter(lambda x: x[0].startswith('val'), dataloaders.items())),
                                     opts, global_step)
                    if log and log['val/acc'] > best_eval:
                        best_ckpt = global_step
                        best_eval = log['val/acc']
                        pbar.set_description(f'{opts.model}: {n_epoch}-{best_ckpt} best_acc-{best_eval * 100:.2f}')
                    model_saver.save(model, global_step)
            if global_step >= opts.num_train_steps:
                break
        if global_step >= opts.num_train_steps or global_step - best_ckpt > 0.1 * opts.num_train_steps:
            break
        n_epoch += 1
        LOGGER.info(f"Step {global_step}: finished {n_epoch} epochs")
    return best_ckpt


def idiom2tree(idiom, select_masks):
    # ans = list(idiom)
    ans = idiom
    for k, select_mask in enumerate(select_masks):
        for idx, v in enumerate(select_mask):
            if v == 1:
                c0 = ans.pop(idx)
                if isinstance(c0, Tree):
                    c0_label = c0.label()
                else:
                    c0_label = c0

                c1 = ans.pop(idx)
                if isinstance(c1, Tree):
                    c1_label = c1.label()
                else:
                    c1_label = c1

                ans.insert(idx, Tree(c0_label + c1_label, (c0, c1)))
            else:
                c = ans.pop(idx)
                if isinstance(c, Tree):
                    c_label = c.label()
                else:
                    c_label = c
                ans.insert(idx, Tree(c_label, (c,)))
    assert len(ans) == 2
    return ans


@torch.no_grad()
def validate_calo(opts, model, val_loader, split, global_step):
    val_loss = 0
    fine_emotion_score = 0
    sentiment_score = 0
    n_ex = 0
    val_mrr = 0
    st = time()
    results = []

    def get_header(key):
        d = calo_inverse_mapping[key]
        return [f'{key}_{d[v]}_{v}' if isinstance(d[v], str) else f'{key}_{d[v][-1]}_{v}' for v in range(len(d))]

    affection_results = []
    with tqdm(range(len(val_loader.dataset) // opts.size), desc=f'{split}-{opts.rank}') as tq:
        for i, batch in enumerate(val_loader):
            qids = batch['qids']
            targets = batch['targets']
            del batch['targets']
            del batch['qids']

            # select_masks, atts, composition_gates = composition
            if batch['input_ids'].dim() == 3:
                input_ids = torch.gather(batch['input_ids'][1], dim=1, index=batch['gather_index'][0])
            else:
                input_ids = torch.gather(batch['input_ids'], dim=1, index=batch['gather_index'])

            _, over_logits, select_masks, (fine_emotion_logits, sentiment_logits) = model(
                **batch, targets=None, compute_loss=False)

            idiom_targets = targets[:, 0]
            coarse_emotion_targets = targets[:, 1]
            fine_emotion_targets = targets[:, 2]
            sentiment_targets = targets[:, 3]

            fine_emotion_score += (
                    fine_emotion_logits.max(dim=-1, keepdim=False)[1] == fine_emotion_targets).sum().item()
            sentiment_score += (
                    sentiment_logits.max(dim=-1, keepdim=False)[1] == sentiment_targets).sum().item()

            if over_logits is not None:
                loss = F.cross_entropy(over_logits, idiom_targets, reduction='sum')
                val_loss += loss.item()
                # tot_score += (scores.max(dim=-1, keepdim=False)[1] == idiom_targets).sum().item()
                max_prob, max_idx = over_logits.max(dim=-1, keepdim=False)

                options = [val_loader.dataset.id2idiom[o] for o in val_loader.dataset.enlarged_candidates]
                for j, (qid, inp, position, answer) in enumerate(zip(qids,
                                                                     # idiom_targets,
                                                                     input_ids,
                                                                     # batch['option_ids'],
                                                                     batch['positions'],
                                                                     max_idx)):
                    # g = over_logits[j].cpu().numpy()
                    # top_k = np.argsort(-g)
                    # val_mrr += 1 / (1 + np.argwhere(top_k == target.item()).item())

                    example = val_loader.dataset.db[qid]
                    idiom = val_loader.dataset.id2idiom[example['idiom']]
                    # idiom = options[target.item()]
                    affection_results.append(
                        [idiom] + fine_emotion_logits[
                            j].cpu().numpy().tolist() + sentiment_logits[j].cpu().numpy().tolist()
                    )
                    if i % 1000 == 0 and select_masks is not None:
                        g = over_logits[j].cpu().numpy()
                        top_k = np.argsort(-g)[:5]
                        print(qid,
                              [options[k] for k in top_k],
                              idiom)
                        # print(len(select_masks), atts.size())
                        s_masks = [select_mask[j].long().cpu().numpy().tolist() for select_mask in select_masks]
                        # s_att = atts[j].cpu().numpy().tolist()

                        # tokens = val_loader.dataset.tokenizer.convert_ids_to_tokens(inp)
                        # start = tokens.index(val_loader.dataset.tokenizer.mask_token)
                        # tokens[position:position + len(idiom)] = list(idiom)
                        tokens = list(idiom)
                        # print(tokens, s_masks, s_att, composition_gates[j].sum())
                        print(tokens, s_masks)
                        try:
                            tree = Tree(' '.join(tokens), idiom2tree(tokens, s_masks))
                            print(TreePrettyPrinter(tree).text(unicodelines=True))
                        except:
                            pass

                        predictions = {
                            # "coarse emotion": {
                            #     "target": calo_inverse_mapping['coarse_emotion'].get(coarse_emotion_targets[j].item(),
                            #                                                          '无'),
                            #     "predictions": {calo_inverse_mapping['coarse_emotion'][k]: v for k, v in
                            #                     enumerate(coarse_emotion_logits[j].cpu().numpy().tolist())}
                            # },
                            "fine emotion": {
                                "target": calo_inverse_mapping['fine_emotion'].get(fine_emotion_targets[j].item(), '无'),
                                "predictions": {calo_inverse_mapping['fine_emotion'][k]: v for k, v in
                                                enumerate(fine_emotion_logits[j].cpu().numpy().tolist())}
                            },
                            "sentiment": {
                                "target": calo_inverse_mapping['sentiment'].get(sentiment_targets[j].item(), '无'),
                                "predictions": {calo_inverse_mapping['sentiment'][k]: v for k, v in
                                                enumerate(sentiment_logits[j].cpu().numpy().tolist())}
                            }
                        }
                        pprint(predictions)

                answers = max_idx.cpu().tolist()
                results.extend(zip(qids, answers))
            else:
                for j, (qid, inp, position) in enumerate(zip(qids, input_ids,
                                                             # batch['option_ids'],
                                                             batch['positions'],
                                                             )):
                    # options = [val_loader.dataset.id2idiom[o.item()] for o in option_ids]
                    example = val_loader.dataset.db[qid]
                    idiom = val_loader.dataset.id2idiom[example['idiom']]
                    affection_results.append(
                        [idiom] + fine_emotion_logits[
                            j].cpu().numpy().tolist() + sentiment_logits[j].cpu().numpy().tolist()
                    )
                    if i % 1000 == 0 and select_masks is not None:
                        print(qid,
                              idiom)
                        s_masks = [select_mask[j].long().cpu().numpy().tolist() for select_mask in select_masks]
                        tokens = list(idiom)
                        # print(tokens, s_masks, s_att, composition_gates[j].sum())
                        print(tokens, s_masks)
                        try:
                            tree = Tree(' '.join(tokens), idiom2tree(tokens, s_masks))
                            print(TreePrettyPrinter(tree).text(unicodelines=True))
                        except:
                            pass

                        predictions = {
                            # "coarse emotion": {
                            #     "target": calo_inverse_mapping['coarse_emotion'].get(coarse_emotion_targets[j].item(),
                            #                                                          '无'),
                            #     "predictions": {calo_inverse_mapping['coarse_emotion'][k]: v for k, v in
                            #                     enumerate(coarse_emotion_logits[j].cpu().numpy().tolist())}
                            # },
                            "fine emotion": {
                                "target": calo_inverse_mapping['fine_emotion'].get(fine_emotion_targets[j].item(), '无'),
                                "predictions": {calo_inverse_mapping['fine_emotion'][k]: v for k, v in
                                                enumerate(fine_emotion_logits[j].cpu().numpy().tolist())}
                            },
                            "sentiment": {
                                "target": calo_inverse_mapping['sentiment'].get(sentiment_targets[j].item(), '无'),
                                "predictions": {calo_inverse_mapping['sentiment'][k]: v for k, v in
                                                enumerate(sentiment_logits[j].cpu().numpy().tolist())}
                            }
                        }
                        pprint(predictions)

            n_ex += len(qids)
            tq.update(len(qids))

    if results:
        out_file = f'{opts.output_dir}/results/{split}_results_{global_step}_rank{opts.rank}.csv'
        with open(out_file, 'w') as f:
            for id_, ans in results:
                f.write(f'{id_},{ans}\n')

    header = ['idiom'] + get_header('fine_emotion') + get_header('sentiment')
    if affection_results:
        out_file = f'{opts.output_dir}/results/{split}_affection_results_{global_step}_rank{opts.rank}.csv'
        pd.DataFrame(affection_results, columns=header).to_csv(out_file)

    val_loss = sum(all_gather_list(val_loss))
    val_mrr = sum(all_gather_list(val_mrr))

    # val_coarse_emotion_score = sum(all_gather_list(coarse_emotion_score))
    val_fine_emotion_score = sum(all_gather_list(fine_emotion_score))
    val_sentiment_score = sum(all_gather_list(sentiment_score))

    n_ex = sum(all_gather_list(n_ex))
    tot_time = time() - st

    val_loss /= n_ex
    val_mrr = val_mrr / n_ex
    # val_coarse_emotion_score = val_coarse_emotion_score / n_ex
    val_fine_emotion_score = val_fine_emotion_score / n_ex
    val_sentiment_score = val_sentiment_score / n_ex

    if results:
        out_file = f'{opts.output_dir}/results/{split}_results_{global_step}.csv'
        if not os.path.isfile(out_file):
            with open(out_file, 'wb') as g:
                for f in glob.glob(f'{opts.output_dir}/results/{split}_results_{global_step}_rank*.csv'):
                    shutil.copyfileobj(open(f, 'rb'), g)

        sum(all_gather_list(opts.rank))

        txt_db = os.path.join('/txt',
                              intermediate_dir(opts.pretrained_model_name_or_path),
                              getattr(opts, f'{split}_txt_db'))
        val_acc = judge(out_file, f'{txt_db}/answer.csv')

    if opts.rank == 0:
        results_files = glob.glob(f'{opts.output_dir}/results/{split}_affection_results_{global_step}_rank*.csv')
        new_affection_results_df = pd.concat(map(pd.read_csv, results_files))
        idiom_num = new_affection_results_df['idiom'].unique().size
        idiom_wise_accs = {}
        for item in new_affection_results_df.groupby('idiom').mean().reset_index().to_dict(orient='records'):
            idiom = item['idiom']
            idiom_id = val_loader.dataset.chengyu_vocab[idiom]
            affections = val_loader.dataset.calo_vocab[idiom_id][0]
            for sub_type in ['fine_emotion', 'sentiment']:
                d = {k: v for k, v in item.items() if k.startswith(sub_type)}
                key = max(d, key=d.get)
                _, pred = key.rsplit('_', 1)
                target = affections[sub_type]
                idiom_wise_accs.setdefault(sub_type, 0)
                idiom_wise_accs[sub_type] += (int(pred) == target) / idiom_num * 100

        val_acc = (val_fine_emotion_score + val_sentiment_score) / 2

        val_log = {f'{split}/loss': val_loss,
                   f'{split}/acc': val_acc,
                   f'{split}/fine_emotion': val_fine_emotion_score * 100,
                   f'{split}/sentiment': val_sentiment_score * 100,
                   f'{split}/mrr': val_mrr,
                   f'{split}/ex_per_s': n_ex / tot_time}

        for k, v in idiom_wise_accs.items():
            val_log[f'{split}/{k}'] = v

        LOGGER.info(f"validation finished in {int(tot_time)} seconds, \n"
                    # f"coarse emotion score: {val_coarse_emotion_score * 100:.2f}, \n"
                    f"fine emotion score: {val_fine_emotion_score * 100:.2f}, \n"
                    f"sentiment score: {val_sentiment_score * 100:.2f}, \n"
                    f"score: {val_acc * 100:.2f}, \n"
                    f"idiom-wise score: {idiom_wise_accs}, "
                    f"mrr: {val_mrr:.3f}")
        return val_log


@torch.no_grad()
def validate_slide(opts, model, val_loader, split, global_step):
    val_loss = 0
    sentiment_score = 0
    n_ex = 0
    val_mrr = 0
    st = time()
    results = []

    def get_header(key):
        d = idioms_inverse_mapping[key]
        return [f'{key}_{d[v]}_{v}' if isinstance(d[v], str) else f'{key}_{d[v][-1]}_{v}' for v in range(len(d))]

    affection_results = []
    with tqdm(range(len(val_loader.dataset) // opts.size), desc=f'{split}-{opts.rank}') as tq:
        for i, batch in enumerate(val_loader):
            qids = batch['qids']
            targets = batch['targets']
            del batch['targets']
            del batch['qids']

            # select_masks, atts, composition_gates = composition
            if batch['input_ids'].dim() == 3:
                input_ids = torch.gather(batch['input_ids'][1], dim=1, index=batch['gather_index'][0])
            else:
                input_ids = torch.gather(batch['input_ids'], dim=1, index=batch['gather_index'])

            _, over_logits, select_masks, sentiment_logits = model(
                **batch, targets=None, compute_loss=False)

            idiom_targets = targets[:, 0]
            sentiment_targets = targets[:, 1]

            sentiment_score += (
                    sentiment_logits.max(dim=-1, keepdim=False)[1] == sentiment_targets).sum().item()

            if over_logits is not None:
                loss = F.cross_entropy(over_logits, idiom_targets, reduction='sum')
                val_loss += loss.item()
                # tot_score += (scores.max(dim=-1, keepdim=False)[1] == idiom_targets).sum().item()
                max_prob, max_idx = over_logits.max(dim=-1, keepdim=False)

                options = [val_loader.dataset.id2idiom[o] for o in val_loader.dataset.enlarged_candidates]
                for j, (qid, inp, position, answer) in enumerate(zip(qids,
                                                                     # idiom_targets,
                                                                     input_ids,
                                                                     # batch['option_ids'],
                                                                     batch['positions'],
                                                                     max_idx)):
                    # g = over_logits[j].cpu().numpy()
                    # top_k = np.argsort(-g)
                    # val_mrr += 1 / (1 + np.argwhere(top_k == target.item()).item())

                    example = val_loader.dataset.db[qid]
                    idiom = val_loader.dataset.id2idiom[example['idiom']]
                    # idiom = options[target.item()]
                    affection_results.append(
                        [idiom] + sentiment_logits[j].cpu().numpy().tolist()
                    )
                    if i % 1000 == 0:
                        g = over_logits[j].cpu().numpy()
                        top_k = np.argsort(-g)[:5]
                        print(qid,
                              [options[k] for k in top_k],
                              idiom)
                        # print(len(select_masks), atts.size())
                        if select_masks is not None:
                            s_masks = [select_mask[j].long().cpu().numpy().tolist() for select_mask in select_masks]
                            # s_att = atts[j].cpu().numpy().tolist()

                            # tokens = val_loader.dataset.tokenizer.convert_ids_to_tokens(inp)
                            # start = tokens.index(val_loader.dataset.tokenizer.mask_token)
                            # tokens[position:position + len(idiom)] = list(idiom)
                            tokens = val_loader.dataset.tokenizer.convert_ids_to_tokens(
                                val_loader.dataset.idiom_input_ids[qid])
                            # print(tokens, s_masks, s_att, composition_gates[j].sum())
                            print(tokens, s_masks)
                            try:
                                tree = Tree(' '.join(tokens), idiom2tree(tokens, s_masks))
                                print(TreePrettyPrinter(tree).text(unicodelines=True))
                            except:
                                pass

                        predictions = {
                            # "coarse emotion": {
                            #     "target": calo_inverse_mapping['coarse_emotion'].get(coarse_emotion_targets[j].item(),
                            #                                                          '无'),
                            #     "predictions": {calo_inverse_mapping['coarse_emotion'][k]: v for k, v in
                            #                     enumerate(coarse_emotion_logits[j].cpu().numpy().tolist())}
                            # },
                            "sentiment": {
                                "target": idioms_inverse_mapping['sentiment'].get(sentiment_targets[j].item(), '无'),
                                "predictions": {idioms_inverse_mapping['sentiment'][k]: v for k, v in
                                                enumerate(sentiment_logits[j].cpu().numpy().tolist())}
                            }
                        }
                        pprint(predictions)

                answers = max_idx.cpu().tolist()
                results.extend(zip(qids, answers))
            else:
                for j, (qid, inp, position) in enumerate(zip(qids, input_ids,
                                                             # batch['option_ids'],
                                                             batch['positions'],
                                                             )):
                    # options = [val_loader.dataset.id2idiom[o.item()] for o in option_ids]
                    example = val_loader.dataset.db[qid]
                    idiom = val_loader.dataset.id2idiom[example['idiom']]
                    affection_results.append(
                        [idiom] + sentiment_logits[j].cpu().numpy().tolist()
                    )
                    if i % 1000 == 0:
                        print(qid,
                              idiom)
                        if select_masks is not None:
                            s_masks = [select_mask[j].long().cpu().numpy().tolist() for select_mask in select_masks]
                            tokens = val_loader.dataset.tokenizer.convert_ids_to_tokens(
                                val_loader.dataset.idiom_input_ids[qid])
                            # print(tokens, s_masks, s_att, composition_gates[j].sum())
                            print(tokens, s_masks)
                            try:
                                tree = Tree(' '.join(tokens), idiom2tree(tokens, s_masks))
                                print(TreePrettyPrinter(tree).text(unicodelines=True))
                            except:
                                pass

                        predictions = {
                            "sentiment": {
                                "target": idioms_inverse_mapping['sentiment'].get(sentiment_targets[j].item(), '无'),
                                "predictions": {idioms_inverse_mapping['sentiment'][k]: v for k, v in
                                                enumerate(sentiment_logits[j].cpu().numpy().tolist())}
                            }
                        }
                        pprint(predictions)

            n_ex += len(qids)
            tq.update(len(qids))

    if results:
        out_file = f'{opts.output_dir}/results/{split}_results_{global_step}_rank{opts.rank}.csv'
        with open(out_file, 'w') as f:
            for id_, ans in results:
                f.write(f'{id_},{ans}\n')

    header = ['idiom'] + get_header('sentiment')
    if affection_results:
        out_file = f'{opts.output_dir}/results/{split}_affection_results_{global_step}_rank{opts.rank}.csv'
        pd.DataFrame(affection_results, columns=header).to_csv(out_file)

    val_loss = sum(all_gather_list(val_loss))
    val_mrr = sum(all_gather_list(val_mrr))

    val_sentiment_score = sum(all_gather_list(sentiment_score))

    n_ex = sum(all_gather_list(n_ex))
    tot_time = time() - st

    val_loss /= n_ex
    val_mrr = val_mrr / n_ex
    val_sentiment_score = val_sentiment_score / n_ex

    if results:
        out_file = f'{opts.output_dir}/results/{split}_results_{global_step}.csv'
        if not os.path.isfile(out_file):
            with open(out_file, 'wb') as g:
                for f in glob.glob(f'{opts.output_dir}/results/{split}_results_{global_step}_rank*.csv'):
                    shutil.copyfileobj(open(f, 'rb'), g)

        sum(all_gather_list(opts.rank))

        txt_db = os.path.join('/txt',
                              intermediate_dir(opts.pretrained_model_name_or_path),
                              getattr(opts, f'{split}_txt_db'))
        val_acc = judge(out_file, f'{txt_db}/answer.csv')

    if opts.rank == 0:
        results_files = glob.glob(f'{opts.output_dir}/results/{split}_affection_results_{global_step}_rank*.csv')
        new_affection_results_df = pd.concat(map(pd.read_csv, results_files))
        idiom_num = new_affection_results_df['idiom'].unique().size
        idiom_wise_accs = {}
        for item in new_affection_results_df.groupby('idiom').mean().reset_index().to_dict(orient='records'):
            idiom = item['idiom']
            idiom_id = val_loader.dataset.vocab[idiom]
            for sub_type in ['sentiment']:
                d = {k: v for k, v in item.items() if k.startswith(sub_type)}
                key = max(d, key=d.get)
                _, pred = key.rsplit('_', 1)
                target = val_loader.dataset.sentiments[idiom_id]
                idiom_wise_accs.setdefault(sub_type, 0)
                idiom_wise_accs[sub_type] += (int(pred) == target) / idiom_num * 100

        val_acc = val_sentiment_score

        val_log = {f'{split}/loss': val_loss,
                   f'{split}/acc': val_acc,
                   f'{split}/sentiment': val_sentiment_score * 100,
                   f'{split}/mrr': val_mrr,
                   f'{split}/ex_per_s': n_ex / tot_time}

        for k, v in idiom_wise_accs.items():
            val_log[f'{split}/{k}'] = v

        LOGGER.info(f"validation finished in {int(tot_time)} seconds, \n"
                    # f"coarse emotion score: {val_coarse_emotion_score * 100:.2f}, \n"
                    f"sentiment score: {val_sentiment_score * 100:.2f}, \n"
                    f"score: {val_acc * 100:.2f}, \n"
                    f"idiom-wise score: {idiom_wise_accs}, "
                    f"mrr: {val_mrr:.3f}")
        return val_log


validate = {
    'calo': validate_calo,
    'slide': validate_slide
}


def evaluation(model, data_loaders: dict, opts, global_step):
    model.eval()
    log = {}
    for split, loader in data_loaders.items():
        LOGGER.info(f"Step {global_step}: start running "
                    f"validation on {split} split...")
        log.update(validate[opts.project](opts, model, loader, split, global_step))
    TB_LOGGER.log_scaler_dict(log)
    model.train()
    return log


def get_best_ckpt(val_data_dir, opts):
    pat = re.compile(r'val_results_(?P<step>\d+)_rank0.csv')
    prediction_files = glob.glob('{}/results/val_results_*_rank0.csv'.format(opts.output_dir))

    top_files = Counter()
    for f in prediction_files:
        acc = judge(f, os.path.join(val_data_dir, 'answer.csv'))
        top_files.update({f: acc})

    print(top_files)

    for f, acc in top_files.most_common(1):
        m = pat.match(os.path.basename(f))
        best_epoch = int(m.group('step'))
        return best_epoch


def main(opts):
    device = torch.device("cuda", hvd.local_rank())
    torch.cuda.set_device(hvd.local_rank())
    rank = hvd.rank()
    opts.rank = rank
    opts.size = hvd.size()
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
        device, n_gpu, hvd.rank(), opts.fp16))

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
            opts.gradient_accumulation_steps))

    set_random_seed(opts.seed)

    # data loaders
    DatasetCls = DATA_REGISTRY[opts.dataset_cls]
    EvalDatasetCls = DATA_REGISTRY[opts.eval_dataset_cls]
    opts.evaluate_embedding = False
    splits, dataloaders = create_dataloaders(DatasetCls, EvalDatasetCls, opts)

    if opts.project == 'calo':
        setattr(opts, 'fine_emotion_weights', dataloaders['train'].dataset.fine_emotion_weights)
        setattr(opts, 'sentiment_weights', dataloaders['train'].dataset.sentiment_weights)
    else:
        opts.weight = torch.tensor(opts.weight)

    # Prepare model
    model = build_model(opts)
    model.to(device)

    if opts.project == 'calo':
        opts.fine_emotion_weights = opts.fine_emotion_weights.tolist()
        opts.sentiment_weights = opts.sentiment_weights.tolist()
    else:
        opts.weight = opts.weight.tolist()

    if opts.mode == 'train':
        best_ckpt = train(model, dataloaders, opts)
    else:
        best_ckpt = get_best_ckpt(dataloaders['val'].dataset.db_dir, opts)

    best_pt = f'{opts.output_dir}/ckpt/model_step_{best_ckpt}.pt'
    model.load_state_dict(torch.load(best_pt), strict=False)
    evaluation(model, dict(filter(lambda x: x[0] != 'train', dataloaders.items())), opts, best_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--model_config",
                        default=None, type=str,
                        help="json file for model architecture")
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained model")
    parser.add_argument("--model", default='paired',
                        choices=['snlive'],
                        help="choose from 2 model architecture")
    parser.add_argument("--mode", default='train',
                        choices=['train', 'infer'],
                        help="choose from 2 mode")

    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model checkpoints will be "
                             "written.")

    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')

    # training parameters
    parser.add_argument("--train_batch_size",
                        default=4096, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--val_batch_size",
                        default=4096, type=int,
                        help="Total batch size for validation. "
                             "(batch by tokens)")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--valid_steps",
                        default=1000,
                        type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps",
                        default=100000,
                        type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adam',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+', type=float,
                        help="beta for adam optimizer")
    parser.add_argument("--dropout",
                        default=0.1,
                        type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm",
                        default=0.25,
                        type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps",
                        default=4000,
                        type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for.")

    # device parameters
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    args = parse_with_config(parser)

    checkpoint = os.path.basename(args.pretrained_model_name_or_path)

    hvd.init()
    n_gpu = hvd.size()
    args.n_gpu = n_gpu

    base_dir = '_'.join(['affection',
                         f'{args.n_gpu * args.gradient_accumulation_steps}',
                         f'{args.num_train_steps}',
                         f'{args.learning_rate}',
                         f'{args.dropout}',
                         f'{args.weight_decay}'])
    args.output_dir = os.path.join(args.output_dir,
                                   f'{args.model}_context-{args.use_context}',
                                   os.path.basename(args.pretrained_model_name_or_path),
                                   os.path.basename(args.config),
                                   base_dir)

    if exists(args.output_dir) and os.listdir(f'{args.output_dir}/ckpt'):
        if args.mode == 'train':
            raise ValueError("Output directory ({}) already exists and is not "
                             "empty.".format(args.output_dir))

    main(args)
