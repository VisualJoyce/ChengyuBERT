"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER finetuning for NLVR2
"""
from os.path import exists, join

import argparse
import json
import numpy as np
import os
import torch
from apex import amp
from horovod import torch as hvd
from time import time
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from transformers import BertConfig

from chengyubert.data import ChengyuDataset, ChengyuEvalDataset, chengyu_collate, chengyu_eval_collate, \
    create_dataloaders
from chengyubert.modeling_bert import ChengyuBert, BertForClozeSingle, BertForClozeDual
from chengyubert.optim import get_lr_sched
from chengyubert.optim.misc import build_optimizer
from chengyubert.utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                                           broadcast_tensors)
from chengyubert.utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from chengyubert.utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
from chengyubert.utils.save import ModelSaver, save_training_meta


def main(opts):
    hvd.init()
    n_gpu = hvd.size()
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

    DatasetCls = ChengyuDataset
    EvalDatasetCls = ChengyuEvalDataset
    collate_fn = chengyu_collate
    eval_collate_fn = chengyu_eval_collate

    if opts.model == 'bertsingle':
        ModelCls = BertForClozeSingle
    elif opts.model == 'bertdual':
        ModelCls = BertForClozeDual
    else:
        ModelCls = ChengyuBert

    # data loaders
    splits, dataloaders = create_dataloaders(LOGGER, DatasetCls, EvalDatasetCls, collate_fn, eval_collate_fn, opts)

    # Prepare model
    bert_config = BertConfig.from_json_file(args.model_config)
    model = ModelCls.from_pretrained(opts.checkpoint,
                                     config=bert_config,
                                     len_idiom_vocab=opts.len_idiom_vocab, model_name=opts.model)
    model.to(device)
    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    model, optimizer = amp.initialize(model, optimizer,
                                      enabled=opts.fp16, opt_level='O2')

    global_step = 0
    if rank == 0:
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

    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
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

            loss = model(**batch,
                         compute_loss=True)
            loss = loss.mean()
            delay_unscale = (step + 1) % opts.gradient_accumulation_steps != 0
            with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale
                                ) as scaled_loss:
                scaled_loss.backward()
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
                    grad_norm = clip_grad_norm_(amp.master_params(optimizer),
                                                opts.grad_norm)
                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)
                optimizer.step()
                optimizer.zero_grad()
                pbar.update(1)

                if global_step % 100 == 0:
                    # monitor training throughput
                    tot_ex = sum(all_gather_list(n_examples))
                    ex_per_sec = int(tot_ex / (time() - start))
                    LOGGER.info(f'Step {global_step}: '
                                f'{tot_ex} examples trained at '
                                f'{ex_per_sec} ex/s')
                    TB_LOGGER.add_scalar('perf/ex_per_s',
                                         ex_per_sec, global_step)

                if global_step % opts.valid_steps == 0:
                    log = evaluation(model, model_saver,
                                     dict(filter(lambda x: x[0].startswith('val'), dataloaders.items())),
                                     opts,
                                     rank, global_step, save_model=global_step > opts.num_train_steps // 2)
                    if log['val/acc'] > best_eval:
                        best_ckpt = global_step
                        best_eval = log['val/acc']
            if global_step >= opts.num_train_steps:
                break
        if global_step >= opts.num_train_steps:
            break
        n_epoch += 1
        LOGGER.info(f"Step {global_step}: finished {n_epoch} epochs")

    best_pt = f'{opts.output_dir}/ckpt/model_step_{best_ckpt}.pt'
    model.load_state_dict(torch.load(best_pt), strict=False)
    evaluation(model, model_saver,
               dict(filter(lambda x: x[0] != 'train', dataloaders.items())),
               opts, rank, best_ckpt, save_model=False)


def evaluation(model, model_saver, data_loaders: dict, opts, rank, global_step, save_model):
    model.eval()
    log = {}
    for split, loader in data_loaders.items():
        LOGGER.info(f"Step {global_step}: start running "
                    f"validation on {split} split...")
        out_file = f'{opts.output_dir}/results/{split}_results_{global_step}_rank{rank}.csv'
        log.update(validate(opts, model, loader, split, out_file))
    TB_LOGGER.log_scaler_dict(log)
    if save_model:
        model_saver.save(model, global_step)
    model.train()
    return log


def judge(pred_file, answer_file):
    if isinstance(pred_file, str):
        pred = open(pred_file).readlines()
    else:
        pred = pred_file.readlines()

    ans = open(answer_file).readlines()
    assert len(pred) == len(ans)

    ans_dict = {}
    pred_dict = {}
    for line in ans:
        line = line.strip().split(',')
        ans_dict[line[0]] = int(line[1])
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


@torch.no_grad()
def validate(opts, model, val_loader, split, out_file):
    val_loss = 0
    tot_score = 0
    n_ex = 0
    val_mrr = 0
    st = time()
    results = []
    with tqdm(range(len(val_loader.dataset))) as tq:
        for i, batch in enumerate(val_loader):
            qids = batch['qids']
            targets = batch['targets']
            del batch['targets']
            del batch['qids']

            scores, over_logits = model(**batch, targets=None, compute_loss=False)
            loss = F.cross_entropy(scores, targets, reduction='sum')
            val_loss += loss.item()
            tot_score += (scores.max(dim=-1, keepdim=False)[1] == targets).sum().item()
            max_prob, max_idx = scores.max(dim=-1, keepdim=False)
            answers = max_idx.cpu().tolist()

            targets = torch.gather(batch['option_ids'], dim=1, index=targets.unsqueeze(1)).cpu().numpy()
            for j, (qid, target) in enumerate(zip(qids, targets)):
                g = over_logits[j].cpu().numpy()
                top_k = np.argsort(-g)
                val_mrr += 1 / (1 + np.argwhere(top_k == target).item())

            results.extend(zip(qids, answers))
            n_ex += len(qids)
            tq.update(len(qids))

    val_loss = sum(all_gather_list(val_loss))
    tot_score = sum(all_gather_list(tot_score))
    n_ex = sum(all_gather_list(n_ex))
    tot_time = time() - st
    val_loss /= n_ex
    val_mrr = val_mrr / n_ex

    with open(out_file, 'w') as f:
        for id_, ans in results:
            f.write(f'{id_},{ans}\n')

    txt_db = getattr(opts, f'{split}_txt_db')
    val_acc = judge(out_file, f'{txt_db}/answer.csv')

    val_log = {f'{split}/loss': val_loss,
               f'{split}/acc': val_acc,
               f'{split}/mrr': val_mrr,
               f'{split}/ex_per_s': n_ex / tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc * 100:.2f}"
                f"score: {val_mrr:.3f}")
    return val_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--train_txt_db",
                        default=None, type=str,
                        help="The input train corpus. (LMDB)")
    parser.add_argument("--train_img_dir",
                        default=None, type=str,
                        help="The input train images.")
    parser.add_argument("--val_txt_db",
                        default=None, type=str,
                        help="The input validation corpus. (LMDB)")
    parser.add_argument("--val_img_dir",
                        default=None, type=str,
                        help="The input validation images.")
    parser.add_argument("--test_txt_db",
                        default=None, type=str,
                        help="The input test corpus. (LMDB)")
    parser.add_argument("--test_img_dir",
                        default=None, type=str,
                        help="The input test images.")
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
    parser.add_argument('--use_img_type', action='store_true',
                        help="expand the type embedding for 2 image types")

    parser.add_argument("--output_dir", default=None, type=str,
                        help="The output directory where the model checkpoints will be "
                             "written.")

    parser.add_argument('--use_distill', action='store_true',
                        help="expand the type embedding for 2 image types")
    parser.add_argument('--distill_temp', type=float, default=None,
                        help="expand the type embedding for 2 image types")
    parser.add_argument('--distill_alpha', type=float, default=None,
                        help="expand the type embedding for 2 image types")
    parser.add_argument("--teacher_model_path",
                        default=None, type=str,
                        help="json file for model architecture")
    parser.add_argument("--teacher_checkpoint",
                        default=None, type=str,
                        help="pretrained model")

    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')

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
    if args.use_distill:
        distill = f'distill_temp-{args.distill_temp}_alpha-{args.distill_alpha}_step-{args.teacher_checkpoint}_'
        args.teacher_model_config = os.path.join(args.teacher_model_path, 'log', 'model.json')
        args.teacher_model_pt = os.path.join(args.teacher_model_path, 'ckpt',
                                             f'model_step_{args.teacher_checkpoint}.pt')
    else:
        distill = ''

    args.output_dir = os.path.join(args.output_dir, args.model, f'{distill}{args.num_train_steps}_{args.learning_rate}')
    if exists(args.output_dir) and os.listdir(f'{args.output_dir}/results'):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output_dir))

    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
