"""
Copyright (c) VisualJoyce.
Licensed under the MIT license.
"""
import argparse
import glob
import json
import os
import shutil
from os.path import exists, join
from time import time

import numpy as np
import torch
from horovod import torch as hvd
from scipy.optimize import linear_sum_assignment
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from chengyubert.data import create_dataloaders, intermediate_dir
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
    splits, dataloaders = create_dataloaders(DatasetCls, EvalDatasetCls, opts)

    # Prepare model
    model = build_model(opts)
    model.to(device)
    # make sure every process has same model parameters in the beginning
    broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    scaler = GradScaler()

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
            del batch['gather_index']
            n_examples += targets.size(0)

            with autocast():
                original_loss, enlarged_loss = model(**batch, compute_loss=True)
                if opts.candidates == 'original':
                    loss = original_loss
                elif opts.candidates == 'enlarged':
                    loss = enlarged_loss
                elif opts.candidates == 'combined':
                    loss = original_loss + enlarged_loss
                else:
                    raise AssertionError("No such loss!")

                loss = loss.mean()

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
                                f'{ex_per_sec} ex/s '
                                f'best_acc-{best_eval * 100:.2f}')
                    TB_LOGGER.add_scalar('perf/ex_per_s',
                                         ex_per_sec, global_step)

                if global_step % opts.valid_steps == 0:
                    log = evaluation(model,
                                     dict(filter(lambda x: x[0].startswith('val'), dataloaders.items())),
                                     opts, global_step)
                    if log['val/acc'] > best_eval:
                        best_ckpt = global_step
                        best_eval = log['val/acc']
                        pbar.set_description(f'{opts.model}: {n_epoch}-{best_ckpt} best_acc-{best_eval * 100:.2f}')
                    model_saver.save(model, global_step)
            if global_step >= opts.num_train_steps:
                break
        if global_step >= opts.num_train_steps:
            break
        n_epoch += 1
        LOGGER.info(f"Step {global_step}: finished {n_epoch} epochs")

    sum(all_gather_list(opts.rank))

    best_pt = f'{opts.output_dir}/ckpt/model_step_{best_ckpt}.pt'
    model.load_state_dict(torch.load(best_pt), strict=False)
    evaluation(model,
               dict(filter(lambda x: x[0] != 'train', dataloaders.items())),
               opts, best_ckpt)


def evaluation(model, data_loaders: dict, opts, global_step):
    model.eval()
    log = {}
    for split, loader in data_loaders.items():
        LOGGER.info(f"Step {global_step}: start running "
                    f"validation on {split} split...")
        log.update(validate(opts, model, loader, split, global_step))
    TB_LOGGER.log_scaler_dict(log)
    model.train()
    return log


def optimize_answer(example_logits):
    for eid in example_logits:
        tags = []
        costs = []
        for tag, logits in example_logits[eid].items():
            tags.append(tag)
            costs.append(logits)
        cost_matrix = np.array(costs)
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        for tag, ind in zip(tags, col_ind):
            yield tag, ind


@torch.no_grad()
def validate(opts, model, val_loader, split, global_step):
    val_loss = 0
    tot_score = 0
    n_ex = 0
    val_mrr = 0
    st = time()
    example_logits = {}
    with open(f'{val_loader.dataset.db_dir}/id2eid.json', 'r') as f:
        id2eid = json.load(f)

    with tqdm(range(len(val_loader.dataset)), desc=split) as tq:
        for i, batch in enumerate(val_loader):
            qids = batch['qids']
            targets = batch['targets']
            del batch['targets']
            del batch['gather_index']
            del batch['qids']

            logits, over_logits, cond_logits = model(**batch, targets=None, compute_loss=False)
            loss = F.cross_entropy(logits, targets, reduction='sum')
            val_loss += loss.item()

            if opts.candidates == 'original':
                logits = logits
            elif opts.candidates == 'enlarged':
                logits = cond_logits
            elif opts.candidates == 'combined':
                logits = logits + cond_logits
            else:
                raise AssertionError("No such loss!")

            # scores, over_logits = model(**batch, targets=None, compute_loss=False)
            # loss = F.cross_entropy(scores, targets, reduction='sum')
            # val_loss += loss.item()
            max_prob, max_idx = logits.max(dim=-1, keepdim=False)
            tot_score += torch.eq(max_idx, targets).sum().item()
            # tot_score += (scores.max(dim=-1, keepdim=False)[1] == targets).sum().item()

            targets = torch.gather(batch['option_ids'], dim=1, index=targets.unsqueeze(1)).cpu().numpy()
            for j, (qid, target, score, over_logit) in enumerate(zip(qids, targets, logits, over_logits)):
                g = over_logit.cpu().numpy()
                top_k = np.argsort(-g)
                val_mrr += 1 / (1 + np.argwhere(top_k == target).item())

                eid = id2eid[qid]
                example_logits.setdefault(eid, {})
                example_logits[eid][qid] = score.cpu().numpy()

            n_ex += len(qids)
            tq.update(len(qids))

    out_file = f'{opts.output_dir}/results/{split}_results_{global_step}_rank{opts.rank}.csv'
    with open(out_file, 'w') as f:
        for id_, ans in optimize_answer(example_logits):
            f.write(f'{id_},{ans}\n')

    val_loss = sum(all_gather_list(val_loss))
    n_ex = sum(all_gather_list(n_ex))
    tot_time = time() - st
    val_loss /= n_ex
    val_mrr = val_mrr / n_ex

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

    val_log = {f'{split}/loss': val_loss,
               f'{split}/acc': val_acc,
               f'{split}/mrr': val_mrr,
               f'{split}/ex_per_s': n_ex / tot_time}
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {val_acc * 100:.2f}, "
                f"mrr: {val_mrr:.3f}")
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

    hvd.init()
    n_gpu = hvd.size()
    args.n_gpu = n_gpu

    args.output_dir = os.path.join(args.output_dir,
                                   f'{args.model}-{args.candidates}',
                                   os.path.basename(args.pretrained_model_name_or_path),
                                   f'competition_{args.n_gpu}_{args.num_train_steps}_{args.learning_rate}')
    if exists(args.output_dir) and os.listdir(f'{args.output_dir}/ckpt'):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output_dir))

    main(args)
