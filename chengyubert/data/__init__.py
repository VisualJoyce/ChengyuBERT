from torch.utils.data import DataLoader

from chengyubert.data.data import ChengyuDataset, ChengyuEvalDataset, chengyu_collate, chengyu_eval_collate, open_lmdb
from chengyubert.data.loader import PrefetchLoader
from chengyubert.data.sampler import DistributedTokenBucketSampler, ContrastiveSampler, ContrastivePairSampler
from chengyubert.utils.const import BUCKET_SIZE


def create_dataloader(txt_path, batch_size, is_train,
                      dset_cls, collate_fn, opts):
    dset = dset_cls(txt_path, opts.max_txt_len, opts)
    sampler = DistributedTokenBucketSampler(
        opts.size, opts.rank, dset.lens,
        bucket_size=BUCKET_SIZE, batch_size=batch_size, droplast=is_train)
    loader = DataLoader(dset, batch_sampler=sampler,
                        num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                        collate_fn=collate_fn)
    return PrefetchLoader(loader)


def create_contrastive_dataloader(txt_path, batch_size, is_train,
                                  dset_cls, collate_fn, opts):
    dset = dset_cls(txt_path, opts.max_txt_len, opts)
    sampler = ContrastiveSampler(opts.size, opts.rank, dset.lens, dset.ids, batch_size, dset.reverse_index,
                                 droplast=is_train)
    loader = DataLoader(dset, batch_sampler=sampler,
                        num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                        collate_fn=collate_fn)
    return PrefetchLoader(loader)


def create_contrastive_pair_dataloader(txt_path, batch_size, is_train,
                                       dset_cls, collate_fn, opts):
    dset = dset_cls(txt_path, opts.max_txt_len, opts)
    sampler = ContrastivePairSampler(opts.size, opts.rank, dset.lens, dset.ids, batch_size, dset.reverse_index,
                                     droplast=is_train)
    loader = DataLoader(dset, batch_sampler=sampler,
                        num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                        collate_fn=collate_fn)
    return PrefetchLoader(loader)


def create_dataloaders(LOGGER, DatasetCls, EvalDatasetCls, collate_fn, eval_collate_fn, opts,
                       pretrain_collate_fn=None, create_dataloader_fn_dict=None, splits=None):
    if create_dataloader_fn_dict is None:
        create_dataloader_fn_dict = {}

    if splits is None:
        splits = []
        for k in dir(opts):
            if k.endswith('_txt_db'):
                splits.append(k.replace('_txt_db', ''))

    print(splits)

    dataloaders = {}
    for split in splits:
        txt_db = getattr(opts, f'{split}_txt_db')
        LOGGER.info(f"Loading {split} Dataset {txt_db}")
        batch_size = getattr(opts, f'{split}_batch_size') if split in ['pretrain', 'train'] else opts.val_batch_size
        dataset_cls = DatasetCls if split in ['pretrain', 'train'] else EvalDatasetCls
        if split == 'pretrain':
            c_fn = pretrain_collate_fn or collate_fn
        else:
            c_fn = collate_fn if split == 'train' else eval_collate_fn
        create_dataloader_fn = create_dataloader_fn_dict.get(split, create_dataloader)
        dataloaders[split] = create_dataloader_fn(txt_db, batch_size, 'train' == split, dataset_cls, c_fn, opts)
    return splits, dataloaders
