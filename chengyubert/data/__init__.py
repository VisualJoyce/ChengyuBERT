from torch.utils.data import DataLoader

from chengyubert.data.data import ChengyuDataset, ChengyuEvalDataset, chengyu_collate, chengyu_eval_collate, open_lmdb
from chengyubert.data.loader import PrefetchLoader
from chengyubert.data.sampler import DistributedTokenBucketSampler
from chengyubert.utils.const import BUCKET_SIZE


def create_dataloader(txt_path, batch_size, is_train,
                      dset_cls, collate_fn, opts):
    dset = dset_cls(txt_path, opts.max_txt_len if is_train else -1, opts)
    sampler = DistributedTokenBucketSampler(
        opts.size, opts.rank, dset.lens,
        bucket_size=BUCKET_SIZE, batch_size=batch_size, droplast=is_train)
    loader = DataLoader(dset, batch_sampler=sampler,
                        num_workers=opts.n_workers, pin_memory=opts.pin_mem,
                        collate_fn=collate_fn)
    return PrefetchLoader(loader)


def create_dataloaders(LOGGER, DatasetCls, EvalDatasetCls, collate_fn, eval_collate_fn, opts, pretrain_collate_fn=None):
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
        dataloaders[split] = create_dataloader(txt_db, batch_size, 'train' == split, dataset_cls, c_fn, opts)
    return splits, dataloaders
