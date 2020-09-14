"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

sampler for length bucketing (batch by tokens)
"""
import random
from collections import deque, Counter

from cytoolz import partition_all
from more_itertools import flatten
from torch.utils.data import Sampler


class TokenBucketSampler(Sampler):
    def __init__(self, lens, bucket_size, batch_size,
                 droplast=False, size_multiple=8):
        self._lens = lens
        self._max_tok = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._size_mul = size_multiple

    def _create_ids(self):
        return list(range(len(self._lens)))

    def _sort_fn(self, i):
        return self._lens[i]

    def __iter__(self):
        ids = self._create_ids()
        random.shuffle(ids)
        buckets = [sorted(ids[i:i + self._bucket_size],
                          key=self._sort_fn, reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]
        # fill batches until max_token (include padding)
        batches = []
        for bucket in buckets:
            max_len = 0
            batch_indices = []
            for indices in partition_all(self._size_mul, bucket):
                max_len = max(max_len, max(self._lens[i] for i in indices))
                if (max_len * (len(batch_indices) + self._size_mul)
                        > self._max_tok):
                    if not batch_indices:
                        raise ValueError(
                            "max_tokens too small / max_seq_len too long")
                    assert len(batch_indices) % self._size_mul == 0
                    batches.append(batch_indices)
                    batch_indices = list(indices)
                else:
                    batch_indices.extend(indices)
            if not self._droplast and batch_indices:
                batches.append(batch_indices)
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        raise ValueError("NOT supported. "
                         "This has some randomness across epochs")


class DistributedTokenBucketSampler(TokenBucketSampler):
    def __init__(self, num_replicas, rank, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rank = rank
        self._num_replicas = num_replicas

    def _create_ids(self):
        return super()._create_ids()[self._rank::self._num_replicas]


class ContrastiveSampler(Sampler):
    def __init__(self, num_replicas, rank, lens, ids, batch_size, reverse_index,
                 droplast=False, size_multiple=8):
        self._rank = rank
        self._num_replicas = num_replicas
        self._lens = lens
        self._ids = {v: k for k, v in enumerate(ids)}
        self._max_tok = batch_size
        self._droplast = droplast
        self._size_mul = size_multiple
        self.reverse_index = reverse_index
        self.contrastive_deque = deque(maxlen=500 * self._num_replicas)

    def _sort_fn(self, i):
        return self._lens[i]

    def _contrastive_bucket(self):
        if len(self.contrastive_deque) == 0:
            self.contrastive_deque.append(random.choices(list(self.reverse_index.keys()),
                                                         k=500 * self._num_replicas)[self._rank::self._num_replicas])

        idiom_ids = self.contrastive_deque.pop()

        bucket = []
        for idx in random.choices(idiom_ids, k=500):
            try:
                qid = random.choice(self.reverse_index[idx])
                bucket.append(self._ids[qid])
            except:
                continue
        return bucket

    def __iter__(self):
        while True:
            bucket = self._contrastive_bucket()
            max_len = 0
            batch_indices = []
            for indices in partition_all(self._size_mul, bucket):
                max_len = max(max_len, max(self._lens[i] for i in indices))
                if (max_len * (len(batch_indices) + self._size_mul)
                        > self._max_tok):
                    if not batch_indices:
                        raise ValueError(
                            "max_tokens too small / max_seq_len too long")
                    assert len(batch_indices) % self._size_mul == 0
                    break
                else:
                    batch_indices.extend(indices)
            yield batch_indices

    def __len__(self):
        raise ValueError("NOT supported. "
                         "This has some randomness across epochs")


class ContrastivePairSampler(Sampler):
    def __init__(self, num_replicas, rank, lens, ids, batch_size, reverse_index,
                 droplast=False, size_multiple=8):
        self._rank = rank
        self._num_replicas = num_replicas
        self._lens = lens
        self._ids = {v: k for k, v in enumerate(ids)}  # from example to example index
        self._max_tok = batch_size
        self._droplast = droplast
        self._size_mul = size_multiple
        self.idiom_counter = Counter()
        self.reverse_index = reverse_index  # from idiom id to example list
        self.contrastive_deque = deque(maxlen=500 * self._num_replicas)

    def _sort_fn(self, i):
        return self._lens[i]

    def _contrastive_bucket(self):
        if len(self.contrastive_deque) == 0:
            total = sum(self.idiom_counter.values()) + 1
            weights = [1 - self.idiom_counter[i] / total for i in range(len(self.reverse_index))]
            c = random.choices(range(len(self.reverse_index)),
                               weights=weights,
                               k=500 * self._num_replicas)
            q = c[self._rank::self._num_replicas]
            self.contrastive_deque.append(q)  # add sampled idiom ids

        idiom_ids = self.contrastive_deque.pop()  # pop idiom ids

        bucket = []
        for idx in random.choices(idiom_ids, k=500):
            try:
                qid_pair = random.choices(self.reverse_index[idx], k=2)  # sample two examples given one idiom id
                bucket.append([(idx, self._ids[qid]) for qid in qid_pair])  # convert example to example index
            except:
                continue
        return bucket

    def __iter__(self):
        while True:
            bucket = self._contrastive_bucket()
            max_len = 0
            batch_indices = []
            for indices in partition_all(self._size_mul, bucket):
                idiom_ids = [j for i in indices for j, _ in i]
                indices = [j for i in indices for _, j in i]
                max_len = max(max_len, max(self._lens[i] for i in indices))
                if (max_len * (len(batch_indices) + self._size_mul) * 2
                        > self._max_tok):
                    if not batch_indices:
                        raise ValueError(
                            "max_tokens too small / max_seq_len too long")
                    assert len(batch_indices) % self._size_mul == 0
                    break
                else:
                    batch_indices.extend(indices)
                    self.idiom_counter.update(idiom_ids)
            assert len(batch_indices) % 2 == 0
            yield batch_indices

    def __len__(self):
        raise ValueError("NOT supported. "
                         "This has some randomness across epochs")
