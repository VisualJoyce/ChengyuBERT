"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Misc utilities
"""
import json
import os
import random
import sys

import numpy as np
import torch

from .logger import LOGGER


class NoOp(object):
    """ useful for distributed training No-Ops """

    def __getattr__(self, name):
        return self.noop

    def noop(self, *args, **kwargs):
        return


def parse_with_config(parser):
    """
    Parse from config files < command lines < system env
    """
    args = parser.parse_args()
    if args.config is not None:
        config_args = json.load(open(args.config))
        override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                         if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
            if os.getenv(k.upper()):
                LOGGER.info(f"Replaced {k} from system environment {k.upper()}.")
                new_v = os.getenv(k.upper())
                if isinstance(v, int):
                    new_v = int(new_v)
                if isinstance(v, float):
                    new_v = float(new_v)
                if isinstance(v, bool):
                    new_v = bool(new_v)
                setattr(args, k, new_v)

    del args.config
    args.model_config = os.path.join(args.pretrained_model_name_or_path, 'config.json')
    return args


VE_ENT2IDX = {
    'contradiction': 0,
    'entailment': 1,
    'neutral': 2
}

VE_IDX2ENT = {
    0: 'contradiction',
    1: 'entailment',
    2: 'neutral'
}


class Struct(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)


def set_dropout(model, drop_p):
    for name, module in model.named_modules():
        # we might want to tune dropout for smaller dataset
        if isinstance(module, torch.nn.Dropout):
            if module.p != drop_p:
                module.p = drop_p
                LOGGER.info(f'{name} set to {drop_p}')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_parent_dir(cur_dir):
    return os.path.abspath(os.path.join(cur_dir, os.path.pardir))
