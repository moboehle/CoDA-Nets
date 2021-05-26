import numpy as np
from torchvision import transforms

from data.augmentations.rand_augment import RandAugment
from data.data_transforms import AddInverse
from modules.losses import CombinedLosses, LogitsBCE
from modules.norms import Squashing
from copy import deepcopy as copy
from training.utils import TopKAcc


class WarmUpLR:

    def __init__(self, trainer):
        self.trainer = trainer
        self.sched = trainer.options["sched_opts"]["sched"]

    def __call__(self, epoch):
        return self.sched[epoch]

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return str(self)


default_config = {
    "base_lr": 2.5e-4,
    "batch_size": 8,
    "logit_temperature": 1,
    "virtual_batch_size": 64,
    "num_classes": 100,
    "num_epochs": 60,
    "lr_steps": 30,
    "decay_factor": 2,
    "embedding_channels": 6,
    "logit_bias": np.log(.1/(1-.1)),
    "augmentation_transforms": transforms.Compose([transforms.RandomCrop(240),
                                                   transforms.RandomHorizontalFlip(),
                                                   RandAugment(n=2, m=10)]),
    "pre_data_transforms": transforms.Resize(256),
    "test_time_transforms": transforms.CenterCrop(240),
    "loss": CombinedLosses(LogitsBCE()),
    "pre_process_img": AddInverse(),
    "data_params": {"class_idcs": list(range(100))},
    "eval_every": 2,
    "eval_batch_f": TopKAcc((1, 5)),
 }


def update_default(params):
    exp = copy(default_config)
    exp.update(params)
    return exp


L_CoDA = {
    "9L-L-CoDA-SQ-{norm}".format(norm=norm): update_default({  # As presented in the paper
        "kernel_sizes": [7] + 8 * [3],
        "stride": [3, 1, 1, 2, 1, 1, 2, 1, 1],
        "padding": [3] + [1] * 8,
        "out_c": [16] + 2 * [32] + 5 * [64] + [100],
        "ranks": [64] * 3 + [128] * 3 + [256] * 3,
        "batch_size": 4,
        "sched_opts": {"sched": list(np.linspace(2.5e-4, 1e-3, 15)) + [1e-3 / (2 ** (i//20)) for i in range(15, 61)]},
        "schedule": WarmUpLR,
        "norm": Squashing,
        "logit_temperature": norm,
    }) for norm in [100000]
}

exps = dict()

exps.update(L_CoDA)


def get_exp_params(exp_name):
    if exp_name not in exps:
        raise NotImplementedError("The configuration for {} is not specified yet.".format(exp_name))
    return exps[exp_name]
