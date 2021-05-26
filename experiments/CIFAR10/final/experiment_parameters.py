import numpy as np
from torchvision import transforms

from data.data_transforms import AddInverse
from modules.losses import CombinedLosses, LogitsBCE, DynamicMatrixLoss
from modules.norms import Squashing, L2Norm
from copy import deepcopy as copy


standard_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4)])


default_config = {
    "base_lr": 2.5e-4,
    "batch_size": 8,
    "virtual_batch_size": 16,
    "num_classes": 10,
    "num_epochs": 200,
    "lr_steps": 30,
    "logit_bias": np.log(.1/.9),
    "logit_temperature": 1,
    "decay_factor": 1.5,
    "stopped": True,
    "embedding_channels": 6,
    "loss": CombinedLosses(LogitsBCE()),
    "clamp": False,
    "regul_matrix": False,
    "pre_process_img": AddInverse(),
}


def update_default(params):
    exp = copy(default_config)
    exp.update(params)
    return exp


SQ_exps = {  # Temperature ablations and evaluated model (T=1000)
    "9L-S-CoDA-SQ-{norm}".format(norm=norm): update_default({
        "kernel_sizes": [3] * 8 + [1],
        "stride": [1, 1, 2, 1, 1, 2, 1, 1, 1],
        "padding": [1] * 8 + [0],
        "out_c": [16] * 2 + [32] * 3 + [64] * 3 + [10],
        "ranks": [32] * 2 + [64] * 3 + [64] * 4,
        "augmentation_transforms": standard_aug,
        "logit_temperature": norm,
        "norm": Squashing
    }) for norm in [10, 50, 100, 500, 1000, 5000, 10000]
}

L2_exps = {  # Model with L2-non-linearity
    "9L-S-CoDA-L2-{norm}".format(norm=norm): update_default({
        "kernel_sizes": [3] * 8 + [1],
        "stride": [1, 1, 2, 1, 1, 2, 1, 1, 1],
        "padding": [1] * 8 + [0],
        "out_c": [16] * 2 + [32] * 3 + [64] * 3 + [10],
        "ranks": [32] * 2 + [64] * 3 + [64] * 4,
        "augmentation_transforms": standard_aug,
        "logit_temperature": norm,
        "stopped": False,
        "norm": L2Norm
    }) for norm in [1000]
}

rank_exps = {  # Rank ablations for parameter-accuracy impact
    "9L-S-CoDA-SQ-rank-{rank_factor}".format(rank_factor=rank_factor): update_default({
        "kernel_sizes": [3] * 8 + [1],
        "stride": [1, 1, 2, 1, 1, 2, 1, 1, 1],
        "padding": [1] * 8 + [0],
        "out_c": [16] * 2 + [32] * 3 + [64] * 3 + [10],
        "ranks": [rank_factor] * 2 + [2 * rank_factor] * 3 + [2 * rank_factor] * 4,
        "stopped": False,
        "augmentation_transforms": standard_aug,
        "logit_temperature": 1000,
        "norm": Squashing
    }) for rank_factor in [4, 8, 16, 9 * 16]
}


regul_exps = {
    "9L-S-CoDA-SQ-lambda-{}".format(_lambda): update_default({
        "kernel_sizes": [3] * 8 + [1],
        "stride": [1, 1, 2, 1, 1, 2, 1, 1, 1],
        "padding": [1] * 8 + [0],
        "out_c": [16] * 2 + [32] * 3 + [64] * 3 + [10],
        "ranks": [32] * 2 + [64] * 3 + [64] * 4,
        "augmentation_transforms": standard_aug,
        "stopped": False,
        "loss": CombinedLosses(LogitsBCE(), DynamicMatrixLoss(w=_lambda, norm="L1")),
        "logit_temperature": 64,  # This is equiv. to T=1 with average pooling..
        "norm": Squashing,
        "regul_exp": True
    }) for _lambda in [0., 0.1, .25, .5, 1.]
}

exps = dict()

exps.update(SQ_exps)
exps.update(L2_exps)
exps.update(rank_exps)
exps.update(regul_exps)


def get_exp_params(exp_name):
    if exp_name not in exps:
        raise NotImplementedError("The configuration for {} is not specified yet.".format(exp_name))
    return exps[exp_name]
