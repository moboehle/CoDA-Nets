import torch

from experiment_utils import get_arguments
from training.training_utils import start_training
from models.simple_coda import make_coda_net


def get_model(exp_params):

    embedding_channels = exp_params["embedding_channels"]
    bn_ch = exp_params["ranks"]
    out_c = exp_params["out_c"]
    ks = exp_params["kernel_sizes"]
    stride = exp_params["stride"]
    padding = exp_params["padding"]
    norm = exp_params["norm"]
    logit_bias = exp_params["logit_bias"]
    final_norm = exp_params["logit_temperature"]
    emb = embedding_channels

    return make_coda_net(emb, out_c, bn_ch, ks, stride, padding, norm, logit_bias, final_norm)


def get_optimizer(model, base_lr):
    opt = torch.optim.Adam(model.parameters(), lr=base_lr)
    opt.base_lr = base_lr
    return opt


if __name__ == "__main__":
    cmd_args = get_arguments()
    start_training(cmd_args, get_model, get_optimizer)
