import numpy as np
import torch


def to_numpy(tensor):
    """
    Converting tensor to numpy.
    Args:
        tensor: torch.Tensor

    Returns:
        Tensor converted to numpy.

    """
    if not isinstance(tensor, torch.Tensor):
        return tensor
    return tensor.detach().cpu().numpy()


def to_numpy_img(tensor):
    """
    Converting tensor to numpy image. Expects a tensor of at most 3 dimensions in the format (C, H, W),
    which is converted to a numpy array with (H, W, C) or (H, W) if C=1.
    Args:
        tensor: torch.Tensor

    Returns:
        Tensor converted to numpy.

    """
    return to_numpy(tensor.permute(1, 2, 0)).squeeze()


def make_exp_name(dataset, base_net, exp_name):
    return "-".join([dataset, base_net, exp_name])


def make_exp_name_from_save_path(save_path):
    dataset, base_net, exp_name = save_path.split("/")[-3:]
    return make_exp_name(dataset, base_net, exp_name)


class Str2List:

    def __init__(self, dtype=int):
        self.dtype = dtype

    def __call__(self, instr):
        return str_to_list(instr, self.dtype)


def str_to_list(s, _type=int):
    """
    Parses a string representation of a list of integers ('[1,2,3]') to a list of integers.
    Used for argument parsers.
    """
    s = s.replace(" ", "")
    assert s.startswith("[") and s.endswith("]"), s
    return np.array(s[1:-1].split(","), dtype=_type).tolist()


def str_to_bool(s):
    """
    Parses a string representation of a list of float values ('[0.1,2,3]') to a list of floats.
    Used for argument parsers.
    """
    return s.lower() in ["yes", "true", "1"]
