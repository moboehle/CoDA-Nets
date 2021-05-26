import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Im2Col(torch.autograd.Function):
    """
    This class does the same as torch.nn.functional.unfold(), but also implements the backward function.
    This allows for regularising the gradient by taking the second derivative.
    """
    @staticmethod
    def forward(ctx, x, kernel_size, padding, stride):
        ctx.shape, ctx.kernel_size, ctx.padding, ctx.stride = (x.shape[2:], kernel_size, padding, stride)
        return F.unfold(x, kernel_size=kernel_size, padding=padding, stride=stride)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.enable_grad():
            shape, ks, padding, stride = ctx.shape, ctx.kernel_size, ctx.padding, ctx.stride
            return (
                F.fold(grad_output, shape, kernel_size=ks, padding=padding, stride=stride),
                None,
                None,
                None
            )


class Col2Im(torch.autograd.Function):
    """
    This class does the same as torch.nn.functional.fold(), but also implements the backward function.
    This allows for regularising the gradient by taking the second derivative.
    """
    @staticmethod
    def forward(ctx, x, shape, kernel_size, padding, stride):
        ctx.kernel_size, ctx.padding, ctx.stride = (kernel_size, padding, stride)
        return F.fold(x, shape, kernel_size=kernel_size, padding=padding, stride=stride)

    @staticmethod
    def backward(ctx, grad_output):
        with torch.enable_grad():
            ks, padding, stride = ctx.kernel_size, ctx.padding, ctx.stride
            return (
                F.unfold(grad_output, kernel_size=ks, padding=padding, stride=stride),
                None,
                None,
                None,
                None
            )


def im2col(x, kernel_size, padding, stride):
    """
    Applies torch.nn.functional.unfold() to x with the given kernel size, padding and stride.
    Args:
        x: Input tensor to unfold.
        kernel_size: Kernel size of unfolding operation.
        padding: Padding used for the unfolding operation.
        stride: Stride used for the unfolding operation.

    Returns: The unfolded tensor as if F.unfold(x, kernel_size=kernel_size, padding=padding, stride=stride)
        had been used.

    """
    return Im2Col.apply(x, kernel_size, padding, stride)


def col2im(x, shape, kernel_size, padding, stride):
    """
    Applies torch.nn.functional.fold() to x with the given kernel size, shape, padding and stride.
    Args:
        x: Input tensor to fold.
        shape: Shape of the output after folding.
        kernel_size: Kernel size of folding operation.
        padding: Padding used for the folding operation.
        stride: Stride used for the folding operation.

    Returns: The unfolded tensor as if F.fold(x, kernel_size=kernel_size, padding=padding, stride=stride)
        had been used.

    """
    return Col2Im.apply(x, shape, kernel_size, padding, stride)


class AdaptiveSumPool2d(nn.AdaptiveAvgPool2d):

    def __init__(self, output_size):
        """
        Same as AdaptiveAvgPool2d, only that the normalisation is undone.
        Args:
            output_size: Adaptive size of the output.
        """
        super().__init__(output_size)
        self.shape = None

    def forward(self, in_tensor):
        self.shape = in_tensor.shape[-2:]
        return super().forward(in_tensor) * np.prod(self.shape)


class MyAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):

    def __init__(self, output_size):
        """
        Same as AdaptiveAvgPool2d, with saving the shape for matrix upscaling.
        Args:
            output_size: Adaptive size of the output.
        """
        super().__init__(output_size)
        self.shape = None

    def forward(self, in_tensor):
        self.shape = in_tensor.shape[-2:]
        return super().forward(in_tensor)


class TrainerAsModule(nn.Module):

    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer
        self.eval()

    def forward(self, in_tensor):
        return self.trainer(in_tensor)


class FinalLayer(nn.Module):

    def __init__(self, norm=1, bias=-5):
        """
        Used to add a bias and a temperature scaling to the final output of a particular model.
        Args:
            norm: inverse temperature, i.e., T^{-1}
            bias: constant shift in logit space for all classes.
        """
        super().__init__()
        assert norm != 0, "Norm 0 means average pooling in the last layer of the old trainer. " \
                          "Please add size.prod() of final layer as img_size_norm to exp_params."
        self.norm = norm
        self.bias = bias

    def forward(self, in_tensor):
        out = (in_tensor.view(*in_tensor.shape[:2])/self.norm + self.bias)
        return out
