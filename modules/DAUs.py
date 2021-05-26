from torch import nn

from modules.utils import im2col, col2im


class DAUWeights(nn.Module):

    def __init__(self, in_channels, num_daus, rank, norm_func,
                 kernel_size=3, stride=1, padding=None, groups=1, bias=True):

        """
        This module calculates the weights per position for a CoDA-ConvLayer.
        It consists mainly of a convolutional layer, which predicts the weightings that are to be applied to the input.

        The parameters kernel_size, stride, padding, groups, and bias are used for the nn.Conv2d.

        Args:
            in_channels: Number of input channels to convolution.
            num_daus: Number of output channels / DAUs in this layer.
            rank (int): The size of the rank to use for the DAUs.
            norm_func (nn.Module): Function to calculate the normalisation of the weights.
            kernel_size: Kernel size for the convolutional layer.
            stride: Stride for the convolutional layer.
            padding: Padding for the convolutional layer. If None, (KS-1)//2 is set for the padding.
            groups: Groups for the convolutional layer.
            bias: Bias for the convolutional layer.
        """
        super().__init__()
        if isinstance(rank, bool):
            assert not rank, "Bottleneck should be either False or the size of the bottleneck, not 'True'."

        self.shape = None
        self.out_channels = num_daus
        self.kernel_size = kernel_size
        if padding is None:
            padding = (kernel_size - 1)//2
        self.padding = padding
        self.stride = stride
        self.groups = groups
        self.weighting_f = nn.Sequential(
            nn.Conv2d(in_channels, rank, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=1, bias=False),
            nn.Conv2d(rank, num_daus * in_channels * kernel_size ** 2,
                      kernel_size=1, stride=1, padding=0, groups=groups, bias=bias)
        )
        self.norm_func = norm_func

    @staticmethod
    def reshape_out(in_tensor, out_c):
        """
        Reshapes the output into the format (B, input_patch_dim, out_c, H, W).
        Args:
            in_tensor: Tensor to be reshaped.
            out_c: The size of the out_c dimension. For the unfolded input, this will be one.
                When reshaping the weight matrix, this will be the number of output channels.

        Returns: The reshaped input tensor.

        """
        return in_tensor.view(in_tensor.shape[0],  # Batch size
                              -1,  # Input size to each convolutional kernel
                              out_c,  # Number of output channels
                              *in_tensor.shape[-2:])  # New height and width

    def forward(self, in_tensor):
        self.shape = in_tensor.shape[-2:]  # Remember input spatial dimensions for folding operation.
        w = self.weighting_f(in_tensor)
        w = self.reshape_out(w, out_c=self.out_channels)
        norm = self.norm_func(w)
        return w, norm

    def unfold(self, in_tensor, shape):
        """
        This method is used to extract patches which are then weighted by the weighting_f.
        Args:
            in_tensor: Input tensor from which to extract patches.
            shape: shape of the weightings to easily extract correct shape.
        """

        return im2col(in_tensor, kernel_size=self.kernel_size,
                      padding=self.padding, stride=self.stride).view(
            in_tensor.shape[0],  # Batch size
            -1,  # Input size to each convolutional kernel
            1,  # Number of output channels
            *shape[-2:]  # New height and width
        )

    def fold(self, in_tensor):
        """
        This method is used in the backward pass to accumulate the weightings of the inputs.
        """
        return col2im(in_tensor, shape=self.shape, kernel_size=self.kernel_size,
                      padding=self.padding, stride=self.stride)
