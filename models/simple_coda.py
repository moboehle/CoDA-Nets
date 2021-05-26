from torch import nn

from modules.coda_conv import CoDAConv2d
from modules.utils import AdaptiveSumPool2d, FinalLayer
from modules.DAUs import DAUWeights


def make_coda_net(in_channels, out_chs, ranks, kernel_sizes, strides, paddings, norm_function,
                  logit_bias, logit_temperature):
    """
    Simple CoDA-Net design. This method just concatenates len(out_cs) CoDA-Conv2d layers with the specified parameters
    and adds a AdaptiveSumPool2d layer as well as a final logit bias and temperature scaling as described in the paper.
    Args:
        in_channels: Number of channels of the input to the network. Typically 3 color channels or, for CoDA-Networks,
                    6 channels (r, g, b, 1-r, 1-g, 1-b).
        out_chs: Number of DAUs per CoDA layer, i.e., the number of output channels of a CoDA-Conv2d.
        ranks: Rank of the DAUs per layer.
        kernel_sizes: Kernel sizes of the DAUs per layer.
        strides: Striding of the DAUs per layer.
        paddings: Padding of the input activation map per layer, typically (ks-1)//2
        norm_function: Function to compute the normalisation of the dynamically computed weight vectors.
            E.g., the squashing function or simple L2 rescaling.
        logit_bias: single logit bias for all classes. The more negative, the more positive evidence is emphasised
            over negative evidence.
        logit_temperature: Logit temperature as described in the paper. By down-scaling the network output with a
            high temperature, the alignment in the DAUs can be increased.

    Returns: nn.Sequential of len(out_chs) CoDAConv2d Layers with additional adaptive sum-pooling as well as
        final bias and temperature layer.

    """

    network_list = []
    emb = in_channels

    # Simple concatenation of CoDAConv2d Layers
    for i in range(len(out_chs)):

        network_list += [CoDAConv2d(DAUWeights(emb, out_chs[i], ranks[i], norm_func=norm_function(),
                                               kernel_size=kernel_sizes[i], stride=strides[i],
                                               padding=paddings[i]))]

        emb = out_chs[i]

    # Final layers of the CoDA-Nets
    network_list += [
        AdaptiveSumPool2d((1, 1)),
        FinalLayer(bias=logit_bias, norm=logit_temperature)
    ]

    return nn.Sequential(*network_list)
