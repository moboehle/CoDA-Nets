import torch
from captum.attr import IntegratedGradients, GuidedBackprop, InputXGradient, Saliency, LayerGradCam, DeepLift
from torch import nn as nn

from interpretability.explanation_methods.utils import CaptumDerivative


class IntGrad(CaptumDerivative, IntegratedGradients):

    def __init__(self, trainer, n_steps=20, internal_batch_size=1):
        CaptumDerivative.__init__(self, trainer, n_steps=n_steps, internal_batch_size=internal_batch_size)
        IntegratedGradients.__init__(self, self.trainer)


class GB(CaptumDerivative, GuidedBackprop):

    def __init__(self, trainer):
        CaptumDerivative.__init__(self, trainer)
        GuidedBackprop.__init__(self, self.trainer)


class IxG(CaptumDerivative, InputXGradient):

    def __init__(self, trainer):
        CaptumDerivative.__init__(self, trainer)
        InputXGradient.__init__(self, self.trainer)


class Grad(CaptumDerivative, Saliency):

    def __init__(self, trainer):
        CaptumDerivative.__init__(self, trainer)
        self.configs.update({"abs": False})
        Saliency.__init__(self, self.trainer)


class GradCam(CaptumDerivative, LayerGradCam):

    def __init__(self, trainer):
        CaptumDerivative.__init__(self, trainer)
        model = self.trainer
        adaptive_idx = 0
        self.configs.update({"relu_attributions": True})  # As in original GradCam paper
        found_avg_pool = False
        for adaptive_idx, mod in enumerate(trainer.model.children()):
            if isinstance(mod, nn.AdaptiveAvgPool2d):
                found_avg_pool = True
                break
        assert found_avg_pool, "This implementation assumes that final spatial dimension is reduced via AvgPool." \
                               "If you want to use it, change the model accordingly or update this code."
        layer = model.trainer.model[adaptive_idx - 1]
        LayerGradCam.__init__(self, model, layer)

    def attribute(self, img, target, **kwargs):
        return LayerGradCam.interpolate(
            CaptumDerivative.attribute(self, img, torch.tensor(target).cuda()),
            img.shape[-2:])


class DeepLIFT(CaptumDerivative, DeepLift):

    def __init__(self, trainer):
        CaptumDerivative.__init__(self, trainer)
        DeepLift.__init__(self, self.trainer)
