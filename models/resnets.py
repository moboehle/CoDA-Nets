import torch
from torch import nn


class Normalise(nn.Module):

    def __init__(self):
        super().__init__()
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406])[None, :, None, None], requires_grad=False)
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225])[None, :, None, None], requires_grad=False)

    def forward(self, in_tensor):
        return (in_tensor-self.mean)/self.std


class MyResNet(nn.Module):
    """
    Just a wrapper class to translate the pre-trained resnet models to a format that works with the
    evaluation schemes and so on.. mainly it introduces an adaptive average pooling and adds the normalisation
    of the input to the model instead of using it as pre-processing, which allows to use the same data loader
    for every model.
    """
    def __init__(self, model, num_classes, normalise=True):
        super().__init__()
        feature_list = [
            *([Normalise()] if normalise else []),
            model.conv1, model.bn1, model.relu, model.maxpool,
            *list(model.layer1),
            *list(model.layer2),
            *list(model.layer3),
            *list(model.layer4)
        ]

        classifier = nn.Conv2d(model.fc.weight.shape[1], num_classes, kernel_size=1)
        classifier.weight.data = model.fc.weight.data[..., None, None][:num_classes]
        classifier.bias.data = model.fc.bias.data[:num_classes]
        feature_list += [classifier]
        feature_list += [nn.AdaptiveAvgPool2d((1, 1))]

        self.model = nn.Sequential(*feature_list)

    def forward(self, in_tensor):
        out = self.model(in_tensor)
        return out.squeeze(-2).squeeze(-1)
