# Adapted from https://github.com/akamaster/pytorch_resnet_cifar10
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import OrderedDict

from project_config import MODEL_ROOT


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


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
    def __init__(self, model, normalise=True, squeeze=False):
        super().__init__()
        self.features = nn.Sequential(
            *([Normalise()] if normalise else []),
            model.conv1, model.bn1, nn.ReLU(),
            *list(model.layer1),
            *list(model.layer2),
            *list(model.layer3),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        classifier = nn.Conv2d(*model.linear.weight.shape[::-1], kernel_size=1)
        classifier.weight.data = model.linear.weight.data[..., None, None]
        classifier.bias.data = model.linear.bias.data
        self.classifier = nn.Sequential(classifier, nn.AdaptiveAvgPool2d((1, 1)))
        self.squeeze = squeeze

    def forward(self, in_tensor):
        out = self.classifier(self.features(in_tensor))
        if self.squeeze:
            return out.squeeze(-1).squeeze(-1)
        return out


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlockNoBN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = (self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Id(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, batch_norm=True, embedding_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(embedding_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(16)
        else:
            self.bn1 = Id()

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def get_resnet(depth, load_pre_trained=True, block=BasicBlock, batch_norm=True, normalise=True, embedding_channels=3,
               num_classes=10, **kwargs):
    _ = kwargs  # Unused.
    num_blocks = {20: [3, 3, 3], 32: [5, 5, 5], 44: [7, 7, 7], 56: [9, 9, 9]}
    file_names = {20: "resnet20-12fca82f.th", 32: "resnet32-d509ac18.th",
                  44: "resnet44-014dd654.th", 56: "resnet56-4bfd9763.th"}
    base_path = MODEL_ROOT
    r = ResNet(block, num_blocks[depth], batch_norm=batch_norm, embedding_channels=embedding_channels,
               num_classes=num_classes)
    assert not (num_classes != 10 and load_pre_trained), "Only models with 10 classes available."
    if load_pre_trained:
        checkpoint = torch.load(base_path + file_names[depth], map_location="cpu")
        ckpt = OrderedDict()
        ckpt.update({k[len("module."):]: v for k, v in checkpoint["state_dict"].items()})
        r.load_state_dict(ckpt)
    model = MyResNet(r, normalise=normalise, squeeze=True)
    return model
