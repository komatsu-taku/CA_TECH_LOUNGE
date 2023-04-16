from typing import Callable, List, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes: int, out_planes: int, stride: int, padding: int) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 
                        kernel_size=3, stride=stride, padding=padding, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int=1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes,
                        kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                downsample: Optional[nn.Module] = None, norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(BasicBlock, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        # 1番目のconv層
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        
        # 2番目のconv層
        self.conv2 = conv3x3(planes, planes, stride=1)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        # 1層目
        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        # 2層目
        out = self.conv2(out)
        out = F.relu(self.bn2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.clone() + identity
        out = F.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, 
                downsample: Optional[nn.Module] = None, norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        # 1番目のConv層(1x1)
        self.conv1 = conv1x1(inplanes, planes, stride=1)
        self.bn1 = norm_layer(planes)

        # 2番目のConv層(3x3)
        # TODO : stride可変にする理由
        self.conv2 = conv3x3(planes, planes, stride=stride, padding=1)
        self.bn2 = norm_layer(planes)

        # 3番目のConv層(1x1)
        self.conv3 = conv1x1(planes, planes*self.expansion, stride=1,)
        self.bn3 = norm_layer(planes * self.expansion)

        self.downsample = downsample # TODO : ダウンサンプルが必要な時は指定

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1層目
        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        # 2層目
        out = self.conv2(out)
        out = F.relu(self.bn2(out))

        # 3層目
        out = self.conv3(out)
        out = F.relu(self.bn3(out))

        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        
        out = out.clone() + identity
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, Bottleneck]], layers: List[int], num_classes: int,
                norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64

        # conv1 : 従来通りの定義
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # conv2_x : サイズ不変のため、stride=1
        self.conv2_x = self._make_layer(block, 64, layers[0])

        # conv3_x : サイズ変更ありのため、stride=2
        self.conv3_x = self._make_layer(block, 128, layers[1], stride=2)
        self.conv4_x = self._make_layer(block, 256, layers[2], stride=2)
        self.conv5_x = self._make_layer(block, 512, layers[3], stride=2)

        # classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]],
                    planes: int, blocks: int, stride: int = 1):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes*block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        
        layers = []
        # 1block目を追加 : size, channel変更あり
        layers.append(
            block(self.inplanes, planes, stride, downsample, norm_layer)
        )
        # chennel数変更
        self.inplanes  = planes * block.expansion

        # 残りのblockの追加 : size, chennelの調整なし
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, stride=1, norm_layer=norm_layer)
            )
        
        return nn.Sequential(*layers)


def resnet18(num_classes: int=10, norm_layer: Optional[Callable[..., nn.Module]] = None) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, norm_layer)

def resnet34(num_classes: int=10, norm_layer: Optional[Callable[..., nn.Module]] = None) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, norm_layer)

def resnet50(num_classes: int=10, norm_layer: Optional[Callable[..., nn.Module]] = None) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, norm_layer)

def resnet101(num_classes: int=10, norm_layer: Optional[Callable[..., nn.Module]] = None) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, norm_layer)

def resnet152(num_classes: int=10, norm_layer: Optional[Callable[..., nn.Module]] = None) -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, norm_layer)