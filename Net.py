
import torch.nn as nn
import torch.nn.functional as F

EPOCH = 30
BATCH_SIZE = 15
LR = 1E-2
num_of_classes = 196


class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, ResBlock=ResBlock, num_classes=num_of_classes):
        super(ResNet18, self).__init__()

        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # [15, 3, 256, 256]
        out = self.conv1(x)  # [15, 64, 256, 256]
        out = self.layer1(out)  # [15, 64, 256, 256]
        out = self.layer2(out)  # [15, 128, 128, 128]
        out = self.layer3(out)  # [15, 256, 64, 64]
        out = self.layer4(out)  # [15, 512, 32, 32]
        out = F.avg_pool2d(out, 32)  # [15, 512, 1, 1]
        out = out.view(out.size(0), -1)  # [15, 512]
        out = self.fc(out)  # [15, 196]
        return out
