import Loader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import gc



root = r"/home/qufy620331060/qufy6_project/Landmarks/Dataset/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test"
EPOCH = 30
BATCH_SIZE = 15
LR = 1E-3
num_of_classes = 196

train_landmarks = Loader.LandmarksDataset(txt=root + '/' + 'list_98pt_rect_attr_train.txt')
test_landmarks = Loader.LandmarksDataset(txt=root + '/' + 'list_98pt_rect_attr_test.txt')

# train_landmarks.all_faces_scale()
# for i in range(10):
#     img, landmarks = train_landmarks[i]
#     Loader.vis_landmarks(img, landmarks)
train_loader = Loader.DataLoader(dataset=train_landmarks, batch_size=BATCH_SIZE, shuffle=False)
test_loader = Loader.DataLoader(dataset=test_landmarks, batch_size=BATCH_SIZE, shuffle=False)


# for img, gt_lms in train_loader:
#     print(img.shape, gt_lms.shape)
#     print(img.max(), gt_lms.max())
#     break

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
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


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda:0")
net = ResNet18().to(device)

criterion = nn.MSELoss()
optimizer = optim.SGD(ResNet18().parameters(), LR)
TrainLoss = []
TestRate = []

for i in range(EPOCH):
    print('第', i + 1, '轮迭代：')
    Sum = 0.
    for step, (data, targets) in enumerate(train_loader):
        data = data.type(torch.FloatTensor).to(device)
        targets = targets.type(torch.FloatTensor).to(device)
        optimizer.zero_grad()  # 把梯度清零
        TrainOutput = net(data)  # 进行一次前向传播
        TrainOutput = TrainOutput.requires_grad_(True)
        lossTRN = criterion(TrainOutput, targets)  # 计算误差
        lossTRN.requires_grad_(True)
        Sum += lossTRN.cpu().data.detach().numpy()

        lossTRN.backward()  # 后向传播

        optimizer.step()  # 进行一次参数更新
    print('第', i + 1, '轮迭代：train finished')
    correct = 0
    for step, (data, targets) in enumerate(test_loader):
        
        data = data.type(torch.FloatTensor).to(device)
        targets = targets.type(torch.FloatTensor).to(device)
        acc = 0.
        TestOutput = net(data)  # <class 'torch.Tensor'>,torch.Size([1, 10])

        TestOutput = TestOutput[0].cpu().detach().numpy()
        TestOutput = torch.tensor(TestOutput.argmax(0))

        TestOutput = np.array(TestOutput)
        np.save('Testout%d.npy'%(i+1), TestOutput)

    print('第', i + 1, '轮迭代loss = ', Sum / len(train_loader))
    TrainLoss.append(Sum / len(train_loader))
    

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print('TrainLoss', TrainLoss)

l = plt.plot(list(range(EPOCH)), TrainLoss, 'r--')
plt.show()
