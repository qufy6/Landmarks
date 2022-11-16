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
LR = 1E-2
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


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda:0")
net = ResNet18().to(device)

criterion = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), LR)
TrainLoss = []
TestRate = []
print("Lenth:", len(train_loader))
for i in range(EPOCH):
    print('The', i + 1, 'iteration：')
    Sum_trn = 0.
    total_step = 0
    for step, (data, targets) in enumerate(train_loader):
        data = data.type(torch.FloatTensor).to(device)
        targets = targets.type(torch.FloatTensor).to(device)
        optimizer.zero_grad()  
        
        TrainOutput = net(data)  
        lossTRN = criterion(TrainOutput, targets) 
        lossTRN.backward()
        optimizer.step()  

        if total_step % 100 == 0:
            print(lossTRN)
            plt.clf() 
            img = data[0].cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]
            landmarks_gt = targets[0].cpu().detach().numpy()*255
            landmarks_pred = TrainOutput[0].cpu().detach().numpy()*255
            plt.scatter(landmarks_gt[:98], landmarks_gt[98:], s=5)
            plt.scatter(landmarks_pred[:98], landmarks_pred[98:], s=5)
            plt.imshow(img)
            plt.savefig('./trainimg/train%d_%d.jpg' % (i, total_step))
            plt.show()
        total_step += 1

        Sum_trn += lossTRN.cpu().detach().numpy()
        
    print('The', i + 1, 'iteration train finished')

    step_test = 0
    for step, (data, targets) in enumerate(test_loader):
        
        data = data.type(torch.FloatTensor).to(device)
        targets = targets.type(torch.FloatTensor).to(device)
        
        TestOutput = net(data)  # <class 'torch.Tensor'>,torch.Size([1, 10])
#        lossTST = criterion(TestOutput, targets) 
#        Sum_trn += lossTST.cpu().detach().numpy()
#        TestOutput = TestOutput[0].cpu().detach().numpy()
#        TestOutput = torch.tensor(TestOutput.argmax(0))
#
#        TestOutput = np.array(TestOutput)
#        np.save('Testout%d.npy' % (i+1), TestOutput)
        if step_test % 100 == 0:
            plt.clf() 
            print(lossTRN)
            img = data[0].cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]
            landmarks_gt = targets[0].cpu().detach().numpy()*255
            landmarks_pred = TrainOutput[0].cpu().detach().numpy()*255
            plt.scatter(landmarks_gt[:98], landmarks_gt[98:], s=5)
            plt.scatter(landmarks_pred[:98], landmarks_pred[98:], s=5)
            plt.imshow(img)
            plt.savefig('./testimg/test%d_%d.jpg' % (i, step_test))
            plt.show()
        step_test += 1

    print('The', i + 1, ' iteration loss = ', Sum_trn / len(train_loader))
    TrainLoss.append(Sum_trn / len(train_loader))
    

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print('TrainLoss', TrainLoss)

l = plt.plot(list(range(EPOCH)), TrainLoss, 'r--')
plt.savefig('./TrainLoss.jpg')
plt.show()
