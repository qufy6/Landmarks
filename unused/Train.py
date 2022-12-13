import Loader
import Net
import torch.nn as nn
import torch.optim as optim
import torch
import os

EPOCH = Net.EPOCH
BATCH_SIZE = Net.BATCH_SIZE
LR = Net.LR
num_of_classes = Net.num_of_classes
save_path = '/home/qufy620331060/qufy6_project/Landmarks/model.pt'

train_landmarks = Loader.LandmarksDataset(txt=Loader.root + '/' + 'list_98pt_rect_attr_train.txt')
train_loader = Loader.DataLoader(dataset=train_landmarks, batch_size=BATCH_SIZE, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = Loader.ResNet18().to(device)

criterion = nn.MSELoss(reduction='mean')
optimizer = optim.SGD(net.parameters(), LR)
print("Length:", len(train_loader))

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

        # if total_step % 100 == 0:
        #     print(lossTRN)
        #     plt.clf()
        #     img = data[0].cpu().numpy().transpose(1, 2, 0)[:, :, ::-1]
        #     landmarks_gt = targets[0].cpu().detach().numpy() * 255
        #     landmarks_pred = TrainOutput[0].cpu().detach().numpy() * 255
        #     plt.scatter(landmarks_gt[:98], landmarks_gt[98:], s=5)
        #     plt.scatter(landmarks_pred[:98], landmarks_pred[98:], s=5)
        #     plt.imshow(img)
        #     plt.savefig('./trainimg/train%d_%d.jpg' % (i, total_step))
        #     plt.show()
        # total_step += 1

        Sum_trn += lossTRN.cpu().detach().numpy()

    print('The', i + 1, 'iteration train finished')
    print('The', i + 1, ' iteration loss = ', Sum_trn / len(train_loader))

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.save(net, save_path)