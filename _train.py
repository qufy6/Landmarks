import numpy as np
from PIL import Image, ImageFilter
import os
import sys
import Loader
import _loader
import _tools
import logging
import torch.optim as optim
import torchvision.transforms as transforms
from torch import nn
import Net
import _root
sys.path.insert(0, '..')  # 将当前目录的父目录插入 Python 的搜索路径中。这样做的目的是为了让 Python 在导入模块时能够在当前目录的父目录中查找模块。
import importlib
import torch.utils.data

super_root = _root.super_root
experiment_name = 'resnet18_baseline'
config_path = '.experiments.{}'.format(experiment_name)

my_config = importlib.import_module(config_path, package='Landmarks')
Config = getattr(my_config, 'Config')  ##==》 Config = my_config.Config
cfg = Config()
os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)

if not os.path.exists(os.path.join(super_root, 'snapshots')):
    os.mkdir(os.path.join(super_root, 'snapshots'))
save_dir = os.path.join(super_root, 'snapshots', experiment_name)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

if not os.path.exists(os.path.join(super_root, 'logs')):
    os.mkdir(os.path.join(super_root, 'logs'))
log_dir = os.path.join(super_root, 'logs', experiment_name)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

logging.basicConfig(filename=os.path.join(log_dir, 'train.log'), level=logging.INFO)
if 1 > 0:  # print and log
    print('###########################################')
    print('net_stride:', cfg.net_stride)
    print('batch_size:', cfg.batch_size)
    print('init_lr:', cfg.init_lr)
    print('num_epochs:', cfg.num_epochs)
    print('decay_steps:', cfg.decay_steps)
    print('input_size:', cfg.input_size)
    print('backbone:', cfg.backbone)
    print('pretrained:', cfg.pretrained)
    print('criterion_cls:', cfg.criterion_cls)
    print('criterion_reg:', cfg.criterion_reg)
    print('cls_loss_weight:', cfg.cls_loss_weight)
    print('reg_loss_weight:', cfg.reg_loss_weight)
    print('num_lms:', cfg.num_lms)
    print('save_interval:', cfg.save_interval)
    print('num_nb:', cfg.num_nb)
    print('use_gpu:', cfg.use_gpu)
    print('gpu_id:', cfg.gpu_id)
    print('###########################################')

    logging.info('###########################################')
    logging.info('net_stride: {}'.format(cfg.net_stride))
    logging.info('batch_size: {}'.format(cfg.batch_size))
    logging.info('init_lr: {}'.format(cfg.init_lr))
    logging.info('num_epochs: {}'.format(cfg.num_epochs))
    logging.info('decay_steps: {}'.format(cfg.decay_steps))
    logging.info('input_size: {}'.format(cfg.input_size))
    logging.info('backbone: {}'.format(cfg.backbone))
    logging.info('pretrained: {}'.format(cfg.pretrained))
    logging.info('criterion_cls: {}'.format(cfg.criterion_cls))
    logging.info('criterion_reg: {}'.format(cfg.criterion_reg))
    logging.info('cls_loss_weight: {}'.format(cfg.cls_loss_weight))
    logging.info('reg_loss_weight: {}'.format(cfg.reg_loss_weight))
    logging.info('num_lms: {}'.format(cfg.num_lms))
    logging.info('save_interval: {}'.format(cfg.save_interval))
    logging.info('num_nb: {}'.format(cfg.num_nb))
    logging.info('use_gpu: {}'.format(cfg.use_gpu))
    logging.info('gpu_id: {}'.format(cfg.gpu_id))
    logging.info('###########################################')

# train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=8,
#                                            pin_memory=True, drop_last=True)
#
# train_model(cfg.det_head, net, train_loader, criterion_cls, criterion_reg, cfg.cls_loss_weight, cfg.reg_loss_weight,
#             cfg.num_nb, optimizer, cfg.num_epochs, scheduler, save_dir, cfg.save_interval, device)

net = Net.ResNet18()
if cfg.use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
net = net.to(device)

criterion_cls = None
if cfg.criterion_cls == 'l2':
    criterion_cls = nn.MSELoss()
elif cfg.criterion_cls == 'l1':
    criterion_cls = nn.L1Loss()
else:
    print('No such cls criterion:', cfg.criterion_cls)

criterion_reg = None
if cfg.criterion_reg == 'l1':
    criterion_reg = nn.L1Loss()
elif cfg.criterion_reg == 'l2':
    criterion_reg = nn.MSELoss()
else:
    print('No such reg criterion:', cfg.criterion_reg)

points_flip = None
points_flip = [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7,
               6, 5, 4, 3, 2, 1, 0, 46, 45, 44, 43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41, 40, 39, 38, 51, 52,
               53, 54, 59, 58, 57, 56, 55, 72, 71, 70, 69, 68, 75, 74, 73, 64, 63, 62, 61, 60, 67, 66, 65, 82, 81,
               80, 79, 78, 77, 76, 87, 86, 85, 84, 83, 92, 91, 90, 89, 88, 95, 94, 93, 97, 96]
assert len(points_flip) == 98

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

if cfg.pretrained:
    optimizer = optim.Adam(net.parameters(), lr=cfg.init_lr)
else:
    optimizer = optim.Adam(net.parameters(), lr=cfg.init_lr, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.decay_steps, gamma=0.1)

labels = _tools.get_label('train.txt') # 包含两个元素，分别表示图像文件名和目标信息（196）
# print(np.array(labels).shape)
# img = Image.open(os.path.join(os.path.join(_root.gen_data_wflw_root, 'WFLW', 'images_train'), labels[0][0][0])).convert('RGB')
# print(np.array(img).shape)
train_data = _loader.LandmarksDataset(os.path.join(_root.gen_data_wflw_root, 'WFLW', 'images_train'),
                                      labels,
                                      cfg.input_size,
                                      cfg.num_lms,
                                      cfg.net_stride,
                                      points_flip,
                                      transforms.Compose([
                                          transforms.RandomGrayscale(0.2),
                                          transforms.ToTensor(),
                                          normalize]))
train_loader = Loader.DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=8,
                                           pin_memory=True, drop_last=True)
print(len(train_loader))

_tools.train_model(net, train_loader, criterion_reg, optimizer, cfg.num_epochs, scheduler, save_dir, cfg.save_interval, device)


