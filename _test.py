import time

import cv2
import numpy as np
from PIL import Image, ImageFilter
import os
import sys
from torch.utils.data import DataLoader
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
from scipy.integrate import simps


def compute_nme(lms_pred, lms_gt, norm):
    lms_pred = lms_pred.reshape((-1, 2))
    lms_gt = lms_gt.reshape((-1, 2))
    nme = np.mean(np.linalg.norm(lms_pred - lms_gt, axis=1)) / norm
    return nme


def compute_fr_and_auc(nmes, thres=0.1, step=0.0001):
    num_data = len(nmes)
    xs = np.arange(0, thres + step, step)
    ys = np.array([np.count_nonzero(nmes <= x) for x in xs]) / float(num_data)
    fr = 1.0 - ys[-1]
    auc = simps(ys, x=xs) / thres
    return fr, auc


def forward_net(net, inputs):
    net.eval()
    with torch.no_grad():
        outputs = net(inputs)
        outputs_x = outputs[:, 0:98]
        outputs_y = outputs[:, 98:196]
    return outputs_x, outputs_y


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

net = Net.ResNet18()
if cfg.use_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
net = net.to(device)

weight_file = os.path.join(save_dir, 'epoch%d.pth' % (cfg.num_epochs - 1))
state_dict = torch.load(weight_file)
net.load_state_dict(state_dict)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
preprocess = transforms.Compose([transforms.Resize((cfg.input_size, cfg.input_size)), transforms.ToTensor(), normalize])

norm_indices = [60, 72]
labels = _tools.get_label('test.txt')

nmes_std = []
nmes_merge = []
norm = None
time_all = 0
for label in labels:
    image_name = label[0]
    lms_gt = label[1]
    norm = np.linalg.norm(lms_gt.reshape(-1, 2)[norm_indices[0]] - lms_gt.reshape(-1, 2)[norm_indices[1]])

    image_path = os.path.join(_root.gen_data_wflw_root, 'WFLW', 'images_test', image_name)
    print(image_path)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (cfg.input_size, cfg.input_size))
    inputs = Image.fromarray(image[:, :, ::-1].astype('uint8'), 'RGB')
    inputs = preprocess(inputs).unsqueeze(0)
    inputs = inputs.to(device)
    t1 = time.time()

    lms_pred_x, lms_pred_y = forward_net(net, inputs)

    # merge neighbor predictions
    lms_pred = torch.cat((lms_pred_x, lms_pred_y), dim=1).flatten()
    tmp_x = torch.mean(lms_pred_x, dim=1).view(-1, 1)
    tmp_y = torch.mean(lms_pred_y, dim=1).view(-1, 1)
    lms_pred_merge = torch.cat((tmp_x, tmp_y), dim=1).flatten()
    t2 = time.time()
    time_all += (t2 - t1)

    lms_pred = lms_pred.cpu().numpy()
    lms_pred_merge = lms_pred_merge.cpu().numpy()

    nme_std = compute_nme(lms_pred, lms_gt, norm)
    nmes_std.append(nme_std)
    nme_merge = compute_nme(lms_pred_merge, lms_gt, norm)
    nmes_merge.append(nme_merge)

print('Total inference time:', time_all)
print('Image num:', len(labels))
print('Average inference time:', time_all / len(labels))


print('nme: {}'.format(np.mean(nmes_merge)))
logging.info('nme: {}'.format(np.mean(nmes_merge)))

fr, auc = compute_fr_and_auc(nmes_merge)
print('fr : {}'.format(fr))
logging.info('fr : {}'.format(fr))
print('auc: {}'.format(auc))
logging.info('auc: {}'.format(auc))
