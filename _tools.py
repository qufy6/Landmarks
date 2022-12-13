import os, cv2
import numpy as np
from PIL import Image, ImageFilter
import logging
import torch
import torch.nn as nn
import random
import time
from scipy.integrate import simps

import _loader
import _root


def get_label(label_file, task_type=None):
    label_path = os.path.join(_root.gen_data_wflw_root, 'WFLW', label_file)
    with open(label_path, 'r') as f:
        labels = f.readlines()
    labels = [x.strip().split() for x in labels]
    if len(labels[0])==1:
        return labels

    labels_new = []
    for label in labels:
        image_name = label[0]
        target = label[1:]
        target = np.array([float(x) for x in target])
        if task_type is None:
            labels_new.append([image_name, target])
        else:
            labels_new.append([image_name, task_type, target])
    return labels_new

def compute_loss_pip(outputs_local_x, outputs_local_y, labels_local_x, labels_local_y,  criterion_reg):

    loss_x = criterion_reg(outputs_local_x, labels_local_x)
    loss_y = criterion_reg(outputs_local_y, labels_local_y)
    return loss_x, loss_y


def train_model(net, train_loader, criterion_reg, optimizer, num_epochs, scheduler, save_dir, save_interval, device):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        logging.info('-' * 10)
        net.train()
        epoch_loss = 0.0

        for i, data in enumerate(train_loader):
            img, labels_x, labels_y = data
            img = img.to(device)
            labels_x = labels_x.to(device)
            labels_y = labels_y.to(device)
            output = net(img)
            pred_x, pred_y = _loader.get_coo(output)
            loss = compute_loss_pip(pred_x, pred_y, labels_x, labels_y, criterion_reg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%10 == 0:
                print('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> '.format(
                    epoch, num_epochs-1, i, len(train_loader)-1, loss.item()))
                logging.info('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> '.format(
                    epoch, num_epochs-1, i, len(train_loader)-1, loss.item()))
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        if epoch%(save_interval-1) == 0 and epoch > 0:
            filename = os.path.join(save_dir, 'epoch%d.pth' % epoch)
            torch.save(net.state_dict(), filename)
            print(filename, 'saved')
        scheduler.step()
    return net

