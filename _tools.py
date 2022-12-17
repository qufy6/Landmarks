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
    if len(labels[0]) == 1:
        return labels

    labels_new = []
    for label in labels:  # 7500
        image_name = label[0]
        target = label[1:]
        target = np.array([float(x) for x in target])
        if task_type is None:
            labels_new.append([image_name, target])
        else:
            labels_new.append([image_name, task_type, target])
    # print(np.array(labels_new).shape)#(7500, 2)
    return labels_new


def compute_loss_pip(outputs_local_x, outputs_local_y, labels_local_x, labels_local_y, criterion_reg):
    loss_x = criterion_reg(outputs_local_x, labels_local_x)
    loss_y = criterion_reg(outputs_local_y, labels_local_y)
    return loss_x, loss_y


def train_model(net, train_loader, criterion_reg, optimizer, num_epochs, scheduler, save_dir, device):
    best_epoch_loss = 1000
    for epoch in range(num_epochs):
        t = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        logging.info('-' * 10)
        net.train()
        epoch_loss = 0.0

        for i, data in enumerate(train_loader):
            img, labels_x, labels_y = data
            img = img.to(device)
            labels_x = labels_x.type(torch.FloatTensor).to(device)
            labels_y = labels_y.type(torch.FloatTensor).to(device)
            output = net(img)  # (batchsize, 196)
            pred_x = output[:, 0: 98]
            pred_y = output[:, 98:196]
            loss_x, loss_y = criterion_reg(pred_x, labels_x), criterion_reg(pred_y, labels_y)
            loss = loss_x + loss_y
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> '.format(
                    epoch, num_epochs - 1, i, len(train_loader) - 1, loss.item()))
                logging.info('[Epoch {:d}/{:d}, Batch {:d}/{:d}] <Total loss: {:.6f}> '.format(
                    epoch, num_epochs - 1, i, len(train_loader) - 1, loss.item()))
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        if epoch_loss < best_epoch_loss:
            best_epoch_loss = epoch_loss
            print("saving checkpoint for epoch ", epoch)
            logging.info('saving checkpoint for epoch {:f} '.format(epoch))
            filename = os.path.join(save_dir, 'best.pth')
            torch.save(net.state_dict(), filename)
        print('running time for one epoch in seconds: ', time.time() - t)
        logging.info('running time for one epoch in seconds: {:f} '.format(time.time() - t))
        print('epoch loss : ', epoch_loss)
        logging.info('epoch loss :{:f} '.format(epoch_loss))
        scheduler.step()
    return net
