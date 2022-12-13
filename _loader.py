import torch.utils.data as data
import torch
from PIL import Image, ImageFilter
import os, cv2
import numpy as np
import random

import torchvision.transforms as transforms
from scipy.stats import norm
from math import floor

import Loader
import _root
import _tools
import _preprocess


def get_coo(coo):
    x = []
    y = []
    for j in range(len(coo)):
        if j % 2 == 0:
            x.append(float(coo[j]))
        else:
            y.append(float(coo[j]))
    return x, y


def random_translate(image, target):
    if random.random() > 0.5:
        image_height, image_width = image.size
        a = 1
        b = 0
        # c = 30 #left/right (i.e. 5/-5)
        c = int((random.random() - 0.5) * 60)
        d = 0
        e = 1
        # f = 30 #up/down (i.e. 5/-5)
        f = int((random.random() - 0.5) * 60)
        image = image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f))
        target_translate = target.copy()
        target_translate = target_translate.reshape(-1, 2)
        target_translate[:, 0] -= 1. * c / image_width
        target_translate[:, 1] -= 1. * f / image_height
        target_translate = target_translate.flatten()
        target_translate[target_translate < 0] = 0
        target_translate[target_translate > 1] = 1
        return image, target_translate
    else:
        return image, target


# 上面的代码块实现了随机平移图像和目标的功能。首先，使用random模块的random函数生成一个随机数，如果这个数大于0.5，则执行平移操作，否则返回原图像和目标。平移操作中，首先使用PIL库的transform函数对图像进行平移操作，具体的平移量由c和f两个变量控制。随后，对目标进行平移操作，使用numpy模块的reshape函数将目标转换为二维数组，再对每个目标进行平移操作。最后，对目标进行约束，限制目标坐标在[0,1]范围内，并返回平移后的图像和目标。
def random_blur(image):
    if random.random() > 0.7:
        image = image.filter(ImageFilter.GaussianBlur(random.random() * 5))
    return image


# 上面的代码块实现了随机模糊图像的功能。首先，使用random模块的random函数生成一个随机数，如果这个数大于0.7，则执行模糊操作，否则返回原图像。模糊操作中，使用PIL库的filter函数，并传入ImageFilter.GaussianBlur参数，具体的模糊程度由参数random.random()*5控制。最后，返回模糊后的图像。
def random_occlusion(image):
    if random.random() > 0.5:
        image_np = np.array(image).astype(np.uint8)
        image_np = image_np[:, :, ::-1]
        image_height, image_width, _ = image_np.shape
        occ_height = int(image_height * 0.4 * random.random())
        occ_width = int(image_width * 0.4 * random.random())
        occ_xmin = int((image_width - occ_width - 10) * random.random())
        occ_ymin = int((image_height - occ_height - 10) * random.random())
        image_np[occ_ymin:occ_ymin + occ_height, occ_xmin:occ_xmin + occ_width, 0] = int(random.random() * 255)
        image_np[occ_ymin:occ_ymin + occ_height, occ_xmin:occ_xmin + occ_width, 1] = int(random.random() * 255)
        image_np[occ_ymin:occ_ymin + occ_height, occ_xmin:occ_xmin + occ_width, 2] = int(random.random() * 255)
        image_pil = Image.fromarray(image_np[:, :, ::-1].astype('uint8'), 'RGB')
        return image_pil
    else:
        return image


# 上面的代码块实现了随机遮挡图像的功能。首先，使用random模块的random函数生成一个随机数，如果这个数大于0.5，则执行遮挡操作，否则返回原图像。遮挡操作中，首先将图像转换为numpy数组，再调整通道顺序，使用numpy模块的shape函数获取图像高度和宽度，并使用random模块的random函数生成遮挡高度和宽度，以及遮挡位置的随机数。随后，根据遮挡位置和大小，在图像的numpy数组中随机填充颜色，最后将numpy数组转换为Image对象，并返回遮挡后的图像。
def random_flip(image, target, points_flip):
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        target = np.array(target).reshape(-1, 2)
        target = target[points_flip, :]
        target[:, 0] = 1 - target[:, 0]
        target = target.flatten()
        return image, target
    else:
        return image, target


# 上面的代码块实现了随机翻转图像和目标的功能。首先，使用random模块的random函数生成一个随机数，如果这个数大于0.5，则执行翻转操作，否则返回原图像和目标。翻转操作中，首先使用PIL库的transpose函数对图像进行翻转操作，然后使用numpy模块的array函数将目标转换为numpy数组，再使用numpy模块的reshape函数将目标转换为二维数组。随后，使用points_flip数组筛选出需要翻转的目标，并对这些目标进行翻转操作。最后，使用numpy模块的flatten函数将二维数组转换为一维数组，并返回翻转后的图像和目标。
def random_rotate(image, target, angle_max):
    if random.random() > 0.5:
        center_x = 0.5
        center_y = 0.5
        landmark_num = int(len(target) / 2)
        target_center = np.array(target) - np.array([center_x, center_y] * landmark_num)
        target_center = target_center.reshape(landmark_num, 2)
        theta_max = np.radians(angle_max)
        theta = random.uniform(-theta_max, theta_max)
        angle = np.degrees(theta)
        image = image.rotate(angle)

        c, s = np.cos(theta), np.sin(theta)
        rot = np.array(((c, -s), (s, c)))
        target_center_rot = np.matmul(target_center, rot)
        target_rot = target_center_rot.reshape(landmark_num * 2) + np.array([center_x, center_y] * landmark_num)
        return image, target_rot
    else:
        return image, target


# 旋转操作中，首先使用random模块的uniform函数生成随机旋转角度，并使用PIL库的rotate函数对图像进行旋转操作。随后，使用numpy模块的array函数将目标转换为numpy数组，再使用numpy模块的matmul函数对目标进行旋转操作。最后，返回旋转后的图像和目标。
def gen_target_pip(target, meanface_indices, target_map, target_local_x, target_local_y, target_nb_x, target_nb_y):
    num_nb = len(meanface_indices[0])  # ( 10,len(meanface_indices)=98 )
    map_channel, map_height, map_width = target_map.shape
    target = target.reshape(-1, 2)
    assert map_channel == target.shape[0]

    for i in range(map_channel):
        mu_x = int(floor(target[i][0] * map_width))
        mu_y = int(floor(target[i][1] * map_height))
        mu_x = max(0, mu_x)
        mu_y = max(0, mu_y)
        mu_x = min(mu_x, map_width - 1)
        mu_y = min(mu_y, map_height - 1)
        target_map[i, mu_y, mu_x] = 1
        shift_x = target[i][0] * map_width - mu_x
        shift_y = target[i][1] * map_height - mu_y
        target_local_x[i, mu_y, mu_x] = shift_x
        target_local_y[i, mu_y, mu_x] = shift_y

        for j in range(num_nb):
            nb_x = target[meanface_indices[i][j]][0] * map_width - mu_x
            nb_y = target[meanface_indices[i][j]][1] * map_height - mu_y
            target_nb_x[num_nb * i + j, mu_y, mu_x] = nb_x
            target_nb_y[num_nb * i + j, mu_y, mu_x] = nb_y

    return target_map, target_local_x, target_local_y, target_nb_x, target_nb_y


class LandmarksDataset(data.Dataset):
    def __init__(self, root, imgs, input_size, num_lms, net_stride, points_flip, transform=None, target_transform=None):
        self.root = root
        self.imgs = imgs
        self.num_lms = num_lms
        self.net_stride = net_stride
        self.points_flip = points_flip
        self.transform = transform
        self.target_transform = target_transform
        self.input_size = input_size

    def __getitem__(self, index):

        img_name, target = self.imgs[index]

        img = Image.open(os.path.join(self.root, img_name)).convert('RGB') #(256, 256, 3)
        img, target = random_translate(img, target)
        img = random_occlusion(img)
        img, target = random_flip(img, target, self.points_flip)
        img, target = random_rotate(img, target, 30)
        img = random_blur(img)
        # print('target',len(target))=196

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        target_x, target_y = get_coo(target)
        return img, np.array(target_x), np.array(target_y)

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    points_flip = [32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7,
                   6, 5, 4, 3, 2, 1, 0, 46, 45, 44, 43, 42, 50, 49, 48, 47, 37, 36, 35, 34, 33, 41, 40, 39, 38, 51, 52,
                   53, 54, 59, 58, 57, 56, 55, 72, 71, 70, 69, 68, 75, 74, 73, 64, 63, 62, 61, 60, 67, 66, 65, 82, 81,
                   80, 79, 78, 77, 76, 87, 86, 85, 84, 83, 92, 91, 90, 89, 88, 95, 94, 93, 97, 96]
    assert len(points_flip) == 98

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    traindataset = LandmarksDataset(os.path.join(_root.gen_data_wflw_root, 'WFLW', 'images_train'),
                                    _tools.get_label('train.txt'),
                                    256,
                                    98,
                                    32,
                                    points_flip,
                                    transforms.Compose([
                                        transforms.RandomGrayscale(0.2),
                                        transforms.ToTensor(),
                                        normalize])
                                    )
    train_loader = Loader.DataLoader(dataset=traindataset, batch_size=20, shuffle=True)

    pass
