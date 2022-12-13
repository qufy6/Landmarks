from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import cv2
import _root

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
root = _root.root
image_root = _root.image_root

def get_coo(coo):
    x = []
    y = []
    for j in range(len(coo)):
        if j % 2 == 0:
            x.append(float(coo[j]))
        else:
            y.append(float(coo[j]))
    return x, y


def get_face(loc):
    x_min = int(loc[0])
    y_min = int(loc[1])
    x_max = int(loc[2])
    y_max = int(loc[3])
    return x_min, y_min, x_max, y_max


class LandmarksDataset(Dataset):
    def __init__(self, txt):
        self.root = root
        self.image_root = image_root
        with open(txt, 'r') as fh:
            landmks = []
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split()
                landmks.append([words[0:196], words[196:200], words[206]])
        self.DataList = landmks
        for index in range(len(landmks)):
            x_min, y_min, x_max, y_max = get_face(self.DataList[index][1])
            for j in range(len(self.DataList[index][0])):
                if j % 2 == 0:
                    self.DataList[index][0][j] = float(self.DataList[index][0][j]) - x_min
                else:
                    self.DataList[index][0][j] = float(self.DataList[index][0][j]) - y_min

    def __getitem__(self, index):
        img_path, label = image_root + '/' + self.DataList[index][2], self.DataList[index][0]
        img = cv2.imread(img_path)
        x, y = get_coo(label)  # get coo
        x_min, y_min, x_max, y_max = get_face(self.DataList[index][1])
        img = img[y_min:y_max, x_min:x_max]
        img = cv2.resize(img, (256, 256))
        x = [256 * item / (x_max - x_min) for item in x]
        y = [256 * item / (y_max - y_min) for item in y]

        return np.transpose(img, (2, 0, 1)) / 255., np.array(x + y) / 256.  # Norm and offset

    def __len__(self):
        return len(self.DataList)

    def all_faces_scale(self):
        x, y = [], []
        for i in range(len(self.DataList)):
            x_min, y_min, x_max, y_max = get_face(self.DataList[i][1])
            x_ = x_max - x_min
            y_ = y_max - y_min
            x.append(x_)
            y.append(y_)
        plt.scatter(x, y, s=5)
        plt.title("Scales of all images")
        plt.show()


def vis_landmarks(img, landmarks):
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img[:, :, ::-1])
    x = landmarks[:int(len(landmarks) / 2)] * 256
    y = landmarks[int(len(landmarks) / 2):] * 256
    plt.scatter(x, y, s=30)
    plt.show()
    return
