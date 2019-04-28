import os
import numpy as np
from config import Config
import torch
import torch.utils.data as data
import cv2
import csv
from tqdm import tqdm
import imutils
mean = [0.9010, 0.9049, 0.9025]
std = [0.1521, 0.1347, 0.1458]


def to_int(x):
    if x.isdecimal():
        return int(x)
    return ord(x) - ord('a') + 10


class CaptchaDataset(data.Dataset):
    """ Captcha dataset warpper
    """

    def get_data(self, id):
        num = id // 5
        img_file = self.img_files[num]
        img_path = os.path.join(self.root_dir, img_file)
        im = cv2.imread(img_path)
        vt = id % 5
        if vt == 1:
            im = imutils.rotate(im, 5)
        if vt == 2:
            im = imutils.rotate(im, -5)
        if vt == 3:
            im = imutils.rotate(im, 0, None, 1.1)
        if vt == 4:
            im = imutils.rotate(im, 0, None, 0.9)
        im = im.astype(np.float32)

        im /= 255.0
        im -= mean
        im /= std
        im = torch.from_numpy(im).float().permute(2, 1, 0)
        label_seq =self.dict_image[str(img_file)]
        label_seq = [to_int(x) + 1 for x in label_seq]
        return im, torch.IntTensor(label_seq)

    def __init__(self, type, mean, std):
        tmp_path = "train"
        cnt_train = Config.cnt_train
        cnt_test = Config.cnt_test
        root_dir = os.path.join(Config.DATA_PATH, tmp_path)
        csv_file = "./data/" + tmp_path + ".csv"
        infile = open(csv_file, mode='r')
        reader = csv.reader(infile)
        self.dict_image = {rows[0]: rows[1] for rows in reader}
        super(CaptchaDataset, self).__init__()
        if not os.path.exists(root_dir):
            raise RuntimeError('cannot find root dir: {}'.format(root_dir))
        self.root_dir = root_dir
        self.img_files = [x for x in os.listdir(self.root_dir) if x.endswith('.png')]
        if type == "train":
            start_ = 0
            end_ = cnt_train
        else:
            start_ = cnt_train
            end_ = cnt_train + cnt_test
        self.img_files = self.img_files[start_:end_]
        infile.close()

    def __len__(self):
        return len(self.img_files) * 5

    def __getitem__(self, ind):
        return self.get_data(ind)


class CaptchaPredictDataSet(data.Dataset):
    def __init__(self, mean, std, st, en):
        tmp_path = "test"
        root_dir = os.path.join(Config.DATA_PATH, tmp_path)
        super(CaptchaPredictDataSet, self).__init__()
        if not os.path.exists(root_dir):
            raise RuntimeError('cannot find root dir: {}'.format(root_dir))
        self.root_dir = root_dir

        img_files = [x for x in os.listdir(self.root_dir) if x.endswith('.png')]
        self.data = []
        self.file_paths = []
        for img_file in tqdm(img_files[st:en], desc="Loading data", leave=False):
            img_path = os.path.join(self.root_dir, img_file)
            im = cv2.imread(img_path).astype(np.float32)
            im /= 255.0
            im -= mean
            im /= std
            # to tensor, H x W x C -> C x H x W
            im = torch.from_numpy(im).float().permute(2, 1, 0)
            self.data.append(im)
            self.file_paths.append(img_file)
        print("load done")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        return self.data[ind], self.file_paths[ind]
