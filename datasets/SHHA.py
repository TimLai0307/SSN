import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io
import h5py

class loading_data(Dataset):
    def __init__(self, data_root, transform=None, train=False, patch=False, flip=False):
        # self定義
        self.root_path = data_root
        self.train_lists = "train.txt"
        self.eval_list = "test.txt"
        # there may exist multiple list files
        self.img_list_file = self.train_lists.split(',')
        if train:
            self.img_list_file = self.train_lists.split(',')
        else:
            self.img_list_file = self.eval_list.split(',')

        self.img_map = {}
        self.img_list = []
        # loads the image/gt pairs
        for _, train_list in enumerate(self.img_list_file):
            train_list = train_list.strip()
            with open(os.path.join(self.root_path, train_list)) as fin:
                for line in fin:
                    if len(line) < 2:
                        continue
                    line = line.strip().split()
                    self.img_map[os.path.join(self.root_path, line[0].strip())] = \
                        os.path.join(self.root_path, line[1].strip())
        self.img_list = sorted(list(self.img_map.keys()))
        # number of samples
        self.nSamples = len(self.img_list)

        self.transform = transform
        self.train = train
        self.patch = patch
        self.flip = flip

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.img_list[index]
        gt_path = self.img_map[img_path]
        # load image and ground truth
        img, den = load_data((img_path, gt_path), self.train)
        # applu augumentation
        if self.transform is not None:
            img = self.transform(img)

        # random crop augumentaiton
        if self.train and self.patch:
            img, den = random_crop(img, den)
        # random flipping
        if random.random() > 0.5 and self.train and self.flip:
            # random flip
            img = torch.Tensor(img[:, :, :, ::-1].copy())
            den = torch.Tensor(den[:, :, ::-1].copy())


        img = torch.Tensor(img)
        den = torch.Tensor(den)
        if not self.train:
            den = den.unsqueeze(0)

        return img, den

def load_data(img_gt_path, train):
    img_path, gt_path = img_gt_path
    # load the images
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # load the ground truth
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])

    return img, target

# random crop augumentation
def random_crop(img, den, num_patch=4):
    half_h = 128
    half_w = 128
    result_img = np.zeros([num_patch, img.shape[0], half_h, half_w])
    result_den = np.zeros([num_patch, half_h, half_w])
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, img.size(1) - half_h)
        start_w = random.randint(0, img.size(2) - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_img[i] = img[:, start_h:end_h, start_w:end_w]
        # copy the cropped points
        result_den[i] = den[start_h:end_h, start_w:end_w]

    return result_img, result_den