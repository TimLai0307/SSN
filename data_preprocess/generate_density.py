import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
from matplotlib import cm as CM
from image import *
import torch
import cv2


def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)

    if gt_count == 0:
        return density

    print('generate density...')
    sigma = 4
    density += scipy.ndimage.filters.gaussian_filter(gt, sigma, mode='constant')
    print('done.')

    return density

root = 'D:/Lai/counting_dataset/ShanghaiTech/part_A/'
target_root = 'D:/Lai/counting_dataset/test/adaptive_kernel/SHHA_256'
cls = 'SHHA_256'

if cls == 'SHHA':
    train = os.path.join(root,'train_data','images')
    test = os.path.join(root,'test_data','images')
    path_sets = [train, test]

    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    for img_path in img_paths:
        print(img_path)
        mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_'))
        img= cv2.imread(img_path)
        gt = mat["image_info"][0,0][0,0][0]
        # min_size = min(img.shape[0], img.shape[1])
        # if min_size < 256:
        #     scale = 256 / min_size
        #     # scale the image and points
        #     img = cv2.resize(img, None, fx=scale, fy=scale)
        #     if len(gt) == 0:
        #         gt = gt
        #     else:
        #         gt *= scale
        k = np.zeros((img.shape[0], img.shape[1]))
        for i in range(0,len(gt)):
            if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
                k[int(gt[i][1]),int(gt[i][0])]=1
        k = gaussian_filter_density(k)
        # print(k.sum())
        with h5py.File(img_path.replace('.jpg','.h5').replace('images','density-map'), 'w') as hf:
                hf['density'] = k
        # cv2.imwrite(img_path, img)
        # plt.imshow(k, cmap=CM.jet)
        # plt.axis('off')
        # count = k.sum()
        # plt.savefig('D:/Lai/counting_dataset/test/result/input/ground-truth/' + str(count) + '.jpg', bbox_inches='tight', pad_inches=0)


elif cls == 'NWPU':
    # train = os.path.join(target_root, 'train')
    # test = os.path.join(target_root, 'tests')
    # path_sets = [train, test]
    path = root

    img_paths = []

    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

    j = 0

    img_paths = img_paths[j:]

    for img_path in img_paths:
        print(img_path, j)

        if j < 3109:
            target_path = target_root + '/train'
            img_name = img_path.split('\\')[-1]
            mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('image', 'mats'))
            img = cv2.imread(img_path)
            max_size = max(img.shape)
            gt = mat['annPoints']
            if max_size > 1920:
                scale = 1920 / max_size
                # scale the image and points
                img = cv2.resize(img, None, fx=scale, fy=scale)
                if len(gt) == 0:
                    gt = gt
                else:
                    gt *= scale
            k = np.zeros((img.shape[0], img.shape[1]))
            for i in range(0, len(gt)):
                if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                    k[int(gt[i][1]), int(gt[i][0])] = 1
            k = gaussian_filter_density(k)
            # print(k.sum())
            save_path = target_path + '/' + img_name
            cv2.imwrite(save_path, img)
            with h5py.File(save_path.replace('.jpg','.h5'), 'w') as hf:
                    hf['density'] = k
        else:
            # mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('test', 'mats'))
            target_path = target_root + '/tests'
            img_name = img_path.split('\\')[-1]
            mat = io.loadmat(img_path.replace('.jpg', '.mat').replace('image', 'mats'))
            img = cv2.imread(img_path)
            max_size = max(img.shape)
            gt = mat['annPoints']
            if max_size > 1920:
                scale = 1920 / max_size
                # scale the image and points
                img = cv2.resize(img, None, fx=scale, fy=scale)
                if len(gt) == 0:
                    gt = gt
                else:
                    gt *= scale
            k = np.zeros((img.shape[0], img.shape[1]))
            for i in range(0, len(gt)):
                if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                    k[int(gt[i][1]), int(gt[i][0])] = 1
            k = gaussian_filter_density(k)
            # print(k.sum())
            save_path = target_path + '/' + img_name
            cv2.imwrite(save_path, img)
            with h5py.File(save_path.replace('.jpg', '.h5'), 'w') as hf:
                hf['density'] = k

        print('den_count = ' + str(k.sum()))
        j += 1
        # plt.imshow(k, cmap=CM.jet)
        # plt.axis('off')
        # count = k.sum()
        # plt.savefig('D:/Lai/counting_dataset/test/result/input/ground-truth/' + str(count) + '.jpg', bbox_inches='tight', pad_inches=0)

elif cls == 'UCF_QNRF':
    train = os.path.join(root, 'Train')
    test = os.path.join(root, 'Test')
    path_sets = [train, test]
    # path_sets = [test]

    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    j = 0

    img_paths = img_paths[j:]

    for img_path in img_paths:
        print(img_path, j)
        if j < 1201:
            target_path = target_root + '/train'
            img_name = img_path.split('\\')[-1]
            mat = io.loadmat(img_path.replace('.jpg', '_ann.mat'))
            img = cv2.imread(img_path)
            max_size = max(img.shape)

            gt = mat['annPoints']
            if max_size > 1920:
                scale = 1920 / max_size
                # scale the image and points
                img = cv2.resize(img, None, fx=scale, fy=scale)
                if len(gt) == 0:
                    gt = gt
                else:
                    gt *= scale
            k = np.zeros((img.shape[0], img.shape[1]))
            for i in range(0, len(gt)):
                if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                    k[int(gt[i][1]), int(gt[i][0])] = 1
            k = gaussian_filter_density(k)
            # print(k.sum())
            save_path = target_path + '/' + img_name
            cv2.imwrite(save_path, img)
            with h5py.File(save_path.replace('.jpg', '.h5'), 'w') as hf:
                hf['density'] = k
        else:
            target_path = target_root + '/test'
            img_name = img_path.split('\\')[-1]
            mat = io.loadmat(img_path.replace('.jpg', '_ann.mat'))
            img = cv2.imread(img_path)
            max_size = max(img.shape)

            gt = mat['annPoints']
            if max_size > 1920:
                scale = 1920 / max_size
                # scale the image and points
                img = cv2.resize(img, None, fx=scale, fy=scale)
                if len(gt) == 0:
                    gt = gt
                else:
                    gt *= scale
            k = np.zeros((img.shape[0], img.shape[1]))
            for i in range(0, len(gt)):
                if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                    k[int(gt[i][1]), int(gt[i][0])] = 1
            k = gaussian_filter_density(k)
            # print(k.sum())
            save_path = target_path + '/' + img_name
            cv2.imwrite(save_path, img)
            with h5py.File(save_path.replace('.jpg', '.h5'), 'w') as hf:
                hf['density'] = k

        print('den_count = ' + str(k.sum()))
        j += 1
