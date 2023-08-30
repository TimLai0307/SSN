import argparse
import datetime
import random
import time
from pathlib import Path
from time import sleep
from tqdm import tqdm, trange
import torch
import torchvision.transforms as standard_transforms
import numpy as np
import scipy.io as io
from PIL import Image
import cv2
from crowd_datasets import build_dataset
from engine import *
from models import build_model_can_plus
import os
import warnings
import h5py
from matplotlib import cm as CM
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")


    parser.add_argument('--output_dir', default='./result/test',
                        help='path where to save')
    parser.add_argument('--weight_path', default='D:/Lai/counting/Crowdcounting_model/ckpt/3.5.7_ep5000_d512x2/best_mae.pth',
                        help='path where the trained weights saved')

    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def main(args, debug=False):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cuda')
    # get the P2PNet
    model = build_model_can_plus(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()
    with torch.no_grad():
        # create the pre-processing transform
        transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        root = 'D:/Lai/counting_dataset/test/result/input/all/image'
        img_list = os.listdir(root)

        progress = tqdm(total=len(img_list))

        maes = []
        mses = []

        for img_name in img_list:
            # load the images
            progress.update(1)
            img_path = os.path.join(root, img_name)
            img = cv2.imread(img_path)
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # round the size
            # width, height = img_raw.size

            # if max(img_raw.size) > 1920:
            #     max_size = max(img_raw.size)
            #     scale = 1920 / max_size
            #     width = int(width * scale)
            #     height = int(height * scale)

            # new_width = width // 128 * 128
            # new_height = height // 128 * 128
            #
            # img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
            # pre-proccessing
            img = transform(img)

            gt_path = img_path.replace('jpg','h5').replace('image', 'gt')
            gt_file = h5py.File(gt_path)
            target = np.asarray(gt_file['density'])
            gt_cnt = target.sum()


            samples = torch.Tensor(img).unsqueeze(0)
            samples = samples.to(device)
            # run inference
            outputs = model(samples)
            count = outputs.detach().cpu().sum().numpy()
            plt.imshow(outputs.squeeze().detach().cpu().numpy(), cmap=CM.jet)
            plt.axis('off')
            plt.savefig('D:/Lai/counting_dataset/test/result/Crowdcounting/SHHA/SSM+AM/' + img_name + '_' + str(count) + '.jpg', bbox_inches='tight', pad_inches=0)

            # filter the predictions
            #print(predict_cnt)

            mae = abs(count - gt_cnt)
            print("pic", img_name, " ", "mae", mae)
            mse = (count - gt_cnt) * (count - gt_cnt)
            maes.append(float(mae))
            mses.append(float(mse))
            # calc MAE, MSE
            sleep(0.01)

        mae = np.mean(maes)
        mse = np.sqrt(np.mean(mses))
        print("maes:", mae, "  ",  "mse:", mse)
        print("min:", min(maes))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)