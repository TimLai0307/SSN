import argparse
import datetime
import random
import time
from pathlib import Path
from time import sleep
from tqdm import tqdm
import torch
import torchvision.transforms as standard_transforms
import numpy as np
import scipy.io as io
from PIL import Image
import cv2
from engine import *
from models import build_model
import os
import warnings
import h5py
from matplotlib import cm as CM
import glob
from matplotlib import pyplot as plt
warnings.filterwarnings('ignore')


mngr = plt.get_current_fig_manager()
mngr.window.wm_geometry("+300+300")

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for evaluation', add_help=False)
    
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    parser.add_argument('--vis', default=False,
                        help='Save the visual results or not')

    parser.add_argument('--data_path', default='D:/Lai/counting_dataset/test/preprocess_data/SHHA/test',
                        help='path of test data')
    parser.add_argument('--output_dir', default='./vis_result',
                        help='path where to save')
    parser.add_argument('--weight_path', default='D:/Lai/counting/Crowdcounting_Theses/ckpt/best_mae_mean.pth',
                        help='path where the trained weights saved')


    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')

    return parser

def main(args, debug=False):

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    print(args)
    device = torch.device('cuda')
    # get
    model = build_model(args)
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

        root = args.data_path
        target_root = args.output_dir
        img_list = glob.glob(os.path.join(root, '*.jpg'))

        progress = tqdm(total=len(img_list))

        maes = []
        mses = []

        for img_path in img_list:
            # load the images
            progress.update(1)
            img_name = img_path.split('\\')[-1].split('.')[0]
            img = cv2.imread(img_path)

            # pre-proccessing
            img = transform(img)

            # padding
            c,h,w = img.size()
            new_h = ((h - 1) // 128 + 1) * 128
            new_w = ((w - 1) // 128 + 1) * 128

            input = np.zeros((c, new_h, new_w))
            input[:, : h, : w] = img

            gt_path = img_path.replace('jpg','h5')
            gt_file = h5py.File(gt_path)
            target = np.asarray(gt_file['density'])
            gt_cnt = target.sum()

            # samples = torch.Tensor(input).unsqueeze(0)
            samples = torch.Tensor(input).unsqueeze(0)
            samples = samples.to(device)

            # run inference
            outputs = model(samples)
            out1 = outputs[1][:, :, : h, : w]
            out2 = outputs[2][:, :, : h, : w]

            out = (out1 + out2)/2

            count = out.detach().cpu().sum().numpy()

            # filter the predictions
            #print(predict_cnt)

            # calc MAE, MSE
            mae = abs(count - gt_cnt)
            mse = (count - gt_cnt) * (count - gt_cnt)
            maes.append(float(mae))
            mses.append(float(mse))

            # vis
            if args.vis:
                plt.axis('off')
                plt.imshow(out.squeeze().detach().cpu().numpy(), cmap=CM.jet)
                plt.savefig(target_root + '/' + img_name + '_count=' + str(count) + '.jpg',
                            bbox_inches='tight',
                            pad_inches=0)


        mae = np.mean(maes)
        mse = np.sqrt(np.mean(mses))
        print("mae:", mae, "  ",  "mse:", mse)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)