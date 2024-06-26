import argparse
import datetime
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, DistributedSampler

from datasets.loading_data import building_data
from engine import *
from models.SSN import build_model
import os
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training P2PNet', add_help=False)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=4000, type=int)
    parser.add_argument('--lr_drop', default=10000, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")

    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    parser.add_argument('--backbone_path', default='', type=str,
                        help="Root of the backbone checkpoint")

    # dataset parameters
    parser.add_argument('--data_root', default='D:/Lai/counting_dataset/test/adaptive_kernel/SHHA',
                        help='path where the dataset is')

    # save_dir
    parser.add_argument('--log_dir', default='./log',
                        help='path where to save, empty for no saving')
    parser.add_argument('--checkpoints_dir', default='./ckpt',
                        help='path where to save checkpoints, empty for no saving')
    parser.add_argument('--tensorboard_dir', default='./runs',
                        help='path where to save, empty for no saving')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=1, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='frequency of evaluation, default setting is evaluating in every 5 epoch')
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')

    return parser

def main(args): #主程式
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)

    # create the logging file
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    run_log_name = os.path.join(args.log_dir, 'run_log.txt')
    with open(run_log_name, "w") as log_file:
        log_file.write('Eval Log %s\n' % time.strftime("%c"))

    # backup the arguments
    print(args)
    with open(run_log_name, "a") as log_file:
        log_file.write("{}".format(args))
    device = torch.device('cuda')
    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # get model
    model, criterion = build_model(args, training=True)
    # move to GPU
    model.to(device)
    criterion.to(device)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    # use different optimation params for different parts of the model
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    # Adam is used by default
    optimizer = torch.optim.Adam(param_dicts, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    # create the training and valiation set
    train_set, val_set = building_data(args.data_root)
    # create the sampler used during training
    sampler_train = torch.utils.data.RandomSampler(train_set)
    sampler_val = torch.utils.data.SequentialSampler(val_set)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    # the dataloader for training
    data_loader_train = DataLoader(train_set, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    data_loader_val = DataLoader(val_set, batch_size=1, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn_crowd, num_workers=args.num_workers)

    if args.frozen_weights is not None:  # 讀pretrain
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
    # resume the weights and training state if exists
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        args.start_epoch = checkpoint['epoch'] + 1


    # 開始訓練
    print("'=======================================Train======================================='")
    start_time = time.time()
    # save the performance during the training
    mae = []
    mse = []
    step = args.start_epoch // args.eval_freq + 1
    # the logger writer
    writer = SummaryWriter(args.tensorboard_dir)

    # training starts here
    for epoch in range(args.start_epoch, args.epochs+1):

        t1 = time.time()
        stat = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)

        # record the training states after every epoch
        if writer is not None:
            with open(run_log_name, "a") as log_file:
                log_file.write("loss/loss@{}: {} \n".format(epoch, stat['loss']))

            writer.add_scalar('loss/loss', stat['loss'], epoch)


        t2 = time.time()
        print('[ep %d][lr %.7f][%.2fs]' % \
              (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        with open(run_log_name, "a") as log_file:
            log_file.write('[ep %d][lr %.7f][%.2fs]' % (epoch, optimizer.param_groups[0]['lr'], t2 - t1))
        # change lr according to the scheduler
        lr_scheduler.step()
        # save latest weights every epoch
        if not os.path.isdir(args.checkpoints_dir):
            os.mkdir(args.checkpoints_dir)
        checkpoint_latest_path = os.path.join(args.checkpoints_dir, 'latest.pth')
        torch.save({
            'model': model_without_ddp.state_dict(),
            'epoch': epoch,
        }, checkpoint_latest_path)
        # run evaluation
        if epoch % args.eval_freq == 0 and epoch != 0:
            t1 = time.time()
            result = evaluate_crowd_no_overlap(model, data_loader_val, device)
            t2 = time.time()

            mae.append(result[0])
            mse.append(result[1])
            # print the evaluation results
            print('=======================================Test=======================================')
            print("mae:", result[0], "mse:", result[1], "time:", t2 - t1, "best mae:", np.min(mae), )
            with open(run_log_name, "a") as log_file:
                log_file.write("mae:{}, mse:{}, time:{}, best mae:{} \n".format(result[0],
                                result[1], t2 - t1, np.min(mae)))
            print('=======================================Test=======================================')
            # recored the evaluation results
            if writer is not None:
                with open(run_log_name, "a") as log_file:
                    log_file.write("metric/mae@{}: {} \n".format(step, result[0]))
                    log_file.write("metric/mse@{}: {} \n".format(step, result[1]))
                writer.add_scalar('metric/mae', result[0], step)
                writer.add_scalar('metric/mse', result[1], step)
                step += 1

            # save the best model since begining
            if abs(np.min(mae) - result[0]) < 0.01:
                checkpoint_best_path = os.path.join(args.checkpoints_dir, 'best_mae.pth')
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'epoch': epoch,
                }, checkpoint_best_path)
    # total time for training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    opt = args.parse_args()
    main(opt)