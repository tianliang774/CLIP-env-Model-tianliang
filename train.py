import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler
from tqdm import tqdm
import os
import argparse
import time

import warnings

from model.Universal_model import Universal_model

warnings.filterwarnings("ignore")

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter


from dataset.dataloader import ImageABDataset
from utils import loss
from utils.utils import TEMPLATE, NUM_CLASS
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

torch.multiprocessing.set_sharing_strategy('file_system')


def train(args, train_loader, model, optimizer, loss_L1_function):
    model.train()
    L1_ave = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, y, name = batch["A"].to(args.device), batch["B"].float().to(args.device), batch['name']
        logit_map = model(x)
        L1 = loss_L1_function(logit_map, y, name, TEMPLATE)*1000
        loss = L1
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (L1_loss=%2.5f)" % (
                args.epoch, step, len(train_loader), L1.item())
        )
        L1_ave += L1.item()
        torch.cuda.empty_cache()
    print('Epoch=%d: L1_loss=%2.5f' % (
        args.epoch, L1_ave / len(epoch_iterator)))

    return L1_ave / len(epoch_iterator)


def process(args):
    rank = 0
    # for distributed training
    if args.dist:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = args.local_rank
        print(rank)

    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)

    # prepare the 2D Universal_model
    model = Universal_model(img_size=(args.roi_x, args.roi_y, args.roi_z),
                            in_channels=1,
                            out_channels=NUM_CLASS,
                            backbone=args.backbone,
                            encoding=args.trans_encoding
                            )

    # Load pre-trained weights
    if args.pretrain is not None:
        model.load_params(torch.load(args.pretrain)["state_dict"])

    if args.trans_encoding == 'word_embedding':
        word_embedding = torch.load(args.word_embedding)
        model.organ_embedding.data = word_embedding.float()
        print('load word embedding')

    model.to(args.device)
    model.train()
    # args.dist->distributed
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[args.device], find_unused_parameters=True)

    # criterion
    loss_L1_function = loss.Multi_L1Loss(NUM_CLASS).to(args.device)
    # setting optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # setting scheduler
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)

    if args.resume:
        checkpoint = torch.load(args.resume)
        if args.dist:
            model.load_state_dict(checkpoint['net'])
        else:
            store_dict = model.state_dict()
            model_dict = checkpoint['net']
            for key in model_dict.keys():
                store_dict['.'.join(key.split('.')[1:])] = model_dict[key]
            model.load_state_dict(store_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])

        print('success resume from ', args.resume)

    torch.backends.cudnn.benchmark = True

    train_dataset = ImageABDataset(args)
    train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True) if args.dist else None
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=int(args.batch_size),
                              shuffle=(train_sampler is None),
                              num_workers=args.num_workers,
                              sampler=train_sampler)

    if rank == 0:
        writer = SummaryWriter(log_dir='out/' + args.log_name)
        print('Writing Tensorboard logs to ', 'out/' + args.log_name)

    while args.epoch < args.max_epoch:
        if args.dist:
            dist.barrier()
            train_sampler.set_epoch(args.epoch)
        scheduler.step()

        loss_l1 = train(args, train_loader, model, optimizer, loss_L1_function)
        if rank == 0:
            writer.add_scalar('train_L1_loss', loss_l1, args.epoch)
            writer.add_scalar('lr', scheduler.get_lr(), args.epoch)

        if (args.epoch % args.store_num == 0 and args.epoch != 0) and rank == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": args.epoch
            }
            if not os.path.isdir('out/' + args.log_name):
                os.mkdir('out/' + args.log_name)
            torch.save(checkpoint, 'out/' + args.log_name + '/epoch_' + str(args.epoch) + '.pth')
            print('save model success')

        args.epoch += 1

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')

    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='unet', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--backbone', default='dinov2', help='backbone [swinunetr or unet or dints or unetpp]')
    parser.add_argument('--resume', default=None, help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default=None,  # swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
                        help='The path of pretrain model. Eg, ./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt')
    parser.add_argument('--trans_encoding', default='word_embedding',
                        help='the type of encoding: rand_embedding or word_embedding')
    parser.add_argument('--word_embedding', default='./pretrained_weights/txt_encoding.pth',
                        help='The path of word embedding')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=2000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=5, type=int, help='Store model how often')
    parser.add_argument('--warmup_epoch', default=100, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_123457891213', 'PAOT_10_inner'])  # 'PAOT', 'felix'
    ### please check this argment carefully
    ### PAOT: include PAOT_123457891213 and PAOT_10
    ### PAOT_123457891213: include 1 2 3 4 5 7 8 9 12 13
    ### PAOT_10_inner: same with NVIDIA for comparison
    ### PAOT_10: original division
    parser.add_argument('--data_root_path', default='/computenodes/node31/team1/jliu/data/ct_data/',
                        help='data root path')
    parser.add_argument('--batch_size', default=1, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=2, type=int, help='sample number in each ct')

    parser.add_argument('--phase', default='train', help='train or validation or test')
    parser.add_argument('--uniform_sample', action="store_true", default=False,
                        help='whether utilize uniform sample strategy')

    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0.005, type=float, help='The percentage of cached data in total')

    args = parser.parse_args()

    process(args=args)


if __name__ == "__main__":
    main()

# python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 train.py --dist True --uniform_sample
