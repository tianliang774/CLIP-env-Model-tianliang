import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import os
import argparse
import time
import cv2 as cv
import collections
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse

from model.Universal_model import Universal_model
from dataset.dataloader import ImageABDataset
from utils.utils import TEMPLATE, NUM_CLASS

torch.multiprocessing.set_sharing_strategy('file_system')


def validation(model, ValLoader, args):
    model.eval()

    test_res_dict = collections.defaultdict(lambda: collections.defaultdict(int))

    for index, batch in enumerate(tqdm(ValLoader)):
        # print('%d processd' % (index))
        x, y, name = batch["A"].to(args.device), batch["B"].float().to(args.device), batch['name'][0]

        fake_B = model(x, args=args)

        task_index = TEMPLATE[name]
        ratio, M = batch['R'].item(), batch['M'].item()

        fake_B = fake_B[:, task_index:task_index + 1, :, :].detach()
        real_B = y.detach()  # fake_B and real_B -->[1, 1, 512, 512]
        # for b in range(x.shape[0]):
        #     save_image(x[b], os.path.join("tres", f'{name}_{index}_{b}_"A".png'))
        #     save_image(fake_B[b], os.path.join("tres", f'{name}_{index}_{b}_"fake_B".png'))
        #     save_image(real_B[b], os.path.join("tres", f'{name}_{index}_{b}_"real_B".png'))
        fake_B = np.array(fake_B[0][0].cpu()) * ratio + M
        real_B = np.array(real_B[0][0].cpu()) * ratio + M
        if name == 'ct2mri':
            fake_B = cv.resize(fake_B, (256, 256))
            real_B = cv.resize(real_B, (256, 256))
        test_res_dict[name]["rmse"] += np.sqrt(mse(fake_B, real_B))
        test_res_dict[name]["psnr"] += psnr(fake_B, real_B, data_range=ratio * 2)
        test_res_dict[name]["ssim"] += ssim(fake_B, real_B, use_sample_covariance=False, sigma=1.5,
                                            gaussian_weights=True,
                                            win_size=11, K1=0.01, K2=0.03, data_range=ratio * 2)
        test_res_dict[name]["num"] += 1

        torch.cuda.empty_cache()

    for t in test_res_dict.keys():
        test_res_dict[t]["rmse"] /= test_res_dict[t]["num"]
        test_res_dict[t]["psnr"] /= test_res_dict[t]["num"]
        test_res_dict[t]["ssim"] /= test_res_dict[t]["num"]
    print(test_res_dict)
    # ave_organ_dice = np.zeros((2, NUM_CLASS))
    #
    # with open('out/' + args.log_name + f'/test_{args.epoch}.txt', 'w') as f:
    #     for key in TEMPLATE.keys():
    #         organ_list = TEMPLATE[key]
    #         content = 'Task%s| ' % (key)
    #         for organ in organ_list:
    #             dice = dice_list[key][0][organ - 1] / dice_list[key][1][organ - 1]
    #             content += '%s: %.4f, ' % (ORGAN_NAME[organ - 1], dice)
    #             ave_organ_dice[0][organ - 1] += dice_list[key][0][organ - 1]
    #             ave_organ_dice[1][organ - 1] += dice_list[key][1][organ - 1]
    #         print(content)
    #         f.write(content)
    #         f.write('\n')
    #     content = 'Average | '
    #     for i in range(NUM_CLASS):
    #         content += '%s: %.4f, ' % (ORGAN_NAME[i], ave_organ_dice[0][i] / ave_organ_dice[1][i])
    #     print(content)
    #     f.write(content)
    #     f.write('\n')
    #     print(np.mean(ave_organ_dice[0] / ave_organ_dice[1]))
    #     f.write('%s: %.4f, ' % ('average', np.mean(ave_organ_dice[0] / ave_organ_dice[1])))
    #     f.write('\n')

    # np.save(save_dir + '/result.npy', dice_list)
    # load
    # dice_list = np.load(/out/epoch_xxx/result.npy, allow_pickle=True)


def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='Nvidia/old_fold0', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--backbone', default='dinov2', help='backbone [swinunetr or unet or dints or unetpp]')
    parser.add_argument('--resume', default='./out/Nvidia/old_fold0/aepoch_500.pth',
                        help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default='./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt',
                        help='The path of pretrain model')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')

    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_123457891213', 'PAOT_10_inner'])  # 'PAOT', 'felix'
    ### please check this argment carefully
    ### PAOT: include PAOT_123457891213 and PAOT_10
    ### PAOT_123457891213: include 1 2 3 4 5 7 8 9 12 13
    ### PAOT_10_inner
    parser.add_argument('--data_root_path', default='/home/jliu288/data/whole_organ/', help='data root path')
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
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
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')

    parser.add_argument('--phase', default='test', help='train or validation or test')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')
    parser.add_argument('--cache_rate', default=0.6, type=float, help='The percentage of cached data in total')

    parser.add_argument('--threshold_organ', default='Pancreas Tumor')
    parser.add_argument('--threshold', default=0.6, type=float)

    args = parser.parse_args()

    # prepare the 3D model
    model = Universal_model(
        in_channels=1,
        out_channels=NUM_CLASS,
        backbone=args.backbone,
        encoding='word_embedding'
    )

    # Load pre-trained weights
    store_dict = model.state_dict()
    checkpoint = torch.load(args.resume)
    load_dict = checkpoint['net']
    # args.epoch = checkpoint['epoch']

    for key, value in load_dict.items():
        name = '.'.join(key.split('.')[1:])
        store_dict[name] = value

    model.load_state_dict(store_dict)
    print('Use pretrained weights')

    model.cuda()

    torch.backends.cudnn.benchmark = True

    test_dataset = ImageABDataset(args)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=int(args.batch_size),
                             shuffle=True,
                             num_workers=args.num_workers,
                             )
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    validation(model, test_loader, args)


if __name__ == "__main__":
    main()
