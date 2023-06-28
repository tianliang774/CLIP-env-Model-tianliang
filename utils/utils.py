import os, sys
# import cc3d
# import fastremap
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
# from pyod.models.knn import KNN
from math import ceil
from scipy.ndimage.filters import gaussian_filter
import warnings
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Union
from scipy import ndimage


NUM_CLASS = 5

TEMPLATE = {
    '4DCBCT': 3,
    'CT': 0,
    'ring': 1,
    'low2high': 2,
    'ct2mri': 4
}


def plot_anomalies(df, x='slice_index', y='array_sum', save_dir=None):
    # categories will be having values from 0 to n
    # for each values in 0 to n it is mapped in colormap
    categories = df['Predictions'].to_numpy()
    colormap = np.array(['g', 'r'])

    f = plt.figure(figsize=(12, 4))
    f = plt.plot(df[x], df['SMA20'], 'b')
    f = plt.plot(df[x], df['upper_bound'], 'y')
    f = plt.scatter(df[x], df[y], c=colormap[categories], alpha=0.3)
    f = plt.xlabel(x)
    f = plt.ylabel(y)
    plt.legend(['Simple moving average', 'upper bound', 'predictions'])
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.clf()



def PSVein_post_process(PSVein_mask, pancreas_mask):
    xy_sum_pancreas = pancreas_mask.sum(axis=0).sum(axis=0)
    z_non_zero = np.nonzero(xy_sum_pancreas)
    z_value = np.min(z_non_zero)  ## the down side of pancreas
    new_PSVein = PSVein_mask.copy()
    new_PSVein[:, :, :z_value] = 0
    return new_PSVein





def dice_score(preds, labels, spe_sen=False):  # on GPU
    ### preds: w,h,d; label: w,h,d
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    preds = torch.where(preds > 0.5, 1., 0.)
    predict = preds.contiguous().view(1, -1)
    target = labels.contiguous().view(1, -1)

    tp = torch.sum(torch.mul(predict, target))
    fn = torch.sum(torch.mul(predict != 1, target))
    fp = torch.sum(torch.mul(predict, target != 1))
    tn = torch.sum(torch.mul(predict != 1, target != 1))

    den = torch.sum(predict) + torch.sum(target) + 1

    dice = 2 * tp / den
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (fp + tn)

    # print(dice, recall, precision)
    if spe_sen:
        return dice, recall, precision, specificity
    else:
        return dice, recall, precision


def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map


def multi_net(net_list, img, task_id):
    # img = torch.from_numpy(img).cuda()

    padded_prediction = net_list[0](img, task_id)
    padded_prediction = F.sigmoid(padded_prediction)
    for i in range(1, len(net_list)):
        padded_prediction_i = net_list[i](img, task_id)
        padded_prediction_i = F.sigmoid(padded_prediction_i)
        padded_prediction += padded_prediction_i
    padded_prediction /= len(net_list)
    return padded_prediction  # .cpu().data.numpy()


def check_data(dataset_check):
    img = dataset_check[0]["image"]
    label = dataset_check[0]["label"]
    print(dataset_check[0]["name"])
    img_shape = img.shape
    label_shape = label.shape
    print(f"image shape: {img_shape}, label shape: {label_shape}")
    print(torch.unique(label[0, :, :, 150]))
    plt.figure("image", (18, 6))
    plt.subplot(1, 2, 1)
    plt.title("image")
    plt.imshow(img[0, :, :, 150].detach().cpu(), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("label")
    plt.imshow(label[0, :, :, 150].detach().cpu())
    plt.show()


if __name__ == "__main__":
    print("hello")
