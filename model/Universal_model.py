from typing import Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from model.Unet import UNet2D
from model.Dinov2 import DinoV2_Generator


class Universal_model(nn.Module):
    def __init__(self, img_size, in_channels, out_channels, backbone='unet', encoding='rand_embedding'):
        # encoding: rand_embedding or word_embedding
        # out_channels: NUM_CLASS
        super().__init__()
        self.backbone_name = backbone
        if backbone == 'unet':
            self.backbone = UNet2D()
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 512),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
            )
        elif backbone == 'dinov2':
            self.backbone = DinoV2_Generator(64)
            self.precls_conv = nn.Sequential(
                nn.GroupNorm(16, 64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 8, kernel_size=1)
            )
            self.GAP = nn.Sequential(
                nn.GroupNorm(16, 384),
                nn.ReLU(inplace=True),
                torch.nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0)
            )
        else:
            raise Exception('{} backbone is not implemented in current version'.format(backbone))

        self.encoding = encoding
        self.out_channels = out_channels

        weight_nums, bias_nums = [], []
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 8)
        weight_nums.append(8 * 1)
        bias_nums.append(8)
        bias_nums.append(8)
        bias_nums.append(1)
        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.controller = nn.Conv2d(256 + 256, sum(weight_nums + bias_nums), kernel_size=1, stride=1, padding=0)
        if self.encoding == 'rand_embedding':
            self.organ_embedding = nn.Embedding(out_channels, 256)
        elif self.encoding == 'word_embedding':
            # It is initialization and it will be replaced by txt_encoding.pth by default
            self.register_buffer('organ_embedding', torch.randn(out_channels, 512))
            self.text_to_vision = nn.Linear(512, 256)
        self.class_num = out_channels

    def load_params(self, model_dict):
        if self.backbone_name == 'swinunetr':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out' not in key:
                    store_dict[key] = model_dict[key]

            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')
        elif self.backbone_name == 'unet':
            store_dict = self.backbone.state_dict()
            for key in model_dict.keys():
                if 'out_tr' not in key:
                    store_dict[key.replace("module.", "")] = model_dict[key]
            self.backbone.load_state_dict(store_dict)
            print('Use pretrained weights')

    def encoding_task(self, task_id):
        N = task_id.shape[0]
        task_encoding = torch.zeros(size=(N, 7))
        for i in range(N):
            task_encoding[i, task_id[i]] = 1
        return task_encoding.cuda()

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                weight_splits[l] = weight_splits[l].reshape(num_insts*channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts*channels)
            else:
                weight_splits[l] = weight_splits[l].reshape(num_insts*1, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts*1)
            # print(weight_splits[l].shape, bias_splits[l].shape)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            # print(i, x.shape, w.shape)
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward(self, x_in):
        dec4, out = self.backbone(x_in)
        # dec4.shape: torch.Size([1, 512, 64, 64])
        # out.shape: torch.Size([1, 64, 512, 512])
        if self.encoding == 'rand_embedding':
            task_encoding = self.organ_embedding.weight
        elif self.encoding == 'word_embedding':
            task_encoding = F.relu(self.text_to_vision(self.organ_embedding))
        # task_encoding torch.Size([32, 256])
        x_feat = self.GAP(dec4).squeeze_(-1).squeeze_(-1)  # x_feat.shape torch.Size([1, 256])
        b = x_feat.shape[0]  # b = batch
        logits_array = []
        for i in range(b):
            x_cond = torch.cat([x_feat[i].unsqueeze(0).repeat(self.out_channels, 1), task_encoding], 1)
            # x_cond: torch.Size([out_channels, 512]) //out_channels = 32
            x_cond = x_cond.unsqueeze(-1).unsqueeze(-1)
            params = self.controller(x_cond)
            params.squeeze_(-1).squeeze_(-1)  # params.shape: torch.Size([32, 153]) 153 = sum(weight_nums + bias_nums)
            head_inputs = self.precls_conv(out[i].unsqueeze(0))  # --> 8 channel layer
            head_inputs = head_inputs.repeat(self.out_channels, 1, 1, 1)
            N, _, H, W = head_inputs.size()
            head_inputs = head_inputs.reshape(1, -1, H, W)  # head_inputs: [1,class_num*8,512,512]
            weights, biases = self.parse_dynamic_params(params, 8, self.weight_nums, self.bias_nums)
            logits = self.heads_forward(head_inputs, weights, biases, N)
            logits_array.append(logits)
        out = torch.cat(logits_array, dim=0)
        return out
