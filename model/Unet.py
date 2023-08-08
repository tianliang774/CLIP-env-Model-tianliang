import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, img_size, act):
        super(LUConv, self).__init__()
        self.layer_norm = nn.LayerNorm([out_chan, img_size, img_size], elementwise_affine=False)
        self.conv1 = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)
        # self.bn1 = ContBatchNorm2d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x_and_args):
        x, args = x_and_args[0], x_and_args[1]
        x = self.layer_norm(self.conv1(x))
        _lambda = args.epoch / args.max_epoch
        out = (1 - _lambda) * self.activation(x) + _lambda * x
        return out


def _make_nConv(in_channel, depth, act, double_channel=False):
    img_size = 512 // (2 ** depth)
    if double_channel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth + 1)), img_size, act)
        layer2 = LUConv(32 * (2 ** (depth + 1)), 32 * (2 ** (depth + 1)), img_size, act)
    else:
        layer1 = LUConv(in_channel, 32 * (2 ** depth), img_size, act)
        layer2 = LUConv(32 * (2 ** depth), 32 * (2 ** depth) * 2, img_size, act)

    return layer1, layer2


class DownTransition(nn.Module):
    def __init__(self, in_channel, depth, act):
        super(DownTransition, self).__init__()
        self.ops1, self.ops2 = _make_nConv(in_channel, depth, act)
        self.maxpool = nn.Conv2d(64 * (2 ** depth), 64 * (2 ** depth), kernel_size=3, padding=1, stride=2)
        # self.maxpool = nn.MaxPool2d(2)
        self.current_depth = depth

    def forward(self, x, args):
        if self.current_depth == 3:
            out = self.ops1((x, args))
            out = self.ops2((out, args))
            out_before_pool = out
        else:
            out = self.ops1((x, args))
            out_before_pool = self.ops2((out, args))
            out = self.maxpool(out_before_pool)
        return out, out_before_pool


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth, act):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose2d(inChans, outChans, kernel_size=2, stride=2)
        self.ops1, self.ops2 = _make_nConv(inChans + outChans // 2, depth, act, double_channel=True)

    def forward(self, x, skip_x, args):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv, skip_x), 1)
        out = self.ops1((concat, args))
        out = self.ops2((out, args))
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):
        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv2d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.final_conv(x))
        return out


class UNet2D(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, n_class=1, act='relu'):
        super(UNet2D, self).__init__()
        self.down_tr64 = DownTransition(1, 0, act)
        self.down_tr128 = DownTransition(64, 1, act)
        self.down_tr256 = DownTransition(128, 2, act)
        self.down_tr512 = DownTransition(256, 3, act)

        self.up_tr256 = UpTransition(512, 512, 2, act)
        self.up_tr128 = UpTransition(256, 256, 1, act)
        self.up_tr64 = UpTransition(128, 128, 0, act)
        # self.out_tr = OutputTransition(64, n_class)

    def forward(self, x, args):
        self.out64, self.skip_out64 = self.down_tr64(x, args)
        self.out128, self.skip_out128 = self.down_tr128(self.out64, args)
        self.out256, self.skip_out256 = self.down_tr256(self.out128, args)
        self.out512, self.skip_out512 = self.down_tr512(self.out256, args)

        self.out_up_256 = self.up_tr256(self.out512, self.skip_out256, args)
        self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128, args)
        self.out_up_64 = self.up_tr64(self.out_up_128, self.skip_out64, args)
        # self.out = self.out_tr(self.out_up_64)

        return self.out512, self.out_up_64
