import torch
import numpy as np
import torch.nn as nn
from option import args
import torch.nn.functional as F


class make_dilation_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dilation_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2 + 1,
                              bias=True, dilation=2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Dilation Residual dense block (DRDB)
class DRDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(DRDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dilation_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, bias=True):
        super(ConvLayer, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias
        self.conv = nn.Sequential(
            nn.Conv2d(self.in_ch, self.out_ch, kernel_size=self.kernel_size, padding=self.padding, bias=self.bias),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class AHDRNet(nn.Module):
    def __init__(self):
        super(AHDRNet, self).__init__()
        self.num_channels = args.num_channels
        self.num_feats = args.num_features
        self.num_layers = args.num_layers
        self.growth = args.growth

        # Attention Network
        self.conv_1 = ConvLayer(2 * self.num_channels, self.num_feats)
        self.att_up = nn.Sequential(
            ConvLayer(2 * self.num_feats, 2 * self.num_feats),
            nn.Conv2d(2 * self.num_feats, self.num_feats, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )
        self.att_down = nn.Sequential(
            ConvLayer(2 * self.num_feats, 2 * self.num_feats),
            nn.Conv2d(2 * self.num_feats, self.num_feats, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid()
        )

        # Merging Network
        self.conv_att = ConvLayer(3 * self.num_feats, self.num_feats)
        self.RDB_1 = DRDB(self.num_feats, self.num_layers, self.growth)
        self.RDB_2 = DRDB(self.num_feats, self.num_layers, self.growth)
        self.RDB_3 = DRDB(self.num_feats, self.num_layers, self.growth)
        self.conv_merge = ConvLayer(3 * self.num_feats, self.num_feats, kernel_size=1, padding=0)
        self.conv_2 = ConvLayer(self.num_feats, self.num_feats)
        self.conv_3 = ConvLayer(self.num_feats, self.num_feats)
        self.conv_out = ConvLayer(self.num_feats, self.num_channels)

    def forward(self, x1, x2, x3):
        x1 = self.conv_1(x1)
        x2 = self.conv_1(x2)
        x3 = self.conv_1(x3)

        feat_up = torch.cat((x1, x2), dim=1)
        feat_up_w = self.att_up(feat_up)
        feat_up = x1 * feat_up_w

        feat_down = torch.cat((x3, x2), dim=1)
        feat_down_w = self.att_down(feat_down)
        feat_down = x3 * feat_down_w

        feat_cat = torch.cat((feat_up, x2 ,feat_down), dim=1)
        feat_cat = self.conv_att(feat_cat)
        feat_1 = self.RDB_1(feat_cat)
        feat_2 = self.RDB_2(feat_1)
        feat_3 = self.RDB_3(feat_2)
        feat_drdb = torch.cat((feat_1, feat_2, feat_3), dim=1)
        feat_drdb = self.conv_merge(feat_drdb)
        feat_drdb = self.conv_2(feat_drdb)
        feat_res = feat_drdb + x2
        feat_res = self.conv_3(feat_res)

        output = self.conv_out(feat_res)

        return output


if __name__ == '__main__':
    a = torch.ones([1, 6, 64, 64])
    b = torch.ones([1, 6, 64, 64])
    c = torch.ones([1, 6, 64, 64])
    net = AHDRNet()
    output = net(a, b, c)
    print(output.shape)


