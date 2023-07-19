import sys
sys.path.insert(0,'..')
import math
import open3d
import torch
import torch.nn as nn
import torch.nn.functional as F
import point
import numpy as np
import time
import torch.optim as optim
from models.net_utils import *


def conv_bn(inp, oup, kernel, stride=1, activation='relu'):
    seq = nn.Sequential(
        nn.Conv2d(inp, oup, kernel, stride),
        nn.BatchNorm2d(oup)
    )
    if activation == 'relu':
        seq.add_module('2', nn.ReLU())
    return seq

def conv1d_bn(inp, oup, kernel, stride=1, activation='relu'):
    seq = nn.Sequential(
        nn.Conv1d(inp, oup, kernel, stride),
        nn.BatchNorm1d(oup)
    )
    if activation == 'relu':
        seq.add_module('2', nn.ReLU())
    return seq


def fc_bn(inp, oup):
    return nn.Sequential(
        nn.Linear(inp, oup),
        nn.BatchNorm1d(oup),
        nn.ReLU()
)


class PointSIFT_module_basic(nn.Module):
    def __init__(self):
        super(PointSIFT_module_basic, self).__init__()


    def group_points(self,xyz,idx):
        b , n , c = xyz.shape
        m = idx.shape[1]
        nsample = idx.shape[2]
        out = torch.zeros((xyz.shape[0],xyz.shape[1], 8,c)).cuda()
        point.group_points(b,n,c,m,nsample,xyz,idx.int(),out)
        return out

    def pointsift_select(self, radius, xyz):
        y = torch.zeros((xyz.shape[0],xyz.shape[1], 8), dtype=torch.int32).cuda()
        point.select_cube(xyz,y,xyz.shape[0],xyz.shape[1],radius)
        return y.long()

    def pointsift_group(self, radius, xyz, points, use_xyz=True):

        B, N, C = xyz.shape
        assert C == 3
        # start_time = time.time()
        idx = self.pointsift_select(radius, xyz)  # B, N, 8
        # print("select SIR 1 ", time.time() - start_time, xyz.shape)

        # start_time = time.time()
        grouped_xyz = self.group_points(xyz, idx)  # B, N, 8, 3
        # print("group SIR SIR 1 ", time.time() - start_time)

        grouped_xyz -= xyz.view(B, N, 1, 3)
        if points is not None:
            grouped_points = self.group_points(points, idx)
            if use_xyz:
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz
        return grouped_xyz, grouped_points, idx

    def pointsift_group_with_idx(self, idx, xyz, points, use_xyz=True):

        B, N, C = xyz.shape
        grouped_xyz = self.group_points(xyz, idx)  # B, N, 8, 3
        grouped_xyz -= xyz.view(B, N, 1, 3)
        if points is not None:
            grouped_points = self.group_points(points, idx)
            if use_xyz:
                grouped_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz
        return grouped_xyz, grouped_points


class PointSIFT_res_module(PointSIFT_module_basic):

    def __init__(self, radius, output_channel, extra_input_channel=0, merge='add', same_dim=False):
        super(PointSIFT_res_module, self).__init__()
        self.radius = radius
        self.merge = merge
        self.same_dim = same_dim

        self.conv1 = nn.Sequential(
            conv_bn(3 + extra_input_channel, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2])
        )

        self.conv2 = nn.Sequential(
            conv_bn(3 + output_channel, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2], activation=None)
        )
        if same_dim:
            self.convt = nn.Sequential(
                nn.Conv1d(extra_input_channel, output_channel, 1),
                nn.BatchNorm1d(output_channel),
                nn.ReLU()
            )

    def forward(self, xyz, points):
        _, grouped_points, idx = self.pointsift_group(self.radius, xyz, points)  # [B, N, 8, 3], [B, N, 8, 3 + C]

        grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()  # B, C, N, 8
        ##print(grouped_points.shape)
        new_points = self.conv1(grouped_points)
        ##print(new_points.shape)
        new_points = new_points.squeeze(-1).permute(0, 2, 1).contiguous()

        _, grouped_points = self.pointsift_group_with_idx(idx, xyz, new_points)
        grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()

        ##print(grouped_points.shape)
        new_points = self.conv2(grouped_points)

        new_points = new_points.squeeze(-1)

        if points is not None:
            points = points.permute(0, 2, 1).contiguous()
            # print(points.shape)
            if self.same_dim:
                points = self.convt(points)
            if self.merge == 'add':
                new_points = new_points + points
            elif self.merge == 'concat':
                new_points = torch.cat([new_points, points], dim=1)

        new_points = F.relu(new_points)
        # new_points = new_points.permute(0, 2, 1).contiguous()

        return xyz, new_points


class PointSIFT_module(PointSIFT_module_basic):

    def __init__(self, radius, output_channel, extra_input_channel=0, merge='add', same_dim=False):
        super(PointSIFT_module, self).__init__()
        self.radius = radius
        self.merge = merge
        self.same_dim = same_dim

        self.conv1 = nn.Sequential(
            conv_bn(3+extra_input_channel, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2]),
            conv_bn(output_channel, output_channel, [1, 2], [1, 2])
        )

        # self.conv1 = nn.Sequential(
        #     conv_bn(3+extra_input_channel, output_channel, [1, 2], [1, 2]),
        #     conv_bn(output_channel, output_channel, [1, 2], [1, 2])
        # )

        # self.conv1 = nn.Sequential(
        #     conv_bn(3+extra_input_channel, output_channel, [1, 2], [1, 2])
        # )

        # self.conv2 = conv_bn(output_channel, output_channel, [1, 1], [1, 1])


    def forward(self, xyz, points):
        _, grouped_points, idx = self.pointsift_group(self.radius, xyz, points)  # [B, N, 8, 3], [B, N, 8, 3 + C]

        grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()  # B, C, N, 8
        ##print(grouped_points.shape)
        new_points = self.conv1(grouped_points)
        # new_points = self.conv2(new_points)

        new_points = new_points.squeeze(-1)
        new_points = new_points.permute(0, 2, 1)

        return xyz, new_points


    @staticmethod
    def get_loss(input, target):
        classify_loss = nn.CrossEntropyLoss()
        loss = classify_loss(input, target)
        return loss

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()