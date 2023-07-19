import torch.nn as nn
import torch
import torch.nn.parallel
import torch.nn.functional as F
from utils.pointnet_util import sample_and_group, PointNetSetAbstraction
# from utils.network_util import PointSIFT_module_first, PointSIFT_module, NetVLADLoupe
from models.sift import PointSIFT_module, PointSIFT_res_module


class PointSIFTEncoder(nn.Module):
    def __init__(self):
        super(PointSIFTEncoder, self).__init__()
        # oxford
        self.ps1 = PointSIFT_module(4, 64)
        self.sa1 = PointNetSetAbstraction(1024, 4, 32, 64 + 3, [64], False)
        self.ps2 = PointSIFT_module(6, 64, 64)
        self.sa2 = PointNetSetAbstraction(256, 6, 16, 64 + 3, [128], False)
        self.ps3 = PointSIFT_module(8, 128, 128)
        self.sa3 = PointNetSetAbstraction(64, 8, 8, 128 + 3, [256], False)
        self.ps4 = PointSIFT_module(10, 256, 256)
        self.sa4 = PointNetSetAbstraction(None, None, None, 256 + 3, [1024], True)

    def forward(self, xyz, points=None):
        B                       = xyz.size(0)
        l1_xyz_ps, l1_points_ps = self.ps1(xyz, points)
        l1_xyz_sa, l1_points_sa = self.sa1(l1_xyz_ps, l1_points_ps)
        l2_xyz_ps, l2_points_ps = self.ps2(l1_xyz_sa, l1_points_sa)
        l2_xyz_sa, l2_points_sa = self.sa2(l2_xyz_ps, l2_points_ps)
        l3_xyz_ps, l3_points_ps = self.ps3(l2_xyz_sa, l2_points_sa)
        l3_xyz_sa, l3_points_sa = self.sa3(l3_xyz_ps, l3_points_ps)
        l4_xyz_ps, l4_points_ps = self.ps4(l3_xyz_sa, l3_points_sa)
        l4_xyz_sa, l4_points_sa = self.sa4(l4_xyz_ps, l4_points_ps)
        l4_points_sa            = l4_points_sa.view(B, -1)

        return l4_points_sa


class PointNetDecoder(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetDecoder, self).__init__()
        self.mlp_fcs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_fcs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, x):
        for i, fc in enumerate(self.mlp_fcs):
            bn = self.mlp_bns[i]
            x  = F.relu(bn(fc(x)))  # [B, D]
        
        return x



class PoseSOE(nn.Module):
    def __init__(self):
        super(PoseSOE, self).__init__()
        self.encoder = PointSIFTEncoder()
        self.decoder = PointNetDecoder(1024, [1024, 1024, 1024, 1024])
        self.fct     = nn.Linear(1024, 3)
        self.fcq     = nn.Linear(1024, 3)

    def forward(self, xyz):
        x = self.encoder(xyz)
        y = self.decoder(x)
        t = self.fct(y)  
        q = self.fcq(y)   

        return t, q