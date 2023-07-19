import torch.nn as nn
import torch
import torch.nn.parallel
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetEncoder, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns   = nn.ModuleList()
        last_channel   = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, x):
        x = x.transpose(2, 1)  # [B, 3, N]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            x  = F.relu(bn(conv(x)))  # [B, D, N]

        x = torch.max(x, 2, keepdim=False)[0]  # [B, D]

        return x


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


class PointNetPose(nn.Module):
    def __init__(self):
        super(PointNetPose, self).__init__()
        self.encoder = PointNetEncoder(3,    [64, 64, 64, 128, 1024])
        self.decoder = PointNetDecoder(1024, [1024, 1024, 1024, 1024])
        self.fct     = nn.Linear(1024, 3)
        self.fcq     = nn.Linear(1024, 3)

    def forward(self, xyz):
        x = self.encoder(xyz)
        y = self.decoder(x)
        t = self.fct(y)  
        q = self.fcq(y)   

        return t, q


if __name__ == '__main__':
    model = PointNetPose()