import torch
import torch.nn.functional as F
from torch import nn


class CriterionPose(nn.Module):
    def __init__(self):
        super(CriterionPose, self).__init__()
        self.t_loss_fn = nn.L1Loss()
        self.q_loss_fn = nn.L1Loss()

    def forward(self, pred_t, pred_q, gt_t, gt_q):
        loss = 1 * self.t_loss_fn(pred_t, gt_t) + 3 * self.q_loss_fn(pred_q, gt_q) 

        return loss


class CriterionlrPose(nn.Module):
    def __init__(self, sat=3.0, saq=3.0, learn_gamma=True):
        super(CriterionlrPose, self).__init__()
        self.t_loss_fn = nn.L1Loss()
        self.q_loss_fn = nn.L1Loss()
        self.sat = nn.Parameter(torch.tensor([sat], requires_grad=learn_gamma, device='cuda:0'))
        self.saq = nn.Parameter(torch.tensor([saq], requires_grad=learn_gamma, device='cuda:0'))

    def forward(self, pred_t, pred_q, gt_t, gt_q):
        loss = torch.exp(-self.sat) * self.t_loss_fn(pred_t, gt_t) + self.sat + torch.exp(-self.saq) * self.q_loss_fn(pred_q, gt_q) + self.saq
        
        return loss