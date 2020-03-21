import torch
from torch import nn
import torch.nn.functional as F


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        pass


class YOLOv2Layer(nn.Module):
    def __init__(self, parent, anchors):
        super(YOLOv2Layer, self).__init__()
        self.device = parent.device
        self.anchors = torch.tensor(anchors, device=self.device)
        self.num_anchors = len(anchors)
        self.grid_size = parent.grid_size
        self.num_features = parent.num_features

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        in_shape = x.shape
        x = x.contiguous().view(-1, self.num_features)

        # Convert t_x and t_y --> x and y (ignoring the offset).
        x[:, :2] = torch.sigmoid(x[:, :2])
        # Add the offset.
        offsets = torch.arange(0, int(x.shape[0] / in_shape[0]), device=self.device)
        h_offsets = offsets / self.grid_size[0] / self.num_anchors
        v_offsets = (offsets - (h_offsets * self.grid_size[0] * self.num_anchors)) / self.num_anchors
        h_offsets = h_offsets.repeat(in_shape[0])
        v_offsets = v_offsets.repeat(in_shape[0])
        x[:, 0] += h_offsets.float()
        x[:, 1] += v_offsets.float()

        # Convert t_w and t_h --> w and h.
        anchors = self.anchors.repeat(in_shape[0] * self.grid_size[0] * self.grid_size[1], 1)

        x[:, 2] = anchors[:, 0] * torch.exp(x[:, 2])
        x[:, 3] = anchors[:, 1] * torch.exp(x[:, 3])

        # Convert t_o --> IoU and get class probabilities.
        x[:, 4] = torch.sigmoid(x[:, 4])

        x = x.contiguous().view(*in_shape)
        x = x.permute(0, 3, 2, 1)

        return x


class Swish(nn.Module):

    def __init__(self, beta=1.0, learnable=True):
        super(Swish, self).__init__()

        # Parameters
        if learnable:
            self.beta = nn.Parameter(beta * torch.ones(1))
            self.beta.requires_grad = True
        else:
            self.beta = beta
            self.beta.requires_grad = False

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, x, target):
        """
        input: [N, C], float32
        target: [N, ], int64
        """
        logpt = F.log_softmax(x, dim=-1)
        pt = torch.exp(logpt)
        logpt = (1 - pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, reduction=self.reduction)

        return loss
