import torch
from torch import nn
import torch.nn.functional as F


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        pass


class YOLOv2Layer(nn.Module):
    def __init__(self, parent, anchors, softmax=False):
        super(YOLOv2Layer, self).__init__()
        self.device = parent.device
        self.anchors = torch.tensor(anchors, device=self.device)
        self.num_anchors = len(anchors)
        self.grid_size = parent.grid_size
        self.num_features = parent.num_features
        self.softmax = softmax

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
        x[:, 0] /= self.grid_size[0]
        x[:, 1] /= self.grid_size[1]

        # Convert t_w and t_h --> w and h.
        anchors = self.anchors.repeat(in_shape[0] * self.grid_size[0] * self.grid_size[1], 1)

        x[:, 2] = anchors[:, 0] * torch.exp(x[:, 2])
        x[:, 3] = anchors[:, 1] * torch.exp(x[:, 3])
        # Convert t_o --> IoU and get class probabilities.
        x[:, 4] = torch.sigmoid(x[:, 4])
        if self.softmax:
            x[:, 5:] = torch.softmax(x[:, 5:].contiguous(), dim=1)
        else:
            x[:, 5:] = torch.sigmoid(x[:, 5:])

        x = x.contiguous().view(*in_shape)
        x = x.permute(0, 3, 2, 1)

        return x


class YOLOv3Layer(nn.Module):
    def __init__(self, parent, anchors, softmax=True):
        super(YOLOv3Layer, self).__init__()
        self.device = parent.device
        self.anchors = torch.tensor(anchors, device=self.device)
        self.num_anchors = len(anchors)
        self.grid_size = parent.get_grid_size()
        self.num_features = parent.num_features
        self.softmax = softmax

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        in_shape = x.shape
        x = x.contiguous().view(-1, self.num_features)

        # Convert t_o --> IoU
        x[:, 4] = torch.sigmoid(x[:, 4])

        # Convert t_x and t_y --> x and y (ignoring the offset).
        x[:, :4] = torch.sigmoid(x[:, :4])
        # Add the offset.
        offsets = torch.arange(0, int(x.shape[0] / in_shape[0]), device=self.device)
        h_offsets = offsets / self.grid_size[0] / self.num_anchors
        v_offsets = (offsets - (h_offsets * self.grid_size[0] * self.num_anchors)) / self.num_anchors
        h_offsets = h_offsets.repeat(in_shape[0])
        v_offsets = v_offsets.repeat(in_shape[0])
        x[:, 0] += h_offsets.float()
        x[:, 1] += v_offsets.float()
        x[:, 0] /= self.grid_size[0]
        x[:, 1] /= self.grid_size[1]

        # Convert t_w and t_h --> w and h.
        anchors = self.anchors.repeat(in_shape[0] * self.grid_size[0] * self.grid_size[1], 1)

        x[:, 2] = anchors[:, 0] * torch.exp(x[:, 2])
        x[:, 3] = anchors[:, 1] * torch.exp(x[:, 3])
        if self.softmax:
            x[:, 5:] = torch.softmax(x[:, 5:].contiguous(), dim=1)
        else:
            x[:, 5:] = torch.sigmoid(x[:, 5:])

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
        return x * nn.Sigmoid()(self.beta * x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduction=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        f_loss = self.alpha * (1 - pt)**self.gamma * bce_loss

        if self.reduction is None:
            return f_loss
        elif self.reduction == 'mean':
            return torch.mean(f_loss)
        elif self.reduction == 'sum':
            return torch.sum(f_loss)
        else:
            raise AssertionError
