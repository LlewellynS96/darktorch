import torch
from torch import nn
import torch.nn.functional as F


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        pass


class YOLOLayer(nn.Module):
    def __init__(self, parent, anchors):
        super(YOLOLayer, self).__init__()
        self.device = parent.device
        self.anchors = torch.tensor(anchors, requires_grad=False, device=self.device)
        self.num_anchors = len(anchors)
        self.num_features = parent.num_features

    def forward(self, x):
        x = x.permute(0, 3, 2, 1)
        in_shape = x.shape
        x = x.contiguous().view(-1, self.num_features)

        # Convert t_x and t_y --> x and y (ignoring the offset).
        x[:, :2] = torch.sigmoid(x[:, :2])
        # Add the offset.
        offsets = torch.arange(0, int(x.shape[0] / in_shape[0]), requires_grad=False, device=self.device)
        h_offsets = offsets / in_shape[2] / self.num_anchors
        v_offsets = (offsets - (h_offsets * in_shape[1] * self.num_anchors)) / self.num_anchors
        h_offsets = h_offsets.repeat(in_shape[0])
        v_offsets = v_offsets.repeat(in_shape[0])
        x[:, 0] += h_offsets.float()
        x[:, 1] += v_offsets.float()

        # Convert t_w and t_h --> w and h.
        anchors = self.anchors.repeat(in_shape[0] * in_shape[2] * in_shape[1], 1)

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
            self.beta = beta * torch.ones(1, requires_grad=True)
        else:
            self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.data.size()
        ws = self.stride
        hs = self.stride
        x = x.view(B, C, int(H / hs), hs, int(W / ws), ws).transpose(3, 4).contiguous()
        x = x.view(B, C, int(H / hs * W / ws), hs * ws).transpose(2, 3).contiguous()
        x = x.view(B, C, hs * ws, int(H / hs), int(W / ws)).transpose(1, 2).contiguous()
        x = x.view(B, hs * ws * C, int(H / hs), int(W / ws))
        return x


class RouteLayer(nn.Module):
    def __init__(self, index, first, second, cache):
        super(RouteLayer, self).__init__()
        self.index = index
        self.cache = cache
        self.first = first
        self.second = second

    def forward(self, x):
        if self.second == 0:
            x = self.cache[self.index + self.first]
        else:
            x = torch.cat((self.cache[self.index + self.first], self.cache[self.index + self.second]), 1)
        return x


class ShortcutLayer(nn.Module):
    def __init__(self, index, source, cache):
        super(ShortcutLayer, self).__init__()
        self.index = index
        self.cache = cache
        self.source = source

    def forward(self, x):
        x = x + self.cache[self.index + self.source]
        return x


class EmptyModule(nn.Module):
    def __init__(self):
        super(EmptyModule, self).__init__()

    def forward(self, x):
        return x
