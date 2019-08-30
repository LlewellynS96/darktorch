import torch
from torch import nn


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
        x = x.permute(0, 2, 3, 1)
        in_shape = x.shape
        x = x.contiguous().view(-1, self.num_features)

        # Convert t_o --> IoU
        x[:, 0] = torch.sigmoid(x[:, 0])

        # Convert t_x and t_y --> x and y (ignoring the offset).
        x[:, 1:3] = torch.sigmoid(x[:, 1:3])
        # Add the offset.
        offsets = torch.arange(0, int(x.shape[0] / in_shape[0]), device=self.device)
        h_offsets = offsets / self.grid_size[0] / self.num_anchors
        v_offsets = (offsets - (h_offsets * self.grid_size[0] * self.num_anchors)) / self.num_anchors
        h_offsets = h_offsets.repeat(in_shape[0])
        v_offsets = v_offsets.repeat(in_shape[0])
        x[:, 1] += h_offsets.float()
        x[:, 2] += v_offsets.float()
        x[:, 1] /= self.grid_size[0]
        x[:, 2] /= self.grid_size[1]

        # Convert t_w and t_h --> w and h.
        anchors = self.anchors.repeat(in_shape[0] * self.grid_size[0] * self.grid_size[1], 1)

        x[:, 3] = anchors[:, 0] * torch.exp(x[:, 3])
        x[:, 4] = anchors[:, 1] * torch.exp(x[:, 4])
        # Add softmax to class probabilities.
        # NB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        x[:, 5:] = torch.sigmoid(x[:, 5:])

        x = x.contiguous().view(*in_shape)
        x = x.permute(0, 3, 1, 2)

        return x


class YOLOv3Layer(nn.Module):
    def __init__(self, parent, anchors):
        super(YOLOv3Layer, self).__init__()
        self.device = parent.device
        self.anchors = torch.tensor(anchors, device=self.device)
        self.num_anchors = len(anchors)
        self.grid_size = parent.get_grid_size()
        self.num_features = parent.num_features

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        in_shape = x.shape
        x = x.contiguous().view(-1, self.num_features)

        # Convert t_o --> IoU
        x[:, 0] = torch.sigmoid(x[:, 0])

        # Convert t_x and t_y --> x and y (ignoring the offset).
        x[:, 1:3] = torch.sigmoid(x[:, 1:3])
        # Add the offset.
        offsets = torch.arange(0, int(x.shape[0] / in_shape[0]), device=self.device)
        h_offsets = offsets / self.grid_size[0] / self.num_anchors
        v_offsets = (offsets - (h_offsets * self.grid_size[0] * self.num_anchors)) / self.num_anchors
        h_offsets = h_offsets.repeat(in_shape[0])
        v_offsets = v_offsets.repeat(in_shape[0])
        x[:, 1] += h_offsets.float()
        x[:, 2] += v_offsets.float()
        x[:, 1] /= self.grid_size[0]
        x[:, 2] /= self.grid_size[1]

        # Convert t_w and t_h --> w and h.
        anchors = self.anchors.repeat(in_shape[0] * self.grid_size[0] * self.grid_size[1], 1)

        x[:, 3] = anchors[:, 0] * torch.exp(x[:, 3])
        x[:, 4] = anchors[:, 1] * torch.exp(x[:, 4])
        # Add softmax to class probabilities.
        # NB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        x[:, 5:] = torch.sigmoid(x[:, 5:])

        x = x.contiguous().view(*in_shape)
        x = x.permute(0, 3, 1, 2)

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
        return x * nn.Sigmoid(self.beta * x)