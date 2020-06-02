import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import NUM_WORKERS
from utils import jaccard, xywh2xyxy, non_maximum_suppression, to_numpy_image, add_bbox_to_image, export_prediction
from layers import *
from time import time

LAMBDA_COORD = 5.
LAMBDA_OBJ = 1.
LAMBDA_CLASS = 1.
LAMBDA_NOOBJ = 1.

USE_CROSS_ENTROPY = False


class YOLOv2tiny(nn.Module):

    def __init__(self, model, name='YOLOv2-tiny', device='cuda'):

        super(YOLOv2tiny, self).__init__()

        self.net_info = {}
        self.blocks = []
        self.layers = nn.ModuleList()
        self.detection_layers = []
        self.name = name
        self.device = device

        self.parse_cfg(model)

        self.channels = int(self.net_info['channels'])
        self.default_image_size = int(self.net_info['width']), int(self.net_info['height'])
        self.image_size = self.default_image_size
        self.num_classes = int(self.blocks[-1]['classes'])
        self.num_features = 5 + self.num_classes

        self.stride = self.calculate_stride()
        self.grid_size = self.calculate_grid_size()

        self.build_modules()

        self.anchors = self.collect_anchors()
        self.num_anchors = len(self.anchors)

        self.lr = float(self.net_info['learning_rate'])
        self.momentum = float(self.net_info['momentum'])
        self.weight_decay = float(self.net_info['decay'])
        self.rescore = True if self.blocks[-1]['rescore'] is '1' else False
        self.noobj_iou_threshold = float(self.blocks[-1]['thresh'])
        self.multi_scale = True if self.blocks[-1]['random'] is '1' else False

        self.steps = list(map(float, self.net_info['steps'].split(',')))
        self.scales = list(map(float, self.net_info['scales'].split(',')))

        self.batch_size = int(self.net_info['batch'])
        self.subdivisions = int(self.net_info['subdivisions'])
        self.max_batches = int(self.net_info['max_batches'])

        assert self.batch_size % self.subdivisions == 0, 'Subdivisions must be factor of batch size.'

        self.subdivision_size = self.batch_size / self.subdivisions

        self.iteration = 0

        self.to(device)

    def set_input_dims(self, dims=3):
        conv = self.layers[0][0]
        assert dims <= conv.weight.data.shape[1]
        self.layers[0][0] = nn.Conv2d(dims, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layers[0][0].weight.data.copy_(conv.weight.data[:, :dims])
        self.to(self.device)
        self.channels = dims

    def forward(self, x):

        assert x.dim() == 4

        self.set_image_size(x.shape[-2:])

        for layer in self.layers:
            x = layer(x)

        return x

    def loss(self, predictions, targets, stats):

        assert predictions.shape == targets.shape

        loss = dict()
        batch_size = targets.shape[0]

        targets = targets.permute(0, 2, 3, 1)
        predictions = predictions.permute(0, 2, 3, 1)

        targets = targets.contiguous().view(batch_size, -1, self.num_features)
        predictions = predictions.contiguous().view(batch_size, -1, self.num_features)

        img_idx = torch.arange(batch_size, dtype=torch.float, device=self.device).reshape(-1, 1) * self.grid_size[0]
        targets[:, :, 0] += 2. * img_idx
        predictions[:, :, 0] += 2. * img_idx
        img_idx = torch.arange(batch_size, dtype=torch.float, device=self.device).reshape(-1, 1) * self.grid_size[1]
        targets[:, :, 1] += 2. * img_idx
        predictions[:, :, 1] += 2. * img_idx

        targets = targets.contiguous().view(-1, self.num_features)
        predictions = predictions.contiguous().view(-1, self.num_features)

        obj_mask = torch.nonzero(targets[:, 4]).flatten()
        num_obj = len(obj_mask)

        if obj_mask.numel() > 0:
            predictions_xyxy = xywh2xyxy(predictions[:, :4].detach())
            targets_xyxy = xywh2xyxy(targets[obj_mask, :4])

            all_ious = jaccard(predictions_xyxy, targets_xyxy)
            ious, _ = torch.max(all_ious, dim=1)
            stats['avg_obj_iou'].append(all_ious[obj_mask].diag().mean().item())

            mask = torch.nonzero(ious > self.noobj_iou_threshold).squeeze()
            targets[mask, 4] = 1.
            noobj_mask = torch.nonzero(targets[:, 4] == 0.).squeeze()

            # anchors = self.anchors[obj_mask % 5]
            loss['coord'] = nn.MSELoss(reduction='sum')(predictions[obj_mask, 0], targets[obj_mask, 0])
            loss['coord'] += nn.MSELoss(reduction='sum')(predictions[obj_mask, 1], targets[obj_mask, 1])
            loss['coord'] += nn.MSELoss(reduction='sum')(torch.sqrt(predictions[obj_mask, 2]),
                                                         torch.sqrt(targets[obj_mask, 2]))
            loss['coord'] += nn.MSELoss(reduction='sum')(torch.sqrt(predictions[obj_mask, 3]),
                                                         torch.sqrt(targets[obj_mask, 3]))
            loss['coord'] *= LAMBDA_COORD / batch_size

            if self.iteration * self.batch_size < 12800:
                # anchors = self.anchors[noobj_mask % 5]
                loss['bias'] = nn.MSELoss(reduction='sum')(predictions[noobj_mask, 0],
                                                           targets[noobj_mask, 0])
                loss['bias'] += nn.MSELoss(reduction='sum')(predictions[noobj_mask, 1],
                                                            targets[noobj_mask, 1])
                loss['bias'] += nn.MSELoss(reduction='sum')(torch.sqrt(predictions[noobj_mask, 2]),
                                                            torch.sqrt(targets[noobj_mask, 2]))
                loss['bias'] += nn.MSELoss(reduction='sum')(torch.sqrt(predictions[noobj_mask, 3]),
                                                            torch.sqrt(targets[noobj_mask, 3]))

                loss['bias'] *= 0.1 / batch_size

            predictions[obj_mask, 5:] = F.log_softmax(predictions[obj_mask, 5:], dim=-1)
            targets_long = torch.argmax(targets[obj_mask, 5:], dim=1)
            if USE_CROSS_ENTROPY:
                loss['class'] = nn.NLLLoss(reduction='sum')(predictions[obj_mask, 5:], targets_long)
            else:
                loss['class'] = nn.MSELoss(reduction='sum')(torch.exp(predictions[obj_mask, 5:]),
                                                            targets[obj_mask, 5:])
            loss['class'] *= LAMBDA_CLASS / batch_size
            stats['avg_class'].append(torch.exp(predictions[obj_mask, 5 + targets_long]).mean().item())

            if self.rescore:
                loss['object'] = nn.MSELoss(reduction='sum')(predictions[obj_mask, 4],
                                                             all_ious[obj_mask, torch.arange(num_obj)].detach())
            else:
                loss['object'] = nn.MSELoss(reduction='sum')(predictions[obj_mask, 4],
                                                             targets[obj_mask, 4])
            loss['object'] *= LAMBDA_OBJ / batch_size
            stats['avg_pobj'].append(predictions[obj_mask, 4].mean().item())

            loss['no_object'] = nn.MSELoss(reduction='sum')(predictions[noobj_mask, 4],
                                                            targets[noobj_mask, 4])
            loss['no_object'] *= LAMBDA_NOOBJ / batch_size
            stats['avg_pnoobj'].append(predictions[noobj_mask, 4].mean().item())
        else:
            loss['object'] = torch.tensor([0.], device=self.device)
            loss['coord'] = torch.tensor([0.], device=self.device)
            loss['class'] = torch.tensor([0.], device=self.device)
            loss['no_object'] = LAMBDA_NOOBJ / batch_size * nn.MSELoss(reduction='sum')(predictions[:, 4],
                                                                                        targets[:, 4])
            if self.iteration * self.batch_size < 12800:
                loss['bias'] = nn.MSELoss(reduction='sum')(predictions[:, 0],
                                                           targets[:, 0])
                loss['bias'] += nn.MSELoss(reduction='sum')(predictions[:, 1],
                                                            targets[:, 1])
                loss['bias'] += nn.MSELoss(reduction='sum')(torch.sqrt(predictions[:, 2]),
                                                            torch.sqrt(targets[:, 2]))
                loss['bias'] += nn.MSELoss(reduction='sum')(torch.sqrt(predictions[:, 3]),
                                                            torch.sqrt(targets[:, 3]))
                loss['bias'] *= 0.1 / batch_size

        loss['total'] = (loss['coord'] + loss['class'] + loss['object'] + loss['no_object'])

        return loss, stats

    def fit(self, train_data, optimizer, scheduler=None, epochs=1,
            val_data=None, checkpoint_frequency=100):

        if scheduler is not None:
            scheduler.last_epoch = self.iteration

        self.train()

        train_dataloader = DataLoader(dataset=train_data,
                                      batch_size=train_data.batch_size,
                                      num_workers=NUM_WORKERS)

        all_train_loss = []
        all_train_stats = []
        all_val_loss = []
        all_val_stats = []

        if val_data is not None:
            self.eval()
            loss, val_stats = self.calculate_loss(val_data, val_data.batch_size, fraction=.5)
            self.train()
            val_data.step()
            all_val_loss.append(loss)
            all_val_stats.append(val_stats)

        # ==========================================================
        # PyCharm does not support nested tqdm progress bars, and
        # thus this feature has been removed.
        # ==========================================================
        # with tqdm(total=epochs,
        #           desc='{} Training PASCAL VOC'.format(self.name),
        #           leave=True,
        #           unit='batches') as outer:
        for epoch in range(1, epochs + 1):
            batch_stats = {'avg_obj_iou': [], 'avg_class': [], 'avg_pobj': [], 'avg_pnoobj': []}
            val_stats = {'avg_obj_iou': [], 'avg_class': [], 'avg_pobj': [], 'avg_pnoobj': []}
            batch_loss = []
            if epoch % checkpoint_frequency == 0 or epoch == epochs:
                train_data.disable_multiscale()
            with tqdm(total=len(train_dataloader),
                      desc='Epoch: [{}/{}]'.format(epoch, epochs),
                      leave=True,
                      unit='batches') as inner:
                for images, _, targets in train_dataloader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    if inner.n % self.subdivisions == 0:
                        optimizer.zero_grad()
                    predictions = self(images)
                    loss, batch_stats = self.loss(predictions, targets, batch_stats)
                    batch_loss.append(loss['total'].item())
                    if self.iteration * self.batch_size < 12800:
                        loss['total'] += loss['bias']
                    loss['total'].backward()
                    weights = np.arange(1, 1 + len(batch_loss))
                    disp_str = ' Training Loss: {:.6f}, '.format(np.average(batch_loss, weights=weights)) + \
                               ' Avg IOU: {:.4f},  '.format(np.average(batch_stats['avg_obj_iou'], weights=weights)) + \
                               ' Avg P|Obj: {:.4f},  '.format(np.average(batch_stats['avg_pobj'], weights=weights)) + \
                               ' Avg P|Noobj: {:.4f},  '.format(np.average(batch_stats['avg_pnoobj'], weights=weights)) + \
                               ' Avg Class: {:.4f}, '.format(np.average(batch_stats['avg_class'], weights=weights)) + \
                               ' Iteration: {:d}'.format(self.iteration)
                    inner.set_postfix_str(disp_str)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1., norm_type='inf')
                    if (inner.n + 1) % self.subdivisions == 0:
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        self.iteration += 1
                    inner.update()
                all_train_loss.append(batch_loss)
                all_train_stats.append(batch_stats)
            train_data.step(self.multi_scale)
            if val_data is not None:
                self.eval()
                loss, val_stats = self.calculate_loss(val_data, val_data.batch_size, val_stats, fraction=.5)
                self.train()
                val_data.step()
                all_val_loss.append(loss)
                all_val_stats.append(val_stats)
                disp_str = ' Training Loss: {:.6f}, '.format(np.average(batch_loss, weights=weights)) + \
                           ' Avg IOU: {:.4f},  '.format(np.average(batch_stats['avg_obj_iou'], weights=weights)) + \
                           ' Avg P|Obj: {:.4f},  '.format(np.average(batch_stats['avg_pobj'], weights=weights)) + \
                           ' Avg P|Noobj: {:.4f},  '.format(np.average(batch_stats['avg_pnoobj'], weights=weights)) + \
                           ' Avg Class: {:.4f}, '.format(np.average(batch_stats['avg_class'], weights=weights)) + \
                           ' Validation Loss: {:.6f}'.format(all_val_loss[-1])
                inner.set_postfix_str(disp_str)
                with open('training_loss.txt', 'a') as fl:
                    fl.writelines('Epoch: {} '.format(epoch) + disp_str + '\n')
            else:
                disp_str = ' Training Loss: {:.6f}, '.format(np.average(batch_loss, weights=weights)) + \
                           ' Avg IOU: {:.4f},  '.format(np.average(batch_stats['avg_obj_iou'], weights=weights)) + \
                           ' Avg P|Obj: {:.4f},  '.format(np.average(batch_stats['avg_pobj'], weights=weights)) + \
                           ' Avg P|Noobj: {:.4f},  '.format(np.average(batch_stats['avg_pnoobj'], weights=weights)) + \
                           ' Avg Class: {:.4f}'.format(np.average(batch_stats['avg_class'], weights=weights))
                inner.set_postfix_str(disp_str)
            # if val_data is not None:
            #     outer.set_postfix_str(' Training Loss: {:.6f},  Validation Loss: {:.6f}'.format(train_loss[-1],
            #                                                                                     val_loss[-1]))
            # else:
            #     outer.set_postfix_str(' Training Loss: {:.6f}'.format(train_loss[-1]))
            # outer.update()
            if epoch % checkpoint_frequency == 0:
                self.save_model(self.name + '_{}.pkl'.format(epoch))

        return all_train_loss, all_train_stats, all_val_loss, all_val_stats

    def set_grid_size(self, x, y):

        assert isinstance(x, int)
        assert isinstance(y, int)
        assert x == y, 'This implementation has only been tested for square grids.'

        self.grid_size = x, y
        for layer in self.detection_layers:
            layer.grid_size = x, y

    def save_model(self, name):
        """
        Save the entire YOLOv2tiny model by using the built-in Python
        pickle module.
        Parameters
        ----------
        name
            The filename where the model should be saved.
        """
        pickle.dump(self, open(name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def calculate_loss(self, data, batch_size, stats=None, fraction=0.1):
        """
        Calculates the loss for a random partition of a given dataset without
        tracking gradients. Useful for displaying the validation loss during
        training.
        Parameters
        ----------
        data : PascalDatasetYOLO
            A dataset object which returns images and targets to use for calculating
            the loss. Only a fraction of the images in the dataset will be tested.
        batch_size : int
            The number of images to load per batch. This should not influence the value
            that the function returns, but will affect performance.
        fraction : float
            The fraction of images from data that the loss should be calculated for.

        Returns
        -------
        float
            The mean loss over the fraction of the images that were sampled from
            the data.
        """
        val_dataloader = DataLoader(dataset=data,
                                    batch_size=batch_size,
                                    num_workers=NUM_WORKERS)

        losses = []
        if stats is None:
            stats = {'avg_obj_iou': [], 'avg_class': [], 'avg_pobj': [], 'avg_pnoobj': []}

        with torch.no_grad():
            for i, (images, _, targets) in enumerate(val_dataloader, 1):
                images = images.to(self.device)
                targets = targets.to(self.device)
                predictions = self(images)
                loss, stats = self.loss(predictions, targets, stats)
                losses.append(loss['total'].item())
                if i > len(val_dataloader) * fraction:
                    break

        return np.mean(losses), stats

    def parse_cfg(self, file):

        file = open(file, 'r')
        lines = file.read().split('\n')
        lines = [l for l in lines if len(l) > 0]
        lines = [l for l in lines if l[0] != '#']
        lines = [l.rstrip().lstrip() for l in lines]

        block = {}
        self.blocks = []

        for line in lines:
            if line[0] == '[':
                if len(block) > 0:
                    self.blocks.append(block)
                    block = {}
                block['type'] = line[1:-1]
            else:
                key, value = line.split('=')
                block[key.rstrip()] = value.lstrip()
        self.blocks.append(block)

        self.net_info = self.blocks[0]
        self.blocks = self.blocks[1:]

        return self.net_info, self.blocks

    def build_modules(self):

        self.layers = nn.ModuleList()

        index = 0
        prev_filters = self.channels
        output_filters = []

        for block in self.blocks:
            module = nn.Sequential()

            if block['type'] == 'convolutional':
                activation = block['activation']
                try:
                    batch_normalize = bool(block['batch_normalize'])
                    bias = False
                except KeyError:
                    batch_normalize = False
                    bias = True

                filters = int(block['filters'])
                padding = bool(block['pad'])
                kernel_size = int(block['size'])
                stride = int(block['stride'])

                if padding:
                    pad = (kernel_size - 1) // 2
                else:
                    pad = 0

                conv_layer = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
                nn.init.normal_(conv_layer.weight, 0., 0.001)
                module.add_module('conv_{0}'.format(index), conv_layer)

                if batch_normalize:
                    bn_layer = nn.BatchNorm2d(filters)
                    module.add_module('batch_norm_{}'.format(index), bn_layer)

                if activation == 'leaky':
                    activation_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
                    module.add_module('leaky_{}'.format(index), activation_layer)
                elif activation == 'swish':
                    activation_layer = Swish(beta=1., learnable=True)
                    module.add_module('swish_{}'.format(index), activation_layer)
                elif activation == 'linear':
                    pass
                else:
                    raise AssertionError('Unknown activation in configuration file.')

            elif block['type'] == 'upsample':
                stride = int(block['stride'])
                upsample_layer = nn.Upsample(scale_factor=stride, mode='nearest')
                module.add_module('upsample_{}'.format(index), upsample_layer)

            elif block['type'] == 'route':
                block['layers'] = block['layers'].split(',')

                start = int(block['layers'][0])
                try:
                    end = int(block['layers'][1])
                except IndexError:
                    end = 0

                if start > 0:
                    start = start - index
                if end > 0:
                    end = end - index

                route_layer = EmptyLayer()
                module.add_module('route_{0}'.format(index), route_layer)

                if end < 0:
                    filters = output_filters[index + start] + output_filters[index + end]
                else:
                    filters = output_filters[index + start]

            elif block['type'] == 'shortcut':
                shortcut_layer = EmptyLayer()
                module.add_module('shortcut_{0}'.format(index), shortcut_layer)

            elif block['type'] == 'maxpool':
                stride = int(block['stride'])
                size = int(block['size'])
                if stride == 1:
                    pad_layer = nn.ReplicationPad2d(padding=(0, 1, 0, 1))
                    module.add_module('pad_layer_{}'.format(index), pad_layer)
                    maxpool_layer = nn.MaxPool2d(size, stride)
                else:
                    maxpool_layer = nn.MaxPool2d(size, stride, padding=0)
                module.add_module('maxpool_{}'.format(index), maxpool_layer)

            elif block['type'] == 'yolo':
                assert False, NotImplementedError

                mask = block['mask'].split(',')
                mask = [int(x) for x in mask]

                width, height = self.image_size
                anchors = block['anchors'].split(',')
                anchors = [int(anchor) for anchor in anchors]
                anchors = [(anchors[i] / width, anchors[i + 1] / height)
                           for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in mask]

                detection_layer = YOLOv3Layer(self, anchors, softmax=True)
                self.detection_layers.append(detection_layer)
                module.add_module('detection_{}'.format(index), detection_layer)

            elif block['type'] == 'region':
                anchors = block['anchors'].split(',')
                anchors = [(float(anchors[i]), float(anchors[i + 1]))
                           for i in range(0, len(anchors), 2)]

                detection_layer = YOLOv2Layer(self, anchors)
                self.detection_layers.append(detection_layer)
                module.add_module('detection_{}'.format(index), detection_layer)

            else:
                raise AssertionError('Unknown block in configuration file.')

            self.layers.append(module)
            prev_filters = filters
            output_filters.append(filters)
            index += 1

        return self.net_info, self.layers

    def load_weights(self, file, only_imagenet=False):

        f = open(file, "rb")
        weights = np.fromfile(f, offset=16, dtype=np.float32)

        while weights[0] == 0.:
            weights = weights[1:]

        ptr = 0

        if not only_imagenet:
            layers = len(self.layers)
        else:
            layers = len(self.layers) - 3

        for i in range(layers):

            module_type = self.blocks[i]["type"]

            if module_type == "convolutional":
                module = self.layers[i]
                try:
                    batch_normalize = int(self.blocks[i]["batch_normalize"])
                except KeyError:
                    batch_normalize = 0

                conv_layer = module[0]

                if batch_normalize:
                    bn_layer = module[1]

                    num_bn_biases = bn_layer.bias.numel()
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_biases = bn_biases.view_as(bn_layer.bias.data)
                    bn_weights = bn_weights.view_as(bn_layer.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn_layer.running_mean)
                    bn_running_var = bn_running_var.view_as(bn_layer.running_var)

                    bn_layer.bias.data.copy_(bn_biases)
                    bn_layer.weight.data.copy_(bn_weights)
                    bn_layer.running_mean.copy_(bn_running_mean)
                    bn_layer.running_var.copy_(bn_running_var)

                else:
                    num_biases = conv_layer.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                    conv_biases = conv_biases.view_as(conv_layer.bias.data)

                    conv_layer.bias.data.copy_(conv_biases)

                num_weights = conv_layer.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv_layer.weight.data)

                conv_layer.weight.data.copy_(conv_weights)

        if not only_imagenet:
            assert ptr == len(weights), 'Weights file does not match model.'
        else:
            # Ensure that the size of the ImageNet backbone matches that of the YOLO backbone.
            # The last layer of the ImageNet backbone (weights and biases) will not be loaded.
            assert ptr == len(weights) - 1000 * 1024 - 1000 or ptr == len(weights), 'Weights file does not match model.'

    def save_weights(self, file):

        f = open(file, 'wb')

        header = np.array([0, 0, 1, 0], dtype=np.int32)
        header.tofile(f)

        # Now, let us save the weights
        for i in range(len(self.layers)):
            module_type = self.blocks[i]["type"]

            if module_type == "convolutional":
                module = self.layers[i]
                try:
                    batch_normalize = int(self.blocks[i]["batch_normalize"])
                except KeyError:
                    batch_normalize = 0

                conv_layer = module[0]

                if batch_normalize:
                    bn_layer = module[1]

                    bn_layer.bias.data.detach().cpu().numpy().tofile(f)
                    bn_layer.weight.data.detach().cpu().numpy().tofile(f)
                    bn_layer.running_mean.detach().cpu().numpy().tofile(f)
                    bn_layer.running_var.detach().cpu().numpy().tofile(f)

                else:
                    conv_layer.bias.data.detach().cpu().numpy().tofile(f)

                conv_layer.weight.data.detach().cpu().numpy().tofile(f)

    def calculate_stride(self):

        stride = 1.
        for block in self.blocks:
            if block['type'] == 'maxpool':
                size = int(block['size'])
                s = int(block['stride'])
                div = (1. - size) / s + 1.
                if div > 0:
                    stride /= div

        return int(stride)

    def collect_anchors(self):

        anchors = []
        for layer in self.detection_layers:
            anchors.append(layer.anchors)

        return torch.cat(anchors)

    def calculate_grid_size(self):

        width, height = self.image_size

        assert width % self.stride == 0
        assert height % self.stride == 0

        return int(width / self.stride), int(height / self.stride)

    def set_image_size(self, xy):

        x, y = xy

        assert isinstance(x, int)
        assert isinstance(y, int)

        self.image_size = x, y
        grid_size = self.calculate_grid_size()
        self.set_grid_size(*grid_size)

    def process_bboxes(self, predictions, image_info, confidence_threshold=0.01, overlap_threshold=0.5, nms=True):

        predictions = predictions.permute(0, 2, 3, 1)

        image_idx_ = []
        bboxes_ = []
        confidence_ = []
        classes_ = []

        for i, prediction in enumerate(predictions):
            prediction = prediction.contiguous().view(-1, self.num_features)
            prediction[:, 5:] = F.softmax(prediction[:, 5:], dim=-1)
            classes = torch.argmax(prediction[:, 5:], dim=-1)
            idx = torch.arange(0, len(prediction))
            confidence = prediction[:, 4] * prediction[idx, 5 + classes]

            mask = confidence > confidence_threshold

            if sum(mask) == 0:
                continue

            bboxes = prediction[mask, :4].clone()
            bboxes[:, ::2] *= self.stride
            bboxes[:, 1::2] *= self.stride
            bboxes = xywh2xyxy(bboxes)

            confidence = confidence[mask]
            classes = classes[mask]

            bboxes[:, ::2] = torch.clamp(bboxes[:, ::2],
                                         min=image_info['padding'][0][i]+1,
                                         max=self.image_size[0] - image_info['padding'][2][i])
            bboxes[:, 1::2] = torch.clamp(bboxes[:, 1::2],
                                          min=image_info['padding'][1][i]+1,
                                          max=self.image_size[1] - image_info['padding'][3][i])

            if nms:
                cls = torch.unique(classes)
                for c in cls:
                    cls_mask = (classes == c).nonzero().flatten()
                    mask = non_maximum_suppression(bboxes[cls_mask], confidence[cls_mask], overlap=overlap_threshold)
                    bboxes_.append(bboxes[cls_mask][mask])
                    confidence_.append(confidence[cls_mask][mask])
                    classes_.append(classes[cls_mask][mask])
                    image_idx_.append([image_info['id'][i]] * len(bboxes[cls_mask][mask]))
            else:
                bboxes_.append(bboxes)
                confidence_.append(confidence)
                classes_.append(classes)
                image_idx_.append([image_info['id'][i]] * len(bboxes))

        if len(bboxes_) > 0:
            bboxes = torch.cat(bboxes_).view(-1, 4)
            classes = torch.cat(classes_).flatten()
            confidence = torch.cat(confidence_).flatten()
            image_idx = [item for sublist in image_idx_ for item in sublist]

            return bboxes, classes, confidence, image_idx
        else:
            return torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device)

    def predict(self, dataset, confidence_threshold=0.1, overlap_threshold=0.5, show=True, export=True):

        self.eval()

        dataloader = DataLoader(dataset=dataset,
                                batch_size=dataset.batch_size,
                                num_workers=NUM_WORKERS)

        image_idx_ = []
        bboxes_ = []
        confidence_ = []
        classes_ = []

        with torch.no_grad():
            with tqdm(total=len(dataloader),
                      desc='Exporting',
                      leave=True) as pbar:
                for data in dataloader:
                    images, image_info = data
                    images = images.to(self.device)
                    predictions = self(images)
                    bboxes, classes, confidences, image_idx = self.process_bboxes(predictions,
                                                                                  image_info,
                                                                                  confidence_threshold,
                                                                                  overlap_threshold,
                                                                                  nms=True)

                    if show:
                        for i, (idx, image) in enumerate(zip(image_info['id'], images)):
                            width = self.image_size[0]
                            height = self.image_size[1]
                            if image.shape[0] == 3:
                                image = to_numpy_image(image, size=(width, height))
                            else:
                                mu = dataset.mu[0]
                                sigma = dataset.sigma[0]
                                image = to_numpy_image(image[0], size=(width, height), mu=mu, sigma=sigma, normalised=False)
                            mask = np.array(image_idx) == idx
                            for bbox, cls, confidence in zip(bboxes[mask], classes[mask], confidences[mask]):
                                name = dataset.classes[cls]
                                add_bbox_to_image(image, bbox, confidence, name)
                            plt.imshow(image)
                            plt.axis('off')
                            plt.show()

                    if export:
                        for idx in range(len(images)):
                            mask = [True if idx_ == image_info['id'][idx] else False for idx_ in image_idx]
                            for bbox, cls, confidence in zip(bboxes[mask], classes[mask], confidences[mask]):
                                name = dataset.classes[cls]
                                ids = image_info['id'][idx]
                                set_name = image_info['dataset'][idx]
                                confidence = confidence.item()
                                bbox[::2] -= image_info['padding'][0][idx]
                                bbox[1::2] -= image_info['padding'][1][idx]
                                bbox[::2] /= image_info['scale'][0][idx]
                                bbox[1::2] /= image_info['scale'][1][idx]
                                x1, y1, x2, y2 = bbox.detach().cpu().numpy()
                                export_prediction(cls=name,
                                                  prefix=self.name,
                                                  image_id=ids,
                                                  left=x1,
                                                  top=y1,
                                                  right=x2,
                                                  bottom=y2,
                                                  confidence=confidence,
                                                  set_name=set_name)

                    bboxes_.append(bboxes)
                    confidence_.append(confidences)
                    classes_.append(classes)
                    image_idx_.append(image_idx)

                    pbar.update()

            if len(bboxes_) > 0:
                bboxes = torch.cat(bboxes_).view(-1, 4)
                classes = torch.cat(classes_).flatten()
                confidence = torch.cat(confidence_).flatten()
                image_idx = [item for sublist in image_idx_ for item in sublist]

                return bboxes, classes, confidence, image_idx
            else:
                return torch.tensor([], device=self.device), \
                       torch.tensor([], device=self.device), \
                       torch.tensor([], device=self.device), \
                       []

    def freeze(self, freeze_last_layer=True):
        last_set = False
        for layer in reversed(self.layers):
            if not last_set and not freeze_last_layer:
                for param in layer.parameters():
                    param.requires_grad = True
                    last_set = True
            else:
                for param in layer.parameters():
                    param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def get_trainable_parameters(self):
        """
        Returns a list of a model's trainable parameters by checking which
        parameters are tracking their gradients.
        Returns
        -------
        list
            A list containing the trainable parameters.
        """
        trainable_parameters = []
        for param in self.parameters():
            if param.requires_grad:
                trainable_parameters.append(param)

        return trainable_parameters
