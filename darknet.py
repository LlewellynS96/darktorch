import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import jaccard, xywh2xyxy, non_maximum_suppression, to_numpy_image, add_bbox_to_image, export_prediction
from layers import *


LAMBDA_COORD = 5.
LAMBDA_OBJ = 1.
LAMBDA_CLASS = 1.
LAMBDA_NOOBJ = 1.

USE_CROSS_ENTROPY = False


class YOLO(nn.Module):

    def __init__(self, model, name='YOLOv2', device='cuda'):

        super(YOLO, self).__init__()

        self.net_info = {}
        self.blocks = []
        self.layers = nn.ModuleList()
        self.name = name
        self.device = device

        self.parse_cfg(model)

        self.channels = int(self.net_info['channels'])
        self.default_image_size = int(self.net_info['width']), int(self.net_info['height'])
        self.image_size = self.default_image_size
        self.num_classes = int(self.blocks[-1]['classes'])
        self.num_features = 5 + self.num_classes

        self.strides = self.calculate_strides()

        self.cache = {}
        self.detection_layers = []
        self.cache_layers = []
        self.build_modules()

        self.num_detectors = len(self.detection_layers)
        self.grid_sizes = [[-1, -1] for _ in range(self.num_detectors)]
        self.set_grid_sizes()

        self.anchors = self.collect_anchors()

        self.lr = float(self.net_info['learning_rate'])
        self.momentum = float(self.net_info['momentum'])
        self.weight_decay = float(self.net_info['decay'])
        try:
            self.noobj_iou_threshold = float(self.blocks[-1]['thresh'])
        except KeyError:
            self.noobj_iou_threshold = float(self.blocks[-1]['ignore_thresh'])
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
        self.layers[0][0] = nn.Conv2d(dims, conv.weight.data.shape[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layers[0][0].weight.data.copy_(conv.weight.data[:, :dims])
        self.to(self.device)
        self.channels = dims

    def forward(self, x):

        assert x.dim() == 4

        self.set_image_size(x.shape[-2:])
        # x = x.unsqueeze(2)

        output = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.cache_layers:
                self.cache[i] = x
            if i in self.detection_layers:
                output.append(x)
        return output

    def loss(self, predictions, targets, stats):
        assert type(predictions) == list
        loss = {}
        for i, (p, t) in enumerate(zip(predictions, targets)):
            assert p.shape == t.shape

            l = {}
            batch_size = t.shape[0]

            t = t.permute(0, 2, 3, 1)
            p = p.permute(0, 2, 3, 1)

            t = t.contiguous().view(batch_size, -1, self.num_features)
            p = p.contiguous().view(batch_size, -1, self.num_features)

            img_idx = torch.arange(batch_size, dtype=torch.float, device=self.device)
            img_idx = img_idx.reshape(-1, 1) * p.shape[2]
            t[:, :, 0] += 2. * img_idx
            p[:, :, 0] += 2. * img_idx
            img_idx = torch.arange(batch_size, dtype=torch.float, device=self.device)
            img_idx = img_idx.reshape(-1, 1) * p.shape[1]
            t[:, :, 1] += 2. * img_idx
            p[:, :, 1] += 2. * img_idx

            t = t.contiguous().view(-1, self.num_features)
            p = p.contiguous().view(-1, self.num_features)

            obj_mask = torch.nonzero(t[:, 4]).flatten()
            num_obj = len(obj_mask)

            if obj_mask.numel() > 0:
                p_xyxy = xywh2xyxy(p[:, :4].detach())
                t_xyxy = xywh2xyxy(t[obj_mask, :4])

                all_ious = jaccard(p_xyxy, t_xyxy)
                ious, _ = torch.max(all_ious, dim=1)
                stats['avg_obj_iou'].append(all_ious[obj_mask].diag().mean().item())

                mask = torch.nonzero(ious > self.noobj_iou_threshold).squeeze()
                t[mask, 4] = 1.
                noobj_mask = torch.nonzero(t[:, 4] == 0.).squeeze()

                l['coord'] = nn.MSELoss(reduction='sum')(p[obj_mask, 0], t[obj_mask, 0])
                l['coord'] += nn.MSELoss(reduction='sum')(p[obj_mask, 1], t[obj_mask, 1])
                l['coord'] += nn.MSELoss(reduction='sum')(torch.sqrt(p[obj_mask, 2]), torch.sqrt(t[obj_mask, 2]))
                l['coord'] += nn.MSELoss(reduction='sum')(torch.sqrt(p[obj_mask, 3]), torch.sqrt(t[obj_mask, 3]))
                l['coord'] *= LAMBDA_COORD / batch_size

                if self.iteration * self.batch_size < 12800:
                    l['bias'] = nn.MSELoss(reduction='sum')(p[noobj_mask, 0], t[noobj_mask, 0])
                    l['bias'] += nn.MSELoss(reduction='sum')(p[noobj_mask, 1], t[noobj_mask, 1])
                    l['bias'] += nn.MSELoss(reduction='sum')(torch.sqrt(p[noobj_mask, 2]),
                                                             torch.sqrt(t[noobj_mask, 2]))
                    l['bias'] += nn.MSELoss(reduction='sum')(torch.sqrt(p[noobj_mask, 3]),
                                                             torch.sqrt(t[noobj_mask, 3]))

                    l['bias'] *= 0.1 / batch_size

                p[obj_mask, 5:] = F.log_softmax(p[obj_mask, 5:], dim=-1)
                t_long = torch.argmax(t[obj_mask, 5:], dim=1)
                if USE_CROSS_ENTROPY:
                    l['class'] = nn.NLLLoss(reduction='sum')(p[obj_mask, 5:], t_long)
                else:
                    l['class'] = nn.MSELoss(reduction='sum')(torch.exp(p[obj_mask, 5:]),
                                                             t[obj_mask, 5:])
                l['class'] *= LAMBDA_CLASS / batch_size
                stats['avg_class'].append(torch.exp(p[obj_mask, 5 + t_long]).mean().item())

                # l['object'] = nn.MSELoss(reduction='sum')(p[obj_mask, 4],
                #                                           all_ious[obj_mask, torch.arange(num_obj)].detach())
                l['object'] = nn.MSELoss(reduction='sum')(p[obj_mask, 4],
                                                          t[obj_mask, 4])
                l['object'] *= LAMBDA_OBJ / batch_size
                stats['avg_pobj'].append(p[obj_mask, 4].mean().item())

                l['no_object'] = nn.MSELoss(reduction='sum')(p[noobj_mask, 4],
                                                             t[noobj_mask, 4])
                l['no_object'] *= LAMBDA_NOOBJ / batch_size
                stats['avg_pnoobj'].append(p[noobj_mask, 4].mean().item())
            else:
                l['object'] = torch.tensor([0.], device=self.device)
                l['coord'] = torch.tensor([0.], device=self.device)
                l['class'] = torch.tensor([0.], device=self.device)
                l['no_object'] = LAMBDA_NOOBJ / batch_size * nn.MSELoss(reduction='sum')(p[:, 4],
                                                                                         t[:, 4])
                if self.iteration * self.batch_size < 12800:
                    l['bias'] = nn.MSELoss(reduction='sum')(p[:, 0],
                                                            t[:, 0])
                    l['bias'] += nn.MSELoss(reduction='sum')(p[:, 1],
                                                             t[:, 1])
                    l['bias'] += nn.MSELoss(reduction='sum')(torch.sqrt(p[:, 2]),
                                                             torch.sqrt(t[:, 2]))
                    l['bias'] += nn.MSELoss(reduction='sum')(torch.sqrt(p[:, 3]),
                                                             torch.sqrt(t[:, 3]))
                    l['bias'] *= 0.1 / batch_size

            l['total'] = (l['coord'] + l['class'] + l['object'] + l['no_object'])
            for k, v, in l.items():
                try:
                    loss[k] = loss[k] + v
                except KeyError:
                    loss[k] = v

        return loss, stats

    def fit(self, train_data, optimizer, scheduler=None, epochs=1,
            val_data=None, checkpoint_frequency=100, num_workers=0):

        if scheduler is not None:
            scheduler.last_epoch = self.iteration

        self.train()
        self.freeze_bn()

        train_dataloader = DataLoader(dataset=train_data,
                                      batch_size=train_data.batch_size,
                                      num_workers=num_workers)

        all_train_loss = []
        all_train_stats = []
        all_val_loss = []
        all_val_stats = []

        if val_data is not None:
            self.eval()
            loss, val_stats = self.calculate_loss(val_data, val_data.batch_size, None, .2, num_workers)
            self.train()
            self.freeze_bn()
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
                    targets = [t.to(self.device) for t in targets]
                    if inner.n % self.subdivisions == 0:
                        optimizer.zero_grad()
                    predictions = self(images)
                    loss, batch_stats = self.loss(predictions, targets, batch_stats)
                    batch_loss.append(loss['total'].item())
                    if self.iteration * self.batch_size < 12800:
                        loss['total'] += loss['bias']
                    loss['total'].backward()
                    weights_0 = np.arange(1, 1 + len(batch_loss))
                    weights_1 = np.arange(1, 1 + len(batch_stats['avg_pobj']))
                    disp_str = ' Training Loss: {:.6f}, '.format(np.average(batch_loss,
                                                                            weights=weights_0)) + \
                               ' Avg IOU: {:.4f},  '.format(np.average(batch_stats['avg_obj_iou'],
                                                                       weights=weights_1)) + \
                               ' Avg P|Obj: {:.4f},  '.format(np.average(batch_stats['avg_pobj'],
                                                                         weights=weights_1)) + \
                               ' Avg P|Noobj: {:.4f},  '.format(np.average(batch_stats['avg_pnoobj'],
                                                                           weights=weights_1)) + \
                               ' Avg Class: {:.4f}, '.format(np.average(batch_stats['avg_class'],
                                                                        weights=weights_1)) + \
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
                loss, val_stats = self.calculate_loss(val_data, val_data.batch_size, val_stats, .2, num_workers)
                self.train()
                self.freeze_bn()
                val_data.step()
                all_val_loss.append(loss)
                all_val_stats.append(val_stats)
                disp_str = ' Training Loss: {:.6f}, '.format(np.average(batch_loss, weights=weights_0)) + \
                           ' Avg IOU: {:.4f},  '.format(np.average(batch_stats['avg_obj_iou'], weights=weights_1)) + \
                           ' Avg P|Obj: {:.4f},  '.format(np.average(batch_stats['avg_pobj'], weights=weights_1)) + \
                           ' Avg P|Noobj: {:.4f},  '.format(np.average(batch_stats['avg_pnoobj'], weights=weights_1)) + \
                           ' Avg Class: {:.4f}, '.format(np.average(batch_stats['avg_class'], weights=weights_1)) + \
                           ' Validation Loss: {:.6f}'.format(all_val_loss[-1])
                inner.set_postfix_str(disp_str)
                with open('{}_training_loss.txt'.format(self.name), 'a') as fl:
                    fl.writelines('Epoch: {} '.format(epoch) + disp_str + '\n')
            else:
                disp_str = ' Training Loss: {:.6f}, '.format(np.average(batch_loss, weights=weights_0)) + \
                           ' Avg IOU: {:.4f},  '.format(np.average(batch_stats['avg_obj_iou'], weights=weights_1)) + \
                           ' Avg P|Obj: {:.4f},  '.format(np.average(batch_stats['avg_pobj'], weights=weights_1)) + \
                           ' Avg P|Noobj: {:.4f},  '.format(np.average(batch_stats['avg_pnoobj'], weights=weights_1)) + \
                           ' Avg Class: {:.4f}'.format(np.average(batch_stats['avg_class'], weights=weights_1))
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

    def set_grid_sizes(self):

        width, height = self.image_size

        for i, _ in enumerate(self.detection_layers):
            assert width % self.strides[i] == 0
            assert height % self.strides[i] == 0
            x = int(width / self.strides[i])
            y = int(height / self.strides[i])
            self.grid_sizes[i] = x, y

    def save_model(self, name):
        """
        Save the entire YOLOv2 model by using the built-in Python
        pickle module.
        Parameters
        ----------
        name
            The filename where the model should be saved.
        """
        pickle.dump(self, open(name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def calculate_loss(self, data, batch_size, stats=None, fraction=0.1, num_workers=0):
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
                                    num_workers=num_workers)

        losses = []
        if stats is None:
            stats = {'avg_obj_iou': [], 'avg_class': [], 'avg_pobj': [], 'avg_pnoobj': []}

        with torch.no_grad():
            for i, (images, _, targets) in enumerate(val_dataloader, 1):
                images = images.to(self.device)
                targets = [t.to(self.device) for t in targets]
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

                first = int(block['layers'][0])
                try:
                    second = int(block['layers'][1])
                except IndexError:
                    second = 0

                if first > 0:
                    first = first - index
                if second > 0:
                    second = second - index

                route_layer = RouteLayer(index, first, second, self.cache)

                self.cache_layers.append(index + first)
                if second < 0:
                    self.cache_layers.append(index + second)

                if second < 0:
                    filters = output_filters[index + first] + output_filters[index + second]
                else:
                    filters = output_filters[index + first]
                module.add_module('route_{0}'.format(index), route_layer)

            elif block['type'] == 'shortcut':
                source = int(block['from'])
                activation = block['activation']
                route_layer = ShortcutLayer(index, source, self.cache)
                self.cache_layers.append(index + source)
                module.add_module('route_{0}'.format(index), route_layer)
                if activation == 'linear':
                    pass
                else:
                    raise AssertionError('Unknown activation for shortcut layer.')

            elif block['type'] == 'reorg':
                stride = int(block['stride'])
                reorg = ReorgLayer(stride)
                module.add_module('reorg_{}'.format(index), reorg)

                filters = prev_filters * stride * stride

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
                mask = block['mask'].split(',')
                mask = [int(x) for x in mask]

                width, height = self.image_size
                anchors = block['anchors'].split(',')
                anchors = [int(anchor) for anchor in anchors]
                anchors = [(anchors[i] / width, anchors[i + 1] / height)
                           for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in mask]
                i = len(self.detection_layers)
                anchors = np.array(anchors) * np.array(self.default_image_size) / self.strides[i]
                detection_layer = YOLOLayer(self, anchors)
                self.detection_layers.append(index)
                module.add_module('detection_{}'.format(index), detection_layer)

            elif block['type'] == 'region':
                anchors = block['anchors'].split(',')
                anchors = [(float(anchors[i]), float(anchors[i + 1]))
                           for i in range(0, len(anchors), 2)]
                detection_layer = YOLOLayer(self, anchors)
                self.detection_layers.append(index)
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
        i = 0

        while ptr < len(weights):

            if only_imagenet and ptr == len(weights) - 1000 * 1024 - 1000:
                break

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
            i += 1

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

    def calculate_strides(self):

        stride = 1.
        strides = []
        for block in self.blocks:
            if block['type'] == 'maxpool':
                size = int(block['size'])
                s = int(block['stride'])
                div = (1. - size) / s + 1.
                if div > 0:
                    stride /= div
            if block['type'] == 'convolutional' and int(block['stride']) > 1:
                s = int(block['stride'])
                div = 1. / s
                if div > 0:
                    stride /= div
            if block['type'] == 'upsample':
                s = int(block['stride'])
                div = 1. * s
                if div > 0:
                    stride /= div
            if block['type'] == 'region' or block['type'] == 'yolo':
                strides.append(int(stride))

        return strides

    def set_image_size(self, xy):

        x, y = xy

        assert isinstance(x, int)
        assert isinstance(y, int)

        self.image_size = x, y
        self.set_grid_sizes()

    def collect_anchors(self):
        anchors = []
        for i in self.detection_layers:
            anchors.append(self.layers[i][0].anchors)

        return anchors

    def process_bboxes(self, predictions, image_info, confidence_threshold=0.01, overlap_threshold=0.5, nms=True):

        image_idx_ = []
        bboxes_ = []
        classes_ = []
        conf_ = []

        for i, predictions_ in enumerate(predictions):
            if i not in [0, 1, 2]:  # Use this for specifying only a subset of detectors
                continue
            predictions_ = predictions_.permute(0, 2, 3, 1)

            for j, prediction in enumerate(predictions_):
                prediction = prediction.contiguous().view(-1, self.num_features)
                prediction[:, 5:] = F.softmax(prediction[:, 5:], dim=-1)
                classes = torch.argmax(prediction[:, 5:], dim=-1)
                idx = torch.arange(0, len(prediction))
                confidence = prediction[:, 4] * prediction[idx, 5 + classes]

                mask = confidence > confidence_threshold

                if sum(mask) == 0:
                    continue

                bboxes = prediction[mask, :4].clone()
                bboxes[:, ::2] *= self.strides[i]
                bboxes[:, 1::2] *= self.strides[i]
                bboxes = xywh2xyxy(bboxes)

                confidence = confidence[mask]
                classes = classes[mask]

                bboxes[:, ::2] = torch.clamp(bboxes[:, ::2],
                                             min=image_info['padding'][0][j]+1,
                                             max=self.image_size[0] - image_info['padding'][2][j])
                bboxes[:, 1::2] = torch.clamp(bboxes[:, 1::2],
                                              min=image_info['padding'][1][j]+1,
                                              max=self.image_size[1] - image_info['padding'][3][j])

                image_idx_.append(j)
                bboxes_.append(bboxes)
                classes_.append(classes)
                conf_.append(confidence)

        bboxes_ = \
            [torch.cat([bboxes_[ii] for ii, k in enumerate(image_idx_) if k == idx]) for idx in np.unique(image_idx_)]
        classes_ = \
            [torch.cat([classes_[ii] for ii, k in enumerate(image_idx_) if k == idx]) for idx in np.unique(image_idx_)]
        conf_ = \
            [torch.cat([conf_[ii] for ii, k in enumerate(image_idx_) if k == idx]) for idx in np.unique(image_idx_)]

        image_idx = []
        bboxes = []
        confidence = []
        classes = []

        for i, idx in enumerate(np.unique(image_idx_)):
            if nms:
                cls = torch.unique(classes_[i])
                for c in cls:
                    cls_mask = (classes_[i] == c).nonzero().flatten()
                    mask = non_maximum_suppression(bboxes_[i][cls_mask], conf_[i][cls_mask], overlap=overlap_threshold)
                    bboxes.append(bboxes_[i][cls_mask][mask])
                    classes.append(classes_[i][cls_mask][mask])
                    confidence.append(conf_[i][cls_mask][mask])
                    image_idx.append([image_info['id'][idx]] * len(bboxes_[i][cls_mask][mask]))
            else:
                bboxes.append(bboxes_[i])
                confidence.append(conf_[i])
                classes.append(classes_[i])
                image_idx.append([image_info['id'][idx]] * len(bboxes_[i]))

        if len(bboxes) > 0:
            bboxes = torch.cat(bboxes).view(-1, 4)
            classes = torch.cat(classes).flatten()
            confidence = torch.cat(confidence).flatten()
            image_idx = [item for sublist in image_idx for item in sublist]

            return bboxes, classes, confidence, image_idx
        else:
            return torch.tensor([], device=self.device), \
                   torch.tensor([], dtype=torch.long, device=self.device), \
                   torch.tensor([], device=self.device), \
                   []

    def predict(self, dataset, confidence_threshold=0.1, overlap_threshold=0.5, show=True, export=True, num_workers=0):

        self.eval()

        dataloader = DataLoader(dataset=dataset,
                                batch_size=dataset.batch_size,
                                num_workers=num_workers)

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
                    # images, image_info, targets = data
                    images = images.to(self.device)
                    predictions = self(images)
                    # predictions = [t.to(self.device) for t in targets]
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
                            if len(image_idx) > 0:
                                mask = np.array(image_idx) == idx
                                for bbox, cls, confidence in zip(bboxes[mask], classes[mask], confidences[mask]):
                                    name = dataset.classes[cls]
                                    # coco = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
                                    # name = coco[cls]
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

    def mini_freeze(self, n=13):
        for layer in list(self.layers)[:n]:
            for param in layer.parameters():
                param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = False
                module.eval()

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
