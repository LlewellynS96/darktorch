import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from time import time
from tqdm import tqdm
from utils import PRINT_LINE_LEN, NUM_WORKERS
from utils import jaccard, xywh2xyxy, non_maximum_suppression, to_numpy_image, add_bbox_to_image, export_prediction
from layers import *


REDUCTION = 'mean'


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
        self.downscale_factor = self.calculate_downscale_factor()
        self.grid_size = self.calculate_grid_size()

        self.build_modules()

        self.anchors = self.collect_anchors()
        self.num_anchors = len(self.anchors)

        self.focal_loss = True

        self.to(device)

    def forward(self, x):

        assert x.dim() == 4

        for layer in self.layers:
            x = layer(x)

        return x

    def loss(self, predictions, targets):

        assert predictions.shape == targets.shape

        batch_size = targets.shape[0]

        lambda_coord = 5.
        lambda_obj = 1.
        lambda_noobj = 0.5

        loss = {}

        targets = targets.permute(0, 2, 3, 1)
        predictions = predictions.permute(0, 3, 2, 1)

        targets = targets.contiguous().view(-1, self.num_features)
        predictions = predictions.contiguous().view(-1, self.num_features)

        obj_mask = torch.nonzero(targets[:, 4]).flatten()

        if obj_mask.numel() > 0:
            predictions_xyxy = xywh2xyxy(predictions[:, :4])
            targets_xyxy = xywh2xyxy(targets[obj_mask, :4])

            ious = jaccard(predictions_xyxy, targets_xyxy)
            ious, _ = torch.max(ious, dim=1)

            loss['object'] = lambda_obj * nn.MSELoss(reduction=REDUCTION)(predictions[obj_mask, 4], ious[obj_mask])

            loss['coord'] = nn.MSELoss(reduction=REDUCTION)(predictions[obj_mask, 0], targets[obj_mask, 0])
            loss['coord'] += nn.MSELoss(reduction=REDUCTION)(predictions[obj_mask, 1], targets[obj_mask, 1])
            loss['coord'] += nn.MSELoss(reduction=REDUCTION)(torch.sqrt(predictions[obj_mask, 2]),
                                                             torch.sqrt(targets[obj_mask, 2]))
            loss['coord'] += nn.MSELoss(reduction=REDUCTION)(torch.sqrt(predictions[obj_mask, 3]),
                                                             torch.sqrt(targets[obj_mask, 3]))
            loss['coord'] *= lambda_coord

            loss['class'] = 0.
            for cls in range(self.num_classes):
                if self.focal_loss == 'focal':
                    loss['class'] += FocalLoss(reduction=REDUCTION)(predictions[obj_mask, 5 + cls],
                                                                    targets[obj_mask, 5 + cls])
                else:
                    loss['class'] += nn.MSELoss(reduction=REDUCTION)(predictions[obj_mask, 5 + cls],
                                                                     targets[obj_mask, 5 + cls])
            loss['class'] /= self.num_classes

            iou_threshold = 0.6
            noobj_mask = torch.nonzero(ious > iou_threshold).squeeze()
            targets[noobj_mask, 4] = 2.
            noobj_mask = torch.nonzero(targets[:, 4] == 0.).squeeze()
            loss['no_object'] = lambda_noobj * nn.MSELoss(reduction=REDUCTION)(predictions[noobj_mask, 4],
                                                                               targets[noobj_mask, 4])
        else:
            loss['object'] = torch.tensor([0.], device=self.device)
            loss['coord'] = torch.tensor([0.], device=self.device)
            loss['class'] = torch.tensor([0.], device=self.device)
            loss['no_object'] = lambda_noobj * nn.MSELoss(reduction=REDUCTION)(predictions[:, 4], targets[:, 4])

        loss['total'] = loss['object'] + loss['coord'] + loss['class'] + loss['no_object']

        return loss

    def fit(self, train_data, optimizer, batch_size=1, epochs=1, verbose=True,
            val_data=None, shuffle=True, multi_scale=True, checkpoint_frequency=100):

        self.train()

        train_dataloader = DataLoader(dataset=train_data,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      num_workers=NUM_WORKERS)

        train_loss = []
        val_loss = []

        for epoch in range(1, epochs + 1):
            batch_loss = []
            start = time()
            if multi_scale:
                random_size = np.random.randint(10, 20) * self.downscale_factor
                self.set_image_size(random_size, random_size, dataset=train_data)
            with tqdm(total=len(train_dataloader),
                      desc='Epoch: [{}/{}]'.format(epoch, epochs),
                      leave=True,
                      unit='batches') as pbar:
                for images, _, targets in train_dataloader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    optimizer.zero_grad()
                    predictions = self(images)
                    loss = self.loss(predictions, targets)
                    batch_loss.append(loss['total'].item())
                    loss['total'].backward()
                    optimizer.step()
                    pbar.set_postfix_str(' Training Loss: {:.6f}'.format(np.mean(batch_loss)))
                    pbar.update(True)
                train_loss.append(np.mean(batch_loss))
                if val_data is not None:
                    self.reset_image_size()
                    val_loss.append(self.calculate_loss(val_data, 2 * batch_size))
                    pbar.set_postfix_str(' Training Loss: {:.6f},  Validation Loss: {:.6f}'.format(train_loss[-1],
                                                                                                 val_loss[-1]))
                else:
                    pbar.set_postfix_str(' Training Loss: {:.6f}'.format(train_loss[-1]))
                pbar.refresh()

            if epoch % checkpoint_frequency == 0:
                self.save_model(self.name + '_{}.pkl'.format(epoch))

        return train_loss, val_loss

    def set_grid_size(self, x, y):

        assert isinstance(x, int)
        assert isinstance(y, int)

        self.grid_size = x, y
        for layer in self.detection_layers:
            layer.grid_size = x, y

    def save_model(self, name):

        pickle.dump(self, open(name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def calculate_loss(self, data, batch_size, fraction=0.05):

        val_dataloader = DataLoader(dataset=data,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=NUM_WORKERS)
        losses = []
        with torch.no_grad():
            for i, (images, _, targets) in enumerate(val_dataloader, 1):
                images = images.to(self.device)
                targets = targets.to(self.device)
                predictions = self(images)
                loss = self.loss(predictions, targets)
                losses.append(loss['total'].item())
                if i > len(val_dataloader) * fraction:
                    break

        return np.mean(losses)

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

        return self.blocks

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
                module.add_module('conv_{0}'.format(index), conv_layer)

                if batch_normalize:
                    bn_layer = nn.BatchNorm2d(filters)
                    module.add_module('batch_norm_{0}'.format(index), bn_layer)

                if activation == 'leaky':
                    activation_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
                    module.add_module('leaky_{0}'.format(index), activation_layer)
                elif activation == 'swish':
                    activation_layer = Swish(beta=1., learnable=True)
                    module.add_module('swish_{0}'.format(index), activation_layer)
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
                anchors = [(float(anchors[i]) / self.grid_size[0], float(anchors[i + 1]) / self.grid_size[1])
                           for i in range(0, len(anchors), 2)]

                detection_layer = YOLOv2Layer(self, anchors, softmax=False)
                self.detection_layers.append(detection_layer)
                module.add_module('detection_{}'.format(index), detection_layer)

            else:
                raise AssertionError('Unknown block in configuration file.')

            self.layers.append(module)
            prev_filters = filters
            output_filters.append(filters)
            index += 1

        return self.net_info, self.layers

    def load_weights(self, file):

        f = open(file, "rb")
        weights = np.fromfile(f, offset=16, dtype=np.float32)

        ptr = 0
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

        assert ptr == len(weights), 'Weights file does not match model.'

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
                except:
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

    def calculate_downscale_factor(self):

        downscale_factor = 1.
        for block in self.blocks:
            if block['type'] == 'maxpool':
                size = int(block['size'])
                stride = int(block['stride'])
                div = (1. - size) / stride + 1.
                if div > 0:
                    downscale_factor /= div

        return int(downscale_factor)

    def collect_anchors(self):

        anchors = []
        for layer in self.detection_layers:
            anchors.append(layer.anchors)

        return torch.cat(anchors)

    def calculate_grid_size(self):

        width, height = self.image_size

        assert width % self.downscale_factor == 0
        assert height % self.downscale_factor == 0

        return int(width / self.downscale_factor), int(height / self.downscale_factor)

    def set_image_size(self, x, y, dataset=None):

        assert isinstance(x, int)
        assert isinstance(y, int)

        grid_size = int(x / self.downscale_factor), int(y / self.downscale_factor)
        self.image_size = x, y
        self.set_grid_size(*grid_size)
        if dataset is not None:
            if isinstance(dataset, (list, tuple)):
                for d in dataset:
                    d.set_image_size(x, y)
                    d.set_grid_size(*grid_size)
            else:
                dataset.set_image_size(x, y)
                dataset.set_grid_size(*grid_size)

    def reset_image_size(self, dataset=None):

        self.set_image_size(*self.default_image_size, dataset=dataset)

    def process_bboxes(self, predictions, confidence_threshold=0.01, overlap_threshold=0.5, nms=True):

        predictions = predictions.permute(0, 2, 3, 1)

        image_idx_ = []
        bboxes_ = []
        confidence_ = []
        classes_ = []

        for i, prediction in enumerate(predictions):
            prediction = prediction.contiguous().view(-1, self.num_features)

            classes = torch.argmax(prediction[:, 5:], dim=-1)
            idx = torch.arange(0, len(prediction))
            confidence = prediction[:, 4] * prediction[idx, 5 + classes]

            mask = confidence > confidence_threshold

            if sum(mask) == 0:
                continue

            bboxes = prediction[mask, :4].clone()
            bboxes = xywh2xyxy(bboxes)

            confidence = confidence[mask]
            classes = classes[mask]

            if nms:
                cls = torch.unique(classes)
                for c in cls:
                    cls_mask = (classes == c).nonzero().flatten()
                    mask = non_maximum_suppression(bboxes[cls_mask], confidence[cls_mask], overlap=overlap_threshold)
                    bboxes_.append(bboxes[cls_mask][mask])
                    confidence_.append(confidence[cls_mask][mask])
                    classes_.append(classes[cls_mask][mask])
                    image_idx_.append(torch.ones(len(bboxes[cls_mask][mask]), device=self.device) * i)
            else:
                bboxes_.append(bboxes)
                confidence_.append(confidence)
                classes_.append(classes)
                image_idx_.append(torch.ones(len(bboxes)) * i)

        if len(bboxes_) > 0:
            bboxes = torch.cat(bboxes_).view(-1, 4)
            classes = torch.cat(classes_).flatten()
            confidence = torch.cat(confidence_).flatten()
            image_idx = torch.cat(image_idx_).flatten()

            bboxes = torch.clamp(bboxes, min=0, max=1)
            bboxes = torch.clamp(bboxes, min=0, max=1)

            return bboxes, classes, confidence, image_idx
        else:
            return torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device)

    def predict(self, dataset, batch_size=10, confidence_threshold=0.1, overlap_threshold=0.5, show=True, export=True):

        self.eval()

        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=NUM_WORKERS)

        image_idx_ = []
        bboxes_ = []
        confidence_ = []
        classes_ = []

        with torch.no_grad():
            for i, (images, image_info, targets) in enumerate(dataloader):
                images = images.to(self.device)
                predictions = self(images)
                # predictions = targets
                bboxes, classes, confidences, image_idx = self.process_bboxes(predictions,
                                                                              confidence_threshold=confidence_threshold,
                                                                              overlap_threshold=overlap_threshold,
                                                                              nms=True)
                if show:
                    for idx, image in enumerate(images):
                        width = image_info['width'][idx]
                        height = image_info['height'][idx]
                        image = to_numpy_image(image, size=(width, height))
                        mask = image_idx == idx
                        for bbox, cls, confidence in zip(bboxes[mask], classes[mask], confidences[mask]):
                            name = dataset.classes[cls]
                            add_bbox_to_image(image, bbox, confidence, name)
                        plt.imshow(image)
                        plt.show()

                if export:
                    for idx in range(len(images)):
                        mask = image_idx == idx
                        for bbox, cls, confidence in zip(bboxes[mask], classes[mask], confidences[mask]):
                            name = dataset.classes[cls]
                            ids = image_info['id'][idx]
                            set_name = image_info['dataset'][idx]
                            confidence = confidence.item()
                            x1, y1, x2, y2 = bbox.detach().cpu().numpy()
                            width = image_info['width'][idx].item()
                            height = image_info['height'][idx].item()
                            x1 *= width
                            x2 *= width
                            y1 *= height
                            y2 *= height
                            export_prediction(cls=name,
                                              image_id=ids,
                                              left=x1,
                                              top=y1,
                                              right=x2,
                                              bottom=y2,
                                              confidence=confidence,
                                              set_name=set_name)

                    ii = int(i / len(dataloader) * PRINT_LINE_LEN)
                    progress = '=' * ii
                    progress += '>'
                    progress += ' ' * (PRINT_LINE_LEN - ii)
                    string = 'Exporting |{}| {:.1f} %'.format(progress, i / len(dataloader) * 100.)
                    print('\r' + string, end='')

                bboxes_.append(bboxes)
                confidence_.append(confidences)
                classes_.append(classes)
                image_idx_.append(image_idx)

        if export:
            progress = '=' * PRINT_LINE_LEN
            string = 'Exporting |{}| {:.1f} %'.format(progress, 100.)
            print('\r' + string, end='')

        if len(bboxes_) > 0:
            bboxes = torch.cat(bboxes_).view(-1, 4)
            classes = torch.cat(classes_).flatten()
            confidence = torch.cat(confidence_).flatten()
            image_idx = torch.cat(image_idx_).flatten()

            return bboxes, classes, confidence, image_idx
        else:
            return torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device)

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

        trainable_parameters = []
        for param in self.parameters():
            if param.requires_grad:
                trainable_parameters.append(param)

        return trainable_parameters
