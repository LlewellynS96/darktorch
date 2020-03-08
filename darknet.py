import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import NUM_WORKERS
from utils import jaccard, xywh2xyxy, non_maximum_suppression, to_numpy_image, add_bbox_to_image, export_prediction
from layers import *

REDUCTION = 'sum'
NOOBJ_IOU_THRESHOLD = 0.6
LAMBDA_COORD = 100.
LAMBDA_OBJ = 5.
LAMBDA_CLASS = 5.
LAMBDA_NOOBJ = 1.

MULTI_SCALE_FREQ = 10.


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

        self.anchors = self.calculate_anchors()
        self.num_anchors = len(self.anchors)

        self.focal_loss = False

        self.to(device)

    def forward(self, x):

        assert x.dim() == 4

        for layer in self.layers:
            x = layer(x)

        x[:, 0::self.num_features, :, :] *= self.image_size[0]
        x[:, 1::self.num_features, :, :] *= self.image_size[1]
        x[:, 2::self.num_features, :, :] *= self.image_size[0]
        x[:, 3::self.num_features, :, :] *= self.image_size[1]

        return x

    def loss(self, predictions, targets):

        assert predictions.shape == targets.shape

        loss = dict()
        batch_size = targets.shape[0]

        targets = targets.permute(0, 2, 3, 1)
        predictions = predictions.permute(0, 3, 2, 1)

        targets = targets.contiguous().view(batch_size, -1, self.num_features)
        predictions = predictions.contiguous().view(batch_size, -1, self.num_features)

        image_idx = torch.arange(batch_size, dtype=torch.float, device=self.device).reshape(-1, 1, 1)
        targets[:, :, :2] += image_idx
        predictions[:, :, :2] += image_idx

        targets = targets.contiguous().view(-1, self.num_features)
        predictions = predictions.contiguous().view(-1, self.num_features)

        n_anch = len(predictions)

        obj_mask = torch.nonzero(targets[:, 4]).flatten()
        n_obj = obj_mask.numel()

        if obj_mask.numel() > 0:
            predictions_xyxy = xywh2xyxy(predictions[:, :4])
            targets_xyxy = xywh2xyxy(targets[obj_mask, :4])

            ious = jaccard(predictions_xyxy, targets_xyxy)
            ious, _ = torch.max(ious, dim=1)

            mask = torch.nonzero(ious > NOOBJ_IOU_THRESHOLD).squeeze()
            targets[mask, 4] = 2.
            noobj_mask = torch.nonzero(targets[:, 4] == 0.).squeeze()

            loss['coord'] = nn.MSELoss(reduction=REDUCTION)(predictions[obj_mask, 0], targets[obj_mask, 0])
            loss['coord'] += nn.MSELoss(reduction=REDUCTION)(predictions[obj_mask, 1], targets[obj_mask, 1])
            loss['coord'] += nn.MSELoss(reduction=REDUCTION)(torch.sqrt(predictions[obj_mask, 2]),
                                                             torch.sqrt(targets[obj_mask, 2]))
            loss['coord'] += nn.MSELoss(reduction=REDUCTION)(torch.sqrt(predictions[obj_mask, 3]),
                                                             torch.sqrt(targets[obj_mask, 3]))
            loss['coord'] *= LAMBDA_COORD / batch_size

            if obj_mask.numel() > 0:
                predictions[obj_mask, 5:] = nn.Softmax(dim=-1)(predictions[obj_mask, 5:])
                loss['class'] = nn.MSELoss(reduction=REDUCTION)(predictions[obj_mask, 5:],
                                                                targets[obj_mask, 5:])
                loss['class'] *= LAMBDA_CLASS / batch_size
            else:
                loss['class'] = 0.

            loss['object'] = nn.MSELoss(reduction=REDUCTION)(predictions[obj_mask, 4],
                                                             ious[obj_mask].detach())

            loss['object'] *= LAMBDA_OBJ / batch_size

            loss['no_object'] = nn.MSELoss(reduction=REDUCTION)(predictions[noobj_mask, 4],
                                                                targets[noobj_mask, 4])
            loss['no_object'] *= LAMBDA_NOOBJ / batch_size

        else:
            loss['object'] = torch.tensor([0.], device=self.device)
            loss['coord'] = torch.tensor([0.], device=self.device)
            loss['class'] = torch.tensor([0.], device=self.device)
            loss['no_object'] = LAMBDA_NOOBJ / n_anch * nn.MSELoss(reduction=REDUCTION)(predictions[:, 4],
                                                                                        targets[:, 4])

        loss['total'] = loss['object'] + loss['coord'] + loss['class'] + loss['no_object']

        return loss

    def fit(self, train_data, optimizer, scheduler=None, batch_size=1, epochs=1,
            val_data=None, shuffle=True, multi_scale=True, checkpoint_frequency=100):

        self.train()

        train_dataloader = DataLoader(dataset=train_data,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      num_workers=NUM_WORKERS)

        train_loss = []
        val_loss = []

        # ==========================================================
        # PyCharm does not support nested tqdm progress bars, and
        # thus this feature has been removed.
        # ==========================================================
        # with tqdm(total=epochs,
        #           desc='{} Training PASCAL VOC'.format(self.name),
        #           leave=True,
        #           unit='batches') as outer:
        for epoch in range(1, epochs + 1):
            batch_loss = []
            with tqdm(total=len(train_dataloader),
                      desc='Epoch: [{}/{}]'.format(epoch, epochs),
                      leave=True,
                      unit='batches') as inner:
                for images, _, targets in train_dataloader:
                    images = images.to(self.device)
                    targets = targets.to(self.device)
                    optimizer.zero_grad()
                    predictions = self(images)
                    loss = self.loss(predictions, targets)
                    batch_loss.append(loss['total'].item())
                    loss['total'].backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5., norm_type='inf')
                    optimizer.step()
                    inner.set_postfix_str(' Training Loss: {:.6f}'.format(np.mean(batch_loss)))
                    inner.update()
                    if inner.n % MULTI_SCALE_FREQ == 0 and multi_scale:
                        random_size = np.random.randint(10, 20) * self.downscale_factor
                        self.set_image_size(random_size, random_size, dataset=train_data)
                train_loss.append(np.mean(batch_loss))
                if val_data is not None:
                    self.reset_image_size(dataset=[train_data, val_data])
                    val_loss.append(self.calculate_loss(val_data, 2 * batch_size))
                    inner.set_postfix_str(' Training Loss: {:.6f},  Validation Loss: {:.6f}'.format(train_loss[-1],
                                                                                                    val_loss[-1]))
                    with open('training_loss.txt', 'a') as fl:
                        fl.writelines('Epoch: {}, Train loss: {:.6f}, Val loss: {:.6f}\n'.format(epoch,
                                                                                                 train_loss[-1],
                                                                                                 val_loss[-1]))
                else:
                    inner.set_postfix_str(' Training Loss: {:.6f}'.format(train_loss[-1]))
            if scheduler is not None:
                scheduler.step()
            # if val_data is not None:
            #     outer.set_postfix_str(' Training Loss: {:.6f},  Validation Loss: {:.6f}'.format(train_loss[-1],
            #                                                                                     val_loss[-1]))
            # else:
            #     outer.set_postfix_str(' Training Loss: {:.6f}'.format(train_loss[-1]))
            # outer.update()
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
        """
        Save the entire YOLOv2tiny model by using the built-in Python
        pickle module.
        Parameters
        ----------
        name
            The filename where the model should be saved.
        """
        pickle.dump(self, open(name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def calculate_loss(self, data, batch_size, fraction=0.05):
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
            assert ptr == len(weights) - 1000 * 1024 - 1000, 'Weights file does not match model.'

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
        self.anchors = self.calculate_anchors()

        if dataset is not None:
            if isinstance(dataset, (list, tuple)):
                for d in dataset:
                    d.set_image_size(x, y)
                    d.set_grid_size(*grid_size)
                    d.set_anchors(self.anchors)
            else:
                dataset.set_image_size(x, y)
                dataset.set_grid_size(*grid_size)
                dataset.set_anchors(self.anchors)

    def reset_image_size(self, dataset=None):

        self.set_image_size(*self.default_image_size, dataset=dataset)

    def calculate_anchors(self):
        return self.collect_anchors() / \
               torch.tensor(self.grid_size, device=self.device) * \
               torch.tensor(self.image_size, device=self.device)

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
            bboxes = xywh2xyxy(bboxes)

            confidence = confidence[mask]
            classes = classes[mask]

            bboxes[:, ::2] = torch.clamp(bboxes[:, ::2],
                                      min=image_info['padding'][0][i],
                                      max=self.image_size[0] - image_info['padding'][2][i])
            bboxes[:, 1::2] = torch.clamp(bboxes[:, 1::2],
                                       min=image_info['padding'][1][i],
                                       max=self.image_size[1] - image_info['padding'][3][i])

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
            with tqdm(total=len(dataloader),
                      desc='Exporting',
                      leave=True) as pbar:
                for images, image_info, targets in dataloader:
                    images = images.to(self.device)
                    predictions = self(images)
                    # predictions = targets
                    bboxes, classes, confidences, image_idx = self.process_bboxes(predictions,
                                                                                  image_info,
                                                                                  confidence_threshold,
                                                                                  overlap_threshold,
                                                                                  nms=True)
                    if show:
                        for idx, image in enumerate(images):
                            width = image_info['width'][idx]
                            height = image_info['height'][idx]
                            width = self.image_size[0]
                            height = self.image_size[1]
                            image = to_numpy_image(image, size=(width, height))
                            mask = image_idx == idx
                            for bbox, cls, confidence in zip(bboxes[mask], classes[mask], confidences[mask]):
                                name = dataset.classes[cls]
                                add_bbox_to_image(image, bbox, confidence, name)
                            plt.imshow(image)
                            # plt.axis('off')
                            plt.show()

                    if export:
                        for idx in range(len(images)):
                            mask = image_idx == idx
                            for bbox, cls, confidence in zip(bboxes[mask], classes[mask], confidences[mask]):
                                name = dataset.classes[cls]
                                ids = image_info['id'][idx]
                                set_name = image_info['dataset'][idx]
                                confidence = confidence.item()
                                ratio = min([float(d) / max([width, height]) for d in self.image_size])
                                bbox[::2] -= image_info['padding'][0]
                                bbox[1::2] -= image_info['padding'][1]
                                bbox[::2] /= ratio
                                bbox[::2] /= ratio
                                x1, y1, x2, y2 = bbox.detach().cpu().numpy()
                                width = image_info['width']
                                height = image_info['height']
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
