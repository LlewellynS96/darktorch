import numpy as np
import cv2
import pickle
import torchsummary
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms
from time import time
from dataset import PascalDatasetYOLO
from utils import PRINT_LINE_LEN, NUM_WORKERS
from utils import jaccard, xywh2xyxy, non_maximum_suppression, to_numpy_image, add_bbox_to_image
from layers import *


class YOLOv2tiny(nn.Module):

    def __init__(self, model, device='cuda'):

        super(YOLOv2tiny, self).__init__()

        self.net_info = {}
        self.layers = nn.ModuleList()
        self.blocks = []
        self.detection_layers = []
        self.device = device

        self.parse_cfg(model)

        self.num_classes = int(self.blocks[-1]['classes'])
        self.num_features = 5 + self.num_classes
        self.channels = int(self.net_info['channels'])
        self.default_image_size = int(self.net_info['width']), int(self.net_info['height'])
        self.image_size = self.default_image_size
        self.downscale_factor = self.calculate_downscale_factor()
        self.grid_size = self.calculate_grid_size()

        self.build_modules()

        self.anchors = self.collect_anchors()
        self.num_anchors = len(self.anchors)

    def forward(self, x):

        assert x.dim() == 4

        for layer in self.layers:
            x = layer(x)

        return x

    def loss(self, predictions, targets):

        assert predictions.shape == targets.shape

        batch_size = targets.shape[0]

        lambda_coord = 10.
        lambda_obj = 2.
        lambda_noobj = 0.2

        loss = {}

        targets = targets.permute(0, 2, 3, 1)
        predictions = predictions.permute(0, 2, 3, 1)

        targets = targets.contiguous().view(-1, self.num_features)
        predictions = predictions.contiguous().view(-1, self.num_features)

        # Create a mask and compile a tensor only containing detectors that are responsible for objects.
        obj_mask = torch.nonzero(targets[:, 4]).flatten()

        if obj_mask.numel() > 0:
            # Convert t_w and t_h --> w and h.
            predictions_xyxy = xywh2xyxy(predictions[:, :4])
            targets_xyxy = xywh2xyxy(targets[obj_mask, :4])

            ious = jaccard(predictions_xyxy, targets_xyxy)
            ious, _ = torch.max(ious, dim=1)

            loss['object'] = lambda_obj * nn.MSELoss(reduction='sum')(predictions[obj_mask, 4], ious[obj_mask])
            loss['coord'] = nn.MSELoss(reduction='sum')(predictions[obj_mask, 0], targets[obj_mask, 0])
            loss['coord'] += nn.MSELoss(reduction='sum')(predictions[obj_mask, 1], targets[obj_mask, 1])
            loss['coord'] += nn.MSELoss(reduction='sum')(torch.sqrt(predictions[obj_mask, 2]),
                                                         torch.sqrt(targets[obj_mask, 2]))
            loss['coord'] += nn.MSELoss(reduction='sum')(torch.sqrt(predictions[obj_mask, 3]),
                                                         torch.sqrt(targets[obj_mask, 3]))
            # Multiply by lambda_coord
            loss['coord'] *= lambda_coord
            # Divide by the number of separate loss components.
            loss['class'] = 0.
            for cls in range(self.num_classes):
                # loss['class'] += nn.MSELoss(reduction='sum')(predictions[obj_mask, 5 + cls], targets[obj_mask, 5 + cls])
                loss['class'] += FocalLoss(reduction='sum')(predictions[obj_mask, 5 + cls], targets[obj_mask, 5 + cls])

            threshold = 0.6
            noobj_mask = torch.nonzero(ious > threshold).squeeze()
            targets[noobj_mask, 4] = 2.
            noobj_mask = torch.nonzero(targets[:, 4] == 0.).squeeze()

            loss['no_object'] = lambda_noobj * nn.MSELoss(reduction='sum')(predictions[noobj_mask, 4],
                                                                           targets[noobj_mask, 4])
        else:
            loss['object'] = torch.tensor([0.], device=self.device)
            loss['coord'] = torch.tensor([0.], device=self.device)
            loss['class'] = torch.tensor([0.], device=self.device)

            loss['no_object'] = lambda_noobj * nn.MSELoss(reduction='sum')(predictions[:, 4], targets[:, 4])

        loss['object'] /= batch_size
        loss['coord'] /= batch_size
        loss['class'] /= batch_size
        loss['no_object'] /= batch_size
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
            for i, (images, targets) in enumerate(train_dataloader, 1):
                # image = images[0].permute(1, 2, 0).numpy()
                # image *= 255.
                # image = image.astype(dtype=np.uint8)
                # plt.imshow(image)
                # plt.show()
                images = images.to(self.device)
                targets = targets.to(self.device)
                optimizer.zero_grad()
                predictions = self(images)
                loss = self.loss(predictions, targets)
                batch_loss.append(loss['total'].item())
                loss['total'].backward()
                optimizer.step()
                if verbose:
                    ii = int(i / len(train_dataloader) * PRINT_LINE_LEN)
                    progress = '=' * ii
                    progress += '>'
                    progress += ' ' * (PRINT_LINE_LEN - ii)
                    string = 'Epoch: [{}/{}] |{}| [{}/{}] Training Loss: {:.6f}'.format(epoch, epochs, progress, i,
                                                                                        len(train_dataloader),
                                                                                        np.mean(batch_loss))
                    print('\r' + string, end='')
            train_loss.append(np.mean(batch_loss))
            progress = '=' * (PRINT_LINE_LEN + 1)
            if val_data is not None:
                self.set_image_size(*self.default_image_size)
                val_loss.append(self.calculate_loss(val_data, 2 * batch_size))
                end = time()
                string = 'Epoch: [{}/{}] |{}| [{}/{}] '.format(epoch, epochs, progress, i, len(train_dataloader))
                string += 'Training Loss: {:.6f} Validation Loss: {:.6f} Duration: {:.1f}s'.format(train_loss[-1],
                                                                                                   val_loss[-1],
                                                                                                   end - start)
            else:
                end = time()
                string = 'Epoch: [{}/{}] |{}| [{}/{}] '.format(epoch, epochs, progress, i, len(train_dataloader))
                string += 'Training Loss: {:.6f} Validation Loss: N/A Duration: {:.1f}s'.format(train_loss[-1],
                                                                                                val_loss[-1],
                                                                                                end - start)
            print('\r' + string, end='\n')
            if epoch % checkpoint_frequency == 0:
                self.save_model('yolov2-tiny-{}-bicycle.pkl'.format(epoch))

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
            for i, (images, targets) in enumerate(val_dataloader, 1):
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

                detection_layer = YOLOv3Layer(self, anchors)
                self.detection_layers.append(detection_layer)
                module.add_module('detection_{}'.format(index), detection_layer)

            elif block['type'] == 'region':
                anchors = block['anchors'].split(',')
                anchors = [(float(anchors[i]) / self.grid_size[0], float(anchors[i + 1]) / self.grid_size[1])
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

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn_layer.bias.numel()

                    # Load the weights
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn_layer.bias.data)
                    bn_weights = bn_weights.view_as(bn_layer.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn_layer.running_mean)
                    bn_running_var = bn_running_var.view_as(bn_layer.running_var)

                    # Copy the data to model
                    bn_layer.bias.data.copy_(bn_biases)
                    bn_layer.weight.data.copy_(bn_weights)
                    bn_layer.running_mean.copy_(bn_running_mean)
                    bn_layer.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv_layer.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv_layer.bias.data)

                    # Finally copy the data
                    conv_layer.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv_layer.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_weights)

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

                    # If the parameters are on GPU, convert them back to CPU
                    # We don't convert the parameter to GPU
                    # Instead. we copy the parameter and then convert it to CPU
                    # This is done as weight are need to be saved during training
                    bn_layer.bias.data.detach().cpu().numpy().tofile(f)
                    bn_layer.weight.data.detach().cpu().numpy().tofile(f)
                    bn_layer.running_mean.detach().cpu().numpy().tofile(f)
                    bn_layer.running_var.detach().cpu().numpy().tofile(f)


                else:
                    conv_layer.bias.data.detach().cpu().numpy().tofile(f)

                # Let us save the weights for the Convolutional layers
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

            bboxes[:, 0::2] *= self.image_size[0]
            bboxes[:, 1::2] *= self.image_size[1]

            bboxes = torch.clamp(bboxes, min=0, max=self.image_size[0])
            bboxes = torch.clamp(bboxes, min=0, max=self.image_size[1])

            return bboxes, classes, confidence, image_idx
        else:
            return torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device)

    def predict(self, dataset, batch_size=10, confidence_threshold=0.1, overlap_threshold=0.5, show=True):

        self.eval()

        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=NUM_WORKERS)

        image_idx_ = []
        bboxes_ = []
        confidence_ = []
        classes_ = []

        with torch.no_grad():
            for i, (images, targets) in enumerate(dataloader):
                images = images.to(self.device)
                predictions = self(images)
                # predictions = targets
                bboxes, classes, confidences, image_idx = self.process_bboxes(predictions,
                                                                              confidence_threshold=confidence_threshold,
                                                                              overlap_threshold=overlap_threshold,
                                                                              nms=True)
                if show:
                    for idx, image in enumerate(images):
                        image = to_numpy_image(image)
                        mask = image_idx == idx
                        for bbox, cls, confidence in zip(bboxes[mask], classes[mask], confidences[mask]):
                            name = dataset.classes[cls]
                            add_bbox_to_image(image, bbox, confidence, name)
                        plt.imshow(image)
                        plt.show()

                bboxes_.append(bboxes)
                confidence_.append(confidences)
                classes_.append(classes)
                image_idx_.append(image_idx)

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


def main():
    torch.random.manual_seed(12345)
    np.random.seed(12345)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 10

    train = False

    model = YOLOv2tiny(model='models/yolov2-tiny-voc.cfg',
                       device=device)
    model = model.to(device)

    train_data = PascalDatasetYOLO(root_dir='../data/VOC2012/',
                                   classes='../data/VOC2012/voc.names',
                                   dataset='train',
                                   skip_truncated=False,
                                   skip_difficult=True,
                                   image_size=model.image_size,
                                   grid_size=model.grid_size,
                                   anchors=model.anchors,
                                   transforms=True
                                   )

    val_data = PascalDatasetYOLO(root_dir='../data/VOC2012/',
                                 classes='../data/VOC2012/voc.names',
                                 dataset='val',
                                 skip_truncated=False,
                                 skip_difficult=True,
                                 image_size=model.image_size,
                                 grid_size=model.grid_size,
                                 anchors=model.anchors,
                                 transforms=True
                                 )

    torchsummary.summary(model, (3, 416, 416))
    model.load_weights('models/yolov2-tiny-voc.weights')
    # model.freeze(freeze_last_layer=False)

    if train:
        optimizer = optim.SGD(model.get_trainable_parameters(), lr=1e-5, momentum=0.99)

        model.fit(train_data=train_data,
                  val_data=val_data,
                  optimizer=optimizer,
                  batch_size=batch_size,
                  epochs=100,
                  verbose=True,
                  multi_scale=True,
                  checkpoint_frequency=100)

        model.save_weights('models/yolov2-tiny-focal-bicycle.weights')

    yolo = model
    # yolo = pickle.load(open('yolov2-tiny-100-bicycle.pkl', 'rb'))
    yolo = yolo.to(device)
    image_size = 416, 416
    yolo.set_image_size(*image_size, dataset=(train_data, val_data))

    torch.random.manual_seed(12345)
    np.random.seed(12345)

    yolo.predict(dataset=val_data,
                 batch_size=1,
                 confidence_threshold=0.5,
                 overlap_threshold=0.4,
                 show=True)


if __name__ == '__main__':
    main()
