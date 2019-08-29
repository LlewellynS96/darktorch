import numpy as np
import cv2
import pickle
import torchsummary
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
from time import time
from utils import BGR_PIXEL_MEANS, PRINT_LINE_LEN, NUM_WORKERS, PascalDatasetYOLO, jaccard, xywh2xyxy
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
        self.image_size = int(self.net_info['width']), int(self.net_info['height'])
        self.downscale_factor = self.get_downscale_factor()
        self.grid_size = self.get_grid_size()

        self.build_modules()

        self.anchors = self.get_anchors()
        self.num_anchors = len(self.anchors)

    def forward(self, x):
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
        obj_mask = torch.nonzero(targets[:, 0]).flatten()

        if obj_mask.numel() > 0:
            # Convert t_w and t_h --> w and h.
            predictions_xyxy = xywh2xyxy(predictions[:, 1:5])
            targets_xyxy = xywh2xyxy(targets[obj_mask, 1:5])

            ious = jaccard(predictions_xyxy, targets_xyxy)
            ious, _ = torch.max(ious, dim=1)

            loss['object'] = lambda_obj * nn.MSELoss(reduction='sum')(predictions[obj_mask, 0], ious[obj_mask])
            loss['coord'] = nn.MSELoss(reduction='sum')(predictions[obj_mask, 1], targets[obj_mask, 1])
            loss['coord'] += nn.MSELoss(reduction='sum')(predictions[obj_mask, 2], targets[obj_mask, 2])
            loss['coord'] += nn.MSELoss(reduction='sum')(torch.sqrt(predictions[obj_mask, 3]),
                                                         torch.sqrt(targets[obj_mask, 3]))
            loss['coord'] += nn.MSELoss(reduction='sum')(torch.sqrt(predictions[obj_mask, 4]),
                                                         torch.sqrt(targets[obj_mask, 4]))
            # Multiply by lambda_coord
            loss['coord'] *= lambda_coord
            # Divide by the number of separate loss components.
            loss['class'] = 0.
            for cls in range(self.num_classes):
                loss['class'] += nn.MSELoss(reduction='sum')(predictions[obj_mask, 5 + cls], targets[obj_mask, 5 + cls])

            threshold = 0.6
            noobj_mask = torch.nonzero(ious > threshold).squeeze()
            targets[noobj_mask, 0] = 2.
            noobj_mask = torch.nonzero(targets[:, 0] == 0.).squeeze()

            loss['no_object'] = lambda_noobj * nn.MSELoss(reduction='sum')(predictions[noobj_mask, 0],
                                                                           targets[noobj_mask, 0])
        else:
            loss['object'] = torch.tensor([0.], device=self.device)
            loss['coord'] = torch.tensor([0.], device=self.device)
            loss['class'] = torch.tensor([0.], device=self.device)

            loss['no_object'] = lambda_noobj * nn.MSELoss(reduction='sum')(predictions[:, 0], targets[:, 0])

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
                train_data.set_image_size(*[random_size] * 2)
                train_data.set_grid_size(*[int(random_size / self.downscale_factor)] * 2)
                self.set_image_size(*train_data.image_size)
                self.set_grid_size(*train_data.grid_size)
            for i, (images, targets) in enumerate(train_dataloader, 1):
                # image = images[0].permute(1, 2, 0).numpy()
                # image += BGR_PIXEL_MEANS
                # image = cv2.cvtColor(image.astype(dtype=np.uint8), cv2.COLOR_BGR2RGB)
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
                self.set_grid_size(13, 13)
                val_loss.append(self.calculate_loss(val_data, batch_size))
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
                self.save_model('yolov2_tiny_{}.pkl'.format(epoch))

        return train_loss, val_loss

    def set_grid_size(self, x, y):
        self.grid_size = x, y
        for layer in self.detection_layers:
            layer.grid_size = x, y

    def set_image_size(self, x, y):
        self.image_size = x, y

    def save_model(self, name):
        pickle.dump(self, open(name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def calculate_loss(self, data, batch_size, fraction=0.2):
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
        lines = file.read().split('\n')  # store the lines in a list
        lines = [l for l in lines if len(l) > 0]
        lines = [l for l in lines if l[0] != '#']
        lines = [l.rstrip().lstrip() for l in lines]

        block = {}
        self.blocks = []

        for line in lines:
            if line[0] == '[':  # This marks the start of a new block
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

        index = 0  # indexing blocks helps with implementing route  layers (skip connections)
        prev_filters = 3
        output_filters = []

        for block in self.blocks:
            module = nn.Sequential()

            # If it's a convolutional layer
            if block['type'] == 'convolutional':
                # Get the info about the layer
                activation = block['activation']
                try:
                    batch_normalize = bool(block['batch_normalize'])
                    bias = False
                except:
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

                # Add the convolutional layer
                conv_layer = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
                module.add_module('conv_{0}'.format(index), conv_layer)

                # Add the Batch Norm Layer
                if batch_normalize:
                    bn_layer = nn.BatchNorm2d(filters)
                    module.add_module('batch_norm_{0}'.format(index), bn_layer)

                # Check the activation.
                # It is either Linear or a Leaky ReLU for YOLO
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

                # Start  of a route
                start = int(block['layers'][0])

                # end, if there exists one.
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
                module.add_module()

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

    def load_weights(self, weightfile):

        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 4 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4. IMages seen
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        # The rest of the values are the weights
        # Let's load them up
        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i + 1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]

                if (batch_normalize):
                    bn = model[1]

                    # Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()

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
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Let us load the weights for the Convolutional layers
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

    def get_downscale_factor(self):
        downscale_factor = 1.
        for block in self.blocks:
            if block['type'] == 'maxpool':
                size = int(block['size'])
                stride = int(block['stride'])
                div = (1. - size) / stride + 1.
                if div > 0:
                    downscale_factor /= div
        return int(downscale_factor)

    def get_anchors(self):
        anchors = []
        for layer in self.detection_layers:
            anchors.append(layer.anchors)
        return torch.cat(anchors)

    def get_grid_size(self):
        width, height = self.image_size

        assert width % self.downscale_factor == 0
        assert height % self.downscale_factor == 0

        return int(width / self.downscale_factor), int(height / self.downscale_factor)


def main():
    torch.random.manual_seed(12345)
    np.random.seed(12345)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 10

    model = YOLOv2tiny(model='models/yolov2-tiny.cfg',
                       device=device)
    model = model.to(device)

    # torchsummary.summary(model, (3, 416, 416))

    train_data = PascalDatasetYOLO(root_dir='../data/VOC2012/',
                                   classes='../data/VOC2012/voc.names',
                                   dataset='train',
                                   skip_truncated=False,
                                   skip_difficult=False,
                                   image_size=model.image_size,
                                   grid_size=model.grid_size,
                                   anchors=model.anchors
                                   )

    val_data = PascalDatasetYOLO(root_dir='../data/VOC2012/',
                                 classes='../data/VOC2012/voc.names',
                                 dataset='val',
                                 skip_truncated=False,
                                 skip_difficult=True,
                                 image_size=model.image_size,
                                 grid_size=model.grid_size,
                                 anchors=model.anchors
                                 )

    optimizer = optim.SGD(model.parameters(), lr=5e-5, momentum=0.98)
    # optimizer = optim.Adam(model.parameters())

    # model.fit(train_data=train_data,
    #           val_data=val_data,
    #           optimizer=optimizer,
    #           batch_size=batch_size,
    #           epochs=500,
    #           verbose=True,
    #           multi_scale=False,
    #           checkpoint_frequency=100)

    # yolo = model
    yolo = pickle.load(open('yolov2_tiny_500.pkl', 'rb'))
    train_data.set_image_size(416, 416)
    train_data.set_grid_size(13, 13)
    yolo.set_image_size(*train_data.image_size)
    yolo.set_grid_size(*train_data.grid_size)

    yolo.eval()

    dataloader = DataLoader(dataset=val_data,
                            batch_size=1,
                            shuffle=True,
                            num_workers=NUM_WORKERS)

    torch.random.manual_seed(12345)
    np.random.seed(12345)

    with torch.no_grad():
        for epoch in range(1):
            for i, data in enumerate(dataloader):
                images, targets = data
                images = images.to(device)
                for image, target in zip(images, targets):
                    # try:
                    #     threshold = float(input('Input a threshold:'))
                    # except:
                    #     threshold = 0.
                    threshold = 0.3

                    annotations = []
                    predictions = yolo(image[np.newaxis])
                    # predictions = target[np.newaxis]
                    image = image.cpu().numpy().astype(dtype=np.float64)
                    image = np.ascontiguousarray(image.transpose(1, 2, 0))
                    predictions = predictions.permute(0, 2, 3, 1)
                    predictions = predictions[0]
                    predictions = predictions.reshape(yolo.grid_size[0], yolo.grid_size[1], yolo.num_anchors, yolo.num_features)
                    print('-' * 20)
                    for x in range(yolo.grid_size[0]):
                        for y in range(yolo.grid_size[1]):
                            for d in range(yolo.num_anchors):
                                if predictions[x, y, d, 0] > threshold:
                                    annotation = []
                                    cls = train_data.classes[torch.argmax(predictions[x, y, d, 5:])]
                                    annotation.append(cls)
                                    xywh = torch.zeros(4)
                                    xywh[0] = predictions[x, y, d, 1]
                                    xywh[1] = predictions[x, y, d, 2]
                                    xywh[2] = predictions[x, y, d, 3]
                                    xywh[3] = predictions[x, y, d, 4]
                                    xyxy = xywh2xyxy(xywh[np.newaxis]).cpu().numpy()
                                    for i in range(4):
                                        annotation.append(int(xyxy[0, i] * train_data.image_size[i % 2]))
                                    print(x, y, d, predictions[x, y, d, 0])
                                    name, xmin, ymin, xmax, ymax = annotation
                                    # Draw a bounding box.
                                    color = np.random.uniform(0., 255., size=3)
                                    cv2.rectangle(image, (xmin, ymax), (xmax, ymin), color, 3)

                                    # Display the label at the top of the bounding box
                                    label_size, base_line = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                                    ymax = max(ymax, label_size[1])
                                    cv2.rectangle(image,
                                                  (xmin, ymax - round(1.5 * label_size[1])),
                                                  (xmin + round(1.5 * label_size[0]),
                                                   ymax + base_line),
                                                  color,
                                                  cv2.FILLED)
                                    cv2.putText(image, '{} {:.2f}'.format(name, predictions[x, y, d, 0]), (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255.-BGR_PIXEL_MEANS, 1)
                    # Convert BGR --> RGB
                    image += BGR_PIXEL_MEANS
                    image = cv2.cvtColor(image.astype(dtype=np.uint8), cv2.COLOR_BGR2RGB)
                    plt.imshow(image)
                    plt.show()


if __name__ == '__main__':
    main()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # anchors = ((0.3, 0.3), (0.2, 0.6), (0.6, 0.2), (0.3, 0.8))
    # classes = ['bicycle']
    # batch_size = 8
    #
    # train_dataset = PascalDatasetYOLO(root_dir='../data/VOC2012/',
    #                                   classes=classes,#, 'car', 'cat', 'person', 'train', 'tvmonitor'],
    #                                   dataset='train',
    #                                   skip_truncated=False,
    #                                   skip_difficult=False,
    #                                   image_size=(416, 416),
    #                                   grid_size=(13, 13),
    #                                   anchors=anchors)
    #
    # torch.random.manual_seed(12345)
    # np.random.seed(12345)
    #
    # yolo = YOLOv2tiny(dataset=train_dataset,
    #                   anchors=anchors,
    #                   batch_size=batch_size,
    #                   device=device)
    #
    # yolo = yolo.to(device)
    #
    # r = range(1000)
    #
    # predictions = None
    # targets = None
    # for i, data in enumerate(yolo.dataloader):
    #     images, targets = data
    #     images = images.to(device)
    #     targets = targets.to(device)
    #     predictions = yolo(images)
    #     break
    #
    # start = time()
    # for i in r:
    #     yolo.loss(predictions, targets)
    # print((time()-start))
