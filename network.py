import torch
import numpy as np
import cv2
import pickle
import torchsummary
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from utils import BGR_PIXEL_MEANS, PascalDatasetYOLO, jaccard, xywh2xyxy
import time


class YOLOv2tiny(nn.Module):

    def __init__(self, num_classes, grid_size=None, anchors=None, device='cuda', batch_size=10, activation='leaky'):

        super(YOLOv2tiny, self).__init__()

        self.device = device
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_features = 5 + self.num_classes

        if anchors is None:
            self.anchors = torch.tensor(np.array(((0.5, 0.5), (0.3, 0.6), (0.6, 0.3)), dtype=np.float32), device=device)
        else:
            self.anchors = torch.tensor(np.array(anchors, dtype=np.float32), device=device)
        self.num_anchors = len(self.anchors)

        if grid_size is None:
            self.grid_size = (13, 13)
        else:
            self.grid_size = grid_size

        if activation == 'leaky':
            conv_layer = ConvBatchLeaky
        elif activation == 'swish':
            conv_layer = ConvBatchSwish
        else:
            AssertionError

        self.downscale_factor = 32

        self.channels = [3, 16, 32, 64, 128, 256]
        self.layers = nn.ModuleList()
        for in_c, out_c in zip(self.channels, self.channels[1:]):
            self.layers.append(conv_layer(in_channels=in_c, out_channels=out_c))
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.channels.append(512)
        self.layers.append(conv_layer(in_channels=self.channels[-2], out_channels=self.channels[-1]))
        self.layers.append(nn.ReplicationPad2d(padding=(0, 1, 0, 1)))
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=1))
        self.channels.append(1024)
        self.layers.append(conv_layer(in_channels=self.channels[-2], out_channels=self.channels[-1]))
        self.channels.append(512)
        self.layers.append(conv_layer(in_channels=self.channels[-2], out_channels=self.channels[-1]))
        self.channels.append(self.num_anchors * (5 + self.num_classes))
        self.layers.append(nn.Conv2d(in_channels=self.channels[-2], out_channels=self.channels[-1], kernel_size=1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def loss(self, predictions, targets):

        assert predictions.shape == targets.shape

        batch_size = targets.shape[0]

        lambda_coord = 10.
        lambda_noobj = 0.2

        loss = {}

        targets = targets.permute(0, 2, 3, 1)
        predictions = predictions.permute(0, 2, 3, 1)

        targets = targets.contiguous().view(-1, self.num_features)
        predictions = predictions.contiguous().view(-1, self.num_features)

        # Create a mask and compile a tensor only containing detectors that are responsible for objects.
        obj_mask = torch.nonzero(targets[:, 0]).flatten()

        # Convert t_o --> IoU
        predictions[:, 0] = torch.sigmoid(predictions[:, 0])

        if obj_mask.numel() > 0:
            # Convert t_x and t_x --> x and y (ignoring the offset).
            predictions[:, 1:3] = torch.sigmoid(predictions[:, 1:3])
            # Add the offset.
            offsets = torch.arange(0, predictions.shape[0], device=self.device)
            h_offsets = offsets / self.grid_size[0] / self.num_anchors
            v_offsets = (offsets - (h_offsets * self.grid_size[0] * self.num_anchors)) / self.num_anchors
            predictions[:, 1] += h_offsets.float()
            predictions[:, 2] += v_offsets.float()
            predictions[:, 1] /= self.grid_size[0]
            predictions[:, 2] /= self.grid_size[1]

            # Convert t_w and t_h --> w and h.
            anchors = self.anchors.repeat(batch_size * self.grid_size[0] * self.grid_size[1], 1)

            predictions[:, 3] = anchors[:, 0] * torch.exp(predictions[:, 3])
            predictions[:, 4] = anchors[:, 1] * torch.exp(predictions[:, 4])
            # Add softmax to class probabilities.
            # NB!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            predictions[obj_mask, 5:] = torch.sigmoid(predictions[obj_mask, 5:])

            predictions_xyxy = xywh2xyxy(predictions[:, 1:5])
            targets_xyxy = xywh2xyxy(targets[obj_mask, 1:5])

            ious = jaccard(predictions_xyxy, targets_xyxy)
            ious, _ = torch.max(ious, dim=1)

            loss['object'] = nn.MSELoss(reduction='sum')(predictions[obj_mask, 0], ious[obj_mask])
            loss['coord'] = nn.MSELoss(reduction='sum')(predictions[obj_mask, 1], targets[obj_mask, 1])
            loss['coord'] += nn.MSELoss(reduction='sum')(predictions[obj_mask, 2], targets[obj_mask, 2])
            loss['coord'] += nn.MSELoss(reduction='sum')(torch.sqrt(predictions[obj_mask, 3]), torch.sqrt(targets[obj_mask, 3]))
            loss['coord'] += nn.MSELoss(reduction='sum')(torch.sqrt(predictions[obj_mask, 4]), torch.sqrt(targets[obj_mask, 4]))
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

            loss['no_object'] = lambda_noobj * nn.MSELoss(reduction='sum')(predictions[noobj_mask, 0], targets[noobj_mask, 0])
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

    def fit(self, train_data, optimizer, batch_size=1, epochs=1, verbose=True, val_data=None, shuffle=True, multi_scale=True, checkpoint_frequency=100):
        self.train()

        train_dataloader = DataLoader(dataset=train_data,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      num_workers=0)

        train_loss = []
        val_loss = []

        for epoch in range(1, epochs + 1):
            batch_loss = []
            if multi_scale:
                random_size = np.random.randint(10, 20) * self.downscale_factor
                train_data.set_image_size(*[random_size] * 2)
                train_data.set_grid_size(*[int(random_size / self.downscale_factor)] * 2)
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
                    line_len = 52
                    ii = int(i / len(train_dataloader) * line_len)
                    progress = '=' * ii
                    progress += '>'
                    progress += ' ' * (line_len - ii)
                    print('\rEpoch: [{}/{}] |{}| [{}/{}] Training Loss: {:.6f}'.format(epoch, epochs, progress, i, len(train_dataloader), np.mean(batch_loss)), end='')
            train_loss.append(np.mean(batch_loss))
            progress = '=' * line_len
            if val_data is not None:
                val_loss.append(self.calculate_loss(val_data, batch_size))
                print('\rEpoch: [{}/{}] |{}| [{}/{}] Training Loss: {:.6f} Validation Loss: {:.6f}'.format(epoch, epochs, progress, i, len(train_dataloader), train_loss[-1], val_loss[-1]), end='\n')
            else:
                print('\rEpoch: [{}/{}] |{}| [{}/{}] Training Loss: {:.6f} Validation Loss: N/A'.format(epoch, epochs, progress, i, len(train_dataloader), train_loss[-1]), end='\n')
            if epoch % checkpoint_frequency == 0:
                self.save_model('yolo_{}_leaky.pkl'.format(epoch))

        return train_loss, val_loss

    def set_grid_size(self, x, y):
        self.grid_size = x, y

    def save_model(self, name):
        pickle.dump(self, open(name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    def calculate_loss(self, data, batch_size):
        val_dataloader = DataLoader(dataset=data,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=0)
        loss = []
        with torch.no_grad():
            for images, targets in val_dataloader:
                images = images.to(self.device)
                targets = targets.to(self.device)
                predictions = self(images)
                loss = self.loss(predictions, targets)
                loss.append(loss['total'].item())

        return np.mean(loss)


class ConvBatchLeaky(nn.Module):
    """ This convenience layer groups a 2D convolution, a batchnorm and a leaky ReLU.
    They are executed in a sequential manner.
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the kernel of the convolution
        stride (int): Stride of the convolution
        padding (int): padding of the convolution
        leaky_slope (number, optional): Controls the angle of the negative slope of the leaky ReLU; Default **0.1**
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, leaky_slope=0.1):
        super(ConvBatchLeaky, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = int(kernel_size / 2)
        self.leaky_slope = leaky_slope

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding,
                      bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(self.leaky_slope, inplace=True),
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, negative_slope={leaky_slope})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x


class ConvBatchSwish(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBatchLeaky, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = int(kernel_size / 2)

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding,
                      bias=False),
            nn.BatchNorm2d(self.out_channels),
            Swish()
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, negative_slope={leaky_slope})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    anchors = ((0.3, 0.3), (0.2, 0.6), (0.6, 0.2), (0.3, 0.8))
    classes = ['bicycle', 'person', 'car']
    image_size = (416, 416)
    grid_size = (13, 13)
    batch_size = 8

    train_data = PascalDatasetYOLO(root_dir='../data/VOC2012/',
                                   classes=classes,#, 'car', 'cat', 'person', 'train', 'tvmonitor'],
                                   dataset='train',
                                   skip_truncated=False,
                                   skip_difficult=False,
                                   image_size=image_size,
                                   grid_size=grid_size,
                                   anchors=anchors
                                   )

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0)

    val_data = PascalDatasetYOLO(root_dir='../data/VOC2012/',
                                   classes=classes,#, 'car', 'cat', 'person', 'train', 'tvmonitor'],
                                   dataset='train',
                                   skip_truncated=False,
                                   skip_difficult=False,
                                   image_size=image_size,
                                   grid_size=grid_size,
                                   anchors=anchors
                                   )

    torch.random.manual_seed(12345)
    np.random.seed(12345)

    model = YOLOv2tiny(num_classes=len(classes),
                       grid_size=grid_size,
                       anchors=anchors,
                       batch_size=batch_size,
                       device=device)

    model = model.to(device)

    torchsummary.summary(model, (3, 416, 416))

    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.98)
    # optimizer = optim.Adam(yolo.parameters())

    # model.fit(train_data=train_data,
    #           val_data=val_data,
    #           optimizer=optimizer,
    #           batch_size=10,
    #           epochs=100,
    #           verbose=True,
    #           checkpoint_frequency=100)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    # anchors = ((0.3, 0.3), (0.2, 0.6), (0.6, 0.2), (0.3, 0.8))
    # classes = ['bicycle', 'person', 'car']
    # batch_size = 8
    #
    # train_dataset = PascalDatasetYOLO(root_dir='../data/VOC2012/',
    #                                   classes=classes,  # , 'car', 'cat', 'person', 'train', 'tvmonitor'],
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
    #                   device=device,
    #                   swish=True)
    #
    # yolo = yolo.to(device)
    # yolo.train()
    #
    # torchsummary.summary(yolo, (3, 416, 416))
    #
    # optimizer = optim.SGD(yolo.parameters(), lr=5e-5, momentum=0.98)
    # # optimizer = optim.Adam(yolo.parameters())
    #
    # losses = []
    # n = 2000
    # for epoch in range(1, n + 1):
    #     random_size = np.random.randint(10, 20) * 32
    #     yolo.dataset.image_size = tuple([random_size] * 2)
    #     yolo.dataset.grid_size = tuple([int(random_size / 32)] * 2)
    #     yolo.grid_size = yolo.dataset.grid_size
    #     print(random_size)
    #     for i, (images, targets) in enumerate(yolo.dataloader, 1):
    #         images = images.to(device)
    #         targets = targets.to(device)
    #         optimizer.zero_grad()
    #         predictions = yolo(images)
    #         loss = yolo.loss(predictions, targets)
    #         losses.append(loss['total'].item())
    #         loss['total'].backward()
    #         optimizer.step()
    #         print('Train Epoch: {} [{}/{}] Loss: {:.6f}'.format(
    #             epoch, i, int(np.ceil(len(train_dataset) / yolo.batch_size)), np.mean(losses[-5:])))
    #     if epoch % 1000 == 0:
    #         pickle.dump(yolo, open('yolo_{}_swish.pkl'.format(epoch), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    #         pickle.dump(losses, open('losses_{}_swish.pkl'.format(epoch), 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    yolo = pickle.load(open('yolo_100_leaky.pkl', 'rb'))
    train_data.set_image_size(416, 416)
    train_data.set_grid_size(13, 13)
    yolo.set_grid_size(*train_data.grid_size)

    yolo.eval()

    torch.random.manual_seed(12345)
    np.random.seed(12345)

    with torch.no_grad():
        for epoch in range(1):
            for i, data in enumerate(train_dataloader):
                images, targets = data
                images = images.to(device)
                for image, target in zip(images, targets):
                    # try:
                    #     threshold = float(input('Input a threshold:'))
                    # except:
                    #     threshold = 0.
                    threshold = 0.001

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
                                if torch.sigmoid(predictions[x, y, d, 0]) > threshold:
                                    annotation = []
                                    cls = train_data.classes[torch.argmax(predictions[x, y, d, 5:])]
                                    annotation.append(cls)
                                    xywh = torch.zeros(4)
                                    xywh[0] = (torch.sigmoid(predictions[x, y, d, 1]) + x) / yolo.grid_size[0]
                                    xywh[1] = (torch.sigmoid(predictions[x, y, d, 2]) + y) / yolo.grid_size[1]
                                    xywh[2] = (yolo.anchors[d, 0] * torch.exp(predictions[x, y, d, 3]))
                                    xywh[3] = (yolo.anchors[d, 1] * torch.exp(predictions[x, y, d, 4]))
                                    xyxy = xywh2xyxy(xywh[np.newaxis]).cpu().numpy()
                                    for i in range(4):
                                        annotation.append(int(xyxy[0, i] * train_data.image_size[i % 2]))
                                    print(x, y, d, torch.sigmoid(predictions[x, y, d, 0]))
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
                                    cv2.putText(image, '{} {:.2f}'.format(name, torch.sigmoid(predictions[x, y, d, 0])), (xmin, ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255.-BGR_PIXEL_MEANS, 1)
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
