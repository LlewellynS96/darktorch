import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms
from utils import jaccard, read_classes, get_annotations, get_letterbox_padding

IOU_MATCH_THRESHOLD = 0.1


class PascalDatasetYOLO(Dataset):
    """
    This object can be configured to return images and annotations
    from a PASCAL VOC dataset in a format that is compatible for training
    the YOLOv2 object detector.
    """

    def __init__(self, anchors, class_file, root_dir='data/VOC2012/', dataset='train', skip_truncated=True,
                 do_transforms=False, skip_difficult=True, image_size=(416, 416), grid_size=(13, 13)):
        """
        Initialise the dataset object with some network and dataset specific parameters.

        Parameters
        ----------
        anchors : Tensor
                A Tensor object of N anchors given by (x1, y1, x2, y2).
        class_file : str
                The path to a text file containing the names of the different classes that
                should be loaded.
        root_dir : str, optional
                The root directory where the PASCAL VOC images and annotations are stored.
        dataset : {'train', 'val', 'trainval', 'test}, optional
                The specific subset of the PASCAL VOC challenge which should be loaded.
        skip_truncated : bool,  optional
                A boolean value to specify whether bounding boxes should be skipped or
                returned for objects that are truncated.
        do_transforms : bool, optional
                A boolean value to determine whether default image augmentation transforms
                should be randomly applied to images.
        skip_difficult : bool,  optional
                A boolean value to specify whether bounding boxes should be skipped or
                returned for objects that have been labeled as 'difficult'.
        image_size : tuple of int, optional
                A tuple (w, h) describing the desired width and height of the images to
                be returned.
        grid_size : tuple of int, optional
                A tuple (x, y) describing how many horizontal and vertical grids comprise
                the YOLO model that will load images from this dataset.
        """
        assert image_size[0] == image_size[1], 'This implementation has only been validated for square images.'
        assert grid_size[0] == grid_size[1], 'This implementation has only been validated for square images.'

        self.classes = read_classes(class_file)

        assert dataset in ['train', 'val', 'trainval', 'test']

        self.num_classes = len(self.classes)
        self.num_features = 5 + self.num_classes

        self.root_dir = root_dir
        self.images_dir = os.path.join(self.root_dir, 'JPEGImages/')
        self.annotations_dir = os.path.join(self.root_dir, 'Annotations')
        self.sets_dir = os.path.join(self.root_dir, 'ImageSets', 'Main')
        self.dataset = dataset

        self.images = []
        self.skip_truncated = skip_truncated
        self.skip_difficult = skip_difficult
        self.image_size = image_size
        self.grid_size = grid_size
        self.anchors = anchors.clone().detach().cpu()
        self.num_anchors = len(self.anchors)

        self.do_transforms = do_transforms

        for cls in self.classes:
            file = os.path.join(self.sets_dir, '{}_{}.txt'.format(cls, dataset))
            with open(file) as f:
                for line in f:
                    image_desc = line.split()
                    if image_desc[1] == '1':
                        self.images.append(image_desc[0])

        self.images = list(set(self.images))  # Remove duplicates.
        self.images.sort()

    def __getitem__(self, index):
        """
        Return some image with its meta information and labeled annotations.

        Parameters
        ----------
        index : int
            The index of the image to be returned.

        Returns
        -------
        image : Tensor
            The image at self.images[index] after some optional transforms have been
            performed as an (w, h, 3) Tensor in the range [0., 1.].
        image_info : dict
            A dictionary object containing meta information about the image.
        target : Tensor
            A Tensor representing the target output of the YOLOv2 network which was
            used to initialise the dataset object.

        """
        img = self.images[index]
        image = Image.open(os.path.join(self.images_dir, img + '.jpg'))
        image_info = {'id': img, 'width': image.width, 'height': image.height, 'dataset': self.dataset}
        random_flip = np.random.random()
        if self.do_transforms:
            image = torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.25, hue=0.05)(image)
            oversize = 0.2
            image_resize = np.array([image_info['width'], image_info['height']] * np.array((1. + oversize)),
                                    dtype=np.int)
            crop_offset = np.random.random(size=2) * oversize * [image_info['width'], image_info['height']]
            crop_offset = crop_offset.astype(dtype=np.int)
            image = image.resize(image_resize)
            if random_flip >= 0.5:
                image = torchvision.transforms.functional.hflip(image)
            image = torchvision.transforms.functional.crop(image,
                                                           *crop_offset,
                                                           image_info['height'], image_info['width'])
        ratio, pad = get_letterbox_padding(image.size, self.image_size)
        image = image.resize([int(dim * ratio) for dim in image.size])
        image = ImageOps.expand(image, pad, fill="black")
        image_info['padding'] = pad

        assert image.size == self.image_size

        annotations = get_annotations(self.annotations_dir, img)
        target = np.zeros((self.grid_size[1], self.grid_size[0], self.num_anchors * self.num_features),
                          dtype=np.float32)
        # For each object in image.
        for annotation in annotations:
            name, height, width, xmin, ymin, xmax, ymax, truncated, difficult = annotation
            if (self.skip_truncated and truncated) or (self.skip_difficult and difficult):
                continue
            if name not in self.classes:
                continue
            cell_dims = self.image_size[1] / self.grid_size[1], self.image_size[0] / self.grid_size[0]
            if self.do_transforms:
                if random_flip >= 0.5:
                    tmp = xmin
                    xmin = width - xmax
                    xmax = width - tmp
                xmin = (xmin * image_resize[0] - crop_offset[0]) / image_info['width'] * self.image_size[0] + pad[0]
                xmax = (xmax * image_resize[0] - crop_offset[0]) / image_info['width'] * self.image_size[0] + pad[0]
                ymin = (ymin * image_resize[1] - crop_offset[1]) / image_info['height'] * self.image_size[1] + pad[1]
                ymax = (ymax * image_resize[1] - crop_offset[1]) / image_info['height'] * self.image_size[1] + pad[1]
                xmin = np.clip(xmin, a_min=pad[0], a_max=self.image_size[0] - pad[2])
                xmax = np.clip(xmax, a_min=pad[0], a_max=self.image_size[0] - pad[2])
                ymin = np.clip(ymin, a_min=pad[1], a_max=self.image_size[1] - pad[3])
                ymax = np.clip(ymax, a_min=pad[1], a_max=self.image_size[1] - pad[3])
                if xmax == xmin or ymax == ymin:
                    continue
            else:
                xmin = xmin * ratio + pad[0]
                xmax = xmax * ratio + pad[0]
                ymin = ymin * ratio + pad[1]
                ymax = ymax * ratio + pad[1]
                xmin = np.clip(xmin, a_min=pad[0], a_max=self.image_size[0] - pad[2])
                xmax = np.clip(xmax, a_min=pad[0], a_max=self.image_size[0] - pad[2])
                ymin = np.clip(ymin, a_min=pad[1], a_max=self.image_size[1] - pad[3])
                ymax = np.clip(ymax, a_min=pad[1], a_max=self.image_size[1] - pad[3])
                if xmax == xmin or ymax == ymin:
                    continue
            xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
            idx = int(np.floor((xmax + xmin) / 2. / cell_dims[0])), int(np.floor((ymax + ymin) / 2. / cell_dims[1]))
            if target[idx[1], idx[0], 4::self.num_features].all() == 0:
                ground_truth = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
                anchors = torch.zeros((self.num_anchors, 4))
                anchors[:, 2:] = self.anchors.clone()
                anchors[:, 0::2] += xmin
                anchors[:, 1::2] += ymin
                ious = jaccard(ground_truth, anchors)
                if ious.max() < IOU_MATCH_THRESHOLD:
                    continue
                assign = np.argmax(ious)
                target[idx[1], idx[0], assign * self.num_features + 0] = (xmin + xmax) / 2.
                target[idx[1], idx[0], assign * self.num_features + 1] = (ymin + ymax) / 2.
                target[idx[1], idx[0], assign * self.num_features + 2] = xmax - xmin
                target[idx[1], idx[0], assign * self.num_features + 3] = ymax - ymin
                target[idx[1], idx[0], assign * self.num_features + 4] = 1.
                target[idx[1], idx[0], assign * self.num_features + 5:(assign + 1) * self.num_features] = \
                    self.encode_categorical(name)
            else:
                print('One cell, two objects.')

        image = torchvision.transforms.ToTensor()(image)

        target = torch.tensor(target)
        target = target.permute(2, 0, 1)

        return image, image_info, target

    def __len__(self):

        return len(self.images)

    def encode_categorical(self, name):
        y = self.classes.index(name)
        yy = np.zeros(self.num_classes)
        yy[y] = 1
        return yy

    def set_image_size(self, x, y):
        self.image_size = x, y

    def set_grid_size(self, x, y):
        self.grid_size = x, y

    def set_anchors(self, anchors):
        self.anchors = anchors.clone().detach().cpu()
