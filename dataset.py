import os
import cv2
import torch
import numpy as np
import xml.etree.ElementTree as Et
from torch.utils.data import Dataset
from utils import BGR_PIXEL_MEANS
from utils import jaccard, read_classes
from PIL import Image


class PascalDatasetYOLO(Dataset):

    def __init__(self, anchors, classes, root_dir='data/VOC2012/', dataset='train', skip_truncated=True,
                 transforms=None, skip_difficult=True, image_size=(416, 416), grid_size=(13, 13)):

        self.classes = read_classes(classes)

        assert set(self.classes).issubset({'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                                           'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                                           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'})

        assert dataset in ['train', 'val', 'trainval']

        self.anchors = anchors.clone().detach().cpu()
        self.num_classes = len(self.classes)
        self.num_anchors = len(self.anchors)
        self.num_features = 5 + self.num_classes

        self.root_dir = root_dir
        self.images_dir = os.path.join(self.root_dir, 'JPEGImages/')
        self.annotations_dir = os.path.join(self.root_dir, 'Annotations')
        self.sets_dir = os.path.join(self.root_dir, 'ImageSets', 'Main')

        self.images = []
        self.skip_truncated = skip_truncated
        self.skip_difficult = skip_difficult
        self.image_size = image_size
        self.grid_size = grid_size

        self.transforms = transforms

        for cls in self.classes:
            file = os.path.join(self.sets_dir, '{}_{}.txt'.format(cls, dataset))
            with open(file) as f:
                for line in f:
                    image_desc = line.split()
                    if image_desc[1] == '1':
                        self.images.append(image_desc[0])

    # def __getitem__(self, index):
    #
    #     img = self.images[index]
    #     image = cv2.imread(os.path.join(self.images_dir, img + '.jpg'), 1)
    #     annotations = self.get_annotations(img)
    #     target = np.zeros((self.grid_size[0], self.grid_size[1], self.num_anchors * self.num_features), dtype=np.float32)
    #     # For each object in image.
    #     for annotation in annotations:
    #         name, xmin, ymin, xmax, ymax, _, _ = annotation
    #         if (self.skip_truncated and annotation[5]) or (self.skip_difficult and annotation[6]):
    #             continue
    #         if name not in self.classes:
    #             continue
    #         cell_dims = 1. / self.grid_size[0], 1. / self.grid_size[1]
    #         idx = int(np.floor((xmax + xmin) / 2. / cell_dims[0])), int(np.floor((ymax + ymin) / 2. / cell_dims[1]))
    #         if target[idx[0], idx[1], ::self.num_features].all() == 0:
    #             ground_truth = torch.tensor(np.array([[xmin, ymin, xmax, ymax]], dtype=np.float32))
    #             anchors = torch.zeros((self.num_anchors, 4))
    #             anchors[:, 2:] = self.anchors.clone()
    #             anchors[:, 0::2] += xmin
    #             anchors[:, 1::2] += ymin
    #             ious = jaccard(ground_truth, anchors)
    #             assign = np.argmax(ious)
    #             target[idx[0], idx[1], assign * self.num_features + 4] = 1.
    #             target[idx[0], idx[1], assign * self.num_features + 0] = (xmin + xmax) / 2.
    #             target[idx[0], idx[1], assign * self.num_features + 1] = (ymin + ymax) / 2.
    #             target[idx[0], idx[1], assign * self.num_features + 2] = xmax - xmin
    #             target[idx[0], idx[1], assign * self.num_features + 3] = ymax - ymin
    #             target[idx[0], idx[1], assign * self.num_features + 5:(assign + 1) * self.num_features] = self.encode_categorical(name)
    #
    #     image = cv2.resize(image, self.image_size)
    #     # Subtract the mean pixel values (across the training set) to zero-mean the image.
    #     image = image.astype(dtype=np.float32)
    #     image -= BGR_PIXEL_MEANS
    #
    #     image = torch.tensor(image)
    #     image = image.permute(2, 0, 1)
    #     target = torch.tensor(target)
    #     target = target.permute(2, 0, 1)
    #
    #     if self.transforms is not None:
    #         image = self.transforms(image)
    #
    #     return image, target

    def __getitem__(self, index):

        img = self.images[index]
        image = Image.open(os.path.join(self.images_dir, img + '.jpg'))
        annotations = self.get_annotations(img)
        target = np.zeros((self.grid_size[0], self.grid_size[1], self.num_anchors * self.num_features),
                          dtype=np.float32)
        # For each object in image.
        for annotation in annotations:
            name, xmin, ymin, xmax, ymax, _, _ = annotation
            if (self.skip_truncated and annotation[5]) or (self.skip_difficult and annotation[6]):
                continue
            if name not in self.classes:
                continue
            cell_dims = 1. / self.grid_size[0], 1. / self.grid_size[1]
            idx = int(np.floor((xmax + xmin) / 2. / cell_dims[0])), int(np.floor((ymax + ymin) / 2. / cell_dims[1]))
            if target[idx[0], idx[1], ::self.num_features].all() == 0:
                ground_truth = torch.tensor(np.array([[xmin, ymin, xmax, ymax]], dtype=np.float32))
                anchors = torch.zeros((self.num_anchors, 4))
                anchors[:, 2:] = self.anchors.clone()
                anchors[:, 0::2] += xmin
                anchors[:, 1::2] += ymin
                ious = jaccard(ground_truth, anchors)
                assign = np.argmax(ious)
                target[idx[0], idx[1], assign * self.num_features + 4] = 1.
                target[idx[0], idx[1], assign * self.num_features + 0] = (xmin + xmax) / 2.
                target[idx[0], idx[1], assign * self.num_features + 1] = (ymin + ymax) / 2.
                target[idx[0], idx[1], assign * self.num_features + 2] = xmax - xmin
                target[idx[0], idx[1], assign * self.num_features + 3] = ymax - ymin
                target[idx[0], idx[1], assign * self.num_features + 5:(assign + 1) * self.num_features] = self.encode_categorical(name)

        image = image.resize(self.image_size)

        target = torch.tensor(target)
        target = target.permute(2, 0, 1)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def __len__(self):

        return len(self.images)

    def get_annotations(self, img):

        file = os.path.join(self.annotations_dir, img + '.xml')
        tree = Et.parse(file)
        root = tree.getroot()

        annotations = []

        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            name = obj.find('name').text
            truncated = int(obj.find('truncated').text)
            difficult = int(obj.find('difficult').text)
            # Get ground truth bounding boxes.
            # NOTE: The creators of the Pascal VOC dataset started counting at 1,
            # and thus the indices have to be corrected.
            xmin = (float(bbox.find('xmin').text) - 1.) / width
            xmax = (float(bbox.find('xmax').text) - 1.) / width
            ymin = (float(bbox.find('ymin').text) - 1.) / height
            ymax = (float(bbox.find('ymax').text) - 1.) / height
            annotations.append((name, xmin, ymin, xmax, ymax, truncated, difficult))

        return annotations

    def encode_categorical(self, name):
        y = self.classes.index(name)
        yy = np.zeros(self.num_classes)
        yy[y] = 1
        return yy

    def set_image_size(self, x, y):
        self.image_size = x, y

    def set_grid_size(self, x, y):
        self.grid_size = x, y