import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from utils import jaccard, read_classes, get_annotations
from PIL import Image


class PascalDatasetYOLO(Dataset):

    def __init__(self, anchors, classes, root_dir='data/VOC2012/', dataset='train', skip_truncated=True,
                 do_transform=None, skip_difficult=True, image_size=(416, 416), grid_size=(13, 13)):

        self.classes = read_classes(classes)

        assert set(self.classes).issubset({'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
                                           'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
                                           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'})

        assert dataset in ['train', 'val', 'trainval', 'test']

        self.anchors = anchors.clone().detach().cpu()
        self.num_classes = len(self.classes)
        self.num_anchors = len(self.anchors)
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

        self.transforms = do_transform

        for cls in self.classes:
            file = os.path.join(self.sets_dir, '{}_{}.txt'.format(cls, dataset))
            with open(file) as f:
                for line in f:
                    image_desc = line.split()
                    if image_desc[1] == '1':
                        self.images.append(image_desc[0])

        self.images = list(set(self.images))  # remove duplicates
        self.images.sort()

    def __getitem__(self, index):

        img = self.images[index]
        # img = '2008_005808'
        image = Image.open(os.path.join(self.images_dir, img + '.jpg'))
        image_info = {'id': img, 'width': image.width, 'height': image.height, 'dataset': self.dataset}
        oversize = .1
        random_flip = np.random.random()
        crop_offset = (np.random.random(size=2) * oversize * self.image_size).astype(dtype=np.int)
        if self.transforms:
            image = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.25, hue=0.05)(image)
            image_resize = np.array(self.image_size * np.array((1. + oversize)), dtype=np.int)
            image = image.resize(image_resize)
            if random_flip >= 0.5:
                image = transforms.functional.hflip(image)
            image = transforms.functional.crop(image,
                                               *crop_offset[::-1],
                                               *self.image_size[::-1])
        else:
            image = image.resize(self.image_size)

        annotations = get_annotations(self.annotations_dir, img)
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
                if self.transforms:
                    if random_flip >= 0.5:
                        tmp = xmin
                        xmin = 1. - xmax
                        xmax = 1. - tmp
                    xmin = (xmin * image_resize[0] - crop_offset[0]) / self.image_size[0]
                    xmax = (xmax * image_resize[0] - crop_offset[0]) / self.image_size[0]
                    ymin = (ymin * image_resize[1] - crop_offset[1]) / self.image_size[1]
                    ymax = (ymax * image_resize[1] - crop_offset[1]) / self.image_size[1]
                    xmin = np.clip(xmin, a_min=0, a_max=1.)
                    xmax = np.clip(xmax, a_min=0, a_max=1.)
                    ymin = np.clip(ymin, a_min=0, a_max=1.)
                    ymax = np.clip(ymax, a_min=0, a_max=1.)
                target[idx[0], idx[1], assign * self.num_features + 0] = (xmin + xmax) / 2.
                target[idx[0], idx[1], assign * self.num_features + 1] = (ymin + ymax) / 2.
                target[idx[0], idx[1], assign * self.num_features + 2] = xmax - xmin
                target[idx[0], idx[1], assign * self.num_features + 3] = ymax - ymin
                target[idx[0], idx[1], assign * self.num_features + 4] = 1.
                target[idx[0], idx[1], assign * self.num_features + 5:(assign + 1) * self.num_features] = self.encode_categorical(name)

        image = transforms.ToTensor()(image)

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