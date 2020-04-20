import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageFilter
import torchvision.transforms
from utils import jaccard, read_classes, get_annotations, get_letterbox_padding


USE_LETTERBOX = False
IOU_MATCH_THRESHOLD = 0.05
SMALL_THRESHOLD = 0.005
MULTI_SCALE_FREQ = 10


class PascalDatasetYOLO(Dataset):
    """
    This object can be configured to return images and annotations
    from a PASCAL VOC dataset in a format that is compatible for training
    the YOLOv2 object detector.
    """

    def __init__(self, anchors, class_file, root_dir, dataset=['train'], batch_size=1, skip_truncated=False, do_transforms=False,
                 skip_difficult=True, image_size=(416, 416), stride=32, return_targets=True, multi_scale=False):
        """
        Initialise the dataset object with some network and dataset specific parameters.

        Parameters
        ----------
        anchors : Tensor
                A Tensor object of N anchors given by (x1, y1, x2, y2).
        class_file : str
                The path to a text file containing the names of the different classes that
                should be loaded.
        root_dir : str or list
                The root directory where the PASCAL VOC images and annotations are stored.
        dataset : list, optional
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

        self.classes = read_classes(class_file)

        self.num_classes = len(self.classes)
        self.num_features = 5 + self.num_classes

        if isinstance(root_dir, str):
            root_dir = [root_dir]
        if isinstance(dataset, str):
            dataset = [dataset]

        assert len(root_dir) == len(dataset)

        self.root_dir = root_dir
        self.images_dir = [os.path.join(r, 'JPEGImages') for r in self.root_dir]
        self.annotations_dir = [os.path.join(r, 'Annotations') for r in self.root_dir]
        self.sets_dir = [os.path.join(r, 'ImageSets', 'Main') for r in self.root_dir]
        self.dataset = dataset

        self.images = []
        self.skip_truncated = skip_truncated
        self.skip_difficult = skip_difficult

        self.anchors = anchors.clone().detach().cpu()
        self.num_anchors = len(self.anchors)

        self.do_transforms = do_transforms
        self.return_targets = return_targets

        self.batch_size = batch_size
        self.multi_scale = multi_scale
        self.stride = stride
        self.default_image_size = image_size

        for d in range(len(dataset)):
            for cls in self.classes:
                file = os.path.join(self.sets_dir[d], '{}_{}.txt'.format(cls, dataset[d]))
                with open(file) as f:
                    for line in f:
                        image_desc = line.split()
                        if image_desc[1] == '1':
                            self.images.append((d, image_desc[0]))

        self.images = list(set(self.images))  # Remove duplicates.
        self.images.sort()

        self.n = len(self.images)

        self.image_size = None
        self.grid_size = None
        if self.multi_scale:
            self.step()
        else:
            self.disable_multiscale()

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
        dataset, img = self.images[index]
        # dataset, img = self.images[index % 3 * 9 + 233]
        # dataset, img = (0, '000026') if index % 2 else (0, '000035')
        # dataset, img = random.choice([(1, '2007_001558'), (1, '2007_000272'), (1, '2007_000999')])
        image = Image.open(os.path.join(self.images_dir[dataset], img + '.jpg'))
        image_info = {'id': img, 'width': image.width, 'height': image.height, 'dataset': self.dataset[dataset]}
        if self.do_transforms:
            random_flip = np.random.random()
            random_blur = np.random.random()
            image = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)(image)
            image = torchvision.transforms.RandomGrayscale(p=0.1)(image)
            oversize = 0.2
            image_resize = np.array([image_info['width'], image_info['height']] * np.array((1. + oversize)),
                                    dtype=np.int)
            crop_offset = np.random.random(size=2) * oversize * [image_info['width'], image_info['height']]
            crop_offset = crop_offset.astype(dtype=np.int)
            image = image.resize(image_resize)
            if random_flip >= 0.5:
                image = torchvision.transforms.functional.hflip(image)
            if random_blur >= 0.9:
                image = image.filter(ImageFilter.GaussianBlur(radius=1))
            image = torchvision.transforms.functional.crop(image,
                                                           *crop_offset[::-1],
                                                           image_info['height'], image_info['width'])
        if USE_LETTERBOX:
            ratio, pad = get_letterbox_padding(image.size, self.image_size[index])
            image = image.resize([int(dim * r) for dim, r in zip(image.size, ratio)])
            image = ImageOps.expand(image, pad, fill=(128, 128, 128))
            image_info['padding'] = pad
            image_info['ratio'] = ratio
        else:
            pad = [0., 0., 0., 0.]
            ratio = [dim2 / float(dim1) for dim1, dim2 in zip(image.size, self.image_size[index])]
            image = image.resize(self.image_size[index])
            image_info['padding'] = pad
            image_info['ratio'] = ratio

        image = torchvision.transforms.ToTensor()(image)
        # if self.do_transforms:
        #     image = torchvision.transforms.RandomErasing(p=0.25,
        #                                                  scale=(0.05, 0.1),
        #                                                  ratio=(0.3, 3.3),
        #                                                  value=(0.5, 0.5, 0.5))(image)

        assert (image.size()[1:] == self.image_size[index]).all()

        if self.return_targets:
            annotations = get_annotations(self.annotations_dir[dataset], img)
            random.shuffle(annotations)
            target = np.zeros((self.grid_size[index, 1], self.grid_size[index, 0], self.num_anchors * self.num_features),
                              dtype=np.float32)
            cell_dims = self.image_size[index, 1] / self.grid_size[index, 1], \
                        self.image_size[index, 0] / self.grid_size[index, 0]
            # For each object in image.
            for annotation in annotations:
                name, height, width, xmin, ymin, xmax, ymax, truncated, difficult = annotation
                if (self.skip_truncated and truncated) or (self.skip_difficult and difficult):
                    continue
                if name not in self.classes:
                    continue
                if self.do_transforms:
                    if random_flip >= 0.5:
                        tmp = xmin
                        xmin = width - xmax
                        xmax = width - tmp
                    xmin = (xmin * (1. + oversize) - crop_offset[0]) * ratio[0] + pad[0]
                    xmax = (xmax * (1. + oversize) - crop_offset[0]) * ratio[0] + pad[0]
                    ymin = (ymin * (1. + oversize) - crop_offset[1]) * ratio[1] + pad[1]
                    ymax = (ymax * (1. + oversize) - crop_offset[1]) * ratio[1] + pad[1]
                else:
                    xmin = xmin * ratio[0] + pad[0]
                    xmax = xmax * ratio[0] + pad[0]
                    ymin = ymin * ratio[1] + pad[1]
                    ymax = ymax * ratio[1] + pad[1]
                xmin = np.clip(xmin, a_min=pad[0]+1, a_max=self.image_size[index, 0] - pad[2])
                xmax = np.clip(xmax, a_min=pad[0]+1, a_max=self.image_size[index, 0] - pad[2])
                ymin = np.clip(ymin, a_min=pad[1]+1, a_max=self.image_size[index, 1] - pad[3])
                ymax = np.clip(ymax, a_min=pad[1]+1, a_max=self.image_size[index, 1] - pad[3])
                xmin, xmax, ymin, ymax = np.round(xmin), np.round(xmax), np.round(ymin), np.round(ymax)
                if xmax == xmin or ymax == ymin:
                    continue
                xmin /= cell_dims[0]
                xmax /= cell_dims[0]
                ymin /= cell_dims[1]
                ymax /= cell_dims[1]
                if xmax - xmin < SMALL_THRESHOLD or ymax - ymin < SMALL_THRESHOLD:
                    continue
                idx = int(np.floor((xmax + xmin) / 2.)), int(np.floor((ymax + ymin) / 2.))

                ground_truth = torch.tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
                anchors = torch.zeros((self.num_anchors, 4))
                anchors[:, 2:] = self.anchors.clone()

                target[:, np.arange(self.grid_size[index][0]), 0::self.num_features] = np.arange(self.grid_size[index][0])[None, :, None] + 0.5
                target[:, :, 1::self.num_features] = np.arange(self.grid_size[index][1])[:, None, None] + 0.5
                target[:, :, 2::self.num_features] = 0.
                target[:, :, 3::self.num_features] = 0.

                anchors[:, 0::2] += xmin
                anchors[:, 1::2] += ymin
                ious = jaccard(ground_truth, anchors)
                if ious.max() < IOU_MATCH_THRESHOLD:
                    continue
                assign = np.argmax(ious)

                anchors[assign, 2] -= xmin
                anchors[assign, 3] -= ymin

                target[idx[1], idx[0], assign * self.num_features + 0] = (xmin + xmax) / 2.
                target[idx[1], idx[0], assign * self.num_features + 1] = (ymin + ymax) / 2.
                target[idx[1], idx[0], assign * self.num_features + 2] = np.log((xmax - xmin) / anchors[assign, 2])
                target[idx[1], idx[0], assign * self.num_features + 3] = np.log((ymax - ymin) / anchors[assign, 3])
                target[idx[1], idx[0], assign * self.num_features + 4] = 1.
                target[idx[1], idx[0], assign * self.num_features + 5:(assign + 1) * self.num_features] = \
                    self.encode_categorical(name)

            target = torch.tensor(target)
            target = target.permute(2, 0, 1)

        if self.return_targets:
            return image, image_info, target
        else:
            return image, image_info

    def __len__(self):
        return len(self.images)

    def disable_multiscale(self):
        self.multi_scale = False
        self.image_size = np.repeat(self.default_image_size, self.n).reshape(-1, 2)
        self.grid_size = np.repeat([s // self.stride for s in self.default_image_size], self.n).reshape(-1, 2)

    def shuffle_images(self):
        random.shuffle(self.images)

    def step(self, multi_scale=None):
        if multi_scale is not None:
            self.multi_scale = multi_scale
        self.shuffle_images()
        if self.multi_scale:
            size = (2 * np.random.randint(4, 10, self.n // (self.batch_size * MULTI_SCALE_FREQ) + 1) + 1)
            self.image_size = np.repeat(size * self.stride, 2 * self.batch_size * MULTI_SCALE_FREQ).reshape(-1, 2)
            self.grid_size = np.repeat(size, 2 * self.batch_size * MULTI_SCALE_FREQ).reshape(-1, 2)
        else:
            self.image_size = np.repeat(self.default_image_size, self.n).reshape(-1, 2)
            self.grid_size = np.repeat([s // self.stride for s in self.default_image_size], self.n).reshape(-1, 2)

    def encode_categorical(self, name):
        y = self.classes.index(name)
        yy = np.zeros(self.num_classes)
        yy[y] = 1
        return yy
