import os
import torch
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from torch.utils.data import Dataset


BGR_PIXEL_MEANS = np.array([103.939, 116.779, 123.68])


class PascalDatasetYOLO(Dataset):

    def __init__(self,
                 root_dir='data/VOC2012/',
                 classes=None,
                 dataset='train',
                 skip_truncated=True,
                 skip_difficult=True,
                 image_size=(416, 416),
                 grid_size=(13, 13),
                 anchors=None
                 ):

        assert set(classes).issubset({'aeroplane',
                                      'bicycle',
                                      'bird',
                                      'boat',
                                      'bottle',
                                      'bus',
                                      'car',
                                      'cat',
                                      'chair',
                                      'cow',
                                      'diningtable',
                                      'dog',
                                      'horse',
                                      'motorbike',
                                      'person',
                                      'pottedplant',
                                      'sheep',
                                      'sofa',
                                      'train',
                                      'tvmonitor'})

        assert dataset in ['train', 'val', 'trainval']

        if classes is None:
            self.classes = ['aeroplane',
                            'bicycle',
                            'bird',
                            'boat',
                            'bottle',
                            'bus',
                            'car',
                            'cat',
                            'chair',
                            'cow',
                            'diningtable',
                            'dog',
                            'horse',
                            'motorbike',
                            'person',
                            'pottedplant',
                            'sheep',
                            'sofa',
                            'train',
                            'tvmonitor'
                            ]
        else:
            self.classes = classes

        if anchors is None:
            # Set the dimensions for the anchors (xmin, ymin, xmax, ymax).
            self.anchors = torch.tensor(np.array(((0.5, 0.5), (0.3, 0.6), (0.6, 0.3)), dtype=np.float32))
        else:
            self.anchors = torch.tensor(np.array(anchors, dtype=np.float32))

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

        for cls in classes:
            file = os.path.join(self.sets_dir, '{}_{}.txt'.format(cls, dataset))
            with open(file) as f:
                for line in f:
                    image_desc = line.split()
                    if image_desc[1] == '1':
                        self.images.append(image_desc[0])

    def __getitem__(self, index):
        img = self.images[index]
        image = cv2.imread(os.path.join(self.images_dir, img + '.jpg'), 1)
        annotations = self.get_annotations(img)
        target = np.zeros((self.grid_size[0], self.grid_size[1], self.num_anchors * self.num_features), dtype=np.float32)
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
                match = np.argmax(ious)
                anchor = self.anchors[match].numpy()
                target[idx[0], idx[1], match * self.num_features] = 1.
                target[idx[0], idx[1], match * self.num_features + 1] = (xmin + xmax) / 2.
                target[idx[0], idx[1], match * self.num_features + 2] = (ymin + ymax) / 2.
                target[idx[0], idx[1], match * self.num_features + 3] = xmax - xmin
                target[idx[0], idx[1], match * self.num_features + 4] = ymax - ymin
                target[idx[0], idx[1], match * self.num_features + 5:(match + 1) * self.num_features] = self.encode_categorical(name)

        image = cv2.resize(image, self.image_size)
        # Subtract the mean pixel values (across the training set) to zero-mean the image.
        image = image.astype(dtype=np.float32)
        image -= BGR_PIXEL_MEANS

        image = torch.tensor(image)
        image = image.permute(2, 0, 1)
        target = torch.tensor(target)
        target = target.permute(2, 0, 1)
        return image, target

    def __len__(self):
        return len(self.images)

    def get_annotations(self, img):
        file = os.path.join(self.annotations_dir, img + '.xml')
        tree = ET.parse(file)
        root = tree.getroot()

        annotations = []

        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            name = obj.find('name').text
            truncated = int(obj.find('truncated').text)
            difficult = int(obj.find('difficult').text)
            # Get ground truth bounding boxes.
            # NOTE: The creators of the Pascal VOC dataset started counting at 1,
            # and thus the indices have to be corrected.
            xmin = (float(bndbox.find('xmin').text) - 1.) / width
            xmax = (float(bndbox.find('xmax').text) - 1.) / width
            ymin = (float(bndbox.find('ymin').text) - 1.) / height
            ymax = (float(bndbox.find('ymax').text) - 1.) / height
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


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The predictions for the image, Shape: [predictions, 4].
        scores: (tensor) The class scores for the image, Shape:[num_priors].
        overlap: (float) The overlap threshold for suppressing unnecessary boxes.
        top_k: (int) The maximum number of predictions to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        iou = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[iou.le(overlap)]
    return keep


def jaccard(boxes_a, boxes_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    Args:
        boxes_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        boxes_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.
    from: https://github.com/chainer/chainercv
    """
    assert boxes_a.shape[1] == 4
    assert boxes_b.shape[1] == 4
    assert type(boxes_a) == torch.Tensor
    assert type(boxes_a) == torch.Tensor

    # top left
    tl = torch.max(boxes_a[:, None, :2], boxes_b[:, :2])
    # bottom right
    br = torch.min(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_a = torch.prod(boxes_a[:, 2:] - boxes_a[:, :2], 1)
    area_b = torch.prod(boxes_b[:, 2:] - boxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en # * ((tl < br).all())

    area_a = torch.clamp(area_a, min=0)

    ious = area_i / (area_a[:, None] + area_b - area_i)

    return ious


def logit(x):
    if x == 1.:
        return np.inf
    elif x == 0:
        return -np.inf
    elif x < 0 or x > 1:
        return np.nan
    else:
        return np.log(x / (1. - x))


def xywh2xyxy(xywh):
    xyxy = torch.zeros_like(xywh)
    half = xywh[:, 2:] / 2.
    xyxy[:, :2] = xywh[:, :2] - half
    xyxy[:, 2:] = xywh[:, :2] + half

    return xyxy
