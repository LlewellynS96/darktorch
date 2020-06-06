import random
import torch
import numpy as np
import cv2
import os
import xml.etree.ElementTree as Et
from functools import reduce
from matplotlib import cm


NUM_WORKERS = 0


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def non_maximum_suppression(boxes, scores, overlap=0.5, top_k=101):
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

    return keep[:count].clone()


def jaccard(boxes_a, boxes_b):
    """
    Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.
    Parameters
    ----------
        boxes_a : Tensor
            An array whose shape is :math:`(N, 4)`. :math:`N` is the number
            of bounding boxes. The dtype should be :obj:`float`.
        boxes_b : Tensor
            An array similar to :obj:`bbox_a`, whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`float`.
    Returns
    -------
        Tensor
            An array whose shape is :math:`(N, K)`. An element at index :math:`(n, k)`
            contains IoUs between :math:`n` th bounding box in :obj:`bbox_a` and
            :math:`k` th bounding box in :obj:`bbox_b`.
    Notes
    -----
        from: https://github.com/chainer/chainercv
    """
    assert boxes_a.shape[1] == 4
    assert boxes_b.shape[1] == 4
    assert isinstance(boxes_a, torch.Tensor)
    assert isinstance(boxes_b, torch.Tensor)

    tl = torch.max(boxes_a[:, None, :2], boxes_b[:, :2])
    br = torch.min(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_a = torch.prod(boxes_a[:, 2:] - boxes_a[:, :2], 1)
    area_b = torch.prod(boxes_b[:, 2:] - boxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en

    # area_a = torch.clamp(area_a, min=0)

    ious = area_i / (area_a[:, None] + area_b - area_i)
    ious[torch.isnan(ious)] = 0.

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
    """
    Converts bounding boxes that are in the form
    (x_c, y_c, w, h) to (x1, y1, x2, y2).
    Parameters
    ----------
    xywh : Tensor
        A tensor whose shape is :math:`(N, 4)` where the elements in the
        0th and 1st column represent the x and y coordinates of the center
        of the bounding box and the 2nd and 3rd column represent the
        width and height of the bounding box.

    Returns
    -------
    xyxy : Tensor
        A tensor whose shape is :math:`(N, 4)` where the elements in the
        0th and 1st column represent the x and y coordinates of the top left
        corner of the bounding box and the 2nd and 3rd column represent the
        x and y coordinates of the bottom right corner of the bounding box.
    """
    xyxy = torch.zeros_like(xywh)
    half = xywh[:, 2:] / 2.
    xyxy[:, :2] = xywh[:, :2] - half
    xyxy[:, 2:] = xywh[:, :2] + half

    return xyxy


def xyxy2xywh(xyxy):
    """
    Converts bounding boxes that are in the form
    (x1, y1, x2, y2) to (x_c, y_c, w, h).
    Parameters
    ----------
    xyxy : Tensor
        A tensor whose shape is :math:`(N, 4)` where the elements in the
        0th and 1st column represent the x and y coordinates of the top left
        corner of the bounding box and the 2nd and 3rd column represent the
        x and y coordinates of the bottom right corner of the bounding box.
    Returns
    -------
    xywh : Tensor
        A tensor whose shape is :math:`(N, 4)` where the elements in the
        0th and 1st column represent the x and y coordinates of the center
        of the bounding box and the 2nd and 3rd column represent the
        width and height of the bounding box.
    """
    xywh = torch.zeros_like(xyxy)
    xywh[:, :2] = (xyxy[:, ::2] + xyxy[:, 1::2]) / 2.
    xywh[:, 2:] = xyxy[:, ::2] - xyxy[:, 1::2]

    return xywh


def read_classes(file):
    """
    Utility function that parses a text file containing all the classes
    that are present in a specific dataset.
    Parameters
    ----------
    file : str
        A string pointing to the text file to be read.

    Returns
    -------
    classes : list
        A list containing the classes read from the text file.
    """
    file = open(file, 'r')
    lines = file.read().split('\n')
    lines = [l for l in lines if len(l) > 0]
    classes = [l for l in lines if l[0] != '#']

    return classes


def to_numpy_image(image, size, mu=0., sigma=1., normalised=True):
    """
    Converts a Tensor in the range [0., 1.] to a resized
    Numpy array in the range [0, 255].
    Parameters
    ----------
    image : Tensor
        A Tensor representing the image.
    size : tuple of int
        The size (w, h) to which the image should be resized.
    normalize : bool
        A flag which indicates whether the image was originally normalized,
        which means that it should be "de-normalized" when converting to an
        array.
    Returns
    -------
    image : ndarray
        A Numpy array representation of the image.
    """
    if image.ndim == 2:
        image = image[None]
    if isinstance(image, torch.Tensor):
        image = np.array(image.permute(1, 2, 0).cpu().numpy())
    image *= np.array(sigma)
    image += np.array(mu)
    if not normalised:
        image -= np.min(image)
        image /= np.max(image)
    if image.shape[2] == 1:
        image = cm.get_cmap('viridis')(image)[:, :, 0, :3]
    image *= 255.
    image = image.astype(dtype=np.uint8)
    image = cv2.resize(image, dsize=size, interpolation=cv2.INTER_CUBIC)

    return image


def add_bbox_to_image(image, bbox, confidence, cls, thickness=3, color=None):
    """
    Adds a visual bounding box with labels to an image in-place.

    Parameters
    ----------
    image : ndarray
        A Numpy array containing the image.
    bbox : array_like
        An array (x1, y1, x2, y2) containing the coordinates of the upper-
        left and bottom-right corners of the bounding box to be added to
        the image. The coordinates should be normalized to the width and
        the height of the image.
    confidence : float
        A value representing the confidence of an object within the bounding
        box. This value will be displayed as part of the label.
    cls : str
        The class to which the object in the bounding box belongs. This
        value will be displayed as part of the label.
    """
    if confidence is not None:
        text = '{} {:.2f}'.format(cls, confidence)
    else:
        text = '{}'.format(cls)
    xmin, ymin, xmax, ymax = bbox
    # Draw a bounding box.
    if color is None:
        color = np.random.uniform(0., 255., size=3)
    cv2.rectangle(image, (xmin, ymax), (xmax, ymin), color, thickness)

    # Display the label at the top of the bounding box
    label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ymax = max(ymax, label_size[1])
    cv2.rectangle(image,
                  (xmin, ymax),
                  (xmin + round(1.5 * label_size[0]),
                   ymax + base_line + round(1.5 * label_size[1])),
                  color,
                  cv2.FILLED)
    cv2.putText(image, text, (xmin, ymax + round(1.5 * label_size[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, [255] * 3, 1)


def export_prediction(cls, image_id, top, left, bottom, right, confidence,
                      prefix='comp4', set_name='test', directory='detections'):
    """
    Exports a single predicted bounding box to a text file by appending it to a file
    in the format specified by the Pascal VOC competition.
    Parameters
    ----------
    cls : str
        The predicted class name of the specified bounding box.
    image_id : str
        The Pascal VOC image ID, i.e. the image's file name.
    top : float
        The y-coordinate of the top-left corner of the predicted bounding box. The value
        should be normalised to the height of the image.
    left : float
        The x-coordinate of the top-left corner of the predicted bounding box. The value
        should be normalised to the width of the image.
    bottom : float
        The y-coordinate of the bottom-right corner of the predicted bounding box. The value
        should be normalised to the height of the image.
    right : float
        The x-coordinate of the bottom-right corner of the predicted bounding box. The value
        should be normalised to the width of the image.
    confidence : float
        A confidence value attached to the prediction, which should be generated by the
        detector.  The value does not have to be normalised, but a greater value corresponds
        to greater confidence.  This value is used when calculating the precision-recall graph
        for the detector.
    prefix : str
        A string value that is prepended to the file where the predictions are stored.  For
        PASCAL VOC competitions this value is 'comp' + the number of the competition being
        entered into, e.g. comp4.
    set_name : str
        The subset for which the predictions were made, e.g. 'val', 'test' etc..
    directory : str
        The directory to which all the prediction files should be saved.

    Returns
    -------
    None
    """
    filename = prefix + '_det_' + set_name + '_' + cls + '.txt'
    filename = os.path.join(directory, filename)

    with open(filename, 'a') as f:
        prediction = [image_id, confidence, np.round(left), np.round(top), np.round(right), np.round(bottom), '\n']
        prediction = map(to_repr, prediction)
        prediction = ' '.join(prediction)
        f.write(prediction)


def to_repr(x):
    """
    A small utility function that converts floats and strings to
    a fixed string format.
    Parameters
    ----------
    x
        An input for which a string representation should be provided.
    Returns
    -------
    str
        A string representation of the input.

    """
    if isinstance(x, (float, np.float, np.float32)):
        return '{:.6f}'.format(x)
    else:
        return str(x)


def get_annotations(annotations_dir, img):

    file = os.path.join(annotations_dir, img + '.xml')
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

        xmin = (float(bbox.find('xmin').text))
        xmax = (float(bbox.find('xmax').text))
        ymin = (float(bbox.find('ymin').text))
        ymax = (float(bbox.find('ymax').text))
        annotations.append((name, height, width, xmin, ymin, xmax, ymax, truncated, difficult))

    return annotations


def find_best_anchors(classes, root_dir, dataset, k=5, max_iter=20, skip_truncated=True, init=13, weighted=True, device='cuda'):

    annotations_dir = [os.path.join(r, 'Annotations') for r in root_dir]
    sets_dir = [os.path.join(r, 'ImageSets', 'Main') for r in root_dir]

    images = []

    for d in range(len(dataset)):
        for cls in classes:
            file = os.path.join(sets_dir[d], '{}_{}.txt'.format(cls, dataset[d]))
            with open(file) as f:
                for line in f:
                    image_desc = line.split()
                    if image_desc[1] == '1':
                        images.append((d, image_desc[0]))

    images = list(set(images))
    bboxes = []

    for image in images:
        annotations = get_annotations(annotations_dir[image[0]], image[1])
        for annotation in annotations:
            name, height, width, xmin, ymin, xmax, ymax, truncated, difficult = annotation
            if skip_truncated and truncated:
                continue
            width = (xmax - xmin) / width
            height = (ymax - ymin) / height
            for i in [2. * d + 1 for d in range(4, 10)]:
                bboxes.append([0., 0., i * width, i * height])

    bboxes = torch.tensor(bboxes, device=device)
    anchors = torch.tensor(([0., 0., init, init] * np.random.random((k, 4))).astype(dtype=np.float32), device=device)

    for _ in range(max_iter):
        ious = jaccard(bboxes, anchors)
        iou_max, idx = torch.max(ious, dim=1)
        for i in range(k):
            if weighted:
                weights = (torch.tensor([1.], device=device) - iou_max[idx == i, None]) ** 2
                anchors[i] = torch.sum(bboxes[idx == i] * weights, dim=0) / torch.sum(weights)  # Weighted k-means

            else:
                anchors[i] = torch.mean(bboxes[idx == i], dim=0)  # Normal k-means

        sort = torch.argsort(anchors[:, 2], dim=0)
        anchors = anchors[sort]

    return anchors[:, 2:]


def get_letterbox_padding(size, desired_size):
    scale = min([float(dim) / max(size) for dim in desired_size])
    new_size = np.array([dim * scale for dim in size], dtype=np.int)

    delta = desired_size - new_size
    padding = (delta[0] // 2, delta[1] // 2, delta[0] - (delta[0] // 2), delta[1] - (delta[1] // 2))

    return [scale] * 2, padding


def exponential_decay_scheduler(optimizer, initial_lr=None, warm_up=1, decay=0.05):
    lr = optimizer.defaults['lr']
    if initial_lr is None:
        initial_lr = lr / 10.
    if warm_up > 0:
        gradient = (lr - initial_lr) / float(warm_up)

    def foo(e):
        if e < warm_up:
            return gradient * e + initial_lr
        else:
            return (1. - decay) ** (e - warm_up)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=foo)

    return scheduler


def const_step_decay_scheduler(optimizer, initial_lr=None, warm_up=1, steps=10, decay=0.1):
    lr = optimizer.defaults['lr']
    if initial_lr is None:
        initial_lr = lr / 10.
    if warm_up > 0:
        gradient = (lr - initial_lr) / float(warm_up)

    def foo(e):
        if e < warm_up:
            return gradient * e + initial_lr
        else:
            return decay ** ((e - warm_up) // steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=foo)

    return scheduler


def step_decay_scheduler(optimizer, steps=None, scales=None):
    if steps is None or scales is None:
        steps = [-1, 100]
        scales = [0.1, 10.]

    def foo(e):
        if e < min(steps):
            return 1.
        for i, s in enumerate(reversed(steps)):
            if e >= s:
                return reduce(lambda x, y: x * y, scales[:len(steps) - i])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=foo)

    return scheduler


def linear_scheduler(optimizer, initial_lr=None, final_lr=None, warm_up=1, iterations=100):
    lr = optimizer.defaults['lr']
    if initial_lr is None:
        initial_lr = lr / 10.
    if final_lr is None:
        initial_lr = lr / 100.
    if warm_up > 0:
        gradient1 = (lr - initial_lr) / float(warm_up)
        gradient2 = (final_lr - lr) / float(iterations - warm_up)

    def foo(e):
        if e < warm_up:
            return gradient1 * e + initial_lr
        else:
            return gradient2 * e + lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=foo)

    return scheduler


def warmup_scheduler(optimizer, initial_lr=None, warm_up=1):
    lr = optimizer.defaults['lr']
    if initial_lr is None:
        initial_lr = lr / 10.
    if warm_up > 0:
        gradient = (lr - initial_lr) / float(warm_up)

    def foo(e):
        if e < warm_up:
            return gradient * e + initial_lr
        else:
            return lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=foo)

    return scheduler


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    classes = read_classes('../../../Data/VOCdevkit/voc.names')
    # a = find_best_anchors(classes,
    #                       k=5,
    #                       max_iter=1000,
    #                       root_dir=['../../../Data/VOCdevkit/VOC2007/', '../../../Data/VOCdevkit/VOC2012/'],
    #                       dataset=['trainval'] * 2,
    #                       skip_truncated=False,
    #                       weighted=True,
    #                       device=device)
    classes = read_classes('../../../Data/SS/ss.names')
    a = find_best_anchors(classes,
                          k=5,
                          max_iter=1000,
                          root_dir=['../../../Data/SS/'],
                          dataset=['train'],
                          skip_truncated=False,
                          init=3.,
                          weighted=True,
                          device=device)

    for x, y in a:
        print('{:.2f},{:.2f}, '.format(x, y), end='')


def set_random_seed(x):
    np.random.seed(x)
    torch.random.manual_seed(x)
    random.seed(x)


if __name__ == '__main__':
    main()
