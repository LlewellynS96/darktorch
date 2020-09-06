import os
import random
import pickle
import torch
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import gizeh
from darknet import YOLO
from utils import export_prediction, jaccard, read_classes, get_annotations, to_numpy_image, add_bbox_to_image, set_random_seed
from dataset import PascalDatasetYOLO

USE_LETTERBOX = False
IOU_MATCH_THRESHOLD = 0.05
SMALL_THRESHOLD = 0.005
MULTI_SCALE_FREQ = 10


class SSDatasetYOLO(Dataset):

    def __init__(self, anchors, class_file, root_dir, mu, sigma, mode, dataset, batch_size=1, skip_truncated=False, do_transforms=False,
                 image_size=(512, 512), skip_difficult=True, strides=32, return_targets=True):

        self.classes = read_classes(class_file)

        self.num_classes = len(self.classes)
        self.num_features = 5 + self.num_classes

        if isinstance(root_dir, str):
            root_dir = [root_dir]
        if isinstance(dataset, str):
            dataset = [dataset]

        assert len(root_dir) == len(dataset)

        self.root_dir = root_dir
        self.raw_dir = [os.path.join(r, 'Raw') for r in self.root_dir]
        self.annotations_dir = [os.path.join(r, 'Annotations') for r in self.root_dir]
        self.sets_dir = [os.path.join(r, 'ImageSets', 'Main') for r in self.root_dir]
        self.dataset = dataset

        self.data = []
        self.skip_truncated = skip_truncated
        self.skip_difficult = skip_difficult

        self.anchors = [a.clone().detach().cpu() for a in anchors]
        self.num_anchors = [len(a) for a in self.anchors]

        self.do_transforms = do_transforms
        self.return_targets = return_targets

        self.batch_size = batch_size
        if isinstance(strides, int):
            strides = [strides]
        self.strides = strides

        assert len(anchors) == len(strides)
        self.num_detectors = len(anchors)

        self.default_image_size = image_size

        self.mode = mode
        self.mu = mu
        self.sigma = sigma

        for d in range(len(dataset)):
            for cls in self.classes:
                file = os.path.join(self.sets_dir[d], '{}_{}.txt'.format(cls, dataset[d]))
                with open(file) as f:
                    for line in f:
                        image_desc = line.split()
                        if image_desc[1] == '1':
                            self.data.append((d, image_desc[0]))

        self.data = list(set(self.data))  # Remove duplicates.
        self.data.sort()

        self.n = len(self.data)

        self.image_size = np.repeat(self.default_image_size, self.n).reshape(-1, 2)
        self.grid_sizes = []
        for ss in self.strides:
            self.grid_sizes.append(np.repeat([s // ss for s in self.default_image_size], self.n).reshape(-1, 2))

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
        dataset, img = self.data[index]
        data = np.load(os.path.join(self.raw_dir[dataset], img + '.npz'))
        signal = data['signal']
        samp_rate = data['samp_rate']
        N_fft = data['N_fft']
        N_overlap = data['N_overlap']
        signal = signal[0] + 1.j * signal[1]
        stft, _, _ = self.stft(signal,
                               N_fft=N_fft,
                               N_overlap=N_overlap,
                               samp_rate=samp_rate)
        if self.mode == 'spectrogram':
            data = np.abs(stft) ** 2
        elif self.mode == 'spectrogram_db':
            data = 10. * np.log10(np.abs(stft) ** 2)
        elif self.mode == 'spectrogram_ap':
            data = [np.abs(stft) ** 2, np.angle(stft)]
        elif self.mode == 'spectrogram_ap_db':
            data = [10. * np.log10(np.abs(stft) ** 2), np.angle(stft)]
        elif self.mode == 'stft':
            data = np.abs(stft)
        elif self.mode == 'stft_iq':
            data = [stft.real, stft.imag]
        elif self.mode == 'stft_ap':
            data = [np.abs(stft), np.angle(stft)]
        else:
            raise ValueError('Unknown mode. Use one of spectrogram, spectrogram_db, '
                             'spectrogram_ap, spectrogram_ap_db, stft_iq or stft_ap.')

        data = torch.tensor(data, dtype=torch.float32)
        if data.ndim == 2:
            data = data[None]
        data = (data - torch.tensor(self.mu)[:, None, None]) / torch.tensor(self.sigma)[:, None, None]

        data_info = {'id': img, 'width': data.shape[2], 'height': data.shape[1], 'dataset': self.dataset[dataset]}

        if self.do_transforms:
            pass
        data_info['padding'] = [0., 0., 0., 0.]
        data_info['scale'] = [1., 1.]

        assert (data.size()[1:] == self.image_size[index]).all()

        if self.return_targets:
            annotations = get_annotations(self.annotations_dir[dataset], img)
            random.shuffle(annotations)
            target = [np.zeros((self.grid_sizes[i][index, 1],
                                self.grid_sizes[i][index, 0],
                                self.num_anchors[i] * self.num_features),
                               dtype=np.float32) for i in range(self.num_detectors)]
            cell_dims = np.array([[self.strides[i], self.strides[i]] for i in range(self.num_detectors)])

            anchors = [torch.zeros((n, 4)) for n in self.num_anchors]
            for i, (a, t) in enumerate(zip(anchors, target)):
                a[:, 2:] = self.anchors[i].clone()
                t[:, np.arange(self.grid_sizes[i][index][0]), 0::self.num_features] = \
                    np.arange(self.grid_sizes[i][index][0])[None, :, None] + 0.5
                t[:, :, 1::self.num_features] = np.arange(self.grid_sizes[i][index][1])[:, None, None] + 0.5
                t[:, :, 2::self.num_features] = a[:, 2]
                t[:, :, 3::self.num_features] = a[:, 3]

            # For each object in image.
            for annotation in annotations:
                name, height, width, xmin, ymin, xmax, ymax, truncated, difficult = annotation
                if (self.skip_truncated and truncated) or (self.skip_difficult and difficult):
                    continue
                if name not in self.classes:
                    continue
                if self.do_transforms:
                    pass
                xmin = np.clip(xmin, a_min=1, a_max=self.image_size[index, 0])
                xmax = np.clip(xmax, a_min=1, a_max=self.image_size[index, 0])
                ymin = np.clip(ymin, a_min=1, a_max=self.image_size[index, 1])
                ymax = np.clip(ymax, a_min=1, a_max=self.image_size[index, 1])
                xmin, xmax, ymin, ymax = np.round(xmin), np.round(xmax), np.round(ymin), np.round(ymax)
                if xmax == xmin or ymax == ymin:
                    continue
                xmin /= cell_dims[:, 0]
                xmax /= cell_dims[:, 0]
                ymin /= cell_dims[:, 1]
                ymax /= cell_dims[:, 1]
                if all(xmax - xmin < (SMALL_THRESHOLD * cell_dims[:, 0])):
                    continue
                if all(ymax - ymin < (SMALL_THRESHOLD * cell_dims[:, 1])):
                    continue
                idx = np.floor((xmax + xmin) / 2.), np.floor((ymax + ymin) / 2.)
                idx = np.array(idx, dtype=np.int).T

                ground_truth = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32).t()
                anchors = [torch.zeros((self.num_anchors[i], 4)) for i in range(self.num_detectors)]
                for i in range(self.num_detectors):
                    anchors[i][:, 2:] = self.anchors[i].clone()
                    anchors[i][:, 0::2] += xmin[i]
                    anchors[i][:, 1::2] += ymin[i]
                anchors = torch.cat(anchors)
                ious = jaccard(ground_truth, anchors)
                if ious.max() < IOU_MATCH_THRESHOLD:
                    continue
                max_iou = 0.
                cumsum_detectors = np.cumsum([0] + self.num_anchors)
                for i in range(self.num_detectors):
                    if ious[i, cumsum_detectors[i]:cumsum_detectors[i + 1]].max() > max_iou:
                        l = i
                        d = ious[i, cumsum_detectors[i]:cumsum_detectors[i + 1]].argmax()
                        max_iou = ious[i, cumsum_detectors[i]:cumsum_detectors[i + 1]].max()

                target[l][idx[l][1], idx[l][0], d * self.num_features + 0] = (xmin[l] + xmax[l]) / 2.
                target[l][idx[l][1], idx[l][0], d * self.num_features + 1] = (ymin[l] + ymax[l]) / 2.
                target[l][idx[l][1], idx[l][0], d * self.num_features + 2] = xmax[l] - xmin[l]
                target[l][idx[l][1], idx[l][0], d * self.num_features + 3] = ymax[l] - ymin[l]
                target[l][idx[l][1], idx[l][0], d * self.num_features + 4] = 1.
                target[l][idx[l][1], idx[l][0], d * self.num_features + 5:(d + 1) * self.num_features] = \
                    self.encode_categorical(name)

            target = [torch.tensor(target[i]).permute(2, 0, 1) for i in range(self.num_detectors)]

            return data, data_info, target
        else:
            return data, data_info

    def __len__(self):
        return self.n

    def shuffle_data(self):
        random.shuffle(self.data)

    def step(self, multi_scale=None):
        self.shuffle_data()

    def encode_categorical(self, name):
        y = self.classes.index(name)
        yy = np.zeros(self.num_classes)
        yy[y] = 1
        return yy

    def disable_multiscale(self):
        pass

    @staticmethod
    def stft(x, N_fft=512, N_overlap=64, samp_rate=10e6):
        f, t, specgram = scipy.signal.stft(x,
                                           fs=samp_rate,
                                           nperseg=N_fft,
                                           noverlap=N_overlap,
                                           return_onesided=False,
                                           boundary=None,
                                           padded=False)
        idx = np.argsort(f)
        specgram = specgram[idx]
        f = f[idx]

        return specgram, f, t


def show_ground_truth(model,
                      dataset,
                      device='cpu',
                      classes='../../../Data/SS/ss.names',
                      overlap_threshold=0.5,
                      show=True,
                      export=True,
                      num_workers=0):

    lst_classes = read_classes(classes)
    colors = create_pascal_label_colormap(len(lst_classes))

    dataset.n = 5

    dataloader = DataLoader(dataset=dataset,
                            batch_size=dataset.batch_size,
                            num_workers=num_workers)

    image_idx_ = []
    bboxes_ = []
    confidence_ = []
    classes_ = []

    with tqdm(total=len(dataloader),
              desc='Exporting',
              leave=True) as pbar:
        for data in dataloader:
            images, image_info, targets = data
            model.set_image_size(images.shape[-2:])
            width, height = model.image_size
            targets = [t.to(device) for t in targets]
            bboxes, classes, confidences, image_idx = model.process_bboxes(targets,
                                                                           image_info,
                                                                           1e-5,
                                                                           overlap_threshold,
                                                                           nms=False)

            if show:
                for i, data in enumerate(zip(image_info['id'], images)):
                    idx, image = data
                    if image.shape[0] == 3:
                        image = to_numpy_image(image, size=(width, height))
                    else:
                        mu = dataset.mu[0]
                        sigma = dataset.sigma[0]
                        image = to_numpy_image(image[0], size=(width, height), mu=mu, sigma=sigma, normalised=False)
                    image = gizeh.ImagePattern(np.array(image))
                    image = gizeh.rectangle(2. * model.default_image_size[0],
                                            2. * model.default_image_size[1],
                                            xy=(0, 0),
                                            fill=image)
                    pdf = gizeh.PDFSurface('detections/{}.pdf'.format(idx),
                                           2. * model.default_image_size[0],
                                           2. * model.default_image_size[1])
                    image.draw(pdf)
                    mask = np.array(image_idx) == idx
                    for bb, cl, co in zip(bboxes[mask], classes[mask], confidences[mask]):
                        rect = [[int(bb[0]), int(bb[1])],
                                [int(bb[2]), int(bb[1])],
                                [int(bb[2]), int(bb[3])],
                                [int(bb[0]), int(bb[3])]]
                        rect = gizeh.polyline(rect, close_path=True, stroke_width=4, stroke=colors[cl])
                        rect.draw(pdf)
                    for bb, cl, co in zip(bboxes[mask], classes[mask], confidences[mask]):
                        w, h = len(lst_classes[cl]) * 7.8, 11
                        rect = gizeh.rectangle(w,
                                               h,
                                               xy=(int(bb[0] + w / 2 - 2),
                                                   max((int(bb[1] - h / 2 + 5)), 5)),  # - 2)),
                                               # fill=colors[cl])
                                               fill=(1, 1, 1, 0.5))

                        rect.draw(pdf)
                        txt = gizeh.text(lst_classes[cl],
                                         # 'Helvetica',
                                         'monospace',
                                         fontsize=12,
                                         xy=(int(bb[0]),
                                             max((int(bb[1]), 5))),  # - 12),
                                         fill=(0., 0., 0.),
                                         v_align='center',  # 'bottom',
                                         h_align='left')
                        txt.draw(pdf)
                    pdf.flush()
                    pdf.finish()

            if export:
                for idx in range(len(images)):
                    mask = [True if idx_ == image_info['id'][idx] else False for idx_ in image_idx]
                    for bbox, cls, confidence in zip(bboxes[mask], classes[mask], confidences[mask]):
                        name = dataset.classes[cls]
                        ids = image_info['id'][idx]
                        set_name = image_info['dataset'][idx]
                        confidence = confidence.item()
                        bbox[::2] -= image_info['padding'][0][idx]
                        bbox[1::2] -= image_info['padding'][1][idx]
                        bbox[::2] /= image_info['scale'][0][idx]
                        bbox[1::2] /= image_info['scale'][1][idx]
                        x1, y1, x2, y2 = bbox.detach().cpu().numpy()
                        export_prediction(cls=name,
                                          prefix=model.name,
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
        image_idx = [item for sublist in image_idx_ for item in sublist]

        return bboxes, classes, confidence, image_idx
    else:
        return torch.tensor([], device=device), \
               torch.tensor([], device=device), \
               torch.tensor([], device=device), \
               []


def find_best_anchors(classes, root_dir, dataset, k=5, max_iter=20, skip_truncated=True, init=(13, 13), weighted=True, multi_scale=False, device='cuda'):

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
            if multi_scale:
                for i in [2. * d + 1 for d in range(4, 10)]:
                    bboxes.append([0., 0., i * width, i * height])
            else:
                bboxes.append([0., 0., 13. * width, 13. * height])

    bboxes = torch.tensor(bboxes, dtype=torch.float64, device=device)
    # anchors = [[0, 0, 3, 3],
    #            [0, 0, 4, 3],
    #            [0, 0, 5, 3],
    #            [0, 0, 4, 4],
    #            [0, 0, 5, 4],
    #            [0, 0, 5, 5],
    #            [0, 0, 6, 5],
    #            [0, 0, 10, 5],
    #            [0, 0, 13, 5]]
    anchors = torch.tensor(([0., 0., init[0], init[1]] * np.random.random((k, 4))).astype(dtype=np.float64), device=device)
    # anchors = torch.tensor(anchors, dtype=torch.float64, device=device)

    for _ in range(max_iter):
        ious = jaccard(bboxes, anchors)
        iou_max, idx = torch.max(ious, dim=1)
        for i in range(k):
            if weighted:
                weights = (torch.tensor([1.], device=device) - iou_max[idx == i, None]) ** 10
                anchors[i] = torch.sum(bboxes[idx == i] * weights, dim=0) / torch.sum(weights)  # Weighted k-means

            else:
                anchors[i] = torch.mean(bboxes[idx == i], dim=0)  # Normal k-means

        sort = torch.argsort(anchors[:, 2], dim=0)
        anchors = anchors[sort]

    return anchors[:, 2:]


def draw_vector_bboxes(device='cuda'):
    # colors = plt.cm.get_cmap('tab20').colors
    colors = create_pascal_label_colormap(3)

    model = YOLO(name='YOLOv2',
                 model='models/yolov2-ss.cfg',
                 device=device)

    model = pickle.load(open('YOLOv2_120.pkl', 'rb'))

    data = PascalDatasetYOLO(root_dir='../../../Data/SS/',
                             class_file='../../../Data/SS/ss.names',
                             dataset='test',
                             batch_size=model.batch_size // model.subdivisions,
                             image_size=model.default_image_size,
                             anchors=model.anchors,
                             strides=model.strides,
                             do_transforms=False,
                             multi_scale=False,
                             return_targets=False
                             )

    bboxes, classes, confidence, image_idx = model.predict(dataset=data,
                                                           confidence_threshold=.5,
                                                           overlap_threshold=.45,
                                                           show=False,
                                                           export=False
                                                           )

    for idx in np.unique(image_idx):
        image = Image.open(os.path.join(data.root_dir[0], 'JPEGImages', idx + '.jpg'))
        image = image.resize(model.default_image_size)
        image = gizeh.ImagePattern(np.array(image))
        image = gizeh.rectangle(2*model.default_image_size[0],
                                2*model.default_image_size[1],
                                xy=(0, 0),
                                fill=image)
        pdf = gizeh.PDFSurface('detections/{}.pdf'.format(idx),
                               model.default_image_size[0],
                               model.default_image_size[1])
        image.draw(pdf)
        mask = np.array(image_idx) == idx
        _bboxes = bboxes[mask]
        _classes = classes[mask]
        _confidence = confidence[mask]
        argsort_x = torch.argsort(_bboxes[:, 0])
        argsort_y = torch.argsort(_bboxes[argsort_x][:, 1])
        _bboxes = _bboxes[argsort_x][argsort_y]
        _classes = _classes[argsort_x][argsort_y]
        _confidence = _confidence[argsort_x][argsort_y]
        for bb, cl, co in zip(_bboxes, _classes, _confidence):
            rect = [[int(bb[0]), int(bb[1])],
                    [int(bb[2]), int(bb[1])],
                    [int(bb[2]), int(bb[3])],
                    [int(bb[0]), int(bb[3])]]
            rect = gizeh.polyline(rect, close_path=True, stroke_width=4, stroke=colors[cl])
            rect.draw(pdf)
        for bb, cl, co in zip(_bboxes, _classes, _confidence):
            w, h = len(data.classes[cl]) * 8.5 + 65, 15
            rect = gizeh.rectangle(w,
                                   h,
                                   xy=(int(bb[0] + w / 2 - 2),
                                       int(bb[1] - h / 2 + 7)),
                                   fill=(1, 1, 1, 0.5))

            rect.draw(pdf)
            txt = gizeh.text('{}: {:.2f}'.format(data.classes[cl], co),
                             'monospace',
                             fontsize=16,
                             xy=(int(bb[0]),
                                 int(bb[1])),  # - 12),
                             fill=(0., 0., 0.),
                             v_align='center',
                             h_align='left')
            txt.draw(pdf)
        pdf.flush()
        pdf.finish()


def create_pascal_label_colormap(num_classes=21):
    """
    Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns

    A colormap for visualizing segmentation results.
    """
    def bit_get(val, idx):
        """
        Gets the bit value.
        Parameters
        ----------
        val: int or numpy int array
            Input value.
        idx:
            Which bit of the input val.
        Returns
        -------
        The "idx"-th bit of input val.
        """
        return (val >> idx) & 1

    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap[:num_classes] / 255.


def main():
    # set_random_seed(0)
    # device = 'cpu'
    # model = YOLO(name='YOLOv3',
    #              model='models/yolov3-voc.cfg',
    #              device=device)
    # dataset = PascalDatasetYOLO(root_dir=['../../../Data/VOCdevkit/VOC2007/',
    #                                       '../../../Data/VOCdevkit/VOC2012/'],
    #                             class_file='../../../Data/VOCdevkit/voc.names',
    #                             dataset=['trainval',
    #                                      'trainval'],
    #                             batch_size=32,
    #                             image_size=model.default_image_size,
    #                             anchors=model.anchors,
    #                             strides=model.strides,
    #                             skip_difficult=False,
    #                             do_transforms=True,
    #                             multi_scale=model.multi_scale
    #                             )
    # show_ground_truth(model,
    #                   dataset,
    #                   show=True,
    #                   export=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    classes = read_classes('../../../Data/AMC/amc.names')
    a = find_best_anchors(classes,
                          k=5,
                          max_iter=100,
                          root_dir=['../../../Data/AMC', '../../../Data/AMC'],
                          dataset=['train', 'test'],
                          skip_truncated=False,
                          weighted=False,
                          multi_scale=False,
                          init=(8., 5.),
                          device=device)

    for x, y in a:
        print('{:.2f},{:.2f}, '.format(x, y), end='')
        # print('[{:.0f},{:.0f}],\n'.format(x / 13. * 512., y / 13. * 512.), end='')

    # draw_vector_bboxes()
    # model = YOLO(name='YOLOv2',
    #              model='models/yolov2-ss.cfg',
    #              device='cuda')
    # dataset = PascalDatasetYOLO(root_dir=['../../../Data/SS/'],
    #                             class_file='../../../Data/SS/ss.names',
    #                             dataset=['train'],
    #                             batch_size=32,
    #                             image_size=model.default_image_size,
    #                             anchors=model.anchors,
    #                             strides=model.strides,
    #                             skip_difficult=False,
    #                             do_transforms=False,
    #                             multi_scale=model.multi_scale
    #                             )
    # show_ground_truth(model, dataset)


if __name__ == '__main__':
    main()
