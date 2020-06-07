import os
import random
import torch
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from darknet import YOLO
from utils import export_prediction, jaccard, read_classes, get_annotations, to_numpy_image, add_bbox_to_image, set_random_seed
from dataset import PascalDatasetYOLO


USE_LETTERBOX = False
IOU_MATCH_THRESHOLD = 0.05
SMALL_THRESHOLD = 0.005
MULTI_SCALE_FREQ = 10


class SSDatasetYOLO(Dataset):

    def __init__(self, anchors, class_file, root_dir, mu, sigma, mode, dataset, batch_size=1, skip_truncated=False, do_transforms=False,
                 image_size=(512, 512), skip_difficult=True, stride=32, return_targets=True):

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

        self.anchors = anchors.clone().detach().cpu()
        self.num_anchors = len(self.anchors)

        self.do_transforms = do_transforms
        self.return_targets = return_targets

        self.batch_size = batch_size
        self.stride = stride
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
        self.grid_size = np.repeat([s // self.stride for s in self.default_image_size], self.n).reshape(-1, 2)

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
            target = np.zeros((self.grid_size[index, 1], self.grid_size[index, 0], self.num_anchors * self.num_features),
                              dtype=np.float32)
            cell_dims = self.stride, self.stride

            anchors = torch.zeros((self.num_anchors, 4))
            anchors[:, 2:] = self.anchors.clone()

            target[:, np.arange(self.grid_size[index][0]), 0::self.num_features] = np.arange(self.grid_size[index][0])[
                                                                                   None, :, None] + 0.5
            target[:, :, 1::self.num_features] = np.arange(self.grid_size[index][1])[:, None, None] + 0.5
            target[:, :, 2::self.num_features] = anchors[:, 2]
            target[:, :, 3::self.num_features] = anchors[:, 3]

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
                xmin /= cell_dims[0]
                xmax /= cell_dims[0]
                ymin /= cell_dims[1]
                ymax /= cell_dims[1]
                if xmax - xmin < (SMALL_THRESHOLD * cell_dims[0]) or ymax - ymin < (SMALL_THRESHOLD * cell_dims[1]):
                    continue
                idx = int(np.floor((xmax + xmin) / 2.)), int(np.floor((ymax + ymin) / 2.))

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

            target = torch.tensor(target)
            target = target.permute(2, 0, 1)

            return data, data_info, target
        else:
            return data, data_info

    def __len__(self):
        return len(self.data)

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
                      overlap_threshold=0.5,
                      show=True,
                      export=True,
                      num_workers=0):

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
                                                                           nms=True)

            if show:
                for i, data in enumerate(zip(image_info['id'], images)):
                    idx, image = data
                    if image.shape[0] == 3:
                        image = to_numpy_image(image, size=(width, height))
                    else:
                        mu = dataset.mu[0]
                        sigma = dataset.sigma[0]
                        image = to_numpy_image(image[0], size=(width, height), mu=mu, sigma=sigma, normalised=False)
                    mask = np.array(image_idx) == idx
                    for bbox, cls, confidence in zip(bboxes[mask], classes[mask], confidences[mask]):
                        name = dataset.classes[cls]
                        add_bbox_to_image(image, bbox, None, name, 2, [0., 255., 0.])
                    plt.imshow(image)
                    plt.axis('off')
                    plt.show()

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


def main():
    set_random_seed(0)
    device = 'cpu'
    # model = YOLO(name='YOLOv2-tiny',
    #              model='models/yolov2-tiny-voc.cfg',
    #              device=device)
    model = YOLO(name='YOLOv3',
                 model='models/yolov3-voc.cfg',
                 device=device)
    dataset = PascalDatasetYOLO(root_dir=['../../../Data/VOCdevkit/VOC2007/',
                                          '../../../Data/VOCdevkit/VOC2012/'],
                                class_file='../../../Data/VOCdevkit/voc.names',
                                dataset=['trainval',
                                         'trainval'],
                                batch_size=32,
                                image_size=model.default_image_size,
                                anchors=model.anchors,
                                strides=model.strides,
                                skip_difficult=False,
                                do_transforms=True,
                                multi_scale=model.multi_scale
                                )
    show_ground_truth(model,
                      dataset,
                      show=True,
                      export=False)


if __name__ == '__main__':
    main()