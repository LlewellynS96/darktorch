# import warnings
# warnings.filterwarnings("ignore")

import pickle
from torch import optim
from dataset import PascalDatasetYOLO
from layers import *
from darknet import YOLO
from utils import step_decay_scheduler, set_random_seed


if __name__ == '__main__':
    set_random_seed(12345)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train = True
    predict = True

    model = YOLO(name='YOLOv2-Tiny',
                 model='models/yolov2-tiny-voc.cfg',
                 device=device)

    train_data = PascalDatasetYOLO(root_dir=['../../../Data/VOCdevkit/VOC2007/',
                                             '../../../Data/VOCdevkit/VOC2012/'],
                                   class_file='../../../Data/VOCdevkit/voc.names',
                                   dataset=['trainval',
                                            'trainval'],
                                   batch_size=model.batch_size // model.subdivisions,
                                   image_size=model.default_image_size,
                                   anchors=model.anchors,
                                   strides=model.strides,
                                   do_transforms=True,
                                   multi_scale=model.multi_scale
                                   )

    val_data = PascalDatasetYOLO(root_dir='../../../Data/VOCdevkit/VOC2007/',
                                 class_file='../../../Data/VOCdevkit/voc.names',
                                 dataset='test',
                                 batch_size=model.batch_size // model.subdivisions,
                                 image_size=model.default_image_size,
                                 anchors=model.anchors,
                                 strides=model.strides,
                                 do_transforms=True,
                                 multi_scale=model.multi_scale
                                 )

    test_data = PascalDatasetYOLO(root_dir='../../../Data/VOCdevkit/VOC2007/',
                                  class_file='../../../Data/VOCdevkit/voc.names',
                                  dataset='test',
                                  batch_size=model.batch_size // model.subdivisions,
                                  image_size=model.default_image_size,
                                  anchors=model.anchors,
                                  strides=model.strides,
                                  do_transforms=False,
                                  multi_scale=False,
                                  return_targets=False
                                  )

    model.load_weights('models/yolov2-tiny.conv.13', only_imagenet=True)
    model = model.to(device)

    if train:
        set_random_seed(12345)

        model.mini_freeze()
        optimizer = optim.SGD(model.get_trainable_parameters(),
                              lr=model.lr,
                              momentum=model.momentum,
                              weight_decay=model.weight_decay,
                              nesterov=True)
        scheduler = step_decay_scheduler(optimizer, steps=model.steps, scales=model.scales)

        losses = model.fit(train_data=train_data,
                           val_data=val_data,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           epochs=120,
                           checkpoint_frequency=120,
                           num_workers=8)

        pickle.dump(losses, open('{}_losses.pkl'.format(model.name), 'wb'))

    if predict:
        set_random_seed(12345)

        model.load_weights('models/yolov2-tiny-voc.weights')

        predictions = model.predict(dataset=test_data,
                                    confidence_threshold=.5,
                                    overlap_threshold=.45,
                                    show=False,
                                    export=False
                                    )
