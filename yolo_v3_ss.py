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

    model = YOLO(name='YOLOv3',
                 model='models/yolov3-ss.cfg',
                 device=device)

    train_data = PascalDatasetYOLO(root_dir='../../../Data/SS/',
                                   class_file='../../../Data/SS/ss.names',
                                   dataset='train',
                                   batch_size=model.batch_size // model.subdivisions,
                                   image_size=model.default_image_size,
                                   anchors=model.anchors,
                                   strides=model.strides,
                                   do_transforms=False,
                                   multi_scale=model.multi_scale
                                   )

    val_data = PascalDatasetYOLO(root_dir='../../../Data/SS/',
                                 class_file='../../../Data/SS/ss.names',
                                 dataset='train',
                                 batch_size=model.batch_size // model.subdivisions,
                                 image_size=model.default_image_size,
                                 anchors=model.anchors,
                                 strides=model.strides,
                                 do_transforms=False,
                                 multi_scale=model.multi_scale
                                 )

    test_data = PascalDatasetYOLO(root_dir='../../../Data/SS/',
                                  class_file='../../../Data/SS/ss.names',
                                  dataset='test',
                                  batch_size=model.batch_size // model.subdivisions,
                                  image_size=model.default_image_size,
                                  anchors=model.anchors,
                                  strides=model.strides,
                                  do_transforms=False,
                                  multi_scale=model.multi_scale
                                  )

    model.load_weights('models/darknet53_448.weights', only_imagenet=True)
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

        model.predict(dataset=test_data,
                      confidence_threshold=.001,
                      overlap_threshold=.45,
                      show=False,
                      export=True
                      )
