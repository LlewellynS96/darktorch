# import warnings
# warnings.filterwarnings("ignore")

import torchsummary
from torch import optim
from dataset import PascalDatasetYOLO
from miscellaneous import SSDatasetYOLO
from layers import *
from darknet import YOLO
from utils import step_decay_scheduler, set_random_seed
import pickle


if __name__ == '__main__':
    set_random_seed(12345)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # device = 'cpu'

    train = False
    freeze = False
    predict = True

    model = YOLO(name='YOLOv2',
                       # model='models/yolov2-tiny-voc.cfg',
                       model='models/yolov2-voc.cfg',
                       device=device)

    # torchsummary.summary(model, (model.channels, *model.default_image_size), device=device)

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
                                  #root_dir='../../../Data/SS/',
                                  #class_file='../../../Data/SS/ss.names',
                                  dataset='test',
                                  batch_size=model.batch_size // model.subdivisions,
                                  image_size=model.default_image_size,
                                  anchors=model.anchors,
                                  strides=model.strides,
                                  do_transforms=False,
                                  multi_scale=False,
                                  return_targets=False
                                  )

    # train_data = SSDatasetYOLO(root_dir='../../../Data/SS/',
    #                            class_file='../../../Data/SS/ss.names',
    #                            dataset='train',
    #                            batch_size=model.batch_size // model.subdivisions,
    #                            image_size=model.default_image_size,
    #                            mu=[-25.52],
    #                            sigma=[8.55],
    #                            mode='spectrogram_db',
    #                            anchors=model.anchors,
    #                            return_targets=True
    #                            )
    #
    # val_data = SSDatasetYOLO(root_dir='../../../Data/SS/',
    #                          class_file='../../../Data/SS/ss.names',
    #                          dataset='test',
    #                          batch_size=model.batch_size // model.subdivisions,
    #                          image_size=model.default_image_size,
    #                          mu=[-25.52],
    #                          sigma=[8.55],
    #                          mode='spectrogram_db',
    #                          anchors=model.anchors,
    #                          return_targets=True
    #                          )
    #
    # test_data = SSDatasetYOLO(root_dir='../../../Data/SS/',
    #                           class_file='../../../Data/SS/ss.names',
    #                           dataset='test',
    #                           batch_size=1, #model.batch_size // model.subdivisions,
    #                           image_size=model.default_image_size,
    #                           mu=[-25.52],
    #                           sigma=[8.55],
    #                           mode='spectrogram_db',
    #                           anchors=model.anchors,
    #                           skip_difficult=False,
    #                           return_targets=False
    #                           )

    # model.load_weights('models/yolov3.weights')
    # model.load_weights('models/yolov2-voc.weights')
    # model.load_weights('models/darknet.weights', only_imagenet=True)
    # model.load_weights('models/yolov2-tiny.conv.13', only_imagenet=True)
    # model.load_weights('models/yolov2-tiny-voc.weights')
    # model.load_weights('models/tiny-yolo-voc_final.weights')
    # model = pickle.load(open('YOLOv2-tiny_120.pkl', 'rb'))
    # model.iteration = 90
    # model.device = device
    model = model.to(device)

    # model.set_input_dims(1)

    if freeze:
        model.freeze(freeze_last_layer=False)
    else:
        model.unfreeze()

    if train:
        set_random_seed(12345)

        model.unfreeze()
        optimizer = optim.SGD(model.get_trainable_parameters(),
                              lr=model.lr,
                              momentum=model.momentum,
                              weight_decay=model.weight_decay,
                              nesterov=True)
        scheduler = step_decay_scheduler(optimizer, steps=model.steps, scales=model.scales)

        model.fit(train_data=train_data,
                  val_data=val_data,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  epochs=120,
                  checkpoint_frequency=30)

        # model.save_weights('models/yolov2-tiny-voc-custom.weights')
        # pickle.dump(model, open('YOLOv2_tiny.pkl', 'wb'))

    if predict:
        set_random_seed(12345)

        # model.set_input_dims(1)
        model = pickle.load(open('YOLOv2_120.pkl', 'rb'))

        model.predict(dataset=test_data,
                      confidence_threshold=.5,
                      overlap_threshold=.45,
                      show=True,
                      export=False
                      )
