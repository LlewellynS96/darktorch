# import warnings
# warnings.filterwarnings("ignore")

import torchsummary
from torch import optim
from dataset import PascalDatasetYOLO
from layers import *
from darknet import YOLOv2tiny
from utils import step_decay_scheduler, set_random_seed
import pickle


if __name__ == '__main__':
    set_random_seed(12345)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    train = True
    freeze = False
    predict = True
    fp16 = False

    model = YOLOv2tiny(name='YOLOv2-tiny',
                       model='models/yolov2-tiny-voc.cfg',
                       device=device)

    torchsummary.summary(model, (model.channels, *model.default_image_size), device=device)

    train_data = PascalDatasetYOLO(root_dir=['../../../Data/VOCdevkit/VOC2007/',
                                             '../../../Data/VOCdevkit/VOC2012/'],
                                   class_file='../../../Data/VOCdevkit/voc.names',
                                   dataset=['trainval',
                                            'trainval'],
                                   batch_size=model.batch_size // model.subdivisions,
                                   image_size=model.default_image_size,
                                   anchors=model.anchors,
                                   do_transforms=True,
                                   multi_scale=model.multi_scale
                                   )

    val_data = PascalDatasetYOLO(root_dir='../../../Data/VOCdevkit/VOC2007/',
                                 class_file='../../../Data/VOCdevkit/voc.names',
                                 dataset='test',
                                 batch_size=model.batch_size // model.subdivisions,
                                 image_size=model.default_image_size,
                                 anchors=model.anchors,
                                 do_transforms=True,
                                 multi_scale=model.multi_scale
                                 )

    test_data = PascalDatasetYOLO(root_dir='../../../Data/VOCdevkit/VOC2007/',
                                  class_file='../../../Data/VOCdevkit/voc.names',
                                  dataset='test',
                                  batch_size=model.batch_size // model.subdivisions,
                                  image_size=model.default_image_size,
                                  anchors=model.anchors,
                                  do_transforms=False,
                                  multi_scale=False,
                                  return_targets=False
                                  )

    # model.load_weights('models/darknet.weights', only_imagenet=True)
    model.load_weights('models/yolov2-tiny.conv.13', only_imagenet=True)
    # model.load_weights('models/yolov2-tiny-voc.weights')
    # model.load_weights('models/tiny-yolo-voc_final.weights')
    # model = pickle.load(open('YOLOv2-tiny_ce.pkl', 'rb'))
    # model.iteration = 90
    # model.device = device
    model = model.to(device)

    if freeze:
        model.freeze(freeze_last_layer=False)
    else:
        model.unfreeze()

    if fp16:
        model.to_fp16()
    else:
        model.to_fp32()

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
                  val_data=train_data,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  epochs=120,
                  checkpoint_frequency=30)

        # model.save_weights('models/yolov2-tiny-voc-custom.weights')
        # pickle.dump(model, open('YOLOv2_tiny.pkl', 'wb'))

    if predict:
        set_random_seed(12345)

        model.predict(dataset=test_data,
                      confidence_threshold=0.001,
                      overlap_threshold=.45,
                      show=False,
                      export=True
                      )
