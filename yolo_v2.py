import numpy as np
import torchsummary
import pickle
from torch import optim
from dataset import PascalDatasetYOLO
from layers import *
from darknet import YOLOv2tiny
from utils import exponential_decay_scheduler, step_decay_scheduler


if __name__ == '__main__':
    torch.random.manual_seed(12345)
    np.random.seed(12345)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    train = False
    freeze = False
    predict = True

    model = YOLOv2tiny(name='YOLOv2-tiny',
                       model='models/yolov2-tiny-voc.cfg',
                       device=device)

    torchsummary.summary(model, (model.channels, *model.default_image_size), device=device)

    train_data = PascalDatasetYOLO(root_dir=['../../../Data/VOC/2007/', '../../../Data/VOC/2012/'],
                                   #  root_dir=['../../../Data/VOC/2012/'],
                                   class_file='../../../Data/VOC/2012/voc.names',
                                   dataset=['trainval', 'trainval'],
                                   #  dataset=['val'],
                                   skip_truncated=False,
                                   skip_difficult=False,
                                   image_size=model.image_size,
                                   grid_size=model.grid_size,
                                   anchors=model.anchors,
                                   do_transforms=True
                                   )

    val_data = PascalDatasetYOLO(root_dir='../../../Data/VOC/2012/',
                                 class_file='../../../Data/VOC/2012/voc.names',
                                 dataset='val',
                                 skip_truncated=False,
                                 skip_difficult=True,
                                 image_size=model.default_image_size,
                                 grid_size=model.grid_size,
                                 anchors=model.anchors,
                                 do_transforms=False
                                 )

    test_data = PascalDatasetYOLO(root_dir='../../../Data/VOC/2007/',
                                  class_file='../../../Data/VOC/2012/voc.names',
                                  dataset='test',
                                  skip_truncated=False,
                                  skip_difficult=False,
                                  image_size=model.default_image_size,
                                  grid_size=model.grid_size,
                                  anchors=model.anchors,
                                  do_transforms=False
                                  )

    model.load_weights('models/darknet.weights', only_imagenet=True)
    # model.load_weights('models/yolov2-tiny-voc.weights')
    model = pickle.load(open('YOLOv2-tiny_50.pkl', 'rb'))
    # model = model.to(device)
    # model.device = device

    if freeze:
        model.freeze(freeze_last_layer=False)

    if train:
        torch.random.manual_seed(12345)
        np.random.seed(12345)

        optimizer = optim.SGD(model.get_trainable_parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4, nesterov=True)

        scheduler = step_decay_scheduler(optimizer, warm_up=5, steps=30, decay=0.1)

        model.fit(train_data=train_data,
                  val_data=val_data,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  batch_size=32,  # preferably 64
                  epochs=50,
                  multi_scale=True,
                  checkpoint_frequency=25)

        # model.save_weights('models/yolov2-tiny-voc-custom.weights')
        # pickle.dump(model, open('YOLOv2_tiny.pkl', 'wb'))

    model.reset_image_size(dataset=(train_data, val_data))
    # model.set_image_size(608, 608, dataset=(train_data, val_data, test_data))

    torch.random.manual_seed(12345)
    np.random.seed(12345)

    if predict:
        model.predict(dataset=test_data,
                      batch_size=64,
                      confidence_threshold=0.01,
                      overlap_threshold=0.45,
                      show=False,
                      export=True
                      )
