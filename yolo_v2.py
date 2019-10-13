import numpy as np
import torchsummary
import pickle
from torch import optim
from dataset import PascalDatasetYOLO
from layers import *
from darknet import YOLOv2tiny


if __name__ == '__main__':
    torch.random.manual_seed(12345)
    np.random.seed(12345)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train = False
    freeze = False
    predict = True

    model = YOLOv2tiny(model='models/yolov2-tiny-voc.cfg',
                       device=device)

    torchsummary.summary(model, (model.channels, *model.default_image_size))

    train_data = PascalDatasetYOLO(root_dir='../data/VOC2012/',
                                   classes='../data/VOC2012/voc.names',
                                   dataset='trainval',
                                   skip_truncated=False,
                                   skip_difficult=False,
                                   image_size=model.image_size,
                                   grid_size=model.grid_size,
                                   anchors=model.anchors,
                                   do_transforms=True
                                   )

    val_data = PascalDatasetYOLO(root_dir='../data/VOC2012/',
                                 classes='../data/VOC2012/voc.names',
                                 dataset='val',
                                 skip_truncated=False,
                                 skip_difficult=False,
                                 image_size=model.default_image_size,
                                 grid_size=model.grid_size,
                                 anchors=model.anchors,
                                 do_transforms=False
                                 )

    test_data = PascalDatasetYOLO(root_dir='../data/VOC2007/',
                                  classes='../data/VOC2012/voc.names',
                                  dataset='test',
                                  skip_truncated=False,
                                  skip_difficult=False,
                                  image_size=model.default_image_size,
                                  grid_size=model.grid_size,
                                  anchors=model.anchors,
                                  do_transforms=False
                                  )

    # model.load_weights('models/darknet.weights', only_imagenet=True)
    model.load_weights('models/yolov2-tiny-voc.weights')

    if freeze:
        model.freeze(freeze_last_layer=False)

    if train:
        torch.random.manual_seed(12345)
        np.random.seed(12345)

        optimizer = optim.SGD(model.get_trainable_parameters(), lr=1e-5, momentum=0.99)

        model.fit(train_data=train_data,
                  val_data=val_data,
                  optimizer=optimizer,
                  batch_size=20,
                  epochs=80,
                  multi_scale=True,
                  checkpoint_frequency=80)

        model.save_weights('models/yolov2-tiny-voc-custom.weights')

    model.reset_image_size(dataset=(train_data, val_data))

    torch.random.manual_seed(12345)
    np.random.seed(12345)

    model = pickle.load(open('YOLOv2-tiny_60.pkl', 'rb'))

    if predict:
        model.predict(dataset=train_data,
                      batch_size=20,
                      confidence_threshold=0.6,
                      overlap_threshold=0.3,
                      show=True,
                      export=False
                      )
