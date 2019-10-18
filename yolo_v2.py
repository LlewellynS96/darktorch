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
    # device = 'cpu'

    train = True
    freeze = False
    predict = True

    model = YOLOv2tiny(model='models/yolov2-tiny-voc.cfg',
                       device=device)

    torchsummary.summary(model, (model.channels, *model.default_image_size), device=device)

    train_data = PascalDatasetYOLO(root_dir='../data/VOC2012/',
                                   classes='../data/VOC2012/voc.names',
                                   dataset='train',
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
                                 skip_difficult=True,
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

    model.load_weights('models/darknet.weights', only_imagenet=True)
    model.load_weights('models/yolov2-tiny-voc.weights')
    # model = pickle.load(open('YOLOv2-tiny_x_40.pkl', 'rb'))
    # model.device = device
    # model.detection_layers[0].device = device
    # model.detection_layers[0].anchors = model.detection_layers[0].anchors.to(device)
    # model.to(device)

    if freeze:
        model.freeze(freeze_last_layer=False)

    if train:
        torch.random.manual_seed(12345)
        np.random.seed(12345)

        optimizer = optim.SGD(model.get_trainable_parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4, nesterov=True)

        model.fit(train_data=train_data,
                  val_data=val_data,
                  optimizer=optimizer,
                  batch_size=40,
                  epochs=20,
                  multi_scale=True,
                  checkpoint_frequency=20)

        # model.save_weights('models/yolov2-tiny-voc-custom.weights')
        # pickle.dump(model, open('models/YOLOv2_tiny.pkl', 'wb'))

    model.reset_image_size(dataset=(train_data, val_data))

    torch.random.manual_seed(12345)
    np.random.seed(12345)

    if predict:
        model.predict(dataset=train_data,
                      batch_size=1,
                      confidence_threshold=0.,
                      overlap_threshold=0.45,
                      show=True,
                      export=False
                      )
