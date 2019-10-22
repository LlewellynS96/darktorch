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

    train = True
    freeze = False
    predict = True

    model = YOLOv2tiny(name='YOLOv2-tiny',
                       model='models/yolov2-tiny-voc.cfg',
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
    # model = pickle.load(open('YOLOv2-tiny_80.pkl', 'rb'))
    # model = model.to(device)
    # model.device = device
    # model.detection_layers[0] = model.detection_layers[0].to(device)
    # model.detection_layers[0].device = device
    # model.detection_layers[0].anchors = model.detection_layers[0].anchors.to(device)

    if freeze:
        model.freeze(freeze_last_layer=False)

    if train:
        torch.random.manual_seed(12345)
        np.random.seed(12345)

        optimizer = optim.SGD(model.get_trainable_parameters(), lr=1e-1, momentum=0.9, weight_decay=5e-4, nesterov=True)

        target_lr = optimizer.defaults['lr']
        initial_lr = 1e-2
        warm_up = 1
        step_size = 0.5
        step_frequency = 20
        gradient = (target_lr - initial_lr) / warm_up

        def f(e):
            if e < warm_up:
                return gradient * e + initial_lr
            else:
                return target_lr * step_size ** ((e - warm_up) // step_frequency)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
        # scheduler = None

        model.fit(train_data=train_data,
                  val_data=val_data,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  batch_size=40,
                  epochs=3,
                  multi_scale=False,
                  checkpoint_frequency=20)

        # model.save_weights('models/yolov2-tiny-voc-custom.weights')
        pickle.dump(model, open('YOLOv2_tiny.pkl', 'wb'))

    model.reset_image_size(dataset=(train_data, val_data))
    # model.set_image_size(320, 320, dataset=(train_data, val_data, test_data))

    torch.random.manual_seed(12345)
    np.random.seed(12345)

    if predict:
        model.predict(dataset=test_data,
                      batch_size=64,
                      confidence_threshold=0.005,
                      overlap_threshold=0.45,
                      show=False,
                      export=True
                      )
