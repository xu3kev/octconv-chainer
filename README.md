# octconv-chainer
Implementation of octave convolution in Chainer (https://arxiv.org/abs/1904.05049)

See `chainer-cifar10/` for training with cifar10 dataset. (The resnet model with octave convolution is `chainer-cifar10/models/oct_resnet.py`.)
Use `train_imagenet_multi.py` for training imagenet. See chainercv imagenet training example for details.

ResNet50 imagenet training for 90 epochs:

| alpha         | validation accuracy | theoretical flop cost|
| ------------- | ------------------- |----------------------|
| 0 (origin)    | 0.762               |    100%              |
| 0.25          | 0.762               |    67%               |
| 0.5           | 0.757               |    44%               |

Reference: https://github.com/d-li14/octconv.pytorch
