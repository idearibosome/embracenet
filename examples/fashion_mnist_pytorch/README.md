# MNIST / Fashion MNIST example

This folder contains example codes that employ PyTorch-based EmbraceNet on the Fashion MNIST dataset.
As in the main paper, the code divides the original images into left and right halves having a size of `4 x 28 pixels and considers them as bimodal datasets.

Place the Fashion MNIST data to the ```data/``` folder and run the code based on the following script snippets.


## Script snippets

Train the original EmbraceNet model:
```shell
python train.py --data_training --cuda_device=0 --train_path=/tmp/embracenet/without_dropout
```

Train the EmbraceNet model with modality dropout (Section 5.1 of the main paper):
```shell
python train.py --data_training --cuda_device=0 --model_dropout --train_path=/tmp/embracenet/with_dropout
```

Validate the EmbraceNet model:
```shell
python validate.py --cuda_device=-1 --restore_path=/tmp/embracenet/with_dropout/model_50000.pth
```

Validate the EmbraceNet model only on left halves (i.e., drop right halves):
```shell
python validate.py --cuda_device=-1 --model_drop_right --restore_path=/tmp/embracenet/with_dropout/model_50000.pth
```

Validate the EmbraceNet model only on right halves (i.e., drop left halves):
```shell
python validate.py --cuda_device=-1 --model_drop_left --restore_path=/tmp/embracenet/with_dropout/model_50000.pth
```

Validate the EmbraceNet model with output self-ensemble (as in [this paper](https://arxiv.org/abs/2004.13918)):
```shell
python validate.py --cuda_device=-1 --restore_path=/tmp/embracenet/with_dropout/model_50000.pth --ensemble_repeats=5
```
