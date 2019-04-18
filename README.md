# EmbraceNet: A robust deep learning architecture for multimodal classification

![EmbraceNet](figures/embracenet_structure.png)

## Introduction

EmbraceNet is a novel multimodal integration architecture for deep learning models, which provides good compatibility with any network structure, in-depth consideration of correlations between different modalities, and seamless handling of missing data.
This repository contains the official TensorFlow-based implementation of the EmbraceNet model, which is explained in the following paper.
- J.-H. Choi, J.-S. Lee. EmbraceNet: A robust deep learning architecture for multimodal classification. Information Fusion, vol. 51, pp. 259-270, Nov. 2019 **[[Paper]](https://doi.org/10.1016/j.inffus.2019.02.010)**
```
@article{choi2019embracenet,
  title={EmbraceNet: A robust deep learning architecture for multimodal classification},
  author={Choi, Jun-Ho and Lee, Jong-Seok},
  journal={Information Fusion},
  volume={51},
  pages={259--270},
  year={2019},
  publisher={Elsevier}
}
```

## Dependencies

- Python 3.6+
- TensorFlow 1.8+

## Getting started

The implementation of the EmbraceNet model is in the ```embracenet/``` folder.
Copy the folder to your code base and import it.
```python
from embracenet import EmbraceNet
```
Here is a code snippet to employ EmbraceNet.
```python
# Create an EmbraceNet object.
embracenet = EmbraceNet(batch_size=16, embracement_size=256)

# Build a pre-processing network for each modality.
# Then, feed the output of the pre-processing network to EmbraceNet.
embracenet.add_modality(input_data=modality1, input_size=512)
embracenet.add_modality(input_data=modality2, input_size=128)

# Integrate the modality data.
embraced_output = embracenet.embrace()

# Build a post-processing network with inputting embraced_output.
```
Please refer to the comments in ```embracenet/embracenet.py``` for more information.

## Example

An example code that employs EmbraceNet to build a classifier of [[Fashion MNIST]](https://github.com/zalandoresearch/fashion-mnist) is included in the ```examples/fashion_mnist/``` folder.
