import argparse
import copy
import gzip
import os

import numpy as np
import cv2 as cv

# data loader for bimodal MNIST and Fashion-MNIST

class BimodalMNISTLoader():
  def __init__(self):
    super().__init__()

  
  def parse_args(self, args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data', help='Base path of the input data.')
    parser.add_argument('--data_training', action='store_true', help='Specify this if it is for training.')

    self.args, remaining_args = parser.parse_known_args(args=args)
    return copy.deepcopy(self.args), remaining_args


  def prepare(self):
    if (self.args.data_training):
      images_full_path = os.path.join(self.args.data_path, 'train-images-idx3-ubyte.gz')
      labels_full_path = os.path.join(self.args.data_path, 'train-labels-idx1-ubyte.gz')
    else:
      images_full_path = os.path.join(self.args.data_path, 't10k-images-idx3-ubyte.gz')
      labels_full_path = os.path.join(self.args.data_path, 't10k-labels-idx1-ubyte.gz')
    
    with gzip.open(labels_full_path, 'rb') as f:
      labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    
    with gzip.open(images_full_path, 'rb') as f:
      images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    
    self.data_list = []
    for i in range(len(labels)):
      image = (((1.0 * np.reshape(images[i, :], (28, 28))) / 255.0) * 2.0) - 1.0
      
      image_left = image[:, 0:14]
      image_right = image[:, 14:28]
      label = int(labels[i])
      
      data = {
        'left': image_left,
        'right': image_right,
        'label': label
      }
      self.data_list.append(data)
  

  def get_num_data(self):
    return len(self.data_list)
  
  
  def get_batch(self, batch_size):
    input_list = []
    truth_list = []

    for _ in range(batch_size):
      input_data, label = self.get_random_data_pair()
      input_list.append(input_data)
      truth_list.append(label)
    
    return input_list, truth_list
  

  def get_random_data_pair(self):
    # select a data
    data_index = np.random.randint(self.get_num_data())

    # retrieve data
    input_data, label, _ = self.get_data_pair(data_index=data_index)
    
    # finalize
    return input_data, label


  def get_data_pair(self, data_index):
    input_data, label = self._get_input_data(index=data_index)

    # finalize
    return input_data, label, data_index


  def _get_input_data(self, index):
    data = self.data_list[index]
    return copy.deepcopy([np.array([data['left']]), np.array([data['right']])]), copy.deepcopy(data['label'])

