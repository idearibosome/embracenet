import gzip
import os

import numpy as np

class DataLoader:
  
  def __init__(self, is_training, base_src):
    self.is_training = is_training
    self.base_src = base_src
    
    self.data_list = []
  
  
  def load_data(self):
    if (self.is_training):
      images_full_path = os.path.join(self.base_src, 'train-images-idx3-ubyte.gz')
      labels_full_path = os.path.join(self.base_src, 'train-labels-idx1-ubyte.gz')
    else:
      images_full_path = os.path.join(self.base_src, 't10k-images-idx3-ubyte.gz')
      labels_full_path = os.path.join(self.base_src, 't10k-labels-idx1-ubyte.gz')
    
    with gzip.open(labels_full_path, 'rb') as f:
      labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    
    with gzip.open(images_full_path, 'rb') as f:
      images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    
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
  
  
  def get_all_data(self):
    return self.data_list
  
  
  def get_random_data(self):
    data_index = np.random.randint(low=0, high=len(self.data_list))
    return self.data_list[data_index]
  
    
    