import numpy as np
import tensorflow as tf

import context
from embracenet import EmbraceNet

class ModelConfig(object):
  pass

class ModelObject(object):
  pass

class Model:
  
  DEFAULT_CONFIG = {
    'batch_size': 16,
    'num_modalities': 2,
    'num_classes': 10,
    'embracement_size': 512,
    'train_max_step': 50000,
    'train_dropout_modality_prob': 0.5
  }
  
  def __init__(self, is_training, config):
    self.is_training = is_training
    
    
    # config
    for (key, value) in self.DEFAULT_CONFIG.items():
      if not hasattr(config, key):
        setattr(config, key, value)
    self.config = config

    
    # model
    self.feeds = ModelObject()
    self.feeds.batch_size = tf.placeholder(tf.int32, shape=(), name='batch_size')
    self.feeds.single = []
    
    self.graph = ModelObject()
    self.graph.single = []
    
    self.embracenet = EmbraceNet(
        batch_size=self.feeds.batch_size,
        embracement_size=self.config.embracement_size
    )
  
  
  def build(self):
    # feeds
    self.feeds.input_left = tf.placeholder(tf.float32)
    self.feeds.input_right = tf.placeholder(tf.float32)
    
    
    # configure EmbraceNet parameters
    def dropped_modality_availabilities():
      multinomial_samples = tf.multinomial(tf.log([[1.0] * self.config.num_modalities]), self.feeds.batch_size) # [1, batch_size]
      multinomial_samples = tf.squeeze(multinomial_samples) # [batch_size]
      availabilities = tf.one_hot(multinomial_samples, self.config.num_modalities, on_value=True, off_value=False, axis=-1, dtype=tf.bool) # [batch_size, num_modalities]
      return availabilities
    
    if (self.is_training):
      dropout_prob = tf.random_uniform([])
      modality_availabilities = tf.cond(
          tf.less(dropout_prob, self.config.train_dropout_modality_prob),
          dropped_modality_availabilities,
          lambda: tf.ones([self.feeds.batch_size, self.config.num_modalities], dtype=tf.bool)
      )
    else:
      modality_availabilities = tf.ones([self.feeds.batch_size, self.config.num_modalities], dtype=tf.bool)
    
    if (not self.is_training):
      modality_availabilities = tf.logical_and(modality_availabilities, np.expand_dims(self.config.test_modality_availabilities, axis=0))
    
    self.modality_availabilities = modality_availabilities
    
    
    # pre-embracement layers
    with tf.variable_scope('left'):
      pre_output, pre_output_size = self._pre_embracement_layers(self.feeds.input_left)
      self.embracenet.add_modality(pre_output, pre_output_size)
      
    with tf.variable_scope('right'):
      pre_output, pre_output_size = self._pre_embracement_layers(self.feeds.input_right)
      self.embracenet.add_modality(pre_output, pre_output_size)
    
    
    # embracement layers
    embracenet_output = self.embracenet.embrace(
        modality_availabilities=self.modality_availabilities
    )
    
    
    # post-embracement layers
    with tf.variable_scope('final'):
      final_output, final_probs, final_classes = self._post_embracement_layers(embracenet_output)
      
      self.graph.output = final_output
      self.graph.output_probs = final_probs
      self.graph.output_classes = final_classes
  
  
  def _pre_embracement_layers(self, input_data):
    output_size = 28 * 14 * 64
    
    with tf.variable_scope('conv1'):
      conv1_input = tf.reshape(input_data, [-1, 28, 14, 1])
      conv1_output = self._cnn_2d_inference(conv1_input, 1, 5, 64, padding='SAME')
      conv1_output = tf.nn.relu(conv1_output)
      conv1_output_pooled = tf.nn.max_pool(conv1_output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')
      output_size = output_size // 4
    
    with tf.variable_scope('conv2'):
      conv2_input = conv1_output_pooled
      conv2_output = self._cnn_2d_inference(conv2_input, 64, 5, 64, padding='SAME')
      conv2_output = tf.nn.relu(conv2_output)
    
    final_output = tf.reshape(conv2_output, [-1, output_size])
    return final_output, output_size
  
  
  def _post_embracement_layers(self, input_data):
    post_input = input_data
    post_output = self._fully_connected(post_input, self.embracenet.embracement_size, self.config.num_classes)
    
    post_output_probs = tf.nn.softmax(post_output)
    post_output_classes = tf.argmax(post_output_probs, axis=1)
    
    return post_output, post_output_probs, post_output_classes
    
  
  @staticmethod
  def _cnn_2d_inference(input_data, input_depth, conv_size, conv_depth, padding='SAME'):
    kernel = tf.get_variable(
        'weights', 
        [conv_size, conv_size, input_depth, conv_depth],
        dtype=tf.float32
    )
    biases = tf.get_variable(
        'biases', 
        [conv_depth], 
        dtype=tf.float32
    )
    conv = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding=padding)
    conv = tf.nn.bias_add(conv, biases)
    
    return conv
  
  
  @staticmethod
  def _fully_connected(input_data, input_size, output_size):
    reshape = tf.reshape(input_data, [-1, input_size])
    weights = tf.get_variable(
        'weights', 
        [input_size, output_size], 
        dtype=tf.float32
    )
    biases = tf.get_variable(
        'biases', 
        [output_size], 
        dtype=tf.float32
    )
    output_data = tf.matmul(reshape, weights) + biases
    
    return output_data

