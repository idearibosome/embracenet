import os

import numpy as np
import tensorflow as tf

from dataloader import DataLoader
from model import Model, ModelConfig

FLAGS = tf.flags.FLAGS

if __name__ == '__main__':
  tf.flags.DEFINE_string('dataset_dir', 'data', 'Base path of the dataset.')
  tf.flags.DEFINE_string('restore_path', '/tmp/embracenet_fashion_mnist/model.ckpt', 'Path of the checkpoint file to be restored.')

  tf.flags.DEFINE_boolean('dropout_left', False, 'Specify this to disable employing the left halves.')
  tf.flags.DEFINE_boolean('dropout_right', False, 'Specify this to disable employing the right halves.')

  tf.flags.DEFINE_string('cuda_device', '-1', 'CUDA device index to be used in training. This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. Specify it as -1 to disable GPUs.')


def main(unused_argv):
  
  # initialize
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_device
  

  # config
  config = ModelConfig()
  config.test_modality_availabilities = [(not FLAGS.dropout_left), (not FLAGS.dropout_right)]
  
  
  # data
  data_loader = DataLoader(
      is_training=False,
      base_src=FLAGS.dataset_dir
  )
  data_loader.load_data()
  test_data_list = data_loader.get_all_data()
  
  
  # graph
  with tf.Graph().as_default():
    model = Model(
        is_training=False,
        config=config
    )    
    model.build()
    
    
    # test & session creation    
    saver = tf.train.Saver()
        
    init = tf.global_variables_initializer()
    
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=False,
        allow_soft_placement=True
    ))
    
    sess.run(init)
    
    saver.restore(sess, FLAGS.restore_path)
    
    tf.train.start_queue_runners(sess=sess)
  
  
  # run session for each data
  total_data_count = 0
  correct_data_count = 0
  
  for (data_index, original_data) in enumerate(test_data_list):
    if (data_index % 1000 == 0):
      print('- %d / %d' % (data_index, len(test_data_list)))
    
    r_left_data_list = []
    r_right_data_list = []
    r_label_list = []
    
    r_left_data_list.append(original_data['left'])
    r_right_data_list.append(original_data['right'])
    r_label_list.append(original_data['label'])
    
    feed_dict = {}
    feed_dict[model.feeds.batch_size] = 1
    feed_dict[model.feeds.input_left] = r_left_data_list
    feed_dict[model.feeds.input_right] = r_right_data_list
    
    _, r_output_classes = sess.run(
        [model.graph.output_probs, model.graph.output_classes],
        feed_dict=feed_dict
    )
    
    r_class_list = r_output_classes.flatten()
    r_label_list = np.array(r_label_list).flatten()
    
    total_data_count += len(r_class_list)
    correct_data_count += np.sum((r_class_list == r_label_list))
  
  accuracy = correct_data_count * 1.0 / total_data_count
  
  
  # print results
  print('checkpoint path: %s' % (FLAGS.restore_path))
  print('accuracy: %f' % (accuracy))
  print('error rate: %f' % (1.0 - accuracy))


if __name__ == '__main__':
  tf.app.run()