import json
import os
import time

from datetime import datetime

import numpy as np
import tensorflow as tf

from dataloader import DataLoader
from model import Model, ModelConfig

FLAGS = tf.flags.FLAGS

if __name__ == '__main__':
  tf.flags.DEFINE_string('dataset_dir', 'data', 'Base path of the dataset.')
  tf.flags.DEFINE_string('train_dir', '/tmp/embracenet_fashion_mnist/', 'Base path of the trained model to be saved.')

  tf.flags.DEFINE_integer('batch_size', 64, 'Base path of the trained model to be saved.')
  tf.flags.DEFINE_float('dropout_modality_prob', 0.5, 'The probability that a modality is invalidated.')

  tf.flags.DEFINE_integer('max_steps', 20000, 'The number of maximum training steps.')
  tf.flags.DEFINE_integer('log_freq', 10, 'The frequency of logging via tf.logging.')
  tf.flags.DEFINE_integer('summary_freq', 200, 'The frequency of logging on TensorBoard.')
  tf.flags.DEFINE_integer('save_freq', 5000, 'The frequency of saving the trained model.')
  tf.flags.DEFINE_integer('save_max_keep', 100, 'The maximum number of recent trained models to keep (i.e., max_to_keep of tf.train.Saver).')
  tf.flags.DEFINE_float('sleep_ratio', 0.05, 'The ratio of sleeping time for each training step, which prevents overheating of GPUs. Specify 0 to disable sleeping.')

  tf.flags.DEFINE_string('cuda_device', '0', 'CUDA device index to be used in training. This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. Specify it as -1 to disable GPUs.')


def main(unused_argv):

  # initialize
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_device

  
  # config
  config = ModelConfig()
  
  # config > set the probability that a modality is invalidated
  config.train_dropout_a_modality_prob = FLAGS.dropout_modality_prob


  # save arguments
  arguments_path = os.path.join(FLAGS.train_dir, 'arguments.json')
  with open(arguments_path, 'w') as f:
    f.write(json.dumps(FLAGS.flag_values_dict(), sort_keys=True, indent=2))
  
  
  # data
  data_loader = DataLoader(
      is_training=True,
      base_src=FLAGS.dataset_dir
  )
  data_loader.load_data()
  
  
  # graph
  with tf.Graph().as_default():
    model = Model(
        is_training=True,
        config=config
    )    
    model.build()
    
    
    # training & session creation
    global_step = tf.train.get_or_create_global_step()
    
    logit_list = model.graph.output
    label_list = tf.placeholder(tf.int64)
    
    loss_list = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit_list, labels=label_list)
    total_loss = tf.reduce_mean(loss_list)
    tf.summary.scalar('total_loss', total_loss)
    
    opt = tf.train.AdamOptimizer(1e-3, epsilon=1e-2)
    grads = opt.compute_gradients(total_loss)
    train_op = opt.apply_gradients(grads, global_step=global_step)
    
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.save_freq)
    summary_op = tf.summary.merge_all()
        
    init = tf.global_variables_initializer()
    
    sess = tf.Session(config=tf.ConfigProto(
        log_device_placement=False,
        allow_soft_placement=True
    ))
    
    sess.run(init)
    
    tf.train.start_queue_runners(sess=sess)
    
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
    
    
    # print num trainable parameters
    print('trainable parameters')
    total_variable_parameters = 0
    for variable in tf.trainable_variables():
      shape = variable.get_shape()
      variable_parameters = 1
      for dim in shape:
        variable_parameters *= dim.value
      print(' - %s: %d' % (variable.name, variable_parameters))
      total_variable_parameters += variable_parameters
    print('total: %d' % (total_variable_parameters))
  
  
  # do iterations
  for step in range(0, FLAGS.max_steps+1):
    start_time = time.time()
    
    
    # data and label
    r_left_data_list = []
    r_right_data_list = []
    r_label_list = []
    for _ in range(FLAGS.batch_size):
      original_data = data_loader.get_random_data()
      
      r_left_data_list.append(original_data['left'])
      r_right_data_list.append(original_data['right'])
      r_label_list.append(original_data['label'])
    
    
    # feed dict
    feed_dict = {}
    feed_dict[model.feeds.batch_size] = FLAGS.batch_size
    feed_dict[model.feeds.input_left] = r_left_data_list
    feed_dict[model.feeds.input_right] = r_right_data_list
    feed_dict[label_list] = r_label_list
    
    
    # run
    if (step > 0) and (step % FLAGS.summary_freq == 0):
      _, r_class_list, r_total_loss, summary_str = sess.run(
          [train_op, model.graph.output_classes, total_loss, summary_op],
          feed_dict=feed_dict
      )
      summary_writer.add_summary(summary_str, step)
    else:
      _, r_class_list, r_total_loss = sess.run(
          [train_op, model.graph.output_classes, total_loss],
          feed_dict=feed_dict
      )
    
    duration = time.time() - start_time
    if (FLAGS.sleep_ratio > 0 and duration > 0):
      time.sleep(min(1.0, duration*FLAGS.sleep_ratio))

    assert not np.isnan(r_total_loss), 'Model diverged with loss = NaN'

    if (step % FLAGS.log_freq == 0):
      sec_per_batch = float(duration)
      accuracy = np.sum(np.array(r_class_list).flatten() == np.array(r_label_list).flatten()) / len(np.array(r_class_list).flatten())
      print('step %d, accuracy = %.6f, loss = %.6f (%.3f sec/batch)' % (step, accuracy, r_total_loss, sec_per_batch))

    if (step > 0) and (step % FLAGS.save_freq == 0):
      checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
      saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
  tf.app.run()