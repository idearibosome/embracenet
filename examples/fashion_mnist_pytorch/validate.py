import argparse
import importlib
import json
import os
import traceback
import time

import numpy as np
import torch

from dataloader import BimodalMNISTLoader
from model import BimodalMNISTModel


def main():
  # parse arguments
  parser = argparse.ArgumentParser()

  parser.add_argument('--cuda_device', type=str, default='-1', help='CUDA device index to be used in training. This parameter may be set to the environment variable \'CUDA_VISIBLE_DEVICES\'. Specify it as -1 to disable GPUs.')

  parser.add_argument('--restore_path', type=str, help='Checkpoint path to be restored.')
  parser.add_argument('--global_step', type=int, default=0, help='Global step of the restored model. Some models may require to specify this.')

  parser.add_argument('--ensemble_repeats', type=int, default=1, help='The number of repeats to obtain inferences from the same data (see "output self-ensemble" in http://arxiv.org/abs/2004.13918).')

  args, remaining_args = parser.parse_known_args()


  # initialize
  os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device

  # data loader
  print('prepare data loader - %s' % (args.dataloader))
  dataloader = BimodalMNISTLoader()
  dataloader_args, remaining_args = dataloader.parse_args(remaining_args)
  dataloader.prepare()

  # model
  print('prepare model - %s' % (args.model))
  model = BimodalMNISTModel()
  model_args, remaining_args = model.parse_args(remaining_args)
  model.prepare(is_training=False, global_step=args.global_step)

  # check remaining args
  if (len(remaining_args) > 0):
    print('WARNING: found unhandled arguments: %s' % (remaining_args))

  # model > restore
  model.restore(ckpt_path=args.restore_path)
  print('restored the model')
  

  # validate
  print('begin validation')
  num_data = dataloader.get_num_data()
  num_correct_data = 0
  for data_index in range(num_data):
    input_data, truth_label, data_name = dataloader.get_data_pair(data_index=data_index)

    model_input_list = np.repeat(np.array([input_data]), repeats=args.ensemble_repeats, axis=0)

    output_prob, output_class = model.predict(input_list=model_input_list)
    output_prob = np.mean(output_prob, axis=0)
    output_class = np.bincount(output_class).argmax()

    is_correct = (output_class == truth_label)
    num_correct_data += 1 if is_correct else 0

    if (data_index % 100 == 0):
      print('%d/%d, %s (acc: %f)' % (data_index+1, num_data, ('O' if is_correct else 'X'), (num_correct_data / (data_index+1))))

  
  # finalize
  print('finished')
  print('- accuracy: %f' % (num_correct_data/num_data))
  print('- error rate: %f' % ((num_data - num_correct_data)/num_data))



if __name__ == '__main__':
  main()