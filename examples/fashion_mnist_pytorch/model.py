import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim

import context
from embracenet_pytorch import EmbraceNet


class BimodalMNISTModel():

  def parse_args(self, args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_learning_rate', type=float, default=1e-4, help='Initial learning rate.')
    parser.add_argument('--model_dropout', action='store_true', help='Specify this to employ modality dropout during training.')
    parser.add_argument('--model_drop_left', action='store_true', help='Specity this to drop left-side modality.')
    parser.add_argument('--model_drop_right', action='store_true', help='Specity this to drop right-side modality.')

    self.args, remaining_args = parser.parse_known_args(args=args)
    return copy.deepcopy(self.args), remaining_args
  

  def prepare(self, is_training, global_step=0):
    # config. parameters
    self.global_step = global_step
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # PyTorch model
    self.model = EmbraceNetBimodalModule(device=self.device, is_training=is_training, args=self.args)
    if (is_training):
      self.optim = optim.Adam(
        filter(lambda p: p.requires_grad, self.model.parameters()),
        lr=self.args.model_learning_rate
      )
      self.loss_fn = nn.NLLLoss()

    # configure device
    self.model = self.model.to(self.device)
      

  def save(self, base_path):
    save_path = os.path.join(base_path, 'model_%d.pth' % (self.global_step))
    torch.save(self.model.state_dict(), save_path)


  def restore(self, ckpt_path):
    self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
  

  def get_model(self):
    return self.model


  def train_step(self, input_list, truth_list, summary=None):
    # numpy to torch
    input_tensor = torch.as_tensor(input_list, dtype=torch.float, device=self.device)  # [batch_size, 2, 1, 28, 14]
    truth_tensor = torch.as_tensor(truth_list, dtype=torch.long, device=self.device)

    # get log softmax and calculate loss
    output_tensor = self.model(input_tensor)
    loss = self.loss_fn(output_tensor, truth_tensor)

    # adjust learning rate
    lr = self.args.model_learning_rate
    for param_group in self.optim.param_groups:
      param_group['lr'] = lr

    # do back propagation
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()

    # finalize
    self.global_step += 1

    # write summary
    if (summary is not None):
      summary.add_scalar('loss', loss, self.global_step)
      summary.add_scalar('lr', lr, self.global_step)

    return loss.item()

  def predict(self, input_list):
    # numpy to torch
    input_tensor = torch.tensor(input_list, dtype=torch.float32, device=self.device)

    # get output
    output_tensor = self.model(input_tensor)

    # finalize
    class_list = output_tensor.argmax(dim=-1).detach().cpu().numpy()
    prob_list = output_tensor.detach().cpu().numpy()

    # finalize
    return prob_list, class_list



class EmbraceNetBimodalModule(nn.Module):
  def __init__(self, device, is_training, args):
    super(EmbraceNetBimodalModule, self).__init__()
    
    # input parameters
    self.device = device
    self.is_training = is_training
    self.args = args

    # pre embracement layers
    self.pre_left = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
      nn.ReLU(inplace=True)
    )
    self.pre_right = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
      nn.ReLU(inplace=True)
    )
    self.pre_output_size = (28 * 14 * 64) // 4

    # embracenet
    self.embracenet = EmbraceNet(device=self.device, input_size_list=[self.pre_output_size, self.pre_output_size], embracement_size=512)

    # post embracement layers
    self.post = nn.Linear(in_features=512, out_features=10)

  
  def forward(self, x):
    # separate x into left/right
    x_left = x[:, 0]
    x_left = self.pre_left(x_left)
    x_left = x_left.view(-1, self.pre_output_size)

    x_right = x[:, 1]
    x_right = self.pre_right(x_right)
    x_right = x_right.view(-1, self.pre_output_size)

    # drop left or right modality
    availabilities = None
    if (self.args.model_drop_left or self.args.model_drop_right):
      availabilities = torch.ones([x.shape[0], 2], device=self.device)
      if (self.args.model_drop_left):
        availabilities[:, 0] = 0
      if (self.args.model_drop_right):
        availabilities[:, 1] = 0

    # dropout during training
    if (self.is_training and self.args.model_dropout):
      dropout_prob = torch.rand(1, device=self.device)[0]
      if (dropout_prob >= 0.5):
        target_modalities = torch.round(torch.rand([x.shape[0]], device=self.device)).to(torch.int64)
        availabilities = nn.functional.one_hot(target_modalities, num_classes=2).float()

    # embrace
    x_embrace = self.embracenet([x_left, x_right], availabilities=availabilities)

    # employ final layers
    x = self.post(x_embrace)

    # output softmax
    return nn.functional.log_softmax(x, dim=-1)
