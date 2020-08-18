import torch
import torch.nn as nn


class EmbraceNet(nn.Module):
  
  def __init__(self, device, input_size_list, embracement_size=256, bypass_docking=False):
    """
    Initialize an EmbraceNet module.
    Args:
      device: A "torch.device()" object to allocate internal parameters of the EmbraceNet module.
      input_size_list: A list of input sizes.
      embracement_size: The length of the output of the embracement layer ("c" in the paper).
      bypass_docking: Bypass docking step, i.e., connect the input data directly to the embracement layer. If True, input_data must have a shape of [batch_size, embracement_size].
    """
    super(EmbraceNet, self).__init__()

    self.device = device
    self.input_size_list = input_size_list
    self.embracement_size = embracement_size
    self.bypass_docking = bypass_docking

    if (not bypass_docking):
      for i, input_size in enumerate(input_size_list):
        setattr(self, 'docking_%d' % (i), nn.Linear(input_size, embracement_size))


  def forward(self, input_list, availabilities=None, selection_probabilities=None):
    """
    Forward input data to the EmbraceNet module.
    Args:
      input_list: A list of input data. Each input data should have a size as in input_size_list.
      availabilities: A 2-D tensor of shape [batch_size, num_modalities], which represents the availability of data for each modality. If None, it assumes that data of all modalities are available.
      selection_probabilities: A 2-D tensor of shape [batch_size, num_modalities], which represents probabilities that output of each docking layer will be selected ("p" in the paper). If None, the same probability of being selected will be used for each docking layer.
    Returns:
      A 2-D tensor of shape [batch_size, embracement_size] that is the embraced output.
    """

    # check input data
    assert len(input_list) == len(self.input_size_list)
    num_modalities = len(input_list)
    batch_size = input_list[0].shape[0]
    

    # docking layer
    docking_output_list = []
    if (self.bypass_docking):
      docking_output_list = input_list
    else:
      for i, input_data in enumerate(input_list):
        x = getattr(self, 'docking_%d' % (i))(input_data)
        x = nn.functional.relu(x)
        docking_output_list.append(x)
    

    # check availabilities
    if (availabilities is None):
      availabilities = torch.ones(batch_size, len(input_list), dtype=torch.float, device=self.device)
    else:
      availabilities = availabilities.float()
    

    # adjust selection probabilities
    if (selection_probabilities is None):
      selection_probabilities = torch.ones(batch_size, len(input_list), dtype=torch.float, device=self.device)
    selection_probabilities = torch.mul(selection_probabilities, availabilities)

    probability_sum = torch.sum(selection_probabilities, dim=-1, keepdim=True)
    selection_probabilities = torch.div(selection_probabilities, probability_sum)


    # stack docking outputs
    docking_output_stack = torch.stack(docking_output_list, dim=-1)  # [batch_size, embracement_size, num_modalities]


    # embrace
    modality_indices = torch.multinomial(selection_probabilities, num_samples=self.embracement_size, replacement=True)  # [batch_size, embracement_size]
    modality_toggles = nn.functional.one_hot(modality_indices, num_classes=num_modalities).float()  # [batch_size, embracement_size, num_modalities]

    embracement_output_stack = torch.mul(docking_output_stack, modality_toggles)
    embracement_output = torch.sum(embracement_output_stack, dim=-1)  # [batch_size, embracement_size]

    return embracement_output
