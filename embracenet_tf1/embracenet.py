import tensorflow as tf

from .ops import EmbraceNetOps

class EmbraceNetObject(object):
  pass

class EmbraceNet:
  
  def __init__(self, batch_size=1, embracement_size=256):
    """
    Initialize an EmbraceNet model.
    Args:
      batch_size: Batch size of the input data.
      embracement_size: Length of the output of the embracement layer ("c" in the paper).
    """
    
    self.batch_size = batch_size
    self.embracement_size = embracement_size
    
    self.selection_probabilities = None
    
    self.graph = EmbraceNetObject()
    self.graph.modalities = []
    
    self.feeds = EmbraceNetObject()
    self.feeds.modalities = []
    
  
  def add_modality(self, input_data, input_size, bypass_docking=False):
    """
    Add a modality to EmbraceNet.
    Args:
      input_data: An input data to feed into EmbraceNet. Must be a 2-D tensor of shape [batch_size, input_size].
      input_size: The second dimension of input_data.
      bypass_docking: Bypass docking step, i.e., connect the input data directly to the embracement layer. If True, input_data must have a shape of [batch_size, embracement_size].
    """
    
    # check input data
    tf_assertions = []
    tf_assertions.append(tf.assert_rank(input_data, 2))
    tf_assertions.append(tf.assert_equal(tf.shape(input_data)[0], self.batch_size))
    with tf.control_dependencies(tf_assertions):
      input_data = tf.identity(input_data)
    
    
    with tf.variable_scope('embracenet'):
      # construct docking layer
      modality_index = len(self.graph.modalities)
      modality_graph = EmbraceNetObject()
      modality_feeds = EmbraceNetObject()
      
      with tf.variable_scope('docking/%d' % modality_index):
        docking_input = input_data
        
        if (bypass_docking):
          modality_graph.docking_output = docking_input
        else:
          docking_output = tf.layers.dense(docking_input, units=self.embracement_size, kernel_initializer=None, bias_initializer=None)
          docking_output = tf.nn.relu(docking_output)
          modality_graph.docking_output = docking_output
            
      
      # finalize
      self.graph.modalities.append(modality_graph)
      self.feeds.modalities.append(modality_feeds)
  
  
  def set_selection_probabilities(self, selection_probabilities=None):
    """
    Set selection probabilities.
    Args:
      selection_probabilities: A 2-D tensor of shape [batch_size, num_modalities], which represents probabilities that output of each docking layer will be selected ("p" in the paper). If None, the same probability of being selected will be used for each docking layer.
    Returns:
      A 2-D tensor of shape [batch_size, embracement_size] that is the embraced output.
    """    
    self.selection_probabilities = selection_probabilities
  
  
  def embrace(self, modality_availabilities=None):
    """
    Perform the embracement operation to outputs of the docking layers.
    Args:
      modality_availabilities: A 2-D tensor of shape [batch_size, num_modalities], which represents the availability of data for each modality. If None, it assumes that data of all modalities are available.
    Returns:
      A 2-D tensor of shape [batch_size, embracement_size] that is the embraced output.
    """
    
    # check docking modalities
    num_modalities = len(self.graph.modalities)
    assert (num_modalities > 0), 'At least 1 modality should be added.'
    
    
    # check selection probabilities
    selection_probabilities = self.selection_probabilities
    
    if (selection_probabilities == None):
      selection_probabilities = tf.ones([self.batch_size, num_modalities], dtype=tf.float32)
    
    tf_assertions = []
    tf_assertions.append(tf.assert_rank(selection_probabilities, 2))
    tf_assertions.append(tf.assert_equal(tf.shape(selection_probabilities)[1], num_modalities))
    with tf.control_dependencies(tf_assertions):
      selection_probabilities = tf.identity(selection_probabilities, name='selection_probabilities')
    
    
    # check modality availabilities
    if (modality_availabilities == None):
      modality_availabilities = tf.ones([self.batch_size, num_modalities], dtype=tf.bool)
    
    tf_assertions = []
    tf_assertions.append(tf.assert_rank(modality_availabilities, 2))
    tf_assertions.append(tf.assert_equal(tf.shape(modality_availabilities)[1], num_modalities))
    with tf.control_dependencies(tf_assertions):
      modality_availabilities = tf.cast(modality_availabilities, tf.bool)
      modality_availabilities = tf.identity(modality_availabilities, name='modality_availabilities')
    
    
    # adjust selection probabilities
    modality_availability_mask = tf.cast(modality_availabilities, tf.float32)
    selection_probabilities = tf.multiply(selection_probabilities, modality_availability_mask)
    
    probabilty_sum = tf.reduce_sum(selection_probabilities, axis=1, keepdims=True)
    selection_probabilities = tf.div(selection_probabilities, probabilty_sum)
    self.graph.selection_probabilities = selection_probabilities
    
    
    # stack docking outputs
    docking_output_stack = tf.stack([modality.docking_output for modality in self.graph.modalities], axis=0)
    
    
    # embrace
    modality_indices = EmbraceNetOps.random_multinomial(selection_probabilities, self.embracement_size)
    self.graph.modality_indices = modality_indices
    modality_toggles = tf.one_hot(modality_indices, num_modalities, axis=0, dtype=tf.float32)
    
    embracement_output_stack = tf.multiply(docking_output_stack, modality_toggles)
    embracement_output = tf.reduce_sum(embracement_output_stack, axis=0)
    
    self.graph.output = embracement_output
    
    
    return self.graph.output
    

