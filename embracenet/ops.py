import tensorflow as tf

class EmbraceNetOps:
  
  @staticmethod
  def fully_connected(input_data, input_size, output_size):
    """
    Create a fully-connected layer.
    Args:
      input_data: An input data to feed into the fully-connected layer. Must be a 2-D tensor of shape [batch_size, input_size].
      input_size: The second dimension of input_data.
      output_size: The second dimension of output data.
    Returns:
      An output data of shape [batch_size, output_size].
    """
    
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
    output_data = tf.matmul(input_data, weights) + biases
    
    return output_data
  
  
  @staticmethod
  def random_multinomial(probabilities, num_samples):
    """
    Generate multinomial samples.
    Args:
      probabilities: Probabilities of each of the different outcomes. Must be a 2-D tensor of shape [batch_size, num_classes].
      num_samples: The number of samples to draw for each batch.
    Returns:
      An output data of shape [batch_size, num_samples].
    """
    
    samples = tf.multinomial(tf.log(probabilities), num_samples)
    
    return samples
  
  