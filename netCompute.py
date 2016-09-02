import tensorflow as tf
from parameters import constants, shared
import numpy as np

def conv_weights_bias(kernel_shape, bias_shape, rng, name_prefix):
    weights = tf.Variable(np.asarray(
                                rng.normal(size = kernel.shape, scale = constants.weigthInitStdev)
                            ).astype(np.float32),
                        name = name_prefix + "_kernel")

    biases    = np.empty(bias_shape, dtype = np.float32)
    biases[:] = constants.biasInitValue
    biases    = tf.Variable(biases, name = name_prefix + "_biases")

    return weights, biases


def fc_weights_bias(nb_inputs, nb_outputs, rng, name_prefix):
    weights = tf.Variable(np.asarray(
                                rng.normal(size = [nb_inputs, nb_outputs],
                                        scale = constants.weigthInitStdev
                                    )
                                ).astype(np.float32),
                          name = name_prefix+"_weights"
                    )

    biases    = np.empty([nb_outputs], dtype = np.float32)
    biases[:] = constants.biasInitValue
    biases    = tf.Variable(biases, name = name_prefix + "_biases")

    return weights, biases


class WeightHolder:
    def __init__(self, nb_actions, name):
        self.name = name
        rng = np.random.RandomState(42)
        
        self.conv1_weights, self.conv1_bias = conv_weights_bias(constants.conv1_shape,
                                                                [constants.conv1_zwidth],
                                                                rng,
                                                                name+"_conv1"
                                                            )
        self.conv2_weights, self.conv2_bias = conv_weights_bias(constants.conv2_shape, 
                                                                [constants.conv2_zwidth],
                                                                rng,
                                                                name+"_conv2"
                                                            )
        self.fcl1_weights, self.fcl1_bias = fc_weights_bias(constants.cnn_output_size, 
                                                            constants.fcl1_nbUnit)
        self.fcl2_weights, self.fcl2_bias = fc_weights_bias(constants.fcl1_nbUnit, nb_actions)

        self.params = [ self.conv1_weights, 
                        self.conv1_bias,
                        self.conv2_weights,
                        self.conv2_bias,
                        self.fcl1_weights,
                        self.fcl1_bias,
                        self.fcl2_weights,
                        self.fcl2_weights ]

        self.placeholders = []
        for param in self.params():
            placeholders.append(tf.placeholder(tf.float32, param.get_shape()))

        self.updates = []
        for param, placeholder in zip(self.params, self.placeholders):
            self.updates.append(param.assign(placeholder))

        self.learning_rate = tf.placeholder(tf.float32)
        self.rmsprop = tf.train.RMSPropOptimizer(self.learning_rate,
                                                 decay   = constants.decay_factor,
                                                 epsilon = constants.epsilon_cancel)
        
    def update(self, network, session):
        for placeholder, updatenode, targetparam in zip(self.placeholders, self.updates, network.params):
            session.run(updatenode, feed_dict = {placeholder : targetparam})


class AgentSubNet:
    def __init__(self, network, name_prefix, inputs):
        self.sess = session
    
        self.inputs = inputs 

        self.conv1_weights = tf.Variable(network.conv1_weights)
        self.conv1_bias    = tf.Variable(network.conv1_bias)
        self.conv2_weights = tf.Variable(network.conv2_weights)
        self.conv2_bias    = tf.Variable(network.conv2_bias)
        self.fcl1_weights  = tf.Variable(network.fcl1_weights)
        self.fcl1_bias     = tf.Variable(network.fcl1_bias)
        self.fcl2_weights  = tf.Variable(network.fcl2_weights)
        self.fcl2_bias     = tf.Variable(network.fcl2_bias)

        self.conv1 = tf.nn.relu(tf.nn.conv2d(self.inputs,
                                             self.conv1_weights,
                                             constants.conv1_strides,
                                             padding = "VALID") + 
                                    self.conv1_bias)
        self.conv2 = tf.nn.relu(tf.nn.conv2d(self.inputs,
                                             self.conv2_weights,
                                             constants.conv2_strides,
                                             padding = "VALID") +
                                        self.conv2_bias)

        self.flatconv2 = tf.reshape(self.conv2, [-1, constants.cnn_output_size])


        self.input = tf.placeholder(dtype = tf.float32)
        
