import tensorflow as tf
from parameters import constants, shared
import numpy as np

def conv_weights_bias(kernel_shape, rng, name_prefix):
    """Create weights and biases shared variables for a convolution layer
        
    biases are initialized with a constant value of constants.biasInitValue
    weights are initialized following a gaussian distribution of mean 0 and standard deviation of constants.weightInitStdev

    Args:
        kernel_shape: List of simension of the kernel [filter_height, filter_width, in_channels, out_channels]
        rng: a numpy random number generator
        name_prefix: The name prefix for the operators

    Returns:
        The weights shared variable and the biases shared variable
    """
    weights = tf.Variable(np.asarray(
                                rng.normal(size = kernel.shape, scale = constants.weightInitStdev)
                            ).astype(np.float32),
                        name = name_prefix + "_kernel")

    bias_shape = kernel_shape[3] 
    biases     = np.empty(bias_shape, dtype = np.float32)
    biases[:]  = constants.biasInitValue
    biases     = tf.Variable(biases, name = name_prefix + "_biases")

    return weights, biases


def fc_weights_bias(nb_inputs, nb_outputs, rng, name_prefix):
    """ Create weights and biases shared variable for fully connected layer
        
    biases are initialized with a constant value of constants.biasInitValue
    weights are initialized following a gaussian distribution of mean 0 and standard deviation of constants.weightInitStdev

    Args:
        nb_inputs:  Number of inputs of the layer
        nb_outputs: Number of outputs of the layer
        rng: a numpy random number generator
        name_prefix: The name prefix for the operators

    Returns:
        The weights shared variable and the biases shared variable
    """
    weights = tf.Variable(np.asarray(
                                rng.normal(size = [nb_inputs, nb_outputs],
                                        scale = constants.weightInitStdev
                                    )
                                ).astype(np.float32),
                          name = name_prefix+"_weights"
                    )

    biases    = np.empty([nb_outputs], dtype = np.float32)
    biases[:] = constants.biasInitValue
    biases    = tf.Variable(biases, name = name_prefix + "_biases")

    return weights, biases


class WeightHolder(object):
    """ Class holding weights and rmsprop runing means for the shared network. 
    
    Attributes:
        params : the shared variable containing the parameters values
        learning_rate: a place holder for the shared rmsprop optimizer learning rate
        rmsprop : the shared rmsprop optimizer
    """
    def __init__(self, nb_actions, name):
        """ Initialise the WeightHolder
        
        Args:
            nb_actions: Number of actions for the considered game
            name: The name prefix for the network variables and optimizer
        """
        rng = np.random.RandomState(42)
        
        conv1_weights, conv1_bias = conv_weights_bias(constants.conv1_shape,
                                                      rng,
                                                      name+"_conv1"
                                                     )
        conv2_weights, conv2_bias = conv_weights_bias(constants.conv2_shape, 
                                                      rng,
                                                      name+"_conv2"
                                                     )
        fcl1_weights, fcl1_bias = fc_weights_bias(constants.cnn_output_size, 
                                                  constants.fcl1_nbUnit)
        fcl2_weights, fcl2_bias = fc_weights_bias(constants.fcl1_nbUnit, nb_actions)

        self.params = [ conv1_weights, 
                        conv1_bias,
                        conv2_weights,
                        conv2_bias,
                        fcl1_weights,
                        fcl1_bias,
                        fcl2_weights,
                        fcl2_bias]

        self._placeholders = []
        for param in self.params():
            self._placeholders.append(tf.placeholder(tf.float32, param.get_shape()))

        self._updates = []
        for param, placeholder in zip(self.params, self._placeholders):
            self._updates.append(param.assign(placeholder))

        self.learning_rate = tf.placeholder(tf.float32)
        self.rmsprop = tf.train.RMSPropOptimizer(self.learning_rate,
                                                 decay   = constants.decay_factor,
                                                 epsilon = constants.epsilon_cancel,
                                                 use_locking = constants.updateLock
                                                 )
        
    def update(self, network, session):
        """ Copy the values of an external network's parameters
        
        Args:
            network: The external network from which the parameters value should be copied
            session: The session on which the initialization should run
        """
        for placeholder, updatenode, targetparam in zip(self._placeholders, self._updates, network.params):
            session.run(updatenode, feed_dict = {placeholder : targetparam})


class DQN(object):
    """ Computation architecture of deep Q network
        
        Args:
            params: list of local copies of parameter values
            inputs: image inputs placeholder
            network: Associated WeightHolder
            output: the output layer of the DQN
    """
    def __init__(self, network, name_prefix, inputs):
        """ Initialize the DQN
        Args:
            network: the associated WeightHolder
            name_prefix: the name_prefix for the DQN operators
            inputs: the tensor which will be used as input
        """
        self.network = network
    
        self.inputs = inputs 

        conv1_weights = tf.Variable(network.conv1_weights)
        conv1_bias    = tf.Variable(network.conv1_bias)
        conv2_weights = tf.Variable(network.conv2_weights)
        conv2_bias    = tf.Variable(network.conv2_bias)
        fcl1_weights  = tf.Variable(network.fcl1_weights)
        fcl1_bias     = tf.Variable(network.fcl1_bias)
        fcl2_weights  = tf.Variable(network.fcl2_weights)
        fcl2_bias     = tf.Variable(network.fcl2_bias)

        self.params = [ conv1_weights,
                        conv1_bias,
                        conv2_weights,
                        conv2_bias,
                        fcl1_weights,
                        fcl1_bias,
                        fcl2_weights,
                        fcl2_bias
                      ]

        conv1 = tf.nn.relu(tf.nn.conv2d(self.inputs,
                                             conv1_weights,
                                             constants.conv1_strides,
                                             padding = "VALID") + 
                                    conv1_bias)
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1,
                                             conv2_weights,
                                             constants.conv2_strides,
                                             padding = "VALID") +
                                        conv2_bias)

        flatconv2 = tf.reshape(conv2, [constants.cnn_output_size])

        fcl1        = tf.nn.relu(tf.matmul(flatconv2, fcl1_weights) + fcl1_bias)
        self.output = tf.matmul(fcl1, fcl2_weights) + fcl2_bias

        self._assign_list = []
        for local, shared in zip(self.params, network.params):
            self._assign_list.append(local.assign(shared))

    def copy_network(self, session):
    """ Update the local parameter values according to the WeightHolder values """
        for assign in self._assign_list:
            session.run(assign)
        

class AgentComputation(object):
    """ Wrapper of useful method for an agent """

    def __init__(self, network_holder, critic_holder, session, ident):
        self.sess = session
        self.inputs = tf.placeholder(tf.float32)

        with tf.device('/cpu:{0}'.format(ident))
            self.network = AgentSubNet(network_holder, ident+"_network", self.inputs, session)
            self.critic  = AgentSubNet(critic_holder,  ident+"_critic",  self.inputs, session)

            self.best_action  = tf.argmax(self.network.output, 0)
            self.score_critic = tf.reduce_max(self.critic.output)

            self.action = tf.placeholder(tf.int32)
            self.label  = tf.placeholder(tf.float32)
        
            accumulators = []
        
            for param in network_holder.params:
                accumulators.append(tf.shared(tf.zeros_like(param)))

            self.reset_acc = [acc.assign(tf.zeros_like(acc)) for acc in accumulators]

            network_score = tf.slice(self.network.output, [self.action], [1])
            loss = 0.5 * tf.square(network_score - self.label) 

            gradients = network_holder.rmsprop.compute_gradients(self.loss,
                                                             self.network.params
                                                            )
            self.acc_op = []  
            for acc, grad in zip(accumulators, gradients):
                acc.assign_add(gradients[0])

            acc_caped = [tf.clip_by_value(  acc, 
                                            -constants.gradient_clip, 
                                            constants.gradient_clip
                                        ) for acc in accumulators]
            applyGradList = [(grad, var) for grad, var in zip(acc_caped, network_holder.params)]

            self.applyGradOp = network_holder.rmsprop.apply_gradients(applyGradList)


