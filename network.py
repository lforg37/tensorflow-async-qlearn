import tensorflow as tf
import random
import parameters

def conv_weights_bias(kernel_shape, bias_shape):
    # Create variable named "weights".
    tf.get_variable("weights", kernel_shape, initializer=tf.random_normal_initializer())
    tf.get_variable("weights_critic", kernel_shape, initializer=tf.random_normal_initializer())

    # Create variable named "biases".
    tf.get_variable("biases", bias_shape, initializer=tf.constant_initializer(0.1))
    tf.get_variable("biases_critic", bias_shape, initializer=tf.constant_initializer(0.1))

def fc_weights_bias(nb_inputs, nb_outputs):
    tf.get_variable("weights", [nb_inputs, nb_outputs], initializer=tf.random_normal_initializer())
    tf.get_variable("weights_critic", [nb_inputs, nb_outputs], initializer=tf.random_normal_initializer())
    tf.get_variable("biases",  [nb_outputs], initializer=tf.constant_initializer(0.1))
    tf.get_variable("biases_critic",  [nb_outputs], initializer=tf.constant_initializer(0.1))

def createGlobalWeights(nb_actions):
    random.seed(None)
    constants.fcl2_nbUnit = nb_actions
    with tf.variable_scope("conv1"):
        conv_weights_bias(constants.conv1_shape, [constants.input_frames])
        tf.get_variable_scope().reuse_variables()
    with tf.variable_scope("conv2"):
        conv_weights_bias(constants.conv2_shape, [constants.conv1_zwidth])
        tf.get_variable_scope().reuse_variables()
    with tf.variable_scope("conv3"):
        conv_weights_bias(constants.conv3_shape, [constants.conv2_zwidth])
        tf.get_variable_scope().reuse_variables()
    with tf.variable.scope("fcl1"):
        fc_weights_bias(constants.cnn_output_size,  constants.fcl1_nbUnit)
        tf.get_variable_scope().reuse_variables()
    with tf.variable.scope("fcl2"):
        fc_weights_bias(constants.fcl1_nbUnit,  shared.nb_actions)
        tf.get_variable_scope().reuse_variables()
    with tf.variable_scope("global"):
        tf.get_variable("counter", initializer=tf.constant_initializer(0))

    inputs        = tf.placeholder(tf.uint8, shape=[input_frames, input_size])
    reward        = tf.placeholder(tf.int32, shape=[1])
    final_state   = tf.placeholder(tf.bool,  shape = [1])
    local_counter = tf.Variable(0)
    state_buffer  = tf.

    reshape = tf.reshape(placeholder, constants.image_shape)

    with tf.variable_scope("global", reuse=True):
        compteur = tf.get_variable("counter")
    with tf.variable_scope("conv1", reuse=True):
        weights_conv1        = tf.get_variable("weights")
        weights_critic_conv1 = tf.get_variable("weights_critic")
        biases_conv1         = tf.get_variable("biases")
        biases_critic_conv1  = tf.get_variable("biases_critic")
    conv1 = tf.nn.relu(tf.nn.conv2d(inputs,
                                    weights_conv1,
                                    strides = constants.conv1_strides)
                                    + biases_conv1)
    conv1_critic = tf.nn.relu(tf.nn.conv2d(inputs,
                                    weights_critic_conv1,
                                    strides = constants.conv1_strides)
                                    + biases_critic_conv1)


    with tf.variable_scope("conv2", reuse=True):
        weights_conv2        = tf.get_variable("weights")
        biases_conv2         = tf.get_variable("biases")
        weights_critic_conv2 = tf.get_variable("weights_critic")
        biases_critic_conv2  = tf.get_variable("biases_critic")
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1,
                                    weights_conv2,
                                    strides = constants.conv2_strides)
                                    + biases_conv2)
    conv2_critic = tf.nn.relu(tf.nn.conv2d(conv1_critic,
                                    weights_critic_conv2,
                                    strides = constants.conv2_strides)
                                    + biases_critic_conv2)

    with tf.variable_scope("conv3", reuse=True):
        weights_conv3        = tf.get_variable("weights")
        biases_conv3         = tf.get_variable("biases")
        weights_critic_conv3 = tf.get_variable("weights_critic")
        biases_critic_conv3  = tf.get_variable("biases_critic")
    conv3 = tf.nn.relu(tf.nn.conv2d(conv2,
                                    weights_conv3,
                                    strides = constants.conv3_strides)
                                    + biases_conv3)
    conv3_criic = tf.nn.relu(tf.nn.conv2d(conv2_critic,
                                    weights_critic_conv3,
                                    strides = constants.conv3_strides)
                                    + biases_critic_conv3)

    flatconv3 = tf.reshape(conv3, [-1, constants.cnn_output_size])
    flatconv3_critic = tf.reshape(conv3_critic, [-1, constants.cnn_output_size])

    with tf.variable_scope("fcl1", reuse=True):
        weights_fcl1        = tf.get_variable("weights")
        biases_fcl1         = tf.get_variable("biases")
        weights_critic_fcl1 = tf.get_variable("weights_critic")
        biases_critic_fcl1  = tf.get_variable("biases_critic")
    fcl1 = tf.nn.relu(tf.matmul(flatconv3, weights_fcl1) + biases_fcl1)
    fcl1_critic = tf.nn.relu(tf.matmul(flatconv3_critic + weights_critic_fcl1) 
            + biases_critic_fcl1)

    with tf.variable_scope("fcl2", reuse=True):
        weights_fcl2        = tf.get_variable("weights")
        biases_fcl2         = tf.get_variable("biases")
        weights_critic_fcl2 = tf.get_variable("weights_critic")
        biases_critic_fcl2  = tf.get_variable("biases_critic")
    fcl2 = tf.nn.relu(tf.matmul(fcl1, weights_fcl2) + biases_fcl2)
    fcl2_critic = tf.relu(tf.matmul(fcl1_critic, weights_critic_fcl2) 
                + biases_critic_fcl2)

    updateCritic = [weights_critic_conv1.assign(weights_conv1),
                    weights_critic_conv2.assign(weights_conv2),
                    weights_critic_conv3.assign(weights_conv3),
                    weights_critic_fcl1.assign(weights_fcl1),
                    weights_critic_fcl2.assign(weights_fcl2),
                    biases_critic_conv1.assign(biases_conv1),
                    biases_critic_conv2.assign(biases_conv2),
                    biases_critic_conv3.assign(biases_conv3),
                    biases_critic_fcl1.assign(biases_fcl1),
                    biases_critic_fcl2.assign(biases_fcl2),
                    ]



class AgentSubNet:
    def __init__(self):
                best_action = tf.argmax(fcl2, 1)
        min_epsilon = tf.constant(constants.epsilon_end, tf.float32)
        max_epsilon = tf.constant(constants.epsilon_init, tf.float32)
        final_frame = tf.constant(constants.final_e_frame)
        epsilon_linear = (max_epsilon - min_epsilon) / final_frame
        epsilon = tf.cond(self.compteur < constants.final_e_frame, epsilon_linear, min_epsilon)
        random_epsilon = tf.random_uniform([1])
        random_action = tf.random_uniform([1], dtype=tf.uint8, maxval = shared.nb_actions)
        self.egreedy_action = tf.cond(epsilon < random_epsilon, best_action, random_epsilon)

        self.optimizer = tf.train.RMSPropOptimizer()

