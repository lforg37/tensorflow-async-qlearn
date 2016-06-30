class constants:
    input_frames    = 4
    input_size      = 84*84
    conv1_nbfilt    = 32
    image_shape     = [-1, 84, 84, 1]
    conv1_shape     = [8, 8, 1, 32]
    conv1_zwidth    = 32
    conv1_strides   = [1, 4, 4, 1]
    conv2_shape     = [4, 4, 32, 64]
    conv2_zwidth    = 64
    conv2_strides   = [1, 2, 2, 1]
    conv3_shape     = [3, 3, 64, 64]
    conv3_strides   = [1, 1, 1, 1]
    conv3_zwidth    = 64
    cnn_output_size = conv3_zwidth * 7 * 7
    fcl1_nbUnit     = 512
    max_noop        = 30
    epsilon_init    = 1.
    epsilon_end     = 0.1
    final_e_frame   = 1000000
    action_repeat   = 4
    discount_factor = 0.99
    nb_thread       = 1

class shared:
    nb_actions      = 0
    game_name       = ''
