class constants:
    input_frames    = 4
    input_size      = 84*84
    conv1_nbfilt    = 32
    image_shape     = [None, 84, 84, 4]
    conv1_shape     = [8, 8, 4, 16]
    conv1_zwidth    = 16
    conv1_strides   = [1, 4, 4, 1]
    conv2_shape     = [4, 4, 16, 32]
    conv2_zwidth    = 32
    conv2_strides   = [1, 2, 2, 1]
    cnn_output_size = conv2_zwidth * 9 * 9 * 4
    fcl1_nbUnit     = 256
    max_noop        = 30
    final_e_frame   = 1000000
    action_repeat   = 4
    discount_factor = 0.99
    decay_factor    = 0.99
    nb_max_frames   = 80000000
    batch_size      = 5
    critic_up_freq  = 10000
    epsilon_start   = 1
    epsilon_cancel  = 0.1
    weightInitStdev = 0.25
    biasInitValue   = 1
    freq_fresh_eps  = 200
    gradient_clip   = 40

    filebas = 'output_agent_'

    updateLock      = True
    lock_T          = True
    read_lock       = False

    debug_main = False
    nb_agent   = 16 


class shared:
    nb_actions      = 0
    game_name       = ''
    T               = 0 
