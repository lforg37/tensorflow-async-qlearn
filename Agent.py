import parameters

class AgentThread(threading.Thread):
    def __init__(self, lock, ale):
        threading.Thread.__init__(self)
        self.lock = lock
        self.ale = ale
        self.lock.acquire()
        ale.reset_game()
        self.ale_state = ale.cloneSystemState()
        self.lock.release()
        
        #Tensorflow graph here
        self.g_inputs = tf.placeholder(tf.uint8, shape=(constants.input_frames, constants.input_size))


    def run(self):
        self.lock.acquire()
        self.ale.restoreSystemState(self.ale_state)
        legal_actions = self.ale.getLegalActionSet()
        go_on = not self.ale.game_over()
        self.lock.release()
        total_reward = 0
        while go_on:
            a = legal_actions[random.randrange(len(legal_actions))]
            self.lock.acquire()
            self.ale.restoreSystemState(self.ale_state)
            reward = self.ale.act(a)
            go_on = not self.ale.game_over()
            self.ale_state = self.ale.cloneSystemState()
            self.lock.release()
            total_reward += reward

