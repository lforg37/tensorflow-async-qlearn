import threading
from parameters import shared, constants
from ale_python_interface import ALEInterface
from random import randrange
from random import random
from network import AgentSubNet
import numpy as np
from image_process import resize_tf
from PIL import Image

class AgentThread(threading.Thread):
    def __init__(self, session, lock, ident):
        self.id = ident
        threading.Thread.__init__(self)
        self.lock = lock
        self.ale = ALEInterface()
        self.ale.setInt(b'random_seed', randrange(0,256,1))
        self.ale.loadROM(shared.game_name)
        self.actions = self.ale.getMinimalActionSet()
        with lock:
            self.network = AgentSubNet(self.id)
        self.session = session
        self.t = 0

    def run(self):
        images     = np.zeros([constants.action_repeat, 210, 160, 1], dtype=np.uint8)

        current_frame = np.empty([210, 160, 1], dtype=np.uint8)
        self.ale.getScreenGrayscale(current_frame)
        #images[:] = current_frame
        state = resize_tf(images)
        #Image.fromarray(state[:,:,1]).convert("L").save("new_resized.png")

        reward_batch = []
        action_batch = []
        state_batch  = []
        score = 0
        with self.lock:
            T = shared.T
            shared.T += 1
        
        while T < constants.nb_max_frames and not self.ale.game_over():
            state_batch.append(state)
            T = shared.nb_actions
            epsilon = constants.epsilon_end
            if T < constants.final_e_frame:
                epsilon = constants.epsilon_init + T * \
                    (constants.epsilon_end - constants.epsilon_init) / constants.final_e_frame

            rnd = random()
            action = randrange(0, shared.nb_actions) if rnd < epsilon \
                     else self.network.computeAction(state[np.newaxis, :, :, :], self.session)
            action_vect = np.zeros([shared.nb_actions])
            action_vect[action] = 1
            action_batch.append(action_vect)

            reward = 0
            i = 0
            while i < constants.action_repeat and not self.ale.game_over():
                reward += self.ale.act(self.actions[action])
                self.ale.getScreenGrayscale(images[i,:,:,:])
                i += 1
            if i < constants.action_repeat:
                for k in range(i, constants.action_repeat):
                    images[k,:,:] = images[i-1, :, :]
            
            state = resize_tf(images)
            #for i in range(0, constants.action_repeat):
            #    name = "state_" + str(self.t) + "_" + str(i) + ".png"
            #    Image.fromarray(state[:,:,i]).convert("L").save(name)
            
            #Evaluating
            discounted_reward = 0 if self.ale.game_over() else self.network.computeCritic(state[np.newaxis, :, :, :], self.session)

            reward_batch.append(reward + constants.discount_factor * discounted_reward)
            
            score += reward
            if self.ale.game_over():
                print("Game ended with score of : "+str(score))
                self.ale.reset_game()
                score = 0
                

            if self.t % constants.batch_size == 0 or self.ale.game_over():
                self.network.update(state_batch, action_batch, reward_batch, self.session)
                reward_batch = []
                action_batch = []
                state_batch  = []
            
            if self.t % constants.critic_up_freq == 0:
                self.network.updateCritic(self.session)

            with self.lock: 
                T = shared.T
                shared.T += 1

            self.t += 1
