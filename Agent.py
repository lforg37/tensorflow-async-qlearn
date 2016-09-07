import threading
from parameters import shared, constants
from ale_python_interface import ALEInterface
from random import randrange, random
import net_compute
import numpy as np
from utils import LockManager
from improc import BilinearInterpolator2D
#from PIL import Image

def sample_final_epsilon():
    """
    Sample a final epsilon value of epsilon-greedy policy
    """
    final_epsilons = np.array([0.1, 0.01, 0.5])
    probabilities = np.array([0.4, 0.3, 0.3])
    return np.random.choice(final_epsilons, 1, p=list(probabilities))[0]

class AgentThread(threading.Thread):
    def __init__(self, mainNet, criticNet, session, tlock, rlock, barrier, ident):
        threading.Thread.__init__(self)
        self.ident = ident
        self.rlock = rlock
        self.barrier = barrier
        self.computation = net_compute.AgentComputation(mainNet, critic_holder, session, ident)
        self.session = session
        self.tlock = tlock

    def run(self):
        ale = ALEInterface()
        ale.setInt(b'random_seed', randrange(0,256,1))
        ale.loadROM(shared.game_name)
        actions = self.ale.getMinimalActionSet()

        f.open(constants.filebase + str(ident), 'w')

        t_lock = LockManager(self.tlock.acquire, self.tlock.release, constants.lock_T)
        reader_lock = LockManager(self.rlock.acquire, self.rlock.release, constants.read_lock)

        t = 0
        scores = []
        score = 0

        epsilon_end  = sample_final_epsilon()
        epsilon      = constants.epsilon_start

        # Epsilon linearlily decrease from one to self.epsilon_end
        # between frame 0 and frame constants.final_e_frame
        eps_decrease = -abs(constants.epsilon_start - epsilon_end) / constants.final_e_frame

        interpolator = BilinearInterpolator2D([210,160],[84,84])
        current_frame = np.empty([210, 160, 1], dtype=np.uint8)
        ale.getScreenGrayscale(current_frame)
        
        next_state    = np.empty([constants.action_repeat, 84, 84, 1], dtype=np.float32)
        interpolator.interpolate(current_frame, next_state[0])
        next_state[1:4] = next_state[0]

        with t_lock:
            T = shared.T
            shared.T += 1

        self.barrier.wait()
    
        while T < constants.nb_max_frames:
            t += 1
            state      = next_state
            next_state = np.empty_like(state)

            # Determination of epsilon for the current frame
            epsilon += eps_decrease
            epsilon = max(epsilon, eps_decrease)

            #Choosing current action based on epsilon greedy behaviour
            rnd = random()
            if rnd < epsilon:
                action = randrange(0, len(actions))
            else:
                with reader_lock:
                    action = self.computation.getBestAction(state.transpose(0,3,1,2))[0]

            reward = 0
            i      = 0

            #repeating constants.action_repeat times the same action 
            #and cumulating the rewards 
            while i < constants.action_repeat and not ale.game_over():
                reward += ale.act(actions[action])
                ale.getScreenGrayscale(current_frame)
                interpolator.interpolate(current_frame, next_state[i])
                i += 1

            while i < constants.action_repeat:
                next_state[i] = next_state[i-1]
                i += 1

            score += reward

            discounted_reward = 0
            if   reward > 0:
                discounted_reward = 1
            elif reward < 0:
                discounted_reward = -1

            if not ale.game_over():
                #Computing the estimated Q value of the new state
                with reader_lock:
                    discounted_reward += constants.discount_factor * \
                                        self.computation.getCriticScore(next_state.transpose(0,3,1,2))[0]

            computation.cumulateGradient(
                        state.transpose(0,3,1,2), 
                        action, 
                        discounted_reward, ident)

            if t != 0 and (t % constants.batch_size == 0 or ale.game_over()):
                #computing learning rate for current frame
                lr = init_learning_rate * (1 - T/constants.nb_max_frames)
                self.computation.applyGradient(lr)
                t = 0

            if T % constants.critic_up_freq == 0:
                f.write("Update critic !\n")
                f.flush()
                self.computation.update_critic()
                
            #Log some statistics about played games
            if ale.game_over():
                f.write("["+str(ident)+"] Game ended with score of : "+str(score) + "\n")
                f.write("["+str(ident)+"] T : "+str(T)+"\n")
                ale.reset_game()
                interpolator.interpolate(current_frame, next_state[0])
                next_state[1:4] = next_state[0]
                scores.append(score)
                if len(scores) >= constants.lenmoy:
                    moy = sum(scores) / len(scores)
                    f.write("Average scores for last 12 games for Agent "+str(ident)+ " : " + str(moy)+"\n")
                    f.flush()
                    scores = []
                score = 0

            with t_lock:
                T = T_glob.value
                T_glob.value += 1
